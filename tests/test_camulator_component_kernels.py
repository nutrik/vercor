from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xarray as xr
from jax.errors import JaxRuntimeError

import vercor.setups._external.camulator_contracts as camulator_contracts_module
import vercor.setups._external.camulator_fields as camulator_fields_module
import vercor.setups._external.camulator_forcing as camulator_forcing_module
import vercor.setups._external.camulator_gcm_state as camulator_gcm_state_module
import vercor.setups._external.camulator_imports as camulator_imports_module
import vercor.setups._external.camulator_init as camulator_init_module
import vercor.setups._external.camulator_land as camulator_land_module
import vercor.setups._external.camulator_output as camulator_output_module
import vercor.setups._external.camulator_runtime as camulator_runtime_module
import vercor.setups._external.camulator_tensors as camulator_tensors_module
import vercor.setups._external.camulator_wind_filter as camulator_wind_filter_module
from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor.exceptions import ComponentError, CouplerError
from vercor.recipes import ATMOSPHERE_TO_LAND_RADIATION_FIELDS
from vercor._runtime.contracts import build_exchange_contracts
from vercor.components.contexts import SetupContext, StepContext
from vercor.components import ComponentSpec, DataComponent, StepResult
from vercor.components._adapter import normalize_component, prepare_component
from vercor.components.runtime_execution import step_component_runtime_state
from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.fields import _flatten_field_items
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputVariable,
    PeriodOutput,
    SnapshotContext,
)
from vercor.output._session import _OutputAccumulator
from vercor.setups import CAMulatorConfig
from vercor.setups._external.camulator import make_camulator_gcm
from vercor.fluxes.vertical_coordinates import get_altitudes_hybrid_sigma_levels
from vercor.physics import PhysicalConstants
from vercor.grids import RectilinearGrid
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.component_state import create_runtime_component_state
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor._runtime.validation import validate_exchange_fields_declared
from vercor.jax_logging import DEFAULT_LOGGER_NAME


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def _record(self, message: object, *args: Any, **kwargs: Any) -> None:
        _ = kwargs
        message_text = str(message)
        self.messages.append(message_text.format(*args) if args else message_text)

    def info(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def error(self, message: object, *args: Any, **kwargs: Any) -> None:
        self._record(message, *args, **kwargs)

    def setLevel(self, level: int | str) -> None:
        _ = level

    def isEnabledFor(self, level: int) -> bool:
        _ = level
        return True


def _runtime_component_state(
    name: str,
    data: dict[str, Any] | None = None,
) -> ComponentRuntimeState:
    _ = name
    return ComponentRuntimeState(
        fields=FieldStore.from_mapping(data or {}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )


def _make_coupler(start: datetime) -> SetupContext:
    return SetupContext(
        start=start,
        dt_seconds=21600,
        run_order=(),
        logger=cast(Any, _RecordingLogger()),
    )


def test_camulator_runtime_field_initializer_returns_jax_arrays() -> None:
    fields = camulator_fields_module.initialize_camulator_runtime_fields((2, 3))

    assert fields
    assert all(isinstance(value, jax.Array) for value in fields.values())
    assert {
        "total_surface_temperature",
        "temperature_3d",
        "specific_humidity_3d",
    } <= set(fields)
    assert fields["temperature"].shape == (2, 3)
    assert_allclose_compact(fields["temperature"], np.zeros((2, 3)))


def test_wind_artifact_filter_config_raises_value_error_for_invalid_values() -> None:
    with pytest.raises(ValueError, match="Dilations must be positive"):
        camulator_wind_filter_module.WindArtifactFilterConfig(
            dilation_zonal=0
        ).validate()

    with pytest.raises(ValueError, match="target_levels must be a sequence"):
        camulator_wind_filter_module.WindArtifactFilterConfig(
            target_levels=cast(Any, 5)
        ).validate()


@pytest.mark.fast_always
def test_camulator_wind_filter_facade_exposes_only_runtime_entrypoints() -> None:
    assert not hasattr(camulator_wind_filter_module, "wind_filter")
    assert not hasattr(camulator_wind_filter_module, "simple_wind_artifact_filter")
    assert "wind_filter" not in camulator_wind_filter_module.__all__
    assert "simple_wind_artifact_filter" not in camulator_wind_filter_module.__all__


@pytest.mark.fast_always
def test_wind_filter_private_owner_returns_shape_stable_artifacts() -> None:
    import vercor.setups._external._camulator_wind_filtering as wind_filtering

    u_wind = torch.zeros(5, 5)
    v_wind = torch.zeros(5, 5)
    u_wind[2, 2] = 5.0

    artifacts = wind_filtering.build_wind_filter_artifacts(
        u_wind,
        v_wind,
        speed_threshold=0.5,
        smooth_sigma=1.0,
        dilation_zonal=1,
        dilation_meridional=1,
        falloff_sigma=1.0,
    )

    assert artifacts.u_filtered.shape == u_wind.shape
    assert artifacts.v_filtered.shape == v_wind.shape
    assert artifacts.gaussian_2d.ndim == 2
    assert artifacts.gaussian_2d.shape == (artifacts.kernel_size, artifacts.kernel_size)
    assert artifacts.smooth_blend_mask.shape == (1, 1, *u_wind.shape)


@pytest.mark.fast_always
def test_wind_artifact_filter_updates_only_selected_variable_levels() -> None:
    tensor = torch.zeros(1, 12, 5, 5)
    tensor[:, 1, 2, 2] = 5.0
    tensor[:, 4, 2, 2] = 5.0
    tensor[:, 7, 2, 2] = 9.0
    before = tensor.clone()

    camulator_wind_filter_module.apply_wind_artifact_filter_to_tensor(
        x=tensor,
        varname_upper=["U", "V", "T", "Qtot"],
        levels_per_var=3,
        mask_level=1,
        target_levels=(1,),
        target_vars=("T",),
        speed_threshold=0.5,
        smooth_sigma=1.0,
        dilation_zonal=1,
        dilation_meridional=1,
        falloff_sigma=1.0,
    )

    target_channel = 7
    unchanged_channels = [channel for channel in range(12) if channel != target_channel]
    assert not torch.equal(tensor[:, target_channel], before[:, target_channel])
    assert torch.equal(tensor[:, unchanged_channels], before[:, unchanged_channels])


@pytest.mark.fast_always
def test_post_process_wind_artifacts_logs_and_skips_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tensor = torch.ones(1, 4, 3, 3)
    before = tensor.clone()

    def fail_filter(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        raise RuntimeError("forced failure")

    monkeypatch.setattr(
        camulator_wind_filter_module,
        "apply_wind_artifact_filter_to_tensor",
        fail_filter,
    )

    conf = {
        "postprocessing": {"wind_artifact_filter": {"activate": True}},
        "data": {"variables": ["U", "V"]},
        "model": {"levels": 2},
    }
    with capture_logger_output(DEFAULT_LOGGER_NAME) as output:
        camulator_wind_filter_module.post_process_wind_artifacts(tensor, conf)

    assert "Wind artifact filtering failed: forced failure" in output.getvalue()
    assert torch.equal(tensor, before)


def test_prepare_camulator_surface_forcing_supports_jit_and_gradients() -> None:
    sea_surface_temperature = jnp.asarray([[jnp.nan, 2.0], [5.0, 7.0]])
    land_surface_temperature = jnp.asarray([[10.0, jnp.nan], [15.0, 20.0]])
    land_mask_coslat = jnp.asarray([[0.5, 1.0], [1.2, 0.0]])

    total_surface_temperature, rescaled_total_surface_temperature = jax.jit(
        camulator_fields_module.prepare_camulator_surface_forcing
    )(
        sea_surface_temperature,
        land_surface_temperature,
        land_mask_coslat,
    )

    expected_total = np.asarray([[10.0, 283.0], [283.0, 27.0]])
    expected_rescaled = (expected_total - np.nanmean(expected_total)) / np.nanstd(
        expected_total
    )
    assert_allclose_compact(total_surface_temperature, expected_total)
    assert_allclose_compact(
        rescaled_total_surface_temperature,
        expected_rescaled,
        rtol=1e-7,
        atol=1e-7,
    )

    weights = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    gradient = jax.grad(
        lambda sst: jnp.sum(
            camulator_fields_module.prepare_camulator_surface_forcing(
                sst,
                jnp.asarray([[10.0, 11.0], [12.0, 13.0]]),
                jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            )[1]
            * weights
        )
    )(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
    assert gradient.shape == sea_surface_temperature.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert_finite_jvp_vjp(
        lambda sst: jnp.sum(
            camulator_fields_module.prepare_camulator_surface_forcing(
                sst,
                jnp.asarray([[10.0, 11.0], [12.0, 13.0]]),
                jnp.asarray([[0.0, 0.0], [1.0, 0.0]]),
            )[1]
            * weights
        ),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        jnp.ones((2, 2)),
    )


def test_camulator_surface_forcing_rejects_infinity() -> None:
    with pytest.raises(
        CouplerError,
        match="CAMulator.*surface temperature.*infinity",
    ):
        cast(
            Any,
            camulator_fields_module.prepare_camulator_surface_forcing,
        ).__wrapped__(
            jnp.asarray([[jnp.inf, 280.0]]),
            jnp.asarray([[jnp.nan, jnp.nan]]),
            jnp.zeros((1, 2)),
        )


def test_camulator_surface_forcing_rejects_zero_variance() -> None:
    with pytest.raises(
        CouplerError,
        match="standard deviation.*strictly positive",
    ):
        cast(
            Any,
            camulator_fields_module.prepare_camulator_surface_forcing,
        ).__wrapped__(
            jnp.full((2, 2), 280.0),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
        )


def test_compiled_camulator_surface_forcing_rejects_infinity() -> None:
    with pytest.raises(
        JaxRuntimeError,
        match="CAMulator.*surface temperature.*infinity",
    ):
        result = jax.jit(camulator_fields_module.prepare_camulator_surface_forcing)(
            jnp.asarray([[jnp.inf, 280.0]]),
            jnp.asarray([[jnp.nan, jnp.nan]]),
            jnp.zeros((1, 2)),
        )
        result[0].block_until_ready()


def test_compiled_camulator_surface_forcing_rejects_zero_variance() -> None:
    with pytest.raises(
        JaxRuntimeError,
        match="standard deviation.*strictly positive",
    ):
        result = jax.jit(camulator_fields_module.prepare_camulator_surface_forcing)(
            jnp.full((2, 2), 280.0),
            jnp.zeros((2, 2)),
            jnp.zeros((2, 2)),
        )
        result[0].block_until_ready()


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_camulator_surface_forcing_rejects_nonfinite_land_mask(
    bad_value: float,
) -> None:
    with pytest.raises(CouplerError, match="CAMulator land mask.*active domain"):
        cast(
            Any,
            camulator_fields_module.prepare_camulator_surface_forcing,
        ).__wrapped__(
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            jnp.asarray([[10.0, 20.0], [30.0, 40.0]]),
            jnp.asarray([[bad_value, 0.0], [0.0, 0.0]]),
        )


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_compiled_camulator_surface_forcing_rejects_nonfinite_land_mask(
    bad_value: float,
) -> None:
    with pytest.raises(JaxRuntimeError, match="CAMulator land mask.*active domain"):
        result = jax.jit(camulator_fields_module.prepare_camulator_surface_forcing)(
            jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            jnp.asarray([[10.0, 20.0], [30.0, 40.0]]),
            jnp.asarray([[bad_value, 0.0], [0.0, 0.0]]),
        )
        result[0].block_until_ready()


def test_prepare_camulator_dynamic_forcing_chunk_supports_jit_and_ordering() -> None:
    values = jnp.arange(2 * 3 * 2 * 2, dtype=jnp.float32).reshape(2, 3, 2, 2)

    prepared = jax.jit(camulator_fields_module.prepare_camulator_dynamic_forcing_chunk)(
        values
    )

    assert prepared.shape == (3, 2, 2, 2)
    assert_allclose_compact(prepared, np.asarray(values).transpose((1, 0, 2, 3)))
    assert_finite_jvp_vjp(
        lambda forcing: jnp.sum(
            camulator_fields_module.prepare_camulator_dynamic_forcing_chunk(forcing)
        ),
        values,
        jnp.ones_like(values),
    )


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_camulator_dynamic_forcing_rejects_nonfinite_source(
    bad_value: float,
) -> None:
    values = jnp.zeros((2, 3, 2, 2)).at[0, 0, 0, 0].set(bad_value)
    with pytest.raises(CouplerError, match="CAMulator dynamic forcing.*active domain"):
        cast(
            Any,
            camulator_fields_module.prepare_camulator_dynamic_forcing_chunk,
        ).__wrapped__(values)


@pytest.mark.parametrize("bad_value", [jnp.nan, jnp.inf, -jnp.inf])
def test_compiled_camulator_dynamic_forcing_rejects_nonfinite_source(
    bad_value: float,
) -> None:
    values = jnp.zeros((2, 3, 2, 2)).at[0, 0, 0, 0].set(bad_value)
    with pytest.raises(
        JaxRuntimeError, match="CAMulator dynamic forcing.*active domain"
    ):
        result = jax.jit(
            camulator_fields_module.prepare_camulator_dynamic_forcing_chunk
        )(values)
        result.block_until_ready()


def test_prepare_camulator_sst_input_supports_jit_and_shape() -> None:
    surface_temperature = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    prepared = jax.jit(camulator_fields_module.prepare_camulator_sst_input)(
        surface_temperature
    )

    assert prepared.shape == (1, 1, 1, 2, 2)
    assert_allclose_compact(prepared[0, 0, 0], np.asarray(surface_temperature))
    assert_finite_jvp_vjp(
        lambda forcing: jnp.sum(
            camulator_fields_module.prepare_camulator_sst_input(forcing)
        ),
        surface_temperature,
        jnp.ones_like(surface_temperature),
    )


def test_torch_tensor_from_jax_array_uses_copied_host_boundary() -> None:
    source = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    tensor = camulator_tensors_module.torch_tensor_from_jax_array(source, "cpu")
    tensor[0, 0] = 99.0

    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"
    assert_allclose_compact(np.asarray(source), np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def _camulator_output_conf(
    *,
    save_forecast: str = "/tmp/camulator-output",
    surface_variables: list[str] | None = None,
    diagnostic_variables: list[str] | None = None,
    save_vars: list[str] | None = None,
    climate_rescale_output: bool = False,
) -> dict[str, Any]:
    predict: dict[str, Any] = {"save_forecast": save_forecast}
    if save_vars is not None:
        predict["save_vars"] = save_vars
    if climate_rescale_output:
        predict["climate_rescale_output"] = True
    return {
        "model": {"levels": 3},
        "data": {
            "variables": ["U", "T"],
            "surface_variables": (
                ["PS"] if surface_variables is None else surface_variables
            ),
            "diagnostic_variables": (
                ["FSNS"] if diagnostic_variables is None else diagnostic_variables
            ),
            "level_ids": [10, 20, 30],
        },
        "predict": predict,
    }


def _camulator_prediction(
    total_channels: int, height: int = 2, width: int = 2
) -> torch.Tensor:
    return torch.arange(
        total_channels * height * width,
        dtype=torch.float32,
    ).reshape(1, total_channels, 1, height, width)


def _camulator_runtime_payload(
    *,
    model_state: torch.Tensor,
    prediction: torch.Tensor | None = None,
    prediction_samples: torch.Tensor | None = None,
) -> Any:
    return camulator_gcm_state_module.CAMulatorRuntimePayload(
        model_state=model_state,
        cursor=camulator_forcing_module.CamulatorRuntimeCursor(),
        output_prediction=prediction,
        output_prediction_samples=prediction_samples,
    )


def test_camulator_prediction_values_normalizes_array_like_tensor_output() -> None:
    class _ArrayLike:
        ndim = 4
        shape = (1, 1, 2, 2)

        def __array__(
            self,
            dtype: np.dtype[Any] | None = None,
            copy: bool | None = None,
        ) -> np.ndarray:
            _ = copy
            return np.ones(self.shape, dtype=dtype)

    class _Prediction:
        def detach(self) -> _Prediction:
            return self

        def cpu(self) -> _Prediction:
            return self

        def numpy(self) -> _ArrayLike:
            return _ArrayLike()

    values = camulator_output_module._prediction_values(
        cast(torch.Tensor, cast(object, _Prediction()))
    )

    assert isinstance(values, np.ndarray)
    assert values.shape == (1, 1, 2, 2)


def test_camulator_native_variables_preserve_credit_channel_order() -> None:
    prediction = _camulator_prediction(total_channels=7)
    data_vars = camulator_output_module.camulator_period_output_variables(
        prediction,
        metadata={
            "T": {"units": "K"},
        },
        conf=_camulator_output_conf(diagnostic_variables=[]),
        state_transformer=None,
    )

    assert data_vars["U"].dims == ("time", "level", "latitude", "longitude")
    assert data_vars["T"].dims == ("time", "level", "latitude", "longitude")
    assert data_vars["PS"].dims == ("time", "latitude", "longitude")
    assert data_vars["T"].attrs["units"] == "K"
    assert_allclose_compact(
        data_vars["U"].values,
        prediction[:, 0:3, 0].reshape(1, 3, 2, 2).numpy(),
    )
    assert_allclose_compact(
        data_vars["T"].values,
        prediction[:, 3:6, 0].reshape(1, 3, 2, 2).numpy(),
    )
    assert_allclose_compact(data_vars["PS"].values, prediction[:, 6, 0].numpy())


def test_camulator_native_variables_support_upper_air_only() -> None:
    prediction = _camulator_prediction(total_channels=6)

    data_vars = camulator_output_module.camulator_period_output_variables(
        prediction,
        metadata={},
        conf=_camulator_output_conf(surface_variables=[], diagnostic_variables=[]),
        state_transformer=None,
    )

    assert tuple(data_vars) == ("U", "T")


def test_camulator_period_output_variables_reduce_time_axis() -> None:
    prediction = torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 1, 2, 2)
    variables = camulator_output_module.camulator_period_output_variables(
        prediction,
        metadata={
            "T": {"units": "K"},
            "FSNS": {"long_name": "surface net shortwave flux"},
        },
        conf=_camulator_output_conf(save_vars=["T", "FSNS"]),
        state_transformer=None,
    )
    frame = OutputFrame(
        variables,
        sample_dimension=camulator_output_module.CAMULATOR_TIME_DIM,
    )
    accumulator = _OutputAccumulator.zeros_from_frame(frame).add_frame(frame)
    means = accumulator.mean_frame().variables
    temperature_index = accumulator.names.index("T")

    assert tuple(accumulator.names) == ("U", "T", "PS", "FSNS")
    assert means["T"].dims == ("level", "latitude", "longitude")
    assert means["T"].attrs["units"] == "K"
    assert means["FSNS"].attrs["long_name"] == "surface net shortwave flux"
    assert_allclose_compact(
        accumulator.counts[temperature_index],
        np.full((3, 2, 2), 2),
    )
    assert_allclose_compact(
        means["T"].values,
        np.mean(prediction[:, 3:6, 0].numpy(), axis=0),
    )


def test_camulator_output_provider_returns_native_frame(
    tmp_path: Path,
) -> None:
    prediction = _camulator_prediction(total_channels=8)
    conf = _camulator_output_conf(save_forecast=str(tmp_path), save_vars=["T", "FSNS"])
    metadata = {
        "T": {"units": "K"},
        "FSNS": {"long_name": "surface net shortwave flux"},
        "latitude": {"units": "degrees_north"},
    }

    resources = SimpleNamespace(
        metadata=metadata,
        conf=conf,
        state_transformer=None,
        latlons=SimpleNamespace(
            latitude=SimpleNamespace(values=np.asarray([-45.0, 45.0])),
            longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        ),
    )
    provider = camulator_output_module.camulator_output_provider(resources)

    frame = provider.sample(
        OutputContext(
            component=cast(Any, None),
            state=cast(Any, None),
            payload=_camulator_runtime_payload(
                model_state=torch.zeros_like(prediction),
                prediction=prediction,
                prediction_samples=prediction,
            ),
            step=0,
            time=datetime(2000, 1, 2),
            dt=timedelta(days=1),
        )
    )

    assert tuple(frame.variables) == ("U", "T", "PS", "FSNS")
    assert frame.variables["T"].dims == (
        "time",
        "level",
        "latitude",
        "longitude",
    )
    assert frame.variables["T"].attrs["units"] == "K"
    assert frame.variables["FSNS"].attrs["long_name"] == "surface net shortwave flux"
    assert_allclose_compact(frame.coordinates["latitude"].values, [-45.0, 45.0])


def test_camulator_output_provider_preserves_every_model_substep() -> None:
    predictions = torch.arange(
        2 * 8 * 2 * 2,
        dtype=torch.float32,
    ).reshape(2, 8, 1, 2, 2)
    resources = SimpleNamespace(
        metadata={},
        conf=_camulator_output_conf(save_vars=["T"]),
        state_transformer=None,
        latlons=SimpleNamespace(
            latitude=SimpleNamespace(values=np.asarray([-45.0, 45.0])),
            longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        ),
    )

    frame = camulator_output_module.camulator_output_provider(resources).sample(
        OutputContext(
            component=cast(Any, None),
            state=cast(Any, None),
            payload=_camulator_runtime_payload(
                model_state=torch.zeros_like(predictions[-1:]),
                prediction=predictions[-1:],
                prediction_samples=predictions,
            ),
            step=0,
            time=datetime(2000, 1, 2),
            dt=timedelta(days=1),
        )
    )

    assert frame.variables["T"].values.shape == (2, 3, 2, 2)
    assert_allclose_compact(
        frame.variables["T"].values,
        predictions[:, 3:6, 0].numpy(),
    )


@pytest.mark.parametrize(
    "config_update, message",
    [
        (
            {"predict": {"interp_pressure": {"pressure_levels": [500.0]}}},
            "interp_pressure",
        ),
        ({"use_ptype": True}, "use_ptype"),
        ({"predict": {"ua_var_encoding": {"zlib": True}}}, "ua_var_encoding"),
        (
            {"predict": {"surface_var_encoding": {"zlib": True}}},
            "surface_var_encoding",
        ),
        (
            {"predict": {"pressure_var_encoding": {"zlib": True}}},
            "pressure_var_encoding",
        ),
        (
            {"predict": {"height_var_encoding": {"zlib": True}}},
            "height_var_encoding",
        ),
    ],
)
def test_camulator_output_rejects_unsupported_credit_only_options(
    config_update: dict[str, Any],
    message: str,
) -> None:
    conf = _camulator_output_conf()
    for section, value in config_update.items():
        if isinstance(value, dict) and isinstance(conf.get(section), dict):
            conf[section].update(value)
        else:
            conf[section] = value

    with pytest.raises(ValueError, match=message):
        camulator_output_module.camulator_period_output_variables(
            _camulator_prediction(total_channels=8),
            metadata={},
            conf=conf,
            state_transformer=None,
        )


def test_camulator_output_wrappers_do_not_import_xarray_or_credit_output() -> None:
    camulator_output_source = Path(
        "vercor/setups/_external/camulator_output.py"
    ).read_text(encoding="utf-8")
    camulator_runtime_source = Path(
        "vercor/setups/_external/camulator_runtime.py"
    ).read_text(encoding="utf-8")
    camulator_imports_source = Path(
        "vercor/setups/_external/camulator_imports.py"
    ).read_text(encoding="utf-8")
    output_session_source = Path("vercor/output/_session.py").read_text(
        encoding="utf-8"
    )

    assert "import xarray" not in camulator_output_source
    assert "credit.output" not in camulator_output_source
    assert "credit.output" not in camulator_imports_source
    assert "accumulate_output_variables(" not in camulator_output_source
    assert "period_mean_output_variables(" not in camulator_output_source
    assert "write_period_average_netcdf(" not in camulator_output_source
    assert "should_write_period_output(" not in camulator_runtime_source
    assert not Path("vercor/output/_component_adapter.py").exists()
    assert "should_write_period_output(" in output_session_source


def test_load_camulator_output_metadata_reads_explicit_yaml(tmp_path: Path) -> None:
    metadata_file = tmp_path / "metadata.yaml"
    metadata_file.write_text("T:\n  units: K\n", encoding="utf-8")

    metadata = camulator_output_module.load_camulator_output_metadata(
        {"predict": {"metadata": str(metadata_file)}}
    )

    assert metadata == {"T": {"units": "K"}}


def test_prepare_static_forcing_tensor_preserves_order_and_shape() -> None:
    forcing_ds = xr.Dataset(
        data_vars={
            "TOPO": (("lat", "lon"), np.asarray([[1.0, 2.0], [3.0, 4.0]])),
            "LAND": (("lat", "lon"), np.asarray([[5.0, 6.0], [7.0, 8.0]])),
        }
    )

    static_forcing = camulator_tensors_module.prepare_static_forcing_tensor(
        forcing_ds, ["LAND", "TOPO"], "cpu"
    )

    assert isinstance(static_forcing, torch.Tensor)
    assert static_forcing.shape == (1, 2, 1, 2, 2)
    assert_allclose_compact(
        static_forcing[0, :, 0],
        np.asarray(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[1.0, 2.0], [3.0, 4.0]],
            ]
        ),
    )


def _state_variable_accessor_conf(static_first: bool = False) -> dict[str, Any]:
    return {
        "data": {
            "variables": ["U", "V"],
            "surface_variables": ["TS", "PS"],
            "diagnostic_variables": ["FSNS"],
            "dynamic_forcing_variables": ["SOLIN"],
            "forcing_variables": ["ORO"],
            "static_variables": ["LAND"],
            "static_first": static_first,
        },
        "model": {"levels": 3},
    }


def test_state_variable_accessor_builds_exact_index_maps() -> None:
    state_accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(),
        tensor_type="state",
    )
    input_accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(static_first=False),
        tensor_type="input",
    )
    static_first_input_accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(static_first=True),
        tensor_type="input",
    )
    output_accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(),
        tensor_type="output",
    )

    u_index = state_accessor.get_var_index("U")
    assert u_index.start_idx == 0
    assert u_index.end_idx == 3
    assert u_index.n_channels == 3
    assert u_index.is_3d
    assert u_index.available
    assert state_accessor.get_var_index("V").start_idx == 3
    assert state_accessor.get_var_index("TS").start_idx == 6
    assert state_accessor.get_var_index("PS").end_idx == 8
    fsns_state_index = state_accessor.get_var_index("FSNS")
    solin_state_index = state_accessor.get_var_index("SOLIN")
    assert not fsns_state_index.available
    assert fsns_state_index.reason == "Diagnostics not in state tensor"
    assert not solin_state_index.available
    assert solin_state_index.reason == "Forcing not in state tensor"

    assert input_accessor.get_var_index("SOLIN").start_idx == 8
    assert input_accessor.get_var_index("ORO").start_idx == 9
    assert input_accessor.get_var_index("LAND").start_idx == 10
    assert static_first_input_accessor.get_var_index("LAND").start_idx == 8
    assert static_first_input_accessor.get_var_index("SOLIN").start_idx == 9
    assert static_first_input_accessor.get_var_index("ORO").start_idx == 10
    fsns_input_index = input_accessor.get_var_index("FSNS")
    assert not fsns_input_index.available
    assert fsns_input_index.reason == "Diagnostics not in input tensor"

    fsns_output_index = output_accessor.get_var_index("FSNS")
    assert fsns_output_index.start_idx == 8
    assert fsns_output_index.end_idx == 9
    assert fsns_output_index.n_channels == 1
    assert not fsns_output_index.is_3d
    assert fsns_output_index.available
    land_output_index = output_accessor.get_var_index("LAND")
    assert not land_output_index.available
    assert land_output_index.reason == "Forcing not in output tensor"


@pytest.mark.fast_always
def test_state_variable_accessor_exposes_typed_indices() -> None:
    accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(),
        tensor_type="state",
    )

    variable_index = accessor.get_var_index("U")

    assert isinstance(variable_index, camulator_tensors_module.TensorVariableIndex)
    assert variable_index.channel_slice == slice(0, 3)
    assert variable_index.start_idx == 0
    assert variable_index.end_idx == 3
    assert variable_index.n_channels == 3
    assert variable_index.is_3d
    assert variable_index.available
    diagnostic_index = accessor.get_var_index("FSNS")
    assert not diagnostic_index.available
    assert diagnostic_index.reason == "Diagnostics not in state tensor"


@pytest.mark.fast_always
def test_state_variable_accessor_tensor_access_uses_typed_index_path() -> None:
    accessor = camulator_tensors_module.StateVariableAccessor(
        _state_variable_accessor_conf(),
        tensor_type="state",
    )
    tensor = torch.arange(1 * 8 * 2 * 2 * 2, dtype=torch.float32).reshape(
        1,
        8,
        2,
        2,
        2,
    )

    assert torch.equal(accessor.get_state_var(tensor, "V"), tensor[:, 3:6, ...])

    replacement = torch.full((1, 1, 2, 2, 2), -1.0)
    accessor.set_state_var(tensor, "TS", replacement)

    assert torch.equal(tensor[:, 6:7, ...], replacement)


@pytest.mark.fast_always
def test_state_variable_accessor_uses_shared_index_map_builders() -> None:
    source = Path("vercor/setups/_external/camulator_tensors.py").read_text(
        encoding="utf-8"
    )

    assert "class TensorVariableIndex" in source
    assert "def _append_indexed_variables(" in source
    assert "def _mark_unavailable_variables(" in source
    assert "def get_var_index(" in source


def test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions() -> (
    None
):
    constants = PhysicalConstants()
    hyai = jnp.asarray([0.00, 0.05, 0.10])
    hybi = jnp.asarray([0.00, 0.20, 1.00])
    hyam = jnp.asarray([0.015, 0.025])
    hybm = jnp.asarray([0.15, 0.25])
    u_wind = jnp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    v_wind = u_wind + 10.0
    surface_temperature = jnp.asarray([[280.0, 281.0], [282.0, 283.0]])
    temperature_3d = jnp.asarray(
        [
            [[270.0, 271.0], [272.0, 273.0]],
            [[280.0, 281.0], [282.0, 283.0]],
        ]
    )
    specific_humidity_3d = jnp.full((2, 2, 2), 0.002)
    net_shortwave_accumulated = jnp.full((2, 2), 21600.0 * 8.0)
    net_longwave_accumulated = jnp.full((2, 2), -21600.0 * 3.0)
    surface_pressure = jnp.full((2, 2), 100000.0)

    mapped_fields = jax.jit(camulator_fields_module.map_camulator_prediction_arrays)(
        constants.earth_radius,
        constants.gravity,
        constants.dry_air_gas_constant,
        constants.water_vapor_mass_ratio_correction,
        constants.dry_air_molecular_weight,
        constants.universal_gas_constant,
        constants.reference_pressure,
        constants.dry_air_kappa,
        constants.stefan_boltzmann_constant,
        100000.0,
        hyai,
        hybi,
        hyam,
        hybm,
        u_wind,
        v_wind,
        surface_temperature,
        temperature_3d,
        specific_humidity_3d,
        net_shortwave_accumulated,
        net_longwave_accumulated,
        surface_pressure,
    )

    assert_allclose_compact(mapped_fields["u_velocity"], np.asarray(u_wind[-1]))
    assert_allclose_compact(mapped_fields["v_velocity"], np.asarray(v_wind[-1]))
    assert_allclose_compact(
        mapped_fields["temperature"], np.asarray(temperature_3d[-1])
    )
    assert_allclose_compact(mapped_fields["specific_humidity"], np.full((2, 2), 0.002))
    assert_allclose_compact(
        mapped_fields["net_shortwave_radiation_flux"],
        np.full((2, 2), 8.0),
    )
    assert_allclose_compact(
        mapped_fields["downward_longwave_radiation_flux"],
        constants.stefan_boltzmann_constant * np.asarray(surface_temperature) ** 4
        - 3.0,
    )
    assert mapped_fields["model_level_height"].shape == (2, 2)
    assert mapped_fields["density"].shape == (2, 2)
    assert mapped_fields["potential_temperature"].shape == (2, 2)
    pressure_interfaces = (
        hyai[:, jnp.newaxis, jnp.newaxis] * 100000.0
        + hybi[:, jnp.newaxis, jnp.newaxis] * surface_pressure[jnp.newaxis, :, :]
    )
    expected_model_level_height = get_altitudes_hybrid_sigma_levels(
        constants,
        temperature_3d.T,
        specific_humidity_3d.T,
        pressure_interfaces.T,
    )[..., 0].T
    assert_allclose_compact(
        mapped_fields["model_level_height"], expected_model_level_height
    )
    assert np.all(np.isfinite(np.asarray(mapped_fields["model_level_height"])))
    assert np.all(np.isfinite(np.asarray(mapped_fields["density"])))
    assert np.all(np.isfinite(np.asarray(mapped_fields["potential_temperature"])))

    def mapped_objective(temperature_values: jax.Array) -> jax.Array:
        fields = camulator_fields_module.map_camulator_prediction_arrays(
            constants.earth_radius,
            constants.gravity,
            constants.dry_air_gas_constant,
            constants.water_vapor_mass_ratio_correction,
            constants.dry_air_molecular_weight,
            constants.universal_gas_constant,
            constants.reference_pressure,
            constants.dry_air_kappa,
            constants.stefan_boltzmann_constant,
            100000.0,
            hyai,
            hybi,
            hyam,
            hybm,
            u_wind,
            v_wind,
            surface_temperature,
            temperature_values,
            specific_humidity_3d,
            net_shortwave_accumulated,
            net_longwave_accumulated,
            surface_pressure,
        )
        return sum(
            (jnp.sum(value) for value in fields.values()),
            start=jnp.asarray(0.0),
        )

    assert_finite_jvp_vjp(
        mapped_objective,
        temperature_3d,
        jnp.ones_like(temperature_3d),
        rtol=1e-5,
        atol=1e-7,
    )


def test_camulator_constructor_builds_jax_backed_grid(monkeypatch: Any) -> None:
    state_kwargs: dict[str, Any] = {}
    monkeypatch.setattr(camulator_imports_module, "load_credit_modules", lambda: None)
    latlons = SimpleNamespace(
        longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        latitude=SimpleNamespace(values=np.asarray([-45.0, 0.0, 45.0])),
    )

    class _RecordingState(camulator_gcm_state_module.CAMulatorGCMSetupState):
        def __init__(self, **kwargs: Any) -> None:
            state_kwargs.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        camulator_gcm_state_module,
        "CAMulatorGCMSetupState",
        _RecordingState,
    )
    monkeypatch.setattr(
        camulator_init_module,
        "initialize_camulator",
        lambda **kwargs: {
            "conf": {
                "data": {
                    "dynamic_forcing_variables": ["U"],
                    "lead_time_periods": 6,
                },
                "predict": {"timesteps_fast_climate": 1},
            },
            "stepper": SimpleNamespace(),
            "forcing_dataset": xr.Dataset(),
            "static_forcing": object(),
            "initial_state": object(),
            "latlons": latlons,
            "metadata": {},
            "device": "cpu",
            "state_transformer": object(),
        },
    )

    component = make_camulator_gcm(
        config=CAMulatorConfig(
            config_path="dummy.yaml",
            device="cpu",
            time_alignment="forcing_start",
            output=OutputSpec(period=PeriodOutput(frequency="month")),
        ),
    )

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.spec.inputs == (
        "sea_surface_temperature",
        "land_surface_temperature",
    )
    assert (
        component.spec.outputs
        == camulator_contracts_module.CAMULATOR_RUNTIME_FIELD_NAMES
    )
    assert_allclose_compact(component.grid.binary_mask, np.ones((3, 2)))
    assert callable(component.spec.output.snapshot_writer)
    assert "spinup_time" not in state_kwargs
    assert "do_spinup" not in state_kwargs
    assert state_kwargs["time_alignment"] == "forcing_start"


@pytest.mark.fast_always
def test_camulator_gcm_setup_returns_initial_runtime_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = datetime(2000, 1, 1)
    initial_model_state = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    dynamic_forcing = xr.Dataset(
        data_vars={
            "F1": (
                ("time", "lat", "lon"),
                np.ones((1, 2, 2), dtype=np.float32),
            )
        },
        coords={"time": [start]},
    )
    physics = xr.Dataset(
        data_vars={
            "hyai": (("interface",), np.asarray([0.0, 1.0])),
            "hyam": (("level",), np.asarray([0.5, 1.0])),
            "hybi": (("interface",), np.asarray([0.0, 1.0])),
            "hybm": (("level",), np.asarray([0.5, 1.0])),
            "LANDM_COSLAT": (
                ("lat", "lon"),
                np.zeros((2, 2), dtype=np.float32),
            ),
        }
    )
    resources = cast(
        Any,
        camulator_gcm_state_module.CAMulatorGCMSetupState.__new__(
            camulator_gcm_state_module.CAMulatorGCMSetupState
        ),
    )
    resources.initial_model_state = initial_model_state
    resources.init_noise = None
    resources.stepper = SimpleNamespace(model=lambda value: value)
    resources.lead_time_periods = 6
    resources.conf = {
        "data": {"save_loc_physics": "physics.nc"},
        "predict": {"start_datetime": start},
    }
    resources.forcing_ds_norm = dynamic_forcing
    resources.df_vars = ["F1"]
    resources.device = "cpu"
    resources.time_alignment = "strict"

    monkeypatch.setattr(
        camulator_gcm_state_module.torch.jit,
        "trace",
        lambda model, dummy_input: model,
    )
    monkeypatch.setattr(
        camulator_gcm_state_module.xr,
        "open_dataset",
        lambda path: physics,
    )
    monkeypatch.setattr(
        camulator_tensors_module,
        "StateVariableAccessor",
        lambda conf, tensor_type: SimpleNamespace(tensor_type=tensor_type),
    )

    result = resources.setup(cast(Any, None), _make_coupler(start))

    assert isinstance(
        result.payload, camulator_gcm_state_module.CAMulatorRuntimePayload
    )
    assert result.payload.model_state is initial_model_state
    assert result.payload.forecast_hour == 1
    assert result.payload.output_prediction is None
    assert result.payload.output_prediction_samples is None
    assert result.payload.cursor.start_ix == 0
    assert result.payload.cursor.model_substeps == 1
    assert result.payload.cursor.timestep_counter == 0


def test_camulator_default_snapshot_uses_native_provider_when_period_provider_is_custom(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(camulator_imports_module, "load_credit_modules", lambda: None)
    latlons = SimpleNamespace(
        longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        latitude=SimpleNamespace(values=np.asarray([-45.0, 45.0])),
    )
    monkeypatch.setattr(
        camulator_init_module,
        "initialize_camulator",
        lambda **kwargs: {
            "conf": {
                "data": {
                    "dynamic_forcing_variables": ["U"],
                    "lead_time_periods": 6,
                },
                "predict": {"timesteps_fast_climate": 1},
            },
            "stepper": SimpleNamespace(),
            "forcing_dataset": xr.Dataset(),
            "static_forcing": object(),
            "initial_state": object(),
            "latlons": latlons,
            "metadata": {},
            "device": "cpu",
            "state_transformer": object(),
        },
    )

    samples: list[str] = []
    sample_payloads: list[Any] = []

    class _Provider:
        def __init__(self, name: str) -> None:
            self.name = name

        def sample(self, context: OutputContext) -> OutputFrame:
            samples.append(self.name)
            sample_payloads.append(context.payload)
            return OutputFrame({"temperature": OutputVariable((), np.asarray(280.0))})

    custom_provider = _Provider("custom")
    native_provider = _Provider("native")
    native_modes: list[bool] = []

    def make_native_provider(state: Any, *, latest_only: bool = False) -> _Provider:
        _ = state
        native_modes.append(latest_only)
        return native_provider

    monkeypatch.setattr(
        camulator_output_module,
        "camulator_output_provider",
        make_native_provider,
    )
    monkeypatch.setattr(
        camulator_output_module,
        "write_netcdf_dataset",
        lambda **kwargs: None,
    )

    component = make_camulator_gcm(
        config=CAMulatorConfig(
            config_path="dummy.yaml",
            device="cpu",
            output=OutputSpec(provider=custom_provider),
        )
    )
    writer = cast(Any, component.spec.output.snapshot_writer)
    prediction = torch.ones((1, 1, 1, 1, 1))
    payload = _camulator_runtime_payload(
        model_state=torch.zeros_like(prediction),
        prediction=prediction,
        prediction_samples=prediction,
    )

    writer(
        SnapshotContext(
            component=component,
            state=cast(Any, None),
            payload=payload,
            output_path=tmp_path / "camulator.snapshot.nc",
            time=datetime(2000, 1, 2),
            logger=None,
        )
    )

    assert samples == ["native"]
    assert sample_payloads == [payload]
    assert native_modes == [True]


def test_add_init_noise_logs_through_injected_logger(
    monkeypatch: Any,
) -> None:
    logger = _RecordingLogger()
    state = torch.ones((1, 1), dtype=torch.float32)
    monkeypatch.setattr(camulator_init_module.torch, "randn_like", torch.zeros_like)

    actual = camulator_init_module.add_init_noise(state, noise_std=0.125, logger=logger)

    assert torch.equal(actual, state)
    assert logger.messages == ["Adding initial condition noise (std=0.125)"]


def test_initialize_camulator_logs_lifecycle_through_injected_logger(
    tmp_path: Any, monkeypatch: Any
) -> None:
    logger = _RecordingLogger()
    config_path = tmp_path / "camulator.yml"
    config_path.write_text("predict: {}\n", encoding="utf-8")

    class _Model:
        def to(self, device: Any) -> "_Model":
            _ = device
            return self

        def eval(self) -> None:
            return None

    class _Transformer:
        def transform_dataset(self, dataset: Any) -> Any:
            return dataset

    class _Dataset:
        def chunk(self, chunks: Any) -> "_Dataset":
            _ = chunks
            return self

    stepper = SimpleNamespace(
        flag_mass=True,
        flag_water=False,
        flag_energy=True,
        enable_wind_filtering=False,
    )

    conf = {
        "data": {
            "forcing_chunk_size": 4,
            "scaler_type": "std_new",
            "static_variables": ["LAND"],
        },
        "loss": {"latitude_weights": "latlon.nc"},
        "predict": {
            "forcing_file": "forcing.nc",
            "init_cond_fast_climate": "initial.pt",
            "mode": None,
        },
    }

    monkeypatch.setattr(camulator_imports_module, "CREDIT_AVAILABLE", True)
    monkeypatch.setattr(
        camulator_init_module.yaml,
        "load",
        lambda config_file, Loader: conf,
    )
    monkeypatch.setattr(
        camulator_imports_module,
        "credit_main_parser",
        lambda parsed, parse_training, parse_predict, print_summary: parsed,
        raising=False,
    )
    monkeypatch.setattr(
        camulator_imports_module, "load_transforms", lambda conf: None, raising=False
    )
    monkeypatch.setattr(
        camulator_imports_module,
        "Normalize_ERA5_and_Forcing",
        lambda conf: _Transformer(),
        raising=False,
    )
    monkeypatch.setattr(
        camulator_imports_module,
        "load_model_name",
        lambda conf, model_name, load_weights: _Model(),
    )
    assert not hasattr(camulator_imports_module, "distributed_model_wrapper")
    assert not hasattr(camulator_imports_module, "load_model_state")
    assert not hasattr(camulator_imports_module, "load_model")
    monkeypatch.setattr(camulator_init_module.os.path, "exists", lambda path: True)
    monkeypatch.setattr(
        camulator_init_module.torch,
        "load",
        lambda path, map_location: torch.zeros((1, 1), dtype=torch.float32),
    )
    monkeypatch.setattr(
        camulator_init_module.xr, "open_dataset", lambda path, **kwargs: _Dataset()
    )
    monkeypatch.setattr(
        camulator_init_module,
        "prepare_static_forcing_tensor",
        lambda forcing_ds, static_variables, device: torch.zeros((1, 1)),
    )
    monkeypatch.setattr(
        camulator_output_module,
        "load_camulator_output_metadata",
        lambda conf: {},
    )
    monkeypatch.setattr(
        camulator_init_module, "CAMulatorStepper", lambda model, conf, device: stepper
    )

    camulator_init_module.initialize_camulator(
        str(config_path),
        model_name="checkpoint.pt",
        device="cpu",
        logger=logger,
    )

    assert logger.messages[:3] == [
        f"Initializing CAMulator from config: {config_path}",
        "Using device: cpu",
        "Loading transforms...",
    ]
    assert "Initialization complete!" in logger.messages


def test_camulator_land_stores_jax_runtime_arrays(
    monkeypatch: Any,
) -> None:
    start = datetime(2000, 1, 1, 0, 0, 0)
    forcing_ds = xr.Dataset(
        data_vars={
            "TS": (
                ("time", "lat", "lon"),
                np.asarray(
                    [
                        [[281.0, 282.0], [283.0, 284.0]],
                        [[285.0, 286.0], [287.0, 288.0]],
                    ]
                ),
            )
        },
        coords={"time": [start, datetime(2000, 1, 1, 6, 0, 0)]},
    )

    monkeypatch.setattr(
        camulator_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (jnp.ones((2, 2)), jnp.zeros((2, 2))),
    )
    monkeypatch.setattr(
        camulator_land_module,
        "load_camulator_forcing_context",
        lambda **kwargs: {
            "conf": {
                "data": {"lead_time_periods": 6},
                "predict": {"start_datetime": start},
            },
            "forcing_dataset_raw": forcing_ds,
        },
    )

    camulator_grid = RectilinearGrid(
        name="atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    ocean_grid = RectilinearGrid(
        name="ocn",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
        binary_mask=jnp.ones((2, 2)),
    )
    component = camulator_land_module.make_camulator_land(
        config_path="dummy.yaml",
        camulator_grid=camulator_grid,
        ocn_grid=ocean_grid,
    )

    component = prepare_component(
        normalize_component(component),
        _make_coupler(start),
        DTypePolicy.from_jax_config(),
    )
    assert component.spec.inputs == (
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
    )
    assert component.spec.outputs == ("land_surface_temperature",)
    assert set(component.spec.initial_fields) == {
        "land_surface_temperature",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
    }
    assert isinstance(component._data["land_surface_temperature"], jax.Array)
    assert_allclose_compact(
        component._data["land_surface_temperature"], np.full((2, 2), 283.0)
    )

    coupler = _make_coupler(start)
    initial_component_state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=ExchangeContract(),
    )
    step_context = StepContext(
        dt_seconds=(datetime(2000, 1, 1, 6, 0, 0) - start).total_seconds(),
        time=start,
        logger=coupler.logger,
    )
    component_state = step_component_runtime_state(
        component,
        initial_component_state,
        step_context,
        allow_host_runtime=True,
    )
    repeated_component_state = step_component_runtime_state(
        component,
        initial_component_state,
        step_context,
        allow_host_runtime=True,
    )
    land_surface_temperature = component_state.fields.get("land_surface_temperature")
    assert isinstance(land_surface_temperature, jax.Array)
    assert_allclose_compact(
        land_surface_temperature,
        np.asarray([[281.0, 282.0], [283.0, 284.0]]),
    )
    assert_allclose_compact(
        repeated_component_state.fields.get("land_surface_temperature"),
        np.asarray([[281.0, 282.0], [283.0, 284.0]]),
    )
    assert isinstance(
        initial_component_state.payload,
        camulator_forcing_module.CamulatorRuntimeCursor,
    )
    assert isinstance(
        component_state.payload,
        camulator_forcing_module.CamulatorRuntimeCursor,
    )
    assert isinstance(
        repeated_component_state.payload,
        camulator_forcing_module.CamulatorRuntimeCursor,
    )
    assert initial_component_state.payload.timestep_counter == 0
    assert component_state.payload.timestep_counter == 1
    assert repeated_component_state.payload.timestep_counter == 1


@pytest.mark.fast_always
def test_camulator_land_declares_radiation_exchange_inputs(
    monkeypatch: Any,
) -> None:
    start = datetime(2000, 1, 1, 0, 0, 0)
    forcing_ds = xr.Dataset(
        data_vars={
            "TS": (
                ("time", "lat", "lon"),
                np.asarray([[[281.0, 282.0], [283.0, 284.0]]]),
            )
        },
        coords={"time": [start]},
    )
    monkeypatch.setattr(
        camulator_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (jnp.ones((2, 2)), jnp.zeros((2, 2))),
    )
    monkeypatch.setattr(
        camulator_land_module,
        "load_camulator_forcing_context",
        lambda **kwargs: {
            "conf": {
                "data": {"lead_time_periods": 6},
                "predict": {"start_datetime": start},
            },
            "forcing_dataset_raw": forcing_ds,
        },
    )

    grid = RectilinearGrid(
        name="grid",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    radiation_fields = tuple(_flatten_field_items(ATMOSPHERE_TO_LAND_RADIATION_FIELDS))
    atmosphere = DataComponent(
        name="ATM",
        grid=grid,
        fields={field_name: jnp.zeros(grid.shape) for field_name in radiation_fields},
        spec=ComponentSpec(outputs=radiation_fields),
    )
    land = camulator_land_module.make_camulator_land(
        config_path="dummy.yaml",
        camulator_grid=grid,
        ocn_grid=grid,
    )
    components = {"ATM": atmosphere, "LND": land}
    contracts = build_exchange_contracts(
        tuple(components),
        (
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
            ),
        ),
        validate_endpoints=True,
    )

    assert contracts["LND"].receives == radiation_fields
    for name, component in components.items():
        validate_exchange_fields_declared(component, contracts[name])


class _FiniteCamulatorBoundaryStepper:
    """Minimal native stepper with counters for boundary-order assertions."""

    def __init__(self) -> None:
        self.build_input_calls = 0
        self.model_calls = 0
        self.postprocessing_calls = 0
        self.shift_calls = 0

    def build_input_with_forcing(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        _ = dynamic_forcing, static_forcing
        self.build_input_calls += 1
        return state.clone()

    def model(self, model_input: torch.Tensor) -> torch.Tensor:
        self.model_calls += 1
        return torch.ones_like(model_input)

    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        _ = model_input
        self.postprocessing_calls += 1
        return prediction

    def shift_state_forward(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        _ = state
        self.shift_calls += 1
        return prediction


class _InvalidSecondInputStepper(_FiniteCamulatorBoundaryStepper):
    def build_input_with_forcing(
        self,
        state: torch.Tensor,
        dynamic_forcing: torch.Tensor,
        static_forcing: torch.Tensor,
    ) -> torch.Tensor:
        model_input = super().build_input_with_forcing(
            state,
            dynamic_forcing,
            static_forcing,
        )
        model_input.flatten()[0] = torch.nan
        return model_input


class _InvalidRawPredictionStepper(_FiniteCamulatorBoundaryStepper):
    def model(self, model_input: torch.Tensor) -> torch.Tensor:
        prediction = super().model(model_input)
        if self.model_calls == 1:
            prediction.flatten()[0] = torch.nan
        return prediction

    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        processed = super()._apply_postprocessing(prediction, model_input)
        return torch.nan_to_num(processed)


class _InvalidPostprocessedPredictionStepper(_FiniteCamulatorBoundaryStepper):
    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
    ) -> torch.Tensor:
        processed = super()._apply_postprocessing(prediction, model_input)
        if self.postprocessing_calls == 1:
            processed.flatten()[0] = torch.nan
        return processed


class _InvalidShiftedStateStepper(_FiniteCamulatorBoundaryStepper):
    def shift_state_forward(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        shifted = super().shift_state_forward(state, prediction).clone()
        if self.shift_calls == 1:
            shifted.flatten()[0] = torch.nan
        return shifted


class _NoOpCamulatorInputAccessor:
    def set_state_var(
        self,
        state: torch.Tensor,
        variable_name: str,
        value: torch.Tensor,
    ) -> None:
        _ = state, variable_name, value


def _camulator_boundary_case(
    stepper: _FiniteCamulatorBoundaryStepper,
) -> tuple[Any, dict[str, jax.Array], Any]:
    """Return a deterministic two-substep CAMulator native-boundary fake."""

    start = datetime(2000, 1, 1, 0, 0, 0)
    dynamic_ds = xr.Dataset(
        data_vars={
            "F1": (
                ("time", "lat", "lon"),
                np.ones((2, 2, 2), dtype=np.float32),
            )
        },
        coords={"time": [start, datetime(2000, 1, 1, 6, 0, 0)]},
    )
    resources = SimpleNamespace(
        dynamic_ds=dynamic_ds,
        device="cpu",
        stepper=stepper,
        static_forcing=torch.zeros((1, 1, 1, 2, 2)),
        LANDM_COSLAT=jnp.zeros((2, 2)),
        accessor_input=_NoOpCamulatorInputAccessor(),
    )
    fields = {
        "sea_surface_temperature": jnp.asarray([[280.0, 281.0], [282.0, 283.0]]),
        "land_surface_temperature": jnp.zeros((2, 2)),
    }
    payload = camulator_gcm_state_module.CAMulatorRuntimePayload(
        model_state=torch.zeros((1, 1, 1, 2, 2)),
        cursor=camulator_forcing_module.CamulatorRuntimeCursor(),
    )
    return resources, fields, payload


def test_camulator_rejects_invalid_model_input_before_second_substep() -> None:
    stepper = _InvalidSecondInputStepper()
    resources, fields, payload = _camulator_boundary_case(stepper)

    with pytest.raises(ComponentError, match="CAMulator model input.*non-finite"):
        camulator_runtime_module.run_camulator_prediction_block(
            resources,
            fields,
            payload,
            block_start=0,
            block_end=2,
            logger=None,
        )

    assert stepper.model_calls == 1


def test_camulator_rejects_raw_prediction_before_postprocessing_or_reuse() -> None:
    stepper = _InvalidRawPredictionStepper()
    resources, fields, payload = _camulator_boundary_case(stepper)

    with pytest.raises(ComponentError, match="CAMulator raw prediction.*non-finite"):
        camulator_runtime_module.run_camulator_prediction_block(
            resources,
            fields,
            payload,
            block_start=0,
            block_end=2,
            logger=None,
        )

    assert stepper.model_calls == 1
    assert stepper.postprocessing_calls == 0
    assert stepper.shift_calls == 0


def test_camulator_rejects_postprocessed_prediction_before_state_reuse() -> None:
    stepper = _InvalidPostprocessedPredictionStepper()
    resources, fields, payload = _camulator_boundary_case(stepper)

    with pytest.raises(
        ComponentError,
        match="CAMulator postprocessed prediction.*non-finite",
    ):
        camulator_runtime_module.run_camulator_prediction_block(
            resources,
            fields,
            payload,
            block_start=0,
            block_end=2,
            logger=None,
        )

    assert stepper.model_calls == 1
    assert stepper.postprocessing_calls == 1
    assert stepper.shift_calls == 0


def test_camulator_rejects_shifted_state_before_next_substep() -> None:
    stepper = _InvalidShiftedStateStepper()
    resources, fields, payload = _camulator_boundary_case(stepper)

    with pytest.raises(
        ComponentError, match="CAMulator shifted model state.*non-finite"
    ):
        camulator_runtime_module.run_camulator_prediction_block(
            resources,
            fields,
            payload,
            block_start=0,
            block_end=2,
            logger=None,
        )

    assert stepper.model_calls == 1
    assert stepper.postprocessing_calls == 1
    assert stepper.shift_calls == 1


def test_camulator_runtime_step_is_reproducible_from_one_payload(
    monkeypatch: Any,
) -> None:
    start = datetime(2000, 1, 1, 0, 0, 0)
    dynamic_ds = xr.Dataset(
        data_vars={
            "F1": (
                ("time", "lat", "lon"),
                np.asarray(
                    [
                        [[1.0, 2.0], [3.0, 4.0]],
                        [[5.0, 6.0], [7.0, 8.0]],
                    ]
                ),
            ),
            "F2": (
                ("time", "lat", "lon"),
                np.asarray(
                    [
                        [[10.0, 20.0], [30.0, 40.0]],
                        [[50.0, 60.0], [70.0, 80.0]],
                    ]
                ),
            ),
        },
        coords={"time": [start, datetime(2000, 1, 1, 6, 0, 0)]},
    )
    captured: dict[str, torch.Tensor] = {}

    class _Stepper:
        def build_input_with_forcing(
            self,
            state: torch.Tensor,
            dynamic_forcing: torch.Tensor,
            static_forcing: torch.Tensor,
        ) -> torch.Tensor:
            _ = static_forcing
            assert bool(torch.all(torch.isfinite(dynamic_forcing)))
            captured["dynamic_forcing"] = dynamic_forcing.detach().cpu()
            return state

        def shift_state_forward(
            self, state: torch.Tensor, prediction: torch.Tensor
        ) -> torch.Tensor:
            _ = state
            return prediction

        def model(self, model_input: torch.Tensor) -> torch.Tensor:
            return torch.full_like(model_input, float(model_input.flatten()[0] + 1))

        def _apply_postprocessing(
            self,
            prediction: torch.Tensor,
            model_input: torch.Tensor,
        ) -> torch.Tensor:
            _ = model_input
            return prediction

    class _StepAccessor:
        def set_state_var(self, state: Any, variable_name: str, value: Any) -> None:
            assert variable_name == "SST"
            assert bool(torch.all(torch.isfinite(value)))
            captured["sst"] = value.detach().cpu()
            first_sst_channel = state.shape[1] - value.shape[1]
            state[:, first_sst_channel:, ...].copy_(value)

    class _OutputAccessor:
        def get_state_var(
            self, state: torch.Tensor, variable_name: str
        ) -> torch.Tensor:
            _ = state, variable_name
            return torch.ones((1, 2, 2, 2), dtype=torch.float32)

    component = cast(
        Any,
        camulator_gcm_state_module.CAMulatorGCMSetupState.__new__(
            camulator_gcm_state_module.CAMulatorGCMSetupState
        ),
    )
    component.model_substeps = 2
    cursor = camulator_forcing_module.CamulatorRuntimeCursor(
        start_ix=0,
        init_str="2000-01-01T00Z",
        model_substeps=2,
        timestep_counter=0,
    )
    component.dynamic_ds = dynamic_ds
    component.device = "cpu"
    component.stepper = _Stepper()
    component.static_forcing = torch.zeros((1, 1, 1, 2, 2))
    component.LANDM_COSLAT = jnp.asarray([[0.0, 1.0], [0.5, 0.0]])
    component.name = "ATM"
    component.grid = RectilinearGrid(
        name="atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    component._dtype_policy = DTypePolicy()
    component._data = camulator_fields_module.initialize_camulator_runtime_fields(
        component.grid.shape,
        component._dtype_policy,
    )
    component._data["sea_surface_temperature"] = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    component._data["land_surface_temperature"] = jnp.asarray(
        [[10.0, 20.0], [30.0, 40.0]]
    )
    component.accessor_input = _StepAccessor()
    component.accessor_output = _OutputAccessor()
    component.latlons = SimpleNamespace(
        latitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
        longitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
    )
    component.conf = _camulator_output_conf(
        surface_variables=[],
        diagnostic_variables=[],
        save_vars=["U"],
    )
    component.lead_time_periods = 6
    component.output_frequency = None
    component.metadata = {}
    component.state_transformer = SimpleNamespace(
        inverse_transform=lambda prediction: prediction
    )
    component.P0 = 100000.0
    component.hyai = torch.ones((1, 2, 1, 1))
    component.hybi = torch.ones((1, 2, 1, 1))
    component.hyam = torch.ones((1, 2, 1, 1))
    component.hybm = torch.ones((1, 2, 1, 1))

    def map_runtime_fields(*args: Any, **kwargs: Any) -> dict[str, jax.Array]:
        _ = args
        captured["runtime_prediction"] = kwargs["prediction"].detach().cpu()
        return {"temperature": jnp.full((2, 2), 9.0)}

    monkeypatch.setattr(
        camulator_fields_module,
        "map_camulator_prediction_to_runtime_fields",
        map_runtime_fields,
    )

    component_state = _runtime_component_state("ATM", component._data)
    step_context = StepContext(
        dt_seconds=float((datetime(2000, 1, 1, 6, 0, 0) - start).total_seconds()),
        time=start,
        logger=cast(Any, _RecordingLogger()),
    )
    initial_payload = camulator_gcm_state_module.CAMulatorRuntimePayload(
        model_state=torch.zeros((1, 6, 1, 2, 2)),
        cursor=cursor,
    )
    first = camulator_runtime_module.step_camulator_runtime(
        component,
        component_state.fields.to_mapping(),
        step_context,
        initial_payload,
    )
    second = camulator_runtime_module.step_camulator_runtime(
        component,
        component_state.fields.to_mapping(),
        step_context,
        initial_payload,
    )
    component_state = component_state.with_fields(
        component_state.fields.set_many(first.fields)
    )

    assert isinstance(first, StepResult)
    assert isinstance(second, StepResult)
    assert first.payload.cursor.timestep_counter == 1
    assert second.payload.cursor.timestep_counter == 1
    assert initial_payload.cursor.timestep_counter == 0
    assert torch.count_nonzero(initial_payload.model_state).item() == 0
    assert all(
        np.all(np.isfinite(np.asarray(value))) for value in first.fields.values()
    )
    assert torch.equal(
        first.payload.output_prediction,
        second.payload.output_prediction,
    )
    assert captured["dynamic_forcing"].shape == (1, 2, 1, 2, 2)
    assert_allclose_compact(
        captured["dynamic_forcing"][0, :, 0],
        np.asarray(
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
    )
    assert captured["sst"].shape == (1, 1, 1, 2, 2)
    assert isinstance(
        component_state.fields.get("total_surface_temperature"), jax.Array
    )
    assert_allclose_compact(
        component_state.fields.get("total_surface_temperature"),
        np.asarray([[11.0, 283.0], [33.0, 44.0]]),
    )
    assert_allclose_compact(
        component_state.fields.get("temperature"), np.full((2, 2), 9.0)
    )
    assert_allclose_compact(
        first.payload.output_prediction,
        torch.full_like(first.payload.output_prediction, 2.0),
    )
    assert_allclose_compact(
        captured["runtime_prediction"],
        torch.full_like(captured["runtime_prediction"], 2.0),
    )
    assert first.payload.output_prediction_samples.shape[0] == 2
    assert_allclose_compact(
        first.payload.output_prediction_samples[:, 0, 0, 0, 0],
        np.asarray([1.0, 2.0]),
    )
