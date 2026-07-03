from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xarray as xr

import vercor.setups.external.camulator_contracts as camulator_contracts_module
import vercor.setups.external.camulator_fields as camulator_fields_module
import vercor.setups.external.camulator_forcing as camulator_forcing_module
import vercor.setups.external.camulator_gcm_state as camulator_gcm_state_module
import vercor.setups.external.camulator_imports as camulator_imports_module
import vercor.setups.external.camulator_init as camulator_init_module
import vercor.setups.external.camulator_land as camulator_land_module
import vercor.setups.external.camulator_output as camulator_output_module
import vercor.setups.external.camulator_runtime as camulator_runtime_module
import vercor.setups.external.camulator_tensors as camulator_tensors_module
import vercor.setups.external.camulator_wind_filter as camulator_wind_filter_module
from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact
from vercor.components.contexts import SetupContext, StepContext
from vercor.output.adapters import ComponentOutputAdapter, component_snapshot_writer
from vercor.output.variables import OutputVariable
from vercor.setups.external.camulator import make_camulator_gcm
from vercor.fluxes.vertical_coordinates import get_altitudes_hybrid_sigma_levels
from vercor.grid import RectilinearGrid
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.component_state import create_runtime_component_state
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.settings import VercorSettings
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
) -> RuntimeComponentState:
    _ = name
    return RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(data or {}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )


def _make_camulator_output_adapter() -> ComponentOutputAdapter:
    return ComponentOutputAdapter(
        empty_error_message=camulator_output_module.CAMULATOR_AVERAGE_EMPTY_ERROR_MESSAGE,
        time_dim=camulator_output_module.CAMULATOR_TIME_DIM,
    )


def _make_coupler(start: datetime) -> SetupContext:
    return SetupContext(
        start=start,
        dt_seconds=21600,
        run_sequence=(),
        settings=VercorSettings(),
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
    import vercor.setups.external._camulator_wind_filtering as wind_filtering

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


def test_prepare_camulator_dynamic_forcing_chunk_supports_jit_and_ordering() -> None:
    values = jnp.arange(2 * 3 * 2 * 2, dtype=jnp.float32).reshape(2, 3, 2, 2)

    prepared = jax.jit(camulator_fields_module.prepare_camulator_dynamic_forcing_chunk)(
        values
    )

    assert prepared.shape == (3, 2, 2, 2)
    assert_allclose_compact(prepared, np.asarray(values).transpose((1, 0, 2, 3)))


def test_prepare_camulator_sst_input_supports_jit_and_shape() -> None:
    surface_temperature = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    prepared = jax.jit(camulator_fields_module.prepare_camulator_sst_input)(
        surface_temperature
    )

    assert prepared.shape == (1, 1, 1, 2, 2)
    assert_allclose_compact(prepared[0, 0, 0], np.asarray(surface_temperature))


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


def test_build_camulator_output_variables_preserves_credit_channel_order() -> None:
    prediction = _camulator_prediction(total_channels=7)
    coords, data_vars = camulator_output_module.build_camulator_output_variables(
        prediction,
        datetime(2000, 1, 1, 6, 0, 0),
        latitude=np.asarray([-45.0, 45.0]),
        longitude=np.asarray([0.0, 90.0]),
        forecast_hour=12,
        metadata={
            "T": {"units": "K"},
            "latitude": {"units": "degrees_north"},
        },
        conf=_camulator_output_conf(diagnostic_variables=[]),
    )

    assert coords["level"].dims == ("level",)
    assert_allclose_compact(coords["level"].values, np.asarray([10, 20, 30]))
    assert coords["latitude"].attrs["units"] == "degrees_north"
    assert data_vars["U"].dims == ("time", "level", "latitude", "longitude")
    assert data_vars["T"].dims == ("time", "level", "latitude", "longitude")
    assert data_vars["PS"].dims == ("time", "latitude", "longitude")
    assert data_vars["forecast_hour"].dims == ()
    assert data_vars["T"].attrs["units"] == "K"
    assert int(np.asarray(data_vars["forecast_hour"].values)) == 12
    assert_allclose_compact(
        data_vars["U"].values,
        prediction[:, 0:3, 0].reshape(1, 3, 2, 2).numpy(),
    )
    assert_allclose_compact(
        data_vars["T"].values,
        prediction[:, 3:6, 0].reshape(1, 3, 2, 2).numpy(),
    )
    assert_allclose_compact(data_vars["PS"].values, prediction[:, 6, 0].numpy())


def test_build_camulator_output_variables_supports_upper_air_only() -> None:
    prediction = _camulator_prediction(total_channels=6)

    _, data_vars = camulator_output_module.build_camulator_output_variables(
        prediction,
        datetime(2000, 1, 1),
        latitude=np.asarray([-45.0, 45.0]),
        longitude=np.asarray([0.0, 90.0]),
        forecast_hour=6,
        metadata={},
        conf=_camulator_output_conf(surface_variables=[], diagnostic_variables=[]),
    )

    assert tuple(data_vars) == ("U", "T", "forecast_hour")


def test_write_camulator_prediction_output_uses_vercor_h5netcdf_boundary(
    tmp_path: Path,
) -> None:
    class _Transformer:
        def __init__(self) -> None:
            self.calls = 0

        def inverse_transform(self, prediction: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return prediction + 100.0

    transformer = _Transformer()
    prediction = _camulator_prediction(total_channels=8)
    conf = _camulator_output_conf(
        save_forecast=str(tmp_path),
        save_vars=["T", "FSNS"],
        climate_rescale_output=True,
    )
    logger = _RecordingLogger()

    camulator_output_module.write_camulator_prediction_output(
        prediction,
        datetime(2000, 1, 1, 6, 0, 0),
        latitude=np.asarray([-45.0, 45.0]),
        longitude=np.asarray([0.0, 90.0]),
        init_str="2000-01-01T00Z",
        lead_time_periods=6,
        forecast_hour=2,
        metadata={
            "T": {"units": "K"},
            "FSNS": {"long_name": "surface net shortwave flux"},
        },
        conf=conf,
        state_transformer=transformer,
        logger=cast(Any, logger),
    )

    output = tmp_path / "2000-01-01T00Z" / "pred_2000-01-01T00Z_012.nc"
    assert logger.messages == [f"Writing output file:  {output}"]
    assert transformer.calls == 1
    with h5netcdf.File(output, "r") as actual:
        assert "U" not in actual.variables
        assert "PS" not in actual.variables
        assert actual.variables["T"].dimensions == (
            "time",
            "level",
            "latitude",
            "longitude",
        )
        assert actual.variables["FSNS"].dimensions == (
            "time",
            "latitude",
            "longitude",
        )
        assert actual.variables["T"].attrs["units"] == "K"
        assert (
            actual.variables["FSNS"].attrs["long_name"] == "surface net shortwave flux"
        )
        assert int(actual.variables["forecast_hour"][()]) == 12
        assert_allclose_compact(
            actual.variables["level"][:],
            np.asarray([10, 20, 30]),
        )
        assert_allclose_compact(
            actual.variables["T"][:],
            (prediction[:, 3:6, 0] + 100.0).reshape(1, 3, 2, 2).numpy(),
        )
        assert_allclose_compact(
            actual.variables["FSNS"][:],
            (prediction[:, 7, 0] + 100.0).numpy(),
        )


def test_camulator_period_output_variables_reduce_time_axis_with_adapter() -> None:
    prediction = torch.arange(2 * 8 * 2 * 2, dtype=torch.float32).reshape(2, 8, 1, 2, 2)
    adapter = _make_camulator_output_adapter()

    adapter.accumulate(
        camulator_output_module.camulator_period_output_variables(
            prediction,
            metadata={
                "T": {"units": "K"},
                "FSNS": {"long_name": "surface net shortwave flux"},
            },
            conf=_camulator_output_conf(save_vars=["T", "FSNS"]),
            state_transformer=None,
        ),
        summation_dim=camulator_output_module.CAMULATOR_TIME_DIM,
    )

    assert tuple(adapter.variables) == ("T", "FSNS")
    assert adapter.variables["T"].dims == ("level", "latitude", "longitude")
    assert adapter.variables["T"].attrs["units"] == "K"
    assert adapter.variables["FSNS"].attrs["long_name"] == "surface net shortwave flux"
    assert_allclose_compact(
        adapter.variables["T"].counts,
        np.full((3, 2, 2), 2),
    )
    assert_allclose_compact(
        adapter.accumulator.mean_samples()["T"].values,
        np.mean(prediction[:, 3:6, 0].numpy(), axis=0),
    )


def test_camulator_output_adapter_persists_mean_dataset(
    tmp_path: Path,
) -> None:
    adapter = _make_camulator_output_adapter()
    first_prediction = _camulator_prediction(total_channels=8)
    second_prediction = first_prediction + 80.0
    conf = _camulator_output_conf(save_forecast=str(tmp_path), save_vars=["T", "FSNS"])
    logger = _RecordingLogger()
    for prediction in (first_prediction, second_prediction):
        adapter.accumulate(
            camulator_output_module.camulator_period_output_variables(
                prediction,
                metadata={
                    "T": {"units": "K"},
                    "FSNS": {"long_name": "surface net shortwave flux"},
                },
                conf=conf,
                state_transformer=None,
            ),
            summation_dim=camulator_output_module.CAMULATOR_TIME_DIM,
        )

    output_time = datetime(2000, 1, 2, 0, 0, 0)
    metadata = {
        "T": {"units": "K"},
        "FSNS": {"long_name": "surface net shortwave flux"},
        "latitude": {"units": "degrees_north"},
    }
    output = camulator_output_module.camulator_average_output_path(
        output_time=datetime(2000, 1, 2, 0, 0, 0),
        init_str="2000-01-01T00Z",
        conf=conf,
    )
    adapter.write_period_average(
        output,
        build_coordinate_variables=lambda variables: (
            camulator_output_module.camulator_average_coordinate_variables(
                variables,
                output_time=output_time,
                latitude=np.asarray([-45.0, 45.0]),
                longitude=np.asarray([0.0, 90.0]),
                metadata=metadata,
                conf=conf,
            )
        ),
        build_data_variables=lambda variables: (
            camulator_output_module.camulator_average_data_variables(
                variables,
                metadata=metadata,
            )
        ),
        logger=cast(Any, logger),
    )

    assert output == str(
        tmp_path / "2000-01-01T00Z" / "camulator.averages.2000-01-02.nc"
    )
    assert logger.messages == [f"Writing output file:  {output}"]
    with h5netcdf.File(output, "r") as actual:
        assert "U" not in actual.variables
        assert "PS" not in actual.variables
        assert actual.variables["T"].dimensions == (
            "time",
            "level",
            "latitude",
            "longitude",
        )
        assert actual.variables["FSNS"].dimensions == (
            "time",
            "latitude",
            "longitude",
        )
        assert actual.variables["T"].attrs["units"] == "K"
        assert (
            actual.variables["FSNS"].attrs["long_name"] == "surface net shortwave flux"
        )
        assert actual.variables["latitude"].attrs["units"] == "degrees_north"
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
        assert_allclose_compact(actual.variables["level"][:], np.asarray([10, 20, 30]))
        assert_allclose_compact(
            actual.variables["T"][:],
            np.mean(
                np.stack(
                    [
                        first_prediction[:, 3:6, 0].reshape(1, 3, 2, 2).numpy(),
                        second_prediction[:, 3:6, 0].reshape(1, 3, 2, 2).numpy(),
                    ]
                ),
                axis=0,
            ),
        )
        assert_allclose_compact(
            actual.variables["FSNS"][:],
            np.mean(
                np.stack(
                    [
                        first_prediction[:, 7, 0].numpy(),
                        second_prediction[:, 7, 0].numpy(),
                    ]
                ),
                axis=0,
            ),
        )
    assert adapter.empty


def test_camulator_record_period_output_accumulates_and_writes_mean_dataset(
    tmp_path: Path,
) -> None:
    adapter = _make_camulator_output_adapter()
    first_prediction = _camulator_prediction(total_channels=8)
    second_prediction = first_prediction + 80.0
    conf = _camulator_output_conf(save_forecast=str(tmp_path), save_vars=["T", "FSNS"])
    metadata = {
        "T": {"units": "K"},
        "FSNS": {"long_name": "surface net shortwave flux"},
        "latitude": {"units": "degrees_north"},
    }

    first_written = camulator_output_module.record_camulator_period_output(
        adapter,
        first_prediction,
        output_time=datetime(2000, 1, 1, 0, 0, 0),
        dt=timedelta(hours=1),
        output_frequency="day",
        latitude=np.asarray([-45.0, 45.0]),
        longitude=np.asarray([0.0, 90.0]),
        init_str="2000-01-01T00Z",
        metadata=metadata,
        conf=conf,
        state_transformer=None,
    )
    second_written = camulator_output_module.record_camulator_period_output(
        adapter,
        second_prediction,
        output_time=datetime(2000, 1, 2, 0, 0, 0),
        dt=timedelta(days=1),
        output_frequency=None,
        latitude=np.asarray([-45.0, 45.0]),
        longitude=np.asarray([0.0, 90.0]),
        init_str="2000-01-01T00Z",
        metadata=metadata,
        conf=conf,
        state_transformer=None,
    )

    output = tmp_path / "2000-01-01T00Z" / "camulator.averages.2000-01-02.nc"
    assert not first_written
    assert second_written
    assert adapter.empty
    with h5netcdf.File(output, "r") as actual:
        assert actual.variables["T"].dimensions == (
            "time",
            "level",
            "latitude",
            "longitude",
        )
        assert actual.variables["T"].attrs["units"] == "K"
        assert (
            actual.variables["FSNS"].attrs["long_name"] == "surface net shortwave flux"
        )
        assert_allclose_compact(actual.variables["latitude"][:], [-45.0, 45.0])


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
    tmp_path: Path,
) -> None:
    conf = _camulator_output_conf(save_forecast=str(tmp_path))
    for section, value in config_update.items():
        if isinstance(value, dict) and isinstance(conf.get(section), dict):
            conf[section].update(value)
        else:
            conf[section] = value

    with pytest.raises(ValueError, match=message):
        camulator_output_module.write_camulator_prediction_output(
            _camulator_prediction(total_channels=8),
            datetime(2000, 1, 1),
            latitude=np.asarray([-45.0, 45.0]),
            longitude=np.asarray([0.0, 90.0]),
            init_str="2000-01-01T00Z",
            lead_time_periods=6,
            forecast_hour=1,
            metadata={},
            conf=conf,
            state_transformer=None,
        )


def test_camulator_output_wrappers_do_not_import_xarray_or_credit_output() -> None:
    camulator_output_source = Path(
        "vercor/setups/external/camulator_output.py"
    ).read_text(encoding="utf-8")
    camulator_runtime_source = Path(
        "vercor/setups/external/camulator_runtime.py"
    ).read_text(encoding="utf-8")
    camulator_imports_source = Path(
        "vercor/setups/external/camulator_imports.py"
    ).read_text(encoding="utf-8")
    output_adapters_source = Path("vercor/output/adapters.py").read_text(
        encoding="utf-8"
    )

    assert "import xarray" not in camulator_output_source
    assert "credit.output" not in camulator_output_source
    assert "credit.output" not in camulator_imports_source
    assert "accumulate_output_variables(" not in camulator_output_source
    assert "period_mean_output_variables(" not in camulator_output_source
    assert "write_period_average_netcdf(" not in camulator_output_source
    assert "should_write_period_output(" not in camulator_runtime_source
    assert "should_write_period_output(" in output_adapters_source


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
    source = Path("vercor/setups/external/camulator_tensors.py").read_text(
        encoding="utf-8"
    )

    assert "class TensorVariableIndex" in source
    assert "def _append_indexed_variables(" in source
    assert "def _mark_unavailable_variables(" in source
    assert "def get_var_index(" in source


def test_map_camulator_prediction_arrays_supports_jit_and_preserves_conventions() -> (
    None
):
    settings = VercorSettings()
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
        settings.earth_radius,
        settings.gravity,
        settings.rdair,
        settings.zvir,
        settings.mwdair,
        settings.rgas,
        settings.p0,
        settings.cappa,
        settings.stefBoltz,
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
        settings.stefBoltz * np.asarray(surface_temperature) ** 4 - 3.0,
    )
    assert mapped_fields["model_level_height"].shape == (2, 2)
    assert mapped_fields["density"].shape == (2, 2)
    assert mapped_fields["potential_temperature"].shape == (2, 2)
    pressure_interfaces = (
        hyai[:, jnp.newaxis, jnp.newaxis] * 100000.0
        + hybi[:, jnp.newaxis, jnp.newaxis] * surface_pressure[jnp.newaxis, :, :]
    )
    expected_model_level_height = get_altitudes_hybrid_sigma_levels(
        settings,
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


def test_camulator_constructor_builds_jax_backed_grid(monkeypatch: Any) -> None:
    latlons = SimpleNamespace(
        longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        latitude=SimpleNamespace(values=np.asarray([-45.0, 0.0, 45.0])),
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
        config_path="dummy.yaml",
        device="cpu",
        output_frequency="month",
    )

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.field_spec.inputs == (
        "sea_surface_temperature",
        "land_surface_temperature",
    )
    assert (
        component.field_spec.outputs
        == camulator_contracts_module.CAMULATOR_RUNTIME_FIELD_NAMES
    )
    assert_allclose_compact(component.grid.binary_mask, np.ones((3, 2)))
    assert callable(component_snapshot_writer(component))


def test_camulator_constructor_logs_save_forecast_path(monkeypatch: Any) -> None:
    latlons = SimpleNamespace(
        longitude=SimpleNamespace(values=np.asarray([0.0, 90.0])),
        latitude=SimpleNamespace(values=np.asarray([-45.0, 0.0, 45.0])),
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
                "predict": {
                    "save_forecast": "/tmp/camulator-output",
                    "timesteps_fast_climate": 1,
                },
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
    with capture_logger_output(DEFAULT_LOGGER_NAME) as stream:
        make_camulator_gcm(
            config_path="dummy.yaml",
            device="cpu",
            output_subfolder_name="member-001",
        )

    assert "Saving outputs to: /tmp/camulator-output/member-001" in stream.getvalue()


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
        raising=False,
    )
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

    component.initialize(_make_coupler(start))
    assert component.field_spec.outputs == ("land_surface_temperature",)
    assert set(component.field_spec.defaults) == {"land_surface_temperature"}
    assert isinstance(component.data["land_surface_temperature"], jax.Array)
    assert_allclose_compact(
        component.data["land_surface_temperature"], np.full((2, 2), 283.0)
    )

    coupler = _make_coupler(start)
    component_state = component.step_host_runtime_state(
        create_runtime_component_state(
            component,
            prefill_missing=True,
            contract=RuntimeComponentContract(),
        ),
        StepContext(
            dt_seconds=(datetime(2000, 1, 1, 6, 0, 0) - start).total_seconds(),
            settings=coupler.settings,
            time=start,
            logger=coupler.logger,
        ),
    )
    land_surface_temperature = component_state.data.get("land_surface_temperature")
    assert isinstance(land_surface_temperature, jax.Array)
    assert_allclose_compact(
        land_surface_temperature,
        np.asarray([[281.0, 282.0], [283.0, 284.0]]),
    )


def test_camulator_step_uses_jax_prepared_forcing_boundaries(
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
            captured["dynamic_forcing"] = dynamic_forcing.detach().cpu()
            return state

        def shift_state_forward(
            self, state: torch.Tensor, prediction: torch.Tensor
        ) -> torch.Tensor:
            _ = prediction
            return state

        def model(self, model_input: torch.Tensor) -> torch.Tensor:
            return model_input + 1.0

        def _apply_postprocessing(
            self,
            prediction: torch.Tensor,
            model_input: torch.Tensor,
        ) -> torch.Tensor:
            _ = model_input
            return prediction

    class _StepAccessor:
        def set_state_var(
            self, state: torch.Tensor, variable_name: str, value: torch.Tensor
        ) -> None:
            _ = state
            assert variable_name == "SST"
            captured["sst"] = value.detach().cpu()

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
    component.runtime_cursor = camulator_forcing_module.CamulatorRuntimeCursor(
        start_ix=0,
        init_str="2000-01-01T00Z",
        model_substeps=2,
        timestep_counter=0,
    )
    component.dynamic_ds = dynamic_ds
    component.device = "cpu"
    component.stepper = _Stepper()
    component.static_forcing = torch.zeros((1, 1, 1, 2, 2))
    component.state = torch.zeros((1, 6, 1, 2, 2))
    component.LANDM_COSLAT = jnp.asarray([[0.0, 1.0], [0.5, 0.0]])
    component.name = "ATM"
    component.grid = RectilinearGrid(
        name="atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    component.settings = VercorSettings()
    component.data = camulator_fields_module.initialize_camulator_runtime_fields(
        component.grid.shape,
        component.settings,
    )
    component.data["sea_surface_temperature"] = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    component.data["land_surface_temperature"] = jnp.asarray(
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
    component.output_adapter = _make_camulator_output_adapter()
    component.forecast_hour = 1
    component.metadata = {}
    component.state_transformer = SimpleNamespace(
        inverse_transform=lambda prediction: prediction
    )
    component.P0 = 100000.0
    component.hyai = torch.ones((1, 2, 1, 1))
    component.hybi = torch.ones((1, 2, 1, 1))
    component.hyam = torch.ones((1, 2, 1, 1))
    component.hybm = torch.ones((1, 2, 1, 1))

    output_calls: dict[str, Any] = {}

    def fake_write_camulator_prediction_output(*args: Any, **kwargs: Any) -> None:
        output_calls["args"] = args
        output_calls["kwargs"] = kwargs

    monkeypatch.setattr(
        camulator_output_module,
        "write_camulator_prediction_output",
        fake_write_camulator_prediction_output,
    )
    monkeypatch.setattr(
        camulator_fields_module,
        "map_camulator_prediction_arrays",
        lambda *args: {"temperature": jnp.full((2, 2), 9.0)},
    )

    component_state = _runtime_component_state("ATM", component.data)
    step_context = StepContext(
        dt_seconds=float((datetime(2000, 1, 1, 6, 0, 0) - start).total_seconds()),
        settings=VercorSettings(),
        time=start,
        logger=cast(Any, _RecordingLogger()),
    )
    updates = camulator_runtime_module.step_camulator_runtime(
        component,
        component_state.data.to_mapping(),
        step_context,
        None,
    )
    component_state = component_state.with_data(component_state.data.set_many(updates))

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
    assert isinstance(component_state.data.get("total_surface_temperature"), jax.Array)
    assert_allclose_compact(
        component_state.data.get("total_surface_temperature"),
        np.asarray([[11.0, 283.0], [33.0, 44.0]]),
    )
    assert_allclose_compact(
        component_state.data.get("temperature"), np.full((2, 2), 9.0)
    )
    assert component.runtime_cursor.timestep_counter == 1
    assert output_calls["kwargs"]["state_transformer"] is component.state_transformer


@pytest.mark.parametrize("output_frequency", [None, "day"])
def test_record_camulator_output_records_latest_snapshot_in_all_output_modes(
    monkeypatch: Any,
    tmp_path: Path,
    output_frequency: str | None,
) -> None:
    component = cast(
        Any,
        camulator_gcm_state_module.CAMulatorGCMSetupState.__new__(
            camulator_gcm_state_module.CAMulatorGCMSetupState
        ),
    )
    component.runtime_cursor = camulator_forcing_module.CamulatorRuntimeCursor(
        start_ix=0,
        init_str="2000-01-01T00Z",
        model_substeps=1,
        timestep_counter=0,
    )
    component.output_frequency = output_frequency
    component.output_adapter = _make_camulator_output_adapter()
    component.latlons = SimpleNamespace(
        latitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
        longitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
    )
    component.metadata = {"T": {"units": "K"}}
    component.conf = _camulator_output_conf(
        save_forecast=str(tmp_path),
        save_vars=["T"],
    )
    component.state_transformer = None
    component.lead_time_periods = 6
    component.forecast_hour = 1
    prediction = torch.arange(2 * 8 * 1 * 2 * 2, dtype=torch.float32).reshape(
        2, 8, 1, 2, 2
    )

    monkeypatch.setattr(
        camulator_output_module,
        "write_camulator_prediction_output",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        camulator_output_module,
        "record_camulator_period_output",
        lambda *args, **kwargs: False,
    )

    camulator_runtime_module.record_camulator_prediction_output(
        component,
        prediction=prediction,
        utc_datetime=datetime(2000, 1, 2),
        logger=cast(Any, _RecordingLogger()),
    )

    assert not component.output_adapter.snapshot_empty
    assert component.output_adapter.snapshot_time == datetime(2000, 1, 2)
    assert tuple(component.output_adapter.snapshot_variables) == ("T",)
    assert component.output_adapter.snapshot_variables["T"].dims == (
        "level",
        "latitude",
        "longitude",
    )
    assert component.output_adapter.snapshot_variables["T"].attrs["units"] == "K"
    assert_allclose_compact(
        component.output_adapter.snapshot_variables["T"].values,
        np.mean(prediction[:, 3:6, 0].numpy(), axis=0),
    )


def test_record_camulator_output_averages_when_frequency_is_configured(
    monkeypatch: Any,
) -> None:
    component = cast(
        Any,
        camulator_gcm_state_module.CAMulatorGCMSetupState.__new__(
            camulator_gcm_state_module.CAMulatorGCMSetupState
        ),
    )
    component.runtime_cursor = camulator_forcing_module.CamulatorRuntimeCursor(
        start_ix=0,
        init_str="2000-01-01T00Z",
        model_substeps=1,
        timestep_counter=0,
    )
    component.output_frequency = "day"
    component.output_adapter = _make_camulator_output_adapter()
    component.latlons = SimpleNamespace(
        latitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
        longitude=SimpleNamespace(values=np.asarray([0.0, 1.0])),
    )
    component.metadata = {"T": {"units": "K"}}
    component.conf = _camulator_output_conf(save_forecast="/tmp/camulator-output")
    component.state_transformer = object()
    component.lead_time_periods = 6
    prediction = torch.ones((1, 8, 1, 2, 2), dtype=torch.float32)

    written: dict[str, Any] = {}

    def fake_record_camulator_period_output(
        adapter: ComponentOutputAdapter,
        prediction_arg: torch.Tensor,
        *,
        output_time: datetime,
        dt: timedelta,
        output_frequency: str | None,
        latitude: object,
        longitude: object,
        init_str: str,
        metadata: dict[str, Any],
        conf: dict[str, Any],
        state_transformer: Any | None,
        logger: Any | None = None,
    ) -> bool:
        _ = logger
        written["adapter"] = adapter
        written["prediction"] = prediction_arg
        written["metadata"] = metadata
        written["conf"] = conf
        written["state_transformer"] = state_transformer
        written["output_time"] = output_time
        written["dt"] = dt
        written["output_frequency"] = output_frequency
        written["latitude"] = latitude
        written["longitude"] = longitude
        written["path"] = (
            f"/tmp/camulator-output/{init_str}/"
            f"camulator.averages.{output_time.strftime('%Y-%m-%d')}.nc"
        )
        adapter.accumulate(
            {"T": OutputVariable(("time", "x"), np.asarray([[1.0]]))},
            summation_dim=camulator_output_module.CAMULATOR_TIME_DIM,
        )
        written["counts"] = adapter.variables["T"].counts.copy()
        adapter.reset()
        return True

    monkeypatch.setattr(
        camulator_output_module,
        "write_camulator_prediction_output",
        lambda *args, **kwargs: pytest.fail("frequency mode must not write increments"),
    )
    monkeypatch.setattr(
        camulator_output_module,
        "record_camulator_period_output",
        fake_record_camulator_period_output,
    )

    camulator_runtime_module.record_camulator_prediction_output(
        component,
        prediction=prediction,
        utc_datetime=datetime(2000, 1, 2),
        logger=cast(Any, _RecordingLogger()),
    )

    assert written["adapter"] is component.output_adapter
    assert written["prediction"] is prediction
    assert written["metadata"] is component.metadata
    assert written["conf"] is component.conf
    assert written["state_transformer"] is component.state_transformer
    assert_allclose_compact(written["counts"], np.asarray([1]))
    assert written["output_time"] == datetime(2000, 1, 2)
    assert written["dt"] == timedelta(hours=6)
    assert written["output_frequency"] == "day"
    assert_allclose_compact(written["latitude"], np.asarray([0.0, 1.0]))
    assert_allclose_compact(written["longitude"], np.asarray([0.0, 1.0]))
    assert written["path"].endswith("/2000-01-01T00Z/camulator.averages.2000-01-02.nc")
    assert component.output_adapter.empty
