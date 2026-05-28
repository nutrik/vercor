from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import xarray as xr

import vercor.setups.external.camulator as camulator_module
import vercor.setups.external.camulator_contracts as camulator_contracts_module
import vercor.setups.external.camulator_fields as camulator_fields_module
import vercor.setups.external.camulator_forcing as camulator_forcing_module
import vercor.setups.external.camulator_imports as camulator_imports_module
import vercor.setups.external.camulator_init as camulator_init_module
import vercor.setups.external.camulator_land as camulator_land_module
import vercor.setups.external.camulator_output as camulator_output_module
import vercor.setups.external.camulator_runtime as camulator_runtime_module
import vercor.setups.external.camulator_tensors as camulator_tensors_module
import vercor.setups.external.camulator_wind_filter as camulator_wind_filter_module
from tests._coverage_support import capture_logger_output
from tests.assertions import assert_allclose_compact
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.setups.external.camulator import make_camulator_gcm
from vercor.setups.external.camulator_fields import (
    _initialize_camulator_runtime_fields,
    _map_camulator_prediction_arrays,
    _prepare_camulator_dynamic_forcing_chunk,
    _prepare_camulator_sst_input,
    _prepare_camulator_surface_forcing,
)
from vercor.setups.external.camulator_tensors import _torch_tensor_from_jax_array
from vercor.fluxes.vertical_coordinates import get_altitudes_hybrid_sigma_levels
from vercor.grid import RectilinearGrid
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.component_state import create_runtime_component_state
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.run_sequence import RunSequence
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


def _make_coupler(start: datetime) -> ComponentInitContext:
    return ComponentInitContext(
        start=start,
        dt_seconds=21600,
        run_sequence=RunSequence(order=[]),
        settings=VercorSettings(),
        logger=cast(Any, _RecordingLogger()),
    )


def test_camulator_runtime_field_initializer_returns_jax_arrays() -> None:
    fields = _initialize_camulator_runtime_fields((2, 3))

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


def test_prepare_camulator_surface_forcing_supports_jit_and_gradients() -> None:
    sea_surface_temperature = jnp.asarray([[jnp.nan, 2.0], [5.0, 7.0]])
    land_surface_temperature = jnp.asarray([[10.0, jnp.nan], [15.0, 20.0]])
    land_mask_coslat = jnp.asarray([[0.5, 1.0], [1.2, 0.0]])

    total_surface_temperature, rescaled_total_surface_temperature = jax.jit(
        _prepare_camulator_surface_forcing
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
            _prepare_camulator_surface_forcing(
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

    prepared = jax.jit(_prepare_camulator_dynamic_forcing_chunk)(values)

    assert prepared.shape == (3, 2, 2, 2)
    assert_allclose_compact(prepared, np.asarray(values).transpose((1, 0, 2, 3)))


def test_prepare_camulator_sst_input_supports_jit_and_shape() -> None:
    surface_temperature = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    prepared = jax.jit(_prepare_camulator_sst_input)(surface_temperature)

    assert prepared.shape == (1, 1, 1, 2, 2)
    assert_allclose_compact(prepared[0, 0, 0], np.asarray(surface_temperature))


def test_torch_tensor_from_jax_array_uses_copied_host_boundary() -> None:
    source = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    tensor = _torch_tensor_from_jax_array(source, "cpu")
    tensor[0, 0] = 99.0

    assert isinstance(tensor, torch.Tensor)
    assert tensor.device.type == "cpu"
    assert_allclose_compact(np.asarray(source), np.asarray([[1.0, 2.0], [3.0, 4.0]]))


def test_camulator_output_helper_delegates_to_credit_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    def fake_make_xarray(
        prediction: torch.Tensor,
        utc_datetime: datetime,
        latitude: object,
        longitude: object,
        conf: dict[str, Any],
    ) -> tuple[xr.Dataset, xr.Dataset]:
        calls["make_xarray"] = (prediction, utc_datetime, latitude, longitude, conf)
        return xr.Dataset(), xr.Dataset()

    def fake_save_netcdf_increment(
        upper_air: xr.Dataset,
        single_level: xr.Dataset,
        init_str: str,
        forecast_hour: int,
        metadata: dict[str, Any],
        conf: dict[str, Any],
    ) -> None:
        calls["save"] = (
            upper_air,
            single_level,
            init_str,
            forecast_hour,
            metadata,
            conf,
        )

    monkeypatch.setattr(
        camulator_output_module,
        "_credit_output_functions",
        lambda: (fake_make_xarray, fake_save_netcdf_increment),
    )

    prediction = torch.ones((1, 1))
    utc_datetime = datetime(2000, 1, 1)
    metadata: dict[str, Any] = {"source": "test"}
    conf: dict[str, Any] = {"data": {}}

    camulator_output_module.write_camulator_prediction_output(
        prediction,
        utc_datetime,
        latitude=[0.0],
        longitude=[1.0],
        init_str="2000010100",
        lead_time_periods=6,
        forecast_hour=2,
        metadata=metadata,
        conf=conf,
    )

    assert calls["make_xarray"][0].shape == prediction.shape
    assert calls["save"][2:] == ("2000010100", 12, metadata, conf)


def test_prepare_static_forcing_tensor_preserves_order_and_shape() -> None:
    forcing_ds = xr.Dataset(
        data_vars={
            "TOPO": (("lat", "lon"), np.asarray([[1.0, 2.0], [3.0, 4.0]])),
            "LAND": (("lat", "lon"), np.asarray([[5.0, 6.0], [7.0, 8.0]])),
        }
    )

    static_forcing = camulator_tensors_module._prepare_static_forcing_tensor(
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

    mapped_fields = jax.jit(_map_camulator_prediction_arrays)(
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

    component = make_camulator_gcm(config_path="dummy.yaml", device="cpu")

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
        "_prepare_static_forcing_tensor",
        lambda forcing_ds, static_variables, device: torch.zeros((1, 1)),
    )
    monkeypatch.setattr(
        camulator_imports_module, "load_metadata", lambda conf: {}, raising=False
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
    assert set(component.field_spec.default_fields) == {"land_surface_temperature"}
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
        RuntimeStepContext(
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

    class _StateManager:
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

    class _Model:
        def __call__(self, model_input: torch.Tensor) -> torch.Tensor:
            return model_input + 1.0

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
        camulator_module._CAMulatorGCMState.__new__(
            camulator_module._CAMulatorGCMState
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
    component.stepper = SimpleNamespace(
        state_manager=_StateManager(),
        model=_Model(),
        _apply_postprocessing=lambda prediction, model_input: prediction,
    )
    component.static_forcing = torch.zeros((1, 1, 1, 2, 2))
    component.state = torch.zeros((1, 1, 1, 2, 2))
    component.LANDM_COSLAT = jnp.asarray([[0.0, 1.0], [0.5, 0.0]])
    component.name = "ATM"
    component.grid = RectilinearGrid(
        name="atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([0.0, 1.0]),
    )
    component.settings = VercorSettings()
    component.data = _initialize_camulator_runtime_fields(
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
    component.conf = {}
    component.lead_time_periods = 6
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

    monkeypatch.setattr(
        camulator_output_module,
        "write_camulator_prediction_output",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        camulator_fields_module,
        "_map_camulator_prediction_arrays",
        lambda *args: {"temperature": jnp.full((2, 2), 9.0)},
    )

    component_state = _runtime_component_state("ATM", component.data)
    step_context = RuntimeStepContext(
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
