from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

import vercor.setups.external.jax_gcm as jax_gcm_module
import vercor.setups.external._jax_gcm_pytree as jax_gcm_pytree_module
import vercor.setups.external.jax_gcm_fields as jax_gcm_fields_module
import vercor.setups.external.jax_gcm_output as jax_gcm_output_module
import vercor.setups.external.jax_gcm_runtime as jax_gcm_runtime_module
import vercor.setups.external.jax_gcm_state as jax_gcm_state_module
import vercor.setups.external.veros_fluxes as veros_fluxes_module
import vercor.setups.external.veros_gcm as veros_gcm_module
import vercor.setups.external.veros_gcm_state as veros_gcm_state_module
import vercor.setups.external.veros_output as veros_output_module
import vercor.setups.external.veros_runtime as veros_runtime_module
import vercor.setups.external.veros_runtime_settings as veros_runtime_settings_module
import vercor.setups.external.veros_setup as veros_setup_module
import vercor.setups.external.veros_state as veros_state_module
from tests._coverage_support import capture_logger_output, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.calendar import DateTime360, DateTime365
from vercor.components.data import DataComponent
from vercor.components.contexts import ComponentSetupContext, ComponentStepContext
from vercor.output.adapters import ComponentOutputAdapter, component_snapshot_writer
from vercor.output.variables import OutputVariable
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.settings import VercorSettings


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.messages.append(message.format(*args) if args else message)


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


class _FakeDynamicsPrediction:
    def __init__(
        self,
        *,
        temperature: np.ndarray,
        u_wind: np.ndarray | None = None,
    ) -> None:
        self.temperature = temperature
        self.u_wind = np.ones_like(temperature) if u_wind is None else u_wind

    def asdict(self) -> dict[str, np.ndarray]:
        return {
            "temperature": self.temperature,
            "u_wind": self.u_wind,
        }


@dataclass
class _PredictionValues:
    dynamics: _FakeDynamicsPrediction
    physics: Any
    times: np.ndarray

    def to_xarray(self) -> xr.Dataset:
        raise AssertionError("JAXGCM output adapter path must not call to_xarray()")


class _FakePhysicsModule:
    UNITS_TABLE_CSV_PATH: str | Path | None = None

    def __init__(self, physics_data: dict[str, np.ndarray] | None = None) -> None:
        self.physics_data = physics_data or {}
        self.cached_coords: Any | None = None

    def cache_coords(self, coords: Any) -> None:
        self.cached_coords = coords

    def data_struct_to_dict(
        self,
        struct: Any,
        nodal_shape: tuple[int, ...],
    ) -> dict[str, np.ndarray]:
        _ = struct, nodal_shape
        return self.physics_data


def _make_jax_gcm_output_coords() -> SimpleNamespace:
    layers = 3
    nodal_shape = (2, 3)
    lon = np.deg2rad(np.asarray([0.0, 180.0]))
    sin_lat = np.sin(np.deg2rad(np.asarray([-45.0, 0.0, 45.0])))
    return SimpleNamespace(
        horizontal=SimpleNamespace(
            nodal_axes=(lon, sin_lat),
            modal_axes=(np.asarray([0]), np.asarray([0])),
            nodal_shape=nodal_shape,
            modal_shape=(1, 1),
        ),
        vertical=SimpleNamespace(centers=np.asarray([0.2, 0.5, 0.8]), layers=layers),
        surface_nodal_shape=nodal_shape,
        asdict=lambda: {},
    )


class _FakeForcing:
    def __init__(self) -> None:
        self.copy_calls: list[dict[str, Any]] = []

    def copy(self, **kwargs: Any) -> "_FakeForcing":
        self.copy_calls.append(kwargs)
        return self


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        _ = exc_type, exc, tb
        return False


class _FakeSettings(dict[str, Any]):
    def __init__(self, metadata: dict[str, Any], values: dict[str, Any]) -> None:
        super().__init__(values)
        object.__setattr__(self, "__metadata__", metadata)

    def unlock(self) -> _NullContext:
        return _NullContext()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "__metadata__":
            object.__setattr__(self, name, value)
        else:
            self[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _FakeVariableStore(SimpleNamespace):
    def unlock(self) -> _NullContext:
        return _NullContext()


class _ConstructedVerosState:
    def __init__(
        self,
        var_meta: dict[str, Any],
        settings_meta: dict[str, Any],
        dimensions: dict[str, Any],
        plugin_interfaces: dict[str, Any] | None = None,
    ) -> None:
        self._var_meta = var_meta
        self._dimensions = dimensions
        self._plugin_interfaces = plugin_interfaces
        self.settings = _FakeSettings(settings_meta, {})
        self._variables: dict[str, Any] = {}
        self.timers: dict[str, Any] = {}
        self.profile_timers: dict[str, Any] = {}


def _make_jax_gcm_output_adapter() -> ComponentOutputAdapter:
    return jax_gcm_output_module.make_jax_gcm_output_adapter()


def _write_jax_gcm_average_output(
    adapter: ComponentOutputAdapter,
    output: str,
    *,
    coords: Any,
    output_time: Any,
    physics_module: Any,
    logger: Any | None = None,
) -> None:
    unit_metadata = jax_gcm_output_module.jax_gcm_unit_metadata(physics_module)

    def build_coordinate_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        _ = variables
        return jax_gcm_output_module.jax_gcm_coordinate_variables(
            coords=coords,
            output_time=output_time,
        )

    def build_data_variables(
        variables: Mapping[str, OutputVariable],
    ) -> dict[str, OutputVariable]:
        return jax_gcm_output_module.jax_gcm_data_variables_with_unit_metadata(
            variables,
            unit_metadata,
        )

    adapter.write_period_average(
        output,
        build_coordinate_variables=build_coordinate_variables,
        build_data_variables=build_data_variables,
        logger=logger,
    )


def _make_veros_output_adapter() -> ComponentOutputAdapter:
    return veros_output_module.make_veros_output_adapter()


def _make_coupler(
    *,
    dt_seconds: float,
    run_order: list[str],
    settings: VercorSettings | None = None,
) -> ComponentSetupContext:
    return ComponentSetupContext(
        start=datetime(2000, 1, 1),
        dt_seconds=dt_seconds,
        logger=cast(Any, _RecordingLogger()),
        settings=settings or VercorSettings(),
        run_sequence=tuple(run_order),
    )


def _make_fake_veros_state(surface_temperature: float = 10.0) -> Any:
    temp = np.full((8, 8, 1, 1), surface_temperature, dtype=float)
    variables = SimpleNamespace(temp=temp, tau=0)
    return SimpleNamespace(variables=variables)


def _make_veros_output_state(offset: float = 0.0) -> Any:
    variables = SimpleNamespace(
        tau=1,
        xt=np.asarray([-2.0, -1.0, 0.0, 4.0, 8.0, 99.0], dtype=float),
        xu=np.asarray([-3.0, -1.0, 1.0, 5.0, 9.0, 99.0], dtype=float),
        yt=np.asarray([-4.0, -2.0, -45.0, 0.0, 45.0, 88.0, 99.0], dtype=float),
        yu=np.asarray([-5.0, -2.0, -40.0, 5.0, 50.0, 88.0, 99.0], dtype=float),
        zt=np.asarray([-100.0, -20.0], dtype=float),
        zw=np.asarray([-150.0, -5.0], dtype=float),
        temp=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + offset,
        salt=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + 50.0 + offset,
        u=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + 100.0 + offset,
        surface_taux=np.arange(6 * 7, dtype=float).reshape(6, 7) + 200.0 + offset,
        psi=np.arange(6 * 7 * 3, dtype=float).reshape(6, 7, 3) + 300.0 + offset,
    )
    return SimpleNamespace(
        settings=SimpleNamespace(enable_streamfunction=True, coord_degree=True),
        variables=variables,
    )


def _make_flux_ready_veros_state() -> Any:
    tau = 0
    return SimpleNamespace(
        variables=SimpleNamespace(
            tau=tau,
            u=np.arange(36.0).reshape(6, 6, 1, 1),
            v=np.arange(36.0, 72.0).reshape(6, 6, 1, 1),
            temp=np.full((6, 6, 1, 1), 7.0, dtype=float),
            maskT=np.ones((6, 6, 1), dtype=float),
        )
    )


def test_jax_gcm_tree_as_real_dtype_converts_tree_leaves() -> None:
    tree = {
        "a": jnp.asarray([1, 2], dtype=jnp.int32),
        "b": jnp.asarray([[3, 4]], dtype=jnp.int32),
    }

    converted = jax_gcm_pytree_module.tree_as_real_dtype(tree)

    assert jnp.issubdtype(converted["a"].dtype, jnp.floating)
    assert jnp.issubdtype(converted["b"].dtype, jnp.floating)
    assert_allclose_compact(converted["a"], np.asarray([1.0, 2.0]))


def test_cleanup_surface_temperature_fields_supports_jit_and_gradients() -> None:
    land_surface_temperature = jnp.asarray([[270.0, jnp.nan], [260.0, 240.0]])
    sea_surface_temperature = jnp.asarray([[jnp.nan, 281.0], [2.0, 3.0]])

    (
        clean_land_surface_temperature,
        clean_sea_surface_temperature,
        total_surface_temperature,
        cold_surface_cells,
    ) = jax.jit(jax_gcm_fields_module.cleanup_surface_temperature_fields)(
        land_surface_temperature,
        sea_surface_temperature,
    )

    assert_allclose_compact(
        clean_land_surface_temperature,
        np.asarray([[270.0, 0.0], [260.0, 240.0]]),
    )
    assert_allclose_compact(
        clean_sea_surface_temperature,
        np.asarray([[0.0, 281.0], [2.0, 3.0]]),
    )
    assert_allclose_compact(
        total_surface_temperature,
        np.asarray([[270.0, 281.0], [262.0, 243.0]]),
    )
    assert_allclose_compact(
        cold_surface_cells,
        np.asarray([[False, False], [False, True]]),
    )

    gradient = jax.grad(
        lambda land: jnp.sum(
            jax_gcm_fields_module.cleanup_surface_temperature_fields(
                land,
                jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
            )[2]
        )
    )(jnp.asarray([[270.0, 271.0], [272.0, 273.0]]))
    assert_allclose_compact(gradient, np.ones((2, 2)))


def test_prepare_surface_temperature_forcing_supports_jit_and_fill_value() -> None:
    total_surface_temperature = jnp.asarray([[270.0, 281.0], [282.0, 567.0]])
    land_fraction_mask = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])

    land_surface_temperature, sea_surface_temperature = jax.jit(
        jax_gcm_fields_module.prepare_surface_temperature_forcing
    )(total_surface_temperature, land_fraction_mask)

    assert_allclose_compact(
        land_surface_temperature,
        np.asarray([[270.0, 288.15], [288.15, 567.0]]),
    )
    assert_allclose_compact(
        sea_surface_temperature,
        np.asarray([[288.15, 281.0], [282.0, 288.15]]),
    )


def test_map_jcm_output_fields_supports_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        jax_gcm_fields_module,
        "compute_sigma_pressure_levels",
        lambda reference_pressure, top_pressure, sigma_levels, normalized_surface_pressure: jnp.asarray(
            [
                jnp.full((2, 2), 90000.0),
                jnp.full((2, 2), 80000.0),
            ]
        ),
    )
    monkeypatch.setattr(
        jax_gcm_fields_module,
        "get_altitudes_sigma_levels",
        lambda temperature, pressure, specific_humidity: jnp.asarray(
            [
                jnp.full((2, 2), 50.0),
                jnp.full((2, 2), 150.0),
            ]
        ),
    )

    mapped_fields = jax.jit(jax_gcm_fields_module.map_jcm_output_fields)(
        2.5e6,
        1.0e5,
        jnp.asarray([0.2, 1.0]),
        28.966,
        8314.47,
        1.0e5,
        0.286,
        jnp.full((2, 2, 2), 5.0),
        jnp.full((2, 2, 2), 2.0),
        jnp.full((2, 2), 40.0),
        jnp.full((2, 2), 30.0),
        jnp.asarray([[0.9, 1.0], [1.1, 1.2]]),
        jnp.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        jnp.asarray(
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ]
        ),
        jnp.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
        jnp.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
    )

    assert_allclose_compact(
        mapped_fields["u_velocity"], np.asarray([[5.0, 7.0], [6.0, 8.0]])
    )
    assert_allclose_compact(
        mapped_fields["v_velocity"], np.asarray([[6.0, 8.0], [7.0, 9.0]])
    )
    assert_allclose_compact(
        mapped_fields["temperature"], np.asarray([[284.0, 286.0], [285.0, 287.0]])
    )
    assert_allclose_compact(
        mapped_fields["specific_humidity"],
        np.asarray([[0.05, 0.07], [0.06, 0.08]]),
    )
    assert_allclose_compact(mapped_fields["sensible_heat_flux"], np.full((2, 2), -10.0))
    assert_allclose_compact(
        mapped_fields["latent_heat_flux"],
        np.full((2, 2), -10000.0),
    )
    assert_allclose_compact(
        mapped_fields["pressure"],
        np.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    assert_allclose_compact(
        mapped_fields["density"],
        28.966
        / 8314.47
        * np.full((2, 2), 80000.0)
        / np.asarray([[284.0, 286.0], [285.0, 287.0]]),
    )
    assert_allclose_compact(
        mapped_fields["potential_temperature"],
        np.asarray([[284.0, 286.0], [285.0, 287.0]]) * (100000.0 / 80000.0) ** 0.286,
    )
    assert_allclose_compact(mapped_fields["model_level_height"], np.full((2, 2), 150.0))


def test_generate_step_function_non_jitted_averages_predictions() -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.save_interval = timedelta(days=2)
    component.coupling_timestep = timedelta(hours=12)

    calls: dict[str, Any] = {}

    class _FakeModel:
        def run_from_state(
            self,
            *,
            initial_state: Any,
            save_interval: float,
            total_time: float,
            forcing: Any,
        ) -> tuple[str, Any]:
            calls["run_from_state"] = (
                initial_state,
                save_interval,
                total_time,
                forcing,
            )
            predictions = SimpleNamespace(
                dynamics={"wind": jnp.asarray([[1.0, 3.0], [5.0, 7.0]])},
                physics={"temp": jnp.asarray([[2.0, 4.0], [6.0, 8.0]])},
            )
            return "next-modal-state", predictions

    component.model = _FakeModel()
    state = jax_gcm_state_module.JCMState(prog={}, phydata={}, metadata="initial-state")

    step_function = component._generate_step_function(jitted=False)
    next_state, predictions = step_function(state, "forcing")

    assert calls["run_from_state"] == ("initial-state", 2.0, 0.5, "forcing")
    assert next_state.metadata == "next-modal-state"
    assert_allclose_compact(next_state.prog["wind"], np.asarray([3.0, 5.0]))
    assert_allclose_compact(next_state.phydata["temp"], np.asarray([4.0, 6.0]))
    assert predictions.physics["temp"].shape == (2, 2)


def test_jax_gcm_constructor_builds_jax_backed_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hgrid = SimpleNamespace(
        longitudes=jnp.deg2rad(jnp.asarray([0.0, 180.0])),
        latitudes=jnp.deg2rad(jnp.asarray([-45.0, 0.0, 45.0])),
    )
    coords = SimpleNamespace(
        horizontal=hgrid,
        vertical=SimpleNamespace(centers=jnp.asarray([0.2, 1.0])),
    )
    terrain = SimpleNamespace(fmask=np.zeros((2, 3), dtype=float))

    class _FakeModel:
        def __init__(
            self,
            coords: Any,
            time_step: float,
            terrain: Any,
            physics: Any,
        ) -> None:
            _ = time_step, physics
            self.coords = coords
            self.terrain = terrain

    monkeypatch.setattr(
        jax_gcm_state_module.Parameters,
        "default",
        staticmethod(lambda: SimpleNamespace()),
    )
    monkeypatch.setattr(
        jax_gcm_state_module,
        "SpeedyPhysics",
        lambda parameters: SimpleNamespace(parameters=parameters),
    )
    monkeypatch.setattr(jax_gcm_state_module, "Model", _FakeModel)

    component = jax_gcm_module.make_jax_gcm(coords=coords, terrain=terrain)

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.field_spec.inputs == (
        "land_surface_temperature",
        "sea_surface_temperature",
    )
    assert component.field_spec.outputs == (
        "land_surface_temperature",
        "sea_surface_temperature",
        "total_surface_temperature",
        *jax_gcm_fields_module.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
        "pressure",
    )
    assert_allclose_compact(component.grid.longitude, np.asarray([0.0, 180.0]))
    assert_allclose_compact(component.grid.latitude, np.asarray([-45.0, 0.0, 45.0]))
    assert_allclose_compact(component.grid.binary_mask, np.ones((3, 2)))
    assert callable(component_snapshot_writer(component))


def test_jax_gcm_initialize_validates_timestep_multiple() -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.spinup_time = timedelta(hours=6)
    component.model_timestep = timedelta(minutes=45)
    component.model = SimpleNamespace()
    component.grid = make_test_grid()

    with pytest.raises(ValueError, match="model_timestep"):
        component.initialize(
            cast(Any, component),
            _make_coupler(dt_seconds=3600.0, run_order=["ATM"]),
        )


def test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.spinup_time = timedelta(hours=2)
    component.model_timestep = timedelta(hours=1)
    component.jitted = False
    component.do_spinup = True
    component.name = "ATM"
    component.grid = make_test_grid()
    component.data = {}
    component.settings = VercorSettings()
    component.save_interval = timedelta(days=1)
    component.output_frequency = None
    component.forcing_data = "provided-forcing"
    component.output_adapter = _make_jax_gcm_output_adapter()
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
            vertical=SimpleNamespace(layers=2),
        ),
        primitive="primitive",
        _prepare_initial_modal_state=lambda: "modal-state",
    )

    physics_calls: dict[str, Any] = {}

    class _FakePhysicsData:
        @staticmethod
        def zeros(shape: tuple[int, int], layers: int) -> dict[str, Any]:
            physics_calls["zeros"] = (shape, layers)
            return {"shape": shape, "layers": layers}

    monkeypatch.setattr(jax_gcm_state_module, "PhysicsData", _FakePhysicsData)
    monkeypatch.setattr(
        jax_gcm_state_module,
        "dynamics_state_to_physics_state",
        lambda modal_state, primitive: {
            "modal_state": modal_state,
            "primitive": primitive,
        },
    )
    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: (lambda state, forcing: (state, "unused")),
    )
    accumulated_predictions: list[Any] = []

    def fake_jax_gcm_prediction_output_variables(
        prediction: Any,
        *,
        coords: Any,
        physics_module: Any | None = None,
    ) -> dict[str, OutputVariable]:
        _ = coords, physics_module
        accumulated_predictions.append(prediction)
        return {"spinup": OutputVariable(("time", "x"), np.asarray([[1.0]]))}

    monkeypatch.setattr(
        jax_gcm_state_module._jax_gcm_output,
        "jax_gcm_prediction_output_variables",
        fake_jax_gcm_prediction_output_variables,
    )

    hook_component = DataComponent.from_fields(
        name="ATM",
        grid=component.grid,
        settings=component.settings,
    )
    coupler = _make_coupler(dt_seconds=3600.0, run_order=["OCN"])
    component.initialize(cast(Any, hook_component), coupler)

    assert component.coupling_timestep == timedelta(hours=1)
    assert component.spinup_steps == 2
    assert physics_calls["zeros"] == ((2, 3), 2)
    assert component.forcing == "provided-forcing"
    assert accumulated_predictions == ["unused", "unused"]
    assert_allclose_compact(
        component.output_adapter.variables["spinup"].counts,
        np.asarray([2]),
    )
    assert isinstance(hook_component.data["sea_surface_temperature"], jax.Array)
    assert hook_component.data["sea_surface_temperature"].shape == component.grid.shape


def test_jax_gcm_initialize_builds_default_forcing_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.spinup_time = timedelta(hours=1)
    component.model_timestep = timedelta(hours=1)
    component.jitted = False
    component.do_spinup = False
    component.name = "ATM"
    component.grid = make_test_grid()
    component.data = {}
    component.settings = VercorSettings()
    component.save_interval = timedelta(days=1)
    component.output_frequency = None
    component.forcing_data = None
    component.output_adapter = _make_jax_gcm_output_adapter()
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
            vertical=SimpleNamespace(layers=2),
        ),
        primitive="primitive",
        _prepare_initial_modal_state=lambda: "modal-state",
    )

    class _FakePhysicsData:
        @staticmethod
        def zeros(shape: tuple[int, int], layers: int) -> dict[str, Any]:
            return {"shape": shape, "layers": layers}

    forcing = _FakeForcing()

    monkeypatch.setattr(jax_gcm_state_module, "PhysicsData", _FakePhysicsData)
    monkeypatch.setattr(
        jax_gcm_state_module,
        "dynamics_state_to_physics_state",
        lambda modal_state, primitive: {
            "modal_state": modal_state,
            "primitive": primitive,
        },
    )
    monkeypatch.setattr(
        jax_gcm_state_module,
        "default_forcing",
        lambda horizontal: forcing,
    )
    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: (lambda state, forcing_data: (state, "unused")),
    )

    hook_component = DataComponent.from_fields(
        name="ATM",
        grid=component.grid,
        settings=component.settings,
    )
    component.initialize(
        cast(Any, hook_component),
        _make_coupler(dt_seconds=3600.0, run_order=["ATM"]),
    )

    assert component.forcing is forcing
    assert forcing.copy_calls == [{"lfluxland": True}]


def test_jax_gcm_step_maps_outputs_and_respects_output_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.name = "ATM"
    component.grid = make_test_grid(name="atm")
    component.settings = VercorSettings()
    component.data = {
        "sea_surface_temperature": np.asarray([[np.nan, 281.0], [282.0, 283.0]]),
        "land_surface_temperature": np.asarray([[270.0, np.nan], [0.0, 284.0]]),
    }
    component.model = SimpleNamespace(
        terrain=SimpleNamespace(
            fmask=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        ),
        coords=_make_jax_gcm_output_coords(),
        physics=SimpleNamespace(),
    )
    component.forcing = _FakeForcing()
    cast(Any, component).sigma_levels = np.asarray([0.2, 1.0], dtype=float)
    component._state = cast(Any, SimpleNamespace(metadata=jnp.asarray(0.0)))
    component.output_adapter = _make_jax_gcm_output_adapter()
    component.output_frequency = "day"

    written: dict[str, Any] = {}

    p = SimpleNamespace(
        surface_flux=SimpleNamespace(
            shf=np.full((2, 2, 2), 5.0, dtype=float),
            evap=np.full((2, 2, 2), 2.0, dtype=float),
            rlds=np.full((2, 2), 40.0, dtype=float),
        ),
        shortwave_rad=SimpleNamespace(rsns=np.full((2, 2), 30.0, dtype=float)),
    )
    d = SimpleNamespace(
        u_wind=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ]
        ),
        v_wind=np.asarray(
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0]],
            ]
        ),
        temperature=np.asarray(
            [
                [[280.0, 281.0], [282.0, 283.0]],
                [[284.0, 285.0], [286.0, 287.0]],
            ]
        ),
        specific_humidity=np.asarray(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ]
        ),
        normalized_surface_pressure=np.asarray([[0.9, 1.0], [1.1, 1.2]], dtype=float),
    )

    prediction = SimpleNamespace(physics=p, dynamics=d)
    component._step_function = cast(
        Any,
        lambda state, forcing: (
            SimpleNamespace(metadata=jnp.asarray(1.0)),
            prediction,
        ),
    )
    monkeypatch.setattr(jax_gcm_runtime_module, "tree_stack", lambda objs: objs[0])
    monkeypatch.setattr(
        jax_gcm_runtime_module,
        "tree_unwrap_leading_dims",
        lambda obj: obj,
    )
    monkeypatch.setattr(jax_gcm_runtime_module, "tree_mean", lambda obj, axis: obj)
    monkeypatch.setattr(
        jax_gcm_fields_module,
        "compute_sigma_pressure_levels",
        lambda reference_pressure, top_pressure, sigma_levels, normalized_surface_pressure: jnp.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    monkeypatch.setattr(
        jax_gcm_fields_module,
        "get_altitudes_sigma_levels",
        lambda temperature, pressure, specific_humidity: jnp.asarray(
            [
                np.full((2, 2), 50.0),
                np.full((2, 2), 150.0),
            ]
        ),
    )
    cast(Any, jax_gcm_fields_module.map_jcm_output_fields).clear_cache()

    def fake_record_jax_gcm_period_output(
        adapter: ComponentOutputAdapter,
        prediction_arg: Any,
        *,
        coords: Any,
        physics_module: Any | None,
        output_time: datetime,
        dt: timedelta,
        output_frequency: str | None,
        logger: Any | None = None,
    ) -> bool:
        _ = logger
        written["adapter"] = adapter
        written["accumulated_prediction"] = prediction_arg
        written["accumulated_coords"] = coords
        written["accumulated_physics_module"] = physics_module
        written["output_time"] = output_time
        written["dt"] = dt
        written["output_frequency"] = output_frequency
        written["path"] = f"jcm.averages.{output_time.strftime('%Y-%m-%d')}.nc"
        adapter.accumulate(
            {"temperature": OutputVariable(("time", "x"), np.asarray([[1.0]]))},
            summation_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
        )
        written["counts"] = adapter.variables["temperature"].counts.copy()
        adapter.reset()
        return True

    monkeypatch.setattr(
        jax_gcm_output_module,
        "record_jax_gcm_period_output",
        fake_record_jax_gcm_period_output,
    )

    coupler = _make_coupler(
        dt_seconds=3600.0, run_order=["ATM"], settings=VercorSettings()
    )
    runtime_data = dict(component.data)
    runtime_incoming: dict[str, Any] = {}
    runtime_outgoing: dict[str, Any] = {}
    runtime_contract = RuntimeComponentContract()
    hook_component = DataComponent.from_fields(
        name="ATM",
        grid=component.grid,
        settings=component.settings,
    )
    jax_gcm_runtime_module.prefill_jax_gcm_runtime_fields(
        component,
        cast(Any, hook_component),
        runtime_data,
        runtime_incoming,
        runtime_outgoing,
        runtime_contract,
    )
    component_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(runtime_data),
        incoming=RuntimeFieldStore.from_mapping(runtime_incoming),
        outgoing=RuntimeFieldStore.from_mapping(runtime_outgoing),
        runtime_payload=jax_gcm_runtime_module.create_jax_gcm_runtime_payload(
            component
        ),
    )
    step_context = ComponentStepContext(
        dt_seconds=timedelta(days=1).total_seconds(),
        settings=coupler.settings,
        time=datetime(2000, 1, 2),
        logger=coupler.logger,
    )
    step_result = jax_gcm_runtime_module.step_jax_gcm_component(
        component,
        component_state.data.to_mapping(),
        step_context,
        component_state.runtime_payload,
    )
    component_state = component_state.with_data(
        component_state.data.set_many(step_result.fields)
    ).with_runtime_payload(step_result.payload)
    forcing_call = component.forcing.copy_calls[-1]
    assert isinstance(forcing_call["stl_am"], jax.Array)
    assert isinstance(forcing_call["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(
        forcing_call["stl_am"],
        np.asarray([[270.0, 288.15], [288.15, 567.0]]),
    )
    assert_allclose_compact(
        forcing_call["sea_surface_temperature"],
        np.asarray([[288.15, 282.0], [281.0, 288.15]]),
    )
    data = component_state.data
    assert_allclose_compact(
        data.get("total_surface_temperature"),
        np.asarray([[270.0, 281.0], [282.0, 567.0]]),
    )
    assert_allclose_compact(
        data.get("u_velocity"), np.asarray([[5.0, 7.0], [6.0, 8.0]])
    )
    assert_allclose_compact(
        data.get("v_velocity"), np.asarray([[6.0, 8.0], [7.0, 9.0]])
    )
    assert_allclose_compact(
        data.get("temperature"), np.asarray([[284.0, 286.0], [285.0, 287.0]])
    )
    assert_allclose_compact(
        data.get("specific_humidity"),
        np.asarray([[0.05, 0.07], [0.06, 0.08]]),
    )
    assert_allclose_compact(data.get("sensible_heat_flux"), np.full((2, 2), -10.0))
    assert_allclose_compact(
        data.get("latent_heat_flux"),
        np.full((2, 2), -2.0 / 1e3 * coupler.settings.latvap * 2.0),
    )
    assert_allclose_compact(
        data.get("pressure"),
        np.asarray(
            [
                np.full((2, 2), 90000.0),
                np.full((2, 2), 80000.0),
            ]
        ),
    )
    assert_allclose_compact(
        data.get("density"),
        coupler.settings.mwdair
        / coupler.settings.rgas
        * np.full((2, 2), 80000.0)
        / np.asarray([[284.0, 286.0], [285.0, 287.0]]),
    )
    assert_allclose_compact(
        data.get("potential_temperature"),
        np.asarray([[284.0, 286.0], [285.0, 287.0]])
        * (coupler.settings.p0 / 80000.0) ** coupler.settings.cappa,
    )
    assert_allclose_compact(data.get("model_level_height"), np.full((2, 2), 150.0))
    assert written["accumulated_prediction"] is prediction
    assert written["accumulated_coords"] is component.model.coords
    assert written["accumulated_physics_module"] is component.model.physics
    assert written["adapter"] is component.output_adapter
    assert_allclose_compact(written["counts"], np.asarray([1]))
    assert written["path"] == "jcm.averages.2000-01-02.nc"
    assert written["output_time"] == datetime(2000, 1, 2)
    assert written["dt"] == timedelta(days=1)
    assert written["output_frequency"] == "day"
    assert component.output_adapter.empty


def test_jax_gcm_write_output_persists_mean_dataset(tmp_path: Path) -> None:
    coords = _make_jax_gcm_output_coords()
    first_temperature = np.arange(18.0).reshape(3, 2, 3)
    second_temperature = first_temperature + 18.0
    physics_data = {
        "wvi_output": np.arange(12.0).reshape(1, 2, 2, 3),
        "hsg_output": np.arange(24.0).reshape(1, 4, 2, 3),
    }
    physics_module = _FakePhysicsModule(physics_data=physics_data)
    predictions = [
        _PredictionValues(
            dynamics=_FakeDynamicsPrediction(
                temperature=first_temperature[np.newaxis, ...],
            ),
            physics=SimpleNamespace(),
            times=np.asarray([0.0]),
        ),
        _PredictionValues(
            dynamics=_FakeDynamicsPrediction(
                temperature=second_temperature[np.newaxis, ...],
            ),
            physics=SimpleNamespace(),
            times=np.asarray([1.0]),
        ),
    ]
    adapter = _make_jax_gcm_output_adapter()
    for prediction in predictions:
        adapter.accumulate(
            jax_gcm_output_module.jax_gcm_prediction_output_variables(
                prediction,
                coords=coords,
                physics_module=physics_module,
            ),
            summation_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
        )
    accumulated_temperature = adapter.variables["temperature"]
    assert isinstance(accumulated_temperature.sum_values, jax.Array)
    assert isinstance(accumulated_temperature.counts, jax.Array)

    output = tmp_path / "jcm_output.nc"
    logger_name = "VerCOR.test.jax-gcm-output"
    logger = logging.getLogger(logger_name)

    with capture_logger_output(logger_name) as stream:
        _write_jax_gcm_average_output(
            adapter,
            str(output),
            coords=coords,
            output_time=datetime(2000, 1, 2),
            physics_module=physics_module,
            logger=logger,
        )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "level", "lat", "lon")
        assert temperature.shape == (1, 3, 3, 2)
        assert np.isclose(np.asarray(temperature)[0, 0, 0, 0], 9.0)
        assert temperature.attrs["units"] == "K"
        assert temperature.attrs["description"] == "temperature"
        assert actual.variables["wvi_output"].dimensions == (
            "time",
            "wvi_id",
            "lat",
            "lon",
        )
        assert actual.variables["hsg_output"].dimensions == (
            "time",
            "hsg_level",
            "lat",
            "lon",
        )
        assert actual.variables["time"].shape == (1,)
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
    assert adapter.empty
    assert physics_module.cached_coords is coords
    assert f"Writing output file:  {output}" in stream.getvalue()


def test_jax_gcm_record_period_output_accumulates_and_writes_mean_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coords = _make_jax_gcm_output_coords()
    first_temperature = np.arange(18.0).reshape(3, 2, 3)
    second_temperature = first_temperature + 18.0
    physics_module = _FakePhysicsModule()
    predictions = [
        _PredictionValues(
            dynamics=_FakeDynamicsPrediction(
                temperature=first_temperature[np.newaxis, ...],
            ),
            physics=SimpleNamespace(),
            times=np.asarray([0.0]),
        ),
        _PredictionValues(
            dynamics=_FakeDynamicsPrediction(
                temperature=second_temperature[np.newaxis, ...],
            ),
            physics=SimpleNamespace(),
            times=np.asarray([1.0]),
        ),
    ]
    adapter = _make_jax_gcm_output_adapter()
    monkeypatch.chdir(tmp_path)

    first_written = jax_gcm_output_module.record_jax_gcm_period_output(
        adapter,
        predictions[0],
        coords=coords,
        physics_module=physics_module,
        output_time=datetime(2000, 1, 1),
        dt=timedelta(hours=1),
        output_frequency="day",
    )
    second_written = jax_gcm_output_module.record_jax_gcm_period_output(
        adapter,
        predictions[1],
        coords=coords,
        physics_module=physics_module,
        output_time=datetime(2000, 1, 2),
        dt=timedelta(days=1),
        output_frequency=None,
    )

    assert not first_written
    assert second_written
    assert adapter.empty
    with h5netcdf.File(tmp_path / "jcm.averages.2000-01-02.nc", "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "level", "lat", "lon")
        assert np.isclose(np.asarray(temperature)[0, 0, 0, 0], 9.0)
        assert temperature.attrs["units"] == "K"
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"


@pytest.mark.parametrize(
    ("output_time", "expected_calendar", "expected_day_of_year", "days_per_year"),
    [
        (DateTime360(2001, 2, 30, 0, 0, 0, 0, 60), "360_day", 60, 360),
        (DateTime365(2001, 3, 1, 0, 0, 0, 0, 60), "noleap", 60, 365),
    ],
)
def test_jax_gcm_write_output_preserves_model_calendar_attrs(
    tmp_path: Path,
    output_time: DateTime360 | DateTime365,
    expected_calendar: str,
    expected_day_of_year: int,
    days_per_year: int,
) -> None:
    coords = _make_jax_gcm_output_coords()
    physics_module = _FakePhysicsModule()
    predictions = [
        _PredictionValues(
            dynamics=_FakeDynamicsPrediction(
                temperature=np.ones((1, 3, 2, 3), dtype=float),
            ),
            physics=SimpleNamespace(),
            times=np.asarray([0.0]),
        )
    ]
    adapter = _make_jax_gcm_output_adapter()
    adapter.accumulate(
        jax_gcm_output_module.jax_gcm_prediction_output_variables(
            predictions[0],
            coords=coords,
            physics_module=physics_module,
        ),
        summation_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
    )
    output = tmp_path / "jcm_model_calendar_output.nc"

    _write_jax_gcm_average_output(
        adapter,
        str(output),
        coords=coords,
        output_time=output_time,
        physics_module=physics_module,
    )

    with h5netcdf.File(output, "r") as actual:
        time = actual.variables["time"]
        assert time.attrs["calendar"] == expected_calendar
        assert time.attrs["day_of_year"] == expected_day_of_year
        assert time.attrs["days_per_year"] == days_per_year
        assert time.attrs["fixed_30_day_months"] == int(expected_calendar == "360_day")


def test_jax_gcm_snapshot_output_uses_final_runtime_payload_not_runtime_data(
    tmp_path: Path,
) -> None:
    coords = _make_jax_gcm_output_coords()
    physics_module = _FakePhysicsModule()
    setup_state = SimpleNamespace(
        output_adapter=_make_jax_gcm_output_adapter(),
        model=SimpleNamespace(coords=coords, physics=physics_module),
    )
    jcm_state = SimpleNamespace(
        prog={
            "temperature": np.arange(18.0).reshape(3, 2, 3),
            "u_wind": np.full((3, 2, 3), 4.0),
        },
        phydata={},
    )
    component_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {"temperature": np.full((3, 2, 3), -999.0)}
        ),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
        runtime_payload=SimpleNamespace(jcm_state=jcm_state),
    )
    output = tmp_path / "ATM.snapshot.nc"

    jax_gcm_output_module.write_jax_gcm_snapshot_output(
        setup_state,
        component_state,
        output,
        datetime(2000, 1, 2),
        logger=None,
    )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "level", "lat", "lon")
        assert temperature.attrs["units"] == "K"
        assert temperature.attrs["description"] == "temperature"
        assert np.asarray(temperature)[0, 0, 0, 0] != -999.0
        assert_allclose_compact(
            np.asarray(temperature)[0],
            np.transpose(jcm_state.prog["temperature"], axes=(0, 2, 1)),
        )


def test_veros_compute_fluxes_zeroes_qnec_for_large_negative_dqfldt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component._veros_state = _make_flux_ready_veros_state()
    component.data = {
        "model_level_height": np.full((2, 2), 100.0),
        "u_velocity": np.full((2, 2), 2.0),
        "v_velocity": np.full((2, 2), 3.0),
        "potential_temperature": np.full((2, 2), 280.0),
        "specific_humidity": np.full((2, 2), 0.01),
        "density": np.full((2, 2), 1.2),
        "temperature": np.full((2, 2), 281.0),
        "net_shortwave_radiation_flux": np.full((2, 2), 10.0),
        "downward_longwave_radiation_flux": np.full((2, 2), 20.0),
    }

    captured: dict[str, np.ndarray] = {}

    def fake_compute_ocean_surface_fluxes(
        settings: VercorSettings,
        mask: np.ndarray,
        model_level_height: np.ndarray,
        u_velocity: np.ndarray,
        v_velocity: np.ndarray,
        potential_temperature: np.ndarray,
        specific_humidity: np.ndarray,
        density: np.ndarray,
        temperature: np.ndarray,
        u_tgrid: np.ndarray,
        v_tgrid: np.ndarray,
        surface_temperature: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        _ = (
            settings,
            model_level_height,
            u_velocity,
            v_velocity,
            potential_temperature,
            specific_humidity,
            density,
            temperature,
        )
        captured["mask"] = mask
        captured["u_tgrid"] = u_tgrid
        captured["v_tgrid"] = v_tgrid
        captured["surface_temperature"] = surface_temperature
        return (
            np.full((2, 2), -1.0),
            np.full((2, 2), -2.0),
            np.full((2, 2), -3.0),
            np.full((2, 2), 4.0),
            np.full((2, 2), 5.0),
            np.full((2, 2), 6.0),
            np.full((2, 2), 7.0),
            np.full((2, 2), 8.0),
            np.full((2, 2), 9.0),
            np.full((2, 2), 10.0),
            np.full((2, 2), 11.0),
            np.full((2, 2), 12.0),
            np.asarray([[-1e10, -1e11], [0.5, -2.0]]),
        )

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_ocean_surface_fluxes",
        fake_compute_ocean_surface_fluxes,
    )

    taux, tauy, qnet, qnec = veros_fluxes_module.compute_fluxes(
        component._veros_state,
        component.data,
        VercorSettings(),
    )

    assert isinstance(taux, jax.Array)
    assert isinstance(tauy, jax.Array)
    assert isinstance(qnet, jax.Array)
    assert isinstance(qnec, jax.Array)
    assert_allclose_compact(captured["mask"], np.ones((2, 2)))
    assert captured["u_tgrid"].shape == (2, 2)
    assert captured["v_tgrid"].shape == (2, 2)
    assert_allclose_compact(captured["surface_temperature"], np.full((2, 2), 280.15))
    assert_allclose_compact(taux, np.full((2, 2), 5.0))
    assert_allclose_compact(tauy, np.full((2, 2), 6.0))
    assert_allclose_compact(qnet, np.full((2, 2), 24.0))
    assert_allclose_compact(qnec, np.asarray([[0.0, 0.0], [-0.5, 2.0]]))


def test_custom_global_four_degree_set_diagnostics_populates_outputs() -> None:
    state = SimpleNamespace(
        settings=SimpleNamespace(dt_tracer=1200.0),
        diagnostics={
            "snapshot": SimpleNamespace(output_frequency=None),
            "overturning": SimpleNamespace(
                output_frequency=None, sampling_frequency=None
            ),
            "energy": SimpleNamespace(output_frequency=None, sampling_frequency=None),
            "averages": SimpleNamespace(
                output_variables=None,
                output_frequency=None,
                sampling_frequency=None,
            ),
        },
    )

    component = object.__new__(veros_setup_module.CustomGlobalFourDegree)
    routine = veros_setup_module.CustomGlobalFourDegree.set_diagnostics.func.__self__
    routine.function(component, state)

    assert state.diagnostics["snapshot"].output_frequency == 365 * 86400.0
    assert state.diagnostics["overturning"].sampling_frequency == 1200.0
    assert state.diagnostics["energy"].sampling_frequency == 86400
    assert state.diagnostics["averages"].output_frequency == 365 * 86400.0
    assert state.diagnostics["averages"].sampling_frequency == 86400
    assert state.diagnostics["averages"].output_variables == [
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "surface_taux",
        "surface_tauy",
        "psi",
        "qnet",
        "qnec",
    ]


def test_configure_veros_runtime_sets_diskless_mode() -> None:
    from veros import runtime_settings

    veros_runtime_settings_module.configure_veros_runtime()

    assert getattr(runtime_settings, "diskless_mode") is True


def test_veros_copy_state_jitted_path_deep_copies_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(veros_state_module, "VerosState", _ConstructedVerosState)

    state = SimpleNamespace(
        _dimensions={"xt": [1, 2]},
        settings=_FakeSettings({"meta": {"precision": "x64"}}, {"dt_tracer": 600.0}),
        _plugin_interfaces={"plugins": ["tracer"]},
        _var_meta={"temp": {"units": "K"}},
        _variables={"temp": [1.0, 2.0]},
        timers={"step": [1.0]},
        profile_timers={"profile": [2.0]},
    )

    copied = veros_state_module.copy_state(state, jitted=True)

    assert copied is not state
    assert copied._dimensions == state._dimensions
    assert copied._dimensions is not state._dimensions
    assert copied._plugin_interfaces == state._plugin_interfaces
    assert copied._plugin_interfaces is not state._plugin_interfaces
    assert copied._var_meta == state._var_meta
    assert copied._var_meta is not state._var_meta
    assert copied._variables == state._variables
    assert copied._variables is not state._variables
    assert copied.timers == state.timers
    assert copied.timers is not state.timers
    assert copied.profile_timers == state.profile_timers
    assert copied.profile_timers is not state.profile_timers
    assert copied.settings["dt_tracer"] == 600.0
    assert copied.settings.__metadata__ == state.settings.__metadata__
    assert copied.settings.__metadata__ is not state.settings.__metadata__


def test_veros_pure_runs_step_on_copied_state(monkeypatch: pytest.MonkeyPatch) -> None:
    original_state = SimpleNamespace(counter=1)
    copied_state = SimpleNamespace(counter=1)

    def fake_copy_state(state: Any, jitted: bool = True) -> Any:
        assert state is original_state
        assert jitted is False
        return copied_state

    monkeypatch.setattr(veros_state_module, "copy_state", fake_copy_state)

    def fake_step(state: Any) -> None:
        state.counter += 1

    result = veros_state_module.pure(original_state, jitted=False, step=fake_step)

    assert result is copied_state
    assert copied_state.counter == 2
    assert original_state.counter == 1


def test_veros_update_veros_interior_supports_jit_and_gradients() -> None:
    array = jnp.zeros((8, 8, 1), dtype=jnp.float64)
    interior = jnp.arange(16.0, dtype=jnp.float64).reshape(4, 4, 1)

    updated = jax.jit(veros_state_module.update_veros_interior)(array, interior)

    assert_allclose_compact(updated[2:-2, 2:-2, :], interior)
    assert np.count_nonzero(np.asarray(updated[:2, :, :])) == 0
    assert np.count_nonzero(np.asarray(updated[-2:, :, :])) == 0
    assert np.count_nonzero(np.asarray(updated[:, :2, :])) == 0
    assert np.count_nonzero(np.asarray(updated[:, -2:, :])) == 0

    gradient = jax.grad(
        lambda payload: jnp.sum(
            veros_state_module.update_veros_interior(array, payload)
        )
    )(interior)
    assert_allclose_compact(gradient, np.ones((4, 4, 1)))


def test_veros_extract_surface_temperature_supports_jit_and_gradients() -> None:
    temperature = jnp.arange(8 * 8 * 2 * 2.0, dtype=jnp.float64).reshape(8, 8, 2, 2)

    surface_temperature = jax.jit(veros_state_module.extract_surface_temperature)(
        temperature, 1
    )

    assert_allclose_compact(
        surface_temperature,
        np.asarray(temperature[2:-2, 2:-2, -1, 1].T) + 273.15,
    )

    gradient = jax.grad(
        lambda payload: jnp.sum(
            veros_state_module.extract_surface_temperature(payload, 0)
        )
    )(temperature)
    expected_gradient = np.zeros((8, 8, 2, 2), dtype=float)
    expected_gradient[2:-2, 2:-2, -1, 0] = 1.0
    assert_allclose_compact(gradient, expected_gradient)


def test_veros_prepare_surface_forcing_fields_shapes_nan_cleanup_and_qnec_gate() -> (
    None
):
    taux = jnp.asarray([[1.0, jnp.nan], [3.0, 4.0]])
    tauy = jnp.asarray([[5.0, 6.0], [7.0, 8.0]])
    qnet = jnp.asarray([[9.0, 10.0], [11.0, jnp.nan]])
    qnec = jnp.asarray([[12.0, 13.0], [14.0, 15.0]])

    prepared = jax.jit(veros_state_module.prepare_surface_forcing_fields)(
        taux, tauy, qnet, qnec, False
    )
    assert isinstance(prepared, veros_state_module.VerosForcingFields)
    assert prepared.taux.shape == (2, 2, 1)
    assert prepared.tauy.shape == (2, 2, 1)
    assert prepared.qnet.shape == (2, 2, 1)
    assert prepared.qnec.shape == (2, 2, 1)
    taux_out, tauy_out, qnet_out, qnec_out = prepared

    assert taux_out.shape == (2, 2, 1)
    assert tauy_out.shape == (2, 2, 1)
    assert qnet_out.shape == (2, 2, 1)
    assert qnec_out.shape == (2, 2, 1)
    assert_allclose_compact(taux_out, np.asarray([[[1.0], [3.0]], [[0.0], [4.0]]]))
    assert_allclose_compact(qnet_out, np.asarray([[[9.0], [11.0]], [[10.0], [0.0]]]))
    assert_allclose_compact(qnec_out, np.zeros((2, 2, 1)))

    restored = veros_state_module.prepare_surface_forcing_fields(
        taux, tauy, qnet, qnec, True
    )[3]
    assert_allclose_compact(
        restored,
        np.asarray([[[12.0], [14.0]], [[13.0], [15.0]]]),
    )


def test_veros_output_snapshot_uses_variable_metadata_and_current_timestep() -> None:
    import vercor.setups.external.veros_output as veros_output_module

    state = _make_veros_output_state()

    snapshot = veros_output_module.extract_veros_output_snapshot(
        state,
        ("temp", "u", "surface_taux", "psi"),
    )

    assert snapshot["temp"].dims == ("xt", "yt", "zt")
    assert snapshot["temp"].attrs["units"] == "deg C"
    assert snapshot["temp"].attrs["long_name"] == "Conservative temperature"
    assert isinstance(snapshot["temp"].values, jax.Array)
    assert_allclose_compact(
        snapshot["temp"].values,
        state.variables.temp[2:-2, 2:-2, :, state.variables.tau],
    )
    assert snapshot["u"].dims == ("xu", "yt", "zt")
    assert_allclose_compact(
        snapshot["u"].values,
        state.variables.u[2:-2, 2:-2, :, state.variables.tau],
    )
    assert snapshot["surface_taux"].dims == ("xu", "yt")
    assert_allclose_compact(
        snapshot["surface_taux"].values,
        state.variables.surface_taux[2:-2, 2:-2],
    )
    assert snapshot["psi"].dims == ("xu", "yu")
    assert snapshot["psi"].attrs["units"] == "m^3/s"
    assert snapshot["psi"].attrs["long_name"] == "Barotropic streamfunction"
    assert_allclose_compact(
        snapshot["psi"].values,
        state.variables.psi[2:-2, 2:-2, state.variables.tau],
    )


def test_veros_write_output_persists_period_mean_and_coordinates(
    tmp_path: Path,
) -> None:
    import vercor.setups.external.veros_output as veros_output_module

    state = _make_veros_output_state()
    snapshots = [
        veros_output_module.extract_veros_output_snapshot(
            _make_veros_output_state(offset=0.0),
            ("temp", "salt", "u", "surface_taux", "psi"),
        ),
        veros_output_module.extract_veros_output_snapshot(
            _make_veros_output_state(offset=20.0),
            ("temp", "salt", "u", "surface_taux", "psi"),
        ),
    ]
    adapter = _make_veros_output_adapter()
    for snapshot in snapshots:
        adapter.accumulate(snapshot)
    accumulated_temperature = adapter.variables["temp"]
    assert isinstance(accumulated_temperature.sum_values, jax.Array)
    assert isinstance(accumulated_temperature.counts, jax.Array)
    output = tmp_path / "veros_output.nc"

    adapter.write_period_average(
        str(output),
        build_coordinate_variables=lambda variables: (
            veros_output_module.veros_average_coordinate_variables(
                veros_state=state,
                output_time=datetime(2000, 1, 2),
                variables=variables,
            )
        ),
    )

    with h5netcdf.File(output, "r") as actual:
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
        assert_allclose_compact(
            np.asarray(actual.variables["xt"]),
            state.variables.xt[2:-2],
        )
        assert_allclose_compact(
            np.asarray(actual.variables["yt"]),
            state.variables.yt[2:-2],
        )
        assert actual.variables["zt"].attrs["positive"] == "up"
        assert actual.variables["temp"].dimensions == ("time", "zt", "yt", "xt")
        assert actual.variables["temp"].shape == (1, 2, 3, 2)
        assert actual.variables["temp"].attrs["units"] == "deg C"
        assert actual.variables["temp"].attrs["long_name"] == (
            "Conservative temperature"
        )
        expected_temp = state.variables.temp[2:-2, 2:-2, :, state.variables.tau] + 10.0
        assert_allclose_compact(
            np.asarray(actual.variables["temp"])[0],
            np.transpose(expected_temp),
        )
        expected_salt = state.variables.salt[2:-2, 2:-2, :, state.variables.tau] + 10.0
        assert actual.variables["salt"].dimensions == ("time", "zt", "yt", "xt")
        assert_allclose_compact(
            np.asarray(actual.variables["salt"])[0],
            np.transpose(expected_salt),
        )
        expected_u = state.variables.u[2:-2, 2:-2, :, state.variables.tau] + 10.0
        assert actual.variables["u"].dimensions == ("time", "zt", "yt", "xu")
        assert_allclose_compact(
            np.asarray(actual.variables["u"])[0],
            np.transpose(expected_u),
        )
        expected_taux = state.variables.surface_taux[2:-2, 2:-2] + 10.0
        assert actual.variables["surface_taux"].dimensions == ("time", "yt", "xu")
        assert_allclose_compact(
            np.asarray(actual.variables["surface_taux"])[0],
            np.transpose(expected_taux),
        )
        expected_psi = state.variables.psi[2:-2, 2:-2, state.variables.tau] + 10.0
        assert actual.variables["psi"].dimensions == ("time", "yu", "xu")
        assert_allclose_compact(
            np.asarray(actual.variables["psi"])[0],
            np.transpose(expected_psi),
        )
        assert actual.variables["psi"].attrs["units"] == "m^3/s"
    assert adapter.empty


def test_veros_snapshot_output_uses_native_state_variables(tmp_path: Path) -> None:
    state = _make_veros_output_state()
    setup_state = SimpleNamespace(
        _veros_state=state,
        output_variables=("temp", "surface_taux"),
        output_adapter=_make_veros_output_adapter(),
    )
    component_state = _runtime_component_state(
        "OCN",
        {"temp": np.full((2, 3, 2), -999.0)},
    )
    output = tmp_path / "OCN.snapshot.nc"

    veros_output_module.write_veros_snapshot_output(
        setup_state,
        component_state,
        output,
        datetime(2000, 1, 2),
        logger=None,
    )

    with h5netcdf.File(output, "r") as actual:
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
        assert actual.variables["temp"].dimensions == ("time", "zt", "yt", "xt")
        assert actual.variables["temp"].attrs["units"] == "deg C"
        assert actual.variables["surface_taux"].dimensions == ("time", "yt", "xu")
        expected_temp = state.variables.temp[2:-2, 2:-2, :, state.variables.tau]
        assert_allclose_compact(
            np.asarray(actual.variables["temp"])[0],
            np.transpose(expected_temp),
        )


def test_veros_record_period_output_accumulates_and_writes_mean_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _make_veros_output_state()
    adapter = _make_veros_output_adapter()
    monkeypatch.chdir(tmp_path)

    first_written = veros_output_module.record_veros_period_output(
        adapter,
        _make_veros_output_state(offset=0.0),
        output_variables=("temp", "salt", "u", "surface_taux", "psi"),
        output_time=datetime(2000, 1, 1),
        dt=timedelta(hours=1),
        output_frequency="day",
    )
    second_written = veros_output_module.record_veros_period_output(
        adapter,
        _make_veros_output_state(offset=20.0),
        output_variables=("temp", "salt", "u", "surface_taux", "psi"),
        output_time=datetime(2000, 1, 2),
        dt=timedelta(days=1),
        output_frequency=None,
    )

    assert not first_written
    assert second_written
    assert adapter.empty
    with h5netcdf.File(tmp_path / "veros.averages.2000-01-02.nc", "r") as actual:
        assert actual.variables["time"].attrs["calendar"] == "proleptic_gregorian"
        assert_allclose_compact(
            np.asarray(actual.variables["xt"]),
            state.variables.xt[2:-2],
        )
        assert actual.variables["temp"].dimensions == ("time", "zt", "yt", "xt")


def test_veros_output_variables_rejects_bare_string() -> None:
    import vercor.setups.external.veros_output as veros_output_module

    with pytest.raises(ValueError, match="output_variables"):
        veros_output_module.normalize_veros_output_variables(
            "temp",
            settings=SimpleNamespace(enable_streamfunction=True),
        )


def test_veros_set_variable_updates_only_interior_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = _FakeVariableStore(temp=np.zeros((8, 8, 1), dtype=float))
    state = SimpleNamespace(variables=variables)

    monkeypatch.setattr(
        veros_state_module, "copy_state", lambda tree, jitted=True: deepcopy(tree)
    )

    updated = veros_state_module.set_variable(
        state,
        "temp",
        jnp.full((4, 4, 1), 9.0),
        jitted=False,
    )

    assert isinstance(updated.variables.temp, np.ndarray)
    assert_allclose_compact(
        updated.variables.temp[2:-2, 2:-2, :], np.full((4, 4, 1), 9.0)
    )
    assert np.count_nonzero(updated.variables.temp[:2, :, :]) == 0
    assert np.count_nonzero(updated.variables.temp[-2:, :, :]) == 0
    assert np.count_nonzero(updated.variables.temp[:, :2, :]) == 0
    assert np.count_nonzero(updated.variables.temp[:, -2:, :]) == 0


def test_veros_initialize_validates_timestep_multiple() -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.dt_tracer = 7.0

    with pytest.raises(ValueError, match="dt_tracer"):
        component.initialize(
            cast(Any, component),
            _make_coupler(dt_seconds=20.0, run_order=["OCN"]),
        )


def test_veros_initialize_can_spin_up_and_extract_surface_temperature() -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.dt_tracer = 10.0
    component.do_spinup = True
    component.spinup_time = timedelta(seconds=20.0)
    component.spinup_steps = 2
    component._veros_state = _make_fake_veros_state(surface_temperature=10.0)
    component.name = "OCN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {}
    component.settings = VercorSettings()
    component.output_adapter = _make_veros_output_adapter()

    step_calls = {"count": 0}

    def fake_step_function(state: Any) -> Any:
        step_calls["count"] += 1
        return state

    component._step_function = fake_step_function

    hook_component = DataComponent.from_fields(
        name="OCN",
        grid=component.grid,
        settings=component.settings,
    )
    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component.initialize(cast(Any, hook_component), coupler)

    assert component.model_substeps == 2
    assert step_calls["count"] == 2
    assert isinstance(hook_component.data["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(
        hook_component.data["sea_surface_temperature"],
        np.full((4, 4), 283.15),
    )


def test_veros_initialize_spinup_accumulates_selected_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.dt_tracer = 10.0
    component.do_spinup = True
    component.spinup_time = timedelta(seconds=20.0)
    component.spinup_steps = 2
    component._veros_state = _make_fake_veros_state(surface_temperature=10.0)
    component.name = "OCN"
    component.output_variables = ("temp",)
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {}
    component.settings = VercorSettings()
    component.output_adapter = _make_veros_output_adapter()

    accumulated_states: list[Any] = []
    accumulated_variables: list[tuple[str, ...]] = []
    step_calls = {"count": 0}

    def fake_step_function(state: Any) -> Any:
        _ = state
        step_calls["count"] += 1
        next_state = _make_fake_veros_state(surface_temperature=10.0)
        next_state.variables.step_id = step_calls["count"]
        return next_state

    def fake_extract_veros_output_snapshot(
        veros_state: Any,
        output_variables: tuple[str, ...],
    ) -> dict[str, OutputVariable]:
        accumulated_states.append(veros_state)
        accumulated_variables.append(output_variables)
        return {"temp": OutputVariable(("x",), np.asarray([1.0]))}

    component._step_function = fake_step_function
    monkeypatch.setattr(
        veros_gcm_state_module._veros_output,
        "extract_veros_output_snapshot",
        fake_extract_veros_output_snapshot,
    )

    hook_component = DataComponent.from_fields(
        name="OCN",
        grid=component.grid,
        settings=component.settings,
    )
    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component.initialize(cast(Any, hook_component), coupler)

    assert [state.variables.step_id for state in accumulated_states] == [1, 2]
    assert accumulated_variables == [("temp",), ("temp",)]
    assert_allclose_compact(
        component.output_adapter.variables["temp"].counts,
        np.asarray([2]),
    )
    assert isinstance(hook_component.data["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(
        hook_component.data["sea_surface_temperature"],
        np.full((4, 4), 283.15),
    )


def test_veros_constructor_builds_jax_backed_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.ones((8, 8, 1), dtype=float)
    mask[2, 3, 0] = 0.0
    state = SimpleNamespace(
        settings=SimpleNamespace(dt_tracer=600.0),
        variables=SimpleNamespace(xt=np.arange(8.0), yt=np.arange(8.0), maskT=mask),
    )

    class _FakeGlobalFourDegree:
        def __init__(self, override: dict[str, Any]) -> None:
            self.override = override
            self.state = state
            self.step = lambda veros_state: None

        def setup(self) -> None:
            return None

    monkeypatch.setattr(
        veros_setup_module, "CustomGlobalFourDegree", _FakeGlobalFourDegree
    )
    monkeypatch.setattr(
        veros_state_module, "copy_state", lambda tree, jitted=True: tree
    )

    component = veros_gcm_module.make_veros_gcm(
        custom_parameters={"dt_tracer": 600.0},
        output_frequency="month",
        output_variables=("temp", "surface_taux"),
        jitted=False,
    )

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.field_spec.inputs == (
        "model_level_height",
        "u_velocity",
        "v_velocity",
        "potential_temperature",
        "specific_humidity",
        "density",
        "temperature",
        "net_shortwave_radiation_flux",
        "downward_longwave_radiation_flux",
    )
    assert component.field_spec.outputs == ("sea_surface_temperature",)
    assert component.grid.binary_mask.shape == (4, 4)
    assert callable(component_snapshot_writer(component))
    expected_mask = np.ones((4, 4))
    expected_mask[1, 0] = 0.0
    assert_allclose_compact(component.grid.binary_mask, expected_mask)


@pytest.mark.parametrize(
    ("restore_to_climatology", "expected_qnec"),
    [
        (False, np.zeros((2, 2, 1))),
        (True, np.asarray([[[3.0], [5.0]], [[4.0], [6.0]]])),
    ],
)
def test_veros_step_sets_forcing_fields_and_refreshes_sst(
    monkeypatch: pytest.MonkeyPatch,
    restore_to_climatology: bool,
    expected_qnec: np.ndarray,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = restore_to_climatology
    component.model_substeps = 2
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.name = "OCN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {"sea_surface_temperature": np.zeros((4, 4), dtype=float)}
    component.settings = VercorSettings()

    set_calls: list[tuple[str, np.ndarray]] = []

    def fake_set_variable(
        state: Any, variable_name: str, variable_value: Any, jitted: bool = True
    ) -> Any:
        _ = jitted
        assert isinstance(variable_value, jax.Array)
        set_calls.append((variable_name, np.asarray(variable_value)))
        return state

    def fake_step_function(state: Any) -> Any:
        state.variables.temp = np.full((8, 8, 1, 1), 15.0, dtype=float)
        return state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, settings: (
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, 12.0]]),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        ),
    )
    monkeypatch.setattr(veros_state_module, "set_variable", fake_set_variable)
    component._step_function = fake_step_function

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component_state = _runtime_component_state("OCN", component.data)
    step_context = ComponentStepContext(
        dt_seconds=20.0,
        settings=coupler.settings,
        time=datetime(2000, 1, 1),
        logger=coupler.logger,
    )
    updates = veros_runtime_module.step_veros_runtime(
        component,
        component_state.data.to_mapping(),
        step_context,
        None,
    )
    component_state = component_state.with_data(component_state.data.set_many(updates))

    expected_names = ["taux", "tauy", "qnet", "qnec"]
    assert [name for name, _ in set_calls] == expected_names
    assert_allclose_compact(
        set_calls[0][1], np.asarray([[[1.0], [3.0]], [[2.0], [4.0]]])
    )
    assert_allclose_compact(set_calls[3][1], expected_qnec)
    assert_allclose_compact(
        component_state.data.get("sea_surface_temperature"),
        np.full((4, 4), 288.15),
    )
    assert isinstance(component_state.data.get("sea_surface_temperature"), jax.Array)


def test_veros_step_records_selected_outputs_and_writes_on_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = False
    component.model_substeps = 0
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.name = "OCN"
    component.output_variables = ("temp",)
    component.output_frequency = "day"
    component.output_adapter = _make_veros_output_adapter()
    component._step_function = lambda state: state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, settings: (
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        ),
    )
    monkeypatch.setattr(
        veros_state_module,
        "set_variable",
        lambda *args, **kwargs: args[0],
    )

    written: dict[str, Any] = {}

    def fake_record_veros_period_output(
        adapter: ComponentOutputAdapter,
        veros_state: Any,
        *,
        output_variables: tuple[str, ...],
        output_time: datetime,
        dt: timedelta,
        output_frequency: str | None,
        logger: Any | None = None,
    ) -> bool:
        _ = logger
        written["adapter"] = adapter
        written["state"] = veros_state
        written["variables"] = output_variables
        written["path"] = f"veros.averages.{output_time.strftime('%Y-%m-%d')}.nc"
        written["output_time"] = output_time
        written["dt"] = dt
        written["output_frequency"] = output_frequency
        adapter.accumulate({"temp": OutputVariable(("x",), np.asarray([1.0]))})
        written["counts"] = adapter.variables["temp"].counts.copy()
        adapter.reset()
        return True

    monkeypatch.setattr(
        veros_runtime_module._veros_output,
        "record_veros_period_output",
        fake_record_veros_period_output,
    )

    context = ComponentStepContext(
        dt_seconds=86400.0,
        settings=VercorSettings(),
        time=datetime(2000, 1, 2),
        logger=None,
    )

    veros_runtime_module.step_veros_runtime(component, {}, context, None)

    assert written["adapter"] is component.output_adapter
    assert written["state"] is component._veros_state
    assert written["variables"] == ("temp",)
    assert_allclose_compact(written["counts"], np.asarray([1]))
    assert written["path"] == "veros.averages.2000-01-02.nc"
    assert written["output_time"] == datetime(2000, 1, 2)
    assert written["dt"] == timedelta(days=1)
    assert written["output_frequency"] == "day"
    assert component.output_adapter.empty


def test_veros_step_skips_output_when_no_variables_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = False
    component.model_substeps = 0
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.name = "OCN"
    component.output_variables = ()
    component.output_frequency = None
    component.output_adapter = _make_veros_output_adapter()
    component._step_function = lambda state: state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, settings: (
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
        ),
    )
    monkeypatch.setattr(
        veros_state_module,
        "set_variable",
        lambda *args, **kwargs: args[0],
    )
    monkeypatch.setattr(
        veros_runtime_module._veros_output,
        "extract_veros_output_snapshot",
        lambda *args, **kwargs: pytest.fail("unexpected Veros output extraction"),
    )

    context = ComponentStepContext(
        dt_seconds=86400.0,
        settings=VercorSettings(),
        time=datetime(2000, 1, 2),
        logger=None,
    )

    veros_runtime_module.step_veros_runtime(component, {}, context, None)

    assert component.output_adapter.empty


def test_veros_step_nan_cleans_forcing_fields_before_set_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = True
    component.model_substeps = 0
    component.jitted = False
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.name = "OCN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {"sea_surface_temperature": np.zeros((4, 4), dtype=float)}
    component.settings = VercorSettings()

    set_calls: list[tuple[str, np.ndarray]] = []

    def fake_set_variable(
        state: Any, variable_name: str, variable_value: Any, jitted: bool = True
    ) -> Any:
        _ = jitted
        assert isinstance(variable_value, jax.Array)
        set_calls.append((variable_name, np.asarray(variable_value)))
        return state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, settings: (
            np.asarray([[1.0, np.nan], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [np.nan, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, np.nan]]),
            np.asarray([[12.0, 13.0], [14.0, np.nan]]),
        ),
    )
    monkeypatch.setattr(veros_state_module, "set_variable", fake_set_variable)
    component._step_function = lambda state: state

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component_state = _runtime_component_state("OCN", component.data)
    step_context = ComponentStepContext(
        dt_seconds=20.0,
        settings=coupler.settings,
        time=datetime(2000, 1, 1),
        logger=coupler.logger,
    )
    _ = veros_runtime_module.step_veros_runtime(
        component,
        component_state.data.to_mapping(),
        step_context,
        None,
    )

    assert [name for name, _ in set_calls] == ["taux", "tauy", "qnet", "qnec"]
    assert_allclose_compact(
        set_calls[0][1], np.asarray([[[1.0], [3.0]], [[0.0], [4.0]]])
    )
    assert_allclose_compact(
        set_calls[1][1], np.asarray([[[5.0], [0.0]], [[6.0], [8.0]]])
    )
    assert_allclose_compact(
        set_calls[2][1], np.asarray([[[9.0], [11.0]], [[10.0], [0.0]]])
    )
    assert_allclose_compact(
        set_calls[3][1], np.asarray([[[12.0], [14.0]], [[13.0], [0.0]]])
    )
