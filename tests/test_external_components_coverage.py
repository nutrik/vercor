from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Literal, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from veros.core.external.solvers import (
    get_linear_solver as veros_get_linear_solver,
)

import vercor.setups._external.jax_gcm as jax_gcm_module
import vercor.setups._external._jax_gcm_pytree as jax_gcm_pytree_module
import vercor.setups._external.jax_gcm_fields as jax_gcm_fields_module
import vercor.setups._external.jax_gcm_output as jax_gcm_output_module
import vercor.setups._external.jax_gcm_runtime as jax_gcm_runtime_module
import vercor.setups._external.jax_gcm_state as jax_gcm_state_module
import vercor.setups._external.veros_fluxes as veros_fluxes_module
import vercor.setups._external.veros_gcm as veros_gcm_module
import vercor.setups._external.veros_gcm_state as veros_gcm_state_module
import vercor.setups._external.veros_output as veros_output_module
import vercor.setups._external.veros_runtime as veros_runtime_module
import vercor.setups._external.veros_runtime_settings as veros_runtime_settings_module
import vercor.setups._external.veros_setup as veros_setup_module
import vercor.setups._external.veros_state as veros_state_module
from tests._coverage_support import capture_logger_output, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.calendar import DateTime360, DateTime365
from vercor.components.contracts import PrefillContext
from vercor.components import StepResult
from vercor.components.data import DataComponent
from vercor.components.contexts import SetupContext, StepContext
from vercor.exceptions import ComponentError
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputVariable,
    PeriodOutput,
    SnapshotContext,
)
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output._period import period_mean_sample_to_output_variable
from vercor.output._session import _OutputAccumulator
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups import VerosConfig
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor.state import ComponentState


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.messages.append(message.format(*args) if args else message)


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
        object.__setattr__(self, "__fields__", metadata.keys())

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
    unlock_calls: int = 0

    def unlock(self) -> _NullContext:
        self.unlock_calls += 1
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
        self._variables: Any = {}
        self.timers: dict[str, Any] = {}
        self.profile_timers: dict[str, Any] = {}

    @property
    def variables(self) -> Any:
        return self._variables


class _FakeVerosStepState:
    def __init__(self, counter: int) -> None:
        self.counter = counter


def _accumulate_frames(frames: Sequence[OutputFrame]) -> _OutputAccumulator:
    first, *remaining = frames
    accumulator = _OutputAccumulator.zeros_from_frame(first).add_frame(first)
    for frame in remaining:
        accumulator = accumulator.add_frame(frame)
    return accumulator


def _period_variables(
    accumulator: _OutputAccumulator,
    *,
    time_dim: str,
    reverse_value_dims: bool = False,
) -> dict[str, OutputVariable]:
    return {
        name: period_mean_sample_to_output_variable(
            sample,
            time_dim=time_dim,
            value_dims=tuple(reversed(sample.dims)) if reverse_value_dims else None,
            dimension_order=accumulator.dimension_order,
        )
        for name, sample in accumulator.mean_frame().variables.items()
    }


def _make_coupler(
    *,
    dt_seconds: float,
    run_order: list[str],
) -> SetupContext:
    return SetupContext(
        start=datetime(2000, 1, 1),
        dt_seconds=dt_seconds,
        logger=cast(Any, _RecordingLogger()),
        run_order=tuple(run_order),
    )


def _make_fake_veros_state(surface_temperature: float = 10.0) -> Any:
    temp = np.full((8, 8, 1, 1), surface_temperature, dtype=float)
    variables = SimpleNamespace(temp=temp, tau=0)
    return SimpleNamespace(variables=variables)


def _make_copyable_fake_veros_state(surface_temperature: float = 10.0) -> Any:
    state = _ConstructedVerosState({}, {}, {}, {})
    state._variables = _FakeVariableStore(
        temp=np.full((8, 8, 1, 1), surface_temperature, dtype=float),
        tau=0,
        **{
            name: np.zeros((6, 6, 1), dtype=float)
            for name in ("taux", "tauy", "qnet", "qnec")
        },
    )
    return state


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
        eke=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + 150.0 + offset,
        tke=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + 175.0 + offset,
        rho=np.arange(6 * 7 * 2 * 3, dtype=float).reshape(6, 7, 2, 3) + 190.0 + offset,
        surface_taux=np.arange(6 * 7, dtype=float).reshape(6, 7) + 200.0 + offset,
        psi=np.arange(6 * 7 * 3, dtype=float).reshape(6, 7, 3) + 300.0 + offset,
    )
    active_variables = (
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "eke",
        "tke",
        "rho",
        "surface_taux",
        "surface_tauy",
        "psi",
    )
    manifest_variables = (*active_variables, "xt", "xu", "yt", "yu", "zt", "zw")
    return SimpleNamespace(
        settings=SimpleNamespace(
            enable_eke=True,
            enable_neutral_diffusion=True,
            enable_streamfunction=True,
            enable_tke=True,
            coord_degree=True,
        ),
        variables=variables,
        var_meta={
            name: SimpleNamespace(
                active=True,
                dims=veros_output_module.veros_variables.VARIABLES[name].dims,
            )
            for name in manifest_variables
        },
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
        "c": True,
    }

    converted = jax_gcm_pytree_module.tree_as_real_dtype(tree)

    assert jnp.issubdtype(converted["a"].dtype, jnp.floating)
    assert jnp.issubdtype(converted["b"].dtype, jnp.floating)
    assert jnp.issubdtype(converted["c"].dtype, jnp.floating)
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


def test_prepare_surface_temperature_forcing_preserves_fractional_cells() -> None:
    total_surface_temperature = jnp.asarray([[270.0, 281.0], [282.0, 567.0]])
    land_fraction_mask = jnp.asarray([[1.0, 0.0], [0.25, 0.75]])

    def forcing_sum(temperature: jax.Array) -> jax.Array:
        land, sea = jax_gcm_fields_module.prepare_surface_temperature_forcing(
            temperature,
            land_fraction_mask,
        )
        return jnp.sum(land + sea)

    land_surface_temperature, sea_surface_temperature = jax.jit(
        jax_gcm_fields_module.prepare_surface_temperature_forcing
    )(total_surface_temperature, land_fraction_mask)
    gradient = jax.grad(forcing_sum)(total_surface_temperature)

    assert_allclose_compact(
        land_surface_temperature,
        np.asarray([[270.0, 288.15], [282.0, 567.0]]),
    )
    assert_allclose_compact(
        sea_surface_temperature,
        np.asarray([[288.15, 281.0], [282.0, 567.0]]),
    )
    assert_allclose_compact(gradient, np.asarray([[1.0, 1.0], [2.0, 2.0]]))


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

    mapped_fields = jax.jit(
        lambda *args: jax_gcm_fields_module.map_jcm_output_fields(
            *args,
            dtype=DTypePolicy(),
        )
    )(
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


def test_generate_step_function_threads_jcm_2_physics_carry() -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.save_interval = timedelta(days=2)
    component.coupling_timestep = timedelta(hours=12)
    component._dtype_policy = DTypePolicy()

    calls: dict[str, Any] = {}

    class _FakeModel:
        def run_from_state_with_carry(
            self,
            initial_state: Any,
            forcing: Any,
            save_interval: float,
            total_time: float,
            output_averages: bool,
            initial_physics_state: Any,
        ) -> tuple[str, str, Any]:
            calls["run"] = (
                initial_state,
                initial_physics_state,
                save_interval,
                total_time,
                output_averages,
                forcing,
            )
            predictions = SimpleNamespace(
                dynamics={"wind": jnp.asarray([[1.0, 3.0], [5.0, 7.0]])},
                physics={"heat": jnp.asarray([[2.0, 4.0], [6.0, 8.0]])},
            )
            return "next-dycore", "next-carry", predictions

    component.model = _FakeModel()
    state = jax_gcm_state_module.JCMState(
        dynamics={},
        physics={},
        dycore_state="initial-dycore",
        physics_carry="initial-carry",
    )

    next_state, predictions = component._generate_step_function(jitted=False)(
        state, "forcing"
    )

    assert calls["run"] == (
        "initial-dycore",
        "initial-carry",
        2.0,
        0.5,
        False,
        "forcing",
    )
    assert next_state.dycore_state == "next-dycore"
    assert next_state.physics_carry == "next-carry"
    assert_allclose_compact(next_state.dynamics["wind"], np.asarray([3.0, 5.0]))
    assert_allclose_compact(next_state.physics["heat"], np.asarray([4.0, 6.0]))
    assert predictions.physics["heat"].shape == (2, 2)


def test_bootstrap_jcm_state_captures_native_state_and_physics_carry() -> None:
    calls = {"bootstrap": 0}

    class _FakeDycore:
        def to_physics_state(self, state: Any) -> dict[str, Any]:
            return {"converted": state}

    class _FakeModel:
        def __init__(self) -> None:
            self.dycore = _FakeDycore()
            self._final_dycore_state: Any = None
            self._final_physics_state: Any = None

        def bootstrap_state(self) -> None:
            calls["bootstrap"] += 1
            self._final_dycore_state = {"vorticity": jnp.asarray(1.0)}
            self._final_physics_state = {"heating": jnp.asarray(2.0)}

    model = _FakeModel()
    state = jax_gcm_state_module._bootstrap_jcm_state(cast(Any, model))

    assert calls == {"bootstrap": 1}
    assert state.dynamics == {"converted": state.dycore_state}
    assert state.dycore_state is model._final_dycore_state
    assert state.physics is model._final_physics_state
    assert state.physics_carry is model._final_physics_state


@pytest.mark.parametrize(
    ("dycore_state", "physics_carry"),
    [(None, object()), (object(), None)],
)
def test_bootstrap_jcm_state_rejects_missing_final_state(
    dycore_state: Any,
    physics_carry: Any,
) -> None:
    model = SimpleNamespace(
        bootstrap_state=lambda: None,
        _final_dycore_state=dycore_state,
        _final_physics_state=physics_carry,
    )

    with pytest.raises(RuntimeError, match="did not initialize"):
        jax_gcm_state_module._bootstrap_jcm_state(cast(Any, model))


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
            _ = time_step
            self.coords = coords
            self.terrain = terrain
            self.physics = physics

    monkeypatch.setattr(
        jax_gcm_state_module.Parameters,
        "default",
        staticmethod(lambda: SimpleNamespace()),
    )
    monkeypatch.setattr(
        jax_gcm_state_module,
        "speedy_physics",
        lambda parameters: _FakePhysicsModule(),
    )
    monkeypatch.setattr(jax_gcm_state_module, "Model", _FakeModel)

    component = jax_gcm_module.make_jax_gcm(coords=coords, terrain=terrain)

    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.spec.inputs == (
        "land_surface_temperature",
        "sea_surface_temperature",
        "soil_moisture",
    )
    assert "soil_moisture" in component.spec.initial_fields
    assert component.spec.outputs == (
        "land_surface_temperature",
        "sea_surface_temperature",
        "total_surface_temperature",
        *jax_gcm_fields_module.JAXGCM_OUTPUT_GRID_FIELD_NAMES,
        "pressure",
    )
    assert_allclose_compact(component.grid.longitude, np.asarray([0.0, 180.0]))
    assert_allclose_compact(component.grid.latitude, np.asarray([-45.0, 0.0, 45.0]))
    assert_allclose_compact(component.grid.binary_mask, np.ones((3, 2)))
    assert callable(component.spec.output.snapshot_writer)


def test_jax_gcm_initialize_validates_timestep_multiple() -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.spinup_time = timedelta(hours=6)
    component.model_timestep = timedelta(minutes=45)
    component.model = SimpleNamespace()
    component.grid = make_test_grid()

    with pytest.raises(ValueError, match="model_timestep"):
        component.setup(
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
    component.name = "CUSTOM_ATMOSPHERE"
    component.grid = make_test_grid()
    component.data = {}
    component._dtype_policy = DTypePolicy()
    component.save_interval = timedelta(days=1)
    component.forcing_data = "provided-forcing"
    initial_dycore_state = {"marker": jnp.asarray(0.0)}
    initial_physics_carry = {"marker": jnp.asarray(10.0)}
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
        ),
        dycore=SimpleNamespace(to_physics_state=lambda state: {"converted": state}),
        _final_dycore_state=initial_dycore_state,
        _final_physics_state=initial_physics_carry,
        bootstrap_state=lambda: None,
    )

    def advance_state(state: Any, forcing: Any) -> tuple[Any, str]:
        _ = forcing
        return (
            jax_gcm_state_module.JCMState(
                dynamics=state.dynamics,
                physics=state.physics,
                dycore_state={"marker": state.dycore_state["marker"] + 1.0},
                physics_carry={"marker": state.physics_carry["marker"] + 1.0},
            ),
            "unused",
        )

    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: advance_state,
    )
    hook_component = DataComponent(
        name="CUSTOM_ATMOSPHERE",
        grid=component.grid,
    )
    coupler = _make_coupler(
        dt_seconds=3600.0,
        run_order=["CUSTOM_ATMOSPHERE"],
    )
    setup_result = component.setup(cast(Any, hook_component), coupler)

    assert component.coupling_timestep == timedelta(hours=1)
    assert component.spinup_steps == 2
    assert component.forcing == "provided-forcing"
    assert float(component._state.dycore_state["marker"]) == 2.0
    assert float(component._state.physics_carry["marker"]) == 12.0
    assert (
        setup_result.fields["sea_surface_temperature"]
        == jax_gcm_fields_module.REFERENCE_SURFACE_TEMPERATURE
    )


def test_jax_gcm_spinup_normalizes_loaded_forcing_to_runtime_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.spinup_time = timedelta(hours=1)
    component.model_timestep = timedelta(hours=1)
    component.jitted = False
    component.do_spinup = True
    component.name = "ATM"
    component.grid = make_test_grid()
    component.data = {}
    component.save_interval = timedelta(days=1)
    forcing_values = {
        field_name: jnp.ones((2, 3), dtype=jnp.float32)
        for field_name in (
            "alb0",
            "sice_am",
            "snowc_am",
            "soilw_am",
            "stl_am",
            "sea_surface_temperature",
        )
    }
    component.forcing_data = jax_gcm_state_module.ForcingData.zeros((2, 3)).copy(
        stl_am=forcing_values["stl_am"],
        sea_surface_temperature=forcing_values["sea_surface_temperature"],
    )
    initial_dycore_state = {
        "temperature": jnp.ones((2, 3), dtype=jnp.float32),
        "mode": jnp.asarray(1, dtype=jnp.int32),
        "marker": jnp.asarray(0.0, dtype=jnp.float32),
    }
    initial_physics_carry = {
        "heating": jnp.zeros((2, 3), dtype=jnp.float32),
        "marker": jnp.asarray(10.0, dtype=jnp.float32),
    }
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
        ),
        dycore=SimpleNamespace(to_physics_state=lambda state: state),
        _final_dycore_state=initial_dycore_state,
        _final_physics_state=initial_physics_carry,
        bootstrap_state=lambda: None,
    )

    spinup_forcing_dtypes: list[set[jnp.dtype[Any]]] = []
    spinup_state_dtypes: list[set[jnp.dtype[Any]]] = []

    def record_spinup_inputs(state: Any, forcing: Any) -> tuple[Any, str]:
        spinup_forcing_dtypes.append(
            {jnp.asarray(leaf).dtype for leaf in jax.tree_util.tree_leaves(forcing)}
        )
        spinup_state_dtypes.append(
            {
                jnp.asarray(leaf).dtype
                for leaf in jax.tree_util.tree_leaves(state)
                if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact)
            }
        )
        return (
            jax_gcm_state_module.JCMState(
                dynamics=state.dynamics,
                physics=state.physics,
                dycore_state={
                    **state.dycore_state,
                    "marker": state.dycore_state["marker"] + 1.0,
                },
                physics_carry={
                    **state.physics_carry,
                    "marker": state.physics_carry["marker"] + 1.0,
                },
            ),
            "unused",
        )

    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: record_spinup_inputs,
    )

    hook_component = DataComponent(name="ATM", grid=component.grid)
    component.setup(
        cast(Any, hook_component),
        SetupContext(
            start=datetime(2000, 1, 1),
            dt_seconds=3600.0,
            logger=cast(Any, _RecordingLogger()),
            run_order=("ATM",),
            dtype=DTypePolicy(enable_x64=True),
        ),
    )

    assert spinup_forcing_dtypes == [{jnp.dtype(jnp.float64)}]
    assert spinup_state_dtypes == [{jnp.dtype(jnp.float64)}]
    assert component._state.dycore_state["mode"].dtype == jnp.dtype(jnp.int32)
    assert float(component._state.dycore_state["marker"]) == 1.0
    assert float(component._state.physics_carry["marker"]) == 11.0


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
    component._dtype_policy = DTypePolicy()
    component.save_interval = timedelta(days=1)
    component.forcing_data = None
    initial_dycore_state = {"marker": jnp.asarray(0.0)}
    initial_physics_carry = {"marker": jnp.asarray(10.0)}
    component.model = SimpleNamespace(
        coords=SimpleNamespace(
            horizontal=SimpleNamespace(nodal_shape=(2, 3)),
        ),
        dycore=SimpleNamespace(to_physics_state=lambda state: {"converted": state}),
        _final_dycore_state=initial_dycore_state,
        _final_physics_state=initial_physics_carry,
        bootstrap_state=lambda: None,
    )

    forcing = _FakeForcing()
    step_calls = {"count": 0}

    def disabled_spinup_step(state: Any, forcing_data: Any) -> tuple[Any, str]:
        _ = forcing_data
        step_calls["count"] += 1
        return state, "unused"

    monkeypatch.setattr(
        jax_gcm_state_module,
        "default_forcing",
        lambda horizontal: forcing,
    )
    monkeypatch.setattr(
        component,
        "_generate_step_function",
        lambda jitted: disabled_spinup_step,
    )

    hook_component = DataComponent(
        name="ATM",
        grid=component.grid,
    )
    component.setup(
        cast(Any, hook_component),
        _make_coupler(dt_seconds=3600.0, run_order=["ATM"]),
    )

    assert component.forcing is forcing
    assert forcing.copy_calls == []
    assert step_calls["count"] == 0


def test_jax_gcm_step_maps_outputs_without_owning_output_cadence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    component.name = "ATM"
    component.grid = make_test_grid(name="atm")
    component._dtype_policy = DTypePolicy()
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
    component._state = jax_gcm_state_module.JCMState(
        dynamics={},
        physics={},
        dycore_state={"marker": jnp.asarray(0.0)},
        physics_carry={"marker": jnp.asarray(10.0)},
    )

    p = {
        "_surface_flux": SimpleNamespace(
            shf=np.full((2, 2, 2), 5.0, dtype=float),
            evap=np.full((2, 2, 2), 2.0, dtype=float),
            rlds=np.full((2, 2), 40.0, dtype=float),
        ),
        "_shortwave_rad": SimpleNamespace(rsns=np.full((2, 2), 30.0, dtype=float)),
    }
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
            jax_gcm_state_module.JCMState(
                dynamics=d,
                physics=p,
                dycore_state={"marker": state.dycore_state["marker"] + 1.0},
                physics_carry={"marker": state.physics_carry["marker"] + 2.0},
            ),
            prediction,
        ),
    )
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

    coupler = _make_coupler(dt_seconds=3600.0, run_order=["ATM"])
    runtime_data = dict(component.data)
    runtime_incoming: dict[str, Any] = {}
    runtime_outgoing: dict[str, Any] = {}
    hook_component = DataComponent(
        name="ATM",
        grid=component.grid,
    )
    prefill_result = jax_gcm_runtime_module.prefill_jax_gcm_runtime_fields(
        component,
        cast(Any, hook_component),
        PrefillContext(
            fields=runtime_data,
            received=runtime_incoming,
            sent=runtime_outgoing,
        ),
    )
    runtime_data.update(cast(Any, prefill_result.fields))
    runtime_incoming.update(cast(Any, prefill_result.received))
    runtime_outgoing.update(cast(Any, prefill_result.sent))
    component_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping(runtime_data),
        received=FieldStore.from_mapping(runtime_incoming),
        sent=FieldStore.from_mapping(runtime_outgoing),
        payload=jax_gcm_runtime_module.create_jax_gcm_runtime_payload(component),
    )
    step_context = StepContext(
        dt_seconds=timedelta(days=1).total_seconds(),
        time=datetime(2000, 1, 2),
        logger=coupler.logger,
    )
    step_result = jax_gcm_runtime_module.step_jax_gcm_component(
        component,
        component_state.fields.to_mapping(),
        step_context,
        component_state.payload,
    )
    component_state = component_state.with_fields(
        component_state.fields.set_many(step_result.fields)
    ).with_payload(step_result.payload)
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
    data = component_state.fields
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
        np.full(
            (2, 2),
            -2.0 / 1e3 * coupler.constants.latent_heat_of_vaporization * 2.0,
        ),
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
        coupler.constants.dry_air_molecular_weight
        / coupler.constants.universal_gas_constant
        * np.full((2, 2), 80000.0)
        / np.asarray([[284.0, 286.0], [285.0, 287.0]]),
    )
    assert_allclose_compact(
        data.get("potential_temperature"),
        np.asarray([[284.0, 286.0], [285.0, 287.0]])
        * (coupler.constants.reference_pressure / 80000.0)
        ** coupler.constants.dry_air_kappa,
    )
    assert_allclose_compact(data.get("model_level_height"), np.full((2, 2), 150.0))
    assert float(step_result.payload.jcm_state.dycore_state["marker"]) == 1.0
    assert float(step_result.payload.jcm_state.physics_carry["marker"]) == 12.0


def test_jax_gcm_runtime_rejects_missing_speedy_diagnostics() -> None:
    with pytest.raises(ComponentError, match="_shortwave_rad"):
        jax_gcm_runtime_module._required_speedy_diagnostics(
            {"_surface_flux": object()},
            component_name="ATM",
        )


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
    frames = tuple(
        OutputFrame(
            jax_gcm_output_module.jax_gcm_prediction_output_variables(
                prediction,
                coords=coords,
                physics_module=physics_module,
            ),
            sample_dimension=jax_gcm_output_module.JAX_GCM_TIME_DIM,
            time_dimension=jax_gcm_output_module.JAX_GCM_TIME_DIM,
            dimension_order=jax_gcm_output_module.JAX_GCM_OUTPUT_DIMENSION_ORDER,
        )
        for prediction in predictions
    )
    accumulator = _accumulate_frames(frames)
    temperature_index = accumulator.names.index("temperature")
    assert isinstance(accumulator.sum_values[temperature_index], jax.Array)
    assert isinstance(accumulator.counts[temperature_index], jax.Array)

    output = tmp_path / "jcm_output.nc"
    logger_name = "VerCOR.test.jax-gcm-output"
    logger = logging.getLogger(logger_name)
    variables = _period_variables(
        accumulator,
        time_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
    )
    data_variables = jax_gcm_output_module.jax_gcm_data_variables_with_unit_metadata(
        variables,
        jax_gcm_output_module.jax_gcm_unit_metadata(physics_module),
    )

    with capture_logger_output(logger_name) as stream:
        write_netcdf_dataset(
            output=str(output),
            coordinate_variables=jax_gcm_output_module.jax_gcm_coordinate_variables(
                coords=coords,
                output_time=datetime(2000, 1, 2),
            ),
            data_variables=data_variables,
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
    assert_allclose_compact(
        accumulator.counts[temperature_index],
        np.full((3, 2, 3), 2),
    )
    assert physics_module.cached_coords is coords
    assert f"Writing output file:  {output}" in stream.getvalue()


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
    frames = tuple(
        OutputFrame(
            jax_gcm_output_module.jax_gcm_prediction_output_variables(
                prediction,
                coords=coords,
                physics_module=physics_module,
            ),
            sample_dimension=jax_gcm_output_module.JAX_GCM_TIME_DIM,
            time_dimension=jax_gcm_output_module.JAX_GCM_TIME_DIM,
            dimension_order=jax_gcm_output_module.JAX_GCM_OUTPUT_DIMENSION_ORDER,
        )
        for prediction in predictions
    )
    accumulator = _accumulate_frames(frames)
    variables = _period_variables(
        accumulator,
        time_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
    )
    output = tmp_path / "jcm_model_calendar_output.nc"

    write_netcdf_dataset(
        output=str(output),
        coordinate_variables=jax_gcm_output_module.jax_gcm_coordinate_variables(
            coords=coords,
            output_time=output_time,
        ),
        data_variables=jax_gcm_output_module.jax_gcm_data_variables_with_unit_metadata(
            variables,
            jax_gcm_output_module.jax_gcm_unit_metadata(physics_module),
        ),
    )

    with h5netcdf.File(output, "r") as actual:
        time = actual.variables["time"]
        assert time.attrs["calendar"] == expected_calendar
        assert time.attrs["day_of_year"] == expected_day_of_year
        assert time.attrs["days_per_year"] == days_per_year
        assert time.attrs["fixed_30_day_months"] == int(expected_calendar == "360_day")
    temperature_index = accumulator.names.index("temperature")
    assert_allclose_compact(
        accumulator.counts[temperature_index],
        np.ones((3, 2, 3)),
    )


def test_jax_gcm_snapshot_output_uses_final_runtime_payload_not_runtime_data(
    tmp_path: Path,
) -> None:
    coords = _make_jax_gcm_output_coords()
    physics_module = _FakePhysicsModule()
    setup_state = SimpleNamespace(
        model=SimpleNamespace(coords=coords, physics=physics_module)
    )
    jcm_state = SimpleNamespace(
        dynamics={
            "temperature": np.arange(18.0).reshape(3, 2, 3),
            "u_wind": np.full((3, 2, 3), 4.0),
        },
        physics={},
    )
    component_state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": np.full((3, 2, 3), -999.0)}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
        payload=SimpleNamespace(jcm_state=jcm_state),
    )
    output = tmp_path / "ATM.snapshot.nc"

    jax_gcm_output_module.write_jax_gcm_snapshot_output(
        setup_state,
        SnapshotContext(
            component=cast(Any, None),
            state=ComponentState._from_runtime("ATM", None, component_state),
            payload=component_state.payload,
            output_path=output,
            time=datetime(2000, 1, 2),
            logger=None,
        ),
    )

    with h5netcdf.File(output, "r") as actual:
        temperature = actual.variables["temperature"]
        assert temperature.dimensions == ("time", "level", "lat", "lon")
        assert temperature.attrs["units"] == "K"
        assert temperature.attrs["description"] == "temperature"
        assert np.asarray(temperature)[0, 0, 0, 0] != -999.0
        assert_allclose_compact(
            np.asarray(temperature)[0],
            np.transpose(jcm_state.dynamics["temperature"], axes=(0, 2, 1)),
        )


def test_jax_gcm_output_provider_samples_post_step_payload() -> None:
    coords = _make_jax_gcm_output_coords()
    physics_module = _FakePhysicsModule()
    setup_state = SimpleNamespace(
        model=SimpleNamespace(coords=coords, physics=physics_module),
    )
    jcm_state = SimpleNamespace(
        dynamics={"temperature": np.arange(18.0).reshape(3, 2, 3)},
        physics={},
    )
    provider = jax_gcm_output_module.jax_gcm_output_provider(setup_state)

    frame = provider.sample(
        OutputContext(
            component=cast(Any, None),
            state=cast(Any, None),
            payload=SimpleNamespace(jcm_state=jcm_state),
            step=3,
            time=datetime(2000, 1, 2),
            dt=timedelta(days=1),
        )
    )

    assert "temperature" in frame.variables
    assert frame.sample_dimension == "time"
    assert_allclose_compact(
        frame.variables["temperature"].values[0],
        jcm_state.dynamics["temperature"],
    )


def test_jax_gcm_unit_metadata_uses_packaged_speedy_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dynamics_root = tmp_path / "jcm"
    physics_root = tmp_path / "speedy"
    dynamics_root.mkdir()
    physics_root.mkdir()
    (dynamics_root / "dynamics_units_table.csv").write_text(
        "Variable,Units,Description\nunique_dynamics,K,dynamics field\n",
        encoding="utf-8",
    )
    (physics_root / "units_table.csv").write_text(
        "Variable,Units,Description\nunique_physics,W m-2,physics field\n",
        encoding="utf-8",
    )
    roots = {
        "jcm": dynamics_root,
        "jcm.physics.speedy": physics_root,
    }
    monkeypatch.setattr(
        jax_gcm_output_module.resources,
        "files",
        lambda package: roots[package],
    )

    metadata = jax_gcm_output_module.jax_gcm_unit_metadata(
        cast(Any, SimpleNamespace(UNITS_TABLE_CSV_PATH=None))
    )

    assert metadata["unique_dynamics"] == {
        "units": "K",
        "description": "dynamics field",
    }
    assert metadata["unique_physics"] == {
        "units": "W m-2",
        "description": "physics field",
    }


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
        constants: PhysicalConstants,
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
            constants,
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
        PhysicalConstants(),
        DTypePolicy(),
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


def test_configure_veros_runtime_sets_diskless_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_settings = SimpleNamespace()
    fake_veros = ModuleType("veros")
    setattr(fake_veros, "runtime_settings", runtime_settings)
    monkeypatch.setitem(sys.modules, "veros", fake_veros)

    veros_runtime_settings_module.configure_veros_runtime()

    assert runtime_settings.backend == "numpy"
    assert runtime_settings.force_overwrite is True
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


@pytest.mark.parametrize("jitted", (False, True))
def test_veros_copy_state_returns_deepcopy_compatible_state(
    monkeypatch: pytest.MonkeyPatch,
    jitted: bool,
) -> None:
    monkeypatch.setattr(veros_state_module, "VerosState", _ConstructedVerosState)
    source = _make_copyable_fake_veros_state()

    copied = veros_state_module.copy_state(source, jitted=jitted)
    copied_again = deepcopy(copied)

    assert isinstance(copied.settings.__fields__, tuple)
    assert copied_again is not copied
    assert copied_again.settings is not copied.settings
    assert copied_again.variables is not copied.variables


def test_veros_pure_reuses_component_solver_for_copied_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_state = _FakeVerosStepState(counter=1)
    copied_states = iter(
        (_FakeVerosStepState(counter=1), _FakeVerosStepState(counter=1))
    )
    solver = object()
    solver_cache: dict[tuple[Any, ...], Any] = {}

    monkeypatch.setattr(veros_get_linear_solver, "cache", solver_cache)
    monkeypatch.setattr(veros_get_linear_solver.__wrapped__, "cache", solver_cache)
    monkeypatch.setattr(
        veros_state_module,
        "copy_state",
        lambda state, jitted=True: next(copied_states),
    )

    stepped_states: list[Any] = []

    def fake_step(state: Any) -> None:
        assert veros_get_linear_solver(state) is solver
        state.counter += 1
        stepped_states.append(state)

    results = tuple(
        veros_state_module.pure(
            original_state,
            jitted=False,
            step=fake_step,
            linear_solver=solver,
        )
        for _ in range(2)
    )

    assert results == tuple(stepped_states)
    assert [state.counter for state in results] == [2, 2]
    assert original_state.counter == 1
    assert solver_cache == {}


@pytest.mark.parametrize("has_prior_entry", (False, True))
def test_veros_pure_restores_solver_cache_when_step_fails(
    monkeypatch: pytest.MonkeyPatch,
    has_prior_entry: bool,
) -> None:
    original_state = _FakeVerosStepState(counter=1)
    copied_state = _FakeVerosStepState(counter=1)
    component_solver = object()
    prior_solver = object()
    solver_cache: dict[tuple[Any, ...], Any] = {}
    key = (copied_state,)
    if has_prior_entry:
        solver_cache[key] = prior_solver

    monkeypatch.setattr(veros_get_linear_solver, "cache", solver_cache)
    monkeypatch.setattr(veros_get_linear_solver.__wrapped__, "cache", solver_cache)
    monkeypatch.setattr(
        veros_state_module,
        "copy_state",
        lambda state, jitted=True: copied_state,
    )

    def failing_step(state: Any) -> None:
        assert veros_get_linear_solver(state) is component_solver
        raise RuntimeError("native step failed")

    with pytest.raises(RuntimeError, match="native step failed"):
        veros_state_module.pure(
            original_state,
            jitted=False,
            step=failing_step,
            linear_solver=component_solver,
        )

    assert solver_cache == ({key: prior_solver} if has_prior_entry else {})


def test_veros_component_solvers_are_isolated_and_release_owner_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_owner = _FakeVerosStepState(counter=1)
    second_owner = _FakeVerosStepState(counter=1)
    first_copied_state = _FakeVerosStepState(counter=1)
    second_copied_state = _FakeVerosStepState(counter=1)
    unrelated_state = _FakeVerosStepState(counter=1)
    first_solver = object()
    second_solver = object()
    unrelated_solver = object()
    solver_cache: dict[tuple[Any, ...], Any] = {
        (first_owner,): first_solver,
        (second_owner,): second_solver,
        (unrelated_state,): unrelated_solver,
    }
    copied_states = {
        first_owner: first_copied_state,
        second_owner: second_copied_state,
    }

    monkeypatch.setattr(veros_get_linear_solver, "cache", solver_cache)
    monkeypatch.setattr(veros_get_linear_solver.__wrapped__, "cache", solver_cache)
    monkeypatch.setattr(
        veros_state_module,
        "copy_state",
        lambda state, jitted=True: copied_states[state],
    )

    assert veros_state_module.get_component_linear_solver(first_owner) is first_solver
    assert solver_cache == {
        (second_owner,): second_solver,
        (unrelated_state,): unrelated_solver,
    }
    assert veros_state_module.get_component_linear_solver(second_owner) is second_solver
    assert solver_cache == {(unrelated_state,): unrelated_solver}

    observed_solvers: list[Any] = []

    def first_step(state: Any) -> None:
        observed_solvers.append(veros_get_linear_solver(state))

    def failing_second_step(state: Any) -> None:
        observed_solvers.append(veros_get_linear_solver(state))
        raise RuntimeError("second owner failed")

    assert (
        veros_state_module.pure(
            first_owner,
            jitted=False,
            step=first_step,
            linear_solver=first_solver,
        )
        is first_copied_state
    )
    with pytest.raises(RuntimeError, match="second owner failed"):
        veros_state_module.pure(
            second_owner,
            jitted=False,
            step=failing_second_step,
            linear_solver=second_solver,
        )

    assert observed_solvers == [first_solver, second_solver]
    assert solver_cache == {(unrelated_state,): unrelated_solver}


@pytest.mark.parametrize("cache_mode", ("not_mapping", "different_mapping"))
def test_veros_component_solver_cache_requires_supported_native_interface(
    monkeypatch: pytest.MonkeyPatch,
    cache_mode: str,
) -> None:
    state = _FakeVerosStepState(counter=1)
    cache: Any = {} if cache_mode == "different_mapping" else object()
    wrapped_cache: Any = {} if cache_mode == "different_mapping" else cache

    monkeypatch.setattr(veros_get_linear_solver, "cache", cache)
    monkeypatch.setattr(veros_get_linear_solver.__wrapped__, "cache", wrapped_cache)

    with pytest.raises(
        RuntimeError,
        match="component-scoped Veros solver caching requires Veros >=1.6.2,<1.7",
    ):
        veros_state_module.get_component_linear_solver(state)


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
    import vercor.setups._external.veros_output as veros_output_module

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


def test_veros_output_provider_exposes_active_native_variable_universe() -> None:
    import vercor.setups._external.veros_output as veros_output_module

    state = _make_veros_output_state()
    state.variables.v = state.variables.u.copy()
    state.variables.w = state.variables.temp.copy()
    state.variables.surface_tauy = state.variables.surface_taux.copy()
    state.variables.line_psin = np.ones((6, 6), dtype=float)
    state.var_meta["line_psin"] = SimpleNamespace(
        active=True,
        dims=("isle", "isle"),
    )
    state.variables.Ai_ez = np.ones((6, 7, 2, 2, 2), dtype=float)
    state.var_meta["Ai_ez"] = SimpleNamespace(
        active=True,
        dims=("xt", "yt", "zt", "tensor1", "tensor2"),
    )
    state.variables.time = np.asarray(259_200.0, dtype=np.float32)
    state.var_meta["time"] = SimpleNamespace(active=True, dims=None)
    local_dimension_resolutions = 0

    def local_dimensions(settings: Any) -> tuple[str, ...]:
        nonlocal local_dimension_resolutions
        _ = settings
        local_dimension_resolutions += 1
        return ("xt", "yt", "months")

    state.variables.sss_clim = np.ones((6, 7, 12), dtype=float)
    state.var_meta["sss_clim"] = SimpleNamespace(
        active=True,
        dims=local_dimensions,
    )
    later_state = deepcopy(state)
    later_state.variables.temp = later_state.variables.temp + 1000.0
    provider = veros_output_module.veros_output_provider()

    first_frame = provider.sample(
        OutputContext(
            component=cast(Any, None),
            state=cast(Any, None),
            payload=state,
            step=0,
            time=datetime(2000, 1, 2),
            dt=timedelta(days=1),
        )
    )
    later_frame = provider.sample(
        OutputContext(
            component=cast(Any, None),
            state=cast(Any, None),
            payload=later_state,
            step=1,
            time=datetime(2000, 1, 3),
            dt=timedelta(days=1),
        )
    )

    assert tuple(first_frame.variables) == (
        "temp",
        "salt",
        "u",
        "v",
        "w",
        "eke",
        "tke",
        "rho",
        "surface_taux",
        "surface_tauy",
        "psi",
    )
    assert "sss_clim" not in first_frame.variables
    assert "line_psin" not in first_frame.variables
    assert_allclose_compact(
        later_frame.variables["temp"].values,
        first_frame.variables["temp"].values + 1000.0,
    )
    assert "Ai_ez" not in first_frame.variables
    assert "time" not in first_frame.variables
    assert "time" in first_frame.coordinates
    assert local_dimension_resolutions == 0
    assert first_frame.variables["temp"].dims == ("zt", "yt", "xt")


def test_veros_write_output_persists_period_mean_and_coordinates(
    tmp_path: Path,
) -> None:
    import vercor.setups._external.veros_output as veros_output_module

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
    frames = tuple(
        OutputFrame(
            snapshot,
            time_dimension=veros_output_module.VEROS_TIME_DIM,
        )
        for snapshot in snapshots
    )
    accumulator = _accumulate_frames(frames)
    temperature_index = accumulator.names.index("temp")
    assert isinstance(accumulator.sum_values[temperature_index], jax.Array)
    assert isinstance(accumulator.counts[temperature_index], jax.Array)
    output = tmp_path / "veros_output.nc"
    variables = _period_variables(
        accumulator,
        time_dim=veros_output_module.VEROS_TIME_DIM,
        reverse_value_dims=True,
    )

    write_netcdf_dataset(
        output=str(output),
        coordinate_variables=veros_output_module.veros_average_coordinate_variables(
            veros_state=state,
            output_time=datetime(2000, 1, 2),
            variables=variables,
        ),
        data_variables=variables,
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
    assert_allclose_compact(
        accumulator.counts[temperature_index],
        np.full((2, 3, 2), 2),
    )


def test_veros_snapshot_output_uses_native_state_payload(tmp_path: Path) -> None:
    state = _make_veros_output_state()
    component_state = _runtime_component_state(
        "OCN",
        {"temp": np.full((2, 3, 2), -999.0)},
    )
    output = tmp_path / "OCN.snapshot.nc"

    veros_output_module.write_veros_snapshot_output(
        SnapshotContext(
            component=cast(Any, None),
            state=ComponentState._from_runtime("OCN", None, component_state),
            payload=state,
            output_path=output,
            time=datetime(2000, 1, 2),
            logger=None,
        ),
        variables=("temp", "surface_taux"),
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


def test_veros_output_variables_rejects_bare_string() -> None:
    import vercor.setups._external.veros_output as veros_output_module

    with pytest.raises(ValueError, match="output_variables"):
        veros_output_module.normalize_veros_output_variables(
            "temp",
            settings=SimpleNamespace(enable_streamfunction=True),
        )


def test_veros_output_variables_rejects_setup_local_field() -> None:
    import vercor.setups._external.veros_output as veros_output_module

    with pytest.raises(
        ValueError,
        match="Unknown Veros output variable 'sss_clim'",
    ):
        veros_output_module.normalize_veros_output_variables(
            ("sss_clim",),
            settings=SimpleNamespace(),
        )


def test_apply_veros_forcing_fields_copies_once_and_updates_all_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variables = _FakeVariableStore(
        **{
            name: np.zeros((8, 8, 1), dtype=float)
            for name in ("taux", "tauy", "qnet", "qnec")
        }
    )
    state = SimpleNamespace(variables=variables)
    copy_calls: list[bool] = []

    def recording_copy(value: Any, jitted: bool = True) -> Any:
        copy_calls.append(jitted)
        return deepcopy(value)

    monkeypatch.setattr(veros_state_module, "copy_state", recording_copy)
    forcing = veros_state_module.VerosForcingFields(
        taux=jnp.ones((4, 4, 1)),
        tauy=jnp.full((4, 4, 1), 2.0),
        qnet=jnp.full((4, 4, 1), 3.0),
        qnec=jnp.full((4, 4, 1), 4.0),
    )

    result = veros_state_module.apply_veros_forcing_fields(
        state,
        forcing,
        jitted=True,
    )

    assert copy_calls == [True]
    assert not hasattr(veros_state_module, "set_variable")
    assert result is not state
    assert result.variables.unlock_calls == 1
    for name, expected in zip(
        ("taux", "tauy", "qnet", "qnec"),
        (1.0, 2.0, 3.0, 4.0),
        strict=True,
    ):
        result_value = getattr(result.variables, name)
        assert isinstance(result_value, np.ndarray)
        assert_allclose_compact(
            result_value[2:-2, 2:-2, :],
            np.full((4, 4, 1), expected),
        )
        assert np.count_nonzero(result_value[:2, :, :]) == 0
        assert np.count_nonzero(result_value[-2:, :, :]) == 0
        assert np.count_nonzero(result_value[:, :2, :]) == 0
        assert np.count_nonzero(result_value[:, -2:, :]) == 0
        assert np.count_nonzero(getattr(state.variables, name)) == 0


def test_veros_initialize_validates_timestep_multiple() -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.dt_tracer = 7.0

    with pytest.raises(ValueError, match="dt_tracer"):
        component.setup(
            cast(Any, component),
            _make_coupler(dt_seconds=20.0, run_order=["OCN"]),
        )


@pytest.mark.parametrize(("do_spinup", "expected_steps"), ((True, 2), (False, 0)))
def test_veros_initialize_spinup_follows_enabled_only(
    do_spinup: bool,
    expected_steps: int,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.dt_tracer = 10.0
    component.do_spinup = do_spinup
    component.spinup_time = timedelta(seconds=20.0)
    component.spinup_steps = 2
    component._veros_state = _make_fake_veros_state(surface_temperature=10.0)
    component.name = "CUSTOM_OCEAN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {}

    step_calls = {"count": 0}

    def fake_step_function(state: Any) -> Any:
        step_calls["count"] += 1
        return state

    component._step_function = fake_step_function

    hook_component = DataComponent(
        name="CUSTOM_OCEAN",
        grid=component.grid,
    )
    coupler = _make_coupler(dt_seconds=20.0, run_order=["CUSTOM_OCEAN"])
    setup_result = component.setup(cast(Any, hook_component), coupler)

    assert component.model_substeps == 2
    assert step_calls["count"] == expected_steps
    assert setup_result.payload is component._veros_state
    assert isinstance(setup_result.fields["sea_surface_temperature"], jax.Array)
    assert_allclose_compact(
        setup_result.fields["sea_surface_temperature"],
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
    monkeypatch.setattr(
        veros_runtime_settings_module,
        "configure_veros_runtime",
        lambda: None,
    )
    component_solver = object()
    solver_states: list[Any] = []

    def fake_get_component_linear_solver(veros_state: Any) -> Any:
        solver_states.append(veros_state)
        return component_solver

    monkeypatch.setattr(
        veros_state_module,
        "get_component_linear_solver",
        fake_get_component_linear_solver,
        raising=False,
    )

    component = veros_gcm_module.make_veros_gcm(
        config=VerosConfig(
            custom_parameters={"dt_tracer": 600.0},
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=("temp", "surface_taux"),
                ),
            ),
            jitted=False,
        ),
    )

    assert solver_states == [state]
    assert isinstance(component.grid.longitude, jax.Array)
    assert isinstance(component.grid.latitude, jax.Array)
    assert isinstance(component.grid.binary_mask, jax.Array)
    assert component.spec.inputs == (
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
    assert component.spec.outputs == ("sea_surface_temperature",)
    assert component.grid.binary_mask.shape == (4, 4)
    assert (
        component.spec.output.snapshot_writer
        is veros_output_module.write_veros_snapshot_output
    )
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
    monkeypatch.setattr(veros_state_module, "VerosState", _ConstructedVerosState)
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = restore_to_climatology
    component.model_substeps = 2
    component.jitted = False
    initial_native_state = _make_copyable_fake_veros_state(surface_temperature=12.0)
    component._veros_state = initial_native_state
    component.name = "OCN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {"sea_surface_temperature": np.zeros((4, 4), dtype=float)}

    def fake_step_function(state: Any) -> Any:
        state.variables.temp = np.full((8, 8, 1, 1), 15.0, dtype=float)
        return state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, constants, dtype: (
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [7.0, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, 12.0]]),
            np.asarray([[3.0, 4.0], [5.0, 6.0]]),
        ),
    )
    component._step_function = fake_step_function

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component_state = _runtime_component_state("OCN", component.data)
    step_context = StepContext(
        dt_seconds=20.0,
        time=datetime(2000, 1, 1),
        logger=coupler.logger,
    )
    result = veros_runtime_module.step_veros_runtime(
        component,
        component_state.fields.to_mapping(),
        step_context,
        initial_native_state,
    )

    assert isinstance(result, StepResult)
    assert result.payload is not initial_native_state
    assert component._veros_state is initial_native_state
    assert_allclose_compact(
        result.payload.variables.taux[2:-2, 2:-2, :],
        np.asarray([[[1.0], [3.0]], [[2.0], [4.0]]]),
    )
    assert_allclose_compact(
        result.payload.variables.tauy[2:-2, 2:-2, :],
        np.asarray([[[5.0], [7.0]], [[6.0], [8.0]]]),
    )
    assert_allclose_compact(
        result.payload.variables.qnet[2:-2, 2:-2, :],
        np.asarray([[[9.0], [11.0]], [[10.0], [12.0]]]),
    )
    assert_allclose_compact(result.payload.variables.qnec[2:-2, 2:-2, :], expected_qnec)
    for name in ("taux", "tauy", "qnet", "qnec"):
        assert np.count_nonzero(getattr(initial_native_state.variables, name)) == 0
    assert_allclose_compact(initial_native_state.variables.temp, 12.0)
    assert_allclose_compact(
        result.fields["sea_surface_temperature"],
        np.full((4, 4), 288.15),
    )
    assert isinstance(result.fields["sea_surface_temperature"], jax.Array)


def test_veros_step_requires_native_runtime_payload() -> None:
    resources = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    context = StepContext(dt_seconds=20.0, time=None, logger=None)

    with pytest.raises(ComponentError, match="native runtime payload"):
        veros_runtime_module.step_veros_runtime(resources, {}, context, None)


def test_veros_step_nan_cleans_forcing_fields_before_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = veros_gcm_state_module.VerosGCMSetupState.__new__(
        veros_gcm_state_module.VerosGCMSetupState
    )
    component.restore_to_climatology = True
    component.model_substeps = 0
    component.jitted = True
    component._veros_state = _make_fake_veros_state(surface_temperature=12.0)
    component.name = "OCN"
    component.grid = make_test_grid(
        name="ocn",
        longitude=np.arange(4.0),
        latitude=np.arange(4.0),
    )
    component.data = {"sea_surface_temperature": np.zeros((4, 4), dtype=float)}

    forcing_calls: list[veros_state_module.VerosForcingFields] = []

    def fake_apply_veros_forcing_fields(
        state: Any,
        forcing_fields: veros_state_module.VerosForcingFields,
        *,
        jitted: bool,
    ) -> Any:
        assert jitted is True
        assert all(isinstance(value, jax.Array) for value in forcing_fields)
        forcing_calls.append(forcing_fields)
        return state

    monkeypatch.setattr(
        veros_fluxes_module,
        "compute_fluxes",
        lambda veros_state, runtime_fields, constants, dtype: (
            np.asarray([[1.0, np.nan], [3.0, 4.0]]),
            np.asarray([[5.0, 6.0], [np.nan, 8.0]]),
            np.asarray([[9.0, 10.0], [11.0, np.nan]]),
            np.asarray([[12.0, 13.0], [14.0, np.nan]]),
        ),
    )
    monkeypatch.setattr(
        veros_state_module,
        "apply_veros_forcing_fields",
        fake_apply_veros_forcing_fields,
    )
    component._step_function = lambda state: state

    coupler = _make_coupler(dt_seconds=20.0, run_order=["ATM"])
    component_state = _runtime_component_state("OCN", component.data)
    step_context = StepContext(
        dt_seconds=20.0,
        time=datetime(2000, 1, 1),
        logger=coupler.logger,
    )
    _ = veros_runtime_module.step_veros_runtime(
        component,
        component_state.fields.to_mapping(),
        step_context,
        component._veros_state,
    )

    assert len(forcing_calls) == 1
    forcing_fields = forcing_calls[0]
    assert_allclose_compact(
        forcing_fields.taux,
        np.asarray([[[1.0], [3.0]], [[0.0], [4.0]]]),
    )
    assert_allclose_compact(
        forcing_fields.tauy,
        np.asarray([[[5.0], [0.0]], [[6.0], [8.0]]]),
    )
    assert_allclose_compact(
        forcing_fields.qnet,
        np.asarray([[[9.0], [11.0]], [[10.0], [0.0]]]),
    )
    assert_allclose_compact(
        forcing_fields.qnec,
        np.asarray([[[12.0], [14.0]], [[13.0], [0.0]]]),
    )
