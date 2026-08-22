from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.errors import JaxRuntimeError

from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.calendar import model_year_seconds, year_type_for_calendar
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    Component,
    TransferPolicy,
    LifecycleHooks,
    DataComponent,
    ComponentSpec,
    StepContext,
    SetupResult,
    StepResult,
)
from vercor.components.runtime_execution import step_component_runtime_state
from vercor.setups._data.era5_atmosphere import make_era5_atmosphere
from vercor.setups._data.era5_land import make_era5_land
from vercor.setups._data.era5_ocean import make_era5_ocean
from vercor.setups._data.erainterim_ocean import make_erainterim_ocean
from vercor.setups._data.jcm_land import make_jcm_land
from vercor.setups._external.jax_gcm_fields import (
    JAXGCM_INPUT_GRID_FIELD_NAMES,
    JAXGCM_OUTPUT_GRID_FIELD_NAMES,
)
import vercor.setups._external.jax_gcm_runtime as jax_gcm_runtime_module
import vercor.setups._external.jax_gcm_state as jax_gcm_state_module
from vercor.setups._external.jax_gcm_state import JCMState
from vercor.setups._slab.atmosphere import make_slab_atmosphere
from vercor.setups._slab.land import make_slab_land
from vercor.setups._slab.ocean import make_slab_ocean
from vercor.setups._slab.seaice import make_slab_seaice
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError, CouplerError
from vercor.exchanges import Exchange
from vercor.fields import _flatten_field_items
from vercor.forcing_index import daily_forcing_index
from vercor.grids import RectilinearGrid
from vercor.regridding import bilinear, conservative
from vercor.runtime import RuntimeOptions
from vercor._runtime.state import ComponentRuntimeState
from vercor.state import RunState
from vercor._runtime.stores import FieldStore
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_periodic_interval,
)
from tests._runtime_helpers import (
    create_runtime_state_from_coupler,
    replace_runtime_topology_maps,
    run_scanned_coupler,
)


class _IdentityRegridder:
    def __init__(
        self,
        source_grid: Any = None,
        target_grid: Any = None,
    ) -> None:
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = source_grid is target_grid

    def regrid(self, field: Any) -> Any:
        return jnp.asarray(field)

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        return jnp.asarray(u), jnp.asarray(v)


def _identity_factory(*args: Any, **kwargs: Any) -> _IdentityRegridder:
    _ = kwargs
    return _IdentityRegridder(*args)


class _FakeJCMForcing(NamedTuple):
    stl_am: jax.Array
    sea_surface_temperature: jax.Array

    def copy(self, **kwargs: Any) -> "_FakeJCMForcing":
        return _FakeJCMForcing(
            stl_am=kwargs.get("stl_am", self.stl_am),
            sea_surface_temperature=kwargs.get(
                "sea_surface_temperature",
                self.sea_surface_temperature,
            ),
        )


class _FakeSurfaceFlux(NamedTuple):
    shf: jax.Array
    evap: jax.Array
    rlds: jax.Array


class _FakeShortwaveRad(NamedTuple):
    rsns: jax.Array


class _FakeDynamicsPrediction(NamedTuple):
    normalized_surface_pressure: jax.Array
    u_wind: jax.Array
    v_wind: jax.Array
    temperature: jax.Array
    specific_humidity: jax.Array


class _FakePrediction(NamedTuple):
    physics: dict[str, Any]
    dynamics: _FakeDynamicsPrediction


class _JAXGCMFixture(NamedTuple):
    component: Component
    state: Any


def _make_data_component(
    component_type: Any,
    *,
    name: str,
    grid: RectilinearGrid,
    data: dict[str, jax.Array],
    receives: tuple[str, ...] = (),
    sends: tuple[str, ...] = (),
    transfer: TransferPolicy | None = None,
) -> Any:
    _ = component_type
    component = DataComponent(
        name=name,
        grid=grid,
        fields=data,
        spec=ComponentSpec(
            inputs=receives,
            outputs=sends,
            transfer=TransferPolicy() if transfer is None else transfer,
        ),
    )
    return component


def _fake_jcm_step(
    state: JCMState,
    forcing: _FakeJCMForcing,
) -> tuple[JCMState, _FakePrediction]:
    surface_temperature = forcing.stl_am + forcing.sea_surface_temperature
    temperature = jnp.stack(
        (surface_temperature + 1.0, surface_temperature + 2.0),
        axis=0,
    )
    humidity = jnp.stack(
        (
            jnp.full_like(surface_temperature, 10.0),
            jnp.full_like(surface_temperature, 20.0),
        ),
        axis=0,
    )
    wind_base = jnp.stack(
        (
            jnp.ones_like(surface_temperature),
            jnp.ones_like(surface_temperature) * 2.0,
        ),
        axis=0,
    )
    surface_flux = jnp.stack(
        (
            surface_temperature,
            surface_temperature * 0.5,
        ),
        axis=-1,
    )
    prediction = _FakePrediction(
        physics={
            "_surface_flux": _FakeSurfaceFlux(
                shf=surface_flux[jnp.newaxis, ...],
                evap=(surface_flux * 0.1)[jnp.newaxis, ...],
                rlds=(surface_temperature + 3.0)[jnp.newaxis, ...],
            ),
            "_shortwave_rad": _FakeShortwaveRad(
                rsns=(surface_temperature + 4.0)[jnp.newaxis, ...],
            ),
        },
        dynamics=_FakeDynamicsPrediction(
            normalized_surface_pressure=jnp.ones_like(surface_temperature)[
                jnp.newaxis, ...
            ],
            u_wind=wind_base[jnp.newaxis, ...],
            v_wind=(wind_base + 1.0)[jnp.newaxis, ...],
            temperature=temperature[jnp.newaxis, ...],
            specific_humidity=humidity[jnp.newaxis, ...],
        ),
    )
    updated_state = JCMState(
        dynamics=jax.tree_util.tree_map(lambda value: value[0], prediction.dynamics),
        physics=jax.tree_util.tree_map(lambda value: value[0], prediction.physics),
        dycore_state={
            "marker": state.dycore_state["marker"] + jnp.sum(surface_temperature)
        },
        physics_carry={
            "marker": state.physics_carry["marker"] + 2.0 * jnp.sum(surface_temperature)
        },
    )
    return updated_state, prediction


def _make_jax_gcm_fixture(grid: RectilinearGrid) -> _JAXGCMFixture:
    state = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    state.name = "ATM"
    state.grid = grid
    state._dtype_policy = DTypePolicy()
    state.data = {
        "sea_surface_temperature": jnp.full(grid.shape, 281.0, dtype=jnp.float64),
        "land_surface_temperature": jnp.full(grid.shape, 3.0, dtype=jnp.float64),
    }
    state.model = type(
        "_FakeJCMModel",
        (),
        {
            "terrain": type(
                "_FakeTerrain",
                (),
                {"fmask": jnp.full((grid.shape[1], grid.shape[0]), 0.5)},
            )()
        },
    )()
    state.sigma_levels = jnp.asarray([0.2, 1.0], dtype=jnp.float64)
    state._state = JCMState(
        dynamics=_FakeDynamicsPrediction(
            normalized_surface_pressure=jnp.ones(grid.shape, dtype=jnp.float64),
            u_wind=jnp.zeros((2, *grid.shape), dtype=jnp.float64),
            v_wind=jnp.zeros((2, *grid.shape), dtype=jnp.float64),
            temperature=jnp.zeros((2, *grid.shape), dtype=jnp.float64),
            specific_humidity=jnp.zeros((2, *grid.shape), dtype=jnp.float64),
        ),
        physics={
            "_surface_flux": _FakeSurfaceFlux(
                shf=jnp.zeros((*grid.shape, 2), dtype=jnp.float64),
                evap=jnp.zeros((*grid.shape, 2), dtype=jnp.float64),
                rlds=jnp.zeros(grid.shape, dtype=jnp.float64),
            ),
            "_shortwave_rad": _FakeShortwaveRad(
                rsns=jnp.zeros(grid.shape, dtype=jnp.float64),
            ),
        },
        dycore_state={"marker": jnp.asarray(0.0)},
        physics_carry={"marker": jnp.asarray(10.0)},
    )
    state.forcing = _FakeJCMForcing(
        stl_am=jnp.zeros((grid.shape[1], grid.shape[0]), dtype=jnp.float64),
        sea_surface_temperature=jnp.zeros(
            (grid.shape[1], grid.shape[0]),
            dtype=jnp.float64,
        ),
    )
    state._step_function = _fake_jcm_step
    component = CallableComponent(
        name="ATM",
        grid=grid,
        step=(
            lambda fields, context, payload: (
                jax_gcm_runtime_module.step_jax_gcm_component(
                    state,
                    fields,
                    context,
                    payload,
                )
            )
        ),
        spec=ComponentSpec(
            inputs=JAXGCM_INPUT_GRID_FIELD_NAMES,
            outputs=(
                "land_surface_temperature",
                "sea_surface_temperature",
                "total_surface_temperature",
                *JAXGCM_OUTPUT_GRID_FIELD_NAMES,
                "pressure",
            ),
            initial_fields={
                **{
                    field_name: 0.0
                    for field_name in (
                        jax_gcm_runtime_module.jax_gcm_default_field_names(
                            include_total_surface_temperature=True
                        )
                    )
                },
                **state.data,
                "pressure": jnp.zeros((state.sigma_levels.size, *grid.shape)),
            },
            lifecycle=LifecycleHooks(
                setup=(
                    lambda component, context: SetupResult(
                        payload=jax_gcm_runtime_module.create_jax_gcm_runtime_payload(
                            state
                        )
                    )
                ),
                prefill=(
                    lambda component, context: (
                        jax_gcm_runtime_module.prefill_jax_gcm_runtime_fields(
                            state,
                            component,
                            context,
                        )
                    )
                ),
                validate=(
                    lambda component, context: (
                        jax_gcm_runtime_module.validate_jax_gcm_runtime_state(
                            state,
                            component,
                            context,
                        )
                    )
                ),
            ),
        ),
    )
    return _JAXGCMFixture(component=component, state=state)


def _make_jax_gcm_component(grid: RectilinearGrid) -> Component:
    return _make_jax_gcm_fixture(grid).component


def _component_state(
    name: str,
    data: dict[str, jax.Array],
    receives: tuple[str, ...],
    sends: tuple[str, ...],
) -> ComponentRuntimeState:
    _ = name
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return ComponentRuntimeState(
        fields=FieldStore.from_mapping(
            {
                field: data.get(field, zeros)
                for field in sorted(set(data) | set(receives) | set(sends))
            }
        ),
        received=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in receives}
        ),
        sent=FieldStore.from_mapping(
            {field: data.get(field, zeros) for field in sends}
        ),
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="slab")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=(
            make_slab_atmosphere(grid),
            make_slab_ocean(grid),
            make_slab_land(grid),
            make_slab_seaice(grid),
        ),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="OCN",
                fields=["sensible_heat_flux", "latent_heat_flux"],
                regridder_factory=cast(Any, _identity_factory),
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=["latent_heat_flux"],
                regridder_factory=cast(Any, _identity_factory),
            ),
            Exchange(
                source="OCN",
                target="ICE",
                fields=["sea_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "ATM",
            "OCN",
            "LND",
            "ICE",
        ),
    )
    key = "OCN->ATM"
    regridders = cast(
        Any,
        {
            key: _IdentityRegridder(),
            "ATM->OCN": _IdentityRegridder(),
            "ATM->LND": _IdentityRegridder(),
            "OCN->ICE": _IdentityRegridder(),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={runtime_key: jnp.ones((2, 2)) for runtime_key in regridders},
    )
    return coupler


def _make_initialized_slab_coupler(steps: int) -> Coupler:
    longitude = np.asarray([0.0, 1.0], dtype=float)
    latitude = np.asarray([-1.0, 1.0], dtype=float)
    ocean_mask = np.ones((2, 2), dtype=float)
    land_mask = np.zeros((2, 2), dtype=float)

    atmosphere_grid = make_test_grid(
        name="ATM",
        longitude=longitude,
        latitude=latitude,
    )
    ocean_grid = make_test_grid(
        name="OCN",
        longitude=longitude,
        latitude=latitude,
        binary_mask=ocean_mask,
    )
    land_grid = make_test_grid(
        name="LND",
        longitude=longitude,
        latitude=latitude,
        binary_mask=land_mask,
    )
    ice_grid = make_test_grid(
        name="ICE",
        longitude=longitude,
        latitude=latitude,
    )

    components = (
        make_slab_atmosphere(atmosphere_grid),
        make_slab_ocean(ocean_grid),
        make_slab_land(land_grid),
        make_slab_seaice(ice_grid),
    )
    exchanges = (
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ICE",
            target="ATM",
            fields=["ice_fraction"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=["sensible_heat_flux", "latent_heat_flux"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=["latent_heat_flux"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ICE",
            fields=["sea_surface_temperature"],
            regridder_factory=bilinear,
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=components,
        exchanges=exchanges,
        run_order=("ATM", "OCN", "LND", "ICE"),
    )
    coupler._initialize_runtime()
    return coupler


def test_coupler_initialize_cascades_float32_precision_to_component_arrays() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)

    for component in coupler._ensure_prepared().components.values():
        assert component._dtype_policy.enable_x64 is False
        assert component.grid.longitude.dtype == jnp.float32
        assert component.grid.latitude.dtype == jnp.float32
        if component.grid.binary_mask is not None:
            assert component.grid.binary_mask.dtype == jnp.float32
        for field_value in component._data.values():
            assert jnp.asarray(field_value).dtype == jnp.float32


def _make_initialized_mixed_grid_slab_coupler(steps: int) -> Coupler:
    atmosphere_longitude = np.asarray([0.0, 1.0], dtype=float)
    atmosphere_latitude = np.asarray([-1.0, 1.0], dtype=float)
    atmosphere_longitude_edges = np.asarray([-0.25, 0.5, 1.25], dtype=float)
    atmosphere_latitude_edges = np.asarray([-1.5, 0.0, 1.5], dtype=float)
    ocean_longitude = np.asarray([0.0, 0.5, 1.0], dtype=float)
    ocean_latitude = np.asarray([-1.0, 0.0, 1.0], dtype=float)
    ocean_longitude_edges = np.asarray([-0.25, 0.25, 0.75, 1.25], dtype=float)
    ocean_latitude_edges = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype=float)
    ocean_mask = np.ones((3, 3), dtype=float)
    land_mask = np.zeros((2, 2), dtype=float)

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atmosphere_longitude,
        latitude=atmosphere_latitude,
        longitude_edges=atmosphere_longitude_edges,
        latitude_edges=atmosphere_latitude_edges,
    )
    ocean_grid = RectilinearGrid(
        name="OCN",
        longitude=ocean_longitude,
        latitude=ocean_latitude,
        longitude_edges=ocean_longitude_edges,
        latitude_edges=ocean_latitude_edges,
        binary_mask=ocean_mask,
    )
    land_grid = RectilinearGrid(
        name="LND",
        longitude=atmosphere_longitude,
        latitude=atmosphere_latitude,
        longitude_edges=atmosphere_longitude_edges,
        latitude_edges=atmosphere_latitude_edges,
        binary_mask=land_mask,
    )
    ice_grid = RectilinearGrid(
        name="ICE",
        longitude=ocean_longitude,
        latitude=ocean_latitude,
        longitude_edges=ocean_longitude_edges,
        latitude_edges=ocean_latitude_edges,
    )

    components = (
        make_slab_atmosphere(atmosphere_grid),
        make_slab_ocean(ocean_grid),
        make_slab_land(land_grid),
        make_slab_seaice(ice_grid),
    )
    exchanges = (
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regridder_factory=conservative,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ICE",
            target="ATM",
            fields=["ice_fraction"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=["sensible_heat_flux", "latent_heat_flux"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=["latent_heat_flux"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ICE",
            fields=["sea_surface_temperature"],
            regridder_factory=bilinear,
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=components,
        exchanges=exchanges,
        run_order=("ATM", "OCN", "LND", "ICE"),
    )
    coupler._initialize_runtime()
    return coupler


def _make_initial_state(
    coupler: Coupler,
    sea_surface_temperature: jax.Array,
) -> RunState:
    zeros = jnp.zeros_like(sea_surface_temperature)
    temperature_2m = jnp.full_like(sea_surface_temperature, 288.15)
    components = (
        _component_state(
            "ATM",
            {
                "temperature_2m": temperature_2m,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
                "u_velocity_10m": zeros,
                "v_velocity_10m": zeros,
                "sea_surface_temperature": sea_surface_temperature,
                "land_surface_temperature": temperature_2m,
                "soil_moisture": zeros,
                "ice_fraction": zeros,
            },
            receives=("sea_surface_temperature",),
            sends=(
                "temperature_2m",
                "sensible_heat_flux",
                "latent_heat_flux",
                "u_velocity_10m",
                "v_velocity_10m",
            ),
        ),
        _component_state(
            "OCN",
            {
                "sea_surface_temperature": sea_surface_temperature,
                "sensible_heat_flux": zeros,
                "latent_heat_flux": zeros,
                "u_velocity_10m": zeros,
                "v_velocity_10m": zeros,
                "u_velocity": zeros,
                "v_velocity": zeros,
                "specific_humidity": zeros,
                "temperature": zeros,
                "model_level_height": zeros,
                "net_shortwave_radiation_flux": zeros,
                "downward_longwave_radiation_flux": zeros,
                "ice_fraction": zeros,
            },
            receives=("sensible_heat_flux", "latent_heat_flux"),
            sends=("sea_surface_temperature",),
        ),
        _component_state(
            "LND",
            {
                "soil_moisture": jnp.full_like(sea_surface_temperature, 0.3),
                "land_surface_temperature": temperature_2m,
                "latent_heat_flux": zeros,
                "sensible_heat_flux": zeros,
            },
            receives=("latent_heat_flux",),
            sends=("soil_moisture", "land_surface_temperature"),
        ),
        _component_state(
            "ICE",
            {
                "ice_fraction": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            receives=("sea_surface_temperature",),
            sends=("ice_fraction",),
        ),
    )
    state = coupler.initial_state()
    for component_name, desired in zip(
        ("ATM", "OCN", "LND", "ICE"),
        components,
        strict=True,
    ):
        current = state._component_state(component_name)
        current = current.with_fields(
            current.fields.replace_many(
                {
                    name: desired.fields.get(name)
                    for name in current.fields.field_names
                    if name in desired.fields
                }
            )
        )
        current = current.with_received(
            current.received.replace_many(
                {
                    name: desired.received.get(name)
                    for name in current.received.field_names
                    if name in desired.received
                }
            )
        )
        current = current.with_sent(
            current.sent.replace_many(
                {
                    name: desired.sent.get(name)
                    for name in current.sent.field_names
                    if name in desired.sent
                }
            )
        )
        state = state._with_component_state(component_name, current)
    return state


def _with_ocean_sst(state: RunState, sea_surface_temperature: jax.Array) -> RunState:
    ocean = state._component_state("OCN")
    ocean = ocean.with_fields(
        ocean.fields.set("sea_surface_temperature", sea_surface_temperature)
    )
    ocean = ocean.with_sent(
        ocean.sent.set("sea_surface_temperature", sea_surface_temperature)
    )
    return state._with_component_state("OCN", ocean)


def _without_store_field(store: FieldStore, field_name: str) -> FieldStore:
    return FieldStore.from_mapping(
        {
            name: value
            for name, value in zip(store.field_names, store.values)
            if name != field_name
        }
    )


def test_run_supports_jit_grad_and_jvp() -> None:
    coupler = _make_coupler(steps=2)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    initial_state = _make_initial_state(coupler, initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    ocean_sst = final_state._component_state("OCN").fields.get(
        "sea_surface_temperature"
    )

    assert ocean_sst.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sst: jax.Array) -> jax.Array:
        state = _make_initial_state(coupler, sst)
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("OCN").fields.get("sea_surface_temperature")
        )

    tangent_seed = jnp.ones_like(initial_sst)
    gradient = jax.grad(loss)(initial_sst)
    _, forward_tangent = jax.jvp(loss, (initial_sst,), (tangent_seed,))
    value, pullback = jax.vjp(loss, initial_sst)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(forward_tangent))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_run_matches_one_step_closed_form_for_slab_ocean() -> None:
    coupler = _make_coupler(steps=1)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    final_state = run_scanned_coupler(
        coupler,
        _make_initial_state(coupler, initial_sst),
    )

    ocean_sst = final_state._component_state("OCN").fields.get(
        "sea_surface_temperature"
    )
    sensible = -10.0 * (288.15 - 286.15)
    latent = -0.5 * sensible
    restoring = (np.asarray(initial_sst) - 288.15) / (30.0 * 86400.0)
    expected = (
        np.asarray(initial_sst)
        + ((sensible + latent) / (1025.0 * 3990.0 * 30.0) + restoring) * 3600.0
    )

    assert_allclose_compact(ocean_sst, expected)


def test_initialized_slab_coupler_creates_jittable_runtime_state() -> None:
    coupler = _make_initialized_slab_coupler(steps=2)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    canonical_state = coupler.initial_state()
    runtime_state_copy = coupler.initial_state()
    assert tuple(runtime_state_copy.components()) == tuple(canonical_state.components())
    initial_state = _with_ocean_sst(canonical_state, initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    ocean_sst = final_state._component_state("OCN").fields.get(
        "sea_surface_temperature"
    )

    assert tuple(final_state.components()) == ("ATM", "OCN", "LND", "ICE")
    assert ocean_sst.shape == (2, 2)
    assert isinstance(ocean_sst, jax.Array)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        state = _with_ocean_sst(initial_state, sea_surface_temperature)
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("OCN").fields.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_initialized_slab_coupler_run_prefills_missing_imports() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)
    ocean = coupler._runtime_components["OCN"]

    final_state = coupler.run()
    ocean_state = final_state._component_state("OCN")

    assert tuple(final_state.components()) == ("ATM", "OCN", "LND", "ICE")
    assert ocean_state.received.get("sensible_heat_flux").shape == ocean.grid.shape
    assert ocean_state.received.get("latent_heat_flux").shape == ocean.grid.shape


def test_jcm_slab_ocean_exchange_recipe_prefills_required_flux_imports() -> None:
    from vercor.recipes import (
        ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
        JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
        JCM_LAND_TO_ATMOSPHERE_FIELDS,
        OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    )

    longitude = np.asarray([0.0, 1.0], dtype=float)
    latitude = np.asarray([-1.0, 1.0], dtype=float)
    ocean_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    land_mask = 1.0 - ocean_mask
    atmosphere_grid = make_test_grid(
        name="ATM",
        longitude=longitude,
        latitude=latitude,
    )
    ocean_grid = make_test_grid(
        name="OCN",
        longitude=longitude,
        latitude=latitude,
        binary_mask=ocean_mask,
    )
    land_grid = make_test_grid(
        name="LND",
        longitude=longitude,
        latitude=latitude,
        binary_mask=land_mask,
    )
    zeros = jnp.zeros(atmosphere_grid.shape)
    atmosphere_exports = tuple(
        _flatten_field_items(
            (
                *JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
                *ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
            )
        )
    )
    atmosphere = _make_data_component(
        DataComponent,
        name="ATM",
        grid=atmosphere_grid,
        data={field_name: zeros for field_name in atmosphere_exports},
        receives=tuple(
            _flatten_field_items(
                (
                    *OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                    *JCM_LAND_TO_ATMOSPHERE_FIELDS,
                )
            )
        ),
        sends=atmosphere_exports,
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(
            atmosphere,
            make_slab_ocean(ocean_grid),
            make_slab_land(land_grid),
        ),
        exchanges=(
            Exchange(
                source="ATM",
                target="OCN",
                fields=JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
        run_order=("OCN", "LND", "ATM"),
    )
    coupler._initialize_runtime()

    final_state = coupler.run()
    ocean_state = final_state._component_state("OCN")

    for field_name in (
        "sensible_heat_flux",
        "latent_heat_flux",
        "u_velocity",
        "v_velocity",
    ):
        assert ocean_state.received.get(field_name).shape == ocean_grid.shape
        assert ocean_state.fields.get(field_name).shape == ocean_grid.shape


def test_scanned_runtime_state_uses_runtime_field_stores() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)
    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)

    assert all(
        isinstance(component_state.fields, FieldStore)
        and isinstance(component_state.received, FieldStore)
        and isinstance(component_state.sent, FieldStore)
        for component_state in initial_state._components
    )

    final_state = run_scanned_coupler(coupler, initial_state)

    assert all(
        isinstance(component_state.fields, FieldStore)
        and isinstance(component_state.received, FieldStore)
        and isinstance(component_state.sent, FieldStore)
        for component_state in final_state._components
    )


def test_mixed_grid_slab_coupler_runs_with_real_regridders_under_jit_grad_and_jvp() -> (
    None
):
    coupler = _make_initialized_mixed_grid_slab_coupler(steps=2)
    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    initial_sst = jnp.linspace(285.15, 287.15, 9, dtype=jnp.float64).reshape((3, 3))
    initial_state = _with_ocean_sst(initial_state, initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )

    atmosphere = final_state._component_state("ATM")
    ocean = final_state._component_state("OCN")
    ice = final_state._component_state("ICE")
    atmosphere_sst = atmosphere.received.get("sea_surface_temperature")
    ocean_sst = ocean.fields.get("sea_surface_temperature")
    ice_sst = ice.received.get("sea_surface_temperature")

    assert atmosphere_sst.shape == (2, 2)
    assert ocean_sst.shape == (3, 3)
    assert ice_sst.shape == (3, 3)
    assert isinstance(atmosphere_sst, jax.Array)
    assert isinstance(ocean_sst, jax.Array)
    assert isinstance(ice_sst, jax.Array)
    assert np.all(np.isfinite(np.asarray(atmosphere_sst)))
    assert np.all(np.isfinite(np.asarray(ocean_sst)))
    assert np.all(np.isfinite(np.asarray(ice_sst)))

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        state = _with_ocean_sst(initial_state, sea_surface_temperature)
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("OCN").fields.get("sea_surface_temperature")
        )

    tangent_seed = jnp.ones_like(initial_sst)
    gradient = jax.grad(loss)(initial_sst)
    _, forward_tangent = jax.jvp(loss, (initial_sst,), (tangent_seed,))
    value, pullback = jax.vjp(loss, initial_sst)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(forward_tangent))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_data_forcing_components_run_inside_runtime() -> None:
    grid = make_test_grid(name="forcing")
    monthly_ocean = jnp.zeros((12, 2, 2), dtype=jnp.float64)
    monthly_ocean = monthly_ocean.at[0].set(
        jnp.asarray([[280.0, 281.0], [282.0, 283.0]], dtype=jnp.float64)
    )
    monthly_land = jnp.zeros((12, 2, 2), dtype=jnp.float64)
    monthly_land = monthly_land.at[0].set(
        jnp.asarray([[285.0, 286.0], [287.0, 288.0]], dtype=jnp.float64)
    )
    ocean = _make_data_component(
        make_era5_ocean,
        name="OCN",
        grid=grid,
        data={"sea_surface_temperature": monthly_ocean},
        sends=("sea_surface_temperature",),
        transfer=TransferPolicy("linear"),
    )
    land = _make_data_component(
        make_era5_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": monthly_land},
        sends=("land_surface_temperature",),
        transfer=TransferPolicy("linear"),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros((2, 2), dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros((2, 2), dtype=jnp.float64),
        },
        receives=("sea_surface_temperature", "land_surface_temperature"),
    )
    exchanges = (
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regridder_factory=cast(Any, _identity_factory),
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regridder_factory=cast(Any, _identity_factory),
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(ocean, land, atmosphere),
        exchanges=exchanges,
        run_order=(
            "OCN",
            "LND",
            "ATM",
        ),
    )
    regridders = cast(
        Any,
        {
            "OCN->ATM": _IdentityRegridder(),
            "LND->ATM": _IdentityRegridder(),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            key: jnp.ones(grid.shape, dtype=jnp.float64) for key in regridders
        },
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere = final_state._component_state("ATM")
    received_ocean = atmosphere.received.get("sea_surface_temperature")
    received_land = atmosphere.received.get("land_surface_temperature")
    expected_ocean = np.asarray(monthly_ocean[0])
    expected_land = np.asarray(monthly_land[0])

    assert received_ocean.shape == grid.shape
    assert received_land.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere.fields.field_names
    assert isinstance(received_ocean, jax.Array)
    assert isinstance(received_land, jax.Array)
    assert_allclose_compact(received_ocean, expected_ocean)
    assert_allclose_compact(received_land, expected_land)

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean = initial_state._component_state("OCN")
        state = initial_state._with_component_state(
            "OCN",
            ocean.with_fields(
                ocean.fields.set("sea_surface_temperature", ocean_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("sea_surface_temperature")
        )

    tangent_seed = jnp.ones_like(monthly_ocean)
    gradient = jax.grad(loss)(monthly_ocean)
    _, forward_tangent = jax.jvp(loss, (monthly_ocean,), (tangent_seed,))
    value, pullback = jax.vjp(loss, monthly_ocean)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == monthly_ocean.shape
    assert_allclose_compact(gradient[0], np.ones((2, 2)))
    assert_allclose_compact(gradient[1:], np.zeros((11, 2, 2)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_public_data_component_monthly_output_validates_and_sends_runtime_slice() -> (
    None
):
    grid = make_test_grid(name="public-data-forcing")
    monthly_ocean = jnp.zeros((12, *grid.shape), dtype=jnp.float64)
    first_month = jnp.asarray(
        [[280.0, 281.0], [282.0, 283.0]],
        dtype=jnp.float64,
    )
    monthly_ocean = monthly_ocean.at[0].set(first_month)
    ocean = DataComponent(
        name="OCN",
        grid=grid,
        fields={"sea_surface_temperature": monthly_ocean},
        spec=ComponentSpec(transfer=TransferPolicy("linear")),
    )
    atmosphere = make_slab_atmosphere(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(ocean, atmosphere),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "OCN",
            "ATM",
        ),
    )
    key = "OCN->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    ocean_state = initial_state._component_state("OCN")
    assert (
        ocean_state.fields.get("sea_surface_temperature").shape == monthly_ocean.shape
    )
    assert ocean_state.sent.get("sea_surface_temperature").shape == grid.shape
    assert_allclose_compact(
        ocean_state.sent.get("sea_surface_temperature"),
        np.asarray(first_month),
    )

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    received_sst = final_state._component_state("ATM").received.get(
        "sea_surface_temperature"
    )

    assert received_sst.shape == grid.shape
    assert_allclose_compact(received_sst, np.asarray(first_month))


def test_daily_data_forcing_sends_time_slice_to_slab_component_with_real_regridder() -> (
    None
):
    grid = make_test_grid(name="daily")
    forcing = jnp.zeros((365, 2, 2), dtype=jnp.float64)
    forcing = forcing.at[1].set(jnp.asarray([[286.0, 287.0], [288.0, 289.0]]))
    atmosphere = make_slab_atmosphere(grid)
    ocean = _make_data_component(
        make_era5_ocean,
        name="OCN",
        grid=grid,
        data={"sea_surface_temperature": forcing},
        sends=("sea_surface_temperature",),
        transfer=TransferPolicy("daily"),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 2), dt_seconds=3600.0, steps=1),
        components=(ocean, atmosphere),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=bilinear,
            ),
        ),
        run_order=(
            "OCN",
            "ATM",
        ),
    )
    key = "OCN->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: bilinear(grid, grid)}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_sst = atmosphere_state.received.get("sea_surface_temperature")
    sensible_heat_flux = atmosphere_state.fields.get("sensible_heat_flux")

    assert_allclose_compact(received_sst, np.asarray(forcing[1]))
    assert sensible_heat_flux.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(sensible_heat_flux)))


def test_erainterim_ocean_monthly_forcing_replays_to_slab_atmosphere_with_real_regridder() -> (
    None
):
    ocean_grid = make_test_grid(
        name="erainterim-ocean",
        longitude=np.asarray([0.0, 1.0, 2.0], dtype=float),
        latitude=np.asarray([-1.0, 1.0], dtype=float),
    )
    atmosphere_grid = make_test_grid(
        name="slab-atmosphere",
        longitude=np.asarray([0.5, 1.5], dtype=float),
        latitude=np.asarray([-1.0, 1.0], dtype=float),
    )
    monthly_ocean = jnp.zeros((12, 2, 3), dtype=jnp.float64)
    first_month = jnp.asarray(
        [[280.0, 282.0, 284.0], [281.0, 283.0, 285.0]],
        dtype=jnp.float64,
    )
    monthly_ocean = monthly_ocean.at[0].set(first_month)
    monthly_ocean = monthly_ocean.at[1].set(first_month + 12.0)
    atmosphere = make_slab_atmosphere(atmosphere_grid)
    ocean = _make_data_component(
        make_erainterim_ocean,
        name="OCN",
        grid=ocean_grid,
        data={"sea_surface_temperature": monthly_ocean},
        sends=("sea_surface_temperature",),
        transfer=TransferPolicy("linear"),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(ocean, atmosphere),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=bilinear,
            ),
        ),
        run_order=(
            "OCN",
            "ATM",
        ),
    )
    key = "OCN->ATM"
    regridder = bilinear(ocean_grid, atmosphere_grid)
    cast(Any, atmosphere)._data = {
        "temperature_2m": jnp.full(atmosphere_grid.shape, 288.15, dtype=jnp.float64),
        "sensible_heat_flux": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "latent_heat_flux": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "u_velocity_10m": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "v_velocity_10m": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "sea_surface_temperature": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
    }
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: regridder}),
        fractional_masks={key: jnp.ones(atmosphere_grid.shape)},
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_sst = atmosphere_state.received.get("sea_surface_temperature")
    sensible_heat_flux = atmosphere_state.fields.get("sensible_heat_flux")
    expected_source = np.asarray(first_month)
    expected_received = regridder.regrid(expected_source)

    assert received_sst.shape == atmosphere_grid.shape
    assert sensible_heat_flux.shape == atmosphere_grid.shape
    assert isinstance(received_sst, jax.Array)
    assert isinstance(sensible_heat_flux, jax.Array)
    assert_allclose_compact(received_sst, expected_received)
    assert np.all(np.isfinite(np.asarray(sensible_heat_flux)))

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state._component_state("OCN")
        state = initial_state._with_component_state(
            "OCN",
            ocean_state.with_fields(
                ocean_state.fields.set("sea_surface_temperature", ocean_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("sea_surface_temperature")
        )

    tangent_seed = jnp.ones_like(monthly_ocean)
    gradient = jax.grad(loss)(monthly_ocean)
    _, forward_tangent = jax.jvp(loss, (monthly_ocean,), (tangent_seed,))
    value, pullback = jax.vjp(loss, monthly_ocean)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == monthly_ocean.shape
    assert np.all(np.isfinite(np.asarray(gradient[0])))
    assert_allclose_compact(gradient[1:], np.zeros((11, 2, 3)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_jcm_land_daily_forcing_replays_to_data_atmosphere_under_jit_and_grad() -> None:
    grid = make_test_grid(name="jcm-land")
    forcing = jnp.zeros((365, 2, 2), dtype=jnp.float64)
    forcing = forcing.at[2].set(jnp.asarray([[286.0, 287.0], [288.0, 289.0]]))
    land = _make_data_component(
        make_jcm_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": forcing},
        sends=("land_surface_temperature",),
        transfer=TransferPolicy("daily"),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        receives=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1),
        components=(land, atmosphere),
        exchanges=(
            Exchange(
                source="LND",
                target="ATM",
                fields=["land_surface_temperature"],
                regridder_factory=bilinear,
            ),
        ),
        run_order=(
            "LND",
            "ATM",
        ),
    )
    key = "LND->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: bilinear(grid, grid)}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_temperature = atmosphere_state.received.get("land_surface_temperature")

    assert received_temperature.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere_state.fields.field_names
    assert isinstance(received_temperature, jax.Array)
    assert_allclose_compact(received_temperature, np.asarray(forcing[2]))

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state._component_state("LND")
        state = initial_state._with_component_state(
            "LND",
            land_state.with_fields(
                land_state.fields.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("land_surface_temperature")
        )

    tangent_seed = jnp.ones_like(forcing)
    gradient = jax.grad(loss)(forcing)
    _, forward_tangent = jax.jvp(loss, (forcing,), (tangent_seed,))
    value, pullback = jax.vjp(loss, forcing)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == forcing.shape
    assert_allclose_compact(gradient[2], np.ones(grid.shape))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((362, 2, 2)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_noleap_daily_forcing_replays_calendar_slice_under_jit_and_grad() -> None:
    grid = make_test_grid(name="noleap-daily")
    forcing = jnp.arange(365 * 2 * 2, dtype=jnp.float64).reshape((365, 2, 2))
    land = _make_data_component(
        make_jcm_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": forcing},
        sends=("land_surface_temperature",),
        transfer=TransferPolicy("daily"),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        receives=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(
            start=datetime(2024, 3, 1),
            dt_seconds=3600.0,
            steps=1,
            calendar="noleap",
        ),
        components=(land, atmosphere),
        exchanges=(
            Exchange(
                source="LND",
                target="ATM",
                fields=["land_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "LND",
            "ATM",
        ),
    )
    key = "LND->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_temperature = atmosphere_state.received.get("land_surface_temperature")

    assert "total_surface_temperature" not in atmosphere_state.fields.field_names
    assert_allclose_compact(received_temperature, np.asarray(forcing[59]))

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state._component_state("LND")
        state = initial_state._with_component_state(
            "LND",
            land_state.with_fields(
                land_state.fields.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("land_surface_temperature")
        )

    tangent_seed = jnp.ones_like(forcing)
    gradient = jax.grad(loss)(forcing)
    _, forward_tangent = jax.jvp(loss, (forcing,), (tangent_seed,))
    value, pullback = jax.vjp(loss, forcing)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert_allclose_compact(gradient[59], np.ones(grid.shape))
    assert_allclose_compact(gradient[:59], np.zeros((59, 2, 2)))
    assert_allclose_compact(gradient[60:], np.zeros((305, 2, 2)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_360_day_daily_forcing_matches_host_calendar_mapping_under_jit_and_grad() -> (
    None
):
    grid = make_test_grid(name="360-daily")
    forcing = jnp.arange(365 * 2 * 2, dtype=jnp.float64).reshape((365, 2, 2))
    land = _make_data_component(
        make_jcm_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": forcing},
        sends=("land_surface_temperature",),
        transfer=TransferPolicy("daily"),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        receives=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(
            start=datetime(2001, 2, 28),
            dt_seconds=3600.0,
            steps=1,
            calendar="360_day",
        ),
        components=(land, atmosphere),
        exchanges=(
            Exchange(
                source="LND",
                target="ATM",
                fields=["land_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "LND",
            "ATM",
        ),
    )
    key = "LND->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    _, runtime_time, _ = next(coupler.clock.iter())
    expected_index = daily_forcing_index(
        runtime_time,
        year_type=year_type_for_calendar(
            coupler.clock.calendar,
            runtime_time.year,
        ),
        no_leap=True,
    )
    expected_slice = forcing[expected_index]

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_temperature = atmosphere_state.received.get("land_surface_temperature")

    assert expected_index == 56
    assert_allclose_compact(expected_slice, np.asarray(forcing[56]))
    assert "total_surface_temperature" not in atmosphere_state.fields.field_names
    assert_allclose_compact(received_temperature, expected_slice)

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state._component_state("LND")
        state = initial_state._with_component_state(
            "LND",
            land_state.with_fields(
                land_state.fields.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("land_surface_temperature")
        )

    tangent_seed = jnp.ones_like(forcing)
    gradient = jax.grad(loss)(forcing)
    _, forward_tangent = jax.jvp(loss, (forcing,), (tangent_seed,))
    value, pullback = jax.vjp(loss, forcing)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert_allclose_compact(gradient[56], np.ones(grid.shape))
    assert_allclose_compact(gradient[:56], np.zeros((56, 2, 2)))
    assert_allclose_compact(gradient[57:], np.zeros((308, 2, 2)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_monthly_forcing_wraps_year_boundary_under_jit_and_grad() -> None:
    grid = make_test_grid(name="monthly-wrap")
    monthly_ocean = jnp.zeros((12, 2, 2), dtype=jnp.float64)
    month_zero = jnp.asarray([[280.0, 281.0], [282.0, 283.0]], dtype=jnp.float64)
    month_eleven = jnp.asarray([[292.0, 293.0], [294.0, 295.0]], dtype=jnp.float64)
    monthly_ocean = monthly_ocean.at[0].set(month_zero)
    monthly_ocean = monthly_ocean.at[11].set(month_eleven)
    ocean = _make_data_component(
        make_era5_ocean,
        name="OCN",
        grid=grid,
        data={"sea_surface_temperature": monthly_ocean},
        sends=("sea_surface_temperature",),
        transfer=TransferPolicy("linear"),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        receives=("sea_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2001, 12, 31, 12), dt_seconds=3600.0, steps=1),
        components=(ocean, atmosphere),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "OCN",
            "ATM",
        ),
    )
    key = "OCN->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    year_type = year_type_for_calendar(
        coupler.clock.calendar,
        coupler.clock.start.year,
    )
    year_seconds = model_year_seconds(year_type)
    (left_index, left_weight), (right_index, right_weight) = get_periodic_interval(
        current_time=datetime_to_seconds_in_year(coupler.clock.start),
        cycle_length=year_seconds,
        rec_spacing=year_seconds / 12.0,
        n_rec=12,
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    received_ocean = atmosphere_state.received.get("sea_surface_temperature")
    expected = np.asarray(
        left_weight * monthly_ocean[left_index]
        + right_weight * monthly_ocean[right_index]
    )

    assert (left_index, right_index) == (11, 0)
    assert received_ocean.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere_state.fields.field_names
    assert_allclose_compact(received_ocean, expected)

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state._component_state("OCN")
        state = initial_state._with_component_state(
            "OCN",
            ocean_state.with_fields(
                ocean_state.fields.set("sea_surface_temperature", ocean_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result._component_state("ATM").received.get("sea_surface_temperature")
        )

    tangent_seed = jnp.ones_like(monthly_ocean)
    gradient = jax.grad(loss)(monthly_ocean)
    _, forward_tangent = jax.jvp(loss, (monthly_ocean,), (tangent_seed,))
    value, pullback = jax.vjp(loss, monthly_ocean)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert_allclose_compact(gradient[0], np.full(grid.shape, right_weight))
    assert_allclose_compact(gradient[11], np.full(grid.shape, left_weight))
    assert_allclose_compact(gradient[1:11], np.zeros((10, 2, 2)))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


@pytest.mark.filterwarnings(
    "error:scatter inputs have incompatible types:FutureWarning"
)
def test_jax_gcm_runs_inside_runtime_under_jit_and_grad() -> None:
    jax.config.update("jax_enable_x64", True)
    grid = make_test_grid(name="jcm-runtime")
    fixture = _make_jax_gcm_fixture(grid)
    component = fixture.component
    original_state = fixture.state._state
    original_forcing = fixture.state.forcing
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    initial_payload = initial_state._component_state("ATM").payload
    assert initial_payload is not None
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    temperature = atmosphere_state.fields.get("temperature")
    sensible_heat_flux = atmosphere_state.fields.get("sensible_heat_flux")

    assert fixture.state._state is original_state
    assert fixture.state.forcing is original_forcing
    assert temperature.shape == grid.shape
    assert sensible_heat_flux.shape == grid.shape
    assert isinstance(temperature, jax.Array)
    assert isinstance(sensible_heat_flux, jax.Array)
    assert np.all(np.isfinite(np.asarray(temperature)))
    assert np.all(np.isfinite(np.asarray(sensible_heat_flux)))
    mapped_float_dtypes = {
        atmosphere_state.fields.get(field_name).dtype
        for field_name in (*JAXGCM_OUTPUT_GRID_FIELD_NAMES, "pressure")
    }
    assert mapped_float_dtypes == {jnp.dtype(jnp.float32)}
    assert atmosphere_state.payload is not None
    initial_float_dtypes = tuple(
        jnp.asarray(leaf).dtype
        for leaf in jax.tree_util.tree_leaves(initial_payload)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    )
    final_float_dtypes = tuple(
        jnp.asarray(leaf).dtype
        for leaf in jax.tree_util.tree_leaves(atmosphere_state.payload)
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
    )
    assert set(initial_float_dtypes) == {jnp.dtype(jnp.float32)}
    assert final_float_dtypes == initial_float_dtypes
    assert float(atmosphere_state.payload.jcm_state.dycore_state["marker"]) != float(
        initial_payload.jcm_state.dycore_state["marker"]
    )
    assert float(atmosphere_state.payload.jcm_state.physics_carry["marker"]) != float(
        initial_payload.jcm_state.physics_carry["marker"]
    )

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        atmosphere = initial_state._component_state("ATM")
        state = initial_state._with_component_state(
            "ATM",
            atmosphere.with_fields(
                atmosphere.fields.set(
                    "sea_surface_temperature",
                    sea_surface_temperature,
                )
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(result._component_state("ATM").fields.get("temperature"))

    sea_surface_temperature = jnp.full(grid.shape, 281.0, dtype=jnp.float32)
    tangent_seed = jnp.ones_like(sea_surface_temperature)
    gradient = jax.grad(loss)(sea_surface_temperature)
    _, forward_tangent = jax.jvp(
        loss,
        (sea_surface_temperature,),
        (tangent_seed,),
    )
    value, pullback = jax.vjp(loss, sea_surface_temperature)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_jax_gcm_runtime_keeps_time_dependent_forcing_payload_shape_stable() -> None:
    grid = make_test_grid(name="jcm-runtime-forcing-template")
    fixture = _make_jax_gcm_fixture(grid)
    component = fixture.component
    forcing_template = _FakeJCMForcing(
        stl_am=jnp.zeros((*grid.shape, 365), dtype=jnp.float64),
        sea_surface_temperature=jnp.zeros((*grid.shape, 365), dtype=jnp.float64),
    )
    fixture.state.forcing = forcing_template
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    payload = atmosphere_state.payload

    assert atmosphere_state.fields.get("temperature").shape == grid.shape
    assert payload is not None
    assert payload.forcing.stl_am.shape == forcing_template.stl_am.shape
    assert (
        payload.forcing.sea_surface_temperature.shape
        == forcing_template.sea_surface_temperature.shape
    )


def test_real_jax_gcm_runtime_seeds_and_advances_jcm_2_carry(
    fast_mode: bool,
) -> None:
    if fast_mode:
        pytest.skip("Real JCM carry regression runs outside --fast")

    from vercor.setups import JAXGCMConfig, load_jcm_inputs, make_jax_gcm

    inputs = load_jcm_inputs()
    component = make_jax_gcm(
        inputs.coords,
        inputs.terrain,
        config=JAXGCMConfig(forcing_data=inputs.forcing, jitted=True),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 86400.0, 2),
        components=(component,),
        run_order=("ATM",),
        runtime=RuntimeOptions(
            dtype=DTypePolicy.from_jax_config(),
            topology=None,
        ),
    )
    initial_state = coupler.initial_state()
    setup_hook = component.spec.lifecycle.setup
    assert setup_hook is not None
    setup_state = cast(Any, setup_hook).__self__
    terrain_mask = np.asarray(setup_state.model.terrain.fmask)
    assert np.any((terrain_mask > 0.0) & (terrain_mask < 1.0))
    initial_payload = initial_state._component_state("ATM").payload
    assert initial_payload is not None
    initial_carry = initial_payload.jcm_state.physics_carry
    assert initial_payload.jcm_state.dycore_state is not None
    assert initial_carry is not None
    assert cast(Any, jax.tree_util.tree_structure(initial_carry)) == (
        jax.tree_util.tree_structure(setup_state.model._final_physics_state)
    )
    initial_float_leaves = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(initial_carry)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert all(np.all(np.isfinite(leaf)) for leaf in initial_float_leaves)

    final_state = run_scanned_coupler(coupler, initial_state)
    final_payload = final_state._component_state("ATM").payload
    assert final_payload is not None
    final_carry = final_payload.jcm_state.physics_carry
    final_float_leaves = [
        np.asarray(leaf)
        for leaf in jax.tree_util.tree_leaves(final_carry)
        if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact)
    ]
    assert cast(Any, jax.tree_util.tree_structure(final_carry)) == (
        jax.tree_util.tree_structure(initial_carry)
    )
    assert len(final_float_leaves) == len(initial_float_leaves)
    assert all(np.all(np.isfinite(leaf)) for leaf in final_float_leaves)
    assert any(
        not np.allclose(initial, final)
        for initial, final in zip(initial_float_leaves, final_float_leaves)
    )
    temperature = final_state._component_state("ATM").fields.get("temperature")
    assert np.all(np.isfinite(np.asarray(temperature)))

    native_surface_temperature = jnp.full(
        (component.grid.shape[1], component.grid.shape[0]),
        288.0,
        dtype=jnp.float64,
    )
    forcing = initial_payload.forcing.copy(
        stl_am=native_surface_temperature,
        sea_surface_temperature=native_surface_temperature,
    )

    def real_jcm_loss(sea_surface_temperature: jax.Array) -> jax.Array:
        next_state, _ = setup_state._step_function(
            initial_payload.jcm_state,
            forcing.copy(sea_surface_temperature=sea_surface_temperature),
        )
        return jnp.sum(next_state.dynamics.temperature)

    loss_value, reverse_gradient = jax.value_and_grad(real_jcm_loss)(
        native_surface_temperature
    )
    _, forward_tangent = jax.jvp(
        real_jcm_loss,
        (native_surface_temperature,),
        (jnp.ones_like(native_surface_temperature),),
    )

    assert np.isfinite(np.asarray(loss_value))
    assert reverse_gradient.shape == native_surface_temperature.shape
    assert np.all(np.isfinite(np.asarray(reverse_gradient)))
    assert np.isfinite(np.asarray(forward_tangent))
    assert np.any(np.asarray(reverse_gradient) != 0.0)
    assert float(jnp.abs(forward_tangent)) > 0.0
    assert_allclose_compact(
        forward_tangent,
        jnp.sum(reverse_gradient),
        rtol=1e-6,
        atol=1e-10,
        equal_nan=False,
        label="real JCM forward/reverse derivative",
    )

    def full_coupler_loss(surface_temperature: jax.Array) -> jax.Array:
        state = initial_state.replace_fields(
            "ATM",
            {
                "sea_surface_temperature": jnp.full(
                    component.grid.shape,
                    surface_temperature,
                )
            },
        )
        result = coupler.run(state)
        return jnp.mean(result.component("ATM").field("temperature"))

    coupled_surface_temperature = jnp.asarray(288.0, dtype=jnp.float64)
    coupled_loss, coupled_reverse = jax.value_and_grad(full_coupler_loss)(
        coupled_surface_temperature
    )
    tangent_seed = jnp.ones_like(coupled_surface_temperature)
    value, pullback = jax.vjp(
        full_coupler_loss,
        coupled_surface_temperature,
    )
    (reverse_vjp,) = pullback(jnp.ones_like(value))
    _, forward_tangent = jax.jvp(
        full_coupler_loss,
        (coupled_surface_temperature,),
        (tangent_seed,),
    )

    assert np.isfinite(np.asarray(coupled_loss))
    assert np.isfinite(np.asarray(coupled_reverse))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert np.isfinite(np.asarray(forward_tangent))
    assert float(jnp.abs(coupled_reverse)) > 0.0
    assert_allclose_compact(
        forward_tangent,
        coupled_reverse,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        label="full Coupler real JCM forward/reverse derivative",
    )
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
        label="full Coupler real JCM JVP/VJP inner product",
    )


def test_data_forcing_replays_into_jax_gcm_runtime_under_jit_grad_and_jvp() -> None:
    grid = make_test_grid(name="data-jcm-runtime")
    sea_surface_temperature = jnp.asarray(
        [[280.0, 281.0], [282.0, 283.0]], dtype=jnp.float64
    )
    ocean = _make_data_component(
        make_era5_ocean,
        name="OCN",
        grid=grid,
        data={"sea_surface_temperature": sea_surface_temperature},
        sends=("sea_surface_temperature",),
    )
    atmosphere = _make_jax_gcm_component(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(ocean, atmosphere),
        exchanges=(
            Exchange(
                source="OCN",
                target="ATM",
                fields=["sea_surface_temperature"],
                regridder_factory=cast(Any, _identity_factory),
            ),
        ),
        run_order=(
            "OCN",
            "ATM",
        ),
    )
    key = "OCN->ATM"
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = create_runtime_state_from_coupler(coupler, prefill_missing=True)
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state._component_state("ATM")
    temperature = atmosphere_state.fields.get("temperature")
    received_sst = atmosphere_state.received.get("sea_surface_temperature")

    assert temperature.shape == grid.shape
    assert received_sst.shape == grid.shape
    assert isinstance(temperature, jax.Array)
    assert isinstance(received_sst, jax.Array)
    assert_allclose_compact(received_sst, sea_surface_temperature)
    assert np.all(np.isfinite(np.asarray(temperature)))

    def loss(forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state._component_state("OCN")
        state = initial_state._with_component_state(
            "OCN",
            ocean_state.with_fields(
                ocean_state.fields.set("sea_surface_temperature", forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(result._component_state("ATM").fields.get("temperature"))

    tangent_seed = jnp.ones_like(sea_surface_temperature)
    gradient = jax.grad(loss)(sea_surface_temperature)
    _, forward_tangent = jax.jvp(
        loss,
        (sea_surface_temperature,),
        (tangent_seed,),
    )
    value, pullback = jax.vjp(loss, sea_surface_temperature)
    (reverse_vjp,) = pullback(jnp.ones_like(value))

    assert gradient.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
    assert np.isfinite(np.asarray(forward_tangent))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert_allclose_compact(
        jnp.vdot(tangent_seed, reverse_vjp),
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )


def test_jax_gcm_runtime_requires_initialized_payload() -> None:
    grid = make_test_grid(name="jcm-uninitialized")
    fixture = _make_jax_gcm_fixture(grid)
    component = fixture.component
    del fixture.state._state
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )

    with pytest.raises(ComponentError, match="component initialization"):
        run_scanned_coupler(coupler)

    component = _make_jax_gcm_component(grid)
    payload_coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )
    state = payload_coupler.initial_state()
    atmosphere = state._component_state("ATM").with_payload(None)
    state = state._with_component_state("ATM", atmosphere)

    with pytest.raises(CouplerError, match="runtime payload.*PyTree structure"):
        run_scanned_coupler(payload_coupler, state)


def test_run_accepts_empty_constructor_configuration() -> None:
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )

    final_state = run_scanned_coupler(coupler)

    assert tuple(final_state.components()) == ()


def test_run_accepts_default_runtime_component() -> None:
    grid = make_test_grid(name="dummy")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(
            cast(
                Any,
                DataComponent(
                    name="ATM",
                    grid=grid,
                    fields={"temperature": 0.0},
                ),
            ),
        ),
        run_order=("ATM",),
    )

    final_state = run_scanned_coupler(coupler)

    assert tuple(final_state.components()) == ("ATM",)


def test_scanned_runtime_rejects_camulator_land_runtime_boundary() -> None:
    grid = make_test_grid(name="camulator")
    camulator_land = CallableComponent(
        name="LND",
        grid=grid,
        step=lambda fields, context, payload: {},
        spec=ComponentSpec(
            outputs=("land_surface_temperature",),
            initial_fields={
                "land_surface_temperature": jnp.zeros(
                    grid.shape,
                    dtype=jnp.float64,
                )
            },
            execution="host",
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(cast(Any, camulator_land),),
        run_order=("LND",),
    )

    state = coupler.initial_state()
    assert isinstance(
        state.component("LND").field("land_surface_temperature"), jax.Array
    )

    with pytest.raises(ComponentError, match="backend='jax'.*host-backed"):
        run_scanned_coupler(coupler, state)


def test_scanned_runtime_rejects_camulator_gcm_runtime_boundary() -> None:
    grid = make_test_grid(name="camulator-gcm")
    camulator = CallableComponent(
        name="ATM",
        grid=grid,
        step=lambda fields, context, payload: {},
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": jnp.ones(grid.shape, dtype=jnp.float64)},
            execution="host",
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(cast(Any, camulator),),
        run_order=("ATM",),
    )

    state = coupler.initial_state()
    assert isinstance(state.component("ATM").field("temperature"), jax.Array)

    with pytest.raises(ComponentError, match="backend='jax'.*host-backed"):
        run_scanned_coupler(coupler, state)


def test_scanned_runtime_rejects_veros_runtime_boundary() -> None:
    grid = make_test_grid(name="veros")
    veros = CallableComponent(
        name="OCN",
        grid=grid,
        step=lambda fields, context, payload: {},
        spec=ComponentSpec(
            outputs=("sea_surface_temperature",),
            initial_fields={
                "sea_surface_temperature": jnp.zeros(
                    grid.shape,
                    dtype=jnp.float64,
                )
            },
            execution="host",
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(cast(Any, veros),),
        run_order=("OCN",),
    )

    state = coupler.initial_state()
    assert isinstance(
        state.component("OCN").field("sea_surface_temperature"), jax.Array
    )

    with pytest.raises(ComponentError, match="backend='jax'.*host-backed"):
        run_scanned_coupler(coupler, state)


def test_run_validates_regridders_and_fractional_masks() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    replace_runtime_topology_maps(coupler, regridders={})

    with pytest.raises(CouplerError, match="initialized regridder"):
        run_scanned_coupler(coupler, state)

    coupler = _make_coupler(steps=1)
    state = RunState._from_runtime(
        component_names=state._component_names,
        component_grids=state._component_grids,
        components=state._components,
        fractional_masks=FieldStore.empty(),
    )

    with pytest.raises(CouplerError, match="fractional mask"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_source_fields_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    ocean = state._component_state("OCN").with_sent(FieldStore.empty())
    state = state._with_component_state("OCN", ocean)

    with pytest.raises(CouplerError, match="runtime sent names"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_slab_required_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    atmosphere = state._component_state("ATM")
    atmosphere = atmosphere.with_fields(
        _without_store_field(atmosphere.fields, "temperature_2m")
    )
    state = state._with_component_state("ATM", atmosphere)

    with pytest.raises(CouplerError, match="runtime fields names.*temperature_2m"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_import_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    atmosphere = state._component_state("ATM")
    atmosphere = atmosphere.with_fields(
        _without_store_field(atmosphere.fields, "sea_surface_temperature")
    )
    state = state._with_component_state("ATM", atmosphere)

    with pytest.raises(
        CouplerError, match="runtime fields names.*sea_surface_temperature"
    ):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_export_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    ocean = state._component_state("OCN")
    ocean = ocean.with_fields(
        _without_store_field(ocean.fields, "sea_surface_temperature")
    )
    state = state._with_component_state("OCN", ocean)

    with pytest.raises(
        CouplerError, match="runtime fields names.*sea_surface_temperature"
    ):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_jax_gcm_preseed_before_scan() -> None:
    grid = make_test_grid(name="jcm-missing-preseed")
    component = _make_jax_gcm_component(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("ATM",),
    )
    state = coupler.initial_state()
    atmosphere = state._component_state("ATM")
    atmosphere = atmosphere.with_fields(
        _without_store_field(atmosphere.fields, "pressure")
    )
    state = state._with_component_state("ATM", atmosphere)

    with pytest.raises(CouplerError, match="runtime fields names.*pressure"):
        run_scanned_coupler(coupler, state)


def test_run_validates_fractional_mask_shape_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(coupler, jnp.full((2, 2), 286.15, dtype=jnp.float64))
    state = RunState._from_runtime(
        component_names=state._component_names,
        component_grids=state._component_grids,
        components=state._components,
        fractional_masks=FieldStore.from_mapping(
            {
                "OCN->ATM": jnp.ones((1, 1), dtype=jnp.float64),
                "ATM->OCN": jnp.ones((2, 2), dtype=jnp.float64),
                "ATM->LND": jnp.ones((2, 2), dtype=jnp.float64),
                "OCN->ICE": jnp.ones((2, 2), dtype=jnp.float64),
            }
        ),
    )

    with pytest.raises(CouplerError, match="fractional mask.*shape"):
        run_scanned_coupler(coupler, state)


@pytest.mark.fast_always
def test_field_store_replacements_reject_shape_changes() -> None:
    store = FieldStore.from_mapping(
        {
            "temperature": jnp.zeros((2, 2)),
            "humidity": jnp.ones((2, 2)),
        }
    )

    with pytest.raises(
        ValueError,
        match=r"temperature.*shape \(1, 2\).*existing shape \(2, 2\)",
    ):
        store.replace("temperature", jnp.ones((1, 2)))
    with pytest.raises(
        ValueError,
        match=r"humidity.*shape \(2, 1\).*existing shape \(2, 2\)",
    ):
        store.replace_many({"humidity": jnp.ones((2, 1))})


@pytest.mark.fast_always
def test_public_run_state_replace_fields_rejects_shape_changes() -> None:
    grid = make_test_grid(name="public-replace-shape")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(DataComponent("MODEL", grid, {"value": 1.0}),),
        run_order=("MODEL",),
    )
    state = coupler.initial_state()

    with pytest.raises(
        ValueError,
        match=r"value.*shape \(1, 2\).*existing shape \(2, 2\)",
    ):
        state.replace_fields("MODEL", {"value": jnp.ones((1, 2))})


@pytest.mark.fast_always
def test_component_step_shape_changes_raise_component_error() -> None:
    grid = make_test_grid(name="component-step-shape")
    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields: {"value": jnp.ones((1, 2))},
        spec=ComponentSpec(
            outputs=("value",), initial_fields={"value": 1.0}, execution="host"
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend="host"),
    )

    with pytest.raises(
        ComponentError,
        match=r"MODEL.*value.*shape \(1, 2\).*existing shape \(2, 2\)",
    ):
        coupler.run()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("returned", "actual_type"),
    (
        pytest.param(7, "int", id="integer"),
        pytest.param(["value"], "list", id="list"),
        pytest.param(None, "NoneType", id="none"),
    ),
)
def test_component_step_rejects_non_mapping_non_step_result_returns(
    returned: object,
    actual_type: str,
) -> None:
    grid = make_test_grid(name=f"malformed-step-{actual_type}")
    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields: cast(Any, returned),
        spec=ComponentSpec(
            outputs=("value",), initial_fields={"value": 1.0}, execution="host"
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend="host"),
    )

    with pytest.raises(
        ComponentError,
        match=rf"Component 'MODEL'.*mapping or StepResult.*{actual_type}",
    ):
        coupler.run()


@pytest.mark.fast_always
def test_component_step_rejects_step_result_with_non_mapping_fields() -> None:
    grid = make_test_grid(name="malformed-step-result-fields")
    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields: StepResult(fields=cast(Any, ["value"])),
        spec=ComponentSpec(
            outputs=("value",), initial_fields={"value": 1.0}, execution="host"
        ),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
        runtime=RuntimeOptions(backend="host"),
    )

    with pytest.raises(
        TypeError,
        match=r"StepResult.fields.*mapping.*list",
    ):
        coupler.run()


def _masked_scalar_component(
    value: jax.Array,
    *,
    step: Any | None = None,
    execution: Literal["jax", "host"] = "jax",
) -> CallableComponent:
    """Build a two-cell component with one inactive grid location."""

    grid = make_test_grid(
        name="masked-scalar",
        longitude=np.asarray([0.0, 1.0]),
        latitude=np.asarray([0.0]),
        binary_mask=np.asarray([[1.0, 0.0]]),
    )
    return CallableComponent(
        "SAFE",
        grid,
        (lambda fields: {"value": fields["value"]}) if step is None else step,
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": value},
            execution=execution,
        ),
    )


def _single_component_coupler(component: CallableComponent) -> Coupler:
    """Build a one-step coupler for numerical runtime-boundary tests."""

    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("SAFE",),
        runtime=RuntimeOptions(
            backend="host" if component.spec.execution == "host" else "jax"
        ),
    )


@pytest.mark.fast_always
def test_initial_state_rejects_active_nan_but_allows_inactive_missing_nan() -> None:
    invalid = _masked_scalar_component(jnp.asarray([[jnp.nan, jnp.nan]]))
    with pytest.raises(
        CouplerError, match="Component 'SAFE'.*field 'value'.*active domain"
    ):
        _single_component_coupler(invalid).initial_state()

    valid = _masked_scalar_component(jnp.asarray([[2.0, jnp.nan]]))
    state = _single_component_coupler(valid).initial_state()
    fields = np.asarray(state.component("SAFE").field("value"))
    assert np.isfinite(fields[0, 0])
    assert np.isnan(fields[0, 1])


@pytest.mark.fast_always
def test_component_step_fails_at_first_active_nonfinite_output() -> None:
    component = _masked_scalar_component(
        jnp.asarray([[2.0, jnp.nan]]),
        step=lambda fields: {"value": fields["value"].at[0, 0].set(jnp.nan)},
        execution="host",
    )
    with pytest.raises(
        CouplerError,
        match="Component 'SAFE' step output field 'value'.*active domain",
    ):
        _single_component_coupler(component).run()


@pytest.mark.fast_always
def test_compiled_component_step_reports_active_nonfinite_output() -> None:
    component = _masked_scalar_component(
        jnp.asarray([[2.0, jnp.nan]]),
        step=lambda fields: {"value": fields["value"].at[0, 0].set(jnp.nan)},
        execution="jax",
    )
    with pytest.raises(
        JaxRuntimeError,
        match="Component 'SAFE' step output field 'value'.*active domain",
    ):
        state = _single_component_coupler(component).run()
        jax.block_until_ready(state._component_state("SAFE").fields.get("value"))


def _state_validation_coupler(
    *,
    dormant_lifecycle: LifecycleHooks | None = None,
) -> Coupler:
    grid = make_test_grid(name="exact-state-components")
    active = DataComponent("ACTIVE", grid, {"value": 1.0})
    dormant = CallableComponent(
        "DORMANT",
        grid,
        lambda fields: {"value": fields["value"]},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 2.0},
            lifecycle=dormant_lifecycle,
        ),
    )
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(active, dormant),
        run_order=("ACTIVE",),
        runtime=RuntimeOptions(topology=None),
    )


@pytest.mark.fast_always
def test_runtime_state_requires_exact_registered_component_names() -> None:
    coupler = _state_validation_coupler()
    state = coupler.initial_state()
    missing = RunState._from_runtime(
        component_names=("ACTIVE",),
        components=(state._component_state("ACTIVE"),),
        fractional_masks=state._fractional_masks,
        component_grids=(state._component_grids[state._component_index("ACTIVE")],),
    )
    extra = RunState._from_runtime(
        component_names=(*state._component_names, "EXTRA"),
        components=(*state._components, state._component_state("ACTIVE")),
        fractional_masks=state._fractional_masks,
        component_grids=(*state._component_grids, state._component_grids[0]),
    )

    with pytest.raises(CouplerError, match="component.*missing.*DORMANT"):
        coupler.run(missing)
    with pytest.raises(CouplerError, match="component.*extra.*EXTRA"):
        coupler.run(extra)


@pytest.mark.fast_always
def test_runtime_state_validates_declared_fields_outside_run_order() -> None:
    coupler = _state_validation_coupler()
    state = coupler.initial_state()
    dormant = state._component_state("DORMANT").with_fields(
        FieldStore.from_mapping({"value": jnp.ones((1, 2))})
    )
    malformed = state._with_component_state("DORMANT", dormant)

    with pytest.raises(
        CouplerError,
        match=r"DORMANT.*value.*shape \(1, 2\).*expected.*grid shape \(2, 2\)",
    ):
        coupler.run(malformed)


@pytest.mark.fast_always
def test_runtime_state_runs_validation_hooks_outside_run_order() -> None:
    validated: list[str] = []

    def validate(component: Component, context: Any) -> None:
        _ = context
        validated.append(component.name)

    coupler = _state_validation_coupler(
        dormant_lifecycle=LifecycleHooks(validate=validate)
    )

    coupler.initial_state()

    assert validated == ["DORMANT"]


@pytest.mark.fast_always
def test_gradient_flows_through_component_payload() -> None:
    grid = make_test_grid(name="payload-gradient")
    component = CallableComponent(
        "MODEL",
        grid,
        lambda fields, context, payload: {"value": fields["value"] * payload},
        spec=ComponentSpec(outputs=("value",), initial_fields={"value": 2.0}),
    )
    state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"value": jnp.full(grid.shape, 2.0)}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )
    context = StepContext(dt_seconds=60.0)

    def loss(payload: jax.Array) -> jax.Array:
        result = step_component_runtime_state(
            component,
            state.with_payload(payload),
            context,
            allow_host_runtime=False,
        )
        return jnp.sum(result.fields.get("value"))

    gradient = jax.grad(loss)(jnp.asarray(3.0))

    assert_allclose_compact(gradient, np.asarray(8.0))
