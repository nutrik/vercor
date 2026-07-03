from __future__ import annotations

from datetime import datetime
from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import DummyComponent, make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components import (
    Component,
    DataComponent,
    HostComponent,
)
from vercor.setups.data.era5_atmosphere import make_era5_atmosphere
from vercor.setups.data.era5_land import make_era5_land
from vercor.setups.data.era5_ocean import make_era5_ocean
from vercor.setups.data.erainterim_ocean import make_erainterim_ocean
from vercor.setups.data.jcm_land import make_jcm_land
from vercor.setups.external.jax_gcm_fields import JAXGCM_OUTPUT_GRID_FIELD_NAMES
import vercor.setups.external.jax_gcm_runtime as jax_gcm_runtime_module
import vercor.setups.external.jax_gcm_output as jax_gcm_output_module
import vercor.setups.external.jax_gcm_state as jax_gcm_state_module
from vercor.output.adapters import ComponentOutputAdapter
from vercor.setups.external.jax_gcm_state import JCMState
from vercor.setups.slab.atmosphere import make_slab_atmosphere
from vercor.setups.slab.land import make_slab_land
from vercor.setups.slab.ocean import make_slab_ocean
from vercor.setups.slab.seaice import make_slab_seaice
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.exchange import Exchange
from vercor.forcing_index import daily_forcing_index
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear, conservative
from vercor.runtime.state import RuntimeComponentState, RuntimeCouplerState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.settings import VercorSettings
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_periodic_interval,
)
from tests._runtime_helpers import replace_runtime_topology_maps, run_scanned_coupler


class _IdentityRegridder:
    def __call__(self, *args: Any) -> Any:
        if len(args) == 1:
            return jnp.asarray(args[0])
        return tuple(jnp.asarray(arg) for arg in args)


def _identity_factory(*args: Any, **kwargs: Any) -> _IdentityRegridder:
    _ = args, kwargs
    return _IdentityRegridder()


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


class _FakePhysicsPrediction(NamedTuple):
    surface_flux: _FakeSurfaceFlux
    shortwave_rad: _FakeShortwaveRad


class _FakeDynamicsPrediction(NamedTuple):
    normalized_surface_pressure: jax.Array
    u_wind: jax.Array
    v_wind: jax.Array
    temperature: jax.Array
    specific_humidity: jax.Array


class _FakePrediction(NamedTuple):
    physics: _FakePhysicsPrediction
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
    imports: tuple[str, ...] = (),
    exports: tuple[str, ...] = (),
    settings: VercorSettings | None = None,
) -> Any:
    _ = component_type
    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields=data,
        settings=settings or VercorSettings(),
    )
    component.declare_fields(inputs=imports, outputs=exports)
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
        physics=_FakePhysicsPrediction(
            surface_flux=_FakeSurfaceFlux(
                shf=surface_flux[jnp.newaxis, ...],
                evap=(surface_flux * 0.1)[jnp.newaxis, ...],
                rlds=(surface_temperature + 3.0)[jnp.newaxis, ...],
            ),
            shortwave_rad=_FakeShortwaveRad(
                rsns=(surface_temperature + 4.0)[jnp.newaxis, ...],
            ),
        ),
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
        prog=state.prog,
        phydata=state.phydata,
        metadata=state.metadata + jnp.sum(surface_temperature),
    )
    return updated_state, prediction


def _make_jax_gcm_fixture(grid: RectilinearGrid) -> _JAXGCMFixture:
    state = jax_gcm_state_module.JAXGCMSetupState.__new__(
        jax_gcm_state_module.JAXGCMSetupState
    )
    state.name = "ATM"
    state.grid = grid
    state.settings = VercorSettings()
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
        prog={"marker": jnp.asarray(0.0)},
        phydata={},
        metadata=jnp.asarray(0.0),
    )
    state.forcing = _FakeJCMForcing(
        stl_am=jnp.zeros((grid.shape[1], grid.shape[0]), dtype=jnp.float64),
        sea_surface_temperature=jnp.zeros(
            (grid.shape[1], grid.shape[0]),
            dtype=jnp.float64,
        ),
    )
    state._step_function = _fake_jcm_step
    state.output_adapter = ComponentOutputAdapter(
        empty_error_message=jax_gcm_output_module.JAX_GCM_AVERAGE_EMPTY_ERROR_MESSAGE,
        time_dim=jax_gcm_output_module.JAX_GCM_TIME_DIM,
        dimension_order=jax_gcm_output_module.JAX_GCM_OUTPUT_DIMENSION_ORDER,
    )
    state.output_frequency = None
    component = Component.from_step(
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
        inputs=("land_surface_temperature", "sea_surface_temperature"),
        outputs=(
            "land_surface_temperature",
            "sea_surface_temperature",
            "total_surface_temperature",
            *JAXGCM_OUTPUT_GRID_FIELD_NAMES,
            "pressure",
        ),
        defaults={
            field_name: 0.0
            for field_name in jax_gcm_runtime_module.jax_gcm_default_field_names(
                include_total_surface_temperature=True
            )
        },
        create_runtime_payload=(
            lambda component: jax_gcm_runtime_module.create_jax_gcm_runtime_payload(
                state
            )
        ),
        prefill_runtime_state_fields=(
            lambda component, data, incoming, outgoing, contract: (
                jax_gcm_runtime_module.prefill_jax_gcm_runtime_fields(
                    state,
                    component,
                    data,
                    incoming,
                    outgoing,
                    contract,
                )
            )
        ),
        validate_runtime_state=(
            lambda component, component_state, contract: (
                jax_gcm_runtime_module.validate_jax_gcm_runtime_state(
                    state,
                    component,
                    component_state,
                    contract,
                )
            )
        ),
    )
    component.seed_fields(state.data)
    return _JAXGCMFixture(component=component, state=state)


def _make_jax_gcm_component(grid: RectilinearGrid) -> Component:
    return _make_jax_gcm_fixture(grid).component


def _component_state(
    name: str,
    data: dict[str, jax.Array],
    imports: tuple[str, ...],
    exports: tuple[str, ...],
) -> RuntimeComponentState:
    _ = name
    zeros = jnp.zeros((2, 2), dtype=jnp.float64)
    return RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                field: data.get(field, zeros)
                for field in sorted(set(data) | set(imports) | set(exports))
            }
        ),
        incoming=RuntimeFieldStore.from_mapping(
            {field: data.get(field, zeros) for field in imports}
        ),
        outgoing=RuntimeFieldStore.from_mapping(
            {field: data.get(field, zeros) for field in exports}
        ),
    )


def _make_coupler(steps: int) -> Coupler:
    grid = make_test_grid(name="slab")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.components = {
        "ATM": make_slab_atmosphere(grid),
        "OCN": make_slab_ocean(grid),
        "LND": make_slab_land(grid),
        "ICE": make_slab_seaice(grid),
    }
    coupler.run_sequence = (
        "ATM",
        "OCN",
        "LND",
        "ICE",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=["sensible_heat_flux", "latent_heat_flux"],
            regrid=cast(Any, _identity_factory),
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=["latent_heat_flux"],
            regrid=cast(Any, _identity_factory),
        ),
        Exchange(
            source="OCN",
            target="ICE",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        ),
    ]
    key = ("OCN", "ATM", "_identity_factory")
    regridders = cast(
        Any,
        {
            key: _IdentityRegridder(),
            ("ATM", "OCN", "_identity_factory"): _IdentityRegridder(),
            ("ATM", "LND", "_identity_factory"): _IdentityRegridder(),
            ("OCN", "ICE", "_identity_factory"): _IdentityRegridder(),
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

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.add_component(make_slab_atmosphere(atmosphere_grid))
    coupler.add_component(make_slab_ocean(ocean_grid))
    coupler.add_component(make_slab_land(land_grid))
    coupler.add_component(make_slab_seaice(ice_grid))
    coupler.add_exchange(
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ICE",
            target="ATM",
            fields=["ice_fraction"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            target="OCN",
            fields=["sensible_heat_flux", "latent_heat_flux"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            target="LND",
            fields=["latent_heat_flux"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="OCN",
            target="ICE",
            fields=["sea_surface_temperature"],
            regrid=bilinear,
        )
    )
    coupler.set_run_order(
        (
            "ATM",
            "OCN",
            "LND",
            "ICE",
        )
    )
    coupler.initialize()
    return coupler


def test_coupler_initialize_cascades_float32_precision_to_component_arrays() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)

    for component in coupler.components.values():
        assert component.settings.enable_x64 is False
        assert component.grid.longitude.dtype == jnp.float32
        assert component.grid.latitude.dtype == jnp.float32
        if component.grid.binary_mask is not None:
            assert component.grid.binary_mask.dtype == jnp.float32
        for field_value in component.data.values():
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

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)
    )
    coupler.add_component(make_slab_atmosphere(atmosphere_grid))
    coupler.add_component(make_slab_ocean(ocean_grid))
    coupler.add_component(make_slab_land(land_grid))
    coupler.add_component(make_slab_seaice(ice_grid))
    coupler.add_exchange(
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=conservative,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ICE",
            target="ATM",
            fields=["ice_fraction"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            target="OCN",
            fields=["sensible_heat_flux", "latent_heat_flux"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="ATM",
            target="LND",
            fields=["latent_heat_flux"],
            regrid=bilinear,
        )
    )
    coupler.add_exchange(
        Exchange(
            source="OCN",
            target="ICE",
            fields=["sea_surface_temperature"],
            regrid=bilinear,
        )
    )
    coupler.set_run_order(
        (
            "ATM",
            "OCN",
            "LND",
            "ICE",
        )
    )
    coupler.initialize()
    return coupler


def _make_initial_state(sea_surface_temperature: jax.Array) -> RuntimeCouplerState:
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
            },
            imports=("sea_surface_temperature",),
            exports=(
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
            },
            imports=("sensible_heat_flux", "latent_heat_flux"),
            exports=("sea_surface_temperature",),
        ),
        _component_state(
            "LND",
            {
                "soil_moisture": jnp.full_like(sea_surface_temperature, 0.3),
                "land_surface_temperature": temperature_2m,
                "latent_heat_flux": zeros,
            },
            imports=("latent_heat_flux",),
            exports=("soil_moisture", "land_surface_temperature"),
        ),
        _component_state(
            "ICE",
            {
                "ice_fraction": zeros,
                "sea_surface_temperature": sea_surface_temperature,
            },
            imports=("sea_surface_temperature",),
            exports=("ice_fraction",),
        ),
    )
    return RuntimeCouplerState(
        component_names=("ATM", "OCN", "LND", "ICE"),
        components=components,
        fractional_masks=RuntimeFieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|OCN|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "ATM|LND|_identity_factory": jnp.ones_like(sea_surface_temperature),
                "OCN|ICE|_identity_factory": jnp.ones_like(sea_surface_temperature),
            }
        ),
    )


def _with_ocean_sst(
    state: RuntimeCouplerState, sea_surface_temperature: jax.Array
) -> RuntimeCouplerState:
    ocean = state.get_component_state("OCN")
    ocean = ocean.with_data(
        ocean.data.set("sea_surface_temperature", sea_surface_temperature)
    )
    ocean = ocean.with_outgoing(
        ocean.outgoing.set("sea_surface_temperature", sea_surface_temperature)
    )
    return state.set_component_state("OCN", ocean)


def _without_store_field(
    store: RuntimeFieldStore, field_name: str
) -> RuntimeFieldStore:
    return RuntimeFieldStore.from_mapping(
        {
            name: value
            for name, value in zip(store.field_names, store.values)
            if name != field_name
        }
    )


def test_run_supports_jit_grad_and_jvp() -> None:
    coupler = _make_coupler(steps=2)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    initial_state = _make_initial_state(initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    ocean_sst = final_state.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    )

    assert ocean_sst.shape == (2, 2)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sst: jax.Array) -> jax.Array:
        state = _make_initial_state(sst)
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    _, tangent = jax.jvp(loss, (initial_sst,), (jnp.ones_like(initial_sst),))

    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(tangent))


def test_run_matches_one_step_closed_form_for_slab_ocean() -> None:
    coupler = _make_coupler(steps=1)
    initial_sst = jnp.full((2, 2), 286.15, dtype=jnp.float64)
    final_state = run_scanned_coupler(coupler, _make_initial_state(initial_sst))

    ocean_sst = final_state.get_component_state("OCN").data.get(
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
    canonical_state = coupler.state()
    runtime_state_copy = coupler.state()
    assert runtime_state_copy.component_names == canonical_state.component_names
    initial_state = _with_ocean_sst(canonical_state, initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    ocean_sst = final_state.get_component_state("OCN").data.get(
        "sea_surface_temperature"
    )

    assert final_state.component_names == ("ATM", "OCN", "LND", "ICE")
    assert ocean_sst.shape == (2, 2)
    assert isinstance(ocean_sst, jax.Array)
    assert np.all(np.isfinite(np.asarray(ocean_sst)))

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        state = _with_ocean_sst(initial_state, sea_surface_temperature)
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_initialized_slab_coupler_run_prefills_missing_imports() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)
    ocean = coupler.components["OCN"]

    final_state = coupler.run()
    ocean_state = final_state.get_component_state("OCN")

    assert final_state.component_names == ("ATM", "OCN", "LND", "ICE")
    assert ocean_state.incoming.get("sensible_heat_flux").shape == ocean.grid.shape
    assert ocean_state.incoming.get("latent_heat_flux").shape == ocean.grid.shape


def test_scanned_runtime_state_uses_runtime_field_stores() -> None:
    coupler = _make_initialized_slab_coupler(steps=1)
    initial_state = coupler.state()

    assert all(
        isinstance(component_state.data, RuntimeFieldStore)
        and isinstance(component_state.incoming, RuntimeFieldStore)
        and isinstance(component_state.outgoing, RuntimeFieldStore)
        for component_state in initial_state.components
    )

    final_state = run_scanned_coupler(coupler, initial_state)

    assert all(
        isinstance(component_state.data, RuntimeFieldStore)
        and isinstance(component_state.incoming, RuntimeFieldStore)
        and isinstance(component_state.outgoing, RuntimeFieldStore)
        for component_state in final_state.components
    )


def test_mixed_grid_slab_coupler_runs_with_real_regridders_under_jit_grad_and_jvp() -> (
    None
):
    coupler = _make_initialized_mixed_grid_slab_coupler(steps=2)
    initial_state = coupler.state()
    initial_sst = jnp.linspace(285.15, 287.15, 9, dtype=jnp.float64).reshape((3, 3))
    initial_state = _with_ocean_sst(initial_state, initial_sst)

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )

    atmosphere = final_state.get_component_state("ATM")
    ocean = final_state.get_component_state("OCN")
    ice = final_state.get_component_state("ICE")
    atmosphere_sst = atmosphere.incoming.get("sea_surface_temperature")
    ocean_sst = ocean.data.get("sea_surface_temperature")
    ice_sst = ice.incoming.get("sea_surface_temperature")

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
            result.get_component_state("OCN").data.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(initial_sst)
    _, tangent = jax.jvp(loss, (initial_sst,), (jnp.ones_like(initial_sst),))

    assert gradient.shape == initial_sst.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.isfinite(np.asarray(tangent))


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
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.settings.year_in_seconds = 12.0
    coupler.components = {
        "OCN": _make_data_component(
            make_era5_ocean,
            name="OCN",
            grid=grid,
            data={"sea_surface_temperature": monthly_ocean},
            exports=("sea_surface_temperature",),
            settings=VercorSettings(apply_time_interpolation=True),
        ),
        "LND": _make_data_component(
            make_era5_land,
            name="LND",
            grid=grid,
            data={"land_surface_temperature": monthly_land},
            exports=("land_surface_temperature",),
            settings=VercorSettings(apply_time_interpolation=True),
        ),
        "ATM": _make_data_component(
            make_era5_atmosphere,
            name="ATM",
            grid=grid,
            data={
                "sea_surface_temperature": jnp.zeros((2, 2), dtype=jnp.float64),
                "land_surface_temperature": jnp.zeros((2, 2), dtype=jnp.float64),
            },
            imports=("sea_surface_temperature", "land_surface_temperature"),
        ),
    }
    coupler.run_sequence = (
        "OCN",
        "LND",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        ),
    ]
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "_identity_factory"): _IdentityRegridder(),
            ("LND", "ATM", "_identity_factory"): _IdentityRegridder(),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            key: jnp.ones(grid.shape, dtype=jnp.float64) for key in regridders
        },
    )

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere = final_state.get_component_state("ATM")
    received_ocean = atmosphere.incoming.get("sea_surface_temperature")
    received_land = atmosphere.incoming.get("land_surface_temperature")
    expected_ocean = np.asarray(monthly_ocean[0])
    expected_land = np.asarray(monthly_land[0])

    assert received_ocean.shape == grid.shape
    assert received_land.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere.data.field_names
    assert isinstance(received_ocean, jax.Array)
    assert isinstance(received_land, jax.Array)
    assert_allclose_compact(received_ocean, expected_ocean)
    assert_allclose_compact(received_land, expected_land)

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean = initial_state.get_component_state("OCN")
        state = initial_state.set_component_state(
            "OCN",
            ocean.with_data(ocean.data.set("sea_surface_temperature", ocean_forcing)),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(monthly_ocean)
    assert gradient.shape == monthly_ocean.shape
    assert_allclose_compact(gradient[0], np.ones((2, 2)))
    assert_allclose_compact(gradient[1:], np.zeros((11, 2, 2)))


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
    ocean = DataComponent.from_fields(
        name="OCN",
        grid=grid,
        fields={"sea_surface_temperature": monthly_ocean},
        settings=VercorSettings(apply_time_interpolation=True),
    )
    atmosphere = make_slab_atmosphere(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.settings.year_in_seconds = 12.0
    coupler.components = {"OCN": ocean, "ATM": atmosphere}
    coupler.run_sequence = (
        "OCN",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        )
    ]
    key = ("OCN", "ATM", "_identity_factory")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = coupler.state()
    ocean_state = initial_state.get_component_state("OCN")
    assert ocean_state.data.get("sea_surface_temperature").shape == monthly_ocean.shape
    assert ocean_state.outgoing.get("sea_surface_temperature").shape == grid.shape
    assert_allclose_compact(
        ocean_state.outgoing.get("sea_surface_temperature"),
        np.asarray(first_month),
    )

    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    received_sst = final_state.get_component_state("ATM").incoming.get(
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
        exports=("sea_surface_temperature",),
        settings=VercorSettings(apply_daily_time_selection=True),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 2), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"OCN": ocean, "ATM": atmosphere}
    coupler.run_sequence = (
        "OCN",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=bilinear,
        )
    ]
    key = ("OCN", "ATM", "bilinear")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: bilinear(grid, grid)}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    atmosphere.data = {
        "temperature_2m": jnp.full(grid.shape, 288.15, dtype=jnp.float64),
        "sensible_heat_flux": jnp.zeros(grid.shape, dtype=jnp.float64),
        "latent_heat_flux": jnp.zeros(grid.shape, dtype=jnp.float64),
        "u_velocity_10m": jnp.zeros(grid.shape, dtype=jnp.float64),
        "v_velocity_10m": jnp.zeros(grid.shape, dtype=jnp.float64),
        "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
    }

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_sst = atmosphere_state.incoming.get("sea_surface_temperature")
    sensible_heat_flux = atmosphere_state.data.get("sensible_heat_flux")

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
        exports=("sea_surface_temperature",),
        settings=VercorSettings(apply_time_interpolation=True),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.settings.year_in_seconds = 12.0
    coupler.components = {"OCN": ocean, "ATM": atmosphere}
    coupler.run_sequence = (
        "OCN",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=bilinear,
        )
    ]
    key = ("OCN", "ATM", "bilinear")
    regridder = bilinear(ocean_grid, atmosphere_grid)
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: regridder}),
        fractional_masks={key: jnp.ones(atmosphere_grid.shape)},
    )
    atmosphere.data = {
        "temperature_2m": jnp.full(atmosphere_grid.shape, 288.15, dtype=jnp.float64),
        "sensible_heat_flux": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "latent_heat_flux": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "u_velocity_10m": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "v_velocity_10m": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
        "sea_surface_temperature": jnp.zeros(atmosphere_grid.shape, dtype=jnp.float64),
    }

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_sst = atmosphere_state.incoming.get("sea_surface_temperature")
    sensible_heat_flux = atmosphere_state.data.get("sensible_heat_flux")
    expected_source = np.asarray(first_month)
    expected_received = regridder(expected_source)

    assert received_sst.shape == atmosphere_grid.shape
    assert sensible_heat_flux.shape == atmosphere_grid.shape
    assert isinstance(received_sst, jax.Array)
    assert isinstance(sensible_heat_flux, jax.Array)
    assert_allclose_compact(received_sst, expected_received)
    assert np.all(np.isfinite(np.asarray(sensible_heat_flux)))

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state.get_component_state("OCN")
        state = initial_state.set_component_state(
            "OCN",
            ocean_state.with_data(
                ocean_state.data.set("sea_surface_temperature", ocean_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(monthly_ocean)
    assert gradient.shape == monthly_ocean.shape
    assert np.all(np.isfinite(np.asarray(gradient[0])))
    assert_allclose_compact(gradient[1:], np.zeros((11, 2, 3)))


def test_jcm_land_daily_forcing_replays_to_data_atmosphere_under_jit_and_grad() -> None:
    grid = make_test_grid(name="jcm-land")
    forcing = jnp.zeros((365, 2, 2), dtype=jnp.float64)
    forcing = forcing.at[2].set(jnp.asarray([[286.0, 287.0], [288.0, 289.0]]))
    land = _make_data_component(
        make_jcm_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": forcing},
        exports=("land_surface_temperature",),
        settings=VercorSettings(apply_daily_time_selection=True),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        imports=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 3), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"LND": land, "ATM": atmosphere}
    coupler.run_sequence = (
        "LND",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=bilinear,
        )
    ]
    key = ("LND", "ATM", "bilinear")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: bilinear(grid, grid)}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_temperature = atmosphere_state.incoming.get("land_surface_temperature")

    assert received_temperature.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere_state.data.field_names
    assert isinstance(received_temperature, jax.Array)
    assert_allclose_compact(received_temperature, np.asarray(forcing[2]))

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state.get_component_state("LND")
        state = initial_state.set_component_state(
            "LND",
            land_state.with_data(
                land_state.data.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("land_surface_temperature")
        )

    gradient = jax.grad(loss)(forcing)
    assert gradient.shape == forcing.shape
    assert_allclose_compact(gradient[2], np.ones(grid.shape))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((362, 2, 2)))


def test_noleap_daily_forcing_replays_calendar_slice_under_jit_and_grad() -> None:
    grid = make_test_grid(name="noleap-daily")
    forcing = jnp.arange(365 * 2 * 2, dtype=jnp.float64).reshape((365, 2, 2))
    land = _make_data_component(
        make_jcm_land,
        name="LND",
        grid=grid,
        data={"land_surface_temperature": forcing},
        exports=("land_surface_temperature",),
        settings=VercorSettings(apply_daily_time_selection=True),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        imports=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(
            start=datetime(2024, 3, 1),
            dt_seconds=3600.0,
            steps=1,
            year_type="noleap",
        )
    )
    coupler.components = {"LND": land, "ATM": atmosphere}
    coupler.run_sequence = (
        "LND",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        )
    ]
    key = ("LND", "ATM", "_identity_factory")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_temperature = atmosphere_state.incoming.get("land_surface_temperature")

    assert "total_surface_temperature" not in atmosphere_state.data.field_names
    assert_allclose_compact(received_temperature, np.asarray(forcing[59]))

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state.get_component_state("LND")
        state = initial_state.set_component_state(
            "LND",
            land_state.with_data(
                land_state.data.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("land_surface_temperature")
        )

    gradient = jax.grad(loss)(forcing)
    assert_allclose_compact(gradient[59], np.ones(grid.shape))
    assert_allclose_compact(gradient[:59], np.zeros((59, 2, 2)))
    assert_allclose_compact(gradient[60:], np.zeros((305, 2, 2)))


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
        exports=("land_surface_temperature",),
        settings=VercorSettings(apply_daily_time_selection=True),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        imports=("land_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(
            start=datetime(2001, 2, 28),
            dt_seconds=3600.0,
            steps=1,
            year_type="360",
        )
    )
    coupler.components = {"LND": land, "ATM": atmosphere}
    coupler.run_sequence = (
        "LND",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="LND",
            target="ATM",
            fields=["land_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        )
    ]
    key = ("LND", "ATM", "_identity_factory")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    _, runtime_time, _ = next(coupler.clock.iter())
    expected_index = daily_forcing_index(
        runtime_time,
        year_type=coupler.clock.year_type,
        no_leap=True,
    )
    expected_slice = forcing[expected_index]

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_temperature = atmosphere_state.incoming.get("land_surface_temperature")

    assert expected_index == 56
    assert_allclose_compact(expected_slice, np.asarray(forcing[56]))
    assert "total_surface_temperature" not in atmosphere_state.data.field_names
    assert_allclose_compact(received_temperature, expected_slice)

    def loss(land_forcing: jax.Array) -> jax.Array:
        land_state = initial_state.get_component_state("LND")
        state = initial_state.set_component_state(
            "LND",
            land_state.with_data(
                land_state.data.set("land_surface_temperature", land_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("land_surface_temperature")
        )

    gradient = jax.grad(loss)(forcing)
    assert_allclose_compact(gradient[56], np.ones(grid.shape))
    assert_allclose_compact(gradient[:56], np.zeros((56, 2, 2)))
    assert_allclose_compact(gradient[57:], np.zeros((308, 2, 2)))


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
        exports=("sea_surface_temperature",),
        settings=VercorSettings(apply_time_interpolation=True),
    )
    atmosphere = _make_data_component(
        make_era5_atmosphere,
        name="ATM",
        grid=grid,
        data={
            "sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
            "land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64),
        },
        imports=("sea_surface_temperature",),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2001, 12, 31, 12), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"OCN": ocean, "ATM": atmosphere}
    coupler.run_sequence = (
        "OCN",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        )
    ]
    key = ("OCN", "ATM", "_identity_factory")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    (left_index, left_weight), (right_index, right_weight) = get_periodic_interval(
        current_time=datetime_to_seconds_in_year(coupler.clock.start),
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    received_ocean = atmosphere_state.incoming.get("sea_surface_temperature")
    expected = np.asarray(
        left_weight * monthly_ocean[left_index]
        + right_weight * monthly_ocean[right_index]
    )

    assert (left_index, right_index) == (11, 0)
    assert received_ocean.shape == grid.shape
    assert "total_surface_temperature" not in atmosphere_state.data.field_names
    assert_allclose_compact(received_ocean, expected)

    def loss(ocean_forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state.get_component_state("OCN")
        state = initial_state.set_component_state(
            "OCN",
            ocean_state.with_data(
                ocean_state.data.set("sea_surface_temperature", ocean_forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(
            result.get_component_state("ATM").incoming.get("sea_surface_temperature")
        )

    gradient = jax.grad(loss)(monthly_ocean)
    assert_allclose_compact(gradient[0], np.full(grid.shape, right_weight))
    assert_allclose_compact(gradient[11], np.full(grid.shape, left_weight))
    assert_allclose_compact(gradient[1:11], np.zeros((10, 2, 2)))


def test_jax_gcm_runs_inside_runtime_under_jit_and_grad() -> None:
    grid = make_test_grid(name="jcm-runtime")
    fixture = _make_jax_gcm_fixture(grid)
    component = fixture.component
    original_state = fixture.state._state
    original_forcing = fixture.state.forcing
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = ("ATM",)

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    temperature = atmosphere_state.data.get("temperature")
    sensible_heat_flux = atmosphere_state.data.get("sensible_heat_flux")

    assert fixture.state._state is original_state
    assert fixture.state.forcing is original_forcing
    assert temperature.shape == grid.shape
    assert sensible_heat_flux.shape == grid.shape
    assert isinstance(temperature, jax.Array)
    assert isinstance(sensible_heat_flux, jax.Array)
    assert np.all(np.isfinite(np.asarray(temperature)))
    assert np.all(np.isfinite(np.asarray(sensible_heat_flux)))
    assert atmosphere_state.runtime_payload is not None

    def loss(sea_surface_temperature: jax.Array) -> jax.Array:
        atmosphere = initial_state.get_component_state("ATM")
        state = initial_state.set_component_state(
            "ATM",
            atmosphere.with_data(
                atmosphere.data.set(
                    "sea_surface_temperature",
                    sea_surface_temperature,
                )
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(result.get_component_state("ATM").data.get("temperature"))

    gradient = jax.grad(loss)(jnp.full(grid.shape, 281.0, dtype=jnp.float64))
    assert gradient.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)


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
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = ("ATM",)

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    payload = atmosphere_state.runtime_payload

    assert atmosphere_state.data.get("temperature").shape == grid.shape
    assert payload is not None
    assert payload.forcing.stl_am.shape == forcing_template.stl_am.shape
    assert (
        payload.forcing.sea_surface_temperature.shape
        == forcing_template.sea_surface_temperature.shape
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
        exports=("sea_surface_temperature",),
    )
    atmosphere = _make_jax_gcm_component(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"OCN": ocean, "ATM": atmosphere}
    coupler.run_sequence = (
        "OCN",
        "ATM",
    )
    coupler.exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["sea_surface_temperature"],
            regrid=cast(Any, _identity_factory),
        )
    ]
    key = ("OCN", "ATM", "_identity_factory")
    replace_runtime_topology_maps(
        coupler,
        regridders=cast(Any, {key: _IdentityRegridder()}),
        fractional_masks={key: jnp.ones(grid.shape, dtype=jnp.float64)},
    )

    initial_state = coupler.state()
    final_state = jax.jit(lambda state: run_scanned_coupler(coupler, state))(
        initial_state
    )
    atmosphere_state = final_state.get_component_state("ATM")
    temperature = atmosphere_state.data.get("temperature")
    received_sst = atmosphere_state.incoming.get("sea_surface_temperature")

    assert temperature.shape == grid.shape
    assert received_sst.shape == grid.shape
    assert isinstance(temperature, jax.Array)
    assert isinstance(received_sst, jax.Array)
    assert_allclose_compact(received_sst, sea_surface_temperature)
    assert np.all(np.isfinite(np.asarray(temperature)))

    def loss(forcing: jax.Array) -> jax.Array:
        ocean_state = initial_state.get_component_state("OCN")
        state = initial_state.set_component_state(
            "OCN",
            ocean_state.with_data(
                ocean_state.data.set("sea_surface_temperature", forcing)
            ),
        )
        result = run_scanned_coupler(coupler, state)
        return jnp.sum(result.get_component_state("ATM").data.get("temperature"))

    gradient = jax.grad(loss)(sea_surface_temperature)
    _, tangent = jax.jvp(
        loss,
        (sea_surface_temperature,),
        (jnp.ones_like(sea_surface_temperature),),
    )

    assert gradient.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
    assert np.isfinite(np.asarray(tangent))


def test_jax_gcm_runtime_requires_initialized_payload() -> None:
    grid = make_test_grid(name="jcm-uninitialized")
    fixture = _make_jax_gcm_fixture(grid)
    component = fixture.component
    del fixture.state._state
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = ("ATM",)

    with pytest.raises(ComponentError, match="component initialization"):
        run_scanned_coupler(coupler)

    component = _make_jax_gcm_component(grid)
    coupler.components = {"ATM": component}
    state = coupler.state()
    atmosphere = state.get_component_state("ATM").with_runtime_payload(None)
    state = state.set_component_state("ATM", atmosphere)

    with pytest.raises(ComponentError, match="runtime payload"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_run_sequence() -> None:
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )

    with pytest.raises(CouplerError, match="run sequence"):
        run_scanned_coupler(coupler)


def test_run_accepts_default_runtime_component() -> None:
    grid = make_test_grid(name="dummy")
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": cast(Any, DummyComponent("ATM", grid))}
    coupler.run_sequence = ("ATM",)

    final_state = run_scanned_coupler(coupler)

    assert final_state.component_names == ("ATM",)


def test_scanned_runtime_rejects_camulator_land_runtime_boundary() -> None:
    grid = make_test_grid(name="camulator")
    camulator_land = HostComponent.from_step(
        name="LND",
        grid=grid,
        step=lambda fields, context, payload: {},
        outputs=("land_surface_temperature",),
        defaults={"land_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64)},
    )
    camulator_land.seed_declared_defaults()
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"LND": cast(Any, camulator_land)}
    coupler.run_sequence = ("LND",)

    assert isinstance(camulator_land.data["land_surface_temperature"], jax.Array)
    state = coupler.state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        run_scanned_coupler(coupler, state)


def test_scanned_runtime_rejects_camulator_gcm_runtime_boundary() -> None:
    grid = make_test_grid(name="camulator-gcm")
    camulator = HostComponent.from_step(
        name="ATM",
        grid=grid,
        step=lambda fields, context, payload: {},
        outputs=("temperature",),
        defaults={"temperature": jnp.ones(grid.shape, dtype=jnp.float64)},
    )
    camulator.seed_declared_defaults()
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": cast(Any, camulator)}
    coupler.run_sequence = ("ATM",)

    assert isinstance(camulator.data["temperature"], jax.Array)
    state = coupler.state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        run_scanned_coupler(coupler, state)


def test_scanned_runtime_rejects_veros_runtime_boundary() -> None:
    grid = make_test_grid(name="veros")
    veros = HostComponent.from_step(
        name="OCN",
        grid=grid,
        step=lambda fields, context, payload: {},
        outputs=("sea_surface_temperature",),
        defaults={"sea_surface_temperature": jnp.zeros(grid.shape, dtype=jnp.float64)},
    )
    veros.seed_declared_defaults()
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"OCN": cast(Any, veros)}
    coupler.run_sequence = ("OCN",)

    assert isinstance(veros.data["sea_surface_temperature"], jax.Array)
    state = coupler.state()

    with pytest.raises(ComponentError, match="host-backed.*Coupler.run"):
        run_scanned_coupler(coupler, state)


def test_run_validates_regridders_and_fractional_masks() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    replace_runtime_topology_maps(coupler, regridders={})

    with pytest.raises(CouplerError, match="initialized regridder"):
        run_scanned_coupler(coupler, state)

    coupler = _make_coupler(steps=1)
    state = RuntimeCouplerState(
        component_names=state.component_names,
        components=state.components,
        fractional_masks=RuntimeFieldStore.empty(),
    )

    with pytest.raises(CouplerError, match="fractional mask"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_source_fields_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    ocean = state.get_component_state("OCN").with_outgoing(RuntimeFieldStore.empty())
    state = state.set_component_state("OCN", ocean)

    with pytest.raises(CouplerError, match="source field"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_slab_required_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    atmosphere = state.get_component_state("ATM")
    atmosphere = atmosphere.with_data(
        _without_store_field(atmosphere.data, "temperature_2m")
    )
    state = state.set_component_state("ATM", atmosphere)

    with pytest.raises(CouplerError, match="required data field 'temperature_2m'"):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_import_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    atmosphere = state.get_component_state("ATM")
    atmosphere = atmosphere.with_data(
        _without_store_field(atmosphere.data, "sea_surface_temperature")
    )
    state = state.set_component_state("ATM", atmosphere)

    with pytest.raises(
        CouplerError, match="required data field 'sea_surface_temperature'"
    ):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_export_data_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    ocean = state.get_component_state("OCN")
    ocean = ocean.with_data(_without_store_field(ocean.data, "sea_surface_temperature"))
    state = state.set_component_state("OCN", ocean)

    with pytest.raises(
        CouplerError, match="required data field 'sea_surface_temperature'"
    ):
        run_scanned_coupler(coupler, state)


def test_run_validates_missing_jax_gcm_preseed_before_scan() -> None:
    grid = make_test_grid(name="jcm-missing-preseed")
    component = _make_jax_gcm_component(grid)
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1)
    )
    coupler.components = {"ATM": component}
    coupler.run_sequence = ("ATM",)
    state = coupler.state()
    atmosphere = state.get_component_state("ATM")
    atmosphere = atmosphere.with_data(_without_store_field(atmosphere.data, "pressure"))
    state = state.set_component_state("ATM", atmosphere)

    with pytest.raises(CouplerError, match="required data field 'pressure'"):
        run_scanned_coupler(coupler, state)


def test_run_validates_fractional_mask_shape_before_scan() -> None:
    coupler = _make_coupler(steps=1)
    state = _make_initial_state(jnp.full((2, 2), 286.15, dtype=jnp.float64))
    state = RuntimeCouplerState(
        component_names=state.component_names,
        components=state.components,
        fractional_masks=RuntimeFieldStore.from_mapping(
            {
                "OCN|ATM|_identity_factory": jnp.ones((1, 1), dtype=jnp.float64),
                "ATM|OCN|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
                "ATM|LND|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
                "OCN|ICE|_identity_factory": jnp.ones((2, 2), dtype=jnp.float64),
            }
        ),
    )

    with pytest.raises(CouplerError, match="fractional mask.*shape"):
        run_scanned_coupler(coupler, state)
