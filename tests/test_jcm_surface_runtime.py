"""Protect native flux channels, surface provenance and fractional terrain."""

from datetime import datetime, timedelta
from inspect import Parameter, signature
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import DataComponent
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups import JAXGCMConfig, Spinup

pytestmark = pytest.mark.fast_always


def make_native_jcm(
    *,
    start: datetime = datetime(2000, 1, 3, 12),
    coupling_seconds: float = 1800.0,
    spinup: Spinup = Spinup(),
    water_albedo: float = 0.1,
) -> tuple[Coupler, Any, Any]:
    """Build actual flat native JCM with prescribed covered surface contributions."""
    config = JAXGCMConfig(
        seaice_coupling=True,
        spinup=spinup,
        model_timestep=timedelta(minutes=30),
        save_interval=timedelta(seconds=coupling_seconds),
        custom_parameters={"mod_radcon.albsea": water_albedo},
    )
    from tests.test_jcm_surface_forcing import native_case
    from vercor.setups import make_jax_gcm
    from dataclasses import replace

    jax.config.update("jax_enable_x64", True)
    coords, terrain, _, forcing = native_case()
    shape = terrain.fmask.shape
    forcing = forcing.copy(
        stl_am=jnp.full(shape, 300.0),
        soilw_am=jnp.full(shape, 0.2),
        sea_surface_temperature=jnp.full(shape, 278.0),
        snowc_am=jnp.zeros(shape),
    )
    atmosphere = make_jax_gcm(
        coords, terrain, config=replace(config, forcing_data=forcing)
    )
    canonical = atmosphere.grid.shape
    tag = jnp.arange(np.prod(canonical)).reshape(canonical) / 10000
    fields = {
        "land_surface_temperature": 300 + tag,
        "soil_moisture": jnp.full(canonical, 0.77),
        "open_water_temperature_contribution": jnp.full(canonical, 0.45 * 290),
        "ice_surface_temperature": jnp.full(canonical, 0.3 * 260),
        "ice_surface_albedo": jnp.full(canonical, 0.3 * 0.7),
        "ice_fraction": jnp.full(canonical, 0.3),
    }
    surface = DataComponent("SURFACE", atmosphere.grid, fields)
    coupler = Coupler(
        Clock(start, coupling_seconds, 1, calendar="noleap"),
        components=(surface, atmosphere),
        run_order=("SURFACE", "ATM"),
        exchanges=(Exchange("SURFACE", "ATM", tuple(fields), quantity="intensive"),),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True), topology=None),
        log_level="error",
    )
    hook = atmosphere.spec.lifecycle.setup
    assert hook is not None
    return coupler, atmosphere, cast(Any, hook).__self__


def test_native_flux_channels_are_selected_once() -> None:
    from vercor.setups._external.jax_gcm_fields import map_jcm_output_fields

    shape = (3, 2)
    profile = jnp.full((2, *shape), 280.0)
    shf = jnp.broadcast_to(jnp.array([10.0, 20.0, 17.5]), (*shape, 3))
    evap = jnp.broadcast_to(jnp.array([1.0, 2.0, 1.75]), (*shape, 3))
    mapped = map_jcm_output_fields(
        2.5e6,
        1e5,
        jnp.array([0.2, 0.95]),
        28.966,
        8314.47,
        1e5,
        0.286,
        shf,
        evap,
        jnp.full(shape, 210.0),
        jnp.full(shape, 180.0),
        jnp.full(shape, 0.9),
        jnp.ones_like(profile),
        jnp.ones_like(profile),
        profile,
        jnp.full_like(profile, 10.0),
        downward_shortwave_radiation_flux=jnp.full(shape, 200.0),
        open_water_albedo=jnp.array(0.1),
        large_scale_precipitation=jnp.zeros(shape),
        convective_precipitation=jnp.zeros(shape),
        temperature_2m=jnp.full(shape, 280.0),
        snow_temperature=jnp.array(273.0),
        rain_temperature=jnp.array(278.0),
        dtype=DTypePolicy(enable_x64=True),
    )
    np.testing.assert_allclose(mapped["sensible_heat_flux"], -17.5)
    np.testing.assert_allclose(mapped["land_sensible_heat_flux"], -10.0)
    np.testing.assert_allclose(mapped["latent_heat_flux"], -4375.0)
    np.testing.assert_allclose(mapped["land_latent_heat_flux"], -2500.0)
    np.testing.assert_allclose(mapped["surface_pressure"], 90000.0)
    np.testing.assert_allclose(mapped["downward_shortwave_radiation_flux"], 200.0)
    np.testing.assert_allclose(mapped["open_water_albedo"], 0.1)


def test_coupled_land_moisture_and_fraction_reach_native_forcing() -> None:
    from vercor.setups._external.jax_gcm_runtime import step_jax_gcm_runtime

    coupler, _, owner = make_native_jcm()
    initial = coupler.initial_state()
    source = initial.component("SURFACE")
    payload = cast(Any, initial._component_state("ATM").payload)
    result, _, applied = step_jax_gcm_runtime(
        owner,
        {**source.fields(), **source.fields(scope="received")},
        payload,
        PhysicalConstants(),
        DTypePolicy(enable_x64=True),
    )
    np.testing.assert_allclose(
        applied.stl_am, np.asarray(source.field("land_surface_temperature")).T
    )
    np.testing.assert_allclose(applied.soilw_am, 0.77)
    np.testing.assert_allclose(applied.sea_surface_temperature, 278.0)
    np.testing.assert_allclose(applied.sice_am, 0.4)
    np.testing.assert_allclose(applied.marine_surface_albedo, 0.34)
    np.testing.assert_allclose(result.fields["marine_ice_fraction"], 0.4)
    np.testing.assert_allclose(result.fields["marine_surface_albedo"], 0.34)
    assert "ice_fraction" not in result.fields
    np.testing.assert_allclose(source.field("ice_fraction"), 0.3)
    np.testing.assert_allclose(payload.forcing.soilw_am, 0.2)
    assert not hasattr(owner.model, "marine_surface_albedo")


def test_coupled_factory_uses_fractional_terrain_without_mutating_inputs() -> None:
    config = JAXGCMConfig(seaice_coupling=True)
    from tests.test_jcm_surface_forcing import native_case
    from vercor.setups import (
        JCMInputs,
        JCMLandAtmosphereConfig,
        make_jcm_land_atmosphere,
    )
    from vercor.grid_masks import create_lnd_mask_from_ocn

    coords, terrain, _, forcing = native_case()
    original = np.asarray(terrain.fmask).copy()
    ocean = RectilinearGrid(
        "hemisphere",
        longitude=np.array([90.0, 270.0]),
        latitude=np.array([-45.0, 45.0]),
        longitude_edges=np.array([0.0, 180.0, 360.0]),
        latitude_edges=np.array([-90.0, 0.0, 90.0]),
        binary_mask=np.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    pair = make_jcm_land_atmosphere(
        ocean,
        inputs=JCMInputs(coords, terrain, forcing),
        config=JCMLandAtmosphereConfig(atmosphere=config),
    )
    _, fraction = create_lnd_mask_from_ocn(
        atm_lat=pair.land.grid.latitude,
        atm_lon=pair.land.grid.longitude,
        ocn_grid=ocean,
    )
    np.testing.assert_allclose(pair.terrain.fmask, np.asarray(fraction).T)
    assert np.any((np.asarray(fraction) > 0) & (np.asarray(fraction) < 1))
    np.testing.assert_array_equal(terrain.fmask, original)
    assert set(pair.atmosphere.spec.inputs) == {
        "land_surface_temperature",
        "soil_moisture",
        "ice_surface_temperature",
        "ice_surface_albedo",
        "open_water_temperature_contribution",
        "ice_fraction",
    }
    assert "land_sensible_heat_flux" in pair.land.spec.inputs
    assert "land_latent_heat_flux" in pair.land.spec.inputs


def test_coupled_policy_is_keyword_only_and_specific_land_recipe_is_explicit() -> None:
    assert JAXGCMConfig().seaice_coupling is False
    assert (
        signature(JAXGCMConfig).parameters["seaice_coupling"].kind
        is Parameter.KEYWORD_ONLY
    )
    from vercor.recipes import JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS

    assert JCM_ATMOSPHERE_TO_LAND_FLUX_FIELDS == (
        "land_latent_heat_flux",
        "land_sensible_heat_flux",
    )


@pytest.mark.parametrize("series_field", [None, "stl_am", "soilw_am"])
def test_land_static_fields_keep_spatial_axes_with_daily_inputs(
    series_field: str | None,
) -> None:
    from types import SimpleNamespace
    from vercor.setups import make_jcm_land
    from tests.test_jcm_surface_forcing import native_case

    coords, _, _, _ = native_case(resolution=21)
    shape = coords.horizontal.nodal_shape
    values = {"stl_am": jnp.full(shape, 280.0), "soilw_am": jnp.full(shape, 0.6)}
    if series_field is not None:
        values[series_field] = jnp.stack([values[series_field] + k for k in range(365)])
    ocean = RectilinearGrid(
        "ocean",
        longitude=np.array([90.0, 270.0]),
        latitude=np.array([-45.0, 45.0]),
        binary_mask=np.zeros((2, 2)),
    )
    land = make_jcm_land(coords, SimpleNamespace(**values), ocean)
    assert land.spec.transfer.time_selection == (
        "current" if series_field is None else "daily"
    )
    for name, native in (
        ("land_surface_temperature", "stl_am"),
        ("soil_moisture", "soilw_am"),
    ):
        field = land.spec.initial_fields[name]
        expected = (
            values[native].T
            if series_field is None
            else jnp.broadcast_to(values[native], (365, *shape)).transpose(0, 2, 1)
        )
        assert jnp.shape(field) == expected.shape
        np.testing.assert_array_equal(field, expected)
