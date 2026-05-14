from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from setups.slab.atmosphere import (
    _bulk_flux_step,
    _default_sea_surface_temperature,
    _surface_wind_10m,
)
from setups.slab.land import _update_soil_moisture
from setups.slab.ocean import _advance_sea_surface_temperature
from setups.slab.seaice import _diagnose_ice_fraction


def test_atmosphere_kernels_support_jit_and_gradients() -> None:
    temperature_2m = jnp.asarray([[288.15, 289.15], [290.15, 291.15]])
    sea_surface_temperature = jnp.asarray([[280.0, 281.0], [282.0, 283.0]])

    default_sst = jax.jit(_default_sea_surface_temperature)(temperature_2m)
    sensible_heat_flux, latent_heat_flux, updated_temperature_2m = jax.jit(
        _bulk_flux_step
    )(temperature_2m, sea_surface_temperature)
    u_velocity_10m, v_velocity_10m = jax.jit(_surface_wind_10m)(
        jnp.asarray([-30.0, 30.0]),
        jnp.asarray([0.0, 90.0]),
    )

    assert_allclose_compact(default_sst, np.full((2, 2), 288.15))
    assert_allclose_compact(
        sensible_heat_flux,
        -10.0 * (np.asarray(temperature_2m) - np.asarray(sea_surface_temperature)),
    )
    assert_allclose_compact(latent_heat_flux, -0.5 * sensible_heat_flux)
    assert_allclose_compact(
        updated_temperature_2m,
        np.asarray(temperature_2m)
        - 0.01 * (np.asarray(temperature_2m) - np.asarray(sea_surface_temperature)),
    )
    assert u_velocity_10m.shape == (2, 2)
    assert v_velocity_10m.shape == (2, 2)

    gradient = jax.grad(lambda sst: jnp.sum(_bulk_flux_step(temperature_2m, sst)[0]))(
        sea_surface_temperature
    )
    assert_allclose_compact(gradient, np.full((2, 2), 10.0))


def test_ocean_kernel_supports_jit_and_matches_closed_form() -> None:
    sst = jnp.asarray([[288.15, 288.15], [288.15, 288.15]])
    sensible_heat_flux = jnp.full((2, 2), 20.0)
    latent_heat_flux = jnp.full((2, 2), 10.0)

    out = jax.jit(_advance_sea_surface_temperature)(
        sst,
        sensible_heat_flux,
        latent_heat_flux,
        3600.0,
        1025.0,
        3990.0,
        30.0,
        1.0 / (30.0 * 86400.0),
        288.15,
    )

    expected = (
        np.asarray(sst)
        + (np.asarray(sensible_heat_flux) + np.asarray(latent_heat_flux))
        / (1025.0 * 3990.0 * 30.0)
        * 3600.0
    )
    assert_allclose_compact(out, expected)

    gradient = jax.grad(
        lambda sensible: jnp.sum(
            _advance_sea_surface_temperature(
                sst,
                sensible,
                latent_heat_flux,
                3600.0,
                1025.0,
                3990.0,
                30.0,
                1.0 / (30.0 * 86400.0),
                288.15,
            )
        )
    )(sensible_heat_flux)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_land_kernel_supports_jit_and_clipping() -> None:
    soil_moisture = jnp.asarray([[0.3, 0.2], [0.1, 0.9]])
    latent_heat_flux = jnp.asarray([[100.0, 200.0], [1e9, -1e9]])

    out = jax.jit(_update_soil_moisture)(soil_moisture, latent_heat_flux, 10.0)

    expected = np.clip(
        np.asarray(soil_moisture) - 1e-9 * np.asarray(latent_heat_flux) * 10.0,
        0.0,
        1.0,
    )
    assert_allclose_compact(out, expected)

    gradient = jax.grad(
        lambda sm: jnp.sum(_update_soil_moisture(sm, jnp.full((2, 2), 100.0), 10.0))
    )(jnp.full((2, 2), 0.3))
    assert_allclose_compact(gradient, np.ones((2, 2)))


def test_seaice_kernel_supports_jit_and_gradient() -> None:
    sea_surface_temperature = jnp.asarray([[270.0, 272.0], [274.0, 276.0]])
    out = jax.jit(_diagnose_ice_fraction)(sea_surface_temperature)

    cold = float(out[0, 0])
    warm = float(out[1, 1])
    assert cold > warm
    assert 0.0 < warm < 1.0

    gradient = jax.grad(lambda sst: jnp.sum(_diagnose_ice_fraction(sst)))(
        sea_surface_temperature
    )
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) < 0.0)
