from __future__ import annotations

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor.setups._slab.atmosphere import (
    _bulk_flux_step,
    _default_sea_surface_temperature,
    _surface_wind_10m,
)
from vercor.setups._slab.land import _update_soil_moisture
from vercor.setups._slab.ocean import _advance_sea_surface_temperature
from vercor.setups._slab.seaice import _diagnose_ice_fraction


@pytest.mark.fast_always
@pytest.mark.parametrize("anomaly", [-1.0, 1.0])
def test_ocean_zero_flux_restores_anomaly_and_analytic_derivatives(
    anomaly: float,
) -> None:
    """A one-day Euler step contracts either anomaly by 29/30."""

    def advance(sst: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            _advance_sea_surface_temperature(
                sst,
                0.0,
                0.0,
                86400.0,
                1025.0,
                3990.0,
                30.0,
                1.0 / (30.0 * 86400.0),
                288.15,
            ),
        )

    sst = jnp.asarray(288.15 + anomaly)
    actual, tangent = jax.jvp(advance, (sst,), (jnp.ones_like(sst),))
    _, pullback = jax.vjp(advance, sst)
    assert_allclose_compact(actual - 288.15, anomaly * 29.0 / 30.0)
    assert_allclose_compact(tangent, 29.0 / 30.0)
    assert_allclose_compact(pullback(jnp.ones_like(sst))[0], 29.0 / 30.0)
    epsilon = 1e-3
    finite_difference = (advance(sst + epsilon) - advance(sst - epsilon)) / (
        2 * epsilon
    )
    assert_allclose_compact(finite_difference, 29.0 / 30.0, rtol=1e-8)


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "air,expected_shf,expected_lhf", [(280.0, -100.0, -50.0), (300.0, 100.0, 50.0)]
)
def test_bulk_flux_transfers_heat_down_temperature_gradient(
    air: float,
    expected_shf: float,
    expected_lhf: float,
) -> None:
    sensible, latent, _ = _bulk_flux_step(jnp.asarray(air), jnp.asarray(290.0))
    assert_allclose_compact(sensible, expected_shf)
    assert_allclose_compact(latent, expected_lhf)
    # Reference equals SST so the direction comes solely from the received fluxes.
    updated = _advance_sea_surface_temperature(
        290.0,
        sensible,
        latent,
        86400.0,
        1025.0,
        3990.0,
        30.0,
        1.0 / (30.0 * 86400.0),
        290.0,
    )
    assert float(updated - 290.0) * expected_shf > 0.0


@pytest.mark.fast_always
@pytest.mark.parametrize("latent,expected", [(-100.0, 0.29136), (100.0, 0.30864)])
def test_soil_moisture_evaporation_dries_and_condensation_wets(
    latent: float,
    expected: float,
) -> None:
    assert_allclose_compact(_update_soil_moisture(0.3, latent, 86400.0), expected)


def test_soil_zero_flux_preserves_moisture() -> None:
    moisture = jnp.asarray([0.0, 0.3, 1.0])
    assert_allclose_compact(_update_soil_moisture(moisture, 0.0, 86400.0), moisture)


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
        10.0 * (np.asarray(temperature_2m) - np.asarray(sea_surface_temperature)),
    )
    assert_allclose_compact(latent_heat_flux, 0.5 * sensible_heat_flux)
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
    assert_allclose_compact(gradient, np.full((2, 2), -10.0))
    assert_finite_jvp_vjp(
        lambda sst: jnp.sum(_bulk_flux_step(jnp.full_like(sst, 288.0), sst)[0]),
        jnp.full((2, 2), 285.0),
        jnp.ones((2, 2)),
    )


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
    assert_finite_jvp_vjp(
        lambda surface_temperature: jnp.sum(
            _advance_sea_surface_temperature(
                surface_temperature,
                jnp.full_like(surface_temperature, 10.0),
                jnp.full_like(surface_temperature, 5.0),
                3600.0,
                1025.0,
                3990.0,
                30.0,
                1.0 / (30.0 * 86400.0),
                288.15,
            )
        ),
        jnp.full((2, 2), 288.0),
        jnp.ones((2, 2)),
    )


def test_land_kernel_supports_jit_and_clipping() -> None:
    soil_moisture = jnp.asarray([[0.3, 0.2], [0.1, 0.9]])
    latent_heat_flux = jnp.asarray([[100.0, 200.0], [1e9, -1e9]])

    out = jax.jit(_update_soil_moisture)(soil_moisture, latent_heat_flux, 10.0)

    expected = np.clip(
        np.asarray(soil_moisture) + 1e-9 * np.asarray(latent_heat_flux) * 10.0,
        0.0,
        1.0,
    )
    assert_allclose_compact(out, expected)

    gradient = jax.grad(
        lambda sm: jnp.sum(_update_soil_moisture(sm, jnp.full((2, 2), 100.0), 10.0))
    )(jnp.full((2, 2), 0.3))
    assert_allclose_compact(gradient, np.ones((2, 2)))
    assert_finite_jvp_vjp(
        lambda moisture: jnp.sum(
            _update_soil_moisture(
                moisture,
                jnp.full_like(moisture, 100.0),
                10.0,
            )
        ),
        jnp.full((2, 2), 0.3),
        jnp.ones((2, 2)),
    )


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
    assert_finite_jvp_vjp(
        lambda surface_temperature: jnp.sum(
            _diagnose_ice_fraction(surface_temperature)
        ),
        jnp.asarray([[270.0, 272.0], [274.0, 276.0]]),
        jnp.ones((2, 2)),
    )
