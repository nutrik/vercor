"""Freshwater signs, reference-salinity conversion, and native Veros forcing."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.setups._external import veros_fluxes
from vercor.setups._external.veros_output import extract_veros_output_snapshot
from vercor.setups._external.veros_setup import CustomGlobalFourDegree


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("rain", "snow", "evap", "expected"),
    [
        (1e-4, 0.0, 0.0, -3.47e-6),
        (0.0, 1e-4, 0.0, -3.47e-6),
        (0.0, 0.0, -1e-4, 3.47e-6),
        (6e-5, 4e-5, -1e-4, 0.0),
        (1e-4, 2e-4, -1e-4, -6.94e-6),
    ],
)
def test_virtual_salt_flux_uses_downward_signed_mass_rates(
    rain: float, snow: float, evap: float, expected: float
) -> None:
    """Catch a reversed evaporation sign or a missing mass-to-depth conversion."""

    result = veros_fluxes._compute_virtual_salt_flux(
        PhysicalConstants(),
        jnp.full((2, 3), rain),
        jnp.full((2, 3), snow),
        jnp.full((2, 3), evap),
        jnp.ones((2, 3)),
    )
    assert_allclose_compact(result, np.full((2, 3), expected), atol=1e-20)


@pytest.mark.fast_always
def test_virtual_salt_flux_masks_missing_land_values_and_has_finite_gradients() -> None:
    """Inactive missing data must not poison either differentiation direction."""

    mask = jnp.asarray([[1.0, 0.0], [1.0, 0.0]])
    rain = jnp.asarray([[1e-4, jnp.nan], [2e-4, jnp.nan]])

    def objective(values: jax.Array) -> jax.Array:
        return jnp.sum(
            veros_fluxes._compute_virtual_salt_flux(
                PhysicalConstants(), values, values * 0.5, -values * 0.25, mask
            )
        )

    value, tangent = jax.jvp(objective, (rain,), (mask,))
    gradient = jax.grad(objective)(rain)
    assert_allclose_compact(value, -1.30125e-5)
    assert_allclose_compact(tangent, -0.08675)
    assert_allclose_compact(gradient, np.asarray([[-0.043375, 0], [-0.043375, 0]]))


@pytest.mark.fast_always
def test_virtual_salt_flux_traces_reference_salinity_and_freshwater_density() -> None:
    """Physical constants must remain differentiable rather than JIT-static."""

    def objective(parameters: jax.Array) -> jax.Array:
        constants = PhysicalConstants(
            surface_salinity_reference=parameters[0],
            freshwater_density=parameters[1],
        )
        return jnp.sum(
            veros_fluxes._compute_virtual_salt_flux(
                constants, jnp.asarray(1e-4), 0.0, 0.0, 1.0
            )
        )

    parameters = jnp.asarray([35.0, 1000.0])
    value, tangent = jax.jvp(objective, (parameters,), (jnp.ones(2),))
    gradient = jax.grad(objective)(parameters)
    step = 1e-3
    finite_difference = (
        objective(parameters + step) - objective(parameters - step)
    ) / (2 * step)
    assert_allclose_compact(value, -3.5e-6)
    assert_allclose_compact(gradient, np.asarray([-1e-7, 3.5e-9]))
    assert_allclose_compact(tangent, finite_difference, rtol=1e-8)


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True])
def test_virtual_salt_flux_obeys_runtime_precision(enable_x64: bool) -> None:
    """A float32 run must not promote new forcing arithmetic to float64."""

    policy = DTypePolicy(enable_x64=enable_x64)
    rain = jnp.full((2, 3), 1e-4, dtype=policy.jax_real)
    result = veros_fluxes._compute_virtual_salt_flux(
        PhysicalConstants(),
        rain,
        jnp.zeros_like(rain),
        jnp.zeros_like(rain),
        jnp.ones_like(rain),
        dtype=policy,
    )
    assert result.dtype == policy.jax_real
    assert_allclose_compact(result, np.full((2, 3), -3.47e-6), rtol=1e-6)


def _global_forcing_model(*, restore: bool = False) -> Any:
    """Build a real small native state without downloading global input data."""

    model = CustomGlobalFourDegree(restore_to_climatology=restore)
    state = model.state
    with state.settings.unlock():
        model.set_parameter(state)
        state.settings.nx, state.settings.ny, state.settings.nz = 3, 2, 1
        state.settings.dt_tracer = 20.0
    state.initialize_variables()
    vs = state.variables
    assert_allclose_compact(vs.coupled_salt_flux, np.zeros((7, 6)))
    with vs.unlock():
        vs.maskT = np.ones_like(vs.maskT)
        vs.temp = np.full_like(vs.temp, 10.0)
        vs.salt = np.full_like(vs.salt, 35.0)
        vs.sss_clim = np.full_like(vs.sss_clim, 36.0)
        vs.dzt = np.full_like(vs.dzt, 50.0)
        vs.dzw = np.full_like(vs.dzw, 50.0)
        vs.qnet = np.ones_like(vs.qnet)
        vs.qnec = np.zeros_like(vs.qnec)
        vs.coupled_salt_flux = np.full_like(vs.coupled_salt_flux, -3.47e-6)
    return model


@pytest.mark.fast_always
@pytest.mark.parametrize("restore", [False, True])
def test_native_callback_applies_coupled_flux_without_accumulating_restoring(
    restore: bool,
) -> None:
    """The actual host callback must call VerCOR's kernel on every substep."""

    model = _global_forcing_model(restore=restore)
    for _ in range(2):
        model.set_forcing(model.state)
        expected = -3.47e-6 + (50.0 / 2592000.0 if restore else 0.0)
        assert_allclose_compact(
            model.state.variables.forc_salt_surface,
            np.full((7, 6), expected),
        )


@pytest.mark.fast_always
def test_native_callback_masks_cold_cooling_water_and_land() -> None:
    """Adding freshwater must preserve the existing global setup's masks."""

    model = _global_forcing_model()
    vs = model.state.variables
    with vs.unlock():
        vs.temp[2, 2, -1, :] = -2.0
        vs.qnet[2, 2, :] = -100.0
        vs.maskT[3, 2, :] = 0.0
    model.set_forcing(model.state)
    assert float(vs.forc_salt_surface[2, 2]) == 0.0
    assert float(vs.forc_salt_surface[3, 2]) == 0.0
    assert float(vs.forc_salt_surface[4, 2]) == pytest.approx(-3.47e-6)


@pytest.mark.fast_always
def test_native_callback_fills_cyclic_halos_after_coupled_flux_assignment() -> None:
    """A regridded interior flux needs native periodic halo values."""

    model = _global_forcing_model()
    vs = model.state.variables
    with vs.unlock():
        vs.coupled_salt_flux = np.zeros_like(vs.coupled_salt_flux)
        vs.coupled_salt_flux[2:-2, 2:-2] = np.asarray(
            [[1e-6, 2e-6], [3e-6, 4e-6], [5e-6, 6e-6]]
        )
    model.set_forcing(model.state)
    assert_allclose_compact(
        vs.forc_salt_surface[:, 2],
        np.asarray([3e-6, 5e-6, 1e-6, 3e-6, 5e-6, 1e-6, 3e-6]),
    )


@pytest.mark.fast_always
def test_native_tracer_receives_virtual_salt_flux_in_top_cell_units() -> None:
    """Catch timestep or top-cell-depth factors applied twice in the adapter."""

    from veros.core.thermodynamics import vertmix_tempsalt

    model = _global_forcing_model()
    state = model.state
    with state.variables.unlock():
        state.variables.kbot = np.ones_like(state.variables.kbot)
    model.set_forcing(state)
    result = vertmix_tempsalt(state)
    assert_allclose_compact(
        result.salt[2:-2, 2:-2, -1, state.variables.taup1],
        np.full((3, 2), 34.999998612),
    )


@pytest.mark.fast_always
def test_vercor_output_reports_unambiguous_salinity_flux_units() -> None:
    """Correct state metadata must reach the coupler's output extraction path."""

    model = _global_forcing_model()
    model.set_forcing(model.state)
    sample = extract_veros_output_snapshot(model.state, ("forc_salt_surface",))
    assert sample["forc_salt_surface"].attrs["units"] == "m g kg-1 s-1"
    assert model.state.var_meta["forc_salt_surface"].units == "m g kg-1 s-1"
