"""End-to-end h0 sensitivity through real prognostic Veris and Veros."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.setups import replace_veris_parameters

pytestmark = [pytest.mark.veros_autodiff, pytest.mark.slow, pytest.mark.fast_always]


def test_current_open_water_contribution_has_native_ice_ocean_derivatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime("jax")
    from tests.test_veris_prognostic_gallery import _prognostic_coupler

    coupler = _prognostic_coupler(
        monkeypatch, execution="jax", steps=2, export_open_water_surface=True
    )
    initial = coupler.initial_state()
    grid = initial.component("OCN").grid
    assert grid is not None
    wet = jnp.asarray(grid.binary_mask)

    def objective(h0: jax.Array) -> jax.Array:
        final = coupler.run(replace_veris_parameters(initial, {"h0": h0}))
        contribution = jnp.asarray(
            final.component("OCN").field("open_water_temperature_contribution")
        )
        return jnp.sum(contribution * wet) / jnp.sum(wet)

    point = jnp.asarray(0.43, dtype=jnp.float64)
    value, forward = jax.jvp(jax.jit(objective), (point,), (jnp.ones_like(point),))
    reverse = jax.grad(objective)(point)
    assert np.isfinite(value) and abs(float(forward)) > 1e-8
    np.testing.assert_allclose(forward, reverse, rtol=1e-8, atol=1e-10)
    differences = [
        (objective(point + h) - objective(point - h)) / (2 * h)
        for h in (1e-2, 1e-3, 1e-4)
    ]
    np.testing.assert_allclose(differences, reverse, rtol=2e-3, atol=1e-7)
    final = coupler.run(initial)
    np.testing.assert_allclose(
        final.component("OCN").field("open_water_temperature_contribution"),
        jnp.where(
            wet > 0,
            (1 - jnp.asarray(final.component("CICE").field("ice_fraction")))
            * jnp.asarray(final.component("OCN").field("sea_surface_temperature")),
            0,
        ),
    )


def test_shared_prognostic_h0_gradient_reaches_ocean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime("jax")
    from tests.test_veris_prognostic_gallery import _prognostic_coupler

    coupler = _prognostic_coupler(monkeypatch, execution="jax", steps=2)
    initial = coupler.initial_state()
    grid = initial.component("OCN").grid
    assert grid is not None
    wet = jnp.asarray(grid.binary_mask)
    original = jnp.asarray(initial.component("OCN").field("sea_surface_salinity"))

    def objective(h0: jax.Array) -> jax.Array:
        final = coupler.run(replace_veris_parameters(initial, {"h0": h0}))
        change = (
            jnp.asarray(final.component("OCN").field("sea_surface_salinity")) - original
        )
        return 1e6 * jnp.sum(change * wet) / jnp.sum(wet)

    h0 = jnp.asarray(0.43, dtype=jnp.float64)
    value, tangent = jax.jvp(jax.jit(objective), (h0,), (jnp.ones_like(h0),))
    reverse = jax.grad(objective)(h0)
    assert np.isfinite(value) and abs(float(tangent)) > 1e-8
    np.testing.assert_allclose(tangent, reverse, rtol=1e-8, atol=1e-10)
    finite = [
        (objective(h0 + step) - objective(h0 - step)) / (2 * step)
        for step in (1e-2, 1e-3, 1e-4)
    ]
    np.testing.assert_allclose(finite, reverse, rtol=2e-3, atol=1e-5)
    np.testing.assert_array_equal(
        initial.component("OCN").field("sea_surface_salinity"), original
    )
