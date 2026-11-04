"""Actual reference winds retain the converged CESM profile and current frame."""

import math
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_shared_surface_fluxes import surface_fields
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants

pytestmark = pytest.mark.fast_always


def test_model_level_config_declares_its_owned_forcing_inputs() -> None:
    from tests.test_veris_component import small_ocean
    from vercor.setups import VerisConfig, make_veris
    from vercor.recipes import (
        MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
        OCEAN_TO_SEAICE_FORCING_FIELDS,
    )
    from vercor.fields import _flatten_field_items

    config = VerisConfig(atmosphere_forcing="model_level")
    grid, geometry = small_ocean()
    component = make_veris(grid, geometry, config=config)
    assert component.spec.inputs == tuple(
        _flatten_field_items(
            (
                *MODEL_LEVEL_ATMOSPHERE_TO_SEAICE_FORCING_FIELDS,
                *OCEAN_TO_SEAICE_FORCING_FIELDS,
            )
        )
    )
    assert "u_velocity_10m" not in component.spec.inputs
    assert "net_shortwave_radiation_flux" not in component.spec.inputs


def bulk_winds(fields: dict[str, jax.Array]) -> tuple[jax.Array, jax.Array]:
    from vercor.setups._surface_fluxes import compute_surface_fluxes

    result = compute_surface_fluxes(
        fields,
        sea_surface_temperature=fields["sea_surface_temperature"],
        u_ocean=fields["ocean_u_velocity"],
        v_ocean=fields["ocean_v_velocity"],
        ocean_mask=jnp.ones((2, 3)),
        constants=PhysicalConstants(),
        dtype=DTypePolicy(enable_x64=True),
    )
    return result.u_velocity_10m, result.v_velocity_10m


def test_reference_wind_identity_and_frame() -> None:
    fields = surface_fields()
    actual = jnp.stack(bulk_winds(fields), axis=-1)
    np.testing.assert_allclose(
        actual, jnp.stack((fields["air_u_velocity"], fields["air_v_velocity"]), axis=-1)
    )
    fields["model_level_height"] = jnp.full((2, 3), 40.0)
    reference = jnp.stack(bulk_winds(fields), axis=-1)
    for air, water, shift in (
        ("air_u_velocity", "ocean_u_velocity", 3.0),
        ("air_v_velocity", "ocean_v_velocity", -2.0),
    ):
        fields[air] += shift
        fields[water] += shift
    shifted = jnp.stack(bulk_winds(fields), axis=-1)
    np.testing.assert_allclose(
        shifted - reference,
        jnp.broadcast_to(jnp.array([3.0, -2.0]), reference.shape),
        atol=1e-13,
    )


@pytest.mark.parametrize("hol", [0.2, 0.0, -0.2])
def test_reference_wind_matches_analytic_stability_profiles(hol: float) -> None:
    from vercor.fluxes.bulk_formula_cesm import _reference_height_wind

    def psi(zeta: float) -> float:
        if zeta > 0:
            return -5 * zeta
        if zeta == 0:
            # Native psimhu uses 1.571 as its integration constant.
            return 0.5 * (1.571 - math.pi / 2)
        x = (1 - 16 * zeta) ** 0.25
        return math.log((1 + x * (2 + x)) * (1 + x * x) / 8) - 2 * math.atan(x) + 1.571

    expected_factor = 1 - 0.1 * (math.log(4) - psi(hol) + psi(hol / 4))
    result = _reference_height_wind(
        PhysicalConstants(),
        jnp.array(9.0),
        jnp.array(3.0),
        jnp.array(1.0),
        jnp.array(-1.0),
        jnp.array(40.0),
        jnp.array(10.0),
        jnp.array(0.04),
        jnp.array(hol),
        jnp.array(psi(hol)),
        jnp.array(1.0 if hol > 0 else 0.5 if hol == 0 else 0.0),
        jnp.array(1.0),
    )
    np.testing.assert_allclose(
        result, (1 + 8 * expected_factor, -1 + 4 * expected_factor), rtol=1e-13
    )


def test_reference_winds_calm_and_masked_nan_gradients() -> None:
    from vercor.fluxes.bulk_formula_cesm import _reference_height_wind

    wet = jnp.array([[1, 0, 1], [0, 1, 1]])

    def evaluate(speed: jax.Array) -> jax.Array:
        clean = jnp.where(wet > 0, speed, jnp.nan)
        zero = jnp.where(wet > 0, 0.0, jnp.nan)
        u, v = _reference_height_wind(
            PhysicalConstants(),
            clean,
            zero,
            zero,
            zero,
            jnp.where(wet > 0, 40.0, jnp.nan),
            jnp.array(10.0),
            jnp.where(wet > 0, 0.04, jnp.nan),
            zero,
            zero,
            zero,
            wet,
        )
        return cast(jax.Array, (u + v).sum())

    point = jnp.array(0.0)
    value, tangent = jax.jvp(jax.jit(evaluate), (point,), (jnp.array(1.0),))
    gradient = jax.grad(evaluate)(point)
    assert float(value) == 0
    assert bool(jnp.isfinite(tangent) & jnp.isfinite(gradient))
    np.testing.assert_allclose(tangent, gradient, rtol=1e-13)
    np.testing.assert_allclose(
        gradient, (evaluate(point + 1e-4) - evaluate(point - 1e-4)) / 2e-4, rtol=1e-12
    )
    fields = surface_fields()
    fields["air_u_velocity"] = fields["ocean_u_velocity"]
    fields["air_v_velocity"] = fields["ocean_v_velocity"]
    u, v = bulk_winds(fields)
    np.testing.assert_allclose(u, fields["air_u_velocity"])
    np.testing.assert_allclose(v, fields["air_v_velocity"])
