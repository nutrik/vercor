"""Check interval means, momentum face coverage and sticky native validity."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_veris_component import cold_fields, small_coupler
from tests.test_veris_rollouts import native_payload
from vercor.components import StepContext
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError

pytestmark = pytest.mark.fast_always


def _controlled_transition(
    monkeypatch: pytest.MonkeyPatch, *, failure: str = ""
) -> tuple[Any, Any, Any]:
    import veris
    import vercor.setups._external.veris_runtime as runtime
    from vercor.setups._external.veris_forcing import VerisForcingPack

    coupler, component, _, _ = small_coupler(steps=1)
    initial = native_payload(coupler.initial_state())
    resources = component.spec.lifecycle.setup.__self__.resources
    zero = jnp.zeros_like(initial.state.Area)
    initial = replace(initial, state=replace(initial.state, Qnet=zero))
    pack = VerisForcingPack({}, zero[2:-2, 2:-2], zero[2:-2, 2:-2])
    monkeypatch.setattr(
        runtime, "prepare_veris_forcing_pack", lambda *args, **kwargs: pack
    )

    def advance(
        state: Any, configuration: Any, constants: Any, forcing: Any, *, physics: str
    ) -> tuple[Any, Any]:
        first = jnp.all(state.Qnet == 0)
        heat = jnp.where(first, 10.0, 30.0)
        fraction = jnp.where(first, 0.25, 0.75)
        traction = jnp.where(first, 2.0, 4.0)
        updated = replace(
            state,
            Qnet=zero - heat,
            AreaW=zero + fraction,
            AreaS=zero + fraction,
            Area=zero,
            sigma1=jnp.where(first & (failure == "hidden_nan"), jnp.nan, zero),
        )
        diagnostics = replace(
            initial.diagnostics,
            OceanStressU=zero + traction,
            OceanStressV=zero + traction,
            ocean_mass_flux=zero + jnp.where(first, 2.0, 4.0),
            ocean_salt_mass_flux=zero + jnp.where(first, 0.001, 0.003),
            external_mass_flux=zero + jnp.where(first, 20.0, 40.0),
            external_heat_flux=zero
            + jnp.where(
                first & (failure == "overflow"), jnp.finfo(zero.dtype).max / 2, heat
            ),
            physics_status=zero + jnp.where(first & (failure == "status"), 2.0, 0.0),
        )
        return updated, diagnostics

    monkeypatch.setattr(veris, "advance", advance)
    context = StepContext(600.0, dtype=DTypePolicy(enable_x64=True))
    return initial, resources, context


def test_two_native_substeps_export_interval_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_runtime import step_veris_runtime

    initial, resources, context = _controlled_transition(monkeypatch)
    result = step_veris_runtime(resources, cold_fields(), context, initial)
    wet = np.asarray(initial.state.iceMask[2:-2, 2:-2].T)
    for name, expected in (
        ("sea_surface_heat_flux", 20.0),
        ("sea_surface_mass_flux", 3.0),
        ("sea_surface_salt_mass_flux", 0.002),
    ):
        np.testing.assert_allclose(result.fields[name], expected * wet, err_msg=name)
    np.testing.assert_allclose(
        result.payload.interval_budget.external_mass_flux, 30.0 * wet
    )
    np.testing.assert_allclose(
        result.payload.interval_budget.external_heat_flux, 20.0 * wet
    )
    np.testing.assert_allclose(result.payload.diagnostics.ocean_mass_flux, 4.0)


def test_stress_uses_mean_of_momentum_face_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_runtime import step_veris_runtime

    initial, resources, context = _controlled_transition(monkeypatch)
    result = step_veris_runtime(resources, cold_fields(), context, initial)
    wet = np.asarray(initial.state.iceMask[2:-2, 2:-2].T) > 0
    east = wet & np.roll(wet, -1, axis=1)
    north = wet & np.concatenate((wet[1:], np.zeros_like(wet[:1])), axis=0)
    np.testing.assert_allclose(result.fields["ocean_u_stress"], 1.75 * east)
    np.testing.assert_allclose(result.fields["ocean_v_stress"], 1.75 * north)
    np.testing.assert_array_equal(result.fields["ice_fraction"], 0.0)


@pytest.mark.parametrize("failure", ["status", "hidden_nan", "overflow"])
@pytest.mark.parametrize("compiled", [False, True])
def test_intermediate_native_failure_cannot_disappear(
    monkeypatch: pytest.MonkeyPatch, failure: str, compiled: bool
) -> None:
    from vercor.setups._external.veris_runtime import step_veris_runtime

    initial, resources, context = _controlled_transition(monkeypatch, failure=failure)

    def run(payload: Any) -> Any:
        return step_veris_runtime(resources, cold_fields(), context, payload)

    if compiled:
        run = jax.jit(run)
    with pytest.raises(
        (CouplerError, jax.errors.JaxRuntimeError), match="Veris.*valid"
    ):
        result = run(initial)
        jax.block_until_ready(result.payload)


def test_surface_fluxes_shift_tagged_faces_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vercor.setups._external.veris_fluxes import veris_surface_fluxes
    from vercor.setups._external.veris_forcing import VerisForcingPack

    initial, _, context = _controlled_transition(monkeypatch)
    state = initial.state
    zero = jnp.zeros_like(state.Area)
    u = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).T
    v = 10 * u
    state = replace(state, AreaW=zero + 0.25, AreaS=zero + 0.75, Area=zero)
    diag = replace(
        initial.diagnostics,
        OceanStressU=zero.at[2:-2, 2:-2].set(u),
        OceanStressV=zero.at[2:-2, 2:-2].set(v),
    )
    pack = VerisForcingPack({}, jnp.zeros_like(u), jnp.zeros_like(v))
    fluxes = veris_surface_fluxes(state, diag, pack, dtype=context.dtype)
    np.testing.assert_array_equal(
        fluxes.ocean_u_stress, [[0.0, 0.75, 0.0], [1.25, 1.5, 1.0]]
    )
    np.testing.assert_array_equal(
        fluxes.ocean_v_stress, [[0.0, 37.5, 45.0], [0.0, 0.0, 0.0]]
    )


@pytest.mark.parametrize("area", [0.0, 0.5, 1.0])
def test_direct_open_water_stress_uses_complementary_face_area(
    monkeypatch: pytest.MonkeyPatch, area: float
) -> None:
    from vercor.setups._external.veris_fluxes import veris_surface_fluxes
    from vercor.setups._external.veris_forcing import VerisForcingPack

    initial, _, context = _controlled_transition(monkeypatch)
    zero = jnp.zeros_like(initial.state.Area)
    state = replace(initial.state, AreaW=zero + area, AreaS=zero + area)
    diag = replace(
        initial.diagnostics, OceanStressU=zero + 2.0, OceanStressV=zero + 4.0
    )
    pack = VerisForcingPack({}, zero[2:-2, 2:-2] + 3.0, zero[2:-2, 2:-2] + 5.0)
    fluxes = veris_surface_fluxes(state, diag, pack, dtype=context.dtype)
    wet = np.asarray(state.iceMask[2:-2, 2:-2].T) > 0
    east = wet & np.roll(wet, -1, axis=1)
    north = wet & np.concatenate((wet[1:], np.zeros_like(wet[:1])), axis=0)
    np.testing.assert_allclose(
        fluxes.ocean_u_stress, (2 * area + 3 * (1 - area)) * east
    )
    np.testing.assert_allclose(
        fluxes.ocean_v_stress, (4 * area + 5 * (1 - area)) * north
    )


def test_feedback_recipe_declares_three_scalars_and_one_vector() -> None:
    from vercor.recipes import SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS
    from vercor.fields import vector

    assert SEAICE_TO_OCEAN_TOTAL_FLUX_FIELDS == (
        "sea_surface_heat_flux",
        "sea_surface_mass_flux",
        "sea_surface_salt_mass_flux",
        vector("ocean_u_stress", "ocean_v_stress"),
    )


@pytest.mark.slow
def test_complete_melt_exports_total_flux_and_closes_column_budgets() -> None:
    from datetime import timedelta
    from vercor.setups import VerisConfig

    coupler, _, _, _ = small_coupler(
        steps=1,
        config=VerisConfig(
            model_timestep=timedelta(seconds=300),
            custom_parameters={"nEVPsteps": 8},
            initial_conditions={"Area": 0.5, "hIceMean": 1e-4, "TSurf": 260.0},
        ),
    )
    initial = coupler.initial_state()
    forcing = cold_fields()
    for name, value in {
        "air_temperature": 290.0,
        "air_potential_temperature": 290.0,
        "sea_surface_temperature": 278.0,
        "air_u_velocity": 0.0,
        "air_v_velocity": 0.0,
        "u_velocity_10m": 0.0,
        "v_velocity_10m": 0.0,
        "net_shortwave_radiation_flux": 800.0,
        "downward_shortwave_radiation_flux": 800.0,
        "downward_longwave_radiation_flux": 350.0,
        "rain_rate": 1e-4,
        "snow_rate": 0.0,
    }.items():
        forcing[name] = jnp.full((2, 3), value)
    final = coupler.run(initial.replace_fields("FORCING", forcing))
    before, after = native_payload(initial), native_payload(final)
    fields = final.component("CICE")
    wet = np.asarray(before.state.iceMask[2:-2, 2:-2].T) > 0
    np.testing.assert_array_equal(fields.field("ice_fraction"), 0.0)
    assert np.all(np.asarray(fields.field("sea_surface_mass_flux"))[wet] > 0)
    p = after.physical_constants
    delta_ice = np.asarray(after.state.hIceMean - before.state.hIceMean)[2:-2, 2:-2].T
    delta_snow = np.asarray(after.state.hSnowMean - before.state.hSnowMean)[
        2:-2, 2:-2
    ].T
    delta_mass = p.rhoIce * delta_ice + p.rhoSnow * delta_snow
    mass_input = 600 * jnp.asarray(fields.field("sea_surface_mass_flux"))
    external_mass = 600 * jnp.asarray(after.interval_budget.external_mass_flux)
    salt_input = 600 * jnp.asarray(fields.field("sea_surface_salt_mass_flux"))
    delta_salt = p.rhoIce * p.saltIce_ref / 1000 * delta_ice
    heat_input = 600 * jnp.asarray(fields.field("sea_surface_heat_flux"))
    external_heat = 600 * jnp.asarray(after.interval_budget.external_heat_flux)
    for residual, scale in (
        (
            mass_input + delta_mass - external_mass,
            jnp.maximum(abs(delta_mass), abs(external_mass)),
        ),
        (salt_input + delta_salt, abs(delta_salt)),
        (
            heat_input - p.lhFusion * delta_mass - external_heat,
            jnp.maximum(abs(p.lhFusion * delta_mass), abs(external_heat)),
        ),
    ):
        assert (
            np.max(np.abs(np.asarray(residual)) / np.maximum(1.0, np.asarray(scale)))
            < 1e-10
        )


def test_public_mass_flux_has_consistent_parameter_derivatives() -> None:
    from vercor.setups import replace_veris_parameters

    coupler, _, _, _ = small_coupler(steps=1)
    initial = coupler.initial_state()

    def loss(h0: jax.Array) -> jax.Array:
        candidate = replace_veris_parameters(initial, {"h0": h0})
        return 1e6 * jnp.mean(
            coupler.run(candidate).component("CICE").field("sea_surface_mass_flux")
        )

    x = jnp.asarray(0.43, dtype=jnp.float64)
    value, tangent = jax.jvp(jax.jit(loss), (x,), (jnp.ones_like(x),))
    reverse = jax.grad(loss)(x)
    assert np.isfinite(value) and abs(float(tangent)) > 1e-8
    np.testing.assert_allclose(tangent, reverse, rtol=1e-9, atol=1e-10)
    differences = [
        (loss(x + step) - loss(x - step)) / (2 * step) for step in (1e-3, 1e-4, 1e-5)
    ]
    np.testing.assert_allclose(differences, reverse, rtol=2e-4, atol=1e-7)
