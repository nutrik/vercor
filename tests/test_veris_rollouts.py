"""Compare real coupled native transitions, forcing reset and smooth adjoints."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_veris_component import cold_fields, small_coupler
from vercor.dtypes import DTypePolicy
from vercor.physics import PhysicalConstants
from vercor.state import RunState
from vercor.setups._external.veris_runtime import VerisPayload
from vercor.setups._external.veris_forcing import prepare_veris_forcing

pytestmark = pytest.mark.slow


def native_payload(state: RunState) -> VerisPayload:
    """Assert the wrapper retains the actual immutable native payload type."""
    payload = state._component_state("CICE").payload
    assert isinstance(payload, VerisPayload)
    return payload


def _direct_initial(geometry: Any, config: Any) -> tuple[Any, Any, Any]:
    veris = pytest.importorskip("veris")
    experiment = veris.Experiment(
        grid=veris.Grid(3, 2),
        geometry=geometry,
        settings={
            **dict(config.custom_parameters or {}),
            "dtype": "float64",
            "deltatDyn": config.model_timestep.total_seconds(),
            "deltatTherm": config.model_timestep.total_seconds(),
            "enable_cyclic_y": False,
        },
        initial_conditions=config.initial_conditions or {},
        output=None,
    )
    return cast(tuple[Any, Any, Any], veris.initialize_experiment(experiment))


@pytest.mark.fast_always
def test_two_wrapper_steps_equal_direct_native_advances() -> None:
    veris = pytest.importorskip("veris")
    coupler, component, geometry, config = small_coupler()
    initial = coupler.initial_state()
    original = native_payload(initial)
    state, conf, constants = _direct_initial(geometry, config)
    forcing = prepare_veris_forcing(
        cold_fields(),
        ocean_mask=component.grid.binary_mask,
        constants=PhysicalConstants(),
        dtype=DTypePolicy(enable_x64=True),
    )
    for _ in range(4):
        state, diagnostics = veris.advance(
            state, conf, constants, forcing, physics="coupled"
        )
    final = coupler.run(initial)
    actual = native_payload(final)
    for name in (
        "Area",
        "hIceMean",
        "hSnowMean",
        "TSurf",
        "uIce",
        "vIce",
        "Qnet",
        "Qsw",
    ):
        np.testing.assert_allclose(
            getattr(actual.state, name),
            getattr(state, name),
            rtol=1e-11,
            atol=1e-12,
            err_msg=name,
        )
    np.testing.assert_allclose(
        actual.diagnostics.EmPmR, diagnostics.EmPmR, rtol=1e-11, atol=1e-12
    )
    mask = np.asarray(component.grid.binary_mask)
    area = np.asarray(final.component("CICE").field("ice_fraction"))
    assert ((area >= 0) & (area <= 1)).all()
    np.testing.assert_array_equal(area[mask == 0], 0.0)
    np.testing.assert_array_equal(original.state.Area[2:-2, 2:-2].T, 0.5 * mask)
    assert not np.allclose(actual.state.hIceMean, original.state.hIceMean)
    assert cast(Any, jax.tree_util.tree_structure(actual)) == cast(
        Any, jax.tree_util.tree_structure(original)
    )
    assert all(
        x.dtype == jnp.float64
        for x in jax.tree_util.tree_leaves(
            (
                actual.state,
                actual.diagnostics,
                actual.physical_constants,
                actual.interval_budget,
            )
        )
    )
    assert actual.checked_native_steps.dtype == jnp.int32
    assert int(actual.checked_native_steps) == 4


def test_each_native_substep_reapplies_prescribed_heat_fluxes() -> None:
    veris = pytest.importorskip("veris")
    coupler, component, geometry, config = small_coupler(steps=1)
    state, conf, constants = _direct_initial(geometry, config)
    forcing = prepare_veris_forcing(
        cold_fields(),
        ocean_mask=component.grid.binary_mask,
        constants=PhysicalConstants(),
        dtype=DTypePolicy(enable_x64=True),
    )
    once, _ = veris.advance(state, conf, constants, forcing)
    correct, _ = veris.advance(once, conf, constants, forcing)
    retained_residual, _ = veris.advance(once, conf, constants, None)
    actual = native_payload(coupler.run()).state
    np.testing.assert_allclose(
        actual.hIceMean, correct.hIceMean, rtol=1e-11, atol=1e-12
    )
    assert (
        np.max(np.abs(np.asarray(correct.hIceMean - retained_residual.hIceMean)))
        > 1e-10
    )


@pytest.mark.parametrize("model_seconds", [150, 300])
def test_native_forcing_rollout_jvp_vjp_and_finite_difference(
    model_seconds: int,
) -> None:
    from vercor.setups import VerisConfig

    config = VerisConfig(
        model_timestep=timedelta(seconds=model_seconds),
        custom_parameters={"nEVPsteps": 8},
        initial_conditions={"Area": 0.5, "hIceMean": 0.25, "TSurf": 260.0},
    )
    coupler, _, _, _ = small_coupler(dt=300.0, steps=2, config=config)
    initial = coupler.initial_state()
    reference = initial.component("FORCING").field("temperature")

    def objective(delta: jax.Array) -> jax.Array:
        fields = {
            "temperature": reference + delta,
            "potential_temperature": reference + delta,
        }
        state = initial.replace_fields("FORCING", fields)
        result = native_payload(coupler.run(state)).state
        return jnp.mean(result.hIceMean[2:-2, 2:-2])

    x = jnp.array(0.0)
    value, forward = jax.jvp(jax.jit(objective), (x,), (jnp.ones_like(x),))
    reverse = jax.grad(objective)(x)
    fd = (objective(x + 1e-3) - objective(x - 1e-3)) / 2e-3
    assert np.isfinite([value, forward, reverse, fd]).all()
    assert abs(float(forward)) > 1e-10
    np.testing.assert_allclose([forward, reverse], fd, rtol=2e-4, atol=1e-10)


def test_zero_ice_wet_cells_receive_forcing_and_form_ice() -> None:
    from vercor.setups import VerisConfig

    config = VerisConfig(
        model_timestep=timedelta(seconds=300), custom_parameters={"nEVPsteps": 8}
    )
    coupler, component, _, _ = small_coupler(steps=2, config=config)
    initial = coupler.initial_state()
    np.testing.assert_array_equal(initial.component("CICE").field("ice_fraction"), 0.0)
    final = coupler.run(initial)
    wet = np.asarray(component.grid.binary_mask) > 0
    temperature = np.asarray(final.component("CICE").field("temperature"))
    np.testing.assert_allclose(temperature[wet], 260.0)
    area = np.asarray(final.component("CICE").field("ice_fraction"))
    assert (area[wet] > 0).any()
    np.testing.assert_array_equal(area[~wet], 0.0)
