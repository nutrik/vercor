"""Reverse-mode acceptance tests for differentiable Veros rollouts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from functools import lru_cache
import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.components import Component
from vercor.dtypes import DTypePolicy
from vercor.recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    Spinup,
    VerosConfig,
    make_jcm_land_atmosphere,
    make_veros_gcm,
)
from vercor.state import RunState
from vercor.topology import SurfaceMaskPolicy

pytestmark = [pytest.mark.slow, pytest.mark.veros_autodiff]


def test_global_zero_surface_stress_forcing_has_finite_derivatives() -> None:
    """The JAX global setup keeps calm-wind TKE forcing differentiable."""

    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
        require_differentiable_veros_capabilities,
    )

    configure_veros_runtime("jax")
    try:
        require_differentiable_veros_capabilities()
    except RuntimeError as error:
        pytest.skip(str(error))
    setup_module = importlib.import_module("vercor.setups._external.veros_setup")
    surface_tke_forcing = setup_module._surface_tke_forcing

    def objective(stress: jax.Array) -> jax.Array:
        return cast(
            jax.Array,
            surface_tke_forcing(stress, stress, jnp.asarray(1024.0)),
        )

    calm_stress = jnp.asarray(0.0, dtype=jnp.float64)
    primal, tangent = jax.jvp(
        objective,
        (calm_stress,),
        (jnp.ones_like(calm_stress),),
    )
    gradient = jax.grad(objective)(calm_stress)

    assert bool(jnp.isfinite(primal))
    assert bool(jnp.isfinite(tangent))
    assert bool(jnp.isfinite(gradient))


def _clock(steps: int) -> Clock:
    return Clock(
        start=datetime(2000, 1, 3),
        dt_seconds=86400.0,
        steps=steps,
        calendar="noleap",
    )


def _jax_runtime(*, with_topology: bool = False) -> RuntimeOptions:
    return RuntimeOptions(
        backend="jax",
        dtype=DTypePolicy(enable_x64=True),
        topology=SurfaceMaskPolicy() if with_topology else None,
    )


@lru_cache(maxsize=1)
def _acc_component() -> Component:
    try:
        return cast(
            Component,
            make_veros_gcm(
                config=VerosConfig(setup="acc", execution="jax"),
            ),
        )
    except RuntimeError as error:
        if "differentiable Veros fork" in str(error):
            pytest.skip(str(error))
        raise


def _ocean_coupler(component: Component, *, steps: int) -> Coupler:
    return Coupler(
        _clock(steps),
        components=(component,),
        run_order=("OCN",),
        runtime=_jax_runtime(),
        log_level="WARNING",
    )


def _ocean_payload(state: RunState) -> Any:
    """Return the required opaque native ocean payload."""

    payload = state._component_state("OCN").payload
    assert payload is not None
    return payload


def _replace_native_veros_variable(
    native_state: Any,
    name: str,
    value: jax.Array,
) -> Any:
    """Copy a native Veros state and replace one test-controlled variable."""

    updated_state = native_state.copy()
    with updated_state.variables.unlock():
        setattr(updated_state.variables, name, value)
    return updated_state


def _replace_ocean_surface_temperature(
    state: RunState,
    temperature: jax.Array,
) -> RunState:
    component_state = state._component_state("OCN")
    native_state = _ocean_payload(state)
    baseline = jnp.asarray(native_state.variables.temp)
    surface_mask = jnp.asarray(native_state.variables.maskT[:, :, -1]) > 0.0
    seeded_surface = jnp.where(
        surface_mask[..., jnp.newaxis],
        temperature,
        baseline[:, :, -1, :],
    )
    seeded_temperature = baseline.at[:, :, -1, :].set(seeded_surface)
    seeded_payload = _replace_native_veros_variable(
        native_state,
        "temp",
        seeded_temperature,
    )
    return state._with_component_state(
        "OCN",
        component_state.with_payload(seeded_payload),
    )


def _replace_ocean_parameter(
    state: RunState,
    name: str,
    value: jax.Array,
) -> RunState:
    component_state = state._component_state("OCN")
    seeded_payload = _replace_native_veros_variable(
        _ocean_payload(state),
        name,
        value,
    )
    return state._with_component_state(
        "OCN",
        component_state.with_payload(seeded_payload),
    )


def _assert_payload_structure_stable(initial: RunState, final: RunState) -> None:
    assert cast(object, jax.tree_util.tree_structure(_ocean_payload(final))) == cast(
        object,
        jax.tree_util.tree_structure(_ocean_payload(initial)),
    )


def _assert_scalar_rollout_autodiff(
    objective: Callable[[jax.Array], jax.Array],
    parameter: float,
    *,
    finite_difference_step: float | None,
    relative_tolerance: float,
    inner_product_tolerance: float = 1e-6,
) -> None:
    """Validate scalar rollout AD, optionally including a centered difference."""

    parameter_array = jnp.asarray(parameter, dtype=jnp.float64)
    tangent = jnp.asarray(0.375, dtype=jnp.float64)

    value = objective(parameter_array)
    gradient = jax.grad(objective)(parameter_array)
    jvp_value, jvp_tangent = jax.jvp(
        objective,
        (parameter_array,),
        (tangent,),
    )
    vjp_value, pullback = jax.vjp(objective, parameter_array)
    (vjp_gradient,) = pullback(jnp.ones_like(vjp_value))

    values = np.asarray(
        jax.device_get(
            jnp.stack(
                (value, gradient, jvp_value, jvp_tangent, vjp_value, vjp_gradient)
            )
        )
    )
    assert np.all(np.isfinite(values))
    assert float(abs(gradient)) > 0.0
    np.testing.assert_allclose(
        jvp_tangent,
        tangent * vjp_gradient,
        rtol=inner_product_tolerance,
        atol=1e-12,
    )

    if finite_difference_step is None:
        return

    step = jnp.asarray(finite_difference_step, dtype=jnp.float64)
    centered_difference = (
        objective(parameter_array + step) - objective(parameter_array - step)
    ) / (2.0 * step)
    assert bool(jnp.isfinite(centered_difference))
    np.testing.assert_allclose(
        gradient,
        centered_difference,
        rtol=relative_tolerance,
        atol=1e-10,
    )


def test_three_step_acc_gradient_with_respect_to_surface_temperature() -> None:
    """ACC has finite agreeing derivatives through three scanned days."""

    coupler = _ocean_coupler(_acc_component(), steps=3)
    initial_state = coupler.initial_state()
    initial_variables = _ocean_payload(initial_state).variables
    assert bool(jnp.asarray(initial_variables.maskT)[18, 11, -1] > 0.0)

    def objective(temperature: jax.Array) -> jax.Array:
        final_state = coupler.run(
            _replace_ocean_surface_temperature(initial_state, temperature),
            output=None,
        )
        final_variables = _ocean_payload(final_state).variables
        final_temperature = jnp.asarray(final_variables.temp)
        return final_temperature[18, 11, -1, final_variables.tau]

    seeded_state = _replace_ocean_surface_temperature(
        initial_state,
        jnp.asarray(7.0, dtype=jnp.float64),
    )
    _assert_payload_structure_stable(seeded_state, coupler.run(seeded_state))
    _assert_scalar_rollout_autodiff(
        objective,
        7.0,
        finite_difference_step=1e-2,
        relative_tolerance=1e-3,
    )


def test_twenty_step_acc_gradient_with_respect_to_native_c_k() -> None:
    """Fork-owned ``c_k`` remains differentiable through twenty scanned days."""

    coupler = _ocean_coupler(_acc_component(), steps=20)
    initial_state = coupler.initial_state()
    initial_variables = _ocean_payload(initial_state).variables
    assert bool(jnp.asarray(initial_variables.maskT)[18, 2, -1] > 0.0)

    def objective(c_k: jax.Array) -> jax.Array:
        final_state = coupler.run(
            _replace_ocean_parameter(initial_state, "c_k", c_k),
            output=None,
        )
        final_variables = _ocean_payload(final_state).variables
        final_temperature = jnp.asarray(final_variables.temp)
        return final_temperature[18, 2, -1, final_variables.tau]

    seeded_state = _replace_ocean_parameter(
        initial_state,
        "c_k",
        jnp.asarray(0.1, dtype=jnp.float64),
    )
    _assert_payload_structure_stable(seeded_state, coupler.run(seeded_state))
    _assert_scalar_rollout_autodiff(
        objective,
        0.1,
        finite_difference_step=1e-3,
        relative_tolerance=1e-2,
        inner_product_tolerance=1e-2,
    )


@lru_cache(maxsize=1)
def _fully_coupled_global_model() -> Coupler:
    try:
        ocean = make_veros_gcm(
            config=VerosConfig(setup="global_4deg", execution="jax"),
        )
    except RuntimeError as error:
        if "differentiable Veros fork" in str(error):
            pytest.skip(str(error))
        raise
    jcm = make_jcm_land_atmosphere(
        ocean.grid,
        config=JCMLandAtmosphereConfig(
            atmosphere=JAXGCMConfig(
                spinup=Spinup(enabled=False),
                jitted=True,
            )
        ),
    )
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
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
    )
    return Coupler(
        _clock(5),
        components=(ocean, jcm.land, jcm.atmosphere),
        exchanges=exchanges,
        run_order=("OCN", "LND", "ATM"),
        runtime=_jax_runtime(with_topology=True),
        log_level="WARNING",
    )


def test_five_step_fully_coupled_global_adjoint() -> None:
    """The JAXGCM-land-global-Veros stack has agreeing JVP and VJP results."""

    coupler = _fully_coupled_global_model()
    initial_state = coupler.initial_state()

    def objective(c_k: jax.Array) -> jax.Array:
        final_state = coupler.run(
            _replace_ocean_parameter(initial_state, "c_k", c_k),
            output=None,
        )
        final_temperature = jnp.asarray(_ocean_payload(final_state).variables.temp)
        return jnp.mean(final_temperature[2:-2, 2:-2, :, :] ** 2)

    seeded_state = _replace_ocean_parameter(
        initial_state,
        "c_k",
        jnp.asarray(0.1, dtype=jnp.float64),
    )
    _assert_payload_structure_stable(seeded_state, coupler.run(seeded_state))
    _assert_scalar_rollout_autodiff(
        objective,
        0.1,
        finite_difference_step=None,
        relative_tolerance=1e-2,
    )
