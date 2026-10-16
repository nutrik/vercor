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


@pytest.mark.fast_always
@pytest.mark.parametrize("restore_temp", [False, True])
@pytest.mark.parametrize("restore_salt", [False, True])
def test_native_restoring_controls_have_independent_values_and_gradients(
    restore_temp: bool,
    restore_salt: bool,
) -> None:
    """Check both native switches before any coupled forcing is prepared."""

    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
        require_differentiable_veros_capabilities,
    )

    configure_veros_runtime("jax")
    require_differentiable_veros_capabilities()
    from vercor.setups._external.veros_setup_jax import DifferentiableGlobalFourDegree

    model = DifferentiableGlobalFourDegree(
        restore_to_climatology_temp=restore_temp,
        restore_to_climatology_salt=restore_salt,
    )
    state = model.state
    with state.settings.unlock():
        model.set_parameter(state)
        state.settings.nx, state.settings.ny, state.settings.nz = 3, 2, 1
    state.initialize_variables()
    with state.variables.unlock():
        vs = state.variables
        vs.maskT = jnp.ones_like(vs.maskT)
        vs.sst_clim = jnp.full_like(vs.sst_clim, 12.0)
        vs.sss_clim = jnp.full_like(vs.sss_clim, 36.0)
        vs.qnet = jnp.full_like(vs.qnet, 5.0)
        vs.qnec = jnp.full_like(vs.qnec, 4.0)
        vs.dzt = jnp.full_like(vs.dzt, 50.0)
        vs.coupled_salt_flux = jnp.full_like(vs.coupled_salt_flux, -3.47e-6)

    def forcing(values: jax.Array) -> jax.Array:
        current = state.copy()
        with current.variables.unlock():
            current.variables.temp = jnp.full_like(current.variables.temp, values[0])
            current.variables.salt = jnp.full_like(current.variables.salt, values[1])
        model.set_forcing(current)
        return jnp.stack(
            (
                current.variables.forc_temp_surface[2, 2],
                current.variables.forc_salt_surface[2, 2],
            )
        )

    values = jnp.asarray([10.0, 35.0])
    heat_capacity = 3991.86795711963 * state.settings.rho_0
    expected = jnp.asarray(
        [
            (13.0 if restore_temp else 5.0) / heat_capacity,
            -3.47e-6 + (50.0 / 2592000.0 if restore_salt else 0.0),
        ]
    )
    expected_jacobian = jnp.asarray(
        [
            [-4.0 / heat_capacity if restore_temp else 0.0, 0.0],
            [0.0, -50.0 / 2592000.0 if restore_salt else 0.0],
        ]
    )
    np.testing.assert_allclose(jax.jit(forcing)(values), expected, rtol=1e-12)
    forward = jax.jacfwd(forcing)(values)
    reverse = jax.jacrev(forcing)(values)
    step = 1e-3
    finite_difference = jnp.stack(
        [
            (forcing(values + step * axis) - forcing(values - step * axis)) / (2 * step)
            for axis in jnp.eye(2)
        ],
        axis=1,
    )
    np.testing.assert_allclose(forward, expected_jacobian, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(finite_difference, forward, rtol=1e-8, atol=1e-16)


@pytest.mark.fast_always
def test_global_setup_keeps_turbulence_parameters_in_native_variables() -> None:
    """Catch fork revisions that still assign variable-owned TKE settings."""

    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
        require_differentiable_veros_capabilities,
    )

    configure_veros_runtime("jax")
    try:
        require_differentiable_veros_capabilities()
    except RuntimeError as error:
        pytest.skip(str(error))
    from vercor.setups._external.veros_setup_jax import (
        DifferentiableGlobalFourDegree,
    )

    model = DifferentiableGlobalFourDegree()
    state = model.state
    with state.settings.unlock():
        model.set_parameter(state)
    state.initialize_variables()

    assert state.settings.enable_tke
    assert float(state.variables.c_k) == pytest.approx(0.1)
    assert float(state.variables.c_eps) == pytest.approx(0.7)


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


@pytest.mark.fast_always
def test_coupled_precipitation_reaches_native_salt_with_matching_jvp_vjp() -> None:
    """Follow precipitation through adapter substeps and native tracer mixing."""

    from types import SimpleNamespace

    from vercor.components import StepContext
    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
        require_differentiable_veros_capabilities,
    )

    configure_veros_runtime("jax")
    try:
        require_differentiable_veros_capabilities()
    except RuntimeError as error:
        pytest.skip(str(error))
    from vercor.setups._external.veros_runtime import step_veros_runtime
    from vercor.setups._external.veros_setup_jax import DifferentiableGlobalFourDegree
    from veros.core.thermodynamics import vertmix_tempsalt

    model = DifferentiableGlobalFourDegree()
    state = model.state
    with state.settings.unlock():
        model.set_parameter(state)
        state.settings.nx, state.settings.ny, state.settings.nz = 3, 2, 1
        state.settings.dt_tracer = 20.0
    state.initialize_variables()
    with state.variables.unlock():
        vs = state.variables
        vs.maskT = jnp.ones_like(vs.maskT)
        vs.kbot = jnp.ones_like(vs.kbot)
        vs.temp = jnp.full_like(vs.temp, 10.0)
        vs.salt = jnp.full_like(vs.salt, 35.0)
        vs.dzt = jnp.full_like(vs.dzt, 50.0)
        vs.dzw = jnp.full_like(vs.dzw, 50.0)

    def native_tracer_step(current: Any) -> Any:
        next_state = current.copy()
        model.set_forcing(next_state)
        with next_state.variables.unlock():
            next_state.variables.update(vertmix_tempsalt(next_state))
            vs = next_state.variables
            vs.salt = vs.salt.at[..., vs.tau].set(vs.salt[..., vs.taup1])
            vs.time = vs.time + next_state.settings.dt_tracer
        return next_state

    resources = SimpleNamespace(
        execution="jax",
        uses_atmosphere_forcing=True,
        model_substeps=2,
        _step_function=native_tracer_step,
    )
    fields = {
        name: jnp.full((2, 3), value)
        for name, value in {
            "model_level_height": 50.0,
            "u_velocity": 2.0,
            "v_velocity": 3.0,
            "potential_temperature": 283.15,
            "specific_humidity": 0.005,
            "density": 1.2,
            "temperature": 283.15,
            "net_shortwave_radiation_flux": 0.0,
            "downward_longwave_radiation_flux": 300.0,
            "snow_rate": 0.0,
        }.items()
    }
    context = StepContext(
        dt_seconds=40.0, time=None, dtype=DTypePolicy(enable_x64=True)
    )

    @jax.jit
    def objective(rain: jax.Array) -> jax.Array:
        def couple(current: Any, _: Any) -> tuple[Any, None]:
            result = step_veros_runtime(
                cast(Any, resources),
                {**fields, "rain_rate": jnp.full((2, 3), rain)},
                context,
                current,
            )
            return result.payload, None

        final, _ = jax.lax.scan(couple, state, xs=None, length=3)
        return cast(jax.Array, final.variables.salt[2, 2, -1, final.variables.tau])

    # Six 20-second native steps in a 50 m top cell: dS/dR = -0.08328.
    rain = jnp.asarray(1e-4)
    value, tangent = jax.jvp(objective, (rain,), (jnp.asarray(1.0),))
    gradient = jax.grad(objective)(rain)
    dry = objective(jnp.asarray(0.0))
    finite_difference = (objective(rain + 1e-5) - objective(rain - 1e-5)) / 2e-5
    np.testing.assert_allclose(value - dry, -8.328e-6, atol=1e-12)
    np.testing.assert_allclose(tangent, -0.08328, rtol=1e-10)
    np.testing.assert_allclose(gradient, -0.08328, rtol=1e-10)
    np.testing.assert_allclose(gradient, finite_difference, rtol=1e-7)


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

    value, pullback = jax.vjp(objective, parameter_array)
    # For a scalar objective, a unit VJP is the reverse-mode gradient.
    (gradient,) = pullback(jnp.ones_like(value))
    jvp_value, jvp_tangent = jax.jvp(
        objective,
        (parameter_array,),
        (tangent,),
    )
    values = np.asarray(
        jax.device_get(jnp.stack((value, gradient, jvp_value, jvp_tangent)))
    )
    assert np.all(np.isfinite(values))
    assert float(abs(gradient)) > 0.0
    np.testing.assert_allclose(
        jvp_tangent,
        tangent * gradient,
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
