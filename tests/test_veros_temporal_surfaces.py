"""Native pressure endpoint and physical coupled timestep regressions.

Stock and differentiable Veros have process-wide backend settings. The normal
suite exercises stock NumPy; a fresh process can set VERCOR_TEMPORAL_VEROS_JAX=1
for the exact differentiable fork without changing installed packages.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import os
from types import SimpleNamespace
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._veros_surface_support import surface_model, surface_setup_class
from tests.assertions import assert_allclose_compact

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("substeps", [1, 2, 3])
def test_runtime_exports_mean_surfaces_and_retains_endpoint(
    execution: Any, substeps: int, caplog: pytest.LogCaptureFixture
) -> None:
    from veros.core.operators import numpy as npx
    from vercor.components import StepContext
    from vercor.setups._external.veros_runtime import step_veros_runtime
    from vercor.setups._external.veros_state import copy_state, pure
    from tests.test_veros_ice_feedback import feedback_fields

    model = surface_model(execution=execution)
    original = model.state
    wet = npx.asarray(original.variables.maskT[2:-2, 2:-2, -1]).T

    def native_step(current: Any, scale: Any) -> Any:
        def transition(state: Any) -> None:
            vs = state.variables
            with vs.unlock():
                index = vs.itt + 1
                vs.temp = npx.full_like(vs.temp, -2 + scale * index**2)
                vs.salt = npx.full_like(vs.salt, 30 + 2 * scale * index)
                vs.u = npx.full_like(vs.u, scale * index**2)
                vs.v = npx.full_like(vs.v, -scale * index)
                vs.ssh = npx.full_like(vs.ssh, 2 * scale * index)
                vs.itt = index
                vs.time = vs.time + 600

        return pure(
            current, step=transition, linear_solver=object(), execution=execution
        )

    advance = jax.jit(native_step) if execution == "jax" else native_step
    resources: Any = SimpleNamespace(
        execution=execution,
        uses_bulk_meteorology=False,
        surface_forcing="surface_fluxes",
        model_substeps=substeps,
        export_open_water_surface=True,
        grid=SimpleNamespace(binary_mask=wet),
    )
    forcing = feedback_fields()
    forcing["ice_fraction"] = jnp.full(wet.shape, 0.25)
    logger = logging.getLogger("VerCOR.test.veros-substeps")
    caplog.set_level(logging.INFO, logger=logger.name)
    context = StepContext(substeps * 600, time=datetime(2000, 1, 1), logger=logger)

    def run(scale: Any, payload: Any = original) -> Any:
        resources._step_function = lambda current: advance(current, scale)
        return step_veros_runtime(
            resources, forcing, context, copy_state(payload, execution=execution)
        )

    result = run(1.0)
    first = (substeps + 1) / 2
    second = (substeps + 1) * (2 * substeps + 1) / 6
    masks = original.variables
    expected = {
        "sea_surface_temperature": 271.15 + second,
        "sea_surface_salinity": (30 + 2 * first) * wet,
        "ocean_u_velocity": second * npx.asarray(masks.maskU[2:-2, 2:-2, -1]).T,
        "ocean_v_velocity": -first * npx.asarray(masks.maskV[2:-2, 2:-2, -1]).T,
        "sea_surface_height": 2 * first * wet,
        "open_water_temperature_contribution": 0.75 * (271.15 + second) * wet,
    }
    for name, value in expected.items():
        actual = np.asarray(result.fields[name])
        if name == "sea_surface_temperature":
            # The canonical SST extractor retains its established land fill.
            actual = actual[np.asarray(wet) > 0]
        assert_allclose_compact(actual, value, rtol=1e-12, atol=1e-12)
    assert int(result.payload.variables.itt) == substeps
    assert_allclose_compact(result.payload.variables.temp, -2 + substeps**2)
    assert int(original.variables.itt) == 0
    continued = run(1.0, result.payload)
    indices = np.arange(substeps + 1, 2 * substeps + 1)
    assert_allclose_compact(
        np.asarray(continued.fields["sea_surface_temperature"])[np.asarray(wet) > 0],
        271.15 + np.mean(indices**2),
    )
    if execution == "jax":

        def objective(scale: Any) -> Any:
            return jnp.sum(run(scale).fields["sea_surface_height"])

        point = jnp.asarray(1.0)
        _, tangent = jax.jvp(objective, (point,), (jnp.ones_like(point),))
        reverse = jax.grad(objective)(point)
        assert_allclose_compact(tangent, 2 * first * np.sum(wet))
        assert_allclose_compact(reverse, tangent)
    assert not [record for record in caplog.records if record.name == logger.name]


@pytest.fixture(scope="module")
def execution() -> Literal["host", "jax"]:
    pytest.importorskip("veros")
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    lane: Literal["host", "jax"] = (
        "jax" if os.environ.get("VERCOR_TEMPORAL_VEROS_JAX") == "1" else "host"
    )
    configure_veros_runtime(lane)
    jax.config.update("jax_enable_x64", True)
    return lane


@pytest.mark.parametrize("substeps", [1, 3])
def test_native_pressure_update_exports_current_endpoint(
    execution: Any, substeps: int
) -> None:
    """Use native pressure correction followed by the native driver's rotation."""
    from veros.core.external.solve_pressure import barotropic_velocity_update
    from veros.core.operators import numpy as npx, update, at
    from vercor.setups._external.veros_state import (
        extract_veros_runtime_surface_fields,
        pure,
    )

    model = surface_model(execution=execution)
    state = model.state
    # Surface pressure has geopotential units only in the pressure formulation.
    assert state.var_meta["psi"].units == "m^2/s^2"
    assert state.var_meta["ssh"].units == "m"
    original = np.asarray(state.variables.psi).copy()

    def native_pressure_step(current: Any, scale: Any) -> None:
        vs = current.variables
        with vs.unlock():
            # A nonuniform current solution differs from the previous slot.
            pressure = (
                scale * (vs.itt + 1.0) * npx.arange(vs.ssh.size).reshape(vs.ssh.shape)
            )
            vs.psi = update(vs.psi, at[..., vs.taup1], pressure)
            vs.update(barotropic_velocity_update(current))
            vs.itt = vs.itt + 1
            vs.time = vs.time + current.settings.dt_tracer
            vs.taum1, vs.tau, vs.taup1 = vs.tau, vs.taup1, vs.taum1

    def advance(current: Any, scale: Any) -> Any:
        return pure(
            current,
            step=lambda value: native_pressure_step(value, scale),
            linear_solver=object(),
            execution=execution,
            synchronize_surface_height=True,
        )

    advance_step = jax.jit(advance) if execution == "jax" else advance
    for _ in range(substeps):
        state = advance_step(state, 1.0)
    vs = state.variables
    expected = np.asarray(vs.psi[..., vs.tau]) / state.settings.grav
    np.testing.assert_allclose(vs.ssh, expected, rtol=1e-14)
    wet = np.asarray(vs.maskT[2:-2, 2:-2, -1]).T
    exported = extract_veros_runtime_surface_fields(state, coupled=True)
    np.testing.assert_allclose(
        exported["sea_surface_height"], expected[2:-2, 2:-2].T * wet
    )
    np.testing.assert_array_equal(
        exported["ocean_u_velocity"],
        np.asarray(vs.u[2:-2, 2:-2, -1, vs.tau]).T
        * np.asarray(vs.maskU[2:-2, 2:-2, -1]).T,
    )
    assert float(vs.time) == substeps * state.settings.dt_tracer
    if execution == "jax":
        np.testing.assert_array_equal(model.state.variables.psi, original)

        def height_sum(scale: jax.Array) -> jax.Array:
            current = advance_step(model.state, scale)
            return jnp.sum(
                extract_veros_runtime_surface_fields(current, coupled=True)[
                    "sea_surface_height"
                ]
            )

        scale = jnp.asarray(1.0)
        primal, tangent = jax.jvp(height_sum, (scale,), (jnp.ones_like(scale),))
        reverse = jax.grad(height_sum)(scale)
        expected_derivative = (
            np.sum(np.arange(vs.ssh.size).reshape(vs.ssh.shape)[2:-2, 2:-2].T * wet)
            / state.settings.grav
        )
        np.testing.assert_allclose(primal, expected_derivative, rtol=1e-13)
        np.testing.assert_allclose(tangent, expected_derivative, rtol=1e-13)
        np.testing.assert_allclose(reverse, expected_derivative, rtol=1e-13)


@pytest.mark.parametrize(
    "mode,parameters,expected",
    [
        ("surface_fluxes", None, (1800.0, 1800.0)),
        ("surface_fluxes", {"dt_mom": 300.0}, (300.0, 300.0)),
        ("surface_fluxes", {"dt_tracer": 86400.0}, (1800.0, 86400.0)),
        ("surface_fluxes", {"dt_mom": 300.0, "dt_tracer": 600.0}, (300.0, 600.0)),
        ("bulk_meteorology", None, (600.0, 600.0)),
    ],
)
def test_coupled_default_uses_physical_time_with_explicit_acceleration_supported(
    monkeypatch: pytest.MonkeyPatch,
    execution: Any,
    mode: Any,
    parameters: Any,
    expected: tuple[float, float],
) -> None:
    from vercor.setups._external import veros_gcm_state

    setup_class = surface_setup_class(execution=execution)
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *_: setup_class
    )
    resources = veros_gcm_state.VerosGCMSetupState(
        execution=execution,
        surface_forcing=mode,
        custom_parameters=parameters,
    )
    settings = resources._veros_state.settings
    assert (settings.dt_mom, settings.dt_tracer) == expected
    from vercor.setups._time_helpers import assign_model_timestep_alignment

    assign_model_timestep_alignment(
        resources,
        86400.0,
        timedelta(seconds=resources.dt_tracer),
        coupling_name="dt",
        model_name="dt_tracer",
    )
    assert resources.model_substeps == int(86400.0 / expected[1])


@pytest.mark.parametrize("tracer_dt", [600.0, 86400.0])
def test_actual_native_stress_uses_momentum_duration(
    execution: Any, tracer_dt: float
) -> None:
    """Native acceleration changes physical impulse per tracer-clock second."""
    from veros.core import momentum
    from veros.core.external.solve_pressure import prepare_forcing
    from veros.core.operators import numpy as npx

    model = surface_model(execution=execution, dt=600.0, temperature=0.0)
    state = model.state
    with state.settings.unlock():
        state.settings.dt_tracer = tracer_dt
    vs = state.variables
    with vs.unlock():
        vs.surface_taux = npx.full_like(vs.surface_taux, 0.2)
        vs.surface_tauy = npx.zeros_like(vs.surface_tauy)
        vs.update(momentum.tend_tauxyf(state))
        # Constant tendency in both AB history slots removes startup weighting.
        vs.du = npx.repeat(vs.du[..., vs.tau, None], 3, axis=-1)
        vs.update(prepare_forcing(state)[0])
    expected = 600.0 * 0.2 / (state.settings.rho_0 * vs.dzt[-1])
    wet = np.asarray(vs.maskU[2:-2, 2:-2, -1])
    np.testing.assert_allclose(
        vs.u[2:-2, 2:-2, -1, vs.taup1], expected * wet, rtol=1e-13
    )
    assert state.settings.dt_mom / state.settings.dt_tracer == 600.0 / tracer_dt


def test_streamfunction_is_never_interpreted_as_height(execution: Any) -> None:
    """An actual streamfunction native state has no active SSH variable."""
    from vercor.setups._external.veros_setup import CustomGlobalFourDegree
    from vercor.setups._external.veros_state import pure

    model = CustomGlobalFourDegree()
    state = model.state
    with state.settings.unlock():
        model.set_parameter(state)
        state.settings.nx, state.settings.ny, state.settings.nz = 3, 2, 1
        state.settings.enable_streamfunction = True
    state.initialize_variables()
    assert state.var_meta["psi"].units == "m^3/s"
    assert not state.var_meta["ssh"].active
    result = pure(
        state,
        step=lambda _: None,
        linear_solver=object(),
        execution=execution,
        synchronize_surface_height=True,
    )
    assert not result.var_meta["ssh"].active


def test_full_native_continuation_preserves_prognostic_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    execution: Any,
) -> None:
    """Compare the entire driver with and without diagnostic synchronization."""
    from vercor.setups._external import veros_gcm_state
    from vercor.setups._external.veros_state import copy_state, pure
    from veros.core.operators import numpy as npx

    setup_class = surface_setup_class(execution=execution, temperature=0.0)
    monkeypatch.setattr(
        veros_gcm_state, "_select_veros_setup_class", lambda *_: setup_class
    )
    resources = veros_gcm_state.VerosGCMSetupState(
        execution=execution,
        surface_forcing="surface_fluxes",
        custom_parameters={"dt_mom": 60.0},
    )
    adapted = resources._veros_state
    with adapted.variables.unlock():
        vs = adapted.variables
        pattern = npx.sin(npx.arange(vs.taux.shape[0])[:, None, None])
        vs.taux = npx.ones_like(vs.taux) * pattern * 1e-3
    reference = copy_state(adapted, execution=execution)

    def native_step(current: Any) -> Any:
        return pure(
            current,
            step=resources.model.step,
            linear_solver=resources._linear_solver,
            execution=execution,
        )

    run_reference = jax.jit(native_step) if execution == "jax" else native_step
    executable_count = None
    for step_number in range(3):
        adapted = resources._step_function(adapted)
        reference = run_reference(reference)
        if execution == "jax":
            current_count = getattr(resources._step_function, "_cache_size")()
            native_count = getattr(run_reference, "_cache_size")()
            assert current_count == native_count
            # The native initial and continued carries have distinct signatures;
            # both lanes must reuse the continued executable after that warmup.
            if step_number > 1:
                assert current_count == executable_count
            executable_count = current_count
    av, rv = adapted.variables, reference.variables
    for name in (
        "temp",
        "salt",
        "u",
        "v",
        "psi",
        "tau",
        "taup1",
        "taum1",
        "time",
        "itt",
    ):
        np.testing.assert_array_equal(getattr(av, name), getattr(rv, name))
    current_height = np.asarray(av.psi[..., av.tau]) / adapted.settings.grav
    np.testing.assert_allclose(av.ssh, current_height, rtol=1e-14)
    assert np.max(np.abs(current_height - np.asarray(rv.ssh))) > 1e-12
    assert float(av.time) == 180.0
