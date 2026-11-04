"""Validate stock JAX native stepping and host payload ownership in isolation."""

from __future__ import annotations

from datetime import datetime
from functools import partial
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from vercor.components import StepContext
from vercor.components._adapter import _copy_owned_pytree
from vercor.dtypes import DTypePolicy
from vercor.setups._external import veros_runtime_settings


def main() -> None:
    """Run authentic kernels and reject mutations of an earlier native state."""
    configure = veros_runtime_settings.configure_veros_runtime
    configure("host", backend="jax")
    # The existing source-budget fixture selects execution on entry. Keep this
    # fresh process on the explicitly chosen native backend while constructing it.
    veros_runtime_settings.configure_veros_runtime = partial(configure, backend="jax")
    from tests._veros_surface_support import surface_model, feedback_fields

    model = surface_model(execution="host")
    veros_runtime_settings.configure_veros_runtime = configure
    from vercor.setups._external.veros_runtime import step_veros_runtime
    from vercor.setups._external.veros_state import (
        copy_state,
        get_component_linear_solver,
        pure,
    )

    # Match the factory's setup-time native payload preparation.
    initial = copy_state(model.state, execution="host")
    temperature = np.asarray(initial.variables.temp).copy()
    resources: Any = SimpleNamespace(
        execution="host",
        surface_forcing="surface_fluxes",
        export_open_water_surface=True,
        model_substeps=2,
        grid=SimpleNamespace(
            binary_mask=jnp.asarray(initial.variables.maskT[2:-2, 2:-2, -1]).T
        ),
        _step_function=partial(
            pure,
            step=model.step,
            linear_solver=get_component_linear_solver(initial),
            execution="host",
            synchronize_surface_height=True,
        ),
    )
    fields = feedback_fields(heat=1000.0, mass=5e-3, salt=1e-4)
    fields["ice_fraction"] = jnp.full((4, 4), 0.25)
    context = StepContext(
        1200.0, time=datetime(2000, 1, 1), dtype=DTypePolicy(enable_x64=True)
    )
    result = step_veros_runtime(resources, fields, context, _copy_owned_pytree(initial))
    assert (
        result.payload.timers is not initial.timers
    ), "ERROR host JAX shares native timers"
    assert (
        result.payload.diagnostics is not initial.diagnostics
    ), "ERROR host JAX shares diagnostics"
    assert isinstance(result.payload.variables.qnet, jax.Array)
    assert result.payload.variables.temp.dtype == jnp.float64
    assert int(result.payload.variables.itt) == int(initial.variables.itt) + 2
    np.testing.assert_array_equal(initial.variables.temp, temperature)
    assert int(initial.variables.itt) == 0
    # Independent unmixed-column source oracle: 2*600 s over the 10 m top
    # layer. Heating and freshening both stabilize this initially uniform column.
    vs = result.payload.variables
    wet = np.asarray(vs.maskT[2:-2, 2:-2, -1]) > 0
    heat_rate = 1000.0 / (initial.settings.rho_0 * 3991.86795711963 * 10.0)
    salt_rate = (-34.7 * 5e-3 + 1000.0 * 1e-4) / 1000.0 / 10.0
    np.testing.assert_allclose(
        np.asarray(vs.temp[2:-2, 2:-2, -1, int(vs.tau)])[wet],
        -2.0 + 1200.0 * heat_rate,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(vs.salt[2:-2, 2:-2, -1, int(vs.tau)])[wet],
        35.0 + 1200.0 * salt_rate,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.fields["sea_surface_temperature"])[wet.T],
        271.15 + 900.0 * heat_rate,
        rtol=1e-12,
        atol=0.0,
    )
    from veros.core.thermodynamics import advect_temp_salt_enthalpy

    compiled = advect_temp_salt_enthalpy.function
    cache_size = compiled._cache_size()
    assert cache_size > 0, "ERROR cache probe must observe the outer native kernel"
    replay = step_veros_runtime(resources, fields, context, _copy_owned_pytree(initial))
    assert (
        compiled._cache_size() == cache_size
    ), "ERROR host JAX retraces unchanged native kernels"
    for name, value in result.fields.items():
        np.testing.assert_array_equal(value, replay.fields[name])
        assert np.all(np.isfinite(value))
    changed = copy_state(initial, execution="host")
    original = initial.settings.K_iso_0
    with changed.settings.unlock():
        changed.settings.K_iso_0 = original + 1.0
    jax.block_until_ready(advect_temp_salt_enthalpy(changed))
    assert (
        compiled._cache_size() == cache_size + 1
    ), "ERROR changed settings must invalidate the native executable"
    assert initial.settings.K_iso_0 == original


if __name__ == "__main__":
    main()
