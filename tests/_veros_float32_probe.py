"""Real native float32 acceptance in an isolated backend/precision process."""

from __future__ import annotations

from datetime import datetime
from functools import partial
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from tests._veros_surface_support import feedback_fields, surface_model
from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.components import StepContext
from vercor.dtypes import DTypePolicy

_FLOAT32 = np.dtype("float32")
_EPS = float(np.finfo(np.float32).eps)
_FIELDS = (
    "sea_surface_temperature",
    "sea_surface_salinity",
    "ocean_u_velocity",
    "ocean_v_velocity",
    "sea_surface_height",
    "open_water_temperature_contribution",
)


def _assert_float32(values: Any, *, owner: str) -> None:
    """Reject promotion of every floating native or canonical array."""
    for name, value in values.items():
        dtype = np.dtype(value.dtype)
        if dtype.kind == "f":
            assert dtype == _FLOAT32, f"ERROR {owner}.{name} promoted to {dtype}"
        assert np.all(np.isfinite(np.asarray(value))), f"ERROR {owner}.{name} finite"


def _assert_native_categories(state: Any) -> None:
    """Preserve native integer time slots and binary wet masks."""
    for name in ("tau", "taup1", "taum1", "itt"):
        assert getattr(state.variables, name).dtype == np.dtype("int32"), name
    for name in ("maskT", "maskU", "maskV"):
        assert getattr(state.variables, name).dtype == np.dtype("bool"), name


def _assert_roundoff(
    actual: Any, expected: Any, *, operations: int, label: str
) -> None:
    """Bound float32 arithmetic by operation count and the reference scale."""
    scale = max(
        float(np.finfo(np.float32).tiny),
        float(np.max(np.abs(np.asarray(expected)))),
    )
    assert_allclose_compact(
        actual,
        expected,
        rtol=0.0,
        atol=operations * _EPS * scale,
        equal_nan=False,
        label=label,
    )


def _resources(model: Any, execution: Literal["host", "jax"]) -> Any:
    """Bind the actual full native driver and component-owned solver."""
    from vercor.setups._external.veros_state import get_component_linear_solver, pure

    native_step = partial(
        pure,
        step=model.step,
        linear_solver=get_component_linear_solver(model.state),
        execution=execution,
        synchronize_surface_height=True,
    )
    return SimpleNamespace(
        execution=execution,
        uses_bulk_meteorology=False,
        surface_forcing="surface_fluxes",
        export_open_water_surface=True,
        model_substeps=1,
        _step_function=jax.jit(native_step) if execution == "jax" else native_step,
        grid=SimpleNamespace(
            binary_mask=jnp.asarray(model.state.variables.maskT[2:-2, 2:-2, -1]).T
        ),
    )


def _forcing() -> dict[str, jax.Array]:
    """Resolve f32 increments while warming/freshening keeps the column stable.

    Freshwater dominates salt addition even at the AD stencil's endpoints.
    Otherwise native convective adjustment would invalidate the unmixed oracle.
    """
    fields = feedback_fields(heat=1000.0, mass=5e-3, salt=1e-4, dtype=jnp.float32)
    fields["ice_fraction"] = jnp.full((4, 4), 0.25, dtype=jnp.float32)
    return fields


def _advance(resources: Any, state: Any, fields: Any) -> Any:
    from vercor.setups._external.veros_runtime import step_veros_runtime
    from vercor.setups._external.veros_state import copy_state

    return step_veros_runtime(
        resources,
        fields,
        StepContext(
            600.0 * resources.model_substeps,
            time=datetime(2000, 1, 1),
            dtype=DTypePolicy(enable_x64=False),
        ),
        copy_state(state, execution=resources.execution),
    )


def _manual_surfaces(state: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Map native slots and pressure directly, independently of the extractor."""
    vs = state.variables
    slot = int(vs.tau)
    wet = np.asarray(vs.maskT[2:-2, 2:-2, -1]).T

    def native(name: str, mask: str) -> np.ndarray[Any, Any]:
        return cast(
            np.ndarray[Any, Any],
            (
                np.asarray(getattr(vs, name)[2:-2, 2:-2, -1, slot], dtype=np.float64)
                * np.asarray(getattr(vs, mask)[2:-2, 2:-2, -1])
            ).T,
        )

    temperature = np.asarray(vs.temp[2:-2, 2:-2, -1, slot], dtype=np.float64).T + 273.15
    return {
        "sea_surface_temperature": temperature,
        "sea_surface_salinity": native("salt", "maskT"),
        "ocean_u_velocity": native("u", "maskU"),
        "ocean_v_velocity": native("v", "maskV"),
        "sea_surface_height": (
            np.asarray(vs.psi[2:-2, 2:-2, slot], dtype=np.float64).T
            / state.settings.grav
            * wet
        ),
        "open_water_temperature_contribution": 0.75 * temperature * wet,
    }


def _replay(resources: Any, initial: Any, fields: Any) -> tuple[Any, Any]:
    """Run real native steps and reduce independently mapped test samples."""
    from vercor.physics import PhysicalConstants
    from vercor.setups._external.veros_fluxes import compute_coupled_fluxes
    from vercor.setups._external.veros_state import (
        apply_veros_forcing_fields,
        copy_state,
        prepare_surface_forcing_fields,
    )

    taux, tauy, heat, salt = compute_coupled_fluxes(
        fields, PhysicalConstants(), DTypePolicy(enable_x64=False)
    )
    reference = apply_veros_forcing_fields(
        copy_state(initial, execution=resources.execution),
        prepare_surface_forcing_fields(taux, tauy, heat, None, salt),
        execution=resources.execution,
        activate_coupled_feedback=True,
    )
    samples = []
    for _ in range(resources.model_substeps):
        reference = resources._step_function(reference)
        samples.append(_manual_surfaces(reference))
    means = {
        name: np.mean([sample[name] for sample in samples], axis=0) for name in _FIELDS
    }
    return reference, means


def _check_means(resources: Any, initial: Any) -> None:
    """Catch endpoint export, interval leakage, masks and changed continuation."""
    fields = _forcing()
    wet = np.asarray(resources.grid.binary_mask)
    assert int(np.count_nonzero(wet)) == 15
    original = {
        name: np.asarray(value).copy() for name, value in initial.variables.items()
    }
    for count in (1, 3):
        resources.model_substeps = count
        current = initial
        for interval in range(2):
            result = _advance(resources, current, fields)
            reference, expected = _replay(resources, current, fields)
            _assert_float32(result.payload.variables, owner="native endpoint")
            _assert_native_categories(result.payload)
            _assert_float32(result.fields, owner="canonical mean")
            for name in _FIELDS:
                _assert_roundoff(
                    result.fields[name],
                    expected[name],
                    operations=count + 3,
                    label=name,
                )
            _check_source_budgets(result, initial, wet, count=count, interval=interval)
            for name, value in reference.variables.items():
                assert_array_equal_compact(
                    getattr(result.payload.variables, name),
                    value,
                    label=f"endpoint {name}",
                )
            assert int(result.payload.variables.itt) == (interval + 1) * count
            assert float(result.payload.variables.time) == (interval + 1) * count * 600
            # SST retains its established land fill; other mapped surfaces vanish.
            for name in _FIELDS:
                fill = 273.15 if name == "sea_surface_temperature" else 0.0
                _assert_roundoff(
                    np.asarray(result.fields[name])[~wet],
                    fill,
                    operations=2,
                    label=f"dry {name}",
                )
            current = result.payload
    for name, value in original.items():
        assert_array_equal_compact(
            getattr(initial.variables, name), value, label=f"original {name}"
        )

    # Nonuniform traction makes the velocity/SSH mean tests nonvacuous.
    x, y = jnp.meshgrid(
        jnp.arange(4, dtype=jnp.float32), jnp.arange(4, dtype=jnp.float32)
    )
    fields["ocean_u_stress"] = 0.05 * jnp.sin((x + 0.25) * jnp.pi / 2)
    fields["ocean_v_stress"] = 0.05 * jnp.cos((y + 0.25) * jnp.pi / 2)
    result = _advance(resources, initial, fields)
    _assert_float32(result.payload.variables, owner="traction native endpoint")
    _assert_float32(result.fields, owner="traction canonical mean")
    _, expected = _replay(resources, initial, fields)
    endpoint = _manual_surfaces(result.payload)
    for name in _FIELDS:
        _assert_roundoff(
            result.fields[name], expected[name], operations=6, label=f"traction {name}"
        )
    for name in ("ocean_u_velocity", "ocean_v_velocity", "sea_surface_height"):
        assert np.max(np.abs(expected[name])) > 1e-7, f"ERROR vacuous traction {name}"
        assert (
            np.max(np.abs(expected[name] - endpoint[name])) > 1e-7
        ), f"ERROR no mean signal {name}"


def _check_source_budgets(
    result: Any, initial: Any, wet: Any, *, count: int, interval: int
) -> None:
    """Check tracer increments from independent SI heat and salt balances."""
    capacity = initial.settings.rho_0 * 3991.86795711963
    increment = {
        "sea_surface_temperature": 600 * 1000.0 / (capacity * 10),
        "sea_surface_salinity": 600 * (-34.7 * 5e-3 + 1000 * 1e-4) / (1000 * 10),
    }
    baseline = {"sea_surface_temperature": 271.15, "sea_surface_salinity": 35.0}
    assert np.all(
        np.asarray(result.payload.variables.kappaH) == 0
    ), "ERROR source oracle requires an unmixed stable column"
    for name, delta in increment.items():
        mean_step = interval * count + (count + 1) / 2
        _assert_roundoff(
            np.asarray(result.fields[name])[wet],
            baseline[name] + mean_step * delta,
            operations=2 * (interval + 1) * count + 3,
            label=f"physical mean {name}",
        )
        _assert_roundoff(
            np.asarray(
                getattr(
                    result.payload.variables,
                    "temp" if name == "sea_surface_temperature" else "salt",
                )[2:-2, 2:-2, -1, result.payload.variables.tau]
            ).T[wet],
            (-2.0 if name == "sea_surface_temperature" else 35.0)
            + (interval + 1) * count * delta,
            operations=2 * (interval + 1) * count + 2,
            label=f"physical endpoint {name}",
        )


def _check_source_derivatives(resources: Any, initial: Any) -> None:
    """Check actual native three-step mean JIT/JVP/VJP and resolved differences."""
    resources.model_substeps = 3

    def response(values: jax.Array) -> jax.Array:
        fields = _forcing()
        for index, name in enumerate(
            (
                "sea_surface_heat_flux",
                "sea_surface_mass_flux",
                "sea_surface_salt_mass_flux",
            )
        ):
            fields[name] = fields[name] * values[index]
        result = _advance(resources, initial, fields)
        return jnp.stack(
            tuple(
                result.fields[name][1, 1]
                for name in ("sea_surface_temperature", "sea_surface_salinity")
            )
        )

    compiled = jax.jit(response)
    values = jnp.ones(3, dtype=jnp.float32)
    capacity = initial.settings.rho_0 * 3991.86795711963
    expected = np.asarray(
        [
            [1200 * 1000 / (capacity * 10), 0, 0],
            [0, -1200 * 34.7 * 5e-3 / (1000 * 10), 1200 * 1e-4 / 10],
        ]
    )
    value = compiled(values)
    _assert_float32({"response": value}, owner="JIT")
    assert_array_equal_compact(value, response(values), label="JIT native source mean")
    forward = jax.jacfwd(compiled)(values)
    reverse = jax.jacrev(compiled)(values)
    _assert_float32({"forward": forward, "reverse": reverse}, owner="AD")
    _assert_roundoff(
        forward, expected, operations=16, label="analytic native source Jacobian"
    )
    _assert_roundoff(
        reverse, expected, operations=16, label="reverse native source Jacobian"
    )
    print(
        f"native f32 Jacobian max abs error={np.max(np.abs(np.asarray(forward) - expected)):.3e}"
    )
    print(
        "native f32 Jacobian "
        + json.dumps(
            {
                "expected": expected.tolist(),
                "forward": np.asarray(forward).tolist(),
                "reverse": np.asarray(reverse).tolist(),
            }
        )
    )
    direction = jnp.asarray([1, -0.5, 0.25], dtype=jnp.float32)
    cotangent = jnp.asarray([0.75, -0.5], dtype=jnp.float32)
    _, tangent = jax.jvp(compiled, (values,), (direction,))
    _, pullback = jax.vjp(compiled, values)
    (gradient,) = pullback(cotangent)
    _assert_roundoff(
        tangent, expected @ np.asarray(direction), operations=16, label="native JVP"
    )
    _assert_roundoff(
        gradient, expected.T @ np.asarray(cotangent), operations=16, label="native VJP"
    )
    # Binary-fraction perturbations are exact in f32; each heat/salt response
    # spans hundreds of ULP. Four ULP per output bounds the native sum/divide
    # and Kelvin conversion, propagated through the centered stencil.
    for step in (0.25, 0.125):
        columns = []
        bounds = []
        for axis in jnp.eye(3, dtype=jnp.float32):
            plus = np.asarray(compiled(values + step * axis), dtype=np.float64)
            minus = np.asarray(compiled(values - step * axis), dtype=np.float64)
            columns.append((plus - minus) / (2 * step))
            bounds.append(
                4
                * np.maximum(
                    np.spacing(plus.astype(np.float32)),
                    np.spacing(minus.astype(np.float32)),
                )
                / step
            )
        finite = np.stack(columns, axis=1)
        bound = np.stack(bounds, axis=1)
        assert np.all(np.isfinite(finite))
        assert np.all(
            np.abs(finite - expected) <= bound
        ), f"ERROR unresolved source stencil step={step}"
        print(
            f"native f32 FD step={step}: max abs error={np.max(np.abs(finite - expected)):.3e}"
        )


def main() -> None:
    """Separate explicit allocation from the native process precision policy.

    The exact fork does not cast every scalar-only kernel result back to its
    configured dtype. Its true f32 driver therefore uses x64=False from startup.
    A separate allocation probe retains pytest's x64 capability, protecting the
    explicit fixture dtype without claiming upstream mixed-precision support.
    """
    allocation_only = sys.argv[2] == "allocation"
    assert bool(jax.config.read("jax_enable_x64")) == allocation_only
    assert os.environ["VEROS_FLOAT_TYPE"] == "float32"
    execution = cast(Literal["host", "jax"], sys.argv[1])
    from vercor.setups._external.veros_runtime_settings import (
        configure_veros_runtime,
        require_differentiable_veros_capabilities,
    )
    from veros import runtime_settings

    configure_veros_runtime(execution)
    assert runtime_settings.float_type == "float32"
    if execution == "jax":
        require_differentiable_veros_capabilities()
    model = surface_model(execution=execution)
    assert model.state.variables.dzt.dtype == np.dtype("float32"), (
        "ERROR native float32 fixture promoted dzt: "
        f"{model.state.variables.dzt.dtype}"
    )
    _assert_float32(model.state.variables, owner="initial native")
    _assert_native_categories(model.state)
    _assert_float32(_forcing(), owner="forcing")
    if allocation_only:
        print("native float32 explicit allocation with x64 capability: passed")
        return
    resources = _resources(model, execution)
    _check_means(resources, model.state)
    if execution == "jax":
        _check_source_derivatives(resources, model.state)
    print(
        f"native float32 {execution}: dtype, means, continuation, masks, sources"
        + (", JIT/JVP/VJP/FD" if execution == "jax" else "")
    )


if __name__ == "__main__":
    main()
