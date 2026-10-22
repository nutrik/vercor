"""Fixed runtime schedules compile once despite cadence and chunk length changes."""

from __future__ import annotations

from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import StaticWorkflow, make_clock, make_component
from vercor import Clock, Coupler
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.runtime import RuntimeOptions, StepPlan
from vercor._interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.grid_geometry import centers_to_edges

pytestmark = pytest.mark.fast_always


def test_repeated_host_output_geometry_does_not_retrace_conditionals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native face coordinates reuse their geometry kernel across samples."""
    calls = 0
    original = jax.lax.cond

    def recording_cond(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(jax.lax, "cond", recording_cond)
    centers = jnp.asarray([0.0, 90.0, 180.0, 270.0], dtype=jnp.float64)
    expected = jnp.asarray([-45.0, 45.0, 135.0, 225.0, 315.0])
    np.testing.assert_allclose(centers_to_edges(centers, "lon"), expected)
    first_calls = calls
    for _ in range(3):
        np.testing.assert_allclose(centers_to_edges(centers, "lon"), expected)
    assert (
        calls == first_calls
    ), f"ERROR output geometry retraced {calls - first_calls} conditional branches"


def _compiled_traces(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count traces of runtime executors, independently of invocation count."""
    traces: list[int] = []
    original = jax.jit

    def recording_jit(function: Any, *args: Any, **kwargs: Any) -> Any:
        if function.__name__ != "scanned_chunk":
            return original(function, *args, **kwargs)

        @wraps(function)
        def traced(*inputs: Any, **options: Any) -> Any:
            traces.append(int(inputs[1].shape[0]))
            return function(*inputs, **options)

        return original(traced, *args, **kwargs)

    monkeypatch.setattr(jax, "jit", recording_jit)
    return traces


def test_nonuniform_schedule_lengths_compile_once_per_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schedules = (("A",),) * 2 + (("B",),) + (("A",),) * 3 + (("B",),) * 2 + (("A",),)
    workflow = StaticWorkflow(
        tuple(StepPlan(i, names) for i, names in enumerate(schedules))
    )
    coupler = Coupler(
        make_clock(len(schedules)),
        components=(make_component("A"), make_component("B")),
        run_order=("A", "B"),
        runtime=RuntimeOptions(workflow=workflow),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    traces = _compiled_traces(monkeypatch)
    final = jax.block_until_ready(coupler.run(seed))
    np.testing.assert_allclose(final.component("A").field("value"), 6.0)
    np.testing.assert_allclose(final.component("B").field("value"), 3.0)
    assert (
        len(traces) == 2
    ), f"ERROR schedule executors compiled {len(traces)} times: {traces}"


def test_exact_blocks_preserve_outer_jit_jvp_and_reverse_derivatives() -> None:
    """Internal blocks must preserve every state advance and derivative."""
    schedules = (("A",),) * 2 + (("B",),) + (("A",),) * 3 + (("B",),) * 2 + (("A",),)
    workflow = StaticWorkflow(
        tuple(StepPlan(i, names) for i, names in enumerate(schedules))
    )
    coupler = Coupler(
        make_clock(len(schedules)),
        components=(make_component("A"), make_component("B")),
        run_order=("A", "B"),
        runtime=RuntimeOptions(workflow=workflow),
        log_level="ERROR",
    )
    seed = coupler.initial_state()

    def objective(value: jax.Array) -> jax.Array:
        state = seed.replace_fields("A", {"value": jnp.full((2, 2), value)})
        final = coupler.run(state)
        return jnp.sum(jnp.square(final.component("A").field("value")))

    value = jnp.asarray(2.0, dtype=jnp.float64)
    primal, forward = jax.jvp(objective, (value,), (jnp.ones_like(value),))
    np.testing.assert_allclose(jax.jit(objective)(value), 256.0)
    np.testing.assert_allclose(primal, 256.0)
    np.testing.assert_allclose(forward, 64.0)
    np.testing.assert_allclose(jax.grad(objective)(value), 64.0)


def test_skewed_schedules_have_no_inactive_work_or_gradient_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long prefix must not multiply storage for later singleton windows."""
    schedules = (("A",),) * 20 + (("B",), ("A",)) * 10
    workflow = StaticWorkflow(
        tuple(StepPlan(i, names) for i, names in enumerate(schedules))
    )
    coupler = Coupler(
        make_clock(len(schedules)),
        components=(make_component("A"), make_component("B")),
        run_order=("A", "B"),
        runtime=RuntimeOptions(workflow=workflow),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    slots: list[int] = []
    original = jax.jit

    def recording_jit(function: Any, *args: Any, **kwargs: Any) -> Any:
        compiled = original(function, *args, **kwargs)
        if function.__name__ != "scanned_chunk":
            return compiled

        def invoke(*inputs: Any, **options: Any) -> Any:
            slots.append(int(inputs[1].shape[0]))
            return compiled(*inputs, **options)

        return invoke

    monkeypatch.setattr(jax, "jit", recording_jit)
    final = jax.block_until_ready(coupler.run(seed))
    np.testing.assert_allclose(final.component("A").field("value"), 30.0)
    np.testing.assert_allclose(final.component("B").field("value"), 10.0)
    assert sum(slots) < 2 * len(
        schedules
    ), f"ERROR {sum(slots)} scan slots for {len(schedules)} actual steps"


def test_calendar_windows_and_incomplete_tail_share_one_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = make_component("A", output=OutputSpec(period=PeriodOutput("month")))
    coupler = Coupler(
        Clock(datetime(2001, 1, 1), 86_400.0, 70),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    traces = _compiled_traces(monkeypatch)
    final = jax.block_until_ready(
        coupler.run(
            seed,
            output=OutputTarget(
                tmp_path,
                write_final_fields=False,
                write_snapshots=False,
            ),
        )
    )
    np.testing.assert_allclose(final.component("A").field("value"), 70.0)
    paths = sorted(tmp_path.glob("a.averages.*.nc"))
    assert len(paths) == 2
    for path, expected in zip(paths, (16.0, 45.5), strict=True):
        with h5netcdf.File(path) as dataset:
            np.testing.assert_allclose(dataset.variables["value"][:], expected)
    assert (
        len(traces) == 1
    ), f"ERROR output executor compiled {len(traces)} times: {traces}"


@pytest.mark.parametrize("vector", [False, True])
def test_host_bilinear_control_flow_reuses_compilation(
    vector: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = 0
    original = BilinearRectilinearInterpolator._extrapolate_scalar

    def extrapolate(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal traces
        traces += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        BilinearRectilinearInterpolator, "_extrapolate_scalar", extrapolate
    )
    interpolator = BilinearRectilinearInterpolator(
        np.asarray([0.0, 1.0]),
        np.asarray([-1.0, 1.0]),
        np.asarray([0.25, 0.75]),
        np.asarray([-0.5, 0.5]),
        periodic_longitude=False,
        extrapolation_mode="nearest",
        fill_value=-123.456,
    )
    source = jnp.ones((2, 2), dtype=jnp.float64)
    for value in (1.0, 2.0, 3.0):
        if vector:
            result = interpolator.apply_vector(source * value, source * value)
        else:
            result = interpolator.apply_scalar(source * value)
            np.testing.assert_allclose(result, value)
        jax.block_until_ready(result)
    expected = 2 if vector else 1
    assert (
        traces == expected
    ), f"ERROR bilinear branch traced {traces} times, expected {expected}"
