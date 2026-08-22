"""Workflow runtime execution tests."""

from __future__ import annotations

from collections.abc import Mapping
import signal
from typing import Any, cast

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vercor._runtime.backends as runtime_backends
import vercor.output._runtime as runtime_output
import vercor.runtime as runtime
from tests._coverage_support import make_test_grid
from tests._workflow_test_support import (
    SequentialBackend as _SequentialBackend,
    StaticWorkflow as _StaticWorkflow,
    make_clock as _clock,
    make_component as _component,
)
from tests.assertions import assert_allclose_compact
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.state import RunState
from vercor._runtime.interrupts import RuntimeInterrupted

pytestmark = pytest.mark.fast_always


def test_driver_rejects_a_plan_outside_its_active_chunk() -> None:
    class InventingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            real_plan = chunk.steps[0]
            invented = runtime.StepPlan(step=real_plan.step, components=())
            return driver.run_step(state, invented)

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=InventingBackend()),
    )

    with pytest.raises(CouplerError, match="run_step.*plan.*active.*chunk"):
        coupler.run()


def test_driver_rejects_duplicate_dispatch_of_one_chunk_plan() -> None:
    class RepeatingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            plan = chunk.steps[0]
            state = driver.run_step(state, plan)
            return driver.run_step(state, plan)

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=RepeatingBackend()),
    )

    with pytest.raises(CouplerError, match="run_step.*already.*executed"):
        coupler.run()


def test_backend_must_execute_every_plan_in_its_chunk() -> None:
    class SkippingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context, chunk, driver
            return state

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=SkippingBackend()),
    )

    with pytest.raises(CouplerError, match="backend.*did not execute.*step 0"):
        coupler.run()


def test_same_schedule_chunk_rejects_reordered_plan_dispatch() -> None:
    class ReorderingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            assert len(chunk.steps) == 3
            return driver.run_step(state, chunk.steps[1])

    coupler = Coupler(
        _clock(steps=3),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=ReorderingBackend()),
    )

    with pytest.raises(
        CouplerError,
        match="run_step.*out of order.*expected step 0.*got step 1",
    ):
        coupler.run()


def test_same_schedule_chunk_reports_all_skipped_plans() -> None:
    class PartiallySkippingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            assert len(chunk.steps) == 3
            return driver.run_step(state, chunk.steps[0])

    coupler = Coupler(
        _clock(steps=3),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=PartiallySkippingBackend()),
    )

    with pytest.raises(CouplerError, match="did not execute.*steps 1, 2"):
        coupler.run()


def test_driver_validates_incoming_state_before_component_dispatch() -> None:
    observed: list[tuple[str, int]] = []

    class InvalidStateBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            plan = chunk.steps[0]
            return driver.run_step(cast(RunState, object()), plan)

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host", observed=observed),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=InvalidStateBackend()),
    )

    with pytest.raises(CouplerError, match="run_step.*state.*RunState.*object"):
        coupler.run()
    assert observed == []


def test_supplied_state_is_validated_before_custom_backend_invocation() -> None:
    backend = _SequentialBackend()
    expected = _component("A", execution="host")
    wrong_grid_component = CallableComponent(
        "A",
        make_test_grid(
            name="workflow-wrong-grid",
            longitude=np.asarray([0.0, 1.0, 2.0]),
        ),
        lambda fields: fields,
        spec=ComponentSpec(outputs=("value",), initial_fields={"value": 0.0}),
    )
    foreign = Coupler(
        _clock(steps=1),
        components=(wrong_grid_component,),
        run_order=("A",),
    ).initial_state()
    coupler = Coupler(
        _clock(steps=1),
        components=(expected,),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=backend),
    )

    with pytest.raises(CouplerError, match="A.*runtime grid name.*workflow-wrong-grid"):
        coupler.run(foreign)
    assert backend.calls == []


@pytest.mark.parametrize(
    ("returned", "actual_type"),
    ((None, "NoneType"), ({"state": "bad"}, "dict")),
)
def test_custom_backend_must_return_run_state_for_each_chunk(
    returned: object,
    actual_type: str,
) -> None:
    class InvalidBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> object:
            _ = state, context, chunk, driver
            return returned

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(
            backend=cast(runtime.ExecutionBackend, InvalidBackend())
        ),
    )

    with pytest.raises(
        CouplerError,
        match=rf"InvalidBackend.*execute.*RunState.*{actual_type}",
    ):
        coupler.run()


def test_schema_invalid_state_is_rejected_before_later_nonempty_backend_chunk() -> None:
    observed: list[tuple[str, int]] = []
    wrong_grid = make_test_grid(
        name="workflow-invalid-intermediate-return",
        longitude=np.asarray([0.0, 1.0, 2.0]),
    )
    foreign_state = Coupler(
        _clock(steps=2),
        components=(
            CallableComponent(
                "A",
                wrong_grid,
                lambda fields: fields,
                spec=ComponentSpec(
                    outputs=("value",),
                    initial_fields={"value": 0.0},
                ),
            ),
            _component("B", execution="host"),
        ),
        run_order=("A", "B"),
    ).initial_state()

    class InvalidFirstChunkBackend:
        def __init__(self) -> None:
            self.calls = 0

        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context
            self.calls += 1
            state = driver.run_step(state, chunk.steps[0])
            return foreign_state

    workflow = _StaticWorkflow(
        (
            runtime.StepPlan(step=0, components=("A",)),
            runtime.StepPlan(step=1, components=("B",)),
        )
    )
    backend = InvalidFirstChunkBackend()
    coupler = Coupler(
        _clock(steps=2),
        components=(
            _component("A", execution="host", observed=observed),
            _component("B", execution="host", observed=observed),
        ),
        run_order=("A", "B"),
        runtime=runtime.RuntimeOptions(workflow=workflow, backend=backend),
    )

    with pytest.raises(
        CouplerError,
        match="A.*runtime grid name.*workflow-invalid-intermediate-return",
    ):
        coupler.run()
    assert backend.calls == 1
    assert observed == [("A", 0)]


def test_auto_and_forced_host_backend_selection_preserve_behavior() -> None:
    auto_observed: list[tuple[str, int]] = []
    host_observed: list[tuple[str, int]] = []
    auto = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host", observed=auto_observed),),
        run_order=("A",),
    )
    forced_host = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="jax", observed=host_observed),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend="host"),
    )

    auto.run()
    forced_host.run()

    assert auto_observed == [("A", 0)]
    assert host_observed == [("A", 0)]


def test_forced_jax_backend_rejects_scheduled_host_component() -> None:
    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend="jax"),
    )

    with pytest.raises(
        ComponentError,
        match=r"RuntimeOptions\(backend='jax'\).*host-backed.*A",
    ):
        coupler.run()


def test_default_output_free_jax_workflow_uses_one_jit_and_one_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jit_calls = 0
    scan_calls = 0
    original_jit = jax.jit
    original_scan = jax.lax.scan

    def recording_jit(*args: Any, **kwargs: Any) -> Any:
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(*args, **kwargs)

    def recording_scan(*args: Any, **kwargs: Any) -> Any:
        nonlocal scan_calls
        scan_calls += 1
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(jax, "jit", recording_jit)
    monkeypatch.setattr(jax.lax, "scan", recording_scan)
    coupler = Coupler(
        _clock(steps=3),
        components=(_component("A"),),
        run_order=("A",),
    )

    final_state = coupler.run()
    jax.block_until_ready(final_state.component("A").field("value"))

    assert jit_calls == 1
    assert scan_calls == 1
    assert_allclose_compact(
        final_state.component("A").field("value"),
        jnp.full((2, 2), 3.0),
    )


def test_alternating_jax_workflow_uses_absolute_steps_and_builds_metadata_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = make_test_grid(name="workflow-jax-alternating-absolute")

    def make_component(name: str) -> CallableComponent:
        def step(
            fields: Mapping[str, Any],
            context: StepContext,
        ) -> Mapping[str, Any]:
            return {"value": fields["value"] + context.step + 1.0}

        return CallableComponent(
            name,
            grid,
            step,
            spec=ComponentSpec(
                inputs=("value",),
                outputs=("value",),
                initial_fields={"value": 0.0},
                execution="jax",
            ),
        )

    workflow = _StaticWorkflow(
        (
            runtime.StepPlan(step=0, components=("A",)),
            runtime.StepPlan(step=1, components=("B",)),
            runtime.StepPlan(step=2, components=("A",)),
        )
    )
    coupler = Coupler(
        _clock(steps=3),
        components=(make_component("A"), make_component("B")),
        run_order=("A", "B"),
        runtime=runtime.RuntimeOptions(workflow=workflow),
    )
    initial_state = coupler.initial_state()
    clock_iter_calls = 0
    step_info_calls = 0
    progress_calls = 0
    original_clock_iter = Clock.iter
    original_step_info = runtime_backends.build_runtime_step_info
    original_progress = runtime_backends.runtime_step_progress_messages

    def recording_clock_iter(self: Clock) -> Any:
        nonlocal clock_iter_calls
        clock_iter_calls += 1
        yield from original_clock_iter(self)

    def recording_step_info(*args: Any, **kwargs: Any) -> Any:
        nonlocal step_info_calls
        step_info_calls += 1
        return original_step_info(*args, **kwargs)

    def recording_progress(*args: Any, **kwargs: Any) -> Any:
        nonlocal progress_calls
        progress_calls += 1
        return original_progress(*args, **kwargs)

    monkeypatch.setattr(Clock, "iter", recording_clock_iter)
    monkeypatch.setattr(
        runtime_backends, "build_runtime_step_info", recording_step_info
    )
    monkeypatch.setattr(
        runtime_backends,
        "runtime_step_progress_messages",
        recording_progress,
    )

    final_state = coupler.run(initial_state)

    assert clock_iter_calls == 1
    assert step_info_calls == 1
    assert progress_calls == 1
    assert_allclose_compact(
        final_state.component("A").field("value"),
        jnp.full(grid.shape, 4.0),
    )
    assert_allclose_compact(
        final_state.component("B").field("value"),
        jnp.full(grid.shape, 2.0),
    )


def test_output_enabled_uniform_workflow_reuses_one_jitted_executor(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = OutputSpec(period=PeriodOutput(frequency="step"))
    coupler = Coupler(
        _clock(steps=3, dt_seconds=86_400.0),
        components=(_component("A", output=output),),
        run_order=("A",),
    )
    initial_state = coupler.initial_state()
    jit_calls = 0
    original_jit = jax.jit

    def recording_jit(*args: Any, **kwargs: Any) -> Any:
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(*args, **kwargs)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jax, "jit", recording_jit)

    final_state = coupler.run(
        initial_state,
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        ),
    )
    jax.block_until_ready(final_state.component("A").field("value"))

    assert jit_calls == 1
    assert len(tuple(tmp_path.glob("a.averages.*.nc"))) == 3
    assert_allclose_compact(
        final_state.component("A").field("value"),
        jnp.full((2, 2), 3.0),
    )


def test_custom_backend_period_output_is_accumulated_and_written_by_core(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    backend = _SequentialBackend()
    output = OutputSpec(period=PeriodOutput(frequency="step"))
    component = _component("A", execution="host", output=output)
    coupler = Coupler(
        _clock(steps=2, dt_seconds=86_400.0),
        components=(component,),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=backend),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    paths = sorted(tmp_path.glob("a.averages.*.nc"))
    assert len(paths) == 2
    for expected, path in enumerate(paths, start=1):
        with h5netcdf.File(path, "r") as dataset:
            assert_allclose_compact(
                np.asarray(dataset.variables["value"]),
                np.full((1, 2, 2), float(expected)),
            )


def test_custom_backend_interruption_is_handled_by_core() -> None:
    class InterruptingBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: runtime.ExecutionContext,
            chunk: runtime.ExecutionChunk,
            driver: runtime.RuntimeDriver,
        ) -> RunState:
            _ = context, chunk, driver
            signal.raise_signal(signal.SIGINT)
            return state

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(backend=InterruptingBackend()),
    )

    with pytest.raises(RuntimeInterrupted, match="SIGINT"):
        coupler.run()


@pytest.mark.parametrize("output_kind", ["final", "snapshot"])
def test_run_owned_final_io_remains_inside_interrupt_scope(
    output_kind: str,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = Coupler(
        _clock(steps=0),
        components=(_component("A"),),
        run_order=("A",),
        log_level="WARNING",
    )
    prepared = cast(Any, coupler)._ensure_prepared()

    def request_interrupt(**kwargs: Any) -> None:
        _ = kwargs
        prepared.interrupts.request(signal.SIGINT)

    if output_kind == "final":
        monkeypatch.setattr(
            runtime_output,
            "write_coupler_runtime_outputs",
            request_interrupt,
        )
    else:
        monkeypatch.setattr(
            runtime_output,
            "write_coupler_component_snapshots",
            request_interrupt,
        )

    with pytest.raises(RuntimeInterrupted, match="SIGINT.*runtime output"):
        coupler.run(
            output=OutputTarget(
                tmp_path,
                write_period=False,
                write_final_fields=output_kind == "final",
                write_snapshots=output_kind == "snapshot",
            )
        )


def test_output_free_workflow_preserves_jvp_and_reverse_mode_gradients() -> None:
    component = _component("A")
    coupler = Coupler(
        _clock(steps=3),
        components=(component,),
        run_order=("A",),
    )
    initial_state = coupler.initial_state()

    def objective(value: jax.Array) -> jax.Array:
        state = initial_state.replace_fields(
            "A",
            {"value": jnp.full(component.grid.shape, value)},
        )
        return jnp.mean(coupler.run(state).component("A").field("value"))

    value = jnp.asarray(2.0)
    tangent_seed = jnp.asarray(1.0)
    primal, tangent = jax.jvp(
        objective,
        (value,),
        (tangent_seed,),
    )
    reverse = jax.grad(objective)(value)
    vjp_primal, pullback = jax.vjp(objective, value)
    (reverse_vjp,) = pullback(jnp.ones_like(vjp_primal))

    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(tangent))
    assert np.isfinite(np.asarray(reverse_vjp))
    assert_allclose_compact(primal, jnp.asarray(5.0))
    assert_allclose_compact(tangent, jnp.asarray(1.0))
    assert_allclose_compact(reverse, jnp.asarray(1.0))
    assert_allclose_compact(tangent, reverse_vjp, equal_nan=False)


def test_payload_dependent_multi_step_scan_preserves_treedef_jvp_and_grad() -> None:
    grid = make_test_grid(name="workflow-payload-gradient")

    def setup(component: object, context: object) -> SetupResult:
        _ = component, context
        return SetupResult(payload={"scale": jnp.asarray(2.0)})

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
        payload: object | None,
    ) -> StepResult:
        _ = context
        scale = cast(dict[str, jax.Array], payload)["scale"]
        return StepResult(
            fields={"value": fields["value"] * scale},
            payload={"scale": scale},
        )

    component = CallableComponent(
        "A",
        grid,
        step,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            initial_fields={"value": 1.0},
            lifecycle=LifecycleHooks(setup=setup),
        ),
    )
    coupler = Coupler(
        _clock(steps=3),
        components=(component,),
        run_order=("A",),
    )
    initial_state = coupler.initial_state()
    initial_tree = jax.tree_util.tree_structure(initial_state)

    def objective(value: jax.Array) -> jax.Array:
        state = initial_state.replace_fields(
            "A",
            {"value": jnp.full(grid.shape, value)},
        )
        result = coupler.run(state)
        assert cast(object, jax.tree_util.tree_structure(result)) == cast(
            object, initial_tree
        )
        return jnp.mean(result.component("A").field("value"))

    value = jnp.asarray(3.0)
    tangent_seed = jnp.asarray(1.0)
    primal, tangent = jax.jvp(
        objective,
        (value,),
        (tangent_seed,),
    )
    reverse = jax.grad(objective)(value)
    vjp_primal, pullback = jax.vjp(objective, value)
    (reverse_vjp,) = pullback(jnp.ones_like(vjp_primal))

    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(tangent))
    assert np.isfinite(np.asarray(reverse_vjp))
    assert_allclose_compact(primal, jnp.asarray(24.0))
    assert_allclose_compact(tangent, jnp.asarray(8.0))
    assert_allclose_compact(reverse, jnp.asarray(8.0))
    assert_allclose_compact(tangent, reverse_vjp, equal_nan=False)


def test_output_free_workflow_keeps_inactive_nan_out_of_jvp_and_vjp() -> None:
    binary_mask = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    grid = make_test_grid(
        name="workflow-inactive-missing-data",
        binary_mask=binary_mask,
    )

    def step(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        _ = context
        return {"value": fields["value"] + 1.0}

    component = CallableComponent(
        "MASKED",
        grid,
        step,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            initial_fields={"value": jnp.asarray([[2.0, jnp.nan], [3.0, jnp.nan]])},
        ),
    )
    coupler = Coupler(
        _clock(steps=2),
        components=(component,),
        run_order=("MASKED",),
    )
    initial_state = coupler.initial_state()
    initial_values = cast(
        jax.Array,
        initial_state.component("MASKED").field("value"),
    )
    active = jnp.asarray(binary_mask > 0.0)
    tangent_seed = jnp.where(
        active,
        jnp.ones_like(initial_values),
        jnp.zeros_like(initial_values),
    )

    assert np.all(np.isfinite(np.asarray(initial_values)[binary_mask > 0.0]))
    assert np.all(np.isnan(np.asarray(initial_values)[binary_mask == 0.0]))

    def objective(values: jax.Array) -> jax.Array:
        state = initial_state.replace_fields("MASKED", {"value": values})
        result = coupler.run(state)
        final_values = result.component("MASKED").field("value")
        return jnp.sum(jnp.where(active, final_values, 0.0))

    eager = objective(initial_values)
    compiled_objective = jax.jit(objective)
    compiled = compiled_objective(initial_values)
    primal, forward_tangent = jax.jvp(
        compiled_objective,
        (initial_values,),
        (tangent_seed,),
    )
    value, pullback = jax.vjp(compiled_objective, initial_values)
    (reverse_vjp,) = pullback(jnp.ones_like(value))
    reverse_projection = jnp.vdot(tangent_seed, reverse_vjp)

    assert np.isfinite(np.asarray(eager))
    assert np.isfinite(np.asarray(compiled))
    assert np.isfinite(np.asarray(primal))
    assert np.isfinite(np.asarray(forward_tangent))
    assert np.all(np.isfinite(np.asarray(reverse_vjp)))
    assert np.isfinite(np.asarray(reverse_projection))
    assert_allclose_compact(eager, jnp.asarray(9.0), equal_nan=False)
    assert_allclose_compact(compiled, eager, equal_nan=False)
    assert_allclose_compact(primal, eager, equal_nan=False)
    assert_allclose_compact(
        reverse_projection,
        forward_tangent,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=False,
    )
