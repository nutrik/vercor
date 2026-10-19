"""Public planning, validation, and scheduling contracts for 0.4 workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import datetime
import inspect
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

import vercor
import vercor.runtime as runtime
from tests._coverage_support import make_test_grid
from tests._workflow_test_support import (
    SequentialBackend as _SequentialBackend,
    StaticWorkflow as _StaticWorkflow,
    make_clock as _clock,
    make_component as _component,
)
from tests.assertions import assert_allclose_compact
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.coupler import Coupler
from vercor.exchanges import Exchange
from vercor.exceptions import CouplerError

pytestmark = pytest.mark.fast_always


def test_runtime_module_owns_only_the_v0_4_workflow_contracts() -> None:
    assert runtime.__all__ == [
        "ExecutionBackend",
        "ExecutionChunk",
        "ExecutionContext",
        "ExecutionPlan",
        "RuntimeDriver",
        "RuntimeOptions",
        "SequentialWorkflow",
        "StepPlan",
        "Workflow",
        "WorkflowContext",
    ]
    assert not hasattr(runtime, "ExecutionMode")
    assert not hasattr(runtime.RuntimeDriver, "step_component")
    assert not hasattr(runtime.ExecutionBackend, "run")
    assert "RuntimeRunContext" not in str(
        inspect.signature(runtime.ExecutionBackend.execute)
    )
    for name in runtime.__all__:
        if name != "RuntimeOptions":
            assert name not in vercor.__all__


def test_runtime_options_uses_backend_and_workflow_defaults() -> None:
    options = runtime.RuntimeOptions()

    assert options.dtype == runtime.RuntimeOptions().dtype
    assert options.backend == "auto"
    assert isinstance(options.workflow, runtime.SequentialWorkflow)
    assert options.topology is None
    assert not hasattr(options, "model_year_seconds")
    assert (
        "model_year_seconds" not in inspect.signature(runtime.RuntimeOptions).parameters
    )
    assert "execution" not in inspect.signature(runtime.RuntimeOptions).parameters
    with pytest.raises(FrozenInstanceError):
        options.backend = "host"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    (
        ({"backend": "gpu"}, ValueError, "backend.*auto.*jax.*host"),
        ({"backend": object()}, TypeError, "backend.*execute"),
        ({"workflow": object()}, TypeError, "workflow.*build"),
    ),
)
def test_runtime_options_validates_v0_4_extension_contracts(
    kwargs: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        runtime.RuntimeOptions(**cast(Any, kwargs))


def test_sequential_workflow_builds_one_frozen_plan_per_clock_step() -> None:
    context = runtime.WorkflowContext(
        clock=_clock(steps=3),
        component_names=("A", "B"),
        default_order=("B", "A"),
    )

    plan = runtime.SequentialWorkflow().build(context)

    assert plan == runtime.ExecutionPlan(
        steps=(
            runtime.StepPlan(step=0, components=("B", "A")),
            runtime.StepPlan(step=1, components=("B", "A")),
            runtime.StepPlan(step=2, components=("B", "A")),
        )
    )
    with pytest.raises(FrozenInstanceError):
        plan.steps = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.steps[0].components = ()  # type: ignore[misc]


def test_sequential_workflow_builds_an_empty_plan_for_zero_steps() -> None:
    context = runtime.WorkflowContext(
        clock=_clock(steps=0),
        component_names=("A",),
        default_order=("A",),
    )

    assert runtime.SequentialWorkflow().build(context) == runtime.ExecutionPlan()


def test_static_planning_containers_copy_sequence_inputs_to_tuples() -> None:
    component_names = ["A"]
    default_order = ["A"]
    step_components = ["A"]
    execution_component_names = ["A"]
    plans = [runtime.StepPlan(step=0, components=cast(Any, step_components))]
    context = runtime.WorkflowContext(
        clock=_clock(steps=1),
        component_names=cast(Any, component_names),
        default_order=cast(Any, default_order),
    )
    execution_plan = runtime.ExecutionPlan(steps=cast(Any, plans))
    chunk = runtime.ExecutionChunk(steps=cast(Any, plans))
    execution_context = runtime.ExecutionContext(
        clock=_clock(steps=1),
        component_names=cast(Any, execution_component_names),
        options=runtime.RuntimeOptions(),
    )

    component_names.append("B")
    default_order.clear()
    step_components.append("B")
    execution_component_names.append("B")
    plans.clear()

    assert context.component_names == ("A",)
    assert context.default_order == ("A",)
    assert execution_plan.steps == (runtime.StepPlan(step=0, components=("A",)),)
    assert chunk.steps == execution_plan.steps
    assert execution_context.component_names == ("A",)


@pytest.mark.parametrize(
    ("steps", "message"),
    (
        ((), "exactly 2.*got 0"),
        (
            (
                (0, ("A",)),
                (1, ("A",)),
                (2, ("A",)),
            ),
            "exactly 2.*got 3",
        ),
        (((1, ("A",)), (0, ("A",))), "step.*position 0.*expected 0.*got 1"),
        (((0, ("A",)), (0, ("A",))), "step.*position 1.*expected 1.*got 0"),
        (((0, ("UNKNOWN",)), (1, ("A",))), "unknown.*UNKNOWN"),
        (((0, ("A", "A")), (1, ("A",))), "step 0.*duplicate.*A"),
    ),
)
def test_workflow_output_is_completely_validated_before_stepping(
    steps: tuple[tuple[int, tuple[str, ...]], ...],
    message: str,
) -> None:
    observed: list[tuple[str, int]] = []
    workflow = _StaticWorkflow(
        tuple(runtime.StepPlan(step=step, components=names) for step, names in steps)
    )
    coupler = Coupler(
        _clock(),
        components=(_component("A", execution="host", observed=observed),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(workflow=workflow, backend="host"),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()

    assert observed == []
    assert len(workflow.contexts) == 1


@pytest.mark.parametrize(
    ("returned", "message"),
    (
        (None, "Workflow.build.*ExecutionPlan.*NoneType"),
        ((0, ("A",)), "Workflow.build.*ExecutionPlan.*tuple"),
    ),
)
def test_workflow_must_return_an_execution_plan(
    returned: object,
    message: str,
) -> None:
    class InvalidWorkflow:
        def build(self, context: object) -> object:
            _ = context
            return returned

    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(
            workflow=cast(runtime.Workflow, InvalidWorkflow())
        ),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()


@pytest.mark.parametrize(
    ("entries", "message"),
    (
        ((object(),), "entry 0.*StepPlan.*object"),
        (
            ((0, (1,)),),
            "step 0.*component names.*str.*int",
        ),
    ),
)
def test_workflow_rejects_invalid_step_plan_entries_and_component_name_types(
    entries: tuple[object, ...],
    message: str,
) -> None:
    first = entries[0]
    steps = (
        (runtime.StepPlan(step=first[0], components=cast(Any, first[1])),)
        if isinstance(first, tuple)
        else entries
    )
    workflow = _StaticWorkflow(steps)
    coupler = Coupler(
        _clock(steps=1),
        components=(_component("A", execution="host"),),
        run_order=("A",),
        runtime=runtime.RuntimeOptions(workflow=workflow, backend="host"),
    )

    with pytest.raises(CouplerError, match=message):
        coupler.run()


def test_custom_workflow_can_reorder_and_omit_registered_components() -> None:
    observed: list[tuple[str, int]] = []
    workflow = _StaticWorkflow(
        (
            runtime.StepPlan(step=0, components=("B", "A")),
            runtime.StepPlan(step=1, components=("A",)),
        )
    )
    backend = _SequentialBackend()
    coupler = Coupler(
        _clock(),
        components=(
            _component("A", execution="host", observed=observed),
            _component("B", execution="host", observed=observed),
        ),
        run_order=("A", "B"),
        runtime=runtime.RuntimeOptions(workflow=workflow, backend=backend),
    )

    final_state = coupler.run()

    assert observed == [("B", 0), ("A", 0), ("A", 1)]
    assert len(backend.calls) == 2
    assert [
        tuple(plan.components for plan in cast(Any, chunk).steps)
        for _, chunk in backend.calls
    ] == [(("B", "A"),), (("A",),)]
    public_context = cast(Any, backend.calls[0][0])
    assert public_context.clock is coupler.clock
    assert public_context.component_names == ("A", "B")
    assert public_context.options.backend is backend
    assert_allclose_compact(
        final_state.component("A").field("value"),
        jnp.full((2, 2), 2.0),
    )
    assert_allclose_compact(
        final_state.component("B").field("value"),
        jnp.full((2, 2), 1.0),
    )


def test_multi_chunk_workflow_preserves_absolute_step_indices_and_times() -> None:
    observed: list[tuple[str, int, object]] = []
    grid = make_test_grid(name="workflow-absolute-chunks")

    def make_component(name: str) -> CallableComponent:
        def step(
            fields: Mapping[str, Any],
            context: StepContext,
        ) -> Mapping[str, Any]:
            observed.append((name, int(context.step), context.time))
            return {"value": fields["value"] + context.step + 1.0}

        return CallableComponent(
            name,
            grid,
            step,
            spec=ComponentSpec(
                inputs=("value",),
                outputs=("value",),
                initial_fields={"value": 0.0},
                execution="host",
            ),
        )

    workflow = _StaticWorkflow(
        (
            runtime.StepPlan(step=0, components=("A",)),
            runtime.StepPlan(step=1, components=("B",)),
            runtime.StepPlan(step=2, components=("A",)),
        )
    )
    backend = _SequentialBackend()
    coupler = Coupler(
        _clock(steps=3),
        components=(make_component("A"), make_component("B")),
        run_order=("A", "B"),
        runtime=runtime.RuntimeOptions(workflow=workflow, backend=backend),
    )

    final_state = coupler.run()

    assert observed == [
        ("A", 0, datetime(2000, 1, 1, 0, 0)),
        ("B", 1, datetime(2000, 1, 1, 0, 1)),
        ("A", 2, datetime(2000, 1, 1, 0, 2)),
    ]
    assert_allclose_compact(
        final_state.component("A").field("value"),
        jnp.full(grid.shape, 4.0),
    )
    assert_allclose_compact(
        final_state.component("B").field("value"),
        jnp.full(grid.shape, 2.0),
    )


def test_workflow_can_schedule_and_prime_a_registered_host_component_omitted_from_default_order() -> (
    None
):
    grid = make_test_grid(name="workflow-planned-producer")
    observed: list[tuple[str, int]] = []

    def step_source(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        observed.append(("SOURCE", int(context.step)))
        return {"signal": fields["signal"] + 1.0}

    source = CallableComponent(
        "SOURCE",
        grid,
        step_source,
        spec=ComponentSpec(
            outputs=("signal",),
            initial_fields={"signal": 5.0},
            execution="host",
        ),
    )

    def step_target(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        observed.append(("TARGET", int(context.step)))
        return {"total": fields["total"] + fields["signal"]}

    target = CallableComponent(
        "TARGET",
        grid,
        step_target,
        spec=ComponentSpec(
            inputs=("signal", "total"),
            outputs=("total",),
            initial_fields={"total": 0.0},
            execution="jax",
        ),
    )
    workflow = _StaticWorkflow(
        (runtime.StepPlan(step=0, components=("TARGET", "SOURCE")),)
    )
    coupler = Coupler(
        _clock(steps=1),
        components=(source, target),
        exchanges=(Exchange("SOURCE", "TARGET", ("signal",), quantity="intensive"),),
        run_order=("TARGET",),
        runtime=runtime.RuntimeOptions(workflow=workflow),
    )

    final_state = coupler.run()

    assert observed == [("TARGET", 0), ("SOURCE", 0)]
    assert_allclose_compact(
        final_state.component("TARGET").field("total"),
        jnp.full(grid.shape, 5.0),
    )


def test_dormant_omitted_host_component_does_not_force_auto_host_backend() -> None:
    traced_steps: list[bool] = []
    grid = make_test_grid(name="workflow-dormant-host")

    def step_active(
        fields: Mapping[str, Any],
        context: StepContext,
    ) -> Mapping[str, Any]:
        traced_steps.append(isinstance(context.step, jax.core.Tracer))
        return {"value": fields["value"] + 1.0}

    active = CallableComponent(
        "ACTIVE",
        grid,
        step_active,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            initial_fields={"value": 0.0},
            execution="jax",
        ),
    )
    dormant = _component("DORMANT", execution="host")
    coupler = Coupler(
        _clock(steps=2),
        components=(active, dormant),
        run_order=("ACTIVE",),
    )

    coupler.run()

    assert traced_steps == [True]
