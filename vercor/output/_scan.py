"""Compose pure numeric output reduction with the shared scanned runtime step.

Only fixed-shape sums, counts and latest coordinates accompany RunState; no
trajectory or host timestamp enters the carry. Host coordination owns cadence
boundaries and writes. Abstract preparation establishes the first post-step
schema without advancing actual model state or accumulating an initial sample.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.errors import JaxRuntimeError

from vercor.output import JaxOutputProvider
from vercor.output._session import _OutputPlan, _OutputSession, initial_output_session
from vercor._runtime.backends import _RuntimeExecutionData, build_jax_runtime_step
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.time import RuntimeStepInfo
from vercor.runtime import ExecutionChunk
from vercor.state import RunState
from vercor.types import RuntimeArray

_OutputCarry = tuple[RunState, _OutputSession]
_JaxOutputExecutor = Callable[
    [_OutputCarry, RuntimeArray, RuntimeStepInfo], _OutputCarry
]


def supports_scanned_output(plan: _OutputPlan) -> bool:
    """Require explicit numeric capability from every enabled provider."""
    return all(
        isinstance(schema.provider, JaxOutputProvider) for schema in plan.schemas
    )


def initial_jax_output_session(
    plan: _OutputPlan,
    state: RunState,
    *,
    components: tuple[str, ...],
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
) -> _OutputSession:
    """Abstractly trace one post-step frame and allocate only zero accumulators."""
    runtime_step = build_jax_runtime_step(
        components,
        context=context,
        execution_data=execution_data,
    )
    step = execution_data.step_indices[0]
    info = cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(lambda x: x[0], execution_data.step_infos),
    )

    def first_frame(seed: RunState) -> _OutputSession:
        post_step = runtime_step(seed, step, info)
        return initial_output_session(plan).accumulate_jax(
            plan,
            post_step,
            step=step,
            dt_seconds=context.clock.dt_seconds,
        )

    abstract = jax.eval_shape(first_frame, state)
    return cast(
        _OutputSession,
        jax.tree_util.tree_map(
            lambda leaf: jnp.zeros(leaf.shape, dtype=leaf.dtype),
            abstract,
        ),
    )


def build_jax_output_executor(
    components: tuple[str, ...],
    *,
    plan: _OutputPlan,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
) -> _JaxOutputExecutor:
    """Compile one schedule's scanned step plus numeric provider accumulation."""
    runtime_step = build_jax_runtime_step(
        components, context=context, execution_data=execution_data
    )

    def scanned_chunk(
        carry: _OutputCarry, indices: RuntimeArray, infos: RuntimeStepInfo
    ) -> _OutputCarry:
        def step(
            carry: _OutputCarry, inputs: tuple[RuntimeArray, Any]
        ) -> tuple[_OutputCarry, None]:
            state, session = carry
            index, info = inputs
            state = runtime_step(state, index, info)
            _validate_step_lifecycle(state, context=context)
            session = session.accumulate_jax(
                plan, state, step=index, dt_seconds=context.clock.dt_seconds
            )
            return (state, session), None

        result, _ = jax.lax.scan(
            jax.checkpoint(step), carry, (indices, infos), length=indices.shape[0]
        )
        return result

    return cast(_JaxOutputExecutor, jax.jit(scanned_chunk))


def _validate_step_lifecycle(state: RunState, *, context: RuntimeRunContext) -> None:
    """Preserve concrete author validation after every actual output step.

    Hooks may use arbitrary Python checks. Pass only the relevant component
    PyTree through an ordered callback; numeric and schema validation remain
    with their existing runtime owners. No hook runs during abstract preparation.
    """
    for name, component in context.dispatch_context.components.items():
        if component.spec.lifecycle.validate is None:
            continue
        contract = context.dispatch_context.contracts[name]

        def validate(
            component_state: Any, component: Any = component, contract: Any = contract
        ) -> None:
            component._validate_runtime_state(component_state, contract)

        jax.debug.callback(validate, state._component_state(name), ordered=True)


def execute_jax_output_chunk(
    state: RunState,
    session: _OutputSession,
    *,
    chunk: ExecutionChunk,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
    executor: _JaxOutputExecutor,
) -> _OutputCarry:
    """Execute a core-owned uniform slice and translate callback interruption."""
    start = chunk.steps[0].step
    stop = chunk.steps[-1].step + 1
    infos = cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(lambda x: x[start:stop], execution_data.step_infos),
    )
    try:
        return executor(
            (state, session), execution_data.step_indices[start:stop], infos
        )
    except JaxRuntimeError as error:
        context.interrupts.raise_if_jax_callback_interrupted(
            error, "compiled scanned output runtime"
        )
