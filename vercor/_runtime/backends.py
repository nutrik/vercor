"""Chunk-oriented built-in and custom runtime backend implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, cast

import jax
from jax.errors import JaxRuntimeError

from vercor.calendar import ModelDateTime
from vercor.dtypes import as_jax_index_array
from vercor.exceptions import CouplerError
from vercor.runtime import (
    ExecutionBackend,
    ExecutionChunk,
    ExecutionContext,
    StepPlan,
)
from vercor.state import RunState
from vercor.types import RuntimeArray
from vercor._runtime.driver import step_runtime_component
from vercor._runtime.progress import (
    log_scanned_component_progress,
    log_scanned_step_progress,
    runtime_component_progress_message,
    runtime_step_progress_messages,
)
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.state_validation import (
    validate_runtime_state as validate_runtime_state_schema,
)
from vercor._runtime.time import (
    RuntimeStepInfo,
    build_runtime_step_info,
)

_ClockStep = tuple[int, datetime | ModelDateTime, timedelta]
_JaxChunkExecutor = Callable[
    [RunState, RuntimeArray, RuntimeStepInfo],
    RunState,
]


@dataclass(frozen=True)
class _RuntimeExecutionData:
    """Store clock-derived metadata built exactly once for one runtime run."""

    clock_steps: tuple[_ClockStep, ...]
    step_indices: RuntimeArray
    step_infos: RuntimeStepInfo
    progress_messages: tuple[str, ...]


def build_runtime_execution_data(
    context: RuntimeRunContext,
) -> _RuntimeExecutionData:
    """Precompute all clock-derived host and JAX metadata for one run."""

    clock_steps = tuple(context.clock.iter())
    return _RuntimeExecutionData(
        clock_steps=clock_steps,
        step_indices=as_jax_index_array(tuple(step for step, _, _ in clock_steps)),
        step_infos=build_runtime_step_info(
            context.clock,
            clock_steps=clock_steps,
        ),
        progress_messages=runtime_step_progress_messages(
            context.clock,
            clock_steps=clock_steps,
        ),
    )


def build_jax_chunk_executor(
    components: tuple[str, ...],
    *,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
) -> _JaxChunkExecutor:
    """Compile one reusable scanned executor for a component schedule."""

    def scanned_chunk(
        runtime_state: RunState,
        step_indices: RuntimeArray,
        step_infos: RuntimeStepInfo,
    ) -> RunState:
        def run_step(
            carry: RunState,
            scan_input: tuple[RuntimeArray, Any],
        ) -> tuple[RunState, None]:
            step_index, step_info = scan_input
            context.interrupts.scanned_checkpoint(
                "scanned runtime step",
                step_index,
            )
            log_scanned_step_progress(
                context.logger,
                step_index,
                execution_data.progress_messages,
            )
            for component_name in components:
                label = f"scanned runtime component {component_name}"
                context.interrupts.scanned_checkpoint(label, step_index)
                log_scanned_component_progress(context.logger, component_name)
                carry = step_runtime_component(
                    carry,
                    component_name,
                    step_info,
                    dispatch_context=context.dispatch_context,
                    allow_host_runtime=False,
                    logger=context.logger,
                    step=step_index,
                )
                context.interrupts.scanned_checkpoint(label, step_index)
            context.interrupts.scanned_checkpoint(
                "scanned runtime step",
                step_index,
            )
            return carry, None

        final_state, _ = jax.lax.scan(
            jax.checkpoint(run_step),
            runtime_state,
            (step_indices, step_infos),
            length=step_indices.shape[0],
        )
        return final_state

    return cast(_JaxChunkExecutor, jax.jit(scanned_chunk))


def execute_jax_chunk(
    state: RunState,
    *,
    chunk: ExecutionChunk,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
    executor: _JaxChunkExecutor,
) -> RunState:
    """Execute one uniform chunk with its schedule's reusable JAX executor."""

    if not chunk.steps:
        return state
    start = chunk.steps[0].step
    stop = chunk.steps[-1].step + 1
    step_infos = jax.tree_util.tree_map(
        lambda value: value[start:stop],
        execution_data.step_infos,
    )

    try:
        return executor(
            state,
            execution_data.step_indices[start:stop],
            cast(RuntimeStepInfo, step_infos),
        )
    except JaxRuntimeError as error:
        context.interrupts.raise_if_jax_callback_interrupted(
            error,
            "compiled scanned runtime",
        )


def execute_host_chunk(
    state: RunState,
    *,
    chunk: ExecutionChunk,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
) -> RunState:
    """Execute one chunk through the validated public driver on Python host."""

    driver = _RuntimeDriverAdapter(context, chunk, execution_data)
    for plan in chunk.steps:
        state = driver.run_step(state, plan)
    driver.ensure_complete()
    return state


def execute_custom_chunk(
    backend: ExecutionBackend,
    state: RunState,
    *,
    public_context: ExecutionContext,
    chunk: ExecutionChunk,
    context: RuntimeRunContext,
    execution_data: _RuntimeExecutionData,
) -> RunState:
    """Execute one custom-backend chunk with strict driver and result checks."""

    _validate_state(state, context=context, owner="Execution backend input")
    driver = _RuntimeDriverAdapter(context, chunk, execution_data)
    context.interrupts.checkpoint("custom runtime chunk")
    result = backend.execute(
        state,
        context=public_context,
        chunk=chunk,
        driver=driver,
    )
    context.interrupts.checkpoint("custom runtime chunk")
    backend_name = backend.__class__.__qualname__
    if not isinstance(result, RunState):
        raise CouplerError(
            f"Execution backend {backend_name}.execute(...) must return RunState; "
            f"got {type(result).__name__}."
        )
    _validate_state(result, context=context, owner="Execution backend result")
    driver.ensure_complete(backend_name=backend_name)
    return result


class _RuntimeDriverAdapter:
    """Validate and execute only the ordered plans in one active chunk."""

    def __init__(
        self,
        context: RuntimeRunContext,
        chunk: ExecutionChunk,
        execution_data: _RuntimeExecutionData,
    ) -> None:
        self._context = context
        self._chunk = chunk
        self._execution_data = execution_data
        self._cursor = 0

    def run_step(self, state: RunState, plan: StepPlan) -> RunState:
        """Advance the next exact chunk plan after strict state validation."""

        _validate_state(
            state,
            context=self._context,
            owner="RuntimeDriver.run_step state",
        )
        expected = self._expected_plan(plan)
        step, time, _ = self._execution_data.clock_steps[expected.step]
        self._context.interrupts.checkpoint("runtime step")
        self._context.logger.info(self._execution_data.progress_messages[expected.step])
        step_info = cast(
            RuntimeStepInfo,
            jax.tree_util.tree_map(
                lambda value: value[expected.step],
                self._execution_data.step_infos,
            ),
        )
        for component_name in expected.components:
            label = f"runtime component {component_name}"
            self._context.interrupts.checkpoint(label)
            self._context.logger.info(
                runtime_component_progress_message(component_name)
            )
            state = step_runtime_component(
                state,
                component_name,
                step_info,
                dispatch_context=self._context.dispatch_context,
                allow_host_runtime=True,
                time=time,
                logger=self._context.logger,
                step=step,
            )
            _validate_state(
                state,
                context=self._context,
                owner="RuntimeDriver.run_step result",
            )
            self._context.interrupts.checkpoint(label)
        self._context.interrupts.checkpoint("runtime step")
        self._cursor += 1
        return state

    def ensure_complete(self, *, backend_name: str = "Host backend") -> None:
        """Reject a backend that returned before consuming the whole chunk."""

        remaining = self._chunk.steps[self._cursor :]  # noqa: E203
        if not remaining:
            return
        label = "step" if len(remaining) == 1 else "steps"
        indices = ", ".join(str(plan.step) for plan in remaining)
        raise CouplerError(
            f"Execution backend {backend_name} did not execute {label} {indices} "
            "from its active chunk."
        )

    def _expected_plan(self, plan: StepPlan) -> StepPlan:
        """Return the next plan or reject forged, repeated, and reordered plans."""

        if not isinstance(plan, StepPlan):
            raise CouplerError(
                "RuntimeDriver.run_step plan must be a StepPlan; "
                f"got {type(plan).__name__}."
            )
        consumed = self._chunk.steps[: self._cursor]
        if any(plan is candidate for candidate in consumed):
            raise CouplerError(
                f"RuntimeDriver.run_step plan for step {plan.step} was already "
                "executed in the active chunk."
            )
        if self._cursor >= len(self._chunk.steps):
            raise CouplerError(
                "RuntimeDriver.run_step plan is outside the active execution chunk."
            )
        expected = self._chunk.steps[self._cursor]
        if plan is expected:
            return expected
        future = self._chunk.steps[self._cursor + 1 :]  # noqa: E203
        if any(plan is candidate for candidate in future):
            raise CouplerError(
                "RuntimeDriver.run_step plan is out of order in the active chunk: "
                f"expected step {expected.step}, got step {plan.step}."
            )
        raise CouplerError(
            "RuntimeDriver.run_step plan is outside the active execution chunk."
        )


def _validate_state(
    state: object,
    *,
    context: RuntimeRunContext,
    owner: str,
) -> None:
    """Validate one incoming or returned state against prepared runtime schema."""

    if not isinstance(state, RunState):
        raise CouplerError(f"{owner} must be a RunState; got {type(state).__name__}.")
    dispatch = context.dispatch_context
    validate_runtime_state_schema(
        state,
        components=dispatch.components,
        exchanges=dispatch.exchanges,
        regridders=dispatch.regridders,
        contracts=dispatch.contracts,
        run_order=tuple(dispatch.components),
    )


__all__ = [
    "build_jax_chunk_executor",
    "build_runtime_execution_data",
    "execute_custom_chunk",
    "execute_host_chunk",
    "execute_jax_chunk",
]
