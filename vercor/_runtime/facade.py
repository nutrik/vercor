from __future__ import annotations

from datetime import datetime
import vercor.output._runtime as _runtime_output
from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.jax_logging import LoggerLike
from vercor._runtime.preparation import (
    create_runtime_state,
    prepare_runtime_state,
    validate_runtime_state,
)
from vercor._runtime.execution import (
    build_validated_execution_plan,
    execute_plan,
)
from vercor._runtime.prepared import PreparedCoupling, prepare_coupling
from vercor._runtime.run_context import RuntimeRunContext
from vercor._runtime.exchange_masks import runtime_exchange_masks
from vercor.state import RunState
from vercor.output import OutputTarget
from vercor.output._session import validate_output_run_state_not_traced


def run(
    runtime_state: RunState,
    *,
    prepared: PreparedCoupling,
    logger: LoggerLike,
    output: OutputTarget | None = None,
) -> RunState:
    """Run a validated runtime state through the selected runtime path."""

    if output is not None and output.enabled:
        validate_output_run_state_not_traced(runtime_state)
    with prepared.interrupts.signal_scope():
        context = RuntimeRunContext(
            run_order=prepared.run_order,
            clock=prepared.clock,
            logger=logger,
            dispatch_context=prepared.dispatch_context,
            interrupts=prepared.interrupts,
            options=prepared.runtime,
            output=output,
        )
        plan = build_validated_execution_plan(context)
        final_state = execute_plan(runtime_state, plan=plan, context=context)
        if output is not None and output.enabled:
            if output.write_final_fields:
                binary_masks, fractional_masks = runtime_exchange_masks(
                    final_state, context=prepared.dispatch_context
                )
                prepared.interrupts.checkpoint("runtime output")
                _runtime_output.write_coupler_runtime_outputs(
                    final_state=final_state,
                    components=prepared.components,
                    exchanges=prepared.exchanges,
                    binary_masks=binary_masks,
                    fractional_masks=fractional_masks,
                    output_dir=output.directory,
                    logger=logger,
                )
                prepared.interrupts.checkpoint("runtime output")
            if output.write_snapshots:
                prepared.interrupts.checkpoint("runtime output")
                _runtime_output.write_coupler_component_snapshots(
                    final_state=final_state,
                    components=prepared.components,
                    output_time=_final_snapshot_time(prepared.clock),
                    output_dir=output.directory,
                    logger=logger,
                )
                prepared.interrupts.checkpoint("runtime output")
        return final_state


def _final_snapshot_time(clock: Clock) -> datetime | ModelDateTime:
    """Return the model time represented by the final runtime state."""

    final_time: datetime | ModelDateTime = clock.start
    for _, time, dt in clock.iter():
        final_time = time + dt
    return final_time


__all__ = [
    "create_runtime_state",
    "prepare_coupling",
    "run",
    "prepare_runtime_state",
    "validate_runtime_state",
]
