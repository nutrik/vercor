from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.components.contexts import StepContext
from vercor.components.runtime_execution import step_component_runtime_state
from vercor.jax_logging import LoggerLike
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.exchange_dispatch import dispatch_component_exchanges
from vercor._runtime.exchange_masks import refresh_runtime_masks
from vercor._runtime.field_transfer import receive_runtime_fields, send_runtime_fields
from vercor.state import RunState
from vercor._runtime.time import RuntimeStepInfo
from vercor.types import RuntimeArray


def step_runtime_component(
    runtime_state: RunState,
    component_name: str,
    step_info: RuntimeStepInfo,
    *,
    dispatch_context: RuntimeDispatchContext,
    allow_host_runtime: bool,
    time: datetime | ModelDateTime | None = None,
    logger: LoggerLike | None = None,
    step: int | RuntimeArray = 0,
) -> RunState:
    """Advance one component through dispatch, receive, step, and send phases."""

    runtime_state = dispatch_component_exchanges(
        runtime_state,
        component_name,
        dispatch_context.destination_exchanges(component_name),
        step_info,
        context=dispatch_context,
    )
    component_state = runtime_state._component_state(component_name)
    component = dispatch_context.components[component_name]
    contract = dispatch_context.contracts[component_name]
    component_state = receive_runtime_fields(
        component_state,
        contract,
    )
    step_context = StepContext(
        dt_seconds=dispatch_context.dt_seconds,
        constants=dispatch_context.constants,
        dtype=dispatch_context.dtype,
        time=time,
        logger=logger,
        step=step,
    )
    component_state = step_component_runtime_state(
        component,
        component_state,
        step_context,
        allow_host_runtime=allow_host_runtime,
    )
    runtime_state = refresh_runtime_masks(
        runtime_state._with_component_state(component_name, component_state),
        context=dispatch_context,
    )
    component_state = send_runtime_fields(
        component,
        component_state,
        step_info,
        contract=contract,
        dtype=dispatch_context.dtype,
    )
    return runtime_state._with_component_state(
        component_name,
        component_state,
    )


def prime_runtime_outgoing(
    runtime_state: RunState,
    run_order: Sequence[str],
    *,
    dispatch_context: RuntimeDispatchContext,
    step_info: RuntimeStepInfo,
) -> RunState:
    """Populate outgoing stores once before the first exchange dispatch."""

    for component_name in run_order:
        component_state = runtime_state._component_state(component_name)
        component_state = send_runtime_fields(
            dispatch_context.components[component_name],
            component_state,
            step_info,
            contract=dispatch_context.contracts[component_name],
            dtype=dispatch_context.dtype,
        )
        runtime_state = runtime_state._with_component_state(
            component_name,
            component_state,
        )
    return runtime_state
