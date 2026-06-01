from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from vercor.calendar import ModelDateTime
from vercor.components.contexts import ComponentStepContext
from vercor.components.runtime_execution import step_component_runtime_state
from vercor.jax_logging import LoggerLike
from vercor.runtime.dispatch_context import RuntimeDispatchContext
from vercor.runtime.exchange_dispatch import dispatch_component_exchanges
from vercor.runtime.field_transfer import receive_runtime_fields, send_runtime_fields
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import RuntimeStepInfo


def step_runtime_component(
    runtime_state: RuntimeCouplerState,
    component_name: str,
    step_info: RuntimeStepInfo,
    *,
    dispatch_context: RuntimeDispatchContext,
    allow_host_runtime: bool,
    time: datetime | ModelDateTime | None = None,
    logger: LoggerLike | None = None,
) -> RuntimeCouplerState:
    """Advance one component through dispatch, receive, step, and send phases."""

    runtime_state = dispatch_component_exchanges(
        runtime_state,
        component_name,
        dispatch_context.destination_exchanges(component_name),
        dispatch_context.regridders,
    )
    component_state = runtime_state.get_component_state(component_name)
    component = dispatch_context.components[component_name]
    contract = dispatch_context.contracts[component_name]
    component_state = receive_runtime_fields(
        component_state,
        contract,
    )
    step_context = ComponentStepContext(
        dt_seconds=dispatch_context.dt_seconds,
        settings=dispatch_context.settings,
        time=time,
        logger=logger,
    )
    component_state = step_component_runtime_state(
        component,
        component_state,
        step_context,
        allow_host_runtime=allow_host_runtime,
    )
    component_state = send_runtime_fields(
        component,
        component_state,
        step_info,
        contract=contract,
    )
    return runtime_state.set_component_state(
        component_name,
        component_state,
    )


def prime_runtime_outgoing(
    runtime_state: RuntimeCouplerState,
    run_sequence: Sequence[str],
    *,
    dispatch_context: RuntimeDispatchContext,
    step_info: RuntimeStepInfo,
) -> RuntimeCouplerState:
    """Populate outgoing stores once before the first exchange dispatch."""

    for component_name in run_sequence:
        component_state = runtime_state.get_component_state(component_name)
        component_state = send_runtime_fields(
            dispatch_context.components[component_name],
            component_state,
            step_info,
            contract=dispatch_context.contracts[component_name],
        )
        runtime_state = runtime_state.set_component_state(
            component_name,
            component_state,
        )
    return runtime_state
