from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from vercor.calendar import ModelDateTime
from vercor.components.base import Component, HostRuntimeComponent
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.contexts import RuntimeStepContext
from vercor.runtime.exchange_dispatch import dispatch_component_exchanges
from vercor.runtime.field_transfer import receive_runtime_fields, send_runtime_fields
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import RuntimeStepInfo
from vercor.settings import VercorSettings


@dataclass(frozen=True)
class RuntimeDispatchContext:
    """Static runtime plumbing shared by per-component dispatch helpers."""

    components: Mapping[str, Component]
    exchanges: Sequence[Exchange]
    exchanges_by_destination: Mapping[str, tuple[Exchange, ...]]
    regridders: Mapping[tuple[str, str, str], Any]
    contracts: Mapping[str, RuntimeComponentContract]
    dt_seconds: float
    settings: VercorSettings

    def destination_exchanges(self, component_name: str) -> tuple[Exchange, ...]:
        """Return exchanges targeting ``component_name``."""

        return self.exchanges_by_destination.get(component_name, ())


def host_component_names(components: Mapping[str, Component]) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if isinstance(component, HostRuntimeComponent)
    ]


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
    step_context = RuntimeStepContext(
        dt_seconds=dispatch_context.dt_seconds,
        settings=dispatch_context.settings,
        time=time,
        logger=logger,
    )
    if allow_host_runtime and isinstance(component, HostRuntimeComponent):
        component_state = component.step_host_runtime_state(
            component_state,
            step_context,
        )
    else:
        component_state = component.step_runtime_state(
            component_state,
            step_context,
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
