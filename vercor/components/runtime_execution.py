from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.components._protocols import HostRuntimeExecutionProtocol

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contexts import ComponentStepContext
    from vercor.runtime.state import RuntimeComponentState


def host_component_names(
    components: Mapping[str, "Component"],
) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if isinstance(component, HostRuntimeExecutionProtocol)
    ]


def step_component_runtime_state(
    component: "Component",
    component_state: "RuntimeComponentState",
    context: "ComponentStepContext",
    *,
    allow_host_runtime: bool,
) -> "RuntimeComponentState":
    """Advance ``component_state`` through the component's selected runtime path."""

    if allow_host_runtime and isinstance(component, HostRuntimeExecutionProtocol):
        return component.step_host_runtime_state(
            component_state,
            context,
        )
    return component.step_runtime_state(
        component_state,
        context,
    )
