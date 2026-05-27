from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.components.base import Component, HostRuntimeComponent

if TYPE_CHECKING:
    from vercor.runtime.contexts import RuntimeStepContext
    from vercor.runtime.state import RuntimeComponentState


def component_requires_host_runtime(component: Component) -> bool:
    """Return whether ``component`` must run through the Python host runtime."""

    return isinstance(component, HostRuntimeComponent)


def host_component_names(components: Mapping[str, Component]) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if component_requires_host_runtime(component)
    ]


def step_component_runtime_state(
    component: Component,
    component_state: RuntimeComponentState,
    context: RuntimeStepContext,
    *,
    allow_host_runtime: bool,
) -> RuntimeComponentState:
    """Advance ``component_state`` through the component's selected runtime path."""

    if allow_host_runtime and isinstance(component, HostRuntimeComponent):
        return component.step_host_runtime_state(
            component_state,
            context,
        )
    return component.step_runtime_state(
        component_state,
        context,
    )
