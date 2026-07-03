from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


@runtime_checkable
class HostRuntimeExecutionProtocol(Protocol):
    """Private structural contract for components that require host stepping."""

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: "StepContext",
    ) -> "RuntimeComponentState":
        """Return this component advanced by one Python host runtime step."""
        ...


__all__ = [
    "HostRuntimeExecutionProtocol",
]
