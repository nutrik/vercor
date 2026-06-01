from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentFieldSpec,
    FieldNames,
)
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.dtypes import PrecisionPolicy
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contexts import ComponentStepContext
    from vercor.runtime.state import RuntimeComponentState


class ComponentRuntimeProtocol(Protocol):
    """Private structural contract shared by component helper modules."""

    name: str
    grid: RectilinearGrid
    settings: VercorSettings

    @property
    def field_spec(self) -> ComponentFieldSpec:
        """Return the component's declared runtime field contract."""
        ...


class ComponentAuthoringProtocol(ComponentRuntimeProtocol, Protocol):
    """Private structural contract for helpers that call authoring methods."""

    _lifecycle_hooks: ComponentLifecycleHooks

    def declare_fields(
        self,
        field_spec: ComponentFieldSpec | None = None,
        *,
        inputs: FieldNames = (),
        outputs: FieldNames = (),
        default_fields: AuthorFieldValues = None,
    ) -> ComponentFieldSpec:
        """Declare runtime data fields for a component."""
        ...

    def seed_declared_defaults(
        self,
        policy: PrecisionPolicy = None,
    ) -> object:
        """Seed declared default fields on a component."""
        ...

    def runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> dict[str, RuntimeArray]:
        """Return runtime data fields as a mapping."""
        ...


class ComponentExecutionProtocol(ComponentRuntimeProtocol, Protocol):
    """Private structural contract for components that step runtime state."""

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: "ComponentStepContext",
    ) -> "RuntimeComponentState":
        """Return this component advanced by one differentiable runtime step."""
        ...


@runtime_checkable
class HostRuntimeExecutionProtocol(ComponentExecutionProtocol, Protocol):
    """Private structural contract for components that require host stepping."""

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: "ComponentStepContext",
    ) -> "RuntimeComponentState":
        """Return this component advanced by one Python host runtime step."""
        ...


__all__ = [
    "ComponentAuthoringProtocol",
    "ComponentExecutionProtocol",
    "ComponentRuntimeProtocol",
    "HostRuntimeExecutionProtocol",
]
