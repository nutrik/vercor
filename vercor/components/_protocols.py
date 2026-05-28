from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentFieldSpec,
    FieldNames,
)
from vercor.dtypes import PrecisionPolicy
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
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

    _lifecycle_hooks: Any

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


__all__ = ["ComponentAuthoringProtocol", "ComponentRuntimeProtocol"]
