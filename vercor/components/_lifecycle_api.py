from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import vercor.components._runtime_fields as _runtime_field_adapters
import vercor.components._runtime_validation as _runtime_field_validation
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.components.contexts import ComponentSetupContext
    from vercor.runtime.contracts import RuntimeComponentContract
    from vercor.runtime.state import RuntimeComponentState


class ComponentLifecycleMixin:
    """Default component lifecycle hook dispatch used by factory and subclasses."""

    def _lifecycle_component(self) -> "Component":
        """Return this mixin instance as the concrete component type."""

        return cast("Component", self)

    def initialize(
        self,
        context: "ComponentSetupContext",
    ) -> None:
        """Optionally initialize component-owned runtime data before coupling."""

        component = self._lifecycle_component()
        hook = component._lifecycle_hooks.initialize
        if hook is not None:
            hook(component, context)
            return
        component.seed_declared_defaults(context.settings)

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state."""

        component = self._lifecycle_component()
        hook = component._lifecycle_hooks.create_runtime_payload
        if hook is not None:
            return hook(component)
        return self._default_runtime_payload()

    def _default_runtime_payload(self) -> Any | None:
        """Return the payload used when no lifecycle hook is installed."""

        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: "RuntimeComponentContract",
    ) -> None:
        """Optionally pre-seed fields required by runtime execution."""

        component = self._lifecycle_component()
        hook = component._lifecycle_hooks.prefill_runtime_state_fields
        if hook is not None:
            hook(component, data, incoming, outgoing, contract)
            return
        _runtime_field_adapters.prefill_declared_runtime_fields(component, data)
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: "RuntimeComponentContract",
    ) -> None:
        """Optionally validate component-specific runtime fields before execution."""

        component = self._lifecycle_component()
        hook = component._lifecycle_hooks.validate_runtime_state
        if hook is not None:
            hook(component, component_state, contract)
            return
        _ = contract
        _runtime_field_validation.validate_declared_runtime_fields(
            component,
            component_state,
        )


__all__ = ["ComponentLifecycleMixin"]
