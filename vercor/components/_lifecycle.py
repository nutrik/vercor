from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vercor.components.contracts import (
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
)


@dataclass(frozen=True)
class ComponentLifecycleHooks:
    """Optional lifecycle callbacks installed by component factories."""

    initialize: ComponentInitializeHook | None = None
    create_runtime_payload: ComponentCreatePayloadHook | None = None
    prefill_runtime_state_fields: ComponentPrefillHook | None = None
    validate_runtime_state: ComponentValidateHook | None = None

    def with_updates(
        self,
        *,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "ComponentLifecycleHooks":
        """Return hooks with supplied callbacks replacing existing callbacks."""

        return ComponentLifecycleHooks(
            initialize=self.initialize if initialize is None else initialize,
            create_runtime_payload=(
                self.create_runtime_payload
                if create_runtime_payload is None
                else create_runtime_payload
            ),
            prefill_runtime_state_fields=(
                self.prefill_runtime_state_fields
                if prefill_runtime_state_fields is None
                else prefill_runtime_state_fields
            ),
            validate_runtime_state=(
                self.validate_runtime_state
                if validate_runtime_state is None
                else validate_runtime_state
            ),
        )


class ComponentLifecycleOwner(Protocol):
    """Private structural contract for components that own lifecycle hooks."""

    _lifecycle_hooks: ComponentLifecycleHooks


def install_lifecycle_hooks(
    component: ComponentLifecycleOwner,
    *,
    hooks: ComponentLifecycleHooks | None = None,
    initialize: ComponentInitializeHook | None = None,
    create_runtime_payload: ComponentCreatePayloadHook | None = None,
    prefill_runtime_state_fields: ComponentPrefillHook | None = None,
    validate_runtime_state: ComponentValidateHook | None = None,
) -> None:
    """Attach optional lifecycle hooks to a factory-created component."""

    base_hooks = component._lifecycle_hooks
    if hooks is not None:
        base_hooks = base_hooks.with_updates(
            initialize=hooks.initialize,
            create_runtime_payload=hooks.create_runtime_payload,
            prefill_runtime_state_fields=hooks.prefill_runtime_state_fields,
            validate_runtime_state=hooks.validate_runtime_state,
        )
    component._lifecycle_hooks = base_hooks.with_updates(
        initialize=initialize,
        create_runtime_payload=create_runtime_payload,
        prefill_runtime_state_fields=prefill_runtime_state_fields,
        validate_runtime_state=validate_runtime_state,
    )


__all__ = [
    "ComponentLifecycleHooks",
    "ComponentLifecycleOwner",
    "install_lifecycle_hooks",
]
