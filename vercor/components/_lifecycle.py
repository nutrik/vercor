from __future__ import annotations

from dataclasses import dataclass

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


__all__ = [
    "ComponentLifecycleHooks",
]
