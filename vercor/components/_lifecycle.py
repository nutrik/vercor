from __future__ import annotations

from collections.abc import Callable
from typing import Any

from vercor.runtime.contexts import ComponentInitContext
from vercor.types import RuntimeArray

ComponentInitializeHook = Callable[[Any, ComponentInitContext], None]
ComponentCreatePayloadHook = Callable[[Any], Any | None]
ComponentPrefillHook = Callable[
    [
        Any,
        dict[str, RuntimeArray],
        dict[str, RuntimeArray],
        dict[str, RuntimeArray],
        Any,
    ],
    None,
]
ComponentValidateHook = Callable[[Any, Any, Any], None]


def install_lifecycle_hooks(
    component: Any,
    *,
    initialize: ComponentInitializeHook | None = None,
    create_runtime_payload: ComponentCreatePayloadHook | None = None,
    prefill_runtime_state_fields: ComponentPrefillHook | None = None,
    validate_runtime_state: ComponentValidateHook | None = None,
) -> None:
    """Attach optional lifecycle hooks to a factory-created component."""

    if initialize is not None:
        setattr(component, "_initialize_hook", initialize)
    if create_runtime_payload is not None:
        setattr(component, "_create_runtime_payload_hook", create_runtime_payload)
    if prefill_runtime_state_fields is not None:
        setattr(
            component,
            "_prefill_runtime_state_fields_hook",
            prefill_runtime_state_fields,
        )
    if validate_runtime_state is not None:
        setattr(component, "_validate_runtime_state_hook", validate_runtime_state)
