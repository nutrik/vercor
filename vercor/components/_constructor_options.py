from __future__ import annotations

from vercor._deprecation import warn_deprecated_name
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.components.contracts import (
    AuthorFieldValues,
    ComponentCreatePayloadHook,
    ComponentHooks,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    FieldNames,
    FieldSpec,
)

_REMOVE_IN = "0.2.0"


def normalize_field_spec(
    *,
    fields: FieldSpec | None,
    inputs: FieldNames = (),
    outputs: FieldNames = (),
    defaults: AuthorFieldValues = None,
    default_fields: AuthorFieldValues = None,
) -> FieldSpec:
    """Normalize public field declaration options to one ``FieldSpec``."""

    if defaults is not None and default_fields is not None:
        raise TypeError("Use either defaults or default_fields, not both")
    if fields is not None and (
        tuple(inputs)
        or tuple(outputs)
        or defaults is not None
        or default_fields is not None
    ):
        raise TypeError(
            "Use either fields=FieldSpec(...) or inputs/outputs/defaults, not both"
        )

    normalized_defaults = defaults
    if default_fields is not None:
        warn_deprecated_name("default_fields", "defaults", remove_in=_REMOVE_IN)
        normalized_defaults = default_fields

    return fields or FieldSpec(
        inputs=inputs,
        outputs=outputs,
        defaults=normalized_defaults or {},
    )


def normalize_lifecycle_hooks(
    *,
    hooks: ComponentHooks | None,
    initialize: ComponentInitializeHook | None,
    create_runtime_payload: ComponentCreatePayloadHook | None,
    prefill_runtime_state_fields: ComponentPrefillHook | None,
    validate_runtime_state: ComponentValidateHook | None,
) -> ComponentLifecycleHooks:
    """Normalize public lifecycle hook options to one private hook container."""

    if hooks is not None and any(
        hook is not None
        for hook in (
            initialize,
            create_runtime_payload,
            prefill_runtime_state_fields,
            validate_runtime_state,
        )
    ):
        raise TypeError(
            "Use either hooks=ComponentHooks(...) or individual hook arguments, not both"
        )

    return ComponentLifecycleHooks(
        initialize=hooks.initialize if hooks is not None else initialize,
        create_runtime_payload=(
            hooks.create_payload if hooks is not None else create_runtime_payload
        ),
        prefill_runtime_state_fields=(
            hooks.prefill if hooks is not None else prefill_runtime_state_fields
        ),
        validate_runtime_state=(
            hooks.validate if hooks is not None else validate_runtime_state
        ),
    )


__all__ = ["normalize_field_spec", "normalize_lifecycle_hooks"]
