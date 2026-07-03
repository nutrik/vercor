from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final, TypeAlias

from vercor._deprecation import deprecated_getattr, warn_deprecated_name
from vercor.components.contexts import (
    SetupContext,
    StepContext,
)
from vercor.components._field_names import unique_field_names as _unique_field_names
from vercor.types import RuntimeArray

KEEP_PAYLOAD: Final = object()
"""Sentinel meaning a component step should preserve the existing payload."""


@dataclass(frozen=True)
class StepResult:
    """Result returned by callable component wrappers.

    Attributes:
        fields: Runtime data fields to update.
        payload: Replacement runtime payload, or ``KEEP_PAYLOAD`` to preserve
            the existing payload. Pass ``None`` explicitly to clear the payload.
    """

    fields: Mapping[str, RuntimeArray] = field(default_factory=dict)
    payload: Any = KEEP_PAYLOAD


ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | StepResult
ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], StepContext, Any | None],
    ComponentStepReturn,
]
AuthorStepCallable: TypeAlias = Callable[..., ComponentStepReturn]
FieldNames: TypeAlias = Iterable[str]
AuthorFieldValues: TypeAlias = Mapping[str, object] | None
ComponentInitializeHook = Callable[[Any, SetupContext], None]
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


@dataclass(frozen=True)
class ComponentHooks:
    """Optional lifecycle hooks for component setup and runtime customization."""

    initialize: ComponentInitializeHook | None = None
    create_payload: ComponentCreatePayloadHook | None = None
    prefill: ComponentPrefillHook | None = None
    validate: ComponentValidateHook | None = None


@dataclass(frozen=True, init=False)
class FieldSpec:
    """Author-facing declaration of a component's runtime data-field contract.

    Attributes:
        inputs: Fields the model expects to read from runtime data.
        outputs: Fields the model may write. These are pre-seeded as grid-shaped
            zeros before traced runtime execution.
        defaults: Field defaults used when runtime state is created.
    """

    inputs: FieldNames = ()
    outputs: FieldNames = ()
    defaults: Mapping[str, object] = field(default_factory=dict)

    def __init__(
        self,
        inputs: FieldNames = (),
        outputs: FieldNames = (),
        defaults: Mapping[str, object] | None = None,
        *,
        default_fields: Mapping[str, object] | None = None,
    ) -> None:
        """Create a field declaration.

        ``default_fields`` is the legacy keyword for ``defaults`` and remains
        accepted during the v2 migration window.
        """

        if defaults is not None and default_fields is not None:
            raise TypeError("Use either defaults or default_fields, not both")
        normalized_defaults = defaults if defaults is not None else default_fields
        if default_fields is not None:
            warn_deprecated_name("default_fields", "defaults", remove_in="0.2.0")
        object.__setattr__(self, "inputs", _unique_field_names(inputs))
        object.__setattr__(self, "outputs", _unique_field_names(outputs))
        object.__setattr__(self, "defaults", dict(normalized_defaults or {}))

    @property
    def default_fields(self) -> Mapping[str, object]:
        """Return legacy ``default_fields`` view of ``defaults``."""

        return self.defaults


__all__ = [
    "AuthorFieldValues",
    "AuthorStepCallable",
    "ComponentCreatePayloadHook",
    "ComponentHooks",
    "ComponentInitializeHook",
    "ComponentPrefillHook",
    "ComponentStepCallable",
    "ComponentStepReturn",
    "ComponentValidateHook",
    "FieldNames",
    "FieldSpec",
    "KEEP_PAYLOAD",
    "SetupContext",
    "StepContext",
    "StepResult",
]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "ComponentFieldSpec": ("vercor.components.contracts.FieldSpec", FieldSpec),
        "ComponentStepResult": ("vercor.components.contracts.StepResult", StepResult),
        "ComponentSetupContext": (
            "vercor.components.contexts.SetupContext",
            SetupContext,
        ),
        "ComponentStepContext": (
            "vercor.components.contexts.StepContext",
            StepContext,
        ),
    },
    remove_in="0.2.0",
)
