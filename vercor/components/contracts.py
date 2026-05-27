from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.types import RuntimeArray

ComponentSetupContext = ComponentInitContext
ComponentStepContext = RuntimeStepContext


@dataclass(frozen=True)
class ComponentStepResult:
    """Result returned by callable component wrappers.

    Attributes:
        fields: Runtime data fields to update.
        payload: Replacement runtime payload. Use a plain mapping return from a
            callable step when the existing payload should be preserved.
    """

    fields: Mapping[str, RuntimeArray]
    payload: Any | None = None


ComponentStepReturn: TypeAlias = Mapping[str, RuntimeArray] | ComponentStepResult
ComponentStepCallable: TypeAlias = Callable[
    [Mapping[str, RuntimeArray], RuntimeStepContext, Any | None],
    ComponentStepReturn,
]
AuthorStepCallable: TypeAlias = Callable[..., ComponentStepReturn]
FieldNames: TypeAlias = Iterable[str]
FieldDefaults: TypeAlias = Mapping[str, RuntimeArray] | None
AuthorFieldValues: TypeAlias = Mapping[str, object] | None


@dataclass(frozen=True)
class ComponentFieldSpec:
    """Author-facing declaration of a component's runtime data-field contract.

    Attributes:
        inputs: Fields the model expects to read from runtime data.
        outputs: Fields the model may write. These are pre-seeded as grid-shaped
            zeros before traced runtime execution.
        default_fields: Field defaults used when runtime state is created.
    """

    inputs: FieldNames = ()
    outputs: FieldNames = ()
    default_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize field-name iterables once while preserving declaration order."""

        object.__setattr__(self, "inputs", _unique_field_names(self.inputs))
        object.__setattr__(self, "outputs", _unique_field_names(self.outputs))
        object.__setattr__(self, "default_fields", dict(self.default_fields))


def _unique_field_names(field_names: FieldNames) -> tuple[str, ...]:
    """Return field names without duplicates while preserving order."""

    unique: list[str] = []
    for field_name in field_names:
        if field_name not in unique:
            unique.append(field_name)
    return tuple(unique)


__all__ = [
    "AuthorFieldValues",
    "AuthorStepCallable",
    "ComponentFieldSpec",
    "ComponentSetupContext",
    "ComponentStepCallable",
    "ComponentStepContext",
    "ComponentStepResult",
    "ComponentStepReturn",
    "FieldDefaults",
    "FieldNames",
]
