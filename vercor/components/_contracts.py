from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.field_layout import validate_component_data_layout
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.types import RuntimeArray


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

        object.__setattr__(self, "inputs", unique_field_names(self.inputs))
        object.__setattr__(self, "outputs", unique_field_names(self.outputs))
        object.__setattr__(self, "default_fields", dict(self.default_fields))


def normalize_author_field_values(
    *,
    component_name: str,
    grid: RectilinearGrid,
    fields: AuthorFieldValues,
    policy: PrecisionPolicy = None,
) -> dict[str, RuntimeArray] | None:
    """Return author-provided fields as canonical runtime arrays.

    The additive authoring facade accepts scalar defaults for common setup cases.
    Scalars are expanded to grid-shaped constants; array-like values are converted
    to JAX arrays and then validated against VerCOR's canonical component-data
    layouts.
    """

    if fields is None:
        return None

    normalized: dict[str, RuntimeArray] = {}
    for field_name, field_value in fields.items():
        field_array = as_jax_real_array(field_value, policy)
        if field_array.shape == ():
            normalized[field_name] = jax_full(grid.shape, field_value, policy)
        else:
            normalized[field_name] = field_array

    validate_component_data_layout(
        component_name=component_name,
        grid_shape=grid.shape,
        data=normalized,
    )
    return normalized


def declared_runtime_field_names(field_spec: ComponentFieldSpec) -> tuple[str, ...]:
    """Return all fields that a declaration validates at runtime."""

    return unique_field_names(
        (
            *field_spec.inputs,
            *field_spec.outputs,
            *tuple(field_spec.default_fields),
        )
    )


def merge_component_outputs(
    field_spec: ComponentFieldSpec,
    output_names: Iterable[str],
) -> ComponentFieldSpec:
    """Return ``field_spec`` with additional output names merged in."""

    return ComponentFieldSpec(
        inputs=field_spec.inputs,
        outputs=unique_field_names((*field_spec.outputs, *tuple(output_names))),
        default_fields=field_spec.default_fields,
    )


def unique_field_names(field_names: FieldNames) -> tuple[str, ...]:
    """Return field names without duplicates while preserving order."""

    unique: list[str] = []
    for field_name in field_names:
        if field_name not in unique:
            unique.append(field_name)
    return tuple(unique)
