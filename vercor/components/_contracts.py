from __future__ import annotations

from collections.abc import Iterable

from vercor.components.contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    ComponentFieldSpec,
    ComponentStepCallable,
    ComponentStepResult,
    ComponentStepReturn,
    FieldDefaults,
    FieldNames,
)
from vercor.components._field_names import unique_field_names
from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.field_layout import validate_component_data_layout
from vercor.grid import RectilinearGrid
from vercor.types import RuntimeArray


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


__all__ = [
    "AuthorFieldValues",
    "AuthorStepCallable",
    "ComponentFieldSpec",
    "ComponentStepCallable",
    "ComponentStepResult",
    "ComponentStepReturn",
    "FieldDefaults",
    "FieldNames",
    "declared_runtime_field_names",
    "merge_component_outputs",
    "normalize_author_field_values",
    "unique_field_names",
]
