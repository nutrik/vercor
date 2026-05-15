from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor.components._contracts import (
    AuthorFieldValues,
    ComponentFieldSpec,
    ComponentStepResult,
    ComponentStepReturn,
    FieldNames,
    declared_runtime_field_names,
    normalize_author_field_values,
    unique_field_names,
)
from vercor.dtypes import PrecisionPolicy, jax_zeros
from vercor.exceptions import ComponentError
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.runtime import RuntimeComponentState


def runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
) -> dict[str, RuntimeArray]:
    """Return runtime data fields as a plain name-to-array mapping."""

    _ = component
    return component_state.data.to_mapping()


def runtime_field(
    component: "Component",
    component_state: "RuntimeComponentState",
    name: str,
) -> RuntimeArray:
    """Return one runtime data field with a component-oriented error."""

    try:
        return component_state.data.get(name)
    except KeyError as exc:
        raise ComponentError(
            f"Runtime data field '{name}' is missing for component '{component.name}'."
        ) from exc


def has_runtime_field(
    component: "Component",
    component_state: "RuntimeComponentState",
    name: str,
) -> bool:
    """Return whether one runtime data field exists."""

    _ = component
    return name in component_state.data


def runtime_field_or(
    component: "Component",
    component_state: "RuntimeComponentState",
    name: str,
    default: object,
    policy: PrecisionPolicy = None,
) -> RuntimeArray:
    """Return one runtime field or a grid-shaped/default array fallback."""

    if name in component_state.data:
        return runtime_field(component, component_state, name)
    normalized = normalize_author_field_values(
        component_name=component.name,
        grid=component.grid,
        fields={name: default},
        policy=component.settings if policy is None else policy,
    )
    if normalized is None:
        raise ComponentError(
            f"Default runtime field '{name}' could not be normalized for "
            f"component '{component.name}'."
        )
    return component_state.data.get_or(name, normalized[name])


def runtime_field_or_zeros_like(
    component: "Component",
    component_state: "RuntimeComponentState",
    name: str,
    like: str | RuntimeArray,
) -> RuntimeArray:
    """Return one runtime field or zeros matching another field/array."""

    try:
        return component_state.data.get_or_zeros_like(name, like)
    except KeyError as exc:
        missing_name = like if isinstance(like, str) else name
        raise ComponentError(
            f"Runtime data field '{missing_name}' is missing for component "
            f"'{component.name}'."
        ) from exc


def with_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
    fields: Mapping[str, RuntimeArray],
) -> "RuntimeComponentState":
    """Return ``component_state`` with existing runtime data fields updated."""

    missing_field = next(
        (field_name for field_name in fields if field_name not in component_state.data),
        None,
    )
    if missing_field is not None:
        raise ComponentError(
            f"Component '{component.name}' returned update for runtime data "
            f"field '{missing_field}', but it is missing from runtime data. "
            "Seed the field with seed_field()/seed_fields(), include it in "
            "factory fields, declare it as an output/default in "
            "from_model()/declare_fields(), or declare it through an "
            "exchange before runtime execution."
        )
    return component_state.with_data(component_state.data.replace_many(fields))


def apply_step_result(
    component: "Component",
    component_state: "RuntimeComponentState",
    result: ComponentStepReturn,
) -> "RuntimeComponentState":
    """Apply a field mapping or ``ComponentStepResult`` to runtime state."""

    if isinstance(result, ComponentStepResult):
        updated_state = with_runtime_fields(component, component_state, result.fields)
        return updated_state.with_runtime_payload(result.payload)

    return with_runtime_fields(component, component_state, result)


def require_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
    *names: str,
) -> None:
    """Validate that named runtime data fields use canonical grid layout."""

    from vercor.runtime.validation import validate_runtime_component_data_field

    for field_name in names:
        validate_runtime_component_data_field(component, component_state, field_name)


def prefill_runtime_fields(
    component: "Component",
    data: dict[str, RuntimeArray],
    field_spec: ComponentFieldSpec | None = None,
    *,
    outputs: FieldNames = (),
    default_fields: AuthorFieldValues = None,
    policy: PrecisionPolicy = None,
) -> None:
    """Prefill a mutable runtime data mapping with declared fields."""

    declared = field_spec or ComponentFieldSpec(
        outputs=outputs,
        default_fields=default_fields or {},
    )
    normalized_defaults = normalize_author_field_values(
        component_name=component.name,
        grid=component.grid,
        fields=declared.default_fields,
        policy=component.settings if policy is None else policy,
    )
    for field_name, field_value in (normalized_defaults or {}).items():
        data.setdefault(field_name, field_value)

    zeros = jax_zeros(
        component.grid.shape,
        component.settings if policy is None else policy,
    )
    for field_name in unique_field_names((*declared.outputs, *tuple(outputs))):
        data.setdefault(field_name, zeros)


def prefill_declared_runtime_fields(
    component: "Component",
    data: dict[str, RuntimeArray],
) -> None:
    """Prefill component data from the component's declared runtime fields."""

    prefill_runtime_fields(component, data, component.field_spec)


def validate_declared_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
) -> None:
    """Validate fields required by the component's declared field contract."""

    declared_fields = declared_runtime_field_names(component.field_spec)
    if declared_fields:
        require_runtime_fields(component, component_state, *declared_fields)
