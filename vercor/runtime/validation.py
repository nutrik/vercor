from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor.exceptions import ComponentError, CouplerError
from vercor.field_names import VALID_EXCHANGE_FIELD_NAMES
from vercor.field_layout import (
    canonical_data_layout_description,
    is_canonical_grid_field_shape,
)
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore

if TYPE_CHECKING:
    from vercor.components.base import Component


def validate_runtime_store_field(
    component: "Component",
    store: RuntimeFieldStore,
    field_name: str,
    store_description: str,
) -> None:
    """Validate that a named runtime store field exists and matches the component grid."""

    expected_shape = component.grid.shape
    if field_name not in store:
        raise CouplerError(
            "Runtime missing "
            f"{store_description} field '{field_name}' for component '{component.name}'"
        )

    field_shape = jnp.asarray(store.get(field_name)).shape
    if field_shape != expected_shape:
        raise CouplerError(
            "Runtime "
            f"{store_description} field '{field_name}' for component '{component.name}' "
            f"has shape {field_shape}, expected {expected_shape}"
        )


def validate_runtime_data_field_exists(
    component: "Component",
    component_state: RuntimeComponentState,
    field_name: str,
) -> None:
    """Validate that a named component data field exists in runtime state."""

    if field_name not in component_state.data:
        raise CouplerError(
            "Runtime missing required data field "
            f"'{field_name}' for component '{component.name}'"
        )


def validate_runtime_grid_data_field(
    component: "Component",
    component_state: RuntimeComponentState,
    field_name: str,
) -> None:
    """Validate that a runtime data field exists and matches the component grid."""

    validate_runtime_data_field_exists(component, component_state, field_name)
    validate_runtime_store_field(
        component,
        component_state.data,
        field_name,
        "required data",
    )


def validate_runtime_component_data_field(
    component: "Component",
    component_state: RuntimeComponentState,
    field_name: str,
) -> None:
    """Validate that a runtime data field exists and has canonical grid layout."""

    validate_runtime_data_field_exists(component, component_state, field_name)
    field_shape = tuple(
        int(size) for size in jnp.asarray(component_state.data.get(field_name)).shape
    )
    if not is_canonical_grid_field_shape(field_shape, component.grid.shape):
        raise CouplerError(
            "Runtime required data field "
            f"'{field_name}' for component '{component.name}' has shape "
            f"{field_shape}; expected canonical grid-field layout "
            f"{canonical_data_layout_description()} with trailing grid shape "
            f"{component.grid.shape}"
        )


def validate_component_runtime_contract_fields(
    component: "Component",
    component_state: RuntimeComponentState,
    contract: RuntimeComponentContract,
) -> None:
    """Validate generic runtime contract fields before component-specific checks."""

    for field_name in contract.imports:
        validate_runtime_store_field(
            component,
            component_state.incoming,
            field_name,
            "imported incoming",
        )
        validate_runtime_grid_data_field(
            component,
            component_state,
            field_name,
        )
    for field_name in contract.exports:
        validate_runtime_data_field_exists(component, component_state, field_name)
        validate_runtime_store_field(
            component,
            component_state.outgoing,
            field_name,
            "exported source",
        )
    for field_name in component_state.incoming.field_names:
        validate_runtime_store_field(
            component,
            component_state.incoming,
            field_name,
            "incoming",
        )


def check_not_empty_import_export_lists(
    component: "Component",
    contract: RuntimeComponentContract,
) -> None:
    """Check that a component's runtime contract has valid field ownership."""

    if not contract.imports:
        raise ComponentError(
            f"Component '{component.name}' has no fields to import defined."
        )
    if not contract.exports:
        raise ComponentError(
            f"Component '{component.name}' has no fields to export defined."
        )

    all_fields = set(contract.all_fields)
    if len(all_fields) < len(contract.all_fields):
        raise ComponentError(
            f"Component '{component.name}' has overlapping fields in import/export lists."
        )


def check_valid_exchange_field_names(
    component: "Component",
    contract: RuntimeComponentContract,
) -> None:
    """Check that a component's runtime contract uses supported exchange fields."""

    for field_name in set(contract.all_fields):
        if field_name not in VALID_EXCHANGE_FIELD_NAMES:
            raise ComponentError(
                f"Field name '{field_name}' in component '{component.name}' is not a recognized exchange variable.\n"
                f"Replace field name '{field_name}' with one of the supported names: {VALID_EXCHANGE_FIELD_NAMES}"
            )
