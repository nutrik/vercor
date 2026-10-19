from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor.exceptions import ComponentError, CouplerError
from vercor.components._contracts import declared_runtime_field_names
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore

if TYPE_CHECKING:
    from vercor.components.contracts import Component


def validate_runtime_store_field(
    component: "Component",
    store: FieldStore,
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
    component_state: ComponentRuntimeState,
    field_name: str,
) -> None:
    """Validate that a named component field exists in runtime state."""

    if field_name not in component_state.fields:
        raise CouplerError(
            "Runtime missing required data field "
            f"'{field_name}' for component '{component.name}'"
        )


def validate_runtime_grid_data_field(
    component: "Component",
    component_state: ComponentRuntimeState,
    field_name: str,
) -> None:
    """Validate that a runtime field exists and matches the component grid."""

    validate_runtime_data_field_exists(component, component_state, field_name)
    validate_runtime_store_field(
        component,
        component_state.fields,
        field_name,
        "required data",
    )


def validate_component_runtime_contract_fields(
    component: "Component",
    component_state: ComponentRuntimeState,
    contract: ExchangeContract,
) -> None:
    """Validate generic runtime contract fields before component-specific checks."""

    for field_name in component.spec.outputs:
        validate_runtime_data_field_exists(component, component_state, field_name)
    for field_name in contract.receives:
        validate_runtime_store_field(
            component,
            component_state.received,
            field_name,
            "received",
        )
        validate_runtime_grid_data_field(
            component,
            component_state,
            field_name,
        )
    for field_name in contract.sends:
        validate_runtime_data_field_exists(component, component_state, field_name)
        validate_runtime_store_field(
            component,
            component_state.sent,
            field_name,
            "sent",
        )
    for field_name in component_state.received.field_names:
        validate_runtime_store_field(
            component,
            component_state.received,
            field_name,
            "received",
        )


def validate_exchange_fields_declared(
    component: "Component",
    contract: ExchangeContract,
) -> None:
    """Check that exchanged fields are declared by the receiving/sending component."""

    declared_fields = set(declared_runtime_field_names(component.spec))

    for field_name in contract.sends:
        if field_name not in declared_fields:
            raise ComponentError(
                f"Exchange send field '{field_name}' for component "
                f"'{component.name}' is not declared. Add it to ComponentSpec "
                "inputs or outputs before coupling."
            )

    for field_name in contract.receives:
        if field_name not in declared_fields:
            raise ComponentError(
                f"Exchange receive field '{field_name}' for component "
                f"'{component.name}' is not declared. Add it to ComponentSpec "
                "inputs or outputs before coupling."
            )
