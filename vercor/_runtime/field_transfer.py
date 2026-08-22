from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.time import RuntimeStepInfo
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contracts import Component, TransferPolicy


def receive_runtime_fields(
    component_state: ComponentRuntimeState,
    contract: ExchangeContract,
) -> ComponentRuntimeState:
    """Move received runtime fields into component fields."""

    return component_state.with_fields(
        component_state.fields.set_many(
            {
                field_name: component_state.received.get(field_name)
                for field_name in contract.receives
            }
        )
    )


def select_runtime_field(
    field: RuntimeArray,
    transfer: "TransferPolicy",
    step_info: RuntimeStepInfo | None,
) -> RuntimeArray:
    """Select one current, linearly interpolated, or daily runtime field."""

    if step_info is None:
        return field

    time_selection = transfer.time_selection
    if time_selection == "linear":
        array = jnp.asarray(field)
        left = jnp.take(array, step_info.monthly_index_left, axis=0)
        right = jnp.take(array, step_info.monthly_index_right, axis=0)
        return (
            step_info.monthly_weight_left * left
            + step_info.monthly_weight_right * right
        )
    if time_selection == "daily":
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)
    return field


def send_runtime_fields(
    component: "Component",
    component_state: ComponentRuntimeState,
    step_info: RuntimeStepInfo | None = None,
    *,
    contract: ExchangeContract,
) -> ComponentRuntimeState:
    """Move component fields into sent runtime fields."""

    selected_fields = {
        field_name: select_runtime_field(
            component_state.fields.get(field_name),
            component.spec.transfer,
            step_info,
        )
        for field_name in contract.sends
    }
    for field_name, value in selected_fields.items():
        require_active_finite(
            value,
            active_mask=component.grid.binary_mask,
            owner=f"Component '{component.name}' sent field '{field_name}'",
        )
    return component_state.with_sent(component_state.sent.set_many(selected_fields))
