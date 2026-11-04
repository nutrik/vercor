from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite
from vercor.dtypes import DTypePolicy
from vercor.components._contracts import validate_time_selected_field_layout
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.component_masks import field_active_mask, resolve_component_masks
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.time import RuntimeStepInfo
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contracts import Component, ComponentMaskSpec, TransferPolicy


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
    *,
    field_name: str | None = None,
    masks: "ComponentMaskSpec | None" = None,
) -> RuntimeArray:
    """Select forcing records while keeping declared grid-shaped masks current."""

    validate_time_selected_field_layout(
        field,
        transfer,
        field_name=field_name,
        masks=masks,
    )
    if step_info is None or (
        masks is not None and field_name in (masks.binary_field, masks.fractional_field)
    ):
        return field

    time_selection = transfer.time_selection
    if time_selection == "linear":
        array = jnp.asarray(field)
        left = jnp.take(array, step_info.monthly_index_left, axis=0)
        right = jnp.take(array, step_info.monthly_index_right, axis=0)
        weight_left = jnp.asarray(step_info.monthly_weight_left, dtype=array.dtype)
        weight_right = jnp.asarray(step_info.monthly_weight_right, dtype=array.dtype)
        return weight_left * left + weight_right * right
    if time_selection == "daily":
        return jnp.take(jnp.asarray(field), step_info.daily_index, axis=0)
    return field


def send_runtime_fields(
    component: "Component",
    component_state: ComponentRuntimeState,
    step_info: RuntimeStepInfo | None = None,
    *,
    contract: ExchangeContract,
    dtype: DTypePolicy | None = None,
) -> ComponentRuntimeState:
    """Move component fields into sent runtime fields."""

    binary, fraction = resolve_component_masks(
        component,
        component_state.fields.to_mapping(),
        DTypePolicy.from_jax_config() if dtype is None else dtype,
    )
    selected_fields = {
        field_name: select_runtime_field(
            component_state.fields.get(field_name),
            component.spec.transfer,
            step_info,
            field_name=field_name,
            masks=component.spec.masks,
        )
        for field_name in contract.sends
    }
    for field_name, value in selected_fields.items():
        require_active_finite(
            value,
            active_mask=field_active_mask(
                component,
                field_name,
                default_mask=component.grid.binary_mask,
                fractional_mask=fraction,
                produced=True,
                binary_mask=binary,
                binary_output_fields=contract.binary_sends,
            ),
            owner=f"Component '{component.name}' sent field '{field_name}'",
        )
    return component_state.with_sent(component_state.sent.set_many(selected_fields))
