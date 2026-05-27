from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.components.setup_validation import validate_component_setup
from vercor.dtypes import jax_zeros
from vercor.field_layout import validate_component_data_layout
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def prefill_runtime_contract_fields(
    component: "Component",
    data: dict[str, RuntimeArray],
    incoming: dict[str, RuntimeArray],
    outgoing: dict[str, RuntimeArray],
    contract: RuntimeComponentContract,
) -> None:
    """Add generic import/export fields required for stable runtime execution."""

    zeros = jax_zeros(component.grid.shape, component.settings)
    for field_name in contract.imports:
        incoming.setdefault(field_name, zeros)
        data.setdefault(field_name, zeros)
    for field_name in contract.exports:
        outgoing.setdefault(field_name, data.get(field_name, zeros))
        data.setdefault(field_name, zeros)


def create_runtime_component_state(
    component: "Component",
    *,
    prefill_missing: bool = False,
    contract: RuntimeComponentContract,
) -> RuntimeComponentState:
    """Create immutable runtime state from a component's seed data."""

    validate_component_setup(component)
    data = dict(component.data)
    incoming: dict[str, RuntimeArray] = {}
    outgoing: dict[str, RuntimeArray] = {}
    if prefill_missing:
        component.prefill_runtime_state_fields(data, incoming, outgoing, contract)
        prefill_runtime_contract_fields(component, data, incoming, outgoing, contract)

    validate_component_data_layout(
        component_name=component.name,
        grid_shape=component.grid.shape,
        data=data,
    )

    return RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(data),
        incoming=RuntimeFieldStore.from_mapping(incoming),
        outgoing=RuntimeFieldStore.from_mapping(outgoing),
        runtime_payload=component.create_runtime_payload(),
    )
