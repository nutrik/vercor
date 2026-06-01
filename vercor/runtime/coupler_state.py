from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from vercor.exceptions import CouplerError
from vercor.exchange import Exchange
from vercor.runtime.component_state import create_runtime_component_state
from vercor.runtime.contracts import (
    RuntimeComponentContract,
    build_runtime_contracts,
    exchange_key_name,
)
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.runtime.validation import (
    validate_component_runtime_contract_fields,
)
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component


def runtime_state_from_components(
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray],
    *,
    contracts: Mapping[str, RuntimeComponentContract] | None = None,
    prefill_missing: bool = False,
) -> RuntimeCouplerState:
    """Create immutable runtime state from component setup objects."""

    runtime_contracts = (
        build_runtime_contracts(
            tuple(components),
            exchanges,
            validate_endpoints=False,
        )
        if contracts is None
        else contracts
    )
    runtime_components = tuple(
        create_runtime_component_state(
            component,
            prefill_missing=prefill_missing,
            contract=runtime_contracts[name],
        )
        for name, component in components.items()
    )
    runtime_fractional_masks = {
        exchange_key_name(*key): value for key, value in fractional_masks.items()
    }
    runtime_binary_masks = {
        exchange_key_name(*key): value for key, value in binary_masks.items()
    }
    return RuntimeCouplerState(
        component_names=tuple(components.keys()),
        components=runtime_components,
        fractional_masks=RuntimeFieldStore.from_mapping(runtime_fractional_masks),
        binary_masks=RuntimeFieldStore.from_mapping(runtime_binary_masks),
    )


def refresh_runtime_contracts(
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    *,
    validate_endpoints: bool = False,
) -> dict[str, RuntimeComponentContract]:
    """Return runtime contracts for the current component and exchange topology."""

    return build_runtime_contracts(
        tuple(components),
        exchanges,
        validate_endpoints=validate_endpoints,
    )


def validate_runtime_state(
    runtime_state: RuntimeCouplerState,
    *,
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    run_sequence: Sequence[str] | None,
) -> None:
    """Validate that runtime state matches the configured coupler topology."""

    if run_sequence is None:
        raise CouplerError("Runtime requires a configured component run sequence")

    run_order = tuple(run_sequence)
    if not run_order:
        raise CouplerError("Runtime requires a non-empty component run sequence")

    runtime_component_names = set(runtime_state.component_names)
    for cname in run_order:
        if cname not in components:
            raise CouplerError(
                f"Run-sequence component '{cname}' is not registered in coupler"
            )
        if cname not in runtime_component_names:
            raise CouplerError(
                f"Run-sequence component '{cname}' is missing from runtime state"
            )

        component = components[cname]
        component_state = runtime_state.get_component_state(cname)
        contract = contracts[cname]
        validate_component_runtime_contract_fields(
            component,
            component_state,
            contract,
        )
        component.validate_runtime_state(
            component_state,
            contract,
        )

    for exchange in exchanges:
        key = (exchange.source, exchange.destination, exchange.interpolation_type)
        if exchange.source not in runtime_component_names:
            raise CouplerError(
                f"Exchange source component '{exchange.source}' is missing from runtime state"
            )
        if exchange.destination not in runtime_component_names:
            raise CouplerError(
                f"Exchange destination component '{exchange.destination}' is missing from runtime state"
            )
        if key not in regridders:
            raise CouplerError(
                "Runtime requires an initialized regridder for exchange "
                f"{exchange.name}"
            )

        mask_name = exchange_key_name(*key)
        if mask_name not in runtime_state.fractional_masks.field_names:
            raise CouplerError(
                "Runtime requires an initialized fractional mask for exchange "
                f"{exchange.name}"
            )
        destination_shape = components[exchange.destination].grid.shape
        mask_shape = jnp.asarray(runtime_state.fractional_masks.get(mask_name)).shape
        if mask_shape != destination_shape:
            raise CouplerError(
                "Runtime fractional mask for exchange "
                f"{exchange.name} has shape {mask_shape}, expected {destination_shape}"
            )
