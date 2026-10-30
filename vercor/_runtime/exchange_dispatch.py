from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite, safe_masked_divide
from vercor._runtime.exchange_masks import exchange_coverage, exchange_mask_operands
from vercor._runtime.field_transfer import send_runtime_fields
from vercor.exceptions import ExchangeError
from vercor.exchanges import Exchange
from vercor.fields import VectorField
from vercor.state import RunState
from vercor._runtime.stores import FieldStore
from vercor._runtime.time import RuntimeStepInfo

if TYPE_CHECKING:
    from vercor._runtime.dispatch_context import RuntimeDispatchContext


def _dispatch_quantity(
    state: RunState,
    exchange: Exchange,
    source_fields: FieldStore,
    received_updates: dict[str, Any],
    context: RuntimeDispatchContext,
) -> None:
    """Remap covered source values once, then apply the declared area meaning."""
    source_fraction, target_binary, target_fraction, overlap = exchange_mask_operands(
        state,
        exchange,
        context=context,
    )
    source_fraction = jnp.asarray(source_fraction)
    overlap = jnp.asarray(overlap)
    active_source = source_fraction > 0
    active_target = (jnp.asarray(target_binary) > 0) & (overlap > 0)
    regrid = context.regridders[exchange.route_id]

    def weighted(name: str) -> Any:
        if name not in source_fields:
            raise ExchangeError(f"Field {name!r} not present in source fields")
        value = source_fields.get(name)
        require_active_finite(
            value,
            active_mask=active_source,
            owner=f"Exchange '{exchange.route_id}' source field '{name}'",
        )
        return jnp.where(active_source, value, 0.0) * source_fraction

    def store(name: str, numerator: Any) -> None:
        require_active_finite(
            numerator,
            active_mask=active_target,
            owner=f"Exchange '{exchange.route_id}' regridded field '{name}'",
        )
        if exchange.quantity == "source_flux":
            value = jnp.where(active_target, numerator, 0.0) * jnp.asarray(
                target_binary
            )
        else:
            value = safe_masked_divide(
                numerator,
                overlap,
                where=active_target,
                inactive_value=0.0,
            )
            if exchange.quantity == "target_flux":
                value = value * jnp.asarray(target_fraction)
        received_updates[name] = value

    for field in exchange.fields:
        if isinstance(field, VectorField):
            u, v = regrid.regrid_vector(weighted(field.u), weighted(field.v))
            store(field.u, u)
            store(field.v, v)
        else:
            store(field, regrid.regrid(weighted(field)))


def dispatch_component_exchanges(
    state: RunState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    step_info: RuntimeStepInfo | None = None,
    *,
    context: RuntimeDispatchContext,
) -> RunState:
    """Select clock-dependent exports, dispatch quantities, and retain support.

    Record sources select at the receiving step's time even when the workflow
    omits their step or schedules them later. Refresh each source's sent store
    once across its routes without advancing author physics. Current exports
    retain the sequential workflow's explicit send ownership. A missing time
    leaves stores untouched for direct dispatch of already-selected exports.
    """
    refreshed_sources: set[str] = set()
    if step_info is not None:
        for exchange in exchanges:
            if exchange.source in refreshed_sources:
                continue
            component = context.components[exchange.source]
            if component.spec.transfer.time_selection == "current":
                continue
            component_state = send_runtime_fields(
                component,
                state._component_state(exchange.source),
                step_info,
                contract=context.contracts[exchange.source],
                dtype=context.dtype,
            )
            state = state._with_component_state(exchange.source, component_state)
            refreshed_sources.add(exchange.source)

    destination_component = state._component_state(destination_name)
    received_updates: dict[str, Any] = {}
    for exchange in exchanges:
        _dispatch_quantity(
            state,
            exchange,
            state._component_state(exchange.source).sent,
            received_updates,
            context,
        )
    destination_component = destination_component.with_received(
        destination_component.received.set_many(received_updates)
    )
    result = state._with_component_state(destination_name, destination_component)
    mask_updates = {
        exchange.route_id: exchange_coverage(state, exchange, context=context)
        for exchange in exchanges
    }
    return result._with_received_masks(
        result._received_fractional_masks.replace_many(mask_updates)
    )
