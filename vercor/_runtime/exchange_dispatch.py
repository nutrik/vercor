from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Sequence

import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite, safe_masked_divide
from vercor._runtime.exchange_masks import exchange_mask_operands
from vercor.exceptions import ExchangeError
from vercor.exchanges import Exchange
from vercor.fields import VectorField
from vercor.state import RunState
from vercor._runtime.stores import FieldStore

if TYPE_CHECKING:
    from vercor._runtime.dispatch_context import RuntimeDispatchContext


def _dispatch_explicit_quantity(
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


def _dispatch_vector_exchange_field(
    source_fields: FieldStore,
    received_updates: dict[str, Any],
    field_name: VectorField,
    regrid: Any,
    route_id: str,
    fractional_mask: Any,
) -> None:
    """Dispatch one vector exchange field into the received update mapping."""

    if not all(name in source_fields for name in (field_name.u, field_name.v)):
        raise ExchangeError(
            f"Not all fields in vector {field_name} are present in source fields"
        )
    u_vector, v_vector = regrid.regrid_vector(
        source_fields.get(field_name.u),
        source_fields.get(field_name.v),
    )
    for vector_name, vector_value in (
        (field_name.u, u_vector),
        (field_name.v, v_vector),
    ):
        require_active_finite(
            vector_value,
            active_mask=fractional_mask,
            owner=(f"exchange '{route_id}' regridded vector field '{vector_name}'"),
        )
    received_updates[field_name.u] = u_vector
    received_updates[field_name.v] = v_vector


def _dispatch_scalar_exchange_field(
    source_fields: FieldStore,
    received_updates: dict[str, Any],
    field_name: str,
    regrid: Any,
    route_id: str,
    fractional_mask: Any,
) -> None:
    """Dispatch one scalar exchange field into the received update mapping."""

    if field_name not in source_fields:
        raise ExchangeError(f"Field {field_name} not present in source fields")
    regridded = regrid.regrid(source_fields.get(field_name))
    require_active_finite(
        regridded,
        active_mask=fractional_mask,
        owner=f"exchange '{route_id}' regridded field '{field_name}'",
    )
    active = jnp.asarray(fractional_mask) > 0
    received_updates[field_name] = jnp.where(active, regridded, 0.0) * jnp.asarray(
        fractional_mask
    )


def dispatch_component_exchanges(
    state: RunState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    regridders: Mapping[str, Any],
    *,
    context: RuntimeDispatchContext | None = None,
) -> RunState:
    """Dispatch destination-specific exchanges into one component."""

    destination_component = state._component_state(destination_name)
    destination_received = destination_component.received
    received_updates: dict[str, Any] = {}

    for exchange in exchanges:
        source_component = state._component_state(exchange.source)
        source_fields = source_component.sent
        if exchange.quantity != "legacy":
            if context is None:
                raise ExchangeError(
                    "Explicit exchange quantities require a runtime context."
                )
            _dispatch_explicit_quantity(
                state, exchange, source_fields, received_updates, context
            )
            continue
        regrid = regridders[exchange.route_id]
        fractional_mask = state._fractional_mask(exchange.route_id)

        for field_name in exchange.fields:
            if isinstance(field_name, VectorField):
                _dispatch_vector_exchange_field(
                    source_fields,
                    received_updates,
                    field_name,
                    regrid,
                    exchange.route_id,
                    fractional_mask,
                )
            else:
                _dispatch_scalar_exchange_field(
                    source_fields,
                    received_updates,
                    field_name,
                    regrid,
                    exchange.route_id,
                    fractional_mask,
                )

    destination_received = destination_received.set_many(received_updates)
    destination_component = destination_component.with_received(destination_received)
    return state._with_component_state(destination_name, destination_component)
