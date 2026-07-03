from __future__ import annotations

from typing import Any, Mapping, Sequence

from vercor.exceptions import ExchangerError
from vercor.exchange import Exchange
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.stores import RuntimeFieldStore


def _dispatch_vector_exchange_field(
    source_fields: RuntimeFieldStore,
    incoming_updates: dict[str, Any],
    field_name: tuple[str, str],
    regrid: Any,
) -> None:
    """Dispatch one vector exchange field into the incoming update mapping."""

    if not all(name in source_fields for name in field_name):
        raise ExchangerError(
            f"Not all fields in vector {field_name} are present in source fields"
        )
    u_vector, v_vector = regrid(
        source_fields.get(field_name[0]),
        source_fields.get(field_name[1]),
    )
    incoming_updates[field_name[0]] = u_vector
    incoming_updates[field_name[1]] = v_vector


def _dispatch_scalar_exchange_field(
    source_fields: RuntimeFieldStore,
    incoming_updates: dict[str, Any],
    field_name: str,
    regrid: Any,
    fractional_mask: Any,
) -> None:
    """Dispatch one scalar exchange field into the incoming update mapping."""

    if field_name not in source_fields:
        raise ExchangerError(f"Field {field_name} not present in source fields")
    incoming_updates[field_name] = (
        regrid(source_fields.get(field_name)) * fractional_mask
    )


def dispatch_component_exchanges(
    state: RuntimeCouplerState,
    destination_name: str,
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
) -> RuntimeCouplerState:
    """Dispatch destination-specific exchanges into one component."""

    destination_component = state.get_component_state(destination_name)
    destination_incoming = destination_component.incoming
    incoming_updates: dict[str, Any] = {}

    for exchange in exchanges:
        source_component = state.get_component_state(exchange.source)
        source_fields = source_component.outgoing
        key = (exchange.source, exchange.target, exchange.interpolation_type)
        regrid = regridders[key]
        fractional_mask = state.get_fractional_mask(*key)

        for field_name in exchange.fields:
            if isinstance(field_name, tuple):
                _dispatch_vector_exchange_field(
                    source_fields,
                    incoming_updates,
                    field_name,
                    regrid,
                )
            else:
                _dispatch_scalar_exchange_field(
                    source_fields,
                    incoming_updates,
                    field_name,
                    regrid,
                    fractional_mask,
                )

    destination_incoming = destination_incoming.set_many(incoming_updates)
    destination_component = destination_component.with_incoming(destination_incoming)
    return state.set_component_state(destination_name, destination_component)
