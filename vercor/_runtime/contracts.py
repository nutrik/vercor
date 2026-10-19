from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.fields import _flatten_field_items


@dataclass(frozen=True)
class ExchangeContract:
    """Coupler-owned runtime import/export metadata for one component."""

    receives: tuple[str, ...] = ()
    sends: tuple[str, ...] = ()
    binary_sends: tuple[str, ...] = ()


def append_unique_runtime_fields(target: list[str], exchange_items: list[str]) -> None:
    """Append exchange field names while preserving first-seen order."""

    target.extend([item for item in exchange_items if item not in target])


def _extend_contract_fields(
    fields: tuple[str, ...],
    new_fields: list[str],
) -> tuple[str, ...]:
    """Return ``fields`` extended by new unique field names."""

    updated = list(fields)
    append_unique_runtime_fields(updated, new_fields)
    return tuple(updated)


def validate_exchange_fan_in(exchanges: Sequence[Exchange]) -> None:
    """Reject ambiguous exchange producers for one target runtime field."""

    producers: dict[tuple[str, str], list[str]] = {}
    for exchange in exchanges:
        flattened_fields = set(_flatten_field_items(exchange.fields))
        for field_name in flattened_fields:
            producers.setdefault((exchange.target, field_name), []).append(
                exchange.route_id
            )

    conflicts = [
        (target, field_name, tuple(sorted(labels)))
        for (target, field_name), labels in producers.items()
        if len(labels) > 1
    ]
    if not conflicts:
        return

    target, field_name, labels = sorted(conflicts)[0]
    formatted_labels = ", ".join(repr(label) for label in labels)
    raise CouplerError(
        f"Exchange fan-in conflict for target '{target}' field '{field_name}': "
        f"conflicting exchanges {formatted_labels}. Use distinct field names or "
        "an aggregator component."
    )


def build_exchange_contracts(
    component_names: Sequence[str],
    exchanges: Sequence[Exchange],
) -> dict[str, ExchangeContract]:
    """Build coupler-owned import/export contracts from exchange declarations."""

    known_components = set(component_names)
    contracts = {name: ExchangeContract() for name in component_names}
    for exchange in exchanges:
        if exchange.source not in known_components:
            raise CouplerError(
                f"Source component '{exchange.source}' not registered in coupler"
            )
        if exchange.target not in known_components:
            raise CouplerError(
                f"Destination component '{exchange.target}' not registered in coupler"
            )
    validate_exchange_fan_in(exchanges)

    for exchange in exchanges:
        flattened_fields = _flatten_field_items(exchange.fields)
        source_contract = contracts[exchange.source]
        contracts[exchange.source] = replace(
            source_contract,
            sends=_extend_contract_fields(source_contract.sends, flattened_fields),
            binary_sends=(
                _extend_contract_fields(source_contract.binary_sends, flattened_fields)
                if exchange.source_coverage == "binary"
                else source_contract.binary_sends
            ),
        )
        destination_contract = contracts[exchange.target]
        contracts[exchange.target] = replace(
            destination_contract,
            receives=_extend_contract_fields(
                destination_contract.receives, flattened_fields
            ),
        )
    return contracts
