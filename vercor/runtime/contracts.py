from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from vercor.exceptions import CouplerError
from vercor.exchange import Exchange


@dataclass(frozen=True)
class RuntimeComponentContract:
    """Coupler-owned runtime import/export metadata for one component."""

    imports: tuple[str, ...] = ()
    exports: tuple[str, ...] = ()

    @property
    def all_fields(self) -> tuple[str, ...]:
        """Return all import/export fields while preserving contract order."""

        return (*self.imports, *self.exports)


def flatten_exchange_fields(
    field_names: Sequence[str | tuple[str, str]],
) -> list[str]:
    """Return scalar field names from scalar and vector exchange declarations."""

    flattened: list[str] = []
    for item in field_names:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


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


def build_runtime_contracts(
    component_names: Sequence[str],
    exchanges: Sequence[Exchange],
    *,
    validate_endpoints: bool,
) -> dict[str, RuntimeComponentContract]:
    """Build coupler-owned import/export contracts from exchange declarations."""

    known_components = set(component_names)
    contracts = {name: RuntimeComponentContract() for name in component_names}
    for exchange in exchanges:
        if exchange.source not in known_components:
            if validate_endpoints:
                raise CouplerError(
                    f"Source component '{exchange.source}' not registered in coupler"
                )
            continue
        if exchange.target not in known_components:
            if validate_endpoints:
                raise CouplerError(
                    f"Destination component '{exchange.target}' not registered in coupler"
                )
            continue

        flattened_fields = flatten_exchange_fields(exchange.fields)
        source_contract = contracts[exchange.source]
        destination_contract = contracts[exchange.target]
        contracts[exchange.source] = RuntimeComponentContract(
            imports=source_contract.imports,
            exports=_extend_contract_fields(
                source_contract.exports,
                flattened_fields,
            ),
        )
        contracts[exchange.target] = RuntimeComponentContract(
            imports=_extend_contract_fields(
                destination_contract.imports,
                flattened_fields,
            ),
            exports=destination_contract.exports,
        )
    return contracts


def exchange_key_name(source: str, destination: str, interpolation_type: str) -> str:
    """Return a stable field-store key for exchange metadata arrays."""

    return f"{source}|{destination}|{interpolation_type}"
