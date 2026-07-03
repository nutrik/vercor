from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vercor.exchange import Exchange
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass(frozen=True)
class RuntimeDispatchContext:
    """Static runtime plumbing shared by per-component dispatch helpers."""

    components: Mapping[str, Component]
    exchanges_by_destination: Mapping[str, tuple[Exchange, ...]]
    regridders: Mapping[tuple[str, str, str], Any]
    contracts: Mapping[str, RuntimeComponentContract]
    dt_seconds: float
    settings: VercorSettings

    def destination_exchanges(self, component_name: str) -> tuple[Exchange, ...]:
        """Return exchanges targeting ``component_name``."""

        return self.exchanges_by_destination.get(component_name, ())


def build_runtime_dispatch_context(
    components: Mapping[str, Component],
    exchanges: Sequence[Exchange],
    regridders: Mapping[tuple[str, str, str], Any],
    contracts: Mapping[str, RuntimeComponentContract],
    *,
    dt_seconds: float,
    settings: VercorSettings,
) -> RuntimeDispatchContext:
    """Return static runtime dispatch plumbing for a configured coupler."""

    exchanges_by_destination: dict[str, list[Exchange]] = {}
    for exchange in exchanges:
        exchanges_by_destination.setdefault(exchange.target, []).append(exchange)

    return RuntimeDispatchContext(
        components=components,
        exchanges_by_destination={
            name: tuple(destination_exchanges)
            for name, destination_exchanges in exchanges_by_destination.items()
        },
        regridders=regridders,
        contracts=contracts,
        dt_seconds=dt_seconds,
        settings=settings,
    )
