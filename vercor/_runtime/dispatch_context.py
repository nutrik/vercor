from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.physics import PhysicalConstants
from vercor.types import RuntimeArray
from vercor._runtime.contracts import ExchangeContract

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


@dataclass(frozen=True)
class RuntimeDispatchContext:
    """Static runtime plumbing shared by per-component dispatch helpers."""

    components: Mapping[str, _ComponentBinding]
    exchanges: tuple[Exchange, ...]
    exchanges_by_destination: Mapping[str, tuple[Exchange, ...]]
    regridders: Mapping[str, Any]
    contracts: Mapping[str, ExchangeContract]
    dt_seconds: float
    constants: PhysicalConstants
    dtype: DTypePolicy
    topology_binary_masks: Mapping[str, RuntimeArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Copy-own the static topology restrictions used by explicit quantities."""
        object.__setattr__(
            self,
            "topology_binary_masks",
            MappingProxyType(dict(self.topology_binary_masks)),
        )

    def destination_exchanges(self, component_name: str) -> tuple[Exchange, ...]:
        """Return exchanges targeting ``component_name``."""

        return self.exchanges_by_destination.get(component_name, ())


def build_runtime_dispatch_context(
    components: Mapping[str, _ComponentBinding],
    exchanges: Sequence[Exchange],
    regridders: Mapping[str, Any],
    contracts: Mapping[str, ExchangeContract],
    *,
    dt_seconds: float,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
    topology_binary_masks: Mapping[str, RuntimeArray] | None = None,
) -> RuntimeDispatchContext:
    """Return static runtime dispatch plumbing for a configured coupler."""

    exchanges_by_destination: dict[str, list[Exchange]] = {}
    for exchange in exchanges:
        exchanges_by_destination.setdefault(exchange.target, []).append(exchange)

    return RuntimeDispatchContext(
        components=components,
        exchanges=tuple(exchanges),
        exchanges_by_destination=MappingProxyType(
            {
                name: tuple(destination_exchanges)
                for name, destination_exchanges in exchanges_by_destination.items()
            }
        ),
        regridders=regridders,
        contracts=contracts,
        dt_seconds=dt_seconds,
        constants=constants,
        dtype=dtype,
        topology_binary_masks=(
            {} if topology_binary_masks is None else topology_binary_masks
        ),
    )
