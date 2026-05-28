from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.topology import ExchangeTopologyState, RuntimeRegridder
from vercor.types import RuntimeArray

CompiledRuntime = Callable[[RuntimeCouplerState], RuntimeCouplerState]


@dataclass
class CouplerRuntimeResources:
    """Mutable runtime-owned resources for one public coupler instance."""

    regridders: dict[tuple[str, str, str], RuntimeRegridder] = field(
        default_factory=dict
    )
    binary_masks: dict[tuple[str, str, str], RuntimeArray] = field(default_factory=dict)
    fractional_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    contracts: dict[str, RuntimeComponentContract] = field(default_factory=dict)
    compiled_runtime_cache: dict[tuple[Any, ...], CompiledRuntime] = field(
        default_factory=dict
    )
    interrupts: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController
    )

    def replace_contracts(
        self,
        contracts: dict[str, RuntimeComponentContract],
    ) -> None:
        """Replace refreshed runtime contracts as one resource update."""

        self.contracts = contracts

    def replace_topology(
        self,
        topology: ExchangeTopologyState,
    ) -> None:
        """Replace exchange topology maps from an initialized topology state."""

        self.replace_topology_maps(
            regridders=topology.regridders,
            binary_masks=topology.binary_masks,
            fractional_masks=topology.fractional_masks,
        )

    def replace_topology_maps(
        self,
        *,
        regridders: dict[tuple[str, str, str], RuntimeRegridder],
        binary_masks: dict[tuple[str, str, str], RuntimeArray],
        fractional_masks: dict[tuple[str, str, str], RuntimeArray],
    ) -> None:
        """Replace exchange topology maps as one grouped resource update."""

        self.regridders = regridders
        self.binary_masks = binary_masks
        self.fractional_masks = fractional_masks

    def clear_compiled_runtime_cache(self) -> None:
        """Clear compiled runtime entries owned by this resource holder."""

        self.compiled_runtime_cache.clear()

    def compiled_runtime_cache_entry_count(self) -> int:
        """Return the number of compiled runtime entries in the cache."""

        return len(self.compiled_runtime_cache)

    def compiled_runtime_cache_values(self) -> tuple[CompiledRuntime, ...]:
        """Return compiled runtime values without exposing the cache mapping."""

        return tuple(self.compiled_runtime_cache.values())


__all__ = ["CompiledRuntime", "CouplerRuntimeResources"]
