from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.run_context import CompiledRuntime
from vercor.runtime.topology import ExchangeTopologyState, RuntimeTopologyMaps


@dataclass(slots=True)
class CouplerRuntimeResources:
    """Mutable runtime-owned resources for one public coupler instance."""

    _topology_maps: RuntimeTopologyMaps = field(
        default_factory=RuntimeTopologyMaps.empty
    )
    _runtime_contracts: dict[str, RuntimeComponentContract] = field(
        default_factory=dict
    )
    _compiled_runtime_cache: dict[tuple[Any, ...], CompiledRuntime] = field(
        default_factory=dict
    )
    _interrupt_controller: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController
    )

    @property
    def topology_maps(self) -> RuntimeTopologyMaps:
        """Return grouped exchange topology maps owned by this holder."""

        return self._topology_maps

    @property
    def runtime_contracts(self) -> dict[str, RuntimeComponentContract]:
        """Return refreshed runtime contracts owned by this holder."""

        return self._runtime_contracts

    @property
    def interrupt_controller(self) -> RuntimeInterruptController:
        """Return the runtime interrupt controller owned by this holder."""

        return self._interrupt_controller

    def replace_contracts(
        self,
        contracts: dict[str, RuntimeComponentContract],
    ) -> None:
        """Replace refreshed runtime contracts as one resource update."""

        self._runtime_contracts = contracts

    def replace_topology(
        self,
        topology: ExchangeTopologyState,
    ) -> None:
        """Replace exchange topology maps from an initialized topology state."""

        self.replace_topology_maps(topology.topology_maps)

    def replace_topology_maps(
        self,
        topology_maps: RuntimeTopologyMaps,
    ) -> None:
        """Replace exchange topology maps as one grouped resource update."""

        self._topology_maps = topology_maps

    def runtime_cache_mapping(
        self,
    ) -> dict[tuple[Any, ...], CompiledRuntime]:
        """Return the mutable compiled runtime cache for cache-owner helpers."""

        return self._compiled_runtime_cache

    def clear_compiled_runtime_cache(self) -> None:
        """Clear compiled runtime entries owned by this resource holder."""

        self._compiled_runtime_cache.clear()

    def compiled_runtime_cache_entry_count(self) -> int:
        """Return the number of compiled runtime entries in the cache."""

        return len(self._compiled_runtime_cache)

    def compiled_runtime_cache_values(self) -> tuple[CompiledRuntime, ...]:
        """Return compiled runtime values without exposing the cache mapping."""

        return tuple(self._compiled_runtime_cache.values())


__all__ = ["CompiledRuntime", "CouplerRuntimeResources"]
