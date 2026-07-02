from __future__ import annotations

from dataclasses import dataclass, field

from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.topology_state import RuntimeTopologyMaps


@dataclass(slots=True)
class CouplerRuntimeResources:
    """Mutable runtime-owned resources for one public coupler instance."""

    topology_maps: RuntimeTopologyMaps = field(
        default_factory=RuntimeTopologyMaps.empty
    )
    runtime_contracts: dict[str, RuntimeComponentContract] = field(default_factory=dict)
    interrupt_controller: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController
    )


__all__ = ["CouplerRuntimeResources"]
