from __future__ import annotations

from dataclasses import dataclass

from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
)
from vercor.types import RuntimeArray

RuntimeRegridder = BilinearRectilinearRegridder | ConservativeRectilinearRegridder


@dataclass(frozen=True)
class RuntimeTopologyMaps:
    """Grouped exchange topology maps used by runtime setup and dispatch."""

    regridders: dict[tuple[str, str, str], RuntimeRegridder]
    binary_masks: dict[tuple[str, str, str], RuntimeArray]
    fractional_masks: dict[tuple[str, str, str], RuntimeArray]

    @classmethod
    def empty(cls) -> "RuntimeTopologyMaps":
        """Return an empty grouped topology-map bundle."""

        return cls(
            regridders={},
            binary_masks={},
            fractional_masks={},
        )

    @classmethod
    def from_mappings(
        cls,
        topology_maps: "RuntimeTopologyMaps | None" = None,
    ) -> "RuntimeTopologyMaps":
        """Return an empty bundle or a copy of an existing topology-map bundle."""

        if topology_maps is None:
            return cls.empty()
        return cls(
            regridders=dict(topology_maps.regridders),
            binary_masks=dict(topology_maps.binary_masks),
            fractional_masks=dict(topology_maps.fractional_masks),
        )


@dataclass(frozen=True)
class SurfaceExchangeMasks:
    """Derived ATM-grid masks for the coupled ocean/land surface policy."""

    ocn_fmask_on_atm_grid: RuntimeArray
    lnd_fmask_on_atm_grid: RuntimeArray
    lnd_bmask_on_atm_grid: RuntimeArray


@dataclass(frozen=True)
class ExchangeTopologyState:
    """Exchange topology maps and derived surface-exchange masks."""

    topology_maps: RuntimeTopologyMaps
    surface_masks: SurfaceExchangeMasks


__all__ = [
    "ExchangeTopologyState",
    "RuntimeRegridder",
    "RuntimeTopologyMaps",
    "SurfaceExchangeMasks",
]
