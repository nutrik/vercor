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
