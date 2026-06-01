from __future__ import annotations

from collections.abc import Mapping
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
        *,
        regridders: Mapping[tuple[str, str, str], RuntimeRegridder] | None = None,
        binary_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
        fractional_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
    ) -> "RuntimeTopologyMaps":
        """Return a grouped topology-map bundle copied from existing mappings."""

        source_regridders = (
            topology_maps.regridders if topology_maps is not None else regridders
        )
        source_binary_masks = (
            topology_maps.binary_masks if topology_maps is not None else binary_masks
        )
        source_fractional_masks = (
            topology_maps.fractional_masks
            if topology_maps is not None
            else fractional_masks
        )
        return cls(
            regridders=dict(source_regridders or {}),
            binary_masks=dict(source_binary_masks or {}),
            fractional_masks=dict(source_fractional_masks or {}),
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
