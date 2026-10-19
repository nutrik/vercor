"""Public regridding facade for VerCOR grid-to-grid transfers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vercor.grids import RectilinearGrid as _RectilinearGrid
from vercor._regridders.bilinear import (
    BilinearRectilinearRegridder as _BilinearRectilinearRegridder,
)
from vercor._regridders.conservative import (
    ConservativeRectilinearRegridder as _ConservativeRectilinearRegridder,
)
from vercor.types import RuntimeArray as _RuntimeArray


@runtime_checkable
class Regridder(Protocol):
    """Public capability for scalar grid-to-grid transfer objects."""

    source_grid: _RectilinearGrid

    @property
    def target_grid(self) -> _RectilinearGrid:
        """Return the destination grid using public target terminology."""

    @property
    def has_identical_grids(self) -> bool:
        """Return whether source and target grids are identical."""

    def regrid(self, field: _RuntimeArray) -> _RuntimeArray:
        """Transfer one scalar field to the target grid."""


@runtime_checkable
class VectorRegridder(Regridder, Protocol):
    """Public capability for scalar and paired-vector grid transfer."""

    def regrid_vector(
        self,
        u: _RuntimeArray,
        v: _RuntimeArray,
    ) -> tuple[_RuntimeArray, _RuntimeArray]:
        """Transfer one vector field pair to the target grid."""


@runtime_checkable
class RegridderFactory(Protocol):
    """Build a scalar-capable regridder for one source/target grid pair."""

    def __call__(
        self,
        source_grid: _RectilinearGrid,
        target_grid: _RectilinearGrid,
    ) -> Regridder:
        """Return a regridder configured for the supplied grids."""


def bilinear(
    source_grid: _RectilinearGrid,
    target_grid: _RectilinearGrid,
    *,
    periodic_longitude: bool = True,
    nan_renorm: bool = True,
    extrapolation_mode: str | None = None,
    idw_k: int = 8,
    idw_eps: float = 1e-12,
    fill_value: float = float("nan"),
) -> Regridder:
    """Build a public bilinear regridder for one source/target grid pair."""

    return _BilinearRectilinearRegridder(
        source_grid,
        target_grid,
        periodic_longitude=periodic_longitude,
        nan_renorm=nan_renorm,
        extrapolation_mode=extrapolation_mode,
        idw_k=idw_k,
        idw_eps=idw_eps,
        fill_value=fill_value,
    )


def conservative(
    source_grid: _RectilinearGrid,
    target_grid: _RectilinearGrid,
    *,
    source_mask: _RuntimeArray | None = None,
    normalize: str = "conservation",
    radius_km: float = 6371.0,
) -> Regridder:
    """Build a public conservative scalar regridder for one grid pair."""

    return _ConservativeRectilinearRegridder(
        source_grid,
        target_grid,
        source_mask=source_mask,
        normalize=normalize,
        radius=radius_km,
    )


__all__ = [
    "Regridder",
    "RegridderFactory",
    "VectorRegridder",
    "bilinear",
    "conservative",
]
