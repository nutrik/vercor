from typing import Any, Optional

from vercor.exceptions import RegridderError
from vercor.grid import RectilinearGrid
from vercor.grid_geometry import centers_to_edges, grids_identical
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.regridders.base import Regridder
from vercor.types import RuntimeArray


class ConservativeRectilinearRegridder(Regridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        destination_grid: RectilinearGrid,
        source_mask: Optional[RuntimeArray] = None,
        normalize: str = "conservation",  # 'conservation' | 'fracarea'
        radius: float = 6371.0,
    ) -> None:

        has_identical_grids = grids_identical(source_grid, destination_grid)
        interpolator = None

        if not has_identical_grids:
            if (
                source_grid.longitude_edges is not None
                and source_grid.latitude_edges is not None
            ):
                src_lon_edges = source_grid.longitude_edges
                src_lat_edges = source_grid.latitude_edges
            else:
                src_lon_edges = centers_to_edges(source_grid.longitude, "lon")
                src_lat_edges = centers_to_edges(source_grid.latitude, "lat")

            if (
                destination_grid.longitude_edges is not None
                and destination_grid.latitude_edges is not None
            ):
                dst_lon_edges = destination_grid.longitude_edges
                dst_lat_edges = destination_grid.latitude_edges
            else:
                dst_lon_edges = centers_to_edges(destination_grid.longitude, "lon")
                dst_lat_edges = centers_to_edges(destination_grid.latitude, "lat")

            interpolator = ConservativeRectilinearRemapper(
                src_lon_edges=src_lon_edges,
                src_lat_edges=src_lat_edges,
                dst_lon_edges=dst_lon_edges,
                dst_lat_edges=dst_lat_edges,
                src_mask=source_mask,
                normalize=normalize,
                radius=radius,
            )

        super().__init__(
            source_grid,
            destination_grid,
            interpolator=interpolator,
            has_identical_grids=has_identical_grids,
        )

    def __call__(self, *args: Any) -> Any:
        """Apply conservative scalar regridding."""

        if len(args) not in (1, 2):
            raise TypeError("Provide scalar_src or (u_src, v_src) as positional args")
        if len(args) == 2:
            raise TypeError(
                "Conservative regridding supports scalar fields only; use bilinear "
                "regridding for vector fields."
            )

        if self.has_identical_grids:
            return args[0]

        interpolator = self.interpolator
        if interpolator is None:
            raise RegridderError("Regridder not properly set up")
        return interpolator.apply_scalar(args[0])


def conservative(
    source_grid: RectilinearGrid,
    destination_grid: RectilinearGrid,
    *,
    source_mask: Optional[RuntimeArray] = None,
    normalize: str = "conservation",
    radius: float = 6371.0,
) -> ConservativeRectilinearRegridder:
    return ConservativeRectilinearRegridder(
        source_grid,
        destination_grid,
        source_mask=source_mask,
        normalize=normalize,
        radius=radius,
    )
