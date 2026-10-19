from typing import Optional
from vercor.grids import RectilinearGrid
from vercor.grid_geometry import centers_to_edges, grids_identical
from vercor._interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor._regridders.base import _BaseRegridder
from vercor.types import RuntimeArray


class ConservativeRectilinearRegridder(_BaseRegridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
        source_mask: Optional[RuntimeArray] = None,
        normalize: str = "conservation",  # 'conservation' | 'fracarea'
        radius: float = 6371.0,
    ) -> None:

        has_identical_grids = grids_identical(source_grid, target_grid)
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
                target_grid.longitude_edges is not None
                and target_grid.latitude_edges is not None
            ):
                dst_lon_edges = target_grid.longitude_edges
                dst_lat_edges = target_grid.latitude_edges
            else:
                dst_lon_edges = centers_to_edges(target_grid.longitude, "lon")
                dst_lat_edges = centers_to_edges(target_grid.latitude, "lat")

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
            target_grid,
            interpolator=interpolator,
            has_identical_grids=has_identical_grids,
        )
