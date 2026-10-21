from typing import Optional
from vercor.grids import RectilinearGrid
from vercor.grid_geometry import (
    _coordinate_allclose,
    centers_to_edges,
    grids_identical,
)
from vercor._interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor._regridders.base import _BaseRegridder
from vercor.types import RuntimeArray


class ConservativeRectilinearRegridder(_BaseRegridder):
    """Remap scalar cell means using declared or independently inferred bounds.

    Equal sampling centers alone do not establish equal cell areas. The no-op
    path also requires equal effective bounds and no explicit exclusion mask.
    """

    def __init__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
        source_mask: Optional[RuntimeArray] = None,
        normalize: str = "conservation",  # 'conservation' | 'fracarea'
        radius: float = 6371.0,
    ) -> None:

        if normalize not in {"conservation", "fracarea"}:
            raise ValueError(
                "normalize must be either 'conservation' or 'fracarea', "
                f"got {normalize!r}"
            )

        src_lon_edges = (
            source_grid.longitude_edges
            if source_grid.longitude_edges is not None
            else centers_to_edges(source_grid.longitude, "lon")
        )
        src_lat_edges = (
            source_grid.latitude_edges
            if source_grid.latitude_edges is not None
            else centers_to_edges(source_grid.latitude, "lat")
        )
        dst_lon_edges = (
            target_grid.longitude_edges
            if target_grid.longitude_edges is not None
            else centers_to_edges(target_grid.longitude, "lon")
        )
        dst_lat_edges = (
            target_grid.latitude_edges
            if target_grid.latitude_edges is not None
            else centers_to_edges(target_grid.latitude, "lat")
        )
        has_identical_grids = (
            source_mask is None
            and grids_identical(source_grid, target_grid)
            and src_lon_edges.shape == dst_lon_edges.shape
            and src_lat_edges.shape == dst_lat_edges.shape
            and _coordinate_allclose(src_lon_edges, dst_lon_edges)
            and _coordinate_allclose(src_lat_edges, dst_lat_edges)
        )
        interpolator = None

        if not has_identical_grids:
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
