from typing import Any, cast

from vercor.exceptions import RegridderError
from vercor.grids import RectilinearGrid
from vercor.grid_geometry import grids_identical
from vercor._interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor._regridders.base import _BaseRegridder


class BilinearRectilinearRegridder(_BaseRegridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        target_grid: RectilinearGrid,
        periodic_longitude: bool = True,
        # keep nan_renorm = True otherwise NaN will propagate to another components
        # during regridding and will keep growing over domains
        nan_renorm: bool = True,
        extrapolation_mode: str | None = None,  # 'nearest' | 'idw'
        idw_k: int = 8,
        idw_eps: float = 1e-12,
        fill_value: float = float("nan"),
    ) -> None:

        has_identical_grids = grids_identical(source_grid, target_grid)
        interpolator = None
        if not has_identical_grids:
            interpolator = BilinearRectilinearInterpolator(
                source_grid.longitude,
                source_grid.latitude,
                target_grid.longitude,
                target_grid.latitude,
                src_mask=source_grid.binary_mask,
                tgt_mask=target_grid.binary_mask,
                periodic_longitude=periodic_longitude,
                nan_renorm=nan_renorm,
                extrapolation_mode=extrapolation_mode,
                idw_k=idw_k,
                idw_eps=idw_eps,
                fill_value=fill_value,
            )

        super().__init__(
            source_grid,
            target_grid,
            interpolator=interpolator,
            has_identical_grids=has_identical_grids,
        )

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        """Apply bilinear vector regridding."""

        if self.has_identical_grids:
            return u, v

        interpolator = self.interpolator
        if interpolator is None:
            raise RegridderError("Regridder not properly set up")
        return cast(tuple[Any, Any], interpolator.apply_vector(u, v))
