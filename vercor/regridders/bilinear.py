from typing import Any

from vercor.exceptions import RegridderError
from vercor.grid import RectilinearGrid
from vercor.grid_geometry import grids_identical
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.regridders.base import Regridder


class BilinearRectilinearRegridder(Regridder):
    def __init__(
        self,
        source_grid: RectilinearGrid,
        destination_grid: RectilinearGrid,
        periodic_longitude: bool = True,
        # keep nan_renorm = True otherwise NaN will propagate to another components
        # during regridding and will keep growing over domains
        nan_renorm: bool = True,
        extrapolation_mode: str | None = None,  # 'nearest' | 'idw'
        idw_k: int = 8,
        idw_eps: float = 1e-12,
        fill_value: float = float("nan"),
    ) -> None:

        has_identical_grids = grids_identical(source_grid, destination_grid)
        interpolator = None
        if not has_identical_grids:
            interpolator = BilinearRectilinearInterpolator(
                source_grid.longitude,
                source_grid.latitude,
                destination_grid.longitude,
                destination_grid.latitude,
                src_mask=source_grid.binary_mask,
                tgt_mask=destination_grid.binary_mask,
                periodic_longitude=periodic_longitude,
                nan_renorm=nan_renorm,
                extrapolation_mode=extrapolation_mode,
                idw_k=idw_k,
                idw_eps=idw_eps,
                fill_value=fill_value,
            )

        super().__init__(
            source_grid,
            destination_grid,
            interpolator=interpolator,
            has_identical_grids=has_identical_grids,
        )

    def __call__(self, *args: Any) -> Any:
        """Apply bilinear scalar or vector regridding."""

        if len(args) not in (1, 2):
            raise TypeError("Provide scalar_src or (u_src, v_src) as positional args")

        if self.has_identical_grids:
            return args if len(args) == 2 else args[0]

        interpolator = self.interpolator
        if interpolator is None:
            raise RegridderError("Regridder not properly set up")
        if len(args) == 1:
            return interpolator.apply_scalar(args[0])
        return interpolator.apply_vector(args[0], args[1])


def bilinear(
    source_grid: RectilinearGrid,
    destination_grid: RectilinearGrid,
    *,
    periodic_longitude: bool = True,
    nan_renorm: bool = True,
    extrapolation_mode: str | None = None,
    idw_k: int = 8,
    idw_eps: float = 1e-12,
    fill_value: float = float("nan"),
) -> BilinearRectilinearRegridder:
    return BilinearRectilinearRegridder(
        source_grid,
        destination_grid,
        periodic_longitude=periodic_longitude,
        nan_renorm=nan_renorm,
        extrapolation_mode=extrapolation_mode,
        idw_k=idw_k,
        idw_eps=idw_eps,
        fill_value=fill_value,
    )
