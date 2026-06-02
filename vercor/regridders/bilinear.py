from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.regridders.base import Regridder, _IdentityInterpolator


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

        super().__init__(source_grid, destination_grid)
        if self.has_identical_grids:
            self.interpolator = _IdentityInterpolator()
            return

        self.interpolator = BilinearRectilinearInterpolator(
            self.source_grid.longitude,
            self.source_grid.latitude,
            self.destination_grid.longitude,
            self.destination_grid.latitude,
            src_mask=self.source_grid.binary_mask,
            tgt_mask=self.destination_grid.binary_mask,
            periodic_longitude=periodic_longitude,
            nan_renorm=nan_renorm,
            extrapolation_mode=extrapolation_mode,
            idw_k=idw_k,
            idw_eps=idw_eps,
            fill_value=fill_value,
        )


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
