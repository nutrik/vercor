from typing import Any

from vercor.grid import RectilinearGrid
from vercor.grid_geometry import grids_identical


class Regridder:
    """Shared grid, interpolator, and display state for concrete regridders.

    Concrete subclasses own their call contracts because scalar/vector support
    differs by regridding method.
    """

    def __init__(
        self,
        source_grid: RectilinearGrid,
        destination_grid: RectilinearGrid,
        *,
        interpolator: Any | None = None,
        has_identical_grids: bool | None = None,
    ) -> None:
        self.source_grid = source_grid
        self.destination_grid = destination_grid
        self._interpolator = interpolator
        self._has_identical_grids = (
            grids_identical(self.source_grid, self.destination_grid)
            if has_identical_grids is None
            else has_identical_grids
        )

    @property
    def has_identical_grids(self) -> bool:
        """Return whether source and destination grids are identical."""

        return self._has_identical_grids

    @property
    def interpolator(self) -> Any | None:
        """Return the concrete interpolator, or ``None`` for identity grids."""

        return self._interpolator

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:"
            f"\n ├──Source grid:"
            f"\n │    ├──Grid type: {self.source_grid.__class__.__name__} ({self.source_grid.name})"
            f"\n │    └──Grid shape: {self.source_grid.shape}"
            f"\n └──Destination grid:"
            f"\n      ├──Grid type: {self.destination_grid.__class__.__name__} ({self.destination_grid.name})"
            f"\n      └──Grid shape: {self.destination_grid.shape}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source_grid={repr(self.source_grid)},"
            f" destination_grid={repr(self.destination_grid)})"
        )
