from typing import Any, Protocol, Tuple, cast

from vercor.exceptions import RegridderError
from vercor.grid import RectilinearGrid
from vercor.grid_geometry import grids_identical


class SupportsScalarVectorInterpolation(Protocol):
    """Protocol for rectilinear interpolators used by regridder wrappers."""

    def apply_scalar(self, src: Any) -> Any:
        """Interpolate one scalar field."""
        ...

    def apply_vector(self, u_src: Any, v_src: Any) -> tuple[Any, Any]:
        """Interpolate one vector field."""
        ...


class Regridder:
    def __init__(
        self, source_grid: RectilinearGrid, destination_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.destination_grid = destination_grid
        self.interpolator: SupportsScalarVectorInterpolation | None = None
        self._has_identical_grids = grids_identical(
            self.source_grid, self.destination_grid
        )

    @property
    def has_identical_grids(self) -> bool:
        """Return whether source and destination grids are identical."""

        return self._has_identical_grids

    def _ensure_ready(self, args: Tuple[Any, ...]) -> None:
        """
        Ensure that the Regridder is properly set up before applying interpolation.
        Checks if the interpolator is initialized and if the correct number of arguments
        are provided (either one for scalar fields or two for vector fields)."""

        if len(args) not in (1, 2):
            raise TypeError("Provide scalar_src or (u_src, v_src) as positional args")

    def __call__(
        self,
        *args: Any,
    ) -> Any:
        """
        Supported calls:
          - apply(scalar_src) -> scalar interpolation
        """

        self._ensure_ready(args)

        # Check if components have identical grids internally and
        # returns fields as-is (from source to destination) if so,
        # avoiding unnecessary computation
        if self.has_identical_grids:
            return args if len(args) == 2 else args[0]

        if self.interpolator is None:
            raise RegridderError("Regridder not properly set up")
        if len(args) == 1:
            return self.interpolator.apply_scalar(args[0])
        return cast(tuple[Any, Any], self.interpolator.apply_vector(args[0], args[1]))

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
