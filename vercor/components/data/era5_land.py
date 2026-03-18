from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vercor.clock import CustomDateTime
from vercor.components import Component, ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERA5Land(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "LND",
        surface_file: Optional[Path] = None,
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            surface_file (Path): path to netCDF file with data at surface level

        Attributes of parent classes to be initialized:
            ComponentForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
        """

        if surface_file is None:
            surface_file = get_forcing_data("era5_land_masked")

        self.DATA_FILES = {
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("lon", where="surface")
        latitude = self._read_forcing("lat", where="surface")
        binary_mask = self._read_forcing("mask", where="surface").T
        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self.settings.apply_time_interpolation = True

        # Units: [K]
        self.data["land_surface_temperature"] = self._read_forcing(
            "skt", where="surface"
        )

    def initialize(self, coupler: "Coupler") -> None:
        pass

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        """
        Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        pass
