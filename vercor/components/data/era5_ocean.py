from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vercor.clock import CustomDateTime
from vercor.components import Component, ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERA5Ocean(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "OCN",
        surface_file: Path = get_forcing_data("surface"),
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

        self.DATA_FILES = {
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("longitude", where="surface")
        latitude = self._read_forcing("latitude", where="surface")[::-1]
        fraction_mask = self._read_forcing("lsm", where="surface", flip_y=True).T[0, ::]
        fraction_mask = np.where(fraction_mask > 0.0, 1.0, 0.0)
        binary_mask = 1 - fraction_mask

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self._settings["apply_time_interpolation"] = True
        # Units: [K]
        self.data["sea_surface_temperature"] = self._read_forcing(
            "sst", where="surface", flip_y=True
        )
        self.data["sea_surface_temperature"] *= np.where(
            binary_mask > 0.0, 1.0, np.nan
        ).T[..., np.newaxis]

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
