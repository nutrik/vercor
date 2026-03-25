from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from vercor.clock import CustomDateTime
from vercor.components import Component, ComponentForcingData
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERAInterimOcean(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "OCN",
        resolution: str = "4deg",
        model_level_file: Optional[Path] = None,
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            model_level_file (Path): path to netCDF file with data at model levels
            surface_file (Path): path to netCDF file with data at surface level

        Attributes of parent classes to be initialized:
            ComponentForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
        """

        if model_level_file is None:
            model_level_file = get_forcing_data(f"erainterim_ocean_{resolution}")

        self.DATA_FILES = {
            "model_level": str(model_level_file),
        }

        longitude = self._read_forcing("xt", where="model_level")
        grid_step = longitude[1] - longitude[0]
        yt_bndry = 89.5 if grid_step == 1 else 90.0
        latitude_slice = slice(10, -10) if grid_step == 1 else slice(3, -3)

        # To cover the whole globe with 1 or 4 degree resolution
        latitude: NDArray = np.arange(-yt_bndry, yt_bndry + grid_step, grid_step)
        sss: NDArray = np.zeros((longitude.size, latitude.size, 12))
        sst: NDArray = np.zeros((longitude.size, latitude.size, 12))

        latitude[latitude_slice] = self._read_forcing("yt", where="model_level")
        longitude = longitude - 90.0 if grid_step == 1 else longitude
        sss[:, latitude_slice, :] = self._read_forcing("sss", where="model_level")
        binary_mask = np.where(sss > 0.0, 1.0, 0.0)[..., 0].T

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self.settings.apply_time_interpolation = True

        sst[:, latitude_slice, :] = self._read_forcing("sst", where="model_level") + 273.15
        sst *= np.where(binary_mask > 0.0, 1.0, np.nan).T[..., np.newaxis]
        # Units: [K]
        self.data["sea_surface_temperature"] = sst

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
