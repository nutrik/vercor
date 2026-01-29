from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from vercor.components import Component, ComponentForcingData
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERAInterimOcean(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "OCN",
        model_level_file: Path = (
            Path(__file__).parent.parent.parent
            / ".."
            / "forcing"
            / "forcing_4deg_global_open_itf.nc"
        ).resolve(),
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

        self.DATA_FILES = {
            "model_level": str(model_level_file),
        }

        # To cover the whole globe with 4 degree resolution
        latitude: NDArray = np.arange(-90.0, 94.0, 4.0)
        sss: NDArray = np.zeros((90, latitude.size, 12))
        sst: NDArray = np.zeros((90, latitude.size, 12))

        longitude = self._read_forcing("xt", where="model_level")
        latitude[3:-3] = self._read_forcing("yt", where="model_level")
        sss[:, 3:-3, :] = self._read_forcing("sss", where="model_level")
        binary_mask = np.where(sss > 0.0, 1.0, 0.0)[..., 0].T

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=binary_mask,
        )

        super().__init__(name, grid=self.grid)

        self._settings["apply_time_interpolation"] = True

        sst[:, 3:-3, :] = self._read_forcing("sst", where="model_level") + 273.15
        sst *= np.where(binary_mask > 0.0, 1.0, np.nan).T[..., np.newaxis]
        self.data["sea_surface_temperature"] = sst

    def initialize(self, coupler: "Coupler") -> None:
        pass

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        """
        Advance to the next time step in the dataset
        using time interpolation from one month to another.
        """
        pass
