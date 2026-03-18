from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.forcing import ForcingData

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid
from vercor.tools import create_lnd_mask_from_ocn

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class JCMLand(Component):
    def __init__(
        self,
        jcm_coords: CoordinateSystem,
        jcm_forcing: ForcingData,
        ocn_grid: RectilinearGrid,
        name: str = "LND",
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            jcm_coords (CoordinateSystem): JCM coordinate system object
            jcm_forcing (ForcingData): JCM forcing data object

        Attributes of parent classes to be initialized:
            Component
                name: str
                grid: RectilinearGrid
        """

        longitude = np.rad2deg(jcm_coords.horizontal.longitudes)
        latitude = np.rad2deg(jcm_coords.horizontal.latitudes)
        lnd_bmask, _ = create_lnd_mask_from_ocn(
            atm_lat=latitude,
            atm_lon=longitude,
            ocn_grid=ocn_grid,
        )

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=lnd_bmask,
        )

        super().__init__(name, grid=self.grid)

        self.settings.get_field_time_slice = True

        # Units: [K]
        self.data["land_surface_temperature"] = jcm_forcing.stl_am.T
        # Units: [???]
        self.data["soil_moisture"] = jcm_forcing.soilw_am.T

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
