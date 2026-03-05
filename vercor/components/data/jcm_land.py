from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.forcing import ForcingData

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid
from vercor.tools import (
    check_remap_conservation,
    check_total_lnd_ocn_mask_sum,
    compute_ocn_lnd_masks_on_atm_grid,
)


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def create_new_jcm_lnd_mask(
    atm_lat: NDArray, atm_lon: NDArray, ocn_grid: RectilinearGrid
) -> tuple[NDArray, NDArray]:
    """Create a new land mask from Ocean & JCM geometry object."""

    from vercor.regridders.conservative import ConservativeRectilinearRegridder

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atm_lon,
        latitude=atm_lat,
    )

    regridder = ConservativeRectilinearRegridder(
        ocn_grid,
        atmosphere_grid,
    )

    ocean_binary_mask = np.asarray(ocn_grid.binary_mask)

    (
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

    check_remap_conservation(regridder, ocean_binary_mask, ocn_fmask_on_atm_grid)

    check_total_lnd_ocn_mask_sum(
        lnd_fmask_on_atm_grid,
        ocn_fmask_on_atm_grid,
    )

    return lnd_bmask_on_atm_grid, lnd_fmask_on_atm_grid


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
        lnd_bmask, _ = create_new_jcm_lnd_mask(
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
