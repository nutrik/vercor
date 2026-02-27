from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from dinosaur.coordinate_systems import CoordinateSystem
from jcm.forcing import ForcingData

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid
from vercor.regridders.helpers import compute_land_mask


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def create_new_jcm_lnd_mask(
    atm_lat: NDArray, atm_lon: NDArray, ocn_grid: RectilinearGrid
) -> tuple[NDArray, NDArray]:
    """Create a new land mask from Ocean & JCM geometry object."""

    from vercor.regridders.conservative import ConservativeRectilinearRegridder
    from vercor.interpolators.conservative_remap_rectilinear import (
        ConservativeRectilinearRemapper,
    )
    from vercor.exceptions import (
        RegridderError,
    )

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atm_lon,
        latitude=atm_lat,
    )

    regridder = ConservativeRectilinearRegridder(
        ocn_grid,
        atmosphere_grid,
    )

    ocean_bmask = np.asarray(ocn_grid.binary_mask)
    ocn_fmask_on_atm_grid = np.asarray(regridder(ocean_bmask))
    ocn_fmask_on_atm_grid = np.clip(ocn_fmask_on_atm_grid, 0.0, 1.0)
    lnd_fmask_on_atm_grid = 1.0 - ocn_fmask_on_atm_grid
    lnd_bmask_on_atm_grid = compute_land_mask(ocn_fmask_on_atm_grid)

    do_not_check_mass = False

    if regridder.interpolator is not None and isinstance(
        regridder.interpolator, ConservativeRectilinearRemapper
    ):
        src_lat = regridder.interpolator.src_lat_b
        dst_lat = regridder.interpolator.dst_lat_b
        if src_lat[-1] != dst_lat[-1] or src_lat[0] != dst_lat[0]:
            do_not_check_mass = True
            print(
                "\nWARNING: Skipping mass conservation check for regridding ocean mask to atmospheric grid "
                "due to different latitude bounds.\n"
            )

        src_total_mass = regridder.interpolator.get_src_total_mass(ocean_bmask)
        dst_total_mass = regridder.interpolator.get_dst_total_mass(
            ocn_fmask_on_atm_grid
        )

        if not do_not_check_mass and not np.isclose(
            src_total_mass, dst_total_mass, atol=1e-7
        ):
            raise RegridderError(
                "Regridding ocean binary mask to atmospheric grid does not conserve total mass "
                f"(source mass: {src_total_mass}, destination mass: {dst_total_mass})"
            )

    fmask_sum = lnd_fmask_on_atm_grid + ocn_fmask_on_atm_grid
    min_fsum = fmask_sum.min()
    max_fsum = fmask_sum.max()
    if not (
        np.isclose(min_fsum, 1.0, atol=1e-3) and np.isclose(max_fsum, 1.0, atol=1e-3)
    ):
        raise RegridderError(
            "Fractional land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
            f"(minimum sum {min_fsum}, maximum sum {max_fsum})"
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

        self._settings["get_field_time_slice"] = True

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
