from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from vercor.components import Component
from vercor.grid import RectilinearGrid


if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Atmosphere(Component):
    """Toy atmosphere: produces surface fluxes and 2m temperature from sea_surface_temperature.
    Inputs: sea_surface_temperature [K]
    Outputs: sensible_heat_flux [W/m2], latent_heat_flux [W/m2], temperature_2m [K]
    """

    def __init__(self, name: str, grid: RectilinearGrid) -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        grid_shape = self.grid.shape
        zeros = np.zeros(grid_shape)

        self.data["temperature_2m"] = 273.15 + 15.0 * np.ones(grid_shape)
        self.data["sensible_heat_flux"] = zeros
        self.data["latent_heat_flux"] = zeros
        self.data["u_velocity_10m"] = zeros
        self.data["v_velocity_10m"] = zeros

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:
        # Bulk formula toy: flux proportional to (temperature_2m - sea_surface_temperature)
        sst = self.data.get("sea_surface_temperature", None)

        if sst is None:
            sst = 273.15 + 15.0 * np.ones(self.grid.shape)

        TA = self.data["temperature_2m"]
        dT = TA - sst
        C = 10.0  # W m-2 K-1, toy exchange coefficient
        SHF = -C * dT  # ocean heat gain positive when sst < TA
        LHF = -0.5 * SHF

        # Update wind (toy)
        lat = np.array(self.grid.latitude)
        lon = np.array(self.grid.longitude) - 180.0
        latitudes, longitudes = np.meshgrid(lat, lon, indexing="ij")
        self.data["u_velocity_10m"] = np.cos(
            np.deg2rad(latitudes)
        )  # zonal flow varying with latitude
        self.data["v_velocity_10m"] = 0.5 * np.sin(
            np.deg2rad(longitudes)
        )  # small meridional perturbation

        self.data["sensible_heat_flux"] = SHF
        self.data["latent_heat_flux"] = LHF

        # Relax temperature_2m toward sst weakly (toy boundary layer)
        self.data["temperature_2m"] = TA - 0.01 * dT
