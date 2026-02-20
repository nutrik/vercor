from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid


if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Land(Component):
    """Toy bucket land model: soil moisture evolves from P-E (here: uses latent_heat_flux sign as proxy).
    Outputs: soil_moisture [0..1]
    Inputs: latent_heat_flux (proxy for evaporation)
    """

    def __init__(self, grid: RectilinearGrid, name: str = "LND") -> None:
        super().__init__(name, grid)

    def initialize(self, coupler: "Coupler") -> None:
        self.data["soil_moisture"] = 0.3 * np.ones(self.grid.shape)
        self.data["land_surface_temperature"] = np.zeros(self.grid.shape) + 288.15

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        latent_heat_flux = self.data["latent_heat_flux"]
        soil_moisture = self.data["soil_moisture"]

        evap = 1e-9 * (
            latent_heat_flux if latent_heat_flux is not None else 0.0
        )  # tiny dt scaling
        soil_moisture = np.clip(soil_moisture - evap * dt.total_seconds(), 0.0, 1.0)
        self.data["soil_moisture"] = soil_moisture
