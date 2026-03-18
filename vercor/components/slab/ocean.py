from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from vercor.clock import CustomDateTime
from vercor.components import Component
from vercor.grid import RectilinearGrid

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class Ocean(Component):
    """Toy slab ocean: updates sea_surface_temperature using sensible_heat_flux (sensible) + latent_heat_flux (latent).
    Outputs: sea_surface_temperature [K]
    Inputs: sensible_heat_flux, latent_heat_flux
    """

    def __init__(
        self, grid: RectilinearGrid, name: str = "OCN", H: float = 30.0
    ) -> None:
        super().__init__(name, grid)

        self.H = H  # mixed-layer depth [m]
        self.rho = 1025.0
        self.cp = 3990.0
        self.lambda_relax = 1.0 / (
            30.0 * 86400.0
        )  # weak restoring to 15C over ~30 days

    def initialize(self, coupler: "Coupler") -> None:
        self.data["sea_surface_temperature"] = 273.15 + 15.0 * np.ones(self.grid.shape)

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        sst = self.data.get("sea_surface_temperature", None)
        if sst is None:
            return

        SHF = self.data.get("sensible_heat_flux", None)
        LHF = self.data.get("latent_heat_flux", None)
        Qnet = np.zeros_like(sst)
        if SHF is not None:
            Qnet += SHF
        if LHF is not None:
            Qnet += LHF
        T0 = 273.15 + 15.0
        dTdt = Qnet / (self.rho * self.cp * self.H) + self.lambda_relax * (sst - T0)
        self.data["sea_surface_temperature"] = sst + dTdt * dt.total_seconds()
