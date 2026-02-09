from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from vercor.components import Component, ComponentForcingData
from vercor.fluxes.utilities import (
    compute_air_density,
    get_altitudes_hybrid_sigma_levels,
    compute_pressure_levels,
    compute_potential_temperature,
)
from vercor.grid import RectilinearGrid
from vercor.tools import get_forcing_data

if TYPE_CHECKING:
    from vercor.coupler import Coupler


class ERA5Atmosphere(Component, ComponentForcingData):
    def __init__(
        self,
        name: str = "ATM",
        model_level_file: Path = get_forcing_data("model_level"),
        surface_file: Path = get_forcing_data("surface"),
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            model_level_file (Path): path to netCDF file with data at model levels
            surface_file (Path): path to netCDF file with data at surface level

        Data description:
            Only the lowest to the ground model levels are available and read (L136, L137)
            See ECMWF IFS documentation on vertical model resolution for more details:
            https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions

        Attributes of parent classes to be initialized:
            ComponentForcingData
                DATA_FILES: dict [str, str]
            Component
                name: str
                grid: RectilinearGrid
        """

        self.DATA_FILES = {
            "model_level": str(model_level_file),
            "surface": str(surface_file),
        }

        longitude = self._read_forcing("longitude", where="model_level")
        latitude = self._read_forcing("latitude", where="model_level")[::-1]

        self.grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
        )

        super().__init__(name, grid=self.grid)

        self._settings["apply_time_interpolation"] = True

        self.data["hyai"] = self._read_forcing("hyai", where="model_level")[
            -3:
        ]  # L135-L137
        self.data["hybi"] = self._read_forcing("hybi", where="model_level")[
            -3:
        ]  # L135-L137
        self.data["hyam"] = self._read_forcing("hyam", where="model_level")[
            -2:
        ]  # L136-L137
        self.data["hybm"] = self._read_forcing("hybm", where="model_level")[
            -2:
        ]  # L136-L137

        lnsp = self._read_forcing("lnsp", where="model_level", flip_y=True)[..., 0, :]
        # Units: [Pa]
        self.data["surface_pressure"] = np.exp(lnsp)
        # Units: [kg/kg]
        self.data["specific_humidity_3d"] = self._read_forcing(
            "q", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        # Units: [K]
        self.data["temperature_3d"] = self._read_forcing(
            "t", where="model_level", flip_y=True
        )[
            ..., 1:, :
        ]  # L136-L137
        # Units: [m/s]
        self.data["u_velocity"] = self._read_forcing(
            "u", where="model_level", flip_y=True
        )[
            :, :, 1, :
        ]  # L136
        # Units: [m/s]
        self.data["v_velocity"] = self._read_forcing(
            "v", where="model_level", flip_y=True
        )[
            :, :, 1, :
        ]  # L136

        # tcc = self._read_forcing("tcc", where="surface", flip_y=True)
        # Units: [W/m²]
        self.data["net_shortwave_radiation_flux"] = self._read_forcing(
            "msnswrf", where="surface", flip_y=True
        )
        # Units: [W/m²]
        self.data["downward_longwave_radiation_flux"] = self._read_forcing(
            "msdwlwrf", where="surface", flip_y=True
        )
        # Units: [kg/kg]
        self.data["specific_humidity"] = self.data["specific_humidity_3d"][
            ..., 0, :
        ]  # L136
        # Units: [K]
        self.data["temperature"] = self.data["temperature_3d"][..., 0, :]  # L136

    def initialize(self, coupler: "Coupler") -> None:
        nlat, nlon = self.grid.shape
        settings = coupler.settings
        ds = self.data

        ds["model_level_height"] = np.zeros((nlon, nlat, 12))
        ds["density"] = np.zeros((nlon, nlat, 12))
        ds["potential_temperature"] = np.zeros((nlon, nlat, 12))

        for m in range(12):
            # Units: [Pa]
            ph = compute_pressure_levels(
                ds["surface_pressure"][..., m], ds["hyai"], ds["hybi"]
            )
            # Units: [Pa]
            pf = compute_pressure_levels(
                ds["surface_pressure"][..., m], ds["hyam"], ds["hybm"]
            )
            # Units: [m]
            self.data["model_level_height"][..., m] = get_altitudes_hybrid_sigma_levels(
                settings,
                ds["temperature_3d"][..., m],
                ds["specific_humidity_3d"][..., m],
                ph[...],
            )[
                ..., 1
            ]  # L136
            # Units: [kg/m³]
            self.data["density"][..., m] = compute_air_density(
                settings, pf[:, :, 0], ds["temperature"][:, :, m]
            )
            # Units: [K]
            self.data["potential_temperature"][..., m] = compute_potential_temperature(
                settings, ds["temperature"][:, :, m], pf[:, :, 0]
            )

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
        # Units: [K]
        self.data["total_surface_temperature"] = np.nan_to_num(
            self.data["land_surface_temperature"], nan=0.0
        ) + np.nan_to_num(self.data["sea_surface_temperature"], nan=0.0)
