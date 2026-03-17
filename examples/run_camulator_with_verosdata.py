from datetime import datetime, timedelta

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import ERAInterimOcean, CAMulatorGCM

from vercor.coupler import RunSequence
from vercor.grid import RectilinearGrid
from vercor.regridders import bilinear
from vercor.components import Component
from vercor.tools import create_lnd_mask_from_ocn
from vercor.clock import CustomDateTime


class CAMulatorLand(Component):
    def __init__(
        self,
        camulator_grid: RectilinearGrid,
        ocn_grid: RectilinearGrid,
        name: str = "LND",
    ) -> None:
        """
        Read all necessary fields from the provided forcing files.

        Arguments:
            name (str): component name
            camulator_grid (RectilinearGrid): CAMulator grid object
            ocn_grid (RectilinearGrid): Ocean component grid object

        Attributes of parent classes to be initialized:
            Component
                name: str
                grid: RectilinearGrid
        """

        longitude = camulator_grid.longitude
        latitude = camulator_grid.latitude
        lnd_bmask, _ = create_lnd_mask_from_ocn(
            atm_lat=latitude,
            atm_lon=longitude,
            ocn_grid=ocn_grid,
        )

        grid = RectilinearGrid(
            name=f"{name.lower()}-grid",
            longitude=longitude,
            latitude=latitude,
            binary_mask=lnd_bmask,
        )

        super().__init__(name, grid=grid)

        # Units: [K]
        self.data["land_surface_temperature"] = np.full(
            self.grid.shape, 283.0, dtype=np.float32
        )

    def initialize(self, coupler: "Coupler") -> None:
        pass

    def step(
        self,
        dt: timedelta,
        time: datetime | CustomDateTime,
        coupler: "Coupler",
    ) -> None:
        pass


if __name__ == "__main__":
    # This ocean data & grid is identical to Veros global 4deg. setup
    ocn = ERAInterimOcean()

    atm = CAMulatorGCM(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yaml",
        model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
        output_subfolder_name="test_veros_00091",
    )

    lnd = CAMulatorLand(atm.grid, ocn.grid)

    clock = Clock(
        start=datetime(1981, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=30,
        year_type="noleap",
    )
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

    # Coupler
    cpl = Coupler(clock=clock)
    components = [ocn, lnd, atm]
    for component in components:
        cpl.register(component)  # type: ignore

    cpl.set_components_run_sequence(run_sequence)

    # Exchanges
    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=[
                ("u_velocity", "v_velocity"),
                "specific_humidity",
                "temperature",
                "model_level_height",
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["sea_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["land_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()
