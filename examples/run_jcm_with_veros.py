from datetime import datetime

import numpy as np

from jcm.geometry import Geometry
from jcm.forcing import ForcingData

from vercor import Clock, Coupler, Exchange
from vercor.components import JCMLand, VerosGCM, JAXGCM
from vercor.components.external.jax_gcm_tools import (
    generate_jcm_forcing_and_topography_files,
)
from vercor.coupler import RunSequence
from vercor.regridders import bilinear


if __name__ == "__main__":
    ocn = VerosGCM(do_spinup=True)

    # Read JCM topography file
    external_files = generate_jcm_forcing_and_topography_files(resolution=31)
    geometry = Geometry.from_file(external_files["terrain"])
    forcing = ForcingData.from_file(external_files["forcing"])

    lnd = JCMLand(ocn.grid, external_files["forcing"])

    # Swap mask in JAXGCM with ocean/land masks from ocean model
    geometry.fmask = np.array(lnd.grid.binary_mask).T  # type: ignore

    # Build components
    atm = JAXGCM(
        geometry,
        forcing_data=forcing,
        do_spinup=True,
        jitted=True,
        output_frequency="month",
    )

    # Clock and sequence
    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=60,
        days_per_year=360,
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
                "model_level_height",
                "density",
                "potential_temperature",
                "temperature",
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
            field_names=["soil_moisture", "land_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["latent_heat_flux", "sensible_heat_flux"],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()
