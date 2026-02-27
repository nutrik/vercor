from datetime import datetime

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import ERAInterimOcean, JCMLand, JAXGCM
from vercor.components.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
)
from vercor.coupler import RunSequence
from vercor.regridders import bilinear


if __name__ == "__main__":
    # This ocean data & grid is identical to Veros global 4deg. setup
    ocn = ERAInterimOcean()

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    lnd = JCMLand(coords, forcing, ocn.grid)

    # Swap mask in JAXGCM with ocean/land masks from ocean model
    terrain.fmask = np.array(lnd.grid.binary_mask).T  # type: ignore

    # Build components
    atm = JAXGCM(coords, terrain, forcing_data=forcing, do_spinup=True, jitted=True)

    # Clock and sequence
    clock = Clock(
        start=datetime(2025, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=10,
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
