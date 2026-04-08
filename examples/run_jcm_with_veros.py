from datetime import datetime

import numpy as np

from vercor import Clock, Coupler, Exchange
from vercor.components import JCMLand, VerosGCM, JAXGCM
from vercor.components.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
    get_default_parameter_values,
)
from vercor.coupler import RunSequence
from vercor.regridders import bilinear

from jcm.physics.speedy.params import Parameters


if __name__ == "__main__":
    optimized_parameters: list = [
        "surface_flux.vgust",
        "convection.rhbl",
        "condensation.rhlsc",
        "surface_flux.cds",
    ]

    custom_jcm_parameters: dict[str, float] = get_default_parameter_values(
        parameters=optimized_parameters,
        default_parameters=Parameters.default(),
    )

    # change the values of the parameters to be optimized here
    # custom_jcm_parameters['surface_flux.vgust'] = 5.01

    ocn = VerosGCM(do_spinup=True)

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    lnd = JCMLand(coords, forcing, ocn.grid)

    # Swap mask in JAXGCM with ocean/land masks from ocean model
    terrain.fmask = np.array(lnd.grid.binary_mask).T  # type: ignore

    # Build components
    atm = JAXGCM(
        coords,
        terrain,
        custom_parameters=custom_jcm_parameters,
        forcing_data=forcing,
        do_spinup=True,
        jitted=True,
        output_frequency="month",
    )

    # Clock and sequence
    # Note that the number of steps is set to 365*100-2,
    # which corresponds to 100 years of simulation with a daily time step,
    # starting from January 3rd, 2000.
    # The -2 accounts for the fact that the simulation starts on January 3rd,
    # because of 2 days spinup of JCM & Veros models,
    # so it will end on December 31st, 2099.
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365 * 100 - 2,
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
