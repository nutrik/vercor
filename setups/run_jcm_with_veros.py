from datetime import datetime

from setups.jax_array_helpers import transposed_host_array
from vercor import Clock, Coupler, Exchange, RunSequence
from setups.data.jcm_land import make_jcm_land
from setups.external.jax_gcm import make_jax_gcm
from setups.external.veros_gcm import make_veros_gcm
from setups.external.jax_gcm_tools import (
    generate_jcm_coords_forcing_topography_files,
    get_default_parameter_values,
)
from setups.exchange_recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
)
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

    ocn = make_veros_gcm(do_spinup=True)

    coords, terrain, forcing = generate_jcm_coords_forcing_topography_files()

    lnd = make_jcm_land(coords, forcing, ocn.grid)

    # Swap mask in JAXGCM with ocean/land masks from ocean model
    terrain.fmask = transposed_host_array(lnd.grid.binary_mask)  # type: ignore

    # Build components
    atm = make_jax_gcm(
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
            field_names=list(ATMOSPHERE_TO_VEROS_FORCING_FIELDS),
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
            field_names=list(JCM_LAND_TO_ATMOSPHERE_FIELDS),
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=list(ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS),
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
