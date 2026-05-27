from datetime import datetime

from vercor import Clock, RunSequence
from vercor.setups.coupler_helpers import (
    ExchangeSpec,
    add_exchange_specs,
    build_coupler,
)
from vercor.setups.external.veros_gcm import make_veros_gcm
from vercor.setups.external.jax_gcm_tools import (
    get_default_parameter_values,
)
from vercor.setups.exchange_recipes import (
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.setups.jcm_setup_helpers import build_jcm_land_atmosphere_components
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

    jcm_setup = build_jcm_land_atmosphere_components(
        ocn.grid,
        custom_parameters=custom_jcm_parameters,
        do_spinup=True,
        jitted=True,
        output_frequency="month",
    )
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere

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
    components = [ocn, lnd, atm]
    cpl = build_coupler(
        clock=clock,
        components=components,
        run_sequence=run_sequence,
    )

    # Exchanges
    add_exchange_specs(
        cpl,
        (
            ExchangeSpec(
                source="ATM",
                destination="OCN",
                field_names=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ATM",
                field_names=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="LND",
                destination="ATM",
                field_names=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="LND",
                field_names=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
