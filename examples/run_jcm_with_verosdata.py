from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.setups import make_erainterim_ocean
from vercor.exchanges import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.setups.jcm_setup_helpers import build_jcm_land_atmosphere_components
from vercor.regridding import bilinear

if __name__ == "__main__":
    # This ocean data & grid is identical to Veros global setup (1deg. or 4deg.)
    ocn = make_erainterim_ocean(resolution="4deg")

    jcm_setup = build_jcm_land_atmosphere_components(
        ocn.grid,
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
    # because of 2 days spinup of JCM model, so it will end on December 31st, 2099.
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365 * 100 - 2,
        year_type="noleap",
    )
    run_sequence = ["OCN", "LND", "ATM"]

    # Coupler
    components = [ocn, lnd, atm]
    cpl = Coupler.from_components(
        clock=clock,
        components=components,
        run_order=run_sequence,
    )

    # Exchanges
    cpl.add_exchanges(
        (
            Exchange(
                source="ATM",
                target="OCN",
                fields=ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="OCN",
                target="ATM",
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regrid=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
