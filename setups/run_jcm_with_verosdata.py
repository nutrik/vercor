from datetime import datetime

from vercor import Clock, Exchange, RunSequence
from setups.coupler_helpers import add_exchanges, build_coupler
from setups.data.erainterim_ocean import make_erainterim_ocean
from setups.jcm_setup_helpers import build_jcm_land_atmosphere_components
from vercor.regridders import bilinear

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
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

    # Coupler
    components = [ocn, lnd, atm]
    cpl = build_coupler(
        clock=clock,
        components=components,
        run_sequence=run_sequence,
    )

    # Exchanges
    add_exchanges(
        cpl,
        (
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
            ),
            Exchange(
                source="OCN",
                destination="ATM",
                field_names=["sea_surface_temperature"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=["soil_moisture", "land_surface_temperature"],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=["latent_heat_flux", "sensible_heat_flux"],
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
