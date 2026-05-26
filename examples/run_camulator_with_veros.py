from datetime import datetime, timedelta

from vercor import Clock, Exchange, RunSequence
from vercor.setups.coupler_helpers import add_exchanges, build_coupler
from vercor.setups.external.camulator import make_camulator_gcm
from vercor.setups.external.camulator_land import make_camulator_land
from vercor.setups.external.veros_gcm import make_veros_gcm
from vercor.setups.exchange_recipes import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)

from vercor.regridders import bilinear

if __name__ == "__main__":
    ocn = make_veros_gcm(
        do_spinup=True,
        custom_parameters={"dt_tracer": timedelta(hours=6).total_seconds()},
    )

    atm = make_camulator_gcm(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yml",
        model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
        output_subfolder_name="test_veros_00091",
    )

    lnd = make_camulator_land(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_land_config.yml",
        camulator_grid=atm.grid,
        ocn_grid=ocn.grid,
    )

    clock = Clock(
        start=datetime(1981, 1, 3, 0, 0, 0),
        dt_seconds=86400.0 // 4,
        steps=100 - 2 * 4,
        year_type="noleap",
    )
    run_sequence = RunSequence(order=["OCN", "LND", "ATM"])

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
                field_names=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                destination="ATM",
                field_names=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
