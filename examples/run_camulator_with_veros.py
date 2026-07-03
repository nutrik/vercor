from datetime import datetime, timedelta

from vercor import Clock, Coupler, Exchange
from vercor.setups import make_camulator_gcm
from vercor.setups import make_camulator_land
from vercor.setups import make_veros_gcm
from vercor.exchanges import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)

from vercor.regridding import bilinear

if __name__ == "__main__":
    ocn = make_veros_gcm(
        do_spinup=True,
        output_frequency="month",
        output_variables=(
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "surface_taux",
            "surface_tauy",
            "psi",
        ),
        custom_parameters={"dt_tracer": timedelta(hours=6).total_seconds()},
    )

    atm = make_camulator_gcm(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yml",
        model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
        output_subfolder_name="camulator_veros_v2_00091",
        output_frequency="month",
    )

    lnd = make_camulator_land(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_land_config.yml",
        camulator_grid=atm.grid,
        ocn_grid=ocn.grid,
    )

    clock = Clock(
        start=datetime(1981, 1, 3, 0, 0, 0),
        dt_seconds=86400.0 // 4,
        steps=365 - 2 * 4,
        year_type="noleap",
    )
    run_sequence = ["OCN", "LND", "ATM"]

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
                fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
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
                fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
                regrid=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
