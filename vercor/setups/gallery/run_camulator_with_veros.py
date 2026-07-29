from datetime import datetime, timedelta

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import CAMulatorConfig, Spinup, VerosConfig, make_camulator_gcm
from vercor.setups import make_camulator_land
from vercor.setups import make_veros_gcm
from vercor.recipes import (
    ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)

from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy

if __name__ == "__main__":
    ocn = make_veros_gcm(
        config=VerosConfig(
            spinup=Spinup(enabled=True),
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=(
                        "temp",
                        "salt",
                        "u",
                        "v",
                        "w",
                        "surface_taux",
                        "surface_tauy",
                        "psi",
                    ),
                ),
            ),
            custom_parameters={"dt_tracer": timedelta(hours=6).total_seconds()},
        ),
    )

    atm = make_camulator_gcm(
        config=CAMulatorConfig(
            config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yml",
            model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
            output=OutputSpec(period=PeriodOutput(frequency="month")),
        ),
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
        calendar="noleap",
    )
    run_order = ["OCN", "LND", "ATM"]

    # Exchanges
    exchanges = (
        Exchange(
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=ATMOSPHERE_TO_LAND_RADIATION_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )

    cpl.run(output=OutputTarget("."))
