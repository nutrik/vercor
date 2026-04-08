from datetime import datetime, timedelta

from vercor import Clock, Coupler, Exchange
from vercor.components import CAMulatorGCM, CAMulatorLand, VerosGCM

from vercor.coupler import RunSequence
from vercor.regridders import bilinear


if __name__ == "__main__":
    ocn = VerosGCM(
        do_spinup=True,
        custom_parameters={"dt_tracer": timedelta(hours=6).total_seconds()},
    )

    atm = CAMulatorGCM(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_config.yml",
        model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
        output_subfolder_name="test_veros_00091",
    )

    lnd = CAMulatorLand(
        config_path="/glade/u/home/rnuterman/veros_coupling/climate/camulator_land_config.yml",
        camulator_grid=atm.grid,
        ocn_grid=ocn.grid,
        model_weights_path="/glade/u/home/rnuterman/veros_coupling/climate/checkpoint.pt00091.pt",
    )

    clock = Clock(
        start=datetime(1981, 1, 3, 0, 0, 0),
        dt_seconds=86400.0 // 4,
        steps=100 - 2 * 4,
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
            field_names=["land_surface_temperature"],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()
