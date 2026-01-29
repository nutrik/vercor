from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.components import ERA5Atmosphere, ERA5Land, VerosGCM
from vercor.coupler import RunSequence
from vercor.regridders import bilinear


if __name__ == "__main__":
    atm = ERA5Atmosphere()
    ocn = VerosGCM()
    lnd = ERA5Land()

    # Clock and sequence
    clock = Clock(start=datetime(2025, 1, 1, 0, 0, 0), dt_seconds=86400.0, steps=360)
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
            field_names=[
                "sea_surface_temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="ATM",
            destination="LND",
            field_names=[
                "temperature",
                "specific_humidity",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.add_exchange(
        Exchange(
            source="LND",
            destination="ATM",
            field_names=[
                "land_surface_temperature",
            ],
            regridder_factory=bilinear,
        )
    )

    cpl.initialize()
    cpl.run()
    cpl.finalize()
