from datetime import datetime

from vercor import Clock, Exchange, RunSequence
from setups.coupler_helpers import add_exchanges, build_coupler
from setups.data.era5_atmosphere import make_era5_atmosphere
from setups.data.era5_land import make_era5_land
from setups.external.veros_gcm import make_veros_gcm
from setups.exchange_recipes import ATMOSPHERE_TO_VEROS_FORCING_FIELDS
from vercor.regridders import bilinear

if __name__ == "__main__":
    atm = make_era5_atmosphere()
    ocn = make_veros_gcm(restore_to_climatology=True)
    lnd = make_era5_land()

    # Clock and sequence
    clock = Clock(
        start=datetime(2000, 1, 1, 0, 0, 0),
        dt_seconds=86400.0,
        steps=365,
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
                field_names=list(ATMOSPHERE_TO_VEROS_FORCING_FIELDS),
                regridder_factory=bilinear,
            ),
            Exchange(
                source="OCN",
                destination="ATM",
                field_names=[
                    "sea_surface_temperature",
                ],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="ATM",
                destination="LND",
                field_names=[
                    "temperature",
                    "specific_humidity",
                ],
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=[
                    "land_surface_temperature",
                ],
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
