from datetime import datetime

from vercor import Clock, Exchange, RunSequence
from vercor.setups.coupler_helpers import add_exchanges, build_coupler
from vercor.setups.data.era5_atmosphere import make_era5_atmosphere
from vercor.setups.data.era5_land import make_era5_land
from vercor.setups.external.veros_gcm import make_veros_gcm
from vercor.setups.exchange_recipes import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
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
                source="ATM",
                destination="LND",
                field_names=ATMOSPHERE_TO_LAND_BASIC_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source="LND",
                destination="ATM",
                field_names=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
