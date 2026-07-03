from datetime import datetime

from vercor import Clock, Coupler, Exchange
from vercor.setups import make_era5_atmosphere
from vercor.setups import make_era5_land
from vercor.setups import make_veros_gcm
from vercor.exchanges import (
    ATMOSPHERE_TO_LAND_BASIC_FIELDS,
    ATMOSPHERE_TO_VEROS_FORCING_FIELDS,
    LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear

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
                source="ATM",
                target="LND",
                fields=ATMOSPHERE_TO_LAND_BASIC_FIELDS,
                regrid=bilinear,
            ),
            Exchange(
                source="LND",
                target="ATM",
                fields=LAND_TO_ATMOSPHERE_SURFACE_FIELDS,
                regrid=bilinear,
            ),
        ),
    )

    cpl.initialize()
    final_state = cpl.run()
    cpl.finalize(final_state)
