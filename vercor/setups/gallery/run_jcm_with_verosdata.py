from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import make_erainterim_ocean
from vercor.recipes import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.setups import JAXGCMConfig, JCMLandAtmosphereConfig, Spinup
from vercor.setups import make_jcm_land_atmosphere
from vercor.regridding import bilinear
from vercor.topology import SurfaceMaskPolicy


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run this setup through the shared VerCOR CLI contract."""

    dtype = DTypePolicy(enable_x64=float_type == "float64")
    dtype._ensure_jax_capability()
    # This ocean data & grid is identical to Veros global setup (1deg. or 4deg.)
    ocn = make_erainterim_ocean(resolution="4deg")

    jcm_setup = make_jcm_land_atmosphere(
        ocn.grid,
        config=JCMLandAtmosphereConfig(
            atmosphere=JAXGCMConfig(
                spinup=Spinup(enabled=True),
                output=OutputSpec(period=PeriodOutput(frequency="month")),
                jitted=True,
            ),
        ),
    )
    lnd = jcm_setup.land
    atm = jcm_setup.atmosphere

    # Clock and sequence
    # Gregorian continuation after two spinup days, ending 2100-01-01.
    clock = Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=(datetime(2100, 1, 1) - datetime(2000, 1, 3)).days,
        calendar="gregorian",
    )
    run_order = ["OCN", "LND", "ATM"]

    # Exchanges
    exchanges = (
        Exchange(
            quantity="intensive",
            source="ATM",
            target="OCN",
            fields=ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source="LND",
            target="ATM",
            fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    components = [ocn, lnd, atm]
    cpl = Coupler(
        clock=clock,
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(dtype=dtype, topology=SurfaceMaskPolicy()),
        log_level=loglevel,
    )

    cpl.run(output=OutputTarget("."))


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
