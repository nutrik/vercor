"""Prognostic Veris/Veros on one exact ocean grid with prescribed ERA5 air.

LND -> CICE -> OCN -> ATM is an explicit sequential coupling: ice uses the
latest available ocean/air state; the ocean consumes that interval's total ice
boundary means; ERA5 advances last for the next interval. Incoming ice routes
are intensive. Outgoing total-area scalars use conservative remapping and
stress uses the vector-capable exact shared-grid identity. Both select binary
source support. Native ocean SST/SSS/currents/SSH are prognostic.
"""

from datetime import datetime

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
    VERIS_TO_OCEAN_FLUX_FIELDS,
)
from vercor.regridding import bilinear, conservative
from vercor.setups import (
    VerisConfig,
    VerosConfig,
    make_era5_atmosphere,
    make_era5_land,
    make_veris_veros,
)


def _make_coupler(
    *,
    loglevel: str,
    float_type: str,
    steps: int = 365,
    veris_config: VerisConfig | None = None,
    ocean_config: VerosConfig | None = None,
) -> Coupler:
    """Assemble shared prognostic geometry and explicitly lagged total feedback."""
    if float_type not in {"float32", "float64"}:
        raise ValueError("float_type must be float32 or float64")
    if veris_config is None:
        veris_config = VerisConfig(
            output=OutputSpec(
                period=PeriodOutput(
                    frequency="month",
                    variables=(
                        "Area",
                        "hIceMean",
                        "hSnowMean",
                        "TSurf",
                        "uIce",
                        "vIce",
                        "sea_surface_heat_flux",
                        "sea_surface_mass_flux",
                        "sea_surface_salt_mass_flux",
                    ),
                )
            )
        )
    pair = make_veris_veros(config=veris_config, ocean_config=ocean_config)
    ice, ocean = pair.seaice, pair.ocean
    atmosphere, land = make_era5_atmosphere(), make_era5_land()
    return Coupler(
        Clock(datetime(2000, 1, 1), 86400.0, steps, calendar="noleap"),
        components=(land, ice, ocean, atmosphere),
        exchanges=(
            Exchange(
                atmosphere.name,
                ice.name,
                ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
            Exchange(
                ocean.name,
                ice.name,
                OCEAN_TO_VERIS_FORCING_FIELDS,
                quantity="intensive",
                regridder_factory=bilinear,
            ),
            Exchange(
                ice.name,
                ocean.name,
                VERIS_TO_OCEAN_FLUX_FIELDS[:-1],
                route_id=f"{ice.name}->{ocean.name}.scalars",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=conservative,
            ),
            Exchange(
                ice.name,
                ocean.name,
                VERIS_TO_OCEAN_FLUX_FIELDS[-1:],
                route_id=f"{ice.name}->{ocean.name}.stress",
                quantity="source_flux",
                source_coverage="binary",
                regridder_factory=bilinear,
            ),
        ),
        run_order=(land.name, ice.name, ocean.name, atmosphere.name),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=float_type == "float64")),
        log_level=loglevel,
    )


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run the prognostic example through VerCOR's packaged CLI contract."""
    _make_coupler(loglevel=loglevel, float_type=float_type).run(
        output=OutputTarget(".")
    )


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
