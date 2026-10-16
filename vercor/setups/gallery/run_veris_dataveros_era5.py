"""Standalone Veris with prescribed global 4° Veros ocean and ERA5 data.

The sequence is LND -> CICE -> OCN -> ATM. Both incoming ice routes carry
intensive values on ocean wet cells, including initially ice-free cells.
Ocean currents, height anomaly and runoff are prescribed zero. Native initial
conditions remain configurable; no ice feedback is sent to data components.
"""

from datetime import datetime

from vercor import Clock, Coupler, Exchange, RuntimeOptions
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.recipes import (
    ATMOSPHERE_TO_VERIS_FORCING_FIELDS,
    OCEAN_TO_VERIS_FORCING_FIELDS,
)
from vercor.regridding import bilinear
from vercor.setups import (
    VerisConfig,
    make_era5_atmosphere,
    make_era5_land,
    make_veris_verosdata,
)


def _make_coupler(
    *,
    loglevel: str,
    float_type: str,
    steps: int = 365,
    veris_config: VerisConfig | None = None,
) -> Coupler:
    """Assemble the same experiment for CLI runs and short native verification."""
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
                    ),
                )
            )
        )
    pair = make_veris_verosdata(config=veris_config)
    ice, ocean = pair.seaice, pair.ocean
    atmosphere = make_era5_atmosphere()
    land = make_era5_land()
    return Coupler(
        clock=Clock(datetime(2000, 1, 1), 86400.0, steps, calendar="noleap"),
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
        ),
        run_order=(land.name, ice.name, ocean.name, atmosphere.name),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=float_type == "float64")),
        log_level=loglevel,
    )


def run_setup(*, loglevel: str, float_type: str) -> None:
    """Run the daily coupled example through the standard VerCOR CLI contract."""
    coupler = _make_coupler(loglevel=loglevel, float_type=float_type)
    coupler.run(output=OutputTarget("."))


if __name__ == "__main__":
    run_setup(loglevel="info", float_type="float64")
