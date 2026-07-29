"""Run the bundled JCM atmosphere/land setup with ERA5 ocean forcing."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RunState,
    RuntimeOptions,
)
from vercor.components import DataComponent
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.recipes import (
    ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
    ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
    JCM_LAND_TO_ATMOSPHERE_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
)
from vercor.regridding import bilinear
from vercor.setups import (
    JAXGCMConfig,
    JCMLandAtmosphereConfig,
    JCMInputs,
    Spinup,
    make_era5_ocean,
    make_jcm_land_atmosphere,
)
from vercor.topology import SurfaceMaskPolicy


def _default_clock(*, steps: int = 365 * 100 - 2) -> Clock:
    """Return the historic noleap clock, optionally with a shorter run."""

    return Clock(
        start=datetime(2000, 1, 3, 0, 0, 0),
        dt_seconds=86400.0,
        steps=steps,
        calendar="noleap",
    )


def build_coupler(
    *,
    ocean: DataComponent | None = None,
    jcm_inputs: JCMInputs | None = None,
    clock: Clock | None = None,
) -> Coupler:
    """Build the example coupler, reusing supplied model/data objects."""

    ocn = make_era5_ocean() if ocean is None else ocean
    jcm_setup = make_jcm_land_atmosphere(
        ocn.grid,
        inputs=jcm_inputs,
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

    return Coupler(
        _default_clock() if clock is None else clock,
        components=[ocn, lnd, atm],
        exchanges=(
            Exchange(
                source=atm.name,
                target=ocn.name,
                fields=ATMOSPHERE_TO_DATA_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source=ocn.name,
                target=atm.name,
                fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source=lnd.name,
                target=atm.name,
                fields=JCM_LAND_TO_ATMOSPHERE_FIELDS,
                regridder_factory=bilinear,
            ),
            Exchange(
                source=atm.name,
                target=lnd.name,
                fields=ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS,
                regridder_factory=bilinear,
            ),
        ),
        run_order=[ocn.name, lnd.name, atm.name],
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
    )


def _parse_args(arguments: Sequence[str] | None) -> argparse.Namespace:
    """Parse optional short-run and initialization-only CLI controls."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-state-only",
        action="store_true",
        help="prepare and validate the coupled initial state without stepping",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the historic 100-year workflow with a short step count",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> RunState:
    """Run the example or return its prepared initial state."""

    args = _parse_args(arguments)
    if args.steps is not None and args.steps < 0:
        raise ValueError("--steps must be non-negative")
    clock = None if args.steps is None else _default_clock(steps=args.steps)
    coupler = build_coupler(clock=clock)
    if args.initial_state_only:
        return coupler.initial_state()

    return coupler.run(output=OutputTarget("."))


if __name__ == "__main__":
    main()
