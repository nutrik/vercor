from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Sequence

import jax

from vercor import Clock, Coupler
from vercor.setups.coupler_helpers import ExchangeSpec, add_exchange_specs
from vercor.setups.slab.atmosphere import make_slab_atmosphere
from vercor.setups.slab.land import make_slab_land
from vercor.setups.slab.ocean import make_slab_ocean
from vercor.setups.slab.seaice import make_slab_seaice
from vercor.dtypes import jax_ones
from vercor.grid_geometry import make_rectilinear_grid
from vercor.regridders import bilinear, conservative
from vercor.runtime.state import RuntimeCouplerState
from vercor.setups.exchange_recipes import (
    LAND_TO_ATMOSPHERE_SOIL_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    SEAICE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
)


@dataclass(frozen=True)
class RuntimeProfileResult:
    """Wall-clock timing for the pure scanned runtime path."""

    run_seconds: float
    final_state_leaves: int


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the runtime profiling harness."""

    parser = argparse.ArgumentParser(
        description="Profile VerCOR's pure JAX scanned runtime on a synthetic slab coupler."
    )
    parser.add_argument(
        "--steps", type=int, default=24, help="Number of runtime steps."
    )
    parser.add_argument(
        "--grid-nx",
        type=int,
        default=32,
        help="Number of longitude cells for the synthetic grids.",
    )
    parser.add_argument(
        "--grid-ny",
        type=int,
        default=16,
        help="Number of latitude cells for the synthetic grids.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Coupler log level. WARNING avoids per-step JAX host callbacks.",
    )
    return parser


def _block_until_ready(value: RuntimeCouplerState) -> RuntimeCouplerState:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def build_slab_coupler(
    *,
    steps: int,
    grid_nx: int,
    grid_ny: int,
    log_level: int | str,
) -> Coupler:
    """Build and initialize a small pure-JAX slab coupler for profiling."""

    atm_grid = make_rectilinear_grid(
        "profile-atm-grid",
        grid_nx,
        grid_ny,
        0.0,
        360.0,
        -90.0,
        90.0,
    )
    ocn_mask = jax_ones((grid_ny, grid_nx)).at[:2, :].set(0.0)
    ocn_grid = make_rectilinear_grid(
        "profile-ocn-grid",
        grid_nx,
        grid_ny,
        0.0,
        360.0,
        -90.0,
        90.0,
        mask=ocn_mask,
    )
    lnd_grid = make_rectilinear_grid(
        "profile-lnd-grid",
        grid_nx,
        grid_ny,
        0.0,
        360.0,
        -90.0,
        90.0,
    )
    ice_grid = make_rectilinear_grid(
        "profile-ice-grid",
        grid_nx,
        grid_ny,
        0.0,
        360.0,
        -90.0,
        90.0,
    )

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        log_level=log_level,
    )
    for component in (
        make_slab_atmosphere(atm_grid),
        make_slab_ocean(ocn_grid),
        make_slab_land(lnd_grid),
        make_slab_seaice(ice_grid),
    ):
        coupler.register(component)
    coupler.set_components_run_sequence(["OCN", "ATM", "LND", "ICE"])
    add_exchange_specs(
        coupler,
        (
            ExchangeSpec(
                source="OCN",
                destination="ATM",
                field_names=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="OCN",
                field_names=SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ATM",
                destination="LND",
                field_names=SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
                regridder_factory=conservative,
            ),
            ExchangeSpec(
                source="LND",
                destination="ATM",
                field_names=LAND_TO_ATMOSPHERE_SOIL_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="OCN",
                destination="ICE",
                field_names=OCEAN_TO_SEAICE_SURFACE_FIELDS,
                regridder_factory=bilinear,
            ),
            ExchangeSpec(
                source="ICE",
                destination="OCN",
                field_names=SEAICE_TO_OCEAN_FIELDS,
                regridder_factory=conservative,
            ),
        ),
    )
    coupler.initialize()
    return coupler


def profile_runtime(
    *,
    steps: int,
    grid_nx: int,
    grid_ny: int,
    log_level: int | str,
) -> RuntimeProfileResult:
    """Run a small timing profile for the scanned runtime."""

    coupler = build_slab_coupler(
        steps=steps,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        log_level=log_level,
    )
    runtime_state = coupler.create_runtime_state()

    start = time.perf_counter()
    final_state = _block_until_ready(coupler.run(runtime_state))
    run_seconds = time.perf_counter() - start

    return RuntimeProfileResult(
        run_seconds=run_seconds,
        final_state_leaves=len(jax.tree_util.tree_leaves(final_state)),
    )


def _format_result(result: RuntimeProfileResult) -> Sequence[str]:
    lines = [
        f"run_s={result.run_seconds:.6f}",
        f"final_state_leaves={result.final_state_leaves}",
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiling harness and print compact timing output."""

    args = build_parser().parse_args(argv)
    result = profile_runtime(
        steps=args.steps,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        log_level=args.log_level,
    )
    for line in _format_result(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
