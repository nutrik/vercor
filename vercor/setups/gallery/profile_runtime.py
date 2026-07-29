from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Any, Sequence

import jax

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import jax_ones
from vercor import RectilinearGrid
from vercor.regridding import bilinear, conservative
from vercor.setups import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.recipes import (
    LAND_TO_ATMOSPHERE_SOIL_FIELDS,
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    SEAICE_TO_OCEAN_FIELDS,
    SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
    SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
)
from vercor.topology import SurfaceMaskPolicy


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


def _block_until_ready(value: Any) -> Any:
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

    atm_grid = RectilinearGrid.uniform(
        "profile-atm-grid",
        nlon=grid_nx,
        nlat=grid_ny,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
    )
    ocn_mask = jax_ones((grid_ny, grid_nx)).at[:2, :].set(0.0)
    ocn_grid = RectilinearGrid.uniform(
        "profile-ocn-grid",
        nlon=grid_nx,
        nlat=grid_ny,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
        binary_mask=ocn_mask,
    )
    lnd_grid = RectilinearGrid.uniform(
        "profile-lnd-grid",
        nlon=grid_nx,
        nlat=grid_ny,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
    )
    ice_grid = RectilinearGrid.uniform(
        "profile-ice-grid",
        nlon=grid_nx,
        nlat=grid_ny,
        longitude=(0.0, 360.0),
        latitude=(-90.0, 90.0),
    )

    exchanges = (
        Exchange(
            source="OCN",
            target="ATM",
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=SLAB_ATMOSPHERE_TO_OCEAN_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS,
            regridder_factory=conservative,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=LAND_TO_ATMOSPHERE_SOIL_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ICE",
            fields=OCEAN_TO_SEAICE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ICE",
            target="OCN",
            fields=SEAICE_TO_OCEAN_FIELDS,
            regridder_factory=conservative,
        ),
    )
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=(
            make_slab_atmosphere(atm_grid),
            make_slab_ocean(ocn_grid),
            make_slab_land(lnd_grid),
            make_slab_seaice(ice_grid),
        ),
        exchanges=exchanges,
        run_order=("OCN", "ATM", "LND", "ICE"),
        runtime=RuntimeOptions(topology=SurfaceMaskPolicy()),
        log_level=log_level,
    )


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
    runtime_state = coupler.initial_state()

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
