from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import statistics
import time
from typing import Any, Literal, Sequence

import jax

from vercor import (
    Clock,
    Coupler,
    Exchange,
    RuntimeOptions,
)
from vercor.dtypes import DTypePolicy, jax_ones
from vercor import RectilinearGrid
from vercor.regridding import bilinear, conservative
from vercor.setups import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.recipes import (
    OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
    OCEAN_TO_SEAICE_SURFACE_FIELDS,
    ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS,
    ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS,
)
from vercor.topology import SurfaceMaskPolicy


@dataclass(frozen=True)
class RuntimeProfileResult:
    """Synchronized setup, first invocation and repeated-run timings."""

    run_seconds: float
    final_state_leaves: int
    setup_seconds: float
    warm_run_seconds: tuple[float, ...]
    backend: Literal["host", "jax"] = "jax"
    invocation: Literal["api", "outer-jit"] = "outer-jit"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for the runtime profiling harness."""

    parser = argparse.ArgumentParser(
        description="Profile VerCOR's host or JAX runtime on a synthetic slab coupler."
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
    parser.add_argument(
        "--repeats", type=int, default=5, help="Warm timed invocations."
    )
    parser.add_argument(
        "--float-type", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument(
        "--trace-dir", type=Path, help="Export a separate warm Perfetto/XProf trace."
    )
    parser.add_argument("--backend", choices=("host", "jax"), default="jax")
    parser.add_argument(
        "--invocation",
        choices=("api", "outer-jit"),
        help="Measure ordinary calls or a retained outer JIT; defaults to API for host and outer JIT for JAX.",
    )
    parser.add_argument("--json", action="store_true", help="Print timings as JSON.")
    return parser


def _block_until_ready(value: Any) -> Any:
    jax.block_until_ready(value)
    jax.effects_barrier()
    return value


def build_slab_coupler(
    *,
    steps: int,
    grid_nx: int,
    grid_ny: int,
    log_level: int | str,
    dtype: DTypePolicy | None = None,
    backend: Literal["host", "jax"] = "jax",
) -> Coupler:
    """Build a small slab coupler with an explicit runtime execution lane."""

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

    atm = make_slab_atmosphere(atm_grid)
    ocn = make_slab_ocean(ocn_grid)
    lnd = make_slab_land(lnd_grid)
    ice = make_slab_seaice(ice_grid)

    exchanges = (
        Exchange(
            quantity="intensive",
            source=ocn.name,
            target=atm.name,
            fields=OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source=atm.name,
            target=ocn.name,
            fields=ATMOSPHERE_TO_OCEAN_TURBULENT_HEAT_FLUX_FIELDS,
            regridder_factory=bilinear,
        ),
        Exchange(
            quantity="intensive",
            source=atm.name,
            target=lnd.name,
            fields=ATMOSPHERE_TO_LAND_LATENT_HEAT_FLUX_FIELDS,
            regridder_factory=conservative,
        ),
        Exchange(
            quantity="intensive",
            source=ocn.name,
            target=ice.name,
            fields=OCEAN_TO_SEAICE_SURFACE_FIELDS,
            regridder_factory=bilinear,
        ),
    )
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps),
        components=(atm, ocn, lnd, ice),
        exchanges=exchanges,
        run_order=(ocn.name, atm.name, lnd.name, ice.name),
        runtime=RuntimeOptions(
            dtype=DTypePolicy() if dtype is None else dtype,
            backend=backend,
            topology=SurfaceMaskPolicy(
                atmosphere=atm.name, ocean=ocn.name, land=lnd.name
            ),
        ),
        log_level=log_level,
    )


def profile_runtime(
    *,
    steps: int,
    grid_nx: int,
    grid_ny: int,
    log_level: int | str,
    dtype: DTypePolicy | None = None,
    repeats: int = 5,
    trace_dir: Path | None = None,
    backend: Literal["host", "jax"] = "jax",
    invocation: Literal["api", "outer-jit"] | None = None,
) -> RuntimeProfileResult:
    """Time setup and cold/warm execution; capture a separate optional trace.

    API invocations include Python orchestration and run-local executor setup.
    An outer JIT measures a retained compiled output-free run instead; this is
    the backward-compatible default for JAX. Host always uses ordinary calls.
    Each invocation starts from the same immutable initial state. Timings wait
    for arrays and callbacks; separate trace capture is never timed.
    """

    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if backend not in ("host", "jax"):
        raise ValueError("backend must be host or jax")
    mode = (
        invocation
        if invocation is not None
        else ("api" if backend == "host" else "outer-jit")
    )
    if mode not in ("api", "outer-jit"):
        raise ValueError("invocation must be api or outer-jit")
    if mode == "outer-jit" and backend != "jax":
        raise ValueError("outer-jit invocation requires the jax backend")
    setup_start = time.perf_counter()
    coupler = build_slab_coupler(
        steps=steps,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        log_level=log_level,
        dtype=dtype,
        backend=backend,
    )
    runtime_state = _block_until_ready(coupler.initial_state())
    setup_seconds = time.perf_counter() - setup_start
    run = jax.jit(coupler.run) if mode == "outer-jit" else coupler.run

    start = time.perf_counter()
    final_state = _block_until_ready(run(runtime_state))
    run_seconds = time.perf_counter() - start

    warm_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        final_state = _block_until_ready(run(runtime_state))
        warm_times.append(time.perf_counter() - start)
    if trace_dir is not None:
        with jax.profiler.trace(trace_dir, create_perfetto_trace=True):
            with jax.profiler.TraceAnnotation("vercor_warm_run"):
                _block_until_ready(run(runtime_state))

    return RuntimeProfileResult(
        run_seconds=run_seconds,
        final_state_leaves=len(jax.tree_util.tree_leaves(final_state)),
        setup_seconds=setup_seconds,
        warm_run_seconds=tuple(warm_times),
        backend=backend,
        invocation=mode,
    )


def _format_result(result: RuntimeProfileResult) -> Sequence[str]:
    lines = [
        f"backend={result.backend}",
        f"invocation={result.invocation}",
        f"setup_s={result.setup_seconds:.6f}",
        f"cold_run_s={result.run_seconds:.6f}",
        f"warm_median_s={statistics.median(result.warm_run_seconds):.6f}",
        f"warm_min_s={min(result.warm_run_seconds):.6f}",
        f"warm_max_s={max(result.warm_run_seconds):.6f}",
        f"warm_repeats={len(result.warm_run_seconds)}",
        f"final_state_leaves={result.final_state_leaves}",
    ]
    return lines


def run_setup(*, loglevel: str, float_type: str) -> int:
    """Run the default profile through the shared CLI contract."""

    result = profile_runtime(
        steps=24,
        grid_nx=32,
        grid_ny=16,
        log_level=loglevel,
        dtype=DTypePolicy(enable_x64=float_type == "float64"),
    )
    for line in _format_result(result):
        print(line)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiling harness and print compact timing output."""

    args = build_parser().parse_args(argv)
    result = profile_runtime(
        steps=args.steps,
        grid_nx=args.grid_nx,
        grid_ny=args.grid_ny,
        log_level=args.log_level,
        dtype=DTypePolicy(enable_x64=args.float_type == "float64"),
        repeats=args.repeats,
        trace_dir=args.trace_dir,
        backend=args.backend,
        invocation=args.invocation,
    )
    if args.json:
        print(json.dumps(asdict(result)))
        return 0
    for line in _format_result(result):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
