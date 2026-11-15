"""Synchronized CPU runtime comparisons, runnable against an archived VerCOR tree.

Run with ``python -m benchmarks.benchmark_runtime --directory PATH``. Cold API
and retained-JIT costs are separate cases. Every invocation resets immutable
numeric inputs; output I/O remains in the timed API while artifact reads do not.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

import vercor
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import CallableComponent, ComponentMaskSpec, ComponentSpec
from vercor.dtypes import DTypePolicy
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.runtime import ExecutionChunk, ExecutionPlan, StepPlan, WorkflowContext

CASE_NAMES = (
    "plan",
    "finite",
    "finite_jit",
    "refresh",
    "refresh_jit",
    "host",
    "jax_api",
    "jax_jit",
    "irregular_host",
    "irregular_jax",
    "output_host",
    "output_jax",
    "remap",
    "vector",
    "slab_host",
    "slab_jax",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One resettable invocation with optional post-timing artifact collection."""

    name: str
    invoke: Callable[[], Any]
    metadata: dict[str, Any]
    batch_size: int = 1
    collect: Callable[[], dict[str, np.ndarray[Any, Any]]] | None = None


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    """Return validated options shared by independent comparison processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--case", choices=CASE_NAMES, action="append", dest="cases")
    for name, default in (
        ("repeats", 9),
        ("steps", 6),
        ("grid-nx", 32),
        ("grid-ny", 16),
    ):
        parser.add_argument(f"--{name}", type=_positive, default=default)
    parser.add_argument("--trace", action="store_true")
    return parser


def _synchronize(value: Any) -> Any:
    jax.block_until_ready(value)
    jax.effects_barrier()
    return value


def _numeric_leaves(value: Any) -> dict[str, np.ndarray[Any, Any]]:
    result = {}
    for index, leaf in enumerate(jax.tree_util.tree_leaves(value)):
        array = np.asarray(leaf)
        if array.dtype.kind in "biufc":
            result[f"leaf_{index:04d}"] = array
    return result


def _capture_trace(case: BenchmarkCase, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with jax.profiler.trace(str(directory), create_perfetto_trace=True):
        with jax.profiler.TraceAnnotation(case.name):
            _synchronize(case.invoke())


def measure_case(
    case: BenchmarkCase, *, repeats: int, trace_directory: Path | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray[Any, Any]]]:
    """Measure cold and warm resets, waiting for arrays and ordered callbacks.

    Cold calls execute once; warm compiled controls batch ten resets and report
    per-invocation seconds. Device transfer, NetCDF reads and trace capture occur
    after the timing boundary. Every sample's numeric leaves are retained.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    samples = []
    outputs = {}
    for index in range(repeats + 1):
        jax.effects_barrier()
        batch_size = 1 if index == 0 else case.batch_size
        started = time.perf_counter()
        for _ in range(batch_size):
            value = _synchronize(case.invoke())
        samples.append((time.perf_counter() - started) / batch_size)
        label = "cold" if index == 0 else f"warm_{index:02d}"
        for key, array in _numeric_leaves(value).items():
            outputs[f"{label}/{key}"] = array
        if case.collect is not None:
            for key, array in case.collect().items():
                outputs[f"{label}/artifact/{key}"] = array
    record = {
        "cold_seconds": samples[0],
        "warm_seconds": samples[1:],
        "median_seconds": statistics.median(samples[1:]),
        "batch_size": case.batch_size,
        "cold_batch_size": 1,
        "metadata": case.metadata,
    }
    if trace_directory is not None:
        _capture_trace(case, trace_directory)
    return record, outputs


def _grid(name: str, nx: int, ny: int) -> RectilinearGrid:
    # Explicit edges avoid inferred polar and one-cell geometric degeneracies.
    lon = np.linspace(0.0, 360.0, nx + 1)
    lat = np.linspace(-90.0, 90.0, ny + 1)
    return RectilinearGrid(
        name,
        longitude=(lon[:-1] + lon[1:]) / 2,
        latitude=(lat[:-1] + lat[1:]) / 2,
        longitude_edges=lon,
        latitude_edges=lat,
    )


def _dynamic_coupler(args: argparse.Namespace, backend: str) -> Coupler:
    grid = _grid("dynamic", args.grid_nx, args.grid_ny)
    dtype = DTypePolicy(enable_x64=args.dtype == "float64")
    pattern = jnp.arange(args.grid_nx * args.grid_ny, dtype=dtype.jax_real).reshape(
        grid.shape
    )
    area = jnp.where(pattern % 5 == 0, 0.0, 0.25 + 0.5 * (pattern % 3) / 3)

    def source(fields: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "area": jnp.mod(fields["area"] + 0.125, 1.0),
            "flux": fields["flux"] * 0.99 + 0.05,
            "binary_flux": fields["binary_flux"] * 0.99 + 0.05,
        }

    components = [
        CallableComponent(
            "SRC",
            grid,
            source,
            spec=ComponentSpec(
                outputs=("area", "flux", "binary_flux"),
                initial_fields={
                    "area": area,
                    "flux": 2.0 + pattern / 100,
                    "binary_flux": 3.0 + pattern / 100,
                },
                masks=ComponentMaskSpec(fractional_field="area"),
            ),
        )
    ]
    routes = []
    for name, quantity in (
        ("I", "intensive"),
        ("S", "source_flux"),
        ("T", "target_flux"),
    ):
        components.append(
            CallableComponent(
                name,
                grid,
                lambda fields: {"total": fields["total"] + fields["flux"]},
                spec=ComponentSpec(
                    inputs=("flux", "binary_flux"),
                    outputs=("area", "total"),
                    initial_fields={"area": 0.75, "total": 0.0, "binary_flux": 0.0},
                    masks=ComponentMaskSpec(fractional_field="area"),
                ),
            )
        )
        routes.append(Exchange("SRC", name, ("flux",), quantity=cast(Any, quantity)))
    routes.append(
        Exchange(
            "SRC",
            "I",
            ("binary_flux",),
            quantity="source_flux",
            source_coverage="binary",
            route_id="binary",
        )
    )
    return Coupler(
        Clock(datetime(2000, 1, 1), 3600.0, args.steps),
        components=components,
        exchanges=routes,
        run_order=("SRC", "I", "S", "T"),
        runtime=RuntimeOptions(dtype=dtype, backend=cast(Any, backend)),
        log_level="ERROR",
    )


@dataclass(frozen=True)
class _IrregularWorkflow:
    def build(self, context: WorkflowContext) -> ExecutionPlan:
        schedules = (("A", "B"), ("A", "B"), ("A",), ("B",), ("B",), ())
        return ExecutionPlan(
            tuple(
                StepPlan(i, schedules[i % len(schedules)])
                for i in range(context.clock.steps)
            )
        )


def _increment_coupler(
    args: argparse.Namespace, backend: str, *, output: bool
) -> Coupler:
    grid = _grid("increment", args.grid_nx, args.grid_ny)
    declaration = (
        OutputSpec(period=PeriodOutput(frequency="day", variables=("value",)))
        if output
        else OutputSpec()
    )
    components = tuple(
        CallableComponent(
            name,
            grid,
            lambda fields: {"value": fields["value"] + 1.0},
            spec=ComponentSpec(
                inputs=("value",),
                outputs=("value",),
                initial_fields={"value": 0.0},
                output=declaration,
            ),
        )
        for name in ("A", "B")
    )
    runtime = RuntimeOptions(
        dtype=DTypePolicy(enable_x64=args.dtype == "float64"),
        backend=cast(Any, backend),
    )
    if not output:
        runtime = replace(runtime, workflow=_IrregularWorkflow())
    return Coupler(
        Clock(
            datetime(2000, 1, 1), 3600.0, max(24, args.steps) if output else args.steps
        ),
        components=components,
        run_order=("A", "B"),
        runtime=runtime,
        log_level="ERROR",
    )


def _output_case(
    name: str, args: argparse.Namespace, coupler: Coupler
) -> BenchmarkCase:
    seed = _synchronize(coupler.initial_state())
    root = args.directory / name / "invocations"
    root.mkdir(parents=True, exist_ok=False)
    index = 0
    current = root
    info: dict[str, Any] = {
        "steps": coupler.clock.steps,
        "invocation": "api",
        "artifacts": [],
    }

    def invoke() -> Any:
        nonlocal index, current
        current = root / f"{index:04d}"
        index += 1
        return coupler.run(
            seed,
            output=OutputTarget(
                current, write_final_fields=False, write_snapshots=False
            ),
        )

    def collect() -> dict[str, np.ndarray[Any, Any]]:
        import xarray as xr

        arrays = {}
        descriptions = {}
        for path in sorted(current.rglob("*.nc")):
            relative = str(path.relative_to(current))
            with xr.open_dataset(
                path, engine="h5netcdf", decode_times=False
            ) as dataset:
                descriptions[relative] = {
                    "attrs": dict(dataset.attrs),
                    "variables": {
                        key: {
                            "dims": list(variable.dims),
                            "attrs": dict(variable.attrs),
                        }
                        for key, variable in dataset.variables.items()
                    },
                }
                for key, variable in dataset.variables.items():
                    array = np.asarray(variable.values)
                    if array.dtype.kind in "biufc":
                        arrays[f"{relative}/{key}"] = array
        if not arrays:
            raise RuntimeError("output case wrote no numeric NetCDF variables")
        info["artifacts"].append(descriptions)
        return arrays

    return BenchmarkCase(name, invoke, info, collect=collect)


def _finite_case(name: str, args: argparse.Namespace) -> BenchmarkCase:
    from vercor._numerical_safety import require_active_finite

    shape = (2, args.grid_ny, args.grid_nx)
    seed = _synchronize(
        jnp.arange(np.prod(shape), dtype=jnp.dtype(args.dtype)).reshape(shape) / 100
    )

    def check(values: Any) -> Any:
        arrays: tuple[Any, ...] = ()
        for i in range(12):
            active = jnp.where((values[0] + i / 100) % 0.07 < 0.03, 0.0, 0.25 + i / 24)
            field = jnp.where(active[None] > 0, values + i, jnp.nan)
            require_active_finite(
                field, active_mask=active, owner=f"benchmark field {i}"
            )
            arrays += (field, active)
        return arrays

    callback = jax.jit(check) if name.endswith("jit") else check
    return BenchmarkCase(
        name,
        lambda: callback(seed),
        {
            "arrays": 12,
            "trailing_mask": True,
            "invocation": "outer-jit" if name.endswith("jit") else "api",
        },
        batch_size=10 if name.endswith("jit") else 1,
    )


def _plan_case() -> BenchmarkCase:
    from vercor._runtime.backends import _RuntimeDriverAdapter

    plans = tuple(StepPlan(i, ("A",)) for i in range(4096))
    chunk = ExecutionChunk(plans)

    def consume() -> Any:
        # No physics/context use in _expected_plan: isolate only identity checks.
        driver = _RuntimeDriverAdapter(cast(Any, None), chunk, cast(Any, None))
        for plan in plans:
            driver._expected_plan(plan)
            driver._cursor += 1
        return driver._cursor

    return BenchmarkCase("plan", consume, {"plans": 4096, "microbenchmark": True})


def _regrid_case(name: str, args: argparse.Namespace) -> BenchmarkCase:
    source = _grid("source", max(2, args.grid_nx), max(2, args.grid_ny))
    target = _grid("target", max(2, args.grid_nx) + 1, max(2, args.grid_ny) + 1)
    x, y = jnp.meshgrid(jnp.deg2rad(source.longitude), jnp.deg2rad(source.latitude))
    seed = _synchronize(jnp.sin(x) * jnp.cos(y) + 2)
    if name == "vector":
        from vercor.regridding import bilinear, VectorRegridder

        remapper = cast(VectorRegridder, bilinear(source, target))
        _synchronize((source, target, cast(Any, remapper).interpolator))
        v = _synchronize(jnp.cos(x) - jnp.sin(y))
        return BenchmarkCase(
            name,
            lambda: remapper.regrid_vector(seed, v),
            {"target_shape": target.shape},
        )
    from vercor._interpolators.conservative_remap_rectilinear import (
        ConservativeRectilinearRemapper,
    )

    remappers = tuple(
        ConservativeRectilinearRemapper(
            cast(Any, source.longitude_edges),
            cast(Any, source.latitude_edges),
            cast(Any, target.longitude_edges),
            cast(Any, target.latitude_edges),
            normalize=normalization,
        )
        for normalization in ("conservation", "fracarea")
    )
    _synchronize((source, target, remappers))
    missing = _synchronize(seed.at[::5, ::7].set(jnp.nan))
    return BenchmarkCase(
        name,
        lambda: tuple(remapper.apply_scalar(missing) for remapper in remappers),
        {"normalization": ["conservation", "fracarea"], "target_shape": target.shape},
    )


def build_case(name: str, args: argparse.Namespace) -> BenchmarkCase:
    """Construct exactly one requested case, leaving unrelated kernels untouched."""
    if name == "plan":
        return _plan_case()
    if name in ("finite", "finite_jit"):
        return _finite_case(name, args)
    if name in ("remap", "vector"):
        return _regrid_case(name, args)
    backend = "host" if name.endswith("host") or name == "host" else "jax"
    if name.startswith("output") or name.startswith("irregular"):
        coupler = _increment_coupler(args, backend, output=name.startswith("output"))
    elif name.startswith("slab"):
        from vercor.setups.gallery.profile_runtime import build_slab_coupler

        original = build_slab_coupler(
            steps=args.steps,
            grid_nx=args.grid_nx,
            grid_ny=args.grid_ny,
            log_level="ERROR",
            dtype=DTypePolicy(enable_x64=args.dtype == "float64"),
        )
        coupler = Coupler(
            original.clock,
            components=original.components.values(),
            exchanges=original.exchanges,
            run_order=original.run_order,
            runtime=replace(original.runtime, backend=cast(Any, backend)),
            constants=original.constants,
            log_level="ERROR",
        )
    elif name in ("refresh", "refresh_jit", "host", "jax_api", "jax_jit"):
        coupler = _dynamic_coupler(args, backend)
    else:
        raise ValueError(f"unknown case: {name}")
    if name.startswith("output"):
        return _output_case(name, args, coupler)
    seed = _synchronize(coupler.initial_state())
    if name.startswith("refresh"):
        from vercor._runtime.exchange_masks import refresh_runtime_masks

        context = coupler._ensure_prepared().dispatch_context

        def run(state: Any) -> Any:
            for _ in range(args.steps):
                area = state.component("SRC").field("area")
                state = state.replace_fields(
                    "SRC", {"area": jnp.mod(area + 0.125, 1.0)}
                )
                state = refresh_runtime_masks(state, context=context)
            return state

    else:
        run = coupler.run
    outer_jit = name in ("refresh_jit", "jax_jit", "slab_jax")
    callback = jax.jit(run) if outer_jit else run
    return BenchmarkCase(
        name,
        lambda: callback(seed),
        {
            "steps": coupler.clock.steps,
            "backend": backend,
            "invocation": "outer-jit" if outer_jit else "api",
        },
        batch_size=10 if name == "refresh_jit" else 1,
    )


def _source_identity() -> dict[str, Any]:
    root = Path(vercor.__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for path in sorted((root / "vercor").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    revision = None
    dirty = None
    if (root / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain", "--", "vercor"],
                text=True,
            ).strip()
        )
    return {
        "root": str(root),
        "revision": revision,
        "sha256": digest.hexdigest(),
        "dirty_source": dirty,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _environment() -> dict[str, Any]:
    versions: dict[str, str | None] = {}
    for package in (
        "jax",
        "jaxlib",
        "numpy",
        "ml_dtypes",
        "xarray",
        "h5netcdf",
        "vercor",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        **versions,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "devices": [str(device) for device in jax.devices()],
        "jax_backend": jax.default_backend(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write timings, provenance and every sampled numeric result to a directory."""
    args = build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    args.directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "dtype": args.dtype,
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "source": _source_identity(),
        "environment": _environment(),
        "cases": {},
    }
    arrays = {}
    for name in dict.fromkeys(args.cases or CASE_NAMES):
        started = time.perf_counter()
        case = build_case(name, args)
        jax.effects_barrier()
        setup_seconds = time.perf_counter() - started
        record, leaves = measure_case(
            case,
            repeats=args.repeats,
            trace_directory=args.directory / "traces" / name if args.trace else None,
        )
        record["setup_seconds"] = setup_seconds
        result["cases"][name] = record
        arrays.update({f"{name}/{key}": value for key, value in leaves.items()})
    cast(Any, np.savez)(args.directory / "outputs.npz", **arrays)
    (args.directory / "results.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
