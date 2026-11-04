"""Measure released Veros kernels and coupled execution in separate processes.

Run each backend in a fresh interpreter. Both cases use the global four-degree
ocean. The ocean-only case supplies zero total surface fluxes with synchronous
1800-second native/coupling clocks; the coupled case uses the global
Veris/Veros/JCM gallery with packaged JCM T31 inputs and no spinup. Native
physics, restoring, solver tolerances and sea-ice numerics retain their defaults.
Only the pinned differentiable fork supports scanned or forced-host JAX
components. Output-free compute timings exclude artifact writing.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Any, Literal

import jax
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Describe one native-backend/execution lane and its saved artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("ocean", "coupled"), default="ocean")
    parser.add_argument("--backend", choices=("numpy", "jax"), default="numpy")
    parser.add_argument(
        "--execution", choices=("host", "jax", "forced-host"), default="host"
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--output",
        action="store_true",
        help="Also verify a separate period/snapshot output run.",
    )
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Reject empty work and unsupported execution/backend combinations."""
    if args.steps < 1 or args.repeats < 1:
        raise ValueError("steps and repeats must be positive")
    if args.execution != "host" and args.backend != "jax":
        raise ValueError(
            "Scanned/forced-host JAX components require the JAX native backend"
        )


def _wait(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    jax.effects_barrier()
    return value


def measure_runs(
    run: Callable[[Any], Any],
    initial: Any,
    *,
    repeats: int,
    verify: Callable[[Any], None] | None = None,
) -> tuple[Any, float, list[float]]:
    """Measure first invocation and synchronized warm resets from one input."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    samples = []
    final = None
    for _ in range(repeats + 1):
        start = time.perf_counter()
        final = _wait(run(initial))
        samples.append(time.perf_counter() - start)
        if verify is not None:
            verify(final)
    return final, samples[0], samples[1:]


def _build_coupler(args: argparse.Namespace) -> Any:
    from vercor import Clock, Coupler, Exchange, RuntimeOptions
    from vercor.dtypes import DTypePolicy
    from vercor.output import OutputSpec, PeriodOutput
    from vercor.setups import (
        JCMAtmosphereConfig,
        JCMLandAtmosphereConfig,
        Spinup,
        VerosConfig,
        make_veros_gcm,
    )

    dtype = DTypePolicy(enable_x64=True)
    dtype._ensure_jax_capability()
    native_execution: Literal["host", "jax"] = (
        "host" if args.execution == "host" else "jax"
    )
    period = (
        OutputSpec(period=PeriodOutput(frequency="step"))
        if args.output
        else OutputSpec()
    )
    if args.case == "ocean":
        from vercor.components import DataComponent
        from vercor.regridding import bilinear

        ocean = make_veros_gcm(
            config=VerosConfig(
                execution=native_execution,
                backend=args.backend,
                output=period,
                surface_forcing="surface_fluxes",
            )
        )
        forcing = DataComponent(
            "FORCING",
            ocean.grid,
            fields={name: np.zeros(ocean.grid.shape) for name in ocean.spec.inputs},
        )
        return Coupler(
            Clock(datetime(2000, 1, 3), 1800.0, args.steps),
            components=(forcing, ocean),
            exchanges=(
                Exchange(
                    quantity="intensive",
                    source=forcing.name,
                    target=ocean.name,
                    fields=ocean.spec.inputs,
                    regridder_factory=bilinear,
                ),
            ),
            run_order=(forcing.name, ocean.name),
            log_level="ERROR",
            runtime=RuntimeOptions(
                dtype=dtype, backend="jax" if args.execution == "jax" else "host"
            ),
        )
    from vercor.setups.gallery.veris_veros_jcm import build_coupler

    coupler = build_coupler(
        clock=Clock(datetime(2000, 1, 3), 86400.0, args.steps, calendar="gregorian"),
        ocean_config=VerosConfig(
            execution=native_execution,
            backend=args.backend,
            surface_forcing="surface_fluxes",
            export_open_water_surface=True,
            output=period,
        ),
        jcm_config=JCMLandAtmosphereConfig(
            atmosphere=JCMAtmosphereConfig(
                seaice_coupling=True,
                spinup=Spinup(enabled=False),
                output=period,
            )
        ),
        log_level="ERROR",
        dtype=dtype,
    )
    # Reassemble before preparing either coupler; components and physical
    # resources are identical, and only the orchestration policy changes.
    return Coupler(
        coupler.clock,
        components=tuple(coupler.components.values()),
        exchanges=coupler.exchanges,
        run_order=coupler.run_order,
        log_level="ERROR",
        runtime=replace(
            coupler.runtime, backend="jax" if args.execution == "jax" else "host"
        ),
    )


def _arrays(state: Any) -> dict[str, np.ndarray[Any, Any]]:
    """Save full canonical fields and every native ocean variable for replay."""
    arrays = {
        f"canonical.{name}.{field}": np.asarray(value)
        for name, component in state.components().items()
        for field, value in component.fields().items()
    }
    native = state._component_state("OCN").payload
    arrays.update(
        {
            f"native.{name}": np.asarray(value)
            for name, value in native.variables.items()
        }
    )
    for name in ("temp", "salt", "u", "v", "w", "psi"):
        if not np.all(np.isfinite(arrays[f"native.{name}"])):
            raise ValueError(f"ERROR nonfinite native prognostic {name}")
    for name, component in state.components().items():
        mask = np.asarray(component.grid.binary_mask)
        for field, value in component.fields().items():
            value = np.asarray(value)
            if np.any(np.isinf(value)):
                raise ValueError(f"ERROR infinite canonical {name}.{field}")
            if value.shape[-2:] == mask.shape and not np.all(
                np.isfinite(value[..., mask > 0])
            ):
                raise ValueError(f"ERROR nonfinite active canonical {name}.{field}")
    return arrays


def _fingerprint(arrays: dict[str, np.ndarray[Any, Any]]) -> str:
    digest = hashlib.sha256()
    for name, array in sorted(arrays.items()):
        digest.update(f"{name}:{array.shape}:{array.dtype}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _provenance() -> dict[str, Any]:
    import veros

    packages = ("veros", "jax", "jaxlib", "numpy", "scipy", "jcm", "veris", "flax")
    direct_url = metadata.distribution("veros").read_text("direct_url.json")
    root = Path(__file__).resolve().parents[1]
    tracked = subprocess.check_output(
        ["rg", "--files", "vercor", "-g", "*.py"], cwd=root, text=True
    ).splitlines()
    source = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in tracked
        if name.endswith(".py")
    }
    return {
        "versions": {name: metadata.version(name) for name in packages},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "devices": [str(device) for device in jax.devices()],
        "veros_module": veros.__file__,
        "jax_module": jax.__file__,
        "veros_direct_url": json.loads(direct_url) if direct_url else None,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
            )
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": source,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    """Run one isolated lane and save timings, provenance and compared arrays."""
    args = build_parser().parse_args()
    validate_arguments(args)
    args.directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    coupler = _build_coupler(args)
    initial = _wait(coupler.initial_state())
    setup_seconds = time.perf_counter() - started
    before = _arrays(initial)
    fingerprint = _fingerprint(before)
    run = jax.jit(coupler.run) if args.execution == "jax" else coupler.run
    repeat_hashes: list[str] = []

    def verify_repeat(state: Any) -> None:
        repeated = _fingerprint(_arrays(state))
        if repeat_hashes and repeated != repeat_hashes[0]:
            raise AssertionError("ERROR benchmark repetitions changed the final state")
        repeat_hashes.append(repeated)

    final, cold, warm = measure_runs(
        run, initial, repeats=args.repeats, verify=verify_repeat
    )
    if _fingerprint(_arrays(initial)) != fingerprint:
        raise AssertionError("ERROR benchmark mutated the initial state")
    after = _arrays(final)
    np.savez_compressed(args.directory / "initial.npz", allow_pickle=False, **before)
    np.savez_compressed(args.directory / "final.npz", allow_pickle=False, **after)
    native = final._component_state("OCN").payload
    native_steps = int(native.variables.itt) - int(
        initial._component_state("OCN").payload.variables.itt
    )
    median = statistics.median(warm)
    from veros import runtime_settings
    from vercor.setups._external.veros_state import get_component_linear_solver

    solver = get_component_linear_solver(native)
    report = {
        "setup": "global_4deg",
        "case": args.case,
        "backend": args.backend,
        "execution": args.execution,
        "steps": args.steps,
        "repeats": args.repeats,
        "dtype": str(native.variables.temp.dtype),
        "ocean_shape": [native.settings.nx, native.settings.ny, native.settings.nz],
        "dt_mom": native.settings.dt_mom,
        "dt_tracer": native.settings.dt_tracer,
        "native_steps": native_steps,
        "coupling_dt_seconds": coupler.clock.dt_seconds,
        "native_solver": type(solver).__name__,
        "tdma": runtime_settings.use_special_tdma,
        "setup_seconds": setup_seconds,
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_median_seconds": median,
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
        "native_steps_per_second": native_steps / median,
        "simulated_days_per_second": args.steps
        * coupler.clock.dt_seconds
        / 86400
        / median,
        "initial_sha256": fingerprint,
        "final_sha256": _fingerprint(after),
        "repeat_final_sha256": repeat_hashes,
        "provenance": _provenance(),
    }
    (args.directory / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.output:
        from vercor.output import OutputTarget

        _wait(coupler.run(initial, output=OutputTarget(args.directory / "output")))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "setup",
                    "case",
                    "backend",
                    "execution",
                    "native_steps",
                    "setup_seconds",
                    "cold_seconds",
                    "warm_median_seconds",
                    "native_steps_per_second",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
