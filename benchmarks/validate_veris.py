"""Measure prescribed global Veris budgets and timestep sensitivity.

Physics runs exclusively through the existing prescribed-data gallery and
Coupler.run. A pure output provider reduces interval rates on device; the
shared bounded period accumulator computes their time means. Actual checked
native-step counts come from the guarded runtime carry, not loop inference.
Inventories use native stored cell areas, not a second spherical quadrature.
This is numerical validation and does not establish observed climate skill.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from vercor import Clock, Coupler
from vercor.components import CallableComponent
from vercor.output import (
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)
from vercor.setups import VerisConfig
from vercor.setups._external.veris_runtime import VerisPayload
from vercor.state import RunState


@dataclass(frozen=True)
class Inventory:
    """Native-area ice/snow volume m3, solid mass/salt kg and enthalpy J."""

    ice_volume: float
    snow_volume: float
    mass: float
    salt: float
    enthalpy: float


@dataclass(frozen=True)
class IntegratedFluxes:
    """Actual ocean/external mass kg, salt kg and zero-layer enthalpy J inputs."""

    ocean_mass: float
    ocean_salt: float
    ocean_heat: float
    external_mass: float
    external_heat: float


@jax.jit
def _native_area(payload: VerisPayload) -> jax.Array:
    reciprocal = payload.state.recip_rA[2:-2, 2:-2].T
    wet = payload.state.iceMask[2:-2, 2:-2].T > 0
    return jnp.where(wet, 1 / jnp.where(reciprocal > 0, reciprocal, 1), 0.0)


@jax.jit
def _inventory_values(payload: VerisPayload) -> jax.Array:
    area = _native_area(payload)
    ice = jnp.sum(payload.state.hIceMean[2:-2, 2:-2].T * area)
    snow = jnp.sum(payload.state.hSnowMean[2:-2, 2:-2].T * area)
    p = payload.physical_constants
    mass = p.rhoIce * ice + p.rhoSnow * snow
    return jnp.stack(
        (ice, snow, mass, p.rhoIce * ice * p.saltIce_ref / 1000, -p.lhFusion * mass)
    )


def inventory(payload: VerisPayload) -> Inventory:
    """Transfer five reduced native inventory scalars to the report host."""
    return Inventory(*map(float, np.asarray(_inventory_values(payload))))


@jax.jit
def _rate_values(fields: dict[str, Any], payload: VerisPayload) -> jax.Array:
    area = _native_area(payload)
    values = (
        fields["sea_surface_mass_flux"],
        fields["sea_surface_salt_mass_flux"],
        fields["sea_surface_heat_flux"],
        payload.interval_budget.external_mass_flux,
        payload.interval_budget.external_heat_flux,
    )
    return jnp.stack(tuple(jnp.sum(value * area) for value in values))


class _BudgetProvider:
    """Pure reduced-rate output through the existing immutable accumulator."""

    def sample(self, context: OutputContext) -> OutputFrame:
        fields = {
            name: context.state.field(name)
            for name in (
                "sea_surface_mass_flux",
                "sea_surface_salt_mass_flux",
                "sea_surface_heat_flux",
            )
        }
        rates = _rate_values(fields, context.payload)
        names = tuple(IntegratedFluxes.__dataclass_fields__)
        return OutputFrame(
            {
                name: OutputVariable(
                    (), rates[index], {"units": "W" if "heat" in name else "kg s-1"}
                )
                for index, name in enumerate(names)
            },
            metadata={"area_convention": "native stored reciprocal area"},
        )


def budget_residuals(
    first: Inventory, last: Inventory, fluxes: IntegratedFluxes
) -> dict[str, float]:
    """Normalize independent mass, actual salt and modeled enthalpy closures."""
    terms = {
        "mass": (last.mass - first.mass, fluxes.ocean_mass, -fluxes.external_mass),
        "actual_salt": (last.salt - first.salt, fluxes.ocean_salt, 0.0),
        "enthalpy": (
            last.enthalpy - first.enthalpy,
            fluxes.ocean_heat,
            -fluxes.external_heat,
        ),
    }
    return {
        name: abs(sum(values)) / max(1.0, *(abs(value) for value in values))
        for name, values in terms.items()
    }


def require_budgets(residuals: dict[str, float], tolerance: float = 1e-10) -> None:
    """Reject a failed/nonfinite closure rather than rescale a physical flux."""
    for name, value in residuals.items():
        if not np.isfinite(value) or value > tolerance:
            raise ValueError(
                f"ERROR {name} budget residual {value:.9g} exceeds {tolerance:.3g}"
            )


def validate_coupler(
    coupler: Coupler, initial: RunState, *, output: Path, component: str = "CICE"
) -> tuple[RunState, dict[str, Any]]:
    """Run existing physics once and verify every native step and interval budget."""
    original = coupler.components[component]
    provider = _BudgetProvider()
    monitored = CallableComponent(
        original.name,
        original.grid,
        original.step,
        spec=replace(
            original.spec,
            output=OutputSpec(provider=provider, period=PeriodOutput(frequency="step")),
        ),
    )
    run = Coupler(
        coupler.clock,
        components=tuple(
            monitored if value.name == component else value
            for value in coupler.components.values()
        ),
        exchanges=coupler.exchanges,
        run_order=coupler.run_order,
        runtime=coupler.runtime,
        constants=coupler.constants,
        log_level="error",
    )
    before = initial._component_state(component).payload
    if not isinstance(before, VerisPayload):
        raise TypeError("validation requires the initialized native Veris payload")
    first = inventory(before)
    output.mkdir(parents=True, exist_ok=True)
    if tuple(output.glob(f"{component.lower()}.averages.*.nc")):
        raise ValueError(
            "validation output must not contain an earlier run's period files"
        )
    started = time.perf_counter()
    final = run.run(
        initial,
        output=OutputTarget(output, write_final_fields=False, write_snapshots=False),
    )
    after = final._component_state(component).payload
    if not isinstance(after, VerisPayload):
        raise TypeError("validation lost its native payload")
    elapsed = time.perf_counter() - started
    configuration = cast(
        Any, original.spec.lifecycle.setup
    ).__self__.resources.configuration
    native_seconds = configuration.deltatTherm
    expected = int(
        round(coupler.clock.steps * coupler.clock.dt_seconds / native_seconds)
    )
    checked = int(after.checked_native_steps) - int(before.checked_native_steps)
    if checked != expected:
        raise ValueError(
            f"ERROR native checked-step count {checked} differs from expected {expected}"
        )
    files = tuple(output.glob(f"{component.lower()}.averages.*.nc"))
    if len(files) != coupler.clock.steps:
        raise ValueError("validation requires one scalar sample per coupling interval")
    totals = {name: 0.0 for name in IntegratedFluxes.__dataclass_fields__}
    for path in sorted(files):
        with xr.open_dataset(path) as dataset:
            for name in totals:
                totals[name] += (
                    float(dataset[name].values.item()) * coupler.clock.dt_seconds
                )
        path.unlink()
    fluxes = IntegratedFluxes(**totals)
    last = inventory(after)
    residuals = budget_residuals(first, last, fluxes)
    require_budgets(residuals)
    area = np.asarray(_native_area(after))
    state = after.state
    report = {
        "checked_native_steps": checked,
        "expected_native_steps": expected,
        "all_native_steps_valid": True,
        "native_guard": "every native input/output leaf finite; hIce/hSnow>=0; "
        "Area in [0,1]; zero dry storage; physics_status==0",
        "dtype": str(state.Area.dtype),
        "native_seconds": native_seconds,
        "native_configuration": asdict(configuration),
        "physical_parameters": {
            name: float(getattr(after.physical_constants, name))
            for name in (
                "hIce_min",
                "saltIce_ref",
                "rhoFresh",
                "rhoIce",
                "rhoSnow",
                "lhFusion",
            )
        },
        "coupling_seconds": coupler.clock.dt_seconds,
        "coupling_steps": coupler.clock.steps,
        "elapsed_seconds": elapsed,
        "initial_inventory": asdict(first),
        "final_inventory": asdict(last),
        "integrated_fluxes": asdict(fluxes),
        "budget_residuals": residuals,
        "wet_area_m2": float(area.sum()),
        "area_sha256": hashlib.sha256(area.tobytes()).hexdigest(),
        "boundary_extrema": {
            name: [
                float(np.min(np.asarray(getattr(state, name)))),
                float(np.max(np.asarray(getattr(state, name)))),
            ]
            for name in ("Area", "hIceMean", "hSnowMean")
        },
        "comparison_vector": {
            "ice_volume": last.ice_volume,
            "snow_volume": last.snow_volume,
            **asdict(fluxes),
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return final, report


def segment_coupler(coupler: Coupler, *, start: int, steps: int) -> Coupler:
    """Reuse owners and the model-calendar timestamp at a coupling boundary."""
    if start < 0 or steps <= 0 or start + steps > coupler.clock.steps:
        raise ValueError("segment must lie inside the original clock")
    date = next(date for index, date, _ in coupler.clock.iter() if index == start)
    stamp = datetime(
        date.year,
        date.month,
        date.day,
        date.hour,
        date.minute,
        date.second,
        date.microsecond,
    )
    return Coupler(
        Clock(stamp, coupler.clock.dt_seconds, steps, calendar=coupler.clock.calendar),
        components=tuple(coupler.components.values()),
        exchanges=coupler.exchanges,
        run_order=coupler.run_order,
        runtime=coupler.runtime,
        constants=coupler.constants,
        log_level="error",
    )


def state_difference(expected: Any, actual: Any) -> float:
    """Compare array carries while retaining matching inactive NaN sentinels."""
    left, tree = jax.tree_util.tree_flatten(expected)
    right, other = jax.tree_util.tree_flatten(actual)
    if cast(Any, tree) != other:
        raise ValueError("ERROR segmented run changed state structure")
    maximum = 0.0
    for a, b in zip(left, right, strict=True):
        first, second = np.asarray(a), np.asarray(b)
        if np.any(np.isinf(first)) or np.any(np.isinf(second)):
            raise ValueError("ERROR segmented run contains infinite state")
        missing = np.isnan(first)
        if not np.array_equal(missing, np.isnan(second)):
            raise ValueError("ERROR segmented run changed missing-data support")
        first = np.where(missing, 0, first).astype(np.float64)
        second = np.where(missing, 0, second).astype(np.float64)
        scale = max(1.0, float(np.max(np.abs(first), initial=0)))
        maximum = max(maximum, float(np.max(np.abs(first - second), initial=0)) / scale)
    return maximum


def validate_segmented(
    coupler: Coupler,
    initial: RunState,
    final: RunState,
    *,
    segment_steps: int,
    output: Path,
) -> dict[str, Any]:
    """Check resumed immutable carries and interval budgets against one full run."""
    if segment_steps <= 0:
        raise ValueError("segment steps must be positive")
    current = initial
    reports = []
    for start in range(0, coupler.clock.steps, segment_steps):
        steps = min(segment_steps, coupler.clock.steps - start)
        segment = segment_coupler(coupler, start=start, steps=steps)
        current, report = validate_coupler(segment, current, output=output / str(start))
        reports.append(report)
    maximum = state_difference(final, current)
    if maximum > 1e-12:
        raise ValueError(f"ERROR segmented state mismatch {maximum:.9g}")
    return {
        "segments": len(reports),
        "maximum_scaled_state_difference": maximum,
        "checked_native_steps": sum(r["checked_native_steps"] for r in reports),
        "segment_budget_residuals": [r["budget_residuals"] for r in reports],
    }


def timestep_refinement(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare successive timestep vectors against the declared roundoff floor."""
    if len(reports) != 3:
        raise ValueError("refinement requires coarse, medium and fine reports")
    result = {}
    for name in reports[0]["comparison_vector"]:
        values = [float(report["comparison_vector"][name]) for report in reports]
        scale = max(1.0, *(abs(value) for value in values))
        floor = (
            64
            * np.finfo(np.dtype(reports[-1]["dtype"])).eps
            * max(report["checked_native_steps"] for report in reports)
            * scale
        )
        coarse, fine = abs(values[1] - values[0]), abs(values[2] - values[1])
        result[name] = {
            "coarse_medium": coarse,
            "medium_fine": fine,
            "roundoff_floor": floor,
            "resolved": bool(fine <= floor or fine < coarse),
        }
    return {
        "resolved": all(value["resolved"] for value in result.values()),
        "quantities": result,
    }


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    native = json.loads(
        importlib.metadata.distribution("veris").read_text("direct_url.json") or "{}"
    )
    return {
        "vercor_commit": git("rev-parse", "HEAD"),
        "vercor_dirty": bool(git("status", "--porcelain")),
        "native_provenance": native,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "devices": [str(device) for device in jax.devices()],
    }


def _forcing_provenance() -> dict[str, Any]:
    from vercor.setups._data.assets import get_forcing_data
    from veros.setups.global_4deg.global_4deg import DATA_FILES

    paths = {
        key: get_forcing_data(key)
        for key in ("era5_model_levels", "era5_surface", "era5_land_masked")
    }
    paths.update({f"veros_{key}": Path(DATA_FILES[key]) for key in DATA_FILES.keys()})
    result = {}
    for key, path in paths.items():
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        result[key] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


def polar_config(
    mask: Any,
    latitude: Any,
    *,
    model_seconds: int,
    evp_steps: int = 400,
    adaptive_evp: bool = False,
    minimum_ice_thickness: float = 1e-5,
) -> VerisConfig:
    """Initialize polar wet ice with 4 g/kg salt to exercise actual salt budgets."""
    if evp_steps <= 0:
        raise ValueError("EVP steps must be positive")
    if not np.isfinite(minimum_ice_thickness) or minimum_ice_thickness <= 0:
        raise ValueError("minimum ice thickness must be finite and positive")
    polar = np.asarray(mask) * (abs(np.asarray(latitude))[None, :] >= 60)
    return VerisConfig(
        model_timestep=timedelta(seconds=model_seconds),
        physical_constants={"saltIce_ref": 4.0, "hIce_min": minimum_ice_thickness},
        custom_parameters={"nEVPsteps": evp_steps, "useAdaptiveEVP": adaptive_evp},
        initial_conditions={
            "Area": 0.5 * polar,
            "hIceMean": 0.25 * polar,
            "TSurf": 260.0,
        },
    )


def _global_coupler(
    days: int,
    model_seconds: int,
    evp_steps: int = 400,
    adaptive_evp: bool = False,
    minimum_ice_thickness: float = 1e-5,
) -> Coupler:
    from vercor.setups._external.veros_data import load_veros_data_snapshot
    from vercor.setups._external.veris import make_veris
    from vercor.setups._data.veros_ocean import make_veros_data_ocean
    from vercor.setups._veris import VerisVerosDataSetup
    from vercor.setups.gallery.run_veris_dataveros_era5 import _make_coupler
    import veris

    snapshot = load_veros_data_snapshot()
    mask = np.asarray(snapshot.geometry["maskT"])[..., -1]
    latitude = np.asarray(snapshot.geometry["latitude"])
    config = polar_config(
        mask,
        latitude,
        model_seconds=model_seconds,
        evp_steps=evp_steps,
        adaptive_evp=adaptive_evp,
        minimum_ice_thickness=minimum_ice_thickness,
    )
    pair = VerisVerosDataSetup(
        make_veris(
            snapshot.grid, veris.OceanGeometry(**snapshot.geometry), config=config
        ),
        make_veros_data_ocean(snapshot),
    )
    return _make_coupler(
        loglevel="error",
        float_type="float64",
        steps=days,
        veris_config=config,
        pair=pair,
    )


def main() -> None:
    """Run explicit global acceptance; annual sweeps are never ordinary unit tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--model-seconds", type=int, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segment-days", type=int, default=0)
    parser.add_argument("--evp-steps", type=int, default=400)
    parser.add_argument("--adaptive-evp", action="store_true")
    parser.add_argument("--minimum-ice-thickness", type=float, default=1e-5)
    arguments = parser.parse_args()
    if (
        arguments.days <= 0
        or arguments.model_seconds <= 0
        or arguments.evp_steps <= 0
        or arguments.segment_days < 0
        or not np.isfinite(arguments.minimum_ice_thickness)
        or arguments.minimum_ice_thickness <= 0
        or 86400 % arguments.model_seconds
    ):
        parser.error(
            "days/EVP steps must be positive; segment days nonnegative; model seconds must divide one day"
        )
    provenance = _git_metadata()
    coupler = _global_coupler(
        arguments.days,
        arguments.model_seconds,
        arguments.evp_steps,
        arguments.adaptive_evp,
        arguments.minimum_ice_thickness,
    )
    initial = coupler.initial_state()
    final, report = validate_coupler(coupler, initial, output=arguments.output)
    if arguments.segment_days:
        report["segmented_equivalence"] = validate_segmented(
            coupler,
            initial,
            final,
            segment_steps=arguments.segment_days,
            output=arguments.output / "segments",
        )
    report.update(provenance)
    report["forcing_files"] = _forcing_provenance()
    report["initial_polar_ice"] = {
        "minimum_absolute_latitude_degrees": 60,
        "Area": 0.5,
        "hIceMean_m": 0.25,
        "TSurf_K": 260,
        "saltIce_ref_g_kg": 4.0,
    }
    report["forcing"] = (
        "real ERA5 monthly climatology and prescribed native Veros monthly SST/SSS; daily noleap coupling"
    )
    report["geometry"] = {
        "shape": list(coupler.components["CICE"].grid.shape),
        "radius_m": float(
            cast(
                Any, coupler.components["CICE"].spec.lifecycle.setup
            ).__self__.geometry.radius
        ),
        "area_convention": "authoritative native stored area; no remapping quadrature substitution",
    }
    report["limitations"] = (
        "No observed climate skill claim; actual salt inventory and "
        "fixed-reference Boussinesq virtual salt are distinct."
    )
    (arguments.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "checked_native_steps",
                    "elapsed_seconds",
                    "budget_residuals",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
