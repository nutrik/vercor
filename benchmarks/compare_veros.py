"""Compare synchronized, same-distribution Veros benchmark artifacts.

Error summaries report actual differences without choosing or relaxing physical
tolerances. Saved full native arrays are the reproducible numerical evidence.
Observed timing bounds describe the measured samples, not confidence intervals.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np


def array_errors(
    reference: Mapping[str, np.ndarray[Any, Any]],
    candidate: Mapping[str, np.ndarray[Any, Any]],
) -> dict[str, Any]:
    """Report changed arrays with absolute and field-scale-relative errors."""
    if set(reference) != set(candidate):
        raise ValueError("Array schemas differ")
    changed = {}
    for name, actual in reference.items():
        other = candidate[name]
        if actual.shape != other.shape or actual.dtype != other.dtype:
            raise ValueError(f"Array shape/dtype differs: {name}")
        if not np.array_equal(np.isnan(actual), np.isnan(other)):
            raise ValueError(f"Missing-data support differs: {name}")
        if np.any(np.isinf(actual)) or np.any(np.isinf(other)):
            raise ValueError(f"Infinite native/canonical values: {name}")
        finite = np.isfinite(actual)
        delta = np.abs(actual[finite].astype(float) - other[finite].astype(float))
        maximum = float(np.max(delta, initial=0.0))
        if maximum:
            scale = float(np.max(np.abs(actual[finite].astype(float)), initial=0.0))
            changed[name] = {
                "max_abs": maximum,
                "reference_scale": scale,
                "scale_relative": maximum / scale if scale else None,
            }
    return {
        "arrays_compared": len(reference),
        "arrays_identical": len(reference) - len(changed),
        "changed": changed,
    }


def compare_lanes(reference: Path, candidate: Path) -> dict[str, Any]:
    """Compare physically matched lanes from the same native model distribution."""
    baseline = json.loads((reference / "result.json").read_text())
    alternative = json.loads((candidate / "result.json").read_text())
    if baseline["provenance"]["versions"] != alternative["provenance"]["versions"]:
        raise ValueError(
            "Model/dependency distributions differ; execution attribution is invalid"
        )
    for name in (
        "setup",
        "steps",
        "dtype",
        "ocean_shape",
        "dt_mom",
        "dt_tracer",
        "coupling_dt_seconds",
        "native_steps",
        "native_solver",
    ):
        if baseline[name] != alternative[name]:
            raise ValueError(f"Physical benchmark configuration differs: {name}")
    first, second = baseline["warm_seconds"], alternative["warm_seconds"]
    first_median, second_median = statistics.median(first), statistics.median(second)
    saving = first_median - second_median
    startup = (
        alternative["setup_seconds"]
        + alternative["cold_seconds"]
        - baseline["setup_seconds"]
        - baseline["cold_seconds"]
    )
    blocks = math.ceil(1.0 + max(0.0, startup) / saving) if saving > 0 else None
    report = {
        "reference": str(reference),
        "candidate": str(candidate),
        "warm_speedup": first_median / second_median,
        "observed_speedup_bounds": [min(first) / max(second), max(first) / min(second)],
        "startup_difference_seconds": startup,
        "estimated_break_even_native_steps": (
            blocks * baseline["native_steps"] if blocks else None
        ),
        "estimated_break_even_simulated_days": (
            blocks * baseline["steps"] * baseline["coupling_dt_seconds"] / 86400.0
            if blocks
            else None
        ),
    }
    for phase in ("initial", "final"):
        with np.load(reference / f"{phase}.npz", allow_pickle=False) as actual:
            with np.load(candidate / f"{phase}.npz", allow_pickle=False) as other:
                report[phase] = array_errors(dict(actual), dict(other))
    return report


def main() -> None:
    """Write a reproducible JSON comparison without claiming cross-version gains."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(compare_lanes(args.reference, args.candidate), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
