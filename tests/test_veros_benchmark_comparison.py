"""Keep performance comparison attribution and numerical error summaries honest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.fast_always


def test_numerical_summary_reports_error_without_moving_tolerances() -> None:
    """Catch dropping nonzero differences or comparing inactive NaNs as numbers."""
    from benchmarks.compare_veros import array_errors

    result = array_errors(
        {"temp": np.array([1.0, 2.0, np.nan])}, {"temp": np.array([1.0, 2.5, np.nan])}
    )
    assert result["arrays_compared"] == 1
    assert result["changed"]["temp"]["max_abs"] == 0.5
    assert result["changed"]["temp"]["scale_relative"] == 0.25


def test_numerical_summary_rejects_changed_missing_support() -> None:
    """A backend that changes the wet/dry NaN domain cannot pass comparison."""
    from benchmarks.compare_veros import array_errors

    with pytest.raises(ValueError, match="support"):
        array_errors({"temp": np.array([1.0, np.nan])}, {"temp": np.array([1.0, 0.0])})


def _lane(path: Path, samples: list[float], *, version: str = "1.7.0") -> None:
    path.mkdir()
    report = dict(
        setup="global_4deg",
        case="ocean",
        steps=4,
        dtype="float64",
        ocean_shape=[90, 40, 15],
        dt_mom=1800.0,
        dt_tracer=1800.0,
        coupling_dt_seconds=1800.0,
        native_steps=4,
        native_solver="SciPySolver",
        warm_seconds=samples,
        setup_seconds=1.0,
        cold_seconds=5.0,
        provenance={"versions": {"veros": version, "jax": "0.11.2"}},
    )
    (path / "result.json").write_text(json.dumps(report))
    for phase in ("initial", "final"):
        np.savez(path / f"{phase}.npz", temp=np.array([1.0, 2.0]))


def test_comparison_uses_warm_samples_and_reports_observed_bounds(
    tmp_path: Path,
) -> None:
    """Catch using cold compilation time as steady-state throughput."""
    from benchmarks.compare_veros import compare_lanes

    _lane(tmp_path / "numpy", [2.0, 3.0, 4.0])
    _lane(tmp_path / "jax", [1.0, 1.5, 2.0])
    result = compare_lanes(tmp_path / "numpy", tmp_path / "jax")
    assert result["warm_speedup"] == 2.0
    assert result["observed_speedup_bounds"] == [1.0, 4.0]
    assert result["final"]["arrays_compared"] == 1


def test_comparison_rejects_different_native_releases(tmp_path: Path) -> None:
    """Prevent attributing stock-versus-fork physics changes to execution alone."""
    from benchmarks.compare_veros import compare_lanes

    _lane(tmp_path / "stock", [2.0])
    _lane(tmp_path / "fork", [1.0], version="0+fork")
    with pytest.raises(ValueError, match="distribution"):
        compare_lanes(tmp_path / "stock", tmp_path / "fork")


def test_comparison_rejects_different_global_workloads(tmp_path: Path) -> None:
    """Global ocean-only and coupled timings represent different amounts of work."""
    from benchmarks.compare_veros import compare_lanes

    _lane(tmp_path / "ocean", [2.0])
    _lane(tmp_path / "coupled", [1.0])
    report = tmp_path / "coupled" / "result.json"
    metadata = json.loads(report.read_text())
    metadata["case"] = "coupled"
    report.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="case"):
        compare_lanes(tmp_path / "ocean", tmp_path / "coupled")
