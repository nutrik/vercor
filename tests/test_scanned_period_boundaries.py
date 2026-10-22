"""Acceptance of scanned period boundaries, independent windows, and cancellation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
import signal
from typing import Any, cast

import h5netcdf
import jax
from jax.errors import JaxRuntimeError
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from tests._coverage_support import make_test_grid
from tests._workflow_test_support import StaticWorkflow, make_component
from vercor import Clock, Coupler, RuntimeOptions
from vercor.calendar import DateTime360, ModelDateTime
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.output import OutputSpec, OutputTarget, PeriodOutput, SnapshotContext
from vercor.runtime import StepPlan
from vercor._runtime.interrupts import RuntimeInterruptController, RuntimeInterrupted


def _record_scan_lengths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe executed runtime scan lengths without replacing compiled work."""

    lengths: list[int] = []
    original_jit = jax.jit

    def recording_jit(function: Any, *args: Any, **kwargs: Any) -> Any:
        compiled = original_jit(function, *args, **kwargs)

        def invoke(*call_args: Any, **call_kwargs: Any) -> Any:
            if len(call_args) >= 3:
                indices = call_args[1]
                if (
                    getattr(indices, "ndim", None) == 1
                    and np.dtype(indices.dtype).kind in "iu"
                ):
                    lengths.append(indices.shape[0])
            return compiled(*call_args, **call_kwargs)

        return invoke

    monkeypatch.setattr(jax, "jit", recording_jit)
    return lengths


def _period_output(frequency: str = "day") -> OutputSpec:
    return OutputSpec(period=PeriodOutput(frequency=cast(Any, frequency)))


def _target(directory: Path) -> OutputTarget:
    return OutputTarget(directory, write_final_fields=False, write_snapshots=False)


def _read_mean(path: Path) -> np.ndarray:
    with h5netcdf.File(path, "r") as dataset:
        return cast(np.ndarray, np.asarray(dataset.variables["value"]))


@pytest.mark.parametrize(
    ("start", "calendar", "dates", "time_calendar"),
    [
        (
            datetime(2000, 2, 28),
            "gregorian",
            ("2000-02-28", "2000-02-29", "2000-03-01"),
            "proleptic_gregorian",
        ),
        (
            datetime(2001, 2, 27),
            "gregorian",
            ("2001-02-27", "2001-02-28", "2001-03-01"),
            "proleptic_gregorian",
        ),
        (
            datetime(2001, 2, 27),
            "noleap",
            ("2001-02-27", "2001-02-28", "2001-03-01"),
            "noleap",
        ),
        (
            DateTime360(2001, 2, 29, 0, 0, 0, 0),
            "360_day",
            ("2001-02-29", "2001-02-30", "2001-03-01"),
            "360_day",
        ),
    ],
)
def test_calendar_daily_resets_preserve_monthly_accumulation(
    start: datetime | ModelDateTime,
    calendar: Any,
    dates: tuple[str, ...],
    time_calendar: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Daily writes retain the monthly sums across the first daily boundary."""

    daily = make_component("daily", output=_period_output())
    monthly = make_component("monthly", output=_period_output("month"))
    coupler = Coupler(
        Clock(
            start=cast(datetime, start),
            dt_seconds=21_600.0,
            steps=12,
            calendar=calendar,
        ),
        components=(daily, monthly),
        run_order=("daily", "monthly"),
        log_level="WARNING",
    )
    initial = coupler.initial_state()
    lengths = _record_scan_lengths(monkeypatch)

    final = coupler.run(initial, output=_target(tmp_path))

    assert lengths == [4, 4, 4]
    paths = sorted(tmp_path.glob("daily.averages.*.nc"))
    assert [path.name for path in paths] == [
        f"daily.averages.{date}.nc" for date in dates
    ]
    for path, expected, date in zip(paths, (2.5, 6.5, 10.5), dates, strict=True):
        assert_allclose_compact(_read_mean(path), np.full((1, 2, 2), expected))
        with h5netcdf.File(path, "r") as dataset:
            time_attrs = dataset.variables["time"].attrs
            assert time_attrs["calendar"] == time_calendar
            assert time_attrs["isoformat"].startswith(f"{date}T00:00:00")
    monthly_paths = sorted(tmp_path.glob("monthly.averages.*.nc"))
    assert [path.name for path in monthly_paths] == [
        f"monthly.averages.{dates[0][:7]}.nc"
    ]
    assert_allclose_compact(_read_mean(monthly_paths[0]), np.full((1, 2, 2), 4.5))
    with h5netcdf.File(monthly_paths[0], "r") as dataset:
        assert (
            dataset.variables["time"]
            .attrs["isoformat"]
            .startswith(f"{dates[0]}T00:00:00")
        )
    assert_allclose_compact(final.component("monthly").field("value"), 12.0)


def test_schedule_splits_keep_window_samples_for_omitted_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schedule split neither loses samples nor advances an omitted component."""

    components = tuple(
        make_component(name, output=_period_output()) for name in ("A", "B", "omitted")
    )
    workflow = StaticWorkflow(
        tuple(
            StepPlan(step=step, components=(name,))
            for step, name in enumerate(("A", "A", "B", "B", "B", "B", "A", "A"))
        )
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=21_600.0, steps=8),
        components=components,
        run_order=("A", "B", "omitted"),
        runtime=RuntimeOptions(workflow=workflow),
        log_level="WARNING",
    )
    initial = coupler.initial_state().replace_fields(
        "omitted", {"value": jnp.full((2, 2), 9.0)}
    )
    lengths = _record_scan_lengths(monkeypatch)

    final = coupler.run(initial, output=_target(tmp_path))

    assert lengths == [2, 2, 2, 2]
    for name, means in (
        ("a", (1.75, 2.75)),
        ("b", (0.75, 3.75)),
        ("omitted", (9.0, 9.0)),
    ):
        paths = sorted(tmp_path.glob(f"{name}.averages.*.nc"))
        assert len(paths) == 2
        for path, mean in zip(paths, means, strict=True):
            assert_allclose_compact(_read_mean(path), np.full((1, 2, 2), mean))
    assert_allclose_compact(final.component("A").field("value"), 4.0)
    assert_allclose_compact(final.component("B").field("value"), 4.0)
    assert_allclose_compact(final.component("omitted").field("value"), 9.0)


def test_repeated_scanned_output_runs_start_with_empty_accumulators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reused coupler cannot leak sums, counts, or state between output runs."""

    component = make_component("model", output=_period_output())
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=21_600.0, steps=8),
        components=(component,),
        run_order=("model",),
        log_level="WARNING",
    )
    initial = coupler.initial_state()
    shifted = initial.replace_fields("model", {"value": jnp.full((2, 2), 100.0)})
    lengths = _record_scan_lengths(monkeypatch)

    first = coupler.run(initial, output=_target(tmp_path / "first"))
    second = coupler.run(shifted, output=_target(tmp_path / "second"))

    assert lengths == [4, 4, 4, 4]
    for directory, means in (("first", (2.5, 6.5)), ("second", (102.5, 106.5))):
        paths = sorted((tmp_path / directory).glob("model.averages.*.nc"))
        assert len(paths) == 2
        for path, mean in zip(paths, means, strict=True):
            assert_allclose_compact(_read_mean(path), np.full((1, 2, 2), mean))
    assert_allclose_compact(first.component("model").field("value"), 8.0)
    assert_allclose_compact(second.component("model").field("value"), 108.0)
    assert_allclose_compact(initial.component("model").field("value"), 0.0)
    assert_allclose_compact(shifted.component("model").field("value"), 100.0)


@pytest.mark.fast_always
def test_interrupt_inside_output_scan_aborts_writes_and_clears_run_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An execution-time interrupt stops a longer scan before host file output."""

    interrupts: RuntimeInterruptController | None = None
    requested = False
    observed: list[int] = []
    snapshots: list[Path] = []

    def request_at_second_step(step: Any) -> None:
        nonlocal requested
        observed.append(int(step))
        if not requested and int(step) == 1:
            requested = True
            assert interrupts is not None
            interrupts.request(signal.SIGINT)

    def step(fields: Mapping[str, Any], context: StepContext) -> Mapping[str, Any]:
        jax.debug.callback(request_at_second_step, context.step, ordered=True)
        return {"value": fields["value"] + 1.0}

    def snapshot(context: SnapshotContext) -> None:
        snapshots.append(context.output_path)

    component = CallableComponent(
        "model",
        make_test_grid(name="scanned-output-interrupt"),
        step,
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            output=OutputSpec(
                period=PeriodOutput(frequency="day"), snapshot_writer=snapshot
            ),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=21_600.0, steps=8),
        components=(component,),
        run_order=("model",),
        log_level="WARNING",
    )
    initial = coupler.initial_state()
    interrupts = coupler._ensure_prepared().interrupts
    handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    lengths = _record_scan_lengths(monkeypatch)

    with pytest.raises(RuntimeInterrupted, match="SIGINT") as caught:
        coupler.run(initial, output=OutputTarget(tmp_path))

    assert lengths == [4]
    assert observed == [0, 1]
    assert isinstance(caught.value.__cause__, JaxRuntimeError)
    assert interrupts.requested_signal is None
    assert snapshots == []
    assert not tuple(tmp_path.iterdir())
    for signum, handler in handlers.items():
        assert signal.getsignal(signum) is handler

    resumed = coupler.run(initial, output=_target(tmp_path / "resumed"))

    assert lengths == [4, 4, 4]
    assert_allclose_compact(resumed.component("model").field("value"), 8.0)
    paths = sorted((tmp_path / "resumed").glob("model.averages.*.nc"))
    assert len(paths) == 2
    for path, mean in zip(paths, (2.5, 6.5), strict=True):
        assert_allclose_compact(_read_mean(path), np.full((1, 2, 2), mean))
