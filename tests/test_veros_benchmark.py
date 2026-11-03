"""Protect synchronization and honest steady-state sampling in the benchmark."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.fast_always


def test_measure_run_includes_device_wait_and_resets_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch asynchronous dispatch timing or advancing a previous replicate."""
    from benchmarks import benchmark_veros as benchmark

    elapsed = [0.0]
    events: list[str] = []
    initial = object()

    class Result:
        def block_until_ready(self) -> None:
            events.append("ready")
            elapsed[0] += 2.0

    def run(state: Any) -> Result:
        assert state is initial
        events.append("run")
        elapsed[0] += 1.0
        return Result()

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: elapsed[0])
    result, cold, warm = benchmark.measure_runs(run, initial, repeats=3)
    assert isinstance(result, Result)
    assert cold == 3.0
    assert warm == [3.0, 3.0, 3.0]
    assert events == ["run", "ready"] * 4


def test_measure_run_rejects_empty_warm_sample() -> None:
    """A benchmark without warm samples must not report steady-state speed."""
    from benchmarks.benchmark_veros import measure_runs

    with pytest.raises(ValueError, match="repeats"):
        measure_runs(lambda state: state, None, repeats=0)


def test_repetition_verification_is_excluded_from_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep reproducibility hashing outside the synchronized measured interval."""
    from benchmarks import benchmark_veros as benchmark

    elapsed = [0.0]
    verified: list[int] = []

    def run(state: int) -> int:
        elapsed[0] += 2.0
        return state + 1

    def verify(state: int) -> None:
        verified.append(state)
        elapsed[0] += 100.0

    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: elapsed[0])
    final, cold, warm = benchmark.measure_runs(run, 0, repeats=2, verify=verify)
    assert (final, cold, warm) == (1, 2.0, [2.0, 2.0])
    assert verified == [1, 1, 1]


@pytest.mark.parametrize(
    "arguments",
    (("--steps", "0"), ("--execution", "jax", "--backend", "numpy")),
)
def test_invalid_benchmark_configuration_is_rejected(
    arguments: tuple[str, ...],
) -> None:
    """Catch timing an empty run or mislabelling NumPy as scanned JAX."""
    from benchmarks.benchmark_veros import build_parser, validate_arguments

    arguments = (*arguments, "--directory", "/private/tmp/vercor-benchmark-test")
    with pytest.raises(ValueError):
        validate_arguments(build_parser().parse_args(arguments))
