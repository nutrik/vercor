"""Exercise benchmark synchronization, reset ownership and real runtime lanes."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.fast_always


def _module() -> Any:
    return importlib.import_module("benchmarks.benchmark_runtime")


@pytest.mark.parametrize("option", ["--repeats", "--steps", "--grid-nx", "--grid-ny"])
def test_parser_rejects_nonpositive(option: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _module().build_parser().parse_args(["--directory", str(tmp_path), option, "0"])


def test_measure_resets_seed_and_excludes_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _module()
    events: list[str] = []
    seed = jnp.asarray(3.0)

    def invoke() -> Any:
        events.append("invoke")
        return seed + 1

    def synchronize(value: Any) -> Any:
        events.append("sync")
        return value

    def trace(case: Any, directory: Path) -> None:
        events.append("trace")
        assert events.count("clock") == 6

    def now() -> float:
        events.append("clock")
        return float(len(events))

    monkeypatch.setattr(benchmark, "_synchronize", synchronize)
    monkeypatch.setattr(benchmark.time, "perf_counter", now)
    monkeypatch.setattr(benchmark, "_capture_trace", trace)
    case = benchmark.BenchmarkCase("reset", invoke, {}, batch_size=2)
    record, outputs = benchmark.measure_case(case, repeats=2, trace_directory=tmp_path)
    assert record["batch_size"] == 2
    assert record["cold_batch_size"] == 1
    assert len(record["warm_seconds"]) == 2
    assert record["cold_seconds"] > 0
    assert events.count("invoke") == 5
    assert events[-1] == "trace"
    assert all(float(value) == 4.0 for value in outputs.values())
    assert float(seed) == 3.0


def test_synchronize_waits_arrays_and_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = _module()
    events: list[str] = []
    monkeypatch.setattr(
        benchmark.jax, "block_until_ready", lambda value: events.append("arrays")
    )
    monkeypatch.setattr(
        benchmark.jax, "effects_barrier", lambda: events.append("effects")
    )
    result = {"x": jnp.asarray(1.0)}
    assert benchmark._synchronize(result) is result
    assert events == ["arrays", "effects"]


@pytest.mark.parametrize(
    "names",
    [
        ("host", "jax_api", "jax_jit"),
        ("irregular_host", "irregular_jax"),
        ("output_host", "output_jax"),
    ],
)
def test_real_runtime_cases_agree(names: tuple[str, ...], tmp_path: Path) -> None:
    benchmark = _module()
    all_outputs = []
    for name in names:
        args = benchmark.build_parser().parse_args(
            [
                "--directory",
                str(tmp_path / name),
                "--case",
                name,
                "--steps",
                "2",
                "--grid-nx",
                "2",
                "--grid-ny",
                "2",
                "--repeats",
                "1",
            ]
        )
        case = benchmark.build_case(name, args)
        first = benchmark._synchronize(case.invoke())
        second = benchmark._synchronize(case.invoke())
        first_arrays = benchmark._numeric_leaves(first)
        second_arrays = benchmark._numeric_leaves(second)
        for key in first_arrays:
            np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        artifacts = case.collect() if case.collect is not None else {}
        if name.startswith("output"):
            assert any("value" in key for key in artifacts)
            assert any("time" in key for key in artifacts)
            assert case.metadata["steps"] >= 24
        all_outputs.append({**first_arrays, **artifacts})
    for outputs in all_outputs[1:]:
        assert outputs.keys() == all_outputs[0].keys()
        for key in outputs:
            np.testing.assert_allclose(
                outputs[key], all_outputs[0][key], equal_nan=True
            )


def test_cli_writes_metadata_and_numeric_results(tmp_path: Path) -> None:
    benchmark = _module()
    assert (
        benchmark.main(
            ["--directory", str(tmp_path), "--case", "plan", "--repeats", "1"]
        )
        == 0
    )
    results = json.loads((tmp_path / "results.json").read_text())
    assert list(results["cases"]) == ["plan"]
    assert results["source"]["sha256"]
    assert results["environment"]["jax"]
    with np.load(tmp_path / "outputs.npz") as arrays:
        assert any(np.asarray(arrays[name]).item() == 4096 for name in arrays)


@pytest.mark.parametrize(
    "name",
    [
        "finite",
        "finite_jit",
        "refresh",
        "refresh_jit",
        "remap",
        "vector",
        "slab_host",
        "slab_jax",
    ],
)
def test_control_cases_return_repeatable_arrays(name: str, tmp_path: Path) -> None:
    benchmark = _module()
    args = benchmark.build_parser().parse_args(
        [
            "--directory",
            str(tmp_path),
            "--case",
            name,
            "--steps",
            "2",
            "--grid-nx",
            "3",
            "--grid-ny",
            "3",
        ]
    )
    case = benchmark.build_case(name, args)
    first = benchmark._numeric_leaves(benchmark._synchronize(case.invoke()))
    second = benchmark._numeric_leaves(benchmark._synchronize(case.invoke()))
    assert first
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    if name.startswith("finite"):
        assert any(np.isnan(array).any() for array in first.values())
    if name in ("remap", "vector"):
        assert len(first) == 2
        assert all(array.shape == (4, 4) for array in first.values())


def test_archive_identity_uses_content_hash_without_git(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    benchmark = _module()
    package = tmp_path / "vercor"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(benchmark.vercor, "__file__", str(source))
    first = benchmark._source_identity()
    source.write_text("x = 2\n")
    second = benchmark._source_identity()
    assert first["revision"] is None
    assert first["dirty_source"] is None
    assert first["sha256"] != second["sha256"]


def test_output_case_rejects_existing_invocation_root(tmp_path: Path) -> None:
    benchmark = _module()
    args = benchmark.build_parser().parse_args(
        ["--directory", str(tmp_path), "--grid-nx", "2", "--grid-ny", "2"]
    )
    benchmark.build_case("output_host", args)
    root = tmp_path / "output_host" / "invocations"
    root.mkdir(parents=True, exist_ok=True)
    sentinel = root / "0000" / "stale.nc"
    sentinel.parent.mkdir()
    original = b"previous campaign artifact must remain unchanged"
    sentinel.write_bytes(original)
    with pytest.raises(FileExistsError):
        benchmark.build_case("output_host", args)
    assert sentinel.read_bytes() == original
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == [
        Path("0000"),
        Path("0000/stale.nc"),
    ]
