"""Selection precedes extraction and reduced frames keep exact native counts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests._workflow_test_support import SequentialBackend, make_clock
from vercor import Coupler
from vercor.exceptions import CouplerError
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.output import (
    JaxOutputContext,
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)
from vercor.output._session import _OutputAccumulator
from vercor.runtime import RuntimeOptions

pytestmark = pytest.mark.fast_always


def _component(
    *,
    variables: tuple[str, ...] = (),
    provider: Any = None,
    execution: Literal["jax", "host"] = "jax",
    observed: list[Any] | None = None,
) -> Any:
    def step(fields: Mapping[str, Any], context: StepContext) -> Mapping[str, Any]:
        if observed is not None:
            observed.append(context.output_variables)
        return {"value": fields["value"] + 1.0}

    return CallableComponent(
        "A",
        make_test_grid(name="selective"),
        step,
        spec=ComponentSpec(
            inputs=("forcing",),
            outputs=("value",),
            initial_fields={"forcing": 7.0, "value": 0.0},
            execution=execution,
            output=OutputSpec(provider=provider, period=PeriodOutput("day", variables)),
        ),
    )


def _target(path: Path) -> OutputTarget:
    return OutputTarget(path, write_final_fields=False, write_snapshots=False)


@pytest.mark.parametrize("backend", ["jax", "host", "auto", "custom"])
def test_default_runtime_output_includes_input_fields(
    backend: Literal["jax", "host", "auto", "custom"], tmp_path: Path
) -> None:
    component = _component(execution="host" if backend == "auto" else "jax")
    runtime_backend = SequentialBackend() if backend == "custom" else backend
    coupler = Coupler(
        make_clock(4, dt_seconds=43_200),
        components=(component,),
        run_order=("A",),
        runtime=RuntimeOptions(backend=runtime_backend),
        log_level="ERROR",
    )
    coupler.run(output=_target(tmp_path))
    paths = sorted(tmp_path.glob("a.averages.*.nc"))
    assert len(paths) == 2
    for path, expected in zip(paths, [1.5, 3.5], strict=True):
        with h5netcdf.File(path) as dataset:
            assert {"value", "forcing"} <= set(dataset.variables)
            np.testing.assert_allclose(dataset.variables["value"], expected)
            np.testing.assert_allclose(dataset.variables["forcing"], 7.0)


@pytest.mark.parametrize("backend", ["jax", "host", "custom"])
def test_unselected_runtime_fields_are_not_extracted(
    backend: Literal["jax", "host", "auto", "custom"],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.output._session as session_module

    def selected_only(values: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unselected runtime field sampled")

    monkeypatch.setattr(session_module, "select_runtime_field", selected_only)
    # Request an input only: no export selector should run at all.
    component = _component(variables=("forcing",))
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200),
        components=(component,),
        run_order=("A",),
        runtime=RuntimeOptions(
            backend=SequentialBackend() if backend == "custom" else backend
        ),
        log_level="ERROR",
    )
    coupler.run(output=_target(tmp_path))
    with h5netcdf.File(next(tmp_path.glob("a.averages.*.nc"))) as dataset:
        assert "value" not in dataset.variables
        np.testing.assert_allclose(dataset.variables["forcing"], 7.0)


@pytest.mark.parametrize("numeric", [False, True])
@pytest.mark.parametrize("backend", ["jax", "host", "custom"])
def test_provider_receives_selection_before_sampling(
    numeric: bool,
    backend: Literal["jax", "host", "auto", "custom"],
    tmp_path: Path,
) -> None:
    class Provider:
        def sample(self, context: OutputContext) -> OutputFrame:
            return self.extract(context)

        def extract(self, context: OutputContext | JaxOutputContext) -> OutputFrame:
            assert context.variables == ("chosen",)
            return OutputFrame({"chosen": OutputVariable((), jnp.asarray(3.0))})

    class NumericProvider(Provider):
        def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
            return self.extract(context)

    component = _component(
        variables=("chosen",), provider=NumericProvider() if numeric else Provider()
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200),
        components=(component,),
        run_order=("A",),
        runtime=RuntimeOptions(
            backend=SequentialBackend() if backend == "custom" else backend
        ),
        log_level="ERROR",
    )
    coupler.run(output=_target(tmp_path))
    with h5netcdf.File(next(tmp_path.glob("a.averages.*.nc"))) as dataset:
        np.testing.assert_allclose(dataset.variables["chosen"], 3.0)


def test_reduced_native_frames_merge_counts_without_intermediate_means() -> None:
    def frame(sums: list[float], counts: list[int]) -> OutputFrame:
        return OutputFrame(
            {"native": OutputVariable(("x",), jnp.asarray(sums))},
            reduction_counts={"native": jnp.asarray(counts, dtype=jnp.int32)},
        )

    first = frame([3.0, 0.0, 2.0], [2, 0, 1])
    accumulator = _OutputAccumulator.zeros_from_frame(first).add_frame(first)
    before = sum(leaf.nbytes for leaf in jax.tree.leaves(accumulator))
    for _ in range(9):
        accumulator = accumulator.add_frame(frame([5.0, 0.0, 9.0], [1, 0, 3]))
    assert sum(leaf.nbytes for leaf in jax.tree.leaves(accumulator)) == before
    np.testing.assert_allclose(accumulator.sum_values[0], [48.0, 0.0, 83.0])
    np.testing.assert_array_equal(accumulator.counts[0], [11, 0, 28])
    np.testing.assert_allclose(
        accumulator.mean_frame().variables["native"].values,
        [48 / 11, np.nan, 83 / 28],
        equal_nan=True,
    )
    np.testing.assert_array_equal(accumulator.reset().counts[0], 0)


@pytest.mark.parametrize(
    "counts, message",
    [
        ({"missing": np.ones(2, dtype=int)}, "unknown"),
        ({"x": np.ones(3, dtype=int)}, "shape"),
        ({"x": np.ones(2, dtype=float)}, "integer"),
        ({"x": np.asarray([1, -1])}, "nonnegative"),
    ],
)
def test_reduced_frame_rejects_invalid_counts(counts: Any, message: str) -> None:
    with pytest.raises((TypeError, ValueError, CouplerError), match=message):
        frame = OutputFrame(
            {"x": OutputVariable(("x",), np.ones(2))}, reduction_counts=counts
        )
        _OutputAccumulator.zeros_from_frame(frame).add_frame(frame)


@pytest.mark.parametrize(
    "enabled, variables", [(False, None), (True, ()), (True, ("value",))]
)
@pytest.mark.parametrize("backend", ["host", "custom"])
def test_native_step_receives_run_specific_output_policy(
    enabled: bool,
    variables: tuple[str, ...] | None,
    backend: Literal["jax", "host", "auto", "custom"],
    tmp_path: Path,
) -> None:
    observed: list[Any] = []
    component = _component(variables=variables or (), observed=observed)
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200),
        components=(component,),
        run_order=("A",),
        runtime=RuntimeOptions(
            backend=SequentialBackend() if backend == "custom" else backend
        ),
        log_level="ERROR",
    )
    coupler.run(output=_target(tmp_path) if enabled else None)
    assert observed == [variables, variables]
