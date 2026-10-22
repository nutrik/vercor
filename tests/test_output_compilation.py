"""Numeric output carry schemas stay stable after the first compilation."""

from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import make_clock, make_component
from vercor import Coupler
from vercor.output import (
    JaxOutputContext,
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("coordinate_kind", ["python", "weak", "strong"])
def test_scalar_coordinates_do_not_recompile_after_period_reset(
    coordinate_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dropping coordinate weak typing at allocation must not change JIT inputs."""
    traces = 0
    invocations = 0
    original_jit = jax.jit

    def recording_jit(function: Any, *args: Any, **kwargs: Any) -> Any:
        if (
            function.__module__ != "vercor.output._scan"
            or function.__name__ != "scanned_chunk"
        ):
            return original_jit(function, *args, **kwargs)

        @wraps(function)
        def trace(*inputs: Any, **options: Any) -> Any:
            nonlocal traces
            traces += 1
            return function(*inputs, **options)

        compiled = original_jit(trace, *args, **kwargs)

        @wraps(function)
        def invoke(*inputs: Any, **options: Any) -> Any:
            nonlocal invocations
            invocations += 1
            return compiled(*inputs, **options)

        return invoke

    class ScalarCoordinateProvider:
        def sample(self, context: OutputContext) -> OutputFrame:
            raise AssertionError("numeric output must use sample_jax")

        def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
            coordinate: Any = 1.0
            if coordinate_kind == "weak":
                coordinate = jnp.asarray(coordinate)
            elif coordinate_kind == "strong":
                coordinate = jnp.asarray(coordinate, dtype=jnp.float32)
            return OutputFrame(
                {"value": OutputVariable(("row", "col"), context.state.field("value"))},
                coordinates={
                    "constant": OutputVariable((), coordinate),
                    "position": OutputVariable((), context.step),
                },
            )

    component = make_component(
        "model",
        output=OutputSpec(
            provider=ScalarCoordinateProvider(), period=PeriodOutput("day")
        ),
    )
    coupler = Coupler(
        make_clock(12, dt_seconds=21_600.0),
        components=(component,),
        run_order=("model",),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    monkeypatch.setattr(jax, "jit", recording_jit)

    final = coupler.run(
        seed,
        output=OutputTarget(tmp_path, write_final_fields=False, write_snapshots=False),
    )
    jax.block_until_ready(final)

    paths = sorted(tmp_path.glob("model.averages.*.nc"))
    assert len(paths) == 3
    for path, mean, position in zip(paths, (2.5, 6.5, 10.5), (3, 7, 11), strict=True):
        with h5netcdf.File(path) as dataset:
            np.testing.assert_allclose(dataset.variables["value"][:], mean)
            assert np.asarray(dataset.variables["constant"]) == 1.0
            assert np.asarray(dataset.variables["position"]) == position
    np.testing.assert_allclose(final.component("model").field("value"), 12.0)
    assert invocations == 3
    assert traces == 1, "ERROR numeric coordinate carry changed compilation signature"
