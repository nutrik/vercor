"""Provider capabilities select sampling independently of runtime backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import h5netcdf
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import (
    SequentialBackend,
    make_clock,
    make_component,
)
from vercor import Coupler
from vercor.components import (
    CallableComponent,
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
    StepResult,
)
from vercor.exceptions import ComponentError
from vercor.output import (
    JaxOutputContext,
    OutputContext,
    OutputFrame,
    OutputSpec,
    OutputTarget,
    OutputVariable,
    PeriodOutput,
)
from vercor.runtime import RuntimeOptions

pytestmark = pytest.mark.fast_always


class NumericProvider:
    """Require the numeric contract even when an ordinary provider is enabled."""

    def sample(self, context: OutputContext) -> OutputFrame:
        raise AssertionError("numeric capability must be used on every backend")

    def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
        values = context.state.field("value")
        samples = jnp.stack((values, values + 2.0))
        samples = samples.at[:, 0, 1].set(jnp.nan)
        samples = samples.at[1, 1, 0].set(
            jnp.where(context.step % 2 == 0, jnp.nan, values[1, 0] + 2.0)
        )
        return OutputFrame(
            {"v": OutputVariable(("sample", "row", "col"), samples, {"units": "K"})},
            coordinates={
                "position": OutputVariable(
                    ("row",), jnp.asarray([context.step, context.step + 1])
                )
            },
            sample_dimension="sample",
            metadata={"source": "numeric"},
        )


class HostProvider:
    """Record concrete post-step contexts and return ordinary runtime fields."""

    def __init__(self) -> None:
        self.observed: list[tuple[int, datetime, timedelta, float]] = []

    def sample(self, context: OutputContext) -> OutputFrame:
        assert type(context.step) is int
        assert isinstance(context.time, datetime)
        self.observed.append(
            (
                context.step,
                context.time,
                context.dt,
                float(context.state.field("value")[0, 0]),
            )
        )
        return OutputFrame(
            {"value": OutputVariable(("row", "col"), context.state.field("value"))}
        )


@pytest.mark.parametrize("backend", ["jax", "host", "auto", "custom"])
@pytest.mark.parametrize("ordinary_provider", [False, True])
def test_numeric_sampling_is_independent_of_backend_and_neighbor_provider(
    backend: str, ordinary_provider: bool, tmp_path: Path
) -> None:
    """Backend-wide fallback must not override a provider's numeric capability."""
    host_provider = HostProvider()
    components = (
        make_component(
            "A",
            output=OutputSpec(provider=NumericProvider(), period=PeriodOutput("day")),
        ),
        make_component(
            "B",
            execution="host" if backend == "auto" else "jax",
            output=OutputSpec(
                provider=host_provider,
                period=PeriodOutput("day") if ordinary_provider else None,
            ),
        ),
    )
    runtime_backend: Any = SequentialBackend() if backend == "custom" else backend
    coupler = Coupler(
        make_clock(5, dt_seconds=43_200.0),
        components=components,
        run_order=("A", "B"),
        runtime=RuntimeOptions(backend=runtime_backend),
        log_level="ERROR",
    )
    for run in range(2):
        target = OutputTarget(
            tmp_path / str(run), write_final_fields=False, write_snapshots=False
        )
        final = coupler.run(output=target)
        np.testing.assert_allclose(final.component("A").field("value"), 5.0)
    paths = sorted(tmp_path.glob("*/a.averages.*.nc"))
    assert len(paths) == 4  # Two complete days per run; no partial-day flush.
    expected = [
        np.asarray([[[2.5, np.nan], [7.0 / 3.0, 2.5]]]),
        np.asarray([[[4.5, np.nan], [13.0 / 3.0, 4.5]]]),
    ]
    for path in paths:
        with h5netcdf.File(path) as dataset:
            day = (
                0
                if dataset.variables["time"].attrs["isoformat"].startswith("2000-01-01")
                else 1
            )
            np.testing.assert_allclose(
                dataset.variables["v"][:], expected[day], equal_nan=True
            )
            np.testing.assert_array_equal(
                dataset.variables["position"][:], [1 + day * 2, 2 + day * 2]
            )
            assert dataset.variables["position"].dtype == np.dtype("int32")
            assert dataset.variables["v"].attrs["units"] == "K"
            assert dataset.attrs["source"] == "numeric"
    if ordinary_provider:
        assert (
            host_provider.observed
            == [
                (
                    step,
                    datetime(2000, 1, 1) + timedelta(hours=12 * (step + 1)),
                    timedelta(hours=12),
                    float(step + 1),
                )
                for step in range(5)
            ]
            * 2
        )
        assert len(list(tmp_path.glob("*/b.averages.*.nc"))) == 4
    else:
        assert host_provider.observed == []


@pytest.mark.parametrize("backend", ["host", "auto", "custom"])
@pytest.mark.parametrize("failure", ["return", "selection", "drift"])
def test_host_numeric_sampling_preserves_component_scoped_errors(
    backend: str, failure: str, tmp_path: Path
) -> None:
    class InvalidNumericProvider(NumericProvider):
        def sample_jax(self, context: JaxOutputContext) -> Any:
            if failure == "return":
                return {}
            return OutputFrame(
                {"v": OutputVariable(("row", "col"), context.state.field("value"))},
                metadata={"step": int(context.step)} if failure == "drift" else None,
            )

    component = make_component(
        "A",
        execution="host" if backend == "auto" else "jax",
        output=OutputSpec(
            provider=InvalidNumericProvider(),
            period=PeriodOutput(
                "day", variables=("missing",) if failure == "selection" else ()
            ),
        ),
    )
    runtime_backend: Any = SequentialBackend() if backend == "custom" else backend
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=(component,),
        run_order=("A",),
        runtime=RuntimeOptions(backend=runtime_backend),
        log_level="ERROR",
    )
    reason = {
        "return": "must return OutputFrame",
        "selection": "unknown output variable",
        "drift": "metadata changed",
    }[failure]
    with pytest.raises(ComponentError, match=f"component 'A'.*{reason}"):
        coupler.run(output=OutputTarget(tmp_path))
    assert not list(tmp_path.glob("*.nc"))


@pytest.mark.parametrize("backend", ["host", "auto"])
@pytest.mark.parametrize("enabled", [False, True])
def test_host_numeric_sampling_does_not_trace_opaque_payloads(
    backend: Literal["host", "auto"], enabled: bool, tmp_path: Path
) -> None:
    """Host models keep opaque native state and sample it only when requested."""

    @dataclass(frozen=True)
    class NativeState:
        value: float

    class NativeProvider(NumericProvider):
        def sample_jax(self, context: JaxOutputContext) -> OutputFrame:
            assert enabled, "disabled output must never sample the provider"
            assert isinstance(context.payload, NativeState)
            return OutputFrame(
                {"native": OutputVariable((), np.asarray(context.payload.value))}
            )

    def setup(component: Any, context: SetupContext) -> SetupResult:
        return SetupResult(fields={"value": 0.0}, payload=NativeState(10.0))

    def step(
        fields: Mapping[str, Any], context: StepContext, payload: NativeState
    ) -> StepResult:
        return StepResult(
            fields={"value": fields["value"] + 1.0},
            payload=NativeState(payload.value + 1.0),
        )

    component = CallableComponent(
        "native",
        make_component("grid").grid,
        step,
        spec=ComponentSpec(
            inputs=("value",),
            outputs=("value",),
            execution="host",
            lifecycle=LifecycleHooks(setup=setup),
            output=OutputSpec(provider=NativeProvider(), period=PeriodOutput("day")),
        ),
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=(component,),
        run_order=("native",),
        runtime=RuntimeOptions(backend=backend),
        log_level="ERROR",
    )
    final = coupler.run(
        output=OutputTarget(
            tmp_path,
            write_period=enabled,
            write_final_fields=False,
            write_snapshots=False,
        )
    )
    np.testing.assert_allclose(final.component("native").field("value"), 2.0)
    if enabled:
        (path,) = tmp_path.glob("native.averages.*.nc")
        with h5netcdf.File(path) as dataset:
            np.testing.assert_allclose(dataset.variables["native"][:], [11.5])
    else:
        assert not list(tmp_path.glob("*.nc"))
