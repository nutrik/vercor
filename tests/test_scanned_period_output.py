"""Boundary scans preserve post-step output means without a stored trajectory."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import make_clock, make_component
from vercor import Coupler
from vercor.dtypes import DTypePolicy
from vercor.components import CallableComponent, ComponentSpec, LifecycleHooks
from vercor.exceptions import ComponentError
import vercor.output as output
from vercor.runtime import RuntimeOptions

pytestmark = pytest.mark.fast_always


def _scan_lengths(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Observe executed scan inputs, including cached compiled invocations."""
    lengths: list[int] = []
    original = jax.jit

    def recording_jit(function: Any, *args: Any, **kwargs: Any) -> Any:
        compiled = original(function, *args, **kwargs)

        def invoke(*inputs: Any, **kw: Any) -> Any:
            lengths.append(int(inputs[1].shape[0]))
            return compiled(*inputs, **kw)

        return invoke

    monkeypatch.setattr(jax, "jit", recording_jit)
    return lengths


def _target(path: Path) -> output.OutputTarget:
    return output.OutputTarget(path, write_final_fields=False, write_snapshots=False)


def _values(path: Path, name: str = "value") -> np.ndarray:
    with h5netcdf.File(path) as dataset:
        return np.asarray(dataset.variables[name])


def test_daily_output_scans_whole_windows_and_incomplete_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A singleton-chunk regression must fail the invocation and mean checks."""
    component = make_component(
        "A", output=output.OutputSpec(period=output.PeriodOutput(frequency="day"))
    )
    coupler = Coupler(
        make_clock(10, dt_seconds=21_600.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    lengths = _scan_lengths(monkeypatch)
    final = coupler.run(seed, output=_target(tmp_path))
    jax.block_until_ready(final)
    assert lengths == [4, 4, 2]
    paths = sorted(tmp_path.glob("a.averages.*.nc"))
    assert len(paths) == 2
    np.testing.assert_allclose(_values(paths[0]), 2.5)
    np.testing.assert_allclose(_values(paths[1]), 6.5)
    np.testing.assert_allclose(final.component("A").field("value"), 10.0)


def test_opted_in_numeric_provider_uses_post_step_native_counts_and_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping samples, counting NaNs or freezing coordinates changes output."""

    class NumericProvider:
        def sample(self, context: output.OutputContext) -> output.OutputFrame:
            raise AssertionError("JAX output must not call the ordinary host sampler")

        def sample_jax(self, context: Any) -> output.OutputFrame:
            value = context.state.field("value")
            samples = jnp.stack((value, value + 2.0))
            samples = samples.at[:, 0, 1].set(jnp.nan)
            samples = samples.at[1, 1, 0].set(
                jnp.where(context.step % 2 == 0, jnp.nan, value[1, 0] + 2.0)
            )
            return output.OutputFrame(
                {
                    "v": output.OutputVariable(
                        ("sample", "row", "col"), samples, {"units": "K"}
                    )
                },
                coordinates={
                    "position": output.OutputVariable(
                        ("row",), jnp.asarray([context.step, context.step + 1])
                    )
                },
                sample_dimension="sample",
                metadata={"source": "numeric"},
            )

    provider = NumericProvider()
    assert hasattr(output, "JaxOutputProvider")
    assert isinstance(provider, getattr(output, "JaxOutputProvider"))
    component = make_component(
        "A",
        output=output.OutputSpec(
            provider=provider,
            period=output.PeriodOutput(frequency="day", variables=("v",)),
        ),
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    lengths = _scan_lengths(monkeypatch)
    final = coupler.run(seed, output=_target(tmp_path))
    jax.block_until_ready(final)
    assert lengths == [2]
    (path,) = tmp_path.glob("a.averages.*.nc")
    expected = np.asarray([[[2.5, np.nan], [7.0 / 3.0, 2.5]]])
    np.testing.assert_allclose(_values(path, "v"), expected, equal_nan=True)
    np.testing.assert_array_equal(_values(path, "position"), [1, 2])
    with h5netcdf.File(path) as dataset:
        assert dataset.attrs["source"] == "numeric"
        assert dataset.variables["v"].attrs["units"] == "K"


def test_host_only_provider_keeps_concrete_post_step_context_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[int, datetime, float]] = []

    class HostProvider:
        def sample(self, context: output.OutputContext) -> output.OutputFrame:
            assert type(context.step) is int
            assert isinstance(context.time, datetime)
            observed.append(
                (context.step, context.time, float(context.state.field("value")[0, 0]))
            )
            return output.OutputFrame(
                {
                    "value": output.OutputVariable(
                        ("row", "col"), context.state.field("value")
                    )
                }
            )

    component = make_component(
        "A",
        output=output.OutputSpec(
            provider=HostProvider(), period=output.PeriodOutput(frequency="day")
        ),
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    lengths = _scan_lengths(monkeypatch)
    coupler.run(seed, output=_target(tmp_path))
    assert lengths == [1, 1]
    assert observed == [
        (0, datetime(2000, 1, 1, 12), 1.0),
        (1, datetime(2000, 1, 2), 2.0),
    ]
    (path,) = tmp_path.glob("a.averages.*.nc")
    np.testing.assert_allclose(_values(path), 1.5)


@pytest.mark.parametrize("kind", ["none", "disabled", "final_only", "zero_steps"])
def test_unused_numeric_provider_is_never_traced_or_sampled(
    kind: str, tmp_path: Path
) -> None:
    class ExplodingProvider:
        def sample(self, context: Any) -> output.OutputFrame:
            raise AssertionError("unexpected provider sampling")

        def sample_jax(self, context: Any) -> output.OutputFrame:
            raise AssertionError("unexpected numeric schema tracing")

    component = make_component(
        "A",
        output=output.OutputSpec(
            provider=ExplodingProvider(), period=output.PeriodOutput(frequency="day")
        ),
    )
    steps = 0 if kind == "zero_steps" else 2
    coupler = Coupler(
        make_clock(steps), components=(component,), run_order=("A",), log_level="ERROR"
    )
    target = (
        None
        if kind == "none"
        else output.OutputTarget(
            tmp_path,
            write_period=kind == "zero_steps",
            write_final_fields=kind == "final_only",
            write_snapshots=False,
        )
    )
    final = coupler.run(output=target)
    np.testing.assert_allclose(final.component("A").field("value"), float(steps))


@pytest.mark.parametrize(
    "failure", ["return", "empty", "selection", "metadata", "sample_error"]
)
def test_numeric_provider_failures_remain_component_scoped(
    failure: str, tmp_path: Path
) -> None:
    class InvalidNumericProvider:
        def sample(self, context: Any) -> output.OutputFrame:
            raise AssertionError("unexpected fallback")

        def sample_jax(self, context: Any) -> Any:
            if failure == "return":
                return {}
            if failure == "sample_error":
                raise ValueError("numeric extraction failed")
            return output.OutputFrame(
                (
                    {}
                    if failure == "empty"
                    else {
                        "value": output.OutputVariable(
                            ("row", "col"), context.state.field("value")
                        )
                    }
                ),
                metadata={"step": context.step} if failure == "metadata" else None,
            )

    component = make_component(
        "A",
        output=output.OutputSpec(
            provider=InvalidNumericProvider(),
            period=output.PeriodOutput(
                frequency="day",
                variables=("missing",) if failure == "selection" else (),
            ),
        ),
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    with pytest.raises(
        ComponentError,
        match="component 'A'.*(OutputFrame|no variables|missing|metadata|numeric extraction failed)",
    ):
        coupler.run(output=_target(tmp_path))
    assert not tuple(tmp_path.glob("*.nc"))


@pytest.mark.parametrize(
    "coordinates",
    [
        np.asarray([3_000_000_000, 3_000_000_001], dtype=np.int64),
        np.asarray([1.000000000001, 1.000000000002], dtype=np.float64),
    ],
)
def test_host_provider_preserves_coordinate_precision_with_x64_disabled(
    coordinates: np.ndarray,
    tmp_path: Path,
) -> None:
    """Numeric carry normalization must not cast ordinary provider coordinates."""

    class HostProvider:
        def sample(self, context: output.OutputContext) -> output.OutputFrame:
            return output.OutputFrame(
                {
                    "value": output.OutputVariable(
                        ("row", "col"), context.state.field("value")
                    )
                },
                coordinates={"id": output.OutputVariable(("row",), coordinates)},
            )

    component = make_component(
        "A",
        output=output.OutputSpec(
            provider=HostProvider(),
            period=output.PeriodOutput(frequency="day"),
        ),
    )
    with jax.enable_x64(False):
        coupler = Coupler(
            make_clock(2, dt_seconds=43_200.0),
            components=(component,),
            run_order=("A",),
            runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
            log_level="ERROR",
        )
        coupler.run(output=_target(tmp_path))
    (path,) = tmp_path.glob("a.averages.*.nc")
    actual = _values(path, "id")
    assert actual.dtype == coordinates.dtype
    np.testing.assert_array_equal(actual, coordinates)


def test_scanned_output_rejects_transient_lifecycle_validation_failure(
    tmp_path: Path,
) -> None:
    """A later valid endpoint must not hide a rejected intermediate state."""
    observed = []

    def validate(component: Any, context: Any) -> None:
        value = float(context.state.field("value")[0, 0])
        observed.append(value)
        if value == 1.0:
            raise ComponentError("transient invalid first step")

    base = make_component("A")
    component = CallableComponent(
        "A",
        base.grid,
        lambda fields, context: {"value": fields["value"] + 1.0},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            lifecycle=LifecycleHooks(validate=validate),
            output=output.OutputSpec(period=output.PeriodOutput(frequency="day")),
        ),
    )
    coupler = Coupler(
        make_clock(4, dt_seconds=21_600.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    with pytest.raises(
        (ComponentError, jax.errors.JaxRuntimeError),
        match="transient invalid first step",
    ):
        final = coupler.run(output=_target(tmp_path))
        jax.block_until_ready(final)
        jax.effects_barrier()
    assert 1.0 in observed
    assert 4.0 not in observed
    assert not tuple(tmp_path.glob("*.nc"))


def test_one_host_provider_preserves_host_sampling_for_all_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using any rather than all numeric capabilities must not trace host code."""
    observed = []

    class HostProvider:
        def sample(self, context: output.OutputContext) -> output.OutputFrame:
            assert type(context.step) is int
            observed.append((context.component.name, context.step))
            return output.OutputFrame(
                {
                    "value": output.OutputVariable(
                        ("row", "col"), context.state.field("value")
                    )
                }
            )

    class NumericProvider(HostProvider):
        def sample_jax(self, context: Any) -> output.OutputFrame:
            raise AssertionError("mixed providers must retain host sampling")

    components = tuple(
        make_component(
            name,
            output=output.OutputSpec(
                provider=provider,
                period=output.PeriodOutput(frequency="day"),
            ),
        )
        for name, provider in (("A", NumericProvider()), ("B", HostProvider()))
    )
    coupler = Coupler(
        make_clock(2, dt_seconds=43_200.0),
        components=components,
        run_order=("A", "B"),
        log_level="ERROR",
    )
    seed = coupler.initial_state()
    lengths = _scan_lengths(monkeypatch)
    coupler.run(seed, output=_target(tmp_path))
    assert lengths == [1, 1]
    assert observed == [("A", 0), ("B", 0), ("A", 1), ("B", 1)]
    for path in tmp_path.glob("*.nc"):
        np.testing.assert_allclose(_values(path), 1.5)


def test_numeric_output_validates_each_actual_step_once(tmp_path: Path) -> None:
    """Abstract preparation and chunk endpoints must not add author-hook calls."""
    observed = []

    def validate(component: Any, context: Any) -> None:
        observed.append(float(context.state.field("value")[0, 0]))

    base = make_component("A")
    component = CallableComponent(
        "A",
        base.grid,
        lambda fields, context: {"value": fields["value"] + 1.0},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            lifecycle=LifecycleHooks(validate=validate),
            output=output.OutputSpec(period=output.PeriodOutput(frequency="day")),
        ),
    )
    coupler = Coupler(
        make_clock(4, dt_seconds=21_600.0),
        components=(component,),
        run_order=("A",),
        log_level="ERROR",
    )
    final = coupler.run(output=_target(tmp_path))
    jax.block_until_ready(final)
    jax.effects_barrier()
    assert [value for value in observed if value != 0.0] == [1.0, 2.0, 3.0, 4.0]
