"""Clock-selected forcing reaches recipients independently of source scheduling."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import SequentialBackend, StaticWorkflow
from tests.assertions import assert_allclose_compact
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import (
    CallableComponent,
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
    TransferPolicy,
)
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor.runtime import ExecutionBackend, StepPlan

Selection = Literal["daily", "linear"]
Lane = Literal["host", "jax", "mixed", "custom"]


def _case(
    selection: Selection,
    lane: Lane,
    order: tuple[str, ...],
    *,
    masked: bool = False,
    generic_source: bool = False,
    schedules: tuple[tuple[str, ...], ...] | None = None,
) -> Coupler:
    grid = RectilinearGrid(
        "records", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0)
    )
    if selection == "daily":
        values = np.arange(365, dtype=float) + 2.0
        values[:3] = (2.0, 5.0, 11.0)
        clock = Clock(datetime(2001, 1, 1), 86400.0, 3, calendar="noleap")
    else:
        values = np.arange(12, dtype=float) * 10.0
        # Equal calendar-year record spacing: Jan 31 10:00 is halfway between
        # the first two record midpoints, giving literal values 5, 15 and 25.
        clock = Clock(datetime(2001, 1, 31, 10), 2628000.0, 3, calendar="noleap")
    records = np.broadcast_to(values[:, None, None], (len(values), 2, 2)).copy()
    area = np.array([[0.25, 0.0], [0.5, 1.0]]) if masked else np.ones((2, 2))
    if masked:
        records[:, 0, 1] = np.nan
    fields = {"forcing": records, "area": area}
    source_spec = ComponentSpec(
        outputs=("forcing", "area"),
        initial_fields=fields,
        transfer=TransferPolicy(selection),
        masks=ComponentMaskSpec(fractional_field="area"),
    )

    def forbidden_source_step(fields: object) -> dict[str, jax.Array]:
        raise AssertionError("ERROR omitted source physics must not execute")

    source = (
        CallableComponent("DATA", grid, forbidden_source_step, spec=source_spec)
        if generic_source
        else DataComponent("DATA", grid, fields, spec=source_spec)
    )
    sink = CallableComponent(
        "SINK",
        grid,
        lambda f: {"seen": f["forcing"], "total": f["total"] + f["forcing"]},
        spec=ComponentSpec(
            inputs=("forcing", "area"),
            outputs=("seen", "total"),
            initial_fields={"seen": 0.0, "total": 0.0},
        ),
    )
    # An actual host-required component forces the auto lane to use its mixed
    # execution path. Its result proves it was included in every plan.
    host = CallableComponent(
        "HOST",
        grid,
        lambda f: {"count": np.asarray(f["count"]) + 1.0},
        spec=ComponentSpec(
            outputs=("count",), initial_fields={"count": 0.0}, execution="host"
        ),
    )
    schedule = (*order, "HOST") if lane == "mixed" else order
    backend: ExecutionBackend | Literal["auto", "jax", "host"] = (
        SequentialBackend()
        if lane == "custom"
        else ("auto" if lane == "mixed" else lane)
    )
    return Coupler(
        clock,
        components=(source, sink, host) if lane == "mixed" else (source, sink),
        exchanges=(
            Exchange("DATA", "SINK", ("forcing",), quantity="source_flux"),
            Exchange(
                "DATA",
                "SINK",
                ("area",),
                quantity="intensive",
                source_coverage="binary",
                route_id="current-area",
            ),
        ),
        run_order=("DATA", "SINK", "HOST") if lane == "mixed" else ("DATA", "SINK"),
        runtime=RuntimeOptions(
            backend=backend,
            dtype=DTypePolicy(enable_x64=True),
            workflow=StaticWorkflow(
                tuple(
                    StepPlan(i, names)
                    for i, names in enumerate(schedules or (schedule,) * 3)
                )
            ),
            topology=None,
        ),
        log_level="error",
    )


@pytest.mark.parametrize(
    "selection,last,total", [("daily", 11.0, 18.0), ("linear", 25.0, 45.0)]
)
@pytest.mark.parametrize("lane", ["host", "jax", "mixed", "custom"])
@pytest.mark.parametrize("order", [("DATA", "SINK"), ("SINK", "DATA"), ("SINK",)])
def test_records_follow_receiving_clock(
    selection: Selection, last: float, total: float, lane: Lane, order: tuple[str, ...]
) -> None:
    """Catch stale sends when a record source follows its target or never steps."""
    coupler = _case(selection, lane, order)
    initial = coupler.initial_state()
    final = coupler.run(initial)
    assert_allclose_compact(final.component("SINK").field("seen"), last, atol=1e-12)
    assert_allclose_compact(final.component("SINK").field("total"), total, atol=1e-12)
    assert_allclose_compact(
        final.component("DATA").field("forcing", scope="sent"), last, atol=1e-12
    )
    np.testing.assert_array_equal(
        final.component("DATA").field("forcing"),
        initial.component("DATA").field("forcing"),
    )
    if lane == "mixed":
        assert_allclose_compact(final.component("HOST").field("count"), 3.0)


@pytest.mark.parametrize("selection,last", [("daily", 11.0), ("linear", 25.0)])
@pytest.mark.parametrize("lane", ["host", "jax"])
def test_generic_record_source_keeps_current_masks_and_inactive_nan(
    selection: Selection, last: float, lane: Lane
) -> None:
    """Refresh transfer policy generically without selecting grid mask records."""
    coupler = _case(selection, lane, ("SINK",), masked=True, generic_source=True)
    initial = coupler.initial_state()
    final = coupler.run(initial)
    area = np.array([[0.25, 0.0], [0.5, 1.0]])
    assert_allclose_compact(
        final.component("SINK").field("seen"), last * area, atol=1e-12
    )
    assert_allclose_compact(final.component("SINK").field("area"), area)
    assert_allclose_compact(final.component("DATA").field("area", scope="sent"), area)
    assert_allclose_compact(final._received_fractional_masks.get("DATA->SINK"), area)
    assert_allclose_compact(final._fractional_mask("DATA->SINK"), area)
    assert np.isnan(final.component("DATA").field("forcing", scope="sent")[0, 1])


@pytest.mark.parametrize("lane", ["host", "jax", "mixed", "custom"])
def test_replaced_records_are_selected_when_source_is_omitted(lane: Lane) -> None:
    """A supplied record array must take effect even if its author never runs."""
    coupler = _case("daily", lane, ("SINK",))
    initial = coupler.initial_state()
    updated = initial.replace_fields("DATA", {"forcing": jnp.full((365, 2, 2), 7.0)})
    final = coupler.run(updated)
    assert_allclose_compact(final.component("SINK").field("total"), 21.0)
    assert_allclose_compact(final.component("DATA").field("forcing", scope="sent"), 7.0)
    assert_allclose_compact(
        initial.component("DATA").field("forcing", scope="sent"), 2.0
    )


@pytest.mark.parametrize("lane", ["host", "jax"])
def test_current_feedback_keeps_explicit_sequential_order(
    lane: Literal["host", "jax"],
) -> None:
    """Refreshing ordinary current sends early would hide evolving feedback lag."""
    grid = RectilinearGrid(
        "feedback", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0)
    )
    source = CallableComponent(
        "MODEL",
        grid,
        lambda f: {"forcing": f["forcing"] + 1.0},
        spec=ComponentSpec(outputs=("forcing",), initial_fields={"forcing": 2.0}),
    )
    targets = tuple(
        CallableComponent(
            name,
            grid,
            lambda f: {"total": f["total"] + f["forcing"]},
            spec=ComponentSpec(
                inputs=("forcing",), outputs=("total",), initial_fields={"total": 0.0}
            ),
        )
        for name in ("BEFORE", "AFTER")
    )
    coupler = Coupler(
        Clock(datetime(2001, 1, 1), 86400.0, 3),
        components=(source, *targets),
        exchanges=tuple(
            Exchange("MODEL", name, ("forcing",), quantity="intensive")
            for name in ("BEFORE", "AFTER")
        ),
        run_order=("BEFORE", "MODEL", "AFTER"),
        runtime=RuntimeOptions(backend=lane, topology=None),
        log_level="error",
    )
    final = coupler.run()
    assert_allclose_compact(final.component("BEFORE").field("total"), 9.0)
    assert_allclose_compact(final.component("AFTER").field("total"), 12.0)


def test_omitted_source_selection_preserves_jit_forward_and_reverse_gradients() -> None:
    """Gradients must use selected supplied records rather than stale sent leaves."""
    coupler = _case("daily", "jax", ("SINK",), masked=True)
    initial = coupler.initial_state()
    records = jnp.nan_to_num(initial.component("DATA").field("forcing"))
    area = initial.component("DATA").field("area")

    def objective(scale: jax.Array) -> jax.Array:
        updated = initial.replace_fields(
            "DATA", {"forcing": jnp.where(area > 0, records * scale, jnp.nan)}
        )
        final = coupler.run(updated)
        return jnp.sum(final.component("SINK").field("total"))

    scale = jnp.asarray(2.0)
    value, tangent = jax.jvp(jax.jit(objective), (scale,), (jnp.ones_like(scale),))
    assert_allclose_compact(value, 63.0)
    assert_allclose_compact(tangent, 31.5)
    assert_allclose_compact(jax.grad(objective)(scale), 31.5)


def test_supplied_invalid_sent_store_is_rejected_before_dispatch_refresh() -> None:
    """Temporal refresh must not conceal invalid supplied outgoing state."""
    coupler = _case("daily", "host", ("SINK",))
    initial = coupler.initial_state()
    component = initial._component_state("DATA")
    invalid = initial._with_component_state(
        "DATA",
        component.with_sent(
            component.sent.set_many({"forcing": jnp.full((2, 2), jnp.nan)})
        ),
    )
    with pytest.raises(CouplerError, match="DATA.*forcing"):
        coupler.run(invalid)


def test_record_refresh_reuses_one_scan_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing selection indices must not retrace an unchanged component schedule."""
    from tests.test_runtime_compilation import _compiled_traces

    coupler = _case(
        "daily",
        "jax",
        ("SINK",),
        schedules=(("SINK",), ("SINK", "DATA"), ("SINK",)),
    )
    initial = coupler.initial_state()
    traces = _compiled_traces(monkeypatch)
    final = jax.block_until_ready(coupler.run(initial))
    assert_allclose_compact(final.component("SINK").field("total"), 18.0)
    # Two schedules, three windows: the repeated SINK schedule reuses its
    # executable despite advancing the forcing record from day zero to two.
    assert traces == [1, 1]
