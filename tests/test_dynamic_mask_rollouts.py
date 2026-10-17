"""Check evolving coverage through sequential rollouts, AD and host output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr

from tests._workflow_test_support import StaticWorkflow
from vercor._runtime.stores import FieldStore
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
from vercor.output import OutputTarget
from vercor.regridding import bilinear
from vercor.runtime import SequentialWorkflow, StepPlan

pytestmark = pytest.mark.fast_always


def _coupler(
    *, execution: str = "jax", steps: int = 2, workflow: Any = None
) -> Coupler:
    grid = RectilinearGrid(
        "ocean", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    source = CallableComponent(
        "SRC",
        grid,
        lambda fields: {"area": fields["area"] + 0.25},
        spec=ComponentSpec(
            outputs=("area", "flux"),
            initial_fields={"area": 0.25, "flux": 2.0},
            masks=ComponentMaskSpec(fractional_field="area"),
            execution=cast(Any, execution),
        ),
    )
    targets = tuple(
        CallableComponent(
            name,
            grid,
            lambda fields: {"total": fields["total"] + fields["flux"]},
            spec=ComponentSpec(
                inputs=("flux",),
                outputs=("total",),
                initial_fields={"total": 0.0},
                execution=cast(Any, execution),
            ),
        )
        for name in ("BEFORE", "AFTER")
    )
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, steps),
        components=(source, *targets),
        exchanges=tuple(
            Exchange(
                "SRC",
                name,
                fields=("flux",),
                quantity="source_flux",
                regridder_factory=bilinear,
            )
            for name in ("BEFORE", "AFTER")
        ),
        run_order=("BEFORE", "SRC", "AFTER"),
        runtime=RuntimeOptions(
            dtype=DTypePolicy(enable_x64=True),
            workflow=SequentialWorkflow() if workflow is None else workflow,
        ),
    )


@pytest.mark.parametrize("execution", ["host", "jax"])
def test_mask_refresh_respects_sequential_order(execution: str) -> None:
    coupler = _coupler(execution=execution)
    initial = coupler.initial_state()
    np.testing.assert_allclose(initial._fractional_mask("SRC->AFTER"), 0.25)
    final = coupler.run(initial)
    np.testing.assert_allclose(final.component("BEFORE").field("total"), 1.5)
    np.testing.assert_allclose(final.component("AFTER").field("total"), 2.5)
    np.testing.assert_allclose(final._fractional_mask("SRC->AFTER"), 0.75)
    np.testing.assert_allclose(final._received_fractional_masks.get("SRC->BEFORE"), 0.5)
    np.testing.assert_allclose(final._received_fractional_masks.get("SRC->AFTER"), 0.75)
    np.testing.assert_allclose(initial.component("SRC").field("area"), 0.25)
    assert cast(Any, jax.tree_util.tree_structure(final)) == cast(
        Any, jax.tree_util.tree_structure(initial)
    )


def test_replaced_mask_fields_refresh_supplied_state() -> None:
    workflow = StaticWorkflow((StepPlan(0, ()), StepPlan(1, ())))
    coupler = _coupler(workflow=workflow)
    initial = coupler.initial_state()
    updated = initial.replace_fields("SRC", {"area": jnp.full((2, 2), 0.5)})
    final = coupler.run(updated)
    np.testing.assert_allclose(final._fractional_mask("SRC->AFTER"), 0.5)
    np.testing.assert_allclose(initial._fractional_mask("SRC->AFTER"), 0.25)


@pytest.mark.parametrize("execution", ["host", "jax"])
@pytest.mark.parametrize("mask_kind", ["binary_field", "fractional_field"])
@pytest.mark.parametrize("order", [("SRC", "DST"), ("DST", "SRC"), ("DST",)])
def test_replaced_current_exports_reactivate_without_stale_sent_values(
    execution: str,
    mask_kind: str,
    order: tuple[str, ...],
) -> None:
    grid = RectilinearGrid("wet", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0))
    source = CallableComponent(
        "SRC",
        grid,
        lambda f: {},
        spec=ComponentSpec(
            outputs=("area", "flux"),
            initial_fields={"area": 0.0, "flux": jnp.nan},
            masks=ComponentMaskSpec(**{mask_kind: "area"}),
            execution=cast(Any, execution),
        ),
    )
    target = CallableComponent(
        "DST",
        grid,
        lambda f: {"total": f["flux"]},
        spec=ComponentSpec(
            inputs=("area", "flux"),
            outputs=("total",),
            initial_fields={"total": 0.0},
            execution=cast(Any, execution),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(source, target),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("area", "flux"),
                quantity="source_flux",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(
            dtype=DTypePolicy(enable_x64=True),
            workflow=StaticWorkflow((StepPlan(0, order),)),
        ),
    )
    initial = coupler.initial_state()
    updated = initial.replace_fields(
        "SRC", {"area": jnp.ones((2, 2)), "flux": jnp.full((2, 2), 2.0)}
    )
    final = coupler.run(updated)
    np.testing.assert_array_equal(final.component("DST").field("total"), 2.0)
    np.testing.assert_array_equal(final.component("DST").field("area"), 1.0)
    np.testing.assert_array_equal(initial._component_state("SRC").sent.get("area"), 0.0)
    assert np.isnan(initial._component_state("SRC").sent.get("flux")).all()


def test_replacing_monthly_records_preserves_outgoing_selection_shape() -> None:
    grid = RectilinearGrid("wet", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0))
    source = DataComponent(
        "SRC",
        grid,
        {"value": jnp.ones((12, 2, 2))},
        spec=ComponentSpec(transfer=TransferPolicy(time_selection="linear")),
    )
    target = CallableComponent(
        "DST", grid, lambda f: {}, spec=ComponentSpec(inputs=("value",))
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1, calendar="noleap"),
        components=(source, target),
        run_order=("SRC", "DST"),
        exchanges=(Exchange("SRC", "DST", ("value",), regridder_factory=bilinear),),
    )
    initial = coupler.initial_state()
    updated = initial.replace_fields("SRC", {"value": jnp.full((12, 2, 2), 3.0)})
    np.testing.assert_array_equal(
        updated._component_state("SRC").sent.get("value"), 1.0
    )
    final = coupler.run(updated)
    np.testing.assert_array_equal(final.component("DST").field("value"), 3.0)


def test_omitted_source_retains_mask() -> None:
    workflow = StaticWorkflow((StepPlan(0, ("AFTER",)), StepPlan(1, ("AFTER",))))
    final = _coupler(workflow=workflow).run()
    np.testing.assert_allclose(final.component("AFTER").field("total"), 1.0)
    np.testing.assert_allclose(final._fractional_mask("SRC->AFTER"), 0.25)


def test_binary_reactivation_requires_finite_exports() -> None:
    grid = RectilinearGrid(
        "wet", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    source = CallableComponent(
        "SRC",
        grid,
        lambda fields: {"active": jnp.ones_like(fields["active"])},
        spec=ComponentSpec(
            outputs=("active", "flux"),
            initial_fields={"active": 0.0, "flux": jnp.nan},
            masks=ComponentMaskSpec(binary_field="active"),
            execution="host",
        ),
    )
    target = CallableComponent(
        "DST", grid, lambda fields: {}, spec=ComponentSpec(inputs=("flux",))
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(source, target),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                fields=("flux",),
                quantity="source_flux",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    initial = coupler.initial_state()
    np.testing.assert_array_equal(initial._fractional_mask("SRC->DST"), 0.0)
    with pytest.raises(CouplerError, match="SRC.*flux"):
        coupler.run(initial)


def test_binary_deactivation_neutralizes_missing_export() -> None:
    grid = RectilinearGrid(
        "wet", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )

    def step(fields: Any) -> Any:
        active = 1.0 - fields["active"]
        return {"active": active, "flux": jnp.where(active > 0, 2.0, jnp.nan)}

    source = CallableComponent(
        "SRC",
        grid,
        step,
        spec=ComponentSpec(
            outputs=("active", "flux"),
            initial_fields={"active": 1.0, "flux": 2.0},
            masks=ComponentMaskSpec(binary_field="active"),
        ),
    )
    target = CallableComponent(
        "DST", grid, lambda fields: {}, spec=ComponentSpec(inputs=("flux",))
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 2),
        components=(source, target),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("flux",),
                quantity="source_flux",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    final = coupler.run()
    np.testing.assert_allclose(final.component("DST").field("flux"), 2.0)
    np.testing.assert_allclose(final._fractional_mask("SRC->DST"), 1.0)


@pytest.mark.parametrize("binary,invalid", [(True, 0.5), (False, 1.5)])
def test_imported_mask_is_validated_before_author_step(
    binary: bool, invalid: float
) -> None:
    grid = RectilinearGrid(
        "wet", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    called: list[bool] = []

    def repair_mask(fields: Any) -> Any:
        called.append(True)
        return {"active": jnp.ones_like(fields["active"])}

    src = CallableComponent(
        "SRC",
        grid,
        lambda f: {},
        spec=ComponentSpec(outputs=("active",), initial_fields={"active": invalid}),
    )
    dst = CallableComponent(
        "DST",
        grid,
        repair_mask,
        spec=ComponentSpec(
            inputs=("active",),
            outputs=("active",),
            initial_fields={"active": 1.0},
            execution="host",
            masks=(
                ComponentMaskSpec(binary_field="active")
                if binary
                else ComponentMaskSpec(fractional_field="active")
            ),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(src, dst),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("active",),
                quantity="intensive",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("DST",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    with pytest.raises(CouplerError, match="DST.*active"):
        coupler.run()
    assert not called


@pytest.mark.parametrize("execution", ["host", "jax"])
@pytest.mark.parametrize("receive", [False, True])
def test_source_reactivation_preserves_stored_input_support(
    execution: str, receive: bool
) -> None:
    grid = RectilinearGrid(
        "wet", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    src = CallableComponent(
        "SRC",
        grid,
        lambda f: {
            "active": jnp.ones_like(f["active"]),
            "flux": jnp.full_like(f["flux"], 2.0),
        },
        spec=ComponentSpec(
            outputs=("active", "flux"),
            initial_fields={"active": 0.0, "flux": jnp.nan},
            masks=ComponentMaskSpec(binary_field="active"),
            execution=cast(Any, execution),
        ),
    )
    dst = CallableComponent(
        "DST",
        grid,
        lambda f: {"total": f["total"] + f["flux"]},
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("total",),
            initial_fields={"flux": jnp.nan, "total": 0.0},
            execution=cast(Any, execution),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(src, dst),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("flux",),
                quantity="source_flux",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("SRC", "DST") if receive else ("SRC",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    final = coupler.run()
    np.testing.assert_allclose(
        final.component("DST").field("total"), 2.0 if receive else 0.0
    )


@pytest.mark.parametrize(
    "bad", [jnp.zeros((1, 2)), jnp.ones((2, 2), dtype=jnp.int32), jnp.full((2, 2), 1.5)]
)
def test_supplied_received_support_must_preserve_schema_and_bounds(
    bad: jax.Array,
) -> None:
    coupler = _coupler()
    state = coupler.initial_state()
    state = state._with_received_masks(
        FieldStore.from_mapping(
            {**state._received_fractional_masks.to_mapping(), "SRC->BEFORE": bad}
        )
    )
    with pytest.raises(CouplerError, match="Runtime received mask"):
        coupler.run(state)


def test_final_output_uses_final_masks(tmp_path: Path) -> None:
    final = _coupler().run(
        output=OutputTarget(tmp_path, write_period=False, write_snapshots=False)
    )
    files = tuple(tmp_path.glob("after.runtime_fields.nc"))
    assert len(files) == 1
    with xr.open_dataset(files[0]) as ds:
        np.testing.assert_allclose(ds["fmask_SRC_AFTER"], 0.75)
        np.testing.assert_array_equal(ds["bmask_SRC_AFTER"], 1.0)
    np.testing.assert_allclose(final._fractional_mask("SRC->AFTER"), 0.75)


def test_fractional_mask_rollout_gradients() -> None:
    coupler = _coupler()
    initial = coupler.initial_state()

    def objective(area: jax.Array) -> jax.Array:
        state = initial.replace_fields("SRC", {"area": jnp.full((2, 2), area)})
        final = coupler.run(state)
        # Include final coverage to verify the cache remains traced as well.
        return jnp.mean(final.component("AFTER").field("total")) + jnp.mean(
            final._fractional_mask("SRC->AFTER")
        )

    x = jnp.array(0.1)
    value, tangent = jax.jvp(jax.jit(objective), (x,), (jnp.ones_like(x),))
    reverse = jax.grad(objective)(x)
    finite_difference = (objective(x + 1e-5) - objective(x - 1e-5)) / 2e-5
    np.testing.assert_allclose(value, 2.5)
    np.testing.assert_allclose([tangent, reverse, finite_difference], 5.0, rtol=1e-9)


def test_ice_free_receiver_keeps_intensive_forcing() -> None:
    grid = RectilinearGrid(
        "wet", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    atmosphere = CallableComponent(
        "ATM",
        grid,
        lambda f: {},
        spec=ComponentSpec(
            outputs=("temperature",), initial_fields={"temperature": 270.0}
        ),
    )
    ice = CallableComponent(
        "CICE",
        grid,
        lambda f: {},
        spec=ComponentSpec(
            inputs=("temperature",),
            outputs=("area",),
            initial_fields={"area": 0.0},
            masks=ComponentMaskSpec(fractional_field="area"),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(atmosphere, ice),
        exchanges=(
            Exchange(
                "ATM",
                "CICE",
                fields=("temperature",),
                quantity="intensive",
                regridder_factory=bilinear,
            ),
        ),
        run_order=("CICE",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    np.testing.assert_allclose(
        coupler.run().component("CICE").field("temperature"), 270.0
    )


@pytest.mark.parametrize("execution", ["host", "jax"])
@pytest.mark.parametrize("omit_source", [False, True])
def test_binary_coverage_retains_last_dispatch_support(
    execution: str, omit_source: bool
) -> None:
    grid = RectilinearGrid("wet", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0))
    source = CallableComponent(
        "SRC",
        grid,
        lambda f: {"area": jnp.zeros_like(f["area"])},
        spec=ComponentSpec(
            outputs=("area", "flux"),
            initial_fields={"area": 0.5, "flux": 12.0},
            masks=ComponentMaskSpec(fractional_field="area"),
            execution=cast(Any, execution),
        ),
    )
    target = CallableComponent(
        "DST",
        grid,
        lambda f: {"total": f["total"] + f["flux"]},
        spec=ComponentSpec(
            inputs=("flux",),
            outputs=("total",),
            initial_fields={"total": 0.0},
            execution=cast(Any, execution),
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(source, target),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("flux",),
                quantity="source_flux",
                source_coverage="binary",
            ),
        ),
        run_order=("DST",) if omit_source else ("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
    )
    initial = coupler.initial_state()
    updated = initial.replace_fields(
        "SRC", {"area": jnp.zeros((2, 2)), "flux": jnp.full((2, 2), 14.0)}
    )
    final = coupler.run(updated)
    np.testing.assert_array_equal(final.component("DST").field("total"), 14.0)
    np.testing.assert_array_equal(final._fractional_mask("SRC->DST"), 1.0)
    np.testing.assert_array_equal(final._received_fractional_masks.get("SRC->DST"), 1.0)
    np.testing.assert_array_equal(
        coupler.run(final).component("DST").field("total"), 28.0
    )
    np.testing.assert_array_equal(initial.component("SRC").field("area"), 0.5)
