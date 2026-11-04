"""Quantity dispatch through real component contracts and controlled remappers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from tests._runtime_helpers import prepared_coupling
from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import ComponentMaskSpec, ComponentSpec, DataComponent
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor.fields import vector
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.exchange_dispatch import dispatch_component_exchanges
from vercor.physics import PhysicalConstants
from vercor.state import RunState


class _ScalingRegridder:
    """Linear scalar scaling and 90-degree vector rotation."""

    def __init__(self, source_grid: Any, target_grid: Any, scale: float = 1.0):
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = source_grid is target_grid
        self.scale = scale

    def regrid(self, field: Any) -> jax.Array:
        return self.scale * jnp.asarray(field)

    def regrid_vector(self, u: Any, v: Any) -> tuple[jax.Array, jax.Array]:
        return self.regrid(-v), self.regrid(u)


def _factory(source_grid: Any, target_grid: Any) -> _ScalingRegridder:
    return _ScalingRegridder(source_grid, target_grid)


def _case(
    *, vectors: bool = False, quantity: Any = "source_flux"
) -> tuple[RunState, Exchange, RuntimeDispatchContext]:
    grid = RectilinearGrid(
        "test", longitude=jnp.array([0.0, 1.0]), latitude=jnp.array([0.0, 1.0])
    )
    names = ("u_velocity", "v_velocity") if vectors else ("temperature",)
    source = DataComponent(
        "OCN",
        grid,
        {name: jnp.ones((2, 2)) for name in (*names, "area")},
        spec=ComponentSpec(
            outputs=(*names, "area"), masks=ComponentMaskSpec(fractional_field="area")
        ),
    )
    target = DataComponent(
        "ATM",
        grid,
        {"area": jnp.full((2, 2), 0.25)},
        spec=ComponentSpec(
            inputs=names,
            outputs=("area",),
            masks=ComponentMaskSpec(fractional_field="area"),
        ),
    )
    route = Exchange(
        "OCN",
        "ATM",
        (vector(*names),) if vectors else names,
        quantity=quantity,
        regridder_factory=_factory,
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60, 1),
        components=(source, target),
        exchanges=(route,),
        run_order=("OCN", "ATM"),
        runtime=RuntimeOptions(topology=None),
        log_level="error",
    )
    return coupler.initial_state(), route, prepared_coupling(coupler).dispatch_context


def _dispatch(
    state: RunState, route: Exchange, context: RuntimeDispatchContext
) -> RunState:
    return dispatch_component_exchanges(state, "ATM", (route,), context=context)


def test_dispatch_component_exchanges_handles_scalar_masks_and_gradients() -> None:
    state, route, context = _case()
    context = replace(
        context,
        regridders={
            route.route_id: _ScalingRegridder(
                context.components["OCN"].grid,
                context.components["ATM"].grid,
                scale=2.0,
            )
        },
    )

    def loss(source: jax.Array, mask: jax.Array) -> jax.Array:
        current = state.replace_fields("OCN", {"temperature": source, "area": mask})
        return jnp.sum(
            _dispatch(current, route, context)
            .component("ATM")
            .field("temperature", scope="received")
        )

    source = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.array([[1.0, 0.5], [0.0, 1.0]])
    value = loss(source, mask)
    dvalue, dmask = jax.grad(loss, argnums=(0, 1))(source, mask)
    assert_allclose_compact(value, 12.0)
    assert_allclose_compact(dvalue, [[2.0, 1.0], [0.0, 2.0]])
    assert_allclose_compact(dmask, [[2.0, 4.0], [0.0, 8.0]])
    assert_allclose_compact(jax.jit(loss)(source, mask), value)


def test_dispatch_component_exchanges_preserves_vector_regridding_behavior() -> None:
    state, route, context = _case(vectors=True, quantity="intensive")
    state = state.replace_fields(
        "OCN",
        {"u_velocity": jnp.full((2, 2), 5.0), "v_velocity": jnp.full((2, 2), -2.0)},
    )
    result = _dispatch(state, route, context).component("ATM")
    assert_allclose_compact(result.field("u_velocity", scope="received"), 2.0)
    assert_allclose_compact(result.field("v_velocity", scope="received"), 5.0)


def test_runtime_dispatch_context_groups_exchanges_by_destination() -> None:
    atm_exchange = Exchange(
        quantity="intensive",
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=_factory,
    )
    land_exchange = Exchange(
        quantity="intensive",
        source="ATM",
        target="LND",
        fields=["temperature"],
        regridder_factory=_factory,
    )

    context = RuntimeDispatchContext(
        components={},
        exchanges=(atm_exchange, land_exchange),
        exchanges_by_destination={
            "ATM": (atm_exchange,),
            "LND": (land_exchange,),
        },
        regridders={},
        contracts={},
        dt_seconds=60.0,
        constants=PhysicalConstants(),
        dtype=DTypePolicy(),
    )
    assert context.exchanges == (atm_exchange, land_exchange)

    assert context.destination_exchanges("ATM") == (atm_exchange,)
    assert context.destination_exchanges("OCN") == ()


@pytest.mark.parametrize("vectors", [False, True])
def test_exchange_neutralizes_inactive_nan_without_poisoning_jvp_or_vjp(
    vectors: bool,
) -> None:
    state, route, context = _case(vectors=vectors)
    names = ("u_velocity", "v_velocity") if vectors else ("temperature",)
    mask = jnp.array([[1.0, 0.0], [1.0, 0.0]])

    def transfer(source: jax.Array) -> jax.Array:
        current = state.replace_fields(
            "OCN", {**{name: source for name in names}, "area": mask}
        )
        return jnp.asarray(
            _dispatch(current, route, context)
            .component("ATM")
            .field(names[0], scope="received")
        )

    value = jnp.array([[2.0, jnp.nan], [2.0, jnp.nan]])
    assert_finite_jvp_vjp(lambda x: jnp.sum(transfer(x)), value, mask)
    expected = [[-2.0, 0.0], [-2.0, 0.0]] if vectors else [[2.0, 0.0], [2.0, 0.0]]
    assert_allclose_compact(transfer(value), expected, equal_nan=False)


@pytest.mark.parametrize("vectors", [False, True])
def test_exchange_rejects_nonfinite_active_regridding_output(vectors: bool) -> None:
    state, route, context = _case(vectors=vectors)
    names = ("u_velocity", "v_velocity") if vectors else ("temperature",)

    class InvalidRegridder(_ScalingRegridder):
        def regrid(self, field: Any) -> jax.Array:
            # Coverage (ones) stays valid; corrupted transferred values do not.
            return jnp.where(jnp.abs(field) > 1.0, jnp.nan, field)

    context = replace(
        context,
        regridders={
            route.route_id: InvalidRegridder(
                context.components["OCN"].grid, context.components["ATM"].grid
            )
        },
    )
    state = state.replace_fields("OCN", {name: jnp.full((2, 2), 2.0) for name in names})
    with pytest.raises(CouplerError, match="OCN->ATM.*regridded.*active domain"):
        _dispatch(state, route, context)
