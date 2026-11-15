"""Audit shared and self-route coverage, differentiation and failure ordering."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._runtime_helpers import prepared_coupling
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import CallableComponent, ComponentMaskSpec, ComponentSpec
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor._runtime import exchange_masks
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor.state import RunState

pytestmark = pytest.mark.fast_always


class _MeanRegridder:
    """Expose coverage remapping even for a self-route on the same grid."""

    def __init__(
        self, source_grid: RectilinearGrid, target_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = False

    def regrid(self, values: Any) -> jax.Array:
        return jnp.broadcast_to(jnp.mean(values), self.target_grid.shape)


def _shared_routes(enable_x64: bool) -> tuple[RunState, RuntimeDispatchContext]:
    policy = DTypePolicy(enable_x64=enable_x64)
    components = []
    for name in ("SRC", "B", "C", "D"):
        grid = RectilinearGrid.uniform(
            name, nlon=2, nlat=2, longitude=(0, 360), latitude=(-90, 90)
        )
        components.append(
            CallableComponent(
                name,
                grid,
                lambda fields: {},
                spec=ComponentSpec(
                    inputs=("flux",),
                    outputs=("area", "flux") if name == "SRC" else ("area",),
                    initial_fields={
                        "area": 0.5 if name == "SRC" else 0.25,
                        "flux": 2.0,
                    },
                    masks=ComponentMaskSpec(fractional_field="area"),
                ),
            )
        )
    routes = tuple(
        Exchange(
            "SRC",
            name,
            ("flux",),
            quantity=cast(Any, quantity),
            source_coverage="binary" if name == "C" else "component",
            regridder_factory=_MeanRegridder,
        )
        for name, quantity in (
            ("SRC", "intensive"),
            ("B", "intensive"),
            ("C", "source_flux"),
            ("D", "target_flux"),
        )
    )
    coupler = Coupler(
        Clock(datetime(2001, 1, 1), 60.0, 1),
        components=components,
        exchanges=routes,
        run_order=("SRC", "B", "C", "D"),
        runtime=RuntimeOptions(dtype=policy),
        log_level="ERROR",
    )
    state = coupler.initial_state()
    context = replace(
        prepared_coupling(coupler).dispatch_context,
        topology_binary_masks={
            "SRC->B": jnp.asarray([[1, 0], [1, 0]], dtype=policy.jax_real)
        },
    )
    return state, context


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("entrypoint", ["refresh", "diagnostics"])
def test_refresh_preserves_shared_and_self_route_policies(
    monkeypatch: pytest.MonkeyPatch, enable_x64: bool, entrypoint: str
) -> None:
    """Changing source support must retain route policies and received support."""
    state, context = _shared_routes(enable_x64)
    remapped: list[str] = []
    original_remap = _MeanRegridder.regrid

    def remap(self: _MeanRegridder, values: Any) -> jax.Array:
        remapped.append(self.target_grid.name)
        return original_remap(self, values)

    monkeypatch.setattr(_MeanRegridder, "regrid", remap)
    for fraction in (0.5, 0.75):
        updated = state.replace_fields(
            "SRC", {"area": jnp.full((2, 2), fraction, dtype=context.dtype.jax_real)}
        )
        if entrypoint == "refresh":
            result = exchange_masks.refresh_runtime_masks(updated, context=context)
            fractions = {
                route.route_id: result._fractional_masks.get(route.route_id)
                for route in context.exchanges
            }
            for route in context.exchanges:
                np.testing.assert_array_equal(
                    result._received_fractional_masks.get(route.route_id),
                    updated._received_fractional_masks.get(route.route_id),
                )
        else:
            _, fractions = exchange_masks.runtime_exchange_masks(
                updated, context=context
            )
        np.testing.assert_array_equal(fractions["SRC->SRC"], jnp.full((2, 2), fraction))
        np.testing.assert_array_equal(
            fractions["SRC->B"], [[fraction, 0], [fraction, 0]]
        )
        np.testing.assert_array_equal(fractions["SRC->C"], jnp.ones((2, 2)))
        np.testing.assert_array_equal(fractions["SRC->D"], jnp.full((2, 2), 0.25))
        assert remapped == ["SRC", "B", "C", "D"]
        remapped.clear()


@pytest.mark.parametrize("enable_x64", [False, True])
def test_refresh_preserves_jit_jvp_and_reverse_derivatives(
    enable_x64: bool,
) -> None:
    """The shared source fraction remains a dynamic differentiable operand."""
    state, context = _shared_routes(enable_x64)

    def objective(fraction: jax.Array) -> jax.Array:
        updated = state.replace_fields("SRC", {"area": jnp.full((2, 2), fraction)})
        refreshed = exchange_masks.refresh_runtime_masks(updated, context=context)
        total = jnp.asarray(0.0, dtype=context.dtype.jax_real)
        for route in context.exchanges:
            total = total + jnp.sum(refreshed._fractional_masks.get(route.route_id))
        return total

    value = jnp.asarray(0.5, dtype=context.dtype.jax_real)
    primal, tangent = jax.jvp(objective, (value,), (jnp.ones_like(value),))
    np.testing.assert_array_equal(primal, 8.0)
    np.testing.assert_array_equal(tangent, 6.0)
    np.testing.assert_array_equal(jax.grad(objective)(value), 6.0)
    np.testing.assert_array_equal(jax.jit(objective)(value), 8.0)


def test_earlier_invalid_remap_prevents_later_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local reuse cannot eagerly evaluate a later route ahead of a failure."""
    state, context = _shared_routes(True)
    observed: list[str] = []

    def invalid(self: _MeanRegridder, values: Any) -> jax.Array:
        observed.append(self.target_grid.name)
        return jnp.full(self.target_grid.shape, jnp.inf)

    monkeypatch.setattr(_MeanRegridder, "regrid", invalid)
    with pytest.raises(CouplerError, match="remapped coverage.*infinity"):
        exchange_masks.refresh_runtime_masks(state, context=context)
    assert observed == ["SRC"]
