"""Check area weighting before remapping and explicit quantity semantics."""

from __future__ import annotations

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
from vercor.exceptions import CouplerError, ExchangeError
from vercor.fields import vector
from vercor.regridding import bilinear, conservative
from vercor.state import RunState
from vercor.topology import ExchangeTopologyPatch
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.exchange_dispatch import dispatch_component_exchanges

pytestmark = pytest.mark.fast_always


class _MeanRegridder:
    """A linear equal-area averaging operator with a 90 degree vector rotation."""

    def __init__(
        self, source_grid: RectilinearGrid, target_grid: RectilinearGrid
    ) -> None:
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = False

    def regrid(self, field: Any) -> jax.Array:
        return jnp.broadcast_to(jnp.mean(field), self.target_grid.shape)

    def regrid_vector(self, u: Any, v: Any) -> tuple[jax.Array, jax.Array]:
        return self.regrid(-v), self.regrid(u)


def _scenario(
    quantity: Any,
    *,
    vectors: bool = False,
    source_area: jax.Array | None = None,
    target_area: float = 0.25,
    topology: Any = None,
    conservative_case: bool = False,
    target_wet: jax.Array | None = None,
    source_wet: jax.Array | None = None,
    regridder_factory: Any = None,
) -> tuple[RunState, Exchange, RuntimeDispatchContext]:
    source_lon = [-135.0, -45.0, 45.0, 135.0] if conservative_case else [0.0, 1.0]
    target_lon = [-90.0, 90.0] if conservative_case else [0.25, 0.75]
    source = RectilinearGrid(
        "source",
        longitude=jnp.array(source_lon),
        latitude=jnp.array([-45.0, 45.0]),
        binary_mask=source_wet,
    )
    target = RectilinearGrid(
        "target",
        longitude=jnp.array(target_lon),
        latitude=jnp.array([-45.0, 45.0]),
        binary_mask=target_wet,
    )
    if source_area is None:
        source_area = jnp.tile(jnp.array([0.25, 0.75]), (2, len(source_lon) // 2))
    flux = jnp.tile(jnp.array([2.0, 10.0]), (2, len(source_lon) // 2))
    src = CallableComponent(
        "SRC",
        source,
        lambda fields: {},
        spec=ComponentSpec(
            outputs=("area", "flux", "u", "v"),
            initial_fields={
                "area": source_area,
                "flux": flux,
                "u": flux,
                "v": -2 * flux,
            },
            masks=ComponentMaskSpec(fractional_field="area"),
        ),
    )
    dst = CallableComponent(
        "DST",
        target,
        lambda fields: {},
        spec=ComponentSpec(
            inputs=("flux", "u", "v"),
            outputs=("area",),
            initial_fields={
                "area": target_area if target_wet is None else target_area * target_wet
            },
            masks=ComponentMaskSpec(fractional_field="area"),
        ),
    )
    exchange = Exchange(
        "SRC",
        "DST",
        fields=(vector("u", "v"),) if vectors else ("flux",),
        quantity=quantity,
        regridder_factory=regridder_factory
        or (conservative if conservative_case else _MeanRegridder),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(src, dst),
        exchanges=(exchange,),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True), topology=topology),
    )
    state = coupler.initial_state()
    return state, exchange, prepared_coupling(coupler).dispatch_context


def _dispatch(
    state: RunState, exchange: Exchange, context: RuntimeDispatchContext
) -> RunState:
    return dispatch_component_exchanges(
        state,
        "DST",
        (exchange,),
        context.regridders,
        context=context,
    )


@pytest.mark.parametrize(
    "quantity,expected",
    [("intensive", 8.0), ("source_flux", 4.0), ("target_flux", 2.0)],
)
def test_quantity_weights_before_remapping(quantity: str, expected: float) -> None:
    state, exchange, context = _scenario(quantity)
    result = _dispatch(state, exchange, context)
    np.testing.assert_allclose(
        result.component("DST").field("flux", scope="received"), expected
    )
    np.testing.assert_array_equal(
        state.component("DST").field("flux", scope="received"), 0.0
    )


@pytest.mark.parametrize(
    "quantity,scale", [("intensive", 8.0), ("source_flux", 4.0), ("target_flux", 2.0)]
)
def test_vector_quantities_preserve_rotation_and_weights(
    quantity: str, scale: float
) -> None:
    state, exchange, context = _scenario(quantity, vectors=True)
    view = _dispatch(state, exchange, context).component("DST")
    np.testing.assert_allclose(view.field("u", scope="received"), 2 * scale)
    np.testing.assert_allclose(view.field("v", scope="received"), scale)


def test_intensive_forcing_does_not_vanish_on_ice_free_target() -> None:
    state, exchange, context = _scenario("intensive", target_area=0.0)
    np.testing.assert_allclose(
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received"),
        8.0,
    )


@pytest.mark.parametrize("quantity", ["intensive", "source_flux", "target_flux"])
def test_empty_source_coverage_returns_finite_zero(quantity: str) -> None:
    state, exchange, context = _scenario(quantity, source_area=jnp.zeros((2, 2)))
    source = state._component_state("SRC")
    state = state._with_component_state(
        "SRC",
        source.with_sent(source.sent.replace_many({"flux": jnp.full((2, 2), jnp.nan)})),
    )
    value = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received")
    )
    np.testing.assert_array_equal(value, 0.0)


def test_inactive_nan_is_neutralized_before_remapping_and_ad() -> None:
    state, exchange, context = _scenario(
        "intensive", source_area=jnp.array([[1.0, 0.0], [1.0, 0.0]])
    )

    def objective(x: jax.Array) -> jax.Array:
        source = state._component_state("SRC")
        flux = jnp.stack((jnp.stack((x, jnp.nan)), jnp.stack((x, jnp.nan))))
        current = state._with_component_state(
            "SRC", source.with_sent(source.sent.replace_many({"flux": flux}))
        )
        return jnp.mean(
            _dispatch(current, exchange, context)
            .component("DST")
            .field("flux", scope="received")
        )

    value, tangent = jax.jvp(jax.jit(objective), (jnp.array(2.0),), (jnp.array(1.0),))
    np.testing.assert_allclose(value, 2.0)
    np.testing.assert_allclose(tangent, 1.0)
    np.testing.assert_allclose(jax.grad(objective)(jnp.array(2.0)), 1.0)


def test_infinity_is_invalid_even_outside_source_coverage() -> None:
    state, exchange, context = _scenario("source_flux", source_area=jnp.zeros((2, 2)))
    source = state._component_state("SRC")
    state = state._with_component_state(
        "SRC",
        source.with_sent(source.sent.replace_many({"flux": jnp.full((2, 2), jnp.inf)})),
    )
    with pytest.raises(CouplerError, match="SRC->DST.*flux"):
        _dispatch(state, exchange, context)


def test_spherical_conservative_source_flux_preserves_integral() -> None:
    state, exchange, context = _scenario("source_flux", conservative_case=True)
    received = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received")
    )
    np.testing.assert_allclose(received, 4.0, rtol=1e-12)
    # Four destination cells each cover a quarter sphere. Source-weighted mean
    # flux is independently (0.25*2 + 0.75*10)/2 = 4 W/m2.
    np.testing.assert_allclose(
        jnp.sum(received) * np.pi * 6_371_000.0**2,
        4.0 * 4.0 * np.pi * 6_371_000.0**2,
        rtol=1e-12,
    )


def test_real_bilinear_remap_neutralizes_dry_target_nan() -> None:
    state, exchange, context = _scenario(
        "intensive",
        target_wet=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
        regridder_factory=bilinear,
    )
    value = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received")
    )
    np.testing.assert_allclose(value, [[0.0, 9.2], [0.0, 9.2]])


def test_real_bilinear_empty_source_support_returns_zero() -> None:
    state, exchange, context = _scenario(
        "intensive",
        source_wet=jnp.zeros((2, 2)),
        source_area=jnp.zeros((2, 2)),
        regridder_factory=bilinear,
    )
    value = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received")
    )
    np.testing.assert_array_equal(value, 0.0)


class _Patch:
    def __init__(self, *, fractional: bool = False) -> None:
        self.fractional = fractional

    def build(self, context: Any) -> ExchangeTopologyPatch:
        if self.fractional:
            return ExchangeTopologyPatch(
                fractional_masks={"SRC->DST": jnp.full((2, 2), 0.5)}
            )
        return ExchangeTopologyPatch(
            binary_masks={"SRC->DST": jnp.array([[0.0, 1.0], [0.0, 1.0]])}
        )


def test_binary_topology_patch_restricts_explicit_route() -> None:
    state, exchange, context = _scenario("intensive", topology=_Patch())
    value = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received")
    )
    np.testing.assert_allclose(value, [[0.0, 8.0], [0.0, 8.0]])


def test_conflicting_fractional_topology_patch_is_rejected() -> None:
    with pytest.raises(CouplerError, match="SRC->DST.*fractional"):
        _scenario("intensive", topology=_Patch(fractional=True))


def test_dynamic_masks_require_explicit_quantity() -> None:
    with pytest.raises(CouplerError, match="SRC->DST.*quantity"):
        _scenario("legacy")


def test_explicit_dispatch_requires_its_runtime_context() -> None:
    state, exchange, context = _scenario("intensive")
    with pytest.raises(ExchangeError, match="context"):
        dispatch_component_exchanges(state, "DST", (exchange,), context.regridders)


def test_unknown_exchange_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="quantity"):
        _scenario("mystery")


@pytest.mark.parametrize("quantity", [None, 0, True])
def test_exchange_quantity_must_be_text(quantity: Any) -> None:
    with pytest.raises(TypeError, match="quantity"):
        _scenario(quantity)


@pytest.mark.parametrize(
    "quantity,coverage",
    [("intensive", 0.5), ("source_flux", 0.5), ("target_flux", 0.25)],
)
def test_refresh_replaces_derived_coverage_without_mutating_state(
    quantity: str, coverage: float
) -> None:
    from vercor._runtime.exchange_masks import refresh_runtime_masks

    state, exchange, context = _scenario(quantity)
    updated = refresh_runtime_masks(state, context=context)
    np.testing.assert_allclose(updated._fractional_mask(exchange.route_id), coverage)
    np.testing.assert_allclose(state._fractional_mask(exchange.route_id), coverage)
    assert cast(
        Any, jax.tree_util.tree_structure(updated)
    ) == jax.tree_util.tree_structure(state)
    replaced = state.replace_fields("SRC", {"area": jnp.full((2, 2), 0.8)})
    refreshed = jax.jit(lambda x: refresh_runtime_masks(x, context=context))(replaced)
    np.testing.assert_allclose(
        refreshed._fractional_mask(exchange.route_id),
        0.25 if quantity == "target_flux" else 0.8,
    )


@pytest.mark.parametrize("target_area,binary", [(0.25, 1.0), (0.0, 0.0)])
def test_diagnostic_masks_match_quantity_coverage(
    target_area: float, binary: float
) -> None:
    from vercor._runtime.exchange_masks import runtime_exchange_masks

    state, exchange, context = _scenario("target_flux", target_area=target_area)
    binary_masks, fractions = runtime_exchange_masks(state, context=context)
    np.testing.assert_array_equal(binary_masks[exchange.route_id], binary)
    np.testing.assert_allclose(fractions[exchange.route_id], target_area)
