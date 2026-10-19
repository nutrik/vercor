"""Check area weighting before remapping and explicit quantity semantics."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._exchange_case_data import QUANTITY_CASES, COVERAGE_GRADIENT_CASES
from tests._runtime_helpers import prepared_coupling
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import CallableComponent, ComponentMaskSpec, ComponentSpec
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
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
    source_coverage: Any = "component",
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
        source_coverage=source_coverage,
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
        context=context,
    )


@pytest.mark.parametrize(
    "quantity,expected",
    QUANTITY_CASES,
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


@pytest.mark.parametrize("quantity,scale", QUANTITY_CASES)
def test_vector_quantities_preserve_rotation_and_weights(
    quantity: str, scale: float
) -> None:
    state, exchange, context = _scenario(quantity, vectors=True)
    view = _dispatch(state, exchange, context).component("DST")
    np.testing.assert_allclose(view.field("u", scope="received"), 2 * scale)
    np.testing.assert_allclose(view.field("v", scope="received"), scale)


@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize(
    "quantity,area_gradient,target_gradient",
    COVERAGE_GRADIENT_CASES,
)
def test_nonuniform_coverage_normalization_and_rotation_gradients(
    quantity: str, area_gradient: list[float], target_gradient: float, vectors: bool
) -> None:
    """Check derivatives independently of the runtime's mask implementation."""
    state, route, context = _scenario(quantity, vectors=vectors)
    area = jnp.array([[0.25, 0.75], [0.25, 0.75]])
    target = jnp.full((2, 2), 0.25)

    def objective(a: jax.Array, t: jax.Array) -> jax.Array:
        current = state.replace_fields("SRC", {"area": a}).replace_fields(
            "DST", {"area": t}
        )
        return jnp.mean(
            _dispatch(current, route, context)
            .component("DST")
            .field("u" if vectors else "flux", scope="received")
        )

    compiled = jax.jit(objective)
    actual_a, actual_t = jax.grad(compiled, argnums=(0, 1))(area, target)
    factor = 2.0 if vectors else 1.0
    np.testing.assert_allclose(
        actual_a, factor * np.tile(area_gradient, (2, 1)), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        actual_t, factor * target_gradient, rtol=1e-12, atol=1e-12
    )
    direction = (jnp.ones_like(area), jnp.full_like(target, 0.5))
    _, forward = jax.jvp(compiled, (area, target), direction)
    reverse = jnp.sum(actual_a) + 0.5 * jnp.sum(actual_t)
    np.testing.assert_allclose(forward, reverse, rtol=1e-12, atol=1e-12)
    h = 1e-5
    finite_difference = (
        compiled(area + h, target + 0.5 * h) - compiled(area - h, target - 0.5 * h)
    ) / (2 * h)
    np.testing.assert_allclose(forward, finite_difference, rtol=1e-8, atol=1e-8)


def test_intensive_forcing_does_not_vanish_on_ice_free_target() -> None:
    state, exchange, context = _scenario("intensive", target_area=0.0)
    np.testing.assert_allclose(
        _dispatch(state, exchange, context)
        .component("DST")
        .field("flux", scope="received"),
        8.0,
    )


@pytest.mark.parametrize("quantity", ["intensive", "source_flux", "target_flux"])
@pytest.mark.parametrize("vectors", [False, True])
def test_empty_source_coverage_returns_finite_zero(
    quantity: str, vectors: bool
) -> None:
    state, exchange, context = _scenario(
        quantity, source_area=jnp.zeros((2, 2)), vectors=vectors
    )
    names = ("u", "v") if vectors else ("flux",)
    source = state._component_state("SRC")
    state = state._with_component_state(
        "SRC",
        source.with_sent(
            source.sent.replace_many(
                {name: jnp.full((2, 2), jnp.nan) for name in names}
            )
        ),
    )
    value = (
        _dispatch(state, exchange, context)
        .component("DST")
        .field(names[0], scope="received")
    )
    np.testing.assert_array_equal(value, 0.0)

    def objective(area: jax.Array) -> jax.Array:
        current = state.replace_fields("SRC", {"area": area})
        return jnp.sum(
            _dispatch(current, exchange, context)
            .component("DST")
            .field(names[0], scope="received")
        )

    compiled = jax.jit(objective)
    zero = jnp.zeros((2, 2))
    output, forward = jax.jvp(compiled, (zero,), (jnp.ones_like(zero),))
    np.testing.assert_array_equal(output, 0.0)
    np.testing.assert_array_equal(forward, 0.0)
    np.testing.assert_array_equal(jax.grad(compiled)(zero), 0.0)


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
            return ExchangeTopologyPatch(  # type: ignore[call-arg]
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
    with pytest.raises(TypeError, match="fractional_masks"):
        _scenario("intensive", topology=_Patch(fractional=True))


def test_all_routes_require_explicit_quantity() -> None:
    with pytest.raises(TypeError, match="quantity"):
        Exchange("SRC", "DST", ("flux",))  # type: ignore[call-arg]


def test_topology_cannot_expand_target_computational_support() -> None:
    class ExpandingPatch:
        def build(self, context: Any) -> ExchangeTopologyPatch:
            return ExchangeTopologyPatch(binary_masks={"SRC->DST": jnp.ones((2, 2))})

    with pytest.raises(CouplerError, match="SRC->DST.*support"):
        _scenario(
            "intensive",
            target_wet=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            topology=ExpandingPatch(),
        )


def test_explicit_dispatch_requires_its_runtime_context() -> None:
    state, exchange, context = _scenario("intensive")
    with pytest.raises(TypeError, match="context"):
        dispatch_component_exchanges(state, "DST", (exchange,))  # type: ignore[call-arg]


@pytest.mark.parametrize("quantity", ["mystery", "legacy"])
def test_unknown_exchange_quantity_is_rejected(quantity: str) -> None:
    with pytest.raises(ValueError, match="quantity"):
        _scenario(quantity)


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


@pytest.mark.parametrize("area", [0.0, 0.5, 1.0])
def test_binary_source_flux_survives_zero_ice(area: float) -> None:
    state, route, context = _scenario(
        "source_flux", source_area=jnp.full((2, 2), area), source_coverage="binary"
    )
    state = state.replace_fields("SRC", {"flux": jnp.full((2, 2), 12.0)})
    result = _dispatch(state, route, context)
    np.testing.assert_allclose(
        result.component("DST").field("flux", scope="received"), 12.0
    )
    np.testing.assert_allclose(
        result._received_fractional_masks.get(route.route_id), 1.0
    )
    # Default source coverage still applies the component's continuous fraction.
    original, default, default_context = _scenario(
        "source_flux", source_area=jnp.full((2, 2), area)
    )
    original = original.replace_fields("SRC", {"flux": jnp.full((2, 2), 12.0)})
    np.testing.assert_allclose(
        _dispatch(original, default, default_context)
        .component("DST")
        .field("flux", scope="received"),
        12.0 * area,
    )


@pytest.mark.parametrize("value", [None, 0, True])
def test_source_coverage_must_be_text(value: Any) -> None:
    with pytest.raises(TypeError, match="source_coverage"):
        _scenario("source_flux", source_coverage=value)


@pytest.mark.parametrize("quantity,coverage", [("source_flux", "ice")])
def test_invalid_source_coverage_is_rejected(quantity: str, coverage: str) -> None:
    with pytest.raises(ValueError, match="source_coverage"):
        _scenario(quantity, source_coverage=coverage)


@pytest.mark.parametrize("boundary", ["send", "supplied", "dispatch"])
def test_binary_send_requires_finite_wet_flux(boundary: str) -> None:
    from vercor._runtime.field_transfer import send_runtime_fields
    from vercor._runtime.state_validation import validate_runtime_state

    state, route, context = _scenario(
        "source_flux", source_area=jnp.zeros((2, 2)), source_coverage="binary"
    )
    source = state._component_state("SRC")
    bad = jnp.full((2, 2), jnp.nan)
    state = state._with_component_state(
        "SRC", source.with_fields(source.fields.replace_many({"flux": bad}))
    )
    with pytest.raises(CouplerError, match="SRC.*flux"):
        if boundary == "send":
            send_runtime_fields(
                context.components["SRC"],
                state._component_state("SRC"),
                contract=context.contracts["SRC"],
                dtype=context.dtype,
            )
        elif boundary == "supplied":
            validate_runtime_state(
                state,
                components=context.components,
                exchanges=context.exchanges,
                contracts=context.contracts,
                regridders=context.regridders,
                run_order=("SRC", "DST"),
            )
        else:
            source = state._component_state("SRC")
            state = state._with_component_state(
                "SRC", source.with_sent(source.sent.replace_many({"flux": bad}))
            )
            _dispatch(state, route, context)


@pytest.mark.parametrize("vectors", [False, True])
def test_binary_coverage_preserves_remap_rotation_and_topology(vectors: bool) -> None:
    state, route, context = _scenario(
        "source_flux",
        source_coverage="binary",
        vectors=vectors,
        source_area=jnp.zeros((2, 2)),
        topology=_Patch(),
    )
    view = _dispatch(state, route, context).component("DST")
    for name, scale in (("u", 12.0), ("v", 6.0)) if vectors else (("flux", 6.0),):
        np.testing.assert_allclose(
            view.field(name, scope="received"), [[0.0, scale], [0.0, scale]]
        )


def test_binary_conservative_flux_and_gradient_ignore_fraction() -> None:
    state, route, context = _scenario(
        "source_flux", source_coverage="binary", conservative_case=True
    )
    np.testing.assert_allclose(
        _dispatch(state, route, context)
        .component("DST")
        .field("flux", scope="received"),
        6.0,
    )

    def loss(x: jax.Array) -> jax.Array:
        current = state.replace_fields(
            "SRC", {"flux": jnp.full((2, 4), x), "area": jnp.full((2, 4), 0.0)}
        )
        return jnp.mean(
            _dispatch(current, route, context)
            .component("DST")
            .field("flux", scope="received")
        )

    x = jnp.array(2.0)
    value, tangent = jax.jvp(jax.jit(loss), (x,), (jnp.ones_like(x),))
    np.testing.assert_allclose([value, tangent, jax.grad(loss)(x)], [2.0, 1.0, 1.0])
    np.testing.assert_allclose((loss(x + 1e-4) - loss(x - 1e-4)) / 2e-4, 1.0, rtol=1e-9)


def test_binary_coverage_empty_support_neutralizes_inactive_nan() -> None:
    state, route, context = _scenario(
        "source_flux",
        source_coverage="binary",
        source_wet=jnp.zeros((2, 2)),
        source_area=jnp.zeros((2, 2)),
        regridder_factory=bilinear,
    )
    state = state.replace_fields("SRC", {"flux": jnp.full((2, 2), jnp.nan)})
    result = _dispatch(state, route, context)
    np.testing.assert_array_equal(
        result.component("DST").field("flux", scope="received"), 0.0
    )


def test_binary_contract_metadata_survives_incoming_and_self_routes() -> None:
    from vercor._runtime.contracts import build_exchange_contracts

    routes = (
        Exchange(
            "SRC", "MID", ("a",), quantity="source_flux", source_coverage="binary"
        ),
        Exchange(
            "MID", "DST", ("b",), quantity="source_flux", source_coverage="binary"
        ),
        Exchange("MID", "MID", ("c",), quantity="intensive"),
    )
    contract = build_exchange_contracts(("SRC", "MID", "DST"), routes)["MID"]
    assert contract.receives == ("a", "c")
    assert contract.sends == ("b", "c")
    assert contract.binary_sends == ("b",)


def test_binary_export_requires_full_support_without_component_mask() -> None:
    from vercor._runtime.component_masks import field_active_mask

    grid = RectilinearGrid("relay", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0))
    relay = CallableComponent(
        "RELAY",
        grid,
        lambda f: {},
        spec=ComponentSpec(inputs=("flux",), outputs=("flux",)),
    )
    # A relay may have received nothing in its last dispatch. Sending that field
    # on a total-area route still requires valid values across its wet domain.
    active = field_active_mask(
        relay,
        "flux",
        default_mask=jnp.zeros((2, 2)),
        fractional_mask=jnp.ones((2, 2)),
        produced=True,
        binary_mask=jnp.ones((2, 2)),
        binary_output_fields=("flux",),
    )
    np.testing.assert_array_equal(active, 1.0)
