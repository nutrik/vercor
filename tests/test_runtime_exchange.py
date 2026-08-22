from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_finite_jvp_vjp
from vercor.exchanges import Exchange
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor.fields import vector
from vercor._runtime.dispatch_context import RuntimeDispatchContext
from vercor._runtime.exchange_dispatch import dispatch_component_exchanges
from vercor._runtime.state import ComponentRuntimeState
from vercor.state import RunState
from vercor._runtime.stores import FieldStore
from vercor.physics import PhysicalConstants


class _ScalingRegridder:
    def __init__(
        self,
        scale: float = 1.0,
        source_grid: Any = None,
        target_grid: Any = None,
    ) -> None:
        self.scale = scale
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.has_identical_grids = source_grid is target_grid

    def regrid(self, field: Any) -> Any:
        return jnp.asarray(field) * self.scale

    def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
        return jnp.asarray(u) + 1.0, jnp.asarray(v) - 1.0


class _NonFiniteRegridder:
    """Return prescribed scalar and vector values from a route regridder."""

    def __init__(
        self,
        scalar_result: jax.Array,
        vector_result: tuple[jax.Array, jax.Array] | None = None,
    ) -> None:
        self.scalar_result = scalar_result
        self.vector_result = vector_result

    def regrid(self, field: Any) -> jax.Array:
        _ = field
        return self.scalar_result

    def regrid_vector(self, u: Any, v: Any) -> tuple[jax.Array, jax.Array]:
        _ = u, v
        if self.vector_result is None:
            return self.scalar_result, self.scalar_result
        return self.vector_result


def _factory(*args: Any, **kwargs: Any) -> _ScalingRegridder:
    _ = kwargs
    source_grid, target_grid = args
    return _ScalingRegridder(
        source_grid=source_grid,
        target_grid=target_grid,
    )


def _component(
    name: str,
    *,
    sent: dict[str, jax.Array],
    received: dict[str, jax.Array] | None = None,
) -> ComponentRuntimeState:
    _ = name
    return ComponentRuntimeState(
        fields=FieldStore.from_mapping({}),
        received=FieldStore.from_mapping(received or {}),
        sent=FieldStore.from_mapping(sent),
    )


def _one_field_exchange_state(mask: jax.Array) -> tuple[RunState, Exchange]:
    """Build one scalar OCN-to-ATM exchange and its immutable state."""

    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=cast(Any, _factory),
    )
    state = RunState._from_runtime(
        component_names=("OCN", "ATM"),
        components=(
            _component("OCN", sent={"temperature": jnp.ones_like(mask)}),
            _component("ATM", sent={}, received={"temperature": jnp.zeros_like(mask)}),
        ),
        fractional_masks=FieldStore.from_mapping({exchange.route_id: mask}),
    )
    return state, exchange


def _dispatch_with_passthrough_regridder(
    source: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Dispatch a scalar field through an identity regridder for AD checks."""

    state, exchange = _one_field_exchange_state(mask)
    source_component = state._component_state("OCN").with_sent(
        FieldStore.from_mapping({"temperature": source})
    )
    state = state._with_component_state("OCN", source_component)
    dispatched = dispatch_component_exchanges(
        state,
        "ATM",
        (exchange,),
        {exchange.route_id: _ScalingRegridder()},
    )
    return dispatched._component_state("ATM").received.get("temperature")


def test_dispatch_component_exchanges_handles_scalar_masks_and_gradients() -> None:
    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=cast(Any, _factory),
    )
    regridders = {"OCN->ATM": _ScalingRegridder(scale=2.0)}

    def loss(source: jax.Array, mask: jax.Array) -> jax.Array:
        state = RunState._from_runtime(
            component_names=("OCN", "ATM"),
            components=(
                _component("OCN", sent={"temperature": source}),
                _component(
                    "ATM", sent={}, received={"temperature": jnp.zeros_like(source)}
                ),
            ),
            fractional_masks=FieldStore.from_mapping({"OCN->ATM": mask}),
        )
        dispatched = dispatch_component_exchanges(
            state,
            "ATM",
            (exchange,),
            regridders,
        )
        return jnp.sum(dispatched._component_state("ATM").received.get("temperature"))

    source = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.asarray([[1.0, 0.5], [0.0, 1.0]])

    out = loss(source, mask)
    grad_source, grad_mask = jax.grad(loss, argnums=(0, 1))(source, mask)

    assert_allclose_compact(out, np.asarray(12.0))
    assert_allclose_compact(grad_source, 2.0 * np.asarray(mask))
    assert_allclose_compact(
        grad_mask,
        2.0 * np.where(np.asarray(mask) > 0, np.asarray(source), 0.0),
    )

    jitted = jax.jit(lambda src, msk: loss(src, msk))(source, mask)
    assert_allclose_compact(jitted, out)


def test_dispatch_component_exchanges_preserves_vector_regridding_behavior() -> None:
    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=[vector("u_velocity", "v_velocity")],
        regridder_factory=cast(Any, _factory),
    )
    regridders = {"OCN->ATM": _ScalingRegridder()}
    u_velocity = jnp.full((2, 2), 5.0)
    v_velocity = jnp.full((2, 2), -2.0)
    state = RunState._from_runtime(
        component_names=("OCN", "ATM"),
        components=(
            _component(
                "OCN",
                sent={"u_velocity": u_velocity, "v_velocity": v_velocity},
            ),
            _component(
                "ATM",
                sent={},
                received={
                    "u_velocity": jnp.zeros((2, 2)),
                    "v_velocity": jnp.zeros((2, 2)),
                },
            ),
        ),
        fractional_masks=FieldStore.from_mapping({"OCN->ATM": jnp.full((2, 2), 0.25)}),
    )

    dispatched = dispatch_component_exchanges(state, "ATM", (exchange,), regridders)
    destination = dispatched._component_state("ATM")

    assert_allclose_compact(
        destination.received.get("u_velocity"), np.full((2, 2), 6.0)
    )
    assert_allclose_compact(
        destination.received.get("v_velocity"), np.full((2, 2), -3.0)
    )


def test_runtime_dispatch_context_groups_exchanges_by_destination() -> None:
    atm_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=cast(Any, _factory),
    )
    land_exchange = Exchange(
        source="ATM",
        target="LND",
        fields=["temperature"],
        regridder_factory=cast(Any, _factory),
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


def test_exchange_dispatch_uses_scalar_and_vector_primitives() -> None:
    source = Path("vercor/_runtime/exchange_dispatch.py").read_text(encoding="utf-8")

    assert "def _dispatch_scalar_exchange_field(" in source
    assert "def _dispatch_vector_exchange_field(" in source
    assert "exchange.destination != destination_name" not in source


def test_exchange_rejects_nonfinite_active_regridding_output() -> None:
    regridder = _NonFiniteRegridder(jnp.asarray([[1.0, jnp.nan]]))
    state, exchange = _one_field_exchange_state(jnp.asarray([[1.0, 1.0]]))
    with pytest.raises(
        CouplerError,
        match="exchange 'OCN->ATM'.*temperature.*active domain",
    ):
        dispatch_component_exchanges(
            state,
            "ATM",
            (exchange,),
            {exchange.route_id: regridder},
        )


def test_exchange_neutralizes_inactive_nan_without_poisoning_jvp_or_vjp() -> None:
    mask = jnp.asarray([[1.0, 0.0]])

    def objective(source: jax.Array) -> jax.Array:
        return jnp.sum(_dispatch_with_passthrough_regridder(source, mask))

    assert_finite_jvp_vjp(
        objective,
        jnp.asarray([[2.0, jnp.nan]]),
        jnp.asarray([[1.0, 0.0]]),
    )
    assert_allclose_compact(
        _dispatch_with_passthrough_regridder(jnp.asarray([[2.0, jnp.nan]]), mask),
        np.asarray([[2.0, 0.0]]),
        equal_nan=False,
    )


def test_exchange_rejects_nonfinite_active_vector_regridding_output() -> None:
    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=[vector("u_velocity", "v_velocity")],
        regridder_factory=cast(Any, _factory),
    )
    mask = jnp.asarray([[1.0, 1.0]])
    state = RunState._from_runtime(
        component_names=("OCN", "ATM"),
        components=(
            _component(
                "OCN",
                sent={
                    "u_velocity": jnp.ones_like(mask),
                    "v_velocity": jnp.ones_like(mask),
                },
            ),
            _component(
                "ATM",
                sent={},
                received={
                    "u_velocity": jnp.zeros_like(mask),
                    "v_velocity": jnp.zeros_like(mask),
                },
            ),
        ),
        fractional_masks=FieldStore.from_mapping({exchange.route_id: mask}),
    )
    regridder = _NonFiniteRegridder(
        jnp.ones_like(mask),
        vector_result=(jnp.asarray([[1.0, jnp.nan]]), jnp.asarray([[jnp.nan, 1.0]])),
    )
    with pytest.raises(
        CouplerError,
        match="exchange 'OCN->ATM'.*u_velocity.*active domain",
    ):
        dispatch_component_exchanges(
            state,
            "ATM",
            (exchange,),
            {exchange.route_id: regridder},
        )
