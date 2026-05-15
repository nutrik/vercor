from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from vercor.exchange import Exchange
from vercor.runtime import (
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
    dispatch_component_exchanges,
)
from vercor.runtime.driver import RuntimeDispatchContext


class _ScalingRegridder:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, *args: Any) -> Any:
        if len(args) == 1:
            return jnp.asarray(args[0]) * self.scale
        return jnp.asarray(args[0]) + 1.0, jnp.asarray(args[1]) - 1.0


def _factory(*args: Any, **kwargs: Any) -> _ScalingRegridder:
    _ = args, kwargs
    return _ScalingRegridder()


def _component(
    name: str,
    *,
    outgoing: dict[str, jax.Array],
    incoming: dict[str, jax.Array] | None = None,
) -> RuntimeComponentState:
    _ = name
    return RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({}),
        incoming=RuntimeFieldStore.from_mapping(incoming or {}),
        outgoing=RuntimeFieldStore.from_mapping(outgoing),
    )


def test_dispatch_component_exchanges_handles_scalar_masks_and_gradients() -> None:
    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=cast(Any, _factory),
    )
    regridders = {("OCN", "ATM", "_factory"): _ScalingRegridder(scale=2.0)}

    def loss(source: jax.Array, mask: jax.Array) -> jax.Array:
        state = RuntimeCouplerState(
            component_names=("OCN", "ATM"),
            components=(
                _component("OCN", outgoing={"temperature": source}),
                _component(
                    "ATM", outgoing={}, incoming={"temperature": jnp.zeros_like(source)}
                ),
            ),
            fractional_masks=RuntimeFieldStore.from_mapping({"OCN|ATM|_factory": mask}),
            binary_masks=RuntimeFieldStore.empty(),
        )
        dispatched = dispatch_component_exchanges(
            state,
            "ATM",
            (exchange,),
            regridders,
        )
        return jnp.sum(
            dispatched.get_component_state("ATM").incoming.get("temperature")
        )

    source = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.asarray([[1.0, 0.5], [0.0, 1.0]])

    out = loss(source, mask)
    grad_source, grad_mask = jax.grad(loss, argnums=(0, 1))(source, mask)

    assert_allclose_compact(out, np.asarray(12.0))
    assert_allclose_compact(grad_source, 2.0 * np.asarray(mask))
    assert_allclose_compact(grad_mask, 2.0 * np.asarray(source))

    jitted = jax.jit(lambda src, msk: loss(src, msk))(source, mask)
    assert_allclose_compact(jitted, out)


def test_dispatch_component_exchanges_preserves_vector_regridding_behavior() -> None:
    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=[("u_velocity", "v_velocity")],
        regridder_factory=cast(Any, _factory),
    )
    regridders = {("OCN", "ATM", "_factory"): _ScalingRegridder()}
    u_velocity = jnp.full((2, 2), 5.0)
    v_velocity = jnp.full((2, 2), -2.0)
    state = RuntimeCouplerState(
        component_names=("OCN", "ATM"),
        components=(
            _component(
                "OCN",
                outgoing={"u_velocity": u_velocity, "v_velocity": v_velocity},
            ),
            _component(
                "ATM",
                outgoing={},
                incoming={
                    "u_velocity": jnp.zeros((2, 2)),
                    "v_velocity": jnp.zeros((2, 2)),
                },
            ),
        ),
        fractional_masks=RuntimeFieldStore.from_mapping(
            {"OCN|ATM|_factory": jnp.full((2, 2), 0.25)}
        ),
        binary_masks=RuntimeFieldStore.empty(),
    )

    dispatched = dispatch_component_exchanges(state, "ATM", (exchange,), regridders)
    destination = dispatched.get_component_state("ATM")

    assert_allclose_compact(
        destination.incoming.get("u_velocity"), np.full((2, 2), 6.0)
    )
    assert_allclose_compact(
        destination.incoming.get("v_velocity"), np.full((2, 2), -3.0)
    )


def test_runtime_dispatch_context_groups_exchanges_by_destination() -> None:
    atm_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=cast(Any, _factory),
    )
    land_exchange = Exchange(
        source="ATM",
        destination="LND",
        field_names=["temperature"],
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
        settings=cast(Any, object()),
    )

    assert context.destination_exchanges("ATM") == (atm_exchange,)
    assert context.destination_exchanges("OCN") == ()


def test_exchange_dispatch_uses_scalar_and_vector_primitives() -> None:
    source = Path("vercor/runtime/exchange_dispatch.py").read_text(encoding="utf-8")

    assert "def _dispatch_scalar_exchange_field(" in source
    assert "def _dispatch_vector_exchange_field(" in source
