"""Current exchange coverage derived from component fields and fixed geometry.

Intensive transfers normalize source-covered values; flux transfers select
either source or target area. All fractions are dimensionless. Remapped mask
diagnostics are clipped to [0, 1], as in static surface-mask remapping; physical
field numerators and covered-area denominators are never clipped or rescaled.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor._numerical_safety import replace_missing_nan
from vercor._runtime.component_masks import resolve_component_masks
from vercor.exchanges import Exchange
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor._runtime.dispatch_context import RuntimeDispatchContext
    from vercor.state import RunState


def exchange_mask_operands(
    state: RunState, exchange: Exchange, *, context: RuntimeDispatchContext
) -> tuple[RuntimeArray, RuntimeArray, RuntimeArray, RuntimeArray]:
    """Return source fraction, target binary/fraction, and remapped overlap."""
    source = context.components[exchange.source]
    target = context.components[exchange.target]
    _, source_fraction = resolve_component_masks(
        source,
        state._component_state(exchange.source).fields.to_mapping(),
        context.dtype,
    )
    target_binary, target_fraction = resolve_component_masks(
        target,
        state._component_state(exchange.target).fields.to_mapping(),
        context.dtype,
    )
    route_binary = context.topology_binary_masks.get(exchange.route_id)
    if route_binary is not None:
        target_binary = jnp.asarray(target_binary) * jnp.asarray(route_binary)
        target_fraction = jnp.asarray(target_fraction) * jnp.asarray(route_binary)
    overlap = replace_missing_nan(
        context.regridders[exchange.route_id].regrid(source_fraction),
        owner=f"Exchange '{exchange.route_id}' remapped coverage",
    )
    overlap = jnp.where(jnp.asarray(target_binary) > 0, overlap, 0.0)
    return source_fraction, target_binary, target_fraction, overlap


def exchange_coverage(
    state: RunState, exchange: Exchange, *, context: RuntimeDispatchContext
) -> RuntimeArray:
    """Resolve diagnostic overlap without attenuating intensive field values."""
    if exchange.quantity == "legacy":
        return state._fractional_mask(exchange.route_id)
    _, binary, fraction, overlap = exchange_mask_operands(
        state, exchange, context=context
    )
    if exchange.quantity == "target_flux":
        return jnp.where(jnp.asarray(overlap) > 0, fraction, 0.0)
    return jnp.clip(jnp.asarray(overlap), 0.0, 1.0) * jnp.asarray(binary)


def refresh_runtime_masks(
    state: RunState, *, context: RuntimeDispatchContext
) -> RunState:
    """Replace derived coverage while preserving field order, shape and dtype."""
    updates = {
        route.route_id: exchange_coverage(state, route, context=context)
        for route in context.exchanges
    }
    return state._with_fractional_masks(state._fractional_masks.replace_many(updates))


def runtime_exchange_masks(
    state: RunState, *, context: RuntimeDispatchContext
) -> tuple[dict[str, RuntimeArray], dict[str, RuntimeArray]]:
    """Return current binary/fractional diagnostics for the configured routes."""
    fractions = {
        route.route_id: exchange_coverage(state, route, context=context)
        for route in context.exchanges
    }
    binary = {
        route.route_id: (
            context.topology_binary_masks[route.route_id]
            if route.quantity == "legacy"
            and route.route_id in context.topology_binary_masks
            else jnp.asarray(
                fractions[route.route_id] > 0, dtype=context.dtype.jax_real
            )
        )
        for route in context.exchanges
    }
    return binary, fractions
