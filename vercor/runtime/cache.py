from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, cast

import jax

from vercor.jax_logging import effective_log_level
from vercor.runtime.run_context import CompiledRuntime, RuntimeRunContext


def compiled_scanned_runtime(
    scanned_runtime: CompiledRuntime,
    *,
    cache: MutableMapping[tuple[Any, ...], CompiledRuntime],
    cache_key: tuple[Any, ...],
    donate_state: bool,
) -> CompiledRuntime:
    """Return a cached JIT-scanned runtime for one static topology key."""

    if cache_key in cache:
        return cache[cache_key]

    if donate_state:
        compiled_runtime = cast(
            CompiledRuntime,
            jax.jit(scanned_runtime, donate_argnums=(0,)),
        )
    else:
        compiled_runtime = cast(
            CompiledRuntime,
            jax.jit(scanned_runtime),
        )
    cache[cache_key] = compiled_runtime
    return compiled_runtime


def compiled_runtime_cache_key(
    *,
    donate_state: bool,
    context: RuntimeRunContext,
) -> tuple[Any, ...]:
    """Return a static cache key for the compiled pure-runtime wrapper."""

    dispatch_context = context.dispatch_context
    return (
        donate_state,
        tuple(
            (name, id(component))
            for name, component in dispatch_context.components.items()
        ),
        tuple(context.run_sequence),
        tuple(
            (
                id(exchange),
                exchange.source,
                exchange.destination,
                exchange.interpolation_type,
                tuple(exchange.field_names),
            )
            for exchange in dispatch_context.exchanges
        ),
        tuple(
            sorted(
                (key, id(value)) for key, value in dispatch_context.regridders.items()
            )
        ),
        id(context.logger),
        id(context.interrupts),
        effective_log_level(context.logger, context.log_level),
        tuple(
            (name, contract.imports, contract.exports)
            for name, contract in sorted(dispatch_context.contracts.items())
        ),
        repr(context.clock.start),
        context.clock.dt_seconds,
        context.clock.steps,
        context.clock.year_type,
        dispatch_context.settings.year_in_seconds,
    )
