from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import jax

from vercor.jax_logging import effective_log_level
from vercor.runtime.state import RuntimeCouplerState

if TYPE_CHECKING:
    from vercor.runtime.run_context import RuntimeRunContext

CompiledRuntime = Callable[[RuntimeCouplerState], RuntimeCouplerState]


@dataclass(slots=True)
class CompiledRuntimeCache:
    """Owner for compiled scanned-runtime callables keyed by static run metadata."""

    _compiled_runtime_cache: dict[tuple[Any, ...], CompiledRuntime] = field(
        default_factory=dict
    )

    def get_or_compile(
        self,
        scanned_runtime: CompiledRuntime,
        *,
        cache_key: tuple[Any, ...],
        donate_state: bool,
    ) -> CompiledRuntime:
        """Return a cached JIT runtime, compiling and storing it when missing."""

        if cache_key in self._compiled_runtime_cache:
            return self._compiled_runtime_cache[cache_key]

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
        self._compiled_runtime_cache[cache_key] = compiled_runtime
        return compiled_runtime

    def get_or_compile_for_context(
        self,
        scanned_runtime: CompiledRuntime,
        *,
        context: RuntimeRunContext,
        donate_state: bool,
    ) -> CompiledRuntime:
        """Return a compiled runtime keyed from one static runtime context."""

        return self.get_or_compile(
            scanned_runtime,
            cache_key=compiled_runtime_cache_key(
                donate_state=donate_state,
                context=context,
            ),
            donate_state=donate_state,
        )

    def clear(self) -> None:
        """Clear compiled runtime entries."""

        self._compiled_runtime_cache.clear()

    def entry_count(self) -> int:
        """Return the number of compiled runtime entries."""

        return len(self._compiled_runtime_cache)

    def values(self) -> tuple[CompiledRuntime, ...]:
        """Return compiled runtime values without exposing the mutable mapping."""

        return tuple(self._compiled_runtime_cache.values())


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


__all__ = ["CompiledRuntime", "CompiledRuntimeCache", "compiled_runtime_cache_key"]
