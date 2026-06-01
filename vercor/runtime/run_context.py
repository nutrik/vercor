from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vercor.clock import Clock
from vercor.jax_logging import LoggerLike, effective_log_level
from vercor.runtime.cache import CompiledRuntimeCache
from vercor.runtime.compilation import CompiledRuntime, RuntimeCompilationKey
from vercor.runtime.dispatch_context import RuntimeDispatchContext
from vercor.runtime.interrupts import RuntimeInterruptController


@dataclass(frozen=True)
class RuntimeRunContext:
    """Static inputs required to execute one configured coupler runtime."""

    run_sequence: Sequence[str]
    clock: Clock
    logger: LoggerLike
    log_level: int | str
    dispatch_context: RuntimeDispatchContext
    runtime_cache: CompiledRuntimeCache
    interrupts: RuntimeInterruptController

    def compiled_runtime_cache_key(
        self,
        *,
        donate_state: bool,
    ) -> RuntimeCompilationKey:
        """Return the static cache key for this run context and donation mode."""

        dispatch_context = self.dispatch_context
        return (
            donate_state,
            tuple(
                (name, id(component))
                for name, component in dispatch_context.components.items()
            ),
            tuple(self.run_sequence),
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
                    (key, id(value))
                    for key, value in dispatch_context.regridders.items()
                )
            ),
            id(self.logger),
            id(self.interrupts),
            effective_log_level(self.logger, self.log_level),
            tuple(
                (name, contract.imports, contract.exports)
                for name, contract in sorted(dispatch_context.contracts.items())
            ),
            repr(self.clock.start),
            self.clock.dt_seconds,
            self.clock.steps,
            self.clock.year_type,
            dispatch_context.settings.year_in_seconds,
        )


__all__ = ["CompiledRuntime", "RuntimeRunContext"]
