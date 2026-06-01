from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vercor.clock import Clock
from vercor.jax_logging import LoggerLike
from vercor.runtime.cache import CompiledRuntime, CompiledRuntimeCache
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


__all__ = ["CompiledRuntime", "RuntimeRunContext"]
