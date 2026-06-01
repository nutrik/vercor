from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from vercor.clock import Clock
from vercor.jax_logging import LoggerLike
from vercor.runtime.dispatch_context import RuntimeDispatchContext
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.state import RuntimeCouplerState

CompiledRuntime = Callable[[RuntimeCouplerState], RuntimeCouplerState]


@dataclass(frozen=True)
class RuntimeRunContext:
    """Static inputs required to execute one configured coupler runtime."""

    run_sequence: Sequence[str]
    clock: Clock
    logger: LoggerLike
    log_level: int | str
    dispatch_context: RuntimeDispatchContext
    compiled_runtime_cache: MutableMapping[tuple[Any, ...], CompiledRuntime]
    interrupts: RuntimeInterruptController
