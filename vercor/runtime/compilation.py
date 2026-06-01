from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

from vercor.runtime.state import RuntimeCouplerState

CompiledRuntime = Callable[[RuntimeCouplerState], RuntimeCouplerState]
RuntimeCompilationKey: TypeAlias = tuple[Any, ...]

__all__ = ["CompiledRuntime", "RuntimeCompilationKey"]
