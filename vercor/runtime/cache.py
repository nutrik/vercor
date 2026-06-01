from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import jax

from vercor.runtime.compilation import CompiledRuntime, RuntimeCompilationKey


@dataclass(slots=True)
class CompiledRuntimeCache:
    """Owner for compiled scanned-runtime callables keyed by static run metadata."""

    _compiled_runtime_cache: dict[RuntimeCompilationKey, CompiledRuntime] = field(
        default_factory=dict
    )

    def get_or_compile(
        self,
        scanned_runtime: CompiledRuntime,
        *,
        cache_key: RuntimeCompilationKey,
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

    def clear(self) -> None:
        """Clear compiled runtime entries."""

        self._compiled_runtime_cache.clear()

    def entry_count(self) -> int:
        """Return the number of compiled runtime entries."""

        return len(self._compiled_runtime_cache)

    def values(self) -> tuple[CompiledRuntime, ...]:
        """Return compiled runtime values without exposing the mutable mapping."""

        return tuple(self._compiled_runtime_cache.values())


__all__ = ["CompiledRuntime", "CompiledRuntimeCache"]
