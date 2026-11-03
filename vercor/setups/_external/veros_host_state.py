"""Own host-JAX Veros metadata without recompiling unchanged native kernels.

Stock Veros puts diagnostics, timers and settings objects in PyTree auxiliary
data. Deep-copying those owners changes native JIT keys even when the physics
and shapes are unchanged. This host-only state subclass compares immutable
setup identity and actual setting values for compilation, while retaining each
payload's independent diagnostic/timer owners outside compiled kernels. Native
variables remain dynamic leaves. It does not make the stock driver traceable.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

import jax
from veros.state import VerosState


@dataclass(frozen=True)
class _HostMetadata:
    definition: object
    settings: tuple[tuple[str, Any], ...]
    attributes: tuple[tuple[str, Any], ...] = field(compare=False, hash=False)


@jax.tree_util.register_pytree_node_class
class _HostJaxState(VerosState):
    """Native host state with value-based static configuration and owned metadata."""

    _vercor_definition: object

    def tree_flatten(self) -> tuple[tuple[Any], _HostMetadata]:
        """Expose native variables dynamically, excluding host bookkeeping from keys."""
        attributes = tuple(
            (name, value) for name, value in vars(self).items() if name != "_variables"
        )
        return (self.variables,), _HostMetadata(
            self._vercor_definition, tuple(self.settings.items()), attributes
        )

    @classmethod
    def tree_unflatten(
        cls, metadata: _HostMetadata, leaves: tuple[Any]
    ) -> _HostJaxState:
        """Rebuild the native container using the current payload's auxiliary owners."""
        state = cls.__new__(cls)
        cast(dict[str, Any], vars(state)).update(metadata.attributes)
        state._variables = leaves[0]
        return state


def copy_host_jax_state(state: Any) -> Any:
    """Copy mutable host owners while sharing immutable buffers/setup declarations.

    Grid dimensions, variable declarations and plugin interfaces are fixed at
    setup. Settings are copied and their actual values participate in JIT keys,
    so changed physical configuration still causes the required compilation.
    Native built-in kernels consume variables/settings, not diagnostic/timer
    bookkeeping; the host driver retains those owners in the returned state.
    """
    if not isinstance(state, _HostJaxState):
        view = _HostJaxState.__new__(_HostJaxState)
        cast(dict[str, Any], vars(view)).update(vars(state))
        view._vercor_definition = object()
        state = view
    memo: dict[int, Any] = {
        id(value): value
        for value in state.variables.values()
        if isinstance(value, jax.Array)
    }
    for owner, names in (
        (
            state,
            ("_vercor_definition", "_var_meta", "_dimensions", "_plugin_interfaces"),
        ),
        (state.variables, ("__dimensions__", "__metadata__", "__fields__")),
    ):
        for name in names:
            value = getattr(owner, name)
            memo[id(value)] = value
    fields = state.settings.__fields__
    memo[id(fields)] = tuple(fields)
    return deepcopy(state, memo)


__all__ = ["copy_host_jax_state"]
