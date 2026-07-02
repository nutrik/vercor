from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax

from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.contracts import exchange_key_name
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeComponentState(PyTreeNodeMixin):
    """Immutable runtime state for one component."""

    pytree_children = ("data", "incoming", "outgoing", "runtime_payload")

    data: RuntimeFieldStore
    incoming: RuntimeFieldStore
    outgoing: RuntimeFieldStore
    runtime_payload: Any | None = None

    def with_data(self, data: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced data."""

        return RuntimeComponentState(
            data=data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_incoming(self, incoming: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced incoming fields."""

        return RuntimeComponentState(
            data=self.data,
            incoming=incoming,
            outgoing=self.outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_outgoing(self, outgoing: RuntimeFieldStore) -> "RuntimeComponentState":
        """Return this component state with replaced outgoing fields."""

        return RuntimeComponentState(
            data=self.data,
            incoming=self.incoming,
            outgoing=outgoing,
            runtime_payload=self.runtime_payload,
        )

    def with_runtime_payload(
        self, runtime_payload: Any | None
    ) -> "RuntimeComponentState":
        """Return this component state with replaced runtime payload."""

        return RuntimeComponentState(
            data=self.data,
            incoming=self.incoming,
            outgoing=self.outgoing,
            runtime_payload=runtime_payload,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeCouplerState(PyTreeNodeMixin):
    """Immutable runtime state for the VerCOR runtime core."""

    pytree_children = ("components", "fractional_masks")
    pytree_aux_data = ("component_names",)

    component_names: tuple[str, ...]
    components: tuple[RuntimeComponentState, ...]
    fractional_masks: RuntimeFieldStore
    component_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate that component names and states stay aligned."""

        if len(self.component_names) != len(self.components):
            raise ValueError("component_names and components must have equal length")
        object.__setattr__(
            self,
            "component_indices",
            {name: index for index, name in enumerate(self.component_names)},
        )

    def _pytree_post_unflatten(self) -> None:
        """Validate that component names and states stay aligned."""

        self.__post_init__()

    def get_component_state(self, name: str) -> RuntimeComponentState:
        """Return one component state by name."""

        try:
            index = self.component_indices[name]
        except KeyError as exc:
            raise KeyError(f"Runtime component {name!r} not found") from exc
        return self.components[index]

    def set_component_state(
        self, name: str, component_state: RuntimeComponentState
    ) -> "RuntimeCouplerState":
        """Return a new coupler state with one component replaced."""

        if name not in self.component_indices:
            raise KeyError(f"Runtime component {name!r} not found")
        components = list(self.components)
        components[self.component_indices[name]] = component_state
        return RuntimeCouplerState(
            component_names=self.component_names,
            components=tuple(components),
            fractional_masks=self.fractional_masks,
        )

    def get_fractional_mask(
        self, source: str, destination: str, interpolation_type: str
    ) -> RuntimeArray:
        """Return the fractional mask for an exchange."""

        return self.fractional_masks.get(
            exchange_key_name(source, destination, interpolation_type)
        )
