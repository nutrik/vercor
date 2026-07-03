from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from vercor.grid import RectilinearGrid
from vercor.runtime.state import RuntimeComponentState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.types import RuntimeArray


@dataclass(frozen=True)
class ComponentView:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: RectilinearGrid
    data: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    incoming: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)
    outgoing: RuntimeFieldStore = field(default_factory=RuntimeFieldStore.empty)

    def field_candidates(self, name: str) -> list[RuntimeArray]:
        """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

        return runtime_field_candidates(self, name)

    def field(self, name: str) -> RuntimeArray:
        """Return the first runtime field named ``name`` from this view."""

        return runtime_field(self, name)

    def iter_store_fields(
        self,
        *store_names: str,
    ) -> Iterator[tuple[str, str, RuntimeArray]]:
        """Yield ``(store_name, field_name, value)`` for selected runtime stores."""

        stores = {
            "data": self.data,
            "incoming": self.incoming,
            "outgoing": self.outgoing,
        }
        selected_store_names = store_names or tuple(stores)
        for store_name in selected_store_names:
            try:
                store = stores[store_name]
            except KeyError as exc:
                raise KeyError(f"Runtime view store {store_name!r} not found") from exc
            for field_name, value in zip(store.field_names, store.values, strict=True):
                yield store_name, field_name, value

    @classmethod
    def from_component_state(
        cls,
        name: str,
        grid: RectilinearGrid,
        component_state: RuntimeComponentState,
    ) -> "ComponentView":
        """Create a field view from component metadata and runtime state."""

        return cls(
            name=name,
            grid=grid,
            data=component_state.data,
            incoming=component_state.incoming,
            outgoing=component_state.outgoing,
        )


RuntimeComponentView = ComponentView


RuntimeFieldSource = ComponentView | RuntimeComponentState


def runtime_field_candidates(
    source: RuntimeFieldSource,
    name: str,
) -> list[RuntimeArray]:
    """Return all runtime fields named ``name`` in data, incoming, outgoing order."""

    candidates: list[RuntimeArray] = []
    for store in (source.data, source.incoming, source.outgoing):
        if name in store:
            candidates.append(store.get(name))
    return candidates


def runtime_field(source: RuntimeFieldSource, name: str) -> RuntimeArray:
    """Return the first runtime field named ``name`` from a view or state."""

    candidates = runtime_field_candidates(source, name)
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {name!r} not found")
