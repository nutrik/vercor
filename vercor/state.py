"""Opaque public run state and immutable component field views."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import jax

from vercor.grids import RectilinearGrid as _RectilinearGrid
from vercor._field_names import freeze_name_sequence as _freeze_name_sequence
from vercor._runtime.stores import FieldStore as _FieldStore
from vercor.types import RuntimeArray as _RuntimeArray

if TYPE_CHECKING:
    from vercor._runtime.state import ComponentRuntimeState as _ComponentRuntimeState

    _StateFieldStore = _FieldStore
else:
    _ComponentRuntimeState = Any
    _StateFieldStore = Any

FieldScope: TypeAlias = Literal["state", "received", "sent"]
FieldLookupScope: TypeAlias = Literal["any", "state", "received", "sent"]


@dataclass(frozen=True, init=False, repr=False)
class RunState:
    """Immutable coupled model state returned by the public coupler facade."""

    _component_names: tuple[str, ...]
    _component_grids: tuple[_RectilinearGrid | None, ...]
    _components: tuple["_ComponentRuntimeState", ...]
    _fractional_masks: _StateFieldStore
    _component_indices: dict[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(self) -> None:
        """Block direct public construction of opaque runtime state."""

        raise TypeError(
            "RunState is created by Coupler.initial_state() or Coupler.run(); "
            "use those Coupler methods to create coupled runtime state."
        )

    @classmethod
    def _from_runtime(
        cls,
        *,
        component_names: tuple[str, ...],
        components: tuple["_ComponentRuntimeState", ...],
        fractional_masks: _FieldStore,
        component_grids: tuple[_RectilinearGrid | None, ...] = (),
    ) -> "RunState":
        """Create an opaque public run state from private runtime containers."""

        state = object.__new__(cls)
        state._initialize_runtime(
            component_names=component_names,
            components=components,
            fractional_masks=fractional_masks,
            component_grids=component_grids,
        )
        return state

    def _initialize_runtime(
        self,
        *,
        component_names: tuple[str, ...],
        components: tuple["_ComponentRuntimeState", ...],
        fractional_masks: _FieldStore,
        component_grids: tuple[_RectilinearGrid | None, ...] = (),
    ) -> None:
        """Assign private runtime containers after construction is authorized."""

        object.__setattr__(self, "_component_names", component_names)
        object.__setattr__(
            self,
            "_component_grids",
            component_grids or tuple(None for _ in component_names),
        )
        object.__setattr__(self, "_components", components)
        object.__setattr__(self, "_fractional_masks", fractional_masks)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Validate that component names and states stay aligned."""

        if len(self._component_names) != len(self._components):
            raise ValueError("component_names and components must have equal length")
        if len(self._component_names) != len(self._component_grids):
            raise ValueError(
                "component_names and component_grids must have equal length"
            )
        object.__setattr__(
            self,
            "_component_indices",
            {name: index for index, name in enumerate(self._component_names)},
        )

    def component(self, name: str) -> "ComponentState":
        """Return a public field view for one component."""

        return ComponentState._from_runtime(
            name,
            self._component_grids[self._component_index(name)],
            self._component_state(name),
        )

    def components(
        self,
        names: Sequence[str] | None = None,
    ) -> Mapping[str, "ComponentState"]:
        """Return public field views for selected components."""

        selected_names = (
            self._component_names
            if names is None
            else _freeze_name_sequence(
                names,
                label="RunState.components names",
            )
        )
        return MappingProxyType({name: self.component(name) for name in selected_names})

    def replace_fields(
        self,
        component: str,
        fields: Mapping[str, _RuntimeArray],
    ) -> "RunState":
        """Return a new run state with existing component fields replaced."""

        component_state = self._component_state(component)
        runtime_store = component_state.fields
        missing = tuple(field for field in fields if field not in runtime_store)
        if missing:
            raise KeyError(f"Runtime field {missing[0]!r} not found")

        updated_store = runtime_store.replace_many(fields)
        updated_component = component_state.with_fields(updated_store)
        return self._with_component_state(component, updated_component)

    def _component_index(self, name: str) -> int:
        try:
            return self._component_indices[name]
        except KeyError as exc:
            raise KeyError(f"Runtime component {name!r} not found") from exc

    def _component_state(self, name: str) -> "_ComponentRuntimeState":
        """Return one component state by name."""

        return self._components[self._component_index(name)]

    def _with_component_state(
        self,
        name: str,
        component_state: "_ComponentRuntimeState",
    ) -> "RunState":
        """Return a new run state with one component replaced."""

        components = list(self._components)
        components[self._component_index(name)] = component_state
        return RunState._from_runtime(
            component_names=self._component_names,
            components=tuple(components),
            fractional_masks=self._fractional_masks,
            component_grids=self._component_grids,
        )

    def _with_fractional_masks(self, masks: _FieldStore) -> "RunState":
        """Return a state with derived route-mask leaves replaced immutably."""
        return RunState._from_runtime(
            component_names=self._component_names,
            component_grids=self._component_grids,
            components=self._components,
            fractional_masks=masks,
        )

    def _fractional_mask(
        self,
        route_id: str,
    ) -> _RuntimeArray:
        """Return the fractional mask for an exchange."""

        return self._fractional_masks.get(route_id)


def _flatten_run_state(
    state: RunState,
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    """Return dynamic runtime leaves and static component-name metadata."""

    return (
        (state._component_grids, state._components, state._fractional_masks),
        state._component_names,
    )


def _unflatten_run_state(
    component_names: tuple[str, ...],
    children: tuple[object, ...],
) -> RunState:
    """Rebuild opaque runtime state after a JAX PyTree transformation."""

    component_grids, components, fractional_masks = children
    return RunState._from_runtime(
        component_names=component_names,
        component_grids=component_grids,  # type: ignore[arg-type]
        components=components,  # type: ignore[arg-type]
        fractional_masks=fractional_masks,  # type: ignore[arg-type]
    )


jax.tree_util.register_pytree_node(
    RunState,
    _flatten_run_state,
    _unflatten_run_state,
)


@dataclass(frozen=True, init=False)
class ComponentState:
    """Explicit component metadata plus runtime fields for diagnostics/output."""

    name: str
    grid: _RectilinearGrid | None
    _fields: _StateFieldStore = field(default_factory=_FieldStore.empty)
    _received: _StateFieldStore = field(default_factory=_FieldStore.empty)
    _sent: _StateFieldStore = field(default_factory=_FieldStore.empty)

    def __init__(
        self,
        name: str,
        grid: _RectilinearGrid | None,
        *,
        fields: Mapping[str, Any] | None = None,
        received: Mapping[str, Any] | None = None,
        sent: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a public component-state view from plain field mappings."""

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "_fields", _field_store_from_mapping(fields))
        object.__setattr__(self, "_received", _field_store_from_mapping(received))
        object.__setattr__(self, "_sent", _field_store_from_mapping(sent))

    def field(
        self,
        name: str,
        *,
        scope: FieldLookupScope = "any",
    ) -> _RuntimeArray:
        """Return one runtime field, optionally constrained to one scope."""

        if scope != "any":
            return self._store(scope).get(name)

        return _field(self, name)

    def fields(
        self,
        *,
        scope: FieldScope = "state",
    ) -> Mapping[str, _RuntimeArray]:
        """Return one field scope as a read-only mapping."""

        return MappingProxyType(self._store(scope).to_mapping())

    def iter_fields(
        self,
        *scopes: FieldScope,
    ) -> Iterator[tuple[FieldScope, str, _RuntimeArray]]:
        """Yield ``(scope, field_name, value)`` for selected field scopes."""

        runtime_stores = self._stores()
        selected_scopes = scopes or tuple(runtime_stores)
        for scope in selected_scopes:
            try:
                store = runtime_stores[scope]
            except KeyError as exc:
                raise KeyError(f"Runtime view scope {scope!r} not found") from exc
            for field_name, value in zip(store.field_names, store.values, strict=True):
                yield scope, field_name, value

    @classmethod
    def _from_runtime(
        cls,
        name: str,
        grid: _RectilinearGrid | None,
        component_state: "_ComponentRuntimeState",
    ) -> "ComponentState":
        """Create a field view from component metadata and runtime state."""

        view = object.__new__(cls)
        object.__setattr__(view, "name", name)
        object.__setattr__(view, "grid", grid)
        object.__setattr__(view, "_fields", component_state.fields)
        object.__setattr__(view, "_received", component_state.received)
        object.__setattr__(view, "_sent", component_state.sent)
        return view

    def _stores(self) -> dict[FieldScope, _FieldStore]:
        return {
            "state": self._fields,
            "received": self._received,
            "sent": self._sent,
        }

    def _store(
        self,
        name: FieldScope,
    ) -> _FieldStore:
        try:
            return self._stores()[name]
        except KeyError as exc:
            raise KeyError(f"Runtime view scope {name!r} not found") from exc


def _field_store_from_mapping(
    fields: Mapping[str, Any] | None,
) -> _FieldStore:
    """Return a runtime field store from public plain field mappings."""

    if fields is None:
        return _FieldStore.empty()
    if isinstance(fields, _FieldStore):
        raise TypeError(
            "ComponentState expects plain field mappings, not runtime stores"
        )
    return _FieldStore.from_mapping(fields)


def _field_candidates(
    source: ComponentState,
    name: str,
) -> list[_RuntimeArray]:
    """Return all runtime fields named ``name`` in fields, received, sent order."""

    candidates: list[_RuntimeArray] = []
    for store in (source._fields, source._received, source._sent):
        if name in store:
            candidates.append(store.get(name))
    return candidates


def _field(source: ComponentState, name: str) -> _RuntimeArray:
    """Return the first runtime field named ``name`` from a view or state."""

    candidates = _field_candidates(source, name)
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {name!r} not found")


__all__ = ["ComponentState", "FieldLookupScope", "FieldScope", "RunState"]
