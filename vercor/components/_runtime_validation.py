from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.components._contracts import declared_runtime_field_names
from vercor.exceptions import CouplerError
from vercor.field_layout import validate_canonical_grid_field_shape

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.runtime.state import RuntimeComponentState


def require_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
    *names: str,
) -> None:
    """Validate that named runtime data fields use canonical grid layout."""

    for field_name in names:
        if field_name not in component_state.data:
            raise CouplerError(
                "Runtime missing required data field "
                f"'{field_name}' for component '{component.name}'"
            )
        try:
            validate_canonical_grid_field_shape(
                field_name=field_name,
                value=component_state.data.get(field_name),
                grid_shape=component.grid.shape,
                owner_description="Runtime required data field",
                owner_name=component.name,
            )
        except ValueError as exc:
            raise CouplerError(str(exc)) from exc


def validate_declared_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
) -> None:
    """Validate fields required by the component's declared field contract."""

    declared_fields = declared_runtime_field_names(component.field_spec)
    if declared_fields:
        require_runtime_fields(component, component_state, *declared_fields)
