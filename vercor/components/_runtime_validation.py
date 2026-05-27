from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from vercor.components._contracts import declared_runtime_field_names
from vercor.exceptions import CouplerError
from vercor.field_layout import (
    canonical_data_layout_description,
    is_canonical_grid_field_shape,
)

if TYPE_CHECKING:
    from vercor.components.base import Component
    from vercor.runtime import RuntimeComponentState


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
        field_shape = tuple(
            int(size)
            for size in jnp.asarray(component_state.data.get(field_name)).shape
        )
        if not is_canonical_grid_field_shape(field_shape, component.grid.shape):
            raise CouplerError(
                "Runtime required data field "
                f"'{field_name}' for component '{component.name}' has shape "
                f"{field_shape}; expected canonical grid-field layout "
                f"{canonical_data_layout_description()} with trailing grid shape "
                f"{component.grid.shape}"
            )


def validate_declared_runtime_fields(
    component: "Component",
    component_state: "RuntimeComponentState",
) -> None:
    """Validate fields required by the component's declared field contract."""

    declared_fields = declared_runtime_field_names(component.field_spec)
    if declared_fields:
        require_runtime_fields(component, component_state, *declared_fields)
