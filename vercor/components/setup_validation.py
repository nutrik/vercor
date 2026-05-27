from __future__ import annotations

from typing import Any

from vercor.exceptions import ComponentError
from vercor.field_layout import validate_component_data_layout
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings


def validate_component_setup(component: Any) -> None:
    """Raise a clear error when a component skipped base initialization."""

    required_attributes = ("name", "grid", "data", "settings")
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(component, attribute)
    ]
    if missing:
        missing_names = ", ".join(missing)
        raise ComponentError(
            f"Component '{component.__class__.__name__}' is missing required setup "
            f"attribute(s): {missing_names}. Call super().__init__(name, grid=...) "
            "from the component constructor before runtime initialization, "
            "execution, or finalization."
        )

    if not isinstance(component.grid, RectilinearGrid):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'grid'; "
            "expected RectilinearGrid."
        )
    if not isinstance(component.data, dict):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'data'; "
            "expected dict[str, RuntimeArray]."
        )
    if not isinstance(component.settings, VercorSettings):
        raise ComponentError(
            f"Component '{component.name}' has invalid setup attribute 'settings'; "
            "expected VercorSettings."
        )
    validate_component_data_layout(
        component_name=component.name,
        grid_shape=component.grid.shape,
        data=component.data,
    )
