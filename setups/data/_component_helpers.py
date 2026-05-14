from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

from vercor.components.base import DataComponent, data_component
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext


def time_interpolated_data_component(
    *,
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object],
    outputs: tuple[str, ...],
    data_files: Mapping[str, str],
    initialize: Callable[[DataComponent, ComponentInitContext], None] | None = None,
) -> DataComponent:
    """Create a data component with the standard time-interpolation metadata."""

    component = data_component(
        name=name,
        grid=grid,
        fields=fields,
        initialize=initialize,
    )
    component.declare_fields(outputs=outputs)
    component.update_settings(apply_time_interpolation=True)
    cast(Any, component).DATA_FILES = dict(data_files)
    return component
