from __future__ import annotations

from collections.abc import Callable, Mapping

from vercor.components import DataComponent, SetupContext
from vercor.grid import RectilinearGrid


def time_interpolated_data_component(
    *,
    name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object],
    outputs: tuple[str, ...],
    data_files: Mapping[str, str],
    initialize: Callable[[DataComponent, SetupContext], None] | None = None,
) -> DataComponent:
    """Create a data component with the standard time-interpolation metadata."""

    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields=fields,
        initialize=initialize,
    )
    component.declare_fields(outputs=outputs)
    component.update_settings(apply_time_interpolation=True)
    component.setup_metadata["DATA_FILES"] = dict(data_files)
    return component
