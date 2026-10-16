"""Prescribed monthly Veros ocean surfaces with explicit zero currents and SSH."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.components import DataComponent
from vercor.output import OutputSpec
from vercor.setups._data._component_helpers import time_interpolated_data_component

if TYPE_CHECKING:
    from vercor.setups._external.veros_data import VerosDataSnapshot


def make_veros_data_ocean(
    snapshot: VerosDataSnapshot, *, name: str = "OCN", output: OutputSpec | None = None
) -> DataComponent:
    """Reuse the snapshot's exact wet grid and canonical monthly surface arrays.

    SST is Kelvin, salinity is g/kg, velocities are m/s, and SSH is m. Native
    face velocities are prescribed zero; no atmospheric inputs or fictitious
    sea-ice feedback are declared on this data component.
    """
    return time_interpolated_data_component(
        name=name,
        grid=snapshot.grid,
        fields=snapshot.fields,
        outputs=tuple(snapshot.fields),
        output=output,
    )
