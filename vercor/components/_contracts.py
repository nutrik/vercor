"""Private component field normalization and declaration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor._field_names import unique_field_names
from vercor.components.contracts import ComponentMaskSpec, ComponentSpec, TransferPolicy
from vercor.dtypes import PrecisionPolicy, as_jax_real_array, jax_full
from vercor.exceptions import ComponentError
from vercor.field_layout import (
    canonical_grid_field_shape,
    validate_component_data_layout,
)
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


def normalize_field_values(
    *,
    component_name: str,
    grid: RectilinearGrid,
    fields: Mapping[str, object] | None,
    policy: PrecisionPolicy = None,
) -> dict[str, RuntimeArray]:
    """Expand scalar author values and validate canonical component layouts."""

    normalized: dict[str, RuntimeArray] = {}
    for field_name, field_value in (fields or {}).items():
        try:
            field_array = as_jax_real_array(field_value, policy)
        except (TypeError, ValueError) as exc:
            raise ComponentError(
                f"Component '{component_name}' data field '{field_name}' must be "
                "a real numeric scalar or array."
            ) from exc
        normalized[field_name] = (
            jax_full(grid.shape, field_value, policy)
            if field_array.shape == ()
            else field_array
        )
    validate_component_data_layout(
        component_name=component_name,
        grid_shape=grid.shape,
        data=normalized,
    )
    return normalized


def declared_runtime_field_names(spec: ComponentSpec) -> tuple[str, ...]:
    """Return all input/output names declared by a component spec."""

    return unique_field_names((*spec.inputs, *spec.outputs))


def validate_time_selected_field_layout(
    value: object,
    transfer: TransferPolicy,
    *,
    field_name: str | None = None,
    masks: ComponentMaskSpec | None = None,
    component_name: str | None = None,
) -> None:
    """Require complete time-first climatologies while preserving static masks.

    Runtime metadata always indexes twelve monthly or 365 no-leap daily
    records, including the mapped daily records used by 360-day clocks.
    Validate array metadata before a take can slice latitude or fill NaNs.
    """

    mode = transfer.time_selection
    if mode == "current" or (
        masks is not None and field_name in (masks.binary_field, masks.fractional_field)
    ):
        return
    records = 12 if mode == "linear" else 365
    shape = canonical_grid_field_shape(value)
    if len(shape) in (3, 4) and shape[0] == records:
        return
    owner = "Runtime" if component_name is None else f"Component '{component_name}'"
    raise ComponentError(
        f"{owner} field '{field_name or '<unnamed>'}' with transfer policy '{mode}' "
        f"has shape {shape}; expected {records} time-first records in "
        "(nTime, nLat, nLon) or (nTime, nLev, nLat, nLon) layout."
    )


def validate_selectable_output_fields(
    *,
    component_name: str,
    spec: ComponentSpec,
    fields: Mapping[str, RuntimeArray],
) -> None:
    """Validate provided output fields against their shared transfer policy."""

    for field_name in spec.outputs:
        if field_name in fields:
            validate_time_selected_field_layout(
                fields[field_name],
                spec.transfer,
                field_name=field_name,
                masks=spec.masks,
                component_name=component_name,
            )


def validate_declared_updates(
    component_name: str,
    updates: Mapping[str, Any],
    declared: tuple[str, ...],
    *,
    phase: str,
) -> None:
    """Reject the first field update absent from the allowed declaration."""

    allowed = set(declared)
    undeclared = next((name for name in updates if name not in allowed), None)
    if undeclared is not None:
        suffix = " declared output" if phase == "step" else " declared field"
        raise ComponentError(
            f"Component '{component_name}' {phase} returned field '{undeclared}' "
            f"that is not a{suffix}."
        )


__all__: list[str] = []
