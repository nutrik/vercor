"""Data-only convenience adapter for the structural component protocol."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor._field_names import unique_field_names
from vercor.components.contracts import ComponentSpec, StepResult
from vercor.components.contexts import StepContext
from vercor.grids import RectilinearGrid
from vercor.types import RuntimeArray


class DataComponent:
    """Compose a no-op component from immutable initial data fields.

    Supplied ``fields`` are merged with ``spec.initial_fields``, defensively
    snapshotted, and declared as outputs. Scalar values expand on ``grid`` and
    all values adopt the runtime dtype during private preparation. The adapter
    never performs an active model step; time selection and host capability are
    declared through the optional ``ComponentSpec``.
    """

    name: str
    grid: RectilinearGrid
    spec: ComponentSpec

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        fields: Mapping[str, object] | None = None,
        *,
        spec: ComponentSpec | None = None,
    ) -> None:
        """Validate configuration and build the merged immutable spec."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(grid, RectilinearGrid):
            raise TypeError("grid must be RectilinearGrid")
        if spec is not None and not isinstance(spec, ComponentSpec):
            raise TypeError("spec must be ComponentSpec or None")
        declaration = ComponentSpec() if spec is None else spec
        supplied_fields = dict(fields or {})
        merged_fields = dict(declaration.initial_fields)
        merged_fields.update(supplied_fields)
        self.name = name
        self.grid = grid
        self.spec = ComponentSpec(
            inputs=declaration.inputs,
            outputs=unique_field_names((*declaration.outputs, *tuple(supplied_fields))),
            initial_fields=merged_fields,
            execution=declaration.execution,
            lifecycle=declaration.lifecycle,
            transfer=declaration.transfer,
            output=declaration.output,
            masks=declaration.masks,
        )

    def step(
        self,
        fields: Mapping[str, RuntimeArray],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, RuntimeArray] | StepResult:
        """Return no updates because data components have no active model step."""

        _ = fields, context, payload
        return {}

    def __repr__(self) -> str:
        return f"DataComponent(name={self.name!r}, grid={self.grid!r})"


__all__ = ["DataComponent"]
