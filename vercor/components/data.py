from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, final

from vercor.components.contracts import AuthorFieldValues
from vercor.components._contracts import (
    merge_component_outputs,
)
from vercor.components.base import Component
from vercor.components.contexts import ComponentStepContext
from vercor.dtypes import PrecisionPolicy
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.runtime.state import RuntimeComponentState


class DataComponent(Component):
    """Base class for data-only components that intentionally do not step.

    Use this for forcing and boundary-condition adapters whose runtime behavior is
    limited to importing/exporting seeded fields through the coupler contract.
    Data components must not own active runtime stepping behavior; compute
    plotting-only diagnostics outside runtime state. Active differentiable models
    should inherit :class:`Component` and implement
    :meth:`Component.step_runtime_state` instead.
    """

    @classmethod
    def from_fields(
        cls,
        name: str,
        grid: RectilinearGrid,
        fields: AuthorFieldValues = None,
        settings: VercorSettings | None = None,
    ) -> "DataComponent":
        """Create a data-only component from user-provided grid fields.

        Scalar field values expand to grid-shaped constants and seeded field
        names are exposed as declared outputs.
        """

        if settings is None:
            component = cls(name=name, grid=grid)
        else:
            component = cls(name=name, grid=grid, settings=settings)
        if fields is not None:
            component.seed_fields(fields)
        return component

    def seed_fields(
        self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> "DataComponent":
        """Seed data fields and expose their names as declared outputs."""

        super().seed_fields(fields, policy=policy)
        self._field_spec = merge_component_outputs(self.field_spec, fields.keys())
        return self

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: ComponentStepContext,
    ) -> "RuntimeComponentState":
        """Return the runtime state unchanged for data-only components."""

        _ = context
        return component_state


__all__ = ["DataComponent"]
