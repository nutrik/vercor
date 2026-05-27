from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vercor.components.contracts import (
    AuthorFieldValues as _AuthorFieldValues,
    AuthorStepCallable as _AuthorStepCallable,
    ComponentFieldSpec as _ComponentFieldSpec,
    FieldNames as _FieldNames,
)
from vercor.components._field_authoring import ComponentFieldAuthoringMixin
from vercor.components._lifecycle import (
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentLifecycleHooks,
    ComponentPrefillHook,
    ComponentValidateHook,
)
from vercor.components._lifecycle_api import ComponentLifecycleMixin
from vercor.components._runtime_access import ComponentRuntimeAccessMixin
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import (
        RuntimeComponentState,
    )


__all__ = [
    "Component",
]


@dataclass
class Component(
    ComponentFieldAuthoringMixin,
    ComponentRuntimeAccessMixin,
    ComponentLifecycleMixin,
    ABC,
):
    """Active differentiable component-author contract for VerCOR model adapters.

    Component instances own mutable setup-time metadata: name, grid, seed data,
    and component-specific settings. During coupling, the coupler copies those
    seed fields into immutable runtime state containers so JAX can trace the
    integration. Active differentiable components must implement
    :meth:`step_runtime_state` while preserving its signature. Data-only forcing
    adapters should inherit :class:`vercor.components.DataComponent`;
    non-differentiable adapters should inherit
    :class:`vercor.components.HostRuntimeComponent`.

    Common exchange-field conventions:
        - fields use SI units
        - surface fluxes are positive downward and negative upward
        - data fields use canonical trailing horizontal dimensions:
          (nLat, nLon), (nTime, nLat, nLon), (nLev, nLat, nLon), or
          (nTime, nLev, nLat, nLon)

    Attributes:
        name: component name
        grid: component grid
        data: internal storage for component data arrays to/from which fields
            seed the runtime state during initialization
        settings: component-specific settings
        setup_metadata: non-runtime setup metadata for adapter provenance or
            diagnostics that must not enter runtime field validation
    """

    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: VercorSettings = field(default_factory=VercorSettings)
    setup_metadata: dict[str, Any] = field(default_factory=dict)
    _field_spec: _ComponentFieldSpec = field(
        default_factory=_ComponentFieldSpec,
        init=False,
        repr=False,
    )
    _lifecycle_hooks: ComponentLifecycleHooks = field(
        default_factory=ComponentLifecycleHooks,
        init=False,
        repr=False,
    )

    @classmethod
    def from_model(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _AuthorStepCallable,
        *,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "Component":
        """Create a differentiable component from a user model callable.

        This author-facing constructor mirrors normal Python alternate
        constructors: ``inputs`` declare fields the model reads, ``outputs``
        declare fields the model writes, and ``default_fields`` declares
        concrete runtime defaults. Scalar default values expand to this
        component's grid shape.
        """

        from vercor.components.factories import _callable_component_from_model

        return _callable_component_from_model(
            runtime_kind="differentiable",
            name=name,
            grid=grid,
            step=step,
            payload=payload,
            settings=settings,
            inputs=inputs,
            outputs=outputs,
            default_fields=default_fields,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill_runtime_state_fields,
            validate_runtime_state=validate_runtime_state,
        )

    @abstractmethod
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Return this differentiable component advanced by one runtime step."""

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f" ├── Name: {self.name}\n"
            f" ├── Runtime fields: Configured by Coupler runtime contract\n"
            f" └── Grid name: {self.grid.name}\n"
            f"     └── Shape: {self.grid.shape}\n"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, grid={repr(self.grid)})"
