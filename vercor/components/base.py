from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vercor._deprecation import warn_deprecated_name
from vercor.components.contracts import (
    AuthorFieldValues as _AuthorFieldValues,
    AuthorStepCallable as _AuthorStepCallable,
    ComponentCreatePayloadHook,
    ComponentHooks as _ComponentHooks,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentStepReturn as _ComponentStepReturn,
    ComponentValidateHook,
    FieldSpec as _FieldSpec,
    FieldNames as _FieldNames,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
)
from vercor.components._constructor_options import (
    normalize_field_spec,
    normalize_lifecycle_hooks,
)
from vercor.components._field_authoring import ComponentFieldAuthoringMixin
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.components._lifecycle_api import ComponentLifecycleMixin
import vercor.components._runtime_fields as _runtime_field_adapters
import vercor.components._runtime_validation as _runtime_field_validation
from vercor.dtypes import PrecisionPolicy
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


__all__ = [
    "Component",
]


@dataclass
class Component(
    ComponentFieldAuthoringMixin,
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
    :class:`vercor.components.HostComponent`.

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
    _field_spec: _FieldSpec = field(
        default_factory=_FieldSpec,
        init=False,
        repr=False,
    )
    _lifecycle_hooks: ComponentLifecycleHooks = field(
        default_factory=ComponentLifecycleHooks,
        init=False,
        repr=False,
    )

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: _AuthorStepCallable,
        *,
        fields: _FieldSpec | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        hooks: _ComponentHooks | None = None,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        defaults: _AuthorFieldValues = None,
        default_fields: _AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "Component":
        """Create a differentiable component from a user step callable.

        ``inputs`` declare fields the model reads, ``outputs`` declare fields
        the model writes, and ``defaults`` declares concrete runtime defaults.
        Scalar default values expand to this component's grid shape.
        """

        field_spec = normalize_field_spec(
            fields=fields,
            inputs=inputs,
            outputs=outputs,
            defaults=defaults,
            default_fields=default_fields,
        )
        lifecycle_hooks = normalize_lifecycle_hooks(
            hooks=hooks,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill_runtime_state_fields,
            validate_runtime_state=validate_runtime_state,
        )
        return _CallableComponent(
            name=name,
            grid=grid,
            step=step,
            payload=payload,
            settings=settings,
            field_spec=field_spec,
            lifecycle_hooks=lifecycle_hooks,
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
        defaults: _AuthorFieldValues = None,
        default_fields: _AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "Component":
        """Create a differentiable component from a user model callable.

        Deprecated compatibility wrapper for :meth:`from_step`.
        """

        warn_deprecated_name(
            f"{cls.__name__}.from_model()",
            f"{cls.__name__}.from_step()",
            remove_in="0.2.0",
        )
        return cls.from_step(
            name=name,
            grid=grid,
            step=step,
            payload=payload,
            settings=settings,
            inputs=inputs,
            outputs=outputs,
            defaults=defaults,
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
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Return this differentiable component advanced by one runtime step."""

    def runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> dict[str, RuntimeArray]:
        """Return runtime data fields as a plain name-to-array mapping."""

        return _runtime_field_adapters.runtime_fields(component_state)

    def runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> RuntimeArray:
        """Return one runtime data field with a component-oriented error."""

        return _runtime_field_adapters.runtime_field(self, component_state, name)

    def has_runtime_field(
        self,
        component_state: "RuntimeComponentState",
        name: str,
    ) -> bool:
        """Return whether one runtime data field exists."""

        return _runtime_field_adapters.has_runtime_field(component_state, name)

    def runtime_field_or(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        default: object,
        policy: PrecisionPolicy = None,
    ) -> RuntimeArray:
        """Return one runtime field or a grid-shaped/default array fallback."""

        return _runtime_field_adapters.runtime_field_or(
            self,
            component_state,
            name,
            default,
            policy,
        )

    def runtime_field_or_zeros_like(
        self,
        component_state: "RuntimeComponentState",
        name: str,
        like: str | RuntimeArray,
    ) -> RuntimeArray:
        """Return one runtime field or zeros matching another field/array."""

        return _runtime_field_adapters.runtime_field_or_zeros_like(
            self,
            component_state,
            name,
            like,
        )

    def with_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        fields: Mapping[str, RuntimeArray],
    ) -> "RuntimeComponentState":
        """Return ``component_state`` with existing runtime data fields updated."""

        return _runtime_field_adapters.with_runtime_fields(
            self,
            component_state,
            fields,
        )

    def apply_step_result(
        self,
        component_state: "RuntimeComponentState",
        result: _ComponentStepReturn,
    ) -> "RuntimeComponentState":
        """Apply a field mapping or ``StepResult`` to runtime state."""

        return _runtime_field_adapters.apply_step_result(
            self,
            component_state,
            result,
        )

    def require_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        *names: str,
    ) -> None:
        """Validate that named runtime data fields use canonical grid layout."""

        _runtime_field_validation.require_runtime_fields(
            self,
            component_state,
            *names,
        )

    def prefill_runtime_fields(
        self,
        data: dict[str, RuntimeArray],
        field_spec: _FieldSpec | None = None,
        *,
        outputs: _FieldNames = (),
        defaults: _AuthorFieldValues = None,
        default_fields: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> None:
        """Prefill a mutable runtime data mapping with declared fields."""

        _runtime_field_adapters.prefill_runtime_fields(
            self,
            data,
            field_spec,
            outputs=outputs,
            defaults=defaults,
            default_fields=default_fields,
            policy=policy,
        )

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


class _CallableComponent(_CallableRuntimeMixin, Component):
    """Differentiable component backed by an author-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: _AuthorStepCallable,
        payload: Any | None,
        settings: VercorSettings | None,
        field_spec: _FieldSpec,
        lifecycle_hooks: ComponentLifecycleHooks,
    ) -> None:
        if settings is None:
            Component.__init__(self, name=name, grid=grid)
        else:
            Component.__init__(
                self,
                name=name,
                grid=grid,
                settings=settings,
            )
        self._initialize_callable_runtime(
            step=step,
            payload=payload,
            field_spec=field_spec,
            lifecycle_hooks=lifecycle_hooks,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed differentiable component one step."""

        return self._step_callable_runtime_state(component_state, context)
