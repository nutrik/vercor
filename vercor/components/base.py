from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self, cast, final

from vercor.dtypes import PrecisionPolicy, jax_full, jax_zeros
from vercor.components._contracts import (
    ComponentStepResult as ComponentStepResult,
)
from vercor.components._contracts import (
    AuthorFieldValues as _AuthorFieldValues,
    AuthorStepCallable as _AuthorStepCallable,
    ComponentFieldSpec,
    ComponentStepReturn as _ComponentStepReturn,
    FieldNames as _FieldNames,
    component_field_spec as _component_field_spec,
    merge_component_outputs as _merge_component_outputs,
    normalize_author_field_values as _normalize_author_field_values,
    unique_field_names as _unique_field_names,
)
from vercor.components import _runtime_fields as _runtime_field_adapters
from vercor.components._validation import (
    validate_component_setup as validate_component_setup,
)
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import ComponentInitContext, RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.runtime import (
        RuntimeComponentContract,
        RuntimeComponentState,
    )


ComponentSetupContext = ComponentInitContext
ComponentStepContext = RuntimeStepContext
ComponentInitializeHook = Callable[[Any, ComponentInitContext], None]
ComponentCreatePayloadHook = Callable[[Any], Any | None]
ComponentPrefillHook = Callable[
    [
        Any,
        dict[str, RuntimeArray],
        dict[str, RuntimeArray],
        dict[str, RuntimeArray],
        Any,
    ],
    None,
]
ComponentValidateHook = Callable[[Any, Any, Any], None]

__all__ = [
    "Component",
    "ComponentFieldSpec",
    "ComponentSetupContext",
    "ComponentStepContext",
    "ComponentStepResult",
    "DataComponent",
    "HostRuntimeComponent",
    "data_component",
    "differentiable_component",
    "host_component",
    "validate_component_setup",
]


def _author_field_spec(
    *,
    inputs: _FieldNames = (),
    outputs: _FieldNames = (),
    default_fields: _AuthorFieldValues = None,
) -> ComponentFieldSpec:
    """Build a component field declaration from author constructor arguments."""

    return ComponentFieldSpec(
        inputs=inputs,
        outputs=outputs,
        default_fields=default_fields or {},
    )


def _install_lifecycle_hooks(
    component: "Component",
    *,
    initialize: ComponentInitializeHook | None = None,
    create_runtime_payload: ComponentCreatePayloadHook | None = None,
    prefill_runtime_state_fields: ComponentPrefillHook | None = None,
    validate_runtime_state: ComponentValidateHook | None = None,
) -> None:
    """Attach optional lifecycle hooks to a factory-created component."""

    if initialize is not None:
        setattr(component, "_initialize_hook", initialize)
    if create_runtime_payload is not None:
        setattr(component, "_create_runtime_payload_hook", create_runtime_payload)
    if prefill_runtime_state_fields is not None:
        setattr(
            component,
            "_prefill_runtime_state_fields_hook",
            prefill_runtime_state_fields,
        )
    if validate_runtime_state is not None:
        setattr(component, "_validate_runtime_state_hook", validate_runtime_state)


def _callable_component_from_model(
    *,
    runtime_kind: str,
    name: str,
    grid: RectilinearGrid,
    step: _AuthorStepCallable,
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
    """Create a callable-backed component from the shared author facade."""

    from vercor.components._callable_wrappers import _create_callable_component

    return _create_callable_component(
        runtime_kind=runtime_kind,
        name=name,
        grid=grid,
        step=step,
        payload=payload,
        settings=settings,
        field_spec=_author_field_spec(
            inputs=inputs,
            outputs=outputs,
            default_fields=default_fields,
        ),
        initialize=initialize,
        create_runtime_payload=create_runtime_payload,
        prefill_runtime_state_fields=prefill_runtime_state_fields,
        validate_runtime_state=validate_runtime_state,
    )


@dataclass
class Component(ABC):
    """Active differentiable component-author contract for VerCOR model adapters.

    Component instances own mutable setup-time metadata: name, grid, seed data,
    and component-specific settings. During coupling, the coupler copies those
    seed fields into immutable runtime state containers so JAX can trace the
    integration. Active differentiable components must implement
    :meth:`step_runtime_state` while preserving its signature. Data-only forcing
    adapters should inherit :class:`DataComponent`; non-differentiable adapters
    should inherit :class:`HostRuntimeComponent`.

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
    """

    name: str
    grid: RectilinearGrid
    data: dict[str, RuntimeArray] = field(default_factory=dict)
    settings: VercorSettings = field(default_factory=VercorSettings)
    _field_spec: ComponentFieldSpec = field(
        default_factory=ComponentFieldSpec,
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

    def declare_fields(
        self,
        field_spec: ComponentFieldSpec | None = None,
        *,
        inputs: _FieldNames = (),
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
    ) -> ComponentFieldSpec:
        """Declare runtime data fields for subclasses using author-facing names.

        The base runtime hooks use this declaration to prefill output/default
        fields and validate required fields. Subclasses with special lifecycle
        needs can still override those hooks directly.
        """

        declared = field_spec or ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            default_fields=default_fields or {},
        )
        self._field_spec = ComponentFieldSpec(
            inputs=declared.inputs,
            outputs=declared.outputs,
            default_fields=_normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=declared.default_fields,
                policy=self.settings,
            )
            or {},
        )
        return self._field_spec

    @property
    def field_spec(self) -> ComponentFieldSpec:
        """Return this component's declared author-facing runtime field contract."""

        return _component_field_spec(self)

    @property
    def field_names(self) -> tuple[str, ...]:
        """Return setup-time field names in insertion order."""

        return tuple(self.data)

    def update_settings(self, **values: object) -> Self:
        """Update component settings by name and return this component.

        This is a small convenience for component constructors that need to set
        one or more existing ``VercorSettings`` values while preserving the
        settings metadata and chainable authoring style.
        """

        for setting_name, setting_value in values.items():
            self.settings.set_value(setting_name, setting_value)
        return self

    def grid_field_defaults(
        self,
        names: _FieldNames,
        value: object = 0.0,
        overrides: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> dict[str, RuntimeArray]:
        """Return grid-shaped default fields for named runtime data fields.

        ``value`` is applied to every name, then ``overrides`` replace specific
        names. Scalars expand to this component's grid shape; array-like values
        are validated against the canonical component-data layouts.
        """

        field_names = _unique_field_names(names)
        defaults: dict[str, object] = {field_name: value for field_name in field_names}
        for field_name, field_value in (overrides or {}).items():
            if field_name not in defaults:
                raise ComponentError(
                    f"Default override field '{field_name}' is not declared for "
                    f"component '{self.name}'."
                )
            defaults[field_name] = field_value

        return (
            _normalize_author_field_values(
                component_name=self.name,
                grid=self.grid,
                fields=defaults,
                policy=self.settings if policy is None else policy,
            )
            or {}
        )

    def seed_field(
        self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed one setup-time grid field and return this component.

        Seeded fields must follow VerCOR's canonical component-data layout so
        runtime state can be created with a stable PyTree structure.
        """

        return self.seed_fields({name: value}, policy=policy)

    def seed_fields(
        self,
        fields: Mapping[str, object],
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed setup-time grid fields and return this component."""

        field_updates = _normalize_author_field_values(
            component_name=self.name,
            grid=self.grid,
            fields=fields,
            policy=self.settings if policy is None else policy,
        )
        self.data.update(field_updates or {})
        return self

    def seed_declared_defaults(
        self,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed this component's declared default fields and return itself."""

        default_fields = _component_field_spec(self).default_fields
        if default_fields:
            self.seed_fields(default_fields, policy=policy)
        return self

    def seed_zero_field(
        self,
        name: str,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed one grid-shaped zero field and return this component."""

        return self.seed_field(
            name,
            jax_zeros(
                self.grid.shape,
                self.settings if policy is None else policy,
            ),
        )

    def seed_zero_fields(
        self,
        names: _FieldNames,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed multiple grid-shaped zero fields and return this component."""

        for name in names:
            self.seed_zero_field(name, policy)
        return self

    def seed_constant_field(
        self,
        name: str,
        value: object,
        policy: PrecisionPolicy = None,
    ) -> "Component":
        """Seed one grid-shaped constant field and return this component."""

        return self.seed_field(
            name,
            jax_full(
                self.grid.shape,
                value,
                self.settings if policy is None else policy,
            ),
        )

    def runtime_fields(
        self,
        component_state: "RuntimeComponentState",
    ) -> dict[str, RuntimeArray]:
        """Return runtime data fields as a plain name-to-array mapping."""

        return _runtime_field_adapters.runtime_fields(self, component_state)

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

        return _runtime_field_adapters.has_runtime_field(self, component_state, name)

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
        """Apply a field mapping or ``ComponentStepResult`` to runtime state."""

        from vercor.components._callable_wrappers import apply_callable_step_result

        return apply_callable_step_result(self, component_state, result)

    def require_runtime_fields(
        self,
        component_state: "RuntimeComponentState",
        *names: str,
    ) -> None:
        """Validate that named runtime data fields use canonical grid layout."""

        _runtime_field_adapters.require_runtime_fields(self, component_state, *names)

    def prefill_runtime_fields(
        self,
        data: dict[str, RuntimeArray],
        field_spec: ComponentFieldSpec | None = None,
        *,
        outputs: _FieldNames = (),
        default_fields: _AuthorFieldValues = None,
        policy: PrecisionPolicy = None,
    ) -> None:
        """Prefill a mutable runtime data mapping with declared fields.

        This helper is intended for ``prefill_runtime_state_fields()`` overrides.
        Default fields are inserted first, then output fields are inserted as
        grid-shaped zeros when they are still missing.
        """

        _runtime_field_adapters.prefill_runtime_fields(
            self,
            data,
            field_spec,
            outputs=outputs,
            default_fields=default_fields,
            policy=policy,
        )

    def initialize(self, context: ComponentInitContext) -> None:
        """Optionally initialize component-owned runtime data before coupling.

        Override this hook when setup depends on coupler context such as start
        time, coupling timestep, run sequence, settings, or logger.
        """

        hook = getattr(self, "_initialize_hook", None)
        if hook is not None:
            hook(self, context)
            return
        self.seed_declared_defaults(context.settings)

    def create_runtime_payload(self) -> Any | None:
        """Return optional immutable payload carried by runtime component state.

        Override this hook for differentiable models that need non-field PyTree
        state, for example model internals or forcing containers.
        """

        hook = getattr(self, "_create_runtime_payload_hook", None)
        if hook is not None:
            return hook(self)
        return None

    def prefill_runtime_state_fields(
        self,
        data: dict[str, RuntimeArray],
        incoming: dict[str, RuntimeArray],
        outgoing: dict[str, RuntimeArray],
        contract: RuntimeComponentContract,
    ) -> None:
        """Optionally pre-seed fields required by runtime execution.

        Override this hook when a component creates fields during stepping and
        those fields must exist before the first JAX scan iteration.
        """

        hook = getattr(self, "_prefill_runtime_state_fields_hook", None)
        if hook is not None:
            hook(self, data, incoming, outgoing, contract)
            return
        _runtime_field_adapters.prefill_declared_runtime_fields(self, data)
        _ = incoming, outgoing, contract

    def validate_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        contract: RuntimeComponentContract,
    ) -> None:
        """Optionally validate component-specific runtime fields before execution.

        Override this hook to report missing payloads, diagnostic fields, or
        non-standard shapes before traced runtime execution begins.
        """

        hook = getattr(self, "_validate_runtime_state_hook", None)
        if hook is not None:
            hook(self, component_state, contract)
            return
        _ = contract
        _runtime_field_adapters.validate_declared_runtime_fields(
            self,
            component_state,
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
        fields: _AuthorFieldValues = None,
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
        _merge_component_outputs(self, fields.keys())
        return self

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Return the runtime state unchanged for data-only components."""

        _ = context
        return component_state


class HostRuntimeComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

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
    ) -> "HostRuntimeComponent":
        """Create a host-runtime component from a Python model callable."""

        return cast(
            "HostRuntimeComponent",
            _callable_component_from_model(
                runtime_kind="host",
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
            ),
        )

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Reject accidental execution on the differentiable scanned runtime."""

        _ = component_state, context
        component_name = getattr(self, "name", self.__class__.__name__)
        raise ComponentError(
            f"Component '{component_name}' is host-backed and cannot run through "
            "the differentiable scanned runtime. Use Coupler.run() so VerCOR can "
            "select the host runtime path, or implement a differentiable Component."
        )

    @abstractmethod
    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""


def data_component(
    name: str,
    grid: RectilinearGrid,
    fields: _AuthorFieldValues = None,
    settings: VercorSettings | None = None,
    *,
    initialize: ComponentInitializeHook | None = None,
    create_runtime_payload: ComponentCreatePayloadHook | None = None,
    prefill_runtime_state_fields: ComponentPrefillHook | None = None,
    validate_runtime_state: ComponentValidateHook | None = None,
) -> DataComponent:
    """Create a data-only component using the author-friendly field facade."""

    component = DataComponent.from_fields(
        name=name,
        grid=grid,
        fields=fields,
        settings=settings,
    )
    _install_lifecycle_hooks(
        component,
        initialize=initialize,
        create_runtime_payload=create_runtime_payload,
        prefill_runtime_state_fields=prefill_runtime_state_fields,
        validate_runtime_state=validate_runtime_state,
    )
    return component


def differentiable_component(
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
) -> Component:
    """Create a differentiable component using the author-friendly facade."""

    return Component.from_model(
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


def host_component(
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
) -> HostRuntimeComponent:
    """Create a host-runtime component using the author-friendly facade."""

    return HostRuntimeComponent.from_model(
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
