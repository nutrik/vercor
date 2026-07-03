from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, final

from vercor._deprecation import deprecated_getattr, warn_deprecated_name
from vercor.components.contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    ComponentHooks,
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    FieldSpec,
    FieldNames,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
)
from vercor.components._constructor_options import (
    normalize_field_spec,
    normalize_lifecycle_hooks,
)
from vercor.components.base import Component
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.contexts import StepContext
    from vercor.runtime.state import RuntimeComponentState


class HostComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

    @classmethod
    def from_step(
        cls,
        name: str,
        grid: RectilinearGrid,
        step: AuthorStepCallable,
        *,
        fields: FieldSpec | None = None,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        hooks: ComponentHooks | None = None,
        inputs: FieldNames = (),
        outputs: FieldNames = (),
        defaults: AuthorFieldValues = None,
        default_fields: AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "HostComponent":
        """Create a host-runtime component from a Python step callable."""

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
        return _CallableHostRuntimeComponent(
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
        step: AuthorStepCallable,
        *,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        inputs: FieldNames = (),
        outputs: FieldNames = (),
        defaults: AuthorFieldValues = None,
        default_fields: AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "HostComponent":
        """Create a host component from a Python model callable.

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

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
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
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""


class _CallableHostRuntimeComponent(_CallableRuntimeMixin, HostComponent):
    """Host-runtime component backed by an author-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        payload: Any | None,
        settings: VercorSettings | None,
        field_spec: FieldSpec,
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

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: StepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return self._step_callable_runtime_state(component_state, context)


__all__ = ["HostComponent"]


__getattr__ = deprecated_getattr(
    __name__,
    {
        "HostRuntimeComponent": ("vercor.components.host.HostComponent", HostComponent),
    },
    remove_in="0.2.0",
)
