from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, final

from vercor.components.contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    ComponentCreatePayloadHook,
    ComponentFieldSpec,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    FieldNames,
)
from vercor.components._callable_wrappers import (
    _CallableRuntimeMixin,
)
from vercor.components.base import Component
from vercor.components._lifecycle import ComponentLifecycleHooks
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.contexts import ComponentStepContext
    from vercor.runtime.state import RuntimeComponentState


class HostRuntimeComponent(Component):
    """Base class for host-backed adapters that cannot run inside JAX scan."""

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
        default_fields: AuthorFieldValues = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> "HostRuntimeComponent":
        """Create a host-runtime component from a Python model callable."""

        field_spec = ComponentFieldSpec(
            inputs=inputs,
            outputs=outputs,
            default_fields=default_fields or {},
        )
        lifecycle_hooks = ComponentLifecycleHooks(
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

    @final
    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: ComponentStepContext,
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
        context: ComponentStepContext,
    ) -> "RuntimeComponentState":
        """Advance this non-differentiable host adapter by one runtime step."""


class _CallableHostRuntimeComponent(_CallableRuntimeMixin, HostRuntimeComponent):
    """Host-runtime component backed by an author-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        payload: Any | None,
        settings: VercorSettings | None,
        field_spec: ComponentFieldSpec,
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
        context: ComponentStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return self._step_callable_runtime_state(component_state, context)


__all__ = ["HostRuntimeComponent"]
