from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, cast, final

from vercor.components.contracts import (
    AuthorFieldValues,
    AuthorStepCallable,
    FieldNames,
)
from vercor.components._lifecycle import (
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
)
from vercor.components.base import Component
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


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

        from vercor.components.factories import _callable_component_from_model

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


__all__ = ["HostRuntimeComponent"]
