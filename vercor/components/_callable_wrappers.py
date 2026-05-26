from __future__ import annotations

from inspect import Parameter, signature
from typing import TYPE_CHECKING, Any, Mapping, cast

from vercor.components._contracts import (
    AuthorStepCallable,
    ComponentFieldSpec,
    ComponentStepCallable,
    ComponentStepReturn,
)
from vercor.components._runtime_fields import apply_step_result
from vercor.exceptions import ComponentError
from vercor.grid import RectilinearGrid
from vercor.runtime.contexts import RuntimeStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray
from vercor.components.base import (
    Component,
    ComponentCreatePayloadHook,
    ComponentInitializeHook,
    ComponentPrefillHook,
    ComponentValidateHook,
    HostRuntimeComponent,
)

if TYPE_CHECKING:
    from vercor.runtime import RuntimeComponentState


def normalize_component_step_callable(
    step: AuthorStepCallable,
) -> ComponentStepCallable:
    """Adapt supported author step signatures to the runtime wrapper shape."""

    try:
        step_signature = signature(step)
    except (TypeError, ValueError) as exc:
        raise ComponentError(
            "Component step callable must expose an inspectable signature that "
            "accepts 1, 2, or 3 positional arguments: fields, optional context, "
            "and optional payload."
        ) from exc

    parameters = tuple(step_signature.parameters.values())
    positional_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    )
    required_positional_parameters = tuple(
        parameter
        for parameter in positional_parameters
        if parameter.default is Parameter.empty
    )
    required_keyword_only_parameters = tuple(
        parameter
        for parameter in parameters
        if parameter.kind == Parameter.KEYWORD_ONLY
        and parameter.default is Parameter.empty
    )
    has_varargs = any(
        parameter.kind == Parameter.VAR_POSITIONAL for parameter in parameters
    )

    if required_keyword_only_parameters:
        required_names = ", ".join(
            parameter.name for parameter in required_keyword_only_parameters
        )
        raise ComponentError(
            "Component step callable has required keyword-only argument(s) "
            f"{required_names}; use 1, 2, or 3 positional arguments instead."
        )

    if has_varargs:
        if len(required_positional_parameters) > 3:
            raise _component_step_signature_error()
        arity = 3
    else:
        if (
            len(positional_parameters) < 1
            or len(positional_parameters) > 3
            or len(required_positional_parameters) > 3
        ):
            raise _component_step_signature_error()
        arity = len(positional_parameters)

    if arity == 1:

        def step_fields_only(
            fields: Mapping[str, RuntimeArray],
            context: RuntimeStepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = context, payload
            return step(fields)

        return step_fields_only

    if arity == 2:

        def step_fields_and_context(
            fields: Mapping[str, RuntimeArray],
            context: RuntimeStepContext,
            payload: Any | None,
        ) -> ComponentStepReturn:
            _ = payload
            return step(fields, context)

        return step_fields_and_context

    def step_fields_context_payload(
        fields: Mapping[str, RuntimeArray],
        context: RuntimeStepContext,
        payload: Any | None,
    ) -> ComponentStepReturn:
        return step(fields, context, payload)

    return step_fields_context_payload


def _component_step_signature_error() -> ComponentError:
    """Return a consistent author-facing error for unsupported step signatures."""

    return ComponentError(
        "Component step callable must accept 1, 2, or 3 positional arguments: "
        "fields, optional context, and optional payload."
    )


class _CallableRuntimeMixin:
    """Shared metadata hooks for callable-backed component wrappers."""

    _step: ComponentStepCallable
    _payload: Any | None

    def _initialize_callable_runtime(
        self,
        *,
        name: str,
        grid: RectilinearGrid,
        step: AuthorStepCallable,
        payload: Any | None,
        settings: VercorSettings | None,
        field_spec: ComponentFieldSpec,
        initialize: ComponentInitializeHook | None,
        create_runtime_payload: ComponentCreatePayloadHook | None,
        prefill_runtime_state_fields: ComponentPrefillHook | None,
        validate_runtime_state: ComponentValidateHook | None,
    ) -> None:
        component = cast("Component", self)
        if settings is None:
            Component.__init__(component, name=name, grid=grid)
        else:
            Component.__init__(component, name=name, grid=grid, settings=settings)
        self._step = normalize_component_step_callable(step)
        self._payload = payload
        component.declare_fields(field_spec)
        from vercor.components.factories import _install_lifecycle_hooks

        _install_lifecycle_hooks(
            component,
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill_runtime_state_fields,
            validate_runtime_state=validate_runtime_state,
        )

    def create_runtime_payload(self) -> Any | None:
        """Return the payload supplied to the callable component factory."""

        hook = getattr(self, "_create_runtime_payload_hook", None)
        if hook is not None:
            return hook(cast("Component", self))
        return self._payload

    def _step_callable_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance callable-backed runtime state using the normalized step."""

        component = cast("Component", self)
        return apply_step_result(
            component,
            component_state,
            self._step(
                component.runtime_fields(component_state),
                context,
                component_state.runtime_payload,
            ),
        )


class _CallableComponent(_CallableRuntimeMixin, Component):
    """Differentiable component backed by a user-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        field_spec: ComponentFieldSpec | None = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> None:
        self._initialize_callable_runtime(
            name=name,
            grid=grid,
            step=step,
            payload=payload,
            settings=settings,
            field_spec=field_spec or ComponentFieldSpec(),
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill_runtime_state_fields,
            validate_runtime_state=validate_runtime_state,
        )

    def step_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed differentiable component one step."""

        return self._step_callable_runtime_state(component_state, context)


class _CallableHostRuntimeComponent(
    _CallableRuntimeMixin,
    HostRuntimeComponent,
):
    """Host-runtime component backed by a user-provided step callable."""

    def __init__(
        self,
        name: str,
        grid: RectilinearGrid,
        *,
        step: AuthorStepCallable,
        payload: Any | None = None,
        settings: VercorSettings | None = None,
        field_spec: ComponentFieldSpec | None = None,
        initialize: ComponentInitializeHook | None = None,
        create_runtime_payload: ComponentCreatePayloadHook | None = None,
        prefill_runtime_state_fields: ComponentPrefillHook | None = None,
        validate_runtime_state: ComponentValidateHook | None = None,
    ) -> None:
        self._initialize_callable_runtime(
            name=name,
            grid=grid,
            step=step,
            payload=payload,
            settings=settings,
            field_spec=field_spec or ComponentFieldSpec(),
            initialize=initialize,
            create_runtime_payload=create_runtime_payload,
            prefill_runtime_state_fields=prefill_runtime_state_fields,
            validate_runtime_state=validate_runtime_state,
        )

    def step_host_runtime_state(
        self,
        component_state: "RuntimeComponentState",
        context: RuntimeStepContext,
    ) -> "RuntimeComponentState":
        """Advance this callable-backed host component one step."""

        return self._step_callable_runtime_state(component_state, context)


def _create_callable_component(
    name: str,
    grid: RectilinearGrid,
    *,
    runtime_kind: str,
    step: AuthorStepCallable,
    payload: Any | None = None,
    settings: VercorSettings | None = None,
    field_spec: ComponentFieldSpec | None = None,
    initialize: ComponentInitializeHook | None = None,
    create_runtime_payload: ComponentCreatePayloadHook | None = None,
    prefill_runtime_state_fields: ComponentPrefillHook | None = None,
    validate_runtime_state: ComponentValidateHook | None = None,
) -> "Component":
    """Create a callable-backed component for the selected runtime kind."""

    wrapper_type: type[Any]
    if runtime_kind == "differentiable":
        wrapper_type = _CallableComponent
    elif runtime_kind == "host":
        wrapper_type = _CallableHostRuntimeComponent
    else:
        raise ValueError(
            f"Unsupported callable component runtime kind {runtime_kind!r}"
        )

    return wrapper_type(
        name=name,
        grid=grid,
        step=step,
        payload=payload,
        settings=settings,
        field_spec=field_spec,
        initialize=initialize,
        create_runtime_payload=create_runtime_payload,
        prefill_runtime_state_fields=prefill_runtime_state_fields,
        validate_runtime_state=validate_runtime_state,
    )
