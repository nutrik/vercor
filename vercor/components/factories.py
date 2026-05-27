from __future__ import annotations

from typing import Any

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
    install_lifecycle_hooks,
)
from vercor.components.base import (
    Component,
)
from vercor.components.data import DataComponent
from vercor.components.host import HostRuntimeComponent
from vercor.grid import RectilinearGrid
from vercor.settings import VercorSettings

__all__ = [
    "data_component",
    "differentiable_component",
    "host_component",
]


def data_component(
    name: str,
    grid: RectilinearGrid,
    fields: AuthorFieldValues = None,
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
    install_lifecycle_hooks(
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
