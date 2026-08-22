from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import jax

from vercor._numerical_safety import require_active_finite
from vercor.components._adapter import _copy_owned_pytree
from vercor.components._runtime_fields import apply_step_result, runtime_fields
from vercor.components.contracts import StepResult, _KEEP_PAYLOAD
from vercor.exceptions import ComponentError

if TYPE_CHECKING:
    from vercor.components.contracts import Component
    from vercor.components.contexts import StepContext
    from vercor._runtime.state import ComponentRuntimeState


def host_component_names(
    components: Mapping[str, "Component"],
) -> list[str]:
    """Return names of components that require the Python host runtime."""

    return [
        name
        for name, component in components.items()
        if component.spec.execution == "host"
    ]


def step_component_runtime_state(
    component: "Component",
    component_state: "ComponentRuntimeState",
    context: "StepContext",
    *,
    allow_host_runtime: bool,
) -> "ComponentRuntimeState":
    """Advance ``component_state`` through the component's selected runtime path."""

    if not allow_host_runtime and component.spec.execution == "host":
        raise ComponentError(
            f"Component '{component.name}' is host-backed and cannot run "
            "through the differentiable scanned runtime. Use Coupler.run() "
            "so VerCOR can select the host runtime path, or implement a "
            "differentiable Component."
        )

    step_payload = (
        _copy_owned_pytree(component_state.payload)
        if allow_host_runtime
        else component_state.payload
    )
    result = component.step(
        runtime_fields(component_state),
        context,
        step_payload,
    )
    if (
        not allow_host_runtime
        and isinstance(result, StepResult)
        and result.payload is not _KEEP_PAYLOAD
        and cast(Any, jax.tree_util.tree_structure(result.payload))
        != cast(Any, jax.tree_util.tree_structure(component_state.payload))
    ):
        raise ComponentError(
            f"Component '{component.name}' changed its payload PyTree structure "
            "inside the differentiable scanned runtime. Return StepResult without "
            "payload to preserve it, replace payload with the same PyTree structure, "
            "or set execution='host' to clear or restructure payload state."
        )
    updated_state = apply_step_result(
        component,
        component_state,
        result,
    )
    for field_name in component.spec.outputs:
        require_active_finite(
            updated_state.fields.get(field_name),
            active_mask=component.grid.binary_mask,
            owner=f"Component '{component.name}' step output field '{field_name}'",
        )
    return updated_state
