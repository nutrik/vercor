from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from vercor._runtime.component_state import create_runtime_component_state
from vercor._runtime.contracts import ExchangeContract
from vercor.state import RunState
from vercor._runtime.stores import FieldStore
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding


def runtime_state_from_components(
    components: Mapping[str, _ComponentBinding],
    fractional_masks: Mapping[str, RuntimeArray],
    *,
    contracts: Mapping[str, ExchangeContract],
    prefill_missing: bool = False,
) -> RunState:
    """Create immutable runtime state using already-prepared component contracts."""

    runtime_components = tuple(
        create_runtime_component_state(
            component,
            prefill_missing=prefill_missing,
            contract=contracts[name],
        )
        for name, component in components.items()
    )
    return RunState._from_runtime(
        component_names=tuple(components.keys()),
        components=runtime_components,
        fractional_masks=FieldStore.from_mapping(fractional_masks),
        component_grids=tuple(component.grid for component in components.values()),
    )
