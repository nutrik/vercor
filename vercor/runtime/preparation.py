from __future__ import annotations

from typing import TYPE_CHECKING

from vercor.runtime.contracts import RuntimeComponentContract, build_runtime_contracts
from vercor.runtime.coupler_state import (
    runtime_state_from_components as _runtime_state_from_components,
)
from vercor.runtime.dispatch_context import build_runtime_dispatch_context
from vercor.runtime.driver import prime_runtime_outgoing
from vercor._run_order import normalize_run_sequence
from vercor.runtime.state_validation import (
    validate_runtime_state as _validate_runtime_state,
)
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.time import initial_runtime_step_info

if TYPE_CHECKING:
    from vercor.runtime.facade import RuntimeFacadeInputs


def runtime_state_from_components(
    *,
    inputs: "RuntimeFacadeInputs",
    prefill_missing: bool,
) -> RuntimeCouplerState:
    """Build immutable runtime state from setup components and exchanges."""

    runtime_contracts = build_runtime_contracts(
        tuple(inputs.components),
        inputs.exchanges,
        validate_endpoints=False,
    )
    inputs.runtime_resources.runtime_contracts = runtime_contracts
    topology_maps = inputs.runtime_resources.topology_maps
    runtime_state = _runtime_state_from_components(
        inputs.components,
        inputs.exchanges,
        topology_maps.fractional_masks,
        contracts=inputs.runtime_resources.runtime_contracts,
        prefill_missing=prefill_missing,
    )
    return runtime_state


def validate_runtime_state(
    runtime_state: RuntimeCouplerState,
    *,
    inputs: "RuntimeFacadeInputs",
) -> dict[str, RuntimeComponentContract]:
    """Validate runtime state and return the contracts used for validation."""

    runtime_contracts = build_runtime_contracts(
        tuple(inputs.components),
        inputs.exchanges,
        validate_endpoints=False,
    )
    inputs.runtime_resources.runtime_contracts = runtime_contracts
    _validate_runtime_state(
        runtime_state,
        components=inputs.components,
        exchanges=inputs.exchanges,
        regridders=inputs.runtime_resources.topology_maps.regridders,
        contracts=inputs.runtime_resources.runtime_contracts,
        run_sequence=tuple(normalize_run_sequence(inputs.run_sequence)),
    )
    return inputs.runtime_resources.runtime_contracts


def create_runtime_state(
    *,
    inputs: "RuntimeFacadeInputs",
    prefill_missing: bool,
) -> RuntimeCouplerState:
    """Create, prime, and validate immutable runtime state."""

    runtime_state = runtime_state_from_components(
        inputs=inputs,
        prefill_missing=prefill_missing,
    )
    run_sequence = normalize_run_sequence(inputs.run_sequence)
    if prefill_missing and tuple(run_sequence):
        dispatch_context = build_runtime_dispatch_context(
            inputs.components,
            inputs.exchanges,
            inputs.runtime_resources.topology_maps.regridders,
            inputs.runtime_resources.runtime_contracts,
            dt_seconds=inputs.clock.dt_seconds,
            settings=inputs.settings,
        )
        runtime_state = prime_runtime_outgoing(
            runtime_state,
            tuple(run_sequence),
            dispatch_context=dispatch_context,
            step_info=initial_runtime_step_info(inputs.clock, inputs.settings),
        )
    validate_runtime_state(
        runtime_state,
        inputs=inputs,
    )
    return runtime_state


def prepare_runtime_state(
    initial_state: RuntimeCouplerState | None,
    *,
    inputs: "RuntimeFacadeInputs",
    validate_state: bool = True,
) -> RuntimeCouplerState:
    """Return a runtime state ready for execution."""

    if initial_state is None:
        return create_runtime_state(
            inputs=inputs,
            prefill_missing=True,
        )
    if validate_state:
        validate_runtime_state(
            initial_state,
            inputs=inputs,
        )
    return initial_state


__all__ = [
    "create_runtime_state",
    "prepare_runtime_state",
    "runtime_state_from_components",
    "validate_runtime_state",
]
