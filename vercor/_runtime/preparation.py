from __future__ import annotations

from vercor._runtime.coupler_state import (
    runtime_state_from_components as _runtime_state_from_components,
)
from vercor._runtime.driver import prime_runtime_outgoing
from vercor._runtime.exchange_masks import refresh_runtime_masks
from vercor._runtime.prepared import PreparedCoupling
from vercor._runtime.state_validation import (
    validate_runtime_state as _validate_runtime_state,
)
from vercor.exceptions import CouplerError
from vercor.state import RunState
from vercor._runtime.time import initial_runtime_step_info


def runtime_state_from_components(
    *,
    prepared: PreparedCoupling,
    prefill_missing: bool,
) -> RunState:
    """Build immutable runtime state from setup components and exchanges."""

    topology_maps = prepared.topology_maps
    runtime_state = _runtime_state_from_components(
        prepared.components,
        prepared.exchanges,
        topology_maps.fractional_masks,
        contracts=prepared.contracts,
        prefill_missing=prefill_missing,
    )
    return runtime_state


def validate_runtime_state(
    runtime_state: RunState,
    *,
    prepared: PreparedCoupling,
) -> None:
    """Validate runtime state against the prepared contracts and topology."""

    _validate_runtime_state(
        runtime_state,
        components=prepared.components,
        exchanges=prepared.exchanges,
        regridders=prepared.topology_maps.regridders,
        contracts=prepared.contracts,
        run_order=prepared.run_order,
    )
    return None


def create_runtime_state(
    *,
    prepared: PreparedCoupling,
    prefill_missing: bool,
) -> RunState:
    """Create, prime, and validate immutable runtime state."""

    runtime_state = runtime_state_from_components(
        prepared=prepared,
        prefill_missing=prefill_missing,
    )
    runtime_state = refresh_runtime_masks(
        runtime_state, context=prepared.dispatch_context
    )
    runtime_state = runtime_state._with_received_masks(runtime_state._fractional_masks)
    component_names = tuple(prepared.components)
    if prefill_missing and component_names:
        runtime_state = prime_runtime_outgoing(
            runtime_state,
            component_names,
            dispatch_context=prepared.dispatch_context,
            step_info=initial_runtime_step_info(prepared.clock),
        )
    validate_runtime_state(
        runtime_state,
        prepared=prepared,
    )
    return runtime_state


def prepare_runtime_state(
    initial_state: RunState | None,
    *,
    prepared: PreparedCoupling,
    validate_state: bool = True,
) -> RunState:
    """Return a runtime state ready for execution."""

    if initial_state is None:
        return create_runtime_state(
            prepared=prepared,
            prefill_missing=True,
        )
    if not isinstance(initial_state, RunState):
        raise CouplerError(
            "Coupler.run state must be a RunState or None; "
            f"got {type(initial_state).__name__}."
        )
    if validate_state:
        validate_runtime_state(
            initial_state,
            prepared=prepared,
        )
    return refresh_runtime_masks(initial_state, context=prepared.dispatch_context)


__all__ = [
    "create_runtime_state",
    "prepare_runtime_state",
    "runtime_state_from_components",
    "validate_runtime_state",
]
