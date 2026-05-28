from __future__ import annotations

import vercor.runtime.facade as runtime_facade
from vercor.coupler import Coupler
from vercor.runtime.state import RuntimeCouplerState


def runtime_facade_inputs(coupler: Coupler) -> runtime_facade.RuntimeFacadeInputs:
    """Return the runtime facade inputs for focused runtime tests."""

    return runtime_facade.RuntimeFacadeInputs(
        coupler.components,
        coupler.exchanges,
        coupler._runtime_resources,
        coupler.run_sequence,
        coupler.clock,
        coupler.settings,
    )


def run_scanned_coupler(
    coupler: Coupler,
    initial_state: RuntimeCouplerState | None = None,
    *,
    validate_state: bool = True,
) -> RuntimeCouplerState:
    """Run a coupler through the canonical scanned runtime for focused tests."""

    prepared = runtime_facade.prepare_runtime_state(
        initial_state,
        inputs=runtime_facade_inputs(coupler),
        validate_state=validate_state,
    )
    return runtime_facade.run_scanned(
        prepared.runtime_state,
        inputs=runtime_facade_inputs(coupler),
        logger=coupler.logger,
    )


def runtime_state_from_coupler_components(
    coupler: Coupler,
    *,
    prefill_missing: bool,
) -> RuntimeCouplerState:
    """Build runtime state from a Coupler's components for focused tests."""

    prepared = runtime_facade.runtime_state_from_components(
        inputs=runtime_facade_inputs(coupler),
        prefill_missing=prefill_missing,
    )
    return prepared.runtime_state
