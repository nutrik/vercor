from __future__ import annotations

import vercor.runtime.facade as runtime_facade
from vercor.coupler import Coupler
from vercor.runtime.state import RuntimeCouplerState


def run_scanned_coupler(
    coupler: Coupler,
    initial_state: RuntimeCouplerState | None = None,
    *,
    validate_state: bool = True,
) -> RuntimeCouplerState:
    """Run a coupler through the canonical scanned runtime for focused tests."""

    prepared = runtime_facade.prepare_runtime_state(
        initial_state,
        components=coupler.components,
        exchanges=coupler.exchanges,
        runtime_resources=coupler._runtime_resources,
        run_sequence=coupler.run_sequence,
        clock=coupler.clock,
        settings=coupler.settings,
        validate_state=validate_state,
    )
    return runtime_facade.run_scanned(
        prepared.runtime_state,
        run_sequence=coupler.run_sequence,
        clock=coupler.clock,
        logger=coupler.logger,
        components=coupler.components,
        exchanges=coupler.exchanges,
        runtime_resources=coupler._runtime_resources,
        settings=coupler.settings,
    )


def runtime_state_from_coupler_components(
    coupler: Coupler,
    *,
    prefill_missing: bool,
) -> RuntimeCouplerState:
    """Build runtime state from a Coupler's components for focused tests."""

    prepared = runtime_facade.runtime_state_from_components(
        components=coupler.components,
        exchanges=coupler.exchanges,
        runtime_resources=coupler._runtime_resources,
        prefill_missing=prefill_missing,
    )
    return prepared.runtime_state
