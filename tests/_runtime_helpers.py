from __future__ import annotations

from dataclasses import replace

import vercor._runtime.facade as runtime_facade
import vercor._runtime.preparation as runtime_preparation
from vercor.coupler import Coupler
from vercor._runtime.prepared import PreparedCoupling
from vercor.state import RunState


def prepared_coupling(coupler: Coupler) -> PreparedCoupling:
    """Return the Coupler's canonical prepared runtime boundary."""

    return coupler._ensure_prepared()


def run_scanned_coupler(
    coupler: Coupler,
    initial_state: RunState | None = None,
) -> RunState:
    """Run a coupler through the canonical scanned runtime for focused tests."""

    coupling = prepared_coupling(coupler)
    prepared_state = runtime_facade.prepare_runtime_state(
        initial_state,
        prepared=coupling,
    )
    jax_coupling = replace(
        coupling,
        runtime=replace(coupling.runtime, backend="jax"),
    )
    return runtime_facade.run(
        prepared_state,
        prepared=jax_coupling,
        logger=coupler.logger,
    )


def runtime_state_from_coupler_components(
    coupler: Coupler,
    *,
    prefill_missing: bool,
) -> RunState:
    """Build runtime state from a Coupler's components for focused tests."""

    return runtime_preparation.runtime_state_from_components(
        prepared=prepared_coupling(coupler),
        prefill_missing=prefill_missing,
    )


def create_runtime_state_from_coupler(
    coupler: Coupler,
    *,
    prefill_missing: bool,
) -> RunState:
    """Create, prime, and validate state using the coupler's installed topology."""

    return runtime_facade.create_runtime_state(
        prepared=prepared_coupling(coupler),
        prefill_missing=prefill_missing,
    )
