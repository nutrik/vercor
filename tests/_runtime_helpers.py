from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vercor.types import RuntimeArray
import vercor.runtime.facade as runtime_facade
from vercor.coupler import Coupler
from vercor.runtime.runner import run_scanned_runtime
from vercor.runtime.state import RuntimeCouplerState
from vercor.runtime.topology_state import RuntimeTopologyMaps


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


def replace_runtime_topology_maps(
    coupler: Coupler,
    *,
    regridders: Mapping[tuple[str, str, str], Any],
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
) -> None:
    """Install synthetic topology maps for focused runtime tests."""

    coupler._runtime_resources.topology_maps = RuntimeTopologyMaps(
        regridders=dict(regridders),
        binary_masks={} if binary_masks is None else dict(binary_masks),
        fractional_masks={} if fractional_masks is None else dict(fractional_masks),
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
    return run_scanned_runtime(
        prepared,
        run_sequence=tuple(coupler.run_sequence),
        clock=coupler.clock,
        settings=coupler.settings,
        logger=coupler.logger,
        dispatch_context=runtime_facade.runtime_dispatch_context(
            inputs=runtime_facade_inputs(coupler),
        ),
        interrupts=coupler._runtime_resources.interrupt_controller,
    )


def runtime_state_from_coupler_components(
    coupler: Coupler,
    *,
    prefill_missing: bool,
) -> RuntimeCouplerState:
    """Build runtime state from a Coupler's components for focused tests."""

    return runtime_facade.runtime_state_from_components(
        inputs=runtime_facade_inputs(coupler),
        prefill_missing=prefill_missing,
    )
