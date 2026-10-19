from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax.numpy as jnp

import vercor._runtime.facade as runtime_facade
import vercor._runtime.preparation as runtime_preparation
from vercor.coupler import Coupler
from vercor._runtime.dispatch_context import build_runtime_dispatch_context
from vercor._runtime.prepared import PreparedCoupling
from vercor.state import RunState
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.types import RuntimeArray


def prepared_coupling(coupler: Coupler) -> PreparedCoupling:
    """Return the Coupler's canonical prepared runtime boundary."""

    return coupler._ensure_prepared()


def replace_runtime_topology_maps(
    coupler: Coupler,
    *,
    regridders: Mapping[str, Any],
    binary_masks: Mapping[str, RuntimeArray] | None = None,
) -> None:
    """Install synthetic topology maps for focused runtime tests."""

    prepared = prepared_coupling(coupler)
    runtime_dtype = prepared.runtime.dtype.jax_real
    topology_maps = RuntimeTopologyMaps(
        regridders=dict(regridders),
        binary_masks=(
            dict(prepared.topology_maps.binary_masks)
            if binary_masks is None
            else {
                route_id: jnp.asarray(mask, dtype=runtime_dtype)
                for route_id, mask in binary_masks.items()
            }
        ),
    )
    dispatch_context = build_runtime_dispatch_context(
        prepared.components,
        prepared.exchanges,
        topology_maps.regridders,
        prepared.contracts,
        dt_seconds=prepared.clock.dt_seconds,
        constants=prepared.constants,
        dtype=prepared.runtime.dtype,
        topology_binary_masks=topology_maps.binary_masks,
    )
    coupler._prepared = replace(
        prepared,
        topology_maps=topology_maps,
        dispatch_context=dispatch_context,
    )


def run_scanned_coupler(
    coupler: Coupler,
    initial_state: RunState | None = None,
    *,
    validate_state: bool = True,
) -> RunState:
    """Run a coupler through the canonical scanned runtime for focused tests."""

    coupling = prepared_coupling(coupler)
    prepared_state = runtime_facade.prepare_runtime_state(
        initial_state,
        prepared=coupling,
        validate_state=validate_state,
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
