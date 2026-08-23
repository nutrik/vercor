"""Single immutable prepared-coupling boundary for runtime orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor.physics import PhysicalConstants, _physical_constants_for_dtype
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.dispatch_context import (
    RuntimeDispatchContext,
    build_runtime_dispatch_context,
)
from vercor._runtime.initialization import initialize_coupler_runtime
from vercor._runtime.interrupts import RuntimeInterruptController
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.runtime import RuntimeOptions

if TYPE_CHECKING:
    from vercor.components._adapter import _ComponentBinding, _ComponentDeclaration


@dataclass(frozen=True)
class PreparedCoupling:
    """Private static runtime resources prepared once for one fixed coupler."""

    components: Mapping[str, "_ComponentBinding"]
    exchanges: tuple[Exchange, ...]
    run_order: tuple[str, ...]
    contracts: Mapping[str, ExchangeContract]
    topology_maps: RuntimeTopologyMaps
    dispatch_context: RuntimeDispatchContext
    clock: Clock
    constants: PhysicalConstants
    runtime: RuntimeOptions
    interrupts: RuntimeInterruptController


def prepare_coupling(
    *,
    components: Mapping[str, "_ComponentDeclaration"],
    exchanges: Sequence[Exchange],
    run_order: Sequence[str],
    clock: Clock,
    constants: PhysicalConstants,
    runtime: RuntimeOptions,
    logger: LoggerLike,
) -> PreparedCoupling:
    """Initialize declarations once and build an immutable runtime boundary."""

    runtime.dtype._ensure_jax_capability()
    immutable_declarations = MappingProxyType(dict(components))
    immutable_exchanges = tuple(exchanges)
    immutable_run_order = tuple(run_order)
    runtime_constants = _physical_constants_for_dtype(constants, runtime.dtype)
    initialized = initialize_coupler_runtime(
        clock=clock,
        components=dict(immutable_declarations),
        exchanges=immutable_exchanges,
        run_order=immutable_run_order,
        constants=runtime_constants,
        dtype=runtime.dtype,
        logger=logger,
        topology_policy=runtime.topology,
    )
    contracts = MappingProxyType(dict(initialized.runtime_contracts))
    immutable_components = initialized.components
    topology_maps = initialized.topology_maps
    dispatch_context = build_runtime_dispatch_context(
        immutable_components,
        immutable_exchanges,
        topology_maps.regridders,
        contracts,
        dt_seconds=clock.dt_seconds,
        constants=runtime_constants,
        dtype=runtime.dtype,
    )
    return PreparedCoupling(
        components=immutable_components,
        exchanges=immutable_exchanges,
        run_order=immutable_run_order,
        contracts=contracts,
        topology_maps=topology_maps,
        dispatch_context=dispatch_context,
        clock=clock,
        constants=runtime_constants,
        runtime=runtime,
        interrupts=RuntimeInterruptController(),
    )


__all__ = ["PreparedCoupling", "prepare_coupling"]
