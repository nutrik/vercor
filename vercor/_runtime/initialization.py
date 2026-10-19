from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.components._adapter import (
    _ComponentBinding,
    _ComponentDeclaration,
    prepare_component,
)
from vercor.components.contexts import SetupContext
from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor.physics import PhysicalConstants
from vercor._runtime.contracts import ExchangeContract, build_exchange_contracts
from vercor._runtime.topology import build_exchange_topology
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor._runtime.validation import (
    check_not_empty_import_export_lists,
    validate_exchange_fields_declared,
)
from vercor.topology import TopologyPolicy

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class RuntimeInitializationState:
    """Validated setup-time state required by the runtime facade."""

    components: MappingProxyType[str, _ComponentBinding]
    runtime_contracts: dict[str, ExchangeContract]
    topology_maps: RuntimeTopologyMaps


def initialize_coupler_runtime(
    *,
    clock: Clock,
    components: dict[str, _ComponentDeclaration],
    exchanges: Sequence[Exchange],
    run_order: Sequence[str],
    constants: PhysicalConstants,
    dtype: DTypePolicy,
    logger: LoggerLike,
    topology_maps: RuntimeTopologyMaps | None = None,
    topology_policy: TopologyPolicy | None = None,
) -> RuntimeInitializationState:
    """Initialize components, contracts, and exchange topology for a coupler."""

    logger.info("Initializing coupler and components")

    logger.info(f"Setting default precision for JAX computations: {dtype.enable_x64}")

    init_context = SetupContext(
        start=clock.start,
        dt_seconds=clock.dt_seconds,
        run_order=run_order,
        constants=constants,
        dtype=dtype,
        logger=logger,
        calendar=clock.calendar,
    )

    prepared_components = {
        name: prepare_component(component, init_context, dtype)
        for name, component in components.items()
    }
    for name in prepared_components:
        logger.info(f"Initialized {name}")

    runtime_contracts = build_exchange_contracts(
        tuple(prepared_components),
        exchanges,
        validate_endpoints=True,
    )

    for name, component in prepared_components.items():
        contract = runtime_contracts[name]
        if contract.all_fields:
            check_not_empty_import_export_lists(component, contract)
            validate_exchange_fields_declared(component, contract)

    topology_maps = build_exchange_topology(
        components=prepared_components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        topology_policy=topology_policy,
        dtype=dtype,
        logger=logger,
    )
    return RuntimeInitializationState(
        components=MappingProxyType(prepared_components),
        runtime_contracts=runtime_contracts,
        topology_maps=topology_maps,
    )
