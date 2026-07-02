from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vercor.clock import Clock
from vercor.components.setup_validation import validate_component_setup
from vercor.components.contexts import ComponentSetupContext
from vercor.dtypes import as_jax_real_array
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.runtime.component_topology import validate_component_topology_names
from vercor.runtime.contracts import RuntimeComponentContract, build_runtime_contracts
from vercor.runtime.topology import build_exchange_topology
from vercor.runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
)
from vercor.runtime.validation import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
)
from vercor.settings import VercorSettings

if TYPE_CHECKING:
    from vercor.components.base import Component


@dataclass(frozen=True)
class RuntimeInitializationState:
    """Validated setup-time state required by the runtime facade."""

    runtime_contracts: dict[str, RuntimeComponentContract]
    topology: ExchangeTopologyState


def apply_run_precision_to_component(
    component: Component,
    settings: VercorSettings,
) -> None:
    """Synchronize component-owned setup arrays with the coupler precision."""

    component.settings.set_value("enable_x64", settings.enable_x64)
    component.grid = component.grid.with_precision(settings)
    component.data = {
        field_name: as_jax_real_array(field_value, settings)
        for field_name, field_value in component.data.items()
    }
    field_spec = component.field_spec
    if field_spec.default_fields:
        component.declare_fields(
            inputs=field_spec.inputs,
            outputs=field_spec.outputs,
            default_fields=field_spec.default_fields,
        )


def initialize_coupler_runtime(
    *,
    clock: Clock,
    components: dict[str, Component],
    exchanges: Sequence[Exchange],
    run_sequence: Sequence[str],
    settings: VercorSettings,
    logger: LoggerLike,
    enable_x64_computations: bool | None = None,
    topology_maps: RuntimeTopologyMaps | None = None,
) -> RuntimeInitializationState:
    """Initialize components, contracts, and exchange topology for a coupler."""

    logger.info(" Initializing coupler and components")

    if enable_x64_computations is not None:
        settings.set_value("enable_x64", enable_x64_computations)

    logger.info(
        f" Setting default precision for JAX computations: {settings.enable_x64}"
    )

    if settings.enable_x64:
        import jax

        jax.config.update("jax_enable_x64", True)

    for component in components.values():
        validate_component_setup(component)

    for component in components.values():
        apply_run_precision_to_component(component, settings)

    init_context = ComponentSetupContext(
        start=clock.start,
        dt_seconds=clock.dt_seconds,
        run_sequence=run_sequence,
        settings=settings,
        logger=logger,
    )

    for name, component in components.items():
        component.initialize(init_context)
        validate_component_setup(component)
        validate_component_topology_names({name: component})
        logger.info(f" Initialized {name}")

    runtime_contracts = build_runtime_contracts(
        tuple(components),
        exchanges,
        validate_endpoints=True,
    )

    for name, component in components.items():
        contract = runtime_contracts[name]
        check_not_empty_import_export_lists(component, contract)
        check_valid_exchange_field_names(component, contract)

    topology = build_exchange_topology(
        components=components,
        exchanges=exchanges,
        topology_maps=topology_maps,
        settings=settings,
        logger=logger,
    )
    return RuntimeInitializationState(
        runtime_contracts=runtime_contracts,
        topology=topology,
    )
