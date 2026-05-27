from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vercor.clock import Clock
from vercor.components._validation import validate_component_setup
from vercor.components.base import Component
from vercor.dtypes import as_jax_real_array
from vercor.exchange import Exchange
from vercor.jax_logging import LoggerLike
from vercor.run_sequence import RunSequence
from vercor.runtime.contexts import ComponentInitContext
from vercor.runtime.contracts import RuntimeComponentContract, build_runtime_contracts
from vercor.runtime.topology import (
    ExchangeTopologyState,
    RuntimeRegridder,
    build_exchange_topology,
    validate_component_topology_names,
)
from vercor.runtime.validation import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
)
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


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


def validate_registered_component_setup(component: Component) -> None:
    """Validate one setup component through the runtime-initialization boundary."""

    validate_component_setup(component)


def initialize_coupler_runtime(
    *,
    clock: Clock,
    components: dict[str, Component],
    exchanges: Sequence[Exchange],
    run_sequence: RunSequence,
    settings: VercorSettings,
    logger: LoggerLike,
    enable_x64_computations: bool | None = None,
    regridders: Mapping[tuple[str, str, str], RuntimeRegridder] | None = None,
    binary_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
    fractional_masks: Mapping[tuple[str, str, str], RuntimeArray] | None = None,
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
        apply_run_precision_to_component(component, settings)

    init_context = ComponentInitContext(
        start=clock.start,
        dt_seconds=clock.dt_seconds,
        run_sequence=run_sequence,
        settings=settings,
        logger=logger,
    )

    for name, component in components.items():
        component.initialize(init_context)
        validate_component_topology_names({name: component})
        logger.info(f" Initialized {name}")

    runtime_contracts = build_runtime_contracts(
        tuple(components),
        exchanges,
        validate_endpoints=True,
    )

    for name, component in components.items():
        validate_component_setup(component)
        contract = runtime_contracts[name]
        check_not_empty_import_export_lists(component, contract)
        check_valid_exchange_field_names(component, contract)

    topology = build_exchange_topology(
        components=components,
        exchanges=exchanges,
        regridders=regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
        settings=settings,
        logger=logger,
    )
    return RuntimeInitializationState(
        runtime_contracts=runtime_contracts,
        topology=topology,
    )
