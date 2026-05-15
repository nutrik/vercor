import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from vercor.clock import Clock
from vercor.components.base import Component, validate_component_setup
from vercor.dtypes import as_jax_real_array
from vercor.exceptions import (
    CouplerError,
    ComponentError,
)
from vercor.exchange import Exchange
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger,
)
from vercor.run_sequence import RunSequence
from vercor.output import write_runtime_component_view_to_netcdf
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeCouplerState,
)
from vercor.runtime.contexts import ComponentInitContext
from vercor.runtime.validation import (
    check_not_empty_import_export_lists,
    check_valid_exchange_field_names,
)
from vercor.runtime.coupler_state import (
    output_masks_for_component,
    refresh_runtime_contracts,
    runtime_dispatch_context,
    runtime_state_from_components,
    validate_runtime_state as validate_coupler_runtime_state,
)
from vercor.runtime.driver import RuntimeDispatchContext, prime_runtime_outgoing
from vercor.runtime.interrupts import RuntimeInterruptController
from vercor.runtime.runner import run_coupler_runtime, run_scanned_runtime
from vercor.runtime.time import initial_runtime_step_info
from vercor.runtime.topology import (
    RuntimeRegridder,
    create_exchange_masks,
    initialize_regridders_and_masks,
    patch_exchange_masks,
    validate_land_mask_consistency,
)
from vercor.runtime.views import RuntimeComponentView
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


def _apply_run_precision_to_component(
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


@dataclass
class Coupler:
    """Public orchestration facade for configured component integrations.

    The coupler owns registration, exchange declarations, run sequence,
    regridder/mask setup, runtime-state creation, execution, and final output.
    The differentiable integration itself operates on immutable runtime state;
    component objects remain setup/configuration adapters rather than the
    traced integration state.

    Attributes:
        clock: Clock instance for managing simulation time
        log_level: logging threshold for coupler logs (e.g., "INFO", "DEBUG", etc.)
        logger: Logger instance for coupler logging
        run_sequence: sequence of component names defining the call (step) order
        components: mapping of component name to component instance
        exchanges: list of all Exchange instances
        settings: VercorSettings instance for coupler settings
        ocn_bmask_on_atm_grid: binary ocean mask regridded onto atmosphere grid
        lnd_bmask_on_atm_grid: binary land mask regridded onto atmosphere grid
        ocn_fmask_on_atm_grid: fractional ocean mask regridded onto atmosphere grid
        lnd_fmask_on_atm_grid: fractional land mask regridded onto atmosphere grid
        _regridders: mapping of (source component name, destination component name)
                to Regridder instance (a pool of all available regridders)
        _binary_masks: mapping of (source component name, destination component name)
                to a binary mask array. This mask is used during regridding of fields
                to ensure that only valid (e.g., ocean or land) points are considered
                during the regridding process.
        _fractional_masks: mapping of (source component name, destination component name)
                to a fractional mask array. This mask is applied during field exchanges
                after regridding to ensure that only the appropriate portion from source
                grid cells of the forcing or boundary conditions is transferred to
                destination grid cells, reflecting the partial coverage of source grid cells
                within destination grid cells.
        _runtime_contracts: mapping of component name to RuntimeComponentContract instance
        _compiled_runtime_cache: mapping of static runtime topology keys to cached compiled runtime functions
        _runtime_interrupts: controller for signaling and handling runtime
            interrupts across host and JAX-traced runtime paths
    """

    clock: Clock
    log_level: int | str = "INFO"
    logger: LoggerLike = field(default_factory=setup_logger)
    run_sequence: RunSequence = field(init=False)
    components: dict[str, Component] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    ocn_bmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_bmask_on_atm_grid: RuntimeArray = field(init=False)
    ocn_fmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_fmask_on_atm_grid: RuntimeArray = field(init=False)
    _regridders: dict[
        tuple[str, str, str],
        RuntimeRegridder,
    ] = field(default_factory=dict)
    _binary_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _fractional_masks: dict[tuple[str, str, str], RuntimeArray] = field(
        default_factory=dict
    )
    _runtime_contracts: dict[str, RuntimeComponentContract] = field(
        default_factory=dict
    )
    _compiled_runtime_cache: dict[
        tuple[Any, ...],
        Callable[[RuntimeCouplerState], RuntimeCouplerState],
    ] = field(default_factory=dict, init=False, repr=False)
    _runtime_interrupts: RuntimeInterruptController = field(
        default_factory=RuntimeInterruptController,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Apply the configured logging threshold at construction time."""

        if isinstance(self.logger, logging.Logger):
            self.logger = JaxCallbackLogger(
                configure_python_logger(self.logger, self.log_level)
            )
        elif isinstance(self.logger, JaxCallbackLogger):
            configure_python_logger(self.logger.logger, self.log_level)

        set_level = getattr(self.logger, "setLevel", None)
        if callable(set_level):
            set_level(self.log_level)

    def register(
        self,
        component: Component,
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        validate_component_setup(component)
        if component.name in self.components:
            raise CouplerError(f"Component {component.name} already registered")

        self.components[component.name] = component
        self.logger.info(f" Registered component {component.name}")

    def add_exchange(self, exchange: Exchange) -> None:
        """
        Add an exchange definition to the coupler.

        Arguments:
            exchange: Exchange instance defining the exchange between components to add
        """

        self.exchanges.append(exchange)
        formatted_field_names = ", ".join(
            ", ".join(item) if isinstance(item, tuple) else item
            for item in exchange.field_names
        )
        self.logger.info(
            f" Added exchange {exchange.name}: Fields ({formatted_field_names})"
        )

    def set_components_run_sequence(self, run_sequence: RunSequence) -> None:
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: RunSequence instance defining the order of components execution
        """

        for cname in run_sequence:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self.run_sequence = run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def initialize(self, enable_x64_computations: Optional[bool] = None) -> None:
        """
        Initialize the coupler and all registered components.
        """

        self.logger.info(" Initializing coupler and components")

        if enable_x64_computations is not None:
            self.settings.set_value("enable_x64", enable_x64_computations)

        self.logger.info(
            f" Setting default precision for JAX computations: {self.settings.enable_x64}"
        )

        if self.settings.enable_x64:
            import jax

            jax.config.update("jax_enable_x64", True)

        for component in self.components.values():
            _apply_run_precision_to_component(component, self.settings)

        init_context = ComponentInitContext(
            start=self.clock.start,
            dt_seconds=self.clock.dt_seconds,
            run_sequence=getattr(self, "run_sequence", RunSequence(order=[])),
            settings=self.settings,
            logger=self.logger,
        )

        # Initialize each component
        for name, component in self.components.items():
            component.initialize(init_context)

            if name not in ("ATM", "OCN", "LND", "ICE"):
                raise ComponentError(
                    f"Incorrect component name: {name}, must be ATM, OCN, LND, or ICE"
                )

            self.logger.info(f" Initialized {name}")

        self._runtime_contracts = refresh_runtime_contracts(
            self.components,
            self.exchanges,
            validate_endpoints=True,
        )

        for name, component in self.components.items():
            validate_component_setup(component)
            contract = self._runtime_contracts[name]
            check_not_empty_import_export_lists(component, contract)
            check_valid_exchange_field_names(component, contract)

        (
            self.ocn_fmask_on_atm_grid,
            self.lnd_fmask_on_atm_grid,
            self.lnd_bmask_on_atm_grid,
        ) = create_exchange_masks(self.components, logger=self.logger)
        validate_land_mask_consistency(
            self.components,
            self.lnd_bmask_on_atm_grid,
        )
        self.logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")

        initialize_regridders_and_masks(
            components=self.components,
            exchanges=self.exchanges,
            regridders=self._regridders,
            binary_masks=self._binary_masks,
            fractional_masks=self._fractional_masks,
            settings=self.settings,
            logger=self.logger,
        )

        patch_exchange_masks(
            binary_masks=self._binary_masks,
            fractional_masks=self._fractional_masks,
            ocn_fmask_on_atm_grid=self.ocn_fmask_on_atm_grid,
            lnd_bmask_on_atm_grid=self.lnd_bmask_on_atm_grid,
            lnd_fmask_on_atm_grid=self.lnd_fmask_on_atm_grid,
        )
        self.logger.info(" Exchange masks patching complete")

    def _runtime_state_from_components(
        self, *, prefill_missing: bool = False
    ) -> RuntimeCouplerState:
        self._runtime_contracts = refresh_runtime_contracts(
            self.components,
            self.exchanges,
            validate_endpoints=False,
        )
        return runtime_state_from_components(
            self.components,
            self.exchanges,
            self._fractional_masks,
            self._binary_masks,
            contracts=self._runtime_contracts,
            prefill_missing=prefill_missing,
        )

    def _validate_runtime_state(self, runtime_state: RuntimeCouplerState) -> None:
        if set(self._runtime_contracts) != set(self.components):
            self._runtime_contracts = refresh_runtime_contracts(
                self.components,
                self.exchanges,
                validate_endpoints=False,
            )

        validate_coupler_runtime_state(
            runtime_state,
            components=self.components,
            exchanges=self.exchanges,
            regridders=self._regridders,
            contracts=self._runtime_contracts,
            run_sequence=(
                tuple(self.run_sequence) if hasattr(self, "run_sequence") else None
            ),
        )

    def _prepare_runtime_state(
        self,
        initial_state: RuntimeCouplerState | None,
        *,
        validate_state: bool = True,
    ) -> RuntimeCouplerState:
        """Return a runtime state ready for execution."""

        runtime_state = (
            self.create_runtime_state(prefill_missing=True)
            if initial_state is None
            else initial_state
        )
        if validate_state:
            self._validate_runtime_state(runtime_state)
        return runtime_state

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        runtime_state = self._runtime_state_from_components(
            prefill_missing=prefill_missing
        )
        if prefill_missing and hasattr(self, "run_sequence"):
            runtime_state = prime_runtime_outgoing(
                runtime_state,
                tuple(self.run_sequence),
                dispatch_context=self._runtime_dispatch_context(),
                step_info=initial_runtime_step_info(self.clock, self.settings),
            )
        self._validate_runtime_state(runtime_state)
        return runtime_state

    def _runtime_dispatch_context(self) -> RuntimeDispatchContext:
        """Return static runtime dispatch plumbing for the current coupler state."""

        return runtime_dispatch_context(
            self.components,
            self.exchanges,
            self._regridders,
            self._runtime_contracts,
            dt_seconds=self.clock.dt_seconds,
            settings=self.settings,
        )

    def runtime_component_view(
        self,
        runtime_state: RuntimeCouplerState,
        name: str,
    ) -> RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        return RuntimeComponentView.from_component_state(
            name,
            self.components[name].grid,
            runtime_state.get_component_state(name),
        )

    def finalize(
        self,
        final_state: RuntimeCouplerState,
        output_file_mask: Optional[Path] = None,
    ) -> None:
        """
        Write final runtime component state to component output files.

        Arguments:
            final_state: runtime state returned by run/create_runtime_state
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for name, component in self.components.items():
            validate_component_setup(component)
            if output_file_mask is None:
                filepath = Path(f"{name.lower()}_component_runtime_fields.nc")
            else:
                filepath = Path(f"{name.lower()}_{output_file_mask}.nc")
            view = self.runtime_component_view(final_state, name)
            write_runtime_component_view_to_netcdf(
                view,
                filepath,
                masks=output_masks_for_component(
                    name,
                    self.exchanges,
                    self._binary_masks,
                    self._fractional_masks,
                ),
            )
            self.logger.info(f" Finalized {name}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"├── Run start: {self.clock.start}\n"
            f"├── Components: "
            + ", ".join(
                f"<{component.__class__.__name__}>({name})"
                for name, component in self.components.items()
            )
            + "\n"
            f"├── Exchanges: {', '.join(exchange.name for exchange in self.exchanges)}\n"
            f"└── Run sequence: {', '.join(self.run_sequence)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_sequence={' -> '.join(self.run_sequence)})"

    def run(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        donate_state: bool = False,
    ) -> RuntimeCouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the cached JIT-scanned runtime.
        Host-backed components run through the Python host bridge. When
        ``donate_state`` is true for pure runs, callers must treat the input
        runtime state as consumed after this method returns.
        """

        runtime_state = self._prepare_runtime_state(initial_state)
        return run_coupler_runtime(
            runtime_state,
            components=self.components,
            run_sequence=tuple(self.run_sequence),
            exchanges=self.exchanges,
            regridders=self._regridders,
            contracts=self._runtime_contracts,
            clock=self.clock,
            settings=self.settings,
            logger=self.logger,
            log_level=self.log_level,
            dispatch_context=self._runtime_dispatch_context(),
            compiled_runtime_cache=self._compiled_runtime_cache,
            interrupts=self._runtime_interrupts,
            donate_state=donate_state,
        )

    def _run_scanned_runtime(
        self,
        initial_state: RuntimeCouplerState | None = None,
        *,
        validate_state: bool = True,
    ) -> RuntimeCouplerState:
        """Run the unified scanned runtime path and return state."""

        runtime_state = self._prepare_runtime_state(
            initial_state,
            validate_state=validate_state,
        )
        return run_scanned_runtime(
            runtime_state,
            run_sequence=tuple(self.run_sequence),
            clock=self.clock,
            settings=self.settings,
            logger=self.logger,
            dispatch_context=self._runtime_dispatch_context(),
            interrupts=self._runtime_interrupts,
        )
