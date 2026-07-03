from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from vercor._deprecation import warn_deprecated_name
from vercor.clock import Clock
from vercor.components.setup_validation import validate_component_setup
from vercor.exceptions import CouplerError
from vercor.exchange import Exchange
from vercor.jax_logging import (
    JaxCallbackLogger,
    LoggerLike,
    configure_python_logger,
    setup_logger as _setup_logger,
)
from vercor._run_order import normalize_run_sequence
import vercor.runtime.facade as _runtime_facade
from vercor.runtime.resources import CouplerRuntimeResources
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.components.base import Component
    import vercor.runtime.state as _runtime_state_module
    import vercor.runtime.views as _runtime_views_module


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
        lnd_bmask_on_atm_grid: binary land mask regridded onto atmosphere grid
        ocn_fmask_on_atm_grid: fractional ocean mask regridded onto atmosphere grid
        lnd_fmask_on_atm_grid: fractional land mask regridded onto atmosphere grid
        _runtime_resources: runtime-owned holder for topology maps, runtime
            contracts, and interrupt controller.
    """

    clock: Clock
    log_level: int | str = "INFO"
    logger: LoggerLike = field(default_factory=_setup_logger)
    run_sequence: Sequence[str] = field(default_factory=tuple)
    components: dict[str, Component] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    lnd_bmask_on_atm_grid: RuntimeArray = field(init=False)
    ocn_fmask_on_atm_grid: RuntimeArray = field(init=False)
    lnd_fmask_on_atm_grid: RuntimeArray = field(init=False)
    _runtime_resources: CouplerRuntimeResources = field(
        default_factory=CouplerRuntimeResources,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Apply the configured logging threshold at construction time."""

        self.run_sequence = normalize_run_sequence(self.run_sequence)

        if isinstance(self.logger, logging.Logger):
            self.logger = JaxCallbackLogger(
                configure_python_logger(self.logger, self.log_level)
            )
        elif isinstance(self.logger, JaxCallbackLogger):
            configure_python_logger(self.logger.logger, self.log_level)

        set_level = getattr(self.logger, "setLevel", None)
        if callable(set_level):
            set_level(self.log_level)

    @classmethod
    def from_components(
        cls,
        *,
        clock: Clock,
        components: Iterable["Component"],
        exchanges: Iterable[Exchange] = (),
        run_order: Sequence[str] = (),
        settings: VercorSettings | None = None,
        log_level: int | str = "INFO",
        logger: LoggerLike | None = None,
    ) -> "Coupler":
        """Create a coupler with components, exchanges, and run order configured."""

        if logger is None:
            coupler = cls(
                clock=clock,
                log_level=log_level,
                settings=settings or VercorSettings(),
            )
        else:
            coupler = cls(
                clock=clock,
                log_level=log_level,
                settings=settings or VercorSettings(),
                logger=logger,
            )
        for component in components:
            coupler.add_component(component)
        for exchange in exchanges:
            coupler.add_exchange(exchange)
        if run_order:
            coupler.set_run_order(run_order)
        return coupler

    @property
    def run_order(self) -> tuple[str, ...]:
        """Return component names in runtime execution order."""

        return tuple(self.run_sequence)

    @run_order.setter
    def run_order(self, run_order: Sequence[str]) -> None:
        """Set component names in runtime execution order."""

        self.run_sequence = normalize_run_sequence(run_order)

    def add_component(
        self,
        component: "Component",
    ) -> None:
        """Register a component with the coupler."""

        self._register_component(component)

    def register(
        self,
        component: "Component",
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

        warn_deprecated_name(
            "Coupler.register()",
            "Coupler.add_component()",
            remove_in="0.2.0",
        )
        self._register_component(component)

    def _register_component(
        self,
        component: "Component",
    ) -> None:
        """Register a component with the coupler without compatibility warnings."""

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
            for item in exchange.fields
        )
        self.logger.info(
            f" Added exchange {exchange.label}: Fields ({formatted_field_names})"
        )

    def add_exchanges(self, exchanges: Iterable[Exchange]) -> None:
        """Add multiple exchange definitions to the coupler."""

        for exchange in exchanges:
            self.add_exchange(exchange)

    def set_run_order(
        self,
        run_order: Sequence[str],
    ) -> None:
        """Set the run order for coupler components."""

        normalized_run_sequence = normalize_run_sequence(run_order)
        for cname in normalized_run_sequence:
            if cname not in self.components.keys():
                raise CouplerError(f"Component {cname} not registered in coupler")
        self.run_sequence = normalized_run_sequence
        self.logger.info(
            f" Set coupler components run sequence: {', '.join(self.run_sequence)}"
        )

    def set_components_run_sequence(
        self,
        run_sequence: Sequence[str],
    ) -> None:
        """
        Set the run sequence for the coupler components.

        Arguments:
            run_sequence: component names defining the order of components execution
        """

        warn_deprecated_name(
            "Coupler.set_components_run_sequence()",
            "Coupler.set_run_order()",
            remove_in="0.2.0",
        )
        self.set_run_order(run_sequence)

    def _runtime_inputs(self) -> _runtime_facade.RuntimeFacadeInputs:
        """Return the repeated runtime facade input bundle for this coupler."""

        self.run_sequence = normalize_run_sequence(self.run_sequence)
        return _runtime_facade.RuntimeFacadeInputs(
            self.components,
            self.exchanges,
            self._runtime_resources,
            self.run_sequence,
            self.clock,
            self.settings,
        )

    def initialize(self, enable_x64_computations: Optional[bool] = None) -> None:
        """
        Initialize the coupler and all registered components.
        """

        initialized = _runtime_facade.initialize_coupler_runtime(
            inputs=self._runtime_inputs(),
            logger=self.logger,
            enable_x64_computations=enable_x64_computations,
        )

        topology = initialized.topology
        surface_masks = topology.surface_masks
        self.ocn_fmask_on_atm_grid = surface_masks.ocn_fmask_on_atm_grid
        self.lnd_fmask_on_atm_grid = surface_masks.lnd_fmask_on_atm_grid
        self.lnd_bmask_on_atm_grid = surface_masks.lnd_bmask_on_atm_grid

    def create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        warn_deprecated_name(
            "Coupler.create_runtime_state()",
            "Coupler.state()",
            remove_in="0.2.0",
        )
        return self._create_runtime_state(prefill_missing=prefill_missing)

    def _create_runtime_state(
        self, *, prefill_missing: bool = True
    ) -> _runtime_state_module.RuntimeCouplerState:
        """Create and validate the immutable state used by the unified runtime."""

        return _runtime_facade.create_runtime_state(
            inputs=self._runtime_inputs(),
            prefill_missing=prefill_missing,
        )

    def state(self, *, prefill: bool = True) -> _runtime_state_module.CouplerState:
        """Create and validate the coupled runtime state."""

        return self._create_runtime_state(prefill_missing=prefill)

    def runtime_component_view(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        name: str,
    ) -> _runtime_views_module.RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        warn_deprecated_name(
            "Coupler.runtime_component_view()",
            "Coupler.view()",
            remove_in="0.2.0",
        )
        return self._runtime_component_view(runtime_state, name)

    def _runtime_component_view(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        name: str,
    ) -> _runtime_views_module.RuntimeComponentView:
        """Return a single object containing component metadata and runtime fields."""

        return _runtime_facade.runtime_component_view(
            components=self.components,
            runtime_state=runtime_state,
            name=name,
        )

    def view(
        self,
        state: _runtime_state_module.CouplerState,
        name: str,
    ) -> _runtime_views_module.ComponentView:
        """Return a component view for diagnostics and output."""

        return self._runtime_component_view(state, name)

    def runtime_component_views(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, _runtime_views_module.RuntimeComponentView]:
        """Return named runtime component views in component or requested order."""

        warn_deprecated_name(
            "Coupler.runtime_component_views()",
            "Coupler.views()",
            remove_in="0.2.0",
        )
        return self._runtime_component_views(runtime_state, names=names)

    def _runtime_component_views(
        self,
        runtime_state: _runtime_state_module.RuntimeCouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, _runtime_views_module.RuntimeComponentView]:
        """Return named runtime component views in component or requested order."""

        return _runtime_facade.runtime_component_views(
            components=self.components,
            runtime_state=runtime_state,
            names=names,
        )

    def views(
        self,
        state: _runtime_state_module.CouplerState,
        names: Sequence[str] | None = None,
    ) -> dict[str, _runtime_views_module.ComponentView]:
        """Return component views for diagnostics and output."""

        return self._runtime_component_views(state, names=names)

    def finalize(
        self,
        final_state: _runtime_state_module.RuntimeCouplerState,
        output_file_mask: Optional[Path] = None,
        *,
        output: Optional[Path] = None,
    ) -> None:
        """
        Write final runtime component state to component output files.

        Arguments:
            final_state: runtime state returned by ``run()`` or ``state()``
            output: optional path mask for output files
        """

        if output is not None:
            if output_file_mask is not None:
                raise TypeError("Use either output or output_file_mask, not both")
            output_file_mask = output
        elif output_file_mask is not None:
            warn_deprecated_name(
                "Coupler.finalize(output_file_mask=...)",
                "Coupler.finalize(output=...)",
                remove_in="0.2.0",
            )
        self.logger.info(" ------------ Finalizing coupler and components ------------")
        _runtime_facade.finalize(
            final_state=final_state,
            inputs=self._runtime_inputs(),
            output_file_mask=output_file_mask,
            logger=self.logger,
        )

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
            f"├── Exchanges: {', '.join(exchange.label for exchange in self.exchanges)}\n"
            f"└── Run sequence: {', '.join(self.run_sequence)}"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(runstart={self.clock.start}, run_sequence={' -> '.join(self.run_sequence)})"

    def run(
        self,
        state: _runtime_state_module.RuntimeCouplerState | None = None,
        *,
        initial_state: _runtime_state_module.RuntimeCouplerState | None = None,
    ) -> _runtime_state_module.RuntimeCouplerState:
        """
        Run all registered components through the unified runtime entrypoint.

        Pure differentiable components run through the JIT-scanned runtime.
        Host-backed components run through the Python host bridge.
        """

        if state is not None and initial_state is not None:
            raise TypeError("Use either state or initial_state, not both")
        if initial_state is not None:
            warn_deprecated_name(
                "Coupler.run(initial_state=...)",
                "Coupler.run(state=...)",
                remove_in="0.2.0",
            )
        inputs = self._runtime_inputs()
        runtime_state = _runtime_facade.prepare_runtime_state(
            state if state is not None else initial_state,
            inputs=inputs,
        )
        return _runtime_facade.run(
            runtime_state,
            inputs=inputs,
            logger=self.logger,
        )
