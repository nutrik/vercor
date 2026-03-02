import logging
from dataclasses import dataclass, field
from datetime import datetime
from logging import Logger
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from vercor.clock import Clock, ModelDateTime
from vercor.components import (
    Atmosphere,
    ERA5Atmosphere,
    ERA5Land,
    ERA5Ocean,
    ERAInterimOcean,
    JCMLand,
    Land,
    Ocean,
    Shared,
    JAXGCM,
)
from vercor.components import TimedNamedArray as TNA
from vercor.components.external.veros_gcm import VerosGCM
from vercor.exceptions import (
    CouplerError,
    ComponentError,
    ExchangerError,
)
from vercor.exchange import Exchange
from vercor.regridders import (
    BilinearRectilinearRegridder,
    ConservativeRectilinearRegridder,
)
from vercor.run_sequence import RunSequence
from vercor.settings import VercorSettings
from vercor.tools import (
    check_total_lnd_ocn_mask_sum,
    get_component,
    grids_identical,
    _append_unique,
    _flatten_fields,
    check_remap_conservation,
    compute_ocn_lnd_masks_on_atm_grid,
)
from vercor.types import AllComponentsType


def setup_logger() -> Logger:
    """
    Setup and return a logger for the Coupler.
    """
    logger = logging.getLogger("VerCOR")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.setLevel(logging.INFO)
    return logger


@dataclass
class Coupler:
    clock: Clock
    logger: Logger = field(default_factory=setup_logger)
    run_sequence: RunSequence = field(init=False)
    components: dict[
        str,
        AllComponentsType,
    ] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    settings: VercorSettings = field(default_factory=VercorSettings)
    ocn_bmask_on_atm_grid: NDArray = field(init=False)
    lnd_bmask_on_atm_grid: NDArray = field(init=False)
    ocn_fmask_on_atm_grid: NDArray = field(init=False)
    lnd_fmask_on_atm_grid: NDArray = field(init=False)
    _regridders: dict[
        tuple[str, str, str],
        BilinearRectilinearRegridder | ConservativeRectilinearRegridder,
    ] = field(default_factory=dict)
    _binary_masks: dict[tuple[str, str, str], NDArray] = field(default_factory=dict)
    _fractional_masks: dict[tuple[str, str, str], NDArray] = field(default_factory=dict)

    """
    Main coupler class to manage components and exchanges between them.

    Attributes:
        clock: Clock instance for managing simulation time
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
                to a binary mask NDArray. This mask is used during regridding of fields
                to ensure that only valid (e.g., ocean or land) points are considered
                during the regridding process.
        _fractional_masks: mapping of (source component name, destination component name)
                to a fractional mask NDArray. This mask is applied during field exchanges
                after regridding to ensure that only the appropriate portion from source
                grid cells of the forcing or boundary conditions is transferred to
                destination grid cells, reflecting the partial coverage of source grid cells
                within destination grid cells.
    """

    def register(
        self,
        component: AllComponentsType,
    ) -> None:
        """
        Register a component with the coupler.

        Arguments:
            component: component instance to register
        """

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

        self.logger.info(
            f" Setting default precision for JAX computations: {self.settings.enable_x64}"
        )
        if enable_x64_computations is not None:
            self.settings.enable_x64 = enable_x64_computations

        if self.settings.enable_x64:
            import jax

            jax.config.update("jax_enable_x64", True)

        # Initialize each component
        for name, component in self.components.items():
            component.initialize(self)

            if name not in ("ATM", "OCN", "LND", "ICE"):
                raise ComponentError(
                    f"Incorrect component name: {name}, must be ATM, OCN, LND, or ICE"
                )

            self.logger.info(f" Initialized {name}")

        # Setup components' import/export field lists based on exchanges
        for exchange in self.exchanges:
            if exchange.source not in self.components:
                raise CouplerError(
                    f"Source component '{exchange.source}' not registered in coupler"
                )
            if exchange.destination not in self.components:
                raise CouplerError(
                    f"Destination component '{exchange.destination}' not registered in coupler"
                )

            source_component = self.components[exchange.source]
            destination_component = self.components[exchange.destination]

            flattened_fields = _flatten_fields(exchange.field_names)
            _append_unique(source_component._fields2export, flattened_fields)
            _append_unique(destination_component._fields2import, flattened_fields)

        for name, component in self.components.items():
            component.check_not_empty_import_export_lists()
            component.check_valid_exchange_field_names()
            # Deposit initial data to be sent from component to coupler
            component.send_fields(self.clock.start, self)

        self._create_exchange_masks()
        self._validate_land_mask_consistency()
        self.logger.info(" LND <--> ATM & OCN <--> ATM masks initialization complete")

        # Build regridders per (source component, destination component) pair
        # initialize binary and fractional masks for each regridding pair
        for exchange in self.exchanges:
            key = (exchange.source, exchange.destination, exchange.interpolation_type)

            if key not in self._regridders:
                self._regridders[key] = exchange.create(
                    self.components[exchange.source].grid,
                    self.components[exchange.destination].grid,
                )
                self._binary_masks[key] = np.ones(
                    self.components[exchange.destination].grid.shape
                )
                self._fractional_masks[key] = np.ones(
                    self.components[exchange.destination].grid.shape
                )
            else:
                self.logger.warning(
                    f" Regridder for exchange {exchange.name} already exists, skipping creation"
                )

        self._patch_exchange_masks()
        self.logger.info(" Exchange masks patching complete")

    def _patch_exchange_masks(self) -> None:
        keys = self._binary_masks.keys()

        for key in keys:
            source, destination, interp_type = key
            if "bilinear" in interp_type:
                if source == "OCN" and destination == "ATM":
                    self._fractional_masks[key] = self.ocn_fmask_on_atm_grid
                elif source == "LND" and destination == "ATM":
                    self._binary_masks[key] = self.lnd_bmask_on_atm_grid
                    self._fractional_masks[key] = self.lnd_fmask_on_atm_grid

    def _create_exchange_masks(self) -> None:
        """
        Create binary and fractional masks for exchanges between
        land, ocean, and atmosphere components.
        """

        land_component = get_component(self.components, (Land, ERA5Land, JCMLand))
        atmosphere_component = get_component(
            self.components, (Atmosphere, ERA5Atmosphere, JAXGCM)
        )
        ocean_component = get_component(
            self.components, (Ocean, ERA5Ocean, ERAInterimOcean, VerosGCM)
        )

        if not grids_identical(land_component.grid, atmosphere_component.grid):
            raise CouplerError(
                "Land and atmospheric components must use identical horizontal grids"
            )

        # Remapping the binary mask from the mask origin component
        # to the destination component grid
        regridder = ConservativeRectilinearRegridder(
            ocean_component.grid,
            atmosphere_component.grid,
        )

        ocean_binary_mask = ocean_component.grid.binary_mask
        if ocean_binary_mask is None:
            raise ComponentError(
                f"Ocean component {ocean_component.name} has no binary mask defined"
            )

        (
            self.ocn_fmask_on_atm_grid,
            self.lnd_fmask_on_atm_grid,
            self.lnd_bmask_on_atm_grid,
        ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

        check_remap_conservation(
            regridder,
            np.asarray(ocean_binary_mask),
            self.ocn_fmask_on_atm_grid,
        )

        check_total_lnd_ocn_mask_sum(
            self.lnd_fmask_on_atm_grid, self.ocn_fmask_on_atm_grid
        )

    def _validate_land_mask_consistency(self) -> None:
        land_component = get_component(self.components, (Land, ERA5Land, JCMLand))
        lnd_mask_from_component = land_component.grid.binary_mask
        if lnd_mask_from_component is not None:
            component_mask = np.asarray(lnd_mask_from_component)
            if component_mask.shape != self.lnd_bmask_on_atm_grid.shape:
                raise CouplerError(
                    "Land binary mask read from component does not match atmospheric grid shape"
                )
            if not np.array_equal(component_mask, self.lnd_bmask_on_atm_grid):
                mismatch = np.count_nonzero(
                    component_mask != self.lnd_bmask_on_atm_grid
                )
                raise CouplerError(
                    "Land binary mask created from remapped ocean mask does not match component-provided mask "
                    f"(mismatched points: {mismatch})"
                )

    def append_masks_to_output(
        self,
        name: str,
        shared_fields: Shared,
    ) -> None:
        """
        Append binary and fractional masks to the output shared fields of component 'name'.

        Arguments:
            name: component name
            shared_fields: Shared instance containing fields to be written to output
        """

        for exchange in self.exchanges:
            if name != exchange.destination:
                continue

            key = (exchange.source, name, exchange.interpolation_type)
            source_destination_name = "_".join(key)

            shared_fields["bmask_" + source_destination_name] = (
                self._binary_masks[key],
                datetime.now(),
                name,
            )

            shared_fields["fmask_" + source_destination_name] = (
                self._fractional_masks[key],
                datetime.now(),
                name,
            )

    def interpolate_and_dispatch_fields(
        self,
        component: AllComponentsType,
        timestamp: datetime | ModelDateTime,
    ) -> None:
        """
        Interpolate and dispatch fields to the given component at the specified timestamp.

        Arguments:
            timestamp: current simulation (coupler's) time
            component: destination component instance to process exchanges for
        """

        for exchange in self.exchanges:
            # Ensure exchange for currently stepping component only
            if exchange.destination != component.name:
                continue

            source_component = self.components[exchange.source]
            destination_component = self.components[exchange.destination]

            self.logger.info(f" Exchange fields: {exchange.name}")

            key = (exchange.source, exchange.destination, exchange.interpolation_type)

            regrid = self._regridders[key]
            fractional_mask = self._fractional_masks[key]

            source_fields = source_component.export_fields()
            destination_fields = Shared()

            # Regridder (regrid) checks if components have identical grids internally and
            # returns fields as-is (from source to destination) if so, avoiding unnecessary computation
            for field_name in exchange.field_names:
                # Figure out if scalar or vector field to be regridded & passed to destination
                if isinstance(field_name, tuple):
                    field_name_set = set(field_name)
                    if not field_name_set.issubset(set(source_fields.fields().keys())):
                        raise ExchangerError(
                            f"Not all fields in vector {field_name} are present in source fields"
                        )
                    (
                        u_vector,
                        v_vector,
                    ) = regrid(
                        getattr(source_fields, field_name[0]).data,
                        getattr(source_fields, field_name[1]).data,
                    )
                    setattr(
                        destination_fields,
                        field_name[0],
                        TNA(u_vector, timestamp, exchange.source),
                    )
                    setattr(
                        destination_fields,
                        field_name[1],
                        TNA(v_vector, timestamp, exchange.source),
                    )
                else:
                    if field_name not in source_fields.fields().keys():
                        raise ExchangerError(
                            f"Field {field_name} not present in source fields"
                        )
                    source_field_data = getattr(source_fields, field_name).data
                    scalar = np.asarray(regrid(source_field_data)) * fractional_mask

                    setattr(
                        destination_fields,
                        field_name,
                        TNA(scalar, timestamp, exchange.source),
                    )

            if not destination_fields.is_empty:
                destination_component.import_fields(destination_fields)
                self.logger.debug(
                    f" Exchanged {destination_fields.field_names}"
                    f" from {exchange.source} to {exchange.destination}"
                )

    def finalize(self, output_file_mask: Optional[Path] = None) -> None:
        """
        Finalize the coupler and all registered components.

        Arguments:
            output_file_mask: optional path mask for output files
        """

        self.logger.info(" ------------ Finalizing coupler and components ------------")
        for name, component in self.components.items():
            component.finalize(self, output_file_mask)
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

    def run(self) -> None:
        """
        Run the coupler and all registered components according to the run sequence.
        """

        # TODO: add setup checks like time step consistency,
        # component's readiness (outgoing fields), etc.
        # Wrap in a class method or function
        for cname in self.run_sequence:
            if self.components[cname].outgoing_fields.is_empty:
                raise ComponentError(
                    f"Component {cname} outgoing fields were not initialized properly."
                )

        for n, time, dt in self.clock.iter():
            self.logger.info(
                f" ====== Step: {n:05d} ====== Date: {time} ====== Δt: {dt} "
            )

            # Step components in declared order
            for cname in self.run_sequence:
                self.interpolate_and_dispatch_fields(self.components[cname], time)

                self.logger.info(f" Run component: {cname}")
                self.components[cname].receive_fields(time)

                self.components[cname].step(dt, time, self)

                self.components[cname].send_fields(time, self)
