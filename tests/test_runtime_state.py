from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests._runtime_helpers import runtime_state_from_coupler_components
from tests.assertions import assert_allclose_compact
from vercor.components import ComponentSpec, TransferPolicy
from vercor.components.data import DataComponent
from vercor.clock import Clock
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exchanges import Exchange
from vercor.exceptions import CouplerError
from vercor.runtime import RuntimeOptions
from vercor.setups._external.jax_gcm_runtime import JAXGCMRuntimePayload
from vercor.setups._external.jax_gcm_state import JCMState
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.component_state import create_runtime_component_state
import vercor._runtime.field_transfer as field_transfer_module
from vercor._runtime.field_transfer import send_runtime_fields
from vercor._runtime.state import ComponentRuntimeState
from vercor._runtime.stores import FieldStore
from vercor._runtime.time import RuntimeStepInfo
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.state import RunState
from vercor.types import RuntimeArray


class _RuntimeSendComponent(DataComponent):
    def __init__(self, transfer: TransferPolicy, *, grid: Any | None = None) -> None:
        super().__init__(
            "DATA",
            make_test_grid(name="runtime-send") if grid is None else grid,
            spec=ComponentSpec(transfer=transfer),
        )


def test_runtime_contract_prefill_uses_runtime_float32_policy() -> None:
    component = DataComponent(
        name="DATA",
        grid=make_test_grid(name="runtime-prefill-policy"),
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=("DATA",),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=False)),
    )
    prepared_component = coupler._ensure_prepared().components["DATA"]
    state = create_runtime_component_state(
        prepared_component,
        prefill_missing=True,
        contract=ExchangeContract(receives=("in_field",), sends=("out_field",)),
    )

    assert state.fields.get("in_field").dtype == jnp.float32
    assert state.received.get("in_field").dtype == jnp.float32
    assert state.fields.get("out_field").dtype == jnp.float32
    assert state.sent.get("out_field").dtype == jnp.float32


def test_runtime_modules_use_current_domain_owners() -> None:
    from vercor.components import SetupContext, StepContext
    import vercor.diagnostics as diagnostics_module
    import vercor.diagnostics.fields as diagnostic_fields_module
    from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
    from vercor.setups._external.camulator import make_camulator_gcm
    from vercor.setups._external.camulator_land import make_camulator_land
    from vercor.setups._external.jax_gcm import make_jax_gcm
    from vercor.setups._external.veros_gcm import make_veros_gcm
    from vercor.setups._jcm import make_jcm_land_atmosphere

    assert SetupContext.__module__ == "vercor.components.contexts"
    assert StepContext.__module__ == "vercor.components.contexts"
    assert RuntimeTopologyMaps.__module__ == "vercor._runtime.topology_state"
    assert (
        diagnostics_module.ComponentMetric is diagnostic_fields_module.ComponentMetric
    )
    assert make_camulator_gcm.__module__ == "vercor.setups._external.camulator"
    assert make_camulator_land.__module__ == ("vercor.setups._external.camulator_land")
    assert make_jax_gcm.__module__ == "vercor.setups._external.jax_gcm"
    assert make_jcm_land_atmosphere.__module__ == "vercor.setups._jcm"
    assert make_veros_gcm.__module__ == "vercor.setups._external.veros_gcm"
    assert compute_ocean_surface_fluxes.__module__ == (
        "vercor.fluxes.bulk_formula_cesm"
    )

    runtime_source = Path("vercor/_runtime/state.py").read_text(encoding="utf-8")
    runtime_driver_source = Path("vercor/_runtime/driver.py").read_text(
        encoding="utf-8"
    )
    runtime_time_source = Path("vercor/_runtime/time.py").read_text(encoding="utf-8")
    runtime_coupler_state_path = Path("vercor/_runtime/coupler_state.py")
    runtime_backends_path = Path("vercor/_runtime/backends.py")
    runtime_execution_path = Path("vercor/_runtime/execution.py")
    runtime_runner_path = Path("vercor/_runtime/runner.py")
    assert runtime_coupler_state_path.exists()
    assert runtime_backends_path.exists()
    assert runtime_execution_path.exists()
    assert not runtime_runner_path.exists()
    runtime_coupler_state_source = runtime_coupler_state_path.read_text(
        encoding="utf-8"
    )
    runtime_backends_source = runtime_backends_path.read_text(encoding="utf-8")
    runtime_execution_source = runtime_execution_path.read_text(encoding="utf-8")
    component_contexts_path = Path("vercor/components/contexts.py")
    assert component_contexts_path.exists()
    component_contexts_source = component_contexts_path.read_text(encoding="utf-8")
    runtime_contracts_path = Path("vercor/_runtime/contracts.py")
    runtime_stores_path = Path("vercor/_runtime/stores.py")
    runtime_exchange_dispatch_path = Path("vercor/_runtime/exchange_dispatch.py")
    runtime_dispatch_context_path = Path("vercor/_runtime/dispatch_context.py")
    runtime_run_context_path = Path("vercor/_runtime/run_context.py")
    runtime_prepared_path = Path("vercor/_runtime/prepared.py")
    runtime_progress_path = Path("vercor/_runtime/progress.py")
    runtime_component_state_path = Path("vercor/_runtime/component_state.py")
    runtime_field_transfer_path = Path("vercor/_runtime/field_transfer.py")
    runtime_validation_path = Path("vercor/_runtime/validation.py")
    runtime_state_validation_path = Path("vercor/_runtime/state_validation.py")
    runtime_topology_path = Path("vercor/_runtime/topology.py")
    runtime_component_topology_path = Path("vercor/_runtime/component_topology.py")
    runtime_initialization_path = Path("vercor/_runtime/initialization.py")
    runtime_preparation_path = Path("vercor/_runtime/preparation.py")
    runtime_facade_path = Path("vercor/_runtime/facade.py")
    assert runtime_contracts_path.exists()
    assert runtime_stores_path.exists()
    assert runtime_exchange_dispatch_path.exists()
    assert runtime_dispatch_context_path.exists()
    assert runtime_run_context_path.exists()
    assert runtime_prepared_path.exists()
    assert runtime_progress_path.exists()
    assert runtime_component_state_path.exists()
    assert runtime_field_transfer_path.exists()
    assert runtime_validation_path.exists()
    assert runtime_state_validation_path.exists()
    assert runtime_topology_path.exists()
    assert not runtime_component_topology_path.exists()
    assert runtime_initialization_path.exists()
    assert runtime_preparation_path.exists()
    assert runtime_facade_path.exists()
    runtime_contracts_source = runtime_contracts_path.read_text(encoding="utf-8")
    runtime_stores_source = runtime_stores_path.read_text(encoding="utf-8")
    runtime_exchange_dispatch_source = runtime_exchange_dispatch_path.read_text(
        encoding="utf-8"
    )
    runtime_dispatch_context_source = runtime_dispatch_context_path.read_text(
        encoding="utf-8"
    )
    runtime_run_context_source = runtime_run_context_path.read_text(encoding="utf-8")
    runtime_prepared_source = runtime_prepared_path.read_text(encoding="utf-8")
    runtime_progress_source = runtime_progress_path.read_text(encoding="utf-8")
    runtime_component_state_source = runtime_component_state_path.read_text(
        encoding="utf-8"
    )
    runtime_field_transfer_source = runtime_field_transfer_path.read_text(
        encoding="utf-8"
    )
    runtime_validation_source = runtime_validation_path.read_text(encoding="utf-8")
    runtime_state_validation_source = runtime_state_validation_path.read_text(
        encoding="utf-8"
    )
    runtime_topology_source = runtime_topology_path.read_text(encoding="utf-8")
    runtime_initialization_source = runtime_initialization_path.read_text(
        encoding="utf-8"
    )
    runtime_preparation_source = runtime_preparation_path.read_text(encoding="utf-8")
    runtime_facade_source = runtime_facade_path.read_text(encoding="utf-8")
    regridder_source = Path("vercor/_regridders/base.py").read_text(encoding="utf-8")
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    component_runtime_execution_path = Path("vercor/components/runtime_execution.py")
    assert component_runtime_execution_path.exists()
    component_runtime_execution_source = component_runtime_execution_path.read_text(
        encoding="utf-8"
    )
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    components_source = Path("vercor/components/__init__.py").read_text(
        encoding="utf-8"
    )
    forcing_data_source = Path("vercor/forcing_data.py").read_text(encoding="utf-8")
    flux_source = Path("vercor/fluxes/bulk_formula_cesm.py").read_text(encoding="utf-8")
    diagnostics_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(Path("vercor/diagnostics").glob("*.py"))
    )
    output_source = Path("vercor/output/_runtime.py").read_text(encoding="utf-8")
    jax_gcm_source = Path("vercor/setups/_external/jax_gcm.py").read_text(
        encoding="utf-8"
    )
    veros_source = Path("vercor/setups/_external/veros_gcm.py").read_text(
        encoding="utf-8"
    )
    veros_setup_source = Path("vercor/setups/_external/veros_setup.py").read_text(
        encoding="utf-8"
    )
    camulator_source = Path("vercor/setups/_external/camulator.py").read_text(
        encoding="utf-8"
    )
    camulator_land_source = Path("vercor/setups/_external/camulator_land.py").read_text(
        encoding="utf-8"
    )
    veros_runtime_settings_source = Path(
        "vercor/setups/_external/veros_runtime_settings.py"
    ).read_text(encoding="utf-8")

    assert "import_fields" not in coupler_source
    assert (
        "def component_requires_host_runtime(" not in component_runtime_execution_source
    )
    assert "def host_component_names(" in component_runtime_execution_source
    assert "def step_component_runtime_state(" in component_runtime_execution_source
    assert "if TYPE_CHECKING:" in component_runtime_execution_source
    assert "from vercor.components.contracts import Component" in (
        component_runtime_execution_source
    )
    assert "_ComponentBinding" not in component_runtime_execution_source
    assert 'component.spec.execution == "host"' in component_runtime_execution_source
    assert "time is not None and isinstance" not in runtime_driver_source
    assert "def _step_runtime_component" not in coupler_source
    assert "def _runtime_step_info_from_times" not in coupler_source
    assert "def _runtime_daily_index" not in coupler_source
    assert "def _build_exchange_contracts" not in coupler_source
    assert "class ExchangeContract" in runtime_contracts_source
    assert "def flatten_exchange_fields" in runtime_contracts_source
    assert "def append_unique_runtime_fields" in runtime_contracts_source
    assert "def build_exchange_contracts" in runtime_contracts_source
    assert "def exchange_key" not in runtime_contracts_source
    assert "class ExchangeContract" not in runtime_source
    assert "def build_exchange_contracts" not in runtime_source
    assert "class FieldStore" in runtime_stores_source
    assert "class FieldStore" not in runtime_source
    assert "class ComponentRuntimeState" in runtime_source
    assert "class RunState" not in runtime_source
    assert "from vercor.state import RunState" not in runtime_source
    assert "CouplerState" not in runtime_source
    assert "RuntimeCouplerState" not in runtime_source
    assert "class RuntimeStepInfo" in runtime_time_source
    assert "class RuntimeStepInfo" not in runtime_source
    assert "def dispatch_component_exchanges" in runtime_exchange_dispatch_source
    assert "def dispatch_component_exchanges" not in runtime_source
    assert "class RuntimeDispatchContext" in runtime_dispatch_context_source
    assert "def build_runtime_dispatch_context(" in runtime_dispatch_context_source
    assert "class RuntimeDispatchContext" not in runtime_driver_source
    assert "class RuntimeDispatchContext" not in runtime_coupler_state_source
    assert "def runtime_step_info_from_times" in runtime_time_source
    assert "def step_runtime_component(" in runtime_driver_source
    assert "allow_host_runtime: bool" in runtime_driver_source
    assert "def compile_runtime" not in coupler_source
    assert "def runtime_state_from_components(" in runtime_coupler_state_source
    assert "def validate_runtime_state(" not in runtime_coupler_state_source
    assert "def validate_runtime_state(" in runtime_state_validation_source
    assert "def runtime_dispatch_context(" not in runtime_coupler_state_source
    assert "from vercor._runtime.state_validation import" not in coupler_source
    assert "from vercor._runtime.state_validation import" not in runtime_facade_source
    assert "from vercor._runtime.state_validation import" in runtime_preparation_source
    assert "import vercor._runtime.facade as _runtime_facade" in coupler_source
    assert "build_runtime_dispatch_context(" not in coupler_source
    assert "build_runtime_dispatch_context(" in runtime_prepared_source
    assert "def output_masks_for_component(" not in runtime_coupler_state_source
    assert "def output_masks_for_component(" in output_source
    assert "from vercor._runtime.coupler_state import" not in output_source
    assert "def refresh_runtime_contracts(" not in runtime_coupler_state_source
    assert "refresh_runtime_contracts(" not in coupler_source
    assert "refresh_runtime_contracts(" not in runtime_preparation_source
    assert "refresh_runtime_contracts(" not in runtime_facade_source
    assert "build_exchange_contracts(" not in coupler_source
    assert "build_exchange_contracts(" in runtime_initialization_source
    assert "build_exchange_contracts(" not in runtime_preparation_source
    assert "def prime_runtime_state(" not in runtime_coupler_state_source
    assert "prime_runtime_state(" not in coupler_source
    assert "prime_runtime_outgoing(" not in coupler_source
    assert "prime_runtime_outgoing(" in runtime_preparation_source
    assert "prime_runtime_outgoing(" not in runtime_facade_source
    assert "def execute_host_chunk(" in runtime_backends_source
    assert "def execute_jax_chunk(" in runtime_backends_source
    assert "def execute_host_chunk(" not in runtime_facade_source
    assert "def execute_jax_chunk(" not in runtime_facade_source
    assert "RuntimeRunContext(" in runtime_facade_source
    assert "build_validated_execution_plan(context)" in runtime_facade_source
    assert "execute_plan(runtime_state, plan=plan, context=context)" in (
        runtime_facade_source
    )
    assert "class RuntimeRunContext" not in runtime_facade_source
    assert "class RuntimeRunContext" in runtime_run_context_source
    assert "class PreparedCoupling" in runtime_prepared_source
    assert "components:" not in runtime_run_context_source
    assert "exchanges:" not in runtime_run_context_source
    assert "regridders:" not in runtime_run_context_source
    assert "contracts:" not in runtime_run_context_source
    assert "settings:" not in runtime_run_context_source
    assert "MutableMapping" not in runtime_run_context_source
    assert "context: RuntimeRunContext" in runtime_backends_source
    assert "from vercor._runtime.run_context import" not in coupler_source
    assert "from vercor._runtime.run_context import" in runtime_facade_source
    assert "from vercor._runtime.prepared import PreparedCoupling" in coupler_source
    assert "def _ensure_prepared(" in coupler_source
    assert "RuntimeInputs" not in runtime_facade_source
    assert "donate_state" not in coupler_source
    assert "donate_state" not in runtime_facade_source
    assert "def _run_host_runtime" not in coupler_source
    run_body = coupler_source.split("def run", 1)[1]
    assert "host_component_names(self._components)" not in run_body
    assert "host_component_names(scheduled_components)" in runtime_execution_source
    assert "def _prepare_runtime_state(" not in coupler_source
    assert "_runtime_facade.prepare_runtime_state(" in run_body
    assert "def _run_scanned_runtime(" not in coupler_source
    assert "run_coupler_runtime(" not in run_body
    assert "_runtime_facade.run(" in run_body
    assert "_runtime_facade.run_scanned(" not in coupler_source
    assert "run_coupler_runtime(" not in runtime_facade_source
    assert "run_scanned_runtime(" not in runtime_facade_source
    assert "jax.lax.scan" not in coupler_source
    assert "jax.debug.callback" not in coupler_source
    assert "jax.debug.callback" not in runtime_facade_source
    assert "jax.debug.callback" in runtime_progress_source
    assert "FieldStore.from_mapping" not in coupler_source
    assert "_runtime_step_progress_message" not in coupler_source
    assert "_runtime_component_progress_message" not in coupler_source
    assert "def runtime_step_progress_message(" not in runtime_facade_source
    assert "def runtime_component_progress_message(" not in runtime_facade_source
    assert "def log_scanned_step_progress(" not in runtime_facade_source
    assert "def log_scanned_component_progress(" not in runtime_facade_source
    assert "def runtime_step_progress_message(" in runtime_progress_source
    assert "def runtime_component_progress_message(" in runtime_progress_source
    assert "def log_scanned_step_progress(" in runtime_progress_source
    assert "def log_scanned_component_progress(" in runtime_progress_source
    assert "def _apply_scalar" not in regridder_source
    assert "def _apply_vector" not in regridder_source
    assert "handlers: dict" not in regridder_source
    assert "_sync_data_from_runtime_state" not in base_source
    assert "_fields2import" not in base_source
    assert "_fields2export" not in base_source
    assert "_fields2import" not in coupler_source
    assert "_fields2export" not in coupler_source
    assert "def to_runtime_component_state" not in base_source
    assert "def receive_runtime_fields" not in base_source
    assert "def send_runtime_fields" not in base_source
    assert "def check_not_empty_import_export_lists" not in base_source
    assert "def check_valid_exchange_field_names" not in base_source
    assert "ComponentForcingData" not in base_source
    assert "h5netcdf" not in base_source
    assert "import numpy" not in base_source
    assert "class ComponentForcingData" not in forcing_data_source
    assert "ComponentForcingData" not in components_source
    assert "def create_runtime_component_state" in runtime_component_state_source
    assert "def prefill_runtime_contract_fields" in runtime_component_state_source
    assert "def receive_runtime_fields" in runtime_field_transfer_source
    assert "def send_runtime_fields" in runtime_field_transfer_source
    assert "def validate_component_runtime_contract_fields" in runtime_validation_source
    assert "def check_not_empty_import_export_lists" in runtime_validation_source
    assert "def validate_exchange_fields_declared" in runtime_validation_source
    assert "prepare_component," in runtime_initialization_source
    assert "name: prepare_component(" in runtime_initialization_source
    assert "def initialize_coupler_runtime(" in runtime_initialization_source
    assert "from vercor.components._adapter import (" in runtime_initialization_source
    assert "validate_component_contract" not in runtime_initialization_source
    assert "from vercor.components.setup_validation import" not in (
        runtime_initialization_source
    )
    assert "from vercor._runtime.initialization import" not in coupler_source
    assert (
        "from vercor._runtime.prepared import PreparedCoupling, prepare_coupling"
        in runtime_facade_source
    )
    assert "def _apply_run_precision_to_component(" not in coupler_source
    assert "from vercor.components._validation import" not in coupler_source
    for source in (
        runtime_driver_source,
        runtime_facade_source,
        runtime_component_state_source,
        runtime_initialization_source,
    ):
        assert "from vercor.components._runtime_execution import" not in source
        assert "from vercor.components._validation import" not in source
    assert "from vercor.components.runtime_execution import" in runtime_driver_source
    assert "from vercor.components.runtime_execution import" in runtime_execution_source
    assert "from vercor.components.setup_validation import" in (
        runtime_component_state_source
    )
    assert "from vercor._runtime._components import" not in coupler_source
    assert "from vercor._runtime._components import" not in runtime_coupler_state_source
    assert "from vercor._runtime._components import" not in runtime_driver_source
    assert "from vercor._runtime._components import" not in runtime_fields_source
    assert 'def empty(cls) -> "ExchangeContract"' not in runtime_source
    assert "ExchangeContract.empty" not in coupler_source
    assert "ExchangeContract.empty" not in runtime_driver_source
    assert "def build_exchange_contracts_for_components" not in runtime_source
    assert "build_exchange_contracts_for_components" not in coupler_source
    assert "RuntimeDispatchContext" in runtime_dispatch_context_source
    assert "dispatch_context: RuntimeDispatchContext" in runtime_driver_source
    assert "contracts.get(" not in runtime_driver_source
    assert "_runtime_contracts.get(" not in coupler_source
    assert "def subset(" not in runtime_source
    assert "def to_mapping(" in runtime_stores_source
    assert "component_state.fields.to_mapping()" not in base_source
    assert "component_state.fields.to_mapping()" in runtime_fields_source
    assert "def merge(" not in runtime_source
    assert "_prepared: _PreparedCoupling | None" in coupler_source
    for field_marker in (
        "    _regridders:",
        "    _binary_masks:",
        "    _fractional_masks:",
        "    _runtime_contracts:",
        "    _runtime_interrupts:",
    ):
        assert field_marker not in coupler_source
    runtime_topology_state_source = Path("vercor/_runtime/topology_state.py").read_text(
        encoding="utf-8"
    )
    runtime_exchange_topology_source = Path(
        "vercor/_runtime/exchange_topology.py"
    ).read_text(encoding="utf-8")
    runtime_surface_masks_source = Path("vercor/_runtime/surface_masks.py").read_text(
        encoding="utf-8"
    )

    assert "def build_exchange_topology_maps(" in runtime_exchange_topology_source
    assert "class ExchangeTopologyState" not in runtime_topology_state_source
    assert "class RuntimeTopologyMaps" in runtime_topology_state_source
    assert "class SurfaceExchangeMasks" not in runtime_topology_state_source
    assert "def build_exchange_topology(" in runtime_topology_source
    assert "def create_surface_exchange_masks(" in runtime_surface_masks_source
    assert "def validate_land_mask_consistency(" in runtime_surface_masks_source
    assert "def apply_surface_exchange_masks(" not in runtime_surface_masks_source
    assert "def create_exchange_masks(" not in runtime_topology_source
    assert "def validate_land_mask_consistency(" not in runtime_topology_source
    assert "def initialize_regridders_and_masks(" not in runtime_topology_source
    assert "def patch_exchange_masks(" not in runtime_topology_source
    assert "def validate_component_topology_names(" not in runtime_topology_source
    assert "def get_component(" not in runtime_topology_source
    assert "def require_component(" not in runtime_topology_source
    assert "def _require_surface_role(" in runtime_surface_masks_source
    assert "from vercor._runtime.topology import" not in coupler_source
    assert "from vercor._runtime.topology_state import" in runtime_prepared_source
    assert "RuntimeTopologyMaps" in runtime_prepared_source
    assert "def _create_exchange_masks(" not in coupler_source
    assert "def _validate_land_mask_consistency(" not in coupler_source
    assert "def _patch_exchange_masks(" not in coupler_source
    assert "compute_ocn_lnd_masks_on_atm_grid" not in coupler_source
    assert "check_total_lnd_ocn_mask_sum" not in coupler_source
    assert "jax_ones(" not in coupler_source
    assert "class SetupContext" not in base_source
    assert "class StepContext" not in base_source
    assert "from vercor.components.contexts import StepContext" in base_source
    assert "from vercor._runtime.contexts import" not in base_source
    assert "class SetupContext" in component_contexts_source
    assert "class StepContext" in component_contexts_source
    assert "SetupContext" in components_source
    assert "StepContext" in components_source
    assert "component.initialize(self)" not in coupler_source
    assert "dt_seconds: float,\n        runtime_settings" not in base_source
    assert "def write_runtime_component_to_netcdf" not in base_source
    assert "write_runtime_component_to_netcdf" not in components_source
    assert "write_runtime_component_view_to_netcdf" not in components_source
    assert "def _write_runtime_component_to_netcdf" not in output_source
    assert "def write_runtime_component_view_to_netcdf" in output_source
    assert "def write_coupler_runtime_outputs" in output_source
    assert not Path("vercor/tools.py").exists()
    assert "class ComponentRuntimeState" not in diagnostics_source
    assert "ComponentRuntimeState =" not in diagnostics_source
    assert "ComponentRuntimeState" not in diagnostics_source
    assert "ComponentState" in diagnostics_source
    assert 'hasattr(store, "field_names")' not in diagnostics_source
    assert "elif field_name in store" not in diagnostics_source
    assert ".fields.get(" not in diagnostics_source
    assert "getattr(" not in diagnostics_source
    assert "def make_jax_gcm" in jax_gcm_source
    assert "def make_veros_gcm" in veros_source
    assert "def make_camulator_gcm" in camulator_source
    assert "def make_camulator_land" in camulator_land_source
    assert "load_camulator_forcing_context" in camulator_land_source
    assert "initialize_camulator(" not in camulator_land_source
    assert "vercor.setups._external.camulator import" not in camulator_land_source
    assert (
        "from vercor.setups._external.veros_runtime_settings import *"
        not in veros_source
    )
    assert "configure_veros_runtime" not in veros_setup_source
    veros_loader_source, veros_factory_source = veros_source.split(
        "def make_veros_gcm(", 1
    )
    assert "import vercor.setups._external.veros_gcm_state" in veros_loader_source
    assert "configure_veros_runtime()" in veros_factory_source
    assert veros_factory_source.index(
        "configure_veros_runtime()"
    ) < veros_factory_source.index("_load_veros_implementation()")
    assert "def configure_veros_runtime" in veros_runtime_settings_source
    signature = camulator_land_source.split("def step(", 1)[1].split(") ->", 1)[0]
    assert "coupler" not in signature
    assert "context" in signature
    assert "logger" not in signature
    assert "runtime_settings" not in signature
    assert "component_state.fields.to_mapping()" not in camulator_land_source
    assert "def compute_ocean_surface_fluxes" in flux_source


@pytest.mark.fast_always
def test_runtime_contracts_include_all_constructor_exchanges() -> None:
    atmosphere = DataComponent(
        name="ATM",
        grid=make_test_grid(name="contract-atm"),
        fields={"temperature": 280.0, "humidity": 0.5},
    )
    ocean = DataComponent(
        name="OCN",
        grid=make_test_grid(name="contract-ocn"),
        fields={"sea_surface_temperature": 281.0},
        spec=ComponentSpec(inputs=("temperature", "humidity")),
    )

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(atmosphere, ocean),
        exchanges=(
            Exchange(
                source="ATM",
                target="OCN",
                fields=("temperature",),
                route_id="atmosphere-temperature",
            ),
            Exchange(
                source="ATM",
                target="OCN",
                fields=("humidity",),
                route_id="atmosphere-humidity",
            ),
        ),
    )

    runtime_state_from_coupler_components(coupler, prefill_missing=True)

    assert coupler._prepared is not None
    assert coupler._prepared.contracts["ATM"].sends == (
        "temperature",
        "humidity",
    )


def test_coupler_prepared_boundary_stores_runtime_state() -> None:
    from vercor._runtime.prepared import PreparedCoupling

    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-runtime-state"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()

    assert isinstance(coupler._prepared, PreparedCoupling)
    assert coupler._prepared.contracts["MODEL"] == ExchangeContract()
    assert coupler._prepared.interrupts is not None


@pytest.mark.fast_always
def test_prepared_runtime_fields_are_frozen_and_read_only() -> None:
    component = DataComponent(
        "MODEL",
        make_test_grid(name="prepared-read-only"),
        {"temperature": 280.0},
    )
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(component,),
        run_order=("MODEL",),
    )
    coupler.initial_state()
    prepared = coupler._prepared

    assert prepared is not None
    with pytest.raises(TypeError):
        prepared.contracts["MODEL"] = ExchangeContract()  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        prepared.run_order = ()  # type: ignore[misc]
    assert not Path("vercor/_runtime/resources.py").exists()

    runtime_facade_source = Path("vercor/_runtime/facade.py").read_text(
        encoding="utf-8"
    )
    assert "runtime_resources" not in runtime_facade_source

    profile_source = Path("vercor/setups/gallery/profile_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "coupler._runtime_resources" not in profile_source


@pytest.mark.fast_always
def test_runtime_topology_maps_copying_stays_at_exchange_topology_boundary() -> None:
    topology_state_source = Path("vercor/_runtime/topology_state.py").read_text(
        encoding="utf-8"
    )
    exchange_topology_source = Path("vercor/_runtime/exchange_topology.py").read_text(
        encoding="utf-8"
    )

    assert "def from_mappings(" not in topology_state_source
    assert "RuntimeTopologyMaps.from_mappings" not in exchange_topology_source
    assert "regridders = {}" in exchange_topology_source

    regridders = cast(Any, {"ATM->OCN": object()})
    binary_masks: dict[str, RuntimeArray] = {"ATM->OCN": jnp.ones((2, 2))}
    fractional_masks: dict[str, RuntimeArray] = {"ATM->OCN": jnp.full((2, 2), 0.5)}
    topology_maps = RuntimeTopologyMaps(
        regridders=regridders,
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )

    copied = RuntimeTopologyMaps(
        regridders=dict(topology_maps.regridders),
        binary_masks=dict(topology_maps.binary_masks),
        fractional_masks=dict(topology_maps.fractional_masks),
    )

    assert copied.regridders == topology_maps.regridders
    assert copied.binary_masks == topology_maps.binary_masks
    assert copied.fractional_masks == topology_maps.fractional_masks
    assert copied.regridders is not topology_maps.regridders
    assert copied.binary_masks is not topology_maps.binary_masks
    assert copied.fractional_masks is not topology_maps.fractional_masks


def test_runtime_package_does_not_reexport_focused_module_symbols() -> None:
    runtime_module = importlib.import_module("vercor._runtime")
    component_state_module = importlib.import_module("vercor._runtime.component_state")

    assert runtime_module.__all__ == []
    assert not hasattr(runtime_module, "ExchangeContract")
    assert not hasattr(runtime_module, "FieldStore")
    assert not hasattr(runtime_module, "RuntimeStepInfo")
    assert not hasattr(runtime_module, "dispatch_component_exchanges")
    assert not Path("vercor/_runtime/components.py").exists()
    assert callable(component_state_module.create_runtime_component_state)


def test_examples_use_coupler_runtime_component_view_factory() -> None:
    slab_driver_source = Path("vercor/setups/gallery/run_slab_driver.py").read_text(
        encoding="utf-8"
    )
    data_driver_source = Path("vercor/setups/gallery/run_data_driver.py").read_text(
        encoding="utf-8"
    )
    jcm_slab_source = Path("vercor/setups/gallery/run_jcm_with_slab.py").read_text(
        encoding="utf-8"
    )

    for source in (slab_driver_source, data_driver_source, jcm_slab_source):
        assert "ComponentRuntimeState.from_coupler_state" not in source
        assert "final_state.components(" in source


def test_examples_import_component_contracts_from_canonical_owner() -> None:
    for path in Path("vercor/setups/gallery").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor import Component" not in source
        assert "from vercor import DataComponent" not in source
        assert "from vercor.components.base import" not in source
        assert "from vercor.components.data import" not in source


def test_runtime_field_store_is_immutable_pytree() -> None:
    store = FieldStore.from_mapping(
        {"temperature": jnp.asarray([[1.0, 2.0], [3.0, 4.0]])}
    )

    updated = store.set("temperature", store.get("temperature") + 1.0)

    assert store.field_names == ("temperature",)
    assert_allclose_compact(
        store.get("temperature"), np.asarray([[1.0, 2.0], [3.0, 4.0]])
    )
    assert_allclose_compact(
        updated.get("temperature"), np.asarray([[2.0, 3.0], [4.0, 5.0]])
    )

    leaves, treedef = jax.tree_util.tree_flatten(updated)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, FieldStore)
    assert restored.field_names == ("temperature",)
    assert_allclose_compact(restored.get("temperature"), updated.get("temperature"))


def test_runtime_field_store_supports_jit_updates_and_mapping_roundtrip() -> None:
    store = FieldStore.from_mapping(
        {
            "a": jnp.asarray([1.0, 2.0]),
            "b": jnp.asarray([3.0, 4.0]),
        }
    )

    def update(value: FieldStore) -> FieldStore:
        return value.set("a", value.get("a") * 2.0).set("b", value.get("b") + 1.0)

    updated = jax.jit(update)(store)

    assert updated.field_names == ("a", "b")
    assert_allclose_compact(updated.get("a"), np.asarray([2.0, 4.0]))
    assert_allclose_compact(updated.get("b"), np.asarray([4.0, 5.0]))


def test_runtime_field_store_uses_index_cache_for_bulk_set_and_pytree_restore() -> None:
    store = FieldStore.from_mapping(
        {
            "temperature": jnp.zeros((2, 2), dtype=jnp.float32),
            "humidity": jnp.ones((2, 2), dtype=jnp.float32),
        }
    )

    assert store.field_indices == {"temperature": 0, "humidity": 1}

    updated = store.set_many(
        {
            "humidity": jnp.full((2, 2), 0.75, dtype=jnp.float64),
            "pressure": jnp.full((2, 2), 101325.0, dtype=jnp.float64),
        }
    )
    leaves, treedef = jax.tree_util.tree_flatten(updated)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert updated.field_names == ("temperature", "humidity", "pressure")
    assert restored.field_indices == {
        "temperature": 0,
        "humidity": 1,
        "pressure": 2,
    }
    assert restored.get("humidity").dtype == jnp.float32
    assert restored.get("pressure").dtype == jnp.float64
    assert_allclose_compact(restored.get("humidity"), np.full((2, 2), 0.75))
    assert_allclose_compact(restored.get("pressure"), np.full((2, 2), 101325.0))


def test_runtime_field_store_replacement_preserves_existing_dtype() -> None:
    store = FieldStore.from_mapping(
        {"temperature": jnp.zeros((2, 2), dtype=jnp.float32)}
    )

    updated = store.set(
        "temperature",
        jnp.ones((2, 2), dtype=jnp.float64),
    )

    assert updated.get("temperature").dtype == jnp.float32
    assert_allclose_compact(updated.get("temperature"), np.ones((2, 2)))


def test_runtime_field_store_exposes_mapping_membership_without_default_fallbacks() -> (
    None
):
    store = FieldStore.from_mapping(
        {
            "temperature": jnp.full((2, 2), 280.0, dtype=jnp.float32),
            "humidity": jnp.full((2, 2), 0.5, dtype=jnp.float32),
        }
    )

    fields = store.to_mapping()

    assert "temperature" in store
    assert "missing" not in store
    with pytest.raises(KeyError, match="Runtime field 'missing' not found"):
        store.get("missing")
    assert tuple(fields) == ("temperature", "humidity")
    assert_allclose_compact(fields["temperature"], np.full((2, 2), 280.0))
    assert not hasattr(store, "get_or")
    assert not hasattr(store, "get_or_zeros_like")


def test_runtime_field_store_replace_helpers_preserve_dtype_and_reject_missing() -> (
    None
):
    store = FieldStore.from_mapping(
        {
            "temperature": jnp.zeros((2, 2), dtype=jnp.float32),
            "humidity": jnp.ones((2, 2), dtype=jnp.float32),
        }
    )

    updated = store.replace("temperature", jnp.full((2, 2), 281.0, dtype=jnp.float64))
    updated = updated.replace_many(
        {"humidity": jnp.full((2, 2), 0.75, dtype=jnp.float64)}
    )

    assert updated.get("temperature").dtype == jnp.float32
    assert updated.get("humidity").dtype == jnp.float32
    assert_allclose_compact(updated.get("temperature"), np.full((2, 2), 281.0))
    assert_allclose_compact(updated.get("humidity"), np.full((2, 2), 0.75))
    with pytest.raises(KeyError, match="Runtime field 'missing' not found"):
        store.replace("missing", jnp.zeros((2, 2)))
    with pytest.raises(KeyError, match="Runtime field 'missing' not found"):
        store.replace_many(
            {"temperature": jnp.ones((2, 2)), "missing": jnp.ones((2, 2))}
        )


def test_runtime_field_store_replace_helpers_are_jit_compatible() -> None:
    store = FieldStore.from_mapping(
        {
            "temperature": jnp.ones((2, 2), dtype=jnp.float32),
            "humidity": jnp.ones((2, 2), dtype=jnp.float32),
        }
    )

    def update(value: FieldStore) -> FieldStore:
        return value.replace_many(
            {
                "temperature": value.get("temperature") + 2.0,
                "humidity": jnp.zeros_like(value.get("temperature")) + 0.25,
            }
        )

    updated = jax.jit(update)(store)

    assert_allclose_compact(updated.get("temperature"), np.full((2, 2), 3.0))
    assert_allclose_compact(updated.get("humidity"), np.full((2, 2), 0.25))


def test_runtime_component_and_coupler_state_are_pytrees() -> None:
    component = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        received=FieldStore.from_mapping(
            {"sea_surface_temperature": jnp.zeros((2, 2))}
        ),
        sent=FieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
    )
    assert not hasattr(component, "name")
    assert not hasattr(component, "fields_to_import")
    assert not hasattr(component, "fields_to_export")
    state = RunState._from_runtime(
        component_names=("ATM",),
        components=(component,),
        fractional_masks=FieldStore.from_mapping({"OCN->ATM": jnp.ones((2, 2))}),
    )

    def update(value: RunState) -> RunState:
        atm = value._component_state("ATM")
        atm = atm.with_fields(
            atm.fields.set("temperature", atm.fields.get("temperature") + 2.0)
        )
        return value._with_component_state("ATM", atm)

    updated = jax.jit(update)(state)

    assert state._component_indices == {"ATM": 0}
    assert updated._component_indices == {"ATM": 0}
    assert tuple(updated.components()) == ("ATM",)
    assert_allclose_compact(
        updated._component_state("ATM").fields.get("temperature"),
        np.full((2, 2), 3.0),
    )


def test_runtime_coupler_state_restores_component_index_cache_after_pytree_roundtrip() -> (
    None
):
    component = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )
    state = RunState._from_runtime(
        component_names=("ATM", "OCN"),
        components=(component, component),
        fractional_masks=FieldStore.empty(),
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored._component_indices == {"ATM": 0, "OCN": 1}
    assert restored._component_state("OCN") is restored._components[1]


def test_runtime_component_state_preserves_optional_payload_under_jit() -> None:
    payload = JAXGCMRuntimePayload(
        jcm_state=JCMState(
            dynamics={},
            physics={},
            dycore_state={"marker": jnp.asarray(1.0)},
            physics_carry={"marker": jnp.asarray(10.0)},
        ),
        forcing={"surface_temperature": jnp.asarray([[2.0, 3.0]])},
    )
    component = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": jnp.ones((1, 2))}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
        payload=payload,
    )

    def update(value: ComponentRuntimeState) -> ComponentRuntimeState:
        runtime_payload = value.payload
        assert isinstance(runtime_payload, JAXGCMRuntimePayload)
        return value.with_payload(
            JAXGCMRuntimePayload(
                jcm_state=JCMState(
                    dynamics=runtime_payload.jcm_state.dynamics,
                    physics=runtime_payload.jcm_state.physics,
                    dycore_state={
                        "marker": runtime_payload.jcm_state.dycore_state["marker"] + 1.0
                    },
                    physics_carry={
                        "marker": runtime_payload.jcm_state.physics_carry["marker"]
                        + 2.0
                    },
                ),
                forcing=runtime_payload.forcing,
            )
        )

    updated = jax.jit(update)(component)

    assert isinstance(updated.payload, JAXGCMRuntimePayload)
    assert_allclose_compact(
        updated.payload.jcm_state.dycore_state["marker"],
        np.asarray(2.0),
    )
    assert_allclose_compact(
        updated.payload.jcm_state.physics_carry["marker"],
        np.asarray(12.0),
    )


def test_shared_runtime_field_selector_applies_every_transfer_policy() -> None:
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [2]),
    )
    forcing = jnp.arange(4 * 2 * 2, dtype=jnp.float64).reshape((4, 2, 2))

    current = field_transfer_module.select_runtime_field(
        forcing,
        TransferPolicy("current"),
        step_info,
    )
    linear = field_transfer_module.select_runtime_field(
        forcing,
        TransferPolicy("linear"),
        step_info,
    )
    daily = field_transfer_module.select_runtime_field(
        forcing,
        TransferPolicy("daily"),
        step_info,
    )
    without_step_metadata = field_transfer_module.select_runtime_field(
        forcing,
        TransferPolicy("linear"),
        None,
    )

    assert current is forcing
    assert without_step_metadata is forcing
    assert_allclose_compact(
        linear,
        0.75 * np.asarray(forcing[0]) + 0.25 * np.asarray(forcing[1]),
    )
    assert_allclose_compact(daily, np.asarray(forcing[2]))


def test_runtime_send_applies_monthly_interpolation_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(TransferPolicy("linear"))
    contract = ExchangeContract(sends=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [0]),
    )
    forcing = jnp.zeros((12, 2, 3), dtype=jnp.float64)
    forcing = forcing.at[0].set(4.0)
    forcing = forcing.at[1].set(8.0)

    def send_loss(field: jax.Array) -> jax.Array:
        state = ComponentRuntimeState(
            fields=FieldStore.from_mapping({"temperature": field}),
            received=FieldStore.empty(),
            sent=FieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.sent.get("temperature"))

    sent_state = jax.jit(
        lambda field: send_runtime_fields(
            component,
            ComponentRuntimeState(
                fields=FieldStore.from_mapping({"temperature": field}),
                received=FieldStore.empty(),
                sent=FieldStore.empty(),
            ),
            step_info,
            contract=contract,
        )
    )(forcing)
    out = sent_state.sent.get("temperature")
    gradient = jax.grad(send_loss)(forcing)

    assert out.shape == (2, 3)
    assert_allclose_compact(out, np.full((2, 3), 5.0))
    assert_allclose_compact(gradient[0], np.full((2, 3), 0.75))
    assert_allclose_compact(gradient[1], np.full((2, 3), 0.25))
    assert_allclose_compact(gradient[2:], np.zeros((10, 2, 3)))


def test_runtime_send_applies_daily_time_slice_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(TransferPolicy("daily"))
    contract = ExchangeContract(sends=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [1.0], [0.0], [2]),
    )
    forcing = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, 2, 2))

    def send_loss(field: jax.Array) -> jax.Array:
        state = ComponentRuntimeState(
            fields=FieldStore.from_mapping({"temperature": field}),
            received=FieldStore.empty(),
            sent=FieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.sent.get("temperature"))

    state = ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": forcing}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )
    sent_state = jax.jit(
        lambda value: send_runtime_fields(
            component,
            value,
            step_info,
            contract=contract,
        )
    )(state)
    gradient = jax.grad(send_loss)(forcing)

    assert_allclose_compact(sent_state.sent.get("temperature"), np.asarray(forcing[2]))
    assert_allclose_compact(gradient[2], np.ones((2, 2)))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((2, 2, 2)))


def _runtime_send_state(forcing: jax.Array) -> ComponentRuntimeState:
    """Create the immutable store layout needed for one outbound field."""

    return ComponentRuntimeState(
        fields=FieldStore.from_mapping({"temperature": forcing}),
        received=FieldStore.empty(),
        sent=FieldStore.empty(),
    )


def _daily_step_info(index: int) -> RuntimeStepInfo:
    """Return scalar step metadata selecting one daily forcing record."""

    return cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(
            lambda value: value[0],
            RuntimeStepInfo.from_sequences([0], [1], [1.0], [0.0], [index]),
        ),
    )


def test_runtime_send_rejects_nonfinite_selected_active_field() -> None:
    component = _RuntimeSendComponent(
        TransferPolicy("daily"),
        grid=make_test_grid(
            name="runtime-send-nonfinite",
            binary_mask=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
        ),
    )
    forcing = jnp.asarray(
        [
            [[1.0, jnp.nan], [2.0, jnp.nan]],
            [[jnp.nan, jnp.nan], [3.0, jnp.nan]],
        ]
    )
    state = _runtime_send_state(forcing)
    with pytest.raises(CouplerError, match="sent field 'temperature'.*active domain"):
        send_runtime_fields(
            component,
            state,
            _daily_step_info(index=1),
            contract=ExchangeContract(sends=("temperature",)),
        )
