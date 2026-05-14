from __future__ import annotations

import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests.assertions import assert_allclose_compact
from vercor.components.base import DataComponent
from vercor.runtime.contexts import ComponentInitContext
from vercor.settings import VercorSettings
from setups.external.jax_gcm import JAXGCMRuntimePayload
from vercor.runtime import (
    RuntimeComponentContract,
    RuntimeComponentState,
    RuntimeCouplerState,
    RuntimeFieldStore,
    RuntimeStepInfo,
)
from vercor.runtime.components import (
    create_runtime_component_state,
    send_runtime_fields,
)


class _RuntimeSendComponent(DataComponent):
    def __init__(self, settings: VercorSettings) -> None:
        super().__init__("DATA", make_test_grid(name="runtime-send"))
        self.settings = settings

    def initialize(self, context: ComponentInitContext) -> None:
        _ = context


def test_runtime_contract_prefill_uses_component_float32_policy() -> None:
    component = DataComponent.from_fields(
        name="DATA",
        grid=make_test_grid(name="runtime-prefill-policy"),
        settings=VercorSettings(enable_x64=False),
    )
    state = create_runtime_component_state(
        component,
        prefill_missing=True,
        contract=RuntimeComponentContract(
            imports=("in_field",), exports=("out_field",)
        ),
    )

    assert state.data.get("in_field").dtype == jnp.float32
    assert state.incoming.get("in_field").dtype == jnp.float32
    assert state.data.get("out_field").dtype == jnp.float32
    assert state.outgoing.get("out_field").dtype == jnp.float32


def test_runtime_module_does_not_own_component_specific_steps() -> None:
    runtime_source = Path("vercor/runtime/state.py").read_text(encoding="utf-8")
    runtime_components_source = Path("vercor/runtime/components.py").read_text(
        encoding="utf-8"
    )
    runtime_driver_source = Path("vercor/runtime/driver.py").read_text(encoding="utf-8")
    runtime_time_source = Path("vercor/runtime/time.py").read_text(encoding="utf-8")
    runtime_coupler_state_path = Path("vercor/runtime/coupler_state.py")
    runtime_runner_path = Path("vercor/runtime/runner.py")
    assert runtime_coupler_state_path.exists()
    assert runtime_runner_path.exists()
    runtime_coupler_state_source = runtime_coupler_state_path.read_text(
        encoding="utf-8"
    )
    runtime_runner_source = runtime_runner_path.read_text(encoding="utf-8")
    runtime_contexts_path = Path("vercor/runtime/contexts.py")
    assert runtime_contexts_path.exists()
    runtime_contexts_source = runtime_contexts_path.read_text(encoding="utf-8")
    runtime_contracts_path = Path("vercor/runtime/contracts.py")
    runtime_stores_path = Path("vercor/runtime/stores.py")
    runtime_exchange_dispatch_path = Path("vercor/runtime/exchange_dispatch.py")
    runtime_component_state_path = Path("vercor/runtime/component_state.py")
    runtime_field_transfer_path = Path("vercor/runtime/field_transfer.py")
    runtime_validation_path = Path("vercor/runtime/validation.py")
    runtime_topology_path = Path("vercor/runtime/topology.py")
    assert runtime_contracts_path.exists()
    assert runtime_stores_path.exists()
    assert runtime_exchange_dispatch_path.exists()
    assert runtime_component_state_path.exists()
    assert runtime_field_transfer_path.exists()
    assert runtime_validation_path.exists()
    assert runtime_topology_path.exists()
    runtime_contracts_source = runtime_contracts_path.read_text(encoding="utf-8")
    runtime_stores_source = runtime_stores_path.read_text(encoding="utf-8")
    runtime_exchange_dispatch_source = runtime_exchange_dispatch_path.read_text(
        encoding="utf-8"
    )
    runtime_component_state_source = runtime_component_state_path.read_text(
        encoding="utf-8"
    )
    runtime_field_transfer_source = runtime_field_transfer_path.read_text(
        encoding="utf-8"
    )
    runtime_validation_source = runtime_validation_path.read_text(encoding="utf-8")
    runtime_topology_source = runtime_topology_path.read_text(encoding="utf-8")
    regridder_source = Path("vercor/regridders/base.py").read_text(encoding="utf-8")
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")
    base_source = Path("vercor/components/base.py").read_text(encoding="utf-8")
    runtime_fields_source = Path("vercor/components/_runtime_fields.py").read_text(
        encoding="utf-8"
    )
    components_source = Path("vercor/components/__init__.py").read_text(
        encoding="utf-8"
    )
    forcing_data_source = Path("vercor/forcing_data.py").read_text(encoding="utf-8")
    flux_source = Path("vercor/fluxes/bulk_formula_cesm.py").read_text(encoding="utf-8")
    diagnostics_source = Path("vercor/diagnostics.py").read_text(encoding="utf-8")
    jax_gcm_source = Path("setups/external/jax_gcm.py").read_text(encoding="utf-8")
    veros_source = Path("setups/external/veros_gcm.py").read_text(encoding="utf-8")
    camulator_source = Path("setups/external/camulator.py").read_text(encoding="utf-8")
    camulator_land_source = Path("setups/data/camulator_land.py").read_text(
        encoding="utf-8"
    )
    veros_runtime_settings_source = Path(
        "setups/external/veros_runtime_settings.py"
    ).read_text(encoding="utf-8")
    windpp_source = Path("setups/external/windpp.py").read_text(encoding="utf-8")

    forbidden_component_markers = (
        "step_slab_component_state",
        "is_supported_differentiable_component",
        "receive_component_fields",
        "send_component_fields",
        "step_component_state",
        "JAXGCMRuntimePayload",
        "VerosGCM",
        "CAMulatorGCM",
        "CAMulatorLand",
    )
    for marker in forbidden_component_markers:
        assert marker not in runtime_source
    assert "import_fields" not in coupler_source
    assert 'hasattr(component, "step_host_runtime_state")' not in coupler_source
    assert "isinstance(component, HostRuntimeComponent)" not in coupler_source
    assert "isinstance(component, HostRuntimeComponent)" in runtime_driver_source
    assert "time is not None and isinstance" not in runtime_driver_source
    assert "def _step_runtime_component" not in coupler_source
    assert "def _runtime_step_info_from_times" not in coupler_source
    assert "def _runtime_daily_index" not in coupler_source
    assert "def _build_runtime_contracts" not in coupler_source
    assert "class RuntimeComponentContract" in runtime_contracts_source
    assert "def flatten_exchange_fields" in runtime_contracts_source
    assert "def append_unique_runtime_fields" in runtime_contracts_source
    assert "def build_runtime_contracts" in runtime_contracts_source
    assert "def exchange_key_name" in runtime_contracts_source
    assert "class RuntimeComponentContract" not in runtime_source
    assert "def build_runtime_contracts" not in runtime_source
    assert "class RuntimeFieldStore" in runtime_stores_source
    assert "class RuntimeFieldStore" not in runtime_source
    assert "class RuntimeComponentState" in runtime_source
    assert "class RuntimeCouplerState" in runtime_source
    assert "class RuntimeStepInfo" in runtime_time_source
    assert "class RuntimeStepInfo" not in runtime_source
    assert "def dispatch_component_exchanges" in runtime_exchange_dispatch_source
    assert "def dispatch_component_exchanges" not in runtime_source
    assert not Path("vercor/runtime_contracts.py").exists()
    assert not Path("vercor/runtime.py").exists()
    assert not Path("vercor/runtime_components.py").exists()
    assert not Path("vercor/runtime_contexts.py").exists()
    assert not Path("vercor/runtime_driver.py").exists()
    assert not Path("vercor/runtime_time.py").exists()
    assert not Path("vercor/runtime_views.py").exists()
    assert "def runtime_step_info_from_times" in runtime_time_source
    assert "def step_runtime_component(" in runtime_driver_source
    assert "allow_host_runtime: bool" in runtime_driver_source
    assert "def step_runtime_component_pure" not in runtime_driver_source
    assert "def step_runtime_component_host_enabled" not in runtime_driver_source
    assert "def compile_runtime" not in coupler_source
    assert "def runtime_state_from_components(" in runtime_coupler_state_source
    assert "def validate_runtime_state(" in runtime_coupler_state_source
    assert "def runtime_dispatch_context(" in runtime_coupler_state_source
    assert "def output_masks_for_component(" in runtime_coupler_state_source
    assert "def run_host_runtime(" in runtime_runner_source
    assert "def run_scanned_runtime(" in runtime_runner_source
    assert "def run_coupler_runtime(" in runtime_runner_source
    assert "def compiled_scanned_runtime(" in runtime_runner_source
    assert "def compiled_runtime_cache_key(" in runtime_runner_source
    assert "def _run_host_runtime" not in coupler_source
    assert "def _compiled_runtime_cache_key" not in coupler_source
    run_body = coupler_source.split("def run", 1)[1]
    scanned_body = coupler_source.split("def _run_scanned_runtime", 1)[1]
    assert "host_component_names(self.components)" not in run_body
    assert "host_component_names(components)" in runtime_runner_source
    assert "run_coupler_runtime(" in run_body
    assert "run_scanned_runtime(" in scanned_body
    assert "jax.lax.scan" not in coupler_source
    assert "jax.debug.callback" not in coupler_source
    assert "RuntimeFieldStore.from_mapping" not in coupler_source
    assert "build_runtime_contracts(" not in coupler_source
    assert "_runtime_step_progress_message" not in coupler_source
    assert "_runtime_component_progress_message" not in coupler_source
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
    assert "class ComponentForcingData" in forcing_data_source
    assert "ComponentForcingData" not in components_source
    assert "def create_runtime_component_state" in runtime_component_state_source
    assert "def prefill_runtime_contract_fields" in runtime_component_state_source
    assert "def receive_runtime_fields" in runtime_field_transfer_source
    assert "def send_runtime_fields" in runtime_field_transfer_source
    assert "def validate_component_runtime_contract_fields" in runtime_validation_source
    assert "def check_not_empty_import_export_lists" in runtime_validation_source
    assert "def check_valid_exchange_field_names" in runtime_validation_source
    assert "def create_runtime_component_state" not in runtime_components_source
    assert "def receive_runtime_fields" not in runtime_components_source
    assert "def send_runtime_fields" not in runtime_components_source
    assert (
        "def validate_component_runtime_contract_fields"
        not in runtime_components_source
    )
    assert "from vercor.runtime.components import" not in coupler_source
    assert "from vercor.runtime.components import" not in runtime_coupler_state_source
    assert "from vercor.runtime.components import" not in runtime_driver_source
    assert "from vercor.runtime.components import" not in runtime_fields_source
    assert 'def empty(cls) -> "RuntimeComponentContract"' not in runtime_source
    assert "RuntimeComponentContract.empty" not in coupler_source
    assert "RuntimeComponentContract.empty" not in runtime_driver_source
    assert "def build_runtime_contracts_for_components" not in runtime_source
    assert "build_runtime_contracts_for_components" not in coupler_source
    assert "RuntimeDispatchContext" in runtime_driver_source
    assert "dispatch_context: RuntimeDispatchContext" in runtime_driver_source
    assert "contracts.get(" not in runtime_driver_source
    assert "_runtime_contracts.get(" not in coupler_source
    assert "def subset(" not in runtime_source
    assert "def to_mapping(" in runtime_stores_source
    assert "component_state.data.to_mapping()" not in base_source
    assert "component_state.data.to_mapping()" in runtime_fields_source
    assert "def merge(" not in runtime_source
    assert "_runtime_contracts" in coupler_source
    assert "def initialize_regridders_and_masks(" in runtime_topology_source
    assert "def create_exchange_masks(" in runtime_topology_source
    assert "def validate_land_mask_consistency(" in runtime_topology_source
    assert "def patch_exchange_masks(" in runtime_topology_source
    assert "from vercor.runtime.topology import" in coupler_source
    assert "compute_ocn_lnd_masks_on_atm_grid" not in coupler_source
    assert "check_total_lnd_ocn_mask_sum" not in coupler_source
    assert "jax_ones(" not in coupler_source
    assert "class ComponentInitContext" not in base_source
    assert "class RuntimeStepContext" not in base_source
    assert "from vercor.runtime.contexts import" in base_source
    assert "class ComponentInitContext" in runtime_contexts_source
    assert "class RuntimeStepContext" in runtime_contexts_source
    assert "ComponentInitContext" not in components_source
    assert "RuntimeStepContext" not in components_source
    assert "component.initialize(self)" not in coupler_source
    assert "dt_seconds: float,\n        runtime_settings" not in base_source
    assert "def write_runtime_component_to_netcdf" not in base_source
    assert "write_runtime_component_to_netcdf" not in components_source
    assert "write_runtime_component_view_to_netcdf" not in components_source
    assert not Path("vercor/tools.py").exists()
    assert "class RuntimeComponentView" not in diagnostics_source
    assert "RuntimeComponentView =" not in diagnostics_source
    assert "RuntimeComponentView" in diagnostics_source
    assert 'hasattr(store, "field_names")' not in diagnostics_source
    assert "elif field_name in store" not in diagnostics_source
    assert "def runtime_contract" not in runtime_components_source
    assert "RuntimeComponentContract | None" not in runtime_components_source
    assert "def make_jax_gcm" in jax_gcm_source
    assert "def make_veros_gcm" in veros_source
    assert "def make_camulator_gcm" in camulator_source
    assert "def make_camulator_land" in camulator_land_source
    assert "load_camulator_forcing_context" in camulator_land_source
    assert "initialize_camulator" not in camulator_land_source
    assert "setups.external.camulator import" not in camulator_land_source
    assert "from setups.external.veros_runtime_settings import *" not in veros_source
    assert (
        "from setups.external.veros_runtime_settings import configure_veros_runtime"
        in veros_source
    )
    assert veros_source.index("configure_veros_runtime()") < veros_source.index(
        "from veros.setups.global_4deg import GlobalFourDegreeSetup"
    )
    assert "def configure_veros_runtime" in veros_runtime_settings_source
    assert "def _step_host_runtime_state" not in base_source
    assert "_step_host_runtime_state" not in runtime_driver_source
    for source in (veros_source, camulator_source, camulator_land_source):
        signature = source.split("def step(", 1)[1].split(") ->", 1)[0]
        assert "coupler" not in signature
        assert "context" in signature
        assert "logger" not in signature
        assert "runtime_settings" not in signature
    assert "def step_runtime_state" not in veros_source
    assert "def step_runtime_state" not in camulator_source
    assert "def step_runtime_state" not in camulator_land_source
    assert "component_state.data.to_mapping()" not in camulator_land_source
    assert "post_process_wind_artifacts_deprecated" not in windpp_source
    assert "old_flux_atmOcn" not in flux_source
    assert "new_flux_atmOcn" not in flux_source
    assert "def compute_ocean_surface_fluxes" in flux_source


def test_runtime_focused_modules_keep_compatibility_reexports() -> None:
    runtime_module = importlib.import_module("vercor.runtime")
    contracts_module = importlib.import_module("vercor.runtime.contracts")
    stores_module = importlib.import_module("vercor.runtime.stores")
    time_module = importlib.import_module("vercor.runtime.time")
    exchange_dispatch_module = importlib.import_module(
        "vercor.runtime.exchange_dispatch"
    )
    components_module = importlib.import_module("vercor.runtime.components")
    component_state_module = importlib.import_module("vercor.runtime.component_state")
    field_transfer_module = importlib.import_module("vercor.runtime.field_transfer")
    validation_module = importlib.import_module("vercor.runtime.validation")

    assert runtime_module.RuntimeComponentContract is (
        contracts_module.RuntimeComponentContract
    )
    assert runtime_module.RuntimeFieldStore is stores_module.RuntimeFieldStore
    assert runtime_module.RuntimeStepInfo is time_module.RuntimeStepInfo
    assert runtime_module.dispatch_component_exchanges is (
        exchange_dispatch_module.dispatch_component_exchanges
    )
    assert components_module.create_runtime_component_state is (
        component_state_module.create_runtime_component_state
    )
    assert components_module.receive_runtime_fields is (
        field_transfer_module.receive_runtime_fields
    )
    assert (
        components_module.send_runtime_fields
        is field_transfer_module.send_runtime_fields
    )
    assert components_module.validate_component_runtime_contract_fields is (
        validation_module.validate_component_runtime_contract_fields
    )


def test_examples_use_coupler_runtime_component_view_factory() -> None:
    slab_driver_source = Path("setups/run_slab_driver.py").read_text(encoding="utf-8")
    data_driver_source = Path("setups/run_data_driver.py").read_text(encoding="utf-8")
    jcm_slab_source = Path("setups/run_jcm_with_slab.py").read_text(encoding="utf-8")

    for source in (slab_driver_source, data_driver_source, jcm_slab_source):
        assert "RuntimeComponentView.from_coupler_state" not in source
        assert "cpl.runtime_component_view(final_state," in source


def test_examples_import_concrete_components_directly() -> None:
    for path in Path("setups").glob("run_*.py"):
        source = path.read_text(encoding="utf-8")
        assert "from vercor.components import" not in source


def test_runtime_field_store_is_immutable_pytree() -> None:
    store = RuntimeFieldStore.from_mapping(
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

    assert isinstance(restored, RuntimeFieldStore)
    assert restored.field_names == ("temperature",)
    assert_allclose_compact(restored.get("temperature"), updated.get("temperature"))


def test_runtime_field_store_supports_jit_updates_and_mapping_roundtrip() -> None:
    store = RuntimeFieldStore.from_mapping(
        {
            "a": jnp.asarray([1.0, 2.0]),
            "b": jnp.asarray([3.0, 4.0]),
        }
    )

    def update(value: RuntimeFieldStore) -> RuntimeFieldStore:
        return value.set("a", value.get("a") * 2.0).set("b", value.get("b") + 1.0)

    updated = jax.jit(update)(store)

    assert updated.field_names == ("a", "b")
    assert_allclose_compact(updated.get("a"), np.asarray([2.0, 4.0]))
    assert_allclose_compact(updated.get("b"), np.asarray([4.0, 5.0]))


def test_runtime_field_store_uses_index_cache_for_bulk_set_and_pytree_restore() -> None:
    store = RuntimeFieldStore.from_mapping(
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
    store = RuntimeFieldStore.from_mapping(
        {"temperature": jnp.zeros((2, 2), dtype=jnp.float32)}
    )

    updated = store.set(
        "temperature",
        jnp.ones((2, 2), dtype=jnp.float64),
    )

    assert updated.get("temperature").dtype == jnp.float32
    assert_allclose_compact(updated.get("temperature"), np.ones((2, 2)))


def test_runtime_field_store_exposes_mapping_membership_and_fallback_helpers() -> None:
    store = RuntimeFieldStore.from_mapping(
        {
            "temperature": jnp.full((2, 2), 280.0, dtype=jnp.float32),
            "humidity": jnp.full((2, 2), 0.5, dtype=jnp.float32),
        }
    )

    fields = store.to_mapping()

    assert "temperature" in store
    assert "missing" not in store
    assert tuple(fields) == ("temperature", "humidity")
    assert_allclose_compact(fields["temperature"], np.full((2, 2), 280.0))
    assert_allclose_compact(
        store.get_or("temperature", jnp.zeros((2, 2))),
        np.full((2, 2), 280.0),
    )
    assert_allclose_compact(
        store.get_or("missing", jnp.full((2, 2), 3.0)),
        np.full((2, 2), 3.0),
    )
    assert_allclose_compact(
        store.get_or_zeros_like("missing", "temperature"),
        np.zeros((2, 2)),
    )
    assert_allclose_compact(
        store.get_or_zeros_like("missing", jnp.ones((2, 2))),
        np.zeros((2, 2)),
    )


def test_runtime_field_store_replace_helpers_preserve_dtype_and_reject_missing() -> (
    None
):
    store = RuntimeFieldStore.from_mapping(
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


def test_runtime_field_store_new_helpers_are_jit_compatible() -> None:
    store = RuntimeFieldStore.from_mapping(
        {
            "temperature": jnp.ones((2, 2), dtype=jnp.float32),
            "humidity": jnp.ones((2, 2), dtype=jnp.float32),
        }
    )

    def update(value: RuntimeFieldStore) -> RuntimeFieldStore:
        return value.replace_many(
            {
                "temperature": value.get("temperature") + 2.0,
                "humidity": value.get_or_zeros_like("missing", "temperature") + 0.25,
            }
        )

    updated = jax.jit(update)(store)

    assert_allclose_compact(updated.get("temperature"), np.full((2, 2), 3.0))
    assert_allclose_compact(updated.get("humidity"), np.full((2, 2), 0.25))


def test_runtime_component_and_coupler_state_are_pytrees() -> None:
    component = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        incoming=RuntimeFieldStore.from_mapping(
            {"sea_surface_temperature": jnp.zeros((2, 2))}
        ),
        outgoing=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
    )
    assert not hasattr(component, "name")
    assert not hasattr(component, "fields_to_import")
    assert not hasattr(component, "fields_to_export")
    state = RuntimeCouplerState(
        component_names=("ATM",),
        components=(component,),
        fractional_masks=RuntimeFieldStore.from_mapping(
            {"OCN|ATM|bilinear": jnp.ones((2, 2))}
        ),
        binary_masks=RuntimeFieldStore.empty(),
    )

    def update(value: RuntimeCouplerState) -> RuntimeCouplerState:
        atm = value.get_component_state("ATM")
        atm = atm.with_data(
            atm.data.set("temperature", atm.data.get("temperature") + 2.0)
        )
        return value.set_component_state("ATM", atm)

    updated = jax.jit(update)(state)

    assert state.component_indices == {"ATM": 0}
    assert updated.component_indices == {"ATM": 0}
    assert updated.component_names == ("ATM",)
    assert_allclose_compact(
        updated.get_component_state("ATM").data.get("temperature"),
        np.full((2, 2), 3.0),
    )


def test_runtime_coupler_state_restores_component_index_cache_after_pytree_roundtrip() -> (
    None
):
    component = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((2, 2))}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
    )
    state = RuntimeCouplerState(
        component_names=("ATM", "OCN"),
        components=(component, component),
        fractional_masks=RuntimeFieldStore.empty(),
        binary_masks=RuntimeFieldStore.empty(),
    )
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored.component_indices == {"ATM": 0, "OCN": 1}
    assert restored.get_component_state("OCN") is restored.components[1]


def test_runtime_component_state_preserves_optional_payload_under_jit() -> None:
    payload = JAXGCMRuntimePayload(
        jcm_state={"metadata": jnp.asarray(1.0)},
        forcing={"surface_temperature": jnp.asarray([[2.0, 3.0]])},
    )
    component = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": jnp.ones((1, 2))}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
        runtime_payload=payload,
    )

    def update(value: RuntimeComponentState) -> RuntimeComponentState:
        runtime_payload = value.runtime_payload
        assert isinstance(runtime_payload, JAXGCMRuntimePayload)
        return value.with_runtime_payload(
            JAXGCMRuntimePayload(
                jcm_state={"metadata": runtime_payload.jcm_state["metadata"] + 1.0},
                forcing=runtime_payload.forcing,
            )
        )

    updated = jax.jit(update)(component)

    assert isinstance(updated.runtime_payload, JAXGCMRuntimePayload)
    assert_allclose_compact(
        updated.runtime_payload.jcm_state["metadata"],
        np.asarray(2.0),
    )


def test_runtime_send_applies_monthly_interpolation_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(VercorSettings(apply_time_interpolation=True))
    contract = RuntimeComponentContract(exports=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [0.75], [0.25], [0]),
    )
    forcing = jnp.zeros((12, 2, 3), dtype=jnp.float64)
    forcing = forcing.at[0].set(4.0)
    forcing = forcing.at[1].set(8.0)

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.outgoing.get("temperature"))

    sent_state = jax.jit(
        lambda field: send_runtime_fields(
            component,
            RuntimeComponentState(
                data=RuntimeFieldStore.from_mapping({"temperature": field}),
                incoming=RuntimeFieldStore.empty(),
                outgoing=RuntimeFieldStore.empty(),
            ),
            step_info,
            contract=contract,
        )
    )(forcing)
    out = sent_state.outgoing.get("temperature")
    gradient = jax.grad(send_loss)(forcing)

    assert out.shape == (2, 3)
    assert_allclose_compact(out, np.full((2, 3), 5.0))
    assert_allclose_compact(gradient[0], np.full((2, 3), 0.75))
    assert_allclose_compact(gradient[1], np.full((2, 3), 0.25))
    assert_allclose_compact(gradient[2:], np.zeros((10, 2, 3)))


def test_runtime_send_applies_daily_time_slice_under_jit_and_grad() -> None:
    component = _RuntimeSendComponent(VercorSettings(get_field_time_slice=True))
    contract = RuntimeComponentContract(exports=("temperature",))
    step_info = jax.tree_util.tree_map(
        lambda value: value[0],
        RuntimeStepInfo.from_sequences([0], [1], [1.0], [0.0], [2]),
    )
    forcing = jnp.arange(5 * 2 * 2, dtype=jnp.float64).reshape((5, 2, 2))

    def send_loss(field: jax.Array) -> jax.Array:
        state = RuntimeComponentState(
            data=RuntimeFieldStore.from_mapping({"temperature": field}),
            incoming=RuntimeFieldStore.empty(),
            outgoing=RuntimeFieldStore.empty(),
        )
        sent = send_runtime_fields(component, state, step_info, contract=contract)
        return jnp.sum(sent.outgoing.get("temperature"))

    state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping({"temperature": forcing}),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.empty(),
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

    assert_allclose_compact(
        sent_state.outgoing.get("temperature"), np.asarray(forcing[2])
    )
    assert_allclose_compact(gradient[2], np.ones((2, 2)))
    assert_allclose_compact(gradient[:2], np.zeros((2, 2, 2)))
    assert_allclose_compact(gradient[3:], np.zeros((2, 2, 2)))
