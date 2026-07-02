from __future__ import annotations

import importlib
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tests._architecture_support import package_import_cycles, source_for
from vercor.runtime.facade import PreparedRuntimeState, RuntimeFacadeInputs
from vercor.runtime.resources import CouplerRuntimeResources


def test_runtime_facade_inputs_bundle_owns_repeated_coupler_runtime_inputs() -> None:
    assert is_dataclass(RuntimeFacadeInputs)
    assert [field.name for field in fields(RuntimeFacadeInputs)] == [
        "components",
        "exchanges",
        "runtime_resources",
        "run_sequence",
        "clock",
        "settings",
    ]


def test_coupler_passes_facade_inputs_instead_of_parameter_clumps() -> None:
    coupler_source = Path("vercor/coupler.py").read_text(encoding="utf-8")

    assert "_runtime_facade.RuntimeFacadeInputs(" in coupler_source
    assert "components=self.components,\n            exchanges=self.exchanges," not in (
        coupler_source
    )


def test_runtime_preparation_module_owns_runtime_state_preparation() -> None:
    preparation_path = Path("vercor/runtime/preparation.py")
    assert preparation_path.exists()

    preparation_source = preparation_path.read_text(encoding="utf-8")
    assert "class PreparedRuntimeState" in preparation_source
    assert "def runtime_state_from_components(" in preparation_source
    assert "def validate_runtime_state(" in preparation_source
    assert "def create_runtime_state(" in preparation_source
    assert "def prepare_runtime_state(" in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source

    assert is_dataclass(PreparedRuntimeState)
    assert [field.name for field in fields(PreparedRuntimeState)] == [
        "runtime_state",
        "runtime_contracts",
    ]


@pytest.mark.fast_always
def test_component_topology_module_owns_component_lookup_helpers() -> None:
    component_topology_source = source_for("vercor/runtime/component_topology.py")
    topology_source = source_for("vercor/runtime/topology.py")
    surface_masks_source = source_for("vercor/runtime/surface_masks.py")
    initialization_source = source_for("vercor/runtime/initialization.py")

    assert "VALID_TOPOLOGY_COMPONENT_NAMES" in component_topology_source
    assert "def validate_component_topology_names(" in component_topology_source
    assert "def get_component(" in component_topology_source
    assert "def validate_component_topology_names(" not in topology_source
    assert "def get_component(" not in topology_source
    assert "from vercor.runtime.component_topology import" not in topology_source
    assert "from vercor.runtime.component_topology import" in surface_masks_source
    assert "from vercor.runtime.component_topology import" in initialization_source


@pytest.mark.fast_always
def test_runtime_topology_state_groups_mutable_maps() -> None:
    topology_state_module = importlib.import_module("vercor.runtime.topology_state")
    topology_state_source = source_for("vercor/runtime/topology_state.py")
    topology_source = source_for("vercor/runtime/topology.py")
    resources_source = source_for("vercor/runtime/resources.py")

    assert hasattr(topology_state_module, "RuntimeTopologyMaps")
    RuntimeTopologyMaps = topology_state_module.RuntimeTopologyMaps
    assert is_dataclass(RuntimeTopologyMaps)
    assert getattr(RuntimeTopologyMaps, "__dataclass_params__").frozen is False
    assert hasattr(RuntimeTopologyMaps, "__slots__")
    assert [field.name for field in fields(RuntimeTopologyMaps)] == [
        "regridders",
        "binary_masks",
        "fractional_masks",
    ]
    assert "class RuntimeTopologyMaps" in topology_state_source
    assert "topology_maps: RuntimeTopologyMaps" in topology_state_source
    assert "class RuntimeTopologyMaps" not in topology_source
    assert "topology.regridders" not in resources_source
    assert "def replace_topology(" not in resources_source


@pytest.mark.fast_always
def test_runtime_topology_policy_boundaries_are_focused() -> None:
    topology_state_module = importlib.import_module("vercor.runtime.topology_state")
    topology_state_source = source_for("vercor/runtime/topology_state.py")
    exchange_topology_source = source_for("vercor/runtime/exchange_topology.py")
    surface_masks_source = source_for("vercor/runtime/surface_masks.py")
    topology_source = source_for("vercor/runtime/topology.py")
    resources_source = source_for("vercor/runtime/resources.py")

    assert hasattr(topology_state_module, "SurfaceExchangeMasks")
    SurfaceExchangeMasks = topology_state_module.SurfaceExchangeMasks
    ExchangeTopologyState = topology_state_module.ExchangeTopologyState
    assert is_dataclass(SurfaceExchangeMasks)
    assert is_dataclass(ExchangeTopologyState)
    assert [field.name for field in fields(SurfaceExchangeMasks)] == [
        "ocn_fmask_on_atm_grid",
        "lnd_fmask_on_atm_grid",
        "lnd_bmask_on_atm_grid",
    ]
    assert [field.name for field in fields(ExchangeTopologyState)] == [
        "topology_maps",
        "surface_masks",
    ]

    assert "RuntimeRegridder =" not in topology_state_source
    assert "BilinearRectilinearRegridder" not in topology_state_source
    assert "ConservativeRectilinearRegridder" not in topology_state_source
    assert "def build_exchange_topology_maps(" in exchange_topology_source
    assert "def create_surface_exchange_masks(" in surface_masks_source
    assert "def validate_land_mask_consistency(" in surface_masks_source
    assert "def apply_surface_exchange_masks(" in surface_masks_source

    for marker in (
        "compute_ocn_lnd_masks_on_atm_grid",
        "check_remap_conservation",
        "check_total_lnd_ocn_mask_sum",
        "ConservativeRectilinearRegridder",
        "jax_ones",
        "def create_exchange_masks(",
        "def validate_land_mask_consistency(",
        "def initialize_regridders_and_masks(",
        "def patch_exchange_masks(",
    ):
        assert marker not in topology_source

    assert "import vercor.runtime.exchange_topology as" in topology_source
    assert "import vercor.runtime.surface_masks as" in topology_source
    assert "from vercor.runtime.topology_state import" in topology_source
    assert "from vercor.runtime.topology_state import" in resources_source


@pytest.mark.fast_always
def test_runtime_resources_expose_simple_public_resource_fields() -> None:
    resources = CouplerRuntimeResources()
    resources_source = source_for("vercor/runtime/resources.py")
    facade_source = source_for("vercor/runtime/facade.py")
    preparation_source = source_for("vercor/runtime/preparation.py")
    run_context_source = source_for("vercor/runtime/run_context.py")

    for old_raw_name in (
        "compiled_runtime_cache",
        "runtime_cache_mapping",
        "interrupts",
    ):
        assert not hasattr(resources, old_raw_name), old_raw_name

    for public_field in (
        "topology_maps",
        "runtime_contracts",
        "interrupt_controller",
    ):
        assert hasattr(resources, public_field), public_field

    assert "slots=True" in resources_source
    assert "_topology_maps:" not in resources_source
    assert "_runtime_contracts:" not in resources_source
    assert "_runtime_cache:" not in resources_source
    assert "_compiled_runtime_cache:" not in resources_source
    assert "_interrupt_controller:" not in resources_source
    assert "def replace_contracts(" not in resources_source
    assert "def replace_topology_maps(" not in resources_source
    assert "runtime_cache_mapping(" not in resources_source
    assert "MutableMapping" not in run_context_source
    assert "runtime_cache" not in run_context_source
    assert "CompiledRuntimeCache" not in run_context_source
    for source in (facade_source, preparation_source):
        for raw_access in (
            "runtime_resources.regridders",
            "runtime_resources.binary_masks",
            "runtime_resources.fractional_masks",
            "runtime_resources.contracts",
            "runtime_resources.compiled_runtime_cache",
            "runtime_resources.runtime_cache_mapping",
            "runtime_resources.interrupts",
        ):
            assert raw_access not in source


@pytest.mark.fast_always
def test_runtime_compilation_cache_is_removed() -> None:
    compilation_path = Path("vercor/runtime/compilation.py")
    cache_path = Path("vercor/runtime/cache.py")
    assert not compilation_path.exists()
    assert not cache_path.exists()

    run_context_source = source_for("vercor/runtime/run_context.py")
    resources_source = source_for("vercor/runtime/resources.py")

    assert "from vercor.runtime.compilation import" not in run_context_source
    assert "from vercor.runtime.compilation import" not in resources_source
    assert "from vercor.runtime.cache import" not in run_context_source
    assert "from vercor.runtime.cache import" not in resources_source
    assert "CompiledRuntime" not in run_context_source
    assert "CompiledRuntimeCache" not in resources_source
    assert "compiled_runtime_cache_key(" not in run_context_source


@pytest.mark.fast_always
def test_runtime_state_validation_module_owns_runtime_topology_validation() -> None:
    state_validation_path = Path("vercor/runtime/state_validation.py")
    coupler_state_source = source_for("vercor/runtime/coupler_state.py")
    preparation_source = source_for("vercor/runtime/preparation.py")
    facade_source = source_for("vercor/runtime/facade.py")
    coupler_source = source_for("vercor/coupler.py")

    assert state_validation_path.exists()
    state_validation_source = state_validation_path.read_text(encoding="utf-8")
    assert "def validate_runtime_state(" in state_validation_source
    assert "def validate_runtime_state(" not in coupler_state_source
    assert "from vercor.runtime.state_validation import" in preparation_source
    assert "from vercor.runtime.state_validation import" not in facade_source
    assert "from vercor.runtime.state_validation import" not in coupler_source


def test_runtime_facade_reexports_preparation_without_owning_it() -> None:
    facade_source = source_for("vercor/runtime/facade.py")
    preparation_source = source_for("vercor/runtime/preparation.py")

    assert "from vercor.runtime.preparation import" in facade_source
    assert "Protocol" not in preparation_source
    assert "RuntimePreparationInputs" not in preparation_source
    assert "if TYPE_CHECKING:" in preparation_source
    assert "from vercor.runtime.facade import RuntimeFacadeInputs" in preparation_source
    assert "class PreparedRuntimeState" not in facade_source
    assert "def runtime_state_from_components(" not in facade_source
    assert "def validate_runtime_state(" not in facade_source
    assert "def create_runtime_state(" not in facade_source
    assert "def prepare_runtime_state(" not in facade_source


@pytest.mark.fast_always
def test_runtime_runner_splits_path_selection_helpers() -> None:
    runner_source = source_for("vercor/runtime/runner.py")
    run_coupler_body = runner_source.split("def run_coupler_runtime(", 1)[1].split(
        "\ndef _run_compiled_scanned_runtime(",
        1,
    )[0]

    assert "def _run_compiled_scanned_runtime(" in runner_source
    assert "def _raise_if_donating_host_runtime(" not in runner_source
    assert "compiled_runtime_cache_key(" not in run_coupler_body
    assert "def compiled_runtime_cache_key(" not in runner_source
    assert "compiled_scanned_runtime," not in runner_source
    assert "return compiled_scanned_runtime(" not in runner_source
    assert "get_or_compile_for_context(" not in runner_source
    assert "context.compiled_runtime_cache_key(" not in runner_source
    assert "get_or_compile(" not in runner_source
    assert "donate_state" not in runner_source
    assert "raise CouplerError(" not in run_coupler_body


def test_runtime_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/runtime", "vercor.runtime") == []
