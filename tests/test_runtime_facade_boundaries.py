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
    assert [field.name for field in fields(RuntimeTopologyMaps)] == [
        "regridders",
        "binary_masks",
        "fractional_masks",
    ]
    assert "class RuntimeTopologyMaps" in topology_state_source
    assert "topology_maps: RuntimeTopologyMaps" in topology_state_source
    assert "class RuntimeTopologyMaps" not in topology_source
    assert "topology.regridders" not in resources_source
    assert "topology.topology_maps" in resources_source


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

    assert "RuntimeRegridder =" in topology_state_source
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
def test_runtime_resources_hide_raw_resource_dictionaries() -> None:
    resources = CouplerRuntimeResources()
    resources_source = source_for("vercor/runtime/resources.py")
    facade_source = source_for("vercor/runtime/facade.py")
    preparation_source = source_for("vercor/runtime/preparation.py")
    run_context_source = source_for("vercor/runtime/run_context.py")

    for raw_name in (
        "regridders",
        "binary_masks",
        "fractional_masks",
        "contracts",
        "compiled_runtime_cache",
        "runtime_cache_mapping",
        "interrupts",
    ):
        assert not hasattr(resources, raw_name), raw_name

    assert "slots=True" in resources_source
    assert "_topology_maps:" in resources_source
    assert "_runtime_contracts:" in resources_source
    assert "_runtime_cache:" in resources_source
    assert "_compiled_runtime_cache:" not in resources_source
    assert "_interrupt_controller:" in resources_source
    assert "runtime_cache_mapping(" not in resources_source
    assert "MutableMapping" not in run_context_source
    assert "runtime_cache: CompiledRuntimeCache" in run_context_source
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
def test_runtime_compilation_cache_has_narrow_context_boundary() -> None:
    compilation_path = Path("vercor/runtime/compilation.py")
    assert compilation_path.exists()

    compilation_source = source_for("vercor/runtime/compilation.py")
    cache_source = source_for("vercor/runtime/cache.py")
    run_context_source = source_for("vercor/runtime/run_context.py")
    resources_source = source_for("vercor/runtime/resources.py")

    assert "CompiledRuntime = Callable[" in compilation_source
    assert "RuntimeCompilationKey: TypeAlias" in compilation_source
    assert "from vercor.runtime.compilation import CompiledRuntime" in cache_source
    assert "from vercor.runtime.compilation import CompiledRuntime" in (
        run_context_source
    )
    assert "from vercor.runtime.compilation import CompiledRuntime" in (
        resources_source
    )
    assert "from vercor.runtime.run_context import CompiledRuntime" not in (
        resources_source
    )
    assert "RuntimeRunContext" not in cache_source
    assert "def get_or_compile_for_context(" not in cache_source
    assert "compiled_runtime_cache_key(" not in cache_source
    assert "def compiled_runtime_cache_key(" in run_context_source


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

    assert "from vercor.runtime.preparation import" in facade_source
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
    assert "def _raise_if_donating_host_runtime(" in runner_source
    assert "compiled_runtime_cache_key(" not in run_coupler_body
    assert "def compiled_runtime_cache_key(" not in runner_source
    assert "compiled_scanned_runtime," not in runner_source
    assert "return compiled_scanned_runtime(" not in runner_source
    assert "get_or_compile_for_context(" not in runner_source
    assert "context.compiled_runtime_cache_key(" in runner_source
    assert "get_or_compile(" in runner_source
    assert "raise CouplerError(" not in run_coupler_body


def test_runtime_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/runtime", "vercor.runtime") == []
