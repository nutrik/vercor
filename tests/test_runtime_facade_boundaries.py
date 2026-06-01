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
    initialization_source = source_for("vercor/runtime/initialization.py")

    assert "VALID_TOPOLOGY_COMPONENT_NAMES" in component_topology_source
    assert "def validate_component_topology_names(" in component_topology_source
    assert "def get_component(" in component_topology_source
    assert "def validate_component_topology_names(" not in topology_source
    assert "def get_component(" not in topology_source
    assert "from vercor.runtime.component_topology import" in topology_source
    assert "from vercor.runtime.component_topology import" in initialization_source


@pytest.mark.fast_always
def test_runtime_topology_state_groups_mutable_maps() -> None:
    topology_module = importlib.import_module("vercor.runtime.topology")
    topology_source = source_for("vercor/runtime/topology.py")
    resources_source = source_for("vercor/runtime/resources.py")

    assert hasattr(topology_module, "RuntimeTopologyMaps")
    RuntimeTopologyMaps = topology_module.RuntimeTopologyMaps
    assert is_dataclass(RuntimeTopologyMaps)
    assert [field.name for field in fields(RuntimeTopologyMaps)] == [
        "regridders",
        "binary_masks",
        "fractional_masks",
    ]
    assert "class RuntimeTopologyMaps" in topology_source
    assert "topology_maps: RuntimeTopologyMaps" in topology_source
    assert "topology.regridders" not in resources_source
    assert "topology.topology_maps" in resources_source


@pytest.mark.fast_always
def test_runtime_resources_hide_raw_resource_dictionaries() -> None:
    resources = CouplerRuntimeResources()
    resources_source = source_for("vercor/runtime/resources.py")
    facade_source = source_for("vercor/runtime/facade.py")
    preparation_source = source_for("vercor/runtime/preparation.py")

    for raw_name in (
        "regridders",
        "binary_masks",
        "fractional_masks",
        "contracts",
        "compiled_runtime_cache",
        "interrupts",
    ):
        assert not hasattr(resources, raw_name), raw_name

    assert "slots=True" in resources_source
    assert "_topology_maps:" in resources_source
    assert "_runtime_contracts:" in resources_source
    assert "_compiled_runtime_cache:" in resources_source
    assert "_interrupt_controller:" in resources_source
    for source in (facade_source, preparation_source):
        for raw_access in (
            "runtime_resources.regridders",
            "runtime_resources.binary_masks",
            "runtime_resources.fractional_masks",
            "runtime_resources.contracts",
            "runtime_resources.compiled_runtime_cache",
            "runtime_resources.interrupts",
        ):
            assert raw_access not in source


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
    assert "raise CouplerError(" not in run_coupler_body


def test_runtime_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/runtime", "vercor.runtime") == []
