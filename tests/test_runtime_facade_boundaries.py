from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tests._architecture_support import package_import_cycles, source_for
from vercor.runtime.facade import PreparedRuntimeState, RuntimeFacadeInputs


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
