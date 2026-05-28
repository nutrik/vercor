from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

from vercor.runtime.facade import RuntimeFacadeInputs


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
