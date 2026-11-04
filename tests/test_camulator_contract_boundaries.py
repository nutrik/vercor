from __future__ import annotations

from pathlib import Path

from vercor.setups._external import camulator_contracts


def test_camulator_runtime_field_names_have_lightweight_contract_owner() -> None:
    assert "air_temperature" in camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES
    assert "total_surface_temperature" in (
        camulator_contracts.CAMULATOR_RUNTIME_FIELD_NAMES
    )
    assert (
        camulator_contracts.camulator_runtime_field_defaults()["air_temperature"] == 0.0
    )


def test_camulator_factory_uses_contract_owner_for_runtime_fields() -> None:
    camulator_source = Path("vercor/setups/_external/camulator.py").read_text(
        encoding="utf-8"
    )

    assert "camulator_contracts" in camulator_source
    assert "_CAMULATOR_RUNTIME_FIELD_NAMES" not in camulator_source
