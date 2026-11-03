"""Bundled native output paths use ordinary providers and core coordination."""

from __future__ import annotations

from pathlib import Path


def _source(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_bundled_factories_install_native_output_providers() -> None:
    jax_gcm = _source("vercor/setups/_external/jax_gcm.py")
    veros = _source("vercor/setups/_external/veros_gcm.py")
    camulator = _source("vercor/setups/_external/camulator.py")

    assert "jax_gcm_output_provider(state)" in jax_gcm
    assert "veros_output_provider()" in veros
    assert "camulator_output_provider(resources)" in camulator
    assert "variables=(" not in veros.split("veros_output_provider()", 1)[0][-200:]


def test_core_output_session_owns_native_output_boundaries() -> None:
    session = _source("vercor/output/_session.py")

    assert "class _OutputSession" in session
    assert "def write_output_boundary(" in session
    assert "output_path = plan.target.directory / filename" in session


def test_native_output_modules_return_output_frames() -> None:
    jax_output = _source("vercor/setups/_external/jax_gcm_output.py")
    veros_output = _source("vercor/setups/_external/veros_output.py")
    camulator_output = _source("vercor/setups/_external/camulator_output.py")

    for source in (jax_output, veros_output, camulator_output):
        assert "def sample(self, context: OutputContext) -> OutputFrame:" in source
        assert "OutputFrame(" in source


def test_camulator_native_period_output_uses_run_level_paths() -> None:
    output_source = _source("vercor/setups/_external/camulator_output.py")
    session_source = _source("vercor/output/_session.py")

    assert "def camulator_output_provider(" in output_source
    assert "def sample(self, context: OutputContext) -> OutputFrame:" in output_source
    assert "payload = context.payload" in output_source
    assert "._output_prediction" not in output_source
    assert "output_path = plan.target.directory / filename" in session_source
