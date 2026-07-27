"""Contracts for the Sphinx documentation configuration."""

from __future__ import annotations

import runpy
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONF_PATH = PROJECT_ROOT / "docs" / "conf.py"


def _run_copied_conf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pyproject_text: str,
) -> dict[str, Any]:
    """Execute the real configuration from a synthetic project location."""
    docs_path = tmp_path / "docs"
    docs_path.mkdir()
    copied_conf = docs_path / "conf.py"
    shutil.copy2(CONF_PATH, copied_conf)
    (tmp_path / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")

    unrelated_working_directory = tmp_path / "working-directory"
    unrelated_working_directory.mkdir()
    monkeypatch.chdir(unrelated_working_directory)

    original_sys_path = sys.path.copy()
    try:
        return runpy.run_path(str(copied_conf))
    finally:
        sys.path[:] = original_sys_path


def test_conf_reads_version_relative_to_its_own_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use checkout metadata even when Sphinx starts elsewhere."""
    config = _run_copied_conf(
        tmp_path,
        monkeypatch,
        '[project]\nname = "synthetic-vercor"\nversion = "9.8.7"\n',
    )

    assert config["version"] == "9.8.7"
    assert config["release"] == "9.8.7"


def test_conf_does_not_mock_installed_core_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Let autodoc import the real numerical stack required by JAX."""
    config = _run_copied_conf(
        tmp_path,
        monkeypatch,
        '[project]\nname = "synthetic-vercor"\nversion = "9.8.7"\n',
    )

    mocked_imports = set(config.get("autodoc_mock_imports", ()))
    assert mocked_imports.isdisjoint({"numpy", "scipy", "h5netcdf", "jax"})


def test_conf_rejects_missing_project_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail instead of substituting a branch name for missing metadata."""
    with pytest.raises(KeyError, match="version"):
        _run_copied_conf(
            tmp_path,
            monkeypatch,
            '[project]\nname = "synthetic-vercor"\n',
        )
