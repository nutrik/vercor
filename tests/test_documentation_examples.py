"""Executable contracts for code published in the user documentation."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "docs" / "_examples"
EXAMPLE_NAMES = (
    "quickstart.py",
    "data_component.py",
    "host_component.py",
    "jax_component.py",
)


@pytest.mark.fast_always
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_documentation_example_executes(name: str) -> None:
    """Execute each complete documentation program."""
    runpy.run_path(str(EXAMPLES_ROOT / name), run_name="__main__")


@pytest.mark.fast_always
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_documentation_example_uses_only_public_vercor_imports(name: str) -> None:
    """Reject private VerCOR imports in published examples."""
    source = (EXAMPLES_ROOT / name).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=name)
    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    imported_modules.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(not module.startswith("vercor._") for module in imported_modules)
