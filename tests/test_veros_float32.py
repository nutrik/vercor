"""Run real native float32 boundaries before native process settings lock."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast_always


def _run_float32_probe(
    execution: str, tmp_path: Path, *, allocation_only: bool = False
) -> None:
    """Select native precision before any import in a fresh interpreter."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests._veros_float32_probe",
            execution,
            "allocation" if allocation_only else "native",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "VEROS_FLOAT_TYPE": "float32",
            "JAX_ENABLE_X64": "true" if allocation_only else "false",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    output = result.stdout + result.stderr
    mode = "allocation" if allocation_only else "native"
    log = tmp_path / f"float32-{execution}-{mode}.log"
    log.write_text(output, encoding="utf-8")
    tail = "\n".join(output.splitlines()[-20:])
    assert result.returncode == 0, f"{tail}\nFull native log: {log}"


@pytest.mark.veros_autodiff
def test_exact_fork_float32_allocation_with_x64_capability(tmp_path: Path) -> None:
    """Catch fixture-created float64 arrays even when x64 is available."""
    _run_float32_probe("jax", tmp_path, allocation_only=True)


@pytest.mark.veros_autodiff
def test_exact_fork_float32_native_boundary(tmp_path: Path) -> None:
    """Check native means, continuation and source AD without f64 promotion."""
    _run_float32_probe("jax", tmp_path)


def test_stock_float32_native_boundary(tmp_path: Path) -> None:
    """Check actual native source budgets and mapped means in stock float32."""
    _run_float32_probe("host", tmp_path)
