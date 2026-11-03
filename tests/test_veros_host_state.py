"""Validate isolated host-JAX native state ownership and executable reuse."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.fast_always


def test_stock_jax_host_native_progression_preserves_initial_state(
    tmp_path: Path,
) -> None:
    """Catch mutable native auxiliary owners leaking across host JAX payloads."""
    result = subprocess.run(
        [sys.executable, "-m", "tests._veros_backend_probe"],
        env={**os.environ, "JAX_ENABLE_X64": "true"},
        capture_output=True,
        text=True,
        check=False,
    )
    log = tmp_path / "stock-jax-host-native.log"
    log.write_text(result.stdout + result.stderr)
    assert (
        result.returncode == 0
    ), f"{(result.stdout + result.stderr)[-2200:]}\nFull log: {log}"


def test_stock_jax_float32_preserves_native_sources_and_continuation(
    tmp_path: Path,
) -> None:
    """Run the established native float32 oracle under stock host-JAX kernels."""
    code = """
import sys
from functools import partial
from vercor.setups._external import veros_runtime_settings as settings
settings.configure_veros_runtime = partial(settings.configure_veros_runtime, backend='jax')
settings.configure_veros_runtime('host')
from vercor.setups._external.veros_state import copy_state
from tests import _veros_surface_support as support
original = support.surface_model
def owned_model(**kwargs):
    model = original(**kwargs)
    model.state = copy_state(model.state, execution='host')
    return model
support.surface_model = owned_model
sys.argv = ['probe', 'host', 'native']
from tests._veros_float32_probe import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "VEROS_FLOAT_TYPE": "float32", "JAX_ENABLE_X64": "false"},
        capture_output=True,
        text=True,
        check=False,
    )
    log = tmp_path / "stock-jax-float32.log"
    log.write_text(result.stdout + result.stderr)
    assert (
        result.returncode == 0
    ), f"{(result.stdout + result.stderr)[-2200:]}\nFull log: {log}"
