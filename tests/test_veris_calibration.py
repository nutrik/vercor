"""Recover native h0 through the reusable public-state calibration workflow."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import importlib
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.setups import replace_veris_parameters

pytestmark = pytest.mark.fast_always


@pytest.fixture(scope="module")
def example() -> ModuleType:
    return importlib.import_module("vercor.setups.gallery.run_veris_calibration")


@pytest.fixture(scope="module")
def training(example: ModuleType) -> tuple[Any, Any]:
    coupler = example._make_coupler(loglevel="error", float_type="float64", steps=4)
    return coupler, coupler.initial_state()


def _observed(coupler: Any, initial: Any, h0: Any) -> jax.Array:
    return jnp.asarray(
        coupler.run(replace_veris_parameters(initial, {"h0": h0}))
        .component("CICE")
        .field("ice_fraction")
    )


@pytest.mark.parametrize("truth", [0.35, 0.43, 0.65])
@pytest.mark.parametrize("start", [0.3, 0.7])
def test_synthetic_h0_recovery_and_holdout(
    example: ModuleType, training: tuple[Any, Any], truth: float, start: float
) -> None:
    coupler, initial = training
    observed = _observed(coupler, initial, truth)
    original = _observed(coupler, initial, 0.5)
    result = example.calibrate_lead_closing_thickness(
        coupler, initial, observed, initial_h0=start
    )
    assert result.converged and abs(float(result.h0) - truth) < 1e-5
    assert np.isfinite(float(result.loss)) and abs(float(result.gradient)) < 1e-7
    np.testing.assert_array_equal(_observed(coupler, initial, 0.5), original)
    holdout = example._make_coupler(
        loglevel="error", float_type="float64", steps=6, holdout=True
    )
    held_initial = holdout.initial_state()
    fitted = _observed(holdout, held_initial, result.h0)
    target = _observed(holdout, held_initial, truth)
    assert float(jnp.max(jnp.abs(fitted - target))) < 1e-7
    assert len(jax.tree_util.tree_leaves(result)) == 6
    with pytest.raises(FrozenInstanceError):
        result.h0 = 0.0


@pytest.mark.parametrize("kind", ["shape", "negative", "nonfinite", "zero", "dry_only"])
def test_calibration_rejects_invalid_weights(
    example: ModuleType, training: tuple[Any, Any], kind: str
) -> None:
    coupler, initial = training
    weights = np.ones((2, 3))
    if kind == "shape":
        weights = np.ones((1, 1))
    elif kind == "negative":
        weights[1, 1] = -1.0
    elif kind == "nonfinite":
        weights[0, 0] = np.nan
    elif kind == "zero":
        weights[:] = 0.0
    else:
        weights[:] = 0.0
        weights[0, 0] = 1.0
    with pytest.raises(ValueError, match="weight"):
        example.calibrate_lead_closing_thickness(
            coupler, initial, _observed(coupler, initial, 0.43), weights=weights
        )


@pytest.mark.parametrize(
    "bounds", [(0.0, 0.8), (0.8, 0.25), (0.4, 0.4), (0.25, np.inf), (0.25,)]
)
def test_calibration_rejects_invalid_bounds(
    example: ModuleType, training: tuple[Any, Any], bounds: Any
) -> None:
    coupler, initial = training
    with pytest.raises(ValueError, match="bounds"):
        example.calibrate_lead_closing_thickness(
            coupler, initial, _observed(coupler, initial, 0.43), bounds=bounds
        )


@pytest.mark.parametrize(
    "observed", [np.ones((1, 1)), np.full((2, 3), np.nan), np.full((2, 3), 1.1)]
)
def test_calibration_rejects_invalid_observations(
    example: ModuleType, training: tuple[Any, Any], observed: np.ndarray
) -> None:
    coupler, initial = training
    with pytest.raises(ValueError, match="observ"):
        example.calibrate_lead_closing_thickness(coupler, initial, observed)


def test_calibration_rejects_insensitive_observation_window(
    example: ModuleType,
) -> None:
    coupler = example._make_coupler(
        loglevel="error", float_type="float64", steps=2, initial_area=1.0, calm=True
    )
    initial = coupler.initial_state()
    observed = _observed(coupler, initial, 0.43)
    with pytest.raises(ValueError, match="insensitive"):
        example.calibrate_lead_closing_thickness(coupler, initial, observed)


def test_float32_recovery_smoke(example: ModuleType) -> None:
    coupler = example._make_coupler(loglevel="error", float_type="float32", steps=4)
    initial = coupler.initial_state()
    observed = _observed(coupler, initial, jnp.asarray(0.43, dtype=jnp.float32))
    result = example.calibrate_lead_closing_thickness(
        coupler, initial, observed, initial_h0=0.3
    )
    assert result.h0.dtype == jnp.float32
    assert abs(float(result.h0) - 0.43) < 2e-3


def test_calibration_includes_initially_ice_free_wet_cells(example: ModuleType) -> None:
    coupler = example._make_coupler(
        loglevel="error", float_type="float64", steps=2, initial_area=0.0
    )
    initial = coupler.initial_state()
    observed = _observed(coupler, initial, 0.43)
    result = example.calibrate_lead_closing_thickness(
        coupler, initial, observed, initial_h0=0.7
    )
    assert result.converged and abs(float(result.h0) - 0.43) < 1e-5


def test_fresh_process_forcing_honors_float64_policy() -> None:
    import subprocess
    import sys

    code = """
import jax, numpy as np
jax.config.update('jax_enable_x64', False)
from vercor.dtypes import DTypePolicy
from vercor.setups.gallery.run_veris_calibration import _cold_fields
fields = _cold_fields((2, 3), policy=DTypePolicy(enable_x64=True))
assert all(value.dtype == np.float64 for value in fields.values()), 'ERROR forcing precision was truncated'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-1500:]
