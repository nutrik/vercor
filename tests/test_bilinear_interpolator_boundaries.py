from __future__ import annotations

import importlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from tests._architecture_support import package_import_cycles, source_for
from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.dtypes import numpy_index_dtype


@pytest.mark.fast_always
def test_bilinear_public_module_delegates_private_implementation_owners() -> None:
    source = source_for("vercor/interpolators/bilinear_rectilinear.py")

    assert "import vercor.interpolators._bilinear_geometry as" in source
    assert "import vercor.interpolators._bilinear_weights as" in source
    assert "import vercor.interpolators._bilinear_extrapolation as" in source
    assert "jnp.searchsorted" not in source
    assert "lax.top_k" not in source
    assert "def _great_circle_distance_rad(" not in source
    assert "def _unit_east_north(" not in source


@pytest.mark.fast_always
def test_private_bilinear_helpers_stay_inside_interpolators_package() -> None:
    for path in Path("vercor").rglob("*.py"):
        if path.parts[:2] == ("vercor", "interpolators"):
            continue

        source = path.read_text(encoding="utf-8")
        assert "vercor.interpolators._bilinear_" not in source, path


@pytest.mark.fast_always
def test_interpolators_package_has_no_top_level_import_cycles() -> None:
    assert package_import_cycles("vercor/interpolators", "vercor.interpolators") == []


@pytest.mark.fast_always
def test_private_bilinear_weight_helper_reproduces_periodic_dateline_indices() -> None:
    helper_path = Path("vercor/interpolators/_bilinear_weights.py")
    assert helper_path.exists(), "private bilinear weight helper must exist"

    weights_module = importlib.import_module("vercor.interpolators._bilinear_weights")
    lon_tgt_deg, lat_tgt_deg = jnp.meshgrid(
        jnp.asarray([359.0]),
        jnp.asarray([5.0]),
    )

    weights = weights_module.compute_bilinear_cell_weights(
        lon_src_deg=jnp.asarray([0.0, 120.0, 240.0]),
        lat_src_deg=jnp.asarray([0.0, 10.0]),
        lon_tgt_deg=lon_tgt_deg,
        lat_tgt_deg=lat_tgt_deg,
        periodic=True,
        lat_ascending=True,
    )

    assert_array_equal_compact(weights.i0, np.array([[2]], dtype=numpy_index_dtype()))
    assert_array_equal_compact(weights.i1, np.array([[0]], dtype=numpy_index_dtype()))
    assert_array_equal_compact(weights.j0, np.array([[0]], dtype=numpy_index_dtype()))
    assert_array_equal_compact(weights.j1, np.array([[1]], dtype=numpy_index_dtype()))
    assert_allclose_compact(weights.fx, np.array([[119.0 / 120.0]]))
    assert_allclose_compact(weights.fy, np.array([[0.5]]))
    assert_allclose_compact(
        weights.w00 + weights.w10 + weights.w01 + weights.w11,
        np.array([[1.0]]),
    )


@pytest.mark.fast_always
def test_private_bilinear_extrapolation_helper_fills_when_no_sources_are_valid() -> (
    None
):
    helper_path = Path("vercor/interpolators/_bilinear_extrapolation.py")
    assert helper_path.exists(), "private bilinear extrapolation helper must exist"

    extrapolation_module = importlib.import_module(
        "vercor.interpolators._bilinear_extrapolation"
    )
    out = extrapolation_module.extrapolate_scalar_field(
        source_values=jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        valid_source_mask=jnp.zeros((2, 2), dtype=bool),
        target_shape=(1, 2),
        target_lon_flat=jnp.deg2rad(jnp.asarray([0.0, 1.0])),
        target_lat_flat=jnp.deg2rad(jnp.asarray([0.0, 0.0])),
        source_lon_flat=jnp.deg2rad(jnp.asarray([0.0, 1.0, 0.0, 1.0])),
        source_lat_flat=jnp.deg2rad(jnp.asarray([0.0, 0.0, 1.0, 1.0])),
        mode="nearest",
        idw_k=2,
        idw_eps=1e-12,
        fill_value=-7.0,
    )

    assert_array_equal_compact(out, np.array([[-7.0, -7.0]]))
