"""Wrapped native longitude edges must retain every eastward ocean strip."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor import RectilinearGrid
from vercor._interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.grid_masks import _surface_fractions_from_ocean
from vercor.regridding import conservative

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize(
    "target_edges, expected",
    [
        ([8.0, 4.0, 0.0], [[3.0, 1.0]]),
        ([360.0, 180.0, 0.0], [[3.0, 1.0]]),
    ],
)
def test_monotonic_descending_destination_retains_cell_order(
    target_edges: list[float], expected: list[list[float]]
) -> None:
    """Already-supported descending target cells must not become eastward wraps."""
    upper = target_edges[0]
    remapper = ConservativeRectilinearRemapper(
        np.array([0.0, upper / 2, upper]),
        np.array([-10.0, 10.0]),
        np.array(target_edges),
        np.array([-10.0, 10.0]),
    )
    assert_allclose_compact(
        remapper.apply_scalar(jnp.array([[1.0, 3.0]])), expected, rtol=0, atol=1e-14
    )


def test_descending_destination_aligns_with_shifted_source_support() -> None:
    """Anchor descending support at its lower edge so one periodic shift suffices."""
    remapper = ConservativeRectilinearRemapper(
        np.array([170.0, 350.0, 530.0]),
        np.array([-10.0, 10.0]),
        np.array([180.0, 0.0, -180.0]),
        np.array([-10.0, 10.0]),
    )
    assert_allclose_compact(
        remapper.apply_scalar(jnp.array([[1.0, 3.0]])),
        np.array([[26.0 / 9.0, 10.0 / 9.0]]),
        rtol=0,
        atol=1e-14,
    )


def test_broad_destination_collects_all_periodic_pieces_after_anchoring() -> None:
    """An explicit 420-degree cell includes repeated 60-degree coverage."""
    remapper = ConservativeRectilinearRemapper(
        np.array([180.0, 360.0, 540.0]),
        np.array([-10.0, 10.0]),
        np.array([179.0, 599.0]),
        np.array([-10.0, 10.0]),
    )
    assert_allclose_compact(
        remapper.apply_scalar(jnp.array([[1.0, 3.0]])),
        np.array([[391.0 / 210.0]]),
        rtol=0,
        atol=1e-14,
    )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("wrapped_axis", ["source", "target", "both"])
@pytest.mark.parametrize("origin", [0.0, 720.0, -720.0])
def test_wrapped_global_edges_retain_coverage_and_first_strip(
    enable_x64: bool, wrapped_axis: str, origin: float
) -> None:
    """Dropping the 360-to-4 cell loses half of the first T31 target column."""
    with jax.enable_x64(enable_x64):
        source_edges = np.arange(91) * 4.0
        target_edges = np.arange(97) * 3.75 - 1.875
        if wrapped_axis in {"source", "both"}:
            source_edges[0] = 360.0
        if wrapped_axis in {"target", "both"}:
            target_edges = target_edges % 360.0
        remapper = ConservativeRectilinearRemapper(
            source_edges + origin,
            np.array([-80.0, 0.0, 80.0]),
            target_edges,
            np.array([-80.0, 0.0, 80.0]),
        )
        remap = jax.jit(remapper.apply_scalar)
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            remap(jnp.ones((2, 90))), np.ones((2, 96)), rtol=0, atol=2 * eps
        )
        strip = jnp.zeros((2, 90)).at[:, 0].set(1.0)
        expected = np.zeros((2, 96))
        expected[:, :2] = [0.5, 17.0 / 30.0]
        assert_allclose_compact(remap(strip), expected, rtol=0, atol=2 * eps)


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("normalize", ["conservation", "fracarea"])
def test_wrapped_interior_seam_preserves_masked_values_and_derivatives(
    enable_x64: bool, normalize: str
) -> None:
    """Unwrap geometry without rolling data, losing masks, or hiding AD paths."""
    with jax.enable_x64(enable_x64):
        remapper = ConservativeRectilinearRemapper(
            np.array([356.0, 0.0, 4.0, 8.0]),
            np.array([-10.0, 10.0]),
            np.array([-4.0, 2.0, 8.0]),
            np.array([-10.0, 10.0]),
            src_mask=np.array([[False, False, True]]),
            normalize=normalize,
        )
        field = jnp.array([[3.0, 9.0, jnp.nan]])
        remap = jax.jit(remapper.apply_scalar)
        scale = 3.0 if normalize == "conservation" else 1.0
        expected = np.array([[5.0, 9.0 / scale]])
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(remap(field), expected, rtol=4 * eps, atol=0)
        tangent = jnp.array([[0.0, 1.0, 0.0]])
        _, jvp = jax.jvp(remap, (field,), (tangent,))
        assert_allclose_compact(
            jvp, np.array([[1.0 / 3.0, 1.0 / scale]]), rtol=4 * eps, atol=0
        )
        gradient = jax.grad(lambda x: jnp.sum(remap(x)))(field)
        assert_allclose_compact(
            gradient,
            np.array([[2.0 / 3.0, 1.0 / 3.0 + 1.0 / scale, 0.0]]),
            rtol=4 * eps,
            atol=0,
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_wrapped_native_strip_integral_uses_its_four_degree_area(
    enable_x64: bool,
) -> None:
    """Conservation diagnostics and destination areas use unwrapped geometry."""
    with jax.enable_x64(enable_x64):
        remapper = ConservativeRectilinearRemapper(
            np.r_[360.0, np.arange(1, 91) * 4.0],
            np.array([-80.0, 0.0, 80.0]),
            (np.arange(97) * 3.75 - 1.875) % 360.0,
            np.array([-80.0, 0.0, 80.0]),
        )
        wet = jnp.zeros((2, 90)).at[0, 0].set(1.0)
        mapped = remapper.apply_scalar(wet)
        # One wet source cell: R^2 * sin(80 degrees) * radians(4 degrees).
        expected = 6371.0**2 * np.sin(np.deg2rad(80.0)) * np.deg2rad(4.0)
        actual = jnp.sum(mapped.reshape(-1) * remapper.dst_areas)
        assert_allclose_compact(
            actual, expected, rtol=2e-6 if not enable_x64 else 2e-14, atol=0
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_wrapped_ocean_footprint_preserves_only_prescribed_polar_marine_area(
    enable_x64: bool,
) -> None:
    """A longitude seam must not manufacture prescribed marine midlatitudes."""
    with jax.enable_x64(enable_x64):
        ocean = RectilinearGrid(
            "ocean",
            longitude=np.arange(90) * 4.0 + 2.0,
            latitude=np.array([-40.0, 40.0]),
            longitude_edges=np.r_[360.0, np.arange(1, 91) * 4.0],
            latitude_edges=np.array([-80.0, 0.0, 80.0]),
            binary_mask=np.ones((2, 90)),
        )
        atmosphere = RectilinearGrid(
            "atmosphere",
            longitude=np.arange(96) * 3.75,
            latitude=np.array([-85.0, -40.0, 40.0, 85.0]),
            longitude_edges=np.arange(97) * 3.75 - 1.875,
            latitude_edges=np.array([-90.0, -80.0, 0.0, 80.0, 90.0]),
        )
        coupled, prescribed, land = _surface_fractions_from_ocean(
            atmosphere, ocean, jnp.full(atmosphere.shape, 0.25)
        )
        assert_allclose_compact(coupled[1:3], np.ones((2, 96)), rtol=0, atol=0)
        assert_allclose_compact(prescribed[1:3], np.zeros((2, 96)), rtol=0, atol=0)
        assert_allclose_compact(prescribed[[0, 3], :], 0.75, rtol=0, atol=0)
        assert_allclose_compact(land[[0, 3], :], 0.25, rtol=0, atol=0)
        field = jnp.arange(180.0).reshape(2, 90)
        identity = conservative(ocean, ocean)
        assert identity.has_identical_grids
        assert identity.regrid(field) is field
