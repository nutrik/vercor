import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.dtypes import jax_index_dtype, jax_real_dtype
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.types import RuntimeArray


def _make_remapper(
    src_lon_edges: RuntimeArray,
    src_lat_edges: RuntimeArray,
    dst_lon_edges: RuntimeArray,
    dst_lat_edges: RuntimeArray,
    **kwargs: object,
) -> ConservativeRectilinearRemapper:
    return ConservativeRectilinearRemapper(
        src_lon_edges=src_lon_edges,
        src_lat_edges=src_lat_edges,
        dst_lon_edges=dst_lon_edges,
        dst_lat_edges=dst_lat_edges,
        **kwargs,  # type: ignore
    )


def test_apply_scalar_shape_mismatch_raises_value_error() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([0.0, 1.0, 2.0]),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        remapper.apply_scalar(np.ones((3, 3), dtype=float))


def test_apply_vector_not_implemented_raises_runtime_error() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([0.0, 1.0, 2.0]),
    )

    with pytest.raises(RuntimeError, match="not implemented"):
        remapper.apply_vector(
            np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)
        )


def test_constant_field_preserved_on_refinement_conservation_mode() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        normalize="conservation",
    )

    src = np.full((2, 2), 7.0)
    out = remapper.apply_scalar(src)

    assert_allclose_compact(out, np.full((4, 4), 7.0), rtol=0.0, atol=1e-14)


def test_remapper_pytree_round_trip_preserves_runtime_arrays_only() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        normalize="fracarea",
    )

    leaves, treedef = jax.tree_util.tree_flatten(remapper)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, ConservativeRectilinearRemapper)
    assert restored.normalize == "fracarea"
    assert restored.radius == remapper.radius
    assert_allclose_compact(restored.src_lon_b, remapper.src_lon_b)
    assert_allclose_compact(restored.src_lat_b, remapper.src_lat_b)
    assert_allclose_compact(restored.dst_lon_b, remapper.dst_lon_b)
    assert_allclose_compact(restored.dst_lat_b, remapper.dst_lat_b)
    assert_allclose_compact(restored.dst_areas, remapper.dst_areas)
    assert not hasattr(restored, "fracarea_norm")
    assert not hasattr(restored, "_n_src_cells")


def test_remapper_accepts_jax_backed_constructor_inputs() -> None:
    remapper = _make_remapper(
        src_lon_edges=jnp.asarray([0.0, 1.0, 2.0]),
        src_lat_edges=jnp.asarray([0.0, 1.0, 2.0]),
        dst_lon_edges=jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        src_mask=jnp.asarray([[False, True], [False, False]]),
        normalize="fracarea",
    )

    assert remapper.src_lon_b.dtype == jax_real_dtype()
    assert remapper.src_lat_b.dtype == jax_real_dtype()
    assert remapper.dst_lon_b.dtype == jax_real_dtype()
    assert remapper.dst_lat_b.dtype == jax_real_dtype()
    assert remapper.dst_indices.dtype == jax_index_dtype()
    assert remapper.src_indices.dtype == jax_index_dtype()
    assert remapper.overlap_weights.dtype == jax_real_dtype()


def test_mass_conserved_between_source_and_destination() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        normalize="conservation",
    )

    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = remapper.apply_scalar(src)

    src_mass = remapper.get_src_total_mass(src)
    dst_mass = remapper.get_dst_total_mass(out)

    # Sparse-matrix assembly and trigonometric area terms accumulate tiny
    # floating-point error; enforce near-machine-precision agreement.
    assert_allclose_compact(dst_mass, src_mass, rtol=1e-12, atol=1e-8)


def test_mass_helpers_accept_jax_arrays() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        normalize="conservation",
    )

    src = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    dst = remapper.apply_scalar(src)

    assert_allclose_compact(
        remapper.get_dst_total_mass(dst),
        remapper.get_src_total_mass(src),
        rtol=1e-12,
        atol=1e-8,
    )


def test_apply_scalar_supports_jax_jit_linearity_and_gradients() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        normalize="conservation",
    )

    src_a = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    src_b = jnp.asarray([[0.5, 1.5], [2.5, 3.5]])
    jitted_apply = jax.jit(remapper.apply_scalar)

    out_a = jitted_apply(src_a)
    out_b = jitted_apply(src_b)
    out_sum = jitted_apply(src_a + src_b)

    assert_allclose_compact(out_sum, out_a + out_b, rtol=1e-12, atol=1e-12)

    gradient = jax.grad(lambda field: jnp.sum(jitted_apply(field)))(src_a)
    assert gradient.shape == src_a.shape
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_descending_lat_bounds_round_trip_orientation() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([2.0, 1.0, 0.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([2.0, 1.0, 0.0]),
        normalize="conservation",
    )

    src = np.array([[10.0, 20.0], [30.0, 40.0]])
    out = remapper.apply_scalar(src)

    assert_allclose_compact(out, src, rtol=0.0, atol=1e-14)


def test_periodic_longitude_overlap_maps_shifted_destination_cells() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 120.0, 240.0, 360.0]),
        src_lat_edges=np.array([0.0, 1.0]),
        dst_lon_edges=np.array([-120.0, 0.0, 120.0, 240.0]),
        dst_lat_edges=np.array([0.0, 1.0]),
        normalize="conservation",
    )

    src = np.array([[10.0, 20.0, 30.0]])
    out = remapper.apply_scalar(src)

    assert_allclose_compact(out, np.array([[30.0, 10.0, 20.0]]), rtol=0.0, atol=1e-14)


def test_periodic_overlap_merges_duplicate_shift_contributions() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 180.0, 360.0]),
        src_lat_edges=np.array([0.0, 1.0]),
        dst_lon_edges=np.array([-30.0, 390.0]),
        dst_lat_edges=np.array([0.0, 1.0]),
        normalize="conservation",
        radius=1.0,
    )

    lat_overlap = np.sin(np.deg2rad(1.0)) - np.sin(np.deg2rad(0.0))
    expected_weight = lat_overlap * np.deg2rad(210.0)

    assert remapper.dst_indices.shape == (2,)
    assert_allclose_compact(remapper.dst_indices, np.array([0, 0]), rtol=0.0, atol=0.0)
    assert_allclose_compact(remapper.src_indices, np.array([0, 1]), rtol=0.0, atol=0.0)
    assert_allclose_compact(
        remapper.overlap_weights,
        np.array([expected_weight, expected_weight]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_source_mask_drops_precomputed_triplets_eagerly() -> None:
    remapper = _make_remapper(
        src_lon_edges=jnp.asarray([0.0, 1.0, 2.0]),
        src_lat_edges=jnp.asarray([0.0, 1.0, 2.0]),
        dst_lon_edges=jnp.asarray([0.0, 1.0, 2.0]),
        dst_lat_edges=jnp.asarray([0.0, 1.0, 2.0]),
        src_mask=jnp.asarray([[True, False], [False, False]]),
        normalize="conservation",
    )

    assert remapper.src_indices.shape == (3,)
    assert remapper.dst_indices.shape == (3,)
    assert_allclose_compact(
        remapper.src_indices, np.array([1, 2, 3]), rtol=0.0, atol=0.0
    )
    assert_allclose_compact(
        remapper.dst_indices, np.array([1, 2, 3]), rtol=0.0, atol=0.0
    )


@pytest.mark.filterwarnings(
    "ignore:Input has data type int64, but the output has been cast to float64\\.:FutureWarning"
)
def test_source_mask_with_fracarea_gives_nan_for_fully_masked_target_cell() -> None:
    src_mask = np.array([[True, False], [False, False]])
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([0.0, 1.0, 2.0]),
        src_mask=src_mask,
        normalize="fracarea",
    )

    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = remapper.apply_scalar(src)

    assert np.isnan(out[0, 0])
    assert_allclose_compact(out[0, 1], 2.0, rtol=0.0, atol=1e-14)
    assert_allclose_compact(out[1, 0], 3.0, rtol=0.0, atol=1e-14)
    assert_allclose_compact(out[1, 1], 4.0, rtol=0.0, atol=1e-14)


def test_nan_handling_differs_between_fracarea_and_conservation() -> None:
    src = np.array([[np.nan, 2.0], [3.0, 4.0]])

    remap_frac = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([0.0, 1.0, 2.0]),
        normalize="fracarea",
    )
    out_frac = remap_frac.apply_scalar(src)

    remap_cons = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0]),
        src_lat_edges=np.array([0.0, 1.0, 2.0]),
        dst_lon_edges=np.array([0.0, 1.0, 2.0]),
        dst_lat_edges=np.array([0.0, 1.0, 2.0]),
        normalize="conservation",
    )
    out_cons = remap_cons.apply_scalar(src)

    assert np.isnan(out_frac[0, 0])
    assert_allclose_compact(out_frac[0, 1:], np.array([2.0]), rtol=0.0, atol=1e-14)
    assert_allclose_compact(out_frac[1, :], np.array([3.0, 4.0]), rtol=0.0, atol=1e-14)

    assert_allclose_compact(out_cons[0, 0], 0.0, rtol=0.0, atol=1e-14)


def test_get_src_areas_shape_and_positive() -> None:
    remapper = _make_remapper(
        src_lon_edges=np.array([0.0, 1.0, 2.0, 3.0]),
        src_lat_edges=np.array([-1.0, 0.0, 1.0]),
        dst_lon_edges=np.array([0.0, 1.5, 3.0]),
        dst_lat_edges=np.array([-1.0, 1.0]),
    )

    areas = remapper.get_src_areas()
    assert areas.shape == (2, 3)
    assert np.all(areas > 0.0)
