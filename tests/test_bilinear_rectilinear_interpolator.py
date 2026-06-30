from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.dtypes import jax_index_dtype
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


def _scalar_interp(
    lon_src: Any,
    lat_src: Any,
    lon_tgt: Any,
    lat_tgt: Any,
    **kwargs: object,
) -> BilinearRectilinearInterpolator:
    return BilinearRectilinearInterpolator(
        lon_src=lon_src,
        lat_src=lat_src,
        lon_tgt=lon_tgt,
        lat_tgt=lat_tgt,
        **kwargs,  # type: ignore
    )


def test_init_rejects_non_monotonic_longitude() -> None:
    lon_src = np.array([0.0, 2.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="lon_src must be strictly monotonic"):
        _scalar_interp(lon_src, lat_src, np.array([0.5]), np.array([0.5]))


def test_init_rejects_non_monotonic_latitude() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="lat_src must be strictly monotonic"):
        _scalar_interp(lon_src, lat_src, np.array([0.5]), np.array([0.5]))


def test_init_accepts_jax_arrays_and_tracks_source_orientation() -> None:
    interp = _scalar_interp(
        jnp.asarray([2.0, 1.0, 0.0]),
        jnp.asarray([10.0, 0.0]),
        jnp.asarray([0.5, 1.5]),
        jnp.asarray([2.5]),
        periodic_longitude=False,
    )

    assert interp._lon_flipped is True
    assert interp.lat_descending is True
    assert_array_equal_compact(interp.lon_src_deg, np.array([0.0, 1.0, 2.0]))


def test_scalar_bilinear_exact_on_2x2_cell() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    lon_tgt = np.array([0.25])
    lat_tgt = np.array([0.75])

    # f(lon, lat) = 2*lon + 3*lat + 5 is exactly reproduced by bilinear interpolation
    src = np.array([[5.0, 7.0], [8.0, 10.0]])

    interp = _scalar_interp(
        lon_src, lat_src, lon_tgt, lat_tgt, periodic_longitude=False
    )
    out = interp.apply_scalar(src)

    assert_allclose_compact(out, np.array([[7.75]]), rtol=0.0, atol=1e-14)


def test_scalar_nan_renorm_true_renormalizes_valid_corners() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    src = np.array([[1.0, np.nan], [3.0, 5.0]])

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([0.5]),
        np.array([0.5]),
        periodic_longitude=False,
        nan_renorm=True,
        extrapolation_mode=None,
        fill_value=-999.0,
    )
    out = interp.apply_scalar(src)

    # Equal corner weights (0.25 each); renorm over valid corners => (1+3+5)/3
    assert_allclose_compact(out, np.array([[3.0]]), rtol=0.0, atol=1e-14)


def test_scalar_nan_renorm_false_falls_back_to_fill_when_corner_invalid() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    src = np.array([[1.0, np.nan], [3.0, 5.0]])

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([0.5]),
        np.array([0.5]),
        periodic_longitude=False,
        nan_renorm=False,
        extrapolation_mode=None,
        fill_value=-999.0,
    )
    out = interp.apply_scalar(src)

    assert_array_equal_compact(out, np.array([[-999.0]]))


def test_scalar_periodic_longitude_wrap_uses_dateline_cell() -> None:
    lon_src = np.array([0.0, 120.0, 240.0])
    lat_src = np.array([0.0, 10.0])
    src = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])

    interp_359 = _scalar_interp(lon_src, lat_src, np.array([359.0]), np.array([5.0]))
    interp_minus1 = _scalar_interp(lon_src, lat_src, np.array([-1.0]), np.array([5.0]))

    out_359 = interp_359.apply_scalar(src)
    out_minus1 = interp_minus1.apply_scalar(src)

    assert_allclose_compact(out_359, out_minus1, rtol=0.0, atol=1e-14)
    assert_array_equal_compact(
        interp_359.i0, np.array([[2]], dtype=np.dtype(jax_index_dtype()))
    )
    assert_array_equal_compact(
        interp_359.i1, np.array([[0]], dtype=np.dtype(jax_index_dtype()))
    )


def test_scalar_descending_latitude_supported() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([10.0, 0.0])  # descending
    src = np.array([[10.0, 20.0], [30.0, 40.0]])

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([0.5]),
        np.array([2.5]),
        periodic_longitude=False,
    )
    out = interp.apply_scalar(src)

    assert_allclose_compact(out, np.array([[30.0]]), rtol=0.0, atol=1e-14)


def test_scalar_target_mask_applies_fill_value() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    tgt_mask = np.array([[True, False], [False, True]])

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        periodic_longitude=False,
        tgt_mask=tgt_mask,
        fill_value=-7.0,
    )
    out = interp.apply_scalar(src)

    assert_array_equal_compact(out, np.array([[1.0, -7.0], [-7.0, 4.0]]))


def test_scalar_shape_mismatch_raises_value_error() -> None:
    interp = _scalar_interp(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.5]),
        np.array([0.5]),
        periodic_longitude=False,
    )

    with pytest.raises(ValueError, match="src field must have shape"):
        interp.apply_scalar(np.ones((3, 3), dtype=float))


def test_scalar_extrapolation_nearest() -> None:
    lon_src = np.array([0.0, 1.0, 2.0])
    lat_src = np.array([0.0, 1.0, 2.0])

    src = np.full((3, 3), 99.0)
    src[0, 0] = 11.0

    src_mask = np.zeros((3, 3), dtype=bool)
    src_mask[0, 0] = True

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([1.5]),
        np.array([1.5]),
        periodic_longitude=False,
        src_mask=src_mask,
        extrapolation_mode="nearest",
        fill_value=-999.0,
    )
    out = interp.apply_scalar(src)

    assert_array_equal_compact(out, np.array([[11.0]]))


def test_scalar_extrapolation_idw_k2_symmetric_mean() -> None:
    lon_src = np.array([0.0, 1.0, 2.0])
    lat_src = np.array([-1.0, 0.0, 1.0])

    src = np.full((3, 3), np.nan)
    src[0, 0] = 2.0
    src[0, 2] = 6.0

    src_mask = np.zeros((3, 3), dtype=bool)
    src_mask[0, 0] = True
    src_mask[0, 2] = True

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([1.0]),
        np.array([0.0]),
        periodic_longitude=False,
        src_mask=src_mask,
        extrapolation_mode="idw",
        idw_k=2,
        idw_eps=1e-12,
    )
    out = interp.apply_scalar(src)

    assert_allclose_compact(out, np.array([[4.0]]), rtol=0.0, atol=1e-12)


def test_scalar_invalid_extrapolation_mode_raises_when_used() -> None:
    lon_src = np.array([0.0, 1.0, 2.0])
    lat_src = np.array([0.0, 1.0, 2.0])

    src = np.full((3, 3), 0.0)
    src[0, 0] = 5.0

    src_mask = np.zeros((3, 3), dtype=bool)
    src_mask[0, 0] = True

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([1.5]),
        np.array([1.5]),
        periodic_longitude=False,
        src_mask=src_mask,
        extrapolation_mode="bogus",
    )

    with pytest.raises(ValueError, match="extrapolation_mode must be"):
        interp.apply_scalar(src)


def test_vector_constant_field_preserved() -> None:
    lon_src = np.array([0.0, 90.0])
    lat_src = np.array([-45.0, 45.0])

    interp = _scalar_interp(lon_src, lat_src, lon_src, lat_src, periodic_longitude=True)

    u_src = np.full((2, 2), 2.0)
    v_src = np.full((2, 2), -1.0)

    u_t, v_t = interp.apply_vector(u_src, v_src)

    assert_allclose_compact(u_t, u_src, rtol=0.0, atol=1e-14)
    assert_allclose_compact(v_t, v_src, rtol=0.0, atol=1e-14)


def test_vector_target_mask_applies_fill_value_to_both_components() -> None:
    lon_src = np.array([0.0, 1.0])
    lat_src = np.array([0.0, 1.0])
    tgt_mask = np.array([[True, False], [False, True]])

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        periodic_longitude=False,
        tgt_mask=tgt_mask,
        fill_value=-9.0,
    )

    u_src = np.array([[1.0, 2.0], [3.0, 4.0]])
    v_src = np.array([[5.0, 6.0], [7.0, 8.0]])

    u_t, v_t = interp.apply_vector(u_src, v_src)

    assert_array_equal_compact(u_t, np.array([[1.0, -9.0], [-9.0, 4.0]]))
    assert_array_equal_compact(v_t, np.array([[5.0, -9.0], [-9.0, 8.0]]))


def test_vector_shape_mismatch_raises_value_error() -> None:
    interp = _scalar_interp(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.5]),
        np.array([0.5]),
        periodic_longitude=False,
    )

    u_src = np.ones((2, 2), dtype=float)
    v_src = np.ones((2, 3), dtype=float)

    with pytest.raises(ValueError, match="must both have shape"):
        interp.apply_vector(u_src, v_src)


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_vector_extrapolation_nearest_for_invalid_bilinear_points() -> None:
    lon_src = np.array([0.0, 1.0, 2.0])
    lat_src = np.array([0.0, 1.0, 2.0])

    src_mask = np.zeros((3, 3), dtype=bool)
    src_mask[0, 0] = True

    u_src = np.full((3, 3), np.nan)
    v_src = np.full((3, 3), np.nan)
    u_src[0, 0] = 2.0
    v_src[0, 0] = -3.0

    interp = _scalar_interp(
        lon_src,
        lat_src,
        np.array([1.5]),
        np.array([1.5]),
        periodic_longitude=False,
        src_mask=src_mask,
        extrapolation_mode="nearest",
        fill_value=-999.0,
    )

    u_t, v_t = interp.apply_vector(u_src, v_src)

    assert_array_equal_compact(u_t, np.array([[2.0]]))
    assert_array_equal_compact(v_t, np.array([[-3.0]]))


def test_interpolator_pytree_round_trip_preserves_jittable_state() -> None:
    interp = _scalar_interp(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.25, 0.75]),
        np.array([0.25, 0.75]),
        periodic_longitude=False,
        src_mask=np.array([[True, False], [True, True]]),
        tgt_mask=np.array([[True, False], [False, True]]),
        extrapolation_mode="nearest",
    )

    leaves, treedef = jax.tree_util.tree_flatten(interp)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    src = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert isinstance(restored, BilinearRectilinearInterpolator)
    assert_allclose_compact(restored.apply_scalar(src), interp.apply_scalar(src))
    assert_array_equal_compact(restored.i0, interp.i0)
    assert_array_equal_compact(restored.i1, interp.i1)


def test_scalar_apply_supports_jax_jit() -> None:
    interp = _scalar_interp(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.25]),
        np.array([0.75]),
        periodic_longitude=False,
    )
    src = np.array([[5.0, 7.0], [8.0, 10.0]])

    out = jax.jit(interp.apply_scalar)(src)

    assert_allclose_compact(out, np.array([[7.75]]), rtol=0.0, atol=1e-14)


def test_vector_apply_supports_jax_jit() -> None:
    lon_src = np.array([0.0, 90.0])
    lat_src = np.array([-45.0, 45.0])
    interp = _scalar_interp(lon_src, lat_src, lon_src, lat_src, periodic_longitude=True)

    u_src = np.full((2, 2), 2.0)
    v_src = np.full((2, 2), -1.0)

    u_t, v_t = jax.jit(interp.apply_vector)(u_src, v_src)

    assert_allclose_compact(u_t, u_src, rtol=0.0, atol=1e-14)
    assert_allclose_compact(v_t, v_src, rtol=0.0, atol=1e-14)


def test_scalar_interpolation_supports_gradients_wrt_source_field() -> None:
    interp = _scalar_interp(
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        np.array([0.25]),
        np.array([0.75]),
        periodic_longitude=False,
    )

    def interpolated_value(src: jax.Array) -> Any:
        return interp.apply_scalar(src)[0, 0]

    grad = jax.grad(interpolated_value)(np.array([[5.0, 7.0], [8.0, 10.0]]))

    assert_allclose_compact(grad, np.array([[0.1875, 0.0625], [0.5625, 0.1875]]))
