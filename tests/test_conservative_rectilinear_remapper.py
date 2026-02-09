import numpy as np
import pytest

from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)


def _make_remapper(
    src_lon_edges: np.ndarray,
    src_lat_edges: np.ndarray,
    dst_lon_edges: np.ndarray,
    dst_lat_edges: np.ndarray,
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

    np.testing.assert_allclose(out, np.full((4, 4), 7.0), rtol=0.0, atol=1e-14)


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
    np.testing.assert_allclose(dst_mass, src_mass, rtol=1e-12, atol=1e-8)


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

    np.testing.assert_allclose(out, src, rtol=0.0, atol=1e-14)


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

    np.testing.assert_allclose(
        out, np.array([[30.0, 10.0, 20.0]]), rtol=0.0, atol=1e-14
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
    np.testing.assert_allclose(out[0, 1], 2.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(out[1, 0], 3.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(out[1, 1], 4.0, rtol=0.0, atol=1e-14)


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
    np.testing.assert_allclose(out_frac[0, 1:], np.array([2.0]), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        out_frac[1, :], np.array([3.0, 4.0]), rtol=0.0, atol=1e-14
    )

    np.testing.assert_allclose(out_cons[0, 0], 0.0, rtol=0.0, atol=1e-14)


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
