import numpy as np
import pytest

from vercor.grid import RectilinearGrid
from vercor.interpolators.conservative_remap_rectilinear import ConservativeRectilinearRemapper
from vercor.regridders.conservative import (
    ConservativeRectilinearRegridder,
    conservative,
)


def _grid(
    name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    lon_edges: np.ndarray | None = None,
    lat_edges: np.ndarray | None = None,
) -> RectilinearGrid:
    return RectilinearGrid(
        name=name,
        longitude=lon,
        latitude=lat,
        longitude_edges=lon_edges,
        latitude_edges=lat_edges,
    )


def test_regridder_constructor_sets_interpolator_and_grids() -> None:
    src = _grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst = _grid("dst", np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    regridder = ConservativeRectilinearRegridder(src, dst)

    assert regridder.source_grid is src
    assert regridder.destination_grid is dst
    assert regridder.interpolator is not None


def test_regridder_constructor_uses_provided_edges_when_available() -> None:
    src_lon = np.array([0.5, 1.5])
    src_lat = np.array([0.5, 1.5])
    src_lon_edges = np.array([0.0, 1.0, 2.0])
    src_lat_edges = np.array([0.0, 1.0, 2.0])

    dst_lon = np.array([0.25, 0.75, 1.25, 1.75])
    dst_lat = np.array([0.25, 0.75, 1.25, 1.75])
    dst_lon_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    dst_lat_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    src = _grid("src", src_lon, src_lat, src_lon_edges, src_lat_edges)
    dst = _grid("dst", dst_lon, dst_lat, dst_lon_edges, dst_lat_edges)

    regridder = ConservativeRectilinearRegridder(
        src, dst, normalize="fracarea", radius=10.0
    )
    interp = regridder.interpolator
    assert interp is not None

    if isinstance(interp, ConservativeRectilinearRemapper):
        np.testing.assert_array_equal(interp.src_lon_b, src_lon_edges)
        np.testing.assert_array_equal(interp.src_lat_b, src_lat_edges)
        np.testing.assert_array_equal(interp.dst_lon_b, dst_lon_edges)
        np.testing.assert_array_equal(interp.dst_lat_b, dst_lat_edges)
        assert interp.normalize == "fracarea"
        np.testing.assert_allclose(interp.radius, 10.0, rtol=0.0, atol=0.0)


def test_regridder_scalar_call_dispatches_and_returns_destination_shape() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    regridder = ConservativeRectilinearRegridder(src, dst)
    src_field = np.array([[1.0, 2.0], [3.0, 4.0]])

    out = regridder(src_field)

    assert isinstance(out, np.ndarray)
    assert out.shape == dst.shape


def test_regridder_vector_call_raises_runtime_error_from_remapper() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    regridder = ConservativeRectilinearRegridder(src, dst)

    with pytest.raises(RuntimeError, match="not implemented"):
        regridder(np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float))


def test_regridder_has_identical_grids_true_for_equal_coords() -> None:
    lon = np.array([0.5, 1.5])
    lat = np.array([0.5, 1.5])
    src = _grid("src", lon.copy(), lat.copy())
    dst = _grid("dst", lon.copy(), lat.copy())

    regridder = ConservativeRectilinearRegridder(src, dst)
    assert regridder.has_identical_grids is True


def test_regridder_has_identical_grids_false_for_different_coords() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5, 2.5]))

    regridder = ConservativeRectilinearRegridder(src, dst)
    assert regridder.has_identical_grids is False


def test_regridder_identical_grid_scalar_short_circuit_returns_input_object() -> None:
    lon = np.array([0.5, 1.5])
    lat = np.array([0.5, 1.5])
    src = _grid("src", lon.copy(), lat.copy())
    dst = _grid("dst", lon.copy(), lat.copy())

    regridder = ConservativeRectilinearRegridder(src, dst)
    src_field = np.array([[1.0, 2.0], [3.0, 4.0]])

    out = regridder(src_field)

    assert out is src_field


def test_regridder_call_with_invalid_arg_count_raises_type_error() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    regridder = ConservativeRectilinearRegridder(src, dst)

    with pytest.raises(TypeError, match="Provide scalar_src"):
        regridder()

    with pytest.raises(TypeError, match="Provide scalar_src"):
        regridder(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))


def test_regridder_source_mask_excludes_masked_cells_in_fracarea_mode() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    # Use a different destination grid so Regridder.__call__ does not short-circuit
    # on has_identical_grids=True.
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    source_mask = np.array([[True, False], [False, False]])
    regridder = ConservativeRectilinearRegridder(
        src,
        dst,
        source_mask=source_mask,
        normalize="fracarea",
    )

    src_field = np.array([[1.0, 2.0], [3.0, 4.0]])
    out: np.ndarray = np.asarray(regridder(src_field))

    assert out.shape == (4, 4)
    assert np.all(np.isnan(out[0:2, 0:2]))
    np.testing.assert_allclose(out[0:2, 2:4], 2.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(out[2:4, 0:2], 3.0, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(out[2:4, 2:4], 4.0, rtol=0.0, atol=1e-14)


def test_conservative_factory_returns_conservative_rectilinear_regridder() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.5, 1.5]))

    regridder = conservative(src, dst)

    assert isinstance(regridder, ConservativeRectilinearRegridder)
    assert regridder.source_grid is src
    assert regridder.destination_grid is dst
