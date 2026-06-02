import importlib
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_array_equal_compact
from vercor.grid import RectilinearGrid
from vercor.regridders.bilinear import BilinearRectilinearRegridder, bilinear
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


def _make_grid(
    name: str,
    lon: Any,
    lat: Any,
    mask: Any | None = None,
) -> RectilinearGrid:
    return RectilinearGrid(name=name, longitude=lon, latitude=lat, binary_mask=mask)


def test_regridder_constructor_sets_interpolator_and_grids() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 1.0, 2.0]))

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    assert regridder.source_grid is src_grid
    assert regridder.destination_grid is dst_grid
    assert regridder.interpolator is not None


def test_regridder_constructor_propagates_interpolator_options() -> None:
    src_mask = np.array([[True, False, True], [True, True, False]])
    dst_mask = np.array([[True, False], [False, True], [True, True]])

    src_grid = _make_grid(
        "src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), mask=src_mask
    )
    dst_grid = _make_grid(
        "dst", np.array([0.0, 2.0]), np.array([0.0, 1.0, 2.0]), mask=dst_mask
    )

    regridder = BilinearRectilinearRegridder(
        src_grid,
        dst_grid,
        periodic_longitude=False,
        nan_renorm=False,
        extrapolation_mode="nearest",
        idw_k=3,
        idw_eps=1e-6,
        fill_value=-99.0,
    )

    interp = regridder.interpolator
    if isinstance(interp, BilinearRectilinearInterpolator):
        assert interp is not None
        assert interp.periodic is False
        assert interp.nan_renorm is False
        assert interp.extrapolation_mode == "nearest"
        assert interp.idw_k == 3
        assert np.isclose(interp.idw_eps, 1e-6)
        assert np.isclose(interp.fill_value, -99.0)
        assert_array_equal_compact(interp.src_mask, src_mask)
        assert_array_equal_compact(interp.tgt_mask, dst_mask)


def test_regridder_scalar_call_dispatches_and_returns_destination_shape() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 2.0]))

    regridder = BilinearRectilinearRegridder(
        src_grid, dst_grid, periodic_longitude=False
    )

    src = np.arange(9.0).reshape(3, 3)
    out = regridder(src)

    assert out.shape == dst_grid.shape


def test_regridder_scalar_accepts_jax_array_input() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 2.0]))

    regridder = BilinearRectilinearRegridder(
        src_grid, dst_grid, periodic_longitude=False
    )

    src = jnp.arange(9.0).reshape(3, 3)
    out = regridder(src)

    assert out.shape == dst_grid.shape


def test_regridder_vector_call_dispatches_and_returns_two_destination_arrays() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 2.0]))

    regridder = BilinearRectilinearRegridder(
        src_grid, dst_grid, periodic_longitude=False
    )

    u_src = np.ones(src_grid.shape, dtype=float)
    v_src = -2.0 * np.ones(src_grid.shape, dtype=float)

    out = regridder(u_src, v_src)

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == dst_grid.shape
    assert out[1].shape == dst_grid.shape


def test_regridder_has_identical_grids_true_for_equal_coords() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)
    assert regridder.has_identical_grids is True


def test_regridder_identical_grid_skips_interpolator_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bilinear_module = importlib.import_module("vercor.regridders.bilinear")

    def fail_if_called(*args: Any, **kwargs: Any) -> object:
        _ = args, kwargs
        raise AssertionError("identity regridder should not build interpolator")

    monkeypatch.setattr(
        bilinear_module, "BilinearRectilinearInterpolator", fail_if_called
    )

    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    assert regridder.has_identical_grids is True
    assert regridder.interpolator is not None


def test_regridder_non_identical_grid_constructs_interpolator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bilinear_module = importlib.import_module("vercor.regridders.bilinear")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_call(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(bilinear_module, "BilinearRectilinearInterpolator", record_call)

    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.0]))

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    assert regridder.has_identical_grids is False
    assert len(calls) == 1


def test_regridder_identical_grid_uses_shared_coordinate_tolerance() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon + 1e-16, lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)
    assert regridder.has_identical_grids is True


def test_regridder_has_identical_grids_false_for_different_coords() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.0]))

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)
    assert regridder.has_identical_grids is False


def test_regridder_identical_grid_scalar_short_circuit_returns_input_object() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    src = np.arange(6.0).reshape(2, 3)
    out = regridder(src)

    assert out is src


def test_regridder_identical_grid_scalar_short_circuit_with_jax_backed_coords() -> None:
    lon = jnp.asarray([0.0, 1.0, 2.0])
    lat = jnp.asarray([0.0, 1.0])
    src_grid = _make_grid("src", lon, lat)
    dst_grid = _make_grid("dst", lon, lat)

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)
    src = jnp.arange(6.0).reshape(2, 3)

    out = regridder(src)

    assert out is src


def test_regridder_identical_grid_vector_short_circuit_returns_input_objects() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    u_src = np.ones((2, 3), dtype=float)
    v_src = -np.ones((2, 3), dtype=float)

    out_u, out_v = regridder(u_src, v_src)

    assert out_u is u_src
    assert out_v is v_src


def test_regridder_call_with_invalid_arg_count_raises_type_error() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    with pytest.raises(TypeError, match="Provide scalar_src"):
        regridder()

    with pytest.raises(TypeError, match="Provide scalar_src"):
        regridder(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)))


def test_bilinear_factory_returns_bilinear_rectilinear_regridder() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    regridder = bilinear(src_grid, dst_grid)

    assert isinstance(regridder, BilinearRectilinearRegridder)
    assert regridder.source_grid is src_grid
    assert regridder.destination_grid is dst_grid


def test_bilinear_factory_forwards_interpolator_options() -> None:
    src_mask = np.array([[True, False, True], [True, True, False]])
    dst_mask = np.array([[True, False], [False, True], [True, True]])
    src_grid = _make_grid(
        "src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), mask=src_mask
    )
    dst_grid = _make_grid(
        "dst", np.array([0.0, 2.0]), np.array([0.0, 1.0, 2.0]), mask=dst_mask
    )

    regridder = bilinear(
        src_grid,
        dst_grid,
        periodic_longitude=False,
        nan_renorm=False,
        extrapolation_mode="nearest",
        idw_k=3,
        idw_eps=1e-6,
        fill_value=-99.0,
    )

    interp = regridder.interpolator
    assert isinstance(interp, BilinearRectilinearInterpolator)
    assert interp.periodic is False
    assert interp.nan_renorm is False
    assert interp.extrapolation_mode == "nearest"
    assert interp.idw_k == 3
    assert np.isclose(interp.idw_eps, 1e-6)
    assert np.isclose(interp.fill_value, -99.0)
    assert_array_equal_compact(interp.src_mask, src_mask)
    assert_array_equal_compact(interp.tgt_mask, dst_mask)
