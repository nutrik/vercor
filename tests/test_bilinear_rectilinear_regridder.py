import importlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.grids import RectilinearGrid
from vercor._regridders.base import _BaseRegridder
from vercor._regridders.bilinear import BilinearRectilinearRegridder
from vercor.regridding import bilinear
from vercor._interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator


def _make_grid(
    name: str,
    lon: Any,
    lat: Any,
    mask: Any | None = None,
) -> RectilinearGrid:
    return RectilinearGrid(name=name, longitude=lon, latitude=lat, binary_mask=mask)


def test_scalar_regrid_is_owned_by_shared_private_base() -> None:
    assert "regrid" not in BilinearRectilinearRegridder.__dict__
    assert BilinearRectilinearRegridder.regrid is _BaseRegridder.regrid


def test_regridder_constructor_sets_interpolator_and_grids() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 1.0, 2.0]))

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    assert regridder.source_grid is src_grid
    assert regridder.target_grid is dst_grid
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

    assert isinstance(regridder, BilinearRectilinearRegridder)
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
    out = regridder.regrid(src)

    assert out.shape == dst_grid.shape


def test_regridder_scalar_accepts_jax_array_input() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 2.0]))

    regridder = BilinearRectilinearRegridder(
        src_grid, dst_grid, periodic_longitude=False
    )

    src = jnp.arange(9.0).reshape(3, 3)
    out = regridder.regrid(src)

    assert out.shape == dst_grid.shape


def test_regridder_vector_call_dispatches_and_returns_two_destination_arrays() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 2.0]), np.array([0.0, 2.0]))

    regridder = BilinearRectilinearRegridder(
        src_grid, dst_grid, periodic_longitude=False
    )

    u_src = np.ones(src_grid.shape, dtype=float)
    v_src = -2.0 * np.ones(src_grid.shape, dtype=float)

    out = regridder.regrid_vector(u_src, v_src)

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert out[0].shape == dst_grid.shape
    assert out[1].shape == dst_grid.shape


def test_regridder_identical_grid_skips_interpolator_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bilinear_module = importlib.import_module("vercor._regridders.bilinear")

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

    regridder = bilinear(src_grid, dst_grid)

    assert regridder.has_identical_grids is True
    assert isinstance(regridder, BilinearRectilinearRegridder)
    assert regridder.interpolator is None


def test_regridder_non_identical_grid_constructs_interpolator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bilinear_module = importlib.import_module("vercor._regridders.bilinear")
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


def test_regridder_identical_grid_scalar_short_circuit_returns_input_object() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    src = np.arange(6.0).reshape(2, 3)
    out = regridder.regrid(src)

    assert out is src


def test_regridder_identical_grid_scalar_short_circuit_with_jax_backed_coords() -> None:
    lon = jnp.asarray([0.0, 1.0, 2.0])
    lat = jnp.asarray([0.0, 1.0])
    src_grid = _make_grid("src", lon, lat)
    dst_grid = _make_grid("dst", lon, lat)

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)
    src = jnp.arange(6.0).reshape(2, 3)

    out = regridder.regrid(src)

    assert out is src


def test_regridder_identical_grid_vector_short_circuit_returns_input_objects() -> None:
    lon = np.array([0.0, 1.0, 2.0])
    lat = np.array([0.0, 1.0])
    src_grid = _make_grid("src", lon.copy(), lat.copy())
    dst_grid = _make_grid("dst", lon.copy(), lat.copy())

    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    u_src = np.ones((2, 3), dtype=float)
    v_src = -np.ones((2, 3), dtype=float)

    out_u, out_v = regridder.regrid_vector(u_src, v_src)

    assert out_u is u_src
    assert out_v is v_src


def test_regridder_is_not_callable() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    regridder = BilinearRectilinearRegridder(src_grid, dst_grid)

    assert not callable(regridder)


def test_bilinear_factory_returns_bilinear_rectilinear_regridder() -> None:
    src_grid = _make_grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst_grid = _make_grid("dst", np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    regridder = bilinear(src_grid, dst_grid)

    assert isinstance(regridder, BilinearRectilinearRegridder)
    assert regridder.source_grid is src_grid
    assert regridder.target_grid is dst_grid


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

    assert isinstance(regridder, BilinearRectilinearRegridder)
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


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
def test_bilinear_periodic_longitude_turns_preserve_scalar_vector_and_source_ad(
    enable_x64: bool,
) -> None:
    """Whole longitude turns must leave interpolation and source weights unchanged."""
    with jax.enable_x64(enable_x64):
        source = _make_grid("source", [0.0, 90.0, 180.0, 270.0], [-1.0, 1.0])
        target = _make_grid("target", [-675.0, -315.0, 45.0, 405.0, 765.0], [0.0])
        regridder = BilinearRectilinearRegridder(source, target)
        values = jnp.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
        compiled = jax.jit(regridder.regrid)
        assert_allclose_compact(compiled(values), 0.5, rtol=2e-6, atol=1e-7)

        # Five identical midpoint samples each use four source corners with
        # weight 1/4, so the VJP of their sum is exactly 5/4 on those corners.
        def objective(field: jax.Array) -> jax.Array:
            return jnp.sum(compiled(field))

        direction = jnp.arange(1.0, 9.0).reshape(2, 4)
        _, forward = jax.jvp(objective, (values,), (direction,))
        reverse = jax.grad(objective)(values)
        expected_gradient = np.array([[1.25, 1.25, 0.0, 0.0]] * 2)
        assert_allclose_compact(reverse, expected_gradient, rtol=2e-6, atol=1e-7)
        assert_allclose_compact(forward, 17.5, rtol=2e-6, atol=1e-7)

        vector_u, vector_v = regridder.regrid_vector(values, -2 * values)
        # Tangent-vector interpolation first averages Cartesian components.
        assert_allclose_compact(vector_u, np.sqrt(2.0) / 4.0, rtol=2e-6, atol=1e-6)
        assert_allclose_compact(
            vector_v, -np.cos(np.deg2rad(1.0)), rtol=2e-6, atol=1e-6
        )


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
def test_bilinear_nonperiodic_longitude_clamps_both_endpoints(
    enable_x64: bool,
) -> None:
    """A target west of the source domain must use its western boundary."""
    with jax.enable_x64(enable_x64):
        source = _make_grid("source", [0.0, 10.0], [-1.0, 1.0])
        target = _make_grid("target", [-5.0, 0.0, 5.0, 10.0, 15.0], [0.0])
        regridder = BilinearRectilinearRegridder(
            source, target, periodic_longitude=False
        )
        assert_allclose_compact(
            jax.jit(regridder.regrid)(jnp.asarray([[0.0, 10.0], [0.0, 10.0]])),
            np.array([[0.0, 0.0, 5.0, 10.0, 10.0]]),
            rtol=2e-6,
            atol=1e-7,
        )
        # With a uniform zonal source vector, projection to the target tangent
        # yields cos(5 degrees) outside either boundary and at the midpoint.
        vector_u, vector_v = regridder.regrid_vector(
            jnp.ones((2, 2)), jnp.zeros((2, 2))
        )
        cosine = np.cos(np.deg2rad(5.0))
        assert_allclose_compact(
            vector_u, np.array([[cosine, 1.0, cosine, 1.0, cosine]]), rtol=2e-6
        )
        assert_allclose_compact(vector_v, 0.0, atol=1e-7)
