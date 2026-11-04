import importlib
from inspect import signature
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact, assert_array_equal_compact
from vercor.grids import RectilinearGrid
from vercor._interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor._regridders.base import _BaseRegridder
from vercor._regridders.conservative import (
    ConservativeRectilinearRegridder,
)
from vercor.regridding import conservative
from vercor.regridding import conservative as public_conservative


def _grid(
    name: str,
    lon: Any,
    lat: Any,
    lon_edges: Any | None = None,
    lat_edges: Any | None = None,
) -> RectilinearGrid:
    return RectilinearGrid(
        name=name,
        longitude=lon,
        latitude=lat,
        longitude_edges=lon_edges,
        latitude_edges=lat_edges,
    )


def test_scalar_regrid_is_owned_by_shared_private_base() -> None:
    assert "regrid" not in ConservativeRectilinearRegridder.__dict__
    assert ConservativeRectilinearRegridder.regrid is _BaseRegridder.regrid


def test_regridder_constructor_sets_interpolator_and_grids() -> None:
    src = _grid("src", np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.0, 1.0]))

    regridder = ConservativeRectilinearRegridder(src, dst)

    assert regridder.source_grid is src
    assert regridder.target_grid is dst
    assert regridder.interpolator is not None


def test_conservative_regridder_api_does_not_expose_noop_fill_value() -> None:
    assert "fill_value" not in signature(ConservativeRectilinearRegridder).parameters
    assert "fill_value" not in signature(conservative).parameters


def test_public_conservative_factory_exposes_radius_km_only() -> None:
    parameters = signature(public_conservative).parameters
    assert "radius_km" in parameters
    assert "radius" not in parameters

    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.25, 0.75]), np.array([0.25, 0.75]))

    with pytest.raises(TypeError, match="radius"):
        public_conservative(src, dst, radius=10.0)  # type: ignore[call-arg]


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
    assert isinstance(regridder, ConservativeRectilinearRegridder)
    interp = regridder.interpolator
    assert interp is not None

    if isinstance(interp, ConservativeRectilinearRemapper):
        assert_array_equal_compact(interp.src_lon_b, src_lon_edges)
        assert_array_equal_compact(interp.src_lat_b, src_lat_edges)
        assert_array_equal_compact(interp.dst_lon_b, dst_lon_edges)
        assert_array_equal_compact(interp.dst_lat_b, dst_lat_edges)
        assert interp.normalize == "fracarea"
        assert_allclose_compact(interp.radius, 10.0, rtol=0.0, atol=0.0)


def test_regridder_constructor_accepts_mixed_numpy_and_jax_edges() -> None:
    src = _grid(
        "src",
        np.array([0.5, 1.5]),
        np.array([0.5, 1.5]),
        jnp.asarray([0.0, 1.0, 2.0]),
        np.array([0.0, 1.0, 2.0]),
    )
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
    )

    regridder = ConservativeRectilinearRegridder(
        src,
        dst,
        source_mask=jnp.asarray([[False, True], [False, False]]),
    )

    assert regridder.interpolator is not None
    out = regridder.regrid(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert np.shape(out) == dst.shape


def test_regridder_scalar_call_dispatches_and_returns_destination_shape() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    regridder = ConservativeRectilinearRegridder(src, dst)
    src_field = np.array([[1.0, 2.0], [3.0, 4.0]])

    out = regridder.regrid(src_field)

    assert np.shape(out) == dst.shape


def test_regridder_exposes_only_the_scalar_capability() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    regridder = ConservativeRectilinearRegridder(src, dst)

    with pytest.raises(AttributeError, match="regrid_vector"):
        getattr(regridder, "regrid_vector")(
            np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)
        )


def test_identical_grid_regridder_remains_scalar_only() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.5, 1.5]))

    regridder = ConservativeRectilinearRegridder(src, dst)

    with pytest.raises(AttributeError, match="regrid_vector"):
        getattr(regridder, "regrid_vector")(
            np.ones((2, 2), dtype=float), np.ones((2, 2), dtype=float)
        )


def test_regridder_identical_grid_skips_remapper_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conservative_module = importlib.import_module("vercor._regridders.conservative")

    def fail_if_called(*args: Any, **kwargs: Any) -> object:
        _ = args, kwargs
        raise AssertionError("identity regridder should not build remapper")

    monkeypatch.setattr(
        conservative_module, "ConservativeRectilinearRemapper", fail_if_called
    )

    lon = np.array([0.5, 1.5])
    lat = np.array([0.5, 1.5])
    src = _grid("src", lon.copy(), lat.copy())
    dst = _grid("dst", lon.copy(), lat.copy())

    regridder = conservative(src, dst)

    assert regridder.has_identical_grids is True
    assert isinstance(regridder, ConservativeRectilinearRegridder)
    assert regridder.interpolator is None


def test_regridder_non_identical_grid_constructs_remapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conservative_module = importlib.import_module("vercor._regridders.conservative")
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_call(*args: Any, **kwargs: Any) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        conservative_module, "ConservativeRectilinearRemapper", record_call
    )

    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5, 2.5]), np.array([0.5, 1.5, 2.5]))

    regridder = ConservativeRectilinearRegridder(src, dst)

    assert regridder.has_identical_grids is False
    assert len(calls) == 1


def test_regridder_identical_grid_scalar_short_circuit_returns_input_object() -> None:
    lon = np.array([0.5, 1.5])
    lat = np.array([0.5, 1.5])
    src = _grid("src", lon.copy(), lat.copy())
    dst = _grid("dst", lon.copy(), lat.copy())

    regridder = ConservativeRectilinearRegridder(src, dst)
    src_field = np.array([[1.0, 2.0], [3.0, 4.0]])

    out = regridder.regrid(src_field)

    assert out is src_field


def test_regridder_identical_grid_scalar_short_circuit_with_jax_backed_coords() -> None:
    lon = jnp.asarray([0.5, 1.5])
    lat = jnp.asarray([0.5, 1.5])
    src = _grid("src", lon, lat)
    dst = _grid("dst", lon, lat)

    regridder = ConservativeRectilinearRegridder(src, dst)
    src_field = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])

    out = regridder.regrid(src_field)

    assert out is src_field


def test_regridder_is_not_callable() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    regridder = ConservativeRectilinearRegridder(src, dst)

    assert not callable(regridder)


@pytest.mark.filterwarnings(
    "ignore:Input has data type int64, but the output has been cast to float64\\.:FutureWarning"
)
def test_regridder_source_mask_excludes_masked_cells_in_fracarea_mode() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    # Use a different destination grid so conservative regridding does not
    # short-circuit on has_identical_grids=True.
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
    out: np.ndarray = np.asarray(regridder.regrid(src_field))

    assert out.shape == (4, 4)
    assert np.all(np.isnan(out[0:2, 0:2]))
    assert_allclose_compact(out[0:2, 2:4], 2.0, rtol=0.0, atol=1e-14)
    assert_allclose_compact(out[2:4, 0:2], 3.0, rtol=0.0, atol=1e-14)
    assert_allclose_compact(out[2:4, 2:4], 4.0, rtol=0.0, atol=1e-14)


def test_regridder_accepts_jax_array_input() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )

    regridder = ConservativeRectilinearRegridder(src, dst)
    out = regridder.regrid(jnp.asarray([[1.0, 2.0], [3.0, 4.0]]))

    assert np.shape(out) == dst.shape
    assert np.all(np.isfinite(np.asarray(out)))


def test_conservative_factory_returns_conservative_rectilinear_regridder() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid("dst", np.array([0.5, 1.5]), np.array([0.5, 1.5]))

    regridder = conservative(src, dst)

    assert isinstance(regridder, ConservativeRectilinearRegridder)
    assert regridder.source_grid is src
    assert regridder.target_grid is dst


@pytest.mark.filterwarnings(
    "ignore:Input has data type int64, but the output has been cast to float64\\.:FutureWarning"
)
def test_conservative_factory_forwards_remapper_options() -> None:
    src = _grid("src", np.array([0.5, 1.5]), np.array([0.5, 1.5]))
    dst = _grid(
        "dst",
        np.array([0.25, 0.75, 1.25, 1.75]),
        np.array([0.25, 0.75, 1.25, 1.75]),
    )
    source_mask = np.array([[True, False], [False, False]])

    regridder = conservative(
        src,
        dst,
        source_mask=source_mask,
        normalize="fracarea",
        radius_km=10.0,
    )

    assert isinstance(regridder, ConservativeRectilinearRegridder)
    interp = regridder.interpolator
    assert isinstance(interp, ConservativeRectilinearRemapper)
    assert interp.normalize == "fracarea"
    assert_allclose_compact(interp.radius, 10.0, rtol=0.0, atol=0.0)

    out = np.asarray(regridder.regrid(np.array([[1.0, 2.0], [3.0, 4.0]])))
    assert np.all(np.isnan(out[0:2, 0:2]))
    assert_allclose_compact(out[0:2, 2:4], 2.0, rtol=0.0, atol=1e-14)


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("axis", ["longitude", "latitude"])
def test_same_centers_with_different_edges_remap_actual_cell_areas(
    enable_x64: bool, axis: str
) -> None:
    """A point-coordinate identity must not bypass unequal conservative cells."""
    with jax.enable_x64(enable_x64):
        longitude = np.array([0.0, 2.0])
        latitude = np.array([-1.0, 1.0])
        source = _grid(
            "source", longitude, latitude, [-1.0, 1.0, 3.0], [-2.0, 0.0, 2.0]
        )
        target = _grid(
            "target",
            longitude,
            latitude,
            [-1.0, 0.5, 3.0] if axis == "longitude" else [-1.0, 1.0, 3.0],
            [-2.0, -0.5, 2.0] if axis == "latitude" else [-2.0, 0.0, 2.0],
        )
        source_field = jnp.asarray([[0.0, 10.0], [0.0, 10.0]])
        expected = np.array([[0.0, 8.0], [0.0, 8.0]])
        if axis == "latitude":
            source_field = source_field.T
            # The upper target spans [-0.5, 2] degrees and receives nonzero
            # flux only over [0, 2], whose spherical area scales with sin(lat).
            upper = (
                10.0
                * np.sin(np.deg2rad(2.0))
                / (np.sin(np.deg2rad(2.0)) + np.sin(np.deg2rad(0.5)))
            )
            expected = np.array([[0.0, 0.0], [upper, upper]])

        regridder = conservative(source, target)
        assert_allclose_compact(
            regridder.regrid(source_field), expected, rtol=2e-6, atol=1e-7
        )


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("partial_grid", ["source", "target"])
@pytest.mark.parametrize("axis", ["longitude", "latitude"])
def test_single_explicit_edge_axis_is_retained(
    enable_x64: bool, partial_grid: str, axis: str
) -> None:
    """Only infer a missing axis instead of discarding the declared other axis."""
    with jax.enable_x64(enable_x64):
        partial = _grid(
            "partial",
            [0.0, 2.0],
            [-1.0, 1.0],
            [-1.0, 0.5, 3.0] if axis == "longitude" else None,
            [-2.0, -0.5, 2.0] if axis == "latitude" else None,
        )
        complete = _grid(
            "complete",
            [-0.25, 1.75] if axis == "longitude" else [0.0, 2.0],
            [-1.25, 0.75] if axis == "latitude" else [-1.0, 1.0],
            [-1.0, 0.5, 3.0] if axis == "longitude" else [-1.0, 1.0, 3.0],
            [-2.0, -0.5, 2.0] if axis == "latitude" else [-2.0, 0.0, 2.0],
        )
        source, target = (
            (partial, complete) if partial_grid == "source" else (complete, partial)
        )
        field = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
        # Both declared geometries have identical effective cell boundaries.
        assert_allclose_compact(
            conservative(source, target).regrid(field), field, rtol=2e-6, atol=1e-7
        )


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
@pytest.mark.parametrize("normalize", ["conservation", "fracarea"])
def test_identical_grid_applies_explicit_source_exclusion_mask(
    enable_x64: bool, normalize: str
) -> None:
    """The conservative identity path must not expose excluded source values."""
    with jax.enable_x64(enable_x64):
        grid = _grid("grid", [0.0, 2.0], [-1.0, 1.0])
        regridder = conservative(
            grid,
            grid,
            source_mask=np.array([[True, False], [False, False]]),
            normalize=normalize,
        )
        excluded = 0.0 if normalize == "conservation" else np.nan
        assert_allclose_compact(
            regridder.regrid(jnp.asarray([[1.0, 2.0], [3.0, 4.0]])),
            np.array([[excluded, 2.0], [3.0, 4.0]]),
            rtol=2e-6,
            atol=1e-7,
        )


@pytest.mark.fast_always
def test_identical_grid_rejects_invalid_conservative_normalization() -> None:
    grid = _grid("grid", [0.0, 2.0], [-1.0, 1.0])
    with pytest.raises(ValueError, match="normalize must be"):
        conservative(grid, grid, normalize="unsupported")


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True], ids=["float32", "float64"])
def test_equivalent_explicit_and_inferred_edges_preserve_identity(
    enable_x64: bool,
) -> None:
    with jax.enable_x64(enable_x64):
        source = _grid("source", [0.0, 2.0], [-1.0, 1.0])
        target = _grid(
            "target", [0.0, 2.0], [-1.0, 1.0], [-1.0, 1.0, 3.0], [-2.0, 0.0, 2.0]
        )
        field = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
        assert conservative(source, target).regrid(field) is field
