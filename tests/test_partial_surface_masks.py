"""Keep native prescribed surfaces outside a regional ocean footprint."""

from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from vercor import Exchange, RectilinearGrid
from vercor.exceptions import CouplerError
from vercor.grid_masks import create_land_mask_from_ocean
from vercor.jax_logging import get_default_logger
from vercor.topology import SurfaceMaskPolicy, TopologyContext

pytestmark = pytest.mark.fast_always


def grids(
    *, fractional_boundary: bool = False
) -> tuple[RectilinearGrid, RectilinearGrid]:
    """Return a global target and an all-wet regional ocean with explicit edges."""
    longitude = jnp.array([0.0, 90.0, 180.0, 270.0])
    atmosphere = RectilinearGrid(
        "atmosphere",
        longitude=longitude,
        latitude=jnp.array([-60.0, -20.0, 20.0, 60.0]),
    )
    ocean = RectilinearGrid(
        "ocean",
        longitude=longitude,
        latitude=(
            jnp.array([-15.0, 15.0])
            if fractional_boundary
            else jnp.array([-20.0, 20.0])
        ),
        latitude_edges=jnp.array([-30.0, 0.0, 30.0]) if fractional_boundary else None,
        binary_mask=jnp.ones((2, 4)),
    )
    return atmosphere, ocean


def test_default_keeps_historical_complement() -> None:
    atmosphere, ocean = grids()
    binary, land = create_land_mask_from_ocean(
        atmosphere.latitude, atmosphere.longitude, ocean
    )
    expected = np.array([1.0, 0.0, 0.0, 1.0])[:, None] * np.ones((1, 4))
    np.testing.assert_array_equal(binary, expected)
    np.testing.assert_allclose(land, expected)


@pytest.mark.parametrize("native_polar_land", [0.0, 0.4, 1.0])
def test_native_polar_land_retains_prescribed_ocean(native_polar_land: float) -> None:
    atmosphere, ocean = grids()
    binary, land = create_land_mask_from_ocean(
        atmosphere.latitude,
        atmosphere.longitude,
        ocean,
        native_land_fraction=jnp.full(atmosphere.shape, native_polar_land),
    )
    expected = np.array([native_polar_land, 0.0, 0.0, native_polar_land])[:, None]
    np.testing.assert_allclose(land, np.broadcast_to(expected, atmosphere.shape))
    np.testing.assert_array_equal(
        binary, np.broadcast_to(expected > 0, atmosphere.shape)
    )


def test_fractional_boundary_partitions_covered_native_and_land_area() -> None:
    from vercor.grid_masks import _surface_fractions_from_ocean

    atmosphere, ocean = grids(fractional_boundary=True)
    covered, prescribed, land = _surface_fractions_from_ocean(
        atmosphere, ocean, jnp.full(atmosphere.shape, 0.25)
    )
    footprint = np.array(
        [
            0.0,
            np.sin(np.deg2rad(30)) / np.sin(np.deg2rad(40)),
            np.sin(np.deg2rad(30)) / np.sin(np.deg2rad(40)),
            0.0,
        ]
    )[:, None]
    footprint = np.broadcast_to(footprint, atmosphere.shape)
    np.testing.assert_allclose(covered, footprint)
    np.testing.assert_allclose(prescribed, (1 - footprint) * 0.75)
    np.testing.assert_allclose(land, (1 - footprint) * 0.25)
    np.testing.assert_allclose(covered + prescribed + land, 1.0)


def test_native_terrain_does_not_replace_land_inside_ocean_rectangle() -> None:
    from vercor.grid_masks import _surface_fractions_from_ocean

    atmosphere, ocean = grids()
    ocean = RectilinearGrid(
        "coastal",
        longitude=ocean.longitude,
        latitude=ocean.latitude,
        binary_mask=jnp.array([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
    )
    covered, prescribed, land = _surface_fractions_from_ocean(
        atmosphere, ocean, jnp.zeros(atmosphere.shape)
    )
    expected_land = np.zeros(atmosphere.shape)
    expected_land[1, 0] = 1.0
    np.testing.assert_allclose(land, expected_land)
    np.testing.assert_array_equal(prescribed[1:3], 0.0)
    np.testing.assert_allclose(covered + prescribed + land, 1.0)


def test_partial_longitude_domain_keeps_prescribed_ocean_on_both_sides() -> None:
    from vercor.grid_masks import _surface_fractions_from_ocean

    atmosphere = RectilinearGrid(
        "shifted-global",
        longitude=jnp.array([-135.0, -45.0, 45.0, 135.0]),
        latitude=jnp.array([-60.0, -20.0, 20.0, 60.0]),
    )
    ocean = RectilinearGrid(
        "regional-longitude",
        longitude=jnp.array([-45.0, 45.0]),
        latitude=jnp.array([-20.0, 20.0]),
        binary_mask=jnp.ones((2, 2)),
    )
    covered, prescribed, land = _surface_fractions_from_ocean(
        atmosphere, ocean, jnp.zeros(atmosphere.shape)
    )
    expected = np.zeros(atmosphere.shape)
    expected[1:3, 1:3] = 1.0
    np.testing.assert_allclose(covered, expected)
    np.testing.assert_allclose(prescribed, 1.0 - expected)
    np.testing.assert_array_equal(land, 0.0)


@pytest.mark.parametrize(
    "bad_land",
    [
        jnp.ones((2, 2)),
        jnp.full((4, 4), -0.1),
        jnp.full((4, 4), 1.1),
        jnp.full((4, 4), jnp.nan),
    ],
)
def test_native_land_requires_finite_grid_shaped_fraction(bad_land: Any) -> None:
    atmosphere, ocean = grids()
    with pytest.raises(ValueError, match="native_land_fraction"):
        create_land_mask_from_ocean(
            atmosphere.latitude,
            atmosphere.longitude,
            ocean,
            native_land_fraction=bad_land,
        )


def topology(land_mask: Any) -> TopologyContext:
    """Build minimal role components without optional atmosphere frameworks."""
    atmosphere, ocean = grids()
    land = RectilinearGrid(
        "land",
        longitude=atmosphere.longitude,
        latitude=atmosphere.latitude,
        binary_mask=land_mask,
    )
    components = {
        name: SimpleNamespace(name=name, grid=grid)
        for name, grid in (("ATM", atmosphere), ("OCN", ocean), ("LND", land))
    }
    return TopologyContext(
        cast(Any, components),
        (Exchange("LND", "ATM", ("land_surface_temperature",), quantity="intensive"),),
        get_default_logger(),
    )


def test_surface_topology_preserves_actual_native_polar_land_support() -> None:
    land = np.zeros((4, 4))
    land[-1] = 1.0
    patch = SurfaceMaskPolicy(mode="required").build(topology(land))
    np.testing.assert_array_equal(patch.binary_masks["LND->ATM"], land)


def test_surface_topology_rejects_wrong_land_inside_covered_ocean() -> None:
    land = np.zeros((4, 4))
    land[1] = 1.0
    with pytest.raises(CouplerError, match="mismatched points"):
        SurfaceMaskPolicy(mode="required").build(topology(land))


def test_jcm_land_accepts_native_geometry_without_changing_forcing_records() -> None:
    from vercor.setups._data.jcm_land import make_jcm_land

    atmosphere, ocean = grids()
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=jnp.deg2rad(atmosphere.longitude),
            latitudes=jnp.deg2rad(atmosphere.latitude),
        )
    )
    forcing = SimpleNamespace(
        stl_am=jnp.full((4, 4), 280.0), soilw_am=jnp.full((4, 4), 0.3)
    )
    component = make_jcm_land(
        coords, forcing, ocean, native_land_fraction=jnp.zeros(atmosphere.shape)
    )
    np.testing.assert_array_equal(component.grid.binary_mask, 0.0)
    np.testing.assert_array_equal(component.spec.initial_fields["land_fraction"], 0.0)
    np.testing.assert_array_equal(
        component.spec.initial_fields["land_surface_temperature"], 280.0
    )
    assert component.spec.transfer.time_selection == "current"
