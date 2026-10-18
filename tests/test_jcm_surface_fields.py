"""Verify independently weighted marine contributions and distinct land forcing."""

from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vercor.dtypes import DTypePolicy

pytestmark = pytest.mark.fast_always


def prepare(
    fields: dict[str, Any],
    land: Any,
    water_albedo: Any = 0.1,
    *,
    dtype: Any = jnp.float64,
) -> Any:
    from vercor.setups._external import jax_gcm_fields

    candidate = getattr(jax_gcm_fields, "prepare_coupled_surface_forcing", None)
    assert callable(candidate), "ERROR missing independent JCM surface composition"
    return candidate(
        fields,
        land,
        jnp.asarray(water_albedo),
        dtype=DTypePolicy(enable_x64=dtype == jnp.float64),
    )


def fields_for(ocean: float = 0.75, ice: float = 0.3) -> dict[str, jax.Array]:
    return {
        name: jnp.full((2, 3), value)
        for name, value in {
            "land_surface_temperature": 300.0,
            "soil_moisture": 0.42,
            "open_water_temperature_contribution": (ocean - ice) * 290.0,
            "ice_surface_temperature": ice * 260.0,
            "ice_surface_albedo": ice * 0.7,
            "ice_fraction": ice,
        }.items()
    }


def test_fractional_coast_composes_independent_surfaces() -> None:
    fields = fields_for()
    surface = prepare(fields, jnp.full((2, 3), 0.25))
    np.testing.assert_allclose(surface.sea_surface_temperature, 278.0)
    np.testing.assert_allclose(surface.marine_surface_albedo, 0.34)
    np.testing.assert_allclose(surface.total_surface_temperature, 283.5)
    np.testing.assert_allclose(surface.land_surface_temperature, 300.0)
    np.testing.assert_allclose(surface.soil_moisture, 0.42)
    np.testing.assert_allclose(surface.marine_ice_fraction, 0.4)
    np.testing.assert_array_equal(fields["ice_fraction"], 0.3)
    leaves = jax.tree_util.tree_leaves(surface)
    assert len(leaves) == 6 and all(leaf.shape == (2, 3) for leaf in leaves)


@pytest.mark.parametrize(
    "ocean,ice,temperature,albedo",
    [(1.0, 0.0, 290.0, 0.1), (1.0, 1.0, 260.0, 0.7), (0.5, 0.5, 260.0, 0.7)],
)
def test_surface_limits_have_no_second_concentration_factor(
    ocean: float, ice: float, temperature: float, albedo: float
) -> None:
    surface = prepare(fields_for(ocean, ice), jnp.full((2, 3), 1 - ocean))
    np.testing.assert_allclose(surface.sea_surface_temperature, temperature)
    np.testing.assert_allclose(surface.marine_surface_albedo, albedo)
    np.testing.assert_allclose(surface.marine_ice_fraction, ice / ocean)
    np.testing.assert_allclose(
        surface.total_surface_temperature, (1 - ocean) * 300 + ocean * temperature
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_composed_fields_follow_runtime_precision(dtype: Any) -> None:
    surface = prepare(fields_for(), jnp.full((2, 3), 0.25), dtype=dtype)
    assert all(leaf.dtype == dtype for leaf in jax.tree_util.tree_leaves(surface))


def test_absent_surfaces_ignore_nan_operands_before_arithmetic() -> None:
    land = jnp.array([[1.0, 0.0, 0.25], [0.25, 0.0, 1.0]])
    ocean = 1 - land
    ice = jnp.array([[0.0, 1.0, 0.0], [0.3, 0.0, 0.0]])
    fields = {
        "land_surface_temperature": jnp.where(land > 0, 300.0, jnp.nan),
        "soil_moisture": jnp.where(land > 0, 0.42, jnp.nan),
        "ice_fraction": jnp.where(ocean > 0, ice, jnp.nan),
        "ice_surface_temperature": jnp.where(ice > 0, ice * 260, jnp.nan),
        "ice_surface_albedo": jnp.where(ice > 0, ice * 0.7, jnp.nan),
        "open_water_temperature_contribution": jnp.where(
            ocean - ice > 0, (ocean - ice) * 290, jnp.nan
        ),
    }
    water_albedo = jnp.where(ocean - ice > 0, 0.1, jnp.nan)
    surface = prepare(fields, land, water_albedo)
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(surface)
    )
    expected = land * 300 + (ocean - ice) * 290 + ice * 260
    np.testing.assert_allclose(surface.total_surface_temperature, expected)
    np.testing.assert_allclose(jnp.where(land == 0, surface.soil_moisture, 0), 0)

    def albedo_loss(alpha: jax.Array) -> jax.Array:
        updated = {
            **fields,
            "ice_surface_albedo": jnp.where(ice > 0, ice * alpha, jnp.nan),
        }
        return cast(
            jax.Array, prepare(updated, land, water_albedo).marine_surface_albedo.sum()
        )

    point = jnp.array(0.7)
    _, forward = jax.jvp(jax.jit(albedo_loss), (point,), (jnp.ones_like(point),))
    reverse = jax.grad(albedo_loss)(point)
    np.testing.assert_allclose([forward, reverse], 1.4, rtol=1e-13)
    np.testing.assert_allclose(
        reverse,
        (albedo_loss(point + 1e-4) - albedo_loss(point - 1e-4)) / 2e-4,
        rtol=1e-9,
    )


def test_existing_static_coverage_thresholds_remain_finite() -> None:
    from vercor.grid_masks import compute_ocn_lnd_masks_on_atm_grid

    raw = jnp.array([[0.000999, 0.001, 0.999, 0.999001]])

    class Remap:
        def regrid(self, _: Any) -> Any:
            return raw

    ocean, land, _ = compute_ocn_lnd_masks_on_atm_grid(
        jnp.ones((1, 4)), cast(Any, Remap())
    )
    fields = {
        "land_surface_temperature": jnp.full(raw.shape, 300.0),
        "soil_moisture": jnp.full(raw.shape, 0.42),
        "ice_fraction": raw * 0.5,
        "ice_surface_temperature": raw * 0.5 * 260,
        "ice_surface_albedo": raw * 0.5 * 0.7,
        "open_water_temperature_contribution": raw * 0.5 * 290,
    }
    surface = prepare(fields, land)
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(surface)
    )
    np.testing.assert_allclose(ocean, [[0.0, 0.001, 0.999, 1.0]])
    assert float(surface.marine_ice_fraction[0, 0]) == 0
    assert bool(
        jnp.all((surface.marine_ice_fraction >= 0) & (surface.marine_ice_fraction <= 1))
    )
    np.testing.assert_allclose(surface.sea_surface_temperature[0, 1:3], 275.0)
    np.testing.assert_array_equal(fields["ice_fraction"], raw * 0.5)


def test_surface_jvp_vjp_matches_independent_area_derivatives_and_fd() -> None:
    def loss(parameters: jax.Array) -> jax.Array:
        water, ice_temperature, albedo, land_temperature, soil, ice = parameters
        fields = {
            "land_surface_temperature": jnp.full((1, 1), land_temperature),
            "soil_moisture": jnp.full((1, 1), soil),
            "ice_fraction": jnp.full((1, 1), ice),
            "ice_surface_temperature": jnp.full((1, 1), ice * ice_temperature),
            "ice_surface_albedo": jnp.full((1, 1), ice * albedo),
            "open_water_temperature_contribution": jnp.full(
                (1, 1), (0.75 - ice) * water
            ),
        }
        surface = prepare(fields, jnp.full((1, 1), 0.25))
        return cast(
            jax.Array,
            (
                surface.total_surface_temperature
                + 100 * surface.marine_surface_albedo
                + surface.soil_moisture
            ).sum(),
        )

    point = jnp.array([290.0, 260.0, 0.7, 300.0, 0.42, 0.3])
    gradient = jax.grad(loss)(point)
    np.testing.assert_allclose(gradient, [0.45, 0.3, 40.0, 0.25, 1.0, 50.0], rtol=1e-13)
    _, forward = jax.jvp(jax.jit(loss), (point,), (jnp.ones_like(point),))
    np.testing.assert_allclose(forward, gradient.sum(), rtol=1e-13)
    directions = jnp.eye(6) * 1e-4
    finite = jnp.array([(loss(point + d) - loss(point - d)) / 2e-4 for d in directions])
    np.testing.assert_allclose(gradient, finite, rtol=1e-8)


@pytest.mark.parametrize("ice", [0.0, 1.0])
def test_finite_surface_endpoint_gradients_match_one_sided_mixture(ice: float) -> None:
    """Retain the limiting derivatives of already weighted finite contributions."""

    def mixture(area: jax.Array) -> jax.Array:
        values = {
            "land_surface_temperature": jnp.full((1, 1), jnp.nan),
            "soil_moisture": jnp.full((1, 1), jnp.nan),
            "ice_fraction": jnp.full((1, 1), area),
            "ice_surface_temperature": jnp.full((1, 1), area * 260),
            "ice_surface_albedo": jnp.full((1, 1), area * 0.7),
            "open_water_temperature_contribution": jnp.full((1, 1), (1 - area) * 290),
        }
        result = prepare(values, jnp.zeros((1, 1)), jnp.array(0.1))
        return jnp.stack(
            (result.sea_surface_temperature[0, 0], result.marine_surface_albedo[0, 0])
        )

    point = jnp.array(ice)
    _, forward = jax.jvp(jax.jit(mixture), (point,), (jnp.ones_like(point),))
    reverse = jax.jacrev(mixture)(point)
    direction = 1 if ice == 0 else -1
    finite = (mixture(point + direction * 1e-5) - mixture(point)) / (direction * 1e-5)
    np.testing.assert_allclose(forward, [-30.0, 0.6], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(finite, forward, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("raw_ocean", [0.001, 0.00101, 0.0011, 0.0012])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("missing_water", [False, True])
def test_full_ice_coast_accepts_roundoff_from_fractional_land_complement(
    raw_ocean: float,
    dtype: Any,
    missing_water: bool,
) -> None:
    """R(W*A) and 1-(1-R(W)) must share full-ice support despite cancellation."""
    from vercor.grid_masks import compute_ocn_lnd_masks_on_atm_grid

    raw = jnp.full((1, 1), raw_ocean, dtype=dtype)

    class Remap:
        def regrid(self, _: Any) -> Any:
            return raw

    _, land, _ = compute_ocn_lnd_masks_on_atm_grid(jnp.ones((1, 1)), cast(Any, Remap()))

    def surface(alpha: jax.Array) -> Any:
        fields = {
            "land_surface_temperature": jnp.full((1, 1), 300.0),
            "soil_moisture": jnp.full((1, 1), 0.42),
            "ice_fraction": raw,
            "ice_surface_temperature": raw * 260,
            "ice_surface_albedo": raw * alpha,
            "open_water_temperature_contribution": jnp.full(
                (1, 1), jnp.nan if missing_water else 0.0
            ),
        }
        return prepare(fields, land, jnp.array(0.1), dtype=dtype)

    point = jnp.array(0.7, dtype=dtype)
    result, forward = jax.jvp(jax.jit(surface), (point,), (jnp.ones_like(point),))
    reverse = jax.grad(lambda alpha: surface(alpha).marine_surface_albedo.sum())(point)
    assert bool(
        jnp.all((result.marine_ice_fraction >= 0) & (result.marine_ice_fraction <= 1))
    )
    assert all(
        bool(jnp.all(jnp.isfinite(x)))
        for x in jax.tree.leaves((result, forward, reverse))
    )
    tolerance = 1e-4 if dtype == jnp.float32 else 1e-12
    np.testing.assert_allclose(result.sea_surface_temperature, 260.0, rtol=tolerance)
    np.testing.assert_allclose(result.marine_surface_albedo, 0.7, rtol=tolerance)
    np.testing.assert_allclose(result.marine_ice_fraction, 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(forward.marine_surface_albedo, reverse, rtol=tolerance)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_roundoff_reconciliation_preserves_finite_coastal_endpoint_derivatives(
    dtype: Any,
) -> None:
    raw = jnp.array(0.0011, dtype=dtype)
    land = jnp.full((1, 1), 1 - raw, dtype=dtype)

    def mixture(area: jax.Array) -> jax.Array:
        fields = {
            "land_surface_temperature": jnp.full((1, 1), 300.0),
            "soil_moisture": jnp.full((1, 1), 0.42),
            "ice_fraction": jnp.full((1, 1), raw * area),
            "ice_surface_temperature": jnp.full((1, 1), raw * area * 260),
            "ice_surface_albedo": jnp.full((1, 1), raw * area * 0.7),
            "open_water_temperature_contribution": jnp.full(
                (1, 1), raw * (1 - area) * 290
            ),
        }
        result = prepare(fields, land, jnp.array(0.1), dtype=dtype)
        return jnp.stack(
            (
                result.sea_surface_temperature[0, 0],
                result.marine_surface_albedo[0, 0],
                result.marine_ice_fraction[0, 0],
            )
        )

    point = jnp.array(1.0, dtype=dtype)
    _, forward = jax.jvp(jax.jit(mixture), (point,), (jnp.ones_like(point),))
    reverse = jax.jacrev(mixture)(point)
    scale = raw / (1 - land[0, 0])
    np.testing.assert_allclose(
        forward, jnp.array([-30.0, 0.6, 1.0]) * scale, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(forward, reverse, rtol=1e-5, atol=1e-6)
    h = 1e-2 if dtype == jnp.float32 else 1e-4
    finite = (mixture(point) - mixture(point - h)) / h
    np.testing.assert_allclose(
        finite, forward, rtol=5e-3 if dtype == jnp.float32 else 1e-8, atol=1e-6
    )


def test_roundoff_reconciliation_rejects_materially_excessive_ice() -> None:
    fields = fields_for(ocean=0.0011, ice=0.0012)
    with pytest.raises(Exception, match="ocean-area bounds"):
        result = prepare(fields, jnp.full((2, 3), 1 - 0.0011))
        jax.block_until_ready(result)
