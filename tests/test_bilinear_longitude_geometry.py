"""Analytic longitude and mask-order checks for bilinear interpolation."""

from collections.abc import Callable
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor import RectilinearGrid
from vercor._interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.regridding import VectorRegridder, bilinear
from vercor.types import RuntimeArray

pytestmark = pytest.mark.fast_always


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("origin", [0.0, -720.0, 720.0])
@pytest.mark.parametrize("descending", [False, True])
def test_native_t31_seam_weights_preserve_precision_and_source_ad(
    enable_x64: bool, origin: float, descending: bool
) -> None:
    """The 356.25-to-360 interval gives weight2/3.75 at target358 degrees."""
    with jax.enable_x64(enable_x64):
        longitude = np.arange(96) * 3.75 + origin
        field = jnp.ones((2, 96)).at[:, 0].set(0.0)
        expected_gradient = np.zeros((2, 96))
        expected_gradient[:, -1] = 1.5 * (2.0 / 3.75)
        expected_gradient[:, 0] = 1.5 * (1.75 / 3.75)
        if descending:
            longitude = longitude[::-1].copy()
            field = field[:, ::-1]
            expected_gradient = expected_gradient[:, ::-1]
        target = RectilinearGrid(
            "target",
            longitude=np.array([-2.0, 358.0, 718.0]) + origin,
            latitude=np.array([0.0]),
        )
        # Public grids require ascending centers; the low-level interpolator
        # separately supports descending source storage.
        interp = BilinearRectilinearInterpolator(
            longitude, [-10.0, 10.0], target.longitude, target.latitude
        )
        apply = jax.jit(interp.apply_scalar)
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(apply(field), 2.0 / 3.75, rtol=0, atol=2 * eps)
        assert_allclose_compact(apply(jnp.ones_like(field)), 1.0, rtol=0, atol=2 * eps)

        def objective(values: jax.Array) -> jax.Array:
            return jnp.sum(apply(values))

        assert_allclose_compact(
            jax.grad(objective)(field), expected_gradient, rtol=0, atol=4 * eps
        )
        _, tangent = jax.jvp(objective, (field,), (jnp.ones_like(field),))
        assert_allclose_compact(tangent, 3.0, rtol=0, atol=4 * eps)


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("descending_lon", [False, True])
@pytest.mark.parametrize("descending_lat", [False, True])
def test_irregular_nonperiodic_degree_weights_reproduce_affine_field(
    enable_x64: bool, descending_lon: bool, descending_lat: bool
) -> None:
    """Degree differences preserve affine fields and clip both axis endpoints."""
    with jax.enable_x64(enable_x64):
        longitude = np.array([100.0, 101.5, 104.0])
        latitude = np.array([70.0, 71.25, 74.0])
        field = jnp.array([[0.0, 1.5, 4.0], [2.5, 4.0, 6.5], [8.0, 9.5, 12.0]])
        if descending_lon:
            longitude, field = longitude[::-1].copy(), field[:, ::-1]
        if descending_lat:
            latitude, field = latitude[::-1].copy(), field[::-1, :]
        interp = BilinearRectilinearInterpolator(
            longitude,
            latitude,
            [95.0, 100.75, 103.0, 110.0],
            [60.0, 70.625, 72.0, 80.0],
            periodic_longitude=False,
        )
        expected = np.array(
            [
                [0.0, 0.75, 3.0, 4.0],
                [1.25, 2.0, 4.25, 5.25],
                [4.0, 4.75, 7.0, 8.0],
                [8.0, 8.75, 11.0, 12.0],
            ]
        )
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            jax.jit(interp.apply_scalar)(field), expected, rtol=2 * eps, atol=2 * eps
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("origin", [0.0, -36000.0, 36000.0])
def test_periodic_vector_bases_use_equivalent_longitude_representations(
    enable_x64: bool, origin: float
) -> None:
    """Whole turns cannot rotate source vectors or spherical target bases."""
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source",
            longitude=np.array([0.0, 90.0, 180.0, 270.0]) + origin,
            latitude=np.array([-1.0, 1.0]),
        )
        target = RectilinearGrid(
            "target",
            longitude=np.array([-36000.0, 0.0, 36000.0]) + 45.0,
            latitude=np.array([0.0]),
        )
        remapper = bilinear(source, target)
        assert isinstance(remapper, VectorRegridder)
        field = jnp.array([[0.0, 1.0, 2.0, 3.0]] * 2)
        actual_u, actual_v = jax.jit(remapper.regrid_vector)(field, -2 * field)
        # Average Cartesian corner vectors, then project at (45 degrees, 0).
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(actual_u, np.sqrt(2.0) / 4.0, rtol=0, atol=4 * eps)
        assert_allclose_compact(
            actual_v, -np.cos(np.deg2rad(1.0)), rtol=0, atol=4 * eps
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("mode", ["nearest", "idw"])
@pytest.mark.parametrize("inactive_nan", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_masked_vector_extrapolation_orients_source_values_once(
    enable_x64: bool, mode: str, inactive_nan: bool, descending: bool
) -> None:
    """A valid eastmost source must retain its values, mask, and derivatives."""
    with jax.enable_x64(enable_x64):
        longitude = np.array([0.0, 1.0, 2.0, 3.0])
        support = np.zeros((2, 4), dtype=bool)
        support[0, 3] = True
        u = jnp.full((2, 4), jnp.nan if inactive_nan else -77.0).at[0, 3].set(30.0)
        v = jnp.full((2, 4), jnp.nan if inactive_nan else 42.0).at[0, 3].set(-3.0)
        gradient = np.zeros((2, 4))
        gradient[0, 3] = 1.0
        if descending:
            longitude = longitude[::-1].copy()
            support = support[:, ::-1].copy()
            u, v = u[:, ::-1], v[:, ::-1]
            gradient = gradient[:, ::-1]
        target = RectilinearGrid(
            "target",
            longitude=np.array([0.5, 1.5]),
            latitude=np.array([0.5]),
            binary_mask=np.array([[1, 0]]),
        )
        scalar_apply: Callable[[RuntimeArray], RuntimeArray]
        vector_apply: Callable[
            [RuntimeArray, RuntimeArray], tuple[RuntimeArray, RuntimeArray]
        ]
        if descending:
            interp = BilinearRectilinearInterpolator(
                longitude,
                [0.0, 1.0],
                target.longitude,
                target.latitude,
                src_mask=support,
                tgt_mask=target.binary_mask,
                periodic_longitude=False,
                extrapolation_mode=mode,
                idw_k=2,
                fill_value=-999.0,
            )
            scalar_apply = interp.apply_scalar
            vector_apply = interp.apply_vector
        else:
            source = RectilinearGrid(
                "source",
                longitude=longitude,
                latitude=np.array([0.0, 1.0]),
                binary_mask=support,
            )
            remapper = bilinear(
                source,
                target,
                periodic_longitude=False,
                extrapolation_mode=mode,
                idw_k=2,
                fill_value=-999.0,
            )
            assert isinstance(remapper, VectorRegridder)
            scalar_apply = remapper.regrid
            vector_apply = remapper.regrid_vector
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        scalar = jax.jit(scalar_apply)(u)
        actual_u, actual_v = jax.jit(vector_apply)(u, v)
        assert_allclose_compact(scalar, [[30.0, -999.0]], rtol=4 * eps, atol=0)
        assert_allclose_compact(actual_u, [[30.0, -999.0]], rtol=4 * eps, atol=0)
        assert_allclose_compact(actual_v, [[-3.0, -999.0]], rtol=4 * eps, atol=0)

        def objective(values: jax.Array) -> jax.Array:
            return jnp.sum(vector_apply(values, v)[0])

        assert_allclose_compact(jax.grad(objective)(u), gradient, rtol=4 * eps, atol=0)
        _, tangent = jax.jvp(objective, (u,), (jnp.ones_like(u),))
        assert_allclose_compact(tangent, 1.0, rtol=4 * eps, atol=0)


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("source_origin", [-180.0, -90.0])
@pytest.mark.parametrize("last_longitude", [90.0, 180.0])
def test_periodic_lookup_retains_in_range_target_precision(
    enable_x64: bool, source_origin: float, last_longitude: float
) -> None:
    """A distant source origin must not quantize a representable target point."""
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source",
            longitude=np.array([source_origin, 0.0, 0.001, last_longitude]),
            latitude=np.array([-1.0, 1.0]),
        )
        target = RectilinearGrid(
            "target", longitude=np.array([0.0003, 0.0007]), latitude=np.array([0.0])
        )
        remapper = bilinear(source, target)
        field = jnp.array([[-1.0, 0.0, 1.0, 2.0]] * 2)
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            jax.jit(remapper.regrid)(field), [[0.3, 0.7]], rtol=0, atol=2 * eps
        )
        derivative = jax.grad(lambda values: jnp.sum(remapper.regrid(values)))(field)
        assert_allclose_compact(
            derivative, np.array([[0.0, 0.5, 0.5, 0.0]] * 2), rtol=0, atol=2 * eps
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("last_longitude", [90.0, 180.0])
def test_wrapped_target_keeps_its_representable_subdegree_phase(
    enable_x64: bool,
    last_longitude: float,
) -> None:
    """Remove whole turns directly instead of subtracting a distant origin twice."""
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source",
            longitude=np.array([-180.0, 0.0, 0.001, last_longitude]),
            latitude=np.array([-1.0, 1.0]),
        )
        target = RectilinearGrid(
            "target", longitude=np.array([360.0003, 360.0007]), latitude=np.array([0.0])
        )
        # float32 input angles have phase10/32768 and23/32768 degrees.
        # float64 input angles have the literal phases reflected below.
        expected = (
            [[0.29999999998153726, 0.6999999999948159]]
            if enable_x64
            else [[0.30517578125, 0.701904296875]]
        )
        field = jnp.array([[-1.0, 0.0, 1.0, 2.0]] * 2)
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            jax.jit(bilinear(source, target).regrid)(field),
            expected,
            rtol=0,
            atol=2 * eps,
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("origin", [0.0, -4e-6])
@pytest.mark.parametrize("mode, expected", [("nearest", 10.0), ("idw", 6.5)])
def test_small_signed_longitudes_keep_distinct_extrapolation_distances(
    enable_x64: bool, origin: float, mode: str, expected: float
) -> None:
    """Canonicalization must retain representable donor separations near zero."""
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source",
            longitude=np.arange(5) * 1e-6 + origin,
            latitude=np.array([0.0, 1.0]),
            binary_mask=np.array([[1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]),
        )
        target = RectilinearGrid(
            "target", longitude=np.array([2.6e-6 + origin]), latitude=np.array([0.0])
        )
        remapper = bilinear(
            source,
            target,
            periodic_longitude=False,
            extrapolation_mode=mode,
            idw_k=2,
            idw_eps=1e-30,
        )
        field = jnp.array([[0.0, -77.0, -77.0, -77.0, 10.0], [-77.0] * 5])
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            jax.jit(remapper.regrid)(field), expected, rtol=4 * eps, atol=0
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("inclusive_last", [False, True])
def test_narrow_periodic_seam_preserves_representable_target_phase(
    enable_x64: bool,
    inclusive_last: bool,
) -> None:
    """Fine seam cells cannot encode tiny phases by adding a full revolution."""
    dtype = np.float64 if enable_x64 else np.float32
    first = dtype(2.6e-6)
    last = dtype(360) if inclusive_last else np.nextafter(dtype(360), dtype(0))
    targets = np.array([0.0, first / dtype(2), 360.0], dtype=dtype)
    # Exact rational geometry of the represented inputs, independent of lookup.
    gap = Fraction(360) - Fraction(float(last))
    width = gap + Fraction(float(first))
    expected = np.array(
        [
            [
                float((gap + Fraction(float(phase))) / width)
                for phase in (targets[0], targets[1], dtype(0))
            ]
        ]
    )
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source",
            longitude=np.array([first, 90, 180, 270, last], dtype=dtype),
            latitude=np.array([-1.0, 1.0]),
        )
        target = RectilinearGrid("target", longitude=targets, latitude=np.array([0.0]))
        field = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0]] * 2)
        eps = float(np.finfo(dtype).eps)
        assert_allclose_compact(
            jax.jit(bilinear(source, target).regrid)(field),
            expected,
            rtol=4 * eps,
            atol=0,
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("origin", [-720.0, 720.0])
@pytest.mark.parametrize("extent", [360.0, 480.0])
def test_inclusive_periodic_endpoints_retain_fine_aliased_phase(
    enable_x64: bool, origin: float, extent: float
) -> None:
    """A duplicate endpoint must not require a precise query to acquire a large offset."""
    dtype = np.float64 if enable_x64 else np.float32
    source_lon = np.array([origin, origin + 0.001, origin + extent], dtype=dtype)
    target_lon = np.array([360.0007], dtype=dtype)
    numerator = Fraction(float(target_lon[0])) - Fraction(360)
    denominator = Fraction(float(source_lon[1])) - Fraction(float(source_lon[0]))
    expected = float(numerator / denominator)
    with jax.enable_x64(enable_x64):
        source = RectilinearGrid(
            "source", longitude=source_lon, latitude=np.array([-1.0, 1.0])
        )
        target = RectilinearGrid(
            "target", longitude=target_lon, latitude=np.array([0.0])
        )
        field = jnp.array([[0.0, 1.0, 0.0]] * 2)
        assert_allclose_compact(
            jax.jit(bilinear(source, target).regrid)(field),
            expected,
            rtol=4 * float(np.finfo(dtype).eps),
            atol=0,
        )


@pytest.mark.parametrize("enable_x64", [False, True])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("extent", [360.0, 540.0])
def test_inclusive_and_broad_periodic_grids_retain_existing_endpoint_convention(
    enable_x64: bool, descending: bool, extent: float
) -> None:
    """Keep the established first-turn lookup for duplicate endpoints and broad grids."""
    with jax.enable_x64(enable_x64):
        longitude = np.array([0.0, 120.0, 240.0, 360.0])
        field = jnp.array([[1.0, 2.0, 3.0, 9.0]] * 2)
        expected = [[1.0, 6.0, 1.0, 6.0]]
        if extent > 360:
            longitude = np.array([0.0, 180.0, 360.0, 540.0])
            field = jnp.array([[1.0, 3.0, 9.0, 27.0]] * 2)
            expected = [[1.0, 7.0, 1.0, 7.0]]
        if descending:
            longitude, field = longitude[::-1].copy(), field[:, ::-1]
        interp = BilinearRectilinearInterpolator(
            longitude, [-1.0, 1.0], [0.0, 300.0, 360.0, 660.0], [0.0]
        )
        eps = float(np.finfo(np.float64 if enable_x64 else np.float32).eps)
        assert_allclose_compact(
            jax.jit(interp.apply_scalar)(field), expected, rtol=4 * eps, atol=0
        )


@pytest.mark.parametrize("enable_x64", [False, True])
def test_wrapped_source_centers_are_rejected_before_any_silent_seam_loss(
    enable_x64: bool,
) -> None:
    """Bilinear uses monotonic centers; the conservative wrapped-edge case fails early."""
    with jax.enable_x64(enable_x64):
        with pytest.raises(ValueError, match="lon_src must be strictly monotonic"):
            BilinearRectilinearInterpolator(
                [360.0, 4.0, 8.0, 12.0], [-10.0, 10.0], [0.0], [0.0]
            )
