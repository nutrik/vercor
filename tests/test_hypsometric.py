import jax
import jax.numpy as jnp
import pytest

from tests.assertions import assert_allclose_compact
from setups.external.jax_gcm import get_altitudes_sigma_levels


def _to_jax_numpy(x: jnp.ndarray) -> jnp.ndarray:
    """Convert numpy/jax arrays to a numpy.ndarray (without requiring jax)."""
    # JAX arrays have __array__ so jnp.asarray works.
    return jnp.asarray(x)


def _float_dtype_of(x: jnp.ndarray) -> jnp.dtype:
    """Return a floating dtype suitable for constructing reference arrays."""
    x_jnp = _to_jax_numpy(x)
    if jnp.issubdtype(x_jnp.dtype, jnp.floating):
        return x_jnp.dtype
    # fallback

    out: jnp.dtype = jnp.float32

    return out


def test_shapes_and_floating_dtype() -> None:
    nlev, nlat, nlon = 5, 3, 4
    T = jnp.full((nlev, nlat, nlon), 280.0, dtype=jnp.float32)
    p = jnp.linspace(100000.0, 50000.0, nlev, dtype=jnp.float32)[
        :, None, None
    ] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)
    q = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)

    z = get_altitudes_sigma_levels(T, p, q)

    z_jnp = _to_jax_numpy(z)
    assert z_jnp.shape == (nlev, nlat, nlon)
    assert jnp.issubdtype(z_jnp.dtype, jnp.floating)  # allow float32 or float64


def test_invalid_ndim_raises() -> None:
    T = jnp.zeros((5, 3), dtype=jnp.float32)  # not 3D
    p = jnp.zeros((5, 3, 4), dtype=jnp.float32)
    q = jnp.zeros((5, 3, 4), dtype=jnp.float32)

    with pytest.raises(ValueError):
        get_altitudes_sigma_levels(T, p, q)


def test_mismatched_shapes_raises() -> None:
    T = jnp.zeros((5, 3, 4), dtype=jnp.float32)
    p = jnp.zeros((5, 3, 4), dtype=jnp.float32)
    q = jnp.zeros((5, 3, 5), dtype=jnp.float32)  # mismatch

    with pytest.raises(ValueError):
        get_altitudes_sigma_levels(T, p, q)


def test_monotonic_pressure_gives_increasing_height() -> None:
    """
    With pressure decreasing with level (k increasing upward),
    heights should be non-decreasing with k.
    """
    nlev, nlat, nlon = 6, 2, 2
    T = jnp.full((nlev, nlat, nlon), 270.0, dtype=jnp.float32)
    p_1d = jnp.array([100000, 85000, 70000, 50000, 30000, 20000], dtype=jnp.float32)
    p = p_1d[:, None, None] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)
    q = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)

    z = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q))
    dz = jnp.diff(z, axis=0)

    # float32-friendly tolerance
    assert jnp.all(dz >= -1e-6)


def test_isothermal_dry_column_matches_analytic() -> None:
    """
    For dry, isothermal atmosphere:
      z(p) - z(p0) = (Rd*T/g) * ln(p0/p)
    Using float32-friendly tolerances.
    """
    g = 9.80665
    Rd = 287.05

    nlev, nlat, nlon = 5, 3, 2
    T0 = 280.0

    # use float32 to match many JAX defaults
    T = jnp.full((nlev, nlat, nlon), T0, dtype=jnp.float32)
    q = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)

    p_1d = jnp.array([100000, 90000, 80000, 70000, 60000], dtype=jnp.float32)
    p = p_1d[:, None, None] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)

    z0 = jnp.float32(123.0)
    z = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, z0=z0, g=g, Rd=Rd))

    dtype = _float_dtype_of(z)
    p0 = dtype.type(p_1d[0])
    factor = dtype.type(Rd * T0 / g)

    z_analytic_1d = z0 + factor * jnp.log(p0 / p_1d.astype(dtype))
    z_analytic = z_analytic_1d[:, None, None] * jnp.ones((1, nlat, nlon), dtype=dtype)

    # float32: ~1e-3 m absolute is already extremely strict; use a bit looser
    assert_allclose_compact(z, z_analytic, rtol=0.0, atol=2e-3)


def test_humidity_increases_thickness() -> None:
    """
    With same T and pressure thickness, adding humidity increases Tv and thus thickness.
    """
    nlev, nlat, nlon = 4, 2, 2
    T = jnp.full((nlev, nlat, nlon), 290.0, dtype=jnp.float32)

    p_1d = jnp.array([100000, 85000, 70000, 60000], dtype=jnp.float32)
    p = p_1d[:, None, None] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)

    q_dry = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)
    q_moist = jnp.full((nlev, nlat, nlon), 0.02, dtype=jnp.float32)

    z_dry = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q_dry))
    z_moist = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q_moist))

    thick_dry = z_dry[-1] - z_dry[0]
    thick_moist = z_moist[-1] - z_moist[0]

    assert jnp.all(thick_moist > thick_dry)


def test_specific_humidity_uses_exact_virtual_temperature_without_mixing_ratio_denominator() -> (
    None
):
    """Specific humidity q should not be treated as water-vapor mixing ratio."""

    g = 9.80665
    Rd = 287.05
    Rv = 461.5
    eps = Rv / Rd

    T = jnp.full((2, 1, 1), 300.0, dtype=jnp.float32)
    p = jnp.asarray([100000.0, 90000.0], dtype=jnp.float32)[:, None, None]
    q = jnp.full((2, 1, 1), 0.02, dtype=jnp.float32)

    z = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, g=g, Rd=Rd, Rv=Rv))

    expected_thickness = (
        (Rd / g) * 300.0 * (1.0 + (eps - 1.0) * 0.02) * jnp.log(p[0, 0, 0] / p[1, 0, 0])
    )
    assert_allclose_compact(z[1, 0, 0] - z[0, 0, 0], expected_thickness, atol=1e-4)


def test_z0_accepts_scalar_and_2d_and_3d() -> None:
    nlev, nlat, nlon = 3, 2, 2
    T = jnp.full((nlev, nlat, nlon), 280.0, dtype=jnp.float32)
    p_1d = jnp.array([100000, 80000, 60000], dtype=jnp.float32)
    p = p_1d[:, None, None] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)
    q = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)

    z_scalar = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, z0=jnp.float32(10.0)))
    assert_allclose_compact(z_scalar[0], 10.0, atol=1e-6)

    z0_2d = jnp.array([[0.0, 5.0], [10.0, 20.0]], dtype=jnp.float32)
    z_2d = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, z0=z0_2d))
    assert_allclose_compact(z_2d[0], z0_2d, atol=1e-6)

    z0_3d = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)
    z0_3d = z0_3d.at[0].set(z0_2d)
    z_3d = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, z0=z0_3d))
    assert_allclose_compact(z_3d[0], z0_2d, atol=1e-6)


def test_bad_z0_shape_raises() -> None:
    nlev, nlat, nlon = 3, 2, 2
    T = jnp.full((nlev, nlat, nlon), 280.0, dtype=jnp.float32)
    p = jnp.linspace(100000.0, 60000.0, nlev, dtype=jnp.float32)[
        :, None, None
    ] * jnp.ones((1, nlat, nlon), dtype=jnp.float32)
    q = jnp.zeros((nlev, nlat, nlon), dtype=jnp.float32)

    bad = jnp.zeros((nlat, nlon, 2), dtype=jnp.float32)
    with pytest.raises(ValueError):
        get_altitudes_sigma_levels(T, p, q, z0=bad)


def test_reconstruction_from_z_matches_pressure_ratio() -> None:
    """
    Consistency check in float32:
      ln(p[k-1]/p[k]) ≈ g*Δz / (Rd*Tv_bar)
    """
    g = 9.80665
    Rd = 287.05
    Rv = 461.5
    eps = Rv / Rd

    nlev, nlat, nlon = 5, 2, 3
    seed = 100
    key = jax.random.key(seed)
    # Keep the same random sample for all variables
    rng = jax.random.uniform(key)

    # float32 inputs
    T = (250.0 + 50.0 * rng * jax.random.uniform(key, (nlev, nlat, nlon))).astype(
        jnp.float32
    )
    q = (0.0 + 0.02 * rng * jax.random.uniform(key, (nlev, nlat, nlon))).astype(
        jnp.float32
    )

    p_surface = (
        100000.0 + 2000.0 * rng * jax.random.uniform(key, (nlat, nlon))
    ).astype(jnp.float32)
    ratios = jnp.array([1.0, 0.85, 0.70, 0.55, 0.40], dtype=jnp.float32)[:, None, None]
    p = ratios * p_surface[None, :, :]

    z = _to_jax_numpy(get_altitudes_sigma_levels(T, p, q, g=g, Rd=Rd, Rv=Rv)).astype(
        jnp.float64
    )
    # Use float64 for the reconstruction arithmetic in NumPy to reduce numerical noise
    T64 = T.astype(jnp.float64)
    q64 = q.astype(jnp.float64)
    p64 = p.astype(jnp.float64)

    Tv = T64 * (1.0 + (eps - 1.0) * q64)
    Tv_bar = 0.5 * (Tv[:-1] + Tv[1:])
    dz = jnp.diff(z, axis=0)

    log_pr_rec = (g * dz) / (Rd * Tv_bar)
    log_pr_true = jnp.log(p64[:-1] / p64[1:])

    # float32-derived z will not satisfy 1e-12; use realistic tolerance
    assert_allclose_compact(log_pr_rec, log_pr_true, rtol=0.0, atol=5e-4)
