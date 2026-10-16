"""Native-grid SPEEDY phase diagnosis, model-level fields, and derivatives."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.assertions import assert_allclose_compact
from vercor.dtypes import DTypePolicy
from vercor.grids import RectilinearGrid
from vercor.physics import PhysicalConstants
from vercor.regridding import bilinear
from vercor.setups._external.jax_gcm_fields import map_jcm_output_fields


def _map_fields(
    temperature_2m: jax.Array,
    *,
    precls: jax.Array | None = None,
    precnv: jax.Array | None = None,
    constants: PhysicalConstants | None = None,
    dtype: DTypePolicy | None = None,
) -> dict[str, jax.Array]:
    """Keep a warm bottom level distinct from native near-surface air."""

    constants = PhysicalConstants() if constants is None else constants
    dtype = DTypePolicy.from_jax_config() if dtype is None else dtype
    shape = temperature_2m.shape
    profile = jnp.full((2, *shape), 290.0)
    return cast(
        dict[str, jax.Array],
        map_jcm_output_fields(
            2.5e6,
            1e5,
            jnp.asarray([0.2, 0.95]),
            28.966,
            8314.47,
            1e5,
            0.286,
            jnp.zeros((*shape, 3)),
            jnp.zeros((*shape, 3)),
            jnp.zeros(shape),
            jnp.zeros(shape),
            jnp.ones(shape),
            jnp.ones_like(profile),
            jnp.full_like(profile, 2.0),
            profile,
            jnp.full_like(profile, 10.0),
            large_scale_precipitation=(
                jnp.full(shape, 8.0) if precls is None else precls
            ),
            convective_precipitation=jnp.full(shape, 2.0) if precnv is None else precnv,
            temperature_2m=temperature_2m,
            snow_temperature=constants.precipitation_snow_temperature,
            rain_temperature=constants.precipitation_rain_temperature,
            dtype=dtype,
        ),
    )


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True])
def test_phase_split_uses_calibrated_lt_midpoint_and_slope(enable_x64: bool) -> None:
    """Catch wrong units, threshold order, sign, slope, or bottom-level air."""

    policy = DTypePolicy(enable_x64=enable_x64)
    temperature = jnp.asarray(
        [[272.77, 272.77], [275.46, 275.46], [278.15, 278.15]],
        dtype=policy.jax_real,
    )
    fields = _map_fields(temperature, dtype=policy)
    expected_snow = np.tile(
        [0.008807970779778824, 0.005, 0.0011920292202211755], (2, 1)
    )
    tolerance = 2e-5 if not enable_x64 else 1e-12
    assert fields["snow_rate"].dtype == policy.jax_real
    assert fields["rain_rate"].dtype == policy.jax_real
    assert_allclose_compact(fields["snow_rate"], expected_snow, rtol=tolerance)
    assert_allclose_compact(fields["rain_rate"], 0.01 - expected_snow, rtol=tolerance)
    assert_allclose_compact(fields["temperature"], 290.0)
    assert_allclose_compact(fields["specific_humidity"], 0.01)
    assert_allclose_compact(fields["u_velocity"], 1.0)
    assert_allclose_compact(fields["v_velocity"], 2.0)
    assert_allclose_compact(fields["density"], 1.141246774474563, rtol=1e-6)


@pytest.mark.fast_always
@pytest.mark.parametrize("enable_x64", [False, True])
def test_phase_split_conserves_both_processes_including_dry_and_extreme_cells(
    enable_x64: bool,
) -> None:
    """Catch dropped precipitation, gram conversion, transpose, and overflow."""

    policy = DTypePolicy(enable_x64=enable_x64)
    temperature = jnp.asarray(
        [[150.0, 400.0], [275.46, 272.77], [278.15, 275.46]],
        dtype=policy.jax_real,
    )
    precls = jnp.asarray([[8.0, 18.0], [0.0, 28.0], [38.0, 0.0]])
    precnv = jnp.asarray([[2.0, 2.0], [0.0, 2.0], [2.0, 0.0]])
    fields = _map_fields(temperature, precls=precls, precnv=precnv, dtype=policy)
    rain, snow = fields["rain_rate"], fields["snow_rate"]
    assert bool(jnp.all(jnp.isfinite(rain) & jnp.isfinite(snow)))
    assert bool(jnp.all((rain >= 0.0) & (snow >= 0.0)))
    assert_allclose_compact(
        rain + snow, np.asarray([[0.01, 0.0, 0.04], [0.02, 0.03, 0.0]])
    )

    def snow_sum(values: jax.Array) -> jax.Array:
        return jnp.sum(
            _map_fields(values, precls=precls, precnv=precnv, dtype=policy)["snow_rate"]
        )

    _, tangent = jax.jvp(snow_sum, (temperature,), (jnp.ones_like(temperature),))
    assert bool(jnp.isfinite(tangent))
    assert bool(jnp.all(jnp.isfinite(jax.grad(snow_sum)(temperature))))


@pytest.mark.fast_always
def test_phase_split_temperature_jvp_vjp_and_finite_difference_agree() -> None:
    """Catch detached temperature, zero gradients, and incorrect derivatives."""

    def phases(temperature: jax.Array) -> jax.Array:
        fields = _map_fields(temperature)
        return jnp.stack((fields["rain_rate"], fields["snow_rate"]))

    temperature = jnp.asarray([[250.0, 272.77], [275.46, 278.15], [300.0, 320.0]])
    direction = jnp.ones_like(temperature)
    values, tangent = jax.jvp(jax.jit(phases), (temperature,), (direction,))
    _, pullback = jax.vjp(phases, temperature)
    weights = jnp.zeros_like(values).at[1].set(1.0)
    gradient = pullback(weights)[0]
    step = 1e-3
    finite_difference = (phases(temperature + step) - phases(temperature - step)) / (
        2 * step
    )
    assert_allclose_compact(tangent, finite_difference, rtol=1e-5, atol=1e-13)
    assert_allclose_compact(gradient.T, tangent[1], atol=1e-18)
    assert_allclose_compact(tangent[1, 0, 1], -0.0018587360594795538)
    assert_allclose_compact(tangent.sum(axis=0), 0.0, atol=1e-18)
    assert bool(jnp.all(jnp.isfinite(gradient)))


@pytest.mark.fast_always
def test_phase_thresholds_remain_traced_physical_parameters() -> None:
    """Catch static thresholds or loss of one of the two calibrated parameters."""

    def objective(thresholds: jax.Array) -> jax.Array:
        constants = replace(
            PhysicalConstants(),
            precipitation_snow_temperature=thresholds[0],
            precipitation_rain_temperature=thresholds[1],
        )
        return jnp.sum(
            _map_fields(jnp.asarray([[276.0]]), constants=constants)["snow_rate"]
        )

    thresholds = jnp.asarray([272.77, 278.15])
    _, tangent = jax.jvp(objective, (thresholds,), (jnp.ones(2),))
    gradient = jax.grad(objective)(thresholds)
    step = 1e-3
    differences = jnp.stack(
        [
            (objective(thresholds + step * axis) - objective(thresholds - step * axis))
            / (2 * step)
            for axis in jnp.eye(2)
        ]
    )
    assert bool(jnp.all(gradient > 0.0))
    assert_allclose_compact(gradient, differences, rtol=1e-7)
    assert_allclose_compact(tangent, jnp.sum(gradient))


@pytest.mark.fast_always
def test_phase_split_precedes_bilinear_exchange() -> None:
    """Catch partitioning interpolated precipitation instead of native cells."""

    native_temperature = jnp.asarray([[272.77, 272.77], [278.15, 278.15]])
    fields = _map_fields(
        native_temperature,
        precls=jnp.asarray([[8.0, 8.0], [28.0, 28.0]]),
    )
    source = RectilinearGrid(
        name="ATM",
        longitude=jnp.asarray([0.0, 20.0]),
        latitude=jnp.asarray([-10.0, 10.0]),
    )
    target = RectilinearGrid(
        name="OCN",
        longitude=jnp.asarray([10.0, 15.0]),
        latitude=jnp.asarray([-5.0, 5.0]),
    )
    transfer = bilinear(source, target, periodic_longitude=False)
    rain = jnp.asarray(transfer.regrid(fields["rain_rate"]))
    snow = jnp.asarray(transfer.regrid(fields["snow_rate"]))
    assert_allclose_compact(snow[:, 0], 0.0061920292202211755)
    assert_allclose_compact(rain[:, 0], 0.013807970779778825)
    assert_allclose_compact(rain + snow, np.asarray([[0.02, 0.025], [0.02, 0.025]]))
