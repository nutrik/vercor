from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from vercor.fluxes.bulk_formula_cesm import (
    compute_ocean_surface_fluxes,
    shr_flux_atmIce,
)
from vercor.fluxes.utilities import (
    cdn,
    compute_air_density,
    compute_potential_temperature,
    psimhu,
    psixhu,
    qsat,
    qsat_august_eqn,
)
from vercor.fluxes.vertical_coordinates import (
    compute_hybrid_pressure_levels,
    get_altitudes_hybrid_sigma_levels,
)
from vercor.settings import VercorSettings


def _ocean_state(shape: tuple[int, int] = (3, 4)) -> dict[str, np.ndarray]:
    """Representative near-surface state over ocean for bulk-flux tests."""

    return {
        "mask": np.ones(shape),
        "zbot": np.full(shape, 10.0),
        "ubot": np.full(shape, 8.0),
        "vbot": np.full(shape, 2.0),
        "thbot": np.full(shape, 299.0),
        "qbot": np.full(shape, 0.010),
        "rbot": np.full(shape, 1.2),
        "tbot": np.full(shape, 299.0),
        "us": np.full(shape, 0.5),
        "vs": np.zeros(shape),
        "ts": np.full(shape, 300.0),
    }


def _finite_difference_scalar_grad(
    fn: Callable[[float], jax.Array], x: float, eps: float = 1e-3
) -> float:
    upper = float(fn(x + eps))
    lower = float(fn(x - eps))
    return (upper - lower) / (2.0 * eps)


def test_qsat_is_positive_and_increases_with_temperature() -> None:
    tk = np.array([260.0, 280.0, 300.0])
    out = qsat(tk)

    assert np.all(out > 0.0)
    assert np.all(np.diff(out) > 0.0)


def test_qsat_august_eqn_behaves_physically_with_t_and_p() -> None:
    ps = np.full(3, 101_325.0)
    tk = np.array([270.0, 285.0, 300.0])
    out_t = qsat_august_eqn(ps, tk)

    assert np.all(np.diff(out_t) > 0.0)

    ps2 = np.array([120_000.0, 101_325.0, 90_000.0])
    tk2 = np.full(3, 290.0)
    out_p = qsat_august_eqn(ps2, tk2)

    assert out_p[0] < out_p[1] < out_p[2]


def test_cdn_and_stability_functions_are_well_behaved() -> None:
    umps = np.array([2.0, 8.0, 15.0])
    drag = cdn(umps)
    assert np.all(drag > 0.0)

    xd = np.array([1.0, 2.0, 4.0])
    psim = psimhu(xd)
    psix = psixhu(xd)

    assert np.isclose(psix[0], 0.0)
    assert np.all(np.diff(psix) > 0.0)
    assert np.all(np.diff(psim) > 0.0)


def test_compute_hybrid_pressure_levels_matches_hybrid_definition() -> None:
    sp = np.array([[100_000.0, 95_000.0], [101_000.0, 99_000.0]])
    hya = np.array([100.0, 1_000.0, 5_000.0])
    hyb = np.array([0.0, 0.2, 0.8])

    ph = compute_hybrid_pressure_levels(sp=sp, hya=hya, hyb=hyb)

    assert ph.shape == (2, 2, 3)
    for k in range(3):
        assert_allclose_compact(ph[:, :, k], hya[k] + hyb[k] * sp)


def test_get_altitudes_hybrid_sigma_levels_returns_finite_increasing_profile() -> None:
    settings = VercorSettings()
    sp = np.full((2, 2), 101_325.0)
    hya = np.array([100.0, 1_000.0, 5_000.0, 10_000.0, 20_000.0])
    hyb = np.array([0.0, 0.1, 0.3, 0.5, 0.8])

    ph = compute_hybrid_pressure_levels(sp=sp, hya=hya, hyb=hyb)
    t = np.full((2, 2, 4), 260.0)
    q = np.full((2, 2, 4), 0.004)

    alt = get_altitudes_hybrid_sigma_levels(settings=settings, t=t, q=q, ph=ph)

    assert alt.shape == (2, 2, 4)
    assert np.all(np.isfinite(alt))
    assert np.all(alt > 0.0)
    assert np.all(np.diff(alt, axis=2) > 0.0)


def test_get_altitudes_hybrid_sigma_levels_handles_zero_top_half_level() -> None:
    settings = VercorSettings()
    ph = jnp.asarray([0.0, 1_000.0, 5_000.0, 100_000.0])[None, None, :]
    t = jnp.full((1, 1, 3), 260.0)
    q = jnp.zeros((1, 1, 3))

    alt = get_altitudes_hybrid_sigma_levels(settings=settings, t=t, q=q, ph=ph)

    top_down_dlog = jnp.asarray(
        [
            jnp.log(ph[0, 0, 1] / 0.1),
            jnp.log(ph[0, 0, 2] / ph[0, 0, 1]),
            jnp.log(ph[0, 0, 3] / ph[0, 0, 2]),
        ]
    )
    top_down_alpha = jnp.asarray(
        [
            jnp.log(2.0),
            1.0 - ph[0, 0, 1] / (ph[0, 0, 2] - ph[0, 0, 1]) * top_down_dlog[1],
            1.0 - ph[0, 0, 2] / (ph[0, 0, 3] - ph[0, 0, 2]) * top_down_dlog[2],
        ]
    )
    moist_temperature_rd = settings.rdair * 260.0
    expected_bottom_up_geopotential = jnp.asarray(
        [
            moist_temperature_rd * top_down_alpha[2],
            moist_temperature_rd * (top_down_dlog[2] + top_down_alpha[1]),
            moist_temperature_rd
            * (top_down_dlog[2] + top_down_dlog[1] + top_down_alpha[0]),
        ]
    )
    geopotential_height = expected_bottom_up_geopotential / settings.gravity
    expected_alt = (
        settings.earth_radius
        * geopotential_height
        / (settings.earth_radius - geopotential_height)
    )

    assert alt.shape == (1, 1, 3)
    assert np.all(np.isfinite(np.asarray(alt)))
    assert_allclose_compact(alt[0, 0, :], expected_alt, rtol=1e-6, atol=1e-6)


def test_density_and_potential_temperature_match_closed_form() -> None:
    settings = VercorSettings()
    pf = np.array([[100_000.0, 90_000.0]])
    t = np.array([[300.0, 280.0]])

    rho = compute_air_density(settings=settings, pf=pf, t=t)
    theta = compute_potential_temperature(settings=settings, tbot=t, pf=pf)

    expected_rho = settings.mwdair / settings.rgas * pf / t
    expected_theta = t * (settings.p0 / pf) ** settings.cappa

    assert_allclose_compact(rho, expected_rho)
    assert_allclose_compact(theta, expected_theta)


def test_flux_utility_kernels_support_jit() -> None:
    settings = VercorSettings()
    tk = jnp.asarray([260.0, 280.0, 300.0])
    ps = jnp.full(3, 101_325.0)
    sp = jnp.asarray([[100_000.0, 95_000.0], [101_000.0, 99_000.0]])
    hya = jnp.asarray([100.0, 1_000.0, 5_000.0])
    hyb = jnp.asarray([0.0, 0.2, 0.8])
    t = jnp.full((2, 2, 4), 260.0)
    q = jnp.full((2, 2, 4), 0.004)
    ph = jax.jit(compute_hybrid_pressure_levels)(
        sp,
        jnp.asarray([100.0, 1_000.0, 5_000.0, 10_000.0, 20_000.0]),
        jnp.asarray([0.0, 0.1, 0.3, 0.5, 0.8]),
    )

    assert_allclose_compact(jax.jit(qsat)(tk), qsat(tk))
    assert_allclose_compact(jax.jit(qsat_august_eqn)(ps, tk), qsat_august_eqn(ps, tk))
    assert_allclose_compact(
        jax.jit(compute_hybrid_pressure_levels)(sp, hya, hyb),
        compute_hybrid_pressure_levels(sp, hya, hyb),
    )
    assert_allclose_compact(
        jax.jit(
            lambda temp, humid, pressure: get_altitudes_hybrid_sigma_levels(
                settings, temp, humid, pressure
            )
        )(t, q, ph),
        get_altitudes_hybrid_sigma_levels(settings, t, q, ph),
    )
    assert_allclose_compact(
        jax.jit(cdn)(jnp.asarray([2.0, 8.0, 15.0])), cdn(jnp.asarray([2.0, 8.0, 15.0]))
    )
    assert_allclose_compact(
        jax.jit(psimhu)(jnp.asarray([1.0, 2.0, 4.0])),
        psimhu(jnp.asarray([1.0, 2.0, 4.0])),
    )
    assert_allclose_compact(
        jax.jit(psixhu)(jnp.asarray([1.0, 2.0, 4.0])),
        psixhu(jnp.asarray([1.0, 2.0, 4.0])),
    )
    assert_allclose_compact(
        jax.jit(lambda pf, temp: compute_air_density(settings, pf, temp))(
            jnp.asarray([[100_000.0, 90_000.0]]),
            jnp.asarray([[300.0, 280.0]]),
        ),
        compute_air_density(
            settings, np.array([[100_000.0, 90_000.0]]), np.array([[300.0, 280.0]])
        ),
    )
    assert_allclose_compact(
        jax.jit(lambda temp, pf: compute_potential_temperature(settings, temp, pf))(
            jnp.asarray([[300.0, 280.0]]),
            jnp.asarray([[100_000.0, 90_000.0]]),
        ),
        compute_potential_temperature(
            settings, np.array([[300.0, 280.0]]), np.array([[100_000.0, 90_000.0]])
        ),
    )


def test_compute_ocean_surface_fluxes_produces_finite_and_physically_consistent_signs() -> (
    None
):
    settings = VercorSettings()
    state = _ocean_state()

    sen, lat, lwup, evap, taux, tauy, *_ = compute_ocean_surface_fluxes(
        settings,
        state["mask"],
        state["zbot"],
        state["ubot"],
        state["vbot"],
        state["thbot"],
        state["qbot"],
        state["rbot"],
        state["tbot"],
        state["us"],
        state["vs"],
        state["ts"],
    )

    for arr in (sen, lat, lwup, evap, taux, tauy):
        assert np.all(np.isfinite(arr))

    # Sea warmer and moister than near-surface air: upward turbulent heat/moisture flux.
    assert np.mean(sen) < 0.0
    assert np.mean(lat) < 0.0
    assert np.mean(evap) < 0.0
    assert np.all(lwup < 0.0)

    assert np.mean(taux) > 0.0
    assert np.mean(tauy) > 0.0


def test_compute_ocean_surface_fluxes_matches_reference_state() -> None:
    settings = VercorSettings()
    state = _ocean_state()

    out = compute_ocean_surface_fluxes(
        settings,
        state["mask"],
        state["zbot"],
        state["ubot"],
        state["vbot"],
        state["thbot"],
        state["qbot"],
        state["rbot"],
        state["tbot"],
        state["us"],
        state["vs"],
        state["ts"],
    )
    expected_values = np.asarray(
        [
            -11.501394048985748,
            -332.8244500213005,
            -459.27,
            -0.0001330765493887647,
            0.08261268009451078,
            0.022030048025202875,
            299.069981902988,
            0.01106368351883935,
            65.17383383111927,
            0.2669262983914628,
            -0.03514565688237375,
            -0.00041545971737861595,
            -55.307247258870916,
        ]
    )

    assert_allclose_compact(
        np.asarray([np.asarray(arr)[0, 0] for arr in out]),
        expected_values,
    )


def test_compute_ocean_surface_fluxes_respects_mask_for_surface_exchange_outputs() -> (
    None
):
    settings = VercorSettings()
    state = _ocean_state(shape=(2, 3))
    state["mask"] = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    out = compute_ocean_surface_fluxes(
        settings,
        state["mask"],
        state["zbot"],
        state["ubot"],
        state["vbot"],
        state["thbot"],
        state["qbot"],
        state["rbot"],
        state["tbot"],
        state["us"],
        state["vs"],
        state["ts"],
    )

    # qref is not fully masked by the current implementation.
    for arr in (out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[8]):
        assert np.all(arr[state["mask"] == 0.0] == 0.0)


def test_flux_kernels_support_jit_and_gradients() -> None:
    settings = VercorSettings()
    state = _ocean_state(shape=(1, 1))
    mask = state["mask"]
    zbot = state["zbot"]
    ubot = state["ubot"]
    vbot = state["vbot"]
    thbot = state["thbot"]
    qbot = state["qbot"]
    rbot = state["rbot"]
    tbot = state["tbot"]
    us = state["us"]
    vs = state["vs"]

    jitted_ocean_flux = jax.jit(
        lambda ts: compute_ocean_surface_fluxes(
            settings,
            mask,
            zbot,
            ubot,
            vbot,
            thbot,
            qbot,
            rbot,
            tbot,
            us,
            vs,
            ts,
        )
    )
    jitted_ice_flux = jax.jit(
        lambda ts: shr_flux_atmIce(
            settings,
            mask,
            zbot,
            ubot,
            vbot,
            thbot,
            qbot,
            rbot,
            tbot,
            ts,
        )
    )

    ts = jnp.asarray([[300.0]])
    eager_ocean = compute_ocean_surface_fluxes(
        settings, mask, zbot, ubot, vbot, thbot, qbot, rbot, tbot, us, vs, ts
    )
    eager_ice = shr_flux_atmIce(
        settings, mask, zbot, ubot, vbot, thbot, qbot, rbot, tbot, ts
    )

    for eager_arr, jitted_arr in zip(eager_ocean, jitted_ocean_flux(ts)):
        assert_allclose_compact(jitted_arr, eager_arr)
    for eager_arr, jitted_arr in zip(eager_ice, jitted_ice_flux(ts)):
        assert_allclose_compact(jitted_arr, eager_arr)

    def scalar_sensible_heat(ts_scalar: float) -> jax.Array:
        ts_array = jnp.full((1, 1), ts_scalar)
        return jnp.sum(
            compute_ocean_surface_fluxes(
                settings,
                mask,
                zbot,
                ubot,
                vbot,
                thbot,
                qbot,
                rbot,
                tbot,
                us,
                vs,
                ts_array,
            )[0]
        )

    grad_value = float(jax.grad(scalar_sensible_heat)(300.0))
    finite_diff = _finite_difference_scalar_grad(scalar_sensible_heat, 300.0)
    assert np.isfinite(grad_value)
    assert np.isclose(grad_value, finite_diff, rtol=2e-2, atol=1e-3)


def test_cold_air_outbreak_mod_strengthens_flux_magnitudes() -> None:
    settings = VercorSettings()
    shape = (2, 3)
    mask = np.ones(shape)
    zbot = np.full(shape, 10.0)
    ubot = np.full(shape, 2.0)
    vbot = np.full(shape, 1.0)
    us = np.zeros(shape)
    vs = np.zeros(shape)
    ts = np.full(shape, 300.0)
    tbot = np.full(shape, 278.0)
    thbot = np.full(shape, 278.0)
    qbot = np.full(shape, 0.006)
    rbot = np.full(shape, 1.25)

    base = compute_ocean_surface_fluxes(
        settings,
        mask,
        zbot,
        ubot,
        vbot,
        thbot,
        qbot,
        rbot,
        tbot,
        us,
        vs,
        ts,
        use_coldair_outbreak_mod=False,
    )
    mod = compute_ocean_surface_fluxes(
        settings,
        mask,
        zbot,
        ubot,
        vbot,
        thbot,
        qbot,
        rbot,
        tbot,
        us,
        vs,
        ts,
        use_coldair_outbreak_mod=True,
    )

    assert np.mean(np.abs(mod[4])) > np.mean(np.abs(base[4]))
    assert np.mean(np.abs(mod[5])) > np.mean(np.abs(base[5]))
    assert np.mean(np.abs(mod[1])) > np.mean(np.abs(base[1]))
    assert np.mean(np.abs(mod[0])) > np.mean(np.abs(base[0]))


def test_shr_flux_atmIce_is_finite_and_masked_outputs_are_zeroed() -> None:
    settings = VercorSettings()
    mask = np.array([[1.0, 0.0], [1.0, 0.0]])
    shape = mask.shape

    zbot = np.full(shape, 10.0)
    ubot = np.full(shape, 6.0)
    vbot = np.full(shape, 1.0)
    thbot = np.full(shape, 266.0)
    qbot = np.full(shape, 0.004)
    rbot = np.full(shape, 1.3)
    tbot = np.full(shape, 266.0)
    ts = np.full(shape, 270.0)

    out = shr_flux_atmIce(
        settings,
        mask,
        zbot,
        ubot,
        vbot,
        thbot,
        qbot,
        rbot,
        tbot,
        ts,
    )

    for arr in out:
        assert np.all(np.isfinite(arr))

    # tref and qref are not fully masked by the current implementation.
    for arr in out[:6]:
        assert np.all(arr[mask == 0.0] == 0.0)

    # Upward longwave from the surface is negative with this sign convention.
    assert np.all(out[2][mask == 1.0] < 0.0)
