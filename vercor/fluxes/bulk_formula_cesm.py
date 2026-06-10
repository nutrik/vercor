import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.utilities import cdn, psimhu, psixhu, qsat
from vercor.settings import VercorSettings


def _as_jax_array(value: ArrayLike) -> jax.Array:
    return as_jax_real_array(value)


def _compute_stability_terms(
    settings: VercorSettings,
    zbot: jax.Array,
    thref: jax.Array,
    qbot: jax.Array,
    ustar: jax.Array,
    tstar: jax.Array,
    qstar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    hol = (
        settings.karman
        * settings.gravity
        * zbot
        * (tstar / thref + qstar / (1.0 / settings.zvir + qbot))
        / ustar**2
    )
    hol = jnp.minimum(jnp.abs(hol), 10.0) * jnp.sign(hol)
    stable = 0.5 + 0.5 * jnp.sign(hol)
    xsq = jnp.maximum(jnp.sqrt(jnp.abs(1.0 - 16.0 * hol)), 1.0)
    xqq = jnp.sqrt(xsq)
    psimh = -5.0 * hol * stable + (1.0 - stable) * psimhu(xqq)
    psixh = -5.0 * hol * stable + (1.0 - stable) * psixhu(xqq)
    return (hol, stable, psimh, psixh)


def _iterate_ocean_exchange(
    settings: VercorSettings,
    zbot: jax.Array,
    thbot: jax.Array,
    qbot: jax.Array,
    vmag: jax.Array,
    delt: jax.Array,
    delq: jax.Array,
    alz: jax.Array,
    rdn: jax.Array,
    ustar: jax.Array,
    tstar: jax.Array,
    qstar: jax.Array,
) -> tuple[
    jax.Array,
    ...,
]:
    hol, stable, psimh, psixh = _compute_stability_terms(
        settings, zbot, thbot, qbot, ustar, tstar, qstar
    )
    rd = rdn / (1.0 + rdn / settings.karman * (alz - psimh))
    u10n = vmag * rd / rdn

    next_rdn = jnp.sqrt(cdn(u10n))
    next_ren = jnp.full_like(next_rdn, 0.0346)
    next_rhn = (1.0 - stable) * 0.0327 + stable * 0.018

    next_rd = next_rdn / (1.0 + next_rdn / settings.karman * (alz - psimh))
    next_rh = next_rhn / (1.0 + next_rhn / settings.karman * (alz - psixh))
    next_re = next_ren / (1.0 + next_ren / settings.karman * (alz - psixh))

    next_ustar = next_rd * vmag
    next_tstar = next_rh * delt
    next_qstar = next_re * delq

    return (
        next_rdn,
        next_rd,
        next_rh,
        next_re,
        hol,
        stable,
        psixh,
        u10n,
        next_ustar,
        next_tstar,
        next_qstar,
    )


def compute_ocean_surface_fluxes(
    settings: VercorSettings,
    mask: ArrayLike,
    zbot: ArrayLike,
    ubot: ArrayLike,
    vbot: ArrayLike,
    thbot: ArrayLike,
    qbot: ArrayLike,
    rbot: ArrayLike,
    tbot: ArrayLike,
    us: ArrayLike,
    vs: ArrayLike,
    ts: ArrayLike,
    use_coldair_outbreak_mod: bool = False,
    missval: float = 0.0,
) -> tuple[jax.Array, ...]:
    """Compute atmosphere-ocean surface fluxes with JAX-compatible iteration control."""

    _ = missval

    mask_array = _as_jax_array(mask)
    zbot_array = _as_jax_array(zbot)
    ubot_array = _as_jax_array(ubot)
    vbot_array = _as_jax_array(vbot)
    thbot_array = _as_jax_array(thbot)
    qbot_array = _as_jax_array(qbot)
    rbot_array = _as_jax_array(rbot)
    tbot_array = _as_jax_array(tbot)
    us_array = _as_jax_array(us)
    vs_array = _as_jax_array(vs)
    ts_array = _as_jax_array(ts)

    zref = 10.0
    ztref = 2.0
    maxscl = 2.0
    td0 = -10.0
    alpha = 1.4
    flux_con_tol = 0.0
    flux_con_max_iter = 2

    tdiff = tbot_array - ts_array
    al2 = jnp.log(zref / ztref)

    vmag = jnp.maximum(
        settings.umin_ocean,
        jnp.sqrt((ubot_array - us_array) ** 2 + (vbot_array - vs_array) ** 2),
    )
    coldair_outbreak_mask = tdiff < td0
    vscl = jnp.minimum(
        1.0 + alpha * (jnp.abs(tdiff - td0) ** 0.5 / jnp.abs(vmag)),
        maxscl,
    )
    use_coldair = jnp.asarray(use_coldair_outbreak_mod)
    vmag = jnp.where(use_coldair & coldair_outbreak_mask, vmag * vscl, vmag)

    ssq = 0.98 * qsat(ts_array) / rbot_array
    delt = thbot_array - ts_array
    delq = qbot_array - ssq
    alz = jnp.log(zbot_array / zref)
    cp = settings.cpdair * (1.0 + settings.cpvir * ssq)

    stable0 = 0.5 + 0.5 * jnp.sign(delt)
    rdn0 = jnp.sqrt(cdn(vmag))
    rhn0 = (1.0 - stable0) * 0.0327 + stable0 * 0.018
    ren0 = jnp.full_like(rdn0, 0.0346)
    ustar0 = rdn0 * vmag
    tstar0 = rhn0 * delt
    qstar0 = ren0 * delq

    init_carry = (
        ustar0 * 2.0,
        rdn0,
        rdn0,
        rhn0,
        ren0,
        jnp.zeros_like(mask_array),
        stable0,
        jnp.zeros_like(mask_array),
        jnp.zeros_like(mask_array),
        ustar0,
        tstar0,
        qstar0,
    )

    def body_fn(_: int, carry: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        (
            ustar_prev,
            rdn,
            rd,
            rh,
            re,
            hol,
            stable,
            psixh,
            u10n,
            ustar,
            tstar,
            qstar,
        ) = carry
        continue_iteration = jnp.any(
            jnp.abs((ustar - ustar_prev) / ustar) > flux_con_tol
        )
        (
            next_rdn,
            next_rd,
            next_rh,
            next_re,
            next_hol,
            next_stable,
            next_psixh,
            next_u10n,
            next_ustar,
            next_tstar,
            next_qstar,
        ) = _iterate_ocean_exchange(
            settings,
            zbot_array,
            thbot_array,
            qbot_array,
            vmag,
            delt,
            delq,
            alz,
            rdn,
            ustar,
            tstar,
            qstar,
        )
        return tuple(
            jnp.where(continue_iteration, new_value, current_value)
            for new_value, current_value in (
                (ustar, ustar_prev),
                (next_rdn, rdn),
                (next_rd, rd),
                (next_rh, rh),
                (next_re, re),
                (next_hol, hol),
                (next_stable, stable),
                (next_psixh, psixh),
                (next_u10n, u10n),
                (next_ustar, ustar),
                (next_tstar, tstar),
                (next_qstar, qstar),
            )
        )

    (
        _ustar_prev,
        _rdn,
        rd,
        rh,
        re,
        hol,
        stable,
        psixh,
        u10n,
        ustar,
        tstar,
        qstar,
    ) = lax.fori_loop(0, flux_con_max_iter, body_fn, init_carry)

    tau = rbot_array * ustar * ustar
    taux = tau * (ubot_array - us_array) / vmag * mask_array
    tauy = tau * (vbot_array - vs_array) / vmag * mask_array
    sen = cp * tau * tstar / ustar * mask_array
    lat = settings.latvap * tau * qstar / ustar * mask_array
    lwup = -settings.stefBoltz * ts_array**4 * mask_array
    evap = lat / settings.latvap * mask_array

    hol_2m = hol * ztref / zbot_array
    xsq = jnp.maximum(1.0, jnp.sqrt(jnp.abs(1.0 - 16.0 * hol_2m)))
    xqq = jnp.sqrt(xsq)
    psix2 = -5.0 * hol_2m * stable + (1.0 - stable) * psixhu(xqq)
    fac = (rh / settings.karman) * (alz + al2 - psixh + psix2)
    tref = (thbot_array - delt * fac - 0.01 * ztref) * mask_array
    fac = (re / settings.karman) * (alz + al2 - psixh + psix2)
    qref = qbot_array - delq * fac * mask_array
    duu10n = u10n * u10n * mask_array

    clha = rbot_array * settings.latvap * vmag * re * rd
    devdt = clha * ssq * 2.166847e-3 / (ts_array * ts_array)
    dflwupdt = -4.0 * settings.ocean_emissivity * settings.stefBoltz * ts_array**3
    dfshdt = -rbot_array * settings.cpdair * vmag * rh * rd
    dflhdt = -settings.latvap * devdt
    df0dt = dflwupdt + dfshdt + dflhdt * mask_array

    return (
        sen,
        lat,
        lwup,
        evap,
        taux,
        tauy,
        tref,
        qref,
        duu10n,
        ustar,
        tstar,
        qstar,
        df0dt,
    )


def shr_flux_atmIce(
    settings: VercorSettings,
    mask: ArrayLike,
    zbot: ArrayLike,
    ubot: ArrayLike,
    vbot: ArrayLike,
    thbot: ArrayLike,
    qbot: ArrayLike,
    rbot: ArrayLike,
    tbot: ArrayLike,
    ts: ArrayLike,
    missval: float = 0.0,
) -> tuple[jax.Array, ...]:
    """Compute atmosphere-sea-ice surface fluxes using JAX-native array math."""

    _ = missval
    _ = tbot

    mask_array = _as_jax_array(mask)
    zbot_array = _as_jax_array(zbot)
    ubot_array = _as_jax_array(ubot)
    vbot_array = _as_jax_array(vbot)
    thbot_array = _as_jax_array(thbot)
    qbot_array = _as_jax_array(qbot)
    rbot_array = _as_jax_array(rbot)
    ts_array = _as_jax_array(ts)

    zref = 10.0
    ztref = 2.0
    zzsice = 0.0005

    vmag = jnp.maximum(settings.umin_ice, jnp.sqrt(ubot_array**2 + vbot_array**2))
    thvbot = thbot_array * (1.0 + settings.zvir * qbot_array)
    ssq = qsat(ts_array) / rbot_array
    delt = thbot_array - ts_array
    delq = qbot_array - ssq
    alz = jnp.log(zbot_array / zref)
    cp = settings.cpdair * (1.0 + settings.cpvir * ssq)
    ltheat = settings.latvap + settings.latice

    rdn = jnp.full_like(vmag, settings.karman / jnp.log(zref / zzsice))
    rhn = rdn
    ren = rdn

    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq
    hol, stable, psimh, psixh = _compute_stability_terms(
        settings, zbot_array, thvbot, qbot_array, ustar, tstar, qstar
    )

    rd = rdn / (1.0 + rdn / settings.karman * (alz - psimh))
    rh = rhn / (1.0 + rhn / settings.karman * (alz - psixh))
    re = ren / (1.0 + ren / settings.karman * (alz - psixh))

    ustar = rd * vmag
    tstar = rh * delt
    qstar = re * delq
    hol, stable, psimh, psixh = _compute_stability_terms(
        settings, zbot_array, thvbot, qbot_array, ustar, tstar, qstar
    )

    rd = rdn / (1.0 + rdn / settings.karman * (alz - psimh))
    rh = rhn / (1.0 + rhn / settings.karman * (alz - psixh))
    re = ren / (1.0 + ren / settings.karman * (alz - psixh))

    ustar = rd * vmag
    tstar = rh * delt
    qstar = re * delq

    tau = rbot_array * ustar * ustar
    taux = tau * ubot_array / vmag * mask_array
    tauy = tau * vbot_array / vmag * mask_array
    sen = cp * tau * tstar / ustar * mask_array
    lat = ltheat * tau * qstar / ustar * mask_array
    lwup = -settings.stefBoltz * ts_array**4 * mask_array
    evap = lat / ltheat * mask_array

    bn = settings.karman / rdn
    bh = settings.karman / rh
    ln0 = jnp.log(1.0 + (ztref / zbot_array) * (jnp.exp(bn) - 1.0))
    ln3 = jnp.log(1.0 + (ztref / zbot_array) * (jnp.exp(bn - bh) - 1.0))
    fac = ((ln0 - ztref / zbot_array * (bn - bh)) / bh) * stable + (
        (ln0 - ln3) / bh
    ) * (1.0 - stable)
    fac = jnp.minimum(jnp.maximum(fac, 0.0), 1.0)

    tref = ts_array + (thbot_array - ts_array) * fac * mask_array
    qref = qbot_array - delq * fac * mask_array

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, ustar, tstar, qstar)
