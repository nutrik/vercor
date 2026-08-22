import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike
from typing import cast

from vercor._numerical_safety import require_active_finite
from vercor.dtypes import as_jax_real_array
from vercor.fluxes.utilities import cdn, psimhu, psixhu, qsat
from vercor.physics import PhysicalConstants


def _prepare_bulk_flux_operand(
    values: jax.Array,
    active_mask: jax.Array,
    *,
    owner: str,
    missing_value: float,
) -> jax.Array:
    """Validate one flux input and neutralize only masked missing-data NaNs."""

    require_active_finite(values, active_mask=active_mask, owner=owner)
    return jnp.where(
        jnp.isnan(values),
        jnp.asarray(missing_value, dtype=values.dtype),
        values,
    )


def _compute_stability_terms(
    constants: PhysicalConstants,
    zbot: jax.Array,
    thref: jax.Array,
    qbot: jax.Array,
    ustar: jax.Array,
    tstar: jax.Array,
    qstar: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    hol = (
        constants.von_karman_constant
        * constants.gravity
        * zbot
        * (
            tstar / thref
            + qstar / (1.0 / constants.water_vapor_mass_ratio_correction + qbot)
        )
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
    constants: PhysicalConstants,
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
        constants, zbot, thbot, qbot, ustar, tstar, qstar
    )
    rd = rdn / (1.0 + rdn / constants.von_karman_constant * (alz - psimh))
    u10n = vmag * rd / rdn

    next_rdn = jnp.sqrt(cdn(u10n))
    next_ren = jnp.full_like(next_rdn, 0.0346)
    next_rhn = (1.0 - stable) * 0.0327 + stable * 0.018

    next_rd = next_rdn / (
        1.0 + next_rdn / constants.von_karman_constant * (alz - psimh)
    )
    next_rh = next_rhn / (
        1.0 + next_rhn / constants.von_karman_constant * (alz - psixh)
    )
    next_re = next_ren / (
        1.0 + next_ren / constants.von_karman_constant * (alz - psixh)
    )

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
    constants: PhysicalConstants,
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

    mask_array = as_jax_real_array(mask)
    zbot_array = as_jax_real_array(zbot)
    ubot_array = as_jax_real_array(ubot)
    vbot_array = as_jax_real_array(vbot)
    thbot_array = as_jax_real_array(thbot)
    qbot_array = as_jax_real_array(qbot)
    rbot_array = as_jax_real_array(rbot)
    tbot_array = as_jax_real_array(tbot)
    us_array = as_jax_real_array(us)
    vs_array = as_jax_real_array(vs)
    ts_array = as_jax_real_array(ts)

    require_active_finite(
        mask_array,
        active_mask=None,
        owner="Ocean bulk-flux input 'mask'",
    )
    active_mask = mask_array > 0.0
    zbot_array = _prepare_bulk_flux_operand(
        zbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'zbot'",
        missing_value=1.0,
    )
    ubot_array = _prepare_bulk_flux_operand(
        ubot_array,
        active_mask,
        owner="Ocean bulk-flux input 'ubot'",
        missing_value=0.0,
    )
    vbot_array = _prepare_bulk_flux_operand(
        vbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'vbot'",
        missing_value=0.0,
    )
    thbot_array = _prepare_bulk_flux_operand(
        thbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'thbot'",
        missing_value=273.15,
    )
    qbot_array = _prepare_bulk_flux_operand(
        qbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'qbot'",
        missing_value=0.0,
    )
    rbot_array = _prepare_bulk_flux_operand(
        rbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'rbot'",
        missing_value=1.0,
    )
    tbot_array = _prepare_bulk_flux_operand(
        tbot_array,
        active_mask,
        owner="Ocean bulk-flux input 'tbot'",
        missing_value=273.15,
    )
    us_array = _prepare_bulk_flux_operand(
        us_array,
        active_mask,
        owner="Ocean bulk-flux input 'us'",
        missing_value=0.0,
    )
    vs_array = _prepare_bulk_flux_operand(
        vs_array,
        active_mask,
        owner="Ocean bulk-flux input 'vs'",
        missing_value=0.0,
    )
    ts_array = _prepare_bulk_flux_operand(
        ts_array,
        active_mask,
        owner="Ocean bulk-flux input 'ts'",
        missing_value=273.15,
    )

    zref = constants.reference_height
    ztref = constants.air_temperature_reference_height
    maxscl = 2.0
    td0 = -10.0
    alpha = 1.4
    flux_con_tol = 0.0
    flux_con_max_iter = 2

    tdiff = tbot_array - ts_array
    al2 = jnp.log(zref / ztref)

    vmag = jnp.maximum(
        constants.ocean_minimum_wind_speed,
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
    cp = constants.dry_air_specific_heat * (
        1.0 + constants.water_vapor_specific_heat_ratio_correction * ssq
    )

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
            constants,
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
    lat = constants.latent_heat_of_vaporization * tau * qstar / ustar * mask_array
    lwup = -constants.stefan_boltzmann_constant * ts_array**4 * mask_array
    evap = lat / constants.latent_heat_of_vaporization * mask_array

    hol_2m = hol * ztref / zbot_array
    xsq = jnp.maximum(1.0, jnp.sqrt(jnp.abs(1.0 - 16.0 * hol_2m)))
    xqq = jnp.sqrt(xsq)
    psix2 = -5.0 * hol_2m * stable + (1.0 - stable) * psixhu(xqq)
    fac = (rh / constants.von_karman_constant) * (alz + al2 - psixh + psix2)
    tref = (thbot_array - delt * fac - 0.01 * ztref) * mask_array
    fac = (re / constants.von_karman_constant) * (alz + al2 - psixh + psix2)
    qref = qbot_array - delq * fac * mask_array
    duu10n = u10n * u10n * mask_array

    clha = rbot_array * constants.latent_heat_of_vaporization * vmag * re * rd
    devdt = clha * ssq * 2.166847e-3 / (ts_array * ts_array)
    dflwupdt = (
        -4.0
        * constants.ocean_emissivity
        * constants.stefan_boltzmann_constant
        * ts_array**3
    )
    dfshdt = -rbot_array * constants.dry_air_specific_heat * vmag * rh * rd
    dflhdt = -constants.latent_heat_of_vaporization * devdt
    df0dt = dflwupdt + dfshdt + dflhdt * mask_array

    return cast(
        tuple[jax.Array, ...],
        (
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
        ),
    )


def shr_flux_atmIce(
    constants: PhysicalConstants,
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

    mask_array = as_jax_real_array(mask)
    zbot_array = as_jax_real_array(zbot)
    ubot_array = as_jax_real_array(ubot)
    vbot_array = as_jax_real_array(vbot)
    thbot_array = as_jax_real_array(thbot)
    qbot_array = as_jax_real_array(qbot)
    rbot_array = as_jax_real_array(rbot)
    tbot_array = as_jax_real_array(tbot)
    ts_array = as_jax_real_array(ts)

    require_active_finite(
        mask_array,
        active_mask=None,
        owner="Ice bulk-flux input 'mask'",
    )
    active_mask = mask_array > 0.0
    zbot_array = _prepare_bulk_flux_operand(
        zbot_array,
        active_mask,
        owner="Ice bulk-flux input 'zbot'",
        missing_value=1.0,
    )
    ubot_array = _prepare_bulk_flux_operand(
        ubot_array,
        active_mask,
        owner="Ice bulk-flux input 'ubot'",
        missing_value=0.0,
    )
    vbot_array = _prepare_bulk_flux_operand(
        vbot_array,
        active_mask,
        owner="Ice bulk-flux input 'vbot'",
        missing_value=0.0,
    )
    thbot_array = _prepare_bulk_flux_operand(
        thbot_array,
        active_mask,
        owner="Ice bulk-flux input 'thbot'",
        missing_value=273.15,
    )
    qbot_array = _prepare_bulk_flux_operand(
        qbot_array,
        active_mask,
        owner="Ice bulk-flux input 'qbot'",
        missing_value=0.0,
    )
    rbot_array = _prepare_bulk_flux_operand(
        rbot_array,
        active_mask,
        owner="Ice bulk-flux input 'rbot'",
        missing_value=1.0,
    )
    tbot_array = _prepare_bulk_flux_operand(
        tbot_array,
        active_mask,
        owner="Ice bulk-flux input 'tbot'",
        missing_value=273.15,
    )
    ts_array = _prepare_bulk_flux_operand(
        ts_array,
        active_mask,
        owner="Ice bulk-flux input 'ts'",
        missing_value=273.15,
    )
    _ = tbot_array

    zref = constants.reference_height
    ztref = constants.air_temperature_reference_height
    zzsice = 0.0005

    vmag = jnp.maximum(
        constants.ice_minimum_wind_speed,
        jnp.sqrt(ubot_array**2 + vbot_array**2),
    )
    thvbot = thbot_array * (
        1.0 + constants.water_vapor_mass_ratio_correction * qbot_array
    )
    ssq = qsat(ts_array) / rbot_array
    delt = thbot_array - ts_array
    delq = qbot_array - ssq
    alz = jnp.log(zbot_array / zref)
    cp = constants.dry_air_specific_heat * (
        1.0 + constants.water_vapor_specific_heat_ratio_correction * ssq
    )
    ltheat = constants.latent_heat_of_vaporization + constants.ice_latent_heat_of_fusion

    rdn = jnp.full_like(
        vmag,
        constants.von_karman_constant / jnp.log(zref / zzsice),
    )
    rhn = rdn
    ren = rdn

    ustar = rdn * vmag
    tstar = rhn * delt
    qstar = ren * delq
    hol, stable, psimh, psixh = _compute_stability_terms(
        constants, zbot_array, thvbot, qbot_array, ustar, tstar, qstar
    )

    rd = rdn / (1.0 + rdn / constants.von_karman_constant * (alz - psimh))
    rh = rhn / (1.0 + rhn / constants.von_karman_constant * (alz - psixh))
    re = ren / (1.0 + ren / constants.von_karman_constant * (alz - psixh))

    ustar = rd * vmag
    tstar = rh * delt
    qstar = re * delq
    hol, stable, psimh, psixh = _compute_stability_terms(
        constants, zbot_array, thvbot, qbot_array, ustar, tstar, qstar
    )

    rd = rdn / (1.0 + rdn / constants.von_karman_constant * (alz - psimh))
    rh = rhn / (1.0 + rhn / constants.von_karman_constant * (alz - psixh))
    re = ren / (1.0 + ren / constants.von_karman_constant * (alz - psixh))

    ustar = rd * vmag
    tstar = rh * delt
    qstar = re * delq

    tau = rbot_array * ustar * ustar
    taux = tau * ubot_array / vmag * mask_array
    tauy = tau * vbot_array / vmag * mask_array
    sen = cp * tau * tstar / ustar * mask_array
    lat = ltheat * tau * qstar / ustar * mask_array
    lwup = -constants.stefan_boltzmann_constant * ts_array**4 * mask_array
    evap = lat / ltheat * mask_array

    bn = constants.von_karman_constant / rdn
    bh = constants.von_karman_constant / rh
    ln0 = jnp.log(1.0 + (ztref / zbot_array) * (jnp.exp(bn) - 1.0))
    ln3 = jnp.log(1.0 + (ztref / zbot_array) * (jnp.exp(bn - bh) - 1.0))
    fac = ((ln0 - ztref / zbot_array * (bn - bh)) / bh) * stable + (
        (ln0 - ln3) / bh
    ) * (1.0 - stable)
    fac = jnp.minimum(jnp.maximum(fac, 0.0), 1.0)

    tref = ts_array + (thbot_array - ts_array) * fac * mask_array
    qref = qbot_array - delq * fac * mask_array

    return cast(
        tuple[jax.Array, ...],
        (sen, lat, lwup, evap, taux, tauy, tref, qref, ustar, tstar, qstar),
    )
