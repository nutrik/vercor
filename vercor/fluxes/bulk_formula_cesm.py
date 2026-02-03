from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from vercor.fluxes.utilities import cdn, psimhu, psixhu, qsat
from vercor.settings import VercorSettings


def old_flux_atmOcn(
    settings: VercorSettings,
    mask: NDArray,
    rbot: NDArray,
    zbot: NDArray,
    ubot: NDArray,
    vbot: NDArray,
    qbot: NDArray,
    tbot: NDArray,
    thbot: NDArray,
    us: NDArray,
    vs: NDArray,
    ts: NDArray,
) -> Tuple[NDArray, ...]:
    """atm/ocn fluxes calculation

    Arguments:
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        rbot (:obj:`ndarray`): atm density           (kg/m^3)
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        tbot (:obj:`ndarray`): atm T                 (K)
        thbot(:obj:`ndarray`): atm potential T       (K)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): ocn temperature       (K)

    Returns:
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N/m^2) # original was typo - N
        tauy (:obj:`ndarray`): surface stress, maridional (N/m^2) # original was typo - N

        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)
        duu10n(:obj:`ndarray`): diag: 10m wind speed squared (m/s)^2

        ustar_sv(:obj:`ndarray`): diag: ustar
        re_sv   (:obj:`ndarray`): diag: sqrt of exchange coefficient (water)
        ssq_sv  (:obj:`ndarray`): diag: sea surface humidity  (kg/kg)

    Reference:
        - Large, W. G., & Pond, S. (1981). Open Ocean Momentum Flux Measurements in Moderate to Strong Winds,
        Journal of Physical Oceanography, 11(3), pp. 324-336
        - Large, W. G., & Pond, S. (1982). Sensible and Latent Heat Flux Measurements over the Ocean,
        Journal of Physical Oceanography, 12(5), 464-482.
        - https://svn-ccsm-release.cgd.ucar.edu/model_versions/cesm1_0_5/models/csm_share/shr/shr_flux_mod.F90
    """

    al2 = np.log(settings.zref / settings.ztref)

    vmag = np.maximum(
        settings.umin_ocean,
        np.sqrt((ubot[...] - us[...]) ** 2 + (vbot[...] - vs[...]) ** 2),
    )

    # sea surface humidity (kg/kg)
    ssq = 0.98 * qsat(ts[...]) / rbot[...]

    # potential temperature diff. (K)
    delt = thbot[...] - ts[...]

    # specific humidity diff. (kg/kg)
    delq = qbot[...] - ssq[...]

    alz = np.log(zbot[...] / settings.zref)
    cp = settings.cpdair * (1.0 + settings.cpvir * ssq[...])

    # first estimate of Z/L and ustar, tstar and qstar

    # neutral coefficients, z/L = 0.0
    stable = 0.5 + 0.5 * np.sign(delt[...])
    rdn = np.sqrt(cdn(vmag[...]))
    rhn = (1.0 - stable) * 0.0327 + stable * 0.018
    ren = 0.0346

    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # compute stability & evaluate all stability functions
    hol = (
        settings.karman
        * settings.gravity
        * zbot[...]
        * (tstar[...] / thbot[...] + qstar[...] / (1.0 / settings.zvir + qbot[...]))
        / ustar[...] ** 2
    )
    hol = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = np.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / settings.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / settings.karman * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # iterate to converge on Z/L, ustar, tstar and qstar

    # compute stability & evaluate all stability functions
    hol = (
        settings.karman
        * settings.gravity
        * zbot[...]
        * (tstar[...] / thbot[...] + qstar[...] / (1.0 / settings.zvir + qbot[...]))
        / ustar[...] ** 2
    )
    hol = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # shift wind speed using old coefficient
    rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
    u10n = vmag[...] * rd[...] / rdn[...]

    # update transfer coeffs at 10m and neutral stability
    rdn = np.sqrt(cdn(u10n[...]))
    ren = 0.0346
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018

    # shift all coeffs to measurement height and stability
    rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
    rh = rhn[...] / (1.0 + rhn[...] / settings.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / settings.karman * (alz[...] - psixh[...]))

    # update ustar, tstar, qstar using updated, shifted coeffs
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # compute the fluxes

    tau = rbot[...] * ustar[...] * ustar[...]

    # momentum flux
    taux = tau[...] * (ubot[...] - us[...]) / vmag[...] * mask[...]
    tauy = tau[...] * (vbot[...] - vs[...]) / vmag[...] * mask[...]

    # heat flux
    sen = cp[...] * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = settings.latvap * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -settings.stefBoltz * ts[...] ** 4 * mask[...]

    # water flux
    evap = lat[...] / settings.latvap * mask[...]

    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared

    hol = hol[...] * settings.ztref / zbot[...]
    xsq = np.maximum(1.0, np.sqrt(np.abs(1.0 - 16.0 * hol[...])))
    xqq = np.sqrt(xsq)
    psix2 = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])
    fac = (rh[...] / settings.karman) * (alz[...] + al2 - psixh[...] + psix2[...])
    tref = thbot[...] - delt[...] * fac[...]

    # pot. temp to temp correction
    tref = (tref[...] - 0.01 * settings.ztref) * mask[...]
    fac = (
        (re[...] / settings.karman)
        * (alz[...] + al2 - psixh[...] + psix2[...])
        * mask[...]
    )
    qref = (qbot[...] - delq[...] * fac[...]) * mask[...]

    # 10m wind speed squared
    duu10n = u10n[...] * u10n[...] * mask[...]

    # Calculate correction term of net ocean heat flux (W/m^2) for data forced ocean
    # total derivative with respect to surface temperature
    # Ported from MITgcm bulkf_formula_lanl.f90
    clha = rbot[...] * settings.latvap * vmag[...] * re[...] * rd[...]
    devdt = clha[...] * ssq[...] * 2.166847e-3 / (ts[...] * ts[...])

    dflwupdt = -4.0 * settings.ocean_emissivity * settings.stefBoltz * (ts[...] ** 3)
    dfshdt = -rbot[...] * settings.cpdair * vmag[...] * rh[...] * rd[...]
    dflhdt = -settings.latvap * devdt[...]
    df0dt = (dflwupdt[...] + dfshdt[...] + dflhdt[...]) * mask[...]

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


def new_flux_atmOcn(
    settings: VercorSettings,
    mask: NDArray,
    zbot: NDArray,
    ubot: NDArray,
    vbot: NDArray,
    thbot: NDArray,
    qbot: NDArray,
    rbot: NDArray,
    tbot: NDArray,
    us: NDArray,
    vs: NDArray,
    ts: NDArray,
    use_coldair_outbreak_mod: bool = False,
    missval: float = 0.0,
) -> Tuple[NDArray, ...]:
    """
    Atm-Ocean flux calculation

    - all fluxes are positive downward
    - net heat flux = net sw + lw up + lw down + sen + lat
    - here, tstar = <WT>/U*, and qstar = <WQ>/U*.
    - wind speeds should all be above a minimum speed (eg. 1.0 m/s)

    Assumptions:
    ------------
        Large:
            - Neutral 10m drag coeff: cdn = .0027/U10 + .000142 + .0000764 U10
            - Neutral 10m stanton number: ctn = .0327 sqrt(cdn), unstable
                                            ctn = .0180 sqrt(cdn), stable
            - Neutral 10m dalton number:  cen = .0346 sqrt(cdn)
            - The saturation humidity of air at T(K): qsat(T)  (kg/m^3)

    Arguments:
    ---------
        mask (:obj:`ndarray`): ocn domain mask       0 <=> out of domain
        zbot (:obj:`ndarray`): atm level height      (m)
        ubot (:obj:`ndarray`): atm u wind            (m/s)
        vbot (:obj:`ndarray`): atm v wind            (m/s)
        thbot(:obj:`ndarray`): atm potential T       (K)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        rbot (:obj:`ndarray`): atm air density       (kg/m^3)
        tbot (:obj:`ndarray`): atm T                 (K)
        us   (:obj:`ndarray`): ocn u-velocity        (m/s)
        vs   (:obj:`ndarray`): ocn v-velocity        (m/s)
        ts   (:obj:`ndarray`): ocn temperature       (K)

        seq_flux_atmocn_minwind (float):  minimum wind speed for atmocn      (m/s)
        missval (float): masked value (optional)

    Returns:
    -------
        sen  (:obj:`ndarray`): heat flux: sensible    (W/m^2)
        lat  (:obj:`ndarray`): heat flux: latent      (W/m^2)
        lwup (:obj:`ndarray`): heat flux: lw upward   (W/m^2)
        evap (:obj:`ndarray`): water flux: evap  ((kg/s)/m^2)
        taux (:obj:`ndarray`): surface stress, zonal      (N/m^2) # original was typo - N
        tauy (:obj:`ndarray`): surface stress, maridional (N/m^2) # original was typo - N
        tref (:obj:`ndarray`): diag:  2m ref height T     (K)
        qref (:obj:`ndarray`): diag:  2m ref humidity (kg/kg)
        duu10n(:obj:`ndarray`): diag: 10m wind speed squared (m/s)^2

        ustar_sv(:obj:`ndarray`): diag: ustar (optional)
        re_sv   (:obj:`ndarray`): diag: sqrt of exchange coefficient (water) (optional)
        ssq_sv  (:obj:`ndarray`): diag: sea surface humidity  (kg/kg) (optional)

    Revision history:
        2002-Jun-10 - B. Kauffman - code migrated from cpl5 to cpl6
        2003-Apr-02 - B. Kauffman - taux & tauy now utilize ocn velocity
        2003-Apr-02 - B. Kauffman - tref,qref,duu10n mods as per Bill Large
        2006-Nov-07 - B. Kauffman - code migrated from cpl6 to share

    Local variables:
    ----------------
        vmag    surface wind magnitude   (m/s)
        ssq     sea surface humidity     (kg/kg)
        delt    potential T difference   (K)
        delq    humidity difference      (kg/kg)
        stable  stability factor
        rdn     sqrt of neutral exchange coeff (momentum)
        rhn     sqrt of neutral exchange coeff (heat)
        ren     sqrt of neutral exchange coeff (water)
        rd      sqrt of exchange coefficient (momentum)
        rh      sqrt of exchange coefficient (heat)
        re      sqrt of exchange coefficient (water)
        ustar   ustar
        real(r8)     :: ustar_prev
        qstar   qstar
        tstar   tstar
        hol     H (at zbot) over L
        xsq     ?
        xqq     ?

        psimh   stability function at zbot (momentum)
        psixh   stability function at zbot (heat and water)
        psix2   stability function at ztref reference height
        alz     ln(zbot/zref)
        al2     ln(zref/ztref)
        u10n    10m neutral wind
        tau     stress at zbot
        cp      specific heat of moist air
        fac     vertical interpolation factor
        spval   local missing value

        --- local functions --------------------------------
        qsat    function: the saturation humididty of air (kg/m^3)
        ++ Large only (formula v*=[c4/U10+c5+c6*U10]*U10 in Large et al. 1994)
        cdn     function: neutral drag coeff at 10m
        ++ Large only (stability functions)
        psimhu  function: unstable part of psimh
        psixhu  function: unstable part of psimx
        Umps    dummy arg ~ wind velocity (m/s)
        Tk      dummy arg ~ temperature (K)
        xd      dummy arg ~ ?

        --- for cold air outbreak calc --------------------------------
        tdiff(nMax)                tbot - ts
        vscl

    """

    hol = np.zeros_like(mask)
    u10n = np.zeros_like(mask)
    duu10n = np.zeros_like(mask)
    psixh = np.zeros_like(mask)

    # reference height           (m)
    zref = 10.0
    # reference height for air T (m)
    ztref = 2.0
    # maximum wind scaling for flux (m/s)
    maxscl = 2.0
    # start t-ts for scaling (K)
    td0 = -10.0
    alpha = 1.4

    # These control convergence of the iterative flux calculation
    # (For Large and Pond scheme only; not UA or COARE).
    flux_con_tol = 0.0
    flux_con_max_iter = 2

    # if (debug > 0 .and. s_loglev > 0) write(s_logunit,F00) "enter"

    # --- for cold air outbreak calc --------------------------------
    tdiff = tbot[...] - ts[...]

    al2 = np.log(zref / ztref)

    # --- compute some needed quantities ---
    vmag = np.maximum(
        settings.umin_ocean,
        np.sqrt((ubot[...] - us[...]) ** 2 + (vbot[...] - vs[...]) ** 2),
    )
    if use_coldair_outbreak_mod:
        # Cold Air Outbreak Modification:
        # Increase windspeed for negative tbot-ts
        # based on Mahrt & Sun 1995,MWR
        coldair_outbreak_mask = tdiff < td0
        vscl = np.minimum(
            1.0 + alpha * (np.abs(tdiff[...] - td0) ** 0.5 / np.abs(vmag[...])),
            maxscl,
        )
        vmag = np.where(coldair_outbreak_mask[...], vmag[...] * vscl[...], vmag[...])

    ssq = 0.98 * qsat(ts[...]) / rbot[...]  # sea surf hum (kg/kg)
    delt = thbot[...] - ts[...]  # pot temp diff (K)
    delq = qbot[...] - ssq[...]  # spec hum dif (kg/kg)
    alz = np.log(zbot[...] / zref)
    cp = settings.cpdair * (1.0 + settings.cpvir * ssq)

    # ------------------------------------------------------------
    # first estimate of Z/L and ustar, tstar and qstar
    # ------------------------------------------------------------
    # --- neutral coefficients, z/L = 0.0 ---
    stable = 0.5 + 0.5 * np.sign(delt[...])
    rdn = np.sqrt(cdn(vmag[...]))
    rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018
    # (1.0-stable) * chxcdu + stable * chxcds
    ren = np.ones_like(rdn) * 0.0346  # cexcd

    rd = rdn[...]  # initial guess for rd
    rh = rhn[...]  # initial guess for rh
    re = ren[...]  # initial guess for re

    # --- ustar, tstar, qstar ---
    ustar = rdn[...] * vmag[...]
    tstar = rhn[...] * delt[...]
    qstar = ren * delq[...]
    ustar_prev = ustar[...] * 2.0
    iter = 0
    while (
        np.any(np.abs((ustar[...] - ustar_prev[...]) / ustar[...]) > flux_con_tol)
        and iter < flux_con_max_iter
    ):
        iter += 1
        ustar_prev = ustar[...]
        # --- compute stability & evaluate all stability functions ---
        hol = (
            settings.karman
            * settings.gravity
            * zbot[...]
            * (tstar[...] / thbot[...] + qstar[...] / (1.0 / settings.zvir + qbot[...]))
            / ustar[...] ** 2
        )
        hol = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
        stable = 0.5 + 0.5 * np.sign(hol[...])
        xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
        xqq = np.sqrt(xsq[...])
        psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq)
        psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq)

        # --- shift wind speed using old coefficient ---
        rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
        u10n = vmag[...] * rd[...] / rdn[...]

        # --- update transfer coeffs at 10m and neutral stability ---
        rdn = np.sqrt(cdn(u10n[...]))
        ren = 0.0346  # cexcd
        rhn = (1.0 - stable[...]) * 0.0327 + stable[...] * 0.018
        # (1.0-stable) * chxcdu + stable * chxcds

        # --- shift all coeffs to measurement height and stability ---
        rd = rdn[...] / (1.0 + rdn[...] / settings.karman * (alz[...] - psimh[...]))
        rh = rhn[...] / (1.0 + rhn[...] / settings.karman * (alz[...] - psixh[...]))
        re = ren / (1.0 + ren / settings.karman * (alz[...] - psixh[...]))

        # --- update ustar, tstar, qstar using updated, shifted coeffs --
        ustar = rd * vmag[...]
        tstar = rh * delt[...]
        qstar = re[...] * delq[...]

    if iter < 1:
        print(ustar, ustar_prev, flux_con_tol, flux_con_max_iter)
        raise RuntimeError("No iterations performed ")

    # ------------------------------------------------------------
    # compute the fluxes
    # ------------------------------------------------------------

    tau = rbot[...] * ustar[...] * ustar[...]

    # --- momentum flux ---
    # x surface stress (N)
    taux = tau[...] * (ubot[...] - us[...]) / vmag[...] * mask[...]
    # y surface stress (N)
    tauy = tau[...] * (vbot[...] - vs[...]) / vmag[...] * mask[...]

    # --- heat flux ---
    # sensible heat flux  (W/m^2)
    sen = cp * tau[...] * tstar[...] / ustar[...] * mask[...]
    # latent heat flux  (W/m^2)
    lat = settings.latvap * tau[...] * qstar[...] / ustar[...] * mask[...]
    # long-wave upward heat flux  (W/m^2)
    lwup = -settings.stefBoltz * ts[...] ** 4 * mask[...]
    # --- water flux ---
    # evaporative water flux ((kg/s)/m^2)
    evap = lat[...] / settings.latvap * mask[...]

    # ------------------------------------------------------------
    # compute diagnositcs: 2m ref T & Q, 10m wind speed squared
    # ------------------------------------------------------------
    hol = hol[...] * ztref / zbot[...]
    xsq = np.maximum(1.0, np.sqrt(np.abs(1.0 - 16.0 * hol[...])))
    xqq = np.sqrt(xsq[...])
    psix2 = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq)
    fac = (rh[...] / settings.karman) * (alz[...] + al2 - psixh[...] + psix2[...])
    # 2m reference height temperature (K)
    tref = thbot[...] - delt[...] * fac[...]
    tref[...] = (tref[...] - 0.01 * ztref) * mask[...]  # pot temp to temp correction
    fac[...] = (re / settings.karman) * (alz[...] + al2 - psixh[...] + psix2[...])
    # 2m reference height humidity (kg/kg)
    qref = qbot[...] - delq[...] * fac[...] * mask[...]
    # 10m wind speed squared (m/s)^2
    duu10n = u10n[...] * u10n[...] * mask[...]  # 10m wind speed squared

    # ------------------------------------------------------------
    # optional diagnostics, needed for water tracer fluxes (dcn)
    # ------------------------------------------------------------
    ustar_sv = ustar[...] * mask[...]  # noqa: F841
    re_sv = re[...] * mask[...]  # noqa: F841
    ssq_sv = ssq[...] * mask[...]  # noqa: F841

    # Calculate correction term of net ocean heat flux (W/m^2) for data forced ocean
    # total derivative with respect to surface temperature
    # Ported from MITgcm bulkf_formula_lanl.f90
    clha = rbot[...] * settings.latvap * vmag[...] * re[...] * rd[...]
    devdt = clha[...] * ssq[...] * 2.166847e-3 / (ts[...] * ts[...])

    dflwupdt = -4.0 * settings.ocean_emissivity * settings.stefBoltz * (ts[...] ** 3)
    dfshdt = -rbot[...] * settings.cpdair * vmag[...] * rh[...] * rd[...]
    dflhdt = -settings.latvap * devdt[...]
    df0dt = dflwupdt[...] + dfshdt[...] + dflhdt[...] * mask[...]

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
    mask: NDArray,
    zbot: NDArray,
    ubot: NDArray,
    vbot: NDArray,
    thbot: NDArray,
    qbot: NDArray,
    rbot: NDArray,
    tbot: NDArray,
    ts: NDArray,
    missval: float = 0.0,
) -> Tuple[NDArray, ...]:
    """
    Atm-SeaIce flux calculation

    Arguments:
    ----------
        mask (:obj:`ndarray`): 0 <=> cell NOT in model domain
        zbot (:obj:`ndarray`): atm level height  (m)
        ubot (:obj:`ndarray`): atm u wind     (m/s)
        vbot (:obj:`ndarray`): atm v wind     (m/s)
        thbot(:obj:`ndarray`): atm potential T   (K)
        qbot (:obj:`ndarray`): atm specific humidity (kg/kg)
        rbot (:obj:`ndarray`): atm air density   (kg/m^3)
        tbot (:obj:`ndarray`): atm T       (K)
        ts   (:obj:`ndarray`): surface temperature

    Returns:
    --------
        sen  (:obj:`ndarray`): sensible      heat flux  (W/m^2)
        lat  (:obj:`ndarray`): latent        heat flux  (W/m^2)
        lwup (:obj:`ndarray`): long-wave upward heat flux  (W/m^2)
        evap (:obj:`ndarray`): evaporative water flux ((kg/s)/m^2)
        taux (:obj:`ndarray`): x surface stress (N/m^2)  # original was typo - N
        tauy (:obj:`ndarray`): y surface stress (N/m^2)  # original was typo - N
        tref (:obj:`ndarray`): 2m reference height temperature
        qref (:obj:`ndarray`): 2m reference height humidity

    Local variables:
    ----------------
        vmag    surface wind magnitude   (m/s)
        thvbot  virtual temperature      (K)
        ssq     sea surface humidity     (kg/kg)
        delt    potential T difference   (K)
        delq    humidity difference      (kg/kg)
        stable  stability factor
        rdn     sqrt of neutral exchange coefficient (momentum)
        rhn     sqrt of neutral exchange coefficient (heat)
        ren     sqrt of neutral exchange coefficient (water)
        rd      sqrt of exchange coefficient (momentum)
        rh      sqrt of exchange coefficient (heat)
        re      sqrt of exchange coefficient (water)
        ustar   ustar
        qstar   qstar
        tstar   tstar
        hol     H (at zbot) over L
        xsq     temporary variable
        xqq     temporary variable
        psimh   stability function at zbot (momentum)
        psixh   stability function at zbot (heat and water)
        alz     ln(zbot/z10)
        ltheat  latent heat for surface
        tau     stress at zbot
        cp      specific heat of moist air

        bn      exchange coef funct for interpolation
        bh      exchange coef funct for interpolation
        fac     interpolation factor
        ln0     log factor for interpolation
        ln3     log factor for interpolation
    """

    zref = 10.0  # ref height           ~ m
    ztref = 2.0  # ref height for air T ~ m
    # spval  = shr_const_spval # special value
    zzsice = 0.0005  # ice surface roughness

    # --- define some needed variables ---
    vmag = np.maximum(settings.umin_ice, np.sqrt(ubot[...] ** 2 + vbot[...] ** 2))
    thvbot = thbot[...] * (1.0 + settings.zvir * qbot[...])  # virtual pot temp (K)
    ssq = qsat(ts[...]) / rbot[...]  # sea surf hum (kg/kg)
    delt = thbot[...] - ts[...]  # pot temp diff (K)
    delq = qbot[...] - ssq[...]  # spec hum dif (kg/kg)
    alz = np.log(zbot[...] / zref)
    cp = settings.cpdair * (1.0 + settings.cpvir * ssq[...])
    ltheat = settings.latvap + settings.latice

    # ----------------------------------------------------------
    # first estimate of Z/L and ustar, tstar and qstar
    # ----------------------------------------------------------

    # --- neutral coefficients, z/L = 0.0 ---
    rdn = settings.karman / np.log(zref / zzsice)
    rhn = rdn
    ren = rdn

    # --- ustar,tstar,qstar ----
    ustar = rdn * vmag[...]
    tstar = rhn * delt[...]
    qstar = ren * delq[...]

    # --- compute stability & evaluate all stability functions ---
    hol = (
        settings.karman
        * settings.gravity
        * zbot[...]
        * (tstar[...] / thvbot[...] + qstar[...] / (1.0 / settings.zvir + qbot[...]))
        / ustar**2
    )
    hol = np.minimum(abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # --- shift all coeffs to measurement height and stability ---
    rd = rdn / (1.0 + rdn / settings.karman * (alz[...] - psimh[...]))
    rh = rhn / (1.0 + rhn / settings.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / settings.karman * (alz[...] - psixh[...]))

    # --- update ustar, tstar, qstar w/ updated, shifted coeffs --
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # ----------------------------------------------------------
    # iterate to converge on Z/L, ustar, tstar and qstar
    # ----------------------------------------------------------

    # --- compute stability & evaluate all stability functions ---
    hol = (
        settings.karman
        * settings.gravity
        * zbot[...]
        * (tstar[...] / thvbot[...] + qstar[...] / (1.0 / settings.zvir + qbot[...]))
        / ustar**2
    )
    hol = np.minimum(np.abs(hol[...]), 10.0) * np.sign(hol[...])
    stable = 0.5 + 0.5 * np.sign(hol[...])
    xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * hol[...])), 1.0)
    xqq = np.sqrt(xsq[...])
    psimh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psimhu(xqq[...])
    psixh = -5.0 * hol[...] * stable[...] + (1.0 - stable[...]) * psixhu(xqq[...])

    # --- shift all coeffs to measurement height and stability ---
    rd = rdn / (1.0 + rdn / settings.karman * (alz[...] - psimh[...]))
    rh = rhn / (1.0 + rhn / settings.karman * (alz[...] - psixh[...]))
    re = ren / (1.0 + ren / settings.karman * (alz[...] - psixh[...]))

    # --- update ustar, tstar, qstar w/ updated, shifted coeffs --
    ustar = rd[...] * vmag[...]
    tstar = rh[...] * delt[...]
    qstar = re[...] * delq[...]

    # ----------------------------------------------------------
    # compute the fluxes
    # ----------------------------------------------------------

    tau = rbot[...] * ustar[...] * ustar[...]

    # --- momentum flux ---
    taux = tau[...] * ubot[...] / vmag[...] * mask[...]
    tauy = tau[...] * vbot[...] / vmag[...] * mask[...]

    # --- heat flux ---
    sen = cp * tau[...] * tstar[...] / ustar[...] * mask[...]
    lat = ltheat * tau[...] * qstar[...] / ustar[...] * mask[...]
    lwup = -settings.stefBoltz * ts[...] ** 4 * mask[...]

    # --- water flux ---
    evap = lat[...] / ltheat * mask[...]

    # ----------------------------------------------------------
    # compute diagnostic: 2m reference height temperature
    # ----------------------------------------------------------

    # Compute function of exchange coefficients. Assume that
    # cn = rdn*rdn, cm=rd*rd and ch=rh*rd, and therefore
    # 1/sqrt(cn[...])=1/rdn and sqrt(cm[...])/ch[...]=1/rh
    bn = settings.karman / rdn
    bh = settings.karman / rh

    # Interpolation factor for stable and unstable cases
    ln0 = np.log(1.0 + (ztref / zbot[...]) * (np.exp(bn) - 1.0))
    ln3 = np.log(1.0 + (ztref / zbot[...]) * (np.exp(bn - bh) - 1.0))
    fac = (ln0[...] - ztref / zbot[...] * (bn - bh)) / bh * stable[...] + (
        ln0[...] - ln3[...]
    ) / bh * (1.0 - stable[...])
    fac[...] = np.minimum(np.maximum(fac[...], 0.0), 1.0)

    # Actual interpolation
    tref = ts[...] + (tbot[...] - ts[...]) * fac[...] * mask[...]
    qref = qbot[...] - delq[...] * fac[...] * mask[...]

    return (sen, lat, lwup, evap, taux, tauy, tref, qref, ustar, tstar, qstar)
