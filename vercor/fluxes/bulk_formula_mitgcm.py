import numpy as np
from numpy.typing import NDArray

from vercor.components import Component
from vercor.settings import VercorSettings


def bulkf_formula_lanl(
    settings: VercorSettings,
    shared_fields: Component,
    uw: NDArray,
    vw: NDArray,
    ta: NDArray,
    qa: NDArray,
    tsf: NDArray,
    ocn_mask: NDArray,
    iceornot: NDArray,
) -> tuple[
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
    NDArray,
]:
    """Calculate bulk formula fluxes over open ocean

        wind stress = (ust,vst) = rhoA * Cd * Ws * (del.u,del.v)
        Sensib Heat flux = fsha = rhoA * Ch * Ws * del.T * CpAir
        Latent Heat flux = flha = rhoA * Ce * Ws * del.Q * Lvap
                        = -Evap * Lvap
        with Ws = wind speed = sqrt(del.u^2 +del.v^2) ;
            del.T = Tair - Tsurf ; del.Q = Qair - Qsurf
            Cd,Ch,Ce = transfer coefficient for momentum, sensible
                    & latent heat flux [no units]

    Arguments:
        uw (:obj:`ndarray`): zonal wind speed (at grid center) [m/s]
        vw (:obj:`ndarray`): meridional wind speed (at grid center) [m/s]
        ta (:obj:`ndarray`): air temperature   [K]     at height ht
        qa (:obj:`ndarray`): specific humidity [kg/kg] at heigth ht
        tsf(:obj:`ndarray`): sea surface temperature [K]
        ocn_mask (:obj:`ndarray`): 0=land, 1=ocean
        iceornot (:obj:`ndarray`): 0=open water, 1=sea-ice, 2=sea-ice with snow

    Returns:
        flwupa (:obj:`ndarray`): upward long wave radiation (>0 upward) [W/m2]
        flha   (:obj:`ndarray`): latent heat flux         (>0 downward) [W/m2]
        fsha   (:obj:`ndarray`): sensible heat flux       (>0 downward) [W/m2]
        df0dT  (:obj:`ndarray`): derivative of heat flux with respect to Tsf [W/m2/K]
        ust    (:obj:`ndarray`): zonal wind stress (at grid center)     [N/m2]
        vst    (:obj:`ndarray`): meridional wind stress (at grid center)[N/m2]
        evp    (:obj:`ndarray`): evaporation rate (over open water) [kg/m2/s]
        ssq    (:obj:`ndarray`): surface specific humidity          [kg/kg]
        dEvdT  (:obj:`ndarray`): derivative of evap. with respect to tsf [kg/m2/s/K]
    """

    # Compute turbulent surface fluxes
    ht = 2.0
    zref = 10.0
    zice = 0.0005
    aln = np.log(ht / zref)
    czol = zref * settings.karman * settings.gravity

    # lath = np.ones_like(ocn_mask) * settings.latvap
    lath = np.where(
        iceornot > 0,
        np.ones_like(iceornot) * (settings.latvap + settings.latfresh),
        np.ones_like(iceornot) * settings.latvap,
    )

    rd = np.zeros_like(iceornot)
    rh = np.zeros_like(iceornot)
    re = np.zeros_like(iceornot)

    # wind speed
    us = np.sqrt(uw[...] * uw[...] + vw[...] * vw[...])
    usm = np.maximum(us[...], 1.0)

    t0 = ta[...] * (1.0 + settings.zvir * qa[...])
    ssq = 3.797915 * np.exp(lath[...] * (7.93252e-6 - 2.166847e-3 / tsf[...])) / 1013.0

    deltap = ta[...] - tsf[...] + settings.gamma_blk * ht
    delq = qa[...] - ssq[...]

    # initialize estimate exchange coefficients
    rdn = settings.karman / np.log(zref / zice)
    rhn = rdn
    ren = rdn

    # calculate turbulent scales
    ustar = rdn * usm[...]
    tstar = rhn * deltap[...]
    qstar = ren * delq[...]

    # iteration with psi-functions to find transfer coefficients
    for _ in range(5):
        huol = (
            czol
            / ustar[...] ** 2
            * (tstar[...] / t0 + qstar[...] / (1.0 / settings.zvir + qa[...]))
        )
        huol = np.minimum(np.abs(huol[...]), 10.0) * np.sign(huol[...])
        stable = 0.5 + 0.5 * np.sign(huol[...])
        xsq = np.maximum(np.sqrt(np.abs(1.0 - 16.0 * huol[...])), 1.0)
        x = np.sqrt(xsq[...])
        psimh = -5.0 * huol[...] * stable[...] + (1.0 - stable[...]) * (
            2.0 * np.log(0.5 * (1.0 + x[...]))
            + 2.0 * np.log(0.5 * (1.0 + xsq[...]))
            - 2.0 * np.arctan(x[...])
            + np.pi * 0.5
        )
        psixh = -5.0 * huol[...] * stable[...] + (1.0 - stable[...]) * (
            2.0 * np.log(0.5 * (1.0 + xsq[...]))
        )

        # update the transfer coefficients
        rd = rdn / (1.0 + rdn * (aln[...] - psimh[...]) / settings.karman)
        rh = rhn / (1.0 + rhn * (aln[...] - psixh[...]) / settings.karman)
        re = rh

        # update ustar, tstar, qstar using updated, shifted coefficients.
        ustar = rd[...] * usm[...]
        qstar = re[...] * delq[...]
        tstar = rh[...] * deltap[...]

    # tau = settings.rhoAir * ustar[...]**2
    # tau = tau * us[...] / usm[...]
    csha = settings.rhoAir * settings.cpdair * us[...] * rh[...] * rd[...]
    clha = settings.rhoAir * lath[...] * us[...] * re[...] * rd[...]

    fsha = csha[...] * deltap[...] * ocn_mask[...]
    flha = clha[...] * delq[...] * ocn_mask[...]
    evp = -flha[...] / lath[...] * ocn_mask[...]

    flwupa = (
        np.where(
            iceornot == 0,
            np.ones_like(iceornot)
            * settings.ocean_emissivity
            * settings.stefBoltz
            * tsf**4,
            np.where(
                iceornot == 2,
                np.ones_like(iceornot)
                * settings.snow_emissivity
                * settings.stefBoltz
                * tsf**4,
                np.ones_like(iceornot)
                * settings.ice_emissivity
                * settings.stefBoltz
                * tsf**4,
            ),
        )
        * ocn_mask[...]
    )

    dflwupdt = np.where(
        iceornot == 0,
        np.ones_like(iceornot)
        * 4.0
        * settings.ocean_emissivity
        * settings.stefBoltz
        * tsf**3,
        np.where(
            iceornot == 2,
            np.ones_like(iceornot)
            * 4.0
            * settings.snow_emissivity
            * settings.stefBoltz
            * tsf**3,
            np.ones_like(iceornot)
            * 4.0
            * settings.ice_emissivity
            * settings.stefBoltz
            * tsf**3,
        ),
    )

    devdt = clha[...] * ssq[...] * 2.166847e-3 / (tsf[...] * tsf[...]) * ocn_mask[...]
    dflhdt = -lath[...] * devdt[...]
    dfshdt = -csha[...] * ocn_mask[...]

    # total derivative with respect to surface temperature
    df0dt = (-dflwupdt[...] + dfshdt[...] + dflhdt[...]) * ocn_mask[...]

    #  wind stress at center points
    bulkf_cdn = 2.7e-3 / usm[...] + 0.142e-3 + 0.0764e-3 * usm[...]
    ust = settings.rhoAir * bulkf_cdn * us[...] * uw[...] * ocn_mask[...]
    vst = settings.rhoAir * bulkf_cdn * us[...] * vw[...] * ocn_mask[...]

    return (flwupa, flha, fsha, df0dt, ust, vst, evp, ssq * ocn_mask[...], devdt)
