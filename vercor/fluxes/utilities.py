from typing import Any
import numpy as np
from numpy.typing import NDArray

from vercor.settings import VercorSettings


def qsat(tk: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    result: NDArray[np.floating[Any]] = 640380.0 / np.exp(5107.4 / tk)
    return result


def qsat_august_eqn(
    ps: NDArray[np.floating[Any]], tk: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Saturated specific humidity (kg/kg)

    Arguments:
        ps (:obj:`ndarray`): atm sfc pressure (Pa)
        tk (:obj:`ndarray`): atm temperature (K)

    Returns:
        :obj:`ndarray`

    Reference:
        Barnier B., L. Siefridt, P. Marchesiello, (1995):
        Thermal forcing for a global ocean circulation model
        using a three-year climatology of ECMWF analyses,
        Journal of Marine Systems, 6, p. 363-380.
    """
    result: NDArray[np.floating[Any]] = (
        0.622 / ps * 10 ** (9.4051 - 2353.0 / tk) * 133.322
    )
    return result


def compute_pressure_levels(
    sp: NDArray[np.floating[Any]],
    hya: NDArray[np.floating[Any]],
    hyb: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Compute pressure levels

    Arguments:
        sp (:obj:`ndarray`): Atmospheric surface pressure
        hya (:obj:`ndarray`): Hybrid sigma level A coefficient for vertical grid
        hyb (:obj:`ndarray`): Hybrid sigma level B coefficient for vertical grid

    Return:
        :obj:`ndarray`
    """

    result: NDArray[np.floating[Any]] = (
        hya[np.newaxis, np.newaxis, :]
        + hyb[np.newaxis, np.newaxis, :] * sp[:, :, np.newaxis]
    )
    return result


def get_altitudes_hybrid_sigma_levels(
    settings: VercorSettings,
    t: NDArray[np.floating[Any]],
    q: NDArray[np.floating[Any]],
    ph: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Computes the altitudes at ECMWF Integrated Forecasting System
    (ECMWF-IFS) model half- and full-levels (for 137 levels model reanalysis: L137)

    Arguments:
        t (:obj:`ndarray`): Atmospheric temperture [K]
        q (:obj:`ndarray`): Atmospheric specific humidity [kg/kg]
        ph (:obj:`ndarray`): Pressure at half model levels [Pa]

    Note:
        The top level of the atmosphere is excluded

    Reference:
        - https://www.ecmwf.int/sites/default/files/elibrary/2015/9210-part-iii-dynamics-and-numerical-procedures.pdf
        - https://confluence.ecmwf.int/display/ECC/compute_geopotential_on_ml.py

    Returns:
        :obj:`ndarray`: Altitudes of the atmospheric full model levels [m]
    """

    # virtual temperature (K)
    tv = t[...] * (1.0 + settings.zvir * q[...])

    # dlog_p[0] = np.log(ph[:, :, 1:] / 0.1)
    # alpha[0] = np.log(2)
    dlog_p = np.log(ph[:, :, 1:] / ph[:, :, :-1])
    alpha = 1.0 - ((ph[:, :, :-1] / (ph[:, :, 1:] - ph[:, :, :-1])) * dlog_p)
    tv *= settings.rdair

    # zh is the geopotential of 'half-levels'
    # integrate zh to next half level
    increment = np.flip(tv * dlog_p, axis=2)
    zh = np.cumsum(increment, axis=2)

    # zf is the geopotential of this full level
    # integrate from previous (lower) half-level zh to the
    # full level
    increment_zh = np.insert(zh, 0, 0, axis=2)
    zf = np.flip(tv * alpha, axis=2) + increment_zh[:, :, :-1]

    alt: NDArray[np.floating[Any]] = (
        settings.earth_radius
        * zf
        / settings.gravity
        / (settings.earth_radius - zf / settings.gravity)
    )

    return alt[:, :, :]


def cdn(umps: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    return 0.0027 / umps + 0.000142 + 0.0000764 * umps


def psimhu(xd: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """

    result: NDArray[np.floating[Any]] = (
        np.log((1.0 + xd * (2.0 + xd)) * (1.0 + xd * xd) / 8.0)
        - 2.0 * np.arctan(xd)
        + 1.571
    )

    return result


def psixhu(xd: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    return 2.0 * np.log((1.0 + xd * xd) / 2.0)


def compute_air_density(
    settings: VercorSettings,
    pf: NDArray[np.floating[Any]],
    t: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Air density (kg/m^3)"""
    result: NDArray[np.floating[Any]] = (
        settings.mwdair / settings.rgas * pf[...] / t[...]
    )
    return result


def compute_potential_temperature(
    settings: VercorSettings,
    tbot: NDArray[np.floating[Any]],
    pf: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """Potential temperature (K)"""
    result: NDArray[np.floating[Any]] = (
        tbot[...] * (settings.p0 / pf[...]) ** settings.cappa
    )
    return result
