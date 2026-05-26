import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.vertical_coordinates import (  # noqa: F401
    compute_hybrid_pressure_levels as compute_pressure_levels,
    compute_hybrid_sigma_full_level_altitudes as _compute_hybrid_sigma_full_level_altitudes,
    get_altitudes_hybrid_sigma_levels,
)
from vercor.settings import VercorSettings


def _as_jax_array(value: ArrayLike) -> jax.Array:
    return as_jax_real_array(value)


def _virtual_temperature_from_specific_humidity(
    temperature: ArrayLike,
    specific_humidity: ArrayLike,
    virtual_temperature_correction: float,
) -> jax.Array:
    """Return virtual temperature for specific humidity in kg/kg."""

    temperature_array = _as_jax_array(temperature)
    specific_humidity_array = _as_jax_array(specific_humidity)
    return temperature_array * (
        1.0 + virtual_temperature_correction * specific_humidity_array
    )


def qsat(tk: ArrayLike) -> jax.Array:
    """The saturation humidity of air (kg/m^3)

    Argument:
        tk (:obj:`ndarray`): temperature (K)
    """
    tk_array = _as_jax_array(tk)
    return 640380.0 / jnp.exp(5107.4 / tk_array)


def qsat_august_eqn(ps: ArrayLike, tk: ArrayLike) -> jax.Array:
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
    ps_array = _as_jax_array(ps)
    tk_array = _as_jax_array(tk)
    return 0.622 / ps_array * 10 ** (9.4051 - 2353.0 / tk_array) * 133.322


def cdn(umps: ArrayLike) -> jax.Array:
    """Neutral drag coeff at 10m

    Argument:
        umps (:obj:`ndarray`): wind speed (m/s)
    """
    umps_array = _as_jax_array(umps)
    return 0.0027 / umps_array + 0.000142 + 0.0000764 * umps_array


def psimhu(xd: ArrayLike) -> jax.Array:
    """Unstable part of psimh

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """

    xd_array = _as_jax_array(xd)
    return (
        jnp.log((1.0 + xd_array * (2.0 + xd_array)) * (1.0 + xd_array * xd_array) / 8.0)
        - 2.0 * jnp.arctan(xd_array)
        + 1.571
    )


def psixhu(xd: ArrayLike) -> jax.Array:
    """Unstable part of psimx

    Argument:
        xd (:obj:`ndarray`): model level height devided by Obukhov length
    """
    xd_array = _as_jax_array(xd)
    return 2.0 * jnp.log((1.0 + xd_array * xd_array) / 2.0)


def compute_air_density(
    settings: VercorSettings,
    pf: ArrayLike,
    t: ArrayLike,
) -> jax.Array:
    """Air density (kg/m^3)"""
    pf_array = _as_jax_array(pf)
    t_array = _as_jax_array(t)
    return settings.mwdair / settings.rgas * pf_array / t_array


def compute_potential_temperature(
    settings: VercorSettings,
    tbot: ArrayLike,
    pf: ArrayLike,
) -> jax.Array:
    """Potential temperature (K)"""
    tbot_array = _as_jax_array(tbot)
    pf_array = _as_jax_array(pf)
    return tbot_array * (settings.p0 / pf_array) ** settings.cappa
