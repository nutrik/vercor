"""Pack SI forcing into standalone Veris physical (x, y) arrays.

Native Qnet/Qsw are positive upward, opposite VerCOR heat inputs. Rain and
water-equivalent snow remain separate and convert kg/m²/s to m/s exactly once;
signed bulk evaporation uses the same density conversion. ATemp/aqh are CESM
tref/qref, diagnosed from consistent model-level inputs at the configured
reference height. Actual ERA5 10 m winds drive ice; neutral speed squared is
never used as a wind observation. Dry temperatures/pressure are safe operands
for native thermodynamics, while dry fluxes and humidity are zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp

from vercor._numerical_safety import require_active_finite
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.physics import PhysicalConstants
from vercor.setups._surface_fluxes import compute_surface_fluxes
from vercor.types import RuntimeArray


@partial(jax.jit, static_argnames=("dtype",))
def prepare_veris_forcing(
    fields: Mapping[str, RuntimeArray],
    *,
    ocean_mask: RuntimeArray,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
) -> dict[str, RuntimeArray]:
    """Return fresh prescribed forcing; callers reapply it every native substep."""
    wet = as_jax_real_array(ocean_mask, dtype) > 0
    clean: dict[str, jax.Array] = {}
    for name, field in fields.items():
        value = as_jax_real_array(field, dtype)
        require_active_finite(value, active_mask=wet, owner=f"Veris forcing '{name}'")
        fallback = (
            273.15
            if name
            in {"temperature", "potential_temperature", "sea_surface_temperature"}
            else (
                1.0
                if name in {"density", "model_level_height"}
                else constants.reference_pressure if name == "surface_pressure" else 0.0
            )
        )
        clean[name] = jnp.where(wet, value, as_jax_real_array(fallback, dtype))

    u, v = clean["ocean_u_velocity"], clean["ocean_v_velocity"]
    # Canonical ocean currents use Veros east/north faces. Close coastal and
    # exterior faces before centering or converting to native west/south faces.
    north_wet = jnp.concatenate((wet[1:], jnp.zeros_like(wet[:1])), axis=0)
    u = jnp.where(wet & jnp.roll(wet, -1, axis=1), u, 0.0)
    v = jnp.where(wet & north_wet, v, 0.0)
    west = jnp.roll(u, 1, axis=1)
    south = jnp.concatenate((jnp.zeros_like(v[:1]), v[:-1]), axis=0)
    u_center = 0.5 * (u + west)
    v_center = 0.5 * (v + south)
    bulk = compute_surface_fluxes(
        clean,
        sea_surface_temperature=clean["sea_surface_temperature"],
        u_ocean=u_center,
        v_ocean=v_center,
        ocean_mask=ocean_mask,
        constants=constants,
        dtype=dtype,
    )
    rho = as_jax_real_array(constants.freshwater_density, dtype)

    def native(value: RuntimeArray, dry: object = 0.0) -> RuntimeArray:
        return jnp.where(wet, value, as_jax_real_array(dry, dtype)).T

    zero = jnp.zeros_like(u)
    return {
        "theta": native(clean["sea_surface_temperature"], 273.15),
        "ocSalt": native(clean["sea_surface_salinity"]),
        "uOcean": native(west),
        "vOcean": native(south),
        "ssh_an": native(clean.get("sea_surface_height", zero)),
        "surfPress": native(clean["surface_pressure"], constants.reference_pressure),
        "uWind": native(clean["u_velocity_10m"]),
        "vWind": native(clean["v_velocity_10m"]),
        "ATemp": native(bulk.tref, 273.15),
        "aqh": native(bulk.qref),
        "SWdown": native(clean["downward_shortwave_radiation_flux"]),
        "LWdown": native(clean["downward_longwave_radiation_flux"]),
        "Qnet": native(-bulk.qnet),
        "Qsw": native(-clean["net_shortwave_radiation_flux"]),
        "precip": native(clean["rain_rate"] / rho),
        "snowfall": native(clean["snow_rate"] / rho),
        "evap": native(bulk.evap / rho),
        "runoff": zero.T,
    }
