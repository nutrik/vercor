"""Pack SI forcing into standalone Veris physical (x, y) arrays.

Native Qnet/Qsw are positive upward, opposite VerCOR heat inputs. Rain and
water-equivalent snow remain separate and convert kg/m²/s to m/s exactly once;
signed bulk evaporation uses the same density conversion. ATemp/aqh are CESM
tref/qref, diagnosed from consistent model-level inputs at the configured
reference height. The default uses observed 10 m winds; model-level mode uses
actual 10 m winds from that same bulk solution and water-specific absorbed
shortwave radiation. Dry temperatures/pressure are safe operands
for native thermodynamics, while dry fluxes and humidity are zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from vercor._numerical_safety import _require_zero, require_active_finite
from vercor.dtypes import DTypePolicy, as_jax_real_array
from vercor.physics import PhysicalConstants
from vercor.setups._surface_fluxes import compute_surface_fluxes
from vercor.types import RuntimeArray


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VerisForcingPack:
    """Native forcing and direct-water stresses on physical west/south faces.

    Arrays use native (x, y) storage without halos; stress is in Pa. Bulk
    tracer-cell stress is averaged to each wet face before concentration
    weighting by the momentum-time face fraction in the flux adapter.
    """

    native_forcing: dict[str, RuntimeArray]
    open_water_u_stress: RuntimeArray
    open_water_v_stress: RuntimeArray


@partial(jax.jit, static_argnames=("dtype", "atmosphere_forcing"))
def prepare_veris_forcing_pack(
    fields: Mapping[str, RuntimeArray],
    *,
    ocean_mask: RuntimeArray,
    constants: PhysicalConstants,
    dtype: DTypePolicy,
    atmosphere_forcing: Literal["observed_10m", "model_level"] = "observed_10m",
) -> VerisForcingPack:
    """Evaluate shared bulk physics once and pack native forcing and face stress."""
    if atmosphere_forcing not in ("observed_10m", "model_level"):
        raise ValueError("atmosphere_forcing must be observed_10m or model_level")
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

    if atmosphere_forcing == "model_level":
        albedo = clean["open_water_albedo"]
        _require_zero(
            jnp.count_nonzero(wet & ((albedo < 0) | (albedo > 1))),
            "Veris open_water_albedo",
            "range [0, 1]",
        )
        clean["net_shortwave_radiation_flux"] = (1 - albedo) * clean[
            "downward_shortwave_radiation_flux"
        ]

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
    forcing = {
        "theta": native(clean["sea_surface_temperature"], 273.15),
        "ocSalt": native(clean["sea_surface_salinity"]),
        "uOcean": native(west),
        "vOcean": native(south),
        "ssh_an": native(clean.get("sea_surface_height", zero)),
        "surfPress": native(clean["surface_pressure"], constants.reference_pressure),
        "uWind": native(
            bulk.u_velocity_10m
            if atmosphere_forcing == "model_level"
            else clean["u_velocity_10m"]
        ),
        "vWind": native(
            bulk.v_velocity_10m
            if atmosphere_forcing == "model_level"
            else clean["v_velocity_10m"]
        ),
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

    west_wet = wet & jnp.roll(wet, 1, axis=1)
    south_wet = wet & jnp.concatenate((jnp.zeros_like(wet[:1]), wet[:-1]), axis=0)
    tau_u = 0.5 * (bulk.taux + jnp.roll(bulk.taux, 1, axis=1))
    tau_v = 0.5 * (
        bulk.tauy
        + jnp.concatenate((jnp.zeros_like(bulk.tauy[:1]), bulk.tauy[:-1]), axis=0)
    )
    return VerisForcingPack(
        forcing, jnp.where(west_wet, tau_u, 0.0).T, jnp.where(south_wet, tau_v, 0.0).T
    )
