from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
from vercor.settings import VercorSettings
from vercor.setups.external.veros_runtime_settings import configure_veros_runtime
from vercor.types import RuntimeArray

configure_veros_runtime()

from veros.state import VerosState  # noqa: E402


def compute_fluxes(
    veros_state: VerosState,
    runtime_fields: Mapping[str, RuntimeArray],
    settings: VercorSettings,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute atmosphere-ocean fluxes from explicit Veros and runtime fields."""

    vs = veros_state.variables

    u_tgrid = 0.5 * (
        as_jax_real_array(vs.u[1:, 2:-2, -1, vs.tau], settings)
        + as_jax_real_array(vs.u[:-1, 2:-2, -1, vs.tau], settings)
    )
    v_tgrid = 0.5 * (
        as_jax_real_array(vs.v[2:-2, 1:, -1, vs.tau], settings)
        + as_jax_real_array(vs.v[2:-2, :-1, -1, vs.tau], settings)
    )

    temp = as_jax_real_array(vs.temp[2:-2, 2:-2, -1, vs.tau], settings).T + 273.15

    (
        senf,
        latf,
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
        dqfldt,
    ) = compute_ocean_surface_fluxes(
        settings,
        as_jax_real_array(vs.maskT[2:-2, 2:-2, -1], settings).T,
        as_jax_real_array(runtime_fields["model_level_height"], settings),
        as_jax_real_array(runtime_fields["u_velocity"], settings),
        as_jax_real_array(runtime_fields["v_velocity"], settings),
        as_jax_real_array(runtime_fields["potential_temperature"], settings),
        as_jax_real_array(runtime_fields["specific_humidity"], settings),
        as_jax_real_array(runtime_fields["density"], settings),
        as_jax_real_array(runtime_fields["temperature"], settings),
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )
    _ = evap, tref, qref, duu10n, ustar, tstar, qstar

    qnet = (
        as_jax_real_array(runtime_fields.get("net_shortwave_radiation_flux"), settings)
        + as_jax_real_array(
            runtime_fields.get("downward_longwave_radiation_flux"), settings
        )
        + lwup
        + senf
        + latf
    )
    qnec = -jnp.where(dqfldt <= -1e10, 0.0, dqfldt)

    return (
        as_jax_real_array(taux, settings),
        as_jax_real_array(tauy, settings),
        as_jax_real_array(qnet, settings),
        as_jax_real_array(qnec, settings),
    )


__all__ = ["compute_fluxes"]
