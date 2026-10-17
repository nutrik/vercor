from __future__ import annotations

from copy import copy
from typing import Any

from veros.core.operators import numpy as npx, update, at
from veros.routines import veros_kernel, veros_routine
from veros.state import KernelOutput
from veros.setups.global_4deg import GlobalFourDegreeSetup
from veros.tools import get_periodic_interval
from veros.variables import Variable
from veros.core.utilities import enforce_boundaries


def _surface_tke_forcing(stress_x, stress_y, density):  # type: ignore[no-untyped-def]
    """Return wind-work TKE forcing with a finite calm-stress JAX derivative."""

    stress_squared = (stress_x / density) ** 2 + (stress_y / density) ** 2
    from veros import runtime_settings

    if runtime_settings.backend == "jax":
        from veros.core.operators import safe_sqrt

        stress_norm = safe_sqrt(stress_squared)
    else:
        stress_norm = npx.sqrt(stress_squared)
    return stress_norm**1.5


class CustomGlobalFourDegree(GlobalFourDegreeSetup):
    """Veros global 4-degree setup with VerCOR-controlled forcing fields."""

    def __init__(
        self,
        override: dict[str, Any] | None = None,
        *,
        restore_to_climatology_temp: bool = False,
        restore_to_climatology_salt: bool = False,
    ) -> None:
        """Keep independent restoring choices as setup configuration."""

        self.restore_to_climatology_temp = restore_to_climatology_temp
        self.restore_to_climatology_salt = restore_to_climatology_salt
        super().__init__(override=override)

    @veros_routine
    def set_parameter(self, state):  # type: ignore[no-untyped-def]
        """Declare coupled salt flux and independent native restoring weights."""

        super().set_parameter(state)
        state.var_meta.update(
            coupled_feedback_active=Variable(
                "Total feedback activation",
                (),
                "1",
                "One after total sea-ice feedback is installed; zero during climatological spinup",
                initial=0.0,
                time_dependent=False,
            ),
            coupled_salt_flux=Variable(
                "Coupled virtual salt flux",
                ("xt", "yt"),
                "m g kg-1 s-1",
                "VerCOR-computed total virtual salt contribution from bulk mass and actual salt",
                initial=0.0,
            ),
            temp_restoring=Variable(
                "Temperature restoring weight",
                (),
                "1",
                "Enable the surface temperature correction to climatology",
                initial=float(self.restore_to_climatology_temp),
                time_dependent=False,
            ),
            salt_restoring=Variable(
                "Salinity restoring weight",
                (),
                "1",
                "Enable the existing 30-day surface salinity restoring term",
                initial=float(self.restore_to_climatology_salt),
                time_dependent=False,
            ),
        )
        salt_metadata = copy(state.var_meta["forc_salt_surface"])
        salt_metadata.units = "m g kg-1 s-1"
        state.var_meta["forc_salt_surface"] = salt_metadata

    @veros_routine
    def set_forcing(self, state):  # type: ignore[no-untyped-def]
        """Apply the adapter kernel instead of the upstream module's kernel."""

        state.variables.update(self.set_forcing_kernel(state))

    @veros_kernel
    def set_forcing_kernel(state):  # type: ignore
        vs = state.variables
        settings = state.settings

        year_in_seconds = 365 * 86400.0
        (n1, f1), (n2, f2) = get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12.0, 12
        )

        vs.surface_taux = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        if settings.enable_tke:
            vs.forc_tke_surface = update(
                vs.forc_tke_surface,
                at[1:-1, 1:-1],
                _surface_tke_forcing(
                    0.5 * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1]),
                    0.5 * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2]),
                    settings.rho_0,
                ),
            )

        cp_0 = 3991.86795711963
        sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
        qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
        qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
        vs.forc_temp_surface = (
            (qnet + vs.temp_restoring * qnec * (sst - vs.temp[:, :, -1, vs.tau]))
            * vs.maskT[:, :, -1]
            / cp_0
            / settings.rho_0
        )

        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface = (
            vs.salt_restoring
            / t_rest
            * (sss - vs.salt[:, :, -1, vs.tau])
            * vs.maskT[:, :, -1]
            * vs.dzt[-1]
        )
        vs.forc_salt_surface = (vs.coupled_salt_flux + vs.forc_salt_surface) * vs.maskT[
            :, :, -1
        ]

        mask = npx.logical_and(
            vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
            vs.forc_temp_surface < 0.0,
        )
        mask = npx.logical_and(mask, vs.coupled_feedback_active == 0)
        vs.forc_temp_surface = npx.where(mask, 0.0, vs.forc_temp_surface)
        vs.forc_salt_surface = npx.where(mask, 0.0, vs.forc_salt_surface)
        vs.forc_salt_surface = enforce_boundaries(
            vs.forc_salt_surface, settings.enable_cyclic_x
        )

        output = dict(
            surface_taux=vs.surface_taux,
            surface_tauy=vs.surface_tauy,
            forc_temp_surface=vs.forc_temp_surface,
            forc_salt_surface=vs.forc_salt_surface,
        )
        if settings.enable_tke:
            output["forc_tke_surface"] = vs.forc_tke_surface
        return KernelOutput(**output)

    @veros_routine
    def set_diagnostics(self, state):  # type: ignore
        settings = state.settings
        state.diagnostics["snapshot"].output_frequency = 365 * 86400.0
        state.diagnostics["overturning"].output_frequency = 365 * 86400.0
        state.diagnostics["overturning"].sampling_frequency = settings.dt_tracer
        state.diagnostics["energy"].output_frequency = 365 * 86400.0
        state.diagnostics["energy"].sampling_frequency = 86400
        average_vars = [
            "temp",
            "salt",
            "u",
            "v",
            "w",
            "surface_taux",
            "surface_tauy",
            "psi",
            "qnet",
            "qnec",
        ]
        state.diagnostics["averages"].output_variables = average_vars
        state.diagnostics["averages"].output_frequency = 365 * 86400.0
        state.diagnostics["averages"].sampling_frequency = 86400


__all__ = ["CustomGlobalFourDegree"]
