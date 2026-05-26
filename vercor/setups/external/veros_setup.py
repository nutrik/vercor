from __future__ import annotations

from vercor.setups.external.veros_runtime_settings import configure_veros_runtime

configure_veros_runtime()

from veros.core.operators import numpy as npx, update, at  # noqa: E402
from veros.routines import veros_kernel, veros_routine  # noqa: E402
from veros.state import KernelOutput  # noqa: E402
from veros.setups.global_4deg import GlobalFourDegreeSetup  # noqa: E402
from veros.tools import get_periodic_interval  # noqa: E402


class CustomGlobalFourDegree(GlobalFourDegreeSetup):
    """Veros global 4-degree setup with VerCOR-controlled forcing fields."""

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
                npx.sqrt(
                    (
                        0.5
                        * (vs.surface_taux[1:-1, 1:-1] + vs.surface_taux[:-2, 1:-1])
                        / settings.rho_0
                    )
                    ** 2
                    + (
                        0.5
                        * (vs.surface_tauy[1:-1, 1:-1] + vs.surface_tauy[1:-1, :-2])
                        / settings.rho_0
                    )
                    ** 2
                )
                ** 1.5,
            )

        cp_0 = 3991.86795711963
        sst = f1 * vs.sst_clim[:, :, n1] + f2 * vs.sst_clim[:, :, n2]
        qnec = f1 * vs.qnec[:, :, n1] + f2 * vs.qnec[:, :, n2]
        qnet = f1 * vs.qnet[:, :, n1] + f2 * vs.qnet[:, :, n2]
        vs.forc_temp_surface = (
            (qnet + qnec * (sst - vs.temp[:, :, -1, vs.tau]))
            * vs.maskT[:, :, -1]
            / cp_0
            / settings.rho_0
        )

        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface = (
            1.0
            / t_rest
            * (sss - vs.salt[:, :, -1, vs.tau])
            * vs.maskT[:, :, -1]
            * vs.dzt[-1]
        )

        mask = npx.logical_and(
            vs.temp[:, :, -1, vs.tau] * vs.maskT[:, :, -1] < -1.8,
            vs.forc_temp_surface < 0.0,
        )
        vs.forc_temp_surface = npx.where(mask, 0.0, vs.forc_temp_surface)
        vs.forc_salt_surface = npx.where(mask, 0.0, vs.forc_salt_surface)

        return KernelOutput(
            surface_taux=vs.surface_taux,
            surface_tauy=vs.surface_tauy,
            forc_tke_surface=vs.forc_tke_surface,
            forc_temp_surface=vs.forc_temp_surface,
            forc_salt_surface=vs.forc_salt_surface,
        )

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
