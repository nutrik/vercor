"""Minimal setup adapters for the differentiable Veros execution lane."""

from __future__ import annotations

from veros import veros_routine
from veros.setups.acc import ACCSetup
from veros.variables import Variable

from vercor.setups._external.veros_setup import CustomGlobalFourDegree


class DifferentiableGlobalFourDegree(CustomGlobalFourDegree):
    """Externally forced global setup with a fixed runtime PyTree structure."""

    @veros_routine
    def set_forcing(self, state):  # type: ignore[no-untyped-def]
        """Apply the JAX-safe external-atmosphere forcing kernel."""

        state.variables.update(self.set_forcing_kernel(state))

    @veros_routine
    def set_diagnostics(self, state):  # type: ignore[no-untyped-def]
        """Disable diagnostics whose lazy accumulators change the state PyTree."""

        state.diagnostics.clear()


class DifferentiableACCSetup(ACCSetup):
    """ACC setup adapted to fork-owned traced turbulence variables."""

    @veros_routine
    def set_parameter(self, state):  # type: ignore[no-untyped-def]
        """Configure ACC without assigning variable-owned physical parameters."""

        settings = state.settings
        settings.identifier = "acc"
        settings.description = "Differentiable ACC setup"

        settings.nx, settings.ny, settings.nz = 30, 42, 15
        settings.dt_mom = 4800
        settings.dt_tracer = 86400 / 2.0
        settings.runlen = 86400 * 365

        settings.x_origin = 0.0
        settings.y_origin = -40.0
        settings.coord_degree = True
        settings.enable_cyclic_x = True

        settings.enable_neutral_diffusion = True
        settings.K_iso_0 = 1000.0
        settings.K_iso_steep = 500.0
        settings.iso_dslope = 0.005
        settings.iso_slopec = 0.01
        settings.enable_skew_diffusion = True

        settings.enable_hor_friction = True
        settings.A_h = (2 * settings.degtom) ** 3 * 2e-11
        settings.enable_hor_friction_cos_scaling = True
        settings.hor_friction_cosPower = 1
        settings.enable_bottom_friction = True
        settings.enable_implicit_vert_friction = True

        settings.enable_tke = True
        settings.alpha_tke = 30.0
        settings.mxl_min = 1e-8
        settings.tke_mxl_choice = 2
        settings.kappaM_min = 2e-4
        settings.kappaH_min = 2e-5
        settings.enable_kappaH_profile = True

        settings.enable_eke = True
        settings.eke_k_max = 1e4
        settings.eke_c_k = 0.4
        settings.eke_c_eps = 0.5
        settings.eke_cross = 2.0
        settings.eke_crhin = 1.0
        settings.eke_lmin = 100.0
        settings.enable_eke_superbee_advection = True
        settings.enable_eke_isopycnal_diffusion = True

        settings.enable_streamfunction = False
        settings.enable_idemix = False
        settings.eq_of_state_type = 3

        state.var_meta.update(
            t_star=Variable(
                "t_star",
                ("yt",),
                "deg C",
                "Reference surface temperature",
            ),
            t_rest=Variable(
                "t_rest",
                ("xt", "yt"),
                "1/s",
                "Surface temperature restoring time scale",
            ),
        )

    @veros_routine
    def set_diagnostics(self, state):  # type: ignore[no-untyped-def]
        """Disable diagnostics whose lazy accumulators change the state PyTree."""

        state.diagnostics.clear()


__all__ = ["DifferentiableACCSetup", "DifferentiableGlobalFourDegree"]
