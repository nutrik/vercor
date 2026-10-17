"""Actual small Veros grid for source-only ocean boundary validation.

Use native setup, forcing and thermodynamics kernels with zero initial flow,
zero diffusivity and no mixing closures. The column remains horizontally
uniform on wet cells so advection contributes zero. No solver or tracer source
is replaced by an analytical stand-in. Imports follow runtime configuration so
stock NumPy and the exact JAX fork can use this fixture in separate processes.
"""

from __future__ import annotations

from typing import Any, Literal


def surface_setup_class(
    *,
    execution: Literal["host", "jax"] = "host",
    temperature: float = -2.0,
    dt: float = 600.0,
    top_depth: float = 10.0,
) -> type[Any]:
    """Define a real 4x4, two-level native setup with a single dry cell."""
    from vercor.setups._external.veros_runtime_settings import configure_veros_runtime

    configure_veros_runtime(execution)
    from veros import veros_routine
    from veros.core.operators import numpy as npx, update, at
    from vercor.setups._external.veros_setup import CustomGlobalFourDegree

    class SurfaceColumn(CustomGlobalFourDegree):
        @veros_routine
        def set_parameter(self, state: Any) -> None:
            super().set_parameter(state)
            settings = state.settings
            settings.identifier = "surface_source_test"
            settings.nx, settings.ny, settings.nz = 4, 4, 2
            settings.dt_mom = settings.dt_tracer = dt
            settings.coord_degree = True
            settings.x_origin, settings.y_origin = 0.0, -2.0
            settings.enable_cyclic_x = True
            settings.enable_streamfunction = False
            settings.enable_neutral_diffusion = False
            settings.enable_skew_diffusion = False
            settings.enable_hor_friction = False
            settings.enable_hor_friction_cos_scaling = False
            settings.enable_implicit_vert_friction = False
            settings.enable_tke = settings.enable_eke = settings.enable_idemix = False
            settings.enable_conserve_energy = False
            settings.eq_of_state_type = 1

        @veros_routine
        def set_grid(self, state: Any) -> None:
            vs = state.variables
            vs.dxt = npx.full_like(vs.dxt, 1.0)
            vs.dyt = npx.full_like(vs.dyt, 1.0)
            vs.dzt = npx.array([30.0, top_depth])

        @veros_routine
        def set_coriolis(self, state: Any) -> None:
            state.variables.coriolis_t = npx.zeros_like(state.variables.coriolis_t)

        @veros_routine
        def set_topography(self, state: Any) -> None:
            vs = state.variables
            vs.kbot = update(npx.ones_like(vs.kbot), at[2, 2], 0)

        @veros_routine
        def set_initial_conditions(self, state: Any) -> None:
            vs = state.variables
            vs.temp = temperature * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.temp)
            vs.salt = 35.0 * vs.maskT[..., npx.newaxis] * npx.ones_like(vs.salt)
            vs.sst_clim = npx.full_like(vs.sst_clim, 0.0)
            vs.sss_clim = npx.full_like(vs.sss_clim, 36.0)
            vs.qnet = npx.zeros_like(vs.qnet)
            vs.qnec = npx.full_like(vs.qnec, 4.0)
            vs.taux = npx.zeros_like(vs.taux)
            vs.tauy = npx.zeros_like(vs.tauy)
            vs.kappaH = npx.zeros_like(vs.kappaH)

        @veros_routine
        def set_diagnostics(self, state: Any) -> None:
            state.diagnostics.clear()

    return SurfaceColumn


def surface_model(
    *,
    execution: Literal["host", "jax"] = "host",
    restore_temp: bool = False,
    restore_salt: bool = False,
    temperature: float = -2.0,
    dt: float = 600.0,
    top_depth: float = 10.0,
) -> Any:
    """Initialize the actual small native model without external data assets."""
    setup = surface_setup_class(
        execution=execution, temperature=temperature, dt=dt, top_depth=top_depth
    )
    model = setup(
        restore_to_climatology_temp=restore_temp,
        restore_to_climatology_salt=restore_salt,
    )
    model.setup()
    return model


def feedback_fields(
    *, heat: float = -20.0, mass: float = 1e-4, salt: float = 0.0
) -> dict[str, Any]:
    """Return total-area downward input with zero stress on a 4x4 domain."""
    import jax.numpy as jnp

    return {
        name: jnp.full((4, 4), value)
        for name, value in (
            ("sea_surface_heat_flux", heat),
            ("sea_surface_mass_flux", mass),
            ("sea_surface_salt_mass_flux", salt),
            ("ocean_u_stress", 0.0),
            ("ocean_v_stress", 0.0),
        )
    }
