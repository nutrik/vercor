from copy import deepcopy

from collections.abc import Mapping
from typing import Any, Callable, cast
import jax
import jax.numpy as jnp
from datetime import timedelta

from setups._time_helpers import align_model_timestep
from setups.external.veros_runtime_settings import configure_veros_runtime
from vercor.components.base import (
    ComponentStepContext,
    HostRuntimeComponent,
    host_component,
)
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.fluxes.bulk_formula_cesm import compute_ocean_surface_fluxes
from vercor.runtime import RuntimeFieldStore
from vercor.runtime.contexts import ComponentInitContext
from vercor.settings import VercorSettings
from vercor.host_arrays import runtime_array_to_host
from vercor.types import RuntimeArray

configure_veros_runtime()

from veros.setups.global_4deg import GlobalFourDegreeSetup  # noqa: E402
from veros.core.operators import numpy as npx, update, at  # noqa: E402
from veros.routines import veros_kernel, veros_routine  # noqa: E402
from veros.state import KernelOutput, VerosState  # noqa: E402
from veros.tools import get_periodic_interval  # noqa: E402

try:
    import veros  # noqa: F401
except ImportError:
    raise ImportError(
        "The VerosGCM component requires the Veros package. Please install it with `pip install veros`."
    )


_VEROS_INPUT_FIELD_NAMES = (
    "model_level_height",
    "u_velocity",
    "v_velocity",
    "potential_temperature",
    "specific_humidity",
    "density",
    "temperature",
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
_VEROS_FIELD_DEFAULTS = {"sea_surface_temperature": 283.15}


class CustomGlobalFourDegree(GlobalFourDegreeSetup):
    @veros_kernel
    def set_forcing_kernel(state):  # type: ignore
        vs = state.variables
        settings = state.settings

        year_in_seconds = 365 * 86400.0
        (n1, f1), (n2, f2) = get_periodic_interval(
            vs.time, year_in_seconds, year_in_seconds / 12.0, 12
        )

        # wind stress
        vs.surface_taux = f1 * vs.taux[:, :, n1] + f2 * vs.taux[:, :, n2]
        vs.surface_tauy = f1 * vs.tauy[:, :, n1] + f2 * vs.tauy[:, :, n2]

        # tke flux
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

        # heat flux : W/m^2 K kg/J m^3/kg = K m/s
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

        # salinity restoring
        t_rest = 30 * 86400.0
        sss = f1 * vs.sss_clim[:, :, n1] + f2 * vs.sss_clim[:, :, n2]
        vs.forc_salt_surface = (
            1.0
            / t_rest
            * (sss - vs.salt[:, :, -1, vs.tau])
            * vs.maskT[:, :, -1]
            * vs.dzt[-1]
        )

        # apply simple ice mask
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


@jax.jit
def _update_veros_interior(
    array: object,
    interior_value: object,
) -> jax.Array:
    array_jax = as_jax_real_array(array)
    interior_value_jax = as_jax_real_array(interior_value)
    return array_jax.at[2:-2, 2:-2, ...].set(interior_value_jax)


@jax.jit
def _prepare_surface_forcing_fields(
    taux: object,
    tauy: object,
    qnet: object,
    qnec: object,
    restore_to_climatology: object,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    restore_to_climatology_jax = jnp.asarray(restore_to_climatology, dtype=bool)

    def _prepare(field: object) -> jax.Array:
        field_jax = as_jax_real_array(field)
        return jnp.nan_to_num(field_jax.T[..., jnp.newaxis])

    taux_prepared = _prepare(taux)
    tauy_prepared = _prepare(tauy)
    qnet_prepared = _prepare(qnet)
    qnec_prepared = _prepare(qnec)
    qnec_prepared = jnp.where(
        restore_to_climatology_jax, qnec_prepared, jnp.zeros_like(qnec_prepared)
    )

    return taux_prepared, tauy_prepared, qnet_prepared, qnec_prepared


@jax.jit
def _extract_surface_temperature(
    temperature: object,
    tau: object,
) -> jax.Array:
    temperature_array = as_jax_real_array(temperature)
    tau_index = as_jax_index_array(tau)
    return temperature_array[2:-2, 2:-2, -1, tau_index].T + 273.15


def compute_fluxes(
    veros_state: VerosState,
    runtime_fields: RuntimeFieldStore,
    settings: VercorSettings,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute atmosphere-ocean fluxes from explicit Veros and runtime fields."""

    vs = veros_state.variables

    # u & v have Arakawa-C grid staggering in Veros
    # require additional interpolation
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
        as_jax_real_array(runtime_fields.get("model_level_height"), settings),
        as_jax_real_array(runtime_fields.get("u_velocity"), settings),
        as_jax_real_array(runtime_fields.get("v_velocity"), settings),
        as_jax_real_array(runtime_fields.get("potential_temperature"), settings),
        as_jax_real_array(runtime_fields.get("specific_humidity"), settings),
        as_jax_real_array(runtime_fields.get("density"), settings),
        as_jax_real_array(runtime_fields.get("temperature"), settings),
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )

    # Signs & directions convention in Veros
    # Negative out:        ↑  LW_up ↑  SENf ↑  LATf ↑
    # Positive in:  SW_net ↓  LW_dw ↓  SENf ↓  LATf ↓

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


def copy_state(tree: VerosState, jitted: bool = True) -> VerosState:
    if jitted:
        dimensions = deepcopy(tree._dimensions)
        settings_meta = deepcopy(tree.settings.__metadata__)
        plugin_interfaces = deepcopy(tree._plugin_interfaces)
        var_meta = deepcopy(tree._var_meta)

        state_copy = VerosState(
            var_meta, settings_meta, dimensions, plugin_interfaces=plugin_interfaces
        )

        with state_copy.settings.unlock():
            for k, v in tree.settings.items():
                state_copy.settings.__setattr__(k, v)

        state_copy._variables = deepcopy(tree._variables)
        state_copy.timers = deepcopy(tree.timers)
        state_copy.profile_timers = deepcopy(tree.profile_timers)
    else:
        state_copy = tree

    return state_copy


def pure(state: VerosState, jitted: bool, step: Callable) -> VerosState:
    """
    Convert an in-place Veros step into a copy-before-mutate boundary helper.
    """
    n_state = copy_state(state, jitted=jitted)
    # This is a function that modifies state object inplace
    step(n_state)

    return n_state


def _extract_veros_runtime_sst(state: VerosState) -> jax.Array:
    """Return the Veros surface temperature field in VerCOR runtime layout."""

    return cast(
        jax.Array,
        _extract_surface_temperature(
            state.variables.temp,
            state.variables.tau,
        ),
    )


def set_variable(
    state: VerosState,
    variable_name: str,
    variable_value: RuntimeArray,
    jitted: bool = True,
) -> VerosState:
    n_state = copy_state(state, jitted=jitted)
    vs = n_state.variables

    with n_state.variables.unlock():
        var = getattr(vs, variable_name)
        updated_var = _update_veros_interior(var, variable_value)
        setattr(vs, variable_name, runtime_array_to_host(updated_var))

    return n_state


def _apply_veros_forcing_fields(
    state: VerosState,
    forcing_fields: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    *,
    jitted: bool,
) -> VerosState:
    """Write prepared VerCOR forcing fields into Veros state variables."""

    updated_state = state
    for variable_name, variable_value in zip(
        ("taux", "tauy", "qnet", "qnec"),
        forcing_fields,
    ):
        updated_state = set_variable(
            updated_state,
            variable_name,
            variable_value,
            jitted=jitted,
        )
    return updated_state


def _advance_veros_substeps(
    state: VerosState,
    *,
    step_function: Callable[[VerosState], VerosState],
    model_substeps: int,
    logger: Any | None,
) -> VerosState:
    """Advance Veros through the configured number of host substeps."""

    updated_state = state
    for i in range(model_substeps):
        if logger is not None:
            logger.info(f" Veros sub-step {i+1} / {model_substeps}")
        updated_state = step_function(updated_state)
    return updated_state


class _VerosGCMState:
    name: str
    data: dict[str, RuntimeArray]
    settings: VercorSettings

    def __init__(
        self,
        name: str = "OCN",
        spinup_time: timedelta = timedelta(days=2),
        custom_parameters: dict[str, Any] | None = None,
        restore_to_climatology: bool = False,
        do_spinup: bool = False,
        jitted: bool = False,
    ) -> None:
        """
        Veros GCM component based on the Global 4-degree setup from Veros.

        Arguments:
            name (str): component name
            spinup_time (timedelta): duration of the initial Veros spinup
            custom_parameters (dict[str, Any]): dictionary of custom parameter values to override
                                                the default settings in the GlobalFourDegreeSetup
            restore_to_climatology (bool): whether to apply restoring to climatology in
                                           the surface temperature (add salinity later) tendency
            do_spinup (bool): whether to perform the initial spinup with ERA-Interim forcing
            jitted (bool): whether to use JIT compilation for the Veros model step function
        """

        self.name = name
        override = custom_parameters or {}

        self.model = CustomGlobalFourDegree(override=override)
        self.model.setup()
        self._veros_state = copy_state(self.model.state, jitted=jitted)
        self._step_function = lambda state: pure(
            state, jitted=jitted, step=self.model.step
        )

        self.do_spinup = do_spinup
        self.spinup_time = spinup_time
        self.restore_to_climatology = restore_to_climatology
        self.jitted = jitted

        self.dt_tracer = getattr(self._veros_state.settings, "dt_tracer")
        self.spinup_steps = int(self.spinup_time.total_seconds() // self.dt_tracer)

        mask = jnp.where(
            jnp.asarray(self._veros_state.variables.maskT[:, :, -1]) > 0.0,
            1.0,
            0.0,
        )

        grid = RectilinearGrid(
            name=name,
            longitude=self._veros_state.variables.xt[2:-2],
            latitude=self._veros_state.variables.yt[2:-2],
            binary_mask=mask[2:-2, 2:-2].T,
        )

        self.grid = grid

    def initialize(
        self,
        component: HostRuntimeComponent,
        context: ComponentInitContext,
    ) -> None:
        dt_seconds = context.dt_seconds
        alignment = align_model_timestep(
            dt_seconds,
            timedelta(seconds=float(self.dt_tracer)),
            coupling_name="dt",
            model_name="dt_tracer",
        )
        self.model_substeps = alignment.model_substeps

        # Initial spinup is performed with ERA-Interim (default) atmospheric forcing
        if self.do_spinup and "ATM" in context.run_sequence.order:
            # Do it similar to CESM spinup when coupling with atmosphere is on
            context.logger.info(
                f" Performing Veros spinup for {self.spinup_time} day(s)..."
            )
            for i in range(self.spinup_steps):
                context.logger.info(f" Step {i+1} / {self.spinup_steps}")
                self._veros_state = self._step_function(self._veros_state)

        component.seed_field(
            "sea_surface_temperature",
            _extract_veros_runtime_sst(self._veros_state),
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> Mapping[str, Any]:
        """Advance the private host-backed Veros boundary."""

        _ = payload
        time = context.time
        logger = context.logger
        if time is None:
            return {}

        runtime_fields = RuntimeFieldStore.from_mapping(fields)

        taux, tauy, qnet, qnec = compute_fluxes(
            self._veros_state,
            runtime_fields,
            context.settings,
        )
        forcing_fields = _prepare_surface_forcing_fields(
            taux, tauy, qnet, qnec, self.restore_to_climatology
        )

        self._veros_state = _apply_veros_forcing_fields(
            self._veros_state,
            forcing_fields,
            jitted=self.jitted,
        )
        self._veros_state = _advance_veros_substeps(
            self._veros_state,
            step_function=self._step_function,
            model_substeps=self.model_substeps,
            logger=logger,
        )

        return {
            "sea_surface_temperature": _extract_veros_runtime_sst(self._veros_state)
        }


def make_veros_gcm(
    name: str = "OCN",
    spinup_time: timedelta = timedelta(days=2),
    custom_parameters: dict[str, Any] | None = None,
    restore_to_climatology: bool = False,
    do_spinup: bool = False,
    jitted: bool = False,
) -> HostRuntimeComponent:
    """Return a host-backed Veros GCM component."""

    state = _VerosGCMState(
        name=name,
        spinup_time=spinup_time,
        custom_parameters=custom_parameters,
        restore_to_climatology=restore_to_climatology,
        do_spinup=do_spinup,
        jitted=jitted,
    )
    defaults = {
        field_name: _VEROS_FIELD_DEFAULTS.get(field_name, 0.0)
        for field_name in ("sea_surface_temperature",)
    }
    return host_component(
        name=name,
        grid=state.grid,
        step=state.step,
        inputs=_VEROS_INPUT_FIELD_NAMES,
        outputs=("sea_surface_temperature",),
        default_fields=defaults,
        initialize=state.initialize,
    )
