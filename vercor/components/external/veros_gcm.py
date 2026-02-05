from copy import deepcopy

from typing import TYPE_CHECKING, Callable
import numpy as np
from numpy.typing import NDArray
from datetime import datetime, timedelta

from vercor.components.external.veros_runtime_settings import *  # noqa: F403,F401

from veros.setups.global_4deg import GlobalFourDegreeSetup
from veros.core.operators import update, at
from veros.state import VerosState

from vercor.components.base import Component
from vercor.grid import RectilinearGrid
from vercor.fluxes.bulk_formula_cesm import new_flux_atmOcn
from vercor.settings import VercorSettings


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def compute_fluxes(
    component_state: "VerosGCM", settings: VercorSettings
) -> tuple[NDArray, NDArray, NDArray, NDArray]:

    cs = component_state
    vs = cs._veros_state.variables

    # u & v have Arakawa-C grid staggering in Veros
    # require additional interpolation
    u_tgrid = 0.5 * (vs.u[1:, 2:-2, -1, vs.tau] + vs.u[:-1, 2:-2, -1, vs.tau])
    v_tgrid = 0.5 * (vs.v[2:-2, 1:, -1, vs.tau] + vs.v[2:-2, :-1, -1, vs.tau])

    temp = vs.temp[2:-2, 2:-2, -1, vs.tau].T + 273.15

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
    ) = new_flux_atmOcn(
        settings,
        np.array(vs.maskT[2:-2, 2:-2, -1].T),
        cs.data["model_level_height"],
        cs.data["u_velocity"],
        cs.data["v_velocity"],
        cs.data["potential_temperature"],
        cs.data["specific_humidity"],
        cs.data["density"],
        cs.data["temperature"],
        u_tgrid[1:-2, :].T,
        v_tgrid[:, 1:-2].T,
        temp,
    )

    # Signs & directions convention in Veros
    # Negative out:        ↑  LW_up ↑  SENf ↑  LATf ↑
    # Positive in:  SW_net ↓  LW_dw ↓  SENf ↓  LATf ↓

    qnet = (
        cs.data["net_shortwave_radiation_flux"]
        + cs.data["downward_longwave_radiation_flux"]
        + lwup
        + senf
        + latf
    )
    qnec = -np.where(dqfldt <= -1e10, 0.0, dqfldt)

    return (taux, tauy, qnet, qnec)


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

        # Replace the above with the following line when Etienne put his fixes in Veros
        # return tree_map(lambda x : x.copy(), tree)
    else:
        state_copy = tree

    return state_copy


def pure(state: VerosState, jitted: bool, step: Callable) -> VerosState:
    """
    Convert the state function into a "pure step" copying the input state
    """
    n_state = copy_state(state, jitted=jitted)
    # This is a function that modifies state object inplace
    step(n_state)

    return n_state


def set_variable(
    state: VerosState, variable_name: str, variable_value: NDArray, jitted: bool = True
) -> VerosState:

    n_state = copy_state(state, jitted=jitted)
    vs = n_state.variables

    with n_state.variables.unlock():
        var = getattr(vs, variable_name)
        var = update(var, at[2:-2, 2:-2, ...], variable_value)
        setattr(vs, variable_name, var)

    return n_state


class VerosGCM(Component):
    def __init__(
        self,
        name: str = "OCN",
        spinup_days: int = 2,
        do_spinup: bool = False,
        jitted: bool = False,
    ) -> None:
        """
        Veros GCM component based on the Global 4-degree setup from Veros.

        Arguments:
            name (str): component name
        """

        self.model = GlobalFourDegreeSetup()
        self.model.setup()
        self._veros_state = copy_state(self.model.state, jitted=jitted)
        self._step_function = lambda state: pure(
            state, jitted=jitted, step=self.model.step
        )

        seconds_per_day = 86400.0
        self.do_spinup = do_spinup
        self.spinup_days = spinup_days
        self.jitted = jitted

        self.dt_tracer = getattr(self._veros_state.settings, "dt_tracer")
        self.spinup_steps = int(seconds_per_day * self.spinup_days // self.dt_tracer)

        mask = np.where(self._veros_state.variables.maskT[:, :, -1] > 0.0, 1.0, 0.0)

        self.grid = RectilinearGrid(
            name=name,
            longitude=self._veros_state.variables.xt[2:-2],
            latitude=self._veros_state.variables.yt[2:-2],
            binary_mask=mask[2:-2, 2:-2].T,
        )

        super().__init__(name, grid=self.grid)

    def initialize(self, coupler: "Coupler") -> None:
        dt_seconds = coupler.clock.dt_seconds
        self.model_substeps = int(dt_seconds // self.dt_tracer)

        if dt_seconds % self.dt_tracer != 0:
            raise ValueError(
                f"dt_tracer ({self.dt_tracer}) must be a multiple of dt ({dt_seconds})"
            )

        # Initial spinup is performed with ERA-Interim (default) atmospheric forcing
        if self.do_spinup and "ATM" in coupler.run_sequence.order:
            # Do it similar to CESM spinup when coupling with atmosphere is on
            coupler.logger.info(
                f" Performing Veros spinup for {self.spinup_days} day(s)..."
            )
            for i in range(self.spinup_steps):
                coupler.logger.info(f" Step {i+1} / {self.spinup_steps}")
                self._veros_state = self._step_function(self._veros_state)

        # Units: [K]
        self.data["sea_surface_temperature"] = (
            self._veros_state.variables.temp[
                2:-2, 2:-2, -1, self._veros_state.variables.tau
            ].T
            + 273.15
        )

    def step(
        self,
        dt: timedelta,
        time: datetime,
        coupler: "Coupler",
    ) -> None:

        taux, tauy, qnet, qnec = compute_fluxes(self, coupler.settings)

        for variable_name, variable_value in {
            "taux": np.nan_to_num(taux.T[..., np.newaxis]),
            "tauy": np.nan_to_num(tauy.T[..., np.newaxis]),
            "qnet": np.nan_to_num(qnet.T[..., np.newaxis]),
            "qnec": np.nan_to_num(qnec.T[..., np.newaxis]),
        }.items():
            self._veros_state = set_variable(
                self._veros_state, variable_name, variable_value, jitted=self.jitted
            )

        for i in range(self.model_substeps):
            coupler.logger.info(f" Veros sub-step {i+1} / {self.model_substeps}")
            self._veros_state = self._step_function(self._veros_state)

        # Units: [K]
        self.data["sea_surface_temperature"] = (
            self._veros_state.variables.temp[
                2:-2, 2:-2, -1, self._veros_state.variables.tau
            ].T
            + 273.15
        )
