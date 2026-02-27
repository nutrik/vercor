from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import tree_math
import xarray as xr

from dinosaur import primitive_equations
from dinosaur.coordinate_systems import CoordinateSystem

from jcm.constants import p0
from jcm.forcing import default_forcing
from jcm.model import ForcingData, Model, Predictions
from jcm.physics.speedy.physics_data import PhysicsData
from jcm.physics_interface import (
    PhysicsState,
    TerrainData,
    dynamics_state_to_physics_state,
)

from vercor.clock import ModelDateTime
from vercor.components.base import Component
from vercor.components.external.jax_gcm_tools import (
    mean_leaf,
    stack_objects,
    unwrap_leading_dims,
    get_altitudes_sigma_levels,
    compute_pressure_levels,
)
from vercor.fluxes.utilities import (
    compute_air_density,
    compute_potential_temperature,
)
from vercor.grid import RectilinearGrid


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def asfloat(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda arr: arr.astype(jnp.float_), tree)


@tree_math.struct
@dataclass
class JCMState:
    prog: PhysicsState
    phydata: Any
    metadata: primitive_equations.State


class JAXGCM(Component):
    """JCM Wrapper"""

    _predictions_list: list[Predictions]
    _step_function: Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]
    _state: JCMState
    forcing: ForcingData

    def __init__(
        self,
        coords: CoordinateSystem,
        terrain: TerrainData,
        name: str = "ATM",
        model_timestep: timedelta = timedelta(minutes=30),
        save_interval: timedelta = timedelta(days=1),
        spinup_time: timedelta = timedelta(days=2),
        forcing_data: Optional[ForcingData] = None,
        # Output frequency in days for saving JCM predictions.
        output_frequency: Optional[str] = None,
        do_spinup: bool = False,
        jitted: bool = True,
    ) -> None:

        self.model = Model(
            coords, time_step=model_timestep.total_seconds() / 60.0, terrain=terrain
        )
        self.forcing_data = forcing_data
        self.output_frequency = output_frequency
        self.model_timestep = model_timestep
        self.save_interval = save_interval
        self.spinup_time = spinup_time
        self.do_spinup = do_spinup
        self.jitted = jitted

        hgrid = self.model.coords.horizontal
        grid = RectilinearGrid(
            name=name,
            longitude=np.rad2deg(hgrid.longitudes),
            latitude=np.rad2deg(hgrid.latitudes),
            binary_mask=np.ones_like(
                self.model.terrain.fmask
            ).transpose(),  # This is used for interpolation, which all points are valid
        )

        self.sigma_levels: NDArray = self.model.coords.vertical.centers

        super().__init__(name, grid)

    def _generate_step_function(
        self, jitted: bool = True
    ) -> Callable[[JCMState, ForcingData], tuple[JCMState, Predictions]]:
        def step_function(
            state: JCMState, forcing: ForcingData
        ) -> tuple[JCMState, Predictions]:
            new_atm_modal_state, predictions = self.model.run_from_state(
                initial_state=state.metadata,
                save_interval=self.save_interval / timedelta(days=1),
                total_time=self.coupling_timestep / timedelta(days=1),
                forcing=forcing,
            )

            # phydata is a stacked object, so I take the mean here.
            # However, this action will be done by jcm in the new jcm PR.
            return (
                JCMState(
                    prog=asfloat(mean_leaf(predictions.dynamics, axis=0)),
                    phydata=asfloat(mean_leaf(predictions.physics, axis=0)),
                    metadata=new_atm_modal_state,
                ),
                predictions,
            )

        return jax.jit(step_function) if jitted else step_function

    def do_jcm_steps(self) -> tuple[Any, Any]:
        _avg_predictions = []
        _predictions: Predictions

        _new_state, _predictions = self._step_function(
            self._state,
            self.forcing,
        )

        self._state = _new_state

        _avg_predictions.append(_predictions)

        self._predictions_list.append(_predictions)

        _avg_predictions = mean_leaf(
            unwrap_leading_dims(stack_objects(_avg_predictions)), axis=0
        )

        return _avg_predictions.physics, _avg_predictions.dynamics

    def initialize(self, coupler: "Coupler") -> None:
        self.coupling_timestep = timedelta(seconds=coupler.clock.dt_seconds)
        self.spinup_steps = int(
            self.spinup_time.total_seconds() // self.coupling_timestep.total_seconds()
        )

        if self.coupling_timestep % self.model_timestep != timedelta(days=0):
            raise ValueError(
                f"model_timestep ({self.model_timestep}) must be a "
                f"multiple of coupling_timestep ({self.coupling_timestep})"
            )

        _modal_state = self.model._prepare_initial_modal_state()
        self._state = JCMState(
            metadata=_modal_state,
            phydata=PhysicsData.zeros(
                self.model.coords.horizontal.nodal_shape,
                self.model.coords.vertical.layers,
            ),
            prog=dynamics_state_to_physics_state(_modal_state, self.model.primitive),
        )

        if self.forcing_data is not None:
            self.forcing = self.forcing_data
        else:
            self.forcing = default_forcing(self.model.coords.horizontal).copy(
                lfluxland=True
            )

        self._step_function = self._generate_step_function(jitted=self.jitted)

        grid_shape = self.grid.shape

        zeros = np.zeros(grid_shape)
        self.data["specific_humidity"] = zeros.copy()
        self.data["net_shortwave_radiation_flux"] = zeros.copy()
        self.data["downward_longwave_radiation_flux"] = zeros.copy()
        self.data["sea_surface_temperature"] = zeros.copy() + 273.15 + 15.0
        self.data["land_surface_temperature"] = zeros.copy()
        self.data["u_velocity"] = zeros.copy()
        self.data["v_velocity"] = zeros.copy()
        self.data["temperature"] = zeros.copy()
        self.data["potential_temperature"] = zeros.copy()
        self.data["density"] = zeros.copy()
        self.data["latent_heat_flux"] = zeros.copy()
        self.data["sensible_heat_flux"] = zeros.copy()
        self.data["model_level_height"] = zeros.copy()

        self._predictions_list = []

        if self.do_spinup and "OCN" in coupler.run_sequence.order:
            coupler.logger.info(
                f" Performing JCM spinup for {self.spinup_time} day(s)..."
            )
            # Spin-up from the default JCM forcing
            for i in range(self.spinup_steps):
                coupler.logger.info(f" JCM spinup step {i+1} / {self.spinup_steps}")
                _, _ = self.do_jcm_steps()

    def step(
        self,
        dt: timedelta,
        time: datetime | ModelDateTime,
        coupler: "Coupler",
    ) -> None:
        settings = coupler.settings

        logger = coupler.logger

        logger.info(
            " Mean of SST: {}".format(
                float(jnp.nanmean(jnp.asarray(self.data["sea_surface_temperature"])))
            )
        )

        self.data["sea_surface_temperature"] = np.nan_to_num(
            self.data["sea_surface_temperature"], nan=0.0
        )
        self.data["land_surface_temperature"] = np.nan_to_num(
            self.data["land_surface_temperature"], nan=0.0
        )

        # Units: [K]
        self.data["total_surface_temperature"] = (
            self.data["land_surface_temperature"] + self.data["sea_surface_temperature"]
        )

        logger.info(
            " Number of cells with (SST + SKT) less than 250.0 K: {}".format(
                int(np.sum(self.data["total_surface_temperature"] < 250.0))
            ),
        )

        land_surface_temperature = (
            self.data["total_surface_temperature"]
            * self.model.terrain.fmask.transpose()
        )
        sea_surface_temperature = self.data["total_surface_temperature"] * (
            1.0 - self.model.terrain.fmask.transpose()
        )

        # Replace zero values with a default temperature (e.g., 288.15 K)
        # to avoid issues in JCM
        land_surface_temperature[land_surface_temperature == 0.0] = 288.15
        sea_surface_temperature[sea_surface_temperature == 0.0] = 288.15

        self.forcing = self.forcing.copy(
            stl_am=jnp.asarray(land_surface_temperature).transpose(),
            sea_surface_temperature=jnp.asarray(sea_surface_temperature).transpose(),
        )

        p, d = self.do_jcm_steps()

        # !!! All the heat and freshwater fluxes are positive upward !!!
        # Units: [m/s]
        self.data["u_velocity"] = np.array(d.u_wind[-1, :, :]).transpose()
        # Units: [m/s]
        self.data["v_velocity"] = np.array(d.v_wind[-1, :, :]).transpose()
        # Units: [K]
        self.data["temperature"] = np.array(d.temperature[-1, :, :]).transpose()
        # Units: [kg/kg] (converted from g/kg)
        self.data["specific_humidity"] = (
            np.array(d.specific_humidity[-1, :, :]).transpose() / 1000.0
        )
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Turn negative to upward fluxes and positive to downward fluxes
        # to comply with ERA5 & Veros conventions
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Units: [W/m²]
        self.data["sensible_heat_flux"] = -(
            np.array(p.surface_flux.shf).sum(axis=2).transpose()
        )
        # Units: [W/m²]
        self.data["latent_heat_flux"] = -(
            np.array(p.surface_flux.evap / 1e3 * settings.latvap)
            .sum(axis=2)
            .transpose()
        )
        # Units: [W/m²]
        self.data["net_shortwave_radiation_flux"] = np.array(
            p.shortwave_rad.rsns
        ).transpose()
        # Units: [W/m²]
        self.data["downward_longwave_radiation_flux"] = np.array(
            p.surface_flux.rlds
        ).transpose()
        # Units: [Pa]
        self.data["pressure"] = np.array(
            compute_pressure_levels(
                jnp.asarray(p0),
                jnp.asarray(0.0),
                self.sigma_levels,
                d.normalized_surface_pressure[:, :].transpose(),
            )
        )
        # Units: [kg/m³]
        self.data["density"] = compute_air_density(
            settings, self.data["pressure"][-1, ...], self.data["temperature"]
        )
        # Units: [K]
        self.data["potential_temperature"] = compute_potential_temperature(
            settings, self.data["temperature"], self.data["pressure"][-1, ...]
        )
        # Units: [m]
        self.data["model_level_height"] = np.array(
            get_altitudes_sigma_levels(
                d.temperature.transpose((0, 2, 1))[::-1, :, :],
                jnp.asarray(self.data["pressure"][::-1, :, :]),
                d.specific_humidity.transpose((0, 2, 1))[::-1, :, :] / 1000.0,
            )
        )[1, :, :]

        if self._should_write_output(time=time, dt=dt):
            date_time = time.strftime("%Y-%m-%d")
            self._write_output(output=f"jcm.averages.{date_time}.nc")

    def _is_period_end(
        self,
        time: datetime | ModelDateTime,
        dt: timedelta,
        frequency: Literal["day", "month", "year"],
    ) -> bool:
        next_time = time + dt

        if frequency == "day":
            return (
                next_time.year != time.year
                or next_time.month != time.month
                or next_time.day != time.day
            )
        if frequency == "month":
            return next_time.year != time.year or next_time.month != time.month

        return next_time.year != time.year

    def _should_write_output(
        self,
        time: datetime | ModelDateTime,
        dt: timedelta,
    ) -> bool:
        if self.output_frequency is None:
            return True

        if not isinstance(self.output_frequency, str):
            return False

        frequency = self.output_frequency.lower()
        if frequency not in ("day", "month", "year"):
            return False

        return self._is_period_end(
            time=time,
            dt=dt,
            frequency=cast(Literal["day", "month", "year"], frequency),
        )

    def _write_output(self, output: str) -> None:
        ds = cast(
            xr.Dataset,
            xr.merge(
                [_prediction.to_xarray() for _prediction in self._predictions_list]
            ),
        )

        print(f"Output file: {output:s}")

        t_end = ds.time.isel(time=-1)
        ds.mean(dim="time", keep_attrs=True, keepdims=True).assign_coords(
            time=[t_end.values]
        ).transpose("time", "wvi_id", "hsg_level", "level", "lat", "lon").to_netcdf(
            output, engine="h5netcdf"
        )

        # Clear the predictions list after saving to disk to free up memory
        self._predictions_list = []
