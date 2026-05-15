from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

import jax
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from setups._time_helpers import assign_model_timestep_alignment
from setups.external.camulator_state import (
    CamulatorRuntimeCursor,
    load_camulator_forcing_context,
)

from vercor.grid import RectilinearGrid
from vercor.components.base import (
    ComponentStepContext,
    HostRuntimeComponent,
    host_component,
)
from vercor.runtime.contexts import ComponentInitContext
from vercor.grid_masks import create_lnd_mask_from_ocn

_CAMULATOR_LAND_OUTPUTS = ("land_surface_temperature",)
_CAMULATOR_LAND_DEFAULT_FIELDS = {"land_surface_temperature": 283.0}


def _prepare_camulator_land_surface_temperature(
    land_surface_temperature: ArrayLike,
) -> jax.Array:
    """Normalize CAMulator land temperature fields for JAX-backed runtime storage."""
    return as_jax_real_array(land_surface_temperature)


@dataclass
class _CAMulatorLandState:
    config_path: str
    conf: Any
    forcing_ds: Any
    lead_time_periods: Any
    coupler_start_datetime: Any | None = None
    coupling_timestep: timedelta | None = None
    model_timestep: timedelta | None = None
    model_substeps: int = 0
    dynamic_ds: Any | None = None
    runtime_cursor: CamulatorRuntimeCursor = field(
        default_factory=CamulatorRuntimeCursor
    )


def make_camulator_land(
    config_path: str,
    camulator_grid: RectilinearGrid,
    ocn_grid: RectilinearGrid,
    name: str = "LND",
) -> HostRuntimeComponent:
    """Return a host-backed CAMulator land forcing component."""

    longitude = camulator_grid.longitude
    latitude = camulator_grid.latitude
    lnd_bmask, _ = create_lnd_mask_from_ocn(
        atm_lat=latitude,
        atm_lon=longitude,
        ocn_grid=ocn_grid,
    )

    forcing_context = load_camulator_forcing_context(config_path=config_path)
    state = _CAMulatorLandState(
        config_path=config_path,
        conf=forcing_context["conf"],
        forcing_ds=forcing_context["forcing_dataset_raw"],
        lead_time_periods=forcing_context["conf"]["data"]["lead_time_periods"],
    )

    grid = RectilinearGrid(
        name=f"{name.lower()}-grid",
        longitude=longitude,
        latitude=latitude,
        binary_mask=lnd_bmask,
    )

    def initialize(
        component: HostRuntimeComponent,
        context: ComponentInitContext,
    ) -> None:
        logger = context.logger
        state.coupler_start_datetime = context.start
        assign_model_timestep_alignment(
            state,
            context.dt_seconds,
            timedelta(hours=state.lead_time_periods),
        )

        state.dynamic_ds = state.forcing_ds[
            [
                "TS",
            ]
        ]

        # IMPORTANT: Use the config's datetime object directly for xarray lookup
        # It might be cftime.DatetimeNoLeap, which xarray expects
        state.runtime_cursor.initialize(
            conf=state.conf,
            dynamic_ds=state.dynamic_ds,
            coupler_start_datetime=state.coupler_start_datetime,
            model_substeps=state.model_substeps,
            logger=logger,
        )

        component.seed_declared_defaults(context.settings)

    def step(
        fields: dict[str, Any],
        context: ComponentStepContext,
        payload: Any | None,
    ) -> dict[str, Any]:
        _ = fields, payload
        time = context.time
        if time is None:
            return {}

        idx = state.runtime_cursor.current_index()
        dynamic_ds = cast(Any, state.dynamic_ds)
        ts = dynamic_ds.isel(time=idx).load()

        state.runtime_cursor.advance()

        return {
            "land_surface_temperature": _prepare_camulator_land_surface_temperature(
                ts["TS"].values
            )
        }

    return host_component(
        name=name,
        grid=grid,
        step=step,
        outputs=_CAMULATOR_LAND_OUTPUTS,
        default_fields=_CAMULATOR_LAND_DEFAULT_FIELDS,
        initialize=initialize,
    )
