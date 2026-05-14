from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

import jax
from jax.typing import ArrayLike

from vercor.dtypes import as_jax_real_array
from setups.external.camulator_state import (
    load_camulator_forcing_context,
    parse_datetime_from_config,
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
    start_ix: int = 0
    init_str: str = ""
    timestep_counter: int = 0


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
        state.coupling_timestep = timedelta(seconds=context.dt_seconds)

        state.model_timestep = timedelta(hours=state.lead_time_periods)
        state.model_substeps = int(
            state.coupling_timestep.total_seconds()
            // state.model_timestep.total_seconds()
        )

        if state.coupling_timestep % state.model_timestep != timedelta(days=0):
            raise ValueError(
                f"model_timestep ({state.model_timestep}) must be a "
                f"multiple of coupling_timestep ({state.coupling_timestep})"
            )

        state.dynamic_ds = state.forcing_ds[
            [
                "TS",
            ]
        ]

        # IMPORTANT: Use the config's datetime object directly for xarray lookup
        # It might be cftime.DatetimeNoLeap, which xarray expects
        start_datetime_raw = state.conf["predict"]["start_datetime"]
        loc = state.dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
        state.start_ix = loc.start if isinstance(loc, slice) else loc
        logger.info(f"Starting integration at time index: {state.start_ix}")

        # Now convert to Python datetime for output formatting (if it's a string or cftime)
        init_dt = parse_datetime_from_config(state.conf)
        state.init_str = init_dt.strftime("%Y-%m-%dT%HZ")

        if state.coupler_start_datetime != init_dt:
            logger.warning(
                f"Coupler start datetime ({state.coupler_start_datetime}) does not match "
                f"CAMulator forcing start datetime ({start_datetime_raw}). "
                f"Using CAMulator start datetime for indexing."
            )

        state.timestep_counter = 0

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

        idx = state.start_ix + state.timestep_counter * state.model_substeps
        dynamic_ds = cast(Any, state.dynamic_ds)
        ts = dynamic_ds.isel(time=idx).load()

        state.timestep_counter += 1

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
