from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, cast

from vercor.components import (
    ComponentSetupContext,
    ComponentStepContext,
    HostRuntimeComponent,
    host_component,
)
from vercor.dtypes import as_jax_real_array
from vercor.grid import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.setups._time_helpers import assign_model_timestep_alignment
from vercor.setups.external.camulator_forcing import (
    CamulatorRuntimeCursor,
    load_camulator_forcing_context,
)

_CAMULATOR_LAND_OUTPUTS = ("land_surface_temperature",)
_CAMULATOR_LAND_DEFAULT_FIELDS = {"land_surface_temperature": 283.0}


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
        context: ComponentSetupContext,
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

        # Use the config datetime directly because xarray may expect cftime.
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

        return {"land_surface_temperature": as_jax_real_array(ts["TS"].values)}

    return host_component(
        name=name,
        grid=grid,
        step=step,
        outputs=_CAMULATOR_LAND_OUTPUTS,
        default_fields=_CAMULATOR_LAND_DEFAULT_FIELDS,
        initialize=initialize,
    )
