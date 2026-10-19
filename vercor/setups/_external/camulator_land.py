from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast

from vercor.components import (
    LifecycleHooks,
    CallableComponent,
    Component,
    ComponentSpec,
    ComponentMaskSpec,
    SetupResult,
    SetupContext,
    StepResult,
    StepContext,
)
from vercor.dtypes import as_jax_real_array
from vercor.grids import RectilinearGrid
from vercor.grid_masks import create_lnd_mask_from_ocn
from vercor.exceptions import ComponentError
from vercor.jax_logging import LoggerLike
from vercor.setups._time_helpers import assign_model_timestep_alignment
from vercor.setups._external.camulator_forcing import (
    CamulatorRuntimeCursor,
    load_camulator_forcing_context,
)

_CAMULATOR_LAND_INPUTS = (
    "net_shortwave_radiation_flux",
    "downward_longwave_radiation_flux",
)
_CAMULATOR_LAND_OUTPUTS = ("land_surface_temperature", "land_fraction")
_CAMULATOR_LAND_DEFAULT_FIELDS = {
    "land_surface_temperature": 283.0,
    **{field_name: 0.0 for field_name in _CAMULATOR_LAND_INPUTS},
}


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


def make_camulator_land(
    config_path: str,
    camulator_grid: RectilinearGrid,
    ocn_grid: RectilinearGrid,
    name: str = "LND",
    *,
    time_alignment: Literal["strict", "forcing_start"] = "strict",
) -> Component:
    """Return a host-backed CAMulator land forcing component."""

    longitude = camulator_grid.longitude
    latitude = camulator_grid.latitude
    lnd_bmask, land_fraction = create_lnd_mask_from_ocn(
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

    def setup(
        component: Component,
        context: SetupContext,
    ) -> SetupResult:
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
        cursor = CamulatorRuntimeCursor.initialize(
            conf=state.conf,
            dynamic_ds=state.dynamic_ds,
            coupler_start_datetime=state.coupler_start_datetime,
            model_substeps=state.model_substeps,
            logger=cast(LoggerLike, logger),
            time_alignment=time_alignment,
        )

        _ = component
        return SetupResult(payload=cursor)

    def step(
        fields: dict[str, Any],
        context: StepContext,
        payload: Any | None,
    ) -> StepResult:
        _ = fields
        if not isinstance(payload, CamulatorRuntimeCursor):
            raise ComponentError(
                "CAMulator land runtime requires an initialized immutable cursor "
                f"payload for component '{name}'"
            )
        time = context.time
        if time is None:
            return StepResult(payload=payload)

        idx = payload.current_index()
        dynamic_ds = cast(Any, state.dynamic_ds)
        ts = dynamic_ds.isel(time=idx).load()

        return StepResult(
            fields={"land_surface_temperature": as_jax_real_array(ts["TS"].values)},
            payload=payload.advanced(),
        )

    return CallableComponent(
        name,
        grid,
        step,
        spec=ComponentSpec(
            inputs=_CAMULATOR_LAND_INPUTS,
            outputs=_CAMULATOR_LAND_OUTPUTS,
            initial_fields={
                **_CAMULATOR_LAND_DEFAULT_FIELDS,
                "land_fraction": land_fraction,
            },
            masks=ComponentMaskSpec(fractional_field="land_fraction"),
            execution="host",
            lifecycle=LifecycleHooks(setup=setup),
        ),
    )
