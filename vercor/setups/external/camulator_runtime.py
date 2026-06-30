"""CAMulator host-runtime stepping and prediction block helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import torch

from vercor.components import ComponentStepContext
from vercor.jax_logging import LoggerLike
import vercor.setups.external.camulator_fields as _camulator_fields
import vercor.setups.external.camulator_output as _camulator_output
import vercor.setups.external.camulator_tensors as _camulator_tensors
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.setups.external.camulator_gcm_state import CAMulatorGCMSetupState


def coerce_camulator_datetime(time_obj: Any) -> datetime:
    """Return a Python datetime from CAMulator/xarray time coordinates."""

    if hasattr(time_obj, "item"):
        time_obj = time_obj.item()

    if isinstance(time_obj, datetime):
        return time_obj

    return datetime(
        time_obj.year,
        time_obj.month,
        time_obj.day,
        time_obj.hour,
        time_obj.minute,
        time_obj.second,
    )


def run_camulator_prediction_block(
    state: "CAMulatorGCMSetupState",
    fields: Mapping[str, Any],
    *,
    block_start: int,
    block_end: int,
    logger: LoggerLike | None,
) -> tuple[torch.Tensor, RuntimeArray]:
    """Run one CAMulator forcing block and return the final prediction and TS."""

    prediction = None
    last_total_surface_temperature: RuntimeArray | None = None

    ds_slice = state.dynamic_ds.isel(time=slice(block_start, block_end)).load()
    ds_slice_times = ds_slice["time"].values

    dynamic_forcing_chunk = _camulator_fields.prepare_camulator_dynamic_forcing_chunk(
        ds_slice.to_array(dim="dynamic_variable").values
    )
    gpu_forcing_chunk = _camulator_tensors.torch_tensor_from_jax_array(
        dynamic_forcing_chunk[:, :, jnp.newaxis, :, :],
        state.device,
        pin_memory=True,
    )

    for t in range(gpu_forcing_chunk.shape[0]):
        utc_datetime = coerce_camulator_datetime(ds_slice_times[t])

        if logger is not None:
            logger.info(
                "    CAMulator step: " f"{state.forecast_hour:05}, time: {utc_datetime}"
            )

        dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

        if state.forecast_hour != 1:
            model_input = state.stepper.state_manager.build_input_with_forcing(
                state.state,
                dynamic_forcing_t,
                state.static_forcing,
            )
        else:
            model_input = state.state

        total_ts, rescaled_total_ts = (
            _camulator_fields.prepare_camulator_surface_forcing(
                fields["sea_surface_temperature"],
                fields["land_surface_temperature"],
                state.LANDM_COSLAT,
            )
        )
        last_total_surface_temperature = total_ts

        if logger is not None:
            logger.info(
                "    Rescaled ts stats - max: "
                f"{float(jnp.max(rescaled_total_ts)):.4f}, min: "
                f"{float(jnp.min(rescaled_total_ts)):.4f}"
            )

        state.accessor_input.set_state_var(
            model_input,
            "SST",
            _camulator_tensors.torch_tensor_from_jax_array(
                _camulator_fields.prepare_camulator_sst_input(rescaled_total_ts),
                state.device,
            ),
        )

        with torch.no_grad():
            prediction = state.stepper.model(model_input.float())

        prediction = state.stepper._apply_postprocessing(prediction, model_input)

        record_camulator_prediction_output(
            state,
            prediction=prediction,
            utc_datetime=utc_datetime,
            logger=logger,
        )

        state.state = state.stepper.state_manager.shift_state_forward(
            state.state,
            prediction,
        )
        state.forecast_hour += 1

    if prediction is None or last_total_surface_temperature is None:
        raise ValueError(
            "No CAMulator timesteps were generated from the forcing slice; "
            "check forcing availability and coupling timestep alignment."
        )

    return cast(torch.Tensor, prediction), last_total_surface_temperature


def record_camulator_prediction_output(
    state: "CAMulatorGCMSetupState",
    *,
    prediction: torch.Tensor,
    utc_datetime: datetime,
    logger: LoggerLike | None,
) -> None:
    """Write CAMulator increment output or record period-average output."""

    output_frequency = getattr(state, "output_frequency", None)
    if output_frequency is None:
        _camulator_output.write_camulator_prediction_output(
            prediction,
            utc_datetime,
            latitude=state.latlons.latitude.values,
            longitude=state.latlons.longitude.values,
            init_str=state.runtime_cursor.init_str,
            lead_time_periods=state.lead_time_periods,
            forecast_hour=state.forecast_hour,
            metadata=state.metadata,
            conf=state.conf,
            state_transformer=state.state_transformer,
            logger=logger,
        )
        return

    _camulator_output.record_camulator_period_output(
        state.output_adapter,
        prediction,
        output_time=utc_datetime,
        dt=timedelta(hours=state.lead_time_periods),
        output_frequency=output_frequency,
        latitude=state.latlons.latitude.values,
        longitude=state.latlons.longitude.values,
        init_str=state.runtime_cursor.init_str,
        metadata=state.metadata,
        conf=state.conf,
        state_transformer=state.state_transformer,
        logger=logger,
    )


def step_camulator_runtime(
    state: "CAMulatorGCMSetupState",
    fields: Mapping[str, Any],
    context: ComponentStepContext,
    payload: Any | None,
) -> Mapping[str, Any]:
    """Advance the private host-backed CAMulator atmosphere boundary."""

    _ = payload
    time = context.time
    logger = context.logger
    if time is None:
        return {}

    block_start = state.runtime_cursor.current_index()
    block_end = block_start + state.model_substeps

    prediction, last_total_surface_temperature = run_camulator_prediction_block(
        state,
        fields,
        block_start=block_start,
        block_end=block_end,
        logger=logger,
    )
    state.runtime_cursor.advance()

    mapped_fields = _camulator_fields.map_camulator_prediction_to_runtime_fields(
        context.settings,
        camulator_reference_pressure=state.P0,
        hyai=state.hyai,
        hybi=state.hybi,
        hyam=state.hyam,
        hybm=state.hybm,
        accessor_output=state.accessor_output,
        state_transformer=state.state_transformer,
        prediction=prediction,
    )

    return {
        "total_surface_temperature": last_total_surface_temperature,
        **mapped_fields,
    }


__all__ = [
    "coerce_camulator_datetime",
    "record_camulator_prediction_output",
    "run_camulator_prediction_block",
    "step_camulator_runtime",
]
