"""CAMulator host-runtime stepping and prediction block helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import jax.numpy as jnp
import torch

from vercor.components import StepContext, StepResult
from vercor.exceptions import ComponentError
import vercor.setups._external.camulator_fields as _camulator_fields
from vercor.setups._external.camulator_gcm_state import CAMulatorRuntimePayload
import vercor.setups._external.camulator_tensors as _camulator_tensors
from vercor.types import RuntimeArray

if TYPE_CHECKING:
    from vercor.jax_logging import LoggerLike
    from vercor.setups._external.camulator_gcm_state import CAMulatorGCMSetupState


def _require_finite_torch_tensor(value: Any, *, owner: str) -> None:
    """Reject a non-Tensor or non-finite value at one native host boundary."""

    if not isinstance(value, torch.Tensor):
        raise ComponentError(
            f"{owner} must be a torch.Tensor; got {type(value).__name__}."
        )
    invalid_count = int(torch.count_nonzero(~torch.isfinite(value)).item())
    if invalid_count:
        raise ComponentError(f"{owner} contains {invalid_count} non-finite value(s).")


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
    resources: "CAMulatorGCMSetupState",
    fields: Mapping[str, Any],
    payload: CAMulatorRuntimePayload,
    *,
    block_start: int,
    block_end: int,
    logger: LoggerLike | None,
) -> tuple[CAMulatorRuntimePayload, RuntimeArray]:
    """Run one CAMulator forcing block and return its next payload and final TS."""

    prediction = None
    prediction_samples: list[torch.Tensor] = []
    last_total_surface_temperature: RuntimeArray | None = None
    model_state = payload.model_state.clone()
    forecast_hour = payload.forecast_hour

    ds_slice = resources.dynamic_ds.isel(time=slice(block_start, block_end)).load()
    ds_slice_times = ds_slice["time"].values

    dynamic_forcing_chunk = _camulator_fields.prepare_camulator_dynamic_forcing_chunk(
        ds_slice.to_array(dim="dynamic_variable").values
    )
    gpu_forcing_chunk = _camulator_tensors.torch_tensor_from_jax_array(
        dynamic_forcing_chunk[:, :, jnp.newaxis, :, :],
        resources.device,
        pin_memory=True,
    )

    for t in range(gpu_forcing_chunk.shape[0]):
        utc_datetime = coerce_camulator_datetime(ds_slice_times[t])

        if logger is not None:
            logger.info(
                "    CAMulator step: " f"{forecast_hour:05}, time: {utc_datetime}"
            )

        dynamic_forcing_t = gpu_forcing_chunk[t].unsqueeze(0)

        if forecast_hour != 1:
            model_input = resources.stepper.build_input_with_forcing(
                model_state,
                dynamic_forcing_t,
                resources.static_forcing,
            )
        else:
            model_input = model_state

        total_ts, rescaled_total_ts = (
            _camulator_fields.prepare_camulator_surface_forcing(
                fields["sea_surface_temperature"],
                fields["land_surface_temperature"],
                resources.LANDM_COSLAT,
            )
        )
        last_total_surface_temperature = total_ts

        if logger is not None:
            logger.info(
                "    Rescaled ts stats - max: "
                f"{float(jnp.max(rescaled_total_ts)):.4f}, min: "
                f"{float(jnp.min(rescaled_total_ts)):.4f}"
            )

        resources.accessor_input.set_state_var(
            model_input,
            "SST",
            _camulator_tensors.torch_tensor_from_jax_array(
                _camulator_fields.prepare_camulator_sst_input(rescaled_total_ts),
                resources.device,
            ),
        )

        model_input = model_input.float()
        _require_finite_torch_tensor(
            model_input,
            owner=f"CAMulator model input at forecast hour {forecast_hour}",
        )
        with torch.no_grad():
            prediction = resources.stepper.model(model_input)
        _require_finite_torch_tensor(
            prediction,
            owner=f"CAMulator raw prediction at forecast hour {forecast_hour}",
        )

        prediction = resources.stepper._apply_postprocessing(prediction, model_input)
        _require_finite_torch_tensor(
            prediction,
            owner=(
                "CAMulator postprocessed prediction at forecast hour "
                f"{forecast_hour}"
            ),
        )
        prediction_samples.append(prediction)

        next_model_state = resources.stepper.shift_state_forward(
            model_state,
            prediction,
        )
        _require_finite_torch_tensor(
            next_model_state,
            owner=f"CAMulator shifted model state at forecast hour {forecast_hour}",
        )
        model_state = next_model_state
        forecast_hour += 1

    if prediction is None or last_total_surface_temperature is None:
        raise ValueError(
            "No CAMulator timesteps were generated from the forcing slice; "
            "check forcing availability and coupling timestep alignment."
        )

    next_payload = replace(
        payload,
        model_state=model_state,
        forecast_hour=forecast_hour,
        output_prediction=cast(torch.Tensor, prediction),
        output_prediction_samples=torch.cat(prediction_samples, dim=0),
    )
    return next_payload, last_total_surface_temperature


def step_camulator_runtime(
    resources: "CAMulatorGCMSetupState",
    fields: Mapping[str, Any],
    context: StepContext,
    payload: Any | None,
) -> StepResult:
    """Advance the private host-backed CAMulator atmosphere boundary."""

    if not isinstance(payload, CAMulatorRuntimePayload):
        raise ComponentError("CAMulator runtime requires a native runtime payload.")
    time = context.time
    logger = context.logger
    if time is None:
        return StepResult(payload=payload)
    if not isinstance(time, datetime):
        raise TypeError("CAMulator runtime requires datetime clock values.")

    block_start = payload.cursor.current_index()
    block_end = block_start + resources.model_substeps

    next_payload, last_total_surface_temperature = run_camulator_prediction_block(
        resources,
        fields,
        payload,
        block_start=block_start,
        block_end=block_end,
        logger=logger,
    )
    next_payload = replace(next_payload, cursor=next_payload.cursor.advanced())
    prediction = cast(torch.Tensor, next_payload.output_prediction)

    mapped_fields = _camulator_fields.map_camulator_prediction_to_runtime_fields(
        context.constants,
        camulator_reference_pressure=resources.P0,
        hyai=resources.hyai,
        hybi=resources.hybi,
        hyam=resources.hyam,
        hybm=resources.hybm,
        accessor_output=resources.accessor_output,
        state_transformer=resources.state_transformer,
        prediction=prediction,
    )

    return StepResult(
        fields={
            "total_surface_temperature": last_total_surface_temperature,
            **mapped_fields,
        },
        payload=next_payload,
    )


__all__ = [
    "coerce_camulator_datetime",
    "run_camulator_prediction_block",
    "step_camulator_runtime",
]
