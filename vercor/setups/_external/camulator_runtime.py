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

    if getattr(getattr(time_obj, "dtype", None), "kind", None) == "M":
        # Nanosecond datetime scalars return integer epochs from item().
        # Python datetime instead needs the scalar's microsecond representation.
        time_obj = time_obj.astype("datetime64[us]").item()
    elif hasattr(time_obj, "item"):
        time_obj = time_obj.item()

    if time_obj is None:
        raise ValueError("CAMulator forcing time must be a valid datetime, not NaT.")
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
    ds_slice = resources.dynamic_ds.isel(time=slice(block_start, block_end))
    requested_steps = block_end - block_start
    if requested_steps <= 0 or ds_slice.sizes["time"] != requested_steps:
        raise ValueError(
            "CAMulator forcing block must contain every requested native timestep; "
            f"requested {requested_steps}, available {ds_slice.sizes['time']}."
        )
    ds_slice_times = tuple(
        coerce_camulator_datetime(value) for value in ds_slice["time"].values
    )
    previous_time = None
    if payload.cursor.init_datetime is not None:
        previous_time = (
            payload.cursor.init_datetime
            + (block_start - payload.cursor.start_ix - 1) * resources.model_timestep
        )
    elif block_start > payload.cursor.start_ix:
        previous_time = coerce_camulator_datetime(
            resources.dynamic_ds["time"].isel(time=block_start - 1).values
        )
    for current_time in ds_slice_times:
        if (
            previous_time is not None
            and current_time - previous_time != resources.model_timestep
        ):
            raise ValueError(
                "CAMulator forcing time spacing must match the native model timestep "
                f"({resources.model_timestep})."
            )
        previous_time = current_time

    ds_slice = ds_slice.load()
    model_state = payload.model_state.clone()
    forecast_hour = payload.forecast_hour

    dynamic_forcing_chunk = _camulator_fields.prepare_camulator_dynamic_forcing_chunk(
        ds_slice.to_array(dim="dynamic_variable").values
    )
    gpu_forcing_chunk = _camulator_tensors.torch_tensor_from_jax_array(
        dynamic_forcing_chunk[:, :, jnp.newaxis, :, :],
        resources.device,
        pin_memory=True,
    )

    for t in range(gpu_forcing_chunk.shape[0]):
        utc_datetime = ds_slice_times[t]

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
                _camulator_fields.prepare_camulator_surface_temperature_input(
                    rescaled_total_ts
                ),
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
    """Export endpoint meteorology and complete-window mean radiation rates.

    Every native forecast window has the same duration. Map each prediction
    before averaging radiation so native J/m² normalization and nonlinear
    surface emission reconstruction occur at that prediction's temperature.
    Native state and output history retain their original endpoint/samples.
    """

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
    predictions = cast(torch.Tensor, next_payload.output_prediction_samples)
    mapped_samples = [
        _camulator_fields.map_camulator_prediction_to_runtime_fields(
            context.constants,
            camulator_reference_pressure=resources.P0,
            native_step_seconds=resources.model_timestep.total_seconds(),
            water_vapor_variable=resources.water_vapor_variable,
            hyai=resources.hyai,
            hybi=resources.hybi,
            hyam=resources.hyam,
            hybm=resources.hybm,
            accessor_output=resources.accessor_output,
            state_transformer=resources.state_transformer,
            prediction=prediction,
        )
        for prediction in predictions.split(1, dim=0)
    ]
    mapped_fields = {
        **mapped_samples[-1],
        **{
            name: jnp.mean(
                jnp.stack([sample[name] for sample in mapped_samples]), axis=0
            )
            for name in (
                "net_shortwave_radiation_flux",
                "downward_longwave_radiation_flux",
            )
        },
    }

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
