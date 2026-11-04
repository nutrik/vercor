"""CAMulator NetCDF output helpers backed by VerCOR's h5netcdf writer."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from importlib import resources
from os.path import expandvars
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import yaml

from vercor.output import OutputContext, OutputFrame, SnapshotContext
from vercor.output._dataset import time_coordinate_variable, used_dimension_names
from vercor.output._netcdf import write_netcdf_dataset
from vercor.output import OutputVariable
from vercor.output._selection import (
    component_output_names,
    selected_output_names,
    with_runtime_fields,
)

if TYPE_CHECKING:
    from vercor.setups._external.camulator_gcm_state import CAMulatorRuntimePayload

CAMULATOR_TIME_DIM = "time"
_LEVEL_NAME = "level"
_LATITUDE_NAME = "latitude"
_LONGITUDE_NAME = "longitude"
_UNSUPPORTED_PREDICT_OPTIONS = (
    "interp_pressure",
    "ua_var_encoding",
    "surface_var_encoding",
    "pressure_var_encoding",
    "height_var_encoding",
)


def load_camulator_output_metadata(conf: Mapping[str, Any]) -> dict[str, Any]:
    """Load CAMulator output metadata from an explicit path or bundled filename."""

    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    configured_metadata = predict_conf.get("metadata") or "era5.yaml"
    metadata_path = expandvars(str(configured_metadata))
    if not os.path.dirname(metadata_path):
        metadata_path = str(resources.files("credit.metadata").joinpath(metadata_path))

    with Path(metadata_path).open(encoding="utf-8") as metadata_file:
        metadata = yaml.load(metadata_file, Loader=yaml.SafeLoader)

    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(
            f"CAMulator metadata file {metadata_path!r} must contain a mapping."
        )
    return dict(metadata)


def camulator_period_output_variables(
    prediction: torch.Tensor,
    *,
    metadata: Mapping[str, Any],
    conf: Mapping[str, Any],
    state_transformer: Any | None,
    variables: Sequence[str] = (),
    decoded_prediction: torch.Tensor | None = None,
) -> dict[str, OutputVariable]:
    """Return CAMulator prediction variables ready for period accumulation."""

    _validate_supported_output_options(conf)
    channels = _prediction_channels(conf)
    names = selected_output_names(channels, variables)
    output_prediction = _prediction_for_output(
        prediction,
        conf=conf,
        state_transformer=state_transformer,
        decoded_prediction=decoded_prediction,
    )
    expected_channels = sum(stop - start for start, stop, _ in channels.values())
    if output_prediction.shape[1] != expected_channels:
        raise ValueError(
            "CAMulator prediction channel count does not match the config: "
            f"expected {expected_channels}, got {output_prediction.shape[1]}."
        )
    data_variables = {}
    for name in names:
        start, stop, dims = channels[name]
        values = _prediction_values(
            output_prediction[:, start:stop], require_single_time=False
        )
        data_variables[name] = _with_metadata(
            OutputVariable(dims, values if _LEVEL_NAME in dims else values[:, 0]),
            name,
            metadata,
        )
    return data_variables


def _prediction_channels(
    conf: Mapping[str, Any],
) -> dict[str, tuple[int, int, tuple[str, ...]]]:
    """Map configured output names to tensor slices without touching values."""
    data_conf = _mapping(conf.get("data", {}), name="data")
    levels = _configured_levels(
        data_conf, _mapping(conf.get("model", {}), name="model")
    )
    upper = _string_sequence(data_conf.get("variables", ()), "data.variables")
    surface = _string_sequence(
        data_conf.get("surface_variables", ()), "data.surface_variables"
    )
    diagnostic = _string_sequence(
        data_conf.get("diagnostic_variables", ()), "data.diagnostic_variables"
    )
    channels: dict[str, tuple[int, int, tuple[str, ...]]] = {
        name: (
            index * levels,
            (index + 1) * levels,
            (CAMULATOR_TIME_DIM, _LEVEL_NAME, _LATITUDE_NAME, _LONGITUDE_NAME),
        )
        for index, name in enumerate(upper)
    }
    first_surface = len(upper) * levels
    channels.update(
        {
            name: (
                first_surface + index,
                first_surface + index + 1,
                (CAMULATOR_TIME_DIM, _LATITUDE_NAME, _LONGITUDE_NAME),
            )
            for index, name in enumerate((*surface, *diagnostic))
        }
    )
    return channels


def camulator_average_coordinate_variables(
    variables: Mapping[str, OutputVariable],
    *,
    output_time: datetime,
    latitude: object,
    longitude: object,
    metadata: Mapping[str, Any],
    conf: Mapping[str, Any],
) -> dict[str, OutputVariable]:
    """Return coordinates used by a CAMulator period-average dataset."""

    level_values = _configured_level_values(conf)
    latitude_values = np.asarray(latitude)
    longitude_values = np.asarray(longitude)
    coordinate_builders: dict[str, Callable[[], OutputVariable]] = {
        _LEVEL_NAME: lambda: OutputVariable((_LEVEL_NAME,), level_values),
        _LATITUDE_NAME: lambda: OutputVariable((_LATITUDE_NAME,), latitude_values),
        _LONGITUDE_NAME: lambda: OutputVariable((_LONGITUDE_NAME,), longitude_values),
    }
    coordinate_variables = {
        CAMULATOR_TIME_DIM: time_coordinate_variable(
            output_time,
            time_dim=CAMULATOR_TIME_DIM,
        )
    }
    for dim in used_dimension_names(variables, excluded_dims=(CAMULATOR_TIME_DIM,)):
        builder = coordinate_builders.get(dim)
        if builder is not None:
            coordinate_variables[dim] = _with_metadata(builder(), dim, metadata)
    return coordinate_variables


def camulator_average_data_variables(
    variables: Mapping[str, OutputVariable],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, OutputVariable]:
    """Return CAMulator data variables with configured metadata attached."""

    return {
        name: _with_metadata(variable, name, metadata)
        for name, variable in variables.items()
    }


class _CAMulatorOutputProvider:
    """Ordinary provider adapting the latest CAMulator prediction tensor."""

    def __init__(self, resources: Any, *, latest_only: bool = False) -> None:
        self._resources = resources
        self._latest_only = latest_only

    def sample(self, context: OutputContext) -> OutputFrame:
        """Extract the latest native prediction with CAMulator metadata."""

        payload = context.payload
        payload = _require_camulator_runtime_payload(payload)
        native, canonical = component_output_names(
            context, _prediction_channels(self._resources.conf)
        )
        if payload.output_prediction is None:
            raise ValueError("CAMulator output requires a completed prediction.")
        if not isinstance(context.time, datetime):
            raise TypeError("CAMulator output requires a datetime timestamp.")
        accumulator = None if self._latest_only else payload.output_accumulator
        counts = {}
        if not native:
            variables = {}
        elif accumulator is not None:
            variables = {
                name: OutputVariable(dims, values, dict(attrs))
                for name, dims, values, attrs in zip(
                    accumulator.names,
                    accumulator.dims,
                    accumulator.sum_values,
                    accumulator.attrs,
                    strict=True,
                )
                if name in native
            }
            counts = {
                name: value
                for name, value in zip(
                    accumulator.names, accumulator.counts, strict=True
                )
                if name in native
            }
            missing = next((name for name in native if name not in variables), None)
            if missing is not None:
                raise ValueError(
                    f"CAMulator native reduction is missing requested variable {missing!r}"
                )
        else:
            variables = camulator_period_output_variables(
                payload.output_prediction,
                metadata=self._resources.metadata,
                conf=self._resources.conf,
                state_transformer=self._resources.state_transformer,
                variables=native,
            )
        frame = OutputFrame(
            camulator_average_data_variables(
                variables, metadata=self._resources.metadata
            ),
            reduction_counts=counts,
            coordinates=camulator_average_coordinate_variables(
                variables,
                output_time=context.time,
                latitude=self._resources.latlons.latitude.values,
                longitude=self._resources.latlons.longitude.values,
                metadata=self._resources.metadata,
                conf=self._resources.conf,
            ),
            sample_dimension=None if accumulator is not None else CAMULATOR_TIME_DIM,
            time_dimension=CAMULATOR_TIME_DIM,
        )
        return with_runtime_fields(
            frame, context, canonical, horizontal_dims=(_LATITUDE_NAME, _LONGITUDE_NAME)
        )


def camulator_output_provider(
    resources: Any,
    *,
    latest_only: bool = False,
) -> _CAMulatorOutputProvider:
    """Return the native CAMulator provider installed by the setup factory."""

    return _CAMulatorOutputProvider(resources, latest_only=latest_only)


def write_camulator_snapshot_output(
    resources: Any,
    provider: _CAMulatorOutputProvider,
    context: SnapshotContext,
) -> None:
    """Write one final CAMulator prediction snapshot through the shared adapter."""

    _ = resources
    payload = context.payload
    payload = _require_camulator_runtime_payload(payload)
    if payload.output_prediction is None:
        return
    frame = provider.sample(
        OutputContext(
            component=context.component,
            state=context.state,
            payload=payload,
            step=0,
            time=context.time,
            dt=timedelta(0),
        )
    )
    write_netcdf_dataset(
        output=str(context.output_path),
        coordinate_variables=frame.coordinates,
        data_variables=frame.variables,
        global_attrs=frame.metadata or None,
        logger=context.logger,
    )


def _require_camulator_runtime_payload(payload: Any) -> "CAMulatorRuntimePayload":
    """Return a validated payload without creating an initialization import cycle."""

    from vercor.setups._external.camulator_gcm_state import CAMulatorRuntimePayload

    if not isinstance(payload, CAMulatorRuntimePayload):
        raise ValueError("CAMulator output requires a native runtime payload.")
    return payload


def _validate_supported_output_options(conf: Mapping[str, Any]) -> None:
    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    for option in _UNSUPPORTED_PREDICT_OPTIONS:
        if option in predict_conf:
            raise ValueError(
                f"CAMulator output option predict.{option} requires an xarray "
                "post-processing path and is not supported by the VerCOR NetCDF writer."
            )
    if bool(conf.get("use_ptype", False)):
        raise ValueError(
            "CAMulator output option use_ptype requires an xarray post-processing "
            "path and is not supported by the VerCOR NetCDF writer."
        )


def _prediction_for_output(
    prediction: torch.Tensor,
    *,
    conf: Mapping[str, Any],
    state_transformer: Any | None,
    decoded_prediction: torch.Tensor | None = None,
) -> torch.Tensor:
    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    if not bool(predict_conf.get("climate_rescale_output", False)):
        return prediction.detach()
    if decoded_prediction is not None:
        return decoded_prediction.detach()
    if state_transformer is None:
        raise ValueError(
            "CAMulator output requires state_transformer when "
            "predict.climate_rescale_output is enabled."
        )
    transformed = cast(
        torch.Tensor,
        state_transformer.inverse_transform(prediction.detach().cpu()),
    )
    return transformed.detach().cpu()


def _prediction_values(
    prediction: torch.Tensor,
    *,
    require_single_time: bool = True,
) -> np.ndarray:
    values = np.asarray(prediction.detach().cpu().numpy())
    if values.ndim == 5:
        if values.shape[2] != 1:
            raise ValueError(
                "CAMulator output predictions must contain one time slice in "
                f"axis 2, got shape {values.shape}."
            )
        values = values[:, :, 0, :, :]
    if values.ndim != 4:
        raise ValueError(
            "CAMulator output predictions must have shape "
            "(time, channels, latitude, longitude) or "
            f"(time, channels, 1, latitude, longitude), got {values.shape}."
        )
    if require_single_time and values.shape[0] != 1:
        raise ValueError(
            "CAMulator output currently writes one forecast time per file, "
            f"got {values.shape[0]} times."
        )
    return values


def _with_metadata(
    variable: OutputVariable,
    name: str,
    metadata: Mapping[str, Any],
) -> OutputVariable:
    if name == CAMULATOR_TIME_DIM or name not in metadata:
        return variable
    attrs = metadata[name]
    if not isinstance(attrs, Mapping):
        return variable
    return OutputVariable(
        variable.dims,
        variable.values,
        {
            **dict(variable.attrs),
            **{attr_name: attr_value for attr_name, attr_value in attrs.items()},
        },
    )


def _level_values(data_conf: Mapping[str, Any], levels: int) -> np.ndarray:
    raw_level_ids = data_conf.get("level_ids")
    if raw_level_ids is None:
        return np.arange(levels, dtype=np.int32)

    level_values = np.asarray(list(raw_level_ids))
    if level_values.shape != (levels,):
        raise ValueError(
            f"CAMulator level_ids must contain {levels} entries, got {level_values.shape}."
        )
    return level_values


def _configured_level_values(conf: Mapping[str, Any]) -> np.ndarray:
    data_conf = _mapping(conf.get("data", {}), name="data")
    model_conf = _mapping(conf.get("model", {}), name="model")
    levels = _configured_levels(data_conf, model_conf)
    return _level_values(data_conf, levels)


def _configured_levels(
    data_conf: Mapping[str, Any],
    model_conf: Mapping[str, Any],
) -> int:
    raw_levels = model_conf.get("levels", data_conf.get("levels"))
    if raw_levels is None:
        raise KeyError("'levels' missing in CAMulator model/data config")
    levels = int(raw_levels)
    if levels <= 0:
        raise ValueError(f"CAMulator levels must be positive, got {levels}.")
    return levels


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise ValueError(f"CAMulator config section {name!r} must be a mapping.")


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"CAMulator config entry {name} must be a sequence.")
    try:
        sequence = tuple(value)
    except TypeError as error:
        raise ValueError(
            f"CAMulator config entry {name} must be a sequence."
        ) from error
    if not all(isinstance(entry, str) for entry in sequence):
        raise ValueError(f"CAMulator config entry {name} must contain strings.")
    return sequence


__all__ = [
    "CAMULATOR_TIME_DIM",
    "camulator_average_coordinate_variables",
    "camulator_average_data_variables",
    "camulator_period_output_variables",
    "load_camulator_output_metadata",
    "camulator_output_provider",
    "write_camulator_snapshot_output",
]
