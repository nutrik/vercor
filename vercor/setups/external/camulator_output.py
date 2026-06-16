"""CAMulator NetCDF output helpers backed by VerCOR's h5netcdf writer."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from importlib import resources
from os.path import expandvars
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from vercor.jax_logging import LoggerLike, get_default_logger
from vercor.output.datasets import time_coordinate_variable
from vercor.output.netcdf import write_netcdf_dataset
from vercor.output.variables import OutputVariable

_TIME_NAME = "time"
_LEVEL_NAME = "level"
_LATITUDE_NAME = "latitude"
_LONGITUDE_NAME = "longitude"
_FORECAST_HOUR_NAME = "forecast_hour"
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


def build_camulator_output_variables(
    prediction: torch.Tensor,
    forecast_datetime: datetime,
    *,
    latitude: object,
    longitude: object,
    forecast_hour: int,
    metadata: Mapping[str, Any],
    conf: Mapping[str, Any],
) -> tuple[dict[str, OutputVariable], dict[str, OutputVariable]]:
    """Return coordinate and data variables for one CAMulator forecast increment."""

    data_conf = _mapping(conf.get("data", {}), name="data")
    model_conf = _mapping(conf.get("model", {}), name="model")
    levels = _configured_levels(data_conf, model_conf)
    upper_names = _string_sequence(data_conf.get("variables", ()), "data.variables")
    surface_names = _string_sequence(
        data_conf.get("surface_variables", ()),
        "data.surface_variables",
    )
    diagnostic_names = _string_sequence(
        data_conf.get("diagnostic_variables", ()),
        "data.diagnostic_variables",
    )

    values = _prediction_values(prediction)
    _validate_prediction_shape(
        values,
        levels=levels,
        n_upper_variables=len(upper_names),
        n_single_level_variables=len(surface_names) + len(diagnostic_names),
    )
    latitude_values = np.asarray(latitude)
    longitude_values = np.asarray(longitude)
    _validate_grid_coordinates(values, latitude_values, longitude_values)

    coordinate_variables = {
        _TIME_NAME: _with_metadata(
            time_coordinate_variable(forecast_datetime, time_dim=_TIME_NAME),
            _TIME_NAME,
            metadata,
        ),
        _LEVEL_NAME: _with_metadata(
            OutputVariable(
                (_LEVEL_NAME,),
                _level_values(data_conf, levels),
            ),
            _LEVEL_NAME,
            metadata,
        ),
        _LATITUDE_NAME: _with_metadata(
            OutputVariable((_LATITUDE_NAME,), latitude_values),
            _LATITUDE_NAME,
            metadata,
        ),
        _LONGITUDE_NAME: _with_metadata(
            OutputVariable((_LONGITUDE_NAME,), longitude_values),
            _LONGITUDE_NAME,
            metadata,
        ),
    }

    data_variables = _prediction_data_variables(
        values,
        upper_names=upper_names,
        single_level_names=surface_names + diagnostic_names,
        levels=levels,
        metadata=metadata,
    )
    data_variables[_FORECAST_HOUR_NAME] = OutputVariable(
        (),
        np.asarray(forecast_hour, dtype=np.int32),
    )
    return coordinate_variables, _filtered_data_variables(data_variables, conf)


def write_camulator_netcdf_increment(
    coordinate_variables: Mapping[str, OutputVariable],
    data_variables: Mapping[str, OutputVariable],
    *,
    init_str: str,
    forecast_hour: int,
    conf: Mapping[str, Any],
    logger: LoggerLike | None = None,
) -> str:
    """Write one CAMulator forecast increment to the configured NetCDF file."""

    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    save_forecast = predict_conf.get("save_forecast")
    if not save_forecast:
        raise KeyError("'save_forecast' missing in CAMulator config")

    save_location = os.path.join(str(save_forecast), init_str)
    os.makedirs(save_location, exist_ok=True)
    output = os.path.join(save_location, f"pred_{init_str}_{forecast_hour:03d}.nc")

    log = logger if logger is not None else get_default_logger()
    log.info(f"Writing output file:  {output:s}")
    write_netcdf_dataset(
        output=output,
        coordinate_variables=coordinate_variables,
        data_variables=data_variables,
        global_attrs={"Conventions": "CF-1.11"},
    )
    return output


def write_camulator_prediction_output(
    prediction: torch.Tensor,
    utc_datetime: datetime,
    *,
    latitude: object,
    longitude: object,
    init_str: str,
    lead_time_periods: int,
    forecast_hour: int,
    metadata: Mapping[str, Any],
    conf: Mapping[str, Any],
    state_transformer: Any | None,
    logger: LoggerLike | None = None,
) -> None:
    """Write one CAMulator prediction increment through the VerCOR output boundary."""

    _validate_supported_output_options(conf)
    output_prediction = _prediction_for_output(
        prediction,
        conf=conf,
        state_transformer=state_transformer,
    )
    lead_forecast_hour = int(lead_time_periods) * int(forecast_hour)
    coordinate_variables, data_variables = build_camulator_output_variables(
        output_prediction,
        utc_datetime,
        latitude=latitude,
        longitude=longitude,
        forecast_hour=lead_forecast_hour,
        metadata=metadata,
        conf=conf,
    )
    write_camulator_netcdf_increment(
        coordinate_variables,
        data_variables,
        init_str=init_str,
        forecast_hour=lead_forecast_hour,
        conf=conf,
        logger=logger,
    )


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
) -> torch.Tensor:
    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    if not bool(predict_conf.get("climate_rescale_output", False)):
        return prediction.detach().cpu()
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


def _prediction_values(prediction: torch.Tensor) -> np.ndarray:
    values = prediction.detach().cpu().numpy()
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
    if values.shape[0] != 1:
        raise ValueError(
            "CAMulator output currently writes one forecast time per file, "
            f"got {values.shape[0]} times."
        )
    return values


def _validate_prediction_shape(
    values: np.ndarray,
    *,
    levels: int,
    n_upper_variables: int,
    n_single_level_variables: int,
) -> None:
    expected_channels = levels * n_upper_variables + n_single_level_variables
    if values.shape[1] != expected_channels:
        raise ValueError(
            "CAMulator prediction channel count does not match the config: "
            f"expected {expected_channels}, got {values.shape[1]}."
        )


def _validate_grid_coordinates(
    values: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> None:
    if latitude.ndim != 1:
        raise ValueError(f"CAMulator latitude must be 1D, got shape {latitude.shape}.")
    if longitude.ndim != 1:
        raise ValueError(
            f"CAMulator longitude must be 1D, got shape {longitude.shape}."
        )
    if values.shape[-2:] != (latitude.shape[0], longitude.shape[0]):
        raise ValueError(
            "CAMulator prediction grid does not match latitude/longitude: "
            f"prediction={values.shape[-2:]}, "
            f"coordinates={(latitude.shape[0], longitude.shape[0])}."
        )


def _prediction_data_variables(
    values: np.ndarray,
    *,
    upper_names: tuple[str, ...],
    single_level_names: tuple[str, ...],
    levels: int,
    metadata: Mapping[str, Any],
) -> dict[str, OutputVariable]:
    variables: dict[str, OutputVariable] = {}
    for index, name in enumerate(upper_names):
        start = index * levels
        stop = start + levels
        variables[name] = _with_metadata(
            OutputVariable(
                (_TIME_NAME, _LEVEL_NAME, _LATITUDE_NAME, _LONGITUDE_NAME),
                values[:, start:stop, :, :],
            ),
            name,
            metadata,
        )

    single_level_start = len(upper_names) * levels
    for index, name in enumerate(single_level_names):
        variables[name] = _with_metadata(
            OutputVariable(
                (_TIME_NAME, _LATITUDE_NAME, _LONGITUDE_NAME),
                values[:, single_level_start + index, :, :],
            ),
            name,
            metadata,
        )
    return variables


def _filtered_data_variables(
    data_variables: Mapping[str, OutputVariable],
    conf: Mapping[str, Any],
) -> dict[str, OutputVariable]:
    predict_conf = _mapping(conf.get("predict", {}), name="predict")
    save_vars = predict_conf.get("save_vars")
    if save_vars is None or len(save_vars) == 0:
        return dict(data_variables)

    selected = set(_string_sequence(save_vars, "predict.save_vars"))
    return {
        name: variable
        for name, variable in data_variables.items()
        if name == _FORECAST_HOUR_NAME or name in selected
    }


def _with_metadata(
    variable: OutputVariable,
    name: str,
    metadata: Mapping[str, Any],
) -> OutputVariable:
    if name == _TIME_NAME or name not in metadata:
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
    "build_camulator_output_variables",
    "load_camulator_output_metadata",
    "write_camulator_netcdf_increment",
    "write_camulator_prediction_output",
]
