"""CAMulator CREDIT output helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import torch

make_xarray: Any | None = None
save_netcdf_increment: Any | None = None


def _credit_output_functions() -> tuple[Any, Any]:
    """Load CREDIT output helpers when CAMulator writes a forecast increment."""

    global make_xarray, save_netcdf_increment

    if make_xarray is not None and save_netcdf_increment is not None:
        return make_xarray, save_netcdf_increment

    try:
        from credit.output import (  # type: ignore[import-not-found]
            make_xarray as loaded_make_xarray,
            save_netcdf_increment as loaded_save_netcdf_increment,
        )
    except ModuleNotFoundError as error:
        raise ImportError(
            "CREDIT output helpers are required to write CAMulator forecasts. "
            "Please install credit to use CAMulator output."
        ) from error

    make_xarray = loaded_make_xarray
    save_netcdf_increment = loaded_save_netcdf_increment
    return make_xarray, save_netcdf_increment


def write_camulator_prediction_output(
    prediction: torch.Tensor,
    utc_datetime: datetime,
    *,
    latitude: object,
    longitude: object,
    init_str: str,
    lead_time_periods: int,
    forecast_hour: int,
    metadata: dict[str, Any],
    conf: dict[str, Any],
) -> None:
    """Write one CAMulator prediction increment through the CREDIT output boundary."""

    credit_make_xarray, credit_save_netcdf_increment = _credit_output_functions()
    upper_air, single_level = credit_make_xarray(
        prediction.cpu(),
        utc_datetime,
        latitude,
        longitude,
        conf,
    )
    credit_save_netcdf_increment(
        upper_air,
        single_level,
        init_str,
        lead_time_periods * forecast_hour,
        metadata,
        conf,
    )
