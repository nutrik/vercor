"""CAMulator config loading and forcing-time cursor helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import xarray as xr
import yaml

from vercor.setups._time_helpers import runtime_forcing_index
from vercor.setups.external import camulator_imports


def load_camulator_forcing_context(config_path: str) -> dict[str, Any]:
    """Load CAMulator config and raw forcing without constructing the model."""

    camulator_imports.load_credit_modules()

    with open(config_path) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)

    conf = camulator_imports.credit_main_parser(
        conf,
        parse_training=False,
        parse_predict=True,
        print_summary=False,
    )
    conf["predict"]["mode"] = None

    forcing_file = conf["predict"]["forcing_file"]
    if not os.path.exists(forcing_file):
        raise FileNotFoundError(f"Forcing file not found: {forcing_file}")

    chunk_size = conf["data"].get("forcing_chunk_size", 32)
    forcing_ds = xr.open_dataset(forcing_file, chunks={"time": chunk_size})
    return {
        "conf": conf,
        "forcing_dataset_raw": forcing_ds.chunk({"time": chunk_size}),
    }


def parse_datetime_from_config(conf: dict[str, Any]) -> datetime:
    """Parse CAMulator start datetime values into Python ``datetime`` objects."""

    raw_dt = conf["predict"]["start_datetime"]

    if isinstance(raw_dt, str):
        return datetime.strptime(raw_dt, "%Y-%m-%d %H:%M:%S")
    if isinstance(raw_dt, datetime):
        return raw_dt
    return datetime(
        raw_dt.year,
        raw_dt.month,
        raw_dt.day,
        raw_dt.hour,
        raw_dt.minute,
        raw_dt.second,
    )


@dataclass(frozen=True)
class CAMulatorForcingCursor:
    """Time-index cursor for CAMulator forcing datasets."""

    start_ix: int
    init_datetime: datetime
    init_str: str


@dataclass
class CamulatorRuntimeCursor:
    """Mutable CAMulator forcing cursor shared by host setup adapters."""

    start_ix: int = 0
    init_datetime: datetime | None = None
    init_str: str = ""
    model_substeps: int = 0
    timestep_counter: int = 0

    def initialize(
        self,
        *,
        conf: dict[str, Any],
        dynamic_ds: Any,
        coupler_start_datetime: object,
        model_substeps: int,
        logger: Any,
    ) -> None:
        """Initialize forcing index metadata and reset the runtime counter."""

        cursor = initialize_camulator_forcing_cursor(
            conf=conf,
            dynamic_ds=dynamic_ds,
            coupler_start_datetime=coupler_start_datetime,
            logger=logger,
        )
        self.start_ix = cursor.start_ix
        self.init_datetime = cursor.init_datetime
        self.init_str = cursor.init_str
        self.model_substeps = int(model_substeps)
        self.timestep_counter = 0

    def current_index(self) -> int:
        """Return the current forcing index for this cursor."""

        return runtime_forcing_index(
            start_ix=self.start_ix,
            timestep_counter=self.timestep_counter,
            model_substeps=self.model_substeps,
        )

    def advance(self) -> None:
        """Advance the cursor by one runtime counter step."""

        self.timestep_counter += 1


def initialize_camulator_forcing_cursor(
    *,
    conf: dict[str, Any],
    dynamic_ds: Any,
    coupler_start_datetime: object,
    logger: Any,
) -> CAMulatorForcingCursor:
    """Initialize CAMulator forcing time indexing from config and xarray indexes."""

    start_datetime_raw = conf["predict"]["start_datetime"]
    loc = dynamic_ds.indexes["time"].get_loc(start_datetime_raw)
    start_ix = loc.start if isinstance(loc, slice) else int(loc)
    logger.info(f"Starting integration at time index: {start_ix}")

    init_datetime = parse_datetime_from_config(conf)
    init_str = init_datetime.strftime("%Y-%m-%dT%HZ")

    if coupler_start_datetime != init_datetime:
        logger.warning(
            f"Coupler start datetime ({coupler_start_datetime}) does not match "
            f"CAMulator forcing start datetime ({start_datetime_raw}). "
            f"Using CAMulator start datetime for indexing."
        )

    return CAMulatorForcingCursor(
        start_ix=start_ix,
        init_datetime=init_datetime,
        init_str=init_str,
    )


__all__ = [
    "CAMulatorForcingCursor",
    "CamulatorRuntimeCursor",
    "initialize_camulator_forcing_cursor",
    "load_camulator_forcing_context",
    "parse_datetime_from_config",
]
