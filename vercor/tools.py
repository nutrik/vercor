from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Optional, Mapping
from urllib.request import urlopen

import numpy as np
from numpy.typing import NDArray

from vercor.clock import (
    DateTime360,
    DateTime365,
    ModelDateTime,
    _DAYS_PER_MONTH_GREGORIAN_NO_LEAP,
    _DAYS_PER_MONTH_GREGORIAN_LEAP,
)
from vercor.exceptions import AssetError, CouplerError
from vercor.grid import RectilinearGrid
from vercor.types import AllComponentsType


if TYPE_CHECKING:
    from vercor.coupler import Coupler


VERCOR_ASSETS_BASE_URL = (
    os.environ.get("VERCOR_ASSETS_BASE_URL")
    or "https://sid.erda.dk/share_redirect/bC5N6nQcbY/"
)

_ASSETS_CACHE_DIR = Path.home() / ".vercor" / "assets"

_FORCING_ASSETS: dict[str, dict[str, str]] = {
    "era5_model_levels": {
        "filename": "era5_198x_ml_4x4deg_monthly_mean.nc",
        "md5": "2ada464b2eb2bf3a7abec7f77a18634c",
    },
    "era5_surface": {
        "filename": "era5_198x_sfc_4x4deg_monthly_mean.nc",
        "md5": "304d547b72b3677f7bc44c71bcf7cb8f",
    },
    "era5_land": {
        "filename": "era5_lnd_skt_1980.nc",
        "md5": "b0877a7715c438b7a17593ad00bb8218",
    },
    "era5_land_masked": {
        "filename": "era5_lnd_skt_masked_1980.nc",
        "md5": "cea9349ee88f1ecb55572f87f065ff9b",
    },
    "erainterim_ocean": {
        "filename": "forcing_4deg_global_open_itf.nc",
        "md5": "cfcc6d8cde8da5a74ecec00309d92dd7",
    },
    "ecmwf_4deg_monthly": {
        "filename": "ecmwf_4deg_monthly_nc4.nc",
        "md5": "d1b4e0e199d7a5883cf7c88d3d6bcb27",
    },
}


def _md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_asset(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def _asset_base_url() -> str | None:
    base_url = VERCOR_ASSETS_BASE_URL
    if base_url is None:
        return None
    stripped = base_url.strip().rstrip("/")
    return stripped if stripped else None


def _ensure_forcing_asset(asset_key: str) -> Path:
    asset = _FORCING_ASSETS[asset_key]
    filename = asset["filename"]
    expected_md5 = asset["md5"]

    cached_path = _ASSETS_CACHE_DIR / filename
    if cached_path.exists():
        if _md5sum(cached_path) == expected_md5:
            return cached_path
        cached_path.unlink()

    _ASSETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    base_url = _asset_base_url()
    if base_url is None:
        raise AssetError(
            "Forcing asset not found in cache and no remote base URL configured. "
            "Set VERCOR_ASSETS_BASE_URL to a server hosting VerCOR forcing assets. "
            f"Missing asset: '{filename}'"
        )

    url = f"{base_url}/{filename}"
    try:
        _download_asset(url, cached_path)
    except Exception as e:
        raise AssetError(
            f"Failed to download forcing asset '{filename}' from '{url}': {e}"
        ) from e

    actual_md5 = _md5sum(cached_path)
    if actual_md5 != expected_md5:
        if cached_path.exists():
            cached_path.unlink()
        raise AssetError(
            f"MD5 mismatch for forcing asset '{filename}': expected {expected_md5}, got {actual_md5}"
        )

    return cached_path


def _flatten_fields(
    field_names: list[str | tuple[str, str]],
) -> list[str]:
    flattened: list[str] = []
    for item in field_names:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _append_unique(target: list[str], exchange_items: list[str]) -> None:
    """
    Arguments:
        target: list of field names in _fields2export & _fields2import
                in the component's exchange variable lists.
        exchange_items: list of field names from exchange rules (exchanger).
    """
    target.extend([item for item in exchange_items if item not in target])


def grids_identical(g0: RectilinearGrid, g1: RectilinearGrid) -> bool:
    return (
        g0.shape == g1.shape
        and np.allclose(g0.latitude, g1.latitude, atol=1e-15)
        and np.allclose(g0.longitude, g1.longitude, atol=1e-15)
    )


def get_component(
    allcomponents: dict[str, AllComponentsType], types: tuple[Any, ...]
) -> AllComponentsType:
    """Helper function to retrieve a registered component of a specific type
    from the coupler's allcomponents dictionary."""

    components: list[AllComponentsType] = [
        component
        for component in allcomponents.values()
        if isinstance(component, types)
    ]

    if len(components) > 1:
        raise CouplerError(
            f"Multiple {components[0].name} components registered; only one supported"
        )

    if not components:
        raise CouplerError(f"No component of types ({types}) registered")

    return components[0]


def get_periodic_interval(
    current_time: float, cycle_length: float, rec_spacing: float, n_rec: int
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """
    Ported from Veros: https://github.com/team-ocean/veros/blob/main/veros/tools/setup.py#L88

    Used for linear interpolation between periodic time intervals.

    One common application is the interpolation of external forcings that are defined
    at discrete times (e.g. one value per month of a standard year) to the current
    time step.

    Arguments:
       current_time (float): Time to interpolate to.
       cycle_length (float): Total length of one periodic cycle.
       rec_spacing (float): Time spacing between each data record.
       n_rec (int): Total number of records available.

    Returns:
       :obj:`tuple` containing (n1, f1), (n2, f2): Indices and weights for the interpolated
       record array.

    Example:
       The following interpolates a record array ``data`` containing 12 monthly values
       to the current time step:

       >>> year_in_seconds = 60. * 60. * 24. * 365.
       >>> current_time = 60. * 60. * 24. * 45. # mid-february
       >>> print(data.shape)
       (360, 180, 12)
       >>> (n1, f1), (n2, f2) = get_periodic_interval(current_time, year_in_seconds, year_in_seconds / 12, 12)
       >>> data_at_current_time = f1 * data[..., n1] + f2 * data[..., n2]

    """
    current_time = current_time % cycle_length
    # use np.array to compute integer indices for the time records
    t_idx_1 = np.array(current_time // rec_spacing, dtype="int")
    t_idx_2 = np.array((1 + t_idx_1) % n_rec, dtype="int")
    weight_2 = (current_time - rec_spacing * t_idx_1) / rec_spacing
    weight_1 = 1.0 - weight_2
    return (t_idx_1, weight_1), (t_idx_2, weight_2)


def datetime_to_seconds_in_year(dt: datetime | ModelDateTime) -> float:
    """Convert a datetime object to the number of seconds since the start of the year.

    Arguments:
        dt: datetime or ModelDateTime object to convert.

    Returns:
        float: Number of seconds since the start of the year.
    """

    if isinstance(dt, datetime):
        year_start = datetime(dt.year, 1, 1)
        return (dt - year_start).total_seconds()

    day_of_year = dt.day_of_year
    if day_of_year is None:
        raise ValueError("ModelDateTime.day_of_year is not initialized")

    return (
        (day_of_year - 1) * 86_400.0
        + dt.hour * 3_600.0
        + dt.minute * 60.0
        + dt.second
        + dt.microsecond / 1_000_000.0
    )


def get_forcing_data(file_type: str) -> Path:
    """Resolve forcing data to cached assets in $HOME/.vercor/assets.

    If needed, assets are downloaded from VERCOR_ASSETS_BASE_URL and validated by MD5.
    """

    if file_type not in _FORCING_ASSETS:
        allowed = ", ".join(sorted(_FORCING_ASSETS.keys()))
        raise AssetError(
            f"Unknown file_type '{file_type}'. Allowed values are: {allowed}"
        )

    return _ensure_forcing_asset(file_type)


def is_leap_year(x: int) -> bool:
    return (x % 4 == 0 and x % 100 != 0) or (x % 400 == 0)


def _custom_360_day_to_gregorian_day_of_year(
    time: datetime | ModelDateTime,
    no_leap: bool,
) -> int:
    month_lengths = _DAYS_PER_MONTH_GREGORIAN_NO_LEAP
    if not no_leap and is_leap_year(time.year):
        month_lengths = _DAYS_PER_MONTH_GREGORIAN_LEAP

    month_length = month_lengths[time.month - 1]
    mapped_day_in_month = ((time.day - 1) * (month_length - 1)) // 29 + 1
    days_before_month = sum(month_lengths[: time.month - 1])
    return days_before_month + mapped_day_in_month


def get_field_time_slice(
    field_name: str,
    data: Mapping[str, NDArray],
    time: datetime | ModelDateTime,
    no_leap: bool = True,
) -> NDArray:
    """Retrieve a field from a component data storage dictionary at a specific time index
    without applying time interpolation. The time index is determined based on the day of the year,
    with an option to ignore leap days (Feb 29) for leap years.

    Arguments:
        field_name: Name of the field to retrieve.
        data: Dictionary containing the component data with time-dependent fields.
        time: datetime or ModelDateTime object representing the time slice to retrieve.
        no_leap: Whether to ignore leap days (Feb 29) when indexing.

    Returns:
        NDArray: The field data at the specified time index.
    """

    if isinstance(time, DateTime360):
        tm_yday = _custom_360_day_to_gregorian_day_of_year(time, no_leap=no_leap)
    elif isinstance(time, DateTime365):
        if time.day_of_year is None:
            raise ValueError("DateTime365.day_of_year is not initialized")
        tm_yday = time.day_of_year
    else:
        tm_yday = time.timetuple().tm_yday

        # Disregard Feb 29 for leap years
        year = time.year

        if no_leap and is_leap_year(year) and tm_yday > 59:
            tm_yday -= 1

    time_index = tm_yday - 1

    out: NDArray = data[field_name][time_index, ...]

    return out


def get_field_at_specific_time(
    field_name: str,
    data: Mapping[str, NDArray],
    coupler: "Coupler",
    current_time: Optional[datetime | ModelDateTime] = None,
) -> NDArray:
    """Retrieve a field from a component data storage dictionary at a specific time,
    applying time interpolation if necessary.

    Arguments:
        field_name: Name of the field to retrieve.
        data: Dictionary containing the component data with time-dependent fields.
        coupler: Coupler instance for time settings.
        current_time: Optional datetime or CustomDateTime object representing the current time.
                      If None, coupler's start time is used.
    Returns:
        NDArray: The field data interpolated to the specified time.
    """

    total_seconds = datetime_to_seconds_in_year(
        coupler.clock.start if current_time is None else current_time
    )

    (n1, f1), (n2, f2) = get_periodic_interval(
        current_time=total_seconds,
        cycle_length=coupler.settings.year_in_seconds,
        rec_spacing=coupler.settings.year_in_seconds / 12.0,
        n_rec=12,
    )

    arr = data[field_name]

    # Use swapaxes to have (lat, lon) ordering
    out: NDArray = (f1 * arr[..., n1] + f2 * arr[..., n2]).swapaxes(-2, -1)

    return out
