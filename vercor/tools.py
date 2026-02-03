from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from vercor.exceptions import CouplerError
from vercor.grid import RectilinearGrid
from vercor.types import AllComponentsType


if TYPE_CHECKING:
    from vercor.coupler import Coupler


def _flatten_fields(
    field_names: List[str | Tuple[str, str]],
) -> List[str]:
    flattened: List[str] = []
    for item in field_names:
        if isinstance(item, tuple):
            flattened.extend(item)
        else:
            flattened.append(item)
    return flattened


def _append_unique(target: List[str], exchange_items: List[str]) -> None:
    """
    Arguments:
        target: List of field names in _fields2export & _fields2import
                in the component's exchange variable lists.
        exchange_items: List of field names from exchange rules (exchanger).
    """
    target.extend([item for item in exchange_items if item not in target])


def grids_identical(g0: RectilinearGrid, g1: RectilinearGrid) -> bool:
    return (
        g0.shape == g1.shape
        and np.allclose(g0.latitude, g1.latitude, atol=1e-15)
        and np.allclose(g0.longitude, g1.longitude, atol=1e-15)
    )


def get_component(
    allcomponents: Dict[str, AllComponentsType], types: Tuple[Any, ...], label: str
) -> AllComponentsType:
    components: List[AllComponentsType] = [
        component
        for component in allcomponents.values()
        if isinstance(component, types)
    ]
    if len(components) > 1:
        names = ", ".join(component.name for component in components)
        raise CouplerError(
            f"Multiple {label} components registered; only one supported (found: {names})"
        )
    if not components:
        raise CouplerError(f"No {label} component registered")
    return components[0]


def get_periodic_interval(
    current_time: float, cycle_length: float, rec_spacing: float, n_rec: int
) -> Tuple[Tuple[NDArray, NDArray], Tuple[NDArray, NDArray]]:
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
    # using npx.array works with both NumPy and JAX
    t_idx_1 = np.array(current_time // rec_spacing, dtype="int")
    t_idx_2 = np.array((1 + t_idx_1) % n_rec, dtype="int")
    weight_2 = (current_time - rec_spacing * t_idx_1) / rec_spacing
    weight_1 = 1.0 - weight_2
    return (t_idx_1, weight_1), (t_idx_2, weight_2)


def datetime_to_seconds_in_year(dt: datetime) -> float:
    """Convert a datetime object to the number of seconds since the start of the year.

    Arguments:
        dt: datetime object to convert.

    Returns:
        float: Number of seconds since the start of the year.
    """
    year_start = datetime(dt.year, 1, 1)
    seconds_since_year_start = (dt - year_start).total_seconds()
    return seconds_since_year_start


def get_forcing_data(file_type: str) -> Path:
    """Return the absolute Paths to the ./forcing directory relative to this file."""

    output = {
        "model_level": (
            Path(__file__).parent
            / ".."
            / "forcing"
            / "era5_198x_ml_4x4deg_monthly_mean.nc"
        ).resolve(),
        "surface": (
            Path(__file__).parent
            / ".."
            / "forcing"
            / "era5_198x_sfc_4x4deg_monthly_mean.nc"
        ).resolve(),
    }

    return output[file_type]


def get_time_slice(
    field_name: str,
    data: Dict[str, NDArray],
    time: datetime,
    no_leap: bool = True,
) -> NDArray:
    """Retrieve a field from a component data storage dictionary at a specific time index.

    Arguments:
        field_name: Name of the field to retrieve.
        data: Dictionary containing the component data with time-dependent fields.
        time: datetime object representing the time slice to retrieve.
        no_leap: Whether to ignore leap days (Feb 29) when indexing.

    Returns:
        NDArray: The field data at the specified time index.
    """

    tm_yday = time.timetuple().tm_yday

    # Disregard Feb 29 for leap years
    year = time.year
    leap = lambda x: (x % 4 == 0 and x % 100 != 0) or (x % 400 == 0)

    if no_leap and leap(year) and tm_yday > 59:
        tm_yday -= 1
    time_index = tm_yday - 1

    out: NDArray = data[field_name][time_index, ...]

    return out


def get_field_at_specific_time(
    field_name: str,
    data: Dict[str, NDArray],
    coupler: "Coupler",
    current_time: Optional[datetime] = None,
) -> NDArray:
    """Retrieve a field from a component data storage dictionary at a specific time,
    applying time interpolation if necessary.

    Arguments:
        field_name: Name of the field to retrieve.
        data: Dictionary containing the component data with time-dependent fields.
        coupler: Coupler instance for time settings.
        current_time: Optional datetime object representing the current time.
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

    # Use transpose to have (lat, lon) ordering
    out: NDArray = (
        f1 * data[f"{field_name}"][..., n1] + f2 * data[f"{field_name}"][..., n2]
    ).transpose()

    return out
