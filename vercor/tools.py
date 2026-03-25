from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING, Any, Callable, Optional, Mapping, Sequence
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
from vercor.exceptions import AssetError, CouplerError, RegridderError
from vercor.grid import RectilinearGrid
from vercor.regridders.conservative import ConservativeRectilinearRegridder
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.regridders.helpers import compute_land_mask
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
    "erainterim_ocean_4deg": {
        "filename": "forcing_4deg_global_open_itf.nc",
        "md5": "cfcc6d8cde8da5a74ecec00309d92dd7",
    },
    "erainterim_ocean_1deg": {
        "filename": "forcing_1deg_global.nc",
        "md5": "1fc86f88acd820da078c8da5873cfa01",
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


def safe_component_nanmean(component: Any, field_name: str) -> float:
    """Return np.nanmean(component.get(field_name)) or NaN when unavailable."""

    try:
        return float(np.nanmean(component.get(field_name)))
    except Exception:
        return float("nan")


def _safe_component_metric_mean(
    component: Any,
    metric: str | Callable[[Any], NDArray | float],
) -> float:
    """Resolve a metric and return a robust mean value as float."""

    if isinstance(metric, str):
        return safe_component_nanmean(component, metric)

    try:
        return float(np.nanmean(metric(component)))
    except Exception:
        return float("nan")


def print_component_field_means_table(
    components: Mapping[str, Any],
    fields: Sequence[tuple[str | Callable[[Any], NDArray | float], str]],
    component_order: Sequence[str] | None = None,
) -> None:
    """Print a means table for component fields with configurable column order.

    Arguments:
        components: Mapping from component label (e.g. "ATM") to component object.
        fields: Sequence of (metric, display_label) rows.
            metric can be either a component field name or a callable.
        component_order: Optional ordered component labels for output columns.
    """

    ordered_names = list(component_order or components.keys())
    ordered_names = [name for name in ordered_names if name in components]

    first_col_width = max(10, max((len(label) for _, label in fields), default=10))
    value_col_width = 15

    header = f"{'Variable':<{first_col_width}} " + " ".join(
        f"{name:>{value_col_width}}" for name in ordered_names
    )
    print(header)
    print("-" * len(header))

    for field_name, label in fields:
        values = [
            _safe_component_metric_mean(components[name], field_name)
            for name in ordered_names
        ]
        value_text = " ".join(f"{value:>{value_col_width}.4f}" for value in values)
        print(f"{label:<{first_col_width}} {value_text}")


def _get_component_plot_data(
    component: Any,
    scalar_field_name: str,
    u_field_name: str,
    v_field_name: str,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Return lon/lat grids and scalar/vector fields for one component."""

    lon = np.asarray(component.grid.longitude)
    lat = np.asarray(component.grid.latitude)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing="ij")
    scalar_field = np.asarray(component.get(scalar_field_name)).T
    u_field = np.asarray(component.get(u_field_name)).T
    v_field = np.asarray(component.get(v_field_name)).T
    return lon_2d, lat_2d, scalar_field, u_field, v_field


def plot_component_scalar_vector_comparison(
    rows: Sequence[tuple[str, Any, str, str, str]],
    *,
    figsize: tuple[float, float] = (15.0, 10.0),
    quiver_scale: float = 100.0,
    cmap: str = "coolwarm",
) -> tuple[Any, NDArray, Any]:
    """Create aligned scalar/vector plots for multiple components.

    Arguments:
        rows: Sequence of tuples containing:
            (label, component, scalar_field_name, u_field_name, v_field_name)
        figsize: Figure size passed to matplotlib.
        quiver_scale: Quiver scale factor for all vector panels.
        cmap: Colormap for scalar panels.

    Returns:
        (fig, axs, scalar_mappable) from matplotlib.
    """

    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("rows must contain at least one component")

    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 2, figsize=figsize, layout="constrained")

    if n_rows == 1:
        axs = np.asarray([axs])
    else:
        axs = np.asarray(axs)

    plot_data = [
        (label, *_get_component_plot_data(component, scalar_name, u_name, v_name))
        for label, component, scalar_name, u_name, v_name in rows
    ]

    scalar_min = float(min(np.nanmin(item[3]) for item in plot_data))
    scalar_max = float(max(np.nanmax(item[3]) for item in plot_data))

    lon_min = float(min(np.nanmin(item[1]) for item in plot_data))
    lon_max = float(max(np.nanmax(item[1]) for item in plot_data))
    lat_min = float(min(np.nanmin(item[2]) for item in plot_data))
    lat_max = float(max(np.nanmax(item[2]) for item in plot_data))

    scalar_mappable = None
    for i, (label, lon_2d, lat_2d, scalar_field, u_field, v_field) in enumerate(
        plot_data
    ):
        scalar_plot = axs[i, 0].pcolormesh(
            lon_2d,
            lat_2d,
            scalar_field,
            shading="auto",
            cmap=cmap,
            vmin=scalar_min,
            vmax=scalar_max,
        )
        if scalar_mappable is None:
            scalar_mappable = scalar_plot

        axs[i, 0].set_title(f"{label} Scalar Field")
        axs[i, 0].set_xlabel("Longitude")
        axs[i, 0].set_ylabel("Latitude")

        axs[i, 1].quiver(
            lon_2d,
            lat_2d,
            u_field,
            v_field,
            scale=quiver_scale,
        )
        axs[i, 1].set_title(f"{label} Vector Field")
        axs[i, 1].set_xlabel("Longitude")
        axs[i, 1].set_ylabel("Latitude")

    for ax in axs.flat:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    if scalar_mappable is None:
        raise ValueError("No scalar field was plotted")

    return fig, axs, scalar_mappable


def grids_identical(g0: RectilinearGrid, g1: RectilinearGrid) -> bool:
    return (
        g0.shape == g1.shape
        and np.allclose(g0.latitude, g1.latitude, atol=1e-15)
        and np.allclose(g0.longitude, g1.longitude, atol=1e-15)
    )


def get_component(
    allcomponents: dict[str, AllComponentsType], types: str
) -> AllComponentsType:
    """Helper function to retrieve a registered component of a specific type
    from the coupler's allcomponents dictionary."""

    components: list[AllComponentsType] = [
        component for component in allcomponents.values() if component.name == types
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


def compute_ocn_lnd_masks_on_atm_grid(
    ocean_binary_mask: NDArray, regridder: ConservativeRectilinearRegridder
) -> tuple[NDArray, NDArray, NDArray]:
    """Compute ocean and land fractional & binary masks on the atmospheric grid by conservative remapping

    Arguments:
        ocean_binary_mask: binary ocean mask on ocean grid (1 for ocean, 0 for land)
        regridder: conservative regridder from ocean grid to atmospheric grid

    Returns:
        ocn_fmask_on_atm_grid: fractional ocean mask on atmospheric grid (values between 0 and 1)
        lnd_fmask_on_atm_grid: fractional land mask on atmospheric grid (values between 0 and 1)
        lnd_bmask_on_atm_grid: binary land mask on atmospheric grid (1 for land, 0 for ocean)
    """

    ocean_bmask = np.asarray(ocean_binary_mask)
    # Conservative remapping of binary mask to atmospheric grid
    # results to fractional mask on atmosphere grid
    ocn_fmask_on_atm_grid = np.asarray(regridder(ocean_bmask))
    ocn_fmask_on_atm_grid = np.clip(ocn_fmask_on_atm_grid, 0.0, 1.0)
    lnd_fmask_on_atm_grid = 1.0 - ocn_fmask_on_atm_grid
    # This lnd_bmask... is needed to double-check that the land model mask from random land component
    # is consistent with the land mask after remapping ocean mask to atmosphere grid
    lnd_bmask_on_atm_grid = compute_land_mask(ocn_fmask_on_atm_grid)

    return ocn_fmask_on_atm_grid, lnd_fmask_on_atm_grid, lnd_bmask_on_atm_grid


def check_total_lnd_ocn_mask_sum(
    lnd_fmask_on_atm_grid: NDArray, ocn_fmask_on_atm_grid: NDArray
) -> None:
    """Check that the fractional land and ocean masks on the atmospheric grid sum to approximately 1 everywhere.

    Arguments:
        lnd_fmask_on_atm_grid: fractional land mask on atmospheric grid (values between 0 and 1)
        ocn_fmask_on_atm_grid: fractional ocean mask on atmospheric grid (values between 0 and 1)
    """

    fmask_sum = lnd_fmask_on_atm_grid + ocn_fmask_on_atm_grid
    min_fsum = fmask_sum.min()
    max_fsum = fmask_sum.max()
    if not (
        np.isclose(min_fsum, 1.0, atol=1e-3) and np.isclose(max_fsum, 1.0, atol=1e-3)
    ):
        raise RegridderError(
            "Fractional land and ocean masks on atmospheric grid must sum to approx. 1 everywhere "
            f"(minimum sum {min_fsum}, maximum sum {max_fsum})"
        )


def check_remap_conservation(
    regridder: ConservativeRectilinearRegridder,
    ocean_binary_mask_on_ocn_grid: NDArray,
    ocn_fmask_on_atm_grid: NDArray,
) -> None:
    """Check that the conservative regridding of the ocean binary mask from ocean grid
    to atmospheric grid conserves total mass (ocean area).

    Arguments:
        regridder: conservative regridder from ocean grid to atmospheric grid
        ocean_binary_mask_on_ocn_grid: binary ocean mask on ocean grid (1 for ocean, 0 for land)
        ocn_fmask_on_atm_grid: fractional ocean mask on atmospheric grid (values between 0 and 1) obtained
            by regridding ocean_binary_mask_on_ocn_grid with the provided regridder
    """

    do_not_check_mass = False

    if regridder.interpolator is not None and isinstance(
        regridder.interpolator, ConservativeRectilinearRemapper
    ):
        src_lat = regridder.interpolator.src_lat_b
        dst_lat = regridder.interpolator.dst_lat_b
        if src_lat[-1] != dst_lat[-1] or src_lat[0] != dst_lat[0]:
            do_not_check_mass = True
            print(
                "Skipping mass conservation check for regridding ocean mask to atmospheric grid "
                "due to different latitude bounds.\n"
            )

        src_total_mass = regridder.interpolator.get_src_total_mass(
            ocean_binary_mask_on_ocn_grid
        )
        dst_total_mass = regridder.interpolator.get_dst_total_mass(
            ocn_fmask_on_atm_grid
        )

        if not do_not_check_mass and not np.isclose(
            src_total_mass, dst_total_mass, atol=1e-6
        ):
            raise RegridderError(
                "Regridding ocean binary mask to atmospheric grid does not conserve total mass "
                f"(source mass: {src_total_mass}, destination mass: {dst_total_mass})"
            )


def create_lnd_mask_from_ocn(
    atm_lat: NDArray, atm_lon: NDArray, ocn_grid: RectilinearGrid
) -> tuple[NDArray, NDArray]:
    """Create a new land mask from Ocean & JCM geometry object."""

    from vercor.regridders.conservative import ConservativeRectilinearRegridder

    atmosphere_grid = RectilinearGrid(
        name="ATM",
        longitude=atm_lon,
        latitude=atm_lat,
    )

    regridder = ConservativeRectilinearRegridder(
        ocn_grid,
        atmosphere_grid,
    )

    ocean_binary_mask = np.asarray(ocn_grid.binary_mask)

    (
        ocn_fmask_on_atm_grid,
        lnd_fmask_on_atm_grid,
        lnd_bmask_on_atm_grid,
    ) = compute_ocn_lnd_masks_on_atm_grid(ocean_binary_mask, regridder)

    check_remap_conservation(regridder, ocean_binary_mask, ocn_fmask_on_atm_grid)

    check_total_lnd_ocn_mask_sum(
        lnd_fmask_on_atm_grid,
        ocn_fmask_on_atm_grid,
    )

    return lnd_bmask_on_atm_grid, lnd_fmask_on_atm_grid
