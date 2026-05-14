from pathlib import Path
from typing import Any, Optional, Tuple
from importlib import resources

import jax
import jax.numpy as jnp

from dinosaur.coordinate_systems import CoordinateSystem

from jcm.forcing import ForcingData
from jcm.terrain import TerrainData
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics.speedy.params import Parameters

from vercor.dtypes import as_jax_real_array
from vercor.fluxes.utilities import _virtual_temperature_from_specific_humidity
from vercor.types import RuntimeArray


def change_jcm_parameter_values(
    parameters: dict[str, float], default_parameters: Parameters
) -> None:

    for key, value in parameters.items():
        parameter_group_name, parameter_name = key.split(".")
        default_parameters.__getattribute__(parameter_group_name).__setattr__(
            parameter_name, as_jax_real_array(value)
        )


def get_default_parameter_values(
    parameters: list[str], default_parameters: Parameters
) -> dict[str, float]:
    output = {}

    for parameter in parameters:
        parameter_group_name, parameter_name = parameter.split(".")
        output[parameter] = default_parameters.__getattribute__(
            parameter_group_name
        ).__getattribute__(parameter_name)

    return output


def compute_pressure_levels(
    reference_pressure: RuntimeArray | float,
    top_pressure: RuntimeArray | float,
    sigma_levels: RuntimeArray,
    normalized_surface_pressure: RuntimeArray,
) -> jax.Array:
    """
    Compute pressure levels from sigma levels and top pressure.

    Arguments:
        reference_pressure: Reference surface pressure p0 [Pa]
        top_pressure: Top pressure p_top [Pa], scalar array
        sigma_levels: Sigma levels, 1D array of shape (nlev,) [-]
        normalized_surface_pressure: Normalized surface pressure [p/ p0],
                                     2D array of shape (nlat, nlon) [-]

    Returns:
        pressure_levels: Pressure levels p [Pa], 3D array of shape (nlev, nlat, nlon)
    """
    p0 = as_jax_real_array(reference_pressure)
    p_top = as_jax_real_array(top_pressure)
    sigma = as_jax_real_array(sigma_levels)
    nps = as_jax_real_array(normalized_surface_pressure)

    if p_top.ndim != 0:
        raise ValueError("top_pressure must be a scalar array")
    if sigma.ndim != 1:
        raise ValueError("sigma_levels must be a 1D array")

    ps = as_jax_real_array(nps * p0)[jnp.newaxis, :, :]

    # Broadcast p_top to the horizontal grid shape (nlat, nlon)
    p_top_bcast = jnp.broadcast_to(p_top, ps.shape)

    # Compute pressure levels
    pressure_levels = p_top_bcast + sigma[:, jnp.newaxis, jnp.newaxis] * (
        ps - p_top_bcast
    )

    return pressure_levels


def get_altitudes_sigma_levels(
    temperature: RuntimeArray,
    pressure: RuntimeArray,
    specific_humidity: RuntimeArray,
    *,
    z0: RuntimeArray | float = 0.0,
    g: float = 9.80665,
    Rd: float = 287.05,
    Rv: float = 461.5,
) -> jax.Array:
    """
    Compute geometric altitude z(p) on pressure levels using the Hypsometric Equation.

    Arguments:
      Inputs must be 3D arrays shaped (nlev, nlat, nlon):
      - temperature [K]
      - pressure [Pa]
      - specific_humidity q [kg/kg]  (a.k.a. "specific humidity")

    Returns:
      - z [m], same shape (nlev, nlat, nlon)

    Assumptions:
      - Level index increases upward (pressure generally decreases with k).
      - z0 is the altitude at the first level (k=0). By default 0 m.
        You may pass:
          * scalar (applied everywhere), or
          * array shaped (nlat, nlon) (spatially varying base height), or
          * array shaped (nlev, nlat, nlon) (full field; only z0[0,:,:] used)

    Formula:
      Δz = (Rd * Tv_bar / g) * ln(p_lower / p_upper)
      where Tv is virtual temperature derived from T and q.

    Notes:
      - Tv is computed from specific humidity as:
          Tv = T * (1 + (Rv/Rd - 1)*q)
        (For typical water-vapor constants this is close to T*(1 + 0.61 q).)
      - Tv_bar between adjacent levels is taken as a simple average.
    """
    T = as_jax_real_array(temperature)
    p = as_jax_real_array(pressure)
    q = as_jax_real_array(specific_humidity)

    if T.ndim != 3 or p.ndim != 3 or q.ndim != 3:
        raise ValueError(
            "temperature, pressure, specific_humidity must all be 3D: (nlev, nlat, nlon)"
        )
    if T.shape != p.shape or T.shape != q.shape:
        raise ValueError(
            "temperature, pressure, specific_humidity must have identical shapes"
        )

    nlev, nlat, nlon = T.shape

    # Virtual temperature for specific humidity q.
    eps = Rv / Rd  # ~1.608
    Tv = _virtual_temperature_from_specific_humidity(T, q, eps - 1.0)

    # Log-pressure thickness between adjacent levels: ln(p[k-1]/p[k])
    # (works even if p is not strictly monotone, but physically it should be)
    log_pr = jnp.log(p[:-1, :, :] / p[1:, :, :])  # shape (nlev-1, nlat, nlon)

    # Layer-mean Tv between k-1 and k
    Tv_bar = 0.5 * (Tv[:-1, :, :] + Tv[1:, :, :])

    # Thickness of each layer
    dz = (Rd / g) * Tv_bar * log_pr  # shape (nlev-1, nlat, nlon)

    # Integrate upward from z0 at k=0
    z = jnp.empty_like(T)
    z0_arr = as_jax_real_array(z0)
    if z0_arr.ndim == 0:
        z = z.at[0, :, :].set(z0_arr)
    elif z0_arr.shape == (nlat, nlon):
        z = z.at[0, :, :].set(z0_arr)
    elif z0_arr.shape == (nlev, nlat, nlon):
        z = z.at[0, :, :].set(z0_arr[0, :, :])
    else:
        raise ValueError("z0 must be a scalar, (nlat,nlon), or (nlev,nlat,nlon)")

    z = z.at[1:, :, :].set(z[0:1, :, :] + jnp.cumsum(dz, axis=0))
    return z


def generate_jcm_coords_forcing_topography_files(
    resolution: int = 31,
    input_data_directory: Optional[Path] = None,
) -> Tuple[CoordinateSystem, TerrainData, ForcingData]:
    """Generate JCM coordinates, forcing and topography files at the specified resolution.

    Arguments:
        resolution: Optional resolution of the JCM files to generate (e.g., 31 for T31)
        input_data_directory: Optional directory to look for existing files and to save generated files.
                              If None, defaults to ~/.vercor/jcm/

    Returns:
        A tuple of (coords, terrain, forcing) objects.
    """

    coords = get_speedy_coords(
        spectral_truncation=resolution
    )  # T31 spectral resolution with 8 vertical levels

    # Read JCM topography file
    # Load realistic orography and land-sea mask, interpolated to T31 grid
    if input_data_directory is None:
        data_dir = resources.files("jcm.data.bc.t30.clim")
    else:
        data_dir = Path(input_data_directory)

    terrain_file = data_dir / "terrain.nc"
    terrain = TerrainData.from_file(terrain_file, coords=coords)

    # Load realistic forcing data (SST, sea ice, soil moisture, etc.) interpolated to T31 grid
    forcing_file = data_dir / "forcing.nc"
    forcing = ForcingData.from_file(forcing_file, coords=coords)

    return (coords, terrain, forcing)


def mean_leaf(
    tree: Any,
    axis: int | list[int],
) -> Any:
    """
    A tool function that does the jnp.mean to leaf nodes.

    Arguments:
        tree : a tree object

    Returns:
        tree_mean : tree with jnp.mean applied to each of its leaf node.
    """
    return jax.tree_util.tree_map(lambda arr: jnp.mean(arr, axis=axis), tree)


def unwrap_leading_dims(
    obj: Any,
    first_n_dim: int = 2,
) -> Any:
    """
    A tool function that unwraps the leading dimensions of jax arrays

    Arguments:
        obj : A structure containining jax arrays

    Returns:
        unwrapped object.
    """

    def _unwrap(arr: jnp.ndarray) -> jnp.ndarray:
        new_shape = (-1,) + arr.shape[first_n_dim:]
        return jnp.reshape(arr, new_shape)

    return jax.tree_util.tree_map(_unwrap, obj)


def stack_objects(
    objs: list[Any],
) -> Any:
    """
    A tool function that stack dataclasses together.

    Arguments:
        objs : A list of objects that need to be stacked

    Returns:
        stacked : Stacked object.
    """
    # objs is a list of pytrees with same structure
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *objs)
    return stacked


def concat_objects(
    objs: list[Any],
    axis: int,
) -> Any:
    """
    A tool function that concats dataclasses together.

    Arguments:
        objs : A list of objects that need to be concat

    Returns:
        concatenated : Concatenated object.
    """
    # objs is a list of pytrees with same structure
    concatenated = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=axis), *objs
    )
    return concatenated
