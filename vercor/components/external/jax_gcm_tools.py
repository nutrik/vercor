from pathlib import Path
from typing import Any, Optional
import subprocess
import os
import sys
import shutil

import jax
import jax.numpy as jnp

from jcm.geometry import Geometry, get_terrain


def compute_pressure_levels(
    reference_pressure: jnp.ndarray,
    top_pressure: jnp.ndarray,
    sigma_levels: jnp.ndarray,
    normalized_surface_pressure: jnp.ndarray,
) -> jnp.ndarray:
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
    p0 = jnp.asarray(reference_pressure, dtype=jnp.float_)
    p_top = jnp.asarray(top_pressure, dtype=jnp.float_)
    sigma = jnp.asarray(sigma_levels, dtype=jnp.float_)
    nps = jnp.asarray(normalized_surface_pressure, dtype=jnp.float_)

    if p_top.ndim != 0:
        raise ValueError("top_pressure must be a scalar array")
    if sigma.ndim != 1:
        raise ValueError("sigma_levels must be a 1D array")

    ps = jnp.asarray(nps * p0, dtype=jnp.float_)[jnp.newaxis, :, :]

    # Broadcast p_top to the horizontal grid shape (nlat, nlon)
    p_top_bcast = jnp.broadcast_to(p_top, ps.shape)

    # Compute pressure levels
    pressure_levels = p_top_bcast + sigma[:, jnp.newaxis, jnp.newaxis] * (
        ps - p_top_bcast
    )

    return pressure_levels


def get_altitudes_sigma_levels(
    temperature: jnp.ndarray,
    pressure: jnp.ndarray,
    specific_humidity: jnp.ndarray,
    *,
    z0: float | jnp.ndarray = 0.0,
    g: float = 9.80665,
    Rd: float = 287.05,
    Rv: float = 461.5,
) -> jnp.ndarray:
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
      - Tv is computed with a more exact relation using q:
          Tv = T * (1 + (Rv/Rd - 1)*q) / (1 - q)
        (For small q, this is close to T*(1 + 0.61 q).)
      - Tv_bar between adjacent levels is taken as a simple average.
    """
    T = jnp.asarray(temperature, dtype=jnp.float_)
    p = jnp.asarray(pressure, dtype=jnp.float_)
    q = jnp.asarray(specific_humidity, dtype=jnp.float_)

    if T.ndim != 3 or p.ndim != 3 or q.ndim != 3:
        raise ValueError(
            "temperature, pressure, specific_humidity must all be 3D: (nlev, nlat, nlon)"
        )
    if T.shape != p.shape or T.shape != q.shape:
        raise ValueError(
            "temperature, pressure, specific_humidity must have identical shapes"
        )

    nlev, nlat, nlon = T.shape

    # Virtual temperature (more exact form using q)
    eps = Rv / Rd  # ~1.608
    Tv = T * (1.0 + (eps - 1.0) * q) / (1.0 - q)
    # Tv = T * (1. + 0.608 * q)

    # Log-pressure thickness between adjacent levels: ln(p[k-1]/p[k])
    # (works even if p is not strictly monotone, but physically it should be)
    log_pr = jnp.log(p[:-1, :, :] / p[1:, :, :])  # shape (nlev-1, nlat, nlon)

    # Layer-mean Tv between k-1 and k
    Tv_bar = 0.5 * (Tv[:-1, :, :] + Tv[1:, :, :])

    # Thickness of each layer
    dz = (Rd / g) * Tv_bar * log_pr  # shape (nlev-1, nlat, nlon)

    # Integrate upward from z0 at k=0
    z = jnp.empty_like(T, dtype=jnp.float_)
    z0_arr = jnp.asarray(z0, dtype=jnp.float_)
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


def generate_jcm_geometry_from_orography(
    orography: jnp.ndarray,
    num_levels: int = 8,
    truncation_number: Optional[int] = None,
) -> Geometry:
    """Initialize all of the speedy model geometry variables from a given terrain file containing orog and lsm.

    Arguments:
        orography: A 2-dimensional array of orography
        num_levels (optional): Number of vertical levels `kx` (default 8).
        truncation_number (optional): Spectral truncation number for surface geopotential.
                                      If None, inferred from nodal_shape.

    Returns:
        Geometry object
    """
    orography, fmask = get_terrain(orography=orography)
    return Geometry.from_grid_shape(
        nodal_shape=orography.shape,
        num_levels=num_levels,
        orography=orography,
        fmask=fmask,
        truncation_number=truncation_number,
    )


def generate_jcm_forcing_and_topography_files(
    resolution: int,
    data_directory: Optional[Path] = None,
) -> dict[str, Path]:

    import jcm

    if not (isinstance(data_directory, Path) or data_directory is None):
        raise TypeError("`data_directory` must be of type `Path` or `None`.")

    if data_directory is None:
        home_data_directory = os.environ.get("HOME", None)

        if home_data_directory is None:
            data_directory = Path.cwd()
        else:
            data_directory = Path(home_data_directory)

        data_directory = data_directory / ".cache/jcm"

        print(f'Using input data directory: "{str(data_directory)}".')

    raw_data_directory = Path(jcm.__file__).parent / "data/bc"

    # Prepare boundary file
    files_to_check = dict(
        terrain=(data_directory / f"terrain_t{resolution:d}.nc").resolve(),
        forcing=(data_directory / f"forcing_t{resolution:d}.nc").resolve(),
    )

    def check_if_file_exist(
        file_dict: dict[str, Path], verbose: bool = True
    ) -> dict[Path, bool]:

        file_status = {file: Path(file).exists() for _, file in file_dict.items()}

        if verbose:
            for file, result in file_status.items():
                print(
                    f"Check file: {str(file):s}...",
                    "found." if result else "not found.",
                )

        return file_status

    file_status = check_if_file_exist(files_to_check)

    if not all(list(file_status.values())):
        print("Some files are missing. Need to generate them.")

        data_directory.mkdir(parents=True, exist_ok=True)
        interpolation_code = (raw_data_directory / "interpolate.py").resolve()

        try:
            subprocess.run(
                [sys.executable, str(interpolation_code), f"{resolution:d}"],
                check=True,
                capture_output=True,
                text=True,
                cwd=data_directory,
            )
        except subprocess.CalledProcessError as e:
            print("Error output:", e.stderr)

        for destination_file in files_to_check.values():
            source_file = Path(raw_data_directory / destination_file.name)
            print(f"Copying: {str(source_file):s} => {str(destination_file):s}")
            shutil.copy(source_file, destination_file)

        new_file_status = check_if_file_exist(files_to_check)
        if not all(list(new_file_status.values())):
            raise FileNotFoundError(
                "Something went wrong. The daily file is not generated. Please check."
            )

    return files_to_check


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
