from __future__ import annotations

from typing import Any, cast

from jax import Array, lax
import jax.numpy as jnp

from vercor.dtypes import as_jax_real_array, jax_linspace
from vercor.grid import RectilinearGrid


def make_rectilinear_grid(
    name: str,
    nlon: int,
    nlat: int,
    longitude_start: float,
    longitude_end: float,
    latitude_start: float,
    latitude_end: float,
    mask: Any | None = None,
) -> RectilinearGrid:
    """Build a rectilinear grid with equally spaced coordinate centers."""

    longitude = jax_linspace(longitude_start, longitude_end, nlon)
    latitude = jax_linspace(latitude_start, latitude_end, nlat)
    return RectilinearGrid(
        name=name,
        longitude=longitude,
        latitude=latitude,
        binary_mask=mask,
    )


def centers_to_edges(centers: Any, grid_type: str) -> Any:
    """Convert 1D latitude or longitude centers to cell edges."""

    centers = as_jax_real_array(centers)

    if centers.size < 2:
        half_width = 0.5
        return jnp.stack((centers[0] - half_width, centers[0] + half_width))

    inner_edges = 0.5 * (centers[:-1] + centers[1:])
    d_start = inner_edges[0] - centers[0]
    d_end = centers[-1] - inner_edges[-1]
    edge_start = centers[0] - d_start
    edge_end = centers[-1] + d_end
    edges = jnp.concatenate(
        (jnp.stack((edge_start,)), inner_edges, jnp.stack((edge_end,)))
    )

    if grid_type == "lat":
        edges = jnp.clip(edges, -90.0, 90.0)
    elif grid_type == "lon":
        span = edges[-1] - edges[0]

        def clamp_lon(overhanging_edges: Array) -> Array:
            return cast(
                Array,
                lax.cond(
                    jnp.min(overhanging_edges) < -5.0,
                    lambda value: jnp.clip(value, -180.0, 180.0),
                    lambda value: jnp.clip(value, 0.0, 360.0),
                    overhanging_edges,
                ),
            )

        edges = lax.cond(
            span > 360.0 + 1e-10,
            clamp_lon,
            lambda value: value,
            edges,
        )

    return cast(Any, edges)
