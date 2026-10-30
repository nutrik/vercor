from __future__ import annotations

from jax import Array
import jax.numpy as jnp


def canonical_longitudes(lon_deg: Array) -> Array:
    """Return signed longitude phases in [-180, 180) without offset cancellation.

    The shared degree frame keeps lookup, tangent bases, and distances
    consistent without forming a full-turn offset for small coordinates.
    """
    # A signed remainder retains small positive/negative longitudes. Adding
    # 180 before reduction would erase their resolution, especially in float32.
    reduced = jnp.fmod(lon_deg, 360.0)
    reduced = jnp.where(reduced >= 180.0, reduced - 360.0, reduced)
    reduced = jnp.where(reduced < -180.0, reduced + 360.0, reduced)
    return reduced


def longitude_radians(lon_deg: Array) -> Array:
    """Reduce whole turns before conversion to preserve spherical phase precision."""
    return jnp.deg2rad(canonical_longitudes(lon_deg))


def unit_east_north(lon_rad: Array, lat_rad: Array) -> tuple[Array, Array]:
    r"""Compute local east/north unit vectors on the unit sphere."""

    slon, clon = jnp.sin(lon_rad), jnp.cos(lon_rad)
    slat, clat = jnp.sin(lat_rad), jnp.cos(lat_rad)

    e_east = jnp.stack((-slon, clon, jnp.zeros_like(lon_rad)), axis=-1)
    e_north = jnp.stack((-slat * clon, -slat * slon, clat), axis=-1)
    return (e_east, e_north)


def great_circle_distance_rad(
    lon1: Array,
    lat1: Array,
    lon2: Array,
    lat2: Array,
) -> Array:
    r"""Return haversine great-circle distance in radians on the unit sphere."""

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    sdlat2 = jnp.sin(dlat * 0.5)
    sdlon2 = jnp.sin(dlon * 0.5)
    a = sdlat2 * sdlat2 + jnp.cos(lat1) * jnp.cos(lat2) * sdlon2 * sdlon2
    a = jnp.clip(a, 0.0, 1.0)
    return 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))


def all_positive(values: Array) -> bool:
    """Return whether all entries are strictly positive as an eager bool."""

    return bool(jnp.all(values > 0.0))


def all_negative(values: Array) -> bool:
    """Return whether all entries are strictly negative as an eager bool."""

    return bool(jnp.all(values < 0.0))
