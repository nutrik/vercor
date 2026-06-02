from __future__ import annotations

from dataclasses import dataclass

from jax import Array
import jax.numpy as jnp

from vercor.dtypes import jax_index_dtype
import vercor.interpolators._bilinear_geometry as _geometry


@dataclass(frozen=True)
class BilinearCellWeights:
    """Precomputed source-cell indices and bilinear weights for target points."""

    i0: Array
    i1: Array
    j0: Array
    j1: Array
    fx: Array
    fy: Array
    w00: Array
    w10: Array
    w01: Array
    w11: Array


def compute_bilinear_cell_weights(
    *,
    lon_src_deg: Array,
    lat_src_deg: Array,
    lon_tgt_deg: Array,
    lat_tgt_deg: Array,
    periodic: bool,
    lat_ascending: bool,
) -> BilinearCellWeights:
    """Compute source-cell indices and normalized bilinear weights."""

    nlon = lon_src_deg.shape[0]
    nlat = lat_src_deg.shape[0]

    if periodic:
        base0 = float(lon_src_deg[0])
        lon_tgt_mapped = _geometry.wrap_longitudes_like(lon_tgt_deg, base0)
    else:
        lon_tgt_mapped = jnp.clip(
            lon_tgt_deg,
            jnp.min(lon_src_deg),
            jnp.max(lon_src_deg),
        )

    i1 = jnp.searchsorted(lon_src_deg, lon_tgt_mapped, side="right")
    i0 = i1 - 1
    if periodic:
        i0 = jnp.mod(i0, nlon)
        i1 = jnp.mod(i1, nlon)
    else:
        i0 = jnp.clip(i0, 0, nlon - 2)
        i1 = i0 + 1

    if lat_ascending:
        j1 = jnp.searchsorted(lat_src_deg, lat_tgt_deg, side="right")
        j0 = j1 - 1
    else:
        lat_inv = jnp.flip(lat_src_deg)
        j1_inv = jnp.searchsorted(lat_inv, lat_tgt_deg, side="right")
        j0_inv = j1_inv - 1
        j0 = (nlat - 1) - jnp.clip(j0_inv, 0, nlat - 2) - 1
        j1 = j0 + 1

    j0 = jnp.clip(j0, 0, nlat - 2)
    j1 = j0 + 1

    lon_src_rad = jnp.deg2rad(lon_src_deg)
    lat_src_rad = jnp.deg2rad(lat_src_deg)
    lon_tgt_rad = jnp.deg2rad(lon_tgt_deg)
    lat_tgt_rad = jnp.deg2rad(lat_tgt_deg)

    lon0 = lon_src_rad[i0]
    lon1 = lon_src_rad[i1]
    dlon = lon1 - lon0
    wrap = i1 <= i0
    dlon = jnp.where(wrap, (lon1 + 2.0 * jnp.pi) - lon0, dlon)

    dlam = lon_tgt_rad - lon0
    dlam = jnp.where(dlam < 0.0, dlam + 2.0 * jnp.pi, dlam)
    fx = jnp.where(dlon != 0.0, dlam / dlon, 0.0)
    fx = jnp.clip(fx, 0.0, 1.0)

    lat0 = lat_src_rad[j0]
    lat1 = lat_src_rad[j1]
    dphi = lat1 - lat0
    fy = jnp.where(dphi != 0.0, (lat_tgt_rad - lat0) / dphi, 0.0)
    fy = jnp.clip(fy, 0.0, 1.0)

    return BilinearCellWeights(
        i0=i0.astype(jax_index_dtype()),
        i1=i1.astype(jax_index_dtype()),
        j0=j0.astype(jax_index_dtype()),
        j1=j1.astype(jax_index_dtype()),
        fx=fx,
        fy=fy,
        w00=(1.0 - fx) * (1.0 - fy),
        w10=fx * (1.0 - fy),
        w01=(1.0 - fx) * fy,
        w11=fx * fy,
    )
