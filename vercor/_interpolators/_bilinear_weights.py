from __future__ import annotations

from dataclasses import dataclass

import jax
from jax import Array, lax
import jax.numpy as jnp

from vercor.dtypes import jax_index_dtype
import vercor._interpolators._bilinear_geometry as _geometry


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


@jax.jit
def _periodic_longitude_indices(
    source: Array, source_phase: Array, target_phase: Array
) -> tuple[Array, Array]:
    """Find first-turn neighbors without adding a large offset to the target.

    Compare source candidates in each target's whole-turn frame. A fixed
    binary search preserves original column and inclusive-endpoint ordering,
    including wider grids, without sorting or a dense target/source matrix.
    This cached kernel is used only while constructing interpolation weights.
    """
    nlon = source.shape[0]
    source_turn = jnp.rint((source[0] - source_phase[0]) / 360.0)
    offsets = 360.0 * (source_turn + (target_phase < source_phase[0]))

    def narrow(_: int, interval: tuple[Array, Array]) -> tuple[Array, Array]:
        left, right = interval
        middle = (left + right) // 2
        active = left < right
        candidate = source[jnp.minimum(middle, nlon - 1)] - offsets
        go_right = candidate <= target_phase
        return (
            jnp.where(active & go_right, middle + 1, left),
            jnp.where(active & ~go_right, middle, right),
        )

    left = jnp.zeros(target_phase.shape, dtype=jax_index_dtype())
    right = jnp.full_like(left, nlon)
    insertion, _ = lax.fori_loop(0, nlon.bit_length(), narrow, (left, right))
    return jnp.mod(insertion - 1, nlon), jnp.mod(insertion, nlon)


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

    lon_lookup = lon_src_deg
    if periodic:
        lon_lookup = _geometry.canonical_longitudes(lon_src_deg)
        lon_tgt_mapped = _geometry.canonical_longitudes(lon_tgt_deg)
        i0, i1 = _periodic_longitude_indices(lon_src_deg, lon_lookup, lon_tgt_mapped)
    else:
        lon_tgt_mapped = jnp.clip(
            lon_tgt_deg,
            jnp.min(lon_src_deg),
            jnp.max(lon_src_deg),
        )

        i1 = jnp.searchsorted(lon_lookup, lon_tgt_mapped, side="right")
        i0 = i1 - 1
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

    # Cell lookup and fractional position must use the same wrapped/clipped
    # longitude, including whole turns and nonperiodic domain endpoints.
    # Ratios are dimensionless: subtract in degrees to avoid cancellation
    # between separately converted radian coordinates near a periodic seam.

    lon0 = lon_lookup[i0]
    lon1 = lon_lookup[i1]
    dlon = lon_src_deg[i1] - lon_src_deg[i0]
    wrap = i1 <= i0
    phase_width = jnp.where(lon1 <= lon0, (180.0 - lon0) + (lon1 + 180.0), lon1 - lon0)
    dlon = jnp.where(wrap, phase_width, dlon)

    dlam = lon_tgt_mapped - lon0
    dlam = jnp.where(dlam < 0.0, (180.0 - lon0) + (lon_tgt_mapped + 180.0), dlam)
    fx = jnp.where(dlon != 0.0, dlam / dlon, 0.0)
    fx = jnp.clip(fx, 0.0, 1.0)

    lat0 = lat_src_deg[j0]
    lat1 = lat_src_deg[j1]
    dphi = lat1 - lat0
    fy = jnp.where(dphi != 0.0, (lat_tgt_deg - lat0) / dphi, 0.0)
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
