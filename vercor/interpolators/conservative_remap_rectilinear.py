from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp

from vercor.dtypes import (
    as_jax_real_array,
    jax_index_dtype,
    jax_real_dtype,
    jax_zeros,
)
from vercor.pytree import PyTreeNodeMixin
from vercor.types import RuntimeArray


@jax.tree_util.register_pytree_node_class
class ConservativeRectilinearRemapper(PyTreeNodeMixin):
    """
    First-order locally conservative area-average remapping.
    Handles arbitrary rectilinear grids, periodicity, and conservation.
    """

    pytree_children = (
        "src_lon_b",
        "src_lat_b",
        "dst_lon_b",
        "dst_lat_b",
        "dst_areas",
        "dst_indices",
        "src_indices",
        "overlap_weights",
    )
    pytree_aux_data = (
        "radius",
        "normalize",
        "_s_lat_flip",
        "_d_lat_flip",
        "n_src_lon",
        "n_src_lat",
        "n_dst_lon",
        "n_dst_lat",
    )

    def __init__(
        self,
        src_lon_edges: RuntimeArray,
        src_lat_edges: RuntimeArray,
        dst_lon_edges: RuntimeArray,
        dst_lat_edges: RuntimeArray,
        src_mask: Optional[RuntimeArray] = None,
        normalize: str = "conservation",
        radius: float = 6371.0,
    ):
        """
        Initialize and precompute remapping weights.

        Arguments:
            src_lon_edges (1D array): Source longitude cell edges (Monotonic).
            src_lat_edges (1D array): Source latitude cell edges.
            dst_lon_edges (1D array): Target longitude cell edges (Monotonic).
            dst_lat_edges (1D array): Target latitude cell edges.
            source_mask (2D array, optional): Boolean mask for source grid (True=Invalid).
            normalize (str):
                'conservation': Normalize by total area of target cell. (Mass Preserving)
                'fracarea'    : Normalize by intersection area. (Value Preserving / Extrapolation)
            radius (float): Radius of the sphere (km).
        """

        if normalize not in {"conservation", "fracarea"}:
            raise ValueError(
                "normalize must be either 'conservation' or 'fracarea', "
                f"got {normalize!r}"
            )

        self.radius = float(radius)
        self.normalize = normalize
        self._normalize_fracarea = normalize == "fracarea"

        # 1. Standardize and store bounds
        self.src_lon_b = as_jax_real_array(src_lon_edges)
        self.src_lat_b, self._s_lat_flip = self._standardize_lat(src_lat_edges)
        self.dst_lon_b = as_jax_real_array(dst_lon_edges)
        self.dst_lat_b, self._d_lat_flip = self._standardize_lat(dst_lat_edges)

        self.n_src_lon = self.src_lon_b.shape[0] - 1
        self.n_src_lat = self.src_lat_b.shape[0] - 1
        self.n_dst_lon = self.dst_lon_b.shape[0] - 1
        self.n_dst_lat = self.dst_lat_b.shape[0] - 1
        self._n_dst_cells = self.n_dst_lat * self.n_dst_lon

        # 2. Compute 1D overlaps
        lon_dst_idx, lon_src_idx, lon_overlap = self._compute_lon_overlaps(
            self.src_lon_b, self.dst_lon_b
        )

        src_sin_lat = jnp.round(jnp.sin(jnp.deg2rad(self.src_lat_b)), 14)
        dst_sin_lat = jnp.round(jnp.sin(jnp.deg2rad(self.dst_lat_b)), 14)
        lat_dst_idx, lat_src_idx, lat_overlap = self._compute_interval_overlaps(
            src_sin_lat, dst_sin_lat
        )

        # 3. Combine 1D overlaps into 2D remapping triplets
        dst_indices = (
            lat_dst_idx[:, None] * self.n_dst_lon + lon_dst_idx[None, :]
        ).reshape(-1)
        src_indices = (
            lat_src_idx[:, None] * self.n_src_lon + lon_src_idx[None, :]
        ).reshape(-1)
        overlap_weights = (
            (self.radius**2) * (lat_overlap[:, None] * lon_overlap[None, :])
        ).reshape(-1)

        # 4. Apply source mask eagerly by dropping invalid source entries
        if src_mask is not None:
            src_mask_array = jnp.asarray(src_mask, dtype=bool)
            if self._s_lat_flip:
                src_mask_array = src_mask_array[::-1, :]
            valid_src = (~src_mask_array).reshape(-1)
            keep_indices = jnp.nonzero(valid_src[src_indices])[0]
            dst_indices = dst_indices[keep_indices]
            src_indices = src_indices[keep_indices]
            overlap_weights = overlap_weights[keep_indices]

        self.dst_indices = dst_indices.astype(jax_index_dtype())
        self.src_indices = src_indices.astype(jax_index_dtype())
        self.overlap_weights = overlap_weights.astype(jax_real_dtype())

        dst_lon_diff = jnp.abs(jnp.diff(jnp.deg2rad(self.dst_lon_b)))
        dst_lat_diff = jnp.abs(jnp.diff(dst_sin_lat))
        dst_areas = (
            (self.radius**2) * jnp.outer(dst_lat_diff, dst_lon_diff)
        ).reshape(-1)
        self.dst_areas = jnp.where(dst_areas <= 1e-15, jnp.inf, dst_areas)

    def _pytree_post_unflatten(self) -> None:
        """Restore derived static remapping state after PyTree unflattening."""

        self._normalize_fracarea = self.normalize == "fracarea"
        self._n_dst_cells = self.n_dst_lon * self.n_dst_lat

    @staticmethod
    def _segment_sum(values: jax.Array, indices: jax.Array, size: int) -> jax.Array:
        return jax_zeros((size,)).at[indices].add(values)

    @staticmethod
    def _standardize_lat(bounds: RuntimeArray) -> tuple[jax.Array, bool]:
        """Ensure latitude bounds are monotonically increasing."""

        bounds_array = as_jax_real_array(bounds)
        is_flipped = bool(bounds_array[0] > bounds_array[-1])
        return (
            jnp.flip(bounds_array) if is_flipped else bounds_array,
            is_flipped,
        )

    @staticmethod
    def _compute_dense_interval_overlaps(
        src_edges: jax.Array, dst_edges: jax.Array
    ) -> jax.Array:
        """Return dense destination-by-source overlap lengths."""

        src_start = src_edges[:-1][None, :]
        src_end = src_edges[1:][None, :]
        dst_start = jnp.minimum(dst_edges[:-1], dst_edges[1:])[:, None]
        dst_end = jnp.maximum(dst_edges[:-1], dst_edges[1:])[:, None]
        overlaps = jnp.maximum(
            0.0, jnp.minimum(dst_end, src_end) - jnp.maximum(dst_start, src_start)
        )
        valid_dst = jnp.abs(dst_end - dst_start) > 1e-15
        return jnp.where(valid_dst, overlaps, 0.0)

    @staticmethod
    def _dense_overlaps_to_triplets(
        overlaps: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Flatten a dense overlap matrix into sparse triplets."""

        keep = overlaps > 1e-15
        dst_indices, src_indices = jnp.nonzero(keep)
        values = overlaps[dst_indices, src_indices]
        return dst_indices, src_indices, values

    def _compute_interval_overlaps(
        self, src_edges: jax.Array, dst_edges: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """1D overlap calculation (latitude in sin-space)."""

        overlap_matrix = self._compute_dense_interval_overlaps(src_edges, dst_edges)
        return self._dense_overlaps_to_triplets(overlap_matrix)

    def _compute_lon_overlaps(
        self, src_edges: jax.Array, dst_edges: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """1D longitude overlap with periodicity check."""

        overlap_matrix = self._compute_dense_interval_overlaps(src_edges, dst_edges)
        for shift in (360.0, -360.0):
            overlap_matrix = overlap_matrix + self._compute_dense_interval_overlaps(
                src_edges, dst_edges + shift
            )
        rows, cols, values = self._dense_overlaps_to_triplets(overlap_matrix)
        return rows, cols, jnp.deg2rad(values)

    def get_src_areas(self) -> jax.Array:
        """Returns the exact source cell areas (useful for mass verification)."""

        dlon = jnp.diff(jnp.deg2rad(self.src_lon_b))
        sin_lat = jnp.sin(jnp.deg2rad(self.src_lat_b))
        dsinlat = jnp.abs(jnp.diff(sin_lat))
        areas = (self.radius**2) * dsinlat[:, None] * dlon[None, :]

        if self._s_lat_flip:
            areas = areas[::-1, :]

        return areas

    def apply_scalar(self, field: Any) -> jax.Array:
        """Apply conservative remapping to a scalar field."""

        field_array = as_jax_real_array(field)
        expected_shape = (self.n_src_lat, self.n_src_lon)
        if field_array.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: {field_array.shape} vs grid {expected_shape}"
            )

        if self._s_lat_flip:
            field_array = field_array[::-1, :]

        flat_field = field_array.reshape(-1)
        clean_field = jnp.where(jnp.isnan(flat_field), 0.0, flat_field)
        weighted_values = self.overlap_weights * clean_field[self.src_indices]
        weighted_sum = self._segment_sum(
            weighted_values,
            self.dst_indices,
            self._n_dst_cells,
        )

        if self._normalize_fracarea:
            valid = jnp.where(jnp.isnan(flat_field), 0.0, 1.0)
            norm = self._segment_sum(
                self.overlap_weights * valid[self.src_indices],
                self.dst_indices,
                self._n_dst_cells,
            )
        else:
            norm = self.dst_areas

        result = jnp.where(norm > 1e-15, weighted_sum / norm, jnp.nan)
        result_grid = result.reshape((self.n_dst_lat, self.n_dst_lon))

        if self._d_lat_flip:
            result_grid = result_grid[::-1, :]

        return result_grid

    def get_src_total_mass(self, field_on_src: Any) -> float:
        """Calculate total mass on source grid given field values."""

        field_array = as_jax_real_array(field_on_src)
        result = jnp.nansum(field_array * self.get_src_areas())
        return float(result)

    def get_dst_total_mass(self, field_on_dst: Any) -> float:
        """Calculate total mass on destination grid given field values."""

        clean_areas = jnp.where(jnp.isinf(self.dst_areas), 0.0, self.dst_areas)
        field_array = as_jax_real_array(field_on_dst)
        result = jnp.nansum(field_array.reshape(-1) * clean_areas)
        return float(result)
