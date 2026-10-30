from __future__ import annotations

from math import ceil, floor
from typing import Any, Optional

import jax
import jax.numpy as jnp

from vercor.dtypes import (
    as_jax_real_array,
    jax_index_dtype,
    jax_real_dtype,
    jax_zeros,
)
from vercor._numerical_safety import safe_masked_divide
from vercor._pytree import PyTreeNodeMixin
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
            src_lon_edges (1D array): Eastward source longitude edges, optionally wrapped.
            src_lat_edges (1D array): Source latitude cell edges.
            dst_lon_edges (1D array): Eastward (optionally wrapped) or descending target edges.
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

        # 1. Standardize and store bounds
        self.src_lon_b = self._standardize_lon(src_lon_edges)
        self.src_lat_b, self._s_lat_flip = self._standardize_lat(src_lat_edges)
        self.dst_lon_b = self._standardize_lon(dst_lon_edges, preserve_descending=True)
        self.dst_lat_b, self._d_lat_flip = self._standardize_lat(dst_lat_edges)

        self.n_src_lon = self.src_lon_b.shape[0] - 1
        self.n_src_lat = self.src_lat_b.shape[0] - 1
        self.n_dst_lon = self.dst_lon_b.shape[0] - 1
        self.n_dst_lat = self.dst_lat_b.shape[0] - 1

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

        # Match the overlap calculation: subtract in degrees before converting.
        # Subtracting radian coordinates instead loses relative precision at
        # large longitudes and can remap unit coverage above physical bounds.
        dst_lon_diff = jnp.deg2rad(jnp.abs(jnp.diff(self.dst_lon_b)))
        dst_lat_diff = jnp.abs(jnp.diff(dst_sin_lat))
        dst_areas = (
            (self.radius**2) * jnp.outer(dst_lat_diff, dst_lon_diff)
        ).reshape(-1)
        self.dst_areas = jnp.where(dst_areas <= 1e-15, jnp.inf, dst_areas)

    @staticmethod
    def _segment_sum(values: jax.Array, indices: jax.Array, size: int) -> jax.Array:
        return jax_zeros((size,)).at[indices].add(values)

    @staticmethod
    @jax.jit
    def _standardize_lon(
        bounds: RuntimeArray, *, preserve_descending: bool = False
    ) -> jax.Array:
        """Unwrap eastward edges without reordering their cells.

        Native periodic halos can supply [360, 4, ..., 360]. Add whole
        revolutions after negative jumps, then anchor the lower end of the
        grid in [-180, 180) to keep periodic overlap arithmetic near zero.
        Positive widths, including explicit cells wider than 180 degrees,
        remain unchanged; shortest-arc unwrapping would alter those cells.
        Entirely descending destinations retain their existing ordering. A
        single wrapped destination cell is ambiguous with that convention;
        supply its eastward bounds unwrapped (e.g. [356, 364]).
        """
        edges = as_jax_real_array(bounds)
        jumps = jnp.diff(edges)
        descending = preserve_descending & jnp.all(jumps <= 0.0)
        turns = jnp.where(jumps < 0.0, jnp.ceil(-jumps / 360.0), 0.0)
        turns = jnp.where(descending, 0.0, turns)
        offsets = jnp.concatenate((jnp.zeros_like(edges[:1]), jnp.cumsum(turns)))
        origin = jnp.where(descending, edges[-1], edges[0])
        origin_turns = jnp.floor((origin + 180.0) / 360.0)
        return edges + 360.0 * (offsets - origin_turns)

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
        """Collect every periodic intersection of the static grid bounds."""

        # Broad cells can straddle more than the immediate neighboring copies.
        # These setup-only limits depend on geometry, never on runtime fields.
        first_turn = ceil(float((src_edges[0] - jnp.max(dst_edges)) / 360.0))
        last_turn = floor(float((src_edges[-1] - jnp.min(dst_edges)) / 360.0))
        overlap_matrix = jax_zeros((self.n_dst_lon, self.n_src_lon))
        for turn in range(first_turn, last_turn + 1):
            overlap_matrix = overlap_matrix + self._compute_dense_interval_overlaps(
                src_edges, dst_edges + 360.0 * turn
            )
        rows, cols, values = self._dense_overlaps_to_triplets(overlap_matrix)
        return rows, cols, jnp.deg2rad(values)

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
        n_dst_cells = self.n_dst_lat * self.n_dst_lon
        weighted_sum = self._segment_sum(
            weighted_values,
            self.dst_indices,
            n_dst_cells,
        )

        if self.normalize == "fracarea":
            valid = jnp.where(jnp.isnan(flat_field), 0.0, 1.0)
            norm = self._segment_sum(
                self.overlap_weights * valid[self.src_indices],
                self.dst_indices,
                n_dst_cells,
            )
        else:
            norm = self.dst_areas

        result = safe_masked_divide(
            weighted_sum,
            norm,
            where=norm > 1e-15,
            inactive_value=jnp.nan,
        )
        result_grid = result.reshape((self.n_dst_lat, self.n_dst_lon))

        if self._d_lat_flip:
            result_grid = result_grid[::-1, :]

        return result_grid
