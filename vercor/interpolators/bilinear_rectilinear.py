from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax

from vercor.dtypes import as_jax_real_array, jax_full, jax_index_dtype
import vercor.interpolators._bilinear_extrapolation as _extrapolation
import vercor.interpolators._bilinear_geometry as _geometry
import vercor.interpolators._bilinear_weights as _weights
from vercor.pytree import PyTreeNodeMixin


@jax.tree_util.register_pytree_node_class
class BilinearRectilinearInterpolator(PyTreeNodeMixin):
    """
    Bilinear interpolator for rectilinear lat/lon grids with:

    - periodic longitude handling
    - ascending or descending latitude
    - non-uniform spacing
    - optional NaN-aware renormalization
    - support for scalar and vector (u,v) fields
    - source/target masks
    - nearest / IDW extrapolation for masked-out areas
    """

    pytree_children = (
        "lon_src_deg",
        "lat_src_deg",
        "lon_src_rad",
        "lat_src_rad",
        "lon_tgt_deg",
        "lat_tgt_deg",
        "lon_tgt_rad",
        "lat_tgt_rad",
        "_lon_tgt_flat",
        "_lat_tgt_flat",
        "src_mask",
        "tgt_mask",
        "i0",
        "i1",
        "j0",
        "j1",
        "fx",
        "fy",
        "w00",
        "w10",
        "w01",
        "w11",
        "_e_east_t",
        "_e_north_t",
        "_e_east_src",
        "_e_north_src",
        "_lon_src_2d",
        "_lat_src_2d",
        "_lon_src_flat",
        "_lat_src_flat",
    )
    pytree_aux_data = (
        "periodic",
        "nan_renorm",
        "extrapolation_mode",
        "idw_k",
        "idw_eps",
        "fill_value",
        "nlon",
        "nlat",
        "nx_source",
        "ny_source",
        "tshape",
        "_lon_flipped",
        "lat_ascending",
        "lat_descending",
    )

    def __init__(
        self,
        lon_src: Any,
        lat_src: Any,
        lon_tgt: Any,
        lat_tgt: Any,
        src_mask: Any | None = None,
        tgt_mask: Any | None = None,
        periodic_longitude: bool = True,
        nan_renorm: bool = True,
        extrapolation_mode: str | None = None,
        idw_k: int = 8,
        idw_eps: float = 1e-12,
        fill_value: float = float("nan"),
    ):
        self.periodic = bool(periodic_longitude)
        self.nan_renorm = bool(nan_renorm)
        self.extrapolation_mode = extrapolation_mode
        self.idw_k = int(idw_k)
        self.idw_eps = float(idw_eps)
        self.fill_value = float(fill_value)

        lon_src_deg = as_jax_real_array(lon_src)
        lat_src_deg = as_jax_real_array(lat_src)
        if lon_src_deg.ndim != 1 or lat_src_deg.ndim != 1:
            raise AssertionError("lon_src, lat_src must be 1-D")

        lon_diff = jnp.diff(lon_src_deg)
        lat_diff = jnp.diff(lat_src_deg)

        lon_ascending = _geometry.all_positive(lon_diff)
        lon_descending = _geometry.all_negative(lon_diff)
        if not (lon_ascending or lon_descending):
            raise ValueError(
                "lon_src must be strictly monotonic (ascending or descending)."
            )
        if lon_descending:
            lon_src_deg = jnp.flip(lon_src_deg)
            self._lon_flipped = True
        else:
            self._lon_flipped = False

        self.lat_ascending = _geometry.all_positive(lat_diff)
        self.lat_descending = _geometry.all_negative(lat_diff)
        if not (self.lat_ascending or self.lat_descending):
            raise ValueError(
                "lat_src must be strictly monotonic (ascending or descending)."
            )

        self.lon_src_deg = lon_src_deg
        self.lat_src_deg = lat_src_deg
        self.lon_src_rad = jnp.deg2rad(lon_src_deg)
        self.lat_src_rad = jnp.deg2rad(lat_src_deg)
        self.nlon = int(lon_src_deg.size)
        self.nlat = int(lat_src_deg.size)
        self.nx_source = self.nlon
        self.ny_source = self.nlat

        lon_tgt_array = as_jax_real_array(lon_tgt)
        lat_tgt_array = as_jax_real_array(lat_tgt)
        lon_tgt_deg, lat_tgt_deg = jnp.meshgrid(lon_tgt_array, lat_tgt_array)
        self.tshape = tuple(int(size) for size in lon_tgt_deg.shape)
        self.lon_tgt_deg = lon_tgt_deg
        self.lat_tgt_deg = lat_tgt_deg
        self.lon_tgt_rad = jnp.deg2rad(lon_tgt_deg)
        self.lat_tgt_rad = jnp.deg2rad(lat_tgt_deg)
        self._lon_tgt_flat = self.lon_tgt_rad.reshape(-1)
        self._lat_tgt_flat = self.lat_tgt_rad.reshape(-1)

        if src_mask is None:
            src_mask_array = jnp.ones((self.ny_source, self.nx_source), dtype=bool)
        else:
            src_mask_array = jnp.broadcast_to(
                jnp.asarray(src_mask, dtype=bool), (self.ny_source, self.nx_source)
            )
        if self._lon_flipped:
            src_mask_array = jnp.flip(src_mask_array, axis=1)
        self.src_mask = src_mask_array

        if tgt_mask is None:
            self.tgt_mask = jnp.ones(self.tshape, dtype=bool)
        else:
            self.tgt_mask = jnp.broadcast_to(
                jnp.asarray(tgt_mask, dtype=bool), self.tshape
            )

        self._precompute_cells_and_weights()

        self._e_east_t, self._e_north_t = _geometry.unit_east_north(
            self.lon_tgt_rad, self.lat_tgt_rad
        )
        lon_src_2d, lat_src_2d = jnp.meshgrid(self.lon_src_rad, self.lat_src_rad)
        self._e_east_src, self._e_north_src = _geometry.unit_east_north(
            lon_src_2d,
            lat_src_2d,
        )
        self._lon_src_2d = lon_src_2d
        self._lat_src_2d = lat_src_2d
        self._lon_src_flat = lon_src_2d.reshape(-1)
        self._lat_src_flat = lat_src_2d.reshape(-1)

    def _precompute_cells_and_weights(self) -> None:
        weights = _weights.compute_bilinear_cell_weights(
            lon_src_deg=self.lon_src_deg,
            lat_src_deg=self.lat_src_deg,
            lon_tgt_deg=self.lon_tgt_deg,
            lat_tgt_deg=self.lat_tgt_deg,
            periodic=self.periodic,
            lat_ascending=self.lat_ascending,
        )

        self.i0 = weights.i0.astype(jax_index_dtype())
        self.i1 = weights.i1.astype(jax_index_dtype())
        self.j0 = weights.j0.astype(jax_index_dtype())
        self.j1 = weights.j1.astype(jax_index_dtype())
        self.fx = weights.fx
        self.fy = weights.fy
        self.w00 = weights.w00
        self.w10 = weights.w10
        self.w01 = weights.w01
        self.w11 = weights.w11

    def _prepare_source_field(self, src: Array) -> Array:
        src_array = as_jax_real_array(src)
        if self._lon_flipped:
            src_array = jnp.flip(src_array, axis=1)
        return src_array

    def _apply_bilinear_scalar(self, src: Array) -> tuple[Array, Array]:
        src_array = self._prepare_source_field(src)
        if src_array.shape != (self.nlat, self.nlon):
            raise ValueError(
                f"src field must have shape (nlat,nlon)=({self.nlat},{self.nlon})"
            )
        valid = _extrapolation.valid_scalar_source_mask(src_array, self.src_mask)

        v00 = src_array[self.j0, self.i0]
        v10 = src_array[self.j0, self.i1]
        v01 = src_array[self.j1, self.i0]
        v11 = src_array[self.j1, self.i1]

        m00 = valid[self.j0, self.i0]
        m10 = valid[self.j0, self.i1]
        m01 = valid[self.j1, self.i0]
        m11 = valid[self.j1, self.i1]

        if self.nan_renorm:
            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = w00 + w10 + w01 + w11

            num = (
                jnp.where(m00, w00 * v00, 0.0)
                + jnp.where(m10, w10 * v10, 0.0)
                + jnp.where(m01, w01 * v01, 0.0)
                + jnp.where(m11, w11 * v11, 0.0)
            )
            out = jnp.where(wsum > 0.0, num / wsum, jnp.nan)
            return out, wsum

        num = self.w00 * v00 + self.w10 * v10 + self.w01 * v01 + self.w11 * v11
        all_ok = m00 & m10 & m01 & m11
        out = jnp.where(all_ok, num, jnp.nan)
        wsum = jnp.where(all_ok, 1.0, 0.0)
        return out, wsum

    def _extrapolate_scalar(self, src: Array, src_mask: Array | None) -> Array:
        if self.extrapolation_mode is None:
            return jax_full(self.tshape, self.fill_value)

        src_array = self._prepare_source_field(src)
        valid = _extrapolation.valid_scalar_source_mask(src_array, src_mask)
        return _extrapolation.extrapolate_scalar_field(
            source_values=src_array,
            valid_source_mask=valid,
            target_shape=self.tshape,
            target_lon_flat=self._lon_tgt_flat,
            target_lat_flat=self._lat_tgt_flat,
            source_lon_flat=self._lon_src_flat,
            source_lat_flat=self._lat_src_flat,
            mode=self.extrapolation_mode,
            idw_k=self.idw_k,
            idw_eps=self.idw_eps,
            fill_value=self.fill_value,
        )

    def apply_scalar(self, src: Any) -> Any:
        out, _ = self._apply_bilinear_scalar(src)
        need = ~jnp.isfinite(out)
        ext = lax.cond(
            jnp.any(need),
            lambda _: self._extrapolate_scalar(src, self.src_mask),
            lambda _: jax_full(self.tshape, self.fill_value),
            operand=None,
        )
        out = jnp.where(need, ext, out)
        out = jnp.where(self.tgt_mask, out, self.fill_value)
        return out.reshape(self.tshape)

    def apply_vector(self, u_src: Any, v_src: Any) -> tuple[Any, Any]:
        u_src_array = self._prepare_source_field(u_src)
        v_src_array = self._prepare_source_field(v_src)
        if u_src_array.shape != (self.nlat, self.nlon) or v_src_array.shape != (
            self.nlat,
            self.nlon,
        ):
            raise ValueError(
                f"(u_src,v_src) must both have shape (nlat,nlon)=({self.nlat},{self.nlon}),"
                f" provided {u_src_array.shape}, {v_src_array.shape}"
            )

        valid = (
            jnp.asarray(self.src_mask, dtype=bool)
            & jnp.isfinite(u_src_array)
            & jnp.isfinite(v_src_array)
        )
        v3 = (u_src_array[..., None] * self._e_east_src) + (
            v_src_array[..., None] * self._e_north_src
        )

        v00 = v3[self.j0, self.i0, :]
        v10 = v3[self.j0, self.i1, :]
        v01 = v3[self.j1, self.i0, :]
        v11 = v3[self.j1, self.i1, :]

        m00 = valid[self.j0, self.i0]
        m10 = valid[self.j0, self.i1]
        m01 = valid[self.j1, self.i0]
        m11 = valid[self.j1, self.i1]

        if self.nan_renorm:
            w00 = self.w00 * m00
            w10 = self.w10 * m10
            w01 = self.w01 * m01
            w11 = self.w11 * m11
            wsum = (w00 + w10 + w01 + w11)[..., None]
            num = (
                jnp.where(m00[..., None], w00[..., None] * v00, 0.0)
                + jnp.where(m10[..., None], w10[..., None] * v10, 0.0)
                + jnp.where(m01[..., None], w01[..., None] * v01, 0.0)
                + jnp.where(m11[..., None], w11[..., None] * v11, 0.0)
            )
            vt3 = jnp.where(wsum > 0.0, num / wsum, jnp.nan)
            need = ~jnp.isfinite(vt3[..., 0])
        else:
            num = (
                self.w00[..., None] * v00
                + self.w10[..., None] * v10
                + self.w01[..., None] * v01
                + self.w11[..., None] * v11
            )
            all_ok = m00 & m10 & m01 & m11
            vt3 = jnp.where(all_ok[..., None], num, jnp.nan)
            need = ~all_ok

        u_t = jnp.sum(vt3 * self._e_east_t, axis=-1)
        v_t = jnp.sum(vt3 * self._e_north_t, axis=-1)

        def apply_vector_extrapolation(_: None) -> tuple[Array, Array]:
            u_fill = self._extrapolate_scalar(u_src_array, valid)
            v_fill = self._extrapolate_scalar(v_src_array, valid)
            return u_fill, v_fill

        def no_vector_extrapolation(_: None) -> tuple[Array, Array]:
            fill = jax_full(self.tshape, self.fill_value)
            return fill, fill

        u_fill, v_fill = lax.cond(
            jnp.any(need),
            apply_vector_extrapolation,
            no_vector_extrapolation,
            operand=None,
        )
        u_t = jnp.where(need, u_fill, u_t)
        v_t = jnp.where(need, v_fill, v_t)

        u_t = jnp.where(self.tgt_mask, u_t, self.fill_value)
        v_t = jnp.where(self.tgt_mask, v_t, self.fill_value)
        return (u_t.reshape(self.tshape), v_t.reshape(self.tshape))
