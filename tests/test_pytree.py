from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tests.assertions import assert_allclose_compact
from vercor.grid import RectilinearGrid
from vercor.interpolators.bilinear_rectilinear import BilinearRectilinearInterpolator
from vercor.interpolators.conservative_remap_rectilinear import (
    ConservativeRectilinearRemapper,
)
from vercor.pytree import PyTreeNodeMixin
from vercor.runtime.state import RuntimeComponentState, RuntimeCouplerState
from vercor.runtime.stores import RuntimeFieldStore
from vercor.runtime.time import RuntimeStepInfo
from vercor.setups.external.jax_gcm_runtime import JAXGCMRuntimePayload


def test_registered_pytree_classes_inherit_shared_flatten_methods() -> None:
    registered_classes = (
        RuntimeStepInfo,
        RuntimeFieldStore,
        RuntimeComponentState,
        RuntimeCouplerState,
        RectilinearGrid,
        BilinearRectilinearInterpolator,
        ConservativeRectilinearRemapper,
        JAXGCMRuntimePayload,
    )

    for registered_class in registered_classes:
        assert issubclass(registered_class, PyTreeNodeMixin)
        assert "tree_flatten" not in registered_class.__dict__
        assert "tree_unflatten" not in registered_class.__dict__


def test_array_only_pytree_round_trip_uses_declared_children() -> None:
    step_info = RuntimeStepInfo.from_sequences(
        monthly_index_left=[0, 1],
        monthly_index_right=[1, 2],
        monthly_weight_left=[0.25, 0.75],
        monthly_weight_right=[0.75, 0.25],
        daily_index=[10, 11],
    )

    leaves, treedef = jax.tree_util.tree_flatten(step_info)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(restored, RuntimeStepInfo)
    assert len(leaves) == 5
    assert_allclose_compact(restored.monthly_index_left, step_info.monthly_index_left)
    assert_allclose_compact(
        restored.monthly_weight_right, step_info.monthly_weight_right
    )


def test_static_pytree_metadata_round_trip_uses_declared_aux_fields() -> None:
    grid = RectilinearGrid(
        name="metadata-grid",
        longitude=np.asarray([0.0, 180.0]),
        latitude=np.asarray([-45.0, 45.0]),
        binary_mask=np.asarray([[1, 0], [0, 1]]),
    )
    store = RuntimeFieldStore.from_mapping(
        {
            "temperature": jnp.asarray([280.0, 281.0]),
            "humidity": jnp.asarray([0.001, 0.002]),
        }
    )

    grid_leaves, grid_treedef = jax.tree_util.tree_flatten(grid)
    store_leaves, store_treedef = jax.tree_util.tree_flatten(store)
    restored_grid = jax.tree_util.tree_unflatten(grid_treedef, grid_leaves)
    restored_store = jax.tree_util.tree_unflatten(store_treedef, store_leaves)

    assert restored_grid.name == "metadata-grid"
    assert restored_store.field_names == ("temperature", "humidity")
    assert_allclose_compact(restored_grid.binary_mask, grid.binary_mask)
    assert_allclose_compact(restored_store.get("humidity"), store.get("humidity"))


def test_remapper_pytree_round_trip_uses_declared_metadata_only() -> None:
    remapper = ConservativeRectilinearRemapper(
        src_lon_edges=jnp.asarray([0.0, 1.0, 2.0]),
        src_lat_edges=jnp.asarray([2.0, 1.0, 0.0]),
        dst_lon_edges=jnp.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        dst_lat_edges=jnp.asarray([2.0, 1.5, 1.0, 0.5, 0.0]),
        normalize="fracarea",
    )

    leaves, treedef = jax.tree_util.tree_flatten(remapper)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored._s_lat_flip is True
    assert restored._d_lat_flip is True
    assert not hasattr(restored, "_normalize_fracarea")
    assert not hasattr(restored, "_n_dst_cells")
    assert not hasattr(restored, "_n_src_cells")
    assert "_pytree_post_unflatten" not in ConservativeRectilinearRemapper.__dict__
    assert_allclose_compact(
        restored.apply_scalar(jnp.ones((2, 2))),
        remapper.apply_scalar(jnp.ones((2, 2))),
    )
