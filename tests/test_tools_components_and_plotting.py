from __future__ import annotations

from typing import Any, cast

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pytest

from tests._tools_support import DummyComponentA, DummyComponentB
from tests.assertions import assert_allclose_compact
import vercor.diagnostics as diagnostics_module
from vercor.exceptions import CouplerError
from vercor.grid import RectilinearGrid
from vercor.runtime import (
    RuntimeComponentState,
    RuntimeFieldStore,
    append_unique_runtime_fields,
    flatten_exchange_fields,
)
from vercor.runtime.views import RuntimeComponentView
from vercor.diagnostics import (
    plot_component_scalar_vector_comparison,
    print_component_field_means_table,
    safe_component_nanmean,
)
from vercor.runtime.topology import get_component
from vercor.grid_masks import (
    grids_identical,
)
from vercor.grid_geometry import grids_identical as geometry_grids_identical

matplotlib.use("Agg")


def test_flatten_fields_and_append_unique() -> None:
    flattened = flatten_exchange_fields(["a", ("b", "c"), "d"])
    assert flattened == ["a", "b", "c", "d"]

    target = ["a", "b"]
    append_unique_runtime_fields(target, ["b", "c", "d", "a"])
    assert target == ["a", "b", "c", "d"]


def test_grids_identical_detects_equal_and_unequal_grids() -> None:
    lon = jnp.array([0.0, 1.0, 2.0])
    lat = jnp.array([-1.0, 0.0])
    g0 = RectilinearGrid("g0", longitude=lon, latitude=lat)
    g1 = RectilinearGrid("g1", longitude=lon + 0.0, latitude=lat + 0.0)
    g2 = RectilinearGrid("g2", longitude=jnp.array([0.0, 1.5, 2.0]), latitude=lat + 0.0)

    assert grids_identical(g0, g1)
    assert not grids_identical(g0, g2)
    assert grids_identical is geometry_grids_identical


def test_get_component_returns_single_and_raises_for_ambiguous_or_missing() -> None:
    allcomponents: dict[str, object] = {
        "a": DummyComponentA(name="ATM"),
        "b": DummyComponentB(name="OCN"),
    }

    selected = get_component(cast(Any, allcomponents), "ATM")
    assert isinstance(selected, DummyComponentA)

    with pytest.raises(CouplerError, match="No component"):
        get_component(cast(Any, allcomponents), "UNKNOWN")

    with pytest.raises(CouplerError, match="Multiple"):
        get_component(
            cast(
                Any,
                {
                    "a": DummyComponentA(name="OCN"),
                    "b": DummyComponentA(name="OCN"),
                },
            ),
            "OCN",
        )


def test_safe_component_nanmean_returns_nan_for_missing_fields() -> None:
    grid = RectilinearGrid(
        "dummy",
        longitude=np.array([0.0, 1.0]),
        latitude=np.array([0.0, 1.0]),
    )
    comp = RuntimeComponentView(
        name="DUMMY",
        grid=grid,
        data=RuntimeFieldStore.from_mapping(
            {"foo": jnp.array([[1.0, jnp.nan], [3.0, 5.0]])}
        ),
    )

    assert np.isclose(safe_component_nanmean(comp, "foo"), 3.0)
    assert np.isnan(safe_component_nanmean(comp, "does_not_exist"))


def test_print_component_field_means_table_with_callable_metric(
    capsys: pytest.CaptureFixture[str],
) -> None:
    grid = RectilinearGrid(
        "dummy",
        longitude=np.array([0.0, 1.0]),
        latitude=np.array([0.0, 1.0]),
    )
    atm = RuntimeComponentView(
        name="ATM",
        grid=grid,
        data=RuntimeFieldStore.from_mapping(
            {
                "u": np.array([[3.0, 4.0], [0.0, 0.0]]),
                "v": np.array([[4.0, 3.0], [0.0, 0.0]]),
                "temp": np.array([[280.0, 282.0], [284.0, 286.0]]),
            }
        ),
    )
    ocn = RuntimeComponentView(
        name="OCN",
        grid=grid,
        data=RuntimeFieldStore.from_mapping(
            {
                "u": np.array([[1.0, 2.0], [0.0, 0.0]]),
                "v": np.array([[2.0, 1.0], [0.0, 0.0]]),
                "temp": np.array([[270.0, 271.0], [272.0, 273.0]]),
            }
        ),
    )

    print_component_field_means_table(
        components={"ATM": atm, "OCN": ocn},
        fields=[
            ("temp", "temp"),
            (
                lambda view: np.sqrt(view.data.get("u") ** 2 + view.data.get("v") ** 2),
                "speed",
            ),
        ],
        component_order=["ATM", "OCN"],
    )

    captured = capsys.readouterr().out
    assert "Variable" in captured
    assert "ATM" in captured
    assert "OCN" in captured
    assert "temp" in captured
    assert "speed" in captured


@pytest.mark.fast_always
def test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes() -> None:
    import matplotlib.pyplot as plt

    atm_grid = RectilinearGrid(
        "atm",
        longitude=jnp.asarray([0.0, 1.0, 2.0]),
        latitude=jnp.asarray([-1.0, 1.0]),
    )
    ocn_grid = RectilinearGrid(
        "ocn",
        longitude=np.array([0.0, 2.0]),
        latitude=np.array([-2.0, 0.0, 2.0]),
    )

    atm = RuntimeComponentView(
        name="ATM",
        grid=atm_grid,
        data=RuntimeFieldStore.from_mapping(
            {
                "scalar": jnp.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]),
                "u": jnp.ones((2, 3)),
                "v": jnp.zeros((2, 3)),
            }
        ),
    )
    ocn = RuntimeComponentView(
        name="OCN",
        grid=ocn_grid,
        data=RuntimeFieldStore.from_mapping(
            {
                "scalar": np.array([[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]]),
                "u": np.zeros((3, 2)),
                "v": np.ones((3, 2)),
            }
        ),
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            ("ATM", atm, "scalar", "u", "v"),
            ("OCN", ocn, "scalar", "u", "v"),
        ],
        figsize=(8.0, 5.0),
        quiver_scale=10.0,
    )

    assert axs.shape == (2, 2)
    assert isinstance(atm.data.get("scalar"), jax.Array)
    assert scalar_mappable is not None
    assert_allclose_compact(axs[0, 0].get_xlim(), axs[1, 0].get_xlim())
    assert_allclose_compact(axs[0, 1].get_xlim(), axs[1, 1].get_xlim())
    assert_allclose_compact(axs[0, 0].get_ylim(), axs[1, 0].get_ylim())

    plt.close(fig)


@pytest.mark.fast_always
def test_plot_component_scalar_vector_comparison_reads_runtime_state_pair() -> None:
    import matplotlib.pyplot as plt

    grid = RectilinearGrid(
        "atm",
        longitude=jnp.asarray([0.0, 1.0, 2.0]),
        latitude=jnp.asarray([-1.0, 1.0]),
    )
    runtime_state = RuntimeComponentState(
        data=RuntimeFieldStore.from_mapping(
            {
                "total_surface_temperature": jnp.array(
                    [[280.0, 281.0, 282.0], [283.0, 284.0, 285.0]]
                ),
                "u_velocity": jnp.ones((2, 3, 2)),
                "v_velocity": jnp.zeros((2, 3, 2)),
            }
        ),
        incoming=RuntimeFieldStore.empty(),
        outgoing=RuntimeFieldStore.from_mapping(
            {
                "u_velocity": jnp.ones((2, 3)),
                "v_velocity": jnp.zeros((2, 3)),
            }
        ),
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                RuntimeComponentView.from_component_state("ATM", grid, runtime_state),
                "total_surface_temperature",
                "u_velocity",
                "v_velocity",
            )
        ],
        figsize=(6.0, 4.0),
        quiver_scale=10.0,
    )

    assert axs.shape == (1, 2)
    assert scalar_mappable is not None

    plt.close(fig)


@pytest.mark.fast_always
def test_plot_component_scalar_vector_comparison_accepts_callable_scalar() -> None:
    import matplotlib.pyplot as plt

    grid = RectilinearGrid(
        "atm",
        longitude=jnp.asarray([0.0, 1.0]),
        latitude=jnp.asarray([-1.0, 1.0]),
    )
    runtime_view = RuntimeComponentView(
        name="ATM",
        grid=grid,
        data=RuntimeFieldStore.from_mapping(
            {
                "u_velocity": jnp.ones(grid.shape),
                "v_velocity": jnp.zeros(grid.shape),
            }
        ),
        incoming=RuntimeFieldStore.from_mapping(
            {
                "land_surface_temperature": jnp.asarray(
                    [[jnp.nan, 270.0], [271.0, jnp.nan]]
                ),
                "sea_surface_temperature": jnp.asarray(
                    [[272.0, jnp.nan], [273.0, 274.0]]
                ),
            }
        ),
    )

    fig, axs, scalar_mappable = plot_component_scalar_vector_comparison(
        rows=[
            (
                "ATM",
                runtime_view,
                diagnostics_module.total_surface_temperature,
                "u_velocity",
                "v_velocity",
            )
        ],
        figsize=(6.0, 4.0),
        quiver_scale=10.0,
    )

    assert axs.shape == (1, 2)
    assert "total_surface_temperature" not in runtime_view.data.field_names
    assert scalar_mappable is not None

    plt.close(fig)


def test_plot_component_scalar_vector_comparison_rejects_empty_rows() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        plot_component_scalar_vector_comparison(rows=[])
