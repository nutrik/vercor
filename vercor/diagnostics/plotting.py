from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from vercor.diagnostics.fields import (
    ComponentMetric,
    component_plot_field,
    component_plot_scalar,
)
from vercor.host_arrays import runtime_array_to_host
from vercor.runtime.views import RuntimeComponentView


def _get_component_plot_data(
    component: RuntimeComponentView,
    scalar: ComponentMetric,
    u_field_name: str,
    v_field_name: str,
) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Return lon/lat grids and scalar/vector fields for one component."""

    grid = component.grid
    lon = runtime_array_to_host(grid.longitude)
    lat = runtime_array_to_host(grid.latitude)
    lon_2d, lat_2d = np.meshgrid(lon, lat, indexing="ij")
    scalar_field = runtime_array_to_host(
        jnp.asarray(component_plot_scalar(component, scalar)).T
    )
    u_field = runtime_array_to_host(
        jnp.asarray(component_plot_field(component, u_field_name)).T
    )
    v_field = runtime_array_to_host(
        jnp.asarray(component_plot_field(component, v_field_name)).T
    )
    return lon_2d, lat_2d, scalar_field, u_field, v_field


def plot_component_scalar_vector_comparison(
    rows: Sequence[tuple[str, RuntimeComponentView, ComponentMetric, str, str]],
    *,
    figsize: tuple[float, float] = (15.0, 10.0),
    quiver_scale: float = 100.0,
    cmap: str = "coolwarm",
) -> tuple[Any, NDArray[Any], Any]:
    """Create aligned scalar/vector plots for multiple components."""

    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("rows must contain at least one component")

    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 2, figsize=figsize, layout="constrained")
    axs = np.asarray([axs]) if n_rows == 1 else np.asarray(axs)

    plot_data = [
        (label, *_get_component_plot_data(component, scalar_name, u_name, v_name))
        for label, component, scalar_name, u_name, v_name in rows
    ]
    scalar_min = float(min(np.nanmin(item[3]) for item in plot_data))
    scalar_max = float(max(np.nanmax(item[3]) for item in plot_data))
    lon_min = float(min(np.nanmin(item[1]) for item in plot_data))
    lon_max = float(max(np.nanmax(item[1]) for item in plot_data))
    lat_min = float(min(np.nanmin(item[2]) for item in plot_data))
    lat_max = float(max(np.nanmax(item[2]) for item in plot_data))

    scalar_mappable = None
    for i, (label, lon_2d, lat_2d, scalar_field, u_field, v_field) in enumerate(
        plot_data
    ):
        scalar_plot = axs[i, 0].pcolormesh(
            lon_2d,
            lat_2d,
            scalar_field,
            shading="auto",
            cmap=cmap,
            vmin=scalar_min,
            vmax=scalar_max,
        )
        if scalar_mappable is None:
            scalar_mappable = scalar_plot

        axs[i, 0].set_title(f"{label} Scalar Field")
        axs[i, 0].set_xlabel("Longitude")
        axs[i, 0].set_ylabel("Latitude")
        axs[i, 1].quiver(lon_2d, lat_2d, u_field, v_field, scale=quiver_scale)
        axs[i, 1].set_title(f"{label} Vector Field")
        axs[i, 1].set_xlabel("Longitude")
        axs[i, 1].set_ylabel("Latitude")

    for ax in axs.flat:
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)

    if scalar_mappable is None:
        raise ValueError("No scalar field was plotted")
    return fig, axs, scalar_mappable
