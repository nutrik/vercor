from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from vercor.runtime.views import (
    RuntimeComponentView,
    RuntimeFieldSource,
    runtime_field,
    runtime_field_candidates,
)
from vercor.types import RuntimeArray

ComponentMetric = str | Callable[[RuntimeComponentView], RuntimeArray | float]


def component_vector_speed(
    component_state: RuntimeFieldSource,
    u_field: str = "u_velocity",
    v_field: str = "v_velocity",
) -> RuntimeArray:
    """Return vector speed from a runtime component state."""

    u = jnp.asarray(runtime_field(component_state, u_field))
    v = jnp.asarray(runtime_field(component_state, v_field))
    return jnp.sqrt(u**2 + v**2)


def combine_surface_temperatures(
    land_surface_temperature: RuntimeArray,
    sea_surface_temperature: RuntimeArray,
) -> RuntimeArray:
    """Merge land and sea surface temperatures while treating NaNs as missing."""

    return jnp.nan_to_num(
        jnp.asarray(land_surface_temperature),
        nan=0.0,
    ) + jnp.nan_to_num(
        jnp.asarray(sea_surface_temperature),
        nan=0.0,
    )


def total_surface_temperature(component: RuntimeComponentView) -> RuntimeArray:
    """Return combined land and sea surface temperature for diagnostics."""

    return combine_surface_temperatures(
        runtime_field(component, "land_surface_temperature"),
        runtime_field(component, "sea_surface_temperature"),
    )


def safe_component_nanmean(component: RuntimeComponentView, field_name: str) -> float:
    """Return a robust NaN-aware mean for a runtime component view field."""

    try:
        return float(jnp.nanmean(jnp.asarray(runtime_field(component, field_name))))
    except Exception:
        return float("nan")


def component_plot_field(
    component: RuntimeComponentView,
    field_name: str,
) -> RuntimeArray:
    """Return a 2D field suitable for plotting when one is available."""

    candidates = runtime_field_candidates(component, field_name)
    for candidate in candidates:
        if jnp.asarray(candidate).ndim == 2:
            return candidate
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {field_name!r} not found")


def component_plot_scalar(
    component: RuntimeComponentView,
    scalar: ComponentMetric,
) -> RuntimeArray | float:
    """Resolve a field name or callable diagnostic for plotting."""

    if isinstance(scalar, str):
        return component_plot_field(component, scalar)
    return scalar(component)


def safe_component_metric_mean(
    component: RuntimeComponentView,
    metric: ComponentMetric,
) -> float:
    """Resolve a metric and return a robust mean value as float."""

    if isinstance(metric, str):
        return safe_component_nanmean(component, metric)
    try:
        return float(jnp.nanmean(jnp.asarray(metric(component))))
    except Exception:
        return float("nan")
