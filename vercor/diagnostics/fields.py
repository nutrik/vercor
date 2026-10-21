from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from vercor.state import (
    ComponentState,
)
from vercor.types import RuntimeArray

ComponentMetric = str | Callable[[ComponentState], RuntimeArray | float]


def component_vector_speed(
    component_state: ComponentState,
    u_field: str = "air_u_velocity",
    v_field: str = "air_v_velocity",
) -> RuntimeArray:
    """Return vector speed from a runtime component state."""

    u = jnp.asarray(component_state.field(u_field))
    v = jnp.asarray(component_state.field(v_field))
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


def total_surface_temperature(component: ComponentState) -> RuntimeArray:
    """Return combined land and sea surface temperature for diagnostics."""

    return combine_surface_temperatures(
        component.field("land_surface_temperature"),
        component.field("sea_surface_temperature"),
    )


def safe_component_nanmean(component: ComponentState, field_name: str) -> float:
    """Return a robust NaN-aware mean for a runtime component view field."""

    try:
        return float(jnp.nanmean(jnp.asarray(component.field(field_name))))
    except Exception:
        return float("nan")


def component_plot_field(
    component: ComponentState,
    field_name: str,
) -> RuntimeArray:
    """Return a 2D field suitable for plotting when one is available."""

    candidates = _component_field_candidates(component, field_name)
    for candidate in candidates:
        if jnp.asarray(candidate).ndim == 2:
            return candidate
    if candidates:
        return candidates[0]
    raise KeyError(f"Field {field_name!r} not found")


def _component_field_candidates(
    component: ComponentState,
    field_name: str,
) -> list[RuntimeArray]:
    """Return all matching fields in public scope-resolution order."""

    candidates: list[RuntimeArray] = []
    for _, name, value in component.iter_fields("state", "received", "sent"):
        if name == field_name:
            candidates.append(value)
    return candidates


def component_plot_scalar(
    component: ComponentState,
    scalar: ComponentMetric,
) -> RuntimeArray | float:
    """Resolve a field name or callable diagnostic for plotting."""

    if isinstance(scalar, str):
        return component_plot_field(component, scalar)
    return scalar(component)


def safe_component_metric_mean(
    component: ComponentState,
    metric: ComponentMetric,
) -> float:
    """Resolve a metric and return a robust mean value as float."""

    if isinstance(metric, str):
        return safe_component_nanmean(component, metric)
    try:
        return float(jnp.nanmean(jnp.asarray(metric(component))))
    except Exception:
        return float("nan")
