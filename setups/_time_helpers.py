from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, TypeVar

from vercor.components.base import Component

SpinupResult = TypeVar("SpinupResult")


@dataclass(frozen=True)
class ModelTimestepAlignment:
    """Coupler/model timestep relationship used by setup adapters."""

    coupling_timestep: timedelta
    model_timestep: timedelta
    model_substeps: int


def align_model_timestep(
    dt_seconds: float,
    model_timestep: timedelta,
    *,
    coupling_name: str = "coupling_timestep",
    model_name: str = "model_timestep",
) -> ModelTimestepAlignment:
    """Validate that a model timestep divides one VerCOR coupling timestep."""

    coupling_timestep = timedelta(seconds=float(dt_seconds))
    if model_timestep.total_seconds() <= 0.0:
        raise ValueError(f"{model_name} ({model_timestep}) must be positive")

    if coupling_timestep % model_timestep != timedelta(days=0):
        raise ValueError(
            f"{model_name} ({model_timestep}) must be a multiple of "
            f"{coupling_name} ({coupling_timestep})"
        )

    return ModelTimestepAlignment(
        coupling_timestep=coupling_timestep,
        model_timestep=model_timestep,
        model_substeps=int(
            coupling_timestep.total_seconds() // model_timestep.total_seconds()
        ),
    )


def assign_model_timestep_alignment(
    target: Any,
    dt_seconds: float,
    model_timestep: timedelta,
    *,
    coupling_name: str = "coupling_timestep",
    model_name: str = "model_timestep",
) -> ModelTimestepAlignment:
    """Store common coupling/model timestep attributes on a setup-state object."""

    alignment = align_model_timestep(
        dt_seconds,
        model_timestep,
        coupling_name=coupling_name,
        model_name=model_name,
    )
    target.coupling_timestep = alignment.coupling_timestep
    target.model_timestep = alignment.model_timestep
    target.model_substeps = alignment.model_substeps
    return alignment


def runtime_forcing_index(
    *,
    start_ix: int,
    timestep_counter: int,
    model_substeps: int,
) -> int:
    """Return the forcing index for a setup adapter's current coupling step."""

    return int(start_ix) + int(timestep_counter) * int(model_substeps)


def run_logged_spinup(
    *,
    steps: int,
    logger: Any,
    intro_message: str,
    step_message: Callable[[int, int], str],
    step: Callable[[int], SpinupResult],
) -> SpinupResult | None:
    """Run a setup spinup loop with consistent one-line progress logging."""

    logger.info(intro_message)
    result: SpinupResult | None = None
    for step_number in range(1, int(steps) + 1):
        logger.info(step_message(step_number, int(steps)))
        result = step(step_number)
    return result


def seed_grid_field_defaults(
    component: Component,
    field_names: Sequence[str],
    context: Any,
    *,
    overrides: dict[str, object] | None = None,
) -> None:
    """Seed a component's grid-shaped default fields through one shared path."""

    component.seed_fields(
        component.grid_field_defaults(
            field_names,
            overrides=overrides,
            policy=context.settings,
        )
    )
