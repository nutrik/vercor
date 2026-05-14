from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta


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
