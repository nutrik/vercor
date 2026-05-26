"""Compatibility facade for CAMulator wind artifact filtering."""

from vercor.setups.external.camulator_wind_filter import (
    WindArtifactFilterConfig,
    apply_wind_artifact_filter_to_tensor,
    load_wind_filter_config,
    post_process_wind_artifacts,
    simple_wind_artifact_filter,
    wind_filter,
)

__all__ = [
    "WindArtifactFilterConfig",
    "apply_wind_artifact_filter_to_tensor",
    "load_wind_filter_config",
    "post_process_wind_artifacts",
    "simple_wind_artifact_filter",
    "wind_filter",
]
