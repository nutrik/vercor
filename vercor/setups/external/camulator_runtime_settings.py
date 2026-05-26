from __future__ import annotations

import os


def configure_camulator_runtime() -> None:
    """Apply environment defaults for CAMulator host-runtime execution."""

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


__all__ = ["configure_camulator_runtime"]
