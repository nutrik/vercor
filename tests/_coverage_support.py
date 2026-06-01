from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import io
import logging

import numpy as np
from numpy.typing import NDArray

from vercor.clock import Clock
from vercor.components import DataComponent
from vercor.grid import RectilinearGrid
from vercor.run_sequence import RunSequence
from vercor.components.contexts import ComponentSetupContext, ComponentStepContext
from vercor.settings import VercorSettings
from vercor.types import RuntimeArray


@contextmanager
def capture_logger_output(
    logger_name: str,
    level: int = logging.INFO,
    set_logger_level: bool = True,
) -> Iterator[io.StringIO]:
    logger = logging.getLogger(logger_name)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, "_vercor_canonical_handler", True)
    previous_level = logger.level
    logger.addHandler(handler)
    if set_logger_level:
        logger.setLevel(level)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        if set_logger_level:
            logger.setLevel(previous_level)
        handler.close()


def make_test_grid(
    name: str = "grid",
    *,
    longitude: NDArray | None = None,
    latitude: NDArray | None = None,
    binary_mask: NDArray | None = None,
) -> RectilinearGrid:
    lon = (
        np.asarray(longitude, dtype=float)
        if longitude is not None
        else np.asarray([0.0, 1.0], dtype=float)
    )
    lat = (
        np.asarray(latitude, dtype=float)
        if latitude is not None
        else np.asarray([-1.0, 1.0], dtype=float)
    )
    mask = None if binary_mask is None else np.asarray(binary_mask)
    return RectilinearGrid(name=name, longitude=lon, latitude=lat, binary_mask=mask)


@dataclass
class CoverageCouplerStub:
    clock: Clock = field(
        default_factory=lambda: Clock(
            start=datetime(2000, 1, 1),
            dt_seconds=60.0,
            steps=1,
        )
    )
    settings: VercorSettings = field(default_factory=VercorSettings)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("coverage-tests")
    )
    run_sequence: RunSequence = field(default_factory=lambda: RunSequence(order=[]))

    def init_context(self) -> ComponentSetupContext:
        return ComponentSetupContext(
            start=self.clock.start,
            dt_seconds=self.clock.dt_seconds,
            run_sequence=self.run_sequence,
            settings=self.settings,
            logger=self.logger,
        )

    def step_context(
        self, *, time: datetime | None = None, with_logger: bool = True
    ) -> ComponentStepContext:
        return ComponentStepContext(
            dt_seconds=self.clock.dt_seconds,
            settings=self.settings,
            time=time,
            logger=self.logger if with_logger else None,
        )


class DummyComponent(DataComponent):
    def initialize(self, context: ComponentSetupContext) -> None:
        _ = context
        self.data.setdefault("temperature", np.zeros(self.grid.shape, dtype=float))


class RecordingRegridder:
    def __init__(
        self,
        *,
        scalar_result: RuntimeArray | None = None,
        vector_result: tuple[RuntimeArray, RuntimeArray] | None = None,
    ) -> None:
        self.scalar_result = scalar_result
        self.vector_result = vector_result
        self.calls: list[tuple[NDArray, ...]] = []

    def __call__(
        self, *args: RuntimeArray
    ) -> RuntimeArray | tuple[RuntimeArray, RuntimeArray]:
        self.calls.append(tuple(np.asarray(arg) for arg in args))
        if len(args) == 1:
            if self.scalar_result is not None:
                return self.scalar_result
            return np.asarray(args[0])

        if self.vector_result is not None:
            return self.vector_result
        return np.asarray(args[0]), np.asarray(args[1])
