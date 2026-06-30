from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RunSequence:
    """Public ordered component-name schedule used by ``Coupler.run()``."""

    order: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[str]:
        return iter(self.order)


def normalize_run_sequence(run_sequence: "RunSequence | Sequence[str]") -> RunSequence:
    """Return ``run_sequence`` as a ``RunSequence`` compatibility wrapper."""

    if isinstance(run_sequence, RunSequence):
        return run_sequence
    if isinstance(run_sequence, str):
        raise TypeError("run_sequence must be a sequence of component names, not str")
    return RunSequence(order=list(run_sequence))
