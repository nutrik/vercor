from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Iterator, overload
import warnings


@dataclass(frozen=True)
class RunSequence(Sequence[str]):
    """Deprecated compatibility adapter for ordered component-name schedules."""

    order: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        warnings.warn(
            "RunSequence is deprecated; pass a plain sequence of component names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, "order", _normalize_run_order(self.order))

    def __iter__(self) -> Iterator[str]:
        return iter(self.order)

    def __len__(self) -> int:
        return len(self.order)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[str, ...]: ...

    def __getitem__(self, index: int | slice) -> str | tuple[str, ...]:
        return tuple(self.order)[index]


def _normalize_run_order(run_sequence: Sequence[str]) -> tuple[str, ...]:
    """Return ``run_sequence`` as an immutable component-name tuple."""

    if isinstance(run_sequence, str):
        raise TypeError("run_sequence must be a sequence of component names, not str")
    return tuple(run_sequence)


def normalize_run_sequence(run_sequence: Sequence[str]) -> tuple[str, ...]:
    """Return ``run_sequence`` as an immutable component-name tuple."""

    return _normalize_run_order(run_sequence)
