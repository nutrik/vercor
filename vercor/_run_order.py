from __future__ import annotations

from collections.abc import Sequence


def normalize_run_sequence(run_sequence: Sequence[str]) -> tuple[str, ...]:
    """Return ``run_sequence`` as an immutable component-name tuple."""

    if isinstance(run_sequence, str):
        raise TypeError("run_sequence must be a sequence of component names, not str")
    return tuple(run_sequence)
