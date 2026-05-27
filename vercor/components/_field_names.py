from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

FieldNames: TypeAlias = Iterable[str]


def unique_field_names(field_names: FieldNames) -> tuple[str, ...]:
    """Return field names without duplicates while preserving order."""

    unique: list[str] = []
    for field_name in field_names:
        if field_name not in unique:
            unique.append(field_name)
    return tuple(unique)


__all__ = ["unique_field_names"]
