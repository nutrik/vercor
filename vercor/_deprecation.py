from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeAlias
import warnings

_AliasTarget: TypeAlias = object | tuple[str, object]


def warn_deprecated_name(old: str, new: str, *, remove_in: str) -> None:
    """Emit a standard deprecation warning for renamed public API symbols."""

    warnings.warn(
        f"{old} is deprecated; use {new}. It will be removed in {remove_in}.",
        DeprecationWarning,
        stacklevel=3,
    )


def deprecated_getattr(
    module_name: str,
    aliases: Mapping[str, _AliasTarget],
    *,
    remove_in: str,
) -> Callable[[str], object]:
    """Return a module ``__getattr__`` serving deprecated aliases with warnings."""

    def module_getattr(name: str) -> object:
        try:
            alias_target = aliases[name]
        except KeyError as exc:
            raise AttributeError(
                f"module {module_name!r} has no attribute {name!r}"
            ) from exc

        replacement, value = _alias_replacement(module_name, name, alias_target)
        warn_deprecated_name(
            f"{module_name}.{name}",
            replacement,
            remove_in=remove_in,
        )
        return value

    return module_getattr


def _alias_replacement(
    module_name: str,
    name: str,
    target: _AliasTarget,
) -> tuple[str, object]:
    """Resolve an alias target and replacement display name."""

    if isinstance(target, tuple) and len(target) == 2 and isinstance(target[0], str):
        return target[0], target[1]

    replacement_name = getattr(target, "__name__", name)
    return f"{module_name}.{replacement_name}", target


__all__ = ["deprecated_getattr", "warn_deprecated_name"]
