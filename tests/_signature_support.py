"""Stable rendering support for source and installed public-signature tests."""

from __future__ import annotations

import re

EXTERNAL_TYPING_ALIAS_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (
        "jax.Array | numpy.ndarray | numpy.bool | numpy.number | bool | int | "
        "float | complex",
        "jax.typing.ArrayLike",
    ),
    (
        "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, "
        "float, complex, jax._src.literals.TypedNdArray]",
        "jax.typing.ArrayLike",
    ),
    (
        "Union[jax.Array, numpy.ndarray, numpy.bool, numpy.number, bool, int, "
        "float, complex]",
        "jax.typing.ArrayLike",
    ),
    (
        "numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[typing.Any]]",
        "numpy.typing.NDArray[typing.Any]",
    ),
    ("NDArray[typing.Any]", "numpy.typing.NDArray[typing.Any]"),
)

_EXTERNAL_TYPING_ALIAS_PATTERN = re.compile(
    r"(?<![\w.])(?:"
    + "|".join(
        re.escape(dependency_rendering)
        for dependency_rendering, _ in EXTERNAL_TYPING_ALIAS_REPLACEMENTS
    )
    + r")(?![\w.])"
)


def canonicalize_external_typing_aliases(rendered: str) -> str:
    """Replace evidenced dependency-sensitive aliases with public tokens."""

    replacements = dict(EXTERNAL_TYPING_ALIAS_REPLACEMENTS)
    return _EXTERNAL_TYPING_ALIAS_PATTERN.sub(
        lambda match: replacements[match.group(0)],
        rendered,
    )
