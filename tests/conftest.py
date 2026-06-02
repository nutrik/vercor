from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Protocol, TypeVar

import jax
import pytest

jax.config.update("jax_enable_x64", True)

_TEST_CACHE_ROOT = Path(tempfile.gettempdir())
_PLOTTING_CACHE_ENV_DEFAULTED = {
    "MPLBACKEND": "MPLBACKEND" not in os.environ,
    "MPLCONFIGDIR": "MPLCONFIGDIR" not in os.environ,
    "XDG_CACHE_HOME": "XDG_CACHE_HOME" not in os.environ,
}
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(_TEST_CACHE_ROOT / "vercor-matplotlib-cache"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(_TEST_CACHE_ROOT / "vercor-xdg-cache"),
)

CaseT = TypeVar("CaseT")


class SelectFastCases(Protocol):
    def __call__(
        self,
        cases: Sequence[CaseT],
        *,
        case_id: Callable[[CaseT], str] = repr,
        target_fraction: float = 0.1,
        min_cases: int = 1,
    ) -> list[CaseT]: ...


def _stable_rank(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_fast_case_subset(
    cases: Sequence[CaseT],
    *,
    nodeid: str,
    case_id: Callable[[CaseT], str] = repr,
    target_fraction: float = 0.1,
    min_cases: int = 1,
) -> list[CaseT]:
    """Return a deterministic subsample of cases for `--fast` runs."""

    if not cases:
        return []

    modulus = max(1, round(1.0 / target_fraction))
    ranked_cases = sorted(
        ((_stable_rank(f"{nodeid}:{case_id(case)}"), case) for case in cases),
        key=lambda item: item[0],
    )

    selected = [case for rank, case in ranked_cases if rank % modulus == 0]
    if len(selected) >= min_cases:
        return selected

    return [case for _, case in ranked_cases[:min_cases]]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Run a deterministic subsample of the unit-test suite.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "fast_always: always include this test in --fast mode",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--fast"):
        return

    items_by_path: dict[str, list[pytest.Item]] = defaultdict(list)
    for item in items:
        items_by_path[str(item.path)].append(item)

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for path_items in items_by_path.values():
        always_keep = [
            item
            for item in path_items
            if item.get_closest_marker("fast_always") is not None
        ]
        candidate_items = [item for item in path_items if item not in always_keep]
        keep_count = (
            max(1, math.ceil(len(candidate_items) * 0.1)) if candidate_items else 0
        )

        ranked_candidates = sorted(
            candidate_items, key=lambda item: _stable_rank(item.nodeid)
        )
        keep_items = always_keep + ranked_candidates[:keep_count]
        keep_ids = {id(item) for item in keep_items}

        for item in path_items:
            if id(item) in keep_ids:
                selected.append(item)
            else:
                deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture
def fast_mode(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--fast"))


@pytest.fixture
def select_fast_cases(
    request: pytest.FixtureRequest, fast_mode: bool
) -> SelectFastCases:
    def _select(
        cases: Sequence[CaseT],
        *,
        case_id: Callable[[CaseT], str] = repr,
        target_fraction: float = 0.1,
        min_cases: int = 1,
    ) -> list[CaseT]:
        if not fast_mode:
            return list(cases)

        return select_fast_case_subset(
            cases,
            nodeid=request.node.nodeid,
            case_id=case_id,
            target_fraction=target_fraction,
            min_cases=min_cases,
        )

    return _select
