# Calendar Year Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `vercor.calendar` the sole owner of calendar-derived model-year duration and remove `RuntimeOptions.model_year_seconds` without redesigning any other runtime behavior.

**Architecture:** A public `YearType` and two pure calendar helpers will define leap, no-leap, and 360-day year duration. Private runtime time metadata will resolve a year type for every timestamp from the existing `Clock.calendar`, so monthly and daily forcing share one calendar source of truth. `RuntimeOptions` will return to owning only dtype, backend, workflow, and topology policy.

**Tech Stack:** Python 3.12+, frozen dataclasses, `enum.StrEnum`, JAX, NumPy, pytest, mypy, Black, and flake8.

## Global Constraints

- Preserve `Clock(calendar="gregorian" | "noleap" | "360_day")`, clock iteration, daily forcing lookup, host/JAX execution, differentiation, and output timestamp behavior.
- Do not add a second run-level year-type or calendar setting.
- Do not retain a compatibility alias for `RuntimeOptions.model_year_seconds`.
- Resolve Gregorian leap status per timestamp because one run may cross calendar years.
- Keep the historical frozen public API contract and archived progress documents unchanged.
- Follow red-green-refactor: no production change may precede its failing regression test.
- Use the direct `scipy` interpreter at `/Users/romannuterman/miniforge3/envs/scipy/bin/python`; the Conda launcher is known to panic in this checkout.
- Run the complete unit suite before each implementation commit, as required by `AGENTS.md`.

---

## File Map

- `vercor/calendar.py`: canonical `YearType`, calendar-to-year-type resolution, and model-year seconds.
- `vercor/forcing_index.py`: compatibility type alias and daily forcing consumption of calendar-owned `YearType`.
- `vercor/clock.py`: existing clock API using calendar validation without owning a second year-type mapping.
- `vercor/_runtime/time.py`: per-timestamp monthly and daily forcing metadata derived from calendar ownership.
- `vercor/_runtime/backends.py`, `vercor/_runtime/preparation.py`: callers of the simplified runtime time API.
- `vercor/runtime/__init__.py`: `RuntimeOptions` after removal of the model-year field and validation.
- `tests/test_tools_time_and_forcing.py`: calendar API, forcing compatibility, and direct runtime metadata regressions.
- `tests/test_coupler_runtime.py`: end-to-end monthly forcing and gradient regressions.
- `tests/test_runtime_api_boundaries.py`, `tests/test_v0_4_workflows.py`, `tests/test_plugin_architecture.py`, `tests/test_final_review_boundaries.py`, `tests/test_v0_4_public_api.py`: corrected ownership and public-boundary assertions.
- `tests/test_api_architecture_review.py`, `tests/test_distribution_boundaries.py`: exclude inherited builtin string methods from VerCOR-owned method inventory.
- `tests/contracts/vercor-0.4.0a1-public-signatures.json`: exact new calendar callables and corrected `RuntimeOptions` signature.
- `DESIGN.md`, `DEPENDENCIES.md`, `docs/api-architecture-review.md`, legacy migration guidance, `PROGRESS.md`: implemented ownership, dependency, migration, and verification evidence.

---

### Task 1: Establish the calendar-owned year API

**Files:**

- Modify: `tests/test_tools_time_and_forcing.py`
- Modify: `tests/test_api_architecture_review.py`
- Modify: `tests/test_distribution_boundaries.py`
- Modify: `vercor/calendar.py`
- Modify: `vercor/forcing_index.py`
- Modify: `vercor/clock.py`
- Modify: `docs/api-architecture-review.md`
- Modify: `tests/contracts/vercor-0.4.0a1-public-signatures.json`

**Interfaces:**

- Produces: `YearType`, `model_year_seconds(year_type)`, and `year_type_for_calendar(calendar, year)` in `vercor.calendar`.
- Preserves: string inputs `"leap"`, `"noleap"`, and `"360"` for `daily_forcing_day_of_year` and `daily_forcing_index`.
- Consumed later by: `vercor._runtime.time` in Task 2.

- [ ] **Step 1: Write failing calendar ownership tests**

Add these imports and tests to `tests/test_tools_time_and_forcing.py`:

```python
from typing import Any, cast

from vercor.calendar import (
    YearType,
    model_year_seconds,
    year_type_for_calendar,
)


@pytest.mark.fast_always
def test_calendar_owns_canonical_year_types_and_durations() -> None:
    assert tuple(YearType) == (
        YearType.GREGORIAN_LEAP,
        YearType.GREGORIAN_NO_LEAP,
        YearType.DAY_360,
    )
    assert YearType.GREGORIAN_LEAP == "leap"
    assert YearType.GREGORIAN_NO_LEAP == "noleap"
    assert YearType.DAY_360 == "360"
    assert model_year_seconds(YearType.GREGORIAN_LEAP) == 366 * 86_400.0
    assert model_year_seconds(YearType.GREGORIAN_NO_LEAP) == 365 * 86_400.0
    assert model_year_seconds(YearType.DAY_360) == 360 * 86_400.0


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("calendar", "year", "expected"),
    (
        ("gregorian", 2000, YearType.GREGORIAN_LEAP),
        ("gregorian", 1900, YearType.GREGORIAN_NO_LEAP),
        ("gregorian", 2001, YearType.GREGORIAN_NO_LEAP),
        ("noleap", 2000, YearType.GREGORIAN_NO_LEAP),
        ("360_day", 2000, YearType.DAY_360),
    ),
)
def test_calendar_resolves_year_type_from_existing_clock_policy(
    calendar: str,
    year: int,
    expected: YearType,
) -> None:
    assert year_type_for_calendar(calendar, year) is expected


@pytest.mark.fast_always
def test_calendar_year_helpers_reject_foreign_policy_values() -> None:
    with pytest.raises(TypeError, match="year_type must be a YearType"):
        model_year_seconds(cast(Any, "leap"))
    with pytest.raises(ValueError, match="calendar must be one of"):
        year_type_for_calendar("julian", 2000)
```

Extend `test_forcing_index_resolves_daily_forcing_calendar_cases` with one
assertion that an enum member is accepted alongside existing strings:

```python
    assert (
        forcing_index_module.daily_forcing_index(
            datetime(2001, 1, 1),
            year_type=YearType.GREGORIAN_LEAP,
        )
        == 0
    )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_tools_time_and_forcing.py -q --fast
```

Expected: collection fails because `YearType`, `model_year_seconds`, and
`year_type_for_calendar` are not exported by `vercor.calendar`.

- [ ] **Step 3: Implement the minimal calendar API**

Add `StrEnum` and the new names to `vercor/calendar.py`:

```python
from enum import StrEnum

__all__ = [
    "CalendarDate",
    "DAYS_PER_MONTH_360",
    "DAYS_PER_MONTH_GREGORIAN_LEAP",
    "DAYS_PER_MONTH_GREGORIAN_NO_LEAP",
    "DateTime360",
    "DateTime365",
    "ModelDateTime",
    "YearType",
    "day_of_year_from_month_day",
    "is_leap_year",
    "model_year_seconds",
    "month_day_from_day_of_year",
    "year_type_for_calendar",
]


class YearType(StrEnum):
    """Identify the year structure used by one simulated timestamp."""

    GREGORIAN_LEAP = "leap"
    GREGORIAN_NO_LEAP = "noleap"
    DAY_360 = "360"


def model_year_seconds(year_type: YearType) -> float:
    """Return the canonical duration in seconds for ``year_type``."""

    if not isinstance(year_type, YearType):
        raise TypeError("year_type must be a YearType")
    if year_type is YearType.GREGORIAN_LEAP:
        return 366 * 86_400.0
    if year_type is YearType.GREGORIAN_NO_LEAP:
        return 365 * 86_400.0
    return 360 * 86_400.0


def year_type_for_calendar(calendar: str, year: int) -> YearType:
    """Resolve one timestamp's year structure from a clock calendar."""

    if calendar == "gregorian":
        if is_leap_year(year):
            return YearType.GREGORIAN_LEAP
        return YearType.GREGORIAN_NO_LEAP
    if calendar == "noleap":
        return YearType.GREGORIAN_NO_LEAP
    if calendar == "360_day":
        return YearType.DAY_360
    raise ValueError("calendar must be one of: 'gregorian', 'noleap', '360_day'")
```

Place the helpers after `is_leap_year` so their dependency is explicit.

- [ ] **Step 4: Make forcing indices consume the calendar type**

Replace the private literal owner in `vercor/forcing_index.py` with a PEP 695
compatibility alias and enum conversion:

```python
from vercor.calendar import (
    CalendarDate as _CalendarDate,
    DAYS_PER_MONTH_GREGORIAN_LEAP as _DAYS_PER_MONTH_GREGORIAN_LEAP,
    DAYS_PER_MONTH_GREGORIAN_NO_LEAP as _DAYS_PER_MONTH_GREGORIAN_NO_LEAP,
    YearType as _YearType,
    day_of_year_from_month_day as _day_of_year_from_month_day,
    is_leap_year as _is_leap_year,
)

type ForcingYearType = _YearType


def _validate_year_type(year_type: str | _YearType) -> _YearType:
    try:
        return _YearType(year_type)
    except ValueError as exc:
        raise ValueError(
            "year_type must be one of: 'leap', 'noleap', '360'"
        ) from exc
```

Compare validated values by enum identity in `daily_forcing_day_of_year`:

```python
    if validated_year_type is _YearType.DAY_360:
        return day_of_year_360_to_gregorian(
            cast(_CalendarDate, time),
            no_leap=no_leap,
        )
    if validated_year_type is _YearType.GREGORIAN_NO_LEAP:
        return noleap_day_of_year(cast(_CalendarDate, time))
```

Keep the public function annotations as `year_type: str`; this preserves the
documented calling contract while `StrEnum` values remain accepted because
they are strings.

- [ ] **Step 5: Delegate Clock calendar validation to the calendar owner**

In `Clock.__init__`, replace the local membership tuple with:

```python
        _calendar.year_type_for_calendar(calendar, start.year)
```

Delete the private `YearType` alias and `_forcing_year_type_for_calendar` from
`vercor/clock.py`. In `Clock.__post_init__` and `_day_of_year_for_start`, branch
directly on the already validated public calendar string:

```python
        if self.calendar != "gregorian":
            datetime_class: type[_calendar.DateTime365] | type[_calendar.DateTime360]
            if self.calendar == "noleap":
                datetime_class = _calendar.DateTime365
            else:
                datetime_class = _calendar.DateTime360
```

```python
        if self.calendar == "360_day":
            if start.day > 30:
                raise ValueError(
                    "for calendar='360_day', start day must be between 1 and 30"
                )
            return (start.month - 1) * 30 + start.day
```

- [ ] **Step 6: Make signature inventory treat inherited builtin methods as builtin**

In both `_canonical_public_method_names` in
`tests/test_api_architecture_review.py` and the installed probe embedded in
`tests/test_distribution_boundaries.py`, add this predicate beside
`inspect.isroutine(method)`:

```python
and getattr(method, "__module__", None) is not None
```

This keeps the existing 55 VerCOR-owned/inherited Python methods while avoiding
freezing every inherited `str` method as a `YearType` behavior.

- [ ] **Step 7: Update the executable public calendar contract**

Update the `vercor.calendar` array in `docs/api-architecture-review.md` to the
exact `__all__` order from Step 3. Change the documented concrete callable
count from 147 to 150.

Add these exact sorted entries to the `exports` object in
`tests/contracts/vercor-0.4.0a1-public-signatures.json`:

```json
"vercor.calendar.YearType": "(*values)",
"vercor.calendar.model_year_seconds": "(year_type: vercor.calendar.YearType) -> float",
"vercor.calendar.year_type_for_calendar": "(calendar: str, year: int) -> vercor.calendar.YearType"
```

- [ ] **Step 8: Verify Task 1 GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_tools_time_and_forcing.py \
  tests/test_clock.py \
  tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py -q --fast
```

Expected: all selected tests pass, existing string forcing cases remain green,
and the source/installed public manifests include the calendar-owned API.

- [ ] **Step 9: Format and run the complete suite before committing**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q
git diff --check
```

Expected: Black exits 0, the full suite has zero failures, and whitespace is
clean.

- [ ] **Step 10: Commit the calendar owner**

```bash
git add vercor/calendar.py vercor/forcing_index.py vercor/clock.py \
  tests/test_tools_time_and_forcing.py tests/test_api_architecture_review.py \
  tests/test_distribution_boundaries.py \
  tests/contracts/vercor-0.4.0a1-public-signatures.json \
  docs/api-architecture-review.md
git commit -m "feat: centralize model-year policy in calendar"
```

---

### Task 2: Remove runtime ownership and derive time metadata from each timestamp

**Files:**

- Modify: `tests/test_tools_time_and_forcing.py`
- Modify: `tests/test_coupler_runtime.py`
- Modify: `tests/test_runtime_api_boundaries.py`
- Modify: `tests/test_v0_4_workflows.py`
- Modify: `tests/test_plugin_architecture.py`
- Modify: `tests/test_final_review_boundaries.py`
- Modify: `tests/test_v0_4_public_api.py`
- Modify: `vercor/_runtime/time.py`
- Modify: `vercor/_runtime/backends.py`
- Modify: `vercor/_runtime/preparation.py`
- Modify: `vercor/runtime/__init__.py`
- Modify: `tests/contracts/vercor-0.4.0a1-public-signatures.json`
- Modify: `DESIGN.md`
- Modify: `DEPENDENCIES.md`
- Modify: `docs/api-architecture-review.md`
- Modify: legacy migration guidance
- Modify: `PROGRESS.md`

**Interfaces:**

- Consumes: the three calendar interfaces from Task 1.
- Produces: `RuntimeOptions(dtype, backend, workflow, topology)` with no
  `model_year_seconds` parameter or attribute.
- Preserves: per-step `RuntimeStepInfo` PyTree structure and downstream field
  transfer interfaces.

- [ ] **Step 1: Write failing runtime ownership and calendar-duration tests**

Add these imports and tests to `tests/test_tools_time_and_forcing.py`:

```python
import jax.numpy as jnp

from vercor._runtime.time import runtime_step_info_from_times


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("time", "calendar", "year_type"),
    (
        (datetime(2000, 7, 2, 12), "gregorian", YearType.GREGORIAN_LEAP),
        (datetime(2001, 7, 2, 12), "gregorian", YearType.GREGORIAN_NO_LEAP),
        (
            DateTime365(2000, 7, 2, 12, 0, 0, 0, 183),
            "noleap",
            YearType.GREGORIAN_NO_LEAP,
        ),
        (
            DateTime360(2000, 7, 2, 12, 0, 0, 0, 182),
            "360_day",
            YearType.DAY_360,
        ),
    ),
)
def test_runtime_monthly_metadata_uses_timestamp_calendar_duration(
    time: datetime | DateTime360 | DateTime365,
    calendar: str,
    year_type: YearType,
) -> None:
    info = runtime_step_info_from_times(
        [time],
        calendar=calendar,
    )
    expected_left, expected_right = get_periodic_interval(
        current_time=datetime_to_seconds_in_year(time),
        cycle_length=model_year_seconds(year_type),
        rec_spacing=model_year_seconds(year_type) / 12.0,
        n_rec=12,
    )

    assert int(info.monthly_index_left[0]) == expected_left[0]
    assert int(info.monthly_index_right[0]) == expected_right[0]
    assert jnp.isclose(info.monthly_weight_left[0], expected_left[1])
    assert jnp.isclose(info.monthly_weight_right[0], expected_right[1])
```

Replace the current default assertion in `tests/test_v0_4_workflows.py` with:

```python
    assert not hasattr(options, "model_year_seconds")
    assert "model_year_seconds" not in inspect.signature(
        runtime.RuntimeOptions
    ).parameters
```

Replace the numeric validation/canonicalization block in
`tests/test_final_review_boundaries.py` with:

```python
@pytest.mark.fast_always
def test_runtime_options_rejects_removed_model_year_owner() -> None:
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        RuntimeOptions(model_year_seconds=365 * 86_400.0)  # type: ignore[call-arg]

    assert not hasattr(RuntimeOptions(), "model_year_seconds")
```

- [ ] **Step 2: Run the ownership tests and verify RED**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_tools_time_and_forcing.py \
  tests/test_v0_4_workflows.py \
  tests/test_final_review_boundaries.py -q --fast
```

Expected: failures show that `runtime_step_info_from_times` does not accept
`calendar` and `RuntimeOptions` still exposes `model_year_seconds`.

- [ ] **Step 3: Simplify RuntimeOptions to its four remaining policies**

In `vercor/runtime/__init__.py`, delete `math.isfinite`, `numbers.Real`, the
`model_year_seconds` docstring entry and field, and the complete validation and
normalization block for that field. The resulting declaration is:

```python
@dataclass(frozen=True)
class RuntimeOptions:
    """Own immutable runtime policy.

    Attributes:
        dtype: Precision policy applied at the prepared runtime boundary.
        backend: ``"auto"`` selects host execution only when a scheduled
            component is host-backed; ``"jax"`` forces scanned JAX execution
            and rejects scheduled host components; ``"host"`` forces the
            Python driver; an :class:`ExecutionBackend` handles core-defined
            chunks through the public driver.
        workflow: Static plan builder. Defaults to :class:`SequentialWorkflow`.
        topology: Optional topology policy, or ``None`` for no topology patch.
    """

    dtype: _dtypes.DTypePolicy = field(default_factory=_dtypes.DTypePolicy)
    backend: Literal["auto", "jax", "host"] | "ExecutionBackend" = "auto"
    workflow: Workflow = field(default_factory=SequentialWorkflow)
    topology: _TopologyPolicy | None = None
```

Retain the existing `__post_init__` checks for these four policies unchanged.

- [ ] **Step 4: Derive runtime metadata from calendar for every timestamp**

Change `runtime_step_info_from_times` in `vercor/_runtime/time.py` to:

```python
def runtime_step_info_from_times(
    times: Sequence[datetime | ModelDateTime],
    *,
    calendar: str,
) -> RuntimeStepInfo:
    """Build calendar-derived time-selection metadata for timestamps."""

    monthly_index_left: list[int] = []
    monthly_index_right: list[int] = []
    monthly_weight_left: list[float] = []
    monthly_weight_right: list[float] = []
    daily_index: list[int] = []

    for time in times:
        year_type = year_type_for_calendar(calendar, time.year)
        year_in_seconds = model_year_seconds(year_type)
        total_seconds = datetime_to_seconds_in_year(time)
        (n1, f1), (n2, f2) = get_periodic_interval(
            current_time=total_seconds,
            cycle_length=year_in_seconds,
            rec_spacing=year_in_seconds / 12.0,
            n_rec=12,
        )
        monthly_index_left.append(n1)
        monthly_index_right.append(n2)
        monthly_weight_left.append(f1)
        monthly_weight_right.append(f2)
        daily_index.append(
            daily_forcing_index(time, year_type=year_type, no_leap=True)
        )

    return RuntimeStepInfo.from_sequences(
        monthly_index_left,
        monthly_index_right,
        monthly_weight_left,
        monthly_weight_right,
        daily_index,
    )
```

Import `model_year_seconds` and `year_type_for_calendar` from
`vercor.calendar`; remove `_forcing_year_type_for_calendar`.

Simplify the three wrappers:

```python
def build_runtime_step_info(
    clock: Clock,
    *,
    clock_steps: Sequence[tuple[int, datetime | ModelDateTime, timedelta]] | None = None,
) -> RuntimeStepInfo:
    """Build scanned-runtime time metadata for every clock step."""

    steps = clock.iter() if clock_steps is None else clock_steps
    times = [time for _, time, _ in steps]
    return runtime_step_info_from_times(times, calendar=clock.calendar)


def initial_runtime_step_info(clock: Clock) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for the first clock step."""

    clock_iter = clock.iter()
    try:
        _, first_time, _ = next(clock_iter)
    except StopIteration:
        first_time = clock.start
    return scalar_runtime_step_info(first_time, clock)


def scalar_runtime_step_info(
    time: datetime | ModelDateTime,
    clock: Clock,
) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for one clock timestamp."""

    batched_step_info = runtime_step_info_from_times(
        [time],
        calendar=clock.calendar,
    )
    return cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(lambda value: value[0], batched_step_info),
    )
```

- [ ] **Step 5: Remove duration arguments from runtime callers**

In `vercor/_runtime/backends.py`, call:

```python
        step_infos=build_runtime_step_info(
            context.clock,
            clock_steps=clock_steps,
        ),
```

In `vercor/_runtime/preparation.py`, call:

```python
            step_info=initial_runtime_step_info(prepared.clock),
```

- [ ] **Step 6: Update end-to-end forcing regressions without synthetic years**

In the three January-first monthly tests in `tests/test_coupler_runtime.py`,
delete `runtime=RuntimeOptions(model_year_seconds=12.0)`. Their expected first
record and gradient behavior is unchanged because elapsed year time is zero.

In `test_monthly_forcing_wraps_year_boundary_under_jit_and_grad`, import the
calendar helpers and replace runtime ownership with:

```python
    year_type = year_type_for_calendar(
        coupler.clock.calendar,
        coupler.clock.start.year,
    )
    year_seconds = model_year_seconds(year_type)
    (left_index, left_weight), (right_index, right_weight) = get_periodic_interval(
        current_time=datetime_to_seconds_in_year(coupler.clock.start),
        cycle_length=year_seconds,
        rec_spacing=year_seconds / 12.0,
        n_rec=12,
    )
```

Update the daily forcing expected-index setup to call
`year_type_for_calendar(coupler.clock.calendar, runtime_time.year)` instead of
the deleted private clock helper.

- [ ] **Step 7: Correct all public ownership assertions**

Make these focused replacements:

- `tests/test_runtime_api_boundaries.py`: assert the attribute and
  constructor parameter are absent.
- `tests/test_plugin_architecture.py`: construct `RuntimeOptions` with topology
  only and assert the model-year attribute is absent.
- `tests/test_v0_4_public_api.py`: use `RuntimeOptions(backend="host")` as the
  read-only replacement value.
- `tests/test_final_review_boundaries.py`: remove imports used only by deleted
  numeric normalization parameterization.
- `tests/test_coupler_runtime.py`: remove the deleted private clock-helper import
  and any now-unused `RuntimeOptions` import only if no other test uses it.

Use this boundary shape wherever the assertion is repeated:

```python
    options = RuntimeOptions()
    assert not hasattr(options, "model_year_seconds")
    assert "model_year_seconds" not in inspect.signature(RuntimeOptions).parameters
```

- [ ] **Step 8: Verify runtime GREEN**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  tests/test_tools_time_and_forcing.py \
  tests/test_coupler_runtime.py \
  tests/test_runtime_api_boundaries.py \
  tests/test_v0_4_workflows.py \
  tests/test_plugin_architecture.py \
  tests/test_final_review_boundaries.py \
  tests/test_v0_4_public_api.py -q --fast
```

Expected: all selected tests pass for leap Gregorian, common Gregorian,
no-leap, and 360-day duration; RuntimeOptions has no duplicate owner.

- [ ] **Step 9: Update executable API evidence and architecture documentation**

Make these exact semantic updates:

- `tests/contracts/vercor-0.4.0a1-public-signatures.json`: replace the
  `vercor.runtime.RuntimeOptions` signature with
  `(dtype: vercor.dtypes.DTypePolicy = <factory>, backend: Union[Literal['auto', 'jax', 'host'], vercor.runtime.ExecutionBackend] = 'auto', workflow: vercor.runtime.Workflow = <factory>, topology: vercor.topology.TopologyPolicy | None = None) -> None`.
- `docs/api-architecture-review.md`: remove `model_year_seconds=31536000.0`
  from the readable signature and describe calendar-owned model-year policy.
- `DESIGN.md`: change configuration ownership so `RuntimeOptions` owns dtype,
  backend, workflow, and topology only; add calendar year type/duration to the
  clock/calendar ownership paragraph.
- Legacy migration guidance: state that callers delete
  `RuntimeOptions(model_year_seconds=...)` and select only `Clock.calendar`.
- `DEPENDENCIES.md`: describe `vercor.calendar` as the year-type/duration owner
  in layer 1 and `_runtime/time.py` as its per-timestamp consumer in layer 3.
- `PROGRESS.md`: prepend a dated current-status bullet recording the ownership
  fix, RED failures, focused GREEN count, fast/full counts, quality gates, and
  commit identifier using the exact outputs observed during execution.

- [ ] **Step 10: Prove no live duplicate owner remains**

Run:

```bash
rg -n "model_year_seconds|_forcing_year_type_for_calendar" \
  vercor tests examples DESIGN.md DEPENDENCIES.md \
  docs/api-architecture-review.md PROGRESS.md
```

Expected live matches are limited to the calendar helper, deliberate absence
assertions, migration/progress text, the approved design/plan, and immutable
historical evidence. There must be no `RuntimeOptions` field, runtime option
read, or private clock year-type mapper.

- [ ] **Step 11: Run all requested quality gates**

Run each command independently and retain its exact exit code and compact
summary:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 \
  vercor examples tests --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q \
  vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest \
  --cov=vercor --cov-branch --cov-report=term tests/ -q
git diff --check
```

Expected: every command exits 0, all tests pass, branch coverage remains at or
above 90%, and no whitespace errors are reported.

- [ ] **Step 12: Commit the runtime ownership correction**

```bash
git add vercor/_runtime/time.py vercor/_runtime/backends.py \
  vercor/_runtime/preparation.py vercor/runtime/__init__.py \
  tests/test_tools_time_and_forcing.py tests/test_coupler_runtime.py \
  tests/test_runtime_api_boundaries.py tests/test_v0_4_workflows.py \
  tests/test_plugin_architecture.py tests/test_final_review_boundaries.py \
  tests/test_v0_4_public_api.py \
  tests/contracts/vercor-0.4.0a1-public-signatures.json \
  DESIGN.md DEPENDENCIES.md docs/api-architecture-review.md \
  PROGRESS.md
git commit -m "refactor: derive model-year duration from calendar"
```

---

### Task 3: Independent review and final verification

**Files:**

- Inspect: all files changed since design commit `627c389`.
- Modify only if review finds a reproducible issue: the smallest affected test,
  implementation, or documentation file.

**Interfaces:**

- Consumes: both implementation commits and the approved design.
- Produces: reviewed, freshly verified repository state with no uncommitted
  changes.

- [ ] **Step 1: Request an independent code-quality review**

Use `superpowers:requesting-code-review` with:

```text
DESCRIPTION: Calendar-owned YearType and per-timestamp model-year duration;
RuntimeOptions model_year_seconds removed; forcing/runtime/tests/docs migrated.
PLAN_OR_REQUIREMENTS: docs/superpowers/specs/2026-07-15-calendar-year-ownership-design.md
and docs/superpowers/plans/2026-07-15-calendar-year-ownership.md
BASE_SHA: 627c389
HEAD_SHA: output of git rev-parse HEAD
```

The review must check calendar correctness across leap transitions, public API
ownership, string forcing compatibility, JAX-safe data flow, DRY/YAGNI/SOLID,
test coverage, documentation consistency, and absence of unrelated changes.

- [ ] **Step 2: Resolve every Critical or Important finding test-first**

For each valid finding, add the smallest failing regression, run it to confirm
RED, apply the minimal fix, rerun focused and full tests, and commit with:

```bash
git add -u
git add vercor tests
git commit -m "fix: address calendar ownership review"
```

If the reviewer reports no Critical or Important findings, make no review-only
code change or empty commit.

- [ ] **Step 3: Run fresh final verification after review**

Run:

```bash
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m black --check \
  vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m flake8 \
  vercor examples tests --count --max-line-length=120 --statistics
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m mypy vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m compileall -q \
  vercor examples tests
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast
/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q
git diff --check
git status --short
```

Expected: every quality command exits 0, fast and full suites have zero
failures, whitespace is clean, and `git status --short` emits no lines.

- [ ] **Step 4: Report evidence without publishing**

Report the implementation and any review-fix commit hashes, focused/fast/full
test counts, coverage percentage, Black/flake8/mypy/compileall status, reviewer
assessment, and clean worktree state. Do not tag, push, publish, upload, or open
a pull request without separate authorization.
