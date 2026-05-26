from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import cast

import jax

from vercor.calendar import ModelDateTime, daily_forcing_index
from vercor.clock import Clock
from vercor.dtypes import as_jax_index_array, as_jax_real_array
from vercor.pytree import PyTreeNodeMixin
from vercor.settings import VercorSettings
from vercor.time_selection import (
    datetime_to_seconds_in_year,
    get_periodic_interval,
)
from vercor.types import RuntimeArray


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RuntimeStepInfo(PyTreeNodeMixin):
    """Precomputed time-selection metadata for one runtime step."""

    pytree_children = (
        "monthly_index_left",
        "monthly_index_right",
        "monthly_weight_left",
        "monthly_weight_right",
        "daily_index",
    )

    monthly_index_left: RuntimeArray
    monthly_index_right: RuntimeArray
    monthly_weight_left: RuntimeArray
    monthly_weight_right: RuntimeArray
    daily_index: RuntimeArray

    @classmethod
    def from_sequences(
        cls,
        monthly_index_left: Sequence[int],
        monthly_index_right: Sequence[int],
        monthly_weight_left: Sequence[float],
        monthly_weight_right: Sequence[float],
        daily_index: Sequence[int],
    ) -> "RuntimeStepInfo":
        """Create scan metadata from host-precomputed index and weight arrays."""

        return cls(
            monthly_index_left=as_jax_index_array(monthly_index_left),
            monthly_index_right=as_jax_index_array(monthly_index_right),
            monthly_weight_left=as_jax_real_array(monthly_weight_left),
            monthly_weight_right=as_jax_real_array(monthly_weight_right),
            daily_index=as_jax_index_array(daily_index),
        )


def runtime_step_info_from_times(
    times: Sequence[datetime | ModelDateTime],
    *,
    year_type: str,
    year_in_seconds: float,
) -> RuntimeStepInfo:
    """Build runtime time-selection metadata for one or more timestamps."""

    monthly_index_left: list[int] = []
    monthly_index_right: list[int] = []
    monthly_weight_left: list[float] = []
    monthly_weight_right: list[float] = []
    daily_index: list[int] = []

    for time in times:
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
        daily_index.append(daily_forcing_index(time, year_type=year_type, no_leap=True))

    return RuntimeStepInfo.from_sequences(
        monthly_index_left,
        monthly_index_right,
        monthly_weight_left,
        monthly_weight_right,
        daily_index,
    )


def build_runtime_step_info(clock: Clock, settings: VercorSettings) -> RuntimeStepInfo:
    """Build scanned-runtime time metadata for every clock step."""

    times = [time for _, time, _ in clock.iter()]
    return runtime_step_info_from_times(
        times,
        year_type=clock.year_type,
        year_in_seconds=settings.year_in_seconds,
    )


def initial_runtime_step_info(
    clock: Clock, settings: VercorSettings
) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for the first clock step."""

    clock_iter = clock.iter()
    try:
        _, first_time, _ = next(clock_iter)
    except StopIteration:
        first_time = clock.start
    return scalar_runtime_step_info(first_time, clock, settings)


def scalar_runtime_step_info(
    time: datetime | ModelDateTime,
    clock: Clock,
    settings: VercorSettings,
) -> RuntimeStepInfo:
    """Return scalar runtime time metadata for one clock timestamp."""

    batched_step_info = runtime_step_info_from_times(
        [time],
        year_type=clock.year_type,
        year_in_seconds=settings.year_in_seconds,
    )
    return cast(
        RuntimeStepInfo,
        jax.tree_util.tree_map(lambda value: value[0], batched_step_info),
    )
