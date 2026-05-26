from __future__ import annotations

from collections.abc import Mapping, Sequence

from vercor.diagnostics.fields import ComponentMetric, safe_component_metric_mean
from vercor.runtime.views import RuntimeComponentView


def print_component_field_means_table(
    components: Mapping[str, RuntimeComponentView],
    fields: Sequence[tuple[ComponentMetric, str]],
    component_order: Sequence[str] | None = None,
) -> None:
    """Print a means table for component fields with configurable column order."""

    ordered_names = list(component_order or components.keys())
    ordered_names = [name for name in ordered_names if name in components]

    first_col_width = max(10, max((len(label) for _, label in fields), default=10))
    value_col_width = 15
    header = f"{'Variable':<{first_col_width}} " + " ".join(
        f"{name:>{value_col_width}}" for name in ordered_names
    )
    print(header)
    print("-" * len(header))

    for field_name, label in fields:
        values = [
            safe_component_metric_mean(components[name], field_name)
            for name in ordered_names
        ]
        value_text = " ".join(f"{value:>{value_col_width}.4f}" for value in values)
        print(f"{label:<{first_col_width}} {value_text}")
