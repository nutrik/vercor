"""Exchange reduction has its own samples and selection on every runtime lane."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5netcdf
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._workflow_test_support import SequentialBackend, make_clock
from tests.assertions import assert_allclose_compact
from vercor import Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import CallableComponent, ComponentSpec, StepContext
from vercor.output import OutputSpec, OutputTarget, PeriodOutput

pytestmark = pytest.mark.fast_always


def test_jcm_maps_into_scan_carry_without_a_mapped_trajectory() -> None:
    from tests.test_jcm_temporal_exchange import _case
    from vercor.dtypes import DTypePolicy
    from vercor.physics import PhysicalConstants
    from vercor.setups._external.jax_gcm_runtime import _mean_jcm_sample_fields

    owner, payload, _ = _case([272.77, 278.15])
    _, samples = owner._step_function(payload.jcm_state, payload.forcing)

    def reduce(history: Any) -> Any:
        return _mean_jcm_sample_fields(
            owner, history, PhysicalConstants(), DTypePolicy()
        )

    result = jax.jit(reduce)(samples)
    assert_allclose_compact(result["air_temperature"], 290)
    assert_allclose_compact(result["snow_rate"], 0.0061920292202211746)
    assert_allclose_compact(result["air_pressure_3d"][:, 0, 0], [19000, 90250])
    jaxpr = jax.make_jaxpr(reduce)(samples)

    def find_scans(graph: Any) -> list[Any]:
        graph = getattr(graph, "jaxpr", graph)
        scans = []
        for equation in graph.eqns:
            if equation.primitive.name == "scan":
                scans.append(equation)
            for parameter in equation.params.values():
                if hasattr(parameter, "jaxpr") or hasattr(parameter, "eqns"):
                    scans.extend(find_scans(parameter))
        return scans

    scans = find_scans(jaxpr)
    assert len(scans) == 1
    carry_shapes = [variable.aval.shape for variable in scans[0].outvars]
    expected_shapes = [leaf.shape for leaf in jax.tree.leaves(result)] + [()]
    assert carry_shapes == expected_shapes, "mapped history retained"
    assert scans[0].params["length"] == 1


@pytest.mark.parametrize("lane", ["host", "jax", "mixed", "custom"])
def test_exchange_samples_and_output_variables_are_independent(
    lane: str, tmp_path: Path
) -> None:
    from vercor._averaging import SampleMean

    grid = RectilinearGrid(
        "shared", longitude=jnp.arange(2.0), latitude=jnp.arange(2.0)
    )

    def step(fields: Mapping[str, Any], context: StepContext) -> Mapping[str, Any]:
        # Each coupling interval contains two native endpoint samples. The
        # endpoint is 3,6,9; the exchange mean is 2,5,8. Output samples only the
        # coupling endpoints of the unrelated diagnostic, averaging 103/106.
        first = fields["endpoint"] + 1
        endpoint = fields["endpoint"] + 3
        mean = SampleMean.from_sample(first).add(endpoint).mean()
        return {"exchange": mean, "diagnostic": 100 + endpoint, "endpoint": endpoint}

    def sink(fields: Mapping[str, Any]) -> Mapping[str, Any]:
        return {"total": fields["total"] + fields["exchange"]}

    for selection in (None, ("diagnostic",), ("exchange",), ()):
        source = CallableComponent(
            "SOURCE",
            grid,
            step,
            spec=ComponentSpec(
                outputs=("exchange", "diagnostic", "endpoint"),
                initial_fields={"exchange": 0.0, "diagnostic": 100.0, "endpoint": 0.0},
                execution="host" if lane == "mixed" else "jax",
                output=OutputSpec(
                    period=PeriodOutput("day", variables=selection or ())
                ),
            ),
        )
        target = CallableComponent(
            "TARGET",
            grid,
            sink,
            spec=ComponentSpec(
                inputs=("exchange",), outputs=("total",), initial_fields={"total": 0.0}
            ),
        )
        backend: Any = (
            SequentialBackend()
            if lane == "custom"
            else ("auto" if lane == "mixed" else lane)
        )
        coupler = Coupler(
            make_clock(3, dt_seconds=43200),
            components=(source, target),
            exchanges=(
                Exchange("SOURCE", "TARGET", ("exchange",), quantity="intensive"),
            ),
            run_order=("SOURCE", "TARGET"),
            runtime=RuntimeOptions(backend=backend, topology=None),
            log_level="ERROR",
        )
        directory = tmp_path / str(selection)
        result = coupler.run(
            output=None if selection is None else OutputTarget(directory)
        )
        assert_allclose_compact(result.component("TARGET").field("total"), 15)
        assert_allclose_compact(result.component("SOURCE").field("endpoint"), 9)
        assert_allclose_compact(result.component("SOURCE").field("exchange"), 8)
        if selection is not None:
            paths = tuple(directory.glob("source.averages.*.nc"))
            assert len(paths) == 1
            with h5netcdf.File(paths[0]) as dataset:
                if not selection or "diagnostic" in selection:
                    np.testing.assert_allclose(
                        dataset.variables["diagnostic"][:], 104.5
                    )
                if not selection or "exchange" in selection:
                    np.testing.assert_allclose(dataset.variables["exchange"][:], 3.5)
                assert ("diagnostic" in dataset.variables) == (
                    not selection or "diagnostic" in selection
                )


def test_camulator_output_requests_do_not_change_any_exchange_field() -> None:
    from dataclasses import replace
    from tests.test_camulator_temporal_fluxes import _native_window_case
    from vercor.setups._external.camulator_runtime import step_camulator_runtime

    resources, fields, context, payload = _native_window_case(3, 3600)
    baseline = step_camulator_runtime(resources, fields, context, payload)
    for selection in ((), ("TS",), ("sea_surface_temperature",)):
        result = step_camulator_runtime(
            resources, fields, replace(context, output_variables=selection), payload
        )
        assert tuple(result.fields) == tuple(baseline.fields)
        for name, value in result.fields.items():
            np.testing.assert_array_equal(value, baseline.fields[name])
        assert result.payload.forecast_hour == 4
        assert result.payload.cursor.current_index() == 3
