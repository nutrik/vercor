"""Check measured native-step counts, independent budgets and segmented carry."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_veris_component import small_coupler
from tests.test_veris_rollouts import native_payload
from vercor import Clock, Coupler
from vercor.setups import replace_veris_parameters

pytestmark = pytest.mark.fast_always


def test_validation_report_tracks_all_native_steps(tmp_path: Path) -> None:
    from benchmarks.validate_veris import validate_coupler

    coupler, _, _, _ = small_coupler(steps=3)
    initial = coupler.initial_state()
    final, report = validate_coupler(coupler, initial, output=tmp_path)
    assert report["checked_native_steps"] == 6
    assert report["expected_native_steps"] == 6
    assert report["native_configuration"]["deltatTherm"] == 300.0
    assert report["all_native_steps_valid"] is True
    assert int(native_payload(initial).checked_native_steps) == 0
    assert int(native_payload(final).checked_native_steps) == 6
    assert max(report["budget_residuals"].values()) <= 1e-10


@pytest.mark.parametrize("factor", [0.0, 0.5, 1000.0, 600.0, 4.0 / 3.0])
def test_validation_rejects_lost_or_duplicated_flux(factor: float) -> None:
    from benchmarks.validate_veris import (
        Inventory,
        IntegratedFluxes,
        budget_residuals,
        require_budgets,
    )

    latent = 334000.0
    first = Inventory(
        ice_volume=1.0, snow_volume=0.0, mass=100.0, salt=0.5, enthalpy=-100 * latent
    )
    last = Inventory(
        ice_volume=1.02, snow_volume=0.0, mass=102.0, salt=0.51, enthalpy=-102 * latent
    )
    flux = IntegratedFluxes(
        ocean_mass=-1.0,
        ocean_salt=-0.01,
        ocean_heat=1000 + 2 * latent,
        external_mass=1.0,
        external_heat=1000.0,
    )
    assert max(budget_residuals(first, last, flux).values()) < 1e-14
    corrupted = replace(flux, ocean_mass=flux.ocean_mass * factor)
    with pytest.raises(ValueError, match="budget"):
        require_budgets(budget_residuals(first, last, corrupted))


def test_validation_rejects_mismatched_area_convention() -> None:
    from benchmarks.validate_veris import (
        Inventory,
        IntegratedFluxes,
        budget_residuals,
        require_budgets,
    )

    first = Inventory(1.0, 0.0, 100.0, 0.5, -33400000.0)
    last = Inventory(1.02, 0.0, 102.0, 0.51, -34068000.0)
    flux = IntegratedFluxes(-1.0002, -0.01, 669000.0, 1.0, 1000.0)
    with pytest.raises(ValueError, match="budget"):
        require_budgets(budget_residuals(first, last, flux))


def test_segmented_validation_preserves_parameters_and_flux_carry(
    tmp_path: Path,
) -> None:
    from benchmarks.validate_veris import validate_coupler, validate_segmented

    full, component, _, _ = small_coupler(steps=3)
    initial = replace_veris_parameters(full.initial_state(), {"h0": jnp.asarray(0.43)})
    uninterrupted, report = validate_coupler(full, initial, output=tmp_path / "full")
    resumed = validate_segmented(
        full, initial, uninterrupted, segment_steps=1, output=tmp_path / "resumed"
    )
    assert resumed["maximum_scaled_state_difference"] <= 1e-12
    assert resumed["checked_native_steps"] == 6
    current = initial
    for start, steps in ((0, 1), (1, 2)):
        clock = Clock(full.clock.start + timedelta(seconds=600 * start), 600.0, steps)
        segment = Coupler(
            clock,
            components=tuple(full.components.values()),
            exchanges=full.exchanges,
            run_order=full.run_order,
            runtime=full.runtime,
            constants=full.constants,
            log_level="error",
        )
        current, partial = validate_coupler(
            segment, current, output=tmp_path / f"segment-{start}"
        )
        assert partial["checked_native_steps"] == 2 * steps
    for left, right in zip(
        jax.tree_util.tree_leaves(native_payload(uninterrupted)),
        jax.tree_util.tree_leaves(native_payload(current)),
        strict=True,
    ):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    assert float(native_payload(current).physical_constants.h0) == 0.43
    assert report["checked_native_steps"] == int(
        native_payload(current).checked_native_steps
    )


@pytest.mark.parametrize("failure", ["status", "hidden_nan"])
def test_validation_rejects_intermediate_native_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str, tmp_path: Path
) -> None:
    from tests.test_veris_feedback import _controlled_transition
    from vercor.setups._external.veris_runtime import step_veris_runtime
    from vercor.exceptions import CouplerError

    initial, resources, context = _controlled_transition(monkeypatch, failure=failure)
    with pytest.raises(
        (CouplerError, jax.errors.JaxRuntimeError), match="Veris.*valid"
    ):
        result = step_veris_runtime(resources, {}, context, initial)
        jax.block_until_ready(result.payload)


def test_refinement_rejects_unresolved_difference() -> None:
    from benchmarks.validate_veris import timestep_refinement

    reports: list[dict[str, Any]] = [
        {
            "comparison_vector": {"ice_volume": x},
            "checked_native_steps": 100,
            "dtype": "float64",
        }
        for x in (1.1, 1.05, 1.025)
    ]
    assert timestep_refinement(reports)["resolved"] is True
    reports[-1]["comparison_vector"]["ice_volume"] = 1.11
    assert timestep_refinement(reports)["resolved"] is False


def test_native_counter_measures_executed_transitions() -> None:
    coupler, _, _, _ = small_coupler(steps=2)
    initial = coupler.initial_state()
    assert int(native_payload(initial).checked_native_steps) == 0
    final = coupler.run(initial)
    assert int(native_payload(final).checked_native_steps) == 4
    assert int(native_payload(initial).checked_native_steps) == 0


def test_segment_clock_preserves_noleap_forcing_dates() -> None:
    from datetime import datetime
    from benchmarks.validate_veris import segment_coupler

    full, _, _, _ = small_coupler(steps=3)
    annual = Coupler(
        Clock(datetime(2000, 1, 1), 86400.0, 365, calendar="noleap"),
        components=tuple(full.components.values()),
        exchanges=full.exchanges,
        run_order=full.run_order,
        runtime=full.runtime,
        constants=full.constants,
    )
    segment = segment_coupler(annual, start=59, steps=30)
    assert segment.clock.start == datetime(2000, 3, 1)
    assert segment.clock.calendar == "noleap"


def test_extended_initial_ice_exercises_nonzero_salt() -> None:
    from benchmarks.validate_veris import polar_config

    config = polar_config(
        np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([-65.0, 45.0]), model_seconds=1800
    )
    assert config.physical_constants is not None
    assert config.initial_conditions is not None
    assert config.physical_constants["saltIce_ref"] == 4.0
    np.testing.assert_array_equal(config.initial_conditions["Area"], [[0.5, 0], [0, 0]])
    np.testing.assert_array_equal(
        config.initial_conditions["hIceMean"], [[0.25, 0], [0, 0]]
    )


@pytest.mark.parametrize("field", ["hIceMean", "hSnowMean", "Area"])
def test_validation_rejects_solid_storage_on_dry_cells(field: str) -> None:
    from vercor.setups._external.veris_runtime import _valid_native_payload

    coupler, _, _, _ = small_coupler(steps=1)
    payload = native_payload(coupler.initial_state())
    assert float(payload.state.iceMask[2, 2]) == 0
    array = getattr(payload.state, field).at[2, 2].set(0.1)
    corrupted = replace(payload, state=replace(payload.state, **{field: array}))
    assert not bool(_valid_native_payload(corrupted))


def test_segmented_comparison_preserves_missing_data_sentinels() -> None:
    from benchmarks.validate_veris import state_difference

    left = {"field": np.array([1.0, np.nan]), "count": np.array(3, dtype=np.int32)}
    assert state_difference(left, left) == 0
    with pytest.raises(ValueError, match="missing"):
        state_difference(left, {**left, "field": np.array([1.0, 0.0])})
    with pytest.raises(ValueError, match="infinite"):
        state_difference(left, {**left, "field": np.array([1.0, np.inf])})


def test_benchmark_authors_explicit_static_momentum_precision() -> None:
    from benchmarks.validate_veris import polar_config

    config = polar_config(
        np.ones((2, 2)), np.array([-65.0, 65.0]), model_seconds=900, evp_steps=1600
    )
    assert config.custom_parameters is not None
    assert config.custom_parameters["nEVPsteps"] == 1600


def test_benchmark_adaptive_momentum_mode_is_explicit() -> None:
    from benchmarks.validate_veris import polar_config

    config = polar_config(
        np.ones((2, 2)), np.array([-65.0, 65.0]), model_seconds=900, adaptive_evp=True
    )
    assert config.custom_parameters is not None
    assert config.custom_parameters["useAdaptiveEVP"] is True


def test_benchmark_cleanup_threshold_is_an_explicit_parameter() -> None:
    from benchmarks.validate_veris import polar_config

    config = polar_config(
        np.ones((2, 2)),
        np.array([-65.0, 65.0]),
        model_seconds=900,
        minimum_ice_thickness=1e-9,
    )
    assert config.physical_constants is not None
    assert config.physical_constants["hIce_min"] == 1e-9
