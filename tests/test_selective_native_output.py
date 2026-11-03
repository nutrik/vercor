"""Bundled native output samples only selected arrays and includes full state."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from tests.test_external_components_coverage import _make_veros_output_state
from tests.test_veris_component import small_coupler
from tests.test_veris_rollouts import native_payload
from vercor.output import JaxOutputContext, OutputContext
from vercor.state import ComponentState

pytestmark = pytest.mark.fast_always


def test_veris_default_includes_initialized_registry_and_canonical_inputs() -> None:
    from veris.model.variables import VARIABLES
    from veris.model.diagnostics import DIAGNOSTICS
    from vercor.setups._external.veris_output import veris_output_provider

    coupler, component, _, _ = small_coupler(steps=0)
    initial = coupler.initial_state()
    payload = native_payload(initial)
    provider = veris_output_provider(
        component.spec.lifecycle.setup.__self__.surface_state_fields
    )
    frame = provider.sample_jax(
        JaxOutputContext(component, initial.component("CICE"), payload, 0, 600)
    )
    assert set(VARIABLES) <= set(frame.variables)
    assert set(DIAGNOSTICS) <= set(frame.variables)
    assert set(initial.component("CICE").fields()) <= set(frame.variables)
    assert frame.variables["sigma12"].dims == ("y_face", "x_face")
    np.testing.assert_array_equal(
        frame.variables["SeaIceMassC"].values, payload.state.SeaIceMassC[2:-2, 2:-2].T
    )


@pytest.mark.parametrize("numeric", [False, True])
def test_veris_selection_skips_endpoint_diagnosis_and_unselected_fields(
    numeric: bool,
) -> None:
    from vercor.setups._external.veris_output import veris_output_provider

    coupler, component, _, _ = small_coupler(steps=0)
    initial = coupler.initial_state()
    payload = native_payload(initial)

    def forbidden(payload: Any) -> Any:
        raise AssertionError("unselected endpoint diagnosis executed")

    provider = veris_output_provider(forbidden)
    if numeric:
        frame = provider.sample_jax(
            JaxOutputContext(
                component,
                initial.component("CICE"),
                payload,
                0,
                600,
                variables=("hIceMean",),
            )
        )
    else:
        frame = provider.sample(
            OutputContext(
                component,
                initial.component("CICE"),
                payload,
                0,
                datetime(2000, 1, 1),
                timedelta(seconds=600),
                variables=("hIceMean",),
            )
        )
    assert tuple(frame.variables) == ("hIceMean",)


class _PoisonArray:
    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unselected native array converted")


def _veros_context(state: Any, variables: tuple[str, ...] = ()) -> JaxOutputContext:
    grid = make_test_grid(name="native-select")
    return JaxOutputContext(
        cast(Any, None),
        ComponentState(
            "OCN", grid, fields={"canonical_flux": 2 * jnp.ones(grid.shape)}
        ),
        state,
        0,
        600,
        variables=variables,
    )


def test_veros_selection_does_not_read_unselected_native_arrays() -> None:
    from vercor.setups._external.veros_output import veros_output_provider

    state = _make_veros_output_state()
    state.variables.salt = _PoisonArray()
    frame = veros_output_provider().sample_jax(_veros_context(state, ("temp",)))
    assert tuple(frame.variables) == ("temp",)


def test_veros_default_includes_canonical_fields_and_implicit_native_axes() -> None:
    from vercor.setups._external.veros_output import veros_output_provider

    state = _make_veros_output_state()
    state.variables.matrix = np.arange(4.0).reshape(2, 2)
    state.var_meta["matrix"] = SimpleNamespace(
        active=True,
        dims=("island", "island"),
        units="1",
        long_description="native matrix",
        extra_attributes={},
    )
    frame = veros_output_provider().sample_jax(_veros_context(state))
    assert "canonical_flux" in frame.variables
    assert "matrix" in frame.variables
    matrix = frame.variables["matrix"]
    assert len(set(matrix.dims)) == 2
    for dim in matrix.dims:
        np.testing.assert_array_equal(frame.coordinates[dim].values, [0, 1])
    np.testing.assert_array_equal(matrix.values, state.variables.matrix.T)


def test_jcm_selection_precedes_array_conversion_and_time_expansion() -> None:
    from tests.test_external_components_coverage import (
        _make_jax_gcm_output_coords,
        _FakePhysicsModule,
    )
    from vercor.setups._external.jax_gcm_output import jax_gcm_output_provider

    provider = jax_gcm_output_provider(
        SimpleNamespace(
            seaice_coupling=False,
            model=SimpleNamespace(
                coords=_make_jax_gcm_output_coords(), physics=_FakePhysicsModule()
            ),
        )
    )
    grid = make_test_grid(name="jcm-select")
    context = JaxOutputContext(
        cast(Any, None),
        ComponentState("ATM", grid),
        SimpleNamespace(
            jcm_state=SimpleNamespace(
                dynamics={
                    "temperature": np.ones((3, 2, 3)),
                    "unselected": _PoisonArray(),
                },
                physics={},
            )
        ),
        0,
        600,
        variables=("temperature",),
    )
    frame = provider.sample_jax(context)
    assert tuple(frame.variables) == ("temperature",)
    np.testing.assert_array_equal(frame.variables["temperature"].values, 1.0)


def test_jcm_default_includes_canonical_fluxes_missing_from_native_state() -> None:
    from tests.test_external_components_coverage import (
        _make_jax_gcm_output_coords,
        _FakePhysicsModule,
    )
    from vercor.setups._external.jax_gcm_output import jax_gcm_output_provider

    provider = jax_gcm_output_provider(
        SimpleNamespace(
            seaice_coupling=False,
            model=SimpleNamespace(
                coords=_make_jax_gcm_output_coords(), physics=_FakePhysicsModule()
            ),
        )
    )
    grid = make_test_grid(name="jcm-union")
    context = JaxOutputContext(
        cast(Any, None),
        ComponentState("ATM", grid, fields={"rain_rate": jnp.ones(grid.shape)}),
        SimpleNamespace(
            jcm_state=SimpleNamespace(
                dynamics={"temperature": np.ones((3, 2, 3))}, physics={}
            )
        ),
        0,
        600,
    )
    frame = provider.sample_jax(context)
    assert frame.variables["rain_rate"].dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(frame.variables["rain_rate"].values, 1.0)


def test_jcm_unselected_physics_is_not_reshaped_by_native_formatter() -> None:
    from jcm.physics.speedy.speedy_terms import speedy_physics
    from tests.test_external_components_coverage import _make_jax_gcm_output_coords
    from vercor.setups._external.jax_gcm_output import jax_gcm_output_provider

    class UnselectedArray(np.ndarray):
        def reshape(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("unselected physics field reshaped")

    native_physics = speedy_physics()
    physics = SimpleNamespace(
        cache_coords=lambda coords: None,
        data_struct_to_dict=native_physics.data_struct_to_dict,
        UNITS_TABLE_CSV_PATH=None,
    )
    provider = jax_gcm_output_provider(
        SimpleNamespace(
            seaice_coupling=False,
            model=SimpleNamespace(
                coords=_make_jax_gcm_output_coords(), physics=physics
            ),
        )
    )
    context = JaxOutputContext(
        cast(Any, None),
        ComponentState("ATM", make_test_grid()),
        SimpleNamespace(
            jcm_state=SimpleNamespace(
                dynamics={"temperature": np.ones((3, 2, 3))},
                physics={
                    "unselected": np.ones(6).view(UnselectedArray),
                    "_typed": SimpleNamespace(extra=np.ones((3, 2, 3, 2))),
                },
            )
        ),
        0,
        600,
        variables=("temperature",),
    )
    frame = provider.sample_jax(context)
    assert tuple(frame.variables) == ("temperature",)


def test_veros_default_includes_initialized_native_diagnostic_registry() -> None:
    from vercor.setups._external.veros_output import veros_output_provider

    state = _make_veros_output_state()
    state.diagnostics = {
        "energy": SimpleNamespace(
            variables=SimpleNamespace(kinetic_energy=np.asarray(17.0)),
            var_meta={
                "kinetic_energy": SimpleNamespace(
                    active=True,
                    dims=None,
                    units="J",
                    long_description="total kinetic energy",
                    extra_attributes={},
                )
            },
        )
    }
    provider = veros_output_provider()
    frame = provider.sample_jax(_veros_context(state))
    assert frame.variables["energy.kinetic_energy"].attrs["units"] == "J"
    np.testing.assert_array_equal(frame.variables["energy.kinetic_energy"].values, 17.0)
    state.variables.salt = _PoisonArray()
    selected = provider.sample_jax(_veros_context(state, ("energy.kinetic_energy",)))
    assert tuple(selected.variables) == ("energy.kinetic_energy",)


def test_veros_repeated_grid_axes_preserve_physical_coordinates() -> None:
    from vercor.setups._external.veros_output import veros_output_provider

    state = _make_veros_output_state()
    state.variables.matrix = np.ones((6, 6))
    state.var_meta["matrix"] = SimpleNamespace(
        active=True,
        dims=("xt", "xt"),
        units="1",
        long_description="native matrix",
        extra_attributes={},
    )
    frame = veros_output_provider().sample_jax(_veros_context(state, ("matrix",)))
    for dim in frame.variables["matrix"].dims:
        np.testing.assert_array_equal(frame.coordinates[dim].values, [0.0, 4.0])


def test_veris_temperature_selection_skips_albedo_diagnosis() -> None:
    from vercor.setups._external.veris_output import veris_output_provider

    coupler, component, _, _ = small_coupler(steps=0)
    initial = coupler.initial_state()
    payload = native_payload(initial)

    def forbidden(payload: Any) -> Any:
        raise AssertionError("unselected albedo diagnosed")

    frame = veris_output_provider(forbidden).sample_jax(
        JaxOutputContext(
            component,
            initial.component("CICE"),
            payload,
            0,
            600,
            variables=("ice_surface_temperature",),
        )
    )
    assert tuple(frame.variables) == ("ice_surface_temperature",)
    np.testing.assert_array_equal(
        frame.variables["ice_surface_temperature"].values,
        payload.state.TSurf[2:-2, 2:-2].T,
    )
