"""Native numeric output keeps payload values and host timestamp semantics."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.components import CallableComponent, ComponentSpec
import vercor.output as output_api
from vercor.output import OutputContext, OutputFrame
from vercor.state import ComponentState

pytestmark = pytest.mark.fast_always


def _numeric_method(provider: Any) -> Callable[..., OutputFrame]:
    method = getattr(provider, "sample_jax", None)
    assert callable(method), "native provider must expose pure sample_jax extraction"
    return cast(Callable[..., OutputFrame], method)


def _component(grid: Any | None = None) -> CallableComponent:
    grid = make_test_grid(name="numeric-native-output") if grid is None else grid
    return CallableComponent("model", grid, lambda value: {}, spec=ComponentSpec())


def _numeric_context(
    component: CallableComponent,
    payload: Any,
    fields: dict[str, Any] | None = None,
    *,
    step: Any = 1,
) -> output_api.JaxOutputContext:
    return output_api.JaxOutputContext(
        component=component,
        state=ComponentState(component.name, component.grid, fields=fields or {}),
        payload=payload,
        step=jnp.asarray(step, dtype=jnp.int32),
        dt_seconds=600.0,
    )


def _contexts(
    component: CallableComponent,
    payload: Any,
    fields: dict[str, Any] | None = None,
) -> tuple[output_api.JaxOutputContext, OutputContext]:
    numeric = _numeric_context(component, payload, fields)
    return (
        numeric,
        OutputContext(
            component=component,
            state=numeric.state,
            payload=payload,
            step=1,
            time=datetime(2000, 2, 29, 12),
            dt=timedelta(seconds=600),
        ),
    )


def _assert_host_frame_matches_numeric(
    numeric: OutputFrame,
    host: OutputFrame,
) -> None:
    assert "time" not in numeric.coordinates
    assert host.coordinates["time"].attrs["isoformat"] == "2000-02-29T12:00:00"
    assert tuple(numeric.variables) == tuple(host.variables)
    assert tuple(numeric.coordinates) == tuple(
        name for name in host.coordinates if name != "time"
    )
    assert numeric.metadata == host.metadata
    assert numeric.sample_dimension == host.sample_dimension
    assert numeric.time_dimension == host.time_dimension
    assert numeric.dimension_order == host.dimension_order
    for name, variable in numeric.variables.items():
        assert variable.dims == host.variables[name].dims
        assert variable.attrs == host.variables[name].attrs
        np.testing.assert_array_equal(variable.values, host.variables[name].values)
    for name, coordinate in numeric.coordinates.items():
        assert coordinate.dims == host.coordinates[name].dims
        assert coordinate.attrs == host.coordinates[name].attrs
        np.testing.assert_array_equal(coordinate.values, host.coordinates[name].values)


class _Physics:
    UNITS_TABLE_CSV_PATH = None

    def cache_coords(self, coords: Any) -> None:
        self.coords = coords

    def data_struct_to_dict(
        self, struct: Any, nodal_shape: tuple[int, ...]
    ) -> dict[str, Any]:
        assert nodal_shape == (3, 2, 3)
        return dict(struct)


def _jcm_provider() -> Any:
    from vercor.setups._external.jax_gcm_output import jax_gcm_output_provider

    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            nodal_axes=(
                jnp.deg2rad(jnp.asarray([0.0, 180.0])),
                jnp.sin(jnp.deg2rad(jnp.asarray([-45.0, 0.0, 45.0]))),
            ),
            modal_axes=(jnp.asarray([0]), jnp.asarray([0])),
            nodal_shape=(2, 3),
            modal_shape=(1, 1),
        ),
        vertical=SimpleNamespace(centers=jnp.asarray([0.2, 0.5, 0.8]), layers=3),
        surface_nodal_shape=(2, 3),
    )
    return jax_gcm_output_provider(
        SimpleNamespace(
            seaice_coupling=True,
            model=SimpleNamespace(coords=coords, physics=_Physics()),
        )
    )


def _jcm_payload(offset: Any) -> Any:
    return SimpleNamespace(
        jcm_state=SimpleNamespace(
            dynamics={"temperature": jnp.arange(18.0).reshape(3, 2, 3) + offset},
            physics={"shortwave": jnp.full((3, 2, 3), 7.0) + offset},
        )
    )


def _marine_fields(offset: Any) -> dict[str, Any]:
    return {
        "sea_surface_temperature": jnp.full((3, 2), 275.0) + offset,
        "marine_surface_albedo": jnp.full((3, 2), 0.3),
        "marine_ice_fraction": jnp.full((3, 2), 0.5),
    }


def test_jcm_numeric_extraction_traces_current_payload_and_preserves_metadata() -> None:
    provider = _jcm_provider()
    numeric_sample = _numeric_method(provider)
    component = _component(make_test_grid(name="numeric-jcm-output"))

    def extract(offset: Any) -> Any:
        context = _numeric_context(
            component, _jcm_payload(offset), _marine_fields(offset)
        )
        frame = numeric_sample(context)
        return (
            frame.variables["temperature"].values,
            frame.variables["sea_surface_temperature"].values,
        )

    abstract = jax.eval_shape(extract, jax.ShapeDtypeStruct((), jnp.float64))
    assert abstract[0].shape == (1, 3, 2, 3)
    compiled = jax.jit(extract)
    for offset in (0.0, 100.0):
        temperature, surface = compiled(jnp.asarray(offset))
        np.testing.assert_array_equal(
            temperature, jnp.arange(18.0).reshape(1, 3, 2, 3) + offset
        )
        np.testing.assert_array_equal(surface, np.full((1, 3, 2), 275.0 + offset))
    numeric_context, host_context = _contexts(
        component, _jcm_payload(2.0), _marine_fields(2.0)
    )
    numeric = numeric_sample(numeric_context)
    _assert_host_frame_matches_numeric(numeric, provider.sample(host_context))
    assert numeric.sample_dimension == "time"
    assert numeric.variables["temperature"].attrs["units"] == "K"
    np.testing.assert_allclose(numeric.coordinates["lat"].values, [-45.0, 0.0, 45.0])
    np.testing.assert_allclose(numeric.coordinates["lon"].values, [0.0, 180.0])


def _veros_payload(tau: Any, offset: Any = 0.0) -> Any:
    from vercor.setups._external.veros_output import veros_variables

    variables = SimpleNamespace(
        tau=tau,
        xt=jnp.asarray([-2.0, -1.0, 0.0, 4.0, 8.0, 99.0]),
        yt=jnp.asarray([-4.0, -2.0, -45.0, 0.0, 45.0, 88.0, 99.0]),
        zt=jnp.asarray([-100.0, -20.0]),
        temp=jnp.arange(6 * 7 * 2 * 3.0).reshape(6, 7, 2, 3) + offset,
    )
    return SimpleNamespace(
        settings=SimpleNamespace(coord_degree=True),
        variables=variables,
        var_meta={
            name: deepcopy(veros_variables.VARIABLES[name])
            for name in ("temp", "xt", "yt", "zt")
        },
    )


def test_veros_numeric_extraction_selects_traced_tau_and_latest_native_values() -> None:
    from vercor.setups._external.veros_output import veros_output_provider

    provider = veros_output_provider()
    numeric_sample = _numeric_method(provider)
    component = _component(make_test_grid(name="numeric-veros-output"))

    def extract(tau: Any, offset: Any) -> Any:
        context = _numeric_context(component, _veros_payload(tau, offset), step=tau)
        frame = numeric_sample(context)
        return frame.variables["temp"].values

    abstract = jax.eval_shape(
        extract,
        jax.ShapeDtypeStruct((), jnp.int32),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    assert abstract.shape == (2, 3, 2)
    compiled = jax.jit(extract)
    native = np.arange(6 * 7 * 2 * 3.0).reshape(6, 7, 2, 3)
    for tau, offset in ((0, 0.0), (2, 100.0), (1, -10.0)):
        actual = compiled(jnp.asarray(tau, dtype=jnp.int32), jnp.asarray(offset))
        np.testing.assert_array_equal(actual, native[2:-2, 2:-2, :, tau].T + offset)
    numeric_context, host_context = _contexts(component, _veros_payload(jnp.asarray(1)))
    numeric = numeric_sample(numeric_context)
    _assert_host_frame_matches_numeric(numeric, provider.sample(host_context))
    assert tuple(numeric.variables) == ("temp",)
    assert numeric.variables["temp"].dims == ("zt", "yt", "xt")
    assert numeric.variables["temp"].attrs["units"] == "deg C"
    np.testing.assert_array_equal(numeric.coordinates["xt"].values, [0.0, 4.0])


def _veris_payload(offset: Any) -> Any:
    from veris.model.diagnostics import DIAGNOSTICS

    native = jnp.arange(42.0).reshape(7, 6)
    return SimpleNamespace(
        state=SimpleNamespace(
            Area=jnp.full_like(native, 0.5),
            hIceMean=jnp.full_like(native, 0.25),
            hSnowMean=native + offset,
            TSurf=260.0 + native,
            uIce=100.0 + native + offset,
            vIce=200.0 + native,
        ),
        diagnostics=SimpleNamespace(
            **{name: 300.0 + native + offset for name in DIAGNOSTICS}
        ),
    )


def _ice_fields(offset: Any) -> dict[str, Any]:
    return {
        name: jnp.arange(6.0).reshape(2, 3) + value + offset
        for name, value in (
            ("sea_surface_heat_flux", 400.0),
            ("sea_surface_mass_flux", 500.0),
            ("sea_surface_salt_mass_flux", 600.0),
            ("ocean_u_stress", 700.0),
            ("ocean_v_stress", 800.0),
            ("ice_surface_temperature", 260.0),
            ("ice_surface_albedo", 0.5),
        )
    }


def test_veris_numeric_extraction_keeps_halos_faces_and_canonical_fields() -> None:
    from vercor.grids import RectilinearGrid
    from vercor.setups._external.veris_output import veris_output_provider

    provider = veris_output_provider()
    numeric_sample = _numeric_method(provider)
    grid = RectilinearGrid(
        "ice-output",
        longitude=jnp.asarray([0.0, 10.0, 20.0]),
        latitude=jnp.asarray([-5.0, 5.0]),
        binary_mask=jnp.ones((2, 3)),
    )
    component = _component(grid)

    def extract(offset: Any) -> Any:
        context = _numeric_context(
            component, _veris_payload(offset), _ice_fields(offset)
        )
        frame = numeric_sample(context)
        return (
            frame.variables["hSnowMean"].values,
            frame.variables["ocean_u_stress"].values,
        )

    abstract = jax.eval_shape(extract, jax.ShapeDtypeStruct((), jnp.float64))
    assert abstract[0].shape == (2, 3)
    compiled = jax.jit(extract)
    native = np.arange(42.0).reshape(7, 6)
    for offset in (0.0, 2.0):
        snow, stress = compiled(jnp.asarray(offset))
        np.testing.assert_array_equal(snow, native[2:-2, 2:-2].T + offset)
        np.testing.assert_array_equal(
            stress, np.arange(6.0).reshape(2, 3) + 700 + offset
        )
    numeric_context, host_context = _contexts(
        component, _veris_payload(3.0), _ice_fields(3.0)
    )
    numeric = numeric_sample(numeric_context)
    _assert_host_frame_matches_numeric(numeric, provider.sample(host_context))
    assert numeric.variables["uIce"].dims == ("y_center", "x_face")
    assert numeric.variables["vIce"].dims == ("y_face", "x_center")
    assert numeric.variables["ocean_u_stress"].dims == ("y_center", "x_east_face")
    assert (
        numeric.variables["ice_surface_albedo"].attrs["cell_methods"] == "time: point"
    )
    np.testing.assert_array_equal(
        numeric.coordinates["x_face"].values, [-5.0, 5.0, 15.0]
    )
    np.testing.assert_array_equal(
        numeric.coordinates["x_east_face"].values, [5.0, 15.0, 25.0]
    )


@pytest.mark.parametrize("native", ["jcm", "veros", "veris"])
def test_numeric_native_provider_requires_its_runtime_payload(native: str) -> None:
    if native == "jcm":
        provider = _jcm_provider()
    elif native == "veros":
        from vercor.setups._external.veros_output import veros_output_provider

        provider = veros_output_provider()
    else:
        from vercor.setups._external.veris_output import veris_output_provider

        provider = veris_output_provider()
    numeric_sample = _numeric_method(provider)
    context, _ = _contexts(_component(), None)
    with pytest.raises(ValueError, match="payload|JCM state"):
        numeric_sample(context)
