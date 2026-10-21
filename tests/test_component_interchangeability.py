"""Physical producer substitution through public component and exchange contracts.

Native conversion/species/face tests remain in their dedicated suites. These
synthetic producers establish exchange/receiver parity, not native model skill.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._exchange_case_data import (
    QUANTITY_CASES,
    COVERAGE_GRADIENT_CASES,
    FIELD_GRADIENT_CASES,
)
from tests.test_component_models_coverage import _install_data_driver_factory_fakes
from tests.test_dynamic_mask_exchanges import _MeanRegridder
from vercor import Clock, Coupler, Exchange, RectilinearGrid, RuntimeOptions
from vercor.components import (
    CallableComponent,
    Component,
    ComponentMaskSpec,
    ComponentSpec,
    DataComponent,
    StepContext,
)
from vercor.dtypes import DTypePolicy
from vercor.exceptions import CouplerError
from vercor.fields import vector
from vercor.fluxes import compute_ocean_surface_fluxes
from vercor.physics import PhysicalConstants
from vercor.setups import (
    make_era5_atmosphere,
    make_era5_land,
    make_era5_ocean,
    make_erainterim_ocean,
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
)
from vercor.setups._external.veros_fluxes import _compute_virtual_salt_flux
from vercor.setups._surface_fluxes import compute_surface_fluxes

pytestmark = pytest.mark.fast_always
KINDS = ("data", "callable", "structural", "host")
AIR = {
    "air_temperature": 280.0,
    "air_potential_temperature": 280.0,
    "air_specific_humidity": 0.002,
    "air_density": 1.2,
    "model_level_height": 10.0,
    "air_u_velocity": 5.0,
    "air_v_velocity": 0.0,
    "net_shortwave_radiation_flux": 100.0,
    "downward_longwave_radiation_flux": 300.0,
    "rain_rate": 1e-3,
    "snow_rate": 2e-4,
}


@dataclass(frozen=True)
class _StructuralProducer:
    name: str
    grid: RectilinearGrid
    spec: ComponentSpec

    def step(
        self, fields: Mapping[str, Any], context: StepContext, payload: Any = None
    ) -> Mapping[str, Any]:
        # A real structural step: every export is read through the runtime store.
        return {name: fields[name] for name in self.spec.outputs}


def _producer(
    kind: str, grid: RectilinearGrid, fields: Mapping[str, Any], *, masked: bool = False
) -> Component:
    spec = ComponentSpec(
        outputs=tuple(fields),
        initial_fields=fields,
        masks=ComponentMaskSpec(fractional_field="area") if masked else None,
        execution="host" if kind == "host" else "jax",
    )
    if kind == "data":
        return DataComponent("SRC", grid, fields, spec=spec)
    if kind == "structural":
        return _StructuralProducer("SRC", grid, spec)

    def step(values: Mapping[str, Any]) -> Mapping[str, Any]:
        if kind == "host":
            # Deliberately host-only; an accidental tracing bridge must fail.
            return {name: np.asarray(values[name]).copy() for name in fields}
        return {name: values[name] for name in fields}

    return CallableComponent("SRC", grid, step, spec=spec)


def _grid(name: str, *, target: bool = False, wide: bool = False) -> RectilinearGrid:
    return RectilinearGrid(
        name,
        longitude=(
            jnp.array([0.25, 0.75])
            if target
            else jnp.arange(4.0) if wide else jnp.array([0.0, 1.0])
        ),
        latitude=jnp.array([-45.0, 45.0]),
    )


def _bulk(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    constants, dtype = PhysicalConstants(), DTypePolicy(enable_x64=True)
    one = jnp.ones_like(fields["air_temperature"])
    bulk = compute_surface_fluxes(
        fields,
        sea_surface_temperature=290 * one,
        u_ocean=0 * one,
        v_ocean=0 * one,
        ocean_mask=one,
        constants=constants,
        dtype=dtype,
    )
    sensible, latent, *_ = compute_ocean_surface_fluxes(
        constants,
        one,
        fields["model_level_height"],
        fields["air_u_velocity"],
        fields["air_v_velocity"],
        fields["air_potential_temperature"],
        fields["air_specific_humidity"],
        fields["air_density"],
        fields["air_temperature"],
        0 * one,
        0 * one,
        290 * one,
    )
    return {
        "sensible": sensible,
        "latent": latent,
        "heat": bulk.qnet,
        "evap": bulk.evap,
        "stress": bulk.taux,
        "salt": _compute_virtual_salt_flux(
            constants, fields["rain_rate"], fields["snow_rate"], bulk.evap, one
        ),
    }


def _bulk_coupler(kind: str, *, backend: Any = "auto") -> Coupler:
    grid = _grid("air")
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(
            _producer(kind, grid, AIR),
            CallableComponent(
                "DST",
                grid,
                _bulk,
                spec=ComponentSpec(
                    inputs=tuple(AIR),
                    outputs=("sensible", "latent", "heat", "evap", "stress", "salt"),
                ),
            ),
        ),
        exchanges=(Exchange("SRC", "DST", tuple(AIR), quantity="intensive"),),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True), backend=backend),
        log_level="error",
    )


@pytest.mark.parametrize("kind", KINDS)
def test_swapping_producer_preserves_bulk_flux_units_and_signs(kind: str) -> None:
    actual = _bulk_coupler(kind).run().component("DST")
    reference = _bulk_coupler("data").run().component("DST")
    for name in (*AIR, "sensible", "latent", "heat", "evap", "stress", "salt"):
        np.testing.assert_allclose(
            actual.field(name), reference.field(name), rtol=1e-12
        )
    assert np.all(actual.field("sensible") < 0)
    assert np.all(actual.field("latent") < 0)
    expected_salt = -34.7 * (1e-3 + 2e-4 + actual.field("evap")) / 1000
    np.testing.assert_allclose(actual.field("salt"), expected_salt, rtol=1e-12)
    assert np.all(expected_salt < 0)


def _coastal_coupler(
    kind: str, quantity: str, *, binary: bool = False, wide: bool = False
) -> Coupler:
    area = jnp.tile(jnp.array([0.25, 0.75]), (2, 2 if wide else 1))
    flux = jnp.tile(jnp.array([2.0, 10.0]), (2, 2 if wide else 1))
    source = _producer(
        kind,
        _grid("source", wide=wide),
        {"area": area, "flux": flux, "u": flux, "v": -2 * flux},
        masked=True,
    )
    # A surface heat capacity of 2 J/(m2 K) makes the receiver tendency
    # an independently known linear physical response to the received W/m2.
    target = CallableComponent(
        "DST",
        _grid("target", target=True),
        lambda f: {"heat_tendency": f["flux"] / 2.0},
        spec=ComponentSpec(
            inputs=("flux", "u", "v"),
            outputs=("area", "heat_tendency"),
            initial_fields={"area": 0.25},
            masks=ComponentMaskSpec(fractional_field="area"),
        ),
    )
    return Coupler(
        Clock(datetime(2000, 1, 1), 60.0, 1),
        components=(source, target),
        exchanges=(
            Exchange(
                "SRC",
                "DST",
                ("flux", vector("u", "v")),
                quantity=cast(Any, quantity),
                source_coverage="binary" if binary else "component",
                regridder_factory=_MeanRegridder,
            ),
        ),
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(dtype=DTypePolicy(enable_x64=True)),
        log_level="error",
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("quantity,expected", QUANTITY_CASES)
def test_swapped_components_preserve_coastal_products_and_received_support(
    kind: str,
    quantity: str,
    expected: float,
) -> None:
    coupler = _coastal_coupler(kind, quantity, wide=True)
    state = coupler.run()
    view = state.component("DST")
    np.testing.assert_allclose(view.field("flux"), expected)
    np.testing.assert_allclose(view.field("u"), 2 * expected)
    np.testing.assert_allclose(view.field("v"), expected)
    np.testing.assert_allclose(view.field("heat_tendency"), expected / 2)
    # Imports retain the dispatch support even when current source support changes.
    zero = jnp.zeros((2, 4))
    inactive = state.replace_fields(
        "SRC",
        {
            "area": zero,
            "flux": zero + jnp.nan,
            "u": zero + jnp.nan,
            "v": zero + jnp.nan,
        },
    )
    np.testing.assert_allclose(
        inactive.component("DST").field("flux", scope="received"), expected
    )
    empty = coupler.run(inactive)
    np.testing.assert_array_equal(empty.component("DST").field("flux"), 0.0)
    active = empty.replace_fields(
        "SRC",
        {
            "area": jnp.tile(jnp.array([[1.0, 0.0], [1.0, 0.0]]), (1, 2)),
            "flux": jnp.tile(jnp.array([[2.0, jnp.nan], [2.0, jnp.nan]]), (1, 2)),
            "u": jnp.tile(jnp.array([[2.0, jnp.nan], [2.0, jnp.nan]]), (1, 2)),
            "v": jnp.tile(jnp.array([[-4.0, jnp.nan], [-4.0, jnp.nan]]), (1, 2)),
        },
    )
    final = coupler.run(active).component("DST")
    np.testing.assert_allclose(
        final.field("flux"),
        {"intensive": 2.0, "source_flux": 1.0, "target_flux": 0.5}[quantity],
    )
    np.testing.assert_allclose(final.field("u"), 2 * final.field("flux"))


@pytest.mark.parametrize("kind", KINDS)
def test_preweighted_veris_export_receives_one_area_factor(kind: str) -> None:
    coupler = _coastal_coupler(kind, "source_flux", binary=True)
    state = coupler.initial_state()
    area = jnp.tile(jnp.array([0.25, 0.75]), (2, 1))
    preweighted = area * jnp.tile(jnp.array([2.0, 10.0]), (2, 1))
    state = state.replace_fields("SRC", {"flux": preweighted})
    np.testing.assert_allclose(coupler.run(state).component("DST").field("flux"), 4.0)


@pytest.mark.parametrize("kind", KINDS[:-1])
@pytest.mark.parametrize(
    "quantity,area_gradient,field_gradient",
    [
        (q, area, field)
        for (q, area, _), (_, field) in zip(
            COVERAGE_GRADIENT_CASES, FIELD_GRADIENT_CASES, strict=True
        )
    ],
)
def test_swapped_jax_producers_preserve_analytic_continuous_derivatives(
    kind: str,
    quantity: str,
    area_gradient: list[float],
    field_gradient: list[float],
) -> None:
    coupler = _coastal_coupler(kind, quantity)
    initial = coupler.initial_state()
    area = jnp.tile(jnp.array([0.25, 0.75]), (2, 1))
    flux = jnp.tile(jnp.array([2.0, 10.0]), (2, 1))

    def objective(a: jax.Array, f: jax.Array) -> jax.Array:
        state = initial.replace_fields("SRC", {"area": a, "flux": f})
        return jnp.mean(coupler.run(state).component("DST").field("heat_tendency"))

    compiled = jax.jit(objective)
    gradients = jax.grad(compiled, argnums=(0, 1))(area, flux)
    np.testing.assert_allclose(
        gradients[0], 0.5 * np.tile(area_gradient, (2, 1)), atol=1e-12
    )
    np.testing.assert_allclose(
        gradients[1], 0.5 * np.tile(field_gradient, (2, 1)), atol=1e-12
    )
    direction = (jnp.ones_like(area), jnp.ones_like(flux))
    _, jvp = jax.jvp(compiled, (area, flux), direction)
    _, pullback = jax.vjp(compiled, area, flux)
    np.testing.assert_allclose(
        jvp, sum(jnp.sum(x) for x in pullback(jnp.array(1.0))), atol=1e-12
    )
    h = 1e-5
    fd = (compiled(area + h, flux + h) - compiled(area - h, flux - h)) / (2 * h)
    np.testing.assert_allclose(jvp, fd, rtol=1e-8, atol=1e-8)


def test_bundled_prescribed_factories_declare_no_physical_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_data_driver_factory_fakes(monkeypatch)
    for factory in (
        make_era5_atmosphere,
        make_era5_land,
        make_era5_ocean,
        make_erainterim_ocean,
    ):
        assert factory().spec.inputs == (), factory.__name__


@pytest.mark.parametrize(
    "factory,expected",
    [
        (make_slab_atmosphere, ("sea_surface_temperature",)),
        (make_slab_land, ("latent_heat_flux",)),
        (make_slab_ocean, ("sensible_heat_flux", "latent_heat_flux")),
    ],
)
def test_slab_declarations_match_physics_reads(
    factory: Any, expected: tuple[str, ...]
) -> None:
    assert factory(_grid("slab")).spec.inputs == expected


def test_missing_producer_field_fails_before_receiving_physics() -> None:
    grid = _grid("missing")
    with pytest.raises(CouplerError, match="air_specific_humidity"):
        Coupler(
            Clock(datetime(2000, 1, 1), 60.0, 1),
            components=(
                _producer(
                    "data",
                    grid,
                    {k: v for k, v in AIR.items() if k != "air_specific_humidity"},
                ),
                CallableComponent(
                    "DST", grid, _bulk, spec=ComponentSpec(inputs=tuple(AIR))
                ),
            ),
            exchanges=(Exchange("SRC", "DST", tuple(AIR), quantity="intensive"),),
            run_order=("SRC", "DST"),
        ).run()


def test_camulator_land_is_prescribed(monkeypatch: pytest.MonkeyPatch) -> None:
    from vercor.setups._external import camulator_land

    monkeypatch.setattr(
        camulator_land,
        "load_camulator_forcing_context",
        lambda **_: {
            "conf": {"data": {"lead_time_periods": 6}},
            "forcing_dataset_raw": None,
        },
    )
    monkeypatch.setattr(
        camulator_land,
        "create_land_mask_from_ocean",
        lambda **_: (jnp.ones((2, 2)), jnp.ones((2, 2))),
    )
    assert (
        camulator_land.make_camulator_land(
            "unused", _grid("air"), _grid("ocean")
        ).spec.inputs
        == ()
    )


def test_host_producer_rejects_differentiable_backend() -> None:
    from vercor.exceptions import ComponentError

    coupler = _bulk_coupler("host", backend="jax")
    with pytest.raises(ComponentError, match="backend='jax'.*host-backed"):
        coupler.run()


def test_data_gallery_displays_explicit_diagnostic_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every gallery-only import reaches a table metric or its surface plot."""
    from types import SimpleNamespace
    from vercor.setups.gallery import run_data_driver as gallery

    grid = _grid("air")
    ocean_grid = RectilinearGrid(
        "ocean",
        longitude=grid.longitude,
        latitude=grid.latitude,
        binary_mask=jnp.array([[1.0, 0.0], [1.0, 0.0]]),
    )
    assert ocean_grid.binary_mask is not None
    land_grid = RectilinearGrid(
        "land",
        longitude=grid.longitude,
        latitude=grid.latitude,
        binary_mask=1 - ocean_grid.binary_mask,
    )
    monkeypatch.setattr(
        gallery, "make_era5_atmosphere", lambda: DataComponent("ATM", grid, AIR)
    )
    monkeypatch.setattr(
        gallery,
        "make_erainterim_ocean",
        lambda: DataComponent("OCN", ocean_grid, {"sea_surface_temperature": 290.0}),
    )
    monkeypatch.setattr(
        gallery,
        "make_era5_land",
        lambda: DataComponent("LND", land_grid, {"land_surface_temperature": 285.0}),
    )
    captured: dict[str, Any] = {}

    class _RunWithoutOutput:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.coupler = Coupler(*args, **kwargs)
            captured["components"] = kwargs["components"]

        def run(self, **kwargs: Any) -> Any:
            return self.coupler.run()

    def table(*, components: Any, fields: Any, component_order: Any) -> None:
        captured["table"] = (components, fields, component_order)

    def plot(**kwargs: Any) -> Any:
        captured["rows"] = kwargs["rows"]
        return SimpleNamespace(colorbar=lambda *a, **k: None), None, None

    monkeypatch.setattr(gallery, "Coupler", _RunWithoutOutput)
    monkeypatch.setattr(gallery, "print_component_field_means_table", table)
    monkeypatch.setattr(gallery, "plot_component_scalar_vector_comparison", plot)
    monkeypatch.setattr(gallery.plt, "show", lambda: None)
    gallery.run_setup(loglevel="error", float_type="float64")
    views, fields, order = captured["table"]
    shown = {name for name, _ in fields if isinstance(name, str)}
    shown.update(
        ("air_u_velocity", "air_v_velocity")
    )  # component_vector_speed + quiver
    shown.add("land_surface_temperature")  # total_surface_temperature plot
    for component in captured["components"]:
        assert component.name in order
        assert set(component.spec.inputs) <= shown
        for name in component.spec.inputs:
            assert np.all(
                np.isfinite(views[component.name].field(name, scope="received"))
            )
    np.testing.assert_allclose(
        views["OCN"].field("air_specific_humidity", scope="received"),
        [[0.002, 0.0], [0.002, 0.0]],
    )
    np.testing.assert_allclose(views["OCN"].field("sea_surface_temperature"), 290.0)
    assert captured["rows"][0][2] is gallery.total_surface_temperature


@pytest.mark.parametrize("face", ["maskU", "maskV", "area_u", "area_v"])
def test_unsupported_native_face_geometry_fails_during_preparation(
    face: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import replace
    from tests.test_veris_component import small_ocean
    from vercor.setups import make_veris
    import veris

    grid, geometry = small_ocean()
    values = np.asarray(getattr(geometry, face))
    malformed = replace(geometry, **{face: values[:-1]})
    advances: list[bool] = []

    def advance(*args: Any, **kwargs: Any) -> Any:
        advances.append(True)
        raise AssertionError("native advance must not run with malformed faces")

    monkeypatch.setattr(veris, "advance", advance)
    component = make_veris(grid, malformed)
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), 3600.0, 1),
        components=(component,),
        run_order=(component.name,),
        log_level="error",
    )
    with pytest.raises(ValueError, match=face):
        coupler.initial_state()
    assert advances == []
