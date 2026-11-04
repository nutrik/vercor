"""VerCOR 0.4 route identity, topology, regridding, and state contracts."""

from __future__ import annotations

from datetime import datetime
from functools import partial
import inspect
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import pytest

from tests._coverage_support import make_test_grid
from vercor import Clock, Coupler
from vercor.components import ComponentSpec, DataComponent, LifecycleHooks, SetupResult
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.fields import vector
import vercor.regridding as regridding
from vercor.runtime import RuntimeOptions
from vercor.state import RunState
import vercor.state as state_module
from vercor.topology import ExchangeTopologyPatch, TopologyContext
from vercor._runtime.stores import FieldStore


def _clock(*, steps: int = 1) -> Clock:
    return Clock(start=datetime(2000, 1, 1), dt_seconds=3600.0, steps=steps)


def _components(*, setup_calls: list[str] | None = None) -> tuple[Any, Any]:
    grid = make_test_grid(name="route-grid")

    def setup(component: Any, context: Any) -> SetupResult:
        _ = context
        if setup_calls is not None:
            setup_calls.append(component.name)
        return SetupResult()

    lifecycle = LifecycleHooks(setup=setup) if setup_calls is not None else None
    source = DataComponent(
        "SRC",
        grid,
        {"scalar": 1.0, "other": 4.0, "u": 2.0, "v": 3.0},
        spec=ComponentSpec(
            outputs=("scalar", "other", "u", "v"),
            lifecycle=lifecycle,
        ),
    )
    target = DataComponent(
        "DST",
        grid,
        {"scalar": 0.0, "other": 0.0, "u": 0.0, "v": 0.0},
        spec=ComponentSpec(
            inputs=("scalar", "other", "u", "v"),
            lifecycle=lifecycle,
        ),
    )
    return source, target


def _coupler(
    *exchanges: Exchange,
    setup_calls: list[str] | None = None,
    topology: Any | None = None,
) -> Coupler:
    source, target = _components(setup_calls=setup_calls)
    return Coupler(
        _clock(),
        components=(source, target),
        exchanges=exchanges,
        run_order=("SRC", "DST"),
        runtime=RuntimeOptions(topology=topology),
    )


@pytest.mark.fast_always
def test_exchange_owns_stable_route_id_and_regridder_factory() -> None:
    signature = inspect.signature(Exchange)
    assert "route_id" in signature.parameters
    assert "regridder_factory" in signature.parameters
    assert "regrid" not in signature.parameters
    assert "label" not in signature.parameters

    default = Exchange("SRC", "DST", ("scalar",), quantity="intensive")
    explicit = Exchange(
        "SRC",
        "DST",
        ("scalar",),
        quantity="intensive",
        route_id="surface-temperature",
        regridder_factory=regridding.bilinear,
    )

    assert default.route_id == "SRC->DST"
    assert default.regridder_factory is regridding.bilinear
    assert explicit.route_id == "surface-temperature"
    assert explicit.regridder_factory is regridding.bilinear
    assert not hasattr(default, "regrid")
    assert not hasattr(default, "label")


@pytest.mark.fast_always
@pytest.mark.parametrize("route_id", ("", "   ", 1, object()))
def test_exchange_rejects_invalid_route_ids(route_id: object) -> None:
    error = TypeError if not isinstance(route_id, str) else ValueError
    with pytest.raises(error, match="route_id"):
        Exchange("SRC", "DST", ("scalar",), quantity="intensive", route_id=route_id)  # type: ignore[arg-type]


@pytest.mark.fast_always
def test_exchange_rejects_noncallable_factory_immediately() -> None:
    with pytest.raises(TypeError, match="regridder_factory.*callable"):
        Exchange(
            "SRC",
            "DST",
            ("scalar",),
            quantity="intensive",
            regridder_factory=object(),  # type: ignore[arg-type]
        )


@pytest.mark.fast_always
def test_route_id_collisions_fail_before_lifecycle_or_factory_calls() -> None:
    setup_calls: list[str] = []
    factory_calls: list[tuple[Any, Any]] = []

    def factory(source_grid: Any, target_grid: Any) -> Any:
        factory_calls.append((source_grid, target_grid))
        return regridding.bilinear(source_grid, target_grid)

    first = Exchange(
        "SRC", "DST", ("scalar",), quantity="intensive", regridder_factory=factory
    )
    second = Exchange(
        "SRC", "DST", ("other",), quantity="intensive", regridder_factory=factory
    )

    with pytest.raises(CouplerError, match=r"route ID 'SRC->DST'.*unique"):
        _coupler(first, second, setup_calls=setup_calls)

    assert setup_calls == []
    assert factory_calls == []


@pytest.mark.fast_always
def test_explicit_route_ids_are_globally_unique_across_endpoints() -> None:
    source, target = _components()
    third = DataComponent(
        "THIRD",
        source.grid,
        {"scalar": 0.0},
        spec=ComponentSpec(inputs=("scalar",)),
    )
    routes = (
        Exchange("SRC", "DST", ("scalar",), quantity="intensive", route_id="duplicate"),
        Exchange(
            "SRC", "THIRD", ("scalar",), quantity="intensive", route_id="duplicate"
        ),
    )
    with pytest.raises(CouplerError, match=r"route ID 'duplicate'.*unique"):
        Coupler(
            _clock(),
            components=(source, target, third),
            exchanges=routes,
            run_order=("SRC", "DST", "THIRD"),
        )


@pytest.mark.fast_always
def test_parameterized_factories_build_independent_route_entries() -> None:
    calls: list[tuple[str, Any, Any, float]] = []

    def factory(
        source_grid: Any,
        target_grid: Any,
        *,
        token: str,
        fill_value: float,
    ) -> Any:
        calls.append((token, source_grid, target_grid, fill_value))
        return regridding.bilinear(
            source_grid,
            target_grid,
            fill_value=fill_value,
        )

    routes = (
        Exchange(
            "SRC",
            "DST",
            ("scalar",),
            quantity="intensive",
            route_id="scalar-route",
            regridder_factory=partial(
                factory,
                token="scalar",
                fill_value=-1.0,
            ),
        ),
        Exchange(
            "SRC",
            "DST",
            ("other",),
            quantity="intensive",
            route_id="other-route",
            regridder_factory=partial(
                factory,
                token="other",
                fill_value=-2.0,
            ),
        ),
    )
    coupler = _coupler(*routes)
    coupler.initial_state()

    assert [call[0] for call in calls] == ["scalar", "other"]
    assert [call[3] for call in calls] == [-1.0, -2.0]
    for _, source_grid, target_grid, _ in calls:
        assert source_grid.name == coupler.components["SRC"].grid.name
        assert target_grid.name == coupler.components["DST"].grid.name
        assert jnp.array_equal(
            source_grid.longitude,
            coupler.components["SRC"].grid.longitude,
        )
        assert jnp.array_equal(
            target_grid.latitude,
            coupler.components["DST"].grid.latitude,
        )
    prepared_regridders = coupler._ensure_prepared().topology_maps.regridders
    assert tuple(prepared_regridders) == ("scalar-route", "other-route")
    assert prepared_regridders["scalar-route"] is not prepared_regridders["other-route"]


@pytest.mark.fast_always
def test_scalar_and_vector_regridder_capabilities_are_independent() -> None:
    scalar_protocol = regridding.Regridder
    vector_protocol = getattr(regridding, "VectorRegridder")
    source, target = _components()
    scalar = regridding.conservative(source.grid, target.grid)
    dual = regridding.bilinear(source.grid, target.grid)

    assert isinstance(scalar, scalar_protocol)
    assert not isinstance(scalar, vector_protocol)
    assert not hasattr(scalar, "regrid_vector")
    assert isinstance(dual, scalar_protocol)
    assert isinstance(dual, vector_protocol)


@pytest.mark.fast_always
def test_vector_route_rejects_scalar_only_regridder_during_preparation() -> None:
    route = Exchange(
        "SRC",
        "DST",
        (vector("u", "v"),),
        quantity="intensive",
        route_id="wind",
        regridder_factory=regridding.conservative,
    )
    coupler = _coupler(route)
    with pytest.raises(CouplerError, match=r"wind.*VectorRegridder"):
        coupler.initial_state()


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "fields",
    (
        ("scalar",),
        ("scalar", vector("u", "v")),
    ),
    ids=("scalar", "mixed"),
)
def test_scalar_declarations_reject_vector_only_regridder_during_preparation(
    fields: tuple[Any, ...],
) -> None:
    class VectorOnly:
        def __init__(self, source_grid: Any, target_grid: Any) -> None:
            self.source_grid = source_grid
            self.target_grid = target_grid
            self.has_identical_grids = source_grid is target_grid

        def regrid_vector(self, u: Any, v: Any) -> tuple[Any, Any]:
            return u, v

    route = Exchange(
        "SRC",
        "DST",
        fields,
        quantity="intensive",
        route_id="scalar-capability",
        regridder_factory=cast(Any, VectorOnly),
    )
    with pytest.raises(CouplerError, match=r"scalar-capability.*Regridder"):
        _coupler(route).initial_state()


@pytest.mark.fast_always
def test_mixed_route_requires_both_scalar_and_vector_capabilities() -> None:
    route = Exchange(
        "SRC",
        "DST",
        ("scalar", vector("u", "v")),
        quantity="intensive",
        route_id="mixed",
        regridder_factory=regridding.bilinear,
    )
    final = _coupler(route).run()
    assert final.component("DST").field("scalar") is not None
    assert final.component("DST").field("u") is not None


@pytest.mark.fast_always
def test_topology_policy_requires_only_build_and_uses_route_string_keys() -> None:
    contexts: list[TopologyContext] = []

    class BuildOnlyPolicy:
        def build(self, context: TopologyContext) -> ExchangeTopologyPatch:
            contexts.append(context)
            return ExchangeTopologyPatch(
                binary_masks={"temperature-route": jnp.zeros((2, 2))}
            )

    route = Exchange(
        "SRC",
        "DST",
        ("scalar",),
        quantity="intensive",
        route_id="temperature-route",
    )
    final = _coupler(route, topology=BuildOnlyPolicy()).run()

    assert len(contexts) == 1
    assert tuple(exchange.route_id for exchange in contexts[0].exchanges) == (
        "temperature-route",
    )
    assert jnp.allclose(final.component("DST").field("scalar"), 0.0)


@pytest.mark.fast_always
def test_topology_patch_mappings_are_route_keyed_and_frozen() -> None:
    patch = ExchangeTopologyPatch(
        binary_masks={"route": jnp.ones((2, 2))},
    )
    assert isinstance(patch.binary_masks, MappingProxyType)
    assert tuple(patch.binary_masks) == ("route",)


@pytest.mark.fast_always
def test_topology_policy_must_return_exchange_topology_patch() -> None:
    class InvalidPolicy:
        def build(self, context: TopologyContext) -> None:
            _ = context
            return None

    route = Exchange(
        "SRC", "DST", ("scalar",), quantity="intensive", route_id="scalar-route"
    )
    with pytest.raises(CouplerError, match="must return ExchangeTopologyPatch"):
        _coupler(route, topology=InvalidPolicy()).initial_state()


@pytest.mark.fast_always
def test_fan_in_error_reports_sorted_route_ids_not_factory_names() -> None:
    routes = (
        Exchange("SRC", "DST", ("scalar",), quantity="intensive", route_id="z-route"),
        Exchange("SRC", "DST", ("scalar",), quantity="intensive", route_id="a-route"),
    )
    with pytest.raises(
        CouplerError,
        match=r"target 'DST'.*field 'scalar'.*'a-route'.*'z-route'",
    ):
        _coupler(*routes)


@pytest.mark.fast_always
def test_run_state_exposes_only_opaque_component_access_and_replacement() -> None:
    state = _coupler().initial_state()
    public_names = {name for name in dir(state) if not name.startswith("_")}

    assert public_names == {"component", "components", "replace_fields"}
    assert tuple(state.components()) == ("SRC", "DST")
    assert not hasattr(state, "component_names")
    assert not hasattr(state, "component_grids")
    assert not hasattr(state, "component_indices")
    assert tuple(state_module.__all__) == (
        "ComponentState",
        "FieldLookupScope",
        "FieldScope",
        "RunState",
    )


@pytest.mark.fast_always
def test_coupler_run_rejects_non_run_state_at_public_boundary() -> None:
    with pytest.raises(CouplerError, match="state must be a RunState"):
        _state_coupler().run(cast(Any, object()))


@pytest.mark.fast_always
def test_opaque_run_state_remains_a_jax_pytree() -> None:
    state = _coupler().initial_state()
    leaves, tree = jax.tree.flatten(state)
    rebuilt = jax.tree.unflatten(tree, leaves)

    assert tuple(rebuilt.components()) == ("SRC", "DST")
    assert str(jax.tree.structure(rebuilt)) == str(jax.tree.structure(state))


def _rebuild_state(
    state: RunState,
    *,
    components: tuple[Any, ...] | None = None,
    grids: tuple[Any, ...] | None = None,
    fractional_masks: FieldStore | None = None,
) -> RunState:
    return RunState._from_runtime(
        component_names=state._component_names,
        components=state._components if components is None else components,
        component_grids=state._component_grids if grids is None else grids,
        fractional_masks=(
            state._fractional_masks if fractional_masks is None else fractional_masks
        ),
    )


def _state_with_wrong_longitude(state: RunState) -> RunState:
    grid = state._component_grids[state._component_index("SRC")]
    assert grid is not None
    changed = type(grid)(
        name=grid.name,
        longitude=grid.longitude + 0.25,
        latitude=grid.latitude,
        longitude_edges=grid.longitude_edges,
        latitude_edges=grid.latitude_edges,
        binary_mask=grid.binary_mask,
    )
    grids = list(state._component_grids)
    grids[state._component_index("SRC")] = changed
    return _rebuild_state(state, grids=tuple(grids))


def _state_with_wrong_field_dtype(state: RunState) -> RunState:
    index = state._component_index("SRC")
    component = state._components[index]
    fields = component.fields
    values = list(fields.values)
    values[fields.field_indices["scalar"]] = jnp.asarray(
        fields.get("scalar"),
        dtype=jnp.int32,
    )
    changed_component = component.with_fields(
        FieldStore(field_names=fields.field_names, values=tuple(values))
    )
    components = list(state._components)
    components[index] = changed_component
    return _rebuild_state(state, components=tuple(components))


def _state_with_out_of_range_mask(state: RunState) -> RunState:
    masks = state._fractional_masks
    values = list(masks.values)
    values[0] = jnp.full(jnp.asarray(values[0]).shape, 1.5, dtype=jnp.float32)
    return _rebuild_state(
        state,
        fractional_masks=FieldStore(
            field_names=masks.field_names,
            values=tuple(values),
        ),
    )


def _state_with_wrong_payload(state: RunState) -> RunState:
    index = state._component_index("SRC")
    components = list(state._components)
    components[index] = components[index].with_payload({"unexpected": jnp.asarray(1)})
    return _rebuild_state(state, components=tuple(components))


def _state_with_wrong_component_order(state: RunState) -> RunState:
    return RunState._from_runtime(
        component_names=tuple(reversed(state._component_names)),
        component_grids=tuple(reversed(state._component_grids)),
        components=tuple(reversed(state._components)),
        fractional_masks=state._fractional_masks,
    )


def _state_with_duplicate_component_name(state: RunState) -> RunState:
    return RunState._from_runtime(
        component_names=("SRC", "SRC"),
        component_grids=state._component_grids,
        components=state._components,
        fractional_masks=state._fractional_masks,
    )


def _replace_source_grid(state: RunState, changed: Any) -> RunState:
    grids = list(state._component_grids)
    grids[state._component_index("SRC")] = changed
    return _rebuild_state(state, grids=tuple(grids))


def _state_with_wrong_grid_name(state: RunState) -> RunState:
    grid = state._component_grids[state._component_index("SRC")]
    assert grid is not None
    changed = type(grid)(
        name="foreign-grid",
        longitude=grid.longitude,
        latitude=grid.latitude,
        longitude_edges=grid.longitude_edges,
        latitude_edges=grid.latitude_edges,
        binary_mask=grid.binary_mask,
    )
    return _replace_source_grid(state, changed)


def _state_with_wrong_coordinate_dtype(state: RunState) -> RunState:
    grid = state._component_grids[state._component_index("SRC")]
    assert grid is not None
    changed = type(grid)(
        name=grid.name,
        longitude=grid.longitude,
        latitude=grid.latitude,
        longitude_edges=grid.longitude_edges,
        latitude_edges=grid.latitude_edges,
        binary_mask=grid.binary_mask,
    )
    object.__setattr__(
        changed, "longitude", jnp.asarray(grid.longitude, dtype=jnp.int32)
    )
    return _replace_source_grid(state, changed)


def _state_with_unexpected_grid_edges(state: RunState) -> RunState:
    grid = state._component_grids[state._component_index("SRC")]
    assert grid is not None
    changed = type(grid)(
        name=grid.name,
        longitude=grid.longitude,
        latitude=grid.latitude,
        longitude_edges=jnp.asarray([-0.5, 0.5, 1.5]),
        latitude_edges=grid.latitude_edges,
        binary_mask=grid.binary_mask,
    )
    return _replace_source_grid(state, changed)


def _state_with_unexpected_binary_mask(state: RunState) -> RunState:
    grid = state._component_grids[state._component_index("SRC")]
    assert grid is not None
    changed = type(grid)(
        name=grid.name,
        longitude=grid.longitude,
        latitude=grid.latitude,
        longitude_edges=grid.longitude_edges,
        latitude_edges=grid.latitude_edges,
        binary_mask=jnp.ones(grid.shape),
    )
    return _replace_source_grid(state, changed)


def _replace_component_store(
    state: RunState,
    component_name: str,
    store_name: str,
    store: FieldStore,
) -> RunState:
    component = state._component_state(component_name)
    method = getattr(component, f"with_{store_name}")
    return state._with_component_state(component_name, method(store))


def _state_with_reordered_fields(state: RunState) -> RunState:
    store = state._component_state("SRC").fields
    return _replace_component_store(
        state,
        "SRC",
        "fields",
        FieldStore(
            field_names=tuple(reversed(store.field_names)),
            values=tuple(reversed(store.values)),
        ),
    )


def _state_with_wrong_received_shape(state: RunState) -> RunState:
    store = state._component_state("DST").received
    values = list(store.values)
    values[0] = jnp.ones((1, 2), dtype=jnp.asarray(values[0]).dtype)
    return _replace_component_store(
        state,
        "DST",
        "received",
        FieldStore(field_names=store.field_names, values=tuple(values)),
    )


def _state_with_wrong_sent_dtype(state: RunState) -> RunState:
    store = state._component_state("SRC").sent
    values = list(store.values)
    values[0] = jnp.asarray(values[0], dtype=jnp.int32)
    return _replace_component_store(
        state,
        "SRC",
        "sent",
        FieldStore(field_names=store.field_names, values=tuple(values)),
    )


def _state_with_nonfinite_mask(state: RunState) -> RunState:
    masks = state._fractional_masks
    values = list(masks.values)
    values[0] = jnp.full_like(jnp.asarray(values[0]), jnp.nan)
    return _rebuild_state(
        state,
        fractional_masks=FieldStore(
            field_names=masks.field_names,
            values=tuple(values),
        ),
    )


def _state_with_wrong_mask_dtype(state: RunState) -> RunState:
    masks = state._fractional_masks
    values = list(masks.values)
    values[0] = jnp.asarray(values[0], dtype=jnp.int32)
    return _rebuild_state(
        state,
        fractional_masks=FieldStore(
            field_names=masks.field_names,
            values=tuple(values),
        ),
    )


_STATE_CORRUPTIONS = (
    pytest.param(_state_with_wrong_longitude, "coordinate", id="longitude"),
    pytest.param(_state_with_wrong_field_dtype, "dtype", id="field-dtype"),
    pytest.param(
        _state_with_out_of_range_mask,
        "fractional mask",
        id="mask-range",
    ),
    pytest.param(_state_with_wrong_payload, "payload", id="payload-schema"),
)


_INCOMING_STATE_INVARIANTS = (
    pytest.param(
        _state_with_wrong_component_order,
        "component order",
        id="component-order",
    ),
    pytest.param(
        _state_with_duplicate_component_name,
        "duplicate SRC",
        id="component-duplicate",
    ),
    pytest.param(_state_with_wrong_grid_name, "grid name", id="grid-name"),
    pytest.param(
        _state_with_wrong_coordinate_dtype,
        "coordinate.*dtype",
        id="coordinate-dtype",
    ),
    pytest.param(
        _state_with_unexpected_grid_edges,
        "longitude_edges.*presence",
        id="edge-presence",
    ),
    pytest.param(
        _state_with_unexpected_binary_mask,
        "binary mask.*presence",
        id="binary-mask-presence",
    ),
    pytest.param(
        _state_with_reordered_fields,
        "runtime fields names",
        id="field-order",
    ),
    pytest.param(
        _state_with_wrong_received_shape,
        "runtime received.*shape",
        id="received-shape",
    ),
    pytest.param(
        _state_with_wrong_sent_dtype,
        "runtime sent.*dtype",
        id="sent-dtype",
    ),
    pytest.param(
        _state_with_nonfinite_mask,
        "fractional mask.*finite",
        id="mask-finite",
    ),
    pytest.param(
        _state_with_wrong_mask_dtype,
        "fractional mask.*dtype",
        id="mask-dtype",
    ),
)


def _state_coupler(*, backend: Any | None = None) -> Coupler:
    route = Exchange(
        "SRC", "DST", ("scalar",), quantity="intensive", route_id="scalar-route"
    )
    runtime = RuntimeOptions(backend=backend) if backend is not None else None
    source, target = _components()
    return Coupler(
        _clock(),
        components=(source, target),
        exchanges=(route,),
        run_order=("SRC", "DST"),
        runtime=runtime,
    )


@pytest.mark.parametrize(
    "corrupt, message", _STATE_CORRUPTIONS + _INCOMING_STATE_INVARIANTS
)
def test_supplied_foreign_state_is_strictly_validated(
    corrupt: Any,
    message: str,
) -> None:
    coupler = _state_coupler()
    foreign = corrupt(coupler.initial_state())

    with pytest.raises(CouplerError, match=message):
        coupler.run(foreign)


@pytest.mark.parametrize("corrupt, message", _STATE_CORRUPTIONS)
def test_driver_rejects_foreign_state_before_component_dispatch(
    corrupt: Any,
    message: str,
) -> None:
    class DriverBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: Any,
            chunk: Any,
            driver: Any,
        ) -> RunState:
            _ = context
            driver.run_step(corrupt(state), chunk.steps[0])
            pytest.fail("invalid state reached component dispatch")

    with pytest.raises(CouplerError, match=message):
        _state_coupler(backend=DriverBackend()).run()


@pytest.mark.parametrize("corrupt, message", _STATE_CORRUPTIONS)
def test_backend_returned_foreign_state_is_strictly_validated(
    corrupt: Any,
    message: str,
) -> None:
    class ReturningBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: Any,
            chunk: Any,
            driver: Any,
        ) -> RunState:
            _ = context
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return cast(RunState, corrupt(state))

    with pytest.raises(CouplerError, match=message):
        _state_coupler(backend=ReturningBackend()).run()


@pytest.mark.parametrize(
    "corrupt, message",
    (
        pytest.param(_state_with_wrong_longitude, "coordinate", id="longitude"),
        pytest.param(
            _state_with_out_of_range_mask,
            "fractional mask",
            id="mask-range",
        ),
    ),
)
def test_transformed_invalid_state_values_are_rejected(
    corrupt: Any,
    message: str,
) -> None:
    coupler = _state_coupler()
    foreign = corrupt(coupler.initial_state())
    compiled = jax.jit(lambda state: coupler.run(state))

    with pytest.raises(Exception, match=message):
        result = compiled(foreign)
        jax.block_until_ready(result)


def test_compatible_foreign_state_remains_accepted() -> None:
    class ReturningBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: Any,
            chunk: Any,
            driver: Any,
        ) -> RunState:
            _ = context
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            leaves, tree = jax.tree.flatten(state)
            return cast(RunState, jax.tree.unflatten(tree, leaves))

    final = _state_coupler(backend=ReturningBackend()).run()
    assert tuple(final.components()) == ("SRC", "DST")


def _interleaved_route_coupler(backend: Any) -> Coupler:
    grid = make_test_grid(name="interleaved-routes")
    source = DataComponent(
        "SRC",
        grid,
        {"scalar": 1.0, "other": 2.0},
        spec=ComponentSpec(outputs=("scalar", "other")),
    )
    first_target = DataComponent(
        "FIRST",
        grid,
        {"scalar": 0.0, "other": 0.0},
        spec=ComponentSpec(inputs=("scalar", "other")),
    )
    second_target = DataComponent(
        "SECOND",
        grid,
        {"other": 0.0},
        spec=ComponentSpec(inputs=("other",)),
    )
    return Coupler(
        _clock(),
        components=(source, first_target, second_target),
        exchanges=(
            Exchange(
                "SRC", "FIRST", ("scalar",), quantity="intensive", route_id="route-1"
            ),
            Exchange(
                "SRC", "SECOND", ("other",), quantity="intensive", route_id="route-2"
            ),
            Exchange(
                "SRC", "FIRST", ("other",), quantity="intensive", route_id="route-3"
            ),
        ),
        run_order=("SRC", "FIRST", "SECOND"),
        runtime=RuntimeOptions(backend=backend),
    )


def test_driver_validation_preserves_interleaved_route_order() -> None:
    class DriverBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: Any,
            chunk: Any,
            driver: Any,
        ) -> RunState:
            _ = context
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return cast(RunState, state)

    final = _interleaved_route_coupler(DriverBackend()).run()
    assert tuple(final.components()) == ("SRC", "FIRST", "SECOND")


def test_backend_return_validation_preserves_interleaved_route_order() -> None:
    class ReturningBackend:
        def execute(
            self,
            state: RunState,
            *,
            context: Any,
            chunk: Any,
            driver: Any,
        ) -> RunState:
            _ = context
            for plan in chunk.steps:
                state = driver.run_step(state, plan)
            return state

    final = _interleaved_route_coupler(ReturningBackend()).run()
    assert tuple(final.components()) == ("SRC", "FIRST", "SECOND")


def test_fractional_mask_names_preserve_canonical_route_order() -> None:
    coupler = _interleaved_route_coupler("auto")
    state = coupler.initial_state()
    masks = state._fractional_masks
    foreign = _rebuild_state(
        state,
        fractional_masks=FieldStore(
            field_names=tuple(reversed(masks.field_names)),
            values=tuple(reversed(masks.values)),
        ),
    )

    with pytest.raises(CouplerError, match="fractional masks names"):
        coupler.run(foreign)


def test_strict_state_validation_remains_jit_jvp_and_reverse_safe() -> None:
    coupler = _state_coupler()
    initial = coupler.initial_state()
    compiled = jax.jit(lambda state: coupler.run(state))
    compiled_state = compiled(initial)
    assert tuple(compiled_state.components()) == ("SRC", "DST")

    def objective(value: jax.Array) -> jax.Array:
        state = initial.replace_fields(
            "SRC",
            {"scalar": jnp.full((2, 2), value)},
        )
        return jnp.mean(coupler.run(state).component("DST").field("scalar"))

    primal, tangent = jax.jvp(objective, (jnp.asarray(2.0),), (jnp.asarray(1.0),))
    reverse = jax.grad(objective)(jnp.asarray(2.0))
    assert jnp.allclose(primal, 2.0)
    assert jnp.allclose(tangent, 1.0)
    assert jnp.allclose(reverse, 1.0)
