from __future__ import annotations

from collections.abc import Iterable, Sequence
import ast
from datetime import datetime
import importlib
import inspect
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, ModuleType
from typing import Any, get_type_hints

import pytest

import vercor
from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import (
    CallableComponent,
    Component,
    ComponentSpec,
    LifecycleHooks,
    SetupContext,
    SetupResult,
    StepContext,
)
from vercor.coupler import Coupler
from vercor.exceptions import CouplerError
from vercor.exchanges import Exchange
from vercor.jax_logging import LoggerLike
from vercor.physics import PhysicalConstants
from vercor.runtime import RuntimeOptions
from vercor.topology import TopologyContext

from tests._distribution_support import _cached_build_pythonpath

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT_EXPORTS = (
    "Clock",
    "Coupler",
    "Exchange",
    "RectilinearGrid",
    "RunState",
    "RuntimeOptions",
)
PUBLIC_MODULE_EXPORTS = {
    "vercor.assets": ("VERCOR_ASSETS_BASE_URL", "ensure_registered_asset"),
    "vercor.calendar": (
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
    ),
    "vercor.clock": ("Clock",),
    "vercor.components": (
        "CallableComponent",
        "Component",
        "ComponentSpec",
        "DataComponent",
        "LifecycleHooks",
        "PrefillContext",
        "PrefillResult",
        "SetupContext",
        "SetupResult",
        "StepContext",
        "StepResult",
        "TransferPolicy",
        "ValidationContext",
    ),
    "vercor.coupler": ("Coupler",),
    "vercor.diagnostics": (
        "ComponentMetric",
        "combine_surface_temperatures",
        "component_vector_speed",
        "plot_component_scalar_vector_comparison",
        "print_component_field_means_table",
        "safe_component_nanmean",
        "total_surface_temperature",
    ),
    "vercor.dtypes": (
        "DTypePolicy",
        "PrecisionPolicy",
        "ShapeLike",
        "as_jax_index_array",
        "as_jax_real_array",
        "dtype_policy",
        "jax_arange",
        "jax_full",
        "jax_index_dtype",
        "jax_linspace",
        "jax_ones",
        "jax_real_dtype",
        "jax_zeros",
    ),
    "vercor.exceptions": (
        "AssetError",
        "ComponentError",
        "CouplerError",
        "ExchangeError",
        "GridError",
        "RegridderError",
    ),
    "vercor.exchanges": ("Exchange",),
    "vercor.field_layout": (
        "CANONICAL_DATA_LAYOUTS",
        "canonical_data_layout_description",
        "canonical_grid_field_shape",
        "canonical_grid_field_shape_error",
        "canonicalize_time_last_level_field",
        "canonicalize_time_last_surface_field",
        "is_canonical_grid_field_shape",
        "validate_canonical_grid_field_shape",
        "validate_component_data_layout",
    ),
    "vercor.fields": (
        "COMMON_FIELD_NAMES",
        "ExchangeField",
        "VectorField",
        "vector",
    ),
    "vercor.fluxes": (
        "cdn",
        "compute_air_density",
        "compute_hybrid_pressure_levels",
        "compute_hybrid_sigma_full_level_altitudes",
        "compute_ocean_surface_fluxes",
        "compute_potential_temperature",
        "compute_sigma_pressure_levels",
        "get_altitudes_hybrid_sigma_levels",
        "get_altitudes_sigma_levels",
        "psimhu",
        "psixhu",
        "qsat",
        "qsat_august_eqn",
        "shr_flux_atmIce",
    ),
    "vercor.forcing_data": ("read_forcing",),
    "vercor.forcing_index": (
        "ForcingYearType",
        "daily_forcing_day_of_year",
        "daily_forcing_index",
        "day_of_year_360_to_gregorian",
        "gregorian_month_lengths",
        "noleap_day_of_year",
    ),
    "vercor.grid_masks": (
        "check_remap_conservation",
        "check_total_lnd_ocn_mask_sum",
        "compute_land_mask",
        "compute_ocn_lnd_masks_on_atm_grid",
        "create_lnd_mask_from_ocn",
    ),
    "vercor.grid_geometry": ("centers_to_edges", "grids_identical"),
    "vercor.grids": ("RectilinearGrid",),
    "vercor.jax_logging": (
        "CANONICAL_LOG_DATE_FORMAT",
        "CANONICAL_LOG_FORMAT",
        "DEFAULT_LOGGER_NAME",
        "JaxCallbackLogger",
        "LoggerLike",
        "configure_python_logger",
        "effective_log_level",
        "emit_host_log",
        "get_default_logger",
        "logger_enabled_for",
        "normalize_log_level",
        "setup_logger",
    ),
    "vercor.output": (
        "OutputContext",
        "OutputFrame",
        "OutputProvider",
        "OutputSpec",
        "OutputTarget",
        "OutputVariable",
        "PeriodOutput",
        "SnapshotContext",
        "SnapshotWriter",
    ),
    "vercor.physics": ("PhysicalConstants",),
    "vercor.recipes": (
        "ATMOSPHERE_TO_DATA_OCEAN_FIELDS",
        "ATMOSPHERE_TO_JCM_LAND_FLUX_FIELDS",
        "ATMOSPHERE_TO_LAND_BASIC_FIELDS",
        "ATMOSPHERE_TO_LAND_RADIATION_FIELDS",
        "ATMOSPHERE_TO_LAND_STATE_FIELDS",
        "ATMOSPHERE_TO_OCEAN_RADIATION_FIELDS",
        "ATMOSPHERE_TO_OCEAN_STATE_FIELDS",
        "ATMOSPHERE_TO_VEROS_FORCING_FIELDS",
        "JCM_ATMOSPHERE_TO_SLAB_OCEAN_FIELDS",
        "JCM_LAND_TO_ATMOSPHERE_FIELDS",
        "LAND_TO_ATMOSPHERE_SOIL_FIELDS",
        "LAND_TO_ATMOSPHERE_SURFACE_FIELDS",
        "OCEAN_TO_ATMOSPHERE_SURFACE_FIELDS",
        "OCEAN_TO_SEAICE_SURFACE_FIELDS",
        "SEAICE_TO_OCEAN_FIELDS",
        "SLAB_ATMOSPHERE_TO_LAND_FLUX_FIELDS",
        "SLAB_ATMOSPHERE_TO_OCEAN_FIELDS",
        "SLAB_ATMOSPHERE_TO_OCEAN_FLUX_FIELDS",
    ),
    "vercor.regridding": (
        "Regridder",
        "RegridderFactory",
        "VectorRegridder",
        "bilinear",
        "conservative",
    ),
    "vercor.runtime": (
        "ExecutionBackend",
        "ExecutionChunk",
        "ExecutionContext",
        "ExecutionPlan",
        "RuntimeDriver",
        "RuntimeOptions",
        "SequentialWorkflow",
        "StepPlan",
        "Workflow",
        "WorkflowContext",
    ),
    "vercor.setups": (
        "CAMulatorConfig",
        "JAXGCMConfig",
        "JCMLandAtmosphereConfig",
        "JCMLandAtmosphereSetup",
        "JCMInputs",
        "Spinup",
        "VerosConfig",
        "load_jcm_inputs",
        "make_slab_atmosphere",
        "make_slab_land",
        "make_slab_ocean",
        "make_slab_seaice",
        "make_jcm_land_atmosphere",
        "make_camulator_gcm",
        "make_camulator_land",
        "make_era5_atmosphere",
        "make_era5_land",
        "make_era5_ocean",
        "make_erainterim_ocean",
        "make_jax_gcm",
        "make_jcm_land",
        "make_veros_gcm",
    ),
    "vercor.state": (
        "ComponentState",
        "FieldLookupScope",
        "FieldScope",
        "RunState",
    ),
    "vercor.time_selection": (
        "datetime_to_seconds_in_year",
        "get_periodic_interval",
    ),
    "vercor.topology": (
        "ExchangeTopologyPatch",
        "SurfaceMaskPolicy",
        "TopologyContext",
        "TopologyPolicy",
    ),
    "vercor.types": ("RuntimeArray",),
}


def _component(name: str = "MODEL", *, setup: Any = None) -> CallableComponent:
    lifecycle = LifecycleHooks() if setup is None else LifecycleHooks(setup=setup)
    return CallableComponent(
        name,
        make_test_grid(name.lower()),
        lambda fields: {"value": fields["value"] + 1.0},
        spec=ComponentSpec(
            outputs=("value",),
            initial_fields={"value": 0.0},
            lifecycle=lifecycle,
        ),
    )


def _clock(*, steps: int = 1) -> Clock:
    return Clock(datetime(2000, 1, 1), 60.0, steps)


def test_root_exports_exactly_the_six_primary_symbols() -> None:
    assert tuple(vercor.__all__) == ROOT_EXPORTS
    assert all(hasattr(vercor, name) for name in ROOT_EXPORTS)


def test_public_module_inventory_exactly_matches_the_manifest() -> None:
    package_root = PROJECT_ROOT / "vercor"
    discovered = {
        f"vercor.{path.stem if path.is_file() else path.name}"
        for path in package_root.iterdir()
        if not path.name.startswith("_")
        and (
            (path.is_file() and path.suffix == ".py")
            or (path.is_dir() and (path / "__init__.py").is_file())
        )
    }

    assert discovered == set(PUBLIC_MODULE_EXPORTS)


@pytest.mark.parametrize("module_name", tuple(PUBLIC_MODULE_EXPORTS))
def test_supported_public_module_has_exact_manifest(module_name: str) -> None:
    module = importlib.import_module(module_name)
    expected = PUBLIC_MODULE_EXPORTS[module_name]
    assert tuple(module.__all__) == expected
    assert all(hasattr(module, name) for name in expected)


def test_public_module_namespaces_do_not_leak_non_exported_vercor_objects() -> None:
    violations: list[str] = []
    for module_name, exports in PUBLIC_MODULE_EXPORTS.items():
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if name.startswith("_") or name in exports:
                continue
            origin = (
                value.__name__
                if isinstance(value, ModuleType)
                else getattr(value, "__module__", None)
            )
            if isinstance(origin, str) and origin.startswith("vercor"):
                violations.append(f"{module_name}.{name} ({origin})")

    assert violations == [], "non-exported public names: " + "; ".join(violations)


def test_advanced_symbols_have_one_canonical_public_owner() -> None:
    owners: dict[str, list[str]] = {}
    for module_name, exports in PUBLIC_MODULE_EXPORTS.items():
        for symbol in exports:
            owners.setdefault(symbol, []).append(module_name)

    allowed_root_conveniences = set(ROOT_EXPORTS)
    duplicates = {
        symbol: modules
        for symbol, modules in owners.items()
        if len(modules) > 1 and symbol not in allowed_root_conveniences
    }
    assert duplicates == {}
    assert "DTypePolicy" not in PUBLIC_MODULE_EXPORTS["vercor.runtime"]
    assert "RunState" not in PUBLIC_MODULE_EXPORTS["vercor.runtime"]
    assert "ComponentState" not in PUBLIC_MODULE_EXPORTS["vercor.runtime"]
    assert owners["PhysicalConstants"] == ["vercor.physics"]


def test_manifested_symbols_are_absent_from_every_non_owner_public_module() -> None:
    modules = {
        module_name: importlib.import_module(module_name)
        for module_name in PUBLIC_MODULE_EXPORTS
    }
    modules["vercor"] = vercor
    violations: list[str] = []

    for owner_name, exports in PUBLIC_MODULE_EXPORTS.items():
        for symbol in exports:
            for module_name, module in modules.items():
                if module_name == owner_name:
                    continue
                if module_name == "vercor" and symbol in ROOT_EXPORTS:
                    continue
                if symbol in vars(module):
                    violations.append(f"{symbol}: {owner_name} -> {module_name}")

    assert violations == [], "non-owner aliases: " + "; ".join(violations[:20])


def test_settings_are_absent_from_primary_assembly() -> None:
    coupler = Coupler(_clock())

    assert not hasattr(coupler, "settings")


@pytest.mark.parametrize("context_type", (SetupContext, StepContext, TopologyContext))
def test_settings_are_absent_from_public_contexts(context_type: type[Any]) -> None:
    assert "settings" not in inspect.signature(context_type).parameters
    assert "settings" not in get_type_hints(context_type)


def test_coupler_constructor_is_the_only_public_assembly_surface() -> None:
    parameters = inspect.signature(Coupler).parameters
    expected = {
        "clock": (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.empty,
            Clock,
        ),
        "components": (inspect.Parameter.KEYWORD_ONLY, (), Iterable[Component]),
        "exchanges": (inspect.Parameter.KEYWORD_ONLY, (), Iterable[Exchange]),
        "run_order": (inspect.Parameter.KEYWORD_ONLY, (), Sequence[str]),
        "runtime": (inspect.Parameter.KEYWORD_ONLY, None, RuntimeOptions | None),
        "constants": (
            inspect.Parameter.KEYWORD_ONLY,
            None,
            PhysicalConstants | None,
        ),
        "logger": (inspect.Parameter.KEYWORD_ONLY, None, LoggerLike | None),
        "log_level": (inspect.Parameter.KEYWORD_ONLY, "INFO", int | str),
    }
    hints = get_type_hints(Coupler.__init__)

    assert tuple(parameters) == tuple(expected)
    for name, (kind, default, annotation) in expected.items():
        assert parameters[name].kind is kind
        assert parameters[name].default == default
        assert hints[name] == annotation
    assert hints["return"] is type(None)
    for name in (
        "add_component",
        "add_exchange",
        "add_exchanges",
        "set_run_order",
        "_invalidate_preparation",
    ):
        assert not hasattr(Coupler, name)


def test_constructor_rejects_duplicate_component_names() -> None:
    with pytest.raises(CouplerError, match="Duplicate component name.*MODEL"):
        Coupler(_clock(), components=(_component(), _component()))


def test_constructor_rejects_invalid_clock_and_log_level_eagerly() -> None:
    with pytest.raises(TypeError, match="clock must be Clock"):
        Coupler(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown logging level"):
        Coupler(_clock(), log_level="LOUD")
    with pytest.raises(TypeError):
        Coupler(_clock(), log_level=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown logging level"):
        Coupler(_clock(), logger=object(), log_level="LOUD")  # type: ignore[arg-type]


def test_constructor_normalizes_log_level_before_configuring_custom_logger() -> None:
    class RecordingLogger:
        def __init__(self) -> None:
            self.levels: list[int | str] = []

        def debug(self, message: object, *args: Any, **kwargs: Any) -> None:
            _ = message, args, kwargs

        def info(self, message: object, *args: Any, **kwargs: Any) -> None:
            _ = message, args, kwargs

        def warning(self, message: object, *args: Any, **kwargs: Any) -> None:
            _ = message, args, kwargs

        def error(self, message: object, *args: Any, **kwargs: Any) -> None:
            _ = message, args, kwargs

        def setLevel(self, level: int | str) -> None:
            self.levels.append(level)

        def isEnabledFor(self, level: int) -> bool:
            _ = level
            return True

    logger = RecordingLogger()
    coupler = Coupler(_clock(), logger=logger, log_level="debug")

    assert logger.levels == [logging.DEBUG]
    assert coupler.log_level == "debug"


@pytest.mark.parametrize(
    ("run_order", "message"),
    [
        (("MODEL", "MODEL"), "Duplicate run-order component.*MODEL"),
        (("UNKNOWN",), "Unknown run-order component.*UNKNOWN"),
    ],
)
def test_constructor_rejects_invalid_nonempty_run_order(
    run_order: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(CouplerError, match=message):
        Coupler(_clock(), components=(_component(),), run_order=run_order)


@pytest.mark.parametrize(
    ("run_order", "message"),
    [
        (None, "run_order must be a sequence"),
        ("MODEL", "run_order must be a sequence"),
        ({"MODEL"}, "run_order must be a sequence"),
        ((name for name in ("MODEL",)), "run_order must be a sequence"),
        (("MODEL", object()), "run_order entries must be non-empty strings"),
        (("",), "run_order entries must be non-empty strings"),
    ],
)
def test_constructor_rejects_invalid_run_order_kinds(
    run_order: object,
    message: str,
) -> None:
    with pytest.raises(CouplerError, match=message):
        Coupler(
            _clock(),
            components=(_component(),),
            run_order=run_order,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("components", None, "components must be an iterable"),
        ("components", "MODEL", "components must be an iterable"),
        ("components", (object(),), "invalid component"),
        ("exchanges", None, "exchanges must be an iterable"),
        ("exchanges", "route", "exchanges must be an iterable"),
        ("exchanges", (object(),), "invalid exchange"),
    ],
)
def test_constructor_rejects_invalid_component_and_exchange_iterables(
    keyword: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {keyword: value}
    with pytest.raises(CouplerError, match=message):
        Coupler(_clock(), **kwargs)  # type: ignore[arg-type]


def test_constructor_rejects_duplicate_exchanges_and_unknown_endpoints() -> None:
    component = _component()
    exchange = Exchange("MODEL", "MODEL", ("value",))
    equal_exchange = Exchange("MODEL", "MODEL", ("value",))
    assert equal_exchange is not exchange
    assert equal_exchange == exchange
    with pytest.raises(
        CouplerError, match="Exchange route ID 'MODEL->MODEL' must be unique"
    ):
        Coupler(
            _clock(),
            components=(component,),
            exchanges=(exchange, equal_exchange),
        )
    with pytest.raises(CouplerError, match="unknown source component.*UNKNOWN"):
        Coupler(
            _clock(),
            components=(component,),
            exchanges=(Exchange("UNKNOWN", "MODEL", ("value",)),),
        )
    with pytest.raises(CouplerError, match="unknown target component.*UNKNOWN"):
        Coupler(
            _clock(),
            components=(component,),
            exchanges=(Exchange("MODEL", "UNKNOWN", ("value",)),),
        )


def test_constructor_rejects_exchange_fan_in_before_component_setup() -> None:
    setup_calls = 0

    def setup(component: Any, context: Any) -> SetupResult:
        nonlocal setup_calls
        _ = component, context
        setup_calls += 1
        return SetupResult()

    with pytest.raises(CouplerError, match="fan-in conflict.*DST.*value"):
        Coupler(
            _clock(),
            components=(
                _component("SRC_A", setup=setup),
                _component("SRC_B", setup=setup),
                _component("DST", setup=setup),
            ),
            exchanges=(
                Exchange("SRC_A", "DST", ("value",), route_id="route-a"),
                Exchange("SRC_B", "DST", ("value",), route_id="route-b"),
            ),
            run_order=("SRC_A", "SRC_B", "DST"),
        )

    assert setup_calls == 0


def test_constructor_rejects_duplicate_route_id_before_component_setup() -> None:
    setup_calls = 0

    def setup(component: Any, context: Any) -> SetupResult:
        nonlocal setup_calls
        _ = component, context
        setup_calls += 1
        return SetupResult()

    with pytest.raises(
        CouplerError, match="Exchange route ID 'SRC->DST' must be unique"
    ):
        Coupler(
            _clock(),
            components=(
                _component("SRC", setup=setup),
                _component("DST", setup=setup),
            ),
            exchanges=(
                Exchange("SRC", "DST", ("value",)),
                Exchange("SRC", "DST", ("other",)),
            ),
            run_order=("SRC", "DST"),
        )

    assert setup_calls == 0


def test_exchange_rejects_noncallable_factory_before_component_setup() -> None:
    setup_calls = 0

    def setup(component: Any, context: Any) -> SetupResult:
        nonlocal setup_calls
        _ = component, context
        setup_calls += 1
        return SetupResult()

    with pytest.raises(TypeError, match="regridder_factory must be callable"):
        Exchange(
            "SRC",
            "DST",
            ("value",),
            regridder_factory=object(),  # type: ignore[arg-type]
        )

    assert setup_calls == 0


@pytest.mark.parametrize(
    ("keyword", "value", "error", "message"),
    [
        ("runtime", object(), TypeError, "runtime must be RuntimeOptions"),
        ("constants", object(), TypeError, "constants must be PhysicalConstants"),
        ("logger", object(), TypeError, "logger must satisfy LoggerLike"),
    ],
)
def test_constructor_validates_runtime_constants_and_logger(
    keyword: str,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        Coupler(_clock(), **{keyword: value})  # type: ignore[arg-type]


def test_public_configuration_views_are_stable_immutable_snapshots() -> None:
    component = _component()
    exchange = Exchange("MODEL", "MODEL", ("value",))
    components = [component]
    exchanges = [exchange]
    run_order = ["MODEL"]
    coupler = Coupler(
        _clock(),
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=RuntimeOptions(),
        constants=PhysicalConstants(),
    )

    components.clear()
    exchanges.clear()
    run_order.clear()

    assert isinstance(coupler.components, MappingProxyType)
    assert coupler.components is coupler.components
    assert tuple(coupler.components) == ("MODEL",)
    assert coupler.components["MODEL"] is component
    assert coupler.exchanges == (exchange,)
    assert coupler.run_order == ("MODEL",)
    with pytest.raises(TypeError):
        coupler.components["OTHER"] = component  # type: ignore[index]


@pytest.mark.parametrize(
    ("attribute", "replacement"),
    [
        ("clock", _clock(steps=2)),
        ("runtime", RuntimeOptions(backend="host")),
        ("constants", PhysicalConstants(gravity=9.7)),
        ("logger", object()),
        ("log_level", "DEBUG"),
    ],
)
def test_coupler_configuration_attributes_are_read_only(
    attribute: str,
    replacement: object,
) -> None:
    coupler = Coupler(_clock())

    with pytest.raises(AttributeError):
        setattr(coupler, attribute, replacement)


@pytest.mark.parametrize("attribute", ("components", "exchanges", "run_order"))
def test_public_assembly_attributes_reject_rebinding(attribute: str) -> None:
    coupler = Coupler(_clock())

    with pytest.raises(AttributeError):
        setattr(coupler, attribute, ())


def test_caller_cannot_mutate_the_clock_after_coupler_construction() -> None:
    clock = _clock()
    coupler = Coupler(clock)

    with pytest.raises(AttributeError):
        clock.steps = 2  # type: ignore[misc]

    assert coupler.clock.steps == 1


def test_setup_runs_once_across_initial_state_and_run_reuse() -> None:
    setup_calls = 0

    def setup(component: Any, context: Any) -> SetupResult:
        nonlocal setup_calls
        _ = component, context
        setup_calls += 1
        return SetupResult()

    coupler = Coupler(
        _clock(),
        components=(_component(setup=setup),),
        run_order=("MODEL",),
    )
    first = coupler.initial_state()
    second = coupler.initial_state()
    coupler.run(first)

    assert setup_calls == 1
    assert tuple(first.components()) == tuple(second.components()) == ("MODEL",)


def test_empty_run_order_is_explicit_setup_only_and_run_is_noop() -> None:
    setup_calls = 0
    validate_calls = 0
    step_calls = 0

    def setup(component: Any, context: Any) -> SetupResult:
        nonlocal setup_calls
        _ = component, context
        setup_calls += 1
        return SetupResult(fields={"value": 4.0})

    def validate(component: Any, context: Any) -> None:
        nonlocal validate_calls
        _ = component, context
        validate_calls += 1

    def step(fields: Any) -> dict[str, Any]:
        nonlocal step_calls
        step_calls += 1
        return {"value": fields["value"] + 1.0}

    component = CallableComponent(
        "MODEL",
        make_test_grid("model"),
        step,
        spec=ComponentSpec(
            outputs=("value",),
            lifecycle=LifecycleHooks(setup=setup, validate=validate),
        ),
    )
    coupler = Coupler(_clock(), components=(component,))
    initial = coupler.initial_state()
    final = coupler.run(initial)

    assert setup_calls == 1
    # Initial construction, supplied-state validation, and the empty workflow's
    # chunk result all retain the same prepared component schema.
    assert validate_calls == 3
    assert step_calls == 0
    assert tuple(initial.components()) == ("MODEL",)
    assert tuple(final.components()) == tuple(initial.components())
    assert float(initial.component("MODEL").field("value")[0, 0]) == 4.0
    assert float(final.component("MODEL").field("value")[0, 0]) == 4.0


def test_public_coupler_annotations_resolve_without_private_types() -> None:
    hints = get_type_hints(Coupler.__init__)
    components_hint = repr(hints["components"])
    components_property = inspect.getattr_static(Coupler, "components")
    assert isinstance(components_property, property)
    assert components_property.fget is not None
    assert "vercor.components.contracts.Component" in components_hint
    assert "vercor._" not in components_hint
    assert "_ComponentDeclaration" not in components_hint
    assert "_ComponentBinding" not in repr(get_type_hints(components_property.fget))
    assert "_Component" not in inspect.getsource(Coupler)


def test_public_callable_annotations_resolve_without_private_types() -> None:
    violations: list[str] = []
    forbidden_names = (
        "_ComponentAdapter",
        "_ComponentBinding",
        "_ComponentDeclaration",
    )

    def check_hints(label: str, callable_object: Any) -> None:
        try:
            hints = get_type_hints(callable_object)
        except NameError as exc:
            violations.append(f"{label}: unresolved annotation ({exc})")
            return
        except TypeError:
            return
        for parameter, hint in hints.items():
            rendered = repr(hint)
            if "vercor._" in rendered or any(
                name in rendered for name in forbidden_names
            ):
                violations.append(f"{label}.{parameter}: {rendered}")

    for module_name, exports in PUBLIC_MODULE_EXPORTS.items():
        module = importlib.import_module(module_name)
        for export_name in exports:
            exported = getattr(module, export_name)
            label = f"{module_name}.{export_name}"
            if inspect.isfunction(exported):
                check_hints(label, exported)
                continue
            if not inspect.isclass(exported):
                continue

            check_hints(label, exported)
            check_hints(f"{label}.__init__", exported.__init__)
            for method_name, descriptor in vars(exported).items():
                if method_name.startswith("_"):
                    continue
                if isinstance(descriptor, property):
                    if descriptor.fget is not None:
                        check_hints(f"{label}.{method_name}", descriptor.fget)
                elif isinstance(descriptor, (classmethod, staticmethod)):
                    check_hints(f"{label}.{method_name}", descriptor.__func__)
                elif inspect.isfunction(descriptor):
                    check_hints(f"{label}.{method_name}", descriptor)

    assert violations == [], "private public hints: " + "; ".join(violations[:20])


def test_regridder_factory_is_one_runtime_protocol_with_public_hints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep runtime and static factory contracts on one public protocol."""

    source = (PROJECT_ROOT / "vercor" / "regridding.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "if TYPE_CHECKING" not in source
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "RegridderFactory"
        for node in tree.body
    )

    external_extension_source = (
        PROJECT_ROOT / "tests" / "fixtures" / "external_extension_test_fixture" / "src"
    )
    assert external_extension_source.is_dir()
    monkeypatch.syspath_prepend(str(external_extension_source))
    from vercor.grids import RectilinearGrid
    from vercor.regridding import Regridder, RegridderFactory, bilinear, conservative
    from external_extension_test_fixture import PluginRegridderFactory

    assert isinstance(PluginRegridderFactory("typed-route"), RegridderFactory)
    hints = get_type_hints(RegridderFactory.__call__)
    assert set(hints) == {"source_grid", "target_grid", "return"}
    assert hints["source_grid"] is RectilinearGrid
    assert hints["target_grid"] is RectilinearGrid
    assert hints["return"] is Regridder
    assert isinstance(bilinear, RegridderFactory)
    assert isinstance(conservative, RegridderFactory)
    assert Exchange("SOURCE", "TARGET", ("field",)).regridder_factory is bilinear


def test_prepared_runtime_has_no_reflective_configuration_snapshot() -> None:
    from vercor._runtime import prepared

    source = inspect.getsource(prepared)
    for marker in (
        "configuration_snapshot",
        "validate_configuration",
        "__dict__",
        "__slots__",
        "is_dataclass",
        "isroutine",
        "_instance_configuration_snapshot",
        "_object_configuration_snapshot",
    ):
        assert marker not in source
    assert (
        "configuration_snapshot" not in prepared.PreparedCoupling.__dataclass_fields__
    )


def test_examples_and_current_plugin_use_direct_constructor_assembly() -> None:
    extension_plugin = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "external_extension_test_fixture"
        / "src/external_extension_test_fixture/plugin.py"
    )
    assert extension_plugin.is_file()
    paths = (
        *sorted((PROJECT_ROOT / "vercor" / "setups" / "gallery").glob("*.py")),
        extension_plugin,
    )
    forbidden = (
        ".add_component(",
        ".add_exchange(",
        ".add_exchanges(",
        ".set_run_order(",
        "vercor.coupling",
        "CouplerSpec",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, (path, marker)


def test_optional_setup_imports_remain_lazy() -> None:
    script = """
import sys
import vercor
import vercor.setups
banned = ('dinosaur', 'jax_gcm', 'veros', 'credit', 'torch', 'tensorflow')
loaded = sorted(name for name in sys.modules if name.split('.')[0].lower() in banned)
assert loaded == [], loaded
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_installed_wheel_preserves_the_complete_public_boundary(
    tmp_path: Path,
) -> None:
    distribution_dir = tmp_path / "dist"
    installed_root = tmp_path / "site-packages"
    distribution_dir.mkdir()
    installed_root.mkdir()
    environment = os.environ.copy()
    build_pythonpath = _cached_build_pythonpath()
    if build_pythonpath:
        environment["PYTHONPATH"] = build_pythonpath

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--outdir",
            str(distribution_dir),
            str(PROJECT_ROOT),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )
    wheel = next(distribution_dir.glob("vercor-*.whl"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-compile",
            "--no-deps",
            "--target",
            str(installed_root),
            str(wheel),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
        check=True,
    )

    probe = f"""
import importlib
from pathlib import Path
import pkgutil
import sys
from types import ModuleType

installed_root = Path({str(installed_root)!r}).resolve()
sys.path.insert(0, str(installed_root))
import vercor

root_exports = {ROOT_EXPORTS!r}
module_exports = {PUBLIC_MODULE_EXPORTS!r}

assert tuple(vercor.__all__) == root_exports
assert Path(vercor.__file__).resolve().is_relative_to(installed_root)

discovered = {{
    f"vercor.{{item.name}}"
    for item in pkgutil.iter_modules(vercor.__path__)
    if not item.name.startswith("_")
}}
assert discovered == set(module_exports), (discovered, set(module_exports))

modules = {{"vercor": vercor}}
for module_name, exports in module_exports.items():
    module = importlib.import_module(module_name)
    modules[module_name] = module
    assert tuple(module.__all__) == tuple(exports), module_name
    assert Path(module.__file__).resolve().is_relative_to(installed_root), module_name
    for name, value in vars(module).items():
        if name.startswith("_") or name in exports:
            continue
        origin = (
            value.__name__
            if isinstance(value, ModuleType)
            else getattr(value, "__module__", None)
        )
        assert not (
            isinstance(origin, str) and origin.startswith("vercor")
        ), (module_name, name, origin)

for owner_name, exports in module_exports.items():
    for symbol in exports:
        for module_name, module in modules.items():
            if module_name == owner_name:
                continue
            if module_name == "vercor" and symbol in root_exports:
                continue
            assert symbol not in vars(module), (symbol, owner_name, module_name)

"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
