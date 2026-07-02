from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vercor.runtime.surface_masks as surface_masks_module
import vercor.coupler as coupler_module
import vercor.output as output_module
import vercor.output.runtime as output_runtime_module
from tests._coverage_support import (
    DummyComponent,
    RecordingRegridder,
    capture_logger_output,
    make_test_grid,
)
from tests._runtime_helpers import (
    replace_runtime_topology_maps,
    run_scanned_coupler,
    runtime_state_from_coupler_components,
)
from tests.assertions import assert_allclose_compact
from vercor.calendar import ModelDateTime
from vercor.clock import Clock
from vercor.components.base import Component
from vercor.components.host import HostRuntimeComponent
from vercor.components.contexts import ComponentStepContext
from vercor.coupler import Coupler
from vercor.exceptions import ComponentError, CouplerError, ExchangerError
from vercor.exchange import Exchange
from vercor.jax_logging import (
    CANONICAL_LOG_DATE_FORMAT,
    CANONICAL_LOG_FORMAT,
    DEFAULT_LOGGER_NAME,
    JaxCallbackLogger,
    get_default_logger,
    setup_logger,
)
from vercor.regridders.bilinear import bilinear
from vercor.regridders.conservative import conservative
from vercor.runtime.contracts import RuntimeComponentContract
from vercor.runtime.exchange_dispatch import dispatch_component_exchanges
from vercor.output import output_masks_for_component
from vercor.output.adapters import register_component_snapshot_writer
from vercor.runtime.surface_masks import (
    apply_surface_exchange_masks,
    create_surface_exchange_masks,
    validate_land_mask_consistency,
)
from vercor.runtime.topology import build_exchange_topology
from vercor.runtime.topology_state import (
    ExchangeTopologyState,
    RuntimeTopologyMaps,
    SurfaceExchangeMasks,
)
from vercor.settings import VercorSettings


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.debug_messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.info_messages.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any) -> None:
        self.warning_messages.append(message.format(*args) if args else message)

    def debug(self, message: str, *args: Any) -> None:
        self.debug_messages.append(message.format(*args) if args else message)


class _RunComponent(Component):
    def __init__(self, name: str, events: list[str], timestamp: datetime) -> None:
        _ = timestamp
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.events = events
        self.data["temperature"] = np.ones((2, 2))

    def step_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        time = context.time
        time_label = "none" if time is None else time.isoformat()
        self.events.append(
            f"step_runtime:{self.name}:{time_label}:{context.dt_seconds}"
        )
        return component_state


class _LoggingRunComponent(Component):
    def __init__(self, name: str) -> None:
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.data["temperature"] = np.ones((2, 2), dtype=float)

    def step_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        assert context.logger is not None
        context.logger.info(
            "scanned {} {}",
            self.name,
            jnp.sum(component_state.data.get("temperature")),
        )
        return component_state


class _HostRunComponent(HostRuntimeComponent):
    def __init__(self, name: str, events: list[str] | None = None) -> None:
        super().__init__(name=name, grid=make_test_grid(name=name.lower()))
        self.events = events
        self.data["temperature"] = np.ones((2, 2))

    def step_host_runtime_state(
        self,
        component_state: Any,
        context: ComponentStepContext,
    ) -> Any:
        if self.events is not None:
            time = context.time
            time_label = "none" if time is None else time.isoformat()
            self.events.append(
                f"step_host:{self.name}:{time_label}:{context.dt_seconds}"
            )
        data = component_state.data.set(
            "temperature",
            component_state.data.get("temperature") + context.dt_seconds,
        )
        self.data["host_event"] = np.asarray(context.dt_seconds)
        return component_state.with_data(data.set("host_time_seen", np.asarray(1.0)))


def make_coupler() -> Coupler:
    return Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1))


def _snapshot_output_time_for_finalize(
    coupler: Coupler,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
    }
    coupler.components = cast(Any, components)
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    captured_snapshots: dict[str, Any] = {}

    def fake_write_outputs(**kwargs: Any) -> None:
        _ = kwargs

    def fake_write_snapshots(**kwargs: Any) -> None:
        captured_snapshots.update(kwargs)

    monkeypatch.setattr(
        output_module, "write_coupler_runtime_outputs", fake_write_outputs
    )
    monkeypatch.setattr(
        output_module, "write_coupler_component_snapshots", fake_write_snapshots
    )

    coupler.finalize(state)

    return captured_snapshots["output_time"]


def _topology_components() -> dict[str, DummyComponent]:
    lnd_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    return {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(
            name="OCN",
            grid=make_test_grid(
                name="ocn",
                binary_mask=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            ),
        ),
        "LND": DummyComponent(
            name="LND",
            grid=make_test_grid(name="lnd", binary_mask=lnd_mask),
        ),
    }


def _dispatch_runtime_fields(
    coupler: Coupler,
    runtime_state: Any,
    component_name: str,
) -> Any:
    return dispatch_component_exchanges(
        runtime_state,
        component_name,
        coupler.exchanges,
        coupler._runtime_resources.topology_maps.regridders,
    )


def _canonical_handler(logger: logging.Logger) -> logging.StreamHandler[Any]:
    for handler in logger.handlers:
        formatter = handler.formatter
        if (
            isinstance(handler, logging.StreamHandler)
            and formatter is not None
            and formatter._fmt == CANONICAL_LOG_FORMAT
        ):
            return handler
    raise AssertionError("canonical VerCOR stream handler is not configured")


def _format_canonical_record(
    handler: logging.StreamHandler[Any],
    name: str,
    level: int,
    message: str,
) -> str:
    assert handler.formatter is not None
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    record.created = datetime(2026, 5, 12, 17, 32, 50).timestamp()
    return handler.format(record)


def test_default_logger_uses_vercor_logger_name() -> None:
    logger = get_default_logger()

    assert DEFAULT_LOGGER_NAME == "VerCOR"
    assert logger.name == DEFAULT_LOGGER_NAME


def test_setup_logger_installs_canonical_owned_handler_format() -> None:
    callback_logger = setup_logger(level="INFO")
    logger = callback_logger.logger

    assert logger is get_default_logger()
    assert logger.propagate is False
    handler = _canonical_handler(logger)
    assert handler.formatter is not None
    assert handler.formatter.datefmt == CANONICAL_LOG_DATE_FORMAT

    formatted = _format_canonical_record(
        handler,
        logger.name,
        logging.INFO,
        "format probe",
    )

    assert formatted == "VerCOR: 2026-05-12 17:32:50 [INFO]: format probe"
    assert "INFO [VerCOR]" not in formatted
    assert "," not in formatted.split(" [INFO]:", maxsplit=1)[0]


def test_setup_logger_routes_child_loggers_through_parent_canonical_handler() -> None:
    child_logger = setup_logger(level="WARNING", name="VerCOR.test.callback").logger
    parent_logger = get_default_logger()

    assert child_logger.name == "VerCOR.test.callback"
    assert child_logger.handlers == []
    assert child_logger.propagate is True
    assert parent_logger.propagate is False

    formatted = _format_canonical_record(
        _canonical_handler(parent_logger),
        child_logger.name,
        logging.WARNING,
        "child warning",
    )

    assert formatted == "VerCOR: 2026-05-12 17:32:50 [WARNING]: child warning"
    assert "VerCOR.test.callback" not in formatted


@pytest.mark.fast_always
def test_coupler_runtime_component_views_returns_ordered_named_views() -> None:
    coupler = make_coupler()
    for component_name in ("ATM", "OCN", "LND"):
        coupler.register(
            DummyComponent(
                name=component_name,
                grid=make_test_grid(name=component_name.lower()),
            )
        )
    coupler.set_components_run_sequence(
        (
            "ATM",
            "OCN",
            "LND",
        )
    )

    runtime_state = coupler.create_runtime_state(prefill_missing=True)

    all_views = coupler.runtime_component_views(runtime_state)
    selected_views = coupler.runtime_component_views(
        runtime_state, names=("LND", "ATM")
    )

    assert tuple(all_views) == ("ATM", "OCN", "LND")
    assert tuple(selected_views) == ("LND", "ATM")
    assert all_views["ATM"].name == "ATM"
    assert selected_views["LND"].grid.name == "lnd"


def test_coupler_configures_injected_python_logger_with_canonical_boundary() -> None:
    logger_name = "VerCOR.test.injected-format"
    injected_logger = logging.getLogger(logger_name)
    injected_logger.handlers.clear()
    injected_logger.propagate = True

    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        logger=injected_logger,
        log_level="INFO",
    )

    assert isinstance(coupler.logger, JaxCallbackLogger)
    assert coupler.logger.logger is injected_logger
    assert injected_logger.handlers == []
    assert injected_logger.propagate is True

    formatted = _format_canonical_record(
        _canonical_handler(get_default_logger()),
        injected_logger.name,
        logging.INFO,
        "injected probe",
    )

    assert formatted == "VerCOR: 2026-05-12 17:32:50 [INFO]: injected probe"


def test_coupler_accepts_log_level_at_instantiation() -> None:
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        log_level="WARNING",
    )

    assert not coupler.logger.isEnabledFor(logging.INFO)
    assert coupler.logger.isEnabledFor(logging.WARNING)


@pytest.mark.fast_always
def test_coupler_module_does_not_reexport_logger_setup_helper() -> None:
    assert not hasattr(coupler_module, "setup_logger")


def test_coupler_wraps_injected_python_logger_for_scanned_runtime() -> None:
    logger_name = "VerCOR.test.injected"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        logger=logging.getLogger(logger_name),
        log_level="INFO",
    )
    coupler.components = {"ATM": cast(Any, _LoggingRunComponent("ATM"))}
    coupler.run_sequence = ("ATM",)

    with capture_logger_output(logger_name, set_logger_level=False) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert final_state.component_names == ("ATM",)
    assert "scanned ATM 4.0" in stream.getvalue()


def test_setup_logger_formats_traced_values_under_scan() -> None:
    logger_name = "VerCOR.test.callback"
    logger = setup_logger(level="INFO", name=logger_name)

    def scanned_total(seed: jax.Array) -> jax.Array:
        def body(total: jax.Array, value: jax.Array) -> tuple[jax.Array, None]:
            updated = total + value
            logger.info("callback value {}", updated)
            return updated, None

        result, _ = jax.lax.scan(body, seed, jnp.asarray([1.0, 2.0]))
        return result

    with capture_logger_output(logger_name, set_logger_level=False) as stream:
        assert jax.jit(scanned_total)(jnp.asarray(0.0)) == 3.0
        jax.effects_barrier()

    log_text = stream.getvalue()
    assert "callback value 1.0" in log_text
    assert "callback value 3.0" in log_text


def test_scanned_runtime_passes_callback_logger_to_components() -> None:
    logger_name = "VerCOR.test.scanned-runtime"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        log_level="INFO",
    )
    coupler.logger = setup_logger(level="INFO", name=logger_name)
    coupler.components = {"ATM": cast(Any, _LoggingRunComponent("ATM"))}
    coupler.run_sequence = ("ATM",)

    with capture_logger_output(logger_name) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert final_state.component_names == ("ATM",)
    assert "scanned ATM 4.0" in stream.getvalue()


def test_scanned_runtime_logs_host_equivalent_progress_messages() -> None:
    logger_name = "VerCOR.test.scanned-runtime-progress"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=2),
        log_level="INFO",
    )
    coupler.logger = setup_logger(level="INFO", name=logger_name)
    coupler.components = {
        "ATM": cast(Any, _LoggingRunComponent("ATM")),
        "OCN": cast(Any, _LoggingRunComponent("OCN")),
    }
    coupler.run_sequence = (
        "ATM",
        "OCN",
    )

    with capture_logger_output(logger_name) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert final_state.component_names == ("ATM", "OCN")
    log_text = stream.getvalue()
    assert (
        " ====== Step: 00000 ====== Date: 2000-01-01 00:00:00 ====== Δt: 0:01:00 "
        in log_text
    )
    assert (
        " ====== Step: 00001 ====== Date: 2000-01-01 00:01:00 ====== Δt: 0:01:00 "
        in log_text
    )
    assert log_text.count(" Run component: ATM") == 2
    assert log_text.count(" Run component: OCN") == 2


def test_scanned_runtime_suppresses_info_below_log_level() -> None:
    logger_name = "VerCOR.test.scanned-runtime-warning"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        log_level="WARNING",
    )
    coupler.logger = setup_logger(level="WARNING", name=logger_name)
    coupler.components = {"ATM": cast(Any, _LoggingRunComponent("ATM"))}
    coupler.run_sequence = ("ATM",)

    with capture_logger_output(logger_name, set_logger_level=False) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert final_state.component_names == ("ATM",)
    log_text = stream.getvalue()
    assert "scanned ATM" not in log_text
    assert " ====== Step:" not in log_text
    assert " Run component:" not in log_text


@pytest.mark.fast_always
def test_coupler_register_and_run_sequence_validation() -> None:
    coupler = make_coupler()
    atmosphere = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    coupler.register(cast(Any, atmosphere))

    with pytest.raises(CouplerError, match="already registered"):
        coupler.register(cast(Any, atmosphere))

    with pytest.raises(CouplerError, match="not registered in coupler"):
        coupler.set_components_run_sequence(
            (
                "ATM",
                "OCN",
            )
        )


@pytest.mark.parametrize(
    ("registered_names", "source", "destination"),
    [
        (["ATM"], "OCN", "ATM"),
        (["OCN"], "OCN", "ATM"),
    ],
)
def test_coupler_initialize_rejects_missing_exchange_endpoints(
    registered_names: list[str],
    source: str,
    destination: str,
) -> None:
    coupler = make_coupler()
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
    }
    for name in registered_names:
        coupler.register(cast(Any, components[name]))

    coupler.add_exchange(
        Exchange(
            source=source,
            destination=destination,
            field_names=["temperature"],
            regridder_factory=bilinear,
        )
    )

    with pytest.raises(CouplerError, match="not registered in coupler"):
        coupler.initialize()


def test_coupler_initialize_happy_path_builds_unique_regridders_and_supports_x64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    logger = _RecordingLogger()
    coupler.logger = cast(Any, logger)
    coupler.settings.enable_x64 = False

    lnd_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(
            name="OCN",
            grid=make_test_grid(
                name="ocn", binary_mask=np.asarray([[0.0, 1.0], [1.0, 0.0]])
            ),
        ),
        "LND": DummyComponent(
            name="LND",
            grid=make_test_grid(name="lnd", binary_mask=lnd_mask),
        ),
        "ICE": DummyComponent(name="ICE", grid=make_test_grid(name="ice")),
    }
    components["ATM"].data.update(
        {
            "downward_longwave_radiation_flux": np.full((2, 2), 1.0),
            "temperature_2m": np.full((2, 2), 2.0),
            "sensible_heat_flux": np.full((2, 2), 3.0),
        }
    )
    components["OCN"].data.update(
        {
            "temperature": np.full((2, 2), 4.0),
            "specific_humidity": np.full((2, 2), 5.0),
        }
    )
    components["LND"].data["soil_moisture"] = np.full((2, 2), 6.0)
    components["ICE"].data["ice_fraction"] = np.full((2, 2), 7.0)

    for component in components.values():
        coupler.register(cast(Any, component))

    exchanges = [
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["specific_humidity"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["downward_longwave_radiation_flux"],
            regridder_factory=conservative,
        ),
        Exchange(
            source="LND",
            destination="ATM",
            field_names=["soil_moisture"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="LND",
            field_names=["temperature_2m"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ICE",
            destination="ATM",
            field_names=["ice_fraction"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="ATM",
            destination="ICE",
            field_names=["sensible_heat_flux"],
            regridder_factory=bilinear,
        ),
    ]
    created_keys: list[tuple[str, str]] = []
    for exchange in exchanges:

        def fake_regridder_factory(
            source_grid: Any,
            destination_grid: Any,
            exchange_name: str = exchange.name,
        ) -> RecordingRegridder:
            _ = exchange_name
            created_keys.append((source_grid.name, destination_grid.name))
            return RecordingRegridder()

        monkeypatch.setattr(
            exchange,
            "regridder_factory",
            cast(Any, fake_regridder_factory),
        )
        coupler.add_exchange(exchange)

    def fake_create_surface_exchange_masks(
        *args: Any, **kwargs: Any
    ) -> SurfaceExchangeMasks:
        _ = args, kwargs
        return SurfaceExchangeMasks(
            ocn_fmask_on_atm_grid=np.full((2, 2), 0.4),
            lnd_fmask_on_atm_grid=np.full((2, 2), 0.6),
            lnd_bmask_on_atm_grid=lnd_mask,
        )

    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        fake_create_surface_exchange_masks,
    )
    jax_calls: list[tuple[str, bool]] = []
    monkeypatch.setitem(
        sys.modules,
        "jax",
        SimpleNamespace(
            config=SimpleNamespace(
                update=lambda key, value: jax_calls.append((key, value))
            )
        ),
    )

    coupler.initialize(enable_x64_computations=True)

    assert coupler.settings.enable_x64 is True
    assert jax_calls == [("jax_enable_x64", True)]
    assert len(created_keys) == 6
    topology_maps = coupler._runtime_resources.topology_maps
    assert len(topology_maps.regridders) == 6
    assert any("already exists" in message for message in logger.warning_messages)
    assert isinstance(
        topology_maps.binary_masks[("ATM", "OCN", "conservative")],
        jax.Array,
    )
    assert isinstance(
        topology_maps.fractional_masks[("ATM", "OCN", "conservative")],
        jax.Array,
    )
    assert coupler._runtime_resources.runtime_contracts[
        "ATM"
    ] == RuntimeComponentContract(
        imports=(
            "temperature",
            "specific_humidity",
            "soil_moisture",
            "ice_fraction",
        ),
        exports=(
            "downward_longwave_radiation_flux",
            "temperature_2m",
            "sensible_heat_flux",
        ),
    )
    assert_allclose_compact(
        topology_maps.fractional_masks[("OCN", "ATM", "bilinear")],
        np.full((2, 2), 0.4),
    )
    assert_allclose_compact(
        topology_maps.binary_masks[("LND", "ATM", "bilinear")],
        lnd_mask,
    )


@pytest.mark.fast_always
def test_build_exchange_topology_returns_explicit_patched_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: SurfaceExchangeMasks(
            ocn_fmask_on_atm_grid=np.full((2, 2), 0.4),
            lnd_fmask_on_atm_grid=np.full((2, 2), 0.6),
            lnd_bmask_on_atm_grid=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    state = build_exchange_topology(
        components=cast(Any, components),
        exchanges=(exchange,),
        settings=VercorSettings(),
        logger=cast(Any, _RecordingLogger()),
    )

    assert isinstance(state, ExchangeTopologyState)
    assert set(state.topology_maps.regridders) == {("OCN", "ATM", "bilinear")}
    assert_allclose_compact(
        state.topology_maps.fractional_masks[("OCN", "ATM", "bilinear")],
        np.full((2, 2), 0.4),
    )
    assert_allclose_compact(
        state.surface_masks.lnd_bmask_on_atm_grid,
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


@pytest.mark.fast_always
def test_build_exchange_topology_preserves_duplicate_regridder_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    logger = _RecordingLogger()
    exchanges = (
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            destination="ATM",
            field_names=["specific_humidity"],
            regridder_factory=bilinear,
        ),
    )
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: SurfaceExchangeMasks(
            ocn_fmask_on_atm_grid=np.ones((2, 2)),
            lnd_fmask_on_atm_grid=np.zeros((2, 2)),
            lnd_bmask_on_atm_grid=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    state = build_exchange_topology(
        components=cast(Any, components),
        exchanges=exchanges,
        settings=VercorSettings(),
        logger=cast(Any, logger),
    )

    assert len(state.topology_maps.regridders) == 1
    assert any("already exists" in message for message in logger.warning_messages)


@pytest.mark.fast_always
def test_build_exchange_topology_does_not_mutate_existing_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    exchange = Exchange(
        source="LND",
        destination="ATM",
        field_names=["soil_moisture"],
        regridder_factory=bilinear,
    )
    existing_regridders: dict[tuple[str, str, str], Any] = {}
    existing_binary_masks: dict[tuple[str, str, str], Any] = {}
    existing_fractional_masks: dict[tuple[str, str, str], Any] = {}
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: SurfaceExchangeMasks(
            ocn_fmask_on_atm_grid=np.zeros((2, 2)),
            lnd_fmask_on_atm_grid=np.full((2, 2), 0.75),
            lnd_bmask_on_atm_grid=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    state = build_exchange_topology(
        components=cast(Any, components),
        exchanges=(exchange,),
        topology_maps=RuntimeTopologyMaps(
            regridders=existing_regridders,
            binary_masks=existing_binary_masks,
            fractional_masks=existing_fractional_masks,
        ),
        settings=VercorSettings(),
        logger=cast(Any, _RecordingLogger()),
    )

    assert existing_regridders == {}
    assert existing_binary_masks == {}
    assert existing_fractional_masks == {}
    assert state.topology_maps.regridders is not existing_regridders
    assert_allclose_compact(
        state.topology_maps.binary_masks[("LND", "ATM", "bilinear")],
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_apply_surface_exchange_masks_updates_only_expected_bilinear_pairs() -> None:
    coupler = make_coupler()
    ocn_key = ("OCN", "ATM", "bilinear")
    lnd_key = ("LND", "ATM", "bilinear")
    other_key = ("OCN", "ATM", "conservative")

    binary_masks = {
        ocn_key: np.zeros((2, 2)),
        lnd_key: np.zeros((2, 2)),
        other_key: np.full((2, 2), 9.0),
    }
    fractional_masks = {
        ocn_key: np.zeros((2, 2)),
        lnd_key: np.zeros((2, 2)),
        other_key: np.full((2, 2), 7.0),
    }
    replace_runtime_topology_maps(
        coupler,
        regridders={},
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )
    coupler.ocn_fmask_on_atm_grid = np.full((2, 2), 0.25)
    coupler.lnd_bmask_on_atm_grid = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    coupler.lnd_fmask_on_atm_grid = np.full((2, 2), 0.75)

    topology_maps = coupler._runtime_resources.topology_maps
    apply_surface_exchange_masks(
        topology_maps,
        surface_masks=SurfaceExchangeMasks(
            ocn_fmask_on_atm_grid=coupler.ocn_fmask_on_atm_grid,
            lnd_fmask_on_atm_grid=coupler.lnd_fmask_on_atm_grid,
            lnd_bmask_on_atm_grid=coupler.lnd_bmask_on_atm_grid,
        ),
    )

    assert_allclose_compact(
        topology_maps.fractional_masks[ocn_key], np.full((2, 2), 0.25)
    )
    assert_allclose_compact(
        topology_maps.binary_masks[lnd_key],
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    assert_allclose_compact(
        topology_maps.fractional_masks[lnd_key], np.full((2, 2), 0.75)
    )
    assert_allclose_compact(topology_maps.binary_masks[other_key], np.full((2, 2), 9.0))
    assert_allclose_compact(
        topology_maps.fractional_masks[other_key], np.full((2, 2), 7.0)
    )


def test_validate_land_mask_consistency_rejects_shape_and_value_mismatches() -> None:
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "LND": DummyComponent(
                name="LND",
                grid=make_test_grid(name="lnd", binary_mask=np.ones((3, 2))),
            )
        },
    )
    coupler.lnd_bmask_on_atm_grid = np.ones((2, 2))

    with pytest.raises(CouplerError, match="does not match atmospheric grid shape"):
        validate_land_mask_consistency(
            coupler.components,
            SurfaceExchangeMasks(
                ocn_fmask_on_atm_grid=np.zeros((2, 2)),
                lnd_fmask_on_atm_grid=np.ones((2, 2)),
                lnd_bmask_on_atm_grid=coupler.lnd_bmask_on_atm_grid,
            ),
        )

    coupler.components["LND"] = cast(
        Any,
        DummyComponent(
            name="LND",
            grid=make_test_grid(
                name="lnd",
                binary_mask=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
            ),
        ),
    )
    coupler.lnd_bmask_on_atm_grid = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(CouplerError, match="mismatched points: 2"):
        validate_land_mask_consistency(
            coupler.components,
            SurfaceExchangeMasks(
                ocn_fmask_on_atm_grid=np.zeros((2, 2)),
                lnd_fmask_on_atm_grid=np.ones((2, 2)),
                lnd_bmask_on_atm_grid=coupler.lnd_bmask_on_atm_grid,
            ),
        )


def test_create_surface_exchange_masks_rejects_non_identical_land_and_atmosphere_grids() -> (
    None
):
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "ATM": DummyComponent(
                name="ATM",
                grid=make_test_grid(name="atm", latitude=np.asarray([0.0, 1.0])),
            ),
            "LND": DummyComponent(
                name="LND",
                grid=make_test_grid(name="lnd", latitude=np.asarray([0.0, 2.0])),
            ),
            "OCN": DummyComponent(
                name="OCN",
                grid=make_test_grid(
                    name="ocn", binary_mask=np.asarray([[1.0, 0.0], [0.0, 1.0]])
                ),
            ),
        },
    )

    with pytest.raises(CouplerError, match="must use identical horizontal grids"):
        create_surface_exchange_masks(coupler.components, logger=setup_logger())


def test_create_surface_exchange_masks_rejects_missing_ocean_binary_mask() -> None:
    coupler = make_coupler()
    coupler.components = cast(
        Any,
        {
            "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
            "LND": DummyComponent(name="LND", grid=make_test_grid(name="lnd")),
            "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
        },
    )

    with pytest.raises(ComponentError, match="has no binary mask defined"):
        create_surface_exchange_masks(coupler.components, logger=setup_logger())


def test_output_masks_for_component_returns_destination_exchange_masks() -> None:
    coupler = make_coupler()
    ocn_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    lnd_exchange = Exchange(
        source="LND",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.exchanges = [ocn_exchange, lnd_exchange]
    binary_masks = {
        ("OCN", "ATM", "bilinear"): np.zeros((2, 2)),
        ("LND", "ATM", "bilinear"): np.ones((2, 2)),
    }
    fractional_masks = {
        ("OCN", "ATM", "bilinear"): np.full((2, 2), 0.25),
        ("LND", "ATM", "bilinear"): np.full((2, 2), 0.75),
    }
    replace_runtime_topology_maps(
        coupler,
        regridders={},
        binary_masks=binary_masks,
        fractional_masks=fractional_masks,
    )

    assert not hasattr(coupler, "_output_masks_for_component")

    masks = output_masks_for_component(
        "ATM",
        coupler.exchanges,
        coupler._runtime_resources.topology_maps.binary_masks,
        coupler._runtime_resources.topology_maps.fractional_masks,
    )

    assert set(masks) == {
        "bmask_OCN_ATM_bilinear",
        "fmask_OCN_ATM_bilinear",
        "bmask_LND_ATM_bilinear",
        "fmask_LND_ATM_bilinear",
    }
    assert_allclose_compact(masks["fmask_LND_ATM_bilinear"], np.full((2, 2), 0.75))


def test_runtime_field_dispatch_handles_scalar_and_vector_paths() -> None:
    coupler = make_coupler()
    source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    source.data["temperature"] = jnp.full((2, 2), 5.0)
    source.data["u_velocity"] = np.full((2, 2), 1.0)
    source.data["v_velocity"] = np.full((2, 2), -1.0)

    scalar_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    vector_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=[("u_velocity", "v_velocity")],
        regridder_factory=conservative,
    )
    coupler.components = cast(Any, {"OCN": source, "ATM": destination})
    coupler.exchanges = [scalar_exchange, vector_exchange]
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "bilinear"): RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            ),
            ("OCN", "ATM", "conservative"): RecordingRegridder(
                vector_result=(
                    np.full((2, 2), 9.0),
                    np.full((2, 2), -9.0),
                )
            ),
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            ("OCN", "ATM", "bilinear"): np.asarray([[1.0, 0.5], [0.0, 1.0]]),
            ("OCN", "ATM", "conservative"): np.ones((2, 2)),
        },
    )

    runtime_state = _dispatch_runtime_fields(
        coupler,
        runtime_state_from_coupler_components(coupler, prefill_missing=True),
        "ATM",
    )
    destination_state = runtime_state.get_component_state("ATM")

    assert_allclose_compact(
        destination_state.incoming.get("temperature"),
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )
    assert isinstance(destination_state.incoming.get("temperature"), jax.Array)
    assert_allclose_compact(
        destination_state.incoming.get("u_velocity"),
        np.full((2, 2), 9.0),
    )
    assert_allclose_compact(
        destination_state.incoming.get("v_velocity"),
        np.full((2, 2), -9.0),
    )


def test_runtime_field_dispatch_accepts_mixed_numpy_and_jax_arrays() -> None:
    coupler = make_coupler()
    source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    source.data["temperature"] = np.full((2, 2), 5.0)

    exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.components = cast(Any, {"OCN": source, "ATM": destination})
    coupler.exchanges = [exchange]
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "bilinear"): RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            )
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            ("OCN", "ATM", "bilinear"): np.asarray([[1.0, 0.5], [0.0, 1.0]]),
        },
    )

    runtime_state = _dispatch_runtime_fields(
        coupler,
        runtime_state_from_coupler_components(coupler, prefill_missing=True),
        "ATM",
    )
    destination_state = runtime_state.get_component_state("ATM")

    assert isinstance(destination_state.incoming.get("temperature"), jax.Array)
    assert_allclose_compact(
        destination_state.incoming.get("temperature"),
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )


def test_runtime_field_dispatch_rejects_missing_scalar_and_vector_fields() -> None:
    coupler = make_coupler()

    scalar_source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    scalar_destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    scalar_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=["temperature"],
        regridder_factory=bilinear,
    )
    coupler.components = cast(Any, {"OCN": scalar_source, "ATM": scalar_destination})
    coupler.exchanges = [scalar_exchange]
    regridders = cast(
        Any,
        {("OCN", "ATM", "bilinear"): RecordingRegridder(scalar_result=np.ones((2, 2)))},
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={("OCN", "ATM", "bilinear"): np.ones((2, 2))},
    )

    with pytest.raises(ExchangerError, match="Field temperature not present"):
        _dispatch_runtime_fields(
            coupler,
            runtime_state_from_coupler_components(coupler, prefill_missing=False),
            scalar_destination.name,
        )

    vector_source = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    vector_destination = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    vector_source.data["u_velocity"] = np.ones((2, 2))
    vector_exchange = Exchange(
        source="OCN",
        destination="ATM",
        field_names=[("u_velocity", "v_velocity")],
        regridder_factory=conservative,
    )
    coupler.components = cast(Any, {"OCN": vector_source, "ATM": vector_destination})
    coupler.exchanges = [vector_exchange]
    regridders = cast(
        Any,
        {
            ("OCN", "ATM", "conservative"): RecordingRegridder(
                vector_result=(np.ones((2, 2)), np.ones((2, 2)))
            )
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={("OCN", "ATM", "conservative"): np.ones((2, 2))},
    )

    with pytest.raises(ExchangerError, match="Not all fields in vector"):
        _dispatch_runtime_fields(
            coupler,
            runtime_state_from_coupler_components(coupler, prefill_missing=False),
            vector_destination.name,
        )


def test_coupler_finalize_writes_runtime_outputs_for_all_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
    }
    coupler.components = cast(Any, components)
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    captured_runtime: dict[str, Any] = {}
    captured_snapshots: dict[str, Any] = {}

    def fake_write_outputs(**kwargs: Any) -> None:
        captured_runtime.update(kwargs)

    def fake_write_snapshots(**kwargs: Any) -> None:
        captured_snapshots.update(kwargs)

    monkeypatch.setattr(
        output_module, "write_coupler_runtime_outputs", fake_write_outputs
    )
    monkeypatch.setattr(
        output_module, "write_coupler_component_snapshots", fake_write_snapshots
    )

    coupler.finalize(state, Path("snapshot"))

    assert captured_runtime["final_state"] is state
    assert captured_runtime["components"] is coupler.components
    assert captured_runtime["exchanges"] is coupler.exchanges
    assert (
        captured_runtime["binary_masks"]
        is coupler._runtime_resources.topology_maps.binary_masks
    )
    assert (
        captured_runtime["fractional_masks"]
        is coupler._runtime_resources.topology_maps.fractional_masks
    )
    assert captured_runtime["output_file_mask"] == Path("snapshot")
    assert captured_runtime["logger"] is coupler.logger
    assert captured_snapshots["final_state"] is state
    assert captured_snapshots["components"] is coupler.components
    assert captured_snapshots["output_time"] == datetime(2000, 1, 1, 0, 0)
    assert captured_snapshots["logger"] is coupler.logger


def test_coupler_finalize_uses_last_executed_runtime_step_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=2))

    output_time = _snapshot_output_time_for_finalize(coupler, monkeypatch)

    assert output_time == datetime(2000, 1, 1, 0, 1)


def test_coupler_finalize_uses_clock_start_without_runtime_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=0))

    output_time = _snapshot_output_time_for_finalize(coupler, monkeypatch)

    assert output_time == coupler.clock.start


def test_output_boundary_builds_runtime_views_filenames_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    components = {
        "ATM": DummyComponent(name="ATM", grid=make_test_grid(name="atm")),
        "OCN": DummyComponent(name="OCN", grid=make_test_grid(name="ocn")),
    }
    coupler.components = cast(Any, components)
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    captured: list[tuple[str, Any, Path, dict[str, Any]]] = []

    def fake_write(
        view: Any,
        filename: Path,
        *,
        masks: dict[str, Any] | None = None,
    ) -> None:
        captured.append((view.name, view, filename, masks or {}))

    monkeypatch.setattr(
        output_runtime_module, "write_runtime_component_view_to_netcdf", fake_write
    )

    output_module.write_coupler_runtime_outputs(
        final_state=state,
        components=coupler.components,
        exchanges=coupler.exchanges,
        binary_masks=coupler._runtime_resources.topology_maps.binary_masks,
        fractional_masks=coupler._runtime_resources.topology_maps.fractional_masks,
        output_file_mask=Path("snapshot"),
        logger=coupler.logger,
    )

    assert [item[0] for item in captured] == ["ATM", "OCN"]
    assert captured[0][1].grid is components["ATM"].grid
    assert captured[0][2] == Path("atm_snapshot.nc")
    assert captured[1][2] == Path("ocn_snapshot.nc")


def test_output_boundary_calls_registered_snapshot_writers_and_skips_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    component = DummyComponent(name="ATM", grid=make_test_grid(name="snapshot-atm"))
    skipped = DummyComponent(name="OCN", grid=make_test_grid(name="snapshot-ocn"))
    coupler = make_coupler()
    coupler.components = cast(Any, {"ATM": component, "OCN": skipped})
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    calls: list[tuple[Any, Path, datetime | ModelDateTime, Any]] = []

    def write_snapshot(
        component_state: Any,
        output: Path,
        output_time: datetime | ModelDateTime,
        logger: Any,
    ) -> None:
        calls.append((component_state, output, output_time, logger))

    register_component_snapshot_writer(component, write_snapshot)

    output_module.write_coupler_component_snapshots(
        final_state=state,
        components=coupler.components,
        output_time=datetime(2000, 1, 1, 0, 1),
        logger=coupler.logger,
    )

    assert calls == [
        (
            state.get_component_state("ATM"),
            Path("atm.snapshot.nc"),
            datetime(2000, 1, 1, 0, 1),
            coupler.logger,
        )
    ]
    assert not (tmp_path / "ocn.snapshot.nc").exists()


def test_coupler_string_representations_include_registered_state() -> None:
    coupler = make_coupler()
    atmosphere = DummyComponent(name="ATM", grid=make_test_grid(name="atm"))
    ocean = DummyComponent(name="OCN", grid=make_test_grid(name="ocn"))
    coupler.register(cast(Any, atmosphere))
    coupler.register(cast(Any, ocean))
    coupler.add_exchange(
        Exchange(
            source="ATM",
            destination="OCN",
            field_names=["temperature"],
            regridder_factory=bilinear,
        )
    )
    coupler.set_components_run_sequence(
        (
            "ATM",
            "OCN",
        )
    )

    rendered = str(coupler)
    representation = repr(coupler)

    assert "Coupler:" in rendered
    assert "<DummyComponent>(ATM)" in rendered
    assert "ATM --(bilinear)--> OCN" in rendered
    assert "ATM, OCN" in rendered
    assert "run_sequence=ATM -> OCN" in representation


def test_coupler_run_happy_path_dispatches_and_steps_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    coupler.logger = cast(Any, _RecordingLogger())
    events: list[str] = []
    atmosphere = _HostRunComponent("ATM", events)
    ocean = _HostRunComponent("OCN", events)
    coupler.components = cast(Any, {"ATM": atmosphere, "OCN": ocean})
    coupler.run_sequence = (
        "ATM",
        "OCN",
    )

    def fake_dispatch(state: Any, component_name: str, *args: Any) -> Any:
        _ = args
        events.append(f"dispatch:{component_name}")
        return state

    def fake_receive(component_state: Any, *args: Any) -> Any:
        _ = component_state, args
        events.append("receive")
        return component_state

    def fake_send(
        component: Any,
        component_state: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        _ = component_state, args, kwargs
        events.append(f"send:{component.name}")
        return component_state

    monkeypatch.setattr(
        "vercor.runtime.driver.dispatch_component_exchanges", fake_dispatch
    )
    monkeypatch.setattr("vercor.runtime.driver.receive_runtime_fields", fake_receive)
    monkeypatch.setattr("vercor.runtime.driver.send_runtime_fields", fake_send)

    coupler.run()

    assert events == [
        "send:ATM",
        "send:OCN",
        "dispatch:ATM",
        "receive",
        "step_host:ATM:2000-01-01T00:00:00:60.0",
        "send:ATM",
        "dispatch:OCN",
        "receive",
        "step_host:OCN:2000-01-01T00:00:00:60.0",
        "send:OCN",
    ]


def test_host_runtime_components_use_explicit_host_contract() -> None:
    coupler = make_coupler()
    host_component = _HostRunComponent("ATM")
    coupler.components = cast(Any, {"ATM": host_component})
    coupler.run_sequence = ("ATM",)

    final_state = coupler.run()
    final_component = final_state.get_component_state("ATM")

    assert isinstance(host_component, HostRuntimeComponent)
    assert "host_time_seen" in final_component.data.field_names
    assert_allclose_compact(
        final_component.data.get("temperature"),
        np.full((2, 2), 61.0),
    )


def test_run_warns_when_host_backed_components_make_loop_nondifferentiable() -> None:
    logger = _RecordingLogger()
    coupler = make_coupler()
    coupler.logger = cast(Any, logger)
    coupler.components = cast(
        Any,
        {
            "ATM": _HostRunComponent("ATM"),
            "OCN": _HostRunComponent("OCN"),
        },
    )
    coupler.run_sequence = ("ATM", "OCN")

    coupler.run()

    assert logger.warning_messages == [
        "Coupled loop is not differentiable because host-backed component(s) "
        "require the Python runtime: ATM, OCN"
    ]


def test_run_rejects_state_donation_for_host_backed_components() -> None:
    coupler = make_coupler()
    coupler.components = cast(Any, {"ATM": _HostRunComponent("ATM")})
    coupler.run_sequence = ("ATM",)

    with pytest.raises(CouplerError, match="donation"):
        coupler.run(donate_state=True)


def test_host_and_scanned_run_use_runtime_component_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = make_coupler()
    assert not hasattr(coupler, "run_differentiable")
    assert not hasattr(coupler, "create_differentiable_state")
    assert not hasattr(coupler, "interpolate_and_dispatch_fields")
    assert not hasattr(coupler, "_dispatch_runtime_fields")
    assert not hasattr(coupler, "_commit_runtime_incoming_fields")

    events: list[str] = []

    def fake_runtime_step(
        state: Any,
        component_name: str,
        step_info: Any,
        *,
        allow_host_runtime: bool,
        **kwargs: Any,
    ) -> Any:
        _ = step_info, kwargs
        mode = "run" if allow_host_runtime else "scan"
        events.append(f"{mode}:{component_name}")
        return state

    monkeypatch.setattr(
        "vercor.runtime.runner.step_runtime_component", fake_runtime_step
    )

    coupler.components = cast(Any, {"ATM": _HostRunComponent("ATM")})
    coupler.run_sequence = ("ATM",)
    coupler.run()
    run_events = list(events)
    events.clear()

    timestamp = coupler.clock.start
    atmosphere = _RunComponent("ATM", [], timestamp)
    ocean = _RunComponent("OCN", [], timestamp)
    coupler.components = cast(Any, {"ATM": atmosphere, "OCN": ocean})
    coupler.run_sequence = (
        "ATM",
        "OCN",
    )
    run_scanned_coupler(coupler)

    assert run_events == ["run:ATM"]
    assert events == ["scan:ATM", "scan:OCN"]
