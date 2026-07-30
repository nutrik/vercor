from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import inspect
import logging
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import vercor._runtime.surface_masks as surface_masks_module
import vercor.coupler as coupler_module
import vercor.output._runtime as output_runtime_module
from tests._coverage_support import (
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
from vercor.clock import Clock
from vercor.components import ComponentSpec, DataComponent
from vercor.components.contexts import StepContext
from vercor.coupler import Coupler
from vercor.dtypes import DTypePolicy
from vercor.exceptions import ComponentError, CouplerError, ExchangeError
from vercor.exchanges import Exchange
from vercor.fields import vector
from vercor.jax_logging import (
    CANONICAL_LOG_DATE_FORMAT,
    CANONICAL_LOG_FORMAT,
    DEFAULT_LOGGER_NAME,
    JaxCallbackLogger,
    get_default_logger,
    normalize_log_level,
    setup_logger,
)
from vercor._regridders.bilinear import bilinear
from vercor._runtime.contracts import ExchangeContract
from vercor._runtime.exchange_dispatch import dispatch_component_exchanges
from vercor.output._runtime import output_masks_for_component
from vercor.output import OutputSpec, OutputTarget, SnapshotContext
from vercor.runtime import RuntimeOptions
from vercor._runtime.surface_masks import (
    create_surface_exchange_masks,
    validate_land_mask_consistency,
)
from vercor.state import ComponentState
from vercor._runtime.topology import build_exchange_topology
from vercor._runtime.topology_state import RuntimeTopologyMaps
from vercor.topology import SurfaceMaskPolicy


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

    def error(self, message: str, *args: Any) -> None:
        _ = message, args

    def setLevel(self, level: int | str) -> None:
        _ = level

    def isEnabledFor(self, level: int) -> bool:
        _ = level
        return True


class _RunComponent:
    def __init__(self, name: str, events: list[str], timestamp: datetime) -> None:
        _ = timestamp
        self.name = name
        self.grid = make_test_grid(name=name.lower())
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": np.ones((2, 2), dtype=float)},
        )
        self.events = events

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        _ = fields, payload
        time = context.time
        time_label = "none" if time is None else time.isoformat()
        self.events.append(f"step:{self.name}:{time_label}:{context.dt_seconds}")
        return {}


class _LoggingRunComponent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.grid = make_test_grid(name=name.lower())
        self.spec = ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": np.ones((2, 2), dtype=float)},
        )

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        _ = payload
        assert context.logger is not None
        context.logger.info(
            "scanned {} {}",
            self.name,
            jnp.sum(fields["temperature"]),
        )
        return {}


class _HostRunComponent:
    def __init__(self, name: str, events: list[str] | None = None) -> None:
        self.name = name
        self.grid = make_test_grid(name=name.lower())
        self.spec = ComponentSpec(
            outputs=("temperature", "host_time_seen"),
            initial_fields={
                "temperature": np.ones((2, 2), dtype=float),
                "host_time_seen": np.zeros((2, 2), dtype=float),
            },
            execution="host",
        )
        self.events = events

    def step(
        self,
        fields: Mapping[str, Any],
        context: StepContext,
        payload: Any | None = None,
    ) -> Mapping[str, Any]:
        _ = payload
        if self.events is not None:
            time = context.time
            time_label = "none" if time is None else time.isoformat()
            self.events.append(
                f"step_host:{self.name}:{time_label}:{context.dt_seconds}"
            )
        return {
            "temperature": fields["temperature"] + context.dt_seconds,
            "host_time_seen": np.ones((2, 2)),
        }


def make_coupler(
    *,
    components: Any = (),
    exchanges: Any = (),
    run_order: Any = (),
    runtime: RuntimeOptions | None = None,
    logger: Any = None,
) -> Coupler:
    return Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=components,
        exchanges=exchanges,
        run_order=run_order,
        runtime=runtime,
        logger=logger,
    )


def _snapshot_output_time_for_run_output(
    coupler: Coupler,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    coupler = Coupler(
        clock=coupler.clock,
        components=(
            cast(
                Any,
                DataComponent(
                    name="ATM",
                    grid=make_test_grid(name="atm"),
                    fields={"temperature": 0.0},
                ),
            ),
        ),
        run_order=("ATM",),
        runtime=coupler.runtime,
        constants=coupler.constants,
        logger=coupler.logger,
        log_level=coupler.log_level,
    )
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    captured_snapshots: dict[str, Any] = {}

    def fake_write_outputs(**kwargs: Any) -> None:
        _ = kwargs

    def fake_write_snapshots(**kwargs: Any) -> None:
        captured_snapshots.update(kwargs)

    monkeypatch.setattr(
        output_runtime_module, "write_coupler_runtime_outputs", fake_write_outputs
    )
    monkeypatch.setattr(
        output_runtime_module, "write_coupler_component_snapshots", fake_write_snapshots
    )

    coupler.run(
        state,
        output=OutputTarget(
            ".",
            write_period=False,
        ),
    )

    return captured_snapshots["output_time"]


def _topology_components() -> dict[str, DataComponent]:
    lnd_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    return {
        "ATM": DataComponent(
            name="ATM",
            grid=make_test_grid(name="atm"),
            fields={"temperature": 0.0},
        ),
        "OCN": DataComponent(
            name="OCN",
            grid=make_test_grid(
                name="ocn",
                binary_mask=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            ),
            fields={"temperature": 0.0},
        ),
        "LND": DataComponent(
            name="LND",
            grid=make_test_grid(name="lnd", binary_mask=lnd_mask),
            fields={"temperature": 0.0},
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
        coupler._ensure_prepared().topology_maps.regridders,
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
    coupler = make_coupler(
        components=tuple(
            DataComponent(
                name=component_name,
                grid=make_test_grid(name=component_name.lower()),
                fields={"temperature": 0.0},
            )
            for component_name in ("ATM", "OCN", "LND")
        ),
        run_order=("ATM", "OCN", "LND"),
    )

    runtime_state = coupler.initial_state(prefill_missing=True)

    all_views = runtime_state.components(coupler.run_order)
    selected_views = runtime_state.components(("LND", "ATM"))

    assert tuple(all_views) == ("ATM", "OCN", "LND")
    assert tuple(selected_views) == ("LND", "ATM")
    assert all_views["ATM"].name == "ATM"
    assert selected_views["LND"].grid is not None
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


def test_normalize_log_level_accepts_trace_as_level_five() -> None:
    assert normalize_log_level("trace") == 5
    coupler = Coupler(
        Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        log_level="trace",
    )

    assert coupler.logger.isEnabledFor(5)
    assert coupler.logger.isEnabledFor(logging.DEBUG)


@pytest.mark.fast_always
def test_coupler_module_does_not_reexport_logger_setup_helper() -> None:
    assert not hasattr(coupler_module, "setup_logger")


def test_coupler_wraps_injected_python_logger_for_scanned_runtime() -> None:
    logger_name = "VerCOR.test.injected"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(cast(Any, _LoggingRunComponent("ATM")),),
        run_order=("ATM",),
        logger=logging.getLogger(logger_name),
        log_level="INFO",
    )
    coupler._ensure_prepared()

    with capture_logger_output(logger_name, set_logger_level=False) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert tuple(final_state.components()) == ("ATM",)
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
        components=(cast(Any, _LoggingRunComponent("ATM")),),
        run_order=("ATM",),
        logger=setup_logger(level="INFO", name=logger_name),
        log_level="INFO",
    )
    coupler._ensure_prepared()

    with capture_logger_output(logger_name) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert tuple(final_state.components()) == ("ATM",)
    assert "scanned ATM 4.0" in stream.getvalue()


def test_scanned_runtime_logs_host_equivalent_progress_messages() -> None:
    logger_name = "VerCOR.test.scanned-runtime-progress"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=2),
        components=(
            cast(Any, _LoggingRunComponent("ATM")),
            cast(Any, _LoggingRunComponent("OCN")),
        ),
        run_order=(
            "ATM",
            "OCN",
        ),
        logger=setup_logger(level="INFO", name=logger_name),
        log_level="INFO",
    )
    coupler._ensure_prepared()

    with capture_logger_output(logger_name) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert tuple(final_state.components()) == ("ATM", "OCN")
    log_text = stream.getvalue()
    assert (
        "====== Step: 00000 ====== Date: 2000-01-01 00:00:00 ====== Δt: 0:01:00 "
        in log_text
    )
    assert (
        "====== Step: 00001 ====== Date: 2000-01-01 00:01:00 ====== Δt: 0:01:00 "
        in log_text
    )
    assert log_text.count("Run component: ATM") == 2
    assert log_text.count("Run component: OCN") == 2


def test_scanned_runtime_suppresses_info_below_log_level() -> None:
    logger_name = "VerCOR.test.scanned-runtime-warning"
    coupler = Coupler(
        clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=1),
        components=(cast(Any, _LoggingRunComponent("ATM")),),
        run_order=("ATM",),
        logger=setup_logger(level="WARNING", name=logger_name),
        log_level="WARNING",
    )
    coupler._ensure_prepared()

    with capture_logger_output(logger_name, set_logger_level=False) as stream:
        final_state = jax.jit(lambda: run_scanned_coupler(coupler))()
        jax.effects_barrier()

    assert tuple(final_state.components()) == ("ATM",)
    log_text = stream.getvalue()
    assert "scanned ATM" not in log_text
    assert "====== Step:" not in log_text
    assert "Run component:" not in log_text


@pytest.mark.fast_always
def test_coupler_constructor_validates_duplicate_components_and_run_order() -> None:
    atmosphere = DataComponent(
        name="ATM",
        grid=make_test_grid(name="atm"),
        fields={"temperature": 0.0},
    )

    with pytest.raises(CouplerError, match="Duplicate component name.*ATM"):
        make_coupler(components=(cast(Any, atmosphere), cast(Any, atmosphere)))

    with pytest.raises(CouplerError, match="Unknown run-order component.*OCN"):
        make_coupler(
            components=(cast(Any, atmosphere),),
            run_order=("ATM", "OCN"),
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
    components = {
        "ATM": DataComponent(
            name="ATM",
            grid=make_test_grid(name="atm"),
            fields={"temperature": 0.0},
        ),
        "OCN": DataComponent(
            name="OCN",
            grid=make_test_grid(name="ocn"),
            fields={"temperature": 0.0},
        ),
    }
    with pytest.raises(CouplerError, match="unknown .* component"):
        make_coupler(
            components=tuple(cast(Any, components[name]) for name in registered_names),
            exchanges=(
                Exchange(
                    source=source,
                    target=destination,
                    fields=["temperature"],
                    regridder_factory=bilinear,
                ),
            ),
        )


def test_coupler_initialize_happy_path_builds_unique_regridders_and_supports_x64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = _RecordingLogger()
    lnd_mask = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    components = {
        "ATM": DataComponent(
            name="ATM",
            grid=make_test_grid(name="atm"),
            fields={
                "temperature": 0.0,
                "downward_longwave_radiation_flux": np.full((2, 2), 1.0),
                "temperature_2m": np.full((2, 2), 2.0),
                "sensible_heat_flux": np.full((2, 2), 3.0),
            },
            spec=ComponentSpec(
                inputs=(
                    "temperature",
                    "specific_humidity",
                    "soil_moisture",
                    "ice_fraction",
                ),
                outputs=(
                    "downward_longwave_radiation_flux",
                    "temperature_2m",
                    "sensible_heat_flux",
                ),
            ),
        ),
        "OCN": DataComponent(
            name="OCN",
            grid=make_test_grid(
                name="ocn", binary_mask=np.asarray([[0.0, 1.0], [1.0, 0.0]])
            ),
            fields={
                "temperature": np.full((2, 2), 4.0),
                "specific_humidity": np.full((2, 2), 5.0),
            },
            spec=ComponentSpec(
                inputs=("downward_longwave_radiation_flux",),
                outputs=("temperature", "specific_humidity"),
            ),
        ),
        "LND": DataComponent(
            name="LND",
            grid=make_test_grid(name="lnd", binary_mask=lnd_mask),
            fields={
                "temperature": 0.0,
                "soil_moisture": np.full((2, 2), 6.0),
            },
            spec=ComponentSpec(
                inputs=("temperature_2m",),
                outputs=("soil_moisture",),
            ),
        ),
        "ICE": DataComponent(
            name="ICE",
            grid=make_test_grid(name="ice"),
            fields={
                "temperature": 0.0,
                "ice_fraction": np.full((2, 2), 7.0),
            },
            spec=ComponentSpec(
                inputs=("sensible_heat_flux",),
                outputs=("ice_fraction",),
            ),
        ),
    }

    created_keys: list[tuple[str, str]] = []

    def recording_regridder_factory(
        interpolation_type: str,
    ) -> Any:
        def factory(source_grid: Any, target_grid: Any) -> RecordingRegridder:
            created_keys.append((source_grid.name, target_grid.name))
            return RecordingRegridder(
                source_grid=source_grid,
                target_grid=target_grid,
            )

        factory.__name__ = interpolation_type
        return cast(Any, factory)

    bilinear_recording = recording_regridder_factory("bilinear")
    conservative_recording = recording_regridder_factory("conservative")

    exchanges = [
        Exchange(
            source="OCN",
            target="ATM",
            fields=["temperature", "specific_humidity"],
            regridder_factory=bilinear_recording,
        ),
        Exchange(
            source="ATM",
            target="OCN",
            fields=["downward_longwave_radiation_flux"],
            regridder_factory=conservative_recording,
        ),
        Exchange(
            source="LND",
            target="ATM",
            fields=["soil_moisture"],
            regridder_factory=bilinear_recording,
        ),
        Exchange(
            source="ATM",
            target="LND",
            fields=["temperature_2m"],
            regridder_factory=bilinear_recording,
        ),
        Exchange(
            source="ICE",
            target="ATM",
            fields=["ice_fraction"],
            regridder_factory=bilinear_recording,
        ),
        Exchange(
            source="ATM",
            target="ICE",
            fields=["sensible_heat_flux"],
            regridder_factory=bilinear_recording,
        ),
    ]
    coupler = make_coupler(
        components=tuple(cast(Any, component) for component in components.values()),
        exchanges=exchanges,
        runtime=RuntimeOptions(
            dtype=DTypePolicy(enable_x64=True),
            topology=SurfaceMaskPolicy(),
        ),
        logger=cast(Any, logger),
    )

    def fake_create_surface_exchange_masks(*args: Any, **kwargs: Any) -> Any:
        _ = args, kwargs
        return (
            np.full((2, 2), 0.4),
            np.full((2, 2), 0.6),
            lnd_mask,
        )

    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        fake_create_surface_exchange_masks,
    )
    coupler._initialize_runtime()

    assert coupler.runtime.dtype.enable_x64 is True
    assert len(created_keys) == 6
    assert coupler._prepared is not None
    topology_maps = coupler._prepared.topology_maps
    assert len(topology_maps.regridders) == 6
    assert isinstance(
        topology_maps.binary_masks["ATM->OCN"],
        jax.Array,
    )
    assert isinstance(
        topology_maps.fractional_masks["ATM->OCN"],
        jax.Array,
    )
    assert coupler._prepared.contracts["ATM"] == ExchangeContract(
        receives=(
            "temperature",
            "specific_humidity",
            "soil_moisture",
            "ice_fraction",
        ),
        sends=(
            "downward_longwave_radiation_flux",
            "temperature_2m",
            "sensible_heat_flux",
        ),
    )
    assert_allclose_compact(
        topology_maps.fractional_masks["OCN->ATM"],
        np.full((2, 2), 0.4),
    )
    assert_allclose_compact(
        topology_maps.binary_masks["LND->ATM"],
        lnd_mask,
    )
    assert not hasattr(coupler, "ocn_fmask_on_atm_grid")
    assert not hasattr(coupler, "lnd_fmask_on_atm_grid")
    assert not hasattr(coupler, "lnd_bmask_on_atm_grid")


@pytest.mark.fast_always
def test_build_exchange_topology_returns_runtime_topology_maps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.4),
            np.full((2, 2), 0.6),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    topology_maps = build_exchange_topology(
        components=cast(Any, components),
        exchanges=(exchange,),
        dtype=DTypePolicy(),
        topology_policy=SurfaceMaskPolicy(),
        logger=cast(Any, _RecordingLogger()),
    )

    assert isinstance(topology_maps, RuntimeTopologyMaps)
    assert set(topology_maps.regridders) == {"OCN->ATM"}
    assert_allclose_compact(
        topology_maps.fractional_masks["OCN->ATM"],
        np.full((2, 2), 0.4),
    )


@pytest.mark.fast_always
def test_build_exchange_topology_rejects_duplicate_topology_keys() -> None:
    components = _topology_components()
    logger = _RecordingLogger()
    exchanges = (
        Exchange(
            source="OCN",
            target="ATM",
            fields=["temperature"],
            regridder_factory=bilinear,
        ),
        Exchange(
            source="OCN",
            target="ATM",
            fields=["specific_humidity"],
            regridder_factory=bilinear,
        ),
    )
    with pytest.raises(
        CouplerError,
        match="Duplicate exchange route ID 'OCN->ATM'",
    ):
        build_exchange_topology(
            components=cast(Any, components),
            exchanges=exchanges,
            dtype=DTypePolicy(),
            logger=cast(Any, logger),
        )


@pytest.mark.fast_always
def test_surface_mask_policy_uses_one_build_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    events: list[str] = []
    original_build = SurfaceMaskPolicy.build

    def recording_build(self: SurfaceMaskPolicy, context: Any) -> Any:
        events.append("build")
        return original_build(self, context)

    monkeypatch.setattr(SurfaceMaskPolicy, "build", recording_build)
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: (
            np.full((2, 2), 0.4),
            np.full((2, 2), 0.6),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    topology_maps = build_exchange_topology(
        components=cast(Any, components),
        exchanges=(exchange,),
        dtype=DTypePolicy(),
        topology_policy=SurfaceMaskPolicy(),
        logger=cast(Any, _RecordingLogger()),
    )

    assert events == ["build"]
    assert_allclose_compact(
        topology_maps.fractional_masks["OCN->ATM"],
        np.full((2, 2), 0.4),
    )


@pytest.mark.fast_always
def test_build_exchange_topology_does_not_mutate_existing_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _topology_components()
    exchange = Exchange(
        source="LND",
        target="ATM",
        fields=["soil_moisture"],
        regridder_factory=bilinear,
    )
    seeded_route = "seeded"
    seeded_regridder = object()
    seeded_binary_mask = np.ones((2, 2))
    seeded_fractional_mask = np.full((2, 2), 0.25)
    existing_regridders: dict[str, Any] = {seeded_route: seeded_regridder}
    existing_binary_masks: dict[str, Any] = {seeded_route: seeded_binary_mask}
    existing_fractional_masks: dict[str, Any] = {seeded_route: seeded_fractional_mask}
    monkeypatch.setattr(
        surface_masks_module,
        "create_surface_exchange_masks",
        lambda *args, **kwargs: (
            np.zeros((2, 2)),
            np.full((2, 2), 0.75),
            np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        ),
    )

    topology_maps = build_exchange_topology(
        components=cast(Any, components),
        exchanges=(exchange,),
        topology_maps=RuntimeTopologyMaps(
            regridders=existing_regridders,
            binary_masks=existing_binary_masks,
            fractional_masks=existing_fractional_masks,
        ),
        dtype=DTypePolicy(),
        topology_policy=SurfaceMaskPolicy(),
        logger=cast(Any, _RecordingLogger()),
    )

    assert existing_regridders == {seeded_route: seeded_regridder}
    assert existing_binary_masks[seeded_route] is seeded_binary_mask
    assert existing_fractional_masks[seeded_route] is seeded_fractional_mask
    assert topology_maps.regridders is not existing_regridders
    assert topology_maps.regridders[seeded_route] is seeded_regridder
    assert topology_maps.binary_masks[seeded_route] is seeded_binary_mask
    assert topology_maps.fractional_masks[seeded_route] is seeded_fractional_mask
    assert_allclose_compact(
        topology_maps.binary_masks["LND->ATM"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )


def test_validate_land_mask_consistency_rejects_shape_and_value_mismatches() -> None:
    shape_coupler = make_coupler(
        components=(
            cast(
                Any,
                DataComponent(
                    name="LND",
                    grid=make_test_grid(name="lnd", binary_mask=np.ones((3, 2))),
                    fields={"temperature": 0.0},
                ),
            ),
        )
    )
    shape_lnd_bmask_on_atm_grid = np.ones((2, 2))

    with pytest.raises(CouplerError, match="does not match atmospheric grid shape"):
        validate_land_mask_consistency(
            shape_coupler._runtime_components,
            shape_lnd_bmask_on_atm_grid,
            policy=SurfaceMaskPolicy(),
        )

    value_coupler = make_coupler(
        components=(
            cast(
                Any,
                DataComponent(
                    name="LND",
                    grid=make_test_grid(
                        name="lnd",
                        binary_mask=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
                    ),
                    fields={"temperature": 0.0},
                ),
            ),
        )
    )
    value_lnd_bmask_on_atm_grid = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(CouplerError, match="mismatched points: 2"):
        validate_land_mask_consistency(
            value_coupler._runtime_components,
            value_lnd_bmask_on_atm_grid,
            policy=SurfaceMaskPolicy(),
        )


def test_create_surface_exchange_masks_rejects_non_identical_land_and_atmosphere_grids() -> (
    None
):
    coupler = make_coupler(
        components=cast(
            Any,
            (
                DataComponent(
                    name="ATM",
                    grid=make_test_grid(name="atm", latitude=np.asarray([0.0, 1.0])),
                    fields={"temperature": 0.0},
                ),
                DataComponent(
                    name="LND",
                    grid=make_test_grid(name="lnd", latitude=np.asarray([0.0, 2.0])),
                    fields={"temperature": 0.0},
                ),
                DataComponent(
                    name="OCN",
                    grid=make_test_grid(
                        name="ocn",
                        binary_mask=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
                    ),
                    fields={"temperature": 0.0},
                ),
            ),
        )
    )

    with pytest.raises(CouplerError, match="must use identical horizontal grids"):
        create_surface_exchange_masks(
            coupler._runtime_components,
            policy=SurfaceMaskPolicy(),
            logger=setup_logger(),
        )


def test_create_surface_exchange_masks_rejects_missing_ocean_binary_mask() -> None:
    coupler = make_coupler(
        components=(
            cast(
                Any,
                DataComponent(
                    name="ATM",
                    grid=make_test_grid(name="atm"),
                    fields={"temperature": 0.0},
                ),
            ),
            cast(
                Any,
                DataComponent(
                    name="LND",
                    grid=make_test_grid(name="lnd"),
                    fields={"temperature": 0.0},
                ),
            ),
            cast(
                Any,
                DataComponent(
                    name="OCN",
                    grid=make_test_grid(name="ocn"),
                    fields={"temperature": 0.0},
                ),
            ),
        )
    )

    with pytest.raises(ComponentError, match="has no binary mask defined"):
        create_surface_exchange_masks(
            coupler._runtime_components,
            policy=SurfaceMaskPolicy(),
            logger=setup_logger(),
        )


def test_output_masks_for_component_returns_destination_exchange_masks() -> None:
    ocn_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    lnd_exchange = Exchange(
        source="LND",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    exchanges = (ocn_exchange, lnd_exchange)
    binary_masks = {
        "OCN->ATM": np.zeros((2, 2)),
        "LND->ATM": np.ones((2, 2)),
    }
    fractional_masks = {
        "OCN->ATM": np.full((2, 2), 0.25),
        "LND->ATM": np.full((2, 2), 0.75),
    }
    assert not hasattr(Coupler, "_output_masks_for_component")

    masks = output_masks_for_component(
        "ATM",
        exchanges,
        binary_masks,
        fractional_masks,
    )

    assert set(masks) == {
        "bmask_OCN_ATM",
        "fmask_OCN_ATM",
        "bmask_LND_ATM",
        "fmask_LND_ATM",
    }
    assert_allclose_compact(masks["fmask_LND_ATM"], np.full((2, 2), 0.75))


def test_output_mask_names_remain_unique_after_route_token_sanitizing() -> None:
    exchanges = (
        Exchange("OCN", "ATM", ("temperature",), route_id="a-b"),
        Exchange("LND", "ATM", ("temperature",), route_id="a_b"),
    )
    binary_masks = {
        "a-b": np.zeros((2, 2)),
        "a_b": np.ones((2, 2)),
    }
    fractional_masks = {
        "a-b": np.full((2, 2), 0.25),
        "a_b": np.full((2, 2), 0.75),
    }

    masks = output_masks_for_component(
        "ATM",
        exchanges,
        binary_masks,
        fractional_masks,
    )

    assert len(masks) == 4
    assert sorted(float(np.mean(value)) for value in masks.values()) == [
        0.0,
        0.25,
        0.75,
        1.0,
    ]


def test_runtime_field_dispatch_handles_scalar_and_vector_paths() -> None:
    source = DataComponent(
        "OCN",
        make_test_grid(name="ocn"),
        fields={
            "temperature": jnp.full((2, 2), 5.0),
            "u_velocity": np.ones((2, 2)),
            "v_velocity": np.ones((2, 2)),
        },
    )
    destination = DataComponent(
        "ATM",
        make_test_grid(name="atm"),
        spec=ComponentSpec(inputs=("temperature", "u_velocity", "v_velocity")),
    )

    scalar_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        route_id="ocn-atm-scalar",
        regridder_factory=bilinear,
    )
    vector_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=[vector("u_velocity", "v_velocity")],
        route_id="ocn-atm-vector",
        regridder_factory=bilinear,
    )
    coupler = make_coupler(
        components=(cast(Any, source), cast(Any, destination)),
        exchanges=(scalar_exchange, vector_exchange),
    )
    regridders = cast(
        Any,
        {
            "ocn-atm-scalar": RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            ),
            "ocn-atm-vector": RecordingRegridder(
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
            "ocn-atm-scalar": np.asarray([[1.0, 0.5], [0.0, 1.0]]),
            "ocn-atm-vector": np.ones((2, 2)),
        },
    )

    runtime_state = _dispatch_runtime_fields(
        coupler,
        runtime_state_from_coupler_components(coupler, prefill_missing=True),
        "ATM",
    )
    destination_state = runtime_state._component_state("ATM")

    assert_allclose_compact(
        destination_state.received.get("temperature"),
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )
    assert isinstance(destination_state.received.get("temperature"), jax.Array)
    assert_allclose_compact(
        destination_state.received.get("u_velocity"),
        np.full((2, 2), 9.0),
    )
    assert_allclose_compact(
        destination_state.received.get("v_velocity"),
        np.full((2, 2), -9.0),
    )


def test_runtime_field_dispatch_accepts_mixed_numpy_and_jax_arrays() -> None:
    source = DataComponent(
        "OCN",
        make_test_grid(name="ocn"),
        fields={"temperature": np.full((2, 2), 5.0)},
    )
    destination = DataComponent(
        "ATM",
        make_test_grid(name="atm"),
        spec=ComponentSpec(inputs=("temperature",)),
    )

    exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    coupler = make_coupler(
        components=(cast(Any, source), cast(Any, destination)),
        exchanges=(exchange,),
    )
    regridders = cast(
        Any,
        {
            "OCN->ATM": RecordingRegridder(
                scalar_result=jnp.asarray([[2.0, 4.0], [6.0, 8.0]])
            )
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={
            "OCN->ATM": np.asarray([[1.0, 0.5], [0.0, 1.0]]),
        },
    )

    runtime_state = _dispatch_runtime_fields(
        coupler,
        runtime_state_from_coupler_components(coupler, prefill_missing=True),
        "ATM",
    )
    destination_state = runtime_state._component_state("ATM")

    assert isinstance(destination_state.received.get("temperature"), jax.Array)
    assert_allclose_compact(
        destination_state.received.get("temperature"),
        np.asarray([[2.0, 2.0], [0.0, 8.0]]),
    )


def test_runtime_field_dispatch_rejects_missing_scalar_and_vector_fields() -> None:
    scalar_source = DataComponent(
        "OCN",
        make_test_grid(name="ocn"),
        spec=ComponentSpec(outputs=("temperature",)),
    )
    scalar_destination = DataComponent(
        "ATM",
        make_test_grid(name="atm"),
        spec=ComponentSpec(inputs=("temperature",)),
    )
    scalar_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=["temperature"],
        regridder_factory=bilinear,
    )
    coupler = make_coupler(
        components=(cast(Any, scalar_source), cast(Any, scalar_destination)),
        exchanges=(scalar_exchange,),
    )
    regridders = cast(
        Any,
        {"OCN->ATM": RecordingRegridder(scalar_result=np.ones((2, 2)))},
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={"OCN->ATM": np.ones((2, 2))},
    )

    with pytest.raises(ExchangeError, match="Field temperature not present"):
        _dispatch_runtime_fields(
            coupler,
            runtime_state_from_coupler_components(coupler, prefill_missing=False),
            scalar_destination.name,
        )

    vector_source = DataComponent(
        "OCN",
        make_test_grid(name="ocn"),
        fields={"u_velocity": np.ones((2, 2))},
        spec=ComponentSpec(outputs=("u_velocity", "v_velocity")),
    )
    vector_destination = DataComponent(
        "ATM",
        make_test_grid(name="atm"),
        spec=ComponentSpec(inputs=("u_velocity", "v_velocity")),
    )
    vector_exchange = Exchange(
        source="OCN",
        target="ATM",
        fields=[vector("u_velocity", "v_velocity")],
        regridder_factory=bilinear,
    )
    coupler = make_coupler(
        components=(cast(Any, vector_source), cast(Any, vector_destination)),
        exchanges=(vector_exchange,),
    )
    regridders = cast(
        Any,
        {
            "OCN->ATM": RecordingRegridder(
                vector_result=(np.ones((2, 2)), np.ones((2, 2)))
            )
        },
    )
    replace_runtime_topology_maps(
        coupler,
        regridders=regridders,
        fractional_masks={"OCN->ATM": np.ones((2, 2))},
    )

    with pytest.raises(ExchangeError, match="Not all fields in vector"):
        _dispatch_runtime_fields(
            coupler,
            runtime_state_from_coupler_components(coupler, prefill_missing=False),
            vector_destination.name,
        )


def test_coupler_run_writes_enabled_outputs_for_all_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = (
        DataComponent(
            name="ATM",
            grid=make_test_grid(name="atm"),
            fields={"temperature": 0.0},
        ),
        DataComponent(
            name="OCN",
            grid=make_test_grid(name="ocn"),
            fields={"temperature": 0.0},
        ),
    )
    coupler = make_coupler(
        components=cast(Any, components),
        run_order=tuple(component.name for component in components),
    )
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    captured_runtime: dict[str, Any] = {}
    captured_snapshots: dict[str, Any] = {}

    def fake_write_outputs(**kwargs: Any) -> None:
        captured_runtime.update(kwargs)

    def fake_write_snapshots(**kwargs: Any) -> None:
        captured_snapshots.update(kwargs)

    monkeypatch.setattr(
        output_runtime_module, "write_coupler_runtime_outputs", fake_write_outputs
    )
    monkeypatch.setattr(
        output_runtime_module, "write_coupler_component_snapshots", fake_write_snapshots
    )

    final_state = coupler.run(
        state,
        output=OutputTarget(
            Path("snapshot"),
            write_period=False,
        ),
    )

    assert captured_runtime["final_state"] is final_state
    captured_components = captured_runtime["components"]
    prepared_components = coupler._ensure_prepared().components
    assert tuple(captured_components) == tuple(prepared_components)
    for name, component in prepared_components.items():
        assert captured_components[name] is component
        assert component._component is coupler._runtime_components[name].component
    assert captured_runtime["exchanges"] is coupler.exchanges
    assert (
        captured_runtime["binary_masks"]
        is coupler._ensure_prepared().topology_maps.binary_masks
    )
    assert (
        captured_runtime["fractional_masks"]
        is coupler._ensure_prepared().topology_maps.fractional_masks
    )
    assert captured_runtime["output_dir"] == Path("snapshot")
    assert captured_runtime["logger"] is coupler.logger
    assert captured_snapshots["final_state"] is final_state
    snapshot_components = captured_snapshots["components"]
    assert tuple(snapshot_components) == tuple(prepared_components)
    for name, component in prepared_components.items():
        assert snapshot_components[name] is component
    assert captured_snapshots["output_time"] == datetime(2000, 1, 1, 0, 1)
    assert captured_snapshots["output_dir"] == Path("snapshot")
    assert captured_snapshots["logger"] is coupler.logger


def test_coupler_run_output_uses_final_runtime_state_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=2))

    output_time = _snapshot_output_time_for_run_output(coupler, monkeypatch)

    assert output_time == datetime(2000, 1, 1, 0, 2)


def test_coupler_run_output_uses_clock_start_without_runtime_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coupler = Coupler(clock=Clock(start=datetime(2000, 1, 1), dt_seconds=60.0, steps=0))

    output_time = _snapshot_output_time_for_run_output(coupler, monkeypatch)

    assert output_time == coupler.clock.start


def test_output_boundary_builds_runtime_views_filenames_and_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = {
        "ATM": DataComponent(
            name="ATM",
            grid=make_test_grid(name="atm"),
            fields={"temperature": 0.0},
        ),
        "OCN": DataComponent(
            name="OCN",
            grid=make_test_grid(name="ocn"),
            fields={"temperature": 0.0},
        ),
    }
    coupler = make_coupler(components=cast(Any, tuple(components.values())))
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    prepared = coupler._ensure_prepared()
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

    output_runtime_module.write_coupler_runtime_outputs(
        final_state=state,
        components=prepared.components,
        exchanges=coupler.exchanges,
        binary_masks=prepared.topology_maps.binary_masks,
        fractional_masks=prepared.topology_maps.fractional_masks,
        logger=coupler.logger,
    )

    assert [item[0] for item in captured] == ["ATM", "OCN"]
    assert captured[0][1].grid is prepared.components["ATM"].grid
    assert captured[0][2] == Path("atm.runtime_fields.nc")
    assert captured[1][2] == Path("ocn.runtime_fields.nc")


def test_output_boundary_calls_registered_snapshot_writers_and_skips_others(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    calls: list[SnapshotContext] = []

    def write_snapshot(context: SnapshotContext) -> None:
        calls.append(context)

    component = DataComponent(
        name="ATM",
        grid=make_test_grid(name="snapshot-atm"),
        fields={"temperature": 0.0},
        spec=ComponentSpec(output=OutputSpec(snapshot_writer=write_snapshot)),
    )
    skipped = DataComponent(
        name="OCN",
        grid=make_test_grid(name="snapshot-ocn"),
        fields={"temperature": 0.0},
    )
    coupler = make_coupler(components=(cast(Any, component), cast(Any, skipped)))
    state = runtime_state_from_coupler_components(coupler, prefill_missing=True)
    prepared_components = coupler._ensure_prepared().components

    output_runtime_module.write_coupler_component_snapshots(
        final_state=state,
        components=prepared_components,
        output_time=datetime(2000, 1, 1, 0, 1),
        logger=coupler.logger,
    )

    assert len(calls) == 1
    assert calls[0].component.name == component.name
    assert calls[0].component.spec is component.spec
    assert isinstance(calls[0].state, ComponentState)
    assert calls[0].state.name == "ATM"
    assert calls[0].output_path == Path("atm.snapshot.nc")
    assert calls[0].time == datetime(2000, 1, 1, 0, 1)
    assert calls[0].logger is coupler.logger
    assert not (tmp_path / "ocn.snapshot.nc").exists()


def test_coupler_string_representations_include_registered_state() -> None:
    atmosphere = DataComponent(
        name="ATM",
        grid=make_test_grid(name="atm"),
        fields={"temperature": 0.0},
    )
    ocean = DataComponent(
        name="OCN",
        grid=make_test_grid(name="ocn"),
        fields={"temperature": 0.0},
    )
    coupler = make_coupler(
        components=(cast(Any, atmosphere), cast(Any, ocean)),
        exchanges=(
            Exchange(
                source="ATM",
                target="OCN",
                fields=["temperature"],
                regridder_factory=bilinear,
            ),
        ),
        run_order=("ATM", "OCN"),
    )

    rendered = str(coupler)
    representation = repr(coupler)

    assert "Coupler:" in rendered
    assert "<DataComponent>(ATM)" in rendered
    assert "ATM->OCN" in rendered
    assert "ATM, OCN" in rendered
    assert "run_order=ATM -> OCN" in representation


def test_coupler_run_happy_path_dispatches_and_steps_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    atmosphere = _HostRunComponent("ATM", events)
    ocean = _HostRunComponent("OCN", events)
    coupler = make_coupler(
        components=(cast(Any, atmosphere), cast(Any, ocean)),
        run_order=(
            "ATM",
            "OCN",
        ),
        logger=cast(Any, _RecordingLogger()),
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
        "vercor._runtime.driver.dispatch_component_exchanges", fake_dispatch
    )
    monkeypatch.setattr("vercor._runtime.driver.receive_runtime_fields", fake_receive)
    monkeypatch.setattr("vercor._runtime.driver.send_runtime_fields", fake_send)

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


def test_host_components_use_explicit_host_contract() -> None:
    host_component = _HostRunComponent("ATM")
    coupler = make_coupler(
        components=(cast(Any, host_component),),
        run_order=("ATM",),
    )

    final_state = coupler.run()
    final_component = final_state._component_state("ATM")

    assert host_component.spec.execution == "host"
    assert "host_time_seen" in final_component.fields.field_names
    assert_allclose_compact(
        final_component.fields.get("temperature"),
        np.full((2, 2), 61.0),
    )


def test_run_warns_when_host_backed_components_make_loop_nondifferentiable() -> None:
    logger = _RecordingLogger()
    coupler = make_coupler(
        components=(
            cast(Any, _HostRunComponent("ATM")),
            cast(Any, _HostRunComponent("OCN")),
        ),
        run_order=("ATM", "OCN"),
        logger=cast(Any, logger),
    )

    coupler.run()

    assert logger.warning_messages == [
        "Coupled loop is not differentiable because host-backed component(s) "
        "require the Python runtime: ATM, OCN"
    ]


def test_run_api_does_not_expose_state_donation_option() -> None:
    signature = inspect.signature(Coupler.run)

    assert "donate_state" not in signature.parameters


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
        "vercor._runtime.backends.step_runtime_component", fake_runtime_step
    )

    host_coupler = Coupler(
        clock=coupler.clock,
        components=(cast(Any, _HostRunComponent("ATM")),),
        run_order=("ATM",),
    )
    host_coupler.run()
    run_events = list(events)
    events.clear()

    timestamp = coupler.clock.start
    atmosphere = _RunComponent("ATM", [], timestamp)
    ocean = _RunComponent("OCN", [], timestamp)
    scan_coupler = Coupler(
        clock=coupler.clock,
        components=(cast(Any, atmosphere), cast(Any, ocean)),
        run_order=(
            "ATM",
            "OCN",
        ),
    )
    run_scanned_coupler(scan_coupler)

    assert run_events == ["run:ATM"]
    assert events == ["scan:ATM", "scan:OCN"]
