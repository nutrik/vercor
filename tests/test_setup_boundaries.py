"""Boundary tests for lazy bundled setup imports and configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import cast

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_MODULE_ROOTS = {"credit", "dinosaur", "jcm", "tensorflow", "torch", "veros"}
RUNTIME_ENVIRONMENT_KEYS = (
    "TF_CPP_MIN_LOG_LEVEL",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
)
ENVIRONMENT_SENTINEL = "vercor-import-boundary"


def _test_package_root() -> Path:
    """Return the package root selected for fresh-process boundary probes."""

    return Path(os.environ.get("VERCOR_TEST_PACKAGE_ROOT", str(PROJECT_ROOT))).resolve()


def _run_setup_probe(statement: str) -> dict[str, object]:
    script = f"""
import json
import os
import sys

for key in {RUNTIME_ENVIRONMENT_KEYS!r}:
    os.environ[key] = {ENVIRONMENT_SENTINEL!r}

{statement}

print(json.dumps({{
    "vercor_file": getattr(sys.modules.get("vercor"), "__file__", None),
    "optional_modules": sorted(
        name for name in sys.modules
        if name.partition(".")[0] in {OPTIONAL_MODULE_ROOTS!r}
    ),
    "runtime_environment": {{
        key: os.environ.get(key) for key in {RUNTIME_ENVIRONMENT_KEYS!r}
    }},
    "setup_modules": sorted(
        name for name in sys.modules if name.startswith("vercor.setups")
    ),
}}))
"""
    package_root = _test_package_root()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, object], json.loads(completed.stdout.splitlines()[-1]))


def _run_missing_dependency_probe(
    *,
    factory_name: str,
    dependency_roots: tuple[str, ...],
    invocation: str,
) -> dict[str, str]:
    script = f"""
import builtins
import json
import vercor.setups as setups

factory = setups.{factory_name}
real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.partition(".")[0] in {dependency_roots!r}:
        raise ModuleNotFoundError("blocked optional dependency: " + name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
try:
    {invocation}
except Exception as error:
    print(json.dumps({{"type": type(error).__name__, "message": str(error)}}))
else:
    print(json.dumps({{"type": "", "message": ""}}))
"""
    package_root = _test_package_root()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return cast(dict[str, str], json.loads(completed.stdout.splitlines()[-1]))


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "statement",
    (
        "import vercor",
        "import vercor.setups",
        "import vercor.setups as setups; dir(setups)",
        (
            "from vercor.setups import (CAMulatorConfig, JAXGCMConfig, "
            "JCMLandAtmosphereConfig, Spinup, VerosConfig); "
            "Spinup(); JAXGCMConfig(); VerosConfig(); "
            "CAMulatorConfig(config_path='config.yml'); JCMLandAtmosphereConfig()"
        ),
    ),
)
def test_public_import_and_config_access_are_optional_dependency_free(
    statement: str,
) -> None:
    result = _run_setup_probe(statement)

    assert result["optional_modules"] == []
    assert result["runtime_environment"] == {
        key: ENVIRONMENT_SENTINEL for key in RUNTIME_ENVIRONMENT_KEYS
    }


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("factory_name", "factory_module"),
    (
        ("make_jax_gcm", "vercor.setups._external.jax_gcm"),
        ("make_veros_gcm", "vercor.setups._external.veros_gcm"),
        ("make_camulator_gcm", "vercor.setups._external.camulator"),
    ),
)
def test_lazy_factory_attribute_access_loads_only_lightweight_factory_module(
    factory_name: str,
    factory_module: str,
) -> None:
    result = _run_setup_probe(
        f"import vercor.setups as setups; factory = setups.{factory_name}; "
        "assert callable(factory)"
    )

    setup_modules = result["setup_modules"]
    assert isinstance(setup_modules, list)
    assert factory_module in setup_modules
    assert result["optional_modules"] == []
    assert result["runtime_environment"] == {
        key: ENVIRONMENT_SENTINEL for key in RUNTIME_ENVIRONMENT_KEYS
    }


@pytest.mark.fast_always
def test_vercor_setups_is_the_only_setup_lazy_export_registry() -> None:
    import vercor.setups as setups
    import vercor.setups._data as data_setups
    import vercor.setups._external as external_setups

    assert "_LAZY_EXPORTS" in vars(setups)
    for private_package in (data_setups, external_setups):
        assert "_LAZY_EXPORTS" not in vars(private_package)
        assert "__getattr__" not in vars(private_package)
        assert not any(name.startswith("make_") for name in dir(private_package))


@pytest.mark.fast_always
@pytest.mark.parametrize(
    ("factory_name", "dependency_roots", "invocation", "message"),
    (
        (
            "make_jax_gcm",
            ("jcm",),
            "factory(None, None)",
            "requires the jcm package.*pip install jcm",
        ),
        (
            "make_veros_gcm",
            ("veros",),
            "factory()",
            "requires the Veros package.*pip install veros",
        ),
        (
            "make_camulator_gcm",
            ("credit", "torch"),
            (
                "factory(config=setups.CAMulatorConfig("
                "config_path='missing.yml', device='cpu'))"
            ),
            "CREDIT modules are required.*credit is installed",
        ),
    ),
)
def test_missing_optional_dependencies_fail_at_factory_invocation(
    factory_name: str,
    dependency_roots: tuple[str, ...],
    invocation: str,
    message: str,
) -> None:
    result = _run_missing_dependency_probe(
        factory_name=factory_name,
        dependency_roots=dependency_roots,
        invocation=invocation,
    )

    assert result["type"] == "ImportError"
    assert re.search(message, result["message"]) is not None


@pytest.mark.fast_always
def test_camulator_enabled_spinup_fails_before_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups._external.camulator_runtime_settings as runtime_settings

    configuration_calls: list[None] = []

    def unexpected_runtime_configuration() -> None:
        configuration_calls.append(None)
        pytest.fail("CAMulator runtime must not be configured for unsupported spinup")

    monkeypatch.setattr(
        runtime_settings,
        "configure_camulator_runtime",
        unexpected_runtime_configuration,
    )

    with pytest.raises(
        ValueError,
        match="CAMulator spinup is not implemented.*Spinup\\(enabled=False\\)",
    ):
        setups.make_camulator_gcm(
            config=setups.CAMulatorConfig(
                config_path="unused.yml",
                spinup=setups.Spinup(enabled=True),
            )
        )

    assert configuration_calls == []


@pytest.mark.fast_always
@pytest.mark.parametrize(
    "module_name",
    (
        "vercor.setups._external.veros_output",
        "vercor.setups._external.veros_fluxes",
        "vercor.setups._external.veros_state",
    ),
)
def test_veros_implementation_import_does_not_configure_runtime(
    module_name: str,
) -> None:
    script = f"""
import importlib
import json
import vercor.setups._external.veros_runtime_settings as runtime_settings

calls = []
runtime_settings.configure_veros_runtime = lambda: calls.append("configure")
importlib.import_module({module_name!r})
print(json.dumps(calls))
"""
    package_root = _test_package_root()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=package_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout.splitlines()[-1]) == []


@pytest.mark.fast_always
def test_veros_factory_configures_once_before_implementation_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups._external.veros_gcm as factory_module
    import vercor.setups._external.veros_runtime_settings as runtime_settings

    events: list[str] = []

    class ImplementationReached(RuntimeError):
        pass

    def configure(execution: str) -> None:
        events.append(f"configure:{execution}")

    def load_implementation() -> object:
        events.append("implementation")
        raise ImplementationReached

    monkeypatch.setattr(runtime_settings, "configure_veros_runtime", configure)
    monkeypatch.setattr(
        factory_module,
        "_load_veros_implementation",
        load_implementation,
    )

    with pytest.raises(ImplementationReached):
        setups.make_veros_gcm()

    assert events == ["configure:host", "implementation"]


@pytest.mark.fast_always
def test_veros_jax_factory_checks_capabilities_before_implementation_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vercor.setups as setups
    import vercor.setups._external.veros_gcm as factory_module
    import vercor.setups._external.veros_runtime_settings as runtime_settings

    events: list[str] = []

    class ImplementationReached(RuntimeError):
        pass

    monkeypatch.setattr(
        runtime_settings,
        "configure_veros_runtime",
        lambda execution: events.append(f"configure:{execution}"),
    )
    monkeypatch.setattr(
        runtime_settings,
        "require_differentiable_veros_capabilities",
        lambda: events.append("capabilities"),
        raising=False,
    )

    def load_implementation() -> object:
        events.append("implementation")
        raise ImplementationReached

    monkeypatch.setattr(
        factory_module,
        "_load_veros_implementation",
        load_implementation,
    )

    with pytest.raises(ImplementationReached):
        setups.make_veros_gcm(config=setups.VerosConfig(execution="jax"))

    assert events == ["configure:jax", "capabilities", "implementation"]


@pytest.mark.fast_always
def test_setup_subprocess_helpers_honor_installed_package_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    installed_root = tmp_path / "site-packages"
    installed_root.mkdir()
    calls: list[dict[str, object]] = []

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        _ = args
        calls.append(kwargs)
        return SimpleNamespace(stdout="{}\n")

    monkeypatch.setenv("VERCOR_TEST_PACKAGE_ROOT", str(installed_root))
    monkeypatch.setattr(subprocess, "run", fake_run)

    _run_setup_probe("import vercor")
    _run_missing_dependency_probe(
        factory_name="make_veros_gcm",
        dependency_roots=("veros",),
        invocation="factory()",
    )

    assert len(calls) == 2
    for call in calls:
        assert call["cwd"] == installed_root
        environment = call["env"]
        assert isinstance(environment, dict)
        assert environment["PYTHONPATH"] == str(installed_root)
