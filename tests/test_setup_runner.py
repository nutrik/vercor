from __future__ import annotations

from pathlib import Path
import sys

import pytest

from vercor import _setup_runner


def test_main_returns_setup_status_with_selected_runtime_options(
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        "    assert loglevel == 'debug'\n"
        "    assert float_type == 'float32'\n"
        "    return 7\n",
        encoding="utf-8",
    )

    status = _setup_runner.main(
        [str(setup), "--loglevel", "debug", "--float-type", "float32"]
    )

    assert status == 7


def test_main_reports_setup_contract_error_as_usage_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(SystemExit) as error:
        _setup_runner.main(
            [str(setup), "--loglevel", "info", "--float-type", "float64"]
        )

    expected_error = (
        f"Error: {setup} must define callable run_setup(*, loglevel, float_type)\n"
    )
    assert error.value.code == 2
    assert capsys.readouterr().err == expected_error


def test_invoke_setup_passes_keyword_options_and_none_means_success(
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        "    assert loglevel == 'debug'\n"
        "    assert float_type == 'float32'\n",
        encoding="utf-8",
    )

    assert (
        _setup_runner._invoke_setup(
            setup,
            loglevel="debug",
            float_type="float32",
        )
        == 0
    )


def test_invoke_setup_returns_integer_status(tmp_path: Path) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n" "    return 7\n",
        encoding="utf-8",
    )

    assert (
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )
        == 7
    )


@pytest.mark.parametrize(
    "source",
    (
        "VALUE = 1\n",
        "run_setup = 3\n",
    ),
)
def test_invoke_setup_requires_callable_contract(
    source: str,
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(source, encoding="utf-8")

    with pytest.raises(_setup_runner.SetupContractError, match="run_setup"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )


@pytest.mark.parametrize(
    "source",
    (
        "def run_setup(loglevel, float_type):\n" "    return None\n",
        "def run_setup(*, loglevel, float_type, extra=None):\n" "    return None\n",
        "def run_setup(**kwargs):\n" "    return None\n",
    ),
)
def test_invoke_setup_requires_exact_keyword_only_signature(
    source: str,
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(source, encoding="utf-8")

    with pytest.raises(_setup_runner.SetupContractError, match="keyword-only"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )


@pytest.mark.parametrize("value", ("True", "'bad'", "object()"))
def test_invoke_setup_rejects_non_status_return(
    value: str,
    tmp_path: Path,
) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n" f"    return {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(_setup_runner.SetupContractError, match="int or None"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )


def test_invoke_setup_skips_main_guard_and_supports_adjacent_imports(
    tmp_path: Path,
) -> None:
    (tmp_path / "helper.py").write_text("VALUE = 9\n", encoding="utf-8")
    marker = tmp_path / "marker.txt"
    setup = tmp_path / "setup.py"
    setup.write_text(
        "from helper import VALUE\n"
        "from pathlib import Path\n"
        "if __name__ == '__main__':\n"
        "    raise AssertionError('main guard ran')\n"
        "def run_setup(*, loglevel, float_type):\n"
        f"    Path({str(marker)!r}).write_text(str(VALUE), encoding='utf-8')\n",
        encoding="utf-8",
    )

    status = _setup_runner._invoke_setup(
        setup,
        loglevel="info",
        float_type="float64",
    )

    assert status == 0
    assert marker.read_text(encoding="utf-8") == "9"


def test_invoke_setup_supports_lazy_adjacent_import_and_restores_sys_path(
    tmp_path: Path,
) -> None:
    (tmp_path / "lazy_setup_helper.py").write_text("VALUE = 11\n", encoding="utf-8")
    marker = tmp_path / "lazy-marker.txt"
    setup = tmp_path / "setup.py"
    setup.write_text(
        "from pathlib import Path\n"
        "def run_setup(*, loglevel, float_type):\n"
        "    from lazy_setup_helper import VALUE\n"
        f"    Path({str(marker)!r}).write_text(str(VALUE), encoding='utf-8')\n",
        encoding="utf-8",
    )
    original_sys_path = list(sys.path)

    status = _setup_runner._invoke_setup(
        setup,
        loglevel="info",
        float_type="float64",
    )

    assert status == 0
    assert marker.read_text(encoding="utf-8") == "11"
    assert sys.path == original_sys_path


def test_invoke_setup_restores_sys_path_when_run_setup_raises(tmp_path: Path) -> None:
    setup = tmp_path / "setup.py"
    setup.write_text(
        "def run_setup(*, loglevel, float_type):\n"
        "    raise RuntimeError('setup failed')\n",
        encoding="utf-8",
    )
    original_sys_path = list(sys.path)

    with pytest.raises(RuntimeError, match="setup failed"):
        _setup_runner._invoke_setup(
            setup,
            loglevel="info",
            float_type="float64",
        )

    assert sys.path == original_sys_path
