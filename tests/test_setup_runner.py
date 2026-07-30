from __future__ import annotations

from pathlib import Path

import pytest

from vercor import _setup_runner


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
