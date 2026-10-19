"""Mypy use site for the external extension test fixture."""

from pathlib import Path

from vercor.exchanges import Exchange
from vercor.regridding import RegridderFactory, bilinear, conservative
from external_extension_test_fixture import PluginRegridderFactory, run_smoke


def accepts_factory(factory: RegridderFactory) -> RegridderFactory:
    """Confirm a fixture factory satisfies the public factory protocol."""

    return factory


typed_factory = accepts_factory(PluginRegridderFactory("typed-route"))
typed_bilinear: RegridderFactory = bilinear
typed_conservative: RegridderFactory = conservative
typed_exchange_factory: RegridderFactory = Exchange(
    "SOURCE", "TARGET", ("field",), quantity="intensive"
).regridder_factory


def exercise_plugin(output_dir: Path) -> dict[str, object]:
    """Return typed smoke evidence from the external extension test fixture."""

    evidence: dict[str, object] = run_smoke(output_dir)
    return evidence
