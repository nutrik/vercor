"""Keep the approved JCM JAX override narrow and retain other dependencies."""

from pathlib import Path
import runpy
from typing import Any
from zipfile import ZipFile

import pytest


def _installer() -> dict[str, Any]:
    return runpy.run_path("tools/install_coupled_models.py")


def _wheel(tmp_path: Path, *, jax_requirement: str = "jax<0.11,>=0.10") -> Path:
    path = tmp_path / "jcm-3.0.0-py3-none-any.whl"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "jcm-3.0.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: jcm\nVersion: 3.0.0\n"
            "Requires-Dist: dinosaur>=1.5.0\n"
            "Requires-Dist: flax>=0.12.1\n"
            f"Requires-Dist: {jax_requirement}\n"
            "Requires-Dist: jax-rrtmgp>=0.5.0\n"
            'Requires-Dist: mam4-jax==0.5.1; extra == "mam4"\n',
        )
    return path


def test_override_preserves_every_other_base_requirement(tmp_path: Path) -> None:
    requirements = _installer()["coupled_requirements"](_wheel(tmp_path))
    assert requirements == [
        "dinosaur>=1.5.0",
        "flax>=0.12.1",
        "jax-rrtmgp>=0.5.0",
        "jax==0.11.2",
    ]


@pytest.mark.parametrize("requirement", ["jax>=0.12", "jax<0.10", "jax>=0.10"])
def test_override_refuses_unreviewed_jax_constraints(
    tmp_path: Path, requirement: str
) -> None:
    with pytest.raises(ValueError, match="JAX requirement"):
        _installer()["coupled_requirements"](
            _wheel(tmp_path, jax_requirement=requirement)
        )


def test_dependency_verification_allows_only_the_approved_exception() -> None:
    verify = _installer()["verify_dependency_report"]
    known = "jcm 3.0.0 has requirement jax<0.11,>=0.10, but you have jax 0.11.2.\n"
    verify(known, 1)
    with pytest.raises(RuntimeError, match="dependency"):
        verify(
            known + "veros has requirement click<=8.3.1, but you have click 8.4.2.\n", 1
        )
    with pytest.raises(RuntimeError, match="dependency"):
        verify(
            "jcm 3.0.0 has requirement jax<0.11,>=0.10, but you have jax 0.12.0.\n", 1
        )
