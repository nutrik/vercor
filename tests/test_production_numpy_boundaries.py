from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERCOR_ROOT = PROJECT_ROOT / "vercor"

ALLOWED_NUMPY_BOUNDARIES = {
    "vercor/diagnostics/plotting.py",
    "vercor/dtypes.py",
    "vercor/forcing_data.py",
    "vercor/host_arrays.py",
    "vercor/setups/external/jax_gcm_output.py",
    "vercor/setups/external/veros_output.py",
    "vercor/types.py",
}


def _imports_numpy(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "numpy" for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("numpy"):
                return True
    return False


def test_numpy_imports_match_explicit_host_boundaries() -> None:
    numpy_imports = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in VERCOR_ROOT.rglob("*.py")
        if _imports_numpy(path)
    }

    assert numpy_imports == ALLOWED_NUMPY_BOUNDARIES
