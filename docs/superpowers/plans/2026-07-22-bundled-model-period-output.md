# Bundled Model Period Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable step-cadence period NetCDF output for every bundled slab and data component while preserving explicit opt-in I/O and third-party component defaults.

**Architecture:** A private bundled-setup helper will construct the existing public `OutputSpec(period=PeriodOutput(frequency="step"))` declaration. Slab factories, the shared time-interpolated data factory, and the direct JCM land data factory will reuse it; VerCOR's existing runtime-field provider and output session will continue to sample declared outputs and perform all accumulation and writes.

**Tech Stack:** Python 3.13, JAX, h5netcdf, pytest, Black, flake8, mypy, coverage.py, Git.

## Global Constraints

- Output remains run-level opt-in: `Coupler.run(output=None)` performs no provider sampling, host transfer, path creation, or file I/O.
- Period cadence for bundled slab and data factories is exactly `step`.
- Default period files contain exactly the component's declared `ComponentSpec.outputs`; input-only and exchanged-only fields are excluded.
- Do not change `ComponentSpec` defaults, public factory signatures, external/native providers, snapshot behavior, or final-field output.
- Use the existing public `OutputSpec` and `PeriodOutput` contracts and the existing private runtime-field provider; do not introduce another provider or writer lifecycle.
- Keep optional JCM and Veros imports lazy and preserve the dependency-free slab boundary.
- Use test-driven development and update `PROGRESS.md` and `DEPENDENCIES.md` as required by `AGENTS.md`.
- Do not tag, push, publish, or create a release.

---

## File Structure

- Create `vercor/setups/_output.py`: sole private constructor for the bundled step-period output declaration.
- Create `tests/test_bundled_period_output.py`: bundled declaration, integration, declared-output selection, and third-party-default regressions.
- Modify `vercor/setups/_slab/atmosphere.py`: attach the shared output declaration.
- Modify `vercor/setups/_slab/land.py`: attach the shared output declaration.
- Modify `vercor/setups/_slab/ocean.py`: attach the shared output declaration.
- Modify `vercor/setups/_slab/seaice.py`: attach the shared output declaration.
- Modify `vercor/setups/_data/_component_helpers.py`: attach the shared declaration to all time-interpolated data components.
- Modify `vercor/setups/_data/jcm_land.py`: attach the shared declaration to the direct daily JCM land data component.
- Modify `README.md`: document bundled defaults and custom-component opt-in.
- Modify `DEPENDENCIES.md`: place the new private setup helper in the topological import order.
- Modify `PROGRESS.md`: record the completed behavior and fresh verification evidence.

### Task 1: Specify bundled output behavior with failing tests

**Files:**
- Create: `tests/test_bundled_period_output.py`

**Interfaces:**
- Consumes: `Clock`, `Coupler`, `RectilinearGrid`, `CallableComponent`, `ComponentSpec`, `OutputSpec`, `OutputTarget`, `PeriodOutput`, bundled slab factories, `time_interpolated_data_component(...)`, and `make_jcm_land(...)`.
- Produces: executable acceptance criteria for step cadence, no custom provider, declared-output-only files, direct JCM data coverage, and unchanged third-party defaults.

- [ ] **Step 1: Add the complete failing behavior test module**

```python
"""Bundled slab and data factories declare generic step-period output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import h5netcdf
import jax.numpy as jnp
import numpy as np
import pytest

from tests._coverage_support import make_test_grid
from vercor.clock import Clock
from vercor.components import CallableComponent, ComponentSpec
from vercor.coupler import Coupler
from vercor.output import OutputSpec, OutputTarget, PeriodOutput
from vercor.setups import (
    make_slab_atmosphere,
    make_slab_land,
    make_slab_ocean,
    make_slab_seaice,
)
from vercor.setups._data import jcm_land as jcm_land_module
from vercor.setups._data._component_helpers import (
    time_interpolated_data_component,
)


def _assert_step_period_output(component: Any) -> None:
    output = component.spec.output
    assert isinstance(output, OutputSpec)
    assert output.provider is None
    assert output.period == PeriodOutput(frequency="step")


@pytest.mark.parametrize(
    "factory",
    (
        make_slab_atmosphere,
        make_slab_land,
        make_slab_ocean,
        make_slab_seaice,
    ),
)
def test_all_bundled_slab_factories_declare_step_period_output(factory: Any) -> None:
    _assert_step_period_output(factory(make_test_grid(name="slab-output")))


def test_shared_data_factory_declares_step_period_output() -> None:
    grid = make_test_grid(name="data-output")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 280.0)},
        inputs=("forcing",),
        outputs=("temperature",),
        initial_fields={"forcing": 1.0},
    )

    _assert_step_period_output(component)


def test_direct_jcm_land_data_factory_declares_step_period_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = make_test_grid(name="jcm-land-output")
    monkeypatch.setattr(
        jcm_land_module,
        "create_lnd_mask_from_ocn",
        lambda **kwargs: (
            jnp.ones(grid.shape),
            jnp.zeros(grid.shape),
        ),
    )
    coords = SimpleNamespace(
        horizontal=SimpleNamespace(
            longitudes=jnp.deg2rad(jnp.asarray([0.0, 180.0])),
            latitudes=jnp.deg2rad(jnp.asarray([-45.0, 45.0])),
        )
    )
    forcing = SimpleNamespace(
        stl_am=jnp.full(grid.shape, 280.0),
        soilw_am=jnp.full(grid.shape, 0.25),
    )

    component = jcm_land_module.make_jcm_land(
        cast(Any, coords),
        cast(Any, forcing),
        grid,
    )

    _assert_step_period_output(component)


def test_slab_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="slab-period")
    component = make_slab_atmosphere(grid, name="ATM")
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "atm.averages.2000-01-01.nc", "r") as dataset:
        assert set(component.spec.outputs).issubset(dataset.variables)
        assert "sea_surface_temperature" not in dataset.variables


def test_data_period_file_contains_declared_outputs_only(tmp_path: Path) -> None:
    grid = make_test_grid(name="data-period")
    component = time_interpolated_data_component(
        name="DATA",
        grid=grid,
        fields={"temperature": jnp.full(grid.shape, 280.0)},
        inputs=("forcing",),
        outputs=("temperature",),
        initial_fields={"forcing": 1.0},
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )

    with h5netcdf.File(tmp_path / "data.averages.2000-01-01.nc", "r") as dataset:
        np.testing.assert_allclose(dataset.variables["temperature"][0], 280.0)
        assert "forcing" not in dataset.variables


def test_custom_components_remain_period_output_opt_in(tmp_path: Path) -> None:
    grid = make_test_grid(name="custom-output")
    component = CallableComponent(
        "CUSTOM",
        grid,
        lambda fields: {"temperature": fields["temperature"] + 1.0},
        spec=ComponentSpec(
            outputs=("temperature",),
            initial_fields={"temperature": 280.0},
        ),
    )
    coupler = Coupler(
        Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=1),
        components=(component,),
        run_order=(component.name,),
        log_level="WARNING",
    )

    assert component.spec.output.period is None
    coupler.run(
        output=OutputTarget(
            tmp_path,
            write_final_fields=False,
            write_snapshots=False,
        )
    )
    assert not tuple(tmp_path.glob("*.averages.*.nc"))
```

- [ ] **Step 2: Run the new module and verify RED**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py -q --tb=short
```

Expected: FAIL because bundled slab/data `component.spec.output.period` is `None`, and the slab/data average files are absent. The custom-component regression passes.

### Task 2: Reuse one output declaration across bundled factories

**Files:**
- Create: `vercor/setups/_output.py`
- Modify: `vercor/setups/_slab/atmosphere.py`
- Modify: `vercor/setups/_slab/land.py`
- Modify: `vercor/setups/_slab/ocean.py`
- Modify: `vercor/setups/_slab/seaice.py`
- Modify: `vercor/setups/_data/_component_helpers.py`
- Modify: `vercor/setups/_data/jcm_land.py`
- Test: `tests/test_bundled_period_output.py`

**Interfaces:**
- Consumes: public `OutputSpec` and `PeriodOutput` constructors.
- Produces: `step_period_output() -> OutputSpec`, a private setup-layer constructor used only by bundled factories.

- [ ] **Step 1: Create the private shared declaration helper**

```python
"""Shared output declarations for bundled setup factories."""

from vercor.output import OutputSpec, PeriodOutput


def step_period_output() -> OutputSpec:
    """Return the generic step-cadence policy for bundled model output."""

    return OutputSpec(period=PeriodOutput(frequency="step"))


__all__: list[str] = []
```

- [ ] **Step 2: Attach the shared declaration in each slab factory**

In each of `vercor/setups/_slab/atmosphere.py`, `land.py`, `ocean.py`, and `seaice.py`, add:

```python
from vercor.setups._output import step_period_output
```

Then add this keyword to the existing `ComponentSpec(...)` call:

```python
output=step_period_output(),
```

- [ ] **Step 3: Attach the shared declaration in the shared data factory**

In `vercor/setups/_data/_component_helpers.py`, add:

```python
from vercor.setups._output import step_period_output
```

Then construct the existing `ComponentSpec` with:

```python
transfer=TransferPolicy(time_selection="linear"),
output=step_period_output(),
```

- [ ] **Step 4: Attach the shared declaration in the direct JCM land factory**

In `vercor/setups/_data/jcm_land.py`, add:

```python
from vercor.setups._output import step_period_output
```

Then construct the existing `ComponentSpec` with:

```python
transfer=TransferPolicy(time_selection="daily"),
output=step_period_output(),
```

- [ ] **Step 5: Run the new tests and verify GREEN**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py -q --tb=short
```

Expected: 9 parameter-expanded tests pass; the NetCDF assertions show that only declared outputs are written.

- [ ] **Step 6: Run the focused output and bundled-model regression set**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/test_bundled_period_output.py tests/test_v0_4_output_providers.py tests/test_native_period_output.py tests/test_slab_kernels.py tests/test_data_component_kernels.py tests/test_component_models_coverage.py -q --fast --tb=short
```

Expected: PASS with no failures.

- [ ] **Step 7: Run the fast suite before committing production behavior**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: PASS with no failures.

- [ ] **Step 8: Commit the tested behavior**

```bash
git add vercor/setups/_output.py vercor/setups/_slab vercor/setups/_data/_component_helpers.py vercor/setups/_data/jcm_land.py tests/test_bundled_period_output.py
git commit -m "feat: enable period output for bundled models"
```

### Task 3: Synchronize documentation and dependency ownership

**Files:**
- Modify: `README.md`
- Modify: `DEPENDENCIES.md`
- Modify: `docs/superpowers/specs/2026-07-22-bundled-model-period-output-design.md`

**Interfaces:**
- Consumes: the bundled factory behavior implemented in Task 2.
- Produces: user guidance and a topologically accurate module inventory; no runtime interface.

- [ ] **Step 1: Document bundled and custom component defaults in README**

After the paragraph explaining `PeriodOutput.variables`, add:

```markdown
Bundled slab and data factories declare step-cadence period output by default.
Their generic provider writes only fields declared in `ComponentSpec.outputs`.
Custom and third-party components remain opt-in and must attach their own
`OutputSpec(period=PeriodOutput(...))`. In every case, files are written only
when `Coupler.run` receives an enabled `OutputTarget`.
```

- [ ] **Step 2: Place the new helper in the dependency order**

In `DEPENDENCIES.md` layer 12, add `vercor/setups/_output.py` beside `vercor/setups/config.py`, and update the layer description to include private bundled output declarations:

```markdown
component declarations and normalization helpers, `DataComponent`, diagnostics,
shared output math/file primitives, frozen setup-specific configuration, and
private bundled output declarations.
```

- [ ] **Step 3: Verify documentation and source whitespace**

Run:

```bash
git diff --check
```

Expected: exit 0 with no output.

- [ ] **Step 4: Commit documentation synchronization**

```bash
git add README.md DEPENDENCIES.md docs/superpowers/specs/2026-07-22-bundled-model-period-output-design.md
git commit -m "docs: describe bundled model period output"
```

### Task 4: Run all quality gates and record evidence

**Files:**
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: the complete implementation and documentation commits.
- Produces: durable project-status evidence; no runtime interface.

- [ ] **Step 1: Apply project formatting**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy python -m black vercor examples tests
```

Expected: exit 0; any formatting changes are limited to files in this feature.

- [ ] **Step 2: Run strict linting**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy python -m flake8 . --count --max-line-length=120 --statistics
```

Expected: exit 0 and error count 0.

- [ ] **Step 3: Run static type checking**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy python -m mypy vercor examples tests
```

Expected: exit 0 with `Success: no issues found`.

- [ ] **Step 4: Compile all Python sources**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy python -m compileall -q vercor examples tests
```

Expected: exit 0 with no errors.

- [ ] **Step 5: Re-run the fast suite after formatting and static checks**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q --fast --tb=short
```

Expected: PASS with no failures.

- [ ] **Step 6: Run the full parallel test suite**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
```

Expected: PASS with no failures.

- [ ] **Step 7: Enforce branch coverage**

Run:

```bash
env CONDA_NO_PLUGINS=true conda run -n scipy pytest tests/ -q -n4 --dist=loadscope --max-worker-restart=0 --cov=vercor --cov-branch --cov-report=term-missing --cov-fail-under=90 --tb=short
```

Expected: PASS with branch coverage at or above 90%.

- [ ] **Step 8: Request an independent code-quality review**

Use `superpowers:requesting-code-review` with the base design commit and current implementation HEAD. Require review of scope compliance, DRY/SOLID boundaries, output semantics, test quality, and accidental public-API changes. Resolve every Critical or Important finding before continuing.

- [ ] **Step 9: Record durable evidence at the top of PROGRESS.md**

Add this bullet beneath `## Current Status`, enriching it with the exact test counts, warning counts, and coverage percentage from Steps 5-7:

```markdown
- Bundled slab/data step-period output completed locally (2026-07-22): one
  private setup declaration now enables the existing generic output provider
  across all slab factories, shared time-interpolated data factories, and the
  direct JCM land data factory. Period files contain declared outputs only;
  custom components remain opt-in and `output=None` remains I/O-free. Focused,
  fast, full, branch-coverage, Black, flake8, mypy, compileall, independent
  review, and whitespace gates passed.
```

- [ ] **Step 10: Verify the final diff and repository state**

Run:

```bash
git diff --check
git status --short
git diff --stat HEAD~2
```

Expected: `git diff --check` exits 0; only `PROGRESS.md` and any review corrections are uncommitted; the diff contains no unrelated files.

- [ ] **Step 11: Commit progress evidence and review corrections**

```bash
git add PROGRESS.md
git add vercor tests README.md DEPENDENCIES.md docs/superpowers/specs/2026-07-22-bundled-model-period-output-design.md
git commit -m "docs: record bundled output verification"
```

- [ ] **Step 12: Verify the committed result**

Run:

```bash
git status --short --branch
git log -4 --oneline
```

Expected: clean working tree on the original branch, with the design, implementation, documentation, and verification commits at HEAD. No push, tag, release, or publication has occurred.
