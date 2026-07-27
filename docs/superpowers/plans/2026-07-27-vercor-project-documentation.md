# VerCOR Project Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Read the Docs the canonical detailed VerCOR manual, with separate researcher and developer learning paths, tested examples, and curated stable and advanced API references.

**Architecture:** Build a task-oriented Sphinx site from focused reStructuredText pages, while MyST renders the maintained Markdown migration, plugin, release-note, and release guides without duplication. Runnable programs under `docs/_examples/` are the single source for tutorial code and are executed by pytest; explicit autodoc member lists keep private implementation modules out of the API reference.

**Tech Stack:** Sphinx 9.0.4, Furo, MyST Parser 5.1.0, reStructuredText, Sphinx autodoc/autosummary, pytest, JAX, and VerCOR's public API.

## Global Constraints

- Read the Docs is the canonical detailed documentation; `README.md` is a concise project overview and entry point.
- Provide separate Earth-system researcher and Python/JAX developer learning paths.
- Keep the stable user-facing API separate from advanced runtime, workflow, backend, and topology contracts.
- All displayed Python tutorials come from executable files under `docs/_examples/`.
- Tutorial examples use documented public interfaces only, deterministic in-memory data, and no network access.
- The researcher quick start requires no optional model package or external dataset.
- Do not change VerCOR runtime behavior, physics, numerics, public exports, or dependency policy for optional models.
- Treat Sphinx warnings as errors and preserve the repository's passing fast suite.
- Follow test-driven development and update `PROGRESS.md` after the complete documentation unit passes verification.

---

## File structure

### Existing files to modify

- `docs/conf.py`: enable MyST, define `.rst` and `.md` source types, and exclude internal/archive Markdown from the built site.
- `docs/requirements.txt`: pin MyST Parser 5.1.0 for Read the Docs.
- `.readthedocs.yaml`: install the VerCOR project before autodoc imports public modules.
- `docs/index.rst`: become the audience-routing landing page and root toctree.
- `README.md`: become the concise repository entry point.
- `tests/test_docs_build.py`: protect navigation, source policy, API boundaries, and the strict HTML build.
- `PROGRESS.md`: record canonical documentation scope and final verification.

### Narrative pages to create

- `docs/introduction.rst`: project description and conceptual architecture.
- `docs/researchers/index.rst`: researcher path entry page.
- `docs/researchers/getting-started.rst`: dependency-light installation and first run.
- `docs/researchers/running.rst`: assemble, run, and inspect simulations.
- `docs/developers/index.rst`: developer path entry page.
- `docs/developers/concepts.rst`: structural component and runtime-state contracts.
- `docs/developers/data-components.rst`: static/monthly/daily forcing components.
- `docs/developers/host-components.rst`: host execution and functional payload state.
- `docs/developers/jax-components.rst`: JAX execution, JIT, PyTrees, and gradients.
- `docs/developers/coupling.rst`: exchanges, route IDs, run order, and regridding.
- `docs/how-to/index.rst`: shared task-oriented guide entry page.
- `docs/how-to/examples.rst`: repository example prerequisites and invocation.
- `docs/how-to/backends.rst`: `auto`, `host`, and `jax` selection.
- `docs/how-to/output.rst`: output opt-in, cadence, final fields, and transform constraints.
- `docs/troubleshooting.rst`: concise symptom/cause/action guidance.
- `docs/project-resources.rst`: toctree for maintained Markdown project documents.

### API pages to create

- `docs/api/index.rst`: reference entry page and stability-tier explanation.
- `docs/api/assembly.rst`: root assembly, clock, coupler, runtime options, and state.
- `docs/api/components.rst`: component authoring contracts and adapters.
- `docs/api/grids-exchanges.rst`: grids, exchange declarations, fields, and regridding.
- `docs/api/output-diagnostics.rst`: output and diagnostic interfaces.
- `docs/api/setups-physics-types.rst`: bundled setup factories, physical constants, and public types.
- `docs/api/advanced.rst`: workflow, backend, driver, and topology extension contracts.

### Executable documentation examples to create

- `docs/_examples/quickstart.py`: bundled slab-ocean quick start.
- `docs/_examples/data_component.py`: static and time-selected data declarations.
- `docs/_examples/host_component.py`: host callable with immutable payload replacement.
- `docs/_examples/jax_component.py`: differentiable output-free JAX run.
- `docs/_examples/coupled_components.py`: forcing and model connected by an exchange.
- `tests/test_documentation_examples.py`: execute examples and reject private VerCOR imports.

---

### Task 1: Sphinx foundation, navigation, and introduction

**Files:**

- Modify: `tests/test_docs_build.py`
- Modify: `docs/conf.py`
- Modify: `docs/requirements.txt`
- Modify: `.readthedocs.yaml`
- Modify: `docs/index.rst`
- Create: `docs/introduction.rst`
- Create: `docs/researchers/index.rst`
- Create: `docs/developers/index.rst`
- Create: `docs/how-to/index.rst`
- Create: `docs/api/index.rst`
- Create: `docs/troubleshooting.rst`
- Create: `docs/project-resources.rst`

**Interfaces:**

- Consumes: current Sphinx configuration and strict HTML build test.
- Produces: dual-source Sphinx configuration and the stable root navigation that all later pages join.

- [ ] **Step 1: Add failing navigation and source-policy contracts**

Extend `tests/test_docs_build.py` with the required root pages and configuration assertions:

```python
EXPECTED_ROOT_PAGES = (
    "introduction",
    "researchers/index",
    "developers/index",
    "how-to/index",
    "api/index",
    "troubleshooting",
    "project-resources",
)


@pytest.mark.fast_always
def test_documentation_has_two_learning_paths_and_reference_sections() -> None:
    """Keep every canonical top-level destination in the root toctree."""
    index_source = (DOCS_ROOT / "index.rst").read_text(encoding="utf-8")

    for page in EXPECTED_ROOT_PAGES:
        assert page in index_source

    assert "For Earth-system researchers" in index_source
    assert "For Python and JAX developers" in index_source


@pytest.mark.fast_always
def test_sphinx_builds_rst_and_selected_markdown_sources() -> None:
    """Keep MyST and archive exclusions explicit in the Sphinx policy."""
    conf_source = (DOCS_ROOT / "conf.py").read_text(encoding="utf-8")
    requirements = (DOCS_ROOT / "requirements.txt").read_text(encoding="utf-8")
    readthedocs = (PROJECT_ROOT / ".readthedocs.yaml").read_text(encoding="utf-8")

    assert '"myst_parser"' in conf_source
    assert '".rst": "restructuredtext"' in conf_source
    assert '".md": "markdown"' in conf_source
    assert "README.md" in conf_source
    assert "progress-archive-*.md" in conf_source
    assert "myst-parser==5.1.0" in requirements
    assert "method: pip" in readthedocs
    assert "path: ." in readthedocs
```

- [ ] **Step 2: Run the contracts to verify RED**

Run:

```bash
conda run -n scipy pytest tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: the two new tests fail because the audience routes, MyST extension, source mapping, exclusions, and dependency are absent.

- [ ] **Step 3: Configure MyST and explicit documentation sources**

In `docs/requirements.txt`, add a newline after the existing final requirement and add:

```text
myst-parser==5.1.0
```

In `docs/conf.py`, add `"myst_parser"` to `extensions`, replace the scalar `source_suffix` with:

```python
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
```

Extend `exclude_patterns` with:

```python
"README.md",
"api-architecture-review.md",
"over-engineering-*.md",
"progress-archive-*.md",
```

Keep `superpowers` excluded. Do not exclude the migration, plugin-authoring, release-note, or release-guide Markdown files.

Extend `.readthedocs.yaml` so the documentation requirements are installed
first and the repository project is installed second:

```yaml
python:
  install:
    - requirements: docs/requirements.txt
    - method: pip
      path: .
```

This makes JAX and VerCOR's other core runtime dependencies available when
autodoc imports public modules.

- [ ] **Step 4: Create the root audience-routing pages**

Replace `docs/index.rst` with:

```rst
VerCOR documentation
====================

VerCOR is a JAX-first coupler for composing atmosphere, ocean, sea-ice, land,
and forcing-data components on a shared clock. It moves fields between model
grids, supports host-side and JAX-native components, and keeps output-free JAX
workflows differentiable.

For Earth-system researchers
----------------------------

Start with :doc:`researchers/index` to install VerCOR, run a small bundled
model, and inspect coupled state and output.

For Python and JAX developers
-----------------------------

Start with :doc:`developers/index` to implement data, host, or differentiable
JAX components and connect them with exchanges.

.. toctree::
   :maxdepth: 2
   :caption: Learn

   introduction
   researchers/index
   developers/index

.. toctree::
   :maxdepth: 2
   :caption: Use and extend

   how-to/index
   troubleshooting
   api/index
   project-resources
```

Create `docs/introduction.rst` with sections titled `What VerCOR is`,
`Why JAX`, `How a run fits together`, and `Execution and differentiation`.
Define components, grids, exchanges, clock, immutable `RunState`, and opt-in
output in plain language. State exactly that output-free JAX workflows can be
differentiated end to end and that scheduling any host component selects host
execution when `backend="auto"`.

Create the three path index pages with brief audience-specific goals and
toctrees. At this stage, their toctrees include only pages created in the same
task; later tasks extend them without creating broken links:

```rst
.. toctree::
   :maxdepth: 1

   ../introduction
```

Create `docs/api/index.rst` with two paragraphs explaining that the main API is
curated from supported public owners, private `vercor._*` modules are excluded,
and runtime/topology extension points are documented separately.

Create `docs/troubleshooting.rst` with a concise opening explanation and the
five symptom headings approved in the design. Each heading initially directs
the reader to the appropriate researcher or developer path without linking to
pages that do not yet exist; Task 4 replaces these summaries with complete
symptom/cause/action entries.

Create `docs/project-resources.rst` with a short description and this initial
navigation. Include every maintained Markdown source immediately so enabling
MyST does not leave discovered documents outside a toctree:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Migration and extension

   migration-0.3-to-0.4
   plugin-authoring

.. toctree::
   :maxdepth: 1
   :caption: Releases

   release-notes-0.4.3
   release-notes-0.4.2
   release-notes-0.4.1
   release-notes-0.4.0
   releasing
```

Task 6 adds reader-oriented descriptions and finalizes README ownership.

- [ ] **Step 5: Install the documentation parser in the development environment**

Run:

```bash
conda run -n scipy python -m pip install "myst-parser==5.1.0"
```

Expected: installation succeeds and `conda run -n scipy python -c "import myst_parser"` exits 0.

- [ ] **Step 6: Run focused GREEN checks**

Run:

```bash
conda run -n scipy pytest tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: all documentation build tests pass with Sphinx warnings treated as errors.

- [ ] **Step 7: Commit the documentation foundation**

```bash
git add .readthedocs.yaml docs/conf.py docs/requirements.txt docs/index.rst \
  docs/introduction.rst docs/researchers/index.rst \
  docs/developers/index.rst docs/how-to/index.rst docs/api/index.rst \
  docs/troubleshooting.rst docs/project-resources.rst \
  tests/test_docs_build.py
git commit -m "docs: establish canonical documentation structure"
```

---

### Task 2: Researcher quick start and running guide

**Files:**

- Create: `tests/test_documentation_examples.py`
- Create: `docs/_examples/quickstart.py`
- Create: `docs/researchers/getting-started.rst`
- Create: `docs/researchers/running.rst`
- Modify: `docs/researchers/index.rst`

**Interfaces:**

- Consumes: public `Clock`, `Coupler`, `RectilinearGrid`, `RunState`, and `make_slab_ocean`.
- Produces: `quickstart.py` as the tested source for the researcher tutorial and later output guidance.

- [ ] **Step 1: Add the failing executable-example contract**

Create `tests/test_documentation_examples.py`:

```python
"""Executable contracts for code published in the user documentation."""

from __future__ import annotations

import ast
import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = PROJECT_ROOT / "docs" / "_examples"
EXAMPLE_NAMES = ("quickstart.py",)


@pytest.mark.fast_always
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_documentation_example_executes(name: str) -> None:
    """Execute each complete documentation program."""
    runpy.run_path(str(EXAMPLES_ROOT / name), run_name="__main__")


@pytest.mark.fast_always
@pytest.mark.parametrize("name", EXAMPLE_NAMES)
def test_documentation_example_uses_only_public_vercor_imports(name: str) -> None:
    """Reject private VerCOR imports in published examples."""
    source = (EXAMPLES_ROOT / name).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=name)
    imported_modules = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    imported_modules.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(not module.startswith("vercor._") for module in imported_modules)
```

- [ ] **Step 2: Run the new test to verify RED**

Run:

```bash
conda run -n scipy pytest tests/test_documentation_examples.py \
  -q -n0 --tb=short
```

Expected: collection or execution fails because `docs/_examples/quickstart.py` does not exist.

- [ ] **Step 3: Write the minimal runnable slab quick start**

Create `docs/_examples/quickstart.py` with a two-by-two grid, two hourly
steps, one bundled slab ocean, and public state inspection:

```python
"""Run a dependency-light bundled VerCOR component."""

from datetime import datetime

import jax.numpy as jnp

from vercor import Clock, Coupler, RectilinearGrid
from vercor.setups import make_slab_ocean


grid = RectilinearGrid.uniform(
    "quickstart",
    nlon=2,
    nlat=2,
    longitude=(0.0, 360.0),
    latitude=(-90.0, 90.0),
)
clock = Clock(datetime(2000, 1, 1), dt_seconds=3600.0, steps=2)
ocean = make_slab_ocean(grid)
coupler = Coupler(clock, components=(ocean,), run_order=(ocean.name,))

final_state = coupler.run()
sea_surface_temperature = final_state.component(ocean.name).field(
    "sea_surface_temperature"
)

assert sea_surface_temperature.shape == grid.shape
assert bool(jnp.all(jnp.isfinite(sea_surface_temperature)))
```

- [ ] **Step 4: Write the researcher pages from the executable source**

Create `docs/researchers/getting-started.rst` with `Requirements`,
`Installation`, `Run your first component`, `Inspect the result`, and `Next
steps`. Document Python 3.12/3.13 and:

```console
python -m pip install "vercor==0.4.3"
```

Include the complete script:

```rst
.. literalinclude:: ../_examples/quickstart.py
   :language: python
   :linenos:
```

Explain `RectilinearGrid`, `Clock`, `Coupler`, and `RunState` after the code.

Create `docs/researchers/running.rst` with `Assemble once`, `Choose component
order`, `Run from a new or existing state`, and `Inspect component fields`.
Use exact public calls:

```python
initial_state = coupler.initial_state()
final_state = coupler.run(initial_state)
ocean_state = final_state.component("OCN")
all_components = final_state.components()
```

Explain that a new `Coupler` is required to change configured components,
exchanges, runtime policy, or run order.

Update `docs/researchers/index.rst` to order:

```rst
getting-started
running
../troubleshooting
```

Task 4 adds the output how-to link after that page exists.

- [ ] **Step 5: Run researcher documentation checks**

Run:

```bash
conda run -n scipy pytest \
  tests/test_documentation_examples.py tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: the quick start executes and the strict Sphinx build passes.

- [ ] **Step 6: Commit the researcher path**

```bash
git add tests/test_documentation_examples.py docs/_examples/quickstart.py \
  docs/researchers
git commit -m "docs: add researcher getting-started path"
```

---

### Task 3: Data, host, and JAX component-authoring guides

**Files:**

- Modify: `tests/test_documentation_examples.py`
- Create: `docs/_examples/data_component.py`
- Create: `docs/_examples/host_component.py`
- Create: `docs/_examples/jax_component.py`
- Create: `docs/developers/concepts.rst`
- Create: `docs/developers/data-components.rst`
- Create: `docs/developers/host-components.rst`
- Create: `docs/developers/jax-components.rst`
- Modify: `docs/developers/index.rst`

**Interfaces:**

- Consumes: `DataComponent`, `CallableComponent`, `ComponentSpec`, `TransferPolicy`, lifecycle/result contracts, `RuntimeOptions`, and immutable `RunState.replace_fields`.
- Produces: three independently executable component-authoring examples and the core developer learning path.

- [ ] **Step 1: Extend the example inventory to produce RED**

Change `EXAMPLE_NAMES` in `tests/test_documentation_examples.py` to:

```python
EXAMPLE_NAMES = (
    "quickstart.py",
    "data_component.py",
    "host_component.py",
    "jax_component.py",
)
```

Run:

```bash
conda run -n scipy pytest tests/test_documentation_examples.py \
  -q -n0 --tb=short
```

Expected: six parameter cases fail because the three new files do not exist.

- [ ] **Step 2: Add the data-component example**

Create `docs/_examples/data_component.py`. Build one two-by-two grid, then
construct:

```python
static_forcing = DataComponent(
    "STATIC",
    grid,
    {"heat_flux": 25.0},
)

monthly_forcing = DataComponent(
    "MONTHLY",
    grid,
    {"temperature": jnp.arange(12.0)[:, None, None] * jnp.ones((12, *grid.shape))},
    spec=ComponentSpec(transfer=TransferPolicy("linear")),
)
```

Create a zero-step `Coupler` for both components, call `initial_state()`, and
assert that the static field expands to `grid.shape` while the monthly field
keeps shape `(12, *grid.shape)`. The prose explains that `linear` selects
adjacent monthly records during exchange and `daily` expects 365 or 360
records according to the configured calendar.

- [ ] **Step 3: Add the host-component example**

Create `docs/_examples/host_component.py` with:

```python
@dataclass(frozen=True)
class HostPayload:
    calls: int = 0


def setup_host(component: object, context: SetupContext) -> SetupResult:
    _ = component, context
    return SetupResult(payload=HostPayload())


def host_step(
    fields: Mapping[str, RuntimeArray],
    context: StepContext,
    payload: object | None,
) -> StepResult:
    _ = context
    if not isinstance(payload, HostPayload):
        raise TypeError("host payload was not initialized")
    return StepResult(
        fields={"counter": fields["counter"] + 1.0},
        payload=HostPayload(payload.calls + 1),
    )
```

Wrap it with `CallableComponent`, declare output `counter`, initial value
`0.0`, `execution="host"`, and the setup lifecycle. Run two steps through
`RuntimeOptions(backend="auto")` and assert the public final field equals
`2.0` everywhere. Keep evolving state in the returned payload, never on the
component author object.

- [ ] **Step 4: Add the differentiable JAX-component example**

Create `docs/_examples/jax_component.py` with a pure step that increments
`temperature` by `heating_rate * context.dt_seconds`. Use
`CallableComponent`, `ComponentSpec(execution="jax")`,
`RuntimeOptions(backend="jax")`, and `output=None`.

Create one initial state outside the differentiated function. Define:

```python
def final_temperature_sum(initial_temperature: jax.Array) -> jax.Array:
    state = initial_state.replace_fields(
        "MODEL",
        {"temperature": jnp.full(grid.shape, initial_temperature)},
    )
    result = coupler.run(state, output=None)
    return jnp.sum(result.component("MODEL").field("temperature"))
```

Evaluate `jax.grad(final_temperature_sum)(jnp.asarray(280.0))` and assert that
it equals the number of grid cells. Also assert that `jax.jit` of the function
returns a finite scalar.

- [ ] **Step 5: Write the developer concepts and component pages**

Create `docs/developers/concepts.rst` explaining the structural component
contract—`name`, `grid`, `spec`, and `step`—plus field declarations, immutable
runtime state, payload ownership, and the distinction between author
configuration and evolving runtime payload.

Create the three component pages. Each page includes its matching script:

```rst
.. literalinclude:: ../_examples/data_component.py
   :language: python
   :linenos:
```

Use the corresponding filename on each page. Required callouts:

- data: scalar expansion, record-axis preservation, `current`, `linear`, and
  `daily`;
- host: `execution="host"`, `backend="auto"`, functional payload replacement,
  and no hidden mutable evolving state;
- JAX: pure array operations, stable PyTree/shape/dtype, no Python branching on
  traced physics, `output=None`, JIT, and gradients.

Update `docs/developers/index.rst` to order:

```rst
concepts
data-components
host-components
jax-components
coupling
../api/index
```

Do not add `coupling` until Task 4 creates it; during Task 3, omit that one
entry so the strict Sphinx build remains green.

- [ ] **Step 6: Run component-guide GREEN checks**

Run:

```bash
conda run -n scipy pytest \
  tests/test_documentation_examples.py tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: all four scripts execute, private imports are absent, and Sphinx builds without warnings.

- [ ] **Step 7: Commit the component-authoring path**

```bash
git add tests/test_documentation_examples.py docs/_examples/data_component.py \
  docs/_examples/host_component.py docs/_examples/jax_component.py \
  docs/developers
git commit -m "docs: add component authoring guides"
```

---

### Task 4: Coupling, operational how-to guides, and troubleshooting

**Files:**

- Modify: `tests/test_documentation_examples.py`
- Create: `docs/_examples/coupled_components.py`
- Create: `docs/developers/coupling.rst`
- Create: `docs/how-to/examples.rst`
- Create: `docs/how-to/backends.rst`
- Create: `docs/how-to/output.rst`
- Modify: `docs/troubleshooting.rst`
- Modify: `docs/researchers/index.rst`
- Modify: `docs/developers/index.rst`
- Modify: `docs/how-to/index.rst`

**Interfaces:**

- Consumes: earlier data/JAX component patterns, `Exchange`, built-in regridder factories, `OutputTarget`, and `OutputSpec`.
- Produces: a complete exchange example plus shared operational guidance used by both learning paths.

- [ ] **Step 1: Add the coupled example to produce RED**

Append `"coupled_components.py"` to `EXAMPLE_NAMES`, run the example tests, and
observe two failures for the absent file.

- [ ] **Step 2: Implement the coupled-components example**

Create one `DataComponent("FORCING", ...)` that outputs `heat_flux` and one
`CallableComponent("MODEL", ...)` whose spec declares `heat_flux` as an input,
`temperature` as an output, and initializes both fields. Connect them with:

```python
Exchange(
    "FORCING",
    "MODEL",
    ("heat_flux",),
    route_id="forcing-to-model",
    regridder_factory=bilinear,
)
```

Use identical two-by-two grids, run order `("FORCING", "MODEL")`, and one
step. Assert through `final_state.component("MODEL").field("temperature")`
that the model received and used the forcing.

- [ ] **Step 3: Write the coupling guide**

Create `docs/developers/coupling.rst` and include the complete example with
`literalinclude`. Explain:

- both endpoints must declare exchanged fields;
- `run_order` controls receive/step/send sequencing;
- the default route ID is `"source->target"`;
- explicit unique route IDs are required for multiple routes between one pair;
- `bilinear` handles scalar/vector interpolation and `conservative` is scalar;
- ambiguous fan-in to the same target field is rejected; and
- `SurfaceMaskPolicy` is for the bundled atmosphere/ocean/land topology, not
  ordinary setup-agnostic graphs.

Add `coupling` to the developer index.

- [ ] **Step 4: Write task-oriented how-to pages**

Create `docs/how-to/examples.rst` with a table containing each repository
example, its purpose, optional dependency, and external-data requirement.
Document invocation as:

```console
python -m examples.run_slab_driver
python -m examples.custom_component_wrapping
```

Warn that CAMulator paths and checkpoints are machine-specific and must be
reconfigured.

Create `docs/how-to/backends.rst` with a comparison table:

| setting | behavior | valid components |
| --- | --- | --- |
| `auto` | selects host when any scheduled component is host-backed | mixed |
| `jax` | compiled JAX execution | JAX only |
| `host` | Python driver | JAX and host |

Use `RuntimeOptions(backend="auto")` as the default recommendation for mixed
graphs and state that forced JAX rejects host components before stepping.

Create `docs/how-to/output.rst` covering:

- `coupler.run(output=None)` for no I/O and transformed runs;
- `OutputTarget(path)` and its three enable/disable flags;
- `OutputSpec` and `PeriodOutput(frequency="step" | "month")`;
- final fields versus period files versus component snapshots; and
- why enabled file output rejects traced runtime state.

Use short API fragments, not a second copy of the quick-start program.

Update `docs/how-to/index.rst` with:

```rst
examples
backends
output
../troubleshooting
```

Add `../how-to/output` between `running` and `../troubleshooting` in
`docs/researchers/index.rst`.

- [ ] **Step 5: Write concise troubleshooting entries**

Create `docs/troubleshooting.rst` with five sections:

1. `An optional setup cannot be imported`
2. `An exchange field is rejected`
3. `A host component fails with the JAX backend`
4. `A compiled payload changes structure`
5. `Output fails under JIT or differentiation`

For each, use `Symptom`, `Cause`, and `Action` paragraphs and cross-reference
the relevant full guide. State that CAMulator has no documented compatible
dependency pin.

- [ ] **Step 6: Run operational-guide checks**

Run:

```bash
conda run -n scipy pytest \
  tests/test_documentation_examples.py tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: all five examples execute and Sphinx resolves all guide cross-references.

- [ ] **Step 7: Commit coupling and how-to documentation**

```bash
git add tests/test_documentation_examples.py \
  docs/_examples/coupled_components.py docs/developers/coupling.rst \
  docs/developers/index.rst docs/researchers/index.rst docs/how-to \
  docs/troubleshooting.rst
git commit -m "docs: add coupling and operational guides"
```

---

### Task 5: Curated stable and advanced Python API references

**Files:**

- Modify: `tests/test_docs_build.py`
- Modify: `docs/api/index.rst`
- Create: `docs/api/assembly.rst`
- Create: `docs/api/components.rst`
- Create: `docs/api/grids-exchanges.rst`
- Create: `docs/api/output-diagnostics.rst`
- Create: `docs/api/setups-physics-types.rst`
- Create: `docs/api/advanced.rst`

**Interfaces:**

- Consumes: explicit `__all__` manifests in VerCOR's public owner modules.
- Produces: curated autodoc pages that expose supported public objects without documenting private modules.

- [ ] **Step 1: Add failing API reference boundaries**

Add to `tests/test_docs_build.py`:

```python
API_PAGES = (
    "assembly.rst",
    "components.rst",
    "grids-exchanges.rst",
    "output-diagnostics.rst",
    "setups-physics-types.rst",
    "advanced.rst",
)


@pytest.mark.fast_always
def test_api_reference_is_curated_and_separates_advanced_contracts() -> None:
    """Document public owners explicitly and keep private modules absent."""
    sources = []
    for name in API_PAGES:
        source = (DOCS_ROOT / "api" / name).read_text(encoding="utf-8")
        assert ".. auto" in source
        assert "vercor._" not in source
        sources.append(source)

    stable_source = "\n".join(sources[:-1])
    advanced_source = sources[-1]
    assert "vercor.runtime" not in stable_source
    assert ".. automodule:: vercor.runtime" in advanced_source
    assert ".. automodule:: vercor.topology" in advanced_source
```

Run the focused test and observe failure because the six API pages are absent.

- [ ] **Step 2: Build the stable reference pages with explicit member lists**

Use `.. automodule::` plus `:members:` lists, not recursive autosummary.

`docs/api/assembly.rst` documents:

- `vercor`: `Clock`, `Coupler`, `Exchange`, `RectilinearGrid`, `RunState`,
  `RuntimeOptions`;
- `vercor.state`: `ComponentState`, `FieldLookupScope`, `FieldScope`; and
- `vercor.calendar`: `CalendarDate`, `DateTime360`, `DateTime365`,
  `ModelDateTime`, and `YearType`.

`docs/api/components.rst` documents the exact
`vercor.components.__all__` names:

```rst
.. automodule:: vercor.components
   :members: CallableComponent, Component, ComponentSpec, DataComponent, LifecycleHooks, PrefillContext, PrefillResult, SetupContext, SetupResult, StepContext, StepResult, TransferPolicy, ValidationContext
   :show-inheritance:
```

`docs/api/grids-exchanges.rst` documents:

- `vercor.grids.RectilinearGrid`;
- `vercor.exchanges.Exchange`;
- `COMMON_FIELD_NAMES`, `ExchangeField`, `VectorField`, and `vector` from
  `vercor.fields`; and
- `Regridder`, `RegridderFactory`, `VectorRegridder`, `bilinear`, and
  `conservative` from `vercor.regridding`.

`docs/api/output-diagnostics.rst` documents these exact members under separate
module headings:

- `vercor.output`: `OutputContext`, `OutputFrame`, `OutputProvider`,
  `OutputSpec`, `OutputTarget`, `OutputVariable`, `PeriodOutput`,
  `SnapshotContext`, and `SnapshotWriter`;
- `vercor.diagnostics`: `ComponentMetric`, `combine_surface_temperatures`,
  `component_vector_speed`, `plot_component_scalar_vector_comparison`,
  `print_component_field_means_table`, `safe_component_nanmean`, and
  `total_surface_temperature`.

`docs/api/setups-physics-types.rst` documents:

- `CAMulatorConfig`, `JAXGCMConfig`, `JCMLandAtmosphereConfig`,
  `JCMLandAtmosphereSetup`, `JCMInputs`, `Spinup`, `VerosConfig`,
  `load_jcm_inputs`, `make_slab_atmosphere`, `make_slab_land`,
  `make_slab_ocean`, `make_slab_seaice`, `make_jcm_land_atmosphere`,
  `make_camulator_gcm`, `make_camulator_land`, `make_era5_atmosphere`,
  `make_era5_land`, `make_era5_ocean`, `make_erainterim_ocean`,
  `make_jax_gcm`, `make_jcm_land`, and `make_veros_gcm` from
  `vercor.setups`;
- `vercor.physics.PhysicalConstants`; and
- `vercor.types.RuntimeArray`.

Use exactly these member lists. If a public owner manifest changes before
execution, stop and reconcile that API change with the approved design instead
of silently expanding the documentation. Do not use `:private-members:` or
`:undoc-members:`.

- [ ] **Step 3: Build the advanced runtime and topology page**

Create `docs/api/advanced.rst` with a warning that these contracts are for
custom schedulers, backends, and exchange topology policies. Document the
exact public members of:

```rst
.. automodule:: vercor.runtime
   :members: ExecutionBackend, ExecutionChunk, ExecutionContext, ExecutionPlan, RuntimeDriver, RuntimeOptions, SequentialWorkflow, StepPlan, Workflow, WorkflowContext
   :show-inheritance:

.. automodule:: vercor.topology
   :members: ExchangeTopologyPatch, SurfaceMaskPolicy, TopologyContext, TopologyPolicy
   :show-inheritance:
```

Keep `RuntimeOptions` linked from the common assembly page, but place the
workflow, plan, chunk, backend, driver, and topology protocol explanations only
in this advanced page.

- [ ] **Step 4: Complete API navigation**

Update `docs/api/index.rst`:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Stable user-facing API

   assembly
   components
   grids-exchanges
   output-diagnostics
   setups-physics-types

.. toctree::
   :maxdepth: 1
   :caption: Advanced extension API

   advanced
```

- [ ] **Step 5: Run API and Sphinx GREEN checks**

Run:

```bash
conda run -n scipy pytest tests/test_docs_build.py \
  -q -n0 --tb=short
```

Expected: API ownership contracts and strict autodoc build pass without import warnings or private-module references.

- [ ] **Step 6: Commit the curated API reference**

```bash
git add tests/test_docs_build.py docs/api
git commit -m "docs: add curated Python API reference"
```

---

### Task 6: Project resources and concise canonical README

**Files:**

- Modify: `tests/test_docs_build.py`
- Modify: `docs/project-resources.rst`
- Modify: `README.md`

**Interfaces:**

- Consumes: hosted Read the Docs project, maintained Markdown guides, and the completed learning paths/API.
- Produces: one canonical documentation entry point from GitHub and one in-site project-resource index.

- [ ] **Step 1: Add failing canonical-ownership contracts**

Add:

```python
@pytest.mark.fast_always
def test_readme_is_a_concise_gateway_to_canonical_documentation() -> None:
    """Keep detailed guidance on Read the Docs instead of in the README."""
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "https://vercor.readthedocs.io/" in readme
    assert "Researcher guide" in readme
    assert "Developer guide" in readme
    assert "Python API" in readme
    assert len(readme.splitlines()) <= 180
    assert "### Create a custom JAX component" not in readme
    assert "### Run a host-side component" not in readme


@pytest.mark.fast_always
def test_project_resources_publish_maintained_markdown() -> None:
    """Expose active project guides without publishing archives."""
    source = (DOCS_ROOT / "project-resources.rst").read_text(encoding="utf-8")
    for page in (
        "migration-0.3-to-0.4",
        "plugin-authoring",
        "release-notes-0.4.3",
        "release-notes-0.4.2",
        "release-notes-0.4.1",
        "release-notes-0.4.0",
        "releasing",
    ):
        assert page in source
```

Run the two tests and observe RED because the README still contains the
detailed manuals and lacks the canonical learning-path links. The resource
inventory assertion already passes and protects the MyST navigation while the
README is simplified.

- [ ] **Step 2: Publish maintained project Markdown through one resource page**

Create `docs/project-resources.rst` with short descriptions and:

```rst
.. toctree::
   :maxdepth: 1
   :caption: Migration and extension

   migration-0.3-to-0.4
   plugin-authoring

.. toctree::
   :maxdepth: 1
   :caption: Releases

   release-notes-0.4.3
   release-notes-0.4.2
   release-notes-0.4.1
   release-notes-0.4.0
   releasing
```

Do not include progress archives, over-engineering audits, the API architecture
review, `docs/README.md`, or `docs/superpowers/`.

- [ ] **Step 3: Replace the README with a concise gateway**

Keep:

- title and CI/coverage badges;
- a two-paragraph project description;
- 6–10 key capabilities;
- Python version requirements;
- core, JCM, and Veros installation commands;
- the minimal slab-ocean example sourced from the same public calls as
  `docs/_examples/quickstart.py`;
- the current `0.4.3` stability/migration note;
- a short optional-dependency warning for CAMulator; and
- links labelled `Researcher guide`, `Developer guide`, `Python API`,
  `Migration guide`, and `Plugin authoring`.

Use the canonical base URL `https://vercor.readthedocs.io/` and version-neutral
`en/latest/` paths for the three main guides. Remove detailed custom-component,
host-backend, exchange, output, configuration, and troubleshooting sections
that now belong to Read the Docs. Keep the README at or below 180 lines.

- [ ] **Step 4: Run README, Markdown, example, and Sphinx checks**

Run:

```bash
conda run -n scipy pytest \
  tests/test_docs_build.py tests/test_documentation_examples.py \
  tests/test_api_architecture_review.py \
  -q -n0 --tb=short
```

Expected: README snippets/import boundaries, MyST pages, executable examples,
and strict Sphinx build all pass.

- [ ] **Step 5: Commit canonical documentation ownership**

```bash
git add README.md docs/project-resources.rst tests/test_docs_build.py
git commit -m "docs: make Read the Docs canonical"
```

---

### Task 7: Final documentation review, progress record, and full verification

**Files:**

- Modify: documentation pages found inaccurate during review
- Modify: `PROGRESS.md`

**Interfaces:**

- Consumes: all completed documentation tasks.
- Produces: verified canonical documentation and durable project status.

- [ ] **Step 1: Review every page as both target audiences**

Read the built navigation in order and confirm:

- the landing page makes the researcher/developer choice obvious;
- the researcher path reaches a successful run before advanced concepts;
- the developer path distinguishes data, host, and JAX components;
- every tutorial states prerequisites, expected result, and next step;
- repeated explanations link to one canonical page;
- API imports match current public owners; and
- no page recommends private imports or optional dependencies as core.

Fix wording, cross-references, and navigation defects with focused edits.

- [ ] **Step 2: Run static documentation hygiene**

Run:

```bash
rg -n --pcre2 \
  'vercor\._(?!\*`` modules are excluded from this reference\.)|TBD|TODO|FIXME|placeholder|implement later' \
  docs \
  --glob '!docs/superpowers/**' \
  --glob '!docs/progress-archive-*.md' \
  --glob '!docs/*audit*.md' \
  --glob '!docs/api-architecture-review.md' \
  --glob '!docs/README.md'
git diff --check
```

Expected: no private import in user documentation, no unfinished marker, and no whitespace error. Mentions that explicitly warn readers not to import private modules are allowed only if they contain no import statement.

- [ ] **Step 3: Run focused documentation verification**

Run:

```bash
conda run -n scipy pytest \
  tests/test_docs_build.py tests/test_documentation_examples.py \
  tests/test_api_architecture_review.py tests/test_plugin_architecture.py \
  tests/test_distribution_boundaries.py \
  -q -n0 --tb=short
```

Expected: all selected tests pass.

- [ ] **Step 4: Run formatting and static checks**

Run:

```bash
conda run -n scipy black --check vercor examples tests docs/_examples
conda run -n scipy flake8 . --count --max-line-length=120 --statistics
conda run -n scipy mypy vercor examples tests
conda run -n scipy python -m compileall -q vercor examples tests docs/_examples
```

Expected: all commands exit 0. If Black reports only its known Python
3.13/configured-target advisory while returning 0, record that advisory rather
than changing unrelated formatting policy.

- [ ] **Step 5: Run the complete fast suite**

Run:

```bash
conda run -n scipy pytest tests/ -q --fast \
  --dist=loadscope --max-worker-restart=0 --durations=25 --tb=short
```

Expected: the full fast selection passes with only already-known third-party warnings.

- [ ] **Step 6: Record durable progress**

Add a dated entry near the top of `PROGRESS.md` stating:

- Read the Docs is now the canonical detailed manual;
- researcher and developer learning paths are published;
- data, host, JAX, coupling, running, output, and troubleshooting guides are present;
- API reference is split into stable and advanced tiers;
- all five tutorial programs execute from public interfaces; and
- exact focused/static/fast verification counts and warnings from Steps 3–5.

- [ ] **Step 7: Re-run final changed-file checks**

Run:

```bash
git diff --check
conda run -n scipy pytest tests/test_docs_build.py \
  tests/test_documentation_examples.py -q -n0 --tb=short
```

Expected: no whitespace errors and all final documentation tests pass.

- [ ] **Step 8: Commit the verified documentation handoff**

```bash
git add README.md PROGRESS.md docs tests/test_docs_build.py \
  tests/test_documentation_examples.py
git commit -m "docs: complete VerCOR project documentation"
```

The staged set must contain only the final documentation review fixes and
`PROGRESS.md`; earlier task commits already own their files. Inspect
`git diff --cached --stat` and `git diff --cached --check` before committing.
