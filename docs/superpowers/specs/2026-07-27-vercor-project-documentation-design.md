# VerCOR Project Documentation Design

## Goal

Make Read the Docs the canonical detailed documentation for VerCOR while
keeping the repository README as a concise project overview and entry point.
Serve Earth-system researchers and Python/JAX developers through separate
learning paths that converge on shared how-to guides and a curated public API
reference.

## Audience and documentation ownership

The documentation has two primary audiences:

1. Earth-system researchers who need to understand what VerCOR does, install
   it, run a bundled configuration, inspect results, and configure output.
2. Python/JAX developers who need to understand VerCOR's public contracts,
   connect their own data or model, preserve runtime-state invariants, and
   retain JAX transformation compatibility where applicable.

Read the Docs owns tutorials, concepts, component-authoring guidance,
troubleshooting, and API reference material. `README.md` retains only the
project summary, key capabilities, installation commands, one minimal example,
the current compatibility note, and prominent links to the canonical learning
paths and API reference.

## Information architecture

The Sphinx navigation is organized as follows:

- **Introduction**
  - what VerCOR is;
  - the problem it solves;
  - its differentiable, component-based architecture; and
  - a concise description of clocks, components, exchanges, grids, runtime
    state, and output.
- **Researcher learning path**
  - installation;
  - a first runnable slab simulation;
  - running and inspecting a coupled simulation; and
  - output and common configuration.
- **Developer learning path**
  - core concepts and public contracts;
  - adding a data component;
  - adding a host component;
  - adding a differentiable JAX component; and
  - coupling components with exchanges and regridding.
- **How-to guides**
  - running repository examples;
  - selecting execution backends;
  - configuring output; and
  - troubleshooting common failures.
- **Python API**
  - a stable user-facing reference grouped by purpose; and
  - a separate advanced reference for runtime, workflow, backend, and topology
    contracts.
- **Project resources**
  - the maintained migration guide;
  - the plugin-authoring guide;
  - release notes for the supported 0.4 releases; and
  - the contributor release guide.

The landing page briefly introduces the project and routes readers into the
researcher or developer path. Each path follows
“understand, install, run, inspect, extend” and ends with links into the shared
how-to and API sections.

## Content design

### Introduction and project description

The introduction explains VerCOR as a JAX-first Earth-system coupler rather
than assuming that readers already understand coupling terminology. It
distinguishes component models, forcing data, exchanges, regridding, the shared
clock, and immutable runtime state. It also states the important boundary:
output-free JAX workflows can be differentiated end to end, while a mixed
host/JAX workflow uses the host execution path.

The architecture description remains conceptual. Detailed ownership and
extension contracts stay in the developer path and API reference.

### Researcher learning path

The researcher quick start uses a small bundled slab component, a deterministic
grid, and a short clock. It requires no external datasets or optional model
packages. The guide shows installation, assembly through `Coupler`, execution,
and inspection through `RunState`.

The follow-on running guide explains component order, exchanges, output
selection, and where bundled examples need optional packages or external data.
It does not present machine-specific example paths as portable commands.

### Developer learning path

The component-authoring material is split into three independent guides:

- **Data component:** construct `DataComponent` with static or time-dependent
  forcing, declare fields, and explain transfer selection.
- **Host component:** wrap a functional Python model with
  `CallableComponent`, declare `execution="host"`, initialize payload through a
  lifecycle hook, and return replacement payload state.
- **JAX component:** implement a pure array step through `CallableComponent` or
  the structural `Component` protocol, run with the JAX backend, and explain
  PyTree, shape, dtype, JIT, differentiation, and `output=None` constraints.

The coupling guide composes small components with `Exchange`, public
regridder factories, declared fields, stable route IDs, and explicit run order.
The guides reuse one small conceptual model so the reader learns new contracts
without repeatedly learning new physics.

### How-to and troubleshooting material

Task-focused pages cover backend selection, output configuration, and safe use
of the repository examples. Expected failures are explained beside the
relevant action and summarized on the troubleshooting page:

- missing optional JCM, Veros, or CAMulator dependencies;
- exchange fields not declared by both endpoints;
- host components forced onto the JAX backend;
- payload PyTree structure, shape, or dtype changes in compiled execution; and
- file output requested inside a transformed JAX computation.

The troubleshooting page links back to the full conceptual explanation instead
of duplicating it.

## Python API design

The main API reference is curated from explicit public exports. It does not
recursively document private modules or treat incidental import paths as
supported API.

The stable user-facing section groups the root assembly objects and the public
owners needed for ordinary use: clocks and coupling, components, grids and
exchanges, state, output, regridding, physical constants, bundled setups,
diagnostics, and public array/types contracts. Each group starts with a short
usage description before the generated signatures and docstrings.

Runtime options used during ordinary `Coupler` construction remain visible in
the main reference. Custom workflow, execution backend, runtime-driver, and
topology-policy contracts are documented under an explicitly labelled
advanced section. This keeps the common path compact without hiding supported
extension points.

## Source and example strategy

New narrative pages use reStructuredText, matching the current Sphinx project.
Add MyST Parser to the documentation requirements and configure Sphinx to build
both `.rst` and `.md` sources. This publishes the existing migration,
plugin-authoring, 0.4 release-note, and contributor release guides in the site
without copying them into parallel reStructuredText files. Generated API pages
use Sphinx autodoc/autosummary and explicit module/member lists.

Runnable tutorial programs live under `docs/_examples/`. The documentation
includes them with `literalinclude`, so displayed code and tested code are
identical. The initial set covers:

- the researcher quick start;
- a data component;
- a host component with functional payload state;
- a differentiable JAX component; and
- two coupled components with an exchange.

Examples use small in-memory arrays, deterministic assertions, no network
access, no external data, and no file output unless the page is specifically
teaching output. They import only documented public interfaces.

## Reader experience

Every tutorial begins with prerequisites and a statement of what the reader
will build. It ends with the expected result, a brief explanation of what
happened, and links to the next relevant page. Researcher pages introduce
scientific concepts before implementation details. Developer pages state
contracts, shapes, runtime behavior, and differentiation constraints
explicitly.

Navigation labels use reader tasks rather than internal package names wherever
possible. API module names remain visible within the reference because readers
need exact import paths.

## Testing and verification

Documentation changes follow the repository's test-first policy:

1. Add focused documentation contracts that initially fail because the full
   navigation, canonical README links, example inventory, or API pages are
   absent.
2. Add executable tests for every script in `docs/_examples/`.
3. Build Sphinx HTML with warnings treated as errors.
4. Verify internal toctree and cross-reference targets through the strict
   Sphinx build.
5. Verify that API pages list intended public owners and do not reference
   private VerCOR modules.
6. Run the repository fast test suite after focused documentation checks pass.

`PROGRESS.md` records the completed documentation scope and exact verification
result.

## Acceptance criteria

The work is complete when:

- the landing page clearly routes researchers and developers;
- the introduction and project description explain VerCOR without requiring
  prior code knowledge;
- a new user can install and run the dependency-light quick start;
- the site separately documents data, host, and JAX component authoring;
- the running, output, coupling, and troubleshooting guidance is discoverable
  from the main navigation;
- the stable user-facing API and advanced runtime/topology API are visibly
  separated;
- tutorial code is sourced from passing executable examples;
- Sphinx builds HTML with warnings treated as errors;
- the existing fast suite remains green;
- the README is concise and points readers to the canonical site; and
- no private module is presented as supported user API.

## Scope boundaries

This project changes documentation sources, documentation examples,
documentation tests, Sphinx documentation dependencies/configuration, the
README, and `PROGRESS.md`. It does not redesign VerCOR's runtime API, change
physics or numerics, add a component registry, add optional model
dependencies, or attempt to make external-data examples self-contained.
