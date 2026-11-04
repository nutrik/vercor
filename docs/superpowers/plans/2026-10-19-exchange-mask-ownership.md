# Exchange mask ownership implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Run released Veris v0.2.1 and JCM 3.0.0 through one explicit,
component-owned exchange coverage model.

**Architecture:** Keep grid geometry immutable, component coverage declared by
ComponentMaskSpec, binary restrictions in topology, and coverage/overlap in the
existing runtime. Reuse the explicit quantity dispatcher for scalars and
vectors, migrating each setup and its native consumer together.

**Tech Stack:** Python 3.12, conda scipy, JAX, pytest, Veris, JCM, Veros, gh.

**Spec:** `docs/superpowers/specs/2026-10-19-exchange-mask-ownership-design.md`
(approved 2026-10-19).

## Global constraints

- Work on `feat/veris-seaice`; preserve unrelated edits; no push/merge/tag/release.
- Veris v0.2.1 resolves to `6e3b56ee14922f4bae760d57772f610078aa1886`.
- JCM is `3.0.0`, Dinosaur is `>=1.5.0`; user-approved combined validation uses JAX `0.11.2`, overriding only JCM's incompatible upper bound if all validation passes.
- JCM-backed runs use Gregorian clocks and explicit native RunState time/step;
  public PhysicsState humidity is already kg/kg.
- Stock Veros and the pinned autodiff fork run in separate processes.
- Required quantity values are `intensive`, `source_flux`, and `target_flux`.
- Coverage enters source products before remapping, once; preweighted fields
  retain binary source coverage. No new mask framework or scientific tuning.
- Preserve current versus last-dispatch masks, inactive-NaN safety, JIT/JVP/VJP,
  immutable state, public module ownership, native face conventions and order.
- Test before changing behavior; fast checks while iterating; full unit tests
  must pass before every commit, including design/documentation commits.
- Update PROGRESS.md and the plan after meaningful work. Put logs/artifacts and
  pytest basetemps under `/private/tmp` or ignored `test_logs`.

## Review focus

1. Mixed coastal cells must retain correct land/ocean physics without squared
   area factors or summed intensive temperatures.
2. Zero coverage and inactive vector NaNs must yield finite values and
   derivatives, including after source reactivation before a recipient runs.
3. Static coverage in daily/monthly forcing must remain grid-shaped in sends,
   priming and output; record-shaped masks must fail validation.
4. A topology patch must never activate a dry target cell or own a fraction.
5. Native JCM API/dependency changes must be exercised by actual coupled runs,
   rather than only mocks; installed Veris provenance must match the release.

## Task 1: Upgrade and validate native dependency contracts

**Files:** `pyproject.toml`, `.github/workflows/python-package.yml`,
`tests/test_veris_gallery.py`, `tests/test_distribution_boundaries.py`,
relevant native JCM tests and `vercor/setups/_external/jax_gcm*.py`.

**Interfaces:** Keep public setup factories/configuration and immutable native
payload ownership; adapt released JCM state/forcing APIs only as necessary.

- [x] Add failing release/version/provenance and dependency-resolution contracts.
- [x] Run the exact failing tests and record RED evidence.
- [x] Update package extras and CI provenance; resolve/install dependencies in
  the scipy environment after the old-environment baseline finishes.
- [x] Exercise released native JCM bootstrap/step/output/calendar and Veris
  setup/step. Add regressions for demonstrated API changes before fixing them.
- [x] Migrate the JCM native resumable clock and humidity units. Test leap-day
  and chunk continuation, reject incompatible calendars, update JCM-backed
  gallery clocks and retain native gram-unit diagnostic conversions.
- [x] Run related native tests; record exact versions and installation receipts.
  Defer commits until the full suite is green.

## Task 2: Finish single runtime ownership and dispatch

**Files:** `vercor/exchanges.py`, `vercor/topology.py`,
`vercor/_runtime/{exchange_topology,topology_policy,topology_state,surface_masks,
dispatch_context,prepared,preparation,exchange_masks,exchange_dispatch}.py`,
`tests/test_dynamic_mask_exchanges.py`, `tests/test_runtime_exchange.py`,
`tests/test_dynamic_mask_rollouts.py`, affected topology/public API tests.

**Interfaces:** `Exchange(..., *, quantity: Literal["intensive", "source_flux",
"target_flux"], source_coverage="component")`; topology maps contain regridders
and binary restrictions; runtime context supplies component masks and geometry.

- [x] Add RED tests for required quantities, removed fractional topology
  ownership, target-support restriction, scalar/vector parity, nonuniform
  coastal AD and empty-overlap JIT/JVP/VJP.
- [x] Remove legacy enum/default/dispatch and topology fractional plumbing;
  require runtime dispatch context and apply the spec's three formulas.
- [x] Preserve current and received coverage stores and existing unequal-order,
  supplied-state, inactive-NaN, binary-export and reactivation regressions.
- [x] Migrate generic test fixtures to explicit intended semantics. Replace
  tests tied to deleted helpers with behavioral contracts, preserving unique
  assertions. Run focused exchange/topology/runtime tests.

## Task 3: Make static land fractions component-owned

**Files:** `vercor/components/contracts.py`,
`vercor/_runtime/field_transfer.py`, `vercor/output/_session.py`,
`vercor/setups/_data/jcm_land.py`,
`vercor/setups/_external/camulator_land.py`, `vercor/setups/_jcm.py`,
existing component, output, lifecycle and JCM surface tests.

**Interfaces:** Extend `select_runtime_field` with explicit field-name/mask
metadata so declared current masks bypass daily/linear record selection.
Both send and output call sites use that selector. Land exposes a declared
fractional coverage field through existing ComponentMaskSpec.

- [x] Add RED daily and linear forcing tests covering current mask fields,
  priming, sends and output; reject record-shaped masks.
- [x] Remove the blanket current-only transfer restriction and pass declared
  mask metadata through the two selector call sites, retaining mask validation.
- [x] Retain computed JCM/CAMulator land fractions; reuse land-owned coverage
  for JCM terrain in both modes. Avoid separately recomputing its fraction.
- [x] Run component-mask, time-selection, output and land-factory tests.

## Task 4: Migrate native consumers and all bundled declarations

**Files:** `vercor/setups/_external/{jax_gcm_fields,jax_gcm_runtime,jax_gcm,
camulator_fields}.py`, `vercor/setups/gallery/*.py`, `vercor/recipes.py`,
examples and external extension fixture, affected setup/gallery tests.

**Interfaces:** Ordinary JCM gets independent intensive native land temperature,
SST and moisture. JCM3 has no published land-specific turbulent channels; remove unused prescribed-land flux routes. CAMulator receives remapped
source-area temperature contributions. Existing coupled JCM marine products
and total Veris feedback retain their audited binary/component conventions.

- [x] Add RED fractional-coast tests for ordinary JCM native forcing channels,
  total-temperature diagnostics and soil feedback; add CAMulator source-product
  remapping tests and executable route-semantic contracts.
- [x] Update ordinary JCM fields/native forcing and remove unavailable land feedback channels;
  remove the old intensive-temperature sum path.
- [x] Migrate CAMulator contribution routes/consumer without changing its
  established ML-mask branch. Use conservative scalar area contributions.
- [x] Audit and declare every gallery/example/fixture field's quantity; split
  mixed-semantics routes. Keep local flux densities intensive and preweighted
  total ocean feedback source_flux with binary coverage. No bundled consumer
  needs target_flux merely to exercise the enum.
- [x] Run all setup, recipe, native-surface, gallery and contract tests; document
  intentional changed expectations and run the fast stock suite.

## Task 5: Complete verification, documentation and delivery

**Files:** `DESIGN.md`, `DEPENDENCIES.md`, `PROGRESS.md`, exchange/topology/setup
docs, architecture manifests, this plan/spec, task evidence under test_logs.

**Interfaces:** All source and installed artifacts must use the same dependency
versions, route contracts and native consumer behavior.

- [x] Record final field-semantic table and intentional numerical/API changes.
  Update active dependency/docs/CI contracts, retaining historical evidence.
- [x] Run Black, strict flake8, mypy, compileall and whitespace checks. Run the
  fast and full stock suite with `-m 'not veros_autodiff'`, and all three native
  AD test files in the exact-fork environment with `-n0 -m veros_autodiff`.
- [x] Collect combined coverage; build wheel/sdist, check Twine and strict
  Sphinx, and exercise installed-package/extension boundaries.
- [x] Complete actual short native coupled JCM/Veris/Veros runs and AD/finite
  difference checks with the release versions. Verify evolved finite fields,
  physical budgets, native output and LND/CICE/OCN/ATM order.
- [x] Obtain an independent whole-branch code/science review under the chosen
  execution workflow, resolve material findings, and rerun affected gates.
- [x] Record exact final evidence/limitations in PROGRESS.md; commit coherent
  changes on feat/veris-seaice only after full unit tests pass. Verify clean
  status, commit SHA and GitHub CLI branch/repository state; report completion.

## Completion evidence

Completed 2026-10-19 on feat/veris-seaice. Full stock2209/2209, fast1284/1284,
exact-fork AD15/15, no skips, combined coverage92.48%. Native source and final
wheel CLI-copy global2day runs agree exactly (2319wet/48native/9routes/13files).
Static, strict docs, package and installed-extension gates pass. Review found
one stale AD fixture route; its contract failure was reproduced and corrected
without weakening gradient assertions. JCM's sole approved JAX metadata
exception remains explicit. See docs/exchange-coverage.rst and PROGRESS.md.

Execution notes: the native JCM release API and exchange consumer migration land
atomically because old tile/clock contracts cannot validate the new consumers.
The shared full suites validate all tasks; task completion checks consume their
fresh XML receipts and confirm unchanged Python bytes rather than repeat them.
The user-selected checkout is retained, with no remote integration requested.
