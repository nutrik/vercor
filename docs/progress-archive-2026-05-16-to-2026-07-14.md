# VerCOR Progress Archive: 2026-05-16 to 2026-07-14

This file preserves the detailed execution log that was active through the
VerCOR 0.4 Task 8 milestone. Earlier history remains in
`progress-archive-2026-04-23-to-2026-05-15.md`.

Use this file to decide what to do next. Use the archive when you need exact
historical commands, failure messages, or detailed validation notes.

## Current Status

- VerCOR 0.4 milestone 1 Task 8 completed locally on 2026-07-14 from base commit
  `b411c35`. The independently installed v0.4 plugin now owns frozen
  configuration and an injected assembly factory, supplies a structural scalar
  regridder on an explicit route ID, returns a non-empty topology patch, builds
  a public workflow, drives the validated runtime through a custom backend, and
  exercises per-step period output, snapshot output, lifecycle identity, and
  immutable `RunState.replace_fields()` without private VerCOR imports. A new
  installed-wheel smoke proves the dependency-free default slab-ocean factory
  returns and runs an ordinary v0.4 structural component. Inventory confirmed
  that Tasks 3-7 had already migrated bundled slab/data/external factories,
  native JAXGCM/Veros/CAMulator providers, every example, and the README
  quick-start contracts, so Task 8 avoided redundant production rewrites and
  tightened the remaining plugin/installed-artifact acceptance boundary.
  The focused setup/example/plugin plus JAXGCM/Veros selection passes 114/114,
  the bounded real one-step JCM+Veros example passes with JCM 1.1.1 and Veros
  1.6.2, the fast suite passes 481/481, and the full suite passes 1067/1067
  with only the two known third-party `FutureWarning`s. Black leaves all 239
  Python files unchanged, strict flake8 reports zero findings, full mypy
  reports no issues in 235 source files, compileall and whitespace checks pass,
  and the installed plugin/default-slab artifact pair passes 2/2. The mandated
  `conda run -n scipy` orientation command again panicked before pytest in
  `conda_rattler_solver` with `pyo3_runtime.PanicException: Attempted to create
  a NULL object.`; all actual checks used the approved direct `scipy` Python.
  One over-broad Black invocation included `README.md`, which Black rejected as
  non-Python without changing files; the corrected Python-only command passed.
- Veros native-variable representability regression fixed locally on
  2026-07-14. The
  native provider now resolves each supported active variable's dimensions
  once, excludes `line_psin`, whose repeated `("isle", "isle")` axes cannot
  satisfy the shared `OutputVariable` contract, and excludes four `Ai_*`
  variables whose `tensor1` and `tensor2` coordinates are unavailable, and
  reserves `time` for the adapter-owned coordinate. The
  focused provider regression and Veros output selection pass 6/6; a bounded one-step
  `run_jcm_with_veros.py` execution passes; the fast suite passes 480 tests with
  586 deselected, and the full suite passes all 1066 tests. Black, flake8, mypy,
  and whitespace checks pass.
- Veros output-universe regression fixed locally on 2026-07-14. The bundled
  provider now excludes setup-local state metadata such as `sss_clim` before
  resolving dimensions, keeping enumeration aligned with Veros's global output
  registry while preserving supported manifest order. Direct characterization
  coverage confirms explicit requests for excluded setup-local names remain
  unknown-variable errors. The focused provider and explicit-rejection
  regressions, focused Veros output coverage, fast suite, full suite, Black,
  flake8, mypy, and whitespace checks pass.
- VerCOR 0.4 milestone 1 Task 7 completed locally on 2026-07-14 from base commit
  `12478cd`. `vercor.output` now exposes exactly the immutable provider/frame,
  period, target, and snapshot contracts; `Coupler.run(..., output=None)` is an
  explicit no-I/O default and `OutputTarget` independently enables period,
  final-field, and snapshot output. One private coordinator owns cadence,
  selection, collision-safe filenames, host NetCDF writes, interruption scope,
  and the sole immutable JAX-PyTree sum/count accumulator. Generic runtime
  fields, JAXGCM, Veros, CAMulator, and third-party providers share identical
  `PeriodOutput.variables` filtering. Provider frames validate rank, shape,
  dtype, dimensions, coordinates, attributes, metadata, and sample/time schema;
  concrete array metadata is canonicalized for safe JIT cache equality while
  coordinate values remain dynamic PyTree leaves. All output failures are
  component/path-scoped, and traced I/O is rejected without compromising JIT or
  gradients for output-free and all-disabled runs.
  Review follow-up made Veros expose its active native variable universe,
  preserved every CAMulator model substep for period means while keeping the
  final prediction for runtime fields and native snapshots, removed obsolete
  output-owned state/configuration and test-only adapter helpers, normalized
  public `Coupler.run` annotations, and reduced signal and final-filename
  ownership to one path. Independent test-quality, gradient, code-quality, and
  documentation reviews are clean. The focused output/runtime/native suite
  passes 274 tests with two known third-party warnings; the final fast suite
  passes 480 tests with 585 deselected and the full suite passes all 1065 tests
  with the same two warnings. Black leaves all 239 files unchanged, strict
  flake8 reports zero findings, full mypy reports no issues in 235 source files,
  and compileall and whitespace checks pass. Exact evidence is in
  `.superpowers/sdd/task-7-v0_4-report.md`; the required commit title is
  `feat!: unify component output providers`.
- VerCOR 0.4 milestone 1 Task 6 completed locally on 2026-07-14 from base commit
  `ed5fccf`. The canonical public runtime surface is now the frozen workflow
  planning and chunk execution contract: `WorkflowContext`, `StepPlan`,
  `ExecutionPlan`, `SequentialWorkflow`, `ExecutionContext`, `ExecutionChunk`,
  `Workflow`, `ExecutionBackend`, `RuntimeDriver`, and `RuntimeOptions`.
  Workflows produce exactly one validated absolute plan per clock step and may
  reorder or omit registered components. The private execution coordinator
  groups uniform output-free schedules, preserves the default one-JIT/one-scan
  path, builds clock-derived metadata once per run, and reuses one run-local
  jitted executor per distinct component schedule across cadence chunks. It
  selects host/JAX from scheduled components only and keeps period-output
  sampling, writes, cancellation, and result validation core-owned. Custom
  backends consume an identity-checked ordered plan ledger through
  `RuntimeDriver.run_step`; forged, repeated, reordered, and skipped plans fail.
  Initial outgoing stores are primed for every registered component so custom
  workflows can schedule producers omitted from the default order. Full-suite
  validation exposed and fixed a JAXGCM payload precision instability when a
  float32 runtime follows process-level x64 enablement; numeric payload leaves
  now normalize at creation while opaque/static leaves remain unchanged.
  TDD reached 33 intended RED failures before implementation. Review-driven
  performance regressions additionally proved that output-enabled and
  alternating workflows previously rebuilt clock metadata and JIT executors
  per chunk. The workflow tests are split by responsibility into a 442-line
  public contract owner and a 661-line execution module, with shared helpers in
  an 89-line private support module; their exact 41-test collection is
  preserved.
  Final gradient review strengthened intermediate corrupt-state rejection to
  prove later nonempty chunks do not run and made the JAX A/B/A regression
  numerically depend on absolute step indices; both pass without production
  changes. The repaired focused workflow/runtime/plugin/docs selection passes
  163 tests with 137 deselected. The fast suite passes 475 tests with 550
  deselected, and the full suite passes 1025 tests with the two known
  third-party `FutureWarning`s. Black leaves all 238
  files unchanged, strict flake8 reports zero findings, full mypy reports no
  issues in 234 source files, compileall and whitespace checks pass. Exact
  evidence is in `.superpowers/sdd/task-6-v0_4-report.md`; the required commit
  title is `feat!: add workflow-driven execution backends`.
- VerCOR 0.4 milestone 1 Task 5 completed locally on 2026-07-14 from base commit
  `166f021`. `Exchange` now owns stable global `route_id` identity and an
  injected `regridder_factory`; default `source->target` collisions fail before
  setup or factory calls. Runtime topology maps and public topology patches are
  route-ID keyed, and `TopologyPolicy` has one `build(context)` method.
  Preparation validates the scalar `Regridder` capability and its
  scalar-plus-vector `VectorRegridder` refinement, while deterministic fan-in diagnostics name
  sorted route IDs. `RunState` exposes only `component()`, `components()`, and
  `replace_fields()` publicly; component names, grids, stores, and fractional
  masks are private PyTree state, and component indices are a derived private
  lookup. Supplied, pre-driver,
  and backend-returned states are strictly checked for exact names/order,
  grids/coordinates/edges, store and JAX-payload schemas, and finite mask
  constraints using transform-safe runtime assertions. The focused Task 5
  contract suite passes 51/51, the fast suite passes 441 tests with 550
  deselected, and the full suite passes 991 tests with only the two known third-party
  `FutureWarning`s. Final static checks and independent reviews are recorded in
  `.superpowers/sdd/task-5-v0_4-report.md`. The required commit title is
  `refactor!: add stable route and state contracts`.
- VerCOR 0.4 milestone 1 Task 4 completed on 2026-07-14 from base commit
  `638cd7a`. The primary package root is now exactly `Clock`, `Coupler`,
  `Exchange`, `RectilinearGrid`, `RunState`, and `RuntimeOptions`; advanced
  component/runtime/physics/grid/output/setup contracts use their canonical
  owner modules. Assembly is constructor-only with owned immutable collection
  snapshots for components, exchanges, and run order; read-only public views
  retain the original author objects, which are treated as immutable
  configuration. Primary
  `vercor.coupling`, `vercor.settings`, `vercor.physical_constants`,
  `vercor.host_arrays`, `vercor.pytree`, and `vercor.interpolators` are removed;
  their retained implementations are either canonical (`vercor.coupler`,
  `vercor.physics`) or private (`vercor._host_arrays`, `vercor._pytree`,
  `vercor._interpolators`). Runtime setup/step/topology contexts and private
  preparation no longer carry Settings or a reflective configuration snapshot.
  An empty run order is explicit setup-only behavior and does not synthesize an
  execution order. At that milestone, output still remained on
  `OutputConfig`/`PeriodOutput`, `Coupler.run()`, and
  `Coupler.write_outputs()`; custom backends still remained
  on `RuntimeOptions.execution` plus `ExecutionBackend.run`. Route IDs were
  outside Task 4; workflow, unified output-provider, and `vercor.compat.v0_3`
  APIs were not yet implemented. Final review closed imported-object/module namespace leaks,
  public annotation resolution, eager graph validation, strict run-order
  typing, installed-wheel manifest coverage, and documentation ownership
  accuracy. The import-order audit covers all 149 production Python modules
  exactly once, with no later-layer edges and only the two documented strongly
  connected pairs. Independent test-quality, gradient, code-quality, and
  documentation reviews are clean. Black, strict flake8, full mypy (230
  files), compileall, and whitespace gates pass; the fast suite passes 417
  tests with 522 deselected, and the full suite passes 939 tests with only the
  two known third-party `FutureWarning`s. The Task 4 commit title is
  `refactor!: simplify the VerCOR public assembly API`.
- VerCOR 0.4 milestone 1 Task 3 protocol-first component authoring completed
  locally on 2026-07-14. `vercor.components.Component` is now the single
  runtime-checkable structural protocol (`name`, `grid`, `spec`, `step`), with
  `CallableComponent` and `DataComponent` as the only concrete convenience
  adapters. Frozen `ComponentSpec` exclusively owns inputs, outputs, initial
  fields, execution capability, lifecycle, transfer, and output policy;
  `LifecycleHooks.setup` returns an immutable-mapping
  `SetupResult(fields, payload)` once during private binding preparation;
  standard payload containers are rebuilt per state, NumPy leaves are copied,
  and opaque object leaves are deep-copied or rejected if they cannot be owned;
  validation callbacks likewise receive a copy-owned payload and cannot mutate
  initial or caller-supplied runtime state.
  The inherited/mixin authoring hierarchy,
  `ComponentLike`, `HostComponent`, component `initialize`/`initial_fields`,
  constructor payloads, duplicate output/import-policy properties, and their
  dead private helper modules are removed from the primary v0.4 implementation.
  Prefill stores are declaration-, shape-, and dtype-normalized before atomic
  update; scanned payload replacements must retain the setup payload PyTree
  structure, while host execution may clear or restructure it. RED coverage
  includes structural/callable/data authoring, invalid names/mappings,
  setup/prefill/step declarations and layouts, scalar expansion, defensive
  NumPy ownership, nested payload ownership, immutable mappings, payload
  preserve/clear/replace PyTrees, and setup-payload JVP/reverse gradients. The
  current installed plugin is migrated to strict-mypy v0.4 authoring; the frozen
  legacy plugin and pre-redesign compatibility baseline remain unchanged and
  demonstrate the intentional break. Final verification passes the exact
  202-test
  component/API/plugin/gradient focus, all 416 fast-selected tests, and all 872
  full-suite tests with only the two known third-party `FutureWarning`s. Black
  leaves all 237 files unchanged, strict flake8 reports zero findings, full
  mypy reports no issues in 233 source files, and compileall passes. Exact
  commands and evidence are in `.superpowers/sdd/task-3-report.md`.
- VerCOR 0.4 milestone 1 Task 2 physical configuration and precision ownership
  completed locally on 2026-07-13. `vercor.physics.PhysicalConstants` is the
  frozen registered PyTree owner for all 25 traced physical values, using
  canonical descriptive names while preserving the legacy numerical defaults.
  Setup and step contexts carry runtime-normalized constants, and
  `RuntimeOptions.dtype` is the sole precision owner: dtype helpers reject
  `Settings`, runtime preparation casts constants and component fields at one
  boundary, and the float64-constant/float32-runtime regression proves both
  float32 kernel execution and a preserved reverse gradient. Flux, ERA5,
  JAXGCM, CAMulator, Veros, component-authoring, and exchange-topology paths now
  receive physics and precision separately; production contains zero direct
  physical reads from `Settings` and zero `Settings` arguments to dtype
  helpers. Task 4 has since removed the primary Settings and legacy physical-
  constants modules; future legacy compatibility remains explicitly deferred.
  TDD began with 7/7 intended missing-
  module failures; the later mixed-precision regression also failed before the
  runtime boundary cast. Final focused physics tests pass 8/8, the requested
  precision-owner cluster passes 63/63, the fast suite passes 467 selected
  tests, and the full suite passes 868 tests with only the two known third-party
  `FutureWarning`s. Black, strict flake8 (0), full mypy (240 files), compileall,
  and whitespace checks pass. Detailed evidence and the exact legacy-to-
  canonical field map are in `.superpowers/sdd/task-2-report.md`.
  Review follow-up on 2026-07-13 connected the previously ignored momentum and
  air-temperature reference heights to both ocean and sea-ice flux kernels,
  made the 25-field constants API keyword-only, documented every field and
  unit, and added immediate `RuntimeOptions.dtype` validation. Zero-dimensional
  NumPy values are now copied to immutable real numeric scalars, non-scalar and
  nonnumeric NumPy/JAX leaves are rejected without materializing valid JAX
  scalars or tracers, and replacement of `coupler.constants` after preparation
  remains mutation-checked without hashing array contents. The review TDD runs
  recorded 9 intended behavior/ownership/construction/documentation failures
  plus 4 intended nonnumeric-scalar failures before production edits. The
  corrected physics/flux focus passes 36 cases, the broader physics/flux/runtime
  selection passes 71, and the optional setup/kernel selection passes 20. The
  post-review repository gates pass all 470 fast-selected and 882 full tests
  with only the two known third-party `FutureWarning`s. Black leaves all 240
  files unchanged, strict flake8 reports zero findings, full mypy reports no
  issues in 240 files, and compileall plus whitespace checks pass.
- VerCOR 0.4 milestone 1 Task 1 compatibility baseline implemented locally on
  2026-07-13. The static manifest pins clean reference commit
  `9f0b9131c889bed5c1c2d8ded260add3cfef9524`, the pre-redesign version, the exact
  48-symbol root surface, 13 canonical owner-module export lists, and 37 public
  signatures needed by the future legacy adapter. The focused test recorded the
  intended missing-manifest RED. Review follow-up then recorded two intended
  failures for the omitted setup/protocol contract and missing clean-reference
  builder. The non-optional test now creates a clean source tree with
  `git archive` of the pinned commit, builds its wheel offline through the
  established distribution cache fallback, and inspects it in an isolated
  bounded subprocess; the evolving live checkout is not compared with the
  pre-redesign manifest, and the later legacy fixture is rejected as baseline
  evidence. A
  follow-up CI audit reproduced `fatal: not a tree object` for the reference
  SHA in a real depth-1 clone and recorded the intended workflow-boundary RED
  (`fetch-depth` absent). The quality job is the only current job that collects
  the baseline test through its full and coverage suites, and its checkout now
  uses `fetch-depth: 0`; an executable YAML assertion protects that boundary.
  A full-history clone archived the exact SHA and passed the assertion plus all
  three baseline tests (4/4). Final default validation passes 464 tests with
  397 deselections in fast mode; the full suite passes 861 tests with only the
  two known third-party `FutureWarning`s.
- Pre-redesign API hardening documented on 2026-07-13. Task 1 established the
  single component contract validator, authoritative `ComponentSpec.lifecycle`,
  and a non-materializing structural prepared-configuration snapshot (focused
  148/148). Lifecycle/spec callables are identity-only so validation event logs
  and counters do not false-invalidate prepared reuse; hidden closure/global/
  default mutable state is outside the supported configuration contract. Task 2
  added deterministic exchange fan-in rejection, legal feedback
  overlap, shape-stable state/backend schemas, strict topology masks, and
  settings/payload gradient regressions (focused 93/93). Task 3 proved a
  superseded installed plugin and a frozen legacy installed plugin and added
  strict CI quality/coverage
  enforcement (focused 50/50; historical branch coverage 90.53%). Task 4's
  strengthened documentation contracts pass 10/10, including exact
  heading/export inventories, line-break-independent stale-API guards, bounded
  installed-plugin proof, and accurate custom-backend comparison scope; its
  broader API/documentation/distribution selection passes 173 cases. Final
  integrated verification reports that Black, strict flake8, focused mypy, and
  whitespace checks pass; the fast suite passes 461/461 with 397 deselected; and
  the full suite passes 858/858 with only the two known third-party
  FutureWarnings.
- Pre-redesign Task 2 runtime-state semantics completed locally on 2026-07-13.
  Exchange contract construction now rejects deterministic scalar/vector fan-in
  conflicts while allowing receive/step/send feedback fields; the slab driver
  retains only its bilinear ocean-to-sea-ice temperature route. Runtime field
  replacement preserves shapes, malformed step returns raise component-oriented
  errors, supplied states require the exact registered component set and validate
  every component, and custom backends return a schema-compatible `RunState`
  independent of originating coupler identity. Topology patches now require
  concrete finite numeric/bool target-shaped masks with binary/fractional range
  checks. Focused tests recorded 24 intended failures from 29 cases before
  production edits plus one additional extra-field schema RED; settings/payload
  gradient and compatible-state acceptance regressions stayed GREEN. Final
  focused ownership verification passes 93 cases, Black leaves all 13 affected
  files unchanged, strict flake8 reports zero findings, mypy reports no issues
  in 13 affected files, the fast suite passes 446/446, and the full suite passes
  842/842 with only the two known third-party FutureWarnings. The final Conda
  launcher attempt hit the known Rattler NULL-object panic before pytest; direct
  execution with the `scipy` environment Python passed all final suites.
- Superseded API consolidation release validation completed locally on
  2026-07-10 with the direct
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python` fallback for the known
  Conda/Rattler launcher panic. The exact eight-section architecture review is
  in `docs/api-architecture-review.md`; README quick starts cover the built-in,
  structural JAX, host, custom backend, custom topology, and lifecycle/output
  workflows using public imports only. The release owns exchange recipes
  directly in `vercor.recipes`, uses `ComponentStepReturn` on every public step
  callback annotation, and reports the superseded release version. Contract tests recorded the
  expected five-test RED before the document/version changes and now pass 5/5;
  the native Veros/CAMulator output regression recorded two expected RED
  failures and now passes 3/3, including a mixed native/generic plan. The first
  full suite exposed nine stale full-only fixtures: invalid undeclared exchange
  fields, missing run orders for validated output, direct post-preparation
  component mutation, prepared-mapping identity assumptions, and order-dependent
  Veros global settings. The fixtures now configure valid public contracts
  before preparation or isolate the global setting helper; the exact nine-node
  reproduction passes 9/9 without weakening runtime validation. Final Black
  reports 233 files unchanged (with the recurring Python 3.13/target-3.14
  warning), flake8 reports zero findings, mypy reports no issues in 233 files,
  compileall passes, the injected/preloaded JCM smoke passes 9/9, the fast suite
  passes 386/386, the full suite passes 782/782, and coverage is 90.52% (91%
  reported) against the 90% gate. Distribution checks pass 10/10. Retained
  artifacts in the historical distribution-build directory are the superseded
  baseline wheel,
  the superseded baseline source archive, and the 0.1.0 public plugin wheel;
  wheel/sdist `py.typed`, installed origin/version checks, plugin
  smoke, and strict external mypy (4 files) pass. JCM 1.1.1 and Veros 1.6.2 are
  available; CREDIT is absent and remains uninstalled/unpinned. The final
  read-only review approved with no remaining Critical, Important, or Minor
  findings after correcting the CI wheel filename, exact 48-symbol root
  inventory, verified legacy migration history, runtime-resolvable public step
  annotations, and CAMulator period-output path wording.
- Native bundled period-output compatibility fixed locally on 2026-07-10.
  Period-enabled Veros and CAMulator factory components now mark their host
  steps as the private owner of native period writes, so generic runtime-field
  validation and schema construction do not reject model-native variables or
  create duplicate files. Period-output detection remains enabled for I/O,
  tracing, and custom-backend policy, and mixed generic/native plans retain
  the generic schema. The new three-test regression, five focused existing
  output/runtime files in fast mode (23/23), Black, flake8, focused mypy, and
  whitespace checks pass using the direct `scipy` executable.
- Task 5 final plugin-artifact follow-up implemented locally on 2026-07-10.
  VerCOR wheel/sdist and the independently packaged public-plugin wheel are now
  built once and uploaded together; every installed CI matrix cell consumes
  both wheels and never invokes plugin source-build tooling. The local
  distribution helper validates the exact plugin wheel path/name alongside the
  VerCOR artifacts, builds both packages with the same offline fallback only
  when no artifacts are supplied, and installs wheels with binary-only pip.
  A clean supplied-artifact regression disables build/flit_core/Conda fallback,
  installs both wheels, and runs the plugin successfully. The focused
  distribution suite passes 10/10 and the full fast suite passes all 379
  selected tests. Black reports 232 files unchanged, flake8 reports zero
  findings, full mypy reports no issues in 232 files, and whitespace checks
  pass. Detailed evidence is recorded in
  `.superpowers/sdd/task-5-plugin-fix-report.md`.
- Task 5 review follow-up implemented locally on 2026-07-10. The invoked Veros
  factory is now the sole owner of one runtime-configuration call before its
  implementation loader; importing Veros output, flux, or state modules has no
  configuration side effect. Distribution tests reuse explicit downloaded
  wheel/sdist paths in CI while retaining an offline local-build fallback, and
  setup probes plus public-plugin mypy resolve only through an installed root
  outside the checkout. The Python 3.12/3.13 base/JCM/Veros matrix now selects
  the paired-JCM replacement/spinup and Veros configuration/spinup regressions
  by exact node. Focused setup/distribution tests and the real artifact
  install/plugin/mypy check pass; the exact matrix selection passes 9/9, setup
  plus distribution boundaries pass 22/22, and the full fast suite passes
  382/382. Black reports 232 files unchanged, flake8 reports zero findings,
  full mypy reports no issues in 232 files, and whitespace checks pass. The
  first full-fast run exposed one stale source-shape assertion for the former
  inline Veros import; it now verifies the configuration-before-loader boundary
  instead. Detailed evidence is in `.superpowers/sdd/task-5-fix-report.md`.
- Task 5 bundled setup/packaging boundary hardening completed locally on
  2026-07-10. `vercor.setups` is the sole lazy factory registry; public import,
  config, and factory-attribute access remain free of JCM/Dinosaur, Veros, and
  CREDIT/Torch/TensorFlow imports and setup-owned environment mutation. Factory
  invocation owns optional imports/configuration; JAXGCM/Veros spinup follows
  only `Spinup.enabled`; unsupported CAMulator spinup fails before setup; paired
  JCM forcing uses `dataclasses.replace`; and the ERA5/JCM example is injectable
  with short/initial-state-only modes. Runtime metadata now separates test/dev
  tools, wheel/sdist carry `vercor/py.typed`, and the independently packaged
  public plugin exercises structural JAX/host components, original-object
  lifecycle hooks, a sequential backend, topology policy, and snapshots against
  an installed wheel. New setup boundary tests pass 12/12, example tests 6/6,
  distribution tests 5/5, and the full fast suite passes 374/374. Offline local
  artifact verification used cached Conda `build`, `flit_core`, and
  `pyproject_hooks`; direct
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python` remains the fallback
  for the known Conda/Rattler launcher panic. JCM 1.1.1 and Veros 1.6.2 were
  available locally; CREDIT was absent, so CAMulator model execution was not
  exercised. Compileall, Black (232 unchanged), flake8 (zero findings), full
  mypy (232 files), artifact install/smoke/mypy, and whitespace checks pass.
- Task 4 final collision review follow-up implemented locally on 2026-07-10.
  Period output paths are now allocated globally across all schemas and
  boundaries before stepping. Globally unique paths remain unchanged,
  same-schema sub-daily paths retain their time/step discriminator, and
  cross-schema collisions receive sanitized component/schema discriminators so
  every real NetCDF payload survives. The strict RED/GREEN regression, nine
  focused naming/cadence/backend compatibility checks, the seven-file
  output/runtime fast suite, and repository-wide fast suite pass. Black reports
  224 files unchanged, flake8 reports zero findings, mypy reports no issues in
  224 source files, and `git diff --check` passes. Detailed evidence is recorded
  in `.superpowers/sdd/task-4-collision-report.md`.
- Task 4 review follow-up implemented locally on 2026-07-10. Same-date
  step-cadence records now receive deterministic time/step filenames without
  changing unique daily names; generic non-grid dimensions are
  variable-qualified; JAXGCM coordinate caching is preparation-only and its
  payload carries exact raw-prediction sums/counts so multi-time/NaN spinup and
  runtime weights match; configured custom backends fail before invocation.
  Focused RED/GREEN regressions, the required output/runtime fast suite, and
  the repository-wide fast suite pass. Black reports 224 files unchanged,
  flake8 reports zero findings, mypy reports no issues in 224 source files,
  and `git diff --check` passes. Detailed evidence is recorded in
  `.superpowers/sdd/task-4-fix-report.md`.
- Task 4 backend-consistent period output implementation completed locally on
  2026-07-10. Generic components and JAXGCM now use private static schemas plus
  immutable JAX sum/count sessions on both host and compiled backends. Cadence
  is precomputed into coalesced scan chunks, completed reductions are written
  only between chunks, unknown generic fields fail during state creation, and
  traced I/O-enabled runs are rejected while the no-output single-scan path is
  unchanged. Required focused parity/cadence, JAXGCM-native, snapshot, tracer,
  and source-boundary checks, full-fast pytest, Black, flake8, focused mypy,
  and whitespace checks passed. The first full-fast pass exposed an inline
  runner tracer error that violated backend delegation ownership; validation
  moved to the private output-session owner and the regression passed.
- Task 3 runtime backend ownership and custom contract validation completed on
  2026-07-10. `vercor._runtime.backends` now owns compiled/pure scanned
  execution, the Python host loop, and custom backend/driver adaptation;
  `runner` is limited to execution selection, host compatibility/warnings,
  signal scope, and delegation, with no backend-to-runner import. Custom
  backends must return `RunState`, and driver calls validate state, prepared
  component membership, concrete scalar integer steps, and clock bounds before
  dispatch while preserving host components, requested timestamps, and
  interrupt checkpoints. Focused and full-fast validation passed; final static
  gates are recorded in `.superpowers/sdd/task-3-report.md`.
- Task 2 prepared-coupling and uniform-topology implementation completed on
  2026-07-10. `Coupler` now lazily owns one frozen private `PreparedCoupling`
  and reuses its contracts, read-only topology maps, destination dispatch, and
  interrupt controller across state creation, supplied-state runs, and output.
  Public mutators invalidate preparation; direct post-preparation component
  configuration mutation raises an actionable error. Every topology policy,
  including `SurfaceMaskPolicy`, follows `applies` then `build`; patches are
  key/shape validated, duplicate topology keys are rejected, and derived
  surface masks are temporary construction values. JAX x64 is documented and
  tested as process capability separate from explicit per-Coupler allocation
  policy. A review follow-up also validates externally supplied output states
  against the same prepared contracts/topology before any output side effect.
  Focused and full-fast validation passed; final static gates are
  recorded in `.superpowers/sdd/task-2-report.md`.
- Task 1 public component contracts and structural bridge completed on
  2026-07-10. `ComponentLike` is now the validated canonical structural
  extension contract; structural lifecycle hooks receive the original user
  object in the documented refresh order; `ComponentStepReturn` is public from
  its component owner package; public annotations expose `ComponentState` and
  `OutputConfig` instead of runtime internals; data-only step construction is
  rejected; runtime lifecycle bridges are private; and runtime execution
  precedence remains spec capability, `HostComponent` host enforcement, then
  whole-run `RuntimeOptions` backend selection. Focused fast (132 tests) and
  full fast (318 tests) pytest passed; final Black, flake8, mypy, and whitespace
  checks are recorded in `.superpowers/sdd/task-1-report.md`.
- Latest local superseded API-boundary validation passed as of
  2026-07-09. The breaking cleanup removes the `vercor.config` compatibility
  owner, moves surface-mask customization to public `vercor.topology`,
  replaces `RuntimeOptions.surface_masks` with
  `RuntimeOptions.topology`, keeps bundled ATM/OCN/LND mask mechanics private
  behind `SurfaceMaskPolicy`, removes setup-specific mask attributes from
  `Coupler`, removes the hidden `vercor.recipes.CouplerSpec` alias, exposes
  owner-module-only output/state type aliases, and cleans public component
  signatures so they no longer advertise underscored private type aliases.
  Validation passed using the direct `scipy` environment executable: focused
  fast API/plugin/setup/topology pytest, full fast pytest, Black, flake8, mypy,
  full pytest, coverage pytest at 90% total, example/package/test
  `compileall`, and `git diff --check`. Black emitted the recurring Python
  3.13/target-3.14 warning; full pytest/coverage emitted only the existing
  external JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.
- Latest local earlier API-boundary validation passed as of 2026-07-09
  using the direct `scipy` environment executable: focused legacy API-boundary
  pytest, custom component/backend smoke check, full fast pytest, full pytest,
  coverage pytest at 90% total, Black, flake8, mypy, example/package/test
  `compileall`, and `git diff --check`. The breaking legacy cleanup moves
  canonical runtime contracts to public `vercor.runtime`, keeps
  `vercor.config` as a compatibility alias, adds public `ExecutionContext` and
  `RuntimeDriver` for custom backends, makes `RuntimeOptions.surface_masks`
  setup-agnostic by default, renames runtime model-year policy to
  `model_year_seconds`, moves data-import policy to `DataComponent`, drives
  host execution through `ComponentSpec.execution`, exposes read-only
  `ComponentInfo`, moves `CouplerSpec` to `vercor.coupling`, updates bundled
  examples to opt into `SurfaceMaskPolicy()`, and bumps the package version to
  the superseded release. Black emitted the recurring Python
  3.13/target-3.14 warning; full
  pytest/coverage emitted only the existing external JAX dtype-promotion
  `FutureWarning` and xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local earlier API/plugin-boundary validation passed as of
  2026-07-09 using the direct `scipy` environment executable: focused
  red/green plugin-architecture pytest, affected API/setup/runtime fast pytest,
  full fast pytest, full pytest, coverage pytest at 90% total, Black, flake8,
  mypy, example/package/test `compileall`, and `git diff --check`. The
  breaking legacy cleanup moves core runtime policy to public `vercor.config`
  (`RuntimeOptions`, `SurfaceMaskPolicy`, `ExecutionBackend`, `DTypePolicy`),
  removes the old `vercor.setup_config` module, replaces `Coupler`'s
  `surface_mask_policy=` keyword with `runtime=RuntimeOptions(...)`, adds
  public `ComponentLike` and `FieldImportPolicy`, normalizes structural custom
  components through private `vercor.components._adapter`, moves time-selection
  data-import behavior from mutable `Settings` to `ComponentSpec.import_policy`,
  freezes component/setup config mappings, adds private runtime backend owners
  for built-in and custom execution paths, adds `CouplerSpec`, and bumps the
  package version to the superseded release. Black emitted the recurring
  Python 3.13/target-3.14 warning; full pytest/coverage emitted only the existing
  external JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.
- Latest local VerCOR 0.8 setup-config/API-boundary validation passed as of
  2026-07-09 using the direct `scipy` environment executable: focused red/green
  lifecycle/setup-boundary/JCM paired-config pytest, affected
  API/setup/runtime fast pytest, full fast pytest, full pytest, coverage pytest
  at 90% total, Black, flake8, mypy, example/package/test `compileall`, and
  `git diff --check`. The breaking cleanup keeps setup-specific config
  dataclasses behind `vercor.setups`, removes their root and
  `vercor.setup_config` exports, adds `JCMLandAtmosphereConfig` for the paired
  JCM land/atmosphere factory, removes the duplicate JCM setup keyword path,
  and initializes component lifecycle hooks for no-exchange component graphs.
  Black emitted the recurring Python 3.13/target-3.14 warning; full
  pytest/coverage emitted only the existing external JAX dtype-promotion
  `FutureWarning` and xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local exchange-declaration audit follow-up validation passed as of
  2026-07-09 using the direct `scipy` environment executable: focused red/green
  CAMulator land radiation-contract pytest, focused JCM land output/contract
  pytest, focused data/slab recipe declaration pytest, full fast pytest, full
  pytest, Black, flake8, mypy, and `git diff --check`. The audit found and
  fixed two additional dirty-tree issues: CAMulator land now declares received
  atmosphere radiation fields/defaults for the bundled CAMulator-Veros recipe,
  and JCM land now keeps received flux defaults as runtime-prefill data rather
  than letting `DataComponent.initialize()` advertise them as data outputs.
- Latest local JCM land/JAXGCM exchange declaration fix validation passed as
  of 2026-07-09 using the direct `scipy` environment executable: red/green
  JCM land recipe contract pytest, red/green JAXGCM constructor contract
  pytest, an initial-state-only `run_jcm_with_era5data.py` reproduction that
  now reaches `INITIAL_STATE_OK`, affected component/external/runtime fast
  pytest, full fast pytest, Black, flake8, mypy, and `git diff --check`. The
  fix declares JCM land's received atmosphere flux fields and JAXGCM's received
  `soil_moisture` field/default, so strict setup-agnostic exchange validation
  accepts the bundled JCM land-atmosphere recipes. The default `conda run`
  orientation command still panicked before pytest in the local
  `conda_rattler_solver` plugin; direct env Python remains the working path.
- Latest local data setup exchange declaration fix validation passed as of
  2026-07-09 using the direct `scipy` environment executable: focused red/green
  data-factory contract pytest, affected component/runtime fast pytest, full
  fast pytest, a `run_data_driver.py` initial-state-only smoke check, Black,
  flake8, mypy, and `git diff --check`. The fix keeps strict setup-agnostic
  exchange validation intact while declaring runtime receive fields/defaults on
  bundled ERA5/ERA-Interim data setup factories, so data-backed imported fields
  remain runtime inputs/fallbacks rather than advertised data outputs.
- Latest local setup-agnostic exchange API validation passed as of 2026-07-09
  using the direct `scipy` environment executable: baseline fast pytest,
  focused red/green setup-agnostic API pytest, affected runtime/API pytest,
  full fast pytest, full pytest, Black, flake8, mypy, coverage pytest at 90%
  total, example/package/test `compileall`, and `git diff --check`. The change
  adds public `SurfaceMaskPolicy`, removes hard-coded exchange field and
  topology component-name validation from generic runtime initialization,
  validates exchanges against component declarations, makes surface-mask setup
  optional/policy-controlled, propagates the actual runtime step index through
  `StepContext.step`, accepts `JAXGCMConfig` in `make_jcm_land_atmosphere`,
  and documents/tests custom named components exchanging custom fields. Black
  emitted the recurring Python 3.13/target-3.14 warning; full pytest/coverage
  emitted only the existing external JAX dtype-promotion `FutureWarning` and
  xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local boundary-first API redesign validation passed as of 2026-07-09
  using the direct `scipy` environment executable: focused
  API-boundary/public-contract pytest, full fast pytest, Black, flake8, mypy,
  full pytest, coverage pytest at 90% total, example `compileall`, and
  `git diff --check`. The breaking `0.7.0` cleanup keeps the root package
  core-only, moves setup implementations behind `_data`, `_external`, `_slab`,
  and `_jcm`, exposes setup factories only through `vercor.setups`, removes
  `ComponentState.field_candidates()`, moves common field vocabulary to
  `COMMON_FIELD_NAMES`, keeps exchange/regridding helper APIs in their owner
  modules, and removes the duplicate `setup_config.OutputFrequency` export.
  Black emitted the recurring Python 3.13/target-3.14 warning; full
  pytest/coverage emitted only the existing external JAX dtype-promotion
  `FutureWarning` and xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local expired exchange exception alias cleanup validation passed as of
  2026-07-09 using the direct `scipy` environment executable: focused
  API-boundary pytest, full fast pytest, Black, flake8, mypy, full pytest,
  coverage pytest at 90% total, and `git diff --check`. The cleanup removes
  the expired module-level `vercor.exceptions.ExchangerError` alias so only
  `ExchangeError` remains on the root and exception-module public surfaces.
  Black emitted the recurring Python 3.13/target-3.14 warning; full
  pytest/coverage emitted only the existing external JAX dtype-promotion
  `FutureWarning` and xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local VerCOR 0.7 API cleanup validation passed as of 2026-07-08 using
  the direct `scipy` environment executable: focused public API/boundary/output
  pytest, affected runtime/external fast pytest, full fast pytest, full pytest,
  coverage pytest at 90% total, Black, flake8, mypy, example `compileall`, and
  `git diff --check`. The breaking cleanup moves `PeriodOutput` into
  `vercor.output`, makes `OutputConfig.period is None` mean disabled and
  `PeriodOutput(frequency="step")` mean every-step output, hides output
  implementation helpers behind underscore modules, renames `JaxGCMConfig` to
  `JAXGCMConfig`, completes the exchange exception rename to `ExchangeError`
  with the expired module-level alias now removed, makes
  `RectilinearGrid` coordinates keyword-only, renames slab-ocean `H` to
  `mixed_layer_depth`, removes duplicate JCM setup helper wrappers, and keeps
  internal component author-normalization aliases private. Black emitted the
  recurring Python 3.13/target-3.14 warning; full pytest/coverage emitted only
  the existing external JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.
- Latest local API vocabulary cleanup validation passed as of 2026-07-08 using
  the direct `scipy` environment executable: focused API/runtime-boundary
  pytest, full fast pytest, full pytest, coverage pytest at 90% total, Black,
  flake8, mypy, example `compileall`, and `git diff --check`. The breaking
  cleanup consolidates component authoring around
  `ComponentSpec(..., lifecycle=LifecycleHooks(...), output=OutputConfig(...))`
  exposed through `component.spec`, removes `component.field_spec`,
  `ComponentSpec(hooks=...)`, constructor `output=...`, and
  `RunState.replace_fields(scope=...)`, renames public hook/state vocabulary to
  `fields` / `received` / `sent` and `receives` / `sends`, and keeps
  runtime-store containers private behind `ComponentState` view methods. Black
  emitted the recurring Python 3.13/target-3.14 warning; full pytest/coverage
  emitted only the existing external JAX dtype-promotion `FutureWarning` and
  xarray merge `FutureWarning` in JAXGCM coverage.
- Latest local expired deprecation residue cleanup validation passed as of
  2026-07-08 using the direct `scipy` environment executable: focused cleanup
  pytest for API boundaries/public API contracts/runtime state/runtime facade,
  Black, flake8, mypy, full fast pytest, full pytest, coverage pytest at 90%
  total, and `git diff --check`. The cleanup keeps VerCOR deprecation shims
  absent while removing remaining transition wording from active source,
  tests, and docs: tuple-vector errors are version-neutral, API contract tests
  no longer use transition-version names, stale audit docs now point to
  `DESIGN.md` and this file, and the external Dinosaur/JAX
  `jax.experimental.shard_map` pytest filter remains intentionally scoped to
  third-party import noise. Black emitted the recurring Python
  3.13/target-3.14 warning; full pytest/coverage emitted only the existing
  external JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.
- Latest local API review rewrite validation passed as of 2026-07-08 using the
  direct `scipy` environment executable: focused API-boundary pytest, Black,
  flake8, mypy, full fast pytest, full pytest, example `compileall`, coverage
  pytest at 90% total, and `git diff --check`. The rewrite is the breaking
  `0.6.0` cleanup: component constructors now take `spec=ComponentSpec(...)`,
  component snapshot output uses `OutputConfig` and public `SnapshotContext`,
  runtime prefill/validation hooks use typed public contexts/results, external
  setup factories accept `Spinup` and `PeriodOutput`, the root
  facade exports the final public config/output types plus `setups`, and active
  design/dependency docs describe the new boundary. Black emitted the recurring
  Python 3.13/target-3.14 warning; full pytest/coverage emitted only the
  existing JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.
- Latest local expired VerCOR deprecation shim removal validation passed as of
  2026-07-08 using the direct `scipy` environment executable: focused cleanup
  pytest, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest
  at 90% total, and source-level absence guards for VerCOR
  `DeprecationWarning`/`warnings.warn` usage. The cleanup removes
  `Coupler.view(...)`, `Coupler.views(...)`,
  `RunState.with_component_fields(...)`,
  `ComponentState.iter_store_fields(...)`, and runtime dispatch support for
  subclass `step_runtime_state(...)` / `step_host_runtime_state(...)`. Black
  emitted the recurring Python 3.13/target-3.14 warning; full pytest/coverage
  emitted only the existing external JAX dtype-promotion `FutureWarning` and
  xarray merge `FutureWarning` in JAXGCM coverage. No VerCOR deprecation
  warnings remain in active validation output.
- Latest local legacy API boundary rewrite validation passed as of 2026-07-08
  using the direct `scipy` environment executable: Black, flake8, mypy, full
  fast pytest, full pytest, coverage pytest at 90% total, example
  `compileall`, and `git diff --check`. Black emitted the recurring Python
  3.13/target-3.14 warning; full pytest/coverage emitted expected deprecation
  warnings from one-release compatibility wrappers plus the recurring JAX
  dtype-promotion `FutureWarning` and existing xarray merge `FutureWarning` in
  the real JAXGCM payload test. The rewrite moves `ComponentOutput` and
  `ComponentSnapshotWriter` to public `vercor.output.adapters`, keeps
  `_ComponentOutputAdapter` private, replaces public runtime-state component
  stepping with mapping-based `step(...)`, blocks removed component `.data` and
  `.setup_metadata` attributes, creates the shared private
  `vercor._field_names` owner, renames the private regridder base to
  `_BaseRegridder`, makes `Exchange.regrid` keyword-only, removes
  `grid_geometry.make_rectilinear_grid`, exports root `vector`, and makes
  `RunState.component(...)` / `RunState.components(...)` the canonical runtime
  view API.
- Latest local legacy API redesign validation passed as of 2026-07-07 using the
  direct `scipy` environment executable: Black, flake8, mypy, fast pytest,
  full pytest, coverage pytest at 90% total, and `git diff --check`. Black
  emitted the recurring Python 3.13/target-3.14 warning; full pytest/coverage
  emitted the recurring JAX dtype-promotion `FutureWarning` and existing xarray
  merge `FutureWarning` in the real JAXGCM payload test. The change removes
  root/module-level grid constructor shims, `_grid.py`/`_exchange.py`, public
  `Coupler.state()`/`initialize()`, public `RunState` runtime-store accessors,
  public component `.data` and `.setup_metadata`, callable regridders, and
  public output snapshot-writer registration. Public access is now
  `RectilinearGrid.uniform/from_coordinates`, `Coupler.initial_state()`,
  `RunState.component(...).field(...)`, explicit `regrid()`/`regrid_vector()`,
  and typed `ComponentOutput`.
- Latest local expired compatibility shim removal validation: focused cleanup
  pytest, full fast pytest, Black, flake8, mypy, full pytest, and git diff
  whitespace check passed as of 2026-07-07 using the direct `scipy`
  environment executable. Black emitted the recurring Python 3.13/target-3.14
  warning; full pytest emitted the recurring JAX dtype-promotion
  `FutureWarning` and the existing xarray merge `FutureWarning` in the real
  JAXGCM payload test.
- Latest local staged public-owner API rewrite validation: focused red/green
  API/settings/state tests, full API-boundary fast pytest, focused settings and
  state/runtime pytest, affected component/regridding fast pytest, example
  py_compile, Black, flake8, mypy, full fast pytest, full pytest, and git diff
  whitespace check passed as of 2026-07-07 using the direct `scipy` environment
  executable. Black emitted the recurring Python 3.13/target-3.14 warning; full
  pytest emitted the
  recurring JAX dtype-promotion `FutureWarning` and the existing xarray merge
  `FutureWarning` in the real JAXGCM payload test.
- Latest local staged API boundary facade validation: baseline fast pytest,
  focused red/green API-boundary pytest, affected runtime/output/regridding
  pytest, example py_compile, Black, flake8, mypy, full fast pytest, full
  pytest, coverage pytest at 90% total, git diff whitespace check, and
  `CONDA_NO_PLUGINS=true conda run -n scipy` fast pytest passed as of
  2026-07-07 using the direct `scipy` environment executable for full-suite
  validation. Black emitted the recurring Python 3.13/target-3.14 warning;
  full pytest/coverage emitted the recurring JAX dtype-promotion
  `FutureWarning` and the existing xarray merge `FutureWarning` in the real
  JAXGCM payload test.
- Latest local setup-facade/API-boundary validation: focused red/green pytest,
  Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-07-07 using the direct `scipy` environment executables.
- Latest archived full validation status: passing as of 2026-05-15.
- Latest archived fast validation status: `pytest tests/ -q --fast --tb=short`
  passed as of 2026-05-15.
- Latest archived static checks: Black, flake8, and mypy passed as of
  2026-05-15.
- Latest local organization-refactor validation: Black, flake8, mypy, and fast
  pytest passed as of 2026-05-26.
- Latest local compatibility-facade cleanup validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local boundary-cohesion validation: Black, flake8, mypy, focused fast
  pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local boundary-import validation: Black, flake8, mypy, focused fast
  pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local cohesion-boundary implementation validation: Black, flake8,
  mypy, focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-dispatch-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-run-boundary validation: Black, flake8, mypy, focused
  fast pytest, full fast pytest, and full pytest passed as of 2026-05-27.
- Latest local runtime-view/component-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local diagnostics-runtime-view validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local component-author API split validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local core-boundary mixin extraction validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local external-adapter runtime-boundary validation: Black, flake8,
  mypy, focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local callable-component boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local external-adapter state-boundary validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-facade/CAMulator-index validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-27.
- Latest local runtime-resource-holder validation: Black, flake8, mypy,
  focused fast pytest, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local obsolete compatibility API cleanup validation: Black, flake8,
  mypy, focused compatibility pytest, full fast pytest, and full pytest passed
  as of 2026-05-28.
- Latest local obsolete compatibility API active-doc audit validation:
  API-boundary fast pytest, full fast pytest, Black, flake8, mypy, and full
  pytest passed as of 2026-05-28.
- Latest local component protocol/resource boundary validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local whole-codebase boundary refactor validation: focused fast pytest,
  full fast pytest, Black, flake8, mypy, and full pytest passed as of
  2026-05-28.
- Latest local runtime-resource boundary refinement validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local runtime-preparation boundary validation: focused boundary
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local component/runtime boundary alias validation: focused fast
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-05-28.
- Latest local runtime-output boundary validation: focused red/green pytest,
  Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-06-01.
- Latest local runtime-state validation boundary validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
- Latest local runtime resource/topology boundary validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
- Latest local compiled-runtime cache boundary validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, and full pytest passed as of
  2026-06-01.
- Latest local runtime compilation cache boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local runtime topology policy boundary validation: focused red/green
  pytest, Black, focused boundary pytest, flake8, mypy, full fast pytest, and
  full pytest passed as of 2026-06-01.
- Latest local component-context boundary validation: focused red/green pytest,
  Black, focused fast pytest, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-01.
- Latest local component execution protocol boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local component lifecycle boundary validation: focused red/green
  pytest, Black, focused fast pytest, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-01.
- Latest local external-adapter helper boundary validation: focused red/green
  pytest, Black, focused fast pytest, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-01.
- Latest local external-adapter setup-state boundary validation: focused
  red/green pytest, Black, focused fast pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-01.
- Latest local logging facade/private-owner boundary validation: baseline fast
  pytest, focused red/green pytest, Black, flake8, mypy, focused logging pytest,
  full fast pytest, and full pytest passed as of 2026-06-02.
- Latest local bilinear interpolator boundary validation: baseline fast pytest,
  focused red/green pytest, Black, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-02.
- Latest local calendar forcing-index boundary validation: baseline fast pytest,
  focused red/green pytest, focused runtime pytest, Black, flake8, mypy, full
  fast pytest, and full pytest passed as of 2026-06-02.
- Latest local asset/forcing-data boundary validation: baseline fast pytest,
  focused red/green pytest, Black, focused post-format pytest, flake8, mypy,
  full fast pytest, and full pytest passed as of 2026-06-02.
- Latest local external adapter factory/setup-state boundary validation:
  focused red/green pytest, Black, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-06-02.
- Latest local CAMulator wind-filter boundary validation: focused red/green
  pytest, Black, focused post-format pytest, flake8, mypy, full fast pytest,
  and full pytest passed as of 2026-06-02.
- Latest local unit-test speedup validation: focused red/green pytest, Black,
  focused post-format pytest, flake8, mypy, full fast pytest with durations,
  full pytest with durations, and coverage pytest passed as of 2026-06-02.
- Latest local JAXGCM h5netcdf average-output validation: focused red/green
  pytest, Black, focused writer/API pytest, flake8, mypy, full fast pytest,
  full pytest, and coverage pytest passed as of 2026-06-03.
- Latest local Veros h5netcdf period-output validation: baseline fast pytest,
  focused red/green pytest, Black, focused post-format pytest, flake8, mypy,
  full fast pytest, full pytest, and coverage pytest passed as of 2026-06-03.
- Latest local streaming period-average output validation: focused red/green
  pytest, Black, focused boundary pytest, flake8, mypy, full fast pytest, full
  pytest, and coverage pytest passed as of 2026-06-04.
- Latest local Veros average dimension-order validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-04.
- Latest local Veros spinup period-average validation: focused red/green
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-08.
- Latest local JAX-backed output-array boundary validation: baseline fast
  pytest, focused red/green pytest, focused mypy, Black, flake8, full mypy,
  focused post-format pytest, full fast pytest, full pytest, and coverage
  pytest passed as of 2026-06-09.
- Latest local internal naming consistency validation: focused pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-10.
- Latest local trivial internal wrapper cleanup validation: focused cleanup
  scan, focused pytest, Black, flake8, mypy, full fast pytest, and full pytest
  passed as of 2026-06-10.
- Latest local unified GCM output package validation: focused red/green pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed as of 2026-06-10.
- Latest local shared h5netcdf output helper validation: focused red/green
  pytest, Black, git diff whitespace check, flake8, mypy, full fast pytest,
  full pytest, and coverage pytest passed as of 2026-06-10.
- Latest local shared JCM/Veros output dataset helper validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-11.
- Latest local shared period-average output writer validation: focused
  red/green pytest, focused output/API pytest, Black, git diff whitespace
  check, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed as of 2026-06-11.
- Latest local over-engineering quick-win cleanup validation: focused red/green
  pytest, focused affected pytest, Black, git diff whitespace check, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-11.
- Latest local over-engineering helper-layer cleanup validation: focused
  red/green pytest, focused affected pytest, Black, git diff whitespace check,
  flake8, mypy, full fast pytest, and full pytest passed as of 2026-06-11.
- Latest local runtime/component over-engineering sweep validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-11.
- Latest local over-engineering audit quick-win cleanup validation: focused
  red/green pytest, focused external/diagnostics pytest, Black, git diff
  whitespace check, flake8, mypy, full fast pytest, and full pytest passed as
  of 2026-06-11.
- Latest local no-break over-engineering cleanup campaign validation: focused
  red/green pytest, Black, git diff whitespace check, flake8, mypy, full fast
  pytest, full pytest, and coverage pytest passed as of 2026-06-12.
- Latest local external output ownership validation: focused red/green pytest,
  focused output pytest, Black, git diff whitespace check, flake8, mypy, full
  fast pytest, and full pytest passed as of 2026-06-12.
- Latest local CAMulator direct h5netcdf output validation: focused red/green
  pytest, focused output pytest, Black, git diff whitespace check, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest passed as of
  2026-06-16.
- Latest local CAMulator period-average output-frequency validation: focused
  red/green pytest, focused CAMulator/shared-output pytest, Black, flake8,
  mypy, full fast pytest, full pytest, coverage pytest, and git diff
  whitespace check passed as of 2026-06-17.
- Latest local centralized NetCDF filename-logging validation: focused
  red/green pytest, focused output pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `conda run -n scipy` fast pytest passed as of
  2026-06-19.
- Latest local internal output/runtime helper simplification validation:
  baseline fast pytest, focused red/green pytest, focused runtime/API pytest,
  Black, flake8, mypy, full fast pytest, full pytest, coverage pytest, and git
  diff whitespace check passed as of 2026-06-19.
- Latest local component output adapter refactor validation: baseline fast
  pytest via direct `scipy` env Python, focused adapter/external/API pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and git diff whitespace
  check passed as of 2026-06-29.
- Latest local unused helper API cleanup validation: focused red/green cleanup
  pytest, focused affected pytest, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, and git diff whitespace check passed as of
  2026-06-29.
- Latest local remaining helper-surface over-engineering cleanup validation:
  focused red/green pytest, Black, flake8, mypy, focused affected pytest, full
  fast pytest, full pytest, and coverage pytest passed as of 2026-06-29.
- Latest local centralized output adapter record-logic validation: focused
  red/green pytest, Black, flake8, mypy, focused post-format pytest, full fast
  pytest, full pytest, and git diff whitespace check passed as of 2026-06-30
  using the direct `scipy` environment executable.
- Latest local simplification-plan quick-win validation: baseline fast pytest,
  focused red/green pytest, focused affected pytest, Black, flake8, mypy, full
  fast pytest, full pytest, and git diff whitespace check passed as of
  2026-06-30 using the direct `scipy` environment executable.
- Latest local conservative scalar-only regridder validation: baseline fast
  pytest, focused red/green pytest, focused affected pytest, Black, flake8,
  mypy, full fast pytest, full pytest, and git diff whitespace check passed as
  of 2026-06-30 using the direct `scipy` environment executable.
- Latest local internal helper type-surface simplification validation: focused
  red/green pytest, Black, flake8, mypy, full fast pytest, full pytest, and
  git diff whitespace check passed as of 2026-06-30 using the direct `scipy`
  environment executable.
- Latest local external setup-step/remapper derived-state simplification
  validation: baseline fast pytest, focused red/green pytest, focused affected
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and git diff
  whitespace check passed as of 2026-06-30 using `conda run -n scipy`.
- Latest local RunSequence deprecation and tuple run-order validation: focused
  red/green API tests, focused affected fast runtime/external/helper/component
  tests, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest,
  and git diff whitespace check passed as of 2026-06-30 using the `scipy`
  environment through `conda run`.
- Latest local legacy component seed helper removal validation: baseline fast
  pytest, focused red/green API/component tests, Black, flake8, mypy, full fast
  pytest, full pytest, git diff whitespace check, and `conda run -n scipy`
  fast pytest passed as of 2026-06-30. The earlier planning-time Conda/Rattler
  panic was not reproduced during final validation.
- Latest local Exchange create-wrapper removal validation: baseline fast
  pytest, focused red/green Exchange tests, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and `conda run -n scipy`
  smoke pytest passed as of 2026-06-30. The earlier planning-time
  Conda/Rattler panic was not reproduced during final validation.
- Latest local concrete regridder call-ownership validation: baseline fast
  pytest, focused red/green boundary tests, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and git diff whitespace check
  passed as of 2026-06-30 using `conda run -n scipy`.
- Latest local component factory helper deprecation validation: focused
  red/green deprecation/API tests, focused affected fast pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and `conda run -n scipy` fast
  smoke pytest passed as of 2026-06-30 using the direct `scipy` environment
  executable for the full suite.
- Latest local deprecated compatibility shim removal validation: focused
  red/green API/component/forcing tests, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, git diff whitespace check, and
  `conda run -n scipy` fast pytest passed as of 2026-06-30 using the direct
  `scipy` environment executable for full-suite validation.
- Latest local non-differentiable host-runtime warning validation: focused
  red/green warning pytest, Black, flake8, mypy, full fast pytest, full pytest,
  and git diff whitespace check passed as of 2026-07-01 using `conda run` in
  the `scipy` environment; no Conda/Rattler fallback was needed.
- Latest local external-native snapshot finalize-output validation: focused
  red/green snapshot tests, affected output/runtime/external/API tests, Black,
  flake8, mypy, full fast pytest, full pytest, coverage pytest, git diff
  whitespace check, and `conda run -n scipy` fast pytest passed as of
  2026-07-01.
- Latest local final snapshot timestamp alignment validation: focused
  red/green finalize pytest, focused output-boundary pytest, Black, flake8,
  mypy, full fast pytest, full pytest, and `conda run -n scipy` full fast
  pytest passed as of 2026-07-02. Direct-environment validation used
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`; the earlier
  Conda/Rattler initialization panic was not reproduced during final smoke and
  fast-suite validation.
  - Fixed final native snapshot timestamps to use the last runtime step time
    yielded by `Clock.iter()` instead of one extra `dt`; zero-step runs keep
    `clock.start`.
  - Corrected the stale registered snapshot-writer filename expectation to
    match the existing lowercase snapshot filename policy.
- Latest local component setup validation wrapper cleanup validation: focused
  red/green setup-validation pytest, focused affected fast pytest, Black,
  flake8, mypy, full fast pytest, and full pytest passed as of 2026-07-02
  using `/Users/romannuterman/miniforge3/envs/scipy/bin/python`.
- Latest local runtime cache/donation API removal validation: focused red/green
  runtime/API/coupler/profile pytest, Black, flake8, mypy, focused affected
  fast pytest, full fast pytest, full pytest, and `conda run -n scipy` fast
  pytest passed as of 2026-07-02.
- Latest local JAXGCM PyTree/lifecycle simplification validation: focused
  red/green pytest, affected fast pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and git diff whitespace check passed as of 2026-07-02
  using `/Users/romannuterman/miniforge3/envs/scipy/bin/python`.
- Latest local over-engineering simplification slice validation: baseline fast
  pytest, focused red/green pytest, focused affected fast pytest, Black,
  flake8, mypy, full fast pytest, full pytest, coverage pytest, and git diff
  whitespace check passed as of 2026-07-02 using `conda run -n scipy`.
- Latest local runtime/CAMulator helper simplification validation: focused
  red/green pytest, focused affected fast pytest, Black, flake8, mypy, full
  fast pytest, full pytest, coverage pytest at 90% total, and git diff
  whitespace check passed as of 2026-07-02 using the direct `scipy`
  environment executable. The session-start `conda run -n scipy pytest tests/
  -v --fast 2>&1 | tail -20` smoke still failed before pytest with the known
  Conda/Rattler `PanicException`.
- Latest local component helper indirection simplification validation:
  baseline fast pytest, focused red/green pytest, focused affected fast pytest,
  Black, flake8, mypy, full fast pytest, full pytest, git diff whitespace
  check, and `conda run -n scipy` fast pytest passed as of 2026-07-03 using
  the direct `scipy` environment executable for the full validation. Full
  pytest emitted the known JAX dtype-promotion `FutureWarning` in the JAXGCM
  runtime gradient test.
- Latest local legacy time-selection field-helper removal validation: baseline
  fast pytest, focused red/green pytest, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest at 90% total, git diff whitespace check, and
  `conda run -n scipy` fast pytest passed as of 2026-07-03 using the direct
  `scipy` environment executable for the full validation. Black emitted the
  known Python 3.13/target-3.14 warning, and full pytest/coverage emitted the
  known JAX dtype-promotion `FutureWarning` in the JAXGCM runtime gradient
  test.
- Latest local remaining legacy API cleanup validation: baseline fast pytest,
  focused red/green pytest, focused affected pytest, Black, flake8, mypy, full
  fast pytest, full pytest, coverage pytest at 90% total, and git diff
  whitespace check passed as of 2026-07-03 using `conda run -n scipy`.
  Black emitted the recurring Python 3.13/target-3.14 warning, and full
  pytest/coverage emitted the recurring JAX dtype-promotion `FutureWarning`.
- Latest local legacy public API facade cleanup validation: focused legacy
  API-boundary pytest, focused regression pytest, Black, flake8, mypy, example py_compile,
  full fast pytest, full pytest, coverage pytest at 90% total, and git diff
  whitespace check passed as of 2026-07-03 using the direct `scipy`
  environment executable. Black emitted the recurring Python 3.13/target-3.14
  warning, and full pytest/coverage emitted the recurring JAX dtype-promotion
  `FutureWarning`.
- Latest local API redesign implementation validation: focused API-boundary
  pytest, affected runtime-exchange pytest, full fast pytest, Black check,
  flake8, mypy, full pytest, coverage pytest at 90% total, and git diff
  whitespace check passed as of 2026-07-03 using the direct `scipy`
  environment executable. Black emitted the recurring Python 3.13/target-3.14
  warning, and full pytest/coverage emitted the recurring JAX dtype-promotion
  `FutureWarning`.
- Latest local expired API cleanup validation: focused red/green
  API/runtime-state pytest, Black, flake8, mypy, full fast pytest, and full
  pytest passed as of 2026-07-06 using the direct `scipy` environment
  executable. Black emitted the recurring Python 3.13/target-3.14 warning, and
  full pytest emitted the recurring JAX dtype-promotion `FutureWarning`.
- Latest local evidence-only deprecation wording cleanup validation: baseline
  fast pytest, focused cleanup pytest, Black, flake8, mypy, full fast pytest,
  full pytest, and git diff whitespace check passed as of 2026-07-06 using the
  direct `scipy` environment executable. Black emitted the recurring Python
  3.13/target-3.14 warning, and full pytest emitted the recurring JAX
  dtype-promotion `FutureWarning`.
- Latest local staged public API redesign validation: focused API/settings/
  clock/setup/coupler pytest, Black, flake8, mypy, full fast pytest, full
  pytest, coverage pytest at 90% total, and the documented
  `conda run -n scipy` fast pytest passed as of 2026-07-06. Black emitted the
  recurring Python 3.13/target-3.14 warning, and full pytest/coverage emitted
  the recurring JAX dtype-promotion `FutureWarning`.
- Latest local pre-redesign API implementation validation: focused pre-redesign
  API-boundary red/green pytest, focused removed-import-path pytest, full fast
  pytest, Black check, flake8, mypy, full pytest, coverage pytest at 90%
  total, and `conda run -n scipy` fast pytest passed as of 2026-07-06 using
  the direct `scipy` environment executable for full-suite validation. Black
  emitted the recurring Python 3.13/target-3.14 warning, and full
  pytest/coverage emitted the recurring JAX dtype-promotion `FutureWarning`.
- Latest local JCM slab example runtime-field validation: focused red/green
  recipe/runtime/JAXGCM-payload pytest, Black, flake8, mypy, full fast pytest,
  full pytest, and `examples/run_jcm_with_slab.py` with `MPLBACKEND=Agg`
  passed as of 2026-07-06 using `env CONDA_NO_PLUGINS=true conda run -n
  scipy`. The fix keeps data-ocean recipes unchanged, adds a JCM-to-slab-ocean
  recipe with flux imports, and stabilizes real JCM physics payload PyTrees.
  Black emitted the recurring Python 3.13/target-3.14 warning; full pytest
  emitted the recurring JAX dtype-promotion `FutureWarning`, and the example
  emitted the expected xarray and non-interactive Agg warnings.
- No active `IN PROGRESS` task is recorded in the archived log.
- No current blocker is recorded in the archived log.
- Recurring known warning: Black may emit the existing Python 3.13 versus
  target-3.14 safety-check warning while still completing successfully.
- Recurring known warning: the JAXGCM runtime gradient test may emit the
  existing JAX dtype promotion `FutureWarning` while the suite still passes.

## Next Session Checklist

1. Read `DESIGN.md` for architecture and public/runtime boundary context.
2. Read `DEPENDENCIES.md` for module ordering before changing code.
3. Run:

   ```bash
   conda run -n scipy pytest tests/ -q --fast --tb=short
   ```

4. If the fast suite passes, pick the next unchecked item from this file or the
   next failing focused test.
5. If the fast suite fails, work from the first failing test and record the
   root cause and fix here.
6. Before stopping, update this file with a compact summary, not a full command
   transcript.

## Follow-Up Candidates

- Do not restore component `.data` or `.setup_metadata`; they are removed public
  surfaces, not compatibility attributes. Component authors use
  `seed_field()`/`seed_fields()` and plugin-owned attributes, while bundled
  adapters keep setup-only details in private `_setup_metadata`.

## Recent Work

### 2026-07-09: Expired Exchange Exception Alias Cleanup

- Removed the expired `vercor.exceptions.ExchangerError` compatibility alias;
  `ExchangeError` is now the only supported exchange exception import path.
- Strengthened API-boundary tests so both root and exception-module surfaces
  keep `ExchangerError` absent, and the source-level deprecation-residue guard
  rejects future reintroduction of the alias or temporary compatibility
  wording.
- Validation run for this change: focused red/green boundary tests, full fast
  pytest, Black, flake8, mypy, full pytest, coverage pytest at 90% total, and
  `git diff --check` passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted the
  recurring Python 3.13/target-3.14 warning; full pytest/coverage emitted only
  the existing external JAX dtype-promotion `FutureWarning` and xarray merge
  `FutureWarning` in JAXGCM coverage.

### 2026-07-08: Legacy Deprecation Cleanup

- Removed the remaining public conservative-regridder compatibility keyword:
  `vercor.regridding.conservative(...)` now accepts `radius_km` only and passes
  that value to the private conservative regridder's internal `radius`
  parameter.
- Added legacy boundary coverage so the public conservative factory keeps
  `radius` absent and active docs do not advertise removed transition APIs such
  as `ComponentView`, `Coupler.state()`, public `Coupler.initialize()`,
  public `Component.data` / `Component.setup_metadata`, or callable
  regridders.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to describe
  `ComponentState`, `Coupler.initial_state()`, non-callable regridders, and
  private setup stores only.
- Validation run for this change: focused red/green boundary tests, focused
  cleanup pytest, Black, flake8, mypy, full fast pytest, full pytest, and git
  diff whitespace check passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted the
  recurring Python 3.13/target-3.14 warning; full pytest emitted the recurring
  JAX dtype-promotion `FutureWarning` and the existing xarray merge
  `FutureWarning` in the real JAXGCM payload test.

### 2026-07-07: Expired Compatibility Shim Removal

- Removed expired 0.5.0 compatibility surfaces: public `CouplerState`,
  runtime `RuntimeCouplerState`, runtime-view `RuntimeComponentView`,
  `rectilinear_grid(...)`, `vercor.field_names`, exchange recipe fallback
  access from `vercor.exchanges`, direct JCM input generator wrappers, and the
  `build_jcm_land_atmosphere_components(...)` setup alias.
- Migrated runtime, diagnostics, output, and tests to the supported owners:
  `RunState`, `ComponentView`, `uniform_rectilinear_grid(...)`,
  `vercor.recipes`, `load_jcm_inputs(...)`, and
  `load_jcm_coords_terrain_forcing(...)`.
- Deleted obsolete shim modules `vercor/field_names.py`,
  `vercor/_runtime/views.py`, and `vercor/_fields.py`; refreshed API-boundary
  tests so removed modules and symbols stay absent without warning tests.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to describe only the supported
  public API.
- Validation run for this change: focused cleanup pytest, full fast pytest,
  Black, flake8, mypy, full pytest, and git diff whitespace check passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted the
  recurring Python 3.13/target-3.14 warning; full pytest emitted the recurring
  JAX dtype-promotion `FutureWarning` and the existing xarray merge
  `FutureWarning` in the real JAXGCM payload test.

### 2026-07-07: Staged Public-Owner API Rewrite

- Moved public ownership for `RectilinearGrid`, `VectorField`, and `Exchange`
  into `vercor.grids`, `vercor.fields`, and `vercor.exchanges`, and moved
  public ownership for `RunState` and `ComponentView` into `vercor.state`,
  with internal modules using focused public owners or private implementation
  imports where appropriate.
- Added `grid_from_coordinates(...)`, strict `Settings(custom={...})` handling
  for custom settings, explicit `Regridder.regrid(...)` /
  `regrid_vector(...)`, and a public `vercor.recipes` facade for bundled
  `*_FIELDS` exchange recipes.
- Tightened `Component` construction so raw `data=` and `setup_metadata=`
  cannot bypass setup validation through the public constructor; staged mutable
  attributes remain for existing setup adapters.
- Updated examples, API/settings tests, `DESIGN.md`, and `DEPENDENCIES.md` for
  the new public/private boundary.
- Validation run for this change: focused red/green API/settings/state pytest,
  full API-boundary fast pytest, focused settings and state/runtime pytest,
  affected component/regridding fast pytest, example py_compile, Black, flake8,
  mypy, full fast pytest, full pytest, and git diff whitespace check passed
  using `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted
  the recurring Python 3.13/target-3.14 warning; full pytest emitted the
  recurring JAX dtype-promotion `FutureWarning` and the existing xarray merge
  `FutureWarning` in the real JAXGCM payload test.

### 2026-07-07: Staged API Boundary Facades

- Added canonical public state names through `vercor.state`: `RunState` and
  `ComponentView`.
- Added public regridding and grid facades: `Regridder`,
  `RegridderFactory`, `bilinear`, `conservative`, and
  `uniform_rectilinear_grid(...)`; concrete regridder classes stay private
  and public factory signatures no longer expose `_regridders` return types.
- Made `vercor.exchanges.Exchange` the public owner in signatures/docs while
  keeping `_exchange` as the implementation module.
- Promoted the output extension surface through `vercor.output` for
  `OutputVariable`, `ComponentOutputAdapter`, and snapshot-writer
  registration while keeping runtime/period/netcdf helpers private-by-module.
- Updated examples, API-boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` to
  describe the staged 0.4.x-compatible public API.
- Validation run for this change: baseline fast pytest, focused red/green
  API-boundary pytest, affected runtime/output/regridding pytest, example
  py_compile, Black, flake8, mypy, full fast pytest, full pytest, coverage
  pytest at 90% total, git diff whitespace check, and
  `CONDA_NO_PLUGINS=true conda run -n scipy` fast pytest passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python` for full-suite
  validation. Black emitted the recurring Python 3.13/target-3.14 warning;
  full pytest/coverage emitted the recurring JAX dtype-promotion
  `FutureWarning` and the existing xarray merge `FutureWarning` in the real
  JAXGCM payload test.

### 2026-07-06: Pre-redesign API Implementation

- Added the pre-redesign public field vocabulary in `vercor.fields`, including
  `VectorField`, `vector(...)`, field-item normalization, field flattening, and
  shared valid-field vocabulary ownership.
- Exposed stable public `CouplerState` and `ComponentView` names, typed
  `Coupler.state()`, `run()`, `view()`, and `views()`, made setup mutators
  chainable, removed the public `initialize(enable_x64_computations=...)`
  override, and renamed `write_outputs(..., snapshots=...)` to
  `write_snapshots`.
- Migrated public grid construction to
  `rectilinear_grid(..., longitude=..., latitude=..., binary_mask=...)`,
  made `DataComponent.from_fields(...)` optional arguments keyword-only, and
  updated examples/tests to use pre-redesign exchange vector fields instead of
  tuple vectors.
- Exported public exception classes from `vercor` and updated facade imports so
  public workflows can use top-level `rectilinear_grid`, `bilinear`,
  `conservative`, `VectorField`, and `vector`.
- Moved implementation modules behind private paths:
  `vercor.grid` -> `vercor._grid`, `vercor.exchange` -> `vercor._exchange`,
  and `vercor.regridders` -> `vercor._regridders`; normal user imports go
  through the public facades.
- Bumped the package version to `0.4.0` for the breaking public API change.
- Validation run for this change: focused pre-redesign API-boundary pytest, focused
  removed-import-path pytest, full fast pytest, Black check, flake8, mypy, full
  pytest, coverage pytest at 90% total, and
  `conda run -n scipy pytest tests/ -q --fast` passed. Black emitted the
  recurring Python 3.13/target-3.14 warning, and full pytest/coverage emitted
  the recurring JAX dtype-promotion `FutureWarning`.

### 2026-07-06: Breaking Public API Cleanup

- Removed active transitional public surfaces from the staged API redesign:
  `VercorSettings`, `Clock(year_type=...)`, `grids.rectilinear(...)`,
  `Coupler.from_components(...)`, `Coupler.run_sequence`,
  mutable `Coupler.components`/`exchanges` setup assignment,
  `Coupler.finalize(...)`, output-helper reexports from `vercor.output`, and
  top-level `CouplerState`/`ComponentView` exports.
- Normalized component authoring to direct `inputs`/`outputs`/`defaults` field
  declarations and `lifecycle=LifecycleHooks(...)` lifecycle installation.
- Moved public regridding imports to `vercor.regridding`; concrete regridder
  classes remain implementation details under `vercor.regridders.*`.
- Updated tests, examples, `DESIGN.md`, and `DEPENDENCIES.md` to the breaking
  API. Validation run for this change: focused API-boundary fast pytest, full
  fast pytest, Black, flake8, mypy, full pytest, coverage pytest at 90% total,
  and `git diff --check` passed with
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted the
  recurring Python 3.13/target-3.14 warning, and full pytest/coverage emitted
  the recurring JAX dtype-promotion `FutureWarning`.

### 2026-07-06: Staged Public API Redesign Compatibility

- Added the legacy-compatible public names from the API redesign plan:
  `Settings`, `SettingSpec`, `Clock(calendar=...)`,
  `grids.rectilinear_grid(...)`, `Coupler(run_order=...)`,
  `Coupler.write_outputs(...)`, `DataComponent.from_fields(outputs=...)`, and
  `make_jcm_land_atmosphere(...)`.
- Kept staged wrappers for existing workflows: `VercorSettings`,
  `Clock(year_type=...)`, `grids.rectilinear(...)`, `Coupler.run_sequence`,
  mutable `Coupler.components`/`exchanges` assignment, `Coupler.finalize(...)`,
  and `build_jcm_land_atmosphere_components(...)`.
- Refactored `Coupler` from a mutable dataclass facade into a normal class with
  read-only component/exchange views, controlled mutation methods, and runtime
  resource invalidation after setup changes.
- Updated examples, public API-boundary tests, setup helper tests, design notes,
  and the dependency map to use the new public names.
- Validation run for this change: focused API/settings/clock/setup/coupler
  pytest, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest
  at 90% total, and `conda run -n scipy pytest tests/ -q --fast --tb=short`
  passed. Black emitted the recurring Python 3.13/target-3.14 warning, and full
  pytest/coverage emitted the recurring JAX dtype-promotion `FutureWarning`.

### 2026-07-06: Evidence-Only Deprecation Wording Cleanup

- Audited active source and tests after the legacy cleanup and found no live
  VerCOR deprecation warning machinery, deprecated wrappers, or shim modules to
  remove.
- Renamed active tests and design text to avoid legacy-looking wording around
  supported behavior: forcing file-to-runtime layout normalization,
  `VercorSettings` attribute updates, and removed-API regression guards.
- Kept the external JAX/Dinosaur deprecation-warning filter in `pyproject.toml`
  because the warning originates from optional `dinosaur` imports of
  `jax.experimental.shard_map`, not from VerCOR source code.
- Validation run for this change: baseline fast pytest, focused cleanup pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and git diff whitespace
  check passed using `/Users/romannuterman/miniforge3/envs/scipy/bin/python`.
  Black emitted the recurring Python 3.13/target-3.14 warning, and full pytest
  emitted the recurring JAX dtype-promotion `FutureWarning`.

### 2026-07-06: Vercor Expired Deprecation Cleanup

- Removed expired legacy public shim surfaces: component-prefixed aliases,
  `HostRuntimeComponent`, `from_model()`, `default_fields`, exchange legacy
  names, short exchange recipe aliases, regridder short aliases, long coupler
  method wrappers, setup orchestration helpers, and the shared deprecation
  helper module.
- Kept supported behavior on canonical APIs: `ComponentSpec(defaults=...)`,
  `Component.from_step(...)`, `HostComponent.from_step(...)`,
  `Exchange(source, target, fields, regrid=...)`, `Coupler.state/view/views`,
  `Coupler.run(state=...)`, `Coupler.finalize(output=...)`, and
  `vercor.exchanges` `*_FIELDS` recipe names.
- Updated API-boundary tests to assert the removed names and modules stay
  absent, refreshed runtime-state boundary coverage, and updated
  `DESIGN.md`/`DEPENDENCIES.md` to describe the 0.2.0-only API.
- Validation run for this change: focused red/green API/runtime-state pytest,
  Black, flake8, mypy, full fast pytest, and full pytest passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. Black emitted the
  recurring Python 3.13/target-3.14 warning, and full pytest emitted the
  recurring JAX dtype-promotion `FutureWarning`.

### 2026-07-03: API Redesign Implementation

- Made short component names canonical in `vercor` and `vercor.components`;
  legacy names such as `HostRuntimeComponent`, `ComponentComponentSpec`, and
  component-prefixed contexts now resolve through deprecating `__getattr__`
  aliases outside `__all__`.
- Added shared `_deprecation` and `components._constructor_options` helpers,
  centralized `defaults`/`default_fields` and lifecycle-hook normalization,
  and moved setup/adapters/examples/tests to `from_step`, `HostComponent`,
  `ComponentSpec`, `StepResult`, `SetupContext`, and `StepContext`.
- Canonicalized exchange and coupler workflows around `target`/`fields`/`regrid`
  plus `Coupler.state/view/views`, leaving old setup helpers and long coupler
  methods as warning wrappers for the deprecation window.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, examples, and API-boundary tests to
  document the canonical public/private split.

### 2026-07-03: Legacy Public API Facade Cleanup

- Added the legacy orchestration facade: `Coupler.from_components(...)`,
  `run_order`, `state()`, `view()`, `views()`, `add_component()`, and
  `add_exchanges()` while retaining compatibility wrappers for the old runtime
  and run-sequence names.
- Collapsed exchange declarations onto `Exchange(source, target, fields,
  regrid, name=None)`, added a stable derived `label`, and kept
  `ExchangeSpec`, `destination`, `field_names`, and `regridder_factory` as
  migration aliases.
- Added the component-author facade names `ComponentSpec`, `StepContext`,
  `SetupContext`, `StepResult`, `KEEP_PAYLOAD`, `Component.from_step(...)`,
  `DataComponent.from_fields(...)`, and `HostComponent.from_step(...)`, with
  old component/runtime names left as compatibility aliases.
- Added shallow public facades for `vercor.exchanges`, `vercor.grids`,
  `vercor.regridding`, and `vercor.setups`; updated examples to use the new
  public APIs instead of setup-helper and runtime-view wiring.
- Validation run for this change: focused legacy API-boundary pytest, focused
  regression pytest, Black, flake8, mypy, example py_compile, full fast
  pytest, full pytest, coverage pytest at 90% total, and `git diff --check`
  passed using the direct `scipy` environment executable. Black emitted the
  recurring Python 3.13/target-3.14 warning, and full pytest/coverage emitted
  the recurring JAX dtype-promotion warning.

### 2026-07-03: Remaining Legacy API Cleanup

- Renamed the daily forcing runtime-control setting from
  `get_field_time_slice` to `apply_daily_time_selection` and updated runtime
  field transfer, JCM land setup, and daily forcing tests to use the new API.
- Removed remaining compatibility-only public surfaces: top-level
  `CustomDateTime`, the vertical-coordinate `compute_pressure_levels` alias,
  `Clock` calendar metadata properties, and conservative-remapper public
  mass/area helpers.
- Moved conservative remap mass checks into private `grid_masks` helpers that
  compute rectilinear spherical cell areas from remapper geometry. Remapper
  tests now verify conservation through `apply_scalar()` and independent
  geometry calculations.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and API-boundary tests so the removed
  names stay absent and the current daily-selection setting is documented.
- Validation run for this change: baseline fast pytest, focused red/green
  pytest, focused affected pytest, Black, flake8, mypy, full fast pytest, full
  pytest, coverage pytest at 90% total, and `git diff --check` passed using
  `conda run -n scipy`. Black emitted the recurring Python 3.13/target-3.14
  warning, and full pytest/coverage emitted the recurring JAX dtype-promotion
  warning.

### 2026-07-03: Legacy Time-Selection Field-Helper Removal

- Removed the test-only direct field selection helpers
  `get_field_time_slice(...)` and `get_field_at_specific_time(...)` from
  `vercor.time_selection`. Runtime time selection now stays on
  `RuntimeStepInfo`, `daily_forcing_index(...)`, and `send_runtime_fields(...)`.
- Rewrote affected tests to assert the new time-selection boundary, use
  `daily_forcing_index(...)` for expected daily slices, and rely on existing
  JIT/gradient runtime coverage for daily slicing and monthly interpolation.
- Removed the now-unused dummy coupler test helper that only existed to satisfy
  the deleted helper's coupler-shaped argument.
- Updated `DEPENDENCIES.md` to describe `vercor.time_selection` as the owner of
  model-year seconds and periodic interpolation index math only.
- Validation run for this change: baseline fast pytest, focused red/green
  pytest, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest,
  git diff whitespace check, and `conda run -n scipy` fast pytest passed.
  Black emitted the recurring Python 3.13/target-3.14 warning, and full
  pytest/coverage emitted the recurring JAX dtype-promotion warning.

### 2026-07-02: Runtime and CAMulator Helper Simplification

- Removed the redundant `FieldStore.get_or(...)` default helper;
  `runtime_field_or(...)` now returns normalized component defaults directly,
  while zero-like fallback behavior stays on the runtime store.
- Replaced value-scanning `get_component(...)` with keyed
  `require_component(...)`, including explicit key/name mismatch errors for
  topology component mappings.
- Removed CAMulator wind-filter loader globals from `camulator_imports.py`.
  `CAMulatorStepper` now imports the internal wind-filter facade directly,
  owns state shifting and forcing concatenation itself, and runtime code calls
  the stepper directly instead of reaching through `stepper.state_manager`.
- Updated architecture tests, `DESIGN.md`, and `DEPENDENCIES.md` to reflect the
  slimmer helper ownership.
- Validation run for this change: focused red tests failed against the old
  helper surfaces before implementation; after cleanup, affected fast pytest,
  Black, flake8, mypy, full fast pytest, full pytest, coverage pytest, and
  `git diff --check` passed using
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python`. The recurring Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-07-02: Over-Engineering Simplification Slice

- Removed `PreparedRuntimeState`; runtime preparation now returns
  `RuntimeCouplerState` directly while refreshed contracts stay on
  `CouplerRuntimeResources`.
- Removed binary masks from `RuntimeCouplerState` scan carry; topology resources
  still own binary masks for final output and mask bookkeeping.
- Slimmed runtime dispatch and component host-runtime detection by dropping
  redundant all-exchanges storage/filtering, the one-line host-runtime
  predicate, and the duplicated differentiable method on the host protocol.
- Inlined one-case output adapter factories into JAXGCM, Veros, and CAMulator
  setup-state constructors, and inlined single-use ERA5/JCM land data prep
  helpers at their public factory boundaries.
- Made `CamulatorRuntimeCursor.initialize(...)` command-only while keeping the
  pure cursor calculation helper value-returning and directly tested.
- Updated architecture tests to verify behavior/public boundaries instead of
  preserving private helper placement.
- Required validation passed:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`,
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short`, and
  `git diff --check`. The existing Black Python 3.13/target-3.14 warning and
  JAX dtype-promotion warning remain.

### 2026-07-02: JAXGCM PyTree and Lifecycle Simplification

- Removed the public `vercor.pytree_utils` helper module and moved its PyTree
  leaf transforms into private `vercor.setups._external._jax_gcm_pytree`, the
  only production owner that needed them.
- Simplified component lifecycle hook setup by deleting the private owner
  protocol, hook merge method, and installer helper. Constructors now build one
  `LifecycleHooks` value, and callable/data wrappers assign it
  directly to the component's private lifecycle field.
- Relaxed architecture-locking tests around those old helper layers while
  keeping public API and removed-module guards.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to reflect the new helper
  ownership.
- Validation run for this change: focused red tests failed against the old
  structure before implementation; after cleanup, affected fast pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and git diff whitespace check
  passed using the direct `scipy` environment executable.

### 2026-07-02: Runtime Cache and Donation API Removal

- Removed `vercor._runtime.cache`, `CompiledRuntimeCache`, runtime cache keys,
  public cache inspection/clearing facades, and the `donate_state` run option.
- Simplified runtime resources to topology maps, runtime contracts, and the
  interrupt controller. `RuntimeRunContext` now carries only static execution
  inputs and shared controllers.
- Pure scanned runs still execute through a one-shot `jax.jit` wrapper, while
  host-backed runs keep the Python bridge and non-differentiability warning.
- Simplified `examples/profile_runtime.py` to report one `run_s` timing plus
  final state leaf count, without cache or donation metrics.
- Validation run for this change: focused red tests failed against the old
  cache/donation API before implementation; after cleanup, Black, flake8,
  mypy, focused affected fast pytest, full fast pytest, full pytest, and
  `conda run -n scipy` fast pytest passed.

### 2026-07-02: Component Setup Validation Wrapper Cleanup

- Removed the redundant `validate_registered_component_setup(...)` pass-through
  wrappers from runtime initialization/facade code. `validate_component_setup(...)`
  remains the single component-owned setup validator.
- `Coupler.register(...)` now calls the canonical validator directly, and
  runtime initialization validates components before precision synchronization
  and immediately after component initialization hooks.
- Removed the finalization-time setup revalidation; finalization now consumes
  validated runtime state and writes outputs.
- Added regression coverage for registration-time validation, direct
  initialization validation before precision sync, and removal of the obsolete
  wrapper name from production source.
- Validation run for this change: focused red tests failed for initialization
  ordering and wrapper-name presence before implementation; after cleanup,
  focused green tests, focused affected fast pytest, Black, flake8, mypy, full
  fast pytest, and full pytest passed with the direct `scipy` environment
  executable. The session-orientation `conda run` command failed before pytest
  with the known Conda/Rattler `PanicException`; Black emitted the existing
  Python 3.13/target-3.14 warning, and full pytest emitted the existing JAX
  dtype-promotion warning.

### 2026-07-01: Component Snapshot Finalize Output

- Refactored `Coupler.finalize(...)` snapshots so runtime finalization only
  calls component-registered native snapshot writers and skips components
  without a provider. The public finalize API and runtime field output files
  are unchanged.
- `ComponentOutputAdapter` now stores one latest snapshot record separately
  from period-average accumulation and writes it through the existing
  period-output NetCDF pipeline using a temporary accumulator.
- JAXGCM snapshots read the final runtime payload `JCMState`, Veros snapshots
  read `VerosGCMSetupState._veros_state`, and CAMulator records the latest
  prediction in both increment-output and period-output modes. Snapshot output
  no longer uses runtime `data` values or `component.spec.outputs`.
- Focused red/green coverage added for adapter snapshots, provider-based
  finalize orchestration, native external snapshot contents, and API-boundary
  checks.
- Validation run for this change: focused red/green snapshot tests, affected
  output/runtime/external/API tests, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, git diff whitespace check, and
  `conda run -n scipy` fast pytest passed. Black emitted the existing Python
  3.13/target-3.14 warning, and full pytest/coverage emitted the existing JAX
  dtype-promotion warning.

### 2026-07-01: Non-Differentiable Host-Runtime Warning

- Added one `Coupler.run()` warning before the Python host runtime starts when
  host-backed components make the full coupled loop non-differentiable.
- The warning lists all host-backed component names in runtime component
  insertion order and reuses the existing `vercor.jax_logging` logger boundary.
- Added regression coverage for multiple host-backed components and documented
  the behavior in `DESIGN.md`.
- Validation run for this change: focused red warning pytest failed before the
  runtime warning existed; after implementation, focused green pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and git diff whitespace check
  passed using `conda run -n scipy`. Black emitted the existing Python
  3.13/target-3.14 warning, and full pytest emitted the existing JAX
  dtype-promotion warning.

### 2026-06-30: Deprecated Compatibility Shim Removal

- Removed the explicit deprecated public shims: `RunSequence`, module-level
  component factory helpers, `vercor.components.factories`, `vercor.run_sequence`,
  and forcing-index delegates from `vercor.calendar`.
- Kept the supported behavior: plain run-order sequences normalize to immutable
  tuples, string run-order values fail early, component authors use
  `DataComponent.from_fields()`, `Component.from_model()`, and
  `HostRuntimeComponent.from_model()`, and daily forcing-index policy lives in
  `vercor.forcing_index`.
- Moved run-order normalization to private `vercor._run_order`, refreshed
  `DESIGN.md`/`DEPENDENCIES.md`, and updated boundary tests so removed modules
  and exports stay absent.
- Validation run for this change: focused red tests first failed on still-present
  exports/modules/delegates; after implementation, focused affected tests,
  Black, flake8, mypy, full fast pytest, full pytest, coverage pytest, git diff
  whitespace check, and `conda run -n scipy pytest tests/ -q --fast --tb=short`
  passed. Black emitted the existing Python 3.13/target-3.14 warning, and full
  pytest/coverage emitted the existing JAX dtype-promotion warning.

### 2026-06-30: Component Factory Helper Deprecation

- Deprecated `data_component()`, `differentiable_component()`, and
  `host_component()` as importable compatibility shims that warn and delegate
  to `DataComponent.from_fields()`, `Component.from_model()`, and
  `HostRuntimeComponent.from_model()`.
- Added lifecycle-hook parameters to `DataComponent.from_fields()` so data-only
  components no longer need the module-level helper for setup/runtime hooks.
- Migrated VerCOR setup factories, the custom component example, and ordinary
  tests to the class constructors; kept focused warning coverage for the
  deprecated public helpers.
- Validation run for this change: focused red deprecation/API tests failed on
  missing warnings, missing data constructor hook keywords, and internal helper
  imports; after implementation, focused affected fast pytest, Black, flake8,
  mypy, full fast pytest, full pytest, and `conda run -n scipy pytest tests/ -q
  --fast` passed. Black emitted the existing Python 3.13/target-3.14
  safety-check warning, and full pytest emitted the existing JAX dtype-promotion
  warning.

### 2026-06-30: Concrete Regridder Call Ownership

- Moved regridder call dispatch out of the shared `Regridder` base and into
  the concrete bilinear and conservative classes. The base now owns only shared
  grid/interpolator/display state.
- Preserved bilinear scalar/vector behavior, conservative scalar-only errors,
  and identical-grid passthrough while removing the conservative `_ensure_ready`
  override pattern.
- Added boundary coverage for the new ownership split and updated dependency
  wording plus stale test comments.
- Validation run for this change: baseline fast pytest passed; the focused
  red boundary test first failed on the existing base `__call__`; after
  implementation, focused affected pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `git diff --check` passed using
  `conda run -n scipy`. Black emitted the existing Python 3.13/target-3.14
  warning, and full pytest emitted the existing JAX dtype-promotion warning.

### 2026-06-30: Exchange Create Wrapper Removal

- Removed the public-looking one-line `Exchange.create()` wrapper so exchange
  declarations remain static configuration and runtime topology construction
  calls `exchange.regridder_factory(...)` directly.
- Updated helper and coupler coverage tests to assert the removed wrapper stays
  absent, preserve factory-name formatting behavior, and patch
  `regridder_factory` for topology recording tests while keeping existing
  interpolation keys.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on `Exchange.create` still existing, then passed after
  removal. The first full pytest run exposed one stale test monkeypatching the
  removed wrapper; after updating that test double, focused affected pytest,
  Black, flake8, mypy, full fast pytest, full pytest, and
  `conda run -n scipy python -m pytest tests/test_helpers_coverage.py -q --fast --tb=short`
  passed. Black emitted the existing Python 3.13/target-3.14 safety-check
  warning, and full pytest emitted the existing JAX dtype-promotion warning.

### 2026-06-30: Legacy Component Seed Helper Removal

- Removed the public `Component.seed_zero_field()`, `seed_zero_fields()`, and
  `seed_constant_field()` helpers so component authors use the canonical
  scalar-expanding `seed_field()` and `seed_fields()` path.
- Updated API-boundary and component coverage tests to assert the collapsed
  authoring surface and to validate zero/constant seeding through
  `seed_field(s)`. Updated `DESIGN.md` component-authoring guidance to match.
- Validation run for this change: baseline direct fast pytest passed; focused
  red pytest failed on the existing `seed_zero_field` API, then focused green
  pytest passed after implementation. Black, flake8, mypy, direct full fast
  pytest, direct full pytest, `git diff --check`, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. Black emitted
  the existing Python 3.13/target-3.14 safety-check warning, and full pytest
  emitted the existing JAX dtype-promotion warning.

### 2026-06-30: External Setup-Step and Remapper Derived-State Simplification

- Removed one-line Veros and CAMulator GCM setup-state `step()` delegates.
  Their factories now pass `partial(...step_*_runtime, state)` directly to
  the host component boundary, matching the existing JAXGCM factory pattern.
- Removed conservative remapper cached derived fields
  `_normalize_fracarea` and `_n_dst_cells`; `apply_scalar()` now derives those
  values locally from declared metadata, so PyTree unflattening no longer
  needs a class-specific post-unflatten hook.
- Updated boundary/PyTree tests to guard the simplified behavior. The first
  full pytest run exposed a stale runtime-state source assertion that assumed
  Veros and CAMulator GCM setup-state files must still define `def step(`;
  that assertion now only inspects the remaining CAMulator land inline step
  and explicitly verifies the removed setup-state delegates stay absent.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on the existing setup-state delegates and remapper cached
  derived attributes. After implementation, focused affected pytest, Black,
  flake8, mypy, full fast pytest, full pytest, and `git diff --check` passed
  using `conda run -n scipy`. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning.

### 2026-06-30: Internal Helper Type-Surface Simplification

- Removed the unused `FieldDefaults` type alias from component contracts and
  the private component-contract reexport layer. Boundary tests now assert that
  the alias stays absent from both surfaces.
- Removed the `RuntimeRegridder` concrete union from
  `vercor._runtime.topology_state`; grouped runtime topology maps now avoid
  importing bilinear/conservative regridder implementations and type the
  regridder map as an internal object container.
- Updated `DEPENDENCIES.md` so runtime topology state no longer lists direct
  bilinear/conservative regridder dependencies.
- Validation run for this change: focused red tests first failed on the
  existing `FieldDefaults` export and `RuntimeRegridder` topology alias. After
  implementation, focused boundary pytest, Black, flake8, mypy, full fast
  pytest, full pytest, and `git diff --check` passed using direct `scipy`
  environment executables. Black emitted the existing Python 3.13/target-3.14
  safety-check warning, and full pytest emitted the existing JAX
  dtype-promotion warning.

### 2026-06-30: Conservative Scalar-Only Regridder Cleanup

- Removed the unsupported `ConservativeRectilinearRemapper.apply_vector()`
  stub so conservative remapping exposes only the scalar operation it
  implements.
- Added a conservative regridder argument guard that rejects vector calls before
  the shared identical-grid fast path, so both identical and non-identical
  conservative vector calls fail with the same scalar-only `TypeError`.
- Validation run for this change: baseline fast pytest passed; focused red
  tests first failed on the existing remapper vector stub, old remapper-origin
  `RuntimeError`, and identical-grid vector passthrough. After implementation,
  focused red/green pytest, affected conservative pytest, Black, flake8, mypy,
  full fast pytest, full pytest, and `git diff --check` passed using direct
  `scipy` environment executables. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning.

### 2026-06-30: Simplification Plan Quick Wins

- Relaxed private architecture-locking tests toward behavior and public-boundary
  assertions where cleanup was implemented.
- Removed unused public NumPy dtype helper functions; tests now derive NumPy
  dtypes with `np.dtype(jax_real_dtype(...))` and
  `np.dtype(jax_index_dtype(...))`.
- Added compatibility-safe plain component-name sequence support for `Coupler`,
  setup helpers, and runtime preparation/facade paths while preserving
  `RunSequence` normalization internally. Updated examples and `DESIGN.md`
  accordingly.
- Simplified regridder internals by removing the unused interpolation protocol
  and subclass post-initialization mutation; concrete regridders now pass their
  interpolator and identical-grid flag into the shared base.
- Removed the callable component request dataclass/init hop; callable component
  constructors now build field specs and lifecycle hooks directly.
- Validation run for this change: the focused red tests first failed on the old
  dtype helpers, mandatory `RunSequence` storage, regridder protocol, and
  callable request dataclass. After implementation, focused affected pytest,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q`,
  and `git diff --check` passed. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and full pytest emitted the existing
  JAX dtype-promotion warning. The earlier baseline `conda run -n scipy ...`
  path still hit the local Conda/Rattler panic before pytest, so validation used
  the direct `scipy` environment executable.

### 2026-06-30: Centralized Output Adapter Record Logic

- Added `ComponentOutputAdapter.record_period_average_if_due()` as the shared
  output path for "accumulate sample, check cadence, write if due" behavior.
- Added JAXGCM, Veros, and CAMulator package-internal output adapter factories
  plus record helpers so model-specific extraction, coordinates, metadata, and
  CAMulator forecast-increment output remain beside each external adapter while
  period-average orchestration goes through the shared adapter boundary.
- Rewired JAXGCM, Veros, and CAMulator setup states and runtime output paths to
  use those helpers, and updated API-boundary and focused output tests to guard
  against local write-closure duplication returning to runtime modules.
- Validation run for this change: focused red tests first failed on the missing
  generic adapter method and missing model-specific factory/record helpers.
  After implementation,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/test_output_adapters.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py tests/test_api_boundaries.py -q --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/black vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/mypy vercor examples tests`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`,
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --tb=short`,
  and `git diff --check` passed. Black emitted the existing Python
  3.13/target-3.14 safety-check warning, and the full suite emitted the
  existing JAX dtype-promotion warning. Earlier in this session, both
  `conda run -n scipy ...` and `conda --no-plugins run -n scipy ...` hit the
  existing Conda/Rattler `PanicException` before pytest, so validation used the
  direct `scipy` environment executable.

### 2026-06-29: Remaining Helper-Surface Over-Engineering Cleanup

- Added `docs/over-engineering-audit-2026-06-29.md` with the requested
  executive summary, findings table, and conservative refactor plan.
- Removed the hidden `Clock._iter_impl` dispatch attribute; `Clock.iter()` now
  branches directly between Gregorian and model-calendar iterators while
  preserving existing calendar behavior.
- Removed unused CAMulator wind-filter convenience exports
  `wind_filter()` and `simple_wind_artifact_filter()` while keeping the
  config-driven runtime post-processing facade and private tensor mechanics.
- Removed the unused `jax_real_array_copy()` dtype helper and kept the active
  dtype policy helpers used by production code.
- Validation run for this change: focused red tests first failed on the old
  clock dispatch attribute, CAMulator wind-filter wrappers, and dtype copy
  helper, then the same focused suite passed after implementation. Black,
  flake8, mypy, focused affected pytest, full fast pytest, full pytest, and
  coverage pytest at 90% total coverage passed with `conda run -n scipy`. The
  earlier orientation smoke check still recorded the existing Conda Rattler
  plugin crash before pytest on one `conda run` invocation; the direct scipy
  environment fallback
  `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m pytest tests/ -q --fast --tb=short`
  passed. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-29: Unused Helper API Cleanup

- Removed unused CAMulator stepper convenience methods and accessor attributes
  so the runtime path owns model input assembly, postprocessing, and state
  shifting directly.
- Removed the one-line period-average accumulation wrapper; the shared output
  adapter now calls its owned accumulator directly.
- Removed the unused generic PyTree concatenation helper and its concat-only
  test coverage while keeping the active PyTree helpers.
- Validation run for this change: focused red cleanup tests failed before the
  production edits for the expected remaining helper surfaces. Focused affected
  pytest, Black, flake8, mypy, full fast pytest, full pytest, coverage pytest
  at 90% total coverage, and `git diff --check` passed using the direct
  `scipy` env Python path. The existing Black Python 3.13/target-3.14 warning
  and JAX dtype-promotion warning remain.

### 2026-06-29: Component Output Adapter Refactor

- Added `vercor.output.ComponentOutputAdapter` as the shared owner for
  external component period-average accumulation, mean conversion, cadence, and
  NetCDF write lifecycle.
- Replaced model-specific period-output wrapper routines in JAXGCM, Veros, and
  CAMulator with small extraction, coordinate, path, and metadata helpers that
  runtimes compose through the adapter. CAMulator immediate forecast-increment
  output remains model-specific.
- Updated architecture/dependency docs and boundary tests so shared
  period-output helper calls live in `vercor.output.adapters`, not in each
  external component output module.
- Validation run for this change: `conda run -n scipy pytest tests/ -v
  --fast` failed before pytest due to the existing Conda rattler plugin
  `PanicException`; `/Users/romannuterman/miniforge3/envs/scipy/bin/python -m
  pytest tests/ -q --fast --tb=short` passed. Focused
  adapter/external/API pytest, Black, flake8, mypy, full fast pytest, full
  pytest, and `git diff --check` passed. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-19: Internal Output/Runtime Helper Simplification

- Removed the redundant `PeriodAverageSample` alias plus
  `samples_from_output_variables()` / `mean_samples_or_raise()` from
  period-average output. Accumulators now accept `OutputVariable` samples
  directly and `period_mean_output_variables()` owns adapter-specific empty
  accumulator errors.
- Removed the unused `runtime.facade.run_scanned()` shortcut and the thin
  `refresh_runtime_contracts()` wrapper. Focused test helpers now call the
  scanned runtime owner directly, and runtime preparation calls
  `build_exchange_contracts()` at the point of use.
- Made `RuntimeTopologyMaps` a mutable slotted dataclass, matching its actual
  setup-time mutation model while leaving the surrounding topology-state
  containers frozen.
- Validation run for this change: baseline fast pytest, focused red/green
  pytest, focused runtime/API pytest, Black, flake8, mypy, full fast pytest,
  full pytest, coverage pytest, and git diff whitespace check passed. The
  existing Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning
  remain.

### 2026-06-19: Centralized NetCDF Filename Logging

- Moved NetCDF filename log emission into the shared `write_netcdf_dataset`
  boundary and routed period-average and CAMulator forecast-increment writers
  through that single logging path.
- Added regression coverage for shared-writer logger injection, exact-once
  period-file logging, CAMulator forecast/average filename logging, and scalar
  data-variable writes. The scalar test covers the h5py rule that scalar
  datasets cannot use gzip filter options.
- Validation run for this change: focused red/green pytest, focused output
  pytest, Black, flake8, mypy, full fast pytest, full pytest, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. The existing
  Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-17: CAMulator Period-Average Output Frequency

- Added `output_frequency` to the CAMulator GCM public factory and setup state.
  `None` preserves per-forecast-increment output; configured `day`, `month`, or
  `year` streams CAMulator prediction tensors into the shared period-average
  accumulator and writes average files under the configured forecast output
  folder.
- Added CAMulator output helpers for period accumulation and average-file
  writing through shared `vercor.output` primitives while keeping CAMulator
  tensor metadata, `predict.save_vars` filtering, and forecast-increment output
  in `camulator_output.py`.
- Updated the CAMulator/Veros example and architecture/dependency docs for the
  unified external output interface.
- Validation run for this change: focused red/green pytest, focused
  CAMulator/shared-output pytest, Black, flake8, mypy, full fast pytest, full
  pytest, coverage pytest, and `git diff --check` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-16: CAMulator Direct h5netcdf Output

- Replaced CAMulator forecast-increment output delegation to CREDIT
  `make_xarray`/NetCDF helpers with VerCOR-owned tensor shaping and direct
  `h5netcdf` writing through the shared output variable boundary.
- Moved CAMulator output metadata loading into `camulator_output.py`, kept
  model/parser/transform CREDIT imports in `camulator_imports.py`, and wired
  runtime output to use the setup state's existing tensor transformer for
  `predict.climate_rescale_output`.
- Added unsupported-option validation for xarray-only CREDIT output features
  such as pressure interpolation, ptype, and CREDIT-specific encoding dicts.
- Red/green notes: new output tests first failed because direct helper APIs and
  the `state_transformer` writer argument were missing; after implementation,
  the fast suite exposed the new NumPy import as an explicit host-output
  boundary, so `tests/test_production_numpy_boundaries.py` now lists
  `camulator_output.py`.
- Validation run for this change: focused CAMulator/output pytest, Black,
  `git diff --check`, flake8, mypy, full fast pytest, full pytest, and coverage
  pytest passed. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-12: External Output Adapter Ownership Boundary

- Moved JAXGCM and Veros-specific period-output adapters from `vercor.output`
  to `vercor.setups._external`, leaving `vercor.output` as the shared
  setup-agnostic output primitive package.
- Updated JAXGCM/Veros runtime and setup-state imports, boundary tests,
  functional output tests, `DESIGN.md`, and `DEPENDENCIES.md` for the clean
  break from `vercor.output.jax_gcm` and `vercor.output.veros`.
- Red/green notes: focused external/API tests first failed with missing
  `vercor.setups._external.jax_gcm_output` before the move, then passed after
  moving modules and rewiring imports. A pre-existing period-file log-message
  test drift on this branch was aligned with the current
  `Writing output file:  ...` message.
- Validation run for this change: `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_period_averages.py tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-12: No-Break Over-Engineering Cleanup Campaign

- Added `docs/over-engineering-audit-2026-06-12.md` with the requested
  executive summary, findings table, and recommended refactor plan.
- Removed three no-break internal simplification targets from the audit:
  `RuntimeTopologyMaps.from_mappings()` is gone and copy semantics now live at
  the exchange-topology boundary, `vercor.output` directly reexports its three
  runtime-output helpers, and one-use period-file builder aliases are inlined
  into `write_period_average_netcdf()`.
- Left public or compatibility-bound simplification candidates as follow-up
  only: `Grid`, `RunSequence`, component authoring/lifecycle layers, calendar
  compatibility delegates, and optional-dependency setup facades.
- Red/green notes: focused runtime tests first failed on the old
  `from_mappings()` helper, and focused API/period-file tests first failed on
  the old period-file alias layer. The same focused suites passed after the
  cleanup.
- Validation run for this change: baseline fast pytest passed before edits.
  After implementation, `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q` passed. Coverage remained
  at 90% total. The existing Black Python 3.13/target-3.14 warning and JAX
  dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Audit Quick-Win Cleanup

- Completed a whole-codebase over-engineering sweep focused on unnecessary
  internal wrappers, single-use helpers, speculative extension points, and
  public surfaces whose complexity is either justified or compatibility-bound.
- Removed three low-risk internal helper layers:
  private diagnostics `view_field*` delegates now call the runtime view lookup
  owner directly, Veros setup binds `_veros_state.pure` directly instead of a
  one-line `advance_veros_model_step()` wrapper, and `VercorSettings` copies
  immutable default records with `dict(DEFAULT_SETTINGS)` instead of rebuilding
  each `Settings` tuple.
- Added boundary tests preventing those helper shapes from returning while
  preserving existing behavior coverage for settings isolation, diagnostics,
  and external adapter boundaries.
- Audit findings deferred as not quick wins: `RuntimeTopologyMaps.from_mappings`
  is internal but has explicit boundary tests and can wait; component
  authoring/lifecycle mixins and callable wrappers protect documented public
  extension APIs; calendar forcing-index delegates remain active compatibility
  imports.
- Red/green notes: focused boundary/settings tests first failed on the old
  diagnostics wrappers, Veros wrapper, and settings copy helper, then passed
  after the cleanup. Focused external component/diagnostics tests also passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_settings.py -q --tb=short`
  failed first on the expected old helper shapes, then passed after the
  cleanup. `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_tools_components_and_plotting.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target 3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Runtime and Component Over-Engineering Sweep

- Inlined compiled-runtime aliases into `vercor._runtime.cache` and removed the
  alias-only `vercor._runtime.compilation` module.
- Simplified `CouplerRuntimeResources` to public dataclass fields and moved
  cache clearing/counting call sites to the cache owner.
- Removed the private runtime-preparation input protocol, the time-selection
  lookup protocols, and adapter-local runtime-state protocols in favor of the
  existing facade/setup-state types.
- Moved component runtime-field convenience methods onto `Component`, deleted
  `_runtime_access.py`, and trimmed `_protocols.py` to the runtime-checkable
  host execution protocol.
- Reused `OutputVariable` for period-average samples through the
  `PeriodAverageSample` compatibility alias.
- Red/green notes: focused boundary tests first failed on the old alias module,
  private resource fields/wrappers, annotation-only protocols, runtime-access
  mixin, and duplicate sample dataclass; after implementation the same focused
  suite passed.
- Validation run for this change: focused red/green pytest, Black, `git diff
  --check`, flake8, mypy, full fast pytest, full pytest, and coverage pytest
  passed. Coverage reported 90% total. The existing Black Python 3.13/target
  3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Helper-Layer Cleanup

- Removed the private `_IdentityInterpolator` helper and let the base regridder
  handle identical-grid scalar/vector passthrough directly before requiring a
  concrete interpolator.
- Narrowed `RuntimeTopologyMaps.from_mappings()` to the only used behavior:
  return an empty bundle or copy an existing topology-map bundle. Removed the
  unused keyword-construction branches.
- Removed the one-line `CouplerRuntimeResources.replace_topology(...)` wrapper;
  the runtime facade now assigns the grouped topology maps directly on runtime
  resources.
- Removed the unused `vercor.output.veros.VerosOutputVariable` alias so Veros
  output uses the shared `OutputVariable` container directly.
- Red/green notes: the focused cleanup tests first failed on the old identity
  helper, resource topology wrapper, broad topology-map constructor, and Veros
  alias; after implementation the same focused suite passed.
- Deferred broader simplifications for public/boundary-tested surfaces:
  `RunSequence`, `Grid`, component authoring mixins, calendar compatibility
  delegates, and setup helper APIs.
- Validation run for this change: focused affected pytest, Black, `git diff
  --check`, flake8, mypy, full fast pytest, and full pytest passed. The existing
  Black Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Over-Engineering Quick-Win Cleanup

- Removed unused PyTree payload from the bilinear interpolator and conservative
  remapper by dropping cached source meshgrids, unused `fracarea_norm`, and the
  unused source-cell count while preserving interpolation/remapping behavior.
- Removed the dead core runtime data-field validator now superseded by the
  component-owned canonical runtime-field validation path.
- Simplified JAXGCM factory wiring so setup state binds runtime-owned lifecycle
  hooks directly, and removed the one-line callback delegate layer from
  `jax_gcm_state`.
- Removed the unused CAMulator `accessor_state` setup attribute while keeping
  the runtime-used input/output accessors.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture/PyTree tests to
  document and enforce the simplified boundaries.
- Red/green notes: the focused cleanup tests first failed on the old cached
  fields, dead validator, callback delegates, and CAMulator accessor; after the
  cleanup the same focused suite passed.
- Validation run for this change: the focused red cleanup tests failed for the
  expected old symbols, then the same focused suite passed after
  implementation. Focused affected pytest, Black, `git diff --check`, flake8,
  mypy, full fast pytest, full pytest, and coverage pytest also passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Shared Period-Average Output Writer

- Added `vercor.output.period_files.write_period_average_netcdf()` as the
  shared log/build/write/clear lifecycle for period-average NetCDF files,
  keeping direct `h5netcdf` access in `vercor.output.netcdf`.
- Refactored JAXGCM and Veros average writers to provide model-specific mean,
  coordinate, and metadata builders to the shared writer while preserving JCM
  shape-derived coordinates/units and Veros native metadata/axis policy.
- Added focused tests for successful writes, data-variable metadata transforms,
  and preserving accumulated samples when a write fails.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture tests to record and
  enforce the shared period-file lifecycle boundary.
- Red/green notes: `tests/test_period_files.py` first failed with missing
  `vercor.output.period_files`; after adding the helper, focused period-file
  tests passed. Mypy then caught overly narrow test callback annotations, which
  were corrected to `Mapping[str, OutputVariable]`.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_files.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_files.py tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_files.py tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-11: Shared JCM/Veros Output Dataset Helpers

- Added `vercor.output.datasets` for shared one-step time-coordinate variables
  and first-use dimension discovery across output-variable maps.
- Extended `vercor.output.period_averages` with shared helpers for accumulating
  `OutputVariable` mappings and converting accumulated period means into
  one-time-step output variables, keeping JCM/Veros-specific extraction,
  metadata, and dimension policy local to their adapters.
- Refactored JAXGCM and Veros period output to use the shared dataset and
  period-output helpers, preserving direct `h5netcdf` output through
  `vercor.output.netcdf` with no xarray conversion in `vercor.output`.
- Hardened the h5netcdf writer to reject reused dimension names with conflicting
  sizes before creating invalid datasets.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and architecture tests to record and
  enforce the shared helper ownership.
- Red/green notes: focused helper tests first failed on missing
  `accumulate_output_variables` and missing `vercor.output.datasets`. After
  adding the helpers and refactoring adapters, the focused output/API suite
  passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_output_datasets.py tests/test_output_netcdf.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Shared h5netcdf Output Helpers

- Added shared period-output conversion helpers in `vercor.output.period_averages`
  for mapping `OutputVariable` values to accumulator samples, applying
  adapter-specific empty-accumulator errors, and reshaping period means into
  one-time-step output variables with explicit dimension ordering.
- Refactored JAXGCM and Veros average writers to use the shared conversion
  helpers while keeping model-specific coordinate extraction, metadata, and
  dimension policy local to their adapters.
- Replaced the final runtime-view `xarray` writer with `OutputVariable` maps and
  the existing `vercor.output.netcdf.write_netcdf_dataset` h5netcdf boundary.
  Tests now read runtime output back with `h5netcdf`, and architecture tests
  assert `vercor.output.runtime` does not import xarray or call `.to_netcdf()`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record shared output helper
  ownership and the direct h5netcdf runtime-output path.
- Red/green notes: new period-helper tests first failed on missing
  `mean_samples_or_raise`; the output ownership test then failed on
  `import xarray as xr` in `vercor.output.runtime`. After adding shared helpers,
  refactoring adapters, and delegating runtime writes to `write_netcdf_dataset`,
  the focused regressions passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip tests/test_api_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`, `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Unified GCM Output Package

- Replaced the parallel JAXGCM/Veros setup-external output modules with the
  canonical `vercor.output` package. Runtime-view final output now lives in
  `vercor.output.runtime` behind lazy top-level reexports, shared period
  accumulation/time/variable/NetCDF helpers live in focused `vercor.output`
  modules, and model-specific period-output adaptation lives in
  `vercor.output.jax_gcm` and `vercor.output.veros`.
- Removed `vercor.setups._external.jax_gcm_output`,
  `vercor.setups._external.period_averages`, and
  `vercor.setups._external.veros_output` without compatibility wrappers. Updated
  JAXGCM/Veros setup and runtime imports plus architecture tests to enforce the
  hard move and centralized `h5netcdf` ownership in `vercor.output.netcdf`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record the new output ownership
  and dependency split. `vercor.output.__init__` keeps runtime-output reexports
  lazy so period-output imports do not pull runtime-state internals into setup
  adapters.
- Red/green notes: the focused output/API suite first failed on
  `vercor.output` still being a module instead of a package. After the package
  migration it passed. The first full suite exposed one stale test monkeypatch
  targeting the lazy top-level facade instead of `vercor.output.runtime`; the
  test was updated to patch the new owner module and the isolated regression
  passed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_jax_gcm_output_frequency.py tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-10: Trivial Internal Wrapper Cleanup

- Removed private pass-through helpers that only forwarded to canonical
  implementations: flux `_as_jax_array()` aliases, JAXGCM/Veros output
  `_jax_array()` aliases, the bilinear interpolator source-mask delegate,
  forcing-data legacy transpose/flip delegates, and the CAMulator land
  temperature-normalization delegate.
- Replaced call sites with direct `as_jax_real_array(...)`,
  `_extrapolation.valid_scalar_source_mask(...)`, and `jnp.flip(...)` calls.
  Public compatibility aliases and documented facade/accessor boundaries were
  left unchanged.
- Removed the stale CAMulator helper-only test while keeping component-level
  JAX-array storage coverage.
- The precise cleanup scan reported no removed-helper definitions or call sites:
  `rg -n "def (_as_jax_array|_jax_array|_prepare_camulator_land_surface_temperature|_legacy_transpose_to_time_last_order|_flip_legacy_latitude_axis)\b|_ensure_src_mask\b|\b(_as_jax_array|_jax_array|_prepare_camulator_land_surface_temperature|_legacy_transpose_to_time_last_order|_flip_legacy_latitude_axis)\(" vercor tests examples`,
- Validation run for this change:
  `conda run -n scipy pytest tests/test_fluxes_utilities.py tests/test_bilinear_rectilinear_interpolator.py tests/test_forcing_data.py tests/test_jax_gcm_output_frequency.py tests/test_data_component_kernels.py tests/test_camulator_component_kernels.py tests/test_external_components_coverage.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `git diff --check`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The broader
  substring cleanup scan still matches preserved public/test names such as
  `torch_tensor_from_jax_array`.

### 2026-06-10: Internal Naming Consistency Pass

- Swept the Python codebase inventory for semantically similar definitions:
  206 Python files, 1,528 functions/methods, and 180 classes were parsed
  successfully. The pass focused on internal/private names only; public and
  ambiguous similarities were left unchanged for API stability.
- Renamed local internal helpers without changing behavior:
  `vercor.fluxes.bulk_formula_cesm._asarray()` is now `_as_jax_array()`,
  the 3-argument callable adapter in
  `vercor.components._callable_wrappers.normalize_component_step_callable()`
  is now `step_fields_context_and_payload()`, and
  `vercor.setups._external.veros_output._coordinate_variable()` is now
  `_extract_coordinate_variable()`.
- Left intentionally parallel names unchanged: JAX/NumPy dtype helpers,
  runtime-cache owner versus resource-facade methods,
  `step_runtime_state()` versus `step_host_runtime_state()`, component factory
  helpers, scalar/vector and inverse helper pairs, and the public historical
  `shr_flux_atmIce()` physics API.
- Updated `DESIGN.md` and `DEPENDENCIES.md` to record the internal naming
  boundary rationale for callable adapters, flux JAX-array normalization, and
  Veros output variable/coordinate extraction. No module dependency order
  changed.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_fluxes_utilities.py tests/test_slab_kernels.py tests/test_component_base_coverage.py tests/test_external_components_coverage.py tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-06-09: JAX-Backed JCM/Veros Output Arrays

- Converted the shared period-average accumulator to store JAX-backed running
  sums, finite-value counts, and mean samples while preserving current
  `nanmean` behavior. Counts now use VerCOR's canonical index dtype.
- Updated JAXGCM and Veros period-output extraction/mean-shaping to keep
  VerCOR-owned output values as JAX arrays and to use `vercor.dtypes` helpers.
  Direct NumPy imports were removed from `jax_gcm_output.py`,
  `period_averages.py`, and `veros_output.py`.
- Added `vercor.host_arrays` helpers for explicit final host transfer and the
  deliberate host `int64` NetCDF time-coordinate exception. The output writers
  now convert to host arrays only at the `h5netcdf` boundary.
- Tightened production NumPy-boundary and output tests so accumulator internals,
  JCM accumulation, and Veros snapshots/accumulation are JAX-backed before file
  writing. Updated `DESIGN.md` and `DEPENDENCIES.md` for the new boundary.
- Validation run for this change:
  baseline
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed after sandbox
  approval for `conda run -n scipy`. Focused red
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_production_numpy_boundaries.py -q --tb=short`
  failed as expected because the accumulator/snapshots were still NumPy-backed
  and the output modules still imported NumPy. After implementation,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_production_numpy_boundaries.py -q --tb=short`,
  `conda run -n scipy mypy vercor/host_arrays.py vercor/setups/_external/period_averages.py vercor/setups/_external/jax_gcm_output.py vercor/setups/_external/veros_output.py tests/test_period_averages.py tests/test_external_components_coverage.py tests/test_production_numpy_boundaries.py`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries -q --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first post-implementation fast suite exposed
  a stale API-boundary assertion that still required NumPy in
  `period_averages.py` and `veros_output.py`; the assertion was updated to make
  `host_arrays.py` the explicit NumPy owner for output host conversion.

### 2026-06-08: Veros Spinup Period-Average Accumulation

- Fixed Veros setup spinup to accumulate selected `output_variables` into the
  same `PeriodAverageAccumulator` used by runtime output, matching the existing
  JAXGCM behavior where spinup samples seed the first averaged output period.
  Spinup still does not write NetCDF files; runtime period gates remain the
  only write boundary.
- Added a shared package-internal Veros output helper that extracts and
  accumulates one Veros state. Runtime output recording and setup spinup now
  use that helper, so extraction/accumulation behavior stays in one owner.
- Added regression coverage proving two Veros spinup steps accumulate two
  selected-output samples without breaking setup-time SST seeding.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs -q --tb=short`
  failed as expected because no spinup states were accumulated. After
  implementation,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs -q --tb=short`,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_initialize_can_spin_up_and_extract_surface_temperature tests/test_external_components_coverage.py::test_veros_initialize_spinup_accumulates_selected_outputs tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first post-implementation focused run exposed
  a test-fixture issue where fake spinup state temperatures drifted by 1 K;
  the fixture was narrowed to vary only `step_id` so the test remains focused
  on period accumulation.

### 2026-06-04: Veros Average Output Dimension Order

- Fixed Veros h5netcdf average output to keep snapshot extraction and
  accumulation in Veros internal array order, then transpose spatial axes once
  at write time. Files now keep VerCOR's lowercase `time` dimension while
  matching native Veros spatial NetCDF order such as
  `temp(time, zt, yt, xt)` and `psi(time, yu, xu)`.
- Expanded Veros average-output coverage for `temp`, `salt`, `u`,
  `surface_taux`, and `psi`. The writer test now asserts each persisted value
  is the elementwise mean of two runtime snapshots after spatial transposition,
  proving period averaging does not reduce a horizontal or vertical axis.
- Black also normalized the existing Veros/JCM example `output_variables` tuple
  formatting in `examples/run_jcm_with_veros.py` while running the requested
  formatter command across `examples`.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`
  failed as expected on the old `("time", "xt", "yt", "zt")` dimension order.
  After implementation,
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_period_averages.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: no failed implementation approach; the focused
  red test caught the intended pre-fix dimension-order regression.

### 2026-06-04: Streaming Period-Average Output Accumulation

- Replaced the JAXGCM and Veros period-output sample buffers with a shared
  host-side `PeriodAverageAccumulator` that stores one running sum and one
  per-element finite-value count array per variable. This preserves current
  `np.nanmean` behavior, including sparse/all-NaN cells, without retaining each
  timestep/snapshot until output.
- Updated JAXGCM output recording to accumulate prediction variables over their
  prediction `time` dimension at record time, then add the NetCDF `time`
  dimension and canonical output ordering only at write time. JAXGCM spinup
  predictions continue seeding the first output period through the accumulator.
- Updated Veros output recording to accumulate selected extracted snapshots and
  write the same `veros.averages.YYYY-MM-DD.nc` mean files with native
  coordinate/metadata preservation. The output writers clear accumulators only
  after successful file writes.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, explicit NumPy/API boundary coverage,
  and accumulator/runtime/writer tests for the new package-internal host output
  boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_period_averages.py -q --tb=short`
  failed as expected on missing `vercor.setups._external.period_averages`.
  After implementation,
  `conda run -n scipy pytest tests/test_period_averages.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py::test_jax_gcm_initialize_uses_provided_forcing_and_can_spin_up tests/test_external_components_coverage.py::test_jax_gcm_step_maps_outputs_and_respects_output_gate tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_jax_gcm_write_output_preserves_model_calendar_attrs tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries tests/test_production_numpy_boundaries.py::test_numpy_imports_match_explicit_host_boundaries tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates -q --tb=short`,
  `conda run -n scipy pytest tests/test_period_averages.py tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py tests/test_jax_gcm_output_frequency.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: an initial focused boundary command used the
  stale test name `test_external_adapter_helpers_stay_in_owner_modules`; rerun
  with `test_setup_helper_and_external_output_ownership_boundaries` covered the
  intended boundary assertions.

### 2026-06-03: Veros h5netcdf Period Output

- Disabled Veros native output machinery in the explicit runtime settings
  boundary by setting `diskless_mode=True` alongside the NumPy backend and
  force-overwrite settings.
- Added opt-in Veros period-output support through `make_veros_gcm` /
  `VerosGCMSetupState` `output_variables` and `output_frequency` arguments.
  Selected Veros variables are extracted with native Veros metadata, current
  timestep selection, ghost-cell removal, and native dimension order, then
  written as period means to `veros.averages.YYYY-MM-DD.nc` via `h5netcdf`.
- Kept output file I/O in the new `vercor.setups._external.veros_output` host
  boundary. Veros runtime now records selected snapshots and flushes through the
  existing JAXGCM day/month/year cadence helper, leaving the SST exchange output
  unchanged when no output variables are selected.
- Validation run for this change:
  baseline
  `conda run -n scipy pytest tests/ -v --fast 2>&1 | tail -20` passed.
  Focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_configure_veros_runtime_sets_diskless_mode tests/test_external_components_coverage.py::test_veros_output_snapshot_uses_variable_metadata_and_current_timestep tests/test_external_components_coverage.py::test_veros_write_output_persists_period_mean_and_coordinates tests/test_external_components_coverage.py::test_veros_output_variables_rejects_bare_string tests/test_external_components_coverage.py::test_veros_constructor_builds_jax_backed_grid tests/test_external_components_coverage.py::test_veros_step_records_selected_outputs_and_writes_on_gate tests/test_external_components_coverage.py::test_veros_step_skips_output_when_no_variables_selected tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries -q --tb=short`
  failed as expected on missing diskless mode, missing `veros_output`, missing
  Veros output API args, and missing runtime output hooks. After implementation,
  focused feature/API/NumPy-boundary checks passed. Then
  `conda run -n scipy pytest tests/test_external_components_coverage.py tests/test_api_boundaries.py tests/test_production_numpy_boundaries.py tests/test_jax_gcm_output_frequency.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first no-op runtime test omitted the
  `_step_function` fake even though `step_veros_runtime` reads it before the
  zero-substep loop; adding an identity fake fixed the fixture. The first writer
  test used fake Veros settings without `coord_degree`, which real Veros
  coordinate metadata requires; adding that setting fixed the fixture. The first
  lint/type pass exposed a Black/E203 slice-spacing conflict and an overly
  precise mutable-sequence type annotation; adding a local `noqa` and matching
  the existing JAXGCM writer's `MutableSequence[Any]` pattern fixed them.

### 2026-06-03: JAXGCM h5netcdf Average Output Writer

- Replaced the JAXGCM averages writer's xarray merge/mean/to_netcdf path with a
  direct `h5netcdf` writer that consumes prediction dynamics/physics/times
  directly, writes runtime-calendar time metadata, preserves JCM unit-table
  metadata, and clears the prediction buffer only after a successful write.
- Updated the host runtime output gate to pass model coordinates, runtime
  timestamp, and the model physics module into the writer. Added coverage that
  `to_xarray()` is not called, h5netcdf dimensions/metadata are persisted, and
  DateTime360/DateTime365 calendar attrs are written.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and the NumPy-boundary allowlist for
  the new output-file host boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset tests/test_api_boundaries.py::test_jax_gcm_average_writer_bypasses_xarray_adapter -q --tb=short`
  failed as expected on the old writer missing the `coords` keyword and still
  importing xarray. After implementation, the focused writer/calendar/runtime
  gate/API checks passed. Then
  `conda run -n scipy pytest tests/test_external_components_coverage.py::test_jax_gcm_write_output_persists_mean_dataset -q --tb=short`,
  `conda run -n scipy pytest tests/test_jax_gcm_output_frequency.py tests/test_external_components_coverage.py tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --tb=short`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first h5netcdf green run used a fake coordinate
  fixture with `layers=3` but only two level centers, causing an incompatible
  NetCDF dimension size; aligning the level coordinate length fixed it. The
  first model-calendar write used a boolean NetCDF attr for
  `fixed_30_day_months`, which h5netcdf rejects in valid NetCDF mode; storing
  the flag as `0`/`1` fixed the file metadata.

### 2026-06-02: Unit-Test Speedup Pass

- Added test-cache defaults in `tests/conftest.py` so Matplotlib and
  fontconfig use writable temp cache paths during pytest while preserving any
  caller-provided environment values.
- Added an internal identity interpolator path for identical-grid bilinear and
  conservative regridders. Identical grids now avoid eager interpolator/remapper
  construction while preserving the existing unchanged-field call behavior.
- Consolidated optional setup-import boundary checks from two subprocesses per
  import case to one isolated subprocess per case, keeping both output-marker
  and heavy-optional-module assertions. Reduced the runtime profile smoke grid
  from 4x3 to 2x2 while keeping the parser/build/run/cache coverage.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py::test_regridder_identical_grid_skips_interpolator_construction tests/test_conservative_rectilinear_regridder.py::test_regridder_identical_grid_skips_remapper_construction -q --tb=short`
  failed as expected on the constructors still building the expensive
  interpolator/remapper. Focused cache red
  `conda run -n scipy pytest tests/test_tools_components_and_plotting.py::test_test_environment_uses_writable_plotting_cache_defaults -q --tb=short`
  failed as expected on missing `MPLBACKEND`.
  After implementation, the same focused checks passed. Then
  `conda run -n scipy pytest tests/test_bilinear_rectilinear_regridder.py tests/test_conservative_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_unrelated_setup_imports_do_not_initialize_optional_adapters -q --tb=short --durations=10`,
  `conda run -n scipy pytest tests/test_tools_components_and_plotting.py::test_plot_component_scalar_vector_comparison_aligns_axes_and_shapes -q --tb=short --durations=5`,
  `conda run -n scipy pytest tests/test_runtime_run_cache.py -q --tb=short --durations=10`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short --durations=25`,
  `conda run -n scipy pytest tests/ -q --tb=short --durations=25`, and
  `conda run -n scipy pytest --cov=vercor tests/ -q --tb=short` passed.
- Final duration checks: the fast-suite plotting test dropped from the prior
  ~15s hotspot to 0.22s in the fast duration table after cache warm-up; the
  optional setup-import probes dropped from ~2.4-2.8s each to about 1.02-1.59s
  in the final fast duration table and 0.95-1.16s in the final full duration
  table; runtime profile smoke dropped from ~4.8s focused to 0.69s focused.
  Coverage remained source-focused at 90% total. The existing Black Python
  3.13/target warning and JAX dtype-promotion warning remain.

### 2026-06-02: CAMulator Wind-Filter Boundary Refactor

- Split CAMulator wind artifact tensor mechanics into private
  `vercor.setups._external._camulator_wind_filtering`, which now owns PyTorch
  mask/kernel artifact construction, field filtering, and selected in-place
  tensor updates.
- Kept `vercor.setups._external.camulator_wind_filter` as the public facade for
  configuration loading/validation, compatibility functions, and the existing
  log-and-skip failure policy used during optional post-processing.
- Added focused behavior and architecture coverage for shape-stable wind-filter
  artifacts, target-only tensor mutation, log-and-skip failure handling, and the
  public-to-private wind-filter boundary. Updated `DESIGN.md` and
  `DEPENDENCIES.md` for the new owner split.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_camulator_component_kernels.py tests/test_api_boundaries.py -q --fast --tb=short`
  failed as expected on missing
  `vercor.setups._external._camulator_wind_filtering` and the missing private
  boundary file assertion. Focused green with the same command passed after
  implementation. Then `conda run -n scipy black vercor examples tests`,
  focused post-format pytest with the same command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short` passed. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first flake8 pass found one Black-formatted
  computed slice with E203 in `_camulator_wind_filtering.py`; rewriting the
  split helper to use explicit `start`/`end` variables fixed the style issue.

### 2026-06-02: External Adapter Factory/Setup-State Boundary Refactor

- Removed the public `JCMState` compatibility aliases from
  `vercor.setups._external` and `vercor.setups._external.jax_gcm`; the canonical
  state bundle owner is now only `vercor.setups._external.jax_gcm_state`.
- Split CAMulator atmosphere setup-state ownership into
  `vercor.setups._external.camulator_gcm_state.CAMulatorGCMSetupState`, leaving
  `vercor.setups._external.camulator` as a thin `make_camulator_gcm(...)`
  factory that binds the host-component lifecycle methods.
- Updated boundary/runtime tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  stricter JAXGCM public surface and CAMulator setup-state owner.
- Validation run for this change:
  initial focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_camulator_component_kernels.py tests/test_coupler_runtime.py -q --fast --tb=short`
  first exposed an over-eager test import of the not-yet-created CAMulator
  setup-state module; after moving that assertion back to the boundary test,
  the same focused red command failed as expected on the remaining `JCMState`
  alias and missing `camulator_gcm_state.py`. Focused green with the same
  command passed after implementation. Then
  `conda run -n scipy black vercor examples tests`,
  focused post-format pytest with the same boundary/runtime command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short` passed. The first
  full `conda run -n scipy pytest tests/ -q --tb=short` exposed one stale
  full-only boundary assertion in `tests/test_runtime_state.py` that still
  looked for `def step(...)` in `camulator.py`; the exact failing test then
  passed after retargeting it to `camulator_gcm_state.py`. Final Black,
  focused boundary/runtime pytest, flake8, mypy, full fast pytest, and full
  pytest all passed. The existing Black Python 3.13/target-3.14 warning and
  JAX dtype-promotion warning remain.
- Failed approaches recorded: the initial top-level import of
  `camulator_gcm_state` in the CAMulator kernel tests caused collection to fail
  before the intended red boundary assertion, and the first full suite run found
  a stale boundary test that still treated the CAMulator factory as the
  setup-state owner. Both were corrected in the test harness.

### 2026-06-02: Asset and Forcing-Data Boundary Refactor

- Refactored `vercor.assets` so the generic asset cache/download/checksum layer
  uses private normalized asset helpers and no longer embeds forcing-specific
  error wording. Concrete forcing registries remain in
  `vercor.setups._data.assets`.
- Split `vercor.forcing_data.read_forcing()` into private path-resolution,
  NetCDF variable lookup, legacy transpose, and latitude-flip helpers while
  preserving successful array behavior. Missing mapping keys and missing
  NetCDF variables now raise distinct `KeyError` messages.
- Added explicit `year_type` validation in `vercor.forcing_index`, with
  `vercor.calendar` compatibility delegates preserving the same behavior.
- Added focused asset and forcing-data test files, updated the existing
  read-forcing coverage expectation, and kept the small focused tests included
  in `--fast` runs.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the refined asset and
  forcing-data boundaries, including removal of the obsolete
  data-component-reader-class wording.
- The broader audit findings for public `JCMState` reexport cleanup and
  CAMulator setup-state splitting were intentionally left out of this
  asset/forcing-data pass and completed in the later 2026-06-02 external
  adapter factory/setup-state boundary refactor.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_assets.py tests/test_forcing_data.py tests/test_tools_time_and_forcing.py::test_forcing_index_rejects_unknown_year_type tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip -q --fast --tb=short`,
  focused green after implementation with the same focused command,
  focused boundary/API
  `conda run -n scipy pytest tests/test_assets.py tests/test_forcing_data.py tests/test_tools_time_and_forcing.py tests/test_component_base_coverage.py::test_read_forcing_and_runtime_write_round_trip tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`, focused post-format with
  the same boundary/API command,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red tests for generic asset wording,
  missing-variable error reporting, and invalid forcing `year_type` validation.

### 2026-06-02: Calendar Forcing-Index Boundary Refactor

- Added `vercor.forcing_index` as the focused owner for daily forcing lookup
  policy, including Gregorian month lengths, noleap mapping, 360-day to
  Gregorian day mapping, one-based forcing day selection, and zero-based daily
  forcing indexes.
- Kept `vercor.calendar` focused on calendar constants, model-calendar datetime
  values, leap-year logic, and month/day conversion while preserving the
  historic forcing-index imports through thin compatibility delegates.
- Updated `vercor.time_selection` and `vercor._runtime.time` to import forcing
  policy from `vercor.forcing_index`, removing the local 360-day wrapper from
  time selection.
- Added boundary and behavior coverage for forcing-index ownership, runtime
  import direction, absence of a `vercor.forcing_index` top-level import cycle,
  and compatibility-delegate parity across Gregorian, noleap, leap-day, and
  360-day daily forcing cases.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new forcing-index owner.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  focused runtime
  `conda run -n scipy pytest tests/test_coupler_runtime.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`, focused post-format
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_tools_time_and_forcing.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red boundary/behavior tests before
  adding `vercor.forcing_index` and moving runtime imports to the new owner.

### 2026-06-02: Bilinear Interpolator Private-Owner Boundary Refactor

- Split bilinear interpolation internals into private owner modules under
  `vercor.interpolators`: `_bilinear_geometry` for spherical geometry and
  orientation checks, `_bilinear_weights` for target-to-source cell lookup and
  bilinear weights, and `_bilinear_extrapolation` for nearest/IDW fill policy
  plus valid-source mask normalization.
- Kept `BilinearRectilinearInterpolator` as the public PyTree facade with the
  same constructor options, public precomputed weight attributes, scalar/vector
  methods, JIT behavior, and regridder integration.
- Added architecture-boundary coverage for private helper ownership, package
  import-cycle absence, private helper import direction, periodic dateline
  weight construction, and empty-valid-source extrapolation fill behavior.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the bilinear private-owner
  boundary and recorded the `calendar.py` forcing-index split as a follow-up.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py -q --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  focused post-format
  `conda run -n scipy pytest tests/test_bilinear_interpolator_boundaries.py tests/test_bilinear_rectilinear_interpolator.py tests/test_bilinear_rectilinear_regridder.py -q --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional red boundary tests before adding the
  private helper owner modules and public facade delegation.

### 2026-06-02: Logging Facade Private-Owner Boundary Refactor

- Split the former monolithic `vercor.jax_logging` implementation into private
  owner modules under `vercor._logging`: `config` for canonical logger setup,
  `protocols` for logger-like contracts and level checks, `host` for host-side
  message formatting/emission, and `callback` for traced-value partitioning plus
  `JaxCallbackLogger`.
- Kept `vercor.jax_logging` as the only production-facing public facade with an
  explicit `__all__`, preserving existing public imports and callback logging
  behavior while preventing production modules from importing private logging
  internals directly.
- Added architecture-boundary coverage for the thin facade, private logging
  package ownership, private package cycle absence, public API preservation, and
  production import direction.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the logging facade/private-owner
  boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  `conda run -n scipy pytest tests/test_logging_boundaries.py -q --tb=short`,
  focused green after implementation
  `conda run -n scipy pytest tests/test_logging_boundaries.py tests/test_coupler_coverage.py -q --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  focused post-format
  `conda run -n scipy pytest tests/test_logging_boundaries.py tests/test_coupler_coverage.py -q --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a migrated
  canonical log-format mismatch (`VerCOR ─ ...` instead of the tested
  `VerCOR: ...` format). The private config constant was corrected before
  continuing validation.

### 2026-06-01: External Adapter Setup-State Boundary Refactor

- Added `vercor.setups._external.jax_gcm_state` as the owner for JAXGCM setup
  state, model construction, spinup, and lifecycle callback wiring; the public
  `jax_gcm.py` module now stays focused on the `make_jax_gcm(...)` factory and
  then-existing `JCMState` reexport, which was later removed in the 2026-06-02
  external adapter factory/setup-state boundary refactor.
- Added `vercor.setups._external.veros_gcm_state` as the owner for Veros setup
  state, grid derivation, spinup, and host step delegation; the public
  `veros_gcm.py` module now stays focused on `make_veros_gcm(...)`.
- Added the named tuple-compatible `VerosForcingFields` container so Veros
  forcing fields have explicit names while preserving existing tuple unpacking.
- Strengthened architecture coverage for external setup-state owners, factory
  thinness, external package import cycles, and named Veros forcing fields.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new external adapter
  setup-state boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_api_boundaries.py::test_setup_helper_and_external_output_ownership_boundaries tests/test_api_boundaries.py::test_jax_gcm_factory_uses_named_runtime_callbacks tests/test_api_boundaries.py::test_external_package_has_no_top_level_import_cycles tests/test_external_components_coverage.py::test_veros_prepare_surface_forcing_fields_shapes_nan_cleanup_and_qnec_gate -q --fast --tb=short`,
  focused green after implementation,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py tests/test_coupler_runtime.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failures were the intentional focused red tests before adding the
  setup-state owner modules and named Veros forcing container.

### 2026-06-01: External Adapter Helper Boundary Refactor

- Renamed exported external-adapter helper functions to public
  package-internal names in JAXGCM, CAMulator, and Veros setup modules, while
  leaving underscored helpers as local implementation details.
- Removed private underscored helpers and setup-state classes from literal
  external adapter `__all__` exports.
- Updated external runtime call sites so adapter modules no longer reach through
  another module's private helper namespace for JAXGCM field mapping,
  CAMulator tensor/field staging, CAMulator optional-dependency loading, or
  Veros host-state mutation helpers.
- Strengthened architecture coverage so external adapter `__all__` exports
  cannot drift back to private names and runtime helpers use the named
  helper boundary.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the explicit external-adapter
  helper boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing public helper names/private `__all__` exports, focused
  green after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_external_components_coverage.py tests/test_camulator_component_kernels.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused red run imported renamed
  CAMulator helpers directly from the module and failed at collection time.
  The test was adjusted to access helpers through the module object so the red
  phase exercised the intended missing boundary attributes.

### 2026-06-01: Component Lifecycle Boundary Refactor

- Added a typed private lifecycle-owner boundary in
  `vercor.components._lifecycle` and narrowed component authoring protocols so
  lifecycle storage is no longer exposed as `Any`.
- Grouped callable factory lifecycle callbacks into one
  `LifecycleHooks` container inside callable construction metadata.
- Centralized runtime-payload hook precedence in
  `ComponentLifecycleMixin`; callable wrappers now provide only the default
  callable payload fallback when no custom payload hook is installed.
- Strengthened boundary and behavior coverage for typed lifecycle ownership,
  callable-wrapper hook dispatch, callable payload preservation, and custom
  payload-hook override behavior.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the lifecycle-owner/callable
  lifecycle boundary.
- Validation run for this change:
  focused red
  `conda run -n scipy pytest tests/test_component_boundary_contracts.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  focused green after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failure was the intentional focused red test run before adding the
  typed lifecycle-owner boundary and grouped callable hook metadata.

### 2026-06-01: Component Execution Protocol Boundary Refactor

- Added private structural execution protocols in `vercor.components._protocols`
  so component host/scanned execution policy no longer imports concrete
  `Component` or `HostRuntimeComponent` classes.
- Updated `vercor.components.runtime_execution` to detect host-backed runtime
  components through `HostRuntimeExecutionProtocol` while preserving the
  existing public helpers and host-runtime error behavior.
- Narrowed runtime-only context imports in component modules to type-checking
  imports where they are only annotation support.
- Strengthened architecture coverage so runtime execution must use private
  protocols and cannot drift back to concrete component-class imports.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for protocol-backed execution
  policy ownership.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing execution protocols and concrete-class runtime
  execution import, focused green boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_component_boundary_contracts.py tests/test_component_base_coverage.py tests/test_api_boundaries.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: no failed implementation approach was encountered;
  the only failure was the intentional focused red test run before adding the
  private execution protocols.

### 2026-06-01: Component Context Boundary Refactor

- Added `vercor.components.contexts` as the canonical owner for
  `ComponentSetupContext` and `ComponentStepContext`.
- Removed the internal `vercor._runtime.contexts` module and replaced production
  and test imports of `ComponentInitContext` / `RuntimeStepContext` with the
  public component-author context names.
- Updated component contracts and package facades so context dataclasses are
  reexported from `vercor.components` and `vercor` through the component-owned
  boundary, while hook type aliases remain in `vercor.components.contracts`.
- Strengthened architecture coverage so context dataclasses cannot drift back
  into runtime ownership and setup adapters remain free of runtime context/store
  imports.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for component-context ownership.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused red
  pytest for the missing `vercor.components.contexts` owner, focused green
  boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a runtime
  import of `ComponentStepContext` in `vercor.components.base`, which widened
  the base module surface. The import is now `TYPE_CHECKING`-only so the base
  module stays narrow while annotations remain type-checkable.

### 2026-06-01: Runtime Topology Policy Boundary Refactor

- Added `vercor._runtime.topology_state` as the neutral owner for grouped
  `RuntimeTopologyMaps`, `SurfaceExchangeMasks`, and `ExchangeTopologyState`.
- Split generic exchange regridder/identity-mask map construction into
  `vercor._runtime.exchange_topology`, and moved ATM/OCN/LND surface-mask
  creation, validation, and bilinear mask patching into
  `vercor._runtime.surface_masks`.
- Reduced `vercor._runtime.topology` to orchestration: it composes generic
  exchange topology maps with surface masks and returns one explicit topology
  state. Runtime resources and initialization now import topology state
  contracts from the neutral state module, and `Coupler.initialize()` reads
  public mask attributes through `topology.surface_masks`.
- Strengthened boundary coverage so topology state, generic exchange maps, and
  surface-mask policy cannot drift back into one mixed-responsibility topology
  module.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the topology-state,
  exchange-topology, surface-mask, and topology-orchestration split.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused
  red pytest for the missing `topology_state`/`surface_masks` split, focused
  green topology/boundary pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_api_boundaries.py tests/test_coupler_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first focused green run exposed a stale
  boundary assertion that still expected `vercor._runtime.topology` to import
  component-topology lookup helpers; it now checks the new surface-mask owner.
  The first flake8 pass reported one stale unused test local left by the import
  move; it was removed before rerunning flake8.

### 2026-06-01: Runtime Compilation Cache Boundary Refactor

- Added `vercor._runtime.compilation` as the neutral owner for
  `CompiledRuntime` and `RuntimeCompilationKey`.
- Moved context-derived compiled-runtime cache-key construction onto frozen
  `RuntimeRunContext`, leaving `CompiledRuntimeCache` focused on compiled
  callable storage, JIT wrapping, clearing, count, and value inspection.
- Updated the scanned runner to pass
  `context.compiled_runtime_cache_key(...)` into `get_or_compile(...)`, and
  updated runtime resources to type compiled cache values through the neutral
  compilation module instead of importing the alias from `run_context`.
- Strengthened boundary coverage so `vercor._runtime.cache` cannot drift back to
  importing or mentioning `RuntimeRunContext`, context-aware cache helpers, or
  context-derived key construction.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the compilation alias,
  run-context key, and cache storage/JIT ownership split.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`, focused
  red pytest for the missing neutral compilation boundary and old runner cache
  helper call, focused green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_api_boundaries.py tests/test_runtime_run_cache.py tests/test_runtime_interrupts.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first focused green run exposed a stale runner
  boundary assertion that forbade `compiled_runtime_cache_key(` anywhere in
  `runner.py`; it now forbids runner-owned key definitions while requiring the
  intended `context.compiled_runtime_cache_key(...)` call in the compiled
  scanned helper.

### 2026-06-01: Compiled Runtime Cache Boundary Refactor

- Added `vercor._runtime.cache.CompiledRuntimeCache` as the owner for compiled
  scanned-runtime cache storage, JIT wrapping, context-derived cache-key lookup,
  clearing, count, and value inspection.
- Changed `RuntimeRunContext` to carry the cache owner instead of a mutable
  mapping, and changed `CouplerRuntimeResources` to store a private
  `_runtime_cache` while delegating its public cache facade methods to that
  owner.
- Removed the raw `runtime_cache_mapping()` accessor and updated the scanned
  runner to ask the cache owner for compiled runtime reuse rather than importing
  cache-key and cache-mutation helpers directly.
- Strengthened boundary coverage so resources no longer expose the raw cache
  mapping, run-context annotations cannot drift back to `MutableMapping`, and
  cache compile/reuse/inspection behavior remains owned by
  `CompiledRuntimeCache`.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the cache-owner boundary.
- Validation run for this change:
  baseline `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  focused red pytest for the missing cache owner and raw mapping leak, focused
  green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_state.py tests/test_runtime_facade_boundaries.py tests/test_runtime_run_cache.py tests/test_runtime_interrupts.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first runner boundary assertion searched for
  `compiled_scanned_runtime(` across the entire runner module and also matched
  the intended private `_run_compiled_scanned_runtime(...)` helper; it now checks
  only the removed cache-helper import/call shape.

### 2026-06-01: Runtime Resource and Topology Boundary Refactor

- Added `vercor._runtime.component_topology` as the owner for default
  topology component-name validation and component lookup, leaving
  `vercor._runtime.topology` focused on exchange regridder/mask setup.
- Added grouped `RuntimeTopologyMaps` and changed `ExchangeTopologyState` to
  carry topology maps as one boundary object instead of exposing three parallel
  map fields.
- Refactored `CouplerRuntimeResources` into a slotted private-field holder with
  explicit topology, contract, runtime-cache, and interrupt accessors; runtime
  facade/preparation code no longer reaches through to raw resource
  dictionaries.
- Made `RuntimeRunContext` frozen to document that run-context identity is a
  static execution input bundle.
- Updated focused boundary coverage plus runtime/topology/output tests so the
  new ownership cannot drift back into `vercor._runtime.topology` or raw
  resource attributes.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for component-topology ownership,
  grouped topology maps, private runtime resources, and the frozen run context.
- Validation run for this change:
  focused red pytest for the missing ownership/resource boundaries,
  focused green pytest after implementation,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy pytest tests/test_runtime_facade_boundaries.py tests/test_runtime_state.py tests/test_coupler_coverage.py -q --fast --tb=short`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first new boundary test imported
  `RuntimeTopologyMaps` at module import time and failed during collection
  instead of as an assertion; it now imports dynamically inside the test. The
  first full fast suite exposed one stale API-boundary assertion that still
  expected topology-name validation in `vercor._runtime.topology`; it now checks
  `vercor._runtime.component_topology`.

### 2026-06-01: Runtime State Validation Boundary Refactor

- Moved configured runtime-state/topology validation from
  `vercor._runtime.coupler_state` into `vercor._runtime.state_validation`, leaving
  coupler-state ownership focused on immutable runtime state assembly and
  runtime-contract refresh.
- Updated `vercor._runtime.preparation` to call the new validation owner while
  preserving its preparation-facing validation wrapper and public runtime
  behavior.
- Strengthened boundary coverage so validation ownership cannot drift back into
  `vercor._runtime.coupler_state` or the public `Coupler`/runtime-facade
  boundary.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new validation ownership.
- Validation run for this change:
  focused red pytest for the missing validation owner,
  focused green pytest after the move,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-06-01: Runtime Output Boundary Refactor

- Moved output mask selection and naming from `vercor._runtime.coupler_state`
  into `vercor.output`, leaving runtime coupler-state ownership focused on
  immutable state assembly, contract refresh, and validation.
- Removed `vercor.output`'s dependency on `vercor._runtime.coupler_state`; final
  output iteration now owns its view construction, file naming, and mask lookup
  in one output boundary.
- Strengthened boundary coverage so `output_masks_for_component(...)` stays in
  `vercor.output` and cannot drift back into runtime-state assembly.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new output-mask ownership.
- Validation run for this change:
  focused red pytest for the ownership move,
  focused green pytest after the move,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-05-28: Component and Runtime Boundary Alias Refactor

- Moved public lifecycle hook type aliases to `vercor.components.contracts` and
  reexported them from `vercor.components` and `vercor`, leaving
  `vercor.components._lifecycle` focused on private hook storage and
  installation.
- Added shared callable component construction metadata in
  `vercor.components._callable_wrappers` so differentiable and host
  `from_model()` paths share field-spec, payload, settings, and lifecycle-hook
  normalization.
- Split `vercor._runtime.runner.run_coupler_runtime()` into smaller helpers for
  compiled scanned execution and host-runtime donation rejection while
  preserving public runtime behavior.
- Strengthened boundary and lifecycle coverage for public hook ownership,
  callable construction ownership, direct `from_model()` lifecycle hooks, and
  runner path-selection helpers.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new component and runtime
  boundary ownership.
- Validation run for this change:
  focused component/runtime-cache fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first runner boundary test split source through
  the next public function and accidentally included the new private helper
  body; it now extracts only `run_coupler_runtime()`.

### 2026-05-28: Runtime Preparation Boundary Refactor

- Added `vercor._runtime.preparation` as the focused owner for prepared runtime
  state construction, contract refresh for prepared states, runtime-state
  validation, and initial outgoing-store priming.
- Kept `vercor._runtime.facade` as the coupler-facing orchestration boundary by
  reexporting preparation helpers while leaving dispatch/run context
  construction, execution delegation, runtime views, and final output delegation
  in the facade.
- Centralized `CompiledRuntime` in `vercor._runtime.run_context` and exchange
  field/factory aliases in `vercor.exchange`, removing duplicate alias
  ownership from runtime resources and setup helper modules.
- Strengthened boundary tests for runtime preparation ownership, facade
  reexports, runtime import-cycle absence, shared compiled-runtime typing, and
  exchange alias ownership.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the new preparation and alias
  ownership boundaries.
- Validation run for this change:
  focused runtime/API/state boundary fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approaches recorded: the first `RuntimePreparationInputs` protocol used
  mutable attributes, which mypy rejected for frozen `RuntimeInputs`; the
  protocol now uses read-only properties. The first full pytest run exposed a
  stale architecture assertion that still expected preparation logic in
  `runtime.facade`; the assertion now checks `runtime.preparation` as the owner.

### 2026-05-28: Runtime Resource Boundary Refinement

- Added public `Coupler.clear_runtime_cache()` and
  `Coupler.runtime_cache_entry_count()` as the small profiling-facing runtime
  cache API.
- Added grouped runtime-resource helpers for topology-map replacement and
  compiled-cache clear/count/value inspection, keeping cache dictionaries and
  synthetic topology setup behind the runtime resource holder.
- Updated the runtime profiling example and focused runtime tests to use the
  public cache facade or named test helpers instead of raw runtime cache and
  topology assignments.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the refined resource boundary.
- Validation run for this change:
  focused runtime-resource/API fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches.

### 2026-05-28: Whole-Codebase Boundary Refactor

- Added shared canonical grid-field shape helpers in `vercor.field_layout`, then
  routed component-facing and runtime-facing required-field validation through
  the shared message/shape policy while preserving existing exception types.
- Narrowed private component helper protocols so runtime helpers no longer
  require setup data storage, and added focused boundary tests for protocol
  ownership.
- Added `vercor._runtime.facade.RuntimeInputs` so `Coupler` passes one
  grouped internal runtime input bundle into facade helpers instead of repeated
  component/exchange/resource parameter clumps.
- Added lightweight CAMulator runtime field contract ownership in
  `vercor.setups._external.camulator_contracts`, leaving tensor/field mapping
  code focused on runtime arrays.
- Split reusable architecture-test helpers out of `tests/test_api_boundaries.py`
  and added focused tests for field layout, component boundaries, runtime facade
  boundaries, and CAMulator contracts.
- Validation run for this change:
  focused new boundary pytest,
  focused API/runtime/component/CAMulator fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the initial shared field-layout error wording broke
  an existing component data-field regex contract; the helper now preserves the
  component-facing message shape while still centralizing shape validation.

### 2026-05-28: Component Protocol and Runtime Resource Boundary Refactor

- Added private component helper protocols in `vercor.components._protocols` so
  runtime-field, validation, lifecycle, and callable-wrapper helpers depend on
  structural component contracts instead of type-only imports from the public
  `Component` base class.
- Added grouped `CouplerRuntimeResources.replace_contracts(...)` and
  `replace_topology(...)` methods, then routed runtime-facade contract/topology
  refreshes through the resource holder instead of assigning individual maps in
  facade code.
- Strengthened boundary tests for component helper protocol ownership, runtime
  resource replacement, and runtime-facade assignment cleanup; updated
  `DESIGN.md` and `DEPENDENCIES.md` for the new ownership map.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_runtime_state.py tests/test_component_base_coverage.py -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- Failed approach recorded: the first full pytest run exposed an overly exact
  existing source-boundary assertion for the former topology regridder import;
  the assertion now checks topology-owner imports without depending on one-line
  import formatting.

### 2026-05-28: Obsolete Compatibility Active-Doc Audit

- Audited live source, tests, examples, `README.md`, `DESIGN.md`,
  `DEPENDENCIES.md`, and active `PROGRESS.md` for obsolete compatibility import
  paths and shim modules while leaving the historical archive untouched.
- Confirmed the removed facade modules and shim paths remain absent from the
  live tree; remaining runtime-payload references use the canonical
  `vercor.setups._external.jax_gcm_runtime` owner or boundary tests that assert
  removed reexports stay removed.
- Tightened API-boundary coverage so active `PROGRESS.md` no longer advertises
  removed compatibility surfaces as current preserved API, then refreshed stale
  progress entries to point at later canonical ownership.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black Python
  3.13/target-3.14 warning and JAX dtype-promotion warning remain.
- No failed implementation approaches. The audit found no remaining live
  obsolete shim modules requiring production-code deletion.

### 2026-05-28: Obsolete Compatibility API Cleanup

- Removed compatibility-only runtime reexports from `vercor._runtime`; code,
  examples, and tests now import runtime contracts, state containers, stores,
  step metadata, and exchange dispatch from their focused owner modules.
- Removed obsolete compatibility aliases and methods:
  `vercor.setups._external.jax_gcm.JAXGCMRuntimePayload`,
  the external setup lazy payload export, `ComponentSettings`,
  `ComponentForcingData._read_forcing()`, CAMulator dictionary metadata
  accessors, and private `Coupler` runtime resource/scanned-run shims.
- Added test-only runtime helpers for focused scanned-runtime and
  state-from-components coverage without restoring production compatibility
  methods.
- Updated boundary tests, runtime/coupler/cache/interrupt/forcing/CAMulator
  tests, `DESIGN.md`, and `DEPENDENCIES.md` for the canonical API paths.
- Validation run for this change:
  focused obsolete-API/CAMulator/forcing pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-28: Runtime Resource Holder Boundary Refactor

- Added `vercor._runtime.resources.CouplerRuntimeResources` as the owner for
  per-coupler runtime topology maps, refreshed runtime contracts, compiled
  runtime cache, and interrupt controller.
- Updated `Coupler` to store one runtime resource holder while keeping
  then-current private runtime test/profiling aliases; a later cleanup removed
  those aliases.
- Routed runtime facade initialization, state creation, validation, dispatch/run
  context construction, execution, and finalization through the resource holder
  instead of repeated map/cache/interrupt arguments.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new runtime
  resource ownership map.
- Validation run for this change:
  focused runtime/Coupler/cache/interrupt fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Facade and CAMulator Tensor Index Refactor

- Added `vercor._runtime.facade` as the high-level orchestration boundary used by
  `Coupler` for runtime-state creation, validation, dispatch/run context
  construction, execution, runtime views, and final output delegation.
- Slimmed `vercor.coupler` so it delegates runtime internals through the facade
  while retaining then-current test delegates; later cleanup removed the
  private compatibility methods.
- Added typed `TensorVariableIndex` metadata for CAMulator tensor access; the
  temporary dictionary metadata accessor was later removed in favor of
  `StateVariableAccessor.get_var_index(...)`.
- Updated boundary/CAMulator tests, `DESIGN.md`, and `DEPENDENCIES.md`.
- Validation run for this change:
  focused runtime/API/Coupler/CAMulator fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: External Adapter State Boundary Refactor

- Added private runtime-state protocols for JAXGCM, Veros, and CAMulator
  runtime helpers so adapter runtime modules no longer accept unbounded setup
  state objects in their public helper signatures.
- Replaced JAXGCM factory lambda lifecycle wiring with named callbacks bound by
  `functools.partial`, and replaced the Veros host step lambda with a named
  private step adapter.
- Kept Veros optional runtime settings lazy by importing `runtime_settings`
  inside `configure_veros_runtime()`, and made CAMulator wind-filter
  configuration fail with explicit `ValueError`s while removing mutable function
  defaults.
- Updated boundary/focused tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  tightened external-adapter state ownership map.
- Validation run for this change:
  focused external/API/CAMulator fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Callable Component Boundary Refactor

- Moved concrete callable-backed component classes next to their owning public
  runtime-kind bases: `vercor.components.base` now owns the differentiable
  callable wrapper and `vercor.components.host` now owns the host-runtime
  callable wrapper.
- Slimmed `vercor.components._callable_wrappers` to shared callable signature
  normalization and runtime step-result application, and kept
  `vercor.components.factories` as public helper delegates instead of an
  internal callable construction owner.
- Strengthened boundary tests to reject the old factory import path, callable
  wrapper imports of concrete component bases, and the removed
  `_create_callable_component` / `CallableComponentRequest` construction path.
- Updated `DESIGN.md` and `DEPENDENCIES.md` for the callable wrapper ownership
  map.
- Validation run for this change:
  focused API/callable fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: External Adapter Runtime Boundary Refactor

- Added focused runtime owners for external adapter behavior:
  `jax_gcm_runtime.py` owns JAXGCM runtime payload, defaults, prefill,
  validation, stepping, and host recording; `camulator_runtime.py` owns
  CAMulator datetime coercion, prediction-block execution, and runtime step
  mapping; `veros_runtime.py` owns Veros flux application, host substeps, and
  SST refresh.
- Slimmed `jax_gcm.py`, `camulator.py`, and `veros_gcm.py` back toward
  optional-dependency loading, model construction, setup initialization, spinup,
  and factory wiring while preserving existing public factories. The JAXGCM
  runtime payload is now owned by `jax_gcm_runtime.py` rather than reexported by
  `jax_gcm.py`.
- Updated boundary/focused tests, `DESIGN.md`, and `DEPENDENCIES.md` for the
  new external-adapter ownership map.
- Validation run for this change:
  focused external/runtime fast pytest,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Core Boundary Mixin Extraction

- Split `Component` helper behavior into focused private modules:
  `_field_names`, `_field_authoring`, `_runtime_access`, and `_lifecycle_api`,
  leaving `vercor.components.base` focused on the abstract component contract
  and callable factory entrypoint.
- Made `Coupler.run_sequence` an explicit empty `RunSequence` by default and
  removed dynamic `hasattr`/`getattr` branches from runtime preparation.
- Routed the slab example's ICE diagnostic read through `RuntimeComponentView`
  instead of direct runtime-state store access.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new helper
  ownership map.
- Validation run for this change:
  focused boundary/component/coupler fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Component Author API Split

- Added canonical public component-author modules:
  `vercor.components.contracts` for field contracts/context aliases,
  `vercor.components.data` for `DataComponent`, and
  `vercor.components.host` for `HostRuntimeComponent`.
- Slimmed `vercor.components.base` to the abstract `Component` contract and
  moved concrete component-kind imports in factories, callable wrappers,
  runtime execution policy, tests, and package facades to the new module owners.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  component-author ownership map.
- Validation run for this change:
  focused component/API fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Diagnostics Runtime View Boundary Refactor

- Added runtime-owned `runtime_field_candidates(...)` and `runtime_field(...)`
  helpers in `vercor._runtime.views`, and routed `RuntimeComponentView` read
  helpers through them.
- Updated diagnostics to use the runtime-view field lookup boundary instead of
  reaching into runtime stores with `.data.get(...)` or `getattr(...)`, while
  preserving `component_vector_speed(...)` compatibility with runtime states.
- Cleaned `examples/run_data_driver.py` diagnostics wiring and kept its
  component typing on the public top-level facade. A first full-suite run caught
  the direct `vercor.components` example import boundary regression; the example
  now imports `Component` from `vercor`.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the shared
  runtime field-resolution ownership.
- Validation run for this change:
  focused diagnostics/runtime-view fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime View and Component Boundary Refactor

- Moved component setup validation and component host/scanned execution policy
  into explicit component-owned bridge modules:
  `vercor.components.setup_validation` and
  `vercor.components.runtime_execution`, removing runtime imports of the old
  private component helper modules.
- Added read helpers to `RuntimeComponentView` and routed diagnostics/output
  field access through that view abstraction instead of iterating runtime store
  internals directly.
- Added `Coupler.runtime_component_views(...)` and updated multi-view examples
  to reuse that public facade.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  ownership map.
- Validation run for this change:
  focused boundary/view fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Run Boundary Refactor

- Moved `RuntimeRunContext` and the compiled-runtime type alias into
  `vercor._runtime.run_context`, leaving static topology on
  `RuntimeDispatchContext` instead of duplicating it in the run context.
- Moved compiled-runtime cache-key and JIT wrapping policy into
  `vercor._runtime.cache`, and moved host/scanned progress formatting plus JAX
  callback logging helpers into `vercor._runtime.progress`.
- Slimmed `vercor._runtime.runner` to run-mode selection, host/scanned loops,
  donation checks, and interrupt translation while preserving `Coupler.run()`
  behavior and runtime PyTree shapes.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for the new
  ownership map.
- Validation run for this change:
  focused runtime fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Dispatch Boundary Refactor

- Moved static runtime dispatch context construction into
  `vercor._runtime.dispatch_context`, leaving `vercor._runtime.coupler_state`
  focused on runtime state assembly, contract refresh, validation, and output
  masks.
- Added private `vercor.components._runtime_execution` for host-component
  detection and host/scanned component step selection, so `vercor._runtime.driver`
  no longer owns `HostRuntimeComponent` classification.
- Updated `Coupler`, runtime runner/driver imports, boundary tests, and the
  architecture ownership docs for the new dispatch and component-execution
  boundaries.
- Validation run for this change:
  focused runtime/component fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Cohesion and Boundary Refactor Implementation

- Made component contract output merging pure, stored factory lifecycle hooks in
  a single private `LifecycleHooks` container, and moved
  component-facing runtime required-field validation into
  `vercor.components._runtime_validation`.
- Added runtime-owned contract refresh, bundled runner execution inputs in
  `RuntimeRunContext`, delegated final-output iteration to `vercor.output`, and
  kept `vercor.coupler.setup_logger` private to the facade implementation.
- Added `ExchangeSpec`, `build_exchanges()`, and `add_exchange_specs()` for
  setup recipes, then migrated examples and the profiling harness away from
  repeated raw `Exchange(...)` wiring.
- Lazied the paired JCM setup helper's optional JCM imports, moved JCM land
  type-only optional imports behind `TYPE_CHECKING`, and extracted focused
  JAXGCM host-recording and CAMulator prediction-block helpers.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, and boundary tests for the new
  ownership map.
- Validation run for this change:
  focused component/runtime/API fast pytest,
  focused setup/CAMulator/external fast pytest,
  focused runtime-cache fast pytest,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Component Boundary Import Refactor

- Moved component-facing required runtime field validation into
  `vercor.components._runtime_fields`, removing its hidden dependency on
  `vercor._runtime.validation` while preserving the existing `CouplerError`
  messages for missing and non-canonical fields.
- Converted annotation-only `Component` imports in the public coupler facade and
  setup/runtime helper modules to `TYPE_CHECKING` imports, keeping runtime
  imports focused on behavior dependencies.
- Added boundary tests that pin `_runtime_fields` away from runtime validation
  internals and guard the planned type-only import shape for coupler/runtime
  modules.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_api_boundaries.py tests/test_component_base_coverage.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Runtime Facade Cohesion Refactor

- Added `vercor._runtime.initialization` as the setup-time boundary for run
  precision synchronization, component initialization contexts, component setup
  validation, runtime contract validation, and exchange-topology handoff.
- Added explicit `ExchangeTopologyState` and `build_exchange_topology(...)` so
  exchange regridders and masks are assembled through a returned state object
  instead of only mutating caller-owned dictionaries.
- Slimmed `Coupler.initialize()` to delegate initialization wiring while
  preserving existing private runtime-state helpers and topology map aliases.
  Later cleanup removed those private compatibility attributes.
- Validation run for this change:
  `conda run -n scipy pytest tests/test_coupler_coverage.py tests/test_runtime_state.py -q --fast --tb=short`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`,
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Boundary Cohesion Refactor

- Moved component lifecycle hook type aliases and hook installation into
  `vercor.components._lifecycle`, leaving `vercor.components.factories` focused
  on public helper construction and breaking the top-level
  `vercor.components` import cycle.
- Switched base component runtime-field adapters to a direct private-module
  import instead of importing through the package namespace.
- Added shared setup-data helpers for positive binary masks and 2D/time-last
  surface-field canonicalization, then routed ERA5 ocean, ERA-Interim ocean,
  and JCM land preparation through those helpers.
- Updated boundary and setup-data tests to cover lifecycle ownership, component
  package import cycles, and shared field-helper behavior.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`, focused API-boundary and
  data-component kernel tests,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-27: Deprecated Compatibility Import Facade Removal

- Removed obsolete one-hop compatibility modules:
  `vercor._runtime.components`, `vercor.setups._data.camulator_land`,
  `vercor.setups._data.forcing`, `vercor.setups._external.camulator_state`,
  `vercor.setups._external.windpp`, and `vercor.setups.jax_array_helpers`.
- Routed remaining imports to canonical owners: runtime component-state,
  field-transfer, and validation helpers; `vercor.forcing_data.read_forcing`;
  calendar datetime classes; vertical-coordinate helpers; grid identity; and
  exchange field names.
- Removed compatibility reexports from `vercor.clock`, `vercor.exchange`,
  `vercor.grid_masks`, and `vercor.fluxes.utilities` while keeping stable
  package aggregators and settings attribute access. The remaining
  settings/forcing aliases were removed by the later obsolete compatibility API
  cleanup.
- Updated boundary tests, `DESIGN.md`, and `DEPENDENCIES.md` for canonical
  ownership. During full validation, corrected a stale Veros runtime-settings
  boundary assertion to point at `vercor.setups._external.veros_setup`.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  focused cleanup tests,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-26: Code Organization Audit Implementation

- Split physical defaults into `vercor.physical_constants` and composed them
  into `VercorSettings`, keeping static runtime controls in `settings.py`.
- Replaced concrete interpolator imports in `vercor.regridders.base` with a
  small scalar/vector interpolation protocol.
- Moved default topology component-name validation into
  `vercor._runtime.topology` so `Coupler` delegates topology policy.
- Split broad external adapters:
  `jax_gcm_fields.py` owns JCM field mapping and surface-temperature helpers,
  `camulator_fields.py`/`camulator_tensors.py`/`camulator_init.py`/
  `camulator_runtime_settings.py` own CAMulator field, tensor, init-noise, and
  environment setup helpers, and `veros_setup.py`/`veros_fluxes.py`/
  `veros_state.py` own Veros setup, flux, and host-state helpers.
- Kept adapter factory/state compatibility while removing moved helper symbols
  from old external adapter facades; narrowed
  `vercor.setups._data.camulator_land` to the public land factory only.
- Centralized common example exchange field lists in
  `vercor._exchange_recipes`, added slab land/ocean recipe separation,
  and widened `Exchange.field_names` to accept immutable recipe sequences.
- Updated `DEPENDENCIES.md` and ownership boundary tests for the new module map.
- Validation run for this change:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/test_api_boundaries.py -q --fast --tb=short`,
  focused external adapter tests, and
  `conda run -n scipy pytest tests/ -q --fast --tb=short`. The existing Black
  Python 3.13/target-3.14 warning remains. Full pytest was not rerun for this
  refactor.

### 2026-05-26: Ownership Boundary Refactor Follow-Up

- Moved model-calendar datetime values into `vercor.calendar`; later cleanup
  removed the `vercor.clock` reexports.
- Split canonical exchange-field vocabulary into `vercor.field_names`, unified
  grid identity in `vercor.grid_geometry`, and removed the runtime daily-index
  wrapper in favor of `vercor.calendar.daily_forcing_index`.
- Consolidated hybrid/sigma pressure and altitude helpers in
  `vercor.fluxes.vertical_coordinates`; later cleanup removed the old flux
  utility import aliases.
- Moved setup helper ownership to `vercor.host_arrays` and
  `vercor.diagnostics.fields`; moved CAMulator land, CAMulator output, CAMulator
  wind filtering, and JAXGCM output helpers under `vercor.setups._external`.
- Updated `DESIGN.md`, `DEPENDENCIES.md`, examples, and boundary tests for the
  new ownership map.
- Required validation passed:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-26: Refactoring Campaign Ownership Split

- Moved reusable setup adapters under the canonical `vercor.setups` package and
  runnable setup scripts under `examples`, removing in-repo reliance on a
  top-level `setups` package.
- Split public component factory helpers into `vercor.components.factories` and
  kept `vercor.components.base` focused on base authoring contracts.
- Routed setup adapter validation and runtime-boundary imports through private
  validation internals or public component context aliases instead of runtime
  stores/contexts.
- Added focused ownership modules for calendar logic, rectilinear grid geometry,
  generic sigma-coordinate helpers, generic PyTree transforms, setup data asset
  registries, and diagnostics fields/tables/plotting.
- Moved mask math into `vercor.grid_masks` and component topology lookup into
  `vercor._runtime.topology`.
- Split CAMulator optional imports, forcing cursors, tensor accessors, stepping,
  and initialization into focused modules before later cleanup removed the
  `camulator_state.py` facade.
- Focused checks passed for API boundaries, component factories, setup imports,
  runtime-boundary imports, shared helper ownership, assets/diagnostics
  separation, and CAMulator decomposition.
- Required validation passed:
  `conda run -n scipy black vercor examples tests`,
  `conda run -n scipy flake8 . --count --exit-zero --max-line-length=120 --statistics`,
  `conda run -n scipy mypy vercor examples tests`,
  `conda run -n scipy pytest tests/ -q --fast --tb=short`, and
  `conda run -n scipy pytest tests/ -q --tb=short`. The existing Black
  Python 3.13/target-3.14 warning and JAX dtype-promotion warning remain.

### 2026-05-15: Conservative Helper and Compatibility Cleanup

- Removed unused private helpers and wrappers that had become one-hop
  compatibility layers: bilinear `_geo_to_cart(...)`, component author
  `_author_field_spec(...)`, contract refresh indirection, runtime wrapper
  helpers, and Coupler topology delegates.
- Collapsed the setup forcing reader to the canonical
  `vercor.forcing_data.read_forcing` boundary while preserving the setup import
  path.
- Inlined the single-use private NetCDF output helper into the public output
  writer.
- Removed obsolete TODO/commented print blocks from conservative rectilinear
  regridder edge derivation.
- Deferred removal of then-intentional compatibility surfaces such as settings
  aliases, component author facades/context aliases, runtime reexports, setup
  lazy exports, setup forcing reexports, private runtime delegates, regridder
  class/factory APIs, and `Exchange.create()`. Later refactors removed the
  obsolete aliases and shims while preserving documented public facades.

### 2026-05-15: Runtime Helper Consolidation

- Added private `Coupler._prepare_runtime_state(...)` so `run()` and
  `_run_scanned_runtime()` share runtime-state creation/reuse and optional
  validation transition.
- Split runtime exchange dispatch into scalar and vector primitives while
  preserving existing behavior: scalar exchanges apply fractional masks, vector
  exchanges do not.
- Added `CamulatorRuntimeCursor` so CAMulator atmosphere and land adapters share
  forcing-index initialization, counter reset, index lookup, and counter
  advancement.
- Moved generic component step-result application into
  `vercor.components._runtime_fields.apply_step_result(...)` so callable
  wrappers and base components use the same primitive.
- Removed the test-only JCM coordinate wrapper and pointed tests at the real
  `_jcm_coordinates_in_degrees(...)` helper.

### 2026-05-15: Maintainability Follow-Up

- Added `setups.jcm_setup_helpers.build_jcm_land_atmosphere_components(...)`
  for repeated JCM setup construction, land-mask patching, and JAXGCM option
  forwarding.
- Refactored CAMulator `StateVariableAccessor` index-map construction through
  shared private primitives.
- Routed remaining multi-exchange runnable setup scripts through
  `setups.coupler_helpers.add_exchanges(...)`.
- Corrected `align_model_timestep(...)` non-divisible error text so it states
  that the model timestep must evenly divide the coupling timestep.
- Updated `DEPENDENCIES.md` for the JCM setup helper.
- Deferred intentionally high-risk audit findings: JAXGCM mirrored
  runtime/setup state, host/scanned runner unification, callable-wrapper
  architecture changes, and component inheritance changes.

## Milestone Timeline

### 2026-05-14: Setup and Adapter Maintainability

- Removed JAXGCM test-only compatibility attachments from factory-created
  components and moved private setup internals into explicit test fixtures.
- Consolidated NetCDF forcing reads behind `vercor.forcing_data.read_forcing`.
- Added explicit `Component.setup_metadata` for setup-only metadata.
- Extended setup lifecycle helpers for timestep assignment, spinup logging,
  forcing-index calculation, and default-field seeding.
- Added runnable setup helpers while keeping exchange recipes explicit.
- Consolidated common setup adapter paths for JAXGCM, Veros, CAMulator, ERA5,
  ERA-Interim, and JCM where behavior is shared.
- Added lazy optional setup imports so missing optional packages fail only when
  those adapters are used.
- Converted concrete setup components toward factory-based construction and
  reduced duplicate host-array and masked-field helpers.

### 2026-05-13: Logging

- Standardized VerCOR logging format and replaced root-logger capture
  expectations with the canonical logging boundary.

### 2026-05-12: Precision, Performance, and API Simplification

- Audited and propagated the centralized dtype policy.
- Optimized runtime profiling/core dispatch paths.
- Corrected hypsometric altitude calculations.
- Forwarded configured regridder factory options consistently.
- Removed redundant component APIs such as `required_fields` and callable field
  seeding.
- Added the shared PyTree mixin used by immutable JAX containers.

### 2026-05-08: Runtime Ownership and Component Boilerplate

- Moved runtime responsibilities into focused runtime modules.
- Extracted Coupler runtime adapter logic and component runtime-field adapters.
- Fixed time-dependent data field runtime validation.
- Tightened component constructor/runtime boilerplate and removed redundant
  authoring delegates.

### 2026-05-07: Component Authoring API

- Split component internals from the public authoring facade.
- Added and refined helper-first component authoring APIs.
- Polished component field declarations, context aliases, callable wrappers,
  default field seeding, and component author introspection.

### 2026-05-06: Settings and Lifecycle Logging

- Added Coupler lifecycle logging.
- Reworked settings into the unified metadata-backed `VercorSettings`
  container with dynamic attribute access.

### 2026-05-05: Runtime Interrupt Handling

- Added compiled runtime wakeup-fd interrupt handling.
- Unified host and scanned runtime interrupt translation.
- Added scanned runtime progress logging.
- Stabilized JAXGCM forcing payload scan shapes.

### 2026-05-04 to 2026-05-01: Data Layout and Data Components

- Centralized VerCOR dtype policy.
- Canonicalized component data dimension order.
- Added ERA5 atmosphere pure data component support.
- Made component author contracts explicit.

### 2026-04-30 to 2026-04-28: Runtime Package and Boundary Refactors

- Added callback-backed JAX runtime logging.
- Split the runtime package into explicit state, contract, context, driver,
  validation, and transfer boundaries.
- Clarified public/runtime API responsibilities.
- Removed residual compatibility markers and simplified runtime bridge
  ownership.
- Added compile-cache and safe buffer donation runtime support.
- Simplified runtime API validation around component-owned grid shapes.

### 2026-04-27 to 2026-04-23: JAX Translation and Unified Runtime Foundation

- Expanded Coupler, Veros, clock, and flux-kernel coverage.
- Translated flux, grid, bilinear, conservative remapping, slab, Veros, JAXGCM,
  CAMulator, ERA5, ERA-Interim, and data-forcing boundaries toward JAX-first
  runtime paths.
- Added differentiable public runtime and hardened mixed-grid/data-forcing
  runtime execution.
- Unified runtime component paths and removed legacy differentiable/wrapper-era
  APIs.
- Fixed wrapper runtime startup prefill and audited runtime tests against the
  canonical API.

## Known Failed Approaches / Corrections

- Do not fix numerical discrepancies with fudge factors. Earlier successful
  fixes traced missing terms, sign/index errors, dtype/layout issues, or
  boundary ownership mistakes instead.
- CAMulator cursor advancement must occur once after each non-empty forcing
  block, not once per model substep.
- Masked surface fields should use `jnp.where(mask, field, jnp.nan)` rather
  than multiply-by-NaN masking, which produced NaN gradients on valid cells.
- Source-boundary assertions should be precise. Several earlier failures came
  from over-broad substring checks that matched intentional helper names.
- Tests that patch host-backed external components should call
  `step_runtime_state()` with explicit `ComponentRuntimeState` objects when
  they are exercising runtime behavior.
- Setup-only metadata belongs in `Component.setup_metadata`, not ad-hoc
  attributes that enter public component/runtime contracts.
- Detailed failed attempts and command outputs are in
  `docs/progress-archive-2026-04-23-to-2026-05-15.md`.

## Validation Policy

- Keep this file compact. Record outcomes, current state, durable lessons, and
  next actions.
- Do not paste repeated per-task validation boilerplate into this file.
- For a normal development unit, record only:
  - the focused tests that mattered,
  - whether fast/full validation passed,
  - any new warnings, regressions, or failed approaches worth preserving.
- If a detailed transcript is needed, create or update a dated archive under
  `docs/` and link it from this file.
- Default development validation remains:

  ```bash
  conda run -n scipy pytest tests/ -q --fast --tb=short
  ```

- Before a commit or handoff, run the relevant focused tests plus the project
  static checks and full test suite when the change affects code behavior.
