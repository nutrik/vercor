# Backend-independent output sampling

**Goal:** Use one provider-selection and accumulation pipeline across JAX,
forced-host, mixed and custom runtime backends.

**Design:** Output configuration, cadence and writes already have one owner.
Keep that owner and the existing public contracts. Prefer a provider's explicit
`sample_jax` capability on every backend, independently per provider; ordinary
providers receive concrete `OutputContext` values. Share frame validation,
variable selection, finite sum/count reduction and schema checks in one loop.
Only a JAX runtime with entirely numeric providers needs abstract schema
preparation and scan-shaped coordinate leaves. Host components are never traced
to prepare output. Preserve NumPy host-coordinate precision.

**Scope:** This bounded refactor changes the existing output session, provider
documentation and regression tests. It adds no backend, public protocol,
dependency, buffering, callback or execution path. Execution grouping remains
an optimization of the JAX backend; all paths sample each completed coupling
step and write only complete periods. The user's instruction authorizes
assessment, implementation and delivery without another approval round.

**Review focus:** numeric providers alongside ordinary providers; opaque host
payloads; concrete timestamps and canonical int32 numeric indices; independent
period resets and incomplete tails; errors and repeated-run isolation.

## Tasks

- [x] Add integration regressions for `jax`, `host`, mixed `auto` and a driver-
  based custom backend, with and without an ordinary neighbor provider.
  Check hand-derived finite-count means, coordinates/dtypes, time metadata,
  incomplete periods, repeated runs and component-scoped failures. Run RED.
- [x] Consolidate `_OutputSession.accumulate` and `accumulate_jax` through one
  private loop. Select numeric sampling per provider, normalize numeric indices
  to the runtime int32 policy and retain ordinary contexts and host coordinates.
  Run new regressions and existing output/workflow suites to GREEN.
- [x] Add opaque host-payload and disabled-output coverage; document provider
  selection independently of execution backend in DESIGN and output guidance.
- [x] Run Black, maintained/repository flake8, mypy, fast/full stock tests and
  isolated Veros AD tests; review the change, resolve findings and record
  evidence in PROGRESS.

Delivery is authorized: commit intended files, push with gh-authenticated Git
and verify the GitHub branch SHA matches HEAD. The final Git history and
independent remote-ref verification are the authoritative delivery record.

## Execution record

- Existing branch and clean worktree verified at `6ac048b`.
- Initial scipy Python3.14 fast suite reproduces the documented OpenMP crash.
  Focused output tests pass17 there. Full validation will reuse the documented,
  previously approved scipy312/native-overlay lane, changing neither environment.
- New regression RED is recorded in `/private/tmp/vercor-unified-red.log`.
  Detailed remaining evidence uses the `/private/tmp/vercor-unified-*` prefix.
- Corrected test fixture: repeated runs use separate directories; the existing
  writer intentionally replaces a matching filename on a later run. Corrected
  RED is 16 failures from ordinary-method dispatch plus one all-JAX baseline pass.
- Shared sampling GREEN38; independent review found no blocking implementation
  issue and suggested checking numeric-provider host coordinates. Extended the
  existing int64/float64/x64-disabled regression to forced-host and custom
  backends; final focused42 pass. Host opaque payloads and disabled targets pass.
- Black305, maintained flake8 zero, mypy305, compileall/diff and strict Sphinx
  pass. Repository lint retains exactly the known200 historical snippet findings.
  A test annotation and RST underline were corrected during these checks.
- Initial full run exposed the existing180-line PROGRESS bound after adding the
  task entry. Exact test RED reproduces181>180; folding two existing constraint
  bullets preserves their meaning and passes GREEN at180. The gate is unchanged.
- Execution remains on the user's requested branch. Existing task authorization
  covers implementation and gh-authenticated push; no worktree or approval menu
  is needed. No environment installation or dependency modification was made.
- Final acceptance: stock2426, fast1476 and exact-forkAD15 pass with zero
  failures/errors/skips. JUnit/log evidence: `/private/tmp/vercor-unified-full-final`,
  `-fast` and `-ad` (`.xml`/`.log`). Final stock and AD raw coverage are separate;
  copy-only combination preserves their hashes and passes92.81572748686357%
  against the unchanged90% threshold. Verification script/report:
  `/private/tmp/vercor-unified-verify.py`, `vercor-unified-verification.json`,
  `vercor-unified-coverage.{log,json}` and `vercor-unified-coverage-input/`.
- New coverage includes25 cases: the new provider/backend tests21 plus four
  numeric host/custom coordinate cases. Reviewed final production source is
  identical to the source used by final stock and native AD gates. Output-free
  runtime and public signatures are unchanged; the optional numeric capability
  now determines extraction on every backend. No outstanding review findings.
