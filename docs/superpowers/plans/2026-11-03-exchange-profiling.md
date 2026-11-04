# Exchange Remapping Profiling Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation;
> obtain one independent whole-change review before delivery.

**Goal:** Reduce measured exchange remapping overhead beyond output handling.

**Architecture:** Reuse dispatch-local coverage operands and compile the existing
sparse accumulation, preserving validation and all physical arithmetic.

**Tech Stack:** JAX, Python, XProf, Perfetto, TensorBoard, pytest, Black, flake8,
mypy, Git/GitHub. Existing isolated scipy312 tooling if scipy aborts.

**Spec:** `docs/superpowers/specs/2026-11-03-exchange-profiling-design.md`

## Global constraints

- Work on the explicitly requested current `feat/veris-seaice` branch.
- Preserve public interfaces, dtype, masking, sequential ownership and AD.
- Retain eager error ordering; no tolerance changes or numerical fudge factors.
- Full tests must pass before commit; push and verify the remote SHA.

## Review focus

- Updated source/target masks between dispatches; no retained-state cache.
- Received support belongs to dispatch-time state, including vector/binary routes.
- Fracarea denominator depends on current finite source samples.
- Equivalent remapper instances reuse compilation; changed geometry remains live.
- Eager infinities/overflow and checkify keep their existing behavior.

### Task 1: Profile and optimize the exchange remapping path

**Files:** `vercor/_runtime/exchange_dispatch.py`,
`vercor/_runtime/exchange_masks.py`,
`vercor/_interpolators/conservative_remap_rectilinear.py`,
`tests/test_dynamic_mask_exchanges.py`,
`tests/test_conservative_rectilinear_remapper.py`, `PROGRESS.md`,
`DEPENDENCIES.md`, and `docs/exchange-profiling*`.

**Interfaces:** `_dispatch_quantity` returns the diagnostic coverage derived from
its resolved operands. `exchange_coverage` uses the same private formula.
`ConservativeRectilinearRemapper._accumulate(field)` returns flattened weighted
sums and normalization areas using the existing PyTree.

- [x] Capture baseline cold/warm timings, resulting arrays and real traces.
- [x] Add a counted scalar/vector dispatch regression and a compiled-accumulation
  reuse regression; observe RED before implementation.
- [x] Add numerical/gradient/error tests for dynamic NaNs, weights and geometry.
- [x] Implement shared coverage derivation and private compiled accumulation.
- [x] Run focused tests; require GREEN with existing contracts unchanged.
- [x] Repeat isolated timings/traces and compare all arrays and compilation events.
- [x] Analyze with XProf/Perfetto and serve both profiles through TensorBoard.
- [x] Obtain fresh independent review; resolve material findings.
- [x] Run Black, strict flake8, mypy, fast/full stock and exact-fork tests.
- [x] Update evidence/progress/dependencies and prepare validated intended files.
- Delivery: commit, push and independently verify the remote SHA; the final
  external receipt and handoff record delivery without a self-referential SHA.

## Execution ledger

- Baseline `abc4494`; clean requested branch. GitHub connector confirms push access.
- Direct scipy Python3.14 orientation aborts workers during imports; use the existing
  documented scipy312 overlay, without modifying installed environments.
- Current-branch maintenance is explicitly authorized; execute inline, preserving
  all existing behavior rather than adding a design-approval pause.
- Independent performance audit identifies duplicate coverage remapping and
  uncompiled conservative accumulation; unrelated-route refresh is excluded.
- Raw scripts, traces, arrays and command logs: `/private/tmp/vercor-exchange-profile`.
- RED14 expected failures: twelve duplicate-coverage guards and two sparse-trace
  guards; two direct error-class controls already pass. Focused fast214 GREEN.
- Failed benchmark setup: uniform accepts center ranges and requires longitude/
  latitude arguments; corrected to explicit nondegenerate global cell bounds.
  Incomplete timings were discarded before the recorded baseline.
- Two float64 process pairs confirm host remapping/dispatch improvements. First
  pair scalar dispatch2.01-2.31x/vector1.40x; repeat2.01-2.31x/vector1.40x;
  conservative accumulation repeat2.10x/3.08x. Complete host intensive1.23x.
  Compiled/scanned controls are variable; no broad scanned speed claim.
- Each comparison saves285arrays;275byte-identical,10scanned leaves differ only
  at roundoff (float64maxabs1.78e-15;float32maxabs9.54e-7,maxrel1.71e-7), with
  shapes/dtypes/NaNs identical. Host and remapper controls are byte-identical.
- Independent review: no actionable Critical/Important/Minor findings. Root owns
  measured performance/equivalence evidence and final checks excluded by reviewer.
- XProf converts12captures; Perfetto reports fewer launches and0warm compiles;
  TensorBoard enumerates12final profiles and serves both dispatch overviews/stats.
- First stock suite2726passed/5failed: bounded-progress181lines, one RST title
  underline, and three native checks needing a writable Veros asset-lock path.
  Preserved log text by joining one continuation; fixed underline and set existing
  VEROS_ASSET_DIR=/private/tmp/vercor-veros170-assets. Production/tests unchanged.
- Final stock2731/fast1774 pass with zero failures/errors/skips. NativeAD17passed
  initially plus the one corrected global-adjoint case1passed accounts for all18.
  Collection/JUnit union matches2749cases without overlap; combined branch
  coverage92.60147285494091%, unchanged90%floor. Black/mypy331, repositorylint0,
  compileall/strictSphinx/diff pass; all331source/test hashes remain unchanged.
- No production change was needed for final gate failures. Asset errors were
  resolved with the documented writable cache; raw failing and corrected logs
  remain in the evidence. No existing test or tolerance was removed or relaxed.
