# Exchange and Output Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Share exchange/output mean arithmetic without sharing field selection or clocks.

**Architecture:** A private numerical foundation supplies finite sums/counts, safe means, and a frozen scalar-count accumulator for mandatory equal-duration native samples. Output keeps schema/selection; adapters keep sampling/physics; JCM folds mapped samples with a scan.

**Tech Stack:** Python, JAX, pytest, Black, flake8, mypy, Git/GitHub; isolated scipy312 native stack.

**Spec:** `docs/superpowers/specs/2026-11-03-exchange-output-reduction-design.md`

## Global Constraints

- Use existing `feat/veris-seaice`; user explicitly requests this branch and push.
- Preserve native endpoints, nonlinear diagnosis ordering, masks and invalid-sample rejection.
- Output selection never changes exchange field selection, cadence, or physics.
- Keep equal-step native counts scalar and output missing-data counts per cell.
- No native dependency edits, new public API, or JCM native stepping replacement.
- Full and fast suites must pass before the commit; verify pushed remote SHA.

## Review Focus

- Native invalid samples must propagate even if a later sample is finite.
- Entirely missing output cells must return NaN with finite inactive gradients.
- JCM mapped reduction must retain every native sample without a mapped trajectory.
- Different output subsets and disabled output must produce identical exchanges.
- Native state/count storage must remain bounded and reset per coupling interval.

### Task 1: Shared numerical reduction and output reuse

**Files:** create `vercor/_averaging.py`, `tests/test_averaging.py`; modify `vercor/output/_period.py`, `_session.py`.

**Interfaces:** `finite_sum_and_count(values, *, axis=None)`; `mean_from_sum_and_count(total, count)`; frozen PyTree `SampleMean(total, count)` with `from_sample`, `add`, and `mean`. Native tree samples are already mapped arrays; count is scalar int32.

- [x] Write RED tests for finite/missing reductions, broadcast/scalar denominators, zero counts, dtype, strict native nonfinites, bounded storage and JIT/AD.
- [x] Implement minimal foundation and delegate output numerical operations to it.
- [x] Run foundation and period/scanned/selective output tests GREEN.

### Task 2: Native exchange reuse and JCM streaming mapping

**Files:** `vercor/setups/_external/{veros_state,jax_gcm_runtime,veris_runtime}.py`; exchange/temporal regression tests.

**Interfaces:** Task 1's scalar-count accumulator in Veros/JCM; shared mean division for Veris's existing weighted integrals. Add private `_mean_jcm_sample_fields` to fold the existing compact history with `lax.scan`, returning no scan trajectory.

- [x] Write regression tests before source changes for native/output independence and finite behavior; retain existing independent nonlinear/endpoint/continuation/AD oracles.
- [x] Replace duplicate native reductions; preserve invalid native sample propagation.
- [x] Run focused exchange/backend/native tests GREEN; inspect bounded scan carry.

### Task 3: Documentation, independent review and delivery

**Files:** `DESIGN.md`, `DEPENDENCIES.md`, `docs/api-architecture-review.md`, `PROGRESS.md`, this plan.

- [x] Document numerical ownership, distinct cadences, strict native versus missing diagnostic samples, and the retained JCM native-history limitation.
- [x] Obtain one independent whole-change code review and resolve substantive findings.
- [x] Run Black, strict repository flake8, mypy, compileall, whitespace, fast/full stock and exact-fork AD tests; retain concise logs outside the checkout.
- [x] Update progress/evidence for the settled, validated source.
- Delivery: commit and push this validated change on the existing branch; the final handoff and external receipt record the verified remote SHA.

## Evidence and decisions

- Baseline source `38810f1`; clean working tree on requested branch. GitHub connector confirms remote branch exists.
- `scipy` Python3.14 fast baseline aborted four native workers before tests. Reuse documented isolated stock stack with scipy312; installed environments remain unchanged.
- User approved the design via the asynchronous question; no further approval gate is needed.
- Execution is inline with a final independent reviewer; full-suite-before-commit policy takes precedence over incremental plan commits.

- RED7/GREEN86 shared-kernel/output tests; RED1 JCM fold then focused93 GREEN. Four runtime lanes preserve exchange values for disabled/all/subset period output. Existing nonlinear, singleton, native continuation, masks and AD oracles pass in focused checks.
- Independent review found no actionable findings; focused reviewer tests passed. Post-review buffer optimization changes scan inputs to scalar indices so native histories remain invariant, with a static singleton bypass; final numerical/scan tests26 pass.
- Compiled CPU/f64 synthetic mapper-only benchmark (64x32, eight sigma levels): at48 samples fused-vmap temporary storage11,010,048bytes versus66,116bytes for streaming, max normalized field difference7.93e-16. Warm medians0.557ms versus1.175ms; singleton both0.089ms. This is a memory/speed tradeoff for the isolated compiled reduction, not an end-to-end model/GPU performance claim. Compact native histories remain. Evidence: /private/tmp/vercor-exchange-benchmark.json and its reproducer script.


- Final gates: full stock2716, exact-forkAD18, fast1755, additional native exact-JAX temporal14 all pass with zero failures/errors/skips. Exact collected-case union2734/2734; all330 source/test SHA256 values unchanged. Combined branch coverage92.58050665521684%; raw stock/AD coverage files/hashes preserved. Black330/mypy330/repositoryflake8zero/compileall/strictSphinx/diff checks pass. Machine receipt: /private/tmp/vercor-exchange-final-results.json.
- Final repeated CPU benchmark confirms temporary-memory figures. At48samples the prior host vmap-plus-per-field-mean call takes1.710ms versus1.244ms for cached streaming; fused compiled reference0.495ms versus1.172ms. Singleton host0.443ms versus0.092ms. These are synthetic mapper/reducer measurements only, not full-model/GPU certification.
