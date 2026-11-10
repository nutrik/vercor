# Released Veros JAX backend and execution comparison

The user selected the focused approach and approved this design on 2026-11-03,
instructing implementation to proceed. Work starts on `feat/veris-seaice` at
`578fd82`.

## Intent and scope

Enable released Veros 1.7.0 JAX kernels through VerCOR, measure NumPy versus
JAX throughput, compare coupled host and scanned JAX execution, and make JAX
the native default only when controlled measurements justify it. Preserve
native physics, existing differentiable-fork support, lazy optional imports,
runtime state ownership, dtype policy, output, and temporal exchange semantics.
Finish with formatting, lint, types, focused/fast/full tests, Git commits on
the current branch, and a GitHub CLI authenticated push with remote SHA checks.

## Approaches

The user chose to separate the native Veros numerical backend from VerCOR
component execution. Stock 1.7.0 can then use NumPy or JAX kernels with host
orchestration. The pinned differentiable fork remains the fully scanned path.
Extending stock Veros to full scanned execution is a larger, separate option:
its native driver contains an array-dependent Python sanity check and it lacks
the capabilities required by the current differentiable adapter. Stock already
has PyTree support; it must not be described as lacking it. Replacing stock
Veros with the fork would not establish release compatibility or a controlled
NumPy/JAX comparison.

## Configuration and implementation boundaries

- Add one keyword-only `VerosConfig.backend` choice for the native backend,
  preserving positional constructor compatibility. Resolve the host default
  from benchmark evidence; `execution="jax"` always requires native JAX and
  the existing fork capabilities. Reject contradictory requests early.
- Keep `execution` responsible for component dispatch, setup selection,
  functional native state copying, and outer JIT. Keep native `backend`
  responsible for Veros operators, forcing-array placement, and compatible
  linear-solver selection. Configure before importing native implementation
  modules; preserve actionable process-wide runtime-lock errors.
- Remove the assumption that any JAX backend supplies fork-only `safe_sqrt`.
  Use the fork operation when available and native stock forward arithmetic
  otherwise; preserve the fork's validated calm-stress derivative behavior.
- Inspect and test actual host/JAX payload ownership before changing it.
  Stock PyTree auxiliary metadata includes mutable diagnostic/timer owners;
  any demonstrated repair stays in the native adapter. Avoid a generic
  runtime redesign and avoid needless device-to-host forcing transfers.
- Retain native setup and solver behavior wherever possible. Record solver
  classes/tolerances and TDMA policy in benchmarks. Do not change numerical
  tolerances or physical settings to improve a timing or numerical comparison.
- Update the optional dependency extra if released JAX use needs additional
  declared requirements, and update maintained API/docs contracts explicitly.

## Benchmark design

All backend combinations run in separate fresh processes. Rebuild isolated
Python 3.12 stock/exact-fork targets under `/private/tmp`; the old temporary
directories contain no package files. Verify imported module paths, versions,
fork commit provenance, JAX/JAXlib, NumPy/SciPy, JCM/Veris versions, dtype,
device, CPU/thread configuration and source revision. Preserve installed
environments. The normal Python 3.14 `scipy` orientation aborted importing an optional native dependency.

Use the real native ACC setup for an asset-independent ocean comparison and
the actual global four-degree Veris/Veros/JCM gallery for the coupled result.
Reduced deterministic native cases may validate contracts but do not replace
the real coupled throughput measurement. Use identical physical setup,
timesteps, initial conditions, forcing, coupling schedule and output policy
within each controlled comparison. Disable spinup for timing, keep output
off for the primary compute comparison, and validate an explicit output smoke.
Report exact dimensions and native steps per coupled interval.

The measurement matrix is:

| Model distribution | Native backend | VerCOR execution | Purpose |
| --- | --- | --- | --- |
| Stock 1.7.0 | NumPy | Host | Release baseline |
| Stock 1.7.0 | JAX | Host | Controlled native-backend benefit |
| Exact pinned fork | JAX | Forced host | Same-version execution baseline |
| Exact pinned fork | JAX | Scanned JAX | Controlled orchestration benefit |

Report setup time, first synchronized invocation including compilation,
individual warm samples, median/spread, native steps/s and simulated days/s.
Warm up every measured path, synchronize all asynchronous leaves and effects,
and time without simultaneous tests or competing benchmark processes. Reset
from the same owned initial state for each replicate; verify it stays unchanged.
Collect at least five warm replicates and repeat the stock comparison in a
second fresh process per backend. Measure sufficiently long blocks to make
dispatch noise small. Record break-even length including startup cost.
State clearly that stock-versus-fork timings are not a same-physics comparison:
the fork predates correctness changes in 1.7.0. Do not attribute that difference
solely to execution.

## Correctness and default decision

Compare full native prognostic fields and canonical exchanged fields for the
stock backends with scale-aware numerical error summaries, masks, finite
values and conserved/source-budget checks appropriate to the short run. Compare
fork host/scanned results under the same fork and setup, retaining existing
JVP/VJP/finite-difference acceptance. Reuse meaningful existing assertions;
do not manufacture broad tolerances from measured disagreement.

Adopt native JAX as the stock host default if the repeated real coupled
benchmark shows a consistent warm throughput benefit above measured timing
uncertainty and the correctness/continuation/output checks pass. Quantify the
uncertainty and report startup break-even so the decision is reviewable. Keep
explicit NumPy selection available. If only ACC improves or no repeatable
coupled benefit is established, retain the current default and document the
workload boundary. A higher compilation cost alone does not disqualify a
demonstrated steady-state benefit; document its consequence for short runs.

## Validation and deliverables

Write regressions before production changes, covering configuration validation,
lazy import order, backend locks, stock global TKE forcing, array placement,
payload independence, native progression, coupled exchanges and output. Run
Black, strict repository flake8, mypy, compileall, whitespace checks, focused
regressions, fast/full stock suites and the separate exact-fork AD lane. Use
concise logs outside the checkout. The current unmodified Python 3.12 baseline
passes Black and flake8; mypy reports two JAX config-attribute errors in the
older installed JAX environment, and the baseline fast test process exited with 13 failed nodes and 32 error
nodes, dominated by mixed backend locks and incomplete overlay metadata.
Those are environment/baseline evidence, not passing product acceptance.

Commit a reproducible benchmark command/harness, machine-readable timing and
numerical summaries, and a concise interpretation with limitations. Document
the API/default decision in existing relevant pages and keep `PROGRESS.md`
and dependency ownership current. Obtain an independent whole-diff review,
fix material findings, then commit intended files and push the requested branch
using GitHub CLI authentication. No merge, tag or release is requested.
