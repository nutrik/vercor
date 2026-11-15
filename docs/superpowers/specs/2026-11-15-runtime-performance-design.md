# Host and JAX runtime performance design

The approved scope is a broad CPU audit of runtime orchestration, numerical
validation, dynamic exchange masks, remapping and output. Preserve physical
arithmetic, precision policy, JVP/VJP, eager errors, lifecycle order and the
existing exact scan scheduler. Work on `perf/host-jax-runtime`; delivery is local
commits, with no inferred push or pull request.

## Approach

Extend the existing profiler rather than introduce a profiling framework. Add
host/API and JAX/API measurements alongside the existing outer-jitted control.
Output-enabled measurements use the API path. Reject an explicit outer JIT with
host execution or enabled output. Time setup, first invocation and synchronized
warm calls separately; capture XSpace/Perfetto outside timed runs and inspect
them with XProf, Perfetto and TensorBoard.

Implement three bounded candidates: constant-time expected-plan identity checks,
private compiled invalid-count arithmetic, and lazy component-mask reuse within
one refresh. Binary mask policy is static. Shape validation and error reporting
remain outside compiled count kernels. Reuse only raw component masks; retain
route remapping, route topology, first-reference order and received support.

Alternatives rejected: removing full-state validation changes author hooks;
persistent caches introduce invalidation/ownership policy; changing scan widths
changes the established compilation and reverse-mode memory contract. These
require separate evidence and are outside this maintenance task.

## Evidence and acceptance

The baseline is `3a48d7fb79d036508b456c9fe4310e6dea74e036`. Compare paired,
uninstrumented processes in alternating order, three repetitions with nine warm
samples each, at float32/float64. Include uniform/irregular schedules, shared
dynamic-mask endpoints, scalar/vector remapping and output off/on. Save output
leaves, revisions, versions and timing distributions. Use fresh output paths.
Native ocean/coupled checks establish integration, not accelerator throughput.
Keep a candidate only if its target median improves by at least 5% in all three
pairs and representative controls have no repeatable regression above 5%.
Report unsupported CPU trace categories as unavailable, never as zero.

Tests pin mechanism counts rather than elapsed time. Cover first-error ordering,
active/inactive NaNs and infinities, mask shapes, both precisions, nested
JIT/checkify, scan/vmap failures, JVP/VJP, compile reuse, route order and custom
backend identity/completeness. Compare all final leaves and output artifacts at
existing tolerances, requiring exact equality where arithmetic is unchanged.

The user approved `scipy312` instead of the incompatible installed `scipy`
runtime. Keep optional profiler packages and the pinned Veros AD fork isolated
under `/private/tmp`, with stock and AD in separate processes. Do not modify
installed packages. Full stock/AD suites, fast tests, combined branch coverage
at least 90%, Black, strict Flake8, mypy, compileall, strict Sphinx, API and
distribution contracts must pass before committing. Update PROGRESS and
DEPENDENCIES and retain reproducible evidence outside the checkout.

## Execution decisions from measured controls

The mask-reuse trial passed all value/AD checks and accelerated eager refresh,
but its isolated compiled refresh control regressed. It is excluded from the
delivered source, retaining additional shared/self-route audit tests. Initial
non-inline count kernels also slowed API JAX output. Explicit JAX inlining
restored that control while retaining fused eager arithmetic; final acceptance
uses a fresh full matrix of the retained count/plan changes.
