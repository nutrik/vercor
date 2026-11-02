# JAX compilation stability audit

Date: 2026-10-22. Branch: `feat/veris-seaice`.

The requirement is stable compilation during a run with fixed array schemas,
including JAX, forced-host, mixed auto and driver-based custom execution.
Initial compilation of each schedule/kernel is necessary. This audit covers
VerCOR-owned execution, interpolation, native adapters and output. Arbitrary
plugin code can create new JIT functions or change its signatures; the Python
host/custom contracts deliberately continue to permit such code. Separate
`Coupler.run` calls retain their existing run-local executor lifetime.

## Reproduced causes and corrections

| Runtime path | Reproduction | Correction |
| --- | --- | --- |
| Output-free JAX | Repeated schedules of lengths 2/1/3/2/1 trace five executors for two schedules. | Reuse one fixed-width executor per schedule and gather dynamic absolute indices for exact common blocks. Two schedule traces. |
| Scanned output | January, February and an incomplete March tail trace three variants of one schedule (31/28/11 steps). | The same fixed-width inputs retain one executable while preserving exact post-step means and incomplete-period behavior. |
| Numeric coordinates | Weak Python/JAX scalar coordinates change the carry signature after abstract zero allocation, tracing twice across equal periods. | Explicit coordinate dtype inside scans matches the initial strong zero carry. Host coordinates keep their original precision. |
| Bilinear host remapping | Local scalar/vector conditional branches trace 3/6 times across three calls. | Stable JIT finishing kernels trace 1/2 branch bodies. Source validation remains outside that boundary to preserve eager, checkify and AD error behavior. |
| Native face-coordinate output | Missing explicit grid edges call the longitude helper on every host sample, recreating its two conditional branches. | A stable shared geometry JIT traces the conditionals once for fixed coordinate shape/dtype and static grid kind. |
| Native Veris on host/mixed/custom | Three coupling intervals recreate and compile the native substep loop three times. | Setup retains one compiled pure transition with dynamic forcing, payload, physical constants and coupling duration. Six native substeps still execute, with one native trace. |

The scan uses the greatest common divisor of each schedule's chunk lengths.
Larger core chunks use multiple internal blocks; output and validation
boundaries remain unchanged. Every slot executes an actual step, preserving
checkpointing and the original reverse-mode path without padding or a new
conditional around RunState. Uniform workflows still scan their whole window;
coprime calendar/schedule lengths need single-step blocks, trading some host
dispatch overhead for a fixed compilation signature. The existing driver still
owns ordering and state/result checks; host authors retain concrete calendar
times and Python step indices.

Directly JIT-decorating the entire bilinear methods changed exception types
under checkify and VJP. Focused tests caught that regression; retaining source
checks at the existing boundary and compiling only the finishing kernels
preserves the original error contract.

## Regression coverage and evidence

`tests/test_runtime_compilation.py` measures actual compiled-body traces rather
than wrapper construction or scan invocation counts. It covers irregular
schedules, unequal months, tails, bilinear scalar/vector kernels and exact-block
outer-JIT/JVP/reverse derivatives. `tests/test_output_compilation.py` checks
Python, weak and strong scalar coordinates, literal means and latest positions.
`tests/test_veris_compilation.py` exercises real native stepping in all four
backend modes and changes physical constants/duration without retracing.
Existing output/validation/gradient suites retain their numerical assertions;
the daily tail regression now uses five exact two-step blocks instead of
widths 4/4/2.

Local command evidence uses `/private/tmp/vercor-recompile-*` plus
`vercor-output-compile-*` and `vercor-host-compilation-*`. The `scipy` Python3.14
baseline reproduces the existing duplicate-OpenMP abort. Checks use the
existing Python3.12 `scipy312` interpreter and a temporary model-only overlay
with RECORD-verified Veros1.7.0, JCM3.0.0 and Veris0.2.1 packages. NumPy2.4.2,
SciPy1.17.1, JAX/JAXlib 0.11.2, Dinosaur1.5.1 and Flax0.12.10 come from scipy312.
No installed environment or dependency declarations are changed. The prior
JCM upper-bound exception remains unchanged. Native Veros autodiff validation
uses the pinned fork in a separate process.

Independent review reproduced a 9.8-fold gradient temporary-memory increase
with maximum-width padding for one long prefix followed by singleton windows.
Whole-executor checkpointing did not improve it. Mean-width bounded padding
reduced that cost, but the full native adjoint gate then found reverse NaNs
from the new conditional around runtime state (finite forward/JVP). Removing
only bilinear JIT boundaries still failed; restoring the original unconditional
scan body passed. Exact common blocks remove both rejected padding approaches.
The skewed-schedule regression checks that scan slots equal actual steps;
existing native adjoint assertions and tolerances remain unchanged.

The final call-site review additionally reproduced repeated longitude geometry
conditionals in native face-coordinate output. The helper now retains a JIT
boundary with static grid kind and dynamic coordinate values. Its repeated-call
regression first exposed an incorrect expected western edge (corrected to
-45 degrees), then reproduced six additional branch traces after warmup.

Two early geometry checks named nonexistent test modules; the actual helper
coverage is in `test_helpers_coverage.py` and the conservative remapper/regridder
suites. The corrected geometry/output focus passed 37 tests.

Final verification: stock **2,441**, fast **1,491** and isolated exact-fork native
AD **15** passed with zero failures, errors or skips. Copy-only combined branch
coverage is **92.82685193539655%**, above the unchanged 90% threshold; both raw
lane hashes are preserved. All 308 source/test module hashes match the final
verification receipt. Black, mypy, maintained-source flake8, compileall,
whitespace, strict Sphinx and wheel/sdist builds pass. Whole-repository lint
retains the existing 200 historical snippet findings; upstream JAX scatter-cast
and JCM/xarray future warnings remain visible. No review findings remain.

The final receipt is `/private/tmp/vercor-recompile-verification.json`, with
JUnit, logs, raw coverage and source hashes under that prefix. `PROGRESS.md`
and Git history record completion and delivery.

## Measured validation overhead (2026-11-02)

Perfetto traces identified successful scalar validation callbacks as a cost in
output-free CPU runs. `_require_zero` and `_require_true` now use a device-side
conditional to call Python only for a failed scalar compiled check. The
deliberately unjitted bilinear scalar source-entry guard retains its direct
callback so eager `checkify` still raises `CouplerError`; its finishing and
extrapolation kernels keep the conditional policy. Numeric requirements and
error messages are unchanged. The host interruption checkpoints remain active. Under `vmap`, JAX can transform
a conditional into a select and execute callbacks for valid batch items too;
this change preserves batched safety but claims no callback avoidance there.
Regression tests cover successful callback avoidance, invalid values in later
scan slots, batched checks, inactive NaNs and existing differentiation paths.

The existing profiling gallery now separates synchronized setup, a cold compiled
invocation, and repeated warm invocations from the same initial state:

```bash
python -m vercor.setups.gallery.profile_runtime --steps 24 --grid-nx 32 \
  --grid-ny 16 --float-type float64 --repeats 9 \
  --trace-dir /private/tmp/vercor-profile --json
```

Warm samples reuse an outer `jax.jit(coupler.run)` of the output-free scan;
they measure its compiled execution rather than rebuilding Python orchestration
or installing signal handlers on each invocation. The cold measurement includes
compilation and one execution. The trace captures one extra synchronized warm
invocation, excluded from timing samples. This follows the
[JAX profiling guidance](https://docs.jax.dev/en/latest/profiling.html); the
Perfetto-compatible trace can be loaded locally in the Perfetto UI or queried
with its [Trace Processor](https://perfetto.dev/docs/analysis/trace-processor).

On an Apple M4 Pro (14 logical cores, 24 GiB RAM), macOS 26.7.2, Python 3.12.15,
JAX/JAXlib 0.11.2, and the identical
24-step 16 × 32 slab problem, nine untraced warm samples gave:

| Precision | Before median (range), ms | After median (range), ms | Ratio |
| --- | ---: | ---: | ---: |
| float64 | 83.242 (82.435–85.120) | 6.464 (6.157–6.726) | 12.88 |
| float32 | 90.151 (86.173–94.432) | 6.852 (6.682–7.312) | 13.16 |

All final-state leaves are bitwise identical, including NaN locations. Perfetto
v59.1 SQL counted 2,506 debug-callback slices before and 240 after, in each
precision. The remaining 240 are the existing interruption checkpoints. Cold
slab cost increased from 0.305 to 0.332 s in float64 and 0.316 to 0.330 s in float32;
this optimization targets repeated execution, not initial compilation. Trace
durations include profiler overhead and are not used as benchmark timings.

Raw timings, final arrays, traces and Perfetto SQL output are under
`/private/tmp/vercor-performance-before-{f32,f64}/` and
`/private/tmp/vercor-performance-final-after-{f32,f64}/`; the same-input
comparison script is `/private/tmp/vercor-performance-compare.py`. The starting
revision is `7c774d487cd23811b3b94e4588c5cabbed2b5433`. These small CPU results do
not establish throughput on accelerators or a century climate run. Peak memory
has not been measured.

### Native coupled comparison

The same starting-revision fixture supplies a 4 × 4, two-level ocean and sea ice,
a T21 JCM atmosphere, two 1,800-second coupling steps, and no output. Stock
Veros 1.7.0 uses the public host invocation; the pinned exact fork uses an outer
JIT for compiled all-JAX execution. Five untraced warm samples per code/lane:

| Lane | Before median (range), ms | After median (range), ms | Interpretation |
| --- | ---: | ---: | --- |
| stock host | 474.388 (468.328–479.708) | 483.089 (482.517–485.981) | no material gain established |
| exact-fork all-JAX | 107.776 (106.581–108.356) | 19.982 (19.498–20.502) | 5.39 times faster |

All 68 canonical fields are bitwise identical before/after in both lanes.
Perfetto SQL counted 2,411 → 55 debug callbacks in all-JAX and 152 → 0 in host;
eager host validation and native solver work still run. The 55 all-JAX callbacks
include 20 interruption checkpoints and 35 preserved interpolation guards. All-JAX cold cost rose
from 7.500 to 9.821 s; host cold cost was 4.472 versus 4.538 s. These are genuine
native model steps on a miniature configuration, not a global climate benchmark.

Timings, fields, traces and SQL output use the exact directories
`/private/tmp/vercor-performance-native-before-jax`,
`/private/tmp/vercor-performance-final-native-after-jax`,
`/private/tmp/vercor-performance-native-before-host` and
`/private/tmp/vercor-performance-final-native-after-host`. The reproducer is
`/private/tmp/vercor-performance-native-compare.py`; its fixture imports are
frozen to the starting revision while product imports select the before/after
tree. An earlier after-host sample overlapped focused tests and was discarded;
the final settled-code repeat above is used. Earlier measurements before the
checkify boundary repair are superseded by
`/private/tmp/vercor-performance-final-comparisons.json`. An approved read-only hardware query identified Apple M4 Pro, 14 logical cores
and 24 GiB RAM; JAX reports the CPU device as `cpu:0`.

## Explicit Veros float32 and test consolidation (2026-11-02)

`tests/test_veros_float32.py` starts isolated processes before native backend or
precision settings lock. The stock and pinned exact-fork native drivers advance
a 4 × 4 two-level ocean with independently mapped surfaces. Tests cover singleton
and three-step means, two-interval reset/continuation, wet/dry cells, nonzero
traction/SSH signals, physical heat/salt source balances, actual floating and
categorical dtypes, and native JIT/JVP/VJP plus two finite-difference stencils.
The fixture now types its depth and forcing arrays explicitly. The regression
first failed because the exact fork retained a float64 depth array.

For full native float32, select `VEROS_FLOAT_TYPE=float32` and
`JAX_ENABLE_X64=false` before starting Python. VerCOR precision controls its own
arrays; native Veros precision is a process setting. A separate allocation-only
probe retains x64 capability and catches fixture promotion. The pinned native AD
fork can promote a diffusivity under x64 capability even when native precision
is float32 (`veros/core/tke.py:17`; its state setter omits casts at
`veros/state.py:236–245`). Full-driver mixed-capability support and native-package
patches are outside this change. Existing float64 tests remain.

The consolidation removes 33 collected functional/API cases while retaining
independent contract expectations:

| Removed repetitions | Retained assertion owner |
| --- | --- |
| 29 public-module manifest cases | complete documented/live manifest check, now also checked against the original hardcoded exports |
| bilinear constructor forwarding | public factory test with the same masks and nondefault options |
| constructor signature subset | complete exact signature snapshot |
| resolved component step annotations | complete exact signature snapshot |
| separate ValidationContext resolution | adjacent public type test, with the identity assertion migrated |

All 158 remaining cases in the affected five files pass. Five in-memory mutation
checks reject coordinated documentation/source export drift, constructor kind,
step annotation, validation-state annotation and bilinear forwarding defects.
Distinct physical, numerical, autodiff, backend and error tests were retained.
The full mapping is `/private/tmp/vercor-test-dedup-final.md`.

All 200 historical flake8 findings were in four ignored generated evidence
snippets under `test_logs`; all 327 tracked Python files already passed the
120-column check. `.flake8` now explicitly excludes that generated directory.
No lint codes are suppressed and archived evidence is preserved. Repository-wide
strict lint reports zero findings.

### Final integrated verification

The settled source passes 2,637 stock tests, 1,677 fast tests and 17 isolated
exact-fork AD tests, with zero failures, errors or skips. Dedicated precision
validation additionally passed 39 stock cases and 16 exact-fork/float64 temporal
cases. Combined coverage is 92.949002% (94.864310% statements; 84.205331%
branches), above the unchanged 90% combined floor and slightly above the prior
92.945248% baseline. Missing line/branch counts remain 475/320 while covered totals increase
from 8,770/1,704 to 8,774/1,706. The removed assertions retain their independent
owners. Both raw coverage lanes retain their original SHA-256 values.

Black and mypy pass 318 source/test modules; repository lint is zero, and
compileall, whitespace and strict Sphinx checks pass. All 318 source/test hashes
remain unchanged through final gates. The standalone profiler CLI also produced
valid JSON with two warm samples and a real Perfetto trace. No optional model or
Python environment packages were modified. The Perfetto v59.1 analysis binary
was downloaded into `/private/tmp` and verified against its published SHA-256.
The final receipt is `/private/tmp/vercor-performance-final-results.json`.
