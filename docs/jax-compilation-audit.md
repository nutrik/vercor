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
SciPy1.17.1, JAX/JAXlib0.11.2, Dinosaur1.5.1 and Flax0.12.10 come from scipy312.
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
