# Linux CPU and CUDA runtime performance design

The user's fresh request extends the earlier macOS-only audit to magni Linux
CPUs and directly accessed Tesla P100 GPUs, using CUDA 12 JAX in a uv-managed
`.venv`. GPU execution is outside SLURM; partition aegir is available for CPU
work. Work stays on `perf/host-jax-runtime`, with tested commits and an ordinary
push to that branch explicitly authorized. The previous study is reference,
not evidence of performance on this machine.

Use the current tip `e15efe0` as the fresh baseline. Profile host orchestration,
JAX API and retained JIT execution, numerical safety, dynamic masks,
scalar/vector regridding, slab physics and output. Reuse the current benchmark
and gallery tools. Separate compilation and setup, synchronized warm timings,
and instrumented XSpace/Perfetto captures. Analyze with XProf, Perfetto SQL and
the TensorBoard Profile plugin. Record hardware, package/source versions,
thread/affinity settings and profiler limitations.

Prefer bounded fusion/vectorization in existing numerical owners when traces
show repeated eager dispatch. Removing validation, changing error ordering,
persistent mutable caches, changing physical arithmetic or scan scheduling
would violate the task's constraints. Retain only candidates with repeated
paired benefit (target at least 5% across three independent comparisons), no
repeatable representative control regression above 5%, and equivalent saved
results at the existing scientific tolerances. Rejected trials remain in the
evidence log, with their reason.

Tests precede implementation changes. Cover both precisions, inactive NaNs,
infinities, masks, error boundaries, eager and nested JIT, vmap/scan, JVP/VJP
and compilation reuse. Full stock and isolated exact-fork AD tests, fast tests,
type checking, linting, formatting and documentation validation precede final
commits. Optional native model version conflicts must be isolated and reported;
do not edit installed packages outside `.venv` or weaken scientific tolerances.
Keep reproducible scripts and compact machine-readable measurements in the
repository; raw traces/arrays and verbose logs live in `test_logs/`.
