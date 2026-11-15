Host and JAX runtime profiling
==============================

The retained changes reduce CPU dispatch and numerical-validation overhead while
preserving every measured state and output array. Numerical safety now compiles
invalid-count arithmetic in small reusable kernels, keeping shape validation and
error reporting at their existing boundaries. The kernels inline into enclosing
JAX programs. The host/custom driver checks the expected plan identity before
scanning rejection paths, making valid plan checks constant time.

Execution modes and measurement
-------------------------------

The gallery profiler accepts ``--backend host|jax`` and
``--invocation api|outer-jit``. Host defaults to ordinary API calls; JAX retains
the previous outer-JIT default. An explicit outer JIT requires JAX. The result
records the selected backend and invocation, so API orchestration costs are
distinguished from retained compiled execution. Existing timing fields remain.
For example::

   python -m vercor.setups.gallery.profile_runtime --backend host --invocation api \
       --steps 24 --float-type float64 --repeats 9 --json
   python -m vercor.setups.gallery.profile_runtime --backend jax --invocation outer-jit \
       --steps 24 --float-type float64 --repeats 9 --json

Both the gallery and developer benchmark wait for arrays and callbacks. Developer
benchmark setup includes case construction and numeric input readiness. Its
``cold_seconds`` means one first invocation after preparation; process imports
and backend discovery are excluded. Gallery setup also includes any lazy backend
initialization. Warm samples reset the same immutable input. The two compiled
micro-controls batch ten resets per warm sample and report per-call time; cold
always measures one invocation. Output cases use ordinary API calls with fresh
directories, and reject reuse of their invocation root. Artifact reads and
trace capture occur outside the timed regions.

``benchmarks/benchmark_runtime.py`` provides sixteen lazy cases spanning valid
plan checks, finite-value validation, dynamic masks and scalar/vector remapping,
complete host/API-JAX/outer-JIT runs, irregular workflows, daily output and
actual slab physics. Seven-step irregular runs repeat a two-step schedule with
a one-step tail and include omitted components. Output cases use at least 24
hourly steps. The finite-check workload includes construction of twelve masked
fields as well as their validation.

Measured CPU results
--------------------

The baseline is ``3a48d7fb79d036508b456c9fe4310e6dea74e036``. Measurements use
three independent paired processes per precision, alternating baseline/candidate
order, nine warm samples, and fresh per-process compilation caches. The local
arm64 macOS CPU reports 14 logical CPUs; JAX/JAXlib are 0.10.2, NumPy is 2.4.2,
and ml_dtypes is 0.6.0. Logging is disabled below ERROR. The following ranges are
paired warm-median speedups for the 32 x 16 workloads:

.. list-table::
   :header-rows: 1
   :widths: 56 22 22

   * - Workload
     - Float64
     - Float32
   * - Four-component host run, seven steps
     - 1.45x--1.46x
     - 1.48x--1.50x
   * - Same coupled run through the JAX API
     - 1.06x--1.07x
     - 1.05x--1.07x
   * - Slab physics on host, seven steps
     - 1.59x--1.62x
     - 1.60x--1.61x
   * - Host daily-output API run
     - 1.13x
     - 1.15x--1.18x
   * - Conservative remapping, two normalization modes
     - 1.83x--1.87x
     - 1.89x--2.06x
   * - Scalar coverage refresh
     - 1.37x--1.40x
     - 1.38x--1.48x

All representative complete-run controls, including retained JAX scans, remain
within the 5% regression limit in every pair. Steady compiled execution has no
general speedup: its small changes vary by case and precision. The float32
finite-check compiled micro-control varies from 0.91x to 1.09x; it is reported
separately from complete-run controls. At 128 x 64 the complete host workload
improves by 1.33x--1.48x across both precisions. The isolated 4,096-plan check is
402x--419x faster; that microbenchmark excludes state validation and physics.

The remapper and output improvements arise from their shared numerical checks;
their numerical algorithms retain their existing owners. Every isolated retained
target improves by more than 5% in all three pairs. Setup and first-invocation
costs are retained in :download:`the complete machine-readable measurements
<runtime-profiling-results.json>` rather than inferred from warm results.

All 36,000 saved array comparisons are byte-identical, including every cold/warm
sample, both precisions, both resolutions, NetCDF numeric variables and model
timestamps. Paired NetCDF dimensions and attributes match exactly. These are
saved comparisons, including repeated leaves, rather than independent physical
observables. Scientific and AD tests independently validate the contracts.

Profiler evidence and rejected approaches
-----------------------------------------

Separate warm captures export XSpace and Perfetto traces. XProf 2.24.2 converts
all fourteen captures to overview, framework-op statistics and trace-viewer
data. Perfetto 0.59.1 uses its SHA-256-verified v59.1 trace processor. CPU
``CommonPjRtLoadedExecutable::ExecutePrepare`` counts decrease as follows:

.. list-table::
   :header-rows: 1

   * - Capture
     - Baseline
     - Retained source
   * - Finite-check workload
     - 240
     - 132
   * - Coverage refresh
     - 1,589
     - 805
   * - Complete host run
     - 14,340
     - 8,595
   * - Retained JAX run
     - 1
     - 1
   * - JAX daily-output API
     - 125
     - 93
   * - Conservative remapping
     - 80
     - 24
   * - Bilinear vector remapping
     - 183
     - 155

These are CPU executable preparations, not accelerator kernel counts. Hardware
accelerator counters are unavailable on this CPU. Trace durations include
instrumentation and are excluded from speedup calculations. TensorBoard 2.21.0
loads the XProf Profile plugin and enumerates all fourteen runs. Overview and
framework-op endpoints return HTTP 200 for both host profiles. Retained JAX
traces have no warm backend compilation;
API JAX output retains one backend compile per call in both source trees.
Capture and synchronization follow the `JAX profiling guide
<https://docs.jax.dev/en/latest/profiling.html>`_; analysis uses
`XProf <https://openxla.org/xprof>`_ and
`Perfetto Trace Processor <https://perfetto.dev/docs/analysis/trace-processor>`_.

Two trials were rejected or corrected. Sharing raw component masks within one
refresh accelerated eager refresh by 1.26x--1.37x, but its compiled refresh
control regressed by 5.7%--8.4% in two float32 pairs. The original mask-resolution
implementation is retained, with additional shared/self-route, AD and failure
ordering tests. Initial non-inline count kernels slowed float64 API JAX output
by 7%--27%; explicit inlining removed that regression. No persistent mask cache,
changed scan widths, removed validation boundaries or altered lifecycle calls
are part of the retained source.

Reproduction and validation
---------------------------

Run the identical benchmark file in separate interpreters with ``PYTHONPATH``
pointing to each source tree; the script records imported source/harness hashes.
For example::

   profile_python=/Users/spj483/miniforge3/envs/scipy312/bin/python
   "$profile_python" -m benchmarks.benchmark_runtime \
       --directory /private/tmp/runtime-candidate --dtype float64 --steps 7 --repeats 9
   PYTHONPATH=/path/to/baseline "$profile_python" /path/to/benchmark_runtime.py \
       --directory /private/tmp/runtime-baseline --dtype float64 --steps 7 --repeats 9

Repeat in reverse order, then original order; repeat with float32. The larger
matrix adds ``--grid-nx 128 --grid-ny 64``. Select cases with repeated ``--case``.
Use an isolated profiler environment and a separate invocation with ``--trace``
for instrumentation. Point ``tensorboard --logdir`` at the resulting trace root.
Raw samples, arrays, source variants, tool responses, scripts and validation logs
are retained under ``/private/tmp/vercor-runtime-performance``.

Runtime validation uses the user-approved scipy312 environment. The temporary
Veris overlay is the actual released commit; all 81 Python files match the
installed 0.2.1 source, whose original distribution lacked provenance metadata.
The exact Veros AD fork runs in a separate process. Installed environments are
unchanged. Static mypy uses scipy/JAX 0.11.2 because pristine scipy312/JAX 0.10.2
has two pre-existing ``Config`` stub errors in an unchanged gallery test.

The final full suites pass 2,669 stock and all 22 marked AD cases, with no
failures, errors or skips. Combined branch-enabled coverage is 94.05%, above the
unchanged 90% floor. All 320 Python hashes remain unchanged through full
verification. Fast stock passes 1,783 cases. Black and mypy pass all 320 Python
files; strict repository Flake8 has zero findings. Compileall, strict Sphinx,
API/distribution contracts and whitespace checks pass. Independent whole-branch
review has no actionable findings.
Native ocean and coupled output runs validate integration; their concurrent
smoke timings are excluded from performance claims. These synthetic CPU results
do not establish accelerator or climate-scale throughput.
