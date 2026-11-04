Exchange remapping profiling
============================

The November 3, 2026 optimization reduces work in field exchange and conservative
remapping. A dispatch resolves coverage once and reuses those operands when
retaining received support. One shared formula still determines support for
intensive values, source-area fluxes and target-area fluxes. No mask persists
across dispatches.

The conservative remapper now compiles its existing vectorized NaN cleanup,
gather, area weighting and sparse scatter sums together. Indices and weights
remain dynamic PyTree leaves. Fractional-area normalization still follows each
field's current finite samples. Shape checks and the shared safe division remain
outside that kernel, preserving eager errors, inactive values and gradients.

Measurement method
------------------

Baseline ``abc4494`` and the working source run in separate processes on the
same macOS arm64 CPU, Python 3.12.15 and JAX/JAXlib 0.11.2. Source grids have
16 x 32 cells; destination grids have 12 x 24 cells. Remapper calls include
missing samples and a shifted periodic destination grid. Dispatch and complete
coupler runs include evolving fractional coverage and dry target cells. Scalar
routes use conservative remapping; vector routes use spherical bilinear rotation.

Each timing synchronizes all returned leaves and outstanding JAX effects.
Nine warm calls follow one cold call; traces are captured separately. A second
fresh process pair repeats float64 measurements. Float32 uses separate processes
with x64 disabled before array creation. Each dispatch timing transfers twelve
times; each complete output-free coupler timing advances six steps from the same
immutable seed. This measures synthetic CPU workloads, without native-model,
accelerator or climate-throughput claims.

Results
-------

The independent float64 repeat gives the following median milliseconds.

.. list-table:: Synchronized warm CPU timings
   :header-rows: 1

   * - Workload
     - Before
     - After
     - Speedup
   * - Twelve conservative remaps
     - 8.025
     - 3.817
     - 2.10x
   * - Twelve fractional-area remaps
     - 11.665
     - 3.784
     - 3.08x
   * - Twelve intensive scalar dispatches
     - 41.086
     - 18.020
     - 2.28x
   * - Twelve source-flux scalar dispatches
     - 35.482
     - 15.348
     - 2.31x
   * - Twelve target-flux scalar dispatches
     - 38.514
     - 19.150
     - 2.01x
   * - Twelve vector dispatches
     - 96.020
     - 68.821
     - 1.40x
   * - Six complete host steps, intensive route
     - 66.668
     - 54.029
     - 1.23x

The other complete host workloads improve by 1.04x to 1.16x in this repeat.
The first process pair measures 2.01x to 2.31x scalar dispatch, 1.40x vector
dispatch and 1.14x to 1.26x complete host execution. Float32 independently
improves both remapper and dispatch workloads; all raw samples are retained in
:download:`exchange-profiling-results.json <exchange-profiling-results.json>`.
Scanned and enclosing-JIT controls have small, variable gains: these measurements
do not establish a consistent compiled-throughput improvement.

Cold invocation includes compilation for a workload's previously unseen shapes.
It is also affected by earlier cases in the same process: these per-case cold
numbers are not independent interpreter startup measurements. For example, the
repeat's first conservative call falls from 268 to 170 ms; the compiled
source-flux coupler rises from 241 to 307 ms. Full cold/warm samples are retained.

Numerical and profiler evidence
-------------------------------

Each pair saves 285 resulting arrays, including received fields, current and
retained coverage, and complete final state. Of these, 275 are byte-identical, including every host and remapper-control
array. Ten scanned-run leaves differ by at most 1.78e-15 in float64
and 9.54e-7 in float32 (maximum float32 relative difference 1.71e-7). NaN
locations, shapes and dtypes match. Existing tolerances remain unchanged.

Perfetto SQL counts ``CommonPjRtLoadedExecutable::ExecutePrepare`` events.
The twelve-remap captures fall from 720 to 492 launches in conservation mode
and 948 to 492 in fractional-area mode. Scalar intensive dispatch falls from
4,104 to 2,328; vector dispatch falls from 8,808 to 6,216. The complete host
intensive run falls from 6,987 to 5,852; its scanned counterpart remains one
compiled launch. None of these warm captures has a backend compilation event.

XProf 2.24.2 converts all twelve XSpace captures to overview, framework operation
statistics and trace-viewer data. Perfetto's Python API 0.59.1 analyzes the JSON
traces using the previously SHA-256-verified v59.1 trace processor. TensorBoard
2.21.0 loads the Profile plugin and serves both dispatch profiles' overview and
operation-statistics data with HTTP 200 responses. Profiling follows the
`JAX profiling guide <https://docs.jax.dev/en/latest/profiling.html>`_ and
`OpenXLA XProf documentation <https://openxla.org/xprof/>`_.

Reproduction
------------

Download :download:`the benchmark <exchange-profiling-benchmark.py.txt>` and
:download:`the trace analyzer <exchange-profiling-analysis.py.txt>` as
``benchmark.py`` and ``analyze.py`` into ``/private/tmp/vercor-exchange-profile``.
The benchmark records versions and hashes of every production Python source.
The baseline imports a separate ``git archive abc4494`` source tree. Use the
same reviewed dependency environment for both processes; the local commands are::

   profile_python=/Users/spj483/miniforge3/envs/scipy312/bin/python
   export PYTHONPATH=/private/tmp/vercor-veros170-stack:/private/tmp/vercor-profile-tools:$PWD
   "$profile_python" /private/tmp/vercor-exchange-profile/benchmark.py --label optimized --trace
   "$profile_python" /private/tmp/vercor-exchange-profile/analyze.py optimized
   "$profile_python" -m tensorboard.main --logdir /private/tmp/vercor-exchange-profile

For the baseline, substitute its archived source directory for ``$PWD`` and use
``--label baseline``. Independent repeats omit ``--trace``; float32 adds
``--dtype float32``. XProf, Perfetto and TensorBoard are optional profiling tools
in an isolated temporary directory, with no production dependency added.
Raw traces, arrays, scripts, saved tool responses and validation logs remain in
``/private/tmp/vercor-exchange-profile`` and the delivered evidence archive.

Validation
----------

Sixteen additional regressions cover all quantity/vector/source-support dispatch
combinations, changing geometry and finite-sample support, executable reuse, and
direct infinity/overflow errors. Existing coverage retains zero-support checkify,
conservation, periodicity, latitude orientation, float32/float64 and JVP/VJP checks.
The independent source/test review has no actionable findings. Final gate counts
are recorded in the machine-readable report and development log.

Final verification passes 2,731 stock tests and 1,774 fast tests. The native lane
passes seventeen cases plus the global adjoint rerun after correcting its asset
cache path, validating all eighteen native autodiff cases. This disjoint union
matches all 2,749 collected cases. Combined branch coverage is 92.6015% with the
unchanged 90% floor. Black and mypy pass for all 331 source/test files; strict
repository flake8 reports zero findings. Compileall, strict Sphinx and whitespace
checks pass. All source/test hashes remain unchanged during final verification.
The ordinary scipy Python 3.14 environment aborts during imports; these gates use
the existing scipy312 dependency overlays and writable temporary Veros assets.
