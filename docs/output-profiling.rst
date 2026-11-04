Output accumulation profiling
=============================

The November 3, 2026 optimization batches one selected output frame's numeric
reductions and accumulator updates into one compiled call. Host schema and
already-reduced count validation retain their existing field-specific errors.
Arrays remain separate tuple leaves, preserving each field's shape and dtype
without stacking or padding. The shared finite sum/count kernel still owns
missing-value arithmetic. Exchange sampling and native physics are unchanged.

Measured result
---------------

These are synthetic output-coordinator measurements on a macOS 26.7.2 arm64
CPU with 14 logical cores, Python 3.12.15 and JAX/JAXlib 0.11.2. Each invocation
accumulates 24 frames from the same immutable initial accumulator. There are
32 float64 fields, alternating between 32 x 64 and 16 x 32 cells, with missing
samples. History frames contain three samples per cell. Already-reduced frames
supply integer counts. Seven synchronized warm invocations follow a separate
cold invocation. Trace capture is excluded from timings.

.. list-table:: Final independent repeat, median milliseconds per 24 frames
   :header-rows: 1

   * - Input
     - Before
     - After
     - Speedup
   * - Point samples
     - 22.170
     - 9.057
     - 2.45x
   * - Three-sample histories
     - 23.076
     - 8.631
     - 2.67x
   * - Already-reduced sums/counts
     - 85.920
     - 58.511
     - 1.47x

A separate process pair measured 2.39x, 2.90x and 1.45x respectively. Both pairs
preserved all 246 saved arrays byte-for-byte, including counts and NaNs. The
experiment also measures one and eight fields; all sizes improved host warm
throughput. Raw samples and production-source hashes are in
:download:`output-profiling-results.json <output-profiling-results.json>`.

Perfetto SQL analysis of the captured host timelines counts executable launches
at ``CommonPjRtLoadedExecutable::ExecutePrepare``. Point/history frames fall from
2,304 launches to 24; already-reduced frames fall from 8,448 to 5,400. Their
remaining validation reductions stay outside the batched kernel. XProf 2.24.2
successfully converted all six XSpace captures to overview, operation statistics
and trace-viewer data. TensorBoard 2.21.0 loaded the Profile plugin, enumerated
both sets of runs, and served both point-profile overviews. Perfetto's Python
API 0.59.1 used the SHA-256-verified v59.1 trace processor.
The point-profile XProf timelines contain no warm backend-compile events.
Capture follows the `JAX profiling documentation
<https://docs.jax.dev/en/latest/profiling.html>`_; conversion and viewing use
the `OpenXLA XProf tools <https://github.com/openxla/xprof>`_.

Tradeoffs and limits
--------------------

The first 32-field invocation increases from about 22 to 42 ms for point
samples, 23 to 53 ms for histories, and 86 to 107 ms for already-reduced
samples. The warm savings amortize this cold cost within three further
24-frame invocations in this experiment. An enclosing JIT already fuses the
original arithmetic: compiled controls show no consistent throughput gain.
These results do not establish native-model, accelerator or climate-run speed.
Float32 correctness is tested separately; float32 speed is unmeasured here.

Reproduction and coverage ownership
-----------------------------------

The local evidence directory ``/private/tmp/vercor-output-profile`` contains
``benchmark.py``, ``analyze.py``, ``trace_counts.py``, synchronized samples,
saved arrays, raw Perfetto/XSpace traces, XProf conversions and TensorBoard
responses. The baseline is commit ``fd137c3``; it is loaded from a separate
source archive, without changing the working branch. The isolated profiling
tools are outside the installed Conda environments.

From the checkout, use the reviewed dependency overlay and temporary profiling
tools (the baseline substitutes its archived source root for ``$PWD``)::

   profile_python=/Users/spj483/miniforge3/envs/scipy312/bin/python
   export PYTHONPATH=/private/tmp/vercor-veros170-stack:/private/tmp/vercor-profile-tools:$PWD
   "$profile_python" /private/tmp/vercor-output-profile/benchmark.py --label current --trace
   "$profile_python" /private/tmp/vercor-output-profile/analyze.py current
   "$profile_python" -m tensorboard.main --logdir /private/tmp/vercor-output-profile

Four basic cases were removed from ``test_v0_4_output_providers.py`` only after
mapping every assertion to stronger unchanged owners:

* ``test_output_none_performs_no_io_or_provider_sampling`` is owned by
  ``test_disabled_output_never_samples_or_snapshots_even_in_output_directory``
  and ``test_unused_numeric_provider_is_never_traced_or_sampled[none]``.
* ``test_output_target_flags_disable_each_output_kind`` is owned by the same
  disabled-output test and
  ``test_all_disabled_target_remains_jit_and_gradient_compatible``.
* ``test_any_enabled_output_rejects_traced_state_before_io`` is owned by
  ``test_each_enabled_io_kind_rejects_traced_state[final]``.
* ``test_daily_output_accumulates_multiple_samples_and_resets_windows`` is
  owned by ``test_default_runtime_output_includes_input_fields`` across four
  backends, plus
  ``test_period_output_precomputes_all_frequency_boundaries_and_resets[day]``.

The shared averaging, native physics, dtype, invalid-input and autodiff tests
remain. New tests independently verify heterogeneous axes, reduced counts,
missing cells, float32/float64 and JVP/VJP behavior of the batched arithmetic.

Final verification passed 2,715 stock tests, 18 exact-fork native tests and
1,758 fast tests, with zero failures, errors or skips. The disjoint stock/native
union matches all 2,733 collected cases. Combined branch coverage is 92.5964%.
Black and mypy pass for all 331 Python source/test files, strict repository
flake8 reports zero findings, and compileall, strict Sphinx and whitespace
checks pass. An independent code review found no actionable issues. The normal
``scipy`` Python 3.14 environment aborted during native imports; verification
used the existing isolated ``scipy312`` overlay documented in the plan.
