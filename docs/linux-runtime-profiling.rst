Linux CPU and CUDA runtime profiling
====================================

This follow-up starts from ``e15efe0604429be4c33c78e4c3cc1904a295c728``.
The earlier :doc:`runtime-profiling` study measured macOS CPU execution.
The new measurements exercise Linux host/API-JAX/outer-JIT execution, evolving
masks, numerical validation, scalar/vector remapping, irregular schedules,
output and slab physics on CPUs and CUDA 12.

All twenty-four CPU/GPU comparison pairs are complete. The full stock and
exact-fork AD lanes account for all 2,713 collected tests, with no failures,
errors or skips. Verification and independent review are recorded below.

Measured changes
----------------

Component mask resolution compiles its existing parent-support count. Central
state validation computes finite/range predicates in one reusable kernel but
retains both checks and their distinct errors. Bilinear interpolation fuses
four-corner gathers, invalid-corner neutralization, weights and Cartesian vector
accumulation. Vector projection joins the existing finishing kernel. Source
guards, shape checks, safe-division checks and extrapolation retain their
existing boundaries. Helpers inline into enclosing compiled programs.

These changes fuse existing vectorized array operations into stable compiled
kernels; they preserve the four-corner interpolation algorithm, route masks,
clock policy, precision ownership, physical parameters and scan scheduling.
Owner strings remain outside numeric cache keys. Changed values continue to
use the same executable and invalid later inputs still fail.

Hardware and procedure
----------------------

The uv-managed ``.venv`` uses Python 3.12.15 and JAX/JAXlib 0.10.2 with
``jax[cuda12]``. Direct magni execution uses a Tesla P100-PCIE-16GB with driver
580.196.07. The host has Xeon E5-2650v4 CPUs; GPU paired processes retain CPU
affinity 16--23. GPU work runs outside SLURM. CPU paired processes run on aegir
node441, Xeon E5-2683v4, using eight allocated logical CPUs with core binding.
OpenMP and OpenBLAS threads are limited to one.

Each precision uses three independent paired processes, alternating
baseline/candidate order, with nine synchronized warm samples. The same saved
harness runs against archived baseline and candidate source, records source
hashes and resets immutable inputs. Twenty cases cover complete execution,
output, physical slab components and isolated numerical controls. Additional
128 x 64 cases test whether gains persist beyond the default 32 x 16 grid.
Setup and first-invocation compilation costs are recorded separately.
Process imports and backend discovery precede those timing regions. Compiled
micro-controls batch ten resets per warm sample; other cases reset once.
Output cases use at least 24 hourly steps and fresh output directories.

Traces and tests run separately from acceptance timings. GPU execution enables
``JAX_PLATFORMS=cuda,cpu`` because callbacks require a local CPU device. The
initial CUDA-only probe failed at that callback boundary. GPU preallocation is
disabled. No GPU work is submitted to SLURM.
Native Veros separately defaults to CPU and updates JAX's platform setting;
native GPU validation therefore also sets ``VEROS_DEVICE=gpu``. Its pre/post
device assertions record the P100 explicitly. A first CUDA-environment-only
native probe selected CPU and is excluded from GPU integration evidence.

Warm results
------------

Ranges below contain the three paired warm-median speedups for the 32 x 16
workloads. Each backend is compared with its own baseline; the table does not
compare absolute CPU and GPU throughput across their different hosts.

.. list-table::
   :header-rows: 1
   :widths: 32 17 17 17 17

   * - Workload
     - CPU float64
     - CPU float32
     - GPU float64
     - GPU float32
   * - Scalar bilinear remapping
     - 7.83x--8.69x
     - 7.59x--8.51x
     - 6.66x--6.75x
     - 6.56x--6.98x
   * - Vector bilinear remapping
     - 8.71x--9.44x
     - 8.64x--9.14x
     - 7.80x--7.98x
     - 7.95x--8.14x
   * - State validation
     - 1.12x--1.26x
     - 1.14x--1.24x
     - 1.21x--1.26x
     - 1.24x--1.28x
   * - Dynamic-mask refresh
     - 1.12x--1.26x
     - 1.11x--1.20x
     - 1.16x--1.17x
     - 1.19x--1.20x
   * - Complete host run
     - 1.13x--1.28x
     - 1.11x--1.21x
     - 1.22x--1.23x
     - 1.22x--1.26x
   * - Complete JAX API run
     - 1.01x--1.02x
     - 1.00x--1.03x
     - 1.00x--1.03x
     - 1.01x--1.01x
   * - Retained outer-JIT run
     - 0.99x--1.04x
     - 0.97x--1.02x
     - 0.99x--1.04x
     - 0.99x--1.02x

At 128 x 64, across both precisions, CPU scalar/vector remapping improves by
7.66x--9.33x / 5.64x--6.79x and host execution by 1.13x--1.20x. GPU
scalar/vector remapping improves by 6.51x--6.96x / 7.53x--8.23x and host
execution by 1.16x--1.27x. The measured changes provide no general retained-JIT
speedup. Larger float64 GPU retained-JIT pairs range from 0.94x to 0.99x.

.. figure:: /_images/linux-runtime-speedups.svg
   :alt: Paired CPU and GPU warm speedups for two grid sizes.

   Markers show the median of three paired warm medians; whiskers show their
   minimum and maximum. The gray band spans the 5% acceptance boundary around
   equal performance. Every device/precision uses its own baseline.

Each retained target exceeds 5% improvement in every pair. No control has a
repeatable time regression above 5% within its three-pair precision/grid group.
Five float64 GPU control groups have one slower outlier, with a maximum
6.6% time regression: daily host/JAX output, conservative remapping and the
larger compiled refresh/rollout. All raw samples, setup/cold costs and these
outliers remain in :download:`the measurements <linux-runtime-profiling-results.json>`.

Cold compilation remains a tradeoff. Across the three float64 small-grid pairs,
median scalar first-call time rises from 277 to 372 ms on CPU and from 308 to
367 ms on GPU. Vector first-call time falls from 1,255 to 677 ms on CPU and
from 1,281 to 744 ms on GPU; its geometry construction still takes about
2.6 seconds. These are first invocations after preparation in the fixed harness
order, separate from process startup. Short-lived regridders can therefore see
a different tradeoff from repeated simulation steps.

An independent audit reads all 82,680 saved array comparisons: 82,331 are
byte-identical and the remaining 349 pass the unchanged scientific tolerances.
These counts include repeated cold/warm leaves, not independent observables.
NetCDF attributes and dimensions match across all 480 saved file descriptions.
CPU vector-interpolation differences are bounded by 8.88e-16 absolute in float64
and 4.77e-7 in float32; the largest relative differences occur near zero. GPU
conservative remapping uses atomic scatter-add accumulation: the unchanged
baseline already differs between cold and warm calls by 8.88e-16 in float64.
Comparisons retain tight scientific tolerances for this independently verified
GPU variability; CPU conservative remapping retains exact comparison.

Profiler evidence
------------------

JAX exports 32 fresh paired warm XSpace/Perfetto captures. XProf 2.24.2 converts
128 tool exports: overview, framework operations and trace viewer for both
backends, plus GPU kernel and memory profiles. Perfetto v59.1 uses its
manifest SHA-256-verified Linux trace processor. Separate uninstrumented
processes provide every speedup above; trace durations are excluded.

.. list-table::
   :header-rows: 1
   :widths: 36 16 16 16 16

   * - Capture
     - CPU preparations before
     - CPU preparations after
     - GPU kernels before
     - GPU kernels after
   * - Scalar bilinear remapping
     - 130
     - 17
     - 97
     - 19
   * - Vector bilinear remapping
     - 155
     - 16
     - 111
     - 20
   * - State validation
     - 129
     - 77
     - 125
     - 73
   * - Dynamic-mask refresh
     - 805
     - 637
     - 658
     - 490
   * - Complete host run
     - 8,595
     - 6,489
     - 8,987
     - 5,947
   * - Retained outer-JIT run
     - 1
     - 1
     - 408
     - 408
   * - JAX output API
     - 93
     - 93
     - 205
     - 205

CPU counts are ``CommonPjRtLoadedExecutable::ExecutePrepare`` events; GPU
counts are XProf kernel occurrences, not host executable calls. GPU vector
remapping also reduces H2D copies from 92 to 14 and D2D copies from 108 to 4.

GPU traces expose launch and transfer overhead alongside kernel work. The
baseline retained-JIT seven-step workload has 2,826 D2H events despite one
outer executable launch in both versions. Small validation conditionals and callbacks remain
an opportunity for a separate policy-preserving investigation; eager arithmetic
fusion alone does not establish a general compiled-rollout speedup.
XProf's occupancy values are theoretical, and no hardware utilization percentage
is inferred from those values.
The GPU host workload retains transfer overhead: H2D copies fall from 4,504 to
2,454, while D2H copies increase from 5,008 to 5,566. Collating validation
predicates without changing error or lifecycle boundaries remains an opportunity.
Output launch counts and per-call output-API compilation remain unchanged.

TensorBoard 2.21.0 loads XProf's enabled Profile plugin. Baseline and candidate
CPU/GPU overview and framework-operation endpoints, plus both GPU kernel-stat
endpoints, return HTTP 200 with actual JSON data: ten final frontend receipts.
Use one ``--logdir`` root: XProf run enumeration was empty with
``--logdir_spec`` in this environment. Captures remain outside timed samples.

Environment and verification
-----------------------------

Install development and profiler packages into ``.venv`` with uv. JCM 3 requires
JAX below 0.11 while Veris 0.2.1 declares JAX at least 0.11.2. Installing Veris
without re-resolving dependencies permits its actual runtime to be tested on
JAX 0.10.2: all ten compilation/rollout checks pass in this shared environment.
Only the alternative exact Veros AD fork is isolated, pinned to
``7a8c964cf00b5aa0713c995edd760643a431b3c9``.

Activate the environment's PATH for subprocesses as well as the test runner.
Artifact tests need ``pip`` and ``flit_core<4`` in addition to ``build``.
Native Veros uses a writable ``VEROS_ASSET_DIR``; the two copied global-ocean
assets retain their native MD5 verification. These correct the seven initial
environment failures without skips or scientific-tolerance changes.

Focused checks pass 143 validation, 153 bilinear CPU and 215 numerical/bilinear
GPU cases. The exact-fork native AD lane passes 21 cases; the additional marked
coupled rollout passes on CPU and on the P100. A global collection audit accounts
for all 22 marked cases, including that rollout outside the usual four-file
command. This native smoke uses actual 4 x 4 ocean/ice and T21 atmosphere models
for two steps with reduced EVP test numerics; its timing is not a throughput claim.
The final stock suite passes 2,691 tests and the fast suite passes 1,805.
Combining stock and all 22 AD cases gives 94.11018% branch-enabled coverage
(95.70406% statements; 86.92142% branches). All 320 source/test/benchmark Python
hashes remain unchanged throughout final verification. The additional native
GPU rollout brings actual GPU checks to 216. Black, strict repository Flake8,
mypy across all 320 Python files, compileall and the established Sphinx ``-W``
build pass. The installed wheel/sdist ownership and typing checks pass.

Independent final review found one important reproduction issue: preparing a
new snapshot omitted the environment freeze consumed by the analysis script.
The runner now captures that freeze before creating its output root. A failing
smoke reproduced the omission; the corrected new-root smoke verifies both
source identities, the environment and the full six-pair invocation plan.
The published resume runner also validates all twenty-four completed pairs.
No review findings remain deferred.

The uv test harness also needed two portable installed-artifact checks. A child
venv receives parent dependency directories through a path-only ``.pth`` after
the wheel is installed, retaining child-wheel precedence and an unset
``PYTHONPATH``. A module-path assertion proves ownership. Installed typing checks
reject project source directories while allowing the interpreter and dependencies
to live under the checkout's ``.venv``.

Reproduction
------------

Activate ``.venv`` for both the runner and subprocesses. Install the runtime with
``uv pip install --python .venv/bin/python -e '.[dev,jcm,veros]' 'jax[cuda12]==0.10.2'``
and install the pinned Veris tag with ``--no-deps``. Profiler versions are
``xprof==2.24.2``, ``tensorboard==2.21.0`` and ``perfetto==0.59.1``; the full
environment freeze is embedded in the measurements. Native checks require
``pip``, ``build``, ``flit_core<4`` and a writable, checksum-verified Veros asset cache.

Archive baseline ``vercor/`` with ``git archive`` and snapshot the candidate.
Run the identical saved benchmark file by absolute filename with each source
directory first on ``PYTHONPATH``. Check its recorded source and harness hashes.
The :download:`paired runner <linux-runtime-profiling-runner.py.txt>` automates
snapshot preparation, the six pairs in each matrix and saved-result comparison::

   python docs/linux-runtime-profiling-runner.py.txt --prepare \
       --root test_logs/linux-runtime-reproduction --lane cpu
   taskset --cpu-list 16-23 python docs/linux-runtime-profiling-runner.py.txt \
       --root test_logs/linux-runtime-reproduction --lane gpu

Use ``--prepare`` only with a new root. Reuse that prepared root for the larger
matrices, passing ``--nx 128 --ny 64`` and the seven ``--case`` selections recorded
in the measurements. ``--resume`` validates completed artifacts and options;
it never overwrites an incomplete invocation. For a single baseline/candidate
comparison, the underlying harness accepts the following form.
For example, using unused output directories::

   JAX_PLATFORMS=cpu PYTHONPATH=/path/to/baseline-source \
       .venv/bin/python /path/to/harness.py --directory /tmp/linux-baseline \
       --dtype float64 --steps 7 --repeats 9
   JAX_PLATFORMS=cpu PYTHONPATH=/path/to/candidate-source \
       .venv/bin/python /path/to/harness.py --directory /tmp/linux-candidate \
       --dtype float64 --steps 7 --repeats 9

Alternate the process order, repeat three times in each precision, then add
``--grid-nx 128 --grid-ny 64`` for the seven larger controls recorded in the JSON.
GPU runs set ``CUDA_VISIBLE_DEVICES=0``, ``JAX_PLATFORMS=cuda,cpu`` and
``XLA_PYTHON_CLIENT_PREALLOCATE=false``, unset ``LD_LIBRARY_PATH`` and retain
``taskset --cpu-list 16-23``. CPU captures used a separate aegir allocation;
GPU captures ran after all timings. Add ``--trace`` only to separate captures.

:download:`The trace exporter <linux-runtime-profiling-traces.py.txt>` consumes
paired ``baseline/`` and ``candidate/`` capture directories, verifies the
Perfetto binary against its packaged manifest, exports XProf tools and runs SQL
counts. Save captures under ``cpu-paired-captures/`` or ``gpu-paired-captures/``
in the prepared root, then run the exporter on each directory to produce its
``analysis.json``.

:download:`The analysis/audit script <linux-runtime-profiling-analysis.py.txt>`
reads saved arrays and metadata, rechecks acceptance, exports the measurements
and generates :download:`the standalone figure <_images/linux-runtime-speedups.svg>`.
Use ``tensorboard --logdir test_logs/linux-runtime`` to inspect the captures.

Raw timings, every saved numeric leaf, source snapshots, traces, tool responses
and complete validation logs are under ``test_logs/linux-runtime/``.
Capture follows the `JAX profiling guide
<https://docs.jax.dev/en/latest/profiling.html>`_; CUDA support follows the
`JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_.
Analysis uses `XProf <https://openxla.org/xprof>`_ and
`Perfetto Trace Processor <https://perfetto.dev/docs/analysis/trace-processor>`_.
