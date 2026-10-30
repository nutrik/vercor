# Coupled component interface and wrapper audit

Audit date: 2026-10-21. Audited baseline: `964dcc0` on
`feat/veris-seaice`. The audit phase changed no production source or tests.
The user subsequently authorized implementation of verified fixes and Git
delivery on the same branch. Findings below retain baseline line numbers so
the original defects remain inspectable after remediation.

## Assessment and evidence boundary

VerCOR has one coherent integration pattern: structural `Component` objects
declare `ComponentSpec`, return field updates or `StepResult`, and keep evolving
native state in a runtime payload. A private binding performs setup, scalar and
dtype normalization, prefill, ownership, and validation. The same driver owns
receive/step/send for host, scanned JAX, and custom backends. One output session
owns sampling, cadence, accumulation, filenames, and writing.

Adding an ordinary component does not require changing core coupling logic.
The public-only installed extension fixture and `docs/plugin-authoring.md`
demonstrate configuration, structural components, payloads, custom remapping,
topology, workflow/backend, period output, and snapshots. A component must still
provide compatible *physical* fields: a common Python method signature alone
does not make a slab, prescribed dataset, and native prognostic model
scientifically interchangeable.

The audit scanned definitions and signatures in all 179 production Python
modules (25,882 lines at the baseline), then traced actual construction,
lifecycle, execution, exchange, output, and gallery call sites. Three independent
read-only reviews covered native adapters, slab/data/output paths, and
exchange/remapping. Numerical probes established the findings below. This is
an interface and implementation audit; it does not certify a native CAMulator
checkpoint, long-term climate skill, or a generic restart format.

The source inventory covers 30 root modules, 29 private runtime modules, 12
component modules, 40 external-adapter modules, 10 prescribed-data modules,
five slab modules, 15 gallery modules, eight setup infrastructure modules,
six output modules, six interpolator modules, four regridder modules, five
logging modules, four diagnostic modules, four flux modules, and one public
runtime module. The complete retained path inventory and dependency owners
remain in `DEPENDENCIES.md` and `docs/source-coupling-audit.md`.

## Interface and wrapper compatibility matrix

All components below expose `name`, `grid`, `spec`, and
`step(fields, context, payload=None) -> Mapping | StepResult`. Grid fields use
canonical trailing `(latitude, longitude)` axes. `ComponentSpec` owns inputs,
outputs, initial fields, execution, lifecycle, transfer, output, and masks.

| Component / factory | Wrapper and lifecycle | State, time, and exchange contract | Output / compatibility |
| --- | --- | --- | --- |
| Structural third-party component | No inheritance required; exact three-argument step checked at preparation | SetupResult seeds payload; mapping preserves it; StepResult replaces it. Host can restructure payload, JAX requires stable tree/shapes/dtypes | Shared output and injected remapping/topology/backend; no model-specific branch in generic dispatch |
| CallableComponent | Normalizes author callables accepting one, two, or three arguments; public method stays uniform | Same binding and runtime as structural plugins | No separate lifecycle or output implementation |
| DataComponent | No-op step; merges declarations and supplied fields; forwards lifecycle, execution, transfer, masks, output | Prescribed current/monthly/daily fields; explicit diagnostic inputs may be added by applications | Shared runtime-field provider; transfer-layout defect F3 |
| Slab atmosphere | CallableComponent, no native payload or special hooks | SST input; air temperature/winds and toy sensible/latent heat outputs | Shared output; lacks native atmospheric bulk-forcing capability |
| Slab ocean | CallableComponent, current fields, optional fractional surface mask | Heat fluxes drive explicit-Euler SST; restoring contracts anomalies | Shared output; SST-only, not the full ocean-to-Veris contract |
| Slab land | CallableComponent, current fields, optional fractional mask | Latent heat updates clipped soil moisture; land temperature prescribed | Shared output; toy moisture closure is not a land energy/water model |
| Slab sea ice | CallableComponent, no native payload | Diagnoses ice fraction from SST; no ice mechanics or total ocean flux exports | Shared output; deliberately not interchangeable with prognostic Veris |
| ERA5 atmosphere | Input-free DataComponent; setup diagnoses pressure, height, density, theta | Twelve monthly records; K, Pa, kg/kg, m/s, W/m2; rain/snow kg/m2/s summed before remapping | Shared monthly output; explicit observed 10 m channels |
| ERA5 land / ocean | Input-free DataComponent, monthly linear selection | Land temperature / SST K; static binary support; mixed-coast ocean exclusion is a discretization choice | Shared output; no physical feedback evolution |
| ERA-Interim ocean | Input-free DataComponent; source layout/longitude conversion | Monthly SST; Celsius to K once; salinity-derived wet support | Shared output; SST-only forcing |
| Prescribed Veros ocean | Setup-only native snapshot, read-only copied data, shared geometry | Monthly SST K, salinity g/kg, zero prescribed currents/SSH; no evolving native owner | Shared output; supports Veris forcing geometry, not prognostic feedback |
| JCM land | Input-free DataComponent, static or daily selection | T K, dimensionless soil availability, explicit fractional land mask; static partner broadcast to daily records | Shared output; daily climatology contract is 365 records |
| JCM atmosphere | CallableComponent, JAX; setup/prefill/validate; native construction at factory | Frozen native state/physics carry/forcing payload; Gregorian epoch and integer native substeps; ordinary intensive SST or explicit water/ice source products | Native provider and snapshot through shared coordinator; unnecessary host mirrors F7; deferred save-interval validation F8 |
| Veros global / ACC | CallableComponent; host stock or capability-gated JAX fork; setup hook | Full native payload; generic host copy before mutation, fork copy for JAX. Global bulk meteorology or total-flux mode; ACC internal forcing | Shared native provider/snapshot. Process-wide backend lock and fixed native climatology need explicit compatibility limits |
| Veris | CallableComponent, JAX; setup hook and pure native advance | Frozen state/diagnostics/constants/budgets; static wet support, evolving Area; explicit face conversion and fresh forcing each substep | Shared native provider/snapshot; interval-mean total ocean fluxes and current per-ice properties |
| CAMulator atmosphere | CallableComponent, host; setup hook, no implemented spinup | Frozen payload with cloned Torch state, cursor, forecast hour, predictions; declared vapor channel; Gregorian datetime forcing | Shared native provider/snapshot; timestamp and partial-block defects F5/F6; native float32 is intentional |
| CAMulator land | CallableComponent, host; setup hook | Immutable forcing cursor; selects prescribed TS at interval start; fractional land support | Default OutputSpec; unlike other land factories, no explicit output argument |
| Coupler / orchestrator | Constructor-only assembly; lazy preparation; initial_state and run | Same receive/step/send for all backends; exact plans/chunks, immutable stores, current versus received coverage, declared precision | One output lifecycle; optional native snapshots, no generic restart loader/finalizer protocol |

Primary implementation anchors: `vercor/components/_protocol.py:96`,
`components/contracts.py:190`, `components/_adapter.py:245`,
`components/runtime_execution.py:34`, `_runtime/driver.py:21`,
`_runtime/backends.py:90`, `_runtime/execution.py:116`, and
`output/_session.py:365`. Paths without a prefix in this paragraph are relative
to `vercor/`.

## Verified inconsistencies, prioritized

Severity: **P1** blocks a supported execution path or silently changes integrated
time; **P2** produces incorrect data for a valid boundary input or violates state
ownership; **P3** reports invalid configuration late. These findings were
established before remediation; final status and regression evidence follow.

| ID | Severity | Baseline location | Trigger, observed behavior, and necessary correction |
| --- | --- | --- | --- |
| F1 | P2 | `vercor/_regridders/conservative.py:21-50` | Identity compares centers only. Same centers with source longitude edges `[-1,1,3]`, target `[-1,.5,3]`, and field `[0,10]` incorrectly return `[0,10]`; exact overlap gives `[0,8]`. A single explicit edge axis is discarded when the other axis is absent. Identical grids also bypass an explicit exclusion mask and normalize validation. Resolve each axis independently; require matching effective cell bounds and absent exclusion mask for identity. Preserve bilinear point-grid identity semantics. |
| F2 | P2 | `vercor/_interpolators/_bilinear_weights.py:42-88` | Indices use clipped/wrapped longitude while weights use raw longitude. Nonperiodic source `[0,10]`, targets `[-5,5,15]` gives `[10,5,10]` instead of `[0,5,10]`; periodic equivalent targets `[45,405,765]` give `[.5,1,1]` instead of `[.5,.5,.5]`. Compute both indices and weights from the same mapped longitude. |
| F3 | P2 | `vercor/components/_adapter.py:264-300`, `_runtime/field_transfer.py:51-60`, `output/_session.py:316-323` | Non-current policy accepts one-record or static fields. At February 1 a one-record monthly/daily output-only component completes and writes four NaNs; a static monthly field slices the latitude axis. Enforce time-first 12-month / 365-day selectable records, exempt grid-shaped declared masks, and validate setup/prefill/selection through shared infrastructure. |
| F4 | P2 | `vercor/output/_netcdf.py:37-43,70` | Public OutputVariable accepts Python lists/tuples/scalars, but writer dimension checks require `.shape` before conversion. A custom provider with list coordinate `[0.,1.]` fails after opening its output file. Normalize once at the host writer boundary and define dimensions from that array. |
| F5 | P1 | `vercor/setups/_external/camulator_runtime.py:37-53,85-86` | NumPy datetime64[ns].item returns an integer; coercion then reads `.year` and fails before inference. Convert datetime-like host values explicitly, validate missing/unsupported times, and preserve standard datetime output. |
| F6 | P1 | `vercor/setups/_external/camulator_runtime.py:73-85,197-208` | A request for two native forcing records with only one remaining executes one model step, then advances a full coupling cursor. Reject empty/partial blocks before inference; retain payload on failure. Native cadence must agree with the declared interval. |
| F7 | P2 | `vercor/setups/_external/jax_gcm_runtime.py:478-524` | Forced-host JCM writes unused evolving `_state` and `forcing` mirrors on setup resources. Runtime and providers already use payload; no current corruption was demonstrated. Remove the second evolving authority and verify repeatability/resource identity. |
| F8 | P3 | `vercor/setups/_external/jax_gcm_state.py:208-240` | A six-hour coupling interval with the default one-day native save interval passes setup but fails on first native execution. Released JCM requires model step to divide save interval and save interval to divide coupling interval. Validate these configuration relations before bootstrap/spinup; do not silently change sampling. |

The conservative mask convention remains **True means excluded**. The fixes must
not invert it or change row normalization. For F3, daily means the existing
no-leap forcing axis: Gregorian leap day repeats the preceding no-leap record,
and a 360-day clock maps dates into the same 365-day record axis
(`vercor/forcing_index.py:32-46,63-84`).

Two additional boundary checks belong to F3/F6. With global x64 capability
enabled, monthly float32 fields were promoted by float64 interpolation weights;
exchange stores cast them back, while period output retained float64. Selection
must preserve the field dtype. Also Clock.start is a standard datetime even for
synthetic calendars; checking its Python type alone cannot reject CAMulator's
unsupported no-leap/360-day calendars. Both CAMulator setup paths must validate
SetupContext.calendar explicitly, before execution or dataset lookup.

## Field and recipe compatibility

| Exchange contract | Supported producers / consumers | Physical requirements and limits |
| --- | --- | --- |
| Atmosphere to ocean bulk forcing | ERA5, JCM, configured CAMulator -> Veros bulk mode | Bottom-level T/theta K, q kg/kg, density kg/m3, height m, wind m/s, radiation W/m2, precipitation kg/m2/s; intensive. Slab atmosphere lacks this contract. |
| Observed 10 m atmosphere to sea ice | ERA5 -> Veris observed_10m | Separate actual 10 m winds plus colocated model-level thermodynamics. JCM/CAMulator cannot supply observed winds by renaming model-level winds. |
| Model-level atmosphere to sea ice | JCM -> Veris model_level | Raw downward shortwave and open-water albedo plus bottom-level air; bulk solve diagnoses reference winds. Current ERA5/CAMulator declarations lack the complete recipe. |
| Ocean to Veris forcing | Prescribed Veros snapshot or total-flux Veros -> Veris | SST K, SSS g/kg, face currents m/s, SSH m; exact shared grid/explicit face shifts. Ordinary SST-only ocean components are incomplete. |
| Veris total ocean boundary | Veris -> Veros surface_fluxes | Interval-mean downward heat W/m2, freshwater and salt kg/m2/s, stress Pa; source_flux with binary source coverage. Scalars conservative, stress vector-capable bilinear. |
| Water/ice surface composition | Veris and open-water Veros -> JCM seaice_coupling | Form area products on the source grid. Per-ice T/albedo use component fraction; preweighted ice/open-water contributions use binary support. |
| Ordinary surface temperatures | Prescribed/slab/native surface -> ordinary JCM or slab ATM | K and intensive. CAMulator instead expects source_flux land/ocean contributions before summing them. Field name equality does not imply equal area semantics. |
| Land state to JCM | JCM/slab land -> JCM | Land temperature K plus dimensionless soil availability. ERA5/CAMulator land provide only temperature. |
| Turbulent heat to slabs | Slab ATM or JCM -> toy ocean/land | Downward W/m2. JCM grid-mean flux applied to toy tiles does not establish separately closed tile budgets. |

One scalar/vector dispatcher owns source weighting, covered normalization and
target scaling (`vercor/_runtime/exchange_dispatch.py:27-76`). Current route
coverage and last-dispatch received support have separate stores; recipient
validation uses the latter. Intensive meteorology remains available on wet
ice-free cells so Veris can form new ice. Ambiguous fan-in is rejected instead
of implicitly adding independent fluxes. These are deliberate, consistent
design choices, not defects.

## Lifecycle, communication, I/O, and restart

Construction policy belongs in frozen setup configuration; native framework
resources may be built at factory time. The one binding runs setup once per
Coupler, copies initial fields/payload, expands scalars and normalizes runtime
fields/grids to RuntimeOptions.dtype. Native payload precision is adapter-owned;
the generic binding copies arbitrary payload leaves rather than blindly casting
native categorical or opaque state.

JCM and Veris use pure native transitions. Veros host and CAMulator require
framework-specific mutable native operations within the copied payload boundary.
Veros host/JAX settings lock process-wide, so separate processes are required.
These adaptations preserve the public interface and are justified. An additional
ABC or another runner hierarchy would duplicate the existing protocol/binding.

Host StepContext includes an absolute datetime; scanned JAX StepContext has
time=None and a traced absolute step index. JAX components must carry native
time in payload or use step/dt, and must not depend on host-only datetime behavior.
Custom workflows may omit a component, but every invocation receives the base
clock dt; omission is not elapsed-time accumulation or general multirate
subcycling. A new multirate contract would need an explicit design.

Sequential feedback is intentionally lagged. In the full gallery,
LND -> CICE -> OCN -> ATM means ice sees the preceding atmosphere and pre-ocean
state, ocean receives current interval ice fluxes, and atmosphere receives
post-ocean open-water plus current ice properties. This order and field provenance
must stay explicit.

All output is opt-in and coordinated outside physics/backend code. Providers
sample post-step state at end-of-step time; period files use averaging-window
start timestamps. Period output selects forcing records; final runtime output
preserves raw record dimensions for inspection. Native snapshots receive final
payloads. **There is no generic restart format or loader**, state clock cursor,
or finalization hook in the 0.4 contract (`DESIGN.md:43`). Snapshots must not be
advertised as universally restartable checkpoints. In-memory continuation uses
a supplied RunState and an appropriately advanced application clock.

## Architectural risks and missing capabilities

These are not claims of reproduced bundled-model failures:

1. Field declarations contain names, not machine-checked units, signs, staggering,
   or area meaning. Keep explicit recipe documentation and producer/consumer
   tests; introduce optional field descriptors only when a concrete plugin
   needs automatic validation. Do not add a registry or Pydantic hierarchy now.
2. A source_flux route can inject a nonconservative regridder. The quantity name
   cannot guarantee a global area budget. Physical budget routes must choose
   conservative conservation normalization explicitly; vector/staggered remap
   across different grids remains unverified.
3. Veros checks exported surfaces centrally but lacks an adapter-wide check for
   hidden deep native numerical state. Defining active native masks is necessary
   before safely rejecting all nonfinite leaves; do not reject valid masked NaNs
   indiscriminately. DESIGN's blanket native-leaf validation claim is too broad.
4. Veros restoring phase uses native elapsed time and a fixed 365-day climatology,
   rather than binding arbitrary Coupler start/calendar. Reusing one Veros author
   component across Couplers may also accumulate enabled spinup. These need
   explicit date/reuse policy and executable native evidence before a change.
5. CAMulator declares latent/sensible heat and zero precipitation without a
   native turbulent/precipitation diagnosis. The supplied Veros bulk path
   recomputes turbulent fluxes, but a third-party route must not treat placeholder
   zero exports as measured physical fluxes. Checkpoint vapor semantics and
   native CREDIT execution remain uncertified here.
6. Native temporal representations differ: Veris total fluxes are interval means,
   JCM exports average saved native samples, CAMulator exchange uses the latest
   prediction and prescribed land uses interval-start TS. Document these choices;
   a common method signature does not establish interval-conservative feedback.
7. Output coordinate schemas validate dimensions/dtypes/attributes, while
   same-schema coordinate values may move between samples. A period mean then
   retains the latest coordinates. Moving-grid support or rejection needs an
   explicit provider policy. Public NumPy buffers can also alias OutputVariable
   values; frozen mappings alone do not freeze arbitrary array buffers.
8. CAMulator land lacks the output argument offered by other land factories;
   wrapper declarations can still be composed by applications. Add it only for a
   concrete need, keeping one OutputSpec owner rather than another I/O pathway.

## Recommended standard and reusable template

Retain Component, LifecycleHooks, SetupResult, StepResult, TransferPolicy,
ComponentMaskSpec, OutputProvider, Regridder/VectorRegridder, Workflow and
ExecutionBackend as the shared structural protocols. No new base class is
needed. Follow the executable public-only template in
[`plugin-authoring.md`](plugin-authoring.md):

1. Frozen plugin configuration owns model-specific construction choices.
2. Factory returns a structural component or CallableComponent, with complete
   physical declarations and explicit host/JAX capability.
3. Setup returns normalized fields and the sole evolving native payload.
4. Step reads imports/context/payload and returns declared updates plus payload;
   it never chooses filenames, cadence, or a second runtime-state authority.
5. Shared driver, exchange dispatcher and output session own communication and I/O.
6. Model-specific conversion utilities own units, axes, faces, native clocks and
   physical capability checks. Keep these narrow and test independently.

The remediation follows that standard: correct existing owners and add focused
regressions. It does not add a component hierarchy, field registry, universal
serializer, duplicate workflow, or speculative physics conversion.

## Verification and delivery

The initial scipy Python 3.14 fast run aborted before test results. The user
approved the existing scipy312 environment, without altering either environment.
Validation uses the previously installed released-native overlay at
`/private/tmp/vercor-coupling-names/native-overlay` (JCM 3.0.0, Veris 0.2.1,
stock Veros 1.7.0) and separate exact-fork Veros processes for AD acceptance.
The sandboxed baseline fast run reached 1,368 tests with one environmental
failure: the installed-gh regression could not bind its isolated loopback socket.
Subsequent suite runs use the authorized unsandboxed test boundary.

Detailed local evidence is retained under `/private/tmp/vercor-interface-audit*`.

### Implemented remediation

All eight verified finding groups are implemented using their existing owners:

| Finding | Result and regression evidence |
| --- | --- |
| F1 | Conservative identity now checks independently resolved cell bounds and absence of an explicit exclusion mask. Both edge axes, source/target declarations, normalization modes and true identity controls are tested in `tests/test_conservative_rectilinear_regridder.py`. |
| F2 | Bilinear cell fractions use mapped longitude. Scalar/vector whole-turn and endpoint tests cover float32/64; independently derived source JVP/VJP expectations remain explicit in `tests/test_bilinear_rectilinear_regridder.py`. |
| F3 | One helper validates setup, prefill and selector layouts; monthly weights adopt field dtype. Tests cover valid surface/level records, static masks, invalid records, float32/64 AD and identical send/output precision. Short synthetic fixtures now use complete climatologies while preserving selected values and gradient-zero regions. |
| F4 | Shared writer converts coordinate/data values to host once before checking dimensions. Public provider runs persist list/tuple/scalar coordinates and attributes. Output-free execution gains no host conversion. |
| F5 | CAMulator converts nanosecond/microsecond datetime scalars and scalar arrays correctly and rejects NaT before inference. |
| F6 | CAMulator checks complete forcing blocks and native timestamp cadence, including initialized nonzero-index cursors. Both setup hooks reject non-Gregorian calendars before noise, mutation or dataset lookup. Failure tests verify no native calls or payload advancement. |
| F7 | JCM no longer mutates setup-owner state/forcing during host execution; its runtime returns StepResult directly. Tests verify unchanged resources, equivalent repeated transitions from one payload, and real native forcing inputs. |
| F8 | JCM validates positive save interval, model-step divisibility and coupling-interval divisibility before bootstrap. Invalid configurations now fail at the owning lifecycle boundary. |

Independent cross-review found no remaining correctness findings in these fixes.
One report correction distinguished provider sample time from averaging-window
file timestamps. Architectural risks listed above remain separate and are not
claimed resolved by these boundary fixes.

### Verified checks

| Check | Result |
| --- | --- |
| Full stock suite | 2,371 passed; zero failures, errors or skips. Baseline was 2,295; 76 regression cases were added. |
| Fast suite | 1,426 passed; zero failures, errors or skips. |
| Exact-fork native AD | 15 passed; zero failures, errors or skips; Veros `7a8c964cf00b5aa0713c995edd760643a431b3c9` in a separate process. |
| Installed wheel/sdist/plugin/CLI probes | Four passed. Tested wheel matches all 179 production modules byte for byte. |
| Combined coverage | 92.68584030% aggregate, above unchanged 90% gate; statements 8,447/8,923 = 94.66547125%; branches 1,640/1,960 = 83.67346939%. |
| Formatting / typing | Black: 300 files unchanged; mypy: 300 files clean. |
| Maintained lint | `vercor tests tools benchmarks`: zero findings. Whole-repository diagnostic command still reports 200 existing findings in ignored historical evidence snippets. |
| Other checks | Strict Sphinx, compileall, whitespace, build and Twine passed. |

Coverage lanes are collected separately; their per-lane zero collection thresholds
do not replace the final unchanged 90% gate. Original and archived raw SHA-256
hashes remain unchanged. The combined branch arcs equal the exact union of both
raw lanes for all 164 measured non-gallery modules. Combination consumed only
disposable copies; preserved raw archives remain available independently.

The first full remediation run found three further fixture/documentation failures:
a three-record daily mask fixture, PROGRESS exceeding its 180-line bound, and an
outdated published-page inventory. The fixture now contains 365 records with the
same tested values, an existing progress continuation was combined without losing
history, and the navigation expectation includes this report. An initial attempt
to trim a trailing blank did not reduce the progress count; the bounded-log test
caught it. The subsequent full suite passed. No numerical assertion, skip,
timeout or coverage exclusion was weakened.

Twine was absent in both existing environments, so it ran from a temporary
target directory; neither conda environment was modified. The AD lane retains
the existing native int64-to-int32 scatter FutureWarning and two upstream JCM/
xarray FutureWarnings. These warnings also appear in the preceding naming-run
receipt and were not suppressed. Native CREDIT/checkpoint execution, Veros deep
state validity/calendar policy, generic restart, and the scientific limits above
remain outside this validation claim.

Git delivery uses the existing `feat/veris-seaice` branch and GitHub CLI
credentials. The final handoff records the pushed commit and independently
verified remote branch SHA.

## Bilinear geometry follow-up (2026-10-30)

The follow-up starts at `13c31ee` on `feat/veris-seaice`. Bilinear interpolation
uses sampling centers, not conservative cell edges. The actual ocean centers
start `[2, 6, 10]` degrees while the native edges start `[360, 4, 8]`. The
conservative missing-strip mechanism therefore does not enter bilinear lookup.
`RectilinearGrid` requires increasing coordinates; the low-level interpolator
also accepts strictly descending source axes and rejects mixed wrapped order
before calculating weights.

The audit traced public factories, cell lookup, fractions, spherical bases,
mask alignment, scalar/vector fallback, and explicit exchange dispatch. It
verified and repaired these defects in the existing owners:

| Finding | Verified behavior | Repair |
| --- | --- | --- |
| Degree/radian cancellation | The T31 seam sample at 358 degrees should retain source-last weight 2/3.75. Float32 at origin 720 instead gives 0.5333236456; an irregular affine latitude/longitude case also loses accuracy. | Calculate dimensionless fractions from degree differences. |
| Periodic lookup/weight phase loss | A representable target 0.0003 on source `[-180,0,0.001,90]` becomes 0.30517578125 instead of 0.3 in float32. At a fine seam, forming a full-turn coordinate can erase the target phase and give 1 instead of 0.9214918432. | Compare source candidates in each target's whole-turn frame without translating the precise target, and compute seam distances as sums of small intervals. Original columns, inclusive endpoints and wider grids retain their first-turn convention. |
| Spherical whole-turn sensitivity | Converting large longitude aliases directly to radians changes tangent bases and fallback distances through phase roundoff. | Reduce degrees with a signed remainder before conversion. This also retains small positive/negative coordinates; offsetting by180 first would erase their resolution. |
| Descending-source vector fallback | With one valid source u=30, scalar nearest/IDW returns 30 but vector fallback can return 0 because values are flipped twice against an already-oriented mask. | The shared extrapolator consumes prepared values. Scalar finishing prepares once; vector finishing uses its already-prepared source. |

One construction-only cached binary search handles periodic neighbors with
fixed iteration count and unchanged source-column indices. It subtracts each
target's whole turn from candidate sources for comparison, instead of adding a
large origin to a precise target. Nonwrapped widths remain raw degree
differences, including wider intervals; wrapped widths use positive signed-phase
differences. Thus fine seams, inclusive endpoints and wider source domains use
one lookup owner without sorting, a dense target/source matrix, or span-specific
fallbacks. The existing PyTree, eager validation, compiled finishing kernels,
and runtime ownership are retained.

The audit preserves nonperiodic endpoint clipping, ascending/descending latitude,
scalar NaN renormalization, target masks, and existing componentwise vector
extrapolation. Identical-grid scalar/vector no-ops remain unchanged. Explicit
exchange dispatch already applies source coverage and target support; it
continues to own those masks rather than introducing a second masking pass.

Targeted regressions use analytic scalar/vector values, exact rational geometry
of represented seam coordinates, independent JVP/VJP expectations, inactive
NaNs, nearest/IDW fallback, both precision policies, and compatibility controls.
The native harness compares actual T31/4-degree coordinate pairs in both
directions with SciPy `RegularGridInterpolator`, independently interpolating
Cartesian vector components before projection. It also observes prescribed
polar surfaces through two-day host and all-JAX runs and checks exact native
ocean/ice identity. Receipts are under `/private/tmp/vercor-bilinear-*`; final
gate results and delivery status are recorded in `PROGRESS.md` and the handoff.

This is a numerical implementation audit. It does not establish a temperature,
budget, or circulation correction from an interpolation error alone. Malformed
input validation and changes to the established vector extrapolation convention
are outside this repair.

Final verification passes **2,586 full stock, 1,636 fast, and 15 exact-fork
autodiff tests**, with zero failures, errors or skips. Combined statement/branch
coverage is **92.922171%**, above the unchanged 90% gate. The 84 new regression
cases and existing focused cases pass 153/153. Independent review passes 1,028
exact-coordinate scalar/index/order oracles per precision and masked vector/VJP
checks across source-axis orders, inclusive endpoints and wider grids.

Both final two-day native runs complete 96 atmospheric evaluations and write
nine shared output files. SciPy agreement is within 3.331e-16 for scalars and
9.992e-16 for vectors, with exact unit coverage and identical host/JAX geometric
receipts. Prescribed polar SST/ice match selected native forcing exactly.
Black, mypy on 312 files, maintained-source flake8, compileall, strict Sphinx, and
wheel/sdist build pass; the wheel matches all 180 production modules. The full
and native suites use the existing isolated `scipy312` setup after the known
`scipy` native-worker abort. Repository-wide lint retains 200 historical
`test_logs` findings. Raw coverage lanes and 312 frozen source/test hashes are
preserved in the final verification receipt.
