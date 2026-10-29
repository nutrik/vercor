Veris–Veros–JCM coupling validation
===================================

Current release integration (2026-10-19) uses Veris v0.2.1 and JCM 3.0.0,
Gregorian clocks and nine routes. JCM 3 removed separate land turbulent channels;
the unused prescribed-land import route is removed. The record below describes
the preceding integration; :doc:`exchange-coverage` documents current contracts
and the reviewed dependency exception.

This record concerns the functional coupling in ``run_veris_veros_jcm.py``.
The execution order is LND, CICE (native Veris), OCN (prognostic Veros), ATM
(JCM). Source code, tests, native data and the shared output coordinator are
reused; no additional model or integration loop was introduced.

Delayed full-ice bounds repair (2026-10-29)
------------------------------------------------------------

A geometry-only reproduction identified inconsistent longitude arithmetic in
``ConservativeRectilinearRemapper``. Overlap widths were computed by subtracting
degree coordinates and then converting the interval to radians. Destination
cell areas instead subtracted already-converted radian coordinates. Cancellation
at large longitudes made the denominator slightly inconsistent with the
overlap numerator: a uniform unit field mapped to the T31 grid reached
``1.0000000000000082``, 37 float64 machine epsilons above one.

The static ocean-mask path bounds its remapped fractions, while physical
``ice_fraction`` retains the conservative remap. As evolving ice approaches
full coverage, the mismatch can exceed JCM's two-epsilon ocean-area guard.
The callback traceback reports that rejection; the callback is not its cause.

The remapper now computes both longitude widths as degree differences before
conversion. This preserves its spherical overlap formula and existing exchange
ownership. JCM's bounds, endpoint reconciliation, and native physics are
unchanged. Six new regression cases fail before the fix and pass afterwards:
float32/64 unit coverage across periodic seams and shifted origins, plus a
compiled full-ice remap into JCM on the Gaussian T31 grid. All 89 focused
remapping/surface cases pass, including existing conservation and gradient
checks. Independent masked-grid checks also retain finite JVP/VJP values and
mass conservation to dtype precision.

The exact supplied Veris parameters are used by the single global validation
harness at ``/private/tmp/vercor-ice-bounds-global.py``. Its receipts, captured
surface inputs and shared monthly output are kept under
``/private/tmp/vercor-ice-bounds-global/``. The first harness preparation stopped
after initialization on a payload-inspection API error, before any coupled
timestep; it was corrected before launching the expensive trajectory. Captured
old-normalization comparisons replay the new trajectory's incoming ice values;
they are diagnostics, not a second historical trajectory.

The run uses the packaged 365-day Gregorian clock, daily coupling, the ordinary
two-day JCM spinup and monthly output. ``VerosConfig(execution="jax")`` selects
the differentiable ocean; all four component specifications declare JAX
execution. ``VerisConfig.physical_constants`` contains exactly::

    {
        "pStar": 18764.412842134217,
        "cStar": 21.70524853277675,
        "PlasDefCoeff": 1.5797477010208008,
        "airIceDrag": 0.0013712803103803446,
        "waterIceDrag": 0.00987660047535896,
        "h0": 0.36901695784821165,
        "uStarBase": 0.008896379124487205,
    }

Validation uses isolated Python 3.12 targets with JAX/JAXlib 0.11.2, Flax
0.12.10, JCM 3.0.0, Veris v0.2.1 and exact differentiable Veros
``7a8c964cf00b5aa0713c995edd760643a431b3c9``. Installed environments are
unchanged. Stock Veros 1.7.0 tests run in separate processes. The complete
stock invocation passed 2,446 cases; its sole RST underline failure passed
after correction (1/1). Fast stock passed 1,497 cases and the exact-fork
AD lane passed 15, with no remaining failures or skips. Black, maintained
source/test flake8, mypy, compileall and strict Sphinx passed. Repository-wide
lint retains 200 pre-existing findings in ignored scientific snippets.

The single corrected global trajectory completed all 365 coupling steps and
8,760 native ice substeps in 1,120.25 seconds, writing 43 shared-output files.
Every surface-input bounds check passed. The largest incoming ice-area excess
was one float64 epsilon, below the unchanged two-epsilon allowance. Replaying
the old area normalization on these incoming values first flags one cell on
day 253 and again flags one cell on day 270. This explains why the error can
appear late as ice coverage evolves; the replay does not certify the exact
original failing trajectory or its date. The six new regression cases also
pass directly in the regular ``scipy`` environment; broad native validation
uses the isolated Python 3.12 stack because ``scipy`` native collection aborts.

Revisions and environment
-------------------------

* VerCOR task base: ``e0ccd9e450e4e40dca39fd652c4eb256426d099f`` on
  ``feat/veris-seaice``. Reviewed feature tip: ``26e16beda7f8f8ffd08b3a767489a898d89169ee``.
  Final checks and publication are recorded below.
* Native Veris base fetched before edits:
  ``6c205a356dd1b25c26430f899cb8998523ab3447``. Shared albedo extraction:
  ``225aa34c4f12c337c7198f67927ca098afae3f9c``, published to the requested
  ``team-ocean/veris`` branch ``feat/vercor-scientific-validation``.
* Separate differentiable Veros lane:
  ``7a8c964cf00b5aa0713c995edd760643a431b3c9``. Stock NumPy and JAX fork
  processes were isolated. Python 3.12 Conda environment ``scipy``, float64;
  pytest temporary roots were outside the checkout.

Requirement evidence
--------------------

.. list-table:: Approved specification acceptance items
   :header-rows: 1
   :widths: 5 35 60

   * - Item
     - Requirement
     - Executable evidence
   * - 1
     - Native albedo equivalence and initiation
     - Native ``test_surface_properties.py`` checks independent category/snow,
       hemisphere/dry/wet truth and committed pre-extraction solver/growth/JVP
       baselines. Maintained native full suite: 1745 pass, 5 platform skips.
   * - 2
     - Distinct surfaces, masks and limits
     - ``test_jcm_surface_fields.py`` covers land/water/ice, zero/partial/full
       cover, absent NaNs, threshold coasts, dtype, analytic gradients and
       endpoint one-sided derivatives. The independent review reproduced full-ice
       coastal cancellation; float32/float64 regressions now verify bounded
       cover, inactive-water NaNs and precision-only reconciliation with
       preserved limiting derivatives. Native tests cover both hemispheres.
   * - 3
     - Source products and current ocean surface
     - ``test_ocean_surface_contributions.py`` proves correlated conservative
       remapping (142 K versus the incorrect product of averages, 145 K),
       tagged extraction order and actual native heating. The four-component
       rollout compares the same-interval composition with current CICE and OCN.
   * - 4
     - Land moisture, fractional terrain and selected fluxes
     - ``test_jcm_surface_runtime.py`` captures native applied fields, preserves
       caller terrain, selects total/land channels and checks signs/units.
       Static and mixed daily land layouts have explicit regressions.
   * - 5
     - Air/radiation provenance and reference winds
     - ``test_reference_winds.py``, ``test_veris_forcing.py`` and
       ``test_shared_surface_fluxes.py`` cover stable/neutral/unstable profiles,
       reference-height identity, Galilean covariance, calm/inactive operands,
       raw/water-absorbed SW, pressure and unchanged public bulk outputs.
   * - 6
     - Dynamic masks and derivatives
     - Existing dynamic-mask disappearance/reactivation tests remain passing.
       New native ice–ocean and ice–JCM–ocean tests check JIT/JVP/VJP/FD. A
       forward-identical stopped CICE-to-ATM remapper isolates a nonzero
       atmosphere-mediated derivative in a later ocean interval.
   * - 7
     - Immutable carry, initialization and compatibility
     - ``test_jcm_initial_forcing.py`` compares native direct diagnostics while
       preserving every dycore/cross-step carry leaf; verifies noleap epochs,
       spinup alignment and repeated-run independence. Native hook tests retain
       ordinary SPEEDY behavior and diagnostic PyTree schemas. Full legacy
       adapter/runtime/output/public contracts pass.
   * - 8
     - Exact assembly and installed use
     - ``test_veris_veros_jcm_gallery.py`` checks all ten routes, one ocean
       initialization, precision before construction, lazy models, public CLI
       copying and mode rejection. An installed wheel's CLI-copied file ran
       outside the checkout with real global models and shared output.
   * - 9
     - Actual coupled run and output
     - Source and installed two-day global runs use default native physics,
       two-day JCM spinup and 2319 wet ocean cells. All owned numerical leaves
       are finite; Veris completes 48 native steps. Ice cover evolves from zero
       to maximum 0.2014795963, marine albedo to 0.2070061255, and final marine
       temperature spans 269.1237778–303.0433263 K. Period files match current
       canonical fields and lat/lon coordinates with K/1 units.

Model boundary and limitations
------------------------------

JCM SPEEDY has one marine temperature. The adapter supplies an area-composed
effective marine temperature and albedo, actual normalized marine concentration,
and separate land temperature/moisture. It does not resolve separate ice/water
turbulent channels. Initial atmospheric diagnostics use native climatological
surface forcing without advancing the native carry; later atmosphere intervals
use coupled incoming surfaces. Air and prescribed land heat inputs retain the
explicit sequential lag. Land receives fluxes but remains prescribed.

Reduced native derivative tests use a 4x4, two-layer source-only ocean fixture,
T21 JCM and eight EVP iterations; these are declared test numerics. The global
smoke uses model defaults. Smooth derivatives are verified away from native hard
wet/dry and phase transitions. Neither smoke nor numerical agreement establishes
long-term climate skill. The earlier native scientific validation and its
explicit refinement settings remain documented in :doc:`veris-validation`.

Detailed commands, XML reports, gradient metrics, review and artifact evidence
are retained locally under ``test_logs/veris-veros-jcm/``. Final checks and
publication are recorded in the completion record below.

Completion record
-----------------

The validated implementation is ``47ba4975c9ec58d2d36e1267346176535a600f07``.
Its ordinary fast-forward push to ``nutrik/vercor:feat/veris-seaice`` was
verified with GitHub CLI; the native branch ref exactly matches the pin above.
The following handoff commit changes only documentation and the bounded
progress log.

Final gates after the independent review fix:

* Stock fast: 1264/1264, 222.517 s.
* Complete stock suite with branch coverage: 2185/2185, 293.981 s.
* Separate exact-fork Veros autodiff coverage: 15/15, 581.901 s.
* Combined coverage: 92.50%; statements 94.61%, branches 82.87%.
  The repository's combined 90% gate passes.
* Black, strict maintained flake8 (zero findings), mypy on 296 files,
  compileall and whitespace checks pass. Flake8 excludes only git-ignored
  historical evidence fragments and this task's scratch workspace.
* Strict Sphinx, wheel/sdist build, Twine and final installed-wheel public
  CLI-copy global smoke pass. The wheel's scientific modules match the
  validated source byte for byte.

The fresh whole-branch review found one Important coastal-coverage defect,
no Critical findings and no Minor findings. Seventeen of nineteen new checks
first failed; two invalid-coverage/derivative controls already passed. After
correction, 99 focused cases and all final gates above pass. Independent native
albedo review additionally passed 16/16 extraction/equivalence cases.

For full ice on tiny coasts, ``1-(1-R(W))`` and independently remapped ice
coverage can differ by floating-point cancellation. Reconciliation is limited
to two machine epsilons of the runtime dtype, before deciding water support.
It preserves limiting identity tangents while rejecting material coverage
violations. The existing 0.001 static support threshold and native physics
parameters are unchanged.

At native albedo 0.70 in the declared reduced three-interval test, JVP is
-0.8314104516324147 and VJP is -0.8314104516328807 mK per albedo unit.
Central differences at 1e-2, 1e-3 and 1e-4 are -0.8314114656589787,
-0.8314104661621968 and -0.8314104510039749. The forward-identical stopped
CICE-to-ATM control has derivative -0.8310681765206815; their difference,
-0.0003422751121991974, isolates a nonzero atmosphere-mediated ocean response.
Prescribed polar marine surfaces
-------------------------------

The paired setup now preserves native JCM terrain and prescribed marine
forcing beyond the coupled ocean's geometric footprint. Boundary cells
combine coupled and prescribed contributions by fractional area. Native SST
and ice retain their original time-selection policy, and prescribed albedo
uses the native SPEEDY diagnosis. Period output and final snapshots include
the same applied marine fields.

Regression coverage includes native land and ocean poles, fractional latitude
and longitude boundaries, float32/float64 composition, endpoint roundoff,
inactive operands, JVP/VJP checks, ordinary and sea-ice coupling, and native
host/JAX integrations. A changing-forcing case uses several model steps per
save interval and several saves per coupling interval to check sampling dates.

The instrumented real global host check completed two coupling days with
96 native atmospheric evaluations. Across 252 prescribed polar cells, applied
SST and ice matched the original selected forcing exactly at every evaluation.
Their original ranges were 237.6954--271.1052 K and 0.086948--0.998946,
respectively. Atmospheric latitude centers span about -87.159 to 87.159 degrees;
the inspected ocean centers span -78 to 78 degrees. Period files and the final
snapshot retain nonzero SST, ice and albedo there. This is a short coupling
and provenance check, not a climate-skill assessment. Local harness and
receipts are ``/private/tmp/vercor-mask-global.py`` and
``/private/tmp/vercor-mask-global-host/verification.json``.

The all-JAX global check repeats the same 96 native evaluations with the exact
Veros differentiable fork and obtains zero polar SST/ice discrepancy as well.
Both lanes write nine shared output files. Its receipt is
``/private/tmp/vercor-mask-global-jax/verification.json``.

The end-to-end ice-albedo acceptance test retains native polar marine forcing
and the original gradient tolerances. At albedo 0.70, JVP and VJP agree at
about -0.552228598 mK per unit albedo. A deterministic convergence probe gives
-0.552228319, -0.552226235 and -0.552195919 at finite-difference widths
``1e-5``, ``1e-6`` and ``1e-7``. Those widths check the local derivative;
the former ``1e-2``--``1e-4`` perturbations give different finite-perturbation
behavior. Both stencil ranges are recorded in the acceptance artifact. The
exact native regime or numerical-solver cause of the coarse discrepancy is
unproven; no native physics, geography or tolerance was changed to obtain
agreement. This check certifies the local derivative, not finite excursions.

Final verification passes 2,472 stock tests and 1,522 fast stock tests with
zero failures, errors or skips. Exact-fork AD acceptance passes all 15 cases
(14 initially, then the corrected local-stencil case). Combined line/branch
coverage is 92.892266 percent, above the unchanged 90 percent gate; raw lane
data hashes are preserved. Black and mypy cover 310 source/test files, strict
maintained-source lint is clear, and compileall, strict Sphinx, wheel/sdist
build, Twine and whitespace checks pass. The wheel's 180 Python files match
tested source byte for byte. Whole-repository lint retains 200 historical
snippet findings. Validation uses isolated Python 3.12 dependencies because
the required scipy baseline crashed workers before collection; installed
environments remain unchanged.
