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
