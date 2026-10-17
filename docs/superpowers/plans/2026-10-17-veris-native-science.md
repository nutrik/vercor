# Native Veris Scientific Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct and independently validate native spherical operators, physical-parameter AD, and conservative ocean boundary diagnostics.

**Architecture:** Change the native owners on an isolated branch from the exact existing dependency pin. Keep numerical controls static, physical values traced, and actual mass/salt diagnostics separate from virtual salinity forcing. Complete this plan before the dependent [VerCOR plan](2026-10-17-vercor-veris-integration.md).

**Tech Stack:** Python >=3.12, JAX, NumPy, pytest, Ruff, ty, native build/verification tools.

**Spec:** [Approved scientific design](../specs/2026-10-17-veris-scientific-validation-design.md), sections 1–3 and native parts of section 5; user approved written-spec planning on 2026-10-17.

## Global Constraints

- Native work starts at `a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e` on `feat/vercor-scientific-validation`; preserve the user's older editable checkout.
- No main/jax-only merge, release, tag or publication is part of this work. The user's dedicated-branch instruction takes precedence over the native guide's normal jax-only commit destination.
- Use pure JAX transitions, immutable PyTrees, explicit SI boundary quantities, lazy dependencies, the existing component lifecycle, and the shared output coordinator.
- Preserve the planar contract and its reference tolerances; keep original independent reference archives unchanged.
- Latitudes are physical degrees, with lengths ny and ny+1 respectively; radius is positive metres.
- Exact discrete identities target normalized float64 residuals at or below 1e-11; continuum checks demonstrate second-order interior refinement.
- `saltIce_ref` is nonnegative and fixed for a trajectory; reject formation from less saline water rather than silently changing existing ice salinity.
- Native `EmPmR` is total bulk ocean mass loss in kg/m2/s. Native salt forcing is `(EmPmR*S_local + 1000*J_salt_in)/rhoFresh` in m (g/kg)/s.
- Host definition construction remains device-free. Compiled kernels return JAX-compatible physics status; host runners and VerCOR check that status before accepting results.
- Write reproducing tests first. Run only one native pytest instance at a time; full correctness precedes every native code commit, without a redundant fast run next to a full run.
- Never add a numerical factor, relax a reference tolerance, skip a required distributed lane, or increase a timeout to obtain a pass.

## Review Focus

1. Optional spherical coordinates must survive definition-file round trips and regional/sharded slicing; partial or invalid descriptions must fail before allocation (N1).
2. Correct face lengths alone can introduce spurious tensor forces; nonuniform-grid work identities and solid-body rotation constrain the combined operators (N1).
3. A parameter update must refresh dependencies without device initialization during definition construction, stale derived leaves, or value-dependent recompilation (N2).
4. Fresh-water formation, salty-ice melting into fresh water, and invalid nonzero-salinity formation have different valid outcomes under JIT (N3).
5. Cleanup, snow-to-ice conversion, and intermediate substep failure must not disappear from mass/salt accounting or runner validation (N3/N4).

## Execution preflight

- [x] Invoke `superpowers:using-git-worktrees` at execution time. Keep VerCOR in its explicitly requested current branch. Create an isolated native checkout under `/private/tmp`, based on upstream `jax-only` at the exact pin; do not switch or edit the user's separate native checkout.
- [x] Read that checkout's `AGENTS.md`, `DESIGN.md`, `CHANGELOG.md`, `requirements-test.txt`, `pytest.ini`, and CI test commands. Record initial/remote SHAs and a clean native status. Claim the current task in its CHANGELOG.
- [x] Create/activate its isolated `.venv` with Python 3.12, install native declared dependencies and `requirements-test.txt`, and verify imports resolve to this checkout. Do not replace dependencies in either user environment.
- [x] Run the native baseline once and save concise logs. Treat existing failures as evidence to diagnose, not permission to weaken final gates.

## Shared native gates

For each code task, run its named RED case before implementation. During iteration use the focused fast selection; before its commit run `ruff check .`, `ruff format --check .`, `ty check .`, the repository's annotation check, `python -m pytest tests/ -q`, and `git diff --check`. All executable CPU/sharded/distributed cases must pass; only genuine unavailable-hardware skips are reported as such. Record counts, warning provenance and unsuccessful approaches in CHANGELOG. Stage only that task's files and journal update.

### Task 1: N1 — Spherical geometry and a consistent strain/stress pair

**Files:**
- Create: native `veris/grid/spherical.py`, `tests/grid/test_spherical_operators.py`.
- Modify: native `veris/grid/geometry.py`, `grid/ocean.py`, `grid/regions.py`, `model/experiment_file.py`, `execution/experiment_initialization.py`, `physics/dynamics/rheology.py`.
- Test: native `tests/grid/test_ocean_setup.py`, `test_model_regions.py`, `test_closed_y_stress.py`; `tests/model/test_experiment_file.py`; `tests/execution/test_model_physics_sharding.py`.
- Document: native `docs/api/initialization.rst`, `DESIGN.md`, `CHANGELOG.md`.

**Interfaces:**
- Consumes: existing `OceanGeometry`, `State`, `Configuration`, west/south face conventions and stored native cell areas.
- Produces: `OceanGeometry(..., latitude=None, latitude_edges=None, radius=None)`; all three optional values select spherical geometry together.
- Produces: `spherical_metric_overrides(geometry: OceanGeometry, *, dtype: str) -> dict[str, Array]` in `veris/grid/spherical.py`, returning the existing State metric/curvature keys for host geometry assembly. No new runner or global precision setting.
- Produces: existing `strainrates(...)` and `stressdiv(...)` signatures with consistent spherical operators; unchanged planar behavior.

- [x] **Write independent failing tests.** Own the synthetic geometry fixture in `test_spherical_operators.py`: nx=ny=8, tracer latitudes 48,52,...,76 degrees, longitude spacing 45 degrees, meridional spacing 4 degrees, R=6.370e6 m, both native timesteps 3600 s, initial hIceMean=1 m/Area=.5, periodic x/closed y. The uniform-north test uses u=0 and v=.1 m/s away from walls. Check:

  ```python
  def test_spherical_zonal_spacing_at_sixty_degrees(spherical_state):
      dx_actual = 1 / spherical_state.recip_dxC[3, 5]
      np.testing.assert_allclose(dx_actual, 6.370e6*np.deg2rad(45)*0.5, rtol=1e-12)
  ```

  In the same file define `test_uniform_northward_transport_matches_face_integral` with expected delta_h=9.786682438863892e-5 m and rtol=1e-11, and `test_area_weighted_strain_stress_work_identity` with normalized residual <=1e-11. The local `spherical_state` fixture supplies the specified initialized geometry. Include read-only NumPy 0D radius input, matching the snapshot's owned scalar representation.

  Add `test_solid_body_rotation_refines_at_second_order`, `test_uniform_isotropic_stress_has_zero_force`, and a nonconstant manufactured tensor case. Use independently evaluated spherical equations; assert observed smooth interior order >=1.8 at 4/2/1-degree resolution. Include southern/nonuniform latitudes, coast masks, and distinct face positions. Add `test_spherical_coordinates_survive_non_square_partitioning` on a 12x8 grid, split separately along x/y on two logical devices; every physical metric/curvature interior must match serial within rtol=1e-12. Add partial-coordinate, shape, nonfinite, monotonicity, invalid-radius, singular-grid and JSON round-trip cases; unsupported singular geometry must fail explicitly rather than produce infinities.
- [x] **Run RED.** `python -m pytest tests/grid/test_spherical_operators.py -q`; expect the missing spherical keywords first, followed by the demonstrated metric/curvature failures once the interface exists. Save failures and their first divergent operator.
- [x] **Implement the geometry and paired operators.** Derive each full staggered length from supplied tracer/edge latitude and equatorial spacing. Initialize curvature and reciprocals through the same owner, propagate coordinates correctly through regional geometry, and derive tensor divergence consistently with the area-weighted strain operator. Use native stored areas and field-specific halo policies. Keep the planar path's arithmetic/order where unaffected.
- [x] **Verify GREEN and compatibility.** Run the named geometry/model/sharding tests and the shared gates. Explicitly compare planar references, serial/two-device spherical operators, coordinate round trips and the nonconstant work identity. A spacing-only repair is not a completed task.
- [x] **Commit.** Update native journal and commit only this unit as `fix: implement consistent spherical sea-ice operators`.

### Task 2: N2 — Dynamic physical parameters with safe host/runtime boundaries

**Files:**
- Create: native `tests/model/test_physical_parameter_pytree.py`, `tests/numerics/test_physical_parameter_gradients.py`.
- Modify: native `veris/model/physical_constants.py`, `model/metadata.py`, `model/diagnostics.py`, `execution/advance.py`, `execution/rollout.py`, `execution/simulation.py`.
- Modify nested parameter users: native `grid/averaging.py`; `numerics/evp.py`, `freedrift.py`, `momentum.py`, `surface_temperature.py`; `physics/dynamics/area_mass.py`, `ocean_stress.py`, `rheology.py`; `physics/transport/advection.py`, `cleanup.py`; `physics/thermodynamics/growth.py`, `heat_flux_cesm.py`, `heat_flux_mitgcm.py`.
- Test: native `tests/model/test_experiment*.py`, `test_precision*.py`, `test_diagnostics.py`; `tests/execution/test_compiled_advance.py`, `test_lifecycle_sharding.py`; existing gradient/nonsmooth/distributed suites.
- Document: native `docs/reference/automatic-differentiation.rst`, `docs/api/model-data.rst`, `DESIGN.md`, `CHANGELOG.md`.

**Interfaces:**
- Produces: a frozen registered `PhysicalConstants` PyTree, with dtype as static metadata and independent physical values as leaves; dependent reciprocals/ratios/trigonometric values are recomputed, not independent leaves.
- Produces: `PhysicalConstants.base_parameter_names() -> tuple[str, ...]` and `with_overrides(self, **parameters: ArrayLike) -> PhysicalConstants`. Reject unknown, derived and dtype overrides; host calls validate domains, while traced calls preserve dynamic values.
- Preserves: `advance(state, configuration, constants, forcing=None, *, physics="coupled")` and the pure rollout API, now supporting parameter JVP/VJP.
- Produces: halo-shaped, runtime-real `Diagnostics.physics_status`: 0 valid, 1 invalid physical parameters; code 2 is reserved for N3's invalid formation state. Include registry metadata. Default/initial status is zero. Host Simulation rejects any nonzero status without accepting the candidate state; pure transforms return explicit status.

- [x] **Write failing parameter/host tests.** Pin the derivative and dependency contract:

  ```python
  def test_parameter_update_refreshes_traced_reciprocal():
      p = PhysicalConstants(h0=0.5)
      def reciprocal(x):
          return p.with_overrides(h0=x).recip_h0
      np.testing.assert_allclose(jax.grad(reciprocal)(0.5), -4.0, rtol=1e-12)
      np.testing.assert_allclose(jax.jit(reciprocal)(0.4), 2.5, rtol=1e-12)
  ```

  Test rhoFresh reciprocals/ratios and turning-angle dependencies, independent-leaf round trips, float32/64, copied caller inputs, invalid concrete values and forbidden overrides. Construct Experiment/JSON definitions with device discovery blocked: they must remain device-free. Instrument a small jitted objective to prove same-schema .4/.5 parameter values trace once. Pin explicit status for traced invalid h0, with no silent acceptance by Simulation.
- [x] **Run RED.** `python -m pytest tests/model/test_physical_parameter_pytree.py tests/numerics/test_physical_parameter_gradients.py -q`; expect absent methods or current scalar/tracer rejection, not fixture errors.
- [x] **Implement base-leaf reconstruction and dynamic kernel plumbing.** Reuse native metadata rather than introduce a second parameter registry. Keep host-only scalar validation separate from tree reconstruction/traced algebra; do not discover devices while building definitions. Remove `phys` from every nested static JIT argument. Pass replicated physical leaves explicitly through sharded transitions and preserve existing communication/checkpoint policies. Return/check physics status at the existing runner boundary.
- [x] **Verify GREEN from parameters upward.** Use h0/concentration, air/water drag/drift, iceConduct/cold thickness or TSurf, and dryIceAlb/illuminated snow-free thickness in identifiable regimes. Require finite nonzero chosen sensitivities, JVP/VJP inner-product agreement and central-difference step sweeps (1e-2,1e-3,1e-4 times the parameter scale; relative agreement <=2e-4 on a stable plateau). Exercise serial, two-device and distributed derivatives through pure advance/rollout. Check hard-transition one-sided behavior separately. Run shared gates.
- [x] **Commit.** Journal the supported parameter and fixed-initial-state semantics; commit as `feat: trace native sea-ice physical parameters`.

### Task 3: N3 — Actual ocean mass/salt exchange and snowfall enthalpy

**Files:**
- Create: native `veris/physics/thermodynamics/budgets.py`, `tests/physics/thermodynamics/test_ocean_budgets.py`.
- Modify: native `physics/thermodynamics/growth.py`, `physics/transport/cleanup.py` where a reproduced cleanup defect requires it, `model/diagnostics.py`, `execution/advance.py`, `execution/simulation.py` status messages.
- Test: native `tests/physics/thermodynamics/test_growth.py`, `tests/model/test_diagnostics.py`, `tests/execution/test_lifecycle_reference.py`, existing conservation/reference tests.
- Document: native diagnostic tables/physics/AD docs, `DESIGN.md`, `CHANGELOG.md`; retain every original reference archive and checksum.

**Interfaces:**
- Produces: frozen PyTree `OceanExchangeFluxes(mass_input, salt_mass_input)` with grid-shaped rates in kg/m2/s and kg salt/m2/s, respectively, positive into ocean.
- Produces: `ocean_exchange_fluxes(*, ice_change: Array, snow_change: Array, external_mass_input: Array, cleanup_ice: Array, cleanup_snow: Array, configuration: Configuration, constants: PhysicalConstants) -> OceanExchangeFluxes` in the budget owner. Ice/snow changes are final minus prepared pre-thermodynamic volume in metres; cleanup is signed raw post-advection minus prepared volume, so total stock change is thermodynamic change minus cleanup. External input includes actual diagnosed sublimation, precipitation and runoff in kg/m2/s. Transport redistribution is not a thermodynamic exchange.
- Extends native Diagnostics with `ocean_mass_flux`, `ocean_salt_mass_flux`; derives corrected `EmPmR=-ocean_mass_flux` and `(EmPmR*ocSalt + 1000*ocean_salt_mass_flux)/rhoFresh`.
- Produces native `Diagnostics.external_mass_flux` and `external_heat_flux`, downward-positive input from atmosphere/land into the modeled ice/ocean column in kg/m2/s and W/m2. Compute these independently from the actual boundary events/heat fluxes before phase change, not by rearranging the final inventory equation; they are required for extended budget verification.
- Extends Growth's maintained result tuple by the two canonical and two external-boundary arrays; update all internal unpackers/annotations atomically. `physics_status=2` identifies new ice formation with `ocSalt < saltIce_ref`; valid salty-ice melting remains allowed.

- [x] **Write analytic RED budgets.** Define `test_pure_freezing_matches_actual_mass_and_salt_budget`; isolate unrelated surface radiation/conduction as the audit probe does, and independently compute mass/enthalpy inventories. At So=35, Si=5, dt=300 s and pure 100 W/m2 freezing:

  ```python
  np.testing.assert_allclose(delta_h, 100*300/(900*334000), rtol=1e-11)
  np.testing.assert_allclose(salt_forcing, 8.982035928143713e-6, rtol=1e-11)
  np.testing.assert_allclose(mass_input, -900*delta_h/300, rtol=1e-11)
  np.testing.assert_allclose(salt_mass_input, mass_input*5/1000, rtol=1e-11)
  ```

  Cover freezing/melting, Si=0/2/5, wet/dry support, rain/snow/evaporation/runoff at A=0/.5/1, nondefault densities and exact latent/residual heat closure. For snowfall P=1e-4 kg/m2/s into ocean at freezing with no other forcing, require ice mass gain P*dt, net ocean mass input=0 and residual heat=0. A warm-water case consumes P*lhFusion and delivers P to liquid water. Reject new saline-ice formation from fresh water under JIT via status=2; permit saline-ice melt into it with finite positive actual salt input. Use Si=0 for fresh-water formation.
- [x] **Run RED.** `python -m pytest tests/physics/thermodynamics/test_ocean_budgets.py -q`; reproduce the 167.6667 salt error and missing open-water snow budget before correcting them.
- [x] **Implement events/budgets at their first owner.** Apply open-water snow enthalpy before phase change. Compute total bulk mass and actual salt from the diagnosed thermodynamic/cleanup changes and atmospheric boundary events; do not derive them from whole coupled-step thickness differences. Use the fixed bulk ice salinity consistently. Neutralize dry operands. Trace negative advection overshoots and small-positive cleanup independently: removing/clipping stock must have the opposite compensating ocean inventory change. Preserve instantaneous native stress semantics and net/shortwave accounting.
- [x] **Verify GREEN and reference provenance.** Add flooding, snow-to-ice conversion, complete melt, cleanup and intermediate invalid-status cases. Verify normalized actual mass/salt and modeled zero-layer enthalpy residuals <=1e-11 in analytic float64 cases against the independently computed external boundary diagnostics. Keep original reference files unchanged; where intended science changes an expected trajectory, retain the old provenance and add explicitly versioned, independently reviewed correction expectations, preserving per-field assertions. Do not blindly regenerate an archive from the changed solver or relax tolerances. Run shared gates.
- [x] **Commit.** Document corrected diagnostic meaning/units and validity domain; commit as `fix: conserve native sea-ice ocean boundary budgets`.

### Task 4: N4 — Long transport acceptance and native handoff

**Files:**
- Create: native `benchmarks/validate_spherical.py`, `tests/physics/transport/test_spherical_longrun.py`.
- Modify: native `tests/execution/test_model_physics_sharding.py`, `tests/distributed_framework_probe.py` for new parameter/status/diagnostic coverage, preserving existing unique cases; `benchmarks/README.md`, `CHANGELOG.md`.
- Run: native package build, existing installed/public-contract tests and full correctness checks.

**Interfaces:**
- Produces: runnable `python -m benchmarks.validate_spherical --steps 1000 --output <json>` with concise summary and machine-readable geometry/dtype/commit, inventory errors, extrema, refinement and status evidence. It reuses native operators; no alternate production solver.
- Produces: a verified native commit SHA and local wheel/source path consumed by the VerCOR plan. No placeholder hash is added to VerCOR.

- [x] **Write harness tests first.** Define `test_long_transport_report_checks_inventory_and_geometry` for a small valid run, `test_wrong_cartesian_metric_fails_spherical_tendency` as the negative control (conservation alone must not pass it), and `test_intermediate_physics_failure_is_retained` for a nonzero status followed by a valid step. Pin finite report diagnostics, an asserted geometry discrepancy and retained failure respectively.
- [x] **Run RED, then implement the focused verifier.** Use a smooth positive periodic advected field and analytically known flow, with thermodynamics/cleanup/ridging inactive. Run at least 1,000 native steps; assert stored-area volume drift <=1e-10 in float64, boundedness, and the manufactured/refinement checks from N1. This tolerance is a roundoff accumulation allowance, not permission to change physics.
- [x] **Exercise supported execution contracts.** Verify serial/two-device agreement and existing multi-process derivatives with dynamic physical leaves and new diagnostics/status. Preserve checkpoint/halo message-order regression assertions. Run only one pytest controller and log summaries instead of arrays.
- [x] **Run final native gates and build.** All shared checks pass, docs build strictly, and `python -m build` plus outside-checkout wheel import/API/advance/parameter-AD smoke succeeds. Record real hardware skips and hosted/local evidence separately. Commit the benchmark/acceptance unit as `test: validate spherical sea-ice transport and parameter execution` only after its full gate passes.
- [x] **Handoff to VerCOR.** Record the exact committed SHA, artifact provenance, native test counts and diagnostics/API contract in both journals. Keep the native branch local until its reviewed passing commit is ready to publish; later push it before installing the immutable remote pin in the final VerCOR transaction. Do not merge it into main or jax-only.

## Self-review and ordering

N1 implements spec section 1 and establishes geometry for N4. N2 implements native section 3 and defines parameter/status interfaces. N3 implements section 2, including the low-salinity domain and snow closure; it consumes N2 status and exports the exact diagnostics used by VerCOR C3/C4. N4 proves native extended conservation/execution and supplies the immutable handoff. Native source/host definitions, scalar and sharded paths, original reference provenance, parameter gradients, physics domains and all five Review Focus cases have named test owners. The dependent VerCOR plan owns the actual global 30-day/annual prescribed-forcing runs and synthetic coupled calibration.
