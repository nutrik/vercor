# Veris scientific validation, calibration and ocean feedback

Status: written specification approved by the user on 2026-10-17, including
synthetic parameter recovery with a reusable calibration workflow, native
Veris fixes on a dedicated branch, and an immutable VerCOR dependency update.
Production implementation awaits plan review and execution-method selection.

## Intent and scope

Complete the three scientific follow-ups from the existing Veris integration:
spherical and extended-duration numerical validation, physical-parameter
calibration, and validated salt/ocean feedback conventions. Correct demonstrated
defects at their native owner rather than duplicate native solvers in VerCOR.
Use pure JAX transitions, immutable PyTrees, explicit SI boundary quantities,
lazy dependencies, the existing component lifecycle, and the shared output
coordinator. Keep implementations small and specific to the demonstrated need.

VerCOR work stays on `feat/veris-seaice`. Native work starts at the pinned
upstream `jax-only` commit `a9d6c0ade52fbae4c0d1da694878943d6dbf7e6e` in an
isolated checkout on `feat/vercor-scientific-validation`; do not modify the
user's separate, older editable `jax-only` checkout. No main/jax-only merge,
release, tag or publication is part of this work. After verification, push the
native branch and repin VerCOR to its exact reachable commit; push VerCOR with
GitHub CLI after its final checks pass.

Scientific accuracy here means independent equation/operator and conservative
budget agreement, refinement, stable extended trajectories, and verified
synthetic inverse recovery. Observed climate skill is not an acceptance claim:
the user selected synthetic calibration, not an observational dataset.

## Audit and approach

Three specialized audits and parent reruns on the exact installed pin found:

| Area | Reproduced discrepancy | Owner |
| --- | --- | --- |
| Spherical spacing | Zonal spacing is twice the physical value at 60 degrees | Native ocean geometry and VerCOR snapshot |
| Meridional transport | Uniform 1 m thickness has zero native tendency rather than 9.78668244e-5 m after one hour in the analytic case | Native face metrics/transport |
| Spherical momentum | Solid-body rotation produces spurious strain; changing lengths alone gives a spurious force under uniform isotropic stress | Native strain/stress divergence |
| Salt | Nonzero ice salinity mixes differently normalized terms; isolated forcing is 167.6667 times its mass budget | Native thermodynamic diagnostics |
| Snow | Open-water snowfall is absent from freshwater accounting | Native thermodynamics |
| Calibration | Native physical constants are host scalars/static nested JIT arguments | Native physical constants and VerCOR resources |
| Feedback | Only final substep diagnostics survive; existing source coverage would multiply total-cell fluxes by concentration again | VerCOR wrapper and exchange declaration |

Detailed reproducible probes and results are in
`test_logs/veris-followup-audit.md` and its three sibling probe/result pairs.
The unchanged VerCOR stock baseline passes 1,955 tests; Black, strict flake8,
mypy and compilation pass. These tests demonstrate compatibility and existing
operation, not the missing independent spherical truth.

The chosen approach is a native correction followed by a thin integration
update. A VerCOR-only guard/calibration addition cannot correct the native
operators. Copying or monkeypatching physics into VerCOR would create competing
owners and is rejected. A generic optimization framework, parameter registry,
or new component lifecycle is unnecessary.

## 1. Native spherical geometry and operators

Extend native `OceanGeometry` with optional `latitude`, `latitude_edges` and
`radius`. Supplying all three selects rectilinear spherical geometry; omitting
all three preserves the existing planar contract and its reference tolerances.
Reject partially supplied descriptions. Latitudes are physical degrees, with
lengths ny and ny+1 respectively; radius is positive metres. Existing spacing
vectors remain equatorial zonal and physical meridional distances in metres;
derive full staggered lengths using the supplied latitude positions. VerCOR
takes coordinates/radius from the same Veros snapshot as the masks/areas; it
must not combine VerCOR's default radius with another radius's geometry.

Construct lengths and reciprocal metrics at their actual C-grid positions,
including longitude cosine factors and curvature. Treat native west/south face
indexing and physical halo boundaries consistently. Correct strain and tensor
stress divergence together; use the area-weighted strain/divergence work
identity to constrain the discrete operators. Correcting only lengths fails
the uniform-isotropic-stress test and is explicitly insufficient.

Validation uses independent manufactured fields: scalar longitude/latitude
gradients, uniform northward transport, solid-body rotation, constant isotropic
stress, and nonconstant symmetric stress. Exact discrete identities target
normalized float64 residuals at or below 1e-11. Smooth continuum comparisons
must exhibit the expected second-order interior refinement, rather than a
tolerance chosen to accommodate the current discrepancy. Include nonuniform
latitude spacing, closed walls, periodic longitude, coastal masks and both
hemispheres. Exercise serial and supported native sharding with the same
geometry/operator definitions; real GPU/multi-host claims require actual
available hardware and observed results.

Conservation uses stored authoritative areas. Veros's midpoint areas differ
from exact spherical integrals by about 0.02031% at four degrees; analytic
tests must name the quadrature and radius instead of treating this known
discretization difference as a bug or correcting it with a fitted factor.

## 2. Native water, salt and heat accounting

Give actual mass and actual salt mass distinct, explicit diagnostic ownership.
New canonical ocean-feedback diagnostics are rates per total horizontal wet-cell
area, including open water. Define total bulk fluid mass input, including its
dissolved salt subset, positive into the ocean in kg/m2/s; actual salt input
is positive into the ocean in kg salt/m2/s. Growth,
melting, snow storage/melt, precipitation, evaporation, runoff, flooding and
transport cleanup must close those budgets with the correct signs.

Give retained `EmPmR` the explicit corrected meaning of total bulk mass loss
from the ocean, including ice exchange, in kg/m2/s. It is the negative of the
canonical mass input and contains no retained-salt reduction. Derive native
`forc_salt_surface` as `(EmPmR*S_local + 1000*J_salt_in)/rhoFresh`, with units
m (g/kg)/s; this local-reference native diagnostic is distinct from VerCOR's
fixed-reference Veros conversion. Never reconstruct canonical mass/salt from
an equivalent freshwater diagnostic. Existing independent reference archives
remain intact; explicitly test scientifically intentional corrections rather
than silently regenerating an oracle from changed code.

Keep the existing uniform bulk ice-salinity model: `saltIce_ref` is nonnegative
and fixed for a trajectory, and all modeled ice contains that salinity.
Formation from water less saline than that prescribed ice is outside this
model's physical domain and must fail clearly, including under transforms;
fresh-water formation uses zero ice salinity. Melting salty ice into fresher
water remains valid and supplies its actual salt. Do not silently reset the
salinity of existing ice as ocean salinity changes. This closes mass/salt
budgets without introducing an unrequested prognostic brine tracer.

For snowfall over open water, use instantaneous melting with its corresponding
latent cooling in ocean energy accounting. Snow is supplied at its modeled
freezing reference; there is no independently prescribed snow-temperature
parameter or invented sensible-heat correction. Test the resulting water and
zero-layer enthalpy budgets, including refreezing at the freezing point. This
is a stated model closure, not a claim that real open-water snowfall always
melts instantaneously.
At the freezing point, an isolated snowfall/refreezing case must retain the
input mass as new ice, have zero net ocean mass transfer and zero residual
ocean heat. Apply the snow enthalpy before solving phase change; appending a
latent cooling term after the growth update would fail that closure.

Native ocean residual heat is upward-positive W/m2. The VerCOR ocean heat input
is its negative. Shortwave already participates in net heat; retain its
penetrating component as a separately named diagnostic and do not add it a
second time to net heat. The existing Veros adapter applies total heat at the
top tracer cell; adding a new vertical optical deposition model is outside
this boundary-flux correction. Check latent plus residual heat against the
prescribed inputs in isolated freezing and melting cases.

Test ice salinity 0 and multiple nonzero values, valid formation and rejected
formation below the prescribed ice salinity, salty-ice melting into fresh
water, rain/snow/evaporation at ice fractions 0/interior/1, dry cells, cleanup
overshoots and snow-to-ice conversion. Test density changes with independently
calculated SI mass budgets and finite zero-local-salinity diagnostics.

## 3. Native physical-parameter AD and calibration

Make native physical parameters dynamic leaves of a frozen registered PyTree.
Keep numerical shape controls, precision policy, iteration counts and boundary
switches static. Validate physical domains at the host definition boundary;
compiled paths use tracer-safe algebra and appropriate transform-safe checks.
Recompute dependent physical quantities from their base parameters with JAX
algebra. Derived reciprocals and ratios are not independent calibration leaves.
Remove physical parameters from nested static JIT arguments throughout the
used native kernels; changing only the wrapper is insufficient.

The native public definition remains a host object. Pure advance/rollout accepts
the physical PyTree, so fitting does not repeatedly initialize geometry or
recompile solely for new values of same-shaped physical leaves. Preserve dtype,
fixed PyTree schemas, output-free execution and existing state/forcing AD.

Add an immutable `VerisConfig.physical_constants` mapping forwarded to native
`Experiment(constants=...)`, distinct from numerical `custom_parameters`.
Keep evolving native state and runtime physical parameters in the Veris payload;
static resources retain only numerical configuration and coupling alignment.
Provide `vercor.setups.replace_veris_parameters(state, parameters, *,
component="CICE") -> RunState` for base physical scalar overrides in a prepared
state, using native metadata, tracer-safe dependent quantities and dtype
normalization. It returns a new state without mutating a component owner or
exposing private payload access. Reject numerical/derived/unknown keys and
wrong component payloads before execution. Geometry and the supplied initial
state remain fixed when differentiating parameters of a continuation; rebuilding
parameter-dependent initial conditions remains an explicit separate operation.
No general payload editor or general parameter subsystem is added to the coupler.

Freshwater mass conversion has one owner: the traced VerCOR
`freshwater_density` supplies both mass-to-water-equivalent conversion and the
native freshwater-density/ratio values used during that transition. Reject
conflicting authored ownership at setup. Do not equate other native and bulk
constants merely because their names resemble each other.

The packaged `run_veris_calibration.py` example fits native lead-closing
thickness `h0` from synthetic concentration observations. Keep a reusable
calibration function in that focused example, using the public replacement
helper and a pure fixed-wet-support weighted residual, including ice-free
cells. Use bounded SciPy minimization with JAX value/gradient evaluation; no new
optimization dependency or general fitting framework is needed. Supply explicit
truth, bounds, scales, observation mask and convergence diagnostics. Verify
several interior truths from multiple initial guesses and a held-out smooth
forcing scenario. The audit's derivative-free recovery of 0.42999977735 m for
truth 0.43 m is feasibility evidence, not the final implementation.

Validate nonzero `h0` sensitivities, representative drag/conductivity/albedo
sensitivities in identifiable forcing regimes, JVP/VJP inner-product agreement,
and directional finite-difference step sweeps. Establish scalar/parameter-layer
agreement before the complete coupled inverse problem. Exact binary transitions
retain hard physics: test branch-local derivatives away from switches and
one-sided/finite behavior at switches. No straight-through derivative or smooth
membership surrogate is introduced.

## 4. Conservative VerCOR ocean feedback

Keep final instantaneous native diagnostics for native output. Separately
accumulate coupling-interval flux means over every native substep using
`sum(dt_native * flux) / dt_coupling`. These are rates; never divide by a
timestep twice. Include heat, total bulk mass, actual salt mass and total
ocean stress. Keep all carries and accumulator shapes fixed for JIT/AD.

Expose these canonical interval-mean feedback fields in the dedicated
`VERIS_TO_OCEAN_FLUX_FIELDS` recipe:

| Field | Units | Positive direction |
| --- | --- | --- |
| `sea_surface_heat_flux` | W/m2 | Into ocean |
| `sea_surface_mass_flux` | kg/m2/s | Into ocean |
| `sea_surface_salt_mass_flux` | kg salt/m2/s | Into ocean |
| `ocean_u_stress`, `ocean_v_stress` | N/m2 | East/north force on ocean |

The prognostic example shares the exact ocean/ice grid and native area
quadrature. Its acceptance proves native-area conservation on that shared
geometry. General different-grid budget assertions additionally require
agreement between native and remapping area conventions; do not infer those
assertions from a same-grid test.

Native ice-ocean stress is per ice-covered face area. Multiply it once by the
face concentration used by that momentum solve, and combine the complementary
direct air-water stress on open water. Use the shared bulk calculation once
for each prescribed forcing pack. Shift native west/south stress to canonical
Veros east/north faces without admitting coastal/exterior flow. A final tracer
ice fraction is not a substitute for the earlier momentum face concentration.
Test changing concentration and traction to distinguish
`mean(A_face*tau_ice)` from `mean(A_face)*mean(tau_ice)`.

Add `Exchange.source_coverage: Literal["component", "binary"] = "component"`.
The default preserves existing behavior; binary coverage selects the current
computational binary support without its fractional concentration. Require
an explicit quantity for the binary override, so compatibility-mode routes
cannot silently change meaning.
Use explicit source-flux quantity and binary source coverage for already
total-area ocean boundary fluxes. Preserve conservative remapping and vector
transformation, topology restrictions, inactive-NaN safety and empty support.
This supports complete melt to zero without dividing by concentration.
Apply the binary override consistently to weighting, remapped coverage,
active validation and stored dispatch support. Vector tests include distinct
valid east/north face supports, coasts and closed exteriors; tracer-cell
support alone does not establish valid face support or staggered conservation.

Add `VerosConfig.surface_forcing: Literal["atmosphere", "coupled"] =
"atmosphere"` for global prognostic Veros. Coupled mode consumes the validated
total boundary fluxes as their sole coupled owner; atmospheric bulk
rain/snow/evaporation/heat is not added again. Atmosphere mode keeps its current
behavior. Reject coupled mode for native-internally-forced ACC. Coupled mode
does not apply the old below-minus-1.8-C cooling proxy that currently zeroes
both heat and salt.

Retain VerCOR's existing fixed `surface_salinity_reference` for rigid-lid virtual
salt conversion (default 34.7 g/kg), explicitly distinct from native local
salinity used in freezing/thermodynamics. If M is downward-positive total mass
input and J is downward-positive actual salt input in kg salt/m2/s, the native
Veros salinity forcing is `(-S_ref*M + 1000*J) / rho_fw` in m (g/kg)/s. Pure
freezing yields `mass_freezing*(S_ref - S_ice)/rho_fw`. This reference convention
matches the existing freshwater pathway and is tested independently; no new
local-versus-reference policy switch is added.
The fixed-reference Boussinesq virtual-salt budget differs from a literal
combined ocean/ice physical-salt inventory when local salinity differs from
the reference. Validate actual native mass/salt inventories separately from
the declared reference conversion, and report that approximation explicitly.

Veros owns conversion of downward heat by its own `rho_0 * cp_0` and subsequent
tracer integration by timestep/top-cell depth. Avoid applying either factor
twice. Preserve independent optional temperature/salinity restoring; diagnose
its contribution separately in budgets. Reject incompatible total-forcing
fields or route combinations before physics.

Expose prognostic Veros surface salinity, canonical face currents and SSH in
addition to SST. Keep the prescribed-data gallery as a prescribed experiment;
add `run_veris_veros_era5.py` as a separate prognostic ocean/ice/atmosphere example
using the same geometry, forcing, flux and output owners. Use the established sequential lagged coupling
order and record exchange timing explicitly rather than create an implicit
solver. Test actual host and exact-fork JAX ocean tracer increments with simple
analytic fluxes, plus a complete small coupled feedback rollout and its smooth
parameter gradients.

## 5. Extended validation and acceptance

After independent corrected operators and budgets pass, run:

- At least 1,000 stable smooth transport steps with thermodynamics/cleanup
  inactive, checking stored-area volume conservation, boundedness and refinement.
- A 30-day prescribed global integration with finite native states, nonnegative
  volume, bounded concentration, zero dry-cell storage, scalar budget summaries,
  interval flux accounting and split-run equivalence.
- An explicit annual prescribed-forcing benchmark and timestep-refinement
  comparison, recording forcing provenance, native/coupling timesteps, radius,
  dtype, initial state, native commit and maximum budget/refinement errors.
- Synthetic parameter recovery, independent gradient checks and complete small
  prognostic feedback budgets. Extended-run stability is not climate skill.

Use fast focused tests during development, write reproducing tests before
fixes, and preserve unique existing assertions. Native full correctness tests
include supported distributed/sharded contracts; do not hide a failure by
increasing timeouts or excluding a difficult lane. Record unavailable hardware
honestly. Native development follows its own Ruff/format/ty and CHANGELOG rules;
VerCOR uses Black, strict flake8, mypy, compileall, bounded PROGRESS updates,
fast/full stock tests and separate exact-fork Veros AD processes. Use the
working Python 3.12 `scipy` environment for VerCOR; keep dependency installations
and native work isolated from the user's environments and editable checkout.

Update public signatures/manifests, output metadata, architecture/dependencies,
examples and documentation for intentional contract additions. After correctness
passes, verify package builds and the installed pinned integration, perform an
independent scientific/gradient/code-quality review, commit small passing units,
push native then VerCOR branches, and verify both remote SHAs with GitHub CLI.
Report measured outcomes, remaining limitations and hosted-CI results only when
actually observed.
