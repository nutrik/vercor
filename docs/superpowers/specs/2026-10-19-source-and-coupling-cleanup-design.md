# Source cleanup and coupling field contracts

Date: 2026-10-19. Branch: `feat/veris-seaice`.

Status: written specification approved on 2026-10-19; implementation plan review follows.
Baseline: `e1d417af9baf841edc8d9a44d498be8caa2aeba2`.

## Intent and acceptance

Require stock Veros `>=1.7.0`, remove obsolete compatibility and duplicate
surfaces across the source tree, correct demonstrated coupling field defects,
and consolidate tests that add no distinct guarantee. Preserve the physical,
numerical, ownership, and autodiff contracts of the remaining implementation.
Follow `AGENTS.md`; develop and commit on `feat/veris-seaice` using GitHub CLI
for GitHub inspection. The existing local baseline commit remains intact.

Acceptance covers the complete maintained source, tests, gallery, dependency
metadata, and active documentation. Each removal must have an actual caller
inventory and a surviving owner; each behavior correction must have a failing
regression before implementation. The result includes an audit record of
removed and retained mechanisms, test consolidation with surviving assertions,
all required checks, and focused Git commits. Publication is a separate action.

## Architecture and alternatives

Keep the existing structural component protocol, immutable runtime state,
prepared coupling, explicit exchange quantities, topology policy, native model
boundaries, and shared output coordinator. Simplify within these owners. A new
field registry or coupling framework would duplicate their responsibilities;
changing only the dependency would leave demonstrated sign defects unresolved.

`CallableComponent` retains its documented one-, two-, and three-argument
callable convenience. Structural `Component.step` follows the full protocol;
the runtime binds that method directly rather than rediscovering private
callable normalization. Native host copies, JAX payload structure checks, lazy
optional imports, output providers, and the JCM forcing extensions remain
because they enforce current execution or physical behavior.

No replacement aliases, fallback preparation routes, runtime unit parser,
generic plugin registry, warning suppression, or parallel field store is added.
Physics values remain traced, precision remains static, and inactive operands
are neutralized before nonlinear arithmetic. Numerical tolerances are derived
from precision or independent physical checks; no empirical correction factors
are introduced to make tests pass.

## Dependency and supported native boundaries

Change the stock extra to exactly `veros = ["veros>=1.7.0"]`. Keep the
`veros_ad` requirement at immutable revision
`7a8c964cf00b5aa0713c995edd760643a431b3c9` and keep its capability checks.
Stock and fork validation run in separate Python processes. Update active
requirements, diagnostics, CI contracts, and documentation without rewriting
historical validation results or archived plans.

The native solver cache and state-copy boundaries carry real ownership
requirements. Preserve them and validate against actual stock Veros 1.7.0;
replace the obsolete `<1.7` diagnostic with a capability-specific error.
Veros 1.7 changes forcing interpolation and native numerical behavior, so
old numerical snapshots cannot be silently regenerated to conceal differences.
Use independent budgets, units, signs, and analytic expectations to distinguish
intended upstream changes from regressions.

## Removal inventory

The audit inventoried 179 Python source modules and 102 test modules. Apply
these removals after checking their callers against the implementation baseline:

| Surface | Final owner or behavior |
| --- | --- |
| `Coupler._initialize_runtime` test-only forwarder | Public lifecycle tests use `initial_state` or `run`; preparation unit tests use the existing private preparation owner. |
| Optional contract rebuilding in `_runtime/coupler_state.py` | Require the contracts already owned by `PreparedCoupling`; remove its unused exchange argument. |
| `build_exchange_contracts(validate_endpoints=...)` and endpoint-skipping branch | Always validate source and target names before building contracts. |
| `check_not_empty_import_export_lists` and its guarded call | Remove the unreachable check; retain declaration and required-field validation. |
| Preseeded `topology_maps` arguments across initialization and topology | Build routes through their declared factories once, validate capabilities, then apply the topology policy. |
| `prepare_runtime_state(validate_state=...)` bypass | Always validate supplied states before refreshing masks. |
| Raw-component fallback in topology context construction | Receive the declared private bindings and expose their author components directly. |
| Mutable setup `_state` fallback in JCM snapshot writing | Require the final runtime payload; missing payload is a component-scoped error. |
| Defaults for guaranteed Veros and JCM setup-state attributes | Read constructor-owned attributes directly; complete test fixtures. |
| Duplicate private bilinear and conservative factories | Public `vercor.regridding` factories construct private concrete implementations directly, preserving public signatures and `radius_km`. |
| `flatten_exchange_fields` forwarding helper | Use the existing `_flatten_field_items` owner. |
| Test-only `tree_as_real_dtype`, `tree_unwrap_leading_dims`, and `tree_stack` | Retain production `tree_as_runtime_dtype` and `tree_mean`; preserve integer, complex, and category dtype guarantees there. |
| Test-only `canonicalize_surface_field` | Use the explicit canonical layout functions. |
| Test-only `initialize_camulator_runtime_fields` | Component preparation owns field storage and initialization. |
| Test-only `prepare_veris_forcing` forwarder | Use the production forcing pack and preserve native forcing, conservation, and gradient assertions. |
| Unused JCM default-parameter and prediction-output convenience wrappers | Retain the current gallery construction and provider/snapshot extraction owners. |
| Public `forcing_index.ForcingYearType` alias | Use canonical `calendar.YearType`; update the public inventory and migration documentation. |

Remove additional wrappers only when the same caller and ownership evidence
demonstrates that they add no value. Record necessary retentions rather than
removing mechanisms solely because they contain `getattr` or the word adapter.
Update `DEPENDENCIES.md`, public manifests, relevant signatures, examples, and
active architecture documentation with the final ownership decisions.

## Canonical field meaning

The existing dispatcher is the sole area-weighting owner. For source coverage
`s`, target binary support `b`, target fractional coverage `t`, and remapper
`R`, form `N = R(where(s > 0, x, 0) * s)` and `d = R(s)`. Receive `N/d` for
`intensive`, `N*b` for `source_flux`, and `(N/d)*t` for `target_flux`, using
finite zero values and derivatives for empty overlap. Weight both vector
components before remapping and rotation. Preweighted Veris feedback,
concentration, and open-water temperature products use binary source coverage.

Static support, evolving fractions, current route diagnostics, and saved
last-dispatch import support retain separate ownership. Mask fields stay
finite and bounded; active missing data and all infinities fail validation.
Keep existing endpoint-roundoff reconciliation bounded by machine precision.

| Fields | Canonical units and sign | Native boundary |
| --- | --- | --- |
| Air, land, ocean, and ice temperatures | K | Add 273.15 once for native Veros Celsius; retain separate land and marine temperatures. |
| Air specific humidity | kg water vapor/kg moist air | JCM 3 already uses kg/kg; do not apply its old gram conversion. Total water is not silently treated as vapor. |
| Air pressure and density | Pa and kg/m3 | Use the pressure and temperature at the same model level. |
| Air reference height and SSH | m | Air fields retain their colocated height; native streamfunction is not SSH. |
| Wind and ocean currents | m/s | Preserve vector orientation and explicit east/north versus west/south face conversion. |
| Rain and water-equivalent snow | kg/m2/s, positive downward | Aggregate ERA5 phases; convert JCM g/m2/s once before remapping. |
| Heat and radiation | W/m2, positive downward into the surface | Negate native upward turbulent heat once; distinguish net shortwave from downward shortwave and longwave. |
| Bulk evaporation mass | kg/m2/s, negative for water loss | Preserve this sign through Veris and virtual salt conversion. |
| Salinity | g/kg | This native numerical convention is an explicit exception to the SI field policy. |
| Actual salt mass flux | kg salt/m2/s, positive into ocean | Keep distinct from virtual salinity forcing. |
| Virtual salinity forcing | m (g/kg)/s | Use `(-S_ref*M + 1000*J)/rho_fw`; Veros owns timestep and depth integration. |
| Stress | Pa, component direction on the ocean | Combine ice/open-water contributions once and retain native face support. |
| Coverage, albedo, soil availability | dimensionless | Coverage and albedo have their respective bounds; local fields and area products are distinct declarations. |

Document producers, consumers, conversions, and route quantities alongside
these contracts. Prescribed components must not advertise physical feedback
through unused imports. Preserve intentional diagnostic display of received
fields in the data gallery and the general `DataComponent` forwarding contract.
Prune unused slab and prescribed-model inputs and adapt their recipes and
galleries where imports currently imply physics that is never evaluated.

## Demonstrated physics corrections

For slab ocean, use
`T_new = T + dt * ((SHF + LHF)/(rho*cp*H) - (T - T_ref)/tau)`.
Keep the existing physical coefficients and document their units. Positive
downward heat warms the slab; warm and cold unforced departures contract for
a stable timestep. Verify the derivative `1 - dt/tau` independently of the
implementation and retain flux, depth, and timestep sensitivities.

For slab atmosphere, use downward sensible heat
`SHF = 10*(T_air - SST)` W/m2. Keep the toy latent magnitude and its existing
temperature dependence, expressed as `LHF = 0.5*SHF` after the sensible-sign
correction. This is a documented toy closure, not a humidity-based evaporation
model. Cold air above warm water produces negative surface heat fluxes.

For slab land, negative downward latent heat means evaporation and reduces
soil moisture; positive latent heat means condensation and increases it.
Retain the existing toy conversion coefficient and clipping bounds, documenting
their meaning. Add direction tests away from clipping plus endpoint and
derivative checks. Do not describe the toy coefficient as a certified water
inventory budget.

## CAMulator initialization and radiation

Decode the latest initial input frame with CREDIT's `inverse_transform_input`,
not its prediction-output inverse transform. Reuse the existing tensor accessor
and thermodynamic/height kernels to obtain current meteorology without a native
model advance, cursor change, noise resampling, or hidden mutable field cache.
Prediction and bootstrap mapping share the same physical conversion owner.

Add one setup-data declaration, `CAMulatorConfig.initial_diagnostics`, a
defensively owned optional mapping containing `time` and native `TS`, `FSNS`,
and `FLNS` arrays. `time` identifies the effective initial timestamp selected
by the existing runtime cursor, including `forcing_start` alignment.
`TS` is K; radiation is signed accumulated J/m2 over the configured native
forecast step, matching the supported ERA5-scaled CAMulator training contract.
During setup, require these diagnostics when they are unavailable in the native
initial input tensor. Validate grid shape, finite values, and time alignment;
missing diagnostics fail before coupling rather than being guessed from a
reference atmosphere or replaced with zero forcing. The mapping supplies
initialization data only; `SetupResult` and the existing runtime payload remain
the sole field/state owners. The gallery must show how its initial diagnostics
come from the corresponding native initial-condition data at the selected start.

Native `Qtot` names total water. The atmosphere bulk contract requires vapor
specific humidity. Add `CAMulatorConfig.water_vapor_variable`, an optional
native channel name whose declaration promises vapor specific humidity in
kg/kg. Require that channel in both initial input and prediction layouts;
the known total-water channel `Qtot` cannot satisfy it. A missing declaration,
unavailable channel, or `Qtot` selection produces a clear setup error;
do not invent a saturation adjustment to recover an unpredicted species.
Document this requirement and update the CAMulator gallery to supply a
compatible configured channel explicitly. A checkpoint without that channel
cannot be certified as a drop-in bulk-forcing atmosphere.

Convert accumulated radiation with
`native_step_seconds = 3600 * conf['data']['lead_time_periods']`.
Keep the existing signed net-longwave to downward-longwave conversion,
including surface emission, with one duration conversion. Reject invalid
durations before arithmetic. Test at least 1-, 3-, and 6-hour native intervals
with fixed independent rate expectations and exercise multiple native
substeps per coupling interval. Support the existing accumulated-energy
training contract; do not infer unmarked W/m2 data or add a unit-guessing path.

The upstream contracts are visible at CREDIT revision
`bc078d7a39300fb8c8fe867a22ec218a5e02da11`:
[input transformation](https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/credit/transforms/transforms_global.py),
[native energy conversion](https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/credit/postblock/gen1.py),
[flux units and signs](https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/credit/postblock/conservation.py),
and [CAMulator channels and timing](https://github.com/NCAR/miles-credit/blob/bc078d7a39300fb8c8fe867a22ec218a5e02da11/climate/camulator_config.yml).
The local environment lacks CREDIT and native checkpoint data. Real native
execution remains a separately identified validation limit; stand-in boundary
tests cannot be reported as native climate validation.

## Interchangeability and test consolidation

A supported swap preserves field meaning, units, layout, support, execution
capability, and initialization. Matching string names alone is insufficient.
Exercise the same receiving closure with prescribed data, callable JAX,
structural JAX, and host components supplying identical physical inputs.
Include fractional coasts, different scalar grids, vector orientation, evolving
and empty coverage, and inactive NaNs. Compare conserved source products and
physical tendencies, not only output shapes. Preserve public component names
and assembly contracts so swaps do not depend on private model classes.

Retain real JCM/ERA5/Veros/Veris producer and consumer conversions, prescribed
versus prognostic ocean surfaces, shared-grid Veris face geometry, multi-step
feedback, JIT, JVP/VJP agreement, and finite differences. Unsupported native
geometries, missing required fields, or unavailable physical species fail at
preparation. Do not claim arbitrary native grids or every bundled model can be
exchanged without satisfying these contracts.

Remove redundant `donate_state` absence checks, root/component export smoke
checks already covered by canonical manifests, duplicate identical-grid flag
checks, dtype agreement checks already covered for both precisions, and a
RunState PyTree smoke already covered by full round-trip tests. Combine the
duplicate run-order and foreign-state test bodies while retaining every input
and corruption case. Keep one canonical dependency metadata assertion.

When touched, replace source-spelling checks for removed private wrappers with
the relevant behavior or existing single-owner check. Retain distinct lazy
imports, import-cycle restrictions, public manifests/signatures, host ownership,
solver-cache restoration, native output, conservation, AD, installed wheel,
sdist, external-extension runtime/typing, and CLI behavior. Test deletion
requires an explicit surviving-assertion map; coverage percentage alone is not
proof of redundancy. Do not add timeouts or skip newly failing cases.

## Verification and delivery

Use the isolated Python 3.12 conda environment named `scipy`, writable
`VEROS_ASSET_DIR` and caches, and `/private/tmp` basetemps/artifacts. The stock
acceptance environment must actually install Veros 1.7.0 or newer; preserve the
separate exact-fork process and provenance checks. Keep the existing reviewed
JCM 3.0.0/JAX 0.11.2 installer exception narrow and unchanged.

Run focused regressions after each change and `--fast` during development.
Before implementation commits, run full stock tests, isolated native AD,
combined coverage with the current 90% gate, Black, strict maintained-source
flake8 plus the requested repository lint report, mypy, compileall, whitespace,
strict Sphinx, wheel/sdist metadata checks, and installed-package/extension
smokes. Run a short real coupled JCM/Veros/Veris gallery from the final built
artifact and inspect its saved fields and native step count. Broaden checks
only when a new failure, change, or unresolved concern justifies them.

Record exact commands, results, failed approaches, scientific limits, removal
decisions, and the test consolidation map in active documentation and
`PROGRESS.md`. Keep progress within its existing 180-line bound. Stage intended
paths only, keep refactoring and behavior corrections in separate logical
commits, and verify the resulting branch and commit history. Do not claim the
full objective complete before every numbered user requirement has current
evidence. Commit the written specification after the full existing unit suite
and fast gate pass; implementation planning follows its review.
