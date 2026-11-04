# Exchange mask ownership and native dependency upgrades

Date: 2026-10-19. Branch: `feat/veris-seaice`.

Status: approved and implemented 2026-10-19; full native validation and independent review complete.

## Intent and acceptance

Use released Veris `v0.2.1` and JCM `3.0.0` for both development and
acceptance runs. Remove legacy exchange dispatch and make component, topology,
runtime, and numerical responsibilities unambiguous. Reuse the existing
component declarations, topology maps, immutable runtime state, and regridders.
Do not introduce another mask framework or runner.

Every bundled route declares its physical area convention. Coverage enters
source products before remapping, once. Scalars and vectors follow the same
quantity rules. Static geometry, evolving coverage, inactive NaNs, and the
coverage of stored imports retain their distinct meanings under host and JAX
execution. Formatting, lint, typing, focused/fast/full tests, native coupled
runs, documentation, and a Git commit are required for delivery.

## Existing behavior and alternatives

The explicit quantity path already implements most of the required numerical
operation. The remaining ambiguity comes from a second legacy dispatcher,
topology-owned fractional masks, and static setups that depend on those masks.
Extending that split would preserve duplicate ownership. A new generic mask
registry would add an unnecessary parallel abstraction. The selected approach
finishes the existing explicit path and removes the obsolete one.

The ordinary JCM consumer currently adds incoming land and sea temperatures.
That assumes area contributions, whereas native JCM exposes separate land
temperature and SST inputs. Merely replacing the route declaration with
`intensive` would overcount coastal temperatures. Likewise, bundled land
components currently discard their computed fractional coverage and depend on
the surface topology policy to restore it later.

## Ownership and public contracts

1. `RectilinearGrid.binary_mask` remains immutable computational geometry.
   `ComponentMaskSpec` remains the sole declaration of component coverage.
   Its existing optional binary field may restrict support within that geometry;
   its fractional field is current, finite, dimensionless, and zero outside
   component support. Binary transitions are discrete; continuous fractions
   remain differentiable.
2. Topology creates regridders, validates compatible surface geometry, and may
   restrict target route support with a binary patch. A policy cannot expand
   support beyond target geometry. `ExchangeTopologyPatch.fractional_masks`
   and private topology fractional maps are removed. `SurfaceMaskPolicy` retains
   geometry/land-support validation and necessary binary restrictions, without
   assigning coverage or changing numerical quantities.
3. `_runtime/component_masks.py` resolves component declarations.
   `_runtime/exchange_masks.py` owns source/target coverage, remapped overlap,
   current route diagnostics, and route validation support. Existing validation
   boundaries consume these decisions instead of inventing independent mask
   rules.
4. `Exchange.quantity` becomes a required keyword: `intensive`, `source_flux`,
   or `target_flux`. There is no implicit or explicit `legacy` mode.
   `source_coverage="component"` remains the default;
   `source_coverage="binary"` explicitly identifies preweighted source fields.
5. `_runtime/exchange_dispatch.py` has one scalar/vector dispatch path using
   the same source weighting, target normalization, and finite-value checks.
   Runtime context is required; there is no context-free legacy fallback.
6. `RunState` retains current route coverage separately from the coverage saved
   with its last dispatched imports. Refreshing current coverage never changes
   stored-import validation support. Public field replacement keeps outgoing
   copies coherent as it does today.

These are intentional contract changes: callers must declare quantity, and
topology policies must move fractional coverage to their owning components.
Public owner/export boundaries otherwise remain unchanged.

## Numerical contract

Let `s` be current source fractional coverage, or source binary support for an
explicitly preweighted field. Let `b` be current target binary support
intersected with the static binary route restriction, `t` target fractional
coverage on that support, and `R` the selected linear remapper. Compute
`d = R(s)` and `n = R(where(s > 0, value, 0) * s)`.

| Quantity | Delivered value where `b > 0` and `d > 0` | Meaning |
| --- | --- | --- |
| `intensive` | `n / d` | Mean over the source-covered overlap |
| `source_flux` | `n` | Source-area contribution per target cell area |
| `target_flux` | `(n / d) * t` | Overlap mean converted to target-covered contribution |

Return finite zero outside active route support. Intensive forcing remains
available on supported targets with zero fractional coverage. Zero source
coverage produces finite zero values and derivatives. Preweighted fields use
binary source coverage so component area is not applied twice.

Neutralize inactive NaNs before multiplication, vector rotation, remapping,
and division. Preserve the existing rejection of infinity everywhere and
active NaNs at author, send, dispatch, and supplied-state boundaries. Both
vector components receive identical weighting and normalization; rotation
remains the regridder's responsibility.

Physical numerators and denominators are not clipped or renormalized to make
tests pass. Diagnostic coverage may retain the documented bounded-roundoff
handling, independently of physical values. Conservative integral claims apply
to conservative remappers; bilinear interpolation does not become conservative
by selecting `source_flux`.

## Bundled setup migration

Audit producers and consumers together, splitting routes when fields have
different semantics. Record the final field-by-field choices in user-facing
documentation and executable setup contracts.

- Atmospheric wind, thermodynamic state, radiation/precipitation forcing per
  local surface area, ocean currents, SST, land temperature, soil moisture,
  albedo, and other local state variables use intensive transfer to native
  surface-specific consumers. Target-area contributions use `target_flux`
  only where the consumer actually expects that area convention.
- Native Veris ocean heat, freshwater/salt, and momentum contributions already
  include ice/open-water weighting. Preserve `source_flux` with binary source
  coverage. Preserve the verified shared-grid stress vector contract and
  explicit native face conversion.
- Existing coupled JCM marine temperature/albedo/ice products use their
  established source-area products before conservative remapping; retain
  binary source coverage for those preweighted products. Land inputs remain
  independent intensive surface values.
- Ordinary JCM receives independent intensive land temperature, SST, and soil
  moisture and passes them to the corresponding native forcing channels.
  Its total surface-temperature diagnostic uses physical land/ocean fractions.
  Use fractional native terrain consistently in both JCM modes; remove the old
  temperature-sum cleanup assumption. Released JCM 3 publishes only delivered
  grid-mean turbulent fluxes; its optional per-tile contract fields are absent.
  Remove unused flux routes into prescribed JCM land and unavailable land-specific
  exports. The slab demonstration retains explicit grid-mean forcing as an
  acknowledged approximation; never reconstruct a land flux by dividing a
  combined land/ocean flux by land fraction.
- CAMulator's combined skin-temperature input requires area contributions.
  Form land/ocean contributions at their sources and remap before combining,
  rather than multiplying separately remapped temperature and coverage.
  Split any local state fields from such contribution routes.
- JCM/CAMulator land components retain their already-computed land fraction
  as a declared current coverage field through `ComponentMaskSpec`.

Static current mask fields must coexist with daily/monthly forcing records.
Allow non-current component transfer policies with mask declarations, and make
the shared runtime selector recognize declared mask fields as current. Apply
that same selector rule in sends and default runtime output. Keep exact
grid-shape/dtype validation for masks; do not infer time axes from heuristics.

Intentional numerical changes include removal of post-remap scalar mask
multiplication, source masking/weighting for vectors, physically normalized
coastal intensive states, consistent native fractional terrain, and any
contribution routes switched to conservative remapping. Document each affected
setup rather than silently preserving legacy expected numbers.

## Dependency contracts

GitHub CLI resolves annotated Veris tag `v0.2.1` to commit
`6e3b56ee14922f4bae760d57772f610078aa1886` (tag object
`a964f002f3f0cee91bfb8e9edd62e8b0a0066bf5`). Use the release in package
metadata and verify installed version/provenance and CI provenance against the
same commit. Replace active references to the earlier native development pin;
preserve explicitly historical validation evidence.

Pin the JCM extra to `jcm==3.0.0` and raise its Dinosaur floor to `>=1.5.0`.
JCM's published metadata requires JAX `>=0.10,<0.11`, but released Veris
requires `>=0.11.2`. The user explicitly approved a documented JCM upper-bound
override only if full validation passes. Combined validation uses JAX `0.11.2`,
retains every other native dependency requirement, and documents the unresolved
upstream metadata conflict. Resolve all other requirements in one compatible
environment rather than suppressing other dependency conflicts. Use a conda environment named
`scipy`, with the established isolated Python 3.12 lane when needed. Keep stock
Veros and the exact autodiff fork in separate processes/environments. Verify
native JCM state, forcing, calendars, diagnostics, output, and coupled surface
hooks against the released API. In particular, JCM 3 requires the exact time
and step in its resumable RunState and uses Gregorian dates rather than its
removed 365-day calendar option. Thread that clock through the immutable
adapter payload, use Gregorian clocks in JCM-backed bundled galleries, and
reject incompatible calendars explicitly. Its public PhysicsState humidity is
already kg/kg: remove the adapter's obsolete division by 1000 while retaining
conversions for diagnostics that remain in native SPEEDY gram units. Test
leap-day/chunk-continuation and independent humidity-unit cases. These are
required release-API migrations; no scientific recalibration or native
model-physics modifications are part of this refactor.

## Validation and delivery

Write failing regressions before behavior changes. Extend the existing test
files rather than building a second harness. Cover fractional mismatched
coasts, zero/evolving coverage and reactivation, source binary coverage,
topology restriction/expansion rejection, scalar/vector parity with rotation,
inactive NaNs/infinity, conservation using independent cell-area budgets,
stored imports across unequal component schedules, and supplied-state checks.
Exercise JIT, JVP, reverse VJP, and finite differences away from discrete mask
transitions. Test static coverage alongside daily and linear forcing in sends
and period output.

Run targeted regressions and the fast stock suite during development. Before
committing, run Black, strict flake8, mypy, compileall, full stock pytest and
the separate exact-fork AD suite. Complete coverage, strict Sphinx, distribution
and installed-package checks, and actual short native JCM/Veris/Veros coupled
runs using the new dependencies. Record exact versions, commands, outcomes,
known limitations, and failed approaches in the task evidence and
`PROGRESS.md`; update `DESIGN.md`, `DEPENDENCIES.md`, exchange/topology docs,
examples, CI and executable API/setup contracts. Use GitHub CLI for remote
inspection and commit the completed work on `feat/veris-seaice` after the
required suite passes. Pushing, merging, tagging, and release publication are
not requested.
