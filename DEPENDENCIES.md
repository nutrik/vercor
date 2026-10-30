# VerCOR Module Dependency Order

The numbered entries below are topological import layers for the current source
tree. A module may import modules in its own layer only where an explicit
strongly connected pair is documented; otherwise production imports point to an
earlier layer. Imports guarded only by `TYPE_CHECKING` are annotation edges and
are not runtime-order constraints. Public owner facades are listed separately
from their concrete private implementations.

Layer 14's `vercor/output/_session.py` owns backend-independent provider
selection and accumulation. It uses layer 1's canonical int32 index conversion
for numeric sampling on both host and JAX runs; ordinary providers retain
concrete calendar contexts. Layer 18's execution coordinator selects scan
acceleration only when the runtime and every enabled provider support it.
This adds no layer or dependency cycle.

Conservative remapping computes destination longitude widths with the same
degree-difference/radian-conversion arithmetic as its overlap weights. The
ice-area bounds repair changes no import or topological layer.
The longitude-wrap repair standardizes source/destination bounds in that same
layer-2 remapper before overlaps and areas, retaining native cell order and the
existing identity bypass. No module or dependency layer is added.

Bilinear layer-1 geometry owns canonical degree/radian longitude phases.
Layer-2 weights compare source candidates in each target's whole-turn frame
during construction and return unchanged source-column indices; scalar/vector fallback shares one orientation
boundary in layer 3. No runtime state, public interface, or layer is added.

Layer 18's execution coordinator selects an exact common scan block size per schedule;
the existing backend owner supplies shared numeric chunk inputs to the output
scan owner. Bilinear extrapolation and Veris setup retain compiled kernels at
their existing layers. Scanned coordinates receive strong numeric dtypes in
the output session. No new module, public protocol or dependency is added.

The 2026-07-20 simplification removed the one-caller
`vercor/_runtime/component_topology.py` and one-use
`vercor/_runtime/runner.py` modules. Surface-role lookup now belongs to
`surface_masks.py`, while `facade.py` directly constructs the run context and
calls the execution owner. Those deleted modules are removed from the order;
the remaining topological layers stay stable.

Release-only tooling is outside the runtime graph:
`tools/validate_release_state.py` is a standard-library-only foundation
consumed by `docs/releasing.md` and `tests/test_release_state_validator.py`.

`vercor/time_selection.py` owns host-side interpolation of centered periodic
records; runtime time metadata consumes it without changing the import layers.

The Veros adapter has two capability-gated branches. Layer 1 owns lazy runtime
configuration plus the stock-compatible global forcing definition; layer 2's
`veros_setup_jax.py` depends on that definition and contains only inherited JAX
setup adaptations. Higher setup-state and factory layers select between them
after the requested runtime is configured and, for JAX, its capabilities are
verified.

Freshwater coupling preserves these layers: the field/recipe foundations
declare rain and snow, layer 4's JCM field mapper normalizes its diagnostics,
and layer 5's Veros flux owner consumes the layer 2 traced physical constants.
The global setup's native salt-flux buffer and restoring coefficient remain
in layer 1; runtime forcing preparation in layers 3 and 15 fills the buffer.
ERA5 aggregation and lifecycle loading stay in its existing layer 15 owner.

1. `vercor/_field_names.py`, `vercor/_interpolators/_bilinear_geometry.py`, `vercor/_logging/__init__.py`, `vercor/_logging/config.py`, `vercor/_pytree.py`, `vercor/_regridders/__init__.py`, `vercor/_run_order.py`, `vercor/_runtime/__init__.py`, `vercor/_runtime/interrupts.py`, `vercor/_setup_runner.py`, `vercor/calendar.py`, `vercor/dtypes.py`, `vercor/exceptions.py`, `vercor/fields.py`, `vercor/setups/_data/__init__.py`, `vercor/setups/_external/__init__.py`, `vercor/setups/_external/camulator_contracts.py`, `vercor/setups/_external/camulator_imports.py`, `vercor/setups/_external/camulator_runtime_settings.py`, `vercor/setups/_external/veros_runtime_settings.py`, `vercor/setups/_external/veros_setup.py`, `vercor/setups/_lazy_imports.py`, `vercor/setups/_time_helpers.py`, and `vercor/types.py` - independent foundations: public field/type/dtype/calendar/error contracts, with `vercor.calendar` as the canonical year-type and model-year-duration owner; shared private text-safe name-sequence freezing, private tuple/PyTree/logging/runtime package bases, optional-adapter configuration/import boundaries, and the standard-library-only private setup runner.
2. `vercor/_numerical_safety.py`, `vercor/_host_arrays.py`, `vercor/_interpolators/_bilinear_extrapolation.py`, `vercor/_interpolators/_bilinear_weights.py`, `vercor/_interpolators/conservative_remap_rectilinear.py`, `vercor/_logging/protocols.py`, `vercor/_runtime/stores.py`, `vercor/_runtime/topology_state.py`, `vercor/assets.py`, `vercor/clock.py`, `vercor/field_layout.py`, `vercor/forcing_data.py`, `vercor/forcing_index.py`, `vercor/grids.py`, `vercor/physics.py`, `vercor/recipes.py`, `vercor/setups/_external/_jax_gcm_pytree.py`, `vercor/setups/_external/jax_gcm_tools.py`, `vercor/setups/_external/veros_setup_jax.py`, and `vercor/time_selection.py` - first consumers of the foundations, including transform-safe numerical validation/masking, private host/PyTree/interpolation mechanics, inherited differentiable-Veros setup adaptations, JCM 3 coordinate/terrain/`ForcingData` loading, and the canonical public `Clock`, `RectilinearGrid`, and `PhysicalConstants` owners.
3. `vercor/_interpolators/bilinear_rectilinear.py`, `vercor/_logging/host.py`, `vercor/_runtime/component_masks.py`, `vercor/_runtime/state.py`, `vercor/_runtime/time.py`, `vercor/components/setup_validation.py`, `vercor/fluxes/utilities.py`, `vercor/fluxes/vertical_coordinates.py`, `vercor/grid_geometry.py`, `vercor/setups/_data/_field_helpers.py`, `vercor/setups/_data/assets.py`, `vercor/setups/_external/camulator_tensors.py`, `vercor/setups/_external/veros_data.py`, and `vercor/setups/_external/veros_state.py`, `vercor/setups/_external/jax_gcm_surface.py` - private numerical/runtime implementations and public grid-geometry/flux utilities built from layers 1-2; `_runtime/time.py` consumes the calendar owner to derive year type and duration for each timestamp.
4. `vercor/_interpolators/__init__.py`, `vercor/_logging/callback.py`, `vercor/_regridders/base.py`, `vercor/fluxes/bulk_formula_cesm.py`, `vercor/setups/_external/camulator_fields.py`, and `vercor/setups/_external/jax_gcm_fields.py` - concrete private interpolation, logging, regridder, flux, and adapter-field implementation owners. CAMulator fields owns the sole pure traced meteorology mapper and the host latest-input bootstrap; setup-resource and cursor annotations use TYPE_CHECKING imports to avoid runtime cycles.
5. `vercor/_regridders/bilinear.py`, `vercor/_regridders/conservative.py`, `vercor/fluxes/__init__.py`, `vercor/jax_logging.py`, and `vercor/setups/_surface_fluxes.py` - concrete private regridders precede the later public regridding facade; `vercor.fluxes` and `vercor.jax_logging` are the public owners over their focused implementations.
6. `vercor/_runtime/progress.py`, `vercor/components/contexts.py`, `vercor/regridding.py`, `vercor/setups/_external/_camulator_wind_filtering.py`, and `vercor/setups/_external/camulator_forcing.py`, `vercor/setups/_external/veros_fluxes.py`, and `vercor/setups/_external/veris_forcing.py` - public setup/step contexts and `Regridder`/`VectorRegridder`/`RegridderFactory` plus `bilinear(...)`/`conservative(...)` facade, built after the concrete private regridders; private runtime progress, wind-filter mechanics, and the CAMulator forcing cursor follow their logging dependencies and remain non-public.
7. `vercor/setups/_external/veris_fluxes.py`, `vercor/exchanges.py`, `vercor/grid_masks.py`, and `vercor/setups/_external/camulator_wind_filter.py` - canonical public `Exchange` owner, public mask utilities, and the adapter wind-filter facade. Exchange field aliases remain owned solely by `vercor.fields`; this layer consumes them.
8. `vercor/_runtime/exchange_masks.py`, `vercor/_runtime/contracts.py` and `vercor/setups/_external/camulator_stepper.py` - private exchange-contract and CAMulator stepping implementations. Runtime consumers use `Exchange.route_id` directly as the sole route identity.
9. `vercor/_runtime/dispatch_context.py`, `vercor/_runtime/exchange_topology.py`, `vercor/_runtime/field_transfer.py`, and `vercor/state.py` - private dispatch/topology implementations; `field_transfer.py` is the shared transfer-policy selection owner; and the canonical public `RunState` and `ComponentState` owner.
10. `vercor/_runtime/exchange_dispatch.py`, `vercor/components/_protocol.py`, and `vercor/diagnostics/fields.py` - destination exchange dispatch, the dependency-light structural component protocol, and state-backed diagnostic fields. The protocol is internally owned but reports `vercor.components.contracts` as its canonical public module; one import-finalization binding resolves its `ComponentSpec` forward annotation after declarations load.
11. `vercor/output/__init__.py` - stable output contracts (`OutputProvider`, frames, specs, targets, variables, cadence, and snapshot contexts) depend only on the component protocol, not the full component declaration module. The former component/output runtime import cycle is removed.
12. `vercor/components/contracts.py`, `vercor/components/_callable_wrappers.py`, `vercor/components/_contracts.py`, `vercor/components/data.py`, `vercor/diagnostics/plotting.py`, `vercor/diagnostics/tables.py`, `vercor/output/_netcdf.py`, `vercor/output/_period.py`, `vercor/setups/config.py`, and `vercor/setups/_output.py` - component declarations and normalization helpers, `DataComponent`, diagnostics, shared output math/file primitives, frozen setup-specific configuration, and private optional-output normalization.
13. `vercor/_runtime/validation.py`, `vercor/components/_adapter.py`, `vercor/components/_runtime_fields.py`, `vercor/components/base.py`, `vercor/diagnostics/__init__.py`, `vercor/output/_dataset.py`, and `vercor/output/_runtime.py` - private validation/binding/output implementations, the public `CallableComponent` implementation, the diagnostics facade, and final runtime-view/snapshot writing.
14. `vercor/_runtime/component_state.py`, `vercor/_runtime/driver.py`, `vercor/_runtime/state_validation.py`, `vercor/components/__init__.py`, `vercor/components/runtime_execution.py`, `vercor/output/_session.py`, the strongly connected pair `vercor/_runtime/surface_masks.py` plus `vercor/topology.py`, `vercor/setups/_data/_component_helpers.py`, `vercor/setups/_data/jcm_land.py`, `vercor/setups/_external/camulator_land.py`, `vercor/setups/_external/camulator_output.py`, `vercor/setups/_external/jax_gcm_output.py`, `vercor/setups/_external/veros_output.py`, `vercor/setups/_external/veris_output.py`, `vercor/setups/_slab/atmosphere.py`, `vercor/setups/_slab/land.py`, `vercor/setups/_slab/ocean.py`, and `vercor/setups/_slab/seaice.py`, `vercor/setups/_external/veros_runtime.py` - runtime state/driver validation, the public component facade, the sole immutable period-output plan/session, topology policy, bundled native providers, data helpers, and slab components. JCM land canonicalizes static and time-first JCM 3 forcing layouts, while native JCM output flattens mapping diagnostics and resolves packaged SPEEDY units. `vercor/output/_session.py` consumes the shared selection owner with run-precomputed `RuntimeStepInfo`; both imports remain from this layer to earlier layers. There is no component-output adapter or second period-file lifecycle.
15. `vercor/_runtime/coupler_state.py`, `vercor/_runtime/topology_policy.py`, `vercor/runtime/__init__.py`, `vercor/setups/_data/era5_atmosphere.py`, `vercor/setups/_data/era5_land.py`, `vercor/setups/_data/era5_ocean.py`, `vercor/setups/_data/erainterim_ocean.py`, `vercor/setups/_data/veros_ocean.py`, `vercor/setups/_external/camulator_init.py`, `vercor/setups/_external/camulator_runtime.py`, `vercor/setups/_external/jax_gcm_runtime.py`, `vercor/setups/_external/veros_gcm_state.py`, `vercor/setups/_external/veris_runtime.py`, and `vercor/setups/_slab/__init__.py` - public `RuntimeOptions`/execution-extension owner after its clock, state, logging, and topology dependencies; private runtime topology/state and concrete setup runtimes. The JCM runtime consumes reduced JCM 3 dynamics plus `_surface_flux` (including `t0`), `_shortwave_rad`, `_condensation`, and `_convection` diagnostics, passes traced phase thresholds to native-grid rain/snow diagnosis, and functionally replaces the immutable payload.
16. `vercor/_runtime/run_context.py`, `vercor/_runtime/topology.py`, `vercor/setups/_external/camulator_gcm_state.py`, `vercor/setups/_external/jax_gcm_state.py`, and `vercor/setups/_external/veros_gcm.py`, `vercor/setups/_external/veris.py`, and `vercor/setups/_external/veris_parameters.py` - frozen run inputs, topology orchestration, and external setup-state/factory implementations. JCM setup owns bootstrap extraction plus carry-aware `run_from_state_with_carry` resumption of native dycore and physics state.
17. `vercor/_runtime/backends.py`, `vercor/_runtime/initialization.py`, `vercor/setups/_external/camulator.py`, and `vercor/setups/_external/jax_gcm.py`, and `vercor/setups/_veris.py` - run-wide clock metadata, schedule-cached JIT-scanned executors, host-driver and custom chunk backend execution, one-shot component preparation and topology initialization, and lazily invoked external factories, including the JCM 3 terrain-typed atmosphere boundary. Supplied, pre-driver, and backend-returned states, together with per-component driver results, are checked against the private prepared binding for exact component/store/route names, grid type/name/coordinates/edges/masks, array shapes and dtypes, and finite binary/fractional mask constraints.
18. `vercor/_runtime/execution.py`, `vercor/_runtime/prepared.py`, and `vercor/setups/_jcm.py` - validated workflow planning/chunk coordination, one frozen private prepared-coupling boundary, and paired JCM 3 forcing/terrain setup ownership. The execution coordinator selects backends only from scheduled components and retains core ownership of output and cancellation boundaries. Preparation retains private frozen component bindings without a reflective configuration snapshot or public prepared graph.
19. `vercor/_runtime/preparation.py` and `vercor/setups/__init__.py` - runtime-state creation/validation and the sole public lazy setup export table, after all eagerly imported slab/JCM factories.
20. `vercor/_runtime/facade.py` - high-level private orchestration for preparation reuse, interrupt-scoped construction and execution of validated workflow plans, explicit output-target validation, final-field output, and snapshots.
21. `vercor/coupler.py` - sole canonical public `Coupler` owner, placed after its private runtime facade. Constructor assembly owns immutable collection/declaration snapshots while public `components` retains the original author objects; those objects are treated as immutable configuration. `initial_state(*, prefill_missing=True)` and `run(state=None, *, output=None)` are the only runtime boundaries; `run_order=()` performs setup/state preparation without advancing a component.
22. `vercor/__init__.py` - final primary convenience facade exporting exactly `Clock`, `Coupler`, `Exchange`, `RectilinearGrid`, `RunState`, and `RuntimeOptions`, in that order.
23. `vercor/setups/gallery/` and `vercor/cli.py` - packaged runnable setup scripts depend on runtime and dtype layers; the installed `vercor` command depends on the gallery resource plus the private runner subprocess boundary. The gallery owns model-specific setup construction and translates log level/dtype explicitly; `run_veris_veros_jcm.py` uses only public pair/configuration/recipe/regridding owners and enables dtype before native construction; the CLI performs Click presentation, shared discovery, duplicate rejection, and exclusive copying without overwrite.
24. `pyproject.toml`, `vercor/py.typed`, `.github/workflows/python-package.yml`, and `tests/fixtures/external_extension_test_fixture/` - VerCOR 0.4.0 packaging and verification, PEP 561 typing, wheel/installed-sdist and Python 3.12/3.13 base/JCM/Veros CI, temporary installed-extension verification, and the macOS smoke. The temporary fixture composes frozen configuration/factory, structural regridding, route-keyed topology, workflow/backend, period/snapshot output, and immutable state replacement.
25. `README.md`, `CHANGELOG.md`, `docs/api-architecture-review.md`, `docs/interface-wrapper-audit.md`, and `docs/releasing.md` - runnable public quick starts, exact eight-section architecture review and complete inventory, component/wrapper compatibility and verified boundary findings, current researcher/developer guidance, release history, and candidate verification procedure, built on the implemented layers above.
26. `docs/superpowers/specs/2026-07-13-vercor-0.4-api-design.md`, `docs/superpowers/plans/2026-07-13-vercor-0.4-api.md`, `docs/superpowers/specs/2026-07-17-vercor-0.4-deprecation-cleanup-design.md`, `tests/test_api_architecture_review.py`, `tests/test_distribution_boundaries.py`, and `tests/test_final_review_boundaries.py` - historical `0.4.0a1` architecture and cleanup baseline, executable documentation/manifests/signatures/snippets, installed artifact boundaries, and focused final-review regressions. At that baseline, Tasks 1-8 plus Task 10 formed the alpha series; Task 9 was explicitly skipped and no legacy adapter namespace was present. Historical cleanup verification passed 632 fast tests, 1220 full tests, and 90.49% branch coverage; current stable release-gate evidence is recorded in `PROGRESS.md`.

Independent global Veros restoring is configured in `vercor/setups/config.py`,
passed through the factory and setup-state owners, and applied through
native scalar temperature/salinity weights in `vercor/setups/_external/veros_setup.py`.
`veros_state.py` owns forcing layouts and cleanup, not restoring policy.
The existing import layers are unchanged.

Physical-contract naming (2026-10-21) preserves these layers and module paths.
Layer 12 owns `JCMAtmosphereConfig`; layer 15 owns
`JCMAtmosphereRuntimePayload`; layer 16 owns `JCMAtmosphereSetupState`; and
layer 17 owns the public `make_jcm_atmosphere` factory. Layer 2 recipes use
physical exchange contracts, while layer 7's public mask helpers spell out
ocean/land/atmosphere and binary/fractional meaning. No new module or runtime
dependency is introduced. See `docs/coupling-fields.rst` for the migration.

The prognostic pair in layer 17 reuses layer-16 Veros owner/component assembly
and geometry-only extraction, then binds sea ice on the same grid. The two
packaged Veris galleries depend on this setup facade, recipes and public state
replacement; the calibration objective accesses no private payload owner.

27. `benchmarks/validate_veris.py`, `tests/test_veris_validation.py`, and
    `docs/veris-validation.rst` consume layers 16–23 for host-only scientific
    acceptance: native-area scalar inventories, guarded transition counts,
    shared period output, model-calendar segmentation and timestep comparisons.

Veris current surface extraction stays in layer 15 with its runtime payload;
source-grid ice weighting belongs to the existing exchange dispatcher. Recipes
declare separate per-ice-area properties and binary-supported concentration.

The ocean setup in layer 15 calls layer 14's pure open-water contribution
kernel during initialization; runtime stepping uses the same kernel after
native SST extraction. Imported ice concentration never replaces ocean support.

The JCM marine forcing extension in layer 3 delegates native ForcingData
selection/copying and SPEEDY forcing. Layer 4 composes independently weighted
surface fields; lifecycle/runtime layers install the term and carry arrays.

28. `tools/install_coupled_models.py` reads released JCM wheel metadata and
    resolves all non-JAX native requirements normally. Only JCM 3.0.0's JAX
    upper bound is overridden for the released Veris 0.2.1 combined lane.
    `tests/test_native_dependency_install.py` guards the exact exception.

Runtime contracts are built once with endpoint and fan-in validation. The state
builder requires those prepared contracts; supplied state is always validated.
Structural component steps bind directly after their three-argument protocol
signature is checked at preparation; only CallableComponent normalizes author
callable arities. Topology is built from declared factories and policy, and
policy contexts expose the original author objects retained by bindings.

The existing component declaration layer owns fractional coverage, topology owns
only binary route restrictions, and the runtime mask resolver feeds the single
scalar/vector quantity dispatcher. No parallel mask framework is introduced.

Canonical-helper cleanup (2026-10-19) preserves these layers: layer 6 public
regridding factories construct layer 5 concrete regridders directly, without
private factory forwarders; conservative's public radius remains `radius_km`.
Layer 1 `calendar.YearType` is the sole enum owner, consumed directly by layer 2
forcing lookup. Layer 2 JCM PyTree normalization/reduction owns inexact precision
and categorical semantics; unused stack/unwrap/all-real helpers are removed.
Layer 3 data helpers retain production masking; explicit layer 2 layout owners
handle time-last arrays. Layer 6 Veris forcing packs provide native forcing and
face stresses from one bulk solution. Layer 14 JCM output consumes final runtime
payloads exclusively; setup resources cannot substitute mutable snapshot state.

Prescribed-contract cleanup (2026-10-19) retains these owners. Layer 3 data
factories and CAMulator land export prescribed fields only; gallery-owned
DataComponent declarations explicitly add diagnostic imports where displayed.
Slab factories declare only fields read by their existing kernels. Layer 6
recipes and galleries retain physical routes and explicit diagnostic routes.
The test-only coastal case data is shared by dispatch and substitution tests;
no production field registry or new runtime layer is introduced.

Final source audit (2026-10-19): all 179 Python modules are retained in their
existing layers; 164 non-gallery modules are enumerated above and the 15 gallery
modules form the application layer. Task 8 changes tests/documentation only.
The complete inventory, remaining adapters with actual callers, and removal
decisions are in `docs/source-coupling-audit.md`. No new runtime dependency or
alternate preparation/output path is introduced.

Scanned numeric output (2026-10-22) adds `vercor/output/_scan.py` after layer 17
and before layer 18. It composes layer 17's shared JAX runtime-step builder with
layer 14's existing immutable output accumulator; it owns neither cadence nor
file writes. The optional numeric provider/context contract remains in the
existing public output owner, with concrete extraction in layer 14. The layer
18 coordinator selects the numeric lane only when every enabled provider opts
in, and splits at the union of output boundaries plus schedule changes. Host
providers and custom backends retain their established single-step behavior.
