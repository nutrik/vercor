# Changelog

All notable changes to VerCOR are recorded here. The project follows semantic
versioning; pre-releases may still refine new contracts.

## [Unreleased]

### Added

- Audit component interfaces, wrappers, physical recipe compatibility, lifecycle,
  and output/restart boundaries in `docs/interface-wrapper-audit.md`.
- Transfer ERA5 rain and water-equivalent snow and JCM total precipitation to
  Veros through zero-seeded `rain_rate` and `snow_rate` exchange fields.
- Compute virtual salt flux in VerCOR from precipitation minus bulk evaporation,
  with traced freshwater density and fixed reference salinity.
- Diagnose JCM rain and water-equivalent snow from native-grid SPEEDY ``t0``
  before remapping, using a sigmoid matched to the calibrated LT midpoint and
  slope, with traced phase-temperature thresholds.

### Changed

- Name bundled lowest-model-level meteorology with six explicit `air_*` keys.
  Distinguish JCM's `air_pressure_3d` from CAMulator's
  `model_level_air_pressure`, and name JCM's phase-only `t0` input
  `near_surface_air_temperature` without asserting a 2 m height.
- Rename shared recipes by physical contract, the JCM-backed factory/config to
  `make_jcm_atmosphere` / `JCMAtmosphereConfig`, and Veros surface-forcing modes
  to `bulk_meteorology` / `surface_fluxes`. Spell out public mask helpers and
  identify CAMulator's combined surface-temperature packer. Former names have
  no compatibility aliases; units, signs, staggering and coverage are retained.
- Derive gallery routes, execution order and topology roles from component
  names. Document the migration and physical contracts in
  `docs/coupling-fields.rst`.

### Fixed

- Preserve independently supplied conservative cell bounds and explicit source
  exclusion masks, including grids with identical centers. Compute bilinear
  weights from the same wrapped/clipped longitude used for source-cell lookup.
- Validate monthly/daily forcing record layouts before selection, and preserve
  field precision in monthly interpolation and period output.
- Write accepted list, tuple, and scalar provider coordinates through the shared
  host-array boundary before defining NetCDF dimensions.
- Accept nanosecond CAMulator forcing times and reject missing records,
  inconsistent native cadence, and unsupported calendars before native stepping.
- Keep JCM evolving state exclusively in payloads and validate native save
  intervals before bootstrap or spinup.
- Distinguish JCM bottom-level air variables from near-surface temperature and
  surface radiation in local adapter names, preserving bulk-flux field meanings.
- Apply the VerCOR global forcing callback in both Veros execution lanes,
  preserve cyclic salt-flux halos, and report unambiguous salinity-flux units.
- Make global salinity restoring an optional additive term controlled by
  `VerosConfig.restore_to_climatology_salt`, disabled by default in the gallery.
- Replace the shared Veros restoring argument with independent
  `restore_to_climatology_temp` and `restore_to_climatology_salt` flags, both
  disabled by default. Apply the switches in the native global callback so
  they also control restoring during spinup; preserve raw `qnec` diagnostics.

## [0.4.6] - 2026-08-27

### Added

- Added capability-gated differentiable JAX execution for the supported
  Veros fork, including stable multi-step forward- and reverse-mode
  gradients.

### Changed

- Made `VerosConfig.execution` the sole Veros execution-policy option and
  removed the redundant `jitted` flag.
- Strengthened CI isolation, provenance, dependency installation, and
  combined coverage for stock and differentiable Veros lanes.

### Fixed

- Copy complete stock-Veros states without enumerating native attributes,
  preserving diagnostics and future additions while retaining the fork's
  native PyTree copy path.
- Corrected JCM sigma-level height evaluation and validated hybrid-sigma
  height calculations.
- Removed unsupported VerCOR release references from the complete tracked
  source, documentation, test, and archive tree.

## [0.4.5] - 2026-08-24

### Changed

- Updated the optional atmosphere integration to Dinosaur 1.3.6 and JCM
  2.0.1, including explicit immutable carry transitions and current SPEEDY
  diagnostics.
- Enforced finite active-domain values and transform-safe masked arithmetic
  across runtime, exchange, regridding, flux, and bundled-model boundaries.

### Fixed

- Preserved complete land and sea temperatures on fractional JCM surface
  cells, restoring finite forward- and reverse-mode coupled gradients.
- Corrected setup-gallery target masks, conservative ocean-mask cutoffs, JAX
  x64 initialization, and portable NetCDF boolean encoding.
- Removed unsolicited per-step JAXGCM surface-temperature logging and its
  unused reductions.

## [0.4.4] - 2026-08-20

### Added

- Packaged the runnable setup gallery and added the installed ``vercor`` CLI
  for listing, safely copying, and running local setup scripts through the
  explicit ``run_setup(*, loglevel, float_type)`` child-process contract.
- Added installed wheel and source-distribution coverage for the private setup
  runner and an external lightweight setup contract.
- Added a canonical Read the Docs manual with researcher and developer learning
  paths, executable examples, and a curated API reference.
- Added a recreatable Conda environment for source installations.

### Changed

- Hardened the tagged-release workflow with bounded GitHub release-visibility
  checks and exact artifact-state validation.
- Made Sphinx derive its displayed version from package metadata and made
  documentation dependencies explicit in CI.

### Fixed

- Normalized JCM state and forcing arrays to the configured runtime dtype before
  optional spinup.
- Hardened setup discovery, copying, child execution, and installed-wheel
  validation across supported CLI paths.

## [0.4.3] - 2026-07-25

### Fixed

- Replaced the invalid `License :: OSI Approved :: Apache-2.0` classifier
  that caused PyPI to reject the immutable `v0.4.2` release attempt with the
  canonical Apache Software License classifier.
- Enforced exactly one canonical license classifier in distribution
  metadata.

### Changed

- GitHub Actions now derives the package version, artifact names, tag guard,
  release-notes path, and publication inventory from `pyproject.toml`.

## [0.4.2] - 2026-07-25

### Fixed

- The `v0.4.1` tag was created from a stale commit and failed before
  publication. It remains immutable and is superseded by the 0.4.2 recovery
  candidate.
- Active package metadata, artifact names, installation guidance, and release
  transcripts now use the new 0.4.2 package and tag identities.

## [0.4.1] - 2026-07-24

### Fixed

- GitHub Actions now proves its effective `contents: write` capability through
  the non-mutating Release notes-generation endpoint instead of interpreting
  repository `permissions.push` for an installation token.
- The failed immutable `v0.4.0` workflow remains preserved; publication
  recovery proceeds through the new `v0.4.1` patch tag.

## [0.4.0] - 2026-07-23

### Added

- Configurable period output for bundled slab and forcing-data models.
- Executable plugin-authoring guidance and temporary installed external-extension
  fixture verification.

### Changed

- Bundled output defaults now require an explicit period policy to write period
  averages.
- The stable extension tier and third-party plugin floor
  `vercor>=0.4.0,<0.5` are formalized.

### Fixed

- Time-dependent forcing output now samples the field selected for the current
  coupling step.
- Period-average filenames and timestamps preserve each averaging window's
  identity.
- Veros and CAMulator runtime payloads are functionally owned and defensively
  copied.
- Veros linear-solver caches remain valid for the lifetime of runtime payloads.
- JAXGCM runtime calculations consistently apply the configured dtype.
- Installed-artifact and NetCDF CI gates are stable across supported lanes.

### Compatibility

- VerCOR supports Python 3.12 and 3.13.
- The 0.4 API does not include legacy compatibility adapters; use the current
  researcher and developer guides.
- No legacy adapter namespace is included.

### Known limitations

- CAMulator still requires a separately installed compatible MILES-CREDIT
  environment; an exact compatible release is not pinned.
- CAMulator spinup remains unsupported.

## [0.4.0a1] - 2026-07-14

### Added

- Structural component authoring with immutable declarations and setup results.
- Stable exchange route IDs, route-keyed topology, and scalar/vector regridder
  capabilities.
- Validated workflow plans, chunk-oriented execution backends, and a public
  runtime driver.
- Unified provider, period, target, and snapshot output contracts.
- Frozen traced `PhysicalConstants` and setup-owned frozen configuration.
- Installed wheel, source-distribution, temporary installed external-extension
  fixture verification, optional-model, and macOS release gates.

### Changed

- The package root now exports exactly six primary conveniences.
- Coupler assembly is constructor-only and prepared configuration is private.
- `RunState` is opaque and exposes immutable field replacement.
- Runtime precision is owned only by `RuntimeOptions.dtype`.
- Bundled slab, JCM, Veros, and CAMulator factories use the 0.4 component and
  output contracts.
- Bundled slab and data factories accept per-component `OutputSpec` overrides;
  omission matches external configurations with no period policy.

### Removed

- Primary legacy aliases, settings, authoring mixins, coupler recipes/mutators,
  callable-derived route identity, backend-owned output, and public preparation
  internals.
- Duplicate native/generic output accumulators and hidden output markers.

### Compatibility

This alpha does not ship legacy compatibility adapters. Use the current
researcher and developer guides for supported workflows.

[Unreleased]: https://github.com/nutrik/vercor/compare/v0.4.6...HEAD
[0.4.6]: https://github.com/nutrik/vercor/compare/v0.4.5...v0.4.6
[0.4.5]: https://github.com/nutrik/vercor/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/nutrik/vercor/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/nutrik/vercor/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/nutrik/vercor/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/nutrik/vercor/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/nutrik/vercor/compare/v0.4.0a1...v0.4.0
[0.4.0a1]: https://github.com/Roman-N/VerCOR/releases/tag/v0.4.0a1
