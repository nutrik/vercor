# Changelog

All notable changes to VerCOR are recorded here. The project follows semantic
versioning; pre-releases may still refine new contracts.

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
- Version 0.4 is intentionally source-breaking for 0.3 applications; follow
  `docs/migration-0.3-to-0.4.md`.
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

- Primary 0.3 aliases, settings, authoring mixins, coupler recipes/mutators,
  callable-derived route identity, backend-owned output, and public preparation
  internals.
- Duplicate native/generic output accumulators and hidden output markers.

### Compatibility

This alpha does not ship legacy adapters. Follow
`docs/migration-0.3-to-0.4.md` to migrate 0.3-only workflows directly.

[0.4.2]: https://github.com/nutrik/vercor/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/nutrik/vercor/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/nutrik/vercor/compare/v0.4.0a1...v0.4.0
[0.4.0a1]: https://github.com/Roman-N/VerCOR/compare/v0.3.2...v0.4.0a1
