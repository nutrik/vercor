# Sphinx Pyproject Version Design

## Goal

Make `docs/conf.py` read VerCOR's documentation version directly from
`[project].version` in the repository-root `pyproject.toml`. This preserves
`pyproject.toml` as the sole executable owner of the package version and avoids
importing `vercor` merely to configure Sphinx.

## Design

Resolve `pyproject.toml` relative to the location of `docs/conf.py`, not the
process working directory. Parse it with Python's standard-library `tomllib`
module and assign the resulting `[project]["version"]` value to both Sphinx's
`version` and `release` settings.

Remove the `vercor.__version__` import and the `0+untagged` fallback. A missing
file, invalid TOML document, missing `[project]` table, or missing `version`
field must fail explicitly during Sphinx configuration instead of silently
using `"main"`.

## Testing

Add a focused configuration contract test that executes the version-loading
portion of `docs/conf.py` against a synthetic project tree. The test will prove
that:

- the path is derived from `docs/conf.py`, independently of the working
  directory;
- the synthetic `[project].version` becomes both `version` and `release`; and
- the configuration no longer imports `vercor.__version__` or contains the
  `0+untagged` fallback.

The test will be observed failing before `docs/conf.py` is changed, then passing
after the minimal implementation. Existing focused documentation/version
contracts and the repository's fast suite will be run afterward. Any
pre-existing unrelated failure will be reported separately.

## Scope

This change affects only Sphinx version configuration and its focused test.
It does not change package metadata, runtime exports, release automation, or
the current project version.
