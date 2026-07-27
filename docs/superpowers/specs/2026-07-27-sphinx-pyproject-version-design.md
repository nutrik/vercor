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
contracts and the repository's fast suite will be run afterward.

The new `.readthedocs.yaml` follows Read the Docs' official template and links
to its external configuration reference whose final filename is the letter
`v`, the schema number `2`, and `.html`. The repository version-policy scanner
currently mistakes that URL segment for a stale VerCOR API label. Add a focused
policy test first, then exempt only a matching token whose character span is
inside that exact official Read the Docs URL in the root configuration file. A
stale API token elsewhere on the same line must remain rejected. Tests and
matcher definitions must construct the external token from separate string
fragments so the repository scanner continues to police its own tracked source.

## Scope

This change affects Sphinx version configuration, the narrow Read the Docs URL
policy correction, and their focused tests. Include all current and newly
created untracked documentation files in the implementation commit:
`.readthedocs.yaml`, `docs/Makefile`, `docs/conf.py`, and
`docs/requirements.txt`. Do not change package metadata, runtime exports,
release automation, or the current project version.
