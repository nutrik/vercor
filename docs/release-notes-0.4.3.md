# VerCOR 0.4.3

VerCOR 0.4.3 republishes the stable protocol-first VerCOR 0.4 functionality
after the immutable `v0.4.2` release attempt failed during publication.

## Release recovery

- PyPI rejected 0.4.2 because
  `License :: OSI Approved :: Apache-2.0` is an invalid license classifier.
- The `v0.4.2` tag remains immutable. No PyPI 0.4.2 version or GitHub Release
  was created.
- Package metadata now uses
  `License :: OSI Approved :: Apache Software License` and tests require
  exactly one canonical license classifier.
- The GitHub workflow derives release identity and artifact names from
  `pyproject.toml`, which remains the sole executable VerCOR version owner.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.3"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 is intentionally
source-breaking for 0.3 applications; follow the
[migration guide](migration-0.3-to-0.4.md). Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT
environment; an exact compatible release is not yet pinned. CAMulator
spinup is not implemented. No legacy 0.3 adapter namespace is included.
