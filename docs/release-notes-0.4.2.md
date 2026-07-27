# VerCOR 0.4.2

VerCOR 0.4.2 is the recovery candidate for the stable protocol-first VerCOR
0.4 functionality.

## Recovery

- The `v0.4.1` tag was created from a stale commit, failed before publication,
  remains immutable, and is superseded by the 0.4.2 candidate.
- The generic release producer guard rejects any version tag that does not
  match the package version.
- Package metadata, workflow artifacts, installation guidance, and release
  commands now consistently use the 0.4.2 package and tag identities.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.2"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 is intentionally
source-breaking for 0.3 applications; follow the
[migration guide](migration-0.3-to-0.4.md). Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT environment;
an exact compatible release is not yet pinned. CAMulator spinup is not
implemented. No legacy 0.3 adapter namespace is included.
