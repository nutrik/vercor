# VerCOR 0.4.1

VerCOR 0.4.1 republishes the stable protocol-first VerCOR 0.4 functionality
with corrected release automation.

## Fixed

- The GitHub Actions release job now verifies its effective
  `contents: write` capability through GitHub's non-mutating Release
  notes-generation endpoint.
- The failed `v0.4.0` workflow and tag remain immutable; this patch uses new
  package and tag identities.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.1"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 is intentionally
source-breaking for 0.3 applications; follow
`docs/migration-0.3-to-0.4.md`. Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT environment;
an exact compatible release is not yet pinned. CAMulator spinup is not
implemented. No legacy 0.3 adapter namespace is included.
