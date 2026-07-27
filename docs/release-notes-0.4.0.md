# VerCOR 0.4.0

VerCOR 0.4.0 stabilizes the protocol-first JAX coupler architecture introduced
in 0.4.0a1.

## Highlights

- Structural component authoring, immutable runtime state, stable exchange
  route IDs, validated workflows/backends, and unified output contracts.
- Configurable period output for bundled slab and forcing-data components.
- Correct time-selected output sampling and period-window file identity.
- Installed wheel, source-distribution, temporary installed external-extension
  fixture verification, optional-model, and differentiation release gates.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.0"
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
