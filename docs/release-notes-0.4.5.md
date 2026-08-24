# VerCOR 0.4.5

VerCOR 0.4.5 updates the bundled atmosphere integration to JCM 2.0.1 and
strengthens finite-value and differentiation guarantees across coupled
Earth-system workflows.

## Highlights

- Migrated the optional atmosphere adapter to Dinosaur 1.3.6 and JCM 2.0.1,
  including immutable dynamics, physics, dycore, and physics-carry state.
- Preserved complete temperatures on fractional surface cells, fixing
  unphysical coastal forcing and non-finite coupled gradients.
- Added a no-NaN active-domain policy and neutralized masked undefined
  arithmetic while retaining NaN as an inactive missing-data sentinel.
- Corrected setup gallery masks, dtype initialization, conservative ocean
  cutoffs, NetCDF boolean output, and CAMulator composition coverage.
- Removed noisy per-step JAXGCM surface-temperature logging.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.5"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 is intentionally
source-breaking for 0.3 applications; follow
`docs/migration-0.3-to-0.4.md`. Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

JCM setups require the pinned `dinosaur==1.3.6` and `jcm==2.0.1` optional
dependencies supplied by `vercor[jcm]`.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT
environment; an exact compatible release is not pinned. CAMulator spinup
remains unsupported. No legacy 0.3 adapter namespace is included.
