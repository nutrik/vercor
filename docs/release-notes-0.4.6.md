# VerCOR 0.4.6

VerCOR 0.4.6 adds verified multi-step differentiable Veros execution and
corrects native-state ownership and atmospheric hybrid-sigma heights.

## Highlights

- Added a capability-gated JAX lane for the supported Veros fork with stable
  forward- and reverse-mode gradients across multiple steps.
- Made `VerosConfig.execution` the single execution-policy option.
- Preserved complete stock-Veros state copies, including diagnostics and
  future native additions.
- Strengthened stock/fork CI provenance, isolation, dependency, and combined
  coverage gates.
- Corrected JCM sigma-level height evaluation and validated hybrid-sigma
  calculations.
- Removed unsupported VerCOR release references from tracked source,
  documentation, tests, and archives.

The stock state copy is owned by the generic runtime; the fork retains its
native PyTree copy path.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.6"
```

## Compatibility

VerCOR requires Python 3.12 or 3.13. The stable public API remains the 0.4
API, and third-party plugins should depend on `vercor>=0.4.0,<0.5` and use
the documented stable extension modules.

JCM setups use the pinned optional dependencies supplied by `vercor[jcm]`.
The differentiable Veros lane requires the supported JAX-capable fork;
ordinary `vercor[veros]` installations continue to use stock host execution.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT
environment; an exact compatible release is not pinned. CAMulator spinup
remains unsupported.
