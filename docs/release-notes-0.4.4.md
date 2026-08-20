# VerCOR 0.4.4

VerCOR 0.4.4 adds an installed command-line workflow for discovering, copying,
and running the packaged setup gallery, alongside expanded documentation for
researchers and component developers.

## Highlights

- The new `vercor` command-line interface lists packaged and external setup
  templates, copies them without overwriting existing work, and runs explicit
  `run_setup(*, loglevel, float_type)` contracts in a child process.
- The Read the Docs manual now provides focused researcher and developer paths,
  executable examples, setup-gallery guidance, and a curated API reference.
- JCM initialization normalizes state and forcing arrays to the configured
  runtime dtype before optional spinup.
- Installed wheel checks cover the setup runner and command-line entry point.
- The release workflow now uses bounded release visibility polling and exact
  artifact-state validation around GitHub mutations.

## Upgrade

```bash
python -m pip install --upgrade "vercor==0.4.4"
```

## Compatibility and migration

VerCOR requires Python 3.12 or 3.13. Version 0.4 remains intentionally
source-breaking for 0.3 applications; follow the
[migration guide](migration-0.3-to-0.4.md). Third-party plugins should depend on
`vercor>=0.4.0,<0.5` and use the documented stable extension modules.

## Known limitations

CAMulator requires a separately installed compatible MILES-CREDIT environment;
an exact compatible release is not yet pinned. CAMulator spinup is not
implemented. No legacy 0.3 adapter namespace is included.
