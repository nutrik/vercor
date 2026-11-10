# Bundled Output Default Alignment Design

## Goal

Align omitted-output behavior across bundled slab, data, JAXGCM and Veros components. Slab and data factories retain complete per-component
`OutputSpec` configurability, but omission no longer silently enables
step-period averages.

## Public Behavior

Every existing slab and data signature remains unchanged:

```python
output: OutputSpec | None = None
```

Omitting `output`, or passing `None`, resolves to:

```python
OutputSpec()
```

Consequently, no `PeriodOutput` is scheduled. Passing an explicit declaration
continues to configure cadence, variable selection, provider, and snapshot
writer:

```python
make_slab_ocean(
    grid,
    output=OutputSpec(
        period=PeriodOutput(
            frequency="month",
            variables=("sea_surface_temperature",),
        )
    ),
)
```

`OutputTarget.write_final_fields` remains a run-level policy and continues to
write final runtime fields even when the component has no period declaration.
`Coupler.run(output=None)` remains entirely I/O-free.

## Scope

The aligned default applies to:

- all four slab factories;
- ERA5 atmosphere, land, and ocean factories;
- the ERA-Interim ocean factory;
- the shared time-interpolated data-component helper;
- direct JCM land construction; and
- `JCMLandAtmosphereConfig.land_output`.

JAXGCM and Veros retain their existing model-native provider and
snapshot-writer installation. Those are adapter-specific representations of
native model state and are not duplicated for slab/data components, whose
runtime fields are already covered by final-field output.

The paired JCM atmosphere retains its historic explicit monthly default. This
is an explicit `OutputSpec` in the paired setup, not omitted-output behavior.
Paired JCM land changes from the former step-period default to `OutputSpec()`.

## Architecture

The private setup output helper becomes a generic optional-declaration
resolver:

```python
def resolve_output(output: OutputSpec | None = None) -> OutputSpec:
    if output is None:
        return OutputSpec()
    if not isinstance(output, OutputSpec):
        raise TypeError("output must be OutputSpec or None")
    return output
```

All slab factories, the shared data helper, and direct JCM land reuse this
resolver. The step-specific helper is removed. `JCMLandAtmosphereConfig` uses
`field(default_factory=OutputSpec)` directly, matching `JAXGCMConfig`,
and `VerosConfig`.

No runtime provider, accumulator, cadence, filename, backend, component
kernel, or NetCDF writer changes. The output session continues to install its
generic runtime-field provider only when an explicit `PeriodOutput` exists.

## Data Flow

1. A factory receives `output=None` or an explicit `OutputSpec`.
2. The resolver converts only `None` to `OutputSpec()` and preserves a supplied
   declaration by identity.
3. `ComponentSpec.output` owns the normalized policy.
4. An enabled `OutputTarget` independently selects final fields, period files,
   and snapshots.
5. Period sampling occurs only when `ComponentSpec.output.period` is not
   `None`.

## Validation and Errors

- A non-`OutputSpec`, non-`None` factory argument raises the same `TypeError`
  at every bundled slab/data boundary.
- Existing `OutputSpec` and `PeriodOutput` constructors continue to validate
  nested providers, cadence, variable names, and snapshot writers.
- Unknown selected provider variables retain component-scoped runtime errors.
- Configuration remains immutable after coupler construction.

## Testing

The red-green cycle covers:

- every slab default changing from step cadence to `period=None`;
- the shared data helper and every public data factory using `period=None` when
  omitted;
- direct and paired JCM land using `period=None` when omitted;
- a bare `OutputTarget` writing no period files for default slab/data
  components;
- run-level final fields still being written for a default slab component;
- explicit step and monthly policies still producing period files;
- monthly averaging still using all coupler-step samples;
- exact variable selection and supplied-`OutputSpec` identity;
- consistent invalid-output errors; and
- unchanged external native provider/snapshot behavior.

Public signatures do not change, so signature fixtures should remain
byte-for-byte stable. Documentation and progress records are updated to remove
the superseded step-default description. Black, strict flake8, mypy,
compileall, focused tests, fast/full pytest, branch coverage, installed
artifact probes, and Git whitespace checks run before completion.

## Compatibility

Factory call syntax is backward compatible. The intentional behavior change is
that callers relying on omitted slab/data step-period files must now request:

```python
OutputSpec(period=PeriodOutput(frequency="step"))
```

This makes omitted-output semantics consistent with the bundled external
component configuration classes and keeps period I/O explicitly declared.
