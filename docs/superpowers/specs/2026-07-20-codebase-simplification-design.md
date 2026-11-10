# VerCOR Codebase Simplification Design

Date: 2026-07-20

## Objective

Simplify VerCOR's private implementation without changing its supported public
API, numerical results, JAX transformation behavior, output formats, optional
dependency boundaries, or third-party plugin contracts. The work combines
low-risk dead-code and indirection removal with two bounded performance
follow-ups: reducing bilinear-interpolator PyTree state and avoiding repeated
full Veros-state copies.

## Constraints

- Preserve the six-symbol package root and every canonical public owner,
  signature, protocol, and output filename format.
- Preserve immutable registered PyTrees, traced physics values, runtime dtype
  ownership, structural component plugins, custom workflows/backends, topology
  policies, and output providers.
- Preserve exact runtime validation coverage and error semantics. This change
  does not remove validation calls whose redundancy has not been profiled.
- Preserve lazy imports for JCM/JAXGCM, Veros, and other optional frameworks.
- Use test-driven changes. Refactoring tests should assert observable behavior
  and invariants rather than require unnecessary private module placement.
- Do not change ignored public physics parameters, public field-name catalogs,
  logging exports, `RectilinearGrid.from_coordinates()`, seeded topology-map
  behavior, or other public/deprecation-sensitive surfaces in this work.

## Design

### 1. Remove dead state and unreachable code

Delete state that has no reader or runtime effect:

- `CallableComponent._author_step`, `_ComponentDeclaration.author_step`, and
  `_ComponentBinding._author_step`;
- `_OutputSchema.index`;
- the unused virtual-temperature helper in `vercor.fluxes.utilities`;
- setup-only `_data_files` and `_hybrid_coefficients` attributes;
- `FieldStore.get_or_zeros_like()` and `RuntimeTopologyMaps.empty()`;


No replacement abstraction will be introduced for deleted state. If a future
feature needs one of these values, it must add it at the real point of use with
tests for that use.

Move `_Grid` mask validation and display behavior directly onto
`RectilinearGrid`, its only production subclass, and remove the private ABC.
The public `RectilinearGrid.from_coordinates()` constructor remains unchanged.

### 2. Consolidate regridding behavior

Move the identical scalar `regrid()` implementation from the bilinear and
conservative subclasses into `_BaseRegridder`. The base already owns the
source/target grids, identity decision, and interpolator, so it is the cohesive
owner of this shared behavior. Bilinear retains only its vector-specific
method; each subclass retains construction of its concrete interpolator.

The public `Regridder`, `VectorRegridder`, and `RegridderFactory` protocols and
public factory signatures remain unchanged. Removing the private free factory
functions is deferred unless implementation shows that it is required for the
shared-method cleanup.

### 3. Simplify output internals

Create one private grid-dimension inference helper in `vercor.output._dataset`
and use it for both default period-provider frames and final runtime fields.
The helper will preserve the existing dimension names byte-for-byte.

Replace repeated `_OutputAccumulator` reconstruction with
`dataclasses.replace()` only if focused eager and `jax.jit` tests demonstrate
identical PyTree structure, schema metadata, sums, counts, coordinate values,
means, and reset behavior. Otherwise retain the explicit construction and
record why it is JAX-sensitive.

Filename tokenization and collision allocation are excluded because their
formats are externally observable and independently tested.

### 4. Remove narrow runtime indirection

- Make `build_exchange_topology()` return `RuntimeTopologyMaps` directly and
  remove the one-field `ExchangeTopologyState`. `RuntimeInitializationState`
  will store `topology_maps` directly.
- Move the checked surface-role component lookup into `surface_masks` and
  remove the one-caller `_NamedComponent` protocol/module.
- Inline `RuntimeRunContext` construction at the run boundary and call the
  existing execution owner's plan-build/execute functions directly, removing
  the one-use runner wrapper.

`RuntimeTopologyMaps`, `RuntimeInitializationState`, `RuntimeRunContext`, and
`RuntimeDispatchContext` remain. They group multiple cohesive values and carry
real immutability, execution, or JAX-boundary meaning.

The optional contract-rebuild path, seeded topology parameters, and backend
validation placement are deferred; they have wider test and correctness
implications than this safety-first pass.

### 5. Reduce bilinear-interpolator PyTree state

Keep `fx` and `fy` local to weight construction rather than storing them on
`BilinearRectilinearInterpolator` as unused target-sized dynamic leaves. Remove
redundant `nx_source` and `ny_source` aliases in favor of the existing `nlon`
and `nlat` values. Treat descending-latitude detection as construction state
unless a real post-construction consumer is found. Remove the duplicate
`self.nlon` constructor assignment. Keep both source-validity calculations:
they belong to distinct bilinear and extrapolation paths and are not dead code.

Focused tests must verify:

- identical scalar and vector interpolation results;
- identical extrapolation and mask behavior;
- PyTree flatten/unflatten and `jax.jit` round trips;
- fewer dynamic leaves without changing required static reconstruction data;
- forward- and reverse-mode gradient behavior already covered by the suite.

If a field proves necessary for correct PyTree reconstruction, it remains and
the reason is documented rather than forcing its removal.

### 6. Copy Veros state once per forcing update

Refactor `apply_veros_forcing_fields()` to copy and unlock the Veros state once,
assign `taux`, `tauy`, `qnet`, and `qnec` to that copy, and return it. Preserve
copy-before-mutate semantics, locking behavior, host/JIT selection, dtype and
shape behavior, and the exact four updated values.

`set_variable()` may remain as a separately tested single-variable helper if it
has a meaningful caller after the refactor; otherwise it will be removed as
dead private code. Tests will assert one copy operation and four assignments,
then compare updated values with the pre-refactor behavior.

## Error handling and compatibility

All supported exceptions and validation messages remain unchanged unless they
name a deleted private type. Removing wrappers must not broaden caught
exceptions or suppress component, topology, output, JAX, or external-framework
errors.

Private-layout tests affected by the cleanup will be replaced with positive
tests for retained invariants: frozen topology mappings, component lookup
diagnostics, validated execution-plan order, runtime context contents, output
dimension parity, PyTree reconstruction, and external-state copy semantics.

## Verification

Development proceeds in small TDD units. Each unit runs focused tests followed
by the fast suite. Before the final commit, run:

1. Black on `vercor`, `examples`, and `tests`.
2. Strict flake8 with the repository's 120-character limit.
3. mypy on `vercor`, `examples`, and `tests`.
4. `compileall` for source, examples, and tests.
5. Focused component, regridding, interpolator, output, runtime/topology,
   JAXGCM and Veros tests.
6. `pytest tests/ -q --fast`.
7. The complete parallel test suite and branch coverage gate.
8. `git diff --check`.
9. `graphify update .` and a final clean working-tree review.

`PROGRESS.md` and `DEPENDENCIES.md` will be updated if implementation changes
the documented module dependency order. The final Git commit will contain only
the approved simplifications, corresponding tests, and current documentation.

## Out of scope

- Public API removal or deprecation.
- Physics-formula or public numerical-signature changes.
- New factories, registries, dependency injection, or configuration systems.
- Output filename policy changes.
- Removing runtime validation without profiling and dedicated boundary tests.
- Broad rewrites of component, workflow/backend, output-provider, interruption,
  calendar, or optional-adapter architecture.
