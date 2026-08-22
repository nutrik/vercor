# No-NaN Differentiation Policy Design

## Summary

VerCOR is a differentiable Earth-system coupler whose JAX execution paths must
produce finite active-domain primal values, forward-mode tangents, and
reverse-mode cotangents. The current code already tests selected gradients and
uses finite masks, but the guarantee is not yet expressed as one source-owned
contract or verified systematically across every bundled component and
numerical subsystem.

This change introduces a fail-fast numerical-safety policy. Intentional NaNs
remain valid missing-data sentinels outside component domains and in
output-only missing-data locations. NaNs or infinities in active computation
domains are errors. The implementation will repair unsafe arithmetic at its
first source, validate runtime handoffs with contextual diagnostics, and add
explicit JIT, JVP, and VJP evidence for every differentiable bundled path.

## Goals

- Guarantee finite active-domain primal values, JVP tangents, and VJP
  cotangents for every implemented JAX numerical kernel and bundled JAX
  component.
- Reject active-domain NaNs and infinities at the earliest owned runtime
  boundary with component, field or payload leaf, and stage context.
- Preserve intentional NaN missing-data semantics outside active component
  domains.
- Prevent inactive branches of masked JAX expressions from contaminating
  reverse-mode derivatives.
- Cover host-only bundled components with strict primal validation and cover
  their standalone JAX preprocessing kernels with JVP and VJP tests.
- Keep the policy private, centralized, and compatible with the stable VerCOR
  0.4 public API.
- Follow TDD and record a component-by-component audit and verification result.

## Non-goals

- Replacing NaN-backed missing-data storage with a new public mask or field
  representation.
- Making host model calls, Python callbacks, file output, or third-party
  opaque payloads differentiable.
- Silently repairing active non-finite values with `nan_to_num`, clipping,
  arbitrary fill values, or numerical fudge factors.
- Enabling JAX's process-global `jax_debug_nans` option, which is incompatible
  with intentional missing-data NaNs and is not a library-owned policy.
- Adding a public runtime switch that allows callers to disable the invariant.
- Refactoring unrelated component, output, workflow, or public API ownership.

## Considered Approaches

### Boundary-only validation

Validating fields after component steps and exchanges would be small and would
stop many non-finite values from reaching later components. It is insufficient
for differentiation: JAX may evaluate or transpose an unsafe inactive branch
of `jnp.where`, so a finite primal output can still have a NaN cotangent.

### Blanket sanitization

Applying `nan_to_num` throughout the runtime would often make outputs finite,
but it would hide invalid physics, make defects harder to localize, and assign
unreviewed derivatives at singularities. This conflicts with the fail-fast and
no-fudge-factor requirements.

### Layered source repair and boundary enforcement

The selected approach combines source-level numerical fixes with centralized
runtime checks. Each unsafe expression is corrected according to its physical
domain, while runtime handoffs identify the first owner that violates the
contract. Explicit JVP and VJP tests verify behavior that primal checks alone
cannot observe.

## Active-Domain Contract

A field's active domain is derived from its owning component grid:

- a grid without `binary_mask` treats every grid cell as active;
- a grid with `binary_mask` treats values greater than zero as active;
- leading time, level, ensemble, or forcing-record dimensions broadcast over
  the trailing grid dimensions; and
- an exchanged field additionally uses the route's positive fractional target
  mask for the exchange handoff.

Active values must be finite. Inactive values may be finite or NaN. Infinity is
never an intentional sentinel and is rejected in both active and inactive
locations. Output-only arrays may retain their existing schema-specific NaN
semantics when they do not re-enter model computation.

Payloads do not have a universal spatial interpretation. VerCOR-owned bundled
payloads will therefore receive component-specific checks wherever their
numeric leaves participate in later computation. Opaque third-party payload
semantics remain the component author's responsibility, but every declared
field crossing the central runtime is checked.

## Numerical-Safety Ownership

A dependency-light private numerical-safety module will own the reusable
active-mask broadcasting, finite predicates, and contextual transform-safe
assertion behavior. It will not become a public API. Lower-level numerical
kernels and the runtime may both depend on it without introducing a reverse
runtime dependency.

Reusable masked-arithmetic helpers will be added only where at least two call
sites share the exact operation and semantics. One-off physics repairs remain
local to their owning kernel. This keeps the policy DRY without creating a
generic numerical abstraction layer that the current code does not need.

Runtime orchestration will reuse the existing preparation, exchange,
field-transfer, and component-step owners. No second runtime validator
hierarchy or duplicate component adapter will be introduced.

## Differentiation-Safe Arithmetic

The audit will inspect divisions, roots, logarithms, powers, normalization,
masked products, and every `jnp.where` whose inactive branch can be undefined.
JAX traces and transposes both branches, so selecting a finite result is not
enough when an unselected expression computes `0 / 0`, a negative root, or an
equivalent singular derivative.

Repairs will use the mathematical domain of each operation. Typical patterns
include giving inactive operands finite neutral values before division or
other nonlinear operations, then applying the output mask. Active invalid
inputs are never hidden. Where a physical limiting value is required, it must
follow from the implemented equation and be covered by a targeted regression;
arbitrary epsilons or scale corrections are forbidden.

## Runtime Enforcement Flow

Preparation checks active initial and lifecycle-produced fields before the
first `RunState` can advance. Each component step then validates the following
handoffs in order:

1. scalar or vector regridding output on the route's active target domain;
2. fields placed in the destination component's received store;
3. received values after the component transfer policy is applied;
4. fields and owned payload leaves returned by the component step; and
5. fields selected into the component's sent store.

This sequence reports the first VerCOR-owned stage that introduces a
non-finite active value instead of allowing a later physics failure to obscure
the cause. Regridders may retain NaNs only outside their active target domain;
missing source coverage for an active target cell is an error.

Eager validation raises `CouplerError`. During compiled JAX execution, the
existing callback-compatible assertion style may be wrapped by JAX's runtime
exception, but the message will preserve the component, field or payload leaf,
and stage. Checks must remain compatible with `jax.jit`, `jax.jvp`, and
`jax.vjp` and must not introduce host conversion of traced arrays.

## Component and Kernel Coverage

The audit covers all implemented numerical owners, including:

- slab atmosphere, ocean, land, and sea-ice kernels and factories;
- ERA5 atmosphere, ocean, and land plus ERA-Interim ocean forcing components;
- bilinear scalar/vector interpolation and conservative remapping;
- exchange masks, dispatch, transfer-policy selection, and composed Coupler
  workflows;
- flux, vertical-coordinate, grid-geometry, and mask utilities;
- JCM field preparation, stepping, output mapping, and real coupled execution;
- Veros forcing preparation, stepping, field extraction, and coupled paths;
- CAMulator JAX preprocessing, field mapping, and wind-filter kernels; and
- common forcing selection, dtype normalization, and immutable runtime state
  paths that participate in differentiation.

Every JAX-executable bundled component and applicable pure numerical kernel
will have eager, JIT, explicit `jax.jvp`, and explicit `jax.vjp` coverage.
Directional JVP and VJP projections will be compared where their inputs and
outputs support a meaningful scalar pairing. Host-only model execution receives
strict primal checks but is not represented as differentiable. Its standalone
JAX kernels still receive transform coverage.

Optional real-model tests run when their dependencies and data are available.
Deterministic fakes enforce the same adapter boundary in the ordinary test
suite, so the policy does not depend solely on optional integration lanes.

## Test-Driven Implementation

Tests will be written before each production change. The first focused test
module will define the central active-domain contract and its eager, JIT, JVP,
and VJP behavior. Subsequent regressions will be placed beside the affected
kernel or component test.

The negative matrix will inject NaN and infinity at preparation, exchange,
receive, step-result, send, and bundled-payload boundaries and assert an
immediate contextual failure. Positive tests will prove that inactive NaN
sentinels remain accepted, active outputs remain finite, and inactive values do
not contaminate active tangents or cotangents.

The audit will maintain a compact component matrix recording primal, JIT, JVP,
VJP, inactive-NaN, and negative-failure evidence. Tests will print only
aggregate diagnostics in accordance with `AGENTS.md`.

## Documentation and Dependency Order

`DESIGN.md` will state the active-domain finite-value and differentiation
invariant. `DEPENDENCIES.md` will place the private numerical-safety owner in
the earliest compatible import layer and update later owners if required.
`PROGRESS.md` will record the completed component matrix, numerical defects and
their first-source repairs, rejected approaches, and concise verification
results.

## Repository and Verification Workflow

The clean obsolete PR worktree is removed and pruned before implementation.
Local `main` is fast-forwarded to merged PR #24 commit `54b29bd`, and work
continues on `audit/no-nan-ad-policy` in the primary checkout without recreating
`.worktrees/`.

Focused tests run after every RED/GREEN change. Before the implementation
commit, verification includes Black, strict flake8, mypy, compileall,
`git diff --check`, focused numerical-safety and component transform tests, the
configured fast and full suites, branch coverage, and package build and
installed-artifact checks applicable to the changed boundary. The complete
diff receives a final requirements and code-quality review. The work is
committed locally; pushing or creating a pull request requires separate user
authorization.

## Acceptance Criteria

- Active initial, exchanged, received, stepped, and sent fields cannot contain
  NaN or infinity without an immediate contextual failure.
- Intentional inactive-domain NaNs remain supported and do not contaminate
  active primal values, JVP tangents, or VJP cotangents.
- Every implemented JAX component and applicable numerical subsystem has
  explicit finite eager, JIT, JVP, and VJP evidence.
- Host-only components reject non-finite active primal values, and their JAX
  preprocessing kernels satisfy the transform contract.
- Unsafe arithmetic is repaired at its first source without sanitization,
  clipping, fudge factors, or unjustified epsilon terms.
- The stable public API, output opt-in behavior, optional-import laziness,
  immutable state model, and dtype policy remain unchanged.
- Project design, dependency order, and progress documentation reflect the
  implemented policy and audit evidence.
- Formatting, linting, type checking, compilation, focused tests, fast/full
  tests, branch coverage, build checks, and whitespace verification pass before
  the final implementation commit.
