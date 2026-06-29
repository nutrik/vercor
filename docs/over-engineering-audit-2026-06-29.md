# Over-Engineering Audit - 2026-06-29

## Executive summary

The remaining high-value simplification targets are small helper surfaces rather
than broad architecture changes. Runtime, output, and component-owner modules
look abstract in places, but most of that structure is documented and
boundary-tested to protect JAX scan state, optional dependency loading, and the
public component-author API. The safe cleanup is to remove unused helper
exports and one hidden clock dispatch attribute while leaving public
compatibility-bound APIs intact.

## Findings

| File | Symbol | Issue | Simpler alternative | Risk | Priority |
|---|---|---|---|---|---|
| `vercor/clock.py` | `Clock._iter_impl` | Stored a callable dispatch target during `__post_init__` only so `iter()` could delegate later. This hid a simple calendar branch in mutable instance state. | Branch directly in `Clock.iter()` between Gregorian and model-calendar iterators. | Low | Quick win |
| `vercor/setups/external/camulator_wind_filter.py` | `wind_filter`, `simple_wind_artifact_filter` | Low-level convenience wrappers were exported but unused by production code; the runtime only needs config loading, failure-tolerant post-processing, and tensor-level application. | Remove the wrappers and keep the production wind-filter facade plus private tensor mechanics. | Medium | Quick win |
| `vercor/dtypes.py` | `jax_real_array_copy` | One-line helper had no production callers and existed only as a test-maintained API surface. | Remove it; future production code can call `jnp.array(..., dtype=jax_real_dtype(policy), copy=True)` at the point of use. | Low | Quick win |
| `vercor/grid.py` | `Grid` | The ABC still has one production implementation and mostly centralizes mask validation plus display behavior. | Keep for now because it is public and boundary-tested; revisit only with a deprecation plan. | Medium | Later |
| `vercor/run_sequence.py` | `RunSequence` | One-field iterable wrapper around `list[str]` is more structure than the runtime strictly needs. | Keep for now; later allow plain sequences at `Coupler` entrypoints while preserving compatibility. | Low | Later |
| `vercor/calendar.py` | Forcing-index compatibility delegates | Thin delegates duplicate apparent API surface for historic import paths. | Keep until old `vercor.calendar` imports can be deprecated. | Low | Avoid unless needed |
| `vercor/components/*` | Lifecycle, factory, and callable wrapper stack | Complex, but it supports documented public component-author APIs, lifecycle hooks, and host-runtime separation. | Avoid broad collapse; simplify only inside a dedicated component API redesign. | High | Avoid unless needed |

## Recommended refactor plan

1. Remove the clock iterator dispatch attribute and keep all existing calendar
   iteration behavior.
2. Remove unused CAMulator low-level wind-filter convenience exports while
   preserving the runtime post-processing entrypoint.
3. Remove the unused dtype copy helper and keep the canonical dtype policy
   helpers that production code uses.
4. Leave `Grid`, `RunSequence`, calendar delegates, and component authoring
   layers as follow-up candidates unless a public compatibility decision is
   made.
5. Keep focused regression tests so removed helper surfaces do not return
   accidentally.
