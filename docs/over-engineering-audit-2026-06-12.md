# Over-Engineering Audit - 2026-06-12

## Executive summary

The codebase has already had several recent simplification passes, so the
remaining high-value opportunities are mostly small internal indirections rather
than broad rewrites. The quick wins are in runtime topology copying and output
module facades. Larger public surfaces such as `Grid`, `RunSequence`, and the
component authoring layer are small or complex enough to invite scrutiny, but
they remain compatibility-bound public APIs and should not be removed without a
separate deprecation plan.

## Findings

| File | Symbol | Issue | Simpler alternative | Risk | Priority |
|---|---|---|---|---|---|
| `vercor/runtime/topology_state.py` | `RuntimeTopologyMaps.from_mappings` | Single-use copy/empty constructor helper made behavior harder to trace and turned a simple copy into a boundary-tested API shape. It mattered because future readers had to inspect two modules to understand initialization semantics. | Remove the classmethod and copy existing maps directly in `build_exchange_topology_maps`. | Low | Quick win |
| `vercor/output/__init__.py` | `_RUNTIME_EXPORTS`, `__getattr__`, `__dir__` | Lazy facade for three lightweight runtime-output exports added import indirection without protecting an optional dependency. It mattered because `vercor.output` behavior was hidden behind dynamic lookup. | Directly import and reexport the same public names from `vercor.output.runtime`. | Low | Quick win |
| `vercor/output/period_files.py` | `MeanVariablesBuilder`, `CoordinateVariablesBuilder`, `DataVariablesBuilder` | Local type aliases were used once, so they obscured the public function signature more than they clarified it. | Inline the `Callable[...]` annotations in `write_period_average_netcdf`. | Low | Quick win |
| `vercor/grid.py` | `Grid` | The ABC has one production implementation and mostly shares display plus mask validation. It matters because it suggests broader grid polymorphism than currently exists. | Keep for public compatibility; consider accepting plain grid protocols or deprecating the ABC only in a future public API cycle. | Medium | Later |
| `vercor/run_sequence.py` | `RunSequence` | One-field iterable dataclass around `list[str]` is more structure than the current runtime needs. It matters because users must construct a wrapper for a simple ordered name list. | Keep for public compatibility; later allow plain sequences at coupler entrypoints while preserving `RunSequence`. | Low | Later |
| `vercor/components/*` | Lifecycle, factory, and mixin layers | The component authoring stack is complex, but it supports documented public extension points, lifecycle hooks, and structural host-runtime dispatch. | Avoid broad collapse; simplify only with a dedicated component API redesign and compatibility policy. | High | Avoid unless needed |

## Recommended refactor plan

1. Remove `RuntimeTopologyMaps.from_mappings` and keep explicit copy behavior in
   `vercor.runtime.exchange_topology`.
2. Replace the lazy `vercor.output` runtime facade with direct reexports while
   preserving the same public `__all__`.
3. Inline one-use period-file builder aliases in the function signature.
4. Leave `Grid`, `RunSequence`, and component authoring surfaces as documented
   follow-up candidates rather than changing public behavior.
5. Continue using focused boundary tests for simplification sweeps so helpers
   do not return accidentally.
