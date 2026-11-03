# Output accumulation profiling and optimization

The requested maintenance work stays on `feat/veris-seaice`, preserves scientific
behavior and public APIs, and ends with checked, committed, pushed changes.
Measure current host output accumulation using synchronized cold and warm runs,
XProf, Perfetto, and TensorBoard before choosing the implementation.

## Approach

Batch the numeric work for one selected output frame into a pure JIT kernel.
Keep field-specific schema and reduced-count validation in the existing output
owner. Tuple leaves preserve heterogeneous shapes and dtypes without padding or
stacking fields. Reuse the existing finite sum/count arithmetic. Preserve
operation order within each field, int32 output counts, missing-cell NaNs,
native exchange sampling, endpoint payloads, and differentiation.

Alternatives considered: adding a JIT around just each field's additions still
dispatches repeatedly; stacking fields forces a common shape/dtype and copies;
rewriting native model stepping has greater scope without profiling evidence.
The implementation choice is the agent's maintenance decision under the user's
explicit optimization request; it introduces no new feature or physical policy.

Only remove basic tests when every assertion has a stronger retained owner.
Keep distinct numerical, physical, autodiff, backend, and failure-boundary cases.
Benchmark isolated from test workloads and separate compile cost from warm
execution. Preserve installed Conda environments; profiling tools and raw
evidence live under `/private/tmp`. No GPU or climate-scale speed claim follows
from these CPU output-coordinator measurements.
