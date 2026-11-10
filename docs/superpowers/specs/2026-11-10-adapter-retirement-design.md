# Host emulator retirement

Remove the host neural atmosphere adapter and its prescribed-land companion
from the tracked project, including factories, configuration, native helpers,
gallery, adapter-specific tests, dependencies, CI references, and documentation.
The user explicitly requested complete removal, a separate branch, full static
and runtime validation, a commit, and a push using GitHub CLI.

The implementation deletes the integration directly. Keeping compatibility
aliases or adding a replacement would retain unwanted ownership and is outside
the requested scope. Shared numerical kernels and the remaining JCM, Veros,
Veris, data and slab integrations keep their existing contracts.

Update exact public manifests, signatures and gallery inventories before code
removal and observe their failures. Retain generic host-execution and payload
tests when their behavior is independent of the retired integration. Remove
dedicated fixtures and dependencies with their last consumers. Clean historical
documentation without relabeling old measurements as new results.

Validate formatting, strict lint, typing, compilation, documentation, focused
tests, fast/full tests and distribution contents. Obtain an independent review,
commit only intended files and verify the pushed remote SHA. Work on
a separate feature branch; preserve installed environments and Git history.
