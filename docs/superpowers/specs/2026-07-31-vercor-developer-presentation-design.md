# VerCOR Developer Presentation Design

## Purpose

Create a 20–25 minute PowerPoint presentation about VerCOR for Earth system
modelers. Most attendees are developers, and some may want to integrate their
own models as VerCOR components. The deck must explain why explicit coupling
matters scientifically while giving prospective integrators a practical and
accurate path into the public VerCOR 0.4 contracts.

Communication job:

> By the end, developers should understand how VerCOR composes Earth-system
> components, where runtime responsibilities begin and end, and what they must
> implement and validate to wrap a model successfully.

The audience has mixed familiarity with JAX and automatic differentiation.
The deck will therefore introduce their consequences in plain language before
using JAX-specific terms.

## Narrative approach

Use the approved architecture-to-integration-to-differentiability progression:

1. Coupling decisions belong in the scientific experiment.
2. VerCOR separates component physics from coupling decisions.
3. Components and exchanges form an explicit experiment graph.
4. A shared clock governs the receive, step, and send cycle.
5. The runtime and component author have distinct responsibilities.
6. A model becomes a component through four public contracts.
7. Evolving model state remains explicit and immutable.
8. Exchanges make field transfer, regridding, topology, and scheduling visible.
9. JAX-native and host components coexist with different capabilities.
10. Integration proceeds through staged validation before coupled experiments.

An atmosphere-ocean-land example will recur across the architecture slides so
that the abstractions remain connected to a recognizable Earth-system setup.

## Slide sequence and pacing

The deck contains 14 slides and targets approximately 22 minutes, leaving
flexibility within the requested 20–25 minute range.

1. **VerCOR: differentiable coupling for Earth-system models** — minimal title
   and one-sentence promise.
2. **Coupling decisions belong in the experiment** — motivate exchanged
   fields, grids, clocks, ordering, and boundary conventions.
3. **Four owners keep coupling explicit** — introduce components, exchanges,
   runtime, and output as separate responsibilities.
4. **A coupled experiment is an explicit component graph** — show atmosphere,
   ocean, land, sea ice, and forcing connected by named fields.
5. **Every step follows receive, step, send** — explain the shared-clock
   execution cycle and run-order consequences.
6. **The runtime owns coordination; components own physics** — map validation,
   state, transfer, scheduling, and output boundaries against component code.
7. **A component is four public contracts** — identity, grid, specification,
   and step.
8. **Wrapping a model can stay structurally small** — show one compact public
   component example and annotate the contract rather than teaching syntax.
9. **Evolving state belongs in an immutable payload** — distinguish author
   configuration from per-run model state.
10. **Exchanges expose transfer and scheduling choices** — cover route identity,
    scalar/vector regridding, topology, masks, and rejected ambiguous fan-in.
11. **Differentiability means tracing cause through the coupled system** — use a
    parameter-to-diagnostic gradient path and explain JAX/AD without assuming
    prior experience.
12. **JAX-native and host components have different capabilities** — compare
    compiled differentiable execution with ordinary Python or external-model
    execution without implying that all mixed runs are differentiable.
13. **Integrate in stages, validate at every boundary** — isolated kernel,
    field/grid validation, exchange validation, short coupled run, and gradient
    validation where applicable.
14. **Make coupling an inspectable part of your model** — close with concrete
    next actions and current boundaries.

Speaker notes will provide a natural talk track, approximate pacing,
transitions, and source blocks. Timing scaffolds will not appear on slides.

## Content balance

- Approximately 60% architecture and integration.
- Approximately 25% scientific motivation and coupling meaning.
- Approximately 15% JAX, automatic differentiation, and capability boundaries.

Only one compact code example will appear. It will use the stable public 0.4
interface and avoid private runtime internals. A prospective integrator should
leave with a credible first checklist, not an exhaustive API tutorial.

## Visual system

Use the bundled Codex Grid layout library as the composition reference because
no external presentation template was supplied. Adapt it to an Earth-system
palette:

- deep ocean blue for the base;
- atmosphere teal;
- land ochre;
- sea-ice pale cyan;
- warm coral for constraints and capability boundaries; and
- warm off-white for explanatory slides.

Typography will meet the presentation-skill minimums: at least 50 pt for the
deck title, 35 pt for slide titles, 24 pt for subheads, and 16 pt for body text.
Slides will use takeaway titles, low text density, equal margins, and varied
flat compositions rather than dashboard-like card grids.

The deck will avoid decorative stock photography. Visual meaning will come
from strong typography and four focused diagrams:

1. Earth-system component and exchange graph.
2. Shared-clock receive-step-send flow.
3. Parameter-to-diagnostic gradient pathway.
4. Staged model-integration and validation flow.

Mermaid Chart will be used to create the requested graph and flow diagrams.
The final deck may use editable native PowerPoint shapes when that produces
clearer slide rendering while preserving the validated Mermaid logic.

## Evidence and source policy

Primary sources are the repository's implemented architecture and maintained
documentation:

- `DESIGN.md`;
- `README.md`;
- `PROGRESS.md`;
- `DEPENDENCIES.md`;
- `docs/plugin-authoring.md`;
- the developer and researcher guides under `docs/`; and
- public examples and API docstrings.

Claims about JAX or external model integrations will remain within what these
sources substantiate. Every slide with a non-trivial claim or external asset
will contain a `[Sources]` block in its speaker notes. The deck will distinguish
stable public 0.4 behavior from development-only work and will not imply that
CAMulator compatibility is pinned.

## Quality and acceptance criteria

The finished artifact must:

- be a local `.pptx` created with `@oai/artifact-tool`;
- contain 14 coherent slides suitable for a 20–25 minute developer-oriented
  talk;
- include speaker notes with a complete talk track, transitions, and sources;
- correctly represent VerCOR's stable architecture and current limitations;
- make JAX and automatic differentiation accessible to a mixed-familiarity
  audience;
- provide a concrete four-contract model-wrapping example and staged validation
  path;
- use Mermaid Chart during diagram development;
- render without clipping, unintended overlap, broken connectors, unresolved
  placeholders, or unexpected title wrapping;
- pass the presentation overflow test; and
- be visually inspected slide by slide at full size before delivery.

## Out of scope

- A live coding demonstration.
- A complete API or plugin-authoring tutorial.
- Performance benchmarks not supported by repository evidence.
- A claim that host-side or output-enabled workflows are differentiable.
- Detailed private runtime, CI, packaging, or release-engineering architecture.
