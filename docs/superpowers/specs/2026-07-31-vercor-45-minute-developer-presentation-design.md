# VerCOR 45-Minute Developer Presentation Design

## Purpose

Create a 45-minute PowerPoint presentation about VerCOR for Earth system
modelers. The slot includes approximately 36 minutes of prepared content and
9 minutes for questions. Most attendees are developers, and some may want to
integrate their own models as VerCOR components. The deck must explain why
explicit coupling matters scientifically while giving prospective integrators
a practical and accurate path into the stable public VerCOR 0.4 contracts.

Communication job:

> By the end, developers should understand how VerCOR composes Earth-system
> components, where runtime responsibilities begin and end, what they must
> implement to wrap a model, and how JAX and automatic differentiation affect
> execution and validation choices.

The audience has mixed familiarity with JAX and automatic differentiation.
The deck will introduce their consequences in plain language before using
JAX-specific terms.

## Approved approach

Use the approved architecture-to-integration walkthrough. An
atmosphere-ocean-land-sea-ice example will recur across the deck so that each
abstraction maps back to a recognizable coupled system.

The narrative has four acts:

1. Coupling choices are part of the scientific experiment.
2. VerCOR makes components, exchanges, execution, and state explicit.
3. A model can be integrated through a small public contract and staged
   validation.
4. Differentiability enables new experiments but has explicit capability
   boundaries.

This approach gives architecture and integration the majority of the time,
while preserving enough scientific motivation and JAX context for a mixed
audience.

## Pacing

The deck contains 18 slides and targets approximately 36 minutes of prepared
content, followed by approximately 9 minutes of questions.

- Scientific motivation and coupling choices: about 6 minutes.
- VerCOR architecture and execution: about 11 minutes.
- Model integration and validation: about 12 minutes.
- JAX, automatic differentiation, boundaries, and close: about 7 minutes.

Speaker notes will contain a natural talk track, slide-level pacing,
transitions, and sources. Timing scaffolds will not appear in audience-facing
slide content.

## Slide sequence

1. **VerCOR: differentiable coupling for Earth-system models** — minimal title
   and one-sentence promise.
2. **Coupling decisions belong in the experiment** — fields, grids, clock,
   ordering, and boundary conventions.
3. **Traditional coupling hides consequential choices** — explain how implicit
   state, transfer logic, and execution order hinder validation.
4. **Four owners keep coupling explicit** — components, exchanges, runtime,
   and output.
5. **A coupled experiment is an explicit graph** — atmosphere, ocean, land,
   sea ice, and forcing connected by named exchanges.
6. **Every clock step follows receive, step, send** — shared-clock execution,
   run-order consequences, and boundary validation.
7. **The runtime coordinates; components own physics** — map responsibility
   boundaries for state, validation, transfer, scheduling, and physics.
8. **A component is four public contracts** — `name`, `grid`, `spec`, and
   `step`.
9. **A wrapper can remain structurally small** — one compact public-interface
   example annotated by contract rather than taught line by line.
10. **Evolving model state belongs in the payload** — distinguish static
    author configuration from immutable per-run state.
11. **Exchanges make transfer choices inspectable** — route identity,
    scalar/vector regridding, masks, topology, and rejected ambiguous fan-in.
12. **Workflows turn the experiment into an execution plan** — scheduling,
    run order, backend selection, and output boundaries.
13. **JAX-native and host components can coexist** — compare compiled
    differentiable execution with ordinary Python or external-model execution.
14. **Automatic differentiation traces cause through the system** — explain a
    parameter-to-simulation-to-diagnostic-to-gradient path in plain language.
15. **Differentiability has explicit boundaries** — output-free JAX workflows,
    stable PyTree structure, traced physics values, and unsupported paths.
16. **Integrate in stages** — isolated kernel, fields and grid, component,
    exchange, and short coupled run.
17. **Validate physics and gradients from the bottom up** — boundary checks,
    reference tests, finite differences, and forward/reverse agreement.
18. **Make coupling an inspectable part of your model** — integration
    checklist, current boundaries, documentation, and transition to questions.

## Content balance

- Approximately 55% architecture and execution.
- Approximately 25% component integration and validation.
- Approximately 10% scientific motivation.
- Approximately 10% JAX and automatic differentiation concepts.

Two compact code-oriented slides are permitted: the four-contract component
surface and one structural wrapper example. All other slides should favor
diagrams, concise comparisons, or takeaway statements. The deck is an
architectural and integration orientation, not a complete API tutorial or live
coding workshop.

## Visual system

Use the bundled Codex Grid layout library as the composition reference because
no external template was supplied. Adapt it to a restrained Earth-system
palette:

- deep ocean blue for structure and backgrounds;
- atmosphere teal;
- land ochre;
- sea-ice pale cyan;
- coral for constraints and capability boundaries; and
- warm off-white for explanatory slides.

Typography will meet the presentation requirements: at least 50 pt for the
deck title, 35 pt for slide titles, 24 pt for subheads, and 16 pt for body
text. Slides will use takeaway titles, equal margins, low text density, and
varied flat compositions rather than dashboard-like card grids.

The deck will avoid decorative stock photography. Visual meaning will come
from strong typography and five focused diagrams:

1. Earth-system component and named-exchange graph.
2. Shared-clock receive-step-send cycle.
3. Runtime-versus-component responsibility boundary.
4. Parameter-to-diagnostic gradient pathway.
5. Staged model-integration and validation flow.

Mermaid Chart will be used to develop and validate the requested graph and
flow logic. The final deck may implement the same logic with editable native
PowerPoint shapes when that renders more clearly. Diagram connectors will be
created before nodes so edges remain behind labels.

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

Every substantive slide will contain a `[Sources]` block in its speaker notes.
Claims about JAX, output, and optional integrations will remain within what the
repository substantiates. The deck will distinguish stable public 0.4 behavior
from development-only work and will not imply that CAMulator compatibility is
pinned.

## Accuracy boundaries

- Use only stable public VerCOR 0.4 interfaces in visible examples.
- State that output-free JAX workflows can remain differentiable end to end;
  do not claim that output-enabled workflows are differentiable.
- Do not imply that host-side or mixed-backend execution is differentiable.
- Explain that compiled execution requires stable payload PyTree structure,
  leaf shapes, and dtypes.
- Describe physical values as traced inputs and precision/execution policy as
  static configuration.
- Avoid performance or scalability claims without repository evidence.
- Describe current optional-model limitations explicitly, including unpinned
  CAMulator compatibility.

## Quality and acceptance criteria

The finished artifact must:

- be a local `.pptx` created with `@oai/artifact-tool`;
- contain exactly 18 coherent slides suitable for a 36-minute developer talk
  plus 9 minutes of questions;
- include speaker notes with a complete talk track, transitions, pacing, and
  `[Sources]` blocks;
- correctly represent VerCOR's stable architecture and current limitations;
- make JAX and automatic differentiation accessible to a mixed-familiarity
  audience;
- provide a concrete four-contract component example and staged integration
  path;
- use Mermaid Chart during diagram development;
- render without clipping, unintended overlap, broken connectors, unresolved
  placeholders, or unexpected title wrapping;
- pass the presentation overflow and unresolved-content checks; and
- be visually inspected slide by slide at full size before delivery.

## Out of scope

- A live coding demonstration.
- A complete API or plugin-authoring tutorial.
- Unsupported performance benchmarks.
- A claim that every mixed VerCOR workflow is differentiable.
- Detailed private runtime, CI, packaging, or release-engineering architecture.

