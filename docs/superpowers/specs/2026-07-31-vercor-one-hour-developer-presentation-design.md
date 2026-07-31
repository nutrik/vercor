# VerCOR One-Hour Developer Presentation Design

## Purpose

Create a one-hour PowerPoint presentation about VerCOR for Earth system
modelers. The slot contains approximately 48 minutes of prepared content and
12 minutes for questions. Most attendees are developers, and some may want to
integrate their own models as VerCOR components. The deck must explain why
explicit coupling matters scientifically while giving prospective integrators
a practical and accurate path into the stable public VerCOR 0.4 contracts.

Communication job:

> By the end, developers should understand how VerCOR composes Earth-system
> components, where runtime responsibilities begin and end, what they must
> implement to wrap a model, and how JAX and automatic differentiation affect
> execution and validation choices.

The audience has mixed familiarity with JAX and automatic differentiation.
The deck will explain their consequences in plain language before introducing
JAX-specific terms.

## Approved narrative approach

Use an architecture-to-integration-to-differentiability walkthrough. An
atmosphere-ocean-land-sea-ice example will recur throughout the deck so that
each abstraction maps to a recognizable coupled Earth-system configuration.

The narrative has four acts:

1. Coupling choices are part of the scientific experiment.
2. VerCOR makes components, exchanges, execution, state, and output explicit.
3. A model can be integrated through a small public contract and staged
   validation.
4. Differentiability enables new experiments but has explicit capability
   boundaries.

This approach gives architecture and integration most of the prepared time,
while preserving enough scientific motivation and JAX context for a mixed
audience.

## Pacing

The deck will contain approximately 22 slides and target 48 minutes of
prepared content followed by 12 minutes of questions.

- Scientific motivation and coupling choices: approximately 7 minutes.
- VerCOR architecture and execution: approximately 14 minutes.
- Model integration and validation: approximately 17 minutes.
- JAX, automatic differentiation, boundaries, and close: approximately
  10 minutes.

Speaker notes will contain a natural talk track, slide-level pacing,
transitions, and sources. Timing scaffolds will not appear in audience-facing
slide content.

## Slide sequence

1. **VerCOR: differentiable coupling for Earth-system models** — minimal title
   and one-sentence promise.
2. **Coupling decisions belong in the experiment** — fields, grids, clock,
   ordering, and boundary conventions.
3. **Hidden coupling creates hidden assumptions** — implicit state, embedded
   transfer logic, and execution order hinder validation.
4. **VerCOR makes five responsibilities explicit** — components, exchanges,
   workflows, runtime state, and output.
5. **A coupled experiment is an explicit graph** — atmosphere, ocean, land,
   sea ice, and forcing connected by named field exchanges.
6. **One clock drives receive, step, and send** — the shared-clock execution
   cycle and consequences of run order.
7. **The runtime coordinates; components own physics** — responsibility
   boundary for validation, transfer, scheduling, state, and model physics.
8. **The stable public surface stays deliberately small** — orient developers
   to the six root exports and canonical extension modules without exposing
   private runtime objects.
9. **A component is four public contracts** — `name`, `grid`, `spec`, and
   `step`.
10. **A wrapper can remain structurally small** — a compact public-interface
    example annotated by contract rather than taught line by line.
11. **Fields and grids define the coupling boundary** — declared inputs,
    outputs, initialization, shapes, dtypes, units, and rectilinear grids.
12. **Evolving model state belongs in the payload** — static author
    configuration versus immutable per-run payload and `StepResult`.
13. **Lifecycle hooks prepare state without hiding it** — setup, prefill,
    validation, and stable identity across lifecycle boundaries.
14. **Exchanges make transfer choices inspectable** — route identity,
    scalar/vector transfer, bilinear/conservative regridding, masks, topology,
    and rejected ambiguous fan-in.
15. **Workflows become validated execution plans** — schedules, run order,
    backend selection, chunks, and output boundaries at the public conceptual
    level.
16. **Output has a single opt-in owner** — providers, cadence, period means,
    snapshots, final fields, and the output-free differentiable path.
17. **JAX changes execution, not the scientific contract** — pure functions,
    PyTrees, JIT compilation, traced values, and static policy in accessible
    language.
18. **Automatic differentiation traces cause through the system** — parameter
    to coupled simulation to diagnostic to gradient.
19. **JAX-native and host components can coexist** — compare compiled,
    differentiable-capable execution with ordinary Python or external-model
    execution.
20. **Integrate in stages** — isolated kernel, fields and grid, component,
    exchange, short coupled run, and optional gradient check.
21. **Validate physics and gradients from the bottom up** — reference tests,
    boundary checks, finite differences, and forward/reverse agreement.
22. **Make coupling inspectable in your model** — integration checklist,
    current boundaries, documentation path, and transition to questions.

The sequence may be adjusted by one slide during production if visual pacing
requires combining or separating adjacent concepts, but the four-act structure
and 48-minute content target will remain unchanged.

## Content balance

- Approximately 50% architecture and execution.
- Approximately 27% component integration and validation.
- Approximately 9% scientific motivation.
- Approximately 14% JAX and automatic differentiation concepts.

Two slides will be code-oriented: the four-contract component surface and one
structural wrapper example. All other slides will favor diagrams, concise
comparisons, progressive sequences, or takeaway statements. The deck is an
architectural and integration orientation, not a complete API tutorial or live
coding workshop.

## Visual system

Use the bundled Codex Grid layout library as the composition reference because
no external template was supplied. Adapt it to a restrained Earth-system
palette:

- deep ocean blue for structure and primary backgrounds;
- atmosphere teal;
- land ochre;
- sea-ice pale cyan;
- coral for constraints and capability boundaries; and
- warm off-white for explanatory and code-oriented slides.

Typography will meet the presentation requirements: at least 50 pt for the
deck title, 35 pt for slide titles, 24 pt for subheads, and 16 pt for body
text. Slides will use takeaway titles, equal margins, low text density, and
varied flat compositions rather than dashboard-like card grids.

The deck will avoid decorative stock photography. Visual meaning will come
from strong typography and five focused diagrams:

1. Earth-system components connected by named field exchanges.
2. Shared-clock receive-step-send execution cycle.
3. Runtime responsibilities versus component responsibilities.
4. Parameter-to-coupled-simulation-to-diagnostic-to-gradient pathway.
5. Staged model integration and validation flow.

Mermaid Chart will be used to develop and validate the requested graph and
flow logic. Final PowerPoint diagrams will remain editable and may use native
PowerPoint shapes when that produces clearer labels and more reliable
rendering. Diagram connectors will be created before nodes so edges remain
behind labels.

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
repository substantiates. External claims or assets, if introduced, must come
from primary sources and be cited in the corresponding slide notes.

## Accuracy boundaries

- Use only stable public VerCOR 0.4 interfaces in visible examples.
- State that output-free JAX workflows can remain differentiable end to end;
  do not claim that output-enabled workflows are differentiable.
- Do not imply that host-side or mixed-backend execution is differentiable.
- Explain that compiled execution requires stable payload PyTree structure,
  leaf shapes, and dtypes.
- Describe physics values as traced inputs and precision/execution policy as
  static configuration.
- Distinguish published `0.4.3` functionality from development-only
  setup-gallery work.
- Describe CAMulator compatibility as optional and not currently pinned.
- Avoid performance or scalability claims without repository evidence.

## Quality and acceptance criteria

The finished artifact must:

- be an editable local `.pptx` created with `@oai/artifact-tool`;
- contain approximately 22 coherent slides suitable for 48 minutes of prepared
  content plus 12 minutes of questions;
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
- pass presentation overflow and unresolved-content checks; and
- be visually inspected slide by slide at full size before delivery.

## Out of scope

- A live coding demonstration.
- A complete API or plugin-authoring tutorial.
- Unsupported performance benchmarks.
- A claim that every mixed VerCOR workflow is differentiable.
- Detailed private runtime, CI, packaging, or release-engineering architecture.
