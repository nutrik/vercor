# VerCOR Research Presentation Design

## Purpose

Create a 20–25 minute PowerPoint presentation introducing VerCOR to Earth
system modelers. Most attendees are researchers rather than scientific-software
developers, while a minority may want to integrate their own model. The deck
must therefore establish the scientific value of explicit, differentiable
coupling before presenting a concise and credible component-integration path.

Communication job:

> By the end, Earth system modelers should understand how VerCOR makes coupling
> an inspectable part of a scientific experiment and know the first steps for
> evaluating whether their model can become a VerCOR component.

The audience is expected to have mixed familiarity with JAX and automatic
differentiation. The presentation will explain the scientific consequence of
differentiability before naming the underlying JAX mechanisms.

## Narrative approach

Use a research-workflow-first learning progression:

1. Coupling choices are part of the scientific hypothesis.
2. VerCOR separates component physics from coupling decisions.
3. Components and exchanges form an explicit experiment graph.
4. A shared clock and receive-step-send cycle govern execution.
5. Regridding, topology, state, and output have clear ownership.
6. Output-free JAX workflows remain differentiable end to end.
7. JAX-native and host-side models can coexist with different capabilities.
8. A model can be integrated by satisfying four explicit contracts and
   validating it in stages.
9. VerCOR's current boundaries are stated plainly.

An atmosphere-ocean-land example will recur across the deck so that the
architecture remains connected to a recognizable Earth-system experiment.

## Slide sequence and pacing

The deck contains 14 slides and targets approximately 22 minutes, leaving
2–3 minutes of flexibility within the requested range.

1. **VerCOR: differentiable coupling for Earth-system experiments** — minimal
   opening and one-sentence promise.
2. **Coupling is part of the scientific hypothesis** — motivate clocks,
   exchanged fields, grids, ordering, and boundary conventions as choices that
   affect results.
3. **VerCOR separates models from coupling decisions** — contrast embedded
   coupling logic with explicit component, exchange, workflow, and output
   ownership.
4. **A coupled experiment is a graph of components and exchanges** — show
   atmosphere, ocean, land, sea ice, and forcing connected by named fields.
5. **One clock advances a transparent receive-step-send cycle** — explain the
   runtime at a conceptual level.
6. **Regridding and topology are explicit scientific choices** — distinguish
   scalar/vector transfer, bilinear/conservative methods, and masks without
   presenting implementation detail.
7. **Immutable state makes experiments traceable and transformable** — connect
   explicit state to reproducibility, validation, checkpoint reasoning, and JAX
   transformations.
8. **Differentiability opens new experimental methods** — explain sensitivity
   analysis, gradient-based calibration, and differentiable surrogate/model
   experiments; avoid promising that every mixed-backend experiment is
   differentiable.
9. **JAX-native and host-side models can coexist—with clear trade-offs** — show
   capability boundaries for compiled differentiable and host execution.
10. **VerCOR already includes useful building blocks** — present slab models,
    forcing data, JCM, Veros, and CAMulator as examples, with optional
    dependency and current-compatibility caveats.
11. **Wrapping a model requires four explicit contracts** — identity/grid,
    field specification, explicit state/payload, and a functional step.
12. **Integrate in stages, then couple with confidence** — isolated component
    tests, field and grid validation, exchange validation, short coupled run,
    and gradient validation where applicable.
13. **VerCOR enables new experiments within clear boundaries** — synthesize
    strengths and current limits, including rejected ambiguous fan-in,
    opt-in output, and the distinction between JAX-native and host workflows.
14. **Make coupling an inspectable part of the experiment** — close with three
    audience actions: explore a bundled setup, map a model to the four
    contracts, and identify a coupling question worth testing.

Speaker notes will provide a natural talk track, slide-level pacing, transition
sentences, and source blocks. Timings will not appear on audience-facing slides.

## Content balance

- Approximately 65% scientific motivation, experimental meaning, and conceptual
  architecture.
- Approximately 25% integration pathway and component contracts.
- Approximately 10% JAX/runtime mechanics needed to understand capability
  boundaries.

Only one compact code example will appear. It will be annotated as a contract,
not presented as a programming tutorial. The deck will avoid private VerCOR
internals and use the stable public 0.4 interface as the integration boundary.

## Visual system

Use the bundled Codex Grid layout library as the composition reference because
no external template was supplied. Preserve its strong hierarchy and varied
silhouettes while adapting the palette to Earth-system science:

- deep ocean blue for the base;
- atmosphere teal;
- land ochre;
- sea-ice pale cyan;
- warm coral for cautions and current boundaries;
- warm off-white backgrounds for high-density explanatory slides.

Typography will meet the presentation skill minimums: at least 50 pt for the
deck title, 35 pt for slide titles, 24 pt for subheads, and 16 pt for body copy.
Slides will use takeaway titles, low text density, equal margins, and a flat
research-presentation composition rather than dashboard-like card grids.

The deck will avoid decorative stock photography. Visual meaning will come from
typography, restrained color, native charts or shapes where appropriate, and
four focused diagrams:

1. Earth-system component and exchange graph.
2. Shared-clock receive-step-send flow.
3. Gradient pathway from parameter through coupled simulation to diagnostic.
4. Staged model-integration flow.

Mermaid Chart will be used to draft and validate the graph/flow logic requested
by the user. Final PowerPoint diagrams may use native shapes if that produces
clearer labels and more reliable slide rendering.

## Evidence and source policy

Primary content sources are the repository's implemented architecture and
maintained documentation:

- `DESIGN.md`
- `README.md`
- `PROGRESS.md`
- `DEPENDENCIES.md`
- `docs/plugin-authoring.md`
- the researcher and developer documentation under `docs/`
- relevant public API docstrings and examples

Claims about JAX or bundled external models will be limited to what VerCOR's
documentation and implementation substantiate. If external claims or assets are
introduced, they must come from primary sources. Every slide with a non-trivial
claim or externally sourced asset will contain a `[Sources]` block in speaker
notes.

The deck will distinguish the currently published `0.4.3` interface from
development-only setup-gallery functionality and will not imply that CAMulator
compatibility is fully pinned.

## Quality and acceptance criteria

The finished artifact must:

- be a local `.pptx` created with `@oai/artifact-tool`;
- contain 14 coherent slides suitable for a 20–25 minute research talk;
- include speaker notes with a complete talk track and source blocks;
- correctly represent VerCOR's stable architecture and current limitations;
- explain differentiability accessibly to a mixed-familiarity audience;
- give prospective integrators a concrete four-contract, staged pathway;
- use the requested Mermaid capability during diagram development;
- render every slide without clipping, unintended overlap, broken connectors,
  unresolved placeholders, or unexpected title wrapping;
- pass the presentation overflow test; and
- be visually inspected slide by slide at full size before delivery.

## Out of scope

- A live coding demonstration.
- A complete API tutorial.
- Performance benchmarks not already supported by repository evidence.
- A claim that host-side or output-enabled workflows are differentiable.
- Detailed internal runtime, CI, packaging, or release-engineering architecture.
