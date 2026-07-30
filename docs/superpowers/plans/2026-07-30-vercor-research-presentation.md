# VerCOR Research Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a 14-slide PowerPoint deck that introduces VerCOR to Earth system researchers and gives prospective integrators a credible first pathway.

**Architecture:** A single JavaScript ES module will create the deck with `@oai/artifact-tool`, using selected Codex Grid layouts as composition references and named helper functions for repeated typography, notes, footers, and diagrams. Repository documentation supplies the scientific claims; Mermaid Chart validates the requested graph and process logic before those diagrams are recreated as editable native PowerPoint shapes.

**Tech Stack:** `@oai/artifact-tool`, JavaScript ES modules, Mermaid Chart, bundled presentation render/overflow tools, LibreOffice/Poppler through the presentation runtime.

## Global Constraints

- Final deck: `/Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx`.
- Build workspace: `/Users/romannuterman/Documents/Science/scodes/Python/vercor/.codex/vercor-research-presentation`.
- Exactly 14 slides, paced for approximately 22 minutes within a 20–25 minute talk.
- Audience balance: approximately 65% scientific motivation and conceptual architecture, 25% model integration, and 10% JAX/runtime mechanics.
- Use the stable public 0.4 interface and distinguish published `0.4.3` behavior from development-only setup-gallery functionality.
- Do not imply that host-side or output-enabled workflows are differentiable.
- Use at least 50 pt for the deck title, 35 pt for slide titles, 24 pt for subheads, and 16 pt for body text.
- Include complete speaker notes with slide-level talk track, pacing, transitions, and `[Sources]` blocks.
- Preserve selected Codex Grid layout hierarchy and media frames while adapting content.
- Create connectors before diagram nodes so arrows remain behind entities.
- Render and visually inspect every slide; fix every unintended overlap, clipping, wrapping, and connector error.

---

### Task 1: Source ledger and diagram logic

**Files:**
- Create: `.codex/vercor-research-presentation/source-notes.txt`
- Create: `.codex/vercor-research-presentation/diagram-notes.txt`

**Interfaces:**
- Consumes: `DESIGN.md`, `README.md`, `PROGRESS.md`, `DEPENDENCIES.md`, `docs/plugin-authoring.md`, `docs/researchers/`, `docs/developers/`, and public examples.
- Produces: claim-to-source mappings and four validated Mermaid diagrams used by the deck builder.

- [ ] **Step 1: Initialize the artifact-tool workspace**

Run:

```bash
/Users/romannuterman/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  /Users/romannuterman/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/setup_artifact_tool_workspace.mjs \
  --workspace /Users/romannuterman/Documents/Science/scodes/Python/vercor/.codex/vercor-research-presentation
```

Expected: the workspace contains a usable `node_modules/@oai/artifact-tool`.

- [ ] **Step 2: Write the source ledger**

Create `source-notes.txt` with one section per slide. Each section records the
exact repository file and heading supporting every non-trivial claim. Record
the local file paths used in notes and the public documentation URL only when
it is already present in the repository.

- [ ] **Step 3: Draft the four Mermaid diagrams**

Create and display these diagrams through Mermaid Chart:

1. Component graph: `Forcing`, `Atmosphere`, `Ocean`, `Land`, and `Sea ice`
   connected by named scalar/vector exchanges.
2. Runtime cycle: `Receive → Step → Send → Validate/output boundary → Advance
   clock`.
3. Gradient path: `Parameters → Coupled simulation → Diagnostic → Gradient →
   Updated experiment`, with a boundary note for host/output paths.
4. Integration stages: `Isolated component → Field/grid validation → Exchange
   validation → Short coupled run → Gradient validation where applicable`.

Expected: all four diagrams render without Mermaid syntax errors and their
final Mermaid text is recorded in `diagram-notes.txt`.

- [ ] **Step 4: Verify the evidence boundary**

Run:

```bash
grep -nE "performance|speedup|faster|benchmark|fully differentiable|CAMulator" \
  .codex/vercor-research-presentation/source-notes.txt
```

Expected: each matched claim is either directly supported by repository
documentation or explicitly qualified; no unsupported benchmark claim exists.

### Task 2: Build the complete editable deck

**Files:**
- Create: `.codex/vercor-research-presentation/build-vercor-deck.mjs`
- Create: `.codex/vercor-research-presentation/rendered/`
- Create: `/Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx`

**Interfaces:**
- Consumes: Task 1 source and diagram notes plus the selected Codex Grid slide
  modules `slide-01`, `slide-02`, `slide-04`, `slide-07`, `slide-08`,
  `slide-09`, `slide-10`, `slide-13`, `slide-15`, `slide-17`, and `slide-26`.
- Produces: the editable 14-slide PowerPoint, one PNG and layout JSON per slide,
  a montage, and an inspect snapshot.

- [ ] **Step 1: Implement deck-level helpers**

In `build-vercor-deck.mjs`, import `Presentation` and `PresentationFile` from
`@oai/artifact-tool`. Define these exact helpers:

```javascript
function addTitle(slide, title, slideNumber, options = {})
function addFooter(slide, slideNumber)
function addTextBox(slide, name, text, position, style = {})
function addSourceNotes(slide, talkTrack, sources)
function addNode(slide, name, label, position, fill, textColor = COLORS.ink)
function connectNodes(slide, source, target, options = {})
async function writeBlob(path, blob)
```

Create a `1280 × 720` presentation, Helvetica Neue/Arial typography, the
approved Earth-system palette, and a white/off-white canvas consistent with
Codex Grid.

- [ ] **Step 2: Implement slides 1–3**

Use:

- slide 1: the `slide-01` sparse cover hierarchy;
- slide 2: the `slide-02` large-message hierarchy with five coupling choices
  placed as a restrained evidence rail;
- slide 3: the `slide-04` two-column hierarchy contrasting embedded and
  explicit coupling ownership.

Add 1.25–1.75 minutes of speaker notes per slide and cite `README.md` and
`DESIGN.md` where applicable.

- [ ] **Step 3: Implement slides 4–6**

Create connectors before nodes.

- slide 4: editable component graph with named field exchanges;
- slide 5: shared-clock receive-step-send cycle;
- slide 6: two-column regridding/topology explanation with bilinear,
  conservative, scalar/vector, and mask distinctions.

Keep each diagram label to at most four words and cite the coupling/topology
sections of `DESIGN.md`.

- [ ] **Step 4: Implement slides 7–9**

- slide 7: immutable state visual using a before/after state sequence;
- slide 8: gradient pathway with accessible plain-language callouts;
- slide 9: JAX-native versus host-side capability comparison.

The slide 8 notes must say that output-free JAX workflows can remain
differentiable end to end. The slide 9 notes must explicitly distinguish
compiled differentiable execution from host execution.

- [ ] **Step 5: Implement slides 10–12**

- slide 10: building blocks overview covering slab components, forcing data,
  JCM, Veros, and CAMulator with compatibility caveats;
- slide 11: four contracts with one compact annotated structural-component
  example;
- slide 12: five-stage integration pathway.

The code example must use only public concepts (`name`, `grid`, `spec`, and
`step`) and remain readable at 18 pt or larger.

- [ ] **Step 6: Implement slides 13–14**

- slide 13: strengths on the left and current boundaries on the right;
- slide 14: close with three audience actions, using the `slide-26` closing
  hierarchy rather than a generic “Thank you” ending.

- [ ] **Step 7: Add notes and export all artifacts**

For every slide, set visible speaker notes containing:

1. a natural talk track;
2. a pacing line;
3. a transition line; and
4. a `[Sources]` block.

Export one PNG and one layout JSON per slide, a montage, an inspect snapshot,
and the final `.pptx`.

- [ ] **Step 8: Run the deck builder**

Run:

```bash
cd /Users/romannuterman/Documents/Science/scodes/Python/vercor/.codex/vercor-research-presentation
/Users/romannuterman/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node \
  build-vercor-deck.mjs
```

Expected: exit code 0; the final PPTX and all 14 slide previews exist.

### Task 3: Render, inspect, and refine

**Files:**
- Modify: `.codex/vercor-research-presentation/build-vercor-deck.mjs`
- Regenerate: `.codex/vercor-research-presentation/rendered/`
- Regenerate: `/Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx`

**Interfaces:**
- Consumes: Task 2 deck and render artifacts.
- Produces: a visually corrected deck with no unintended overflow or overlap.

- [ ] **Step 1: Render the exported PowerPoint independently**

Run:

```bash
/Users/romannuterman/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/romannuterman/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/render_slides.py \
  /Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx
```

Expected: 14 independent slide PNGs.

- [ ] **Step 2: Create and inspect the montage**

Run:

```bash
/Users/romannuterman/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/romannuterman/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/create_montage.py \
  --input_dir /Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers \
  --output_file /Users/romannuterman/Documents/Science/scodes/Python/vercor/.codex/vercor-research-presentation/rendered/final-montage.png
```

Inspect the montage for narrative rhythm, repeated silhouettes, palette drift,
and inconsistent footers.

- [ ] **Step 3: Inspect every slide at full size**

Open each of the 14 independent PNGs. Record issues in
`.codex/vercor-research-presentation/qa-ledger.txt` using the format:

```text
slide N | object/region | issue | correction
```

Correct title wrapping, text clipping, font size, connector routing, label
collision, inconsistent alignment, and blurry or unresolved assets.

- [ ] **Step 4: Run overflow and content checks**

Run:

```bash
/Users/romannuterman/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  /Users/romannuterman/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations/container_tools/slides_test.py \
  /Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx
```

Run:

```bash
grep -RniE "lorem|placeholder|TODO|TBD|undefined|NaN" \
  .codex/vercor-research-presentation/rendered \
  .codex/vercor-research-presentation/inspect.ndjson
```

Expected: no overflow errors and no unresolved placeholder text.

- [ ] **Step 5: Regenerate until clean**

Re-run Task 2 Step 8 and Tasks 3 Steps 1–4 after every correction. Stop only
when the independent render, all full-size slide inspections, the montage, and
the overflow test are clean.

### Task 4: Final evidence and project handoff

**Files:**
- Modify: `PROGRESS.md`
- Verify: `/Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx`

**Interfaces:**
- Consumes: the clean Task 3 artifact and QA ledger.
- Produces: a concise progress record and verified final deliverable.

- [ ] **Step 1: Record completion in `PROGRESS.md`**

Add a dated current-status entry stating that the 14-slide VerCOR research deck
was created, that Mermaid validated the diagram logic, and that all slides
passed independent rendering, full-size visual inspection, and overflow checks.
Do not claim codebase test changes.

- [ ] **Step 2: Verify final file and slide count**

Run:

```bash
ls -lh /Users/romannuterman/.codex/visualizations/2026/07/30/019fb474-dabb-7182-b651-f98b0fc13484/vercor-earth-system-modelers.pptx
```

Run a final presentation inspect and confirm exactly 14 slides and speaker notes
on all 14.

- [ ] **Step 3: Check repository scope**

Run:

```bash
git status --short
git diff --check
```

Expected: only the intentional plan/progress documentation changes are tracked;
build intermediates remain under ignored `.codex/`, and the final presentation
is outside the repository.
