# Worked-Example "Trace" Diagrams - Design

Date: 2026-06-17
Status: approved

## Goal

Add a **worked-example trace** to selected lessons: take one concrete input and
walk it through the lesson's method step by step, showing the **actual
intermediate values** (not just structure). This complements the guide's
existing diagrams, which mostly show *structure* (what the pieces are) rather
than *a run* (what one example becomes at each step).

The point a reader should get: "oh, THIS is what the method actually does to a
concrete thing."

## Scope

12 lessons (Tier 1 = strongest fit, Tier 2 = good fit). One trace per lesson,
placed in the main reading flow (not hidden in a "Going deeper" accordion),
authored bilingually (zh + en) with the same per-section `<p>` parity, CJK, and
escaping rules as the rest of the guide.

NOT in scope: Tier-3 lessons (L1/L2/L3 overview narrative; L7/L8/L10/L15/L16/L17
structural; L13/L14 already have step diagrams). No changes to lesson prose
beyond inserting the trace and a 1-2 sentence lead-in.

## Two visual styles

Decision rule: **A by default; C where the method depends on special geometry.**

- **Style A - `.trace` HTML/CSS component (systematized).** A row of "station"
  panels; each station shows a step label + the concrete value as mono cells;
  between stations an arrow carries the operation/formula. One reusable CSS
  block in `src/shell.py`, theme-consistent, dark-mode automatic. Used for
  linear value pipelines.
- **Style C - hybrid: `.trace` wrapper + inline bespoke SVG.** For
  geometry-heavy methods (matrices, state machines, graphs, low-rank
  bottlenecks, memory reinterpretation), hand-draw an inline SVG inside the
  trace wrapper. The SVG must: use the guide's palette via literal colors that
  read on both light and dark backgrounds (or be wrapped so dark mode is
  acceptable), keep text minimal (labels only; prose stays in HTML), and use
  ASCII-only text in the en half.

Pure full-bespoke-everything (old "B") is NOT chosen - A is the systematic base.

## Per-lesson assignment

| Lesson | Worked example | Style |
|---|---|---|
| L4 attention | one token's Q scored against history K -> softmax -> weighted sum of V (with numbers) | C (score bars + weighting lines) |
| L11 mul_mat | `[3,2] x [3,2]` real numbers, one row.col -> one result cell, k eliminated | C (SVG matrix grids) |
| L12 dequantize | one q4_0 block: bytes -> nibble -> `(q-8)*d` -> floats | A |
| L20 tokenize | `"Hi shi-jie"` -> split/merge -> token ids (incl. byte fallback) | A |
| L21 sampling | logits -> temperature -> top-k -> softmax -> top-p -> sample | A |
| L23 grammar | generate `{"a":1}` char by char: mask illegal -> allow -> advance FSM | C (SVG state machine) |
| L5 transpose | `[3,2]` -> `[2,3]` swap ne/nb, data unchanged | C (SVG grid + memory strip) |
| L6 quant round-trip | 32 weights -> quantize -> dequantize -> error | A |
| L9 build graph | `y = W2*(W1*x)` grows into a DAG as ops are called | C (SVG graph nodes) |
| L19 KV cache | step n -> n+1 -> n+2 cells filling, history reused | A |
| L22 chat template | `[{system},{user}]` -> ChatML string with markers | A |
| L24 LoRA | `x -> W*x + scale*B*(A*x)` low-rank bypass | C (SVG big->small->big bottleneck) |

6 x A (pure component), 6 x C (component + bespoke SVG).

## The `.trace` component (Style A)

New CSS in `src/shell.py` (alongside the existing diagram classes). Markup:

```html
<div class="trace">
  <div class="tcap"><b>追踪：<title></b> <one-line what-this-shows></div>
  <div class="stations">
    <div class="stn"><h5>(1) step name</h5>
      <div class="cellrow"><span class="vc">val</span>...</div>
      <div class="tlab">caption</div></div>
    <div class="op">op<br>param</div>
    <div class="stn">...</div>
    ...
  </div>
</div>
```

Reuses existing value-cell visual language (`.vc` like the current `.cell`,
with `.hot`/`.dim`/`.blue` variants) so it feels native. The wrapper gets a
subtle distinct accent (e.g. a left rule or a "worked example" cap) so readers
register it as a run-through, not a structural schematic.

`.trace` is added to `check_html.py`'s `DIAGRAM_CLASSES` so traces count toward
the visual-density check.

## Constraints (same as the rest of the guide)

- Bilingual: every trace authored in BOTH zh and en; per-`<h2>`-section
  (`<p>`+`<p `) parity stays equal (the trace is a `<div>`, invisible to the
  `<p>` count, so inserting it does not change parity - but the 1-2 sentence
  lead-in `<p>` must be added to BOTH languages).
- zh CJK stays >= 4000 per lesson (the lead-in adds a little zh; if a lesson is
  near the floor, add a sentence of zh prose to stay >= 4000).
- en pure ASCII except card-tag emoji; SVG en text ASCII-only; math uses
  `*`/`x`/`-`/`...` not the unicode forms.
- Escaping: chat markers (`<|im_start|>`) become `&lt;|im_start|&gt;` in SVG and
  HTML; no double-escapes.
- Wall-free preserved (a trace is a visual block, so it also breaks prose runs -
  net positive).
- Placement: insert near the section that introduces the method, after the
  structural diagram if one exists, so the reader sees structure then a run.
- Build + check_html + check_links must stay clean; rebuild regenerates
  lessons/*.html.

## Process

Per the established guide workflow: author directly (or via subagent on
claude-opus-4.8), verify (build/validators/parity/CJK/escaping), then run the
two-stage review (spec-compliance then code-quality, both opus-4.8) per task,
commit, merge `--no-ff`. Likely batched: CSS component first (with one A
prototype to lock the look), then the A lessons, then the C lessons (each SVG
verified individually for dark-mode legibility and ASCII en text).

## Success criteria

- 12 lessons each gain one worked-example trace showing concrete intermediate
  values, in the agreed A/C style.
- All existing checks stay green; bilingual parity, CJK, escaping, wall-free
  intact.
- The traces are visually distinct from structural diagrams and read as
  "one example, stepped through".
