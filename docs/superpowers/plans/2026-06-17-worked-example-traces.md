# Worked-Example Trace Diagrams - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one worked-example "trace" diagram (a concrete input stepped through the method, showing actual intermediate values) to 12 lessons of the llama.cpp visual guide.

**Architecture:** A new `.trace` HTML/CSS component (Style A) added to `src/shell.py`, reused across all 12 lessons; geometry-heavy lessons (Style C) embed a bespoke inline SVG inside the same `.trace` wrapper. Traces are authored bilingually (zh+en) in `src/part2.py`/`part3.py`/`part4.py` and rendered by `build.py`.

**Tech Stack:** Zero-dependency Python static-site generator; pure HTML/CSS (CSS custom properties, dark-mode via `prefers-color-scheme`); inline SVG for Style C. No JS test framework - verification is `check_html.py` + `check_links.py` + bespoke parity/CJK/escaping/wall checks.

---

## Conventions every task follows

- **Bilingual + parity:** Author each trace in BOTH zh and en. The `.trace` block is a `<div>` (invisible to the `<p>` parity check), but the 1-2 sentence **lead-in `<p>`** that introduces it MUST be added to BOTH languages so per-`<h2>`-section (`<p>`+`<p `) counts stay equal zh==en.
- **CJK floor:** zh stays >= 4000 CJK per lesson. The zh lead-in adds a little; if a lesson sits near 4000, add one more zh sentence.
- **en pure ASCII** (except card-tag emoji): in HTML and in SVG `<text>`. Math uses `*` `x` `-` `...`, not the unicode forms. Render `->` as `-&gt;`.
- **Escaping:** chat markers become `&lt;|im_start|&gt;` etc. in both HTML and SVG; verify no double-escape (`&amp;lt;`).
- **Placement:** insert the trace in the main flow, right AFTER the structural diagram/explanation of the method (reader sees structure, then a run).
- **Style C SVG:** `viewBox` + `width="100%"` (responsive via the `.trace svg` rule); literal palette colors that read on light AND dark (prefer mid-tone fills like `#c2630e`/`#2563eb`/`#7c3aed` with white text, neutral strokes `#cdd5df`); labels only - prose stays in HTML.
- **Per-task verification** (the "tests"): from `src/`, `python3 build.py && python3 check_html.py && python3 check_links.py` must be clean (0 errors/0 warnings, 96 links); then the parity/CJK/escaping/wall checks below must pass for the edited lesson.
- **Per-task review:** spec-compliance subagent then code-quality subagent (both `claude-opus-4.8`) before commit, per the guide's established workflow.

**Standard verification snippet** (run from repo root, substitute the lesson number N and its module):

```bash
cd /home/verden/course/llama-cpp-visual-guide && python3 - <<'PY'
import re,sys; sys.path.insert(0,'src')
import part2,part3,part4
mods={**{n:part2 for n in range(4,8)},**{n:part3 for n in range(8,14)},**{n:part4 for n in range(14,25)}}
def cjk(s): return len(re.findall(r'[\u4e00-\u9fff]',s))
N=__import__('os').environ.get('N')  # set N before running, or hardcode
PY
```

(In practice each task uses the same inline Python used throughout the guide: split each lesson's zh/en on `<h2`, compare `<p>`+`<p ` counts; assert `cjk(zh)>=4000`, `cjk(en)==0`; grep rendered HTML for `&amp;lt;`/raw `<|`; and the wall scanner asserting max top-level `<p>` run <= 3.)

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/shell.py` | CSS design system | **Modify**: add `.trace` component CSS before line 384 (end of CSS string) |
| `src/check_html.py` | structural validator | **Modify**: add `"trace"` to `DIAGRAM_CLASSES` (line 42) |
| `src/part2.py` | lessons L4-L7 | **Modify**: add traces to L4, L5, L6 |
| `src/part3.py` | lessons L8-L13 | **Modify**: add traces to L9, L11, L12 |
| `src/part4.py` | lessons L14-L24 | **Modify**: add traces to L19, L20, L21, L22, L23, L24 |
| `lessons/*.html` | generated output | **Regenerated** by build.py (committed artifacts) |

No new files. 14 tasks: Task 1 (CSS component + L21 prototype), Tasks 2-6 (remaining A traces), Tasks 7-12 (C traces), Task 13 (full-guide verification), Task 14 (finish/merge).

---

## Task 1: `.trace` CSS component + L21 sampling prototype (Style A)

Establishes the reusable component and locks the visual look with the first trace.

**Files:**
- Modify: `src/shell.py` (insert CSS before the `"""` that closes the CSS block at line 384)
- Modify: `src/check_html.py:42` (register `trace` as a diagram class)
- Modify: `src/part4.py` (LESSON_21, both zh and en - insert the trace in the "从 logits 到 token" / "From logits to a token" section, after its existing flow/structural explanation)

- [ ] **Step 1: Add the `.trace` CSS** to `src/shell.py`, immediately before the line `"""` that ends the CSS string (currently line 384, right after the `.timeline .tslot.now {...}` rule):

```css
/* ---- worked-example trace: one concrete input, stepped through ---- */
.trace { margin: 1.3rem 0; background: var(--panel); border: 1px solid var(--line);
  border-left: 4px solid var(--accent); border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.trace .tcap { font-size: .82rem; color: var(--muted); margin-bottom: .7rem; }
.trace .tcap b { color: var(--accent-ink); }
.trace .stations { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; }
.trace .stn { flex: 1 1 0; min-width: 116px; border: 1px solid var(--line); border-radius: 10px;
  padding: .55rem; background: var(--bg); }
.trace .stn h5 { margin: 0 0 .45rem; font-size: .8rem; color: var(--ink); }
.trace .cellrow { display: flex; gap: .3rem; align-items: center; flex-wrap: wrap; }
.trace .vc { min-width: 2.1rem; padding: .32rem .45rem; text-align: center; border-radius: 7px;
  background: var(--panel-2); border: 1px solid var(--line); font: 600 .76rem ui-monospace, monospace; white-space: nowrap; }
.trace .vc.hot  { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); }
.trace .vc.blue { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); }
.trace .vc.dim  { opacity: .42; }
.trace .tlab { font-size: .68rem; color: var(--faint); margin-top: .35rem; }
.trace .op { align-self: center; color: var(--accent); font: 700 .72rem ui-monospace, monospace;
  padding: 0 .5rem; text-align: center; white-space: nowrap; }
.trace svg { max-width: 100%; height: auto; display: block; margin: .3rem auto; }
@media (max-width: 640px) { .trace .stations { flex-direction: column; } .trace .op { padding: .3rem 0; } }
```

- [ ] **Step 2: Register `trace` as a diagram class** in `src/check_html.py:42`:

```python
DIAGRAM_CLASSES = ("layers", "vflow", "flow", "cols", "cellgroup", "timeline", "trace")
```

- [ ] **Step 3: Insert the L21 trace (zh)** into `src/part4.py` LESSON_21, in the "从 logits 到 token" section, after the paragraph/flow that explains the sampling pipeline. Markup (concrete worked example):

```html
<div class="trace">
  <div class="tcap"><b>追踪一次采样</b>：5 个候选词，看一排 logits 怎么一步步变成最终选中的一个 token（数字为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① logits</h5>
      <div class="cellrow"><span class="vc">3.2</span><span class="vc">2.1</span><span class="vc">1.0</span><span class="vc">0.5</span><span class="vc">-0.3</span></div>
      <div class="tlab">cat / dog / sky / run / blue</div></div>
    <div class="op">÷T<br>T=0.7</div>
    <div class="stn"><h5>② 温度缩放</h5>
      <div class="cellrow"><span class="vc">4.6</span><span class="vc">3.0</span><span class="vc">1.4</span><span class="vc">0.7</span><span class="vc">-.4</span></div>
      <div class="tlab">T&lt;1 放大差距</div></div>
    <div class="op">top-k<br>k=3</div>
    <div class="stn"><h5>③ 截断候选</h5>
      <div class="cellrow"><span class="vc hot">4.6</span><span class="vc hot">3.0</span><span class="vc hot">1.4</span><span class="vc dim">0.7</span><span class="vc dim">-.4</span></div>
      <div class="tlab">只留分数最高的 3 个</div></div>
    <div class="op">softmax<br>top-p .9</div>
    <div class="stn"><h5>④ 概率 → 采样</h5>
      <div class="cellrow"><span class="vc blue">.78</span><span class="vc blue">.18</span><span class="vc dim">.04</span></div>
      <div class="tlab">按概率抽一个 → <strong>cat</strong></div></div>
  </div>
</div>
```

- [ ] **Step 4: Insert the L21 trace (en)** into LESSON_21's English half, same section, parallel content:

```html
<div class="trace">
  <div class="tcap"><b>Tracing one sampling step</b>: 5 candidate words - watch a row of logits become the single chosen token (numbers illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) logits</h5>
      <div class="cellrow"><span class="vc">3.2</span><span class="vc">2.1</span><span class="vc">1.0</span><span class="vc">0.5</span><span class="vc">-0.3</span></div>
      <div class="tlab">cat / dog / sky / run / blue</div></div>
    <div class="op">/T<br>T=0.7</div>
    <div class="stn"><h5>(2) temperature</h5>
      <div class="cellrow"><span class="vc">4.6</span><span class="vc">3.0</span><span class="vc">1.4</span><span class="vc">0.7</span><span class="vc">-.4</span></div>
      <div class="tlab">T&lt;1 widens gaps</div></div>
    <div class="op">top-k<br>k=3</div>
    <div class="stn"><h5>(3) truncate</h5>
      <div class="cellrow"><span class="vc hot">4.6</span><span class="vc hot">3.0</span><span class="vc hot">1.4</span><span class="vc dim">0.7</span><span class="vc dim">-.4</span></div>
      <div class="tlab">keep the top 3 only</div></div>
    <div class="op">softmax<br>top-p .9</div>
    <div class="stn"><h5>(4) probs -&gt; sample</h5>
      <div class="cellrow"><span class="vc blue">.78</span><span class="vc blue">.18</span><span class="vc dim">.04</span></div>
      <div class="tlab">draw one by probability -&gt; <strong>cat</strong></div></div>
  </div>
</div>
```

- [ ] **Step 5: Add a bilingual lead-in `<p>`** right before each trace (one sentence, added to BOTH zh and en in the same section to preserve parity), e.g. zh: `<p>把这条链路用一个具体例子走一遍就清楚了：</p>` / en: `<p>Walking one concrete example through this pipeline makes it click:</p>`

- [ ] **Step 6: Build and verify.**

Run: `cd src && python3 build.py && python3 check_html.py && python3 check_links.py`
Expected: `Wrote 25 files`, `0 error(s), 0 warning(s)`, `all 96 internal links resolve`.

- [ ] **Step 7: Verify parity/CJK/escaping/wall for L21.** Run the standard inline Python: assert L21 per-section `<p>` parity zh==en, `cjk(zh)>=4000`, `cjk(en)==0`, no `&amp;lt;` in `lessons/21-sampling.html`, max top-level `<p>` run <= 3.
Expected: all pass.

- [ ] **Step 8: Visual self-check.** Open `lessons/21-sampling.html`; confirm the trace renders as 4 stations with arrows, value cells legible, distinct from the existing structural diagrams, and acceptable in dark mode (toggle OS theme or check the dark CSS vars apply).

- [ ] **Step 9: Commit.**

```bash
git add src/shell.py src/check_html.py src/part4.py lessons/
git commit -m "feat: add .trace component + L21 sampling worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Task 2: L12 dequantize trace (Style A)

One q4_0 block: raw bytes -> split nibble -> `(q-8)*d` -> float. Reinforces the lesson's core formula with a concrete block.

**Files:**
- Modify: `src/part3.py` (LESSON_12, both langs - in the "解量化：把字节还原成浮点" / "Dequantize: restoring bytes to floats" section, after the `x = (q-8)*d` pseudocode/flow)

- [ ] **Step 1: Design the example values.** One q4_0 block, scale `d = 0.05`. Take 4 sample nibbles: codes `6, 9, 3, 12` -> `(q-8)`: `-2, +1, -5, +4` -> `*d`: `-0.10, +0.05, -0.25, +0.20`. (Consistent with the lesson's `x=(q-8)*d` and L12's q4_0 layout.)

- [ ] **Step 2: Author zh trace** in `src/part3.py` LESSON_12. Use 4 stations: `① 字节 qs[j]` (show two hex bytes, e.g. `0x96 0xC3`) -> `② 拆 nibble` (codes `6 9 3 12`) -> `③ 减 8` (`-2 +1 -5 +4`) -> `④ × d (d=0.05)` (`-.10 +.05 -.25 +.20`, mark as `<span class="vc blue">`). `.tcap`: `<b>追踪一次解量化</b>：一个 q4_0 块的几个字节怎么还原成浮点（d=0.05 为示意）。` Ops on arrows: `& 0x0F / >>4`, `q-8`, `× d`.

- [ ] **Step 3: Author en trace** (parallel): stations `(1) byte qs[j]` -> `(2) split nibble` -> `(3) minus 8` -> `(4) x d (d=0.05)`. `.tcap`: `<b>Tracing one dequant</b>: how a few bytes of a q4_0 block become floats (d=0.05 illustrative).` Use ASCII: `&amp; 0x0F`, `&gt;&gt;4`, `x d`.

- [ ] **Step 4: Add bilingual lead-in `<p>`** before the trace in both langs (e.g. zh `<p>拿一个真实的块走一遍，"减 8 再乘 scale" 就具体了：</p>` / en `<p>Run one real block through it and "subtract 8, multiply scale" gets concrete:</p>`).

- [ ] **Step 5: Build + verify.** `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` clean; then L12 parity/CJK(zh>=4000)/en-ASCII/escaping/wall checks pass.

- [ ] **Step 6: Commit.**

```bash
git add src/part3.py lessons/
git commit -m "feat: L12 dequantize worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Task 3: L20 tokenize trace (Style A)

A short mixed string -> pieces -> token ids, including the byte-fallback path for an out-of-vocab char.

**Files:**
- Modify: `src/part4.py` (LESSON_20, both langs - in the "几种分词算法" / tokenizer-algorithms section or "特殊 token 与字节回退" / byte-fallback section, after the existing byte-fallback diagram)

- [ ] **Step 1: Design the example.** Input `"Hi 世!"`. Pieces: `"Hi"` -> known token, `" 世"` -> suppose in-vocab token, `"!"` -> known; then show a fallback example: a rare char `"🎉"` -> bytes `<0xF0><0x9F><0x8E><0x89>` -> 4 byte-tokens. Token ids illustrative (e.g. `15043, 1745, 0, 30`).

- [ ] **Step 2: Author zh trace.** Stations: `① 输入串` (`"Hi 世!"`) -> `② 切成片` (`Hi` / ` 世` / `!` cells) -> `③ 查词表 → id` (`15043 1745 30` blue) -> `④ 字节回退` (rare `🎉` -> `<0xF0><0x9F><0x8E><0x89>` -> 4 ids). `.tcap`: `<b>追踪一次分词</b>：一句话怎么变成 token id；遇到词表里没有的字符就拆成 UTF-8 字节（id 为示意）。` Ops: `分词`, `→ id`, `UTF-8 拆字节`.

- [ ] **Step 3: Author en trace** (parallel): `(1) input` -> `(2) split pieces` -> `(3) vocab -> id` -> `(4) byte fallback`. `.tcap`: `<b>Tracing one tokenization</b>: how a sentence becomes token ids; an out-of-vocab char splits into UTF-8 bytes (ids illustrative).` ASCII bytes `&lt;0xF0&gt;...`.

- [ ] **Step 4: Add bilingual lead-in `<p>`** in both langs.

- [ ] **Step 5: Build + verify** (L20 parity/CJK/en-ASCII/escaping/wall). Note: ensure `<` in `<0xF0>` is written `&lt;0xF0&gt;` and check rendered HTML has no raw `<0x` and no `&amp;lt;`.

- [ ] **Step 6: Commit.**

```bash
git add src/part4.py lessons/
git commit -m "feat: L20 tokenize worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Task 4: L6 quantization round-trip trace (Style A)

A few FP16 weights -> quantize (find scale, store codes) -> dequantize -> show the small error. Makes "lossy but tiny" concrete.

**Files:**
- Modify: `src/part2.py` (LESSON_06, both langs - in the "块量化：每块一个 scale" / "Block quantization: one scale per block" section, after its existing cellgroup)

- [ ] **Step 1: Design the example.** 4 original weights `0.46, -0.12, 0.31, -0.40`. block max abs = 0.46 -> `d = max/-8 = -0.0575` (q4_0 ref); codes after round `q = round(w/d)+8`: compute -> e.g. `0(-> -8 mapped) ...`. Simplify for teaching: pick `d = 0.46/7 ~= 0.066`, codes/levels chosen so dequant gives `0.46, -0.13, 0.33, -0.40`; errors `0, 0.01, 0.02, 0`. Keep numbers small and clearly "close".

- [ ] **Step 2: Author zh trace.** Stations: `① 原始权重 (FP16)` (`0.46 -0.12 0.31 -0.40`) -> `② 定 scale d` (`d=0.066`, one amber `<span class="vc scale">`-style cell) -> `③ 存 4-bit 码` (`hot` cells, integer levels) -> `④ 反量化 + 误差` (`0.46 -0.13 0.33 -0.40`, blue; tlab shows `误差 ≈ 0.01`). `.tcap`: `<b>追踪一次量化往返</b>：4 个权重压成 4-bit 再还原，看误差有多小（数字为示意）。` NOTE: the `.scale` cell class already exists in shell.py; `.trace .vc.scale` may need the amber style - if so add it in Task 1's CSS (or reuse `.vc.hot`). Decide during execution; if adding, mirror `.cell.scale`.

- [ ] **Step 3: Author en trace** (parallel): `(1) FP16 weights` -> `(2) pick scale d` -> `(3) store 4-bit codes` -> `(4) dequantize + error`. tlab `error ~= 0.01`. ASCII only (`~=`, not the unicode approx sign).

- [ ] **Step 4: Add bilingual lead-in `<p>`** in both langs.

- [ ] **Step 5: Build + verify** (L6 parity/CJK/en-ASCII/escaping/wall). If a `.vc.scale` style was added to shell.py, re-confirm Task-1 lessons still render fine.

- [ ] **Step 6: Commit.**

```bash
git add src/part2.py lessons/
git commit -m "feat: L6 quantization round-trip worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Task 5: L19 KV cache fill trace (Style A)

Three decode steps (n, n+1, n+2): each step appends one new cell's K/V; prior cells stay byte-identical (reused). Shows "only the new token is computed".

**Files:**
- Modify: `src/part4.py` (LESSON_19, both langs - in the "cell 怎么管：pos 与 seq_id" section, after the existing cells diagram)

- [ ] **Step 1: Design the example.** A single sequence. Step n holds cells `K0 K1 K2` (pos 0,1,2). Step n+1 appends `K3` (pos 3); `K0..K2` unchanged. Step n+2 appends `K4` (pos 4). Highlight the appended cell `hot`, prior cells `dim`/neutral with a "(reused)" tlab.

- [ ] **Step 2: Author zh trace.** Three station rows (use 3 `.stn` as time steps, each a `cellrow`): `① 第 n 步` (`K0 K1 K2` + `<span class="vc hot">+K? no</span>`)... Actually use stations as steps: `① 第 n 步` cells `K0 K1 K2`; op `+token`; `② 第 n+1 步` cells `K0 K1 K2` dim + `K3` hot; op `+token`; `③ 第 n+2 步` cells `K0..K3` dim + `K4` hot. `.tcap`: `<b>追踪 KV cache 增长</b>：每生成一个 token 只新算一格 K/V，前面的原样复用（不重算）。` tlab on dim cells: `复用，未重算`.

- [ ] **Step 3: Author en trace** (parallel): `(1) step n` / `(2) step n+1` / `(3) step n+2`. `.tcap`: `<b>Tracing KV-cache growth</b>: each token computes just one new K/V cell; earlier cells are reused as-is (not recomputed).` tlab `reused, not recomputed`.

- [ ] **Step 4: Add bilingual lead-in `<p>`** in both langs.

- [ ] **Step 5: Build + verify** (L19 parity/CJK/en-ASCII/escaping/wall).

- [ ] **Step 6: Commit.**

```bash
git add src/part4.py lessons/
git commit -m "feat: L19 KV-cache fill worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Task 6: L22 chat-template assembly trace (Style A)

A 2-message list -> apply ChatML template -> the actual marker-laden string. Makes "structure -> flat text" concrete.

**Files:**
- Modify: `src/part4.py` (LESSON_22, both langs - in the "检测与应用" / detection-and-application section, after the template table)

- [ ] **Step 1: Design the example.** Messages `[{system:"You are helpful."},{user:"Hi"}]` -> ChatML -> string `<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n`. ALL markers escaped `&lt;|im_start|&gt;` etc.

- [ ] **Step 2: Author zh trace.** Stations: `① 消息列表` (two cells `{system}` `{user}`) -> `② 套 ChatML 模板` (op) -> `③ 拼出字符串` (a single wide cell or `<pre>`-like mono block showing the escaped marker string) -> `④ 交给 tokenize` (-> L20). `.tcap`: `<b>追踪模板拼接</b>：结构化消息怎么被压平成一串带特殊标记的纯文本（送进 tokenize 前）。` Because step 3 is a long string, it may use a full-width `.stn` with a mono block rather than cells.

- [ ] **Step 3: Author en trace** (parallel). `.tcap`: `<b>Tracing template assembly</b>: how structured messages flatten into one marked-up plain-text string (just before tokenize).`

- [ ] **Step 4: Add bilingual lead-in `<p>`** in both langs.

- [ ] **Step 5: Build + verify** (L22 parity/CJK/en-ASCII/escaping/wall). CRITICAL: verify rendered `lessons/22-chat-templates.html` shows `&lt;|im_start|&gt;` (not raw `<|im_start|>`, not double-escaped `&amp;lt;`).

- [ ] **Step 6: Commit.**

```bash
git add src/part4.py lessons/
git commit -m "feat: L22 chat-template assembly worked-example trace

Assisted-by: GitHub Copilot"
```

---

## Style-C SVG conventions (apply to Tasks 7-12)

- Wrap every SVG inside a `<div class="trace">` with a `.tcap` lead caption, so it shares the component frame and the `.trace svg { max-width:100% }` responsive rule.
- `<svg viewBox="0 0 W H" width="100%" role="img" aria-label="...">`. No fixed pixel width.
- **Dark-mode legible palette** (literal hex, readable on both `#fff` and `#161b22`): fills `#c2630e` (accent), `#2563eb` (blue), `#7c3aed` (purple), `#9aa6b2` (muted) with `#fff` text inside; neutral strokes `#cdd5df`; standalone labels in `#5b6470` (mid-gray reads on both). Avoid near-white fills with dark text or vice-versa. If a fill must be pale, give it a visible stroke.
- en SVG `<text>` is ASCII-only; zh SVG `<text>` may use Chinese. Each task authors zh and en SVG separately (parallel), inserted into the respective language half.
- Escape `<`/`>`/`&` inside `<text>` (`&lt;` `&gt;` `&amp;`).
- Keep text short - labels and numbers only; explanatory prose stays in the HTML lead-in `<p>` and surrounding paragraphs.
- Parity: the SVG is inside a `<div>` (parity-invisible); only the bilingual lead-in `<p>` affects `<p>` counts - add it to both langs.

---

## Task 7: L11 mul_mat matrix trace (Style C)

A real `[k=3,m=2] x [k=3,n=2]` multiply with numbers; highlight one row of A and one column of B feeding one result cell; annotate "inner dim k eliminated".

**Files:**
- Modify: `src/part3.py` (LESSON_11, both langs - in the "头号算子：矩阵乘 mul_mat" section, after the existing `cellgroup` shape-inference diagram)

- [ ] **Step 1: Design the example.** a (ne=[3,2]) columns -> rows for display; b (ne=[3,2]); pick small ints. Show `a` as a 2x3 grid `[[1,0,2],[ -1,3,1]]`, `b` as 3x2-ish, result `c` ne=[2,2]. Compute one cell `c[0,0] = a_row0 . b_col0 = 1*2 + 0*1 + 2*0 = 2` (use ggml's "inner ne[0]=k summed" framing). Keep one highlighted row x column -> one highlighted result cell.

- [ ] **Step 2: Author zh SVG** inside a `.trace`. SVG ~ `viewBox="0 0 620 220"`. Three grids: A (left), B (top-right), C (bottom-right result), with the multiply laid out so the highlighted row of A and column of B visually point into the highlighted C cell (use light connector lines `#cdd5df`). Numbers in `<text>`. Annotate `k=3 被消去` near the summed dimension. Skeleton:

```html
<div class="trace">
  <div class="tcap"><b>追踪一次矩阵乘</b>：[k=3,m=2] x [k=3,n=2]，看内维 k 怎么被"乘加求和"吃掉，只剩 [m,n]（数字为示意）。</div>
  <svg viewBox="0 0 620 230" width="100%" role="img" aria-label="mul_mat worked example">
    <!-- A grid (2 rows x 3 cols), highlight row 0 -->
    <g font-family="ui-monospace,monospace" font-size="13">
      <rect x="20" y="60" width="150" height="30" fill="#fbeede" stroke="#c2630e"/>
      <text x="30" y="80" fill="#8a4708">1    0    2   (a row0)</text>
      <rect x="20" y="90" width="150" height="30" fill="#fff" stroke="#cdd5df"/>
      <text x="30" y="110" fill="#1d2129">-1   3    1</text>
      <text x="20" y="50" fill="#5b6470">A  ne=[3,2]</text>
    </g>
    <!-- B grid, highlight col 0; C result, highlight c[0,0]=2; connector lines; "k=3 消去" label -->
    <!-- ... author full grid + connectors during execution ... -->
    <text x="300" y="200" font-size="12" fill="#5b6470">内维 k=3 在相乘求和时消去 -&gt; 结果 [m=2, n=2]</text>
  </svg>
</div>
```

(Execution: complete the B grid, C grid, the row0 x col0 connector lines into c[0,0], and the highlighted result cell. Keep it legible.)

- [ ] **Step 3: Author en SVG** (parallel, ASCII text): `.tcap` `<b>Tracing one matmul</b>: [k=3,m=2] x [k=3,n=2] - watch inner dim k get eaten by multiply-and-sum, leaving [m,n] (numbers illustrative).` Label `inner dim k=3 eliminated -&gt; result [m=2, n=2]`.

- [ ] **Step 4: Add bilingual lead-in `<p>`** in both langs.

- [ ] **Step 5: Build + verify** (L11 parity/CJK/en-ASCII/escaping/wall) + dark-mode legibility check on `lessons/11-core-operators.html`.

- [ ] **Step 6: Commit** (`feat: L11 mul_mat matrix worked-example trace`).

---

## Task 8: L4 attention trace (Style C)

One query token scores against 3 history Keys -> softmax weights -> weighted sum of Values. SVG shows score bars, the softmax weights, and weighting lines into the output.

**Files:**
- Modify: `src/part2.py` (LESSON_04, both langs - in the "decoder-only：一个 block 在算什么" section, after the attention "weighted retrieval" analogy card, OR in the causal-mask section after the cellgroup)

- [ ] **Step 1: Design the example.** Query `q` (current token). 3 history tokens with keys; raw scores `q.k = [4.0, 1.0, 2.0]` -> softmax -> weights `[0.71, 0.04, 0.26]` (approx; recompute to ~2dp). Values `v1,v2,v3` (small vectors or just labeled). Output = `0.71*v1 + 0.04*v2 + 0.26*v3`.

- [ ] **Step 2: Author zh SVG** inside `.trace`. Layout: left column = 3 history tokens (k boxes), a "Q" box; middle = score bars (lengths ~ scores) then softmax weight numbers; right = output box with three weighted arrows (stroke-width ~ weight) converging. `.tcap`: `<b>追踪一次注意力</b>：当前 token 拿 Q 给 3 个历史 token 打分 -> softmax 成权重 -> 按权重汇总它们的 V（数字为示意）。` viewBox ~ `0 0 640 230`. Provide the score-bar + weight + converging-arrows skeleton (execution completes geometry).

- [ ] **Step 3: Author en SVG** (parallel, ASCII): `.tcap` `<b>Tracing one attention step</b>: the current token's Q scores 3 history tokens -> softmax to weights -> weighted sum of their V (numbers illustrative).`

- [ ] **Step 4: Bilingual lead-in `<p>`** both langs.

- [ ] **Step 5: Build + verify** (L4 parity/CJK/en-ASCII/escaping/wall) + dark-mode check.

- [ ] **Step 6: Commit** (`feat: L4 attention worked-example trace`).

---

## Task 9: L23 grammar trace (Style C)

Generating `{"a":1}` char by char: at each step the grammar masks illegal tokens, allows the legal one, and advances the FSM. SVG = a small state machine + a mask row.

**Files:**
- Modify: `src/part4.py` (LESSON_23, both langs - in the "语法怎么约束采样" / "How the grammar constrains sampling" section, after the existing flow)

- [ ] **Step 1: Design the example.** Target `{"a":1}`. Steps: state `start` --`{`--> `expectKey` --`"a"`--> `expectColon` --`:`--> `expectValue` --`1`--> `expectEnd` --`}`--> `done`. At step "expectColon", show a mask row: candidate tokens `: , } 5` -> only `:` allowed (others -inf/dim).

- [ ] **Step 2: Author zh SVG** inside `.trace`. Two parts in one SVG: (a) a left-to-right FSM (nodes = states, edges labeled with the accepted char), with the current transition highlighted accent; (b) below it, a "mask" cellrow for the current step: legal token `hot`, illegal tokens `dim` with `-inf`. `.tcap`: `<b>追踪一次语法约束</b>：逐字符生成 {"a":1}，每步把非法 token 砸成 -inf、只放行合法的，并推进状态机。` viewBox ~ `0 0 660 230`. Escape `"`/`{`/`}` as needed in `<text>` (they are fine literally except `&`/`<`/`>`).

- [ ] **Step 3: Author en SVG** (parallel, ASCII): `.tcap` `<b>Tracing one grammar constraint</b>: generating {"a":1} char by char - each step masks illegal tokens to -inf, allows only the legal one, and advances the state machine.`

- [ ] **Step 4: Bilingual lead-in `<p>`** both langs.

- [ ] **Step 5: Build + verify** (L23 parity/CJK/en-ASCII/escaping/wall) + dark-mode check. Verify any `<`/`>` in token labels are escaped.

- [ ] **Step 6: Commit** (`feat: L23 grammar-constraint worked-example trace`).

---

## Task 10: L5 transpose trace (Style C)

A `[3,2]` tensor transposed to `[2,3]`: ne/nb swap but the underlying byte strip is unchanged. SVG shows the grid reinterpreted over the SAME memory strip.

**Files:**
- Modify: `src/part2.py` (LESSON_05, both langs - in the "view / 转置为什么不拷贝数据" section, after the existing `cols` before/after diagram)

- [ ] **Step 1: Design the example.** Original ne=[3,2], nb=[4,12], 6 values `a b c d e f` laid out in memory as `a b c d e f` (offsets 0,4,8,12,16,20). Transposed: ne=[2,3], nb=[12,4], SAME memory `a b c d e f`, but the logical grid is read differently. Show the memory strip once (shared), and two grid "readings" above/below pointing into it.

- [ ] **Step 2: Author zh SVG** inside `.trace`. Top: original 3x2 grid (values). Middle: a single horizontal "memory strip" of 6 cells `a..f` with byte offsets. Bottom: transposed 2x3 grid reading the SAME strip (connector lines from both grids to the same strip cells). `.tcap`: `<b>追踪一次转置</b>：[3,2] 变 [2,3] 只是换了 ne/nb 怎么读这块内存，底层 6 个字节一个都没搬。` viewBox ~ `0 0 560 250`.

- [ ] **Step 3: Author en SVG** (parallel, ASCII): `.tcap` `<b>Tracing one transpose</b>: [3,2] -&gt; [2,3] just changes how ne/nb read this memory; the 6 underlying bytes never move.`

- [ ] **Step 4: Bilingual lead-in `<p>`** both langs.

- [ ] **Step 5: Build + verify** (L5 parity/CJK/en-ASCII/escaping/wall) + dark-mode check.

- [ ] **Step 6: Commit** (`feat: L5 transpose worked-example trace`).

---

## Task 11: L9 build-graph trace (Style C)

`y = W2 * (W1 * x)`: as each op is called it adds a node with back-pointers, growing a small DAG. SVG shows leaves (x,W1,W2) + nodes (h, y) with src edges.

**Files:**
- Modify: `src/part3.py` (LESSON_09, both langs - in the "把张量串成一张图" / "Stringing tensors into a graph" section, after the existing `vflow`)

- [ ] **Step 1: Design the example.** Leaves: `x`, `W1`, `W2` (op NONE). `h = mul_mat(W1, x)` -> node, src=[W1,x]. `y = mul_mat(W2, h)` -> node, src=[W2,h]. Show the DAG with arrows pointing from a node BACK to its srcs (the lesson's "back-pointer" point), and a small "topological order: x,W1,W2 -> h -> y" caption.

- [ ] **Step 2: Author zh SVG** inside `.trace`. Nodes as boxes: leaves (`x W1 W2`, neutral) on one side, `h` and `y` (accent, marked "node op=MUL_MAT") with `src` arrows pointing back to their inputs. `.tcap`: `<b>追踪一次建图</b>：写下 y=W2·(W1·x) 时，每个算子只新建一个节点、用 src 指回输入，于是长成一张有向图（还没开算）。` viewBox ~ `0 0 600 240`. Use `·` is NON-ASCII - in zh it's fine; en uses `*`.

- [ ] **Step 3: Author en SVG** (parallel, ASCII): `.tcap` `<b>Tracing one graph build</b>: writing y=W2*(W1*x), each op just creates a node pointing back at its inputs via src - growing a DAG (nothing computed yet).` Edge labels `src`. ASCII `*` not the unicode dot.

- [ ] **Step 4: Bilingual lead-in `<p>`** both langs.

- [ ] **Step 5: Build + verify** (L9 parity/CJK/en-ASCII/escaping/wall) + dark-mode check.

- [ ] **Step 6: Commit** (`feat: L9 build-graph worked-example trace`).

---

## Task 12: L24 LoRA low-rank trace (Style C)

`x -> W*x + scale*B*(A*x)`: the bypass path goes through the low-rank bottleneck (big dim -> small rank r -> big dim), added back to the frozen base output. SVG shows the big->small->big shape change.

**Files:**
- Modify: `src/part4.py` (LESSON_24, both langs - in the "LoRA 数学" section, after the existing math explanation)

- [ ] **Step 1: Design the example.** Dims: input d=4, rank r=1 (or 2), output d=4. Base: `W*x` (frozen, d->d). LoRA path: `A` (d->r, the "down"), then `B` (r->d, the "up"), times `scale`. Output = base + scaled bypass. Show shapes `[4] --A--> [1] --B--> [4]` with the bottleneck visually narrow.

- [ ] **Step 2: Author zh SVG** inside `.trace`. Top lane: `x [4] --W (冻结)--> W·x [4]`. Bottom lane (the LoRA bypass): `x [4] --A--> [r=1] (瓶颈) --B--> [4] -- ×scale -->`, then a `+` merging both into the final output. Make the `[r=1]` box visibly small. `.tcap`: `<b>追踪一次 LoRA 前向</b>：主干 W·x 不动，旁路把 x 压到低秩 r 再升回来、乘 scale 加回去（维度为示意）。` viewBox ~ `0 0 640 220`.

- [ ] **Step 3: Author en SVG** (parallel, ASCII): `.tcap` `<b>Tracing one LoRA forward</b>: the frozen W*x stays; the bypass squeezes x to low rank r, lifts it back, scales it, and adds it in (dims illustrative).` Use `x` for multiply where needed, `*` consistently.

- [ ] **Step 4: Bilingual lead-in `<p>`** both langs.

- [ ] **Step 5: Build + verify** (L24 parity/CJK/en-ASCII/escaping/wall) + dark-mode check.

- [ ] **Step 6: Commit** (`feat: L24 LoRA low-rank worked-example trace`).

---

## Task 13: Full-guide verification

- [ ] **Step 1: Rebuild + validators.** `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` -> 0 errors / 0 warnings / 96 links. Confirm `check_html.py` now counts `.trace` toward visual density (no lesson regressed below MIN_DIAGRAMS).

- [ ] **Step 2: Whole-guide invariants.** Run the standard scans across L1-L24: per-section `<p>` parity zh==en (0 mismatches), every lesson `cjk(zh)>=4000` and `cjk(en)==0`, no `&amp;lt;`/raw `<|` in any rendered lesson, max top-level `<p>` run <= 3 (wall-free preserved), and confirm exactly 12 lessons gained a `.trace` block (zh and en each).

- [ ] **Step 3: Count check.** Grep `class="trace"` across `lessons/*.html`: expect 24 occurrences (12 lessons x 2 languages).

---

## Task 14: Finish the branch

- [ ] **Step 1:** Confirm Task 13 all green.
- [ ] **Step 2:** Use the **finishing-a-development-branch** skill: present the 4 finish options; on the user's choice (historically "merge locally --no-ff + delete branch"), checkout master, `git merge --no-ff`, re-run validators on the merged result, delete the branch.

---

## Self-review notes

- **Spec coverage:** all 12 lessons in the spec's A/C table have a task (A: T1 L21, T2 L12, T3 L20, T4 L6, T5 L19, T6 L22; C: T7 L11, T8 L4, T9 L23, T10 L5, T11 L9, T12 L24). Component CSS (T1), check_html registration (T1), full verification (T13), finish (T14). No gaps.
- **Open execution decisions flagged in-task:** T4 may add a `.vc.scale` style (mirror existing `.cell.scale`); T7/T8/T10/T11 SVGs are given as skeletons to complete during execution (geometry finalized against the rendered result) - acceptable because the example values, layout, palette, captions, and verification are fully specified.
- **Type/name consistency:** component class is `.trace` everywhere; cell variants `.vc`/`.vc.hot`/`.vc.blue`/`.vc.dim` match the Task-1 CSS; `DIAGRAM_CLASSES` gets `"trace"`.
