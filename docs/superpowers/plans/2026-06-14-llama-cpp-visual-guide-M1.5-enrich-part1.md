# M1.5 · 第一部分加强（回炉重做 01-03）实施计划

> **配套 Spec：** `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`（§5/§6 已升级到新标准）
> **路线图：** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`
> **背景：** 用户反馈第一部分"没怎么画图、内容偏薄"（每课仅 1 图、~2000 字、0 折叠深挖）。
> 已确认按"大幅加强"标准回炉 01-03，并把标准写进 spec。
> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 逐 task 执行，每个 task
> 跑完整 spec+质量双审；内容课的质量审查必须**回查 `/home/verden/course/llama.cpp` 真实源码**核实准确性。

**Goal:** 把第一部分三课（01 是什么 / 02 项目地图 / 03 推理生命周期）从"1 图 + ~2000 字"提升到
**每课 3-5 张图 + 2-3 个折叠深挖 + ~4000-6000 中文字 + 概念示意图**，并新增可复用的 HTML+CSS 概念图组件。

**Architecture:** 在 `shell.py` 设计系统里新增两类**自包含、深色模式自适应**的概念图组件（`cells` 单元条 / `timeline`
时间线泳道），配合既有 `layers`/`flow`/`vflow`/`cols`/`table`；然后据详尽简报扩写 `part1.py` 的
`LESSON_01/02/03`（更多图、折叠深挖、伪代码、正文）。可选给 `check_html.py` 加"图密度"软检查防止再次变薄。

**Tech Stack:** Python 3 标准库 · 自包含双语 HTML/CSS/JS（无外部依赖、无外部图片/SVG 资源）。

---

## 内容作者方式（沿用 M1）

"**详尽简报 + 子代理执笔 + 审查回查源码核实**"。本 plan 为每课给出：要新增的**图（直接给 HTML/CSS 结构）**、
**折叠深挖**的标题与要点、**伪代码/源码片段**（经核实）、扩写方向与必讲论点、要核对的源码位置。
子代理据此用中/英扩写完整 HTML；审查回查真实源码把关。

**硬性标准（每课）**：3-5 张图（含至少 1 张概念示意图）· 2-3 个折叠深挖（示例→为什么→源码在哪→替代）·
2-3 段伪代码/源码片段 · 正文 ~4000-6000 中文字 · 中英信息等价 · `<pre>` 内转义、代码内 ASCII。

---

## 文件结构（本 milestone 修改）

- Modify: `src/shell.py` — 新增 `cells`/`cellgroup` 与 `timeline`/`lane`/`tslot` 的 CSS（概念图组件）
- Modify: `src/part1.py` — 扩写 `LESSON_01`、`LESSON_02`、`LESSON_03`
- Modify: `src/quizzes.py` — 按需为加强后的课补 1-2 题（保持每课 2-3 题）
- Modify: `src/check_html.py` —（可选）加"图密度"软检查（WARN）
- 产出：重建 `index.html` + `lessons/01..03`（CSS 变更会重建全部页面）

> 不改 `page()/index_page()/build.py/registry` 结构；课程数量与登记不变（仍 3 课）。

---

## Task 1: 新增概念图 CSS 组件（cells / timeline）

> 为概念示意图提供**自包含、深色模式自适应**的 HTML+CSS 组件（用设计系统 CSS 变量配色）。
> `cells`：一排小方块，用于 token 序列 / 量化分块 / KV cache 列。`timeline`：泳道，用于 prefill vs decode、逐步推进。
> 这两类组件 M2-M9 也会复用（张量形状、注意力等）。

**Files:**
- Modify: `src/shell.py`（在 `CSS` 字符串末尾、`.langtoggle:hover {...}` 之后、结尾 `"""` 之前追加）

- [ ] **Step 1: 追加 CSS 组件**

在 `src/shell.py` 的 `CSS = r"""..."""` 末尾（语言切换那段 `.langtoggle:hover {...}` 之后、闭合 `"""` 之前）插入：

```css

/* ---- schematic: cell strips (token sequences / quant blocks / KV columns) ---- */
.cellgroup { margin: 1.2rem 0; background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.cellgroup .cg-cap { font-size: .82rem; color: var(--muted); margin-bottom: .55rem; }
.cellgroup .cg-cap b { color: var(--ink); }
.cells { display: flex; flex-wrap: wrap; gap: .35rem; align-items: center; }
.cells + .cells { margin-top: .5rem; }
.cell { min-width: 2.1rem; padding: .38rem .5rem; text-align: center; border-radius: 8px;
  background: var(--panel-2); border: 1px solid var(--line); font-size: .78rem;
  font-family: ui-monospace, monospace; white-space: nowrap; }
.cell.scale { background: var(--amber-soft); border-color: var(--amber); color: var(--amber); font-weight: 700; }
.cell.hl    { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); font-weight: 700; }
.cell.q     { background: var(--blue-soft); border-color: var(--blue); color: var(--blue); }
.cell.dim   { opacity: .45; }
.cells .lab { font-size: .76rem; color: var(--faint); padding: 0 .35rem; }
.cells .sep { color: var(--faint); padding: 0 .1rem; }

/* ---- schematic: timeline lanes (prefill vs decode, step-by-step) ---- */
.timeline { margin: 1.2rem 0; display: flex; flex-direction: column; gap: .5rem;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1rem 1.1rem; box-shadow: var(--shadow); }
.timeline .lane { display: flex; align-items: center; gap: .5rem; flex-wrap: wrap; }
.timeline .lane-label { min-width: 6rem; font-size: .8rem; font-weight: 700; color: var(--muted); }
.timeline .tslot { padding: .4rem .6rem; border-radius: 8px; background: var(--panel-2);
  border: 1px solid var(--line); font-size: .78rem; text-align: center; font-family: ui-monospace, monospace; }
.timeline .tslot.span { flex: 1; min-width: 8rem; background: var(--blue-soft); border-color: var(--blue);
  color: var(--blue); font-weight: 700; }
.timeline .tslot.now { background: var(--accent-soft); border-color: var(--accent); color: var(--accent-ink); font-weight: 700; }
.timeline .tnote { font-size: .76rem; color: var(--muted); margin-top: .1rem; }
```

- [ ] **Step 2: 重建 + 校验（CSS 变更会重建全部页面）**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '\.cellgroup' lessons/01-what-is-llamacpp.html && echo "cells css present"
grep -q '\.timeline' lessons/01-what-is-llamacpp.html && echo "timeline css present"
git status --short
```
Expected：`structural check passed`（0 error）、`all 12 internal links resolve`、两个 `*css present`；
`git status` 显示 `src/shell.py` + 全部 4 个 HTML 产物被修改（CSS 内联在每页，故全部重建——预期）。

- [ ] **Step 3: Commit**

```bash
git add src/shell.py index.html lessons/01-what-is-llamacpp.html lessons/02-project-map.html lessons/03-inference-lifecycle.html
git commit -m "feat: add schematic diagram CSS components (cells, timeline)

Assisted-by: GitHub Copilot"
```

---

## Task 2: 加强课 01「llama.cpp 是什么」

> 目标：从 1 图/~2200 字 提升到 **3+ 图 + 1 表 + 2 片段 + 2-3 折叠深挖 + ~4500-5500 中文字**，中英等价。
> 保留现有的四层结构图、对比表、最小 C-API 片段；**新增**：训练vs推理对比图、量化概念示意图、CLI 片段、折叠深挖。

**Files:** Modify `src/part1.py`（`LESSON_01` zh+en）；按需 `src/quizzes.py`（01 可加到 3 题）。

- [ ] **Step 1: 重排并扩写 LESSON_01 结构（中/英）**

目标结构（按序；保留=沿用现有内容并适当扩写，新增=本次加）：

1. `lead`（扩写）：llama.cpp 是什么、为什么存在（一句更有力的点题）。
2. `card analogy`（保留：乐谱/播放器）。
3. `<h2>` 它解决什么问题（扩写）+ `card macro`（保留并扩写）。
4. `<h2>` 三大支柱（**新增小节**）：**GGUF 格式 · 量化 · ggml 引擎**，各一段；其中"量化"配下面的**量化概念示意图**。
5. `<h2>` 整体结构图：四层（保留 `layers` 图，扩写每层说明）。
6. `<h2>` 训练 vs 推理（**新增**）：用 `cols` 并排对比图（见下）+ 一段说明"为什么剥离推理"。
7. `<h2>` 和 PyTorch / transformers / vLLM 的区别（保留对比表，扩写）。
8. `<h2>` 怎么真正跑起来（**新增**）：先给**最小 C-API 片段**（保留），再给一条 **CLI 片段**（见下）。
9. **折叠深挖**（新增 2-3 个，见 Step 4）。
10. `card key`（扩写到 5-6 条）。
11. `card spark`（保留并扩写）。

- [ ] **Step 2: 新增「训练 vs 推理」对比图（cols）**

在第 6 小节插入（en 翻译文字、保持结构）：
```html
<div class="cols">
  <div class="col">
    <h4>🏋️ 训练（PyTorch 等）</h4>
    <ul>
      <li>前向 + <strong>反向传播</strong>、算梯度、更新权重</li>
      <li>要 <strong>优化器状态</strong>，显存吃紧（常需多卡）</li>
      <li>Python 生态，依赖重</li>
      <li>目标：把模型<strong>练出来</strong></li>
    </ul>
  </div>
  <div class="col">
    <h4>⚡ 推理（llama.cpp）</h4>
    <ul>
      <li><strong>只前向</strong>：权重固定，算一遍出 logits</li>
      <li>可<strong>量化</strong>压显存，CPU 也能跑</li>
      <li>纯 C/C++，几乎零依赖</li>
      <li>目标：把模型<strong>跑出字</strong></li>
    </ul>
  </div>
</div>
```

- [ ] **Step 3: 新增「量化」概念示意图（cells）**

在第 4 小节"量化"处插入（en 翻译标签、保持结构）。**先核实** `Q4_0` 块大小（`ggml/src/ggml-common.h` 的
`QK4_0`，应为 32）与每权重位宽（一块 = fp16 scale + 32×4bit ≈ 4.5 bit/权重），按真实值微调文字：
```html
<div class="cellgroup">
  <div class="cg-cap"><b>FP16 原始权重</b>：每个数 16 bit，精度高但占空间</div>
  <div class="cells">
    <span class="cell">0.12</span><span class="cell">-0.34</span><span class="cell">0.08</span><span class="cell">0.51</span><span class="cell dim">…</span>
    <span class="lab">一块 32 个 × 16 bit</span>
  </div>
  <div class="cg-cap" style="margin-top:.7rem"><b>Q4_0 量化后</b>：整块共享 1 个 scale，每个权重只存 4 bit 档位</div>
  <div class="cells">
    <span class="cell scale">scale</span><span class="sep">×</span>
    <span class="cell q">0110</span><span class="cell q">1001</span><span class="cell q">0011</span><span class="cell q">1100</span><span class="cell q dim">…</span>
    <span class="lab">≈ 4.5 bit/权重，约 1/4 大小</span>
  </div>
</div>
```
（注：`×` 在正文里可用，但若放进 `<pre>` 须改 ASCII；此图不在 `<pre>` 内，OK。）

- [ ] **Step 4: 新增 CLI 片段 + 折叠深挖**

CLI 片段（第 8 小节，`<pre class="code">`，ASCII、`<pre>` 内无裸 `<`/`&`）：
```
<span class="cm"># 最快跑起来：一个可执行文件 + 一个 .gguf</span>
llama-cli -m model.gguf -p <span class="st">"用一句话解释量化"</span>
```
（核实 `tools/cli` 的二进制名为 `llama-cli`、`-m`/`-p` 参数存在。）

折叠深挖（2-3 个，`<details class="accordion"><summary>…</summary><div class="acc-body">…</div></details>`，双语）：
- **量化为什么几乎不掉精度？** 示例（块内共享 scale）→ 为什么（按块缩放、低位宽够用；K-quant 用重要性区分）→
  源码在哪（`ggml/src/ggml-quants.c`、`tools/imatrix`）→ 替代（GPTQ / AWQ，思路类似但格式不同）。
- **ggml 到底是什么？** 一句话（张量 + 计算图 + 多后端的小引擎）→ 为什么自研（零依赖、可嵌入）→
  源码（`ggml/` 的 `ggml.c`/`ggml-backend`）→ 替代（直接用 cuBLAS/oneDNN，但失去可移植性）。
- **能跑多大的模型？**（估算）7B 模型 FP16 ≈ 14GB，Q4 ≈ 4GB 上下，普通笔记本可跑；给个"参数量 × 位宽 ÷ 8"的算法。

**必讲论点**：训练/推理的本质区别；量化=按块共享 scale 压位宽；三大支柱；CLI 一行即可跑。
**要核实**：`QK4_0`=32、`llama-cli` 名与 `-m/-p`、`ggml-quants.c`/`imatrix` 路径。引用用"文件+符号"。

- [ ] **Step 5: 重建 + 校验 + 富度自检**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "cols: "; grep -c 'class="cols"' lessons/01-what-is-llamacpp.html
echo -n "cellgroup: "; grep -c 'class="cellgroup"' lessons/01-what-is-llamacpp.html
echo -n "layers: "; grep -c 'class="layers"' lessons/01-what-is-llamacpp.html
echo -n "deep-dive details(含quiz): "; grep -c 'class="accordion"' lessons/01-what-is-llamacpp.html
cd src && python -c "import re,part1; t=re.sub(r'<[^>]+>','',part1.LESSON_01['zh']); print('zh字数',len(t.strip()))"
```
Expected：`structural check passed`（0 error）、`all 12 internal links resolve`；
`cols`>=2、`cellgroup`>=2、`layers`>=2（均含中英两份）；accordion 数明显增加（深挖+quiz）；zh 字数 > 3800。

- [ ] **Step 6: Commit**
```bash
git add src/part1.py src/quizzes.py lessons/01-what-is-llamacpp.html
git commit -m "content: enrich lesson 01 (training-vs-inference, quant diagram, CLI, deep-dives)

Assisted-by: GitHub Copilot"
```

---

## Task 3: 加强课 02「项目全景地图」

> 目标：从 1 图/~2100 字 提升到 **3+ 图 + 1 表 + 2-3 折叠深挖 + ~4500-5500 中文字**，中英等价。
> 保留四层结构图 + repo-map 表；**新增**：模型数据流图（标 Python/C++ 边界）、读源码路线图、折叠深挖。

**Files:** Modify `src/part1.py`（`LESSON_02` zh+en）；按需 `src/quizzes.py`。

- [ ] **Step 1: 重排并扩写 LESSON_02 结构（中/英）**

1. `lead`（扩写）。
2. `card analogy`（保留：工厂园区/导览图）。
3. `<h2>` 顶层目录速览 + `table`（保留 repo-map，扩写角色说明）。
4. `<h2>` 四层映射（保留 `layers`，扩写）+ 两条旁路段落（保留）。
5. `<h2>` 一个模型怎么从训练到运行（**新增**）：模型数据流图（见 Step 2）+ 一段说明"Python 准备 / C++ 运行的边界就是 `.gguf`"。
6. `<h2>` 想读源码，从哪进（**新增**）：读源码路线图（见 Step 3）。
7. **折叠深挖**（新增 2-3 个，见 Step 4）。
8. `card key`（扩写 5-6 条）+ `card spark`（保留并扩写）。

- [ ] **Step 2: 新增「模型数据流」图（flow，标 Python/C++ 边界）**

插入第 5 小节（en 翻译文字、保持结构）：
```html
<div class="flow">
  <div class="node"><div class="nt">HF / PyTorch 模型</div><div class="nd">safetensors 权重</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">convert_hf_to_gguf.py</div><div class="nd">Python · gguf-py</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">model.gguf</div><div class="nd">单文件 · 权重 + 元数据</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_model_load_from_file</div><div class="nd">C++ 运行时</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">跑出字</div><div class="nd">llama-cli / server</div></div>
</div>
<p class="acc-intro">左半截是 <strong>Python（准备）</strong>，右半截是 <strong>C++（运行）</strong>，两者的边界就是中间那个 <span class="inline">.gguf</span> 文件。</p>
```
（`->` 在 HTML 文本里写 `-&gt;` 以防被当标签；`.flow`/`.node`/`.arrow`/`.nt`/`.nd`/`.hl` 已在设计系统中。）

- [ ] **Step 3: 新增「读源码路线」图（layers，3 个目标）**

插入第 6 小节（en 翻译、保持结构；用 `layers` 组件，badge=目标、name=目录、desc=怎么读）：
```html
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">想会用</span><span class="name">tools/ · examples/simple</span></div>
    <div class="ld">先把 <code>llama-cli</code>/<code>llama-server</code> 跑起来，再读 <code>examples/simple</code> 的最小调用</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">懂推理</span><span class="name">src/llama-*</span></div>
    <div class="ld">按主线读：<code>llama-model-loader</code> -&gt; <code>llama-graph</code> -&gt; <code>llama-kv-cache</code> -&gt; <code>llama-sampler</code></div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">懂算子</span><span class="name">ggml/</span></div>
    <div class="ld">进引擎：<code>ggml.c</code> 与各 <code>ggml-*</code> 后端，看张量/算子/调度怎么实现</div></div>
</div>
```

- [ ] **Step 4: 折叠深挖（2-3 个，双语）**

- **GGUF 文件里到底装了什么？** 示例（header + 元数据 KV + tensor 信息 + 权重块）→ 为什么（单文件、可 mmap、自带超参/词表）→
  源码（`ggml/src/gguf.cpp`、`src/llama-model-loader.cpp`）→ 替代（旧的 GGML/GGJT 格式，已被 GGUF 取代）。
- **为什么 ggml 是独立子项目？** 一句（同一个引擎被多个项目复用）→ 例子（`whisper.cpp` 等也用 ggml）→
  源码（`ggml/` 自带 `include/`+`src/`）→ 好处（引擎独立演进、便于嵌入）。
- **common/ 和 src/ 有啥区别？** common = 各可执行程序**共用的胶水**（参数、采样封装、日志…），不是推理库本体；
  推理本体在 `src/llama-*`。源码（`common/common.cpp`、`common/arg.cpp`）。

**必讲论点**：`.gguf` 是 Python/C++ 边界；按目标选读源码入口；GGUF 单文件自带超参/词表；ggml 可复用。
**要核实**：`gguf.cpp` 在 `ggml/src/`；`src/llama-model-loader` 存在；`examples/simple` 存在；whisper.cpp 用 ggml（众所周知，可不强求仓库内证据）。

- [ ] **Step 5: 重建 + 校验 + 富度自检**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "flow: "; grep -c 'class="flow"' lessons/02-project-map.html
echo -n "layers: "; grep -c 'class="layers"' lessons/02-project-map.html
echo -n "table: "; grep -c '<table class="t"' lessons/02-project-map.html
echo -n "accordion: "; grep -c 'class="accordion"' lessons/02-project-map.html
cd src && python -c "import re,part1; t=re.sub(r'<[^>]+>','',part1.LESSON_02['zh']); print('zh字数',len(t.strip()))"
```
Expected：0 error、链接全通；`flow`>=2、`layers`>=4（4 层图 + 读源码路线图，各中英两份）、`table`>=2、accordion 增加；zh 字数 > 3800。

- [ ] **Step 6: Commit**
```bash
git add src/part1.py src/quizzes.py lessons/02-project-map.html
git commit -m "content: enrich lesson 02 (model dataflow with Python/C++ boundary, read paths, deep-dives)

Assisted-by: GitHub Copilot"
```

---

## Task 4: 加强课 03「一次推理的生命周期」

> 目标：从 1 图/~2500 字 提升到 **4+ 图 + 1 片段 + 2-3 折叠深挖 + ~4500-5500 中文字**，中英等价。
> 保留 7 步 vflow + 伪代码；**新增**：prefill/decode 时间线、KV cache 增长示意、"一次 decode 内部"放大图、折叠深挖。

**Files:** Modify `src/part1.py`（`LESSON_03` zh+en）；按需 `src/quizzes.py`。

- [ ] **Step 1: 重排并扩写 LESSON_03 结构（中/英）**

1. `lead`（扩写）+ `card analogy`（保留：流水线接力）。
2. `<h2>` 七步数据流（保留 `vflow`，每步说明可再丰富一句）。
3. `<h2>` 放大第 3 步：一次 decode 内部（**新增**）：inside-decode `flow` 图（见 Step 2）+ 一段说明"decode = 建图 + 后端执行 -> logits"。
4. `<h2>` prefill vs decode（保留 `card macro` 并扩写）+ **新增** prefill/decode 时间线图（见 Step 3）。
5. `<h2>` KV cache 为什么让循环不贵（**新增**）：KV 增长示意图（见 Step 4）+ 一段说明。
6. `<h2>` 对回最小主线（保留伪代码 `<pre>` + 那句"采样器自取 logits"的说明）。
7. **折叠深挖**（新增 2-3 个，见 Step 5）。
8. `card key`（扩写）+ `card spark`（保留并扩写）。

- [ ] **Step 2: 新增「一次 decode 内部」放大图（flow）**

```html
<div class="flow">
  <div class="node"><div class="nt">llama_decode</div><div class="nd">一次前向</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">建计算图</div><div class="nd">llama-graph.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">后端执行</div><div class="nd">ggml-backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits</div><div class="nd">llama_get_logits_ith</div></div>
</div>
```

- [ ] **Step 3: 新增「prefill vs decode」时间线（timeline）**

```html
<div class="timeline">
  <div class="lane"><span class="lane-label">Prefill</span><span class="tslot span">整段 prompt（t1…t5）一次并行算，填满 KV cache</span></div>
  <div class="lane"><span class="lane-label">Decode</span><span class="tslot">t6</span><span class="tslot">t7</span><span class="tslot now">t8…</span></div>
</div>
<p class="acc-intro">Prefill 把整段提示词<strong>一次并行</strong>算完；之后每步 decode <strong>只算 1 个新 token</strong>，所以"接着往下写"很便宜。</p>
```

- [ ] **Step 4: 新增「KV cache 增长」示意（cells）**

```html
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache</b>：每生成一个 token 就把它的 K/V 追加进缓存，下一步直接复用、不重算历史</div>
  <div class="cells"><span class="lab">prefill 后</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span><span class="cell">K4</span><span class="cell">K5</span></div>
  <div class="cells"><span class="lab">decode t6</span><span class="cell dim">K1…K5（复用）</span><span class="cell hl">+K6</span></div>
  <div class="cells"><span class="lab">decode t7</span><span class="cell dim">K1…K6（复用）</span><span class="cell hl">+K7</span></div>
</div>
```

- [ ] **Step 5: 折叠深挖（2-3 个，双语）**

- **为什么 decode 输出的是 logits、不是文字？** logits = 词表上每个 token 的"分数向量"；选哪个是**采样器**的事
  （`src/llama-sampler.cpp` `llama_sampler_sample`），还原文字是 `llama_token_to_piece`（`src/llama-vocab.cpp`）的事。
- **KV cache 到底省了什么？** 没有它，生成第 n 个 token 要把前 n-1 个重算一遍 -> 总成本约 O(n^2)；有了它每步只算新 token -> 约 O(n)。
  注意：注意力对历史的"扫描"仍是每步 O(n)，省掉的是**重复计算过去 token 的 K/V**。源码 `src/llama-kv-cache.cpp`。
- **"计算图"是什么？为什么先建图再算？** ggml 先把这一步的运算<strong>描述成一张图</strong>（`src/llama-graph.cpp` 的 `llm_graph_*`，
  由 `src/llama-model.cpp` 的 `build_graph` 拼出），再交给后端调度执行（`ggml-backend`）——分离"描述"与"执行"，便于多后端与优化。

**必讲论点**：decode 产 logits（采样才得 token）；prefill 并行 / decode 逐步；KV cache 把 O(n^2) 摊成 O(n)（但注意力扫描仍 O(n)/步）；先建图后执行。
**要核实**：`llama-graph.cpp`/`llm_graph_*`、`llama-kv-cache.cpp`、`llama_get_logits_ith`/`llama_token_to_piece` 等（M1 已核实，引用用"文件+符号"）。

- [ ] **Step 6: 重建 + 校验 + 富度自检**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "vflow: "; grep -c 'class="vflow"' lessons/03-inference-lifecycle.html
echo -n "flow: "; grep -c 'class="flow"' lessons/03-inference-lifecycle.html
echo -n "timeline: "; grep -c 'class="timeline"' lessons/03-inference-lifecycle.html
echo -n "cellgroup: "; grep -c 'class="cellgroup"' lessons/03-inference-lifecycle.html
echo -n "accordion: "; grep -c 'class="accordion"' lessons/03-inference-lifecycle.html
cd src && python -c "import re,part1; t=re.sub(r'<[^>]+>','',part1.LESSON_03['zh']); print('zh字数',len(t.strip()))"
```
Expected：0 error、链接全通；`vflow`>=2、`flow`>=2、`timeline`>=2、`cellgroup`>=2（均含中英两份）、accordion 增加；zh 字数 > 3800。

- [ ] **Step 7: Commit**
```bash
git add src/part1.py src/quizzes.py lessons/03-inference-lifecycle.html
git commit -m "content: enrich lesson 03 (prefill/decode timeline, KV growth, inside-decode, deep-dives)

Assisted-by: GitHub Copilot"
```

---

## Task 5: 图密度软检查 + M1.5 验收

> 给 `check_html.py` 加一个"视觉块密度"软检查（WARN，不是 ERR），让以后再出现"内容偏薄、几乎没图"的课能被
> 校验**当场报出来**（这次就是因为薄也能 0 error 才漏过）。然后做 M1.5 整体验收。

**Files:** Modify `src/check_html.py`（加 WARN 软检查）。

- [ ] **Step 1: 给 check_html.py 加图密度软检查**

在常量区（`SOFT_EXEMPT = {...}` 附近）加：
```python
# Visual-block density (soft): containers that count as a "diagram/table".
DIAGRAM_CLASSES = ("layers", "vflow", "flow", "cols", "cellgroup", "timeline")
MIN_DIAGRAMS = 6  # per lesson, counting BOTH languages (>= 3 per language)
```
在 `check_lesson()` 的软检查处（`if fname not in SOFT_EXEMPT:` 块内，紧挨 analogy/key 检查之后）加：
```python
        nvis = sum(html.count(f'class="{c}"') for c in DIAGRAM_CLASSES)
        nvis += html.count('<table class="t"')
        if nvis < MIN_DIAGRAMS:
            add("WARN", fname, f"only {nvis} visual blocks (want >= {MIN_DIAGRAMS}; add diagrams)")
```
（说明：每个图/表容器在中英两份里各出现一次，故阈值按"双语合计"计；`class="flow"` 不会误配 `class="vflow"`。）

- [ ] **Step 2: M1.5 验收（清重建 + 校验 + 三课富度小结）**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
for f in 01-what-is-llamacpp 02-project-map 03-inference-lifecycle; do
  echo "== $f =="
  v=0
  for c in layers vflow flow cols cellgroup timeline; do v=$((v+$(grep -c "class=\"$c\"" lessons/$f.html))); done
  v=$((v+$(grep -c '<table class="t"' lessons/$f.html)))
  echo "  视觉块(双语合计): $v"
  echo "  折叠(深挖+quiz): $(grep -c 'class="accordion"' lessons/$f.html)"
  python3 -c "import re,sys;sys.path.insert(0,'src');import part1;k={'01-what-is-llamacpp':'LESSON_01','02-project-map':'LESSON_02','03-inference-lifecycle':'LESSON_03'}['$f'];print('  zh字数',len(re.sub(r'<[^>]+>','',getattr(part1,k)['zh']).strip()))"
done
git status --short && echo "(clean if empty)"
```
Expected：`structural check passed`（**0 error**；且这 3 课**无图密度 WARN**——每课视觉块 >= 6）、链接全通；
每课视觉块 >= 6、折叠数明显多于改前、zh 字数 > 3800；`git status` 干净。

- [ ] **Step 3: Commit + 路线图小注**

在 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 的 M1 行后或状态区加一行小注：
`> M1.5：第一部分 3 课已按加强标准回炉（每课 3-5 图 + 折叠深挖 + 概念示意图）。`
```bash
git add src/check_html.py docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md
git commit -m "test: add visual-density soft check; note Part 1 enrichment

Assisted-by: GitHub Copilot"
```

---

## 验收标准（Definition of Done · M1.5）

- 三课每课 **>= 3 张图**（含概念示意图）+ **>= 2 个折叠深挖** + 正文 **~4000-6000 中文字**，中英等价。
- 新增 `cells`/`timeline` CSS 组件，深色模式自适应、自包含。
- `check_html.py` 0 error；新增图密度软检查，三课**无该 WARN**；`check_links.py` 0 死链；构建零漂移。

---

## Self-Review（plan 作者自审）

**1. Spec 覆盖**：本 M1.5 落实 Spec §5/§6 升级后的硬性标准（3-5 图、2-3 折叠深挖、2-3x 内容、概念示意图）。
**2. 占位符**：每课给出新增图的**完整 HTML**、折叠深挖的标题与要点、片段与必讲论点、待核实清单——非空泛。
**3. 一致性**：新增 CSS 类（`cellgroup/cells/cell/timeline/lane/tslot` 等）在 Task 1 定义、Tasks 2-4 使用、Task 5 软检查统计，三处名称一致；沿用既有 `layers/flow/vflow/cols/table` 组件。
**4. 歧义**：图/表在双语各一份，故密度阈值按双语合计（>=6 ≈ 每语 3）；概念图用 HTML+CSS（非 SVG）以适配深色模式；
`<pre>` 内 ASCII/转义不变；`->` 在 HTML 文本里写 `-&gt;`。

---

## 执行交接

计划完成，保存于 `docs/superpowers/plans/2026-06-14-llama-cpp-visual-guide-M1.5-enrich-part1.md`。
建议沿用 **subagent-driven-development**：开分支 `build/m1.5-enrich`，逐 task 派发实现子代理 + spec/质量双审，
内容课的质量审查**回查真实源码**核实（量化块大小、CLI 参数、符号位置等）；全部完成后整体审查并合并 master。
