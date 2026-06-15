# M1.6 · 第一部分「中文深挖」实施计划

> **配套 Spec：** `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`（§5/§6）
> **前置：** M1.5 已合并（每课 4 视觉块 + 3 折叠深挖），但**纯中文字仅 ~1900-2280/课**，约为
> spec 字面目标「~4000-6000 中文字」的一半（之前字数口径误把英文代码/路径也算进去）。
> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development；每 task 跑完整 spec+质量双审，
> 内容课的质量审查**回查 `/home/verden/course/llama.cpp` 真实源码**。

**Goal:** 把第一部分三课的**纯中文正文加到 ~4000+ 汉字/课**（约翻倍，是真深度不是灌水）；给**课 02 补 1 段代码片段**、
给**课 01/02 各补 1 张真图**（达到 4 张真图）；并把字数口径改成**按 CJK 汉字计**（spec 措辞 + check_html 软检查）。

**Architecture:** 不动生成器结构。`check_html.py` 加"中文字密度"软检查（按 CJK 计，WARN）；`part1.py` 扩写
`LESSON_01/02/03` 的中文正文与深挖、各加 1 图、课 02 加 1 段 `<pre>`；中英保持等价（英文同步加深）。

**Tech Stack:** Python 3 标准库 · 自包含双语 HTML/CSS/JS。

---

## 字数口径（本 milestone 起统一）

- **"中文字" = CJK 汉字**（Unicode `\u4e00-\u9fff`），用 `len(re.findall(r'[\u4e00-\u9fff]', zh))` 计。
- 每课**目标 ~4000+ CJK**（深挖折叠里的中文也计入）。check_html 加软 WARN：CJK < 阈值则提示（不是 ERR）。
- 英文不按字数卡（英文自然约 2x 字符），只要求**信息与中文等价**。

## 深挖原则（不是灌水）

扩写要加**真信息**：每节多讲"**为什么这么设计 / 它在解决什么 / 换种做法会怎样 / 常见误区**"；
深挖折叠从 3-4 句扩到一小段；多举**具体例子与数字**；保持准确（对照源码）。避免重复已有句子、避免空话。

---

## 文件结构（本 milestone 修改）

- Modify: `src/check_html.py` — 加"CJK 中文字密度"软检查（WARN）
- Modify: `src/part1.py` — 扩写 `LESSON_01/02/03` 中文正文/深挖；课 01/02 各加 1 图；课 02 加 1 段 `<pre>`
- Modify: `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md` — §6 注明"中文字 = CJK 汉字"
- 产出：重建 `lessons/01..03`（及 index 若计数变化）

> 不改 `shell.py` 结构（如需新图样式才动 CSS）、`build.py`、`registry`、课程数量与登记。

---

## Task 1: CJK 字数口径（check_html 软检查 + spec 措辞）

> 把"中文字"统一为 **CJK 汉字**计数，并给 check_html 加一个"中文密度"软检查（WARN），防止"看着长、其实中文很少"。

**Files:** Modify `src/check_html.py`、`docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`。

- [ ] **Step 1: check_html.py 加 CJK 软检查**

在常量区（`MIN_DIAGRAMS = 6` 附近）加：
```python
MIN_CJK = 3000  # per-lesson zh CJK chars (soft floor; authoring target ~4000+)
```
在 `main()` 的 registry 内容循环里（已有 `MIN_CONTENT` 那段，按 `fname`/`c` 遍历 CONTENT 处），在
非空检查之后、`SOFT_EXEMPT` 允许的前提下追加：
```python
        if fname not in SOFT_EXEMPT:
            cjk = len(re.findall(r"[\u4e00-\u9fff]", c.get("zh", "")))
            if cjk < MIN_CJK:
                add("WARN", fname, f"only {cjk} CJK chars in zh (want >= {MIN_CJK})")
```
（`re` 已 import；`c` 是该课的 `{"zh","en"}`。该检查只看中文侧，英文不卡字数。）

- [ ] **Step 2: spec §6 注明口径**

在 spec §6 规则 3「尽量详细」里，把"~4000-6000 中文字"一句改为明确口径：
把
```
3. **尽量详细（硬性）**：每课正文目标 **~4000-6000 中文字**（约为"薄基线"的 2-3 倍），并配 **2-3 个折叠深挖**，
```
改为
```
3. **尽量详细（硬性）**：每课**纯中文正文目标 ~4000+ 汉字（CJK，按 `\u4e00-\u9fff` 计；不含英文/代码/路径）**，
   并配 **2-3 个折叠深挖**，
```

- [ ] **Step 3: 验证软检查有牙齿（内存内，不写盘）**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py | tail -3 && python check_links.py
python -c "
import check_html as C, re
# 现有三课目前 ~1900-2280 CJK，应当各触发 1 条 CJK WARN（直到 M1.6 后续 task 深挖到 >=3000）
import registry
for fn in ('01-what-is-llamacpp.html','02-project-map.html','03-inference-lifecycle.html'):
    cjk=len(re.findall(r'[\u4e00-\u9fff]', registry.CONTENT[fn]['zh']))
    print(fn, 'CJK', cjk, 'WARN?' , cjk < C.MIN_CJK)
"
```
Expected：`check_html` 此刻会对三课各报 1 条 `CJK chars in zh` 的 **WARN**（因为还没深挖），但**仍 0 error、structural check passed**；
脚本打印三课当前 CJK 数（~1900-2280）且 `WARN? True`。这证明检查生效；后续 task 把中文加到 >=3000 后 WARN 消失。

- [ ] **Step 4: Commit**
```bash
cd /home/verden/course/llama-cpp-visual-guide
git add src/check_html.py docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md
git commit -m "test: add CJK prose-density soft check; clarify char metric in spec

Assisted-by: GitHub Copilot"
```

---

## Task 2: 课 01 中文深挖（~4000+ CJK）+ 第 4 张图

> 现状 ~2280 CJK、3 真图。目标 **~4000+ CJK、4 真图**。扩写是加真信息（见子话题清单），不是灌水。

**Files:** Modify `src/part1.py`（`LESSON_01` zh+en，中英同步加深）。

- [ ] **Step 1: 按子话题给各节加深（中/英）**

在保留现有结构的前提下，给下列地方补**实质内容**（每条 1-3 句，融进对应小节或其折叠深挖）：
- **为什么是 C/C++（不是 Python/Rust/Go）**：零运行时、可静态编译成单个可执行文件、能嵌进任何 App、跨平台。
- **GGUF 为什么"单文件"好**：免配置、可 `mmap` 按需加载、自带超参 / 词表 / chat 模板，换模型只换一个文件。
- **量化为什么能省又不太掉质量**：权重数量远大于激活；低位宽对"生成下一个词"的影响小；按块共享 scale 保住动态范围；代价是轻微质量损失（用重要性矩阵可缓解）。
- **ggml 为什么自研**：现成训练框架太重、依赖多；要可移植到 CPU 和各家 GPU、要零依赖、要能嵌入。
- **"只前向"具体意味着什么**：没有反向/梯度/优化器状态；显存只需放权重 + KV cache；因此可量化、可在 CPU 跑。
- **实际能跑多大**：7B Q4 权重 ~4 GB，再加随上下文增长的 KV cache，普通笔记本可跑；70B 需更多内存或多机/offload。
- **生态定位**：Ollama / LM Studio 等桌面工具底层很多就是用 llama.cpp（一句话点出它的地基地位）。

同时把 **3 个折叠深挖**各自从几句扩到一小段（多给例子和数字）。中文目标 ~4000+ CJK；英文同步加深、信息等价。

- [ ] **Step 2: 加第 4 张真图（量化 -> 便携 flow）**

在"三大支柱/量化"或"怎么跑起来"附近插入（en 翻译文字、保持结构）：
```html
<div class="flow">
  <div class="node"><div class="nt">FP16 模型</div><div class="nd">7B ≈ 14 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">量化 Q4</div><div class="nd">≈ 4 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">.gguf 单文件</div><div class="nd">+ 一个可执行文件</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">本地跑</div><div class="nd">笔记本 / 手机 / 服务器</div></div>
</div>
```

- [ ] **Step 3: 重建 + 校验 + CJK 自检**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "真图(flow+cols+cellgroup+layers): "
v=0; for c in flow cols cellgroup layers; do v=$((v+$(grep -c "class=\"$c\"" lessons/01-what-is-llamacpp.html))); done; echo $((v/2))" /单语"
cd src && python -c "import re,part1; print('CJK', len(re.findall(r'[\u4e00-\u9fff]', part1.LESSON_01['zh'])))"
```
Expected：`structural check passed`（0 error；**课 01 不再有 CJK WARN**）、链接全通；单语真图 >= 4（flow+cols+cellgroup+layers）；CJK >= 3800（目标 ~4000+）。`<pre>` 仍安全。

- [ ] **Step 4: Commit**
```bash
git add src/part1.py lessons/01-what-is-llamacpp.html
git commit -m "content: deepen lesson 01 Chinese prose (~4000+ CJK) and add quant-to-portability diagram

Assisted-by: GitHub Copilot"
```

---

## Task 3: 课 02 中文深挖（~4000+ CJK）+ 代码片段 + 第 4 张图（GGUF 概念图）

> 现状 ~1960 CJK、3 真图、**0 代码片段**。目标 **~4000+ CJK、4 真图（含 1 张概念图）、1 段 `<pre>`**。

**Files:** Modify `src/part1.py`（`LESSON_02` zh+en）。

- [ ] **Step 1: 按子话题给各节加深（中/英）**
- **四块的职责边界与"为什么分开"**：引擎(ggml) / 推理(src/llama-*) / 工具(tools) / 准备(gguf-py·convert) 各管一件事的好处。
- **公共 API 收在单头文件的好处**：对外面只暴露 `llama.h`，内部随便改；稳定的接口面好被别的语言绑定（Python/Go/Rust binding）。
- **ggml 为什么能独立复用**：`scripts/sync-ggml.sh` 从上游 `ggml-org/ggml` 同步；`whisper.cpp` 等也用它；依赖**单向**（`src/llama-*` 包含 `ggml.h`，反过来没有）。
- **GGUF 里到底装了什么**：magic+版本、元数据 KV（举例 `block_count`/`embedding_length`/`tokenizer.ggml.*`/`tokenizer.chat_template`）、张量信息（名/形状/类型/偏移）、张量数据；加载靠 `mmap` + 偏移。
- **common vs src 的区别**：common 是各程序**共用胶水**（`arg`/`sampling`/`chat`/`log`/`download`）；删掉 common，`src/llama-*` 仍能编（依赖方向 tools -> common -> llama.h -> src/llama-* -> ggml，单向）。
- **新增一个模型支持大概要动哪里**：`llama-arch` 注册结构、`llama-graph` 搭前向、`convert_*.py` 写转换（预告 part4/part8）。
- 把 **3 个折叠深挖**与"读源码路线"各条都再展开一句。中文目标 ~4000+ CJK；英文同步、等价。

- [ ] **Step 2: 加代码片段（准备 -> 运行 管道，`<pre class="code">`）**

插入"模型怎么从训练到运行"一节（ASCII；`<pre>` 内不得有裸 `<`/`&`；`->` 直接写即可，`>` 无需转义）：
```
<span class="cm"># 模型从准备到运行的完整管道</span>
<span class="cm"># 1) 转换：HuggingFace 模型 -> GGUF（Python 侧）</span>
python convert_hf_to_gguf.py ./my-model            <span class="cm"># 产出 my-model.gguf（FP16）</span>
<span class="cm"># 2) 量化（可选，压小）</span>
llama-quantize my-model.gguf my-model-Q4.gguf Q4_0
<span class="cm"># 3) 运行（C++ 侧）</span>
llama-cli -m my-model-Q4.gguf -p <span class="st">"你好"</span>
```
（核实 `convert_hf_to_gguf.py`、`llama-quantize`、`llama-cli` 调用形式；量化档位名如 `Q4_0` 对照 `llama-quantize` 用法。）

- [ ] **Step 3: 加第 4 张真图（GGUF 文件结构概念图，cells）**

在"GGUF 里装了什么"附近插入（en 翻译标签）：
```html
<div class="cellgroup">
  <div class="cg-cap"><b>GGUF 文件结构</b>（单文件，按顺序排布）</div>
  <div class="cells">
    <span class="cell hl">magic + 版本</span><span class="cell">元数据 KV（超参 / 词表 / chat 模板）</span><span class="cell">张量信息（名 / 形状 / 类型 / 偏移）</span><span class="cell q">张量数据（权重块）</span>
  </div>
  <div class="cg-cap" style="margin-top:.5rem">加载时按"张量信息"里的偏移，用 <span class="mono">mmap</span> 映射到对应"张量数据"块，<strong>按需取用、不全量拷贝</strong>。</div>
</div>
```

- [ ] **Step 4: 重建 + 校验 + CJK 自检**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "真图(flow+layers+cellgroup): "
v=0; for c in flow layers cellgroup; do v=$((v+$(grep -c "class=\"$c\"" lessons/02-project-map.html))); done; echo $((v/2))" /单语"
echo -n "pre: "; grep -c '<pre class="code"' lessons/02-project-map.html
cd src && python -c "import re,part1; print('CJK', len(re.findall(r'[\u4e00-\u9fff]', part1.LESSON_02['zh'])))
for pre in re.findall(r'<pre[^>]*>(.*?)</pre>', part1.LESSON_02['zh'], re.S):
    c=re.sub(r'</?(?:span|strong|b|em|u|a)\b[^>]*>','',pre); assert not re.search(r'<(?!/)',c),'raw <'
print('pre safe')"
```
Expected：0 error（**课 02 不再有 CJK WARN**）、链接全通；单语真图 >= 4（layers×2 + flow + cellgroup）；`pre` >= 2（中英各 1）；CJK >= 3800；`pre safe`。

- [ ] **Step 5: Commit**
```bash
git add src/part1.py lessons/02-project-map.html
git commit -m "content: deepen lesson 02 Chinese prose (~4000+ CJK), add pipeline snippet and GGUF-layout diagram

Assisted-by: GitHub Copilot"
```

---

## Task 4: 课 03 中文深挖（~4000+ CJK）

> 现状 ~1890 CJK、已有 4 真图（vflow + flow + timeline + cellgroup）。本 task **只加深中文正文**，不加新图。

**Files:** Modify `src/part1.py`（`LESSON_03` zh+en）。

- [ ] **Step 1: 按子话题给各节加深（中/英）**
- **七步各自再讲透一点**：分词为什么不是按字/词而是 subword；组批里 `pos`/`seq_id` 是什么、为什么需要；
  "前向"在算什么（embedding -> 多层 transformer -> 输出层）；取 logits 是取最后一个位置的那一行。
- **prefill 为什么能并行 / decode 为什么必须串行**：prompt 里每个 token 的 K/V 互不依赖，可一次并行算；
  但"下一个词"依赖"已生成的词"，所以 decode 只能一步一个、串行。
- **K/V 到底是什么、为什么能缓存**：注意力里每个历史 token 会算出一对 Key/Value；它们一旦算出就不再变，
  所以缓存起来下次直接复用；代价是 KV cache 占的显存随上下文长度线性增长（长上下文很吃显存）。
- **为什么"先建图后执行"**：ggml 先把这一步的算子拼成一张图（只描述、不计算），再交后端调度执行；
  好处是同一张图能换不同后端跑、能做内存规划与算子融合等优化。
- **logits 到底是什么、采样怎么选**：logits 是词表上每个 token 的"打分"；采样器按 temperature / top-k / top-p
  等策略从中挑一个（预告后面专门讲采样的课）。
- **token != 词**：`llama_token_to_piece` 还原出来的是"词片"，多个片才拼成一个完整的词/汉字。
- 把 **3 个折叠深挖**各自再展开一小段（多给"为什么 / 例子 / 数字"）。中文目标 ~4000+ CJK；英文同步、等价。

- [ ] **Step 2: 重建 + 校验 + CJK 自检**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
echo -n "真图(vflow+flow+timeline+cellgroup): "
v=0; for c in vflow flow timeline cellgroup; do v=$((v+$(grep -c "class=\"$c\"" lessons/03-inference-lifecycle.html))); done; echo $((v/2))" /单语"
cd src && python -c "import re,part1; print('CJK', len(re.findall(r'[\u4e00-\u9fff]', part1.LESSON_03['zh'])))
for pre in re.findall(r'<pre[^>]*>(.*?)</pre>', part1.LESSON_03['zh'], re.S):
    c=re.sub(r'</?(?:span|strong|b|em|u|a)\b[^>]*>','',pre); assert not re.search(r'<(?!/)',c),'raw <'
print('pre safe')"
```
Expected：0 error（**课 03 不再有 CJK WARN**）、链接全通；单语真图 >= 4；CJK >= 3800；`pre safe`。

- [ ] **Step 3: Commit**
```bash
git add src/part1.py lessons/03-inference-lifecycle.html
git commit -m "content: deepen lesson 03 Chinese prose (~4000+ CJK)

Assisted-by: GitHub Copilot"
```

---

## Task 5: M1.6 验收

> 端到端确认三课中文都到位、图与片段齐全、CJK 软检查全过。

**Files:** 无新增。

- [ ] **Step 1: 清重建 + 校验 + 三课 CJK/图/片段小结**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
for f in 01-what-is-llamacpp 02-project-map 03-inference-lifecycle; do
  echo "== $f =="
  # 真图（排除 table）
  v=0; for c in layers vflow flow cols cellgroup timeline; do v=$((v+$(grep -c "class=\"$c\"" lessons/$f.html))); done
  echo "  真图/单语: $((v/2))   代码片段/单语: $(( $(grep -c '<pre class="code"' lessons/$f.html) /2 ))"
  python3 -c "import re,sys;sys.path.insert(0,'src');import part1;k={'01-what-is-llamacpp':'LESSON_01','02-project-map':'LESSON_02','03-inference-lifecycle':'LESSON_03'}['$f'];print('  CJK 汉字:', len(re.findall(r'[\u4e00-\u9fff]', getattr(part1,k)['zh'])))"
done
git status --short && echo "(clean if empty)"
```
Expected：`structural check passed`（**0 error 且三课均无 CJK WARN、无图密度 WARN**）、链接全通；
每课**真图 >= 4**、CJK **>= 3800（目标 ~4000+）**；课 02 代码片段 >= 1（单语）；`git status` 干净。

- [ ] **Step 2: 路线图小注（可选）**
在 roadmap 把 M1.5 那行小注补一句：`并经 M1.6 把每课纯中文加深到 ~4000+ 汉字、课02补代码片段与 GGUF 概念图。`
```bash
cd /home/verden/course/llama-cpp-visual-guide
git add docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md
git commit -m "docs: note M1.6 prose deepening in roadmap

Assisted-by: GitHub Copilot"
```

---

## 验收标准（Definition of Done · M1.6）

- 三课**纯中文 >= 3800 CJK（目标 ~4000+）**，无 CJK 软 WARN；中英信息等价。
- 课 01/02/03 各 **>= 4 张真图**；课 02 有 **1 段代码片段**与 **1 张概念图（GGUF 结构）**。
- `check_html.py` 0 error、新增 CJK 软检查生效；`check_links.py` 0 死链；构建零漂移。
- 字数口径已按 CJK 汉字（spec §6 已注明）。

---

## Self-Review（plan 作者自审）

**1. Spec 覆盖**：落实 §6 升级后的"纯中文 ~4000+ 汉字"硬标准与口径澄清；图/片段满足 §5/§6。
**2. 占位符**：每课给出**具体加深子话题清单**（真信息，非"写更多"）、新图/片段的完整 HTML、待核实点——非空泛。
**3. 一致性**：CJK 计数法（`[\u4e00-\u9fff]`）在 check_html 软检查、各 task 自检、spec 措辞三处一致；
新图复用既有 `flow`/`cellgroup`（无需新 CSS）。
**4. 歧义**：软检查阈值 `MIN_CJK=3000`（防线），各 task 自检按 `>=3800`（贴近 ~4000 目标）；课 03 已有 4 图、本轮只加深正文。

---

## 执行交接

计划完成，保存于 `docs/superpowers/plans/2026-06-14-llama-cpp-visual-guide-M1.6-prose-depth.md`。
建议沿用 **subagent-driven-development**：开分支 `build/m1.6-prose`，逐 task 派发实现子代理 + spec/质量双审，
内容课质量审查**回查真实源码**核实；全部完成后整体审查并合并 master。
