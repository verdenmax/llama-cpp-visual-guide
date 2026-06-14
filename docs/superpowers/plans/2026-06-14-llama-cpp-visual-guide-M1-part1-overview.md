# M1 · 第一部分（宏观全景）实施计划

> **配套 Spec：** `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`
> **路线图：** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`
> **前置：** M0 脚手架已完成并合并到 master（双语生成器 + 校验器 + 样板课 01）。
> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 逐 task 执行；
> 每个 task 跑完整 spec+质量双重审查。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 写出第一部分「宏观全景」三课（01 llama.cpp 是什么 / 02 项目全景地图 / 03 一次推理的生命周期），
并把 M0 final-review 的两组延后项（构建守卫 + 标题转义；课 01 润色）一并落地；同时引入**双语自测 quiz** 基础设施，
让每课从 M1 起就"内容完整"。完成后 `index.html` 列出 3 课、全部双语、校验全绿。

**Architecture:** 沿用 M0 生成器：在 `shell.PAGES` 追加 02/03 两课登记；内容写进 `src/part1.py`
（`LESSON_02` / `LESSON_03`）；`registry.CONTENT` 与 `shell.SUBTITLES` 追加映射；新增 `src/quizzes.py`
（双语题库 + `render(fname, lang)`）并接入 `build.py`；轻量加固 `build.py` 与 `shell.head_meta`。

**Tech Stack:** Python 3 标准库 · 自包含双语 HTML/CSS/JS（无外部依赖）。

---

## 这次确立的"内容作者方式"（重要，贯穿 M1-M9）

M0 的样板课把**整篇内容写死在 plan 里**、实现子代理只做誊写。对 40 课规模这不可持续。从 M1 起改为
**"详尽内容简报 + 子代理执笔 + 审查核实"**：

- 本 plan 为每课提供**精确到可执行**的简报：固定的卡片顺序与小节、**要画的图（直接给出 HTML 标记）**、
  **要放的代码/伪代码片段（给出经核实的真实片段）**、**本课要点**、**quiz 题目**、**必须讲到的论点清单**、
  以及**要对照核实的源码文件/符号**。
- 实现子代理据此**用中/英两种语言执笔完整 HTML**，文风对齐 M0 的课 01；并**对照真实源码核实**后标注"文件 + 符号"。
- spec 审查子代理核对结构与"无多无少"；质量审查子代理**回查 `/home/verden/course/llama.cpp` 真实源码**核实技术准确性
  （M0 已验证这套审查能抓出 `quantize`→`llama-quantize` 这类错误）。

> 准则（同 Spec §6）：多图、多伪代码/源码简化片段、尽量详细；源码引用以"文件 + 符号名"为主、不写死行号；
> `<pre>` 内 `<`/`>`/`&` 必须转义为 `&lt;`/`&gt;`/`&amp;`；代码片段内 ASCII 优先（不用 em-dash / unicode 箭头）。

---

## 文件结构（本 milestone 创建/修改）

- Modify: `src/shell.py` — `PAGES` 追加 02/03；`SUBTITLES` 追加 02/03；`head_meta()` 增加 `&/</>` 转义
- Modify: `src/build.py` — 友好的 CONTENT 缺失守卫；接入 quizzes（每课 quiz 追加到双语正文）
- Modify: `src/part1.py` — 课 01 润色；新增 `LESSON_02`、`LESSON_03`
- Modify: `src/registry.py` — `CONTENT` 追加 02/03
- Create: `src/quizzes.py` — 双语题库 `QUIZZES` + `render(fname, lang)`
- 产出（构建生成，提交）：`index.html`、`lessons/02-*.html`、`lessons/03-*.html`（01 重建）

---

## 课程文件名与登记（本 milestone）

| 课 | filename | title_zh | title_en | 副标题 zh / en |
| --- | --- | --- | --- | --- |
| 01 | `01-what-is-llamacpp.html` | llama.cpp 是什么 | What is llama.cpp | （沿用 M0）|
| 02 | `02-project-map.html` | 项目全景地图 | The project map | ggml / src·llama / common / tools / 转换脚本 · the whole tree at a glance |
| 03 | `03-inference-lifecycle.html` | 一次推理的生命周期 | Lifecycle of one inference | prompt -> 分词 -> 计算图 -> logits -> 采样 -> token · the full data flow |

> 三课同属 `part_zh="第一部分 · 宏观全景"` / `part_en="Part 1 · The Big Picture"`，
> 所以 index 仍是"1 个部分"。

---

## Task 1: 基础设施加固（标题转义 + 构建守卫）

> 来自 M0 final-review 的两条延后 Minor：① `head_meta()` 只转义了 `"`，未转义 `&/</>`，未来标题含
> `C & C++`、`vector<T>` 会破坏 `<title>`/`og:title`；② `build.py` 在 PAGES 有、CONTENT 无时抛裸 `KeyError`。

**Files:**
- Modify: `src/shell.py`（新增 `esc()`；`head_meta()` 与两处 `<title>` 用 `esc()`）
- Modify: `src/build.py`（CONTENT 缺失时友好退出）

- [ ] **Step 1: 在 shell.py 新增 esc() 助手**

在 `head_meta()` 定义**之前**插入：

```python
def esc(s):
    """Escape plain text for an HTML text/attribute context.

    For chrome/meta strings that are NOT meant to carry inline markup (page
    titles, descriptions). Do NOT use on lesson body content or bi() inputs,
    which may legitimately contain inline tags.
    """
    return (
        str(s).replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;")
    )
```

- [ ] **Step 2: head_meta() 用 esc() 全量转义**

把 `head_meta()` 开头的：
```python
    t = title.replace('"', "&quot;")
    d = description.replace('"', "&quot;")
```
替换为：
```python
    t = esc(title)
    d = esc(description)
```

- [ ] **Step 3: 两处 `<title>` 标签用 esc()**

`page()` 与 `index_page()` 各有一行 `<title>{title_tag}</title>`。把这两处分别改为：
```python
<title>{esc(title_tag)}</title>
```
（`page()` 的 `title_tag` 是 f-string 局部变量；`index_page()` 的 `title_tag` 同名。两处都改。）

- [ ] **Step 4: build.py 友好守卫**

在 `build()` 的循环里，把：
```python
    for page in shell.PAGES:
        fname = page[0]
        html = shell.page(fname, CONTENT[fname], home_href="../index.html")
```
替换为：
```python
    for page in shell.PAGES:
        fname = page[0]
        if fname not in CONTENT:
            sys.exit(f"build error: no registry.CONTENT entry for {fname!r} (declared in shell.PAGES)")
        html = shell.page(fname, CONTENT[fname], home_href="../index.html")
```

- [ ] **Step 5: 重建并验证无回归（产物零漂移）**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide && git status --short
```
Expected: `structural check passed`（0 error）、`all N internal links resolve`；`git status` 仅显示 `src/shell.py`、
`src/build.py` 被修改（`index.html`/`lessons/*` **不应**变化，因为现有标题无特殊字符，转义后字节相同）。

- [ ] **Step 6: 验证 esc() 行为与守卫**

Run:
```bash
cd src && python -c "
import shell, sys
assert shell.esc('C & C++ <x> \"q\"') == 'C &amp; C++ &lt;x&gt; &quot;q&quot;', shell.esc('C & C++ <x>')
print('esc ok')
"
(cd src && python -c "
import build, shell
shell.PAGES.append(('99-x.html','z','e','p','pe'))
try:
    build.build(); print('NO GUARD')
except SystemExit as e:
    print('guard ok:', 'no registry.CONTENT entry' in str(e))
")
```
Expected: `esc ok`, then `guard ok: True`. (The second snippet mutates an in-memory copy of PAGES only; it does not change files.)

- [ ] **Step 7: Commit**

```bash
cd /home/verden/course/llama-cpp-visual-guide
git add src/shell.py src/build.py
git commit -m "harden: escape titles in head_meta and guard missing CONTENT in build

Assisted-by: GitHub Copilot"
```

---

## Task 2: 双语 quiz 基础设施（quizzes.py + 接入 build.py）

> 引入与 langchain 模板同构、但**双语化**的题库：每题 `q/opts/why` 都带 `{"zh","en"}`；
> 选项按 `(fname, 题号)` 确定性洗牌——**zh/en 用同一套洗牌**，所以正确答案字母在两种语言里一致。
> 本 task 同时把课 01 的 quiz 一并写好（课 01 内容 M0 已就绪），让接入立即有真实产物可验。

**Files:**
- Create: `src/quizzes.py`
- Modify: `src/build.py`（import quizzes；每课 quiz 追加到双语正文）

- [ ] **Step 1: 创建 src/quizzes.py**

Create `src/quizzes.py`:

```python
"""Per-lesson bilingual self-test (自测题): design-insight multiple-choice + open prompts.

Schema per lesson::

    "NN-file.html": {
        "mcq": [
            {
                "q":   {"zh": "...", "en": "..."},
                "opts": [{"zh": "...", "en": "..."}, ...],
                "answer": 1,                      # 0-based index into opts (as written)
                "why": {"zh": "...", "en": "..."},
            },
        ],
        "open": [{"zh": "...", "en": "..."}],
    }

``render(fname, lang)`` turns it into HTML that build.py appends to the bottom of
each language's lesson body. Options are deterministically shuffled per question
(same permutation for zh and en, so the correct letter matches across languages).
"""
import hashlib

_HEAD = {"zh": "🧪 自测 · 想一想为什么这么设计", "en": "🧪 Self-test - think about the design"}
_SEE = {"zh": "看答案与解析", "en": "Show answer & explanation"}
_CLICK = {"zh": "点击展开", "en": "click to expand"}
_ANS = {"zh": "答案：", "en": "Answer: "}
_OPEN = {
    "zh": "💭 发散思考（没有标准答案，动手或动脑想想）",
    "en": "💭 Open questions (no single right answer - just think or try)",
}


def _shuffle(opts, answer, seed):
    """Deterministically permute opts (stable across builds); return
    (new_opts, new_answer_index) so the correct option lands in a varied slot."""
    order = sorted(
        range(len(opts)),
        key=lambda i: hashlib.md5(f"{seed}:{i}".encode("utf-8")).hexdigest(),
    )
    return [opts[i] for i in order], order.index(answer)


QUIZZES = {
    "01-what-is-llamacpp.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama.cpp 把“推理”从“训练框架”里彻底剥离出来。这个定位选择，最主要换来了什么？",
                    "en": "llama.cpp deliberately separates inference from the training framework. What does that choice mainly buy?",
                },
                "opts": [
                    {"zh": "更高的训练精度", "en": "Higher training accuracy"},
                    {
                        "zh": "甩掉对 Python 生态与大显存的依赖，能在消费级硬件上本地、离线、低成本跑",
                        "en": "Dropping the Python-ecosystem and large-VRAM dependency, so it runs locally/offline/cheaply on consumer hardware",
                    },
                    {"zh": "自动帮你下载模型", "en": "It downloads models for you automatically"},
                    {"zh": "让模型变得更聪明", "en": "It makes the model itself smarter"},
                ],
                "answer": 1,
                "why": {
                    "zh": "只做推理，就能用零依赖 C/C++ + 量化 + 自研 ggml 引擎压掉重依赖——这正是它“一个文件到处跑”的根本原因。",
                    "en": "By doing inference only, it can use zero-dependency C/C++ + quantization + the ggml engine to shed heavy deps - the root reason it 'runs anywhere from a single file'.",
                },
            },
            {
                "q": {
                    "zh": "整体四层结构里，真正“把算子算在硬件上（SIMD / GPU kernel）”的是哪一层？",
                    "en": "In the four-layer structure, which layer actually 'runs the ops on hardware (SIMD / GPU kernels)'?",
                },
                "opts": [
                    {"zh": "工具层 tools/", "en": "tools/ layer"},
                    {"zh": "推理层 src/llama-*", "en": "inference layer src/llama-*"},
                    {"zh": "引擎层 ggml", "en": "the ggml engine layer"},
                    {"zh": "后端层 CPU/CUDA/Metal/Vulkan", "en": "the backend layer CPU/CUDA/Metal/Vulkan"},
                ],
                "answer": 3,
                "why": {
                    "zh": "ggml 负责描述张量与计算图、调度算子；真正落到硬件上的乘加由各“后端”实现。",
                    "en": "ggml describes tensors/graphs and schedules ops; the actual hardware math is implemented by each backend.",
                },
            },
        ],
        "open": [
            {
                "zh": "什么场景你会选 llama.cpp，而不是 vLLM？反过来又是什么场景？把你的判断依据写下来。",
                "en": "When would you pick llama.cpp over vLLM, and when the reverse? Write down the criteria you'd use.",
            },
        ],
    },
}


def render(fname, lang):
    """Return the self-test HTML block for ``fname`` in ``lang`` ('' if none)."""
    data = QUIZZES.get(fname)
    if not data:
        return ""
    out = ['<div class="selftest">', f'<h2>{_HEAD[lang]}</h2>']
    for i, item in enumerate(data.get("mcq", []), 1):
        shuffled, ans = _shuffle(item["opts"], item["answer"], f"{fname}:{i}")
        opts = "\n".join(f"    <li>{o[lang]}</li>" for o in shuffled)
        letter = chr(65 + ans)
        out.append(
            f'<div class="quiz">\n'
            f'  <div class="qn">{i}. {item["q"][lang]}</div>\n'
            f'  <ol class="opts">\n{opts}\n  </ol>\n'
            f'  <details class="accordion">\n'
            f'    <summary>{_SEE[lang]} <span class="hint">{_CLICK[lang]}</span></summary>\n'
            f'    <div class="acc-body"><div class="qa"><div class="a">'
            f'<strong>{_ANS[lang]}{letter}</strong>。{item["why"][lang]}'
            f"</div></div></div>\n"
            f"  </details>\n"
            f"</div>"
        )
    opens = data.get("open", [])
    if opens:
        lis = "\n".join(f"    <li>{o[lang]}</li>" for o in opens)
        out.append(
            '<div class="card spark">\n'
            f'  <div class="tag">{_OPEN[lang]}</div>\n'
            f"  <ul>\n{lis}\n  </ul>\n"
            "</div>"
        )
    out.append("</div>")
    return "\n".join(out)
```

- [ ] **Step 2: 接入 build.py（每课 quiz 追加到双语正文）**

在 `src/build.py` 顶部 import 区，把：
```python
import shell  # noqa: E402
from registry import CONTENT  # noqa: E402
```
替换为：
```python
import shell  # noqa: E402
import quizzes  # noqa: E402
from registry import CONTENT  # noqa: E402
```

然后把 `build()` 循环里（Task 1 已加守卫后的）这段：
```python
    for page in shell.PAGES:
        fname = page[0]
        if fname not in CONTENT:
            sys.exit(f"build error: no registry.CONTENT entry for {fname!r} (declared in shell.PAGES)")
        html = shell.page(fname, CONTENT[fname], home_href="../index.html")
```
替换为：
```python
    for page in shell.PAGES:
        fname = page[0]
        if fname not in CONTENT:
            sys.exit(f"build error: no registry.CONTENT entry for {fname!r} (declared in shell.PAGES)")
        base = CONTENT[fname]
        content = {
            "zh": base["zh"] + quizzes.render(fname, "zh"),
            "en": base["en"] + quizzes.render(fname, "en"),
        }
        html = shell.page(fname, content, home_href="../index.html")
```

- [ ] **Step 3: 单元验证 render() 双语一致**

Run:
```bash
cd src && python -c "
import quizzes
zh = quizzes.render('01-what-is-llamacpp.html','zh')
en = quizzes.render('01-what-is-llamacpp.html','en')
assert '自测' in zh and 'Self-test' in en
assert zh.count('<details') == en.count('<details') == 2
assert zh.count('<summary') == 2 and en.count('<summary') == 2
# correct-answer letter must be identical across languages (same shuffle)
import re
lz = re.findall(r'答案：([A-D])', zh); le = re.findall(r'Answer: ([A-D])', en)
assert lz == le and len(lz) == 2, (lz, le)
print('quiz render ok; letters', lz)
"
```
Expected: `quiz render ok; letters ['?', '?']`（两个字母，zh/en 相同）。

- [ ] **Step 4: 重建 + 校验，确认 quiz 进入双语产物**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -c 'class="selftest"' lessons/01-what-is-llamacpp.html
grep -c 'class="quiz"' lessons/01-what-is-llamacpp.html
```
Expected: `structural check passed`（0 error）、`all N internal links resolve`；`selftest` 出现 **2** 次
（中/英各一份）、`quiz` 出现 **4** 次（2 题 x 双语）。

- [ ] **Step 5: Commit**

```bash
git add src/quizzes.py src/build.py index.html lessons/01-what-is-llamacpp.html
git commit -m "feat: add bilingual quiz infrastructure and lesson 01 self-test

Assisted-by: GitHub Copilot"
```

---

## Task 3: 课 01 润色（并入 ST5 review 三条）

> 来自 M0 ST5 质量审查的可选润色：① 量化位宽措辞更准确（K-quant 还有 2/3/6 bit）；
> ② 最小片段里引入 `vocab`/`smpl`，使其自洽；③ 英文 transformers 定位 "Model hub" -> "Model library" 对齐中文"模型库"。
> 符号已核实：`llama_model_get_vocab`、`llama_sampler_chain_init`、`llama_sampler_chain_default_params`、
> `llama_sampler_chain_add`、`llama_sampler_init_greedy`（`include/llama.h`，核验 2026-06-14）。

**Files:**
- Modify: `src/part1.py`（`LESSON_01` 的 zh/en 字符串）
- 产出：重建 `lessons/01-what-is-llamacpp.html`

- [ ] **Step 1: zh 片段引入 vocab/smpl**

把 zh 片段中：
```
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// 新接口</span>

<span class="cm">// 1) prompt 切成 token</span>
```
替换为：
```
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// 新接口</span>

const llama_vocab *vocab = <span class="fn">llama_model_get_vocab</span>(model);
llama_sampler     *smpl  = <span class="fn">llama_sampler_chain_init</span>(<span class="fn">llama_sampler_chain_default_params</span>());
<span class="fn">llama_sampler_chain_add</span>(smpl, <span class="fn">llama_sampler_init_greedy</span>());  <span class="cm">// 最简：贪心采样</span>

<span class="cm">// 1) prompt 切成 token</span>
```

- [ ] **Step 2: en 片段引入 vocab/smpl**

把 en 片段中：
```
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// new API</span>

<span class="cm">// 1) split the prompt into tokens</span>
```
替换为：
```
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// new API</span>

const llama_vocab *vocab = <span class="fn">llama_model_get_vocab</span>(model);
llama_sampler     *smpl  = <span class="fn">llama_sampler_chain_init</span>(<span class="fn">llama_sampler_chain_default_params</span>());
<span class="fn">llama_sampler_chain_add</span>(smpl, <span class="fn">llama_sampler_init_greedy</span>());  <span class="cm">// simplest: greedy</span>

<span class="cm">// 1) split the prompt into tokens</span>
```

- [ ] **Step 3: 量化位宽措辞（zh + en）**

zh：把 `<strong>量化</strong>（把权重压成 4/5/8 bit）` 替换为
`<strong>量化</strong>（把权重压成 4/5/8 bit 等低位宽，K-quant 甚至能到 2/3/6 bit）`。

en：把 `<strong>quantization</strong> (compressing weights to 4/5/8 bits)` 替换为
`<strong>quantization</strong> (compressing weights to e.g. 4/5/8 bits, down to 2/3/6-bit K-quants)`。

- [ ] **Step 4: en transformers 定位措辞**

把 en 对比表里 `<td>Model hub / high-level wrapper</td>` 替换为
`<td>Model library / high-level wrapper</td>`（对齐中文"模型库 / 高层封装"）。

- [ ] **Step 5: 重建 + 校验 + 片段自洽检查**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd src && python -c "
import part1
for lang in ('zh','en'):
    s = part1.LESSON_01[lang]
    assert 'llama_model_get_vocab' in s and 'llama_sampler_chain_init' in s, (lang,'snippet')
    assert 'llama_sampler_init_greedy' in s, (lang,'greedy')
assert 'Model hub' not in part1.LESSON_01['en'], 'model hub still present'
assert '2/3/6' in part1.LESSON_01['zh'] and '2/3/6' in part1.LESSON_01['en'], 'kquant wording'
print('refine ok')
"
```
Expected: `structural check passed`（0 error）、`all N internal links resolve`、`refine ok`。

- [ ] **Step 6: Commit**

```bash
cd /home/verden/course/llama-cpp-visual-guide
git add src/part1.py lessons/01-what-is-llamacpp.html
git commit -m "polish: refine lesson 01 (quant wording, self-contained snippet, transformers label)

Assisted-by: GitHub Copilot"
```

---

## Task 4: 课 02「项目全景地图 / The project map」

> 新课，按"详尽简报 + 子代理执笔"产出。先做登记编辑（PAGES/SUBTITLES/registry），
> 再据简报用中/英执笔 `LESSON_02`，并加该课 quiz。**目录与角色已核实**（见下，核验 2026-06-14）。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 02 一项）
- Modify: `src/registry.py`（`CONTENT` 加 02）
- Modify: `src/part1.py`（新增 `LESSON_02`）
- Modify: `src/quizzes.py`（`QUIZZES` 加 02）
- 产出：`index.html`（现 2 课）、`lessons/02-project-map.html`

- [ ] **Step 1: 登记 PAGES**

在 `shell.py` 的 `PAGES` 列表里，01 那条之后追加：
```python
    ("02-project-map.html", "项目全景地图", "The project map",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
```

- [ ] **Step 2: 登记 SUBTITLES**

在 `shell.py` 的 `SUBTITLES` 字典里追加：
```python
    "02-project-map.html": ("ggml / src·llama / common / tools / 转换脚本",
                            "ggml / src·llama / common / tools / converters"),
```

- [ ] **Step 3: 登记 registry**

在 `registry.py` 的 `CONTENT` 字典里追加：
```python
    "02-project-map.html": part1.LESSON_02,
```

- [ ] **Step 4: 在 part1.py 执笔 LESSON_02（中/英双语）**

新增 `LESSON_02 = {"zh": r"""...""", "en": r"""..."""}`，文风对齐课 01。**结构（按序）**：

1. `<p class="lead">`：仓库第一眼看着很大，但其实是清晰分层的。给你一张"园区地图"，照着读就不迷路。
2. `<div class="card analogy">`（🔌 生活类比）：把仓库想成一座**工厂园区**，目录就是**园区导览图**——
   每个车间（目录）只干一件事，知道地图就知道该去哪。
3. `<h2>` 顶层目录速览 + `<table class="t">`（两列：目录 / 作用）。**用下列已核实的行**（en 自行翻译，保持同序同表）：
   - `ggml/` — 自研**张量引擎**：张量 · 计算图 · 算子 · 后端调度；独立子项目（自带 `include/` 与 `src/`）
   - `ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …` — 各**硬件后端**实现（还有 hip/sycl/musa/opencl 等十余种）
   - `src/` — **llama 推理库**：`llama-model-loader` · `llama-graph` · `llama-kv-cache` · `llama-sampler` · `llama-vocab` · `llama-chat` · `llama-grammar` · `llama-quant` …
   - `include/` — **公共 C API**：`llama.h`；`llama-cpp.h`（C++ 薄封装 + RAII）
   - `common/` — **复用工具**：`arg`（参数解析）· `sampling`（采样封装）· `chat` · `log` · `download` · `json-schema-to-grammar` …
   - `tools/` — **可执行程序**：`llama-cli` · `llama-server` · `llama-quantize` · `mtmd`（多模态）· `perplexity` · `llama-bench` …
   - `examples/` — 小型示例程序（如 `simple`）
   - `gguf-py/` — **Python 的 GGUF 读写库**
   - `convert_hf_to_gguf.py` 等 — **HuggingFace -> GGUF** 转换脚本（共 4 个 `convert_*.py`）
   - `models/` · `tests/` · `docs/` · `grammars/` · `cmake/` — 模型数据 / 测试 / 文档 / GBNF 示例 / 构建系统
4. `<h2>` 它们怎么对上"四层" + 一个 `<div class="layers">`（**复用课 01 的四层组件**，但每层标注真实目录）：
   - `l-app`（工具/应用）：`tools/`（cli · server · quantize · mtmd …）、`examples/`
   - `l-part`（推理库）：`src/llama-*`、`include/llama.h`
   - `l-main`（引擎）：`ggml/`（`ggml.c` · `gguf.cpp` · `ggml-alloc` · `ggml-backend`）
   - `l-core`（后端）：`ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …`
   - 紧跟一段话说明**两条旁路**：① 模型准备 `gguf-py/` + `convert_*.py`（Python）产出 `.gguf` 喂给引擎；
     ② 支撑设施 `common/`（胶水）、`tests/` `docs/` `cmake/`。
5. `<div class="card detail">`（🔬 细节 / 源码对应）：用一句话把"读源码该从哪进"指出来——
   想懂**用法**看 `tools/` 与 `examples/simple`；想懂**推理逻辑**看 `src/llama-*`；想懂**底层算子**看 `ggml/`；
   **对外契约**只有一个 `include/llama.h`。（此课不放 `<pre>` 代码块，避免树形字符；目录用表格/行内 `code` 呈现。）
6. `<div class="card key">`（✅ 本课要点 / Key points）。**要点（双语，照此写）**：
   - 仓库 = `ggml`（引擎）+ `src/llama-*`（推理库）+ `common`（胶水）+ `tools`（程序）+ `gguf-py`/`convert_*`（模型准备）。
   - 对外公共 API 只有一个 `include/llama.h`，是整个项目的"对外契约"。
   - `ggml` 是**独立可复用引擎**，`llama` 只是它的一个使用者（同一个 ggml 也被别的项目用）。
   - **模型准备（Python 转换）与运行（C++ 推理）彻底分离**，桥梁就是 `.gguf` 文件。
7. `<div class="card spark">`（💡 设计亮点 / Design insight）：**引擎与模型逻辑分层** + **公共 API 收口于单一头文件** +
   **Python 准备 / C++ 运行解耦** —— 让 ggml 能独立演进、llama 能被轻量嵌入、模型转换不拖累运行时。

**必须讲到的论点**：四层映射到真实目录；`include/llama.h` 是唯一对外 API；ggml 是独立子项目；`.gguf` 是 Python 与 C++ 的边界。
**要核实**（对照 `/home/verden/course/llama.cpp`）：上述目录名与 `tools/` 程序名（`llama-cli`/`llama-server`/`llama-quantize`）、
`include/` 仅 `llama.h`+`llama-cpp.h`、`convert_*.py` 数量；引用以"目录 + 文件名"为主、不写行号。

- [ ] **Step 5: 在 quizzes.py 增加 02 的双语 quiz**

在 `QUIZZES` 字典里追加键 `"02-project-map.html"`，含 2 道 mcq + 1 道 open（schema 同 01）：
- **MCQ1**：q `{"zh":"整个项目对外的公共 C API 主要收在哪里？","en":"Where does the project's public C API mainly live?"}`；
  opts（answer=正确项下标）：`include/llama.h`（✅正确）/ `src/llama.cpp` / `common/common.h` / `ggml/include/ggml.h`；
  why `{"zh":"对外契约只有 include/llama.h（外加 llama-cpp.h 的 C++ 薄封装）；src 与 ggml 是内部实现。","en":"The public contract is just include/llama.h (plus the llama-cpp.h C++ wrapper); src and ggml are internal."}`
- **MCQ2**：q `{"zh":"把一个 HuggingFace 模型变成能被 llama.cpp 运行的文件，靠的是哪部分？","en":"What turns a HuggingFace model into a file llama.cpp can run?"}`；
  opts：`gguf-py/ + convert_*.py（Python 转换脚本）`（✅）/ `tools/llama-quantize` / `src/llama-model-loader` / `ggml 后端`；
  why `{"zh":"转换在 Python 侧（gguf-py + convert_*.py）产出 .gguf；C++ 运行时只负责加载已是 GGUF 的文件。","en":"Conversion happens in Python (gguf-py + convert_*.py) producing .gguf; the C++ runtime only loads already-GGUF files."}`
- **OPEN**：`{"zh":"如果要新增一个采样策略，你认为应该改哪个目录？为什么不是 ggml/？","en":"If you were adding a new sampling strategy, which directory would you change - and why not ggml/?"}`

- [ ] **Step 6: 重建 + 校验**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 2 课 · 1 个部分' index.html && echo "index shows 2 lessons"
grep -q 'href="lessons/02-project-map.html"' index.html && echo "toc links 02"
for L in zh en; do echo "$L:"; grep -c 'class="lang-'$L'"' lessons/02-project-map.html; done
```
Expected: `structural check passed`（0 error）、`all N internal links resolve`、`index shows 2 lessons`、`toc links 02`，
且 02 页 `lang-zh`/`lang-en` 均 >=1。check_html 的"both languages / 本课要点 / analogy / 计数"全过。

- [ ] **Step 7: Commit**

```bash
git add src/shell.py src/registry.py src/part1.py src/quizzes.py index.html lessons/02-project-map.html
git commit -m "feat: add lesson 02 project map (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 5: 课 03「一次推理的生命周期 / Lifecycle of one inference」

> 新课，全章最适合画**数据流图**。把课 01 的最小 C-API 主线"放慢镜头"，看一个 token 怎么从 prompt 生出来。
> **符号已核实**（`include/llama.h` + `src/`，核验 2026-06-14）：`llama_tokenize` · `llama_batch_get_one` ·
> `llama_decode`（`src/llama-context.cpp`）· `llama_get_logits_ith` · `llama_sampler_sample`（`src/llama-sampler.cpp`）·
> `llama_vocab_is_eog` · `llama_token_to_piece`（`src/llama-vocab.cpp`）· 建图 `src/llama-graph.cpp`（`llm_graph_*`）·
> KV cache `src/llama-kv-cache.cpp`。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 03）
- Modify: `src/registry.py`（`CONTENT` 加 03）
- Modify: `src/part1.py`（新增 `LESSON_03`）
- Modify: `src/quizzes.py`（`QUIZZES` 加 03）
- 产出：`index.html`（现 3 课）、`lessons/03-inference-lifecycle.html`

- [ ] **Step 1: 登记 PAGES**（02 那条之后追加）
```python
    ("03-inference-lifecycle.html", "一次推理的生命周期", "Lifecycle of one inference",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
```

- [ ] **Step 2: 登记 SUBTITLES**
```python
    "03-inference-lifecycle.html": ("prompt -> 分词 -> 计算图 -> logits -> 采样 -> token",
                                    "prompt -> tokenize -> graph -> logits -> sample -> token"),
```

- [ ] **Step 3: 登记 registry**
```python
    "03-inference-lifecycle.html": part1.LESSON_03,
```

- [ ] **Step 4: 在 part1.py 执笔 LESSON_03（中/英双语）**

新增 `LESSON_03 = {"zh": r"""...""", "en": r"""..."""}`，文风对齐课 01。**结构（按序）**：

1. `<p class="lead">`：课 01 给了最小主线，这一课把它**放慢镜头**——看一个 token 如何从 prompt 一步步生出来，再被接回去继续。
2. `<div class="card analogy">`（🔌 生活类比）：像**流水线接力**——prompt 整段进料，过几个工位吐出一个字；
   再把这个字**接回队尾**，开始下一棒。一次只多产出一个字。
3. `<h2>` 七步数据流 + 一个 `<div class="vflow">`（**竖向编号步骤**；每步 `num` + `h4` 标题 + `p` 说明 + `.mono` 文件）。**七步（照此写，en 同序翻译）**：
   1. **分词 Tokenize** — prompt 文本 -> token id 序列 · `src/llama-vocab.cpp`（`llama_tokenize`）
   2. **组批 Batch** — token + 位置 `pos` + 序列 `seq_id` 打包成一次输入 · `src/llama-batch.cpp`（`llama_batch_get_one`）
   3. **解码 Decode（前向）** — `llama_decode` 跑一次前向 · `src/llama-context.cpp`；
      内部：**建计算图** `src/llama-graph.cpp`（`llm_graph_*`）+ `src/llama-model.cpp`，再**在后端执行**（`ggml-backend`）
   4. **取 logits** — 拿到"下一个 token 的分数向量" · `llama_get_logits_ith`
   5. **采样 Sample** — 从 logits 里选一个 token · `src/llama-sampler.cpp`（`llama_sampler_sample`，sampler chain）
   6. **判结束 + 还原文字** — `llama_vocab_is_eog` 判结束符；`llama_token_to_piece` 把 token 还原成文字片段 · `src/llama-vocab.cpp`
   7. **回灌 + 循环** — 新 token 作为下一步输入再 `decode`；过去的 K/V 存在 `src/llama-kv-cache.cpp`，**无需重算**
4. `<div class="card macro">`（🌍 宏观理解）：**prefill vs decode + 为什么循环不贵**。
   第一次把**整段 prompt 并行**算（prefill，填满 KV cache）；之后每步 decode **只算 1 个新 token**，
   靠 KV cache 复用过去的 K/V，不必重算整段历史。
5. `<div class="card detail">`（🔬 细节 / 源码对应）：把七步对回课 01 的调用，放一段**极简伪代码** `<pre class="code">`
   （引用真实符号；`<pre>` 内不得有裸 `<`/`&`，本片段无需特殊字符）：
```
<span class="cm">// 课 01 主线的"慢镜头"：每一步对应一个调用</span>
tokens = <span class="fn">llama_tokenize</span>(vocab, prompt)         <span class="cm">// 1 分词</span>
batch  = <span class="fn">llama_batch_get_one</span>(tokens)           <span class="cm">// 2 组批</span>
<span class="kw">loop</span>:
    <span class="fn">llama_decode</span>(ctx, batch)                  <span class="cm">// 3 前向(内部建图 + 后端执行)</span>
    logits = <span class="fn">llama_get_logits_ith</span>(ctx, -1)     <span class="cm">// 4 取 logits</span>
    id     = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1) <span class="cm">// 5 采样</span>
    <span class="kw">if</span> <span class="fn">llama_vocab_is_eog</span>(vocab, id): <span class="kw">break</span>  <span class="cm">// 6 结束?</span>
    print(<span class="fn">llama_token_to_piece</span>(vocab, id))      <span class="cm">// 6 还原文字</span>
    batch = <span class="fn">llama_batch_get_one</span>([id])          <span class="cm">// 7 回灌; KV cache 记住过去</span>
```
   （en 版把注释翻成英文，结构不变。）
6. `<div class="card key">`（✅ 本课要点 / Key points）：
   - 一次推理 = **分词 -> 组批 -> 解码(前向) -> 取 logits -> 采样 -> 还原文字 -> 回灌循环**。
   - "解码一次"内部 = **建计算图 + 在后端执行**；输出是**下一个 token 的 logits**（不是文字、也不是选好的 token）。
   - **prefill** 把整段 prompt 一次并行算；之后每步**只算一个新 token**。
   - **KV cache** 记住过去的 K/V，是"循环不重算、跑得动"的关键。
7. `<div class="card spark">`（💡 设计亮点 / Design insight）：**自回归 + KV cache** 把每步从"重算整段历史"降为
   "只算一个新 token"，避免了朴素实现的 O(n^2) 重复计算——这是本地能跑大模型的关键之一。

**必须讲到的论点**：decode 的输出是 logits（采样才得 token）；prefill/decode 区别；KV cache 让循环便宜。
**要核实**：上述文件/符号名对照 `/home/verden/course/llama.cpp`；引用以"文件 + 符号"为主、不写行号。

- [ ] **Step 5: 在 quizzes.py 增加 03 的双语 quiz**

在 `QUIZZES` 追加 `"03-inference-lifecycle.html"`，2 mcq + 1 open：
- **MCQ1**：q `{"zh":"llama_decode 跑一次前向，直接产出的是什么？","en":"What does a single llama_decode forward pass directly produce?"}`；
  opts：`下一个 token 的 logits（分数向量）`（✅）/ `最终要显示的文字` / `一个已经选好的 token` / `更新后的模型权重`；
  why `{"zh":"decode 只算出 logits；选哪个 token 是采样器 llama_sampler_sample 的事，还原文字是 llama_token_to_piece 的事。","en":"decode only yields logits; picking a token is llama_sampler_sample's job, turning it into text is llama_token_to_piece's."}`
- **MCQ2**：q `{"zh":"自回归循环里，为什么每生成一个新 token 不必把整段历史重算一遍？","en":"In the autoregressive loop, why doesn't each new token require recomputing the whole history?"}`；
  opts：`因为 KV cache 缓存了过去 token 的 K/V`（✅）/ `因为 prompt 很短` / `因为用了 GPU` / `因为权重被量化了`；
  why `{"zh":"prefill 填满 KV cache 后，每步 decode 只算新 token 的 Q 并复用缓存的 K/V，省掉对历史的重复计算。","en":"After prefill fills the KV cache, each decode step only computes the new token's Q and reuses cached K/V, skipping recomputation over history."}`
- **OPEN**：`{"zh":"如果完全不用 KV cache，生成第 1000 个 token 的成本会怎么变化？这对本地推理意味着什么？","en":"Without any KV cache, how would the cost of generating the 1000th token change - and what would that mean for local inference?"}`

- [ ] **Step 6: 重建 + 校验**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 3 课 · 1 个部分' index.html && echo "index shows 3 lessons"
grep -q 'class="vflow"' lessons/03-inference-lifecycle.html && echo "has vflow diagram"
for L in zh en; do echo "$L:"; grep -c 'class="lang-'$L'"' lessons/03-inference-lifecycle.html; done
```
Expected: `structural check passed`（0 error）、`all N internal links resolve`、`index shows 3 lessons`、`has vflow diagram`，
03 页双语均 >=1。

- [ ] **Step 7: Commit**

```bash
git add src/shell.py src/registry.py src/part1.py src/quizzes.py index.html lessons/03-inference-lifecycle.html
git commit -m "feat: add lesson 03 inference lifecycle (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 6: M1 验收（清重建 + 校验 + 标记里程碑）

**Files:** 无新增；端到端验证 M1。

- [ ] **Step 1: 清重建 + 双校验**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
```
Expected：`Wrote 4 files ...`（index + 3 课）、`structural check passed`（**0 error**，理想 0 warning）、`all N internal links resolve`。

- [ ] **Step 2: 三课与双语自查**
```bash
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 3 课 · 1 个部分' index.html && echo "count ok"
for f in 01-what-is-llamacpp 02-project-map 03-inference-lifecycle; do
  echo "== $f =="
  grep -c 'class="lang-zh"' lessons/$f.html; grep -c 'class="lang-en"' lessons/$f.html
  grep -c 'class="selftest"' lessons/$f.html   # 每课应有 2（中/英各一份 quiz）
done
git status --short && echo "(empty = clean)"
```
Expected：`count ok`；每课 `lang-zh`/`lang-en` 均 >=1、`selftest` == 2；`git status` 干净（产物已提交）。

- [ ] **Step 3: 标记路线图 M1 完成**

编辑 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`：
- 状态追踪：`- [ ] M1 第一部分（宏观全景）` -> `- [x] M1 第一部分（宏观全景）`。
- 里程碑总表 M1 行：状态 `待写` -> `完成`。

```bash
cd /home/verden/course/llama-cpp-visual-guide
git add docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md
git commit -m "docs: mark M1 part-1 milestone complete

Assisted-by: GitHub Copilot"
```

---

## 验收标准（Definition of Done · M1）

- `index.html` 列出 **3 课**（`共 3 课 · 1 个部分`），全部双语、顶部可切换。
- 课 01 已润色（量化措辞 / 自洽片段 / transformers 标签）且带 quiz；课 02、03 为新课，含**结构图/数据流图**、
  源码对应（文件 + 符号、对照真实仓库核实）、本课要点、设计亮点、双语 quiz。
- `check_html.py` 0 error；`check_links.py` 0 死链；每课 `selftest` 双份。
- 基础设施加固到位：`head_meta`/`<title>` 转义、`build.py` 友好守卫、双语 quiz 基础设施可用。
- `git status` 干净；路线图 M1 标记完成。

---

## Self-Review（plan 作者自审）

**1. Spec 覆盖**：本 M1 覆盖 Spec §4 第一部分课 1-3、§5 页面解剖（卡片齐全 + 图 + quiz）、§6 内容准则
（多图、伪代码/源码片段、文件+符号不写行号、`<pre>` 转义、代码内 ASCII）、§7 配套中的 quiz、§8 的 M1 条目。
build_print/PDF、CI、README 仍属 M9，未在 M1 范围。

**2. 占位符扫描**：登记编辑、加固、quiz 均给出完整可执行代码；两课正文为**详尽简报**（结构 + 图谱 + 已核实的目录/符号 +
双语要点 + 双语 quiz 原文 + 必讲论点 + 待核实清单），非空泛占位——子代理可据此直接执笔，审查可据此判"无多无少"。

**3. 类型 / 命名一致性**：
- `PAGES` 5 元组、`SUBTITLES`/`CONTENT` 键、`LESSON_02`/`LESSON_03` 的 `{"zh","en"}` dict 形状与 M0 一致。
- `quizzes.QUIZZES` schema：`mcq:[{q:{zh,en}, opts:[{zh,en}], answer:int, why:{zh,en}}], open:[{zh,en}]`；
  `render(fname, lang)` 与 `build.py` 的 `quizzes.render(fname, "zh"/"en")` 调用一致；选项洗牌 zh/en 同序、答案字母一致。
- 文件名 `02-project-map.html`、`03-inference-lifecycle.html` 在 PAGES/SUBTITLES/registry/quizzes/产物五处一致。
- `esc()` 仅用于 chrome/meta，不用于正文与 `bi()`（避免吃掉内联标记）。

**4. 歧义检查**：内容采用"详尽简报 + 子代理执笔 + 审查回查源码"（已与用户确认）；每课明确"要核实"的源码位置，
要求引用"文件 + 符号"、不写行号；`<pre>` 片段已确保无裸 `<`/`&`。check_html 的 `MIN_CONTENT`/双语/计数门槛对新课均适用。

---

## 执行交接

计划完成，保存于 `docs/superpowers/plans/2026-06-14-llama-cpp-visual-guide-M1-part1-overview.md`。
建议沿用 M0 的 **subagent-driven-development**：先开分支 `build/m1-part1`，逐 task 派发实现子代理 + spec/质量双审，
每个内容课额外要求质量审查**回查 `/home/verden/course/llama.cpp` 真实源码**核实技术准确性；全部完成后做整体审查并合并到 master。
