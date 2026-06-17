# M5 · 第五部分（公共 API 与工具）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 逐课执行本计划（每课一个 task，task 内步骤用 `- [ ]` 勾选跟踪）。
> **配套设计：** `docs/superpowers/specs/2026-06-17-m5-part5-tools-design.md`（父级总设计 `...2026-06-13-...-design.md`，roadmap `...2026-06-13-...-roadmap.md`）。

**Goal:** 为图解教程补齐**第五部分 · 公共 API 与工具**（第 25-30 课，共 6 课），视角从"内部原理"转向"对外接口 + 工具程序"，每课沿用 M1-M4 基线并在初稿内嵌 worked-example trace 图。

**Architecture:** 沿用既有零依赖 Python 静态站点生成器。**新建** `src/part5.py`，写 `LESSON_25..30` 双语内容字典；每课在 `src/shell.py`（`PAGES`/`SUBTITLES`）、`src/registry.py`（`CONTENT`）、`src/quizzes.py`（`QUIZZES`）登记；`python3 src/build.py` 生成 HTML，`check_html.py`/`check_links.py` 校验。index 课数 24 -> 30，部分数 4 -> **5**（新增"第五部分"）。trace 复用既有 `.trace` 组件（Style A 纯 HTML 站点流 / Style C 内联 SVG）。

**Tech Stack:** Python 3（仅标准库）· 自包含 HTML/CSS/JS · 既有 `.trace` 组件。

---

## 里程碑范围（M5 = 课 25-30）

| 课 | slug（产出文件） | 标题 zh / en | 内嵌 trace（风格） |
| --- | --- | --- | --- |
| 25 | `25-c-api.html` | C API 总览 / The C API | 典型调用序列（A） |
| 26 | `26-common.html` | common 工具层 / The common layer | 命令行参数 -> params（A） |
| 27 | `27-llama-cli.html` | llama-cli / llama-cli | 生成主循环一轮（A） |
| 28 | `28-llama-server.html` | llama-server / llama-server | 连续批处理 slot x 时间（C，SVG） |
| 29 | `29-quantize-tool.html` | quantize 工具 / The quantize tool | imatrix 加权量化一块（A） |
| 30 | `30-eval-bench.html` | 评测与基准 / Evaluation & benchmarks | 一段序列算 PPL（A） |

> 全部在同一个"第五部分 · 公共 API 与工具 / Part 5 · Public API & tools"标签下。L28 只做架构总览，深度调度留 L35（第七部分）。

## 统一交付标准（每课硬性达标，与 M1-M4 一致）

每课 Step 4 写 `LESSON_NN = {"zh": r'''...''', "en": r'''...'''}`，须满足：

- **结构**：导语 `<p>` + 教学卡片（`macro`/`detail`/`analogy`/`key`/`spark` 至少各类酌用，>=2 张深挖 `<details>`）+ **>=3 个图示**（`flow`/`vflow`/`cols`/`cellgroup`/`layers`/`timeline`/`trace` 之一，双语合计，单语 >=3）+ **>=2 段伪代码或真实源码简化片段**（`<pre class="code">`）+ **1 个内嵌 worked-example trace**（见各课）。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（trace 的 `.trace` div 与内联 `<svg>` 不计入 `<p>`；trace 前的导语 `<p>` 要中英同节各加一个，或并入既有段以守恒）。
- **中文密度**：zh CJK >= 4000；**en CJK == 0**（纯 ASCII；代码/SVG 文本也用 `-`/`->`/`...`，不用 em-dash / unicode 箭头；英文 SVG `<text>` 纯 ASCII）。
- **无文字墙**：连续顶层 `<p>` <= 3（遇墙用图/卡片/trace 打断）。
- **转义**：渲染 HTML 里 `<`/`>`/`&` 须转义（`&lt;`/`&gt;`/`&amp;`）；**无双重转义**（无 `&amp;lt;`）；HTTP/JSON/代码片段里的 `<`、`{`、`"` 按需处理（`{`/`}`/`"` 在 HTML 文本中合法，`<`/`>` 必转义）。
- **trace 规范**：内联 `<svg>` 须是合法 XML（`viewBox` + `width="100%"` + `role="img"` + `aria-label`，zh aria-label 用中文、en 纯 ASCII）；深色模式可读（白底配 `#1d2129`、accent `#c2630e` 配 `#fff`、标签 `#5b6470`，沿用 trace 系列色板）；trace 不与 `<div class="card">` 紧邻（中间须有 `<p>`/`<h2>`）。
- **源码引用**：以"文件 + 符号名"为主、不写死行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验日期 2026-06-17）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测（沿用既有 `QUIZZES` 格式）。
- **登记**：`shell.PAGES`（filename, zh/en 短标题, `第五部分 · 公共 API 与工具`/`Part 5 · Public API & tools`）、`shell.SUBTITLES`（zh/en 副标题）、`registry.CONTENT`（filename -> `part5.LESSON_NN`）。

## 执行方式

- superpowers:subagent-driven-development，**一课一个 task**（Task 1=课25 ... Task 6=课30，顺序执行；Task 7 收尾）。
- 每个 task：实现子代理 -> **spec 合规审查子代理 -> 质量审查子代理**（两段审查，沿用偏好），修复回环后再标完成。子代理一律用当前主会话模型。
- 控制器把每个 task 的完整文本喂给子代理（不让子代理读 plan 文件）。
- 全程对照真实源码；commit 用 `Assisted-by: GitHub Copilot`（非 Co-authored-by）。
- 分支：在 master 上从本 plan 提交后，新建 `feature/part5-tools` 分支做实现（沿用前序惯例）。

---

## Task 1: 课 25「C API 总览 / The C API」

**Files:** **新建** `src/part5.py`（写 `LESSON_25`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `25-c-api.html`。

**源码事实（核实 2026-06-17，`/home/verden/course/llama.cpp`，引用文件+符号、无行号）：**
- 头文件 `include/llama.h`（C ABI，`LLAMA_API` 导出）。不透明句柄：`llama_model`（只读知识，可多会话共享）、`llama_context`（每会话状态：KV cache/计算资源）、`llama_vocab`（`llama_model_get_vocab` 取）、`llama_sampler`、`llama_memory_t`、`llama_adapter_lora`。
- 典型调用序列：`llama_backend_init()` -> `llama_model_load_from_file(path, params)`（多分片用 `llama_model_load_from_splits`）-> `llama_init_from_model(model, ctx_params)`（注意 `llama_new_context_with_model` 已 `DEPRECATED`，名字改了语义没变）-> `llama_model_get_vocab` -> `llama_tokenize(vocab, text, ...)` -> 填 `llama_batch`（简单情形 `llama_batch_get_one`）-> `llama_decode(ctx, batch)` -> `llama_get_logits(ctx)` -> 采样器链（`llama_sampler_chain_params` 建链、`llama_sampler_sample(smpl, ctx, idx)` 选 token）-> `llama_token_to_piece(vocab, token, ...)` -> 循环；收尾 `llama_free(ctx)` / `llama_model_free(model)` / `llama_backend_free()`。
- C++ 便利头 `include/llama-cpp.h`（仅 30 行）：`std::unique_ptr` 别名 `llama_model_ptr`/`llama_context_ptr`/`llama_sampler_ptr`/`llama_adapter_lora_ptr`，deleter 调对应 `llama_*_free`，自动释放。

- [ ] **Step 1-3: 登记**（三处）

```python
# shell.py PAGES 追加（第五部分起点）：
("25-c-api.html", "C API 总览", "The C API", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES 追加：
"25-c-api.html": ("include/llama.h：句柄、调用序列、C++ RAII 包装", "include/llama.h: handles, call sequence, the C++ RAII wrappers"),
# registry.py CONTENT 追加：
"25-c-api.html": part5.LESSON_25,
# registry.py 顶部 import 追加：import part5
```

- [ ] **Step 4: 执笔 `LESSON_25`（双语，新建 `src/part5.py`，文件头部加注释 `# Part 5 · 公共 API 与工具`）。结构：**
  - 导语 `<p>`：C API 是 llama.cpp 的"总闸"——cli / server / 各语言绑定全都通过 `include/llama.h` 这套稳定 C 函数驱动模型；前四部分讲的内部机件，这一课起从"怎么对外用"重新串一遍。
  - `<h2>` 句柄与所有权：四个不透明句柄。`llama_model`=只读知识（可共享，呼应 L14/L17），`llama_context`=每会话状态（KV cache，呼应 L17/L19），`llama_vocab`（L20）、`llama_sampler`（L21）。C 里手动 `_free`；C++ 用 `llama-cpp.h` 的 `_ptr`（`unique_ptr`）自动释放。**图**：`cols` 或 `layers` 展示 model（只读，多 ctx 共享）vs context（每会话一份）。
  - `<h2>` 典型调用序列：`backend_init -> load_model -> init_from_model -> get_vocab -> tokenize -> decode -> get_logits -> sample -> token_to_piece -> 循环 -> free`。**伪代码**（`<pre class="code">`，简化自真实调用，ASCII，注明对应 `llama.h` 符号）。**深挖前先放 trace（见下）**。
  - **内嵌 worked-example trace（Style A，复用 `.trace` 组件）**：标题"追踪一次 C API 调用 / Tracing one C-API call"。一个最小例子 `"Hi"` 走完整条链：站点 `① load 模型`（`model.gguf -> llama_model*`）-> `② 建 context`（`llama_context* n_ctx=512`）-> `③ tokenize "Hi"`（`-> [15043]` 示意）-> `④ decode -> logits`（`logits[n_vocab]`）-> `⑤ sample`（`-> token 1820`）-> `⑥ token_to_piece`（`1820 -> " there"`，蓝色结果）。tcap 注"（id 为示意）"。导语 `<p>` 引入，前后用 `<p>`/`<h2>` 隔开不与 card 紧邻。
  - `<h2>` 为什么是 C ABI：稳定 ABI + 跨语言绑定（Python/Go/Rust/Node 全绑这套）+ 不透明指针隐藏实现，是 llama.cpp 能被到处嵌入的根。**card analogy**：C 头像"插座标准"。
  - `<h2>` 折叠深挖（`<details>` >=2）：(1) `llama_batch` 怎么填（呼应 L18 批处理：`token`/`pos`/`seq_id`/`logits` 标志）；(2) API 的弃用与演进——`llama_new_context_with_model` -> `llama_init_from_model`，老符号留 `DEPRECATED` 兼容；告诉读者读源码看 `DEPRECATED(...)` 宏识别。
  - 硬性：zh CJK>=4000、en CJK==0、逐段对齐、>=3 图（含 trace）、>=2 深挖、>=2 片段。

- [ ] **Step 5: quiz（25）** `quizzes.py` 追加 2-4 题双语，例：「`llama_model` 与 `llama_context` 哪个可多会话共享？」「C++ 里怎么自动释放句柄？（`llama-cpp.h` 的 `_ptr`）」「典型调用序列里 tokenize 之后、sample 之前是哪一步？（decode 拿 logits）」。

- [ ] **Step 6: 重建+校验**：`python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py`，全绿；index 变"共 25 课 · **5** 个部分"；硬性达标；trace `<svg>` 无（Style A）；grep 渲染 `25-c-api.html` 确认无 `&amp;lt;` 双重转义、`.trace` 计数=2。

- [ ] **Step 7: commit**：`feat: add lesson 25 C API overview (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 2: 课 26「common 工具层 / The common layer」

**Files:** `src/part5.py`（追加 `LESSON_26`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `26-common.html`。

**源码事实（核实 2026-06-17）：**
- `common/`（不属公共 API，是各 tool 共享的"胶水层"，把裸 C API 包成好用的 C++）。`common/common.h`：`struct common_params`（一站式配置大结构，含嵌套 `struct common_params_sampling sampling`）；`common_init()`；`struct common_init_result`（一次拿到 model + context + sampler）；`common_batch_clear`/`common_batch_add`（填 `llama_batch` 的助手，呼应 L18）；`common_token_to_piece`（解码助手，呼应 L20）。
- 参数解析 `common/arg.{h,cpp}`：`struct common_arg`（一条命令行选项的定义，带 `.set_examples()`/`.set_env()` 等链式构建）；`common_params_parse(argc, argv, params, example)` 把 argv 解析进 `common_params`；`enum llama_example` 区分不同 tool 的参数集（cli/server/...）。
- 采样包装 `common/sampling.{h,cpp}`：`struct common_sampler`（把裸 `llama_sampler` 链 + 语法 GBNF 包在一起）；`common_sampler_init(model, params.sampling)` 按 `common_params_sampling.samplers`（`enum common_sampler_type` 列表）建链；`common_sampler_sample(gsmpl, ctx, idx, grammar_first)`、`common_sampler_accept(...)`。上层 cli/server 用的是这层，不直接碰 `llama_sampler_*`。
- 下载/缓存 `common/download.{h,cpp}`：`common_download_split_repo_tag("repo:tag")`、HF 仓库下载与本地缓存（`common_list_cached_models`、`common_download_progress`/回调）。让 `-hf user/repo` 直接拉模型。
- 还有 `common/log.{h,cpp}`（分级日志）、`common/console.{h,cpp}`（终端着色/交互）。

- [ ] **Step 1-3: 登记**

```python
# shell.py PAGES：
("26-common.html", "common 工具层", "The common layer", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES：
"26-common.html": ("common：参数解析、采样包装、下载缓存——各 tool 的共享胶水", "common: arg parsing, sampler wrapper, downloads - the shared glue for every tool"),
# registry.py CONTENT：
"26-common.html": part5.LESSON_26,
```

- [ ] **Step 4: 执笔 `LESSON_26`（双语）。结构：**
  - 导语：裸 C API（L25）好用但啰嗦——填 batch、建采样链、解析命令行、下载模型，每个 tool 都要做。`common/` 把这些重复活封成一层共享胶水；cli/server 都站在它肩上。注意：common **不是**公共 API（不保证 ABI 稳定），是 llama.cpp 自带工具的内部公共库。
  - `<h2>` 一站式配置 `common_params`：一个大结构装下所有可调项（模型路径、prompt、`n_predict`、`n_ctx`、嵌套 `sampling`…）。`common_init()` 吃它、吐 `common_init_result`（model+ctx+sampler 一次到位）。**图**：`vflow` 或 `cols` 展示 `common_params -> common_init -> {model, context, sampler}`。
  - `<h2>` 命令行怎么变配置：`common_arg` 声明每个选项 + `common_params_parse` 把 argv 填进 `common_params`；`enum llama_example` 让同一套机制为不同 tool 暴露不同参数子集。**伪代码**：一条 `common_arg` 的链式声明 + `common_params_parse` 调用。**trace 见下**。
  - **内嵌 trace（Style A）**：标题"追踪一次参数解析 / Tracing one arg parse"。站点：`① argv`（cells `-m model.gguf` `-p "Hi"` `-n 16` `--temp 0.7`）-> op `common_params_parse` -> `② common_params 字段`（`model=model.gguf` `prompt="Hi"` `n_predict=16` `sampling.temp=0.70`，蓝色）-> op `common_init` -> `③ 就绪`（cell `{model, ctx, sampler}`）。tcap 注"（示意）"。
  - `<h2>` 采样包装 `common_sampler`：把 L21 的裸 `llama_sampler` 链 + L23 的 GBNF 语法包成一个对象；`common_sampler_init` 按 `params.sampling.samplers` 顺序建链，`common_sampler_sample` 一把梭。**card detail** 标注它和 `llama_sampler_*` 的关系（呼应 L21）。
  - `<h2>` 下载与缓存：`-hf user/repo:tag` 怎么变成本地文件（`common_download_split_repo_tag` + HF 缓存）。**card spark 实战**：第一次跑某模型为什么会下载、缓存在哪。
  - `<h2>` 折叠深挖（>=2）：(1) 为什么 common 不算"公共 API"（ABI 不保证、是工具内部库，绑定者应直接用 `llama.h`）；(2) log/console 两个小工具（分级日志、终端着色），调试时怎么用。
  - 硬性同 Task 1。

- [ ] **Step 5: quiz（26）**：「common 属于公共 API 吗？（否，是工具共享内部库）」「谁把 argv 变成 `common_params`？」「cli/server 调采样用哪层？（`common_sampler`，非裸 `llama_sampler`）」。

- [ ] **Step 6: 重建+校验**（同上；index "共 26 课 · 5 个部分"；硬性达标；grep 渲染确认 `--temp`、`"Hi"` 等无裸 `<`、无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 26 common layer (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 3: 课 27「llama-cli / llama-cli」

**Files:** `src/part5.py`（追加 `LESSON_27`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `27-llama-cli.html`。

**源码事实（核实 2026-06-17）：**
- `tools/cli/`：`main.cpp`（极薄，`int main(argc,argv)` 转入 cli 逻辑）+ `cli.cpp`（主体）。入口 `common_init()` -> `common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)`。
- **重要现状**：现代 `llama-cli` **复用 server 的引擎**——`cli.cpp` `#include "server-common.h"`/`"server-context.h"`/`"server-task.h"`，`CMakeLists.txt` `target_link_libraries(... server-context ...)`。即 cli 是套在共享 `server_context`（slot/task，L28 详讲）上的**命令行 + 交互外壳**，不再是一份独立的裸 `llama_decode` 主循环。
- 交互/生成：`cli.cpp` 里 `// interactive loop` 后 `while (true)`；按 `n_predict` 上限、`antiprompt`（反向提示/reverse prompt）、EOG（生成结束符，呼应 L20/L21）停。流式：每选定一个 token 就 `common_token_to_piece` 还原并即时打印。
- 一句话定位：llama-cli = 给共享推理引擎套一层"读 stdin / 流式打印 stdout / 交互对话"的壳，是上手 llama.cpp 最直接的工具。

- [ ] **Step 1-3: 登记**

```python
# shell.py PAGES：
("27-llama-cli.html", "llama-cli", "llama-cli", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES：
"27-llama-cli.html": ("命令行/交互外壳：跑在共享引擎上的生成主循环", "the CLI/interactive shell: a generation loop on the shared engine"),
# registry.py CONTENT：
"27-llama-cli.html": part5.LESSON_27,
```

- [ ] **Step 4: 执笔 `LESSON_27`（双语）。结构：**
  - 导语：llama-cli 是大多数人第一个跑起来的 llama.cpp 程序——`llama-cli -m model.gguf -p "..."` 就能生成。这一课看它内部怎么把"命令行 + 一个生成循环"拼起来；也借它看清 llama.cpp 的工具是怎么搭在 common（L26）和引擎之上的。
  - `<h2>` 一条命令的旅程：`main.cpp -> cli.cpp`；`common_init` + `common_params_parse(..., LLAMA_EXAMPLE_CLI)` 把命令行变 `common_params`（L26）。**图**：`vflow` 步骤图：解析参数 -> 载入模型/建上下文 -> 编码 prompt -> 生成循环 -> 流式输出。
  - `<h2>` 生成主循环：tokenize prompt -> 喂引擎 decode -> 采样下一 token（`common_sampler`，L26/L21）-> `token_to_piece` 流式打印 -> 追加回上下文 -> 重复，直到 `n_predict` / EOG / 反向提示命中。**伪代码**：精简的 while 生成循环（ASCII，注明对应符号）。**trace 见下**。
  - **内嵌 trace（Style A）**：标题"追踪生成主循环一轮 / Tracing one generation step"。把 L19（KV 增长）/L21（采样）在 cli 视角下串一轮：站点 `① 已生成`（cells `The` `cat`）-> op `decode 末位` -> `② logits` -> op `common_sampler_sample` -> `③ 选定 token`（`sat`，hot）-> op `token_to_piece + 打印` -> `④ 流式输出`（cell `"...cat sat"`，蓝色）-> op `回环/检查停` -> `⑤ 停？`（`n_predict? EOG? antiprompt?`）。tcap 注"（示意）"。
  - `<h2>` 跑在共享引擎上：**现状重点**——现代 cli `#include "server-context.h"` 并链接 `server-context`，复用 server 的 `server_context`（slot/task）。**card macro**：cli 和 server 其实共用一台引擎，只是外壳不同（cli=命令行/交互，server=HTTP）；这也是为什么下一课 server 的很多概念在这里已埋下伏笔。
  - `<h2>` 交互模式：`-i`/对话模式、反向提示（antiprompt）打断、`console`（L26）着色。**card spark 实战**：常用参数 `-n`/`-c`/`--temp`/`-i` 各管什么。
  - `<h2>` 折叠深挖（>=2）：(1) cli 复用 server-context 的来龙去脉（历史上是独立 main 循环，后统一到引擎，减少重复）；(2) EOG/停止条件细节（呼应 L20 的 EOG 集合）。
  - 硬性同上。

- [ ] **Step 5: quiz（27）**：「llama-cli 内部复用了哪个组件的引擎？（server-context）」「生成循环靠什么停？（n_predict / EOG / 反向提示）」「命令行怎么变成配置？（common_params_parse, L26）」。

- [ ] **Step 6: 重建+校验**（同上；index "共 27 课 · 5 个部分"；硬性达标；grep 渲染无裸标签/双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 27 llama-cli (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 4: 课 28「llama-server / llama-server」（重点；trace 为 Style C SVG）

**Files:** `src/part5.py`（追加 `LESSON_28`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `28-llama-server.html`。

**源码事实（核实 2026-06-17）：**
- `tools/server/` 模块化：`main.cpp`（入口）、`server-context.{h,cpp}`（引擎：`struct server_context`，私有实现 `server_context_impl`）、`server-queue.h`（`server_queue`：`std::deque<server_task> queue_tasks`、`post()`/`recv()`、`callback_update_slots`）、`server-http.{h,cpp}`（HTTP 层）、`server-chat.{h,cpp}`（OpenAI/Anthropic 兼容转换，如 `server_chat_convert_anthropic_to_oai`）、`server-common.{h,cpp}`。
- **slot**：`struct server_slot`（server-context.cpp），每个 slot = 一条并行序列（`seq_id` = slot id），有自己的 `slot_state`：`SLOT_STATE_IDLE / STARTED / PROCESSING_PROMPT / DONE_PROMPT / GENERATING`（+ `WAIT_OTHER`）。slot 数 = `params.n_parallel`（`--parallel N`）；每 slot `n_ctx_slot = llama_n_ctx_seq(ctx)`（总上下文切给各 slot）。
- **连续批处理（continuous batching）**：`update_slots` 把**多个 slot 的当前 token 拼进同一个 `llama_batch`**——`common_batch_add(batch, token, pos, { slot.id }, ...)` 给每个 token 打上所属 `seq_id`；处于 `PROCESSING_PROMPT`（prefill）和 `GENERATING`（解码）的 slot **共享同一个 batch**，一次 `llama_decode(batch)` 同时推进所有活跃 slot 的序列。这是 server 高吞吐的核心：GPU 一次前向服务多请求，而不是一个一个排队。
- **请求流**：HTTP 请求 -> `server_task` -> `server_queue`（任务队列）-> 分配给空闲 `server_slot` -> `update_slots` 连续批处理循环 -> `server_task_result`（可流式）-> HTTP 响应。OpenAI 兼容端点（`/v1/chat/completions` 等）由 `server-chat` 把对话/工具调用在 OpenAI schema 与内部表示之间转换。
- 范围：本课只做**架构总览**；更深的调度/吞吐取舍（prefill/decode 交错、batch 容量、抢占）留 **L35（第七部分）**。

- [ ] **Step 1-3: 登记**

```python
# shell.py PAGES：
("28-llama-server.html", "llama-server", "llama-server", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES：
"28-llama-server.html": ("HTTP + OpenAI 兼容 + slot 连续批处理：把引擎变成服务", "HTTP + OpenAI-compatible + slot continuous batching: the engine as a service"),
# registry.py CONTENT：
"28-llama-server.html": part5.LESSON_28,
```

- [ ] **Step 4: 执笔 `LESSON_28`（双语）。结构：**
  - 导语：llama-server 把推理引擎变成一个 HTTP 服务——多用户、多请求同时连，还兼容 OpenAI 接口（现成 SDK 直接连）。它最精彩的地方是 **slot + 连续批处理**：一次前向同时服务多条请求。这一课看它的架构总览（深度调度留 L35）。
  - `<h2>` 整体架构：HTTP 请求 -> `server_task` -> `server_queue` -> 空闲 `server_slot` -> `update_slots`（连续批处理）-> `server_task_result` -> 响应。**图**：`vflow`/`flow` 横向架构图（HTTP 层 / 任务队列 / 引擎+slots / 结果）。注明对应 `server-http`/`server-queue`/`server-context`/`server-chat` 模块。
  - `<h2>` slot 是什么：`--parallel N` 开 N 个 slot，每个 slot 一条独立序列（自己的 `seq_id` + KV 区，呼应 L19），有状态机（IDLE -> PROCESSING_PROMPT -> GENERATING -> IDLE）。**图**：`timeline` 或 `cellgroup` 展示 slot 状态机。
  - `<h2>` 连续批处理（核心）：处于 prefill 和 decode 的 slot 把各自的 token 拼进**同一个 batch**，一次 `llama_decode` 全推进。**伪代码**：精简的 `update_slots`（遍历活跃 slot -> `common_batch_add(batch, tok, pos, {slot.id})` -> 一次 `llama_decode` -> 每 slot 取自己那行 logits 采样）。**trace 见下**。
  - **内嵌 trace（Style C，内联 SVG，复用 `.trace` 框 + `.trace svg` 响应式）**：标题"追踪一次连续批处理 / Tracing one continuous-batch step"。**概念**：3 个请求占 3 个 slot，某一步的 batch 同时含它们各自的 token；一次 `llama_decode` 后每 slot 各得下一 token。
    - **布局**（viewBox ~ `0 0 640 250`）：左侧 3 行 = slot0/slot1/slot2，标注各自 `seq` 与状态（slot0 GENERATING、slot1 PROCESSING_PROMPT/prefill、slot2 GENERATING）；中间一个"合并 batch"框，里面是混合的 token 格子（颜色按所属 slot 区分：accent/blue/purple），每格标 `seq=k`；一个 `llama_decode` 箭头汇入；右侧每 slot 各引出"下一 token"。底部注"一次前向，多请求共享 / one forward pass, shared by all requests"。
    - **Style-C 规范**（同 trace 系列）：`<svg viewBox=... width="100%" role="img" aria-label=...>`，zh aria-label 中文、en 纯 ASCII；字面色板（白 `#ffffff`/`#1d2129`、accent `#c2630e`/`#fff`、blue `#2563eb`、purple `#7c3aed`、muted `#9aa6b2`、label `#5b6470`），深色可读；`<text>` 内 `<`/`>`/`&` 转义；英文 SVG 文本纯 ASCII；合法 XML。实现时可先用 Python 算坐标生成、校验 well-formed + ASCII 再嵌入（参考既有 6 个 SVG）。
  - `<h2>` OpenAI 兼容：`/v1/chat/completions` 等端点；`server-chat` 在 OpenAI schema 与内部之间转换（含工具调用），所以现成客户端能直接连。**card spark 实战**：`llama-server -m ... --port 8080` 起服务，curl `/v1/chat/completions`。
  - `<h2>` 折叠深挖（>=2）：(1) slot 满了会怎样（排队 `queue_tasks_deferred`，呼应队列）；(2) 为什么连续批处理比"逐请求"吞吐高（GPU 一次前向摊到多请求；细节指向 L35）。
  - 硬性同上 + trace 为合法 SVG。

- [ ] **Step 5: quiz（28）**：「连续批处理的核心是什么？（多 slot 的 token 拼进同一 batch、一次 decode 全推进）」「slot 数由哪个参数定？（--parallel）」「server 怎么兼容 OpenAI？（server-chat 转换 /v1/... 端点）」。

- [ ] **Step 6: 重建+校验**（同上；index "共 28 课 · 5 个部分"；硬性达标；**`<svg` 计数 = 2、两个 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII**；grep 渲染无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 28 llama-server (bilingual) with SVG trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 5: 课 29「quantize 工具 / The quantize tool」

**Files:** `src/part5.py`（追加 `LESSON_29`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `29-quantize-tool.html`。

**源码事实（核实 2026-06-17）：**
- `tools/quantize/`：`main.cpp`（薄）+ `quantize.cpp`（主体，`#include "imatrix-loader.h"`）。用法 `llama-quantize in.gguf out.gguf Q4_K_M`；内部调公共 API `llama_model_quantize(in, out, &params)`。`quantize.cpp` 有一张档位表把名字 -> `llama_ftype` -> 体积/ppl 说明（如 `Q4_0`/`Q5_K`/`IQ2_*`/`MXFP4_MOE`…，含 "+0.xx ppl @ Llama-3-8B" 实测）。
- `include/llama.h` `struct llama_model_quantize_params`：`ftype`（目标类型）、`nthread`、`output_tensor_type`/`token_embedding_type`（个别张量可单独定精度）、`allow_requantize`、`pure`、`dry_run`（只算最终体积不真压）、`keep_split`、**`const struct llama_model_imatrix_data * imatrix`**（重要性矩阵指针）、`tt_overrides`（按张量覆盖类型）、`prune_layers`。
- `tools/imatrix/`：`imatrix.cpp`，用法 `llama-imatrix -m model.gguf -f calib.txt -o imatrix.gguf`。机制：用 eval-callback 钩子 `collect_imatrix(ggml_tensor* t, ...)` 在前向时累计每个权重张量**每列的激活幅度**（`values`/`counts`），存成 imatrix.gguf。量化时把它喂给 quantize -> 同样 bit 数下，把精度优先留给"被激活得多=重要"的权重列，质量更好。
- 串联：L6/L12 讲了量化的"原理与字节布局"；本课讲"用工具怎么压 + imatrix 怎么用更高质量地压"。

- [ ] **Step 1-3: 登记**

```python
# shell.py PAGES：
("29-quantize-tool.html", "quantize 工具", "The quantize tool", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES：
"29-quantize-tool.html": ("llama-quantize + imatrix：把模型压小，并用重要性矩阵保质量", "llama-quantize + imatrix: shrink the model, keep quality via an importance matrix"),
# registry.py CONTENT：
"29-quantize-tool.html": part5.LESSON_29,
```

- [ ] **Step 4: 执笔 `LESSON_29`（双语）。结构：**
  - 导语：前面（L6/L12）讲了量化"为什么能压、字节怎么排"。这一课是"怎么用工具压"：`llama-quantize` 一行命令把 fp16 模型变成 Q4_K_M 等小格式；再加上 imatrix（重要性矩阵），同样 bit 数还能把质量再拉回来。
  - `<h2>` 量化工具怎么用：`llama-quantize in.gguf out.gguf <ftype>` -> 公共 API `llama_model_quantize` + `llama_model_quantize_params`（`ftype` 选档、`dry_run` 试算体积、个别张量可单独定精度）。**图**：档位表（`cellgroup` 或 `table.t`）列几个代表 ftype 的 bpw 与体积/ppl 取舍（呼应 L06 的"挑档位"）。
  - `<h2>` imatrix 重要性矩阵：不是所有权重一样重要。`llama-imatrix -m ... -f calib.txt -o imatrix.gguf` 用校准文本跑模型、`collect_imatrix` 收集每列激活幅度 -> 哪些权重"用得多"。量化时 `--imatrix imatrix.gguf` 喂进去，精度优先留给重要列。**伪代码**：imatrix 生成（eval-callback 累计）+ 喂给 quantize（ASCII，注明 `collect_imatrix`/`llama_model_quantize_params.imatrix`）。**trace 见下**。
  - **内嵌 trace（Style A）**：标题"追踪一次 imatrix 加权量化 / Tracing one imatrix-weighted quantize"。站点：`① 一行权重`（cells `0.50` `0.02` `0.48` `-0.03`）-> op `imatrix 重要性` -> `② 重要性`（cells `高` `低` `高` `低` / `hi` `lo` `hi` `lo`，重要的 hot）-> op `按重要性量化` -> `③ 4-bit 码`（cells，重要列舍入更准）-> op `还原` -> `④ 误差`（`≈0` `0.02` `≈0` `0.02`，蓝色；tlab "误差被推给不重要的列 / error pushed onto unimportant weights"）。tcap 注"（示意）"。
  - `<h2>` 为什么有用：同样 4-bit，imatrix 让"重要权重少丢精度、不重要的多担误差"，整体困惑度（ppl）更低。**card macro**：这就是社区 imatrix 量化（如 IQ 系列）质量好的原因。
  - `<h2>` 折叠深挖（>=2）：(1) ftype 命名速读（`Q4_K_M` 的 K=K-quant、M=中等；`IQ2_*` 是带 imatrix 的超低 bit）；(2) `dry_run`/`keep_split`/按张量覆盖等实用 params。
  - 硬性同上。**注意**：trace 里若用 `≈` 是 unicode（zh 可用；en 用 `~=` 或 `<=`）。

- [ ] **Step 5: quiz（29）**：「imatrix 是干什么的？（记录权重列的重要性，量化时优先保重要的）」「怎么试算量化后体积而不真压？（dry_run）」「L06/L12 与本课的分工？（原理 vs 工具用法）」。

- [ ] **Step 6: 重建+校验**（同上；index "共 29 课 · 5 个部分"；硬性达标；grep 渲染无双重转义；**en SVG/trace 无（Style A），但确认英文段无 unicode `≈`**）。

- [ ] **Step 7: commit**：`feat: add lesson 29 quantize tool (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 6: 课 30「评测与基准 / Evaluation & benchmarks」

**Files:** `src/part5.py`（追加 `LESSON_30`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `30-eval-bench.html`。

**源码事实（核实 2026-06-17）：**
- `tools/perplexity/`：`main.cpp` + `perplexity.cpp`。`softmax(logits)`、`log_softmax(n_vocab, logits, tok)` 取**真实下一词的 log 概率**；`struct results_perplexity{ ppl_value }`。困惑度 PPL = `exp( -(1/N) * Σ log P(真实下一词) )` = `exp(平均负 log-prob)`，模型对真实文本越"不惊讶"越低。也支持 HellaSwag / Winogrande 等任务级（多选准确率，按 `n_chunk` 切块）。
- `tools/llama-bench/`：`llama-bench.cpp` + `main.cpp`。`struct cmd_params`：`n_prompt`（默认 512，**pp** = prompt/prefill 处理速度）、`n_gen`（默认 128，**tg** = token 生成速度）。测吞吐 t/s，报 `avg()` ± stddev，可跨配置（量化档/线程/后端）对比。
- 闭环：L29 档位表里的 "+0.xx ppl" 正是 perplexity 测出来的"质量代价"；量化省下的速度由 llama-bench 量。所以本课给前面所有"压/调"提供**度量标尺**，也为第五部分收尾。
- PPL 速算自检：`-logP=[1.2,0.5]` -> 平均 0.85 -> `exp(0.85)≈2.34`。

- [ ] **Step 1-3: 登记**

```python
# shell.py PAGES：
("30-eval-bench.html", "评测与基准", "Evaluation & benchmarks", "第五部分 · 公共 API 与工具", "Part 5 · Public API & tools"),
# shell.py SUBTITLES：
"30-eval-bench.html": ("perplexity 量质量、llama-bench 量速度：选模型/量化档的两把尺子", "perplexity for quality, llama-bench for speed: the two rulers for choosing models/quants"),
# registry.py CONTENT：
"30-eval-bench.html": part5.LESSON_30,
```

- [ ] **Step 4: 执笔 `LESSON_30`（双语）。结构：**
  - 导语：把模型跑起来、压小之后，怎么知道一个量化档/配置"好不好"？要两把尺子——**质量**（perplexity）和**速度**（llama-bench）。这一课讲它们各量什么、怎么读，并给第五部分收个尾。
  - `<h2>` perplexity 量质量：PPL = `exp(平均负 log-prob)`——拿真实文本，看模型给"真实下一词"打多高概率；越自信越对，PPL 越低。**伪代码**：遍历序列 -> `log_softmax(logits, 真实token)` 取 log P -> 累加 -> `exp(-平均)`（ASCII，注 `perplexity.cpp` 的 `log_softmax`）。**trace 见下**。
  - **内嵌 trace（Style A）**：标题"追踪一次困惑度计算 / Tracing one perplexity calc"。站点：`① 序列`（cells `The` `cat` `sat`）-> op `每步真实下一词 logP` -> `② -logP`（cells `1.2` `0.5`，蓝色；tlab "模型对真实下一词的惊讶度"）-> op `平均` -> `③ 0.85` -> op `exp` -> `④ PPL ≈ 2.34`（hot；tlab "越低越好"）。tcap 注"（数字为示意）"。
  - `<h2>` llama-bench 量速度：两类基准 **pp**（`n_prompt`，prefill 吞吐）与 **tg**（`n_gen`，生成吞吐），报 t/s 的 avg±stddev，跨量化档/线程/后端对比。**图**：`cols` 或 `cellgroup` 对照 pp（一次喂很多 token）vs tg（一个一个生成）两种工作负载。**card spark 实战**：`llama-bench -m model.gguf -p 512 -n 128`。
  - `<h2>` 两把尺子一起看（闭环 + 第五部分收尾）：量化（L29）省了速度、加了一点 ppl；只有同时看 ppl（质量）和 bench（速度）才能选对档。**card macro**：回顾第五部分——L25 C API（怎么调）-> L26 common（共享胶水）-> L27 cli / L28 server（怎么跑）-> L29 quantize（怎么压）-> L30 评测（怎么衡量好坏）。从"内部原理"到"对外怎么用、怎么压、怎么评"，一条线走完。
  - `<h2>` 折叠深挖（>=2）：(1) PPL 的直觉——"有效分支因子"（PPL=N 约等于模型在 N 个等概率选项间犹豫）；(2) 任务级评测（HellaSwag/Winogrande 多选准确率）与 ppl 的区别，何时用哪个。
  - 硬性同上。

- [ ] **Step 5: quiz（30）**：「perplexity 怎么算、越高越好还是越低？（exp(平均负 logP)，越低越好）」「llama-bench 的 pp 和 tg 分别量什么？（prefill / 生成 吞吐）」「为什么选量化档要同时看 ppl 和 bench？（质量 vs 速度的取舍）」。

- [ ] **Step 6: 重建+校验**（同上；index "共 30 课 · 5 个部分"；硬性达标；grep 渲染无双重转义；英文段无 unicode `≈`，en 用 `~=`）。

- [ ] **Step 7: commit**：`feat: add lesson 30 evaluation & benchmarks (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 7: 收尾（roadmap 勾选 + 全量验证 + 完成分支）

**Files:** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（勾 M5）。

- [ ] **Step 1: 更新 roadmap**：里程碑总表 M5 行"状态"`待写` -> `完成`；"状态追踪" `- [ ] M5 ...` -> `- [x] M5 ...`。commit `docs: mark M5 (part 5) done`。
- [ ] **Step 2: 全量验证**（master 合并前在分支上跑）：
  - `python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py` = `Wrote 31 files`、`0 error(s), 0 warning(s)`、`all internal links resolve`。
  - index 显示"共 **30** 课 · **5** 个部分"；第五部分 6 课都在导航里。
  - 全 30 课：按 `<h2>` 分节 `<p>` 计数中英相等（parity）；zh CJK >= 4000、en CJK == 0；无连续 `<p>` > 3；无 `&amp;lt;`/`&amp;gt;`/`&amp;amp;` 双重转义、无裸 `<|`/`<0x`。
  - 第五部分新增 trace：`class="trace"` 全站净增 12（6 课 x 2 语言）；`<svg` 净增 2（仅 L28）；新增 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII、深色色板。
- [ ] **Step 3: 第五部分整体复审**（可选但建议）：派一个 superpowers:code-reviewer 子代理（当前模型）复审 `master..HEAD` 的 6 课一致性（标题/卡片/图/trace 风格统一、源码引用准确、双语纪律、范围未越界——不碰 1-4 部分/build 基础设施/M9）。
- [ ] **Step 4: 完成分支**：用 superpowers:finishing-a-development-branch，先过验证门，再按用户选择（历史偏好：本地 `--no-ff` 合并 master + 删分支）。

---

## 计划自审（writing-plans self-review）

- **Spec 覆盖**：设计 spec 的 6 课 (25-30) 各有一个 Task（1-6）；"trace 初稿内嵌""L28 总览/深挖留 L35""每课交付物""不变量""执行方式"均在头部统一节 + 各 Task 落实。✓
- **占位符扫描**：无 TBD/TODO；每课给了真实文件+符号、具体结构、trace 设想与示例值、登记代码、quiz 方向、验证命令。lesson 正文 4000+ CJK 由实现子代理执笔（属内容创作，不可能在 plan 里预写全文；plan 给到结构+源码事实+trace 规格，符合 M1-M4 既有 plan 的粒度）。✓
- **类型/命名一致**：6 课 slug、`LESSON_25..30`、`part5.py`、PAGES/SUBTITLES/CONTENT 三处登记字段在各 Task 一致；index 计数每课 +1（25->30）、部分数在 Task 1 由 4->5（新增第五部分标签）。✓
- **源码事实一致**：L25 `llama_init_from_model`（非弃用的 `llama_new_context_with_model`）；L27 cli 复用 `server-context`，与 L28 `server_context`/`server_slot`/连续批处理一致；L29 `llama_model_quantize_params.imatrix` 与 L30 perplexity 的 "+ppl" 闭环；均按 2026-06-17 真实源核实。✓
- **范围**：聚焦 M5 六课；明确不做 PDF/CI/README(M9)、server 深度调度(L35)、其它 tools（mtmd/rpc/tts/...）。✓
