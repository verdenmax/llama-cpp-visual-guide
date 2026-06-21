# 第八部分（实战与贡献，课 38-39）+ 第九部分（速查，课 40）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐 task 执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 给图解教程补齐第八部分「实战与贡献」2 课（从 HF 转换模型、编译/调试/测试/贡献）与第九部分「速查」1 课（术语表·概念索引），共 3 课收官，双语、硬核落地。

**Architecture:** 沿用 M1-M7 的零依赖 Python 静态站点生成器。新建 `src/part8.py`（`LESSON_38/39`）与 `src/part9.py`（`LESSON_40`），改 `src/registry.py`/`src/shell.py`/`src/quizzes.py` 登记，重建全部 HTML。新增两个部分标签（第八、第九部分）。

**Tech Stack:** Python 3（生成器 + 校验脚本 `check_html.py`/`check_links.py`）；手写 HTML 片段；内联 SVG（L40 概念依赖图，Style C）；rsvg-convert + chromium（SVG/页面目检）。

---

## 统一交付标准（每课硬性，照搬 M7）

- **结构**：导语 `<p>` + 卡片（macro/analogy/key/spark 酌用，≥2 张 `<details>` 深挖；L40 索引课可少用 `<details>`、以表为主）+ **≥3 图示**（`cols`/`layers`/`table.t`/`trace`，单语 ≥3，含 ≥1 个 trace）+ **≥2 段真实代码** `<pre class="code">`（L40 以术语表/链接为主，代码段可少）+ 各课规定的内嵌图。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace`、内联 `<svg>`、`<table>` 不计入）。
- **中文密度**：zh CJK ≥ 4000；**en CJK == 0**（纯 ASCII：用 `-`/`->`/`...`/`~`/`+/-`/`x`，禁 em-dash/unicode 箭头/`≈`/`±`/`×`/`…`/`·`/`Δ`）。
- **无文字墙**：连续顶层 `<p>` ≤ 3。
- **转义**：代码里 `<`/`>`/`&` 写成 `&lt;`/`&gt;`/`&amp;`；无双重转义 `&amp;lt;`。**quiz 字符串同样须预转义**（`quizzes.py` 的 q/opt/why 原样插值，含 `<` 的词如 `<image>` 须写 `&lt;image&gt;`）。
- **trace**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn/.cellrow/.vc[.hot/.blue]/.op/.tlab`）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>`（zh aria-label 中文、en 纯 ASCII），合法 XML，**深浅 `.trace` 背景都可读**（深色 `#161b22`：深色文字只放白底框内，自由文字用中间色 `#5b6470`/accent/blue/purple），trace 不与 `.card` 紧邻。
- **源码引用**：以"文件 + 符号名"为主、不写行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验 2026-06-19）。
- **quiz**：`quizzes.py` 写该课 3 MCQ + 1 开放题双语自测。
- **登记**：`registry.CONTENT`（`import part8`/`import part9` + filename -> `LESSON_NN`）；`shell.PAGES`（filename、zh/en 短标题、部分标签）；`shell.SUBTITLES`；index 自动变"共 40 课 · 9 个部分"。

## 执行方式（M5/M6/M7 经验）

- superpowers:subagent-driven-development，**一课一个 task**（Task 1=课38、Task 2=课39、Task 3=课40；Task 4 收尾）。
- 每个 task：实现子代理 -> **spec 合规审查子代理 -> 质量审查子代理**（两段审查），修复回环后再标完成。子代理一律用当前主会话模型，显式传 `model`。
- **关键经验**：后台 general-purpose 子代理写整课常中途失败（"completed"却零文件写入）。控制器**每次都要独立核验 git 状态**（不信报告）；若实现子代理失败，则由控制器亲自照模板执笔，仍跑完整 spec+质量双重审查。
- **关键 Style-C SVG（L40 概念依赖图）**：先用 Python 预生成并校验（well-formed XML + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 渲染目检），再喂给实现（沿用 M6/M7 做法）。
- **HTML 是被 git 跟踪的**：每课提交须 `git add` 源文件 **+ 重建后的全部 HTML**（`index.html` + `lessons/*.html`），提交后 `git status` 必须干净。
- commit 用 `Assisted-by: GitHub Copilot`（非 Co-authored-by）。分支：在 master 上从本 plan 提交后，新建 `feature/part8-9-contrib-glossary` 分支做实现。

---

## Task 1: 课 38「从 HF 转换模型 / Converting HF models to GGUF」

**Files:** **新建** `src/part8.py`（写 `LESSON_38`，文件头 `"""Content for Part 8 (practice & contributing)."""`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `38-convert-hf.html`。trace 为 **Style A（无 SVG）**；另含 GGUF 字节布局 `layers` 图。

**源码事实（核实 2026-06-19，文件+符号、无行号）：**
- **薄 CLI** `convert_hf_to_gguf.py`（仓库根，约 300 行）：`parse_args()` 的 `--outtype f32/f16/bf16/q8_0/tq1_0/tq2_0/auto`（转换时即可量化）；`main()` 里 `ModelBase.load_hparams(dir)` 读 `config.json` 的 `architectures` -> `get_model_architecture()` -> `get_model_class(name, mmproj)` 分发到注册子类 -> 实例化 -> `model_instance.write()`。**关键结构事实**：转换机制已从单文件重构进 `conversion/` 包；本课讲包布局。
- `conversion/base.py`：`class ModelBase`——`_model_classes`（按 `ModelType` TEXT/MMPROJ 存的注册表）、`@classmethod register(*names)`（装饰器工厂，把每个 HF 架构名写进 `_model_classes`）、`get_tensors()`（遍历 safetensors/pt 分片，yield `(name, tensor)`）、`map_tensor_name()`（经 `gguf.TensorNameMap` 把 HF 名翻成 GGUF 名）、`set_gguf_parameters()`（基类 `NotImplementedError`，子类必须重写）、`modify_tensors()`（子类重塑/拆分/permute）、`prepare_tensors()`（主循环：遍历 `get_tensors` -> `modify_tensors` -> 定 `data_qtype` -> `self.gguf_writer.add_tensor(new_name, data, raw_dtype=...)`）、`write()`（编排：`prepare_tensors` -> `prepare_metadata` -> `write_header_to_file` -> `write_kv_data_to_file` -> `write_tensors_to_file` -> `close`）。`class TextModel(ModelBase)`：`self.tensor_map = gguf.get_tensor_name_map(arch, block_count)`、`set_vocab()`、`set_gguf_parameters()`（写 block_count/n_embd/n_head/rope 等 KV）。
- `conversion/llama.py`：`@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM", "MixtralForCausalLM", ...)` -> `class LlamaModel(TextModel)`（`model_arch = gguf.MODEL_ARCH.LLAMA`）；静态 `permute()`（Q/K 权重重排，HF 与 llama.cpp 的 RoPE 排布不同）；`modify_tensors()`（按需 permute）。
- `gguf-py/gguf/constants.py`：`GGUF_MAGIC=0x46554747`（"GGUF"）、`GGUF_VERSION=3`、`GGUF_DEFAULT_ALIGNMENT=32`；`class Keys`（`general.architecture`/`general.alignment`/`general.file_type` 等元数据键命名空间）；`GGMLQuantizationType`（Q4_0=2 ... Q4_K=12 ... 张量 dtype 码）；`GGUFValueType`（KV 值类型标记 UINT8=0 ... ARRAY=9 ... FLOAT64=12）。
- `gguf-py/gguf/gguf_writer.py`：`class GGUFWriter`（`WriterState` 状态机 NO_FILE->EMPTY->HEADER->KV_DATA->TI_DATA->WEIGHTS）；`write_header_to_file()`（写四字段：magic u32、version u32、tensor_count u64、kv_count u64）；`write_kv_data_to_file()`（逐 KV：key 字符串 + 带 `GGUFValueType` 标记的值）；`write_ti_data_to_file()`（逐张量信息：name + n_dims + 各维 + dtype 码 + offset，offset 按 `ggml_pad(nbytes, alignment)` 推进）；`write_tensors_to_file()`（写完张量信息后对齐到 32B，再逐张量写原始字节）。
- `gguf-py/gguf/tensor_mapping.py`：`class TensorNameMap`（`__init__(arch, n_blocks)` 构建 `HF 别名 -> (MODEL_TENSOR, 规范名)` 映射，`{bid}` 按层展开；`get_name()`）。
- 指向 `docs/development/HOWTO-add-model.md`：新增一个架构 = 在 `conversion/` 加一个 `@ModelBase.register` 子类、实现 `set_gguf_parameters`/`modify_tensors`。

- [ ] **Step 1-3: 登记**（三处 + import）

```python
# registry.py 顶部新增: import part8
# registry.py CONTENT 追加:
"38-convert-hf.html": part8.LESSON_38,
# shell.py PAGES 追加（第八部分起点）:
("38-convert-hf.html", "从 HF 转换模型", "Converting HF models", "第八部分 · 实战与贡献", "Part 8 · Practice & contributing"),
# shell.py SUBTITLES 追加:
"38-convert-hf.html": ("convert_hf_to_gguf.py 薄 CLI + conversion 包 + gguf-py 写盘字节布局", "convert_hf_to_gguf.py thin CLI + conversion package + gguf-py byte layout"),
```

- [ ] **Step 4: 执笔 `LESSON_38`（双语，新建 `src/part8.py`）。结构：**
  - 导语 `<p>`：你下载的 HF 模型是一堆 `.safetensors` + `config.json`，llama.cpp 只吃 `.gguf`。这一课把"转换"拆开：谁分发到对应架构、张量名怎么翻译、元数据与权重最终怎么落成一个文件。
  - `<h2>` 一条命令背后：`convert_hf_to_gguf.py` 现在只是薄 CLI，真正干活的是 `conversion/` 包。**真实代码**：`main` 里 `load_hparams` -> 读 `architectures` -> `get_model_class` 分发。
    - **trace（Style A·站点流）**：HF 目录 -> `load_hparams` 读架构 -> `get_model_class` 选 `LlamaModel` -> `set_gguf_parameters` 写超参 -> `prepare_tensors` 逐张量 `modify_tensors`+改名 -> `GGUFWriter` 写盘 -> `model.gguf`。
  - `<h2>` 注册表与架构分发：`@ModelBase.register("LlamaForCausalLM", ...)` 怎么把"HF 架构名 -> Python 类"登记进 `_model_classes`；新增模型就是加一个注册子类（指向 HOWTO-add-model）。**真实代码**：装饰器工厂 + `LlamaModel(TextModel)` 子类骨架。
  - `<h2>` 张量改名与超参：`set_gguf_parameters`（n_layer/n_embd/rope 写成 GGUF KV）+ `modify_tensors`/`map_tensor_name`（经 `TensorNameMap` 把 `model.layers.0.self_attn.q_proj.weight` 翻成 `blk.0.attn_q.weight`，Llama 还要 `permute` Q/K）。**真实代码 + 名称对照**（`.cols`：HF 名 vs GGUF 名，至少 3-4 行映射）。
  - `<h2>` GGUF 文件长什么样：`GGUFWriter` 写盘四段——header（magic/version/tensor_count/kv_count）-> KV 段（带类型标记）-> 张量信息段（名/维度/dtype/offset）-> 对齐 32B -> 原始张量数据。**图（layers）**：GGUF 字节布局自上而下分段堆叠（header / KV / tensor-info / padding / tensor-data）。`--outtype` 在这步决定每张量存 f16/q8_0/...（呼应 L06/L12）。**真实代码**：`write_header_to_file` 的 4 字段 struct.pack 骨架。
  - `<h2>` 折叠深挖（≥2）：(1) 为什么要"对齐"到 32B（mmap 时按边界对齐才能零拷贝映射，呼应 L14 mmap）；(2) `set_vocab` 与 tokenizer（SentencePiece/BPE 词表 + special token 怎么一并写进 GGUF KV，呼应 L20）。
  - 硬性：zh CJK≥4000、en CJK==0、逐节对齐、≥3 图（含 1 trace + 1 layers）、≥2 深挖、≥2 真实代码片段（Python）。

- [ ] **Step 5: quiz（38）** 3 MCQ + 1 开放：「`convert_hf_to_gguf.py` 现在的角色？（薄 CLI，机制在 conversion 包）」「`@ModelBase.register` 干什么？（把 HF 架构名登记到子类，实现分发）」「GGUF 文件 header 头四个字段？（magic/version/tensor_count/kv_count）」「开放：新增一个未支持的架构，大致要在 conversion 包里做哪几步？」

- [ ] **Step 6: 重建+校验**：`cd src && python3 build.py && python3 check_html.py && python3 check_links.py` 全绿；index 变"共 38 课 · 8 个部分"；硬性达标；trace=2（zh/en 各 1）、`<svg`=0（Style A）；grep 渲染无 `&amp;lt;` 双重转义；Python 代码片段里 `<`/`&` 已转义。

- [ ] **Step 7: commit**：`feat: add lesson 38 converting HF models to GGUF (bilingual) with pipeline trace + quiz` + `Assisted-by: GitHub Copilot`（暂存 4 源文件 + 重建的全部 HTML，提交后 git status 干净）。

## Task 2: 课 39「编译·调试·测试·贡献 / Build, debug, test, contribute」

**Files:** `src/part8.py`（**追加** `LESSON_39`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `39-build-contribute.html`。trace 为 **Style A（无 SVG）**；另含后端->开关 `table.t` 与 PR 检查清单 `layers`/`cols`。

**源码事实（核实 2026-06-19，文件+符号、无行号）：**
- `docs/build.md`：CPU 基本路径 `cmake -B build` 然后 `cmake --build build --config Release`；后端开关 `-DGGML_CUDA=ON`（CUDA）、`-DGGML_METAL=OFF`（关 Metal）、`-DGGML_VULKAN=ON`（Vulkan）、`-DGGML_HIP`/`-DGGML_SYCL`/`-DGGML_MUSA` 等；CUDA 架构 `-DCMAKE_CUDA_ARCHITECTURES`；产物落 `build/bin/`（`llama-cli`/`llama-server` 等）。Debug 构建用 `--config Debug` / `-DCMAKE_BUILD_TYPE=Debug`。
- `tests/CMakeLists.txt`：`function(llama_build source)`、`function(llama_test target ...)`（包装 CMake `add_test`）、`function(llama_build_and_test source ...)`（编译 + 注册测试）。代表用例：`test-tokenizer-0`（按词表参数化）、`test-backend-ops.cpp`（**所有 ggml 算子的跨后端正确性/性能基准**）、`test-quantize-fns.cpp`/`test-quantize-perf.cpp`、`test-sampling.cpp`、`test-rope.cpp`。统一用 `ctest`（在 `build/` 里）跑。
- `CONTRIBUTING.md`：开篇 **AI 使用政策**——"**不接受全部或主要由 AI 生成的 PR**"，AI 先写再人改仍算 AI 生成；贡献者要能**独立解释与维护**自己的代码。PR 规范：**一个 PR 一个功能**、改 ggml 要跑/扩 `test-backend-ops`、**CPU 支持优先**；维护者 **squash-merge**、提交标题格式 `<module> : <title> (#NNNN)`。编码规范：跟现有风格、`clang-format`（clang-tools v15+）、命名规范。
- `.github/workflows/`（约 50 个）：分后端构建矩阵 `build-cpu.yml`/`build-cuda-ubuntu.yml`/`build-cuda-windows.yml`/`build-vulkan.yml`/`build-sycl.yml`/`build-apple.yml`/`build-android.yml` 等；质量门 `code-style.yml`（clang-format）、`editorconfig.yml`、`python-lint.yml`、`python-type-check.yml`、`build-sanitize.yml`（ASan/UBSan）；`server.yml`/`docker.yml`/`release.yml`。本课只**概览**不逐个展开。

- [ ] **Step 1-3: 登记**（三处，import 已在 Task 1 加好）

```python
# registry.py CONTENT 追加:
"39-build-contribute.html": part8.LESSON_39,
# shell.py PAGES 追加:
("39-build-contribute.html", "编译·调试·贡献", "Build & contribute", "第八部分 · 实战与贡献", "Part 8 · Practice & contributing"),
# shell.py SUBTITLES 追加:
"39-build-contribute.html": ("CMake 多后端构建 · ctest/test-backend-ops · CONTRIBUTING 的 AI 政策 · clang-format", "CMake multi-backend build, ctest/test-backend-ops, CONTRIBUTING AI policy, clang-format"),
```

- [ ] **Step 4: 执笔 `LESSON_39`（双语，追加进 `src/part8.py`）。结构：**
  - 导语 `<p>`：读懂了源码，怎么动手？这一课走一遍真实开发回路——编译出二进制、跑测试确认没坏、按规范改、提一个能被接受的 PR；特别强调本仓库**对 AI 生成 PR 的明确政策**。
  - `<h2>` 编译：CMake 一套命令编译；不同后端不同开关；Debug/Release 区别；产物 `build/bin/`。**图（table.t）**：后端（CPU/CUDA/Metal/Vulkan/...）-> CMake 开关 -> 适用平台。**真实命令**（`<pre class="code">`：`cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release`）。
    - **trace（Style A·站点流）**："一次贡献的生命周期"：clone -> `cmake -B build` 配置 -> `cmake --build` 编译 -> 改代码 -> `ctest`/`test-backend-ops` 验证 -> `clang-format` -> 提 PR（一功能 / CPU 优先）-> CI 矩阵跑全平台 -> squash-merge。
  - `<h2>` 测试：`ctest` 怎么跑；`llama_test`/`llama_build_and_test` 宏怎么把一个 `test-*.cpp` 注册成用例；几类代表测试（tokenizer / backend-ops / quantize / sampling）各测什么。**真实代码**：`tests/CMakeLists.txt` 的宏调用 + 一条 `ctest -R test-backend-ops` 命令。
  - `<h2>` 调试：Debug 构建 + sanitizer（`build-sanitize`）；`test-backend-ops` 在加新算子/改 ggml 时拿 CPU 当 ground truth 的角色（CONTRIBUTING 点名）；日志/`--verbose`。**点到 + 指向**。
  - `<h2>` 贡献规范（重点）：`CONTRIBUTING` 的 **AI 政策**（不接受全/主要 AI 生成的 PR、要能独立解释与维护）、一个 PR 一个功能、CPU 优先、提交标题格式、`clang-format`。**图（cols/layers）**：一个合格 PR 的检查清单（功能单一 / 跑过 test-backend-ops / clang-format / 能自述）。
  - `<h2>` 折叠深挖（≥2）：(1) 为什么"CPU 支持优先"（CPU 是参考实现 + `test-backend-ops` 的 ground truth，新后端要对齐它，呼应 L31/L33）；(2) CI 矩阵为什么这么多（每个后端一套独立构建，呼应 L33 后端调度）。
  - 硬性：zh CJK≥4000、en CJK==0、逐节对齐、≥3 图（含 1 trace + 1 table.t）、≥2 深挖、≥2 真实代码片段（shell/CMake）。
  - **注意**：本课会引用本仓库自身的 AI 政策；保持中立陈述事实，不夹带评论。

- [ ] **Step 5: quiz（39）** 3 MCQ + 1 开放：「编译 CUDA 版的 CMake 开关？（`-DGGML_CUDA=ON`）」「改了 ggml 算子，CONTRIBUTING 要求跑哪个测试？（`test-backend-ops`）」「本仓库对 AI 生成的 PR 的政策？（不接受全部/主要由 AI 生成的 PR）」「开放：为什么新后端要求 CPU 支持优先 + 对齐 test-backend-ops？」

- [ ] **Step 6: 重建+校验**：`cd src && python3 build.py && python3 check_html.py && python3 check_links.py` 全绿；index 变"共 39 课 · 8 个部分"；硬性达标；trace=2、`<svg`=0；grep 渲染无 `&amp;lt;`；shell/CMake 代码片段里 `<`/`&`（若有）已转义。

- [ ] **Step 7: commit**：`feat: add lesson 39 build debug test contribute (bilingual) with workflow trace + quiz` + `Assisted-by: GitHub Copilot`（暂存 4 源文件 + 重建的全部 HTML，提交后 git status 干净）。

## Task 3: 课 40「术语表·概念索引 / Glossary & concept index」

**Files:** **新建** `src/part9.py`（写 `LESSON_40`，文件头 `"""Content for Part 9 (quick reference)."""`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `40-glossary.html`。**含 Style C·SVG 概念依赖图**（控制器预生成）+ 全书学习路径 trace（Style A）+ 分类术语表（多张 `table.t`）。这是收官课、索引型课。

**源码事实（每个术语标"单一最佳定义点"，核实 2026-06-19）：**
- 核心数据结构：GGUF（`ggml/include/gguf.h` `GGUF_MAGIC`/`gguf_context`；py `gguf-py/gguf/constants.py`）、ggml_tensor（`ggml/include/ggml.h` `struct ggml_tensor`）、cgraph 计算图（`ggml/src/ggml-impl.h` `struct ggml_cgraph`）、量化类型（`ggml/include/ggml.h` `enum ggml_type`，`GGML_TYPE_Q4_K`）。
- 推理流程：context（`src/llama-context.h` `struct llama_context`）、batch（`include/llama.h` `llama_batch`）、KV cache（`src/llama-kv-cache.h` `class llama_kv_cache`）、vocab/tokenizer（`src/llama-vocab.h` `struct llama_vocab`）、sampler（`src/llama-sampler.cpp` `llama_sampler_init`，**注意不是** `llama-sampling.cpp`）、RoPE（`ggml/include/ggml.h` `ggml_rope`）。
- 内核与后端：backend（`ggml/include/ggml-backend.h` `ggml_backend_t`）、CPU/CUDA 后端（呼应 L31/L32）、后端调度（呼应 L33）。
- 进阶机制与工具：MoE（`src/llama-graph.cpp` `build_moe_ffn`）、speculative（`common/speculative.h` `common_speculative`）、mtmd 多模态（`tools/mtmd/mtmd.h` `mtmd_context`）、SSM/Mamba（`ggml/include/ggml.h` `ggml_ssm_scan`）、GGUF 转换（`convert_hf_to_gguf.py` + `conversion/`）。
- 每个术语关联其主讲课号：tensor->L05、cgraph->L09/L10、backend->L31/L33、量化->L06/L12、context->L17、batch->L18、KV cache->L19、vocab->L20、sampler->L21、RoPE->L15、MoE->L35、speculative->L34、mtmd->L36、SSM->L37、GGUF 转换->L38。（链接前先 grep 现有 `lessons/*.html` 文件名核实 slug，避免死链。）

- [ ] **Step 0（控制器先做）: 预生成概念依赖图 SVG**。用 Python 生成 `l40_zh.svg` / `l40_en.svg`（viewBox 约 `0 0 760 360`）：节点画核心概念的依赖层次——底层 `ggml_tensor`（地基）-> `cgraph`（由张量组成）-> `backend`（执行图）；右侧 `llama_context` 持有 `KV cache`，`batch` 驱动 `decode`，末端 `vocab`/`sampler` 收尾；用箭头表示"谁建立在谁之上"。色板：白底框 `#ffffff` + ink `#1d2129`，accent `#c2630e`/blue `#2563eb`/purple `#7c3aed`，自由文字 `#5b6470`。**校验**：`xml.dom.minidom` 可解析、英文 SVG 纯 ASCII（箭头用 `->`，无 `≈`/`×`/`·`）、坐标不溢出 viewBox、`rsvg-convert` 渲染目检在深浅背景都可读。存入会话 files 目录后再嵌入课文。

- [ ] **Step 1-3: 登记**（三处 + import）

```python
# registry.py 顶部新增: import part9
# registry.py CONTENT 追加:
"40-glossary.html": part9.LESSON_40,
# shell.py PAGES 追加（第九部分起点）:
("40-glossary.html", "术语表·索引", "Glossary & index", "第九部分 · 速查", "Part 9 · Quick reference"),
# shell.py SUBTITLES 追加:
"40-glossary.html": ("全书 40 课术语一句话查 + 概念依赖图 + 点链接跳到对应课", "40-lesson glossary one-liners + concept dependency map + jump links"),
```

- [ ] **Step 4: 执笔 `LESSON_40`（双语，新建 `src/part9.py`）。结构：**
  - 导语 `<p>`：到这里全书 40 课讲完了。这一课不教新东西，是一张**速查地图**：先看 9 个部分怎么层层递进，再看核心概念之间谁依赖谁，最后用分类术语表"一句话查 + 点链接跳回对应课"。
  - `<h2>` 全书地图（9 部分学习路径）：**trace（Style A·站点流）**：第一部分 宏观全景 -> 第二部分 前置基础 -> 第三部分 ggml 引擎 -> 第四部分 llama 内部 -> 第五部分 API 与工具 -> 第六部分 底层内核 -> 第七部分 进阶专题 -> 第八部分 实战贡献 -> 第九部分 速查；每站一句话"这部分解决什么"。
  - `<h2>` 概念依赖图：**图（Style C·SVG，嵌 Step 0 预生成的 `l40_{zh,en}.svg`）**：核心概念依赖关系（tensor 是地基、cgraph 由张量组成、backend 执行图、context 持有 KV cache、batch 驱动 decode、sampler/vocab 收尾）。一段 `<p>` 讲怎么读这张图。
  - `<h2>` 分类术语表：**多张 `table.t`**，按类分（① 核心数据结构 ② 推理流程 ③ 内核与后端 ④ 进阶机制与工具）；每行：术语 | 一句话定义 | 源码位置（`<span class="mono">file</span>`）| 跳转（`<a href="NN-slug.html">第 NN 课</a>` 站内链接）。每类 4-6 行；**定义要写够"丰富一句话"**（每词 ~40-70 CJK：是什么 + 为什么重要），以撑起 zh CJK≥4000。
  - `<h2>` 怎么用这份速查 + 收官：怎么按需查、想深入就点链接回硬核课；一句收尾呼应总目标"读懂 + 动手 + 贡献 llama.cpp"。
  - 硬性：zh CJK≥4000（术语表丰富定义是主力）、en CJK==0、逐节 `<p>` 对齐、≥3 图（学习路径 trace + 概念依赖 SVG（在 `.trace` 内）+ ≥2 张 table.t）；索引课 `<details>` 可省或 1 张；代码段可省（以表与链接为主）。
  - **链接纪律**：所有 `<a href>` 指向真实存在的 `NN-slug.html`（先 grep `lessons/` 核实文件名）；`check_links.py` 必须全过。

- [ ] **Step 5: quiz（40）** 3 MCQ + 1 开放：「概念依赖图里谁是地基？（`ggml_tensor`）」「计算图（cgraph）的真实结构定义在哪？（`ggml/src/ggml-impl.h`，公开头只前向声明）」「KV cache 由谁持有？（`llama_context`）」「开放：用一句话各概括 9 个部分分别解决什么问题。」

- [ ] **Step 6: 重建+校验**：`cd src && python3 build.py && python3 check_html.py && python3 check_links.py` 全绿；index 变"共 40 课 · 9 个部分"；硬性达标；新增 trace=4（学习路径 zh/en + 概念图 SVG 所在 `.trace` zh/en）、`<svg`=2（zh/en 概念图，`xml.dom.minidom` 可解析、英文 SVG 纯 ASCII）；**所有术语表链接 `check_links.py` 全解**；grep 渲染无 `&amp;lt;`；chromium 目检概念图深浅背景可读。

- [ ] **Step 7: commit**：`feat: add lesson 40 glossary and concept index (bilingual) with dependency SVG + quiz` + `Assisted-by: GitHub Copilot`（暂存 5 源文件 + 重建的全部 HTML，提交后 git status 干净）。

## Task 4: 收尾（roadmap 勾选 + 全量验证 + 整体复审 + 完成分支）

**Files:** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（勾 M8）。

- [ ] **Step 1: 更新 roadmap**：里程碑总表 M8 行"状态"`待写` -> `完成`；状态追踪 `- [ ] M8 ...` -> `- [x] M8 ...`。commit `docs: mark M8 (part 8 & 9) done` + `Assisted-by: GitHub Copilot`。
- [ ] **Step 2: 全量验证**（master 合并前在分支上跑）：
  - `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` = `Wrote 41 files`、`structural check passed`（0 error/warning）、`all internal links resolve`。
  - index 显示"共 **40** 课 · **9** 个部分"；第八部分 2 课、第九部分 1 课都在导航里。
  - 三课（38-40）：按 `<h2>` 分节 `<p>` 计数中英相等；zh CJK ≥ 4000、en CJK == 0、en 无 unicode（`≈`/`±`/`×`/箭头/em-dash/`…`/`·`/`Δ`）；无连续顶层 `<p>` > 3；无 `&amp;lt;`/`&amp;gt;`/`&amp;amp;` 双重转义；quiz 字符串无裸 `<`。
  - 新增 trace：`class="trace"` 全站净增 8（L38 1×2 + L39 1×2 + L40 2×2）；`<svg` 净增 2（L40 概念图 zh/en），均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII、深色色板；用 chromium 目检三课渲染正常。
- [ ] **Step 3: 第八/九部分整体复审**（建议）：派一个 superpowers:code-reviewer 子代理（当前模型 `claude-opus-4.8`）复审 `master..HEAD` 的 3 课跨课一致性（标题/卡片/图/trace 风格统一、真实源码引用准确、双语纪律、范围未越界——只改 part8/part9 + 登记，不碰 1-37 课内容/build 基础设施；三处范围取舍——L38 量化算法点到、L39 CI 只概览、L40 速查不重复深讲——已落实；L40 所有跳转链接真实有效；三课形成"动手实战 + 速查收官"的连贯收尾）。修复任何确证问题后 amend。
- [ ] **Step 4: 完成分支**：用 superpowers:finishing-a-development-branch，先过验证门，再按既定偏好：本地 `--no-ff` 合并 master + 删分支（无需再询问）。合并后在 master 上复跑全量验证确认干净。

---

## 计划自审（writing-plans self-review）

- **Spec 覆盖**：设计 §每课设计 的 L38/L39/L40 三课 -> Task 1/2/3；统一交付标准 -> 各 task Step 4 硬性 + Step 6 校验；roadmap 勾选/全量验证/整体复审/完成分支 -> Task 4。三处范围取舍（L38 量化点到、L39 CI 概览、L40 速查不重复深讲）在对应 task 的源码事实与结构里均落实。L40 "两者并重"（分类术语表 + 概念依赖图）在 Task 3 Step 0（预生成 SVG）+ Step 4（术语表 + 概念图两节）落实。✓ 无遗漏。
- **占位符扫描**：无 TBD/TODO；各 task 的源码事实为真实"文件+符号"（已核实，含"convert 已重构进 conversion 包""sampler 是 llama-sampler.cpp 非 llama-sampling.cpp"两处关键纠偏）、登记字符串为可直接粘贴的精确内容、quiz 给了具体题目与答案要点。✓
- **类型/命名一致**：第八部分两课统一 `src/part8.py` 的 `LESSON_38/39`、`import part8`、part 标签 `第八部分 · 实战与贡献`/`Part 8 · Practice & contributing`、文件名 `38-convert-hf`/`39-build-contribute`；第九部分 `src/part9.py` 的 `LESSON_40`、`import part9`、标签 `第九部分 · 速查`/`Part 9 · Quick reference`、文件名 `40-glossary`。Task 1 新建 part8.py、Task 2 追加；Task 3 新建 part9.py。✓
- **风险点**：(1) L40 术语表链接死链——Step 4/6 点名先 grep `lessons/` 核实 slug + `check_links.py` 全过；(2) L40 索引课 CJK 偏低——靠"丰富一句话定义"撑，必要时分轮扩写（沿用 M7 经验，逐节镜像扩写）；(3) L40 概念图 SVG——Task 3 Step 0 控制器先 Python 预生成校验（well-formed + 英文 ASCII + 深色可读）再嵌；(4) 子代理写整课可能失败——执行方式已写明"独立核验 git + 失败则控制器亲自执笔"；(5) L39 引用本仓库 AI 政策——Step 4 点名中立陈述、不夹带评论。
