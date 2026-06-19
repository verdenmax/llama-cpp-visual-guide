# M8 · 第八部分（实战与贡献）+ 第九部分（速查）设计文档

> **配套：** 父级总设计 `2026-06-13-llama-cpp-visual-guide-design.md`（§第八/九部分）· roadmap `2026-06-13-llama-cpp-visual-guide-roadmap.md`（M8 行）。
> **执笔基线：** 沿用 M1-M7 的零依赖 Python 静态站点生成器与双语机制；本里程碑新建 `src/part8.py`（写 `LESSON_38/39`）与 `src/part9.py`（写 `LESSON_40`）。

## Goal

为图解教程补齐**第八部分 · 实战与贡献**（2 课：38-39）与**第九部分 · 速查**（1 课：40），共 **3 课**。视角从"读懂源码"转向"动手用 + 参与贡献 + 快速查阅"：把一个 HuggingFace 模型亲手转成 GGUF（L38）、把仓库编译/调试/测试/提 PR 的真实流程走一遍（L39）、再用一份分类术语表 + 概念依赖图把全书 40 课串成可速查的知识网（L40）。这是收官三课，呼应总设计的目标读者"准备阅读/调试/贡献 llama.cpp 的开发者"。

## 总基调（沿用 M6/M7 硬核路线）

- **硬核但落地**：L38 逐行讲真实的 `conversion/` 包与 `gguf-py` 写盘逻辑；L39 给真实可跑的 CMake/ctest 命令与 `CONTRIBUTING` 政策；L40 每个术语都标"源码位置 + 课号链接"，不空谈。
- **用图把流程讲透**：L38/L39 各内嵌 1 个 worked-example trace（Style A · 站点流）；L40 含 1 张概念依赖图（Style C · 手绘 SVG）+ 全书学习路径 trace + 分类术语表。
- 沿用 M1-M7 双语基线（见下"统一交付标准"）。
- **与已写内容不重叠**：原父设计 Part 8 的"投机解码/多模态/性能内存"已在 L30/L34/L36 覆盖；本部分聚焦"转换模型"与"编译贡献"两件前面没讲的实战事；L40 是纯索引/速查，不重复讲解。

## 范围取舍（与用户确认）

1. **L38 转换（重大结构发现）**：根目录 `convert_hf_to_gguf.py` 已重构为 **298 行薄 CLI 包装**，真正机制在新的 **`conversion/` 包**（`conversion/base.py` 的 `ModelBase`/`TextModel`、`@ModelBase.register` 注册表、每个架构一个模块如 `conversion/llama.py`）。本课**讲这个包布局**，不是单文件；`gguf-py` 的 `GGUFWriter` 讲到**字节布局**（header/KV/tensor-info/对齐/data），但具体量化算法只点到（呼应 L25/L26）。
2. **L39 贡献**：CMake 多后端、`ctest`/`test-backend-ops`、`clang-format`、`CONTRIBUTING` 的 **AI 政策**都讲；CI 矩阵（~50 个 workflow）只做**概览**，不逐个文件展开。
3. **L40 速查（与用户确认：两者并重）**：既有**完整分类术语表**（每词一句话定义 + 源码位置 + 跳转课号），也有**一张大的概念依赖图**（Style C SVG，画核心概念之间的依赖关系）；术语只"一句话查"，深讲仍指回对应课。

## 里程碑范围（M8 = 课 38-40，跨两个部分）

| 课 | slug（产出文件） | 标题 zh / en | 部分标签 | 内嵌图（风格） |
| --- | --- | --- | --- | --- |
| 38 | `38-convert-hf.html` | 从 HF 转换模型 / Converting HF models to GGUF | 第八部分 · 实战与贡献 | HF 权重 -> conversion 分发 -> 写元数据/张量 -> GGUF 文件（A·站点流）+ GGUF 字节布局（layers） |
| 39 | `39-build-contribute.html` | 编译·调试·测试·贡献 / Build, debug, test, contribute | 第八部分 · 实战与贡献 | clone -> cmake 构建 -> 改 -> ctest 验证 -> clang-format -> 提 PR -> CI（A·站点流） |
| 40 | `40-glossary.html` | 术语表·概念索引 / Glossary & concept index | 第九部分 · 速查 | 全书 9 部分学习路径（A·站点流）+ 概念依赖图（C·SVG）+ 分类术语表 |

> 新增两个部分标签 `第八部分 · 实战与贡献 / Part 8 · Practice & contributing`、`第九部分 · 速查 / Part 9 · Quick reference`；index 课数 37->40、部分数 7->9（index 自动从 PAGES 标签推导）。

## 每课设计

### 课 38 · 从 HF 转换模型（`convert_hf_to_gguf.py` + `conversion/` 包 + `gguf-py`）

**源码事实（核实 2026-06-19，文件+符号为准）：** 薄 CLI `convert_hf_to_gguf.py`（`parse_args` 的 `--outtype`、`main` 里 `ModelBase.load_hparams` -> `get_model_architecture` -> `get_model_class` 分发、`model_instance.write()`）。`conversion/base.py`：`class ModelBase`（`_model_classes` 注册表、`@classmethod register(*names)` 装饰器工厂）、`get_tensors()`、`map_tensor_name()`（经 `gguf.TensorNameMap` 把 HF 名翻成 GGUF 名）、`set_gguf_parameters()`（基类 `NotImplementedError`）、`modify_tensors()`（子类重塑/拆分/permute）、`prepare_tensors()`（主循环：遍历张量 -> `modify_tensors` -> 选量化类型 -> `gguf_writer.add_tensor`）、`write()`（`prepare_tensors`->`prepare_metadata`->`write_header_to_file`->`write_kv_data_to_file`->`write_tensors_to_file`）；`class TextModel(ModelBase)` 建 `tensor_map`、`set_vocab/set_gguf_parameters`。`conversion/llama.py`：`@ModelBase.register("LlamaForCausalLM", ...)` -> `class LlamaModel(TextModel)`、`permute()`、`modify_tensors()`。`gguf-py/gguf/`：`constants.py`（`GGUF_MAGIC`/`GGUF_VERSION=3`/`GGUF_DEFAULT_ALIGNMENT=32`、`Keys`、`GGMLQuantizationType`、`GGUFValueType`）、`gguf_writer.py`（`GGUFWriter`：`write_header_to_file` 写 magic/version/tensor_count/kv_count、`write_kv_data_to_file` 写类型标记的 KV、`write_ti_data_to_file` 写张量信息、`write_tensors_to_file` 对齐后写 data）、`tensor_mapping.py`（`TensorNameMap`）。指向 `docs/development/HOWTO-add-model.md`（新增一个架构 = 在 `conversion/` 加一个注册子类）。

**结构：**
- 导语：你下载的 HF 模型是一堆 `.safetensors` + `config.json`；llama.cpp 只吃 `.gguf`。这一课把"转换"这件事拆开：谁来分发到对应架构、张量名怎么翻译、元数据和权重最终怎么落成一个文件。
- `<h2>` 一条命令背后：`convert_hf_to_gguf.py` 现在只是薄 CLI；真正干活的是 `conversion/` 包。**真实代码**：`main` 里 `load_hparams` -> 读 `config.json` 的 `architectures` -> `get_model_class` 分发到注册的子类。
  - **trace（Style A·站点流）**：HF 目录 -> `load_hparams` 读架构 -> `get_model_class` 选 `LlamaModel` -> `set_gguf_parameters` 写超参 -> `prepare_tensors` 逐张量 `modify_tensors` + 改名 -> `GGUFWriter` 写盘 -> `model.gguf`。
- `<h2>` 注册表与架构分发：`@ModelBase.register("LlamaForCausalLM", ...)` 怎么把"HF 架构名 -> Python 类"登记进 `_model_classes`；新增一个模型就是加一个注册子类（指向 HOWTO-add-model）。**真实代码**：装饰器工厂 + 一个 `LlamaModel` 子类骨架。
- `<h2>` 张量改名与超参：`set_gguf_parameters`（n_layer/n_embd/rope 等写成 GGUF KV）+ `modify_tensors`/`map_tensor_name`（经 `TensorNameMap` 把 `model.layers.0.self_attn.q_proj.weight` 翻成 `blk.0.attn_q.weight`，Llama 还要 `permute` Q/K）。**真实代码 + 名称对照**（`.cols`：HF 名 vs GGUF 名）。
- `<h2>` GGUF 文件长什么样：`GGUFWriter` 的写盘四段——header（magic/version/tensor_count/kv_count）-> KV 段（带类型标记）-> 张量信息段（名/维度/dtype/offset）-> 对齐到 32B -> 原始张量数据。**图（layers）**：GGUF 字节布局自上而下的分段堆叠。`--outtype` 在这一步决定每个张量存成 f16/q8_0/...（呼应 L25/L26 量化）。
- `<h2>` 折叠深挖（≥2）：(1) 为什么要"对齐"（mmap 时按 32B 边界对齐，呼应 L05 mmap）；(2) `set_vocab` 与 tokenizer（SentencePiece/BPE 词表怎么一并写进 GGUF，呼应 L07/L08）。
- **范围取舍落实**：量化算法本身只点到（指回 L25/L26）；只讲转换流程不抠每种 dtype 的打包。

### 课 39 · 编译·调试·测试·贡献（`docs/build.md` + `tests/` + `CONTRIBUTING.md` + CI）

**源码事实：** `docs/build.md`：CPU `cmake -B build && cmake --build build --config Release`、`-DGGML_CUDA=ON`/`-DGGML_METAL=OFF`/`-DGGML_VULKAN=ON` 等后端开关、产物落 `build/bin/`。`tests/CMakeLists.txt`：`llama_test(...)`/`llama_build_and_test(...)` 宏注册 `ctest` 用例（`test-tokenizer-0`、`test-backend-ops.cpp`、`test-quantize-fns.cpp`、`test-sampling.cpp` 等）。`CONTRIBUTING.md`：**AI 使用政策**（不接受全/主要由 AI 生成的 PR；AI 改写仍算 AI 生成）、PR 规范（一个 PR 一个功能、CPU 支持优先、ggml 改动要跑 `test-backend-ops`）、维护者 squash-merge + 标题 `<module> : <title> (#NNNN)`、编码规范（`clang-format` v15+、命名）。`.github/workflows/`（~50 个）：分后端构建矩阵 + `code-style.yml`/`python-lint.yml`/`build-sanitize.yml` 等质量门。

**结构：**
- 导语：读懂了源码，怎么动手？这一课走一遍真实开发回路——编译出二进制、跑测试确认没坏、按规范改、提一个能被接受的 PR。特别强调本仓库**对 AI 生成 PR 的明确政策**。
- `<h2>` 编译：CMake 一套命令编译；不同后端用不同开关（`-DGGML_CUDA=ON` 等）；Debug/Release 区别；产物在 `build/bin/`。**图（table.t）**：后端 -> CMake 开关 -> 适用平台。**真实命令**（`<pre class="code">`）。
  - **trace（Style A·站点流）**：clone -> `cmake -B build` 配置 -> `cmake --build` 编译 -> 改代码 -> `ctest`/`test-backend-ops` 验证 -> `clang-format` -> 提 PR（一功能/CPU 优先）-> CI 矩阵跑全平台。
- `<h2>` 测试：`ctest` 怎么跑；`llama_test`/`llama_build_and_test` 宏怎么把一个 `test-*.cpp` 注册成用例；几类代表测试（tokenizer / backend-ops / quantize / sampling）各测什么。**真实代码**：`tests/CMakeLists.txt` 的宏 + 一条 `ctest` 命令。
- `<h2>` 调试：Debug 构建 + sanitizer（`build-sanitize`）；`test-backend-ops` 在加新算子/改 ggml 时的角色（CONTRIBUTING 点名）；`--verbose`/日志。**点到为止 + 指向**。
- `<h2>` 贡献规范（重点）：`CONTRIBUTING` 的 **AI 政策**（不接受全/主要 AI 生成的 PR、要能独立解释与维护自己的代码）、一个 PR 一个功能、CPU 优先、提交标题格式、`clang-format`。**图（layers/cols）**：一个合格 PR 的检查清单 / CI 质量门。
- `<h2>` 折叠深挖（≥2）：(1) 为什么"CPU 支持优先"（参考实现 + `test-backend-ops` 拿 CPU 当 ground truth，呼应 L31/L33）；(2) CI 矩阵为什么这么多（每个后端一套构建，呼应 L33 后端调度）。
- **范围取舍落实**：CI 只概览不逐 workflow 展开。

### 课 40 · 术语表·概念索引（全书 40 课 · 速查）

**源码事实（每个术语标"单一最佳定义点"）：** GGUF（`ggml/include/gguf.h` / `gguf-py/gguf/constants.py`）、ggml_tensor（`ggml/include/ggml.h`）、backend（`ggml-backend.h`）、KV cache（`src/llama-kv-cache.h`）、batch（`include/llama.h` `llama_batch`）、context（`src/llama-context.h`）、cgraph（`ggml/src/ggml-impl.h`）、ggml_type/量化（`ggml/include/ggml.h`）、vocab（`src/llama-vocab.h`）、sampler（`src/llama-sampler.cpp`，注意不是 `llama-sampling.cpp`）、RoPE（`ggml_rope`）、MoE（`build_moe_ffn`）、speculative（`common/speculative.h`）、mtmd（`tools/mtmd/mtmd.h`）、SSM（`ggml_ssm_scan`）等 ~25 个核心词。每词关联其主讲课号。

**结构：**
- 导语：到这里全书 40 课讲完了。这一课不教新东西，是一张**速查地图**：先看 9 个部分怎么层层递进，再看核心概念之间谁依赖谁，最后用分类术语表"一句话查 + 点链接跳回对应课"。
- `<h2>` 全书地图（9 部分学习路径）：**trace（Style A·站点流）**：第一部分 宏观全景 -> 第二部分 前置基础 -> 第三部分 ggml 引擎 -> 第四部分 llama 内部 -> 第五部分 API 与工具 -> 第六部分 底层内核 -> 第七部分 进阶专题 -> 第八部分 实战贡献 -> 第九部分 速查；每站一句话说"这部分解决什么"。
- `<h2>` 概念依赖图：**图（Style C·SVG）**：核心概念的依赖关系——`ggml_tensor` 是地基，`cgraph` 由张量组成，`backend` 执行图，`llama_context` 持有 KV cache，`batch` 驱动 decode，`sampler`/`vocab` 收尾；用箭头画"谁建立在谁之上"。深浅背景都可读、英文区纯 ASCII。
- `<h2>` 分类术语表：若干 **`table.t`**，按类分（核心数据结构 / 推理流程 / 内核与后端 / 进阶机制与工具）；每行：术语 | 一句话定义 | 源码位置 | 跳到第几课（`<a href>` 站内链接）。
- `<h2>` 怎么用这份速查 + 收官：怎么按需查、怎么继续深入（指回硬核课）；一句收尾呼应总目标。
- **形态落实（与用户确认两者并重）**：完整分类术语表 + 一张概念依赖 SVG 都给齐。

## 统一交付标准（每课硬性达标，与 M1-M7 一致）

每课 `LESSON_NN = {"zh": r'''...''', "en": r'''...'''}`，须满足：

- **结构**：导语 `<p>` + 教学卡片（macro/analogy/key/spark 酌用，≥2 张深挖 `<details>`；L40 索引课可少用 `<details>`、以表为主）+ **≥3 个图示**（`cols`/`layers`/`table.t`/`trace` 等，单语 ≥3，含 ≥1 个 trace）+ **≥2 段真实/简化代码片段**（`<pre class="code">`；L40 以术语表/链接为主，代码段可少）+ 各课规定的内嵌图。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace` div、内联 `<svg>`、`<table>` 不计入）。
- **中文密度**：zh CJK ≥ 4000；**en CJK == 0**（纯 ASCII；代码/SVG/图示文本用 `-`/`->`/`...`/`~`/`+/-`，不用 em-dash/unicode 箭头/`≈`/`±`/`×`/`·`/`Δ`）。L40 术语表靠"丰富一句话定义"把 CJK 写够。
- **无文字墙**：连续顶层 `<p>` <= 3。
- **转义**：渲染 `<`/`>`/`&` 须转义（`&lt;`/`&gt;`/`&amp;`）；无双重转义（`&amp;lt;`）。**quiz 字符串同样须预转义**（`quizzes.py` 的 q/opt/why 原样插值，`<image>` 须写 `&lt;image&gt;`）。
- **trace 规范**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn` 等）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>`（zh aria-label 中文、en 纯 ASCII），合法 XML、坐标不溢出 viewBox、深浅两种 `.trace` 背景下都可读（深色文字只放白底框内、自由文字用中间色 `#5b6470`），trace 不与 `<div class="card">` 紧邻。
- **源码引用**：以"文件 + 符号名"为主、不写死行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验日期 2026-06-19）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测（3 MCQ + 1 开放）。
- **登记**：`shell.PAGES`（filename、zh/en 短标题、部分标签）、`shell.SUBTITLES`、`registry.CONTENT`（filename -> `part8/part9.LESSON_NN`）；index 课数 37->40、部分数 7->9（index 自动从 PAGES 推导）。

## 与 roadmap 衔接

- 完成后：roadmap M8 行"状态"`待写`->`完成`、状态追踪 `- [ ] M8`->`- [x] M8`。
- 执行：superpowers:subagent-driven-development（一课一个 task，顺序执行；收尾 task 勾 roadmap + 全量验证 + 第八/九部分整体复审 + 完成分支）。**注意**：M5/M6/M7 经验表明后台 general-purpose 子代理写整课常中途失败（零写入），实现时由控制器亲自照模板执笔，仍保留完整 spec+质量双重审查。
- 关键 Style-C SVG（L40 概念依赖图）先用 Python 预生成并校验（well-formed + 英文 ASCII + 坐标不溢出 + 深色可读），再喂给实现（沿用 M6/M7 做法）。
