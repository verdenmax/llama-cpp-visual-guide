# M4a · 第四部分（上）llama 推理内部 — 加载与运行时核心 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 产出本可视化指南第四部分（上）共 6 课（14-19），讲清"一个 GGUF 模型如何被加载、组织成架构、搭成前向图、装进上下文、按批处理、并用 KV cache 高效自回归"——即 ggml 之上的 `src/llama-*` 运行时核心。

**Architecture:** 沿用现有零依赖 Python 静态站点生成器。新建 `src/part4.py` 承载 `LESSON_14..19` 双语内容字典；每课在 `src/shell.py`（PAGES/SUBTITLES）、`src/registry.py`（CONTENT，新增 `import part4`）、`src/quizzes.py`（QUIZZES）登记；`python3 src/build.py` 生成 HTML，`check_html.py`/`check_links.py` 校验。index 部分数从 3 -> 4。

**Tech Stack:** Python 静态站点生成器（build.py/shell.py/registry.py/quizzes.py/check_*.py）；内容在 part4.py；复用 CSS 图组件（layers/vflow/flow/cols/cellgroup/timeline/`table.t`）。

---

## 里程碑范围（M4a = 课 14-19）

| 课 | 标题（zh / en） | 主源文件 |
| --- | --- | --- |
| 14 | 模型加载 / Model loading | `src/llama-model-loader.{h,cpp}` · `src/llama.cpp` |
| 15 | 架构与超参 / Architecture & hyperparameters | `src/llama-arch.{h,cpp}` · `src/llama-hparams.h` |
| 16 | 构建计算图 / Building the compute graph | `src/llama-graph.{h,cpp}` · `src/models/*.cpp` |
| 17 | 上下文与会话 / Context & session | `src/llama-context.{h,cpp}` · `include/llama.h` |
| 18 | 批处理 / Batching | `include/llama.h` · `src/llama-batch.{h,cpp}` |
| 19 | KV cache / The KV cache | `src/llama-kv-cache.{h,cpp}` · `src/llama-kv-cells.h` |

衔接：本部分承接第三部分（ggml 引擎：L08 context/arena、L09 图、L10 执行、L12 量化块、L13 GGUF）；M4b（20-24 vocab/sampler/chat/grammar/LoRA）随后。

## 统一交付标准（每课硬性达标，与 M1-M3 一致）

- **zh 正文纯中文 CJK ≥ 4000**（`[\u4e00-\u9fff]` 计数）；**en 正文 0 个 CJK**，且与 zh **逐段对齐**（按 `<h2>` 切分，每段 `<p>` 数 zh==en）。
- **3-5 张图**，取自既有复用组件（layers/vflow/flow/cols/cellgroup/timeline/`table.t`），含 **≥1 张概念/结构图**；两种语言图数相同。
- **2-3 个 `<pre>` 代码片段**：至少一个"简化自真实源码"的片段，**引用 文件+符号、不带行号**；可含伪代码。
- **2-3 个 `<details class="accordion">` 深挖**。
- 卡片齐全：`lead`、`analogy`（🔌）、`key`（✅ 关键要点）、`spark`（💡 设计洞察）。
- quiz：3 个 MCQ（`answer` 为 0 基正确项下标，构建器会确定性洗牌）+ 1 个 open，全双语 `{"zh","en"}`。
- ASCII 守则：代码块内不用 `→`/`×`/`÷`/`…`，用 `->`/`x`/`/`/`...`；散文沿用既有风格（`·`、`×`、`—` 可用，但**不用 `→`**）。

## 执行方式

按用户既定：**subagent 驱动**（每课跑 spec 合规 + 代码质量两段审查子代理，全程 `claude-opus-4.8`）。鉴于内容子代理写大段 HTML 易卡死，**控制者直接执笔每课内容**，再对每课跑 spec + 质量双审子代理（只读）。分支 `build/m4a-part4`，6 课顺序产出，验收后合并回 master。

---

## 验证过的源码事实（2026-06-15 对真实源树核验；引用"文件+符号"，无行号）

> ⚠ 源码已演进，以下为**当前真相**，若与旧教程/直觉冲突，以此为准。写每课时严格据此，勿臆造已不存在的名字。

**L14 模型加载** — `src/llama-model-loader.{h,cpp}` / `src/llama.cpp`
- `struct llama_model_loader` 持有 GGUF：`gguf_context * metadata`（由 `gguf_context_ptr metadata_ptr` 拥有），构造时 `gguf_init_from_file(...)` 填充。
- 张量清单：`weights_map`（`std::map<std::string, llama_tensor_weight, ...>`）；嵌套 `struct llama_tensor_weight{ idx(来源文件号), offs, ggml_tensor * tensor }`。**是 map，不是 vector**。
- mmap：`bool use_mmap` + `llama_mmaps mappings` + `llama_files files`；数据加载 `load_data_for` / `load_all_data`。
- 读 KV/超参：模板 `get_key`（按 `enum llm_kv` 或 `std::string` 重载）+ `get_arr`/`get_arr_n`/`get_key_or_arr`；成员 `LLM_KV llm_kv` 把枚举映射到 GGUF 键串。
- 分片 SPLIT：`LLM_KV_SPLIT_COUNT="split.count"`（另 `split.no`/`split.tensors.count`）；读入 `uint16_t n_split`；文件名由 `llama_split_path`（`src/llama.cpp`）按 `"%s-%05d-of-%05d.gguf"` 拼。
- 入口：`llama_model_load_from_file` / `llama_model_load_from_splits`（`src/llama.cpp`）-> `llama_model_load_from_file_impl` -> 静态 `llama_model_load`。

**L15 架构与超参** — `src/llama-arch.{h,cpp}` / `src/llama-hparams.h`
- `enum llm_arch`（`LLM_ARCH_LLAMA`/`LLM_ARCH_QWEN2`…）+ `LLM_ARCH_NAMES`（`{LLM_ARCH_LLAMA,"llama"}`）。
- `enum llm_kv` + `LLM_KV_NAMES`：`LLM_KV_BLOCK_COUNT->"%s.block_count"`、`LLM_KV_EMBEDDING_LENGTH->"%s.embedding_length"`；键串由 `struct LLM_KV::operator()` 用 `format(模板, 架构名)` 生成。
- `enum llm_tensor` + `LLM_TENSOR_NAMES`：`{LLM_TENSOR_TOKEN_EMBD,"token_embd"}`、`{LLM_TENSOR_OUTPUT_NORM,"output_norm"}`、`{LLM_TENSOR_ATTN_Q,"blk.%d.attn_q"}`；名字构造器是 `struct LLM_TN`/`LLM_TN_IMPL`（`tn(...)` 是 `LLM_TN::operator()`，**不是自由函数**）。
- `struct llama_hparams`（`src/llama-hparams.h`）：标量字段 `n_embd`、`n_ctx_train`、`rope_freq_base_train`、`f_norm_rms_eps`。
- ⚠ `n_layer()` / `n_head(il)` / `n_head_kv(il)` / `n_ff(il)` / `n_rot(il)` / `n_embd_head_k(il)` / `n_embd_head_v(il)` 是**访问器方法**，不是标量字段；后备存储 `n_head_arr`/`n_head_kv_arr`/`n_ff_arr`（`std::array<…, LLAMA_MAX_LAYERS>`），层数存 `n_layer_all`。
- ⚠ `n_vocab` **不在** `llama_hparams`；词表大小来自 `llama_vocab::n_tokens()`（GGUF 键 `LLM_KV_VOCAB_SIZE="%s.vocab_size"`）。

**L16 构建计算图** — `src/llama-graph.{h,cpp}` / `src/models/*.cpp` / `src/llama-model.cpp`
- ⚠ 旧 `llm_build_context`/`llm_build_llama` **已不存在**。图基类是 `struct llm_graph_context`（`src/llama-graph.h`）；**每个架构一个文件** `src/models/<arch>.cpp`。
- 每架构是 `struct llama_model_<arch> : llama_model_base`（`src/models/models.h`），含 `load_arch_hparams`/`load_arch_tensors`/`build_arch_graph(const llm_graph_params&)`；实际图是嵌套模板 `llama_model_<arch>::graph<embed> : llm_graph_context`。
- 派发：`llama_model::build_graph(const llm_graph_params&)`（`src/llama-model.cpp`）调虚函数 `build_arch_graph(...)`，再 `build_pooling`/`build_sampling`/`build_dense_out`，返回 `res->get_gf()`（`ggml_cgraph *`）。
- 复用积木是 `llm_graph_context` 的方法（`src/llama-graph.h`）：`build_norm`/`build_ffn`/`build_attn`（多重载）+ `build_attn_mha`；输入构造 `build_inp_embd`/`build_inp_pos` 等。
- 图输入类型 `class llm_graph_input_*`（派生自 `llm_graph_input_i`）：`_embd`/`_pos`/`_pos_bucket`/`_out_ids`/`_attn_no_cache`/`_attn_kv`/`_attn_kv_iswa`/`_mem_hybrid`。
- 图容器 `struct llm_graph_result` 持 `ggml_cgraph * gf`，`get_gf()`——即第三部分的 `ggml_cgraph`。

**L17 上下文与会话** — `src/llama-context.{h,cpp}` / `include/llama.h`
- `struct llama_context`（`src/llama-context.h`）持有：`llama_cparams cparams`；内存抽象 `llama_memory_ptr memory`（⚠ 泛化的"memory"，不是裸 KV 字段）；调度器 `ggml_backend_sched_ptr sched`。
- ⚠ 输出缓冲是 `buffer_view<float> logits` 与 `buffer_view<float> embd`（不是 `std::vector<float>`）。
- `struct llama_context_params`（`include/llama.h`）公开字段：`n_ctx`/`n_batch`/`n_ubatch`/`n_seq_max`/`n_threads`/`n_threads_batch`/`rope_scaling_type`/`pooling_type`/`attention_type`/`flash_attn_type`/`rope_freq_base`/`rope_freq_scale`/`yarn_*`/`type_k`/`type_v`/`embeddings`/`offload_kqv`；内部镜像 `struct llama_cparams`（`src/llama-cparams.h`）。
- `llama_decode`/`llama_encode`；输出访问 `llama_get_logits`/`llama_get_logits_ith`/`llama_get_embeddings*`（`include/llama.h`）。

**L18 批处理** — `include/llama.h` / `src/llama-batch.{h,cpp}`
- 公开 `struct llama_batch`（`include/llama.h`）：`n_tokens`、`llama_token * token`、`float * embd`、`llama_pos * pos`、`int32_t * n_seq_id`、`llama_seq_id ** seq_id`、`int8_t * logits`（**per-token 输出标志**，源码注释 `// TODO: rename this to "output"`）。
- `llama_batch_get_one`/`llama_batch_init`/`llama_batch_free`。
- ⚠ `struct llama_sbatch` **已被** `class llama_batch_allocr`（`src/llama-batch.h`）取代：`init(...)` 净化/填充 `llama_batch`，再 `split_simple(n_ubatch)`/`split_equal(...)`/`split_seq(...)` 切成 `llama_ubatch`；微批 `struct llama_ubatch` 的输出标志字段叫 `int8_t * output`（非 logits）。

**L19 KV cache** — `src/llama-kv-cache.{h,cpp}` / `src/llama-kv-cells.h` / `src/llama-memory*.{h,cpp}`
- ⚠ 统一缓存类是 `class llama_kv_cache`（`src/llama-kv-cache.h/.cpp`，**不是** `llama_kv_cache_unified`），派生自 `class llama_memory_i`（`src/llama-memory.h`）；每次 decode 的上下文是 `class llama_kv_cache_context`。
- cell/slot 状态：`class llama_kv_cells`（`src/llama-kv-cells.h`）——`pos`（`std::vector<llama_pos>`）、`used` 集合、按序列 `seq_pos`、可选 2D `x/y`（M-RoPE）、`size()`；缓存暴露滚动 `head()`；后备 `llama_kv_cells_vec v_cells`。
- 序列操作是 `llama_kv_cache` 方法（override `llama_memory_i`）：`seq_rm`/`seq_cp`/`seq_keep`/`seq_add`（上下文移位）/`seq_div`；多序列靠每 cell 的 seq 集合。
- ⚠ 公开 C API 已从 `llama_kv_self_seq_*` **改名为** `llama_memory_seq_*`：`llama_memory_seq_rm`/`_cp`/`_add`，并 `llama_get_memory` 返回 `llama_memory_t`（旧 `llama_kv_self_*` 已从 `include/llama.h` 移除）。
- ⚠ 显式 defrag **已移除**（`llama_kv_cache` 无 defrag；`llama_context_params::defrag_thold` 标记 `[DEPRECATED]`）；上下文移位仍在（`seq_add` + `build_graph_shift`）。
- ⚠ 变体文件（当前真实存在）：iSWA 滑窗 `class llama_kv_cache_iswa`（`src/llama-kv-cache-iswa.{h,cpp}`）；recurrent `class llama_memory_recurrent`（`src/llama-memory-recurrent.{h,cpp}`）；hybrid `class llama_memory_hybrid`（`src/llama-memory-hybrid.{h,cpp}`，另有 `llama_memory_hybrid_iswa`）；基接口 `llama_memory_i`。

---

## Task 1: 课 14「模型加载 / Model loading」

> 第四部分第 1 课，开篇。承接 L13（GGUF 格式）与 L08（ggml_context）：讲 `llama_model_loader` 怎么把一个 `.gguf` 读成内存里带名字的张量清单 + 就位的数据指针，并处理分片。

**Files:** `src/part4.py`（**新建**，含 `LESSON_14`）、`src/registry.py`（新增 `import part4` + 登记）、`src/shell.py`（PAGES/SUBTITLES）、`src/quizzes.py`。产出 `14-model-loading.html`。

- [ ] **Step 1: 新建 `src/part4.py`** 顶部 `# -*- coding: utf-8 -*-`，定义 `LESSON_14 = {"zh": r"""...""", "en": r"""..."""}`（内容见 Step 4）。
- [ ] **Step 2: `src/registry.py`** 顶部加 `import part4`，CONTENT 末尾加 `"14-model-loading.html": part4.LESSON_14,`。
- [ ] **Step 3: 登记 PAGES/SUBTITLES（`src/shell.py`）**
```python
("14-model-loading.html", "模型加载", "Model loading",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"14-model-loading.html": ("llama_model_loader · GGUF metadata/张量清单 · mmap · 分片",
                          "llama_model_loader; GGUF metadata/tensor map; mmap; splits"),
```

- [ ] **Step 4: 执笔 LESSON_14（双语）**。**结构**：
1. `lead`：第三部分把 GGUF 文件（L13）和 ggml 引擎讲清了；这一课上到 `llama` 层，看 `llama_model_loader` 怎么把一个 `.gguf` 真正<strong>加载成内存里的模型</strong>——读 metadata、建张量清单、按 mmap 让数据就位、并处理多文件分片。
2. `analogy`（🔌）：loader 像<strong>收货验货员</strong>：先看箱单（GGUF metadata + tensor info）、核对货品清单（`weights_map` 按名字索引每个张量）、再按单提货（mmap 让每个张量 data 指针落到文件对应位置）；分片就是一批货分装多箱，验货员按 `of-N` 编号逐箱核对。
3. `<h2>` 加载总览 + **图1【加载流程】**（`vflow` 结构图，本课结构图）：`gguf_init_from_file` 读 metadata + tensor infos -> 建 `weights_map` -> （按 `use_mmap`）映射文件 -> 逐张量 `load_data_for`/`load_all_data` 让 `data` 指针就位。
4. **代码1（简化自 `src/llama-model-loader.h`）**：
```
struct llama_tensor_weight { uint16_t idx; size_t offs; ggml_tensor * tensor; };  // 来源文件号 + 偏移 + 张量
struct llama_model_loader {
    gguf_context * metadata;                          // GGUF 头(KV + tensor infos)
    std::map<std::string, llama_tensor_weight> weights_map;  // 按张量名索引
    bool use_mmap;  llama_mmaps mappings;             // 零拷贝数据
};
```
   讲：metadata 持有 GGUF 头；`weights_map` 是<strong>按名字</strong>的张量清单（不是 vector）；mmap 承接数据。
5. `<h2>` 读超参与张量 + 正文：超参由模板 `get_key(llm_kv, ...)` 从 GGUF KV 读出（呼应 L13 自描述、预告 L15 的 `llm_kv` 键）；每个张量按名字进 `weights_map`，名字对应 L13 的 tensor info（name/dims/type/offset）。**代码2（伪代码，加载流程）**：
```
ml = llama_model_loader(path)              # gguf_init_from_file 读头部
ml.get_key(LLM_KV_BLOCK_COUNT, n_layer)    # 从 KV 读超参(L15)
for name, w in ml.weights_map:             # 遍历张量清单
    t = create_tensor(name, w.tensor->ne)  # 在 ggml_context 里建元数据(L08)
    ml.load_data_for(t)                    # use_mmap: data 指进映射; 否则读入
```
6. `<h2>` 分片：大模型拆成多片 + **图2【分片 -> 一个逻辑模型】**（`cellgroup` 或 `cols`）：`model-00001-of-00003.gguf` | `00002` | `00003` -> 加载器当成<strong>一个逻辑模型</strong>读。正文：文件名格式 `"%s-%05d-of-%05d.gguf"`（`llama_split_path`）；`split.count`（`LLM_KV_SPLIT_COUNT`）告诉有几片；`weights_map` 里 `idx` 记每个张量来自第几片。
7. `<h2>` 入口与衔接 + **图3【入口调用链】**（`flow`）：`llama_model_load_from_file` / `llama_model_load_from_splits` -> `..._impl` -> 静态 `llama_model_load` -> 得到 `llama_model`（权重张量的 data 已指进 mmap）。正文：加载完，下一课（L15）讲这些张量怎么按架构/超参组织、怎么靠名字对上模型结构。
8. **深挖1**"为什么用 mmap 而不是把权重全读进内存？"：呼应 L13——零拷贝、秒加载、多进程共享、按页惰性载入；`use_mmap` 可关（某些后端/平台）。
9. **深挖2**"`weights_map` 为什么按名字索引、而不是按顺序？"：因为张量要<strong>按名字</strong>对应到架构里的具体位置（`blk.0.attn_q.weight` 等，L15）；分片时还要跨多个文件把同名张量统一查到，map 比下标稳。
10. **深挖3**"分片到底怎么对上的？"：`split.no`/`split.count`/`split.tensors.count` 三个 KV + 文件名的 `of-N` 编号；loader 按编号挨个打开、把各片的张量并进同一张 `weights_map`。
11. `key`（✅）：`llama_model_loader` = 读 GGUF `metadata`（`get_key` 取超参）+ `weights_map`（按名字的张量清单）+ mmap 数据；分片用 `"%s-%05d-of-%05d.gguf"` + `split.count`；入口 `llama_model_load_from_file` -> `_impl` -> `llama_model_load`。
12. `spark`（💡）：loader 把"<strong>解析格式</strong>"和"<strong>使用模型</strong>"解耦——它只负责把字节变成<strong>带名字的张量清单 + 就位的数据指针</strong>；至于这些张量怎么接成一张前向网络，是 L15（架构）和 L16（建图）的事。一个清晰的边界，让"支持新格式细节"和"支持新架构"互不打扰。

必须讲到：`llama_model_loader` 的职责；`weights_map` 按名字、mmap 数据；`get_key` 读超参；分片命名与 `split.count`；入口调用链。

- [ ] **Step 5: quiz（14）**：
- MCQ1 "`llama_model_loader` 主要做什么？" -> 正确："读 GGUF 的 metadata（超参）和 tensor infos、建按名字的张量清单、（按 use_mmap）把权重数据 mmap 或读入"；干扰：训练模型 / 量化权重 / 编译 GPU kernel。
- MCQ2 "一个被分片的大模型，文件名长什么样？" -> 正确："`model-00001-of-00003.gguf` 这种 `of-N` 编号"；干扰：随机哈希名 / 一个 .zip / 永远是单文件。
- MCQ3 "加载器怎么知道模型有多少层、多大维度？" -> 正确："用 `get_key(llm_kv, ...)` 从 GGUF 的 metadata KV 里读（自描述）"；干扰：猜测 / 读外部 config.json / 在代码里硬编码。
- OPEN "结合 L13，说说 `llama_model_loader` 为什么用 mmap 加载权重数据能做到'秒加载'又省内存。"

- [ ] **Step 6: 重建+校验**（`python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py`；index 变 "共 14 课 · 4 个部分"；CJK≥4000、en CJK=0、逐段对齐、≥3 图、≥2 深挖、≥2 片段）。
- [ ] **Step 7: commit**（`feat: add lesson 14 model loading (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 2: 课 15「架构与超参 / Architecture & hyperparameters」

> 第四部分第 2 课。L14 把张量加载成"带名字的清单"，但它们怎么组织成一个<strong>具体架构</strong>（llama/qwen2…）？这一课讲 `llama-arch`（`LLM_ARCH_*` / `LLM_KV_*` / `LLM_TENSOR_*` 命名约定）与 `llama-hparams`（超参）——把"一堆张量"变成"一个可建图的模型"的说明书。接 L14、L13。

**Files:** `src/part4.py`（追加 `LESSON_15`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `15-architecture-hparams.html`。

- [ ] **Step 1-3: 登记**
```python
("15-architecture-hparams.html", "架构与超参", "Architecture & hyperparameters",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"15-architecture-hparams.html": ("llm_arch · llama_hparams · LLM_TENSOR_NAMES 命名约定",
                                 "llm_arch; llama_hparams; LLM_TENSOR_NAMES naming"),
"15-architecture-hparams.html": part4.LESSON_15,
```

- [ ] **Step 4: 执笔 LESSON_15（双语）**。**结构**：
1. `lead`：L14 把权重加载成"带名字的张量清单"，可这些张量怎么知道自己是<strong>哪种架构</strong>、每层有哪些部件？这一课讲 `llama-arch`（架构标识 + KV 键 + 张量名三套约定）和 `llama-hparams`（几层/多宽/几个头）。
2. `analogy`（🔌）：arch + hparams 像<strong>建筑图纸 + 规格表</strong>：图纸（`llm_arch` + 张量命名约定）说明"这是哪种楼、每层有哪些构件、各叫什么名"；规格表（`hparams`）给出"几层、多宽、每层几个注意力头"。loader 按图纸的名字提货、按规格表的数搭楼。
3. `<h2>` 架构标识 `llm_arch` + **图1【架构名 -> 一套约定】**（`flow`）：GGUF 的 `general.architecture`（L13）= `"llama"` -> 查 `LLM_ARCH_NAMES` 得 `LLM_ARCH_LLAMA` -> 选定这套 KV/张量约定与建图函数（L16）。**代码1（简化自 `src/llama-arch.{h,cpp}`）**：
```
enum llm_arch { LLM_ARCH_LLAMA, LLM_ARCH_QWEN2, /* ... */ };
// LLM_ARCH_NAMES: { LLM_ARCH_LLAMA, "llama" }, { LLM_ARCH_QWEN2, "qwen2" }, ...
```
4. `<h2>` 超参 `llama_hparams` + **图2【关键超参】**（`<table class="t">`）：键 / 含义，列 `n_embd`(隐藏维)、`n_layer()`(层数)、`n_head(il)`(每层头数)、`n_head_kv(il)`(KV 头数/GQA)、`n_ff(il)`(FFN 维)、`n_rot(il)`(RoPE 维)、`rope_freq_base_train`、`f_norm_rms_eps`。**代码2（简化自 `src/llama-hparams.h`）**，并点明：
```
struct llama_hparams {
    uint32_t n_embd; uint32_t n_ctx_train; float rope_freq_base_train; float f_norm_rms_eps;
    std::array<uint32_t, LLAMA_MAX_LAYERS> n_head_arr, n_head_kv_arr, n_ff_arr;  // 按层
    uint32_t n_layer() const;  uint32_t n_head(uint32_t il) const;  // 访问器, 不是字段
};
```
   ⚠ 必须讲清两个坑：`n_layer()`/`n_head(il)` 是<strong>方法</strong>（背后是按层数组），因为 GQA/滑窗层各层可能不同；`n_vocab` <strong>不在</strong> hparams，来自 `llama_vocab::n_tokens()`（L20）。
5. `<h2>` 张量命名约定 + **图3【张量名模板】**（`cellgroup` 或 `cols`）：`token_embd`（词嵌入）、`blk.%d.attn_q`（第 d 层 Q 投影）、`output_norm`（输出归一）。正文：名字由 `LLM_TENSOR_NAMES` + 构造器 `LLM_TN`（`tn(LLM_TENSOR_ATTN_Q, "weight", il)`）拼出；loader（L14）正是按这些名字在 `weights_map` 里查张量、对上模型结构。
6. 正文：把"自描述"在架构层<strong>兑现一遍</strong>——`general.architecture` 选 arch；`%s.*` 的 KV（如 `llama.block_count`）用架构名填模板、由 `get_key` 读出超参；张量按 `LLM_TENSOR_NAMES` 命名一一对上。三套约定一咬合，"一堆张量"就成了"一个具体模型"。
7. **深挖1**"为什么 `n_head` 写成 `n_head(il)` 带层号？"：因为现代架构里不同层的头数/类型可能不同（GQA、滑窗层、混合注意力），按层取最通用；后备是 `n_head_arr` 这类按层数组。
8. **深挖2**"加一个新架构要改哪几处？"：① `enum llm_arch` 加一项 + `LLM_ARCH_NAMES` 加名字；② 填 hparams 读取（`load_arch_hparams`）；③ 写一份 `src/models/<arch>.cpp` 建图（L16）；④ 张量名映射。大量积木（attn/ffn/norm）复用，引擎主干不动。
9. **深挖3**"张量命名约定为什么这么重要？"：它是 loader（跨分片按名字查）、`gguf-py` 转换脚本（L02 写文件时）、和建图（L16 按名字取权重）三方的<strong>共同契约</strong>；换架构就是换一套拼名字的规则（呼应 L13 自描述）。
10. `key`（✅）：`llm_arch` 由 `general.architecture` 选定（`"llama"`…）；`llama_hparams` 给规格（`n_layer()`/`n_head(il)` 是<strong>方法</strong>、`n_vocab` 来自 vocab）；`LLM_TENSOR_NAMES`+`LLM_TN` 定张量名（`token_embd`/`blk.N.attn_q`/`output_norm`）；三者把"张量清单"变成"具体模型"。
11. `spark`（💡）：把"模型长什么样"编码成<strong>三张表</strong>（架构名 / KV 键模板 / 张量名模板），于是一套引擎代码能读懂几十种架构——加新模型多半是"填表 + 写一份建图"，引擎主干纹丝不动。这正是 L05/L12 那条"结构不变、可换的部分集中在表里"的思路，在架构层的again。

必须讲到：`llm_arch` 与 `general.architecture` 的关系；`llama_hparams` 关键字段且 `n_layer()/n_head(il)` 是方法、`n_vocab` 不在其中；`LLM_TENSOR_NAMES`/`LLM_TN` 命名约定与 loader 的衔接。

- [ ] **Step 5: quiz（15）**：
- MCQ1 "GGUF 里哪个 KV 决定按哪套架构建图？" -> 正确："`general.architecture`（如 `\"llama\"`、`\"qwen2\"`）"；干扰：`general.name` / `general.file_type` / `version`。
- MCQ2 "为什么 hparams 把头数写成 `n_head(il)` 带层号？" -> 正确："不同层的头数/注意力类型可能不同（GQA、滑窗），按层取最通用"；干扰：写错了 / 为了更快 / 随机。
- MCQ3 "加载器怎么把文件里的张量对应到模型结构？" -> 正确："靠 `LLM_TENSOR_NAMES` 的命名约定（`token_embd`/`blk.N.attn_q`…）按名字在 weights_map 里查"；干扰：按文件顺序 / 按张量大小 / 随机。
- OPEN "`llm_arch`、`llama_hparams`、`LLM_TENSOR_NAMES` 三者各管什么？它们怎么合起来把'一堆张量'变成'一个具体可建图的模型'？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 15 课 · 4 个部分"；commit `feat: add lesson 15 architecture and hyperparameters (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 3: 课 16「构建计算图 / Building the compute graph」

> 第四部分第 3 课。L14 加载了张量、L15 给了架构与超参；这一课把它们<strong>接成一张 transformer 前向计算图</strong>——`llama-graph` 的 `build_*` 积木如何把权重张量串成 attention+FFN 的 `ggml_cgraph`（衔接 L09 惰性建图、L11 算子、L04 注意力数学）。

**Files:** `src/part4.py`（追加 `LESSON_16`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `16-build-graph.html`。

- [ ] **Step 1-3: 登记**
```python
("16-build-graph.html", "构建计算图", "Building the compute graph",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"16-build-graph.html": ("llm_graph_context · build_attn/build_ffn · src/models/<arch> · ggml_cgraph",
                        "llm_graph_context; build_attn/build_ffn; src/models/<arch>; ggml_cgraph"),
"16-build-graph.html": part4.LESSON_16,
```

- [ ] **Step 4: 执笔 LESSON_16（双语）**。**结构**（衔接 L09/L11/L04，勿重复其内容，按引用）：
1. `lead`：有了张量（L14）和架构超参（L15），这一课把它们<strong>接成一张前向图</strong>：`llama-graph` 提供 `build_norm`/`build_attn`/`build_ffn` 这些标准件，每个架构在 `src/models/<arch>.cpp` 里决定"按什么顺序拼"，产出的是一张 ggml 计算图（L09）。
2. `analogy`（🔌）：建图像照着图纸（L15 架构）用<strong>标准件搭模型</strong>：`build_attn`/`build_ffn`/`build_norm` 是标准件；`src/models/<arch>.cpp` 是"这种楼的拼装说明书"；拼出来的不是结果，而是一张<strong>待执行的图</strong>（L09 先建后算）。
3. `<h2>` 谁来建图 + **图1【建图派发链】**（`flow`）：`llama_model::build_graph(params)` -> 虚函数 `build_arch_graph(...)`（在 `src/models/<arch>.cpp`）-> 用 `llm_graph_context` 的 `build_*` 积木 -> `llm_graph_result.get_gf()` = `ggml_cgraph *`。**代码1（简化自 `src/llama-model.cpp` / `src/models/llama.cpp`）**：
```
ggml_cgraph * llama_model::build_graph(const llm_graph_params & p) {
    auto res = build_arch_graph(p);   // 虚: 派发到 src/models/<arch>.cpp
    // ... build_pooling / build_dense_out ...
    return res->get_gf();             // ggml_cgraph(L09)
}
```
4. `<h2>` 一层 transformer 怎么搭 + **图2【一个 block 的图】**（`vflow`）：`build_norm` -> `build_attn`（Q/K/V 投影 + rope + 读写 KV cache + soft_max）-> 残差 -> `build_norm` -> `build_ffn` -> 残差。**代码2（伪代码，一层的拼法）**：
```
cur = build_norm(inpL, attn_norm_w)          # RMSNorm(L11)
cur = build_attn(cur, wq, wk, wv, wo, kv)    # Q/K/V mul_mat + rope + KV + softmax(L11/L04/L19)
inpL = cur + inpL                            # 残差
cur = build_norm(inpL, ffn_norm_w)
cur = build_ffn(cur, w_gate, w_up, w_down)   # 前馈(L11 mul_mat)
inpL = cur + inpL                            # 残差 -> 下一层
```
5. `<h2>` 复用积木与图输入 + **图3【积木 + 输入】**（`layers` 或 `cols`）：积木 `build_attn`/`build_ffn`/`build_norm` 是 `llm_graph_context` 的方法；图输入 `llm_graph_input_*`（`_embd`/`_pos`/`_attn_kv`）把"外部数据"接进图。正文：这些积木产出的都是 ggml 张量（L11 算子的 op/src），串起来正是 L09 那张图。
6. 正文：把<strong>建图与执行的分离</strong>收个尾——`build_*` 只"声明"算子（填 op/src、不计算，L09 惰性），最后 `get_gf()` 拿到 `ggml_cgraph`，交给 L10 的后端执行。每个架构的差异，只浓缩在 `src/models/<arch>.cpp` 的"拼法"里。
7. **深挖1**"为什么每个架构单独一个 `src/models/<arch>.cpp`？"：差异隔离——共享 `llm_graph_context` 的积木，加一个架构就是加一个文件、实现 `build_arch_graph`，<strong>不动别人</strong>。
8. **深挖2**"`build_attn` 内部到底做了什么？"：把 L04 的注意力数学翻译成 L11 的算子串——Q/K/V 各一次 `mul_mat`、`rope` 注入位置、把 K/V 写进 KV cache 并读回（L19）、`soft_max_ext` 加因果掩码、再 `mul_mat` 汇总 V、输出投影。一个 `build_attn` 调用 = 一整套注意力子图。
9. **深挖3**"建图和执行是怎么彻底分开的？"：`build_graph` 只建不算（L09 惰性）；`get_gf()` 交出 `ggml_cgraph`；L10 的 `ggml_backend_*` 才真正执行。正因如此，<strong>同一张图能换后端跑</strong>（CPU/CUDA/Metal），上层逻辑只写一遍。
10. `key`（✅）：`llama_model::build_graph` 派发到 `src/models/<arch>.cpp` 的 `build_arch_graph`；用 `llm_graph_context` 的 `build_norm`/`build_attn`/`build_ffn` 把权重串成一张 `ggml_cgraph`（经 `llm_graph_result.get_gf()`）；<strong>只建不算</strong>（L09），交 L10 执行。
11. `spark`（💡）：把"每种架构怎么前向"写成一份 `src/models/<arch>.cpp`，把"怎么算注意力/FFN"沉淀成 `llm_graph_context` 的可复用积木——于是新架构只是"用标准件换个拼法"，而底层 ggml（L08-L12）<strong>根本不知道</strong>上面跑的是 llama 还是 qwen，照样执行（L10）。这就是这套分层最漂亮的地方。

必须讲到：`build_graph` 的派发链与 `src/models/<arch>.cpp`；`llm_graph_context` 的 `build_attn`/`build_ffn`/`build_norm` 积木；一层的拼法；产出 `ggml_cgraph`、只建不算交 L10。

- [ ] **Step 5: quiz（16）**：
- MCQ1 "llama 层怎么为不同架构建出不同的前向图？" -> 正确："`llama_model::build_graph` 派发到每个架构自己的 `src/models/<arch>.cpp` 的 `build_arch_graph`，复用 `llm_graph_context` 的 `build_*` 积木"；干扰：一个巨型 if-else / 每个架构一个独立引擎 / 运行时编译模型。
- MCQ2 "`build_graph` 产出什么、交给谁？" -> 正确："一张 `ggml_cgraph`（只建不算），交给后端执行（L10）"；干扰：直接产出文本 / 立即算出结果 / 一个 .gguf 文件。
- MCQ3 "一层 transformer 在图里大致是什么顺序？" -> 正确："norm -> attn（QKV+rope+KV+softmax）-> 残差 -> norm -> ffn -> 残差"；干扰：只有一个 mul_mat / 完全随机 / 先 ffn 后 attn 且无 norm。
- OPEN "`build_attn` 这种积木被复用、加新架构只写一份 `src/models/<arch>.cpp`——这种结构对'支持很多模型'有什么好处？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 16 课 · 4 个部分"；commit `feat: add lesson 16 building the compute graph (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 4: 课 17「上下文与会话 / Context & session」

> 第四部分第 4 课。能建图的模型（L14-16）还需要一个<strong>运行时</strong>来真正跑、管状态、出 logits——这就是 `llama_context`：持有 cparams、KV cache（memory）、后端调度器（sched）、输出缓冲；`llama_decode` 跑一步、`llama_get_logits_ith` 取结果。接 L16、L10、L19。

**Files:** `src/part4.py`（追加 `LESSON_17`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `17-context-session.html`。

- [ ] **Step 1-3: 登记**
```python
("17-context-session.html", "上下文与会话", "Context & session",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"17-context-session.html": ("llama_context · cparams · memory/sched/logits · llama_decode",
                            "llama_context; cparams; memory/sched/logits; llama_decode"),
"17-context-session.html": part4.LESSON_17,
```

- [ ] **Step 4: 执笔 LESSON_17（双语）**。**结构**（4 个 `<h2>`）：
1. `lead`：模型能建图了，但谁来真正"跑"它、记住对话进度、把结果交出来？是 `llama_context`——一个<strong>有状态的运行时</strong>，持有这次会话的配置、KV cache、后端调度器和输出缓冲。
2. `analogy`（🔌）：如果 `llama_model` 是<strong>图纸 + 零件库</strong>（静态、只读、可共享），`llama_context` 就是<strong>施工现场</strong>（有状态、每个会话一个）：现场放着这次施工的进度（KV cache）、工具调度（sched）、产出（logits）。同一份图纸能同时开多个工地——一个 model 配多个 context。
3. `<h2>` model vs context + **图1【只读权重 vs 有状态会话】**（`cols`）：model（权重 · 只读 · 一份 · 可被多 context 共享）对照 context（cparams + KV + sched + logits · 有状态 · 每会话一个）。正文点明这是 llama.cpp 多会话/多并发的根基。
4. `<h2>` context 里有什么 + **图2【context 的组成】**（`layers`）：`llama_context` 持有 `llama_cparams cparams` / `llama_memory_ptr memory`（KV cache，L19）/ `ggml_backend_sched_ptr sched`（L10）/ 输出缓冲 `buffer_view<float> logits` 与 `embd`。**代码1（简化自 `src/llama-context.h`）**：
```
struct llama_context {
    llama_cparams        cparams;   // 这次会话的配置
    llama_memory_ptr     memory;    // KV cache 等(L19)
    ggml_backend_sched_ptr sched;   // 多后端调度(L10)
    buffer_view<float>   logits;    // 输出: 下一 token 的分数
};
```
5. `<h2>` cparams 配置 + **图3【常用 cparams】**（`<table class="t">`）：`n_ctx`(上下文长度)、`n_batch`/`n_ubatch`(逻辑/物理批大小)、`n_seq_max`(并行序列数)、`n_threads`、`type_k`/`type_v`(KV 的量化类型)、`offload_kqv`、`pooling_type`。**代码2（简化自 `include/llama.h` `llama_context_params`）**。正文：这些参数怎么权衡显存与速度（`n_ctx` 越大 KV 越占内存——衔接 L19）。
6. `<h2>` 一步推理与取结果（并入上一节或单独，控制总 h2=4-5）：`llama_decode(ctx, batch)` 内部 = 切 ubatch（L18）-> `build_graph`（L16）-> `sched` 执行（L10）-> 更新 KV（L19）-> logits 写入缓冲；`llama_get_logits_ith(ctx, i)` 取第 i 个 token 的 logits（`n_vocab` 维）。**伪代码** 串一遍这条链。
7. **深挖1**"为什么 model 和 context 要分开？"：model 只读、可被多个 context <strong>共享同一份权重</strong>（省内存，配合 L13 的 mmap）；context 有状态、每会话独立。一份几 GB 权重 + 每会话一份轻量 KV，是多并发的关键。
8. **深挖2**"logits 是什么？为什么只在某些 token 上有？"：logits = 下一个 token 的<strong>未归一分数</strong>（`n_vocab` 维，喂给 L21 采样）。只在"需要预测下一个"的位置才算——prefill 阶段往往只要最后一个 token 的 logits，由 batch 的 `logits/output` 标志控制（L18），省掉无用计算。
9. **深挖3**"context 怎么把 L16/L10/L19 串成一步？"：`llama_decode` 是总指挥——用 `llama_batch_allocr` 把 batch 切成 ubatch（L18），调 `build_graph`（L16）搭这一步的图，交 `sched` 执行（L10），把新 token 的 K/V 写进 KV cache（L19），最后把 logits 放进输出缓冲供取用。
10. `key`（✅）：`llama_model` 只读可共享；`llama_context` 有状态（cparams + `memory`/KV + `sched` + `logits`）、每会话一个；`llama_decode` 跑一步前向；`llama_get_logits_ith` 取 logits；cparams 调 `n_ctx`/`n_batch`/`type_k` 等权衡显存与速度。
11. `spark`（💡）：把"<strong>不变的知识</strong>"（权重 model）和"<strong>会话的状态</strong>"（KV/进度 context）拆成两个对象——于是一份几 GB 权重能被许多会话共享，每个会话只额外背一份轻量 KV。这正是 `llama-server` 能扛多并发的根基（第五部分细讲）。

必须讲到：model（只读/共享）vs context（有状态/每会话）；context 持有 cparams/memory/sched/logits；cparams 关键字段；`llama_decode` 一步 + `llama_get_logits_ith`。

- [ ] **Step 5: quiz（17）**：
- MCQ1 "`llama_model` 和 `llama_context` 的区别？" -> 正确："model 是只读权重（可被多个 context 共享），context 是有状态运行时（KV/sched/logits，每会话一个）"；干扰：是一回事 / context 存权重 / model 存 KV cache。
- MCQ2 "`llama_decode` 做什么？" -> 正确："跑一步前向（建图 + 执行 + 更新 KV），算出 logits"；干扰：加载模型 / 直接采样出 token / 释放内存。
- MCQ3 "取第 i 个 token 的 logits 用哪个？" -> 正确："`llama_get_logits_ith(ctx, i)`"；干扰：`llama_get_model` / `llama_tokenize` / `llama_free`。
- OPEN "为什么把'权重'（model）和'会话状态'（context）分成两个对象？这对一台机器服务很多用户有什么好处？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 17 课 · 4 个部分"；commit `feat: add lesson 17 context and session (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 5: 课 18「批处理 / Batching」

> 第四部分第 5 课。L17 的 `llama_decode` 吃的是一个 `llama_batch`。这一课讲批处理：一个 batch 怎么装多个 token（各带 pos/seq_id/输出标志）、内部怎么被 `llama_batch_allocr` 切成 `ubatch` 喂给图。接 L17、预告 L19。

**Files:** `src/part4.py`（追加 `LESSON_18`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `18-batching.html`。

- [ ] **Step 1-3: 登记**
```python
("18-batching.html", "批处理", "Batching",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"18-batching.html": ("llama_batch · pos/seq_id/logits 标志 · llama_batch_allocr -> ubatch",
                     "llama_batch; pos/seq_id/logits flags; llama_batch_allocr -> ubatch"),
"18-batching.html": part4.LESSON_18,
```

- [ ] **Step 4: 执笔 LESSON_18（双语）**。**结构**：
1. `lead`：`llama_decode` 一次吃一个 `llama_batch`。这一课拆开它：一个 batch 怎么同时装多个 token、每个 token 怎么带上"第几位（pos）、属于哪条序列（seq_id）、要不要输出（logits 标志）"，以及内部怎么被切成小批喂给计算图。
2. `analogy`（🔌）：batch 像一张<strong>点单</strong>：每个 token 是一道菜，`pos` 是上菜顺序、`seq_id` 是哪一桌（多序列）、`logits` 标志是"这道要不要打包带走（输出）"。厨房（decode）按物理产能（`n_ubatch`）把大单拆成一锅锅小批做（ubatch）。
3. `<h2>` llama_batch 是什么 + **图1【batch 的字段】**（`<table class="t">` 或 `cellgroup`）：`n_tokens`、`token[]`、`pos[]`、`n_seq_id[]`/`seq_id[][]`、`logits[]`（`int8` 输出标志）。**代码1（简化自 `include/llama.h`）**：
```
struct llama_batch {
    int32_t       n_tokens;
    llama_token * token;     // 词 id
    llama_pos   * pos;       // 每个 token 的位置
    int32_t     * n_seq_id;  llama_seq_id ** seq_id;  // 属于哪条/哪些序列
    int8_t      * logits;    // 标志: 这个 token 是否输出 logits(源码注释: rename to "output")
};
```
4. `<h2>` 输出标志：不是每个 token 都出 logits + **图2【只有标记位输出】**（`cellgroup`）：一行 token，只有 `logits=1` 的位置才算输出投影（prefill 整段 prompt 往往只最后一个；多序列各自最后一个）。衔接 L17（logits 缓冲）。
5. `<h2>` 切成 ubatch + **图3【batch -> ubatch】**（`flow`）：`llama_batch_allocr.init(batch)` 净化/填默认 -> `split_simple`/`split_equal`/`split_seq`(`n_ubatch`) -> 多个 `llama_ubatch` -> 逐个送进图执行。**代码2（伪代码）**，⚠ 明确用 `llama_batch_allocr`（旧 `llama_sbatch` 已移除）：
```
alloc = llama_batch_allocr()
alloc.init(batch)                       # 校验 pos/seq_id, 填默认
for ub in alloc.split_simple(n_ubatch): # 按物理批大小切
    decode_ubatch(ub)                   # 建图(L16)+执行(L10)
```
6. 正文：辨清 `n_batch`（逻辑上一次提交多少 token）与 `n_ubatch`（物理上一次真正算多少）；便捷构造 `llama_batch_get_one`（把一串 token 包成最简 batch）。
7. **深挖1**"`pos` 和 `seq_id` 各有什么用？"：`pos` 是位置，喂给 rope（L16）和 KV cache 的 cell（L19）；`seq_id` 标明属于哪条序列——一个 batch/context 可<strong>并行多条序列</strong>，它们共享权重、各自有 KV。
8. **深挖2**"为什么要 `ubatch` 这层切分？"：硬件一次能高效计算的<strong>物理批大小有限</strong>（`n_ubatch`）；把一个大 batch 切成若干 ubatch 逐个跑，兼顾吞吐与显存。逻辑提交（n_batch）与物理执行（n_ubatch）由此解耦。
9. **深挖3**"输出标志怎么省算力？"：decode 阶段每步只新增 1 个 token、只它要 logits；prefill 整段 prompt 也只要最后一个的 logits。标志让引擎<strong>跳过其余位置的输出投影</strong>（一次 `n_vocab` 维大矩阵乘），呼应 L03/L04 的 prefill/decode。
10. `key`（✅）：`llama_batch` 装多 token（`token`/`pos`/`seq_id`/`logits` 标志）；`logits` 标志选谁输出（省算力）；`llama_batch_allocr` 把 batch 切成 `ubatch`（按 `n_ubatch`）喂图；`n_batch`（逻辑）vs `n_ubatch`（物理）。
11. `spark`（💡）：把"<strong>喂什么</strong>"（batch：哪些 token、哪条序列、谁要输出）和"<strong>怎么分块算</strong>"（ubatch 切分）解耦——于是同一套 `decode` 既能跑单条对话的逐字 decode、也能多序列并行、还能 prefill 整段 prompt，全靠一个统一的 batch 接口描述意图。

必须讲到：`llama_batch` 字段；`logits` 输出标志的意义与省算力；`llama_batch_allocr` 切 `ubatch`；`n_batch` vs `n_ubatch`；`pos`/`seq_id` 用途。

- [ ] **Step 5: quiz（18）**：
- MCQ1 "`llama_batch` 的 `logits` 字段是干嘛的？" -> 正确："一个 per-token 标志，标记哪些 token 需要算输出 logits"；干扰：存放算好的 logits / 存权重 / 采样温度。
- MCQ2 "为什么要把 batch 切成 ubatch？" -> 正确："硬件一次能高效处理的物理批大小有限（`n_ubatch`），大批切成小批逐个算"；干扰：为了加密 / 为多线程随机切 / 没有意义。
- MCQ3 "batch 里的 `seq_id` 表示什么？" -> 正确："这个 token 属于哪条序列（支持多序列并行）"；干扰：token 的 id / token 的位置 / 输出标志。
- OPEN "结合 L03 的 prefill/decode，说说 batch 的 `logits` 标志怎么帮引擎省掉不必要的计算。"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 18 课 · 4 个部分"；commit `feat: add lesson 18 batching (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 6: 课 19「KV cache / The KV cache」

> 第四部分第 6 课（M4a 收尾）。自回归每步只新算 1 个 token，全靠缓存先前 token 的 K/V——这就是 KV cache（L03/L04 反复提到）。这一课钻进 `llama-kv-cache`：cell 管理、上下文移位、多序列、以及变体。接 L04、L18、L17（type_k/type_v）。

**Files:** `src/part4.py`（追加 `LESSON_19`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `19-kv-cache.html`。

- [ ] **Step 1-3: 登记**
```python
("19-kv-cache.html", "KV cache", "The KV cache",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"19-kv-cache.html": ("llama_kv_cache · cell(pos/seq_id) · 上下文移位 · 多序列 · 变体",
                     "llama_kv_cache; cell(pos/seq_id); context shift; multi-seq; variants"),
"19-kv-cache.html": part4.LESSON_19,
```

- [ ] **Step 4: 执笔 LESSON_19（双语）**。**结构**（4-5 个 `<h2>`）：
1. `lead`：自回归每步只新算 1 个 token，全靠把先前 token 的 K/V <strong>缓存</strong>起来（L04 证明过这与重算整段精确等价）。这一课钻进 `llama-kv-cache`：cell 怎么管、上下文满了怎么移位、多序列怎么共存、以及为长上下文准备的变体。
2. `analogy`（🔌）：KV cache 像一本<strong>会议纪要</strong>：每来一个发言（token），把要点（K/V）记在新的一行（cell），标上第几句（pos）、谁说的（seq_id）。下次接话只需读纪要，不必把整场会重听一遍；会议太长就滚动旧页（滑窗）或划掉某人的发言（`seq_rm`）。
3. `<h2>` 为什么需要 KV cache + **图1【重算 vs 缓存】**（`cellgroup` 或 `timeline`）：没缓存 = 每步重算整段（随长度平方增长）；有缓存 = 每步只算新 token、读历史 K/V（线性）。衔接 L04（精确等价）、L03（decode 快）。
4. `<h2>` cell 怎么管 + **图2【cells: pos + seq_id】**（`cellgroup`）：`llama_kv_cells` 每个 cell 存 `pos` + 所属 `seq_id` 集合；`head()` 是滚动写入指针；`size()` ≈ `n_ctx`。**代码1（简化自 `src/llama-kv-cache.h` / `src/llama-kv-cells.h`）**，⚠ 用 `llama_kv_cache`（非 `_unified`）、cells 在 `llama-kv-cells.h`：
```
class llama_kv_cells {              // src/llama-kv-cells.h
    std::vector<llama_pos> pos;     // 每个 cell 的位置
    // 每个 cell 还记: 属于哪些 seq_id
};
class llama_kv_cache : llama_memory_i {  // src/llama-kv-cache.h
    llama_kv_cells_vec v_cells;     // 实际存储(可多序列)
    // head(): 滚动写指针
};
```
5. `<h2>` 序列操作与上下文移位 + **图3【seq 操作】**（`flow` 或 `timeline`）：`seq_rm`/`seq_cp`/`seq_keep`/`seq_add`(移位)/`seq_div`。**代码2（伪代码）**，⚠ 公开 C API 是 `llama_memory_seq_*`（经 `llama_get_memory`），旧 `llama_kv_self_*` 已移除：
```
mem = llama_get_memory(ctx)
llama_memory_seq_rm (mem, seq, p0, p1)   # 删一段 KV
llama_memory_seq_add(mem, seq, p0, p1, d) # 上下文移位: pos 平移
```
   正文：<strong>上下文移位</strong> = 上下文满了，丢掉最旧一段、把剩下的 `pos` 往前挪、腾地方继续生成（不是重算）。
6. `<h2>` 多序列与变体（可并入上一节，控制 h2 数）+ **图4/表【变体】**（`<table class="t">`）：一个 cache 靠每 cell 的 seq 集合支持<strong>多序列共享</strong>；变体应对长上下文——`llama_kv_cache_iswa`（滑窗）、`llama_memory_recurrent`（RNN 类，固定状态）、`llama_memory_hybrid`（混合），均实现基接口 `llama_memory_i`。
7. **深挖1**"KV cache 为什么这么吃显存？"：每个 token、每一层、每个 KV 头都要存一份 K 和 V；`n_ctx` 越大越占。所以 `type_k`/`type_v` 可把 KV <strong>量化</strong>存（L17 cparams），`n_ctx` 要按需权衡。
8. **深挖2**"上下文移位（context shift）到底是什么？"：当生成长度超过 `n_ctx`，引擎丢掉最旧的一段 KV、把保留部分的 `pos` 整体前移（`seq_add`），腾出尾部空间继续生成——避免"满了就停"，也不必重算整段。
9. **深挖3**"为什么有这么多变体？"：标准全注意力 KV 随长度线性增长仍然很大；滑窗（iSWA）只保留最近一窗、recurrent 用固定大小状态不随长度涨、hybrid 混搭——各为不同架构与长上下文场景做取舍。它们都实现 `llama_memory_i`，可整体替换。
10. `key`（✅）：KV cache 缓存历史 K/V，让自回归每步只算新 token（线性而非平方）；cell 存 `pos`+`seq_id`（`llama_kv_cells`）、`head` 滚动；`seq_rm`/`seq_add` 等管序列与移位；公开 API `llama_memory_seq_*`；变体 iswa/recurrent/hybrid 应对长上下文。
11. `spark`（💡）：把"算过的别再算"做成一块带 cell 管理的缓存——自回归从每步重算降到增量更新，这是大模型能逐字流式输出的根本。而把它抽象成 `llama_memory_i` 接口，又让"怎么记忆"（滑窗/recurrent/hybrid）能整体替换、引擎其余部分不动——正是 L12 `type_traits` 那种解耦思路，在"记忆"层的回响。M4a（加载与运行时核心）到此结束。

必须讲到：KV cache 解决的问题（线性 vs 平方）；`llama_kv_cells` 的 cell（pos/seq_id）+ head；序列操作与上下文移位；公开 API `llama_memory_seq_*`（非 `llama_kv_self_*`）；变体及 `llama_memory_i`。

- [ ] **Step 5: quiz（19）**：
- MCQ1 "KV cache 解决什么问题？" -> 正确："缓存先前 token 的 K/V，让自回归每步只算新 token（不重算整段）"；干扰：压缩权重 / 缓存 logits / 加速模型加载。
- MCQ2 "一个 cell 主要记录什么？" -> 正确："这个位置的 `pos`、属于哪些 `seq_id`（以及对应的 K/V）"；干扰：权重 / 文件偏移 / 采样概率。
- MCQ3 "公开 C API 里删除某序列的 KV 用哪个？" -> 正确："`llama_memory_seq_rm`（经 `llama_get_memory`）"；干扰：`llama_kv_self_rm`（已改名移除）/ `llama_free` / `llama_decode`。
- OPEN "KV cache 很吃显存。结合 L17 的 `type_k`/`type_v` 和本课的滑窗变体，说说有哪些办法控制 KV 的内存。"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 19 课 · 4 个部分"；commit `feat: add lesson 19 KV cache (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 7: M4a 验收（清重建 + 密度/CJK 审计 + 里程碑）

> 无新增内容；端到端验证第四部分（上）6 课全部达标、与前三部分衔接无回归。

**Files:** 无新增（仅可能补一处副标题/交叉引用的一致性微调）。

- [ ] **Step 1: 清重建 + 双校验（产物零漂移）**
```bash
cd /home/verden/course/llama-cpp-visual-guide
rm -f index.html lessons/*.html
cd src && python3 build.py && python3 check_html.py && python3 check_links.py
cd .. && git status --short        # 期望干净
```
Expected：0 error / 0 warning、全链接解析、`git status` 干净。

- [ ] **Step 2: 6 课密度 / CJK / 图 / 片段 / 深挖 审计**
```bash
cd /home/verden/course/llama-cpp-visual-guide && python3 - <<'PY'
import re, sys; sys.path.insert(0, "src"); import registry
DIAG = ("layers","vflow","flow","cols","cellgroup","timeline")
for f in ["14-model-loading.html","15-architecture-hparams.html","16-build-graph.html",
          "17-context-session.html","18-batching.html","19-kv-cache.html"]:
    z = registry.CONTENT[f]["zh"]; e = registry.CONTENT[f]["en"]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", z)); encjk = len(re.findall(r"[\u4e00-\u9fff]", e))
    diag = max(sum(z.count(f'class="{c}"') for c in DIAG)+z.count('<table class="t"'),
               sum(e.count(f'class="{c}"') for c in DIAG)+e.count('<table class="t"'))
    acc = z.count('class="accordion"'); pre = z.count("<pre")
    zs=re.split(r'<h2',z); es=re.split(r'<h2',e)
    par = len(zs)==len(es) and all((a.count('<p>')+a.count('<p '))==(b.count('<p>')+b.count('<p ')) for a,b in zip(zs,es))
    ok = cjk>=4000 and encjk==0 and diag>=3 and acc>=2 and pre>=2 and par
    print(f"{f:30s} CJK={cjk:5d} enCJK={encjk} diag>={diag} acc={acc} pre={pre} par={'Y' if par else 'N'}  {'OK' if ok else 'CHECK'}")
PY
```
Expected：每课 `CJK>=4000`、`enCJK=0`、`diag>=3`、`acc>=2`、`pre>=2`、`par=Y`。任一 `CHECK` 回对应 Task 补足。

- [ ] **Step 3: 导航与交叉引用检查**
```bash
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 19 课 · 4 个部分' index.html && echo "index: 19 课 · 4 部分"
for n in 14 15 16 17 18 19; do grep -ql "href=\"lessons/$n-" index.html && echo "toc links $n"; done
grep -RoE '第 [0-9]+ 课' lessons/ | awk -F'第 | 课' '{if($2+0>40) print "OUT:", $0}'   # 期望无输出
```
Expected：index 显示 `19 课 · 4 个部分`、14-19 均在目录、无越界 `第 N 课` 引用、prev/next 链完整（13->14 跨部分边界正常）。

- [ ] **Step 4: 标记里程碑 + 合并**

把 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 的 M4 行/复选标注为"M4a 完成"（M4b 仍待写），commit（`docs: mark M4a (Part 4a load & runtime core) complete` + `Assisted-by: GitHub Copilot`）。随后按 finishing-a-development-branch 把 `build/m4a-part4` 合并回 master（`--no-ff`）并删分支。

---

## Self-Review（计划自检，作者本人执行）

**1. Spec 覆盖**：第四部分（上）6 课（14 模型加载 / 15 架构超参 / 16 建图 / 17 上下文 / 18 批处理 / 19 KV cache）= Spec §"第四部分"前 6 条，逐一对应 Task 1-6；加强标准（3-5 图含概念图、2-3 片段、2-3 深挖、~4000+ CJK、en 逐段对齐无 CJK）写进"统一交付标准"并由 Task 7 Step 2 脚本量化校验。M4b（20-24 vocab/sampler/chat/grammar/LoRA）另起一份计划。✅

**2. Placeholder 扫描**：每个 Task 的 Step 4 给了逐卡片/逐图/逐片段/逐深挖的具体简报与真实源码片段（`weights_map`/`llm_arch`/`llm_graph_context`/`llama_context`/`llama_batch`/`llama_kv_cells` 等实体），Step 5 给完整 quiz，Step 6-7 给精确命令与期望。无 TBD/TODO/"类似上文"占位。✅

**3. 类型/名称一致性**：六课文件名（`14-model-loading`/`15-architecture-hparams`/`16-build-graph`/`17-context-session`/`18-batching`/`19-kv-cache`）在 PAGES/SUBTITLES/registry/quizzes/校验命令中一致；part 标签统一 `第四部分 · llama 推理内部` / `Part 4 · Inside llama inference`；`part4.py` 由 Task 1 创建、Task 2-6 追加；registry 的 `import part4` 仅 Task 1 引入。index 部分数 3 -> 4。✅

**4. 防重复**：L16 vs L09/L11（L09 惰性建图、L11 算子怎么算；L16 只讲"怎么把权重按架构拼成图、谁来建"，按引用不重述）、L19 vs L04（L04 注意力数学/KV 等价；L19 讲 cell 管理/移位/变体的工程实现）、L17 vs L10（L10 后端执行；L17 讲 context 怎么把建图/执行/KV 串成一步并管会话状态）均在 task 标题/导语划清边界。✅

**5. 源码事实**：均来自 2026-06-15 对真实源树的核验（explore 子代理逐条确认并修正了多处陈旧名：`llm_build_context`->`llm_graph_context` + `src/models/*.cpp`、`llama_sbatch`->`llama_batch_allocr`、`llama_kv_cache_unified`->`llama_kv_cache`、`llama_kv_self_*`->`llama_memory_seq_*`、hparams `n_layer/n_head` 为方法、`n_vocab` 不在 hparams、defrag 已移除）。引用一律"文件+符号"。✅
