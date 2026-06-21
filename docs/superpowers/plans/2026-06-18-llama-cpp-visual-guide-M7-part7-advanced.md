# 第七部分（进阶专题，课 34-37）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐 task 执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 给图解教程补齐第七部分「进阶专题」共 4 课（投机解码、MoE、多模态、状态空间模型），每课硬核逐行真实代码 + 1 个 worked-example trace，双语。

**Architecture:** 沿用 M1-M6 的零依赖 Python 静态站点生成器。新建 `src/part7.py`（`LESSON_34..37`），改 `src/registry.py`/`src/shell.py`/`src/quizzes.py` 登记，重建全部 HTML。

**Tech Stack:** Python 3（生成器 + 校验脚本 `check_html.py`/`check_links.py`）；手写 HTML 片段；内联 SVG（Style C）；rsvg-convert + chromium（SVG/页面目检）。

---

## 统一交付标准（每课硬性，照搬 M6）

- **结构**：导语 `<p>` + 卡片（macro/analogy/detail/key/spark 酌用，≥2 张 `<details>` 深挖）+ **≥3 图示**（`cols`/`layers`/`table.t`/`trace`，单语 ≥3，含 1 个 trace）+ **≥2 段真实代码** `<pre class="code">` + **1 个 worked-example trace**。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace` 与内联 `<svg>` 不计入）。
- **中文密度**：zh CJK ≥ 4000；**en CJK == 0**（纯 ASCII：用 `-`/`->`/`...`/`~`/`+/-`/`x`，禁 em-dash/unicode 箭头/`≈`/`±`/`×`/`…`）。
- **无文字墙**：连续顶层 `<p>` ≤ 3。
- **转义**：代码里 `<`/`>`/`&` 写成 `&lt;`/`&gt;`/`&amp;`；无双重转义 `&amp;lt;`。
- **trace**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn/.cellrow/.vc[.hot/.blue]/.op/.tlab`）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>`（zh aria-label 中文、en 纯 ASCII），合法 XML，**深浅 `.trace` 背景都可读**（深色 `#161b22`：深色文字只放白底框内，自由文字用中间色 `#5b6470`/accent/blue/purple），trace 不与 `.card` 紧邻。
- **源码引用**：以"文件 + 符号名"为主、不写行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验 2026-06-18）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测。
- **登记**：`registry.CONTENT`（`import part7` + filename -> `part7.LESSON_NN`）；`shell.PAGES`（filename、zh/en 短标题、`第七部分 · 进阶专题`/`Part 7 · Advanced topics`）；`shell.SUBTITLES`；index 自动变"共 37 课 · 7 个部分"。

## 执行方式（M5/M6 经验）

- superpowers:subagent-driven-development，**一课一个 task**（Task 1=课34 ... Task 4=课37；Task 5 收尾）。
- 每个 task：实现子代理 -> **spec 合规审查子代理 -> 质量审查子代理**（两段审查），修复回环后再标完成。子代理一律用当前主会话模型，显式传 `model`。
- **关键经验**：后台 general-purpose 子代理写整课常中途失败（"completed"却零文件写入）。控制器**每次都要独立核验 git 状态**（不信报告）；若实现子代理失败，则由控制器亲自照模板执笔，仍跑完整 spec+质量双重审查。
- **关键 Style-C SVG（L35 MoE 路由、L37 SSM 扫描）**：先用 Python 预生成并校验（well-formed XML + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 渲染目检），再喂给实现（沿用 M6 L31/L32 做法）。
- **HTML 是被 git 跟踪的**：每课提交须 `git add` 源文件 **+ 重建后的全部 HTML**（`index.html` + `lessons/*.html`），提交后 `git status` 必须干净。
- commit 用 `Assisted-by: GitHub Copilot`（非 Co-authored-by）。分支：在 master 上从本 plan 提交后，新建 `feature/part7-advanced` 分支做实现。

---

## Task 1: 课 34「投机解码 / Speculative decoding」

**Files:** **新建** `src/part7.py`（写 `LESSON_34`，文件头 `"""Content for Part 7 (advanced topics)."""`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `34-speculative-decoding.html`。trace 为 **Style A（无 SVG）**。

**源码事实（核实 2026-06-18，文件+符号、无行号）：**
- `common/speculative.h`：opaque `common_speculative`；`common_speculative_init(common_params_speculative & params, uint32_t n_seq)`、`common_speculative_begin(spec, seq_id, prompt)`、`common_speculative_draft(spec)`（生成草稿）、`common_speculative_accept(spec, seq_id, uint16_t n_accepted)`（告知接受了几个）。`struct common_speculative_draft_params`：`drafting`、`n_max`、`n_past`、`id_last`、`prompt`、`result`（上次 `_draft()` 生成的草稿 token）。
- draft 类型（`speculative.cpp`）：`COMMON_SPECULATIVE_TYPE_NONE` / `_NGRAM_SIMPLE` / `_NGRAM_MAP_K` / `_NGRAM_MAP_K4V` / `_NGRAM_MOD` / `_NGRAM_CACHE`，外加 draft-model（小模型）路径；多实现可链式（一个出草稿后停后续）。
- **draft-model 路径核心**（`speculative.cpp`）：循环 `llama_decode(ctx_dft, batch)` -> 取 `cur_p->data[0]`（最可能 token），当 `cur_p->data[0].p < params.p_min` 或达到 `n_max` 时停（即"草稿模型贪心续写、置信度不够就收手"）。参数 `n_max`/`n_min`/`p_min`。
- **验证/接受**（由调用方完成，如 server/examples）：把草稿的 K 个 token 一次性追加进 target 的 `llama_batch`，**一次 `llama_decode` 并行算出 target 在这 K 个位置的预测** -> 从头逐位比对 -> 接受"匹配前缀"，并在第一个不匹配处白送 1 个（target 自己的预测本就正确）-> 调 `common_speculative_accept(n_accepted)`。
- 为什么快：decode 是**访存密集/带宽瓶颈**（L30/L32），一次 `llama_decode` 验证 K 个 token 和生成 1 个 token 几乎一样快——这就是免费的并行空间。接受率高则加速明显；接受率低（草稿老被拒）反而更慢（白跑 draft）。

- [ ] **Step 1-3: 登记**（三处 + import）

```python
# registry.py 顶部新增: import part7
# registry.py CONTENT 追加:
"34-speculative-decoding.html": part7.LESSON_34,
# shell.py PAGES 追加（第七部分起点）:
("34-speculative-decoding.html", "投机解码", "Speculative decoding", "第七部分 · 进阶专题", "Part 7 · Advanced topics"),
# shell.py SUBTITLES 追加:
"34-speculative-decoding.html": ("draft model / n-gram 提候选 · target 并行验证 · 接受率", "draft model / n-gram propose, target verifies in parallel, acceptance rate"),
```

- [ ] **Step 4: 执笔 `LESSON_34`（双语，新建 `src/part7.py`）。结构：**
  - 导语 `<p>`：自回归一次只出一个 token，慢在"串行 + 访存密集"（呼应 L30 decode memory-bound）。投机解码让一个小模型先"猜"一串，大模型一次性"批改"，把多步串行压成一步并行。
  - `<h2>` 为什么能加速：decode 瓶颈是带宽不是算力（L30/L32），所以"验证 K 个"和"生成 1 个"几乎一样快——免费的并行空间。**图**：串行 decode（N 次）vs 投机（1 次 draft + 1 次 verify）。
    - **trace（Style A·站点流）**："追踪一轮投机"：draft 提 K 个候选 -> target 一次并行验证（K 个位置的预测）-> 从头比对 -> 接受匹配前缀 + 1 bonus -> 不匹配处之后丢弃、下一轮从接受点续。
  - `<h2>` 草稿从哪来：(1) draft model（小模型、同词表）——**真实代码**：`llama_decode(ctx_dft)` 贪心续写、`p_min` 门控、`n_max` 上限；(2) n-gram（从已生成文本里查最近的续接，零额外模型）。
  - `<h2>` 接受、拒绝与接受率：逐位比对决定接受几个；接受率怎么决定加速比；`p_min` 阈值的取舍；接受率低反而变慢。**真实代码**：`common_speculative_accept(n_accepted)` 的语义 + 调用方验证循环骨架。
  - `<h2>` 折叠深挖（≥2）：(1) 为什么能白送 1 个 bonus token（target 在第一个不匹配位置的预测本就是要采的那个）；(2) 投机解码**不改变输出分布**（验证保证等价于直接从 target 采样，附简化说明；草稿只影响速度、不影响结果）。
  - 硬性：zh CJK≥4000、en CJK==0、逐节对齐、≥3 图（含 1 trace）、≥2 深挖、≥2 真实代码片段。

- [ ] **Step 5: quiz（34）** 2-4 题：「投机解码为什么能加速？（decode 带宽瓶颈，验证 K 个≈生成 1 个）」「接受率低时会怎样？（白跑 draft，反而更慢）」「为什么不改变输出质量？（target 验证保证等价于直接采样）」「draft 有哪两种来源？（小 draft model / n-gram）」。

- [ ] **Step 6: 重建+校验**：`cd src && python3 build.py && python3 check_html.py && python3 check_links.py` 全绿；index 变"共 34 课 · 7 个部分"；硬性达标；trace=2（`class="trace"` 的 zh/en）、`<svg`=0（Style A）；grep 渲染无 `&amp;lt;` 双重转义。

- [ ] **Step 7: commit**：`feat: add lesson 34 speculative decoding (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`（暂存 4 源文件 + 重建的全部 HTML，提交后 git status 干净）。

---

## Task 2: 课 35「MoE 专家混合 / Mixture of experts」

**Files:** `src/part7.py`（追加 `LESSON_35`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `35-moe.html`。trace 为 **Style C·SVG**（控制器预生成）。

**源码事实（核实 2026-06-18）：**
- `src/llama-graph.cpp` `llm_graph_context::build_moe_ffn(...)`（参数 `n_expert`/`n_expert_used`/`gating_op`）。核心流程（逐行可读）：
  1. `logits = build_lora_mm(gate_inp, cur)` —— router/gate 线性层，得 `[n_expert, n_tokens]` 打分。
  2. `probs = ggml_soft_max(logits)`（或 `ggml_sigmoid`，由 `gating_op` 决定）—— 每个专家的概率。
  3. `selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used)` —— 选 top-k 专家（如 8 选 2），得 `[n_expert_used, n_tokens]`。
  4. `weights = ggml_get_rows(ctx0, probs, selected_experts)` —— 取选中专家的门控权重；再 `ggml_soft_max`/`ggml_div(ggml_sum_rows)` 归一化、`ggml_scale(w_scale)`。
  5. `ggml_mul_mat_id(ctx0, up_exps, cur, selected_experts)`（及 `gate_exps`/`down_exps`）—— **只对选中的专家做矩阵乘**（`ids = selected_experts` 指明每 token 走哪些专家，间接寻址）。
  6. 按 `weights` 加权合并各专家输出。
- `hparams.n_expert`（专家总数，如 8/128）、`hparams.n_expert_used`（每 token 用几个，如 2/8）。部分模型有 expert group（`n_expert_groups`/`n_group_used`）、shared expert。
- **关键 intuition**：稠密 FFN 每 token 过整层；MoE 把 FFN 拆成 N 个专家、每 token 只走 k 个 -> **总参数大涨、单 token 计算量几乎不变**。代价：专家权重仍要全部驻留显存。

> **控制器预研（实现前完成）**：用 Python 预生成并校验 trace 的 Style-C SVG（"一个 token 的 MoE 路由"：token -> router 打分 -> 8 个专家里 top-2 高亮（带权重 w1/w2）-> 两专家各算 -> 按权重加权求和成输出）；校验合法 XML + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 目检，存到 session files，再逐字喂给实现。

- [ ] **Step 1-3: 登记**

```python
# registry.py CONTENT 追加（import part7 已在 Task 1 加过）:
"35-moe.html": part7.LESSON_35,
# shell.py PAGES:
("35-moe.html", "MoE 专家混合", "Mixture of experts", "第七部分 · 进阶专题", "Part 7 · Advanced topics"),
# shell.py SUBTITLES:
"35-moe.html": ("router 门控 -> top-k 选专家 -> ggml_mul_mat_id 稀疏算 -> 加权合并", "router gating -> top-k experts -> ggml_mul_mat_id sparse compute -> weighted combine"),
```

- [ ] **Step 4: 执笔 `LESSON_35`（双语）。结构：**
  - 导语：稠密 FFN 每个 token 都过整层；MoE 把 FFN 拆成 N 个"专家"，每 token 只走其中 k 个（如 8 选 2）——参数量大涨、单 token 计算量却几乎不变。这是当下大模型（Mixtral/DeepSeek/Qwen-MoE）扩容的主流手段。
  - `<h2>` 路由（router / gating）：一个小线性层给每 token 打分 -> softmax/sigmoid -> top-k 选专家。**真实代码**：`gate -> ggml_soft_max -> ggml_argsort_top_k -> ggml_get_rows(weights) -> 归一化`。
    - **trace（Style C·SVG，控制器预生成）**："追踪一个 token 的路由"：token -> router 打分 -> 8 专家 top-2 高亮（w1/w2）-> 两专家各算 -> 加权求和。
  - `<h2>` 稀疏地算（核心）：`ggml_mul_mat_id(w, cur, ids)` 怎么用 `ids`（= `selected_experts`）只算被选中的专家、而不是算完 8 个扔 6 个（间接寻址、省 6/8 的 FFN 算力）。**真实代码逐行**。**图**：稠密 FFN vs MoE 的算力对比（`cols`/`layers`）。
  - `<h2>` 为什么这么设计：用"激活稀疏"换"参数容量"——总参数大（记得多）、每步算得少（跑得快）；但显存/带宽的代价（所有专家权重都要驻留、按 token 路由导致访存不规整）。呼应 L33 的 offload（MoE 模型尤其吃显存）。
  - `<h2>` 折叠深挖（≥2）：(1) 负载均衡——为什么要让专家被均匀选中（否则一部分专家闲置、容量浪费），训练期的 auxiliary loss / 容量因子（点到）；(2) shared expert 与 expert group（`n_expert_groups`）等变体（点到，指向 `build_moe_ffn` 的对应分支）。
  - 硬性同 Task 1（但含 1 个 `<svg>` 的 zh/en）。

- [ ] **Step 5: quiz（35）** 2-4 题：「MoE 相比稠密 FFN，参数量和单 token 算力各怎么变？（参数大涨、算力几乎不变）」「`ggml_mul_mat_id` 的作用？（按 ids 只算选中的专家）」「8 选 2 里 router 怎么选？（gate 打分 -> softmax -> top-k）」「MoE 省了算力，却在什么上更贵？（显存：所有专家权重都要驻留）」。

- [ ] **Step 6: 重建+校验**（同 Task 1；index "共 35 课 · 7 个部分"；trace=2、`<svg`=2（仅 MoE 路由 trace 的 zh/en）、两 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII；无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 35 mixture of experts (bilingual) with routing SVG trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 3: 课 36「多模态 / Multimodal」

**Files:** `src/part7.py`（追加 `LESSON_36`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `36-multimodal.html`。trace 为 **Style A（无 SVG）**。

**源码事实（核实 2026-06-18）：**
- `tools/mtmd/`（mtmd = multimodal）。管线（`mtmd.h` + `mtmd-helper.h` 文档化）：
  1. `mtmd_init_from_file(mmproj_fname, ...)` —— 加载 projector（mmproj 文件）。
  2. `mtmd_tokenize(...)` —— 把"含 `<image>`/`<audio>` 标记 + bitmap 的输入"切成 `mtmd_input_chunks`，每个 chunk 是 `MTMD_INPUT_CHUNK_TYPE_TEXT` / `_IMAGE` / `_AUDIO`；image chunk 变成 `mtmd_image_tokens`。
  3. `mtmd_encode_chunk(...)` —— 对 image chunk 跑 clip(ViT) + projector，产出视觉 embedding。
  4. `mtmd_get_output_embd(...)` —— 取出这些 embedding。
  5. `llama_decode(...)` —— image embedding 和 text token 一起喂进 LLM（`mtmd_helper_eval_chunks` 把这套自动串起来：text chunk 直接 decode、image chunk 先 encode 再 decode）。
- `tools/mtmd/clip.{h,cpp}`：`clip_image_encode(ctx, n_threads, img, out_vec)`（跑视觉编码器）；`clip_n_output_tokens(ctx, img)`（一张图产出几个 embedding token）；`clip_n_mmproj_embd(ctx)`（projector 输出维度 = LLM 的 embedding 维度）。
- **关键**：LLM 只认 token embedding；多模态就是把"一张图"变成"N 个 embedding"塞进同一序列。**projector（mmproj）是桥**：把 clip 输出的视觉特征投影到 LLM 的 embedding 空间（维度/分布对齐）。M-RoPE 等还要处理图像 embedding 占的"位置"（`mtmd_helper_get_n_pos`）。

- [ ] **Step 1-3: 登记**

```python
# registry.py CONTENT 追加:
"36-multimodal.html": part7.LESSON_36,
# shell.py PAGES:
("36-multimodal.html", "多模态", "Multimodal", "第七部分 · 进阶专题", "Part 7 · Advanced topics"),
# shell.py SUBTITLES:
"36-multimodal.html": ("mtmd 管线: 图像 -> clip(ViT) -> projector(mmproj) -> embedding -> 与文本交织", "mtmd pipeline: image -> clip(ViT) -> projector(mmproj) -> embeddings -> interleave with text"),
```

- [ ] **Step 4: 执笔 `LESSON_36`（双语）。结构：**
  - 导语：LLM 只懂 token embedding；多模态就是把"一张图"也变成"一串 embedding"，塞进同一个序列里和文本一起算。这一课看 ggml 的 `mtmd` 怎么做这件事。
  - `<h2>` 总管线：图像 -> 切 patches -> clip(ViT) 编码成视觉特征 -> projector 投影到 LLM embedding 维度 -> 得到 N 个 image embedding -> 按 `<image>` 占位插进文本序列 -> `llama_decode`。**真实代码**：`mtmd_tokenize` -> `mtmd_encode_chunk` -> `mtmd_get_output_embd` 的管线骨架（含 chunk 类型分支）。
    - **trace（Style A·站点流）**："追踪一张图进 LLM"：一张图 -> patches -> clip 编码 -> projector -> N 个 embedding -> 与文本 token 交织成一个序列 -> `llama_decode`。
  - `<h2>` projector 是关键桥：为什么需要它（clip 视觉特征的维度/分布 != LLM embedding，不投影塞不进去）；`clip_n_mmproj_embd` = LLM embd 维度；`clip_n_output_tokens` = 一张图占几个 token。常见 projector 类型（线性 / MLP / resampler，点到）。**真实代码**：projector/encode 调用（`clip_image_encode`）。
  - `<h2>` 折叠深挖（≥2）：(1) **clip 的 ViT 内部只点不逐行**（范围取舍说明：标准视觉 transformer，指向 `clip.cpp`，告诉读者复杂度来源）；(2) 图像 embedding 怎么占"位置"与进 KV cache（M-RoPE / `mtmd_helper_get_n_pos`，呼应 L16/L19）。
  - **范围取舍落实**：clip ViT 内部当标准视觉 transformer，概念带过。
  - 硬性同 Task 1（Style A，无 `<svg>`）。

- [ ] **Step 5: quiz（36）** 2-4 题：「多模态的核心是把图像变成什么？（一串能塞进序列的 embedding）」「projector(mmproj) 干嘛的？（把视觉特征投影到 LLM 的 embedding 空间）」「mtmd_tokenize 把输入切成什么？（text/image/audio chunk）」「一张图进 LLM 后占的是什么？（N 个 token 位置的 embedding）」。

- [ ] **Step 6: 重建+校验**（同 Task 1，Style A、无 `<svg>`：trace=2、`<svg`=0；index "共 36 课 · 7 个部分"；硬性达标；无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 36 multimodal (bilingual) with pipeline trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 4: 课 37「状态空间模型 Mamba/RWKV / State-space models」

**Files:** `src/part7.py`（追加 `LESSON_37`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `37-state-space.html`。trace 为 **Style C·SVG**（控制器预生成）。

**源码事实（核实 2026-06-18）：**
- `ggml/include/ggml.h`：`ggml_ssm_conv(ctx, sx, c)`（因果 1D 卷积，扫描前的局部混合）；`ggml_ssm_scan(ctx, s, x, dt, A, B, C, ids)`（**选择性扫描**：状态 `s`、输入 `x`、时间步 `dt`(Δ)、状态矩阵 `A`/`B`/`C`、序列 `ids`）—— 这就是 SSM 的核心递推 `h_t = f(A,B,C,Δ; h_{t-1}, x_t)`、`y_t = C·h_t`。
- `src/llama-graph.cpp`：`llm_graph_context::build_rs(...)`（recurrent state：把上一步状态读出、更新、写回）；RWKV 还有 `build_rwkv_token_shift_load/store`。
- `src/llama-memory-recurrent.{h,cpp}`：recurrent state cache（**替代 KV cache**），`state_write`/`state_read`、每序列固定大小状态。`hparams.ssm_d_state`（状态维度）、`ssm_d_conv`（卷积宽度）、`ssm_d_inner`。
- `src/llama-memory-hybrid.cpp`：混合架构（有的模型 SSM 层 + 注意力层混用）。
- **关键对照**：注意力要存全部 KV（显存随序列**线性**涨，L19）、每步看全历史（算力 O(n^2)）；SSM 用一个**固定大小的递推状态** h_t 概括历史 —— 显存 **O(1)**、不随序列长度涨，每步只看上一个状态（算力 O(n)）。代价：状态是"有损压缩"，长程精确检索不如注意力。

> **控制器预研（实现前完成）**：用 Python 预生成并校验 trace 的 Style-C SVG（"SSM 状态沿时间扫描"：输入 x_0,x_1,...,x_t 依次进来，状态 h 原地递推更新 h_0 -> h_1 -> ... -> h_t，每步只依赖上一个 h 和当前 x；可与"注意力的 KV 一路变长"做并排对照）；校验同 M6（合法 XML + 英文 ASCII + 不溢出 viewBox + 深色可读 + `rsvg-convert` 目检），存 session files，再逐字喂给实现。

- [ ] **Step 1-3: 登记**

```python
# registry.py CONTENT 追加:
"37-state-space.html": part7.LESSON_37,
# shell.py PAGES:
("37-state-space.html", "状态空间模型", "State-space models", "第七部分 · 进阶专题", "Part 7 · Advanced topics"),
# shell.py SUBTITLES:
"37-state-space.html": ("Mamba/RWKV: 递推状态替代 KV cache · ggml_ssm_conv/scan · O(1) 显存", "Mamba/RWKV: recurrent state instead of KV cache; ggml_ssm_conv/scan; O(1) memory"),
```

- [ ] **Step 4: 执笔 `LESSON_37`（双语）。结构：**
  - 导语：transformer 的注意力要存全部 KV（显存随序列线性涨，L19）；状态空间模型（Mamba/RWKV）改用一个**固定大小的递推状态** h_t，由 h_{t-1} 和当前输入算出——显存 O(1)、不随序列长度涨。这是注意力之外的另一条技术路线。
  - `<h2>` 递推 vs 注意力：注意力"每步看全历史"（O(n) 显存、O(n^2) 算）；SSM"每步只看上一个状态"（O(1) 显存、O(n) 算）。**图/对照**（`cols`/`layers`）：KV cache 一路变长 vs 状态原地更新。
    - **trace（Style C·SVG，控制器预生成）**："追踪一次状态扫描"：x_0..x_t 依次进，h 原地递推 h_0 -> h_1 -> ... -> h_t，每步只依赖上一个 h 和当前 x。
  - `<h2>` ggml 怎么实现：`ggml_ssm_conv`（因果卷积、局部混合）+ `ggml_ssm_scan(s,x,dt,A,B,C,ids)`（选择性扫描）两个算子；`build_rs` 怎么把状态从 recurrent cache 读出/写回（呼应 L09/L10 计算图 + L17/L19 cache）。**真实代码**：算子调用 + `build_rs` 骨架。
  - `<h2>` selective scan（概念）：`A`/`B`/`C`/`Δ`(dt) 各是什么；"选择性"指这些量**随输入变化**（输入相关的门控，让模型学会"记住/忘记"）。**简化伪代码**（在线扫描 `h = A*h + B*x; y = C*h`），不逐行真实 kernel。
  - `<h2>` 折叠深挖（≥2）：(1) 为什么 SSM 适合长序列、代价是什么（状态是有损压缩，长程精确检索/copy 任务不如注意力）；(2) 混合架构（SSM 层 + 注意力层混用，`llama-memory-hybrid`，取两者之长）。
  - 收尾 `<p>`：第七部分到此收束——从投机解码、MoE、多模态到状态空间模型，四个"标准 transformer 之外"的进阶机制讲完了；下一站第八部分讲实战与贡献（转换模型、编译调试测试、参与贡献）。
  - **范围取舍落实**：selective-scan 内核数学当概念 + 伪代码。
  - 硬性同 Task 1（含 1 个 `<svg>` 的 zh/en）。

- [ ] **Step 5: quiz（37）** 2-4 题：「SSM 相比注意力，显存怎么变？（O(1) 固定状态，不随序列涨）」「SSM 用什么替代 KV cache？（固定大小的递推状态 h_t）」「`ggml_ssm_scan` 算的是什么？（选择性扫描，按 A/B/C/Δ 递推更新状态）」「SSM 的代价是什么？（状态有损压缩，长程精确检索不如注意力）」。

- [ ] **Step 6: 重建+校验**（同 Task 2，含 `<svg>`：index "共 37 课 · 7 个部分"；trace=2、`<svg`=2（仅扫描 trace 的 zh/en）、合法 XML、英文 SVG 纯 ASCII；无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 37 state-space models (bilingual) with scan SVG trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 5: 收尾（roadmap 勾选 + 全量验证 + 完成分支）

**Files:** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（勾 M7）。

- [ ] **Step 1: 更新 roadmap**：里程碑总表 M7 行"状态"`待写` -> `完成`；状态追踪 `- [ ] M7 ...` -> `- [x] M7 ...`。commit `docs: mark M7 (part 7) done`。
- [ ] **Step 2: 全量验证**（master 合并前在分支上跑）：
  - `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` = `Wrote 38 files`、`structural check passed`（0 error/warning）、`all internal links resolve`。
  - index 显示"共 **37** 课 · **7** 个部分"；第七部分 4 课都在导航里。
  - 第七部分 4 课（34-37）：按 `<h2>` 分节 `<p>` 计数中英相等；zh CJK ≥ 4000、en CJK == 0、en 无 unicode（`≈`/`±`/`×`/箭头/em-dash/`…`）；无连续顶层 `<p>` > 3；无 `&amp;lt;`/`&amp;gt;`/`&amp;amp;` 双重转义。
  - 第七部分新增 trace：`class="trace"` 全站净增 8（4 课 × 2 语言）；`<svg` 净增 4（L35 MoE 路由 + L37 SSM 扫描，各 zh/en）；新增 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII、深色色板；用 chromium 目检 4 课渲染正常。
- [ ] **Step 3: 第七部分整体复审**（建议）：派一个 superpowers:code-reviewer 子代理（当前模型）复审 `master..HEAD` 的 4 课跨课一致性（标题/卡片/图/trace 风格统一、真实源码引用准确、双语纪律、范围未越界——只改 part7 + 登记，不碰 1-33 课内容/build 基础设施/其它部分；两处范围取舍——多模态 clip ViT 概念化、Mamba selective-scan 伪代码——已落实；四课形成"标准 transformer 之外的进阶机制"连贯主线、各自回链相关旧课 L19/L30/L32/L33 准确）。
- [ ] **Step 4: 完成分支**：用 superpowers:finishing-a-development-branch，先过验证门，再按用户选择（历史偏好：本地 `--no-ff` 合并 master + 删分支）。

---

## 计划自审（writing-plans self-review）

- **Spec 覆盖**：设计 §每课设计 的 L34/L35/L36/L37 四课 -> Task 1/2/3/4；统一交付标准 -> 各 task Step 4 硬性 + Step 6 校验；roadmap 勾选/全量验证/完成分支 -> Task 5。两个范围取舍（多模态 clip ViT 概念化、Mamba selective-scan 伪代码）在 Task 3/Task 4 的源码事实与结构里均落实。✓ 无遗漏。
- **占位符扫描**：无 TBD/TODO；各 task 的源码事实为真实"文件+符号"（已核实）、登记字符串为可直接粘贴的精确内容、quiz 给了具体题目与答案要点。✓
- **类型/命名一致**：四课统一 `src/part7.py` 的 `LESSON_34..37`、`import part7`、part 标签 `第七部分 · 进阶专题`/`Part 7 · Advanced topics`、文件名 `34-speculative-decoding`/`35-moe`/`36-multimodal`/`37-state-space`。Task 1 新建 part7.py（文件头注释），Task 2-4 追加，命名一致。✓
- **风险点**：(1) 真实代码转义——MoE/SSM 的 ggml 调用、C 代码里 `<`/`&` 须在 r-string 里写成 `&lt;`/`&amp;`，已在统一交付标准 + 各 Step 6 点名；(2) zh CJK 密度——硬核课 `<span class="mono">`/代码占比高，CJK 易偏低，已在交付标准提醒"写实、目标 4200+ 一次过、必要时分轮扩写"；(3) 子代理写整课可能失败——执行方式已写明"独立核验 git + 失败则控制器亲自执笔"；(4) 关键 SVG 预生成——Task 2/4 已写明控制器先 Python 预生成校验、深浅背景都可读。





