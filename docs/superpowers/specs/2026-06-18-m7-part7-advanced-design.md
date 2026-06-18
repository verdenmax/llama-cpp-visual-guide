# M7 · 第七部分（进阶专题）设计文档

> **配套：** 父级总设计 `2026-06-13-llama-cpp-visual-guide-design.md`（§第七部分）· roadmap `2026-06-13-llama-cpp-visual-guide-roadmap.md`（M7 行）。
> **执笔基线：** 沿用 M1-M6 的零依赖 Python 静态站点生成器与双语机制；本部分新建 `src/part7.py`，写 `LESSON_34..37`。

## Goal

为图解教程补齐**第七部分 · 进阶专题（硬核）**，共 **4 课（34-37）**，讲透 llama.cpp 里四个"前六部分没碰过的新机制/新架构"：投机解码、MoE 专家混合、多模态、状态空间模型。视角从前面的"标准 transformer 推理 + 底层内核"扩展到"非标准架构与加速技巧"。

## 总基调（与用户确认）

- **硬核路线（depth=B，同 M6）**：尽量展开并逐行讲解真实代码（`common/speculative`、`ggml_mul_mat_id`/`build_moe_ffn`、`tools/mtmd`、`ggml_ssm_*`/`build_rs`）；实现分散/复杂处做**最小简化**但保持真实，并标注来源文件+符号。
- **用图把硬核讲透**：每课内嵌 **1 个 worked-example trace**，按主题选 **Style A（站点流，纯 HTML）** 或 **Style C（手绘 SVG）**。
- 沿用 M1-M6 双语基线（见下"统一交付标准"）。
- **与已写内容不重叠**：原父设计 M7 的"连续批处理/性能内存优化"已被 L18/L28/L30/L32/L33 覆盖，故替换为四个全新主题。

## 范围取舍（与用户确认）

1. **多模态（L36）**：`mtmd` 管线 + projector（mmproj）+ image embedding 与文本 token 的合并上**真实代码**；`clip` 的 ViT 编码器内部当作标准视觉 transformer，**只点到、不逐行**（否则一课讲不完一个 ViT）。
2. **状态空间模型（L37）**：`ggml_ssm_conv`/`ggml_ssm_scan` 的**调用**、递推状态、`build_rs` 的 recurrent state cache 上**真实代码**；selective-scan 的内核数学（`A`/`B`/`C`/`Δ` 的逐元素扫描）当**概念 + 简化伪代码**，不逐行抠 kernel。

## 里程碑范围（M7 = 课 34-37）

| 课 | slug（产出文件） | 标题 zh / en | 内嵌 trace（风格） |
| --- | --- | --- | --- |
| 34 | `34-speculative-decoding.html` | 投机解码 / Speculative decoding | draft 提 K 个 → target 并行验证 → 接受前缀（A·站点流） |
| 35 | `35-moe.html` | MoE 专家混合 / Mixture of experts | 一个 token：router → top-k 选专家 → 加权合并（C·SVG） |
| 36 | `36-multimodal.html` | 多模态 / Multimodal | 图像 → patches → clip → projector → embedding → 交织（A·站点流） |
| 37 | `37-state-space.html` | 状态空间模型 / State-space models | 递推状态 h_0 -> h_1 -> ... -> h_t 扫描（C·SVG） |

> 全部在"第七部分 · 进阶专题 / Part 7 · Advanced topics"标签下。

## 每课设计

### 课 34 · 投机解码（`common/speculative.{h,cpp}`）

**源码事实（核实 2026-06-18，以"文件+符号"为准，无行号）：** `common/speculative.h`：`common_speculative`、`common_speculative_init(params, n_seq)`、`common_speculative_begin(spec, seq_id, prompt)`、`common_speculative_draft_params`、`common_params_speculative`（`n_draft`/`p_min` 等）、多种 draft 类型（draft model / n-gram）。核心循环：用小 draft model（或 n-gram）一次提出 K 个候选 token -> 大 target model **一次 `llama_decode` 并行验证这 K 个** -> 从头比对，接受到第一个不匹配处、再额外白送 1 个（target 在该位置的预测）。

**结构：**
- 导语：自回归解码一次只出一个 token，慢在"串行+访存密集"（呼应 L30 的 decode memory-bound）。投机解码用一个小模型先"猜"一串，大模型一次性"批改"，把多步串行压成一步并行。
- `<h2>` 为什么能加速：decode 的瓶颈是带宽不是算力（L30/L32），所以"验证 K 个 token"和"生成 1 个 token"几乎一样快——这就是免费的并行空间。
  - **trace（Style A·站点流）**：draft 提 K 个候选 -> target 一次并行验证 -> 从头比对 -> 接受匹配前缀 + 1 bonus -> 拒绝处之后丢弃、重来。
- `<h2>` draft 从哪来：(1) draft model（一个小模型，和 target 同词表）；(2) n-gram（从已生成文本里查最近出现的续接）。真实代码：`common_speculative_*` 的 draft 生成与参数。
- `<h2>` 接受/拒绝与接受率：逐位置比对、接受率怎么决定加速比；`p_min` 等阈值；接受率低时反而变慢（draft 全被拒）。
- `<h2>` 折叠深挖（≥2）：(1) 为什么接受 1 bonus token（target 在第一个不匹配位置的预测本就是对的）；(2) 投机解码不改变输出分布（数学上等价于直接采样，附简化说明）。

### 课 35 · MoE 专家混合（`src/llama-graph.cpp` `build_moe_ffn` + `ggml_mul_mat_id`）

**源码事实：** `llm_graph_context::build_moe_ffn(...)`（参数 `n_expert`/`n_expert_used`）：先用 router/gate 线性层得到每个 token 对各专家的打分 -> `ggml_soft_max` + top-k 选出 `n_expert_used` 个专家（如 8 选 2）-> 用 `ggml_mul_mat_id(ctx, w, cur, ids)` **只对选中的专家做矩阵乘**（`ids` 指明每个 token 走哪些专家）-> 按 gate 权重加权合并专家输出。`hparams.n_expert`/`n_expert_used`。

**结构：**
- 导语：稠密 FFN 每个 token 都过整层；MoE 把 FFN 拆成 N 个"专家"，每个 token 只走其中 k 个（如 8 选 2）——参数量大涨、单 token 计算量却几乎不变。
- `<h2>` 路由（router/gating）：一个小线性层给每个 token 打分 -> softmax -> top-k 选专家。真实代码（gate -> softmax -> top-k）。**图**：一个 token 的路由打分。
  - **trace（Style C·SVG）**：一个 token 进来 -> router 打分 -> 8 个专家里 top-2 高亮（带权重）-> 两个专家各算 -> 按权重加权求和成输出。
- `<h2>` 稀疏地算：`ggml_mul_mat_id` 怎么只算被选中的专家（`ids` 张量 + 间接寻址），而不是算完 8 个再扔 6 个。真实代码逐行。
- `<h2>` 为什么这么设计：用"激活稀疏"换"参数容量"——总参数大（记得多）、每步算得少（跑得快）；显存与带宽的取舍（专家权重仍要全部驻留）。
- `<h2>` 折叠深挖（≥2）：(1) 负载均衡（为什么要让专家被均匀选中，auxiliary loss / 容量）；(2) shared expert / 不同 MoE 变体（点到为止）。

### 课 36 · 多模态（`tools/mtmd/`）

**源码事实：** `tools/mtmd/`：`clip.cpp`（`clip_image_encode` 跑视觉编码器）、`mtmd.cpp`/`mtmd-helper.{h,cpp}`（管线：`mtmd_tokenize` 把含 `<image>` 标记的输入切成 text/image chunk -> `mtmd_encode_chunk` 对 image chunk 跑 clip+projector -> `mtmd_get_output_embd` 取出 image embedding -> 和 text token 一起 `llama_decode`）。projector（mmproj）把视觉特征投影到语言模型的 embedding 空间。

**结构：**
- 导语：LLM 只懂 token embedding；多模态就是把"一张图"也变成"一串 embedding"，塞进同一个序列里。这一课看 ggml 的 `mtmd` 怎么做这件事。
- `<h2>` 总管线：图像 -> 切 patches -> clip(ViT) 编码成视觉特征 -> projector 投影到 LLM 的 embedding 维度 -> 得到 N 个"image token embedding" -> 按 `<image>` 占位插进文本序列。**真实代码**：`mtmd_tokenize`/`mtmd_encode_chunk`/`mtmd_get_output_embd` 的管线骨架。
  - **trace（Style A·站点流）**：一张图 -> patches -> clip 编码 -> projector -> N 个 embedding -> 与文本 token 交织成一个序列 -> llama_decode。
- `<h2>` projector 是关键桥：为什么需要它（视觉特征维度/分布 != LLM embedding）；常见类型（线性/MLP、resampler 等，点到）。真实代码：projector 前向。
- `<h2>` 折叠深挖（≥2）：(1) clip 的 ViT 内部只点不逐行（范围取舍说明 + 指向 `clip.cpp`）；(2) 图像 embedding 在 KV cache / 位置编码上的处理（占多少"位置"）。
- **范围取舍落实**：clip ViT 内部当标准视觉 transformer，概念带过。

### 课 37 · 状态空间模型 Mamba/RWKV（`ggml_ssm_*` + `build_rs`）

**源码事实：** `ggml/include/ggml.h`：`ggml_ssm_conv(...)`、`ggml_ssm_scan(...)`（选择性扫描算子）。`src/llama-graph.cpp`：`build_rs(...)`/`build_rs_inp()`（recurrent state：把上一步的状态读出、更新、写回）。`src/llama-memory-recurrent.cpp`：recurrent state cache（替代 KV cache）。Mamba 用一个**随序列推进的固定大小状态** h_t 概括历史，而不是像注意力那样保留全部 KV。

**结构：**
- 导语：transformer 的注意力要存全部 KV（显存随序列线性涨，L19）；状态空间模型（Mamba/RWKV）改用一个**固定大小的递推状态**，h_t 由 h_{t-1} 和当前输入算出——显存 O(1)、不随序列长度涨。
- `<h2>` 递推 vs 注意力：注意力是"每步看全历史"（O(n) 显存、O(n^2) 算）；SSM 是"每步只看上一个状态"（O(1) 显存、O(n) 算）。**图/对照**：KV cache 一路变长 vs 状态原地更新。
  - **trace（Style C·SVG）**：序列 x_0,x_1,...,x_t 依次进来，状态 h 原地递推更新 h_0 -> h_1 -> ... -> h_t，每步只依赖上一个 h 和当前 x（画出"状态沿时间扫描"）。
- `<h2>` ggml 怎么实现：`ggml_ssm_conv`（局部因果卷积）+ `ggml_ssm_scan`（选择性扫描）两个算子；`build_rs` 怎么把状态从 recurrent cache 读出/写回（呼应 L09/L10 的计算图 + L17/L19 的 cache）。**真实代码**：算子调用 + build_rs 骨架。
- `<h2>` selective scan（概念）：A/B/C/Δ 是什么、"选择性"指输入相关的门控；简化伪代码（在线扫描），不逐行真实 kernel。
- `<h2>` 折叠深挖（≥2）：(1) 为什么 SSM 适合长序列、又有什么代价（状态是"有损压缩"，长程精确检索不如注意力）；(2) 混合架构（有的模型 SSM 层 + 注意力层混用，`llama-memory-hybrid`）。
- **范围取舍落实**：selective-scan 内核数学当概念 + 伪代码。

## 统一交付标准（每课硬性达标，与 M1-M6 一致）

每课 `LESSON_NN = {"zh": r'''...''', "en": r'''...'''}`，须满足：

- **结构**：导语 `<p>` + 教学卡片（macro/analogy/detail/key/spark 酌用，≥2 张深挖 `<details>`）+ **≥3 个图示**（`cols`/`layers`/`table.t`/`trace` 等，单语 ≥3，含 1 个 trace）+ **≥2 段真实/简化代码片段**（`<pre class="code">`）+ **≥1 个内嵌 worked-example trace**（见各课）。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace` div 与内联 `<svg>` 不计入）。
- **中文密度**：zh CJK ≥ 4000；**en CJK == 0**（纯 ASCII；代码/SVG/图示文本用 `-`/`->`/`...`/`~`/`+/-`，不用 em-dash/unicode 箭头/`≈`/`±`/`×`）。
- **无文字墙**：连续顶层 `<p>` ≤ 3。
- **转义**：渲染 `<`/`>`/`&` 须转义（`&lt;`/`&gt;`/`&amp;`）；无双重转义（`&amp;lt;`）。
- **trace 规范**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn/.cellrow/.vc[.hot/.blue/.dim]/.op/.tlab`）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>`（zh aria-label 中文、en 纯 ASCII），合法 XML，字面色板（白 `#ffffff`/ink `#1d2129`、accent `#c2630e`、blue `#2563eb`、purple `#7c3aed`、label `#5b6470` 等），**深浅两种 `.trace` 背景下都可读**（深色 `#161b22`：深色文字只放在白底框内、自由文字用中间色），trace 不与 `<div class="card">` 紧邻。
- **源码引用**：以"文件 + 符号名"为主、不写死行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验日期 2026-06-18）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测。
- **登记**：`shell.PAGES`（filename、zh/en 短标题、`第七部分 · 进阶专题`/`Part 7 · Advanced topics`）、`shell.SUBTITLES`、`registry.CONTENT`（filename -> `part7.LESSON_NN`）；index 课数 33->37、部分数 6->7（index 自动从 PAGES 推导部分数，新增"第七部分"标签即生效）。

## 与 roadmap 衔接

- 完成后：roadmap M7 行"状态"`待写`->`完成`、状态追踪 `- [ ]`->`- [x]`。
- 执行：superpowers:subagent-driven-development（一课一个 task，顺序执行；收尾 task 勾 roadmap + 全量验证 + 完成分支）。**注意**：M5/M6 经验表明后台 general-purpose 子代理写整课常中途失败（零写入），实现时若失败则由控制器亲自照模板执笔，仍保留完整 spec+质量双重审查。
- 关键 Style-C SVG（L35 MoE 路由、L37 SSM 扫描）建议先用 Python 预生成并校验（well-formed + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 渲染目检），再喂给实现（沿用 M6 做法）。
