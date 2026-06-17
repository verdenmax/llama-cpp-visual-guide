# M5 · 第五部分（公共 API 与工具）· 设计

> 父级设计：`docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`（§第五部分定义了这 6 课）
> 路线图：`docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（M5 = 待写）
> 本文件只记录 M5 这一里程碑相对父级设计的**细化与新决策**；总体架构/双语机制/内容准则沿用父级设计与 M1-M4 已确立的基线。

## 目标

为图解教程补齐**第五部分 · 公共 API 与工具**，共 6 课（第 25-30 课），把视角从"内部原理"（1-4 部分）转到"对外接口 + 工具程序"。每课沿用 M1-M4 基线，并在初稿就内嵌 worked-example trace（流程展示）图。

## 范围（6 课）

每课对应当前 `llama.cpp` 真实源码（已核对路径，引用以"文件 + 符号名"为主、不写死行号；对照真实代码核实）：

| 课 | 主题 | 核心源码（已核实存在） | 内嵌 trace 设想 | 风格 |
| --- | --- | --- | --- | --- |
| 25 | C API 总览 | `include/llama.h` · `include/llama-cpp.h` | 一次典型调用序列：backend_init -> model_load -> new_context -> tokenize -> decode -> sample -> token_to_piece | A |
| 26 | common 工具层 | `common/common.cpp` · `common/arg.cpp` · `common/sampling.*` · `common/download.cpp` · `common/log.*` | 一串命令行参数 -> `common_params` 结构（或采样链按配置装配） | A |
| 27 | llama-cli | `tools/cli` | 生成主循环一轮：读入 -> tokenize -> decode -> sample -> 流式打印 -> 回环 | A |
| 28 | llama-server | `tools/server` | **重点**：连续批处理——3 个请求 -> slot 分配 -> 一个 batch 交错解码各自的下一 token（slot x 时间网格） | C（SVG） |
| 29 | quantize 工具 | `tools/quantize` · `tools/imatrix` | imatrix 加权量化一块：重要性权重 -> 偏向重要权重定 scale -> 量化码 | A 或 C |
| 30 | 评测与基准 | `tools/perplexity` · `tools/llama-bench` | 一小段序列上算 PPL：logits -> 真实下一词 log-prob -> 平均 -> exp | A |

### 关键取舍

- **L28 只做架构总览**（HTTP / OpenAI 兼容端点 / slot 概念 / 连续批处理直觉 / 调度概览）。更深的 slot 调度与 prefill/decode 交错吞吐细节，按父级 spec 留给 **L35（第七部分 · 进阶专题）**，避免与之重复。
- **trace 初稿内嵌**：复用已有 `.trace` 组件（M1-M4 之后补的 worked-example trace 系列建立）。Style A = 纯 HTML 站点流；Style C = 手绘内联 SVG（仅在依赖特殊几何/网格时用，如 L28 的 slot x 时间网格）。每课"合适处"放 1 张；不强求每课都有花哨 SVG。
- **视角差异**：第五部分偏"怎么用 / 工具怎么跑"，所以伪代码更偏**调用方代码**（如何调 API、命令行长什么样、HTTP 请求/响应），真实源码片段标注 `tools/...` / `common/...` / `include/llama.h`。

## 每课交付物（沿用 roadmap §统一交付清单）

1. `src/part5.py` 写 `LESSON_25..30 = {"zh": ..., "en": ...}`：导语 + 教学卡片（macro/detail/analogy/key/spark）+ 结构/流程图 + 伪代码或真实源码简化片段 + 折叠深挖 + 合适处的 worked-example trace。
2. `src/shell.PAGES` 登记该课（filename、zh/en 短标题、`第五部分 · 公共 API 与工具` / `Part 5 · Public API & tools`）。
3. `src/registry.py` 登记 `文件名 -> LESSON_xx`。
4. `src/quizzes.py` 写该课双语自测（2-4 题，沿用既有格式）。
5. `index_page` 副标题表登记 zh/en 副标题。
6. `build.py` + `check_html.py` + `check_links.py` 全绿后 commit。

文件名（延续既有 NN-slug 命名）：`25-c-api.html` · `26-common.html` · `27-llama-cli.html` · `28-llama-server.html` · `29-quantize-tool.html` · `30-eval-bench.html`（具体 slug 在 plan 里定稿）。

## 不变量（每课结束都要过）

- `check_html.py` 0 error / 0 warning（结构 / 导航链 / 计数 / 防漂移 / 中文密度）、`check_links.py` 0 死链。
- 中英两份均存在且可切换；按 `<h2>` 分节的 `<p>` 计数中英严格相等（parity）。
- 中文 CJK >= 4000；英文 CJK == 0（纯 ASCII，代码片段也用 `-`/`->`/`...` 而非 unicode）。
- 无文字墙：连续顶层 `<p>` <= 3。
- trace/SVG：内嵌 SVG 须是合法 XML；英文 SVG 文本纯 ASCII，中文可用 unicode；深色模式可读（沿用 trace 系列的字面色板）；不与 `card` 紧邻（中间须有 `<p>`/`<h2>`）。

## 执行方式

- superpowers:subagent-driven-development，**一课一个 task**（L25 -> L30 顺序）。
- 每个 task 跑完整两段审查：spec 合规子代理 -> 质量子代理（沿用偏好）。子代理一律用当前主会话模型。
- 6 课全绿后，更新 roadmap 勾掉 M5（并在状态表标"完成"）。

## 明确不做（YAGNI）

- 不在 M5 写 PDF / CI / README（那是 M9）。
- 不深挖 server 调度（L35）、不写多模态 mtmd / rpc / tts 等其它 tools（不在父级 spec 的 Part 5 范围内）。
- 不重构既有 1-4 部分或 build/check 基础设施。
