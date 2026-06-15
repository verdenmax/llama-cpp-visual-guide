# llama.cpp 图解教程（llama-cpp-visual-guide）· 设计文档（Spec）

> 状态：已与用户确认设计方向，待用户复核本 spec 后进入 writing-plans。
> 日期：2026-06-13
> 仿照项目：`/home/verden/course/langchain-visual-guide`
> 讲解对象仓库：`/home/verden/course/llama.cpp`（`ggml-org/llama.cpp`）

---

## 1. 目标与受众

做一套面向**完全新手**的可视化（HTML 图解）教程，带读者**从零开始、一点点**理解
整个 llama.cpp 项目：既有**宏观全景与整体结构图**，也深入到 **ggml 引擎 / llama 推理内部 / 底层内核**
的源码细节，并讲清楚"**每个东西是什么功能、为什么要这么写**"。

- **受众**：
  - 完全没接触过 llama.cpp / ggml、想从零入门的新手
  - 想先建立宏观认知、再深入内部源码的学习者
  - 准备阅读 / 调试 / 贡献 llama.cpp 源码的开发者
- **读者收获**：从"会用"到"懂原理"的完整路径，以及一份可随时查阅的源码导航地图。

> 注意：本教程是**独立的第三方学习材料**，与 llama.cpp 官方无隶属关系；
> 也**绝不**向 llama.cpp 仓库提交任何内容（该仓库明确禁止 AI 生成内容）。

---

## 2. 已确认的关键决策

| 决策点 | 结论 |
| --- | --- |
| 深度重心 | **全覆盖**：宏观 → 用法 → 内部源码 → 底层内核，做成超大型教程 |
| 项目位置 | 新建独立目录 `/home/verden/course/llama-cpp-visual-guide`，单独 git 仓库 |
| 语言 | **中英双语**，顶部按钮即时切换（zh ⇄ en），`localStorage` 记住选择 |
| 配套功能 | **全保留**：每课自测 quiz + 下载 PDF + GitHub Pages 自动部署 CI |
| 课程组织 | **方案 A·单线递进**：一条从零到深的主线 |
| 内容风格 | **多画图**（结构图/流程图，图加深理解）、**多伪代码 + 源码简化片段**、每部分**尽量详细** |
| 交付方式 | **分多个 milestone** 逐步交付，不必一次写完 |

---

## 3. 架构（复用模板 + 换皮 + 双语）

完全复用 langchain-visual-guide 的成熟架构：**纯 Python 3、零第三方依赖的生成器**，
产出**自包含、可 `file://` 直接打开**的 HTML。

### 3.1 仓库结构

```
llama-cpp-visual-guide/
├── index.html                  ← 目录页（入口），从这里开始
├── lessons/                    ← 生成的课程页（每课一个 HTML，内嵌中/英两份内容）
│   ├── 01-what-is-llamacpp.html
│   └── …  40-glossary.html
├── src/                        ← 纯 Python 无依赖生成器（可重建全部 HTML / PDF）
│   ├── shell.py                设计系统(CSS) + PAGES + page()/index_page() + 语言切换 JS
│   ├── registry.py             文件名 → {"zh":…, "en":…} 内容映射（单一事实源）
│   ├── part1.py … part9.py     各部分课程内容（每课 LESSON_xx = {"zh":…, "en":…}）
│   ├── quizzes.py              每课自测（双语）
│   ├── build.py                站点构建（→ index.html + lessons/）
│   ├── build_print.py          PDF 构建（→ print-zh.html / print-en.html，折叠全展开）
│   ├── check_html.py           CI：校验生成 HTML 与 src/ 无漂移
│   └── check_links.py          CI：校验内部链接无死链
├── .github/workflows/
│   ├── deploy.yml              CI：构建站点 + 渲染双语 PDF + 部署 Pages
│   └── ci.yml                  CI：防回归（重建校验漂移 + 死链）
├── README.md
└── LICENSE                     （MIT）
```

### 3.2 双语机制（顶部即时切换）

- 每课页面同时渲染两块内容：`<div class="lang-zh">…</div>` 与 `<div class="lang-en">…</div>`。
- 顶部一个 `中 / EN` 切换按钮：JS 切换 `<html lang>` 或 body class，CSS 控制只显示一种语言。
- 选择写入 `localStorage`，**翻到下一课仍保持**同一语言。
- 默认语言：中文（首次访问）。
- 站点/课程的**外壳文案**（导航、进度、按钮、目录副标题）也双语化，跟随切换。

### 3.3 换皮（llama.cpp 主题）

- 主题色：从 langchain 的绿色改为 llama.cpp 风格的一套（暖橙 + 石墨蓝），保留深色模式。
- 角标/favicon：`λ` → llama.cpp 风格标记（🦙 或 `ggml` 单色字标）。
- 站点标题：「llama.cpp 图解教程 · 从零理解整个项目」/「llama.cpp Visual Guide」。
- **其余设计系统全部沿用**：卡片（macro/detail/analogy/key/warn/spark）、流程图（flow/vflow）、
  分层架构（layers）、两栏对比（cols）、表格、代码块（带高亮 span）、折叠（accordion）、quiz、
  顶部进度条、上一课/下一课导航、目录搜索。

---

## 4. 课程大纲（9 部分 · 40 课）

> 副标题给方向；每课的真实源码引用以"**文件 + 符号名**"为主（**不写死行号**，避免随上游更新失效）。
> 内容准则见第 6 节：**多图、多伪代码 / 源码简化片段、尽量详细**。

### 第一部分 · 宏观全景
1. **llama.cpp 是什么** — 解决什么问题 · 和 PyTorch/transformers/vLLM 的区别 · C/C++ 零依赖哲学
2. **项目全景地图** — ggml + `src/llama-*` + common + tools + 转换脚本，目录导览（**整体结构图**）
3. **一次推理的生命周期** — prompt → 分词 → 计算图 → logits → 采样 → token → 文字 全景数据流

### 第二部分 · 前置基础（从零打底）
4. **大模型推理基础** — decoder-only 极简回顾 · prefill vs decode · 为什么要 KV cache · 自回归
5. **张量是什么** — shape/stride/行优先 · `ggml_tensor` 字段直观理解
6. **量化入门** — 为什么量化（显存/带宽）· 块量化思想 · Q4_0/Q8_0/K-quant 一览
7. **构建系统与后端** — CMake · 后端（CPU/CUDA/Metal/Vulkan…）· 怎么编译 · 产物

### 第三部分 · ggml 张量引擎
8. **ggml 核心对象** — `ggml_context` · `ggml_tensor` · 内存池 · no-malloc 设计
9. **计算图：惰性构建** — 先建图后执行 · `ggml_cgraph` · op 节点 · src 反向指针
10. **图的执行与调度** — `ggml_backend` · `ggml_backend_sched` · ggml-alloc 内存复用
11. **核心算子** — matmul / rope / softmax / attention · 形状推导
12. **量化格式细节** — 块布局 · super-block · `ggml-quants.c` · 解量化
13. **GGUF 文件格式** — header / metadata KV / tensor info / 对齐 · `gguf.cpp` · mmap

### 第四部分 · llama 推理内部
14. **模型加载** — `llama-model-loader` · 读 metadata 与权重 · mmap · 分片
15. **架构与超参** — `llama-arch`（LLM_ARCH_*）· `llama-hparams` · 张量命名约定
16. **构建计算图** — `llama-graph` · 把权重接成 transformer 前向图
17. **上下文与会话** — `llama-context` · cparams · logits 输出
18. **批处理** — `llama-batch` · token/pos/seq_id · logits 标记
19. **KV cache** — `llama-kv-cache` · cell 管理 · 上下文移位 · 多序列 · 变体（iswa/recurrent/hybrid）
20. **分词器** — `llama-vocab` · BPE/SPM/WPM · `unicode.cpp` · 特殊 token · detokenize
21. **采样** — `llama-sampler` · sampler chain · temperature/top-k/top-p/min-p/repetition
22. **聊天模板** — `llama-chat` · 内置模板 · role 拼装
23. **语法约束生成** — `llama-grammar` · GBNF · json-schema-to-grammar · 受限解码
24. **LoRA 适配器** — `llama-adapter` · 热插拔 · 权重叠加

### 第五部分 · 公共 API 与工具
25. **C API 总览** — `include/llama.h` · 句柄类型 · 典型调用序列 · `llama-cpp.h` C++ 包装
26. **common 工具层** — `common.cpp` · arg 解析 · sampling 包装 · log · 下载/HF 缓存
27. **llama-cli** — `tools/cli` · 主循环 · 参数
28. **llama-server** — `tools/server` · HTTP · OpenAI 兼容 · slot · 连续批处理 · 调度
29. **quantize 工具** — `tools/quantize` · `llama-quant` · imatrix 重要性矩阵
30. **评测与基准** — perplexity · llama-bench

### 第六部分 · 底层内核（硬核）
31. **CPU 后端** — `ggml-cpu` · SIMD（AVX/NEON）· 向量点积 · 量化矩阵乘 · 多线程
32. **CUDA 后端** — `ggml-cuda` · kernel 结构 · dequant+matmul · flash attention · 显存
33. **Metal/Vulkan/其他** — `ggml-metal` · `ggml-vulkan` · 后端注册 · dl 动态加载 · RPC

### 第七部分 · 进阶专题
34. **投机解码** — `common/speculative` · draft model · n-gram · 接受率
35. **连续批处理与调度** — server slot · prefill/decode 交错 · 吞吐
36. **多模态** — `tools/mtmd` · 图像/音频投影 · clip · mmproj
37. **性能与内存优化** — mmap · offload · n-gpu-layers · flash-attn

### 第八部分 · 实战与贡献
38. **从 HF 转换模型** — `convert_hf_to_gguf.py` · `gguf-py` · 新增模型流程（HOWTO-add-model）
39. **编译/调试/测试/贡献** — build.md · 测试 · 调试 · CONTRIBUTING（含 AI 政策）

### 第九部分 · 速查
40. **术语表 · 概念索引** — 全书术语一句话查 + 点链接跳到对应课

---

## 5. 每课页面结构（页面解剖）

沿用模板的教学元素，每课大致包含（按需取用，不强制全有）：

- **顶部**：进度条 + 部分标签 + `NN / 40` + 语言切换按钮。
- **Hero**：部分名 + 课程标题。
- **导语**：一段话点题。
- **教学卡片**：
  - 🌍 **宏观理解**（macro）— 大局观、为什么这样设计。
  - 🔬 **细节 / 源码对应**（detail）— 指向真实文件 + 符号（如 `src/llama-graph.cpp` 的 `build_graph`）。
  - 🔌 **生活类比**（analogy）— 用日常事物帮助理解抽象概念。
  - ✅ **关键要点**（key）— 每课小结。
  - 💡 **设计亮点**（spark）— 该课最精妙的设计思想。
  - ⚠️ **坑 / 注意**（warn，按需）。
- **图（硬性要求，重点）**：**每课 3-5 张图**，类型尽量多样：
  - **结构 / 分层图**（`layers`）、**流程图**（`flow` 横向 / `vflow` 纵向步骤）、**对比图**（`cols` 并排）、
  - **概念示意图**（用 **HTML+CSS** 画的原理草图，如量化分块、KV cache 增长、prefill/decode 时间线；
    用设计系统的 CSS 变量配色，**深色模式自适应**，自包含、不依赖外部图片/SVG 资源）。
  - 宏观/结构课必须有"**整体结构图**"；原理课必须有至少 1 张**概念示意图**。
- **代码（重点）**：每课 **2-3 段**伪代码 / 从源码简化的真实片段（带高亮），讲清"为什么这么写"。
- **折叠深挖**（accordion，标准元素）：**每课 2-3 个**可展开的深入卡片，新手可跳过；
  推荐结构「**示例 → 为什么这样设计 → 源码在哪（文件+符号）→ 还有什么替代**」。
- **末尾自测**（quiz）：2-4 题双语，点开看解析。
- **底部导航**：上一课 / 下一课。

---

## 6. 内容准则

1. **多图（硬性）**：**每课 3-5 张图**，类型多样化（分层 / 流程 / 对比 / 概念示意），而非只画一张分层图。
   宏观/结构课必须有"整体结构图"，原理课必须有至少 1 张"概念示意图"。概念图优先用 **HTML+CSS**（设计系统
   CSS 变量配色、深色模式自适应、自包含），不用硬编码颜色的 SVG。
2. **多代码**：每课 **2-3 段**——以**伪代码**讲清思路，再给**从源码简化的真实片段**对照（标注来源文件 + 符号）。
3. **尽量详细（硬性）**：每课**纯中文正文目标 ~4000+ 汉字（CJK，按 `\u4e00-\u9fff` 计；不含英文/代码/路径）**，
   并配 **2-3 个折叠深挖**，
   把"是什么 / 为什么这么写 / 源码在哪 / 还有什么替代方案"讲透；宁多勿少。
4. **源码引用**：以"**文件 + 符号名**"为主，**不写死行号**（行号会随上游更新失效）。
5. **准确性**：所有源码引用对照 llama.cpp 仓库**真实代码核实**；标注核验日期与版本锚点。
6. **ASCII 优先**：正文/代码避免 em-dash `—`、unicode 箭头 `→` 等（用 `-`、`->`、`...`）——
   *仅指课程"代码片段"内部*；中文叙述与既有模板中的排版符号不受此限。
7. **自包含**：页面无外部 JS/CSS 依赖，相对链接，支持 `file://` 与任意静态服务器。
8. **双语对齐**：中/英两份内容**信息等价**（不要求逐字对译，但要点不能缺）。

---

## 7. 配套功能

- **quizzes.py**：每课 2-4 题自测，双语，折叠看解析。
- **build_print.py**：把全 40 课合成单页、折叠全展开、自动分页，分别产出
  `print-zh.html` 与 `print-en.html`，供无头 Chrome 打成
  `llama-cpp-visual-guide-zh.pdf` / `-en.pdf`。
- **CI**：
  - `deploy.yml`：push 即重建站点 + 渲染双语 PDF（安装 CJK/emoji 字体）+ 部署 GitHub Pages；
    打 `v*` 标签时发布 Release 附带双语 PDF。
  - `ci.yml`：每次 push/PR 重跑 `build.py` 校验提交 HTML 与 `src/` 无漂移；跑 `check_links.py` 防死链。
- **首次启用 Pages**：仓库 Settings → Pages → Source 选 GitHub Actions（一次性，需仓库 owner 操作）。

---

## 8. 里程碑（分步交付）

按"先骨架、再内容分批"的节奏，每个 milestone 都能独立构建、可在浏览器查看：

- **M0 · 脚手架**：`src/shell.py`（设计系统 + 双语切换 + 导航）、`registry.py`、`build.py`、
  1-2 课样板内容 + index 页跑通；确定主题色与品牌、双语切换交互。
- **M1 · 第一部分（宏观全景，3 课）**：含整体结构图，确立"图 + 伪代码 + 详尽"的内容基线。
- **M2 · 第二部分（前置基础，4 课）**。
- **M3 · 第三部分（ggml 引擎，6 课）**。
- **M4 · 第四部分（llama 推理内部，11 课）**。
- **M5 · 第五部分（公共 API 与工具，6 课）**。
- **M6 · 第六部分（底层内核，3 课）**。
- **M7 · 第七部分（进阶专题，4 课）**。
- **M8 · 第八部分（实战与贡献，2 课）+ 第九部分（术语表，1 课）**。
- **M9 · 配套收尾**：quizzes 补全、`build_print.py` 双语 PDF、CI（deploy.yml / ci.yml）、README。

> 每个 milestone 内部再按课拆分；写实施计划时**分步逐段**，不一次写完。

---

## 9. 非目标 / YAGNI

- 不做账号、评论、后端服务、搜索引擎索引等动态功能（站点是纯静态）。
- 不逐行翻译/搬运 llama.cpp 全部源码；只挑**讲清原理所需**的简化片段。
- 不追求覆盖每一个后端的每个 kernel；底层内核部分以 CPU/CUDA 为主线，其它后端做概览。
- 不向 llama.cpp 上游提交任何内容。

---

## 10. 成功标准

- 一个完全没接触过 llama.cpp 的人，能顺着 1 → 40 课**一点点读懂**整个项目的结构与原理。
- 每课都能回答："这个组件**是什么功能**、**为什么这么写**、对应**哪些源码文件/符号**"。
- 站点可 `file://` 直接打开，也可部署到 GitHub Pages；可下载中/英 PDF。
- 中英双语切换顺滑、跨课保持；CI 防漂移与死链。
