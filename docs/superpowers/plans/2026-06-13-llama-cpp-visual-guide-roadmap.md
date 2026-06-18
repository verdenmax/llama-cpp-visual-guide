# llama.cpp 图解教程 · 实施路线图（Roadmap）

> **配套 Spec：** `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`
> **For agentic workers:** 本文件是**总览**，记录全部 milestone。每个 milestone 有**独立的详细 plan**
> （`docs/superpowers/plans/2026-06-13-...-M*.md`），用 superpowers:subagent-driven-development 或
> superpowers:executing-plans 逐个执行。

**Goal:** 用纯 Python（零依赖）生成器，做出一套中英双语、可 `file://` 打开、可部署 Pages、可导出 PDF 的
llama.cpp 图解教程（9 部分 40 课）。

**Architecture:** 复用 langchain-visual-guide 架构：`src/shell.py`（设计系统 CSS + `PAGES` + `page()`/
`index_page()` + 双语切换 JS）→ `registry.py`（文件名 → `{"zh","en"}` 内容）→ `part1..part9.py`（每课内容）→
`quizzes.py`（双语自测）→ `build.py`（站点）/`build_print.py`（双语 PDF）→ `check_html.py`/`check_links.py`（CI 校验）。

**Tech Stack:** Python 3（仅标准库）· 自包含 HTML/CSS/JS · GitHub Actions（Pages + 无头 Chrome 打 PDF）。

---

## 里程碑总表

| MS | 名称 | 覆盖课 | 主要产出 | 详细 plan 文件 | 状态 |
| --- | --- | --- | --- | --- | --- |
| **M0** | 脚手架 | 样板 1-2 课 | `shell.py`(双语) · `registry.py` · `build.py` · 样板课 · `index.html` · `check_*.py` 跑通 | `...-M0-scaffold.md` | 完成 |
| **M1** | 第一部分 · 宏观全景 | 1-3 | 3 课（含整体结构图），确立"图+伪代码+详尽"基线 | `...-M1-part1-overview.md` | 完成 |
| **M2** | 第二部分 · 前置基础 | 4-7 | 4 课 | `...-M2-part2-foundations.md` | 完成 |
| **M3** | 第三部分 · ggml 引擎 | 8-13 | 6 课 | `...-M3-part3-ggml.md` | 完成 |
| **M4a** | 第四部分上 · llama 推理内部 | 14-19 | 6 课（加载/架构/计算图/上下文/批处理/KV cache） | `...-M4a-part4-llama-internals.md` | 完成 |
| **M4b** | 第四部分下 · llama 推理内部 | 20-24 | 5 课（词表/采样/对话模板/语法/LoRA） | `...-M4b-part4-llama-internals.md` | 完成 |
| **M5** | 第五部分 · 公共 API 与工具 | 25-30 | 6 课 | `...-M5-part5-tools.md` | 完成 |
| **M6** | 第六部分 · 底层内核 | 31-33 | 3 课（CPU/CUDA 主线 + 其它后端概览） | `...-M6-part6-kernels.md` | 完成 |
| **M7** | 第七部分 · 进阶专题 | 34-37 | 4 课 | `...-M7-part7-advanced.md` | 完成 |
| **M8** | 第八部分 · 实战与贡献 + 第九部分 · 速查 | 38-40 | 3 课（含术语表） | `...-M8-part8-9-contrib-glossary.md` | 待写 |
| **M9** | 配套收尾 | - | quizzes 补全 · `build_print.py` 双语 PDF · `deploy.yml`/`ci.yml` · README · LICENSE | `...-M9-pdf-ci-readme.md` | 待写 |

> 课时合计：3+4+6+11+6+3+(2+1) = **40 课**。

> M1.5：第一部分 3 课已按加强标准回炉（每课 3-5 图 + 折叠深挖 + 概念示意图）。

> M1.6：再把每课纯中文加深到 ~4000+ 汉字（CJK 计），课02补"转换->量化->运行"代码片段与 GGUF 结构概念图；check_html 加中文密度软检查。

---

## 依赖与顺序

```
M0 (脚手架, 一切的基础)
  --> M1 (确立内容基线)
        --> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8   (各部分内容, 顺序递进)
                                                       --> M9 (配套收尾: 全部课就位后做 PDF/CI)
```

- **M0 必须先完成**：它定义双语机制、设计系统、`PAGES`/`registry` 结构、构建与校验流程，
  后续每个内容 milestone 只是在 `partN.py` + `quizzes.py` 里加内容、在 `shell.PAGES` 里登记课程。
- **M1 确立内容基线**：第一部分写完后，沉淀一套"每课模板"（卡片顺序、图怎么画、伪代码/源码片段怎么放、
  双语怎么对齐），后续 M2-M8 照此基线产出，减少返工。
- **M9 最后做**：PDF/CI 依赖全部 40 课与 index 稳定后再收尾。

---

## 每个内容 milestone（M1-M8）的统一交付清单

每个内容 milestone 的详细 plan 都会按"**每课一组任务**"展开，每课产出：

1. 在 `shell.PAGES` 登记该课（filename, zh/en 短标题, zh/en 部分标签）。
2. 在 `partN.py` 写 `LESSON_xx = {"zh": ..., "en": ...}`：导语 + 教学卡片（macro/detail/analogy/key/spark）
   + **结构/流程图** + **伪代码或源码简化片段** + 折叠深挖。
3. 在 `registry.py` 登记 `文件名 -> LESSON_xx`。
4. 在 `quizzes.py` 写该课双语自测（2-4 题）。
5. 在 `index_page` 的副标题表登记该课副标题（zh/en）。
6. 运行 `build.py` + `check_html.py` + `check_links.py`，全绿后 commit。

> 校验门槛（每课/每 milestone 结束都要过）：`check_html.py` 0 error（结构/导航链/计数/防漂移）、
> `check_links.py` 0 死链、中英两份内容均存在且可切换。

---

## 内容准则（贯穿所有 milestone，详见 Spec §6）

- **多图**：每课尽量 1-2 张结构/流程/分层图；宏观课要有整体结构图。
- **多代码**：伪代码讲思路 + 从源码简化的真实片段（标注文件 + 符号）。
- **尽量详细**：把"是什么 / 为什么这么写 / 还有什么替代"讲透。
- **源码引用以"文件 + 符号名"为主，不写死行号**；对照真实代码核实，标注核验日期。
- 代码片段内部 ASCII 优先（`-`/`->`/`...`），不用 em-dash / unicode 箭头。

---

## 状态追踪

- [x] M0 脚手架
- [x] M1 第一部分（宏观全景）
- [x] M2 第二部分（前置基础）
- [x] M3 第三部分（ggml 引擎）
- [x] M4 第四部分（llama 推理内部）— M4a（14-19）+ M4b（20-24）完成
- [x] M5 第五部分（公共 API 与工具）
- [x] M6 第六部分（底层内核）
- [x] M7 第七部分（进阶专题）
- [ ] M8 第八部分 + 第九部分（实战/贡献 + 术语表）
- [ ] M9 配套收尾（PDF / CI / README）
