# M9 · 配套收尾（PDF · CI · README · LICENSE）设计文档

> **配套：** 父级总设计 `2026-06-13-llama-cpp-visual-guide-design.md`（§M9）· roadmap `2026-06-13-llama-cpp-visual-guide-roadmap.md`（M9 行）。
> **执笔基线：** 沿用 M1-M8 的零依赖 Python 静态站点生成器；M9 不新增课程，只补"打包/发布/工程化"的配套。

## Goal

四十课内容已全部完成并合并。M9 给这个图解教程补齐最后一圈"配套"，让它从"一堆能跑的脚本 + HTML"变成一个**可发布、可打印、可被他人 clone 即用、有明确许可与维护入口**的完整项目。具体四件事：(1) 双语**打印版**（`build_print.py` 生成 `print_zh.html`/`print_en.html`，浏览器一键导 PDF）；(2) **CI**（`ci.yml` 自动验证站点能构建且校验全过）；(3) **部署**（`deploy.yml` 把站点发布到 GitHub Pages）；(4) **README + LICENSE**（项目说明 + 双许可）。

## 总基调

- **零依赖优先**：`build_print.py` 沿用纯标准库，产物是**自包含的打印友好 HTML**（内联同一套 CSS + 打印分页），任何浏览器 `Ctrl/Cmd+P` 即可导出 PDF；真 PDF 只作为 CI 里的**可选**步骤（`chromium --print-to-pdf`），不作为本地硬依赖。
- **不破坏现有站点**：M9 只**新增**文件（`src/build_print.py`、`README.md`、`LICENSE`、`LICENSE-CONTENT`、`.github/workflows/*.yml`、产出的 `print_*.html`），不改 `build.py`/`shell.py`/`registry.py` 等既有逻辑；现有 40 课与 index 一字不动。
- **沿用校验门**：CI 跑的就是本地一直在用的 `build.py` + `check_html.py` + `check_links.py`，外加"构建后 git 无改动"（确保提交的 HTML 与源同步）。

## 范围取舍（与用户确认）

1. **打印产物形态**（与用户确认）：生成**两份独立**的打印友好 HTML —— `print_zh.html`（全 40 课中文 + 逐课分页）、`print_en.html`（全 40 课英文）。用浏览器 `Ctrl/Cmd+P` 导出 PDF；CI 里**可选**用 `chromium --print-to-pdf` 生成真 PDF 作为构件（artifact）。**不**引入 weasyprint/wkhtmltopdf 这类重依赖。
2. **许可**（与用户确认）：**双许可** —— 代码（Python 生成器 / 校验脚本）用 **MIT**；教学内容（课程 HTML / 文字 / 图）用 **CC BY 4.0**。两个文件：`LICENSE`（MIT）、`LICENSE-CONTENT`（CC BY 4.0），README 里说清分界。
3. **部署**：GitHub Pages via Actions。**注意**（已知约束）：仓库所有者须在 Settings -> Pages -> Source 选 "GitHub Actions" **手动启用一次**，`configure-pages` 无法用 GITHUB_TOKEN 自动建站（缺 admin 权限），只能部署。README/design 里写明这一步。
4. **quizzes 补全**：核查发现 **40/40 课均已有 quiz**（每课 3 MCQ + 1 开放），此项已满足，M9 只做"核验"不新增。

## 里程碑范围（M9 = 配套收尾，无新增课）

| 产出 | 文件 | 作用 |
| --- | --- | --- |
| 打印生成器 | `src/build_print.py` | 把 40 课 + index 拼成 `print_zh.html`/`print_en.html`，逐课分页、打印 CSS |
| 打印产物 | `print_zh.html` / `print_en.html` | 自包含、可直接 `Ctrl+P` 导 PDF（也被 Pages 一并发布） |
| CI | `.github/workflows/ci.yml` | push/PR 时跑 build + check_html + check_links + "构建无 diff" |
| 部署 | `.github/workflows/deploy.yml` | push 到 master 时构建并发布到 GitHub Pages |
| 项目说明 | `README.md` | 是什么 / 怎么看 / 怎么构建 / 结构 / 双许可 / 部署须知 |
| 许可 | `LICENSE`（MIT）· `LICENSE-CONTENT`（CC BY 4.0） | 代码 + 内容双许可 |

## 各组件设计

### `src/build_print.py`（打印版生成器）

- **接口**：`python3 build_print.py` -> 在仓库根写出 `print_zh.html`、`print_en.html`，打印行数/课数到 stdout（仿 `build.py` 风格）。
- **复用**：`import registry`（拿 `CONTENT` 有序课表）、`import shell`（拿 PAGES/SUBTITLES 与 **同一套 CSS**，保证打印版和在线版观感一致；若 CSS 在 shell 里是常量则直接取，否则从 `shell` 暴露的渲染函数里取）。`import quizzes`（每课打印版也带上自测题，但**折叠展开**为静态可读形式）。
- **结构**：单 HTML，`<head>` 内联站点 CSS + **打印专用 CSS**（`@media print`：每课 `page-break-before: always`；隐藏在线版的导航/进度条/语言切换/交互按钮；`.trace`/`table.t`/`svg` 保证不被分页截断 `break-inside: avoid`）。`<body>` 里按 `registry.CONTENT` 顺序，逐课输出"标题 + 该语言正文 + quiz（静态展开）"，课与课之间分页。
- **语言**：`print_zh.html` 只取每课 `["zh"]`、`print_en.html` 只取 `["en"]`（不靠 `data-lang` 运行时切换——打印版是静态的）。
- **范围取舍落实**：不内联 SVG 之外的外链；产物自包含、可单文件分发。

### `.github/workflows/ci.yml`

- **触发**：`push`（master）+ `pull_request`。
- **步骤**：checkout -> setup-python（3.x，无 pip 依赖）-> `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` -> `python3 build_print.py` -> `git diff --exit-code`（确保仓库内已提交的 HTML 与源同步、构建无遗漏）。
- **可选**：装 chromium，跑 `chromium --headless --print-to-pdf` 把 `print_zh.html`/`print_en.html` 转成 PDF，作为 artifact 上传（`actions/upload-artifact`）。失败不阻断主校验（`continue-on-error` 或单独 job）。

### `.github/workflows/deploy.yml`

- **触发**：`push`（master）+ `workflow_dispatch`。权限 `pages: write`、`id-token: write`。
- **步骤**：checkout -> `cd src && python3 build.py && python3 build_print.py`（构建出最新站点 + 打印版）-> `actions/configure-pages` -> `actions/upload-pages-artifact`（path 为仓库根，含 `index.html`/`lessons/`/`print_*.html`）-> `actions/deploy-pages`。
- **已知约束**：所有者须先在 Settings -> Pages 选 "GitHub Actions" 启用一次（见上）。

### `README.md`

- 中英双语简介（项目是什么、面向谁、覆盖 llama.cpp 哪些主题）；一张"九个部分"目录表（可链接到在线站点 / 各课）；**怎么看**（在线 Pages 链接占位 + 本地 `cd src && python3 build.py` 后开 `index.html`）；**怎么打印**（`python3 build_print.py` -> 开 `print_zh.html` -> Ctrl+P）；**项目结构**（`src/` 生成器、`lessons/` 产出、`docs/` 设计与计划）；**双许可说明**（代码 MIT / 内容 CC BY 4.0）；**部署须知**（手动启用 Pages 一次）；一句"本指南是对 llama.cpp 的第三方学习材料，非官方、不含其源码"。

### `LICENSE` / `LICENSE-CONTENT`

- `LICENSE`：标准 MIT 全文（年份 2026、版权人占位为仓库所有者）。
- `LICENSE-CONTENT`：CC BY 4.0（用官方简短指引 + 链接，或嵌入 deed 摘要 + 指向 `https://creativecommons.org/licenses/by/4.0/`）。

## 统一交付标准（M9 适配）

- **零新依赖**：`build_print.py` 仅用标准库；CI 不 `pip install` 任何东西（chromium 经 apt/action 装，且仅用于可选 PDF）。
- **不碰既有逻辑**：除新增文件外，`src/*.py`（除新建 `build_print.py`）、`lessons/*`、`index.html` 内容不变；M9 提交后 `build.py` 重跑应无 diff。
- **打印版自检**：`print_zh.html`/`print_en.html` 用 `check_html` 思路自检（HTML 结构合法、无未转义 `<`、无双重转义）；用 chromium 目检分页正常、trace/表/SVG 不被截断；中文版含中文、英文版纯 ASCII（除卡片 emoji）。
- **CI/部署 YAML**：用 ASCII；action 版本固定（如 `actions/checkout@v4`）；deploy 权限最小化。
- **英文/ASCII 纪律**：README/LICENSE 的英文段、YAML、`print_en.html` 均纯 ASCII（无 unicode 箭头/破折号/中点）。

## 与 roadmap 衔接

- 完成后：roadmap M9 行"状态"`待写`->`完成`、状态追踪 `- [ ] M9`->`- [x] M9`；总进度可标注"全部里程碑完成"。
- 执行：superpowers:subagent-driven-development（一组件一个 task，顺序执行；收尾 task 勾 roadmap + 全量验证 + 整体复审 + 完成分支）。M9 是工程文件、非硬核课，控制器可亲自执笔，仍跑 spec+质量双重审查。
- 部署/CI 的 YAML 与 build_print 先本地核验（`build_print.py` 跑通 + chromium 目检 PDF；YAML 用 `python -c "import yaml"` 或 actionlint 思路静态检查语法）再提交。
