# M9 配套收尾（PDF · CI · README · LICENSE）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐 task 执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 给已完成的 40 课图解教程补齐配套：双语打印版（`build_print.py` -> `print_zh.html`/`print_en.html`）、CI（`ci.yml`）、部署（`deploy.yml`）、`README.md` 与双许可（`LICENSE` MIT + `LICENSE-CONTENT` CC BY 4.0）。

**Architecture:** 沿用零依赖 Python 静态站点生成器。M9 只**新增**文件、不改既有逻辑：新建 `src/build_print.py`（复用 `registry.CONTENT` + `shell.CSS` + `quizzes`），新建仓库根的 `README.md`/`LICENSE`/`LICENSE-CONTENT` 与 `.github/workflows/{ci,deploy}.yml`。

**Tech Stack:** Python 3 标准库；GitHub Actions（actions/checkout@v4、setup-python、configure-pages/upload-pages-artifact/deploy-pages）；chromium（仅 CI 里可选生成真 PDF）。

---

## 统一交付标准（M9 适配）

- **零新依赖**：`build_print.py` 仅标准库；CI 不 `pip install`（chromium 经 apt/action 装，仅用于可选 PDF）。
- **不碰既有逻辑**：除新增文件外，`src/*.py`（除新建 `build_print.py`）、`lessons/*`、`index.html`、`shell.py`/`registry.py`/`quizzes.py`/`build.py`/`check_*.py` 内容不变；提交后 `python3 build.py` 重跑应无 diff。
- **打印版自检**：`print_zh.html`/`print_en.html` HTML 结构合法、无未转义裸 `<`、无双重转义；chromium 目检逐课分页、trace/表/SVG 不被截断；zh 版含中文、en 版纯 ASCII（除卡片 emoji）。
- **ASCII 纪律**：README/LICENSE 英文段、所有 YAML、`print_en.html` 纯 ASCII（无 unicode 箭头/破折号/中点）。
- **YAML 合法**：每个 workflow 用 `python3 -c "import yaml; yaml.safe_load(open(f))"`（若无 pyyaml 则用结构目检）确认可解析；action 版本固定。

## 执行方式

- superpowers:subagent-driven-development，一组件一个 task（Task 1=build_print、Task 2=README+LICENSE、Task 3=CI/deploy、Task 4=收尾）。
- 每个 task：实现 -> **spec 合规审查 -> 质量审查**（两段审查），修复回环后再标完成。子代理用当前主会话模型，显式传 `model`。
- M9 是工程文件、非硬核课，控制器可亲自执笔，仍跑完整 spec+质量双重审查。
- commit 用 `Assisted-by: GitHub Copilot`。分支：在 master 上从本 plan 提交后，新建 `feature/m9-pdf-ci-readme` 分支做实现。
- **HTML/产物是被 git 跟踪的**：`build_print.py` 产出的 `print_*.html` 要 `git add`，提交后 `git status` 干净。

---

## Task 1: `src/build_print.py`（双语打印版生成器）

**Files:** **新建** `src/build_print.py`。产出仓库根 `print_zh.html`、`print_en.html`。复用 `registry.CONTENT`、`shell.CSS`/`shell.PAGES`、`quizzes.render`。

**关键事实（已核实）：** `build.py` 逐课 `content = CONTENT[fname][lang] + quizzes.render(fname, lang)`，再 `shell.page()` 包壳。`shell.CSS`（src/shell.py:139）是模块级常量，可直接 `shell.CSS` 取整套站点 CSS。`shell.PAGES` 是有序元组 `(fname, zh_title, en_title, zh_part, en_part)`。`quizzes.render()` 把答案放进 `<details class="accordion">`（默认折叠）；课文里的"深挖"也是 `<details class="accordion">`。打印版要把这些 `<details>` 全部 `open`，让答案/深挖可见。课文正文（`CONTENT[fname][lang]`）是**单语裸 HTML 片段**（卡片/section/trace 等），不带 data-lang 包裹，直接嵌入即可被 `shell.CSS` 的基础样式渲染。

- [ ] **Step 1: 写 `src/build_print.py`（完整实现）**

```python
"""Generate print-friendly bilingual HTML: print_zh.html and print_en.html.

Each file is self-contained (inlines shell.CSS + print CSS), contains a TOC plus
all lessons in order, one page per lesson, with every <details> expanded so quiz
answers and deep-dives are visible. Open in a browser and Ctrl/Cmd+P to a PDF.

Usage:
    cd src && python build_print.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import shell  # noqa: E402
import quizzes  # noqa: E402
from registry import CONTENT  # noqa: E402

TITLE = {"zh": "llama.cpp 图解学习指南 - 打印版", "en": "llama.cpp Visual Guide - Print Edition"}
INTRO = {
    "zh": "全 40 课 - 逐课分页。用浏览器 Ctrl/Cmd+P 即可导出 PDF。",
    "en": "All 40 lessons - one page each. Use Ctrl/Cmd+P in a browser to export a PDF.",
}
TOC = {"zh": "目录", "en": "Contents"}

PRINT_CSS = """
body { max-width: 820px; margin: 0 auto; padding: 1.6rem; background: #fff; }
.print-toc { margin: 1rem 0 2rem; }
.print-toc li { margin: .2rem 0; }
.lesson-print { padding-top: .5rem; }
@media print {
  .lesson-print { page-break-before: always; }
  .lesson-print:first-of-type { page-break-before: avoid; }
  .trace, table.t, svg, pre, .layers, .cols, .card, details { break-inside: avoid; }
  a { color: inherit; text-decoration: none; }
}
details[open] > summary { list-style: none; }
"""


def _expand_details(html):
    # show quiz answers and deep-dives in the static print version
    return html.replace('<details class="accordion">', '<details class="accordion" open>')


def build_lang(lang):
    htmllang = "zh-CN" if lang == "zh" else "en"
    head = (
        f'<!doctype html>\n<html lang="{htmllang}" data-lang="{lang}">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{TITLE[lang]}</title>\n"
        f"<style>{shell.CSS}\n{PRINT_CSS}</style>\n</head>\n<body>\n"
    )
    parts = [f'<h1>{TITLE[lang]}</h1>\n<p style="color:var(--muted)">{INTRO[lang]}</p>']
    toc = [f'<div class="print-toc"><h2>{TOC[lang]}</h2>\n<ol>']
    for page in shell.PAGES:
        title = page[1] if lang == "zh" else page[2]
        toc.append(f"  <li>{title}</li>")
    toc.append("</ol></div>")
    parts.append("\n".join(toc))
    for page in shell.PAGES:
        fname = page[0]
        title = page[1] if lang == "zh" else page[2]
        body = _expand_details(CONTENT[fname][lang])
        quiz = _expand_details(quizzes.render(fname, lang))
        parts.append(f'<section class="lesson-print">\n<h1>{title}</h1>\n{body}\n{quiz}\n</section>')
    return head + "\n".join(parts) + "\n</body>\n</html>\n"


def build():
    written = []
    for lang in ("zh", "en"):
        html = build_lang(lang)
        out = os.path.join(ROOT, f"print_{lang}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        written.append(f"print_{lang}.html")
    return written


if __name__ == "__main__":
    done = build()
    n_lessons = len(shell.PAGES)
    print(f"Wrote {len(done)} print files ({n_lessons} lessons each):", ", ".join(done))
```

- [ ] **Step 2: 跑生成器**：`cd src && python3 build_print.py`，期望 `Wrote 2 print files (40 lessons each): print_zh.html, print_en.html`。

- [ ] **Step 3: 自检（HTML 合法 + 单语纯净）**：

```bash
cd /home/verden/course/llama-cpp-visual-guide
python3 - <<'EOF'
import html.parser
for f in ("print_zh.html","print_en.html"):
    s=open(f,encoding="utf-8").read()
    # well-formed-ish: parser doesn't raise
    class P(html.parser.HTMLParser):
        pass
    P().feed(s)
    assert "&amp;lt;" not in s and "&amp;gt;" not in s, f+": double-escape"
    print(f, "OK len", len(s), "lessons", s.count('class="lesson-print"'))
# en print pure-ASCII (except card emoji)
en=open("print_en.html",encoding="utf-8").read()
bad=[hex(ord(c)) for c in en if ord(c)>127 and c not in "🌍🔌✅💡🔬·…—×≈±"]
# allow only the 4 standard card emojis:
bad=[hex(ord(c)) for c in en if ord(c)>127 and c not in "🌍🔌✅💡"]
assert not bad, ("en non-ascii", bad[:10])
print("en print pure-ASCII OK")
EOF
```

期望：两个文件各 40 个 `lesson-print` section、无双重转义、`print_en.html` 纯 ASCII（除 4 个卡片 emoji）。

- [ ] **Step 4: chromium 目检分页**（控制器目检）：`chromium --headless --print-to-pdf=/tmp/guide_zh.pdf print_zh.html`，确认能生成 PDF、逐课分页、trace/表/SVG 不被截断、`<details>` 已展开（答案可见）。

- [ ] **Step 5: commit**：`git add src/build_print.py print_zh.html print_en.html` + `feat: add build_print.py generating print_zh/print_en (bilingual print edition)` + `Assisted-by: GitHub Copilot`。提交后 `git status` 干净。

## Task 2: `README.md` + `LICENSE` + `LICENSE-CONTENT`

**Files:** **新建** 仓库根 `README.md`、`LICENSE`（MIT）、`LICENSE-CONTENT`（CC BY 4.0）。版权人 `verdenmax`（取自 `git config user.name`）。

- [ ] **Step 1: 写 `LICENSE`（MIT 全文）**

```text
MIT License

Copyright (c) 2026 verdenmax

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

----------------------------------------------------------------------------
The MIT license above covers the CODE of this project (the Python generators
and validation scripts under src/). The educational CONTENT (lesson text and
diagrams rendered into index.html, lessons/*.html and print_*.html) is licensed
separately under CC BY 4.0 - see LICENSE-CONTENT.
```

- [ ] **Step 2: 写 `LICENSE-CONTENT`（CC BY 4.0 通知）**

```text
Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2026 verdenmax

The educational CONTENT of this project - the lesson prose and diagrams that are
authored in src/part*.py and src/quizzes.py and rendered into index.html,
lessons/*.html and print_*.html - is licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).

You are free to:
  - Share - copy and redistribute the material in any medium or format
  - Adapt - remix, transform, and build upon the material for any purpose,
    even commercially

Under the following term:
  - Attribution - You must give appropriate credit, provide a link to this
    license, and indicate if changes were made.

Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode
Human-readable summary: https://creativecommons.org/licenses/by/4.0/

NOTE: This guide is third-party educational material ABOUT llama.cpp. It does
not include llama.cpp source code; llama.cpp itself is MIT-licensed by its own
authors (https://github.com/ggml-org/llama.cpp).
```

- [ ] **Step 3: 写 `README.md`（双语）**。结构（用真实内容，非占位）：
  - 标题 + 一句话双语简介："llama.cpp 图解学习指南 / A visual, bilingual guide to llama.cpp internals - 40 lessons from 'what is llama.cpp' to 'how to contribute'."
  - **声明**：第三方学习材料、非官方、不含 llama.cpp 源码（指向 ggml-org/llama.cpp）。
  - **谁适合读 / What it covers**：九个部分一句话各概括（对应 L40 全书地图）。
  - **目录表**：九个部分 + 课号范围（markdown 表，照 roadmap/L40）。
  - **怎么看 / View**：(a) 在线（GitHub Pages，URL 占位 `https://<owner>.github.io/<repo>/`）；(b) 本地 `cd src && python3 build.py` 后用浏览器打开 `index.html`（零依赖，Python 3 即可）。
  - **打印 / Print**：`cd src && python3 build_print.py` -> 打开 `print_zh.html` 或 `print_en.html` -> `Ctrl/Cmd+P` 导出 PDF。
  - **项目结构 / Structure**：`src/`（生成器 part*.py + shell.py + registry.py + quizzes.py + build.py + build_print.py + check_*.py）、`lessons/`（产出 HTML）、`docs/superpowers/`（设计与计划）、`index.html`/`print_*.html`（产出）。
  - **构建与校验 / Build & validate**：`python3 build.py && python3 check_html.py && python3 check_links.py`。
  - **部署须知 / Deploy note**：GitHub Pages via Actions —— **所有者须先在 Settings -> Pages -> Source 选 "GitHub Actions" 启用一次**（configure-pages 无法自动建站）。
  - **许可 / License**：代码 MIT（LICENSE）、内容 CC BY 4.0（LICENSE-CONTENT）。
  - 英文段全部纯 ASCII（用 `-`/`->`，不用 unicode 破折号/箭头）。

- [ ] **Step 4: 校验**：`README.md` 链接/路径正确；`LICENSE`/`LICENSE-CONTENT`/README 英文纯 ASCII（`grep -nP "[^\x00-\x7F]" LICENSE LICENSE-CONTENT` 应只在 README 的中文段命中，LICENSE 两文件应 0 命中）。`build.py` 重跑无 diff（未碰站点）。

- [ ] **Step 5: commit**：`git add README.md LICENSE LICENSE-CONTENT` + `docs: add README and dual license (code MIT, content CC BY 4.0)` + `Assisted-by: GitHub Copilot`。

## Task 3: `.github/workflows/ci.yml` + `deploy.yml`

**Files:** **新建** `.github/workflows/ci.yml`、`.github/workflows/deploy.yml`。纯 ASCII，action 版本固定。

- [ ] **Step 1: 写 `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  build-and-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.x'
      - name: Build site and print editions
        run: |
          cd src
          python3 build.py
          python3 build_print.py
      - name: Validate HTML and links
        run: |
          cd src
          python3 check_html.py
          python3 check_links.py
      - name: Ensure committed output is in sync with sources
        run: git diff --exit-code

  print-pdf:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.x'
      - name: Build print editions
        run: |
          cd src
          python3 build.py
          python3 build_print.py
      - name: Install chromium
        run: sudo apt-get update && sudo apt-get install -y chromium-browser || sudo apt-get install -y chromium
      - name: Render PDFs
        run: |
          BIN="$(command -v chromium || command -v chromium-browser)"
          "$BIN" --headless --no-sandbox --print-to-pdf=guide_zh.pdf print_zh.html
          "$BIN" --headless --no-sandbox --print-to-pdf=guide_en.pdf print_en.html
      - uses: actions/upload-artifact@v4
        with:
          name: guide-pdfs
          path: '*.pdf'
```

- [ ] **Step 2: 写 `.github/workflows/deploy.yml`**

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.x'
      - name: Build site and print editions
        run: |
          cd src
          python3 build.py
          python3 build_print.py
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: '.'
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 3: 校验 YAML 语法**：

```bash
cd /home/verden/course/llama-cpp-visual-guide
python3 - <<'EOF'
try:
    import yaml
    for f in (".github/workflows/ci.yml", ".github/workflows/deploy.yml"):
        yaml.safe_load(open(f))
        print(f, "parses OK")
except ImportError:
    # no pyyaml: structural sanity only (keys present, indentation)
    for f in (".github/workflows/ci.yml", ".github/workflows/deploy.yml"):
        s=open(f).read()
        assert s.startswith("name:") and "jobs:" in s and "runs-on: ubuntu-latest" in s, f
        print(f, "structural check OK (pyyaml absent)")
EOF
grep -nP "[^\x00-\x7F]" .github/workflows/ci.yml .github/workflows/deploy.yml || echo "workflows pure-ASCII OK"
```

期望：两个 YAML 可解析（或结构检查过）、纯 ASCII。

- [ ] **Step 4: commit**：`git add .github/workflows/ci.yml .github/workflows/deploy.yml` + `ci: add GitHub Actions for validation (ci.yml) and Pages deploy (deploy.yml)` + `Assisted-by: GitHub Copilot`。
  - **注意**：本地无法真正触发 GitHub Actions；语法/结构本地核验即可。部署须所有者在 GitHub Settings -> Pages 手动启用一次（写在 README）。

## Task 4: 收尾（roadmap 勾选 + 全量验证 + 整体复审 + 完成分支）

**Files:** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（勾 M9）。

- [ ] **Step 1: 更新 roadmap**：里程碑总表 M9 行"状态"`待写` -> `完成`；状态追踪 `- [ ] M9 ...` -> `- [x] M9 ...`；可在顶部标注"全部里程碑（M0-M9）完成"。commit `docs: mark M9 done - all milestones complete` + `Assisted-by: GitHub Copilot`。
- [ ] **Step 2: 全量验证**（master 合并前在分支上跑）：
  - `cd src && python3 build.py && python3 check_html.py && python3 check_links.py` = 40 课 + index、0 error/warning、206 链接全解。
  - `python3 build_print.py` = `Wrote 2 print files (40 lessons each)`；`print_zh.html`/`print_en.html` 自检过（结构合法、无双重转义、en 纯 ASCII、各 40 个 lesson-print）。
  - `git status` 干净（站点 HTML + print_*.html 均已提交）。
  - 三个 YAML 可解析/结构过、纯 ASCII；README/LICENSE/LICENSE-CONTENT 英文段纯 ASCII。
- [ ] **Step 3: 整体复审**（建议）：派一个 superpowers:code-reviewer 子代理（当前模型 `claude-opus-4.8`）复审 `master..HEAD`：(1) `build_print.py` 正确复用 `registry/shell/quizzes`、产物自包含且逐课分页、`<details>` 已展开；(2) README 路径/命令/部署须知准确，LICENSE 双许可表述无误、CC BY 4.0 链接正确；(3) 两个 workflow YAML 合法、action 版本固定、权限最小、`git diff --exit-code` 门有意义；(4) **范围未越界**——只新增 `build_print.py`/`README`/`LICENSE`×2/两个 workflow/`print_*.html`，未改 40 课内容、未改 `build.py`/`shell.py`/`registry.py`/`quizzes.py`/`check_*.py` 逻辑（重跑 `build.py` 无 diff）。修复确证问题后 amend。
- [ ] **Step 4: 完成分支**：用 superpowers:finishing-a-development-branch，先过验证门，再按既定偏好：本地 `--no-ff` 合并 master + 删分支（无需再询问）。合并后在 master 上复跑全量验证确认干净。**这是全书最后一个里程碑：合并后项目完结。**

---

## 计划自审（writing-plans self-review）

- **Spec 覆盖**：设计 §各组件设计 的 build_print.py / README / LICENSE+LICENSE-CONTENT / ci.yml+deploy.yml -> Task 1/2/3；roadmap 勾选/全量验证/整体复审/完成分支 -> Task 4。两处用户确认（双许可、两份打印 HTML）在 Task 2 Step 1-2 与 Task 1 落实。"quizzes 补全"已核实 40/40 满足，设计与计划均说明无需新增。✓ 无遗漏。
- **占位符扫描**：无 TBD/TODO；Task 1 给了 `build_print.py` 完整可粘贴实现，Task 2 给了 LICENSE/LICENSE-CONTENT 全文与 README 逐节真实要点，Task 3 给了两个完整 workflow YAML。✓
- **类型/命名一致**：产物文件名 `print_zh.html`/`print_en.html` 在设计、Task 1（生成）、Task 3（CI/deploy 构建）、Task 4（验证）四处一致；版权人 `verdenmax` 在 LICENSE 两文件一致；workflow 名 `ci.yml`/`deploy.yml` 一致。`build_print.py` 复用的 `shell.CSS`/`shell.PAGES`/`registry.CONTENT`/`quizzes.render` 均已核实存在。✓
- **风险点**：(1) `git diff --exit-code` 门要求 `print_*.html` 已提交且与源同步——Task 1 Step 5 已 `git add` 产物；(2) CI 装 chromium 在 ubuntu-latest 可能不稳——`print-pdf` job 设 `continue-on-error: true`，不阻断主校验；(3) Pages 首次须手动启用——README + 设计 + Task 3 Step 4 均点名；(4) build_print 嵌入单语裸 HTML 片段——已核实课文正文不带 data-lang 包裹、可被 shell.CSS 直接渲染，打印 CSS 另加分页/避免截断；(5) 本地无法真跑 Actions——YAML 仅本地语法/结构核验，部署效果交所有者在 GitHub 上验证。
