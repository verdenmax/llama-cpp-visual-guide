# M0 · 脚手架 实施计划（llama.cpp 图解教程）

> **配套 Spec：** `docs/superpowers/specs/2026-06-13-llama-cpp-visual-guide-design.md`
> **路线图：** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`
> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development（推荐）
> 或 superpowers:executing-plans 逐 task 执行。步骤用 `- [ ]` 复选框跟踪。

**Goal:** 搭起纯 Python（零依赖）生成器骨架，跑通"双语设计系统 + 导航 + 1 课样板 + index 目录页 + 结构/死链校验"，
能 `python build.py` 产出可在浏览器 `file://` 打开、可中英切换的最小站点。

**Architecture:** 复用 langchain-visual-guide 架构并做双语化改造：`src/shell.py`（CSS 设计系统 + `PAGES` 双语
元数据 + `page()`/`index_page()` + 语言切换 JS）→ `registry.py`（filename → `{"zh","en"}`）→ `part1.py`（样板课）
→ `build.py`（站点构建）→ `check_html.py`/`check_links.py`（CI 校验）。本 milestone 不含 quizzes/PDF/CI 工作流（留给 M9）。

**Tech Stack:** Python 3 标准库 · 自包含 HTML/CSS/JS（无外部依赖）。

---

## 文件结构（本 milestone 创建/修改）

- Create: `src/shell.py` — 设计系统 CSS、`PAGES`、`FAVICON`、`page()`、`index_page()`、语言切换 JS、外壳双语文案
- Create: `src/registry.py` — `CONTENT` 映射：filename → `{"zh","en"}`
- Create: `src/part1.py` — 样板课内容 `LESSON_01`（双语 dict）
- Create: `src/build.py` — 构建站点（index.html + lessons/）
- Create: `src/check_html.py` — 结构/导航/计数/防漂移校验
- Create: `src/check_links.py` — 内部死链校验
- Create: `.gitignore` — 忽略 `__pycache__/` 等
- 产出（构建生成，提交）：`index.html`、`lessons/01-what-is-llamacpp.html`

> 设计取舍：M0 只放 **1 课样板**（01）跑通全链路；真正的第一部分 3 课内容在 M1 写。
> `part2..part9.py` 等后续 milestone 再建，避免空文件。

---

## Task 1: 项目骨架与 .gitignore

**Files:**
- Create: `.gitignore`
- Create: `src/` 目录（占位，后续任务填充）

- [ ] **Step 1: 创建 .gitignore**

Create `.gitignore`:

```gitignore
__pycache__/
*.pyc
.DS_Store
print-zh.html
print-en.html
*.pdf
.venv/
```

> 说明：`index.html` 与 `lessons/*.html` 是**提交的产物**（GitHub Pages 直接用），不忽略。
> `print-*.html` 与 `*.pdf` 是 PDF 中间/产物（M9 才生成），先忽略。

- [ ] **Step 2: 建 src 目录**

Run:
```bash
mkdir -p src lessons
```
Expected: 目录创建成功（`lessons/` 先占位，build 时会写入）。

- [ ] **Step 3: 验证 git 仓库就绪**

Run:
```bash
git -C . rev-parse --is-inside-work-tree && git status --short
```
Expected: 输出 `true`，且看到 `.gitignore` 为未跟踪文件。

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore for llama.cpp visual guide

Assisted-by: GitHub Copilot"
```

---

## Task 2: 从模板复制 shell.py 作为基底

**Files:**
- Create: `src/shell.py`（复制自 langchain-visual-guide 模板）
- 参考：`/home/verden/course/langchain-visual-guide/src/shell.py`

> 设计取舍：`shell.py` 的 CSS 设计系统（~240 行）几乎可整体复用，只需改主题色与品牌、再叠加双语机制。
> 因此先**原样复制**模板作为基底（本 task），后续 task 再做"换肤"（Task 3）与"双语化"（Task 4-6）。
> 这样每步改动小、可追踪，符合 DRY。

- [ ] **Step 1: 复制模板 shell.py**

Run:
```bash
cp /home/verden/course/langchain-visual-guide/src/shell.py src/shell.py
```
Expected: `src/shell.py` 出现，约 500 行。

- [ ] **Step 2: 验证可被 Python 导入**

Run:
```bash
cd src && python -c "import shell; print('PAGES:', len(shell.PAGES), 'INDEX:', shell.INDEX_FILE)"
```
Expected: 打印 `PAGES: 27 INDEX: index.html`（此刻仍是模板的 27 课，后续 task 会替换为我们的课表）。

- [ ] **Step 3: Commit（vendored 基底）**

```bash
git add src/shell.py
git commit -m "chore: vendor shell.py design system from template as base

Assisted-by: GitHub Copilot"
```

---

## Task 3: 换肤（主题色 + favicon）

**Files:**
- Modify: `src/shell.py`（favicon SVG、`--accent` 调色板、theme-color）

> 取舍：仅改**视觉身份**（绿色 -> llama.cpp 暖橙），不动品牌文案/结构——品牌文案在 Task 5-6 重写
> `page()`/`index_page()` 时一并双语化，避免重复改。主色用暖橙 `#c2630e`（深色模式 `#e8923f`）。

- [ ] **Step 1: 改 favicon（绿色 λ -> 暖橙 "ll"）**

Edit `src/shell.py`，把：
```python
    "<rect width='32' height='32' rx='7' fill='#1a7f64'/>"
    "<text x='16' y='23' font-family='system-ui,sans-serif' font-size='20'"
    " font-weight='700' fill='#fff' text-anchor='middle'>λ</text></svg>"
```
替换为：
```python
    "<rect width='32' height='32' rx='7' fill='#c2630e'/>"
    "<text x='16' y='22' font-family='system-ui,sans-serif' font-size='15'"
    " font-weight='800' fill='#fff' text-anchor='middle'>ll</text></svg>"
```

- [ ] **Step 2: 改浅色调色板 accent 行**

把：
```python
  --accent: #1a7f64; --accent-soft: #e4f3ee; --accent-ink: #0f5c48;
```
替换为：
```python
  --accent: #c2630e; --accent-soft: #fbeede; --accent-ink: #8a4708;
```

- [ ] **Step 3: 改深色调色板 accent 行**

把：
```python
    --accent: #3fb892; --accent-soft: #14302a; --accent-ink: #8ee0c6;
```
替换为：
```python
    --accent: #e8923f; --accent-soft: #3a2410; --accent-ink: #f3b673;
```

- [ ] **Step 4: 改 theme-color meta（绿色 -> 暖橙）**

把 `head_meta()` 内：
```python
        f'<meta name="theme-color" content="#1a7f64">\n'
```
替换为：
```python
        f'<meta name="theme-color" content="#c2630e">\n'
```

- [ ] **Step 5: 验证旧主题色已清除、模块可导入**

Run:
```bash
cd src && ! grep -nE '#1a7f64|#3fb892' shell.py && python -c "import shell; print('ok')"
```
Expected: 无匹配输出（grep 失败即旧色已清除），随后打印 `ok`。

- [ ] **Step 6: Commit**

```bash
git add src/shell.py
git commit -m "style: retheme design system to llama.cpp warm-amber

Assisted-by: GitHub Copilot"
```

---

## Task 4: 双语机制核心（PAGES 5 元组 + bi() + 语言切换 CSS/JS）

**Files:**
- Modify: `src/shell.py`（`PAGES` 改双语 5 元组、新增 `bi()`、追加语言切换 CSS、新增 `LANG_JS`/`LANG_BOOT`）

> 双语方案：`<html data-lang="zh">` 默认中文；CSS 用 `html[data-lang=en] .lang-zh{display:none}` /
> `html[data-lang=zh] .lang-en{display:none}` 控制显隐；顶部按钮 JS 翻转 `data-lang` 并写 `localStorage`；
> `<head>` 内防闪脚本在首屏绘制前读取上次语言。每课内容渲染中/英两份 `div`，外壳文案用 `bi(zh,en)` 双份 span。

- [ ] **Step 1: 把 PAGES 改成双语 5 元组**

把 `shell.py` 里整个 `PAGES = [ ... ]` 列表（27 条模板课，到 `INDEX_FILE = "index.html"` 之前的 `]`）
替换为本 milestone 的课表（M0 仅 1 课样板，后续 milestone 追加）：

```python
# Ordered list of all pages:
# (filename, title_zh, title_en, part_zh, part_en)
PAGES = [
    ("01-what-is-llamacpp.html", "llama.cpp 是什么", "What is llama.cpp",
     "第一部分 · 宏观全景", "Part 1 · The Big Picture"),
]
```

- [ ] **Step 2: 新增 bi() 双语行内助手**

在 `PAGES` 定义之后、`INDEX_FILE = "index.html"` 之前插入：

```python
def bi(zh, en):
    """Inline bilingual pair; only the active language is shown (CSS-controlled)."""
    return f'<span class="lang-zh">{zh}</span><span class="lang-en">{en}</span>'
```

- [ ] **Step 3: 追加语言切换 CSS**

在 `CSS = r"""..."""` 字符串**末尾**（结尾 `"""` 之前，紧接 `.pdf-btn:hover {...}` 那行后）插入：

```css

/* ---- bilingual language switch ---- */
html[data-lang="en"] .lang-zh { display: none !important; }
html[data-lang="zh"] .lang-en { display: none !important; }
.langtoggle { font-size:.72rem; font-weight:700; color:var(--accent-ink);
  background:var(--accent-soft); border:1px solid var(--accent); border-radius:999px;
  padding:.22rem .7rem; cursor:pointer; line-height:1.4; white-space:nowrap; }
.langtoggle:hover { background:var(--accent); color:#fff; }
```

- [ ] **Step 4: 新增 LANG_JS 与 LANG_BOOT 常量**

在 `NAV_SCRIPT = """..."""` 常量定义之后插入：

```python
LANG_JS = """
function lcvgSetLang(l){
  var d=document.documentElement;
  d.dataset.lang=l; d.lang=(l==='en'?'en':'zh-CN');
  try{localStorage.setItem('lcvg-lang',l);}catch(e){}
}
function lcvgToggleLang(){
  lcvgSetLang(document.documentElement.dataset.lang==='en'?'zh':'en');
}
"""

# Runs in <head> before first paint to avoid a flash of the wrong language.
LANG_BOOT = (
    "<script>try{var l=localStorage.getItem('lcvg-lang');"
    "if(l==='en'){document.documentElement.dataset.lang='en';"
    "document.documentElement.lang='en';}}catch(e){}</script>"
)
```

- [ ] **Step 5: 验证可导入且标记就位**

Run:
```bash
cd src && python -c "import shell; print(len(shell.PAGES), shell.bi('中','EN'))" \
  && grep -q 'data-lang="en"' shell.py \
  && grep -q 'lcvgToggleLang' shell.py \
  && echo "markers ok"
```
Expected: 打印 `1 <span class="lang-zh">中</span><span class="lang-en">EN</span>`，随后 `markers ok`。

- [ ] **Step 6: Commit**

```bash
git add src/shell.py
git commit -m "feat: add bilingual switch mechanism (PAGES schema, bi(), lang CSS/JS)

Assisted-by: GitHub Copilot"
```

---

## Task 5: 重写 page() 为双语渲染

**Files:**
- Modify: `src/shell.py`（整体替换 `def page(...)`；并把 `head_meta` 里 og:site_name 改成本项目品牌）

> 取舍：去掉模板里 `standalone`/`data-nav` 的二态分支——本站只产出静态相对链接（`file://` 与 Pages 都适用），
> 更简单。每课渲染 `<div class="lang-zh">` 与 `<div class="lang-en">` 两份正文；外壳文案用 `bi()`；
> `<head>` 注入 `LANG_BOOT` 防闪、页尾注入 `LANG_JS` 切换逻辑。

- [ ] **Step 1: 改 head_meta 的 og:site_name 品牌**

把 `head_meta()` 内：
```python
        f'<meta property="og:site_name" content="LangChain 图解教程">\n'
```
替换为：
```python
        f'<meta property="og:site_name" content="llama.cpp 图解教程">\n'
```

- [ ] **Step 2: 整体替换 page() 函数**

把整个 `def page(...)` 函数（从 `def page(filename, content, standalone=False, home_href=None):`
到它的 `return html`，即 `def index_page(` 之前的全部内容）替换为：

```python
def page(filename, content, home_href="../index.html"):
    """Wrap one lesson's bilingual content in the full HTML shell.

    ``content`` is a dict ``{"zh": html, "en": html}``. Both are emitted; CSS
    shows only the active language. Navigation uses plain relative ``href``
    links so the site works via file:// and any static server (lessons share
    one directory; home defaults to ``../index.html``).
    """
    idx = next(i for i, p in enumerate(PAGES) if p[0] == filename)
    fname, title_zh, title_en, part_zh, part_en = PAGES[idx]
    total = len(PAGES)
    pct = int((idx + 1) / total * 100)
    home = home_href

    if idx > 0:
        p = PAGES[idx - 1]
        prev_link = (
            f'<a class="prev" href="{p[0]}"><div class="dir">{bi("← 上一课", "← Prev")}</div>'
            f'<div class="ttl">{bi(p[1], p[2])}</div></a>'
        )
    else:
        prev_link = (
            f'<a class="prev" href="{home}"><div class="dir">{bi("← 返回", "← Back")}</div>'
            f'<div class="ttl">{bi("目录", "Contents")}</div></a>'
        )
    if idx + 1 < total:
        p = PAGES[idx + 1]
        next_link = (
            f'<a class="next" href="{p[0]}"><div class="dir">{bi("下一课 →", "Next →")}</div>'
            f'<div class="ttl">{bi(p[1], p[2])}</div></a>'
        )
    else:
        next_link = (
            f'<a class="next" href="{home}"><div class="dir">{bi("完成 →", "Done →")}</div>'
            f'<div class="ttl">{bi("返回目录", "Back to index")}</div></a>'
        )

    title_tag = f"{idx+1:02d} · {title_zh} / {title_en} - llama.cpp 图解教程"
    desc = f"{part_zh}｜{title_zh} - llama.cpp 图解教程（中英双语，配真实源码对应、折叠深挖与设计亮点）"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN" data-lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{LANG_BOOT}
<title>{title_tag}</title>
{head_meta(title_tag, desc, og_type="article")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <a class="home" href="{home}">🦙 <b class="lang-zh">llama.cpp 图解教程</b><b class="lang-en">llama.cpp Visual Guide</b></a>
    <span class="pill">{bi(part_zh, part_en)}</span>
    <span class="pill">{idx+1:02d} / {total:02d}</span>
    <button class="langtoggle" onclick="lcvgToggleLang()" aria-label="switch language"><span class="lang-zh">EN</span><span class="lang-en">中</span></button>
  </div>
  <div class="progress"><span style="width:{pct}%"></span></div>
</div>
<div class="wrap">
  <div class="hero">
    <div class="part">{bi(part_zh, part_en)}</div>
    <h1><span class="lang-zh">{title_zh}</span><span class="lang-en">{title_en}</span></h1>
  </div>
  <div class="lang-zh">{content["zh"]}</div>
  <div class="lang-en">{content["en"]}</div>
  <div class="footnav">{prev_link}{next_link}</div>
</div>
<script>{LANG_JS}</script>
</body></html>"""
    return html
```

- [ ] **Step 3: 验证可导入**

Run:
```bash
cd src && python -c "import shell; print('page ok' if callable(shell.page) else 'no')"
```
Expected: 打印 `page ok`（此刻还不能整页渲染，缺 content；下个 task 提供样板课后在 build 中验证）。

- [ ] **Step 4: Commit**

```bash
git add src/shell.py
git commit -m "feat: rewrite page() for bilingual rendering and llama.cpp branding

Assisted-by: GitHub Copilot"
```

---

## Task 6: 重写 index_page() 为双语目录页

**Files:**
- Modify: `src/shell.py`（新增 `SUBTITLES`、整体替换 `def index_page(...)`、微调 `SEARCH_JS` 计数文案）

> 取舍：目录页是所有 milestone 共用基础设施，本 task 一次做完整（双语 TOC 分组 + 搜索 + hero）。
> M0 不放"下载 PDF"按钮（PDF 在 M9），避免现在出现死链。计数 pill 保留中文子串 `共 N 课 · N 个部分`
> 供 `check_html.py` 正则校验。

- [ ] **Step 1: 微调 SEARCH_JS 计数（去掉语言相关后缀）**

把 `SEARCH_JS` 内：
```python
    count.textContent = t ? (n+' \u8bfe') : '';
```
替换为：
```python
    count.textContent = t ? String(n) : '';
```

- [ ] **Step 2: 整体替换 index_page()，并在其前新增 SUBTITLES**

把整个 `def index_page(...)` 函数（从 `def index_page(` 到文件末尾）替换为：

```python
# Per-lesson TOC subtitle: filename -> (zh, en). Missing entries render blank.
SUBTITLES = {
    "01-what-is-llamacpp.html": ("解决什么问题 · 零依赖哲学",
                                 "What problem it solves; zero-dep philosophy"),
}


def index_page(lesson_prefix="lessons/"):
    """Build the bilingual index (table of contents). Always relative links."""
    order = []   # ordered list of (part_zh, part_en)
    groups = {}  # part_zh -> [(num, fname, title_zh, title_en), ...]
    for i, (fname, tz, te, pz, pe) in enumerate(PAGES):
        if pz not in groups:
            groups[pz] = []
            order.append((pz, pe))
        groups[pz].append((i + 1, fname, tz, te))

    blocks = []
    for pz, pe in order:
        blocks.append(f'<div class="toc-part">{bi(pz, pe)}</div>')
        for num, fname, tz, te in groups[pz]:
            sz, se = SUBTITLES.get(fname, ("", ""))
            blocks.append(
                f'<a href="{lesson_prefix}{fname}"><span class="n">{num:02d}</span>'
                f'<span class="tt"><span class="lang-zh">{tz}</span>'
                f'<span class="lang-en">{te}</span></span>'
                f'<span class="ts"><span class="lang-zh">{sz}</span>'
                f'<span class="lang-en">{se}</span></span></a>'
            )
    toc = "\n".join(blocks)
    total = len(PAGES)
    nparts = len(order)

    title_tag = "llama.cpp 图解教程 · 从零理解整个项目 / llama.cpp Visual Guide"
    desc = ("从零理解整个 llama.cpp 项目的中英双语图解教程：宏观结构、用法、ggml 引擎、"
            "llama 推理内部、底层内核，每课配真实源码对应、折叠深挖与设计亮点。")

    return f"""<!DOCTYPE html>
<html lang="zh-CN" data-lang="zh"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{LANG_BOOT}
<title>{title_tag}</title>
{head_meta(title_tag, desc, og_type="website")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <span class="home">🦙 <b class="lang-zh">llama.cpp 图解教程</b><b class="lang-en">llama.cpp Visual Guide</b></span>
    <span class="pill"><span class="lang-zh">共 {total} 课 · {nparts} 个部分</span><span class="lang-en">{total} lessons · {nparts} parts</span></span>
    <button class="langtoggle" onclick="lcvgToggleLang()" aria-label="switch language"><span class="lang-zh">EN</span><span class="lang-en">中</span></button>
  </div>
  <div class="progress"><span style="width:100%"></span></div>
</div>
<div class="wrap">
  <div class="hero index">
    <div class="part">{bi("从零开始 · 面向完全新手", "From scratch · for complete beginners")}</div>
    <h1><span class="lang-zh">用图解理解整个 llama.cpp 项目</span><span class="lang-en">Understand the whole llama.cpp project, visually</span></h1>
    <p class="lead"><span class="lang-zh">这套教程带你<strong>层层深入</strong>：先建立<strong>宏观全景</strong>，再学会<strong>使用</strong>，
    然后深入 <strong>ggml 引擎</strong>与 <strong>llama 推理内部</strong>，最后直抵<strong>底层内核</strong>。每课配真实源码对应、图解与设计亮点。</span>
    <span class="lang-en">A layered tour: build the <strong>big picture</strong> first, learn to <strong>use</strong> it,
    then dive into the <strong>ggml engine</strong> and <strong>llama inference internals</strong>, down to the <strong>low-level kernels</strong>. Every lesson maps to real source, with diagrams and design insights.</span></p>
    <div class="legend">
      <span><i style="background:var(--blue)"></i>{bi("宏观理解", "Big picture")}</span>
      <span><i style="background:var(--purple)"></i>{bi("细节 / 源码", "Details / source")}</span>
      <span><i style="background:var(--amber)"></i>{bi("生活类比", "Analogy")}</span>
      <span><i style="background:var(--accent)"></i>{bi("关键要点", "Key points")}</span>
    </div>
    <p style="margin:.8rem 0 0;color:var(--faint);font-size:.8rem">{bi("📌 对照 llama.cpp 仓库真实源码核实 · 源码引用以“文件 + 符号名”为主（行号随上游更新而变）", "📌 Verified against the real llama.cpp source; references cite file + symbol (line numbers drift upstream)")}</p>
  </div>
  <div class="toc-search">
    <input id="q" type="search" placeholder="🔎 搜索课程 / Search lessons" autocomplete="off" aria-label="search">
    <span class="qcount" id="qcount"></span>
  </div>
  <div class="toc">{toc}</div>
  <div class="toc-empty" id="tocempty">{bi("没有匹配的课程，换个关键词试试。", "No matching lessons, try another keyword.")}</div>
</div>
<script>{LANG_JS}</script>
<script>{SEARCH_JS}</script>
</body></html>"""
```

- [ ] **Step 3: 验证可导入并能渲染 index**

Run:
```bash
cd src && python -c "import shell; h=shell.index_page(); print('toc' in h, 'lcvgToggleLang' in h, '共 1 课' in h)"
```
Expected: 打印 `True True True`。

- [ ] **Step 4: Commit**

```bash
git add src/shell.py
git commit -m "feat: rewrite index_page() as bilingual table of contents

Assisted-by: GitHub Copilot"
```

---

## Task 7: 样板课 part1.py（双语 · 含结构图与源码片段）

**Files:**
- Create: `src/part1.py`（`LESSON_01 = {"zh": ..., "en": ...}`）

> 这是**内容基线样板**：导语 + 生活类比 + 宏观理解 + **整体结构图（layers）** + 对比表 +
> 细节/源码对应（**核实过的最小 C API 片段**）+ 本课要点 + 设计亮点。M1 会在此基础上完善 01 并补 02/03。
> 校验要求：含 `card analogy` 与 `本课要点`（满足 check_html 软检查）；`<pre>` 内 `<`/`&` 必须转义为 `&lt;`/`&amp;`。
> 源码符号已对照 `include/llama.h` 核实：`llama_model_load_from_file` · `llama_init_from_model`
> （`llama_new_context_with_model` 已弃用）· `llama_tokenize` · `llama_batch_get_one` · `llama_decode` ·
> `llama_sampler_sample` · `llama_model_free`（核验日期 2026-06-13）。

- [ ] **Step 1: 创建 src/part1.py**

Create `src/part1.py`:

```python
"""Content for Part 1 (macro overview). M0 ships lesson 01 as the baseline."""

LESSON_01 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
llama.cpp 是一个用<strong>纯 C/C++</strong> 写的<strong>大模型推理引擎</strong>：把已经训练好的大语言模型
（以 <span class="inline">GGUF</span> 格式存放）<strong>高效地跑起来</strong>，在普通 CPU、甚至手机上也能推理，
有 GPU 就更快。它不训练模型，只专注"<strong>把模型跑出字来</strong>"。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把训练好的大模型想成一张<strong>乐谱</strong>（权重）。PyTorch 像<strong>录音棚</strong>：能作曲、能录、设备重。
  llama.cpp 像一台<strong>便携播放器</strong>：不作曲，只<strong>把乐谱高保真地播放出来</strong>，还特别省电、到处能用。
</div>

<h2>它到底解决什么问题</h2>
<p>研究界的模型大多用 Python + PyTorch，依赖重、显存吃紧、难以部署到普通设备。llama.cpp 的目标正相反：</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  用<strong>零外部依赖的 C/C++</strong> + <strong>量化</strong>（把权重压成 4/5/8 bit）+ <strong>自研张量引擎 ggml</strong>，
  让大模型能在<strong>消费级硬件</strong>上本地、离线、低成本地推理。一个可执行文件 + 一个 <span class="inline">.gguf</span> 文件即可运行。
</div>

<h3>整体结构图：四层自底向上</h3>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">工具</span><span class="name">tools/ · examples/</span></div>
    <div class="ld">面向用户：<span class="mono">llama-cli</span> 命令行、<span class="mono">llama-server</span> HTTP 服务、<span class="mono">quantize</span> 量化器</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">推理</span><span class="name">src/llama-*</span></div>
    <div class="ld">模型加载 · 计算图 · KV cache · 采样 · 分词 · 聊天模板（把"模型"变成"会话"）</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">引擎</span><span class="name">ggml</span></div>
    <div class="ld">张量 · 计算图 · 算子（matmul/rope/softmax…）· 后端调度 · 量化格式</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">后端</span><span class="name">CPU · CUDA · Metal · Vulkan …</span></div>
    <div class="ld">把算子真正算在硬件上（SIMD / GPU kernel）</div></div>
</div>

<h2>和 PyTorch / transformers / vLLM 的区别</h2>
<table class="t">
  <tr><th>项目</th><th>定位</th><th>语言 / 依赖</th><th>典型场景</th></tr>
  <tr><td><strong>PyTorch</strong></td><td>训练 + 推理框架</td><td>Python，重</td><td>科研、训练</td></tr>
  <tr><td><strong>transformers</strong></td><td>模型库 / 高层封装</td><td>Python，重</td><td>快速实验</td></tr>
  <tr><td><strong>vLLM</strong></td><td>GPU 高吞吐服务</td><td>Python + CUDA</td><td>云端大并发</td></tr>
  <tr><td><strong>llama.cpp</strong></td><td>轻量本地推理</td><td>C/C++，几乎零依赖</td><td>本地 / 边缘 / 嵌入</td></tr>
</table>

<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  一次最小推理在 C API 里就是这几步（简化自 <span class="inline">include/llama.h</span>，伪代码骨架）：
<pre class="code"><span class="cm">// 简化自 include/llama.h 的最小推理流程</span>
llama_backend_init();

llama_model   *model = <span class="fn">llama_model_load_from_file</span>(<span class="st">"model.gguf"</span>, mparams);
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// 新接口</span>

<span class="cm">// 1) prompt 切成 token</span>
int n = <span class="fn">llama_tokenize</span>(vocab, prompt, /*...*/, tokens, /*...*/);

<span class="cm">// 2) 自回归解码循环</span>
llama_batch batch = <span class="fn">llama_batch_get_one</span>(tokens, n);
<span class="kw">while</span> (generating) {
    <span class="fn">llama_decode</span>(ctx, batch);                 <span class="cm">// 前向：算出下一 token 的 logits</span>
    llama_token id = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1);   <span class="cm">// 采样</span>
    <span class="kw">if</span> (llama_vocab_is_eog(vocab, id)) <span class="kw">break</span>; <span class="cm">// 结束符</span>
    batch = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);      <span class="cm">// 新 token 喂回去</span>
}

<span class="fn">llama_model_free</span>(model);</pre>
  <p style="margin:.5rem 0 0">这条主线（加载 → 分词 → 解码循环 → 采样）就是后面所有课的骨架，第 03 课会展开完整生命周期。</p>
</div>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>llama.cpp = <strong>纯 C/C++ 的大模型推理引擎</strong>，只负责"跑"，不负责"训练"。</li>
    <li>三大支柱：<strong>GGUF 格式</strong> + <strong>量化</strong> + <strong>ggml 张量引擎</strong>。</li>
    <li>整体四层：<strong>后端 → ggml → llama 推理 → 工具</strong>。</li>
    <li>定位：<strong>本地 / 边缘 / 低成本</strong>，对照 PyTorch（训练）、vLLM（云端高并发）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  把"<strong>推理</strong>"从"<strong>训练框架</strong>"里彻底剥离，再用<strong>量化 + 自研引擎</strong>压掉对 Python 生态与大显存的依赖
  ——这就是它能"一个文件到处跑"的根本原因。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
llama.cpp is an <strong>LLM inference engine written in plain C/C++</strong>: it takes an already-trained
model (stored as a <span class="inline">GGUF</span> file) and <strong>runs it efficiently</strong> - on an
ordinary CPU, even a phone, and faster with a GPU. It does not train models; it only focuses on
"<strong>turning a model into text</strong>".
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of a trained model as a <strong>music score</strong> (the weights). PyTorch is the <strong>recording studio</strong>:
  it can compose and record, but it is heavy. llama.cpp is a <strong>portable player</strong>: it does not compose,
  it just <strong>plays the score faithfully</strong> - using little power and running almost anywhere.
</div>

<h2>What problem does it solve</h2>
<p>Research models mostly use Python + PyTorch: heavy dependencies, hungry for VRAM, hard to deploy on
ordinary devices. llama.cpp aims for the opposite:</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  With <strong>zero-dependency C/C++</strong> + <strong>quantization</strong> (compressing weights to 4/5/8 bits) +
  its own tensor engine <strong>ggml</strong>, it makes LLMs run <strong>locally, offline, and cheaply on consumer
  hardware</strong>. One executable plus one <span class="inline">.gguf</span> file is enough.
</div>

<h3>Structure map: four layers, bottom-up</h3>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">tools</span><span class="name">tools/ · examples/</span></div>
    <div class="ld">User-facing: <span class="mono">llama-cli</span>, the <span class="mono">llama-server</span> HTTP service, the <span class="mono">quantize</span> tool</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">infer</span><span class="name">src/llama-*</span></div>
    <div class="ld">Model loading · compute graph · KV cache · sampling · tokenizer · chat templates</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">engine</span><span class="name">ggml</span></div>
    <div class="ld">Tensors · compute graph · ops (matmul/rope/softmax...) · backend scheduling · quant formats</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">backend</span><span class="name">CPU · CUDA · Metal · Vulkan ...</span></div>
    <div class="ld">Actually runs the ops on hardware (SIMD / GPU kernels)</div></div>
</div>

<h2>How it differs from PyTorch / transformers / vLLM</h2>
<table class="t">
  <tr><th>Project</th><th>Role</th><th>Lang / deps</th><th>Typical use</th></tr>
  <tr><td><strong>PyTorch</strong></td><td>Training + inference framework</td><td>Python, heavy</td><td>Research, training</td></tr>
  <tr><td><strong>transformers</strong></td><td>Model hub / high-level wrapper</td><td>Python, heavy</td><td>Fast experiments</td></tr>
  <tr><td><strong>vLLM</strong></td><td>High-throughput GPU serving</td><td>Python + CUDA</td><td>Cloud, high concurrency</td></tr>
  <tr><td><strong>llama.cpp</strong></td><td>Lightweight local inference</td><td>C/C++, near-zero deps</td><td>Local / edge / embedded</td></tr>
</table>

<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A minimal inference run is just these steps in the C API (simplified from
  <span class="inline">include/llama.h</span>, pseudo-code skeleton):
<pre class="code"><span class="cm">// simplified minimal inference flow from include/llama.h</span>
llama_backend_init();

llama_model   *model = <span class="fn">llama_model_load_from_file</span>(<span class="st">"model.gguf"</span>, mparams);
llama_context *ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);  <span class="cm">// new API</span>

<span class="cm">// 1) split the prompt into tokens</span>
int n = <span class="fn">llama_tokenize</span>(vocab, prompt, /*...*/, tokens, /*...*/);

<span class="cm">// 2) autoregressive decode loop</span>
llama_batch batch = <span class="fn">llama_batch_get_one</span>(tokens, n);
<span class="kw">while</span> (generating) {
    <span class="fn">llama_decode</span>(ctx, batch);                 <span class="cm">// forward: logits for the next token</span>
    llama_token id = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1);   <span class="cm">// sample</span>
    <span class="kw">if</span> (llama_vocab_is_eog(vocab, id)) <span class="kw">break</span>; <span class="cm">// end-of-generation</span>
    batch = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);      <span class="cm">// feed the new token back</span>
}

<span class="fn">llama_model_free</span>(model);</pre>
  <p style="margin:.5rem 0 0">This main line (load -> tokenize -> decode loop -> sample) is the skeleton for every later lesson;
  lesson 03 expands it into the full lifecycle.</p>
</div>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>llama.cpp = <strong>an LLM inference engine in plain C/C++</strong> - it only "runs", it does not "train".</li>
    <li>Three pillars: <strong>GGUF format</strong> + <strong>quantization</strong> + <strong>the ggml tensor engine</strong>.</li>
    <li>Four layers: <strong>backend -> ggml -> llama inference -> tools</strong>.</li>
    <li>Niche: <strong>local / edge / low-cost</strong>, vs PyTorch (training) and vLLM (cloud, high concurrency).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  It cleanly separates <strong>inference</strong> from the <strong>training framework</strong>, then uses
  <strong>quantization + a custom engine</strong> to drop the dependency on the Python ecosystem and large VRAM -
  that is why it can "run anywhere from a single file".
</div>
""",
}
```

- [ ] **Step 2: 验证可导入且中英两份非空**

Run:
```bash
cd src && python -c "import part1; d=part1.LESSON_01; print('zh' in d, 'en' in d, len(d['zh'])>500, len(d['en'])>500)"
```
Expected: 打印 `True True True True`。

- [ ] **Step 3: Commit**

```bash
git add src/part1.py
git commit -m "feat: add bilingual sample lesson 01 (content baseline)

Assisted-by: GitHub Copilot"
```

---

## Task 8: registry.py（内容映射）

**Files:**
- Create: `src/registry.py`

> 单一事实源：`CONTENT[filename] = {"zh","en"}`。后续 milestone 只在这里追加映射 + 在 `partN.py` 写内容。

- [ ] **Step 1: 创建 src/registry.py**

Create `src/registry.py`:

```python
"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and (later)
build_print.py both import this so the lesson set stays in sync with
shell.PAGES.
"""
import part1

# Filename -> {"zh": ..., "en": ...}. Keep keys in sync with shell.PAGES.
CONTENT = {
    "01-what-is-llamacpp.html": part1.LESSON_01,
}
```

- [ ] **Step 2: 验证映射键与 PAGES 对齐**

Run:
```bash
cd src && python -c "import shell, registry; ps={p[0] for p in shell.PAGES}; cs=set(registry.CONTENT); print('aligned' if ps==cs else ('PAGES-only:',ps-cs,'CONTENT-only:',cs-ps))"
```
Expected: 打印 `aligned`。

- [ ] **Step 3: Commit**

```bash
git add src/registry.py
git commit -m "feat: add registry mapping filenames to bilingual content

Assisted-by: GitHub Copilot"
```

---

## Task 9: build.py 并首次构建跑通

**Files:**
- Create: `src/build.py`
- 产出（生成并提交）：`index.html`、`lessons/01-what-is-llamacpp.html`

> M0 的 build 不接 quizzes（留给 M9）；每课直接把 `{"zh","en"}` 交给 `shell.page()` 双份渲染。

- [ ] **Step 1: 创建 src/build.py**

Create `src/build.py`:

```python
"""Build the llama.cpp visual guide as a standalone bilingual static site.

Layout produced (relative to project root):

    index.html            entry point (table of contents)
    lessons/NN-*.html     lesson pages (each embeds zh + en; CSS toggles)

Pages use relative links so the site works via file:// or any static server.

Usage:
    cd src && python build.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LESSONS_DIR = os.path.join(ROOT, "lessons")
sys.path.insert(0, HERE)

import shell  # noqa: E402
from registry import CONTENT  # noqa: E402


def build():
    os.makedirs(LESSONS_DIR, exist_ok=True)
    written = []
    for page in shell.PAGES:
        fname = page[0]
        html = shell.page(fname, CONTENT[fname], home_href="../index.html")
        with open(os.path.join(LESSONS_DIR, fname), "w", encoding="utf-8") as f:
            f.write(html)
        written.append(os.path.join("lessons", fname))
    with open(os.path.join(ROOT, shell.INDEX_FILE), "w", encoding="utf-8") as f:
        f.write(shell.index_page(lesson_prefix="lessons/"))
    written.append(shell.INDEX_FILE)
    return written


if __name__ == "__main__":
    done = build()
    print("Wrote", len(done), "files under", ROOT)
    for f in done:
        print("  -", f)
```

- [ ] **Step 2: 运行构建**

Run:
```bash
cd src && python build.py
```
Expected: 打印 `Wrote 2 files under ...`，列出 `lessons/01-what-is-llamacpp.html` 与 `index.html`。

- [ ] **Step 3: 核对产物含双语与切换控件**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide
grep -q 'class="lang-zh"' lessons/01-what-is-llamacpp.html \
 && grep -q 'class="lang-en"' lessons/01-what-is-llamacpp.html \
 && grep -q 'lcvgToggleLang' lessons/01-what-is-llamacpp.html \
 && grep -q 'data-lang="zh"' index.html \
 && echo "build markers ok"
```
Expected: 打印 `build markers ok`。

- [ ] **Step 4: Commit（生成器 + 产物）**

```bash
git add src/build.py index.html lessons/01-what-is-llamacpp.html
git commit -m "feat: add build.py and generate initial bilingual site

Assisted-by: GitHub Copilot"
```

---

## Task 10: check_html.py（双语感知结构校验）

**Files:**
- Create: `src/check_html.py`

> 关键适配：① PAGES 现为 5 元组，part 取 `p[3]`；② 校验每课都含 `lang-zh` 与 `lang-en` 两块；
> ③ 跨课引用 `第 N 课` 按**最终规划的 40 课**判定范围（`MAX_LESSON=40`），使增量里程碑里"前向引用"
> 不被误报；④ 导航链/计数沿用真实 `len(PAGES)`。

- [ ] **Step 1: 创建 src/check_html.py**

Create `src/check_html.py`:

```python
"""Structural / consistency regression guard for the generated HTML.

Run after build.py:
    cd src && python check_html.py

Exits non-zero on any ERROR (used by CI). WARN/INFO print but don't fail.
Checks each lesson + index:
* balanced tags (div/details/table/pre/summary) and details<->summary
* a <title> + meta description; exactly one <h1> per lesson
* both languages present (lang-zh and lang-en blocks)
* no unescaped '<' inside <pre> code blocks
* cross-references "第 N 课" within 1..MAX_LESSON (forward refs allowed)
* nav prev/next chain matches shell.PAGES order
* index TOC lists every page; '共 N 课 · N 个部分' pill matches PAGES
* (WARN) every lesson has a key-points card and an analogy card
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import shell  # noqa: E402

PAGES = shell.PAGES
ORDER = [p[0] for p in PAGES]
TOTAL = len(PAGES)
MAX_LESSON = 40  # planned final lesson count; cross-refs may point forward

PRE_INLINE = ("span", "strong", "b", "em", "u", "a")
SOFT_EXEMPT = {"40-glossary.html"}

issues = []


def add(sev, f, msg):
    issues.append((sev, f, msg))


def check_balance(name, html, tag):
    o = len(re.findall(rf"<{tag}[\s>]", html))
    c = len(re.findall(rf"</{tag}>", html))
    if o != c:
        add("ERR", name, f"<{tag}> unbalanced: {o} open / {c} close")


def check_lesson(fname, html):
    for tag in ("div", "details", "table", "pre", "summary"):
        check_balance(fname, html, tag)
    nd = len(re.findall(r"<details", html))
    ns = len(re.findall(r"<summary", html))
    if nd != ns:
        add("ERR", fname, f"details({nd}) != summary({ns})")
    h1 = len(re.findall(r"<h1", html))
    if h1 != 1:
        add("WARN", fname, f"{h1} <h1> (expected 1)")
    if "<title>" not in html:
        add("ERR", fname, "missing <title>")
    if 'name="description"' not in html:
        add("ERR", fname, "missing meta description")
    if 'class="lang-zh"' not in html:
        add("ERR", fname, "missing lang-zh content")
    if 'class="lang-en"' not in html:
        add("ERR", fname, "missing lang-en content")
    if fname not in SOFT_EXEMPT:
        if "本课要点" not in html and "Key points" not in html:
            add("WARN", fname, "no key-points card")
        if "card analogy" not in html:
            add("WARN", fname, "no analogy card")

    for pre in re.findall(r"<pre[^>]*>(.*?)</pre>", html, re.S):
        cleaned = re.sub(r"</?(?:%s)\b[^>]*>" % "|".join(PRE_INLINE), "", pre)
        if re.search(r"<(?!/)", cleaned):
            m = re.search(r"<(?!/).{0,20}", cleaned)
            add("ERR", fname, f"unescaped '<' in <pre>: {m.group(0)!r}")
            break

    for m in re.finditer(r"第\s*([0-9、,，~\-－\s]+?)\s*课", html):
        nums = [int(x) for x in re.findall(r"[0-9]+", m.group(1))]
        over = [n for n in nums if n == 0 or n > MAX_LESSON]
        if over:
            add("ERR", fname, f"course ref out of range: {m.group(0)!r} -> {over}")

    if fname in ORDER:
        idx = ORDER.index(fname)
        if idx + 1 < TOTAL and f'href="{ORDER[idx + 1]}"' not in html:
            add("ERR", fname, f"next link missing -> {ORDER[idx + 1]}")
        if idx > 0 and f'href="{ORDER[idx - 1]}"' not in html:
            add("ERR", fname, f"prev link missing -> {ORDER[idx - 1]}")


def main():
    for page in PAGES:
        fname = page[0]
        path = os.path.join(ROOT, "lessons", fname)
        if not os.path.exists(path):
            add("ERR", fname, "lesson file missing (run build.py)")
            continue
        check_lesson(fname, open(path, encoding="utf-8").read())

    index_path = os.path.join(ROOT, shell.INDEX_FILE)
    idx = open(index_path, encoding="utf-8").read()
    for page in PAGES:
        fname, tz, te = page[0], page[1], page[2]
        if fname not in idx:
            add("ERR", "index.html", f"TOC missing entry {fname}")
        if tz not in idx:
            add("WARN", "index.html", f"TOC missing zh title {tz!r}")
        if te not in idx:
            add("WARN", "index.html", f"TOC missing en title {te!r}")
    m = re.search(r"共 (\d+) 课 · (\d+) 个部分", idx)
    if m:
        if int(m.group(1)) != TOTAL:
            add("ERR", "index.html", f"count says {m.group(1)} but PAGES has {TOTAL}")
        nparts = len({p[3] for p in PAGES})
        if int(m.group(2)) != nparts:
            add("ERR", "index.html", f"parts says {m.group(2)} but PAGES has {nparts}")
    else:
        add("WARN", "index.html", "could not find '共 N 课 · N 个部分' pill")

    errs = [i for i in issues if i[0] == "ERR"]
    warns = [i for i in issues if i[0] == "WARN"]
    rank = {"ERR": 0, "WARN": 1, "INFO": 2}
    for sev, f, msg in sorted(issues, key=lambda x: rank[x[0]]):
        print(f"  [{sev}] {f}: {msg}")
    print(f"\nChecked {TOTAL} lessons + index - {len(errs)} error(s), {len(warns)} warning(s).")
    if errs:
        print("structural check FAILED")
        return 1
    print("structural check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: 运行结构校验**

Run:
```bash
cd src && python check_html.py
```
Expected: 末行 `structural check passed`，0 error（允许 0 个 warning）。

- [ ] **Step 3: Commit**

```bash
git add src/check_html.py
git commit -m "test: add bilingual-aware structural HTML checker

Assisted-by: GitHub Copilot"
```

---

## Task 11: check_links.py（内部死链校验）

**Files:**
- Create: `src/check_links.py`

> 与模板基本一致，仅把"部署时才生成、源码树里允许缺失"的 PDF 名改成本项目的双语 PDF（M9 才产出）。

- [ ] **Step 1: 创建 src/check_links.py**

Create `src/check_links.py`:

```python
"""Verify every internal relative link in the built site resolves to a file.

Checks index.html and lessons/*.html: each relative href ending in .html must
point to an existing file (resolved relative to the page's directory).
External (http), anchors (#), data: and the generated PDFs are skipped.

Exit code 1 if any broken link is found. No third-party dependencies.

Usage:
    cd src && python check_links.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

HREF_RE = re.compile(r'href="([^"]+)"')
SKIP_PREFIXES = ("http://", "https://", "#", "mailto:", "data:")
# Generated at deploy time (M9); not present in a plain source checkout.
ALLOW_MISSING = {
    "llama-cpp-visual-guide-zh.pdf",
    "llama-cpp-visual-guide-en.pdf",
}


def page_files():
    yield os.path.join(ROOT, "index.html")
    lessons = os.path.join(ROOT, "lessons")
    if os.path.isdir(lessons):
        for name in sorted(os.listdir(lessons)):
            if name.endswith(".html"):
                yield os.path.join(lessons, name)


def check():
    broken = []
    checked = 0
    for path in page_files():
        base = os.path.dirname(path)
        with open(path, encoding="utf-8") as f:
            html = f.read()
        for href in HREF_RE.findall(html):
            if href.startswith(SKIP_PREFIXES):
                continue
            target = href.split("#", 1)[0]
            if not target or target in ALLOW_MISSING:
                continue
            checked += 1
            resolved = os.path.normpath(os.path.join(base, target))
            if not os.path.exists(resolved):
                broken.append((os.path.relpath(path, ROOT), href))
    return checked, broken


if __name__ == "__main__":
    checked, broken = check()
    if broken:
        print(f"{len(broken)} broken link(s):")
        for page, href in broken:
            print(f"  {page} -> {href}")
        sys.exit(1)
    print(f"all {checked} internal links resolve")
```

- [ ] **Step 2: 运行死链校验**

Run:
```bash
cd src && python check_links.py
```
Expected: 打印 `all N internal links resolve`（N >= 2：01 的上/下导航回 index + index 的 TOC 链接）。

- [ ] **Step 3: Commit**

```bash
git add src/check_links.py
git commit -m "test: add internal link checker

Assisted-by: GitHub Copilot"
```

---

## Task 12: M0 验收（干净重建 + 双语自查）

**Files:**
- 无新增；端到端验证 M0 成果。

- [ ] **Step 1: 干净重建并连跑两项校验**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src \
 && python build.py && python check_html.py && python check_links.py
```
Expected: 依次看到 `Wrote 2 files...`、`structural check passed`、`all N internal links resolve`。

- [ ] **Step 2: 确认产物干净（无 src/ 漂移）**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide && git status --short
```
Expected: 无输出（工作区干净；若 `index.html`/`lessons/*` 有改动，说明上次构建未提交，需 `git add` 后重跑校验再提交）。

- [ ] **Step 3: 浏览器自查（人工）**

Run（本地静态预览）：
```bash
cd /home/verden/course/llama-cpp-visual-guide && python -m http.server 8000
```
然后浏览器打开 `http://localhost:8000/`，人工确认：
- 目录页显示 1 课（01），暖橙主题、深浅色随系统；
- 点进 01：进度条、上/下导航、卡片、**整体结构图（四层）**、对比表、代码片段正常；
- 顶部 `EN` 按钮：点一下整页切到英文，再点回中文；**刷新后语言保持**；翻到 index 再进 01 语言仍保持。

> 自查完成后 `Ctrl-C` 结束预览服务器。此步为人工确认，不阻塞自动校验。

- [ ] **Step 4: 标记里程碑完成（更新路线图勾选）**

在 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 的"状态追踪"里，把
`- [ ] M0 脚手架` 改成 `- [x] M0 脚手架`，并把总表 M0 行状态由 `计划中` 改为 `完成`。

Run（提交收尾）：
```bash
cd /home/verden/course/llama-cpp-visual-guide
git add docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md
git commit -m "docs: mark M0 scaffold milestone complete

Assisted-by: GitHub Copilot"
```

---

## 验收标准（Definition of Done · M0）

- `python build.py` 产出 `index.html` + `lessons/01-what-is-llamacpp.html`，可 `file://` 打开。
- `python check_html.py` 0 error；`python check_links.py` 0 死链。
- 每页含 `lang-zh` 与 `lang-en` 两份内容，顶部按钮即时切换、`localStorage` 跨课/刷新保持。
- 主题为 llama.cpp 暖橙、favicon 已换、品牌文案为 llama.cpp。
- 样板课 01 含：生活类比 + 宏观理解 + **整体结构图** + 对比表 + 源码片段（核实过的 C API）+ 本课要点 + 设计亮点。
- `git status` 干净（产物已提交、无 src 漂移）。

---

## Self-Review（plan 作者自审）

**1. Spec 覆盖**：本 M0 对应 Spec §3（架构 + 双语机制 + 换皮）、§5（页面解剖，样板课已含全部卡片类型）、
§8 的 M0 条目。Spec §4 完整课表、§6 内容准则、§7 配套（quiz/PDF/CI）属于 M1-M9，不在 M0 范围，已在路线图登记。

**2. 占位符扫描**：无 TBD/TODO；每个改动步骤均给出完整代码或精确到行的替换。

**3. 类型/命名一致性**：
- `PAGES` 5 元组 `(fname, title_zh, title_en, part_zh, part_en)` 在 `page()`、`index_page()`、`check_html.py`
  三处一致使用（part 取 `p[3]`）。
- `CONTENT[fname]` 为 `{"zh","en"}` dict，`page(filename, content)` 接收同一形状，`build.py` 据此调用。
- 语言开关三件套 `data-lang` / `.lang-zh` / `.lang-en` / `lcvgToggleLang` / `LANG_BOOT` 在 CSS、`page()`、
  `index_page()`、`check_html.py` 中名称一致。
- `bi(zh,en)` 仅用于外壳文案；课程正文用 `<div class="lang-zh">`/`<div class="lang-en">` 包裹。

**4. 歧义检查**：`page()` 去掉了模板的 `standalone`/`data-nav` 分支，统一相对 `href`（已在 Task 5 注明）；
跨课引用范围用 `MAX_LESSON=40` 支持前向引用（Task 10 注明）；M0 不含 quiz/PDF/CI（多处注明留给 M9）。

