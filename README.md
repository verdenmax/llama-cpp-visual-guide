# llama.cpp Visual Guide / llama.cpp 图解学习指南

A visual, bilingual (English + 中文) guide to the internals of [llama.cpp](https://github.com/ggml-org/llama.cpp) - **40 lessons** that take you from "what is llama.cpp" all the way to "how to convert a model and contribute a PR".

> **Disclaimer:** This is **third-party, unofficial** educational material *about* llama.cpp. It contains **no llama.cpp source code**; it explains llama.cpp by quoting small, cited snippets. llama.cpp itself is MIT-licensed by its own authors.

Every lesson is self-contained, embeds both languages (toggle in the page), and uses hand-drawn diagrams, worked-example traces, real (cited) code, and a short self-test quiz.

---

## What it covers

The guide is organized into nine parts that build up step by step:

| Part | Topic | Lessons |
| --- | --- | --- |
| 1 | Overview - what llama.cpp is and why | L01-03 |
| 2 | Foundations - inference basics, tensors, quantization, building | L04-07 |
| 3 | The ggml engine - graphs, operators, GGUF | L08-13 |
| 4 | llama inference internals - loading, the decode loop, KV cache, sampling | L14-24 |
| 5 | Public API & tools - C API, common, llama-cli, llama-server, quantize, bench | L25-30 |
| 6 | Low-level kernels - CPU/CUDA backends, backend dispatch | L31-33 |
| 7 | Advanced topics - speculative decoding, MoE, multimodal, state-space models | L34-37 |
| 8 | Practice & contributing - converting HF models, build/debug/test/contribute | L38-39 |
| 9 | Quick reference - glossary & concept index | L40 |

## How to view

**Online:** published via GitHub Pages at **https://verdenmax.github.io/llama-cpp-visual-guide/**.

**Locally** (zero dependencies, just Python 3):

```bash
cd src
python3 build.py
# then open ../index.html in a browser
```

## How to print / export a PDF

```bash
cd src
python3 build_print.py
# open ../print_zh.html (Chinese) or ../print_en.html (English) in a browser,
# then File -> Print -> Save as PDF (Ctrl/Cmd+P). Each lesson starts on a new page.
```

## Project structure

```
src/            generators + tooling (pure Python 3, no dependencies)
  part1.py .. part9.py   lesson content (bilingual), one module per part
  quizzes.py             per-lesson self-test questions
  shell.py               page shell + the shared CSS
  registry.py            ordered filename -> content map
  build.py               builds index.html + lessons/*.html
  build_print.py         builds print_zh.html + print_en.html
  check_html.py          structural HTML validation
  check_links.py         internal link validation
lessons/        generated lesson pages (committed, kept in sync)
index.html      generated table of contents (committed)
print_*.html    generated print editions (committed)
docs/superpowers/   design specs and implementation plans
```

## Build & validate

```bash
cd src
python3 build.py          # regenerate index.html + lessons/*.html
python3 build_print.py    # regenerate print_zh.html + print_en.html
python3 check_html.py     # structural checks (0 error / 0 warning expected)
python3 check_links.py    # all internal links must resolve
```

The generated HTML is committed and kept in sync with the sources; a re-run of `build.py` should produce no diff.

## License

Dual-licensed:

- **Code** (everything under `src/`) - MIT, see [LICENSE](LICENSE).
- **Content** (the lesson text and diagrams rendered into `index.html`, `lessons/*.html`, `print_*.html`) - CC BY 4.0, see [LICENSE-CONTENT](LICENSE-CONTENT).

---

## 中文说明

这是一份 [llama.cpp](https://github.com/ggml-org/llama.cpp) 内部原理的**图解、双语**学习指南，共 **40 课**，从"llama.cpp 是什么"一路讲到"怎么转换模型、怎么提一个 PR"。

> **声明：** 本项目是**第三方、非官方**的学习材料，**不包含 llama.cpp 源码**，只通过引用少量、标注来源的代码片段来讲解。llama.cpp 本身由其作者以 MIT 许可发布。

每一课都自成一体、内嵌中英双语（页内可切换），用手绘图、worked-example 追踪图、真实（标注来源的）代码和一段自测题来讲清一个概念。

**九个部分**（层层递进）：① 宏观全景（L01-03）② 前置基础（L04-07）③ ggml 引擎（L08-13）④ llama 推理内部（L14-24）⑤ 公共 API 与工具（L25-30）⑥ 底层内核（L31-33）⑦ 进阶专题（L34-37）⑧ 实战与贡献（L38-39）⑨ 速查（L40）。

**怎么看：** 在线版见 **https://verdenmax.github.io/llama-cpp-visual-guide/**；本地零依赖，`cd src && python3 build.py` 后用浏览器打开 `index.html`。

**怎么打印：** `cd src && python3 build_print.py`，再打开 `print_zh.html`（中文）或 `print_en.html`（英文），用 `Ctrl/Cmd+P` 导出 PDF，每课自动分页。

**许可：** 双许可 —— 代码（`src/`）用 MIT（见 LICENSE），教学内容（课程文字与图）用 CC BY 4.0（见 LICENSE-CONTENT）。
