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
        if fname not in CONTENT:
            sys.exit(f"build_print error: no registry.CONTENT entry for {fname!r} (declared in shell.PAGES)")
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
