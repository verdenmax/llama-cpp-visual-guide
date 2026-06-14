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
