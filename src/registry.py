"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and (later)
build_print.py both import this so the lesson set stays in sync with
shell.PAGES.
"""
import part1

# Filename -> {"zh": ..., "en": ...}. Keep keys in sync with shell.PAGES.
CONTENT = {
    "01-what-is-llamacpp.html": part1.LESSON_01,
    "02-project-map.html": part1.LESSON_02,
}
