"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and (later)
build_print.py both import this so the lesson set stays in sync with
shell.PAGES.
"""
import part1
import part2
import part3

# Filename -> {"zh": ..., "en": ...}. Keep keys in sync with shell.PAGES.
CONTENT = {
    "01-what-is-llamacpp.html": part1.LESSON_01,
    "02-project-map.html": part1.LESSON_02,
    "03-inference-lifecycle.html": part1.LESSON_03,
    "04-llm-inference-basics.html": part2.LESSON_04,
    "05-tensors.html": part2.LESSON_05,
    "06-quantization-intro.html": part2.LESSON_06,
    "07-build-and-backends.html": part2.LESSON_07,
    "08-ggml-core-objects.html": part3.LESSON_08,
    "09-compute-graph.html": part3.LESSON_09,
    "10-graph-execution.html": part3.LESSON_10,
    "11-core-operators.html": part3.LESSON_11,
    "12-quant-formats.html": part3.LESSON_12,
}
