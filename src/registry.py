"""Single source of truth: ordered map of output filename -> bilingual content.

Each value is a dict ``{"zh": html, "en": html}``. build.py and (later)
build_print.py both import this so the lesson set stays in sync with
shell.PAGES.
"""
import part1
import part2
import part3
import part4

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
    "13-gguf-format.html": part3.LESSON_13,
    "14-model-loading.html": part4.LESSON_14,
    "15-architecture-hparams.html": part4.LESSON_15,
    "16-build-graph.html": part4.LESSON_16,
    "17-context-session.html": part4.LESSON_17,
    "18-batching.html": part4.LESSON_18,
    "19-kv-cache.html": part4.LESSON_19,
    "20-vocabulary.html": part4.LESSON_20,
    "21-sampling.html": part4.LESSON_21,
}
