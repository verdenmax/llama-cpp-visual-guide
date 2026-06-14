"""Per-lesson bilingual self-test (自测题): design-insight multiple-choice + open prompts.

Schema per lesson::

    "NN-file.html": {
        "mcq": [
            {
                "q":   {"zh": "...", "en": "..."},
                "opts": [{"zh": "...", "en": "..."}, ...],
                "answer": 1,                      # 0-based index into opts (as written)
                "why": {"zh": "...", "en": "..."},
            },
        ],
        "open": [{"zh": "...", "en": "..."}],
    }

``render(fname, lang)`` turns it into HTML that build.py appends to the bottom of
each language's lesson body. Options are deterministically shuffled per question
(same permutation for zh and en, so the correct letter matches across languages).

Quiz text (q/opts/why) is raw HTML in a text context (like the lesson body):
write literal ``<``/``&`` as ``&lt;``/``&amp;`` (or wrap code in ``<code>``).
"""
import hashlib

_HEAD = {"zh": "🧪 自测 · 想一想为什么这么设计", "en": "🧪 Self-test - think about the design"}
_SEE = {"zh": "看答案与解析", "en": "Show answer & explanation"}
_CLICK = {"zh": "点击展开", "en": "click to expand"}
_ANS = {"zh": "答案：", "en": "Answer: "}
_SEP = {"zh": "。", "en": ". "}
_OPEN = {
    "zh": "💭 发散思考（没有标准答案，动手或动脑想想）",
    "en": "💭 Open questions (no single right answer - just think or try)",
}


def _shuffle(opts, answer, seed):
    """Deterministically permute opts (stable across builds); return
    (new_opts, new_answer_index) so the correct option lands in a varied slot."""
    order = sorted(
        range(len(opts)),
        key=lambda i: hashlib.md5(f"{seed}:{i}".encode("utf-8")).hexdigest(),
    )
    return [opts[i] for i in order], order.index(answer)


QUIZZES = {
    "01-what-is-llamacpp.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama.cpp 把“推理”从“训练框架”里彻底剥离出来。这个定位选择，最主要换来了什么？",
                    "en": "llama.cpp deliberately separates inference from the training framework. What does that choice mainly buy?",
                },
                "opts": [
                    {"zh": "更高的训练精度", "en": "Higher training accuracy"},
                    {
                        "zh": "甩掉对 Python 生态与大显存的依赖，能在消费级硬件上本地、离线、低成本跑",
                        "en": "Dropping the Python-ecosystem and large-VRAM dependency, so it runs locally/offline/cheaply on consumer hardware",
                    },
                    {"zh": "自动帮你下载模型", "en": "It downloads models for you automatically"},
                    {"zh": "让模型变得更聪明", "en": "It makes the model itself smarter"},
                ],
                "answer": 1,
                "why": {
                    "zh": "只做推理，就能用零依赖 C/C++ + 量化 + 自研 ggml 引擎压掉重依赖——这正是它“一个文件到处跑”的根本原因。",
                    "en": "By doing inference only, it can use zero-dependency C/C++ + quantization + the ggml engine to shed heavy deps - the root reason it 'runs anywhere from a single file'.",
                },
            },
            {
                "q": {
                    "zh": "整体四层结构里，真正“把算子算在硬件上（SIMD / GPU kernel）”的是哪一层？",
                    "en": "In the four-layer structure, which layer actually 'runs the ops on hardware (SIMD / GPU kernels)'?",
                },
                "opts": [
                    {"zh": "工具层 tools/", "en": "tools/ layer"},
                    {"zh": "推理层 src/llama-*", "en": "inference layer src/llama-*"},
                    {"zh": "引擎层 ggml", "en": "the ggml engine layer"},
                    {"zh": "后端层 CPU/CUDA/Metal/Vulkan", "en": "the backend layer CPU/CUDA/Metal/Vulkan"},
                ],
                "answer": 3,
                "why": {
                    "zh": "ggml 负责描述张量与计算图、调度算子；真正落到硬件上的乘加由各“后端”实现。",
                    "en": "ggml describes tensors/graphs and schedules ops; the actual hardware math is implemented by each backend.",
                },
            },
            {
                "q": {
                    "zh": "Q4_0 量化为什么能大幅压缩体积，却几乎不掉精度？",
                    "en": "Why can Q4_0 quantization shrink the size so much yet barely lose accuracy?",
                },
                "opts": [
                    {"zh": "因为它直接丢掉了不重要的网络层", "en": "It simply drops the unimportant network layers"},
                    {
                        "zh": "整块权重共享一个 scale、按小块贴合数值范围，低位宽档位已够用",
                        "en": "A whole block shares one scale and hugs that block's value range, so the low-bit levels are enough",
                    },
                    {"zh": "因为它用 GPU 重新训练了权重", "en": "It retrains the weights on a GPU"},
                    {"zh": "因为模型本来就不需要精度", "en": "Because the model never needed precision anyway"},
                ],
                "answer": 1,
                "why": {
                    "zh": "量化按块共享缩放因子，每块贴合自己的数值范围；权重对微小误差不敏感，4 bit 档位足够，于是省约 4 倍空间而精度几乎不变。进一步的 K-quant 还可选配合重要性矩阵（imatrix）给量化误差加权，让更关键的权重被更精确地保留——位宽不变，只是误差被加权，并非分到更多比特。",
                    "en": "Quantization shares a scale per block, each fitting its own value range; weights tolerate tiny errors and 4-bit levels suffice, so it saves ~4x space with almost no accuracy change. K-quants can optionally pair with an importance matrix (imatrix) that weights the quantization error so important weights are preserved more faithfully - the bit-width is unchanged, the error is weighted rather than bits reallocated.",
                },
            },
        ],
        "open": [
            {
                "zh": "什么场景你会选 llama.cpp，而不是 vLLM？反过来又是什么场景？把你的判断依据写下来。",
                "en": "When would you pick llama.cpp over vLLM, and when the reverse? Write down the criteria you'd use.",
            },
        ],
    },
    "02-project-map.html": {
        "mcq": [
            {
                "q": {
                    "zh": "整个项目对外的公共 C API 主要收在哪里？",
                    "en": "Where does the project's public C API mainly live?",
                },
                "opts": [
                    {"zh": "<code>src/llama.cpp</code>", "en": "<code>src/llama.cpp</code>"},
                    {"zh": "<code>include/llama.h</code>", "en": "<code>include/llama.h</code>"},
                    {"zh": "<code>common/common.h</code>", "en": "<code>common/common.h</code>"},
                    {"zh": "<code>ggml/include/ggml.h</code>", "en": "<code>ggml/include/ggml.h</code>"},
                ],
                "answer": 1,
                "why": {
                    "zh": "对外契约只有 include/llama.h（外加 llama-cpp.h 的 C++ 薄封装）；src 与 ggml 是内部实现。",
                    "en": "The public contract is just include/llama.h (plus the llama-cpp.h C++ wrapper); src and ggml are internal.",
                },
            },
            {
                "q": {
                    "zh": "把一个 HuggingFace 模型变成能被 llama.cpp 运行的文件，靠的是哪部分？",
                    "en": "What turns a HuggingFace model into a file llama.cpp can run?",
                },
                "opts": [
                    {"zh": "<code>tools/llama-quantize</code>", "en": "<code>tools/llama-quantize</code>"},
                    {"zh": "<code>src/llama-model-loader</code>", "en": "<code>src/llama-model-loader</code>"},
                    {
                        "zh": "<code>gguf-py/</code> + <code>convert_*.py</code>（Python 转换脚本）",
                        "en": "<code>gguf-py/</code> + <code>convert_*.py</code> (Python conversion scripts)",
                    },
                    {"zh": "ggml 后端", "en": "the ggml backends"},
                ],
                "answer": 2,
                "why": {
                    "zh": "转换在 Python 侧（gguf-py + convert_*.py）产出 .gguf；C++ 运行时只负责加载已是 GGUF 的文件。",
                    "en": "Conversion happens in Python (gguf-py + convert_*.py) producing .gguf; the C++ runtime only loads already-GGUF files.",
                },
            },
            {
                "q": {
                    "zh": "<code>common/</code> 在整个项目里扮演什么角色？",
                    "en": "What role does <code>common/</code> play in the project?",
                },
                "opts": [
                    {
                        "zh": "推理库本体：模型加载、计算图、采样都在这里",
                        "en": "The inference core: model loading, the compute graph and sampling all live here",
                    },
                    {
                        "zh": "各可执行程序共用的“胶水”（参数解析、采样封装、日志…），推理本体在 <code>src/llama-*</code>",
                        "en": "The shared “glue” for the executables (arg parsing, sampler wrapper, logging...); the inference core is in <code>src/llama-*</code>",
                    },
                    {"zh": "ggml 的一部分，负责底层算子", "en": "Part of ggml, handling the low-level ops"},
                    {"zh": "一组 Python 转换脚本", "en": "A set of Python conversion scripts"},
                ],
                "answer": 1,
                "why": {
                    "zh": "common 把各程序重复要写的胶水（arg、采样封装、日志…）抽出来给 tools/ 复用；真正的加载/计算图/采样在 src/llama-*，对外只经 include/llama.h。",
                    "en": "common factors out the boilerplate the programs repeat (arg, sampler wrapper, logging...) for tools/ to reuse; the real loading/graph/sampling is in src/llama-*, exposed only via include/llama.h.",
                },
            },
        ],
        "open": [
            {
                "zh": "如果要新增一个采样策略，你认为应该改哪个目录？为什么不是 ggml/？",
                "en": "If you were adding a new sampling strategy, which directory would you change - and why not ggml/?",
            },
        ],
    },
    "03-inference-lifecycle.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama_decode 跑一次前向，直接产出的是什么？",
                    "en": "What does a single llama_decode forward pass directly produce?",
                },
                "opts": [
                    {"zh": "最终要显示的文字", "en": "The final text to display"},
                    {"zh": "一个已经选好的 token", "en": "An already-chosen token"},
                    {
                        "zh": "下一个 token 的 logits（分数向量）",
                        "en": "The next token's logits (a score vector)",
                    },
                    {"zh": "更新后的模型权重", "en": "The updated model weights"},
                ],
                "answer": 2,
                "why": {
                    "zh": "decode 只算出 logits；选哪个 token 是采样器 llama_sampler_sample 的事，还原文字是 llama_token_to_piece 的事。",
                    "en": "decode only yields logits; picking a token is llama_sampler_sample's job, turning it into text is llama_token_to_piece's.",
                },
            },
            {
                "q": {
                    "zh": "自回归循环里，为什么每生成一个新 token 不必把整段历史重算一遍？",
                    "en": "In the autoregressive loop, why doesn't each new token require recomputing the whole history?",
                },
                "opts": [
                    {"zh": "因为 prompt 很短", "en": "Because the prompt is short"},
                    {
                        "zh": "因为 KV cache 缓存了过去 token 的 K/V",
                        "en": "Because the KV cache stores past tokens' K/V",
                    },
                    {"zh": "因为用了 GPU", "en": "Because a GPU is used"},
                    {"zh": "因为权重被量化了", "en": "Because the weights are quantized"},
                ],
                "answer": 1,
                "why": {
                    "zh": "prefill 填满 KV cache 后，每步 decode 只算新 token 的 Q 并复用缓存的 K/V，省掉对历史的重复计算。",
                    "en": "After prefill fills the KV cache, each decode step only computes the new token's Q and reuses cached K/V, skipping recomputation over history.",
                },
            },
        ],
        "open": [
            {
                "zh": "如果完全不用 KV cache，生成第 1000 个 token 的成本会怎么变化？这对本地推理意味着什么？",
                "en": "Without any KV cache, how would the cost of generating the 1000th token change - and what would that mean for local inference?",
            },
        ],
    },
}


def render(fname, lang):
    """Return the self-test HTML block for ``fname`` in ``lang`` ('' if none)."""
    data = QUIZZES.get(fname)
    if not data or not (data.get("mcq") or data.get("open")):
        return ""
    out = ['<div class="selftest">', f'<h2>{_HEAD[lang]}</h2>']
    for i, item in enumerate(data.get("mcq", []), 1):
        shuffled, ans = _shuffle(item["opts"], item["answer"], f"{fname}:{i}")
        opts = "\n".join(f"    <li>{o[lang]}</li>" for o in shuffled)
        letter = chr(65 + ans)
        out.append(
            f'<div class="quiz">\n'
            f'  <div class="qn">{i}. {item["q"][lang]}</div>\n'
            f'  <ol class="opts">\n{opts}\n  </ol>\n'
            f'  <details class="accordion">\n'
            f'    <summary>{_SEE[lang]} <span class="hint">{_CLICK[lang]}</span></summary>\n'
            f'    <div class="acc-body"><div class="qa"><div class="a">'
            f'<strong>{_ANS[lang]}{letter}</strong>{_SEP[lang]}{item["why"][lang]}'
            f"</div></div></div>\n"
            f"  </details>\n"
            f"</div>"
        )
    opens = data.get("open", [])
    if opens:
        lis = "\n".join(f"    <li>{o[lang]}</li>" for o in opens)
        out.append(
            '<div class="card spark">\n'
            f'  <div class="tag">{_OPEN[lang]}</div>\n'
            f"  <ul>\n{lis}\n  </ul>\n"
            "</div>"
        )
    out.append("</div>")
    return "\n".join(out)


def _validate():
    """Fail fast on authoring mistakes in QUIZZES (clear message names the lesson)."""
    for fname, data in QUIZZES.items():
        for qi, item in enumerate(data.get("mcq", []), 1):
            opts = item["opts"]
            if not (0 <= item["answer"] < len(opts)):
                raise ValueError(
                    f"quizzes[{fname!r}] Q{qi}: answer {item['answer']} out of range 0..{len(opts) - 1}"
                )
            for o in opts:
                if not ({"zh", "en"} <= o.keys()):
                    raise ValueError(f"quizzes[{fname!r}] Q{qi}: an option is missing zh/en")
            if not ({"zh", "en"} <= item["q"].keys() and {"zh", "en"} <= item["why"].keys()):
                raise ValueError(f"quizzes[{fname!r}] Q{qi}: q/why missing zh/en")
        for oi, o in enumerate(data.get("open", []), 1):
            if not ({"zh", "en"} <= o.keys()):
                raise ValueError(f"quizzes[{fname!r}] open{oi}: missing zh/en")


_validate()
