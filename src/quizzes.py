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
                    "zh": "“一次 llama_decode 前向”在内部大致是怎么完成的？",
                    "en": "Internally, how is a single llama_decode forward pass carried out?",
                },
                "opts": [
                    {"zh": "直接查表返回下一个 token", "en": "It looks the next token up in a table directly"},
                    {
                        "zh": "先建计算图，再交后端执行，最后得到 logits",
                        "en": "It builds a compute graph, runs it on the backend, then yields logits",
                    },
                    {"zh": "在 Python 里跑一遍前向", "en": "It runs a forward pass in Python"},
                    {"zh": "重新加载并量化模型权重", "en": "It reloads and quantizes the model weights"},
                ],
                "answer": 1,
                "why": {
                    "zh": "decode 内部先由 llama-graph.cpp 的 llm_graph_*（经 build_graph）把这步描述成计算图，再交 ggml-backend 调度到硬件执行，算完用 llama_get_logits_ith 取出 logits。",
                    "en": "Inside decode, llm_graph_* in llama-graph.cpp (via build_graph) describes the step as a compute graph, ggml-backend schedules it on hardware, then llama_get_logits_ith reads out the logits.",
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
    "04-llm-inference-basics.html": {
        "mcq": [
            {
                "q": {
                    "zh": "为什么说 KV cache 是“精确优化”而不是“近似”？",
                    "en": "Why is the KV cache an exact optimization rather than an approximation?",
                },
                "opts": [
                    {
                        "zh": "因为因果掩码下旧 token 看不到新 token，它们的 K/V 不随新 token 改变",
                        "en": "Because under the causal mask, old tokens cannot see new tokens, so their K/V never change as new tokens arrive",
                    },
                    {"zh": "因为它把 logits 也缓存了", "en": "Because it also caches the logits"},
                    {"zh": "因为权重被量化了", "en": "Because the weights are quantized"},
                    {"zh": "因为只缓存了最后一层", "en": "Because it only caches the last layer"},
                ],
                "answer": 0,
                "why": {
                    "zh": "因果掩码使旧 token 的 K/V 与未来无关，故缓存它们是精确的、与从头重算逐位相等，不是近似。",
                    "en": "The causal mask makes old tokens' K/V independent of the future, so caching them is exact - bit for bit equal to recomputing - not approximate.",
                },
            },
            {
                "q": {
                    "zh": "decoder-only 模型里，token 之间“互相交流”主要发生在？",
                    "en": "In a decoder-only model, where do tokens mainly \"talk to each other\"?",
                },
                "opts": [
                    {"zh": "带因果掩码的自注意力层", "en": "The self-attention layer (with the causal mask)"},
                    {"zh": "FFN（前馈层）", "en": "The FFN (feed-forward layer)"},
                    {"zh": "RMSNorm", "en": "RMSNorm"},
                    {"zh": "输出投影", "en": "The output projection"},
                ],
                "answer": 0,
                "why": {
                    "zh": "FFN、归一化、投影都对每个 token 独立处理，只有注意力让 token 互相看。",
                    "en": "FFN, norm, and projection all process each token independently; only attention lets tokens see each other.",
                },
            },
            {
                "q": {
                    "zh": "decode 时为什么用 llama_get_logits_ith(ctx, -1) 只取最后一个位置？",
                    "en": "During decode, why does llama_get_logits_ith(ctx, -1) read only the last position?",
                },
                "opts": [
                    {
                        "zh": "因为预测“下一个 token”只需要最后一个位置的输出，其余位置不用算输出",
                        "en": "Because predicting the next token needs only the last position's output; other positions need no output",
                    },
                    {"zh": "因为前面的位置算错了", "en": "Because the earlier positions were computed wrong"},
                    {"zh": "因为 -1 表示取平均", "en": "Because -1 means take the average"},
                    {"zh": "因为只有最后一层有 logits", "en": "Because only the last layer has logits"},
                ],
                "answer": 0,
                "why": {
                    "zh": "自回归每步只产出一个新 token，对应的就是序列最后一个位置那一行 logits。",
                    "en": "Autoregression emits one new token per step, which corresponds to the logits row at the sequence's last position.",
                },
            },
        ],
        "open": [
            {
                "zh": "如果去掉因果掩码（允许看到未来 token），自回归生成还成立吗？KV cache 还精确吗？",
                "en": "If you removed the causal mask (letting tokens see the future), would autoregressive generation still hold? Would the KV cache still be exact?",
            },
        ],
    },
    "05-tensors.html": {
        "mcq": [
            {
                "q": {
                    "zh": "在 ggml 里把一个张量“转置”，主要改变了什么？",
                    "en": "Transposing a ggml tensor mainly changes what?",
                },
                "opts": [
                    {
                        "zh": "只交换 ne[]/nb[]（步长），复用同一块 data，不搬数据",
                        "en": "Only swaps ne[]/nb[] (strides), reusing the same data - no data is moved",
                    },
                    {"zh": "复制出一块新内存", "en": "Copies out a new block of memory"},
                    {"zh": "改变了 type", "en": "Changes the type"},
                    {"zh": "重新量化了权重", "en": "Re-quantizes the weights"},
                ],
                "answer": 0,
                "why": {
                    "zh": "转置是改元数据的视图操作：交换 ne/nb、用 view_src 指回原张量，data 一个字节都不动，所以零拷贝。",
                    "en": "Transpose is a metadata-only view: swap ne/nb, point view_src back to the original, and data is untouched - hence zero-copy.",
                },
            },
            {
                "q": {
                    "zh": "ggml 张量里哪一维是“最内/连续（步长最小）”的维？",
                    "en": "Which dimension of a ggml tensor is the innermost/contiguous (smallest stride) one?",
                },
                "opts": [
                    {"zh": "ne[0]", "en": "ne[0]"},
                    {"zh": "ne[3]", "en": "ne[3]"},
                    {"zh": "由 type 决定", "en": "Decided by the type"},
                    {"zh": "都一样", "en": "All the same"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ggml 行优先，nb[0]=ggml_type_size(type) 最小，ne[0] 在内存里连续摆放；这和 numpy/PyTorch 的约定相反。",
                    "en": "ggml is row-major: nb[0]=ggml_type_size(type) is smallest and ne[0] is laid out contiguously - the opposite of numpy/PyTorch.",
                },
            },
            {
                "q": {
                    "zh": "为什么量化类型（如 Q4_0）的张量不能像普通数组那样按单个元素下标随便取？",
                    "en": "Why can't a quantized-type tensor (e.g. Q4_0) be indexed element-by-element like a plain array?",
                },
                "opts": [
                    {
                        "zh": "因为它按“块”打包存储，单个权重无法独立寻址，要按块解量化",
                        "en": "Because it packs values by block, so a single weight is not independently addressable - you dequantize by block",
                    },
                    {"zh": "因为它没有 ne 字段", "en": "Because it has no ne field"},
                    {"zh": "因为它的 data 是空的", "en": "Because its data is empty"},
                    {"zh": "因为它只能有一维", "en": "Because it can only be one-dimensional"},
                ],
                "answer": 0,
                "why": {
                    "zh": "量化类型把一整块（如 32 个权重）压成定长字节，nb 公式里的 ne[0]/ggml_blck_size(type) 正是“一排有多少块”。",
                    "en": "A quantized type packs a whole block (e.g. 32 weights) into fixed bytes; the ne[0]/ggml_blck_size(type) in the nb formula is exactly 'blocks per row'.",
                },
            },
        ],
        "open": [
            {
                "zh": "给一个 ne=[4,3]（ne[0]=4）的 F32 张量，nb[0] 与 nb[1] 各是多少字节？（F32 = 4 字节）",
                "en": "For an F32 tensor with ne=[4,3] (ne[0]=4), what are nb[0] and nb[1] in bytes? (F32 = 4 bytes)",
            },
        ],
    },
    "06-quantization-intro.html": {
        "mcq": [
            {
                "q": {
                    "zh": "块量化（每块一个 scale）为什么比“整个张量共用一个 scale”更准？",
                    "en": "Why is block quantization (one scale per block) more accurate than one scale for the whole tensor?",
                },
                "opts": [
                    {
                        "zh": "每块自带 scale，块内动态范围更小，近似误差更小",
                        "en": "Each block has its own scale, so its dynamic range is smaller and the approximation error is smaller",
                    },
                    {"zh": "因为用了更多 bit", "en": "Because it uses more bits"},
                    {"zh": "因为压缩率更高", "en": "Because the compression ratio is higher"},
                    {"zh": "因为完全不丢数据", "en": "Because it loses no data at all"},
                ],
                "answer": 0,
                "why": {
                    "zh": "全局一个 scale 没法兼顾大值和小值；切成小块、每块各配 scale，局部范围小、贴得准，误差自然更小。",
                    "en": "One global scale cannot serve large and small values at once; per-block scales fit local ranges tightly, so error shrinks.",
                },
            },
            {
                "q": {
                    "zh": "Q4_0 每个权重平均约几 bit？（每块 32 权重 = 2 字节 scale + 16 字节量化值）",
                    "en": "About how many bits per weight does Q4_0 use? (per 32-weight block = 2-byte scale + 16 bytes of quants)",
                },
                "opts": [
                    {"zh": "约 4.5 bit", "en": "About 4.5 bit"},
                    {"zh": "正好 4 bit", "en": "Exactly 4 bit"},
                    {"zh": "8 bit", "en": "8 bit"},
                    {"zh": "2 bit", "en": "2 bit"},
                ],
                "answer": 0,
                "why": {
                    "zh": "(2 + 16) 字节 × 8 / 32 = 4.5 bit；多出的 0.5 bit 是每块都要分摊的那个 scale。",
                    "en": "(2 + 16) bytes x 8 / 32 = 4.5 bit; the extra 0.5 bit is the per-block scale amortized over the block.",
                },
            },
            {
                "q": {
                    "zh": "imatrix（重要性矩阵）在量化里起什么作用？",
                    "en": "What role does imatrix (importance matrix) play in quantization?",
                },
                "opts": [
                    {
                        "zh": "用校准数据按权重的重要性来加权量化误差，让重要权重更准；不改变位宽分配",
                        "en": "It uses calibration data to weight quantization error by weight importance, keeping important weights more accurate; it does not change bit-width allocation",
                    },
                    {"zh": "决定每个权重用几 bit", "en": "It decides how many bits each weight gets"},
                    {"zh": "它本身是一种新的量化格式", "en": "It is itself a new quantization format"},
                    {"zh": "它给模型增加显存占用", "en": "It increases the model's memory footprint"},
                ],
                "answer": 0,
                "why": {
                    "zh": "imatrix 改变的是“误差怎么分配”，让精度用在重要权重上；位宽由量化档位（如 Q4_K）决定，与 imatrix 无关。",
                    "en": "imatrix changes how error is distributed, spending precision on important weights; bit-width is set by the quant tier (e.g. Q4_K), independent of imatrix.",
                },
            },
        ],
        "open": [
            {
                "zh": "同样压到约 4bit，为什么 Q4_K 通常比 Q4_0 困惑度更低？（提示：super-block 与子块 scale）",
                "en": "At roughly 4 bits, why does Q4_K usually have lower perplexity than Q4_0? (hint: super-block and per-sub-block scales)",
            },
        ],
    },
    "07-build-and-backends.html": {
        "mcq": [
            {
                "q": {
                    "zh": "想编一个支持 NVIDIA GPU 的 llama.cpp，应该怎么做？",
                    "en": "How do you build a llama.cpp with NVIDIA GPU support?",
                },
                "opts": [
                    {"zh": "配置时加 cmake -B build -DGGML_CUDA=ON", "en": "Configure with cmake -B build -DGGML_CUDA=ON"},
                    {"zh": "pip install cuda", "en": "pip install cuda"},
                    {"zh": "运行时加 --gpu 参数", "en": "Add a --gpu flag at runtime"},
                    {"zh": "手动改源码里的 if", "en": "Manually edit the ifs in the source"},
                ],
                "answer": 0,
                "why": {
                    "zh": "后端是编译期的 CMake 开关：配置时用 -DGGML_CUDA=ON 把 CUDA 后端编进去，再 cmake --build。",
                    "en": "Backends are compile-time CMake switches: configure with -DGGML_CUDA=ON to compile in the CUDA backend, then cmake --build.",
                },
            },
            {
                "q": {
                    "zh": "ggml 的“后端”(CPU/CUDA/Metal/...)主要解决什么问题？",
                    "en": "What does ggml's \"backend\" layer (CPU/CUDA/Metal/...) mainly solve?",
                },
                "opts": [
                    {
                        "zh": "让同一张计算图能派发到不同硬件执行，把“算什么”和“在哪算”解耦",
                        "en": "It lets the same compute graph dispatch to different hardware, decoupling \"what to compute\" from \"where\"",
                    },
                    {"zh": "决定模型的精度", "en": "It decides the model's precision"},
                    {"zh": "负责量化权重", "en": "It quantizes the weights"},
                    {"zh": "解析 GGUF 文件", "en": "It parses GGUF files"},
                ],
                "answer": 0,
                "why": {
                    "zh": "后端是“执行层”的统一抽象：上层只描述计算图，具体在 CPU/GPU 上怎么算交给各后端实现。",
                    "en": "The backend is a uniform \"execution layer\": the upper layer only describes the graph; each backend implements how to compute it on CPU/GPU.",
                },
            },
            {
                "q": {
                    "zh": "运行时的 -ngl 参数是干什么的？",
                    "en": "What does the runtime -ngl flag do?",
                },
                "opts": [
                    {
                        "zh": "决定把模型多少层卸载到 GPU 上算（其余留在 CPU）",
                        "en": "It decides how many model layers to offload to the GPU (the rest stay on CPU)",
                    },
                    {"zh": "选择量化档位", "en": "It selects the quantization tier"},
                    {"zh": "设置线程数", "en": "It sets the thread count"},
                    {"zh": "设置上下文长度", "en": "It sets the context length"},
                ],
                "answer": 0,
                "why": {
                    "zh": "-ngl（n-gpu-layers）控制 GPU 层卸载：显存够就多放，不够就 CPU/GPU 混合跑。",
                    "en": "-ngl (n-gpu-layers) controls GPU layer offload: put more if VRAM allows, otherwise run CPU/GPU mixed.",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么“选哪些后端”是编译期开关，而不是运行时把所有后端都带上？（提示：依赖、体积、零依赖哲学）",
                "en": "Why are backends a compile-time switch rather than bundling all of them at runtime? (hint: dependencies, size, the zero-dependency philosophy)",
            },
        ],
    },
    "08-ggml-core-objects.html": {
        "mcq": [
            {
                "q": {
                    "zh": "ggml 会为每个张量都单独 malloc 一次内存吗？",
                    "en": "Does ggml malloc memory separately for every single tensor?",
                },
                "opts": [
                    {
                        "zh": "不会，张量从 ggml_context 预分配的 arena 里 bump 切出，不做 per-tensor malloc",
                        "en": "No - tensors are bump-carved from the arena pre-allocated by ggml_context, with no per-tensor malloc",
                    },
                    {"zh": "会，每建一个张量就 malloc 一次", "en": "Yes, it mallocs once per tensor created"},
                    {"zh": "用垃圾回收器自动管理", "en": "It uses a garbage collector"},
                    {"zh": "把每个张量都存到磁盘", "en": "It stores each tensor to disk"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ggml_init 一次性备好一块 arena，ggml_new_object 在里面往后推游标就地放下；元数据和数据都从这块池子切，避免成千上万次 malloc。",
                    "en": "ggml_init prepares one arena up front; ggml_new_object bumps a cursor and places in situ. Metadata and data are both carved from this pool, avoiding thousands of mallocs.",
                },
            },
            {
                "q": {
                    "zh": "ggml_init_params 里 no_alloc=true 意味着什么？",
                    "en": "What does no_alloc=true in ggml_init_params mean?",
                },
                "opts": [
                    {
                        "zh": "只分配张量元数据、不分配数据缓冲，为“先建图、后由后端分配”铺路",
                        "en": "Allocate tensor metadata only, no data buffer - paving the way for \"build the graph first, let the backend allocate later\"",
                    },
                    {"zh": "什么都不分配", "en": "Allocate nothing at all"},
                    {"zh": "关闭量化", "en": "Turn off quantization"},
                    {"zh": "把 context 设为只读", "en": "Make the context read-only"},
                ],
                "answer": 0,
                "why": {
                    "zh": "建计算图只需要形状和依赖，用不到真正的数据内存；no_alloc 让 ctx 只存元数据，数据等图建好后由后端统一分配（L10）。",
                    "en": "Building a graph needs only shapes and dependencies, not real data memory; no_alloc keeps the ctx metadata-only, and the backend allocates data once the graph is built (L10).",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么 arena 满了 ggml 选择直接 abort，而不是自动扩容？这对使用者提出了什么要求？",
                "en": "Why does ggml abort outright when the arena is full instead of auto-growing? What does that demand of the user?",
            },
        ],
    },
    "09-compute-graph.html": {
        "mcq": [
            {
                "q": {
                    "zh": "调用 c = ggml_mul_mat(a, b) 时发生了什么？",
                    "en": "What happens when you call c = ggml_mul_mat(a, b)?",
                },
                "opts": [
                    {
                        "zh": "新建一个结果张量并记下 op=MUL_MAT、src=[a, b]，但不做任何乘法",
                        "en": "It builds a result tensor and records op=MUL_MAT, src=[a, b], but does no multiplication",
                    },
                    {"zh": "立刻算出矩阵乘的结果数字", "en": "It immediately computes the matmul result numbers"},
                    {"zh": "修改了 a 的内容", "en": "It modifies the contents of a"},
                    {"zh": "把结果写到磁盘", "en": "It writes the result to disk"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ggml 是惰性建图：算子函数只建结果张量、填 op/src 反向指针，真正的运算留到执行阶段（下一课）。",
                    "en": "ggml builds graphs lazily: an operator function only builds the result tensor and fills op/src back-pointers; the real math waits for execution (next lesson).",
                },
            },
            {
                "q": {
                    "zh": "计算图里 leafs 和 nodes 的区别是？",
                    "en": "What is the difference between leafs and nodes in the compute graph?",
                },
                "opts": [
                    {
                        "zh": "leafs 是输入/权重/常量（op==NONE），nodes 是算子结果（按拓扑序计算）",
                        "en": "leafs are inputs/weights/constants (op==NONE); nodes are operator results (computed in topological order)",
                    },
                    {"zh": "leafs 是输出，nodes 是输入", "en": "leafs are outputs, nodes are inputs"},
                    {"zh": "两者没有区别", "en": "There is no difference"},
                    {"zh": "nodes 是叶子，leafs 是树枝", "en": "nodes are leaves, leafs are branches"},
                ],
                "answer": 0,
                "why": {
                    "zh": "判据是有没有“来历”：op==NONE 的是叶子（输入/常量，直接用其数据），有 op 的是节点（要按依赖顺序算出来）。",
                    "en": "The criterion is whether it has an origin: op==NONE means a leaf (input/constant, used directly); having an op means a node (computed in dependency order).",
                },
            },
            {
                "q": {
                    "zh": "ggml_build_forward_expand 从输出张量出发做了什么？",
                    "en": "What does ggml_build_forward_expand do, starting from the output tensor?",
                },
                "opts": [
                    {
                        "zh": "沿 src 指针递归回溯，把所有依赖按拓扑序收进图，保证执行时输入先于输出算好",
                        "en": "It recurses back along src pointers, collecting all dependencies in topological order so inputs are computed before outputs at execution",
                    },
                    {"zh": "立刻执行整张图", "en": "It immediately executes the whole graph"},
                    {"zh": "把图保存成 GGUF 文件", "en": "It saves the graph to a GGUF file"},
                    {"zh": "随机打乱节点顺序", "en": "It randomly shuffles the node order"},
                ],
                "answer": 0,
                "why": {
                    "zh": "“先递归收集依赖、再放自己”天然产生拓扑序：排在前面的不依赖后面的，执行时从头算到尾即可。",
                    "en": "\"Recurse to collect dependencies first, then add itself\" naturally yields topological order: earlier never depends on later, so execution just goes front to back.",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么 ggml 要“先建图、后执行”，而不是边调用边算？至少说出两个好处。",
                "en": "Why does ggml \"build the graph first, execute later\" instead of computing as it goes? Name at least two benefits.",
            },
        ],
    },
    "10-graph-execution.html": {
        "mcq": [
            {
                "q": {
                    "zh": "ggml-alloc 为什么能大幅复用内存、压低峰值？",
                    "en": "Why can ggml-alloc reuse memory heavily and crush the peak?",
                },
                "opts": [
                    {
                        "zh": "因为惰性建图提供了完整的图，能预知每个张量的生命周期，用完即归还供后续复用",
                        "en": "Because lazy building gives the complete graph, so it foresees each tensor's lifetime and returns memory once done for later reuse",
                    },
                    {"zh": "因为用了量化", "en": "Because it uses quantization"},
                    {"zh": "因为内存很便宜", "en": "Because memory is cheap"},
                    {"zh": "因为它不存任何中间结果", "en": "Because it stores no intermediate results"},
                ],
                "answer": 0,
                "why": {
                    "zh": "有了完整的图才能算出每个张量“最后一次被用”在哪，过了那点立刻回收；best-fit 复用 + 合并空闲块把峰值压到很低。",
                    "en": "Only the complete graph lets it compute where each tensor is last used; past that it reclaims immediately, and best-fit reuse + merging free blocks crushes the peak.",
                },
            },
            {
                "q": {
                    "zh": "ggml_backend_sched 主要负责什么？",
                    "en": "What is ggml_backend_sched mainly responsible for?",
                },
                "opts": [
                    {
                        "zh": "把一张图拆开、把算子指派到合适的后端设备，并在设备间拷贝张量",
                        "en": "Splitting a graph, assigning operators to suitable backend devices, and copying tensors between devices",
                    },
                    {"zh": "解析 GGUF 文件", "en": "Parsing GGUF files"},
                    {"zh": "量化权重", "en": "Quantizing weights"},
                    {"zh": "决定采样策略", "en": "Deciding the sampling strategy"},
                ],
                "answer": 0,
                "why": {
                    "zh": "sched 是“包工头”：拆图、按设备指派算子、在 CPU/GPU 边界自动插入拷贝——这正是 -ngl 把部分层放 GPU 的底层机制。",
                    "en": "sched is the \"general contractor\": split, assign operators by device, and auto-insert copies at CPU/GPU boundaries - the mechanism behind -ngl putting some layers on GPU.",
                },
            },
            {
                "q": {
                    "zh": "ggml_backend_graph_compute 执行一张图时，叶子（权重、输入）会被计算吗？",
                    "en": "When ggml_backend_graph_compute runs a graph, are leafs (weights, inputs) computed?",
                },
                "opts": [
                    {
                        "zh": "不会，叶子是现成的数据，只被算子读取；只有节点（算子结果）才按拓扑序逐个计算",
                        "en": "No - leafs are ready-made data, only read by operators; only nodes (operator results) are computed one by one in topological order",
                    },
                    {"zh": "会，每个张量都要算一遍", "en": "Yes, every tensor is computed"},
                    {"zh": "只算叶子，不算节点", "en": "Only leafs are computed, not nodes"},
                    {"zh": "随机选一半来算", "en": "A random half is computed"},
                ],
                "answer": 0,
                "why": {
                    "zh": "叶子是权重/输入等现成数据，执行时只被读取；执行引擎只对节点从头到尾算一遍。",
                    "en": "Leafs are ready-made data like weights/inputs, only read at execution; the engine computes only the nodes, front to back.",
                },
            },
        ],
        "open": [
            {
                "zh": "把模型一半层放 GPU、一半留 CPU（-ngl 设一半）时，ggml_backend_sched 在背后大概做了哪些事？",
                "en": "When you put half the layers on GPU and keep half on CPU (-ngl set to half), what does ggml_backend_sched roughly do behind the scenes?",
            },
        ],
    },
    "11-core-operators.html": {
        "mcq": [
            {
                "q": {
                    "zh": "ggml_mul_mat(a, b) 对形状的核心要求是？",
                    "en": "What is the core shape requirement of ggml_mul_mat(a, b)?",
                },
                "opts": [
                    {
                        "zh": "a 和 b 的内维 ne[0] 必须相等（这一维在相乘时被消去）",
                        "en": "a and b must have equal inner dim ne[0] (this dim is eliminated in the multiply)",
                    },
                    {"zh": "a 和 b 形状必须完全相同", "en": "a and b must have identical shapes"},
                    {"zh": "b 必须是方阵", "en": "b must be square"},
                    {"zh": "没有任何要求", "en": "There is no requirement"},
                ],
                "answer": 0,
                "why": {
                    "zh": "mul_mat 要求 a->ne[0]==b->ne[0]；结果 ne={a.ne[1], b.ne[1], ...}。因 ggml 行优先、ne[0] 最内，这条规则方向和数学“行×列”相反。",
                    "en": "mul_mat requires a->ne[0]==b->ne[0]; result ne={a.ne[1], b.ne[1], ...}. Since ggml is row-major with ne[0] innermost, this reads reversed from math's rows x columns.",
                },
            },
            {
                "q": {
                    "zh": "soft_max_ext 里的 mask 起什么作用？",
                    "en": "What does the mask in soft_max_ext do?",
                },
                "opts": [
                    {
                        "zh": "给分数加上掩码，例如把未来位置设为 -inf 以实现因果掩码",
                        "en": "Adds a mask to the scores, e.g. setting future positions to -inf to implement the causal mask",
                    },
                    {"zh": "对输入做归一化", "en": "Normalizes the input"},
                    {"zh": "把权重量化", "en": "Quantizes the weights"},
                    {"zh": "缩放学习率", "en": "Scales the learning rate"},
                ],
                "answer": 0,
                "why": {
                    "zh": "soft_max_ext 融合 softmax(a*scale + mask)：scale 通常 1/sqrt(d) 防止分数过大，mask 加因果掩码（未来位 -inf，softmax 后权重为 0）。",
                    "en": "soft_max_ext fuses softmax(a*scale + mask): scale is usually 1/sqrt(d) to keep scores in range, mask adds the causal mask (future at -inf, weight 0 after softmax).",
                },
            },
            {
                "q": {
                    "zh": "为什么说一个 ggml 算子有“两处代码”？",
                    "en": "Why is a ggml operator said to have \"two pieces of code\"?",
                },
                "opts": [
                    {
                        "zh": "一处在 ggml.c 建图、定 op/src 与输出形状；另一处在后端（如 ggml-cpu 的 compute_forward）真正计算",
                        "en": "One in ggml.c builds the graph, defining op/src and output shape; the other in a backend (e.g. ggml-cpu's compute_forward) actually computes",
                    },
                    {"zh": "调试版和发布版", "en": "Debug and release builds"},
                    {"zh": "前端网页和后端服务器", "en": "Frontend web page and backend server"},
                    {"zh": "训练和推理", "en": "Training and inference"},
                ],
                "answer": 0,
                "why": {
                    "zh": "建图侧（ggml.c）只定形状、填 op/src，不算；计算侧（各后端的 ggml_compute_forward_*）真算。两者靠 enum ggml_op + switch(op) 对接。",
                    "en": "The build side (ggml.c) only defines the shape and fills op/src, no compute; the compute side (each backend's ggml_compute_forward_*) actually computes. They meet via enum ggml_op + switch(op).",
                },
            },
            {
                "q": {
                    "zh": "执行时，后端怎么知道每个节点该调哪个算子实现？",
                    "en": "At execution, how does the backend know which operator implementation to call for each node?",
                },
                "opts": [
                    {
                        "zh": "用一个大 switch(node->op) 按算子编号派发到对应的 ggml_compute_forward_*",
                        "en": "A big switch(node->op) dispatches by operator number to the matching ggml_compute_forward_*",
                    },
                    {"zh": "靠文件名匹配", "en": "By matching file names"},
                    {"zh": "随机选一个", "en": "It picks one at random"},
                    {"zh": "每次都重新编译", "en": "It recompiles each time"},
                ],
                "answer": 0,
                "why": {
                    "zh": "建图时算子编号记在 tensor->op；执行时后端用 switch(op) 跳到对应实现。加新算子 = 加一个 enum 值 + 写一份 forward 并接进 switch。",
                    "en": "The operator number is recorded in tensor->op at build; at execution the backend uses switch(op) to jump to the implementation. Adding an operator = add an enum value + write a forward wired into the switch.",
                },
            },
        ],
        "open": [
            {
                "zh": "ggml 的 mul_mat 形状规则为什么读起来和你在数学课学的“行×列”方向相反？（提示：L05 的维度顺序）",
                "en": "Why does ggml's mul_mat shape rule read reversed from the \"rows x columns\" you learned in math? (hint: L05's dimension order)",
            },
        ],
    },
    "12-quant-formats.html": {
        "mcq": [
            {
                "q": {
                    "zh": "q4_0 的一块 18 字节是怎么构成的？",
                    "en": "How are the 18 bytes of a q4_0 block made up?",
                },
                "opts": [
                    {
                        "zh": "2 字节的 half scale + 16 字节装 32 个 4-bit 量化值",
                        "en": "a 2-byte half scale + 16 bytes holding 32 4-bit quantized values",
                    },
                    {"zh": "18 个权重，每个 1 字节", "en": "18 weights, one byte each"},
                    {"zh": "16 字节 scale + 2 字节量化值", "en": "16 bytes of scale + 2 bytes of values"},
                    {"zh": "全是 int8，没有 scale", "en": "all int8, no scale"},
                ],
                "answer": 0,
                "why": {
                    "zh": "block_q4_0 = {ggml_half d; uint8_t qs[16]}：2 字节 scale + 16 字节装 32 个 4-bit 值（每字节两个 nibble），共 18 字节、平均每权重 4.5 bit。",
                    "en": "block_q4_0 = {ggml_half d; uint8_t qs[16]}: a 2-byte scale + 16 bytes holding 32 4-bit values (two nibbles per byte), 18 bytes total, 4.5 bits per weight on average.",
                },
            },
            {
                "q": {
                    "zh": "K-quant（如 q4_K）为什么在相同位宽下比 q4_0 更准？",
                    "en": "Why is K-quant (e.g. q4_K) more accurate than q4_0 at the same bit width?",
                },
                "opts": [
                    {
                        "zh": "用 256 的超块：整体 d/dmin 之外，每个子块还有更细的 scale，局部更贴合",
                        "en": "It uses a 256 super-block: beyond the overall d/dmin, each sub-block has a finer scale, fitting locally better",
                    },
                    {"zh": "它其实用了更多的 bit", "en": "It actually uses more bits"},
                    {"zh": "它的量化值不打包", "en": "Its quantized values are not packed"},
                    {"zh": "它完全不丢失任何信息", "en": "It loses no information at all"},
                ],
                "answer": 0,
                "why": {
                    "zh": "q4_K 是两层 scale：超块 d/dmin 定大范围、8 个子块各有 6-bit scale/min 做局部微调，加上 dmin 的偏移量化，同样 4.5 bit 却比 q4_0 单层 scale 更准。",
                    "en": "q4_K has a two-level scale: super-block d/dmin set the broad range, 8 sub-blocks each carry a 6-bit scale/min for local fine-tuning, plus dmin's offset - the same 4.5 bits but more accurate than q4_0's single scale.",
                },
            },
            {
                "q": {
                    "zh": "ggml 怎么让算子统一处理几十种量化类型？",
                    "en": "How does ggml let operators handle dozens of quantization types uniformly?",
                },
                "opts": [
                    {
                        "zh": "用 ggml_type_traits 表 + to_float/from_float_ref 函数指针，算子按 traits 解量化，无需为每种类型各写一遍",
                        "en": "A ggml_type_traits table + to_float/from_float_ref function pointers; operators dequantize per traits, no need to rewrite per type",
                    },
                    {"zh": "为每种量化类型写一个专门的算子", "en": "Write a dedicated operator for each quantization type"},
                    {"zh": "运行时为每种类型即时编译代码", "en": "JIT-compile code for each type at runtime"},
                    {"zh": "把所有权重都转成 F32 存盘", "en": "Convert all weights to F32 on disk"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ggml_type_traits 为每种 type 登记一行（blck_size、type_size、to_float、from_float_ref 等）；算子查表、调函数指针解量化，加新类型只需填一行 + 写解/量化函数。",
                    "en": "ggml_type_traits registers one row per type (blck_size, type_size, to_float, from_float_ref, ...); operators look it up and call the function pointers, so a new type needs only one row + its dequant/quant functions.",
                },
            },
        ],
        "open": [
            {
                "zh": "q4_K 和 q4_0 都是约 4-bit，但内存布局差别很大。试着说出至少两点结构上的不同。（提示：超块/两层 scale/dmin/字节数）",
                "en": "q4_K and q4_0 are both ~4-bit, yet their memory layouts differ a lot. Name at least two structural differences. (hint: super-block / two-level scale / dmin / byte count)",
            },
        ],
    },
    "13-gguf-format.html": {
        "mcq": [
            {
                "q": {
                    "zh": "GGUF 文件里的 metadata KV 主要存什么？",
                    "en": "What does the metadata KV in a GGUF file mainly store?",
                },
                "opts": [
                    {
                        "zh": "模型的自描述信息——架构、层数/维度等超参、词表、聊天模板，让加载器无需猜测结构",
                        "en": "the model's self-describing info - architecture, hyperparameters like layer count/dimension, vocab, chat template - so the loader need not guess the structure",
                    },
                    {"zh": "只有权重的原始字节", "en": "only the raw bytes of the weights"},
                    {"zh": "只有一个版本号", "en": "only a version number"},
                    {"zh": "模型的源代码", "en": "the model's source code"},
                ],
                "answer": 0,
                "why": {
                    "zh": "metadata 是一串带类型（gguf_type）的键值对，自描述地存架构、超参、词表、聊天模板等；加载器直接读 KV 就能建图，无需外部配置或猜测。",
                    "en": "Metadata is a run of typed (gguf_type) key-value pairs that self-describe architecture, hyperparameters, vocab, chat template, etc.; the loader reads the KVs to build the graph, with no external config or guessing.",
                },
            },
            {
                "q": {
                    "zh": "llama.cpp 用 mmap 加载 GGUF 权重的好处是？",
                    "en": "What is the benefit of llama.cpp loading GGUF weights with mmap?",
                },
                "opts": [
                    {
                        "zh": "只读映射文件、零拷贝，按需分页载入，不必把几 GB 权重先读进内存再拷一遍",
                        "en": "read-only file mapping, zero-copy, paged in on demand - no need to read several GB of weights into memory and copy them again",
                    },
                    {"zh": "可以在运行时修改权重", "en": "it lets you modify the weights at runtime"},
                    {"zh": "会自动对权重再做量化", "en": "it auto-quantizes the weights again"},
                    {"zh": "会加密权重数据", "en": "it encrypts the weight data"},
                ],
                "answer": 0,
                "why": {
                    "zh": "mmap 把文件只读映射进地址空间，张量 data 指针直接指向磁盘页，用到哪页 OS 才载入；省去了把几 GB 权重整体读入再拷贝的开销，这就是“秒加载”。",
                    "en": "mmap maps the file read-only into the address space, with tensor data pointers pointing straight at disk pages loaded only when touched; it avoids reading and copying several GB of weights wholesale - that is the 'instant load'.",
                },
            },
            {
                "q": {
                    "zh": "GGUF 文件开头的 magic 和当前 version 是？",
                    "en": "What are the magic and current version at the start of a GGUF file?",
                },
                "opts": [
                    {"zh": "magic 是 \"GGUF\"，当前 version 是 3", "en": "the magic is \"GGUF\" and the current version is 3"},
                    {"zh": "magic 是 \"GGML\"，version 是 1", "en": "the magic is \"GGML\", version 1"},
                    {"zh": "magic 是 \"LLMA\"，version 是 2", "en": "the magic is \"LLMA\", version 2"},
                    {"zh": "没有 magic，直接就是权重", "en": "there is no magic, just weights directly"},
                ],
                "answer": 0,
                "why": {
                    "zh": "GGUF_MAGIC = \"GGUF\"（开头 4 字节），GGUF_VERSION = 3（见 ggml/include/gguf.h）；加载器一上来就核对它们，不对就拒绝或报错。",
                    "en": "GGUF_MAGIC = \"GGUF\" (the first 4 bytes), GGUF_VERSION = 3 (see ggml/include/gguf.h); the loader checks them immediately and refuses or errors if they do not match.",
                },
            },
        ],
        "open": [
            {
                "zh": "GGUF 把超参和词表都写进文件自描述，相比“权重文件 + 外部 config”的老办法有什么好处？（提示：可移植/可扩展/加载）",
                "en": "GGUF writes hyperparameters and vocab into the file as self-description. What are the advantages over the old \"weights file + external config\" approach? (hint: portability / extensibility / loading)",
            },
        ],
    },
    "14-model-loading.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama_model_loader 主要做什么？",
                    "en": "What does llama_model_loader mainly do?",
                },
                "opts": [
                    {
                        "zh": "读 GGUF 的 metadata（超参）和 tensor infos、建按名字的张量清单、（按 use_mmap）把权重数据 mmap 或读入",
                        "en": "read GGUF metadata (hyperparameters) and tensor infos, build a name-indexed tensor list, and (per use_mmap) mmap or read the weight data",
                    },
                    {"zh": "训练这个模型", "en": "train the model"},
                    {"zh": "把权重重新量化", "en": "re-quantize the weights"},
                    {"zh": "编译 GPU kernel", "en": "compile GPU kernels"},
                ],
                "answer": 0,
                "why": {
                    "zh": "loader 不计算，只把字节整理成可用模型：gguf_init_from_file 读头部、weights_map 按名字登记每个张量、use_mmap 时让 data 指针指进文件映射（L13 零拷贝）。",
                    "en": "The loader computes nothing; it organizes bytes into a usable model: gguf_init_from_file reads the header, weights_map registers each tensor by name, and with use_mmap the data pointers point into the file mapping (L13 zero-copy).",
                },
            },
            {
                "q": {
                    "zh": "一个被分片的大模型，文件名长什么样？",
                    "en": "What do the filenames of a split large model look like?",
                },
                "opts": [
                    {"zh": "model-00001-of-00003.gguf 这种 of-N 编号", "en": "of-N numbering like model-00001-of-00003.gguf"},
                    {"zh": "随机哈希名", "en": "random hash names"},
                    {"zh": "一个 .zip 压缩包", "en": "a single .zip archive"},
                    {"zh": "永远是单文件，不能分片", "en": "always a single file, never split"},
                ],
                "answer": 0,
                "why": {
                    "zh": "分片文件名由 llama_split_path 按 \"%s-%05d-of-%05d.gguf\" 拼出；split.count 记总片数；loader 按编号逐片打开、并进同一张 weights_map。",
                    "en": "Split filenames are built by llama_split_path as \"%s-%05d-of-%05d.gguf\"; split.count records the total; the loader opens each by number and merges into one weights_map.",
                },
            },
            {
                "q": {
                    "zh": "加载器怎么知道模型有多少层、多大维度？",
                    "en": "How does the loader know the model's layer count and dimensions?",
                },
                "opts": [
                    {
                        "zh": "用 get_key(llm_kv, ...) 从 GGUF 的 metadata KV 里读（自描述）",
                        "en": "it reads them from the GGUF metadata KVs via get_key(llm_kv, ...) (self-describing)",
                    },
                    {"zh": "靠猜测", "en": "by guessing"},
                    {"zh": "读一个外部 config.json", "en": "by reading an external config.json"},
                    {"zh": "在代码里硬编码", "en": "hard-coded in the code"},
                ],
                "answer": 0,
                "why": {
                    "zh": "超参全在 GGUF 的 metadata KV 里（L13 自描述）；loader 用模板方法 get_key 把键映射到具体超参字段，无需外部配置或猜测。",
                    "en": "Hyperparameters live in the GGUF metadata KVs (L13 self-description); the loader's templated get_key maps a key to a specific field, with no external config or guessing.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L13，说说 llama_model_loader 为什么用 mmap 加载权重数据能做到“秒加载”又省内存。（提示：零拷贝/按页/共享）",
                "en": "Drawing on L13, explain why llama_model_loader's mmap loading of weight data achieves 'instant load' and saves memory. (hint: zero-copy / paging / sharing)",
            },
        ],
    },
    "15-architecture-hparams.html": {
        "mcq": [
            {
                "q": {
                    "zh": "GGUF 里哪个 KV 决定按哪套架构建图？",
                    "en": "Which GGUF KV decides which architecture's graph to build?",
                },
                "opts": [
                    {"zh": "general.architecture（如 \"llama\"、\"qwen2\"）", "en": "general.architecture (e.g. \"llama\", \"qwen2\")"},
                    {"zh": "general.name", "en": "general.name"},
                    {"zh": "general.file_type", "en": "general.file_type"},
                    {"zh": "version", "en": "version"},
                ],
                "answer": 0,
                "why": {
                    "zh": "loader 读出 general.architecture 字符串，在 LLM_ARCH_NAMES 里查得 LLM_ARCH_LLAMA 等枚举；这个枚举决定后续用哪套 KV/张量约定与建图函数（L16）。",
                    "en": "The loader reads the general.architecture string and looks it up in LLM_ARCH_NAMES to get an enum like LLM_ARCH_LLAMA; that enum decides the KV/tensor conventions and graph builder used (L16).",
                },
            },
            {
                "q": {
                    "zh": "为什么 hparams 把头数写成 n_head(il) 带层号？",
                    "en": "Why does hparams write head count as n_head(il) with a layer index?",
                },
                "opts": [
                    {
                        "zh": "不同层的头数/注意力类型可能不同（GQA、滑窗），按层取最通用",
                        "en": "different layers may have different head counts/attention types (GQA, sliding-window), so per-layer is most general",
                    },
                    {"zh": "写错了，应该是字段", "en": "it is a typo; it should be a field"},
                    {"zh": "为了让推理更快", "en": "to make inference faster"},
                    {"zh": "随机决定的", "en": "decided at random"},
                ],
                "answer": 0,
                "why": {
                    "zh": "现代架构里各层注意力配置可能不同（GQA 让 KV 头少于 Q 头、混合架构逐层不同），所以头数按层存进 n_head_arr，n_head(il) 是按层取值的访问器方法（不是字段）。",
                    "en": "In modern architectures per-layer attention configs can differ (GQA gives fewer KV than Q heads; hybrids differ by layer), so head counts are stored per layer in n_head_arr, and n_head(il) is an accessor method (not a field) fetching by layer.",
                },
            },
            {
                "q": {
                    "zh": "加载器怎么把文件里的张量对应到模型结构？",
                    "en": "How does the loader map a file's tensors onto the model structure?",
                },
                "opts": [
                    {
                        "zh": "靠 LLM_TENSOR_NAMES 的命名约定（token_embd / blk.N.attn_q ...）按名字在 weights_map 里查",
                        "en": "by the LLM_TENSOR_NAMES naming convention (token_embd / blk.N.attn_q ...), looking up weights_map by name",
                    },
                    {"zh": "按文件里的张量顺序", "en": "by the tensor order in the file"},
                    {"zh": "按张量大小排序", "en": "by sorting on tensor size"},
                    {"zh": "随机匹配", "en": "by random matching"},
                ],
                "answer": 0,
                "why": {
                    "zh": "张量名遵循 LLM_TENSOR_NAMES 模板（blk.%d 里填层号），由 LLM_TN 的 tn() 拼出；建图按名字去 weights_map 取权重。名字是稳定契约，跨工具、跨分片都不怕。",
                    "en": "Tensor names follow LLM_TENSOR_NAMES templates (blk.%d filled with the layer index), built by LLM_TN's tn(); graph-building fetches weights from weights_map by name. A name is a stable contract, robust across tools and splits.",
                },
            },
        ],
        "open": [
            {
                "zh": "llm_arch、llama_hparams、LLM_TENSOR_NAMES 三者各管什么？它们怎么合起来把“一堆张量”变成“一个具体可建图的模型”？",
                "en": "What do llm_arch, llama_hparams, and LLM_TENSOR_NAMES each govern? How do they combine to turn 'a pile of tensors' into 'a concrete, graph-able model'?",
            },
        ],
    },
    "16-build-graph.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama 层怎么为不同架构建出不同的前向图？",
                    "en": "How does the llama layer build different forward graphs for different architectures?",
                },
                "opts": [
                    {
                        "zh": "llama_model::build_graph 派发到每架构自己的 build_arch_graph（src/models/&lt;arch&gt;.cpp），复用 llm_graph_context 的 build_* 积木",
                        "en": "llama_model::build_graph dispatches to each architecture's build_arch_graph (src/models/&lt;arch&gt;.cpp), reusing llm_graph_context's build_* blocks",
                    },
                    {"zh": "一个巨型 if-else", "en": "one giant if-else"},
                    {"zh": "每个架构一个独立引擎", "en": "a separate engine per architecture"},
                    {"zh": "运行时编译整个模型", "en": "compiling the whole model at runtime"},
                ],
                "answer": 0,
                "why": {
                    "zh": "build_graph 是稳定入口，调虚函数 build_arch_graph 派发到各架构的 src/models/&lt;arch&gt;.cpp；真正干活的 build_attn/build_ffn/build_norm 是基类 llm_graph_context 的共享积木。",
                    "en": "build_graph is the stable entry; it calls the virtual build_arch_graph, dispatching to each architecture's src/models/&lt;arch&gt;.cpp, while the real workers build_attn/build_ffn/build_norm are shared blocks on the base llm_graph_context.",
                },
            },
            {
                "q": {
                    "zh": "build_graph 产出什么、交给谁？",
                    "en": "What does build_graph produce, and hand to whom?",
                },
                "opts": [
                    {"zh": "一张 ggml_cgraph（只建不算），交给后端执行（L10）", "en": "a ggml_cgraph (built, not computed), handed to the backend to execute (L10)"},
                    {"zh": "直接产出文本", "en": "text output directly"},
                    {"zh": "立即算出结果", "en": "the computed result immediately"},
                    {"zh": "一个 .gguf 文件", "en": "a .gguf file"},
                ],
                "answer": 0,
                "why": {
                    "zh": "build_* 只填算子的 op/src（L09 惰性建图），get_gf() 交出一张 ggml_cgraph；真正逐节点执行是 L10 后端的事。所以同一张图能换后端跑。",
                    "en": "build_* only fills operators' op/src (L09 lazy build); get_gf() hands out a ggml_cgraph; actual node-by-node execution is the L10 backend's job. So the same graph runs on any backend.",
                },
            },
            {
                "q": {
                    "zh": "一层 transformer 在图里大致是什么顺序？",
                    "en": "Roughly what order is one transformer layer in the graph?",
                },
                "opts": [
                    {"zh": "norm -> attn（QKV+rope+KV+softmax）-> 残差 -> norm -> ffn -> 残差", "en": "norm -> attn (QKV+rope+KV+softmax) -> residual -> norm -> ffn -> residual"},
                    {"zh": "只有一个 mul_mat", "en": "just one mul_mat"},
                    {"zh": "完全随机", "en": "completely random"},
                    {"zh": "先 ffn 后 attn 且无 norm", "en": "ffn before attn, with no norm"},
                ],
                "answer": 0,
                "why": {
                    "zh": "一个 block 的骨架固定：build_norm -> build_attn -> 残差 -> build_norm -> build_ffn -> 残差；循环 n_layer() 层，每层按名字取权重（L15）。",
                    "en": "A block's skeleton is fixed: build_norm -> build_attn -> residual -> build_norm -> build_ffn -> residual; looped n_layer() times, fetching weights by name per layer (L15).",
                },
            },
        ],
        "open": [
            {
                "zh": "build_attn 这种积木被复用、加新架构只写一份 src/models/&lt;arch&gt;.cpp——这种结构对“支持很多模型”有什么好处？",
                "en": "build_attn-style blocks are reused, and a new architecture only writes one src/models/&lt;arch&gt;.cpp - what are the benefits of this structure for 'supporting many models'?",
            },
        ],
    },
    "17-context-session.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama_model 和 llama_context 的区别是？",
                    "en": "What is the difference between llama_model and llama_context?",
                },
                "opts": [
                    {
                        "zh": "model 是只读权重（可被多 context 共享），context 是有状态运行时（KV/sched/logits，每会话一个）",
                        "en": "model is read-only weights (shareable by many contexts); context is a stateful runtime (KV/sched/logits, one per session)",
                    },
                    {"zh": "是一回事，只是名字不同", "en": "they are the same thing, just different names"},
                    {"zh": "context 存权重，model 存 KV", "en": "context stores weights, model stores KV"},
                    {"zh": "model 有状态，context 只读", "en": "model is stateful, context is read-only"},
                ],
                "answer": 0,
                "why": {
                    "zh": "权重只读、几个 GB，多会话共享同一份最省内存；KV cache、当前位置是每会话不同的状态，必须各存一份。所以 model 只读可共享、context 有状态每会话一个。",
                    "en": "Weights are read-only and several GB, so sharing one copy across sessions saves the most memory; KV cache and current position are per-session state stored separately. So model is read-only/shareable, context is stateful/per-session.",
                },
            },
            {
                "q": {
                    "zh": "llama_decode 做什么？",
                    "en": "What does llama_decode do?",
                },
                "opts": [
                    {"zh": "跑一步前向（建图 + 执行 + 更新 KV），算出 logits", "en": "runs one forward step (build graph + execute + update KV), producing logits"},
                    {"zh": "加载模型", "en": "loads the model"},
                    {"zh": "直接采样出一个 token", "en": "directly samples a token"},
                    {"zh": "释放内存", "en": "frees memory"},
                ],
                "answer": 0,
                "why": {
                    "zh": "llama_decode 吃一个 batch，内部切 ubatch（L18）-> build_graph（L16）-> sched 执行（L10）-> 更新 KV（L19）-> 把 logits 写进输出缓冲。采样是下一步（L21）的事。",
                    "en": "llama_decode eats a batch, internally splitting ubatch (L18) -> build_graph (L16) -> sched execute (L10) -> update KV (L19) -> write logits to the output buffer. Sampling is the next step's job (L21).",
                },
            },
            {
                "q": {
                    "zh": "取第 i 个 token 的 logits 用哪个？",
                    "en": "Which call gets the i-th token's logits?",
                },
                "opts": [
                    {"zh": "llama_get_logits_ith(ctx, i)", "en": "llama_get_logits_ith(ctx, i)"},
                    {"zh": "llama_get_model", "en": "llama_get_model"},
                    {"zh": "llama_tokenize", "en": "llama_tokenize"},
                    {"zh": "llama_free", "en": "llama_free"},
                ],
                "answer": 0,
                "why": {
                    "zh": "llama_get_logits_ith(ctx, i) 取第 i 个被标记输出的位置的 logits——一个 n_vocab 维向量，交给采样（L21）挑词。",
                    "en": "llama_get_logits_ith(ctx, i) reads the logits of the i-th flagged-output position - an n_vocab-dimensional vector handed to sampling (L21) to pick a word.",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么把“权重”（model）和“会话状态”（context）分成两个对象？这对一台机器服务很多用户有什么好处？",
                "en": "Why split 'weights' (model) and 'session state' (context) into two objects? What does this gain for one machine serving many users?",
            },
        ],
    },
    "18-batching.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama_batch 的 logits 字段是干嘛的？",
                    "en": "What is the logits field of llama_batch for?",
                },
                "opts": [
                    {"zh": "一个 per-token 标志，标记哪些 token 需要算输出 logits", "en": "a per-token flag marking which tokens need output logits computed"},
                    {"zh": "存放算好的 logits", "en": "stores the already-computed logits"},
                    {"zh": "存权重", "en": "stores weights"},
                    {"zh": "采样温度", "en": "the sampling temperature"},
                ],
                "answer": 0,
                "why": {
                    "zh": "logits 是个开关数组（源码注释将改名 output）：标了的位置才做输出投影（隐藏向量 -> 词表大小的大矩阵乘）。prefill 往往只标最后一个，省掉大量无用投影。",
                    "en": "logits is a switch array (source comment will rename to output): only flagged positions do the output projection (a hidden-vector -> vocab-size matmul). Prefill often flags only the last, saving many useless projections.",
                },
            },
            {
                "q": {
                    "zh": "为什么要把 batch 切成 ubatch？",
                    "en": "Why split a batch into ubatches?",
                },
                "opts": [
                    {"zh": "硬件一次能高效处理的物理批大小有限（n_ubatch），大批切成小批逐个算", "en": "the physical batch size hardware can efficiently process is limited (n_ubatch); a big batch is split into small ones computed one by one"},
                    {"zh": "为了加密", "en": "for encryption"},
                    {"zh": "为多线程随机切", "en": "to split randomly for multithreading"},
                    {"zh": "没有意义", "en": "no reason"},
                ],
                "answer": 0,
                "why": {
                    "zh": "n_batch 是逻辑批（一次能提交多少），n_ubatch 是物理批（硬件一次高效算多少）。llama_batch_allocr 把大的逻辑批 split 成若干 &lt;= n_ubatch 的物理批逐个喂图，两者解耦。",
                    "en": "n_batch is the logical batch (how much you can submit at once), n_ubatch the physical batch (how much hardware efficiently computes at once). llama_batch_allocr splits the big logical batch into several &lt;= n_ubatch physical batches fed one by one, the two decoupled.",
                },
            },
            {
                "q": {
                    "zh": "batch 里的 seq_id 表示什么？",
                    "en": "What does seq_id in a batch represent?",
                },
                "opts": [
                    {"zh": "这个 token 属于哪条序列（支持多序列并行）", "en": "which sequence this token belongs to (supports multi-sequence parallelism)"},
                    {"zh": "token 的 id", "en": "the token's id"},
                    {"zh": "token 的位置", "en": "the token's position"},
                    {"zh": "输出标志", "en": "the output flag"},
                ],
                "answer": 0,
                "why": {
                    "zh": "seq_id 标明每个 token 属于哪条（或哪些）序列。一个 batch/context 可同时装多条序列，它们共享权重和调度、各有各的 KV（按 seq_id 区分，L19）。",
                    "en": "seq_id marks which sequence(s) each token belongs to. One batch/context can hold several sequences at once, sharing weights and scheduling, each with its own KV (distinguished by seq_id, L19).",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L03 的 prefill/decode，说说 batch 的 logits 标志怎么帮引擎省掉不必要的计算。",
                "en": "Drawing on L03's prefill/decode, explain how the batch's logits flag helps the engine skip unnecessary computation.",
            },
        ],
    },
    "19-kv-cache.html": {
        "mcq": [
            {
                "q": {
                    "zh": "KV cache 解决什么问题？",
                    "en": "What problem does the KV cache solve?",
                },
                "opts": [
                    {"zh": "缓存先前 token 的 K/V，让自回归每步只算新 token（不重算整段）", "en": "caches prior tokens' K/V so autoregression computes only the new token each step (no whole-segment recompute)"},
                    {"zh": "压缩权重", "en": "compresses weights"},
                    {"zh": "缓存 logits", "en": "caches logits"},
                    {"zh": "加速模型加载", "en": "speeds up model loading"},
                ],
                "answer": 0,
                "why": {
                    "zh": "没缓存每步要重算前面所有 token 的 K/V（随长度平方涨）；有缓存则每步只算新 token、读历史 K/V（线性）。L04 证明二者数值等价但快一个数量级，是 decode 快的根本。",
                    "en": "Without a cache each step recomputes all prior tokens' K/V (quadratic in length); with one, each step computes only the new token and reads historical K/V (linear). L04 proves they are numerically equivalent but an order of magnitude faster - the root of decode's speed.",
                },
            },
            {
                "q": {
                    "zh": "一个 cell 主要记录什么？",
                    "en": "What does a cell mainly record?",
                },
                "opts": [
                    {"zh": "这个位置的 pos、属于哪些 seq_id（以及对应的 K/V）", "en": "this position's pos, which seq_ids it belongs to (and the corresponding K/V)"},
                    {"zh": "权重", "en": "weights"},
                    {"zh": "文件偏移", "en": "a file offset"},
                    {"zh": "采样概率", "en": "sampling probabilities"},
                ],
                "answer": 0,
                "why": {
                    "zh": "llama_kv_cells 管理一格格 cell，每个 cell 记 pos（喂 rope/因果掩码）和所属 seq_id（支持多序列）；head 是滚动写指针。pos+seq_id 是 cell 的核心标识。",
                    "en": "llama_kv_cells manages the grid of cells; each records pos (fed to rope/causal mask) and its seq_id (for multi-sequence); head is the rolling write pointer. pos+seq_id are a cell's core identity.",
                },
            },
            {
                "q": {
                    "zh": "公开 C API 里删除某序列的 KV 用哪个？",
                    "en": "Which public C API removes a sequence's KV?",
                },
                "opts": [
                    {"zh": "llama_memory_seq_rm（经 llama_get_memory）", "en": "llama_memory_seq_rm (via llama_get_memory)"},
                    {"zh": "llama_kv_self_rm（已改名移除）", "en": "llama_kv_self_rm (renamed and removed)"},
                    {"zh": "llama_free", "en": "llama_free"},
                    {"zh": "llama_decode", "en": "llama_decode"},
                ],
                "answer": 0,
                "why": {
                    "zh": "序列操作的公开 API 是 llama_memory_seq_*（seq_rm/seq_cp/seq_add 等），经 llama_get_memory 拿到记忆对象；旧名 llama_kv_self_* 已改名移除，看老教程要小心。",
                    "en": "The public sequence-op API is llama_memory_seq_* (seq_rm/seq_cp/seq_add etc.), via llama_get_memory; the old llama_kv_self_* names are renamed and removed, so beware old tutorials.",
                },
            },
        ],
        "open": [
            {
                "zh": "KV cache 很吃显存。结合 L17 的 type_k/type_v 和本课的滑窗变体，说说有哪些办法控制 KV 的内存。",
                "en": "The KV cache is VRAM-hungry. Drawing on L17's type_k/type_v and this lesson's sliding-window variant, what are the ways to control KV memory?",
            },
        ],
    },
    "20-vocabulary.html": {
        "mcq": [
            {
                "q": {
                    "zh": "tokenize 做什么？",
                    "en": "What does tokenize do?",
                },
                "opts": [
                    {"zh": "把文本字符串切成一串 token id（喂给模型）", "en": "cuts a text string into a list of token ids (to feed the model)"},
                    {"zh": "把 token 还原成文本", "en": "turns tokens back into text"},
                    {"zh": "训练一张新词表", "en": "trains a new vocabulary"},
                    {"zh": "给每个 token 打分", "en": "scores each token"},
                ],
                "answer": 0,
                "why": {
                    "zh": "tokenize 是编码方向：文本 -> token id 序列，喂进模型。反方向（id -> 文本）是 token_to_piece/detokenize 干的。",
                    "en": "tokenize is the encoding direction: text -> a list of token ids, fed into the model. The reverse (id -> text) is done by token_to_piece/detokenize.",
                },
            },
            {
                "q": {
                    "zh": "字节回退（byte fallback）的作用是什么？",
                    "en": "What is byte fallback for?",
                },
                "opts": [
                    {"zh": "让任何 UTF-8 字符都能被编码（拆成字节 token），避免未登录词 OOV", "en": "lets any UTF-8 char be encoded (split into byte tokens), avoiding out-of-vocabulary OOV"},
                    {"zh": "压缩词表大小", "en": "compresses the vocab size"},
                    {"zh": "加速推理", "en": "speeds up inference"},
                    {"zh": "删除特殊 token", "en": "removes special tokens"},
                ],
                "answer": 0,
                "why": {
                    "zh": "词表里没有的字符按 UTF-8 拆成字节、每字节一个 &lt;0xXX&gt; token（256 个必覆盖），于是再罕见的字符也能无损编码，消灭 OOV；代价是生僻字占多个 token。",
                    "en": "A char absent from the vocab is split into UTF-8 bytes, one &lt;0xXX&gt; token per byte (256 of them, guaranteed to cover), so even rare chars encode losslessly, abolishing OOV; the cost is rare chars take several tokens.",
                },
            },
            {
                "q": {
                    "zh": "取词表大小，当前应该用哪个 API？",
                    "en": "Which API should you use today to get the vocab size?",
                },
                "opts": [
                    {"zh": "llama_vocab_n_tokens（旧 llama_n_vocab 已弃用）", "en": "llama_vocab_n_tokens (old llama_n_vocab is deprecated)"},
                    {"zh": "llama_n_ctx", "en": "llama_n_ctx"},
                    {"zh": "strlen", "en": "strlen"},
                    {"zh": "llama_n_embd", "en": "llama_n_embd"},
                ],
                "answer": 0,
                "why": {
                    "zh": "词表大小是词表的属性，权威 API 是 llama_vocab_n_tokens；旧名 llama_n_vocab 已标 DEPRECATED。n_ctx/n_embd 是别的量。",
                    "en": "Vocab size is a vocab property; the authoritative API is llama_vocab_n_tokens; the old llama_n_vocab is marked DEPRECATED. n_ctx/n_embd are different quantities.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L21，描述一次对话生成里 tokenize 和 token_to_piece 各在什么时候被调用、各处理哪个方向。",
                "en": "Drawing on L21, describe when tokenize and token_to_piece are each called in one chat generation, and which direction each handles.",
            },
        ],
    },
    "21-sampling.html": {
        "mcq": [
            {
                "q": {
                    "zh": "greedy 采样选哪个 token？",
                    "en": "Which token does greedy sampling pick?",
                },
                "opts": [
                    {"zh": "logit 最大的那个（argmax，确定性）", "en": "the one with the max logit (argmax, deterministic)"},
                    {"zh": "随机一个", "en": "a random one"},
                    {"zh": "最后一个", "en": "the last one"},
                    {"zh": "logit 最小的那个", "en": "the one with the min logit"},
                ],
                "answer": 0,
                "why": {
                    "zh": "greedy 永远取 logit 最大的候选（argmax），同样输入永远同样输出；dist 才是按概率随机抽。要复现/严谨用 greedy，要多样性用 dist。",
                    "en": "greedy always takes the max-logit candidate (argmax); same input always same output. dist is the one that draws randomly by probability. Use greedy for reproducibility/rigor, dist for diversity.",
                },
            },
            {
                "q": {
                    "zh": "top_p（核采样）保留哪些候选？",
                    "en": "Which candidates does top_p (nucleus) keep?",
                },
                "opts": [
                    {"zh": "按概率从高到低累加、达到阈值 p 的最小候选集合", "en": "the smallest set whose cumulative probability (high to low) reaches threshold p"},
                    {"zh": "固定的前 50 个", "en": "a fixed top 50"},
                    {"zh": "概率大于 p 的全部", "en": "all with probability greater than p"},
                    {"zh": "全部候选", "en": "all candidates"},
                ],
                "answer": 0,
                "why": {
                    "zh": "top_p 按概率累加到达 p 为止，候选数随分布自适应（尖时少、平时多）；top_k 才是固定个数。两者常配合：先 top_k 砍长尾，再 top_p 收口。",
                    "en": "top_p accumulates probability until reaching p, so the candidate count adapts to the distribution (few when peaked, many when flat); top_k is the fixed-count one. They often pair: top_k chops the tail, top_p closes adaptively.",
                },
            },
            {
                "q": {
                    "zh": "把采样做成\"链\"（chain）的主要好处是什么？",
                    "en": "What is the main benefit of making sampling a 'chain'?",
                },
                "opts": [
                    {"zh": "可组合——按顺序施加多个独立、可配置的采样器", "en": "composability - apply several independent, configurable samplers in order"},
                    {"zh": "跑得更快", "en": "it runs faster"},
                    {"zh": "省内存", "en": "it saves memory"},
                    {"zh": "只能用一个采样器", "en": "it allows only one sampler"},
                ],
                "answer": 0,
                "why": {
                    "zh": "链把采样策略变成数据：每个采样器是独立小部件、自带状态，顺序可调、增删自由，用户调参就能拼出任意策略，引擎主干不动。快/省内存不是它的设计目的。",
                    "en": "The chain turns the strategy into data: each sampler is an independent part with its own state, order is adjustable, add/remove is free; users tune parameters to assemble any strategy without touching the engine. Speed/memory are not its design goal.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L20 和 L23，说说采样器为什么是在\"词表的 token 空间\"里工作，以及 grammar 如何作为一种\"掩码\"来约束这一步。",
                "en": "Drawing on L20 and L23, explain why a sampler works in 'the vocabulary's token space', and how grammar acts as a 'mask' to constrain this step.",
            },
        ],
    },
    "22-chat-templates.html": {
        "mcq": [
            {
                "q": {
                    "zh": "对话模板（chat template）做什么？",
                    "en": "What does a chat template do?",
                },
                "opts": [
                    {"zh": "把带角色的消息列表拼成该模型约定格式的提示词字符串", "en": "assemble a role-tagged message list into a prompt string in the model's agreed format"},
                    {"zh": "把文本切成 token", "en": "cut text into tokens"},
                    {"zh": "给回答打分", "en": "score the reply"},
                    {"zh": "压缩对话历史", "en": "compress the conversation history"},
                ],
                "answer": 0,
                "why": {
                    "zh": "模板负责把 system/user/assistant 的消息列表，按这个模型训练时的格式（插入特殊标记）拼成一段字符串，再交给词表 tokenize。切 token 是 L20、打分是 L21 的事。",
                    "en": "The template assembles the system/user/assistant message list into a string per the model's training format (inserting special markers), then hands it to the vocab to tokenize. Cutting tokens is L20, scoring is L21.",
                },
            },
            {
                "q": {
                    "zh": "ChatML 模板用哪对标记包裹每条消息？",
                    "en": "Which pair of markers does ChatML use to wrap each message?",
                },
                "opts": [
                    {"zh": "&lt;|im_start|&gt; 和 &lt;|im_end|&gt;", "en": "&lt;|im_start|&gt; and &lt;|im_end|&gt;"},
                    {"zh": "[INST] 和 [/INST]", "en": "[INST] and [/INST]"},
                    {"zh": "&lt;s&gt; 和 &lt;/s&gt;", "en": "&lt;s&gt; and &lt;/s&gt;"},
                    {"zh": "{{ 和 }}", "en": "{{ and }}"},
                ],
                "answer": 0,
                "why": {
                    "zh": "ChatML 用 &lt;|im_start|&gt;role ... &lt;|im_end|&gt; 包每条消息；[INST]/[/INST] 是 Llama-2 的；&lt;s&gt;/&lt;/s&gt; 是序列起止符；{{ }} 是 Jinja 语法。",
                    "en": "ChatML wraps each message as &lt;|im_start|&gt;role ... &lt;|im_end|&gt;; [INST]/[/INST] is Llama-2's; &lt;s&gt;/&lt;/s&gt; are sequence delimiters; {{ }} is Jinja syntax.",
                },
            },
            {
                "q": {
                    "zh": "add_ass（add assistant）为真时会做什么？",
                    "en": "What does add_ass (add assistant) do when true?",
                },
                "opts": [
                    {"zh": "在末尾追加 assistant 起始标记，让模型接着生成回答", "en": "append the assistant start marker at the end so the model continues generating the reply"},
                    {"zh": "删除 system 消息", "en": "remove the system message"},
                    {"zh": "把回答翻译成英文", "en": "translate the reply into English"},
                    {"zh": "关闭采样", "en": "turn off sampling"},
                ],
                "answer": 0,
                "why": {
                    "zh": "add_ass 在拼好的提示词末尾补上 assistant 的起始标记（不含内容），相当于把话筒递给模型，让它从\"该助手说话\"处续写。聊天时开、纯补全时关。",
                    "en": "add_ass appends the assistant's start marker (no content) at the end of the assembled prompt, like handing over the mic so the model continues from 'the assistant's turn'. On for chat, off for plain completion.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L20，说说为什么要\"先套对话模板、再 tokenize\"，如果把顺序反过来会出什么问题。",
                "en": "Drawing on L20, explain why you 'apply the chat template first, then tokenize', and what goes wrong if you reverse the order.",
            },
        ],
    },
    "23-grammar.html": {
        "mcq": [
            {
                "q": {
                    "zh": "GBNF 语法约束怎么起作用？",
                    "en": "How does a GBNF grammar constraint work?",
                },
                "opts": [
                    {"zh": "采样时把不符合语法的 token 的 logit 设成负无穷（掩码），模型只能选合法 token", "en": "at sampling time it sets the logit of grammar-illegal tokens to negative infinity (a mask), so the model can only pick legal tokens"},
                    {"zh": "生成完用正则校验", "en": "validate with a regex after generation"},
                    {"zh": "微调模型", "en": "fine-tune the model"},
                    {"zh": "改 prompt 提示", "en": "change the prompt"},
                ],
                "answer": 0,
                "why": {
                    "zh": "grammar 在采样前掩掉非法候选（logit 设负无穷、概率归零），选定后推进语法状态。这样每步都合法、输出必然合法，不靠事后校验，也不用微调或改 prompt。",
                    "en": "The grammar masks illegal candidates before sampling (logit to negative infinity, probability zeroed), then advances the grammar state after a pick. So every step is legal and the output is necessarily valid - no after-the-fact check, no fine-tuning or prompt change.",
                },
            },
            {
                "q": {
                    "zh": "GBNF 文法的入口（起始）规则叫什么？",
                    "en": "What is the entry (start) rule of a GBNF grammar called?",
                },
                "opts": [
                    {"zh": "root", "en": "root"},
                    {"zh": "main", "en": "main"},
                    {"zh": "start", "en": "start"},
                    {"zh": "entry", "en": "entry"},
                ],
                "answer": 0,
                "why": {
                    "zh": "GBNF 约定入口规则叫 root，文法从它开始展开（就像程序从 main 开始）。读 GBNF 从 root 出发顺着 ::= 往下看最省力。",
                    "en": "GBNF conventionally names the entry rule root, where the grammar starts expanding (like a program at main). The easiest way to read a GBNF is to start from root and follow ::= downward.",
                },
            },
            {
                "q": {
                    "zh": "相比'生成完再校验'，token 级语法掩码的好处是？",
                    "en": "Compared to 'check after generation', what is the benefit of token-level grammar masking?",
                },
                "opts": [
                    {"zh": "每一步都保证合法，不会生成到一半才发现非法而重来", "en": "every step is guaranteed legal, never finding illegality halfway and having to retry"},
                    {"zh": "占内存更少", "en": "it uses less memory"},
                    {"zh": "不需要词表", "en": "it needs no vocabulary"},
                    {"zh": "让模型更有创意", "en": "it makes the model more creative"},
                ],
                "answer": 0,
                "why": {
                    "zh": "事后校验慢且可能反复失败、不保证收敛；token 级掩码把关口前移到每一步，生成出来必然合法、一次成型。它仍要用词表的 token 空间工作。",
                    "en": "After-the-fact checking is slow, may fail repeatedly, and is not guaranteed to converge; token-level masking moves the gate to every step, so output is necessarily valid in one go. It still works in the vocab's token space.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L21，说说 grammar 作为一种'掩码采样器'为什么常被放在采样链之外、按 grammar_first 单独施加。",
                "en": "Drawing on L21, explain why grammar, as a 'mask sampler', is often placed outside the sampler chain and applied separately per grammar_first.",
            },
        ],
    },
    "24-lora-adapters.html": {
        "mcq": [
            {
                "q": {
                    "zh": "LoRA 怎么改变模型行为？",
                    "en": "How does LoRA change model behavior?",
                },
                "opts": [
                    {"zh": "冻结原权重，用两个小矩阵 A、B 给权重加一个低秩增量 scale·B·A", "en": "freeze the original weights and add a low-rank delta scale*B*A via two small matrices A, B"},
                    {"zh": "重训全部权重", "en": "retrain all the weights"},
                    {"zh": "换一个词表", "en": "swap the vocabulary"},
                    {"zh": "改采样温度", "en": "change the sampling temperature"},
                ],
                "answer": 0,
                "why": {
                    "zh": "LoRA 冻结原权重 W，只学小矩阵 A、B，输出 = W·x + scale·B·A·x。适配器只有几 MB，远比重训全部权重轻；它和换词表、调温度是完全不同的事。",
                    "en": "LoRA freezes the original W and learns only small matrices A, B; output = W*x + scale*B*A*x. The adapter is just a few MB, far lighter than retraining all weights; it is wholly different from swapping the vocab or tuning temperature.",
                },
            },
            {
                "q": {
                    "zh": "当前 llama.cpp 给上下文挂载 LoRA 用哪个 API？",
                    "en": "Which API attaches a LoRA to the context in current llama.cpp?",
                },
                "opts": [
                    {"zh": "批量的 llama_set_adapters_lora（单数 set/rm/clear 已移除，n=0 清空）", "en": "the batched llama_set_adapters_lora (singular set/rm/clear removed, n=0 clears)"},
                    {"zh": "单数的 llama_set_adapter_lora", "en": "the singular llama_set_adapter_lora"},
                    {"zh": "llama_lora_apply", "en": "llama_lora_apply"},
                    {"zh": "重新加载模型", "en": "reload the model"},
                ],
                "answer": 0,
                "why": {
                    "zh": "当前 API 是复数批量的 llama_set_adapters_lora（一次可挂多个、各带 scale，n=0 清空）；早期单数的 set/rm/clear 三件套已被它取代、不复存在。",
                    "en": "The current API is the plural, batched llama_set_adapters_lora (attach several at once, each with a scale; n=0 clears); the early singular set/rm/clear trio is replaced by it and no longer exists.",
                },
            },
            {
                "q": {
                    "zh": "LoRA 和控制向量（control vector）的主要区别？",
                    "en": "The main difference between LoRA and a control vector?",
                },
                "opts": [
                    {"zh": "LoRA 给权重加低秩增量（改 matmul），控制向量沿固定方向平移激活（加进残差流）", "en": "LoRA adds a low-rank delta to weights (changing matmul); a control vector shifts activations along a fixed direction (added to the residual stream)"},
                    {"zh": "两者完全一样", "en": "they are exactly the same"},
                    {"zh": "LoRA 改词表、cvec 改采样", "en": "LoRA changes the vocab, cvec changes sampling"},
                    {"zh": "都要重训模型", "en": "both require retraining the model"},
                ],
                "answer": 0,
                "why": {
                    "zh": "LoRA 动权重（低秩增量折进 matmul，build_lora_mm），控制向量动激活（沿方向平移残差流，set_adapter_cvec）。两者都不碰基础权重、即插即用，但层面不同。",
                    "en": "LoRA acts on weights (a low-rank delta folded into matmul, build_lora_mm); a control vector acts on activations (shifting the residual stream along a direction, set_adapter_cvec). Both leave base weights untouched and are plug-and-play, but at different levels.",
                },
            },
        ],
        "open": [
            {
                "zh": "结合 L16，说说为什么挂上 LoRA 能'不复制权重'就生效——build_lora_mm 在建图的哪一步把增量加进来。",
                "en": "Drawing on L16, explain why attaching a LoRA takes effect 'without copying weights' - at which graph-build step build_lora_mm adds the delta.",
            },
        ],
    },
    "25-c-api.html": {
        "mcq": [
            {
                "q": {
                    "zh": "llama_model 和 llama_context，哪个能被多个会话安全共享？",
                    "en": "Of llama_model and llama_context, which can be safely shared across sessions?",
                },
                "opts": [
                    {"zh": "llama_model：加载后只读，多个 context 可共用一份", "en": "llama_model: read-only after load, many contexts can share one copy"},
                    {"zh": "llama_context：它装着会话状态，最适合共享", "en": "llama_context: it holds session state, best for sharing"},
                    {"zh": "两个都不能共享，必须各自独立", "en": "neither can be shared, each must be independent"},
                    {"zh": "两个都能随便共享，无所谓", "en": "both can be shared freely, it does not matter"},
                ],
                "answer": 0,
                "why": {
                    "zh": "llama_model 是只读知识（权重加载后不变），能被多个 context 共享，加载一次到处推理；llama_context 装的是每会话私有状态（KV cache、计算缓冲、采样位置），必须一个会话一个。",
                    "en": "llama_model is read-only knowledge (weights never change after load), shareable across contexts - load once, infer in many places; llama_context holds per-session private state (KV cache, compute buffers, sampling position), so it must be one per session.",
                },
            },
            {
                "q": {
                    "zh": "在典型调用序列里，tokenize 之后、sample 之前，缺了哪一步？",
                    "en": "In the typical call sequence, what step sits between tokenize and sample?",
                },
                "opts": [
                    {"zh": "decode 一遍计算图、取出 logits（llama_decode -> llama_get_logits）", "en": "decode the graph once and read the logits (llama_decode -> llama_get_logits)"},
                    {"zh": "直接释放模型", "en": "free the model directly"},
                    {"zh": "再加载一次词表", "en": "load the vocab again"},
                    {"zh": "重新初始化后端", "en": "reinitialize the backend"},
                ],
                "answer": 0,
                "why": {
                    "zh": "顺序是 tokenize -> decode -> get_logits -> sample：分词得到 token 后，必须先 llama_decode 跑一遍计算图、用 llama_get_logits 取出每个 token 的分数，采样链才有分数可挑。",
                    "en": "The order is tokenize -> decode -> get_logits -> sample: after tokenizing, you must llama_decode the graph and read per-token scores via llama_get_logits before the sampler chain has anything to pick from.",
                },
            },
            {
                "q": {
                    "zh": "在 C++ 里想让句柄自动释放，最省心的做法是？",
                    "en": "In C++, the tidiest way to release handles automatically is?",
                },
                "opts": [
                    {"zh": "用 include/llama-cpp.h 的 _ptr 别名（unique_ptr，带匹配的 _free 删除器）", "en": "use include/llama-cpp.h's _ptr aliases (unique_ptr with matching _free deleters)"},
                    {"zh": "什么都不做，句柄会被垃圾回收", "en": "do nothing, handles are garbage-collected"},
                    {"zh": "在每个函数末尾手写 llama_free，绝不会忘", "en": "hand-write llama_free at the end of every function, never forgetting"},
                    {"zh": "调用 llama_backend_free 一次性清掉所有句柄", "en": "call llama_backend_free to wipe all handles at once"},
                ],
                "answer": 0,
                "why": {
                    "zh": "include/llama-cpp.h 给每个句柄定义了 unique_ptr 别名（llama_model_ptr 等），删除器会调用匹配的 _free，句柄一出作用域就自动释放，不怕漏放或顺序写反。C/C++ 都没有 GC，手动 _free 容易出错。",
                    "en": "include/llama-cpp.h defines a unique_ptr alias per handle (llama_model_ptr etc.) whose deleter calls the matching _free, so a handle frees the moment it leaves scope - no missed frees or wrong order. C/C++ has no GC, and manual _free is error-prone.",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么 llama.cpp 对外暴露的是 C 接口而不是 C++ 类？结合 ABI 稳定性和跨语言绑定（Python/Go/Rust/Node）说说 opaque 句柄起了什么作用。",
                "en": "Why does llama.cpp expose a C interface rather than C++ classes? Drawing on ABI stability and cross-language bindings (Python/Go/Rust/Node), explain what the opaque handles buy you.",
            },
        ],
    },
    "26-common.html": {
        "mcq": [
            {
                "q": {
                    "zh": "common 是 llama.cpp 公共 API 的一部分吗？",
                    "en": "Is common part of llama.cpp's public API?",
                },
                "opts": [
                    {"zh": "不是：它是给自家工具用的内部共享库，没有 ABI 稳定承诺", "en": "No: it is an internal shared library for the project's own tools, with no ABI-stability promise"},
                    {"zh": "是：它和 include/llama.h 一样是对外稳定契约", "en": "Yes: like include/llama.h, it is a stable outward contract"},
                    {"zh": "是：第三方语言绑定都应该直接依赖 common", "en": "Yes: third-party language bindings should all depend on common directly"},
                    {"zh": "算半个：取决于编译时是否开启某个开关", "en": "Half: it depends on a compile-time switch"},
                ],
                "answer": 0,
                "why": {
                    "zh": "common 是工具共享的内部便利库，结构体布局与签名会随项目需要改动，没有 ABI 稳定承诺；对外稳定契约是 include/llama.h，第三方绑定应直接对着它写。",
                    "en": "common is a tool-shared internal convenience library; its layouts and signatures change as the project needs, with no ABI-stability promise. The stable outward contract is include/llama.h, which third-party bindings should target directly.",
                },
            },
            {
                "q": {
                    "zh": "谁负责把命令行 argv 变成填好的 common_params？",
                    "en": "Who turns the command-line argv into a filled common_params?",
                },
                "opts": [
                    {"zh": "common_params_parse：按 enum llama_example 选定选项集，逐个调用 common_arg 的回调写入字段", "en": "common_params_parse: it picks the option set by enum llama_example and calls each common_arg callback to write fields"},
                    {"zh": "common_init()：它在全局初始化时顺便解析命令行", "en": "common_init(): it parses the command line as part of global init"},
                    {"zh": "common_sampler_init：它解析所有采样相关的参数", "en": "common_sampler_init: it parses all sampling-related arguments"},
                    {"zh": "没有谁，main() 里手写一大堆 if-else 分支", "en": "Nobody; main() hand-writes a big pile of if-else branches"},
                ],
                "answer": 0,
                "why": {
                    "zh": "common_params_parse(argc, argv, params, ex) 遍历 argv、按名字匹配 common_arg、调用回调写进 common_params；第 4 个参数 ex（enum llama_example）决定这次认哪套选项。common_init() 只做全局初始化（日志、build 信息）。",
                    "en": "common_params_parse(argc, argv, params, ex) walks argv, matches a common_arg by name, and calls callbacks to write into common_params; the 4th arg ex (enum llama_example) decides which option set applies. common_init() only does global init (logging, build info).",
                },
            },
            {
                "q": {
                    "zh": "llama-cli / llama-server 做采样时，用的是哪一层？",
                    "en": "When llama-cli / llama-server sample, which layer do they use?",
                },
                "opts": [
                    {"zh": "common_sampler：它把 L21 的 llama_sampler 链和 L23 的 GBNF 语法裹成一个对象", "en": "common_sampler: it wraps L21's llama_sampler chain and L23's GBNF grammar into one object"},
                    {"zh": "直接用裸 llama_sampler_*，完全不经过 common", "en": "the raw llama_sampler_* directly, bypassing common entirely"},
                    {"zh": "每个工具各自手写一条独立的采样链", "en": "each tool hand-writes its own separate sampler chain"},
                    {"zh": "Python 绑定里实现的采样器", "en": "the sampler implemented in the Python bindings"},
                ],
                "answer": 0,
                "why": {
                    "zh": "cli/server 用的是 common_sampler 这层：common_sampler_init 按 params.sampling.samplers 的顺序建链，common_sampler_sample 一步完成取分、过语法、采样。需要时还能用 common_sampler_get 取回底层裸 llama_sampler 链。",
                    "en": "cli/server use the common_sampler layer: common_sampler_init builds the chain in params.sampling.samplers order, and common_sampler_sample does logits, grammar, and sampling in one step. You can still recover the raw llama_sampler chain via common_sampler_get when needed.",
                },
            },
        ],
        "open": [
            {
                "zh": "假设你要为 llama.cpp 写一个 Rust 绑定：你会依赖 common，还是只用 include/llama.h？结合 common“对内不对外”、没有 ABI 稳定承诺这两点，说说你的理由。",
                "en": "Say you are writing a Rust binding for llama.cpp: would you depend on common, or use include/llama.h only? Argue from common being inward-facing and carrying no ABI-stability promise.",
            },
        ],
    },
    "27-llama-cli.html": {
        "mcq": [
            {
                "q": {
                    "zh": "现代 llama-cli 内部复用了哪个组件的引擎？",
                    "en": "Which component's engine does a modern llama-cli reuse internally?",
                },
                "opts": [
                    {"zh": "server-context（server_context）：cli #include server-context.h 并链接 server-context，与 server 同引擎、异壳", "en": "server-context (server_context): cli #includes server-context.h and links server-context, sharing server's engine with a different shell"},
                    {"zh": "它有一份完全独立的裸 llama_decode 主循环，与 server 无关", "en": "it has a fully standalone bare llama_decode main loop, unrelated to server"},
                    {"zh": "Python 绑定提供的引擎", "en": "an engine provided by the Python bindings"},
                    {"zh": "ggml 的计算图执行器，绕过 llama 层", "en": "ggml's graph executor, bypassing the llama layer"},
                ],
                "answer": 0,
                "why": {
                    "zh": "现代 cli 不再自带裸主循环：cli.cpp #include 了 server-common.h / server-context.h / server-task.h，CMake 也链接 server-context，直接复用 server 的 server_context（slot/task）。cli 与 server 是“同引擎、异壳”。",
                    "en": "A modern cli no longer carries its own bare loop: cli.cpp #includes server-common.h / server-context.h / server-task.h and CMake links server-context, reusing server's server_context (slot/task). cli and server are 'same engine, different shells'.",
                },
            },
            {
                "q": {
                    "zh": "llama-cli 的生成主循环靠什么停下来？",
                    "en": "What makes llama-cli's generation main loop stop?",
                },
                "opts": [
                    {"zh": "三者之一：n_predict 写满、模型吐出 EOG 结束符、或交互模式下命中反向提示（antiprompt）", "en": "any of three: n_predict filled, the model emits an EOG token, or a reverse prompt (antiprompt) is hit in interactive mode"},
                    {"zh": "只有一种：必须等模型自己吐出 EOG", "en": "only one: it must wait for the model to emit EOG"},
                    {"zh": "固定生成 2048 个 token 后无条件停止", "en": "it always stops unconditionally after exactly 2048 tokens"},
                    {"zh": "由操作系统的定时器中断决定", "en": "an operating-system timer interrupt decides"},
                ],
                "answer": 0,
                "why": {
                    "zh": "三个停止条件：n_predict 计数归零（你设的长度上限）、llama_vocab_is_eog 判定的结束符（模型自己喊停）、交互模式下命中你设的反向提示。忘了设 -n 又遇上模型不吐 EOG，就可能一直生成。",
                    "en": "Three stop conditions: n_predict counting to zero (your length cap), an end-of-generation token detected by llama_vocab_is_eog (the model stopping itself), or hitting your reverse prompt in interactive mode. Forget -n and meet a model that will not emit EOG, and it may run forever.",
                },
            },
            {
                "q": {
                    "zh": "llama-cli 怎么把命令行变成内部配置？",
                    "en": "How does llama-cli turn the command line into internal config?",
                },
                "opts": [
                    {"zh": "common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI) 把 argv 填进 common_params（L26）", "en": "common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI) fills argv into common_params (L26)"},
                    {"zh": "cli 自己手写一大堆 if-else 直接解析 argv", "en": "cli hand-writes a big pile of if-else to parse argv itself"},
                    {"zh": "从一个 JSON 配置文件读取，命令行被忽略", "en": "it reads a JSON config file; the command line is ignored"},
                    {"zh": "llama_model_load_from_file 顺便解析命令行", "en": "llama_model_load_from_file parses the command line along the way"},
                ],
                "answer": 0,
                "why": {
                    "zh": "cli 复用 common 的参数解析：common_params_parse 按第 4 个参数 LLAMA_EXAMPLE_CLI 选定 cli 的选项集，逐个调用 common_arg 回调把 argv 写进 common_params，再交给 common_init_from_params 产出 model+ctx+sampler。",
                    "en": "cli reuses common's arg parsing: common_params_parse picks cli's option set via the 4th argument LLAMA_EXAMPLE_CLI, calls each common_arg callback to write argv into common_params, then hands it to common_init_from_params to produce model+ctx+sampler.",
                },
            },
        ],
        "open": [
            {
                "zh": "既然 cli 和 server 复用同一台 server_context 引擎、只是外壳不同，试着说说：把“引擎”与“外壳”分开，对维护和加新特性各有什么好处？如果要再写一个 gRPC 版的 llama 服务，你会怎么搭？",
                "en": "Since cli and server reuse the same server_context engine and differ only in shell, explain: what does separating 'engine' from 'shell' buy you for maintenance and for adding features? If you had to write a gRPC version of a llama service, how would you structure it?",
            },
        ],
    },
    "28-llama-server.html": {
        "mcq": [
            {
                "q": {
                    "zh": "连续批处理（continuous batching）的核心做法是什么？",
                    "en": "What is the core mechanism of continuous batching?",
                },
                "opts": [
                    {"zh": "把多个活跃 slot 的 token 用 common_batch_add(..., {slot.id}) 拼进同一个 batch，一次 llama_decode 同时推进所有序列", "en": "pack many active slots' tokens into one batch via common_batch_add(..., {slot.id}), and one llama_decode advances all sequences at once"},
                    {"zh": "为每条请求开一个独立进程，各自加载一份权重并行跑", "en": "spawn an independent process per request, each loading its own weights and running in parallel"},
                    {"zh": "把请求排成队列，GPU 严格一条接一条地处理", "en": "queue the requests and have the GPU process them strictly one after another"},
                    {"zh": "把模型权重复制 N 份到显存，每份服务一条请求", "en": "copy the model weights N times into VRAM, one copy per request"},
                ],
                "answer": 0,
                "why": {
                    "zh": "连续批处理把处于 prefill / decode 的多个 slot 的 token 用 common_batch_add 打上各自的 seq_id 拼进同一 batch，一次 llama_decode 借注意力掩码同时推进所有活跃序列——一次前向、服务多人，这是 server 高吞吐的核心。复制权重/多进程恰恰是它要避免的。",
                    "en": "Continuous batching tags each active slot's token with its seq_id via common_batch_add and packs them into one batch; one llama_decode then advances all active sequences using the attention mask - one forward, serve many, the heart of server throughput. Copying weights / many processes is exactly what it avoids.",
                },
            },
            {
                "q": {
                    "zh": "server 的 slot 数量由哪个参数决定？",
                    "en": "Which parameter sets server's number of slots?",
                },
                "opts": [
                    {"zh": "--parallel N（源码里的 n_parallel）", "en": "--parallel N (n_parallel in the source)"},
                    {"zh": "--ctx-size / -c", "en": "--ctx-size / -c"},
                    {"zh": "--threads / -t", "en": "--threads / -t"},
                    {"zh": "--n-predict / -n", "en": "--n-predict / -n"},
                ],
                "answer": 0,
                "why": {
                    "zh": "--parallel N 开 N 个 slot（n_parallel），每个 slot 是一条独立并行序列，有自己的 seq_id + KV + 状态机。-c 设的是总上下文（再切给各 slot 成 n_ctx_slot），-t 是线程数，-n 是生成长度，都不决定 slot 数。",
                    "en": "--parallel N opens N slots (n_parallel), each an independent parallel sequence with its own seq_id + KV + state machine. -c sets the total context (divided among slots as n_ctx_slot), -t is thread count, -n is generation length - none set the slot count.",
                },
            },
            {
                "q": {
                    "zh": "llama-server 怎么做到兼容 OpenAI 客户端？",
                    "en": "How does llama-server stay compatible with OpenAI clients?",
                },
                "opts": [
                    {"zh": "server-chat 在 OpenAI 的 schema 与引擎内部表示间转换，暴露 /v1/chat/completions 等端点", "en": "server-chat converts between OpenAI's schema and the engine's internal representation, exposing endpoints like /v1/chat/completions"},
                    {"zh": "它把请求转发给 OpenAI 的云端服务器", "en": "it forwards requests to OpenAI's cloud servers"},
                    {"zh": "它要求客户端改用 llama.cpp 专有协议", "en": "it requires clients to switch to a llama.cpp proprietary protocol"},
                    {"zh": "靠 llama.h 的 C API 直接说 HTTP", "en": "the llama.h C API speaks HTTP directly"},
                ],
                "answer": 0,
                "why": {
                    "zh": "server-chat 负责把 OpenAI 的 JSON schema（messages、tools 等）与引擎内部表示来回转换，并暴露 /v1/chat/completions 等与 OpenAI 一致的端点；于是现成的 OpenAI 客户端/SDK 只要把 base URL 指向 llama-server 就能用，无需改协议、更不经过 OpenAI 云端。",
                    "en": "server-chat converts between OpenAI's JSON schema (messages, tools, ...) and the engine's internal representation and exposes OpenAI-identical endpoints like /v1/chat/completions; an existing OpenAI client/SDK just points its base URL at llama-server, with no protocol change and no detour through OpenAI's cloud.",
                },
            },
        ],
        "open": [
            {
                "zh": "为什么 server 用“一份权重 + 多 slot + 连续批处理”，而不是“多进程、每进程一份权重”？请从显存占用和吞吐两个角度说明，并谈谈这套设计的代价（比如延迟、公平、实现复杂度）。",
                "en": "Why does server use 'one set of weights + many slots + continuous batching' instead of 'many processes, one set of weights each'? Argue from VRAM usage and throughput, and discuss the costs of this design (e.g. latency, fairness, implementation complexity).",
            },
        ],
    },
    "29-quantize-tool.html": {
        "mcq": [
            {
                "q": {
                    "zh": "imatrix（重要性矩阵）是做什么用的？",
                    "en": "What is the imatrix (importance matrix) for?",
                },
                "opts": [
                    {"zh": "记录每个权重列的重要性（激活幅度），量化时把精度优先留给重要的列", "en": "records each weight column's importance (activation magnitude) so quantize keeps precision for important columns first"},
                    {"zh": "把模型权重再压缩一倍，不损失任何精度", "en": "compresses the weights another 2x with zero precision loss"},
                    {"zh": "记录每个 token 的生成概率，用于采样", "en": "records each token's generation probability for sampling"},
                    {"zh": "存储模型的超参数（层数、维度等）", "en": "stores the model's hyperparameters (layers, dims, etc.)"},
                ],
                "answer": 0,
                "why": {
                    "zh": "imatrix 用校准文本跑模型、由 collect_imatrix 累计每个权重张量每列的激活幅度，得出哪些列“重要”。量化时把它喂进去（--imatrix），同样比特下精度优先留给重要列、误差推给不重要列，整体困惑度更低。它不额外压缩、也与采样/超参无关。",
                    "en": "imatrix runs the model on calibration text and collect_imatrix accumulates each weight tensor's per-column activation magnitude, telling which columns are 'important'. Fed in at quantize time (--imatrix), at the same bits it keeps precision for important columns and pushes error onto unimportant ones, lowering overall perplexity. It does not compress further and is unrelated to sampling/hyperparameters.",
                },
            },
            {
                "q": {
                    "zh": "不真的压缩，只想试算量化后体积，用哪个旗标？",
                    "en": "Which flag trial-computes the quantized size without actually compressing?",
                },
                "opts": [
                    {"zh": "--dry-run（对应 params.dry_run）", "en": "--dry-run (params.dry_run)"},
                    {"zh": "--keep-split", "en": "--keep-split"},
                    {"zh": "--imatrix", "en": "--imatrix"},
                    {"zh": "--output-tensor-type", "en": "--output-tensor-type"},
                ],
                "answer": 0,
                "why": {
                    "zh": "--dry-run（params.dry_run）只计算并打印量化后的最终体积、并不真的压，方便你在选档位时快速对比几个档位的体积。--keep-split 保持分片，--imatrix 喂重要性矩阵，--output-tensor-type 按张量定精度，都不是“只试算体积”。",
                    "en": "--dry-run (params.dry_run) only computes and prints the final quantized size without really compressing, handy for comparing a few levels' sizes when choosing. --keep-split keeps shards, --imatrix feeds the importance matrix, --output-tensor-type sets per-tensor precision - none merely trial-compute size.",
                },
            },
            {
                "q": {
                    "zh": "L06/L12 和这一课（L29）的分工是什么？",
                    "en": "How do L06/L12 and this lesson (L29) divide up?",
                },
                "opts": [
                    {"zh": "L06/L12 讲量化的原理与字节布局；L29 讲怎么用工具压、以及 imatrix 怎么更高质量地压", "en": "L06/L12 cover quantization's principle and byte layout; L29 covers how to compress with the tool and how imatrix compresses with higher quality"},
                    {"zh": "完全重复，L29 只是把前面再讲一遍", "en": "they fully overlap; L29 just repeats the earlier lessons"},
                    {"zh": "L06/L12 讲工具用法，L29 讲底层数学", "en": "L06/L12 cover tool usage, L29 covers the underlying math"},
                    {"zh": "L29 讲采样，和量化无关", "en": "L29 covers sampling, unrelated to quantization"},
                ],
                "answer": 0,
                "why": {
                    "zh": "L06/L12 是“懂原理”（为什么能压、各格式字节怎么排）；L29 是“会操作”（llama-quantize 一键压、各档位取舍、imatrix 用校准数据把同样比特花得更聪明）。两者互补，不重复。",
                    "en": "L06/L12 is 'understand the principle' (why compression works, how each format's bytes are laid out); L29 is 'operate the tool' (one-command llama-quantize, level trade-offs, imatrix spending the same bits more cleverly via calibration data). They are complementary, not repetitive.",
                },
            },
        ],
        "open": [
            {
                "zh": "你有一张 8GB 显存的显卡，想跑一个 13B 模型。结合这一课的“先看显存、再看用途、超低比特认 imatrix”，说说你会怎么选量化档位？为什么很多人说“同样大小宁可选大模型的低档量化”？",
                "en": "You have an 8GB GPU and want to run a 13B model. Using this lesson's 'VRAM first, then use, ultra-low-bit demands imatrix', explain how you would pick a quant level. Why do many say 'at the same size, prefer a bigger model's lower level'?",
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
