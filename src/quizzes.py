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
