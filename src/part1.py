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
  用<strong>零外部依赖的 C/C++</strong> + <strong>量化</strong>（把权重压成 4/5/8 bit 等低位宽，K-quant 甚至能到 2/3/6 bit）+ <strong>自研张量引擎 ggml</strong>，
  让大模型能在<strong>消费级硬件</strong>上本地、离线、低成本地推理。一个可执行文件 + 一个 <span class="inline">.gguf</span> 文件即可运行。
</div>

<h3>整体结构图：四层自底向上</h3>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">工具</span><span class="name">tools/ · examples/</span></div>
    <div class="ld">面向用户：<span class="mono">llama-cli</span> 命令行、<span class="mono">llama-server</span> HTTP 服务、<span class="mono">llama-quantize</span> 量化器</div></div>
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

const llama_vocab *vocab = <span class="fn">llama_model_get_vocab</span>(model);
llama_sampler     *smpl  = <span class="fn">llama_sampler_chain_init</span>(<span class="fn">llama_sampler_chain_default_params</span>());
<span class="fn">llama_sampler_chain_add</span>(smpl, <span class="fn">llama_sampler_init_greedy</span>());  <span class="cm">// 最简：贪心采样</span>

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
  <p style="margin:.5rem 0 0">这条主线（加载 → 分词 → 解码循环 → 采样）就是后面所有课的骨架，后面会专门用一课展开完整生命周期。</p>
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
  With <strong>zero-dependency C/C++</strong> + <strong>quantization</strong> (compressing weights to e.g. 4/5/8 bits, down to 2/3/6-bit K-quants) +
  its own tensor engine <strong>ggml</strong>, it makes LLMs run <strong>locally, offline, and cheaply on consumer
  hardware</strong>. One executable plus one <span class="inline">.gguf</span> file is enough.
</div>

<h3>Structure map: four layers, bottom-up</h3>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">tools</span><span class="name">tools/ · examples/</span></div>
    <div class="ld">User-facing: <span class="mono">llama-cli</span>, the <span class="mono">llama-server</span> HTTP service, the <span class="mono">llama-quantize</span> tool</div></div>
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
  <tr><td><strong>transformers</strong></td><td>Model library / high-level wrapper</td><td>Python, heavy</td><td>Fast experiments</td></tr>
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

const llama_vocab *vocab = <span class="fn">llama_model_get_vocab</span>(model);
llama_sampler     *smpl  = <span class="fn">llama_sampler_chain_init</span>(<span class="fn">llama_sampler_chain_default_params</span>());
<span class="fn">llama_sampler_chain_add</span>(smpl, <span class="fn">llama_sampler_init_greedy</span>());  <span class="cm">// simplest: greedy</span>

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
  a later lesson expands it into the full lifecycle.</p>
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

LESSON_02 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
第一次打开 llama.cpp，几百个文件、十几个目录，很容易发懵。其实它<strong>分层非常清晰</strong>：
先给你一张"校园地图"，认清每个目录在干什么，后面读源码就不会迷路。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把整个仓库想成一座<strong>工厂园区</strong>，而<strong>目录就是地图</strong>：每个车间（目录）只干一件事——
  有的造引擎（<span class="mono">ggml</span>）、有的负责装配（<span class="mono">src/llama-*</span>）、有的是对外门店（<span class="mono">tools/</span>）。
  先认地图，比一头扎进某个车间更重要。
</div>

<h2>顶层目录速览</h2>
<p>站在仓库根目录，按"作用"把主要目录读一遍：</p>
<table class="t">
  <tr><th>目录</th><th>作用</th></tr>
  <tr><td class="mono">ggml/</td><td>自研<strong>张量引擎</strong>：张量 · 计算图 · 算子 · 后端调度；独立子项目（自带 <code>include/</code> 与 <code>src/</code>）</td></tr>
  <tr><td class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …</td><td>各<strong>硬件后端</strong>（还有 hip / sycl / musa / opencl 等十余种）</td></tr>
  <tr><td class="mono">src/</td><td><strong>llama 推理库</strong>：<code>llama-model-loader</code> · <code>llama-graph</code> · <code>llama-kv-cache</code> · <code>llama-sampler</code> · <code>llama-vocab</code> · <code>llama-chat</code> · <code>llama-grammar</code> · <code>llama-quant</code> …</td></tr>
  <tr><td class="mono">include/</td><td><strong>公共 C API</strong>：<code>llama.h</code>；<code>llama-cpp.h</code>（C++ 薄封装 + RAII）</td></tr>
  <tr><td class="mono">common/</td><td><strong>复用工具</strong>：<code>arg</code>（参数解析）· <code>sampling</code>（采样封装）· <code>chat</code> · <code>log</code> · <code>download</code> · <code>json-schema-to-grammar</code> …</td></tr>
  <tr><td class="mono">tools/</td><td><strong>可执行程序</strong>：<span class="mono">llama-cli</span> · <span class="mono">llama-server</span> · <span class="mono">llama-quantize</span> · <code>llama-mtmd-cli</code>（多模态）· <code>llama-perplexity</code> · <span class="mono">llama-bench</span> …</td></tr>
  <tr><td class="mono">examples/</td><td>小型示例程序（如 <code>simple</code>）</td></tr>
  <tr><td class="mono">gguf-py/</td><td><strong>Python 的 GGUF 读写库</strong></td></tr>
  <tr><td class="mono">convert_hf_to_gguf.py 等</td><td><strong>Python 转换脚本</strong>（共 4 个 <code>convert_*.py</code>，主力是 HuggingFace -> GGUF）</td></tr>
  <tr><td class="mono">models/ · tests/ · docs/ · grammars/ · cmake/</td><td>模型数据 / 测试 / 文档 / GBNF 示例 / 构建系统</td></tr>
</table>

<h2>它们怎么对上"四层"</h2>
<p>把上面这些目录映射回上一课的"四层"结构，就一目了然：</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">工具 / 应用</span><span class="name">tools/ · examples/</span></div>
    <div class="ld"><span class="mono">tools/</span>（cli · server · quantize · mtmd …）、<span class="mono">examples/</span>：面向用户的命令行、服务与示例</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">推理库</span><span class="name">src/llama-*</span></div>
    <div class="ld"><span class="mono">src/llama-*</span> 加上对外头文件 <span class="mono">include/llama.h</span>：加载 · 计算图 · KV cache · 采样 · 分词 · 聊天模板</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">引擎</span><span class="name">ggml</span></div>
    <div class="ld"><span class="mono">ggml/</span>（<span class="mono">ggml.c</span> · <span class="mono">gguf.cpp</span> · <span class="mono">ggml-alloc</span> · <span class="mono">ggml-backend</span>）：张量 · 计算图 · 算子 · 调度 · GGUF 格式</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">后端</span><span class="name">CPU · CUDA · Metal · Vulkan …</span></div>
    <div class="ld"><span class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …</span>：把算子真正算在硬件上</div></div>
</div>
<p>除了这条主干，还有两条<strong>支线</strong>：① <strong>模型准备</strong>——<span class="mono">gguf-py/</span> 加 <span class="mono">convert_*.py</span>（Python）把 HuggingFace 模型转成 <span class="inline">.gguf</span>，再喂给引擎；② <strong>配套支撑</strong>——<span class="mono">common/</span>（把库粘成程序的胶水）以及 <span class="mono">tests/</span> · <span class="mono">docs/</span> · <span class="mono">cmake/</span>。</p>

<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  想读源码却不知道从哪下手？按目标找入口：想看<strong>怎么用</strong> → <span class="mono">tools/</span> 与 <span class="mono">examples/simple</span>；想看<strong>推理逻辑</strong> → <span class="mono">src/llama-*</span>；想看<strong>底层算子</strong> → <span class="mono">ggml/</span>；想看<strong>对外契约</strong> → 只有一个 <span class="mono">include/llama.h</span>。先认入口，再逐层往下钻。
</div>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>仓库 = <strong>ggml</strong>（引擎）+ <strong>src/llama-*</strong>（推理库）+ <strong>common</strong>（胶水）+ <strong>tools</strong>（程序）+ <strong>gguf-py / convert_*</strong>（模型准备）。</li>
    <li>对外只有一个公共 C API：<span class="mono">include/llama.h</span>——这就是整个项目的<strong>外部契约</strong>。</li>
    <li><strong>ggml</strong> 是独立、可复用的引擎；<strong>llama</strong> 只是它的众多使用者之一。</li>
    <li>模型<strong>准备</strong>（Python 转换）与<strong>运行</strong>（C++ 推理）完全分离，桥梁就是 <span class="inline">.gguf</span> 文件。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  引擎与模型逻辑分层 + 单头文件公共 API + Python 准备 / C++ 运行解耦——于是 ggml 能独立演进、llama 轻量地嵌进来用、转换脚本也不会拖累运行时。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Open llama.cpp for the first time and a few hundred files across a dozen directories can feel overwhelming.
In fact it is <strong>cleanly layered</strong>: here is a "campus map" so you know what each directory does and
never get lost reading the source.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of the whole repo as a <strong>factory campus</strong>, and the <strong>directories are the map</strong>: each
  workshop (directory) does exactly one job - some build the engine (<span class="mono">ggml</span>), some do the
  assembly (<span class="mono">src/llama-*</span>), some are the storefront (<span class="mono">tools/</span>).
  Reading the map first beats diving head-first into one workshop.
</div>

<h2>Top-level directories at a glance</h2>
<p>Standing at the repo root, here are the main directories by what they do:</p>
<table class="t">
  <tr><th>Directory</th><th>Role</th></tr>
  <tr><td class="mono">ggml/</td><td>The in-house <strong>tensor engine</strong>: tensors · compute graph · ops · backend scheduling; a standalone sub-project (ships its own <code>include/</code> and <code>src/</code>)</td></tr>
  <tr><td class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan ...</td><td>The individual <strong>hardware backends</strong> (plus hip / sycl / musa / opencl and a dozen more)</td></tr>
  <tr><td class="mono">src/</td><td>The <strong>llama inference library</strong>: <code>llama-model-loader</code> · <code>llama-graph</code> · <code>llama-kv-cache</code> · <code>llama-sampler</code> · <code>llama-vocab</code> · <code>llama-chat</code> · <code>llama-grammar</code> · <code>llama-quant</code> ...</td></tr>
  <tr><td class="mono">include/</td><td>The <strong>public C API</strong>: <code>llama.h</code>; <code>llama-cpp.h</code> (a thin C++ wrapper + RAII)</td></tr>
  <tr><td class="mono">common/</td><td><strong>Reusable helpers</strong>: <code>arg</code> (argument parsing) · <code>sampling</code> (sampler wrapper) · <code>chat</code> · <code>log</code> · <code>download</code> · <code>json-schema-to-grammar</code> ...</td></tr>
  <tr><td class="mono">tools/</td><td>The <strong>executable programs</strong>: <span class="mono">llama-cli</span> · <span class="mono">llama-server</span> · <span class="mono">llama-quantize</span> · <code>llama-mtmd-cli</code> (multimodal) · <code>llama-perplexity</code> · <span class="mono">llama-bench</span> ...</td></tr>
  <tr><td class="mono">examples/</td><td>Small example programs (e.g. <code>simple</code>)</td></tr>
  <tr><td class="mono">gguf-py/</td><td>The <strong>Python GGUF read/write library</strong></td></tr>
  <tr><td class="mono">convert_hf_to_gguf.py, etc.</td><td><strong>Python conversion scripts</strong> (4 <code>convert_*.py</code>; the main one is HuggingFace -> GGUF)</td></tr>
  <tr><td class="mono">models/ · tests/ · docs/ · grammars/ · cmake/</td><td>Model data / tests / docs / GBNF examples / build system</td></tr>
</table>

<h2>How they map onto the "four layers"</h2>
<p>Mapping those directories back onto the four-layer structure from the previous lesson makes it click:</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">tools &amp; apps</span><span class="name">tools/ · examples/</span></div>
    <div class="ld"><span class="mono">tools/</span> (cli · server · quantize · mtmd ...) and <span class="mono">examples/</span>: the user-facing CLI, server and samples</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">inference lib</span><span class="name">src/llama-*</span></div>
    <div class="ld"><span class="mono">src/llama-*</span> plus the public header <span class="mono">include/llama.h</span>: loading · compute graph · KV cache · sampling · tokenizer · chat templates</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">engine</span><span class="name">ggml</span></div>
    <div class="ld"><span class="mono">ggml/</span> (<span class="mono">ggml.c</span> · <span class="mono">gguf.cpp</span> · <span class="mono">ggml-alloc</span> · <span class="mono">ggml-backend</span>): tensors · compute graph · ops · scheduling · the GGUF format</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">backends</span><span class="name">CPU · CUDA · Metal · Vulkan ...</span></div>
    <div class="ld"><span class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan ...</span>: actually run the ops on hardware</div></div>
</div>
<p>Besides this main trunk there are two <strong>side-paths</strong>: (1) <strong>model prep</strong> - <span class="mono">gguf-py/</span> plus <span class="mono">convert_*.py</span> (Python) turn a HuggingFace model into a <span class="inline">.gguf</span> file fed to the engine; (2) <strong>support</strong> - <span class="mono">common/</span> (the glue that turns the library into programs) plus <span class="mono">tests/</span> · <span class="mono">docs/</span> · <span class="mono">cmake/</span>.</p>

<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Not sure where to start reading? Pick an entry point by goal: to see <strong>how it is used</strong> -> <span class="mono">tools/</span> and <span class="mono">examples/simple</span>; the <strong>inference logic</strong> -> <span class="mono">src/llama-*</span>; the <strong>low-level ops</strong> -> <span class="mono">ggml/</span>; the <strong>single public contract</strong> -> just <span class="mono">include/llama.h</span>. Find the entry first, then drill down.
</div>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>The repo = <strong>ggml</strong> (engine) + <strong>src/llama-*</strong> (inference lib) + <strong>common</strong> (glue) + <strong>tools</strong> (programs) + <strong>gguf-py / convert_*</strong> (model prep).</li>
    <li>The only public C API is <span class="mono">include/llama.h</span> - the project's <strong>external contract</strong>.</li>
    <li><strong>ggml</strong> is an independent, reusable engine; <strong>llama</strong> is just one of its users.</li>
    <li>Model <strong>prep</strong> (Python conversion) and <strong>run</strong> (C++ inference) are fully separated; the bridge is the <span class="inline">.gguf</span> file.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Engine / model-logic layering + a single-header public API + Python-prep / C++-run decoupling - so ggml can
  evolve independently, llama embeds lightly, and conversion never burdens the runtime.
</div>
""",
}

LESSON_03 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
课 01 给了你一条最小主线（加载 -> 分词 -> 解码循环 -> 采样）。这一课把它<strong>放慢成慢镜头</strong>：
看清<strong>一个 token</strong> 是怎么从 prompt 一步步算出来的，又怎么被<strong>回灌</strong>进队尾，驱动下一步。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一次推理想成<strong>接力赛 / 流水线</strong>：整段 prompt 从起点进入，依次经过几个工位（分词、前向、采样……），
  终点吐出<strong>一个字</strong>；这个字再被接到队伍<strong>尾巴</strong>上，开始下一棒。每跑一圈，只多产出一个字。
</div>

<h2>七步数据流</h2>
<p>把"一段 prompt 变出下一个 token"拆开，正好是这 7 步，从左到右流过去：</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>分词 Tokenize</h4>
    <p>把 prompt 文本切成一串 token id 序列——模型只认数字 id，不认字符。</p>
    <p class="mono">src/llama-vocab.cpp · llama_tokenize</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>组批 Batch</h4>
    <p>把 token 连同它的位置 <span class="mono">pos</span> 和序列号 <span class="mono">seq_id</span> 打包成一次模型输入。</p>
    <p class="mono">src/llama-batch.cpp · llama_batch_get_one</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>解码 Decode（前向）</h4>
    <p><span class="mono">llama_decode</span> 跑一次前向；内部先<strong>建计算图</strong>，再交给<strong>后端</strong>真正算在硬件上。</p>
    <p class="mono">src/llama-context.cpp · llama_decode；建图 src/llama-graph.cpp（llm_graph_*）+ src/llama-model.cpp；执行 ggml-backend</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>取 logits</h4>
    <p>从这次前向里拿到"下一个 token 的分数向量"——词表里每个 token 各有一个分。</p>
    <p class="mono">src/llama-context.cpp · llama_get_logits_ith</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>采样 Sample</h4>
    <p>采样器链（sampler chain）按策略（贪心 / top-k / top-p……）从 logits 里选出一个 token。</p>
    <p class="mono">src/llama-sampler.cpp · llama_sampler_sample</p>
  </div></div>
  <div class="step"><div class="num">6</div><div class="sc">
    <h4>判结束 + 还原文字</h4>
    <p>先用 <span class="mono">llama_vocab_is_eog</span> 判断是不是结束符；不是，就用 <span class="mono">llama_token_to_piece</span> 把 token 还原成文字输出。</p>
    <p class="mono">src/llama-vocab.cpp · llama_vocab_is_eog · llama_token_to_piece</p>
  </div></div>
  <div class="step"><div class="num">7</div><div class="sc">
    <h4>回灌 + 循环</h4>
    <p>把新 token 作为下一步输入再 <span class="mono">decode</span>；过去 token 的 K/V 已存在 KV cache 里，无需重算。</p>
    <p class="mono">src/llama-kv-cache.cpp</p>
  </div></div>
</div>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这个循环其实分两种节奏：<strong>第一次前向（prefill）</strong>把<strong>整段 prompt 并行地</strong>一次算完，
  顺手把每个 token 的 K/V 填进 <strong>KV cache</strong>；之后每一步 <strong>decode</strong> 只算<strong>一个新 token</strong>，
  直接复用缓存里过去的 K/V，<strong>不再重算整段历史</strong>——这就是循环能跑得快、能在本地跑得起来的关键。
</div>

<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  把这 7 步对回上一课的 C API 主线，就是下面这段伪代码骨架，每一行正好对应一步：
<pre class="code"><span class="cm">// 课 01 主线的"慢镜头"：每一步对应一个调用</span>
tokens = <span class="fn">llama_tokenize</span>(vocab, prompt)         <span class="cm">// 1 分词</span>
batch  = <span class="fn">llama_batch_get_one</span>(tokens)           <span class="cm">// 2 组批</span>
<span class="kw">loop</span>:
    <span class="fn">llama_decode</span>(ctx, batch)                  <span class="cm">// 3 前向(内部建图 + 后端执行)</span>
    logits = <span class="fn">llama_get_logits_ith</span>(ctx, -1)     <span class="cm">// 4 取 logits</span>
    id     = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1) <span class="cm">// 5 采样</span>
    <span class="kw">if</span> <span class="fn">llama_vocab_is_eog</span>(vocab, id): <span class="kw">break</span>  <span class="cm">// 6 结束?</span>
    print(<span class="fn">llama_token_to_piece</span>(vocab, id))      <span class="cm">// 6 还原文字</span>
    batch = <span class="fn">llama_batch_get_one</span>([id])          <span class="cm">// 7 回灌; KV cache 记住过去</span></pre>
  <p style="margin:.5rem 0 0">循环体就是"自回归"引擎：每转一圈吐一个 token，直到 <span class="mono">llama_vocab_is_eog</span> 命中结束符才停。</p>
</div>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>一次推理 = <strong>分词 -> 组批 -> 解码(前向) -> 取 logits -> 采样 -> 还原文字 -> 回灌循环</strong>。</li>
    <li>"一次 decode"内部 = <strong>建计算图 + 后端执行</strong>；它的产出是<strong>下一个 token 的 logits</strong>（不是文字，也不是已经选好的 token）。</li>
    <li><strong>prefill</strong> 把整段 prompt 一次性并行算完；之后每一步只算<strong>一个新 token</strong>。</li>
    <li><strong>KV cache</strong> 记住过去的 K/V——这是"循环不重算、能在本地跑"的关键。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  <strong>自回归 + KV cache</strong> 把每一步从"重算整段历史"变成"只算一个新 token"，
  避开了朴素实现里 O(n^2) 的重复计算——这正是本地大模型推理能跑得动的关键原因之一。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Lesson 01 gave you the minimal main line (load -> tokenize -> decode loop -> sample). This lesson plays it in
<strong>slow motion</strong>: how <strong>one token</strong> is produced from the prompt step by step, then
<strong>fed back</strong> to the tail of the queue to drive the next step.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of one inference as a <strong>relay race / assembly line</strong>: the whole prompt enters at the start,
  passes a few stations (tokenize, forward, sample...), and the finish line emits <strong>one character</strong>;
  that character is appended to the <strong>tail</strong> of the queue to start the next leg. One extra character per loop.
</div>

<h2>The 7-step data flow</h2>
<p>Break "turn a prompt into the next token" apart and it is exactly these 7 steps, flowing left to right:</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>Tokenize</h4>
    <p>Cut the prompt text into a sequence of token ids - the model understands numeric ids, not characters.</p>
    <p class="mono">src/llama-vocab.cpp · llama_tokenize</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>Batch</h4>
    <p>Pack the tokens together with their position <span class="mono">pos</span> and sequence id <span class="mono">seq_id</span> into one model input.</p>
    <p class="mono">src/llama-batch.cpp · llama_batch_get_one</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>Decode (forward)</h4>
    <p><span class="mono">llama_decode</span> runs one forward pass; internally it first <strong>builds the compute graph</strong>, then hands it to the <strong>backend</strong> to run on hardware.</p>
    <p class="mono">src/llama-context.cpp · llama_decode; graph build src/llama-graph.cpp (llm_graph_*) + src/llama-model.cpp; execution ggml-backend</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>Get logits</h4>
    <p>Read out the "score vector for the next token" from this forward pass - one score per token in the vocabulary.</p>
    <p class="mono">src/llama-context.cpp · llama_get_logits_ith</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>Sample</h4>
    <p>The sampler chain picks one token from the logits by some strategy (greedy / top-k / top-p...).</p>
    <p class="mono">src/llama-sampler.cpp · llama_sampler_sample</p>
  </div></div>
  <div class="step"><div class="num">6</div><div class="sc">
    <h4>Check end + detokenize</h4>
    <p>First use <span class="mono">llama_vocab_is_eog</span> to test for an end token; if not, <span class="mono">llama_token_to_piece</span> turns the token back into text.</p>
    <p class="mono">src/llama-vocab.cpp · llama_vocab_is_eog · llama_token_to_piece</p>
  </div></div>
  <div class="step"><div class="num">7</div><div class="sc">
    <h4>Feed-back + loop</h4>
    <p>The new token becomes the next step's input and we <span class="mono">decode</span> again; past tokens' K/V already live in the KV cache, so nothing is recomputed.</p>
    <p class="mono">src/llama-kv-cache.cpp</p>
  </div></div>
</div>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The loop actually has two rhythms: the <strong>first forward pass (prefill)</strong> computes the
  <strong>whole prompt in parallel</strong> once, filling each token's K/V into the <strong>KV cache</strong>;
  after that each <strong>decode</strong> step computes only <strong>one new token</strong>, reusing the past K/V
  from the cache instead of <strong>recomputing the whole history</strong> - the key to why the loop is fast and
  can run locally.
</div>

<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Map these 7 steps back onto the previous lesson's C API main line and it is just this pseudo-code skeleton,
  one line per step:
<pre class="code"><span class="cm">// "slow motion" of lesson 01's main line: one call per step</span>
tokens = <span class="fn">llama_tokenize</span>(vocab, prompt)         <span class="cm">// 1 tokenize</span>
batch  = <span class="fn">llama_batch_get_one</span>(tokens)           <span class="cm">// 2 batch</span>
<span class="kw">loop</span>:
    <span class="fn">llama_decode</span>(ctx, batch)                  <span class="cm">// 3 forward (build graph + run on backend)</span>
    logits = <span class="fn">llama_get_logits_ith</span>(ctx, -1)     <span class="cm">// 4 get logits</span>
    id     = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1) <span class="cm">// 5 sample</span>
    <span class="kw">if</span> <span class="fn">llama_vocab_is_eog</span>(vocab, id): <span class="kw">break</span>  <span class="cm">// 6 end?</span>
    print(<span class="fn">llama_token_to_piece</span>(vocab, id))      <span class="cm">// 6 detokenize</span>
    batch = <span class="fn">llama_batch_get_one</span>([id])          <span class="cm">// 7 feed-back; KV cache remembers the past</span></pre>
  <p style="margin:.5rem 0 0">The loop body is the "autoregressive" engine: each turn emits one token, until <span class="mono">llama_vocab_is_eog</span> hits an end token.</p>
</div>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>One inference = <strong>tokenize -> batch -> decode(forward) -> get logits -> sample -> detokenize -> feed-back loop</strong>.</li>
    <li>"One decode" internally = <strong>build compute graph + run on backend</strong>; its output is the <strong>next token's logits</strong> (not text, not an already-chosen token).</li>
    <li><strong>Prefill</strong> computes the whole prompt in parallel once; afterwards each step computes only <strong>one new token</strong>.</li>
    <li>The <strong>KV cache</strong> remembers past K/V - the key to "loop without recompute, runnable locally".</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  <strong>Autoregression + the KV cache</strong> turn each step from "recompute the whole history" into
  "compute just one new token", avoiding the naive O(n^2) recomputation - one of the key reasons local LLM
  inference is feasible at all.
</div>
""",
}
