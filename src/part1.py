"""Content for Part 1 (macro overview). M0 ships lesson 01 as the baseline."""

LESSON_01 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
llama.cpp 是一个用<strong>纯 C/C++</strong> 写的<strong>大模型推理引擎</strong>：把已经训练好的大语言模型
（以 <span class="inline">GGUF</span> 格式存放）<strong>高效地跑起来</strong>，在普通 CPU、甚至手机上也能推理，
有 GPU 就更快。它<strong>不训练</strong>模型，只专注"<strong>把模型跑出字来</strong>"。一个可执行文件加一个
<span class="inline">.gguf</span> 文件，就能在本地、离线、低成本地和大模型对话——这就是它最迷人的地方。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把训练好的大模型想成一张<strong>乐谱</strong>（权重）。PyTorch 像<strong>录音棚</strong>：能作曲、能录、设备重。
  llama.cpp 像一台<strong>便携播放器</strong>：不作曲，只<strong>把乐谱高保真地播放出来</strong>，还特别省电、到处能用。
</div>

<h2>它到底解决什么问题</h2>
<p>研究界的模型大多用 Python + PyTorch，依赖重、显存吃紧、难以部署到普通设备：想在自己的笔记本上跑一个
7B 模型，常被"装不上环境""显存不够""要下载几十 GB 权重"挡在门外。llama.cpp 的目标正相反——把推理这一件事做到极致地<strong>轻</strong>：</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  用<strong>零外部依赖的 C/C++</strong> + <strong>量化</strong>（把权重压成 4/5/8 bit 等低位宽，K-quant 甚至能到 2/3/6 bit）+ <strong>自研张量引擎 ggml</strong>，
  让大模型能在<strong>消费级硬件</strong>上本地、离线、低成本地推理。不需要 Python、不需要 CUDA 工具链、不需要联网，
  一个可执行文件 + 一个 <span class="inline">.gguf</span> 文件即可运行。
</div>

<h2>三大支柱：GGUF · 量化 · ggml</h2>
<p>llama.cpp 能"一个文件到处跑"，靠的是三块拼在一起的基石。这里先各看一眼，后面每一块都会有专门的课展开：</p>

<h3>① GGUF：一个文件装下整个模型</h3>
<p><span class="inline">GGUF</span> 是 llama.cpp 的<strong>模型文件格式</strong>：把权重、超参数（层数、维度、词表大小）、
分词器、聊天模板等<strong>全部打包进一个文件</strong>。加载时直接 <span class="mono">mmap</span> 进内存，不再需要额外的配置文件或
Python 代码——拿到一个 <span class="inline">.gguf</span>，引擎就知道"这是什么模型、该怎么跑"。</p>

<h3>② 量化：把权重压小，精度几乎不掉</h3>
<p>原始权重通常是 16 bit 浮点（FP16），一个 7B 模型就要约 14 GB。<strong>量化</strong>把每个权重压成更低的位宽
（如 4 bit），体积直接降到约 1/4，普通笔记本的内存也装得下。代价是一点点精度损失，但靠"<strong>按块共享缩放</strong>"
的设计，损失小到几乎察觉不到：</p>

<div class="cellgroup">
  <div class="cg-cap"><b>FP16 原始权重</b>：每个数 16 bit，精度高但占空间</div>
  <div class="cells">
    <span class="cell">0.12</span><span class="cell">-0.34</span><span class="cell">0.08</span><span class="cell">0.51</span><span class="cell dim">…</span>
    <span class="lab">一块 32 个 × 16 bit</span>
  </div>
  <div class="cg-cap" style="margin-top:.7rem"><b>Q4_0 量化后</b>：整块共享 1 个 scale，每个权重只存 4 bit 档位；反量化 = scale × (码值 − 8)</div>
  <div class="cells">
    <span class="cell scale">scale</span><span class="sep">×</span>
    <span class="cell q">0110</span><span class="cell q">1001</span><span class="cell q">0011</span><span class="cell q">1100</span><span class="cell q dim">…</span>
    <span class="lab">≈ 4.5 bit/权重，约 1/4 大小</span>
  </div>
</div>

<h3>③ ggml：底层的张量计算引擎</h3>
<p><span class="mono">ggml</span> 是 llama.cpp <strong>自研的张量库</strong>：定义张量、把一次推理描述成<strong>计算图</strong>，
再把图里的算子（矩阵乘、softmax、rope……）派发到不同<strong>后端</strong>（CPU 的 SIMD、CUDA、Metal、Vulkan……）真正算出来。
它<strong>零依赖、可嵌入、可移植</strong>，是整个项目能跑遍各种硬件的根本。</p>

<h2>整体结构图：四层自底向上</h2>
<p>把上面三块支柱按"谁依赖谁"摞起来，llama.cpp 就是一座清晰的<strong>四层塔</strong>，从底层硬件一路往上到用户工具：</p>
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
<p>读源码时记住这条线：<strong>后端</strong>提供算力，<strong>ggml</strong> 把计算组织成图，<strong>src/llama-*</strong>
把图拼成"会话级"的推理逻辑，最外面的 <strong>tools/</strong> 才是你直接敲的命令。下一课会把这四层对应到具体目录。</p>

<h2>训练 vs 推理：llama.cpp 站在哪一边</h2>
<p>一个大模型的一生分两段：先<strong>训练</strong>（把它教会），再<strong>推理</strong>（用它干活）。这是需求完全不同的两件事：</p>
<div class="cols">
  <div class="col">
    <h4>🏋️ 训练（PyTorch 等）</h4>
    <ul>
      <li>前向 + <strong>反向传播</strong>、算梯度、更新权重</li>
      <li>要 <strong>优化器状态</strong>，显存吃紧（常需多卡）</li>
      <li>Python 生态，依赖重</li>
      <li>目标：把模型<strong>练出来</strong></li>
    </ul>
  </div>
  <div class="col">
    <h4>⚡ 推理（llama.cpp）</h4>
    <ul>
      <li><strong>只前向</strong>：权重固定，算一遍出 logits</li>
      <li>可<strong>量化</strong>压显存，CPU 也能跑</li>
      <li>纯 C/C++，几乎零依赖</li>
      <li>目标：把模型<strong>跑出字</strong></li>
    </ul>
  </div>
</div>
<p>llama.cpp 只做<strong>推理</strong>这一半。正因为不必支持反向传播和优化器，它能彻底丢掉训练框架的重依赖，
把整个引擎压成一份纯 C/C++ 代码——这正是它能在你电脑上轻装跑起来的前提。模型怎么"练出来"不归它管，
那是 PyTorch 等训练框架的活。</p>

<h2>和 PyTorch / transformers / vLLM 的区别</h2>
<p>同样和大模型打交道，这几个项目其实站在不同的位置。横向对比一下，就能看清 llama.cpp 独特的生态位：</p>
<table class="t">
  <tr><th>项目</th><th>定位</th><th>语言 / 依赖</th><th>典型场景</th></tr>
  <tr><td><strong>PyTorch</strong></td><td>训练 + 推理框架</td><td>Python，重</td><td>科研、训练</td></tr>
  <tr><td><strong>transformers</strong></td><td>模型库 / 高层封装</td><td>Python，重</td><td>快速实验</td></tr>
  <tr><td><strong>vLLM</strong></td><td>GPU 高吞吐服务</td><td>Python + CUDA</td><td>云端大并发</td></tr>
  <tr><td><strong>llama.cpp</strong></td><td>轻量本地推理</td><td>C/C++，几乎零依赖</td><td>本地 / 边缘 / 嵌入</td></tr>
</table>
<p>一句话总结：要<strong>训练 / 做研究</strong>选 PyTorch，要<strong>云端高并发服务</strong>选 vLLM，要在<strong>本地 / 边缘 / 嵌入式</strong>设备上轻量地把模型跑起来，就选 llama.cpp。它们不是互相取代，而是各司其职。</p>

<h2>怎么真正跑起来</h2>
<p>最快的方式不用写一行代码：下载一个 <span class="inline">.gguf</span>，用命令行工具 <span class="mono">llama-cli</span> 直接对话：</p>
<pre class="code"><span class="cm"># 最快跑起来：一个可执行文件 + 一个 .gguf</span>
llama-cli -m model.gguf -p <span class="st">"用一句话解释量化"</span></pre>
<p>其中 <span class="mono">-m</span> 指定模型文件，<span class="mono">-p</span> 给出提示词；回车之后模型就开始一个字一个字地往外蹦。
那这条命令背后到底发生了什么？拆开看，就是 C API 里的几步：</p>

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
  <p style="margin:.5rem 0 0">这条主线（加载 -&gt; 分词 -&gt; 解码循环 -&gt; 采样）就是后面所有课的骨架，后面会专门用一课展开完整生命周期。</p>
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个常见问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 量化为什么几乎不掉精度？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>示例：</strong>上面的 Q4_0 把 32 个权重分成一<strong>块</strong>，整块共享一个 scale；块内每个权重只存一个 4 bit 的"档位"，
    用时再乘回 scale 还原。关键在于<strong>缩放是按小块算的</strong>，每块都能贴合自己那段数值的范围。</p>
    <p><strong>为什么够用：</strong>神经网络权重大多挤在 0 附近、对单个权重的微小误差并不敏感；按块缩放 + 低位宽，
    就能在"省 4 倍空间"和"几乎不掉精度"之间取得平衡。更进一步的 <strong>K-quant</strong> 常<strong>可选</strong>配合<strong>重要性矩阵（imatrix）</strong>：
    用它给每个权重的量化<strong>误差加权</strong>，让更关键的权重被<strong>更精确地保留</strong>（位宽不变——是误差被加权，而非给它更多比特）。</p>
    <p><strong>源码：</strong>量化与反量化的核心实现在 <span class="mono">ggml/src/ggml-quants.c</span>；重要性矩阵由 <span class="mono">tools/imatrix</span> 统计产出。</p>
    <p><strong>替代：</strong>GPTQ、AWQ 等也是主流量化方案，思路类似（按块 / 按通道缩放），只是格式与具体算法不同。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> ggml 到底是什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>ggml 是一个<strong>张量 + 计算图 + 多后端</strong>的小引擎——定义数据（张量）、把运算组织成图，
    再把图调度到 CPU / GPU 上执行。</p>
    <p><strong>为什么自研：</strong>为了<strong>零依赖、可嵌入、可移植</strong>。不绑定庞大的深度学习框架，一份 C 代码就能编译进任何程序、
    跑遍各种硬件——这是它能"到处跑"的工程基础。</p>
    <p><strong>源码：</strong>核心在 <span class="mono">ggml/</span> 下的 <span class="mono">ggml.c</span>（张量与计算图）和 <span class="mono">ggml-backend</span>（后端抽象与调度）。</p>
    <p><strong>替代：</strong>也可以直接调用 cuBLAS、oneDNN 这类厂商库，但会绑死特定硬件、失去可移植性。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 我的电脑能跑多大的模型？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>有个粗略但好用的估算：<strong>权重占用 ≈ 参数量 × 每权重位宽 ÷ 8</strong>（字节）。把位宽代进去就能心算：</p>
    <ul>
      <li><strong>7B、FP16</strong>（16 bit）≈ 7e9 × 16 ÷ 8 ≈ <strong>14 GB</strong>——多数笔记本扛不住。</li>
      <li><strong>7B、Q4</strong>（约 4.5 bit）≈ <strong>4 GB</strong> 上下——普通笔记本就能本地跑。</li>
    </ul>
    <p>这就是量化最直接的意义：把"装不下"变成"装得下"。实际还要再留一点余量给 <strong>KV cache</strong> 和上下文开销
    （上下文越长占用越多），但量级上，<strong>Q4 让 7B 从"需要显卡"降到"普通内存即可"</strong>。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>llama.cpp = <strong>纯 C/C++ 的大模型推理引擎</strong>，只负责"跑"，不负责"训练"。</li>
    <li>三大支柱：<strong>GGUF 格式</strong>（一个文件装下整个模型）+ <strong>量化</strong>（压小体积）+ <strong>ggml 引擎</strong>（多后端张量计算）。</li>
    <li>整体四层：<strong>后端 -&gt; ggml -&gt; llama 推理 -&gt; 工具</strong>，自底向上。</li>
    <li>训练 vs 推理是两件事：它<strong>只做推理</strong>（只前向），所以能甩掉训练框架的重依赖。</li>
    <li>量化按<strong>块共享 scale</strong>，约省 4 倍空间而几乎不掉精度（Q4 让 7B 从约 14 GB 降到约 4 GB）。</li>
    <li>定位：<strong>本地 / 边缘 / 低成本</strong>，对照 PyTorch（训练）、vLLM（云端高并发）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  把"<strong>推理</strong>"从"<strong>训练框架</strong>"里彻底剥离，再用<strong>量化 + 自研引擎</strong>压掉对 Python 生态与大显存的依赖——
  于是模型<strong>准备</strong>（Python 转 GGUF）和模型<strong>运行</strong>（C/C++ 推理）完全解耦，引擎得以编译成一份零依赖的可执行文件。
  这就是它能"<strong>一个文件到处跑</strong>"的根本原因。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
llama.cpp is an <strong>LLM inference engine written in plain C/C++</strong>: it takes an already-trained
model (stored as a <span class="inline">GGUF</span> file) and <strong>runs it efficiently</strong> - on an
ordinary CPU, even a phone, and faster with a GPU. It does not train models; it only focuses on
"<strong>turning a model into text</strong>". One executable plus one <span class="inline">.gguf</span> file
lets you chat with an LLM locally, offline, and cheaply - and that is what makes it so appealing.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of a trained model as a <strong>music score</strong> (the weights). PyTorch is the <strong>recording studio</strong>:
  it can compose and record, but it is heavy. llama.cpp is a <strong>portable player</strong>: it does not compose,
  it just <strong>plays the score faithfully</strong> - using little power and running almost anywhere.
</div>

<h2>What problem does it solve</h2>
<p>Research models mostly use Python + PyTorch: heavy dependencies, hungry for VRAM, hard to deploy on
ordinary devices. Trying to run a 7B model on your own laptop, you often hit "can't install the
environment", "not enough VRAM", or "tens of GB of weights to download". llama.cpp aims for the
opposite - making inference, that one job, extremely <strong>light</strong>:</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  With <strong>zero-dependency C/C++</strong> + <strong>quantization</strong> (compressing weights to e.g. 4/5/8 bits, down to 2/3/6-bit K-quants) +
  its own tensor engine <strong>ggml</strong>, it makes LLMs run <strong>locally, offline, and cheaply on consumer
  hardware</strong>. No Python, no CUDA toolchain, no network needed: one executable plus one
  <span class="inline">.gguf</span> file is enough.
</div>

<h2>Three pillars: GGUF · quantization · ggml</h2>
<p>The reason llama.cpp can "run anywhere from a single file" is three building blocks fitting together.
Here is a first glance at each; later lessons expand every one of them:</p>

<h3>① GGUF: one file holds the whole model</h3>
<p><span class="inline">GGUF</span> is llama.cpp's <strong>model file format</strong>: it bundles the weights,
hyper-parameters (layers, dimensions, vocab size), tokenizer, chat template, and more <strong>into a single
file</strong>. Loading just <span class="mono">mmap</span>s it into memory - no extra config files or Python code.
Hand the engine one <span class="inline">.gguf</span> and it knows "what model this is and how to run it".</p>

<h3>② Quantization: shrink the weights, keep the accuracy</h3>
<p>Raw weights are usually 16-bit floats (FP16), so a 7B model needs about 14 GB. <strong>Quantization</strong>
packs each weight into a lower bit-width (e.g. 4 bits), cutting the size to roughly 1/4 - small enough for an
ordinary laptop's RAM. The cost is a tiny accuracy loss, but a "<strong>per-block shared scale</strong>"
design keeps that loss almost unnoticeable:</p>

<div class="cellgroup">
  <div class="cg-cap"><b>FP16 raw weights</b>: each number 16 bits - high precision, but bulky</div>
  <div class="cells">
    <span class="cell">0.12</span><span class="cell">-0.34</span><span class="cell">0.08</span><span class="cell">0.51</span><span class="cell dim">…</span>
    <span class="lab">a block of 32 x 16 bit</span>
  </div>
  <div class="cg-cap" style="margin-top:.7rem"><b>After Q4_0</b>: the whole block shares one scale; each weight stores just a 4-bit level; dequant = scale × (code − 8)</div>
  <div class="cells">
    <span class="cell scale">scale</span><span class="sep">×</span>
    <span class="cell q">0110</span><span class="cell q">1001</span><span class="cell q">0011</span><span class="cell q">1100</span><span class="cell q dim">…</span>
    <span class="lab">~4.5 bit/weight, about 1/4 the size</span>
  </div>
</div>

<h3>③ ggml: the low-level tensor engine</h3>
<p><span class="mono">ggml</span> is llama.cpp's <strong>in-house tensor library</strong>: it defines tensors,
describes one inference run as a <strong>compute graph</strong>, then dispatches the ops in that graph
(matmul, softmax, rope...) to different <strong>backends</strong> (CPU SIMD, CUDA, Metal, Vulkan...) to actually
compute. It is <strong>zero-dependency, embeddable, and portable</strong> - the root reason the project runs
across so much hardware.</p>

<h2>Structure map: four layers, bottom-up</h2>
<p>Stack those three pillars by "who depends on whom" and llama.cpp becomes a clean <strong>four-layer tower</strong>,
from the hardware at the bottom up to the user-facing tools:</p>
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
<p>Keep this line in mind when reading the source: the <strong>backend</strong> provides compute, <strong>ggml</strong>
organizes the math into a graph, <strong>src/llama-*</strong> assembles that graph into "session-level" inference
logic, and the outermost <strong>tools/</strong> is what you actually type. The next lesson maps these four layers
onto concrete directories.</p>

<h2>Training vs inference: which side is llama.cpp on</h2>
<p>An LLM's life has two phases: first <strong>training</strong> (teaching it), then <strong>inference</strong>
(putting it to work). These are two jobs with completely different needs:</p>
<div class="cols">
  <div class="col">
    <h4>🏋️ Training (PyTorch, etc.)</h4>
    <ul>
      <li>Forward + <strong>backprop</strong>, compute gradients, update weights</li>
      <li>Needs <strong>optimizer state</strong>, VRAM-hungry (often multi-GPU)</li>
      <li>Python ecosystem, heavy dependencies</li>
      <li>Goal: <strong>produce</strong> the model</li>
    </ul>
  </div>
  <div class="col">
    <h4>⚡ Inference (llama.cpp)</h4>
    <ul>
      <li><strong>Forward only</strong>: weights are fixed, one pass yields logits</li>
      <li>Can <strong>quantize</strong> to save memory; even a CPU runs it</li>
      <li>Plain C/C++, near-zero dependencies</li>
      <li>Goal: <strong>turn</strong> the model into text</li>
    </ul>
  </div>
</div>
<p>llama.cpp does only the <strong>inference</strong> half. Precisely because it need not support backprop or
optimizers, it can drop the heavy training-framework dependencies and compress the whole engine into a single
slab of plain C/C++ - the prerequisite for running light on your machine. How the model is "produced" is not
its concern; that is the job of training frameworks like PyTorch.</p>

<h2>How it differs from PyTorch / transformers / vLLM</h2>
<p>All of these deal with LLMs, yet they sit at different positions. A side-by-side comparison makes
llama.cpp's distinct niche clear:</p>
<table class="t">
  <tr><th>Project</th><th>Role</th><th>Lang / deps</th><th>Typical use</th></tr>
  <tr><td><strong>PyTorch</strong></td><td>Training + inference framework</td><td>Python, heavy</td><td>Research, training</td></tr>
  <tr><td><strong>transformers</strong></td><td>Model library / high-level wrapper</td><td>Python, heavy</td><td>Fast experiments</td></tr>
  <tr><td><strong>vLLM</strong></td><td>High-throughput GPU serving</td><td>Python + CUDA</td><td>Cloud, high concurrency</td></tr>
  <tr><td><strong>llama.cpp</strong></td><td>Lightweight local inference</td><td>C/C++, near-zero deps</td><td>Local / edge / embedded</td></tr>
</table>
<p>In one line: pick PyTorch to <strong>train / do research</strong>, vLLM for <strong>high-concurrency cloud
serving</strong>, and llama.cpp to run a model lightly on <strong>local / edge / embedded</strong> devices. They
do not replace each other - each has its job.</p>

<h2>How to actually run it</h2>
<p>The fastest way needs no code at all: download a <span class="inline">.gguf</span> and chat right from the
command line with <span class="mono">llama-cli</span>:</p>
<pre class="code"><span class="cm"># fastest way to run: one executable + one .gguf</span>
llama-cli -m model.gguf -p <span class="st">"explain quantization in one sentence"</span></pre>
<p>Here <span class="mono">-m</span> points at the model file and <span class="mono">-p</span> gives the prompt;
press enter and the model starts emitting text token by token. So what actually happens behind that command?
Unpacked, it is just these few steps in the C API:</p>

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

<h2>Go deeper (optional)</h2>
<p class="acc-intro">Three common questions below - open them if you want to dig in; skip them if you just want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does quantization barely lose accuracy? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>Example:</strong> the Q4_0 above splits 32 weights into one <strong>block</strong> that shares a single scale;
    each weight in the block stores only a 4-bit "level", multiplied back by the scale when used. The key is that
    <strong>scaling is done per small block</strong>, so each block hugs the value range of its own slice.</p>
    <p><strong>Why it is enough:</strong> network weights mostly cluster near 0 and are insensitive to tiny per-weight
    errors; per-block scaling + low bit-width strikes a balance between "4x smaller" and "barely any accuracy loss".
    <strong>K-quants</strong> can <strong>optionally</strong> pair with an <strong>importance matrix (imatrix)</strong>: it
    <strong>weights</strong> each weight's quantization error so the more important weights are <strong>preserved more
    faithfully</strong> (bit-width is unchanged - the error is weighted, bits are not reallocated).</p>
    <p><strong>Source:</strong> the core quant/dequant code lives in <span class="mono">ggml/src/ggml-quants.c</span>;
    the importance matrix is produced by <span class="mono">tools/imatrix</span>.</p>
    <p><strong>Alternatives:</strong> GPTQ and AWQ are mainstream too, with a similar idea (per-block / per-channel
    scaling) - only the format and exact algorithm differ.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What exactly is ggml? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> ggml is a small engine of <strong>tensors + compute graph + multiple backends</strong> -
    it defines data (tensors), organizes the math into a graph, then schedules that graph onto CPU / GPU to execute.</p>
    <p><strong>Why in-house:</strong> for <strong>zero dependencies, embeddability, and portability</strong>. By not binding
    to a huge deep-learning framework, a single slab of C can compile into any program and run across all kinds of hardware -
    the engineering basis for "running anywhere".</p>
    <p><strong>Source:</strong> the core is under <span class="mono">ggml/</span>: <span class="mono">ggml.c</span> (tensors and
    compute graph) and <span class="mono">ggml-backend</span> (backend abstraction and scheduling).</p>
    <p><strong>Alternatives:</strong> you could call vendor libraries like cuBLAS or oneDNN directly, but that locks you to
    specific hardware and loses portability.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How big a model can my machine run? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>A rough but handy estimate: <strong>weight footprint ~= parameters x bits-per-weight / 8</strong> (bytes). Plug in the
    bit-width and you can do it in your head:</p>
    <ul>
      <li><strong>7B, FP16</strong> (16 bit) ~= 7e9 x 16 / 8 ~= <strong>14 GB</strong> - too much for most laptops.</li>
      <li><strong>7B, Q4</strong> (~4.5 bit) ~= <strong>4 GB</strong> or so - an ordinary laptop runs it locally.</li>
    </ul>
    <p>That is quantization's most direct payoff: turning "won't fit" into "fits". In practice leave some headroom for the
    <strong>KV cache</strong> and context overhead (longer context uses more), but in order of magnitude,
    <strong>Q4 takes 7B from "needs a GPU" down to "ordinary RAM is fine"</strong>.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>llama.cpp = <strong>an LLM inference engine in plain C/C++</strong> - it only "runs", it does not "train".</li>
    <li>Three pillars: <strong>GGUF format</strong> (one file holds the whole model) + <strong>quantization</strong> (shrink the size) + <strong>the ggml engine</strong> (multi-backend tensor compute).</li>
    <li>Four layers: <strong>backend -> ggml -> llama inference -> tools</strong>, bottom-up.</li>
    <li>Training vs inference are two jobs: it does <strong>inference only</strong> (forward only), so it sheds the training framework's heavy deps.</li>
    <li>Quantization shares a <strong>scale per block</strong>, ~4x smaller with barely any accuracy loss (Q4 takes 7B from ~14 GB to ~4 GB).</li>
    <li>Niche: <strong>local / edge / low-cost</strong>, vs PyTorch (training) and vLLM (cloud, high concurrency).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  It cleanly separates <strong>inference</strong> from the <strong>training framework</strong>, then uses
  <strong>quantization + a custom engine</strong> to drop the dependency on the Python ecosystem and large VRAM -
  so model <strong>prep</strong> (Python to GGUF) and model <strong>run</strong> (C/C++ inference) fully decouple, and the
  engine compiles into a single zero-dependency executable. That is why it can "run anywhere from a single file".
</div>
""",
}

LESSON_02 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
第一次打开 llama.cpp，几百个文件、十几个目录，很容易发懵：到底先看哪儿？其实它<strong>分层非常清晰</strong>，每个目录各司其职。
这一课先给你一张"校园地图"——认清顶层目录在干什么，再把它们对回上一课的"四层"，接着看清<strong>一个模型从训练到运行</strong>
要穿过哪几个目录、边界落在哪里。读完这张图，无论你是想跑工具还是钻源码，都不会迷路。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把整个仓库想成一座<strong>工厂园区</strong>，而<strong>目录就是地图</strong>：每个车间（目录）只干一件事——
  有的造引擎（<span class="mono">ggml</span>）、有的负责装配（<span class="mono">src/llama-*</span>）、有的是对外门店（<span class="mono">tools/</span>），
  还有的是把原料运进园区的<strong>码头</strong>（<span class="mono">convert_*.py</span> 把外面的模型转成 <span class="inline">.gguf</span>）。
  先认地图、看清车间之间怎么衔接，比一头扎进某个车间更重要。
</div>

<h2>顶层目录速览</h2>
<p>站在仓库根目录，先别急着点开文件，按"<strong>这个目录到底在干什么</strong>"把主要目录扫一遍。下面这张表就是地图的图例：</p>
<table class="t">
  <tr><th>目录</th><th>作用</th></tr>
  <tr><td class="mono">ggml/</td><td>自研<strong>张量引擎</strong>：张量 · 计算图 · 算子 · 后端调度；独立子项目（自带 <code>include/</code> 与 <code>src/</code>），是整个项目最底层、也最"硬"的一块</td></tr>
  <tr><td class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …</td><td>各<strong>硬件后端</strong>，把算子真正算在 CPU / GPU 上（还有 hip / sycl / musa / opencl 等十余种）</td></tr>
  <tr><td class="mono">src/</td><td><strong>llama 推理库</strong>：把引擎拼成"会话级"推理——<code>llama-model-loader</code> · <code>llama-graph</code> · <code>llama-kv-cache</code> · <code>llama-sampler</code> · <code>llama-vocab</code> · <code>llama-chat</code> · <code>llama-grammar</code> · <code>llama-quant</code> …</td></tr>
  <tr><td class="mono">include/</td><td><strong>公共 C API</strong>：<code>llama.h</code>（唯一对外契约）；<code>llama-cpp.h</code>（C++ 薄封装 + RAII）</td></tr>
  <tr><td class="mono">common/</td><td><strong>复用工具 / 胶水</strong>：<code>arg</code>（参数解析）· <code>sampling</code>（采样封装）· <code>chat</code> · <code>log</code> · <code>download</code> · <code>json-schema-to-grammar</code> …；给程序用，不是推理库本体</td></tr>
  <tr><td class="mono">tools/</td><td><strong>可执行程序</strong>：<span class="mono">llama-cli</span> · <span class="mono">llama-server</span> · <span class="mono">llama-quantize</span> · <code>llama-mtmd-cli</code>（多模态）· <code>llama-perplexity</code> · <span class="mono">llama-bench</span> …</td></tr>
  <tr><td class="mono">examples/</td><td>小型示例程序：<code>simple</code> 用两百多行演示最小推理，是最佳"可读"入口</td></tr>
  <tr><td class="mono">gguf-py/</td><td><strong>Python 的 GGUF 读写库</strong>：转换脚本靠它写出 <span class="inline">.gguf</span></td></tr>
  <tr><td class="mono">convert_hf_to_gguf.py 等</td><td><strong>Python 转换脚本</strong>（共 4 个 <code>convert_*.py</code>：3 个转换器 + 1 个 tokenizer 哈希维护脚本；主力是 HuggingFace -&gt; GGUF）</td></tr>
  <tr><td class="mono">models/ · tests/ · docs/ · grammars/ · cmake/</td><td>模型数据 / 测试 / 文档 / GBNF 示例 / 构建系统</td></tr>
</table>
<p>一个简单的判断法：<strong>越往下越"硬"</strong>——<span class="mono">ggml/</span> 偏数学与硬件，<span class="mono">src/</span> 偏模型与会话逻辑，<span class="mono">tools/</span> 与 <span class="mono">common/</span> 偏"给人用"。而真正对外暴露的，自始至终只有 <span class="mono">include/llama.h</span> 这一个公共头文件。</p>

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

<h2>一个模型怎么从训练到运行</h2>
<p>把目录串起来看，一个模型的"一生"其实是一条很直的流水线：左边在 Python 里准备，右边在 C++ 里运行，中间靠一个文件交接。顺着这条线走一遍，就知道每个目录在整条链路里站在哪一站：</p>
<div class="flow">
  <div class="node"><div class="nt">HF / PyTorch 模型</div><div class="nd">safetensors 权重</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">convert_hf_to_gguf.py</div><div class="nd">Python · gguf-py</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">model.gguf</div><div class="nd">单文件 · 权重 + 元数据</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_model_load_from_file</div><div class="nd">C++ 运行时</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">跑出字</div><div class="nd">llama-cli / server</div></div>
</div>
<p class="acc-intro">左半截是 <strong>Python（准备）</strong>，右半截是 <strong>C++（运行）</strong>，两者的边界就是中间那个 <span class="inline">.gguf</span> 文件。</p>
<p>这条边界很关键：转换脚本只在<strong>准备</strong>阶段跑一次，产出 <span class="inline">.gguf</span> 后就退场；运行时<strong>完全不碰 Python</strong>，只认这一个文件。所以同一个 <span class="inline">.gguf</span>，既能喂给 <span class="mono">llama-cli</span>，也能喂给 <span class="mono">llama-server</span> 或 <span class="mono">examples/simple</span>——它们共享同一套加载与推理代码。</p>

<h2>想读源码，从哪进</h2>
<p>想读源码，却不知道从哪下手？与其从第一个文件啃到最后一个，不如<strong>先想清楚你的目标</strong>，再选对应的入口往下钻。常见的三种目标，正好对应三个入口：</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">想会用</span><span class="name">tools/ · examples/simple</span></div>
    <div class="ld">先把 <code>llama-cli</code>/<code>llama-server</code> 跑起来，再读 <code>examples/simple</code> 的最小调用</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">懂推理</span><span class="name">src/llama-*</span></div>
    <div class="ld">按主线读：<code>llama-model-loader</code> -&gt; <code>llama-graph</code> -&gt; <code>llama-kv-cache</code> -&gt; <code>llama-sampler</code></div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">懂算子</span><span class="name">ggml/</span></div>
    <div class="ld">进引擎：<code>ggml.c</code> 与各 <code>ggml-*</code> 后端，看张量/算子/调度怎么实现</div></div>
</div>
<p>不管从哪条路进，记住对外只有一个<strong>公共契约</strong> <span class="mono">include/llama.h</span>：搞不清某个能力归谁管时，先回到这个头文件，看它把哪些函数暴露给了外面。先认入口，再逐层往下钻，比漫无目的地翻文件高效得多。</p>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想把这张地图看透的同学点开看；只想记住主干的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> GGUF 文件里到底装了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>示例：</strong>一个 <span class="inline">.gguf</span> 从头到尾大致分四段：<strong>文件头</strong>（magic + 版本 + 张量数 / 元数据条数）-&gt; <strong>元数据 KV</strong>（一串键值对：层数、维度、词表大小、分词器、聊天模板…）-&gt; <strong>张量信息表</strong>（每个权重张量的名字、形状、类型、偏移）-&gt; <strong>权重数据块</strong>（真正的数值，按张量信息里的偏移排布）。</p>
    <p><strong>为什么这么设计：</strong>全部塞进<strong>一个文件</strong>，加载时直接 <span class="mono">mmap</span> 进内存、按偏移随用随取，不必先解压或拷贝（CPU 推理时近乎零拷贝；用 GPU 后端则权重还会再拷进显存）；超参数与词表都<strong>自带</strong>，引擎读完头部就知道"这是什么模型、该怎么搭计算图"，<strong>免配置文件、免 Python</strong>。</p>
    <p><strong>源码：</strong>读写与解析在 <span class="mono">ggml/src/gguf.cpp</span>（<code>gguf_kv</code> / <code>gguf_tensor_info</code> 等结构）；把这些元数据接到 llama 模型上、按 key 取超参的，是 <span class="mono">src/llama-model-loader.cpp</span>。</p>
    <p><strong>替代：</strong>更早的 <strong>GGML / GGJT</strong> 等老格式也干过同样的活，但字段零散、版本兼容差，<strong>已被 GGUF 取代</strong>（仓库里还留着一个 <code>convert_llama_ggml_to_gguf.py</code> 专门把老格式转过来）。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么 ggml 是独立子项目？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>因为<strong>同一个引擎被多个项目复用</strong>——ggml 不只为 llama.cpp 服务，所以它被切成一个能单独存在的子项目。</p>
    <p><strong>例子：</strong>同作者的 <span class="mono">whisper.cpp</span>（语音转文字）等项目也直接拿 ggml 当计算引擎；它们和 llama.cpp 共享同一套张量、算子与后端代码，只是上层逻辑不同。</p>
    <p><strong>源码：</strong><span class="mono">ggml/</span> 自带完整的 <code>include/</code> 与 <code>src/</code>，对外的张量 / 计算图 / 后端接口是独立的一套，不依赖 <span class="mono">src/llama-*</span> 里的任何东西——依赖是<strong>单向</strong>的：llama 用 ggml，反过来不成立。</p>
    <p><strong>好处：</strong>引擎可以<strong>独立演进</strong>（加新算子、新后端不必动 llama），也<strong>便于嵌入</strong>到任何想要本地张量计算的程序里；llama 只是它众多使用者中的一个。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> common/ 和 src/ 有啥区别？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong><span class="mono">common/</span> 是各个可执行程序<strong>共用的胶水</strong>，<strong>不是</strong>推理库本体；真正的推理逻辑住在 <span class="mono">src/llama-*</span>。</p>
    <p><strong>它管什么：</strong>命令行参数解析、把 C API 的采样接口包成更顺手的封装、日志、下载模型、聊天模板拼接……这些是"把库变成一个能用的程序"要反复写的活，抽到 <span class="mono">common/</span> 让 <span class="mono">tools/</span> 里每个程序都能复用。</p>
    <p><strong>它不管什么：</strong>加载权重、搭计算图、KV cache、真正的采样算法——这些都在 <span class="mono">src/llama-*</span> 里，对外只通过 <span class="mono">include/llama.h</span> 暴露。换句话说，删掉 <span class="mono">common/</span> 推理库照样能用，只是你得自己手写一堆样板代码。</p>
    <p><strong>源码：</strong>参数解析看 <span class="mono">common/arg.cpp</span>，其余通用工具看 <span class="mono">common/common.cpp</span>。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>仓库 = <strong>ggml</strong>（引擎）+ <strong>src/llama-*</strong>（推理库）+ <strong>common</strong>（胶水）+ <strong>tools / examples</strong>（程序）+ <strong>gguf-py / convert_*</strong>（模型准备）。</li>
    <li>这些目录对回<strong>四层</strong>：后端 -&gt; ggml -&gt; src/llama-* -&gt; tools，自底向上各管一段。</li>
    <li>一个模型的一生是条流水线：<strong>Python 准备</strong> -&gt; <span class="inline">.gguf</span> -&gt; <strong>C++ 运行</strong>，边界就是那个单文件。</li>
    <li>读源码<strong>按目标选入口</strong>：想会用看 <span class="mono">tools/</span> 与 <span class="mono">examples/simple</span>，懂推理看 <span class="mono">src/llama-*</span>，懂算子看 <span class="mono">ggml/</span>。</li>
    <li>对外只有一个公共 C API：<span class="mono">include/llama.h</span>——整个项目的<strong>外部契约</strong>。</li>
    <li><strong>ggml</strong> 是独立、可复用的引擎；<strong>common</strong> 是胶水、不是推理本体——两者都别和 <span class="mono">src/llama-*</span> 搞混。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  引擎与模型逻辑分层 + 单头文件公共 API + Python 准备 / C++ 运行解耦——于是 ggml 能独立演进、被 <span class="mono">whisper.cpp</span> 等项目复用，llama 轻量地嵌进来用，转换脚本也不会拖累运行时。一张<strong>清晰的目录地图</strong>背后，其实是一组刻意划好的<strong>边界</strong>：谁依赖谁、谁对外、谁只是胶水，全摆在明面上。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Open llama.cpp for the first time and a few hundred files across a dozen directories can feel overwhelming -
where do you even start? In fact it is <strong>cleanly layered</strong>, every directory with one job. This lesson
hands you a "campus map": learn what the top-level directories do, map them back onto the four layers from the
previous lesson, then watch how <strong>a model travels from training to running</strong> - which directories it
passes through and where the boundary falls. Once you have read the map you will not get lost, whether you want
to run a tool or dig into the source.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of the whole repo as a <strong>factory campus</strong>, and the <strong>directories are the map</strong>: each
  workshop (directory) does exactly one job - some build the engine (<span class="mono">ggml</span>), some do the
  assembly (<span class="mono">src/llama-*</span>), some are the storefront (<span class="mono">tools/</span>), and
  some are the <strong>loading dock</strong> that brings raw material in (<span class="mono">convert_*.py</span> turns
  an outside model into a <span class="inline">.gguf</span>). Reading the map - and seeing how the workshops
  connect - beats diving head-first into one workshop.
</div>

<h2>Top-level directories at a glance</h2>
<p>Standing at the repo root, do not rush to open files; first scan the main directories by "<strong>what does this
directory actually do</strong>". The table below is the map's legend:</p>
<table class="t">
  <tr><th>Directory</th><th>Role</th></tr>
  <tr><td class="mono">ggml/</td><td>The in-house <strong>tensor engine</strong>: tensors · compute graph · ops · backend scheduling; a standalone sub-project (ships its own <code>include/</code> and <code>src/</code>) - the lowest and "hardest" layer of the whole project</td></tr>
  <tr><td class="mono">ggml/src/ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan ...</td><td>The individual <strong>hardware backends</strong> that actually run the ops on CPU / GPU (plus hip / sycl / musa / opencl and a dozen more)</td></tr>
  <tr><td class="mono">src/</td><td>The <strong>llama inference library</strong>: assembles the engine into "session-level" inference - <code>llama-model-loader</code> · <code>llama-graph</code> · <code>llama-kv-cache</code> · <code>llama-sampler</code> · <code>llama-vocab</code> · <code>llama-chat</code> · <code>llama-grammar</code> · <code>llama-quant</code> ...</td></tr>
  <tr><td class="mono">include/</td><td>The <strong>public C API</strong>: <code>llama.h</code> (the only external contract); <code>llama-cpp.h</code> (a thin C++ wrapper + RAII)</td></tr>
  <tr><td class="mono">common/</td><td><strong>Reusable helpers / glue</strong>: <code>arg</code> (argument parsing) · <code>sampling</code> (sampler wrapper) · <code>chat</code> · <code>log</code> · <code>download</code> · <code>json-schema-to-grammar</code> ...; for the programs, not the inference library itself</td></tr>
  <tr><td class="mono">tools/</td><td>The <strong>executable programs</strong>: <span class="mono">llama-cli</span> · <span class="mono">llama-server</span> · <span class="mono">llama-quantize</span> · <code>llama-mtmd-cli</code> (multimodal) · <code>llama-perplexity</code> · <span class="mono">llama-bench</span> ...</td></tr>
  <tr><td class="mono">examples/</td><td>Small example programs: <code>simple</code> demonstrates minimal inference in a couple hundred lines - the best "readable" entry point</td></tr>
  <tr><td class="mono">gguf-py/</td><td>The <strong>Python GGUF read/write library</strong>: the conversion scripts use it to write out a <span class="inline">.gguf</span></td></tr>
  <tr><td class="mono">convert_hf_to_gguf.py, etc.</td><td><strong>Python conversion scripts</strong> (4 <code>convert_*.py</code>: 3 converters + 1 tokenizer-hash updater; the main one is HuggingFace -&gt; GGUF)</td></tr>
  <tr><td class="mono">models/ · tests/ · docs/ · grammars/ · cmake/</td><td>Model data / tests / docs / GBNF examples / build system</td></tr>
</table>
<p>A simple rule of thumb: <strong>the lower you go, the "harder" it gets</strong> - <span class="mono">ggml/</span> leans toward math and hardware, <span class="mono">src/</span> toward model and session logic, <span class="mono">tools/</span> and <span class="mono">common/</span> toward "for people to use". The only thing ever exposed to the outside, start to finish, is the single public header <span class="mono">include/llama.h</span>.</p>

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

<h2>How a model travels from training to running</h2>
<p>String the directories together and a model's "life" is really a straight pipeline: prepared in Python on the left, run in C++ on the right, handed over through one file in the middle. Walk it once and you will see which station each directory occupies along the whole chain:</p>
<div class="flow">
  <div class="node"><div class="nt">HF / PyTorch model</div><div class="nd">safetensors weights</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">convert_hf_to_gguf.py</div><div class="nd">Python · gguf-py</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">model.gguf</div><div class="nd">single file · weights + metadata</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_model_load_from_file</div><div class="nd">C++ runtime</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">emit text</div><div class="nd">llama-cli / server</div></div>
</div>
<p class="acc-intro">The left half is <strong>Python (prepare)</strong>, the right half is <strong>C++ (run)</strong>, and the boundary between them is exactly that <span class="inline">.gguf</span> file in the middle.</p>
<p>That boundary matters: the conversion script runs <strong>once</strong> during prep, emits the <span class="inline">.gguf</span>, and then bows out; the runtime <strong>never touches Python</strong> and knows only this one file. So the same <span class="inline">.gguf</span> can feed <span class="mono">llama-cli</span>, <span class="mono">llama-server</span>, or <span class="mono">examples/simple</span> alike - they share the same loading and inference code.</p>

<h2>Want to read the source - where to enter</h2>
<p>Want to read the source but not sure where to start? Rather than chewing from the first file to the last, <strong>decide your goal first</strong>, then pick the matching entry point and drill down. Three common goals map to three entries:</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">to use it</span><span class="name">tools/ · examples/simple</span></div>
    <div class="ld">First get <code>llama-cli</code>/<code>llama-server</code> running, then read the minimal call in <code>examples/simple</code></div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">to grok inference</span><span class="name">src/llama-*</span></div>
    <div class="ld">Follow the main line: <code>llama-model-loader</code> -&gt; <code>llama-graph</code> -&gt; <code>llama-kv-cache</code> -&gt; <code>llama-sampler</code></div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">to grok the ops</span><span class="name">ggml/</span></div>
    <div class="ld">Into the engine: <code>ggml.c</code> and the <code>ggml-*</code> backends - how tensors/ops/scheduling are implemented</div></div>
</div>
<p>Whichever path you take, remember there is only one <strong>public contract</strong>, <span class="mono">include/llama.h</span>: when you cannot tell which part owns some capability, go back to this header and see which functions it exposes to the outside. Find the entry first, then drill down layer by layer - far more efficient than flipping through files at random.</p>

<h2>Go deeper (optional)</h2>
<p class="acc-intro">Three questions below - open them if you want to see the whole map clearly; skip them if you just want the main trunk.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What is actually inside a GGUF file? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>Example:</strong> a <span class="inline">.gguf</span> is roughly four sections end to end: a <strong>header</strong> (magic + version + tensor count / metadata count) -&gt; <strong>metadata KV</strong> (a list of key-value pairs: layers, dimensions, vocab size, tokenizer, chat template...) -&gt; a <strong>tensor info table</strong> (each weight tensor's name, shape, type, offset) -&gt; <strong>weight data blocks</strong> (the actual numbers, laid out by the offsets in the tensor info).</p>
    <p><strong>Why this design:</strong> packing everything into <strong>one file</strong> means loading just <span class="mono">mmap</span>s it into memory and reads on demand by offset - no unpacking or copying first (near-zero-copy for CPU inference; with a GPU backend the weights are then copied into VRAM); the hyper-parameters and vocab are <strong>built in</strong>, so once the engine reads the header it knows "what model this is and how to build the compute graph", with <strong>no config files and no Python</strong>.</p>
    <p><strong>Source:</strong> reading/parsing lives in <span class="mono">ggml/src/gguf.cpp</span> (the <code>gguf_kv</code> / <code>gguf_tensor_info</code> structs); wiring that metadata onto a llama model and fetching hyper-parameters by key is <span class="mono">src/llama-model-loader.cpp</span>.</p>
    <p><strong>Alternatives:</strong> the earlier <strong>GGML / GGJT</strong> formats did the same job, but with scattered fields and poor version compatibility - now <strong>superseded by GGUF</strong> (the repo still keeps a <code>convert_llama_ggml_to_gguf.py</code> just to migrate the old format over).</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is ggml a standalone sub-project? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> because <strong>the same engine is reused by several projects</strong> - ggml does not serve only llama.cpp, so it is carved out as a sub-project that can stand on its own.</p>
    <p><strong>Example:</strong> same-author projects like <span class="mono">whisper.cpp</span> (speech-to-text) use ggml directly as their compute engine; they share the very same tensor, op, and backend code as llama.cpp, only the upper layer differs.</p>
    <p><strong>Source:</strong> <span class="mono">ggml/</span> ships its own complete <code>include/</code> and <code>src/</code>; its tensor / graph / backend interface is a self-contained set that depends on nothing in <span class="mono">src/llama-*</span> - the dependency is <strong>one-way</strong>: llama uses ggml, never the reverse.</p>
    <p><strong>Benefit:</strong> the engine can <strong>evolve independently</strong> (new ops or backends without touching llama) and is <strong>easy to embed</strong> in any program that wants local tensor compute; llama is just one of its many users.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What is the difference between common/ and src/? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> <span class="mono">common/</span> is the <strong>glue shared</strong> by the executable programs - it is <strong>not</strong> the inference library itself; the real inference logic lives in <span class="mono">src/llama-*</span>.</p>
    <p><strong>What it handles:</strong> command-line argument parsing, wrapping the C API's sampler into something handier, logging, downloading models, assembling chat templates... the boilerplate every "turn the library into a usable program" needs, factored into <span class="mono">common/</span> so each program in <span class="mono">tools/</span> can reuse it.</p>
    <p><strong>What it does not:</strong> it does <strong>not</strong> load weights, build the compute graph, manage the KV cache, or implement the actual sampling algorithms - those all live in <span class="mono">src/llama-*</span> and are exposed only through <span class="mono">include/llama.h</span>. In other words, delete <span class="mono">common/</span> and the inference library still works; you would just hand-write a pile of boilerplate yourself.</p>
    <p><strong>Source:</strong> for argument parsing see <span class="mono">common/arg.cpp</span>; for the rest of the shared helpers see <span class="mono">common/common.cpp</span>.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>The repo = <strong>ggml</strong> (engine) + <strong>src/llama-*</strong> (inference lib) + <strong>common</strong> (glue) + <strong>tools / examples</strong> (programs) + <strong>gguf-py / convert_*</strong> (model prep).</li>
    <li>Those directories map back onto the <strong>four layers</strong>: backend -&gt; ggml -&gt; src/llama-* -&gt; tools, bottom-up, each owning one slice.</li>
    <li>A model's life is a pipeline: <strong>Python prep</strong> -&gt; <span class="inline">.gguf</span> -&gt; <strong>C++ run</strong>; the single file is the boundary.</li>
    <li>Read the source <strong>by goal</strong>: to use it look at <span class="mono">tools/</span> and <span class="mono">examples/simple</span>, to grok inference <span class="mono">src/llama-*</span>, to grok the ops <span class="mono">ggml/</span>.</li>
    <li>The only public C API is <span class="mono">include/llama.h</span> - the project's <strong>external contract</strong>.</li>
    <li><strong>ggml</strong> is an independent, reusable engine; <strong>common</strong> is glue, not the inference core - do not confuse either with <span class="mono">src/llama-*</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Engine / model-logic layering + a single-header public API + Python-prep / C++-run decoupling - so ggml can
  evolve independently and be reused by projects like <span class="mono">whisper.cpp</span>, llama embeds lightly, and
  conversion never burdens the runtime. Behind one <strong>clean directory map</strong> sits a set of deliberately
  drawn <strong>boundaries</strong>: who depends on whom, who faces outward, who is just glue - all out in the open.
</div>
""",
}

LESSON_03 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
课 01 给了你一条最小主线（加载 -&gt; 分词 -&gt; 解码循环 -&gt; 采样）。这一课把它<strong>放慢成慢镜头</strong>：
看清<strong>一个 token</strong> 是怎么从 prompt 一步步算出来的，又怎么被<strong>回灌</strong>进队尾、驱动下一步。
我们先走一遍<strong>七步数据流</strong>，再<strong>放大其中最重的"一次 decode"</strong>——看清它内部其实是"先建计算图、再交后端执行、最后吐出 logits"；
最后用 <strong>prefill / decode 两种节奏</strong>和 <strong>KV cache</strong>，讲清为什么这个循环能在本地<strong>便宜地一圈圈转下去</strong>。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把一次推理想成<strong>接力赛 / 流水线</strong>：整段 prompt 从起点进入，依次经过几个工位（分词、前向、采样……），
  终点吐出<strong>一个字</strong>；这个字再被接到队伍<strong>尾巴</strong>上，开始下一棒。每跑一圈，只多产出一个字——
  而队伍前面那些<strong>已经算过的字</strong>不用重新排队，这正是后面 <strong>KV cache</strong> 要替我们守住的"便宜"。
</div>

<h2>七步数据流</h2>
<p>把"一段 prompt 变出下一个 token"拆开，正好是这 7 步，从上到下顺次流过去：</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>分词 Tokenize</h4>
    <p>把 prompt 文本切成一串 token id 序列——模型只认数字 id，不认字符；同一句话用不同分词器切出的 id 可能完全不同。</p>
    <p class="mono">src/llama-vocab.cpp · llama_tokenize</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>组批 Batch</h4>
    <p>把 token 序列包成一次输入（batch）；用 <span class="mono">llama_batch_get_one</span> 时，位置 <span class="mono">pos</span> 与序列 <span class="mono">seq_id</span> 由 <span class="mono">llama_decode</span> 自动补（位置顺序排、序列固定为 0），需要多序列 / 自定义位置时才用 <span class="mono">llama_batch_init</span>。</p>
    <p class="mono">src/llama-batch.cpp · llama_batch_get_one</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>解码 Decode（前向）</h4>
    <p><span class="mono">llama_decode</span> 跑一次前向；内部先<strong>建计算图</strong>，再交给<strong>后端</strong>真正算在硬件上。这一步最重，下一节会专门把它<strong>放大</strong>看。</p>
    <p class="mono">src/llama-context.cpp · llama_decode；建图 src/llama-graph.cpp（llm_graph_*）+ src/llama-model.cpp；执行 ggml-backend</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>取 logits</h4>
    <p>从这次前向里拿到"下一个 token 的分数向量"——词表里每个 token 各有一个分。注意此刻还<strong>没有</strong>选定任何 token。</p>
    <p class="mono">src/llama-context.cpp · llama_get_logits_ith</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>采样 Sample</h4>
    <p>采样器链（sampler chain）按策略（贪心 / top-k / top-p……）从 logits 里选出一个 token；策略不同，同一份 logits 也会选出不同的字。</p>
    <p class="mono">src/llama-sampler.cpp · llama_sampler_sample</p>
  </div></div>
  <div class="step"><div class="num">6</div><div class="sc">
    <h4>判结束 + 还原文字</h4>
    <p>先用 <span class="mono">llama_vocab_is_eog</span> 判断是不是结束符；不是，就用 <span class="mono">llama_token_to_piece</span> 把 token 还原成文字输出。</p>
    <p class="mono">src/llama-vocab.cpp · llama_vocab_is_eog · llama_token_to_piece</p>
  </div></div>
  <div class="step"><div class="num">7</div><div class="sc">
    <h4>回灌 + 循环</h4>
    <p>把新 token 作为下一步输入再 <span class="mono">decode</span>；过去 token 的 K/V 已存在 KV cache 里，无需重算——于是循环每转一圈只多算一个 token。</p>
    <p class="mono">src/llama-kv-cache.cpp</p>
  </div></div>
</div>

<h2>放大第 3 步：一次 decode 内部</h2>
<p>七步里最"重"的是第 3 步 <span class="mono">decode</span>。别把它当成一个黑盒——拆开看，一次前向内部正好是三小步串起来：</p>
<div class="flow">
  <div class="node"><div class="nt">llama_decode</div><div class="nd">一次前向</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">建计算图</div><div class="nd">llama-graph.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">后端执行</div><div class="nd">ggml-backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits</div><div class="nd">llama_get_logits_ith</div></div>
</div>
<p>所以"一次 decode"= <strong>建计算图 + 后端执行 -&gt; logits</strong>：先由 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">llm_graph_*</span>（经 <span class="mono">src/llama-model.cpp</span> 的 <span class="mono">build_graph</span> 拼出）把这一步运算<strong>描述成一张图</strong>，再交 <span class="mono">ggml-backend</span> 调度到硬件上<strong>真正算</strong>，算完用 <span class="mono">llama_get_logits_ith</span> 取出"下一个 token 的分数向量"。它的产出是 <strong>logits</strong>——不是文字，也不是已经选好的 token。</p>

<h2>prefill vs decode：两种节奏</h2>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这个循环其实分两种节奏：<strong>第一次前向（prefill）</strong>把<strong>整段 prompt 并行地</strong>一次算完，
  顺手把每个 token 的 K/V 填进 <strong>KV cache</strong>；之后每一步 <strong>decode</strong> 只算<strong>一个新 token</strong>，
  直接复用缓存里过去的 K/V，<strong>不再重算整段历史</strong>——这就是循环能跑得快、能在本地跑得起来的关键。
</div>
<p>把这两种节奏摆到一条时间线上，差别一眼就清楚：prefill 是<strong>一段宽</strong>的并行块，decode 是<strong>一格一格</strong>往后接的小步。</p>
<div class="timeline">
  <div class="lane"><span class="lane-label">Prefill</span><span class="tslot span">整段 prompt（t1…t5）一次并行算，填满 KV cache</span></div>
  <div class="lane"><span class="lane-label">Decode</span><span class="tslot">t6</span><span class="tslot">t7</span><span class="tslot now">t8…</span></div>
</div>
<p class="acc-intro">Prefill 把整段提示词<strong>一次并行</strong>算完；之后每步 decode <strong>只算 1 个新 token</strong>，所以"接着往下写"很便宜。</p>

<h2>KV cache 为什么让循环不贵</h2>
<p>承上：decode 之所以每步只算一个新 token，靠的就是 <strong>KV cache</strong>。它把每个算过的 token 的 <strong>K/V</strong> 存下来，下一步直接复用，免去重算整段历史：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache</b>：每生成一个 token 就把它的 K/V 追加进缓存，下一步直接复用、不重算历史</div>
  <div class="cells"><span class="lab">prefill 后</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span><span class="cell">K4</span><span class="cell">K5</span></div>
  <div class="cells"><span class="lab">decode t6</span><span class="cell dim">K1…K5（复用）</span><span class="cell hl">+K6</span></div>
  <div class="cells"><span class="lab">decode t7</span><span class="cell dim">K1…K6（复用）</span><span class="cell hl">+K7</span></div>
</div>
<p>每一行只多出<strong>一个高亮新格</strong>（<span class="mono">+K6</span>、<span class="mono">+K7</span>），前面灰掉的部分都是"<strong>复用、不重算</strong>"。没有这层缓存，生成第 n 个 token 就得把前面 n-1 个全重算一遍；有了它，每步的<strong>新增计算</strong>基本是常数——这就是自回归循环能在本地便宜地一直转下去的原因。</p>

<h2>对回最小主线</h2>
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
  <p style="margin:.5rem 0 0">第 4 步的 <span class="mono">logits</span>：真实代码里采样器会自己从 <span class="mono">ctx</span> 取，这里单独列出只为把"产出 logits"这一步看清楚。</p>
  <p style="margin:.5rem 0 0">循环体就是"自回归"引擎：每转一圈吐一个 token，直到 <span class="mono">llama_vocab_is_eog</span> 命中结束符才停。</p>
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个常见问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 decode 输出的是 logits、不是文字？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>一次前向只算到"打分"为止。<span class="mono">logits</span> 是<strong>词表上每个 token 的"分数向量"</strong>——词表多大它就多长，每个 token 一个分，谁高谁低而已，还没"拍板"。</p>
    <p><strong>选哪个是另一步：</strong>从这串分数里挑出一个 token，是<strong>采样器</strong>的事（<span class="mono">src/llama-sampler.cpp</span> 的 <span class="mono">llama_sampler_sample</span>，按贪心 / top-k / top-p 等策略选）；把选中的 token 再<strong>还原成文字</strong>，是 <span class="mono">llama_token_to_piece</span>（<span class="mono">src/llama-vocab.cpp</span>）的事。</p>
    <p><strong>为什么这么分：</strong>把"打分 / 选词 / 还原文字"三件事拆开，采样策略就能随意替换而不动前向——同一份 logits，换个采样器就有不同风格的输出。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> KV cache 到底省了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>省的是"重算过去"：</strong>没有它，生成第 n 个 token 要把前 n-1 个<strong>重算一遍</strong> -&gt; 总成本约 <span class="mono">O(n^2)</span>；有了它，每步只算<strong>新 token</strong> 的 K/V 并追加进缓存 -&gt; 约 <span class="mono">O(n)</span>。</p>
    <p><strong>注意别夸大：</strong>注意力对历史的<strong>扫描</strong>仍是每步 <span class="mono">O(n)</span>（要看过去所有 token），省掉的是<strong>重复计算过去 token 的 K/V</strong>，不是把注意力也变成常数。</p>
    <p><strong>源码：</strong>缓存的分配、写入与复用在 <span class="mono">src/llama-kv-cache.cpp</span>；上下文越长，这块占用越大，也是本地推理要预留内存的地方。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> "计算图"是什么？为什么先建图再算？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong><span class="mono">ggml</span> 先把这一步运算<strong>描述成一张图</strong>（节点是算子：matmul、rope、softmax……，边是数据流），再交后端按图执行。</p>
    <p><strong>谁来建：</strong>图由 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">llm_graph_*</span> 搭骨架、由 <span class="mono">src/llama-model.cpp</span> 的 <span class="mono">build_graph</span> 按具体模型结构拼出；建好后交 <span class="mono">ggml-backend</span> 调度执行。</p>
    <p><strong>为什么分两步：</strong>把"<strong>描述</strong>运算"与"<strong>执行</strong>运算"分开，同一张图就能落到不同后端（CPU / CUDA / Metal……）上跑，也便于做内存复用、算子融合等优化——这是 ggml 能"一处描述、多端执行"的根。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 本课要点</div>
  <ul>
    <li>一次推理 = <strong>分词 -&gt; 组批 -&gt; 解码(前向) -&gt; 取 logits -&gt; 采样 -&gt; 还原文字 -&gt; 回灌循环</strong>。</li>
    <li>"一次 decode"内部 = <strong>建计算图 + 后端执行</strong>；它的产出是<strong>下一个 token 的 logits</strong>（不是文字，也不是已经选好的 token）——选词靠采样器，还原文字靠 <span class="mono">llama_token_to_piece</span>。</li>
    <li><strong>prefill</strong> 把整段 prompt 一次性<strong>并行</strong>算完；之后每一步 <strong>decode</strong> 只算<strong>一个新 token</strong>。</li>
    <li><strong>KV cache</strong> 记住过去的 K/V，把朴素实现的 <span class="mono">O(n^2)</span> 重算摊成约 <span class="mono">O(n)</span>（但注意力对历史的扫描仍是每步 <span class="mono">O(n)</span>）——这是"循环不重算、能在本地跑"的关键。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计亮点</div>
  <strong>自回归 + KV cache</strong> 把每一步从"重算整段历史"变成"只算一个新 token"，
  避开了朴素实现里 <span class="mono">O(n^2)</span> 的重复计算；再加上 <strong>decode 内部"先建图、后执行"</strong>的分层，
  让同一套推理既能把单步成本压到接近常数，又能落到各种后端上跑——这正是本地大模型推理能跑得动的关键原因之一。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Lesson 01 gave you the minimal main line (load -&gt; tokenize -&gt; decode loop -&gt; sample). This lesson plays it in
<strong>slow motion</strong>: how <strong>one token</strong> is produced from the prompt step by step, then
<strong>fed back</strong> to the tail of the queue to drive the next step. We first walk the <strong>7-step data flow</strong>,
then <strong>zoom into the heaviest part - "one decode"</strong> - to see it is really "build a compute graph, run it on the
backend, then emit logits"; finally we use the <strong>prefill / decode rhythms</strong> and the <strong>KV cache</strong> to
explain why this loop can keep turning <strong>cheaply, round after round</strong>, on local hardware.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of one inference as a <strong>relay race / assembly line</strong>: the whole prompt enters at the start,
  passes a few stations (tokenize, forward, sample...), and the finish line emits <strong>one character</strong>;
  that character is appended to the <strong>tail</strong> of the queue to start the next leg. One extra character per loop -
  and the characters <strong>already computed</strong> at the front never have to queue up again, which is exactly the
  "cheapness" the <strong>KV cache</strong> will preserve for us later.
</div>

<h2>The 7-step data flow</h2>
<p>Break "turn a prompt into the next token" apart and it is exactly these 7 steps, flowing top to bottom:</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>Tokenize</h4>
    <p>Cut the prompt text into a sequence of token ids - the model understands numeric ids, not characters; the same sentence can split into completely different ids under a different tokenizer.</p>
    <p class="mono">src/llama-vocab.cpp · llama_tokenize</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>Batch</h4>
    <p>Wrap the token sequence into one input (batch); with <span class="mono">llama_batch_get_one</span>, <span class="mono">pos</span>/<span class="mono">seq_id</span> are auto-assigned by <span class="mono">llama_decode</span> (sequential positions, sequence 0) - use <span class="mono">llama_batch_init</span> only when you need multiple sequences / custom positions.</p>
    <p class="mono">src/llama-batch.cpp · llama_batch_get_one</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>Decode (forward)</h4>
    <p><span class="mono">llama_decode</span> runs one forward pass; internally it first <strong>builds the compute graph</strong>, then hands it to the <strong>backend</strong> to run on hardware. This is the heaviest step - the next section <strong>zooms into</strong> it.</p>
    <p class="mono">src/llama-context.cpp · llama_decode; graph build src/llama-graph.cpp (llm_graph_*) + src/llama-model.cpp; execution ggml-backend</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>Get logits</h4>
    <p>Read out the "score vector for the next token" from this forward pass - one score per token in the vocabulary. At this point <strong>no</strong> token has been chosen yet.</p>
    <p class="mono">src/llama-context.cpp · llama_get_logits_ith</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>Sample</h4>
    <p>The sampler chain picks one token from the logits by some strategy (greedy / top-k / top-p...); a different strategy can pick a different token from the very same logits.</p>
    <p class="mono">src/llama-sampler.cpp · llama_sampler_sample</p>
  </div></div>
  <div class="step"><div class="num">6</div><div class="sc">
    <h4>Check end + detokenize</h4>
    <p>First use <span class="mono">llama_vocab_is_eog</span> to test for an end token; if not, <span class="mono">llama_token_to_piece</span> turns the token back into text.</p>
    <p class="mono">src/llama-vocab.cpp · llama_vocab_is_eog · llama_token_to_piece</p>
  </div></div>
  <div class="step"><div class="num">7</div><div class="sc">
    <h4>Feed-back + loop</h4>
    <p>The new token becomes the next step's input and we <span class="mono">decode</span> again; past tokens' K/V already live in the KV cache, so nothing is recomputed - each turn of the loop adds just one token of work.</p>
    <p class="mono">src/llama-kv-cache.cpp</p>
  </div></div>
</div>

<h2>Zoom into step 3: inside one decode</h2>
<p>The heaviest of the 7 steps is step 3, <span class="mono">decode</span>. Don't treat it as a black box - opened up, one forward pass is exactly three little steps chained together:</p>
<div class="flow">
  <div class="node"><div class="nt">llama_decode</div><div class="nd">one forward pass</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">build graph</div><div class="nd">llama-graph.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">run on backend</div><div class="nd">ggml-backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits</div><div class="nd">llama_get_logits_ith</div></div>
</div>
<p>So "one decode" = <strong>build graph + run on backend -&gt; logits</strong>: first <span class="mono">llm_graph_*</span> in <span class="mono">src/llama-graph.cpp</span> (assembled by <span class="mono">build_graph</span> in <span class="mono">src/llama-model.cpp</span>) <strong>describes</strong> this step's computation <strong>as a graph</strong>, then <span class="mono">ggml-backend</span> schedules it onto hardware to <strong>actually compute</strong>, and afterwards <span class="mono">llama_get_logits_ith</span> reads out the "score vector for the next token". Its output is <strong>logits</strong> - not text, and not an already-chosen token.</p>

<h2>prefill vs decode: two rhythms</h2>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The loop actually has two rhythms: the <strong>first forward pass (prefill)</strong> computes the
  <strong>whole prompt in parallel</strong> once, filling each token's K/V into the <strong>KV cache</strong>;
  after that each <strong>decode</strong> step computes only <strong>one new token</strong>, reusing the past K/V
  from the cache instead of <strong>recomputing the whole history</strong> - the key to why the loop is fast and
  can run locally.
</div>
<p>Put the two rhythms on one timeline and the difference is obvious at a glance: prefill is <strong>one wide</strong> parallel block, decode is <strong>cell-by-cell</strong> steps appended after it.</p>
<div class="timeline">
  <div class="lane"><span class="lane-label">Prefill</span><span class="tslot span">whole prompt (t1…t5) computed in parallel at once, filling the KV cache</span></div>
  <div class="lane"><span class="lane-label">Decode</span><span class="tslot">t6</span><span class="tslot">t7</span><span class="tslot now">t8…</span></div>
</div>
<p class="acc-intro">Prefill computes the whole prompt <strong>in parallel</strong> in one pass; afterwards each decode step computes <strong>just 1 new token</strong>, so "keep writing" is cheap.</p>

<h2>Why the KV cache keeps the loop cheap</h2>
<p>Following on: the reason decode computes only one new token per step is the <strong>KV cache</strong>. It stores each computed token's <strong>K/V</strong> so the next step reuses them directly, sparing a recompute of the whole history:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache</b>: each generated token appends its K/V to the cache; the next step reuses them directly instead of recomputing history</div>
  <div class="cells"><span class="lab">after prefill</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span><span class="cell">K4</span><span class="cell">K5</span></div>
  <div class="cells"><span class="lab">decode t6</span><span class="cell dim">K1…K5 (reuse)</span><span class="cell hl">+K6</span></div>
  <div class="cells"><span class="lab">decode t7</span><span class="cell dim">K1…K6 (reuse)</span><span class="cell hl">+K7</span></div>
</div>
<p>Each row adds only <strong>one highlighted new cell</strong> (<span class="mono">+K6</span>, <span class="mono">+K7</span>); everything greyed out before it is "<strong>reused, not recomputed</strong>". Without this cache, generating the n-th token would recompute all n-1 before it; with it, the <strong>added work</strong> per step is essentially constant - that is why an autoregressive loop can keep running cheaply on local hardware.</p>

<h2>Back to the minimal main line</h2>
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
  <p style="margin:.5rem 0 0">In real code the sampler reads the logits from <span class="mono">ctx</span> itself; the explicit step-4 <span class="mono">logits</span> line is shown only to make the "produce logits" step visible.</p>
  <p style="margin:.5rem 0 0">The loop body is the "autoregressive" engine: each turn emits one token, until <span class="mono">llama_vocab_is_eog</span> hits an end token.</p>
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three common questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does decode output logits, not text? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> a forward pass only goes as far as "scoring". <span class="mono">logits</span> is a <strong>score vector over the whole vocabulary</strong> - as long as the vocab is big, with one score per token; it just says who is higher or lower, nothing is "decided" yet.</p>
    <p><strong>Picking is a separate step:</strong> choosing one token out of those scores is the <strong>sampler</strong>'s job (<span class="mono">llama_sampler_sample</span> in <span class="mono">src/llama-sampler.cpp</span>, by greedy / top-k / top-p...); turning the chosen token <strong>back into text</strong> is <span class="mono">llama_token_to_piece</span>'s job (<span class="mono">src/llama-vocab.cpp</span>).</p>
    <p><strong>Why split it:</strong> separating "score / pick / detokenize" lets the sampling strategy be swapped freely without touching the forward pass - same logits, a different sampler, a different style of output.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What exactly does the KV cache save? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>It saves "recomputing the past":</strong> without it, generating the n-th token would <strong>recompute the previous n-1</strong> -&gt; total cost about <span class="mono">O(n^2)</span>; with it, each step only computes the <strong>new token</strong>'s K/V and appends it -&gt; about <span class="mono">O(n)</span>.</p>
    <p><strong>Don't overstate it:</strong> attention's <strong>scan</strong> over history is still <span class="mono">O(n)</span> per step (it must look at all past tokens); what is saved is <strong>recomputing past tokens' K/V</strong>, not turning attention itself into a constant.</p>
    <p><strong>Source:</strong> allocation, writing and reuse of the cache live in <span class="mono">src/llama-kv-cache.cpp</span>; the longer the context, the bigger this footprint - and the memory you must reserve for local inference.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What is a "compute graph", and why build then run? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> <span class="mono">ggml</span> first <strong>describes</strong> this step's computation <strong>as a graph</strong> (nodes are operators: matmul, rope, softmax...; edges are data flow), then the backend executes the graph.</p>
    <p><strong>Who builds it:</strong> the graph skeleton comes from <span class="mono">llm_graph_*</span> in <span class="mono">src/llama-graph.cpp</span>, assembled per the concrete model structure by <span class="mono">build_graph</span> in <span class="mono">src/llama-model.cpp</span>; once built it is handed to <span class="mono">ggml-backend</span> to schedule and run.</p>
    <p><strong>Why two steps:</strong> separating "<strong>describe</strong>" from "<strong>execute</strong>" lets the same graph run on different backends (CPU / CUDA / Metal...), and makes optimizations like memory reuse and operator fusion possible - the root of ggml's "describe once, run on many backends".</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>One inference = <strong>tokenize -&gt; batch -&gt; decode(forward) -&gt; get logits -&gt; sample -&gt; detokenize -&gt; feed-back loop</strong>.</li>
    <li>"One decode" internally = <strong>build compute graph + run on backend</strong>; its output is the <strong>next token's logits</strong> (not text, not an already-chosen token) - picking is the sampler's job, detokenizing is <span class="mono">llama_token_to_piece</span>'s.</li>
    <li><strong>Prefill</strong> computes the whole prompt <strong>in parallel</strong> once; afterwards each <strong>decode</strong> step computes only <strong>one new token</strong>.</li>
    <li>The <strong>KV cache</strong> remembers past K/V, amortizing the naive <span class="mono">O(n^2)</span> recompute down to about <span class="mono">O(n)</span> (though attention's scan over history is still <span class="mono">O(n)</span> per step) - the key to "loop without recompute, runnable locally".</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  <strong>Autoregression + the KV cache</strong> turn each step from "recompute the whole history" into
  "compute just one new token", avoiding the naive <span class="mono">O(n^2)</span> recomputation; together with decode's internal
  <strong>"build graph, then execute"</strong> split, the same inference can both shrink per-step cost toward constant and
  run on a variety of backends - one of the key reasons local LLM inference is feasible at all.
</div>
""",
}
