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

<div class="cols">
  <div class="col"><h4>研究界常见栈</h4><p>Python + PyTorch · 依赖重 · 吃显存 · 常要 CUDA · 难部署到普通设备</p></div>
  <div class="col"><h4>llama.cpp</h4><p>零依赖 C/C++ · 量化省显存 · CPU 也能跑 · 离线 · 一个可执行文件 + 一个 .gguf</p></div>
</div>
<p><strong>为什么偏偏是 C/C++，而不是 Python、Rust 或 Go？</strong>推理引擎最在意两件事——<strong>零运行时</strong>与<strong>可嵌入</strong>，
而 C/C++ 恰好两样都占。它直接编译成机器码，不背 Python 解释器、也没有 Go/Java 那种垃圾回收和虚拟机运行时，<strong>启动即算</strong>、延迟可控；
它能<strong>静态链接成一个独立可执行文件</strong>，拷到另一台同架构机器上、不装任何依赖就能跑。更关键的是几乎所有语言都能通过
<span class="mono">C ABI</span> 调用它，于是这套引擎可以被<strong>嵌进桌面 App、安卓/iOS、浏览器（编译成 WASM）甚至嵌入式设备</strong>，
这是重运行时语言很难做到的。Rust 其实也能达成类似目标，但项目起步时 C/C++ 的编译器与各家 GPU 工具链最成熟、移植阻力最小。
官方 README 开宗明义就把目标写成"<strong>以最小的依赖</strong>实现顶尖性能"，纯 C/C++ 正是这句话的支点。</p>

<h2>三大支柱：GGUF · 量化 · ggml</h2>
<p>llama.cpp 能"一个文件到处跑"，靠的是三块拼在一起的基石。这里先各看一眼，后面每一块都会有专门的课展开：</p>

<h3>① GGUF：一个文件装下整个模型</h3>
<p><span class="inline">GGUF</span> 是 llama.cpp 的<strong>模型文件格式</strong>：把权重、超参数（层数、维度、词表大小）、
分词器、聊天模板等<strong>全部打包进一个文件</strong>。加载时直接 <span class="mono">mmap</span> 进内存，不再需要额外的配置文件或
Python 代码——拿到一个 <span class="inline">.gguf</span>，引擎就知道"这是什么模型、该怎么跑"。</p>
<p><strong>"单文件"到底好在哪？</strong>传统做法要同时凑齐权重分片、<span class="mono">config.json</span>、<span class="mono">tokenizer.json</span>、
生成配置等一堆文件，少一个就跑不起来、版本错一个就对不上；GGUF 把它们<strong>全焊进一个文件</strong>，于是<strong>免配置</strong>——
引擎读文件头里以键值对存放的元数据（超参、词表、chat 模板）就知道该怎么跑。加载时用 <span class="mono">mmap</span> 把文件
<strong>按需映射</strong>进内存，用到哪一页操作系统才读哪一页，启动快、还能让多个进程<strong>共享同一份只读权重</strong>省内存。
最实在的好处是<strong>换模型只换一个文件</strong>：把 <span class="inline">.gguf</span> 一换、命令行参数原封不动，跑的就是另一个模型了，分发和管理都简单到极点。</p>

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
<p><strong>为什么能"既省又几乎不掉质量"？</strong>三件事叠在一起：其一，模型里<strong>权重的数量远多于运行时的激活值</strong>，
把权重压成低位宽，省下的空间最多、对计算精度的牵连却最小；其二，"预测下一个词"看的是各候选 logits 的<strong>相对高低</strong>，
对单个权重的微小误差并不敏感，低位宽带来的抖动大多被淹没；其三，<strong>按块共享 scale</strong> 让每一小块都贴合自己那段数值的范围，
牢牢<strong>保住动态范围</strong>。代价确实存在——一点点质量损失，但可用<strong>重要性矩阵</strong>把更关键的权重保留得更准，把损失再压下去。</p>
<p>把这套压缩放进真实场景，就是一条"<strong>从重到轻</strong>"的流水线：同一个模型，量化后体积骤降，再配上一个可执行文件，就能从数据中心搬到你的笔记本上：</p>
<div class="flow">
  <div class="node"><div class="nt">FP16 模型</div><div class="nd">7B ≈ 14 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">量化 Q4</div><div class="nd">≈ 4 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">.gguf 单文件</div><div class="nd">+ 一个可执行文件</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">本地跑</div><div class="nd">笔记本 / 手机 / 服务器</div></div>
</div>

<h3>③ ggml：底层的张量计算引擎</h3>
<p><span class="mono">ggml</span> 是 llama.cpp <strong>自研的张量库</strong>：定义张量、把一次推理描述成<strong>计算图</strong>，
再把图里的算子（矩阵乘、softmax、rope……）派发到不同<strong>后端</strong>（CPU 的 SIMD、CUDA、Metal、Vulkan……）真正算出来。
它把“<strong>描述运算</strong>”和“<strong>在硬件上执行</strong>”分开，于是同一套模型代码不改，就能跑遍 CPU 和各种 GPU。</p>

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
<p>方向上有个对称的美感：你的<strong>请求自上而下</strong>穿过四层（工具收到提示词 -&gt; 推理层组织成计算图 -&gt; ggml 安排算子 -&gt; 后端落到硬件），
算出的<strong>结果再自下而上</strong>冒回来、最终变成屏幕上的文字。读源码时若一时迷路，回到这张图、先认清"自己正站在哪一层"，往往就不慌了。</p>

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
<p><strong>"只前向"具体意味着什么？</strong>训练时为了在<strong>反向传播</strong>里回算梯度，要把前向过程中每一层的中间结果都留着，
还要给每个权重维护<strong>优化器状态</strong>（如 Adam 的一阶、二阶动量），算下来显存常是权重本身的好几倍。推理把这些<strong>全砍掉</strong>了：
没有反向、没有梯度、没有优化器状态，权重在加载后<strong>只读、不再变化</strong>，显存里实质上只剩两样东西——<strong>权重</strong>和 <strong>KV cache</strong>。
正因为权重只读，才可以放心地把它<strong>量化</strong>成低位宽而不必担心影响训练；也正因为卸掉了训练那套重负担，整件事才轻到
<strong>能在一颗普通 CPU 上跑起来</strong>。这就是"只做推理"换来的全部底气。</p>

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
<p>还有个常被忽略的事实，最能说明它的"<strong>地基</strong>"地位：许多你熟悉的本地大模型桌面工具——<strong>Ollama</strong>、<strong>LM Studio</strong>、Jan、KoboldCpp、LocalAI 等——
底层很大程度上就在调用 llama.cpp（它们大多就列在 llama.cpp 自己 README 的"UIs"清单里）。也就是说，你也许从没直接敲过它的命令，
却很可能<strong>每天都在间接用它</strong>；它更像整条本地推理生态的<strong>发动机</strong>，而不是一个孤立的命令行玩具。</p>

<h2>怎么真正跑起来</h2>
<p>最快的方式不用写一行代码：下载一个 <span class="inline">.gguf</span>，用命令行工具 <span class="mono">llama-cli</span> 直接对话：</p>
<pre class="code"><span class="cm"># 最快跑起来：一个可执行文件 + 一个 .gguf</span>
llama-cli -m model.gguf -p <span class="st">"用一句话解释量化"</span></pre>
<p>其中 <span class="mono">-m</span> 指定模型文件，<span class="mono">-p</span> 给出提示词；回车之后模型就开始一个字一个字地往外蹦。
想要一个能用浏览器访问的"本地 ChatGPT"，把 <span class="mono">llama-cli</span> 换成 <span class="mono">llama-server</span> 即可——它把同一套推理逻辑包成 HTTP 接口，对外提供兼容 OpenAI 的 API。
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
<div class="flow">
  <div class="node"><div class="nt">加载模型</div><div class="nd">llama_model_load<br>_from_file</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">分词</div><div class="nd">llama_tokenize<br>文本 -&gt; token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">解码循环</div><div class="nd">llama_decode<br>算下一 token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">采样</div><div class="nd">sampler_sample<br>挑一个词</div></div>
</div>
<p>这里有个容易忽略的细节：解码循环每一轮只把<strong>上一个新 token</strong> 喂回去，而不是把整段历史重算一遍——靠的正是
<strong>KV cache</strong> 把先前每个 token 算出的键/值缓存了下来。所以第一次要把整段 prompt 整体过一遍（<strong>prefill</strong>），<strong>首个字出现前要稍等一下</strong>（这就是“首 token 延迟”）；
之后逐字生成（<strong>decode</strong>）却很快，每步只做一个 token 的前向。<span class="mono">llama_decode</span> 负责"前向算一步"，
<span class="mono">llama_sampler_sample</span> 负责"按概率挑一个词"，两者一前一后交替，就织出了你看到的逐字输出。</p>

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
    <p><strong>算笔账：</strong>Q4_0 的一块正好是 <span class="mono">QK4_0 = 32</span> 个权重，存储上是 1 个 FP16 的 scale（2 字节）+ 32 个 4 bit 码值（16 字节）= 共 <strong>18 字节</strong>；
    平摊到每个权重就是 <strong>18×8÷32 = 4.5 bit</strong>，这正是"约 1/4 大小"的由来（原始 FP16 是 16 bit）。块大小取 32 是个折中：太大则一个 scale 盖不住整块的数值范围、误差变大，太小则每块都要单存一个 scale、开销摊不薄。</p>
    <p><strong>实战里怎么选：</strong>真正常用的往往不是最朴素的 Q4_0，而是 <span class="mono">Q4_K_M</span> 这类 <strong>K-quant</strong>——
    它对不同层用不同位宽、并把 scale 本身也量化，质量/体积比更好；想更接近原始精度就上 Q5_K_M、Q6_K，想更省内存就降到 Q3_K_M。
    一句话：<strong>位宽越高越像原模型、越低越省内存</strong>，Q4 附近通常是体感上的"甜点档"。</p>
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
    <p><strong>换个角度看自研的必要性：</strong>若直接搬 PyTorch 这类训练框架，会背上几百 MB 的依赖和一整套 Python 运行时，根本塞不进手机或嵌入式设备；
    而推理真正用到的其实只是一小撮算子（矩阵乘、softmax、rope、各种归一化……）。ggml 索性只实现这一小撮，再给每种硬件配一套后端——
    CPU 的 SIMD 指令、CUDA、Metal、Vulkan、HIP 等——同一张计算图<strong>换个后端就能换硬件</strong>。这种可移植性，是绑死某一家厂商的闭源库怎么都换不来的。</p>
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
    <p><strong>再往大了看：</strong>同一条公式套到 <strong>70B</strong> 上，FP16 约 140 GB、即便 Q4 也要 <strong>约 40 GB</strong>——单张消费级显卡根本装不下；
    这时要么换内存更大的机器，要么把一部分层<strong>卸载（offload）</strong>到 CPU 内存，或干脆用多机/多卡来分担。还要记住 <strong>KV cache</strong> 会<strong>随上下文长度线性增长</strong>：
    上下文拉得很长时，它的占用甚至能和权重本身同一个量级，所以"能跑多大"从来不只看权重，得给上下文留足余量。</p>
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

<div class="cols">
  <div class="col"><h4>Typical research stack</h4><p>Python + PyTorch · heavy deps · VRAM-hungry · often needs CUDA · hard to deploy on ordinary devices</p></div>
  <div class="col"><h4>llama.cpp</h4><p>zero-dep C/C++ · quantization saves VRAM · runs on CPU too · offline · one executable + one .gguf</p></div>
</div>
<p><strong>Why C/C++ specifically, and not Python, Rust, or Go?</strong> An inference engine cares about two things above all - a
<strong>zero runtime</strong> and being <strong>embeddable</strong> - and C/C++ delivers both. It compiles straight to machine code, with no
Python interpreter and none of the garbage collection or VM runtime of Go/Java, so it <strong>computes the moment it starts</strong> with
predictable latency. It can be <strong>statically linked into one standalone executable</strong> that runs on another same-architecture machine
with zero dependencies installed. Crucially, almost any language can call it through the <span class="mono">C ABI</span>, so the engine can be
<strong>embedded into desktop apps, Android/iOS, the browser (compiled to WASM), even embedded devices</strong> - hard to pull off in a
heavy-runtime language. Rust could reach a similar goal, but when the project started the C/C++ compilers and each vendor's GPU toolchains were the
most mature and the easiest to port to. The official README states the goal up front as "<strong>minimal setup</strong> with top performance", and
plain C/C++ is the linchpin of that sentence.</p>

<h2>Three pillars: GGUF · quantization · ggml</h2>
<p>The reason llama.cpp can "run anywhere from a single file" is three building blocks fitting together.
Here is a first glance at each; later lessons expand every one of them:</p>

<h3>① GGUF: one file holds the whole model</h3>
<p><span class="inline">GGUF</span> is llama.cpp's <strong>model file format</strong>: it bundles the weights,
hyper-parameters (layers, dimensions, vocab size), tokenizer, chat template, and more <strong>into a single
file</strong>. Loading just <span class="mono">mmap</span>s it into memory - no extra config files or Python code.
Hand the engine one <span class="inline">.gguf</span> and it knows "what model this is and how to run it".</p>
<p><strong>What is so good about "one file"?</strong> The traditional way needs weight shards, <span class="mono">config.json</span>,
<span class="mono">tokenizer.json</span>, a generation config and more all present at once - miss one and nothing runs, mismatch a version and things break.
GGUF <strong>welds them all into a single file</strong>, so it is <strong>config-free</strong>: the engine reads the metadata stored as key-value pairs in
the file header (hyper-parameters, vocab, chat template) and knows exactly how to run. Loading <span class="mono">mmap</span>s the file in
<strong>on demand</strong> - the OS reads a page only when it is touched - so startup is fast and multiple processes can <strong>share one read-only
copy</strong> of the weights to save memory. The most practical payoff: <strong>swapping models means swapping one file</strong> - replace the
<span class="inline">.gguf</span>, leave the command-line flags untouched, and you are running a different model. Distribution and management become
trivially simple.</p>

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
<p><strong>Why can it "save space yet barely lose quality"?</strong> Three things stack up. First, a model has <strong>far more weights than runtime
activations</strong>, so compressing the weights to a low bit-width saves the most space while touching compute precision the least. Second,
"predicting the next token" depends on the <strong>relative ranking</strong> of the candidate logits, which is insensitive to tiny per-weight errors -
the jitter from low bit-width mostly washes out. Third, a <strong>per-block shared scale</strong> lets each small block hug the value range of its own
slice, firmly <strong>preserving the dynamic range</strong>. There is a cost - a little quality loss - but an <strong>importance matrix</strong> can keep
the more critical weights more accurate and push that loss down further.</p>
<p>Put that compression in a real setting and it is a "<strong>heavy-to-light</strong>" pipeline: the same model shrinks sharply after quantization,
and paired with one executable it can move from the data center onto your laptop:</p>
<div class="flow">
  <div class="node"><div class="nt">FP16 model</div><div class="nd">7B ≈ 14 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">Quantize Q4</div><div class="nd">≈ 4 GB</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">.gguf single file</div><div class="nd">+ one executable</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">Run locally</div><div class="nd">laptop / phone / server</div></div>
</div>

<h3>③ ggml: the low-level tensor engine</h3>
<p><span class="mono">ggml</span> is llama.cpp's <strong>in-house tensor library</strong>: it defines tensors,
describes one inference run as a <strong>compute graph</strong>, then dispatches the ops in that graph
(matmul, softmax, rope...) to different <strong>backends</strong> (CPU SIMD, CUDA, Metal, Vulkan...) to actually
compute. By separating "<strong>describing the math</strong>" from "<strong>running it on hardware</strong>", the same
model code runs unchanged across CPU and all kinds of GPUs.</p>

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
<p>There is a pleasing symmetry to the direction: your <strong>request flows top-down</strong> through the four layers (a tool receives the prompt
-&gt; the inference layer organizes it into a compute graph -&gt; ggml arranges the ops -&gt; a backend lands them on hardware), and the computed
<strong>result flows bottom-up</strong> back into text on your screen. If you ever get lost reading the source, return to this picture and first pin
down "which layer am I in" - it usually settles the panic.</p>

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
<p><strong>What does "forward only" concretely mean?</strong> To recompute gradients during <strong>backprop</strong>, training keeps every layer's
intermediate results from the forward pass, and maintains <strong>optimizer state</strong> per weight (e.g. Adam's first- and second-moment momentum) -
adding up to several times the memory of the weights themselves. Inference <strong>cuts all of that</strong>: no backward pass, no gradients, no
optimizer state, and the weights are <strong>read-only and never change</strong> after loading, so memory really only holds two things -
<strong>weights</strong> and the <strong>KV cache</strong>. Because the weights are read-only, you can safely <strong>quantize</strong> them to a low
bit-width without worrying about training; and because the heavy training burden is gone, the whole thing is light enough to <strong>run on a single
ordinary CPU</strong>. That is the entire confidence that "inference only" buys.</p>

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
<p>One often-overlooked fact best captures its "<strong>foundation</strong>" status: many of the local-LLM desktop tools you know -
<strong>Ollama</strong>, <strong>LM Studio</strong>, Jan, KoboldCpp, LocalAI and more - are, to a large degree, calling llama.cpp underneath (most of
them are listed right in llama.cpp's own README under "UIs"). In other words, you may never have typed its commands directly, yet you very likely
<strong>use it indirectly every day</strong>; it is less a standalone command-line toy and more the <strong>engine</strong> under the whole
local-inference ecosystem.</p>

<h2>How to actually run it</h2>
<p>The fastest way needs no code at all: download a <span class="inline">.gguf</span> and chat right from the
command line with <span class="mono">llama-cli</span>:</p>
<pre class="code"><span class="cm"># fastest way to run: one executable + one .gguf</span>
llama-cli -m model.gguf -p <span class="st">"explain quantization in one sentence"</span></pre>
<p>Here <span class="mono">-m</span> points at the model file and <span class="mono">-p</span> gives the prompt;
press enter and the model starts emitting text token by token. Want a browser-accessible "local ChatGPT"? Swap
<span class="mono">llama-cli</span> for <span class="mono">llama-server</span> - it wraps the same inference logic into an HTTP service exposing an
OpenAI-compatible API. So what actually happens behind that command?
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
<div class="flow">
  <div class="node"><div class="nt">load model</div><div class="nd">llama_model_load<br>_from_file</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tokenize</div><div class="nd">llama_tokenize<br>text -&gt; token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">decode loop</div><div class="nd">llama_decode<br>next token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">sample</div><div class="nd">sampler_sample<br>pick a word</div></div>
</div>
<p>One easily missed detail: each turn of the decode loop only feeds back the <strong>single new token</strong> rather than recomputing the whole
history - thanks to the <strong>KV cache</strong>, which stores the keys/values already computed for every prior token. That is why the <strong>first</strong>
token only appears after the whole prompt has been passed through once (<strong>prefill</strong> - the "time to first token"), while generating word by word afterward (<strong>decode</strong>) is fast, each step
doing the forward pass for just one token. <span class="mono">llama_decode</span> is "run one forward step" and
<span class="mono">llama_sampler_sample</span> is "pick a word by probability"; the two alternate to weave the token-by-token output you see.</p>

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
    <p><strong>Do the math:</strong> a Q4_0 block is exactly <span class="mono">QK4_0 = 32</span> weights, stored as one FP16 scale (2 bytes) + 32
    four-bit codes (16 bytes) = <strong>18 bytes</strong> total; amortized per weight that is <strong>18x8/32 = 4.5 bit</strong>, which is where "about
    1/4 the size" comes from (raw FP16 is 16 bit). A block size of 32 is a compromise: too large and one scale cannot cover the block's value range so
    error grows; too small and every block must store its own scale, so the overhead does not amortize.</p>
    <p><strong>What people actually pick:</strong> the common choice is usually not the plain Q4_0 but a <strong>K-quant</strong> like
    <span class="mono">Q4_K_M</span> - it uses different bit-widths for different layers and quantizes the scales themselves too, for a better
    quality/size ratio; go to Q5_K_M or Q6_K to get closer to the original precision, or down to Q3_K_M to save more memory. In a line:
    <strong>higher bit-width is closer to the original model, lower is more memory-thrifty</strong>, and around Q4 is usually the sweet spot.</p>
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
    <p><strong>Another angle on why in-house:</strong> dragging in a training framework like PyTorch would mean hundreds of MB of dependencies and a
    whole Python runtime - it would never fit into a phone or an embedded device; yet inference really uses only a small handful of ops (matmul,
    softmax, rope, various normalizations...). ggml simply implements that handful, then gives each kind of hardware its own backend - CPU SIMD, CUDA,
    Metal, Vulkan, HIP and so on - so the <strong>same compute graph swaps hardware just by swapping the backend</strong>. That portability is something
    binding yourself to one vendor's closed library can never buy.</p>
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
    <p><strong>Scaling up:</strong> apply the same formula to <strong>70B</strong> and FP16 is about 140 GB, while even Q4 still needs
    <strong>about 40 GB</strong> - more than a single consumer GPU can hold; then you either move to a machine with more memory, or
    <strong>offload</strong> some layers to CPU RAM, or simply split the work across multiple machines/GPUs. Remember too that the
    <strong>KV cache</strong> <strong>grows linearly with context length</strong>: stretch the context very long and its footprint can reach the same
    order of magnitude as the weights themselves, so "how big can I run" is never about the weights alone - leave enough headroom for context.</p>
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

<h2>四块各管一段：为什么要拆开</h2>
<p>目录虽多，<strong>真正的主角只有四块</strong>，每块咬死一件事、互不越界。先认清这四条边界，再回头看那张目录表，剩下的就全是细节了：</p>
<table class="t">
  <tr><th>这一块</th><th>只干一件事</th><th>大致落在</th></tr>
  <tr><td><strong>引擎</strong></td><td>造算力：定义张量、把运算组织成计算图、调度到各后端真正算出来</td><td class="mono">ggml/</td></tr>
  <tr><td><strong>推理库</strong></td><td>把算力拼成"会话"：加载权重、搭图、KV cache、采样、分词、聊天模板</td><td class="mono">src/llama-*</td></tr>
  <tr><td><strong>程序 / 胶水</strong></td><td>给人用：命令行、HTTP 服务、量化器，外加把库粘成程序的通用件</td><td class="mono">tools/ · common/</td></tr>
  <tr><td><strong>模型准备</strong></td><td>把外部模型搬进来：HuggingFace 权重转成 <span class="inline">.gguf</span></td><td class="mono">gguf-py/ · convert_*.py</td></tr>
</table>
<p><strong>为什么非要拆这么开？</strong>因为这四件事的<strong>变化节奏完全不同</strong>：后端算子要追着新硬件、新指令集不断改；推理逻辑要随新模型结构演进；命令行参数与服务接口随用户需求增删；转换脚本则要跟着上游模型格式跑。把它们焊死在一起，改一处就得提心吊胆会不会崩另一处。拆开之后，<strong>各自独立演进、独立测试、独立复用</strong>——给 <span class="mono">ggml</span> 加一个新后端，不必动 <span class="mono">src/llama-*</span> 一行；新增一个模型结构，也碰不到底层算子。这就是"<strong>边界清晰</strong>"最实在的回报。</p>

<h3>为什么公共 API 只留一个头文件</h3>
<p>还有个刻意的设计值得单拎出来：上百个内部文件里，真正<strong>对外公开</strong>的只有 <span class="mono">include/llama.h</span> 一个头。把对外的口子收得这么窄，换来三重好处：其一，<strong>内部随便改</strong>——只要 <span class="mono">llama.h</span> 里的函数签名不变，<span class="mono">src/llama-*</span> 内部怎么重构、换数据结构、调算法，都不会惊动外面的使用者；其二，<strong>对外契约稳定</strong>，使用方升级版本时心里有底，不必追着内部细节东奔西跑；其三，因为暴露的是一套 <span class="mono">C ABI</span>，几乎<strong>所有语言都能绑定</strong>——Python、Go、Rust、Node 等都能透过这个 C 接口调用引擎，社区里大量的语言绑定正是这么搭起来的。把对外收成一个小口，内部才换来放手重构的自由。</p>
<p>顺带提一句：紧挨着 <span class="mono">llama.h</span> 还有个 <span class="mono">llama-cpp.h</span>，它只是给 C++ 用户的一层<strong>薄封装</strong>——用 RAII（智能指针）自动管理 <span class="mono">llama_model</span> / <span class="mono">llama_context</span> 的释放，省去手动 <code>free</code>。它<strong>并不扩大</strong>对外暴露面，只是把同一个 C 接口包得更顺手，所以"对外只有一个契约"这句话依然成立。</p>

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
<p>为什么叫"支线"？因为<strong>一次推理请求只在主干四层里上下穿行</strong>，根本不会跑进这两条线：模型准备在<strong>跑之前</strong>就一次性做完了（产出 <span class="inline">.gguf</span> 便退场），配套支撑则像<strong>脚手架</strong>围在主干周围——<span class="mono">common/</span> 帮程序少写样板，<span class="mono">tests/</span> · <span class="mono">docs/</span> · <span class="mono">cmake/</span> 管测试、文档与构建。把"运行时会经过的"和"运行前 / 运行外的"分清楚，读源码时就不会把脚手架错当成承重墙。</p>

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
<p>把这条流水线落到<strong>真实命令</strong>上，最常见的就是"<strong>转换 -&gt; 量化 -&gt; 运行</strong>"三步。下面这段是最小可跑的骨架——左边 Python 准备、右边 C++ 运行，中间仍旧靠那个 <span class="inline">.gguf</span> 交接：</p>
<pre class="code"><span class="cm"># 模型从准备到运行的完整管道</span>
<span class="cm"># 1) 转换：HuggingFace 模型 -> GGUF（Python 侧，--outfile 指定输出名）</span>
python convert_hf_to_gguf.py ./my-model --outfile my-model.gguf   <span class="cm"># 16 位浮点（默认 auto）</span>
<span class="cm"># 2) 量化（可选，压小）</span>
llama-quantize my-model.gguf my-model-Q4.gguf Q4_0
<span class="cm"># 3) 运行（C++ 侧）</span>
llama-cli -m my-model-Q4.gguf -p <span class="st">"你好"</span></pre>
<p>三步正好落在三个目录：第一步在 <span class="mono">gguf-py/</span> + <span class="mono">convert_*.py</span>（Python）里跑，吐出一个 FP16 的 <span class="inline">.gguf</span>；第二步 <span class="mono">llama-quantize</span>（来自 <span class="mono">tools/</span>）把它压成 <span class="mono">Q4_0</span> 这类低位宽版本，体积骤降到约四分之一；第三步 <span class="mono">llama-cli</span> 加载量化后的文件，真正"跑出字"。两点值得记牢：<strong>量化是可选的</strong>——不在意体积，直接拿第一步的 <span class="inline">.gguf</span> 去跑也行；而第二、三步<strong>全程不碰 Python</strong>，只认那一个文件，这正是前面那句"边界就在 <span class="inline">.gguf</span>"落到命令上的样子。</p>
<p><span class="mono">convert_hf_to_gguf.py</span> 这一步具体在做什么？它读入 HuggingFace 目录里的 <span class="mono">config.json</span>（超参）、分词器与 <span class="mono">safetensors</span>（权重），按模型架构把张量改名、必要时转置，再连同元数据一起<strong>写成一个 <span class="inline">.gguf</span></strong>。换句话说，这条"码头"把外部世界五花八门的模型，统一翻译成引擎只认的那一种格式——之后 C++ 侧就再不必关心它原本长什么样了。</p>
<p>顺带把体积感建立起来：第一步产出的 FP16 文件，7B 模型约 14 GB；第二步 <span class="mono">Q4_0</span> 量化后降到约 4 GB——同一条命令链，跑完就把一个"原本要显卡"的模型压成了"普通内存就能装"的文件。这也正呼应上一课那条"<strong>从重到轻</strong>"的流水线，只是这次落在了真实命令上。</p>

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
<p>如果只想挑<strong>一个</strong>地方开始，首推 <span class="mono">examples/simple</span>：它用两百行左右把"加载模型 -&gt; 分词 -&gt; 解码循环 -&gt; 采样 -&gt; 输出"这条主线完整跑了一遍，没有服务、多模态那些枝节干扰。把它<strong>从头读到尾一遍</strong>，再把每个调用对回上面四层——这个函数属于推理库还是引擎、走的是 <span class="mono">llama.h</span> 里哪个接口——整张地图就从"看过"变成"走过"了。</p>
<p>不管从哪条路进，记住对外只有一个<strong>公共契约</strong> <span class="mono">include/llama.h</span>：搞不清某个能力归谁管时，先回到这个头文件，看它把哪些函数暴露给了外面。先认入口，再逐层往下钻，比漫无目的地翻文件高效得多。</p>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面四个问题，想把这张地图看透的同学点开看；只想记住主干的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> GGUF 文件里到底装了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>示例：</strong>一个 <span class="inline">.gguf</span> 从头到尾大致分四段，按顺序紧挨着排在<strong>同一个文件</strong>里：</p>
    <div class="cellgroup">
      <div class="cg-cap"><b>GGUF 文件结构</b>（单文件，按顺序排布）</div>
      <div class="cells">
        <span class="cell hl">magic + 版本</span><span class="cell">元数据 KV（超参 / 词表 / chat 模板）</span><span class="cell">张量信息（名 / 形状 / 类型 / 偏移）</span><span class="cell q">张量数据（权重块）</span>
      </div>
      <div class="cg-cap" style="margin-top:.5rem">加载时按"张量信息"里的偏移，用 <span class="mono">mmap</span> 映射到对应"张量数据"块，<strong>按需取用、不全量拷贝</strong>。</div>
    </div>
    <p>逐段拆开看：① <strong>文件头</strong>是四字节 magic <span class="mono">"GGUF"</span> 加一个版本号（当前为 3），后面记着"有多少个张量、多少条元数据"；② <strong>元数据 KV</strong> 是一长串键值对——超参如 <span class="mono">&lt;arch&gt;.block_count</span>（层数）、<span class="mono">&lt;arch&gt;.embedding_length</span>（隐藏维度），词表如 <span class="mono">tokenizer.ggml.tokens</span>，会话模板如 <span class="mono">tokenizer.chat_template</span>，都塞在这一段；③ <strong>张量信息</strong>逐个登记每个权重张量的名字、形状、类型，以及它在文件里的<strong>偏移</strong>；④ 最后才是真正的 <strong>张量数据</strong>——一块块权重数值，位置正由上面那些偏移指过去。</p>
    <p><strong>为什么这么设计：</strong>全部塞进<strong>一个文件</strong>，加载时直接 <span class="mono">mmap</span> 进内存、按偏移随用随取，不必先解压或整体拷贝（CPU 推理时近乎零拷贝；用 GPU 后端则权重还会再拷进显存）。超参数与词表都<strong>自带</strong>，引擎读完头部就知道"这是什么模型、该怎么搭计算图"，<strong>免配置文件、免 Python</strong>。把元数据排在权重<strong>前面</strong>也有讲究：引擎先读一小段头部把结构看清楚，再决定怎么映射后面那一大坨权重数据。还有个额外好处：因为是只读映射，<strong>多个进程能共享同一份权重内存</strong>，同机起多个实例时省内存又省加载时间。</p>
    <p><strong>源码：</strong>读写与解析在 <span class="mono">ggml/src/gguf.cpp</span>（<code>gguf_kv</code> / <code>gguf_tensor_info</code> 等结构）；元数据键名的常量集中定义在 <span class="mono">gguf-py/gguf/constants.py</span>；把这些元数据接到 llama 模型上、按 key 取超参的，是 <span class="mono">src/llama-model-loader.cpp</span>。</p>
    <p><strong>替代：</strong>更早的 <strong>GGML / GGJT</strong> 等老格式也干过同样的活，但字段零散、版本兼容差，<strong>已被 GGUF 取代</strong>（仓库里还留着一个 <code>convert_llama_ggml_to_gguf.py</code> 专门把老格式转过来）。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么 ggml 是独立子项目？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>因为<strong>同一个引擎被多个项目复用</strong>——ggml 不只为 llama.cpp 服务，所以它被切成一个能单独存在的子项目。</p>
    <p><strong>例子：</strong>同作者的 <span class="mono">whisper.cpp</span>（语音转文字）等项目也直接拿 ggml 当计算引擎；它们和 llama.cpp 共享同一套张量、算子与后端代码，只是上层逻辑不同。</p>
    <p><strong>怎么保持同步：</strong>ggml 有自己<strong>独立的上游仓库</strong>（<span class="mono">ggml-org/ggml</span>），llama.cpp 里的 <span class="mono">ggml/</span> 其实是它的一份镜像；仓库自带的 <span class="mono">scripts/sync-ggml.sh</span> 就负责把上游最新代码<strong>同步</strong>过来。于是引擎在自己的仓库里演进，各使用方（llama.cpp、whisper.cpp）再各自拉取，谁都不绑死谁。</p>
    <p><strong>源码：</strong><span class="mono">ggml/</span> 自带完整的 <code>include/</code> 与 <code>src/</code>，对外的张量 / 计算图 / 后端接口是独立的一套，不依赖 <span class="mono">src/llama-*</span> 里的任何东西——依赖是严格<strong>单向</strong>的：<span class="mono">src/llama-*</span> 里用 <code>#include "ggml.h"</code> 调引擎，反过来 ggml 从不 include llama 的任何头文件。</p>
    <p><strong>好处：</strong>引擎可以<strong>独立演进</strong>（加新算子、新后端不必动 llama），也<strong>便于嵌入</strong>到任何想要本地张量计算的程序里；llama 只是它众多使用者中的一个。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> common/ 和 src/ 有啥区别？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong><span class="mono">common/</span> 是各个可执行程序<strong>共用的胶水</strong>，<strong>不是</strong>推理库本体；真正的推理逻辑住在 <span class="mono">src/llama-*</span>。</p>
    <p><strong>它管什么：</strong>命令行参数解析、把 C API 的采样接口包成更顺手的封装、日志、下载模型、聊天模板拼接……这些是"把库变成一个能用的程序"要反复写的活，抽到 <span class="mono">common/</span> 让 <span class="mono">tools/</span> 里每个程序都能复用。</p>
    <p><strong>它不管什么：</strong>加载权重、搭计算图、KV cache、真正的采样算法——这些都在 <span class="mono">src/llama-*</span> 里，对外只通过 <span class="mono">include/llama.h</span> 暴露。换句话说，删掉 <span class="mono">common/</span>，<span class="mono">src/llama-*</span> 推理库照样能编译、照样能用，只是你得自己手写一堆样板代码。</p>
    <p><strong>依赖方向：</strong>这条链是严格<strong>单向</strong>的——<span class="mono">tools/</span> 用 <span class="mono">common/</span>，<span class="mono">common/</span> 透过 <span class="mono">include/llama.h</span> 调 <span class="mono">src/llama-*</span>，<span class="mono">src/llama-*</span> 再往下压到 <span class="mono">ggml</span>，即 <span class="mono">tools -&gt; common -&gt; llama.h -&gt; src/llama-* -&gt; ggml</span>，越往右越底层、从不回头反向依赖。认准这个方向，遇到任何一个符号都知道"该去哪一层找"。</p>
    <p><strong>源码：</strong>参数解析看 <span class="mono">common/arg.cpp</span>，其余通用工具看 <span class="mono">common/common.cpp</span>。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">4</span> 新增一个模型支持，大概动哪几处？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>顺着这张地图，给 llama.cpp 加一个新模型结构，通常只在<strong>三处</strong>落笔——一处登记、一处搭图、一处转换。</p>
    <p><strong>① 登记架构：</strong>在 <span class="mono">src/llama-arch</span> 里把新架构注册进来，并声明它用到的各类张量名字（有哪几种权重、各自叫什么）。这一步相当于在引擎的"花名册"上添个新成员。</p>
    <p><strong>② 搭前向图：</strong>在 <span class="mono">src/llama-graph</span> 提供的构件之上，把模型一次前向要做的事拼成计算图——嵌入、各层的注意力与前馈、归一化、输出。复用现成的 <code>build_attn</code> / <code>build_norm</code> 等积木，往往不必从零写算子。</p>
    <p><strong>③ 写转换器：</strong>在 <span class="mono">convert_hf_to_gguf.py</span> 里为这个模型加一段转换逻辑，把 HuggingFace 的权重与超参，按上面登记的张量名写进 <span class="inline">.gguf</span>。</p>
    <p><strong>这里只是预告：</strong>三步各自的细节，后面"<strong>模型加载</strong>""<strong>计算图</strong>"相关的课会专门展开；此刻只需记住——新增模型不是漫天改动，而是沿着<strong>登记 -&gt; 搭图 -&gt; 转换</strong>这条窄路走一遍。</p>
    <p><strong>为什么能这么省事？</strong>正是前面那套分层在兜底：算子、后端、KV cache、采样这些<strong>通用机制</strong>早已写好、且与具体模型无关，新模型只需描述"我的结构长什么样"，把现成积木重新搭一遍即可，无需重造引擎。这就是"边界清晰"在<strong>扩展性</strong>上的回报——加模型是<strong>沿既有接缝填空</strong>，而非动土重建。</p>
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

<h2>Four blocks, each owning one slice - why split them</h2>
<p>The table lists many directories, but <strong>there are really only four protagonists</strong>, each locked onto one job and never crossing into another's. Get these four boundaries straight and the rest of that directory table becomes mere detail:</p>
<table class="t">
  <tr><th>This block</th><th>Does exactly one thing</th><th>Roughly lives in</th></tr>
  <tr><td><strong>Engine</strong></td><td>Provide compute: define tensors, organize ops into a compute graph, schedule to the backends that actually run it</td><td class="mono">ggml/</td></tr>
  <tr><td><strong>Inference lib</strong></td><td>Assemble compute into a "session": load weights, build the graph, KV cache, sampling, tokenizing, chat templates</td><td class="mono">src/llama-*</td></tr>
  <tr><td><strong>Programs / glue</strong></td><td>For people to use: the CLI, the HTTP server, the quantizer, plus the shared bits that turn a library into a program</td><td class="mono">tools/ · common/</td></tr>
  <tr><td><strong>Model prep</strong></td><td>Bring outside models in: turn HuggingFace weights into a <span class="inline">.gguf</span></td><td class="mono">gguf-py/ · convert_*.py</td></tr>
</table>
<p><strong>Why split them so hard?</strong> Because the four jobs <strong>change at completely different rhythms</strong>: backend ops chase new hardware and instruction sets; inference logic evolves with new model architectures; CLI flags and server APIs come and go with user needs; conversion scripts track upstream model formats. Weld them together and touching one means worrying you broke another. Split apart, <strong>each can evolve, be tested, and be reused on its own</strong> - add a new backend to <span class="mono">ggml</span> without touching a line of <span class="mono">src/llama-*</span>; add a new model architecture without reaching down to the low-level ops. That is the most concrete payoff of "<strong>clean boundaries</strong>".</p>

<h3>Why the public API is a single header</h3>
<p>One deliberate design is worth singling out: of the hundreds of internal files, the only thing <strong>exposed to the outside</strong> is the single header <span class="mono">include/llama.h</span>. Keeping the outward opening this narrow buys three things: first, <strong>change the internals freely</strong> - as long as the function signatures in <span class="mono">llama.h</span> stay put, however <span class="mono">src/llama-*</span> refactors internally, swaps data structures, or tweaks algorithms, no outside user is disturbed; second, <strong>a stable external contract</strong>, so consumers upgrade with confidence instead of chasing internal details; third, because what is exposed is a <span class="mono">C ABI</span>, <strong>almost any language can bind to it</strong> - Python, Go, Rust, Node and more all call the engine through this C interface, which is exactly how the many community language bindings are built. Narrow the outward opening to one small neck, and the internals earn the freedom to be refactored at will.</p>
<p>One aside: right next to <span class="mono">llama.h</span> sits <span class="mono">llama-cpp.h</span>, a thin <strong>convenience wrapper</strong> for C++ users - it uses RAII (smart pointers) to free <span class="mono">llama_model</span> / <span class="mono">llama_context</span> automatically, sparing you the manual <code>free</code>. It does <strong>not</strong> widen the exposed surface; it only wraps the same C interface more ergonomically, so "only one external contract" still holds.</p>

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
<p>Why call them "side-paths"? Because <strong>an inference request only travels up and down the four-layer trunk</strong> - it never runs into these two. Model prep is done <strong>once, before you run</strong> (it emits the <span class="inline">.gguf</span> and steps aside), while support sits around the trunk like <strong>scaffolding</strong>: <span class="mono">common/</span> spares programs from boilerplate, and <span class="mono">tests/</span> · <span class="mono">docs/</span> · <span class="mono">cmake/</span> handle testing, docs, and the build. Tell apart "what the runtime passes through" from "what runs before or around it" and you won't mistake the scaffolding for a load-bearing wall when reading the source.</p>

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
<p>Put this pipeline onto <strong>real commands</strong> and the common case is the three steps "<strong>convert -&gt; quantize -&gt; run</strong>". The block below is the minimal runnable skeleton - Python prepares on the left, C++ runs on the right, and the handover is still that <span class="inline">.gguf</span> in the middle:</p>
<pre class="code"><span class="cm"># the full pipeline, from prep to run</span>
<span class="cm"># 1) convert: HuggingFace model -> GGUF (Python; --outfile sets the name)</span>
python convert_hf_to_gguf.py ./my-model --outfile my-model.gguf   <span class="cm"># 16-bit float (auto by default)</span>
<span class="cm"># 2) quantize (optional, to shrink)</span>
llama-quantize my-model.gguf my-model-Q4.gguf Q4_0
<span class="cm"># 3) run (C++ side)</span>
llama-cli -m my-model-Q4.gguf -p <span class="st">"Hello"</span></pre>
<p>The three steps land in three directories: step one runs in <span class="mono">gguf-py/</span> + <span class="mono">convert_*.py</span> (Python) and spits out an FP16 <span class="inline">.gguf</span>; step two <span class="mono">llama-quantize</span> (from <span class="mono">tools/</span>) compresses it into a low-bit version like <span class="mono">Q4_0</span>, shrinking it to about a quarter of the size; step three <span class="mono">llama-cli</span> loads the quantized file and actually "emits text". Two things to remember: <strong>quantization is optional</strong> - if you don't care about size, just run the <span class="inline">.gguf</span> from step one; and steps two and three <strong>never touch Python</strong>, knowing only that one file - which is exactly what "the boundary is the <span class="inline">.gguf</span>" looks like once it lands on real commands.</p>
<p>What does <span class="mono">convert_hf_to_gguf.py</span> actually do in that step? It reads the <span class="mono">config.json</span> (hyper-parameters), the tokenizer, and the <span class="mono">safetensors</span> (weights) from the HuggingFace directory, renames tensors per the model architecture (transposing where needed), and <strong>writes it all out as one <span class="inline">.gguf</span></strong> together with the metadata. In other words, this "loading dock" translates the outside world's motley models into the single format the engine recognizes - after which the C++ side need never care what they originally looked like.</p>
<p>While we are here, build some size intuition: the FP16 file from step one is about 14 GB for a 7B model; after <span class="mono">Q4_0</span> in step two it drops to about 4 GB - the same command chain turns a model that "used to need a GPU" into a file that "fits in ordinary RAM". This echoes the "<strong>heavy to light</strong>" pipeline from the previous lesson, only this time landed on real commands.</p>

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
<p>If you want just <strong>one</strong> place to start, <span class="mono">examples/simple</span> is the top pick: in roughly two hundred lines it runs the whole main line - "load model -&gt; tokenize -&gt; decode loop -&gt; sample -&gt; output" - with no server or multimodal side-branches to distract you. Read it <strong>top to bottom once</strong>, then map each call back onto the four layers above - does this function belong to the inference lib or the engine, which interface in <span class="mono">llama.h</span> does it go through - and the whole map goes from "seen" to "walked".</p>
<p>Whichever path you take, remember there is only one <strong>public contract</strong>, <span class="mono">include/llama.h</span>: when you cannot tell which part owns some capability, go back to this header and see which functions it exposes to the outside. Find the entry first, then drill down layer by layer - far more efficient than flipping through files at random.</p>

<h2>Go deeper (optional)</h2>
<p class="acc-intro">Four questions below - open them if you want to see the whole map clearly; skip them if you just want the main trunk.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What is actually inside a GGUF file? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>Example:</strong> a <span class="inline">.gguf</span> is roughly four sections end to end, packed in order inside <strong>one file</strong>:</p>
    <div class="cellgroup">
      <div class="cg-cap"><b>GGUF file layout</b> (single file, laid out in order)</div>
      <div class="cells">
        <span class="cell hl">magic + version</span><span class="cell">metadata KV (hparams / vocab / chat template)</span><span class="cell">tensor info (name / shape / type / offset)</span><span class="cell q">tensor data (weight blocks)</span>
      </div>
      <div class="cg-cap" style="margin-top:.5rem">On load, the offsets in "tensor info" point <span class="mono">mmap</span> at the matching "tensor data" blocks - <strong>read on demand, no full copy</strong>.</div>
    </div>
    <p>Section by section: (1) the <strong>header</strong> is a four-byte magic <span class="mono">"GGUF"</span> plus a version number (currently 3), followed by "how many tensors, how many metadata entries"; (2) the <strong>metadata KV</strong> is a long list of key-value pairs - hyper-parameters like <span class="mono">&lt;arch&gt;.block_count</span> (layer count) and <span class="mono">&lt;arch&gt;.embedding_length</span> (hidden size), the vocab like <span class="mono">tokenizer.ggml.tokens</span>, the chat template like <span class="mono">tokenizer.chat_template</span>, all sit here; (3) the <strong>tensor info</strong> records each weight tensor's name, shape, type, and its <strong>offset</strong> in the file; (4) only last comes the actual <strong>tensor data</strong> - block after block of weight values, located exactly by those offsets.</p>
    <p><strong>Why this design:</strong> packing everything into <strong>one file</strong> means loading just <span class="mono">mmap</span>s it into memory and reads on demand by offset - no unpacking or whole-file copy first (near-zero-copy for CPU inference; with a GPU backend the weights are then copied into VRAM). The hyper-parameters and vocab are <strong>built in</strong>, so once the engine reads the header it knows "what model this is and how to build the compute graph", with <strong>no config files and no Python</strong>. Putting metadata <strong>before</strong> the weights is deliberate too: the engine reads a small header first to understand the structure, then decides how to map the big blob of weight data that follows. A bonus: because the mapping is read-only, <strong>multiple processes can share the same weight memory</strong>, saving RAM and load time when you run several instances on one machine.</p>
    <p><strong>Source:</strong> reading/parsing lives in <span class="mono">ggml/src/gguf.cpp</span> (the <code>gguf_kv</code> / <code>gguf_tensor_info</code> structs); the metadata key-name constants are defined in <span class="mono">gguf-py/gguf/constants.py</span>; wiring that metadata onto a llama model and fetching hyper-parameters by key is <span class="mono">src/llama-model-loader.cpp</span>.</p>
    <p><strong>Alternatives:</strong> the earlier <strong>GGML / GGJT</strong> formats did the same job, but with scattered fields and poor version compatibility - now <strong>superseded by GGUF</strong> (the repo still keeps a <code>convert_llama_ggml_to_gguf.py</code> just to migrate the old format over).</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is ggml a standalone sub-project? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> because <strong>the same engine is reused by several projects</strong> - ggml does not serve only llama.cpp, so it is carved out as a sub-project that can stand on its own.</p>
    <p><strong>Example:</strong> same-author projects like <span class="mono">whisper.cpp</span> (speech-to-text) use ggml directly as their compute engine; they share the very same tensor, op, and backend code as llama.cpp, only the upper layer differs.</p>
    <p><strong>How they stay in sync:</strong> ggml has its own <strong>standalone upstream repo</strong> (<span class="mono">ggml-org/ggml</span>), and the <span class="mono">ggml/</span> inside llama.cpp is really a mirror of it; the repo's own <span class="mono">scripts/sync-ggml.sh</span> is what <strong>syncs</strong> the latest upstream code over. So the engine evolves in its own repo and each consumer (llama.cpp, whisper.cpp) pulls it in separately - nobody is locked to anybody.</p>
    <p><strong>Source:</strong> <span class="mono">ggml/</span> ships its own complete <code>include/</code> and <code>src/</code>; its tensor / graph / backend interface is a self-contained set that depends on nothing in <span class="mono">src/llama-*</span> - the dependency is strictly <strong>one-way</strong>: <span class="mono">src/llama-*</span> does <code>#include "ggml.h"</code> to use the engine, while ggml never includes any llama header.</p>
    <p><strong>Benefit:</strong> the engine can <strong>evolve independently</strong> (new ops or backends without touching llama) and is <strong>easy to embed</strong> in any program that wants local tensor compute; llama is just one of its many users.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What is the difference between common/ and src/? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> <span class="mono">common/</span> is the <strong>glue shared</strong> by the executable programs - it is <strong>not</strong> the inference library itself; the real inference logic lives in <span class="mono">src/llama-*</span>.</p>
    <p><strong>What it handles:</strong> command-line argument parsing, wrapping the C API's sampler into something handier, logging, downloading models, assembling chat templates... the boilerplate every "turn the library into a usable program" needs, factored into <span class="mono">common/</span> so each program in <span class="mono">tools/</span> can reuse it.</p>
    <p><strong>What it does not:</strong> it does <strong>not</strong> load weights, build the compute graph, manage the KV cache, or implement the actual sampling algorithms - those all live in <span class="mono">src/llama-*</span> and are exposed only through <span class="mono">include/llama.h</span>. In other words, delete <span class="mono">common/</span> and <span class="mono">src/llama-*</span> still compiles and still works; you would just hand-write a pile of boilerplate yourself.</p>
    <p><strong>Dependency direction:</strong> the chain is strictly <strong>one-way</strong> - <span class="mono">tools/</span> use <span class="mono">common/</span>, <span class="mono">common/</span> calls <span class="mono">src/llama-*</span> through <span class="mono">include/llama.h</span>, and <span class="mono">src/llama-*</span> presses down onto <span class="mono">ggml</span>, i.e. <span class="mono">tools -&gt; common -&gt; llama.h -&gt; src/llama-* -&gt; ggml</span>, lower the further right, never doubling back. Fix this direction in your head and any symbol tells you "which layer to look in".</p>
    <p><strong>Source:</strong> for argument parsing see <span class="mono">common/arg.cpp</span>; for the rest of the shared helpers see <span class="mono">common/common.cpp</span>.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">4</span> Adding support for a new model - roughly where do you touch? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> following this map, adding a new model architecture to llama.cpp usually means edits in just <strong>three places</strong> - one to register, one to build the graph, one to convert.</p>
    <p><strong>(1) Register the architecture:</strong> in <span class="mono">src/llama-arch</span>, register the new architecture and declare the tensor names it uses (which weights, and what each is called). This step is like adding a new member to the engine's "roster".</p>
    <p><strong>(2) Build the forward graph:</strong> on top of the building blocks <span class="mono">src/llama-graph</span> provides, assemble what one forward pass does into a compute graph - embeddings, each layer's attention and feed-forward, normalization, output. Reusing ready-made bricks like <code>build_attn</code> / <code>build_norm</code> usually means you don't write ops from scratch.</p>
    <p><strong>(3) Write the converter:</strong> in <span class="mono">convert_hf_to_gguf.py</span>, add a conversion path for the model, writing the HuggingFace weights and hyper-parameters into a <span class="inline">.gguf</span> under the tensor names registered above.</p>
    <p><strong>This is only a preview:</strong> the details of each step are expanded in the later "<strong>model loading</strong>" and "<strong>compute graph</strong>" lessons; for now just remember - adding a model is not a sprawling change but a walk down the narrow path <strong>register -&gt; build graph -&gt; convert</strong>.</p>
    <p><strong>Why so cheap?</strong> Exactly because the layering underneath has your back: the <strong>generic machinery</strong> - ops, backends, KV cache, sampling - is already written and model-agnostic; a new model only describes "what my structure looks like" and re-assembles existing bricks, with no need to rebuild the engine. That is the payoff of "clean boundaries" for <strong>extensibility</strong> - adding a model is <strong>filling in along existing seams</strong>, not breaking ground to rebuild.</p>
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

<p>这 7 步里有几处容易一带而过、其实值得多看一眼。<strong>第 1 步"分词"切出的是 subword（子词），既不是按单字、也不是按整词</strong>：llama.cpp 的分词器走的是 BPE / SPM / WordPiece 这类<strong>子词</strong>算法（见 <span class="mono">include/llama.h</span> 的 <span class="mono">LLAMA_VOCAB_TYPE_SPM / _BPE / _WPM</span>），一个常见英文单词可能正好是一个 token，而生僻词或一个汉字往往被拆成<strong>好几个子词片</strong>。这也正好解释了第 6 步为什么要用 <span class="mono">llama_token_to_piece</span> 把 token 还原成"<strong>词片</strong>"——多个片拼起来才是一个完整的词或汉字，所以你看到的输出是"<strong>一个 token 一个 token 地往外蹦</strong>"，而不是规规矩矩一个字一个字地出。换句话说，<strong>token 并不等于"词"</strong>，它只是模型词表里的一个最小单位。这也带来一个很实际的后果：<strong>token 数和字符数往往对不上</strong>——同一段话，中文、英文、代码切出的 token 数可能差很多，而上下文窗口、计费、速度都是按 <strong>token</strong> 算的，不是按字数算的。</p>
<p><strong>第 2 步"组批"里的 <span class="mono">pos</span> 与 <span class="mono">seq_id</span></strong> 也值得点破。<span class="mono">pos</span> 是每个 token 在序列里的<strong>位置下标</strong>（第 0、1、2…个），模型靠它给注意力补上"谁先谁后"的位置信息；<span class="mono">seq_id</span> 标的则是这个 token<strong>属于哪一条序列</strong>。为什么需要后者？因为一次 batch 里可能<strong>同时塞进多条互不相干的对话</strong>（服务端并发、批量推理正是这么干的），<span class="mono">seq_id</span> 让它们各自的 K/V 互不串台、各算各的。用 <span class="mono">llama_batch_get_one</span> 时这两样会由 <span class="mono">llama_decode</span> 自动补好（位置顺序排、序列固定为 0）；只有要多序列或自定义位置时，才需要 <span class="mono">llama_batch_init</span> 手动填（字段定义见 <span class="mono">include/llama.h</span> 的 <span class="mono">llama_batch</span>）。</p>

<h2>放大第 3 步：一次 decode 内部</h2>
<p>七步里最"重"的是第 3 步 <span class="mono">decode</span>。别把它当成一个黑盒——拆开看，一次前向内部正好是三小步串起来：</p>
<div class="flow">
  <div class="node"><div class="nt">llama_decode</div><div class="nd">一次前向</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">建计算图</div><div class="nd">llama-graph.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">后端执行</div><div class="nd">ggml-backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits（产出）</div><div class="nd">用 llama_get_logits_ith 取出</div></div>
</div>
<p>所以"一次 decode"= <strong>建计算图 + 后端执行 -&gt; logits</strong>：先由 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">llm_graph_*</span>（经 <span class="mono">src/llama-model.cpp</span> 的 <span class="mono">build_graph</span> 拼出）把这一步运算<strong>描述成一张图</strong>，再交 <span class="mono">ggml-backend</span> 调度到硬件上<strong>真正算</strong>，算完用 <span class="mono">llama_get_logits_ith</span> 取出"下一个 token 的分数向量"。它的产出是 <strong>logits</strong>——不是文字，也不是已经选好的 token。</p>
<p>有人会问：既然每步 decode 都要"<strong>先建图</strong>"，那这张图是不是每次都从头搭、很费？其实不必担心——<strong>建图只是搭一张"算子骨架"</strong>（描述谁连谁、张量多大），并不真的搬运权重数据，开销远小于真正的矩阵乘；而且 decode 每步只处理<strong>一个新 token</strong>，这张图本身也很小。所以"建图 + 执行"里<strong>真正的重头始终是后端执行那一段</strong>，建图更像每圈开跑前快速摆好的赛道。</p>
<p><strong>那"前向"内部到底在算什么？</strong>顺着 <span class="mono">src/models/llama.cpp</span> 的建图读，是一条很直的链：先把 token id 经 <span class="mono">build_inp_embd</span> 查成<strong>词向量</strong>（embedding），再让它穿过<strong>多层 transformer block</strong>（每层大致是"自注意力 + 前馈网络"，层数由 <span class="mono">n_layer</span> 决定），末了过一道输出归一化、由<strong>输出层（lm_head）投影回词表大小</strong>，得到每个候选 token 的分数。这里还藏着一个省算力的细节：decode 时其实只需要<strong>最后一个位置</strong>那一行结果——<span class="mono">build_inp_out_ids</span>（<span class="mono">src/llama-graph.cpp</span>）负责只挑出"要输出的位置"，所以第 4 步取 logits，取的正是"<strong>最后一个位置</strong>"对应的那一行分数（这也是 <span class="mono">llama_get_logits_ith(ctx, -1)</span> 里那个 <span class="mono">-1</span> 的含义）。</p>

<h2>prefill vs decode：两种节奏</h2>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这个循环其实分两种节奏：<strong>第一次前向（prefill）</strong>把<strong>整段 prompt 并行地</strong>一次算完
  （很长的 prompt 会切成几个 ubatch 分块、各自一次并行，默认每块 512 token），
  顺手把每个 token 的 K/V 填进 <strong>KV cache</strong>；之后每一步 <strong>decode</strong> 只算<strong>一个新 token</strong>，
  直接复用缓存里过去的 K/V，<strong>不再重算整段历史</strong>——这就是循环能跑得快、能在本地跑得起来的关键。
</div>
<p>把这两种节奏摆到一条时间线上，差别一眼就清楚：prefill 是<strong>一段宽</strong>的并行块，decode 是<strong>一格一格</strong>往后接的小步。</p>
<div class="timeline">
  <div class="lane"><span class="lane-label">Prefill</span><span class="tslot span">整段 prompt（t1…t5）一次并行算，填满 KV cache</span></div>
  <div class="lane"><span class="lane-label">Decode</span><span class="tslot">t6</span><span class="tslot">t7</span><span class="tslot now">t8…</span></div>
</div>
<p><strong>为什么 prefill 能并行、decode 却必须串行？</strong>关键在"谁依赖谁"。prefill 面对的是<strong>已经完整给定</strong>的 prompt，里面每个 token 都是已知的，它们的 K/V <strong>彼此不依赖</strong>，于是可以一口气并行算完、一次把缓存填满。decode 正相反：要算"<strong>下一个词</strong>"，前提是"<strong>上一个词已经写出来了</strong>"——第 n+1 步的输入，恰是第 n 步采样刚选中的那个 token。这种"<strong>后一步依赖前一步的产出</strong>"的链条天然<strong>无法并行</strong>，只能一步一个地往后挪。这也是单条对话生成速度上不去的根因之一：哪怕硬件还很闲，decode 也得乖乖排队、一格一格地走。</p>
<p class="acc-intro">Prefill 把整段提示词<strong>一次并行</strong>算完；之后每步 decode <strong>只算 1 个新 token</strong>，所以"接着往下写"很便宜。</p>
<p><strong>"分块"其实分两层，别混了：</strong>一次 <span class="mono">llama_decode</span> 能接收的 token 数有个<strong>逻辑上限</strong> <span class="mono">n_batch</span>（默认 2048），这是你一次最多能塞进去的量；真正落到硬件上算时，又会按<strong>物理块</strong> <span class="mono">n_ubatch</span>（默认 512）进一步切开、一块块并行跑（两个默认值都在 <span class="mono">common/common.h</span>）。所以遇到很长的 prompt，prefill 既可能被拆成<strong>多次</strong> <span class="mono">decode</span>（受 <span class="mono">n_batch</span> 约束），每次内部又再切成若干 <span class="mono">n_ubatch</span> 小块；但无论怎么切，<strong>同一块之内始终是并行</strong>的，这正是 prefill 远比逐个 decode 快的原因。</p>

<h2>KV cache 为什么让循环不贵</h2>
<p>承上：decode 之所以每步只算一个新 token，靠的就是 <strong>KV cache</strong>。它把每个算过的 token 的 <strong>K/V</strong> 存下来，下一步直接复用，免去重算整段历史：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache</b>：每生成一个 token 就把它的 K/V 追加进缓存，下一步直接复用、不重算历史</div>
  <div class="cells"><span class="lab">prefill 后</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span><span class="cell">K4</span><span class="cell">K5</span></div>
  <div class="cells"><span class="lab">decode t6</span><span class="cell dim">K1…K5（复用）</span><span class="cell hl">+K6</span></div>
  <div class="cells"><span class="lab">decode t7</span><span class="cell dim">K1…K6（复用）</span><span class="cell hl">+K7</span></div>
</div>
<p>每一行只多出<strong>一个高亮新格</strong>（<span class="mono">+K6</span>、<span class="mono">+K7</span>），前面灰掉的部分都是"<strong>复用、不重算</strong>"。没有这层缓存，生成第 n 个 token 就得把前面 n-1 个全重算一遍；有了它，每步的<strong>新增计算</strong>基本是常数——这就是自回归循环能在本地便宜地一直转下去的原因。</p>
<p><strong>那 K/V 到底是什么、又凭什么能缓存？</strong>注意力机制里，每个 token 都会被算出三样东西：Query（拿去"问"的向量）、Key（被查的"标签"）、Value（携带的"内容"）。算"当前 token 该关注谁"时，要拿它的 Query 去和<strong>所有历史 token 的 Key</strong> 逐一比对，再按比对出的权重，把对应的 <strong>Value</strong> 加权汇总起来。关键就在这里：<strong>历史 token 的 K 和 V 一旦算出便不再改变</strong>（它们只取决于那个 token 本身和它的位置），所以完全可以<strong>存下来反复复用</strong>——这正是 KV cache 缓存的东西。代价也很直接：缓存要为<strong>每一层、每个历史 token</strong> 各存一份 K 和 V，占用的内存/显存随<strong>上下文长度线性增长</strong>；上下文开到几万 token 时，KV cache 会吃掉相当可观的一块内存，这也是"上下文窗口"为什么总有上限、长上下文为什么格外吃硬件的原因之一（分配、写入与复用都在 <span class="mono">src/llama-kv-cache.cpp</span>）。</p>

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
  <p style="margin:.5rem 0 0"><strong>"自回归"三个字落到这段代码上</strong>就很具体了：第 7 步把刚生成的 <span class="mono">id</span> 重新包成 batch 喂回去，下一圈的输入就<strong>含了上一圈的输出</strong>——模型一边写、一边把自己写出来的字当作新的上下文继续往下写。也正因为每圈只回灌<strong>一个</strong>新 token、过去的 K/V 又都在缓存里，<strong>每一圈的实际计算量几乎是常数</strong>，循环才能这样一圈圈稳稳地转下去，而不是越写越慢。</p>
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个常见问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 decode 输出的是 logits、不是文字？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong>一次前向只算到"打分"为止。<span class="mono">logits</span> 是<strong>词表上每个 token 的"分数向量"</strong>——词表多大它就多长，每个 token 一个分，谁高谁低而已，还没"拍板"。</p>
    <p><strong>选哪个是另一步：</strong>从这串分数里挑出一个 token，是<strong>采样器</strong>的事（<span class="mono">src/llama-sampler.cpp</span> 的 <span class="mono">llama_sampler_sample</span>，按贪心 / top-k / top-p 等策略选）；把选中的 token 再<strong>还原成文字</strong>，是 <span class="mono">llama_token_to_piece</span>（<span class="mono">src/llama-vocab.cpp</span>）的事。</p>
    <p><strong>为什么这么分：</strong>把"打分 / 选词 / 还原文字"三件事拆开，采样策略就能随意替换而不动前向——同一份 logits，换个采样器就有不同风格的输出。</p>
    <p><strong>采样具体怎么挑：</strong>最简单的<strong>贪心（greedy）</strong>直接选分数最高的那个 token；但实际生成更常用<strong>温度（temperature）</strong>先把这串分数"摊平或拉尖"以调节随机性，再用 <strong>top-k</strong>（只在分数前 k 名里挑）、<strong>top-p</strong>（只在累计概率刚够 p 的那一小撮里挑）把候选范围收窄，最后按概率随机抽一个。所以"<strong>同一份 logits、换个采样器</strong>"才能给出从一板一眼到天马行空的不同风格——这部分后面会有<strong>专门一课</strong>展开，这里只需先记住"logits 负责打分、采样器负责拍板"。</p>
    <p><strong>再补一句"分数"的性质：</strong>logits 是<strong>未归一化的原始打分</strong>，可正可负、加起来也不等于 1，并不是现成的概率。要变成"每个 token 的概率"，还得再过一道 <strong>softmax</strong>（指数化后归一化）；温度其实正作用在这一步之前——把 logits 整体放大或缩小，softmax 出来的分布就更尖或更平。想通这层，就明白"贪心"为什么能跳过 softmax 直接取最大值：只比大小的话，归不归一化都不影响谁最大。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> KV cache 到底省了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>省的是"重算过去"：</strong>没有它，每生成一个新 token 都要把前面所有 token <strong>重算一遍</strong> -&gt; <strong>重算成本</strong>约 <span class="mono">O(n^2)</span>；有了它每步只算<strong>新 token</strong> 的 K/V 并追加进缓存 -&gt; <strong>重算降到</strong> <span class="mono">O(n)</span>。</p>
    <p><strong>给个体感：</strong>假设上下文已经有 1000 个 token，现在要生成第 1001 个。没有缓存，这一步得把前面 1000 个 token 的 K/V <strong>全部重算一遍</strong>；有了缓存，只需算<strong>第 1001 个</strong>这一个 token 的 K/V，再追加到缓存末尾，省下的几乎是整段历史的重复前向。把每一步都这么省下来，一整段生成累计省掉的计算量，就从 <span class="mono">O(n^2)</span> 量级压到 <span class="mono">O(n)</span> 量级——这正是自回归生成能在本地"<strong>一直往下写而不越写越慢</strong>"的根本。</p>
    <p><strong>注意别夸大：</strong>注意力对历史的<strong>扫描</strong>仍是每步 <span class="mono">O(n)</span>（要看过去所有 token），省掉的是<strong>重复计算过去 token 的 K/V</strong>，不是把注意力也变成常数。</p>
    <p><strong>为什么长上下文这么"吃"硬件：</strong>KV cache 的占用大致正比于"<strong>层数 × 上下文长度 × 每层 K/V 的宽度</strong>"——层数和宽度由模型定死，唯一会涨的就是上下文长度。于是把上下文从几千开到几万，KV cache 就要成倍变大，往往成为权重之外最显眼的一块内存。这也解释了为什么本地跑长上下文时，光有"<strong>装得下权重</strong>"的内存还不够，得额外给 KV cache 留足空间；想省，就只能在<strong>更短的上下文</strong>、<strong>更省的缓存量化</strong>或<strong>共享 K/V 的注意力结构</strong>之间做权衡。</p>
    <p><strong>源码：</strong>缓存的分配、写入与复用在 <span class="mono">src/llama-kv-cache.cpp</span>；上下文越长，这块占用越大，也是本地推理要预留内存的地方。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> "计算图"是什么？为什么先建图再算？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>一句话：</strong><span class="mono">ggml</span> 先把这一步运算<strong>描述成一张图</strong>（节点是算子：matmul、rope、softmax……，边是数据流），再交后端按图执行。</p>
    <p><strong>谁来建：</strong>图由 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">llm_graph_*</span> 搭骨架、由 <span class="mono">src/llama-model.cpp</span> 的 <span class="mono">build_graph</span> 按具体模型结构拼出；建好后交 <span class="mono">ggml-backend</span> 调度执行。</p>
    <p><strong>为什么分两步：</strong>把"<strong>描述</strong>运算"与"<strong>执行</strong>运算"分开，同一张图就能落到不同后端（CPU / CUDA / Metal……）上跑，也便于做内存复用、算子融合等优化——这是 ggml 能"一处描述、多端执行"的根。</p>
    <p><strong>这样分到底换来什么：</strong>因为图只是"<strong>描述</strong>"、并不绑定具体硬件，同一套模型结构<strong>不改一行</strong>就能落到 CPU、CUDA、Metal、Vulkan 等不同后端上跑；而且图一旦建好，调度器还能在<strong>真正开算之前</strong>统筹全局——做<strong>内存复用</strong>（算完即可丢弃的中间张量不必各占一块显存）、<strong>算子融合</strong>（把几个小算子并成一个、少几趟读写）、以及把彼此无依赖的分支并行起来等优化。这种"<strong>先把整张图看全、再决定怎么算</strong>"的余地，正是"边算边定"的即时执行模式很难拥有的。</p>
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

<p>A few spots in these 7 steps are easy to skim past but worth a second look. <strong>Step 1 "tokenize" cuts the text into subwords - not whole words, and not single characters</strong>: llama.cpp's tokenizers use subword algorithms like BPE / SPM / WordPiece (see <span class="mono">LLAMA_VOCAB_TYPE_SPM / _BPE / _WPM</span> in <span class="mono">include/llama.h</span>). A common English word may be exactly one token, while a rare word or a single CJK character is often split into <strong>several subword pieces</strong>. That is also why step 6 needs <span class="mono">llama_token_to_piece</span> to turn a token back into a "<strong>piece</strong>" - several pieces glue together into one full word or character, so output comes out "<strong>one token at a time</strong>" rather than one tidy character at a time. In other words, <strong>a token is not a "word"</strong>; it is just the smallest unit in the model's vocabulary. This has a very practical consequence: <strong>token counts and character counts rarely match</strong> - the same passage tokenizes into very different counts for Chinese, English, or code, and the context window, billing and speed are all measured in <strong>tokens</strong>, not characters.</p>
<p><strong>The <span class="mono">pos</span> and <span class="mono">seq_id</span> in step 2 "batch"</strong> are also worth spelling out. <span class="mono">pos</span> is each token's <strong>position index</strong> in the sequence (0, 1, 2...), which the model uses to give attention its "who comes before whom" information; <span class="mono">seq_id</span> marks <strong>which sequence</strong> a token belongs to. Why need the latter? Because one batch may <strong>hold several unrelated conversations at once</strong> (exactly what server-side concurrency and batched inference do), and <span class="mono">seq_id</span> keeps their K/V from crossing wires, each computed on its own. With <span class="mono">llama_batch_get_one</span> these two are auto-filled by <span class="mono">llama_decode</span> (sequential positions, sequence 0); only for multiple sequences or custom positions do you fill them by hand with <span class="mono">llama_batch_init</span> (fields defined in <span class="mono">llama_batch</span> in <span class="mono">include/llama.h</span>).</p>

<h2>Zoom into step 3: inside one decode</h2>
<p>The heaviest of the 7 steps is step 3, <span class="mono">decode</span>. Don't treat it as a black box - opened up, one forward pass is exactly three little steps chained together:</p>
<div class="flow">
  <div class="node"><div class="nt">llama_decode</div><div class="nd">one forward pass</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">build graph</div><div class="nd">llama-graph.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">run on backend</div><div class="nd">ggml-backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits (produced)</div><div class="nd">read via llama_get_logits_ith</div></div>
</div>
<p>So "one decode" = <strong>build graph + run on backend -&gt; logits</strong>: first <span class="mono">llm_graph_*</span> in <span class="mono">src/llama-graph.cpp</span> (assembled by <span class="mono">build_graph</span> in <span class="mono">src/llama-model.cpp</span>) <strong>describes</strong> this step's computation <strong>as a graph</strong>, then <span class="mono">ggml-backend</span> schedules it onto hardware to <strong>actually compute</strong>, and afterwards <span class="mono">llama_get_logits_ith</span> reads out the "score vector for the next token". Its output is <strong>logits</strong> - not text, and not an already-chosen token.</p>
<p>You might ask: if every decode step has to "<strong>build a graph</strong>" first, is rebuilding it from scratch each time expensive? No need to worry - <strong>building the graph only assembles an "operator skeleton"</strong> (describing what connects to what, and tensor sizes); it does not actually move weight data, so its cost is far below the real matrix multiplies. And since each decode step processes only <strong>one new token</strong>, the graph itself is small. So within "build + run", <strong>the real heavy part is always the backend execution</strong>; building the graph is more like quickly laying out the track before each lap.</p>
<p><strong>So what does the "forward pass" actually compute inside?</strong> Reading the graph build in <span class="mono">src/models/llama.cpp</span> it is a very straight chain: first the token id is looked up into a <strong>word vector</strong> (embedding) via <span class="mono">build_inp_embd</span>, then it passes through <strong>several transformer blocks</strong> (each roughly "self-attention + feed-forward network", with the number of layers set by <span class="mono">n_layer</span>), and finally goes through an output norm and is <strong>projected by the output layer (lm_head) back to vocabulary size</strong>, giving a score for each candidate token. There is also a compute-saving detail hidden here: during decode you only need the result at the <strong>last position</strong> - <span class="mono">build_inp_out_ids</span> (<span class="mono">src/llama-graph.cpp</span>) selects just the "positions to output", so step 4 reads exactly the row of scores for the "<strong>last position</strong>" (which is what the <span class="mono">-1</span> in <span class="mono">llama_get_logits_ith(ctx, -1)</span> means).</p>

<h2>prefill vs decode: two rhythms</h2>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The loop actually has two rhythms: the <strong>first forward pass (prefill)</strong> computes the
  <strong>whole prompt in parallel</strong> once (very long prompts are split into a few ubatch chunks,
  each computed in one parallel pass; default 512 tokens per chunk), filling each token's K/V into the <strong>KV cache</strong>;
  after that each <strong>decode</strong> step computes only <strong>one new token</strong>, reusing the past K/V
  from the cache instead of <strong>recomputing the whole history</strong> - the key to why the loop is fast and
  can run locally.
</div>
<p>Put the two rhythms on one timeline and the difference is obvious at a glance: prefill is <strong>one wide</strong> parallel block, decode is <strong>cell-by-cell</strong> steps appended after it.</p>
<div class="timeline">
  <div class="lane"><span class="lane-label">Prefill</span><span class="tslot span">whole prompt (t1…t5) computed in parallel at once, filling the KV cache</span></div>
  <div class="lane"><span class="lane-label">Decode</span><span class="tslot">t6</span><span class="tslot">t7</span><span class="tslot now">t8…</span></div>
</div>
<p><strong>Why can prefill run in parallel while decode must be serial?</strong> It comes down to "what depends on what". Prefill faces an <strong>already fully given</strong> prompt where every token is known; their K/V <strong>do not depend on each other</strong>, so they can all be computed in parallel in one shot and fill the cache at once. Decode is the opposite: to compute the "<strong>next word</strong>", the precondition is that "<strong>the previous word has already been written</strong>" - the input to step n+1 is exactly the token that step n's sampling just chose. This "<strong>each step depends on the previous step's output</strong>" chain is inherently <strong>impossible to parallelize</strong>; it can only inch forward one step at a time. That is one root reason a single conversation's generation speed has a ceiling: even if the hardware is idle, decode still has to queue up and go cell by cell.</p>
<p class="acc-intro">Prefill computes the whole prompt <strong>in parallel</strong> in one pass; afterwards each decode step computes <strong>just 1 new token</strong>, so "keep writing" is cheap.</p>
<p><strong>"Chunking" actually happens at two levels, don't mix them up:</strong> one <span class="mono">llama_decode</span> call has a <strong>logical cap</strong> <span class="mono">n_batch</span> (default 2048) on how many tokens you can submit at once; when it actually runs on hardware, it is further split into <strong>physical chunks</strong> of <span class="mono">n_ubatch</span> (default 512) and run chunk by chunk in parallel (both defaults live in <span class="mono">common/common.h</span>). So for a very long prompt, prefill may be split into <strong>several</strong> <span class="mono">decode</span> calls (bounded by <span class="mono">n_batch</span>), each internally cut into several <span class="mono">n_ubatch</span> chunks; but however it is cut, <strong>within one chunk it is still parallel</strong> - which is exactly why prefill is far faster than decoding one token at a time.</p>

<h2>Why the KV cache keeps the loop cheap</h2>
<p>Following on: the reason decode computes only one new token per step is the <strong>KV cache</strong>. It stores each computed token's <strong>K/V</strong> so the next step reuses them directly, sparing a recompute of the whole history:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache</b>: each generated token appends its K/V to the cache; the next step reuses them directly instead of recomputing history</div>
  <div class="cells"><span class="lab">after prefill</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span><span class="cell">K4</span><span class="cell">K5</span></div>
  <div class="cells"><span class="lab">decode t6</span><span class="cell dim">K1…K5 (reuse)</span><span class="cell hl">+K6</span></div>
  <div class="cells"><span class="lab">decode t7</span><span class="cell dim">K1…K6 (reuse)</span><span class="cell hl">+K7</span></div>
</div>
<p>Each row adds only <strong>one highlighted new cell</strong> (<span class="mono">+K6</span>, <span class="mono">+K7</span>); everything greyed out before it is "<strong>reused, not recomputed</strong>". Without this cache, generating the n-th token would recompute all n-1 before it; with it, the <strong>added work</strong> per step is essentially constant - that is why an autoregressive loop can keep running cheaply on local hardware.</p>
<p><strong>So what exactly are K/V, and why can they be cached?</strong> In the attention mechanism, every token gets three things computed: a Query (the vector it uses to "ask"), a Key (the "label" it is matched against), and a Value (the "content" it carries). To compute "who should the current token attend to", you take its Query and compare it against <strong>the Keys of all past tokens</strong>, then weight-sum the corresponding <strong>Values</strong> by the resulting weights. Here is the key point: <strong>a past token's K and V never change once computed</strong> (they depend only on that token itself and its position), so they can simply be <strong>stored and reused</strong> - which is exactly what the KV cache holds. The cost is direct too: the cache must store one K and one V for <strong>every layer and every past token</strong>, so its memory footprint <strong>grows linearly with context length</strong>; push the context to tens of thousands of tokens and the KV cache eats a sizable chunk of memory - one reason a "context window" always has an upper bound and long context is especially hardware-hungry (allocation, writing and reuse all live in <span class="mono">src/llama-kv-cache.cpp</span>).</p>

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
  <p style="margin:.5rem 0 0"><strong>"Autoregressive" gets concrete on this very code</strong>: step 7 wraps the just-generated <span class="mono">id</span> back into a batch and feeds it in, so the next lap's input <strong>contains the previous lap's output</strong> - the model writes while treating what it just wrote as new context to keep writing. And precisely because each lap feeds back only <strong>one</strong> new token while the past K/V all sit in the cache, <strong>the actual work per lap is nearly constant</strong>, which is why the loop can keep turning steadily round after round instead of slowing down as it goes.</p>
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three common questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does decode output logits, not text? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> a forward pass only goes as far as "scoring". <span class="mono">logits</span> is a <strong>score vector over the whole vocabulary</strong> - as long as the vocab is big, with one score per token; it just says who is higher or lower, nothing is "decided" yet.</p>
    <p><strong>Picking is a separate step:</strong> choosing one token out of those scores is the <strong>sampler</strong>'s job (<span class="mono">llama_sampler_sample</span> in <span class="mono">src/llama-sampler.cpp</span>, by greedy / top-k / top-p...); turning the chosen token <strong>back into text</strong> is <span class="mono">llama_token_to_piece</span>'s job (<span class="mono">src/llama-vocab.cpp</span>).</p>
    <p><strong>Why split it:</strong> separating "score / pick / detokenize" lets the sampling strategy be swapped freely without touching the forward pass - same logits, a different sampler, a different style of output.</p>
    <p><strong>How sampling actually picks:</strong> the simplest, <strong>greedy</strong>, just takes the highest-scoring token; but real generation more often uses <strong>temperature</strong> to first "flatten or sharpen" the scores to tune randomness, then <strong>top-k</strong> (pick only among the top k scores) and <strong>top-p</strong> (pick only within the smallest set whose cumulative probability just reaches p) to narrow the candidates, and finally samples one by probability. That is how "<strong>same logits, a different sampler</strong>" can range from buttoned-up to wildly creative - a <strong>dedicated lesson</strong> will unpack this later; here just remember "logits do the scoring, the sampler makes the call".</p>
    <p><strong>One more note on what "scores" are:</strong> logits are <strong>unnormalized raw scores</strong> - they can be positive or negative and do not sum to 1, so they are not ready-made probabilities. Turning them into "a probability per token" takes a <strong>softmax</strong> (exponentiate then normalize); temperature acts right before this step - scaling the logits up or down makes the softmax distribution sharper or flatter. Once you see this, it is clear why <strong>greedy</strong> can skip softmax and take the max directly: if you only compare magnitudes, normalizing or not does not change which one is largest.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What exactly does the KV cache save? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>It saves "recomputing the past":</strong> without it, every new token re-runs all previous tokens -&gt; the <strong>recompute cost</strong> is ~<span class="mono">O(n^2)</span>; with it each step only computes the <strong>new token</strong>'s K/V and appends it -&gt; <strong>recompute drops to</strong> <span class="mono">O(n)</span>.</p>
    <p><strong>A feel for it:</strong> say the context already has 1000 tokens and you want to generate the 1001st. Without the cache, this step would <strong>recompute the K/V of all 1000</strong> prior tokens; with it, you only compute the K/V of <strong>the 1001st</strong> token and append it to the end - saving nearly a whole history's worth of repeated forward work. Save that at every step and the total work for a full generation drops from the <span class="mono">O(n^2)</span> ballpark to the <span class="mono">O(n)</span> ballpark - the very reason autoregressive generation can "<strong>keep writing without getting slower and slower</strong>" locally.</p>
    <p><strong>Don't overstate it:</strong> attention's <strong>scan</strong> over history is still <span class="mono">O(n)</span> per step (it must look at all past tokens); what is saved is <strong>recomputing past tokens' K/V</strong>, not turning attention itself into a constant.</p>
    <p><strong>Why long context is so hardware-hungry:</strong> the KV cache footprint is roughly proportional to "<strong>layers x context length x the K/V width per layer</strong>" - layers and width are fixed by the model, so the only thing that grows is context length. Push the context from a few thousand to tens of thousands and the KV cache grows in proportion, often the biggest block of memory after the weights themselves. That is why running long context locally needs more than enough memory to "<strong>fit the weights</strong>" - you must reserve extra room for the KV cache; to save it, you can only trade among a <strong>shorter context</strong>, <strong>cheaper cache quantization</strong>, or <strong>attention that shares K/V</strong>.</p>
    <p><strong>Source:</strong> allocation, writing and reuse of the cache live in <span class="mono">src/llama-kv-cache.cpp</span>; the longer the context, the bigger this footprint - and the memory you must reserve for local inference.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What is a "compute graph", and why build then run? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>In one line:</strong> <span class="mono">ggml</span> first <strong>describes</strong> this step's computation <strong>as a graph</strong> (nodes are operators: matmul, rope, softmax...; edges are data flow), then the backend executes the graph.</p>
    <p><strong>Who builds it:</strong> the graph skeleton comes from <span class="mono">llm_graph_*</span> in <span class="mono">src/llama-graph.cpp</span>, assembled per the concrete model structure by <span class="mono">build_graph</span> in <span class="mono">src/llama-model.cpp</span>; once built it is handed to <span class="mono">ggml-backend</span> to schedule and run.</p>
    <p><strong>Why two steps:</strong> separating "<strong>describe</strong>" from "<strong>execute</strong>" lets the same graph run on different backends (CPU / CUDA / Metal...), and makes optimizations like memory reuse and operator fusion possible - the root of ggml's "describe once, run on many backends".</p>
    <p><strong>What this split actually buys:</strong> because the graph is only a "<strong>description</strong>" and not bound to specific hardware, the same model structure runs on different backends (CPU, CUDA, Metal, Vulkan...) <strong>without changing a line</strong>; and once the graph is built, the scheduler can take a global view <strong>before any computation starts</strong> - doing <strong>memory reuse</strong> (intermediate tensors that can be freed right after use need not each hold their own block), <strong>operator fusion</strong> (merging several small ops into one, saving a few read/write passes), and parallelizing branches that have no dependency on each other. This room to "<strong>see the whole graph first, then decide how to compute</strong>" is exactly what eager, compute-as-you-go execution struggles to have.</p>
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
