"""Content for Part 5 (public API & tools)."""

LESSON_25 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前四部分，我们把一个 <span class="mono">.gguf</span> 文件怎么被加载、怎么建图、怎么推理、怎么采样和约束，整条内部机器拆了个遍（L14-L24）。可这些机器，外面的人到底要怎么<strong>驱动</strong>它？答案是 <span class="mono">include/llama.h</span> 里那套 C 函数——它是 llama.cpp 的<strong>总开关</strong>：<span class="mono">llama-cli</span>、<span class="mono">llama-server</span>，以及 Python / Go / Rust / Node 各种语言绑定，全都通过这同一套稳定的 C 接口去加载模型、喂入 token、取出结果。
</p>
<p style="color:var(--muted);margin-top:.4rem">从这一课起我们换个视角：前四部分讲的是"里面怎么转"，从现在开始，我们沿着"<strong>外面怎么用</strong>"这条线，把整个项目重新串一遍。这一课先认识这套 C API 的三块基石——<strong>opaque 句柄</strong>（你只拿到指针，看不到内部）、<strong>典型调用序列</strong>（从初始化到释放的固定套路），以及 C++ 那层 <strong>RAII 包装</strong>（自动帮你释放句柄）。这三块东西看似零散，其实环环相扣：句柄是你操作的对象，调用序列是你操作的顺序，RAII 包装则替你收尾。把它们连起来，你就拿到了读懂任何上层代码的一把钥匙。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  C API 是整个项目对外的<strong>唯一正门</strong>。无论上层是命令行、HTTP 服务，还是别的语言，最终都要落到这几十个 C 函数上。把"对外接口"和"内部实现"彻底分开，正是 llama.cpp 能被到处嵌入的根本前提：接口稳定，内部就能放手重构。换句话说，你在前四部分见到的所有复杂——计算图、KV cache、量化内核——都被收拢在这道门后面；门外的人不必懂这些，只要会调这几十个函数，就能让模型跑起来。这一课就是带你站到门外，看清这道门长什么样、又该怎么推开。
</div>

<h2>句柄与所有权</h2>
<p>这套 C API 不会把模型的内部结构体摊开给你，而是只递给你几个 <strong>opaque 句柄</strong>（不透明句柄）：你拿到的是一个指针，它指向的内容由库内部掌管，你看不到、也不该去碰它的字段。这种"只给指针、藏起实现"的好处后面专门讲，先认认这几个最常打交道的句柄。这么设计不是为难你：把字段藏起来，库才能在不惊动调用方的前提下随意调整内部布局，你也不会因为手一抖改了某个本不该碰的字段而把状态搞乱。你要做的，只是把句柄当成一张"取货凭证"——拿着它去调对应的函数，至于货架后面怎么摆，完全不用操心。</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">只读</span><span class="name">llama_model</span></div><div class="ld">模型权重与元数据；加载后只读，可被多个 context 共享</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">会话</span><span class="name">llama_context</span></div><div class="ld">单个会话的状态：KV cache、计算缓冲、采样位置</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">词表</span><span class="name">llama_vocab</span></div><div class="ld">分词与还原用的词表；经 llama_model_get_vocab 取得</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">采样</span><span class="name">llama_sampler</span></div><div class="ld">采样器与采样链，从 logits 里挑出下一个 token</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">记忆</span><span class="name">llama_memory_t</span></div><div class="ld">上下文记忆（KV cache 等）的句柄；经 llama_get_memory 取得</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">适配器</span><span class="name">llama_adapter_lora</span></div><div class="ld">LoRA 适配器（L24），即插即用地微调权重</div></div>
</div>
<p>这里 <span class="mono">llama_model</span> 装的是<strong>只读的知识</strong>：模型权重一旦加载就不再改变（呼应 L14 的加载、L17 的只读共享），因此它能被多个会话安全地共用同一份——加载一次，到处推理，显存只占一份。这一点在服务端尤其值钱：一台机器上同时来了几十个请求，它们可以共享同一个 <span class="mono">llama_model</span>，各自只开一个轻量的 <span class="mono">llama_context</span>，而不必把几 GB 的权重复制几十份。把"重而不变"的权重和"轻而多变"的会话状态拆开，正是这套 API 省内存、能并发的关键。</p>
<p>而 <span class="mono">llama_context</span> 是<strong>每个会话各自的状态</strong>：它装着 KV cache、计算资源、logits 缓冲（呼应 L17 的上下文、L19 的 KV cache），一个会话开一个，彼此互不干扰。词表 <span class="mono">llama_vocab</span> 经 <span class="mono">llama_model_get_vocab</span> 从模型里取出（L20），采样器 <span class="mono">llama_sampler</span> 则是 L21 那条采样链。值得留意 <span class="mono">llama_vocab</span> 这个句柄：它从属于 model（毕竟分词规则是模型自带的），所以不用单独加载、也不用单独释放，取个指针来用即可。这种"从一个句柄派生出另一个句柄"的关系在 C API 里很常见，理清谁拥有谁，释放时才不会出错。</p>
<div class="cols">
  <div class="col"><h4>llama_model（只读）</h4><p>模型权重与元数据，加载后不再改变。<strong>可被多个 context 共享</strong>：加载一次、多处推理、省内存。</p></div>
  <div class="col"><h4>llama_context（每会话）</h4><p>一个会话的私有状态：KV cache、计算缓冲、采样位置。<strong>一个会话一个</strong>，开销随上下文长度增长。</p></div>
</div>
<p>句柄是你拿到的，自然也得由你负责<strong>释放</strong>。在纯 C 里，这意味着手动调用对应的 <span class="mono">_free</span>：<span class="mono">llama_free</span> 放掉 context、<span class="mono">llama_model_free</span> 放掉 model、<span class="mono">llama_backend_free</span> 收尾全局后端，而且顺序不能乱（先放依赖方、再放被依赖方）：context 是从 model 建出来的、依赖 model，所以必须先释放 context，再释放 model；反过来先放 model，context 就成了悬空指针。漏放会内存泄漏，乱放会崩溃——纯 C 的世界里，这些都得你自己盯着。</p>
<pre class="code"><span class="cm">// C: 你拿到每个句柄, 也得自己释放 (include/llama.h)</span>
llama_model   * model = <span class="fn">llama_model_load_from_file</span>(path, mparams);
llama_context * ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);
<span class="cm">// ... 使用 ...</span>
<span class="fn">llama_free</span>(ctx);              <span class="cm">// 先放 context</span>
<span class="fn">llama_model_free</span>(model);      <span class="cm">// 再放 model</span>

<span class="cm">// C++: include/llama-cpp.h 用 unique_ptr 包住句柄, 出作用域自动释放</span>
<span class="kw">llama_model_ptr</span>   model(<span class="fn">llama_model_load_from_file</span>(path, mparams));
<span class="kw">llama_context_ptr</span> ctx(<span class="fn">llama_init_from_model</span>(model.get(), cparams));
<span class="cm">// ... 使用 ... 作用域结束自动调用 llama_free / llama_model_free</span></pre>
<p>C++ 用户有更省心的选择。<span class="mono">include/llama-cpp.h</span> 这个仅 30 行左右的小头文件，给每个句柄定义了一个 <span class="mono">std::unique_ptr</span> 别名——<span class="mono">llama_model_ptr</span>、<span class="mono">llama_context_ptr</span>、<span class="mono">llama_sampler_ptr</span>、<span class="mono">llama_adapter_lora_ptr</span>，各自带一个会调用匹配 <span class="mono">_free</span> 的删除器。句柄一出作用域就自动释放，再不怕漏掉哪个、也不怕释放顺序写反。把句柄按"先建后毁"的栈顺序声明，析构就会自动按相反顺序进行——刚好满足前面说的"先放 context 再放 model"。等于把容易出错的手动管理交给编译器去保证，这正是 C++ RAII 的拿手好戏。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  <span class="mono">include/llama-cpp.h</span> 全文就四个 <span class="mono">unique_ptr</span> 别名 + 四个 deleter 结构体，每个 deleter 里只有一行：调用对应的 <span class="mono">llama_model_free</span> / <span class="mono">llama_free</span> / <span class="mono">llama_sampler_free</span> / <span class="mono">llama_adapter_lora_free</span>。这就是 C++ 那层"自动释放"的全部秘密——没有魔法，只是把手动 _free 交给了 unique_ptr 的析构。值得学的是这种思路：与其新造一套复杂的资源管理器，不如复用语言已有的机制，用最少的代码把 C 句柄"裹"成 C++ 对象。三十行不到，就让一套手动 API 用起来像现代 C++。
</div>

<h2>典型调用序列</h2>
<p>认识了句柄，再看它们怎么<strong>串起来</strong>用。几乎所有用 llama.cpp 的程序，骨架都是同一条流水线：先初始化后端，加载模型，建上下文，取词表，把文字分词，喂进去解码，拿到 logits，采样出一个 token，转回文字，循环，最后逐个释放。无论是几十行的最小示例，还是 <span class="mono">llama-server</span> 这种成熟服务，骨架都跳不出这条线；区别只在于服务端会把"加载"和"循环"拆到不同线程、再加上缓存与并发调度而已。先把这条主干刻进脑子，再看任何上层代码都不会迷路。</p>
<div class="flow">
  <div class="node"><div class="nt">backend_init</div><div class="nd">全局后端</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">load_model</div><div class="nd">加载权重</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">init_from_model</div><div class="nd">建 context</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_vocab</div><div class="nd">取词表</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tokenize</div><div class="nd">文字-&gt;id</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">decode</div><div class="nd">跑计算图</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_logits</div><div class="nd">取分数</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">sample</div><div class="nd">选 token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">to_piece</div><div class="nd">id-&gt;文字</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">loop</div><div class="nd">循环生成</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">free</div><div class="nd">释放</div></div>
</div>
<p>开头几步是一次性的准备：<span class="mono">llama_backend_init</span> 起全局后端；<span class="mono">llama_model_load_from_file</span> 按路径加载模型（如果模型拆成了多个分片文件，改用 <span class="mono">llama_model_load_from_splits</span>）；<span class="mono">llama_init_from_model</span> 基于模型建出一个 context；<span class="mono">llama_model_get_vocab</span> 取出词表备用。这几步里，参数对象 <span class="mono">mparams</span>、<span class="mono">cparams</span> 决定了很多关键设置：模型参数里有要不要 mmap、放多少层到 GPU（呼应 L07 的 -ngl）；上下文参数里有 n_ctx（上下文窗口多大）、n_batch（一批最多喂多少）等。换句话说，命令行上那些选项，最后都会变成这两个结构体里的字段传进来。</p>
<pre class="code"><span class="cm"># 伪代码: 一次完整的 C-API 生成循环 (简化自 include/llama.h)</span>
<span class="fn">llama_backend_init</span>();                                <span class="cm"># 全局后端初始化</span>
model = <span class="fn">llama_model_load_from_file</span>(path, mparams);   <span class="cm"># 多分片 -&gt; llama_model_load_from_splits</span>
ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);       <span class="cm"># 旧名 llama_new_context_with_model (DEPRECATED)</span>
vocab = <span class="fn">llama_model_get_vocab</span>(model);              <span class="cm"># 只读词表</span>
smpl  = <span class="fn">llama_sampler_chain_init</span>(sparams);          <span class="cm"># 建采样链, 再 add top_k/top_p/temp/dist</span>

n     = <span class="fn">llama_tokenize</span>(vocab, prompt, tokens, ...);  <span class="cm"># 文字 -&gt; token id</span>
batch = <span class="fn">llama_batch_get_one</span>(tokens, n);             <span class="cm"># 最简单的单序列 batch</span>
<span class="kw">while</span> (more) {
    <span class="fn">llama_decode</span>(ctx, batch);                       <span class="cm"># 跑一遍计算图</span>
    logits = <span class="fn">llama_get_logits</span>(ctx);               <span class="cm"># 每个词表 token 一个分数</span>
    id     = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1);   <span class="cm"># 采样链选出下一个 token</span>
    piece  = <span class="fn">llama_token_to_piece</span>(vocab, id, ...);   <span class="cm"># token -&gt; 文字片段</span>
    batch  = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);           <span class="cm"># 把它喂回去, 继续循环</span>
}
<span class="fn">llama_free</span>(ctx); <span class="fn">llama_model_free</span>(model); <span class="fn">llama_backend_free</span>();   <span class="cm"># 逐个释放</span></pre>
<p>然后进入<strong>自回归主循环</strong>：<span class="mono">llama_decode</span> 跑一遍计算图，<span class="mono">llama_get_logits</span> 取出这一步每个 token 的分数，采样链经 <span class="mono">llama_sampler_sample</span> 挑出下一个 token，<span class="mono">llama_token_to_piece</span> 把它转回文字片段，再用 <span class="mono">llama_batch_get_one</span> 喂回去解码下一步。循环结束后，按 context -&gt; model -&gt; backend 的顺序逐个释放。这里能再次看到句柄拆分的意义：循环每转一圈，model 都纹丝不动，真正在变的只是 context 里的 KV 缓存和位置计数器——这也正是"重而不变的权重"和"轻而多变的状态"分家的好处。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  注意 <span class="mono">llama_init_from_model</span> 这个名字。它早期叫 <span class="mono">llama_new_context_with_model</span>，现在被标成 <span class="mono">DEPRECATED</span>、改了名但<strong>语义完全不变</strong>（细节见文末折叠）。读老代码或老教程时，把这两个名字看成同一件事即可。这类改名往往是为了让函数名更准确地表达意图——新名字点明了"从一个模型初始化出上下文"这件事，比旧名字更直白。
</div>
<p>光看流程图还不够具体。我们用一个最小的例子——给模型喂一个 <span class="mono">"Hi"</span>——把整条链真正走一遍，看清每一步手里到底拿着什么：下面这条追踪把抽象的函数名换成了具体的输入输出，顺着六个站走下来，你会发现每一步都只是"拿上一步的产物、调一个函数、得到下一步的产物"，并不神秘。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次 C API 调用</b>：用最小例子 "Hi" 把整条链从加载一路走到出字（id 为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 加载模型</h5>
      <div class="cellrow"><span class="vc">model.gguf</span></div>
      <div class="tlab">load_from_file -&gt; llama_model*</div></div>
    <div class="op">init<br>from_model</div>
    <div class="stn"><h5>② 建 context</h5>
      <div class="cellrow"><span class="vc">llama_context*</span></div>
      <div class="tlab">n_ctx=512</div></div>
    <div class="op">tokenize</div>
    <div class="stn"><h5>③ 分词 "Hi"</h5>
      <div class="cellrow"><span class="vc hot">15043</span></div>
      <div class="tlab">"Hi" -&gt; [15043]</div></div>
    <div class="op">decode</div>
    <div class="stn"><h5>④ 解码取分</h5>
      <div class="cellrow"><span class="vc">logits[n_vocab]</span></div>
      <div class="tlab">每 token 一个分数</div></div>
    <div class="op">sample</div>
    <div class="stn"><h5>⑤ 采样</h5>
      <div class="cellrow"><span class="vc hot">1820</span></div>
      <div class="tlab">采样链选出 token</div></div>
    <div class="op">to_piece</div>
    <div class="stn"><h5>⑥ 转文字</h5>
      <div class="cellrow"><span class="vc blue">" there"</span></div>
      <div class="tlab">1820 -&gt; " there"</div></div>
  </div>
</div>

<h2>为什么是一套 C ABI</h2>
<p>退一步问：为什么对外暴露的偏偏是 <strong>C</strong> 接口，而不是更现代的 C++ 类？答案是 <strong>ABI 稳定性</strong>。C 的函数签名和内存布局，是各语言、各编译器之间最稳妥的"最小公约数"，一旦定下来就很少变；而 C++ 的类布局、名字修饰会随编译器和版本漂移，根本不适合做跨语言的稳定边界。同一个 C++ 类，用 GCC 和 Clang 编出来的二进制接口都可能对不上；而 C 的调用约定几十年如一日地稳定，几乎每种编程语言都内建了"调用 C 函数"的能力，这才让一份引擎能被这么多语言复用。</p>
<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  C 头文件就像一份<strong>"插座标准"</strong>：孔位、电压被定死了，至于墙后面的电网怎么改造、用什么发电，都和你的插头无关。只要插座不变，全世界的电器都能插上就用。家里的电器从不关心电是煤电、风电还是核电，因为插座这层标准早把"怎么用"和"怎么实现"划清了界限。llama.cpp 的 C API 就是这块插座——内部实现天翻地覆，外面的调用方却一行都不用改。
</div>
<p>稳定的 C 接口配上 <strong>opaque 指针</strong>，效果加倍：调用方只看见一个指针，看不见也碰不到背后的 C++ 类、字段布局、模板。于是库作者可以放心重构内部，只要那几十个 C 函数的签名不动，所有调用方就都安然无恙。这也是为什么 llama.cpp 内部能频繁地改算法、换数据结构、加新后端，外面用 Python 绑定的人却几乎从不需要跟着改代码。</p>
<div class="cols">
  <div class="col"><h4>对外：稳定</h4><p>C 函数签名、opaque 句柄、枚举值。这是各语言绑定依赖的<strong>契约</strong>，轻易不动。</p></div>
  <div class="col"><h4>对内：自由</h4><p>指针背后的 C++ 类、数据布局、算法实现。可随时重构、优化，调用方<strong>毫无感知</strong>。</p></div>
</div>
<p>正是这套"稳定的表面 + 自由的内部"，让 llama.cpp 能被嵌进几乎任何地方：Python、Go、Rust、Node 等语言的绑定，全都是对这同一套 C 函数做一层薄封装。一处稳定的 C ABI，撑起了上面整片多语言生态——这也是它能"到处跑"的根本原因。你在手机 App、桌面软件、云端服务里看到的各种"本地大模型"，往下挖到底，调用的多半就是这几个 C 函数；正因为底座足够稳，这份投入才能一年年地复利式回报。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这个"对外稳定、对内自由"的主题，其实贯穿整套教程：L14 的只读权重、L17 的会话状态、L21 的可插拔采样器……llama.cpp 反复在做同一件事——把<strong>不变的契约</strong>和<strong>可变的实现</strong>分开。C API 只是这一思想在"项目边界"上的又一次体现。看懂了这条主线，你再去读任何一个模块，都会下意识地去找：哪部分是对外的承诺、哪部分是可以随时换掉的内脏。
</div>

<h2>深入：批次与 API 演进</h2>
<p>最后用两个折叠，补两块容易绊住新手的细节：分词出来的 token 到底怎么填进批次，以及为什么源码里总冒出 <span class="mono">DEPRECATED</span>。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> llama_batch 是怎么填的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>分词得到的 token 不会直接喂给 <span class="mono">llama_decode</span>，而是先装进一个 <span class="mono">llama_batch</span>（呼应 L18 的批处理）。这个结构告诉 decode：这批有几个 token、它们的 id、各自在序列里的位置 <span class="mono">pos</span>、属于哪个序列 <span class="mono">seq_id</span>、以及哪些 token 需要在算完后输出 <span class="mono">logits</span>。之所以要打包成批，是因为 GPU 一次算一大片比逐个算高效得多；把零散的 token 拼成一个批，正是 L18 讲的"用并行换吞吐"在 API 层的落点。</p>
    <pre class="code"><span class="cm">// llama_decode 吃进去的输入结构 (简化自 include/llama.h)</span>
<span class="kw">struct</span> <span class="fn">llama_batch</span> {
    int32_t        n_tokens;   <span class="cm">// 这批有多少个 token</span>
    llama_token  * token;      <span class="cm">// token id 数组 (L18/L20)</span>
    llama_pos    * pos;        <span class="cm">// 每个 token 在序列里的位置</span>
    llama_seq_id** seq_id;     <span class="cm">// 每个 token 属于哪个序列</span>
    int8_t       * logits;     <span class="cm">// 1=该 token 输出 logits, 0=跳过</span>
};</pre>
    <p>多数简单场景用不着手填这么多字段——<span class="mono">llama_batch_get_one(tokens, n)</span> 会替你把"单序列、从头排位置、只输出最后一个"的常见情形一次填好。只有要并行多序列、或自定义位置时，才需要自己逐字段填。那个 <span class="mono">logits</span> 字段是个<strong>标志数组</strong>：置 1 的 token 才会在 decode 后给出 logits，其余跳过以省算力。这也解释了一个常见疑问：明明喂进去 100 个提示词 token，为什么只在最后一个位置拿 logits？因为前 99 个只是来"填 KV 缓存"的，我们并不需要它们的预测分数，自然就把那些位置的 <span class="mono">logits</span> 标志设成 0。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么源码里总看到 DEPRECATED？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>翻开 <span class="mono">include/llama.h</span>，你会撞见 <span class="mono">llama_new_context_with_model</span> 被包在一个 <span class="mono">DEPRECATED(...)</span> 宏里。它就是今天 <span class="mono">llama_init_from_model</span> 的<strong>旧名字</strong>——语义完全一样，只是换了个更准确的名称。为了不破坏已有代码，旧符号被保留下来、只打上"已弃用"的标记。要分清"弃用"和"删除"：弃用只是编译时给个警告，代码照样能编、能跑；而删除才是真正的断点，老代码会直接编不过。这个缓冲期就是留给大家从容迁移的。</p>
    <p>这种演进在 C API 里很常见（<span class="mono">llama_model_load_from_file</span> 之于更早的 <span class="mono">llama_load_model_from_file</span> 也是一例）。读源码时养成一个习惯：看到 <span class="mono">DEPRECATED(...)</span> 包着的声明，就知道"这是为兼容保留的旧门面，新代码该用它后面 hint 里指向的那个新名字"。这样你既能读懂老教程，又不会在新项目里用错 API。一个稳定的库正是这样小步演进的：既不冻死接口、也不动辄推倒重来，而是用"弃用 - 保留 - 最终移除"这套节奏，让生态有时间跟上。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>C API 在 <span class="mono">include/llama.h</span>、以 <span class="mono">LLAMA_API</span> 导出；对外只给 <strong>opaque 句柄</strong>：<span class="mono">llama_model</span> / <span class="mono">llama_context</span> / <span class="mono">llama_vocab</span> / <span class="mono">llama_sampler</span> / <span class="mono">llama_memory_t</span> / <span class="mono">llama_adapter_lora</span>。</li>
    <li><span class="mono">llama_model</span> = 只读知识（可跨会话共享）；<span class="mono">llama_context</span> = 每会话状态（KV cache、计算资源）。</li>
    <li>典型序列：<span class="mono">backend_init -&gt; load_model -&gt; init_from_model -&gt; get_vocab -&gt; tokenize -&gt; decode -&gt; get_logits -&gt; sample -&gt; token_to_piece -&gt; 循环 -&gt; free</span>。</li>
    <li>释放：C 手动 <span class="mono">llama_free</span> / <span class="mono">llama_model_free</span> / <span class="mono">llama_backend_free</span>；C++ 用 <span class="mono">llama-cpp.h</span> 的 <span class="mono">_ptr</span>（unique_ptr）自动释放。</li>
    <li>选 C ABI 是为<strong>稳定 + 跨语言绑定 + opaque 指针隐藏实现</strong>；<span class="mono">llama_new_context_with_model</span> 已 <span class="mono">DEPRECATED</span>，等同 <span class="mono">llama_init_from_model</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把对外接口定成一套<strong>稳定的 C ABI</strong>、再用 opaque 句柄把实现藏起来，是 llama.cpp 能"一份引擎、处处可用"的根本设计。它和你在 ggml 后端接口（L10）、采样器接口（L21）里见过的是同一种思维：对外承诺一个最小而稳定的契约，对内保留随意重构的自由。读懂这一层，你就明白为什么从命令行到各种语言绑定，最终都汇聚到 <span class="mono">include/llama.h</span> 这几十个函数上——它们才是整个项目真正的<strong>公共入口</strong>。带着这把钥匙往后看：第五部分接下来要讲的命令行工具、HTTP 服务、各类绑定，本质上都是在这套 C 函数之上各搭各的楼，地基却是同一块。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Across the first four parts we took apart the whole internal machine - how a <span class="mono">.gguf</span> file is loaded, how the graph is built, how it infers, samples, and is constrained (L14-L24). But how does the outside world actually <strong>drive</strong> that machine? The answer is the set of C functions in <span class="mono">include/llama.h</span> - llama.cpp's <strong>main switch</strong>: <span class="mono">llama-cli</span>, <span class="mono">llama-server</span>, and the Python / Go / Rust / Node bindings all load the model, feed in tokens, and read out results through this one stable C interface.
</p>
<p style="color:var(--muted);margin-top:.4rem">From this lesson on we switch angle: the first four parts were about "how it turns inside"; from now on we re-thread the whole project along "<strong>how you use it from outside</strong>". This lesson meets the three cornerstones of that C API - <strong>opaque handles</strong> (you only get a pointer, never the internals), the <strong>typical call sequence</strong> (the fixed arc from init to free), and the C++ <strong>RAII wrappers</strong> (which free your handles automatically). These three look scattered but interlock: the handles are what you operate on, the call sequence is the order you operate in, and the RAII wrappers clean up after you. Connect them and you hold a key to reading any higher-level code.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The C API is the project's <strong>one front door</strong> to the outside. Whether the layer above is a command line, an HTTP service, or another language, it all bottoms out in these few dozen C functions. Cleanly splitting the "outward interface" from the "inner implementation" is exactly what lets llama.cpp be embedded everywhere: keep the interface stable and you are free to refactor the inside. In other words, all the complexity you met across the first four parts - the compute graph, the KV cache, the quantization kernels - is gathered behind this one door; the people outside need not understand any of it, they only call these few dozen functions to get the model running. This lesson takes you outside that door, to see what it looks like and how to push it open.
</div>

<h2>Handles and ownership</h2>
<p>This C API does not spread the model's internal structs out for you; it only hands you a few <strong>opaque handles</strong>: what you get is a pointer whose contents the library owns internally - you cannot see, and should not touch, its fields. Why "just a pointer, implementation hidden" is good comes later; first meet the handles you deal with most. This is not to make life hard: hiding the fields lets the library reshuffle its internal layout without disturbing callers, and keeps you from corrupting state by poking a field you were never meant to touch. All you do is treat a handle like a claim ticket - carry it to the matching function, and never mind how the shelves behind the counter are arranged.</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">read-only</span><span class="name">llama_model</span></div><div class="ld">model weights and metadata; read-only after load, shareable across contexts</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">session</span><span class="name">llama_context</span></div><div class="ld">one session's state: KV cache, compute buffers, sampling position</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">vocab</span><span class="name">llama_vocab</span></div><div class="ld">token tables for (de)tokenizing; obtained via llama_model_get_vocab</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">sampler</span><span class="name">llama_sampler</span></div><div class="ld">samplers and the sampler chain, picking the next token from logits</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">memory</span><span class="name">llama_memory_t</span></div><div class="ld">handle to context memory (KV cache etc.); obtained via llama_get_memory</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">adapter</span><span class="name">llama_adapter_lora</span></div><div class="ld">a LoRA adapter (L24), plug-and-play weight tweaks</div></div>
</div>
<p>Here <span class="mono">llama_model</span> holds <strong>read-only knowledge</strong>: model weights never change once loaded (echoing L14's loading and L17's read-only sharing), so many sessions can safely share one copy - load once, infer in many places, only one copy in memory. This pays off especially on the server side: when dozens of requests arrive on one machine, they can share a single <span class="mono">llama_model</span> and each open only a lightweight <span class="mono">llama_context</span>, instead of duplicating the multi-GB weights dozens of times. Splitting the "heavy but constant" weights from the "light but changing" session state is the key to this API saving memory and scaling concurrently.</p>
<p>And <span class="mono">llama_context</span> is <strong>per-session state</strong>: it holds the KV cache, compute resources, and logits buffers (echoing L17's context and L19's KV cache), one per session, each isolated from the rest. The vocab <span class="mono">llama_vocab</span> is pulled from the model via <span class="mono">llama_model_get_vocab</span> (L20), and the sampler <span class="mono">llama_sampler</span> is L21's sampler chain. Note the <span class="mono">llama_vocab</span> handle: it belongs to the model (the tokenizer rules ship with the model, after all), so you neither load nor free it separately - you just take a pointer and use it. This "one handle derived from another" relationship is common in the C API, and being clear on who owns whom is what keeps your frees correct.</p>
<div class="cols">
  <div class="col"><h4>llama_model (read-only)</h4><p>Model weights and metadata, unchanged after load. <strong>Shareable across contexts</strong>: load once, infer in many places, save memory.</p></div>
  <div class="col"><h4>llama_context (per session)</h4><p>One session's private state: KV cache, compute buffers, sampling position. <strong>One per session</strong>, cost grows with context length.</p></div>
</div>
<p>The handles are yours, so freeing them is your job too. In plain C this means calling the matching <span class="mono">_free</span> by hand: <span class="mono">llama_free</span> releases the context, <span class="mono">llama_model_free</span> the model, <span class="mono">llama_backend_free</span> tears down the global backend - and the order matters (free the dependent first, then what it depended on): a context is built from a model and depends on it, so you must free the context first and the model second; do it the other way and the context becomes a dangling pointer. Miss a free and you leak memory, get the order wrong and you crash - in plain C, all of this is on you to watch.</p>
<pre class="code"><span class="cm">// C: you own every handle, and must free it yourself (include/llama.h)</span>
llama_model   * model = <span class="fn">llama_model_load_from_file</span>(path, mparams);
llama_context * ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);
<span class="cm">// ... use ...</span>
<span class="fn">llama_free</span>(ctx);              <span class="cm">// free the context first</span>
<span class="fn">llama_model_free</span>(model);      <span class="cm">// then the model</span>

<span class="cm">// C++: include/llama-cpp.h wraps a handle in unique_ptr, freed at scope exit</span>
<span class="kw">llama_model_ptr</span>   model(<span class="fn">llama_model_load_from_file</span>(path, mparams));
<span class="kw">llama_context_ptr</span> ctx(<span class="fn">llama_init_from_model</span>(model.get(), cparams));
<span class="cm">// ... use ... scope end auto-calls llama_free / llama_model_free</span></pre>
<p>C++ users get a tidier option. <span class="mono">include/llama-cpp.h</span>, a tiny header of about 30 lines, defines a <span class="mono">std::unique_ptr</span> alias for each handle - <span class="mono">llama_model_ptr</span>, <span class="mono">llama_context_ptr</span>, <span class="mono">llama_sampler_ptr</span>, <span class="mono">llama_adapter_lora_ptr</span> - each with a deleter that calls the matching <span class="mono">_free</span>. A handle is released the moment it leaves scope, so you never miss one or get the free order wrong. Declare the handles in stack order, built before destroyed, and destruction runs in reverse order automatically - exactly the "free the context, then the model" rule from earlier. You hand the error-prone manual bookkeeping to the compiler to enforce, which is precisely what C++ RAII is good at.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  All of <span class="mono">include/llama-cpp.h</span> is four <span class="mono">unique_ptr</span> aliases plus four deleter structs, each deleter a single line: call the matching <span class="mono">llama_model_free</span> / <span class="mono">llama_free</span> / <span class="mono">llama_sampler_free</span> / <span class="mono">llama_adapter_lora_free</span>. That is the whole secret of the C++ "auto release" - no magic, just handing the manual _free to unique_ptr's destructor. The lesson worth taking is the approach: rather than build a complex new resource manager, reuse the language's existing mechanism and "wrap" the C handle into a C++ object with the least code. In under thirty lines, a manual API starts to feel like modern C++.
</div>

<h2>The typical call sequence</h2>
<p>With the handles known, see how they <strong>string together</strong>. Almost every program using llama.cpp shares the same skeleton pipeline: init the backend, load the model, build a context, get the vocab, tokenize the text, feed it in to decode, take the logits, sample one token, turn it back into text, loop, and finally free things one by one. Whether it is a tiny example of a few dozen lines or a mature service like <span class="mono">llama-server</span>, the skeleton never escapes this line; the only difference is that the server splits "load" and "loop" across threads and adds caching and concurrent scheduling. Burn this trunk into your head and you will not get lost in any higher-level code.</p>
<div class="flow">
  <div class="node"><div class="nt">backend_init</div><div class="nd">global backend</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">load_model</div><div class="nd">load weights</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">init_from_model</div><div class="nd">build context</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_vocab</div><div class="nd">get vocab</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tokenize</div><div class="nd">text-&gt;id</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">decode</div><div class="nd">run graph</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_logits</div><div class="nd">get scores</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">sample</div><div class="nd">pick token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">to_piece</div><div class="nd">id-&gt;text</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">loop</div><div class="nd">keep generating</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">free</div><div class="nd">release</div></div>
</div>
<p>The first few steps are one-time setup: <span class="mono">llama_backend_init</span> starts the global backend; <span class="mono">llama_model_load_from_file</span> loads the model by path (if the model is split into shard files, use <span class="mono">llama_model_load_from_splits</span> instead); <span class="mono">llama_init_from_model</span> builds a context from the model; <span class="mono">llama_model_get_vocab</span> pulls out the vocab for later. Across these steps the parameter objects <span class="mono">mparams</span> and <span class="mono">cparams</span> decide a lot: the model params carry whether to mmap and how many layers to offload to the GPU (echoing L07's -ngl); the context params carry n_ctx (how large the context window is), n_batch (how many tokens at most per batch), and more. In other words, those command-line options all end up as fields in these two structs passed in here.</p>
<pre class="code"><span class="cm"># pseudocode: one full C-API generation loop (simplified from include/llama.h)</span>
<span class="fn">llama_backend_init</span>();                                <span class="cm"># global backend init</span>
model = <span class="fn">llama_model_load_from_file</span>(path, mparams);   <span class="cm"># multi-shard -&gt; llama_model_load_from_splits</span>
ctx   = <span class="fn">llama_init_from_model</span>(model, cparams);       <span class="cm"># old name llama_new_context_with_model (DEPRECATED)</span>
vocab = <span class="fn">llama_model_get_vocab</span>(model);              <span class="cm"># read-only vocab</span>
smpl  = <span class="fn">llama_sampler_chain_init</span>(sparams);          <span class="cm"># build a chain, then add top_k/top_p/temp/dist</span>

n     = <span class="fn">llama_tokenize</span>(vocab, prompt, tokens, ...);  <span class="cm"># text -&gt; token ids</span>
batch = <span class="fn">llama_batch_get_one</span>(tokens, n);             <span class="cm"># simplest single-sequence batch</span>
<span class="kw">while</span> (more) {
    <span class="fn">llama_decode</span>(ctx, batch);                       <span class="cm"># run the compute graph</span>
    logits = <span class="fn">llama_get_logits</span>(ctx);               <span class="cm"># one score per vocab token</span>
    id     = <span class="fn">llama_sampler_sample</span>(smpl, ctx, -1);   <span class="cm"># the chain picks the next token</span>
    piece  = <span class="fn">llama_token_to_piece</span>(vocab, id, ...);   <span class="cm"># token -&gt; text piece</span>
    batch  = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);           <span class="cm"># feed it back, keep looping</span>
}
<span class="fn">llama_free</span>(ctx); <span class="fn">llama_model_free</span>(model); <span class="fn">llama_backend_free</span>();   <span class="cm"># free one by one</span></pre>
<p>Then comes the <strong>autoregressive main loop</strong>: <span class="mono">llama_decode</span> runs the graph once, <span class="mono">llama_get_logits</span> reads out a score per token for this step, the chain via <span class="mono">llama_sampler_sample</span> picks the next token, <span class="mono">llama_token_to_piece</span> turns it back into a text piece, and <span class="mono">llama_batch_get_one</span> feeds it back to decode the next step. When the loop ends, free in the order context -&gt; model -&gt; backend. Here you see the point of splitting handles again: on every turn of the loop the model never moves, and the only things that change are the KV cache and the position counter inside the context - exactly the payoff of separating the "heavy, constant weights" from the "light, changing state".</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Note the name <span class="mono">llama_init_from_model</span>. It used to be <span class="mono">llama_new_context_with_model</span>, now marked <span class="mono">DEPRECATED</span> - renamed, but with <strong>identical semantics</strong> (details in the fold below). Reading old code or tutorials, just treat the two names as the same thing. Such renames are usually about making the function name state intent more accurately - the new name spells out "initialize a context from a model", which is plainer than the old one.
</div>
<p>A flow chart is still not concrete enough. Let us walk one minimal example - feeding the model a single <span class="mono">"Hi"</span> - through the whole chain, to see exactly what we hold at each step: the trace below swaps abstract function names for concrete inputs and outputs, and walking the six stations you find every step is just "take the previous step's product, call one function, get the next step's product" - nothing mysterious.</p>
<div class="trace">
  <div class="tcap"><b>Tracing one C-API call</b>: a minimal "Hi" walks the whole chain from load to output (ids are illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) load model</h5>
      <div class="cellrow"><span class="vc">model.gguf</span></div>
      <div class="tlab">load_from_file -&gt; llama_model*</div></div>
    <div class="op">init<br>from_model</div>
    <div class="stn"><h5>(2) build context</h5>
      <div class="cellrow"><span class="vc">llama_context*</span></div>
      <div class="tlab">n_ctx=512</div></div>
    <div class="op">tokenize</div>
    <div class="stn"><h5>(3) tokenize "Hi"</h5>
      <div class="cellrow"><span class="vc hot">15043</span></div>
      <div class="tlab">"Hi" -&gt; [15043]</div></div>
    <div class="op">decode</div>
    <div class="stn"><h5>(4) decode -&gt; logits</h5>
      <div class="cellrow"><span class="vc">logits[n_vocab]</span></div>
      <div class="tlab">one score per token</div></div>
    <div class="op">sample</div>
    <div class="stn"><h5>(5) sample</h5>
      <div class="cellrow"><span class="vc hot">1820</span></div>
      <div class="tlab">the chain picks a token</div></div>
    <div class="op">to_piece</div>
    <div class="stn"><h5>(6) token_to_piece</h5>
      <div class="cellrow"><span class="vc blue">" there"</span></div>
      <div class="tlab">1820 -&gt; " there"</div></div>
  </div>
</div>

<h2>Why a C ABI</h2>
<p>Step back and ask: why expose a <strong>C</strong> interface rather than more modern C++ classes? The answer is <strong>ABI stability</strong>. C's function signatures and memory layout are the safest "lowest common denominator" across languages and compilers, and once fixed they rarely change; C++ class layouts and name mangling drift with compiler and version, unfit for a stable cross-language boundary. The same C++ class can produce binary interfaces that do not line up between GCC and Clang; C's calling convention, by contrast, has stayed stable for decades, and nearly every programming language has a built-in ability to "call a C function" - which is what lets one engine be reused from so many languages.</p>
<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A C header is like a <strong>"power-socket standard"</strong>: the hole pattern and voltage are fixed, while how the grid behind the wall is rebuilt or what generates the power has nothing to do with your plug. As long as the socket stays the same, appliances everywhere just plug in. The appliances at home never care whether the electricity comes from coal, wind, or nuclear, because the socket standard already drew a clean line between "how to use" and "how it is implemented". llama.cpp's C API is that socket - the internals can be overhauled wholesale while callers outside change not a single line.
</div>
<p>A stable C interface plus <strong>opaque pointers</strong> doubles the effect: the caller sees only a pointer, never the C++ classes, field layouts, or templates behind it. So the library authors can refactor the inside freely, and as long as those few dozen C function signatures hold, every caller stays unaffected. This is also why llama.cpp can frequently change algorithms, swap data structures, and add new backends internally, while people using the Python bindings outside almost never have to change their code along with it.</p>
<div class="cols">
  <div class="col"><h4>Outward: stable</h4><p>C function signatures, opaque handles, enum values. This is the <strong>contract</strong> every language binding depends on, rarely changed.</p></div>
  <div class="col"><h4>Inward: free</h4><p>The C++ classes, data layouts, and algorithms behind the pointer. Refactor and optimize anytime, callers <strong>none the wiser</strong>.</p></div>
</div>
<p>It is exactly this "stable surface plus free interior" that lets llama.cpp be embedded almost anywhere: the Python, Go, Rust, and Node bindings are all a thin wrapper over this same set of C functions. One stable C ABI holds up the whole multi-language ecosystem above it - the root reason it "runs everywhere". The various "local LLMs" you see in phone apps, desktop software, and cloud services, dug all the way down, mostly call these same few C functions; because the base is stable enough, this investment pays compounding returns year after year.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This "stable outward, free inward" theme actually runs through the whole guide: L14's read-only weights, L17's session state, L21's pluggable samplers... llama.cpp keeps doing the same thing - separating the <strong>unchanging contract</strong> from the <strong>changeable implementation</strong>. The C API is just one more expression of that idea, this time at the project's boundary. Once you see this through-line, reading any module you will instinctively look for which part is the outward promise and which part is the swappable guts.
</div>

<h2>Deep dive: the batch and API evolution</h2>
<p>Finally, two folds to fill in two details that often trip up newcomers: how tokenized tokens actually get filled into a batch, and why <span class="mono">DEPRECATED</span> keeps showing up in the source.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> How is llama_batch filled? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Tokens from tokenizing are not fed straight to <span class="mono">llama_decode</span>; they first go into a <span class="mono">llama_batch</span> (echoing L18's batching). This struct tells decode: how many tokens this batch carries, their ids, each token's position <span class="mono">pos</span> in the sequence, which sequence <span class="mono">seq_id</span> it belongs to, and which tokens should output <span class="mono">logits</span> after compute. The reason to pack into a batch is that a GPU computing one big slab at once is far more efficient than one token at a time; stitching scattered tokens into a batch is exactly where L18's "trade parallelism for throughput" lands at the API layer.</p>
    <pre class="code"><span class="cm">// the input struct llama_decode consumes (simplified from include/llama.h)</span>
<span class="kw">struct</span> <span class="fn">llama_batch</span> {
    int32_t        n_tokens;   <span class="cm">// how many tokens this batch carries</span>
    llama_token  * token;      <span class="cm">// the token ids (L18/L20)</span>
    llama_pos    * pos;        <span class="cm">// position of each token in its sequence</span>
    llama_seq_id** seq_id;     <span class="cm">// which sequence each token belongs to</span>
    int8_t       * logits;     <span class="cm">// 1=output logits for this token, 0=skip</span>
};</pre>
    <p>Most simple cases need not fill all these fields by hand - <span class="mono">llama_batch_get_one(tokens, n)</span> fills the common "single sequence, positions from the start, output only the last" case for you in one go. Only parallel multi-sequence work or custom positions need per-field filling. That <span class="mono">logits</span> field is a <strong>flag array</strong>: only tokens set to 1 yield logits after decode, the rest are skipped to save compute. This also answers a common question: when you feed in 100 prompt tokens, why take logits only at the last position? Because the first 99 are only there to "fill the KV cache" - we do not need their prediction scores, so we set the <span class="mono">logits</span> flag to 0 at those positions.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Why does DEPRECATED keep showing up? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Open <span class="mono">include/llama.h</span> and you will run into <span class="mono">llama_new_context_with_model</span> wrapped in a <span class="mono">DEPRECATED(...)</span> macro. It is simply the <strong>old name</strong> of today's <span class="mono">llama_init_from_model</span> - identical semantics, just a more accurate name. To avoid breaking existing code, the old symbol is kept and merely tagged "deprecated". Tell "deprecated" apart from "removed": deprecation only emits a compile-time warning while the code still compiles and runs; removal is the real cutoff, where old code simply fails to build. That grace period is exactly what lets everyone migrate at a comfortable pace.</p>
    <p>Such evolution is common in the C API (<span class="mono">llama_model_load_from_file</span> versus the earlier <span class="mono">llama_load_model_from_file</span> is another case). Build a habit when reading source: a declaration wrapped in <span class="mono">DEPRECATED(...)</span> means "this is an old facade kept for compatibility, new code should use the new name its hint points to". That way you can both read old tutorials and avoid using the wrong API in new projects. A stable library evolves in exactly these small steps: neither freezing the interface forever nor tearing it down on a whim, but using the "deprecate - keep - eventually remove" rhythm to give the ecosystem time to catch up.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>The C API lives in <span class="mono">include/llama.h</span>, exported via <span class="mono">LLAMA_API</span>; it hands out only <strong>opaque handles</strong>: <span class="mono">llama_model</span> / <span class="mono">llama_context</span> / <span class="mono">llama_vocab</span> / <span class="mono">llama_sampler</span> / <span class="mono">llama_memory_t</span> / <span class="mono">llama_adapter_lora</span>.</li>
    <li><span class="mono">llama_model</span> = read-only knowledge (shareable across sessions); <span class="mono">llama_context</span> = per-session state (KV cache, compute resources).</li>
    <li>Typical sequence: <span class="mono">backend_init -&gt; load_model -&gt; init_from_model -&gt; get_vocab -&gt; tokenize -&gt; decode -&gt; get_logits -&gt; sample -&gt; token_to_piece -&gt; loop -&gt; free</span>.</li>
    <li>Freeing: C calls <span class="mono">llama_free</span> / <span class="mono">llama_model_free</span> / <span class="mono">llama_backend_free</span> by hand; C++ uses <span class="mono">llama-cpp.h</span>'s <span class="mono">_ptr</span> (unique_ptr) for automatic release.</li>
    <li>A C ABI buys <strong>stability + cross-language bindings + opaque pointers hiding the implementation</strong>; <span class="mono">llama_new_context_with_model</span> is <span class="mono">DEPRECATED</span>, same as <span class="mono">llama_init_from_model</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Pinning the outward interface to a <strong>stable C ABI</strong> and hiding the implementation behind opaque handles is the core design that lets llama.cpp be "one engine, usable everywhere". It is the same thinking you saw in the ggml backend interface (L10) and the sampler interface (L21): promise a minimal, stable contract outward while keeping full freedom to refactor inward. Grasp this layer and you see why everything - from the command line to every language binding - converges onto those few dozen functions in <span class="mono">include/llama.h</span>; they are the project's real <strong>public entry point</strong>. Carry this key forward: the command-line tools, HTTP service, and assorted bindings that Part 5 covers next are each their own building raised on top of this same set of C functions, but the foundation is one and the same.
</div>
""",
}
