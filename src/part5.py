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

LESSON_26 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课（L25）我们认识了 <span class="mono">include/llama.h</span> 那套 C API——它确实能驱动整台机器，可真用起来挺"啰嗦"：填一个 <span class="mono">llama_batch</span> 要逐字段设 token/pos/seq_id；搭一条采样链要一节节 <span class="mono">llama_sampler_chain_add</span>；把命令行解析成参数、再把模型从网上拉下来，每个工具都得从头写一遍。这些反复出现的杂活，正是 <span class="mono">common/</span> 这一层要替你包掉的。
</p>
<p style="color:var(--muted);margin-top:.4rem"><span class="mono">common/</span> 是 llama.cpp 自带的"公共胶水层"：它把上面那些零碎的 C 调用，封装成一组顺手的 C++ helper，让 <span class="mono">llama-cli</span>、<span class="mono">llama-server</span> 这些自带工具都站在它的肩膀上，不必各写各的样板。但有件事要先说清楚：<strong>common 不是公共 API</strong>——它没有 <span class="mono">llama.h</span> 那样的 ABI 稳定承诺，只是项目"内部"给自家工具用的便利库；第三方语言绑定该直接对着 <span class="mono">llama.h</span> 写，而不是依赖 common。</p>
<p style="color:var(--muted)">这一层到底替你包了哪些杂活？大致可分四类：把命令行解析成一个大配置，按配置把模型与上下文一次性初始化好，把采样链和语法约束包成一个顺手的采样器，以及在你只给出一个仓库名时自动下载并缓存模型文件。本课就顺着这四件事走一遍，最后再花点篇幅讲清"为什么 common 不算公共 API"，以及两个天天在用、却很少被单独提起的小帮手——日志与终端着色。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  common 夹在 <span class="mono">llama.h</span>（稳定地基）和各个 tool（楼上的房间）之间，是一层"只对内、不对外"的脚手架。它的全部价值就两个字：<strong>复用</strong>。把每个工具都要做的重复动作——解析参数、配采样器、下载模型、打日志——统统收拢到一处，工具的 <span class="mono">main()</span> 因此能瘦到几乎只剩业务逻辑。读懂 common，你就读懂了 cli/server 这些工具"短小"的秘密：不是它们做得少，而是杂活早被 common 提前包圆了。换个角度看，common 的存在让"地基"得以保持精简：凡是为了顺手、却不值得写进稳定 C API 的东西，都可以安心搁在 common 这层，将来要改也不会惊动外部用户。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把这套分层想成一家连锁餐饮：<span class="mono">llama.h</span> 是<strong>水电管网</strong>（稳定、对所有人一致），common 是<strong>中央厨房</strong>（把洗菜、切配、调酱这些每家店都要做的预处理一次做好），各个 tool 则是<strong>门店</strong>（只管按订单出菜）。门店不必各自洗菜，中央厨房也从不直接面对顾客——它只服务自家门店。这正是 common"对内不对外"的含义：它是给自家工具用的后厨，不是开给外人的接口。厨房改了刀工、换了酱料配方，受影响的只有自家门店，不会波及马路上的顾客；可水电管网若要改规格，全城接它的人都得跟着动——这正是"对内可随时调整、对外必须稳定"的分别。
</div>

<h2>一站式配置：common_params</h2>
<p>common 的核心，是一个<strong>大配置结构体</strong> <span class="mono">common_params</span>（见 <span class="mono">common/common.h</span>）。模型路径、提示词、要生成多少 token、上下文开多大，还有一整块嵌套的采样设置，几乎所有"旋钮"都塞进这一个结构体里。它还自带一套合理的默认值，于是你只需覆盖真正关心的那几项，其余的留空也能跑起来。这样一来，工具不再到处传一大把零散参数，而是只围着这一个 <span class="mono">common_params</span> 转：解析命令行时按引用（<span class="mono">common_params &amp;</span>）往里填，初始化时再把它整个取出来用——同一个对象从头贯穿到尾，省去层层透传的麻烦。</p>
<pre class="code"><span class="cm">// 一个大结构体装下所有旋钮 (简化自 common/common.h)</span>
<span class="kw">struct</span> <span class="fn">common_params</span> {
    std::string  prompt;              <span class="cm">// -p  提示词</span>
    int32_t      n_predict;           <span class="cm">// -n  最多生成多少 token</span>
    int32_t      n_ctx;               <span class="cm">// -c  上下文窗口大小</span>
    common_params_sampling sampling;  <span class="cm">// 嵌套: samplers / temp / grammar</span>
    common_params_model    model;     <span class="cm">// model.path / model.hf_repo ...</span>
    <span class="cm">// ... 还有几十个字段</span>
};</pre>
<p>留意那两个<strong>嵌套子结构</strong>：<span class="mono">sampling</span>（即 <span class="mono">common_params_sampling</span>，装着采样链顺序、温度、语法等）和 <span class="mono">model</span>（即 <span class="mono">common_params_model</span>，装着本地路径 <span class="mono">model.path</span> 与 HF 仓库 <span class="mono">model.hf_repo</span>）。把相关的旋钮收进各自的小结构体，既让 <span class="mono">common_params</span> 不至于变成一锅乱炖，也方便采样、下载这些模块各取所需。</p>
<p>有了配置，怎么把它变成跑得起来的对象？分两步。先调一次 <span class="mono">common_init()</span> 做<strong>全局初始化</strong>（起日志系统、打印 build 信息）——这步和具体模型无关，整个程序只做一次；真正干活的是 <span class="mono">common_init_from_params(params)</span>，它接过填好的 <span class="mono">common_params</span>，一次性产出一个 <span class="mono">common_init_result</span>，里面装着加载好的 model、建好的 context、配好的 sampler（分别经 <span class="mono">.model()</span> / <span class="mono">.context()</span> / <span class="mono">.sampler()</span> 取用）。之所以拆成两步，是因为全局初始化必须先于任何日志输出发生，否则早期的报错可能就被悄悄吞掉。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>common_params（填好的配置）</h4>
    <p>命令行解析后的结果：-m 模型、-p 提示词、-n 长度，还有整块采样设置，全在这一个结构体里。</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>common_init_from_params(params)</h4>
    <p>内部跑的正是 L25 那条手写序列：加载模型、建上下文、搭采样链，一口气全办了。</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>common_init_result -&gt; {model, context, sampler}</h4>
    <p>开箱即用的三件套，经 .model() / .context() / .sampler() 取出，工具直接拿去推理。</p>
  </div></div>
</div>
<p>这一步的意义，是把 L25 那条容易写错的长序列——<span class="mono">llama_model_load_from_file</span> 加载、<span class="mono">llama_init_from_model</span> 建上下文、再一节节搭采样链——整个收进了 <span class="mono">common_init_from_params</span> 内部。它还顺手照看了出错与释放：加载失败时返回一个空结果让你及早发现，成功时这个 <span class="mono">common_init_result</span> 则"持有"那几个对象、析构时一并释放，你不必再手动逐个 free。工具的 <span class="mono">main()</span> 因此干净许多：填好 <span class="mono">common_params</span>，调一次初始化，就拿到了三件套，省掉一大串样板。这也呼应了 L17 的 cparams、L14 的模型加载——那些底层步骤一个没少，只是被收进了这层胶水里。</p>

<h2>命令行如何变成配置</h2>
<p><span class="mono">common_params</span> 里的字段是从哪来的？大多来自<strong>命令行</strong>。<span class="mono">common/arg.{h,cpp}</span> 把每一个命令行选项都描述成一个 <span class="mono">common_arg</span> 对象：它记着选项的名字（可同时给 <span class="mono">-m</span> 短写和 <span class="mono">--model</span> 长写）、一段帮助文本，以及一个"拿到值就往 <span class="mono">common_params</span> 里写"的回调。所有 <span class="mono">common_arg</span> 的帮助文本还会被自动汇总，拼成你敲 <span class="mono">--help</span> 时看到的那张说明表，省得另写一份文档、还总忘了同步。声明时更能用链式 builder 微调行为——<span class="mono">.set_examples()</span> 限定它属于哪些工具、<span class="mono">.set_env()</span> 允许从环境变量取值、<span class="mono">.set_excludes()</span> 把某些工具排除在外。</p>
<pre class="code"><span class="cm">// 声明 "-m / --model" 这个选项 (简化自 common/arg.cpp)</span>
<span class="fn">add_opt</span>(<span class="fn">common_arg</span>(
    {<span class="st">"-m"</span>, <span class="st">"--model"</span>}, <span class="st">"FNAME"</span>, <span class="st">"model path to load"</span>,
    [](common_params &amp; params, <span class="kw">const</span> std::string &amp; value) {
        params.model.path = value;          <span class="cm">// 拿到值就写进 common_params</span>
    }
).<span class="fn">set_examples</span>({LLAMA_EXAMPLE_COMMON}).<span class="fn">set_env</span>(<span class="st">"LLAMA_ARG_MODEL"</span>));

<span class="cm">// 把整条 argv 解析进 params; 第 4 个参数选哪套工具的选项集</span>
<span class="fn">common_params_parse</span>(argc, argv, params, LLAMA_EXAMPLE_CLI);</pre>
<p>真正把命令行"灌"进结构体的，是 <span class="mono">common_params_parse(argc, argv, params, ex)</span>：它遍历 <span class="mono">argv</span>，按名字找到对应的 <span class="mono">common_arg</span>，挨个调用回调写进 <span class="mono">params</span>。第 4 个参数 <span class="mono">ex</span> 是一个 <span class="mono">enum llama_example</span>（如 <span class="mono">LLAMA_EXAMPLE_CLI</span>、<span class="mono">LLAMA_EXAMPLE_SERVER</span>、<span class="mono">LLAMA_EXAMPLE_COMMON</span>），它决定<strong>这次解析认哪些选项</strong>——同一套机制，因此能给不同工具暴露不同的参数子集：cli 有 cli 的选项、server 有 server 的，互不打架，而标成 <span class="mono">COMMON</span> 的那批则是大家共享的基本选项。若环境变量与命令行同时给了值，命令行优先——这样在 CI 里用环境变量设默认、临时在命令行覆盖，就显得很自然。</p>
<p>光说不够直观。下面顺着一条最小命令行 <span class="mono">-m model.gguf -p "Hi" -n 16 --temp 0.7</span>，看它怎么一步步变成 <span class="mono">common_params</span> 字段、再被交去产出三件套：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次参数解析</b>：最小命令行如何被 common_params_parse 填进结构体、再交给 common_init_from_params 产出三件套（数值为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① argv</h5>
      <div class="cellrow"><span class="vc">-m model.gguf</span><span class="vc">-p "Hi"</span><span class="vc">-n 16</span><span class="vc">--temp 0.7</span></div>
      <div class="tlab">原始命令行</div></div>
    <div class="op">common_<br>params_parse</div>
    <div class="stn"><h5>② common_params 字段</h5>
      <div class="cellrow"><span class="vc blue">model.path=model.gguf</span><span class="vc blue">prompt="Hi"</span><span class="vc blue">n_predict=16</span><span class="vc blue">sampling.temp=0.70</span></div>
      <div class="tlab">每个选项写进对应字段</div></div>
    <div class="op">common_init<br>from_params</div>
    <div class="stn"><h5>③ 就绪</h5>
      <div class="cellrow"><span class="vc hot">{model, ctx, sampler}</span></div>
      <div class="tlab">开箱即用的三件套</div></div>
  </div>
</div>

<h2>采样包装：common_sampler</h2>
<p>采样这件事，L21 讲过它的底层是一条 <span class="mono">llama_sampler</span> 链，L23 又讲了 GBNF 语法约束。这两样本是分开的：链负责按概率挑词，语法负责"只准挑合法的词"。common 把它们<strong>裹成一个对象</strong> <span class="mono">common_sampler</span>（见 <span class="mono">common/sampling.{h,cpp}</span>）：采样链 + 语法捆在一起，对外只露出一个句柄，省得每个工具自己去操心"先过语法还是先采样"这类细节。<span class="mono">common_sampler_init(model, params.sampling)</span> 会照着 <span class="mono">params.sampling.samplers</span> 里列出的顺序（一串 <span class="mono">common_sampler_type</span>，如 <span class="mono">COMMON_SAMPLER_TYPE_TOP_K</span> / <span class="mono">TOP_P</span> / <span class="mono">TEMPERATURE</span>）把链一节节搭起来。</p>
<div class="flow">
  <div class="node"><div class="nt">top_k</div><div class="nd">留前 k 个</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">top_p</div><div class="nd">核采样</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">temp</div><div class="nd">温度缩放</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">dist</div><div class="nd">按分布抽样</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">token</div><div class="nd">选出下一个</div></div>
</div>
<p>用起来更省事：<span class="mono">common_sampler_sample(gsmpl, ctx, idx, grammar_first)</span> 一次调用，就把"取 logits、过语法掩码、走完整条链、挑出 token"全办了；选完再用 <span class="mono">common_sampler_accept(...)</span> 把这个 token 喂回去，更新重复惩罚和语法状态。<span class="mono">llama-cli</span>、<span class="mono">llama-server</span> 用的都是<strong>这一层</strong>，而不是直接碰 L21 那套裸 <span class="mono">llama_sampler_*</span>。上图只画了默认链里的几节（真实默认还含 penalties、dry、min_p 等），但顺序的道理一样：先一层层筛掉不要的候选，最后按分布抽一个。</p>
<pre class="code"><span class="cm">// cli/server 都用这层, 不直接碰裸 llama_sampler_* (简化自 common/sampling.h)</span>
common_sampler * smpl = <span class="fn">common_sampler_init</span>(model, params.sampling);
<span class="cm">// ... 每步 ...</span>
llama_token id = <span class="fn">common_sampler_sample</span>(smpl, ctx, -1);   <span class="cm">// 取分-&gt;过语法-&gt;采样, 一步到位</span>
<span class="fn">common_sampler_accept</span>(smpl, id, <span class="cm">/*is_generated=*/</span> <span class="kw">true</span>); <span class="cm">// 反馈: 更新惩罚与语法状态</span></pre>
<p>注意这是个<strong>每步都要走一遍</strong>的循环：每生成一个 token，就 <span class="mono">common_sampler_sample</span> 选一个、再 <span class="mono">common_sampler_accept</span> 反馈一次，如此往复，直到遇上结束符或写满预定长度。<span class="mono">accept</span> 这一步至关重要——重复惩罚要靠它记住"已经出过哪些词"，语法状态也要靠它推进到下一个合法位置。参数里的 <span class="mono">grammar_first</span> 则控制"语法掩码"先于还是后于其它采样器生效：多数情况下用默认即可，只有当语法很严、又想让温度等先发挥作用时才需要调整。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  <span class="mono">common_sampler</span> 不是另起炉灶，而是<strong>薄薄一层壳</strong>：内部仍是 L21 那条 <span class="mono">llama_sampler</span> 链，只是额外捆上语法、候选缓冲和一点状态管理。需要时还能用 <span class="mono">common_sampler_get(gsmpl)</span> 把底层那条裸 <span class="mono">llama_sampler</span> 链取回来。换句话说，它和 <span class="mono">llama_sampler_*</span> 不是二选一，而是"上层便利 + 下层内核"的关系——和 common 整体之于 <span class="mono">llama.h</span> 的关系如出一辙。日常使用几乎不必直接碰那条裸链，但知道它"随时能拆开"这一点，会在你要做特殊采样时留出一条后路。
</div>

<h2>下载与缓存</h2>
<p>还有一桩重复的杂活：弄到模型文件本身。<span class="mono">common/download.{h,cpp}</span> 让你能直接写 <span class="mono">-hf user/repo:tag</span> 从 Hugging Face（一个公开的模型托管站）拉模型，省去"先手动下载、再用 <span class="mono">-m</span> 指路"那两步。第一步是 <span class="mono">common_download_split_repo_tag("repo:tag")</span>，把 <span class="mono">repo:tag</span> 这种写法拆成仓库名和标签两段——标签常用来挑某一种量化（如 <span class="mono">Q4_K_M</span>）；接着按这个信息去 HF 上把文件下载下来，并在本地建一份缓存。遇到被切成多片的大模型（多个 <span class="mono">.gguf</span> 分卷），它也会把各片一并取齐。</p>
<div class="flow">
  <div class="node"><div class="nt">-hf user/repo:tag</div><div class="nd">命令行写法</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">split_repo_tag</div><div class="nd">拆出 repo 与 tag</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">HF 下载</div><div class="nd">首次才联网</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">本地缓存</div><div class="nd">~/.cache/llama.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">model.path</div><div class="nd">当成本地文件用</div></div>
</div>
<p>这条流水线最关键的一步是<strong>先查缓存</strong>：真正联网下载前，它会先看本地是否已有这个文件、且与远端版本一致，若有就直接跳过下载。所以上图里"HF 下载"那一格，其实只有首次运行才真的会走，之后都被缓存短路掉了。</p>
<p>缓存是关键：模型只在<strong>第一次</strong>用到时才下载，落进本地缓存目录后，以后每次启动都直接命中、秒开。缓存默认在 <span class="mono">~/.cache/llama.cpp/</span>（可用环境变量 <span class="mono">LLAMA_CACHE</span> 覆盖）；<span class="mono">common_list_cached_models()</span> 能列出已经缓存的模型；下载过程则由 <span class="mono">common_download_callback</span>（带 <span class="mono">on_start</span> / <span class="mono">on_update</span> / <span class="mono">on_done</span> 三个回调，配合 <span class="mono">common_download_progress</span>）驱动那条进度条。这份缓存还是各工具<strong>共享</strong>的：cli、server、bench 指向同一个仓库时，命中的都是同一份本地副本，不会各下一遍。</p>
<div class="card spark">
  <div class="tag">💡 动手试试</div>
  动手试一下：第一次跑 <span class="mono">llama-cli -hf &lt;user&gt;/&lt;repo&gt;</span> 会看到一段下载进度，之后再跑同一个就瞬间启动——因为文件已经躺在 <span class="mono">~/.cache/llama.cpp/</span> 里了。想把缓存搬到大硬盘？设一个 <span class="mono">LLAMA_CACHE=/path/to/dir</span> 即可；想知道缓存了哪些，去那个目录翻一翻、或在代码里调 <span class="mono">common_list_cached_models()</span>。这套"下一次就免下载"的机制，正是 common 把"弄到模型"这桩杂活也一并包圆的体现：你只管写 <span class="mono">-hf</span>，下载、命名、缓存、复用，它都替你想好了。当然，第一次仍要联网、也得留足磁盘空间；跑通一次之后，就再不必为"模型在哪"操心了。
</div>

<h2>深入：边界与两个小帮手</h2>
<p>最后用两个折叠，补两个常被问到的点：为什么反复强调"common 不是公共 API"，以及那些不起眼、却天天在用的小工具（日志与终端）。这两点，一个关乎"边界"——你到底该依赖哪一层，一个关乎"手感"——调试时是谁在默默帮你，放在一起正好给本课收个尾。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么说 common 不算"公共 API"？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">include/llama.h</span> 是有 ABI 稳定承诺的对外契约（L25），各语言绑定都照着它写；而 <span class="mono">common/</span> 没有这种承诺——它的结构体布局、函数签名会随项目需要随时改动，因为它本就是<strong>给自家工具用的内部便利库</strong>，不在"对外保证"的范围里。所以一条实用的分界是：写第三方绑定（Python/Go/Rust）请直接对着 <span class="mono">llama.h</span>，把 common 当成"可以参考、但别依赖"的示例；只有当你给 llama.cpp <strong>贡献自带工具</strong>（往 <span class="mono">tools/</span> 或 <span class="mono">examples/</span> 里加东西）时，才该站上 common 的肩膀。分清这条线，能帮你躲开"绑定依赖了 common，下次一更新就编不过"的坑。</p>
    <p>一个简单的判断法：如果你的代码只要 <span class="mono">#include "llama.h"</span> 就够用，就别去 include common 里的头文件；只有当你在写自带工具、确实想复用参数解析或采样封装时，才把 common 一起编进来。这条线也解释了为什么本课开头一再强调它——把"对外稳定"和"对内便利"分清楚，是用好整个项目的前提。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> log 与 console：两个不起眼的帮手 <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">common/log.{h,cpp}</span> 提供<strong>分级日志</strong>：<span class="mono">LOG_INF</span> / <span class="mono">LOG_WRN</span> / <span class="mono">LOG_ERR</span> / <span class="mono">LOG_DBG</span> 分别对应信息、警告、错误、调试四档，底层由一个带后台线程的 <span class="mono">common_log</span> 统一输出（异步打印，不拖慢主线程）。调试时把日志级别调高、多打几行 <span class="mono">LOG_DBG</span>，就能看清模型加载、批次、采样每一步到底发生了什么，比到处插 <span class="mono">printf</span> 干净得多。日志级别可用 <span class="mono">-v</span> 或环境变量调节，无需重新编译。</p>
    <p><span class="mono">common/console.{h,cpp}</span> 则管<strong>终端交互与着色</strong>：<span class="mono">console::set_display(...)</span> 切换不同的显示类别（提示、用户输入、错误等各有颜色），<span class="mono">console::readline(...)</span> 处理一行输入（含多行与 UTF-8）。<span class="mono">llama-cli</span> 里那个带颜色、能正常敲中文的交互界面，就是靠它撑起来的。这两个小工具谈不上"核心"，却实实在在让工具用着顺手、调着省心——也是 common"把杂活包圆"的一部分。调试交互式会话时，颜色能正常显示、退格与中文输入都不串位，靠的正是这层不起眼的封装。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">common/</span> 是 llama.cpp 自带工具的<strong>共享胶水层</strong>，把裸 C API 包成顺手的 C++；它<strong>不是</strong>公共 API（无 ABI 稳定承诺），第三方绑定应直接用 <span class="mono">llama.h</span>。</li>
    <li><span class="mono">common_params</span> 一个大结构体装下所有旋钮；<span class="mono">common_init()</span> 做全局初始化，<span class="mono">common_init_from_params(params)</span> 产出 <span class="mono">common_init_result</span>（model + context + sampler）。</li>
    <li>命令行：每个选项是一个 <span class="mono">common_arg</span>（链式 <span class="mono">.set_examples/.set_env/.set_excludes</span>）；<span class="mono">common_params_parse(argc, argv, params, ex)</span> 按 <span class="mono">enum llama_example</span> 把 argv 填进 <span class="mono">common_params</span>。</li>
    <li>采样：<span class="mono">common_sampler</span> 把 L21 的 <span class="mono">llama_sampler</span> 链 + L23 的 GBNF 语法裹成一个对象；<span class="mono">common_sampler_init</span> 按 <span class="mono">samplers</span> 顺序建链，<span class="mono">common_sampler_sample</span> 一步采样。cli/server 用这层、不用裸采样器。</li>
    <li>下载：<span class="mono">-hf user/repo:tag</span> 经 <span class="mono">common_download_split_repo_tag</span> + HF 缓存变成本地文件；首次下载、之后命中 <span class="mono">~/.cache/llama.cpp/</span>。</li>
    <li>小帮手：<span class="mono">common/log</span> 给分级日志（<span class="mono">LOG_INF/WRN/ERR/DBG</span>，异步输出），<span class="mono">common/console</span> 管终端着色与交互输入——调试时省心不少。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  common 的设计哲学，一句话：<strong>复用胜过重写</strong>。它没有发明任何新机制，只是把每个工具都要做的重复动作——填参数、配采样、下模型、打日志——抽到一处，让 cli/server 的 <span class="mono">main()</span> 短到一眼能读完。但它刻意<strong>不</strong>把自己伪装成对外接口：稳定契约留给 <span class="mono">llama.h</span>，common 只做"对内顺手"。这条"对外稳定、对内便利"的分工，和你在 L25 见到的"稳定的表面 + 自由的内部"是同一套思路——只不过这次，common 站在了"便利"的那一端。看懂这层，第五部分接下来的 cli 与 server，就只是 common 之上各搭各的楼了。把这一层吃透，你会发现后面的工具课大多是在看"如何组合 common 的零件"，而不是又一套全新机制。说到底，common 教给我们的，是一种"把重复的杂活抽出来、把稳定的承诺留在边界上"的工程素养——这份在"对外稳定"与"对内便利"之间拿捏分寸的眼光，比记住任何单个函数名都更值得带走。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson (L25) we met the C API in <span class="mono">include/llama.h</span> - it really can drive the whole machine, but using it is rather "verbose": filling one <span class="mono">llama_batch</span> means setting token/pos/seq_id field by field; building a sampler chain means adding it node by node with <span class="mono">llama_sampler_chain_add</span>; parsing the command line into params and pulling a model off the network - every tool writes all of this from scratch. Those repeated chores are exactly what the <span class="mono">common/</span> layer packages away for you.
</p>
<p style="color:var(--muted);margin-top:.4rem"><span class="mono">common/</span> is llama.cpp's built-in "shared glue layer": it wraps those scattered C calls into a set of handy C++ helpers, so bundled tools like <span class="mono">llama-cli</span> and <span class="mono">llama-server</span> all stand on its shoulders instead of each writing their own boilerplate. But one thing must be clear up front: <strong>common is not the public API</strong> - it carries no ABI-stability promise like <span class="mono">llama.h</span> does, it is just an internal convenience library for the project's own tools; third-party language bindings should write against <span class="mono">llama.h</span> directly rather than depend on common.</p>
<p style="color:var(--muted)">So what chores does this layer actually package? Roughly four kinds: parse the command line into one big config, initialize the model and context from that config in one shot, wrap the sampler chain and grammar constraint into one handy sampler, and download and cache the model file when you hand it only a repo name. This lesson walks those four things in turn, then spends a little time making clear "why common is not the public API", plus two small helpers used every day yet rarely mentioned on their own - logging and terminal coloring.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  common sits between <span class="mono">llama.h</span> (the stable foundation) and each tool (the rooms upstairs), a layer of scaffolding that faces inward only, never outward. Its whole value is one word: <strong>reuse</strong>. Gather the repeated actions every tool needs - parse args, configure samplers, download models, print logs - into one place, and a tool's <span class="mono">main()</span> shrinks to almost nothing but business logic. Understand common and you understand the secret behind how "small" cli/server look: not that they do less, but that the chores were packaged up ahead of time by common. Seen another way, common lets the "foundation" stay lean: anything handy to have but not worth freezing into the stable C API can sit safely up here in common, free to change later without disturbing outside users.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the layering as a restaurant chain: <span class="mono">llama.h</span> is the <strong>plumbing and wiring</strong> (stable, identical for everyone), common is the <strong>central kitchen</strong> (it does the washing, chopping, and sauce-prep every branch would otherwise repeat), and each tool is a <strong>storefront</strong> (just plate up to order). The storefronts never wash vegetables themselves, and the central kitchen never faces customers directly - it only serves the chain's own branches. That is what common's "inward, not outward" means: it is the back kitchen for the project's own tools, not an interface opened to outsiders. If the kitchen changes its knife work or swaps a sauce recipe, only the chain's own branches feel it, never the customers on the street; but if the plumbing changes spec, everyone in the city wired to it must follow - exactly the split between "freely adjustable inward, necessarily stable outward".
</div>

<h2>One-stop config: common_params</h2>
<p>At common's core is one <strong>big config struct</strong>, <span class="mono">common_params</span> (see <span class="mono">common/common.h</span>). The model path, the prompt, how many tokens to generate, how large the context is, plus a whole nested block of sampling settings - nearly every "knob" is packed into this one struct. It also carries a set of sane defaults, so you only override the few things you actually care about and leave the rest to run as-is. So a tool no longer passes a fistful of scattered parameters around; it just revolves around this single <span class="mono">common_params</span>: fill it by reference (<span class="mono">common_params &amp;</span>) while parsing the command line, then pull the whole thing out at init time - one object threaded end to end, sparing you layers of pass-through.</p>
<pre class="code"><span class="cm">// one big struct holds every knob (simplified from common/common.h)</span>
<span class="kw">struct</span> <span class="fn">common_params</span> {
    std::string  prompt;              <span class="cm">// -p  the prompt</span>
    int32_t      n_predict;           <span class="cm">// -n  how many tokens to generate</span>
    int32_t      n_ctx;               <span class="cm">// -c  context window size</span>
    common_params_sampling sampling;  <span class="cm">// nested: samplers / temp / grammar</span>
    common_params_model    model;     <span class="cm">// model.path / model.hf_repo ...</span>
    <span class="cm">// ... dozens more fields</span>
};</pre>
<p>Note the two <strong>nested sub-structs</strong>: <span class="mono">sampling</span> (that is <span class="mono">common_params_sampling</span>, holding the sampler-chain order, temperature, grammar, and so on) and <span class="mono">model</span> (that is <span class="mono">common_params_model</span>, holding the local <span class="mono">model.path</span> and the HF <span class="mono">model.hf_repo</span>). Tucking related knobs into their own little structs keeps <span class="mono">common_params</span> from turning into one big stew, and lets modules like sampling and download each take just what they need.</p>
<p>With the config in hand, how does it become runnable objects? In two steps. First call <span class="mono">common_init()</span> once for <strong>global init</strong> (start the logging system, print build info) - this step is model-independent and runs once per program; the real work is done by <span class="mono">common_init_from_params(params)</span>, which takes the filled <span class="mono">common_params</span> and produces, in one shot, a <span class="mono">common_init_result</span> holding the loaded model, the built context, and the configured sampler (taken via <span class="mono">.model()</span> / <span class="mono">.context()</span> / <span class="mono">.sampler()</span>). The two steps are split because global init must happen before any logging, or early errors could be swallowed silently.</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>common_params (the filled config)</h4>
    <p>The result of parsing the command line: -m model, -p prompt, -n length, plus the whole sampling block, all in this one struct.</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>common_init_from_params(params)</h4>
    <p>Inside, it runs exactly L25's hand-written sequence: load the model, build the context, build the sampler chain - all at once.</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>common_init_result -&gt; {model, context, sampler}</h4>
    <p>The ready-to-use trio, taken via .model() / .context() / .sampler(), handed straight to inference.</p>
  </div></div>
</div>
<p>The point of this step is that L25's long, easy-to-misfire sequence - <span class="mono">llama_model_load_from_file</span> to load, <span class="mono">llama_init_from_model</span> to build the context, then the sampler chain node by node - is folded entirely inside <span class="mono">common_init_from_params</span>. It also looks after errors and cleanup along the way: on a failed load it returns an empty result so you notice early, and on success the <span class="mono">common_init_result</span> "owns" those objects and frees them together on destruction, so you need not free each one by hand. A tool's <span class="mono">main()</span> is much cleaner for it: fill <span class="mono">common_params</span>, call init once, and the trio is ready, sparing you a long stretch of boilerplate. This echoes L17's cparams and L14's model loading - none of those low-level steps are gone, they are just tucked into this glue layer.</p>

<h2>How the command line becomes config</h2>
<p>Where do the fields in <span class="mono">common_params</span> come from? Mostly from the <strong>command line</strong>. <span class="mono">common/arg.{h,cpp}</span> describes each command-line option as a <span class="mono">common_arg</span> object: it records the option's names (a short <span class="mono">-m</span> and a long <span class="mono">--model</span> at once), a line of help text, and a callback that "writes the value into <span class="mono">common_params</span>". The help text of every <span class="mono">common_arg</span> is also gathered automatically into the table you see when you type <span class="mono">--help</span>, sparing a separate document that always drifts out of sync. At declaration you can further tune behavior with chainable builders - <span class="mono">.set_examples()</span> limits which tools it belongs to, <span class="mono">.set_env()</span> lets it read from an environment variable, <span class="mono">.set_excludes()</span> rules certain tools out.</p>
<pre class="code"><span class="cm">// declare the "-m / --model" option (simplified from common/arg.cpp)</span>
<span class="fn">add_opt</span>(<span class="fn">common_arg</span>(
    {<span class="st">"-m"</span>, <span class="st">"--model"</span>}, <span class="st">"FNAME"</span>, <span class="st">"model path to load"</span>,
    [](common_params &amp; params, <span class="kw">const</span> std::string &amp; value) {
        params.model.path = value;          <span class="cm">// the value goes into common_params</span>
    }
).<span class="fn">set_examples</span>({LLAMA_EXAMPLE_COMMON}).<span class="fn">set_env</span>(<span class="st">"LLAMA_ARG_MODEL"</span>));

<span class="cm">// parse the whole argv into params; the 4th arg picks which tool's option set</span>
<span class="fn">common_params_parse</span>(argc, argv, params, LLAMA_EXAMPLE_CLI);</pre>
<p>What actually "pours" the command line into the struct is <span class="mono">common_params_parse(argc, argv, params, ex)</span>: it walks <span class="mono">argv</span>, finds the matching <span class="mono">common_arg</span> by name, and calls each callback to write into <span class="mono">params</span>. The 4th argument <span class="mono">ex</span> is an <span class="mono">enum llama_example</span> (like <span class="mono">LLAMA_EXAMPLE_CLI</span>, <span class="mono">LLAMA_EXAMPLE_SERVER</span>, <span class="mono">LLAMA_EXAMPLE_COMMON</span>) that decides <strong>which options this parse recognizes</strong> - one mechanism, so it can expose different parameter subsets to different tools: cli has cli's options, server has server's, with no clashes, while the ones marked <span class="mono">COMMON</span> are the shared basics everyone gets. When an environment variable and the command line both supply a value, the command line wins - so setting defaults via env in CI and overriding on the command line ad hoc feels natural.</p>
<p>Words alone are not concrete enough. Below we follow one minimal command line, <span class="mono">-m model.gguf -p "Hi" -n 16 --temp 0.7</span>, and watch it turn step by step into <span class="mono">common_params</span> fields, then get handed off to produce the trio:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one arg parse</b>: how a minimal command line is filled into the struct by common_params_parse, then handed to common_init_from_params to produce the trio (values are illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) argv</h5>
      <div class="cellrow"><span class="vc">-m model.gguf</span><span class="vc">-p "Hi"</span><span class="vc">-n 16</span><span class="vc">--temp 0.7</span></div>
      <div class="tlab">raw command line</div></div>
    <div class="op">common_<br>params_parse</div>
    <div class="stn"><h5>(2) common_params fields</h5>
      <div class="cellrow"><span class="vc blue">model.path=model.gguf</span><span class="vc blue">prompt="Hi"</span><span class="vc blue">n_predict=16</span><span class="vc blue">sampling.temp=0.70</span></div>
      <div class="tlab">each option written to its field</div></div>
    <div class="op">common_init<br>from_params</div>
    <div class="stn"><h5>(3) ready</h5>
      <div class="cellrow"><span class="vc hot">{model, ctx, sampler}</span></div>
      <div class="tlab">the ready-to-use trio</div></div>
  </div>
</div>

<h2>The sampler wrapper: common_sampler</h2>
<p>As for sampling, L21 covered how its underlying form is a <span class="mono">llama_sampler</span> chain, and L23 covered GBNF grammar constraints. These two were separate by nature: the chain picks words by probability, the grammar enforces "only legal words allowed". common <strong>wraps them into one object</strong>, <span class="mono">common_sampler</span> (see <span class="mono">common/sampling.{h,cpp}</span>): the sampler chain plus the grammar bundled together, exposing only one handle outward, so each tool need not fuss over details like "apply the grammar before or after sampling". <span class="mono">common_sampler_init(model, params.sampling)</span> builds the chain node by node, following the order listed in <span class="mono">params.sampling.samplers</span> (a list of <span class="mono">common_sampler_type</span>, like <span class="mono">COMMON_SAMPLER_TYPE_TOP_K</span> / <span class="mono">TOP_P</span> / <span class="mono">TEMPERATURE</span>).</p>
<div class="flow">
  <div class="node"><div class="nt">top_k</div><div class="nd">keep top k</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">top_p</div><div class="nd">nucleus</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">temp</div><div class="nd">temperature scale</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">dist</div><div class="nd">sample by dist</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">token</div><div class="nd">pick the next</div></div>
</div>
<p>Using it is tidier: <span class="mono">common_sampler_sample(gsmpl, ctx, idx, grammar_first)</span> does "take logits, apply the grammar mask, run the whole chain, pick a token" in one call; afterward <span class="mono">common_sampler_accept(...)</span> feeds the token back to update repetition penalties and grammar state. <span class="mono">llama-cli</span> and <span class="mono">llama-server</span> both use <strong>this layer</strong>, not the raw <span class="mono">llama_sampler_*</span> from L21. The diagram draws only a few nodes of the default chain (the real default also includes penalties, dry, min_p, and more), but the logic of the order is the same: sieve out unwanted candidates layer by layer, then draw one by distribution at the end.</p>
<pre class="code"><span class="cm">// cli/server use this layer, not raw llama_sampler_* (simplified from common/sampling.h)</span>
common_sampler * smpl = <span class="fn">common_sampler_init</span>(model, params.sampling);
<span class="cm">// ... each step ...</span>
llama_token id = <span class="fn">common_sampler_sample</span>(smpl, ctx, -1);   <span class="cm">// logits-&gt;grammar-&gt;sample in one call</span>
<span class="fn">common_sampler_accept</span>(smpl, id, <span class="cm">/*is_generated=*/</span> <span class="kw">true</span>); <span class="cm">// feedback: update penalties and grammar state</span></pre>
<p>Note this is a loop you run <strong>every step</strong>: for each generated token you <span class="mono">common_sampler_sample</span> to pick one, then <span class="mono">common_sampler_accept</span> to feed it back, over and over until an end token or the planned length is reached. That <span class="mono">accept</span> step matters - the repetition penalty relies on it to remember "which words already appeared", and the grammar state relies on it to advance to the next legal position. The <span class="mono">grammar_first</span> argument controls whether the "grammar mask" applies before or after the other samplers: the default is fine most of the time, and you only adjust it when the grammar is strict yet you still want temperature and friends to act first.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  <span class="mono">common_sampler</span> does not start from scratch; it is a <strong>thin shell</strong>: inside is still L21's <span class="mono">llama_sampler</span> chain, just bundled with the grammar, a candidate buffer, and a little state management. When needed you can even call <span class="mono">common_sampler_get(gsmpl)</span> to get the underlying raw <span class="mono">llama_sampler</span> chain back. In other words, it and <span class="mono">llama_sampler_*</span> are not an either-or but an "upper-level convenience plus lower-level core" relationship - exactly like common as a whole relates to <span class="mono">llama.h</span>. In everyday use you hardly ever touch that raw chain directly, but knowing it "can be opened up any time" leaves you an escape hatch when you need some special sampling.
</div>

<h2>Download and cache</h2>
<p>There is one more repeated chore: getting the model file itself. <span class="mono">common/download.{h,cpp}</span> lets you write <span class="mono">-hf user/repo:tag</span> to pull a model straight from Hugging Face (a public model-hosting site), sparing the two-step dance of "download by hand, then point <span class="mono">-m</span> at it". The first step is <span class="mono">common_download_split_repo_tag("repo:tag")</span>, which splits the <span class="mono">repo:tag</span> form into a repo name and a tag - the tag often picks a particular quantization (like <span class="mono">Q4_K_M</span>); then it downloads the file from HF using that info and builds a local cache. For a big model split into several shards (multiple <span class="mono">.gguf</span> parts) it fetches all the parts together too.</p>
<div class="flow">
  <div class="node"><div class="nt">-hf user/repo:tag</div><div class="nd">command-line form</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">split_repo_tag</div><div class="nd">split repo and tag</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">HF download</div><div class="nd">network on first run</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">local cache</div><div class="nd">~/.cache/llama.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">model.path</div><div class="nd">use as a local file</div></div>
</div>
<p>The key step on this pipeline is to <strong>check the cache first</strong>: before any real network download, it looks at whether the file already exists locally and matches the remote version, and if so it skips the download entirely. So the "HF download" box above really only runs on the very first run; later runs are short-circuited by the cache.</p>
<p>The cache is the key: a model downloads only the <strong>first</strong> time it is used, and once it lands in the local cache directory, every later start hits it directly and opens instantly. The cache defaults to <span class="mono">~/.cache/llama.cpp/</span> (overridable via the <span class="mono">LLAMA_CACHE</span> environment variable); <span class="mono">common_list_cached_models()</span> lists the already-cached models; and the download itself is driven by <span class="mono">common_download_callback</span> (with three callbacks <span class="mono">on_start</span> / <span class="mono">on_update</span> / <span class="mono">on_done</span>, paired with <span class="mono">common_download_progress</span>) that powers the progress bar. This cache is also <strong>shared</strong> across tools: when cli, server, and bench point at the same repo, they all hit the same local copy rather than each downloading their own.</p>
<div class="card spark">
  <div class="tag">💡 Hands-on</div>
  Try it: the first run of <span class="mono">llama-cli -hf &lt;user&gt;/&lt;repo&gt;</span> shows a download progress bar, and running the same one again starts instantly - because the file already sits in <span class="mono">~/.cache/llama.cpp/</span>. Want to move the cache to a bigger disk? Set <span class="mono">LLAMA_CACHE=/path/to/dir</span>. Want to see what is cached? Browse that directory, or call <span class="mono">common_list_cached_models()</span> in code. This "no download next time" mechanism is common packaging up the "get the model" chore too: you just write <span class="mono">-hf</span>, and downloading, naming, caching, and reuse are all handled for you. Of course the first time still needs the network and enough disk space; once it has run through once, you never worry about "where the model is" again.
</div>

<h2>Deep dive: the boundary and two small helpers</h2>
<p>Finally, two folds for two often-asked points: why we keep stressing that "common is not the public API", and those unglamorous little utilities used every day (logging and the terminal). One is about the "boundary" - which layer you should actually depend on - and the other about the "feel" - who quietly helps you while debugging - and together they round off the lesson.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why does common not count as a "public API"? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">include/llama.h</span> is the outward contract with an ABI-stability promise (L25), and every language binding writes against it; <span class="mono">common/</span> carries no such promise - its struct layouts and function signatures change whenever the project needs, because it is an <strong>internal convenience library for the project's own tools</strong>, outside the "outward guarantee". So a practical dividing line is: for third-party bindings (Python/Go/Rust) write against <span class="mono">llama.h</span> directly and treat common as "fine to read, but do not depend on" sample code; only when you <strong>contribute a bundled tool</strong> to llama.cpp (adding something under <span class="mono">tools/</span> or <span class="mono">examples/</span>) should you stand on common's shoulders. Drawing this line keeps you clear of the trap where "a binding depended on common, and the next update no longer compiles".</p>
    <p>A simple test: if your code only needs <span class="mono">#include "llama.h"</span>, do not reach for common's headers; pull common in only when you are writing a bundled tool and genuinely want to reuse the arg parsing or the sampler wrapper. This line also explains why the lesson stresses it from the start - keeping "stable outward" and "convenient inward" apart is the prerequisite for using the whole project well.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> log and console: two unglamorous helpers <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">common/log.{h,cpp}</span> provides <strong>leveled logging</strong>: <span class="mono">LOG_INF</span> / <span class="mono">LOG_WRN</span> / <span class="mono">LOG_ERR</span> / <span class="mono">LOG_DBG</span> map to info, warning, error, and debug, all emitted through a <span class="mono">common_log</span> backed by a worker thread (asynchronous printing, so the main thread is not slowed). When debugging, raise the log level and print a few more <span class="mono">LOG_DBG</span> lines, and you can see exactly what happens at each step of model loading, batching, and sampling - far cleaner than scattering <span class="mono">printf</span> everywhere. The log level can be tuned with <span class="mono">-v</span> or an environment variable, no recompile needed.</p>
    <p><span class="mono">common/console.{h,cpp}</span> handles <strong>terminal interaction and coloring</strong>: <span class="mono">console::set_display(...)</span> switches display categories (prompt, user input, error, each with its own color), and <span class="mono">console::readline(...)</span> handles a line of input (including multi-line and UTF-8). The colored, input-capable interactive interface in <span class="mono">llama-cli</span> is held up by exactly this. Neither tool is "core", yet both genuinely make the tools pleasant to use and easy to debug - also part of common "packaging the chores". When you debug an interactive session, the colors showing correctly and backspace and UTF-8 input not garbling all come from this unglamorous wrapper.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">common/</span> is the <strong>shared glue layer</strong> for llama.cpp's bundled tools, wrapping the raw C API into handy C++; it is <strong>not</strong> the public API (no ABI-stability promise), and third-party bindings should use <span class="mono">llama.h</span> directly.</li>
    <li><span class="mono">common_params</span> is one big struct holding every knob; <span class="mono">common_init()</span> does global init, and <span class="mono">common_init_from_params(params)</span> produces a <span class="mono">common_init_result</span> (model + context + sampler).</li>
    <li>Command line: each option is a <span class="mono">common_arg</span> (chainable <span class="mono">.set_examples/.set_env/.set_excludes</span>); <span class="mono">common_params_parse(argc, argv, params, ex)</span> fills argv into <span class="mono">common_params</span> per <span class="mono">enum llama_example</span>.</li>
    <li>Sampling: <span class="mono">common_sampler</span> wraps L21's <span class="mono">llama_sampler</span> chain plus L23's GBNF grammar into one object; <span class="mono">common_sampler_init</span> builds the chain in <span class="mono">samplers</span> order, and <span class="mono">common_sampler_sample</span> samples in one step. cli/server use this layer, not the raw sampler.</li>
    <li>Download: <span class="mono">-hf user/repo:tag</span> becomes a local file via <span class="mono">common_download_split_repo_tag</span> + the HF cache; first run downloads, later runs hit <span class="mono">~/.cache/llama.cpp/</span>.</li>
    <li>Small helpers: <span class="mono">common/log</span> gives leveled logging (<span class="mono">LOG_INF/WRN/ERR/DBG</span>, async), and <span class="mono">common/console</span> handles terminal coloring and interactive input - a real relief when debugging.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  common's design philosophy in a phrase: <strong>reuse over rewrite</strong>. It invents no new mechanism; it just lifts the actions every tool repeats - fill params, configure sampling, download models, print logs - into one place, so cli/server's <span class="mono">main()</span> shrinks to something you can read at a glance. But it deliberately does <strong>not</strong> disguise itself as an outward interface: the stable contract is left to <span class="mono">llama.h</span>, and common only does "handy on the inside". This split of "stable outward, convenient inward" is the same thinking as L25's "stable surface plus free interior" - only this time, common stands on the "convenient" end. Grasp this layer and Part 5's coming cli and server are just buildings each raised on top of common. Master it and you will find the coming tool lessons are mostly about "how to assemble common's parts", not yet another brand-new mechanism. In the end, what common teaches is a kind of engineering taste - "lift the repeated chores out, keep the stable promise at the boundary" - and that sense of where to draw the line between "stable outward" and "convenient inward" is worth more than memorizing any single function name.
</div>
"""
}

LESSON_27 = {
     "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
<span class="mono">llama-cli</span> 多半是你第一个真正跑起来的 llama.cpp 程序——一句 <span class="mono">llama-cli -m model.gguf -p "从前有座山"</span>，模型就开始往下续写。它简单到几乎不用解释，可正因为简单，它是看清"一条命令如何变成满屏文字"的最佳样本。
</p>
<p style="color:var(--muted);margin-top:.4rem">这一课我们钻进它内部，把"解析命令行 + 一个生成循环"这套骨架拆开看；也借它看清 llama.cpp 的工具到底是怎么搭在 <span class="mono">common</span>（L26）和推理引擎之上的——工具课的"读法"，从这一课开始定调。</p>
<p style="color:var(--muted)">还有一点要先剧透：现代的 <span class="mono">llama-cli</span> 已经<strong>不是</strong>一份独立的裸 <span class="mono">llama_decode</span> 主循环了，它直接复用了 <span class="mono">llama-server</span> 的那台引擎。所以这一课既是"上手第一课"，也悄悄替下一课的 server 埋好了伏笔。</p>

<div class="card macro">
   <div class="tag">🌍 宏观理解</div>
   一句话给 cli 定位：它是<strong>给共享推理引擎套上的一层"命令行外壳"</strong>。引擎负责真正的重活——加载模型、维护 KV、前向、采样；cli 只负责"把人的意图喂进去、把模型的输出递出来"：从命令行和 stdin 读到你的 prompt，驱动引擎一轮轮生成，再把每个新词即时打到屏幕上。看懂这层"壳与引擎"的分工，你就明白为什么 cli 的源码出奇地短——重活早被引擎和 common 包圆了，cli 自己要写的，只剩"读输入、转一圈、流式输出"这点活。换个角度说，cli 的"短"恰恰是整套分层设计交出的成绩单：底层 C API（L25）足够稳，中间的 common（L26）足够厚，到了工具这一层自然就能薄。所以读 cli 的源码，与其说是在读"一个程序"，不如说是在读"前面那几层到底替它省下了多少重复劳动"。这也是我们把它排在第五部分靠前的原因——它像一面镜子，照见 C API 与 common 这些铺垫的价值。你越熟悉前四部分讲过的内部机件——加载、KV、批处理、采样——就越会惊讶于 cli 能把它们用得这么轻：几乎所有重活都发生在别处，它只负责把人和引擎接起来。
</div>

<div class="card analogy">
   <div class="tag">🔌 生活类比</div>
   把 cli 想成一台自动售货机的<strong>面板</strong>：你按下按钮（敲命令行 / 输入文字），机器内部（共享引擎）一件件出货（生成 token），面板只管把货<strong>一件件递出来</strong>（流式打印），不必关心里头是怎么造的。下一课的 server 则是<strong>同一台机器</strong>换了个"网络下单"的面板：货还是那批货，引擎还是那台引擎，只是接单与出货的方式从"按钮"变成了"HTTP 请求"。面板可以有很多种，机器只有一台——这正是 cli 与 server 的关系。这个类比还能再往前推一步：同一台机器，将来完全可以再接上第三块、第四块面板——比如一个 gRPC 服务、一个桌面 GUI——而机器内部的配方、火候、工序，一行都不必改。这正是"面板可换、机器唯一"的威力：你想支持一种新的接入方式，只要再写一层薄薄的壳，把请求翻译成引擎认得的格式就行，完全不用重写推理逻辑。记住这幅画面，下一课看 server 时你就会明白，它无非是换上了一块"能同时接很多张订单"的面板，灶台后面那台机器，还是你在 cli 里见过的那一台。
</div>

<h2>一条命令的旅程</h2>
<p>程序入口在 <span class="mono">tools/cli/main.cpp</span>，但它薄得几乎只有一行——把活儿立刻转交给 <span class="mono">tools/cli/cli.cpp</span>。cli.cpp 一上来先调 <span class="mono">common_init()</span> 起好日志系统，再用 <span class="mono">common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)</span> 把命令行<strong>灌进</strong> <span class="mono">common_params</span>（正是 L26 那套机制）。留意第 4 个参数 <span class="mono">LLAMA_EXAMPLE_CLI</span>：它告诉解析器"按 cli 这套选项集来认参数"，于是 cli 专属的旗标和共享旗标都能被正确识别。</p>
<pre class="code"><span class="cm">// 入口: 薄薄的 main 转入 cli.cpp (简化自 tools/cli)</span>
<span class="fn">common_init</span>();                                  <span class="cm">// 全局初始化: 起日志</span>
<span class="kw">if</span> (!<span class="fn">common_params_parse</span>(argc, argv, params, <span class="cm">/*ex=*/</span> LLAMA_EXAMPLE_CLI))
     <span class="kw">return</span> 1;                                   <span class="cm">// argv -&gt; common_params</span>
<span class="kw">auto</span> res = <span class="fn">common_init_from_params</span>(params);     <span class="cm">// 概念上拿到 model+ctx+sampler (cli 实经引擎)</span></pre>
<p>参数就位后，剩下的事几乎可以排成一条直线：按 <span class="mono">common_params</span> 载入模型、建好上下文，把 prompt 编码成 token 喂进去，进入生成循环，一边采样一边把新词流式打印。下图把这条主线画成五步——前两步你应该眼熟，正是 L25 那条手写序列，如今被 <span class="mono">common_init_from_params</span> 收进了一行。</p>
<div class="vflow">
   <div class="step"><div class="num">1</div><div class="sc">
     <h4>解析参数</h4>
     <p>common_params_parse 把 argv 按 LLAMA_EXAMPLE_CLI 填进 common_params。</p>
   </div></div>
   <div class="step"><div class="num">2</div><div class="sc">
     <h4>载入模型 + 建上下文</h4>
     <p>common_init_from_params 一口气加载 model、建好 context、配好 sampler。</p>
   </div></div>
   <div class="step"><div class="num">3</div><div class="sc">
     <h4>编码 prompt</h4>
     <p>把提示词 tokenize 成 token 序列，喂进上下文当作"已生成"的开头。</p>
   </div></div>
   <div class="step"><div class="num">4</div><div class="sc">
     <h4>生成循环</h4>
     <p>decode -&gt; 采样 -&gt; 接回上下文，一圈圈转，直到撞上停止条件。</p>
   </div></div>
   <div class="step"><div class="num">5</div><div class="sc">
     <h4>流式输出</h4>
     <p>每选定一个 token 就还原成文字、即时打印，不必等整段生成完。</p>
   </div></div>
</div>
<p>这条主线里，前两步（解析、初始化）几乎全被 common 包圆了（L26），真正属于 cli 自己的"主戏"，是第 4 步那个一圈圈转的<strong>生成循环</strong>——它把 L19 的 KV 增长、L21 的采样、L20 的 detokenize 串成了一个你能亲眼看到输出的闭环。说得直白些，前四个部分讲过的所有内部机件，到这里终于汇成了一个你能一行行看着它吐字的循环——很多人正是从 cli 开始"爱上"读 llama.cpp 源码的，因为它把抽象的推理，变成了屏幕上实实在在跳动的文字。下面就把这个循环单独拎出来看。</p>

<h2>生成主循环</h2>
<p>生成的本质是一个<strong>循环</strong>：每一轮让引擎前向一次（<span class="mono">llama_decode</span>）、拿到"下一个词的打分"（logits），用采样器（<span class="mono">common_sampler</span>，L26/L21）挑出一个 token，用 <span class="mono">common_token_to_piece</span> 把它还原成文字、即时打印，再把这个新 token 接回上下文，进入下一轮。如此往复，直到撞上三个"停"条件之一。</p>
<pre class="code"><span class="cm">// 生成主循环的本质 (现位于引擎 server_context 内, 非 cli.cpp 自有)</span>
<span class="kw">while</span> (n_remain != 0) {
     <span class="fn">llama_decode</span>(ctx, batch);                        <span class="cm">// 前向: 末位拿 logits</span>
     llama_token id = <span class="fn">common_sampler_sample</span>(smpl, ctx, -1); <span class="cm">// 采样下一个</span>
     <span class="fn">common_sampler_accept</span>(smpl, id, <span class="kw">true</span>);            <span class="cm">// 反馈: 更新惩罚/语法</span>
     <span class="kw">if</span> (<span class="fn">llama_vocab_is_eog</span>(vocab, id)) <span class="kw">break</span>;         <span class="cm">// 结束符 -&gt; 停</span>
     fputs(<span class="fn">common_token_to_piece</span>(ctx, id).c_str(), stdout); <span class="cm">// 流式打印</span>
     batch = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);              <span class="cm">// 新 token 接回去</span>
     n_remain--;                                         <span class="cm">// n_predict 计数</span>
}</pre>
<p>三个"停下来"的理由要记牢：写满了预定长度（<span class="mono">n_predict</span> 计数到 0）、模型自己吐出了<strong>结束符</strong>（EOG，呼应 L20/L21，用 <span class="mono">llama_vocab_is_eog</span> 判断）、或在交互模式下命中了你设的<strong>反向提示</strong>（antiprompt）。循环里那句 <span class="mono">common_sampler_accept</span> 也一步都不能省——它把刚选的 token 反馈回采样器，更新重复惩罚、推进语法状态（L23），下一轮才采得对。这一点初学时最容易忽略：很多人以为"采样"不过是挑个词那么简单，却忘了采样器其实是<strong>带记忆</strong>的——它要记住已经出过哪些词，好施加重复惩罚；要记住语法走到了哪一步，好约束下一个合法 token。一旦漏掉 accept，这些记忆就停在原地不更新，生成很快就会重复、跑偏甚至卡死。</p>
<p>把这一圈用一个最小例子走一遍最直观：假设上下文里已经有 "The cat"，看它如何生成下一个词、并把它流式吐到屏幕上。</p>
<div class="trace">
   <div class="tcap"><b>追踪生成主循环一轮</b>：从已生成的 token 出发，decode 拿 logits、采样选词、还原打印，再判断是否要停（数值为示意）。</div>
   <div class="stations">
     <div class="stn"><h5>① 已生成</h5>
       <div class="cellrow"><span class="vc">The</span><span class="vc">cat</span></div>
       <div class="tlab">上下文里的 token</div></div>
     <div class="op">decode<br>末位</div>
     <div class="stn"><h5>② logits</h5>
       <div class="cellrow"><span class="vc">logits[n_vocab]</span></div>
       <div class="tlab">下一词的打分</div></div>
     <div class="op">common_<br>sampler_sample</div>
     <div class="stn"><h5>③ 选定 token</h5>
       <div class="cellrow"><span class="vc hot">sat</span></div>
       <div class="tlab">采样挑出</div></div>
     <div class="op">token_to_piece<br>+ 打印</div>
     <div class="stn"><h5>④ 流式输出</h5>
       <div class="cellrow"><span class="vc blue">"The cat sat"</span></div>
       <div class="tlab">即时写到屏幕</div></div>
     <div class="op">回环<br>检查停</div>
     <div class="stn"><h5>⑤ 停?</h5>
       <div class="cellrow"><span class="vc">n_predict? EOG? antiprompt?</span></div>
       <div class="tlab">否则回到 ①</div></div>
   </div>
</div>

<h2>跑在共享引擎上</h2>
<p>现在揭开开头那个剧透。如果你翻开 <span class="mono">tools/cli/cli.cpp</span> 的头部，会看到它直接 <span class="mono">#include</span> 了 server 的几个头文件——<span class="mono">server-common.h</span>、<span class="mono">server-context.h</span>、<span class="mono">server-task.h</span>；<span class="mono">tools/cli/CMakeLists.txt</span> 里也把它链接到了 <span class="mono">server-context</span> 这个库。换句话说，现代 <span class="mono">llama-cli</span> 并没有自己再写一份裸的 <span class="mono">llama_decode</span> 主循环，而是<strong>复用了 server 那台引擎</strong> <span class="mono">server_context</span>（带 slot 与 task 的那一套，下一课细讲）。</p>
<pre class="code"><span class="cm">// tools/cli/cli.cpp 顶部: 直接复用 server 的引擎</span>
<span class="kw">#include</span> <span class="st">"server-common.h"</span>
<span class="kw">#include</span> <span class="st">"server-context.h"</span>   <span class="cm">// server_context: slot / task / KV</span>
<span class="kw">#include</span> <span class="st">"server-task.h"</span>

<span class="cm"># tools/cli/CMakeLists.txt: 链接 server-context 库</span>
target_link_libraries(${TARGET} PUBLIC server-context llama-common ...)</pre>
<p>这件事意味着什么？cli 和 server 其实<strong>共用同一台"发动机"</strong>，只是套了不同的"壳"：cli 的壳是命令行 + 交互终端，server 的壳是 HTTP + 多请求。引擎完全一样（加载、KV、批处理、采样都走同一套 <span class="mono">server_context</span>），区别只在"怎么把请求喂进去、怎么把结果递出来"。</p>
<div class="cols">
   <div class="col"><h4>llama-cli（命令行壳）</h4><p>从 argv / stdin 读 prompt，把生成的 token 流式打到 stdout；适合上手、脚本、交互对话。</p></div>
   <div class="col"><h4>llama-server（HTTP 壳）</h4><p>从 HTTP 请求收 prompt，把结果按 OpenAI 兼容格式返回；适合多用户、做服务。</p></div>
</div>
<p>历史上 cli 曾是一份独立的 <span class="mono">main.cpp</span> 生成循环，后来才统一到这台共享引擎上——好处是少维护一份几乎重复的代码，引擎里修一个 bug，cli 和 server 两边都跟着受益。也正因为如此，下一课 server 的不少概念（slot、连续批处理）其实你在 cli 里已经"隔着壳"用上了，只是没察觉而已。</p>
<div class="card macro">
   <div class="tag">🌍 宏观理解</div>
   "同引擎、异壳"是理解整个第五部分工具的一把钥匙。<span class="mono">server_context</span> 是那台发动机，cli 与 server 是两副不同的车壳：你换壳（换交互方式），但发动机不动。这种设计的价值在于<strong>单一事实源</strong>——推理逻辑只有一份实现，所有工具共享；要优化吞吐、修采样 bug、加新特性，只动引擎一处，全家受益。所以别把 cli 看成"另一套实现"，它更像 server 的一个轻量前台。把这把钥匙揣好，下一课直接拆发动机本身。再补一句这套设计的代价与回报。把引擎抽成共享组件，短期看是多添了一层抽象，读代码要多绕一道弯；可长期看，它换回的是"改一次、处处生效"的巨大便利。设想一下：要是 cli 和 server 各管各的生成循环，那么每修一个采样的边界 bug、每加一种新的停止条件，你都得在两个地方分别动手，还要时时提防两边行为不一致、悄悄跑偏。共享引擎把这种"双份维护"的负担一笔勾销了。这种"宁可多一层抽象，也要消灭重复"的取舍，是成熟工程里反复出现的母题，值得你在自己的项目里也留个心眼。
</div>

<h2>交互模式</h2>
<p>给命令行加上 <span class="mono">-i</span>，cli 就从"一次性续写"变成"来回对话"：它会在你按回车后，把你的输入编码进上下文，再继续生成，如此一问一答。打断生成靠<strong>反向提示</strong>（antiprompt / reverse prompt）——你设一个字符串（比如 <span class="mono">"User:"</span>），模型一旦要生成到它，就停下来、把话筒交还给你。终端的着色、退格、中文输入，则由 L26 的 <span class="mono">console</span> 在背后撑着。</p>
<p>几个最常打交道的旗标值得对着主循环记一下：<span class="mono">-n</span> 限制最多生成多少 token（就是 <span class="mono">n_predict</span>，管循环转几圈），<span class="mono">-c</span> 设上下文窗口多大（<span class="mono">n_ctx</span>，呼应 L17/L19），<span class="mono">--temp</span> 调采样温度（L21，管挑词那一步），<span class="mono">-i</span> 进交互。把每个旗标和循环里的某一步对上号，你就能预测它到底改变了什么。</p>
<div class="card spark">
   <div class="tag">💡 动手试试</div>
   最值得记的四个旗标：<span class="mono">-m</span> 指模型、<span class="mono">-p</span> 给提示词、<span class="mono">-n</span> 限生成长度、<span class="mono">-i</span> 进交互。想体会"壳与引擎"的分别，可以同一个模型先 <span class="mono">llama-cli -m x.gguf -p "讲个笑话" -n 64</span> 跑一次性续写，再 <span class="mono">llama-cli -m x.gguf -i</span> 进交互聊几句——你会发现底下那台引擎一模一样，变的只是你和它打交道的方式。再加上 <span class="mono">-c</span> 调上下文、<span class="mono">--temp</span> 调温度，把它们和这一课的生成循环对着看，"参数 -&gt; 循环行为"的因果就一目了然了。还可以再做个小实验加深印象：把 <span class="mono">--temp</span> 分别设成 0 和 1.2，各跑一次同样的 prompt，你会直观看到温度如何左右"挑词"那一步——设 0 时它几乎每次都吐一模一样的话，设 1.2 时则天马行空、花样百出。再把 <span class="mono">-n</span> 调得很小（比如 4），看它怎么话没说完就被硬生生截断，这就是 <span class="mono">n_predict</span> 这道闸门在起作用。把这些旗标一个个亲手拨动、对照生成循环看效果，远比死记每个参数的定义来得有用——你建立起来的，是"参数到行为"的肌肉记忆，将来调任何模型都用得上。
</div>

<h2>深入：复用的来龙去脉与"何时算停"</h2>
<p>最后用两个折叠，补两个常被追问的点：cli 复用 server 引擎的来龙去脉，以及"到底什么时候算生成结束"。前者关乎"架构为什么这么演化"，后者关乎一个你每次跑都会遇到、却未必说得清的细节。这两个问题看似琐碎，却分别对应着读源码时最常冒出的两种困惑：一种是"这段代码为什么长这样"（历史与权衡），一种是"它到底什么时候停"（运行时行为）。把它们说清楚，你再去翻 cli 的真实源码就不会被绕晕。</p>
<details class="accordion">
   <summary><span class="badge-num">1</span> cli 为什么要复用 server 的引擎？ <span class="hint">点击展开</span></summary>
   <div class="acc-body">
     <p>早期的 llama.cpp 里，cli（当时叫 <span class="mono">main</span>）和 server 各有一份生成循环：各自调 <span class="mono">llama_decode</span>、各自管 KV、各自处理停止条件。两份代码做的事高度重叠，却要分别维护——改一处采样逻辑，得记得两边都改，很容易漏。后来项目把引擎抽成共享的 <span class="mono">server_context</span>（slot/task 那套），让 cli 也站上去：cli 退化成"开一个 slot、喂一条序列、流式取回"的瘦客户端。好处是<strong>单一事实源</strong>——生成逻辑只剩一份实现，bug 修一处、特性加一处，cli 与 server 同时受益。</p>
     <p>所以当你在 cli 里看到 <span class="mono">server_task</span>、<span class="mono">server_slot</span> 这些名字时不必奇怪：它们不是"server 专用"，而是"引擎的词汇"。这也解释了为什么把 cli 放在 server（L28）<strong>前面</strong>讲——先在简单的命令行场景里见过这台引擎，下一课再看它如何同时服务多个 HTTP 请求，就顺理成章了。</p>
   </div>
</details>
<details class="accordion">
   <summary><span class="badge-num">2</span> "生成结束"到底由谁说了算？ <span class="hint">点击展开</span></summary>
   <div class="acc-body">
     <p>循环停下来有三种情形，触发者各不相同。其一是<strong>长度到顶</strong>：你用 <span class="mono">-n</span> 设的 <span class="mono">n_predict</span> 计数归零，这是"你"喊停。其二是<strong>模型自己喊停</strong>：它生成了一个 EOG（end-of-generation）token，比如 <span class="mono">&lt;/s&gt;</span> 或某些聊天模板里的 <span class="mono">&lt;|im_end|&gt;</span>——cli 用 <span class="mono">llama_vocab_is_eog(vocab, id)</span> 判断，呼应 L20 里那个"结束符集合"。其三是<strong>反向提示命中</strong>：交互模式下，模型快要生成到你设的 antiprompt 时被截停，把控制权还给你。</p>
     <p>这三者的优先级与细节，正是"为什么有时它早早就停 / 为什么停不下来"的根源：忘了设 <span class="mono">-n</span> 又遇上模型不肯吐 EOG，就可能一直生成；而某些模型的 EOG token 若没被模板正确标注，也会让它"刹不住车"。理解这三个闸门，你就能对症下药地控制生成长度。</p>
   </div>
</details>

<div class="card key">
   <div class="tag">✅ 关键要点</div>
   <ul>
     <li><span class="mono">llama-cli</span> = 给共享推理引擎套的一层<strong>命令行/交互外壳</strong>：读 stdin、驱动生成、流式打印 stdout，是上手 llama.cpp 最直接的工具。</li>
     <li>入口 <span class="mono">main.cpp</span>（薄）-&gt; <span class="mono">cli.cpp</span>；<span class="mono">common_init</span> + <span class="mono">common_params_parse(..., LLAMA_EXAMPLE_CLI)</span> 把命令行变成 <span class="mono">common_params</span>（L26）。</li>
     <li>生成主循环：<span class="mono">decode</span> 拿 logits -&gt; <span class="mono">common_sampler_sample</span> 选 token -&gt; <span class="mono">common_sampler_accept</span> 反馈 -&gt; <span class="mono">common_token_to_piece</span> 流式打印 -&gt; 接回上下文，循环往复。</li>
     <li>三个停止条件：<span class="mono">n_predict</span> 写满、EOG 结束符（<span class="mono">llama_vocab_is_eog</span>）、交互模式下命中反向提示（antiprompt）。</li>
     <li><strong>现状重点</strong>：现代 cli 复用 server 的引擎——<span class="mono">#include "server-context.h"</span> 并链接 <span class="mono">server-context</span>，与 server 同引擎、异壳（cli=命令行，server=HTTP）。</li>
   </ul>
</div>

<div class="card spark">
   <div class="tag">💡 设计洞察</div>
   cli 这一课真正想留给你的，不是某个旗标的用法，而是"<strong>壳与引擎分离</strong>"这一架构直觉。同一台 <span class="mono">server_context</span>，套上命令行壳就是 cli，套上 HTTP 壳就是 server——交互方式千变万化，推理内核始终如一。这种"把稳定的核做厚、把多变的壳做薄"的思路，和 L25 的"稳定 C ABI + 自由内部"、L26 的"对外稳定 + 对内便利"是同一条线索的延续：好的系统总在努力分清"哪些该统一、哪些该各异"。把这层想透，你看第五部分剩下的工具，就不会再把它们当成一个个孤立程序，而会看见底下那台被反复复用的引擎——下一课，我们就正面把它拆开。最后留一个值得反复咀嚼的问题给你：下次自己设计系统时，该怎么判断"哪一部分做成稳定的核、哪一部分做成可换的壳"？cli 给的答案朴素而有力——把"所有接入方式都共享的那部分"（也就是推理逻辑）沉进核里，把"每种接入方式各不相同的那部分"（命令行还是 HTTP）留在壳上。这条看似简单的分界线，其实适用于绝大多数需要支持多种入口的软件：Web 框架的路由与业务、数据库的协议层与存储引擎，背后都是同一种智慧。把它内化成你自己的设计直觉，你带走的就不只是"llama-cli 怎么用"，而是一种能迁移到任何项目的判断力。
</div>
""",
     "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
<span class="mono">llama-cli</span> is most likely the first llama.cpp program you ever run - one line, <span class="mono">llama-cli -m model.gguf -p "Once upon a time"</span>, and the model starts continuing the text. It is so simple it barely needs explaining, but that very simplicity makes it the best sample for seeing "how one command turns into a screen full of text".
</p>
<p style="color:var(--muted);margin-top:.4rem">This lesson digs inside it and takes apart the skeleton of "parse the command line + one generation loop"; it also uses cli to see how llama.cpp's tools actually sit on top of <span class="mono">common</span> (L26) and the inference engine - the "way to read" the tool lessons is set here.</p>
<p style="color:var(--muted)">One more spoiler up front: a modern <span class="mono">llama-cli</span> is <strong>no longer</strong> a standalone bare <span class="mono">llama_decode</span> main loop - it directly reuses <span class="mono">llama-server</span>'s engine. So this lesson is both "your first hands-on lesson" and a quiet setup for the server lesson next.</p>

<div class="card macro">
   <div class="tag">🌍 Big picture</div>
   cli in one line: it is <strong>a "command-line shell" wrapped around the shared inference engine</strong>. The engine does the real heavy lifting - load the model, maintain the KV, run the forward pass, sample; cli only "feeds in the human's intent and hands out the model's output": read your prompt from the command line and stdin, drive the engine round after round, and print each new word to the screen immediately. Understand this "shell vs engine" split and you see why cli's source is surprisingly short - the heavy lifting was packaged up by the engine and common, leaving cli with just "read input, turn a loop, stream output". Put differently, cli's "shortness" is the report card of the whole layered design: the C API underneath (L25) is stable enough, the common layer in the middle (L26) is thick enough, so by the tool layer it can afford to be thin. Reading cli's source is thus less like reading "a program" and more like reading "how much repeated labor the layers below saved it". That is also why we place it early in Part 5 - it is a mirror reflecting the value of the C API and common groundwork. The more familiar you are with the internal machinery of the first four parts - loading, KV, batching, sampling - the more it surprises you how lightly cli uses them: almost all the heavy lifting happens elsewhere, and it only wires the human to the engine.
</div>

<div class="card analogy">
   <div class="tag">🔌 Analogy</div>
   Think of cli as the <strong>panel</strong> of a vending machine: you press buttons (type the command line / input text), the machine inside (the shared engine) dispenses items one at a time (generates tokens), and the panel just <strong>hands them out one by one</strong> (streams the print), never minding how they were made inside. Next lesson's server is the <strong>same machine</strong> with an "order over the network" panel instead: same goods, same engine, only the way of taking orders and dispensing changes from "buttons" to "HTTP requests". There can be many panels, but only one machine - that is exactly the cli-and-server relationship. The analogy stretches one step further: the same machine could later take a third or fourth panel - a gRPC service, a desktop GUI - while the recipe, the heat, the steps inside change not one line. That is the power of "panels swappable, machine singular": to support a new way in, you write one more thin shell that translates requests into what the engine understands, with no rewrite of the inference logic. Hold this picture, and next lesson server makes sense at once - it is merely fitted with a panel that "takes many orders at once", while the machine behind the stove is the very one you met in cli.
</div>

<h2>A command's journey</h2>
<p>The entry point is <span class="mono">tools/cli/main.cpp</span>, but it is so thin it is almost one line - it immediately hands off to <span class="mono">tools/cli/cli.cpp</span>. cli.cpp first calls <span class="mono">common_init()</span> to start the logging system, then uses <span class="mono">common_params_parse(argc, argv, params, LLAMA_EXAMPLE_CLI)</span> to <strong>pour</strong> the command line into <span class="mono">common_params</span> (exactly L26's mechanism). Note the 4th argument <span class="mono">LLAMA_EXAMPLE_CLI</span>: it tells the parser "recognize parameters by cli's option set", so cli-specific flags and shared flags are both parsed correctly.</p>
<pre class="code"><span class="cm">// entry: the thin main hands off to cli.cpp (simplified from tools/cli)</span>
<span class="fn">common_init</span>();                                  <span class="cm">// global init: start logging</span>
<span class="kw">if</span> (!<span class="fn">common_params_parse</span>(argc, argv, params, <span class="cm">/*ex=*/</span> LLAMA_EXAMPLE_CLI))
     <span class="kw">return</span> 1;                                   <span class="cm">// argv -&gt; common_params</span>
<span class="kw">auto</span> res = <span class="fn">common_init_from_params</span>(params);     <span class="cm">// conceptually model+ctx+sampler (cli does it via the engine)</span></pre>
<p>With the parameters in place, the rest lines up almost straight: load the model and build the context from <span class="mono">common_params</span>, encode the prompt into tokens and feed it in, enter the generation loop, and stream out new words as you sample. The diagram below draws this main line as five steps - the first two should look familiar, they are exactly L25's hand-written sequence, now folded into one line of <span class="mono">common_init_from_params</span>.</p>
<div class="vflow">
   <div class="step"><div class="num">1</div><div class="sc">
     <h4>Parse args</h4>
     <p>common_params_parse fills argv into common_params per LLAMA_EXAMPLE_CLI.</p>
   </div></div>
   <div class="step"><div class="num">2</div><div class="sc">
     <h4>Load model + build context</h4>
     <p>common_init_from_params loads the model, builds the context, configures the sampler in one go.</p>
   </div></div>
   <div class="step"><div class="num">3</div><div class="sc">
     <h4>Encode prompt</h4>
     <p>Tokenize the prompt into a token sequence, fed in as the "already generated" opening.</p>
   </div></div>
   <div class="step"><div class="num">4</div><div class="sc">
     <h4>Generation loop</h4>
     <p>decode -&gt; sample -&gt; append back, turning round and round until a stop condition hits.</p>
   </div></div>
   <div class="step"><div class="num">5</div><div class="sc">
     <h4>Stream output</h4>
     <p>Each chosen token is restored to text and printed at once, no waiting for the whole thing.</p>
   </div></div>
</div>
<p>On this main line, the first two steps (parse, init) are almost entirely packaged by common (L26); the part that truly belongs to cli, its "main act", is step 4's looping <strong>generation loop</strong> - it threads L19's KV growth, L21's sampling, and L20's detokenize into a closed loop whose output you can watch live. Put plainly, all the internal machinery the first four parts covered finally converges here into a loop you can watch emit text line by line - many people first "fall for" reading llama.cpp's source from cli, because it turns abstract inference into words actually dancing on the screen. Let us pull that loop out and look at it alone.</p>

<h2>The generation main loop</h2>
<p>Generation is in essence a <strong>loop</strong>: each round runs one forward pass through the engine (<span class="mono">llama_decode</span>) to get "the next word's scores" (logits), uses the sampler (<span class="mono">common_sampler</span>, L26/L21) to pick a token, restores it to text with <span class="mono">common_token_to_piece</span> and prints it at once, then appends this new token back to the context and goes round again. So it repeats, until it hits one of three "stop" conditions.</p>
<pre class="code"><span class="cm">// the essence of the generation loop (now inside the engine, server_context)</span>
<span class="kw">while</span> (n_remain != 0) {
     <span class="fn">llama_decode</span>(ctx, batch);                        <span class="cm">// forward: logits at last pos</span>
     llama_token id = <span class="fn">common_sampler_sample</span>(smpl, ctx, -1); <span class="cm">// sample the next one</span>
     <span class="fn">common_sampler_accept</span>(smpl, id, <span class="kw">true</span>);            <span class="cm">// feedback: penalties/grammar</span>
     <span class="kw">if</span> (<span class="fn">llama_vocab_is_eog</span>(vocab, id)) <span class="kw">break</span>;         <span class="cm">// end-of-gen -&gt; stop</span>
     fputs(<span class="fn">common_token_to_piece</span>(ctx, id).c_str(), stdout); <span class="cm">// stream print</span>
     batch = <span class="fn">llama_batch_get_one</span>(&amp;id, 1);              <span class="cm">// append new token</span>
     n_remain--;                                         <span class="cm">// n_predict countdown</span>
}</pre>
<p>Keep the three "stop" reasons in mind: the set length is reached (<span class="mono">n_predict</span> counts down to 0), the model itself emits an <strong>end-of-generation</strong> token (EOG, echoing L20/L21, tested with <span class="mono">llama_vocab_is_eog</span>), or in interactive mode it hits the <strong>reverse prompt</strong> (antiprompt) you set. That <span class="mono">common_sampler_accept</span> line cannot be skipped either - it feeds the just-chosen token back to the sampler, updating repetition penalties and advancing grammar state (L23), so the next round samples correctly. This is the easiest thing to overlook as a beginner: many think "sampling" is just picking a word, forgetting the sampler is actually <strong>stateful</strong> - it must remember which words already appeared to apply repetition penalties, and where the grammar has advanced to constrain the next legal token. Skip accept and that memory freezes in place, so generation soon repeats, drifts, or even stalls.</p>
<p>Walking one turn with a minimal example is the most vivid: suppose the context already holds "The cat", and watch how it generates the next word and streams it to the screen.</p>
<div class="trace">
   <div class="tcap"><b>Tracing one generation step</b>: starting from the already-generated tokens, decode for logits, sample a word, restore and print, then decide whether to stop (values are illustrative).</div>
   <div class="stations">
     <div class="stn"><h5>(1) generated</h5>
       <div class="cellrow"><span class="vc">The</span><span class="vc">cat</span></div>
       <div class="tlab">tokens in context</div></div>
     <div class="op">decode<br>last pos</div>
     <div class="stn"><h5>(2) logits</h5>
       <div class="cellrow"><span class="vc">logits[n_vocab]</span></div>
       <div class="tlab">scores for next word</div></div>
     <div class="op">common_<br>sampler_sample</div>
     <div class="stn"><h5>(3) chosen token</h5>
       <div class="cellrow"><span class="vc hot">sat</span></div>
       <div class="tlab">picked by sampling</div></div>
     <div class="op">token_to_piece<br>+ print</div>
     <div class="stn"><h5>(4) streamed out</h5>
       <div class="cellrow"><span class="vc blue">"The cat sat"</span></div>
       <div class="tlab">written to screen now</div></div>
     <div class="op">loop<br>check stop</div>
     <div class="stn"><h5>(5) stop?</h5>
       <div class="cellrow"><span class="vc">n_predict? EOG? antiprompt?</span></div>
       <div class="tlab">else back to (1)</div></div>
   </div>
</div>

<h2>Running on the shared engine</h2>
<p>Now lift the spoiler from the start. If you open the top of <span class="mono">tools/cli/cli.cpp</span>, you will see it directly <span class="mono">#include</span>s several of server's headers - <span class="mono">server-common.h</span>, <span class="mono">server-context.h</span>, <span class="mono">server-task.h</span>; and <span class="mono">tools/cli/CMakeLists.txt</span> links it against the <span class="mono">server-context</span> library. In other words, a modern <span class="mono">llama-cli</span> does not write its own bare <span class="mono">llama_decode</span> main loop anymore, it <strong>reuses server's engine</strong> <span class="mono">server_context</span> (the slot-and-task machinery, detailed next lesson).</p>
<pre class="code"><span class="cm">// top of tools/cli/cli.cpp: reuse server's engine directly</span>
<span class="kw">#include</span> <span class="st">"server-common.h"</span>
<span class="kw">#include</span> <span class="st">"server-context.h"</span>   <span class="cm">// server_context: slot / task / KV</span>
<span class="kw">#include</span> <span class="st">"server-task.h"</span>

<span class="cm"># tools/cli/CMakeLists.txt: link the server-context library</span>
target_link_libraries(${TARGET} PUBLIC server-context llama-common ...)</pre>
<p>What does this mean? cli and server actually <strong>share one "engine"</strong>, just wrapped in different "shells": cli's shell is the command line plus an interactive terminal, server's shell is HTTP plus many requests. The engine is identical (loading, KV, batching, sampling all go through the same <span class="mono">server_context</span>); the only difference is "how requests are fed in and how results are handed out".</p>
<div class="cols">
   <div class="col"><h4>llama-cli (command-line shell)</h4><p>reads the prompt from argv / stdin, streams generated tokens to stdout; great for getting started, scripts, interactive chat.</p></div>
   <div class="col"><h4>llama-server (HTTP shell)</h4><p>takes the prompt from an HTTP request, returns results in an OpenAI-compatible shape; great for many users, running a service.</p></div>
</div>
<p>Historically cli was a standalone <span class="mono">main.cpp</span> generation loop, and only later was unified onto this shared engine - the gain is one less near-duplicate copy to maintain, and a bug fixed in the engine benefits cli and server alike. For the same reason, many of next lesson's server concepts (slots, continuous batching) you are in fact already using in cli "through the shell", just without noticing.</p>
<div class="card macro">
   <div class="tag">🌍 Big picture</div>
   "Same engine, different shells" is a key to the whole of Part 5's tools. <span class="mono">server_context</span> is that engine, and cli and server are two different car bodies: you swap the body (the way you interact), but the engine stays put. The value of this design is a <strong>single source of truth</strong> - the inference logic has exactly one implementation, shared by all tools; to optimize throughput, fix a sampling bug, or add a feature, you touch the engine in one place and the whole family benefits. So do not see cli as "another implementation", it is more like a lightweight front desk for server. Pocket this key, and next lesson we take the engine itself apart. One more word on this design's cost and reward. Extracting the engine into a shared component adds, short term, one more layer of abstraction and one more hop to follow while reading; but long term it buys the huge convenience of "fix once, effective everywhere". Imagine cli and server each minding their own generation loop: every boundary bug in sampling, every new stop condition would have to be changed in two places, while you guard against the two drifting out of sync. The shared engine wipes out that "double maintenance" entirely. This trade of "rather one more abstraction than any duplication" is a recurring motif in mature engineering, worth keeping an eye out for in your own projects.
</div>

<h2>Interactive mode</h2>
<p>Add <span class="mono">-i</span> to the command line and cli turns from "one-shot continuation" into "back-and-forth conversation": after you press enter, it encodes your input into the context and keeps generating, taking turns. Interrupting generation is done with the <strong>reverse prompt</strong> (antiprompt) - you set a string (say <span class="mono">"User:"</span>), and the moment the model is about to generate up to it, it stops and hands the microphone back to you. The terminal's coloring, backspace, and UTF-8 input are held up behind the scenes by L26's <span class="mono">console</span>.</p>
<p>A few of the flags you deal with most are worth noting against the main loop: <span class="mono">-n</span> caps how many tokens to generate (that is <span class="mono">n_predict</span>, governing how many times the loop turns), <span class="mono">-c</span> sets the context window size (<span class="mono">n_ctx</span>, echoing L17/L19), <span class="mono">--temp</span> tunes the sampling temperature (L21, governing the pick-a-word step), and <span class="mono">-i</span> enters interactive. Match each flag to a step in the loop and you can predict exactly what it changes.</p>
<div class="card spark">
   <div class="tag">💡 Hands-on</div>
   The four flags most worth remembering: <span class="mono">-m</span> for the model, <span class="mono">-p</span> for the prompt, <span class="mono">-n</span> to cap length, <span class="mono">-i</span> for interactive. To feel the "shell vs engine" split, take one model and first run <span class="mono">llama-cli -m x.gguf -p "tell a joke" -n 64</span> for a one-shot continuation, then <span class="mono">llama-cli -m x.gguf -i</span> to chat a few turns - you will find the engine underneath is identical, only the way you talk to it changes. Add <span class="mono">-c</span> for context and <span class="mono">--temp</span> for temperature, line them up against this lesson's generation loop, and the "param -&gt; loop behavior" cause and effect becomes plain. Try one more small experiment to cement it: set <span class="mono">--temp</span> to 0 and then to 1.2, running the same prompt each time, and you will see directly how temperature swings the "pick a word" step - at 0 it spits almost the same words every time, at 1.2 it runs wild and varied. Then set <span class="mono">-n</span> very small (say 4) and watch it get cut off mid-sentence - that is the <span class="mono">n_predict</span> gate at work. Turning these flags by hand one by one and watching the effect against the generation loop beats memorizing each definition - what you build is muscle memory of "param to behavior" that carries to any model.
</div>

<h2>Deep dive: the story of reuse and "when to stop"</h2>
<p>Finally two folds for two often-asked points: the backstory of cli reusing server's engine, and "when exactly generation counts as finished". The first is about "why the architecture evolved this way", the second about a detail you meet every run yet may not be able to explain. These two questions look trivial, yet each answers one of the two confusions that most often surface while reading source: one is "why is this code shaped this way" (history and trade-offs), the other "when exactly does it stop" (runtime behavior). Make them clear and you will not get lost when you open cli's real source.</p>
<details class="accordion">
   <summary><span class="badge-num">1</span> Why does cli reuse server's engine? <span class="hint">click to expand</span></summary>
   <div class="acc-body">
     <p>In early llama.cpp, cli (then called <span class="mono">main</span>) and server each had their own generation loop: each calling <span class="mono">llama_decode</span>, each managing the KV, each handling stop conditions. The two bodies of code did highly overlapping things yet were maintained separately - change one piece of sampling logic and you had to remember to change both, easy to miss. Later the project extracted the engine into a shared <span class="mono">server_context</span> (the slot/task machinery) and let cli stand on it too: cli degenerates into a thin client that "opens one slot, feeds one sequence, streams it back". The gain is a <strong>single source of truth</strong> - generation logic has one implementation, a bug fixed once and a feature added once benefit cli and server together.</p>
     <p>So when you see names like <span class="mono">server_task</span> and <span class="mono">server_slot</span> in cli, do not be surprised: they are not "server-only", they are "the engine's vocabulary". This also explains why cli is taught <strong>before</strong> server (L28) - meet the engine first in the simple command-line setting, and next lesson, seeing it serve many HTTP requests at once follows naturally.</p>
   </div>
</details>
<details class="accordion">
   <summary><span class="badge-num">2</span> Who decides "generation is done"? <span class="hint">click to expand</span></summary>
   <div class="acc-body">
     <p>The loop stops in three cases, each with a different trigger. First, <strong>length cap reached</strong>: the <span class="mono">n_predict</span> you set with <span class="mono">-n</span> counts to zero - "you" call stop. Second, <strong>the model calls stop itself</strong>: it generates an EOG (end-of-generation) token, such as <span class="mono">&lt;/s&gt;</span> or some chat templates' <span class="mono">&lt;|im_end|&gt;</span> - cli tests it with <span class="mono">llama_vocab_is_eog(vocab, id)</span>, echoing L20's "end-of-generation set". Third, <strong>reverse prompt hit</strong>: in interactive mode, the model is cut off just as it is about to generate up to your antiprompt, handing control back to you.</p>
     <p>The priority and details of these three are the root of "why it sometimes stops early / why it will not stop": forget to set <span class="mono">-n</span> and meet a model unwilling to emit EOG, and it may generate forever; and if some model's EOG token is not correctly marked by the template, it too "cannot hit the brakes". Understand these three gates and you can control generation length to the point.</p>
   </div>
</details>

<div class="card key">
   <div class="tag">✅ Key points</div>
   <ul>
     <li><span class="mono">llama-cli</span> = a <strong>command-line/interactive shell</strong> over the shared inference engine: read stdin, drive generation, stream stdout - the most direct way to get started with llama.cpp.</li>
     <li>Entry <span class="mono">main.cpp</span> (thin) -&gt; <span class="mono">cli.cpp</span>; <span class="mono">common_init</span> + <span class="mono">common_params_parse(..., LLAMA_EXAMPLE_CLI)</span> turns the command line into <span class="mono">common_params</span> (L26).</li>
     <li>The generation main loop: <span class="mono">decode</span> for logits -&gt; <span class="mono">common_sampler_sample</span> picks a token -&gt; <span class="mono">common_sampler_accept</span> feeds back -&gt; <span class="mono">common_token_to_piece</span> streams print -&gt; append back to context, round and round.</li>
     <li>Three stop conditions: <span class="mono">n_predict</span> filled, EOG token (<span class="mono">llama_vocab_is_eog</span>), or reverse prompt (antiprompt) hit in interactive mode.</li>
     <li><strong>Current state</strong>: a modern cli reuses server's engine - <span class="mono">#include "server-context.h"</span> and links <span class="mono">server-context</span>, same engine as server with a different shell (cli=command-line, server=HTTP).</li>
   </ul>
</div>

<div class="card spark">
   <div class="tag">💡 Design insight</div>
   What this cli lesson really wants to leave you is not the use of some flag, but the architectural instinct of "<strong>separate the shell from the engine</strong>". The same <span class="mono">server_context</span>, wrapped in a command-line shell, is cli; wrapped in an HTTP shell, is server - the way you interact varies endlessly, the inference core stays one and the same. This idea of "make the stable core thick and the changing shell thin" continues the same thread as L25's "stable C ABI plus free interior" and L26's "stable outward plus convenient inward": good systems always work to tell apart "what should be unified and what should differ". Think this through, and you will stop seeing Part 5's remaining tools as isolated programs, and start seeing the one engine reused beneath them all - next lesson, we take it apart head-on. One last question worth chewing on: next time you design a system yourself, how do you decide "which part to make the stable core and which the swappable shell"? cli's answer is plain and strong - sink "the part all entry methods share" (the inference logic) into the core, and leave "the part each entry method differs in" (command line vs HTTP) on the shell. This seemingly simple dividing line fits most software that must support multiple entries: a web framework's routing vs business logic, a database's protocol layer vs storage engine - the same wisdom underneath. Internalize it as your own design instinct and you walk away with not just "how to use llama-cli", but a judgment that transfers to any project.
</div>
""",
}

LESSON_28 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课的 <span class="mono">llama-cli</span> 给共享引擎套了个"命令行壳"；这一课的 <span class="mono">llama-server</span> 给同一台引擎换上"HTTP 壳"——把推理变成一个网络服务：多个用户、多条请求同时连进来，还兼容 OpenAI 的接口，现成的 SDK 改个地址就能用。
</p>
<p style="color:var(--muted);margin-top:.4rem">既然 cli 和 server 共用一台引擎（L27 已揭晓），这一课我们就正面看这台引擎在"要同时伺候很多请求"时是怎么转的。它最精彩、也最值得记住的一招，叫<strong>连续批处理</strong>（continuous batching）：一次前向，同时推进多条请求。</p>
<p style="color:var(--muted)">本课只做<strong>架构总览</strong>——把请求从进门到出门的主干道走通，把 slot 与连续批处理这两个核心概念讲清。更深的调度取舍（prefill 与 decode 如何交错、batch 容量、抢占与公平）留到第七部分的 L35 再展开。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  server 的本事可以浓缩成一句话：<strong>把"一台引擎"变成"一个能同时服务很多人的服务"</strong>。难点不在"收发 HTTP"——那是现成的；难点在"一台 GPU、一份模型权重，怎么同时伺候十几条请求还都不慢"。答案就是连续批处理：与其让请求排长队、GPU 一次只算一条，不如把多条请求的"当前这一步"打包进同一次前向，一次算完、各取所需。读懂这一点，你就抓住了 server 区别于 cli 的根本——cli 一次只伺候你一个人，server 要在同一台引擎上把"并发"这件事做漂亮。它不是把引擎复制很多份，而是让一份引擎学会"分身"同时照顾多条对话。再往深一层看，这件事之所以可能，靠的是 KV cache（L19）的隔离：每条请求有自己独立的一段 KV，互不污染，它们才能安全地共用同一次前向计算而不"串台"。所以 server 的并发，本质上是"<strong>计算共享、状态隔离</strong>"——一次前向把算力摊给所有人，而每个人的对话历史各自存好、谁也看不见谁。这两件事缺一不可：只共享不隔离会乱套，只隔离不共享就退回到一条条排队的笨办法。把这对搭配记牢，后面所有关于 slot、batch、调度的细节，都是在这两条原则上做文章；也正因如此，server 真正的难点从来不是"怎么收 HTTP 请求"，而是"怎么让一台 GPU 在严格隔离各请求状态的前提下，仍能一次前向把大家一起往前推"。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把 server 想成一家<strong>餐厅后厨</strong>：每张餐桌（slot）坐一桌客人（一条请求），厨房（引擎）只有一个。笨办法是一桌一桌做——做完第一桌才招呼第二桌，客人全在干等。连续批处理像一个<strong>会"并灶"的大厨</strong>：他把此刻所有桌"下一道菜要下的料"一起下锅、一次翻炒，再分到各桌去。灶台（GPU 的一次前向）开一次火，同时把好几桌的菜往前推进了一步。桌子的数量（<span class="mono">--parallel N</span>）决定了后厨能同时招呼几桌；而那位会并灶的大厨，就是 <span class="mono">update_slots</span>。这个类比还藏着一个容易被忽略的点：并灶之所以划算，前提是各桌的"下一道菜"恰好能同时下锅——也就是各请求都正好处在"该算下一个 token"的节拍上。真实的 server 里，有的桌还在吃前菜（prefill 一大段 prompt）、有的已在逐字上主菜（生成），大厨得把这些不同阶段的活儿巧妙编排进同一锅。这也是"连续"二字的分量：它不是把请求一次性凑齐再开火，而是每一轮都重新决定"这一锅放谁的料"——有桌吃完就撤下、新客来了就补位，灶台一刻不停地转。本课先尝到"并灶"的甜头，至于大厨具体怎么排班、怎么在"多接客"和"每桌都快"之间拿捏分寸，留到 L35。
</div>

<h2>整体架构：一条请求的旅程</h2>
<p>先把一条请求从进门到出门的主干道走一遍。一个 HTTP 请求进来后，会被包装成一个 <span class="mono">server_task</span>（一个待办任务）丢进 <span class="mono">server_queue</span>（任务队列）；调度器把它分配给一个空闲的 <span class="mono">server_slot</span>；然后 <span class="mono">update_slots</span> 这个连续批处理循环不断推进它，每出一个 token 就产出一个 <span class="mono">server_task_result</span>（可以流式地一段段发回）；最后由 HTTP 层拼成响应回给客户端。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>HTTP 层 (server-http)</h4>
    <p>收下请求、解析 JSON；最后把结果（可流式）写回客户端。</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>任务队列 (server-queue)</h4>
    <p>请求包成 server_task 进 server_queue；post() 投递、recv() 取出，调度给空闲 slot。</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>引擎 + slots (server-context)</h4>
    <p>server_context 持有一组 server_slot；update_slots 连续批处理循环推进所有活跃 slot。</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>结果 (server_task_result)</h4>
    <p>每步产出一个结果，流式回传；server-chat 负责 OpenAI 兼容的格式转换。</p>
  </div></div>
</div>
<p>这套<strong>模块化</strong>是 server 好读的关键：<span class="mono">server-http</span> 只管网络、<span class="mono">server-queue</span> 只管排队、<span class="mono">server-context</span> 只管推理、<span class="mono">server-chat</span> 只管 OpenAI 兼容转换。各司其职，互不纠缠——你想看"请求怎么排队"就翻 queue，想看"怎么生成"就翻 context，不必在一坨大文件里大海捞针。这种切分也呼应了 L27：cli 复用的正是中间这块 <span class="mono">server_context</span> 引擎。读 server 源码时，这张模块地图就是你的导航：迷路了就回来看一眼，先定位"我此刻关心的是网络、排队、推理还是兼容"，再钻进对应的那个文件，而不必把整座 server 一口气从头啃到尾。</p>

<h2>slot 是什么</h2>
<p>slot 是理解 server 的第一块基石。启动时 <span class="mono">--parallel N</span> 会开 <span class="mono">N</span> 个 slot（源码里叫 <span class="mono">n_parallel</span>），每个 slot 就是一条<strong>独立的并行序列</strong>：有自己的 <span class="mono">seq_id</span>、自己的一块 KV 区（呼应 L19 的 KV cache），还有一个小小的<strong>状态机</strong>。总上下文会切给各 slot，每个分到的那份叫 <span class="mono">n_ctx_slot</span>。</p>
<div class="flow">
  <div class="node"><div class="nt">IDLE</div><div class="nd">空闲, 可接新请求</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">STARTED</div><div class="nd">分到任务</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">PROCESSING_PROMPT</div><div class="nd">吃 prompt (prefill)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">GENERATING</div><div class="nd">逐 token 生成</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">IDLE</div><div class="nd">完成, 放回池子</div></div>
</div>
<p>一个 slot 的一生就是这个圈：空闲（<span class="mono">IDLE</span>）时待命；接到任务后进入 <span class="mono">STARTED</span>，开始吃 prompt（<span class="mono">PROCESSING_PROMPT</span>，也就是 prefill）；prompt 吃完（<span class="mono">DONE_PROMPT</span>）就转入 <span class="mono">GENERATING</span> 一个一个吐 token；生成完毕（撞上结束符或长度上限）就回到 <span class="mono">IDLE</span>，等下一个请求。N 个 slot 各自独立地在这个圈里转，互不干扰——这就是 server 能"同时开很多条对话"的底子。每个 slot 都有自己的 KV，所以这条对话的上下文不会和那条串味。还有一点值得记：正因为 slot 数是固定的，server 启动时就把这 N 份 KV 区一次性划好，运行时不再频繁向显存申请、释放——既快又稳，但也意味着 N 一旦定下，能同时跑几条就定死了，想再多接就只能排队。</p>

<h2>连续批处理（核心）</h2>
<p>现在到了 server 最精彩的一招。设想 3 条请求同时在跑：有的 slot 在 prefill（吃 prompt）、有的在 decode（吐词）。笨办法是一条一条来，GPU 一次只服务一个 slot，其余干等。<strong>连续批处理</strong>反其道而行：它把当下所有活跃 slot"这一步要算的 token"统统拼进<strong>同一个</strong> <span class="mono">llama_batch</span>，一次 <span class="mono">llama_decode</span> 把它们的序列<strong>全部往前推进一步</strong>。</p>
<pre class="code"><span class="cm">// 连续批处理的核心 (精简自 server-context.cpp 的 update_slots)</span>
<span class="fn">common_batch_clear</span>(batch);
<span class="kw">for</span> (slot : slots) {                       <span class="cm">// 遍历所有活跃 slot</span>
    <span class="kw">if</span> (slot.state == GENERATING || slot.state == PROCESSING_PROMPT)
        <span class="fn">common_batch_add</span>(batch, slot.token, slot.pos, { slot.id }); <span class="cm">// 打上 seq_id=slot.id</span>
}
<span class="fn">llama_decode</span>(ctx, batch);                 <span class="cm">// 一次前向, 推进所有活跃序列</span>
<span class="kw">for</span> (slot : slots)
    slot.next = <span class="fn">common_sampler_sample</span>(slot.smpl, ctx, slot.i_logits); <span class="cm">// 各取自己那行</span></pre>
<p>关键就在那句 <span class="mono">common_batch_add(batch, token, pos, { slot.id })</span>：它给每个 token 打上"我属于哪个 slot"的 <span class="mono">seq_id</span> 标签。于是一个 batch 里混着好几个 slot 的 token，<span class="mono">llama_decode</span> 借助注意力掩码让每条序列只看自己的历史，算完后每个 slot 再按自己的行号取走那一行 logits 去采样。下面这张图把"某一步"定格下来看：3 个 slot 的 token 如何挤进同一个 batch、一次 decode 后又如何各自分到下一个 token。看图时请特别留意中间那个"合并 batch"框：它不是按请求分成三段，而是把三个 slot 的 token 真正<strong>混</strong>在一格一格里，只靠每格标注的 <span class="mono">seq</span> 区分归属——正是这种"混装一锅、靠标签认人"的做法，让一次前向同时喂进了三条请求的活儿。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次连续批处理</b>：slot0/slot2 在生成、slot1 在预填充，三者的 token 拼进同一个 batch；一次 llama_decode 后，每个 slot 各取下一个 token（数值为示意）。</div>
  <svg viewBox="0 0 640 250" width="100%" role="img" aria-label="连续批处理示例：三个 slot 的 token 合并进同一 batch，一次解码全推进">
<g font-family="ui-monospace,monospace">
<text x="84" y="30" text-anchor="middle" fill="#5b6470" font-size="11">3 个 slot = 3 条序列</text>
<text x="320" y="30" text-anchor="middle" fill="#5b6470" font-size="11">合并 batch（按 slot 上色）</text>
<text x="595" y="30" text-anchor="middle" fill="#5b6470" font-size="11">各取下一 token</text>
<rect x="8" y="48" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="48" width="6" height="40" rx="3" fill="#c2630e"/>
<text x="24" y="65" fill="#1d2129" font-weight="700" font-size="12">slot0</text>
<rect x="74" y="54" width="78" height="16" rx="8" fill="#c2630e"/><text x="113" y="66" text-anchor="middle" fill="#ffffff" font-size="10">生成</text>
<text x="24" y="81" fill="#5b6470" font-size="10">seq=0 pos=41</text>
<rect x="8" y="100" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="100" width="6" height="40" rx="3" fill="#2563eb"/>
<text x="24" y="117" fill="#1d2129" font-weight="700" font-size="12">slot1</text>
<rect x="74" y="106" width="78" height="16" rx="8" fill="#2563eb"/><text x="113" y="118" text-anchor="middle" fill="#ffffff" font-size="10">预填充</text>
<text x="24" y="133" fill="#5b6470" font-size="10">seq=1 pos=0</text>
<rect x="8" y="152" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="152" width="6" height="40" rx="3" fill="#7c3aed"/>
<text x="24" y="169" fill="#1d2129" font-weight="700" font-size="12">slot2</text>
<rect x="74" y="158" width="78" height="16" rx="8" fill="#7c3aed"/><text x="113" y="170" text-anchor="middle" fill="#ffffff" font-size="10">生成</text>
<text x="24" y="185" fill="#5b6470" font-size="10">seq=2 pos=57</text>
<line x1="160" y1="68" x2="207" y2="92" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 96 L 205 96 L 209 89 z" fill="#9aa6b2"/>
<line x1="160" y1="120" x2="206" y2="118" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 118 L 206 122 L 206 114 z" fill="#9aa6b2"/>
<line x1="160" y1="172" x2="207" y2="144" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 140 L 209 148 L 205 141 z" fill="#9aa6b2"/>
<rect x="214" y="44" width="212" height="120" rx="8" fill="#ffffff" stroke="#cdd5df"/>
<rect x="226" y="84" width="32" height="38" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="242" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="242" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s0</text>
<rect x="264" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="280" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="280" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="302" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="318" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="318" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="340" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="356" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="356" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="378" y="84" width="32" height="38" rx="5" fill="#7c3aed" stroke="#7c3aed"/><text x="394" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="394" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s2</text>
<text x="320" y="140" text-anchor="middle" fill="#5b6470" font-size="10">prefill 放多个，生成各放 1 个</text>
<text x="320" y="180" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">一次 llama_decode(batch)</text>
<line x1="320" y1="164" x2="320" y2="166" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 320 174 L 316 166 L 324 166 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="550" y2="68" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 66 L 551 72 L 549 64 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="550" y2="117" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 118 L 550 121 L 550 113 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="551" y2="166" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 170 L 549 170 L 553 163 z" fill="#9aa6b2"/>
<rect x="558" y="51" width="74" height="34" rx="5" fill="#ffffff" stroke="#c2630e"/>
<text x="595" y="66" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="11">下一 tok</text>
<text x="595" y="79" text-anchor="middle" fill="#5b6470" font-size="9">slot0</text>
<rect x="558" y="103" width="74" height="34" rx="5" fill="#ffffff" stroke="#2563eb"/>
<text x="595" y="118" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="11">下一 tok</text>
<text x="595" y="131" text-anchor="middle" fill="#5b6470" font-size="9">slot1</text>
<rect x="558" y="155" width="74" height="34" rx="5" fill="#ffffff" stroke="#7c3aed"/>
<text x="595" y="170" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">下一 tok</text>
<text x="595" y="183" text-anchor="middle" fill="#5b6470" font-size="9">slot2</text>
<text x="320" y="234" text-anchor="middle" fill="#5b6470" font-size="11">一次前向，多请求共享</text>
</g></svg>
</div>

<h2>OpenAI 兼容</h2>
<p>server 还有一个让它格外好用的特性：它直接长出了一套和 <strong>OpenAI API 一样</strong>的端点，比如 <span class="mono">/v1/chat/completions</span>。这意味着任何为 OpenAI 写的客户端、SDK、前端，只要把请求地址（base URL）指向你的 llama-server，就能直接连上本地模型，几乎不用改代码。负责这层"翻译"的是 <span class="mono">server-chat</span>：它在 OpenAI 的 JSON schema（messages、tools 等）和引擎内部表示之间来回转换，连工具调用（tool calls）也照顾到了。这一层翻译看似不起眼，却是 server 能融入现有生态的关键——你已有的工具链、监控面板、客户端代码，几乎都能不改一行就复用，把"换成本地模型"的迁移成本压到了最低。</p>
<pre class="code"><span class="cm"># 起一个服务, 然后像调 OpenAI 一样调它</span>
llama-server -m model.gguf --port 8080   <span class="cm"># 默认开 N 个 slot (--parallel)</span>

curl http://localhost:8080/v1/chat/completions \
  -H <span class="st">"Content-Type: application/json"</span> \
  -d <span class="st">'{"messages":[{"role":"user","content":"你好"}]}'</span></pre>
<div class="card spark">
  <div class="tag">💡 动手试试</div>
  最能体会 server 价值的一步：<span class="mono">llama-server -m model.gguf --port 8080</span> 起服务，然后开几个终端同时 <span class="mono">curl /v1/chat/completions</span>——你会看到它们<strong>同时</strong>都有响应，而不是排队一个个来，这就是连续批处理在背后并灶。再把 <span class="mono">--parallel</span> 调大或调小，观察"能同时伺候几条请求"怎么变。因为端点是 OpenAI 兼容的，你甚至可以拿现成的 OpenAI Python SDK，把 <span class="mono">base_url</span> 指到 <span class="mono">http://localhost:8080/v1</span> 直接用——本地模型，云端 API 的手感。想更直观地看见连续批处理的威力，可以做个对比实验：先用 <span class="mono">--parallel 1</span> 起服务（只有一个 slot），同时发 4 条请求，你会看到它们基本排队、一条接一条地出字；再用 <span class="mono">--parallel 4</span> 起、同样发 4 条，这次它们几乎一起开始、一起往外蹦字。同一台机器、同一个模型，只因为多开了几个 slot、让引擎"并灶"，整体观感就天差地别。这个小实验最能把"批处理换吞吐"从一句口号，变成你亲眼见过的事实；再进一步，一边压测一边盯着 server 打印的吞吐日志，你还能亲手找到那个"再加 slot 也不见更快"的拐点——那就是这张卡的算力上限。
</div>

<h2>为什么不直接开很多进程？</h2>
<p>读到这里，你可能会冒出一个很自然的问题：要同时服务很多请求，为什么不干脆开很多个 llama 进程，一个进程伺候一条请求，岂不简单？答案藏在<strong>显存</strong>里。模型权重动辄几个 GB 到几十 GB，每开一个独立进程，就要把这份权重<strong>再加载一份</strong>进显存——开 8 个进程，同一份权重就被占了 8 遍，普通显卡根本扛不住。</p>
<p>server 的做法恰恰相反：<strong>一份权重，多条会话</strong>。模型只加载一次（呼应 L25 的 <span class="mono">llama_model</span> 只读、可共享），所有 slot 共用这同一份权重，各自只额外占一小块 KV（L19）。于是显存开销从"权重 × 进程数"变成了"权重 × 1 + KV × slot 数"——而一块 KV 比整份权重小得多。这就是为什么"单引擎多 slot"能在一张卡上塞下远比"多进程"更多的并发。</p>
<p>更何况多进程还各算各的前向，享受不到连续批处理那"一次前向、服务多人"的吞吐红利。所以"单引擎 + 多 slot + 连续批处理"这套组合，不是为了写起来优雅，而是被显存和吞吐这两条硬约束逼出来的最优解：省显存（共享权重）又高吞吐（共享前向），一举两得。当然多进程也有它的好处——隔离更彻底（一个崩了不连累别的）、部署更省心；但在"一张卡尽量多伺候几条请求"这个最常见的目标下，单引擎多 slot 几乎总是更划算，这也是主流推理服务器（不只 llama.cpp）几乎都选这条路的原因。</p>

<h2>深入：排队与吞吐</h2>
<p>最后两个折叠，回答两个一问就到点子上的问题：slot 不够用了会怎样，以及连续批处理为什么能把吞吐做高。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> slot 占满了，新请求怎么办？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>slot 的数量是固定的（<span class="mono">--parallel N</span>）。如果 N 个 slot 全在忙，又来了新请求，它不会被丢弃，而是<strong>排队等待</strong>：留在 <span class="mono">server_queue</span> 里（实现上有一条"延后任务"的队列），直到某个 slot 生成完、回到 <span class="mono">IDLE</span>，调度器再把它取出来分配进去。所以 <span class="mono">--parallel</span> 这个数字是一种取舍：调大，能同时接的请求更多，但每条分到的 KV 上下文（<span class="mono">n_ctx_slot</span>）和算力被摊薄；调小则相反。怎么权衡要看你的显存和负载——这正是 L35 要细讲的调度话题。一个实用的经验法则是：先按"单条请求大概要多长上下文"估出每个 slot 至少要留多少 KV，再拿剩余显存除以它，得到的大致就是 N 的上限——超过这个数，要么爆显存，要么每条分到的上下文被压得不够用。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 连续批处理为什么比"逐请求"快？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>关键在于 GPU 的特性：它做一次大矩阵运算（一次前向）的开销，<strong>几乎</strong>不随"同时算几条序列"线性增长——一次算 1 条和一次算 8 条，耗时差得远没有 8 倍。所以把多条请求的 token 合进一次 <span class="mono">llama_decode</span>，等于让同一次前向的成本摊给了好几条请求，单位时间能吐出的 token 总量（吞吐）大大提高。代价是实现复杂了（要管 seq_id、注意力掩码、各 slot 的进度），还有一些公平与延迟的权衡——但"一次前向、服务多人"这个核心收益，足以让它成为现代推理服务的标配。更深的取舍留给 L35。顺带一提，这也解释了"延迟"和"吞吐"为什么常是一对冤家：连续批处理拉高了整体吞吐，却可能让单条请求因为要和别人挤同一次前向而稍稍变慢——到底偏向哪头，取决于你是想让一个人尽快拿到答案，还是想让一整批人平均都不等太久。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama-server</span> 给共享引擎套"HTTP 壳"，把推理变成<strong>网络服务</strong>：多请求并发 + OpenAI 兼容端点。</li>
    <li>请求旅程：HTTP -&gt; <span class="mono">server_task</span> -&gt; <span class="mono">server_queue</span> -&gt; 空闲 <span class="mono">server_slot</span> -&gt; <span class="mono">update_slots</span> -&gt; <span class="mono">server_task_result</span> -&gt; 响应；模块化分层（http/queue/context/chat）。</li>
    <li><span class="mono">slot</span>：<span class="mono">--parallel N</span> 开 N 条并行序列，各有 seq_id + KV + 状态机（IDLE -&gt; STARTED -&gt; PROCESSING_PROMPT -&gt; GENERATING -&gt; IDLE）。</li>
    <li><strong>连续批处理</strong>（核心）：<span class="mono">common_batch_add(..., {slot.id})</span> 把多个 slot 的 token 拼进同一 batch，一次 <span class="mono">llama_decode</span> 推进所有活跃序列——一次前向、服务多人。</li>
    <li>OpenAI 兼容：<span class="mono">server-chat</span> 转换 <span class="mono">/v1/chat/completions</span> 等端点，现成 OpenAI 客户端改 base URL 即可连。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  server 这一课的精华，是"<strong>批处理换吞吐</strong>"这个朴素而深刻的工程思想。单看一条请求，连续批处理并不会让它更快（延迟没变小）；但把视角放大到"整台服务器单位时间能服务多少 token"，它就是数量级的提升。这背后是对硬件的深刻顺应——GPU 擅长"一次算一大批"，那就把请求攒成批喂给它，而不是逼它一条条算。从 cli 到 server，你看到的是同一台引擎的两种用法：cli 追求"一个人用得顺手"，server 追求"一群人用得高效"。而把"一群人"伺候好的关键，从来不是把引擎复制很多份，而是让一份引擎学会"并灶"。把这个思想记住，等到 L35 拆解更细的调度时，你会发现一切取舍都围着它转。再把它放进更大的图景：从 L25 的 C API、L26 的 common、L27 的 cli，到这一课的 server，第五部分其实在反复讲同一件事——如何把"一份稳定的核"用在越来越复杂的场景里。cli 让一个人用得顺，server 让一群人用得起，而支撑这一切的，始终是前四部分打磨出来的那台引擎。当你下次面对"如何让一个系统服务更多用户"时，不妨先问一句：我的"那一次前向"是什么？能不能把多个请求的它合并起来一次做完？它的反面也值得警惕——如果一个系统天然无法批处理、每个请求都得独占资源跑到底，那它的扩展性从架构之初就被卡死了。能不能批，往往一开始就决定了上限。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson's <span class="mono">llama-cli</span> wrapped the shared engine in a "command-line shell"; this lesson's <span class="mono">llama-server</span> fits the same engine with an "HTTP shell" - turning inference into a network service: many users and many requests connect at once, and it speaks the OpenAI API, so an existing SDK just needs a new address.
</p>
<p style="color:var(--muted);margin-top:.4rem">Since cli and server share one engine (revealed in L27), this lesson looks head-on at how that engine runs when it must serve many requests at once. Its most brilliant and most memorable move is <strong>continuous batching</strong>: one forward pass advances many requests together.</p>
<p style="color:var(--muted)">This lesson is an <strong>architecture overview</strong> only - walk the main road a request takes from door to door, and make the two core concepts, slots and continuous batching, clear. The deeper scheduling trade-offs (how prefill and decode interleave, batch capacity, preemption and fairness) wait for L35 in Part 7.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  server's skill boils down to one line: <strong>turn "one engine" into "a service that serves many people at once"</strong>. The hard part is not "send and receive HTTP" - that is off the shelf; the hard part is "one GPU, one copy of the model weights, serving a dozen requests at once without any of them crawling". The answer is continuous batching: rather than queue requests and let the GPU compute one at a time, pack each request's "current step" into the same forward pass, compute once, and hand each its share. Grasp this and you have server's essential difference from cli - cli serves only you, while server must do "concurrency" well on the very same engine. It does not copy the engine many times; it teaches one engine to "split itself" across many conversations. Look one level deeper and this is possible thanks to KV-cache (L19) isolation: each request has its own slice of KV, uncontaminated, so they can safely share the same forward computation without "crossing wires". So server's concurrency is essentially "<strong>shared compute, isolated state</strong>" - one forward spreads the compute across everyone, while each conversation's history is stored apart, invisible to the others. Neither half can be dropped: share without isolation and it descends into chaos; isolate without sharing and you are back to the clumsy one-at-a-time queue. Hold this pairing and every later detail about slots, batches, and scheduling is just elaboration on these two principles; that is also why server's real hard part is never "how to receive HTTP requests", but "how to let one GPU, while strictly isolating each request's state, still advance everyone together in one forward".
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of server as a restaurant <strong>kitchen</strong>: each table (slot) seats one party (one request), and there is only one kitchen (the engine). The clumsy way is table by table - finish table one before greeting table two, everyone else waiting. Continuous batching is like a chef who can <strong>cook several woks at once</strong>: he throws "the next ingredient every table needs right now" into one pan, stir-fries once, and splits it out to each table. The stove (one GPU forward pass) lights once and advances several tables' dishes by a step. The number of tables (<span class="mono">--parallel N</span>) sets how many parties the kitchen serves at once; and that multi-wok chef is <span class="mono">update_slots</span>. The analogy hides a point easy to miss: cooking many woks pays off only if each table's "next dish" can go into the pan at the same moment - that is, each request is right on the beat of "compute the next token". In a real server, some tables are still on appetizers (prefilling a long prompt) and some already plating mains token by token (generating), and the chef must weave these different stages into one pan. That is the weight of the word "continuous": it does not gather all requests before lighting the fire, but every round re-decides "whose ingredients go in this pan" - clearing a table that finished and seating a newcomer, the stove turning without pause. This lesson tastes the sweetness of "many woks"; how the chef actually schedules and weighs "more guests" against "every table fast" is left to L35.
</div>

<h2>Overall architecture: a request's journey</h2>
<p>First walk a request's main road from door to door. An incoming HTTP request is wrapped into a <span class="mono">server_task</span> (a to-do) and dropped into the <span class="mono">server_queue</span> (a task queue); the scheduler assigns it to an idle <span class="mono">server_slot</span>; then the <span class="mono">update_slots</span> continuous-batching loop keeps advancing it, producing a <span class="mono">server_task_result</span> per token (streamable in pieces); finally the HTTP layer assembles the response back to the client.</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>HTTP layer (server-http)</h4>
    <p>Take the request, parse JSON; at the end write the result (streamable) back to the client.</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>Task queue (server-queue)</h4>
    <p>The request becomes a server_task in server_queue; post() submits, recv() takes out, scheduled to an idle slot.</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>Engine + slots (server-context)</h4>
    <p>server_context holds a set of server_slot; update_slots advances all active slots by continuous batching.</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>Result (server_task_result)</h4>
    <p>Each step yields a result, streamed back; server-chat handles OpenAI-compatible format conversion.</p>
  </div></div>
</div>
<p>This <strong>modularity</strong> is why server reads well: <span class="mono">server-http</span> minds only the network, <span class="mono">server-queue</span> only the queue, <span class="mono">server-context</span> only inference, <span class="mono">server-chat</span> only OpenAI compatibility. Each to its own, untangled - want to see "how requests queue" open queue, want "how it generates" open context, no needle-in-a-haystack in one giant file. This split also echoes L27: what cli reuses is exactly that middle <span class="mono">server_context</span> engine. When you read server's source, this module map is your navigation: lost, come back and glance at it, first locate "is what I care about now the network, the queue, inference, or compatibility", then dive into the matching file, rather than gnawing through the whole server front to back in one go.</p>

<h2>What a slot is</h2>
<p>A slot is the first cornerstone for understanding server. At startup <span class="mono">--parallel N</span> opens <span class="mono">N</span> slots (called <span class="mono">n_parallel</span> in the source), and each slot is an <strong>independent parallel sequence</strong>: with its own <span class="mono">seq_id</span>, its own slice of KV (echoing L19's KV cache), and a small <strong>state machine</strong>. The total context is divided among slots, and each one's share is <span class="mono">n_ctx_slot</span>.</p>
<div class="flow">
  <div class="node"><div class="nt">IDLE</div><div class="nd">free, can take a request</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">STARTED</div><div class="nd">assigned a task</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">PROCESSING_PROMPT</div><div class="nd">eat prompt (prefill)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">GENERATING</div><div class="nd">emit token by token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">IDLE</div><div class="nd">done, back to pool</div></div>
</div>
<p>A slot's life is this circle: idle (<span class="mono">IDLE</span>) on standby; on a task it enters <span class="mono">STARTED</span> and begins eating the prompt (<span class="mono">PROCESSING_PROMPT</span>, that is prefill); once the prompt is eaten (<span class="mono">DONE_PROMPT</span>) it turns to <span class="mono">GENERATING</span> and emits tokens one by one; when generation finishes (an end token or the length cap) it returns to <span class="mono">IDLE</span> and waits for the next request. The N slots each turn this circle independently, without interfering - that is the basis for server "running many conversations at once". Each slot has its own KV, so one conversation's context never bleeds into another's. One more thing worth noting: because the slot count is fixed, server carves out these N KV regions once at startup and no longer keeps asking VRAM for memory and freeing it at runtime - fast and stable, but it also means once N is set, how many can run at once is set too, and anything beyond must queue.</p>

<h2>Continuous batching (the core)</h2>
<p>Now to server's most brilliant move. Picture 3 requests running at once: some slots are in prefill (eating the prompt), some in decode (emitting words). The clumsy way takes them one at a time, the GPU serving only one slot while the rest wait. <strong>Continuous batching</strong> does the opposite: it packs "the tokens to compute this step" from all currently active slots into the <strong>same</strong> <span class="mono">llama_batch</span>, and one <span class="mono">llama_decode</span> advances all their sequences <strong>by one step together</strong>.</p>
<pre class="code"><span class="cm">// the heart of continuous batching (condensed from update_slots in server-context.cpp)</span>
<span class="fn">common_batch_clear</span>(batch);
<span class="kw">for</span> (slot : slots) {                       <span class="cm">// iterate all active slots</span>
    <span class="kw">if</span> (slot.state == GENERATING || slot.state == PROCESSING_PROMPT)
        <span class="fn">common_batch_add</span>(batch, slot.token, slot.pos, { slot.id }); <span class="cm">// tag with seq_id=slot.id</span>
}
<span class="fn">llama_decode</span>(ctx, batch);                 <span class="cm">// one forward, advance all active sequences</span>
<span class="kw">for</span> (slot : slots)
    slot.next = <span class="fn">common_sampler_sample</span>(slot.smpl, ctx, slot.i_logits); <span class="cm">// each reads its own row</span></pre>
<p>The key is that line <span class="mono">common_batch_add(batch, token, pos, { slot.id })</span>: it tags each token with "which slot I belong to" as a <span class="mono">seq_id</span>. So one batch holds tokens from several slots, <span class="mono">llama_decode</span> uses the attention mask to let each sequence see only its own history, and afterward each slot reads its own row of logits to sample. The diagram below freezes "one step": how 3 slots' tokens squeeze into one batch, and how after one decode each gets its own next token. When you read it, look closely at the middle "merged batch" box: it is not split into three request-segments, but truly <strong>mixes</strong> the three slots' tokens cell by cell, telling them apart only by the <span class="mono">seq</span> tag on each cell - it is exactly this "mix into one pan, recognize by tag" that lets one forward feed in the work of three requests at once.</p>
<div class="trace">
  <div class="tcap"><b>Tracing one continuous-batch step</b>: slot0/slot2 are generating, slot1 is prefilling; their tokens pack into one batch, and after one llama_decode each slot gets its next token (values are illustrative).</div>
  <svg viewBox="0 0 640 250" width="100%" role="img" aria-label="continuous batching example: tokens from three slots merged into one batch, one decode advances all">
<g font-family="ui-monospace,monospace">
<text x="84" y="30" text-anchor="middle" fill="#5b6470" font-size="11">3 slots = 3 seqs</text>
<text x="320" y="30" text-anchor="middle" fill="#5b6470" font-size="11">merged batch (colored by slot)</text>
<text x="595" y="30" text-anchor="middle" fill="#5b6470" font-size="11">next tokens</text>
<rect x="8" y="48" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="48" width="6" height="40" rx="3" fill="#c2630e"/>
<text x="24" y="65" fill="#1d2129" font-weight="700" font-size="12">slot0</text>
<rect x="74" y="54" width="78" height="16" rx="8" fill="#c2630e"/><text x="113" y="66" text-anchor="middle" fill="#ffffff" font-size="10">gen</text>
<text x="24" y="81" fill="#5b6470" font-size="10">seq=0 pos=41</text>
<rect x="8" y="100" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="100" width="6" height="40" rx="3" fill="#2563eb"/>
<text x="24" y="117" fill="#1d2129" font-weight="700" font-size="12">slot1</text>
<rect x="74" y="106" width="78" height="16" rx="8" fill="#2563eb"/><text x="113" y="118" text-anchor="middle" fill="#ffffff" font-size="10">prefill</text>
<text x="24" y="133" fill="#5b6470" font-size="10">seq=1 pos=0</text>
<rect x="8" y="152" width="152" height="40" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<rect x="8" y="152" width="6" height="40" rx="3" fill="#7c3aed"/>
<text x="24" y="169" fill="#1d2129" font-weight="700" font-size="12">slot2</text>
<rect x="74" y="158" width="78" height="16" rx="8" fill="#7c3aed"/><text x="113" y="170" text-anchor="middle" fill="#ffffff" font-size="10">gen</text>
<text x="24" y="185" fill="#5b6470" font-size="10">seq=2 pos=57</text>
<line x1="160" y1="68" x2="207" y2="92" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 96 L 205 96 L 209 89 z" fill="#9aa6b2"/>
<line x1="160" y1="120" x2="206" y2="118" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 118 L 206 122 L 206 114 z" fill="#9aa6b2"/>
<line x1="160" y1="172" x2="207" y2="144" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 214 140 L 209 148 L 205 141 z" fill="#9aa6b2"/>
<rect x="214" y="44" width="212" height="120" rx="8" fill="#ffffff" stroke="#cdd5df"/>
<rect x="226" y="84" width="32" height="38" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="242" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="242" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s0</text>
<rect x="264" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="280" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="280" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="302" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="318" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="318" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="340" y="84" width="32" height="38" rx="5" fill="#2563eb" stroke="#2563eb"/><text x="356" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="356" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s1</text>
<rect x="378" y="84" width="32" height="38" rx="5" fill="#7c3aed" stroke="#7c3aed"/><text x="394" y="100" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="11">tok</text><text x="394" y="115" text-anchor="middle" fill="#ffffff" font-size="9">s2</text>
<text x="320" y="140" text-anchor="middle" fill="#5b6470" font-size="10">prefill adds many, each gen adds 1</text>
<text x="320" y="180" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">one llama_decode(batch)</text>
<line x1="320" y1="164" x2="320" y2="166" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 320 174 L 316 166 L 324 166 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="550" y2="68" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 66 L 551 72 L 549 64 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="550" y2="117" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 118 L 550 121 L 550 113 z" fill="#9aa6b2"/>
<line x1="426" y1="104" x2="551" y2="166" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 558 170 L 549 170 L 553 163 z" fill="#9aa6b2"/>
<rect x="558" y="51" width="74" height="34" rx="5" fill="#ffffff" stroke="#c2630e"/>
<text x="595" y="66" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="11">next tok</text>
<text x="595" y="79" text-anchor="middle" fill="#5b6470" font-size="9">slot0</text>
<rect x="558" y="103" width="74" height="34" rx="5" fill="#ffffff" stroke="#2563eb"/>
<text x="595" y="118" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="11">next tok</text>
<text x="595" y="131" text-anchor="middle" fill="#5b6470" font-size="9">slot1</text>
<rect x="558" y="155" width="74" height="34" rx="5" fill="#ffffff" stroke="#7c3aed"/>
<text x="595" y="170" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">next tok</text>
<text x="595" y="183" text-anchor="middle" fill="#5b6470" font-size="9">slot2</text>
<text x="320" y="234" text-anchor="middle" fill="#5b6470" font-size="11">one forward pass, shared by all requests</text>
</g></svg>
</div>

<h2>OpenAI compatibility</h2>
<p>server has one more feature that makes it especially handy: it grows a set of endpoints <strong>identical to the OpenAI API</strong>, such as <span class="mono">/v1/chat/completions</span>. This means any client, SDK, or front-end written for OpenAI only needs to point its request URL (base URL) at your llama-server to talk to a local model, with almost no code change. The layer doing this "translation" is <span class="mono">server-chat</span>: it converts back and forth between OpenAI's JSON schema (messages, tools, and so on) and the engine's internal representation, tool calls included. This translation layer looks humble but is the key to server fitting into the existing ecosystem - your existing tool chains, monitoring dashboards, and client code can almost all be reused without changing a line, cutting the cost of "switching to a local model" to a minimum.</p>
<pre class="code"><span class="cm"># start a service, then call it like OpenAI</span>
llama-server -m model.gguf --port 8080   <span class="cm"># opens N slots by default (--parallel)</span>

curl http://localhost:8080/v1/chat/completions \
  -H <span class="st">"Content-Type: application/json"</span> \
  -d <span class="st">'{"messages":[{"role":"user","content":"Hello"}]}'</span></pre>
<div class="card spark">
  <div class="tag">💡 Hands-on</div>
  The step that best conveys server's value: <span class="mono">llama-server -m model.gguf --port 8080</span> to start the service, then open a few terminals and <span class="mono">curl /v1/chat/completions</span> at the same time - you will see them all respond <strong>concurrently</strong>, not one at a time in a queue; that is continuous batching cooking several woks behind the scenes. Then turn <span class="mono">--parallel</span> up or down and watch "how many requests it serves at once" change. Because the endpoints are OpenAI-compatible, you can even take the off-the-shelf OpenAI Python SDK, point <span class="mono">base_url</span> at <span class="mono">http://localhost:8080/v1</span>, and use it directly - a local model with the feel of a cloud API. To see continuous batching's power directly, run a comparison: first start with <span class="mono">--parallel 1</span> (one slot) and send 4 requests at once - they basically queue, emitting one after another; then start with <span class="mono">--parallel 4</span> and send 4 again - this time they begin together and spill out words together. Same machine, same model, yet just opening a few slots and letting the engine "cook many woks" makes the overall feel night and day. This small experiment best turns "batching for throughput" from a slogan into a fact seen with your own eyes; go further, watch the throughput logs while you load-test, and you can find by hand the knee where "more slots no longer means faster" - that is this card's compute ceiling.
</div>

<h2>Why not just run many processes?</h2>
<p>By here a natural question may surface: to serve many requests at once, why not simply run many llama processes, one process per request - is that not simpler? The answer hides in <strong>VRAM</strong>. Model weights run from a few GB to tens of GB, and each independent process must load <strong>another copy</strong> of those weights into VRAM - run 8 processes and the same weights are paid for 8 times, which an ordinary GPU cannot bear.</p>
<p>server does the opposite: <strong>one set of weights, many sessions</strong>. The model loads once (echoing L25's read-only, shareable <span class="mono">llama_model</span>), all slots share these same weights, and each only takes one extra small slice of KV (L19). So VRAM cost goes from "weights x process count" to "weights x 1 + KV x slot count" - and one KV slice is far smaller than a full set of weights. That is why "one engine, many slots" fits far more concurrency on a single card than "many processes".</p>
<p>Besides, many processes each run their own forward pass and miss continuous batching's "one forward, serve many" throughput dividend. So the combo of "one engine + many slots + continuous batching" is not for elegance, but the optimum forced by two hard constraints, VRAM and throughput: save VRAM (shared weights) and high throughput (shared forward), two wins at once. Many processes do have merits - cleaner isolation (one crash spares the rest), simpler deployment; but under the most common goal of "serve as many requests as possible on one card", one engine with many slots almost always wins, which is why mainstream inference servers (not just llama.cpp) nearly all take this road.</p>

<h2>Deep dive: queueing and throughput</h2>
<p>Two final folds answering two questions that cut to the point: what happens when slots run out, and why continuous batching makes throughput high.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Slots are full - what happens to a new request? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The number of slots is fixed (<span class="mono">--parallel N</span>). If all N slots are busy and a new request arrives, it is not dropped but <strong>queued to wait</strong>: it stays in the <span class="mono">server_queue</span> (the implementation has a "deferred tasks" queue) until some slot finishes generating and returns to <span class="mono">IDLE</span>, then the scheduler takes it out and assigns it. So <span class="mono">--parallel</span> is a trade-off: larger means more concurrent requests, but each slot's KV context (<span class="mono">n_ctx_slot</span>) and compute are thinner; smaller is the opposite. How to weigh it depends on your VRAM and load - exactly the scheduling topic L35 covers in detail. A handy rule of thumb: estimate from "how much context one request needs" how much KV each slot must reserve, then divide remaining VRAM by it, and that is roughly the ceiling for N - beyond it you either blow VRAM or squeeze each one's context too thin.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Why is continuous batching faster than "per request"? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The key is a property of the GPU: the cost of one big matrix op (one forward pass) <strong>barely</strong> grows with "how many sequences computed at once" - one sequence versus eight is nowhere near 8x the time. So merging many requests' tokens into one <span class="mono">llama_decode</span> spreads the cost of that single forward over several requests, and the total tokens produced per unit time (throughput) rises sharply. The price is more complex code (managing seq_ids, the attention mask, each slot's progress), plus some fairness and latency trade-offs - but the core gain of "one forward, serve many" is enough to make it standard for modern inference services. The deeper trade-offs are left to L35. By the way, this also explains why "latency" and "throughput" are often at odds: continuous batching lifts overall throughput, yet may make a single request slightly slower for having to share one forward with others - which way you lean depends on whether you want one person to get an answer as fast as possible, or a whole batch of people to each wait not too long on average.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama-server</span> fits the shared engine with an "HTTP shell", turning inference into a <strong>network service</strong>: concurrent requests + OpenAI-compatible endpoints.</li>
    <li>Request journey: HTTP -&gt; <span class="mono">server_task</span> -&gt; <span class="mono">server_queue</span> -&gt; idle <span class="mono">server_slot</span> -&gt; <span class="mono">update_slots</span> -&gt; <span class="mono">server_task_result</span> -&gt; response; modular layering (http/queue/context/chat).</li>
    <li><span class="mono">slot</span>: <span class="mono">--parallel N</span> opens N parallel sequences, each with a seq_id + KV + state machine (IDLE -&gt; STARTED -&gt; PROCESSING_PROMPT -&gt; GENERATING -&gt; IDLE).</li>
    <li><strong>Continuous batching</strong> (core): <span class="mono">common_batch_add(..., {slot.id})</span> packs many slots' tokens into one batch, one <span class="mono">llama_decode</span> advances all active sequences - one forward, serve many.</li>
    <li>OpenAI compatibility: <span class="mono">server-chat</span> converts <span class="mono">/v1/chat/completions</span> and other endpoints; an existing OpenAI client just changes the base URL.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  the essence of this server lesson is the plain yet profound engineering idea of "<strong>batching for throughput</strong>". For a single request, continuous batching does not make it faster (latency is unchanged); but zoom out to "how many tokens the whole server produces per unit time" and it is an order-of-magnitude gain. Behind it is a deep deference to the hardware - the GPU excels at "computing one big batch at once", so gather requests into batches and feed it, instead of forcing it to compute one by one. From cli to server you see two uses of the same engine: cli chases "smooth for one person", server chases "efficient for a crowd". And the key to serving "a crowd" well is never to copy the engine many times, but to teach one engine to "cook many woks". Hold this idea, and when L35 dissects finer scheduling, you will find every trade-off revolves around it. Place it in a bigger picture: from L25's C API, L26's common, L27's cli, to this lesson's server, Part 5 keeps telling one story - how to use "one stable core" in ever more complex settings. cli makes it smooth for one person, server makes it affordable for a crowd, and underneath it all is the engine honed across the first four parts. Next time you face "how to make a system serve more users", first ask: what is my "one forward pass"? Can I merge many requests' version of it into one shot? Its inverse is worth heeding too - if a system inherently cannot batch and each request must hog resources to the end, its scalability is capped from the very birth of the architecture. Whether you can batch often sets the ceiling from the start.
</div>
""",
}

LESSON_29 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前面的 L6 和 L12 已经讲透了量化的"原理"——为什么几个比特就能近似一个浮点数、各种格式的字节又是怎么排布的。这一课换个角度，讲"<strong>怎么用工具真把模型压小</strong>"：一行 <span class="mono">llama-quantize</span> 命令，就能把一个十几 GB 的 fp16 模型变成三五 GB 的 Q4_K_M；再配上 imatrix（重要性矩阵），同样的比特数还能把掉下去的质量再拉回来一截。
</p>
<p style="color:var(--muted);margin-top:.4rem">换句话说，L6/L12 是"懂原理"，这一课是"会操作"：知道每个旗标在调什么、不同档位是怎么取舍体积与质量的、以及 imatrix 这把"质量回血"的钥匙到底怎么用。量化是让大模型能在普通显卡、甚至纯 CPU 上跑起来的关键一步，而这一课就是教你亲手把它完成。</p>
<p style="color:var(--muted)">这一课的两个主角是 <span class="mono">tools/quantize</span>（压缩工具本体）和 <span class="mono">tools/imatrix</span>（生成重要性矩阵的配套工具）。我们先看怎么用 quantize 一键压缩、它背后调的是哪个公共 API，再看 imatrix 凭什么能在不加比特的前提下把质量做得更好。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  量化工具干的事，本质上是一次"<strong>有损压缩</strong>"：把每个权重从 16 位浮点，换成 4 位、5 位这种更省地方的表示。省下来的是实打实的显存和带宽——模型小一半，加载快一倍，能塞进的显卡也更便宜。代价是精度的损失，但这个损失是可控的：档位（ftype）让你在"压多狠"和"留多少质量"之间自由选点，而 imatrix 则像一个聪明的预算分配器，把有限的比特优先花在最重要的权重上。读懂这一课，你就握住了"把模型搬到自己机器上"最常用的那把工具——绝大多数你在网上下到的 GGUF 量化模型，都是这一步的产物。换个角度说，这一课把 L6/L12 学到的"原理"真正变成了你手上能用的"手艺"。而且这门手艺门槛很低：你不必懂量化算法内部的数学，只要会调几个旗标、知道各档位的取舍，就能压出一个能用的模型——真正的复杂度都被 llama-quantize 这个工具和它背后的公共 API 包圆了，你站在现成的肩膀上即可。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把量化想成<strong>压缩一张照片</strong>：原图（fp16）清晰但很大，存成 JPEG（量化）会小很多，但画质有损。"档位"就像 JPEG 的质量滑块——拉到 90% 几乎看不出区别、文件中等，拉到 30% 就很小但开始糊。而 imatrix 更像一种"<strong>智能压缩</strong>"：它先分析这张图哪里是人脸、哪里是空白背景，然后把字节预算多留给人脸、少留给背景。同样的文件大小，重点区域更清楚——这正是 imatrix 在权重世界里做的事：把精度优先留给"被用得最多"的那些权重，把误差更多地丢给那些"无关紧要"的角落。这个类比还能再推一步：JPEG 之所以能在画质损失很小的情况下大幅压缩，靠的是"人眼对某些细节不敏感"这一先验；imatrix 异曲同工，它靠的是"模型对某些权重不敏感"这一在真实数据上测出来的先验。两者都说明一个道理：<strong>压缩从来不是均匀地砍，而是知道哪里能砍、哪里不能砍</strong>。你越懂数据里"哪些重要"，就越能在同样的体积里留住更多质量——好的量化方案，往往不是算法更花哨，而是"测得更准"。
</div>

<h2>量化工具怎么用</h2>
<p>最常见的用法只有一行：<span class="mono">llama-quantize in.gguf out.gguf Q4_K_M</span>——输入一个 fp16/fp32 的 GGUF，指定一个目标档位（这里是 Q4_K_M），它就吐出一个压缩好的小 GGUF。入口在 <span class="mono">tools/quantize/main.cpp</span>（很薄），主体逻辑在 <span class="mono">quantize.cpp</span>，而真正干活的是它调用的<strong>公共 API</strong> <span class="mono">llama_model_quantize(in, out, &amp;params)</span>——注意这是 L25 那套 <span class="mono">llama.h</span> 里的函数，所以量化能力对外也是开放的，并不只有命令行能用，你完全可以在自己的程序里调它。</p>
<pre class="code"><span class="cm">// 量化工具的核心: 一行命令背后调的公共 API (简化自 tools/quantize/quantize.cpp)</span>
llama_model_quantize_params params = <span class="fn">llama_model_quantize_default_params</span>();
params.ftype   = LLAMA_FTYPE_MOSTLY_Q4_K_M;  <span class="cm">// 目标档位</span>
params.imatrix = imatrix_data;               <span class="cm">// 可选: 喂入重要性矩阵</span>
params.dry_run = false;                       <span class="cm">// true 则只算体积, 不真压</span>
<span class="fn">llama_model_quantize</span>(<span class="st">"in.gguf"</span>, <span class="st">"out.gguf"</span>, &amp;params);</pre>
<p>那个 <span class="mono">params</span>（<span class="mono">llama_model_quantize_params</span>）藏着不少实用旋钮：<span class="mono">ftype</span> 选目标档位；<span class="mono">dry_run</span> 设成 true 就只<strong>试算</strong>压缩后多大、并不真的压（选档位时特别省事）；<span class="mono">output_tensor_type</span> / <span class="mono">token_embedding_type</span> 能给个别关键张量单独定一个更高的精度；<span class="mono">keep_split</span> 保持分片结构。换句话说，量化不是"一刀切到底"，而是可以精细到每一类张量、甚至每一层的。</p>
<p>那么"档位"到底是什么？它就是一个 <span class="mono">llama_ftype</span> 枚举值，对应一种"平均每个权重用几个比特"（bpw）的方案。<span class="mono">quantize.cpp</span> 里有一张表，把每个档位的名字、bpw、以及实测的体积/困惑度代价列在一起。下面挑几个有代表性的档位，看它们在"体积"和"质量"之间各站在哪：</p>
<div class="cols">
  <div class="col"><h4>Q8_0（约 8.5 bpw）</h4><p>几乎无损，体积大；适合对质量极敏感、显存又够的场景。</p></div>
  <div class="col"><h4>Q4_K_M（约 4.8 bpw）</h4><p>社区最常用的"甜点档"：体积小一大半，质量损失很小，日常首选。</p></div>
  <div class="col"><h4>IQ2_XS（约 2.3 bpw）</h4><p>超低比特、极致省显存；靠 imatrix 撑质量，否则会明显变差。</p></div>
</div>
<p>挑档位的直觉和 L06 一脉相承：bpw 越低，模型越小、跑得越省，但精度损失越大，困惑度（ppl，下一课讲）越高。大多数人会落在 Q4_K_M 这类"甜点档"上——体积已经小到能塞进消费级显卡，质量却几乎看不出退步。只有当显存特别紧张时，才会往 IQ2 这种超低比特走，而那时 imatrix 就成了救命稻草。所以"挑档位"从来不是挑最小的，而是在你的显存预算下，挑那个质量还撑得住的最小档。</p>

<h2>imatrix 重要性矩阵</h2>
<p>这里有个朴素但关键的观察：<strong>不是所有权重都一样重要</strong>。有些权重在模型干活时几乎总被强烈激活、对输出影响很大；有些则常年"打酱油"。如果量化时一视同仁地给所有权重同样的精度，就太浪费了——重要的权重精度不够会明显伤质量，而给不重要的权重留高精度又是白费比特。imatrix（importance matrix，重要性矩阵）就是来解决这个"<strong>比特预算怎么分</strong>"的问题的。打个比方，这就像考试时间有限：与其每道题都花同样多时间，不如把时间多花在分值高的大题上、小题快速带过——总分自然更高。imatrix 干的就是给权重"按分值分配精度"的活儿：先搞清楚哪些权重是"大题"，再把宝贵的比特预算重点投给它们。没有这份"分值表"，量化就只能盲目地一视同仁，难免把精度浪费在无关紧要的地方。</p>
<div class="flow">
  <div class="node"><div class="nt">校准文本</div><div class="nd">几百段代表性文本</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">跑模型 + collect_imatrix</div><div class="nd">累计每列激活幅度</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">imatrix.gguf</div><div class="nd">每个权重列的重要性</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">quantize --imatrix</div><div class="nd">按重要性分配精度</div></div>
</div>
<p>怎么知道哪些权重重要？办法很直接：拿一批<strong>校准文本</strong>（calibration text，几百段有代表性的语料）真的跑一遍模型，在前向过程中用一个 eval-callback 钩子 <span class="mono">collect_imatrix</span> 把每个权重张量<strong>每一列的激活幅度</strong>累加起来（源码里存成 <span class="mono">Stats</span> 的 values / counts）。被激活得越多越强的列，就越"重要"。跑完后，这些统计被存成一个 <span class="mono">imatrix.gguf</span> 文件，等量化时再喂回去。</p>
<pre class="code"><span class="cm"># 第一步: 用校准文本生成重要性矩阵 (tools/imatrix)</span>
llama-imatrix -m model.gguf -f calib.txt -o imatrix.gguf
<span class="cm"># 内部: 前向时 collect_imatrix(t, ...) 累计每个权重张量每列的激活幅度</span>

<span class="cm"># 第二步: 量化时把它喂进去, 精度优先留给重要的列</span>
llama-quantize --imatrix imatrix.gguf in.gguf out.gguf IQ2_XS</pre>
<p>有了这份重要性清单，量化时就能<strong>因材施教</strong>：在同样的比特预算下，给重要的权重列分配更准的量化（让它们舍入误差更小），把不可避免的误差更多地推给那些"无关紧要"的列。下面用一个最小例子，看一行权重在 imatrix 加权下是怎么被量化的：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次 imatrix 加权量化</b>：同一行权重，重要的列（imatrix 判定）量化得更准，误差被推给不重要的列（数值为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 一行权重</h5>
      <div class="cellrow"><span class="vc">0.50</span><span class="vc">0.02</span><span class="vc">0.48</span><span class="vc">-0.03</span></div>
      <div class="tlab">原始 fp16 值</div></div>
    <div class="op">imatrix<br>重要性</div>
    <div class="stn"><h5>② 重要性</h5>
      <div class="cellrow"><span class="vc hot">高</span><span class="vc dim">低</span><span class="vc hot">高</span><span class="vc dim">低</span></div>
      <div class="tlab">哪些列被激活得多</div></div>
    <div class="op">按重要性<br>量化</div>
    <div class="stn"><h5>③ 4-bit 码</h5>
      <div class="cellrow"><span class="vc hot">0.50</span><span class="vc">0.06</span><span class="vc hot">0.48</span><span class="vc">-0.07</span></div>
      <div class="tlab">重要列舍入更准</div></div>
    <div class="op">还原<br>看误差</div>
    <div class="stn"><h5>④ 误差</h5>
      <div class="cellrow"><span class="vc blue">~=0</span><span class="vc">0.04</span><span class="vc blue">~=0</span><span class="vc">0.04</span></div>
      <div class="tlab">误差被推给不重要的列</div></div>
  </div>
</div>

<h2>为什么这样更好</h2>
<p>道理其实一句话就能说清：<strong>同样的比特，花在刀刃上</strong>。普通量化把误差均匀摊给所有权重；imatrix 量化则让重要的权重几乎不损失精度，把误差集中倒给那些本来就影响不大的权重。结果就是：在完全相同的体积（比特数）下，模型整体的困惑度（ppl）更低、表现更接近原始的 fp16。比特数没变，质量却回来了一截——这就是 imatrix 的魔力，也是"测量一下再优化"这种笨功夫换来的实在好处。更妙的是，这一切对使用者完全透明：你下载一个带 imatrix 的量化模型，加载、推理的代码一行都不用改，质量却凭空好了一截——所有的聪明都发生在"压缩那一刻"，用的时候只管享受成果。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这正是社区里那些"<strong>imatrix 量化</strong>"（尤其是 IQ 系列，如 IQ2_XS、IQ3_M）质量出奇好的原因。在 2-3 bpw 这种超低比特下，不用 imatrix 的模型往往已经明显变笨，而带 imatrix 的同档位却还能保持相当可用——差别就在于"误差倒给谁"。所以你在 Hugging Face 上看到标着 imatrix 或 "IQ" 的量化版本，背后都跑过一遍校准文本、生成过一份重要性矩阵。理解了这一层，你下次挑量化模型时就有了判断：同样的档位，带 imatrix 的通常更值得选；而越往超低比特走，有没有 imatrix 的差距就越大。值得提醒的是，imatrix 的质量也取决于<strong>校准文本选得好不好</strong>：如果校准语料和你实际的用途差很远（比如拿纯英文语料校准、却主要用来写中文代码），测出来的"重要性"就可能不贴合，效果打折扣。所以社区里讲究的量化作者，会用覆盖多种语言、多种任务的混合语料来生成 imatrix。这也提醒你：imatrix 不是魔法，它只是把"用代表性数据测出来的重要性"如实地用上了——数据有多代表，它就有多准。
</div>

<h2>怎么给自己选档位</h2>
<p>讲了这么多档位，到底该给自己选哪个？一个实用的决策顺序是：<strong>先看显存</strong>。把"模型大小"粗略估成"参数量 × bpw / 8"，再对照你显卡的显存——能宽裕放下的，就尽量选高一点的档位（质量更好）；放不下的，才往下压。比如一个 8B 模型，Q8_0 约 8GB、Q4_K_M 约 4.5GB、IQ2 约 2.5GB，你的卡有多大，基本就框定了可选的范围。</p>
<p>在显存允许的范围内，<strong>再看用途</strong>。要它写代码、做推理这种"差一点就错"的任务，质量优先，尽量别低于 Q4_K_M；只是闲聊、续写这种容错高的场景，往低压一两档通常也无伤大雅。还有个常被忽略的点：<strong>同样大小，宁可选更大模型的低档量化，也别选小模型的高档</strong>——一个 13B 的 Q4 往往比一个 7B 的 Q8 更聪明，哪怕它俩体积差不多。这是社区反复验证过的经验法则。</p>
<p>最后，只要往超低比特（IQ2、IQ3）走，就<strong>一定优先选带 imatrix</strong> 的版本；普通 Q4/Q5 这类中高档，带不带 imatrix 差别没那么大，但带上通常也只赚不亏。把"显存框范围、用途定底线、超低比特认 imatrix"这三步记住，你就能在满屏的量化文件名里快速锁定最适合自己的那一个。</p>
<div class="card detail">
  <div class="tag">🔬 细节</div>
  顺带说一句，量化通常是<strong>一次性</strong>的：你压好一个 GGUF，之后每次加载都直接用这个小文件，不必每次重压。所以为一次压缩多花点心思（试几个档位、生成一份 imatrix）很值——这点前期成本，会被之后无数次的快速加载和省下的显存反复摊薄。这也是为什么社区愿意为热门模型精心制作各档位的量化版本，供大家按需取用：辛苦一次，方便众人。
</div>

<h2>深入：档位命名与实用旗标</h2>
<p>最后两个折叠，补两个动手时一定会撞上的实际问题：那些古怪的档位名到底怎么读，以及除了选档位还有哪些实用旗标。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Q4_K_M、IQ2_XS……这些名字怎么读？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>档位名是有规律的。<span class="mono">Q4_0</span> 里的 <span class="mono">Q</span> 是 quantize、<span class="mono">4</span> 是每权重约 4 比特、<span class="mono">0</span> 是早期的简单方案。<span class="mono">Q4_K_M</span> 里多出的 <span class="mono">K</span> 表示这是"<strong>K-quant</strong>"（一种更聪明的分块量化，质量更好），<span class="mono">M</span> 是 medium（中等档，另有 S=small、L=large 微调体积）。而 <span class="mono">IQ2_XS</span> 里的 <span class="mono">IQ</span> 表示"<strong>带 imatrix 的超低比特</strong>"方案，<span class="mono">2</span> 是约 2 比特，<span class="mono">XS</span> 是 extra small。一句话速记：<strong>Q=基础、K=更聪明的分块、IQ=超低比特靠 imatrix、后缀 S/M/L=同档里的大小微调</strong>。看懂命名，你就能从一长串文件名里一眼挑出想要的那个，不必每个都去试。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 除了选档位，还有哪些实用旗标？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>最常用的是 <span class="mono">--dry-run</span>（对应 <span class="mono">params.dry_run</span>）：它只<strong>计算并打印</strong>量化后的最终体积，并不真的压——在你纠结"选哪个档位才塞得进显存"时，先 dry-run 几个档位对比体积，比真压一遍快太多了。<span class="mono">--keep-split</span> 让输出保持和输入一样的分片结构（大模型常被切成多个 <span class="mono">.gguf</span> 分卷）。还有 <span class="mono">--output-tensor-type</span> / <span class="mono">--token-embedding-type</span> 能单独给输出层、词嵌入这两个对质量影响大的张量定更高的精度——很多高质量量化就是靠"主体压狠一点、关键张量留高一点"这种混合策略做出来的。这些旗标背后，正是前面 <span class="mono">llama_model_quantize_params</span> 里那些字段，命令行只是把它们暴露出来而已。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama-quantize in.gguf out.gguf &lt;ftype&gt;</span> 一键压缩，背后调公共 API <span class="mono">llama_model_quantize</span> + <span class="mono">llama_model_quantize_params</span>。</li>
    <li>档位（<span class="mono">ftype</span>）= 每权重几比特（bpw）的方案；bpw 越低越小越省、但 ppl 越高。Q4_K_M 是常用"甜点档"。</li>
    <li>imatrix：用校准文本跑模型、<span class="mono">collect_imatrix</span> 累计每列激活幅度 -&gt; <span class="mono">imatrix.gguf</span>；量化时 <span class="mono">--imatrix</span> 喂入，精度优先留给重要列。</li>
    <li>同样比特下，imatrix 让"重要权重少丢精度、不重要的多担误差"，整体 ppl 更低——这是 IQ 系列质量好的原因。</li>
    <li>实用旗标：<span class="mono">--dry-run</span>（只试算体积）、<span class="mono">--keep-split</span>（保持分片）、<span class="mono">--output-tensor-type</span> 等（按张量定精度）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  量化工具这一课，藏着一个反复出现的工程智慧：<strong>面对有限的预算，与其平均分配，不如按重要性分配</strong>。imatrix 不增加一个比特，只是把同样的比特花得更聪明——这和缓存把热数据放近、调度器把算力给关键任务，是同一种思路。它也提醒我们：很多"免费的午餐"其实来自"先花点力气测量、再据此优化"。生成 imatrix 要先跑一遍校准文本（花点时间），换来的却是同等体积下更好的质量（长期受益）。从 L6/L12 的"原理"到这一课的"工具"，你现在不仅知道量化是什么，还知道怎么把它用到最好——下一课，我们就用困惑度这把尺子，亲手量一量量化到底损失了多少。再往大里说，这种"先测量、再按重要性分配"的思路，在计算机科学里到处都是：JIT 编译器先看哪些代码热、再重点优化它；数据库先统计哪些查询频繁、再为它们建索引。它们和 imatrix 共享同一条信念——<strong>与其凭空猜，不如用真实运行数据说话</strong>。把这条信念带在身上，你以后遇到任何"资源有限、又想要最好效果"的问题，都会本能地先问一句：能不能先测一测，看看力气该往哪儿使？这，比记住任何一个量化档位的名字都更有用。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
L6 and L12 already covered quantization's "principle" - why a few bits can approximate a float, and how each format lays out its bytes. This lesson takes a different angle: <strong>how to actually shrink a model with the tool</strong>. One <span class="mono">llama-quantize</span> command turns a dozen-GB fp16 model into a 3-5 GB Q4_K_M; add imatrix (the importance matrix) and the same bit width claws back a chunk of the lost quality.
</p>
<p style="color:var(--muted);margin-top:.4rem">In other words, L6/L12 is "understand the principle", this lesson is "operate the tool": knowing what each flag tunes, how different levels trade size against quality, and how to use imatrix, that "quality-restoring" key. Quantization is the crucial step that lets big models run on ordinary GPUs or even pure CPU, and this lesson teaches you to do it by hand.</p>
<p style="color:var(--muted)">The two stars here are <span class="mono">tools/quantize</span> (the compressor itself) and <span class="mono">tools/imatrix</span> (the companion that builds the importance matrix). We first see how quantize compresses in one command and which public API it calls underneath, then why imatrix can raise quality without adding any bits.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  what the quantize tool does is essentially one <strong>lossy compression</strong>: turn each weight from 16-bit float into a thriftier 4-bit or 5-bit representation. What you save is real VRAM and bandwidth - half the model size, twice the load speed, a cheaper GPU it fits on. The price is lost precision, but that loss is controllable: the level (ftype) lets you pick any point between "how hard to compress" and "how much quality to keep", and imatrix acts like a smart budget allocator, spending the limited bits first on the most important weights. Understand this lesson and you hold the most-used tool for "moving a model onto your own machine" - the vast majority of GGUF quantized models you download are the product of this step. Put differently, this lesson turns the "principle" of L6/L12 into a craft you can actually use. And the bar is low: you need not understand the inner math of quantization algorithms - just tune a few flags and know each level's trade-off, and you can compress a usable model; the real complexity is all packaged up by the llama-quantize tool and the public API behind it, so you stand on ready-made shoulders.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of quantization as <strong>compressing a photo</strong>: the original (fp16) is sharp but large; saved as JPEG (quantized) it is much smaller, but lossy. The "level" is like JPEG's quality slider - at 90% you can barely tell the difference and the file is medium; at 30% it is tiny but starts to smear. imatrix is more like <strong>smart compression</strong>: it first analyzes which parts of the image are the face and which are blank background, then gives more byte budget to the face and less to the background. At the same file size, the important region is clearer - exactly what imatrix does in the world of weights: keep precision first for the "most-used" weights, and dump more error into the "irrelevant" corners. The analogy stretches one step further: JPEG can compress hugely with little visible loss because of the prior that "the human eye is insensitive to certain detail"; imatrix does the same, leaning on the prior, measured on real data, that "the model is insensitive to certain weights". Both say one thing: <strong>compression is never an even cut, but knowing where you can cut and where you cannot</strong>. The better you understand "what matters" in the data, the more quality you keep at the same size - a good quant scheme is often not a fancier algorithm but "a truer measurement".
</div>

<h2>How the quantize tool is used</h2>
<p>The most common use is one line: <span class="mono">llama-quantize in.gguf out.gguf Q4_K_M</span> - feed an fp16/fp32 GGUF, name a target level (here Q4_K_M), and it emits a compressed small GGUF. The entry is <span class="mono">tools/quantize/main.cpp</span> (thin), the body logic is in <span class="mono">quantize.cpp</span>, and the real work is the <strong>public API</strong> it calls, <span class="mono">llama_model_quantize(in, out, &amp;params)</span> - note this is a function from L25's <span class="mono">llama.h</span>, so the quantization capability is public too, not only the command line; you can call it from your own program.</p>
<pre class="code"><span class="cm">// the quantize tool's heart: the public API behind one command (simplified from tools/quantize/quantize.cpp)</span>
llama_model_quantize_params params = <span class="fn">llama_model_quantize_default_params</span>();
params.ftype   = LLAMA_FTYPE_MOSTLY_Q4_K_M;  <span class="cm">// target level</span>
params.imatrix = imatrix_data;               <span class="cm">// optional: feed in the importance matrix</span>
params.dry_run = false;                       <span class="cm">// true = only compute size, do not really compress</span>
<span class="fn">llama_model_quantize</span>(<span class="st">"in.gguf"</span>, <span class="st">"out.gguf"</span>, &amp;params);</pre>
<p>That <span class="mono">params</span> (<span class="mono">llama_model_quantize_params</span>) hides several practical knobs: <span class="mono">ftype</span> picks the target level; <span class="mono">dry_run</span> set to true only <strong>trial-computes</strong> how big the result would be without really compressing (very handy when picking a level); <span class="mono">output_tensor_type</span> / <span class="mono">token_embedding_type</span> can give a few key tensors their own higher precision; <span class="mono">keep_split</span> keeps the shard structure. In other words, quantization is not "one blunt cut", but can be tuned per tensor class, even per layer.</p>
<p>So what is a "level"? It is a <span class="mono">llama_ftype</span> enum value, mapping to a scheme of "how many bits per weight on average" (bpw). <span class="mono">quantize.cpp</span> has a table listing each level's name, bpw, and measured size/perplexity cost. Below are a few representative levels and where they stand between "size" and "quality":</p>
<div class="cols">
  <div class="col"><h4>Q8_0 (~8.5 bpw)</h4><p>nearly lossless, large; for quality-critical cases with enough VRAM.</p></div>
  <div class="col"><h4>Q4_K_M (~4.8 bpw)</h4><p>the community's favorite "sweet spot": much smaller, tiny quality loss, the everyday default.</p></div>
  <div class="col"><h4>IQ2_XS (~2.3 bpw)</h4><p>ultra-low-bit, extreme VRAM thrift; leans on imatrix for quality, else clearly worse.</p></div>
</div>
<p>The intuition for picking a level follows L06: the lower the bpw, the smaller and thriftier the model, but the greater the precision loss and the higher the perplexity (ppl, next lesson). Most people land on a "sweet spot" like Q4_K_M - small enough for consumer GPUs, yet barely any visible regression. Only when VRAM is very tight do you go toward ultra-low-bit IQ2, and there imatrix becomes the lifeline. So "picking a level" is never picking the smallest, but picking the smallest level whose quality still holds up under your VRAM budget.</p>

<h2>The imatrix importance matrix</h2>
<p>Here is a plain but crucial observation: <strong>not all weights matter equally</strong>. Some are almost always strongly activated and heavily affect the output; others mostly "sit around". Quantizing them all to the same precision is wasteful - too little precision on important weights clearly hurts quality, while high precision on unimportant ones wastes bits. imatrix (importance matrix) exists to solve this "<strong>how to split the bit budget</strong>" problem. By analogy, it is like a timed exam: rather than spend equal time on every question, spend more on the high-mark big questions and breeze through the small ones - the total score is naturally higher. imatrix does exactly this "allocate precision by marks" job for weights: first figure out which weights are the "big questions", then pour the precious bit budget mainly into them. Without this "mark sheet", quantization can only blindly treat all alike, inevitably wasting precision on places that hardly matter.</p>
<div class="flow">
  <div class="node"><div class="nt">calibration text</div><div class="nd">a few hundred passages</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">run + collect_imatrix</div><div class="nd">accumulate per-column activation</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">imatrix.gguf</div><div class="nd">importance of each column</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">quantize --imatrix</div><div class="nd">allocate precision by importance</div></div>
</div>
<p>How do we know which weights are important? Directly: take a batch of <strong>calibration text</strong> (a few hundred representative passages) and actually run the model, and during the forward pass an eval-callback hook <span class="mono">collect_imatrix</span> accumulates <strong>each column's activation magnitude</strong> for every weight tensor (stored in the source as <span class="mono">Stats</span> values / counts). The more strongly a column is activated, the more "important" it is. When done, these stats are saved into an <span class="mono">imatrix.gguf</span> file, to be fed back at quantize time.</p>
<pre class="code"><span class="cm"># step 1: build the importance matrix from calibration text (tools/imatrix)</span>
llama-imatrix -m model.gguf -f calib.txt -o imatrix.gguf
<span class="cm"># inside: during the forward pass collect_imatrix(t, ...) accumulates each weight tensor's per-column activation</span>

<span class="cm"># step 2: feed it at quantize time, precision goes first to important columns</span>
llama-quantize --imatrix imatrix.gguf in.gguf out.gguf IQ2_XS</pre>
<p>With this importance list, quantization can <strong>teach to each according to its aptitude</strong>: under the same bit budget, give important weight columns a more accurate quantization (smaller rounding error), and push the unavoidable error more onto the "irrelevant" columns. Below a minimal example shows how one row of weights is quantized under imatrix weighting:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one imatrix-weighted quantize</b>: the same row of weights, important columns (per imatrix) quantized more accurately, error pushed onto unimportant ones (values are illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) a row of weights</h5>
      <div class="cellrow"><span class="vc">0.50</span><span class="vc">0.02</span><span class="vc">0.48</span><span class="vc">-0.03</span></div>
      <div class="tlab">original fp16 values</div></div>
    <div class="op">imatrix<br>importance</div>
    <div class="stn"><h5>(2) importance</h5>
      <div class="cellrow"><span class="vc hot">hi</span><span class="vc dim">lo</span><span class="vc hot">hi</span><span class="vc dim">lo</span></div>
      <div class="tlab">which columns activate a lot</div></div>
    <div class="op">quantize by<br>importance</div>
    <div class="stn"><h5>(3) 4-bit codes</h5>
      <div class="cellrow"><span class="vc hot">0.50</span><span class="vc">0.06</span><span class="vc hot">0.48</span><span class="vc">-0.07</span></div>
      <div class="tlab">important columns round truer</div></div>
    <div class="op">dequant<br>see error</div>
    <div class="stn"><h5>(4) error</h5>
      <div class="cellrow"><span class="vc blue">~=0</span><span class="vc">0.04</span><span class="vc blue">~=0</span><span class="vc">0.04</span></div>
      <div class="tlab">error pushed onto unimportant columns</div></div>
  </div>
</div>

<h2>Why this is better</h2>
<p>The reason fits in a line: <strong>the same bits, spent where they count</strong>. Plain quantization spreads error evenly across all weights; imatrix quantization lets important weights lose almost no precision and dumps the error onto weights that hardly mattered anyway. The result: at exactly the same size (bit count), the model's overall perplexity (ppl) is lower and its behavior closer to the original fp16. Same bits, yet quality comes back a notch - that is imatrix's magic, and the real payoff of the plain effort of "measure first, then optimize". Better still, all of this is transparent to the user: you download an imatrix quant, change not a line of your load-and-infer code, yet quality is better out of nowhere - all the cleverness happens "at the moment of compression", and when you use it you simply enjoy the result.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  this is exactly why the community's "<strong>imatrix quants</strong>" (especially the IQ series, like IQ2_XS, IQ3_M) are surprisingly good. At ultra-low 2-3 bpw, a model without imatrix is often clearly dumber, while the same level with imatrix stays quite usable - the difference is "who the error is dumped on". So when you see a quant on Hugging Face marked imatrix or "IQ", a calibration-text run and an importance matrix lie behind it. Understand this and next time you pick a quant you have a rule: at the same level, the imatrix one is usually the better choice; and the lower the bit width, the bigger the gap between having imatrix and not. Worth a reminder: imatrix's quality also depends on <strong>how well the calibration text is chosen</strong> - if the calibration corpus is far from your actual use (say, calibrating on pure English but mainly writing Chinese code), the measured "importance" may not fit and the effect is diluted. So careful quant authors in the community build the imatrix from a mixed corpus covering many languages and tasks. It also reminds you: imatrix is no magic, it merely faithfully applies "importance measured on representative data" - as representative as the data is, that accurate it is.
</div>

<h2>Choosing a level for yourself</h2>
<p>After all this talk of levels, which should you actually pick? A practical decision order is: <strong>look at VRAM first</strong>. Roughly estimate "model size" as "parameter count x bpw / 8", compare it with your GPU's VRAM - if it fits with room to spare, pick a higher level (better quality); only when it does not fit do you compress further down. For example, an 8B model is about 8GB at Q8_0, 4.5GB at Q4_K_M, 2.5GB at IQ2 - how big your card is roughly frames the range of choices.</p>
<p>Within what VRAM allows, <strong>then look at the use</strong>. For "a small slip is a real error" tasks like coding or reasoning, prioritize quality and try not to go below Q4_K_M; for high-tolerance scenes like casual chat or continuation, dropping a level or two is usually harmless. One often-overlooked point: <strong>at the same size, prefer a low level of a bigger model over a high level of a smaller one</strong> - a 13B Q4 is often smarter than a 7B Q8 even if they are about the same size. This is a rule of thumb the community has verified again and again.</p>
<p>Finally, whenever you go to ultra-low bits (IQ2, IQ3), <strong>always prefer the imatrix version</strong>; for mid-to-high levels like plain Q4/Q5 the difference with or without imatrix is smaller, though having it is usually only a gain. Remember these three steps - "VRAM frames the range, use sets the floor, ultra-low-bit demands imatrix" - and you can quickly lock onto the one best suited to you from a screen full of quant file names.</p>
<div class="card detail">
  <div class="tag">🔬 Detail</div>
  By the way, quantization is usually <strong>one-time</strong>: you compress a GGUF once, and every later load just uses that small file, no re-compressing each time. So spending a bit more care on one compression (trying a few levels, building an imatrix) pays off - that upfront cost is amortized again and again by countless later fast loads and the VRAM saved. That is also why the community happily crafts each level's quant for popular models for everyone to grab as needed: toil once, ease for many.
</div>

<h2>Deep dive: level naming and practical flags</h2>
<p>Two final folds for two practical issues you will surely hit hands-on: how to read those odd level names, and what useful flags exist besides picking a level.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Q4_K_M, IQ2_XS... how do you read these names? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The names follow a pattern. In <span class="mono">Q4_0</span>, <span class="mono">Q</span> is quantize, <span class="mono">4</span> is about 4 bits per weight, <span class="mono">0</span> is the early simple scheme. The extra <span class="mono">K</span> in <span class="mono">Q4_K_M</span> means it is a "<strong>K-quant</strong>" (a smarter block quantization, better quality), and <span class="mono">M</span> is medium (with S=small, L=large fine-tuning the size). In <span class="mono">IQ2_XS</span>, <span class="mono">IQ</span> means an "<strong>ultra-low-bit scheme with imatrix</strong>", <span class="mono">2</span> is about 2 bits, <span class="mono">XS</span> is extra small. A one-line memo: <strong>Q=base, K=smarter blocks, IQ=ultra-low-bit via imatrix, suffix S/M/L=size tweak within a level</strong>. Read the naming and you can pick the one you want at a glance from a long list of file names, without trying each.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Besides picking a level, what useful flags are there? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The most useful is <span class="mono">--dry-run</span> (matching <span class="mono">params.dry_run</span>): it only <strong>computes and prints</strong> the final quantized size without really compressing - when you are torn over "which level fits VRAM", dry-running a few levels to compare sizes is far faster than really compressing each. <span class="mono">--keep-split</span> keeps the output's shard structure the same as the input (big models are often split into several <span class="mono">.gguf</span> shards). And <span class="mono">--output-tensor-type</span> / <span class="mono">--token-embedding-type</span> can give the output layer and token embeddings - two quality-sensitive tensors - their own higher precision; many high-quality quants come from exactly this mix of "compress the body harder, keep key tensors higher". Behind these flags are those fields in the earlier <span class="mono">llama_model_quantize_params</span>; the command line merely exposes them.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama-quantize in.gguf out.gguf &lt;ftype&gt;</span> compresses in one command, calling the public API <span class="mono">llama_model_quantize</span> + <span class="mono">llama_model_quantize_params</span>.</li>
    <li>A level (<span class="mono">ftype</span>) = a bits-per-weight (bpw) scheme; lower bpw = smaller and thriftier but higher ppl. Q4_K_M is the common "sweet spot".</li>
    <li>imatrix: run the model on calibration text, <span class="mono">collect_imatrix</span> accumulates per-column activation -&gt; <span class="mono">imatrix.gguf</span>; feed it via <span class="mono">--imatrix</span> at quantize time, precision goes first to important columns.</li>
    <li>At the same bits, imatrix makes "important weights lose less precision, unimportant ones bear more error", lowering overall ppl - why the IQ series is good quality.</li>
    <li>Practical flags: <span class="mono">--dry-run</span> (only trial-compute size), <span class="mono">--keep-split</span> (keep shards), <span class="mono">--output-tensor-type</span> etc. (per-tensor precision).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  this quantize lesson hides a recurring engineering wisdom: <strong>facing a limited budget, allocate by importance rather than evenly</strong>. imatrix adds not one bit; it just spends the same bits more cleverly - the same thinking as a cache keeping hot data near, or a scheduler giving compute to critical tasks. It also reminds us that many "free lunches" actually come from "spend a little effort measuring first, then optimize on that". Building an imatrix means running calibration text first (a time cost), but it buys better quality at the same size (a lasting gain). From L6/L12's "principle" to this lesson's "tool", you now not only know what quantization is, but how to use it best - next lesson, we take perplexity as a ruler and measure by hand just how much quantization actually loses. Zoom out and this "measure first, then allocate by importance" idea is everywhere in computer science: a JIT compiler first sees which code is hot, then optimizes it; a database first counts which queries are frequent, then builds indexes for them. They share imatrix's one belief - <strong>rather than guess in the void, let real runtime data speak</strong>. Carry this belief and any future "limited resources, yet want the best result" problem will make you instinctively ask first: can I measure a bit and see where the effort should go? That is more useful than memorizing any single quant level's name.
</div>
""",
}
