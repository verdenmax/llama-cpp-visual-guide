"""Content for Part 4 (inside llama inference)."""

LESSON_14 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
第三部分我们把 GGUF 文件格式（L13）和 ggml 引擎（L08-L12）讲透了。这一课上到 <span class="mono">llama</span> 层——看 <span class="mono">llama_model_loader</span> 怎么把一个 <span class="mono">.gguf</span>
真正<strong>加载成内存里的模型</strong>：读 metadata、把权重整理成一张按名字索引的张量清单、用 mmap 让数据零拷贝就位，并处理"一个模型拆成多文件"的分片。
</p>
<p style="color:var(--muted);margin-top:.4rem">为什么要专门讲加载？因为它是<strong>从磁盘字节到可用模型的第一道关</strong>：L13 教会我们 GGUF 长什么样，但"看懂格式"和"把它变成一个能建图、能推理的 <span class="mono">llama_model</span>"之间，
还隔着 loader 这一层。读懂它，你就接上了"文件"和"模型"两端。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  <span class="mono">llama_model_loader</span> 像一个<strong>收货验货员</strong>：先看箱单（GGUF 的 metadata 和 tensor info）、再核对货品清单（把每个张量按<strong>名字</strong>登记进 <span class="mono">weights_map</span>）、
  最后按单提货（mmap 让每个张量的 <span class="mono">data</span> 指针落到文件里对应的位置）。分片就是一批货分装好几个箱子，验货员按箱子上的 <span class="mono">of-N</span> 编号逐箱核对、并成一份总清单。
</div>

<h2>加载总览：loader 在做什么</h2>
<p>打开一个 <span class="mono">.gguf</span>，loader 大致走四步：读头部、建清单、映射数据、让指针就位。它不"算"任何东西，只负责把磁盘上的字节<strong>整理成内存里有名有姓、随时可取的张量</strong>。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>gguf_init_from_file</h4><p>读 GGUF 头：metadata（KV 超参，L13）+ 每个张量的 tensor info（name / dims / type / offset）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>建 weights_map</h4><p>把每个张量按<strong>名字</strong>登记：来源文件号 idx、在文件里的偏移 offs、以及张量本身。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>mmap 文件</h4><p>（use_mmap 时）把整个文件只读映射进地址空间，准备零拷贝取数（L13）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>load_data_for</h4><p>逐张量让 data 指针落到映射上对应位置（或在不用 mmap 时读入）。</p></div></div>
</div>
<p>把这张图落到结构体上（简化自 <span class="mono">src/llama-model-loader.h</span>）：</p>
<pre class="code"><span class="cm">// 简化自 src/llama-model-loader.h</span>
<span class="kw">struct</span> <span class="fn">llama_tensor_weight</span> { uint16_t idx; size_t offs; ggml_tensor * tensor; }; <span class="cm">// 来源文件号 + 偏移 + 张量</span>
<span class="kw">struct</span> <span class="fn">llama_model_loader</span> {
    gguf_context_ptr metadata_ptr;                       <span class="cm">// GGUF 头(KV + tensor infos)</span>
    std::map&lt;std::string, llama_tensor_weight&gt; weights_map; <span class="cm">// 按张量名索引的清单</span>
    bool use_mmap;  llama_mmaps mappings;                <span class="cm">// 零拷贝数据</span>
};</pre>
<p>逐个看：<span class="mono">metadata_ptr</span> 持有整份 GGUF 头，所有超参、词表、模板都在里头（L13 的自描述）；<span class="mono">weights_map</span> 是这一课的主角——一张<strong>按名字</strong>的张量清单，
键是张量名（如 <span class="mono">blk.0.attn_q.weight</span>），值记着这张量来自第几个文件（<span class="mono">idx</span>，为分片准备）、在文件里的偏移（<span class="mono">offs</span>）、以及对应的 <span class="mono">ggml_tensor</span>。</p>
<p>注意一个关键设计：清单用的是 <strong>map（按名字）</strong>，不是 vector（按顺序）。为什么？因为接下来（L15）这些张量要<strong>按名字</strong>对应到模型架构里的具体位置；而且分片时同一个逻辑模型的张量散在多个文件里，
按名字查比按下标稳得多。loader 在这里就把"散落的字节"收敛成了"一张能按名字点名的清单"。</p>
<p>把 loader 放到整个推理流程里看，它的位置很特别：<strong>加载只发生一次</strong>，而后面的 decode（L17）会被反复调用成千上万次。正因为只做一次，loader 可以"慢工出细活"地把模型整理周全——
读全 metadata、建好完整清单、接好所有数据指针；之后得到的 <span class="mono">llama_model</span> 是<strong>只读</strong>的，可以被许多次推理、甚至许多个会话反复复用，不必再碰磁盘。把"一次性的重活"和"反复做的快活"分开，是这套设计省时省力的根源。</p>
<p>"整理成清单"这一步看似平淡，其实是把后面一切操作变简单的关键。磁盘上的张量只是一段段连续字节，彼此之间没有任何"我是谁"的信息；loader 给它们配上名字、记下位置、建好索引，
之后无论是按架构建图（L16）、还是热插拔 LoRA（L24 预告），都能<strong>按名字精确点到某一个张量</strong>。先把混沌整理成有序，后面才能从容操作——这是工程里很常见、却常被忽视的一步。</p>
<p>这里也能看清几个层次的分工：最底下是 <span class="mono">gguf</span> 库（L13），只懂"GGUF 这种文件怎么解析"，连"权重"是什么都不关心；往上是 <span class="mono">ggml</span>（L08-L12），只懂张量和计算，不关心文件；
而 loader 正好<strong>夹在两者之间</strong>——它用 gguf 库读出原始信息，再用 ggml 建出张量，把"文件世界"翻译成"张量世界"。理解 loader，其实就是理解这道翻译是怎么发生的。</p>
<p>你平时跑 llama.cpp 时，启动那一大段刷屏的日志——"loaded meta data with N key-value pairs"、一行行张量名和类型——多半就是 loader 在汇报它的工作：读了多少 KV、识别出哪种架构、每个张量多大、用没用上 mmap。
下次看到这些日志，你就知道屏幕背后正是这一课讲的流程在跑。</p>
<p>要补一句：上面"四步"是<strong>概念上</strong>的顺序，真实代码里它们常交织在一起——比如读 tensor info 的同时就建好了 weights_map 的条目，建条目时就记下了 mmap 里的偏移。
教学上分成四步是为了看清职责，工程上则是一遍扫描尽量把能做的都做了。理解了每一步"在干什么"，看真实代码时就不会被它们交错的写法绕晕。</p>

<h2>读超参与张量</h2>
<p>loader 怎么知道模型有几层、多宽？不靠猜，全从 GGUF 的 metadata KV 里读——用一个模板方法 <span class="mono">get_key</span>，把"键"映射到具体的超参字段。键本身是自描述的（L13），下一课（L15）会讲这些键怎么按架构拼出来。</p>
<pre class="code"><span class="cm"># 伪代码: loader 的典型用法</span>
ml = <span class="fn">llama_model_loader</span>(path)              <span class="cm"># 内部 gguf_init_from_file 读头部</span>
ml.<span class="fn">get_key</span>(LLM_KV_BLOCK_COUNT, n_layer)   <span class="cm"># 从 KV 读超参(L15)</span>
<span class="kw">for</span> name, w <span class="kw">in</span> ml.weights_map:             <span class="cm"># 遍历张量清单</span>
    t = <span class="fn">create_tensor</span>(name, w.tensor-&gt;ne)   <span class="cm"># 在 ggml_context 里建元数据(L08)</span>
    ml.<span class="fn">load_data_for</span>(t)                    <span class="cm"># use_mmap: data 指进映射; 否则读入</span></pre>
<p>这段把前几课串了起来：<span class="mono">get_key</span> 读出的超参，下一课（L15）会装进 <span class="mono">llama_hparams</span>；<span class="mono">create_tensor</span> 在 <span class="mono">ggml_context</span>（L08）里只建<strong>元数据</strong>
（形状/类型，配合 L08 讲的 <span class="mono">no_alloc</span>）；<span class="mono">load_data_for</span> 才让真正的权重数据就位——而"就位"在 mmap 下就是<strong>把 data 指针指进文件映射</strong>，一个字节都不拷（L13 的零拷贝）。</p>
<p>所以 loader 读出来的每个张量，都对得上 L13 里那条 tensor info（name / dims / type / offset）：名字进了 <span class="mono">weights_map</span> 的键，dims/type 建成 ggml 张量的元数据，offset 则告诉 <span class="mono">load_data_for</span>
去文件的哪个位置取数。L13 讲"文件里怎么存"，这一课讲"loader 怎么把它读回内存"，两课正好首尾相接。</p>
<p>这里藏着一个和 L13"头部轻、尾部重"呼应的巧思：loader <strong>先</strong>读那一小段头部（metadata + tensor info，通常几十 KB），就把模型的全部结构搞清楚了——几层、多宽、有哪些张量、各在文件哪个位置；
<strong>之后</strong>才按需去碰那几个 GB 的权重数据。正因为"描述"和"数据"在文件里是分开的，loader 才能用很小的代价先建好整张清单和骨架，把真正的大块留到 mmap 按页惰性载入。这一步的轻量，直接决定了大模型能不能"秒开"。</p>
<p>还要留意 <span class="mono">create_tensor</span> 这一步的轻——它在 <span class="mono">ggml_context</span> 里建的只是张量的<strong>元数据</strong>（形状、类型、名字），并不为那几 GB 的浮点数据另开缓冲（这正是 L08 讲的 <span class="mono">no_alloc</span>）。
于是装下"整个模型的骨架"只要几 MB 的 context，真正占地方的权重则由 mmap 映射承接。元数据归元数据、数据归数据，两者分头安放——这是 L08 内存观在加载阶段的直接兑现，也让"先把结构建全、数据按需就位"成为可能。</p>
<p>再把 <span class="mono">offs</span> 这个字段说细一点：它记的是这个张量的数据在文件里的<strong>字节偏移</strong>（对应 L13 tensor info 的 offset）。有了它，<span class="mono">load_data_for</span> 才能在 mmap 映射里<strong>直接算出</strong>这个张量从哪开始——
映射基址 + 数据段起点 + offs，一步定位，不用顺序扫描。这正是 L13 讲的"按 offset 定位张量"在加载代码里的落地。</p>
<p>读张量时 loader 还会做一些<strong>一致性检查</strong>：比如某个张量的形状要和架构期望的对得上、类型是否被支持。这些检查放在加载期做最划算——一旦放行，后面建图、推理就可以默认"张量都是对的"，
不必每步重新提防。把校验集中在入口，是让后续代码能写得干净利落的前提。</p>

<h2>分片：一个模型拆成多文件</h2>
<p>特别大的模型（几十上百 GB）常被拆成<strong>多个文件</strong>，方便下载与分发。loader 把这些片当成<strong>一个逻辑模型</strong>读：按编号挨个打开、把各片的张量并进同一张 <span class="mono">weights_map</span>。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>分片 -&gt; 一个逻辑模型</b>：按 of-N 编号逐片打开，张量并进同一张 weights_map</div>
  <div class="cells"><span class="lab">磁盘</span><span class="cell hl">model-00001-of-00003</span><span class="cell">00002-of-00003</span><span class="cell">00003-of-00003</span><span class="lab">-&gt; 一个逻辑模型</span></div>
</div>
<p>分片靠几样东西对上：文件名格式 <span class="mono">"%s-%05d-of-%05d.gguf"</span>（由 <span class="mono">llama_split_path</span> 拼，<span class="mono">src/llama.cpp</span>）告诉你"第几片、共几片"；元数据键 <span class="mono">split.count</span>
（<span class="mono">LLM_KV_SPLIT_COUNT</span>）记总片数；而每个张量在 <span class="mono">weights_map</span> 里的 <span class="mono">idx</span> 字段，正是记着"我来自第几片"。</p>
<p>这也解释了前面为什么 <span class="mono">weights_map</span> 要按名字：分片把同一个模型的张量分散到多个文件，唯有按名字才能跨文件把它们统一查到、拼成完整的一份。对使用者来说，分不分片几乎无感——你只管给入口一个路径，loader 在背后把碎片拼好。</p>
<p>为什么要分片？一是<strong>下载与分发</strong>友好：一个 200 GB 的模型切成几十个几 GB 的片，断点续传、并行下载、镜像同步都更容易；二是有些文件系统对单文件大小有上限，分片能绕开；
三是方便<strong>按需取用</strong>。你在 HuggingFace 上看到的大模型，权重文件往往就是 <span class="mono">...-of-00010.gguf</span> 这样一长串。</p>
<p>一个自然的疑问：分片之后，前面说的 mmap 零拷贝还成立吗？成立——loader 给每一片各做一个映射（<span class="mono">mappings</span> 是一组映射、不是一个），每个张量的 <span class="mono">idx</span> 记着它属于哪一片，
<span class="mono">load_data_for</span> 就去对应那片的映射里取数。所以"分片"和"零拷贝"是正交的两件事：分片解决"文件太大不好搬"，mmap 解决"数据太大不想拷"，两者叠加，超大模型既好分发、又能秒加载。</p>
<p>顺便消除一个误解：分片不是什么特殊模式。单文件模型其实就是"只有一片"的退化情形——<span class="mono">split.count</span> 为 1（或干脆没有这个键）。loader 的代码<strong>统一</strong>按"可能有多片"来写，单文件只是片数为一的特例。
这种"把特例当成通例的一种"的写法，让代码更简单、也更少出错。</p>
<p>各家工具（包括 L02 的 <span class="mono">gguf-py</span>）在导出大模型时，会按一个目标分片大小自动切，并把 <span class="mono">split.*</span> 这几个键写进每一片——loader 读出来就能无缝拼回。
写入方和读取方共享同一套分片约定，正是 GGUF 自描述精神（L13）在"多文件"维度上的延伸。</p>

<h2>入口与衔接</h2>
<p>从外面看，加载就是一个函数调用。它的调用链很直白：</p>
<div class="flow">
  <div class="node"><div class="nt">llama_model_load_from_file</div><div class="nd">公开入口<br>(或 _from_splits)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">..._impl</div><div class="nd">内部实现</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">llama_model_load</div><div class="nd">用 loader 读<br>metadata + 张量</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_model</div><div class="nd">权重 data 已指进 mmap</div></div>
</div>
<p>公开入口 <span class="mono">llama_model_load_from_file</span>（以及分片版 <span class="mono">llama_model_load_from_splits</span>）都汇到 <span class="mono">..._impl</span>，再调静态的 <span class="mono">llama_model_load</span>，由它创建 loader、读出 metadata 与张量，
最终返回一个 <span class="mono">llama_model</span>——里面的权重张量，data 指针已经指进了 mmap 映射，随取随用。</p>
<p>加载到此为止。下一课（L15）我们就接着问：这一堆"带名字的张量"，怎么知道自己属于哪种架构（llama？qwen2？）、每层有哪些部件、怎么按超参组织起来——也就是 loader 读出的东西，到底要<strong>按什么图纸</strong>拼成一个模型。</p>
<p>稍微展开一下加载完拿到的 <span class="mono">llama_model</span> 到底是什么：它是一个<strong>只读</strong>的对象，装着所有权重张量（data 指针指进 mmap）、加上从 metadata 读出的超参（L15 会装进 <span class="mono">llama_hparams</span>）和词表（L20）。
它不含任何"会话状态"——没有 KV cache、没有当前算到哪。这正是下一层（L17）要把 <span class="mono">llama_model</span> 和 <span class="mono">llama_context</span> 分开的伏笔：知识（权重）只读可共享，状态（KV/进度）每会话一份。加载这一课，交付的就是那份"只读的知识"。</p>
<p>顺带一提：加载失败是有明确信号的——magic 不对、version 不认识、某个必需张量缺失，loader 都会当场报错而不是带病继续。这种"加载期就把问题暴露出来"的做法，比"跑到一半才崩"友好得多，也是把校验集中在 loader 这一层的好处。</p>
<p>把这一课收个尾：loader 站在"格式"和"模型"之间，向下它只关心 GGUF 的字节怎么排（L13），向上它只交付一份干净的、按名字可查的、数据已就位的张量集合。它不懂什么是注意力、不懂 llama 和 qwen 有什么不同——这些是上面几课的事。
正是这种"各司其职、边界清晰"的分层，让 llama.cpp 能一边支持越来越多的新格式细节、一边支持越来越多的新架构，而两边的改动很少互相牵连。读懂了加载，你就握住了从磁盘到模型的第一环。</p>
<p>最后看一眼那条"入口 -&gt; _impl -&gt; _load"的调用链为什么要分这么多层。最外层 <span class="mono">llama_model_load_from_file</span> 是<strong>稳定的公开 C API</strong>，要长期不变、给各种语言绑定调用；
中间的 <span class="mono">_impl</span> 和静态 <span class="mono">llama_model_load</span> 是内部实现，可以随时重构。把"对外承诺"和"对内实现"分开，是库设计的基本功——你调的是一个十年不变的名字，它背后怎么演进你不必关心。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么用 mmap，而不是把权重全读进内存？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>这正是 L13 讲过的零拷贝。权重动辄几个 GB，如果老老实实 <span class="mono">read()</span> 进一块内存，既慢又占地方。mmap 则把文件<strong>映射</strong>进地址空间，张量 data 指针看着像普通内存、实则指向磁盘页，真正用到哪页操作系统才载入。</p>
    <p>好处有三：启动几乎不花时间在"搬数据"上（秒加载）；物理内存按需、可被系统回收；多个进程映射同一个文件还能<strong>共享</strong>同一份物理页——同机起多个实例时省内存。所以 loader 默认 <span class="mono">use_mmap=true</span>，只有某些后端或平台才关掉它。</p>
    <p>换句话说，loader 在这一步做的不是"把权重读进来"，而是"<strong>把权重的位置接好</strong>"。数据始终躺在文件里，需要时才一页页流进来，这是大模型能在普通机器上跑起来的关键之一。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> weights_map 为什么按名字索引，而不是按顺序？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为张量最终要<strong>按名字</strong>对应到模型架构里的具体位置。下一课会看到，<span class="mono">blk.0.attn_q.weight</span> 这种名字是有约定的（LLM_TENSOR_NAMES），建图时正是拿名字去 <span class="mono">weights_map</span> 里取对应权重。</p>
    <p>按顺序（下标）则很脆：换个导出工具、张量排列略有不同，下标就全错位了；而名字是稳定的契约。更要紧的是<strong>分片</strong>——同一个逻辑模型的张量散在多个文件里，只有按名字才能跨文件把它们统一查到。</p>
    <p>所以 map 既稳又省心：不管张量物理上躺在哪个文件、第几位，只要名字对得上，就能精确点名。loader 把"物理排布"和"逻辑名字"解耦，是后续一切按名字操作的基础。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 分片到底怎么对上的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>靠三个元数据键加一套文件名约定。键有 <span class="mono">split.no</span>（这是第几片）、<span class="mono">split.count</span>（一共几片）、<span class="mono">split.tensors.count</span>（总张量数）；文件名则是 <span class="mono">"...-00001-of-00003.gguf"</span> 这种 of-N 编号。</p>
    <p>loader 拿到首片，从 <span class="mono">split.count</span> 知道还有几片，再用 <span class="mono">llama_split_path</span> 按编号拼出其余文件名、挨个打开，把每片的张量并进同一张 <span class="mono">weights_map</span>；每个张量的 <span class="mono">idx</span> 记下它来自第几片，方便 <span class="mono">load_data_for</span> 时去对的文件取数。</p>
    <p>对调用方来说，分片几乎透明：给一个路径（或用 <span class="mono">_from_splits</span> 给一组），loader 在背后把碎片拼成一份完整模型。这种"物理上分、逻辑上合"的设计，让超大模型既好分发、又不增加使用复杂度。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama_model_loader</span> = 读 GGUF <span class="mono">metadata</span>（用 <span class="mono">get_key</span> 取超参）+ 建 <span class="mono">weights_map</span>（按<strong>名字</strong>的张量清单）+ mmap 数据。</li>
    <li><span class="mono">weights_map</span> 用 map 按名字索引（不是按顺序），为"按名字建图"（L15/L16）和分片跨文件查找打底。</li>
    <li>权重数据默认 <strong>mmap 零拷贝</strong>就位（L13），<span class="mono">load_data_for</span> 让 data 指针指进映射。</li>
    <li>分片：文件名 <span class="mono">"%s-%05d-of-%05d.gguf"</span> + <span class="mono">split.count</span>，loader 当成一个逻辑模型读。</li>
    <li>入口 <span class="mono">llama_model_load_from_file</span> -&gt; <span class="mono">_impl</span> -&gt; <span class="mono">llama_model_load</span>，最终返回 <span class="mono">llama_model</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  loader 把"<strong>解析格式</strong>"和"<strong>使用模型</strong>"干净地隔开——它只负责把字节变成<strong>带名字的张量清单 + 就位的数据指针</strong>，至于这些张量怎么接成一张前向网络，是 L15（架构）和 L16（建图）的事。
  正因为这道边界清晰，"支持新的格式细节"和"支持新的模型架构"才能各自演进、互不打扰。一个好的加载层，存在感越低越好：它把脏活做完，让上层只看到一个干净的模型。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Part 3 took apart the GGUF format (L13) and the ggml engine (L08-L12). This lesson moves up to the <span class="mono">llama</span> layer - seeing how <span class="mono">llama_model_loader</span> turns a <span class="mono">.gguf</span>
into an <strong>in-memory model</strong>: reading metadata, organizing the weights into a name-indexed tensor list, using mmap to bring data into place zero-copy, and handling "one model split across several files".
</p>
<p style="color:var(--muted);margin-top:.4rem">Why a whole lesson on loading? Because it is the <strong>first gate from disk bytes to a usable model</strong>: L13 taught us what GGUF looks like, but between "understanding the format" and "turning it into a graph-able,
inferable <span class="mono">llama_model</span>" sits this loader layer. Read it and you connect the two ends - "file" and "model".</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  <span class="mono">llama_model_loader</span> is like a <strong>goods-receiving clerk</strong>: first read the manifest (GGUF metadata and tensor info), then check the inventory (register each tensor <strong>by name</strong> into <span class="mono">weights_map</span>),
  and finally pick stock by the list (mmap lands each tensor's <span class="mono">data</span> pointer at its place in the file). A split model is one shipment packed into several boxes; the clerk checks them by the <span class="mono">of-N</span> number on each box and merges them into one list.
</div>

<h2>Loading overview: what the loader does</h2>
<p>Opening a <span class="mono">.gguf</span>, the loader roughly takes four steps: read the header, build the list, map the data, point the pointers. It computes nothing - it just organizes disk bytes into <strong>named, ready-to-fetch tensors in memory</strong>.</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>gguf_init_from_file</h4><p>Read the GGUF header: metadata (KV hyperparameters, L13) + each tensor's info (name / dims / type / offset).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>build weights_map</h4><p>Register each tensor <strong>by name</strong>: source file idx, offset offs in the file, and the tensor itself.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>mmap the file</h4><p>(when use_mmap) map the whole file read-only into the address space, ready for zero-copy fetch (L13).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>load_data_for</h4><p>Per tensor, land the data pointer at its place in the mapping (or read it in when not using mmap).</p></div></div>
</div>
<p>Landing that figure on the struct (simplified from <span class="mono">src/llama-model-loader.h</span>):</p>
<pre class="code"><span class="cm">// simplified from src/llama-model-loader.h</span>
<span class="kw">struct</span> <span class="fn">llama_tensor_weight</span> { uint16_t idx; size_t offs; ggml_tensor * tensor; }; <span class="cm">// source file idx + offset + tensor</span>
<span class="kw">struct</span> <span class="fn">llama_model_loader</span> {
    gguf_context_ptr metadata_ptr;                       <span class="cm">// GGUF header(KV + tensor infos)</span>
    std::map&lt;std::string, llama_tensor_weight&gt; weights_map; <span class="cm">// list indexed by tensor name</span>
    bool use_mmap;  llama_mmaps mappings;                <span class="cm">// zero-copy data</span>
};</pre>
<p>Field by field: <span class="mono">metadata_ptr</span> holds the whole GGUF header, with all hyperparameters, vocab, and template inside (L13's self-description); <span class="mono">weights_map</span> is this lesson's star - a <strong>name-indexed</strong> tensor list whose
key is the tensor name (e.g. <span class="mono">blk.0.attn_q.weight</span>) and whose value records which file the tensor came from (<span class="mono">idx</span>, for splits), its offset in the file (<span class="mono">offs</span>), and the matching <span class="mono">ggml_tensor</span>.</p>
<p>Note a key design: the list is a <strong>map (by name)</strong>, not a vector (by order). Why? Because next (L15) these tensors must be matched <strong>by name</strong> to specific positions in the model architecture; and with splits, one logical model's tensors are
scattered across several files, where look-up by name is far steadier than by index. Right here the loader collapses "scattered bytes" into "a list you can call by name".</p>
<p>Placed in the whole inference pipeline, the loader's position is special: <strong>loading happens once</strong>, while the decode that follows (L17) is called thousands of times. Precisely because it runs once, the loader can take its time to organize the model thoroughly -
read all metadata, build the complete list, wire up every data pointer; the resulting <span class="mono">llama_model</span> is <strong>read-only</strong> and can be reused across many inferences, even many sessions, without touching disk again. Separating "the one-time heavy work" from "the repeated fast work" is the root of this design's efficiency.</p>
<p>This "organize into a list" step looks mundane but is the key to making everything later simple. On disk, tensors are just runs of contiguous bytes with no "who am I" information; the loader gives them names, records positions, and builds an index,
so that whether building a graph by architecture (L16) or hot-swapping a LoRA (L24), the engine can <strong>point precisely at one tensor by name</strong>. Turning chaos into order first is what lets everything afterward proceed calmly - a common but underappreciated step in engineering.</p>
<p>You can also see the division of labor across layers here: at the bottom is the <span class="mono">gguf</span> library (L13), which only knows "how to parse a GGUF file" and cares nothing for what a "weight" is; above it is <span class="mono">ggml</span> (L08-L12), which only knows tensors and computation, not files;
and the loader sits <strong>exactly between them</strong> - using the gguf library to read raw info, then ggml to build tensors, translating the "file world" into the "tensor world". Understanding the loader is really understanding how that translation happens.</p>
<p>When you run llama.cpp, that wall of startup logs - "loaded meta data with N key-value pairs", lines of tensor names and types - is mostly the loader reporting its work: how many KVs it read, which architecture it recognized, how big each tensor is, whether mmap was used.
Next time you see those logs, you will know the process behind the screen is exactly what this lesson describes.</p>
<p>One caveat: the "four steps" above are a <strong>conceptual</strong> order; in real code they often interleave - e.g. reading a tensor info also builds its weights_map entry, and building the entry records the offset into the mmap.
Splitting into four steps is for seeing the responsibilities clearly; in engineering, one pass does as much as it can at once. Once you grasp what each step "is doing", the interleaved real code will not confuse you.</p>

<h2>Reading hyperparameters and tensors</h2>
<p>How does the loader know how many layers, how wide? Not by guessing - it reads it all from the GGUF metadata KVs, via a templated <span class="mono">get_key</span> that maps a "key" to a specific hyperparameter field. The keys are self-describing (L13); the next lesson (L15) covers how they are templated per architecture.</p>
<pre class="code"><span class="cm"># pseudocode: typical loader usage</span>
ml = <span class="fn">llama_model_loader</span>(path)              <span class="cm"># internally gguf_init_from_file reads the header</span>
ml.<span class="fn">get_key</span>(LLM_KV_BLOCK_COUNT, n_layer)   <span class="cm"># read a hyperparameter from the KVs(L15)</span>
<span class="kw">for</span> name, w <span class="kw">in</span> ml.weights_map:             <span class="cm"># walk the tensor list</span>
    t = <span class="fn">create_tensor</span>(name, w.tensor-&gt;ne)   <span class="cm"># build metadata in ggml_context(L08)</span>
    ml.<span class="fn">load_data_for</span>(t)                    <span class="cm"># use_mmap: data points into the map; else read in</span></pre>
<p>This snippet ties the earlier lessons together: the hyperparameters <span class="mono">get_key</span> reads will be packed into <span class="mono">llama_hparams</span> next lesson (L15); <span class="mono">create_tensor</span> builds only <strong>metadata</strong>
(shape/type) in the <span class="mono">ggml_context</span> (L08, with the <span class="mono">no_alloc</span> idea); and <span class="mono">load_data_for</span> brings the real weight data into place - which, under mmap, just means <strong>pointing the data pointer into the file mapping</strong>, copying not a byte (L13's zero-copy).</p>
<p>So every tensor the loader reads lines up with that tensor info from L13 (name / dims / type / offset): the name becomes a key in <span class="mono">weights_map</span>, dims/type build the ggml tensor's metadata, and the offset tells <span class="mono">load_data_for</span>
where in the file to fetch from. L13 covered "how it is stored in the file", this lesson covers "how the loader reads it back into memory" - the two meet end to end.</p>
<p>Here is a clever touch echoing L13's "light head, heavy tail": the loader <strong>first</strong> reads that small header (metadata + tensor info, usually tens of KB) and thereby learns the model's entire structure - how many layers, how wide, which tensors, where each sits in the file;
<strong>only then</strong> does it touch those GB of weight data on demand. Because "description" and "data" are separated in the file, the loader can build the whole list and skeleton at tiny cost and leave the real bulk to mmap's lazy, page-by-page loading. The lightness of this step directly decides whether a large model can "open instantly".</p>
<p>Note also how light the <span class="mono">create_tensor</span> step is - in the <span class="mono">ggml_context</span> it builds only a tensor's <strong>metadata</strong> (shape, type, name), reserving no buffer for those GB of float data (exactly L08's <span class="mono">no_alloc</span>).
So holding "the whole model's skeleton" takes only a few MB of context, while the weights that actually take space are carried by the mmap mapping. Metadata as metadata, data as data, placed separately - the direct cash-out of L08's memory model at load time, and what makes "build the full structure first, bring data into place on demand" possible.</p>
<p>A bit more on the <span class="mono">offs</span> field: it records this tensor's data <strong>byte offset</strong> in the file (matching L13's tensor info offset). With it, <span class="mono">load_data_for</span> can <strong>directly compute</strong> where this tensor starts in the mmap mapping -
mapping base + data-section start + offs, located in one step, no sequential scan. This is exactly L13's "addressing tensors by offset" realized in loading code.</p>
<p>While reading tensors the loader also does some <strong>consistency checks</strong>: a tensor's shape must match what the architecture expects, its type must be supported. Doing these at load time is most economical - once they pass, graph-building and inference can assume "the tensors are all correct"
without re-guarding at every step. Centralizing validation at the gate is what lets the later code stay clean.</p>

<h2>Splits: one model across several files</h2>
<p>Very large models (tens to hundreds of GB) are often split into <strong>multiple files</strong> for easier download and distribution. The loader reads these shards as <strong>one logical model</strong>: opening them by number and merging each shard's tensors into the same <span class="mono">weights_map</span>.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>shards -&gt; one logical model</b>: open by of-N number, merge tensors into one weights_map</div>
  <div class="cells"><span class="lab">disk</span><span class="cell hl">model-00001-of-00003</span><span class="cell">00002-of-00003</span><span class="cell">00003-of-00003</span><span class="lab">-&gt; one logical model</span></div>
</div>
<p>Splits line up via a few things: the filename format <span class="mono">"%s-%05d-of-%05d.gguf"</span> (built by <span class="mono">llama_split_path</span>, <span class="mono">src/llama.cpp</span>) tells you "which shard, of how many"; the metadata key <span class="mono">split.count</span>
(<span class="mono">LLM_KV_SPLIT_COUNT</span>) records the total shard count; and each tensor's <span class="mono">idx</span> field in <span class="mono">weights_map</span> records "which shard I came from".</p>
<p>This also explains why <span class="mono">weights_map</span> is keyed by name: splits scatter one model's tensors across files, and only by name can they be looked up uniformly across files and stitched into a complete whole. To the caller, split-or-not is nearly invisible - you just hand the entry point a path, and the loader stitches the pieces behind the scenes.</p>
<p>Why split at all? First, <strong>download and distribution</strong> friendliness: a 200 GB model cut into dozens of few-GB shards is easier to resume, download in parallel, and mirror; second, some file systems cap single-file size, which splitting sidesteps;
third, it eases <strong>selective loading</strong>. The large models you see on HuggingFace often have weight files like a long <span class="mono">...-of-00010.gguf</span> series.</p>
<p>A natural question: after splitting, does the earlier mmap zero-copy still hold? It does - the loader makes one mapping per shard (<span class="mono">mappings</span> is a set, not a single map), each tensor's <span class="mono">idx</span> records which shard it belongs to,
and <span class="mono">load_data_for</span> fetches from that shard's mapping. So "splitting" and "zero-copy" are orthogonal: splitting solves "the file is too big to move around", mmap solves "the data is too big to copy"; together, a huge model is both easy to distribute and instant to load.</p>
<p>Clearing up a misconception: splitting is not a special mode. A single-file model is just the degenerate "one shard" case - <span class="mono">split.count</span> is 1 (or the key is simply absent). The loader's code is written <strong>uniformly</strong> as "there may be multiple shards", and a single file is just the one-shard special case.
Treating the special case as one instance of the general case keeps the code simpler and less error-prone.</p>
<p>Tooling (including L02's <span class="mono">gguf-py</span>) splits a large model automatically by a target shard size when exporting, writing the <span class="mono">split.*</span> keys into every shard - which the loader reads to stitch them back seamlessly.
Writer and reader sharing one split convention is GGUF's self-describing spirit (L13) extended into the "multiple files" dimension.</p>

<h2>Entry points and the hand-off</h2>
<p>From the outside, loading is one function call, with a straightforward chain:</p>
<div class="flow">
  <div class="node"><div class="nt">llama_model_load_from_file</div><div class="nd">public entry<br>(or _from_splits)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">..._impl</div><div class="nd">internal impl</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">llama_model_load</div><div class="nd">loader reads<br>metadata + tensors</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_model</div><div class="nd">weight data points into mmap</div></div>
</div>
<p>The public entry <span class="mono">llama_model_load_from_file</span> (and the split version <span class="mono">llama_model_load_from_splits</span>) both funnel into <span class="mono">..._impl</span>, then call the static <span class="mono">llama_model_load</span>, which creates the loader, reads metadata and tensors,
and returns a <span class="mono">llama_model</span> - whose weight tensors already have their data pointers pointing into the mmap mapping, ready to use.</p>
<p>Loading ends here. The next lesson (L15) asks what comes next: this pile of "named tensors" - how does it know which architecture it belongs to (llama? qwen2?), which parts each layer has, how it is organized by the hyperparameters - that is, <strong>by what blueprint</strong> the loader's output is assembled into a model.</p>
<p>A bit more on what the <span class="mono">llama_model</span> you get after loading actually is: a <strong>read-only</strong> object holding all weight tensors (data pointers into mmap), plus the hyperparameters read from metadata (L15 packs them into <span class="mono">llama_hparams</span>) and the vocab (L20).
It contains no "session state" - no KV cache, no notion of where computation currently is. That foreshadows why the next layer (L17) separates <span class="mono">llama_model</span> from <span class="mono">llama_context</span>: knowledge (weights) is read-only and shareable, state (KV/progress) is per-session. What this loading lesson delivers is exactly that "read-only knowledge".</p>
<p>By the way: load failures are signaled clearly - a wrong magic, an unknown version, a missing required tensor, and the loader errors out on the spot rather than limping on. This "surface problems at load time" approach is far friendlier than "crash halfway through", and is a benefit of centralizing validation in the loader layer.</p>
<p>To close this lesson: the loader stands between "format" and "model" - downward it only cares how GGUF's bytes are laid out (L13), upward it only delivers a clean, name-addressable set of tensors with data in place. It knows nothing of attention, nothing of how llama differs from qwen - those belong to the lessons above.
This very "each does its job, with clear boundaries" layering is what lets llama.cpp keep supporting more format details on one side and more architectures on the other, with the two rarely entangling. Understand loading and you hold the first link from disk to model.</p>
<p>Finally, a look at why the "entry -&gt; _impl -&gt; _load" chain has so many layers. The outermost <span class="mono">llama_model_load_from_file</span> is the <strong>stable public C API</strong>, meant to stay unchanged for the long run and be called by bindings in many languages;
the inner <span class="mono">_impl</span> and static <span class="mono">llama_model_load</span> are implementation that can be refactored anytime. Separating "the outward promise" from "the inward implementation" is library-design basics - you call a name that holds for years, and need not care how it evolves underneath.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why mmap rather than reading all weights into memory? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>This is exactly L13's zero-copy. Weights are often several GB; dutifully <span class="mono">read()</span>-ing them into a buffer is slow and space-hungry. mmap instead <strong>maps</strong> the file into the address space - a tensor's data pointer looks like ordinary memory but points at disk pages, loaded by the OS only when actually touched.</p>
    <p>Three benefits: startup spends almost no time "moving data" (instant load); physical memory is on-demand and reclaimable; and multiple processes mapping the same file can <strong>share</strong> the same physical pages - saving memory when running several instances on one machine. So the loader defaults to <span class="mono">use_mmap=true</span>, turned off only on certain backends or platforms.</p>
    <p>In other words, what the loader does here is not "read the weights in" but "<strong>wire up where the weights are</strong>". The data stays in the file and streams in page by page on demand - one key to running large models on ordinary machines.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is weights_map keyed by name, not by order? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because tensors must ultimately be matched <strong>by name</strong> to specific positions in the model architecture. Next lesson you will see that names like <span class="mono">blk.0.attn_q.weight</span> follow a convention (LLM_TENSOR_NAMES), and graph-building fetches the matching weight from <span class="mono">weights_map</span> by name.</p>
    <p>By order (index) it would be fragile: a different export tool, a slightly different tensor arrangement, and every index is off. A name is a stable contract. More importantly, with <strong>splits</strong>, one logical model's tensors are scattered across files - only by name can they be looked up uniformly.</p>
    <p>So a map is both steady and convenient: no matter which file a tensor physically sits in, or in what position, as long as the name matches it can be called out precisely. The loader decouples "physical layout" from "logical name" - the basis for everything that follows operating by name.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How exactly do splits line up? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Via three metadata keys plus a filename convention. The keys are <span class="mono">split.no</span> (which shard this is), <span class="mono">split.count</span> (how many in total), and <span class="mono">split.tensors.count</span> (total tensor count); the filename is the of-N number <span class="mono">"...-00001-of-00003.gguf"</span>.</p>
    <p>Given the first shard, the loader learns the count from <span class="mono">split.count</span>, uses <span class="mono">llama_split_path</span> to build the other filenames by number, opens each, and merges its tensors into the same <span class="mono">weights_map</span>; each tensor's <span class="mono">idx</span> records which shard it came from, so <span class="mono">load_data_for</span> fetches from the right file.</p>
    <p>To the caller, splits are nearly transparent: hand over a path (or a set via <span class="mono">_from_splits</span>), and the loader stitches the pieces into one complete model. This "physically split, logically whole" design lets huge models be easy to distribute without adding usage complexity.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama_model_loader</span> = read GGUF <span class="mono">metadata</span> (hyperparameters via <span class="mono">get_key</span>) + build <span class="mono">weights_map</span> (a <strong>name-indexed</strong> tensor list) + mmap the data.</li>
    <li><span class="mono">weights_map</span> is a map keyed by name (not by order), underpinning "build-by-name" (L15/L16) and cross-file lookup for splits.</li>
    <li>Weight data defaults to <strong>mmap zero-copy</strong> in place (L13); <span class="mono">load_data_for</span> points the data pointer into the mapping.</li>
    <li>Splits: filename <span class="mono">"%s-%05d-of-%05d.gguf"</span> + <span class="mono">split.count</span>; the loader reads them as one logical model.</li>
    <li>Entry <span class="mono">llama_model_load_from_file</span> -&gt; <span class="mono">_impl</span> -&gt; <span class="mono">llama_model_load</span>, returning a <span class="mono">llama_model</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The loader cleanly separates "<strong>parsing the format</strong>" from "<strong>using the model</strong>" - it only turns bytes into a <strong>name-indexed tensor list + data pointers in place</strong>; how those tensors are wired into a forward network is L15's (architecture) and L16's (graph) job.
  Because that boundary is clear, "supporting new format details" and "supporting new model architectures" can each evolve without disturbing the other. A good loading layer is best when barely noticed: it does the dirty work so the upper layers see only a clean model.
</div>
""",
}

LESSON_15 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课 loader 把权重读成了一张"带名字的张量清单"，可这些张量怎么知道自己属于<strong>哪种架构</strong>（llama？qwen2？）、每层有哪些部件？这一课讲两样东西：<span class="mono">llama-arch</span>（架构标识 +
键名约定 + 张量名约定三套表）和 <span class="mono">llama-hparams</span>（几层、多宽、每层几个头）——它们合起来，是把"一堆张量"翻译成"一个具体可建图的模型"的<strong>说明书</strong>。
</p>
<p style="color:var(--muted);margin-top:.4rem">为什么把这两件事单拎一课？因为它们是 llama.cpp 能用<strong>一套代码读懂几十种模型</strong>的秘密。同样是 transformer，llama 和 qwen2 的差别，本质上就藏在"用哪套约定、几层多宽、张量怎么命名"里。
搞懂这一课，你就明白新模型是怎么"插进"这个引擎的。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  架构与超参像一套<strong>建筑图纸 + 规格表</strong>：图纸（<span class="mono">llm_arch</span> + 张量命名约定）说明"这是哪种楼、每层有哪些构件、各叫什么名"；规格表（<span class="mono">hparams</span>）给出具体尺寸——
  "几层、多宽、每层几个注意力头"。上一课的 loader 就像备好了一堆贴着标签的建材，而这一课的图纸和规格表，告诉它<strong>按什么名字提哪块料、按什么尺寸搭起来</strong>。
</div>

<h2>架构标识：llm_arch</h2>
<p>一切从一个问题开始：拿到一个模型，怎么知道它是 llama 还是 qwen2？答案在 L13 讲过的那个 GGUF 元数据键 <span class="mono">general.architecture</span> 里。它是一个字符串（比如 <span class="mono">"llama"</span>），
loader 读出它、在一张名字表里一查，就得到对应的<strong>架构枚举</strong> <span class="mono">LLM_ARCH_LLAMA</span>。这个枚举一旦定下，后面用哪套键名、哪套张量名、调哪个建图函数（L16），就全都定了。</p>
<div class="flow">
  <div class="node"><div class="nt">general.architecture</div><div class="nd">GGUF 里的字符串<br>= "llama"</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">LLM_ARCH_LLAMA</div><div class="nd">查 LLM_ARCH_NAMES<br>得到架构枚举</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">一套约定</div><div class="nd">KV 键 / 张量名 / 建图(L16)</div></div>
</div>
<p>落到源码（简化自 <span class="mono">src/llama-arch.{h,cpp}</span>）：</p>
<pre class="code"><span class="cm">// 简化自 src/llama-arch.h / src/llama-arch.cpp</span>
<span class="kw">enum</span> llm_arch { LLM_ARCH_LLAMA, LLM_ARCH_QWEN2, <span class="cm">/* ... 几十种 ... */</span> };
<span class="cm">// LLM_ARCH_NAMES: 枚举 &lt;-&gt; 字符串</span>
{ LLM_ARCH_LLAMA, <span class="st">"llama"</span> }, { LLM_ARCH_QWEN2, <span class="st">"qwen2"</span> }, <span class="cm">/* ... */</span></pre>
<p>这张 <span class="mono">LLM_ARCH_NAMES</span> 表是双向的桥：写文件时（L02 转换脚本）把架构枚举翻成字符串写进 GGUF，读文件时把字符串翻回枚举。一个小小的字符串，就是<strong>整个后续流程的总开关</strong>——
它一变，loader 找的键、建图用的函数全跟着变。所以你可以把 <span class="mono">general.architecture</span> 理解成模型在对引擎说："请按 llama 这套规矩来对待我。"</p>
<p>为什么要用一个枚举、而不是到处用字符串比较？因为枚举又快又不易写错，且能当 <span class="mono">switch</span> / 表格的下标。llama.cpp 里大量"按架构分情况"的逻辑——读哪些超参、张量怎么命名、建图怎么拼——都靠这个枚举来分流。
把"我是谁"收敛成一个枚举值，是让后面所有"按架构区别对待"的代码都能写得整齐的前提。</p>
<p>顺带感受一下这套机制的容量：llama.cpp 支持的架构早已是<strong>几十种</strong>——llama、qwen、mistral、phi、gemma、deepseek、stablelm…… 全都靠这一个 <span class="mono">llm_arch</span> 枚举区分。
新模型层出不穷，可它们绝大多数都是 transformer 的变体、差异有限；于是"再多一种架构"在引擎眼里，往往只是枚举里多一个值、表里多几行。这种"用一个枚举撑起一个生态"的容量，正是表驱动设计的威力。</p>
<p>除了 <span class="mono">general.architecture</span>，GGUF 头里还有一批 <span class="mono">general.*</span> 的通用元信息（L13 提过）：模型名、整体量化档位、量化版本等。它们和架构枚举一起，构成了"这到底是个什么模型"的完整自我介绍。
loader 一上来读的，正是这批通用信息加上架构相关的超参——前者认出"是谁"，后者量出"多大"。</p>
<p>这条"字符串 &lt;-&gt; 枚举"的桥也解释了 L02 转换脚本在做的事之一：把一个 HuggingFace 模型转成 GGUF 时，脚本要判断它是哪种架构、把对应的字符串（如 <span class="mono">"llama"</span>）写进 <span class="mono">general.architecture</span>。
写入方和读取方共用同一张 <span class="mono">LLM_ARCH_NAMES</span>，一写一读才能严丝合缝。这也是为什么有时一个全新架构的模型，需要先给 llama.cpp 加上对它的支持，转换脚本才认得、才转得出来。</p>

<h2>超参 llama_hparams</h2>
<p>知道了是哪种架构，还得知道<strong>具体尺寸</strong>：这个 llama 是 7B 还是 70B？多少层、多宽、每层几个头？这些数叫<strong>超参</strong>（hyperparameters），装在 <span class="mono">llama_hparams</span> 里，全部由上一课的 <span class="mono">get_key</span> 从 GGUF 的 KV 读出。</p>
<table class="t">
  <tr><th>超参</th><th>含义</th></tr>
  <tr><td><span class="mono">n_embd</span></td><td>隐藏维度（一个 token 向量多宽）</td></tr>
  <tr><td><span class="mono">n_layer()</span></td><td>层数（多少个 transformer block）</td></tr>
  <tr><td><span class="mono">n_head(il)</span></td><td>第 il 层的注意力头数</td></tr>
  <tr><td><span class="mono">n_head_kv(il)</span></td><td>第 il 层的 KV 头数（GQA 时 &lt; n_head）</td></tr>
  <tr><td><span class="mono">n_ff(il)</span></td><td>第 il 层 FFN 的中间维度</td></tr>
  <tr><td><span class="mono">rope_freq_base_train</span></td><td>RoPE 的频率基（位置编码，L16）</td></tr>
</table>
<pre class="code"><span class="cm">// 简化自 src/llama-hparams.h</span>
<span class="kw">struct</span> llama_hparams {
    uint32_t n_embd;  uint32_t n_ctx_train;  float rope_freq_base_train;  float f_norm_rms_eps;
    std::array&lt;uint32_t, LLAMA_MAX_LAYERS&gt; n_head_arr, n_head_kv_arr, n_ff_arr; <span class="cm">// 按层存</span>
    uint32_t n_layer() const;  uint32_t n_head(uint32_t il = 0) const;        <span class="cm">// 访问器, 不是字段!</span>
};</pre>
<p>这里有两个<strong>极易踩的坑</strong>，务必记牢。第一：<span class="mono">n_layer()</span>、<span class="mono">n_head(il)</span> 这些不是普通字段，而是<strong>访问器方法</strong>（注意有括号）。为什么？因为现代架构里，
不同层的头数、注意力类型可能不一样（比如 GQA、滑窗层），所以头数按层存进 <span class="mono">n_head_arr</span> 这样的数组，取的时候要带层号 <span class="mono">il</span>。把它当成定值字段去用，迟早出错。</p>
<p>第二个坑：<span class="mono">n_vocab</span>（词表大小）<strong>不在</strong> <span class="mono">llama_hparams</span> 里！它属于分词器，来自 <span class="mono">llama_vocab::n_tokens()</span>（L20 会讲）。很多人想当然以为词表大小是个超参，
结果在 hparams 里翻半天找不到。记住：hparams 管的是"网络的形状"，词表是另一码事，归分词器管。</p>
<p>这些超参一旦读出来，就成了整个推理的"尺寸基准"：建图（L16）时按 <span class="mono">n_layer()</span> 决定堆几层、按 <span class="mono">n_head(il)</span> 切多少个注意力头、按 <span class="mono">n_embd</span> 定各处矩阵的形状；
KV cache（L19）按层数和 KV 头数算该开多大。可以说，hparams 是把"一个抽象的 transformer"具体化成"这一个模型"的那组数字。</p>
<p>顺便厘清"参数量"和超参的关系。我们常说的 7B、70B，指的是模型权重里浮点数的<strong>总个数</strong>（70 亿、700 亿）；而这个总数，正是由超参算出来的——大致是 <span class="mono">n_layer</span> × 每层各权重矩阵尺寸之和，
而每个矩阵的尺寸又由 <span class="mono">n_embd</span>、<span class="mono">n_ff</span> 等决定。所以超参不是一堆孤立的数字，它们<strong>共同决定了模型有多大</strong>。读懂超参，你就能从一个模型的几个数，估出它要吃多少显存。</p>
<p>表里没列全的超参还有不少，各有用处：<span class="mono">n_ctx_train</span> 是模型训练时的上下文长度（你能开多长上下文的参考上限）；<span class="mono">f_norm_rms_eps</span> 是 RMSNorm 里防止除零的小常数（L11 的归一化用到）；<span class="mono">n_rot</span> 是 RoPE 实际旋转的维数。
这些数看着琐碎，却个个都会在建图（L16）时被某个算子精确用到——少一个、错一个，算出来的就不是这个模型了。</p>
<p>再说一句那个默认参数 <span class="mono">il = 0</span>：它让"对每层都一样"的简单模型用起来很省事——不传层号，默认取第 0 层（也就是所有层）的值。所以接口虽然是"按层取"，对老实的同构模型并不啰嗦。
这是个体贴的设计：复杂情况能表达，简单情况不添乱。</p>
<p>这里特别值得记住 <span class="mono">n_head_kv</span> 的意义：它直接决定 KV cache 有多大。GQA 的思路就是让多个 Q 头<strong>共享</strong>一组 K/V，于是 KV 头数远少于 Q 头数，KV cache 也就成倍变小（L19 会细讲）。
所以读一个模型的超参时，<span class="mono">n_head</span> 和 <span class="mono">n_head_kv</span> 的比值，几乎就告诉了你"这个模型对长上下文友不友好"。一个看似不起眼的超参，背后是显存与速度的大账。</p>

<h2>张量命名约定</h2>
<p>最后一块拼图：loader 读出的张量怎么<strong>按名字</strong>对上模型结构？靠一套命名约定。每个张量在文件里都有名字，而这些名字不是随便起的，遵循 <span class="mono">LLM_TENSOR_NAMES</span> 定义的模板。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>张量名模板（LLM_TENSOR_NAMES）</b>：按部件 + 层号命名，建图时照名取权重</div>
  <div class="cells"><span class="lab">名字</span><span class="cell hl">token_embd</span><span class="cell">blk.0.attn_q</span><span class="cell">blk.0.ffn_gate</span><span class="cell">output_norm</span></div>
</div>
<p>看这几个名字就懂了规律：<span class="mono">token_embd</span> 是词嵌入表（开头那层），<span class="mono">blk.0.attn_q</span> 是第 0 层的注意力 Q 投影、<span class="mono">blk.0.ffn_gate</span> 是第 0 层 FFN 的门控、<span class="mono">output_norm</span> 是最后的输出归一。
名字里 <span class="mono">blk.%d</span> 的 <span class="mono">%d</span> 会被层号填进去——这正是 <span class="mono">n_layer()</span> 派上用场的地方：循环 0 到 n_layer()，每层按模板拼出该层各张量的名字。</p>
<p>名字由一个叫 <span class="mono">LLM_TN</span> 的小构造器拼出来（它的 <span class="mono">tn(...)</span> 调用，把"部件 + 后缀 + 层号"组装成完整张量名）。建图（L16）时，每要一个权重，就用这个构造器拼出名字、去上一课的 <span class="mono">weights_map</span> 里查——
这就把"加载"和"建图"两课用名字这根线<strong>串了起来</strong>：loader 按名字存，建图按名字取，中间靠 <span class="mono">LLM_TENSOR_NAMES</span> 这份共同约定对齐。</p>
<p>这也回答了上一课留的悬念——为什么 <span class="mono">weights_map</span> 要按名字索引。因为名字是<strong>稳定的契约</strong>：换个导出工具、张量排列变了也不怕，只要名字这套约定不变，建图就总能精确取到它要的那块权重。
名字这层抽象，把"权重物理上躺在哪"和"逻辑上是哪个部件"彻底解耦了。</p>
<p>把一层 transformer 的张量名列全，规律就更清楚了：注意力部分有 <span class="mono">attn_q</span>/<span class="mono">attn_k</span>/<span class="mono">attn_v</span>/<span class="mono">attn_output</span> 四个投影、加一个 <span class="mono">attn_norm</span>；FFN 部分有 <span class="mono">ffn_gate</span>/<span class="mono">ffn_up</span>/<span class="mono">ffn_down</span> 三个矩阵、加一个 <span class="mono">ffn_norm</span>。
每一层都按这套模板复制一遍，前面加 <span class="mono">blk.层号.</span>。看懂这张"一层有哪些权重"的清单，你就看懂了 transformer 一个 block 的全部可学习参数。</p>
<p>为什么用 <span class="mono">.</span> 点号分层级命名（<span class="mono">blk.0.attn_q.weight</span>）？因为这天然形成一棵<strong>层级树</strong>：<span class="mono">blk</span> 下是各层、层下是各部件、部件下是 weight/bias。这种命名既清晰、又方便按前缀批量匹配——
比如想找第 0 层的所有权重，匹配 <span class="mono">blk.0.</span> 前缀即可。一个好的命名约定，不只是"起个名"，而是把结构信息编码进了名字本身。</p>
<p>这套命名还是<strong>跨架构通用</strong>的：不管 llama 还是 qwen2，第 0 层的注意力 Q 投影都叫 <span class="mono">blk.0.attn_q</span>。正因为大家共享同一套名字，针对张量的通用工具（量化、转换、可视化）才能不区分架构地处理任意模型。</p>
<p>再补一点 <span class="mono">LLM_TN</span> 的细节：它拼出的完整名字通常还带个后缀，区分 <span class="mono">.weight</span> 和 <span class="mono">.bias</span>——同一个部件可能有权重、也可能有偏置。所以 <span class="mono">tn(LLM_TENSOR_ATTN_Q, "weight", il)</span> 拼出的是 <span class="mono">blk.il.attn_q.weight</span>。
把"部件名模板 + 后缀 + 层号"三者交给一个构造器统一拼，既避免了到处手写字符串容易出的错，也让"改个命名规则"只需动一处。</p>
<p>顺便提一句：有些张量是<strong>可选</strong>的——比如不少现代模型的线性层没有 bias，那 <span class="mono">.bias</span> 那个张量在文件里就根本不存在。建图时按架构知道"这层该有哪些张量"，缺的可选项就跳过。
命名约定加上"哪些必需、哪些可选"的知识，才完整描述了一个架构的张量构成。</p>

<h2>自描述如何在架构层兑现</h2>
<p>把三样东西连起来看，L13 说的"<strong>自描述</strong>"就在架构层完整兑现了：<span class="mono">general.architecture</span> 选定 arch；带 <span class="mono">%s</span> 的 KV 键（如 <span class="mono">llama.block_count</span>）用架构名填模板、由 <span class="mono">get_key</span> 读出超参；
张量按 <span class="mono">LLM_TENSOR_NAMES</span> 命名一一对上。三套约定一咬合，loader 读出的"一堆张量"就成了"一个有名有姓、有形有状的具体模型"，随时可以交给 L16 建图。</p>
<p>值得回味的是这种设计的"<strong>表驱动</strong>"味道：架构是一张名字表、键是一张键名表、张量是一张张量名表。引擎的主干代码不写死任何一种模型，而是"照表办事"。于是支持一个新模型，多半不是改引擎，而是<strong>往这几张表里加几行 + 写一份建图</strong>（L16）。</p>
<p>设想一下如果<strong>不</strong>用表驱动会怎样：每支持一种新模型，就得在引擎主干里写一堆 <span class="mono">if (arch == "llama") ... else if (arch == "qwen2") ...</span> 的分支，读超参、命名张量、建图处处都要改。模型一多，这些分支就会织成一张谁也不敢动的网。
表驱动把这些差异从"散落在代码各处的 if"收进"集中的几张表 + 一份建图文件"，于是主干代码读起来始终是"照表办事"，清清爽爽。</p>
<p>这套"<strong>识别架构 -&gt; 读超参 -&gt; 按名取张量</strong>"的三步，其实是任何"通用模型加载器"都绕不开的骨架：先搞清是什么、再量出多大、最后把零件对号入座。llama.cpp 把这三步做得极其干净，正是它能在短时间内追上一个又一个新模型的工程根基。
下一课，我们就拿着这份"图纸 + 规格表 + 零件清单"，真正动手把它们拼成一张能算的前向计算图。</p>
<p>把这一课和上一课连起来看，会发现一条清晰的主线：L14 的 loader 把字节变成"带名字的张量"，L15 的 arch/hparams 给这些张量配上"图纸和规格"。到这里，模型已经从"磁盘上的一个文件"变成了"内存里一个有名有姓、知道自己几层多宽的对象"——
只差最后一步：把这些零件按图纸真正拼成一张能算的网络。那正是下一课的事。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 n_head 写成 n_head(il) 带层号？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为现代架构里，不同层的注意力配置可能不一样。最常见的是 <strong>GQA</strong>（分组查询注意力）：Q 头多、KV 头少，于是 <span class="mono">n_head_kv(il)</span> 会小于 <span class="mono">n_head(il)</span>，用来省 KV cache（L19）。</p>
    <p>还有些架构是<strong>混合</strong>的：某些层用全注意力、某些层用滑动窗口，各层的头数、窗口大小都可能不同。要表达这种"逐层不同"，最干净的办法就是把这些数存成<strong>按层的数组</strong>（<span class="mono">n_head_arr</span> 等），取的时候带上层号 <span class="mono">il</span>。</p>
    <p>所以 <span class="mono">n_head(il)</span> 是个方法、不是字段——它背后从数组里按层号取值。对大多数老实的同构模型，每层都一样，<span class="mono">il</span> 取默认 0 即可；但接口设计成按层取，才容得下那些"逐层不同"的新架构。这是个"为通用性留余地"的典型取舍。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 加一个新架构，要改哪几处？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>大致四步，且大多是"填表"：① 在 <span class="mono">enum llm_arch</span> 里加一项、在 <span class="mono">LLM_ARCH_NAMES</span> 里加它的名字；② 实现这架构的超参读取（从 KV 把它特有的几个超参读进 hparams）；③ 把它用到的张量在张量名表里登记；④ 写一份建图（L16，<span class="mono">src/models/&lt;arch&gt;.cpp</span>）。</p>
    <p>关键是<strong>第④步往往很轻</strong>：因为绝大多数架构用的是同一批积木（注意力、FFN、归一化），新架构多半只是"换个拼法"，能复用现成的 <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>（L16）。只有遇到真正新颖的结构，才需要补一两个新算子（L11）。</p>
    <p>这就是为什么 llama.cpp 能跟上层出不穷的新模型：大部分"新架构"在工程上其实是"<strong>填几张表 + 复用积木</strong>"，引擎主干纹丝不动。把"模型的多样性"收进表和建图文件，把"不变的机制"留在主干——这是这套设计最省力的地方。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 张量命名约定为什么这么重要？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为它是<strong>三方共享的契约</strong>。写入方（L02 的转换脚本）按这套名字把权重写进 GGUF；读取方（L14 的 loader）按名字建 <span class="mono">weights_map</span>；建图方（L16）按名字取权重。三方只要都遵守同一套 <span class="mono">LLM_TENSOR_NAMES</span>，就能严丝合缝地对上。</p>
    <p>正因为有这套约定，"权重"和"代码"才解耦了：你换个工具导出、张量在文件里顺序不同，都不影响——只要名字对得上，建图就能取到对的料。名字成了模型各部件的"身份证"，比"第几个张量"这种脆弱的下标稳得多。</p>
    <p>它也呼应了 L13 的自描述精神：模型不光自带超参、词表，连"每块权重是哪个部件"都用名字标得明明白白。读懂了命名约定，你再去看任何模型的张量列表，都能<strong>一眼认出</strong>哪个是第几层的什么——这是读 llama.cpp 模型的一项基本功。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llm_arch</span> 由 GGUF 的 <span class="mono">general.architecture</span> 选定（<span class="mono">"llama"</span> -&gt; <span class="mono">LLM_ARCH_LLAMA</span>），决定后续用哪套约定与建图。</li>
    <li><span class="mono">llama_hparams</span> 给规格（n_embd / n_layer() / n_head(il) ...）；<span class="mono">n_layer()</span>/<span class="mono">n_head(il)</span> 是<strong>方法</strong>（按层），<span class="mono">n_vocab</span> <strong>不在</strong>其中（来自 vocab，L20）。</li>
    <li>张量按 <span class="mono">LLM_TENSOR_NAMES</span> 命名（<span class="mono">token_embd</span>/<span class="mono">blk.N.attn_q</span>/<span class="mono">output_norm</span>），由 <span class="mono">LLM_TN</span> 的 <span class="mono">tn()</span> 拼出。</li>
    <li>三者咬合 = 自描述兑现：选 arch、填模板读超参、按名字对张量，"一堆张量"成"具体模型"。</li>
    <li>加新架构 ≈ 往这几张表加几行 + 写一份建图（L16），引擎主干不动。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"模型长什么样"编码成<strong>三张表</strong>——架构名表、键名表、张量名表——于是一套引擎代码能读懂几十种架构。这是典型的<strong>表驱动</strong>设计：把"会变的部分"（不同模型的差异）集中进数据（表），让"不变的机制"（读取、建图、执行）留在代码里。
  它和 L05/L12 那条"结构不变、可换的部分集中起来"的思路一脉相承——只不过这次可换的不是数据类型，而是<strong>整个模型架构</strong>。读懂了这一课，你就明白了 llama.cpp"海纳百川"的底层套路。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson the loader read the weights into a "name-indexed tensor list", but how do those tensors know which <strong>architecture</strong> they belong to (llama? qwen2?), and which parts each layer has? This lesson covers two things: <span class="mono">llama-arch</span>
(architecture identity + key-name convention + tensor-name convention - three tables) and <span class="mono">llama-hparams</span> (how many layers, how wide, how many heads per layer) - together the <strong>blueprint</strong> that turns "a pile of tensors" into "a concrete, graph-able model".
</p>
<p style="color:var(--muted);margin-top:.4rem">Why a whole lesson on these two? Because they are the secret behind llama.cpp reading <strong>dozens of models with one codebase</strong>. Both are transformers, yet the difference between llama and qwen2 essentially hides in "which conventions, how many layers/wide, how tensors are named".
Get this lesson and you understand how a new model "plugs into" the engine.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Architecture and hyperparameters are like a <strong>blueprint + spec sheet</strong>: the blueprint (<span class="mono">llm_arch</span> + tensor naming convention) says "which kind of building, which parts per floor, what each is named"; the spec sheet (<span class="mono">hparams</span>) gives the dimensions -
  "how many floors, how wide, how many attention heads per floor". Last lesson's loader is like a stack of labeled materials ready to go, and this lesson's blueprint and spec sheet tell it <strong>which part to fetch by what name, and at what size to assemble</strong>.
</div>

<h2>Architecture identity: llm_arch</h2>
<p>It all starts with one question: given a model, how do you know it is llama or qwen2? The answer is in that GGUF metadata key from L13, <span class="mono">general.architecture</span>. It is a string (e.g. <span class="mono">"llama"</span>); the loader reads it, looks it up in a name table,
and gets the matching <strong>architecture enum</strong> <span class="mono">LLM_ARCH_LLAMA</span>. Once that enum is fixed, which key names, which tensor names, and which graph builder (L16) to use are all fixed.</p>
<div class="flow">
  <div class="node"><div class="nt">general.architecture</div><div class="nd">string in GGUF<br>= "llama"</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">LLM_ARCH_LLAMA</div><div class="nd">look up LLM_ARCH_NAMES<br>get the arch enum</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">one set of conventions</div><div class="nd">KV keys / tensor names / graph(L16)</div></div>
</div>
<p>In source (simplified from <span class="mono">src/llama-arch.{h,cpp}</span>):</p>
<pre class="code"><span class="cm">// simplified from src/llama-arch.h / src/llama-arch.cpp</span>
<span class="kw">enum</span> llm_arch { LLM_ARCH_LLAMA, LLM_ARCH_QWEN2, <span class="cm">/* ... dozens ... */</span> };
<span class="cm">// LLM_ARCH_NAMES: enum &lt;-&gt; string</span>
{ LLM_ARCH_LLAMA, <span class="st">"llama"</span> }, { LLM_ARCH_QWEN2, <span class="st">"qwen2"</span> }, <span class="cm">/* ... */</span></pre>
<p>This <span class="mono">LLM_ARCH_NAMES</span> table is a two-way bridge: on write (L02's conversion script) it turns the arch enum into a string written into GGUF; on read it turns the string back into an enum. A tiny string is the <strong>master switch for everything downstream</strong> -
change it and the keys the loader seeks and the function used to build the graph all change with it. So you can read <span class="mono">general.architecture</span> as the model telling the engine: "please treat me by the llama rules."</p>
<p>Why an enum rather than string comparisons everywhere? Because an enum is fast, hard to mistype, and can index a <span class="mono">switch</span> / a table. The many "branch by architecture" decisions in llama.cpp - which hyperparameters to read, how tensors are named, how the graph is assembled - all route off this enum.
Collapsing "who am I" into one enum value is what lets all the later "treat each architecture differently" code stay tidy.</p>
<p>Get a feel for this mechanism's capacity: llama.cpp already supports <strong>dozens</strong> of architectures - llama, qwen, mistral, phi, gemma, deepseek, stablelm... - all distinguished by this one <span class="mono">llm_arch</span> enum.
New models keep appearing, but the vast majority are transformer variants with limited differences; so "one more architecture" is, to the engine, often just one more enum value and a few more table rows. That capacity of "one enum holding up an ecosystem" is the power of table-driven design.</p>
<p>Beyond <span class="mono">general.architecture</span>, the GGUF header carries a set of generic <span class="mono">general.*</span> metadata (mentioned in L13): model name, overall quantization level, quantization version, and so on. Together with the architecture enum, they form a complete self-introduction of "what this model even is".
What the loader reads first is exactly this generic info plus the architecture-specific hyperparameters - the former recognizes "who", the latter measures "how big".</p>
<p>This "string &lt;-&gt; enum" bridge also explains one thing L02's conversion script does: when converting a HuggingFace model to GGUF, the script must decide which architecture it is and write the matching string (e.g. <span class="mono">"llama"</span>) into <span class="mono">general.architecture</span>.
Writer and reader share the same <span class="mono">LLM_ARCH_NAMES</span>, so write and read mesh perfectly. This is also why a brand-new architecture sometimes needs support added to llama.cpp first, before the conversion script recognizes and can convert it.</p>

<h2>Hyperparameters: llama_hparams</h2>
<p>Knowing the architecture, you still need the <strong>concrete dimensions</strong>: is this llama 7B or 70B? How many layers, how wide, how many heads per layer? These numbers are the <strong>hyperparameters</strong>, held in <span class="mono">llama_hparams</span>, all read from the GGUF KVs by last lesson's <span class="mono">get_key</span>.</p>
<table class="t">
  <tr><th>hyperparameter</th><th>meaning</th></tr>
  <tr><td><span class="mono">n_embd</span></td><td>hidden dimension (how wide a token vector is)</td></tr>
  <tr><td><span class="mono">n_layer()</span></td><td>layer count (how many transformer blocks)</td></tr>
  <tr><td><span class="mono">n_head(il)</span></td><td>attention head count of layer il</td></tr>
  <tr><td><span class="mono">n_head_kv(il)</span></td><td>KV head count of layer il (&lt; n_head under GQA)</td></tr>
  <tr><td><span class="mono">n_ff(il)</span></td><td>FFN intermediate dim of layer il</td></tr>
  <tr><td><span class="mono">rope_freq_base_train</span></td><td>RoPE frequency base (position encoding, L16)</td></tr>
</table>
<pre class="code"><span class="cm">// simplified from src/llama-hparams.h</span>
<span class="kw">struct</span> llama_hparams {
    uint32_t n_embd;  uint32_t n_ctx_train;  float rope_freq_base_train;  float f_norm_rms_eps;
    std::array&lt;uint32_t, LLAMA_MAX_LAYERS&gt; n_head_arr, n_head_kv_arr, n_ff_arr; <span class="cm">// stored per layer</span>
    uint32_t n_layer() const;  uint32_t n_head(uint32_t il = 0) const;        <span class="cm">// accessors, not fields!</span>
};</pre>
<p>There are two <strong>easy traps</strong> here, worth memorizing. First: <span class="mono">n_layer()</span>, <span class="mono">n_head(il)</span> are not plain fields but <strong>accessor methods</strong> (note the parentheses). Why? Because in modern architectures different layers may have different head counts or attention types (GQA, sliding-window layers),
so head counts are stored per layer in arrays like <span class="mono">n_head_arr</span>, fetched with a layer index <span class="mono">il</span>. Treat it as a constant field and you will eventually be wrong.</p>
<p>Second trap: <span class="mono">n_vocab</span> (vocab size) is <strong>not</strong> in <span class="mono">llama_hparams</span>! It belongs to the tokenizer, from <span class="mono">llama_vocab::n_tokens()</span> (L20). Many assume vocab size is a hyperparameter and then search hparams in vain.
Remember: hparams govern "the shape of the network"; the vocab is a separate matter, owned by the tokenizer.</p>
<p>Once read, these hyperparameters become the inference's "size baseline": graph-building (L16) stacks layers by <span class="mono">n_layer()</span>, splits heads by <span class="mono">n_head(il)</span>, sets matrix shapes by <span class="mono">n_embd</span>;
the KV cache (L19) sizes itself by layer count and KV head count. In short, hparams are the set of numbers that turn "an abstract transformer" into "this particular model".</p>
<p>While we are at it, untangle "parameter count" from hyperparameters. The 7B, 70B we casually say refers to the <strong>total count</strong> of floats in the model's weights (7 billion, 70 billion); and that total is computed from the hyperparameters - roughly <span class="mono">n_layer</span> x the sum of each layer's weight-matrix sizes,
where each matrix's size is set by <span class="mono">n_embd</span>, <span class="mono">n_ff</span>, etc. So hyperparameters are not isolated numbers; together they <strong>determine how big the model is</strong>. Read the hyperparameters and you can estimate a model's VRAM appetite from just a few numbers.</p>
<p>Plenty of hyperparameters the table omits each have their use: <span class="mono">n_ctx_train</span> is the context length the model was trained at (a reference ceiling for how long a context you can open); <span class="mono">f_norm_rms_eps</span> is the small constant in RMSNorm that avoids divide-by-zero (used by L11's normalization); <span class="mono">n_rot</span> is the number of dimensions RoPE actually rotates.
These look trivial, yet each is used precisely by some operator at graph time (L16) - miss one or get one wrong, and what you compute is no longer this model.</p>
<p>One more word on that default <span class="mono">il = 0</span>: it makes the common "same for every layer" model effortless - pass no index, and it defaults to layer 0's (i.e. every layer's) value. So although the interface is "fetch per layer", it is not verbose for honest homogeneous models.
A considerate design: it can express the complex case without cluttering the simple one.</p>
<p>Worth specially remembering is what <span class="mono">n_head_kv</span> means: it directly decides how big the KV cache is. GQA's idea is to let several Q heads <strong>share</strong> one set of K/V, so KV heads are far fewer than Q heads, and the KV cache shrinks several-fold (L19 covers this).
So reading a model's hyperparameters, the ratio of <span class="mono">n_head</span> to <span class="mono">n_head_kv</span> nearly tells you "how friendly this model is to long context". A seemingly minor hyperparameter, with a big VRAM-and-speed account behind it.</p>

<h2>Tensor naming convention</h2>
<p>The last piece: how do the loader's tensors line up <strong>by name</strong> with the model structure? Via a naming convention. Every tensor in the file has a name, and these names are not arbitrary - they follow the templates defined in <span class="mono">LLM_TENSOR_NAMES</span>.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>tensor-name templates (LLM_TENSOR_NAMES)</b>: named by part + layer index; graph-building fetches weights by name</div>
  <div class="cells"><span class="lab">name</span><span class="cell hl">token_embd</span><span class="cell">blk.0.attn_q</span><span class="cell">blk.0.ffn_gate</span><span class="cell">output_norm</span></div>
</div>
<p>These names reveal the pattern: <span class="mono">token_embd</span> is the token-embedding table (the first layer), <span class="mono">blk.0.attn_q</span> is layer 0's attention Q projection, <span class="mono">blk.0.ffn_gate</span> is layer 0's FFN gate, <span class="mono">output_norm</span> is the final output norm.
The <span class="mono">%d</span> in <span class="mono">blk.%d</span> is filled with the layer index - exactly where <span class="mono">n_layer()</span> earns its keep: loop 0 to n_layer(), and per layer build that layer's tensor names from the templates.</p>
<p>Names are built by a small constructor called <span class="mono">LLM_TN</span> (its <span class="mono">tn(...)</span> call assembles "part + suffix + layer index" into a full tensor name). At graph time (L16), each time a weight is needed, this constructor builds the name and looks it up in last lesson's <span class="mono">weights_map</span> -
which <strong>strings together</strong> the "loading" and "graph" lessons via names: the loader stores by name, the graph fetches by name, aligned through the shared <span class="mono">LLM_TENSOR_NAMES</span> convention.</p>
<p>This also answers last lesson's cliffhanger - why <span class="mono">weights_map</span> is keyed by name. Because a name is a <strong>stable contract</strong>: a different export tool, a different tensor order, no problem - as long as the naming convention holds, graph-building can always fetch the exact weight it wants.
That layer of naming fully decouples "where a weight physically sits" from "which part it logically is".</p>
<p>List a transformer layer's tensor names in full and the pattern is clearer still: the attention part has four projections <span class="mono">attn_q</span>/<span class="mono">attn_k</span>/<span class="mono">attn_v</span>/<span class="mono">attn_output</span> plus an <span class="mono">attn_norm</span>; the FFN part has three matrices <span class="mono">ffn_gate</span>/<span class="mono">ffn_up</span>/<span class="mono">ffn_down</span> plus an <span class="mono">ffn_norm</span>.
Every layer replicates this template, prefixed with <span class="mono">blk.&lt;index&gt;.</span>. Understand this "what weights a layer has" list and you understand all the learnable parameters of one transformer block.</p>
<p>Why dot-separated hierarchical names (<span class="mono">blk.0.attn_q.weight</span>)? Because it naturally forms a <strong>hierarchy tree</strong>: under <span class="mono">blk</span> are the layers, under a layer the parts, under a part weight/bias. Such naming is both clear and convenient for prefix matching in bulk -
to find all of layer 0's weights, match the <span class="mono">blk.0.</span> prefix. A good naming convention is not just "giving a name"; it encodes structural information into the name itself.</p>
<p>This naming is also <strong>cross-architecture</strong>: whether llama or qwen2, layer 0's attention Q projection is named <span class="mono">blk.0.attn_q</span>. Because everyone shares one naming set, generic tensor tools (quantization, conversion, visualization) can process any model without caring about architecture.</p>
<p>A bit more <span class="mono">LLM_TN</span> detail: the full name it builds usually carries a suffix too, distinguishing <span class="mono">.weight</span> from <span class="mono">.bias</span> - a part may have a weight and possibly a bias. So <span class="mono">tn(LLM_TENSOR_ATTN_Q, "weight", il)</span> builds <span class="mono">blk.il.attn_q.weight</span>.
Handing "part-name template + suffix + layer index" to one constructor avoids the errors of hand-writing strings everywhere, and means "changing a naming rule" touches just one place.</p>
<p>By the way: some tensors are <strong>optional</strong> - many modern models' linear layers have no bias, so the <span class="mono">.bias</span> tensor simply does not exist in the file. At graph time, the architecture knows "which tensors this layer should have", and missing optional ones are skipped.
The naming convention plus knowledge of "which are required, which optional" together fully describe an architecture's tensor makeup.</p>

<h2>How self-description cashes out at the architecture layer</h2>
<p>Connect the three things and L13's "<strong>self-description</strong>" cashes out fully at the architecture layer: <span class="mono">general.architecture</span> selects the arch; <span class="mono">%s</span> keys (like <span class="mono">llama.block_count</span>) fill the template with the arch name and are read by <span class="mono">get_key</span> into hyperparameters;
tensors line up by their <span class="mono">LLM_TENSOR_NAMES</span> names. Once the three conventions mesh, the loader's "pile of tensors" becomes "a concrete model with names, shapes, and sizes", ready to hand to L16 for graph-building.</p>
<p>What is worth savoring is the <strong>table-driven</strong> flavor of this design: architecture is a name table, keys are a key-name table, tensors are a tensor-name table. The engine's trunk hard-codes no single model; it "acts by the tables". So supporting a new model is mostly not editing the engine, but <strong>adding a few rows to these tables + writing one graph builder</strong> (L16).</p>
<p>Imagine if it were <strong>not</strong> table-driven: every new model would mean a pile of <span class="mono">if (arch == "llama") ... else if (arch == "qwen2") ...</span> branches in the engine trunk - reading hyperparameters, naming tensors, building graphs, all needing edits everywhere. With enough models, those branches weave into a web no one dares touch.
Table-driven design gathers these differences from "ifs scattered through the code" into "a few centralized tables + one graph file", so the trunk code always reads as "act by the tables" - clean and clear.</p>
<p>These three steps - "<strong>recognize the architecture -&gt; read hyperparameters -&gt; fetch tensors by name</strong>" - are really the inescapable skeleton of any "general model loader": first figure out what it is, then measure how big, finally slot the parts into place. llama.cpp does these three exceptionally cleanly, the engineering basis for it catching up to one new model after another so quickly.
Next lesson, carrying this "blueprint + spec sheet + parts list", we actually assemble them into a runnable forward compute graph.</p>
<p>Connecting this lesson with the last reveals a clear through-line: L14's loader turns bytes into "named tensors", and L15's arch/hparams give those tensors "a blueprint and spec". By here, the model has gone from "a file on disk" to "an in-memory object with names, knowing how many layers and how wide it is" -
one step short: actually assembling these parts by the blueprint into a runnable network. That is exactly the next lesson.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why is n_head written as n_head(il) with a layer index? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because in modern architectures different layers may have different attention configs. The most common is <strong>GQA</strong> (grouped-query attention): many Q heads, few KV heads, so <span class="mono">n_head_kv(il)</span> is smaller than <span class="mono">n_head(il)</span>, used to shrink the KV cache (L19).</p>
    <p>Some architectures are <strong>hybrid</strong>: some layers use full attention, others sliding windows, with different head counts and window sizes per layer. To express this "per-layer difference", the cleanest way is to store these numbers as <strong>per-layer arrays</strong> (<span class="mono">n_head_arr</span> etc.), fetched with the layer index <span class="mono">il</span>.</p>
    <p>So <span class="mono">n_head(il)</span> is a method, not a field - behind it, a value is pulled from an array by layer index. For most honest homogeneous models every layer is the same and <span class="mono">il</span> defaults to 0; but designing the interface to fetch per layer is what accommodates those "per-layer different" new architectures. A classic "leave room for generality" trade-off.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What does adding a new architecture touch? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Roughly four steps, mostly "filling tables": (1) add an entry to <span class="mono">enum llm_arch</span> and its name to <span class="mono">LLM_ARCH_NAMES</span>; (2) implement this architecture's hyperparameter reading (read its specific hyperparameters from the KVs into hparams); (3) register the tensors it uses in the tensor-name table; (4) write one graph builder (L16, <span class="mono">src/models/&lt;arch&gt;.cpp</span>).</p>
    <p>The key is that <strong>step (4) is often light</strong>: because the vast majority of architectures use the same building blocks (attention, FFN, normalization), a new architecture is mostly "a different arrangement", reusing the existing <span class="mono">build_attn</span>/<span class="mono">build_ffn</span> (L16). Only a truly novel structure needs one or two new operators (L11).</p>
    <p>This is why llama.cpp keeps up with the endless stream of new models: most "new architectures" are, in engineering terms, "<strong>fill a few tables + reuse blocks</strong>", with the engine trunk untouched. Folding "model diversity" into tables and graph files, and keeping "the invariant machinery" in the trunk - that is where this design saves the most effort.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Why is the tensor naming convention so important? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because it is a <strong>three-party contract</strong>. The writer (L02's conversion script) writes weights into GGUF by these names; the reader (L14's loader) builds <span class="mono">weights_map</span> by name; the graph builder (L16) fetches weights by name. As long as all three honor the same <span class="mono">LLM_TENSOR_NAMES</span>, they line up perfectly.</p>
    <p>Because of this convention, "weights" and "code" are decoupled: export with a different tool, a different tensor order in the file - none of it matters, as long as names match, graph-building fetches the right material. A name becomes each part's "ID card", far steadier than the fragile "which-th tensor" index.</p>
    <p>It also echoes L13's self-description spirit: a model carries not only its hyperparameters and vocab but even labels "which part each weight is" clearly by name. Read the naming convention and, looking at any model's tensor list, you can <strong>recognize at a glance</strong> which is what of which layer - a basic skill for reading llama.cpp models.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llm_arch</span> is selected by GGUF's <span class="mono">general.architecture</span> (<span class="mono">"llama"</span> -&gt; <span class="mono">LLM_ARCH_LLAMA</span>), deciding the conventions and graph builder used.</li>
    <li><span class="mono">llama_hparams</span> gives the spec (n_embd / n_layer() / n_head(il) ...); <span class="mono">n_layer()</span>/<span class="mono">n_head(il)</span> are <strong>methods</strong> (per layer), and <span class="mono">n_vocab</span> is <strong>not</strong> among them (from vocab, L20).</li>
    <li>Tensors are named by <span class="mono">LLM_TENSOR_NAMES</span> (<span class="mono">token_embd</span>/<span class="mono">blk.N.attn_q</span>/<span class="mono">output_norm</span>), built by <span class="mono">LLM_TN</span>'s <span class="mono">tn()</span>.</li>
    <li>The three meshing = self-description cashed out: pick arch, fill templates to read hyperparameters, line up tensors by name; "a pile of tensors" becomes "a concrete model".</li>
    <li>Adding a new architecture ~= add a few rows to these tables + write one graph builder (L16), engine trunk untouched.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Encoding "what a model looks like" into <strong>three tables</strong> - an arch-name table, a key-name table, a tensor-name table - lets one engine codebase read dozens of architectures. This is classic <strong>table-driven</strong> design: fold "the parts that vary" (differences between models) into data (tables), and keep "the invariant machinery" (reading, graph-building, execution) in code.
  It is of a piece with that L05/L12 idea of "structure fixed, the swappable parts gathered together" - only this time what is swappable is not a data type but <strong>an entire model architecture</strong>. Get this lesson and you understand llama.cpp's underlying recipe for "taking in all rivers".
</div>
""",
}

LESSON_16 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
有了加载好的权重（L14）和架构超参（L15），终于可以<strong>把它们接成一张真正能算的前向计算图</strong>了。这一课讲 <span class="mono">llama-graph</span>：它提供 <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span>
这些"标准件"，每个架构在 <span class="mono">src/models/&lt;arch&gt;.cpp</span> 里决定"按什么顺序把它们拼起来"，最终产出一张第三部分讲过的 <span class="mono">ggml_cgraph</span>（L09）。
</p>
<p style="color:var(--muted);margin-top:.4rem">这一课是<strong>承上启下的枢纽</strong>：上面（L14/L15）把模型整理成了"有名有姓、有形有状"的张量集合，下面（L09/L10/L11）是 ggml 怎么建图、怎么执行、怎么算每个算子。这一课正是把两端接起来的那道桥——
它把"一个具体模型"翻译成"一张 ggml 计算图"。读懂它，你就看清了 llama.cpp 是怎么把一堆权重变成"能跑"的。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  建图像照着图纸（L15 架构）用<strong>标准件搭模型</strong>：<span class="mono">build_attn</span>（搭一套注意力）、<span class="mono">build_ffn</span>（搭一套前馈）、<span class="mono">build_norm</span>（搭一层归一化）就是预制好的标准件；
  <span class="mono">src/models/&lt;arch&gt;.cpp</span> 是"这种楼的拼装说明书"，告诉你这些件按什么顺序、用哪些权重拼。而且拼出来的<strong>不是结果，而是一张待执行的图</strong>——就像搭好的不是已通电的电路，而是一张电路图，真正通电（计算）是 L10 的事。
</div>

<h2>谁来建图：从 build_graph 说起</h2>
<p>建图的总入口是 <span class="mono">llama_model::build_graph</span>。它本身很薄，主要做一件事：<strong>把活派发给当前架构</strong>。因为不同架构（llama、qwen2…）的前向流程不一样，所以真正的建图逻辑放在每个架构<strong>各自的文件</strong> <span class="mono">src/models/&lt;arch&gt;.cpp</span> 里。</p>
<div class="flow">
  <div class="node"><div class="nt">llama_model::build_graph</div><div class="nd">总入口(薄)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">build_arch_graph</div><div class="nd">虚函数 -&gt; src/models/&lt;arch&gt;.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llm_graph_context 的积木</div><div class="nd">build_attn / build_ffn ...</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_gf()</div><div class="nd">= ggml_cgraph(L09)</div></div>
</div>
<p>落到源码（简化自 <span class="mono">src/llama-model.cpp</span>）：</p>
<pre class="code"><span class="cm">// 简化自 src/llama-model.cpp</span>
ggml_cgraph * llama_model::<span class="fn">build_graph</span>(<span class="kw">const</span> llm_graph_params &amp; p) <span class="kw">const</span> {
    <span class="kw">auto</span> llm = <span class="fn">build_arch_graph</span>(p);   <span class="cm">// 虚: 派发到 src/models/&lt;arch&gt;.cpp</span>
    <span class="cm">// ... build_pooling / build_dense_out ...</span>
    <span class="kw">return</span> llm-&gt;res-&gt;<span class="fn">get_gf</span>();        <span class="cm">// ggml_cgraph(L09)</span>
}</pre>
<p>这里有两层抽象要分清：<span class="mono">build_graph</span> 是<strong>稳定的总入口</strong>（不管什么架构，外面都调它）；<span class="mono">build_arch_graph</span> 是<strong>虚函数</strong>，每个架构重写自己那一份。这正是 L15 表驱动思路的延续——
"哪种架构"决定调哪份建图代码。而所有架构的建图，都基于一个共同的基类 <span class="mono">llm_graph_context</span>，它身上挂着 <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span> 这些人人可用的积木方法。</p>
<p>为什么把建图按架构拆成一个个文件，而不是写成一个巨大的 <span class="mono">switch</span>？因为每种架构的前向多少有点不同，分开写各自清爽、互不干扰；而<strong>共性</strong>（注意力、前馈、归一化怎么搭）则抽到基类的积木里复用。
这种"差异分文件、共性进基类"的组织，让加一个新架构基本只用新写一个 <span class="mono">src/models/&lt;arch&gt;.cpp</span>，不必碰别人。</p>
<p>这里也顺势澄清 <span class="mono">build_graph</span> 和 ggml 的关系：它<strong>不是</strong> ggml 的一部分，而是 llama 层站在 ggml 之上写的"组装逻辑"。ggml（L08-L11）提供张量、算子、建图原语；<span class="mono">build_graph</span> 用这些原语，按 transformer 的结构拼出一张具体的图。
所以这一课本质是在讲"<strong>怎么用 ggml 这套积木，搭出一个真正的大模型</strong>"。</p>
<p>那 <span class="mono">build_graph</span> 什么时候被调用？答案是<strong>每一步推理都调一次</strong>——下一课会讲的 <span class="mono">llama_decode</span>，内部第一件事就是为这一步搭出计算图。听起来很费：每生成一个 token 都要重搭一次图？
其实不然，图的"结构"很轻（只是一串算子的声明、不含计算），搭起来很快；而且对形状相同的步骤，这张图还能被复用，不必每次从头来。</p>
<p>传给 <span class="mono">build_graph</span> 的 <span class="mono">llm_graph_params</span> 里，装着搭这一步图所需的上下文：这一步要处理哪些 token（来自 L18 的 batch）、KV cache 当前状态、用哪个后端等等。换句话说，<span class="mono">build_graph</span> 不是凭空搭图，而是<strong>针对"这一步要算什么"</strong>搭出恰好够用的图。
这也是为什么 prefill（一次算整段 prompt）和 decode（一次算一个 token）虽然用同一套建图代码，搭出的图大小却不同。</p>
<p>代码里那行 <span class="mono">build_pooling / build_dense_out</span> 注释也值得一提：除了主体的 N 层 block，建图还会按需接上一些"收尾"步骤——比如做 embedding 任务时的池化、某些模型额外的输出层。这些是<strong>可选的尾巴</strong>，普通文本生成多半用不到，但它们和主体共用同一套建图框架，按架构和任务接进同一张图。</p>

<h2>一层 transformer 怎么搭</h2>
<p>模型的主体是 N 个一模一样的 transformer block 叠起来。看清<strong>一层</strong>怎么搭，就看懂了整个前向。一层的骨架很固定：归一化 -&gt; 注意力 -&gt; 残差 -&gt; 归一化 -&gt; 前馈 -&gt; 残差。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>build_norm</h4><p>对输入做 RMSNorm，把数值稳到合理范围（L11 的归一化算子）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>build_attn</h4><p>一整套注意力：Q/K/V 投影 + rope 注入位置 + 读写 KV cache + softmax + 输出投影（L11/L04/L19）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>残差相加</h4><p>把注意力输出加回输入（残差连接），让信息和梯度都好流动。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>build_norm + build_ffn</h4><p>再归一化一次，然后前馈网络（gate/up/down 三个矩阵乘，L11）。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>残差相加</h4><p>把前馈输出加回去，得到这一层的输出，喂给下一层。</p></div></div>
</div>
<p>把这套骨架写成伪代码，几乎就是 <span class="mono">src/models/llama.cpp</span> 里循环体的样子：</p>
<pre class="code"><span class="cm"># 伪代码: 一层的拼法(对应 src/models/llama.cpp 的循环体)</span>
cur  = <span class="fn">build_norm</span>(inpL, attn_norm_w)            <span class="cm"># RMSNorm(L11)</span>
cur  = <span class="fn">build_attn</span>(inp_attn, cur, wq,wk,wv,wo)    <span class="cm"># Q/K/V + rope + KV + softmax(L11/L04/L19)</span>
inpL = cur + inpL                              <span class="cm"># 残差</span>
cur  = <span class="fn">build_norm</span>(inpL, ffn_norm_w)
cur  = <span class="fn">build_ffn</span>(cur, w_gate, w_up, w_down)      <span class="cm"># 前馈(L11 mul_mat)</span>
inpL = cur + inpL                              <span class="cm"># 残差 -> 下一层</span></pre>
<p>循环 0 到 <span class="mono">n_layer()</span>（L15 那个访问器方法），每层都拼这么一套，权重就用 L15 的命名约定按名字取（<span class="mono">blk.il.attn_q.weight</span> 等）。可以看到，"建图"在这一层非常机械——就是把固定的积木按固定顺序、喂以每层各自的权重，连成一长串算子。</p>
<p>这也把前面三课漂亮地收束了：L14 按名字备好张量、L15 给出每层该取哪些权重和多少层，L16 在这里把它们按 transformer 的结构真正拼起来。三课合起来，回答的就是"<strong>一个模型怎么从一堆权重变成一张能算的图</strong>"。</p>
<p>那两处<strong>残差相加</strong>（cur + inpL）别看简单，却是深层 transformer 能训练、能工作的关键之一：它让每一层在"原始输入"的基础上只学一个"增量"，信息和梯度都能顺着这条捷径直通到底，不至于在几十层里衰减殆尽。建图时它就是一个普通的 ggml 加法算子，但其意义远不止一次加法。</p>
<p>整个前向并非只有重复的层。<strong>开头</strong>有一步把 token id 查成词向量（<span class="mono">token_embd</span> 那张表，图输入之一）；<strong>结尾</strong>在最后一层之后，还有一次 <span class="mono">output_norm</span> 归一化、和一次投影到词表大小的 <span class="mono">output</span>（算出 logits，L17）。
所以完整的图是"输入嵌入 -&gt; N 层 block -&gt; 输出归一 -&gt; 投影出 logits"，中间那 N 层才是我们重点拆的对象。</p>
<p>值得点明 prefill 和 decode 在建图上的关系：两者用<strong>同一套</strong>建图代码，区别只在喂进去的 batch（L18）——prefill 一次喂整段 prompt 的多个 token、decode 一次只喂一个新 token。图的"形状"随 token 数变，但"结构"（每层怎么拼）完全一样。
这正是统一建图的好处：一套逻辑，既管"首次把 prompt 过一遍"，又管"之后逐字生成"。</p>
<p>还要留意一点：建图代码里<strong>看不到具体的数值</strong>。<span class="mono">build_attn</span>、<span class="mono">build_ffn</span> 操作的全是"还没算的张量"——它们只是在说"把这个权重和那个输入做矩阵乘，结果叫 cur"。真正的浮点数要等 L10 执行时才填进去。
所以读建图代码，你读到的是<strong>数据流的形状</strong>，而不是数据本身——这也是 L09 惰性建图最直观的体感。</p>
<p>退一步看，这一整张前向图其实就是一个<strong>有向无环图</strong>（DAG）：token 向量从输入叶子流入，经过一层层 block 的算子变换，最后流到 logits。每个算子是图上一个节点、箭头表示"谁喂给谁"。
L09 已经讲过这种图的本质，这一课只是让你看到：原来一个真实大模型的前向，落到图上就是这么一张结构清晰、层层堆叠的 DAG。</p>

<h2>复用积木与图输入</h2>
<p>支撑这套拼装的，是 <span class="mono">llm_graph_context</span> 上的一批<strong>复用积木</strong>和<strong>图输入</strong>。积木是 <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span> 这些方法；图输入是把"外部数据"接进图的入口。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">图输入</span><span class="name">llm_graph_input_*</span></div><div class="ld">把外部数据接进图：embd（词向量）· pos（位置）· attn_kv（KV cache）</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">积木</span><span class="name">build_attn / build_ffn / build_norm</span></div><div class="ld">把权重 + 输入拼成一段子图，内部都是 L11 的算子</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">产物</span><span class="name">llm_graph_result -&gt; ggml_cgraph</span></div><div class="ld">所有积木串起来，get_gf() 交出最终的图(L09)</div></div>
</div>
<p>图输入（<span class="mono">llm_graph_input_*</span>）是个容易被忽略却很关键的概念。一张计算图光有"算子"还不够，还得有"入口"——token 的词向量从哪进来、每个 token 的位置从哪来、KV cache 接在哪。
这些就是各种 <span class="mono">build_inp_embd</span>/<span class="mono">build_inp_pos</span> 建出来的输入节点。它们是图的"叶子"（L09 讲过的 leafs），每步推理把新数据填进去，图就能算出新结果。</p>
<p>而所有积木产出的，都是 L11 讲的 ggml 张量（<span class="mono">mul_mat</span>、<span class="mono">soft_max_ext</span> 等算子的输出）。一个 <span class="mono">build_attn</span> 调用，内部就是十几个 ggml 算子按注意力数学（L04）串成的一小段子图。
把许多这样的子图首尾相连，就长成了整个模型的前向图——这正是 L09 说的"算子串成图"，只不过这里是站在 llama 层、按 transformer 结构有组织地串。</p>
<p>顺带看一眼 <span class="mono">build_ffn</span> 内部：现代 llama 类模型的前馈不是简单的"一升一降"，而是 <strong>SwiGLU</strong> 式的——<span class="mono">gate</span> 和 <span class="mono">up</span> 两个矩阵各把输入投影一次，<span class="mono">gate</span> 那路过一个激活函数后与 <span class="mono">up</span> 逐元素相乘，再由 <span class="mono">down</span> 投影回去。
这就是为什么一层 FFN 有 gate/up/down 三个权重矩阵（L15 命名约定里见过）。<span class="mono">build_ffn</span> 把这套固定套路封好，建图时一句话搞定。</p>
<p>再说说图输入和"叶子"的关系。L09 讲过，图里分两类节点：算出来的<strong>节点</strong>和不计算、只被读取的<strong>叶子</strong>。权重是叶子（加载时就备好了，L14），而图输入（词向量、位置）也是叶子——只不过它们的数据是<strong>每步填新的</strong>。
建图时把这些叶子的位置占好，执行时把当前这一步的数据填进去，同一张图就能算出不同的结果。</p>
<p>你会注意到 <span class="mono">build_attn</span> 有<strong>好几个重载</strong>。为什么？因为注意力有不少变体：要不要用 KV cache（prefill 的某些路径不用、decode 必用）、是标准多头还是 GQA、用不用滑动窗口……与其每种各写一遍完整注意力，
不如把"公共骨架 + 可选差异"做成几个重载，让各架构按需挑用。这又是一处"把差异收进可选项、把共性沉淀成积木"的体现。</p>
<p>图输入被做成一族类（<span class="mono">llm_graph_input_*</span> 都派生自一个共同接口）也有讲究：不同的输入有不同的"填法"——词向量要按 token id 查表、位置要按当前进度生成、KV 掩码要按因果规则算。
把每种输入的"怎么填"封进各自的类，执行前统一调一遍，图就准备好了。这让"图里需要哪些外部输入"变得可扩展、可组合。</p>

<h2>建图与执行，泾渭分明</h2>
<p>最后强调这一课最重要的一点：<span class="mono">build_graph</span> 只<strong>建</strong>、不<strong>算</strong>。它把算子的 op 和 src 填好（L09 的惰性建图），最后 <span class="mono">get_gf()</span> 交出一张 <span class="mono">ggml_cgraph</span>，至于真正逐节点执行，是 L10 后端的事。</p>
<p>这种"建图归建图、执行归执行"的分离，回报是巨大的：同一张图，能原封不动地跑在 CPU、CUDA、Metal 等天差地别的硬件上（L10 的后端调度），上层的模型逻辑只写一遍。也正因为建图不碰具体计算，
换一个后端、加一种新硬件，都<strong>不用动建图代码</strong>。L16 负责"拼出正确的图"，L10 负责"在某种硬件上把图算快"，两者各司其职，合起来才是完整的推理。</p>
<p>正因为建图只产出"结构"、不含数据，这张图在很多情况下还能被<strong>缓存复用</strong>：连续的 decode 步骤，每步都是"一个新 token"，图的结构一模一样，于是引擎可以复用上一张图的骨架、只换喂进去的输入，省下反复搭图的开销。这是"惰性建图 + 结构与数据分离"带来的又一个红利。</p>
<p>把这一课放回整个推理循环里看：每生成一个 token，<span class="mono">llama_decode</span>（L17）大致就是"<strong>建图（L16）-&gt; 后端执行（L10）-&gt; 得到 logits -&gt; 采样（L21）出下一个 token</strong>"这么一圈。L16 是这圈里"把模型变成可算的图"那一环。
理解了它，你就把"加载好的模型"和"真正跑起来的推理"接上了。下一课，我们就进到 <span class="mono">llama_context</span>，看这一圈是怎么转起来的。</p>
<p>所以这一课真正要带走的，是一个<strong>心智模型</strong>：模型推理 = 按架构把权重拼成一张计算图（L16）+ 在某后端上执行这张图（L10）。拼图的逻辑写一遍、能跑遍所有硬件；这就是 llama.cpp 既轻便又通用的根。把这句话记牢，第四部分后面几课其实都是在它的脉络上继续展开。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么每个架构要单独一个 src/models/&lt;arch&gt;.cpp？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>为了<strong>差异隔离</strong>。各架构的前向流程多少有别：有的注意力带偏置、有的 FFN 用不同激活、有的层间还插了别的东西。把每种架构的"拼法"放进各自的文件，改一个不会波及另一个，读起来也清爽——一个文件就是一种模型的完整前向。</p>
    <p>而真正干活的积木（<span class="mono">build_attn</span> 等）是<strong>共享</strong>的，住在基类 <span class="mono">llm_graph_context</span> 里。所以这些架构文件大多很短：无非是"按这个架构的顺序，调几次共享积木、喂上对的权重"。共性进基类、差异进文件，是这套设计能容纳几十种架构还不乱的关键。</p>
    <p>这也呼应了 L15：加一个新架构，建图这步往往就是<strong>新写一个不长的 <span class="mono">src/models/&lt;arch&gt;.cpp</span></strong>，复用现成积木。只有遇到真正新颖的结构，才需要往基类加一两个新积木、甚至往 ggml 加一两个新算子（L11）。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> build_attn 内部到底做了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>它把 L04 的注意力数学，翻译成 L11 的一串算子。大致是：先用三次 <span class="mono">mul_mat</span> 把输入投影成 Q、K、V；给 Q、K 施加 <span class="mono">rope</span> 注入位置；把这一步的 K、V <strong>写进 KV cache</strong>、再把历史的 K、V <strong>读回来</strong>（L19）。</p>
    <p>接着算注意力分数（Q 和 K 的矩阵乘）、用 <span class="mono">soft_max_ext</span> 加因果掩码并归一成权重、再用一次 <span class="mono">mul_mat</span> 按权重把 V 汇总；最后一次输出投影。一个 <span class="mono">build_attn</span> 调用，就这样把<strong>一整套注意力</strong>拼成了一段子图。</p>
    <p>所以你之前学的东西在这里全用上了：L04 的数学是蓝本、L11 的算子是砖块、L19 的 KV cache 是让它每步只算新 token 的关键。<span class="mono">build_attn</span> 就是把这三者按正确顺序焊到一起的那个"组装工"。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 建图和执行是怎么彻底分开的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>靠 L09 的惰性建图。<span class="mono">build_*</span> 这些积木<strong>不计算</strong>，只创建张量、填好它的 <span class="mono">op</span>（我是哪种算子）和 <span class="mono">src</span>（我的输入是谁）。一圈拼下来，得到的是一张<strong>只描述了"算什么、依赖谁"的图</strong>，里面一个数都还没算。</p>
    <p>然后 <span class="mono">get_gf()</span> 把这张 <span class="mono">ggml_cgraph</span> 交出去，由 L10 的后端按拓扑序逐节点真正计算。建图侧只关心"逻辑结构对不对"，执行侧只关心"在这块硬件上怎么算得快"——两边的关注点完全分开。</p>
    <p>好处是<strong>解耦带来的自由</strong>：模型逻辑（建图）写一遍，就能跑在所有后端上；要支持新硬件，只在执行侧加一个后端，建图代码一行不改。这正是 ggml/llama 这套分层最值钱的地方，也是它能同时跑在你的笔记本 CPU 和数据中心 GPU 上的根本原因。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama_model::build_graph</span> 派发到每架构自己的 <span class="mono">build_arch_graph</span>（<span class="mono">src/models/&lt;arch&gt;.cpp</span>），返回一张 <span class="mono">ggml_cgraph</span>（经 <span class="mono">res-&gt;get_gf()</span>）。</li>
    <li>积木 <span class="mono">build_norm</span>/<span class="mono">build_attn</span>/<span class="mono">build_ffn</span> 是基类 <span class="mono">llm_graph_context</span> 的方法，被各架构复用。</li>
    <li>一层 = norm -&gt; attn（QKV+rope+KV+softmax）-&gt; 残差 -&gt; norm -&gt; ffn -&gt; 残差；循环 <span class="mono">n_layer()</span> 层，按名字取权重（L15）。</li>
    <li>图输入 <span class="mono">llm_graph_input_*</span> 把词向量/位置/KV 接进图，是图的"叶子"（L09）。</li>
    <li><strong>只建不算</strong>（L09 惰性）：<span class="mono">build_graph</span> 拼出图，L10 后端才执行；同一图可换后端跑。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"每种架构怎么前向"写成一份 <span class="mono">src/models/&lt;arch&gt;.cpp</span>，把"怎么算注意力/前馈"沉淀成 <span class="mono">llm_graph_context</span> 的可复用积木——于是新架构只是"用标准件换个拼法"。更妙的是，底层 ggml（L08-L12）<strong>根本不知道</strong>上面跑的是 llama 还是 qwen，
  它只看到一张普通的计算图、照样执行（L10）。模型的多样性收在建图层、计算的通用性留在 ggml 层——这道干净的分界，正是 llama.cpp 既能海纳百川、又能一套引擎跑天下的底层秘密。下一课，我们看这张图被装进 <span class="mono">llama_context</span> 后，怎么真正跑起一步推理。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
With the loaded weights (L14) and the architecture hyperparameters (L15), we can finally <strong>wire them into a forward compute graph that actually computes</strong>. This lesson covers <span class="mono">llama-graph</span>: it provides "standard parts" like <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span>,
each architecture decides "in what order to assemble them" in <span class="mono">src/models/&lt;arch&gt;.cpp</span>, and the result is a <span class="mono">ggml_cgraph</span> from Part 3 (L09).
</p>
<p style="color:var(--muted);margin-top:.4rem">This lesson is the <strong>pivot connecting both sides</strong>: above (L14/L15) the model became a set of tensors "with names, shapes, and sizes"; below (L09/L10/L11) is how ggml builds graphs, executes, and computes each operator. This lesson is the bridge joining the two -
it translates "a concrete model" into "a ggml compute graph". Read it and you see how llama.cpp turns a pile of weights into something "runnable".</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Building the graph is like assembling a model from <strong>standard parts</strong> per the blueprint (L15 architecture): <span class="mono">build_attn</span> (one attention set), <span class="mono">build_ffn</span> (one feed-forward set), <span class="mono">build_norm</span> (one normalization) are the prefab parts;
  <span class="mono">src/models/&lt;arch&gt;.cpp</span> is "the assembly manual for this kind of building", saying in what order and with which weights to assemble them. And what comes out is <strong>not a result but a graph waiting to run</strong> - like building not a powered circuit but a circuit diagram; actually powering it (computing) is L10's job.
</div>

<h2>Who builds the graph: starting from build_graph</h2>
<p>The main entry for graph-building is <span class="mono">llama_model::build_graph</span>. It is itself thin, doing mainly one thing: <strong>dispatching the work to the current architecture</strong>. Because different architectures (llama, qwen2...) have different forward flows, the real graph-building logic lives in each architecture's <strong>own file</strong> <span class="mono">src/models/&lt;arch&gt;.cpp</span>.</p>
<div class="flow">
  <div class="node"><div class="nt">llama_model::build_graph</div><div class="nd">main entry (thin)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">build_arch_graph</div><div class="nd">virtual -&gt; src/models/&lt;arch&gt;.cpp</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llm_graph_context blocks</div><div class="nd">build_attn / build_ffn ...</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">get_gf()</div><div class="nd">= ggml_cgraph(L09)</div></div>
</div>
<p>In source (simplified from <span class="mono">src/llama-model.cpp</span>):</p>
<pre class="code"><span class="cm">// simplified from src/llama-model.cpp</span>
ggml_cgraph * llama_model::<span class="fn">build_graph</span>(<span class="kw">const</span> llm_graph_params &amp; p) <span class="kw">const</span> {
    <span class="kw">auto</span> llm = <span class="fn">build_arch_graph</span>(p);   <span class="cm">// virtual: dispatch to src/models/&lt;arch&gt;.cpp</span>
    <span class="cm">// ... build_pooling / build_dense_out ...</span>
    <span class="kw">return</span> llm-&gt;res-&gt;<span class="fn">get_gf</span>();        <span class="cm">// ggml_cgraph(L09)</span>
}</pre>
<p>Two layers of abstraction to separate here: <span class="mono">build_graph</span> is the <strong>stable main entry</strong> (whatever the architecture, outside code calls it); <span class="mono">build_arch_graph</span> is a <strong>virtual function</strong> each architecture overrides. This continues L15's table-driven idea -
"which architecture" decides which graph code runs. And every architecture's graph-building is based on a shared base class <span class="mono">llm_graph_context</span>, which carries the reusable block methods <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span>.</p>
<p>Why split graph-building into files per architecture rather than one giant <span class="mono">switch</span>? Because each architecture's forward differs somewhat; writing them separately keeps each clean and non-interfering, while the <strong>commonality</strong> (how attention/FFN/norm are built) is lifted into reusable base-class blocks.
This "differences per file, commonality in the base" organization means adding a new architecture is basically writing one new <span class="mono">src/models/&lt;arch&gt;.cpp</span> without touching others.</p>
<p>Let us also clarify <span class="mono">build_graph</span>'s relation to ggml: it is <strong>not</strong> part of ggml, but the "assembly logic" the llama layer writes on top of ggml. ggml (L08-L11) provides tensors, operators, and graph-building primitives; <span class="mono">build_graph</span> uses these primitives to assemble a concrete graph by the transformer structure.
So this lesson is essentially about "<strong>how to use ggml's building blocks to assemble a real large model</strong>".</p>
<p>When is <span class="mono">build_graph</span> called? The answer is <strong>once per inference step</strong> - <span class="mono">llama_decode</span> (next lesson) builds this step's compute graph as its first act. Sounds costly: rebuild a graph for every generated token?
Not really - the graph's "structure" is light (just a chain of operator declarations, no computation), so it builds fast; and for steps of the same shape the graph can be reused, no need to start from scratch each time.</p>
<p>The <span class="mono">llm_graph_params</span> passed to <span class="mono">build_graph</span> carries the context needed to build this step's graph: which tokens this step processes (from L18's batch), the KV cache's current state, which backend, and so on. In other words, <span class="mono">build_graph</span> does not build out of thin air but <strong>builds just enough graph for "what this step computes"</strong>.
This is why prefill (computing a whole prompt at once) and decode (one token at a time), though sharing the same graph code, build graphs of different sizes.</p>
<p>That <span class="mono">build_pooling / build_dense_out</span> comment in the code is worth a mention: beyond the main N blocks, graph-building also appends some "wrap-up" steps as needed - pooling for embedding tasks, extra output layers for certain models. These are <strong>optional tails</strong>, usually unused in plain text generation, but they share the same graph framework, attached to the same graph by architecture and task.</p>

<h2>How one transformer layer is built</h2>
<p>The model's body is N identical transformer blocks stacked. See clearly how <strong>one layer</strong> is built and you understand the whole forward. A layer's skeleton is fixed: norm -&gt; attention -&gt; residual -&gt; norm -&gt; feed-forward -&gt; residual.</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>build_norm</h4><p>RMSNorm the input, stabilizing values into a sane range (L11's normalization operator).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>build_attn</h4><p>A whole attention set: Q/K/V projection + rope position + read/write KV cache + softmax + output projection (L11/L04/L19).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>residual add</h4><p>Add the attention output back to the input (residual connection), letting information and gradients flow well.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>build_norm + build_ffn</h4><p>Normalize again, then the feed-forward network (gate/up/down, three matmuls, L11).</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>residual add</h4><p>Add the FFN output back, producing this layer's output, fed to the next layer.</p></div></div>
</div>
<p>Written as pseudocode, this skeleton is nearly the loop body in <span class="mono">src/models/llama.cpp</span>:</p>
<pre class="code"><span class="cm"># pseudocode: one layer (mirrors the loop body in src/models/llama.cpp)</span>
cur  = <span class="fn">build_norm</span>(inpL, attn_norm_w)            <span class="cm"># RMSNorm(L11)</span>
cur  = <span class="fn">build_attn</span>(inp_attn, cur, wq,wk,wv,wo)    <span class="cm"># Q/K/V + rope + KV + softmax(L11/L04/L19)</span>
inpL = cur + inpL                              <span class="cm"># residual</span>
cur  = <span class="fn">build_norm</span>(inpL, ffn_norm_w)
cur  = <span class="fn">build_ffn</span>(cur, w_gate, w_up, w_down)      <span class="cm"># feed-forward(L11 mul_mat)</span>
inpL = cur + inpL                              <span class="cm"># residual -> next layer</span></pre>
<p>Loop 0 to <span class="mono">n_layer()</span> (L15's accessor method), assembling this set per layer, fetching weights by L15's naming convention (<span class="mono">blk.il.attn_q.weight</span> etc.). As you can see, "building the graph" at this level is very mechanical - feeding fixed blocks in a fixed order with each layer's own weights, chaining a long run of operators.</p>
<p>This nicely closes the last three lessons: L14 prepared tensors by name, L15 said which weights each layer takes and how many layers, and here L16 actually assembles them by the transformer structure. The three together answer "<strong>how a model goes from a pile of weights to a runnable graph</strong>".</p>
<p>Those two <strong>residual adds</strong> (cur + inpL) look trivial but are one key to deep transformers training and working: they let each layer learn only an "increment" on top of the "original input", so information and gradients flow straight down this shortcut without decaying away across dozens of layers. At graph time it is just an ordinary ggml add operator, but its meaning is far more than one addition.</p>
<p>The whole forward is not only repeated layers. At the <strong>start</strong>, one step looks up token ids into token vectors (the <span class="mono">token_embd</span> table, one of the graph inputs); at the <strong>end</strong>, after the last layer, there is an <span class="mono">output_norm</span> and a projection to vocab size <span class="mono">output</span> (computing logits, L17).
So the full graph is "input embedding -&gt; N blocks -&gt; output norm -&gt; project to logits", with those N middle layers being what we focus on dissecting.</p>
<p>Worth noting the relation between prefill and decode at graph time: both use the <strong>same</strong> graph code, differing only in the batch fed in (L18) - prefill feeds a whole prompt's many tokens at once, decode feeds one new token at a time. The graph's "shape" varies with token count, but its "structure" (how each layer is assembled) is identical.
This is the benefit of unified graph-building: one logic handles both "passing the prompt through once" and "generating word by word afterward".</p>
<p>One more thing: <strong>no concrete numbers appear</strong> in the graph code. <span class="mono">build_attn</span>, <span class="mono">build_ffn</span> operate entirely on "not-yet-computed tensors" - they merely say "matmul this weight with that input, call the result cur". The actual floats are filled in only when L10 executes.
So reading graph code, what you read is <strong>the shape of the data flow</strong>, not the data itself - the most intuitive feel of L09's lazy build.</p>
<p>Step back and this whole forward graph is really a <strong>directed acyclic graph</strong> (DAG): token vectors flow in from input leaves, transform through layer after layer of block operators, and finally flow to logits. Each operator is a node, arrows mean "who feeds whom".
L09 covered the essence of such graphs; this lesson just lets you see that a real large model's forward, landed on a graph, is exactly such a clearly-structured, layer-stacked DAG.</p>

<h2>Reusable blocks and graph inputs</h2>
<p>Underpinning this assembly is a set of <strong>reusable blocks</strong> and <strong>graph inputs</strong> on <span class="mono">llm_graph_context</span>. The blocks are methods like <span class="mono">build_attn</span>/<span class="mono">build_ffn</span>/<span class="mono">build_norm</span>; the graph inputs are the entry points that wire "external data" into the graph.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">graph inputs</span><span class="name">llm_graph_input_*</span></div><div class="ld">wire external data into the graph: embd (token vectors) · pos (positions) · attn_kv (KV cache)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">blocks</span><span class="name">build_attn / build_ffn / build_norm</span></div><div class="ld">assemble weights + inputs into a subgraph, internally L11 operators</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">product</span><span class="name">llm_graph_result -&gt; ggml_cgraph</span></div><div class="ld">all blocks chained; get_gf() hands out the final graph(L09)</div></div>
</div>
<p>Graph inputs (<span class="mono">llm_graph_input_*</span>) are an easily-overlooked but crucial concept. A compute graph needs more than "operators" - it needs "entry points": where token vectors enter, where each token's position comes from, where the KV cache attaches.
These are the input nodes built by <span class="mono">build_inp_embd</span>/<span class="mono">build_inp_pos</span> and friends. They are the graph's "leaves" (the leafs from L09); each inference step fills new data into them, and the graph computes a new result.</p>
<p>And everything the blocks produce are the ggml tensors from L11 (outputs of operators like <span class="mono">mul_mat</span>, <span class="mono">soft_max_ext</span>). One <span class="mono">build_attn</span> call is internally a small subgraph of a dozen-odd ggml operators chained by the attention math (L04).
Connect many such subgraphs end to end and you grow the model's entire forward graph - exactly L09's "operators chained into a graph", only here from the llama layer, organized by the transformer structure.</p>
<p>A glance inside <span class="mono">build_ffn</span>: modern llama-style models' feed-forward is not a simple "up then down" but <strong>SwiGLU</strong>-style - the <span class="mono">gate</span> and <span class="mono">up</span> matrices each project the input, the <span class="mono">gate</span> path passes an activation and is multiplied element-wise with <span class="mono">up</span>, then <span class="mono">down</span> projects back.
This is why one FFN layer has three weight matrices gate/up/down (seen in L15's naming convention). <span class="mono">build_ffn</span> wraps this fixed routine, done in one line at graph time.</p>
<p>More on graph inputs and "leaves". L09 covered two kinds of nodes: computed <strong>nodes</strong> and non-computed, only-read <strong>leaves</strong>. Weights are leaves (prepared at load, L14), and graph inputs (token vectors, positions) are leaves too - except their data is <strong>filled fresh each step</strong>.
Graph-building reserves these leaves' positions, execution fills in this step's data, and the same graph computes different results.</p>
<p>You will notice <span class="mono">build_attn</span> has <strong>several overloads</strong>. Why? Because attention has many variants: with or without KV cache (some prefill paths skip it, decode always uses it), standard multi-head or GQA, with or without a sliding window... Rather than write a full attention for each,
"a common skeleton + optional differences" is made into a few overloads each architecture picks from. Another instance of "fold differences into options, distill commonality into blocks".</p>
<p>Making graph inputs a family of classes (the <span class="mono">llm_graph_input_*</span> all derive from a common interface) is deliberate too: different inputs have different "fill methods" - token vectors look up by token id, positions are generated by current progress, the KV mask is computed by the causal rule.
Wrapping each input's "how to fill" into its own class, called uniformly before execution, readies the graph. This makes "which external inputs the graph needs" extensible and composable.</p>

<h2>Build and execute, sharply separated</h2>
<p>Finally, this lesson's most important point: <span class="mono">build_graph</span> only <strong>builds</strong>, never <strong>computes</strong>. It fills in the operators' op and src (L09's lazy build), then <span class="mono">get_gf()</span> hands out a <span class="mono">ggml_cgraph</span>; the actual node-by-node execution is the L10 backend's job.</p>
<p>This "build is build, execute is execute" separation pays off enormously: the same graph runs unchanged on wildly different hardware - CPU, CUDA, Metal (L10's backend scheduling) - with the upper model logic written once. And precisely because graph-building touches no concrete computation,
switching backends or adding new hardware needs <strong>no change to graph-building code</strong>. L16 "assembles the correct graph", L10 "computes the graph fast on some hardware" - each to its job, together making complete inference.</p>
<p>Precisely because graph-building yields only "structure", not data, this graph can in many cases be <strong>cached and reused</strong>: in consecutive decode steps each is "one new token", the graph's structure is identical, so the engine can reuse the previous graph's skeleton and only swap the inputs, saving the cost of rebuilding. Another dividend of "lazy build + structure-data separation".</p>
<p>Put this lesson back into the whole inference loop: per generated token, <span class="mono">llama_decode</span> (L17) is roughly the round "<strong>build graph (L16) -&gt; backend execute (L10) -&gt; get logits -&gt; sample (L21) the next token</strong>". L16 is the "turn the model into a computable graph" link in that round.
Understand it and you have joined "the loaded model" to "inference actually running". Next lesson, we enter <span class="mono">llama_context</span> to see how this round turns.</p>
<p>So what to truly take from this lesson is a <strong>mental model</strong>: model inference = assemble weights into a compute graph by architecture (L16) + execute that graph on some backend (L10). The assembly logic, written once, runs across all hardware; that is the root of llama.cpp being both lightweight and universal. Hold onto this, and the rest of Part 4 really unfolds along its thread.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why a separate src/models/&lt;arch&gt;.cpp per architecture? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>For <strong>difference isolation</strong>. Architectures' forward flows differ somewhat: some attentions carry a bias, some FFNs use a different activation, some insert other things between layers. Putting each architecture's "assembly" in its own file means changing one does not ripple to another, and it reads cleanly - one file is one model's complete forward.</p>
    <p>Meanwhile the blocks doing the real work (<span class="mono">build_attn</span> etc.) are <strong>shared</strong>, living in the base class <span class="mono">llm_graph_context</span>. So these architecture files are mostly short: just "in this architecture's order, call a few shared blocks and feed the right weights". Commonality in the base, differences in files, is the key to hosting dozens of architectures without chaos.</p>
    <p>This also echoes L15: adding a new architecture, the graph-building step is usually <strong>writing one not-long <span class="mono">src/models/&lt;arch&gt;.cpp</span></strong>, reusing existing blocks. Only a truly novel structure needs one or two new blocks in the base, or even one or two new ggml operators (L11).</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What does build_attn actually do inside? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>It translates L04's attention math into a chain of L11 operators. Roughly: three <span class="mono">mul_mat</span>s project the input into Q, K, V; <span class="mono">rope</span> injects position into Q, K; this step's K, V are <strong>written into the KV cache</strong>, and the historical K, V are <strong>read back</strong> (L19).</p>
    <p>Then compute attention scores (Q-by-K matmul), <span class="mono">soft_max_ext</span> adds the causal mask and normalizes to weights, another <span class="mono">mul_mat</span> weight-sums V; finally an output projection. One <span class="mono">build_attn</span> call thus assembles a <strong>whole attention set</strong> into a subgraph.</p>
    <p>So everything you learned earlier is used here: L04's math is the blueprint, L11's operators are the bricks, L19's KV cache is what lets each step compute only the new token. <span class="mono">build_attn</span> is the "assembler" welding these three together in the right order.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How are build and execute fully separated? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Via L09's lazy build. The <span class="mono">build_*</span> blocks <strong>do not compute</strong>; they only create a tensor and fill its <span class="mono">op</span> (which operator I am) and <span class="mono">src</span> (who my inputs are). One pass of assembly yields a graph that <strong>only describes "what to compute and what depends on what"</strong>, with not a number computed yet.</p>
    <p>Then <span class="mono">get_gf()</span> hands out this <span class="mono">ggml_cgraph</span>, and the L10 backend computes it node by node in topological order. The build side cares only about "is the logical structure correct"; the execute side only about "how to compute fast on this hardware" - their concerns fully separated.</p>
    <p>The payoff is <strong>the freedom of decoupling</strong>: write the model logic (graph) once, and it runs on all backends; to support new hardware, just add a backend on the execute side, with not one line of graph code changed. This is the most valuable part of the ggml/llama layering, and the root reason it runs on your laptop CPU and a datacenter GPU alike.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama_model::build_graph</span> dispatches to each architecture's <span class="mono">build_arch_graph</span> (<span class="mono">src/models/&lt;arch&gt;.cpp</span>), returning a <span class="mono">ggml_cgraph</span> (via <span class="mono">res-&gt;get_gf()</span>).</li>
    <li>Blocks <span class="mono">build_norm</span>/<span class="mono">build_attn</span>/<span class="mono">build_ffn</span> are methods on the base <span class="mono">llm_graph_context</span>, reused by every architecture.</li>
    <li>One layer = norm -&gt; attn (QKV+rope+KV+softmax) -&gt; residual -&gt; norm -&gt; ffn -&gt; residual; looped <span class="mono">n_layer()</span> times, fetching weights by name (L15).</li>
    <li>Graph inputs <span class="mono">llm_graph_input_*</span> wire token vectors/positions/KV into the graph as its "leaves" (L09).</li>
    <li><strong>Build only, no compute</strong> (L09 lazy): <span class="mono">build_graph</span> assembles the graph, the L10 backend executes; the same graph runs on any backend.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Writing "how each architecture does its forward" as a single <span class="mono">src/models/&lt;arch&gt;.cpp</span>, and distilling "how to compute attention/FFN" into <span class="mono">llm_graph_context</span>'s reusable blocks - so a new architecture is just "a different arrangement of standard parts". Better still, the underlying ggml (L08-L12) <strong>has no idea</strong> whether llama or qwen runs above;
  it sees just an ordinary compute graph and executes it (L10). Model diversity gathered in the graph layer, computational generality kept in the ggml layer - this clean boundary is the underlying secret to llama.cpp taking in all rivers while one engine runs them all. Next lesson, we see how this graph, packed into a <span class="mono">llama_context</span>, actually runs one inference step.
</div>
""",
}

LESSON_17 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
能建图的模型（L14-16）还差一个<strong>运行时</strong>，才能真正跑起来、记住对话进度、把结果交出去——这就是 <span class="mono">llama_context</span>。它是一个<strong>有状态</strong>的对象，持有这次会话的配置（cparams）、
KV cache（memory）、后端调度器（sched）和输出缓冲；<span class="mono">llama_decode</span> 跑一步前向、<span class="mono">llama_get_logits_ith</span> 取出结果。
</p>
<p style="color:var(--muted);margin-top:.4rem">这一课是把前面"静态的模型"激活成"会跑的推理"的关键一环。它也回答了一个很实际的问题：为什么 llama.cpp 把"模型"和"上下文"分成两个对象？想清楚这件事，你就明白了 llama-server 能同时服务很多用户的底层原因。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  如果 <span class="mono">llama_model</span> 是<strong>图纸 + 零件库</strong>（静态、只读、可共享），<span class="mono">llama_context</span> 就是<strong>施工现场</strong>（有状态、每个会话一个）：现场放着这次施工的进度（KV cache）、工具调度（sched）、和产出（logits）。
  同一份图纸，能同时开好几个工地——一个 model 配多个 context，各跑各的对话，谁也不影响谁。
</div>

<h2>model 与 context：只读知识 vs 有状态会话</h2>
<p>这是这一课最该先想清的一刀切：<strong>权重是只读的知识，会话是有状态的进度</strong>，两者被故意分成两个对象。</p>
<div class="cols">
  <div class="col"><h4>llama_model（只读知识）</h4><p>权重 · 超参 · 词表 · <strong>只读</strong> · 一份就够 · 可被<strong>多个 context 共享</strong></p></div>
  <div class="col"><h4>llama_context（有状态会话）</h4><p>cparams + KV cache + sched + logits · <strong>有状态</strong> · <strong>每会话一个</strong> · 记着这次对话算到哪</p></div>
</div>
<p>为什么要这么分？因为权重几个 GB、加载一次就不变，理应<strong>共享</strong>；而"对话进度"（KV cache、当前位置）是每个会话各不相同的<strong>状态</strong>，必须各存一份。把不变的知识和会变的状态拆开，一份权重就能撑起许多并发会话——这正是服务器多用户的根基。</p>
<p>打个比方：模型像一本字典（人人可查、内容不变），上下文像每个人手里的草稿纸（各写各的、互不干扰）。你绝不会给每个查字典的人各印一本字典，但每个人都需要自己的草稿纸。llama.cpp 这一刀，切的正是"共享的知识"和"私有的状态"。</p>
<p>"一个 model 配多个 context"不是空话，而是天天在发生的事。<span class="mono">llama-server</span> 同时服务多个用户时，就是一份权重 + 每个请求一个 context；甚至单个程序里想并行跑几条不同的对话，也是开几个 context。
它们共享那份只读的权重，各自维护自己的 KV 和进度，互不串扰。理解了这点，你就明白"加载一次、服务很多"是怎么做到的。</p>
<p>实现上，<span class="mono">llama_context</span> 内部持有一个指向 <span class="mono">llama_model</span> 的引用——它不复制权重，只是"借用"。所以新建一个 context 的代价很小：分配一些会话状态（主要是 KV cache 的空间），权重那几个 GB 一个字节都不用再读、不用再拷。
这也是为什么 server 加一个并发连接，增量内存主要就是那一份 KV，而不是整个模型。</p>
<p>这"知识 vs 状态"的分法，其实是计算机里一条很通用的设计原则：把<strong>无状态、可共享</strong>的部分和<strong>有状态、需隔离</strong>的部分分开。Web 服务器把静态资源和会话 session 分开、数据库把只读快照和事务状态分开，都是同一个思路。
llama.cpp 把它用在了推理上：权重是无状态的"程序"，context 是有状态的"进程"。</p>
<p>这里的"会话"（session）一词值得点明：它就是一次"<strong>连续的对话或生成过程</strong>"。同一个会话里，后面的话能记得前面说过什么（靠 KV cache）；换一个会话，就是一张白纸重新开始。
所以 context 本质上承载的是"<strong>一次连贯对话的全部记忆</strong>"——它在，对话的上下文就在；它一释放，这次对话就被忘得干干净净。</p>
<p>顺带一提，"上下文"这个词在这里有两层意思容易混：一是 <span class="mono">llama_context</span> 这个对象，二是 <span class="mono">n_ctx</span> 那个"能记多少 token"的上下文长度。前者是装会话状态的容器，后者是这个容器能装下的对话有多长。本课讲的主要是前者，后者的细节留到 L19。</p>

<h2>context 里有什么</h2>
<p>掀开 <span class="mono">llama_context</span> 看看，它主要持有四样东西：配置、记忆、调度器、输出。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">配置</span><span class="name">llama_cparams cparams</span></div><div class="ld">这次会话的参数：上下文多长、批多大、几个线程……</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">记忆</span><span class="name">llama_memory_ptr memory</span></div><div class="ld">KV cache（L19）：记着这次对话先前每个 token 的 K/V</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">调度</span><span class="name">ggml_backend_sched_ptr sched</span></div><div class="ld">多后端调度器（L10）：决定图的哪部分在 CPU/GPU 上算</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">输出</span><span class="name">buffer_view&lt;float&gt; logits / embd</span></div><div class="ld">输出缓冲：装这一步算出的 logits（下一 token 的分数）</div></div>
</div>
<pre class="code"><span class="cm">// 简化自 src/llama-context.h</span>
<span class="kw">struct</span> llama_context {
    llama_cparams          cparams;  <span class="cm">// 这次会话的配置</span>
    llama_memory_ptr       memory;   <span class="cm">// KV cache 等(L19)</span>
    ggml_backend_sched_ptr sched;    <span class="cm">// 多后端调度(L10)</span>
    buffer_view&lt;float&gt;     logits;   <span class="cm">// 输出: 下一 token 的分数</span>
};</pre>
<p>这四样合起来，正好是"跑一步推理"所需的全部状态：<span class="mono">cparams</span> 说"按什么规格跑"，<span class="mono">memory</span> 记"之前算过什么"，<span class="mono">sched</span> 管"在哪块硬件上算"，<span class="mono">logits</span> 接"算出来的结果"。
注意 <span class="mono">memory</span> 是个泛化的"记忆"抽象（不只是裸的 KV cache，还能是 recurrent、hybrid 等变体，L19 细讲），所以字段名是 memory 而非 kv_cache——这是为长上下文与新型架构留的余地。</p>
<p>还要点一句：<span class="mono">context</span> <strong>不</strong>持有权重——权重在 <span class="mono">model</span> 里，context 只<strong>引用</strong>它。这正是上一节"分开"的体现：一个轻量的 context 背着会话状态、指向那份重而只读的权重，于是开多个 context 几乎不额外占权重的内存。</p>
<p>为什么 <span class="mono">sched</span>（后端调度器）也要放进 context、而不是放进 model？因为调度是带<strong>会话状态</strong>的：它要管这次会话的中间张量内存怎么分配复用（L10 的 ggml-alloc）、图的哪部分在哪块设备上算。
不同会话各跑各的图，自然各需要一个调度器。把它放进 context，正好和"每会话一份状态"的设计对齐。</p>
<p>输出为什么是 <span class="mono">buffer_view&lt;float&gt;</span> 这种"视图"，而不是一个普通数组？因为输出的大小是动态的——这一步标了几个位置要输出，就有几行 logits（每行 <span class="mono">n_vocab</span> 个数）。
用一个轻量的视图指向底层缓冲，既能灵活表示"这一步有几行输出"，又避免反复分配。<span class="mono">llama_get_logits_ith</span> 取的，就是这个视图里第 i 行。</p>
<p>顺带说说 context 的生死：它由 <span class="mono">llama_init_from_model(model, cparams)</span> 创建（这时就按 cparams 把 KV cache 等开好），用完由 <span class="mono">llama_free</span> 释放。model 活得久（整个服务期间），context 可以来去（一个请求一个）。
这种"长命的 model + 短命的 context"的生命周期搭配，正是服务器处理一波又一波请求的常态。</p>
<p>还有个实现细节：你传进去的 <span class="mono">llama_context_params</span> 会被<strong>拷一份</strong>存进 context（内部叫 <span class="mono">llama_cparams</span>）。这样 context 一旦建好，它这次会话的规格就<strong>定下来了</strong>，不会因为你后来改了外面那份参数而变。
每个 context 各自记着自己的规格，互不影响——这又是"每会话独立"的一处体现。</p>
<p>那个和 logits 并列的 <span class="mono">embd</span> 缓冲也顺带提一句：它装的是<strong>嵌入向量</strong>输出——做 embedding 任务（把整句话变成一个向量）时用它，而不是 logits。所以 context 的输出口其实有两个：要"下一个词"就看 logits，要"句子的向量表示"就看 embd。同一套 decode 机制，按任务取不同的输出。</p>

<h2>cparams：怎么配这次会话</h2>
<p>创建 context 时，你用 <span class="mono">llama_context_params</span>（cparams）告诉它这次会话怎么跑。这些参数大多是在<strong>显存与速度之间做权衡</strong>。</p>
<table class="t">
  <tr><th>参数</th><th>含义</th></tr>
  <tr><td><span class="mono">n_ctx</span></td><td>上下文长度（能记多少 token；越大 KV cache 越占显存）</td></tr>
  <tr><td><span class="mono">n_batch</span> / <span class="mono">n_ubatch</span></td><td>逻辑 / 物理批大小（一次提交多少 / 一次真正算多少，L18）</td></tr>
  <tr><td><span class="mono">n_seq_max</span></td><td>最多并行几条序列</td></tr>
  <tr><td><span class="mono">n_threads</span></td><td>用几个 CPU 线程</td></tr>
  <tr><td><span class="mono">type_k</span> / <span class="mono">type_v</span></td><td>KV cache 的数据类型（可量化以省显存）</td></tr>
  <tr><td><span class="mono">offload_kqv</span></td><td>是否把 KV 相关计算放到 GPU</td></tr>
  <tr><td><span class="mono">pooling_type</span></td><td>embedding 任务时怎么把 token 向量汇成句向量</td></tr>
</table>
<pre class="code"><span class="cm">// 简化自 include/llama.h 的 llama_context_params</span>
<span class="kw">struct</span> llama_context_params {
    uint32_t n_ctx;      uint32_t n_batch;   uint32_t n_ubatch;
    uint32_t n_seq_max;  int32_t  n_threads;
    ggml_type type_k, type_v;   <span class="cm">// KV 的量化类型(省显存)</span>
    bool offload_kqv;           <span class="cm">// KV 计算放 GPU?</span>
};</pre>
<p>这里最该上手感的是 <span class="mono">n_ctx</span> 和 <span class="mono">type_k/type_v</span>：它们直接决定 KV cache 吃多少显存。<span class="mono">n_ctx</span> 翻倍，KV cache 大致翻倍；把 <span class="mono">type_k/type_v</span> 从 16 位降到 8 位，KV 占用又能减半（代价是一点点精度）。
所以"能开多长上下文"不是模型单方面决定的，而是你按手头显存，在 <span class="mono">n_ctx</span> 和 KV 量化之间调出来的——这条线会在 L19 讲 KV cache 时再展开。</p>
<p><span class="mono">n_seq_max</span> 这个参数关系到一个常被忽略的能力：一个 context 可以<strong>同时跑多条序列</strong>。比如批量给几个不同 prompt 各生成回答，可以放进同一个 context、用不同的 seq_id 区分（L18），共享这份权重和这套调度。
<span class="mono">n_seq_max</span> 就是上限。这让"一个 context 服务多个并发对话"成为可能，是比"一对话一 context"更省的玩法。</p>
<p>cparams 和模型本身的默认值也有联系。很多参数你可以填 0 表示"用模型的默认"——比如 <span class="mono">n_ctx</span> 填 0，就取模型训练时的上下文长度（L15 的 <span class="mono">n_ctx_train</span>）。
这让你既能省心地用默认值，又能在需要时按显存覆盖它。配置的灵活性，就藏在这些"0 表示跟随模型"的约定里。</p>
<p>表里没列全的 cparams 还有一些专门用途，比如 <span class="mono">pooling_type</span>（做 embedding 任务时怎么把 token 向量汇成一个句向量）、各种 RoPE 缩放参数（把上下文外推到训练长度之外）。
普通文本生成多半用默认就行，但它们的存在说明：context 不只服务"生成下一个词"，也能服务 embedding、长上下文外推等多种任务。</p>
<p>一个实用提醒：context 的内存占用，<strong>大头往往是 KV cache</strong>，而 KV cache 的大小由 <span class="mono">n_ctx</span>、层数、KV 头数、<span class="mono">type_k/type_v</span> 共同决定。所以当你发现显存不够，调小 <span class="mono">n_ctx</span> 或量化 KV，往往比换模型更立竿见影。这条经验，在 L19 会有更细的账。</p>
<p>为什么这些参数放在<strong>建 context 时</strong>配、而不是加载 model 时配？因为它们是"<strong>这次会话怎么跑</strong>"的事，而不是"模型是什么"的事。同一个 model，你可以用不同的 cparams 开多个 context：一个开长上下文、一个开短的，一个多线程、一个少线程，各按各的场景来。
把会话参数和模型解耦，正是为了这种灵活。</p>

<h2>一步推理与取结果</h2>
<p>万事俱备，跑推理就是反复调 <span class="mono">llama_decode</span>。它吃一个 batch（L18），内部把建图、执行、更新 KV 一气呵成，最后把 logits 放进 context 的输出缓冲。</p>
<pre class="code"><span class="cm"># 伪代码: 一步推理(llama_decode 内部)</span>
<span class="fn">llama_decode</span>(ctx, batch)              <span class="cm"># 跑一步前向</span>
<span class="cm">#   -&gt; 切 ubatch(L18) -&gt; build_graph(L16) -&gt; sched 执行(L10) -&gt; 更新 KV(L19)</span>
<span class="cm">#   -&gt; logits 写进 ctx 的输出缓冲</span>
p = <span class="fn">llama_get_logits_ith</span>(ctx, i)    <span class="cm"># 取第 i 个 token 的 logits(n_vocab 维)</span></pre>
<p>取结果用 <span class="mono">llama_get_logits_ith(ctx, i)</span>，拿到第 i 个 token 的 logits——一个 <span class="mono">n_vocab</span> 维的向量，是"下一个词"的未归一分数，接下来交给采样（L21）挑一个词。这就把 L16 的建图、L10 的执行、L19 的 KV，
通过 <span class="mono">llama_decode</span> 这一个函数<strong>串成了一步完整的推理</strong>。</p>
<p>所以 <span class="mono">llama_context</span> 在整套机制里扮演的是"<strong>总指挥 + 状态本</strong>"：它知道这次会话的所有配置、记着算到哪、调度着硬件、收着输出。理解了它，你就把前面那些零件（model、graph、backend、KV）真正<strong>组装成了一台能转的推理机</strong>。</p>
<p>这里也顺势把 L03 的 prefill/decode 接上：两个阶段<strong>都</strong>是调 <span class="mono">llama_decode</span>，区别只在喂的 batch。prefill 一次喂整段 prompt（多个 token），把它们的 K/V 一口气填进 KV cache，只取最后一个的 logits；decode 之后每次只喂一个新 token、取它的 logits。
同一个函数，两种节奏，全靠 batch 来表达。</p>
<p>把整个自回归循环画出来就是：<span class="mono">llama_decode</span> 算出 logits -&gt; 采样（L21）挑一个 token -&gt; 把这个新 token 包成一个新 batch 喂回 <span class="mono">llama_decode</span> -&gt; 再出 logits…… 如此往复，逐字蹦出回答。
context 在这整个循环里<strong>一直在</strong>：它的 KV cache 一步步变长、它的输出缓冲一步步刷新。一句对话的生成，就是这个循环在一个 context 上转了很多圈。</p>
<p>补一句 <span class="mono">llama_encode</span>：有些模型（如带 encoder 的）需要先 encode、再 decode，所以 API 里既有 <span class="mono">llama_decode</span> 也有 <span class="mono">llama_encode</span>。对最常见的 decoder-only 大模型（L04），你基本只会用到 <span class="mono">llama_decode</span>。知道有这么个分工即可，不必深究。</p>
<p>所以这一课交付的，是一个<strong>从"零件"到"机器"</strong>的跃迁：前几课造出了 model（L14-15）和 graph（L16）这些零件，这一课用 context 把它们装进一个能反复转动的循环里。读到这儿，你已经能在脑子里跑通一次完整推理：加载模型 -&gt; 建 context -&gt; 反复 decode + 采样 -&gt; 逐字输出。
剩下几课，是把这台机器的几个关键部件（batch、KV cache、采样……）再各自拆开看细节。</p>
<p>再强调一遍"<strong>状态在推进</strong>"这件事，因为它是理解自回归的关键。每次 <span class="mono">llama_decode</span> 不是从头重算，而是<strong>在 context 现有状态上往前走一步</strong>：KV cache 里已经有前面所有 token 的 K/V，这一步只需算新 token、把它的 K/V 追加进去。
正因为状态被 context 一直记着，decode 才能做到"每步只算一个 token"这么快（L03/L19）。</p>
<p>最后用一句话收束 context 的角色：它是那个让"静态模型"变成"动态推理"的开关。没有它，model 只是一堆躺着的权重；有了它，权重才被一步步驱动起来、吐出一个个 token。下一课起，我们就钻进这台机器的具体部件，先从喂给 decode 的 batch 开始。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 model 和 context 要分开？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>核心是"<strong>共享只读、各存状态</strong>"。权重是只读的、几个 GB，多个会话<strong>共用同一份</strong>最省内存（配合 L13 的 mmap，连物理内存都能跨进程共享）；而 KV cache、当前位置这些是<strong>每会话不同的状态</strong>，必须各存一份。</p>
    <p>分开之后，一台机器上一份权重就能撑起很多并发会话：每来一个用户/请求，新建一个轻量的 context（只背自己的 KV），权重那几个 GB 一动不动地被大家共享。这正是 <span class="mono">llama-server</span> 能多并发、省内存的根基。</p>
    <p>反过来想，如果不分开、把权重和状态揉成一个对象，那每个会话都得复制一份几 GB 的权重——服务几十个用户就要几十份权重，根本扛不住。一个看似简单的"拆成两个对象"的设计，撑起了整个多用户服务的可行性。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> logits 是什么？为什么只在某些 token 上有？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">logits</span> 是模型对"下一个 token 该是谁"打出的一组<strong>未归一分数</strong>，长度等于词表大小 <span class="mono">n_vocab</span>。它还不是概率（没归一），但分数越高的词越可能被选中；采样（L21）就是拿这组 logits 去挑一个词。</p>
    <p>关键是：<strong>不是每个 token 都要算 logits</strong>。算 logits 要做一次"隐藏向量 -&gt; 词表大小"的大矩阵乘，挺贵。而 prefill 阶段把整段 prompt 过一遍时，中间那些 token 的 logits 根本用不到——我们只要<strong>最后一个</strong> token 的 logits（用来预测下一个）。</p>
    <p>所以哪些位置算 logits，由 batch 的输出标志控制（L18）。decode 阶段每步只新增一个 token、只它要 logits；prefill 整段只要末位。<span class="mono">llama_get_logits_ith(ctx, i)</span> 就是去取第 i 个被标记输出的位置的 logits。这套"按需算输出"的设计，省下了大量无用的大矩阵乘。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> context 怎么把 L16/L10/L19 串成一步？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">llama_decode</span> 是那个"总指挥"。它拿到这一步的 batch（L18），先用批处理逻辑把它切成物理可算的 ubatch；对每个 ubatch，调 <span class="mono">build_graph</span>（L16）搭出这一步的计算图。</p>
    <p>然后把图交给 context 里的 <span class="mono">sched</span>（L10 的后端调度器）真正执行；执行过程中，注意力算子会把这一步新 token 的 K/V <strong>写进 context 的 KV cache</strong>（L19），并读回历史 K/V。算完，把输出 logits 写进 context 的输出缓冲。</p>
    <p>一圈下来，context 的状态就<strong>往前推进了一步</strong>：KV cache 多记了一个 token、输出缓冲有了新 logits。下一次 <span class="mono">llama_decode</span> 接着在这个状态上推进。正是 context 把这些散落的机制（建图、执行、KV、输出）攒在一起、按顺序驱动，才有了"一步接一步"的自回归生成。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama_model</span> 是<strong>只读</strong>的权重/超参/词表（一份、可被多 context 共享）；<span class="mono">llama_context</span> 是<strong>有状态</strong>的会话（每会话一个）。</li>
    <li>context 持有 <span class="mono">cparams</span>（配置）+ <span class="mono">memory</span>（KV cache，L19）+ <span class="mono">sched</span>（后端调度，L10）+ <span class="mono">logits</span>（输出缓冲）；不持有权重，只引用 model。</li>
    <li>cparams 调 <span class="mono">n_ctx</span>/<span class="mono">n_batch</span>/<span class="mono">type_k</span>/<span class="mono">type_v</span> 等，多在显存与速度间权衡（<span class="mono">n_ctx</span>、KV 量化直接影响显存）。</li>
    <li><span class="mono">llama_decode</span> 跑一步前向（切 ubatch -&gt; 建图 -&gt; 执行 -&gt; 更新 KV -&gt; 出 logits）；<span class="mono">llama_get_logits_ith</span> 取第 i 个 token 的 logits。</li>
    <li>分开 model 与 context = 一份权重撑多会话，是 <span class="mono">llama-server</span> 多并发的根基。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"<strong>不变的知识</strong>"（权重 model）和"<strong>会话的状态</strong>"（KV/进度 context）拆成两个对象——这一刀看似平常，回报却极大：一份几 GB 的权重能被许多会话共享，每个会话只额外背一份轻量的状态。
  于是同一台机器、同一份模型，能同时服务很多用户。这正是"<strong>状态与数据分离</strong>"这条老道理在推理引擎里的又一次体现，也是从"能跑一个对话"到"能扛一个服务"之间，那道最关键的设计分水岭。下一课，我们就看喂给 <span class="mono">llama_decode</span> 的那个 batch 到底长什么样。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
A graph-able model (L14-16) still needs a <strong>runtime</strong> to actually run, remember conversation progress, and hand results out - that is <span class="mono">llama_context</span>. It is a <strong>stateful</strong> object holding this session's config (cparams),
the KV cache (memory), the backend scheduler (sched), and output buffers; <span class="mono">llama_decode</span> runs one forward step, <span class="mono">llama_get_logits_ith</span> reads the result.
</p>
<p style="color:var(--muted);margin-top:.4rem">This lesson is the key link that activates "a static model" into "running inference". It also answers a very practical question: why does llama.cpp split "model" and "context" into two objects? Think this through and you understand the underlying reason llama-server can serve many users at once.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  If <span class="mono">llama_model</span> is a <strong>blueprint + parts library</strong> (static, read-only, shareable), <span class="mono">llama_context</span> is a <strong>construction site</strong> (stateful, one per session): the site holds this build's progress (KV cache), tool scheduling (sched), and output (logits).
  The same blueprint can run several sites at once - one model with several contexts, each running its own conversation, none disturbing the others.
</div>

<h2>model vs context: read-only knowledge vs stateful session</h2>
<p>This is the cut to grasp first: <strong>weights are read-only knowledge, a session is stateful progress</strong>, and the two are deliberately split into two objects.</p>
<div class="cols">
  <div class="col"><h4>llama_model (read-only knowledge)</h4><p>weights · hyperparameters · vocab · <strong>read-only</strong> · one is enough · <strong>shareable by many contexts</strong></p></div>
  <div class="col"><h4>llama_context (stateful session)</h4><p>cparams + KV cache + sched + logits · <strong>stateful</strong> · <strong>one per session</strong> · remembers where this conversation is</p></div>
</div>
<p>Why split this way? Because weights are several GB, loaded once and unchanging, and ought to be <strong>shared</strong>; while "conversation progress" (KV cache, current position) is <strong>per-session state</strong> that must be stored separately. Splitting unchanging knowledge from changing state, one copy of weights can hold up many concurrent sessions - the basis of multi-user serving.</p>
<p>An analogy: the model is like a dictionary (everyone consults it, its content fixed), the context like each person's scratch paper (each writes their own, none interfering). You would never print a separate dictionary per reader, but everyone needs their own scratch paper. llama.cpp's cut is exactly between "shared knowledge" and "private state".</p>
<p>"One model with several contexts" is no slogan but a daily reality. When <span class="mono">llama-server</span> serves many users at once, it is one copy of weights + one context per request; even within a single program, running several different conversations in parallel means opening several contexts.
They share the read-only weights, each maintaining its own KV and progress, none interfering. Understand this and you see how "load once, serve many" is done.</p>
<p>In implementation, <span class="mono">llama_context</span> holds a reference to <span class="mono">llama_model</span> internally - it copies no weights, just "borrows". So creating a context is cheap: allocate some session state (mainly KV cache space), with not a byte of those several-GB weights re-read or re-copied.
This is why a server adding one concurrent connection adds memory mainly for that one KV, not the whole model.</p>
<p>This "knowledge vs state" split is actually a very general design principle in computing: separate the <strong>stateless, shareable</strong> parts from the <strong>stateful, must-isolate</strong> parts. Web servers separate static assets from session state, databases separate read-only snapshots from transaction state - the same idea.
llama.cpp applies it to inference: weights are a stateless "program", the context a stateful "process".</p>
<p>The word "session" here is worth pinning down: it is one "<strong>continuous conversation or generation process</strong>". Within one session, later words remember what was said before (via the KV cache); a different session is a blank sheet starting over.
So a context essentially carries "<strong>all the memory of one coherent conversation</strong>" - while it lives, the conversation's context lives; once freed, this conversation is forgotten clean.</p>
<p>By the way, "context" here has two easily-confused senses: one is the <span class="mono">llama_context</span> object, the other is <span class="mono">n_ctx</span>, the "how many tokens it can remember" context length. The former is the container holding session state, the latter how long a conversation that container can hold. This lesson is mainly about the former; the latter's details wait for L19.</p>

<h2>What is inside a context</h2>
<p>Open up <span class="mono">llama_context</span> and it mainly holds four things: config, memory, scheduler, output.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">config</span><span class="name">llama_cparams cparams</span></div><div class="ld">this session's params: how long the context, how big the batch, how many threads...</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">memory</span><span class="name">llama_memory_ptr memory</span></div><div class="ld">KV cache (L19): remembers this conversation's prior tokens' K/V</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">scheduling</span><span class="name">ggml_backend_sched_ptr sched</span></div><div class="ld">multi-backend scheduler (L10): decides which graph parts compute on CPU/GPU</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">output</span><span class="name">buffer_view&lt;float&gt; logits / embd</span></div><div class="ld">output buffer: holds this step's logits (the next token's scores)</div></div>
</div>
<pre class="code"><span class="cm">// simplified from src/llama-context.h</span>
<span class="kw">struct</span> llama_context {
    llama_cparams          cparams;  <span class="cm">// this session's config</span>
    llama_memory_ptr       memory;   <span class="cm">// KV cache etc.(L19)</span>
    ggml_backend_sched_ptr sched;    <span class="cm">// multi-backend scheduling(L10)</span>
    buffer_view&lt;float&gt;     logits;   <span class="cm">// output: next token's scores</span>
};</pre>
<p>These four together are exactly the state needed to "run one inference step": <span class="mono">cparams</span> says "at what spec to run", <span class="mono">memory</span> remembers "what was computed before", <span class="mono">sched</span> manages "on which hardware to compute", <span class="mono">logits</span> receives "the computed result".
Note <span class="mono">memory</span> is a generalized "memory" abstraction (not only a raw KV cache, but also variants like recurrent, hybrid, L19), so the field is named memory rather than kv_cache - room left for long context and new architectures.</p>
<p>One more point: the <span class="mono">context</span> does <strong>not</strong> hold the weights - those live in <span class="mono">model</span>, and the context only <strong>references</strong> it. This is exactly the previous section's "split" in action: a lightweight context carries session state and points at the heavy, read-only weights, so opening many contexts costs almost no extra weight memory.</p>
<p>Why is <span class="mono">sched</span> (the backend scheduler) also placed in the context, not the model? Because scheduling carries <strong>session state</strong>: it manages how this session's intermediate-tensor memory is allocated and reused (L10's ggml-alloc), and which graph parts compute on which device.
Different sessions run their own graphs, so each naturally needs a scheduler. Placing it in the context aligns exactly with the "one bit of state per session" design.</p>
<p>Why is output a <span class="mono">buffer_view&lt;float&gt;</span> "view" rather than a plain array? Because output size is dynamic - however many positions this step flags for output, that many rows of logits (each <span class="mono">n_vocab</span> numbers).
A lightweight view pointing into an underlying buffer both flexibly expresses "how many output rows this step has" and avoids repeated allocation. <span class="mono">llama_get_logits_ith</span> reads the i-th row of this view.</p>
<p>A word on a context's life and death: it is created by <span class="mono">llama_init_from_model(model, cparams)</span> (opening the KV cache etc. per cparams then), and freed by <span class="mono">llama_free</span> when done. The model is long-lived (the whole service duration), the context comes and goes (one per request).
This "long-lived model + short-lived context" lifecycle pairing is exactly how a server handles wave after wave of requests.</p>
<p>An implementation detail: the <span class="mono">llama_context_params</span> you pass in is <strong>copied</strong> into the context (internally <span class="mono">llama_cparams</span>). So once a context is built, its session spec is <strong>fixed</strong>, not changing because you later edit that outer params struct.
Each context remembers its own spec, none interfering - another showing of "per-session independence".</p>
<p>The <span class="mono">embd</span> buffer alongside logits deserves a mention: it holds <strong>embedding vector</strong> output - used for embedding tasks (turning a whole sentence into one vector) instead of logits. So a context actually has two output ports: for "the next word" read logits, for "the sentence's vector representation" read embd. The same decode mechanism, different outputs by task.</p>

<h2>cparams: configuring this session</h2>
<p>When creating a context, you tell it how to run via <span class="mono">llama_context_params</span> (cparams). Most of these parameters trade off <strong>VRAM against speed</strong>.</p>
<table class="t">
  <tr><th>parameter</th><th>meaning</th></tr>
  <tr><td><span class="mono">n_ctx</span></td><td>context length (how many tokens it can remember; bigger = more KV cache VRAM)</td></tr>
  <tr><td><span class="mono">n_batch</span> / <span class="mono">n_ubatch</span></td><td>logical / physical batch size (how much submitted / actually computed at once, L18)</td></tr>
  <tr><td><span class="mono">n_seq_max</span></td><td>max number of parallel sequences</td></tr>
  <tr><td><span class="mono">n_threads</span></td><td>how many CPU threads</td></tr>
  <tr><td><span class="mono">type_k</span> / <span class="mono">type_v</span></td><td>KV cache data type (can be quantized to save VRAM)</td></tr>
  <tr><td><span class="mono">offload_kqv</span></td><td>whether to put KV-related compute on the GPU</td></tr>
  <tr><td><span class="mono">pooling_type</span></td><td>how to pool token vectors into a sentence vector for embedding tasks</td></tr>
</table>
<pre class="code"><span class="cm">// simplified from llama_context_params in include/llama.h</span>
<span class="kw">struct</span> llama_context_params {
    uint32_t n_ctx;      uint32_t n_batch;   uint32_t n_ubatch;
    uint32_t n_seq_max;  int32_t  n_threads;
    ggml_type type_k, type_v;   <span class="cm">// KV quant type(save VRAM)</span>
    bool offload_kqv;           <span class="cm">// KV compute on GPU?</span>
};</pre>
<p>The two to build intuition for are <span class="mono">n_ctx</span> and <span class="mono">type_k/type_v</span>: they directly decide how much VRAM the KV cache eats. Double <span class="mono">n_ctx</span> and the KV cache roughly doubles; drop <span class="mono">type_k/type_v</span> from 16-bit to 8-bit and KV usage halves again (at a little precision cost).
So "how long a context you can open" is not decided by the model alone but tuned by you, given your VRAM, between <span class="mono">n_ctx</span> and KV quantization - a thread we pick up again in L19's KV cache.</p>
<p><span class="mono">n_seq_max</span> relates to an often-overlooked capability: one context can <strong>run multiple sequences at once</strong>. For example, generating answers for several different prompts in a batch can go into the same context, distinguished by different seq_ids (L18), sharing these weights and this scheduling.
<span class="mono">n_seq_max</span> is the upper bound. This makes "one context serving several concurrent conversations" possible, a more frugal approach than "one conversation per context".</p>
<p>cparams also relate to the model's own defaults. Many parameters you can set to 0 to mean "use the model's default" - e.g. <span class="mono">n_ctx</span> set to 0 takes the model's trained context length (L15's <span class="mono">n_ctx_train</span>).
This lets you both rely on defaults effortlessly and override by VRAM when needed. Configuration flexibility hides in these "0 means follow the model" conventions.</p>
<p>cparams the table omits have specialized uses too, such as <span class="mono">pooling_type</span> (how to pool token vectors into one sentence vector for embedding tasks) and various RoPE scaling parameters (extrapolating context beyond the trained length).
Plain text generation mostly uses defaults, but their existence shows the context serves not only "generate the next word" but also embedding, long-context extrapolation, and more.</p>
<p>A practical reminder: a context's memory is <strong>mostly the KV cache</strong>, whose size is set jointly by <span class="mono">n_ctx</span>, layer count, KV head count, and <span class="mono">type_k/type_v</span>. So when you hit a VRAM wall, shrinking <span class="mono">n_ctx</span> or quantizing KV is often more effective than swapping models. We will do this account in more detail in L19.</p>
<p>Why are these parameters configured <strong>at context creation</strong>, not at model load? Because they are about "<strong>how this session runs</strong>", not "what the model is". With one model, you can open several contexts with different cparams: one with long context, one short, one many-threaded, one few-threaded, each to its scenario.
Decoupling session parameters from the model is exactly for this flexibility.</p>

<h2>One inference step and reading the result</h2>
<p>With everything ready, running inference is repeatedly calling <span class="mono">llama_decode</span>. It eats a batch (L18), internally does graph-building, execution, and KV update in one go, then puts the logits into the context's output buffer.</p>
<pre class="code"><span class="cm"># pseudocode: one inference step (inside llama_decode)</span>
<span class="fn">llama_decode</span>(ctx, batch)              <span class="cm"># run one forward step</span>
<span class="cm">#   -&gt; split ubatch(L18) -&gt; build_graph(L16) -&gt; sched execute(L10) -&gt; update KV(L19)</span>
<span class="cm">#   -&gt; logits written into ctx's output buffer</span>
p = <span class="fn">llama_get_logits_ith</span>(ctx, i)    <span class="cm"># get the i-th token's logits(n_vocab-dim)</span></pre>
<p>Read the result with <span class="mono">llama_get_logits_ith(ctx, i)</span>, getting the i-th token's logits - an <span class="mono">n_vocab</span>-dimensional vector, the unnormalized scores for "the next word", handed next to sampling (L21) to pick one. This strings L16's graph-building, L10's execution, and L19's KV,
through this one <span class="mono">llama_decode</span> function, <strong>into one complete inference step</strong>.</p>
<p>So <span class="mono">llama_context</span> plays the role of "<strong>conductor + state ledger</strong>" in the whole machine: it knows all this session's config, remembers where it is, schedules the hardware, and collects the output. Understand it and you have truly <strong>assembled those earlier parts (model, graph, backend, KV) into a running inference machine</strong>.</p>
<p>This is also where L03's prefill/decode connects: <strong>both</strong> phases call <span class="mono">llama_decode</span>, differing only in the batch fed. Prefill feeds a whole prompt at once (many tokens), filling their K/V into the KV cache in one go, taking only the last one's logits; decode then feeds one new token each time and takes its logits.
One function, two rhythms, all expressed via the batch.</p>
<p>Drawing the whole autoregressive loop: <span class="mono">llama_decode</span> computes logits -&gt; sampling (L21) picks a token -&gt; that new token is wrapped into a new batch fed back to <span class="mono">llama_decode</span> -&gt; logits again... and so on, popping out the answer word by word.
The context is <strong>present throughout</strong> this loop: its KV cache grows step by step, its output buffer refreshes step by step. Generating one reply is this loop turning many times on one context.</p>
<p>A note on <span class="mono">llama_encode</span>: some models (e.g. with an encoder) need to encode first, then decode, so the API has both <span class="mono">llama_decode</span> and <span class="mono">llama_encode</span>. For the most common decoder-only large models (L04), you basically only use <span class="mono">llama_decode</span>. Just know this division exists, no need to dig in.</p>
<p>So what this lesson delivers is a leap <strong>from "parts" to "machine"</strong>: earlier lessons built parts like the model (L14-15) and graph (L16); this lesson uses the context to pack them into a loop that can turn repeatedly. By here you can run a full inference in your head: load model -&gt; build context -&gt; repeatedly decode + sample -&gt; output word by word.
The remaining lessons take this machine's key components (batch, KV cache, sampling...) apart one by one for the details.</p>
<p>Emphasize once more that "<strong>state advances</strong>", because it is the key to understanding autoregression. Each <span class="mono">llama_decode</span> is not a recompute from scratch but <strong>advancing one step on the context's existing state</strong>: the KV cache already holds all prior tokens' K/V, and this step only computes the new token and appends its K/V.
Precisely because the context keeps the state throughout, decode can be so fast as "compute only one token per step" (L03/L19).</p>
<p>To close on the context's role in one sentence: it is the switch that turns "a static model" into "dynamic inference". Without it, the model is just a pile of resting weights; with it, the weights are driven step by step, popping out token after token. From the next lesson, we dig into this machine's concrete components, starting with the batch fed to decode.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why split model and context? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The core is "<strong>share read-only, store state separately</strong>". Weights are read-only and several GB, and many sessions <strong>sharing one copy</strong> saves the most memory (with L13's mmap, even physical memory can be shared across processes); while KV cache and current position are <strong>per-session state</strong> that must be stored separately.</p>
    <p>Split this way, one copy of weights on a machine holds up many concurrent sessions: each user/request gets a new lightweight context (carrying only its own KV), while those several GB of weights stay shared and untouched. This is the basis for <span class="mono">llama-server</span> being concurrent and memory-frugal.</p>
    <p>Conversely, if not split - if weights and state were one object - every session would copy several GB of weights; serving dozens of users would need dozens of weight copies, simply unsustainable. A deceptively simple "split into two objects" upholds the feasibility of the whole multi-user service.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What are logits? Why only on some tokens? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">logits</span> are a set of <strong>unnormalized scores</strong> the model assigns for "who the next token should be", of length equal to the vocab size <span class="mono">n_vocab</span>. They are not yet probabilities (not normalized), but higher-scoring words are more likely to be chosen; sampling (L21) takes these logits to pick a word.</p>
    <p>The key: <strong>not every token needs logits computed</strong>. Computing logits means a big "hidden vector -&gt; vocab size" matmul, quite costly. And when prefill passes a whole prompt through, the middle tokens' logits are simply unused - we only want the <strong>last</strong> token's logits (to predict the next one).</p>
    <p>So which positions compute logits is controlled by the batch's output flag (L18). In decode each step adds one token and only it needs logits; prefill needs only the final position. <span class="mono">llama_get_logits_ith(ctx, i)</span> gets the logits of the i-th flagged-output position. This "compute output on demand" design saves a lot of useless big matmuls.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How does context string L16/L10/L19 into one step? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">llama_decode</span> is the "conductor". It takes this step's batch (L18), first uses batching logic to split it into physically-computable ubatches; for each ubatch, it calls <span class="mono">build_graph</span> (L16) to assemble this step's compute graph.</p>
    <p>Then it hands the graph to the context's <span class="mono">sched</span> (L10's backend scheduler) to actually execute; during execution, attention operators <strong>write this step's new token's K/V into the context's KV cache</strong> (L19) and read back historical K/V. When done, it writes the output logits into the context's output buffer.</p>
    <p>One round and the context's state <strong>advances by one step</strong>: the KV cache has remembered one more token, the output buffer has new logits. The next <span class="mono">llama_decode</span> continues advancing on this state. It is exactly the context gathering these scattered mechanisms (graph-building, execution, KV, output) and driving them in order that produces "step after step" autoregressive generation.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama_model</span> is <strong>read-only</strong> weights/hyperparameters/vocab (one copy, shareable by many contexts); <span class="mono">llama_context</span> is a <strong>stateful</strong> session (one per session).</li>
    <li>The context holds <span class="mono">cparams</span> (config) + <span class="mono">memory</span> (KV cache, L19) + <span class="mono">sched</span> (backend scheduling, L10) + <span class="mono">logits</span> (output buffer); it holds no weights, only references the model.</li>
    <li>cparams tunes <span class="mono">n_ctx</span>/<span class="mono">n_batch</span>/<span class="mono">type_k</span>/<span class="mono">type_v</span> etc., mostly trading VRAM against speed (<span class="mono">n_ctx</span> and KV quantization directly affect VRAM).</li>
    <li><span class="mono">llama_decode</span> runs one forward step (split ubatch -&gt; build graph -&gt; execute -&gt; update KV -&gt; emit logits); <span class="mono">llama_get_logits_ith</span> reads the i-th token's logits.</li>
    <li>Splitting model and context = one copy of weights holds up many sessions, the basis of <span class="mono">llama-server</span>'s concurrency.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Splitting "<strong>unchanging knowledge</strong>" (weights, model) from "<strong>session state</strong>" (KV/progress, context) into two objects - this seemingly ordinary cut pays off enormously: one copy of several-GB weights can be shared by many sessions, each carrying only a lightweight bit of state.
  So the same machine, the same model, can serve many users at once. This is another showing of the old truth "<strong>separate state from data</strong>" in an inference engine, and the most crucial design watershed between "can run one conversation" and "can hold up a service". Next lesson, we look at what that batch fed to <span class="mono">llama_decode</span> actually looks like.
</div>
""",
}



