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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  把 loader 放到整个推理流程里看，它的位置很特别：<strong>加载只发生一次</strong>，而后面的 decode（L17）会被反复调用成千上万次。正因为只做一次，loader 可以"慢工出细活"地把模型整理周全——读全 metadata、建好完整清单、接好所有数据指针；之后得到的 <span class="mono">llama_model</span> 是<strong>只读</strong>的，可以被许多次推理、甚至许多个会话反复复用，不必再碰磁盘。把"一次性的重活"和"反复做的快活"分开，是这套设计省时省力的根源。
</div>
<p>"整理成清单"这一步看似平淡，其实是把后面一切操作变简单的关键。磁盘上的张量只是一段段连续字节，彼此之间没有任何"我是谁"的信息；loader 给它们配上名字、记下位置、建好索引，
之后无论是按架构建图（L16）、还是热插拔 LoRA（L24 预告），都能<strong>按名字精确点到某一个张量</strong>。先把混沌整理成有序，后面才能从容操作——这是工程里很常见、却常被忽视的一步。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  这里也能看清几个层次的分工：最底下是 <span class="mono">gguf</span> 库（L13），只懂"GGUF 这种文件怎么解析"，连"权重"是什么都不关心；往上是 <span class="mono">ggml</span>（L08-L12），只懂张量和计算，不关心文件；而 loader 正好<strong>夹在两者之间</strong>——它用 gguf 库读出原始信息，再用 ggml 建出张量，把"文件世界"翻译成"张量世界"。理解 loader，其实就是理解这道翻译是怎么发生的。
</div>
<p>你平时跑 llama.cpp 时，启动那一大段刷屏的日志——"loaded meta data with N key-value pairs"、一行行张量名和类型——多半就是 loader 在汇报它的工作：读了多少 KV、识别出哪种架构、每个张量多大、用没用上 mmap。
下次看到这些日志，你就知道屏幕背后正是这一课讲的流程在跑。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  上面"四步"是<strong>概念上</strong>的顺序，真实代码里它们常交织在一起——比如读 tensor info 的同时就建好了 weights_map 的条目，建条目时就记下了 mmap 里的偏移。教学上分成四步是为了看清职责，工程上则是一遍扫描尽量把能做的都做了。理解了每一步"在干什么"，看真实代码时就不会被它们交错的写法绕晕。
</div>

<h2>读超参与张量</h2>
<p>loader 怎么知道模型有几层、多宽？不靠猜，全从 GGUF 的 metadata KV 里读——用一个模板方法 <span class="mono">get_key</span>，把"键"映射到具体的超参字段。键本身是自描述的（L13），下一课（L15）会讲这些键怎么按架构拼出来。</p>
<div class="cols">
  <div class="col"><h4>读超参</h4><p><span class="mono">get_key("llama.block_count")</span> 等 -&gt; 填进 <span class="mono">llama_hparams</span> 的字段（n_layer / n_embd ...，L15）。读的是 GGUF 头里的 <strong>metadata KV</strong>。</p></div>
  <div class="col"><h4>读张量</h4><p>按名字进 <span class="mono">weights_map</span> -&gt; <span class="mono">create_tensor</span> 建元数据 -&gt; <span class="mono">load_data_for</span> 落数据（mmap 零拷贝）。读的是 GGUF 头里的 <strong>tensor info</strong> 与数据段。</p></div>
</div>
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
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  还要留意 <span class="mono">create_tensor</span> 这一步的轻——它在 <span class="mono">ggml_context</span> 里建的只是张量的<strong>元数据</strong>（形状、类型、名字），并不为那几 GB 的浮点数据另开缓冲（这正是 L08 讲的 <span class="mono">no_alloc</span>）。于是装下"整个模型的骨架"只要几 MB 的 context，真正占地方的权重则由 mmap 映射承接。元数据归元数据、数据归数据，两者分头安放——这是 L08 内存观在加载阶段的直接兑现，也让"先把结构建全、数据按需就位"成为可能。
</div>
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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  为什么要分片？一是<strong>下载与分发</strong>友好：一个 200 GB 的模型切成几十个几 GB 的片，断点续传、并行下载、镜像同步都更容易；二是有些文件系统对单文件大小有上限，分片能绕开；三是方便<strong>按需取用</strong>。你在 HuggingFace 上看到的大模型，权重文件往往就是 <span class="mono">...-of-00010.gguf</span> 这样一长串。
</div>
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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  顺带一提：加载失败是有明确信号的——magic 不对、version 不认识、某个必需张量缺失，loader 都会当场报错而不是带病继续。这种"加载期就把问题暴露出来"的做法，比"跑到一半才崩"友好得多，也是把校验集中在 loader 这一层的好处。
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Placed in the whole inference pipeline, the loader's position is special: <strong>loading happens once</strong>, while the decode that follows (L17) is called thousands of times. Precisely because it runs once, the loader can take its time to organize the model thoroughly - read all metadata, build the complete list, wire up every data pointer; the resulting <span class="mono">llama_model</span> is <strong>read-only</strong> and can be reused across many inferences, even many sessions, without touching disk again. Separating "the one-time heavy work" from "the repeated fast work" is the root of this design's efficiency.
</div>
<p>This "organize into a list" step looks mundane but is the key to making everything later simple. On disk, tensors are just runs of contiguous bytes with no "who am I" information; the loader gives them names, records positions, and builds an index,
so that whether building a graph by architecture (L16) or hot-swapping a LoRA (L24), the engine can <strong>point precisely at one tensor by name</strong>. Turning chaos into order first is what lets everything afterward proceed calmly - a common but underappreciated step in engineering.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  You can also see the division of labor across layers here: at the bottom is the <span class="mono">gguf</span> library (L13), which only knows "how to parse a GGUF file" and cares nothing for what a "weight" is; above it is <span class="mono">ggml</span> (L08-L12), which only knows tensors and computation, not files; and the loader sits <strong>exactly between them</strong> - using the gguf library to read raw info, then ggml to build tensors, translating the "file world" into the "tensor world". Understanding the loader is really understanding how that translation happens.
</div>
<p>When you run llama.cpp, that wall of startup logs - "loaded meta data with N key-value pairs", lines of tensor names and types - is mostly the loader reporting its work: how many KVs it read, which architecture it recognized, how big each tensor is, whether mmap was used.
Next time you see those logs, you will know the process behind the screen is exactly what this lesson describes.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  The "four steps" above are a <strong>conceptual</strong> order; in real code they often interleave - e.g. reading a tensor info also builds its weights_map entry, and building the entry records the offset into the mmap. Splitting into four steps is for seeing the responsibilities clearly; in engineering, one pass does as much as it can at once. Once you grasp what each step "is doing", the interleaved real code will not confuse you.
</div>

<h2>Reading hyperparameters and tensors</h2>
<p>How does the loader know how many layers, how wide? Not by guessing - it reads it all from the GGUF metadata KVs, via a templated <span class="mono">get_key</span> that maps a "key" to a specific hyperparameter field. The keys are self-describing (L13); the next lesson (L15) covers how they are templated per architecture.</p>
<div class="cols">
  <div class="col"><h4>Read hyperparameters</h4><p><span class="mono">get_key("llama.block_count")</span> etc. -&gt; fill fields of <span class="mono">llama_hparams</span> (n_layer / n_embd ..., L15). Reads the <strong>metadata KV</strong> in the GGUF header.</p></div>
  <div class="col"><h4>Read tensors</h4><p>By name into <span class="mono">weights_map</span> -&gt; <span class="mono">create_tensor</span> builds metadata -&gt; <span class="mono">load_data_for</span> places data (mmap zero-copy). Reads the <strong>tensor info</strong> in the GGUF header, then the data section.</p></div>
</div>
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
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Note also how light the <span class="mono">create_tensor</span> step is - in the <span class="mono">ggml_context</span> it builds only a tensor's <strong>metadata</strong> (shape, type, name), reserving no buffer for those GB of float data (exactly L08's <span class="mono">no_alloc</span>). So holding "the whole model's skeleton" takes only a few MB of context, while the weights that actually take space are carried by the mmap mapping. Metadata as metadata, data as data, placed separately - the direct cash-out of L08's memory model at load time, and what makes "build the full structure first, bring data into place on demand" possible.
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Why split at all? First, <strong>download and distribution</strong> friendliness: a 200 GB model cut into dozens of few-GB shards is easier to resume, download in parallel, and mirror; second, some file systems cap single-file size, which splitting sidesteps; third, it eases <strong>selective loading</strong>. The large models you see on HuggingFace often have weight files like a long <span class="mono">...-of-00010.gguf</span> series.
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  By the way: load failures are signaled clearly - a wrong magic, an unknown version, a missing required tensor, and the loader errors out on the spot rather than limping on. This "surface problems at load time" approach is far friendlier than "crash halfway through", and is a benefit of centralizing validation in the loader layer.
</div>
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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  顺带感受一下这套机制的容量：llama.cpp 支持的架构早已是<strong>几十种</strong>——llama、qwen、mistral、phi、gemma、deepseek、stablelm…… 全都靠这一个 <span class="mono">llm_arch</span> 枚举区分。新模型层出不穷，可它们绝大多数都是 transformer 的变体、差异有限；于是"再多一种架构"在引擎眼里，往往只是枚举里多一个值、表里多几行。这种"用一个枚举撑起一个生态"的容量，正是表驱动设计的威力。
</div>
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
<div class="card warn">
  <div class="tag">⚠ 两个易踩的坑</div>
  <strong>①</strong> <span class="mono">n_layer()</span>、<span class="mono">n_head(il)</span> 这些不是普通字段，而是<strong>访问器方法</strong>（注意有括号）——现代架构里不同层的头数、注意力类型可能不一样（比如 GQA、滑窗层），头数按层存进 <span class="mono">n_head_arr</span> 这样的数组，取的时候要带层号 <span class="mono">il</span>；把它当成定值字段去用，迟早出错。<br>
  <strong>②</strong> <span class="mono">n_vocab</span>（词表大小）<strong>不在</strong> <span class="mono">llama_hparams</span> 里！它属于分词器，来自 <span class="mono">llama_vocab::n_tokens()</span>（L20 会讲）——记住：hparams 管的是"网络的形状"，词表是另一码事，归分词器管。
</div>
<p>这些超参一旦读出来，就成了整个推理的"尺寸基准"：建图（L16）时按 <span class="mono">n_layer()</span> 决定堆几层、按 <span class="mono">n_head(il)</span> 切多少个注意力头、按 <span class="mono">n_embd</span> 定各处矩阵的形状；
KV cache（L19）按层数和 KV 头数算该开多大。可以说，hparams 是把"一个抽象的 transformer"具体化成"这一个模型"的那组数字。</p>
<p>顺便厘清"参数量"和超参的关系。我们常说的 7B、70B，指的是模型权重里浮点数的<strong>总个数</strong>（70 亿、700 亿）；而这个总数，正是由超参算出来的——大致是 <span class="mono">n_layer</span> × 每层各权重矩阵尺寸之和，
而每个矩阵的尺寸又由 <span class="mono">n_embd</span>、<span class="mono">n_ff</span> 等决定。所以超参不是一堆孤立的数字，它们<strong>共同决定了模型有多大</strong>。读懂超参，你就能从一个模型的几个数，估出它要吃多少显存。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  表里没列全的超参还有不少，各有用处：<span class="mono">n_ctx_train</span> 是模型训练时的上下文长度（你能开多长上下文的参考上限）；<span class="mono">f_norm_rms_eps</span> 是 RMSNorm 里防止除零的小常数（L11 的归一化用到）；<span class="mono">n_rot</span> 是 RoPE 实际旋转的维数。这些数看着琐碎，却个个都会在建图（L16）时被某个算子精确用到——少一个、错一个，算出来的就不是这个模型了。
</div>
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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这也回答了上一课留的悬念——为什么 <span class="mono">weights_map</span> 要按名字索引。因为名字是<strong>稳定的契约</strong>：换个导出工具、张量排列变了也不怕，只要名字这套约定不变，建图就总能精确取到它要的那块权重。名字这层抽象，把"权重物理上躺在哪"和"逻辑上是哪个部件"彻底解耦了。
</div>
<p>把一层 transformer 的张量名列全，规律就更清楚了：注意力部分有 <span class="mono">attn_q</span>/<span class="mono">attn_k</span>/<span class="mono">attn_v</span>/<span class="mono">attn_output</span> 四个投影、加一个 <span class="mono">attn_norm</span>；FFN 部分有 <span class="mono">ffn_gate</span>/<span class="mono">ffn_up</span>/<span class="mono">ffn_down</span> 三个矩阵、加一个 <span class="mono">ffn_norm</span>。
每一层都按这套模板复制一遍，前面加 <span class="mono">blk.层号.</span>。看懂这张"一层有哪些权重"的清单，你就看懂了 transformer 一个 block 的全部可学习参数。</p>
<p>为什么用 <span class="mono">.</span> 点号分层级命名（<span class="mono">blk.0.attn_q.weight</span>）？因为这天然形成一棵<strong>层级树</strong>：<span class="mono">blk</span> 下是各层、层下是各部件、部件下是 weight/bias。这种命名既清晰、又方便按前缀批量匹配——
比如想找第 0 层的所有权重，匹配 <span class="mono">blk.0.</span> 前缀即可。一个好的命名约定，不只是"起个名"，而是把结构信息编码进了名字本身。</p>
<p>这套命名还是<strong>跨架构通用</strong>的：不管 llama 还是 qwen2，第 0 层的注意力 Q 投影都叫 <span class="mono">blk.0.attn_q</span>。正因为大家共享同一套名字，针对张量的通用工具（量化、转换、可视化）才能不区分架构地处理任意模型。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  再补一点 <span class="mono">LLM_TN</span> 的细节：它拼出的完整名字通常还带个后缀，区分 <span class="mono">.weight</span> 和 <span class="mono">.bias</span>——同一个部件可能有权重、也可能有偏置。所以 <span class="mono">tn(LLM_TENSOR_ATTN_Q, "weight", il)</span> 拼出的是 <span class="mono">blk.il.attn_q.weight</span>。把"部件名模板 + 后缀 + 层号"三者交给一个构造器统一拼，既避免了到处手写字符串容易出的错，也让"改个命名规则"只需动一处。
</div>
<p>顺便提一句：有些张量是<strong>可选</strong>的——比如不少现代模型的线性层没有 bias，那 <span class="mono">.bias</span> 那个张量在文件里就根本不存在。建图时按架构知道"这层该有哪些张量"，缺的可选项就跳过。
命名约定加上"哪些必需、哪些可选"的知识，才完整描述了一个架构的张量构成。</p>

<h2>自描述如何在架构层兑现</h2>
<p>把三样东西连起来看，L13 说的"<strong>自描述</strong>"就在架构层完整兑现了：<span class="mono">general.architecture</span> 选定 arch；带 <span class="mono">%s</span> 的 KV 键（如 <span class="mono">llama.block_count</span>）用架构名填模板、由 <span class="mono">get_key</span> 读出超参；
张量按 <span class="mono">LLM_TENSOR_NAMES</span> 命名一一对上。三套约定一咬合，loader 读出的"一堆张量"就成了"一个有名有姓、有形有状的具体模型"，随时可以交给 L16 建图。</p>
<p>值得回味的是这种设计的"<strong>表驱动</strong>"味道：架构是一张名字表、键是一张键名表、张量是一张张量名表。引擎的主干代码不写死任何一种模型，而是"照表办事"。于是支持一个新模型，多半不是改引擎，而是<strong>往这几张表里加几行 + 写一份建图</strong>（L16）。</p>
<div class="cols">
  <div class="col"><h4>架构名表</h4><p><span class="mono">LLM_ARCH_NAMES</span>：<span class="mono">"llama"</span> -&gt; <span class="mono">LLM_ARCH_LLAMA</span>。先认出"是什么模型"。</p></div>
  <div class="col"><h4>键名表</h4><p><span class="mono">LLM_KV</span>：<span class="mono">llama.block_count</span> 等 -&gt; 由 <span class="mono">get_key</span> 读成超参。再量出"有多大"。</p></div>
  <div class="col"><h4>张量名表</h4><p><span class="mono">LLM_TENSOR_NAMES</span>：<span class="mono">blk.N.attn_q</span> 等 -&gt; 对上每块权重。最后"零件对号入座"。</p></div>
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Get a feel for this mechanism's capacity: llama.cpp already supports <strong>dozens</strong> of architectures - llama, qwen, mistral, phi, gemma, deepseek, stablelm... - all distinguished by this one <span class="mono">llm_arch</span> enum. New models keep appearing, but the vast majority are transformer variants with limited differences; so "one more architecture" is, to the engine, often just one more enum value and a few more table rows. That capacity of "one enum holding up an ecosystem" is the power of table-driven design.
</div>
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
<div class="card warn">
  <div class="tag">⚠ Two easy traps</div>
  <strong>(1)</strong> <span class="mono">n_layer()</span>, <span class="mono">n_head(il)</span> are not plain fields but <strong>accessor methods</strong> (note the parentheses) - in modern architectures different layers may have different head counts or attention types (GQA, sliding-window), so head counts are stored per layer in arrays like <span class="mono">n_head_arr</span>, fetched with a layer index <span class="mono">il</span>; treat it as a constant field and you will eventually be wrong.<br>
  <strong>(2)</strong> <span class="mono">n_vocab</span> (vocab size) is <strong>not</strong> in <span class="mono">llama_hparams</span>! It belongs to the tokenizer, from <span class="mono">llama_vocab::n_tokens()</span> (L20) - remember: hparams govern "the shape of the network"; the vocab is a separate matter, owned by the tokenizer.
</div>
<p>Once read, these hyperparameters become the inference's "size baseline": graph-building (L16) stacks layers by <span class="mono">n_layer()</span>, splits heads by <span class="mono">n_head(il)</span>, sets matrix shapes by <span class="mono">n_embd</span>;
the KV cache (L19) sizes itself by layer count and KV head count. In short, hparams are the set of numbers that turn "an abstract transformer" into "this particular model".</p>
<p>While we are at it, untangle "parameter count" from hyperparameters. The 7B, 70B we casually say refers to the <strong>total count</strong> of floats in the model's weights (7 billion, 70 billion); and that total is computed from the hyperparameters - roughly <span class="mono">n_layer</span> x the sum of each layer's weight-matrix sizes,
where each matrix's size is set by <span class="mono">n_embd</span>, <span class="mono">n_ff</span>, etc. So hyperparameters are not isolated numbers; together they <strong>determine how big the model is</strong>. Read the hyperparameters and you can estimate a model's VRAM appetite from just a few numbers.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Plenty of hyperparameters the table omits each have their use: <span class="mono">n_ctx_train</span> is the context length the model was trained at (a reference ceiling for how long a context you can open); <span class="mono">f_norm_rms_eps</span> is the small constant in RMSNorm that avoids divide-by-zero (used by L11's normalization); <span class="mono">n_rot</span> is the number of dimensions RoPE actually rotates. These look trivial, yet each is used precisely by some operator at graph time (L16) - miss one or get one wrong, and what you compute is no longer this model.
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This also answers last lesson's cliffhanger - why <span class="mono">weights_map</span> is keyed by name. Because a name is a <strong>stable contract</strong>: a different export tool, a different tensor order, no problem - as long as the naming convention holds, graph-building can always fetch the exact weight it wants. That layer of naming fully decouples "where a weight physically sits" from "which part it logically is".
</div>
<p>List a transformer layer's tensor names in full and the pattern is clearer still: the attention part has four projections <span class="mono">attn_q</span>/<span class="mono">attn_k</span>/<span class="mono">attn_v</span>/<span class="mono">attn_output</span> plus an <span class="mono">attn_norm</span>; the FFN part has three matrices <span class="mono">ffn_gate</span>/<span class="mono">ffn_up</span>/<span class="mono">ffn_down</span> plus an <span class="mono">ffn_norm</span>.
Every layer replicates this template, prefixed with <span class="mono">blk.&lt;index&gt;.</span>. Understand this "what weights a layer has" list and you understand all the learnable parameters of one transformer block.</p>
<p>Why dot-separated hierarchical names (<span class="mono">blk.0.attn_q.weight</span>)? Because it naturally forms a <strong>hierarchy tree</strong>: under <span class="mono">blk</span> are the layers, under a layer the parts, under a part weight/bias. Such naming is both clear and convenient for prefix matching in bulk -
to find all of layer 0's weights, match the <span class="mono">blk.0.</span> prefix. A good naming convention is not just "giving a name"; it encodes structural information into the name itself.</p>
<p>This naming is also <strong>cross-architecture</strong>: whether llama or qwen2, layer 0's attention Q projection is named <span class="mono">blk.0.attn_q</span>. Because everyone shares one naming set, generic tensor tools (quantization, conversion, visualization) can process any model without caring about architecture.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A bit more <span class="mono">LLM_TN</span> detail: the full name it builds usually carries a suffix too, distinguishing <span class="mono">.weight</span> from <span class="mono">.bias</span> - a part may have a weight and possibly a bias. So <span class="mono">tn(LLM_TENSOR_ATTN_Q, "weight", il)</span> builds <span class="mono">blk.il.attn_q.weight</span>. Handing "part-name template + suffix + layer index" to one constructor avoids the errors of hand-writing strings everywhere, and means "changing a naming rule" touches just one place.
</div>
<p>By the way: some tensors are <strong>optional</strong> - many modern models' linear layers have no bias, so the <span class="mono">.bias</span> tensor simply does not exist in the file. At graph time, the architecture knows "which tensors this layer should have", and missing optional ones are skipped.
The naming convention plus knowledge of "which are required, which optional" together fully describe an architecture's tensor makeup.</p>

<h2>How self-description cashes out at the architecture layer</h2>
<p>Connect the three things and L13's "<strong>self-description</strong>" cashes out fully at the architecture layer: <span class="mono">general.architecture</span> selects the arch; <span class="mono">%s</span> keys (like <span class="mono">llama.block_count</span>) fill the template with the arch name and are read by <span class="mono">get_key</span> into hyperparameters;
tensors line up by their <span class="mono">LLM_TENSOR_NAMES</span> names. Once the three conventions mesh, the loader's "pile of tensors" becomes "a concrete model with names, shapes, and sizes", ready to hand to L16 for graph-building.</p>
<p>What is worth savoring is the <strong>table-driven</strong> flavor of this design: architecture is a name table, keys are a key-name table, tensors are a tensor-name table. The engine's trunk hard-codes no single model; it "acts by the tables". So supporting a new model is mostly not editing the engine, but <strong>adding a few rows to these tables + writing one graph builder</strong> (L16).</p>
<div class="cols">
  <div class="col"><h4>Arch-name table</h4><p><span class="mono">LLM_ARCH_NAMES</span>: <span class="mono">"llama"</span> -&gt; <span class="mono">LLM_ARCH_LLAMA</span>. First recognize "which model".</p></div>
  <div class="col"><h4>Key-name table</h4><p><span class="mono">LLM_KV</span>: <span class="mono">llama.block_count</span> etc. -&gt; read into hyperparameters by <span class="mono">get_key</span>. Then measure "how big".</p></div>
  <div class="col"><h4>Tensor-name table</h4><p><span class="mono">LLM_TENSOR_NAMES</span>: <span class="mono">blk.N.attn_q</span> etc. -&gt; line up each weight. Finally "slot the parts into place".</p></div>
</div>
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
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里也顺势澄清 <span class="mono">build_graph</span> 和 ggml 的关系：它<strong>不是</strong> ggml 的一部分，而是 llama 层站在 ggml 之上写的"组装逻辑"。ggml（L08-L11）提供张量、算子、建图原语；<span class="mono">build_graph</span> 用这些原语，按 transformer 的结构拼出一张具体的图。所以这一课本质是在讲"<strong>怎么用 ggml 这套积木，搭出一个真正的大模型</strong>"。
</div>
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
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  整个前向并非只有重复的层。<strong>开头</strong>有一步把 token id 查成词向量（<span class="mono">token_embd</span> 那张表，图输入之一）；<strong>结尾</strong>在最后一层之后，还有一次 <span class="mono">output_norm</span> 归一化、和一次投影到词表大小的 <span class="mono">output</span>（算出 logits，L17）。所以完整的图是"输入嵌入 -&gt; N 层 block -&gt; 输出归一 -&gt; 投影出 logits"，中间那 N 层才是我们重点拆的对象。
</div>
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
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  顺带看一眼 <span class="mono">build_ffn</span> 内部：现代 llama 类模型的前馈不是简单的"一升一降"，而是 <strong>SwiGLU</strong> 式的——<span class="mono">gate</span> 和 <span class="mono">up</span> 两个矩阵各把输入投影一次，<span class="mono">gate</span> 那路过一个激活函数后与 <span class="mono">up</span> 逐元素相乘，再由 <span class="mono">down</span> 投影回去。这就是为什么一层 FFN 有 gate/up/down 三个权重矩阵（L15 命名约定里见过）。<span class="mono">build_ffn</span> 把这套固定套路封好，建图时一句话搞定。
</div>
<p>再说说图输入和"叶子"的关系。L09 讲过，图里分两类节点：算出来的<strong>节点</strong>和不计算、只被读取的<strong>叶子</strong>。权重是叶子（加载时就备好了，L14），而图输入（词向量、位置）也是叶子——只不过它们的数据是<strong>每步填新的</strong>。
建图时把这些叶子的位置占好，执行时把当前这一步的数据填进去，同一张图就能算出不同的结果。</p>
<p>你会注意到 <span class="mono">build_attn</span> 有<strong>好几个重载</strong>。为什么？因为注意力有不少变体：要不要用 KV cache（prefill 的某些路径不用、decode 必用）、是标准多头还是 GQA、用不用滑动窗口……与其每种各写一遍完整注意力，
不如把"公共骨架 + 可选差异"做成几个重载，让各架构按需挑用。这又是一处"把差异收进可选项、把共性沉淀成积木"的体现。</p>
<p>图输入被做成一族类（<span class="mono">llm_graph_input_*</span> 都派生自一个共同接口）也有讲究：不同的输入有不同的"填法"——词向量要按 token id 查表、位置要按当前进度生成、KV 掩码要按因果规则算。
把每种输入的"怎么填"封进各自的类，执行前统一调一遍，图就准备好了。这让"图里需要哪些外部输入"变得可扩展、可组合。</p>

<h2>建图与执行，泾渭分明</h2>
<p>最后强调这一课最重要的一点：<span class="mono">build_graph</span> 只<strong>建</strong>、不<strong>算</strong>。它把算子的 op 和 src 填好（L09 的惰性建图），最后 <span class="mono">get_gf()</span> 交出一张 <span class="mono">ggml_cgraph</span>，至于真正逐节点执行，是 L10 后端的事。</p>
<div class="cols">
  <div class="col"><h4>建图（L16）</h4><p><span class="mono">build_graph</span> 只填 op/src，产出一张 <span class="mono">ggml_cgraph</span> <strong>结构</strong>——不碰数据、不做计算。写一遍，跨硬件通用。</p></div>
  <div class="col"><h4>执行（L10）</h4><p>后端 <span class="mono">sched</span> 拿这张图<strong>逐节点算</strong>，CPU / CUDA / Metal 各自把它跑快。换后端，不动建图。</p></div>
</div>
<p>这种"建图归建图、执行归执行"的分离，回报是巨大的：同一张图，能原封不动地跑在 CPU、CUDA、Metal 等天差地别的硬件上（L10 的后端调度），上层的模型逻辑只写一遍。也正因为建图不碰具体计算，
换一个后端、加一种新硬件，都<strong>不用动建图代码</strong>。L16 负责"拼出正确的图"，L10 负责"在某种硬件上把图算快"，两者各司其职，合起来才是完整的推理。</p>
<p>正因为建图只产出"结构"、不含数据，这张图在很多情况下还能被<strong>缓存复用</strong>：连续的 decode 步骤，每步都是"一个新 token"，图的结构一模一样，于是引擎可以复用上一张图的骨架、只换喂进去的输入，省下反复搭图的开销。这是"惰性建图 + 结构与数据分离"带来的又一个红利。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  把这一课放回整个推理循环里看：每生成一个 token，<span class="mono">llama_decode</span>（L17）大致就是"<strong>建图（L16）-&gt; 后端执行（L10）-&gt; 得到 logits -&gt; 采样（L21）出下一个 token</strong>"这么一圈。L16 是这圈里"把模型变成可算的图"那一环。理解了它，你就把"加载好的模型"和"真正跑起来的推理"接上了。下一课，我们就进到 <span class="mono">llama_context</span>，看这一圈是怎么转起来的。
</div>
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
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Let us also clarify <span class="mono">build_graph</span>'s relation to ggml: it is <strong>not</strong> part of ggml, but the "assembly logic" the llama layer writes on top of ggml. ggml (L08-L11) provides tensors, operators, and graph-building primitives; <span class="mono">build_graph</span> uses these primitives to assemble a concrete graph by the transformer structure. So this lesson is essentially about "<strong>how to use ggml's building blocks to assemble a real large model</strong>".
</div>
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
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  The whole forward is not only repeated layers. At the <strong>start</strong>, one step looks up token ids into token vectors (the <span class="mono">token_embd</span> table, one of the graph inputs); at the <strong>end</strong>, after the last layer, there is an <span class="mono">output_norm</span> and a projection to vocab size <span class="mono">output</span> (computing logits, L17). So the full graph is "input embedding -&gt; N blocks -&gt; output norm -&gt; project to logits", with those N middle layers being what we focus on dissecting.
</div>
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
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A glance inside <span class="mono">build_ffn</span>: modern llama-style models' feed-forward is not a simple "up then down" but <strong>SwiGLU</strong>-style - the <span class="mono">gate</span> and <span class="mono">up</span> matrices each project the input, the <span class="mono">gate</span> path passes an activation and is multiplied element-wise with <span class="mono">up</span>, then <span class="mono">down</span> projects back. This is why one FFN layer has three weight matrices gate/up/down (seen in L15's naming convention). <span class="mono">build_ffn</span> wraps this fixed routine, done in one line at graph time.
</div>
<p>More on graph inputs and "leaves". L09 covered two kinds of nodes: computed <strong>nodes</strong> and non-computed, only-read <strong>leaves</strong>. Weights are leaves (prepared at load, L14), and graph inputs (token vectors, positions) are leaves too - except their data is <strong>filled fresh each step</strong>.
Graph-building reserves these leaves' positions, execution fills in this step's data, and the same graph computes different results.</p>
<p>You will notice <span class="mono">build_attn</span> has <strong>several overloads</strong>. Why? Because attention has many variants: with or without KV cache (some prefill paths skip it, decode always uses it), standard multi-head or GQA, with or without a sliding window... Rather than write a full attention for each,
"a common skeleton + optional differences" is made into a few overloads each architecture picks from. Another instance of "fold differences into options, distill commonality into blocks".</p>
<p>Making graph inputs a family of classes (the <span class="mono">llm_graph_input_*</span> all derive from a common interface) is deliberate too: different inputs have different "fill methods" - token vectors look up by token id, positions are generated by current progress, the KV mask is computed by the causal rule.
Wrapping each input's "how to fill" into its own class, called uniformly before execution, readies the graph. This makes "which external inputs the graph needs" extensible and composable.</p>

<h2>Build and execute, sharply separated</h2>
<p>Finally, this lesson's most important point: <span class="mono">build_graph</span> only <strong>builds</strong>, never <strong>computes</strong>. It fills in the operators' op and src (L09's lazy build), then <span class="mono">get_gf()</span> hands out a <span class="mono">ggml_cgraph</span>; the actual node-by-node execution is the L10 backend's job.</p>
<div class="cols">
  <div class="col"><h4>Build (L16)</h4><p><span class="mono">build_graph</span> only fills op/src, yielding a <span class="mono">ggml_cgraph</span> <strong>structure</strong> - touching no data, doing no compute. Written once, universal across hardware.</p></div>
  <div class="col"><h4>Execute (L10)</h4><p>The backend <span class="mono">sched</span> takes the graph and <strong>computes node by node</strong>; CPU / CUDA / Metal each run it fast. Swap backends, leave build untouched.</p></div>
</div>
<p>This "build is build, execute is execute" separation pays off enormously: the same graph runs unchanged on wildly different hardware - CPU, CUDA, Metal (L10's backend scheduling) - with the upper model logic written once. And precisely because graph-building touches no concrete computation,
switching backends or adding new hardware needs <strong>no change to graph-building code</strong>. L16 "assembles the correct graph", L10 "computes the graph fast on some hardware" - each to its job, together making complete inference.</p>
<p>Precisely because graph-building yields only "structure", not data, this graph can in many cases be <strong>cached and reused</strong>: in consecutive decode steps each is "one new token", the graph's structure is identical, so the engine can reuse the previous graph's skeleton and only swap the inputs, saving the cost of rebuilding. Another dividend of "lazy build + structure-data separation".</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Put this lesson back into the whole inference loop: per generated token, <span class="mono">llama_decode</span> (L17) is roughly the round "<strong>build graph (L16) -&gt; backend execute (L10) -&gt; get logits -&gt; sample (L21) the next token</strong>". L16 is the "turn the model into a computable graph" link in that round. Understand it and you have joined "the loaded model" to "inference actually running". Next lesson, we enter <span class="mono">llama_context</span> to see how this round turns.
</div>
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
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  实现上，<span class="mono">llama_context</span> 内部持有一个指向 <span class="mono">llama_model</span> 的引用——它不复制权重，只是"借用"。所以新建一个 context 的代价很小：分配一些会话状态（主要是 KV cache 的空间），权重那几个 GB 一个字节都不用再读、不用再拷。这也是为什么 server 加一个并发连接，增量内存主要就是那一份 KV，而不是整个模型。
</div>
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
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  输出为什么是 <span class="mono">buffer_view&lt;float&gt;</span> 这种"视图"，而不是一个普通数组？因为输出的大小是动态的——这一步标了几个位置要输出，就有几行 logits（每行 <span class="mono">n_vocab</span> 个数）。用一个轻量的视图指向底层缓冲，既能灵活表示"这一步有几行输出"，又避免反复分配。<span class="mono">llama_get_logits_ith</span> 取的，就是这个视图里第 i 行。
</div>
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
<div class="card spark">
  <div class="tag">💡 实战</div>
  cparams 和模型本身的默认值也有联系。很多参数你可以填 0 表示"用模型的默认"——比如 <span class="mono">n_ctx</span> 填 0，就取模型训练时的上下文长度（L15 的 <span class="mono">n_ctx_train</span>）。这让你既能省心地用默认值，又能在需要时按显存覆盖它。配置的灵活性，就藏在这些"0 表示跟随模型"的约定里。
</div>
<p>表里没列全的 cparams 还有一些专门用途，比如 <span class="mono">pooling_type</span>（做 embedding 任务时怎么把 token 向量汇成一个句向量）、各种 RoPE 缩放参数（把上下文外推到训练长度之外）。
普通文本生成多半用默认就行，但它们的存在说明：context 不只服务"生成下一个词"，也能服务 embedding、长上下文外推等多种任务。</p>
<p>一个实用提醒：context 的内存占用，<strong>大头往往是 KV cache</strong>，而 KV cache 的大小由 <span class="mono">n_ctx</span>、层数、KV 头数、<span class="mono">type_k/type_v</span> 共同决定。所以当你发现显存不够，调小 <span class="mono">n_ctx</span> 或量化 KV，往往比换模型更立竿见影。这条经验，在 L19 会有更细的账。</p>
<p>为什么这些参数放在<strong>建 context 时</strong>配、而不是加载 model 时配？因为它们是"<strong>这次会话怎么跑</strong>"的事，而不是"模型是什么"的事。同一个 model，你可以用不同的 cparams 开多个 context：一个开长上下文、一个开短的，一个多线程、一个少线程，各按各的场景来。
把会话参数和模型解耦，正是为了这种灵活。</p>

<h2>一步推理与取结果</h2>
<p>万事俱备，跑推理就是反复调 <span class="mono">llama_decode</span>。它吃一个 batch（L18），内部把建图、执行、更新 KV 一气呵成，最后把 logits 放进 context 的输出缓冲。</p>
<div class="flow">
  <div class="node"><div class="nt">llama_batch</div><div class="nd">这步喂的 token<br>(L18)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">llama_decode</div><div class="nd">建图(L16)+执行(L10)<br>+更新 KV(L19)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits 缓冲</div><div class="nd">写进 context<br>输出缓冲</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_get_logits_ith</div><div class="nd">取第 i 个位置<br>(n_vocab 维)</div></div>
</div>
<pre class="code"><span class="cm"># 伪代码: 一步推理(llama_decode 内部)</span>
<span class="fn">llama_decode</span>(ctx, batch)              <span class="cm"># 跑一步前向</span>
<span class="cm">#   -&gt; 切 ubatch(L18) -&gt; build_graph(L16) -&gt; sched 执行(L10) -&gt; 更新 KV(L19)</span>
<span class="cm">#   -&gt; logits 写进 ctx 的输出缓冲</span>
p = <span class="fn">llama_get_logits_ith</span>(ctx, i)    <span class="cm"># 取第 i 个 token 的 logits(n_vocab 维)</span></pre>
<p>取结果用 <span class="mono">llama_get_logits_ith(ctx, i)</span>，拿到第 i 个 token 的 logits——一个 <span class="mono">n_vocab</span> 维的向量，是"下一个词"的未归一分数，接下来交给采样（L21）挑一个词。这就把 L16 的建图、L10 的执行、L19 的 KV，
通过 <span class="mono">llama_decode</span> 这一个函数<strong>串成了一步完整的推理</strong>。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  所以 <span class="mono">llama_context</span> 在整套机制里扮演的是"<strong>总指挥 + 状态本</strong>"：它知道这次会话的所有配置、记着算到哪、调度着硬件、收着输出。理解了它，你就把前面那些零件（model、graph、backend、KV）真正<strong>组装成了一台能转的推理机</strong>。
</div>
<p>这里也顺势把 L03 的 prefill/decode 接上：两个阶段<strong>都</strong>是调 <span class="mono">llama_decode</span>，区别只在喂的 batch。prefill 一次喂整段 prompt（多个 token），把它们的 K/V 一口气填进 KV cache，只取最后一个的 logits；decode 之后每次只喂一个新 token、取它的 logits。
同一个函数，两种节奏，全靠 batch 来表达。</p>
<p>把整个自回归循环画出来就是：<span class="mono">llama_decode</span> 算出 logits -&gt; 采样（L21）挑一个 token -&gt; 把这个新 token 包成一个新 batch 喂回 <span class="mono">llama_decode</span> -&gt; 再出 logits…… 如此往复，逐字蹦出回答。
context 在这整个循环里<strong>一直在</strong>：它的 KV cache 一步步变长、它的输出缓冲一步步刷新。一句对话的生成，就是这个循环在一个 context 上转了很多圈。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  补一句 <span class="mono">llama_encode</span>：有些模型（如带 encoder 的）需要先 encode、再 decode，所以 API 里既有 <span class="mono">llama_decode</span> 也有 <span class="mono">llama_encode</span>。对最常见的 decoder-only 大模型（L04），你基本只会用到 <span class="mono">llama_decode</span>。知道有这么个分工即可，不必深究。
</div>
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
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  In implementation, <span class="mono">llama_context</span> holds a reference to <span class="mono">llama_model</span> internally - it copies no weights, just "borrows". So creating a context is cheap: allocate some session state (mainly KV cache space), with not a byte of those several-GB weights re-read or re-copied. This is why a server adding one concurrent connection adds memory mainly for that one KV, not the whole model.
</div>
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
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Why is output a <span class="mono">buffer_view&lt;float&gt;</span> "view" rather than a plain array? Because output size is dynamic - however many positions this step flags for output, that many rows of logits (each <span class="mono">n_vocab</span> numbers). A lightweight view pointing into an underlying buffer both flexibly expresses "how many output rows this step has" and avoids repeated allocation. <span class="mono">llama_get_logits_ith</span> reads the i-th row of this view.
</div>
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
<div class="card spark">
  <div class="tag">💡 Tip</div>
  cparams also relate to the model's own defaults. Many parameters you can set to 0 to mean "use the model's default" - e.g. <span class="mono">n_ctx</span> set to 0 takes the model's trained context length (L15's <span class="mono">n_ctx_train</span>). This lets you both rely on defaults effortlessly and override by VRAM when needed. Configuration flexibility hides in these "0 means follow the model" conventions.
</div>
<p>cparams the table omits have specialized uses too, such as <span class="mono">pooling_type</span> (how to pool token vectors into one sentence vector for embedding tasks) and various RoPE scaling parameters (extrapolating context beyond the trained length).
Plain text generation mostly uses defaults, but their existence shows the context serves not only "generate the next word" but also embedding, long-context extrapolation, and more.</p>
<p>A practical reminder: a context's memory is <strong>mostly the KV cache</strong>, whose size is set jointly by <span class="mono">n_ctx</span>, layer count, KV head count, and <span class="mono">type_k/type_v</span>. So when you hit a VRAM wall, shrinking <span class="mono">n_ctx</span> or quantizing KV is often more effective than swapping models. We will do this account in more detail in L19.</p>
<p>Why are these parameters configured <strong>at context creation</strong>, not at model load? Because they are about "<strong>how this session runs</strong>", not "what the model is". With one model, you can open several contexts with different cparams: one with long context, one short, one many-threaded, one few-threaded, each to its scenario.
Decoupling session parameters from the model is exactly for this flexibility.</p>

<h2>One inference step and reading the result</h2>
<p>With everything ready, running inference is repeatedly calling <span class="mono">llama_decode</span>. It eats a batch (L18), internally does graph-building, execution, and KV update in one go, then puts the logits into the context's output buffer.</p>
<div class="flow">
  <div class="node"><div class="nt">llama_batch</div><div class="nd">tokens fed this step<br>(L18)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">llama_decode</div><div class="nd">build(L16)+execute(L10)<br>+update KV(L19)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">logits buffer</div><div class="nd">into context's<br>output buffer</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">llama_get_logits_ith</div><div class="nd">get i-th position<br>(n_vocab-dim)</div></div>
</div>
<pre class="code"><span class="cm"># pseudocode: one inference step (inside llama_decode)</span>
<span class="fn">llama_decode</span>(ctx, batch)              <span class="cm"># run one forward step</span>
<span class="cm">#   -&gt; split ubatch(L18) -&gt; build_graph(L16) -&gt; sched execute(L10) -&gt; update KV(L19)</span>
<span class="cm">#   -&gt; logits written into ctx's output buffer</span>
p = <span class="fn">llama_get_logits_ith</span>(ctx, i)    <span class="cm"># get the i-th token's logits(n_vocab-dim)</span></pre>
<p>Read the result with <span class="mono">llama_get_logits_ith(ctx, i)</span>, getting the i-th token's logits - an <span class="mono">n_vocab</span>-dimensional vector, the unnormalized scores for "the next word", handed next to sampling (L21) to pick one. This strings L16's graph-building, L10's execution, and L19's KV,
through this one <span class="mono">llama_decode</span> function, <strong>into one complete inference step</strong>.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  So <span class="mono">llama_context</span> plays the role of "<strong>conductor + state ledger</strong>" in the whole machine: it knows all this session's config, remembers where it is, schedules the hardware, and collects the output. Understand it and you have truly <strong>assembled those earlier parts (model, graph, backend, KV) into a running inference machine</strong>.
</div>
<p>This is also where L03's prefill/decode connects: <strong>both</strong> phases call <span class="mono">llama_decode</span>, differing only in the batch fed. Prefill feeds a whole prompt at once (many tokens), filling their K/V into the KV cache in one go, taking only the last one's logits; decode then feeds one new token each time and takes its logits.
One function, two rhythms, all expressed via the batch.</p>
<p>Drawing the whole autoregressive loop: <span class="mono">llama_decode</span> computes logits -&gt; sampling (L21) picks a token -&gt; that new token is wrapped into a new batch fed back to <span class="mono">llama_decode</span> -&gt; logits again... and so on, popping out the answer word by word.
The context is <strong>present throughout</strong> this loop: its KV cache grows step by step, its output buffer refreshes step by step. Generating one reply is this loop turning many times on one context.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A note on <span class="mono">llama_encode</span>: some models (e.g. with an encoder) need to encode first, then decode, so the API has both <span class="mono">llama_decode</span> and <span class="mono">llama_encode</span>. For the most common decoder-only large models (L04), you basically only use <span class="mono">llama_decode</span>. Just know this division exists, no need to dig in.
</div>
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

LESSON_18 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课的 <span class="mono">llama_decode</span> 每次吃一个 <span class="mono">llama_batch</span>。这一课就拆开它：一个 batch 怎么同时装多个 token、每个 token 怎么带上"<strong>第几位、属于哪条序列、要不要输出</strong>"，
以及内部怎么被 <span class="mono">llama_batch_allocr</span> 切成小批（ubatch）喂给计算图。batch 是你和引擎之间那张"<strong>这一步要算什么</strong>"的订单。
</p>
<p style="color:var(--muted);margin-top:.4rem">为什么 batch 值得单讲？因为它是<strong>一个统一的接口</strong>：单条对话逐字 decode、多序列并行、prefill 整段 prompt——这些看着很不同的场景，到引擎眼里都只是"喂进来一个 batch"，区别全在 batch 里怎么填。搞懂 batch，你就懂了引擎"一次该算什么"是怎么被描述的。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  <span class="mono">llama_batch</span> 像一张<strong>点单</strong>：每个 token 是一道菜，<span class="mono">pos</span> 是上菜顺序（第几位）、<span class="mono">seq_id</span> 是哪一桌（多序列并行）、<span class="mono">logits</span> 标志是"这道要不要打包带走（输出结果）"。
  厨房（decode）拿到一张大单，会按自己一次能做几道菜的产能（<span class="mono">n_ubatch</span>），把它拆成一锅锅小批（ubatch）逐批做出来。
</div>

<h2>llama_batch 是什么</h2>
<p>先看这张订单的字段。一个 <span class="mono">llama_batch</span> 里，几个并行的数组共同描述"这一步要处理哪些 token、各自什么情况"。</p>
<table class="t">
  <tr><th>字段</th><th>含义</th></tr>
  <tr><td><span class="mono">n_tokens</span></td><td>这一批有多少个 token</td></tr>
  <tr><td><span class="mono">token[]</span></td><td>每个 token 的词 id</td></tr>
  <tr><td><span class="mono">pos[]</span></td><td>每个 token 的位置（喂给 rope 和 KV cache）</td></tr>
  <tr><td><span class="mono">n_seq_id[]</span> / <span class="mono">seq_id[][]</span></td><td>每个 token 属于哪条（或哪些）序列</td></tr>
  <tr><td><span class="mono">logits[]</span></td><td>标志：这个 token 是否要算输出 logits</td></tr>
</table>
<pre class="code"><span class="cm">// 简化自 include/llama.h</span>
<span class="kw">struct</span> llama_batch {
    int32_t        n_tokens;
    llama_token *  token;     <span class="cm">// 词 id</span>
    llama_pos   *  pos;       <span class="cm">// 每个 token 的位置</span>
    int32_t     *  n_seq_id;  llama_seq_id ** seq_id;  <span class="cm">// 属于哪条序列</span>
    int8_t      *  logits;    <span class="cm">// 标志: 是否输出 logits(源码注释: rename to "output")</span>
};</pre>
<p>注意这是几个<strong>平行数组</strong>：<span class="mono">token[i]</span>、<span class="mono">pos[i]</span>、<span class="mono">seq_id[i]</span>、<span class="mono">logits[i]</span> 合起来，描述第 i 个 token 的全部信息。这种"结构数组"的布局，让一次塞进很多 token 变得简单——你只要把这几个数组按相同的下标填好，引擎就知道这一步要处理哪些 token、各自怎么对待。</p>
<p>最常用的构造是 <span class="mono">llama_batch_get_one</span>：把一串 token id 包成一个最简 batch（位置自动从 0 递增、单序列、只最后一个出 logits），适合"喂一段 prompt 或一个新 token"这种最常见的情形。需要更精细控制（多序列、自定义位置）时，再用 <span class="mono">llama_batch_init</span> 手动填那几个数组。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  为什么用几个平行数组、而不是一个"token 对象"的数组？这是性能上的考量。把所有 token 的同一种属性（比如所有 pos）连续放在一起，对 CPU 缓存友好、也方便整批一次性传给后端、整列做向量化处理。这种"<strong>结构数组</strong>"（SoA）布局在高性能数值代码里很常见，和 L05 张量"一块连续内存"的思路一脉相承。
</div>
<p>字段里还有个 <span class="mono">embd</span>（上面简化时略过了）：大多数时候你喂的是 token 的<strong>词 id</strong>（走 <span class="mono">token</span>），但有些场景（比如多模态、或外部已算好嵌入）想直接喂<strong>嵌入向量</strong>，就走 <span class="mono">embd</span>。两者二选一：要么给 id 让模型自己查嵌入，要么直接给嵌入。普通文本生成走 token 即可。</p>
<p>还要理解 batch 的定位：它是一个<strong>纯数据的输入容器</strong>，没有任何方法、不做任何计算。它只负责"把这一步要算的东西描述清楚"，真正的活全在 <span class="mono">llama_decode</span> 里。把"描述要算什么"和"真正去算"分成两个东西（batch 和 decode），是个清爽的接口设计——你填一张表，引擎照表干活。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  这几个数组是"<strong>按列对齐</strong>"的：第 i 个 token 的词 id、位置、序列、输出标志，分别在四个数组的第 i 位。填 batch 时最容易错的，就是某个数组少填一位、或下标对不齐——一旦错位，引擎就会把某个 token 的位置安到另一个 token 头上。所以手动填 batch 时，务必让这几个数组的长度和顺序严格一致。
</div>
<p>具体感受一下：你在聊天框里发一句话，上层会先把它分词成一串 token id（L20），再把这些 id 填进一个 batch 的 <span class="mono">token</span> 数组、位置填进 <span class="mono">pos</span>、都归到同一个 <span class="mono">seq_id</span>，最后只在末位标 <span class="mono">logits</span>。这个 batch 一交给 <span class="mono">llama_decode</span>，模型就开始算"你这句话之后该接什么"。日常每一次对话，背后都是这样一张张 batch 在流转。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这套"一张表喂进去"的接口还有个隐性好处：它把"准备数据"和"跑模型"<strong>解耦</strong>了。准备 batch 可以在 CPU 上慢慢攒、可以由上层框架（甚至别的语言）来做，而 <span class="mono">llama_decode</span> 只认这张表、不关心它怎么来的。正因为接口这么简单清晰，各种语言绑定、各种上层服务才能轻松地接到 llama.cpp 上。
</div>

<h2>输出标志：不是每个 token 都出 logits</h2>
<p>那个 <span class="mono">logits</span> 数组是这一课的一个关键。它是个<strong>开关数组</strong>：<span class="mono">logits[i]=1</span> 表示"我要第 i 个 token 的输出 logits"，<span class="mono">=0</span> 则表示"算它，但不用给我它的 logits"。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>只有标记位才出 logits</b>：prefill 整段 prompt，往往只标最后一个</div>
  <div class="cells"><span class="lab">token</span><span class="cell">t0 (0)</span><span class="cell">t1 (0)</span><span class="cell">t2 (0)</span><span class="cell hl">t3 (1)</span><span class="lab">只 t3 出 logits</span></div>
</div>
<p>为什么要这个开关？因为算 logits 不便宜——它是一次"隐藏向量 -&gt; 词表大小（<span class="mono">n_vocab</span>）"的大矩阵乘。而很多 token 的 logits 我们根本用不到：prefill 把整段 prompt 过一遍时，中间那些 token 只是为了把 K/V 填进 KV cache，<strong>只有最后一个</strong>的 logits 才用来预测下一个词。</p>
<p>所以这个标志直接省算力：标了几个位置，就只算几次输出投影，其余 token 算到隐藏向量为止、不做那次大矩阵乘。decode 阶段每步只新增一个 token、只它要 logits；prefill 整段只要末位。这套"<strong>按需算输出</strong>"，正是 L17 讲 logits 时说的"只在某些 token 上有"的来源。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  从实现看，引擎会数一遍 batch 里标了多少个输出位置，记成 <span class="mono">n_outputs</span>，然后只为这么多行准备输出缓冲、只做这么多次输出投影。L17 讲的那个 <span class="mono">buffer_view&lt;float&gt; logits</span>，其行数正是 <span class="mono">n_outputs</span>、每行 <span class="mono">n_vocab</span> 个数。所以"标了几个"直接决定了"输出缓冲多大、投影做几次"。
</div>
<p>顺带解释源码里那句 <span class="mono">rename this to "output"</span> 的注释：<span class="mono">logits</span> 这个名字其实有点窄——这个标志控制的是"<strong>要不要这个位置的输出</strong>"，而输出既可以是 logits（生成任务），也可以是嵌入向量（embedding 任务，L17 的 embd）。叫 output 更准确。知道这点，你看到 logits 这个字段名时就不会被它字面意思框住。</p>
<p>多序列场景下，这个标志更显灵活：同时给三个不同 prompt 各做 prefill，可以把它们的 token 全塞进一个 batch，只在<strong>每条序列各自的最后一个</strong> token 上标 1。一次 decode，三条序列的下一个词 logits 都拿到了。这种"一批里多条序列、各取各的输出"的玩法，正是服务器批量处理多个请求的基础。</p>
<div class="card warn">
  <div class="tag">⚠ 性能坑</div>
  如果你<strong>每个</strong> token 都标了输出，会怎样？引擎就会老老实实为每个位置都做一次词表大小的投影——prefill 一段长 prompt 时，这是巨大的浪费。所以除非你真的需要每个位置的输出（某些特殊任务），否则务必只标你要的那几个。这个小小的 <span class="mono">int8</span> 数组填得对不对，直接关系到 prefill 快不快。
</div>
<p>顺带一提，<span class="mono">logits</span> 标志和 <span class="mono">seq_id</span> 配合，能表达很精细的需求：比如一个 batch 里有三条序列，你可以只要其中两条的输出、第三条只填 KV 不要输出。这种"<strong>逐 token 级别的精确控制</strong>"，是把多个不同请求高效拼在一起算的前提——服务器正是靠这种精细，才能在一次 decode 里同时推进很多条对话。</p>
<p>把输出标志这件事和显存也连一下：<span class="mono">n_outputs</span> 越大，输出缓冲（每行 n_vocab 个 float）就越占内存。对词表几十万的大模型，多标几行输出，缓冲就多吃不少。所以"只标该标的"，省的不只是计算，还有那块输出缓冲的内存。又一个"精打细算"体现在一个小标志上的例子。</p>

<h2>切成 ubatch：从逻辑批到物理批</h2>
<p>你提交的 batch（逻辑上"这一步要算这么多 token"）不一定能一次性塞进硬件算。引擎用 <span class="mono">llama_batch_allocr</span> 把它<strong>校验、补全、再切成物理可算的小批</strong>（ubatch），逐个送进计算图。</p>
<div class="flow">
  <div class="node"><div class="nt">llama_batch</div><div class="nd">你提交的逻辑批</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">batch_allocr.init</div><div class="nd">校验 pos/seq_id<br>填默认</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">split_simple(n_ubatch)</div><div class="nd">按物理批大小切</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">逐个 ubatch</div><div class="nd">建图(L16)+执行(L10)</div></div>
</div>
<pre class="code"><span class="cm"># 伪代码: batch 切成 ubatch(llama_decode 内部)</span>
alloc = <span class="fn">llama_batch_allocr</span>()
alloc.<span class="fn">init</span>(batch)                       <span class="cm"># 校验 pos/seq_id, 填默认</span>
<span class="kw">for</span> ub <span class="kw">in</span> alloc.<span class="fn">split_simple</span>(n_ubatch):  <span class="cm"># 按物理批大小切成 ubatch</span>
    <span class="fn">decode_ubatch</span>(ub)                   <span class="cm"># 建图(L16)+执行(L10)</span></pre>
<p>这里要分清两个"批大小"：<span class="mono">n_batch</span> 是<strong>逻辑</strong>批——你一次最多能提交多少 token；<span class="mono">n_ubatch</span> 是<strong>物理</strong>批——硬件一次真正高效处理多少 token。前者方便你"一次多交点活"，后者受限于硬件，allocr 就负责把大的逻辑批切成若干个物理批逐个算。</p>
<p>除了最简单的 <span class="mono">split_simple</span>，allocr 还有 <span class="mono">split_equal</span>、<span class="mono">split_seq</span> 等切法，应对多序列等更复杂的排布。但核心思想都一样：<strong>把"你想算的"翻译成"硬件能一口口吃下的"</strong>。这层切分，让上层只管描述意图、不用操心硬件一次能吃多少。</p>
<p>多说一句 <span class="mono">init</span> 这一步的"校验、补全"。它会检查你填的 pos、seq_id 合不合法（比如位置不能乱、序列号不能越界），并为你没填的字段补上合理默认（比如 pos 留空就按顺序自动编号）。这层把关，让上层调用方少踩坑——很多"喂错 batch"的错误，会在这里被当场拦下，而不是带着错继续算。</p>
<table class="t">
  <tr><th>切法</th><th>怎么排</th><th>适用</th></tr>
  <tr><td><span class="mono">split_simple</span></td><td>按顺序切</td><td>普通单序列（最常见）</td></tr>
  <tr><td><span class="mono">split_equal</span></td><td>多序列时各序列尽量均匀分布到每个 ubatch</td><td>多序列并行</td></tr>
  <tr><td><span class="mono">split_seq</span></td><td>把同一序列的 token 切到一起</td><td>recurrent 类模型（L19 变体）</td></tr>
</table>
<p>为什么要分这么细？因为像 recurrent（L19 变体）这类模型对"同序列 token 要连续"有要求，不同切法是为了照顾不同模型的约束。普通模型用 <span class="mono">split_simple</span> 就够。</p>
<p>"逻辑批 vs 物理批"这层区分，其实是计算机里很常见的"<strong>提交</strong>和<strong>执行</strong>解耦"：你按方便提交一大批，系统按自己的节奏分批执行。数据库的批量插入、GPU 的 kernel 启动，都是类似套路。llama.cpp 把它用在 token 上：你按一句话、一段 prompt 的粒度提交，引擎按硬件能吃的粒度执行。</p>
<p>每个 ubatch 会触发一次完整的"建图 + 执行"：<span class="mono">llama_decode</span> 对每个 ubatch 调 <span class="mono">build_graph</span>（L16）搭出针对这批 token 的图、交后端执行（L10）。所以"一个大 batch"在内部可能变成"好几张图轮流跑"。理解了这点，你就明白为什么 batch 切分发生在 decode 内部、而不用你操心——它是连接"你的订单"和"实际计算"的那道自动工序。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  实践里 <span class="mono">n_batch</span> 和 <span class="mono">n_ubatch</span> 怎么设？一般 <span class="mono">n_batch</span> 设大些（让你能一次提交长 prompt），<span class="mono">n_ubatch</span> 按显存和后端调——大了一次算得多更快、但更吃显存。两者都是 L17 的 cparams 成员，在建 context 时定。对大多数人，用默认值就好；只有在压榨吞吐或显存吃紧时，才需要细调它俩。
</div>
<p>把 batch 放回整条链路：是<strong>你</strong>（或上层框架）准备 batch -&gt; <span class="mono">llama_decode</span> 吃 batch、切 ubatch、建图执行 -&gt; logits 出来 -&gt; 采样（L21）挑词 -&gt; 把新词包成下一个 batch…… batch 就是这条循环里"每一圈的输入"。读懂它，你就握住了和引擎对话的那张"订单格式"。</p>
<p>一个常见的节奏：开头 prefill 时，把整段 prompt 的几十上百个 token 一次塞进一个<strong>大 batch</strong>（高效地一次填满 KV cache）；之后 decode 时，每步只喂一个新 token 的<strong>小 batch</strong>。同一个 <span class="mono">llama_batch</span> 结构，一会儿装很多、一会儿装一个——它的弹性，正好贴合 prefill/decode 这一快一慢的两段节奏（L03）。</p>
<p>顺带埋个伏笔：服务器为了榨干吞吐，会玩一种"<strong>连续批处理</strong>"（continuous batching）——把<strong>多个用户、不同进度</strong>的请求，按 seq_id 拼进同一个 batch 一起算，谁生成完了就把谁换出、把新请求换进。这套高级调度（第五部分会提）能成立，底层正是因为 batch 支持"一批里多条序列、各自独立"。你现在学的这个朴素的 batch 结构，撑起的是相当复杂的服务能力。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  "批处理"（batch）这个词本身也点明了思路：与其一个 token 一个 token 地单独喂、单独算（那样每次的固定开销都要重付一遍），不如<strong>攒一批一起算</strong>，把固定开销摊薄。prefill 之所以能比 decode 快很多，正是因为它把整段 prompt 攒成一大批并行算；而 decode 受"必须等上一个词"的制约（L04），只能一个一个来，享受不到批的红利。
</div>
<p>还有个实践细节：decode 循环里，每步那个"只装一个新 token"的小 batch，常常是<strong>复用同一块</strong> batch 内存反复填的——不必每步都重新分配。配合上 L16 说的"图结构可复用"，连续 decode 其实相当轻量：同一张图、同一个 batch 壳子，每步只换里头那一个 token id 和位置。这就是逐字生成能那么快的工程细节之一。</p>
<p>再把 ubatch 这个词的来历点破：u 是 "micro"（微）的意思，<span class="mono">ubatch</span> 就是"微批"。它和 <span class="mono">batch</span> 的关系，就像"你下的一整单"和"厨房一锅锅做的小份"。这个命名本身就提示了它的角色——它是 batch 在物理执行层面被切细后的产物，是真正一次性送进硬件计算的最小单位。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> pos 和 seq_id 各有什么用？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">pos</span> 是每个 token 的<strong>位置</strong>。它有两个去处：一是喂给 rope（L16），让注意力知道两个 token 相距多远；二是写进 KV cache 的 cell（L19），标记"这个 K/V 是第几位的"。所以 pos 填错，位置编码和缓存都会乱。</p>
    <p><span class="mono">seq_id</span> 标明每个 token 属于<strong>哪条序列</strong>。一个 batch（和一个 context）可以<strong>同时</strong>装好几条不同的序列——比如同时给三个不同 prompt 生成回答。它们共享这份权重和这套调度，但各有各的 KV（按 seq_id 区分，L19），互不串味。</p>
    <p>正是 <span class="mono">pos</span> + <span class="mono">seq_id</span> 这两样，让 batch 能精确表达"哪个 token、在哪条序列的第几位"。有了这个，多序列并行、同一序列里续写，都能在一个统一的 batch 接口里表达出来。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么要 ubatch 这层切分？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为硬件一次能高效处理的 token 数是<strong>有限</strong>的（受显存、计算单元规模限制），这个上限就是 <span class="mono">n_ubatch</span>。如果你一次提交了很多 token（大的逻辑批），不切就可能塞不下、或者塞下了也不高效。</p>
    <p>所以 allocr 把大的逻辑批切成若干个不超过 <span class="mono">n_ubatch</span> 的物理批，逐个算、把结果拼起来。对你来说，提交多少 token（<span class="mono">n_batch</span>）是"我想一次交多少活"的事；硬件一次算多少（<span class="mono">n_ubatch</span>）是"机器一口能吃多少"的事——两者解耦，互不打架。</p>
    <p>这层切分还带来灵活：同样一段 prompt，在显存大的机器上可以用大 <span class="mono">n_ubatch</span> 一次多算、更快；显存小就用小 <span class="mono">n_ubatch</span> 多切几次、慢一点但跑得起来。把"逻辑意图"和"物理执行"分开，正是这种"同一份代码适配不同硬件"的底气。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 输出标志怎么省算力？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>关键在于"算 logits"是一次<strong>昂贵</strong>的操作：把最后的隐藏向量投影到词表大小（几万维），是一次大矩阵乘。如果每个 token 都做这一步，prefill 一段长 prompt 就要做几百上千次无用的大投影。</p>
    <p>而我们真正需要 logits 的位置很少：decode 阶段每步只新增一个 token、只它要预测下一个；prefill 整段也只要最后一个。<span class="mono">logits</span> 标志就让引擎<strong>只在标了的位置</strong>做输出投影，其余 token 算到隐藏向量就停，省下大量大矩阵乘。</p>
    <p>这和 L03 讲的 prefill/decode 节奏正好对应：prefill 是"把整段 prompt 一次过完、只取末位 logits"，decode 是"逐字生成、每步取新词的 logits"。两种节奏，都靠这个标志数组在 batch 层面精确表达"这一步谁要输出"。一个 <span class="mono">int8</span> 数组，省下的是实打实的算力。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama_batch</span> 用几个<strong>平行数组</strong>装多 token：<span class="mono">token</span>/<span class="mono">pos</span>/<span class="mono">seq_id</span>/<span class="mono">logits</span>，第 i 列描述第 i 个 token。</li>
    <li><span class="mono">logits</span> 是<strong>输出标志</strong>（源码注释将改名 output）：只在标了的位置算输出投影，省掉大量大矩阵乘。</li>
    <li><span class="mono">pos</span> 喂 rope/KV（位置），<span class="mono">seq_id</span> 标序列——支持<strong>多序列并行</strong>（共享权重、各有 KV）。</li>
    <li><span class="mono">llama_batch_allocr</span> 把逻辑批 <span class="mono">init</span> 后 <span class="mono">split_*</span> 成物理 <span class="mono">ubatch</span>（&lt;= <span class="mono">n_ubatch</span>）逐个喂图。</li>
    <li><span class="mono">n_batch</span>（逻辑：一次提交多少）vs <span class="mono">n_ubatch</span>（物理：一次真正算多少），两者解耦。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"<strong>喂什么</strong>"（batch：哪些 token、哪条序列、谁要输出）和"<strong>怎么分块算</strong>"（ubatch 切分）干净地分开——于是同一套 <span class="mono">llama_decode</span> 既能跑单条对话的逐字 decode、也能多序列并行、还能 prefill 整段 prompt，
  全靠一个统一的 batch 接口来描述意图。一个好的接口，就该这样：用一种结构，表达尽可能多的场景，把"想算什么"和"怎么算"解耦。下一课，我们看这些 token 的 K/V 被记到哪——KV cache。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson's <span class="mono">llama_decode</span> eats one <span class="mono">llama_batch</span> each time. This lesson takes it apart: how a batch holds many tokens at once, how each token carries "<strong>which position, which sequence, output or not</strong>",
and how it is internally split by <span class="mono">llama_batch_allocr</span> into small batches (ubatch) fed to the compute graph. The batch is the "<strong>what to compute this step</strong>" order between you and the engine.
</p>
<p style="color:var(--muted);margin-top:.4rem">Why a whole lesson on the batch? Because it is <strong>one unified interface</strong>: word-by-word decode of a single conversation, multi-sequence parallelism, prefill of a whole prompt - these seemingly different scenarios are, to the engine, just "a batch fed in", with all the difference in how the batch is filled. Get the batch and you get how "what to compute at once" is described.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  <span class="mono">llama_batch</span> is like an <strong>order ticket</strong>: each token is a dish, <span class="mono">pos</span> is the serving order (which position), <span class="mono">seq_id</span> is which table (multi-sequence parallelism), the <span class="mono">logits</span> flag is "take this one to go (output the result) or not".
  The kitchen (decode), given a big order, splits it into small batches (ubatch) by how many dishes it can make at once (<span class="mono">n_ubatch</span>), cooking them batch by batch.
</div>

<h2>What is a llama_batch</h2>
<p>First, this order's fields. In a <span class="mono">llama_batch</span>, several parallel arrays together describe "which tokens this step processes, each in what situation".</p>
<table class="t">
  <tr><th>field</th><th>meaning</th></tr>
  <tr><td><span class="mono">n_tokens</span></td><td>how many tokens in this batch</td></tr>
  <tr><td><span class="mono">token[]</span></td><td>each token's word id</td></tr>
  <tr><td><span class="mono">pos[]</span></td><td>each token's position (fed to rope and the KV cache)</td></tr>
  <tr><td><span class="mono">n_seq_id[]</span> / <span class="mono">seq_id[][]</span></td><td>which sequence(s) each token belongs to</td></tr>
  <tr><td><span class="mono">logits[]</span></td><td>flag: whether this token computes output logits</td></tr>
</table>
<pre class="code"><span class="cm">// simplified from include/llama.h</span>
<span class="kw">struct</span> llama_batch {
    int32_t        n_tokens;
    llama_token *  token;     <span class="cm">// word id</span>
    llama_pos   *  pos;       <span class="cm">// each token's position</span>
    int32_t     *  n_seq_id;  llama_seq_id ** seq_id;  <span class="cm">// which sequence</span>
    int8_t      *  logits;    <span class="cm">// flag: output logits?(source comment: rename to "output")</span>
};</pre>
<p>Note these are several <strong>parallel arrays</strong>: <span class="mono">token[i]</span>, <span class="mono">pos[i]</span>, <span class="mono">seq_id[i]</span>, <span class="mono">logits[i]</span> together describe all of the i-th token's info. This "struct-of-arrays" layout makes stuffing many tokens at once simple - just fill these arrays by the same index and the engine knows which tokens this step processes and how to treat each.</p>
<p>The most common constructor is <span class="mono">llama_batch_get_one</span>: it wraps a run of token ids into a minimal batch (positions auto-increment from 0, single sequence, only the last emits logits), suited to the very common case of "feed a prompt or one new token". When you need finer control (multi-sequence, custom positions), use <span class="mono">llama_batch_init</span> to fill those arrays manually.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Why several parallel arrays rather than one array of "token objects"? It is a performance consideration. Putting all tokens' same attribute (say all pos) contiguously is CPU-cache-friendly and convenient for passing a whole batch to the backend at once and vectorizing column-wise. This "<strong>struct-of-arrays</strong>" (SoA) layout is common in high-performance numeric code, of a piece with L05's tensors being "one contiguous block".
</div>
<p>There is also an <span class="mono">embd</span> field (omitted in the simplification above): most of the time you feed token <strong>word ids</strong> (via <span class="mono">token</span>), but some scenarios (e.g. multimodal, or externally pre-computed embeddings) want to feed <strong>embedding vectors</strong> directly, via <span class="mono">embd</span>. The two are either-or: give ids and let the model look up embeddings, or give embeddings directly. Plain text generation uses token.</p>
<p>Understand the batch's role too: it is a <strong>pure-data input container</strong>, with no methods and no computation. It only "describes what to compute this step", with all the real work in <span class="mono">llama_decode</span>. Splitting "describe what to compute" from "actually compute" into two things (batch and decode) is a clean interface design - you fill a form, the engine works by the form.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  These arrays are "<strong>column-aligned</strong>": the i-th token's word id, position, sequence, and output flag are at index i of the four arrays respectively. The easiest mistake filling a batch is underfilling one array or misaligning indices - once misaligned, the engine puts one token's position onto another. So when filling a batch manually, keep these arrays strictly equal in length and order.
</div>
<p>Concretely: you send a sentence in a chat box, the upper layer first tokenizes it into a run of token ids (L20), fills those ids into a batch's <span class="mono">token</span> array, positions into <span class="mono">pos</span>, all under one <span class="mono">seq_id</span>, and finally flags <span class="mono">logits</span> only on the last. Hand this batch to <span class="mono">llama_decode</span> and the model starts computing "what follows your sentence". Every everyday conversation is, behind the scenes, such batches flowing.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This "feed one form in" interface has a hidden benefit too: it <strong>decouples</strong> "preparing data" from "running the model". Preparing a batch can be done slowly on the CPU, by an upper framework (even another language), while <span class="mono">llama_decode</span> only recognizes this form and does not care how it came to be. Precisely because the interface is so simple and clear, all kinds of language bindings and upper services can easily plug into llama.cpp.
</div>

<h2>Output flag: not every token emits logits</h2>
<p>That <span class="mono">logits</span> array is a key point of this lesson. It is a <strong>switch array</strong>: <span class="mono">logits[i]=1</span> means "I want the i-th token's output logits", <span class="mono">=0</span> means "compute it, but I do not need its logits".</p>
<div class="cellgroup">
  <div class="cg-cap"><b>only flagged positions emit logits</b>: prefilling a whole prompt, often only the last is flagged</div>
  <div class="cells"><span class="lab">token</span><span class="cell">t0 (0)</span><span class="cell">t1 (0)</span><span class="cell">t2 (0)</span><span class="cell hl">t3 (1)</span><span class="lab">only t3 emits logits</span></div>
</div>
<p>Why this switch? Because computing logits is not cheap - it is a big "hidden vector -&gt; vocab size (<span class="mono">n_vocab</span>)" matmul. And many tokens' logits we never use: when prefill passes a whole prompt through, those middle tokens are just there to fill K/V into the KV cache, and <strong>only the last one's</strong> logits are used to predict the next word.</p>
<p>So this flag directly saves compute: however many positions are flagged, that many output projections are done, while other tokens stop at the hidden vector without that big matmul. In decode each step adds one token and only it needs logits; prefill needs only the final position. This "<strong>compute output on demand</strong>" is the source of L17's "logits only on some tokens".</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  In implementation, the engine counts how many output positions the batch flags, records it as <span class="mono">n_outputs</span>, then prepares output buffer for only that many rows and does only that many output projections. L17's <span class="mono">buffer_view&lt;float&gt; logits</span> has exactly <span class="mono">n_outputs</span> rows, each <span class="mono">n_vocab</span> numbers. So "how many are flagged" directly decides "how big the output buffer and how many projections".
</div>
<p>By the way, that source comment <span class="mono">rename this to "output"</span>: the name <span class="mono">logits</span> is actually a bit narrow - this flag controls "<strong>do we want this position's output</strong>", and output can be logits (generation tasks) or embedding vectors (embedding tasks, L17's embd). "output" is more accurate. Knowing this, the field name logits will not box you in by its literal meaning.</p>
<p>In multi-sequence scenarios this flag shows more flexibility: prefilling three different prompts at once, you can stuff all their tokens into one batch and flag 1 only on <strong>each sequence's last</strong> token. One decode, and all three sequences' next-word logits are obtained. This "multiple sequences in one batch, each taking its own output" is the basis for a server batch-processing multiple requests.</p>
<div class="card warn">
  <div class="tag">⚠ Performance trap</div>
  What if you flagged output on <strong>every</strong> token? The engine would dutifully do a vocab-size projection at every position - a huge waste when prefilling a long prompt. So unless you truly need every position's output (some special tasks), flag only the few you want. Whether this tiny <span class="mono">int8</span> array is filled right directly bears on how fast prefill is.
</div>
<p>By the way, the <span class="mono">logits</span> flag together with <span class="mono">seq_id</span> can express very fine needs: say a batch has three sequences, you can want output from only two and have the third just fill KV without output. This "<strong>per-token precise control</strong>" is the precondition for efficiently splicing multiple different requests to compute together - it is exactly this granularity that lets a server advance many conversations in one decode.</p>
<p>Tie the output flag to VRAM too: the larger <span class="mono">n_outputs</span>, the more the output buffer (n_vocab floats per row) takes. For a large model with a vocab of hundreds of thousands, flagging a few more output rows eats notably more buffer. So "flag only what should be flagged" saves not only compute but also that output-buffer memory. Another example of frugality embodied in one tiny flag.</p>

<h2>Splitting into ubatch: from logical batch to physical batch</h2>
<p>The batch you submit (logically "this step computes this many tokens") may not fit into the hardware in one go. The engine uses <span class="mono">llama_batch_allocr</span> to <strong>validate, fill in, then split it into physically-computable small batches</strong> (ubatch), fed one by one into the compute graph.</p>
<div class="flow">
  <div class="node"><div class="nt">llama_batch</div><div class="nd">your logical batch</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">batch_allocr.init</div><div class="nd">validate pos/seq_id<br>fill defaults</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">split_simple(n_ubatch)</div><div class="nd">split by physical batch size</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">each ubatch</div><div class="nd">build graph(L16)+execute(L10)</div></div>
</div>
<pre class="code"><span class="cm"># pseudocode: batch split into ubatch (inside llama_decode)</span>
alloc = <span class="fn">llama_batch_allocr</span>()
alloc.<span class="fn">init</span>(batch)                       <span class="cm"># validate pos/seq_id, fill defaults</span>
<span class="kw">for</span> ub <span class="kw">in</span> alloc.<span class="fn">split_simple</span>(n_ubatch):  <span class="cm"># split into ubatch by physical batch size</span>
    <span class="fn">decode_ubatch</span>(ub)                   <span class="cm"># build graph(L16)+execute(L10)</span></pre>
<p>Distinguish two "batch sizes": <span class="mono">n_batch</span> is the <strong>logical</strong> batch - how many tokens you can submit at most at once; <span class="mono">n_ubatch</span> is the <strong>physical</strong> batch - how many tokens the hardware efficiently processes at once. The former lets you "hand over more work at once", the latter is hardware-limited, and the allocr splits a big logical batch into several physical batches computed one by one.</p>
<p>Beyond the simplest <span class="mono">split_simple</span>, the allocr has <span class="mono">split_equal</span>, <span class="mono">split_seq</span>, and more, for multi-sequence and other complex arrangements. But the core idea is the same: <strong>translate "what you want to compute" into "what the hardware can swallow bite by bite"</strong>. This split lets the upper layer just describe intent, not worry about how much hardware eats at once.</p>
<p>A bit more on <span class="mono">init</span>'s "validate, fill in". It checks whether the pos and seq_id you filled are legal (e.g. positions cannot be disordered, sequence ids cannot overflow), and fills sensible defaults for fields you left out (e.g. leaving pos empty auto-numbers in order). This gatekeeping spares callers pitfalls - many "wrong batch" errors are caught here on the spot rather than computing on with the error.</p>
<table class="t">
  <tr><th>Split</th><th>How it arranges</th><th>Used for</th></tr>
  <tr><td><span class="mono">split_simple</span></td><td>cut in order</td><td>plain single sequence (most common)</td></tr>
  <tr><td><span class="mono">split_equal</span></td><td>with multiple sequences, distribute each evenly across ubatches</td><td>multi-sequence parallel</td></tr>
  <tr><td><span class="mono">split_seq</span></td><td>group one sequence's tokens together</td><td>recurrent-style models (L19 variant)</td></tr>
</table>
<p>Why so fine-grained? Because models like recurrent (an L19 variant) require "same-sequence tokens be contiguous", and different splits accommodate different model constraints. Plain models use <span class="mono">split_simple</span>.</p>
<p>This "logical vs physical batch" distinction is actually computing's common "<strong>submit</strong> and <strong>execute</strong> decoupling": you submit a big batch for convenience, the system executes in batches at its own pace. Database bulk inserts and GPU kernel launches are similar patterns. llama.cpp applies it to tokens: you submit at the granularity of a sentence or a prompt, the engine executes at the granularity hardware can eat.</p>
<p>Each ubatch triggers a full "build graph + execute": <span class="mono">llama_decode</span> calls <span class="mono">build_graph</span> (L16) per ubatch to assemble the graph for that batch of tokens and hands it to the backend (L10). So "one big batch" may internally become "several graphs run in turn". Understand this and you see why batch splitting happens inside decode, with no worry on your part - it is the automatic step connecting "your order" and "actual computation".</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  In practice, how to set <span class="mono">n_batch</span> and <span class="mono">n_ubatch</span>? Generally set <span class="mono">n_batch</span> larger (so you can submit a long prompt at once), and tune <span class="mono">n_ubatch</span> by VRAM and backend - larger computes more at once, faster, but more VRAM-hungry. Both are L17's cparams members, fixed at context creation. For most people, defaults are fine; only when squeezing throughput or tight on VRAM do you fine-tune the two.
</div>
<p>Put the batch back into the whole pipeline: <strong>you</strong> (or an upper framework) prepare a batch -&gt; <span class="mono">llama_decode</span> eats the batch, splits ubatch, builds and executes -&gt; logits come out -&gt; sampling (L21) picks a word -&gt; the new word is wrapped into the next batch... The batch is "each round's input" in this loop. Read it and you hold the "order format" for conversing with the engine.</p>
<p>A common rhythm: at the start, prefill stuffs a whole prompt's tens-to-hundreds of tokens into one <strong>big batch</strong> (efficiently filling the KV cache at once); then decode feeds a <strong>small batch</strong> of one new token each step. The same <span class="mono">llama_batch</span> structure, sometimes holding many, sometimes one - its elasticity fits exactly the fast/slow two-phase rhythm of prefill/decode (L03).</p>
<p>A foreshadow: to squeeze throughput, servers play a "<strong>continuous batching</strong>" - splicing <strong>multiple users' requests at different progress</strong> into one batch by seq_id to compute together, swapping out whoever finishes and swapping in new requests. That advanced scheduling (mentioned in Part 5) works precisely because the batch supports "multiple independent sequences in one batch". The plain batch structure you are learning now upholds quite complex serving capability.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The word "batch" itself hints at the idea: rather than feeding and computing one token at a time (paying the fixed overhead anew each time), <strong>gather a batch to compute together</strong> and amortize the fixed overhead. The reason prefill is much faster than decode is precisely that it gathers a whole prompt into one big batch computed in parallel; decode, constrained by "must wait for the previous word" (L04), can only go one by one, missing the batch dividend.
</div>
<p>Another practical detail: in the decode loop, that small batch "holding just one new token" each step is often <strong>refilled into the same</strong> batch memory - no need to reallocate every step. Together with L16's "reusable graph structure", consecutive decode is quite lightweight: the same graph, the same batch shell, each step swapping only that one token id and position. This is one of the engineering details behind word-by-word generation being so fast.</p>
<p>Unpack the word ubatch: the u means "micro", so <span class="mono">ubatch</span> is "micro-batch". Its relation to <span class="mono">batch</span> is like "the whole order you placed" versus "the small portions the kitchen cooks pot by pot". The name itself hints at its role - it is the product of a batch split fine at the physical-execution level, the smallest unit actually sent to hardware to compute at once.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What are pos and seq_id for? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">pos</span> is each token's <strong>position</strong>. It has two destinations: one, fed to rope (L16) so attention knows how far apart two tokens are; two, written into the KV cache cell (L19), marking "which position this K/V is". So a wrong pos messes up both position encoding and the cache.</p>
    <p><span class="mono">seq_id</span> marks which <strong>sequence</strong> each token belongs to. One batch (and one context) can hold <strong>several</strong> different sequences at once - say, generating answers for three different prompts simultaneously. They share these weights and this scheduling but each has its own KV (distinguished by seq_id, L19), none mixing flavors.</p>
    <p>It is exactly <span class="mono">pos</span> + <span class="mono">seq_id</span> that let a batch precisely express "which token, at which position of which sequence". With this, multi-sequence parallelism and continuing within one sequence are both expressible through one unified batch interface.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why the ubatch split layer? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because the number of tokens hardware can efficiently process at once is <strong>limited</strong> (by VRAM and compute-unit scale), and that limit is <span class="mono">n_ubatch</span>. If you submit many tokens at once (a big logical batch), without splitting it might not fit, or fit but inefficiently.</p>
    <p>So the allocr splits a big logical batch into several physical batches no larger than <span class="mono">n_ubatch</span>, computing each and stitching results. To you, how many tokens to submit (<span class="mono">n_batch</span>) is "how much work I want to hand over"; how many hardware computes at once (<span class="mono">n_ubatch</span>) is "how much the machine eats in one bite" - the two decoupled, not clashing.</p>
    <p>This split also brings flexibility: the same prompt, on a high-VRAM machine, can use a large <span class="mono">n_ubatch</span> to compute more at once, faster; with low VRAM, a small <span class="mono">n_ubatch</span> splits more times, slower but runnable. Separating "logical intent" from "physical execution" is exactly the confidence behind "the same code fitting different hardware".</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How does the output flag save compute? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The key is that "computing logits" is an <strong>expensive</strong> operation: projecting the final hidden vector to vocab size (tens of thousands of dims) is a big matmul. If every token did this, prefilling a long prompt would do hundreds or thousands of useless big projections.</p>
    <p>And the positions where we truly need logits are few: in decode each step adds one token and only it predicts the next; prefill of a whole segment needs only the last. The <span class="mono">logits</span> flag makes the engine do the output projection <strong>only at flagged positions</strong>, stopping other tokens at the hidden vector, saving a lot of big matmuls.</p>
    <p>This corresponds exactly to L03's prefill/decode rhythm: prefill is "pass a whole prompt once, take only the last position's logits", decode is "generate word by word, take the new word's logits each step". Both rhythms express "who outputs this step" precisely at the batch level via this flag array. An <span class="mono">int8</span> array, saving real compute.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama_batch</span> holds many tokens via <strong>parallel arrays</strong>: <span class="mono">token</span>/<span class="mono">pos</span>/<span class="mono">seq_id</span>/<span class="mono">logits</span>, the i-th column describing the i-th token.</li>
    <li><span class="mono">logits</span> is the <strong>output flag</strong> (source comment will rename to output): output projection only at flagged positions, saving many big matmuls.</li>
    <li><span class="mono">pos</span> feeds rope/KV (position), <span class="mono">seq_id</span> marks the sequence - supporting <strong>multi-sequence parallelism</strong> (shared weights, separate KV).</li>
    <li><span class="mono">llama_batch_allocr</span> <span class="mono">init</span>s the logical batch then <span class="mono">split_*</span>s it into physical <span class="mono">ubatch</span>es (&lt;= <span class="mono">n_ubatch</span>) fed to the graph one by one.</li>
    <li><span class="mono">n_batch</span> (logical: how much submitted at once) vs <span class="mono">n_ubatch</span> (physical: how much actually computed at once), the two decoupled.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Cleanly separating "<strong>what to feed</strong>" (batch: which tokens, which sequence, who outputs) from "<strong>how to compute in chunks</strong>" (ubatch split) - so the same <span class="mono">llama_decode</span> can run a single conversation's word-by-word decode, multi-sequence parallelism, or prefill of a whole prompt,
  all via one unified batch interface describing intent. A good interface is just this: one structure expressing as many scenarios as possible, decoupling "what to compute" from "how to compute". Next lesson, we look at where these tokens' K/V are remembered - the KV cache.
</div>
""",
}

LESSON_19 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
自回归每步只新算一个 token，全靠把先前 token 的 K/V <strong>缓存</strong>起来——这就是 KV cache（L03/L04 反复提到的那个"记忆"）。这一课钻进 <span class="mono">llama-kv-cache</span>：cell 怎么管理、上下文满了怎么<strong>移位</strong>、多条序列怎么共存，
以及为长上下文准备的各种<strong>变体</strong>。它是第四部分的收尾，也是大模型"记得住前文"的物理基础。
</p>
<p style="color:var(--muted);margin-top:.4rem">为什么 KV cache 值得收尾一讲？因为它是<strong>显存大户</strong>，也是长上下文、多并发这些实际能力的关键。前面 L17 说 context 持有它、L18 说 batch 往里写——这一课把这个"记忆"本身彻底拆开看清楚。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  KV cache 像一本<strong>会议纪要</strong>：每来一个发言（token），就把它的要点（K/V）记在新的一行（cell），标上这是第几句（<span class="mono">pos</span>）、谁说的（<span class="mono">seq_id</span>）。下次接话，只需<strong>读纪要</strong>，
  不必把整场会重听一遍。会议太长记不下了，就滚动翻页（滑窗）或划掉某人的发言（<span class="mono">seq_rm</span>）。这本纪要，就是模型"记得住上文"的那本账。
</div>

<h2>为什么需要 KV cache</h2>
<p>先说清它解决什么。注意力要让"当前 token"去看"之前所有 token"，需要每个历史 token 的 K（键）和 V（值）。<strong>没有缓存</strong>，每生成一个新词，都得把前面所有 token 的 K/V 重算一遍——长度每涨一点，重算量就平方级地涨。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>重算 vs 缓存</b>：缓存把"每步重算整段"降成"每步只算新 token"</div>
  <div class="cells"><span class="lab">没缓存</span><span class="cell">每步重算 t0..tn 的 K/V</span><span class="lab">随长度平方涨</span></div>
  <div class="cells"><span class="lab">有缓存</span><span class="cell hl">只算新 token 的 K/V</span><span class="cell">读历史 K/V</span><span class="lab">随长度线性</span></div>
</div>
<p><strong>有了缓存</strong>，先前每个 token 的 K/V 算过一次就存着，每步只需算<strong>新 token</strong> 的 K/V、把它追加进缓存，再读出全部历史 K/V 做注意力。于是每步的计算量从"重算整段"降成"只算一个"，这正是 L04 证明过的：缓存和重算<strong>数值上完全等价</strong>，但快了一个数量级。</p>
<p>这也直接解释了 L03 说的"decode 为什么快"：decode 每步只新增一个 token，靠 KV cache 复用历史，所以一步只做一个 token 的前向。可以说，<strong>没有 KV cache，就没有实用的自回归生成</strong>——它是把"理论上能算"变成"实际跑得动"的那块关键拼图。</p>
<p>先快速回忆 K/V 是什么（L04/L11）：注意力里，当前 token 用自己的 Q（查询）去和<strong>每个历史 token 的 K（键）</strong>做点积、算出"该关注谁"，再按这个权重把<strong>各历史 token 的 V（值）</strong>加权汇总。所以"看历史"这件事，需要的正是每个历史 token 的 K 和 V——把它们缓存下来，就不用每步重算。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  那为什么单单缓存 K/V、不缓存别的？因为历史 token 的 K/V <strong>算出来就不再变</strong>：第 3 个 token 的 K/V，不管后面又来了多少新词，它都是那个值，理应只算一次、存着复用。而当前 token 的 Q 是每步新算的（针对"此刻要预测谁"），没必要缓存。这种"<strong>不变的就缓存、每步变的就现算</strong>"的划分，是 KV cache 高效的根本。
</div>
<p>prefill 阶段（L03）正是<strong>一次性把整段 prompt 的 K/V 填进缓存</strong>：把 prompt 的几十上百个 token 一批喂进去（L18），并行算出它们各自的 K/V、全部写进 cell。填完，缓存里就有了整段 prompt 的记忆，之后 decode 逐字生成时，每个新词都能直接读到这份历史，不必回头重算 prompt。</p>
<p>顺带把"K/V"这俩字母的来历也点一下：它们来自数据库式的"键-值"（key-value）类比——K 像索引（拿 Q 去和它匹配、找相关的位置），V 像被取出的内容（按匹配权重汇总）。这个类比不必抠太死，但它能帮你记住：缓存 K/V，就是缓存"每个历史位置的可被检索的内容"。</p>
<p>把"平方 vs 线性"的差距再具体感受一下：生成第 1000 个 token 时，没缓存的话要把前 999 个的 K/V 全重算一遍，越往后每步越慢、整体是平方级；有缓存则第 1000 步和第 10 步一样，都只算一个新 token，整体线性。对动辄几千 token 的长对话，这个差距就是"跑得动"和"卡死"的分界。</p>

<h2>cell 怎么管：pos 与 seq_id</h2>
<p>KV cache 内部是一格格的 <strong>cell</strong>，每个 cell 存一个位置的 K/V。管理这些 cell 的是 <span class="mono">llama_kv_cells</span>：它记着每个 cell 的位置 <span class="mono">pos</span> 和所属的序列 <span class="mono">seq_id</span>；缓存还有个滚动写指针 <span class="mono">head</span>，标记下一个该往哪写。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>cells：每格记 pos + seq_id</b>，head 是滚动写指针</div>
  <div class="cells"><span class="lab">cell</span><span class="cell">pos0/seqA</span><span class="cell">pos1/seqA</span><span class="cell">pos2/seqB</span><span class="cell hl">head -&gt; 空</span></div>
</div>
<pre class="code"><span class="cm">// 简化自 src/llama-kv-cells.h / src/llama-kv-cache.h</span>
<span class="kw">class</span> llama_kv_cells {            <span class="cm">// 管理一格格 cell</span>
    std::vector&lt;llama_pos&gt; pos;    <span class="cm">// 每个 cell 的位置</span>
    <span class="cm">// 每个 cell 还记: 属于哪些 seq_id</span>
};
<span class="kw">class</span> llama_kv_cache : llama_memory_i {  <span class="cm">// src/llama-kv-cache.h</span>
    llama_kv_cells_vec v_cells;   <span class="cm">// 实际存储(可多序列)</span>
    <span class="cm">// head(): 滚动写指针</span>
};</pre>
<p>注意 <span class="mono">llama_kv_cache</span> 派生自 <span class="mono">llama_memory_i</span>——这个基接口很重要，后面讲变体时会回到它。每个 cell 带 <span class="mono">pos</span>，是因为注意力的 rope（L16）和因果掩码都要知道"这个 K/V 是第几位的"；每个 cell 带 <span class="mono">seq_id</span>，是为了支持多序列（下面讲）。</p>
<p>每步 decode，<span class="mono">build_attn</span>（L16）算出新 token 的 K/V 后，就写进 <span class="mono">head</span> 指的那个 cell、把 head 往后挪一格；做注意力时，再从所有属于本序列的 cell 里读出历史 K/V。所以 KV cache 不是被动的存储，而是<strong>每步都在增长、每步都被读取</strong>的活动记忆。把连续几步画出来，"只新算一格"就一目了然：</p>
<div class="trace">
  <div class="tcap"><b>追踪 KV cache 增长</b>：每生成一个 token 只新算一格 K/V，前面的原样复用、不重算（K0、K1… 表示各位置的 K/V）。</div>
  <div class="stations">
    <div class="stn"><h5>① 第 n 步</h5>
      <div class="cellrow"><span class="vc">K0</span><span class="vc">K1</span><span class="vc">K2</span></div>
      <div class="tlab">已有 3 格</div></div>
    <div class="op">+1 token</div>
    <div class="stn"><h5>② 第 n+1 步</h5>
      <div class="cellrow"><span class="vc dim">K0</span><span class="vc dim">K1</span><span class="vc dim">K2</span><span class="vc hot">K3</span></div>
      <div class="tlab">只新算 K3，前 3 格复用</div></div>
    <div class="op">+1 token</div>
    <div class="stn"><h5>③ 第 n+2 步</h5>
      <div class="cellrow"><span class="vc dim">K0</span><span class="vc dim">K1</span><span class="vc dim">K2</span><span class="vc dim">K3</span><span class="vc hot">K4</span></div>
      <div class="tlab">只新算 K4，前 4 格复用</div></div>
  </div>
</div>
<p>cell 和 token 是<strong>一一对应</strong>的：序列里第 i 个 token，就占缓存里某个 cell，存着它（其实是每一层都有一份）的 K 和 V。所以"缓存有多大"约等于"能记多少个 token 的 K/V"，这也是 <span class="mono">n_ctx</span>（上下文长度，L17）的含义——它就是缓存能容纳的 cell 数上限。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  <span class="mono">head</span> 这个滚动写指针还有个作用：当某些 cell 被释放（比如某条序列结束、被 <span class="mono">seq_rm</span> 删掉），它们就空出来了，新 token 可以<strong>复用</strong>这些空位，而不必一味往后涨。所以缓存不是只进不出的，而是像一块可回收的"记忆田"：用过的格子腾出来，又能种新的。这让多序列来来去去时，缓存能被反复利用。
</div>
<p>做注意力时"只读本序列的 cell"也值得说清：缓存里可能混着好几条序列的 cell，但当前 token 只该看<strong>自己这条序列</strong>、且<strong>位置在自己之前</strong>的 K/V。前者靠 <span class="mono">seq_id</span> 过滤、后者靠因果掩码（L11 的 <span class="mono">soft_max_ext</span>）。两道过滤一叠加，就保证了"各序列互不串味、每个 token 只看得到过去"。</p>
<p>再强调一句"每一层都有一份"：一个 transformer 有几十层，每一层都有自己的注意力、各自要缓存一套 K/V。所以一个 token 的"记忆"其实是<strong>几十份</strong>（每层一份）K/V 的集合。这也是为什么深一点的模型 KV cache 特别大——层数直接乘进了缓存大小里（深挖 1 的乘式里那个"层数"就是它）。</p>

<h2>序列操作与上下文移位</h2>
<p>KV cache 不只是"往里写"，还能被<strong>编辑</strong>。一组序列操作让你删、复制、保留、平移某条序列的 K/V，对应公开 C API 的 <span class="mono">llama_memory_seq_*</span>（经 <span class="mono">llama_get_memory</span> 拿到记忆对象）。</p>
<div class="flow">
  <div class="node"><div class="nt">seq_rm</div><div class="nd">删一段 KV</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">seq_cp / seq_keep</div><div class="nd">复制 / 只留某序列</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">seq_add</div><div class="nd">上下文移位<br>pos 平移</div></div>
</div>
<pre class="code"><span class="cm"># 伪代码: 序列操作(公开 C API)</span>
mem = <span class="fn">llama_get_memory</span>(ctx)
<span class="fn">llama_memory_seq_rm</span> (mem, seq, p0, p1)      <span class="cm"># 删 [p0,p1) 这段 KV</span>
<span class="fn">llama_memory_seq_add</span>(mem, seq, p0, p1, d)   <span class="cm"># 上下文移位: 把 pos 平移 d</span></pre>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  这套公开 API 的名字是 <span class="mono">llama_memory_seq_*</span>——旧版本叫 <span class="mono">llama_kv_self_seq_*</span>，已经改名移除了，看老教程时容易踩这个坑。改名本身也透露了演进方向：从"KV cache"这个具体概念，抽象成了更一般的"memory"。
</div>
<p>这里最值得理解的是<strong>上下文移位</strong>（context shift）。当对话长到超过 <span class="mono">n_ctx</span>，缓存满了怎么办？一种办法就是：丢掉最旧的一段 K/V，把剩下那些 token 的 <span class="mono">pos</span> 整体往前挪（<span class="mono">seq_add</span> 一个负的位移），腾出尾部空间继续生成。这样既不用"满了就停"，也不必重算整段——只是把记忆的窗口往前滑了一下。</p>
<p><span class="mono">seq_cp</span>（复制序列）有个巧妙用途：<strong>共享前缀</strong>。比如一个很长的 system prompt，要同时生成好几个不同回答，可以先把它算一遍、存进序列 0，再 <span class="mono">seq_cp</span> 复制给序列 1、2、3……于是几条序列共享同一段前缀的 KV，不用各算一遍。这是服务器省算力的常用招数。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  <span class="mono">seq_keep</span>（只保留某序列）则常用于<strong>清场</strong>：服务器处理完一批请求，想只留下某一条、把其余的 KV 全清掉，一句 <span class="mono">seq_keep</span> 就行。这些序列操作合起来，让 KV cache 不是一块只能整体清空的死内存，而是可以<strong>按序列精细增删</strong>的活内存——这正是高并发服务调度的底层支撑。
</div>
<p>再回到那次改名：从 <span class="mono">llama_kv_self_*</span> 到 <span class="mono">llama_memory_seq_*</span>，不只是换个名字，而是概念的<strong>抽象升级</strong>。早期只有一种"KV cache"，所以 API 就叫 kv；后来出现了 recurrent 这种"不是 KV、但也是记忆"的东西，于是把名字提升到更一般的 memory。一次改名，记录的是这个引擎从"只支持 transformer"到"也能容纳别的架构"的成长。</p>
<p>那几个序列操作里常见的 <span class="mono">p0</span>、<span class="mono">p1</span> 是<strong>位置范围</strong>：很多操作不是对整条序列、而是对"第 p0 到 p1 位"这一段做。比如只删一段、只移一段。这种"按位置区间操作"的精细度，让引擎能做很多花活——比如只回滚最近几个 token（撤销）、只移动中间一段。把记忆做成可按区间编辑的，灵活性就出来了。</p>

<h2>多序列与变体</h2>
<p>因为每个 cell 都带 <span class="mono">seq_id</span>，<strong>一个 KV cache 能同时装多条序列</strong>：它们的 cell 混在同一块缓存里、靠 seq_id 区分，各做各的注意力（只看自己序列的 cell）。这就是 L18 多序列、服务器多并发的内存基础。</p>
<table class="t">
  <tr><th>变体</th><th>思路</th><th>文件</th></tr>
  <tr><td>标准</td><td>全注意力，每个 token 都缓存</td><td><span class="mono">llama-kv-cache</span></td></tr>
  <tr><td>iSWA 滑窗</td><td>只保留最近一窗的 K/V</td><td><span class="mono">llama-kv-cache-iswa</span></td></tr>
  <tr><td>recurrent</td><td>固定大小状态，不随长度涨</td><td><span class="mono">llama-memory-recurrent</span></td></tr>
  <tr><td>hybrid</td><td>混合上面几种</td><td><span class="mono">llama-memory-hybrid</span></td></tr>
</table>
<p>为什么需要这么多变体？因为标准全注意力的 KV cache 虽然把计算降成了线性，<strong>内存却仍随长度线性增长</strong>——上下文越长越占显存。滑窗（iSWA）只留最近一窗、recurrent 用固定状态、hybrid 混搭，都是为<strong>长上下文</strong>省内存的不同取舍。它们都实现同一个基接口 <span class="mono">llama_memory_i</span>，于是可以整体替换、引擎其余部分不变。</p>
<p>多序列"不串味"再强调一遍，因为它是并发的关键：三条序列的 cell 虽然挤在同一块缓存里，但每条序列做注意力时，<span class="mono">seq_id</span> 过滤让它<strong>只看见自己的 cell</strong>，仿佛缓存里只有它一条。于是一块物理缓存，逻辑上被切成了互不可见的多份。这种"<strong>物理共享、逻辑隔离</strong>"，和 L17 的 context 共享权重是同一种省内存哲学。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  稍微展开滑窗（iSWA）：它的想法是"<strong>太远的历史就别记了</strong>"——只保留最近一个固定大小窗口内的 K/V，更旧的丢掉。这对很多任务够用（近处的上下文最重要），却能把 KV cache 的内存从"随长度涨"压成"恒定一个窗口"。有些模型是<strong>逐层混合</strong>的：一部分层用滑窗、一部分用全注意力，兼顾省内存和长程记忆。
</div>
<p>recurrent 变体则更彻底：它对应的是 Mamba 这类<strong>非 transformer</strong> 的架构，用一个<strong>固定大小的状态</strong>来概括"到目前为止的全部历史"，状态大小<strong>完全不随上下文长度变</strong>。这从根本上绕开了 KV cache 随长度涨的问题，代价是状态是"压缩过的历史"、不像全注意力那样能精确回看每个 token。把它也纳入 <span class="mono">llama_memory_i</span>，正是这套抽象的威力——连"记忆的根本机制都不同"的架构，都能接进同一个引擎。</p>
<p>把 KV cache 放回整条推理链：L18 的 batch 把新 token 喂进来 -&gt; L16 的 <span class="mono">build_attn</span> 算出它的 K/V、写进 KV cache，又从 KV cache 读出历史 -&gt; 算完更新缓存、推进一步。KV cache 就是那块被<strong>每一步反复读写</strong>的记忆，是自回归循环里"承上启下"的状态核心。前面所有部件，最后都围着它转。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  最后给第四部分（上）画个句号。回头看这六课，其实是顺着"一个模型怎么活起来"走了一遍：L14 把文件加载成张量、L15 认出它是什么架构、L16 把它拼成计算图、L17 用 context 把它装进运行时、L18 用 batch 喂它该算什么、L19 用 KV cache 让它记住前文。六块拼在一起，就是一台能<strong>持续逐字生成</strong>的推理机的内部全貌。下半部分（M4b：分词、采样、聊天模板、语法、LoRA）会继续往"文本进、文本出"的两端展开。
</div>
<p>还有一点值得知道：KV cache 的内存布局，和 L11 提过的 <strong>flash attention</strong> 这类优化是配合的——把 K/V 在内存里排得规整，注意力内核才能高效地一块块读、边读边算。所以 KV cache 不只是"存得下"就行，它<strong>怎么排</strong>也直接影响注意力算得快不快。存储布局和计算内核，在这一层是互相迁就的一对。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> KV cache 为什么这么吃显存？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为它要存的东西很多：<strong>每个 token、每一层、每个 KV 头</strong>，都要存一份 K 和一份 V。把这些乘起来——上下文长度 × 层数 × KV 头数 × 每头维度 × 2（K 和 V）——就是 KV cache 的大小。上下文一长，这个乘积就很可观，常常比模型权重之外最大的那块内存还大。</p>
    <p>所以有两条省显存的路（L17 cparams 里见过）：一是把 KV 量化存（<span class="mono">type_k</span>/<span class="mono">type_v</span> 从 16 位降到 8 位甚至更低，直接减半再减半）；二是减小 <span class="mono">n_ctx</span>（少缓存几个 token）。L15 还提过 GQA——让 KV 头数远少于 Q 头数，从源头上就把 KV cache 缩小了。</p>
    <p>理解了"KV cache 大小 = 长度 × 层 × KV头 × ..."这个乘式，你就能从一个模型的超参，估出开多长上下文会吃多少显存——这是部署大模型时一笔最该会算的账。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 上下文移位到底是怎么回事？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>场景是这样：你和模型聊了很久，token 数眼看要超过 <span class="mono">n_ctx</span>（缓存装不下了）。最朴素的做法是停下来，但那体验很差。上下文移位提供了另一条路：<strong>丢掉最旧的一段对话</strong>（删掉那些 cell），把剩下保留部分的 <span class="mono">pos</span> 整体往前移，让位置重新从小开始排，尾部就空出了新位置。</p>
    <p>关键是这只动<strong>位置标记</strong>、不重算 K/V 本身——<span class="mono">seq_add</span> 把一段 token 的 pos 平移一个量，缓存里的 K/V 内容不变，只是它们"对应的位置"变了。配合 rope 的相对位置性质，移位后模型还能正常往下接。所以它是一种"用很小代价续命"的手段。</p>
    <p>当然，丢掉最旧的对话意味着模型会"忘记"开头说过的话——这是滑动窗口式记忆的固有代价。要不要移位、丢多少，是在"无限对话"和"记住全部"之间的权衡。顺带一提，更激进的整理（defrag）在新版里已被简化移除，<span class="mono">defrag_thold</span> 参数也标了弃用。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 为什么要把"记忆"抽象成 llama_memory_i？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为"怎么记住前文"其实有很多种策略，而引擎的其余部分（建图、执行、采样）<strong>不该关心</strong>用的是哪种。把它们共同的行为（写入新 token、读出历史、删/移序列）抽象成一个基接口 <span class="mono">llama_memory_i</span>，标准 KV cache、滑窗、recurrent、hybrid 都去实现它。</p>
    <p>于是"换一种记忆策略"就变成"换一个实现 <span class="mono">llama_memory_i</span> 的类"，<span class="mono">llama_context</span> 持有的那个 <span class="mono">memory</span>（L17）指向哪个实现，引擎照常调同一套接口。这正是 L17 说"字段名叫 memory 而非 kv_cache"的原因——它留好了容纳各种记忆策略的余地。</p>
    <p>这又是一处熟悉的解耦：和 L12 的 <span class="mono">type_traits</span>（用接口容纳几十种量化）、L16 的 <span class="mono">build_arch_graph</span>（用虚函数容纳几十种架构）一脉相承。把"会变的策略"收进一个统一接口，把"不变的主干"留在外面——这是贯穿整个 llama.cpp 的设计母题，到 KV cache 这里又见到一次。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>KV cache 缓存历史 K/V，让自回归每步<strong>只算新 token</strong>（线性而非平方）——decode 快的根本（L03/L04）。</li>
    <li>cell 存 <span class="mono">pos</span>+<span class="mono">seq_id</span>（由 <span class="mono">llama_kv_cells</span> 管），<span class="mono">head</span> 是滚动写指针；<span class="mono">llama_kv_cache</span> 派生自 <span class="mono">llama_memory_i</span>。</li>
    <li>序列操作 <span class="mono">seq_rm</span>/<span class="mono">seq_cp</span>/<span class="mono">seq_keep</span>/<span class="mono">seq_add</span>（移位）；公开 API 是 <span class="mono">llama_memory_seq_*</span>（旧名 <span class="mono">llama_kv_self_*</span> 已移除）。</li>
    <li><strong>上下文移位</strong>：丢旧段、平移 <span class="mono">pos</span>，腾空间继续生成，不重算。</li>
    <li>每 cell 带 seq_id =&gt; <strong>多序列共享</strong>一块缓存；变体 iswa/recurrent/hybrid 为长上下文省内存，均实现 <span class="mono">llama_memory_i</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"算过的别再算"做成一块带 cell 管理的缓存——自回归从"每步重算整段"降到"每步增量更新"，这是大模型能逐字流式输出的根本。而把这块记忆抽象成 <span class="mono">llama_memory_i</span> 接口，又让"<strong>怎么记忆</strong>"（全注意力/滑窗/recurrent/hybrid）能整体替换、引擎其余不变——
  正是 L12、L16 那条"统一接口容纳多种策略"的母题，在"记忆"层的又一次回响。第四部分（上）到此结束：从加载（L14）、架构（L15）、建图（L16）、上下文（L17）、批处理（L18）到 KV cache（L19），你已经看清了一个 GGUF 模型如何变成一台能持续生成的推理机。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Autoregression computes only one new token per step, all thanks to <strong>caching</strong> prior tokens' K/V - that is the KV cache (the "memory" L03/L04 kept mentioning). This lesson digs into <span class="mono">llama-kv-cache</span>: how cells are managed, how to <strong>shift</strong> when the context fills,
how multiple sequences coexist, and the various <strong>variants</strong> for long context. It closes Part 4 and is the physical basis for a large model "remembering the earlier text".
</p>
<p style="color:var(--muted);margin-top:.4rem">Why close with the KV cache? Because it is a <strong>VRAM heavyweight</strong> and the key to real capabilities like long context and concurrency. L17 said the context holds it, L18 said the batch writes into it - this lesson takes that "memory" itself fully apart.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  The KV cache is like <strong>meeting minutes</strong>: each speaker (token) gets their points (K/V) written on a new line (cell), tagged with which utterance (<span class="mono">pos</span>) and who spoke (<span class="mono">seq_id</span>). To continue the conversation, just <strong>read the minutes</strong>,
  no need to re-hear the whole meeting. When the meeting runs too long to fit, scroll the pages (sliding window) or strike out someone's remarks (<span class="mono">seq_rm</span>). These minutes are the ledger by which the model "remembers the earlier text".
</div>

<h2>Why a KV cache is needed</h2>
<p>First, what it solves. Attention has "the current token" look at "all earlier tokens", needing each historical token's K (key) and V (value). <strong>Without a cache</strong>, generating each new word recomputes all prior tokens' K/V - and as length grows a bit, the recompute grows quadratically.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>recompute vs cache</b>: caching turns "recompute the whole segment each step" into "compute only the new token each step"</div>
  <div class="cells"><span class="lab">no cache</span><span class="cell">recompute t0..tn K/V each step</span><span class="lab">grows quadratically</span></div>
  <div class="cells"><span class="lab">with cache</span><span class="cell hl">compute only the new token's K/V</span><span class="cell">read historical K/V</span><span class="lab">grows linearly</span></div>
</div>
<p><strong>With a cache</strong>, each prior token's K/V is computed once and stored, and each step only computes the <strong>new token's</strong> K/V, appends it to the cache, and reads all historical K/V for attention. So per-step compute drops from "recompute the whole segment" to "compute one" - exactly what L04 proved: caching and recompute are <strong>numerically identical</strong>, but an order of magnitude faster.</p>
<p>This also directly explains L03's "why decode is fast": decode adds only one token per step, reusing history via the KV cache, so a step does just one token's forward. In short, <strong>without the KV cache there is no practical autoregressive generation</strong> - it is the key piece turning "computable in theory" into "actually runnable".</p>
<p>A quick recap of what K/V are (L04/L11): in attention, the current token uses its own Q (query) to dot-product with <strong>each historical token's K (key)</strong>, computing "who to attend to", then weight-sums <strong>each historical token's V (value)</strong> by those weights. So "looking at history" needs exactly each historical token's K and V - cache them and you avoid recomputing each step.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Why cache only K/V and nothing else? Because a historical token's K/V <strong>do not change once computed</strong>: the 3rd token's K/V, no matter how many new words follow, stay that value, deserving to be computed once and reused. The current token's Q is recomputed each step (for "who to predict right now"), no need to cache. This "<strong>cache the unchanging, compute the per-step-changing fresh</strong>" division is the root of the KV cache's efficiency.
</div>
<p>The prefill phase (L03) is exactly <strong>filling the cache with the whole prompt's K/V at once</strong>: feed the prompt's tens-to-hundreds of tokens as a batch (L18), compute their K/V in parallel, and write them all into cells. Once filled, the cache holds the whole prompt's memory, and later when decode generates word by word, each new word reads this history directly without recomputing the prompt.</p>
<p>A note on where the letters "K/V" come from: a database-style "key-value" analogy - K is like an index (Q matches against it to find relevant positions), V like the retrieved content (summed by match weights). Do not press the analogy too hard, but it helps you remember: caching K/V is caching "each historical position's retrievable content".</p>
<p>Feel the "quadratic vs linear" gap concretely: generating the 1000th token, without a cache you would recompute all prior 999 tokens' K/V, each step slower than the last, quadratic overall; with a cache, step 1000 is like step 10, both computing just one new token, linear overall. For long conversations of thousands of tokens, this gap is the line between "runnable" and "frozen".</p>

<h2>How cells are managed: pos and seq_id</h2>
<p>Inside, the KV cache is a grid of <strong>cells</strong>, each storing one position's K/V. Managing them is <span class="mono">llama_kv_cells</span>: it records each cell's position <span class="mono">pos</span> and the sequence <span class="mono">seq_id</span> it belongs to; the cache also has a rolling write pointer <span class="mono">head</span>, marking where to write next.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>cells: each records pos + seq_id</b>, head is the rolling write pointer</div>
  <div class="cells"><span class="lab">cell</span><span class="cell">pos0/seqA</span><span class="cell">pos1/seqA</span><span class="cell">pos2/seqB</span><span class="cell hl">head -&gt; empty</span></div>
</div>
<pre class="code"><span class="cm">// simplified from src/llama-kv-cells.h / src/llama-kv-cache.h</span>
<span class="kw">class</span> llama_kv_cells {            <span class="cm">// manages the grid of cells</span>
    std::vector&lt;llama_pos&gt; pos;    <span class="cm">// each cell's position</span>
    <span class="cm">// each cell also records: which seq_ids it belongs to</span>
};
<span class="kw">class</span> llama_kv_cache : llama_memory_i {  <span class="cm">// src/llama-kv-cache.h</span>
    llama_kv_cells_vec v_cells;   <span class="cm">// actual storage(multi-sequence capable)</span>
    <span class="cm">// head(): rolling write pointer</span>
};</pre>
<p>Note <span class="mono">llama_kv_cache</span> derives from <span class="mono">llama_memory_i</span> - this base interface matters, and we return to it for the variants. Each cell carries <span class="mono">pos</span> because attention's rope (L16) and causal mask both need to know "which position this K/V is"; each cell carries <span class="mono">seq_id</span> to support multiple sequences (below).</p>
<p>Each decode step, after <span class="mono">build_attn</span> (L16) computes the new token's K/V, it writes them into the cell <span class="mono">head</span> points to and advances head by one; for attention, it reads historical K/V from all cells belonging to this sequence. So the KV cache is not passive storage but an active memory that <strong>grows every step and is read every step</strong>. Draw a few consecutive steps and "only one new cell" becomes obvious:</p>
<div class="trace">
  <div class="tcap"><b>Tracing KV-cache growth</b>: each token computes just one new K/V cell; earlier cells are reused as-is, not recomputed (K0, K1... are each position's K/V).</div>
  <div class="stations">
    <div class="stn"><h5>(1) step n</h5>
      <div class="cellrow"><span class="vc">K0</span><span class="vc">K1</span><span class="vc">K2</span></div>
      <div class="tlab">3 cells so far</div></div>
    <div class="op">+1 token</div>
    <div class="stn"><h5>(2) step n+1</h5>
      <div class="cellrow"><span class="vc dim">K0</span><span class="vc dim">K1</span><span class="vc dim">K2</span><span class="vc hot">K3</span></div>
      <div class="tlab">only K3 is new, first 3 reused</div></div>
    <div class="op">+1 token</div>
    <div class="stn"><h5>(3) step n+2</h5>
      <div class="cellrow"><span class="vc dim">K0</span><span class="vc dim">K1</span><span class="vc dim">K2</span><span class="vc dim">K3</span><span class="vc hot">K4</span></div>
      <div class="tlab">only K4 is new, first 4 reused</div></div>
  </div>
</div>
<p>Cells and tokens are <strong>one-to-one</strong>: the i-th token in a sequence occupies some cell in the cache, storing its (one per layer, actually) K and V. So "how big the cache is" roughly equals "how many tokens' K/V it can remember" - which is the meaning of <span class="mono">n_ctx</span> (context length, L17): the upper bound on cells the cache can hold.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  The <span class="mono">head</span> rolling write pointer has another role: when some cells are freed (e.g. a sequence ends and is removed by <span class="mono">seq_rm</span>), they become empty, and new tokens can <strong>reuse</strong> these slots rather than only growing forward. So the cache is not write-only but like a recyclable "memory field": used cells are freed and can be sown anew. This lets the cache be reused as sequences come and go.
</div>
<p>"Reading only this sequence's cells" during attention is worth clarifying: the cache may mix several sequences' cells, but the current token should see only <strong>its own sequence</strong>'s K/V at <strong>positions before itself</strong>. The former is filtered by <span class="mono">seq_id</span>, the latter by the causal mask (L11's <span class="mono">soft_max_ext</span>). The two filters together ensure "sequences do not mix flavors, and each token sees only the past".</p>
<p>Emphasize "one per layer": a transformer has dozens of layers, each with its own attention, each caching its own set of K/V. So a token's "memory" is actually a collection of <strong>dozens</strong> of K/V (one per layer). This is why a deeper model's KV cache is especially big - layer count multiplies straight into the cache size (the "layer count" in Dig-deeper 1's product is exactly this).</p>

<h2>Sequence operations and context shift</h2>
<p>The KV cache is not just "write into" - it can be <strong>edited</strong>. A set of sequence operations lets you remove, copy, keep, or shift a sequence's K/V, corresponding to the public C API <span class="mono">llama_memory_seq_*</span> (obtaining the memory object via <span class="mono">llama_get_memory</span>).</p>
<div class="flow">
  <div class="node"><div class="nt">seq_rm</div><div class="nd">remove a span of KV</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">seq_cp / seq_keep</div><div class="nd">copy / keep only one sequence</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">seq_add</div><div class="nd">context shift<br>pos shifted</div></div>
</div>
<pre class="code"><span class="cm"># pseudocode: sequence operations (public C API)</span>
mem = <span class="fn">llama_get_memory</span>(ctx)
<span class="fn">llama_memory_seq_rm</span> (mem, seq, p0, p1)      <span class="cm"># remove the [p0,p1) span of KV</span>
<span class="fn">llama_memory_seq_add</span>(mem, seq, p0, p1, d)   <span class="cm"># context shift: shift pos by d</span></pre>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  The public API names are <span class="mono">llama_memory_seq_*</span> - older versions called them <span class="mono">llama_kv_self_seq_*</span>, now renamed and removed, so old tutorials trip on this. The rename also reveals the direction: from the concrete "KV cache" concept to a more general "memory".
</div>
<p>The most worthwhile thing here is <strong>context shift</strong>. When a conversation grows past <span class="mono">n_ctx</span> and the cache fills, what then? One way: drop the oldest span of K/V, shift the remaining tokens' <span class="mono">pos</span> forward as a whole (a negative <span class="mono">seq_add</span>), and free tail space to keep generating. This avoids "stop when full" without recomputing the whole segment - it just slides the memory window forward a bit.</p>
<p><span class="mono">seq_cp</span> (copy a sequence) has a clever use: <strong>shared prefix</strong>. Say a long system prompt must generate several different answers at once - compute it once into sequence 0, then <span class="mono">seq_cp</span> it to sequences 1, 2, 3... so several sequences share the same prefix's KV without each recomputing it. A common server compute-saving trick.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  <span class="mono">seq_keep</span> (keep only one sequence) is often used to <strong>clear the field</strong>: a server, done with a batch of requests, wanting to keep only one and clear all others' KV, does it with one <span class="mono">seq_keep</span>. These sequence operations together make the KV cache not a dead memory clearable only wholesale, but a live memory that can be <strong>finely added to and removed by sequence</strong> - the underlying support for high-concurrency service scheduling.
</div>
<p>Back to that rename: from <span class="mono">llama_kv_self_*</span> to <span class="mono">llama_memory_seq_*</span> is not just a name change but a conceptual <strong>abstraction upgrade</strong>. Early on there was only one "KV cache", so the API was called kv; later came recurrent, a "not-KV but still memory" thing, so the name was lifted to the more general memory. One rename records this engine's growth from "supporting only transformers" to "also accommodating other architectures".</p>
<p>The <span class="mono">p0</span>, <span class="mono">p1</span> common in those sequence operations are a <strong>position range</strong>: many operations act not on a whole sequence but on the "positions p0 to p1" span - removing only a span, shifting only a span. This "operate by position interval" granularity lets the engine do many tricks - rolling back only the last few tokens (undo), shifting only a middle span. Making memory editable by interval is where the flexibility comes from.</p>

<h2>Multiple sequences and variants</h2>
<p>Because each cell carries <span class="mono">seq_id</span>, <strong>one KV cache can hold multiple sequences at once</strong>: their cells mix in the same cache, distinguished by seq_id, each doing its own attention (seeing only its own sequence's cells). This is the memory basis for L18's multi-sequence and a server's concurrency.</p>
<table class="t">
  <tr><th>variant</th><th>idea</th><th>file</th></tr>
  <tr><td>standard</td><td>full attention, cache every token</td><td><span class="mono">llama-kv-cache</span></td></tr>
  <tr><td>iSWA sliding window</td><td>keep only the most recent window's K/V</td><td><span class="mono">llama-kv-cache-iswa</span></td></tr>
  <tr><td>recurrent</td><td>fixed-size state, not growing with length</td><td><span class="mono">llama-memory-recurrent</span></td></tr>
  <tr><td>hybrid</td><td>a mix of the above</td><td><span class="mono">llama-memory-hybrid</span></td></tr>
</table>
<p>Why so many variants? Because while standard full-attention's KV cache makes compute linear, its <strong>memory still grows linearly with length</strong> - the longer the context, the more VRAM. Sliding window (iSWA) keeps only a recent window, recurrent uses fixed state, hybrid mixes - all different trade-offs to save memory for <strong>long context</strong>. They all implement the same base interface <span class="mono">llama_memory_i</span>, so they can be swapped wholesale with the rest of the engine unchanged.</p>
<p>Emphasize multi-sequence "no mixing" once more, as it is the key to concurrency: three sequences' cells crowd into the same cache, but when each sequence does attention, the <span class="mono">seq_id</span> filter lets it <strong>see only its own cells</strong>, as if the cache held only it. So one physical cache is logically split into mutually-invisible portions. This "<strong>physically shared, logically isolated</strong>" is the same memory-saving philosophy as L17's context sharing weights.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Expand on the sliding window (iSWA) a bit: its idea is "<strong>do not remember history too far back</strong>" - keep only the K/V within a recent fixed-size window, dropping older ones. This suffices for many tasks (nearby context matters most) yet compresses the KV cache memory from "growing with length" to "a constant window". Some models are <strong>layer-mixed</strong>: some layers use a sliding window, some full attention, balancing memory savings and long-range memory.
</div>
<p>The recurrent variant goes further: it corresponds to <strong>non-transformer</strong> architectures like Mamba, using a <strong>fixed-size state</strong> to summarize "all history so far", the state size <strong>not changing with context length at all</strong>. This fundamentally sidesteps the KV cache's growth-with-length problem, at the cost of the state being "compressed history" - not able to look back precisely at each token as full attention can. Folding it too into <span class="mono">llama_memory_i</span> is exactly this abstraction's power - even architectures whose "fundamental memory mechanism differs" can plug into the same engine.</p>
<p>Put the KV cache back into the whole inference chain: L18's batch feeds a new token in -&gt; L16's <span class="mono">build_attn</span> computes its K/V, writes them into the KV cache, and reads history back out -&gt; after computing, updates the cache and advances one step. The KV cache is that memory <strong>read and written every step</strong>, the state core that "links past and future" in the autoregressive loop. All the earlier components, in the end, revolve around it.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Finally, a full stop for Part 4a. Looking back at these six lessons, they walk through "how a model comes alive": L14 loads the file into tensors, L15 recognizes which architecture it is, L16 assembles it into a compute graph, L17 packs it into a runtime via the context, L18 feeds it what to compute via the batch, L19 lets it remember the earlier text via the KV cache. The six together are the full internal picture of an inference machine that <strong>keeps generating word by word</strong>. The second half (M4b: tokenizer, sampling, chat template, grammar, LoRA) continues out toward the "text in, text out" ends.
</div>
<p>One more thing worth knowing: the KV cache's memory layout works together with optimizations like <strong>flash attention</strong> (mentioned in L11) - laying K/V tidily in memory is what lets the attention kernel read block by block and compute as it reads efficiently. So the KV cache is not just about "fitting"; <strong>how it is laid out</strong> also directly affects how fast attention computes. Storage layout and compute kernel are a mutually-accommodating pair at this layer.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does the KV cache eat so much VRAM? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because it stores a lot: <strong>every token, every layer, every KV head</strong> needs one K and one V. Multiply these - context length x layer count x KV head count x per-head dim x 2 (K and V) - and that is the KV cache size. As context grows long, this product is considerable, often the biggest memory block aside from the model weights.</p>
    <p>So there are two routes to save VRAM (seen in L17's cparams): one, store KV quantized (<span class="mono">type_k</span>/<span class="mono">type_v</span> from 16-bit to 8-bit or lower, halving and halving again); two, reduce <span class="mono">n_ctx</span> (cache fewer tokens). L15 also mentioned GQA - making KV heads far fewer than Q heads, shrinking the KV cache at the source.</p>
    <p>Understand "KV cache size = length x layers x KV heads x ..." and you can estimate from a model's hyperparameters how much VRAM a given context length eats - the most worth-knowing account when deploying a large model.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What exactly is context shift? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The scenario: you have chatted with the model for a while, and the token count is about to exceed <span class="mono">n_ctx</span> (the cache cannot hold more). The naive move is to stop, but that is a poor experience. Context shift offers another way: <strong>drop the oldest span of conversation</strong> (remove those cells), shift the remaining part's <span class="mono">pos</span> forward as a whole so positions restart from small, freeing new positions at the tail.</p>
    <p>The key is this only moves <strong>position tags</strong>, not recomputing the K/V themselves - <span class="mono">seq_add</span> shifts a span of tokens' pos by an amount, the cached K/V content unchanged, only their "corresponding positions" changing. With rope's relative-position nature, the model continues normally after the shift. So it is a "buy more life at small cost" technique.</p>
    <p>Of course, dropping the oldest conversation means the model "forgets" what was said at the start - the inherent cost of sliding-window memory. Whether to shift and how much to drop is a trade-off between "endless conversation" and "remember everything". By the way, more aggressive compaction (defrag) has been simplified away in newer versions, and the <span class="mono">defrag_thold</span> parameter is marked deprecated.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Why abstract "memory" into llama_memory_i? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because "how to remember the earlier text" actually has many strategies, and the rest of the engine (graph-building, execution, sampling) <strong>should not care</strong> which is used. Abstracting their common behavior (write a new token, read history, remove/shift a sequence) into a base interface <span class="mono">llama_memory_i</span>, the standard KV cache, sliding window, recurrent, and hybrid all implement it.</p>
    <p>So "switch memory strategies" becomes "switch the class implementing <span class="mono">llama_memory_i</span>", and whichever implementation the <span class="mono">memory</span> held by <span class="mono">llama_context</span> (L17) points to, the engine calls the same interface as usual. This is exactly why L17 said "the field is named memory not kv_cache" - it left room to hold various memory strategies.</p>
    <p>This is another familiar decoupling: of a piece with L12's <span class="mono">type_traits</span> (one interface holding dozens of quantizations) and L16's <span class="mono">build_arch_graph</span> (a virtual function holding dozens of architectures). Folding "the varying strategy" into a unified interface and keeping "the invariant trunk" outside - a design motif running through all of llama.cpp, seen once more at the KV cache.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>The KV cache caches historical K/V so autoregression <strong>computes only the new token</strong> each step (linear, not quadratic) - the root of decode's speed (L03/L04).</li>
    <li>Cells store <span class="mono">pos</span>+<span class="mono">seq_id</span> (managed by <span class="mono">llama_kv_cells</span>), <span class="mono">head</span> is the rolling write pointer; <span class="mono">llama_kv_cache</span> derives from <span class="mono">llama_memory_i</span>.</li>
    <li>Sequence ops <span class="mono">seq_rm</span>/<span class="mono">seq_cp</span>/<span class="mono">seq_keep</span>/<span class="mono">seq_add</span> (shift); the public API is <span class="mono">llama_memory_seq_*</span> (the old <span class="mono">llama_kv_self_*</span> is removed).</li>
    <li><strong>Context shift</strong>: drop the old span, shift <span class="mono">pos</span>, free space to keep generating, no recompute.</li>
    <li>Each cell carrying seq_id =&gt; <strong>multiple sequences share</strong> one cache; variants iswa/recurrent/hybrid save memory for long context, all implementing <span class="mono">llama_memory_i</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Making "do not recompute what was computed" into a cell-managed cache - autoregression drops from "recompute the whole segment each step" to "incremental update each step", the root of a large model's word-by-word streaming. And abstracting this memory into the <span class="mono">llama_memory_i</span> interface lets "<strong>how to remember</strong>" (full attention / sliding window / recurrent / hybrid) be swapped wholesale with the rest of the engine unchanged -
  exactly L12's and L16's "one interface holding many strategies" motif, echoing once more at the memory layer. Part 4a ends here: from loading (L14), architecture (L15), graph-building (L16), context (L17), batching (L18), to the KV cache (L19), you have seen how a GGUF model becomes a machine that keeps generating.
</div>
""",
}

LESSON_20 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
到这里，M4a 的模型已经能"算"了——加载（L14）、定架构（L15）、建图（L16）、装进上下文（L17）、按批处理（L18）、用 KV cache 高效自回归（L19），最后吐出一排 logits。可有个根本问题一直被绕过去：模型从头到尾只认<strong>数字</strong>（token id），它根本不认识"你好"这两个字。文本怎么变成数字、数字又怎么变回文本？这一课的主角 <span class="mono">llama_vocab</span>（词表），就是文本世界和 token 世界之间唯一的翻译官。
</p>
<p style="color:var(--muted);margin-top:.4rem">它干两件互逆的事：<span class="mono">tokenize</span> 把字符串切成一串 token id 喂进模型；<span class="mono">detokenize</span>/<span class="mono">token_to_piece</span> 把模型吐出的 token id 还原成文字片段、拼回人能读的句子。一次完整的对话生成，进口要过它（把你的话变成 id），出口也要过它（把模型选出的 id 变成字）。没有这层翻译，"模型只会算数字"和"人类只说文字"这两个世界就永远接不上。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  词表像一本<strong>双向密码本</strong>：编码时，把人话按一套固定规则切成一串号码（token id）；解码时，再按号码查回对应的文字片段。不同模型用的"切法"不一样（SentencePiece、BPE、WordPiece……），但本质都是同一本"号码 &lt;-&gt; 文本片段"的对照表——查得过去，也查得回来。
</div>

<h2>为什么需要词表</h2>
<div class="flow">
  <div class="node"><div class="nt">文本</div><div class="nd">"你好"</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">tokenize</div><div class="nd">编码</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">token id</div><div class="nd">[9707, ...]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">token_to_piece</div><div class="nd">解码</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">文本</div><div class="nd">"好"</div></div>
</div>
<p>模型内部全是数字。它的输入是一串 token id、输出也是一个 token id 的概率分布；从头到尾，它<strong>碰不到、也不需要</strong>字符串。词表就横在文本世界和 token 世界的边界上，进出两头都得过它这一关——这是它必须存在的根本原因。</p>
<p>为什么不直接按"字"或"字母"喂模型？因为太细则序列太长（一个汉字若干字节、一句话上千步）、太粗则词表爆炸（穷举所有词不现实）。<strong>子词（subword）</strong>切分是个折中：常见词整块给一个 id，生僻词拆成几个常见片段。于是词表大小可控（几万），序列长度也合理。</p>
<p>这也解释了为什么"同一句话，不同模型切出的 token 数不一样"。切分规则是模型训练时就定死的、随权重一起存在 GGUF 里（L13 的自描述）；用的时候必须用<strong>同一套</strong>词表，错一套，切出来的 id 就对不上模型学过的东西，输出立刻变成乱码。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  所以词表不是可有可无的配件，而是和权重<strong>绑定</strong>的一部分。加载模型（L14）时，词表也一并从 GGUF 里读出来；它定义了这个模型的"token 空间"——后面采样（L21）选下一个词，选的也正是这张词表里的某个 id。
</div>
<p>换个角度看，词表其实是模型和人之间"约定俗成"的接口。模型在训练时反复见过的每一个片段，都对应词表里的一个 id；它学到的所有规律，都建立在"这些片段"之上。所以词表一旦定下，就等于划定了模型"认得的世界"——它能流利处理的，永远是这张词表切得出来的片段的组合。</p>
<p>也正因如此，词表的好坏会直接影响表现。一套切得好的词表，能让常用表达只占很少的 token，让模型把注意力花在"意思"而不是"拼写"上；切得糙则会把简单的词拆得七零八落，既浪费序列长度、又抬高学习难度。可以说，分词是模型训练和推理共同的起跑线。</p>
<p>不妨用一个数字建立直觉：一个英文单词平均约切成 1.3 个 token，一个汉字常占 1 到 2 个 token，而一段几百字的提示词，往往对应上千个 token。模型的"上下文长度"（L17 的 n_ctx）数的就是 token、不是字符；你能塞进多少对话，最终由分词后的 token 数决定。想明白这点，才能理解为什么"同样几屏文字，有时就超了上下文"。</p>

<h2>llama_vocab 是什么</h2>
<pre class="code"><span class="cm">// 简化自 src/llama-vocab.h</span>
<span class="kw">struct</span> <span class="fn">llama_vocab</span> {
    uint32_t <span class="fn">n_tokens</span>() <span class="kw">const</span>;                          <span class="cm">// 词表大小(L15 的 n_vocab 来源)</span>
    int32_t  <span class="fn">tokenize</span>(<span class="kw">const char</span> * text, ...) <span class="kw">const</span>;       <span class="cm">// 文本 -&gt; token id</span>
    int32_t  <span class="fn">token_to_piece</span>(llama_token id, <span class="kw">char</span> * buf, ...) <span class="kw">const</span>; <span class="cm">// id -&gt; 文本片段</span>
<span class="kw">private</span>:
    <span class="kw">struct</span> impl;                              <span class="cm">// pimpl: 藏起分词器实现</span>
    std::unique_ptr&lt;impl&gt; pimpl;
};</pre>
<p><span class="mono">llama_vocab</span> 对外是一套统一接口：<span class="mono">n_tokens()</span> 给词表大小、<span class="mono">tokenize</span> 编码、<span class="mono">token_to_piece</span> 解码，还有一堆查询单个 token 属性的方法。但它把"具体用哪种分词算法"的实现细节，全藏在一个私有的 <span class="mono">impl</span> 结构里——这就是 <strong>pimpl</strong>（pointer to implementation）手法。</p>
<p>为什么要 pimpl？因为分词算法五花八门（下一节细讲），每种的内部数据结构、合并规则都不同。把它们统统塞进 <span class="mono">impl</span>，对外只露 <span class="mono">tokenize</span>/<span class="mono">token_to_piece</span> 这层薄薄的接口，于是<strong>用的人完全无感</strong>：不管底下是 SPM 还是 BPE，调用方式一模一样。换算法、改实现，都不会惊动上层代码。</p>
<p>这和你前面见过的解耦是一个味道：L14 的 loader 把"解析格式"和"使用模型"分开，L17 的 context 把"只读知识"和"会话状态"分开。这里则是把"分词的脏活"和"统一的接口"分开。一道清晰的边界，让复杂性被关在盒子里。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  还有一组以 <span class="mono">get_add_bos()</span>/<span class="mono">get_add_eos()</span> 为代表的标志位，记录"这个模型 tokenize 时该不该自动前置 BOS、后置 EOS"。这些也是随模型定的（写在 GGUF 里），词表忠实地照做——又一次体现 L13 的自描述精神。
</div>
<p>再具体看这套接口的分工。编码这头，<span class="mono">tokenize</span> 要处理的细节其实不少：要不要加空格前缀、要不要规范化、遇到连续空白怎么合并——这些都是不同分词器各自的"脾气"，但全被收进了 <span class="mono">impl</span>。上层只管把字符串递进去、把 id 取出来，完全不必关心底下在折腾什么。</p>
<p>解码那头同样有讲究。<span class="mono">token_to_piece</span> 不是简单"查表取字符串"，它还要处理特殊 token 该不该显示、字节 token 怎么按 UTF-8 拼、首词要不要补空格这些琐碎规则。把它们也一并封进词表，是为了让"还原文本"在任何模型上都一致正确——你只管循环调用、拼接结果。</p>

<h2>几种分词算法</h2>
<table class="t">
  <tr><th>类型</th><th>算法</th><th>代表模型</th></tr>
  <tr><td>SPM</td><td>SentencePiece（字节级 BPE + 字节回退）</td><td>LLaMA</td></tr>
  <tr><td>BPE</td><td>字节级 byte-pair 合并</td><td>GPT-2 / Qwen</td></tr>
  <tr><td>WPM</td><td>WordPiece</td><td>BERT</td></tr>
  <tr><td>UGM</td><td>Unigram</td><td>T5</td></tr>
  <tr><td>RWKV</td><td>贪心匹配</td><td>RWKV</td></tr>
  <tr><td>PLAMO2</td><td>Aho-Corasick + 动态规划</td><td>PLaMo-2</td></tr>
</table>
<p>主流分词算法就那么几种，<span class="mono">enum llama_vocab_type</span> 把它们一一列出。它们的差别在"怎么把词拆成子词、怎么合并"，但对上层都是同一个 <span class="mono">tokenize</span>。GGUF 的 tokenizer 元数据（L13）决定这个模型用哪种。</p>
<p><strong>SPM</strong>（SentencePiece）是 LLaMA 系的传统，基于字节级 BPE 且自带字节回退；<strong>BPE</strong>（byte-pair encoding）是 GPT-2 系的字节级合并；<strong>WPM</strong>（WordPiece）是 BERT 系；<strong>UGM</strong>（Unigram）是 T5 系；<strong>RWKV</strong> 用贪心匹配；还有较新的 <strong>PLAMO2</strong>（Aho-Corasick + 动态规划）。这些缩写背后，是不同的历史生态和语言/效率取舍。</p>
<p>为什么有这么多？因为不同模型家族沿用各自生态的工具链，而每种算法在多语言、代码、压缩率上各有长短。llama.cpp 不强求统一，而是用一套接口（pimpl）把它们都兼容进来——这正是它能跑几十种模型的工程基础之一。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  你不需要记住每种的细节，只要建立一个直觉：<strong>同一句话，换种算法就换种切法</strong>，token 数和边界都可能不同；但只要"编码用的词表"和"模型训练用的词表"是同一套，往返就严丝合缝。
</div>
<p>举个直观的例子体会差异。"unhappiness"这个词，BPE 可能切成"un"+"happiness"或"un"+"happy"+"ness"，靠的是训练时统计出来的高频合并；Unigram 则从一个大候选集里、按概率挑出最可能的一种切分。两条路线殊途同归，都想用尽量少的片段覆盖尽量多的文本，只是挑片段的"哲学"不同。</p>
<p>对中文这种没有天然空格的语言，分词更见功夫。字节级方案会先把汉字降到 UTF-8 字节再合并，于是不依赖"词的边界"也能工作——这也是为什么一个主要用英文训练的模型，往往也能磕磕绊绊地处理中文：因为最底层它认的是字节，而不是某种语言的"词"。</p>
<p>还有一个常被问到的点：词表该做多大？太小则每个词都得拆成很多片段、序列变长、推理变慢；太大则嵌入表和输出层都跟着膨胀、显存吃紧。所以词表大小是个折中，主流模型大多落在几万到十几万这个区间。它一旦定下，就深深影响着模型的体量与速度——又一次印证"词表和模型是绑在一起的"。</p>

<h2>特殊 token 与字节回退</h2>
<p>词表里除了普通的"文字片段 token"，还有一类<strong>特殊 token</strong>：<span class="mono">BOS</span>（序列开始）、<span class="mono">EOS</span>（序列结束）、<span class="mono">EOT</span>（一轮结束）、<span class="mono">PAD</span>/<span class="mono">SEP</span>/<span class="mono">UNK</span> 等。它们不对应具体文字，而是<strong>控制标记</strong>，由访问器 <span class="mono">token_bos()</span>/<span class="mono">token_eos()</span>/<span class="mono">token_eot()</span> 取出。比如模型生成出 EOS，就意味着"我说完了"，上层据此停止。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>字节回退</b>：词表里没有的字符 -&gt; 按 UTF-8 拆成字节 -&gt; 每字节一个 &lt;0xXX&gt; token</div>
  <div class="cells"><span class="lab">生僻字/emoji</span><span class="cell hl">词表外</span><span class="lab">-&gt;</span><span class="cell">&lt;0xF0&gt;</span><span class="cell">&lt;0x9F&gt;</span><span class="cell">&lt;0x8E&gt;</span><span class="cell">&lt;0x89&gt;</span></div>
</div>
<p>那遇到词表里压根没有的字符怎么办（生僻字、emoji）？靠<strong>字节回退</strong>（byte fallback）：把这个字符按 UTF-8 拆成若干字节，每个字节映射到一个形如 <span class="mono">&lt;0xF0&gt;</span> 的字节 token（带 <span class="mono">LLAMA_TOKEN_ATTR_BYTE</span> 属性）。于是<strong>任何</strong> UTF-8 文本最差也能逐字节编码，永远不会"无法编码"。这一手解决的是经典的 <strong>OOV</strong>（未登录词）难题。一次往返大致如下：</p>
<pre class="code"><span class="cm"># 伪代码: tokenize 往返</span>
ids = vocab.<span class="fn">tokenize</span>(<span class="st">"Hello"</span>, add_special=<span class="kw">True</span>)   <span class="cm"># 可自动前置 BOS</span>
<span class="cm"># ids = [&lt;bos&gt;, 9906, ...]</span>
text = <span class="st">""</span>
<span class="kw">for</span> id <span class="kw">in</span> ids:
    text += vocab.<span class="fn">token_to_piece</span>(id)               <span class="cm"># 逐 token 还原拼接</span></pre>
<p>把上面这段 "Hello" 真正走一遍，编码就具体了：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次分词</b>：一句话怎么变成 token id；遇到词表里没有的字符（生僻字、emoji）就按 UTF-8 拆成字节（id 为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 输入串</h5>
      <div class="cellrow"><span class="vc">"Hello!"</span></div>
      <div class="tlab">一句话</div></div>
    <div class="op">分词</div>
    <div class="stn"><h5>② 切成片</h5>
      <div class="cellrow"><span class="vc">Hello</span><span class="vc">!</span></div>
      <div class="tlab">命中词表的片段</div></div>
    <div class="op">查表<br>&rarr; id</div>
    <div class="stn"><h5>③ token id</h5>
      <div class="cellrow"><span class="vc blue">9906</span><span class="vc blue">30</span></div>
      <div class="tlab">每片一个 id</div></div>
    <div class="op">词表外<br>UTF-8</div>
    <div class="stn"><h5>④ 字节回退</h5>
      <div class="cellrow"><span class="vc">&lt;0xF0&gt;</span><span class="vc">&lt;0x9F&gt;</span><span class="vc">&lt;0x8E&gt;</span><span class="vc">&lt;0x89&gt;</span></div>
      <div class="tlab">生僻 emoji = 4 个字节 token</div></div>
  </div>
</div>
<p><span class="mono">tokenize</span> 时可以让它自动前置 BOS（由前面那个 <span class="mono">get_add_bos()</span> 标志控制）；解码时则逐个 token 调 <span class="mono">token_to_piece</span> 把片段拼回去。注意字节 token 还原时要按 UTF-8 把几个字节<strong>拼起来</strong>才是一个完整字符——这也是为什么解码要逐步累积、而不是"一个 token 一个字"。</p>
<p>特殊 token 之所以重要，是因为它们承载着"文字之外"的结构信息。一段对话里，谁说的、一轮在哪结束、要不要停下，都靠这些标记界定（后面 L22 的对话模板，正是在大量使用它们）。可以把普通 token 看成"内容"、特殊 token 看成"标点和段落标记"——少了后者，模型就分不清对话的骨架。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  字节回退还有个容易被忽略的好处：它让词表可以做得相对"小"而不担心覆盖不全。既然有 256 个字节 token 兜底，词表就不必为塞下每个生僻字而无限膨胀，只收高频片段即可，罕见的交给字节去拼。这是"常见的走捷径、罕见的走通路"的务实设计，和很多系统里"快路径 + 慢路径"异曲同工。
</div>
<p>这里还要点出一个细节：判断"该不该停"靠的不是单一的 EOS，而是一组"生成结束"（EOG）标记。不同模型用的结束标记不一样，有的用 EOS、有的用 EOT、有的两者皆可；词表里用一个专门的判断（是否属于 EOG 集合）来统一处理。上层只要问一句"这个 token 是不是结束符"，就能正确收尾，而不必记住每个模型的具体约定。</p>

<h2>C API 与衔接</h2>
<p>从 C API 用词表，路径很直白：先 <span class="mono">llama_model_get_vocab(model)</span> 从模型拿到词表，再 <span class="mono">llama_vocab_n_tokens(vocab)</span> 问大小、<span class="mono">llama_tokenize</span> 编码、<span class="mono">llama_token_to_piece</span>/<span class="mono">llama_detokenize</span> 解码。这些函数都收一个 <span class="mono">const llama_vocab *</span>。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  2023 年老教程里那一批名字<strong>大多已经弃用</strong>。取词表大小的 <span class="mono">llama_n_vocab</span> 被标了 DEPRECATED，改用 <span class="mono">llama_vocab_n_tokens</span>；取特殊 token 的 <span class="mono">llama_token_bos</span>/<span class="mono">llama_token_eos</span> 等，统统改名成了 <span class="mono">llama_vocab_bos</span>/<span class="mono">llama_vocab_eos</span>。看老代码时要留个心眼。
</div>
<p>为什么要这么大动干戈地改名？因为这些操作本质上是<strong>词表的</strong>方法，而不是模型的——把它们从 <span class="mono">llama_*</span> 统一收进 <span class="mono">llama_vocab_*</span>，名实相符，也呼应了内部 <span class="mono">llama_vocab</span> 已经独立成型这件事。改名虽然烦，但让 API 更清晰。</p>
<p>把这一课接回主线：你输入的文字，先经 L22 的对话模板拼好格式，再经词表 <span class="mono">tokenize</span> 成 id，进模型算出 logits（L17），由采样器（L21）在<strong>这张词表的 token 空间</strong>里选出下一个 id，最后再经 <span class="mono">token_to_piece</span> 变回文字显示给你。词表正是这条回路一进一出的两道门。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>你的文字</h4><p>一句话或一段提示词。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>对话模板(L22) + tokenize</h4><p>先拼好格式，再切成 token id 序列。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>模型算 logits(L17)</h4><p>前向一遍，输出每个 token 一个分数。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>采样选 id(L21)</h4><p>在这张词表的 token 空间里挑下一个。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>token_to_piece -&gt; 文字</h4><p>把 id 还原成文字，显示给你。</p></div></div>
</div>
<p>顺带澄清一个常见疑惑：tokenize 的结果是不是唯一的？对确定的词表和同一套规则，答案是肯定的——同样的输入永远切出同样的 id 序列，这正是编码/解码能可靠往返的前提。采样（L21）带来的随机性，发生在"选下一个 token"那一步，和分词无关；分词本身是完全确定的。</p>
<p>最后留一个串起全局的视角：词表是这套推理系统里少数"人能直接看懂"的部分。权重是一堆浮点、计算图是一串算子，唯独词表，你能把 id 一个个查回文字、亲眼看到模型"读到了什么、想说什么"。调试模型行为时，先把 token 打印出来看看，常常是最快的入手点。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么有这么多分词算法（SPM/BPE/WPM…）？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>直接原因是<strong>历史与生态</strong>：LLaMA 系沿用 SentencePiece，GPT 系用字节级 BPE，BERT 系用 WordPiece，T5 系用 Unigram。每个模型家族训练时用什么，推理时就得用什么——词表和权重是配套的，换不得。</p>
    <p>更深一层是<strong>取舍</strong>：不同算法在多语言覆盖、对代码/数字的友好度、压缩率（同样文本切成多少 token）上各有高下。比如字节级 BPE 对任何语言都鲁棒（先降到字节），WordPiece 对英文形态友好。没有银弹，所以百花齐放。</p>
    <p>llama.cpp 的态度是<strong>全都支持</strong>：用 pimpl 把各算法的实现差异藏起来，对上层暴露同一个 <span class="mono">tokenize</span>。于是它不挑模型——这正是一个"通用推理引擎"该有的样子，和 L15 表驱动支持多架构是同一种胸怀。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 字节回退到底解决什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>解决 <strong>OOV</strong>（out-of-vocabulary，未登录词）。任何固定词表都不可能穷尽世界上所有字符（新 emoji、生僻字、各种符号层出不穷）。没有兜底机制的话，遇到没见过的字符就只能丢一个 <span class="mono">&lt;UNK&gt;</span>，信息彻底丢失。</p>
    <p>字节回退的兜底很优雅：UTF-8 本身就是字节序列，把任意字符拆成 1-4 个字节，每个字节对应一个 <span class="mono">&lt;0xXX&gt;</span> token（共 256 个，必然覆盖）。于是"词表外"这个概念被消灭了——再罕见的字符也能被无损编码，只是占的 token 多一点。</p>
    <p>代价值得一提：一个生僻字可能占 3-4 个字节 token，比常见字"贵"几倍。所以模型处理大量生僻字/某些语言时，token 消耗会明显偏高——这也是有些语言"显得更费 token"的底层原因之一。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 为什么 n_vocab 在词表里、不在 hparams（L15）？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为词表大小是<strong>词表的属性</strong>，由 tokenizer 决定，而不是网络结构的属性。L15 的 <span class="mono">llama_hparams</span> 描述"网络多少层、多宽、几个头"；词表描述"token 空间多大、怎么切分"。两者职责不同，理应分家。</p>
    <p>实践中确实有这个坑：L15 特意强调过 <span class="mono">n_vocab</span> 不在 <span class="mono">hparams</span> 里，权威来源是 <span class="mono">llama_vocab::n_tokens()</span>。虽然嵌入层和输出层的形状要用到词表大小（它们的一个维度就是 <span class="mono">n_vocab</span>），但这个数的"主人"是词表。</p>
    <p>这种"谁的属性归谁管"的划分，让代码各司其职：改词表不动 hparams、改网络结构不动词表。边界清晰，是这套代码能长期维护的隐形功臣——你在 L14（loader vs 模型）、L17（model vs context）已经反复见到同一种纪律。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">llama_vocab</span> 是文本 &lt;-&gt; token 的双向翻译官：<span class="mono">tokenize</span> 编码、<span class="mono">token_to_piece</span>/<span class="mono">detokenize</span> 解码。</li>
    <li>它是 <strong>pimpl</strong>：把具体分词算法藏进 <span class="mono">impl</span>，对外只露统一接口。</li>
    <li>类型 <span class="mono">enum llama_vocab_type</span>：SPM/BPE/WPM/UGM/RWKV/PLAMO2，差在"怎么切"，由 GGUF 元数据选定。</li>
    <li>特殊 token（<span class="mono">token_bos/eos/eot</span>）是控制标记；<strong>字节回退</strong> <span class="mono">&lt;0xXX&gt;</span> 让任何 UTF-8 都能编码、消灭 OOV。</li>
    <li>C API：<span class="mono">llama_model_get_vocab</span> -&gt; <span class="mono">llama_vocab_n_tokens</span>/<span class="mono">llama_tokenize</span>/<span class="mono">llama_token_to_piece</span>；旧名 <span class="mono">llama_n_vocab</span>/<span class="mono">llama_token_bos</span> 已弃用、改 <span class="mono">llama_vocab_*</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  词表把"<strong>文本世界</strong>"和"<strong>token 世界</strong>"干净地隔开——模型只在 token 空间里活，永远不碰字符串；几种截然不同的切分算法，被 pimpl 一藏，在上层眼里就是同一个 <span class="mono">tokenize</span>。这层薄薄的翻译官，是"模型只会算数字"和"人类只说文字"之间唯一的桥。读懂了它，你就明白每次对话一进一出，文字是怎么悄悄变成数字、又变回文字的。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
By now M4a's model can "compute" - loading (L14), architecture (L15), graph-building (L16), context (L17), batching (L18), efficient autoregression via the KV cache (L19), finally emitting a row of logits. But one basic question kept getting skipped: from start to finish the model only knows <strong>numbers</strong> (token ids); it has no idea what the characters "hi" are. How does text turn into numbers, and numbers back into text? This lesson's star, <span class="mono">llama_vocab</span> (the vocabulary), is the sole translator between the text world and the token world.
</p>
<p style="color:var(--muted);margin-top:.4rem">It does two inverse jobs: <span class="mono">tokenize</span> cuts a string into a list of token ids to feed the model; <span class="mono">detokenize</span>/<span class="mono">token_to_piece</span> turns the token ids the model emits back into text pieces, reassembled into a human-readable sentence. A full chat generation passes through it on the way in (your words become ids) and on the way out (the chosen ids become characters). Without this translation, "the model only does numbers" and "humans only speak text" never connect.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A vocabulary is like a <strong>two-way codebook</strong>: encoding cuts human speech by a fixed set of rules into a string of numbers (token ids); decoding looks each number back up to its text piece. Different models use different "cuts" (SentencePiece, BPE, WordPiece...), but at heart it is the same "number &lt;-&gt; text-piece" table - it maps forward, and it maps back.
</div>

<h2>Why a vocabulary is needed</h2>
<div class="flow">
  <div class="node"><div class="nt">text</div><div class="nd">"hi"</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">tokenize</div><div class="nd">encode</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">token id</div><div class="nd">[9707, ...]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">token_to_piece</div><div class="nd">decode</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">text</div><div class="nd">"hi"</div></div>
</div>
<p>Inside, the model is all numbers. Its input is a list of token ids and its output is a probability distribution over a token id; from end to end it <strong>never touches, and never needs</strong>, strings. The vocabulary sits exactly on the border between the text world and the token world, and both directions must pass through it - that is why it must exist.</p>
<p>Why not feed the model by "character" or "letter" directly? Too fine and the sequence is too long (one CJK glyph is several bytes, one sentence thousands of steps); too coarse and the vocab explodes (enumerating all words is hopeless). <strong>Subword</strong> splitting is the compromise: common words get one id whole, rare words split into a few common pieces. So the vocab size stays manageable (tens of thousands) and the sequence length stays reasonable.</p>
<p>This also explains why "the same sentence, split by different models, yields a different token count". The splitting rules are fixed at training time and stored with the weights in GGUF (L13's self-description); at use time you must use the <strong>same</strong> vocab - the wrong one and the ids no longer match what the model learned, and the output instantly turns to garbage.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  So the vocabulary is not an optional accessory but a part <strong>bound</strong> to the weights. When the model loads (L14), the vocab is read out of GGUF along with it; it defines this model's "token space" - and when sampling (L21) later picks the next word, it is picking some id from exactly this vocabulary.
</div>
<p>From another angle, the vocabulary is really an agreed-upon interface between the model and people. Every piece the model saw repeatedly during training maps to one id in the vocab; all the patterns it learned are built on "those pieces". So once the vocab is fixed, it delimits the model's "known world" - what it handles fluently is always combinations of pieces this vocab can produce.</p>
<p>For that reason the vocab's quality directly affects performance. A well-cut vocab lets common expressions take very few tokens, letting the model spend attention on "meaning" rather than "spelling"; a crude one shatters simple words into fragments, wasting sequence length and raising the learning difficulty. Tokenization is, in a sense, the shared starting line of both training and inference.</p>
<p>Build intuition with a number: an English word averages about 1.3 tokens, a CJK glyph often takes 1 to 2 tokens, and a prompt of a few hundred characters often maps to over a thousand tokens. The model's "context length" (L17's n_ctx) counts tokens, not characters; how much conversation you can fit is ultimately decided by the post-tokenization token count. Grasp this and you see why "the same few screens of text sometimes overflows the context".</p>

<h2>What llama_vocab is</h2>
<pre class="code"><span class="cm">// simplified from src/llama-vocab.h</span>
<span class="kw">struct</span> <span class="fn">llama_vocab</span> {
    uint32_t <span class="fn">n_tokens</span>() <span class="kw">const</span>;                          <span class="cm">// vocab size (source of L15's n_vocab)</span>
    int32_t  <span class="fn">tokenize</span>(<span class="kw">const char</span> * text, ...) <span class="kw">const</span>;       <span class="cm">// text -&gt; token id</span>
    int32_t  <span class="fn">token_to_piece</span>(llama_token id, <span class="kw">char</span> * buf, ...) <span class="kw">const</span>; <span class="cm">// id -&gt; text piece</span>
<span class="kw">private</span>:
    <span class="kw">struct</span> impl;                              <span class="cm">// pimpl: hides the tokenizer internals</span>
    std::unique_ptr&lt;impl&gt; pimpl;
};</pre>
<p>Outwardly <span class="mono">llama_vocab</span> is one unified interface: <span class="mono">n_tokens()</span> gives the vocab size, <span class="mono">tokenize</span> encodes, <span class="mono">token_to_piece</span> decodes, plus a batch of methods to query a single token's attributes. But it hides all the "which tokenizer algorithm exactly" implementation detail inside a private <span class="mono">impl</span> struct - this is the <strong>pimpl</strong> (pointer to implementation) idiom.</p>
<p>Why pimpl? Because tokenizer algorithms vary wildly (next section), each with different internal data structures and merge rules. Stuffing them all into <span class="mono">impl</span> and exposing only the thin <span class="mono">tokenize</span>/<span class="mono">token_to_piece</span> interface means <strong>the caller feels nothing</strong>: whether SPM or BPE underneath, the call is identical. Swapping algorithms or changing internals never disturbs the upper code.</p>
<p>This is the same flavor of decoupling you have seen before: L14's loader splits "parse the format" from "use the model", L17's context splits "read-only knowledge" from "session state". Here it splits "the dirty work of tokenizing" from "the unified interface". A clear boundary keeps the complexity locked in a box.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  There is also a set of flags led by <span class="mono">get_add_bos()</span>/<span class="mono">get_add_eos()</span>, recording "should this model auto-prepend BOS, auto-append EOS when tokenizing". These too are set per model (written in GGUF), and the vocab faithfully obeys - once more L13's self-description spirit.
</div>
<p>Look more concretely at this interface's division of labor. On the encoding side, <span class="mono">tokenize</span> handles quite a few details: whether to add a space prefix, whether to normalize, how to merge consecutive whitespace - each tokenizer's own "temperament", all gathered into <span class="mono">impl</span>. The upper layer just hands in a string and takes out ids, never minding what churns below.</p>
<p>The decoding side is equally subtle. <span class="mono">token_to_piece</span> is not a plain "look up a string"; it also handles whether a special token should show, how byte tokens join by UTF-8, whether the first word needs a leading space. Sealing these into the vocab too keeps "restoring text" consistently correct across any model - you just call in a loop and concatenate.</p>

<h2>A few tokenizer algorithms</h2>
<table class="t">
  <tr><th>Type</th><th>Algorithm</th><th>Example model</th></tr>
  <tr><td>SPM</td><td>SentencePiece (byte-level BPE + byte fallback)</td><td>LLaMA</td></tr>
  <tr><td>BPE</td><td>byte-level byte-pair merges</td><td>GPT-2 / Qwen</td></tr>
  <tr><td>WPM</td><td>WordPiece</td><td>BERT</td></tr>
  <tr><td>UGM</td><td>Unigram</td><td>T5</td></tr>
  <tr><td>RWKV</td><td>greedy matching</td><td>RWKV</td></tr>
  <tr><td>PLAMO2</td><td>Aho-Corasick + dynamic programming</td><td>PLaMo-2</td></tr>
</table>
<p>There are only a handful of mainstream tokenizer algorithms, and <span class="mono">enum llama_vocab_type</span> lists them out. They differ in "how to split a word into subwords and how to merge", but to the upper layer they are all the same <span class="mono">tokenize</span>. GGUF's tokenizer metadata (L13) decides which one this model uses.</p>
<p><strong>SPM</strong> (SentencePiece) is the LLaMA-family tradition, based on byte-level BPE with built-in byte fallback; <strong>BPE</strong> (byte-pair encoding) is the GPT-2-family byte-level merging; <strong>WPM</strong> (WordPiece) is the BERT family; <strong>UGM</strong> (Unigram) is the T5 family; <strong>RWKV</strong> uses greedy matching; and the newer <strong>PLAMO2</strong> (Aho-Corasick + dynamic programming). Behind these abbreviations lie different historical ecosystems and language/efficiency trade-offs.</p>
<p>Why so many? Because different model families inherit their own ecosystem's toolchain, and each algorithm has strengths and weaknesses across multilingual coverage, code, and compression rate. llama.cpp does not force uniformity; it makes them all compatible behind one interface (pimpl) - one of the engineering foundations for running dozens of models.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  You need not memorize each one's details, only build an intuition: <strong>the same sentence, a different algorithm means a different cut</strong>, with possibly different token counts and boundaries; but as long as "the vocab used to encode" and "the vocab the model trained with" are the same, the round-trip fits perfectly.
</div>
<p>A concrete example brings out the difference. The word "unhappiness", BPE might cut into "un"+"happiness" or "un"+"happy"+"ness", relying on high-frequency merges counted at training time; Unigram instead picks, from a large candidate set, the most probable single split by probability. The two routes converge - both want to cover the most text with the fewest pieces - they just differ in the "philosophy" of choosing pieces.</p>
<p>For a language like Chinese with no natural spaces, tokenization shows its craft. Byte-level schemes first drop a glyph to UTF-8 bytes and then merge, so they work without relying on "word boundaries" - which is also why a model trained mostly on English can often stumble through Chinese: at the bottom it knows bytes, not any language's "words".</p>
<p>Another frequently asked point: how big should the vocab be? Too small and every word splits into many pieces, lengthening sequences and slowing inference; too large and the embedding table and output layer bloat with it, straining VRAM. So vocab size is a compromise, and mainstream models mostly land between tens of thousands and a hundred-odd thousand. Once fixed, it deeply shapes the model's size and speed - once more proof that "the vocab and the model are bound together".</p>

<h2>Special tokens and byte fallback</h2>
<p>Beyond ordinary "text-piece tokens", the vocab has a class of <strong>special tokens</strong>: <span class="mono">BOS</span> (begin sequence), <span class="mono">EOS</span> (end sequence), <span class="mono">EOT</span> (end of turn), <span class="mono">PAD</span>/<span class="mono">SEP</span>/<span class="mono">UNK</span>, etc. They map to no specific text but are <strong>control markers</strong>, fetched by accessors <span class="mono">token_bos()</span>/<span class="mono">token_eos()</span>/<span class="mono">token_eot()</span>. For example, when the model emits EOS it means "I am done", and the upper layer stops accordingly.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Byte fallback</b>: a char not in the vocab -&gt; split into UTF-8 bytes -&gt; one &lt;0xXX&gt; token per byte</div>
  <div class="cells"><span class="lab">rare glyph/emoji</span><span class="cell hl">out of vocab</span><span class="lab">-&gt;</span><span class="cell">&lt;0xF0&gt;</span><span class="cell">&lt;0x9F&gt;</span><span class="cell">&lt;0x8E&gt;</span><span class="cell">&lt;0x89&gt;</span></div>
</div>
<p>So what about a character the vocab simply does not have (rare glyphs, emoji)? <strong>Byte fallback</strong>: split the character into its UTF-8 bytes and map each byte to a byte token shaped like <span class="mono">&lt;0xF0&gt;</span> (carrying the <span class="mono">LLAMA_TOKEN_ATTR_BYTE</span> attribute). So <strong>any</strong> UTF-8 text can, worst case, be encoded byte by byte, and is never "unencodable". This solves the classic <strong>OOV</strong> (out-of-vocabulary) problem. A round-trip looks roughly like:</p>
<pre class="code"><span class="cm"># pseudocode: a tokenize round-trip</span>
ids = vocab.<span class="fn">tokenize</span>(<span class="st">"Hello"</span>, add_special=<span class="kw">True</span>)   <span class="cm"># may auto-prepend BOS</span>
<span class="cm"># ids = [&lt;bos&gt;, 9906, ...]</span>
text = <span class="st">""</span>
<span class="kw">for</span> id <span class="kw">in</span> ids:
    text += vocab.<span class="fn">token_to_piece</span>(id)               <span class="cm"># rebuild piece by piece</span></pre>
<p>Walk that "Hello" through it for real and encoding gets concrete:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one tokenization</b>: how a sentence becomes token ids; a char not in the vocab (rare glyph, emoji) splits into UTF-8 bytes (ids illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) input</h5>
      <div class="cellrow"><span class="vc">"Hello!"</span></div>
      <div class="tlab">a sentence</div></div>
    <div class="op">tokenize</div>
    <div class="stn"><h5>(2) pieces</h5>
      <div class="cellrow"><span class="vc">Hello</span><span class="vc">!</span></div>
      <div class="tlab">in-vocab pieces</div></div>
    <div class="op">look up<br>-&gt; id</div>
    <div class="stn"><h5>(3) token id</h5>
      <div class="cellrow"><span class="vc blue">9906</span><span class="vc blue">30</span></div>
      <div class="tlab">one id per piece</div></div>
    <div class="op">OOV<br>UTF-8</div>
    <div class="stn"><h5>(4) byte fallback</h5>
      <div class="cellrow"><span class="vc">&lt;0xF0&gt;</span><span class="vc">&lt;0x9F&gt;</span><span class="vc">&lt;0x8E&gt;</span><span class="vc">&lt;0x89&gt;</span></div>
      <div class="tlab">rare emoji = 4 byte tokens</div></div>
  </div>
</div>
<p><span class="mono">tokenize</span> can auto-prepend BOS (controlled by that <span class="mono">get_add_bos()</span> flag); decoding then calls <span class="mono">token_to_piece</span> per token to stitch pieces back. Note that restoring byte tokens means <strong>joining</strong> several bytes by UTF-8 to form one complete character - which is why decoding accumulates step by step, not "one token, one character".</p>
<p>Special tokens matter because they carry structural information "beyond the text". In a conversation, who spoke, where a turn ends, whether to stop - all are delimited by these markers (L22's chat template, a later lesson, uses them heavily). Think of ordinary tokens as "content" and special tokens as "punctuation and paragraph marks" - without the latter, the model cannot tell the conversation's skeleton.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Byte fallback has an easily overlooked benefit too: it lets the vocab stay relatively "small" without fearing incomplete coverage. Since 256 byte tokens are the safety net, the vocab need not bloat endlessly to fit every rare glyph - it keeps only high-frequency pieces and leaves the rare ones to bytes. This "shortcut for the common, full path for the rare" is the same pragmatic spirit as many systems' "fast path + slow path".
</div>
<p>One more detail to call out: deciding "whether to stop" relies not on a single EOS but on a set of "end-of-generation" (EOG) markers. Different models use different end markers - some EOS, some EOT, some either; the vocab handles them uniformly with a dedicated test (whether a token belongs to the EOG set). The upper layer need only ask "is this token a terminator", finishing correctly without memorizing each model's specific convention.</p>

<h2>The C API and the hand-off</h2>
<p>Using the vocab from the C API is straightforward: first <span class="mono">llama_model_get_vocab(model)</span> to get the vocab from the model, then <span class="mono">llama_vocab_n_tokens(vocab)</span> for the size, <span class="mono">llama_tokenize</span> to encode, <span class="mono">llama_token_to_piece</span>/<span class="mono">llama_detokenize</span> to decode. These all take a <span class="mono">const llama_vocab *</span>.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  That batch of names from 2023-era tutorials is <strong>mostly deprecated</strong>. The vocab-size getter <span class="mono">llama_n_vocab</span> is marked DEPRECATED, replaced by <span class="mono">llama_vocab_n_tokens</span>; the special-token getters <span class="mono">llama_token_bos</span>/<span class="mono">llama_token_eos</span> etc. were all renamed to <span class="mono">llama_vocab_bos</span>/<span class="mono">llama_vocab_eos</span>. Watch out when reading old code.
</div>
<p>Why such a sweeping rename? Because these operations are essentially <strong>vocabulary</strong> methods, not model ones - folding them from <span class="mono">llama_*</span> into <span class="mono">llama_vocab_*</span> makes name match substance, echoing how <span class="mono">llama_vocab</span> has internally become its own thing. Renames are annoying but make the API clearer.</p>
<p>Connecting this lesson back to the main line: your input text is first formatted by L22's chat template, then <span class="mono">tokenize</span>d into ids by the vocab, enters the model to compute logits (L17), the sampler (L21) picks the next id in <strong>this vocabulary's token space</strong>, and finally <span class="mono">token_to_piece</span> turns it back into text shown to you. The vocab is exactly the two gates, in and out, of this loop.</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>your text</h4><p>A sentence or a prompt.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>chat template (L22) + tokenize</h4><p>First format it, then cut into a token id sequence.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>model computes logits (L17)</h4><p>One forward pass, one score per token.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>sampling picks an id (L21)</h4><p>Pick the next one in this vocab's token space.</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>token_to_piece -&gt; text</h4><p>Turn the id back into text, shown to you.</p></div></div>
</div>
<p>A common doubt worth clearing: is tokenize's result unique? For a fixed vocab and the same rules, yes - the same input always cuts into the same id sequence, which is exactly the premise that encoding/decoding round-trips reliably. The randomness sampling (L21) brings happens at the "pick the next token" step, unrelated to tokenization; tokenization itself is fully deterministic.</p>
<p>One last whole-picture view: the vocab is one of the few parts of this inference system "a human can read directly". Weights are a pile of floats, the compute graph a chain of operators; only the vocab lets you look ids back into text and see with your own eyes "what the model read, what it wants to say". When debugging model behavior, printing the tokens first is often the fastest way in.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why so many tokenizer algorithms (SPM/BPE/WPM...)? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The direct reason is <strong>history and ecosystem</strong>: the LLaMA family inherits SentencePiece, the GPT family uses byte-level BPE, the BERT family uses WordPiece, the T5 family uses Unigram. Whatever a model family trains with, it must infer with - vocab and weights are a matched set, not swappable.</p>
    <p>One layer deeper is <strong>trade-offs</strong>: algorithms differ in multilingual coverage, friendliness to code/numbers, and compression rate (how many tokens the same text becomes). Byte-level BPE is robust for any language (it drops to bytes first); WordPiece is friendly to English morphology. No silver bullet, hence the variety.</p>
    <p>llama.cpp's stance is <strong>support them all</strong>: hide each algorithm's implementation difference behind pimpl and expose the same <span class="mono">tokenize</span> upward. So it is not picky about models - exactly what a "general inference engine" should be, the same breadth as L15's table-driven multi-architecture support.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What does byte fallback actually solve? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>It solves <strong>OOV</strong> (out-of-vocabulary). No fixed vocab can exhaust every character in the world (new emoji, rare glyphs, all kinds of symbols keep appearing). Without a fallback, an unseen character can only yield a single <span class="mono">&lt;UNK&gt;</span>, losing the information entirely.</p>
    <p>Byte fallback's safety net is elegant: UTF-8 is itself a byte sequence, so split any character into 1-4 bytes, each mapping to a <span class="mono">&lt;0xXX&gt;</span> token (256 of them, guaranteed to cover). The concept of "out of vocab" is thus abolished - even the rarest character is encoded losslessly, just at a few more tokens.</p>
    <p>The cost is worth noting: a rare glyph may take 3-4 byte tokens, several times "pricier" than a common one. So a model processing lots of rare glyphs / certain languages spends noticeably more tokens - one underlying reason some languages "seem more token-hungry".</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Why is n_vocab in the vocab, not in hparams (L15)? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because the vocab size is a <strong>property of the vocabulary</strong>, decided by the tokenizer, not a property of the network structure. L15's <span class="mono">llama_hparams</span> describes "how many layers, how wide, how many heads"; the vocab describes "how big the token space is, how to split". Different responsibilities, rightly separated.</p>
    <p>In practice this is a real trap: L15 deliberately stressed that <span class="mono">n_vocab</span> is not in <span class="mono">hparams</span>, the authoritative source being <span class="mono">llama_vocab::n_tokens()</span>. Although the embedding and output layers' shapes use the vocab size (one of their dimensions is <span class="mono">n_vocab</span>), that number's "owner" is the vocab.</p>
    <p>This "whose property, whose responsibility" division lets the code stay clean: changing the vocab does not touch hparams, changing the network does not touch the vocab. Clear boundaries are the invisible hero of long-term maintainability - the same discipline you saw again and again in L14 (loader vs model) and L17 (model vs context).</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">llama_vocab</span> is the two-way translator between text and tokens: <span class="mono">tokenize</span> encodes, <span class="mono">token_to_piece</span>/<span class="mono">detokenize</span> decodes.</li>
    <li>It is <strong>pimpl</strong>: the concrete tokenizer algorithm is hidden in <span class="mono">impl</span>, exposing only a unified interface.</li>
    <li>Type <span class="mono">enum llama_vocab_type</span>: SPM/BPE/WPM/UGM/RWKV/PLAMO2, differing in "how to cut", selected by GGUF metadata.</li>
    <li>Special tokens (<span class="mono">token_bos/eos/eot</span>) are control markers; <strong>byte fallback</strong> <span class="mono">&lt;0xXX&gt;</span> lets any UTF-8 be encoded, abolishing OOV.</li>
    <li>C API: <span class="mono">llama_model_get_vocab</span> -&gt; <span class="mono">llama_vocab_n_tokens</span>/<span class="mono">llama_tokenize</span>/<span class="mono">llama_token_to_piece</span>; old names <span class="mono">llama_n_vocab</span>/<span class="mono">llama_token_bos</span> are deprecated, use <span class="mono">llama_vocab_*</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The vocabulary cleanly separates the "<strong>text world</strong>" from the "<strong>token world</strong>" - the model lives only in token space and never touches strings; several utterly different splitting algorithms, hidden by pimpl, look like the same <span class="mono">tokenize</span> from above. This thin translator is the only bridge between "the model only does numbers" and "humans only speak text". Understand it, and you see how, every turn in and out, text quietly becomes numbers and becomes text again.
</div>
""",
}

LESSON_21 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课词表把文字变成 token id 喂进模型，模型一路计算（M4a），最后在<strong>输出层</strong>吐出一排 <span class="mono">logits</span>——词表里每个 token 各对应一个原始分数，分数越高代表模型越"看好"它当下一个词。可下一个词只能有一个，怎么从几万个分数里挑出它？这一课讲的<strong>采样</strong>（sampling），就是"从一排分数到一个 token"的最后一步。
</p>
<p style="color:var(--muted);margin-top:.4rem">采样远不止"选最大的"那么简单。每次都选最高分，模型会死板、重复、毫无新意；可纯随机又会胡言乱语。真正的采样是一套<strong>裁剪 + 塑形 + 抽选</strong>的组合拳：先划掉没希望的候选、再调分布的软硬、压一压老重复的词，最后才按概率抽一个。llama.cpp 把这些手段做成一个个可插拔的<strong>采样器</strong>，串成一条<strong>采样链</strong>。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  采样像<strong>摇号抽奖</strong>：<span class="mono">logits</span> 是每个号码的原始权重，温度调节"凭实力还是凭运气"，top-k/top-p 先划掉没希望的号，惩罚项压低最近老出现的号，最后 <span class="mono">dist</span> 按权重摇一个出来——或者 <span class="mono">greedy</span> 干脆选权重最大的那个。同一堆号码，配不同的规则，摇出来的"性格"就完全不同。
</div>

<h2>从 logits 到 token</h2>
<div class="flow">
  <div class="node"><div class="nt">logits</div><div class="nd">每个 token 一个分数</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">penalties</div><div class="nd">压低重复</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">top_k / top_p</div><div class="nd">裁剪候选</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">temp</div><div class="nd">塑形分布</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">dist</div><div class="nd">选一个 token</div></div>
</div>
<p>先看清采样的输入和输出。输入是一个<strong>候选数组</strong>：词表里每个 token 一条记录，含 token id、它的 logit（原始分数）、以及待会儿算出来的概率 p。输出是其中<strong>一个</strong>被选中的 id。采样要做的，就是在这个数组上一通操作，最后挑出一条。</p>
<p>这个候选数组在 llama.cpp 里叫 <span class="mono">llama_token_data_array</span>，每条记录是 <span class="mono">llama_token_data</span>。整条采样管线，本质上就是<strong>不断改写这个数组</strong>：有的采样器把某些候选的 logit 砸成负无穷（等于划掉），有的重新算概率、重新排序，最后一步从中选定一个、把它的下标记在数组的 <span class="mono">selected</span> 字段上。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  为什么要分这么多步、而不是一步选完？因为"选下一个词"需求多样：写代码要严谨（偏确定），写诗要发散（偏随机），还要避免老车轱辘话来回说。把这些需求拆成一个个独立的小操作、按需组合，远比写一个巨大的"全能采样函数"灵活。
</div>
<p>于是管线大致长这样：先用惩罚压低重复，再用 top-k/top-p 砍掉长尾候选，接着用温度调节剩下分布的软硬，最后用 dist 按概率抽一个（或 greedy 直接取最大）。每一步只做一件小事，叠起来就是一套完整的采样策略。</p>
<p>要强调的是，这套管线只动 logits/概率、<strong>不碰模型本身</strong>。模型每步老老实实算出同一排 logits，至于怎么从中选词，全由采样这层说了算。所以"换个生成风格"根本不用动模型，调调采样参数即可——这也是同一个模型能时而严谨、时而天马行空的原因。</p>
<p>不妨把这排 logits 想象成一座<strong>高低起伏的山脉</strong>：模型越看好的词，峰就越高。采样要做的，就是按这座山的形状来取舍——只在高峰附近选（保守），还是连山脚的小丘也给点机会（发散）。后面的每个采样器，其实都在<strong>重塑这座山的轮廓</strong>，再决定从哪儿落子。这个画面记住了，后面的 top-k、温度就都好理解了。把这条链路用一个具体例子走一遍就清楚了：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次采样</b>：5 个候选词，看一排 logits 怎么一步步变成最终选中的一个 token（数字为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① logits</h5>
      <div class="cellrow"><span class="vc">3.2</span><span class="vc">2.1</span><span class="vc">1.0</span><span class="vc">0.5</span><span class="vc">-0.3</span></div>
      <div class="tlab">cat / dog / sky / run / blue</div></div>
    <div class="op">÷T<br>T=0.7</div>
    <div class="stn"><h5>② 温度缩放</h5>
      <div class="cellrow"><span class="vc">4.6</span><span class="vc">3.0</span><span class="vc">1.4</span><span class="vc">0.7</span><span class="vc">-.4</span></div>
      <div class="tlab">T&lt;1 放大差距</div></div>
    <div class="op">top-k<br>k=3</div>
    <div class="stn"><h5>③ 截断候选</h5>
      <div class="cellrow"><span class="vc hot">4.6</span><span class="vc hot">3.0</span><span class="vc hot">1.4</span><span class="vc dim">0.7</span><span class="vc dim">-.4</span></div>
      <div class="tlab">只留分数最高的 3 个</div></div>
    <div class="op">softmax<br>top-p .9</div>
    <div class="stn"><h5>④ 概率 → 采样</h5>
      <div class="cellrow"><span class="vc blue">.78</span><span class="vc blue">.18</span><span class="vc dim">.04</span></div>
      <div class="tlab">按概率抽一个 → <strong>cat</strong></div></div>
  </div>
</div>

<h2>采样器接口</h2>
<pre class="code"><span class="cm">// 简化自 include/llama.h</span>
<span class="kw">struct</span> <span class="fn">llama_sampler_i</span> {
    <span class="kw">const char</span> * (*name)  (...);                          <span class="cm">// 名字(可空)</span>
    <span class="kw">void</span> (*accept)(llama_sampler *, llama_token);          <span class="cm">// 喂回选中 token(可空)</span>
    <span class="kw">void</span> (*apply) (llama_sampler *, llama_token_data_array * cur_p); <span class="cm">// 改/排候选(必需)</span>
    <span class="kw">void</span> (*reset)(llama_sampler *);                        <span class="cm">// 清状态(可空)</span>
};
<span class="kw">struct</span> <span class="fn">llama_sampler</span> { <span class="kw">const</span> llama_sampler_i * iface; llama_sampler_context_t ctx; };</pre>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">必需</span><span class="name">apply(cur_p)</span></div><div class="ld">改写候选数组：划掉 / 重排 / 重算概率——每个采样器的核心</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">可空</span><span class="name">accept · reset · name · clone · free</span></div><div class="ld">accept 把选中 token 喂回有状态采样器；其余按需实现</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">状态</span><span class="name">llama_sampler_context_t ctx</span></div><div class="ld">每个采样器私有的账本：penalties 记历史、mirostat 记反馈</div></div>
</div>
<p>看看一个采样器到底是什么。<span class="mono">llama_sampler_i</span> 就是一组函数指针：<span class="mono">apply</span>（核心，改写候选数组）、<span class="mono">accept</span>（把选中的 token 喂回来给有状态采样器记账）、<span class="mono">reset</span>（清状态），还有 name/clone/free。配上一块状态 <span class="mono">ctx</span>，就构成一个 <span class="mono">llama_sampler</span>。</p>
<p>这里 <span class="mono">apply</span> 是唯一<strong>必需</strong>的——它拿到候选数组，按自己的规则改一改（划掉一些、重排一下、重算概率）。<span class="mono">accept</span> 可空，只有"有记忆"的采样器才用得上：惩罚项要记住前面出过哪些 token、mirostat 要根据反馈调参，它们都靠 accept 把"刚选中的 token"收进自己的状态。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这种"一组函数指针 + 一块状态"的设计，你应该眼熟——它就是 L10 后端、L19 记忆接口那套<strong>接口 + 实现</strong>的又一次运用。每个采样器只要实现这几个函数，就能被统一调度；引擎不在乎你内部是 top-k 还是 mirostat，只管按顺序调 apply。
</div>
<p>把采样器抽象成统一接口，最大的好处是<strong>可组合</strong>。既然它们长一个样，就能像积木一样排成一队，挨个作用在同一个候选数组上。下一节的"采样链"，就是这种可组合性的直接产物。</p>
<p>顺带说状态 <span class="mono">ctx</span>：它是每个采样器私有的小账本。无状态的采样器（如 top_k）ctx 几乎是空的；有状态的（penalties/mirostat/grammar）则把历史、参数、反馈都存在这里。采样器之间互不干扰，各记各的账。</p>
<p>为什么接口里好几个函数都标着"可空"？因为不是每个采样器都用得上每件事。像 top_k 这种纯粹"裁一刀"的，根本不需要记忆，也就不必实现 accept；而 reset 只在复用同一个采样器跑多段生成时才有意义。把这些做成可选，让最简单的采样器可以只写一个 apply，既省事又清晰——接口只要求"必需的那件事"，其余按需。</p>

<h2>采样链</h2>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>chain_init</h4><p>建一条空的采样链（<span class="mono">llama_sampler_chain</span>）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>chain_add 若干采样器</h4><p>按顺序加入 penalties、top_k、top_p、temp、dist……每个都是独立采样器。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>sample</h4><p>对候选数组按加入顺序逐个 apply，最后一个（dist/greedy）选出 token。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>accept</h4><p>把选中 token 喂回链，让有状态采样器（penalties 等）记住它。</p></div></div>
</div>
<p>采样链 <span class="mono">llama_sampler_chain</span> 本身也是一个采样器——它内部装着一串子采样器，它的 <span class="mono">apply</span> 就是<strong>按加入顺序</strong>把每个子采样器的 apply 挨个跑一遍。这是典型的"组合模式"：一条链对外看也是一个采样器，对内是许多采样器的队列。</p>
<pre class="code"><span class="cm"># 伪代码: 组一条采样链</span>
chain = <span class="fn">llama_sampler_chain_init</span>(params)
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_penalties</span>(...))    <span class="cm"># 压低重复</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_top_k</span>(40))         <span class="cm"># 留前 40</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_top_p</span>(0.95, 1))     <span class="cm"># 核采样</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_temp</span>(0.8))          <span class="cm"># 温度</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_dist</span>(seed))         <span class="cm"># 按概率随机选</span>
id = <span class="fn">llama_sampler_sample</span>(chain, ctx, -1)         <span class="cm"># 跑全链 -&gt; 返回 token id</span></pre>
<p>用起来很直观：先 <span class="mono">chain_init</span> 建空链，再 <span class="mono">chain_add</span> 按想要的顺序把采样器一个个塞进去，最后 <span class="mono">llama_sampler_sample</span> 一把梭——它读出某位置的 logits、组成候选数组、跑完整条链、返回选中的 token id，并顺手把这个 token <span class="mono">accept</span> 回去。</p>
<p><strong>顺序很重要</strong>。同样几个采样器，排列不同，结果可能不同：一般先做惩罚和裁剪（缩小候选集），再做温度（调软硬），最后才是 dist/greedy（真正选定）。把"选定"放最后，是因为前面每一步都在为这"临门一脚"准备一个更合理的候选分布。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  注意链会<strong>接管</strong>加进来的采样器的所有权——一旦 add 进去，释放链时会一并释放它们，你不用单独操心。这种"加进去就交给链管"的约定，让组装采样策略很省心：拼好一条链，用完整体释放即可。
</div>
<p>这套"链"的设计，本质上是把"采样策略"变成了<strong>数据</strong>（一串采样器配置），而不是写死的代码。于是用户在命令行/配置里调几个参数，就能拼出千变万化的采样行为，引擎主干一行都不用改——又是一次"会变的部分集中起来"的体现。</p>
<p>举个顺序影响结果的例子。假如你把温度放在 top-p <strong>之前</strong>，温度会先把分布整体烫平、再让 top-p 去圈范围，圈出来的核就偏大、偏发散；反过来先 top-p 圈定再升温，则是在一个已经收紧的小集合里调随机性，结果更可控。同样的零件、不同的次序，最终"性格"就有微妙差别——这正是把顺序交给用户配置的价值。</p>
<p>实际项目里，这条链常有个约定俗成的默认顺序。llama.cpp 的上层（<span class="mono">common</span>）大致按"惩罚 -&gt; 裁剪（top-k/top-p/min-p 等）-&gt; 温度 -&gt; dist"来排。你不必死记，但记住那个大原则就够了：<strong>先缩小候选、再调软硬、最后才抽签</strong>。绝大多数采样策略，都是在这条主轴上加加减减。</p>

<h2>常见采样器与 greedy vs dist</h2>
<table class="t">
  <tr><th>采样器</th><th>作用</th></tr>
  <tr><td>greedy</td><td>选 logit 最大的（argmax，确定）</td></tr>
  <tr><td>dist</td><td>按概率随机选（靠 seed）</td></tr>
  <tr><td>top_k</td><td>只留前 k 个候选</td></tr>
  <tr><td>top_p</td><td>核采样：留累积概率达 p 的最小集</td></tr>
  <tr><td>min_p</td><td>留概率不低于"最大值 × p"的候选</td></tr>
  <tr><td>temp</td><td>缩放 logits，调随机性</td></tr>
  <tr><td>penalties</td><td>压低重复/高频/已出现的 token</td></tr>
  <tr><td>mirostat</td><td>动态调温，稳住困惑度</td></tr>
</table>
<p>来认认常用的几个采样器。它们各管一段：有的负责"裁"（缩小候选集），有的负责"塑"（改分布形状），有的负责"选"（最终拍板）。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  先说最终拍板的两个：<span class="mono">greedy</span> 永远选 logit 最大的那个——确定性，同样输入永远同样输出，适合要复现、要严谨的场景；<span class="mono">dist</span> 则把 logits 经 softmax 变成概率，再按概率<strong>随机</strong>抽一个，带来多样性，靠随机种子 seed 控制。一条链最后接 greedy 还是 dist，决定这次生成是"确定"还是"随机"。
</div>
<p>再说"裁"的两位主力：<span class="mono">top_k</span> 留分数最高的固定 k 个、其余划掉；<span class="mono">top_p</span>（核采样）按概率从高到低累加、留到累计达 p 为止——候选数随分布自适应（分布尖时留得少、平时留得多）。两者常配合：先 top_k 砍掉长尾，再 top_p 自适应收口。</p>
<p>"塑"的代表是温度 <span class="mono">temp</span>：把 logits 除以一个温度值 T 再 softmax。T 小则分布更尖（更确定），T 大则更平（更随机）。它不改候选集，只改"软硬"。还有 <span class="mono">penalties</span> 压低重复、<span class="mono">mirostat</span> 动态调温稳住困惑度等，各有专长。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  2023 年那套全局采样函数（<span class="mono">llama_sample_top_k</span>/<span class="mono">llama_sample_top_p</span>/<span class="mono">llama_sample_temperature</span> 等）<strong>已经全部移除</strong>，统一换成了"采样器对象 + 链"这套模型。看老教程别再找那些函数了。
</div>
<p>单独说说 <span class="mono">penalties</span> 这一类，因为它最贴近日常体验。它盯着最近生成过的 token，对老重复的词施加惩罚（调低 logit），于是模型不容易陷进"复读机"式的循环。常见的有重复惩罚、频率惩罚、存在惩罚几种口味，分别对应"出现过就罚""出现越多越罚""只要出现就一视同仁地罚"。调它们，能在"连贯"和"啰嗦"之间找平衡。</p>

<h2>接回主回路与衔接</h2>
<p>把采样接回主回路：每生成一个 token，<span class="mono">llama_decode</span>（L17）算出 logits，采样链从中选一个 id，这个 id 一边经词表（L20）变回文字显示、一边被包成新 batch 喂回 <span class="mono">llama_decode</span> 进入下一步。采样就是自回归循环里"<strong>挑下一个词</strong>"那一环。</p>
<p>还有一个和 grammar（L23）的衔接要先打招呼：语法约束本质上也是一个采样器（它的 apply 把不合语法的 token 划掉）。但它通常<strong>不</strong>塞进主链，而是作为独立对象、按 <span class="mono">grammar_first</span> 决定在链前还是链后单独施加——这一课熟悉了采样器接口，L23 再看 grammar 就水到渠成。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  所以这一课真正要带走的，是一个心智模型：<strong>采样 = 在候选数组上排一队小变换，最后选一个</strong>。模型决定"每个词多大概率合适"，采样决定"这次到底挑谁"。理解了它，你就理解了为什么同一个模型、同一句提示，调调参数就能从"一本正经"变到"天马行空"。
</div>
<p>顺带点一个实用细节：要让随机生成<strong>可复现</strong>，关键在 <span class="mono">dist</span> 的那个随机种子 seed。固定 seed、固定采样参数，同一段提示就能跑出完全一样的结果——这在调试、对比实验时极有用。反过来，想要每次都不一样，让 seed 随时间变即可。确定性到底掌握在你手里。</p>
<p>最后澄清一个常见误解：采样调不出模型本来没有的能力。它只能在模型给出的那排 logits 上做文章——好的采样能让一个模型<strong>扬长避短</strong>（少出昏招、保持多样），但变不出模型压根学不会的知识。所以效果不好时，先分清是"模型不行"还是"采样没调好"：前者要换模型/微调（L24），后者调调参数即可。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 温度（temperature）到底在做什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>温度 T 的作用是缩放 logits：把每个分数除以 T，再 softmax 变概率。T=1 是原样；T 小于 1，大分数被进一步放大、小分数被压扁，分布更<strong>尖锐</strong>，模型更倾向选最可能的词（更确定、更保守）；T 大于 1 则把差距拉平，分布更<strong>平坦</strong>，冷门词也有机会（更随机、更有创意）。</p>
    <p>两个极端很有意思：T 趋近 0，分布尖到只剩最大那个，温度采样就<strong>退化成 greedy</strong>；T 很大时分布趋于均匀，几乎是瞎猜。所以温度是一个连续的"确定 &lt;-&gt; 随机"旋钮，greedy 不过是它的一个极端特例。</p>
    <p>要记住温度<strong>只改软硬、不改候选集</strong>——它不删任何 token，只重新分配大家的概率。删候选是 top-k/top-p 的活。两类操作正交，组合才好用：先用 top-p 圈定合理候选集，再用温度调这个集合内部的随机程度。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> top-k 和 top-p 有何不同？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">top_k</span> 留<strong>固定个数</strong>：把候选按分数排序，留前 k 个、其余划掉。简单直接，但有个毛病——分布尖时 k 个里混进很多没希望的，分布平时又可能把好候选挡在门外。它不看分布形状，只数个数。</p>
    <p><span class="mono">top_p</span>（核采样 nucleus）留<strong>累积概率达 p 的最小集合</strong>：按概率从高到低累加，加到超过 p 就停。它的候选数是<strong>自适应</strong>的——分布尖时可能只留两三个，分布平时可能留几十个。这种"按概率密度收口"往往比固定 k 更合理。</p>
    <p>实践中常<strong>两者叠用</strong>：先 top_k（比如 40）砍掉绝大多数长尾、控制开销，再 top_p（比如 0.95）在剩下的里自适应收口。一个管"最多留多少"，一个管"按质量留多少"，配合起来既快又稳。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 为什么做成"链"而不是一个大函数？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为<strong>可组合 + 可配置 + 状态隔离</strong>。每个采样器是独立小部件、自带状态（penalties 记历史、mirostat 记反馈、dist 记随机数），顺序可调、增删自由。用户在配置里写一串采样器名字和参数，引擎照单拼出一条链——想要什么策略就拼什么，不必改一行引擎代码。</p>
    <p>对比"一个写死的大采样函数"：那样每加一种新手段都得改主函数、各种 if 越堆越多，参数也纠缠不清。拆成链之后，新增一种采样器只是多写一个独立实现，对已有的零影响。这正是 L11 算子、L16 建图积木一脉相承的"小部件组合出复杂行为"。</p>
    <p>还有个细节：grammar（L23）这种采样器通常<strong>不进主链</strong>，而是按 <span class="mono">grammar_first</span> 在链前或链后单独施加。这说明"链"也不死板——它给特殊约束留了在合适位置插入的余地，足够灵活。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>采样 = 在候选数组 <span class="mono">llama_token_data_array</span> 上裁剪塑形、最后选一个 token。</li>
    <li>采样器 <span class="mono">llama_sampler_i</span>：<span class="mono">apply</span>（必需，改候选）+ <span class="mono">accept</span>（喂回选中 token）+ reset；配状态 <span class="mono">ctx</span> 成 <span class="mono">llama_sampler</span>。</li>
    <li>采样链：<span class="mono">chain_init</span> -&gt; <span class="mono">chain_add</span> 若干采样器 -&gt; <span class="mono">sample</span>，按<strong>顺序</strong>逐个 apply。</li>
    <li><span class="mono">greedy</span>=选最大（确定）、<span class="mono">dist</span>=按概率随机；<span class="mono">top_k</span>/<span class="mono">top_p</span> 裁候选、<span class="mono">temp</span> 调软硬、<span class="mono">penalties</span>/<span class="mono">mirostat</span> 各有专长。</li>
    <li>旧全局 <span class="mono">llama_sample_*</span> 已移除，统一为采样器对象 + 链；grammar（L23）是链外的特殊采样器。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把采样从"一个写死的大函数"拆成"一串可插拔的小变换"，是典型的<strong>责任链 / 管道</strong>设计——和你在 ggml 算子链（L09）、建图积木（L16）里见过的是同一种味道：用小而独立的部件，组合出复杂多变的行为。于是"换一种生成风格"只是换链里几个环、调几个数，模型和引擎主干纹丝不动。读懂采样，你就握住了把模型从"一本正经"调到"天马行空"的那几个旋钮。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson the vocab turned text into token ids fed to the model; the model computes all the way through (M4a) and finally, at the <strong>output layer</strong>, emits a row of <span class="mono">logits</span> - one raw score per token in the vocab, a higher score meaning the model "favors" it more as the next word. But the next word can be only one, so how do you pick it from tens of thousands of scores? This lesson's topic, <strong>sampling</strong>, is that last step "from a row of scores to one token".
</p>
<p style="color:var(--muted);margin-top:.4rem">Sampling is far more than "pick the max". Always picking the top score makes the model rigid, repetitive, dull; but pure randomness babbles. Real sampling is a combo of <strong>prune + shape + draw</strong>: first cut hopeless candidates, then tune the distribution's softness, push down words that keep repeating, and only then draw one by probability. llama.cpp makes these means into pluggable <strong>samplers</strong>, strung into a <strong>sampler chain</strong>.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Sampling is like a <strong>lottery draw</strong>: <span class="mono">logits</span> are each ticket's raw weight, temperature tunes "by skill or by luck", top-k/top-p first strike out hopeless tickets, the penalty pushes down tickets that keep showing up lately, and finally <span class="mono">dist</span> draws one by weight - or <span class="mono">greedy</span> simply takes the heaviest. The same pile of tickets, with different rules, draws an entirely different "personality".
</div>

<h2>From logits to a token</h2>
<div class="flow">
  <div class="node"><div class="nt">logits</div><div class="nd">one score per token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">penalties</div><div class="nd">damp repeats</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">top_k / top_p</div><div class="nd">prune candidates</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">temp</div><div class="nd">shape distribution</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">dist</div><div class="nd">pick one token</div></div>
</div>
<p>First, see sampling's input and output clearly. The input is a <strong>candidate array</strong>: one record per token in the vocab, holding the token id, its logit (raw score), and a probability p computed later. The output is <strong>one</strong> selected id among them. What sampling does is work over this array and finally pick one record.</p>
<p>This candidate array is called <span class="mono">llama_token_data_array</span> in llama.cpp, each record a <span class="mono">llama_token_data</span>. The whole pipeline is essentially <strong>repeatedly rewriting this array</strong>: some samplers slam certain candidates' logit to negative infinity (a strike-out), some recompute probabilities and re-sort, and the last step selects one, recording its index in the array's <span class="mono">selected</span> field.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Why so many steps instead of picking in one shot? Because "pick the next word" has varied needs: code wants rigor (lean deterministic), poetry wants divergence (lean random), and you must avoid rehashing the same phrases. Splitting these needs into independent small operations to combine on demand is far more flexible than one giant "do-it-all sampling function".
</div>
<p>So the pipeline looks roughly like this: penalties damp repeats first, top-k/top-p chop the long tail, then temperature tunes the softness of what remains, and finally dist draws one by probability (or greedy takes the max). Each step does one small thing; stacked together they are a complete sampling strategy.</p>
<p>Worth stressing: this pipeline touches only logits/probabilities, <strong>never the model itself</strong>. The model dutifully computes the same row of logits each step; how a word is chosen from them is entirely up to the sampling layer. So "change the generation style" needs no change to the model, just sampling parameters - which is why one model can be rigorous one moment and wildly imaginative the next.</p>
<p>Picture this row of logits as a <strong>mountain range of peaks and valleys</strong>: the more the model favors a word, the higher its peak. Sampling chooses by the shape of this range - pick only near the high peaks (conservative), or give the foothills a chance too (divergent). Every later sampler is really <strong>reshaping this range's outline</strong> before deciding where to land. Hold this picture, and top-k and temperature later all become easy. Walking one concrete example through this pipeline makes it click:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one sampling step</b>: 5 candidate words - watch a row of logits become the single chosen token (numbers illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) logits</h5>
      <div class="cellrow"><span class="vc">3.2</span><span class="vc">2.1</span><span class="vc">1.0</span><span class="vc">0.5</span><span class="vc">-0.3</span></div>
      <div class="tlab">cat / dog / sky / run / blue</div></div>
    <div class="op">/T<br>T=0.7</div>
    <div class="stn"><h5>(2) temperature</h5>
      <div class="cellrow"><span class="vc">4.6</span><span class="vc">3.0</span><span class="vc">1.4</span><span class="vc">0.7</span><span class="vc">-.4</span></div>
      <div class="tlab">T&lt;1 widens gaps</div></div>
    <div class="op">top-k<br>k=3</div>
    <div class="stn"><h5>(3) truncate</h5>
      <div class="cellrow"><span class="vc hot">4.6</span><span class="vc hot">3.0</span><span class="vc hot">1.4</span><span class="vc dim">0.7</span><span class="vc dim">-.4</span></div>
      <div class="tlab">keep the top 3 only</div></div>
    <div class="op">softmax<br>top-p .9</div>
    <div class="stn"><h5>(4) probs -&gt; sample</h5>
      <div class="cellrow"><span class="vc blue">.78</span><span class="vc blue">.18</span><span class="vc dim">.04</span></div>
      <div class="tlab">draw one by probability -&gt; <strong>cat</strong></div></div>
  </div>
</div>

<h2>The sampler interface</h2>
<pre class="code"><span class="cm">// simplified from include/llama.h</span>
<span class="kw">struct</span> <span class="fn">llama_sampler_i</span> {
    <span class="kw">const char</span> * (*name)  (...);                          <span class="cm">// name (nullable)</span>
    <span class="kw">void</span> (*accept)(llama_sampler *, llama_token);          <span class="cm">// feed back chosen token (nullable)</span>
    <span class="kw">void</span> (*apply) (llama_sampler *, llama_token_data_array * cur_p); <span class="cm">// edit/rank candidates (required)</span>
    <span class="kw">void</span> (*reset)(llama_sampler *);                        <span class="cm">// clear state (nullable)</span>
};
<span class="kw">struct</span> <span class="fn">llama_sampler</span> { <span class="kw">const</span> llama_sampler_i * iface; llama_sampler_context_t ctx; };</pre>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">required</span><span class="name">apply(cur_p)</span></div><div class="ld">rewrite the candidate array: strike out / re-rank / recompute probs - each sampler's core</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">nullable</span><span class="name">accept / reset / name / clone / free</span></div><div class="ld">accept feeds the chosen token back to stateful samplers; the rest as needed</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">state</span><span class="name">llama_sampler_context_t ctx</span></div><div class="ld">each sampler's private ledger: penalties keeps history, mirostat feedback</div></div>
</div>
<p>See what a sampler actually is. <span class="mono">llama_sampler_i</span> is just a set of function pointers: <span class="mono">apply</span> (the core, rewrites the candidate array), <span class="mono">accept</span> (feeds the chosen token back so stateful samplers can keep tally), <span class="mono">reset</span> (clear state), plus name/clone/free. With a state blob <span class="mono">ctx</span>, it forms a <span class="mono">llama_sampler</span>.</p>
<p>Here <span class="mono">apply</span> is the only <strong>required</strong> one - it takes the candidate array and edits it by its own rule (strike some out, re-rank, recompute probabilities). <span class="mono">accept</span> is nullable, needed only by samplers with memory: the penalty must remember which tokens appeared before, mirostat must adjust by feedback; both use accept to take "the just-chosen token" into their state.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This "a set of function pointers + a state blob" design should look familiar - it is another use of that <strong>interface + implementation</strong> pattern from L10's backends and L19's memory interface. Each sampler need only implement these functions to be scheduled uniformly; the engine does not care whether you are top-k or mirostat inside, it just calls apply in order.
</div>
<p>Abstracting samplers into a unified interface buys <strong>composability</strong> above all. Since they look alike, they can line up like blocks, each acting on the same candidate array. Next section's "sampler chain" is the direct product of this composability.</p>
<p>A word on the state <span class="mono">ctx</span>: it is each sampler's private ledger. A stateless sampler (like top_k) has an almost-empty ctx; a stateful one (penalties/mirostat/grammar) keeps history, parameters, feedback here. Samplers do not interfere with one another, each keeping its own books.</p>
<p>Why are several interface functions marked "nullable"? Because not every sampler needs everything. A pure "one cut" sampler like top_k needs no memory and thus need not implement accept; reset matters only when reusing one sampler across several generations. Making these optional lets the simplest sampler write just an apply - tidy and clear; the interface demands only "the required thing", the rest on demand.</p>

<h2>The sampler chain</h2>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>chain_init</h4><p>Build an empty sampler chain (<span class="mono">llama_sampler_chain</span>).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>chain_add several samplers</h4><p>Add penalties, top_k, top_p, temp, dist... in order; each is an independent sampler.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>sample</h4><p>apply over the candidate array in add-order; the last (dist/greedy) picks the token.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>accept</h4><p>Feed the chosen token back into the chain so stateful samplers (penalties etc.) remember it.</p></div></div>
</div>
<p>The sampler chain <span class="mono">llama_sampler_chain</span> is itself a sampler - it holds a list of child samplers, and its <span class="mono">apply</span> simply runs each child's apply <strong>in add-order</strong>. This is the classic "composite" pattern: a chain looks like one sampler outside, while inside it is a queue of many.</p>
<pre class="code"><span class="cm"># pseudocode: build a sampler chain</span>
chain = <span class="fn">llama_sampler_chain_init</span>(params)
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_penalties</span>(...))    <span class="cm"># damp repeats</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_top_k</span>(40))         <span class="cm"># keep top 40</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_top_p</span>(0.95, 1))     <span class="cm"># nucleus</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_temp</span>(0.8))          <span class="cm"># temperature</span>
chain.<span class="fn">add</span>(<span class="fn">llama_sampler_init_dist</span>(seed))         <span class="cm"># draw by probability</span>
id = <span class="fn">llama_sampler_sample</span>(chain, ctx, -1)         <span class="cm"># run the whole chain -&gt; return a token id</span></pre>
<p>It is straightforward to use: <span class="mono">chain_init</span> an empty chain, <span class="mono">chain_add</span> samplers in the order you want, and finally <span class="mono">llama_sampler_sample</span> does it all - it reads a position's logits, forms the candidate array, runs the whole chain, returns the chosen token id, and conveniently <span class="mono">accept</span>s that token back.</p>
<p><strong>Order matters</strong>. The same few samplers in a different arrangement can give different results: generally do penalties and pruning first (shrink the candidate set), then temperature (tune softness), and dist/greedy last (the actual selection). Putting "selection" last is because every earlier step is preparing a more reasonable candidate distribution for that "final kick".</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Note the chain <strong>takes ownership</strong> of the samplers added to it - once added, freeing the chain frees them too, so you need not track them separately. This "add it and the chain manages it" convention makes assembling a strategy carefree: build a chain, free it whole when done.
</div>
<p>This "chain" design essentially turns "the sampling strategy" into <strong>data</strong> (a list of sampler configs) rather than hardcoded code. So a user tuning a few parameters on the command line / config can assemble endlessly varied sampling behavior with not a line of the engine trunk changed - once more "gather the parts that vary".</p>
<p>An example of order affecting the result. If you put temperature <strong>before</strong> top-p, temperature first flattens the whole distribution and then top-p draws the range, so the nucleus comes out larger and more divergent; conversely, top-p fencing first then heating tunes randomness within an already-tightened small set, more controllable. Same parts, different order, subtly different "personality" - exactly the value of leaving order to user config.</p>
<p>In real projects this chain often has a conventional default order. llama.cpp's upper layer (<span class="mono">common</span>) roughly arranges "penalties -&gt; pruning (top-k/top-p/min-p etc.) -&gt; temperature -&gt; dist". You need not memorize it, but the big principle suffices: <strong>shrink candidates first, tune softness next, draw last</strong>. The vast majority of sampling strategies are just additions and subtractions along this main axis.</p>

<h2>Common samplers and greedy vs dist</h2>
<table class="t">
  <tr><th>Sampler</th><th>What it does</th></tr>
  <tr><td>greedy</td><td>pick the max logit (argmax, deterministic)</td></tr>
  <tr><td>dist</td><td>draw randomly by probability (via seed)</td></tr>
  <tr><td>top_k</td><td>keep only the top k candidates</td></tr>
  <tr><td>top_p</td><td>nucleus: keep the smallest set with cumulative prob p</td></tr>
  <tr><td>min_p</td><td>keep candidates with prob no less than "max x p"</td></tr>
  <tr><td>temp</td><td>scale logits, tune randomness</td></tr>
  <tr><td>penalties</td><td>damp repeated/frequent/seen tokens</td></tr>
  <tr><td>mirostat</td><td>dynamically tune temperature to hold perplexity</td></tr>
</table>
<p>Meet the common samplers. Each owns a stage: some "prune" (shrink the candidate set), some "shape" (change the distribution's form), some "select" (the final call).</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  First the two that make the final call: <span class="mono">greedy</span> always picks the max logit - deterministic, same input always same output, good for reproducible, rigorous scenarios; <span class="mono">dist</span> softmaxes logits into probabilities and then draws one <strong>randomly</strong> by probability, bringing diversity, controlled by a random seed. Whether a chain ends in greedy or dist decides if this generation is "deterministic" or "random".
</div>
<p>Then the two pruning mainstays: <span class="mono">top_k</span> keeps the fixed top k by score and strikes out the rest; <span class="mono">top_p</span> (nucleus) accumulates by probability from high to low, keeping until the cumulative reaches p - the candidate count is <strong>adaptive</strong> (few when the distribution is peaked, many when flat). The two often pair: top_k chops the long tail, then top_p closes adaptively.</p>
<p>The "shape" representative is temperature <span class="mono">temp</span>: divide logits by a temperature T then softmax. Small T makes the distribution peakier (more deterministic), large T flatter (more random). It does not change the candidate set, only the softness. There are also <span class="mono">penalties</span> to damp repeats, <span class="mono">mirostat</span> to dynamically tune temperature and hold perplexity, each with a specialty.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  The 2023-era global sampling functions (<span class="mono">llama_sample_top_k</span>/<span class="mono">llama_sample_top_p</span>/<span class="mono">llama_sample_temperature</span> etc.) are <strong>all removed</strong>, unified into this "sampler object + chain" model. Do not go looking for those functions in old tutorials.
</div>
<p>A word on the <span class="mono">penalties</span> family, since it is closest to everyday experience. It watches recently generated tokens and penalizes oft-repeated words (lowering their logit), so the model is less likely to fall into a "broken record" loop. Common flavors are repeat, frequency, and presence penalties - "penalize if seen", "penalize more the more it appears", "penalize once seen, flatly". Tuning them balances "coherent" against "verbose".</p>

<h2>Back to the main loop and the hand-off</h2>
<p>Connecting sampling back to the main loop: per generated token, <span class="mono">llama_decode</span> (L17) computes logits, the chain picks one id from them, and this id is both turned back into text via the vocab (L20) for display and wrapped into a new batch fed back to <span class="mono">llama_decode</span> for the next step. Sampling is the "<strong>pick the next word</strong>" link in the autoregressive loop.</p>
<p>One hand-off with grammar (L23) to flag early: a grammar constraint is essentially a sampler too (its apply strikes out tokens that break the grammar). But it usually does <strong>not</strong> go into the main chain; it is a separate object applied before or after the chain per <span class="mono">grammar_first</span> - having learned the sampler interface here, grammar in L23 will come naturally.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  So what to truly take from this lesson is a mental model: <strong>sampling = line up a queue of small transforms over the candidate array, then pick one</strong>. The model decides "how likely each word is appropriate", sampling decides "who exactly gets picked this time". Understand it and you see why one model, one prompt, can go from "buttoned-up" to "wildly free" just by tuning parameters.
</div>
<p>A practical detail in passing: to make random generation <strong>reproducible</strong>, the key is <span class="mono">dist</span>'s random seed. Fix the seed and the sampling parameters, and the same prompt runs to identical results - invaluable for debugging and comparison experiments. Conversely, to vary each run, let the seed change with time. Determinism is firmly in your hands.</p>
<p>Finally, clear a common misconception: sampling cannot conjure abilities the model lacks. It can only work on the row of logits the model gives - good sampling lets a model <strong>play to its strengths</strong> (fewer blunders, kept diversity), but cannot invent knowledge the model never learned. So when results are poor, first tell "the model is weak" from "the sampling is mistuned": the former needs a different model / fine-tuning (L24), the latter just parameter tweaks.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What does temperature actually do? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Temperature T scales logits: divide each score by T, then softmax into probabilities. T=1 is as-is; T below 1 amplifies big scores further and squashes small ones, making the distribution <strong>peakier</strong>, the model leaning toward the most likely word (more deterministic, more conservative); T above 1 flattens the gaps, making it <strong>flatter</strong> so longshots get a chance (more random, more creative).</p>
    <p>The two extremes are interesting: as T approaches 0, the distribution peaks down to just the max, and temperature sampling <strong>degenerates into greedy</strong>; at very large T the distribution nears uniform, almost blind guessing. So temperature is a continuous "deterministic &lt;-&gt; random" knob, and greedy is merely one extreme special case of it.</p>
    <p>Remember temperature <strong>only changes softness, not the candidate set</strong> - it deletes no token, only redistributes everyone's probability. Deleting candidates is top-k/top-p's job. The two operations are orthogonal and combine well: use top-p to fence a reasonable candidate set, then temperature to tune the randomness within that set.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> How do top-k and top-p differ? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">top_k</span> keeps a <strong>fixed count</strong>: sort candidates by score, keep the top k, strike out the rest. Simple and direct, but with a flaw - when the distribution is peaked, many hopeless ones sneak into the k, and when it is flat, good candidates may be shut out. It does not look at the distribution's shape, only counts.</p>
    <p><span class="mono">top_p</span> (nucleus) keeps the <strong>smallest set whose cumulative probability reaches p</strong>: accumulate by probability from high to low, stopping once it passes p. Its candidate count is <strong>adaptive</strong> - maybe just two or three when peaked, dozens when flat. This "closing by probability density" is often more reasonable than a fixed k.</p>
    <p>In practice the two are often <strong>stacked</strong>: top_k (say 40) chops the vast long tail and bounds cost, then top_p (say 0.95) closes adaptively among the rest. One governs "how many at most", the other "how many by quality"; together they are both fast and steady.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Why a "chain" rather than one big function? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because of <strong>composability + configurability + state isolation</strong>. Each sampler is an independent small part with its own state (penalties keeps history, mirostat keeps feedback, dist keeps the RNG), with adjustable order and free add/remove. A user writes a list of sampler names and parameters in config, and the engine assembles a chain to order - whatever strategy you want, without changing a line of engine code.</p>
    <p>Compare "one hardcoded big sampling function": there every new sampling means edits to the main function, ever more ifs, tangled parameters. Split into a chain, adding a new sampler is just one more independent implementation, with zero impact on the existing ones. This is the same "small parts compose complex behavior" as L11's operators and L16's graph blocks.</p>
    <p>One more detail: a sampler like grammar (L23) usually does <strong>not</strong> enter the main chain but is applied before or after per <span class="mono">grammar_first</span>. This shows the "chain" is not rigid either - it leaves room to insert special constraints at the right spot, flexible enough.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Sampling = prune and shape the candidate array <span class="mono">llama_token_data_array</span>, then pick one token.</li>
    <li>Sampler <span class="mono">llama_sampler_i</span>: <span class="mono">apply</span> (required, edits candidates) + <span class="mono">accept</span> (feed back chosen token) + reset; with state <span class="mono">ctx</span> it forms a <span class="mono">llama_sampler</span>.</li>
    <li>Sampler chain: <span class="mono">chain_init</span> -&gt; <span class="mono">chain_add</span> several samplers -&gt; <span class="mono">sample</span>, applying each <strong>in order</strong>.</li>
    <li><span class="mono">greedy</span>=pick the max (deterministic), <span class="mono">dist</span>=random by probability; <span class="mono">top_k</span>/<span class="mono">top_p</span> prune, <span class="mono">temp</span> tunes softness, <span class="mono">penalties</span>/<span class="mono">mirostat</span> have specialties.</li>
    <li>Old global <span class="mono">llama_sample_*</span> are removed, unified into sampler object + chain; grammar (L23) is a special sampler outside the chain.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Splitting sampling from "one hardcoded big function" into "a string of pluggable small transforms" is the classic <strong>chain-of-responsibility / pipeline</strong> design - the same flavor you saw in ggml's operator chain (L09) and graph blocks (L16): small independent parts composing complex, varied behavior. So "change the generation style" is just swapping a few links and tuning a few numbers, with the model and engine trunk untouched. Understand sampling, and you hold the very knobs that turn a model from "buttoned-up" to "wildly free".
</div>
""",
}

LESSON_22 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
M4a 让模型能算，L20 把文字切成 token，L21 教它怎么选下一个词。可还有个关键问题没解决：你在聊天框里一句一句地说，模型怎么知道"哪句是你说的、哪句是它说的、一轮从哪到哪"？这一课讲<strong>对话模板</strong>——把一串带角色的消息（system/user/assistant），按这个模型认得的格式，拼成一段带特殊标记的提示词字符串。
</p>
<p style="color:var(--muted);margin-top:.4rem">这步看似不起眼，却极其关键：每个模型在训练时，对话都是按某种固定格式喂进去的（ChatML、Llama-2 各不相同）。推理时你必须用<strong>同一种</strong>格式，模型才认得出"轮次"和"角色"。格式拼错，模型轻则答非所问，重则完全不在状态。对话模板就负责把消息正确装进"这个模型的信封"。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  对话模板像<strong>公文的信封格式</strong>：同样一句话，不同机构有不同的抬头和落款。ChatML 把每条消息裹成 <span class="mono">&lt;|im_start|&gt;角色 ... &lt;|im_end|&gt;</span>，Llama-2 用 <span class="mono">[INST] ... [/INST]</span>。模板做的，就是把你的消息装进这个模型训练时认得的那种信封——装错了信封，收信人就读不懂。
</div>

<h2>为什么需要模板</h2>
<div class="flow">
  <div class="node"><div class="nt">消息列表</div><div class="nd">[{system},{user}]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">apply_template</div><div class="nd">套模板</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">提示词串</div><div class="nd">"&lt;|im_start|&gt;..."</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tokenize</div><div class="nd">L20</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">model</div><div class="nd">L17</div></div>
</div>
<p>先看清这步在整条链路里的位置。你的对话是一串结构化消息：每条有个角色（system/user/assistant）和一段内容。但模型吃的不是这种结构，而是一长串 token（L20）。中间必须有一步把"消息列表"压平成"一段字符串"，这一步就是对话模板。</p>
<p>拼出来的字符串里，除了各条消息的正文，还插了一堆<strong>特殊标记</strong>：标明每条消息从哪开始、到哪结束、是谁说的。这些标记对应词表里的特殊 token（L20），模型正是靠它们识别"现在轮到 assistant 说话了""这一轮用户说完了"。</p>
<p>顺序是：消息列表 -&gt; 套模板拼成字符串 -&gt; 交给词表 tokenize -&gt; 进模型。模板负责"结构到文本"，tokenize 负责"文本到 token"，两步接力、缺一不可。这也是为什么这一课紧跟在词表（L20）后面——它的产物正是 tokenize 的输入。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  模板拼出来的是<strong>纯文本</strong>，模型并不知道什么"角色""轮次"的高级概念，它只是学会了"看到 <span class="mono">&lt;|im_start|&gt;</span> 这种标记，就该切换说话人"。模板只是忠实地复刻那个格式。
</div>
<p>反过来想，如果不套模板、直接把用户的话 tokenize 进去会怎样？模型会以为这是一段普通文本的续写，而不是"一轮对话求回应"。它可能继续替用户往下编，而不是作为助手来回答——因为少了那些界定角色和轮次的标记，它根本不知道"该自己说话了"。模板的有无，直接决定模型是"补全"还是"对话"。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  模型本身并不"理解"对话，它只是个超级强大的文本续写器。是对话模板和训练数据一起，把"续写"这件事<strong>伪装</strong>成了"对话"。你看到的一问一答，在模型眼里始终是"给定前文、预测下一个 token"。这个认识很重要——它能帮你理解后面很多看似神奇的行为，其实都只是续写规律的体现。
</div>
<p>正因为对话被编码成了纯文本，很多有趣的事才成为可能：你可以把"系统提示"写进 system 消息里，给模型定个人设；可以把前几轮对话原样拼进去，让它"记住"上下文（其实是每次都把历史重新喂一遍）；甚至可以伪造一段助手的话塞进去，引导它往某个方向接。这些灵活玩法，全建立在"对话不过是一段精心格式化的文本"这个事实上。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  模板里的特殊标记，必须是这个模型词表（L20）里真实存在的 token，模型才认得。所以模板和词表是<strong>配套</strong>的——ChatML 的 <span class="mono">&lt;|im_start|&gt;</span> 之所以好使，是因为对应模型的词表里就有这么一个专门的 token。换个不认识这标记的模型硬套 ChatML，反而会把标记拆成一堆碎字节，适得其反。
</div>

<h2>内建模板表</h2>
<pre class="code"><span class="cm">// 简化自 src/llama-chat.h</span>
<span class="kw">enum</span> <span class="fn">llm_chat_template</span> {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_GEMMA,
    <span class="cm">/* ... 五十多种 ... */</span>
    LLM_CHAT_TEMPLATE_UNKNOWN,
};</pre>
<p>llama.cpp 内置了一大批常见模型的模板，全列在枚举 <span class="mono">llm_chat_template</span> 里——CHATML、LLAMA_2（还有若干变体）、LLAMA_3、GEMMA、MISTRAL、PHI 等等，加起来五十多种。每一种对应一套具体的"标记 + 拼法"。</p>
<p>为什么要硬编码这么多？因为不同模型家族的对话格式是它们训练时定死的，五花八门。把常见的都内置进来，用户拿到一个主流模型，引擎多半能<strong>自动认出</strong>它该用哪套格式，开箱即用，不用手动指定。</p>
<table class="t">
  <tr><th>模板</th><th>消息标记</th></tr>
  <tr><td>ChatML</td><td>&lt;|im_start|&gt;role ... &lt;|im_end|&gt;</td></tr>
  <tr><td>Llama-2</td><td>[INST] ... [/INST]</td></tr>
  <tr><td>Llama-3</td><td>&lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;</td></tr>
  <tr><td>Gemma</td><td>&lt;start_of_turn&gt;role ... &lt;end_of_turn&gt;</td></tr>
</table>
<p>看几个代表就懂了：ChatML（很多模型用）拿 <span class="mono">&lt;|im_start|&gt;</span>/<span class="mono">&lt;|im_end|&gt;</span> 包消息；Llama-2 用 <span class="mono">[INST]</span>/<span class="mono">[/INST]</span> 框用户指令；Gemma 用 <span class="mono">&lt;start_of_turn&gt;</span>；Llama-3 用 <span class="mono">&lt;|start_header_id|&gt;</span> 标角色。标记不同，但意图一样：界定角色和轮次边界。</p>
<p>这套"把每个模型的格式收进一张枚举表"的做法，你应该眼熟——和 L15 把架构收进 <span class="mono">LLM_ARCH</span> 表、L20 把分词器类型收进 <span class="mono">vocab_type</span> 是同一种思路：把"会变的差异"集中成数据，让通用代码照表办事。</p>
<p>你可能会问：模型自己不知道该用哪套格式吗，还要引擎来猜？还真不一定知道。GGUF 文件里<strong>可能</strong>带一个模板字段（很多新模型会写），但也有不少模型没写、或写得不规范。于是 llama.cpp 一边支持读取模型自带的模板，一边内置这几十种常见格式兜底——两手准备，尽量让用户不必手动操心。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这几十种模板看着多，其实大同小异，无非是"用什么符号标角色、用什么符号断轮次、system 消息放哪"几个维度的排列组合。把它们一一编码进枚举，是一种"用工程量换通用性"的取舍：写的时候累一点，换来的是"一个引擎通吃主流模型"的便利。这种"宁可自己多写、也要让用户省心"的态度，贯穿了 llama.cpp 的很多设计。
</div>
<p>顺便说，枚举里那个 <span class="mono">LLM_CHAT_TEMPLATE_UNKNOWN</span> 哨兵也有讲究：当检测既匹配不上名字、又认不出特征时，就落到它。这时引擎会提示"没认出模板"，提醒用户手动指定一个，而不是默默用错格式蒙混过去。给"认不出"留一个明确的出口，是健壮设计的常见手法。</p>
<p>一个常被忽略的细节是 system 消息的处理。不同模板对"系统提示"放哪、怎么标记，分歧最大：有的像 ChatML 一样单列一条 system 消息，有的（如某些 Llama-2 变体）要把它揉进第一条 user 消息里，还有的根本不支持独立的 system。所以同一段系统提示，套不同模板出来的位置可能差很远——这也是为什么换模型时，光改提示词内容还不够，得让模板替你摆对位置。</p>

<h2>检测与应用</h2>
<pre class="code"><span class="cm"># 伪代码: 套用对话模板</span>
tmpl = <span class="fn">llm_chat_detect_template</span>(template_str)   <span class="cm"># 先按名, 再按内容特征猜</span>
dest = <span class="st">""</span>
<span class="fn">llm_chat_apply_template</span>(tmpl, messages, dest, add_ass=<span class="kw">True</span>)
<span class="cm"># add_ass: 末尾追加 assistant 起始标记, 让模型接着写回答</span></pre>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>模板字符串/名字</h4><p>来自模型 GGUF 元数据或用户指定。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>detect_template</h4><p>先按名精确匹配，认不出就看是否含 &lt;|im_start|&gt;/[INST] 等特征子串。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>apply_template</h4><p>按选定模板，把消息逐条裹上标记、首尾拼成一段字符串。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>add_ass 递话筒</h4><p>为真则末尾追加 assistant 起始标记，让模型接着写回答。</p></div></div>
</div>
<p>有了这张表，剩下两件事：一是<strong>认出</strong>该用哪套模板，二是<strong>套用</strong>它把消息拼出来。</p>
<p>认出靠 <span class="mono">llm_chat_detect_template</span>：它先按名字精确匹配（模型 GGUF 里常自带一个模板名/模板串），认不出就退而看模板内容里有没有 <span class="mono">&lt;|im_start|&gt;</span>、<span class="mono">[INST]</span> 这类特征子串，按特征猜。套用靠 <span class="mono">llm_chat_apply_template</span>：给它模板枚举、消息列表、一个输出字符串，它就按这套格式拼好。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  这里有个参数值得专门说：<span class="mono">add_ass</span>（add assistant）。为真时，拼完所有消息后，会在末尾再追加 assistant 的<strong>起始标记</strong>（但不含内容）——相当于把话筒递给模型，让它从"该 assistant 说话"的位置开始生成回答。要模型续写回答时打开它；只想补全已有文本时关掉。
</div>
<p>消息本身的结构很简单：<span class="mono">llama_chat_message</span> 就两个字段，<span class="mono">role</span>（角色字符串，如 "user"）和 <span class="mono">content</span>（内容）。一串这样的消息，就是 <span class="mono">apply_template</span> 的输入；它在内部按选定模板，把每条消息裹上对应标记、首尾拼接，吐出最终那段提示词。拿一组具体消息走一遍就清楚了：</p>
<div class="trace">
  <div class="tcap"><b>追踪模板拼接</b>：结构化消息怎么被压平成一串带特殊标记的纯文本（送进 tokenize 之前的最后一步；内容为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 消息列表</h5>
      <div class="cellrow"><span class="vc">system: "You are helpful."</span><span class="vc">user: "Hi"</span></div>
      <div class="tlab">2 条带角色的结构化消息</div></div>
    <div class="op">套 ChatML</div>
    <div class="stn"><h5>② 压平 + 加标记</h5>
      <div class="cellrow"><span class="vc hot">&lt;|im_start|&gt;system</span><span class="vc">You are helpful.</span><span class="vc hot">&lt;|im_end|&gt;</span><span class="vc hot">&lt;|im_start|&gt;user</span><span class="vc">Hi</span><span class="vc hot">&lt;|im_end|&gt;</span><span class="vc hot">&lt;|im_start|&gt;assistant</span></div>
      <div class="tlab">标记包住每条正文，结尾把话筒递给 assistant</div></div>
    <div class="op">→ tokenize</div>
    <div class="stn"><h5>③ 交给 L20</h5>
      <div class="cellrow"><span class="vc blue">tokenize</span></div>
      <div class="tlab">这串纯文本再切成 token id</div></div>
  </div>
</div>
<p>检测这一步其实暗藏玄机。最理想的情况是模型 GGUF 里写明了模板名，一查便知；但现实里常常只给出一段模板<strong>内容</strong>（Jinja 文本），没有名字。这时只能靠"内容里有没有某些特征标记"来反推——看到 <span class="mono">[INST]</span> 就猜 Llama-2、看到 <span class="mono">&lt;|im_start|&gt;</span> 就猜 ChatML。这种基于特征的启发式不是百分百可靠，但覆盖了绝大多数情况。</p>
<p>套用这一步也比看上去讲究。同一套格式，system 消息有的拼在最前、有的并进第一条 user 消息、有的干脆不支持；多轮对话里，历史消息要不要重复加标记、最后一轮怎么收尾，每种模板都有自己的规矩。<span class="mono">apply_template</span> 把这些细节按模板类型一一处理妥当，你只管递进去一个消息列表，它还你一段格式严丝合缝的提示词。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  调试对话效果时，不妨把 <span class="mono">apply_template</span> 拼出来的那段字符串<strong>原样打印</strong>出来看看。很多"模型不好好回答"的问题，根子就在拼出来的提示词格式不对——少了个标记、system 放错了位置、add_ass 忘了开。先看清喂进去的到底长什么样，往往比反复调参数更快定位问题。
</div>

<h2>两条路：内建 vs Jinja</h2>
<div class="cols">
  <div class="col"><h4>内建（llama-chat.cpp）</h4><p>固定枚举、纯字符串拼接、零依赖、快；只认预定义的几十种。C API <span class="mono">llama_chat_apply_template</span> 走这条。</p></div>
  <div class="col"><h4>Jinja（common/jinja）</h4><p>渲染<strong>任意</strong>模板：模型自带的 Jinja chat_template 原样执行，最忠实。<span class="mono">common/chat.cpp</span> 封装，还支持工具调用。</p></div>
</div>
<p>内建模板只覆盖"已知的那些模型"。要是来了个全新模型、带着自己独特的模板呢？这就引出第二条路：<strong>Jinja</strong>。</p>
<p>内建这条路（<span class="mono">src/llama-chat.cpp</span>）是纯 C++ 字符串拼接，固定枚举、零依赖、快，但只认预定义的那几十种。C API <span class="mono">llama_chat_apply_template</span> 走的就是这条。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  C API <span class="mono">llama_chat_apply_template</span> <strong>只收模板字符串、不带模型参数</strong>（旧版带 <span class="mono">llama_model *</span> 的重载已移除），注释也明说"不用 jinja，只支持预定义列表"。
</div>
<p>Jinja 这条路（vendored 在 <span class="mono">common/jinja/</span>）能渲染<strong>任意</strong>模板：模型在 GGUF 里自带的 chat_template（往往是一段 Jinja 文本）可以被原样执行，最忠实。上层 <span class="mono">common/chat.cpp</span> 的 <span class="mono">common_chat_templates</span> 封装了它，还顺带支持工具调用、输出解析等高级花样。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  两条路分工很清楚：内建管"已知模型、要快要稳"，Jinja 管"任意模型、要忠实"。衔接上，无论哪条路，拼好的提示词都要再经 L20 的 tokenize、进 L17 的 decode——对话模板只是把"消息"变成"字符串"的那一棒。
</div>
<p>为什么不干脆<strong>全</strong>用 Jinja，省得维护几十种内建模板？因为 Jinja 是一套完整的模板语言，要带一个解释器、要解析执行任意逻辑，开销和复杂度都不小。对那些格式早已固定的常见模型，用几行 C++ 直接拼，又快又稳、还没有解析任意模板带来的安全顾虑。所以内建这条"快路"有它不可替代的价值。</p>
<p>反过来，为什么又非要有 Jinja 不可？因为模型层出不穷，总有内建表里没有的新格式、或带着复杂条件逻辑的模板（比如"有 system 就这样拼、没有就那样拼""带工具定义时再加一段"）。这些用固定枚举根本表达不了，只能靠一个真正的模板引擎去执行。两条路各补各的短，合起来才既覆盖广、又跑得快。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  对话模板这一层，本质上是在弥合"人类的对话观"和"模型的文本观"之间的鸿沟。人觉得对话是你一言我一语的结构，模型只认一条连续的 token 流。模板就是这两种世界观之间的翻译协议——而它居然能用"一张枚举表 + 一个可选的 Jinja 引擎"就基本搞定，足见把复杂性收进数据是多么有力的一招。
</div>
<p>还值得一提的是工具调用（function calling）这类高级用法。当你想让模型调用外部工具时，工具的定义、调用的格式、返回的拼接，都要按特定约定塞进提示词——这远超内建模板"拼几条消息"的能力，正是 common 层结合 Jinja 与语法约束（L23）来做的。所以对话模板不只是"聊天"，它还是更复杂的"结构化交互"的地基。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> add_ass（add assistant）这个开关到底干嘛？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>它决定要不要在拼好的提示词<strong>末尾</strong>，补上 assistant 角色的起始标记（如 ChatML 的 <span class="mono">&lt;|im_start|&gt;assistant</span>），但不含任何内容。等于在纸上写好"助手："然后把笔递过去——模型自然会从这个位置接着往下写回答。</p>
    <p>为真是<strong>聊天</strong>的常态：你想要模型作为助手回应，就得把"该它说话"的起点标出来。为假则用于<strong>补全</strong>：你只想让模型接着某段已有文本往下写，不需要切换到 assistant 身份。一个开关，区分了"对话"和"续写"两种用法。</p>
    <p>这也解释了一个常见现象：如果忘了开 add_ass，模型有时会"自言自语"地替用户多说几句，而不是直接回答——因为提示词停在了用户那一轮，没给它"轮到你了"的信号。理解这个开关，能省掉不少"模型怎么不好好回答"的困惑。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么 C API 的 apply_template 不再带 model 参数？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为套模板这件事，需要的只是"<strong>模板字符串 + 消息列表</strong>"，跟整个模型没关系。模板串本身可以来自 GGUF 元数据、也可以由用户直接指定；把 model 从参数里拿掉，函数职责更单一、更好测试、也更灵活。</p>
    <p>这是个典型的<strong>解耦</strong>动作，和 L20 把 <span class="mono">llama_token_bos</span> 等改成 <span class="mono">llama_vocab_*</span> 一个道理：让每个 API 只依赖它真正需要的东西。旧版带 <span class="mono">llama_model *</span> 的重载已经移除，看老代码时别再按那个签名调用。</p>
    <p>注释里还点明：这个 C API <strong>不走 jinja</strong>，只支持内建的预定义模板列表。换句话说，它是"快而专"的那条路；要 jinja 的全部灵活性，得上 <span class="mono">common/chat.cpp</span> 那层。API 的边界划得很清楚。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 内建模板 vs Jinja，何时用哪个？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>原则上：模型在 GGUF 里<strong>自带了 chat_template</strong>（一段 Jinja 文本）、或你需要工具调用这类高级特性时，走 Jinja（<span class="mono">common/chat.cpp</span> 的 <span class="mono">common_chat_templates</span>，开 <span class="mono">use_jinja</span>），最忠实于模型作者的意图。</p>
    <p>反过来，模型是<strong>已知的主流款</strong>、或你想要零依赖、要快要稳时，用内建枚举即可——几十种常见格式都覆盖了，纯 C++ 拼接没有额外开销。命令行小工具、嵌入式场景，往往选这条。</p>
    <p>两者不是对立，而是<strong>分层覆盖</strong>："已知模型"由内建快速搞定，"任意模型"由 Jinja 兜底。这种"常见走快路、罕见走通路"的设计，和 L20 词表的"高频片段直接收、罕见字符靠字节回退"是同一种务实智慧。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>对话模板把带角色的消息列表 -&gt; 该模型约定格式的提示词字符串，再交给 L20 tokenize。</li>
    <li>内建模板枚举 <span class="mono">llm_chat_template</span>（五十多种）；<span class="mono">llm_chat_detect_template</span>（先按名、再按特征子串）+ <span class="mono">llm_chat_apply_template</span>（渲染，<span class="mono">add_ass</span> 控制是否递话筒）。</li>
    <li><span class="mono">llama_chat_message</span> 只有 <span class="mono">role</span> + <span class="mono">content</span> 两个字段。</li>
    <li>C API <span class="mono">llama_chat_apply_template</span> <strong>只收模板字符串、无 model 参数</strong>，且只走内建、不用 jinja。</li>
    <li>两条路：内建（<span class="mono">llama-chat.cpp</span>，快/已知模型）vs Jinja（<span class="mono">common/jinja</span> + <span class="mono">common/chat.cpp</span>，任意模型/工具调用）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  对话模板把"<strong>模型的对话方言</strong>"收进一张表——同样的消息，换个模型就换个信封，引擎主干不必关心。它和 L15 的"表驱动架构"、L20 的"表驱动分词"是同一种智慧：把"每个模型各不相同的部分"沉淀成数据/模板，让通用代码照着办。而内建与 Jinja 两条路，又是"常见走快路、罕见走通路"的经典分层。读懂它，你就明白为什么同一个 llama.cpp 能流利地说几十种模型的"话"。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
M4a let the model compute, L20 cut text into tokens, L21 taught it to pick the next word. But one key question remains: you speak sentence by sentence in a chat box, so how does the model know "which line is yours, which is its own, where one turn starts and ends"? This lesson covers <strong>chat templates</strong> - assembling a list of role-tagged messages (system/user/assistant), in the format this model recognizes, into a prompt string with special markers.
</p>
<p style="color:var(--muted);margin-top:.4rem">This step looks minor but is crucial: every model was trained with conversations fed in some fixed format (ChatML, Llama-2 differ). At inference you must use the <strong>same</strong> format for the model to recognize "turns" and "roles". Get the format wrong and the model is, at best, off-topic; at worst, completely out of character. The chat template is what packs messages correctly into "this model's envelope".</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A chat template is like an <strong>official envelope format</strong>: the same words get different headers and sign-offs at different institutions. ChatML wraps each message as <span class="mono">&lt;|im_start|&gt;role ... &lt;|im_end|&gt;</span>, Llama-2 uses <span class="mono">[INST] ... [/INST]</span>. What the template does is pack your message into the envelope this model learned to recognize at training - wrong envelope, and the recipient cannot read it.
</div>

<h2>Why a template is needed</h2>
<div class="flow">
  <div class="node"><div class="nt">message list</div><div class="nd">[{system},{user}]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">apply_template</div><div class="nd">apply template</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">prompt string</div><div class="nd">"&lt;|im_start|&gt;..."</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tokenize</div><div class="nd">L20</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">model</div><div class="nd">L17</div></div>
</div>
<p>First see where this step sits in the whole pipeline. Your conversation is a list of structured messages: each has a role (system/user/assistant) and some content. But the model doesn't eat this structure, but rather a long string of tokens (L20). There must be a step that flattens "the message list" into "one string", and that step is the chat template.</p>
<p>The assembled string, besides each message's body, inserts a pile of <strong>special markers</strong>: marking where each message starts, ends, and who spoke. These markers correspond to special tokens in the vocab (L20), and the model relies on them to recognize "it is the assistant's turn now", "the user is done with this turn".</p>
<p>The order is: message list -&gt; apply the template into a string -&gt; hand to the vocab to tokenize -&gt; into the model. The template handles "structure to text", tokenize handles "text to tokens" - two relays, neither dispensable. This is also why this lesson follows the vocab (L20) closely - its product is exactly tokenize's input.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  What the template assembles is <strong>plain text</strong>; the model has no high-level notion of "role" or "turn", it merely learned that "seeing a marker like <span class="mono">&lt;|im_start|&gt;</span> means switch speakers". The template just faithfully reproduces that format.
</div>
<p>Conversely, what if you skip the template and tokenize the user's words directly? The model would think it is the continuation of some ordinary text, not "a turn of dialogue asking for a reply". It might keep writing on the user's behalf rather than answer as the assistant - because without the markers delimiting role and turn, it has no idea "it is its turn to speak". The presence or absence of a template directly decides whether the model "completes" or "converses".</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The model itself does not "understand" conversation; it is just an extremely powerful text continuer. It is the chat template together with the training data that <strong>disguises</strong> "continuation" as "conversation". The question-and-answer you see is, to the model, always "given the prefix, predict the next token". This realization matters - it helps you understand why many seemingly magical behaviors later are just continuation patterns at work.
</div>
<p>Precisely because conversation is encoded as plain text, many interesting things become possible: you can write a "system prompt" into the system message to give the model a persona; you can splice prior turns verbatim so it "remembers" context (really, the history is re-fed each time); you can even insert a fabricated assistant line to steer where it continues. These flexible tricks all rest on the fact that "a conversation is just carefully formatted text".</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  The template's special markers must be tokens that truly exist in this model's vocab (L20) for the model to recognize them. So template and vocab are a <strong>matched set</strong> - ChatML's <span class="mono">&lt;|im_start|&gt;</span> works because that model's vocab has a dedicated token for it. Force ChatML onto a model that does not know the marker and it gets split into a pile of byte fragments, backfiring.
</div>

<h2>The built-in template table</h2>
<pre class="code"><span class="cm">// simplified from src/llama-chat.h</span>
<span class="kw">enum</span> <span class="fn">llm_chat_template</span> {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_GEMMA,
    <span class="cm">/* ... fifty-odd of them ... */</span>
    LLM_CHAT_TEMPLATE_UNKNOWN,
};</pre>
<p>llama.cpp ships a big batch of common models' templates, all listed in the enum <span class="mono">llm_chat_template</span> - CHATML, LLAMA_2 (plus several variants), LLAMA_3, GEMMA, MISTRAL, PHI, and so on, fifty-odd in total. Each corresponds to a concrete "markers + assembly rule".</p>
<p>Why hardcode so many? Because each model family's chat format is fixed at its training time, and they vary widely. Building the common ones in means that, given a mainstream model, the engine can mostly <strong>auto-detect</strong> which format to use, working out of the box without manual specification.</p>
<table class="t">
  <tr><th>Template</th><th>Message markers</th></tr>
  <tr><td>ChatML</td><td>&lt;|im_start|&gt;role ... &lt;|im_end|&gt;</td></tr>
  <tr><td>Llama-2</td><td>[INST] ... [/INST]</td></tr>
  <tr><td>Llama-3</td><td>&lt;|start_header_id|&gt;role&lt;|end_header_id|&gt;</td></tr>
  <tr><td>Gemma</td><td>&lt;start_of_turn&gt;role ... &lt;end_of_turn&gt;</td></tr>
</table>
<p>A few representatives make it clear: ChatML (used by many models) wraps messages with <span class="mono">&lt;|im_start|&gt;</span>/<span class="mono">&lt;|im_end|&gt;</span>; Llama-2 frames user instructions with <span class="mono">[INST]</span>/<span class="mono">[/INST]</span>; Gemma uses <span class="mono">&lt;start_of_turn&gt;</span>; Llama-3 marks roles with <span class="mono">&lt;|start_header_id|&gt;</span>. Different markers, same intent: delimit role and turn boundaries.</p>
<p>This "gather every model's format into one enum table" should look familiar - the same idea as L15 gathering architectures into the <span class="mono">LLM_ARCH</span> table and L20 gathering tokenizer types into <span class="mono">vocab_type</span>: concentrate "the differences that vary" into data, and let generic code act by the table.</p>
<p>You might ask: does the model not know which format to use, needing the engine to guess? Not necessarily. The GGUF file <strong>may</strong> carry a template field (many new models write one), but plenty of models omit it or write it loosely. So llama.cpp both supports reading the model's own template and builds in these dozens of common formats as a backstop - a two-pronged setup to spare the user manual fuss.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  These dozens of templates look many but are largely alike, just permutations of a few dimensions: "which symbol marks the role, which breaks the turn, where the system message goes". Encoding them one by one into an enum is an "engineering effort for generality" trade-off: a bit more work to write, in exchange for "one engine handling mainstream models". This "rather write more ourselves than burden the user" attitude runs through much of llama.cpp's design.
</div>
<p>By the way, the enum's <span class="mono">LLM_CHAT_TEMPLATE_UNKNOWN</span> sentinel has a purpose too: when detection matches neither a name nor a feature, it lands here. The engine then signals "template not recognized", prompting the user to specify one manually rather than silently using a wrong format. Leaving a clear exit for "cannot recognize" is a common robust-design technique.</p>
<p>An often-overlooked detail is how the system message is handled. Templates diverge most on where the "system prompt" goes and how it is marked: some, like ChatML, list a separate system message; some (like certain Llama-2 variants) fold it into the first user message; some support no standalone system at all. So the same system prompt can land in very different places under different templates - which is also why, switching models, changing the prompt text alone is not enough; the template must place it correctly for you.</p>

<h2>Detection and application</h2>
<pre class="code"><span class="cm"># pseudocode: apply a chat template</span>
tmpl = <span class="fn">llm_chat_detect_template</span>(template_str)   <span class="cm"># by name first, then guess by content</span>
dest = <span class="st">""</span>
<span class="fn">llm_chat_apply_template</span>(tmpl, messages, dest, add_ass=<span class="kw">True</span>)
<span class="cm"># add_ass: append the assistant start marker so the model writes the reply</span></pre>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>template string/name</h4><p>From the model's GGUF metadata or user-specified.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>detect_template</h4><p>Match by name first; failing that, look for feature substrings like &lt;|im_start|&gt;/[INST].</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>apply_template</h4><p>By the chosen template, wrap each message in markers and concatenate into one string.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>add_ass hands the mic</h4><p>If true, append the assistant start marker so the model continues the reply.</p></div></div>
</div>
<p>With this table, two things remain: <strong>recognize</strong> which template to use, and <strong>apply</strong> it to assemble the messages.</p>
<p>Recognizing uses <span class="mono">llm_chat_detect_template</span>: it first matches by name exactly (the model's GGUF often carries a template name/string), and failing that, looks for feature substrings like <span class="mono">&lt;|im_start|&gt;</span> or <span class="mono">[INST]</span> in the template body and guesses by feature. Applying uses <span class="mono">llm_chat_apply_template</span>: give it the template enum, the message list, and an output string, and it assembles per that format.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  One parameter deserves a special mention: <span class="mono">add_ass</span> (add assistant). When true, after all messages are assembled it appends the assistant's <strong>start marker</strong> at the end (but no content) - like handing the mic to the model, letting it start generating the reply from where "the assistant should speak". Turn it on when you want the model to continue a reply; off when you only want to complete existing text.
</div>
<p>The message's own structure is simple: <span class="mono">llama_chat_message</span> has just two fields, <span class="mono">role</span> (a role string, e.g. "user") and <span class="mono">content</span> (the content). A list of such messages is the input to <span class="mono">apply_template</span>; internally, per the chosen template, it wraps each message in its markers, concatenates head to tail, and emits that final prompt. Walk a concrete message set through it to see:</p>
<div class="trace">
  <div class="tcap"><b>Tracing template assembly</b>: how structured messages flatten into one marked-up plain-text string (the last step before tokenize; content illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) message list</h5>
      <div class="cellrow"><span class="vc">system: "You are helpful."</span><span class="vc">user: "Hi"</span></div>
      <div class="tlab">2 structured role-tagged messages</div></div>
    <div class="op">apply ChatML</div>
    <div class="stn"><h5>(2) flatten + mark</h5>
      <div class="cellrow"><span class="vc hot">&lt;|im_start|&gt;system</span><span class="vc">You are helpful.</span><span class="vc hot">&lt;|im_end|&gt;</span><span class="vc hot">&lt;|im_start|&gt;user</span><span class="vc">Hi</span><span class="vc hot">&lt;|im_end|&gt;</span><span class="vc hot">&lt;|im_start|&gt;assistant</span></div>
      <div class="tlab">markers wrap each body; ends handing the mic to assistant</div></div>
    <div class="op">-&gt; tokenize</div>
    <div class="stn"><h5>(3) hand to L20</h5>
      <div class="cellrow"><span class="vc blue">tokenize</span></div>
      <div class="tlab">this plain text then splits into token ids</div></div>
  </div>
</div>
<p>Detection actually hides subtlety. Ideally the GGUF states the template name and a lookup settles it; but in reality it often gives only a template <strong>body</strong> (Jinja text) with no name. Then one can only infer from "whether certain feature markers appear in the body" - see <span class="mono">[INST]</span> and guess Llama-2, see <span class="mono">&lt;|im_start|&gt;</span> and guess ChatML. This feature-based heuristic is not 100% reliable but covers the vast majority.</p>
<p>Application is also fussier than it looks. For the same format, the system message may go at the very front, be merged into the first user message, or not be supported at all; in multi-turn dialogue, whether history repeats the markers and how the last turn closes - each template has its own rules. <span class="mono">apply_template</span> handles these details per template type, so you just hand in a message list and it returns a precisely formatted prompt.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  When debugging chat behavior, print the string <span class="mono">apply_template</span> produces <strong>verbatim</strong> and look at it. Many "the model won't answer properly" issues are rooted in a wrong assembled prompt - a missing marker, the system placed wrong, add_ass forgotten. Seeing exactly what is fed in often locates the problem faster than repeatedly tuning parameters.
</div>

<h2>Two paths: built-in vs Jinja</h2>
<div class="cols">
  <div class="col"><h4>Built-in (llama-chat.cpp)</h4><p>fixed enum, pure string assembly, zero deps, fast; recognizes only the predefined few dozen. The C API <span class="mono">llama_chat_apply_template</span> takes this path.</p></div>
  <div class="col"><h4>Jinja (common/jinja)</h4><p>renders <strong>any</strong> template: the model's own Jinja chat_template runs as-is, most faithful. <span class="mono">common/chat.cpp</span> wraps it and even supports tool calls.</p></div>
</div>
<p>The built-in templates cover only "the known models". What about a brand-new model bringing its own unique template? That leads to the second path: <strong>Jinja</strong>.</p>
<p>The built-in path (<span class="mono">src/llama-chat.cpp</span>) is pure C++ string assembly - fixed enum, zero deps, fast, but recognizes only the predefined few dozen. The C API <span class="mono">llama_chat_apply_template</span> takes this path.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  The C API <span class="mono">llama_chat_apply_template</span> <strong>only takes a template string, no model parameter</strong> (the old <span class="mono">llama_model *</span> overload is removed), and its comment plainly says "no jinja, only the predefined list".
</div>
<p>The Jinja path (vendored in <span class="mono">common/jinja/</span>) can render <strong>any</strong> template: the chat_template a model carries in GGUF (often a piece of Jinja text) runs as-is, most faithful. The upper layer <span class="mono">common/chat.cpp</span>'s <span class="mono">common_chat_templates</span> wraps it and also supports tool calls, output parsing, and other advanced tricks.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The two paths divide cleanly: built-in for "known models, fast and steady", Jinja for "any model, faithful". For the hand-off, whichever path, the assembled prompt still goes through L20's tokenize and into L17's decode - the chat template is merely the leg that turns "messages" into "a string".
</div>
<p>Why not just use Jinja for <strong>everything</strong> and skip maintaining dozens of built-in templates? Because Jinja is a full template language, requiring an interpreter and executing arbitrary logic, with non-trivial cost and complexity. For common models whose format is long fixed, a few lines of C++ assembling directly is faster, steadier, and free of the security concerns of running arbitrary templates. So the built-in "fast path" has irreplaceable value.</p>
<p>Conversely, why is Jinja indispensable? Because models keep appearing, and there are always new formats absent from the built-in table, or templates with complex conditional logic ("assemble this way if there is a system message, that way if not", "add a section when tool definitions are present"). A fixed enum simply cannot express these; only a real template engine can execute them. Each path covers the other's weakness; together they are both broad and fast.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The chat-template layer essentially bridges the gap between "the human view of conversation" and "the model's view of text". People see dialogue as a back-and-forth structure; the model knows only one continuous token stream. The template is the translation protocol between these worldviews - and that it largely manages with "one enum table + an optional Jinja engine" shows how powerful settling complexity into data really is.
</div>
<p>Also worth mentioning is the advanced use of tool calling (function calling). When you want the model to call an external tool, the tool definitions, the call format, and the splicing of returns must all be packed into the prompt per a specific convention - far beyond the built-in template's "stitch a few messages", and exactly what the common layer does by combining Jinja with grammar constraints (L23). So chat templates are not only "chat"; they are the foundation of more complex "structured interaction" too.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What does the add_ass (add assistant) switch actually do? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>It decides whether to append, at the <strong>end</strong> of the assembled prompt, the assistant role's start marker (like ChatML's <span class="mono">&lt;|im_start|&gt;assistant</span>) with no content. It is like writing "Assistant:" on the page and handing over the pen - the model naturally continues writing the reply from that spot.</p>
    <p>True is the norm for <strong>chat</strong>: if you want the model to respond as the assistant, you must mark the start of "its turn to speak". False is for <strong>completion</strong>: when you only want the model to continue some existing text, with no switch to the assistant identity. One switch separates "converse" from "continue".</p>
    <p>This also explains a common phenomenon: forget add_ass and the model sometimes "talks to itself", adding a few more lines on the user's behalf instead of answering directly - because the prompt stopped at the user's turn, with no "your turn" signal. Understanding this switch saves much "why won't the model answer properly" confusion.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why does the C API apply_template no longer take a model parameter? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because applying a template needs only "<strong>a template string + a message list</strong>", nothing to do with the whole model. The template string can come from GGUF metadata or be given by the user directly; dropping model from the parameters makes the function's job more single-purpose, easier to test, and more flexible.</p>
    <p>This is a classic <strong>decoupling</strong> move, the same idea as L20 renaming <span class="mono">llama_token_bos</span> etc. to <span class="mono">llama_vocab_*</span>: let each API depend only on what it truly needs. The old overload taking <span class="mono">llama_model *</span> is removed, so do not call it by that signature when reading old code.</p>
    <p>The comment also makes plain: this C API <strong>does not use jinja</strong>, only the built-in predefined template list. In other words, it is the "fast and specialized" path; for jinja's full flexibility you go to the <span class="mono">common/chat.cpp</span> layer. The API draws its boundary clearly.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Built-in vs Jinja, when to use which? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>In principle: when the model <strong>carries its own chat_template</strong> in GGUF (a piece of Jinja text), or you need advanced features like tool calls, go Jinja (<span class="mono">common/chat.cpp</span>'s <span class="mono">common_chat_templates</span> with <span class="mono">use_jinja</span>), most faithful to the model author's intent.</p>
    <p>Conversely, when the model is a <strong>known mainstream one</strong>, or you want zero deps, fast and steady, the built-in enum suffices - it covers dozens of common formats, with no extra cost of pure C++ assembly. Command-line tools and embedded scenarios often pick this path.</p>
    <p>The two are not opposed but a <strong>layered coverage</strong>: "known models" handled fast by built-in, "any model" backstopped by Jinja. This "common takes the fast path, rare takes the full path" is the same pragmatic wisdom as L20's vocab "keep high-frequency pieces directly, rare chars via byte fallback".</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>A chat template turns a role-tagged message list -&gt; a prompt string in the model's agreed format, then handed to L20 tokenize.</li>
    <li>Built-in template enum <span class="mono">llm_chat_template</span> (fifty-odd); <span class="mono">llm_chat_detect_template</span> (by name, then feature substring) + <span class="mono">llm_chat_apply_template</span> (render, <span class="mono">add_ass</span> controls whether to hand over the mic).</li>
    <li><span class="mono">llama_chat_message</span> has just two fields, <span class="mono">role</span> + <span class="mono">content</span>.</li>
    <li>The C API <span class="mono">llama_chat_apply_template</span> <strong>takes only a template string, no model parameter</strong>, and goes built-in only, not jinja.</li>
    <li>Two paths: built-in (<span class="mono">llama-chat.cpp</span>, fast / known models) vs Jinja (<span class="mono">common/jinja</span> + <span class="mono">common/chat.cpp</span>, any model / tool calls).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The chat template gathers "<strong>a model's conversational dialect</strong>" into a table - the same messages, a different model, a different envelope, with the engine trunk none the wiser. It is the same wisdom as L15's "table-driven architecture" and L20's "table-driven tokenization": settle "the part each model differs in" into data/templates and let generic code act by it. And the built-in vs Jinja two paths are the classic "common takes the fast path, rare takes the full path" layering. Understand it, and you see why one llama.cpp can fluently speak the "tongue" of dozens of models.
</div>
""",
}

LESSON_23 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
到这里，模型已经能聊天了（L20-L22）。但很多真实场景要的不只是"聊得通"，而是<strong>格式严格正确</strong>：调一个 API 要合法的 JSON、填一张表要规定的字段、抽取信息要固定的结构。模型靠概率生成，难免偶尔跑偏——这一课讲的 <strong>GBNF 语法约束</strong>，就是给生成套上一副"护栏"，让它<strong>不可能</strong>产出格式非法的东西。
</p>
<p style="color:var(--muted);margin-top:.4rem">它的思路很巧：不是生成完再检查、不合格就重来（那样既慢又不保险），而是在<strong>每一步采样时</strong>就把"此刻语法不允许的 token"统统划掉，模型只能在合法的路上往前走。于是无论模型多想跑偏，它都迈不出语法的边界——结果<strong>永远合法</strong>，一次成型。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  语法约束像<strong>表单上的下拉框</strong>：你不能随便填，只能从给定选项里挑。grammar 在每一步采样前，把"此刻不该出现的 token"全部置灰（设成负无穷），模型只能从剩下的合法选项里选一个。一步一个下拉框，连起来就保证整段输出严格符合你定义的格式。
</div>

<h2>为什么需要语法</h2>
<div class="flow">
  <div class="node"><div class="nt">logits</div><div class="nd">所有候选</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">grammar.apply</div><div class="nd">非法 -&gt; -inf</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">采样 L21</div><div class="nd">只从合法里选</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">合法 token</div><div class="nd">绝不越界</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">grammar.accept</div><div class="nd">推进语法</div></div>
</div>
<p>先看清它在采样管线里的位置。L21 讲过，采样是在一排 logits 上裁剪塑形、最后选一个 token。grammar 就是往这条管线里插一个<strong>掩码</strong>环节：在选之前，先把所有"此刻语法不允许"的候选的 logit 砸成负无穷，于是它们的概率变成 0，绝无可能被选中。</p>
<p>选定一个合法 token 之后，还有一步：<strong>推进</strong>语法状态。语法就像一台状态机，刚才放行的那个 token 让它往前走了一格，下一步该允许哪些 token 也随之更新。一掩一进，逐 token 地把整段输出牢牢锁在语法的轨道上。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  为什么非得在 token 级别做、不能事后检查？设想生成一段 JSON，写到一半冒出个非法字符——事后检查只能整段作废重来，又慢又可能反复失败。token 级掩码则保证<strong>每一步都合法</strong>，根本不给"写错"的机会，一次就成。这是"约束前置"对"事后补救"的彻底胜利。
</div>
<p>还有个微妙的好处：因为非法 token 被设成负无穷、概率归零，剩下的合法 token 会重新归一化概率。也就是说，约束不仅"禁止非法"，还让模型在<strong>合法范围内</strong>按它原本的偏好挑——既守了规矩，又尽量保留了模型的判断。强约束和模型智能，在这里并不冲突。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  这一步发生在词表的 token 空间里（L20）：grammar 要判断的是"这个 token 接上去，整段文本还合不合语法"。所以它和词表、采样器是紧密咬合的——语法用词表的 token 说话，用采样器的接口干活。理解了这层关系，你就明白为什么这一课紧跟在采样（L21）后面。
</div>
<p>换个角度感受它的价值。没有语法约束时，让模型输出 JSON，你只能在 prompt 里恳求"请只返回合法 JSON、不要多余文字"，然后祈祷它听话——大模型多数时候听，但偶尔会画蛇添足加段解释、漏个引号、把数字写成中文。这种"绝大多数对、偶尔翻车"在生产环境里恰恰最致命，因为你得为那 1% 的翻车写一堆容错。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  语法约束把这件事从"祈祷"变成"保证"。一旦套上 JSON 文法，模型<strong>物理上</strong>就吐不出非法的东西——该是引号的位置只能是引号，该是数字的地方只能是数字。那 1% 的翻车被从根上消灭了，下游代码可以放心地直接解析，不必再写一层防御。这种确定性，正是把大模型接进严肃系统的前提。
</div>
<p>你可能担心：约束这么死，会不会把模型"框傻"了？不会。语法只规定<strong>结构</strong>（哪儿能放什么），不规定<strong>内容</strong>（具体放什么值）。在合法的位置上，模型依然按自己的理解去填——该填用户名就填用户名、该填年龄就填合理的数。结构由你定死，内容仍由模型的智能决定，两者各司其职。</p>

<h2>GBNF 是什么</h2>
<pre class="code"><span class="cm"># 简化自 grammars/json.gbnf</span>
root   ::= object
object ::= <span class="st">"{"</span> ws ( string <span class="st">":"</span> ws value )? <span class="st">"}"</span>
value  ::= object | string | number | <span class="st">"true"</span> | <span class="st">"false"</span>
string ::= <span class="st">"\""</span> [^<span class="st">"</span>]* <span class="st">"\""</span></pre>
<p>那"语法"本身怎么写？用 <strong>GBNF</strong>（GGML BNF），一种类 BNF 的文法描述。它的核心是一条条<strong>规则</strong>：用 <span class="mono">::=</span> 定义"某个名字可以展开成什么"，用 <span class="mono">|</span> 表示"多选一"，用 <span class="mono">[...]</span> 描述一类字符，再配上重复、分组等记号。</p>
<p>上面这段（简化自仓库里的 <span class="mono">grammars/json.gbnf</span>）描述了一个极简 JSON：入口是 <span class="mono">root</span>，它展开成一个 <span class="mono">object</span>；object 是花括号里包着键值对；value 可以是 object、字符串、数字或字面量。规则可以<strong>递归</strong>（object 里又能套 value、value 又能是 object），于是有限的几条规则就能描述无限层嵌套的结构。</p>
<table class="t">
  <tr><th>语法</th><th>含义</th></tr>
  <tr><td>::=</td><td>定义一条规则</td></tr>
  <tr><td>|</td><td>多选一（备选）</td></tr>
  <tr><td>[...]</td><td>字符类（如 [a-z]）</td></tr>
  <tr><td>[^...]</td><td>取反字符类</td></tr>
  <tr><td>* + ?</td><td>重复 0+ / 1+ / 可选</td></tr>
  <tr><td>( )</td><td>分组</td></tr>
  <tr><td>root</td><td>入口规则</td></tr>
</table>
<p>这张表列了 GBNF 最常用的记号。它们组合起来，几乎能描述任何"结构化"的输出格式：JSON、特定语法的代码、固定模板的回答……仓库的 <span class="mono">grammars/</span> 目录里就放着 JSON、国际象棋着法等现成例子，可以直接拿来用或改。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  入口规则约定叫 <span class="mono">root</span>——文法从这里开始展开，就像程序从 main 开始。读一份 GBNF，最好的办法就是从 root 出发，顺着 <span class="mono">::=</span> 一层层往下看每个名字能变成什么，很快就能在脑子里把它"跑"一遍。
</div>
<p>BNF 这套记法其实历史悠久，是描述编程语言文法的经典工具；GBNF 是它的一个轻量方言，专为"约束生成"裁剪定制。如果你见过编程语言的文法定义，会对这套 <span class="mono">::=</span> 规则一见如故；没见过也不要紧，把它当成"一套描述合法字符串长什么样的积木"就行。</p>
<p>写 GBNF 有个实用心法：<strong>从大到小、逐层拆解</strong>。先想清最外层的结构（比如"一个对象"），写成 root；再把它依赖的部分（键、值、空白）各写一条规则；遇到"可以是好几种之一"的就用 <span class="mono">|</span>，遇到"重复若干次"的就用 <span class="mono">*</span>/<span class="mono">+</span>。一层层拆到最底层的字符类，一份文法就成了。仓库里的现成例子是最好的模板。</p>

<h2>语法怎么约束采样</h2>
<pre class="code"><span class="cm"># 伪代码: 掩码 + 推进</span>
<span class="cm"># 采样前: 掩掉非法候选 (llama_grammar_apply_impl)</span>
<span class="kw">for</span> cand <span class="kw">in</span> cur_p:
    <span class="kw">if</span> <span class="kw">not</span> grammar_allows(stacks, cand.id):
        cand.logit = -INFINITY        <span class="cm"># 非法 -&gt; 永不会被选中</span>
<span class="cm"># 选定 token 后: 推进语法状态 (llama_grammar_accept_impl)</span>
grammar.<span class="fn">accept</span>(chosen_token)          <span class="cm"># 沿规则栈往前走一步</span></pre>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>logits</h4><p>这一步模型给出的所有候选 token 分数。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>grammar.apply 掩码</h4><p>按规则栈把不合语法的候选 logit 设成 -inf（含语法未走完时的 EOG）。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>采样选定</h4><p>采样器（L21）只能从剩下的合法候选里选一个 id。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>grammar.accept 推进</h4><p>把选中的 token 喂回，规则栈往前走一格，回到第 1 步循环。</p></div></div>
</div>
<p>把 GBNF 文法变成"采样时的掩码"，靠两个内部函数：<span class="mono">llama_grammar_apply_impl</span>（掩码）和 <span class="mono">llama_grammar_accept_impl</span>（推进）。</p>
<p><span class="mono">apply</span> 这一步：它拿着语法当前的状态（一组<strong>规则栈</strong> <span class="mono">stacks</span>，记着"现在展开到哪、接下来合法的是什么"），逐个检查候选 token——能接上的留着，接不上的把 logit 设成负无穷。EOG（结束符）在语法还没走完时也会被掩掉，免得模型半途而废。</p>
<p><span class="mono">accept</span> 这一步：采样真正选定一个 token 后，把它喂回语法，让规则栈<strong>往前推进</strong>到新状态。下一轮 apply 就基于这个新状态再算一遍合法集。两步交替，像沿着文法的轨道一步步走，每一步都只踩在合法的枕木上。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  内部还细致处理了 <strong>UTF-8</strong>：一个字符可能跨多个字节 token（L20 的字节回退），语法用一个 <span class="mono">partial_utf8</span> 缓冲来拼接半个字符，等拼完整再判断合不合规。这些细节你不用记，但知道"它考虑到了多字节字符"就够了——正是这种周到，让约束在真实多语言文本上也站得住。
</div>
<p>这里值得停下来体会"<strong>状态机</strong>"这个比喻。一份文法被加载后，运行时维护的不是"整段文本"，而是"当前走到文法的哪个位置、接下来允许哪些字符"。每接受一个 token，这个位置就往前挪；它就像一个在文法图上移动的光标，光标所在处决定了下一步的合法集。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  正因为状态是逐步推进的，<strong>同一个 token 在不同位置合不合法是不同的</strong>。比如在 JSON 里，刚写完 <span class="mono">{</span> 时只允许引号（开始一个键）或 <span class="mono">}</span>（空对象），而写完一个完整键值对后又只允许逗号或 <span class="mono">}</span>。grammar 每一步都根据当前状态算出这个"此刻合法集"，再据此掩码。约束不是一成不变的，而是<strong>随上下文动态变化</strong>的。
</div>
<p>拿生成 {"a":1} 这个最小例子走一遍：每到一步，语法只放行合法的那个字符、把其它候选砸成 -inf，选完再推进状态机——逐字符把输出锁在合法的轨道上。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次语法约束</b>：逐字符生成 {"a":1}，每步把非法 token 砸成 -inf、只放行合法的，并推进状态机。</div>
  <svg viewBox="0 0 660 250" width="100%" role="img" aria-label="语法约束状态机示例">
<g font-family="ui-monospace,monospace" font-size="13">
<line x1="91" y1="61" x2="125" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 125 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="110" y="40" text-anchor="middle" fill="#5b6470" font-size="13">{</text>
<line x1="201" y1="61" x2="235" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 235 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="220" y="40" text-anchor="middle" fill="#5b6470" font-size="13">"a"</text>
<line x1="311" y1="61" x2="345" y2="61" stroke="#c2630e" stroke-width="2.6"/>
<path d="M 345 61 l -7 -3.5 v 7 z" fill="#c2630e"/>
<text x="330" y="40" text-anchor="middle" fill="#c2630e" font-size="13">:</text>
<line x1="421" y1="61" x2="455" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 455 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="440" y="40" text-anchor="middle" fill="#5b6470" font-size="13">1</text>
<line x1="531" y1="61" x2="565" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 565 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="550" y="40" text-anchor="middle" fill="#5b6470" font-size="13">}</text>
<rect x="19" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="55" y="66" text-anchor="middle" fill="#1d2129" font-size="12">起始</text>
<rect x="129" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="165" y="66" text-anchor="middle" fill="#1d2129" font-size="12">进对象</text>
<rect x="239" y="46" width="72" height="30" rx="14" fill="#c2630e" stroke="#c2630e"/><text x="275" y="66" text-anchor="middle" fill="#fff" font-size="12">得键</text>
<rect x="349" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="385" y="66" text-anchor="middle" fill="#1d2129" font-size="12">得值</text>
<rect x="459" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="495" y="66" text-anchor="middle" fill="#1d2129" font-size="12">待收尾</text>
<rect x="569" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="605" y="66" text-anchor="middle" fill="#1d2129" font-size="12">完成</text>
<text x="19" y="138" fill="#5b6470" font-size="12">此刻状态 need ':'，候选 token 谁能留下：</text>
<rect x="19" y="150" width="58" height="36" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="48" y="174" text-anchor="middle" fill="#fff" font-weight="700">:</text>
<text x="48" y="202" text-anchor="middle" fill="#c2630e" font-size="11">放行</text>
<rect x="89" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="118" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">,</text>
<line x1="99" y1="168" x2="137" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="118" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<rect x="159" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="188" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">}</text>
<line x1="169" y1="168" x2="207" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="188" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<rect x="229" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="258" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">5</text>
<line x1="239" y1="168" x2="277" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="258" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<text x="313" y="174" fill="#5b6470" font-size="12">&#8594; 推进状态</text>
</g></svg>
</div>
<p>有人会问：每步都遍历几万个候选去判断合不合法，不会很慢吗？实现上做了不少优化——把文法预编译成高效结构、对候选按规则栈快速筛、缓存中间结果等等。多数情况下这点开销相对模型的一次前向（L17）几乎可忽略。所以你尽管放心用语法约束，它换来的可靠性，远大于那一点点代价。</p>
<p>举个落地的例子：很多"让大模型当后端"的应用，都靠语法约束保证它返回能被程序直接吃下的 JSON。你定义好返回结构的文法、挂上 grammar 采样器，模型这一头就成了一个"永远输出合法结构"的可靠组件。没有它，你得在模型和程序之间塞一层解析、纠错、重试的胶水；有了它，那层胶水基本可以省掉。这就是约束带来的实打实的工程价值。</p>

<h2>元素类型与作为采样器</h2>
<div class="cellgroup">
  <div class="cg-cap"><b>llama_gretype</b>：GBNF 规则被编译成的底层元素类型</div>
  <div class="cells"><span class="lab">类型</span><span class="cell">CHAR 字面</span><span class="cell">CHAR_RNG_UPPER 范围</span><span class="cell">CHAR_NOT 取反</span><span class="cell">RULE_REF 引用</span><span class="cell">ALT 备选</span><span class="cell">END 收尾</span></div>
</div>
<p>文法在加载时会被<strong>编译</strong>成一串底层元素，类型由 <span class="mono">enum llama_gretype</span> 定义。你写的每条 <span class="mono">::=</span> 规则，最终都被翻译成这样一串元素，供运行时高效匹配。</p>
<p>这些类型就是 GBNF 记号的"机器码"：<span class="mono">CHAR</span> 是一个字面字符，<span class="mono">CHAR_RNG_UPPER</span> 配合表示一个范围（如 a-z），<span class="mono">CHAR_NOT</span> 是取反类，<span class="mono">RULE_REF</span> 是"引用另一条规则"，<span class="mono">ALT</span> 是备选分隔，<span class="mono">END</span> 收尾。把人写的文法降到这一层，是为了让运行时能快速判断"下一个字符合不合法"。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  那 grammar 怎么接进采样？通过一个<strong>采样器</strong>：<span class="mono">llama_sampler_init_grammar(vocab, 文法串, root)</span> 返回的就是 L21 那套 <span class="mono">llama_sampler</span>——它的 <span class="mono">apply</span> 调掩码、<span class="mono">accept</span> 调推进。换句话说，grammar 本质上就是<strong>一个特殊的采样器</strong>，完美复用了 L21 的接口。
</div>
<p>不过它通常<strong>不混进主采样链</strong>，而是作为独立对象，按一个 <span class="mono">grammar_first</span> 标志决定在链前还是链后单独施加。这是因为约束和"调温度/裁候选"那些塑形操作性质不同，需要灵活安排先后。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  惰性变体 <span class="mono">llama_sampler_init_grammar_lazy</span> 已弃用，改用 <span class="mono">llama_sampler_init_grammar_lazy_patterns</span>。
</div>
<p>为什么要先把人写的文法<strong>编译</strong>成这串底层元素，而不是直接拿原文匹配？因为运行时每生成一个 token 都要判一次合法性，必须快。把文法预先拆成 <span class="mono">CHAR</span>/<span class="mono">RULE_REF</span> 这些规整的元素，运行时就能用简单高效的方式推进和匹配，而不必反复解析原始的文法文本。这是"编译期多花点、运行期省大头"的经典权衡。</p>
<p>再品一下 grammar"就是个采样器"的妙处。L21 把采样设计成一串可插拔的小变换，当时你可能没料到，"语法约束"这种听起来完全不同的东西，居然能<strong>原封不动</strong>地套进同一个 <span class="mono">apply</span>/<span class="mono">accept</span> 接口。这就是好接口的价值：它预留的扩展点，能容纳设计时根本没想到的新玩法。</p>
<p>最后把这一课放回整张图：从 L20 的词表、L21 的采样、L22 的对话模板，到这一课的语法约束，你已经集齐了"控制模型输出"的整套工具——控制<strong>怎么分词</strong>、<strong>怎么选词</strong>、<strong>怎么组织对话</strong>、<strong>怎么约束结构</strong>。下一课 L24 再讲 LoRA，就连"<strong>怎么微调模型行为</strong>"也补上了。第四部分的拼图，只差最后一块。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> grammar 和采样器（L21）到底是什么关系？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>grammar 本质上<strong>就是一个采样器</strong>。<span class="mono">llama_sampler_init_grammar</span> 返回的是 L21 那个 <span class="mono">llama_sampler</span> 结构：它的 <span class="mono">apply</span> 实现成"掩掉非法 token"，<span class="mono">accept</span> 实现成"推进语法状态"。所以它完美套进了 L21 那套统一接口，引擎按一样的方式调度它。</p>
    <p>区别在于它<strong>有状态、且约束力强</strong>。普通采样器（top-k/温度）只是塑形概率，grammar 却能把整批 token 直接判死刑。也因为它有自己的语法状态要维护（走到哪一步了），不像无状态的 top-k 那么随意——这也是它常被单独管理、而非混进主链的原因。</p>
    <p>这种"用同一个接口容纳天差地别的实现"，正是 L21 责任链设计的威力：温度、惩罚、语法约束长相一致，却能做截然不同的事。读懂了这点，你就明白为什么往采样里加一种全新的约束，几乎不用动引擎主干。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 惰性 / 触发语法（lazy）是干嘛的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>有时你<strong>不想一上来就约束</strong>，而是等某个信号出现后才开始。最典型的是工具调用：让模型先自由地说话，一旦它说出某个触发词（比如表示"我要调用工具了"的标记），再切到严格的 JSON 约束，逼它把参数写成合法格式。</p>
    <p>这就是<strong>惰性语法</strong>：<span class="mono">llama_grammar</span> 里的 <span class="mono">lazy</span>/<span class="mono">awaiting_trigger</span>/<span class="mono">trigger_patterns</span> 字段实现这点——约束先"待命"，把输出缓冲着，直到匹配上触发条件才真正生效、开始掩码。对应的采样器是 <span class="mono">llama_sampler_init_grammar_lazy_patterns</span>。</p>
    <p>为什么有用？因为现实任务常是"<strong>先自由、后严格</strong>"：模型先用自然语言思考/回应，需要结构化输出时才上约束。惰性语法让你不必从第一个 token 就锁死格式，既保留了模型的灵活，又在关键处保证了结构。这是把"约束"和"自由"按需切换的巧妙设计。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 为什么 token 级掩码胜过"事后校验"？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>事后校验是"生成完整段、再用正则/解析器检查合不合格，不合格就重来"。问题很明显：一是<strong>慢</strong>，一次不合格就得整段重生成，可能反复失败；二是<strong>不保证收敛</strong>，模型可能怎么试都凑不出合法的，陷入死循环。</p>
    <p>token 级掩码把关口前移到<strong>每一步</strong>：每选一个 token 都保证此刻合法，于是生成出来的<strong>必然</strong>是合法的，一次成型，无需重试。它用"每步一点点约束"换来了"整体永远正确"，既快又稳。</p>
    <p>这背后是个通用的工程智慧：<strong>把错误挡在产生之前，远胜于产生之后再补救</strong>。你在 L14 见过加载期的一致性检查（早查早安心）、在编译型语言里见过类型检查，都是同一个道理。grammar 把这套思路用在了生成上。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>GBNF 语法约束 = 在每步采样时把<strong>不合语法的 token 掩成负无穷</strong>，模型只能选合法的，输出<strong>必然合法</strong>。</li>
    <li>GBNF 规则：<span class="mono">::=</span> 定义、<span class="mono">|</span> 备选、<span class="mono">[...]</span> 字符类、<span class="mono">* + ?</span> 重复、<span class="mono">root</span> 入口；可递归。</li>
    <li>两个动作：<span class="mono">apply</span>（掩码非法候选）+ <span class="mono">accept</span>（推进规则栈）；文法编译成 <span class="mono">enum llama_gretype</span> 元素。</li>
    <li>接入采样：<span class="mono">llama_sampler_init_grammar</span>（grammar 就是个特殊采样器），通常按 <span class="mono">grammar_first</span> 在主链外施加；惰性用 <span class="mono">..._grammar_lazy_patterns</span>（旧 <span class="mono">_grammar_lazy</span> 弃用）。</li>
    <li>token 级掩码 &gt; 事后校验：每步都合法、一次成型，不会生成到一半才发现非法。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  语法约束把"<strong>结构正确</strong>"从"事后祈祷"变成"<strong>生成时保证</strong>"——在 token 级别就堵死所有非法路径。它把"约束"漂亮地装进了 L21 的采样器接口：grammar 不过是又一个 <span class="mono">apply</span>/<span class="mono">accept</span> 的实现，却让"自由生成"和"严格格式"在同一套机制里和谐共处。这正是 llama.cpp 让大模型<strong>可靠输出结构化数据</strong>的钥匙——也是把它接进真实软件系统的关键一步。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
By now the model can chat (L20-L22). But many real scenarios want more than "talks fine" - they want <strong>strictly correct format</strong>: calling an API needs valid JSON, filling a form needs the required fields, extracting info needs a fixed structure. A model generates by probability and inevitably wanders off sometimes - this lesson's <strong>GBNF grammar constraint</strong> puts a "guardrail" on generation, making it <strong>impossible</strong> to produce format-invalid output.
</p>
<p style="color:var(--muted);margin-top:.4rem">Its idea is clever: not generate-then-check-and-retry-if-bad (slow and unreliable), but at <strong>each sampling step</strong> strike out every "token the grammar disallows right now", so the model can only move forward on the legal path. So however much the model wants to wander, it cannot step past the grammar's boundary - the result is <strong>always valid</strong>, right the first time.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A grammar constraint is like a <strong>dropdown on a form</strong>: you cannot type freely, only pick from the given options. Before each sampling step, the grammar greys out (sets to negative infinity) "the tokens that should not appear right now", so the model can only pick one of the remaining legal options. One dropdown per step, strung together, guarantees the whole output strictly matches the format you defined.
</div>

<h2>Why a grammar is needed</h2>
<div class="flow">
  <div class="node"><div class="nt">logits</div><div class="nd">all candidates</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">grammar.apply</div><div class="nd">illegal -&gt; -inf</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">sampling L21</div><div class="nd">pick from legal only</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">legal token</div><div class="nd">never out of bounds</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">grammar.accept</div><div class="nd">advance grammar</div></div>
</div>
<p>First see where it sits in the sampling pipeline. L21 covered that sampling prunes and shapes a row of logits and finally picks one token. The grammar inserts a <strong>mask</strong> stage into this pipeline: before picking, slam to negative infinity the logit of every candidate "the grammar disallows right now", so their probability becomes 0, impossible to be chosen.</p>
<p>After a legal token is picked, one more step: <strong>advance</strong> the grammar state. The grammar is like a state machine; the token just allowed moved it one notch forward, and which tokens are legal next updates accordingly. Mask then advance, token by token, locks the whole output firmly onto the grammar's track.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Why must it be done at the token level, not checked afterward? Imagine generating a JSON and an illegal character pops up halfway - an after-the-fact check can only scrap the whole thing and retry, slow and possibly failing repeatedly. Token-level masking guarantees <strong>every step is legal</strong>, never giving "write it wrong" a chance, done in one go. This is the decisive win of "constrain up front" over "patch afterward".
</div>
<p>There is a subtle bonus too: because illegal tokens are set to negative infinity with probability zeroed, the remaining legal tokens re-normalize their probabilities. That is, the constraint not only "forbids the illegal" but lets the model pick <strong>within the legal range</strong> by its own preference - keeping the rules while preserving the model's judgment as much as possible. Strong constraint and model intelligence do not conflict here.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  This step happens in the vocab's token space (L20): the grammar must judge "with this token appended, is the whole text still legal". So it meshes tightly with the vocab and the sampler - the grammar speaks in the vocab's tokens and works through the sampler's interface. Understand this relationship and you see why this lesson follows sampling (L21) closely.
</div>
<p>Feel its value from another angle. Without a grammar constraint, to get JSON out of the model you can only beg in the prompt "please return only valid JSON, no extra text" and pray it obeys - a large model mostly does, but occasionally gilds the lily with an explanation, drops a quote, or writes a number as words. This "mostly right, occasionally derailed" is precisely the most lethal in production, because you must write a pile of error-handling for that 1% derailment.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  A grammar constraint turns this from "praying" into "guaranteeing". Once a JSON grammar is on, the model <strong>physically</strong> cannot emit anything illegal - where a quote belongs only a quote can go, where a number belongs only a number can. That 1% derailment is eliminated at the root, and downstream code can parse directly without a defensive layer. This certainty is the prerequisite for wiring a large model into serious systems.
</div>
<p>You might worry: with such a rigid constraint, does it "frame the model into stupidity"? No. A grammar dictates only <strong>structure</strong> (what can go where), not <strong>content</strong> (the actual values). At a legal position, the model still fills by its own understanding - a username where a username goes, a sensible number where an age goes. You fix the structure, the model's intelligence still decides the content, each to its job.</p>

<h2>What GBNF is</h2>
<pre class="code"><span class="cm"># simplified from grammars/json.gbnf</span>
root   ::= object
object ::= <span class="st">"{"</span> ws ( string <span class="st">":"</span> ws value )? <span class="st">"}"</span>
value  ::= object | string | number | <span class="st">"true"</span> | <span class="st">"false"</span>
string ::= <span class="st">"\""</span> [^<span class="st">"</span>]* <span class="st">"\""</span></pre>
<p>So how is "the grammar" itself written? In <strong>GBNF</strong> (GGML BNF), a BNF-like grammar description. Its core is <strong>rules</strong>: <span class="mono">::=</span> defines "what a name can expand into", <span class="mono">|</span> means "one of several", <span class="mono">[...]</span> describes a class of characters, plus repetition, grouping and other notations.</p>
<p>The snippet above (simplified from the repo's <span class="mono">grammars/json.gbnf</span>) describes a minimal JSON: the entry is <span class="mono">root</span>, expanding into an <span class="mono">object</span>; an object is key-value pairs inside braces; a value can be an object, string, number or literal. Rules can be <strong>recursive</strong> (an object can nest a value, a value can be an object), so a handful of rules describe infinitely nested structures.</p>
<table class="t">
  <tr><th>Syntax</th><th>Meaning</th></tr>
  <tr><td>::=</td><td>define a rule</td></tr>
  <tr><td>|</td><td>one of several (alternation)</td></tr>
  <tr><td>[...]</td><td>character class (e.g. [a-z])</td></tr>
  <tr><td>[^...]</td><td>negated character class</td></tr>
  <tr><td>* + ?</td><td>repeat 0+ / 1+ / optional</td></tr>
  <tr><td>( )</td><td>grouping</td></tr>
  <tr><td>root</td><td>entry rule</td></tr>
</table>
<p>This table lists GBNF's most common notations. Combined, they can describe almost any "structured" output format: JSON, code in a particular syntax, fixed-template answers... The repo's <span class="mono">grammars/</span> directory ships ready examples like JSON and chess moves, to use directly or adapt.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  The entry rule is conventionally called <span class="mono">root</span> - the grammar starts expanding here, like a program starts at main. The best way to read a GBNF is to start from root and follow <span class="mono">::=</span> down level by level, seeing what each name becomes; you can quickly "run" it in your head.
</div>
<p>BNF as a notation is in fact long-standing, a classic tool for describing programming-language grammars; GBNF is a lightweight dialect of it, trimmed and tailored for "constraining generation". If you have seen a programming language's grammar definition, you will take to these <span class="mono">::=</span> rules at first sight; if not, no matter - just treat it as "a set of blocks describing what a legal string looks like".</p>
<p>There is a practical knack to writing GBNF: <strong>top-down, decompose layer by layer</strong>. First think out the outermost structure (say "an object"), written as root; then write a rule for each part it depends on (key, value, whitespace); use <span class="mono">|</span> for "one of several", <span class="mono">*</span>/<span class="mono">+</span> for "repeat some times". Decompose down to the bottom character classes and a grammar is done. The ready examples in the repo are the best templates.</p>

<h2>How the grammar constrains sampling</h2>
<pre class="code"><span class="cm"># pseudocode: mask + advance</span>
<span class="cm"># before sampling: mask out illegal candidates (llama_grammar_apply_impl)</span>
<span class="kw">for</span> cand <span class="kw">in</span> cur_p:
    <span class="kw">if</span> <span class="kw">not</span> grammar_allows(stacks, cand.id):
        cand.logit = -INFINITY        <span class="cm"># illegal -&gt; can never be chosen</span>
<span class="cm"># after a token is chosen: advance the grammar state (llama_grammar_accept_impl)</span>
grammar.<span class="fn">accept</span>(chosen_token)          <span class="cm"># step forward along the rule stacks</span></pre>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>logits</h4><p>All candidate token scores the model gives this step.</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>grammar.apply mask</h4><p>By the rule stacks, set illegal candidates' logit to -inf (incl. EOG while the grammar is not yet complete).</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>sampling picks</h4><p>The sampler (L21) can pick an id only from the remaining legal candidates.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>grammar.accept advance</h4><p>Feed the chosen token back, the rule stacks step forward one notch, loop to step 1.</p></div></div>
</div>
<p>Turning a GBNF grammar into "a mask at sampling time" relies on two internal functions: <span class="mono">llama_grammar_apply_impl</span> (mask) and <span class="mono">llama_grammar_accept_impl</span> (advance).</p>
<p>The <span class="mono">apply</span> step: holding the grammar's current state (a set of <strong>rule stacks</strong>, the <span class="mono">stacks</span> field, recording "where the expansion is now, what is legal next"), it checks each candidate token - keep those that fit, set those that do not to negative infinity. EOG (the terminator) is also masked while the grammar is not yet complete, lest the model quit halfway.</p>
<p>The <span class="mono">accept</span> step: once sampling actually picks a token, feed it back to the grammar so the rule stacks <strong>advance</strong> to a new state. The next apply then recomputes the legal set from this new state. The two alternate, like walking along the grammar's track, each step stepping only on a legal sleeper.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Internally it also carefully handles <strong>UTF-8</strong>: one character may span several byte tokens (L20's byte fallback), so the grammar uses a <span class="mono">partial_utf8</span> buffer to stitch a half character and judges legality once it is whole. You need not memorize these details, but knowing "it accounts for multi-byte characters" is enough - this thoroughness is what lets the constraint hold up on real multilingual text.
</div>
<p>It is worth pausing here to savor the "<strong>state machine</strong>" metaphor. Once a grammar is loaded, what the runtime maintains is not "the whole text" but "where in the grammar it is now, which characters are allowed next". Each accepted token nudges this position forward; it is like a cursor moving over the grammar graph, and where the cursor sits decides the legal set for the next step.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Because the state advances step by step, <strong>the same token can be legal or not at different positions</strong>. In JSON, for instance, right after <span class="mono">{</span> only a quote (starting a key) or <span class="mono">}</span> (empty object) is allowed, while after a complete key-value pair only a comma or <span class="mono">}</span> is. The grammar computes this "legal set right now" from the current state each step and masks accordingly. The constraint is not fixed but <strong>changes dynamically with context</strong>.
</div>
<p>Walk the smallest example, generating {"a":1}, step by step: at each step the grammar lets through only the legal next character, slams every other candidate to -inf, then advances the state machine - locking the output onto the legal track character by character.</p>
<div class="trace">
  <div class="tcap"><b>Tracing one grammar constraint</b>: generating {"a":1} char by char - each step masks illegal tokens to -inf, allows only the legal one, and advances the state machine.</div>
  <svg viewBox="0 0 660 250" width="100%" role="img" aria-label="grammar constraint state machine">
<g font-family="ui-monospace,monospace" font-size="13">
<line x1="91" y1="61" x2="125" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 125 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="110" y="40" text-anchor="middle" fill="#5b6470" font-size="13">{</text>
<line x1="201" y1="61" x2="235" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 235 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="220" y="40" text-anchor="middle" fill="#5b6470" font-size="13">"a"</text>
<line x1="311" y1="61" x2="345" y2="61" stroke="#c2630e" stroke-width="2.6"/>
<path d="M 345 61 l -7 -3.5 v 7 z" fill="#c2630e"/>
<text x="330" y="40" text-anchor="middle" fill="#c2630e" font-size="13">:</text>
<line x1="421" y1="61" x2="455" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 455 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="440" y="40" text-anchor="middle" fill="#5b6470" font-size="13">1</text>
<line x1="531" y1="61" x2="565" y2="61" stroke="#9aa6b2" stroke-width="1.4"/>
<path d="M 565 61 l -7 -3.5 v 7 z" fill="#9aa6b2"/>
<text x="550" y="40" text-anchor="middle" fill="#5b6470" font-size="13">}</text>
<rect x="19" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="55" y="66" text-anchor="middle" fill="#1d2129" font-size="12">start</text>
<rect x="129" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="165" y="66" text-anchor="middle" fill="#1d2129" font-size="12">obj</text>
<rect x="239" y="46" width="72" height="30" rx="14" fill="#c2630e" stroke="#c2630e"/><text x="275" y="66" text-anchor="middle" fill="#fff" font-size="12">key</text>
<rect x="349" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="385" y="66" text-anchor="middle" fill="#1d2129" font-size="12">val</text>
<rect x="459" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="495" y="66" text-anchor="middle" fill="#1d2129" font-size="12">end</text>
<rect x="569" y="46" width="72" height="30" rx="14" fill="#ffffff" stroke="#cdd5df"/><text x="605" y="66" text-anchor="middle" fill="#1d2129" font-size="12">done</text>
<text x="19" y="138" fill="#5b6470" font-size="12">state need ':' - which candidate tokens survive:</text>
<rect x="19" y="150" width="58" height="36" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="48" y="174" text-anchor="middle" fill="#fff" font-weight="700">:</text>
<text x="48" y="202" text-anchor="middle" fill="#c2630e" font-size="11">keep</text>
<rect x="89" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="118" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">,</text>
<line x1="99" y1="168" x2="137" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="118" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<rect x="159" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="188" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">}</text>
<line x1="169" y1="168" x2="207" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="188" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<rect x="229" y="150" width="58" height="36" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="258" y="174" text-anchor="middle" fill="#5b6470" font-weight="700">5</text>
<line x1="239" y1="168" x2="277" y2="168" stroke="#9aa6b2" stroke-width="2"/>
<text x="258" y="202" text-anchor="middle" fill="#9aa6b2" font-size="11">-inf</text>
<text x="313" y="174" fill="#5b6470" font-size="12">&#8594; advance</text>
</g></svg>
</div>
<p>One might ask: scanning tens of thousands of candidates each step to judge legality, is that not slow? The implementation does plenty of optimization - precompiling the grammar into efficient structures, quickly filtering candidates by the rule stacks, caching intermediate results, and so on. In most cases this cost is nearly negligible against the model's one forward pass (L17). So use grammar constraints freely; the reliability they buy far outweighs the small price.</p>
<p>A concrete grounded example: many "let the large model be a backend" applications rely on grammar constraints to guarantee it returns JSON a program can consume directly. Define the grammar for the return structure, attach the grammar sampler, and the model end becomes a reliable component that "always outputs a legal structure". Without it, you must stuff a layer of parse, correct, and retry glue between the model and the program; with it, that glue layer can largely be dropped. This is the concrete engineering value a constraint brings.</p>

<h2>Element types and being a sampler</h2>
<div class="cellgroup">
  <div class="cg-cap"><b>llama_gretype</b>: the low-level element types a GBNF rule compiles into</div>
  <div class="cells"><span class="lab">type</span><span class="cell">CHAR literal</span><span class="cell">CHAR_RNG_UPPER range</span><span class="cell">CHAR_NOT negate</span><span class="cell">RULE_REF ref</span><span class="cell">ALT alternate</span><span class="cell">END close</span></div>
</div>
<p>A grammar is <strong>compiled</strong> at load time into a string of low-level elements, whose types are defined by <span class="mono">enum llama_gretype</span>. Every <span class="mono">::=</span> rule you write is ultimately translated into such a string of elements for the runtime to match efficiently.</p>
<p>These types are GBNF notation's "machine code": <span class="mono">CHAR</span> is a literal character, <span class="mono">CHAR_RNG_UPPER</span> pairs up to express a range (like a-z), <span class="mono">CHAR_NOT</span> is a negated class, <span class="mono">RULE_REF</span> is "reference another rule", <span class="mono">ALT</span> is an alternation separator, <span class="mono">END</span> closes. Lowering the human-written grammar to this level lets the runtime quickly decide "is the next character legal".</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  So how does the grammar plug into sampling? Through a <strong>sampler</strong>: <span class="mono">llama_sampler_init_grammar(vocab, grammar_str, root)</span> returns exactly L21's <span class="mono">llama_sampler</span> - its <span class="mono">apply</span> calls the mask, its <span class="mono">accept</span> calls the advance. In other words, the grammar is essentially <strong>a special sampler</strong>, perfectly reusing L21's interface.
</div>
<p>But it usually does <strong>not</strong> mix into the main sampler chain; it is a separate object applied before or after the chain per a <span class="mono">grammar_first</span> flag. This is because constraint differs in nature from shaping ops like "tune temperature / prune candidates", needing flexible ordering.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  The lazy variant <span class="mono">llama_sampler_init_grammar_lazy</span> is deprecated, use <span class="mono">llama_sampler_init_grammar_lazy_patterns</span>.
</div>
<p>Why compile the human-written grammar into this string of low-level elements first, rather than matching the raw text directly? Because the runtime judges legality once per generated token and must be fast. Pre-splitting the grammar into tidy elements like <span class="mono">CHAR</span>/<span class="mono">RULE_REF</span> lets the runtime advance and match in a simple, efficient way, without re-parsing the raw grammar text. This is the classic "spend a bit at compile time, save the bulk at run time" trade-off.</p>
<p>Savor again the elegance of the grammar "being a sampler". L21 designed sampling as a string of pluggable small transforms; at the time you may not have foreseen that "grammar constraint", something that sounds utterly different, could slot <strong>unchanged</strong> into the same <span class="mono">apply</span>/<span class="mono">accept</span> interface. That is the value of a good interface: the extension point it reserves can hold new tricks never imagined at design time.</p>
<p>Finally, put this lesson back into the whole picture: from L20's vocab, L21's sampling, L22's chat templates, to this lesson's grammar constraint, you have now gathered the full toolkit for "controlling the model's output" - controlling <strong>how to tokenize</strong>, <strong>how to pick words</strong>, <strong>how to organize the conversation</strong>, <strong>how to constrain the structure</strong>. Next lesson L24 covers LoRA, adding even "<strong>how to fine-tune the model's behavior</strong>". Only the last piece of Part 4's puzzle remains.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What is the relationship between grammar and the sampler (L21)? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>A grammar essentially <strong>is a sampler</strong>. <span class="mono">llama_sampler_init_grammar</span> returns that L21 <span class="mono">llama_sampler</span> struct: its <span class="mono">apply</span> is implemented as "mask out illegal tokens", its <span class="mono">accept</span> as "advance the grammar state". So it slots perfectly into L21's unified interface, and the engine schedules it the same way.</p>
    <p>The difference is it is <strong>stateful and forceful</strong>. An ordinary sampler (top-k/temperature) only shapes probabilities; the grammar can sentence whole batches of tokens to death. And because it has its own grammar state to maintain (which step it is at), it is not as casual as stateless top-k - which is also why it is often managed separately rather than mixed into the main chain.</p>
    <p>This "one interface holding wildly different implementations" is exactly the power of L21's chain-of-responsibility design: temperature, penalties, grammar constraint look alike yet do utterly different things. Grasp this and you see why adding a brand-new constraint to sampling barely touches the engine trunk.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What are lazy / triggered grammars for? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Sometimes you <strong>do not want to constrain from the start</strong>, but only begin after some signal appears. The classic case is tool calling: let the model speak freely first, and once it emits a trigger word (say a marker meaning "I am about to call a tool"), switch to a strict JSON constraint, forcing it to write the arguments in valid format.</p>
    <p>That is the <strong>lazy grammar</strong>: the <span class="mono">lazy</span>/<span class="mono">awaiting_trigger</span>/<span class="mono">trigger_patterns</span> fields in <span class="mono">llama_grammar</span> implement it - the constraint first "stands by", buffering output, until the trigger condition matches, then truly takes effect and starts masking. The matching sampler is <span class="mono">llama_sampler_init_grammar_lazy_patterns</span>.</p>
    <p>Why useful? Because real tasks are often "<strong>free first, strict later</strong>": the model thinks/responds in natural language first, and only constrains when structured output is needed. A lazy grammar lets you not lock the format from the first token, keeping the model's flexibility while guaranteeing structure where it counts. It is a clever design for switching "constraint" and "freedom" on demand.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Why does token-level masking beat "after-the-fact checking"? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>After-the-fact checking is "generate the whole thing, then check legality with a regex/parser, and retry if bad". The problems are obvious: one, it is <strong>slow</strong> - one failure means regenerating the whole thing, possibly failing repeatedly; two, it does <strong>not guarantee convergence</strong> - the model may never stumble onto a legal one, stuck in a loop.</p>
    <p>Token-level masking moves the gate forward to <strong>every step</strong>: every chosen token is guaranteed legal at that moment, so what is generated is <strong>necessarily</strong> legal, done in one go, no retry needed. It trades "a little constraint each step" for "always correct overall" - both fast and steady.</p>
    <p>Behind this is a general engineering wisdom: <strong>blocking an error before it arises far beats patching it afterward</strong>. You saw load-time consistency checks in L14 (check early, rest easy), and type checks in compiled languages - all the same idea. The grammar applies this to generation.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>GBNF grammar constraint = at each sampling step, <strong>mask illegal tokens to negative infinity</strong>, so the model can only pick legal ones and the output is <strong>necessarily valid</strong>.</li>
    <li>GBNF rules: <span class="mono">::=</span> define, <span class="mono">|</span> alternation, <span class="mono">[...]</span> char class, <span class="mono">* + ?</span> repetition, <span class="mono">root</span> entry; can recurse.</li>
    <li>Two actions: <span class="mono">apply</span> (mask illegal candidates) + <span class="mono">accept</span> (advance the rule stacks); a grammar compiles into <span class="mono">enum llama_gretype</span> elements.</li>
    <li>Into sampling: <span class="mono">llama_sampler_init_grammar</span> (the grammar is a special sampler), usually applied outside the main chain per <span class="mono">grammar_first</span>; lazy uses <span class="mono">..._grammar_lazy_patterns</span> (old <span class="mono">_grammar_lazy</span> deprecated).</li>
    <li>Token-level masking &gt; after-the-fact checking: every step legal, done in one go, never finding illegality halfway.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  A grammar constraint turns "<strong>structural correctness</strong>" from "praying afterward" into "<strong>guaranteed at generation</strong>" - blocking every illegal path right at the token level. It packs "constraint" beautifully into L21's sampler interface: a grammar is just another <span class="mono">apply</span>/<span class="mono">accept</span> implementation, yet it lets "free generation" and "strict format" coexist harmoniously in one mechanism. This is exactly llama.cpp's key to making a large model <strong>reliably output structured data</strong> - and a crucial step to wiring it into real software systems.
</div>
""",
}

LESSON_24 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
第四部分一路走来，你已经能让模型加载、推理、分词、采样、套对话格式、约束输出（L14-L23）。最后一块拼图：如果想让模型<strong>学会新风格、新任务</strong>呢？全量微调要重训并存下整套几十 GB 的权重，又贵又笨。这一课讲 <strong>LoRA</strong>——一种轻量得多的办法，用两个小矩阵给权重打个"补丁"，几 MB 就能改变模型行为。
</p>
<p style="color:var(--muted);margin-top:.4rem">LoRA 的精髓是<strong>低秩</strong>：它不动原权重，只学一个<strong>低秩增量</strong>（两个小矩阵 A、B 相乘），加到原权重的输出上。因为秩很低，这两个矩阵小得可怜（适配器常只有几 MB），却能逼近全量微调的效果。更妙的是，它即插即用——想要某种风格就挂上对应适配器，不想要随时卸下，还能几个叠着用。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  LoRA 像给镜头套<strong>滤镜</strong>：原镜头（基础权重）一点不动，套上一片轻巧的滤镜（A、B 组成的低秩增量），成像风格就变了；不喜欢随时摘下，也能叠加几片。而<strong>控制向量</strong>则像一个调色旋钮，沿某个固定方向整体平移画面色调。两者都不动底片，只在出片时做手脚。
</div>

<h2>为什么需要 LoRA</h2>
<div class="cols">
  <div class="col"><h4>全量微调</h4><p>更新<strong>所有</strong>权重，存一整套（几十 GB），每个任务一份。又贵又笨。</p></div>
  <div class="col"><h4>LoRA</h4><p>冻结原权重 W，只学两个<strong>小</strong>矩阵 A、B。适配器往往几 MB，可叠加、可卸下。</p></div>
</div>
<p>先想清楚全量微调的痛点。一个几十亿参数的模型，要让它适配一个新任务，传统做法是继续训练、更新它的<strong>所有</strong>权重，然后把这一整套新权重存下来。问题是：每个任务都得存一份几十 GB 的模型，训练也要很大显存——又贵、又占地方、又难分享。</p>
<p>LoRA 换了个思路：<strong>冻结</strong>原权重一个字节都不改，只在旁边学两个小矩阵 A、B。要用的时候，把 A、B 算出的增量临时加到原权重上即可。于是你存的、传的、加载的"适配器"，就只有 A、B 这两个小矩阵，常常只有几 MB——和几十 GB 的全量微调一比，省了好几个数量级。</p>
<p>这背后的关键假设是：微调给权重带来的改变，往往集中在一个<strong>低维子空间</strong>里。换句话说，"让模型学会某个新任务"所需的调整，没那么多自由度，用一个低秩矩阵就能很好地近似。正是这个洞察，让"只学两个小矩阵"成为可能，且效果出奇地好。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  实际收益是实打实的：一个基础模型 + 一堆几 MB 的 LoRA，就能变出无数"专精版本"——写代码的、扮角色的、特定领域的，随用随挂。基础模型只读、只存一份，差异全在那些小适配器里。这种"一份底座、多个补丁"的格局，正是 LoRA 流行的根本原因。
</div>
<p>打个更接地气的比方。全量微调像是为了改一句话，把整本书重新印一遍；LoRA 则像在原书上贴几张便签——书没动，便签却足以表达你的修改。要换一种修改，撕掉便签换一批即可，原书永远是那一本。这种"原件不动、改动外挂"的思路，是 LoRA 一切便利的源头。</p>
<p>它带来的协作红利也很大。社区里，大家共享的不再是几十 GB 的整模型，而是几 MB 的适配器——下载快、存储省、还能像插件一样自由组合。一个流行的基础模型周围，往往围着成百上千个各显神通的 LoRA，这种繁荣正是"轻量、可分享"换来的。</p>
<p>当然 LoRA 不是万能的。它擅长"在已有能力上做风格化、领域化的调整"，但要让模型学会一项它<strong>完全没有</strong>的全新本领，低秩增量的表达力可能就不够，那时还得靠更重的训练。明白它的边界，才能用在刀刃上——多数"调性、格式、领域"层面的需求，LoRA 都能漂亮地接住。</p>

<h2>LoRA 数学</h2>
<div class="flow">
  <div class="node"><div class="nt">x</div><div class="nd">输入</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">A</div><div class="nd">降到秩 r</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">B</div><div class="nd">升回原维</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">x scale</div><div class="nd">缩放强度</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">+ W·x</div><div class="nd">加到基础输出</div></div>
</div>
<p>具体怎么算？设原权重是 W、输入是 x。基础输出就是 W·x（一次普通的矩阵乘）。LoRA 在它旁边加一条<strong>低秩支路</strong>：先用 A 把 x 降到一个很低的维度（秩 r），再用 B 升回原来的维度，得到 B·(A·x)；乘上一个缩放系数 scale，加到 W·x 上。最终输出 = W·x + scale·B·A·x。</p>
<p>把这条公式按"维度"画出来，低秩瓶颈就一目了然：主干 W·x 维度不变；旁路先被 A 压到很窄的秩 r，再被 B 升回来，乘 scale 加回主干。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次 LoRA 前向</b>：主干 W·x 不动，旁路把 x 压到低秩 r 再升回来、乘 scale 加回去（维度为示意）。</div>
  <svg viewBox="0 0 660 226" width="100%" role="img" aria-label="LoRA 前向示例：低秩旁路">
<g font-family="ui-monospace,monospace">
<text x="150" y="20" fill="#5b6470" font-size="11">主干 W&#183;x（冻结，不更新）</text>
<text x="150" y="206" fill="#5b6470" font-size="11">旁路：把 x 压到低秩 r 再升回来</text>
<rect x="20" y="82" width="46" height="60" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="43" y="117" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">x</text><text x="43" y="155" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
<line x1="66" y1="100" x2="147" y2="67" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 150 66 L 144 73 L 141 65 z" fill="#9aa6b2"/>
<rect x="150" y="46" width="60" height="40" rx="5" fill="#e7edf5" stroke="#2563eb"/><text x="180" y="71" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">W</text><text x="180" y="99" text-anchor="middle" fill="#5b6470" font-size="10">冻结</text>
<line x1="210" y1="66" x2="247" y2="72" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 250 72 L 241 75 L 243 67 z" fill="#9aa6b2"/>
<rect x="250" y="46" width="58" height="60" rx="5" fill="#e7edf5" stroke="#2563eb"/><text x="279" y="81" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">W&#183;x</text><text x="279" y="119" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
<line x1="66" y1="124" x2="123" y2="149" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 126 150 L 117 150 L 120 143 z" fill="#9aa6b2"/>
<rect x="126" y="150" width="46" height="34" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="149" y="172" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">A</text><text x="149" y="197" text-anchor="middle" fill="#5b6470" font-size="10">4-&gt;r</text>
<line x1="172" y1="167" x2="203" y2="175" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 206 176 L 197 178 L 199 170 z" fill="#9aa6b2"/>
<rect x="206" y="168" width="44" height="18" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="228" y="182" text-anchor="middle" fill="#fff" font-weight="700" font-size="12">r=1</text>
<line x1="250" y1="176" x2="283" y2="168" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 286 167 L 279 173 L 277 165 z" fill="#9aa6b2"/>
<rect x="286" y="150" width="46" height="34" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="309" y="172" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">B</text><text x="309" y="197" text-anchor="middle" fill="#5b6470" font-size="10">r-&gt;4</text>
<line x1="332" y1="160" x2="370" y2="134" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 372 132 L 368 140 L 363 133 z" fill="#9aa6b2"/>
<rect x="372" y="102" width="56" height="60" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="400" y="137" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">[4]</text>
<line x1="308" y1="76" x2="467" y2="98" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 470 98 L 462 101 L 463 93 z" fill="#9aa6b2"/>
<line x1="428" y1="132" x2="467" y2="109" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 470 108 L 465 115 L 461 108 z" fill="#9aa6b2"/>
<text x="452" y="150" text-anchor="middle" fill="#c2630e" font-size="10">&#215;scale</text>
<rect x="470" y="84" width="40" height="40" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="490" y="109" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="18">+</text>
<line x1="510" y1="104" x2="553" y2="104" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 556 104 L 548 108 L 548 100 z" fill="#9aa6b2"/>
<rect x="556" y="74" width="58" height="60" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="585" y="109" text-anchor="middle" fill="#fff" font-weight="700" font-size="14">y</text><text x="585" y="147" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
</g></svg>
</div>
<pre class="code"><span class="cm">// 简化自 src/llama-graph.cpp build_lora_mm</span>
res = <span class="fn">ggml_mul_mat</span>(w, cur);                  <span class="cm">// 基础权重输出 W·x</span>
<span class="kw">for</span> (lora : active_adapters) {
    ab = <span class="fn">ggml_mul_mat</span>(b, <span class="fn">ggml_mul_mat</span>(a, cur)); <span class="cm">// 低秩两步 B·(A·x)</span>
    ab  = <span class="fn">ggml_scale</span>(ab, scale);             <span class="cm">// scale = alpha/rank * 用户比例</span>
    res = <span class="fn">ggml_add</span>(res, ab);                 <span class="cm">// 叠加增量</span>
}</pre>
<p>上面这段（简化自 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">build_lora_mm</span>）就是它在建图（L16）时干的事：先算基础的 <span class="mono">mul_mat(w, cur)</span>，再对每个生效的适配器，算 <span class="mono">B·(A·cur)</span>、乘 scale、加回去。注意这一切发生在<strong>计算图</strong>里——增量是临时算出来叠加的，并没有真去改 W 那几个 GB 的权重。</p>
<p>那个 scale 也有讲究：它由适配器的 <span class="mono">alpha</span> 和秩 r 算出（大致是 <span class="mono">alpha/rank</span>，再乘上用户给的比例）。这个比例就是你挂载时能调的"<strong>强度</strong>"旋钮——调大，适配器的影响更强；调小，更接近原模型。把强度做成可调，让你能在"原汁原味"和"完全变身"之间平滑过渡。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  为什么是 A 降维、B 升维这<strong>两步</strong>，而不是直接学一个同样大小的增量矩阵？因为直接学一个 d×d 的满秩矩阵，参数量和原权重一样大，就失去意义了。拆成 d×r 和 r×d 两个瘦长矩阵（r 远小于 d），参数量从 d×d 降到 2dr——这正是"低秩"省参数的数学本质。
</div>
<p>再把"秩"这个词说透一点。一个矩阵的秩，粗略地说就是它"真正独立的方向"有多少。满秩意味着各个方向都用上了，信息量最大但也最占参数；低秩则是说"其实只用了少数几个方向就够描述这次改动"。LoRA 赌的就是：适配一个任务所需的改动，本质上是低秩的，于是用 r 个方向（A、B 的中间维度就是 r）足以近似。</p>
<p>还有个常被忽略的细节：A、B 的初始化是不对称的。通常 B 初始化为全零、A 随机初始化，于是训练刚开始时增量 B·A 为零——也就是说，挂上一个<strong>没训练过</strong>的 LoRA，对模型毫无影响，和不挂一样。训练过程才慢慢让这个增量长出有用的方向。这个"从零开始、平滑加入"的设计，让 LoRA 训练既稳定又安全。</p>

<h2>加载与应用</h2>
<pre class="code"><span class="cm"># 伪代码: 加载并挂载 LoRA</span>
adapter = <span class="fn">llama_adapter_lora_init</span>(model, <span class="st">"style.gguf"</span>)   <span class="cm"># 读 A/B 张量</span>
<span class="fn">llama_set_adapters_lora</span>(ctx, [adapter], n=1, scales=[0.8])  <span class="cm"># 批量挂载, 各带 scale</span>
<span class="cm"># ... decode 若干步, 输出带上这个风格 ...</span>
<span class="fn">llama_set_adapters_lora</span>(ctx, [], n=0, NULL)              <span class="cm"># n=0 =&gt; 清空, 卸下全部</span></pre>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">加载</span><span class="name">llama_adapter_lora_init(model, "x.gguf")</span></div><div class="ld">从 GGUF 读出 A/B 张量，按目标权重名索引</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">挂载</span><span class="name">llama_set_adapters_lora(ctx, [..], n, scales)</span></div><div class="ld">批量挂到 context、各带 scale；不复制权重（n=0 清空）</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">生效</span><span class="name">build_lora_mm（每次 decode 建图）</span></div><div class="ld">按张量名把 scale·B·A 折进相关 matmul</div></div>
</div>
<p>用起来很简单：<span class="mono">llama_adapter_lora_init</span> 从一个 <span class="mono">.gguf</span> 适配器文件读出 A、B 张量，得到一个适配器对象；再用 <span class="mono">llama_set_adapters_lora</span> 把它挂到 context（L17）上、并给一个 scale。之后的每次 decode，建图时就会自动把这个适配器的增量折进去。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  挂载用的是<strong>复数、批量</strong>的 <span class="mono">llama_set_adapters_lora</span>（一次可以挂多个适配器、各带一个 scale）。早期那套<strong>单数</strong>的 <span class="mono">llama_set_adapter_lora</span>/<span class="mono">rm</span>/<span class="mono">clear</span> 已经不存在了——清空适配器就是调批量版、传 <span class="mono">n=0</span>。看老代码时别再找单数那几个。
</div>
<p>"能同时挂多个、各带 scale"不是摆设，而是<strong>能力叠加</strong>的基础：你可以把"中文风格"和"法律领域"两个 LoRA 同时挂上、各给一个权重，让模型同时具备两种特长。批量接口天然支持这种组合——这也是为什么它被设计成一组 <span class="mono">{适配器, scale}</span>，而不是一次只能挂一个。</p>
<p>还要强调那个"<strong>不复制权重</strong>"：挂载只是在 context 上记下"现在生效哪些适配器、各什么 scale"，真正的叠加发生在每次 decode 建图时（<span class="mono">build_lora_mm</span>）。所以挂上、卸下 LoRA 几乎是零成本的——不涉及那几十 GB 权重的任何拷贝或修改，切换风格快得像换个滤镜。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  适配器为什么也用 <span class="mono">.gguf</span> 格式（L13）？因为 LoRA 本质上也是"一堆带名字的张量"（A、B 矩阵，按它们要修改的目标权重命名），和模型权重是同一类东西。复用 GGUF 这套自描述格式，意味着加载器（L14）几乎能照搬——读元数据、按名字建张量清单，连工具链都是现成的。一种格式通吃，省了重复造轮子。
</div>
<p>挂载时按目标张量名对号入座，也呼应了 L15 的命名约定。每个 LoRA 张量的名字，记着它要修改的是哪一层的哪个权重（比如某层的 attn_q）。建图时 <span class="mono">build_lora_mm</span> 算到那个权重，就去适配器里按名字找有没有对应的 A、B，有就把增量加上。名字再一次成了把"权重"和"补丁"对上的关键。</p>
<p>这种设计还带来一个好处：同一个 LoRA 文件，能套到任何<strong>结构兼容</strong>的基础模型上——因为它修改的目标是按名字指定的，不绑死某个具体模型实例。于是社区里一个针对某架构训练的 LoRA，常常能直接用在该架构的不同微调版本上。名字驱动的松耦合，让适配器的复用范围大大扩展。</p>
<p>顺带提一个实践中的常见组合：很多人用一个量化过的基础模型（L12）+ 一个 LoRA 适配器来跑，既享受量化省下的显存、又靠适配器获得任务特长。llama.cpp 对这种"量化底座 + LoRA"是支持的——适配器的增量在建图时按需叠加，和底座怎么量化基本正交。省显存和可定制，两个好处可以同时要。</p>

<h2>控制向量与衔接</h2>
<table class="t">
  <tr><th></th><th>改什么</th><th>怎么生效</th></tr>
  <tr><td>LoRA</td><td>权重（低秩增量 scale·B·A）</td><td>折进 matmul（build_lora_mm）</td></tr>
  <tr><td>控制向量</td><td>激活（沿固定方向平移）</td><td>加进残差流（set_adapter_cvec）</td></tr>
</table>
<p>除了 LoRA，还有一种更轻的"调味"手段：<strong>控制向量</strong>（control vector，cvec）。它不动权重，而是直接在某些层的<strong>激活</strong>（残差流）上，加一个固定方向的向量——好比给模型的"思路"轻轻推一把，让它整体偏向某种语气或倾向（更正式、更乐观之类）。</p>
<p>两者的区别值得记牢：LoRA 改的是<strong>权重</strong>（给 matmul 加低秩增量，影响那一层的全部计算），表达力强、能学复杂适配；控制向量改的是<strong>激活</strong>（沿一个方向平移残差流），更轻、更像"调味"，擅长沿某个语义方向微调风格。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  C API 上，cvec 用 <span class="mono">llama_set_adapter_cvec</span>（旧的 <span class="mono">llama_apply_adapter_cvec</span> 已移除）。
</div>
<p>把这一课接回第四部分的主线：适配器挂在 <span class="mono">llama_context</span>（L17）上，每次 <span class="mono">llama_decode</span> 建图（L16）时，<span class="mono">build_lora_mm</span> 把增量折进相关的 matmul。所以 LoRA/cvec 不是另起炉灶的新系统，而是<strong>嵌在已有推理回路里</strong>的一层薄薄的"行为调节"——复用了你前面学的建图、上下文这些机制。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  至此，第四部分（llama 推理内部）就完整了：从一个 .gguf 被<strong>加载</strong>成模型（L14-15），<strong>搭成计算图</strong>（L16），装进<strong>上下文</strong>按批用 KV 高效<strong>推理</strong>（L17-19），再到<strong>分词</strong>、<strong>采样</strong>、<strong>对话模板</strong>、<strong>语法约束</strong>、以及这一课的<strong>轻量微调</strong>（L20-24）。你已经把"一个大模型如何被驱动、控制、改造"从头到尾走了一遍。
</div>
<p>控制向量是怎么"算"出来的，值得一提。它往往不需要训练，而是用<strong>对比</strong>的办法：拿一批"正面例子"（比如语气正式的文本）和一批"负面例子"（随意的文本），分别跑过模型、取某层的激活，两组激活的<strong>差</strong>的方向，就大致是"正式"这个概念在模型内部的方向。把这个方向向量加进残差流，就能把输出往"更正式"推。简单、直接、还不用训练。</p>
<p>退一步看 LoRA 和控制向量的共同点：它们都践行了同一条原则——<strong>基础模型只读，改动外挂且可叠加</strong>。这跟 L17 把"只读权重"和"会话状态"分开、L21 把采样策略做成可插拔的链，是一脉相承的设计哲学。整个第四部分，其实都在反复演奏这一个主题：把不变的沉淀下来，把可变的拆出去，于是系统既稳固又灵活。</p>
<p>第四部分到此收尾。再往后（第五部分）我们会跳出 llama 内部，去看这些能力是怎么通过公共 API 和命令行工具暴露给你用的——你已经懂了引擎盖下的机理，接下来就是学会怎么开这辆车。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么"低秩"就够用？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>经验和理论都指向同一个观察：微调给权重带来的变化，往往落在一个<strong>低维子空间</strong>里。也就是说，"适配某个任务"所需的方向并不多，用一个秩很低（比如 r=8 或 16）的矩阵就能很好地张成。于是花极小的参数，就能逼近全量微调的效果。</p>
    <p>直觉上也说得通：基础模型已经学到了海量通用能力，适配新任务更像是在它之上做"小幅修正"，而不是推倒重来。小幅修正的自由度本就不高，低秩矩阵正好够用。这也是为什么 r 通常取得很小，再大收益也递减。</p>
    <p>这是个非常划算的取舍：参数量从 d×d 降到 2×d×r（r 远小于 d），可能小几百倍，效果却所失无几。用一点点近似换来巨大的成本下降——LoRA 之所以能在消费级硬件上微调大模型，根子就在这。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么挂载 API 是批量复数 llama_set_adapters_lora？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为现实里你常想<strong>同时挂多个</strong> LoRA：一个管语言风格、一个管领域知识，各给一个 scale 叠加使用。接口若一次只能挂一个，就表达不了这种组合。于是它天然被设计成一组 <span class="mono">{适配器, scale}</span> 的批量形式。</p>
    <p>这也简化了语义：挂载、替换、清空，全用同一个批量 setter 表达——传新的一组就是替换，传空（<span class="mono">n=0</span>）就是清空。不需要单独的 add/remove/clear 三件套，一个函数搞定所有情况，干净利落。</p>
    <p>所以旧教程里那套单数的 <span class="mono">llama_set_adapter_lora</span>/<span class="mono">llama_rm_adapter_lora</span>/<span class="mono">llama_clear_adapter_lora</span> 已经被这一个批量函数取代、不复存在了。看到老代码按单数签名调用，要知道那是过时的写法。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> LoRA 和控制向量，到底差在哪？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>层面不同。LoRA 动的是<strong>权重</strong>：在某些 matmul 上加一个低秩增量，于是那一层的<strong>整个</strong>线性变换都被改写了，表达力强，能学相对复杂的适配（新风格、新格式、新领域）。代价是它要训练、要存 A/B 矩阵。</p>
    <p>控制向量动的是<strong>激活</strong>：直接在残差流上加一个固定方向的向量，相当于沿某个语义轴（"正式 vs 随意""乐观 vs 悲观"）把模型的状态推一推。它更轻、更直接，往往不需要训练（可以从对比样本里算出方向），但表达力也更有限——擅长"调味"，不擅长"教新本事"。</p>
    <p>一句话：LoRA 是"<strong>低秩权重补丁</strong>"，控制向量是"<strong>激活方向偏置</strong>"。一个改算子怎么算，一个改数据往哪偏。它们都不碰基础权重、都即插即用，是同一类"轻量行为调节"的两种风味，按需要的表达力和成本来选。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>LoRA = 冻结原权重 W，只学小矩阵 A、B，输出 = W·x + <strong>scale·B·A·x</strong>（低秩增量）；适配器常仅几 MB。</li>
    <li>数学在建图时实现（<span class="mono">build_lora_mm</span>，<span class="mono">src/llama-graph.cpp</span>）：<span class="mono">res = W·x</span>，再 <span class="mono">+ scale·B·(A·x)</span>；scale 来自 <span class="mono">alpha/rank</span> × 用户比例。</li>
    <li>加载 <span class="mono">llama_adapter_lora_init</span>；挂载用<strong>批量</strong> <span class="mono">llama_set_adapters_lora</span>（单数 set/rm/clear 已移除，<span class="mono">n=0</span> 清空）。</li>
    <li>控制向量 <span class="mono">llama_set_adapter_cvec</span>：沿固定方向平移<strong>激活</strong>；LoRA 改<strong>权重</strong>。两者都不复制权重、即插即用。</li>
    <li>适配器挂在 context（L17）、decode 建图（L16）时折进 matmul，<strong>不改</strong>基础权重。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  LoRA 把"<strong>改变模型行为</strong>"从"重训整套权重"降到"加一片几 MB 的低秩滤镜"——基础模型只读、增量即插即用。它和第四部分反复出现的主题一脉相承：把<strong>只读的知识</strong>（权重）和<strong>可变的部分</strong>（适配器、上下文、采样策略）分开，于是一份大模型能被千变万化地复用。学到这里，你已走完第四部分——从一个 .gguf 文件被加载，到它如何被驱动、约束、并轻量改造成你想要的样子。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
All through Part 4, you can now load, infer, tokenize, sample, apply chat formats, and constrain output (L14-L23). The last piece of the puzzle: what if you want the model to <strong>learn a new style or task</strong>? Full fine-tuning means retraining and storing a whole set of tens-of-GB weights - expensive and clumsy. This lesson covers <strong>LoRA</strong> - a far lighter approach that patches the weights with two small matrices, changing model behavior in mere megabytes.
</p>
<p style="color:var(--muted);margin-top:.4rem">LoRA's essence is <strong>low rank</strong>: it leaves the original weights untouched and learns only a <strong>low-rank delta</strong> (the product of two small matrices A and B), added to the original weights' output. Because the rank is low, these two matrices are tiny (an adapter is often just a few MB), yet they approximate full fine-tuning's effect. Better still, it is plug-and-play - attach the adapter for a style you want, drop it anytime, and even stack several.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  LoRA is like putting a <strong>filter</strong> on a lens: the original lens (base weights) does not move at all, you put on a light filter (the low-rank delta of A and B) and the look changes; do not like it, take it off anytime, and you can stack a few. A <strong>control vector</strong>, by contrast, is like a color-grading knob, shifting the whole picture's tone along one fixed direction. Neither touches the negative, just tweaks at print time.
</div>

<h2>Why LoRA is needed</h2>
<div class="cols">
  <div class="col"><h4>Full fine-tuning</h4><p>Update <strong>all</strong> weights, store a whole set (tens of GB), one per task. Expensive and clumsy.</p></div>
  <div class="col"><h4>LoRA</h4><p>Freeze the original weights W, learn only two <strong>small</strong> matrices A, B. An adapter is often a few MB, stackable, removable.</p></div>
</div>
<p>First, see full fine-tuning's pain point clearly. To adapt a billions-of-parameters model to a new task, the traditional way is to keep training and update <strong>all</strong> its weights, then store this whole new weight set. The problem: each task needs its own tens-of-GB model, training takes lots of VRAM - expensive, space-hungry, and hard to share.</p>
<p>LoRA takes a different tack: <strong>freeze</strong> the original weights, not a byte changed, and learn just two small matrices A, B alongside. To use it, temporarily add the delta computed from A and B onto the original weights. So the "adapter" you store, share, and load is just those two small matrices A and B, often only a few MB - compared to tens of GB of full fine-tuning, several orders of magnitude smaller.</p>
<p>The key assumption behind this is: the change fine-tuning brings to the weights often concentrates in a <strong>low-dimensional subspace</strong>. In other words, the adjustment needed to "make the model learn a task" has not that many degrees of freedom, and a low-rank matrix approximates it well. It is exactly this insight that makes "learn just two small matrices" possible, and surprisingly effective.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The real payoff is concrete: one base model + a pile of few-MB LoRAs conjures countless "specialized versions" - a coder, a role-player, a domain expert, attached on demand. The base model is read-only, stored once, and all the difference lives in those small adapters. This "one base, many patches" pattern is the fundamental reason LoRA caught on.
</div>
<p>A more down-to-earth analogy. Full fine-tuning is like reprinting a whole book to change one sentence; LoRA is like sticking a few notes onto the original book - the book is untouched, yet the notes suffice to express your edits. To change an edit, peel the notes and swap a new batch; the original book is forever that one book. This "original untouched, edits attached" thinking is the source of all of LoRA's convenience.</p>
<p>The collaboration dividend is large too. In the community, what people share is no longer the tens-of-GB whole model but few-MB adapters - fast to download, cheap to store, and freely combinable like plugins. A popular base model is often surrounded by hundreds or thousands of LoRAs each with its own trick, a flourishing bought precisely by "lightweight and shareable".</p>
<p>Of course LoRA is not omnipotent. It excels at "stylistic, domain-specific adjustments on top of existing ability", but to make the model learn a brand-new skill it <strong>utterly lacks</strong>, the low-rank delta's expressiveness may fall short, and heavier training is then needed. Knowing its boundary lets you use it where it counts - most needs at the "tone, format, domain" level, LoRA catches beautifully.</p>

<h2>The LoRA math</h2>
<div class="flow">
  <div class="node"><div class="nt">x</div><div class="nd">input</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">A</div><div class="nd">down to rank r</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">B</div><div class="nd">back up to dim</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">x scale</div><div class="nd">scale strength</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">+ W*x</div><div class="nd">add to base output</div></div>
</div>
<p>How exactly is it computed? Let the original weight be W and the input x. The base output is W*x (an ordinary matmul). LoRA adds a <strong>low-rank branch</strong> alongside it: first A drops x to a very low dimension (rank r), then B lifts it back to the original dimension, giving B*(A*x); multiply by a scale factor and add onto W*x. The final output = W*x + scale*B*A*x.</p>
<p>Draw this formula by "dimension" and the low-rank bottleneck pops out: the base W*x keeps its width; the bypass is first squeezed by A to a narrow rank r, lifted back by B, scaled, and added to the base.</p>
<div class="trace">
  <div class="tcap"><b>Tracing one LoRA forward</b>: the frozen W*x stays; the bypass squeezes x to low rank r, lifts it back, scales it, and adds it in (dims illustrative).</div>
  <svg viewBox="0 0 660 226" width="100%" role="img" aria-label="LoRA forward worked example">
<g font-family="ui-monospace,monospace">
<text x="150" y="20" fill="#5b6470" font-size="11">base W*x  (frozen, not trained)</text>
<text x="150" y="206" fill="#5b6470" font-size="11">bypass: squeeze x to low rank r, lift back</text>
<rect x="20" y="82" width="46" height="60" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="43" y="117" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">x</text><text x="43" y="155" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
<line x1="66" y1="100" x2="147" y2="67" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 150 66 L 144 73 L 141 65 z" fill="#9aa6b2"/>
<rect x="150" y="46" width="60" height="40" rx="5" fill="#e7edf5" stroke="#2563eb"/><text x="180" y="71" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">W</text><text x="180" y="99" text-anchor="middle" fill="#5b6470" font-size="10">frozen</text>
<line x1="210" y1="66" x2="247" y2="72" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 250 72 L 241 75 L 243 67 z" fill="#9aa6b2"/>
<rect x="250" y="46" width="58" height="60" rx="5" fill="#e7edf5" stroke="#2563eb"/><text x="279" y="81" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">W*x</text><text x="279" y="119" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
<line x1="66" y1="124" x2="123" y2="149" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 126 150 L 117 150 L 120 143 z" fill="#9aa6b2"/>
<rect x="126" y="150" width="46" height="34" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="149" y="172" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">A</text><text x="149" y="197" text-anchor="middle" fill="#5b6470" font-size="10">4-&gt;r</text>
<line x1="172" y1="167" x2="203" y2="175" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 206 176 L 197 178 L 199 170 z" fill="#9aa6b2"/>
<rect x="206" y="168" width="44" height="18" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="228" y="182" text-anchor="middle" fill="#fff" font-weight="700" font-size="12">r=1</text>
<line x1="250" y1="176" x2="283" y2="168" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 286 167 L 279 173 L 277 165 z" fill="#9aa6b2"/>
<rect x="286" y="150" width="46" height="34" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="309" y="172" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">B</text><text x="309" y="197" text-anchor="middle" fill="#5b6470" font-size="10">r-&gt;4</text>
<line x1="332" y1="160" x2="370" y2="134" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 372 132 L 368 140 L 363 133 z" fill="#9aa6b2"/>
<rect x="372" y="102" width="56" height="60" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="400" y="137" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="14">[4]</text>
<line x1="308" y1="76" x2="467" y2="98" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 470 98 L 462 101 L 463 93 z" fill="#9aa6b2"/>
<line x1="428" y1="132" x2="467" y2="109" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 470 108 L 465 115 L 461 108 z" fill="#9aa6b2"/>
<text x="452" y="150" text-anchor="middle" fill="#c2630e" font-size="10">*scale</text>
<rect x="470" y="84" width="40" height="40" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="490" y="109" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="18">+</text>
<line x1="510" y1="104" x2="553" y2="104" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 556 104 L 548 108 L 548 100 z" fill="#9aa6b2"/>
<rect x="556" y="74" width="58" height="60" rx="5" fill="#c2630e" stroke="#c2630e"/><text x="585" y="109" text-anchor="middle" fill="#fff" font-weight="700" font-size="14">y</text><text x="585" y="147" text-anchor="middle" fill="#5b6470" font-size="10">[4]</text>
</g></svg>
</div>
<pre class="code"><span class="cm">// simplified from src/llama-graph.cpp build_lora_mm</span>
res = <span class="fn">ggml_mul_mat</span>(w, cur);                  <span class="cm">// base weight output W*x</span>
<span class="kw">for</span> (lora : active_adapters) {
    ab = <span class="fn">ggml_mul_mat</span>(b, <span class="fn">ggml_mul_mat</span>(a, cur)); <span class="cm">// low-rank two steps B*(A*x)</span>
    ab  = <span class="fn">ggml_scale</span>(ab, scale);             <span class="cm">// scale = alpha/rank * user ratio</span>
    res = <span class="fn">ggml_add</span>(res, ab);                 <span class="cm">// add the delta</span>
}</pre>
<p>The snippet above (simplified from <span class="mono">src/llama-graph.cpp</span>'s <span class="mono">build_lora_mm</span>) is what it does at graph-build time (L16): first compute the base <span class="mono">mul_mat(w, cur)</span>, then for each active adapter compute <span class="mono">B*(A*cur)</span>, multiply by scale, and add back. Note all this happens in the <strong>compute graph</strong> - the delta is computed and added on the fly, never actually modifying those GB of W weights.</p>
<p>That scale matters too: it is computed from the adapter's <span class="mono">alpha</span> and the rank r (roughly <span class="mono">alpha/rank</span>, times a user-given ratio). This ratio is the "<strong>strength</strong>" knob you can tune at attach time - turn it up for a stronger adapter influence, down to stay closer to the original model. Making strength adjustable lets you glide smoothly between "as-is" and "fully transformed".</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Why the <strong>two steps</strong> of A down, B up, rather than learning one delta matrix of the same size directly? Because learning a full-rank d x d matrix directly has as many parameters as the original weight, defeating the point. Splitting into two slim matrices d x r and r x d (r far smaller than d) drops the parameters from d squared to 2dr - this is the mathematical essence of "low rank" saving parameters.
</div>
<p>Let me spell out the word "rank" a bit more. A matrix's rank is, roughly, how many "truly independent directions" it has. Full rank means all directions are used, maximal information but also maximal parameters; low rank says "actually only a few directions suffice to describe this change". LoRA's bet is exactly that the change needed to adapt a task is essentially low-rank, so r directions (the inner dimension of A and B is r) approximate it well enough.</p>
<p>One often-missed detail: the initialization of A and B is asymmetric. Usually B is initialized to all zeros and A randomly, so at the start of training the delta B*A is zero - that is, attaching an <strong>untrained</strong> LoRA has no effect on the model, the same as attaching none. Only training gradually grows useful directions in this delta. This "start from zero, join smoothly" design makes LoRA training both stable and safe.</p>

<h2>Loading and applying</h2>
<pre class="code"><span class="cm"># pseudocode: load and attach a LoRA</span>
adapter = <span class="fn">llama_adapter_lora_init</span>(model, <span class="st">"style.gguf"</span>)   <span class="cm"># read A/B tensors</span>
<span class="fn">llama_set_adapters_lora</span>(ctx, [adapter], n=1, scales=[0.8])  <span class="cm"># batch attach, each with scale</span>
<span class="cm"># ... decode a few steps, output carries this style ...</span>
<span class="fn">llama_set_adapters_lora</span>(ctx, [], n=0, NULL)              <span class="cm"># n=0 =&gt; clear, detach all</span></pre>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">load</span><span class="name">llama_adapter_lora_init(model, "x.gguf")</span></div><div class="ld">read A/B tensors from GGUF, indexed by target weight name</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">attach</span><span class="name">llama_set_adapters_lora(ctx, [..], n, scales)</span></div><div class="ld">batch-attach to the context, each with a scale; no weight copy (n=0 clears)</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">effect</span><span class="name">build_lora_mm (per-decode graph build)</span></div><div class="ld">fold scale*B*A into the relevant matmul by tensor name</div></div>
</div>
<p>It is simple to use: <span class="mono">llama_adapter_lora_init</span> reads the A, B tensors from a <span class="mono">.gguf</span> adapter file, giving an adapter object; then <span class="mono">llama_set_adapters_lora</span> attaches it to the context (L17) with a scale. Every subsequent decode automatically folds this adapter's delta in at graph-build time.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  Attaching uses the <strong>plural, batched</strong> <span class="mono">llama_set_adapters_lora</span> (you can attach several adapters at once, each with a scale). The early <strong>singular</strong> <span class="mono">llama_set_adapter_lora</span>/<span class="mono">rm</span>/<span class="mono">clear</span> no longer exist - clearing adapters is calling the batched version with <span class="mono">n=0</span>. Do not go looking for those singular ones in old code.
</div>
<p>"Attach several at once, each with a scale" is no ornament but the basis of <strong>stacking abilities</strong>: you can attach a "Chinese style" and a "legal domain" LoRA at once, each with a weight, giving the model both specialties. The batched interface naturally supports this combination - which is why it is designed as a set of <span class="mono">{adapter, scale}</span>, not one-at-a-time.</p>
<p>Stress that "<strong>no weight copy</strong>" again: attaching merely records on the context "which adapters are active now, each at what scale"; the real addition happens at each decode's graph build (<span class="mono">build_lora_mm</span>). So attaching and detaching a LoRA is nearly free - involving no copy or modification of those GB of weights, switching styles as fast as swapping a filter.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Why is an adapter also in <span class="mono">.gguf</span> format (L13)? Because a LoRA is essentially also "a bunch of named tensors" (the A, B matrices, named after the target weights they modify), the same kind of thing as model weights. Reusing GGUF's self-describing format means the loader (L14) can be reused almost verbatim - read metadata, build the tensor list by name, even the toolchain is ready-made. One format fits all, sparing reinvented wheels.
</div>
<p>Matching by target tensor name at attach time also echoes L15's naming convention. Each LoRA tensor's name records which layer's which weight it modifies (say a layer's attn_q). At graph build, when <span class="mono">build_lora_mm</span> reaches that weight, it looks up the adapter by name for a matching A, B, and adds the delta if found. Names once again become the key that pairs "weight" with "patch".</p>
<p>This design brings another benefit: the same LoRA file can apply to any <strong>structurally compatible</strong> base model - because its modification targets are specified by name, not bound to a specific model instance. So a community LoRA trained for one architecture can often be used directly on different fine-tuned versions of that architecture. Name-driven loose coupling vastly expands an adapter's reuse range.</p>
<p>A common combination in practice worth mentioning: many people run a quantized base model (L12) + a LoRA adapter, enjoying the VRAM saved by quantization while gaining task specialty from the adapter. llama.cpp supports this "quantized base + LoRA" - the adapter's delta is added on demand at graph build, largely orthogonal to how the base is quantized. Saving VRAM and staying customizable, you can have both.</p>

<h2>Control vectors and the hand-off</h2>
<table class="t">
  <tr><th></th><th>What it changes</th><th>How it takes effect</th></tr>
  <tr><td>LoRA</td><td>weights (low-rank delta scale*B*A)</td><td>folded into matmul (build_lora_mm)</td></tr>
  <tr><td>Control vector</td><td>activations (shift along a fixed direction)</td><td>added to the residual stream (set_adapter_cvec)</td></tr>
</table>
<p>Besides LoRA, there is an even lighter "seasoning" means: the <strong>control vector</strong> (cvec). It touches no weights but adds a fixed-direction vector directly onto the <strong>activations</strong> (the residual stream) of certain layers - like nudging the model's "train of thought", tilting it overall toward some tone or tendency (more formal, more optimistic, and so on).</p>
<p>The difference is worth remembering: LoRA changes <strong>weights</strong> (adding a low-rank delta to matmul, affecting that layer's entire computation), expressive, able to learn complex adaptations; the control vector changes <strong>activations</strong> (shifting the residual stream along one direction), lighter, more like "seasoning", good at fine-tuning style along a semantic direction.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  In the C API, cvec uses <span class="mono">llama_set_adapter_cvec</span> (the old <span class="mono">llama_apply_adapter_cvec</span> is removed).
</div>
<p>Connecting this lesson back to Part 4's main line: the adapter is attached to <span class="mono">llama_context</span> (L17), and at each <span class="mono">llama_decode</span> graph build (L16), <span class="mono">build_lora_mm</span> folds the delta into the relevant matmul. So LoRA/cvec is not a new system started from scratch but a thin layer of "behavior tuning" <strong>embedded in the existing inference loop</strong> - reusing the graph-build and context mechanisms you learned earlier.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  With that, Part 4 (inside llama inference) is complete: from a .gguf being <strong>loaded</strong> into a model (L14-15), <strong>assembled into a compute graph</strong> (L16), packed into a <strong>context</strong> and inferred efficiently in batches with the KV cache (L17-19), to <strong>tokenization</strong>, <strong>sampling</strong>, <strong>chat templates</strong>, <strong>grammar constraints</strong>, and this lesson's <strong>lightweight fine-tuning</strong> (L20-24). You have now walked end to end through "how a large model is driven, controlled, and reshaped".
</div>
<p>How a control vector is "computed" is worth a mention. It often needs no training but uses a <strong>contrastive</strong> method: take a batch of "positive examples" (say formal-toned text) and a batch of "negative examples" (casual text), run each through the model and take a layer's activations; the direction of the <strong>difference</strong> between the two activation sets is roughly the direction of the concept "formal" inside the model. Add this direction vector to the residual stream and you push the output toward "more formal". Simple, direct, and training-free.</p>
<p>Step back to the common ground of LoRA and control vectors: both practice the same principle - <strong>the base model is read-only, the changes are attached and stackable</strong>. This is of one piece with L17 separating "read-only weights" from "session state" and L21 making the sampling strategy a pluggable chain. All of Part 4, really, plays this one theme over and over: settle the invariant, split out the mutable, so the system is both solid and flexible.</p>
<p>Part 4 ends here. Beyond it (Part 5) we step out of llama's internals to see how these abilities are exposed to you through the public API and command-line tools - having understood the machinery under the hood, next is learning to drive the car.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why is "low rank" enough? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Experience and theory point to the same observation: the change fine-tuning brings to the weights often lands in a <strong>low-dimensional subspace</strong>. That is, "adapting to a task" needs few directions, well spanned by a very low-rank matrix (say r=8 or 16). So with tiny parameters you approximate full fine-tuning's effect.</p>
    <p>It makes intuitive sense too: the base model already learned vast general ability, and adapting to a new task is more like a "small correction" on top of it than a rebuild from scratch. A small correction has inherently few degrees of freedom, and a low-rank matrix is just enough. This is also why r is usually small - bigger brings diminishing returns.</p>
    <p>It is a very cost-effective trade: parameters drop from d x d to 2 x d x r (r far smaller than d), possibly hundreds of times smaller, with the effect barely diminished. Trading a little approximation for a huge cost cut - this is the root of why LoRA can fine-tune large models on consumer hardware.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is the attach API the batched plural llama_set_adapters_lora? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Because in reality you often want to <strong>attach several</strong> LoRAs at once: one for language style, one for domain knowledge, each with a scale, used together. An interface that attaches only one at a time cannot express this combination. So it is naturally designed as a batched set of <span class="mono">{adapter, scale}</span>.</p>
    <p>It also simplifies the semantics: attach, replace, clear are all expressed by the same batched setter - pass a new set to replace, pass empty (<span class="mono">n=0</span>) to clear. No need for a separate add/remove/clear trio; one function handles every case, clean and tidy.</p>
    <p>So the singular <span class="mono">llama_set_adapter_lora</span>/<span class="mono">llama_rm_adapter_lora</span>/<span class="mono">llama_clear_adapter_lora</span> from old tutorials are replaced by this one batched function and no longer exist. Seeing old code call them by the singular signature, know that is an outdated form.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> LoRA vs control vectors, what exactly differs? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Different levels. LoRA acts on <strong>weights</strong>: adding a low-rank delta to certain matmuls, so that layer's <strong>entire</strong> linear transform is rewritten - expressive, able to learn relatively complex adaptations (new style, new format, new domain). The cost is it needs training and storing the A/B matrices.</p>
    <p>The control vector acts on <strong>activations</strong>: adding a fixed-direction vector directly to the residual stream, like nudging the model's state along a semantic axis ("formal vs casual", "optimistic vs pessimistic"). It is lighter and more direct, often needing no training (the direction can be computed from contrasting samples), but also more limited in expressiveness - good at "seasoning", not at "teaching new skills".</p>
    <p>In a sentence: LoRA is a "<strong>low-rank weight patch</strong>", the control vector is an "<strong>activation-direction bias</strong>". One changes how an operator computes, the other where the data leans. Both leave the base weights untouched and are plug-and-play, two flavors of the same "lightweight behavior tuning" - choose by the expressiveness and cost you need.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>LoRA = freeze the original weights W, learn only small matrices A, B; output = W*x + <strong>scale*B*A*x</strong> (a low-rank delta); adapters are often just a few MB.</li>
    <li>The math is implemented at graph build (<span class="mono">build_lora_mm</span>, <span class="mono">src/llama-graph.cpp</span>): <span class="mono">res = W*x</span>, then <span class="mono">+ scale*B*(A*x)</span>; scale comes from <span class="mono">alpha/rank</span> x a user ratio.</li>
    <li>Load with <span class="mono">llama_adapter_lora_init</span>; attach with the <strong>batched</strong> <span class="mono">llama_set_adapters_lora</span> (singular set/rm/clear removed, <span class="mono">n=0</span> clears).</li>
    <li>Control vector <span class="mono">llama_set_adapter_cvec</span>: shifts <strong>activations</strong> along a fixed direction; LoRA changes <strong>weights</strong>. Both copy no weights and are plug-and-play.</li>
    <li>The adapter is attached to the context (L17) and folded into matmul at decode graph build (L16), <strong>not changing</strong> the base weights.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  LoRA brings "<strong>changing model behavior</strong>" down from "retraining a whole weight set" to "adding a few-MB low-rank filter" - the base model read-only, the delta plug-and-play. It is of one piece with Part 4's recurring theme: separate the <strong>read-only knowledge</strong> (weights) from the <strong>mutable parts</strong> (adapters, context, sampling strategy), so one large model can be reused in endless variations. Reaching here, you have finished Part 4 - from a .gguf file being loaded, to how it is driven, constrained, and lightly reshaped into what you want.
</div>
""",
}

