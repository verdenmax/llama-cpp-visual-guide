"""Content for Part 3 (the ggml engine)."""

LESSON_08 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
第二部分给了你张量、量化、后端的<strong>直觉</strong>；从第三部分起，我们正式拆开 ggml 这台引擎。第一站是"<strong>内存</strong>"——
ggml 到底怎么管理成千上万个张量的内存？答案出人意料地朴素：一个叫 <span class="mono">ggml_context</span> 的<strong>内存池</strong>，
加上一套"绝不零散 malloc"的分配哲学。看懂这一层，后面的计算图、执行、算子才有立足之地。
</p>
<p style="color:var(--muted);margin-top:.4rem">先说清这一课的<strong>问题意识</strong>：一次推理要凭空造出成千上万个张量——权重、中间激活、KV cache，全是张量。
如果每造一个就向操作系统讨一次内存，光是"申请、记账、归还"的开销就足以拖慢整个引擎。ggml 的回答是：<strong>别零买，批发</strong>。
它一次性圈下一大块内存，自己在里面精打细算地切——这块内存就叫 <span class="mono">ggml_context</span>。这一课就讲它怎么圈地、怎么切、以及为什么这么设计。</p>
<p>顺便把一个名词对上号：这种"<strong>一次圈一大块、内部自己切、用完整体释放</strong>"的内存管理方式，在系统编程里有个通用名字叫
<strong>arena（竞技场 / 区域）分配器</strong>，也叫<strong>线性分配器</strong>或<strong>区域分配器</strong>。它不是 ggml 的独创，而是高性能程序里常见的老把戏；
ggml 只是把它用在了"管理一张计算图的所有张量"这个恰到好处的场景上。后面我们说 arena，指的就是 <span class="mono">ggml_context</span> 持有的那块大内存。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  <span class="mono">ggml_context</span> 像一块<strong>预先划好的停车场</strong>：开门营业时一次性圈下一整片地（<span class="mono">mem_size</span>），
  之后每停一辆车（建一个张量或对象）就往后挪一个车位（bump 游标），<strong>不必每次都跑去物业重新申请地皮</strong>（malloc）。
  收工时整片场地一次清空，干净利落。把"反复找物业"换成"自己在圈好的地里挪车位"，正是 ggml 内存管理的全部精髓。把这个比喻记牢，这一课就成功了一半。
</div>

<h2>地基三件套</h2>
<p>ggml 引擎最底层，其实就三样东西扣在一起：一份<strong>配置</strong>（你想要多大的池子）、一个<strong>内存池本体</strong>、以及池子里一个挨一个排着的
<strong>对象</strong>（每个对象包着一个张量或一张计算图）。先看它们的关系：</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">配置</span><span class="name">ggml_init_params</span></div><div class="ld">mem_size（多大）· mem_buffer（用谁的内存）· no_alloc（要不要给数据留位）</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">内存池</span><span class="name">ggml_context</span></div><div class="ld">持有一整块 arena，记着用到哪了（游标）、对象链表头尾</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">对象</span><span class="name">ggml_object -&gt; ggml_tensor / ggml_cgraph</span></div><div class="ld">池子里一个接一个排开的对象，每个包着一个张量或一张图</div></div>
</div>
<p>这一层的关键词是<strong>"池"</strong>：ggml 不会零散地为每个张量去找系统要内存，而是<strong>一次性拿一大块、自己在里面切</strong>。
下面三节就把这三件套逐一讲清。</p>
<p>这里先埋一个贯穿全课的对照：<strong>"元数据"和"数据"是两回事</strong>。元数据是"<strong>这个张量长什么样</strong>"——它的形状 ne、步长 nb、类型 type、
以及 op/src 这些（L05 讲过的字段），加起来不过几十上百字节；数据则是"<strong>那一大片真正的浮点数</strong>"，一个权重矩阵可能就是几十 MB。
<span class="mono">ggml_context</span> 这块 arena，很多时候<strong>只装元数据</strong>，把笨重的数据留给后端去管。带着这个"轻元数据 / 重数据"的分野往下读，很多设计就顺理成章了。</p>

<h2>ggml_context：一个内存池</h2>
<p>一切从 <span class="mono">ggml_init(params)</span> 开始。你在 <span class="mono">ggml_init_params</span> 里告诉 ggml 三件事，它就回给你一个
<span class="mono">ggml_context</span>——一块已经备好的内存池：</p>
<pre class="code"><span class="cm">// 简化自 ggml/include/ggml.h</span>
<span class="kw">struct</span> ggml_init_params {
    size_t mem_size;     <span class="cm">// arena 总大小</span>
    <span class="kw">void</span> * mem_buffer;   <span class="cm">// 传 NULL 则由 ggml 内部分配这块内存</span>
    <span class="kw">bool</span>   no_alloc;     <span class="cm">// true = 只建张量"元数据", 不为张量"数据"留位</span>
};

ctx = <span class="fn">ggml_init</span>(params);   <span class="cm">// 一次性拿到整块 arena</span>
<span class="cm">// ... 在 ctx 里建很多张量 ...</span>
<span class="fn">ggml_free</span>(ctx);             <span class="cm">// 一次性整体释放</span></pre>
<div class="flow">
  <div class="node"><div class="nt">ggml_init(params)</div><div class="nd">圈一块 arena<br>(mem_size)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">ggml_context</div><div class="nd">内存池: arena + 游标<br>+ 对象链表</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">ggml_free(ctx)</div><div class="nd">整块一次性还掉</div></div>
</div>
<p><span class="mono">ggml_init</span> 做的事很直接：先为 <span class="mono">ggml_context</span> 这个管理结构本身分配一点空间，
然后<strong>准备好那块大 arena</strong>——如果你传了 <span class="mono">mem_buffer</span>，它就用你给的内存；传 <span class="mono">NULL</span>，
它就自己 <span class="mono">ggml_aligned_malloc(mem_size)</span> 要一块对齐过的内存（源码见 <span class="mono">ggml/src/ggml.c</span>）。
<span class="mono">ggml_free</span> 则把这块 arena 整体还掉（只有当这块内存是 ggml 自己分配的、即 <span class="mono">mem_buffer_owned</span> 时才释放）。
"<strong>能让你传入 mem_buffer</strong>"这一点很重要：它意味着 ggml 可以在别人给的内存上工作，方便嵌入到各种环境、或复用一块缓冲反复建图。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  顺带说说<strong>对齐</strong>。<span class="mono">ggml_aligned_malloc</span> 要的不是普通内存，而是<strong>对齐到特定边界</strong>的内存——因为后端的 SIMD 指令（AVX、NEON 等，L07 提过）往往要求数据地址对齐才能高效甚至正确地读取。arena 内部每切一个对象，也会按 <span class="mono">GGML_MEM_ALIGN</span> 对齐。你可以把 arena 理解成一条<strong>带刻度的尺子</strong>，每个对象都落在整齐的刻度上，而不是随手乱放——这点整齐，换来的是计算时的速度。
</div>
<p>所以严格说，<span class="mono">ggml_aligned_malloc</span> 与普通 <span class="mono">malloc</span> 的区别就在"<strong>对齐</strong>"二字：普通 malloc 只保证够大、不保证地址落在某个边界上；
而 ggml 要的内存，起始地址必须是某个对齐值（如 16 或 32 字节）的整数倍，这样后端才能放心地用对齐版的 SIMD 加载指令一次搬一大批数。对齐这件小事，体现的是 ggml"<strong>处处为后端计算让路</strong>"的取向。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  还有一个常被问到的问题：<strong>一个程序里能开几个 <span class="mono">ggml_context</span>？</strong>答案是<strong>多个</strong>，而且这很常见。比如可以用一个 ctx 装<strong>模型权重</strong>（活得久，整个推理期间都在）、另一个 ctx 装<strong>每步推理的计算图</strong>（活得短，算完就清）。不同生命周期的东西放进不同的池子，<strong>该长留的长留、该速清的速清</strong>，互不干扰——这也是 arena 模型带来的便利：一次 <span class="mono">ggml_free</span> 就能精准回收一整批同寿命的对象。
</div>
<p>这套机制在 llama.cpp 里随处可见。加载一个模型时，loader 会先按 GGUF 头里记的张量数量和大小，<strong>估出需要多大的元数据 arena</strong>，
<span class="mono">ggml_init</span> 出一个（通常 <span class="mono">no_alloc=true</span> 的）context，再把每个权重张量在里面登记一遍；权重的真正数据则由后端缓冲承接（甚至直接 mmap 自文件，见 L13）。
每跑一步推理，又会用另一个 context 临时搭出这一步的计算图、算完即弃。所以你大可以把 <span class="mono">ggml_context</span> 想成 ggml 世界里<strong>最基本的"工作台"</strong>：
要干活，先支一个台子；活干完，连台带料一起收走。理解了它，你就理解了 ggml 所有数据结构"住在哪里"。</p>

<h2>no-malloc：bump 分配</h2>
<p>拿到 arena 之后，每建一个张量、一张图，ggml 都<strong>不</strong>再去找系统要内存，而是在这块 arena 里"<strong>往后推一格</strong>"。这就是 bump（碰撞指针）分配：
维护一个"用到哪了"的游标，来一个对象就把游标往右挪、就地放下：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>bump 分配</b>：所有对象在同一块 arena 里一个接一个排开，游标只进不退</div>
  <div class="cells"><span class="lab">arena</span><span class="cell hl">obj1</span><span class="cell hl">obj2</span><span class="cell hl">obj3</span><span class="cell dim">空闲 …</span><span class="lab">游标 -&gt;</span></div>
  <div class="cells"><span class="lab">链表</span><span class="cell dim">begin</span><span class="sep">-&gt;</span><span class="cell">obj1</span><span class="sep">-&gt;</span><span class="cell">obj2</span><span class="sep">-&gt;</span><span class="cell">obj3</span><span class="sep">-&gt;</span><span class="cell dim">end</span><span class="lab">begin/end 是头尾指针</span></div>
</div>
<p>每个对象前面都挂一个小小的 <span class="mono">ggml_object</span> 头（记着自己的偏移、大小、指向下一个对象的指针），它们串成一条链表；
游标永远停在最后一个对象的末尾。新建对象时，就从游标处往后切一块。把这个过程写成伪代码就一目了然：</p>
<p>多说一句那个 <span class="mono">ggml_object</span> 头里到底装了什么：自己在 arena 里的<strong>偏移</strong> <span class="mono">offs</span>、占用的<strong>大小</strong> <span class="mono">size</span>、
指向<strong>下一个对象</strong>的指针 <span class="mono">next</span>，外加一个标记"这是张量还是图"的类型字段。<span class="mono">ggml_context</span> 自己则记着链表的头尾
（<span class="mono">objects_begin</span> / <span class="mono">objects_end</span>）和已放对象数 <span class="mono">n_objects</span>。有了尾指针，"在末尾追加"就是 O(1)，这是 bump 快的又一面。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  为什么这种"只加不减"的游标能行得通？因为<strong>建图阶段几乎只增不删</strong>——你是在一口气把整张计算图搭出来，中途很少需要单独释放某个张量。既然没有"挖东墙补西墙"的需求，那最简单的分配器（一个往前推的游标）就够用了，连记录空闲块、合并碎片这些复杂逻辑都省了。这是一种典型的"<strong>用使用场景的特点，换分配器的极致简单</strong>"：等到 L10 真正要<strong>复用</strong>内存时，才会上更聪明的分配器；而这里的建图阶段，朴素的 bump 反而最合适。
</div>
<pre class="code"><span class="cm"># 对应 ggml/src/ggml.c 的 ggml_new_object / ggml_new_tensor_impl</span>
<span class="kw">def</span> <span class="fn">new_object</span>(ctx, size):
    cur = ctx.objects_end.offs + ctx.objects_end.size   <span class="cm"># 当前游标</span>
    <span class="kw">if</span> cur + size &gt; ctx.mem_size:                       <span class="cm"># 池子不够了</span>
        abort("arena 空间不足")                          <span class="cm"># 不扩容, 直接报错!</span>
    obj = place_at(ctx.mem_buffer + cur)                <span class="cm"># 就地放下</span>
    link_into(ctx.objects, obj)                         <span class="cm"># 接入链表尾</span>
    <span class="kw">return</span> obj</pre>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  这里有两个要点。其一，<strong>张量的元数据（那个 <span class="mono">ggml_tensor</span> 结构）和它的数据缓冲，都从这同一块 arena 里切</strong>——没有"每个张量单独 <span class="mono">malloc</span> 一次"这回事。其二，<strong>arena 不会自动扩容</strong>：游标一旦撞到边界，ggml 直接 <span class="mono">abort</span>。所以使用者要<strong>事先把池子估得足够大</strong>。ggml 提供了 <span class="mono">ggml_tensor_overhead()</span> 帮你算"每个张量的元数据要占多少字节"，建图时常按 <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> 个节点的规模留余量。
</div>
<p>举个具体感受："元数据"到底有多轻？一个 <span class="mono">ggml_tensor</span> 结构加上对象头，<span class="mono">ggml_tensor_overhead()</span> 量出来不过几百字节。
就算一张图有上千个张量，元数据加起来也才几百 KB——和动辄几个 GB 的<strong>权重数据</strong>相比，几乎可以忽略。这再次印证了"轻元数据 / 重数据"的分野：
在 <span class="mono">no_alloc=true</span> 下，<span class="mono">ggml_context</span> 这块 arena 只需开<strong>几 MB</strong> 装下整张图的骨架就绰绰有余，真正吃内存的数据另有去处。</p>
<p>把两种分配方式并排一看，arena 的好处就很直观了：</p>
<div class="cols">
  <div class="col"><h4>每张量各自 malloc（ggml <strong>不</strong>这么做）</h4><p>上千次系统调用，开销高；小块散落各处、易碎片；释放要一个个 free，容易漏；内存不连续、对缓存不友好。</p></div>
  <div class="col"><h4>arena + bump（ggml 的做法）</h4><p>一次大分配，开销摊薄；对象紧挨着、缓存友好；游标只进不退，分配快到只是"加个数"；收工一把 <span class="mono">ggml_free</span> 全清。</p></div>
</div>
<p>还有个容易被忽略的细节：bump 分配<strong>天然带来确定性</strong>。因为对象严格按建立顺序一个挨一个排开，<strong>同样的建图代码，每次跑出来的内存布局都一模一样</strong>，
这对调试、复现、以及后端按固定偏移读写都很有用。相比之下，<span class="mono">malloc</span> 返回的地址是<strong>不可预测</strong>的，每次运行都可能不同。
ggml 这种"<strong>可预测的连续布局</strong>"，是它能把一张图整体搬到别的内存（比如先在 CPU 上规划、再映射到 GPU 缓冲）的隐形前提。</p>

<h2>no_alloc：只建元数据，不占数据</h2>
<p>回头看 <span class="mono">ggml_init_params</span> 里那个 <span class="mono">no_alloc</span>。当它为 <span class="mono">true</span> 时，ggml 在 arena 里
<strong>只给张量的"元数据"留位</strong>（type、ne、nb、op、src 这些，L05 讲过），<strong>不为张量的"数据"（那一大片浮点数）分配缓冲</strong>。</p>
<div class="cols">
  <div class="col"><h4>no_alloc = false</h4><p>arena 里给元数据<strong>和</strong>数据缓冲都留位——张量能直接装下那片浮点数据。</p></div>
  <div class="col"><h4>no_alloc = true</h4><p>只给<strong>元数据</strong>（type/ne/nb/op/src）留位，数据缓冲留给后端——"先描述、后分配"。</p></div>
</div>
<p>为什么要这样？因为很多时候，我们想做的只是<strong>"先把计算图搭出来"</strong>——这一步只需要知道每个张量的形状和依赖关系，根本还用不到真正的数据内存。
等图建好、看清全貌，再交给<strong>后端</strong>统一分配真正的数据缓冲（这正是 L09 惰性建图、L10 内存复用的前提）。所以 <span class="mono">no_alloc=true</span>
是"<strong>先描述、后分配</strong>"这套玩法的开关，你会在 llama.cpp 加载模型、搭计算图时反复见到它。</p>
<p>把这一课的内存观<strong>收个尾</strong>：一次完整的使用，是这样一条线——<span class="mono">ggml_init</span> 圈地（拿到 arena）-&gt; 在 ctx 里
<strong>建张量、搭计算图</strong>（bump 切元数据）-&gt; 交给后端<strong>分配真正的数据并执行</strong>（L09、L10）-&gt; <span class="mono">ggml_free</span> 一把清空。
你会发现，<span class="mono">ggml_context</span> 始终扮演"<strong>轻量的脚手架</strong>"：它让搭建过程几乎不花内存、不碎不漏，把真正的重活（几 GB 的权重数据）
留到看清全局之后、由更懂硬件的后端来扛。地基只负责"把架子稳稳搭起来"，这正是它该有的样子。</p>
<p>所以这一课你真正要带走的，不是几个函数名，而是一个<strong>心智模型</strong>：ggml 里所有张量、所有图，都<strong>住在某个 <span class="mono">ggml_context</span> 的 arena 里</strong>；
它们靠 bump 一个挨一个排开、不单独 malloc；元数据在 ctx 里很轻，数据在后端缓冲里很重；用完整块一清。带着这个模型，下一课我们就去看：在这块 arena 上建起来的张量，是怎么靠 op/src 串成一张<strong>计算图</strong>的。</p>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么不直接 malloc / new 一个个分配张量？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>一次推理会建出成千上万个张量。如果每个都单独向系统申请内存，会有三个麻烦：<strong>分配器开销</strong>（每次 malloc/free 都有成本）、
    <strong>内存碎片</strong>（零散的小块散落各处）、以及<strong>释放繁琐</strong>（要一个个记得 free）。</p>
    <p>arena + bump 把这些一举解决：<strong>一次大分配</strong>摊薄了申请成本；对象在内存里<strong>紧挨着</strong>，对 CPU 缓存友好；
    收工时 <span class="mono">ggml_free</span> <strong>一把全清</strong>，不会漏。代价是你得<strong>预估池子大小</strong>——但对"建一张图"这种规模可预测的场景，这点代价非常划算。</p>
    <p>其实这种取舍在系统软件里很普遍：<strong>通用分配器</strong>（malloc）什么场景都能用，但什么都不特别快；<strong>arena</strong> 放弃了"随时单独释放任意一块"的灵活性，
    换来分配近乎免费、释放一步到位。ggml 之所以敢用 arena，正因为它的使用模式恰好匹配——一张图里的张量<strong>同生共死</strong>，    要么一起留着算，要么算完一起丢，几乎不存在"中途单独删一个张量"的需求。把通用工具换成贴合场景的专用工具，是性能工程里最常见的提速手法之一。早年很多游戏引擎、编译器内部都用同一招管理临时对象，原理和 ggml 这里如出一辙。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> arena 不够会怎样？怎么估它的大小？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>游标撞到 <span class="mono">mem_size</span> 边界时，ggml 不会偷偷扩容，而是<strong>当场报错失败</strong>（调试版会直接 <span class="mono">abort</span>，发布版则返回空指针并打印警告；无论哪种，都<strong>不悄悄换语义</strong>）。
    所以 <span class="mono">mem_size</span> 必须事先估够。</p>
    <p>估法也不神秘：<strong>元数据部分</strong> ≈ 张量个数 × <span class="mono">ggml_tensor_overhead()</span>（外加对象头）；
    <strong>数据部分</strong>（若 <span class="mono">no_alloc=false</span>）≈ 各张量字节数之和（用 L05 的 <span class="mono">ggml_nbytes</span> 思路）。
    建图常按 <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> 个节点留出余量，省得精打细算。值得一提的是，这个"宁可一次开大、也不要中途不够"的态度，和它"超限直接 abort"的刚硬是一致的：ggml 把"内存够不够"这件事<strong>前置到建池子那一刻</strong>，让后面的分配永远稳稳当当、没有意外。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> ggml_context 的内存和"后端内存"是一回事吗？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>不是，这是新手最容易混的一点。<span class="mono">ggml_context</span> 的 arena 在很多场景里只放<strong>元数据</strong>（配合 <span class="mono">no_alloc=true</span>）；
    而张量真正的<strong>数据</strong>，是由<strong>后端缓冲</strong>（如 CPU 内存、CUDA 显存）来分配的——那是 L10 里 ggml-alloc 的活儿。</p>
    <p>换句话说，存在<strong>两层内存</strong>：一层是 ctx 里轻量的"图骨架/元数据"，另一层是后端里厚重的"张量数据"。把这两层分开，正是 ggml 能
    "<strong>在 CPU 上搭好一张图、再把数据分配到 GPU 上去算</strong>"的关键。后面两课会把这条线接上。</p>
    <p>一个好记的划分：<strong>ctx 管"图长什么样"，后端管"数有多少"</strong>。前者轻、可预测、用 arena 一把搭一把清；后者重、要复用、由 ggml-alloc 精打细算（L10）。
    新手只要记住"看到 <span class="mono">ggml_context</span> 别以为权重就在里面"，就避开了八成的内存困惑。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">ggml_context</span> 是一个<strong>预分配的内存池（arena）</strong>；<span class="mono">ggml_init</span> 一次性备好，<span class="mono">ggml_free</span> 一次性释放。</li>
    <li><strong>no-malloc / bump 分配</strong>：<span class="mono">ggml_new_object</span> 在 arena 里往后推游标、就地放下，<strong>不为每个张量单独 malloc</strong>；对象串成 <span class="mono">ggml_object</span> 链表。</li>
    <li>arena <strong>不自动扩容</strong>，超出即报错失败（调试版 <span class="mono">abort</span>、发布版返回 NULL+警告）；用 <span class="mono">ggml_tensor_overhead()</span> 估大小，图常按 <span class="mono">GGML_DEFAULT_GRAPH_SIZE=2048</span> 留余量。</li>
    <li><span class="mono">no_alloc=true</span> 时<strong>只建元数据、不占数据</strong>，为"先建图、后由后端分配"铺路。</li>
    <li>存在<strong>两层内存</strong>：ctx 的元数据 arena，与后端的张量数据缓冲（L10）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  用"<strong>一次大分配 + 内部 bump</strong>"替代"成千上万次 malloc"——简单、快、可整体迁移、一把释放。正是这块轻量的内存池，让 ggml 能把一张计算图
  几乎零成本地<strong>搭起来</strong>，再整体交给某个后端去<strong>分配数据、执行</strong>。地基朴素，却撑起了上面所有的精巧——记住"一次圈地、内部挪车位、用完整清"这三步，你就握住了 ggml 内存管理的全部要义。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Part 2 gave you the <strong>intuition</strong> for tensors, quantization, and backends; from Part 3 on, we take the ggml engine apart. First stop: <strong>memory</strong> -
how does ggml manage the memory of thousands of tensors? The answer is surprisingly plain: a <strong>memory pool</strong> called <span class="mono">ggml_context</span>,
plus a "never scatter-malloc" allocation philosophy. Get this layer and the compute graph, execution, and operators all have ground to stand on.
</p>
<p style="color:var(--muted);margin-top:.4rem">First, the <strong>problem at hand</strong>: one inference conjures up thousands of tensors out of thin air - weights, intermediate activations, the KV cache,
all tensors. If each one went to the operating system for memory, the overhead of "request, bookkeep, return" alone would drag the whole engine down. ggml's answer is:
<strong>don't buy retail, buy wholesale</strong>. It fences off one big block at once and carves it carefully inside - that block is <span class="mono">ggml_context</span>. This
lesson is about how it fences, how it carves, and why it is designed this way.</p>
<p>While we are at it, let me name the technique: this way of "<strong>grab one big block, carve internally, free wholesale</strong>" has a common name in systems
programming - an <strong>arena (or region) allocator</strong>, also called a <strong>linear allocator</strong>. It is not ggml's invention but a well-worn trick in
high-performance code; ggml just applies it to the perfectly-suited scenario of "managing all the tensors of one compute graph". When we say arena below, we mean that big
block held by <span class="mono">ggml_context</span>.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  <span class="mono">ggml_context</span> is like a <strong>pre-marked parking lot</strong>: at opening you fence off a whole plot at once (<span class="mono">mem_size</span>),
  then each car you park (every tensor or object) just moves forward one slot (a bump cursor), <strong>without running to the office to requisition land each time</strong>
  (malloc). At close, the whole lot clears in one go - clean and tidy. Swapping "keep calling the office" for "shuffle slots in your own fenced lot" is the whole essence of
  ggml memory management. Hold onto this image and you are halfway through the lesson.
</div>

<h2>The three foundation pieces</h2>
<p>At ggml's lowest layer there are really just three things buckled together: a <strong>config</strong> (how big a pool you want), the <strong>pool itself</strong>, and the
<strong>objects</strong> lined up one after another inside it (each wrapping a tensor or a compute graph). First, how they relate:</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">config</span><span class="name">ggml_init_params</span></div><div class="ld">mem_size (how big) · mem_buffer (whose memory) · no_alloc (reserve data space?)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">pool</span><span class="name">ggml_context</span></div><div class="ld">holds one whole arena, tracking the cursor (how far used) and the object list head/tail</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">objects</span><span class="name">ggml_object -&gt; ggml_tensor / ggml_cgraph</span></div><div class="ld">objects laid out one after another, each wrapping a tensor or a graph</div></div>
</div>
<p>The keyword here is <strong>"pool"</strong>: ggml does not scatter-request memory per tensor; it <strong>grabs one big block once and carves inside it</strong>. The next
three sections walk through these three pieces.</p>
<p>Let me plant a contrast that runs through the whole lesson: <strong>"metadata" and "data" are two different things</strong>. Metadata is "<strong>what this tensor looks
like</strong>" - its shape ne, strides nb, type, and op/src (the fields from L05) - adding up to mere tens or hundreds of bytes; data is "<strong>the big slab of actual
floats</strong>", where one weight matrix might be tens of MB. The <span class="mono">ggml_context</span> arena very often holds <strong>only metadata</strong>, leaving the heavy
data to the backend. Read on with this "light metadata / heavy data" split in mind and many design choices fall into place.</p>

<h2>ggml_context: a memory pool</h2>
<p>It all starts with <span class="mono">ggml_init(params)</span>. You tell ggml three things in <span class="mono">ggml_init_params</span> and it hands back a
<span class="mono">ggml_context</span> - a ready memory pool:</p>
<pre class="code"><span class="cm">// simplified from ggml/include/ggml.h</span>
<span class="kw">struct</span> ggml_init_params {
    size_t mem_size;     <span class="cm">// total arena size</span>
    <span class="kw">void</span> * mem_buffer;   <span class="cm">// NULL = ggml allocates this block internally</span>
    <span class="kw">bool</span>   no_alloc;     <span class="cm">// true = build tensor "metadata" only, no data buffer</span>
};

ctx = <span class="fn">ggml_init</span>(params);   <span class="cm">// grab the whole arena at once</span>
<span class="cm">// ... build many tensors in ctx ...</span>
<span class="fn">ggml_free</span>(ctx);             <span class="cm">// free it all at once</span></pre>
<div class="flow">
  <div class="node"><div class="nt">ggml_init(params)</div><div class="nd">claim one arena<br>(mem_size)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">ggml_context</div><div class="nd">pool: arena + cursor<br>+ object list</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">ggml_free(ctx)</div><div class="nd">return the whole<br>block at once</div></div>
</div>
<p><span class="mono">ggml_init</span> is straightforward: it allocates a little space for the <span class="mono">ggml_context</span> management struct itself, then <strong>prepares
that big arena</strong> - if you passed a <span class="mono">mem_buffer</span> it uses your memory; pass <span class="mono">NULL</span> and it does
<span class="mono">ggml_aligned_malloc(mem_size)</span> itself (see <span class="mono">ggml/src/ggml.c</span>). <span class="mono">ggml_free</span> returns the whole arena
(only freeing the block if ggml allocated it itself, i.e. <span class="mono">mem_buffer_owned</span>). That "<strong>you can pass in mem_buffer</strong>" matters: ggml can work
on memory someone else gave it, handy for embedding into various environments or reusing one buffer to build graphs repeatedly.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A word on <strong>alignment</strong>. <span class="mono">ggml_aligned_malloc</span> wants not just any memory but memory <strong>aligned to a particular boundary</strong> - because backend SIMD instructions (AVX, NEON, from L07) often require aligned addresses to read efficiently or even correctly. Each object carved inside the arena is also aligned to <span class="mono">GGML_MEM_ALIGN</span>. Think of the arena as a <strong>ruler with tick marks</strong>: every object lands on a tidy tick rather than wherever - and that bit of tidiness buys speed at compute time.
</div>
<p>So strictly, the difference between <span class="mono">ggml_aligned_malloc</span> and plain <span class="mono">malloc</span> is just
"<strong>alignment</strong>": plain malloc only guarantees big-enough, not that the address falls on a boundary; ggml's memory must start at a multiple of some alignment (16 or
32 bytes), so the backend can confidently use aligned SIMD loads to move a batch at once. This small thing reflects ggml's bias of "<strong>always making way for backend
compute</strong>".</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  One more often-asked question: <strong>how many <span class="mono">ggml_context</span>s can a program open?</strong> The answer is <strong>several</strong>, and that is common. For instance, one ctx for the <strong>model weights</strong> (long-lived, present the whole inference) and another for <strong>each step's compute graph</strong> (short-lived, cleared once computed). Putting things of different lifetimes into different pools lets <strong>the long-lived stay and the short-lived clear fast</strong>, without interfering - another convenience of the arena model: one <span class="mono">ggml_free</span> precisely reclaims a whole batch of same-lifetime objects.
</div>
<p>This mechanism is everywhere in llama.cpp. When loading a model, the loader first <strong>estimates how big a metadata arena</strong> it needs from the tensor count and sizes
in the GGUF header, <span class="mono">ggml_init</span>s a (usually <span class="mono">no_alloc=true</span>) context, and registers every weight tensor in it; the weights' real data is
taken up by backend buffers (or even mmap'd straight from the file, see L13). Each inference step uses another context to temporarily build that step's compute graph, discarded
once done. So you can picture <span class="mono">ggml_context</span> as ggml's most basic <strong>"workbench"</strong>: to work, set up a bench; when done, clear bench and materials
together. Understand it and you understand "where" all of ggml's data structures live.</p>

<h2>no-malloc: bump allocation</h2>
<p>Once it has the arena, every tensor or graph it builds does <strong>not</strong> go back to the system for memory; it just "<strong>pushes forward one slot</strong>" inside
the arena. That is bump (pointer-bump) allocation: keep a "how far used" cursor, and for each object move the cursor right and place it in situ:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>bump allocation</b>: all objects line up in the same arena, the cursor only moves forward</div>
  <div class="cells"><span class="lab">arena</span><span class="cell hl">obj1</span><span class="cell hl">obj2</span><span class="cell hl">obj3</span><span class="cell dim">free …</span><span class="lab">cursor -&gt;</span></div>
  <div class="cells"><span class="lab">list</span><span class="cell dim">begin</span><span class="sep">-&gt;</span><span class="cell">obj1</span><span class="sep">-&gt;</span><span class="cell">obj2</span><span class="sep">-&gt;</span><span class="cell">obj3</span><span class="sep">-&gt;</span><span class="cell dim">end</span><span class="lab">begin/end are head/tail pointers</span></div>
</div>
<p>Each object is prefixed with a tiny <span class="mono">ggml_object</span> header (recording its offset, size, and a pointer to the next object); they form a linked list,
and the cursor always rests at the end of the last object. A new object is carved from the cursor forward. As pseudocode it is clear at a glance:</p>
<p>A word more on what that <span class="mono">ggml_object</span> header holds: its <strong>offset</strong> <span class="mono">offs</span> in the arena, the <strong>size</strong>
<span class="mono">size</span> it occupies, a pointer <span class="mono">next</span> to the <strong>next object</strong>, plus a type field marking "tensor or graph".
<span class="mono">ggml_context</span> itself tracks the list head/tail (<span class="mono">objects_begin</span> / <span class="mono">objects_end</span>) and the object count
<span class="mono">n_objects</span>. With a tail pointer, "append at the end" is O(1) - another face of why bump is fast.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  And why does this "only-add, never-remove" cursor work? Because <strong>the build phase is almost append-only</strong> - you are constructing the whole compute graph in one go, rarely needing to free a single tensor mid-way. With no "rob Peter to pay Paul" need, the simplest allocator (a forward-pushing cursor) suffices, sparing all the complexity of tracking free blocks and merging fragments. This is a classic "<strong>trade the scenario's traits for the allocator's utter simplicity</strong>": not until L10 actually needs to <strong>reuse</strong> memory does a smarter allocator come in; here in the build phase, plain bump is the best fit.
</div>
<pre class="code"><span class="cm"># cf. ggml_new_object / ggml_new_tensor_impl in ggml/src/ggml.c</span>
<span class="kw">def</span> <span class="fn">new_object</span>(ctx, size):
    cur = ctx.objects_end.offs + ctx.objects_end.size   <span class="cm"># current cursor</span>
    <span class="kw">if</span> cur + size &gt; ctx.mem_size:                       <span class="cm"># pool is out of room</span>
        abort("arena out of space")                     <span class="cm"># no growth, just fail!</span>
    obj = place_at(ctx.mem_buffer + cur)                <span class="cm"># place in situ</span>
    link_into(ctx.objects, obj)                         <span class="cm"># append to the list</span>
    <span class="kw">return</span> obj</pre>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  Two points here. One, <strong>a tensor's metadata (the <span class="mono">ggml_tensor</span> struct) and its data buffer are both carved from this same arena</strong> - there is no "malloc once per tensor". Two, <strong>the arena does not auto-grow</strong>: the moment the cursor hits the edge, ggml <span class="mono">abort</span>s. So the user must <strong>size the pool large enough up front</strong>. ggml offers <span class="mono">ggml_tensor_overhead()</span> to compute "how many bytes each tensor's metadata takes", and graphs commonly leave headroom for <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> nodes.
</div>
<p>For a concrete feel: just how light is "metadata"? A <span class="mono">ggml_tensor</span> struct plus object header, as measured by
<span class="mono">ggml_tensor_overhead()</span>, is only a few hundred bytes. Even a graph with thousands of tensors totals just a few hundred KB of metadata - next to the multiple
GB of <strong>weight data</strong>, practically nothing. This again confirms the "light metadata / heavy data" split: under <span class="mono">no_alloc=true</span>, the
<span class="mono">ggml_context</span> arena need only be a <strong>few MB</strong> to comfortably hold the whole graph's skeleton; the truly memory-hungry data lives elsewhere.</p>
<p>Put the two allocation styles side by side and the arena's benefits are obvious:</p>
<div class="cols">
  <div class="col"><h4>malloc per tensor (ggml does <strong>not</strong> do this)</h4><p>thousands of syscalls, high overhead; small blocks scattered everywhere, prone to fragmentation; freeing means one-by-one, easy to leak; non-contiguous memory, cache-unfriendly.</p></div>
  <div class="col"><h4>arena + bump (ggml's way)</h4><p>one big allocation, overhead amortized; objects adjacent, cache-friendly; the cursor only moves forward, allocation as fast as "add a number"; at close, one <span class="mono">ggml_free</span> clears all.</p></div>
</div>
<p>One easily-missed detail: bump allocation <strong>naturally yields determinism</strong>. Because objects line up strictly in creation order, <strong>the same graph-building
code produces the exact same memory layout every run</strong> - useful for debugging, reproduction, and for the backend reading/writing at fixed offsets. By contrast,
<span class="mono">malloc</span> returns <strong>unpredictable</strong> addresses that may differ each run. ggml's "<strong>predictable contiguous layout</strong>" is the invisible
premise that lets it move a whole graph to other memory (e.g. plan on CPU first, then map to a GPU buffer).</p>

<h2>no_alloc: metadata only, no data</h2>
<p>Back to that <span class="mono">no_alloc</span> in <span class="mono">ggml_init_params</span>. When it is <span class="mono">true</span>, ggml reserves space in the arena
<strong>only for a tensor's "metadata"</strong> (type, ne, nb, op, src - from L05), <strong>not for the tensor's "data" (that big slab of floats)</strong>.</p>
<div class="cols">
  <div class="col"><h4>no_alloc = false</h4><p>the arena reserves room for metadata <strong>and</strong> the data buffer - tensors can hold that float data directly.</p></div>
  <div class="col"><h4>no_alloc = true</h4><p>only <strong>metadata</strong> (type/ne/nb/op/src) gets room; the data buffer is left to the backend - "describe first, allocate later".</p></div>
</div>
<p>Why? Because often all we want is to <strong>"build the compute graph first"</strong> - that step only needs each tensor's shape and dependencies, not real data memory yet.
Once the graph is built and the whole picture is clear, we hand it to the <strong>backend</strong> to allocate the real data buffers in one pass (exactly the premise of L09's
lazy build and L10's memory reuse). So <span class="mono">no_alloc=true</span> is the switch for the "<strong>describe first, allocate later</strong>" approach, which you will
see again and again as llama.cpp loads models and builds graphs.</p>
<p>To <strong>wrap up</strong> this lesson's view of memory: a full usage is one line - <span class="mono">ggml_init</span> fences the land (gets the arena) -&gt; <strong>build
tensors and the compute graph</strong> in the ctx (bump-carve metadata) -&gt; hand to the backend to <strong>allocate real data and execute</strong> (L09, L10) -&gt;
<span class="mono">ggml_free</span> clears it all. You will notice <span class="mono">ggml_context</span> always plays the role of a <strong>lightweight scaffold</strong>: it makes the
building process cost almost no memory, no fragmentation, no leaks, leaving the truly heavy lifting (the multi-GB weight data) for after the whole picture is clear, carried by a
more hardware-aware backend. The foundation only "holds the frame steady" - exactly as it should.</p>
<p>So what you should really take away is not a few function names but a <strong>mental model</strong>: every tensor and every graph in ggml <strong>lives in some
<span class="mono">ggml_context</span> arena</strong>; they line up via bump, no individual malloc; metadata is light in the ctx, data is heavy in the backend buffer; clear the
whole block when done. With this model, the next lesson goes to see how the tensors built on this arena are strung via op/src into a <strong>compute graph</strong>.</p>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why not just malloc / new each tensor individually? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>One inference builds thousands of tensors. Requesting memory individually for each brings three headaches: <strong>allocator overhead</strong> (every malloc/free
    has a cost), <strong>fragmentation</strong> (scattered small blocks all over), and <strong>tedious freeing</strong> (you must remember to free each one).</p>
    <p>arena + bump solves all at once: <strong>one big allocation</strong> amortizes the request cost; objects sit <strong>adjacent</strong> in memory, cache-friendly; and at
    close <span class="mono">ggml_free</span> <strong>clears everything in one shot</strong>, nothing leaked. The cost is having to <strong>estimate the pool size</strong> - but for
    the predictable scale of "building one graph", that is a very worthwhile trade.</p>
    <p>This trade is common in systems software: a <strong>general allocator</strong> (malloc) works for any scenario but is not especially fast at any; an <strong>arena</strong>
    gives up the flexibility of "free any one block anytime" in exchange for near-free allocation and one-step release. ggml dares to use an arena precisely because its usage
    pattern fits - tensors in a graph <strong>live and die together</strong>, kept to compute or dropped at once, with almost no need to "delete one tensor mid-way". Swapping a
    general tool for a scenario-fitting specialized one is one of the most common speedups in performance engineering; old game engines and compilers used the same trick for
    temporary objects, on the very same principle as ggml here.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What if the arena runs out? How do you size it? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>When the cursor hits the <span class="mono">mem_size</span> edge, ggml does not silently grow; it <strong>fails on the spot</strong> (debug builds <span class="mono">abort</span> outright; release builds return a NULL pointer and print a warning - either way it does <strong>not</strong> quietly
    change semantics). So <span class="mono">mem_size</span> must be estimated big enough up front.</p>
    <p>The estimate is not mysterious: the <strong>metadata part</strong> ~= number of tensors x <span class="mono">ggml_tensor_overhead()</span> (plus object headers);
    the <strong>data part</strong> (if <span class="mono">no_alloc=false</span>) ~= the sum of each tensor's bytes (the L05 <span class="mono">ggml_nbytes</span> idea). When building a
    graph, people commonly just leave headroom for <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> nodes rather than count precisely. Notably, this "rather open big than
    fall short mid-way" attitude matches its hard-line "abort on overflow": ggml <strong>front-loads the "is there enough memory" question to pool-creation time</strong>, so every
    later allocation is rock-steady, no surprises.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Is ggml_context's memory the same as "backend memory"? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>No - this is the thing beginners most often conflate. The <span class="mono">ggml_context</span> arena in many scenarios holds only <strong>metadata</strong> (with
    <span class="mono">no_alloc=true</span>); a tensor's real <strong>data</strong> is allocated by a <strong>backend buffer</strong> (CPU memory, CUDA VRAM) - that is L10's
    ggml-alloc work.</p>
    <p>In other words, there are <strong>two layers of memory</strong>: a light "graph skeleton / metadata" layer in the ctx, and a heavy "tensor data" layer in the backend.
    Separating these two is exactly what lets ggml "<strong>build a graph on the CPU, then allocate the data on the GPU to compute</strong>". The next two lessons connect this line.</p>
    <p>An easy division to remember: <strong>the ctx manages "what the graph looks like", the backend manages "how much data there is"</strong>. The former is light, predictable,
    set up and cleared with an arena; the latter is heavy, must be reused, and is carefully managed by ggml-alloc (L10). Beginners need only remember "seeing
    <span class="mono">ggml_context</span> does not mean the weights are inside it" to dodge eighty percent of memory confusion.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">ggml_context</span> is a <strong>pre-allocated memory pool (arena)</strong>; <span class="mono">ggml_init</span> sets it up once, <span class="mono">ggml_free</span> releases it once.</li>
    <li><strong>no-malloc / bump allocation</strong>: <span class="mono">ggml_new_object</span> pushes the cursor forward in the arena and places in situ, <strong>no per-tensor malloc</strong>; objects form a <span class="mono">ggml_object</span> linked list.</li>
    <li>The arena <strong>does not auto-grow</strong>; overflow fails on the spot (debug <span class="mono">abort</span>, release returns NULL+warning); size it with <span class="mono">ggml_tensor_overhead()</span>, graphs leave headroom for <span class="mono">GGML_DEFAULT_GRAPH_SIZE=2048</span>.</li>
    <li><span class="mono">no_alloc=true</span> builds <strong>metadata only, no data</strong>, paving the way for "build the graph first, let the backend allocate later".</li>
    <li>There are <strong>two memory layers</strong>: the ctx metadata arena, and the backend tensor-data buffer (L10).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Replacing "thousands of mallocs" with "<strong>one big allocation + internal bump</strong>" - simple, fast, wholesale-movable, freed in one shot. It is this lightweight
  memory pool that lets ggml <strong>build</strong> a compute graph at near-zero cost, then hand it wholesale to a backend to <strong>allocate data and execute</strong>. A plain
  foundation, yet it carries all the cleverness above it - remember "fence once, shuffle slots inside, clear all when done" and you hold the whole gist of ggml memory management.
</div>
""",
}


LESSON_09 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课你知道了张量从 <span class="mono">ggml_context</span> 的内存池里来。但这里有个会让很多人意外的事实：当你写下
<span class="mono">c = ggml_mul_mat(a, b)</span> 时，那个矩阵乘<strong>根本没有发生</strong>——ggml 只是默默记下了一句"c 是由 a 和 b 经矩阵乘得到的"。
这一课就讲这种"<strong>先记账、不动手</strong>"的惰性建图，它是整个 ggml 引擎最精巧的设计之一。
</p>
<p style="color:var(--muted);margin-top:.4rem">这件事为什么重要？因为它<strong>颠覆了你对"调用一个函数就该立刻得到结果"的直觉</strong>。在 ggml 里，调用算子更像是在
<strong>下订单、写清单</strong>，而不是当场交付货物。整张神经网络的前向过程，会先被完整地"<strong>记成一张图</strong>"，
然后才在某个时刻被一次性地、有计划地算出来。理解了这个"记账在前、计算在后"的两段式，你才算真正看懂 ggml 为什么能又快又省、还能跑遍各种硬件。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  这就像<strong>写菜谱</strong>而不是马上做菜：你先把"先切菜、再热油、然后下锅、最后装盘"的步骤和它们之间的先后依赖，
  一条条写成一张流程图；等到真要开火（执行）时，才照着这张图把菜做出来。<strong>建图 = 写菜谱，执行 = 照着做</strong>。
  好处是：菜谱写好后，你可以先通读一遍、优化火候顺序、甚至换个厨房来做——这些都是"先写下来"才换得到的余地。
</div>
<p>顺便说一句，"惰性"（lazy）在编程里是个褒义词，不是"偷懒"的意思，而是"<strong>不到非算不可的那一刻，绝不提前算</strong>"。这种推迟，往往能让程序在真正动手前
看清全局、做出更聪明的安排。ggml 把这个思想用在了计算图上：所有算子调用都先攒着，等图齐了再一次性算——这正是它高效的源头之一。</p>

<h2>一次算子调用，到底发生了什么</h2>
<p>先把最核心的误解纠正过来。在很多框架里，<span class="mono">c = a @ b</span> 写下去，乘法立刻就算完了，<span class="mono">c</span> 里装着结果数字。
但在 ggml 里完全不是这样：<span class="mono">ggml_mul_mat(a, b)</span> 返回的 <span class="mono">c</span>，此刻<strong>还是一个空壳</strong>——
它知道自己的形状、知道自己将由谁经什么运算得到，但<strong>里面一个数都还没算</strong>。它记下的，只是"<strong>身世</strong>"：</p>
<p>请特别留意"<strong>反向</strong>"这两个字。在你脑子里画神经网络时，箭头通常是<strong>从输入流向输出</strong>的（数据怎么走）；
但 ggml 在张量里存的指针方向恰好<strong>相反</strong>——是结果<strong>指回</strong>它的输入。为什么反着存？因为执行时 ggml 最关心的问题是"<strong>要算出这个结果，我得先有谁</strong>"，
顺着结果往回找输入，正好一步到位。这就像顺着一个人往上查父母、再查祖父母，比从祖先往下逐代点名要直接得多。这个"<strong>从输出回溯输入</strong>"的方向，
是后面建图、求导都反复用到的关键。</p>
<div class="flow">
  <div class="node"><div class="nt">a</div><div class="nd">输入张量</div></div>
  <div class="node"><div class="nt">b</div><div class="nd">输入张量</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">c = mul_mat(a, b)</div><div class="nd">c.op = MUL_MAT<br>c.src = [a, b]</div></div>
</div>
<p>看那个高亮的结果 <span class="mono">c</span>：它的 <span class="mono">op</span> 字段记着"我是用矩阵乘得到的"，<span class="mono">src[0]</span> 和
<span class="mono">src[1]</span> 两个指针分别指回 <span class="mono">a</span> 和 <span class="mono">b</span>（注意箭头方向——是结果<strong>指回</strong>输入，
所以叫"反向指针"）。这两样东西，L05 介绍 <span class="mono">ggml_tensor</span> 字段时就见过，当时只说"记录它怎么来的"，现在你看到它真正的用途了。
把源码摊开看，每个算子函数都是同一个套路：</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里值得停一下，体会一下这个设计有多统一：无论是矩阵乘、加法、归一化还是注意力，几百个算子函数<strong>清一色都是"建张量 + 填 op/src + 返回"</strong>这个三步模板。正因为如此一致，ggml 才能用<strong>同一套建图、同一套执行</strong>机制处理所有算子——加一个新算子，主要就是定义一个新的 <span class="mono">op</span> 枚举值、再写它的形状推导和计算实现，建图这一环完全不用改。这种"<strong>用统一模板装下千变万化</strong>"的克制，是 ggml 代码读起来不乱的重要原因。
</div>
<pre class="code"><span class="cm">// 简化自 ggml/src/ggml.c 的 ggml_mul_mat</span>
<span class="kw">struct</span> ggml_tensor * <span class="fn">ggml_mul_mat</span>(ctx, a, b) {
    result = <span class="fn">ggml_new_tensor</span>(ctx, GGML_TYPE_F32, ...);  <span class="cm">// 只建一个空的结果张量</span>
    result-&gt;op     = GGML_OP_MUL_MAT;                   <span class="cm">// 记下"我是怎么来的"</span>
    result-&gt;src[0] = a;                                 <span class="cm">// 记下输入 1</span>
    result-&gt;src[1] = b;                                 <span class="cm">// 记下输入 2</span>
    <span class="kw">return</span> result;                                      <span class="cm">// 直接返回, 一个乘法都没做!</span>
}</pre>
<p><span class="mono">ggml_add</span>、<span class="mono">ggml_rms_norm</span>、<span class="mono">ggml_soft_max</span>……几乎所有算子都长这样：
<strong>建一个结果张量、填好 op 和 src、返回</strong>。真正的浮点运算，要等到后面"执行"时才发生（下一课的事）。
有些算子还需要额外参数（比如 rope 的旋转角度、softmax 的缩放系数），这些会用 <span class="mono">ggml_set_op_params_*</span> 之类的辅助函数
存进结果张量的 <span class="mono">op_params</span> 里——但同样，只是"记下来"，不计算。</p>
<p>换个角度想，这个"空壳"结果张量其实是一张<strong>借条（IOU）</strong>：它向你承诺"将来我会等于 a 乘 b 的结果"，但<strong>现在还没兑现</strong>。
你可以拿这张借条继续往下写——把它当作下一个算子的输入，再得到一张新借条；如此层层叠叠，直到写出最终输出。整个过程里，<strong>没有任何真实数字被算出来</strong>，
你手里攒下的，是一摞环环相扣的借条。等到"执行"那一刻，ggml 才会顺着这摞借条，从最底层开始，把每一张都兑现成真实的数据。这种"<strong>先开借条、后统一兑现</strong>"，
正是惰性（lazy）二字的含义，也是这一课从头到尾在反复打磨的那个核心直觉。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  这也解释了一个新手常踩的坑：在 ggml 里，<strong>建完图就去读结果张量的数据，是读不到东西的</strong>——借条还没兑现呢。必须先把图交给后端执行（下一课），结果张量的 <span class="mono">data</span> 才会被填上真实数值。把"建图"和"执行"分成两个明确的阶段，是用好 ggml API 的第一课。
</div>

<h2>把张量串成一张图</h2>
<p>一个算子记下两三个 src，看起来不起眼；但当你把整个模型的前向过程都写出来，这些 src 指针就<strong>层层相扣，连成了一张有向图</strong>。
看一个最小的例子——两层线性变换 <span class="mono">y = W2 · (W1 · x)</span>：</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>叶子：x, W1, W2</h4>
    <p>输入和权重，它们的 <span class="mono">op</span> 是 NONE（没有"来历"，是图的起点）。</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>h = mul_mat(W1, x)</h4>
    <p>第一个算子结果，<span class="mono">src = [W1, x]</span>；它是个"节点"。</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>y = mul_mat(W2, h)</h4>
    <p>第二个算子结果，<span class="mono">src = [W2, h]</span>；注意它依赖上一步的 <span class="mono">h</span>。</p>
  </div></div>
</div>
<p>这张图的妙处在于：从输出 <span class="mono">y</span> 出发，顺着 src 指针往回走，就能<strong>找到算出它所需的一切</strong>——
y 依赖 W2 和 h，h 又依赖 W1 和 x。ggml 用 <span class="mono">ggml_build_forward_expand(graph, y)</span> 做的正是这件事：
从你指定的输出张量出发，<strong>沿 src 递归回溯，把所有依赖按"先算谁后算谁"的顺序（拓扑排序）收集进一张 <span class="mono">ggml_cgraph</span></strong>：</p>
<pre class="code"><span class="cm"># 对应 ggml/src/ggml.c 的 ggml_build_forward_expand / ggml_visit_parents_graph</span>
<span class="kw">def</span> <span class="fn">build_forward</span>(graph, t):
    <span class="kw">for</span> s <span class="kw">in</span> t.src:            <span class="cm"># 先把依赖都收进来</span>
        <span class="fn">build_forward</span>(graph, s)   <span class="cm"># 递归回溯</span>
    <span class="kw">if</span> t.op == NONE <span class="kw">and</span> <span class="kw">not</span> t.is_param:
        graph.leafs.append(t)     <span class="cm"># 输入/常量 -&gt; 叶子</span>
    <span class="kw">else</span>:
        graph.nodes.append(t)     <span class="cm"># 算子结果 -&gt; 节点(按依赖顺序排好)</span></pre>
<p>因为是"先递归收集依赖、再把自己放进去"，最后 <span class="mono">graph.nodes</span> 里的节点天然就是<strong>拓扑有序</strong>的：
排在前面的，一定不依赖排在后面的。这样执行时只要从头到尾依次算，每算一个节点，它的输入<strong>保证已经算好了</strong>。
<span class="mono">ggml_cgraph</span> 本身就是几个数组：<span class="mono">nodes</span>（算子结果）、<span class="mono">leafs</span>（输入/常量）、
计数 <span class="mono">n_nodes</span>/<span class="mono">n_leafs</span>、容量 <span class="mono">size</span>（默认 <span class="mono">GGML_DEFAULT_GRAPH_SIZE=2048</span>）。</p>
<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  "<strong>拓扑排序</strong>"这个词听起来唬人，其实道理就是一句大白话：<strong>要用到的东西，必须先准备好</strong>。做菜时你不能在切菜之前就下锅，算 <span class="mono">y</span> 之前必须先有 <span class="mono">h</span>。拓扑排序就是把所有步骤排成一个合法的先后顺序，让每一步用到的输入都在它之前已经备齐。一张图可能有<strong>不止一种</strong>合法顺序（比如两个互不依赖的分支谁先谁后都行），但只要满足"依赖在前"，执行起来结果就一样。ggml 的回溯式建图，自动帮你算出了这样一个合法顺序，你完全不用操心。它内部还用一个"已访问"集合避免把同一个张量重复收进图——当多个算子<strong>共享同一个输入</strong>时（这在神经网络里太常见了），那个输入只会被收一次、也只会被算一次。
</div>
<p>把这套机制放回真实的 llama.cpp 里看：加载一个模型后，每跑一步推理，llama.cpp 都会用一长串算子调用（embedding、几十层的注意力和 FFN、最后的输出投影）
<strong>搭出这一步的完整计算图</strong>——可能有上千个节点。这一大串调用，没有一个真的在算，全是在<strong>填 op/src、连依赖</strong>；
直到图搭完、交给后端，才一次性算出这一步的 logits。所以你之前学的"一次 decode 内部是先建图、再执行"（L03），到这里就有了精确的含义：
<strong>建图 = 这一串惰性的算子调用，执行 = 后端按拓扑序把图算完</strong>。</p>

<h2>nodes 与 leafs：图的两类居民</h2>
<p>建图时，ggml 把遇到的张量分成两类放好。判据很简单：看它有没有"来历"（<span class="mono">op</span> 是不是 NONE）：</p>
<table class="t">
  <tr><th>类别</th><th>是什么</th><th>判据</th><th>执行时</th></tr>
  <tr><td><strong>leafs（叶子）</strong></td><td>输入、权重、常量</td><td><span class="mono">op == NONE</span>（没有算子来历）</td><td>不计算，直接用它的数据</td></tr>
  <tr><td><strong>nodes（节点）</strong></td><td>算子的结果</td><td>有 <span class="mono">op</span>（如 MUL_MAT）</td><td>按拓扑序逐个计算</td></tr>
</table>
<p>用上面那个例子：<span class="mono">x</span>、<span class="mono">W1</span>、<span class="mono">W2</span> 是叶子（它们是给定的，不用算）；
<span class="mono">h</span>、<span class="mono">y</span> 是节点（要算出来）。执行引擎只对 <span class="mono">nodes</span> 逐个动手，<span class="mono">leafs</span> 提供原料即可。
源码里这套分类逻辑在 <span class="mono">ggml_visit_parents_graph</span>（<span class="mono">ggml/src/ggml.c</span>）：碰到 <span class="mono">op==NONE</span> 且不是参数的张量，
就当叶子；否则当节点。理解了这条线，你看 ggml 调试输出里那一长串 nodes/leafs 就不再陌生。</p>
<p>为什么非要把这两类分开？因为它们在执行时的<strong>待遇完全不同</strong>。叶子是"<strong>已知量</strong>"——权重早就从模型文件里加载好了，输入也是你给定的，
它们的数据现成就在那儿，执行引擎<strong>碰都不用碰</strong>，直接拿来当原料。节点才是"<strong>未知量</strong>"——要靠算子把输入加工出来，是执行引擎真正干活的地方。
把"现成的"和"待算的"分开放，执行时就一目了然：跳过所有叶子，只对节点从头到尾算一遍，整张图就算完了。</p>
<p>还有个容易混淆的小问题：同一个张量，会不会既是这张图的叶子、又是另一张图的节点？会的。比如某层的输出 <span class="mono">h</span>，在"算 h 的那张子图"里它是节点（要算出来），
但如果你把它当作另一段计算的<strong>给定输入</strong>，它就成了那段计算的叶子。叶子和节点不是张量的<strong>固有属性</strong>，而是它<strong>在当前这张图里扮演的角色</strong>——
是起点（叶子），还是中间/末端的产物（节点）。想通这一层，你对"图"的理解就更灵活了。这也呼应了上一节的"族谱"比喻：同一个人，在自己这一支里是后辈，在更年轻一辈眼里又是长辈，全看你以谁为参照。</p>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> "先建图、后执行"到底买到了什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>延迟计算看似多此一举，其实换来了三样宝贵的东西。<strong>其一，全局视野</strong>：建完图，ggml 能一眼看到"总共要算哪些、谁依赖谁"，
    于是能精打细算地<strong>复用内存</strong>（下一课的 ggml-alloc 正靠这个把峰值内存压到极低）、还能合并或重排算子。</p>
    <p><strong>其二，多后端通用</strong>：图只是"该算什么"的纯描述，不绑定任何硬件，于是<strong>同一张图能交给 CPU、CUDA、Metal 任意后端去执行</strong>（呼应 L01、L07）。
    <strong>其三，能反向求导</strong>：有了显式的依赖图，按链式法则反着走一遍就能自动算梯度（训练用）。这三样，都建立在"先不算、先记下来"之上。</p>
    <p>反过来想，如果<strong>不</strong>建图、边调用边算，会失去什么？你会陷入"<strong>只见树木、不见森林</strong>"：算每一步时都不知道后面还要算什么、哪些中间结果以后还用得到，
    于是只能<strong>保守地把每个中间结果都留着</strong>（内存爆炸），也<strong>没机会</strong>把相邻算子合并、或挑个更优的执行顺序。即时计算（eager）写起来直观，但把优化的余地全堵死了；
    惰性建图牺牲了一点"所见即所得"的直觉，换来的是<strong>一整张可供优化的蓝图</strong>。对追求极致性能的推理引擎来说，这笔交易非常值。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> src 反向指针，和 L05 说的那个 op/src 是一回事吗？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>是同一个，但这一课你才真正看到它的<strong>用途</strong>。L05 讲 <span class="mono">ggml_tensor</span> 字段时，只是告诉你"有 op 和 src 这两样，记录张量怎么来的"；
    当时它们还像是孤立的标签。</p>
    <p>到这一课，这些 src 指针<strong>串了起来</strong>：每个张量都记得自己的父节点，于是整张计算图，本质上就是"<strong>张量们靠 src 互相牵着手连成的一张网</strong>"。
    <span class="mono">op</span> 说"这一步做什么运算"，<span class="mono">src</span> 说"输入从哪来"——两者合起来，一个张量就既是"一块数据"，又是"图里的一个运算节点"。
    这个双重身份，是读懂 ggml 的关键。把"数据"和"图节点"这两个身份合在一个结构体里，是 ggml 区别于"先定义网络结构、再灌数据"那类框架的一个鲜明特点。</p>
    <p>一个形象的说法：<span class="mono">op</span> 和 <span class="mono">src</span> 让每个张量都自带一张"<strong>出生证明</strong>"，写着"我是谁、由哪些张量经什么运算生出来的"。
    把所有张量的出生证明顺着 <span class="mono">src</span> 串起来，就还原出了整个家族的<strong>族谱</strong>——这正是计算图。建图，本质上就是<strong>把散落的出生证明汇总成一本族谱</strong>。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 图建好后存在哪？会很占内存吗？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>不占多少。<span class="mono">ggml_cgraph</span> 本身只是几个<strong>指针数组</strong>——<span class="mono">nodes</span> 和 <span class="mono">leafs</span> 里装的是
    指向张量的指针，不是张量数据的拷贝。一张几千节点的图，这些指针数组也就几十 KB。</p>
    <p>张量的<strong>元数据</strong>（形状、op、src）则在 L08 说的 <span class="mono">ggml_context</span> arena 里，同样很轻。真正占内存的是张量的<strong>数据</strong>
    （那一大片浮点数）——而在惰性建图阶段，配合 <span class="mono">no_alloc=true</span>，这些数据<strong>还没分配</strong>呢！要等下一课，看清整张图后，才由后端统一分配。
    所以"建图"这一步，是出了名的<strong>轻</strong>。</p>
    <p>这也带来一个实践上的好处：因为图这么轻、建起来这么便宜，llama.cpp 可以在<strong>每一步推理时都重新搭一张新图</strong>，而不必费心去复用上一步的图。
    每步的 token 数、KV cache 长度可能都不一样，与其小心翼翼地改旧图，不如干脆<strong>重建一张</strong>——反正只是填一串指针，几乎不花时间。"轻量到可以随手重建"，是惰性建图的一个隐藏福利。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>算子函数（<span class="mono">ggml_mul_mat</span> 等）<strong>只建结果张量、填 op/src，不做计算</strong>；真正的运算留到执行阶段。</li>
    <li><span class="mono">ggml_build_forward_expand</span> 从输出张量出发、<strong>沿 src 回溯做拓扑排序</strong>，把依赖按"先算谁"的顺序收进 <span class="mono">ggml_cgraph</span>。</li>
    <li><strong>leafs</strong> = 输入/权重/常量（<span class="mono">op==NONE</span>，不计算）；<strong>nodes</strong> = 算子结果（按拓扑序逐个计算）。</li>
    <li>惰性建图换来三样东西：<strong>整体内存复用</strong>（L10）、<strong>多后端通用</strong>、<strong>自动求导</strong>。</li>
    <li><span class="mono">ggml_cgraph</span> 只是<strong>指针数组</strong>，很轻；张量数据此刻往往还没分配。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  先把整段运算<strong>画成一张图、再统一执行</strong>——这一步小小的"延迟"，换来了巨大的全局视野：内存可以复用、算子可以调度到不同硬件、还能反向求导。
  ggml 的威力，恰恰<strong>从"先不算"开始</strong>。下一课，我们就让这张图真正跑起来。带着一个问题去读下一课：既然图已经把"该算什么、谁依赖谁"说得清清楚楚，
  那么把它<strong>真正算出来</strong>，又需要解决哪些新问题？答案是两个——<strong>内存怎么分配得省</strong>，以及<strong>算子怎么分派到不同硬件</strong>。这正是第十课的主题。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson you learned tensors come from <span class="mono">ggml_context</span>'s memory pool. But here is a fact that surprises many: when you write
<span class="mono">c = ggml_mul_mat(a, b)</span>, that matrix multiply <strong>does not happen at all</strong> - ggml merely quietly records "c is produced from a and b by matmul".
This lesson is about that "<strong>book it, don't do it</strong>" lazy graph building, one of the most elegant designs in the whole ggml engine.
</p>
<p style="color:var(--muted);margin-top:.4rem">Why does this matter? Because it <strong>upends your intuition that "calling a function should give a result immediately"</strong>. In ggml, calling an operator is more
like <strong>placing an order, writing a list</strong> than handing over goods on the spot. A whole network's forward pass is first fully "<strong>recorded as a graph</strong>", and
only later computed at once, by plan. Grasp this two-stage "record first, compute later" and you truly see why ggml can be fast, frugal, and run across all kinds of hardware.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  It is like <strong>writing a recipe</strong> rather than cooking right away: you first write out the steps and their dependencies - "chop first, heat the oil, then stir-fry,
  finally plate" - as a flow chart; only when you actually light the stove (execute) do you cook by that chart. <strong>Building the graph = writing the recipe, executing = cooking
  by it</strong>. The upside: once the recipe is written, you can read it through, optimize the order of heat, or even cook in a different kitchen - all the room that "writing it down
  first" buys you.
</div>
<p>By the way, "lazy" is a compliment in programming, not "slacking" - it means "<strong>never compute ahead of the moment you truly must</strong>". This deferral often lets a
program see the whole picture before acting, and make smarter arrangements. ggml applies this idea to the compute graph: all operator calls are saved up, then computed once the
graph is complete - one of the roots of its efficiency.</p>

<h2>What actually happens in one operator call</h2>
<p>First, correct the core misconception. In many frameworks, writing <span class="mono">c = a @ b</span> computes the multiply immediately, and <span class="mono">c</span> holds the
result numbers. In ggml it is nothing like that: the <span class="mono">c</span> returned by <span class="mono">ggml_mul_mat(a, b)</span> is <strong>still an empty shell</strong> right now -
it knows its shape, knows by whom and by what op it will be produced, but <strong>not a single number is computed yet</strong>. All it records is its "<strong>origin</strong>":</p>
<p>Pay special attention to the word "<strong>back</strong>". When you picture a neural network, arrows usually flow <strong>from input to output</strong> (how data travels); but the
pointers ggml stores inside a tensor point the <strong>opposite</strong> way - the result <strong>points back</strong> to its inputs. Why store it backwards? Because at execution ggml's
key question is "<strong>to compute this result, whom must I have first</strong>", and following a result back to its inputs answers that in one step. It is like tracing a person up to
their parents, then grandparents - far more direct than calling the roll downward from an ancestor. This "<strong>from output back to inputs</strong>" direction is the key reused again
and again in graph building and differentiation.</p>
<div class="flow">
  <div class="node"><div class="nt">a</div><div class="nd">input tensor</div></div>
  <div class="node"><div class="nt">b</div><div class="nd">input tensor</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">c = mul_mat(a, b)</div><div class="nd">c.op = MUL_MAT<br>c.src = [a, b]</div></div>
</div>
<p>Look at the highlighted result <span class="mono">c</span>: its <span class="mono">op</span> field records "I was produced by matmul", and the two pointers
<span class="mono">src[0]</span> and <span class="mono">src[1]</span> point back to <span class="mono">a</span> and <span class="mono">b</span> (note the arrow direction - the result
<strong>points back</strong> to its inputs, hence "back-pointers"). You met these two when L05 introduced the <span class="mono">ggml_tensor</span> fields, where we only said "they record
how it arose"; now you see their real use. Open the source and every operator function follows the same routine:</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  It is worth pausing to feel how uniform this design is: whether matmul, add, normalization, or attention, the hundreds of operator functions are <strong>uniformly "build a tensor + fill op/src + return"</strong>, this same three-step template. Precisely because of this consistency, ggml can handle all operators with <strong>one graph-building and one execution</strong> mechanism - adding a new operator is mainly defining a new <span class="mono">op</span> enum value plus writing its shape inference and compute implementation; the graph-building part needs no change. This restraint of "<strong>one uniform template holding endless variety</strong>" is a big reason ggml's code reads cleanly.
</div>
<pre class="code"><span class="cm">// simplified from ggml_mul_mat in ggml/src/ggml.c</span>
<span class="kw">struct</span> ggml_tensor * <span class="fn">ggml_mul_mat</span>(ctx, a, b) {
    result = <span class="fn">ggml_new_tensor</span>(ctx, GGML_TYPE_F32, ...);  <span class="cm">// just build an empty result tensor</span>
    result-&gt;op     = GGML_OP_MUL_MAT;                   <span class="cm">// record "how I arose"</span>
    result-&gt;src[0] = a;                                 <span class="cm">// record input 1</span>
    result-&gt;src[1] = b;                                 <span class="cm">// record input 2</span>
    <span class="kw">return</span> result;                                      <span class="cm">// return - not one multiply done!</span>
}</pre>
<p><span class="mono">ggml_add</span>, <span class="mono">ggml_rms_norm</span>, <span class="mono">ggml_soft_max</span>... nearly every operator looks like this:
<strong>build a result tensor, fill in op and src, return</strong>. The real floating-point math waits for "execution" later (next lesson). Some operators need extra parameters
(rope's rotation angle, softmax's scale), stored into the result's <span class="mono">op_params</span> via helpers like <span class="mono">ggml_set_op_params_*</span> - but again, only
"recorded", not computed.</p>
<p>Think of it another way: this "empty shell" result tensor is really an <strong>IOU</strong>: it promises you "I will equal a times b in the future", but <strong>has not paid up
yet</strong>. You can keep writing with this IOU - use it as the next operator's input and get a new IOU; layering on and on until you write the final output. Throughout, <strong>no real
numbers are computed</strong>; what you accumulate is a stack of interlocking IOUs. Only at "execution" does ggml follow this stack, from the bottom up, redeeming each into real data. This
"<strong>issue IOUs first, redeem them all later</strong>" is the meaning of lazy, and the core intuition this whole lesson keeps polishing.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  This also explains a pitfall beginners hit: in ggml, <strong>reading a result tensor's data right after building the graph gets you nothing</strong> - the IOU is not redeemed yet. You must first hand the graph to the backend to execute (next lesson) before the result tensor's <span class="mono">data</span> is filled with real values. Splitting "build" and "execute" into two clear phases is the first lesson of using the ggml API well.
</div>

<h2>Stringing tensors into a graph</h2>
<p>One operator recording two or three srcs looks unremarkable; but once you write out a whole model's forward pass, these src pointers <strong>interlock layer by layer into a
directed graph</strong>. Take the smallest example - two linear transforms <span class="mono">y = W2 . (W1 . x)</span>:</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>leafs: x, W1, W2</h4>
    <p>inputs and weights; their <span class="mono">op</span> is NONE (no "origin", the graph's starting points).</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>h = mul_mat(W1, x)</h4>
    <p>the first operator result, <span class="mono">src = [W1, x]</span>; it is a "node".</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>y = mul_mat(W2, h)</h4>
    <p>the second operator result, <span class="mono">src = [W2, h]</span>; note it depends on the previous <span class="mono">h</span>.</p>
  </div></div>
</div>
<p>The beauty of this graph: starting from the output <span class="mono">y</span> and walking back along src pointers, you can <strong>find everything needed to compute it</strong> -
y depends on W2 and h, and h depends on W1 and x. That is exactly what <span class="mono">ggml_build_forward_expand(graph, y)</span> does: starting from the output tensor you specify,
<strong>recurse back along src and collect all dependencies in "who-computes-first" order (topological sort) into a <span class="mono">ggml_cgraph</span></strong>:</p>
<pre class="code"><span class="cm"># cf. ggml_build_forward_expand / ggml_visit_parents_graph in ggml/src/ggml.c</span>
<span class="kw">def</span> <span class="fn">build_forward</span>(graph, t):
    <span class="kw">for</span> s <span class="kw">in</span> t.src:            <span class="cm"># pull in all dependencies first</span>
        <span class="fn">build_forward</span>(graph, s)   <span class="cm"># recurse back</span>
    <span class="kw">if</span> t.op == NONE <span class="kw">and</span> <span class="kw">not</span> t.is_param:
        graph.leafs.append(t)     <span class="cm"># input/constant -&gt; leaf</span>
    <span class="kw">else</span>:
        graph.nodes.append(t)     <span class="cm"># operator result -&gt; node (in dependency order)</span></pre>
<p>Because it is "recurse to collect dependencies first, then add itself", the nodes in <span class="mono">graph.nodes</span> end up naturally <strong>topologically ordered</strong>:
anything earlier never depends on anything later. So at execution you just compute front to back, and for each node its inputs are <strong>guaranteed already computed</strong>.
<span class="mono">ggml_cgraph</span> itself is just a few arrays: <span class="mono">nodes</span> (operator results), <span class="mono">leafs</span> (inputs/constants), counts
<span class="mono">n_nodes</span>/<span class="mono">n_leafs</span>, and capacity <span class="mono">size</span> (default <span class="mono">GGML_DEFAULT_GRAPH_SIZE=2048</span>).</p>
<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  "<strong>Topological sort</strong>" sounds intimidating but the idea is plain: <strong>what you need must be ready first</strong>. When cooking you cannot hit the pan before chopping; before computing <span class="mono">y</span> you must have <span class="mono">h</span>. Topological sort arranges all steps into a legal order so each step's inputs are ready before it. A graph may have <strong>more than one</strong> legal order (two independent branches can go either first), but as long as "dependencies first" holds, the result is the same. ggml's backtracking build computes such a legal order automatically, no worry for you. It also uses a "visited" set internally to avoid collecting the same tensor twice - when multiple operators <strong>share one input</strong> (extremely common in networks), that input is collected once and computed once.
</div>
<p>Put this back into real llama.cpp: after loading a model, each inference step has llama.cpp <strong>build that step's full compute graph</strong> with a long string of operator calls
(embedding, dozens of layers of attention and FFN, the final output projection) - possibly thousands of nodes. None of this big string actually computes; it all <strong>fills op/src and
wires dependencies</strong>; only once the graph is built and handed to the backend are that step's logits computed at once. So the "one decode is build-then-execute inside" you learned (L03)
now has a precise meaning: <strong>building = this string of lazy operator calls, executing = the backend computing the graph in topological order</strong>.</p>

<h2>nodes and leafs: the graph's two residents</h2>
<p>While building, ggml sorts the tensors it meets into two kinds. The criterion is simple: does it have an "origin" (is <span class="mono">op</span> NONE)?</p>
<table class="t">
  <tr><th>kind</th><th>what</th><th>criterion</th><th>at execution</th></tr>
  <tr><td><strong>leafs</strong></td><td>inputs, weights, constants</td><td><span class="mono">op == NONE</span> (no operator origin)</td><td>not computed, its data is used directly</td></tr>
  <tr><td><strong>nodes</strong></td><td>operator results</td><td>has an <span class="mono">op</span> (e.g. MUL_MAT)</td><td>computed one by one in topological order</td></tr>
</table>
<p>With the example above: <span class="mono">x</span>, <span class="mono">W1</span>, <span class="mono">W2</span> are leafs (given, not computed); <span class="mono">h</span>,
<span class="mono">y</span> are nodes (to be computed). The execution engine only acts on <span class="mono">nodes</span>; <span class="mono">leafs</span> just supply raw material. In source this
classification lives in <span class="mono">ggml_visit_parents_graph</span> (<span class="mono">ggml/src/ggml.c</span>): a tensor with <span class="mono">op==NONE</span> and not a param becomes a leaf,
otherwise a node. Grasp this and the long list of nodes/leafs in ggml's debug output is no longer a mystery.</p>
<p>Why insist on separating these two kinds? Because their <strong>treatment at execution is completely different</strong>. Leafs are "<strong>known quantities</strong>" - weights were long since
loaded from the model file, inputs are given by you; their data is right there, and the execution engine <strong>need not touch them</strong>, using them as raw material. Nodes are the "<strong>unknowns
</strong>" - they must be produced from inputs by operators, where the execution engine actually works. Keeping "ready-made" and "to-be-computed" apart makes execution obvious: skip all leafs,
compute only nodes front to back, and the whole graph is done.</p>
<p>One more easily-confused point: can the same tensor be a leaf of this graph and a node of another? Yes. A layer's output <span class="mono">h</span> is a node in "the subgraph that computes h"
(it must be computed), but if you treat it as a <strong>given input</strong> to another computation, it becomes that computation's leaf. Leaf and node are not a tensor's <strong>inherent
property</strong> but the <strong>role it plays in the current graph</strong> - a starting point (leaf) or a mid/end product (node). See this and your grasp of "graphs" grows more flexible. It
echoes the "family tree" metaphor: the same person is a junior in their own branch and an elder to a younger generation, all depending on your reference point.</p>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What does "build first, execute later" actually buy? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Deferring computation looks redundant but buys three precious things. <strong>One, a global view</strong>: once the graph is built, ggml sees at a glance "everything to be
    computed and who depends on whom", so it can carefully <strong>reuse memory</strong> (next lesson's ggml-alloc rides on this to crush peak memory) and merge or reorder operators.</p>
    <p><strong>Two, backend-agnostic</strong>: the graph is a pure description of "what to compute", bound to no hardware, so <strong>the same graph can go to CPU, CUDA, or Metal to
    execute</strong> (echoing L01, L07). <strong>Three, autodiff</strong>: with an explicit dependency graph, walking it backwards by the chain rule computes gradients automatically (for
    training). All three rest on "don't compute yet, just record".</p>
    <p>Conversely, what would you lose by <strong>not</strong> building a graph, computing as you call? You would be stuck "<strong>seeing trees, not the forest</strong>": computing each step with no
    idea what comes later or which intermediates are still needed, so you can only <strong>conservatively keep every intermediate</strong> (memory blow-up) and have <strong>no chance</strong> to merge
    adjacent operators or pick a better order. Eager computation is intuitive to write but seals off all room for optimization; lazy building sacrifices a bit of "what you see is what you get" for
    <strong>a whole blueprint open to optimization</strong>. For an inference engine chasing peak performance, that is a very worthwhile trade.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Are these src back-pointers the same op/src from L05? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The same ones - but only now do you see their <strong>use</strong>. When L05 covered the <span class="mono">ggml_tensor</span> fields, it just told you "there are op and src that record
    how a tensor arose"; back then they seemed like isolated labels.</p>
    <p>This lesson <strong>strings them together</strong>: each tensor remembers its parents, so the whole compute graph is essentially "<strong>tensors holding hands via src into a
    web</strong>". <span class="mono">op</span> says "what op this step does", <span class="mono">src</span> says "where inputs come from" - together, a tensor is both "a block of data" and
    "a compute node in the graph". This dual identity is the key to reading ggml. Fusing the two identities - "data" and "graph node" - into one struct is a hallmark distinguishing ggml from
    frameworks that "define the network structure first, then pour in data".</p>
    <p>A vivid way to put it: <span class="mono">op</span> and <span class="mono">src</span> give every tensor its own "<strong>birth certificate</strong>", stating "who I am, and from which tensors by
    what operation I was born". String all the birth certificates along <span class="mono">src</span> and you reconstruct the whole family's <strong>genealogy</strong> - that is the compute graph. Building
    the graph is essentially <strong>gathering scattered birth certificates into one genealogy book</strong>.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> Where is the built graph stored? Does it use much memory? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Not much. <span class="mono">ggml_cgraph</span> itself is just a few <strong>pointer arrays</strong> - <span class="mono">nodes</span> and <span class="mono">leafs</span> hold pointers to
    tensors, not copies of tensor data. A graph of a few thousand nodes is just tens of KB of pointer arrays.</p>
    <p>A tensor's <strong>metadata</strong> (shape, op, src) sits in L08's <span class="mono">ggml_context</span> arena, also light. What really takes memory is a tensor's
    <strong>data</strong> (that big slab of floats) - and during lazy graph building, with <span class="mono">no_alloc=true</span>, that data is <strong>not even allocated yet</strong>! It
    waits for the next lesson, where after seeing the whole graph the backend allocates it in one pass. So "building the graph" is famously <strong>light</strong>.</p>
    <p>This brings a practical perk: because the graph is so light and cheap to build, llama.cpp can <strong>build a fresh graph at every inference step</strong>, without bothering to reuse the
    previous step's. Each step's token count and KV-cache length may differ, so rather than carefully patching the old graph, it just <strong>rebuilds one</strong> - it is only filling a string of
    pointers, costing almost no time. "Light enough to rebuild casually" is a hidden bonus of lazy graph building.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Operator functions (<span class="mono">ggml_mul_mat</span> etc.) <strong>only build a result tensor and fill op/src, no computation</strong>; the real math waits for execution.</li>
    <li><span class="mono">ggml_build_forward_expand</span> starts from the output tensor and <strong>recurses back along src in topological order</strong>, collecting dependencies into a <span class="mono">ggml_cgraph</span>.</li>
    <li><strong>leafs</strong> = inputs/weights/constants (<span class="mono">op==NONE</span>, not computed); <strong>nodes</strong> = operator results (computed one by one in topological order).</li>
    <li>Lazy building buys three things: <strong>whole-graph memory reuse</strong> (L10), <strong>backend-agnostic execution</strong>, and <strong>autodiff</strong>.</li>
    <li><span class="mono">ggml_cgraph</span> is just <strong>pointer arrays</strong>, very light; tensor data is often not even allocated at this point.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Draw the whole computation as a graph first, then execute it as a whole - this small "delay" buys an enormous global view: memory can be reused, operators can be scheduled to
  different hardware, and gradients can be computed backwards. ggml's power begins precisely <strong>from "don't compute yet"</strong>. Next lesson, we make this graph actually run. Read the next lesson with a question in mind: since the graph already
  spells out "what to compute and who depends on whom", what new problems arise in <strong>actually computing it</strong>? Two - <strong>how to allocate memory frugally</strong>, and <strong>how to
  dispatch operators to different hardware</strong>. That is exactly lesson 10's theme.
</div>
""",
}


LESSON_10 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课，我们把整张计算图<strong>搭</strong>了出来——但它此刻还只是一具"指针搭成的骨架"，里面一个数都没算。这一课，就让这张图真正
<strong>跑起来</strong>：先给它分配内存，再按拓扑序逐个算出每个节点；如果你的机器上既有 CPU 又有 GPU，还要在多个后端之间协调分工。
这一课接上 L07（后端是什么）和 L09（图是什么），把"先建图、后执行"的后半截补完整。
</p>
<p style="color:var(--muted);margin-top:.4rem">换句话说，前两课我们一直在"<strong>纸上谈兵</strong>"——L08 备好内存池、L09 画好计算图，但<strong>没有一个真实的数字被算出来</strong>。
这一课是"<strong>临门一脚</strong>"：把图变成结果。你会看到，这一脚踢得相当讲究：不是简单地从头到尾算一遍就完事，而是要<strong>精打细算地用内存</strong>、还要<strong>聪明地把活儿分给不同硬件</strong>。
正是这两件事，把"能算"变成了"又快又省地算"，也让 ggml 配得上"<strong>高性能推理引擎</strong>"这个名号。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  执行一张计算图，像<strong>施工队照着图纸盖楼</strong>：先看懂整张图纸、算出总共要多少材料、堆放在哪（<strong>内存分配</strong>），
  再按"先打地基、后盖楼层"的顺序一层层施工（<strong>按拓扑序算节点</strong>）；要是场地不够大，就把已经用不上的脚手架拆掉、把场地<strong>腾出来给后面的工序复用</strong>（<strong>内存复用</strong>）。
  图纸（L09）已经画好，这一课讲的就是"怎么照图施工"。施工的两大讲究——<strong>省料</strong>和<strong>分工</strong>——正是这一课的两条主线。
</div>

<h2>执行三步走</h2>
<p>把"让一张图算出结果"这件事拆开，正好是三步，顺次发生：</p>
<div class="flow">
  <div class="node"><div class="nt">建图</div><div class="nd">L09: 填 op/src</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">规划内存</div><div class="nd">ggml-alloc</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">逐节点执行</div><div class="nd">backend compute</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">输出</div><div class="nd">结果张量有数了</div></div>
</div>
<p>第一步建图是上一课的事。这一课的主角是中间两步：<strong>规划内存</strong>（决定每个张量的数据放在缓冲区的哪个位置）和<strong>逐节点执行</strong>
（按拓扑序把每个算子真正算出来）。这两步做完，图里那些原本空着的结果张量，<span class="mono">data</span> 就被填上了真实的数字——你终于能读到结果了。
听起来直白，但"规划内存"这一步藏着 ggml 一个相当漂亮的优化，值得细看。我们先讲"省料"（内存），再讲"分工"（后端与调度）。</p>
<p>为什么"规划内存"要<strong>单独成为一步</strong>、而不是边算边随手分配？这正是惰性建图的红利所在。如果边算边分配，你算到第 6 层时，根本不知道第 20 层会不会还要用第 6 层的输出，
只能<strong>保守地把它留着</strong>。而现在图是完整的，规划器可以在<strong>真正动手算之前</strong>，先把整张图从头到尾扫一遍，把"<strong>每块内存什么时候用、什么时候能让出来</strong>"算得明明白白，
再开始执行。"先规划、后执行"，就像出门前先把行李箱怎么装规划好，而不是边走边往里塞——前者总能塞得更紧。</p>

<h2>内存复用：ggml-alloc 的妙招</h2>
<p>先问个问题：一张有上千个节点的图，是不是要为每个节点的输出都单独留一块内存？如果真这么做，峰值内存会大得吓人。
但 ggml 不这么干——它发现，<strong>很多中间结果是"用完即弃"的</strong>：第 5 层的输出喂给第 6 层之后，就再也用不到了，那块内存完全可以<strong>让第 8 层的输出来复用</strong>。
这正是 <span class="mono">ggml-alloc</span>（图分配器 <span class="mono">ggml_gallocr</span>）干的事：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>内存复用</b>：张量 A 用完后，它占的那块内存被后来的张量 C 接手（同一块地，先住 A、后住 C）</div>
  <div class="cells"><span class="lab">时刻 1</span><span class="cell hl">A</span><span class="cell">B</span><span class="cell dim">空闲</span></div>
  <div class="cells"><span class="lab">时刻 2</span><span class="cell dim">A 已弃</span><span class="cell">B</span><span class="cell hl">C 复用 A 的地</span></div>
</div>
<p>它凭什么能这么精准地复用？<strong>全靠 L09 那张完整的图</strong>。因为图把"谁依赖谁"说得一清二楚，分配器可以<strong>预先推算出每个张量的"生命周期"</strong>
——它从哪个节点开始被需要、到哪个节点之后就再没人用了。一旦某个张量"寿终正寝"，它占的内存立刻被归还，供后面的张量复用。把这套逻辑写成伪代码：</p>
<pre class="code"><span class="cm"># 对应 ggml/src/ggml-alloc.c 的 gallocr 规划逻辑(简化)</span>
<span class="kw">def</span> <span class="fn">plan</span>(graph):
    <span class="kw">for</span> t <span class="kw">in</span> graph.nodes:                 <span class="cm"># 按拓扑序</span>
        t.offset = free_blocks.<span class="fn">best_fit</span>(nbytes(t))  <span class="cm"># 从空闲块里找一块复用</span>
        <span class="kw">for</span> s <span class="kw">in</span> t.src:
            <span class="kw">if</span> <span class="fn">last_use</span>(s) == t:          <span class="cm"># s 在这之后再没人用了</span>
                free_blocks.<span class="fn">give_back</span>(s)  <span class="cm"># 归还它的内存, 供后面复用</span></pre>
<p>这里的关键词是 <span class="mono">best_fit</span>（从空闲块里挑一块大小最合适的）和 <span class="mono">give_back</span>（张量用完就归还内存）。
ggml-alloc 内部维护一组"空闲块"，分配时找最合适的复用，释放时把相邻的空闲块合并成更大的块。<strong>因为提前看到了整张图，它能把内存复用到极致</strong>——
实际跑下来，峰值内存常常只有"每个张量各占一块"的几分之一。这就是 L09 强调"先建图"<strong>真正的回报之一</strong>：没有完整的图，就没法做这种全局的内存规划。</p>
<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  这件事用一个生活场景就能体会：想象一条很长的流水线，每道工序都会产出一个半成品交给下一道。<strong>笨办法</strong>是给每个半成品都准备一个专属货架，流水线越长、货架越多，仓库迟早爆满。<strong>聪明办法</strong>是：一个半成品被下一道工序取走后，它的货架立刻腾出来，给后面的半成品用——因为你<strong>提前知道</strong>整条流水线的全貌，知道每个半成品什么时候"功成身退"。ggml-alloc 就是这个聪明的仓库管理员，而 L09 的计算图，就是它手里那张"<strong>全流程图</strong>"。少了这张图，它就只能用笨办法。
</div>
<p>顺带说一个常见误区：内存复用<strong>不会</strong>影响计算结果的正确性。有人担心"A 的地盘给了 C，会不会把 A 的数据弄乱？"——不会，因为复用只发生在 A <strong>确定不再被任何人需要之后</strong>。
分配器严格按生命周期办事：只要还有谁可能读 A，A 的内存就绝不会被征用。所以内存复用是一种<strong>完全无损</strong>的优化，省的是空间，动不了结果——这一点和 L06 说的"量化是有损、KV cache 是无损"那个区分，是同一种思维。</p>

<h2>逐节点执行：后端登场</h2>
<p>内存规划好了，终于到了"真正算"的一步。这一步由<strong>后端</strong>（L07 讲过的 CPU/CUDA/Metal 等）来执行。最简单的情形是只用一个后端：</p>
<pre class="code"><span class="cm">// 简化的用法; 涉及 ggml-backend.h / ggml-cpu.h / ggml-alloc.h</span>
ggml_backend_t be = <span class="fn">ggml_backend_cpu_init</span>();   <span class="cm">// 选一个后端(也可以是 cuda/metal...)</span>
<span class="fn">ggml_gallocr_alloc_graph</span>(galloc, graph);       <span class="cm">// 按规划真正分配内存</span>
<span class="fn">ggml_backend_graph_compute</span>(be, graph);         <span class="cm">// 逐节点执行!</span></pre>
<div class="flow">
  <div class="node"><div class="nt">graph.nodes</div><div class="nd">按拓扑序遍历</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">取 node 的 src</div><div class="nd">输入必已算好</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">调后端核函数</div><div class="nd">CPU SIMD / GPU kernel</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">填进 node.data</div><div class="nd">直到最后一个节点</div></div>
</div>
<p><span class="mono">ggml_backend_graph_compute</span> 做的事，就是<strong>按拓扑序遍历 <span class="mono">graph.nodes</span>，对每个节点调用该后端对应的算子实现</strong>
（matmul 调 matmul 核、softmax 调 softmax 核……这些核函数是 L11、第六部分的主题）。因为 L09 保证了拓扑序，每算到一个节点，它的输入<strong>必定已经算好</strong>，
所以从头到尾扫一遍就完事。算完最后一个节点，输出张量里就有结果了。叶子（权重、输入）则全程不计算，只是被算子读取。</p>
<p>这里要破除一个幻觉：<span class="mono">ggml_backend_graph_compute</span> 这个函数名听起来很"重"，好像它自己在做天大的事，其实它更像一个<strong>循环 + 派发器</strong>——
真正的苦力活（一次矩阵乘里成千上万次乘加）是在每个算子的<strong>核函数</strong>里完成的，而核函数是各后端各自实现的（CPU 用 SIMD、CUDA 用 GPU kernel）。
所以"执行一张图"在 ggml 这一层很薄：<strong>按顺序遍历节点、对每个节点喊一声"该你了"</strong>；至于"怎么算得快"，是下一课（L11 算子）和第六部分（内核）的主题。
这种"<strong>调度归调度、计算归计算</strong>"的分层，正是 ggml 能把同一张图跑在天差地别的硬件上的原因。</p>
<p>再点破一个容易忽略的细节：执行<strong>不是从叶子开始算的</strong>，而是直接从第一个<strong>节点</strong>开始。叶子（权重、输入）在执行前就已经备好数据了——权重是从模型文件加载的、
输入是你喂进去的，它们不需要"算"。所以 <span class="mono">ggml_backend_graph_compute</span> 的循环只遍历 <span class="mono">graph.nodes</span>，对每个节点取出它的 src（输入已就绪）、调核函数算出结果、填进它的 data。
一圈下来，从第一个节点到最后一个节点，整张图就算完了。理解这一点，你就明白为什么 L09 要把叶子和节点分开存——<strong>正是为了让执行循环能干净利落地"只算节点"</strong>。</p>

<h2>多后端调度：ggml_backend_sched</h2>
<p>但现实往往更复杂：你可能想把模型的<strong>一部分层放 GPU、其余留 CPU</strong>（还记得 L07 的 <span class="mono">-ngl</span> 吗？）。这时一个后端不够用了，
需要一个"<strong>调度器</strong>"来协调多个后端，它就是 <span class="mono">ggml_backend_sched</span>：</p>
<div class="cols">
  <div class="col"><h4>它要解决的问题</h4><p>一张图里，有的算子该在 GPU 上跑、有的在 CPU 上跑；数据在两种内存里，跨设备时要搬运。谁来统筹？</p></div>
  <div class="col"><h4>sched 的三件事</h4><p><strong>① 拆图</strong>：把图切成若干段，按设备归类；<strong>② 指派</strong>：每段算子分给合适的后端；<strong>③ 拷贝</strong>：在 CPU/GPU 边界自动插入数据搬运。</p></div>
</div>
<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  简单说，<span class="mono">ggml_backend_sched</span> 就是个"<strong>包工头</strong>"：拿到整张图后，它把活儿<strong>分派</strong>给手下不同的"工种"（后端），谁擅长干什么就给谁，还负责在工种之间<strong>传递半成品</strong>（跨设备拷贝张量）。这正是你用 <span class="mono">-ngl 20</span> 把 20 层放进 GPU 时，背后默默发生的事——sched 把这 20 层的算子指派给 CUDA 后端、其余留给 CPU，并在两者交界处安排好数据搬运。你只填了一个数字，它替你搞定了一切协调。
</div>
<p>顺便厘清 <span class="mono">ggml_backend_graph_compute</span> 和 <span class="mono">ggml_backend_sched</span> 的关系：前者是"<strong>单后端</strong>"的执行——一张图、一个设备，直接从头算到尾；
后者是"<strong>多后端</strong>"的总指挥——它先把图拆成几段，每段再各自交给 <span class="mono">ggml_backend_graph_compute</span> 在对应设备上执行。所以 sched 是<strong>更上一层</strong>的协调者，
单后端执行是它手里的基本工具。只用 CPU 时你可能直接用前者；要混合 CPU/GPU，就得请出后者。理清这层包含关系，你看 ggml 的执行代码就不会绕晕。</p>
<p>为什么要分多个后端，而不是统统塞给 GPU？因为<strong>显存常常装不下整个模型</strong>。一个量化后还有几十 GB 的大模型，你的显卡可能只放得下一半的层；
剩下的层只能留在 CPU 内存里、用 CPU 算。这种"<strong>一半 GPU、一半 CPU</strong>"的混合执行，正是消费级硬件跑大模型的常态，也是 <span class="mono">ggml_backend_sched</span> 存在的根本理由。
它让你能<strong>按显存大小灵活地切</strong>：显存多就多放几层进 GPU、少就少放几层，剩下的 CPU 兜底，总能跑起来——只是放进 GPU 的层越多、整体越快。</p>
<div class="card warn">
  <div class="tag">⚠ 性能坑</div>
  也正因为有跨设备拷贝这件事，"放多少层进 GPU"并不是越多越好的简单题。每跨一次 CPU/GPU 边界，都要把数据搬一趟，<strong>搬运本身有开销</strong>。如果切得太碎、来回搬太多次，省下的计算时间可能还不够还搬运的债。所以实践中，往往是<strong>把连续的一大段层整体放进 GPU</strong>（减少边界），而不是东放一层、西放一层。这些权衡 sched 帮你处理了大部分，但理解它，能帮你在显存紧张时更聪明地设 <span class="mono">-ngl</span>。
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么内存能复用得这么省？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>核心还是那句话：<strong>靠 L09 的完整图</strong>。即时计算（边算边丢）时，你没法预知"这个中间结果以后还用不用"，只能保守地都留着，内存自然省不下来。</p>
    <p>而有了完整的图，分配器能<strong>精确算出每个张量"最后一次被用"是在哪个节点</strong>。过了那个点，这块内存立刻可回收。再配合 <span class="mono">best_fit</span> 挑最合适的空闲块、
    把相邻空闲块合并，整张图的峰值内存就被压到很低。可以说，<strong>惰性建图省内存，省的就是这一笔</strong>——这也是为什么训练/推理框架都爱用计算图。</p>
    <p>顺便给个量级感：对一个几十层的大模型，"每个中间张量各占一块"和"复用"两种做法，<strong>峰值内存能差好几倍</strong>。在显存本就紧张的消费级显卡上，这"好几倍"往往就是
    "<strong>跑得起来</strong>"和"<strong>爆显存跑不起来</strong>"的分界线。所以内存复用不是锦上添花的小优化，而是很多模型能在你机器上跑起来的<strong>前提条件</strong>之一。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> sched 怎么决定哪段放哪个后端？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>大体看两点：一是<strong>张量当前在谁的内存里</strong>（数据已经在 GPU 上，自然倾向于在 GPU 算）；二是<strong>这个算子该后端支不支持</strong>。
    如果某个算子 GPU 后端还没实现，sched 会让它<strong>回退到 CPU</strong> 算完（这就是 L07 提过的 fallback），保证整张图总能跑完。</p>
    <p>每当一段的输出要喂给另一种设备上的下一段时，sched 就在<strong>边界处自动插入一个"拷贝"节点</strong>，把数据从一种内存搬到另一种。这些拷贝是有开销的，
    所以"切在哪、放哪些层到 GPU"会影响性能——但这些细节 sched 都替你处理了，你通常只需给一个 <span class="mono">-ngl</span> 数字。</p>
    <p>还有一种更细的切法叫"<strong>按张量切分</strong>"（tensor split）：把<strong>同一个</strong>大矩阵乘，按列拆成几块，分给多张 GPU 同时算，再把结果拼起来。
    这适合<strong>多卡</strong>场景，能把一个单步算子的负载摊到几张卡上。无论是"按层切"还是"按张量切"，背后都是 sched 在统筹——你只需通过 <span class="mono">--split-mode</span> 等参数告诉它怎么切，
    剩下的拆图、指派、拷贝、拼接，都由它默默完成。这种"<strong>把复杂的多设备协调藏在一个简单接口后面</strong>"的设计，正是 ggml 好用的地方。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> reserve 和 alloc 两步是干嘛的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>这是 ggml 一个"配置一次、反复执行"的典型套路。<strong>reserve（预留）</strong>：先用图<strong>预演</strong>一遍，量出"这张图最多要占多少内存"，
    然后一次性把这块缓冲开好。<strong>alloc（分配）</strong>：之后每次执行，都在这块已开好的缓冲里<strong>复用</strong>，把各张量摆到规划好的位置上。</p>
    <p>为什么分两步？因为开大块内存（尤其是 GPU 显存）很贵，不能每次执行都重新开一遍。先 reserve 一次、量好峰值、开好缓冲，后面成千上万步推理就<strong>反复复用同一块缓冲</strong>，
    省掉了反复大分配的开销。这和 L08 的 arena 思想一脉相承：<strong>大块内存一次拿好，内部反复腾挪</strong>。</p>
    <p>把这条线串起来看，你会发现 ggml 从头到尾贯穿着同一个信念：<strong>大额的、昂贵的操作（找系统要内存）尽量只做一次，之后全靠内部复用</strong>。
    L08 的 arena 如此、这里的 reserve/alloc 如此、L10 的内存复用也如此。理解了这个一以贯之的"<strong>一次拿好、反复复用</strong>"，你就抓住了 ggml 性能设计的灵魂。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>执行 = <strong>规划内存</strong>（ggml-alloc）+ <strong>按拓扑序逐节点 compute</strong>（后端）；算完结果张量的 <span class="mono">data</span> 才有数。</li>
    <li><strong>ggml-alloc</strong> 靠 L09 的完整图<strong>预知每个张量的生命周期</strong>，用完即归还、best-fit 复用，把峰值内存压到很低。</li>
    <li><span class="mono">ggml_backend_graph_compute</span> 在<strong>单个后端</strong>上逐节点执行；叶子不算、只被读取。</li>
    <li><span class="mono">ggml_backend_sched</span> 协调<strong>多后端</strong>：拆图、把算子指派到设备、跨设备自动拷贝——这是 <span class="mono">-ngl</span> 的底层。</li>
    <li><strong>reserve/alloc 两步</strong>：先预演量出峰值、开好缓冲，之后反复复用，省掉反复大分配。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  L09 那个"先不算"的延迟，在这一课<strong>连本带利地还了回来</strong>：正因为提前拿到了整张图，内存才能被算得死死地复用、算子才能被聪明地分派到不同硬件。
  "<strong>先描述、后执行</strong>"从来不是麻烦，而是把<strong>全局优化的主动权</strong>牢牢攥在手里。ggml 引擎最核心的三课——内存（L08）、建图（L09）、执行（L10）——到这里就拼齐了。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson we <strong>built</strong> the whole compute graph - but it is still just a "skeleton of pointers", not a number computed in it. This lesson makes that graph actually
<strong>run</strong>: first allocate its memory, then compute each node in topological order; and if your machine has both a CPU and a GPU, coordinate the division of labor across multiple
backends. This connects L07 (what a backend is) and L09 (what the graph is), completing the second half of "build first, execute later".
</p>
<p style="color:var(--muted);margin-top:.4rem">In other words, the last two lessons were all "<strong>theory on paper</strong>" - L08 prepared the memory pool, L09 drew the compute graph, but <strong>not a single real number was
computed</strong>. This lesson is the "<strong>final kick</strong>": turning the graph into results. And you will see this kick is rather refined: not simply computing front to back, but <strong>using
memory frugally</strong> and <strong>cleverly splitting the work across hardware</strong>. These two things turn "can compute" into "compute fast and frugally", earning ggml the name "<strong>high-performance
inference engine</strong>".</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Executing a compute graph is like <strong>a construction crew building from blueprints</strong>: first read the whole blueprint, compute how much material is needed and where to stack it
  (<strong>memory allocation</strong>), then build floor by floor in order "foundation first, then floors" (<strong>compute nodes in topological order</strong>); and if the site is too small,
  tear down scaffolding no longer needed and <strong>free the space for later steps to reuse</strong> (<strong>memory reuse</strong>). The blueprint (L09) is drawn; this lesson is about "building
  from it". The two crafts of building - <strong>saving materials</strong> and <strong>dividing labor</strong> - are this lesson's two main threads.
</div>

<h2>Three execution steps</h2>
<p>Break "make a graph produce a result" apart and it is exactly three steps, in sequence:</p>
<div class="flow">
  <div class="node"><div class="nt">build</div><div class="nd">L09: fill op/src</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">plan memory</div><div class="nd">ggml-alloc</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">compute nodes</div><div class="nd">backend compute</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">output</div><div class="nd">result tensors have data</div></div>
</div>
<p>The first step, building, was last lesson. This lesson stars the middle two: <strong>planning memory</strong> (deciding where each tensor's data sits in the buffer) and <strong>computing
nodes</strong> (actually computing each operator in topological order). After these two, the once-empty result tensors get their <span class="mono">data</span> filled with real numbers - you can
finally read results. Sounds plain, but "planning memory" hides a rather beautiful ggml optimization worth a close look. We cover "saving materials" (memory) first, then "dividing labor" (backends and
scheduling).</p>
<p>Why is "planning memory" <strong>its own step</strong> rather than allocating on the fly as you compute? This is exactly the dividend of lazy building. Allocating on the fly, by layer 6 you have no idea
whether layer 20 will still need layer 6's output, so you can only <strong>conservatively keep it</strong>. But now the graph is complete, so the planner can - <strong>before actually computing</strong> -
sweep the whole graph front to back and work out clearly "<strong>when each block is used and when it can be released</strong>", then start executing. "Plan first, execute later" is like planning how to
pack a suitcase before leaving rather than stuffing it as you walk - the former always packs tighter.</p>

<h2>Memory reuse: ggml-alloc's trick</h2>
<p>First a question: for a graph with thousands of nodes, must each node's output get its own block of memory? Doing so would make peak memory frighteningly large. But ggml does not - it
notices <strong>many intermediates are "use-once and discard"</strong>: layer 5's output, once fed to layer 6, is never needed again, so its memory can perfectly well <strong>be reused by layer
8's output</strong>. That is exactly what <span class="mono">ggml-alloc</span> (the graph allocator <span class="mono">ggml_gallocr</span>) does:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Memory reuse</b>: once tensor A is done, its block is taken over by a later tensor C (same plot, A lives then C)</div>
  <div class="cells"><span class="lab">time 1</span><span class="cell hl">A</span><span class="cell">B</span><span class="cell dim">free</span></div>
  <div class="cells"><span class="lab">time 2</span><span class="cell dim">A done</span><span class="cell">B</span><span class="cell hl">C reuses A's plot</span></div>
</div>
<p>How can it reuse so precisely? <strong>All thanks to L09's complete graph</strong>. Because the graph spells out "who depends on whom", the allocator can <strong>pre-compute each tensor's
"lifetime"</strong> - from which node it starts being needed, to which node after which no one uses it. Once a tensor "dies", its memory is immediately returned for later tensors to reuse. As
pseudocode:</p>
<pre class="code"><span class="cm"># cf. the gallocr planning logic in ggml/src/ggml-alloc.c (simplified)</span>
<span class="kw">def</span> <span class="fn">plan</span>(graph):
    <span class="kw">for</span> t <span class="kw">in</span> graph.nodes:                 <span class="cm"># topological order</span>
        t.offset = free_blocks.<span class="fn">best_fit</span>(nbytes(t))  <span class="cm"># find a reusable free block</span>
        <span class="kw">for</span> s <span class="kw">in</span> t.src:
            <span class="kw">if</span> <span class="fn">last_use</span>(s) == t:          <span class="cm"># s is used by no one after this</span>
                free_blocks.<span class="fn">give_back</span>(s)  <span class="cm"># return its memory for reuse</span></pre>
<p>The keywords are <span class="mono">best_fit</span> (pick the most suitably-sized free block) and <span class="mono">give_back</span> (return a tensor's memory once done). ggml-alloc keeps a set
of "free blocks", finds the best fit when allocating, and merges adjacent free blocks when releasing. <strong>Because it saw the whole graph in advance, it reuses memory to the hilt</strong> - in
practice peak memory is often a fraction of "each tensor its own block". This is <strong>one of the real payoffs</strong> of L09's "build first": without the complete graph, this global memory
planning would be impossible.</p>
<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  An everyday scene makes it click: imagine a long assembly line where each station produces a half-product handed to the next. The <strong>dumb way</strong> is to give every half-product its own dedicated shelf - the longer the line, the more shelves, until the warehouse overflows. The <strong>smart way</strong>: once a half-product is taken by the next station, its shelf frees immediately for later half-products - because you <strong>know in advance</strong> the whole line and when each half-product "retires". ggml-alloc is that smart warehouse manager, and L09's compute graph is the "<strong>full-process chart</strong>" in its hand. Without that chart, it could only use the dumb way.
</div>
<p>A common misconception in passing: memory reuse does <strong>not</strong> affect correctness. Some worry "giving A's plot to C - won't it corrupt A's data?" - no, because reuse only happens after A
is <strong>certainly needed by no one</strong>. The allocator strictly follows lifetimes: as long as anyone might still read A, A's memory is never requisitioned. So memory reuse is a <strong>completely
lossless</strong> optimization - it saves space without touching results, the same thinking as L06's distinction "quantization is lossy, the KV cache is lossless".</p>

<h2>Computing nodes: the backend steps in</h2>
<p>With memory planned, we finally reach "actually compute". This step is carried out by a <strong>backend</strong> (the CPU/CUDA/Metal from L07). The simplest case uses one backend:</p>
<pre class="code"><span class="cm">// simplified usage; spans ggml-backend.h / ggml-cpu.h / ggml-alloc.h</span>
ggml_backend_t be = <span class="fn">ggml_backend_cpu_init</span>();   <span class="cm">// pick a backend (could be cuda/metal...)</span>
<span class="fn">ggml_gallocr_alloc_graph</span>(galloc, graph);       <span class="cm">// actually allocate per the plan</span>
<span class="fn">ggml_backend_graph_compute</span>(be, graph);         <span class="cm">// compute node by node!</span></pre>
<div class="flow">
  <div class="node"><div class="nt">graph.nodes</div><div class="nd">walk in topological order</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">take node's src</div><div class="nd">inputs already computed</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">call backend kernel</div><div class="nd">CPU SIMD / GPU kernel</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">fill node.data</div><div class="nd">until the last node</div></div>
</div>
<p>What <span class="mono">ggml_backend_graph_compute</span> does is <strong>walk <span class="mono">graph.nodes</span> in topological order, calling that backend's operator implementation for each node</strong>
(matmul calls the matmul kernel, softmax the softmax kernel... these kernels are the topic of L11 and Part 6). Because L09 guarantees topological order, when each node is reached its inputs are
<strong>certainly already computed</strong>, so one front-to-back pass does it. After the last node, the output tensor holds the result. Leafs (weights, inputs) are never computed, only read by
operators.</p>
<p>Dispel one illusion here: the name <span class="mono">ggml_backend_graph_compute</span> sounds "heavy", as if it does something enormous itself, but it is really more of a <strong>loop +
dispatcher</strong> - the real grunt work (the thousands of multiply-adds in one matmul) happens inside each operator's <strong>kernel</strong>, and kernels are implemented by each backend separately (CPU
with SIMD, CUDA with GPU kernels). So "executing a graph" is thin at this ggml layer: <strong>walk the nodes in order and shout "your turn" at each</strong>; "how to compute fast" is the topic of the next
lesson (L11 operators) and Part 6 (kernels). This "<strong>scheduling is scheduling, computing is computing</strong>" layering is exactly why ggml can run the same graph on wildly different hardware.</p>
<p>One more easily-missed detail: execution <strong>does not start from the leafs</strong> but straight from the first <strong>node</strong>. Leafs (weights, inputs) already have their data before execution -
weights loaded from the model file, inputs fed by you; they need no "computing". So <span class="mono">ggml_backend_graph_compute</span>'s loop only walks <span class="mono">graph.nodes</span>, taking each node's
src (inputs ready), calling the kernel to compute the result, filling its data. One pass from the first node to the last and the whole graph is done. Grasp this and you see why L09 stores leafs and nodes
separately - <strong>precisely so the execution loop can cleanly "compute only nodes"</strong>.</p>

<h2>Multi-backend scheduling: ggml_backend_sched</h2>
<p>But reality is often more complex: you may want <strong>some layers on GPU, the rest on CPU</strong> (remember L07's <span class="mono">-ngl</span>?). Now one backend is not enough; you need a
"<strong>scheduler</strong>" to coordinate multiple backends - that is <span class="mono">ggml_backend_sched</span>:</p>
<div class="cols">
  <div class="col"><h4>The problem it solves</h4><p>In one graph, some operators should run on GPU, some on CPU; data lives in two memories, needing transfer across devices. Who coordinates?</p></div>
  <div class="col"><h4>sched's three jobs</h4><p><strong>1. Split</strong> the graph into segments, grouped by device; <strong>2. Assign</strong> each segment's operators to a suitable backend; <strong>3. Copy</strong> - auto-insert data transfers at CPU/GPU boundaries.</p></div>
</div>
<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Simply put, <span class="mono">ggml_backend_sched</span> is a "<strong>general contractor</strong>": given the whole graph, it <strong>assigns</strong> the work to different "trades" (backends) under it - whoever is good at what gets it - and handles <strong>passing half-finished goods</strong> between trades (cross-device tensor copies). This is exactly what happens behind the scenes when you put 20 layers on GPU with <span class="mono">-ngl 20</span> - sched assigns those 20 layers' operators to the CUDA backend, leaves the rest to the CPU, and arranges the data transfers at the boundary. You filled in one number; it handled all the coordination for you.
</div>
<p>Let me clarify the relation between <span class="mono">ggml_backend_graph_compute</span> and <span class="mono">ggml_backend_sched</span>: the former is "<strong>single-backend</strong>" execution - one
graph, one device, computed straight front to back; the latter is the "<strong>multi-backend</strong>" conductor - it first splits the graph into segments, then hands each to
<span class="mono">ggml_backend_graph_compute</span> to execute on its device. So sched is the <strong>higher-level</strong> coordinator, and single-backend execution is the basic tool in its hand. On CPU only
you might use the former directly; to mix CPU/GPU you call in the latter. Get this containment straight and ggml's execution code will not dizzy you.</p>
<p>Why multiple backends rather than just cramming everything onto the GPU? Because <strong>VRAM often cannot hold the whole model</strong>. A big model that is still tens of GB after quantization may only fit
half its layers on your card; the rest must stay in CPU memory and compute on CPU. This "<strong>half GPU, half CPU</strong>" hybrid execution is the norm for running big models on consumer hardware, and
the fundamental reason <span class="mono">ggml_backend_sched</span> exists. It lets you <strong>cut flexibly by VRAM size</strong>: more VRAM, put more layers on GPU; less, fewer, with the CPU as backstop, so
it always runs - just faster the more layers go on GPU.</p>
<div class="card warn">
  <div class="tag">⚠ Performance trap</div>
  And precisely because of cross-device copies, "how many layers on GPU" is not a simple "more is better". Each crossing of a CPU/GPU boundary means moving data, and <strong>the move itself has a cost</strong>. Cut too finely and shuttle too often, and the compute time saved may not repay the moving debt. So in practice one usually <strong>puts a large contiguous span of layers on the GPU as a whole</strong> (fewer boundaries) rather than one layer here, one there. sched handles most of these trade-offs, but understanding it helps you set <span class="mono">-ngl</span> more wisely when VRAM is tight.
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why can memory be reused so frugally? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Still the same line: <strong>thanks to L09's complete graph</strong>. With eager computation (compute and discard as you go), you cannot foresee "will this intermediate be needed later", so
    you conservatively keep them all, and memory cannot be saved.</p>
    <p>With the complete graph, the allocator can <strong>compute exactly at which node each tensor is "last used"</strong>. Past that point, its memory is immediately reclaimable. Combined with
    <span class="mono">best_fit</span> picking the most suitable free block and merging adjacent free blocks, the whole graph's peak memory is crushed. You could say <strong>lazy building saves memory,
    and this is the saving</strong> - which is why training/inference frameworks love compute graphs.</p>
    <p>For a sense of scale: for a many-layer big model, "each intermediate tensor its own block" vs "reuse" can differ in peak memory by <strong>several times</strong>. On a consumer GPU already tight on
    VRAM, that "several times" is often the line between "<strong>it runs</strong>" and "<strong>out of VRAM, won't run</strong>". So memory reuse is not a nice-to-have tweak but one of the <strong>preconditions
    </strong> for many models to run on your machine at all.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> How does sched decide which segment goes to which backend? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Roughly two things: one, <strong>whose memory the tensor currently lives in</strong> (data already on GPU naturally favors computing on GPU); two, <strong>whether that backend supports this
    operator</strong>. If some operator is not yet implemented in the GPU backend, sched lets it <strong>fall back to CPU</strong> (the fallback mentioned in L07), guaranteeing the whole graph always
    runs.</p>
    <p>Whenever a segment's output must feed the next segment on a different device, sched <strong>auto-inserts a "copy" node at the boundary</strong>, moving data from one memory to another. These
    copies have a cost, so "where to cut, which layers on GPU" affects performance - but sched handles these details, and you usually just give a single <span class="mono">-ngl</span> number.</p>
    <p>There is also a finer cut called "<strong>tensor split</strong>": split <strong>one</strong> big matmul by columns into several pieces, give them to multiple GPUs to compute at once, then stitch the
    results. This suits <strong>multi-card</strong> setups, spreading a single operator's load across several cards. Whether "split by layer" or "split by tensor", sched coordinates behind the scenes - you just
    tell it how to cut via parameters like <span class="mono">--split-mode</span>, and the rest - splitting, assigning, copying, stitching - it does quietly. This "<strong>hide complex multi-device coordination
    behind a simple interface</strong>" design is exactly what makes ggml pleasant to use.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What are the reserve and alloc steps for? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>This is a classic ggml "configure once, execute repeatedly" pattern. <strong>reserve</strong>: first <strong>rehearse</strong> with the graph to measure "how much memory this graph needs at most",
    then open that buffer once. <strong>alloc</strong>: on each later execution, <strong>reuse</strong> this already-opened buffer, placing tensors at their planned positions.</p>
    <p>Why two steps? Because opening big blocks (especially GPU VRAM) is expensive; you cannot reopen it every execution. Reserve once, measure the peak, open the buffer, and the thousands of later
    inference steps <strong>reuse the same buffer over and over</strong>, sparing repeated big allocations. This is of a piece with L08's arena idea: <strong>grab a big block once, shuffle within it
    repeatedly</strong>.</p>
    <p>String this thread together and you find one belief running through all of ggml: <strong>do the large, expensive operations (asking the system for memory) as few times as possible - ideally once -
    then reuse internally</strong>. L08's arena is like this, reserve/alloc here is like this, L10's memory reuse is like this. Grasp this consistent "<strong>grab once, reuse repeatedly</strong>" and you have
    the soul of ggml's performance design.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Execution = <strong>plan memory</strong> (ggml-alloc) + <strong>compute nodes in topological order</strong> (backend); only then do result tensors' <span class="mono">data</span> hold numbers.</li>
    <li><strong>ggml-alloc</strong> uses L09's complete graph to <strong>foresee each tensor's lifetime</strong>, returning memory once done and reusing via best-fit, crushing peak memory.</li>
    <li><span class="mono">ggml_backend_graph_compute</span> computes node by node on <strong>a single backend</strong>; leafs are not computed, only read.</li>
    <li><span class="mono">ggml_backend_sched</span> coordinates <strong>multiple backends</strong>: split the graph, assign operators to devices, auto-copy across devices - the underpinning of <span class="mono">-ngl</span>.</li>
    <li><strong>reserve/alloc two steps</strong>: rehearse to measure the peak and open the buffer, then reuse repeatedly, sparing repeated big allocations.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  L09's "don't compute yet" delay <strong>pays back with interest</strong> in this lesson: precisely because the whole graph was obtained in advance, memory can be reused tightly and operators
  cleverly dispatched to different hardware. "<strong>Describe first, execute later</strong>" was never a hassle but a way to keep the <strong>initiative for global optimization</strong> firmly in hand.
  The ggml engine's three core lessons - memory (L08), graph building (L09), execution (L10) - now fit together.
</div>
""",
}


LESSON_11 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
计算图是由<strong>算子</strong>搭起来的。这一课，我们挑出 transformer 里最核心的几个算子——矩阵乘 <span class="mono">mul_mat</span>、归一化
<span class="mono">rms_norm</span>、位置编码 <span class="mono">rope</span>、带掩码的 <span class="mono">soft_max_ext</span>，看清它们各算什么、
<strong>输入输出的形状怎么对上</strong>，以及一个算子在 CPU 上"真正落地计算"的地方在哪。这一课把 L04 的注意力数学和 L09/L10 的图与执行，
用具体的算子串到了一起。
</p>
<p style="color:var(--muted);margin-top:.4rem">前三课我们一直在讲"<strong>容器和流程</strong>"——内存怎么放（L08）、图怎么搭（L09）、图怎么跑（L10），但<strong>始终没碰"每个算子具体在算什么"</strong>。
这一课补上这一块。不过要说明：我们<strong>不</strong>逐行去抠某个矩阵乘的循环（那是第六部分内核课的事），而是站在"<strong>会读、会搭网络</strong>"的高度，搞清楚四件事——
这些算子各自<strong>做什么</strong>、形状<strong>怎么对</strong>、注意力<strong>怎么由它们拼成</strong>、以及它们<strong>在哪真正落地算</strong>。学完这一课，你再看 llama.cpp 里那些建图代码，就能大致读懂每一行在拼什么。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  算子就像一块块<strong>乐高积木</strong>：每块都有固定的凸点和凹槽（输入、输出的形状），只有<strong>形状对得上</strong>，两块才能拼在一起。
  建一个模型，就是按形状把这些积木拼成一座塔；要是哪两块形状对不上，拼装（建图时的断言检查）当场就会失败、提醒你拼错了。
  懂了每块积木的"接口形状"，你就懂了怎么读、怎么搭一个网络。而所有积木里，<strong>矩阵乘是那块最大、最关键的底座</strong>，所以我们从它讲起。
</div>

<h2>头号算子：矩阵乘 mul_mat</h2>
<p>神经网络里<strong>绝大部分的计算量，都花在矩阵乘上</strong>——注意力、前馈层、输出投影，本质都是矩阵乘。所以 <span class="mono">ggml_mul_mat</span>
是当之无愧的头号算子。它的形状规则，是这一课<strong>最该记牢</strong>的一条：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>mul_mat 形状推导</b>：内维 ne[0] 必须相等(被消去)，结果取两者的"另一维"</div>
  <div class="cells"><span class="lab">a</span><span class="cell hl">k</span><span class="cell">m</span><span class="lab">ne=[k, m]</span></div>
  <div class="cells"><span class="lab">b</span><span class="cell hl">k</span><span class="cell">n</span><span class="lab">ne=[k, n]</span></div>
  <div class="cells"><span class="lab">结果</span><span class="cell">m</span><span class="cell">n</span><span class="lab">ne=[m, n]，k 被消去</span></div>
</div>
<p>看那两个高亮的 <span class="mono">k</span>：<strong>a 和 b 在 ne[0]（内维）上必须相等</strong>，这个相等的维在相乘时被"消去"，结果的形状由两者各自的"另一维"拼成。
落到源码（<span class="mono">ggml/src/ggml.c</span> 的 <span class="mono">ggml_mul_mat</span> / <span class="mono">ggml_can_mul_mat</span>）：</p>
<pre class="code"><span class="cm">// 断言: a 的内维 == b 的内维 (ne[0] 相等)</span>
GGML_ASSERT(a-&gt;ne[0] == b-&gt;ne[0]);          <span class="cm">// k 必须对上, 否则建图就报错</span>
<span class="cm">// 结果形状: 取 a 的"另一维"、b 的"另一维", 高两维来自 b</span>
ne = { a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3] };  <span class="cm">// 结果类型固定 F32</span></pre>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  这里有个<strong>容易栽跟头</strong>的点：ggml 的形状规则，读起来和你数学课上学的"行 × 列"<strong>方向是反的</strong>。原因是 L05 讲过的——ggml <strong>行优先、ne[0] 是最内维</strong>，所以"内维相等"对应的其实是数学里"左矩阵的列数 == 右矩阵的行数"。只要牢记 L05 那句口诀"<strong>ne[0] 永远是最贴着内存、变化最快的那一维</strong>"，就不会把行当成列。此外，结果的高两维（<span class="mono">ne[2]</span>、<span class="mono">ne[3]</span>）来自 b，且支持<strong>广播</strong>（a 的对应维可以是 b 的整数分之一），这正是多头注意力里"一组权重作用于多个头"的实现方式。
</div>
<p>为什么矩阵乘这么重要，值得单拎出来讲？因为它<strong>又重又频繁</strong>。一个 7B 模型，每生成一个 token，要做几百次矩阵乘，每次都是几千乘几千的大矩阵相乘——
模型的绝大部分参数（那几个 GB 的权重）都是以"矩阵乘里的那个权重矩阵"的身份存在的。所以你之前学的所有东西，到头来几乎都在为矩阵乘服务：量化（L06）是为了让权重矩阵更小、搬得更快，
后端（L07）是为了让矩阵乘算得更快，内存复用（L10）是为了腾地方装矩阵乘的中间结果。<strong>看懂 mul_mat，就看懂了推理的主战场。</strong></p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  再多说一句形状里那个"<strong>消去</strong>"。为什么内维相等、还被消去？因为矩阵乘的本质，就是拿 a 的一行和 b 的一行（在 ggml 的布局下）<strong>逐元素相乘再求和</strong>——那条被"相乘求和"吃掉的维，就是内维 k。它在结果里不复存在，只留下 a、b 各自的"另一维"组成结果的形状。理解了"k 被求和吃掉"，你就明白为什么两个 <span class="mono">[k, ...]</span> 的张量乘出来是 <span class="mono">[m, n]</span>，而不是别的——这不是死记的规则，而是"求和把一维压没了"的自然结果。
</div>
<p>把矩阵乘的形状这条线收个尾：在 ggml 里你会反复看到形如 <span class="mono">cur = ggml_mul_mat(ctx, model.layers[i].wq, cur)</span> 的代码——拿这一层的 Q 权重矩阵去乘当前的隐藏状态，
得到 Query。整座 transformer 的建图，骨架上就是一串这样的 mul_mat，中间穿插着归一化、rope、softmax。所以只要你能<strong>对着权重的形状，推出每个 mul_mat 的输出形状</strong>，
你就能顺着代码把整个模型的数据流"走"一遍。这正是这一课开头说的"会读、会搭网络"的具体含义——而它的核心，就是 mul_mat 这条形状规则。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里也顺势点明 ggml 的一个取舍：它的算子不像有些框架那样"什么都能广播、什么形状都自动对齐"，而是<strong>把形状约束定得相当严格</strong>，对不上就当场断言失败。为什么宁可严格、也不要"智能地自动适配"？因为推理引擎最怕<strong>悄悄出错</strong>——一个被自动广播"凑合"过去的形状错误，可能让模型输出一堆看似正常实则错误的结果，极难排查。严格的断言把错误<strong>挡在建图阶段、暴露在第一现场</strong>，反而让整个系统更可靠。这是性能工程里常见的态度：<strong>宁可早失败、响亮地失败，也不要带病运行。</strong>
</div>

<h2>三个常客：rms_norm / rope / soft_max_ext</h2>
<p>除了矩阵乘，注意力层里还反复出现三个算子。把 L04 讲的注意力，用 ggml 算子串起来，大致是这样一条流水线：</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>rms_norm(x)</h4><p>先把输入归一化，稳住数值尺度（L04 说的"训练稳定"就靠它）。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>mul_mat(Wq/Wk/Wv, ·)</h4><p>投影出 Query / Key / Value 三个张量。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>rope(q, k)</h4><p>给 Q、K 注入位置信息（L04 说的 RoPE 旋转）。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>soft_max_ext(scores, mask)</h4><p>算注意力分数、施加因果掩码、归一成权重（L04 的 -inf 掩码就在这）。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>mul_mat(V, weights)</h4><p>按权重把 Value 加权汇总，得到注意力输出。</p></div></div>
</div>
<p>注意上面流水线里 mul_mat 出现了<strong>好几次</strong>——投影 Q/K/V 是三次 mul_mat、最后按权重汇总 Value 又是一次。这正印证了前面说的"矩阵乘是主战场"：一个注意力层里，
真正吃算力的几乎全是这些 mul_mat，而 rms_norm、rope、softmax 更像是穿插其间的"调味"步骤，单独看都不重，但少了谁注意力都不对。把这条流水线和 L04 的注意力数学对照着看，
你会发现"<strong>数学公式</strong>"和"<strong>ggml 算子序列</strong>"几乎是一一对应的——这也是为什么说看懂算子，就看懂了模型怎么落地成代码。</p>
<p>这三个常客的签名都很直白（<span class="mono">ggml/include/ggml.h</span>）：</p>
<pre class="code"><span class="fn">ggml_rms_norm</span>(ctx, a, eps);                       <span class="cm">// 按最后一维做 RMS 归一化, eps 防止除零</span>
<span class="fn">ggml_rope_ext</span>(ctx, a, pos, ff, n_dims, mode, ...); <span class="cm">// 按位置 pos 旋转, 注入相对位置信息</span>
<span class="fn">ggml_soft_max_ext</span>(ctx, a, mask, scale, max_bias); <span class="cm">// 融合: softmax(a*scale + mask)</span></pre>
<p>逐个一句话：<span class="mono">rms_norm</span> 把一行向量按其均方根缩放到稳定范围，比 LayerNorm 更省（不用算均值）；<span class="mono">rope</span> 不是给位置加一个"序号向量"，
而是<strong>按位置旋转</strong> Q、K，让注意力分数自带"两个 token 相距多远"的信息；<span class="mono">soft_max_ext</span> 是个<strong>融合算子</strong>，把"乘缩放系数 + 加掩码 + softmax"三步并成一次，
省内存又省带宽。注意 <span class="mono">scale</span> 通常是 <span class="mono">1/sqrt(d)</span>（防止分数过大）、<span class="mono">mask</span> 装的就是因果掩码（未来位置为 -inf）。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  为什么要把这三个算子单独点出来？因为它们<strong>体现了 ggml 算子设计的两个常见手法</strong>。一是<strong>融合</strong>：<span class="mono">soft_max_ext</span> 把本可以拆成三四个算子的事（缩放、加掩码、求指数、归一）压成一个，少建几个中间张量、少搬几趟数据——在 decode 这种"带宽比算力更紧张"的场景（L04 说过），融合的收益尤其明显。二是<strong>专用化</strong>：transformer 几乎离不开归一化和位置编码，ggml 干脆为它们提供 <span class="mono">rms_norm</span>、<span class="mono">rope_ext</span> 这样的<strong>专用算子</strong>，而不是让你用一堆基础算子拼。专用算子既好读、又给了后端"整段优化"的机会。这两手——该融合的融合、该专用的专用——贯穿 ggml 的算子库。
</div>
<p>顺带澄清一个容易混的点：<span class="mono">ggml_rope</span> 和 <span class="mono">ggml_rope_ext</span> 是同一族，后者多了一串参数（<span class="mono">freq_base</span>、<span class="mono">freq_scale</span> 等），
用来支持 YaRN 这类<strong>长上下文扩展</strong>技术——简单说，就是通过调整旋转的"频率"，让一个原本只在 4K 上下文训练的模型，也能在几万 token 的长上下文上工作。你现在不必深究这些参数，
只要知道"<strong>位置编码也是可以调的，调它能换来更长的上下文</strong>"，这个认识就够了。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  把这三个算子和 mul_mat 放在一起，你就掌握了读懂任何 transformer 建图代码所需的"<strong>核心词汇表</strong>"：<span class="mono">mul_mat</span>（投影、注意力打分、汇总）、<span class="mono">rms_norm</span>（每个子层前的归一化）、<span class="mono">rope</span>（位置）、<span class="mono">soft_max_ext</span>（注意力权重）。再加上加法（残差）、逐元素乘（门控）这几个基础算子，一个 transformer block 的建图代码，<strong>九成的行你都能认出来在干什么</strong>。剩下的一成是各家模型的小花样，但万变不离这套核心算子——这正是这一课最实在的收获。
</div>

<h2>一个算子，两处代码</h2>
<p>最后破除一个常见困惑：一个算子在 ggml 里其实有<strong>两处</strong>代码，分工明确。一处负责"建图"（定义形状、填 op/src，L09），另一处负责"真正算"（在某个后端上跑数）：</p>
<div class="cols">
  <div class="col"><h4>建图侧：ggml.c</h4><p><span class="mono">ggml_mul_mat(ctx, a, b)</span>：只<strong>定义</strong>结果张量的形状、填好 op 和 src，<strong>不算</strong>。每个算子在这里都有一个"构造函数"。</p></div>
  <div class="col"><h4>计算侧：ggml-cpu / ggml-cuda …</h4><p><span class="mono">ggml_compute_forward_mul_mat(...)</span>：<strong>真正</strong>把矩阵乘算出来。CPU 用 SIMD、CUDA 用 GPU kernel，各后端各写一份。</p></div>
</div>
<p>这两处通过 <span class="mono">enum ggml_op</span> 这个"算子编号"对接：建图时把编号记在 <span class="mono">tensor-&gt;op</span> 里，执行时后端用一个大 <span class="mono">switch(op)</span>
把每个节点<strong>派发</strong>到对应的 <span class="mono">ggml_compute_forward_*</span>（CPU 端在 <span class="mono">ggml/src/ggml-cpu/</span>）。所以"算子很多"并不可怕——它们共享同一套建图与派发框架，
<strong>加一个新算子，主要就是加一个 enum 值 + 写一份 forward 实现</strong>。这种"声明与实现分离"的设计，正是同一张图能在 CPU、CUDA、Metal 上各自高效跑起来的根本。</p>
<p>这个"两处代码"的分工，回头看也解释了前面几课的很多设计。L09 说算子函数"只填 op/src 不计算"——那是因为它<strong>只是建图侧</strong>，计算侧的代码根本不在那儿。
L10 说后端"逐节点 compute"——那个 compute，就是在<strong>计算侧</strong>按 op 派发、逐个调 forward。所以建图侧和计算侧，恰好对应了 L09 的"建图"和 L10 的"执行"两个阶段；
一个算子横跨这两个阶段，在建图侧露个脸（定形状）、在计算侧出全力（真算）。把这条线理顺，ggml 的整个执行流程在你脑子里就<strong>串成一根完整的链</strong>了。</p>
<p>这也是为什么 ggml 能<strong>把模型逻辑和硬件加速彻底分开</strong>：写一个新模型，你只在建图侧用现成算子拼一拼，完全不碰任何 CPU/GPU 的计算代码；
而优化某个算子在某种硬件上的速度，你只改计算侧那一份 forward，不影响任何模型。这种"<strong>模型作者和内核作者各管一摊、互不打扰</strong>"的分工，是 ggml 这类引擎能被广泛复用、又能持续优化的组织学基础。</p>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 mul_mat 的形状规则看起来和数学反着来？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为 ggml 是<strong>行优先</strong>、<span class="mono">ne[0]</span> 是最内（最贴内存、变化最快）的维。数学里我们写"<span class="mono">A(m×k) · B(k×n) = C(m×n)</span>"，
    要求"A 的列数 k == B 的行数 k"。但在 ggml 里，那个连续的 k 维被放在了 <span class="mono">ne[0]</span>，于是规则写成"<span class="mono">a.ne[0] == b.ne[0]</span>"。</p>
    <p>换句话说，<strong>数学的"行/列"和 ggml 的 ne 维度顺序是反过来的</strong>——这正是 L05 那个"维度顺序和 PyTorch 相反"的坑在算子层的体现。
    记住 L05 的口诀"ne[0] 最贴内存"，再看任何 ggml 算子的形状约束，都不会再绕晕。</p>
    <p>给个实操建议：读 ggml 建图代码时，<strong>把每个张量的 ne 在草稿纸上标出来</strong>，顺着算子一个个推导形状，遇到 mul_mat 就检查"两个内维对上没有"。
    这是 ggml 编程最有效的排错法——大多数建图 bug，都是某处形状对不上、被那句 <span class="mono">GGML_ASSERT</span> 当场拦下。形状推导手熟了，你读再复杂的模型建图代码也不慌。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> soft_max_ext 的 mask 和 scale 到底干嘛？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">scale</span> 是<strong>缩放系数</strong>，通常取 <span class="mono">1/sqrt(d)</span>（d 是每个头的维度）。注意力分数是 Q 和 K 的点积，维度越高、点积越容易变得很大，
    softmax 后会过于"尖锐"（几乎一边倒）；先乘一个 <span class="mono">1/sqrt(d)</span> 把分数压回合理范围，梯度和数值都更稳。</p>
    <p><span class="mono">mask</span> 则是把<strong>因果掩码</strong>加进分数：未来位置加上 <span class="mono">-inf</span>，softmax 后权重就变成 0（L04 讲过）。
    <span class="mono">max_bias</span> 控制 ALiBi 这类相对位置偏置，不用时为 0。<span class="mono">soft_max_ext</span> 把"乘 scale、加 mask、做 softmax"<strong>融合成一个算子</strong>，
    避免了生成多个庞大的中间张量——这是推理引擎里很常见的"算子融合"优化。</p>
    <p>顺带把<strong>融合</strong>这件事说透一点。不融合的话，softmax 这一步要先建一个"乘了 scale 的张量"、再建一个"加了 mask 的张量"、再建一个"算了指数的张量"……每一步都要在内存里
    实打实地写出一个和分数矩阵一样大的中间结果，既占内存又费带宽。融合算子则把这几步<strong>在一个循环里一气呵成</strong>，中间值只在寄存器/缓存里转一圈，根本不落地成大张量。
    对注意力这种"分数矩阵随上下文长度平方增长"的算子，融合省下的内存和带宽相当可观——这也是为什么 llama.cpp 还有 flash-attention 这类更激进的融合实现。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 算子这么多，ggml 怎么管得过来？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>靠一个统一的<strong>枚举 + 派发</strong>机制。每个算子是 <span class="mono">enum ggml_op</span> 里的一个值（GGML_OP_MUL_MAT、GGML_OP_SOFT_MAX……）；
    建图时这个值被记进 <span class="mono">tensor-&gt;op</span>。执行时，后端遍历每个节点，用一个大 <span class="mono">switch(node-&gt;op)</span> 跳到对应的 <span class="mono">ggml_compute_forward_*</span> 实现。</p>
    <p>所以加一个新算子的工作量是<strong>可控的</strong>：① 在 enum 里加一个值；② 写一个建图构造函数（定形状、填 src）；③ 在每个你关心的后端里写一份 forward 实现，并接进那个 switch。
    框架的其它部分（建图、内存规划、调度）<strong>完全不用动</strong>。这种"<strong>开放扩展、封闭修改</strong>"的结构，是 ggml 能持续长出几百个算子、还不乱套的原因。</p>
    <p>这套机制也解释了为什么 ggml 能<strong>支持那么多不同架构的模型</strong>。Llama、Qwen、Mistral、Gemma…… 这些模型的差异，本质上就是"用哪些算子、按什么顺序拼"——
    而它们用到的算子，绝大多数是<strong>共享的同一批</strong>（矩阵乘、归一化、注意力那几样）。所以新增一个模型架构，往往<strong>一个新算子都不用加</strong>，只是在建图侧换个拼法；
    偶尔遇到某个架构有独特设计，才补一两个新算子。正是这个共享的算子库，让 llama.cpp 能跟上层出不穷的新模型，而不必每来一个就大改一遍引擎。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">mul_mat</span> 要求<strong>内维 ne[0] 相等</strong>（被消去），结果 <span class="mono">ne={a.ne[1], b.ne[1], ...}</span>，类型 F32，高维支持广播。</li>
    <li>形状规则读起来<strong>和数学"行×列"方向相反</strong>，因为 ggml 行优先、ne[0] 最内（L05 的坑）。</li>
    <li><span class="mono">rms_norm</span> 稳数值、<span class="mono">rope</span> 注入位置、<span class="mono">soft_max_ext</span> 融合"缩放+掩码+softmax"成注意力权重。</li>
    <li>每个算子<strong>两处代码</strong>：建图侧（ggml.c 定形状/填 op/src）+ 计算侧（后端的 <span class="mono">ggml_compute_forward_*</span> 真算）。</li>
    <li>执行靠 <span class="mono">enum ggml_op</span> + 大 <span class="mono">switch(op)</span> 派发；加新算子 = 加 enum + 写 forward。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把一个算子拆成"<strong>声明形状</strong>"和"<strong>各后端各自实现</strong>"两半——前者让建图轻量、还能在拼装时当场查错，后者让同一个算子在 CPU/CUDA/Metal 上各有最优实现。
  模型逻辑只写一遍、硬件加速写多份，正是这种<strong>声明与实现解耦</strong>的红利。下一课，我们钻进这些算子真正吃下去的"料"——量化格式的字节细节。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
A compute graph is built from <strong>operators</strong>. This lesson picks the core few in a transformer - matmul <span class="mono">mul_mat</span>, normalization
<span class="mono">rms_norm</span>, position encoding <span class="mono">rope</span>, masked <span class="mono">soft_max_ext</span> - to see what each computes, <strong>how input
and output shapes line up</strong>, and where an operator "actually lands and computes" on the CPU. This lesson strings L04's attention math and L09/L10's graph and execution
together with concrete operators.
</p>
<p style="color:var(--muted);margin-top:.4rem">The last three lessons were all about "<strong>containers and flow</strong>" - how memory is placed (L08), how the graph is built (L09), how the graph runs (L10) - but <strong>never
touched "what each operator actually computes"</strong>. This lesson fills that in. To be clear: we will <strong>not</strong> pore over the loop of some matmul line by line (that is Part 6's
kernel lesson); instead, from the height of "<strong>being able to read and build networks</strong>", we nail down four things - what these operators <strong>do</strong>, how shapes
<strong>line up</strong>, how attention is <strong>assembled</strong> from them, and where they <strong>actually land and compute</strong>. After this lesson, the graph-building code in llama.cpp
becomes mostly readable - you can tell what each line is assembling.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Operators are like <strong>Lego bricks</strong>: each has fixed studs and sockets (input and output shapes), and only when <strong>shapes match</strong> can two bricks join.
  Building a model is assembling these bricks into a tower by shape; if two bricks' shapes do not match, assembly (the assertion check at graph-build time) fails on the spot, telling
  you the fit is wrong. Understand each brick's "interface shape" and you understand how to read and build a network. And of all the bricks, <strong>matmul is the biggest, most crucial base</strong>, so we start there.
</div>

<h2>The number-one operator: matmul mul_mat</h2>
<p>In a neural network, <strong>the vast majority of compute goes into matrix multiplication</strong> - attention, feed-forward, output projection are all essentially matmuls.
So <span class="mono">ggml_mul_mat</span> is the undisputed number-one operator. Its shape rule is the one thing this lesson is <strong>most worth memorizing</strong>:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>mul_mat shape inference</b>: the inner dim ne[0] must be equal (eliminated); the result takes each one's "other dim"</div>
  <div class="cells"><span class="lab">a</span><span class="cell hl">k</span><span class="cell">m</span><span class="lab">ne=[k, m]</span></div>
  <div class="cells"><span class="lab">b</span><span class="cell hl">k</span><span class="cell">n</span><span class="lab">ne=[k, n]</span></div>
  <div class="cells"><span class="lab">result</span><span class="cell">m</span><span class="cell">n</span><span class="lab">ne=[m, n]; k eliminated</span></div>
</div>
<p>Look at the two highlighted <span class="mono">k</span>: <strong>a and b must be equal on ne[0] (the inner dim)</strong>, this equal dim is "eliminated" in the multiply, and the
result's shape is formed from each one's "other dim". In source (<span class="mono">ggml_mul_mat</span> / <span class="mono">ggml_can_mul_mat</span> in <span class="mono">ggml/src/ggml.c</span>):</p>
<pre class="code"><span class="cm">// assert: a's inner dim == b's inner dim (ne[0] equal)</span>
GGML_ASSERT(a-&gt;ne[0] == b-&gt;ne[0]);          <span class="cm">// k must match, else graph-build errors</span>
<span class="cm">// result shape: take a's "other dim", b's "other dim", high dims from b</span>
ne = { a-&gt;ne[1], b-&gt;ne[1], b-&gt;ne[2], b-&gt;ne[3] };  <span class="cm">// result type is always F32</span></pre>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  There is a <strong>tripping point</strong> here: ggml's shape rule reads in the <strong>opposite direction</strong> from the "rows x columns" you learned in math class. The reason is from L05 - ggml is <strong>row-major, ne[0] is the innermost dim</strong>, so "inner dims equal" actually corresponds to math's "left matrix's columns == right matrix's rows". Just keep L05's mnemonic "<strong>ne[0] is always the memory-adjacent, fastest-changing dim</strong>" and you will not mistake rows for columns. Also, the result's high dims (<span class="mono">ne[2]</span>, <span class="mono">ne[3]</span>) come from b and support <strong>broadcasting</strong> (a's matching dim can be an integer fraction of b's) - exactly how "one set of weights applied to multiple heads" is implemented in multi-head attention.
</div>
<p>Why is matmul so important it deserves its own section? Because it is <strong>both heavy and frequent</strong>. A 7B model does hundreds of matmuls per generated token, each a
thousands-by-thousands matrix multiply - the vast majority of the model's parameters (those several GB of weights) exist as "the weight matrix in a matmul". So almost everything you have learned
ultimately serves matmul: quantization (L06) to make weight matrices smaller and faster to move, backends (L07) to compute matmuls faster, memory reuse (L10) to make room for matmul intermediates.
<strong>Understand mul_mat and you understand the main battlefield of inference.</strong></p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  One more word on that "<strong>elimination</strong>" in the shape. Why is the inner dim equal and then eliminated? Because matrix multiply is essentially taking a row of a and a row of b (under ggml's layout) and <strong>multiplying element-wise then summing</strong> - the dim eaten by that "multiply-and-sum" is the inner dim k. It is gone in the result, leaving only a's and b's "other dim" to form the result shape. Once you get "k is eaten by the sum", you see why two <span class="mono">[k, ...]</span> tensors multiply into <span class="mono">[m, n]</span> and nothing else - not a rule to memorize but the natural result of "summing collapses one dim".
</div>
<p>To wrap up the matmul-shape thread: in ggml you will repeatedly see code like <span class="mono">cur = ggml_mul_mat(ctx, model.layers[i].wq, cur)</span> - multiplying this layer's Q weight matrix by
the current hidden state to get the Query. The whole transformer's graph build is, skeletally, a string of such mul_mats interleaved with normalization, rope, softmax. So as long as you can <strong>derive
each mul_mat's output shape from the weight shapes</strong>, you can "walk" the entire model's data flow through the code. This is the concrete meaning of "reading and building networks" from the lesson's
opening - and at its core is this one mul_mat shape rule.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This is also a good moment to note a ggml trade-off: its operators do not "broadcast anything, auto-align any shape" like some frameworks; instead it <strong>sets shape constraints quite strictly</strong>, asserting failure on the spot when things do not match. Why prefer strict over "smart auto-adaptation"? Because an inference engine fears <strong>silent errors</strong> most - a shape error papered over by auto-broadcast could make the model output a pile of plausible-looking but wrong results, extremely hard to trace. Strict assertions <strong>block errors at graph-build, exposing them at the first scene</strong>, making the whole system more reliable. This is a common attitude in performance engineering: <strong>fail early and loudly rather than run sick</strong>.
</div>

<h2>Three regulars: rms_norm / rope / soft_max_ext</h2>
<p>Besides matmul, three operators recur in the attention layer. Stringing L04's attention with ggml operators gives roughly this pipeline:</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>rms_norm(x)</h4><p>normalize the input first, stabilizing the numeric scale (L04's "training stability" rests on it).</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>mul_mat(Wq/Wk/Wv, ·)</h4><p>project out the Query / Key / Value tensors.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>rope(q, k)</h4><p>inject position info into Q, K (L04's RoPE rotation).</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>soft_max_ext(scores, mask)</h4><p>compute attention scores, apply the causal mask, normalize to weights (L04's -inf mask is here).</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>mul_mat(V, weights)</h4><p>weight-sum the Values by the weights to get the attention output.</p></div></div>
</div>
<p>Note that mul_mat appears <strong>several times</strong> in the pipeline above - projecting Q/K/V is three mul_mats, and the final weight-sum of Values is another. This confirms the earlier "matmul is
the main battlefield": in one attention layer, almost all the real compute is these mul_mats, while rms_norm, rope, softmax are more like "seasoning" steps interspersed - each light on its own, yet
attention is wrong without any of them. Compare this pipeline with L04's attention math and you find "<strong>the formula</strong>" and "<strong>the ggml operator sequence</strong>" map almost one-to-one -
which is why understanding operators means understanding how a model lands as code.</p>
<p>These three regulars have plain signatures (<span class="mono">ggml/include/ggml.h</span>):</p>
<pre class="code"><span class="fn">ggml_rms_norm</span>(ctx, a, eps);                       <span class="cm">// RMS-normalize over the last dim, eps avoids divide-by-zero</span>
<span class="fn">ggml_rope_ext</span>(ctx, a, pos, ff, n_dims, mode, ...); <span class="cm">// rotate by position pos, injecting relative position</span>
<span class="fn">ggml_soft_max_ext</span>(ctx, a, mask, scale, max_bias); <span class="cm">// fused: softmax(a*scale + mask)</span></pre>
<p>One line each: <span class="mono">rms_norm</span> scales a row vector by its root-mean-square into a stable range, cheaper than LayerNorm (no mean to compute);
<span class="mono">rope</span> does not add an "index vector" per position but <strong>rotates</strong> Q, K by position, so attention scores carry "how far apart two tokens are";
<span class="mono">soft_max_ext</span> is a <strong>fused operator</strong> merging "multiply scale + add mask + softmax" into one, saving memory and bandwidth. Note <span class="mono">scale</span>
is usually <span class="mono">1/sqrt(d)</span> (to keep scores from getting too large), and <span class="mono">mask</span> holds the causal mask (future positions at -inf).</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Why single out these three operators? Because they <strong>exemplify two common techniques of ggml operator design</strong>. One is <strong>fusion</strong>: <span class="mono">soft_max_ext</span> compresses what could be three or four operators (scale, add mask, exponentiate, normalize) into one, building fewer intermediate tensors and moving data fewer times - in decode, where "bandwidth is tighter than compute" (from L04), fusion's payoff is especially clear. The other is <strong>specialization</strong>: transformers can hardly do without normalization and position encoding, so ggml just provides <strong>dedicated operators</strong> like <span class="mono">rms_norm</span> and <span class="mono">rope_ext</span> rather than making you assemble them from basic ops. Dedicated operators are both readable and give the backend a chance to "optimize the whole segment". These two moves - fuse what should be fused, specialize what should be specialized - run through ggml's operator library.
</div>
<p>A clarification in passing: <span class="mono">ggml_rope</span> and <span class="mono">ggml_rope_ext</span> are the same family, the latter with a string of extra parameters
(<span class="mono">freq_base</span>, <span class="mono">freq_scale</span>, etc.) supporting long-context extension techniques like YaRN - in short, by adjusting the rotation "frequency", a model originally
trained at 4K context can work at tens of thousands of tokens. You need not study these parameters now; just knowing "<strong>position encoding is tunable, and tuning it buys longer context</strong>" is
enough.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Put these three operators together with mul_mat and you have the "<strong>core vocabulary</strong>" needed to read any transformer's graph-build code: <span class="mono">mul_mat</span> (projection, attention scoring, summing), <span class="mono">rms_norm</span> (normalization before each sub-layer), <span class="mono">rope</span> (position), <span class="mono">soft_max_ext</span> (attention weights). Add a few basic operators like add (residual) and element-wise multiply (gating), and you can recognize <strong>ninety percent of the lines</strong> in a transformer block's graph-build code. The remaining ten percent are each model's little tweaks, but they never stray from this core operator set - exactly this lesson's most practical takeaway.
</div>

<h2>One operator, two pieces of code</h2>
<p>Finally, dispel a common confusion: an operator in ggml actually has <strong>two</strong> pieces of code, with a clear division. One does "graph building" (define the shape, fill
op/src, L09), the other does "actually compute" (run the numbers on some backend):</p>
<div class="cols">
  <div class="col"><h4>build side: ggml.c</h4><p><span class="mono">ggml_mul_mat(ctx, a, b)</span>: only <strong>defines</strong> the result tensor's shape, fills op and src, <strong>no compute</strong>. Every operator has a "constructor" here.</p></div>
  <div class="col"><h4>compute side: ggml-cpu / ggml-cuda ...</h4><p><span class="mono">ggml_compute_forward_mul_mat(...)</span>: <strong>actually</strong> computes the matmul. CPU with SIMD, CUDA with GPU kernels, one per backend.</p></div>
</div>
<p>These two meet through <span class="mono">enum ggml_op</span>, the "operator number": graph-build records the number in <span class="mono">tensor-&gt;op</span>, and at execution the backend uses
one big <span class="mono">switch(op)</span> to <strong>dispatch</strong> each node to the matching <span class="mono">ggml_compute_forward_*</span> (CPU side in <span class="mono">ggml/src/ggml-cpu/</span>).
So "many operators" is not scary - they share one graph-build and dispatch framework, and <strong>adding a new operator is mainly adding an enum value + writing a forward
implementation</strong>. This "declaration separated from implementation" design is the very reason the same graph can run efficiently on CPU, CUDA, and Metal each.</p>
<p>This "two pieces of code" division, in hindsight, explains much of the earlier lessons' design. L09 said operator functions "only fill op/src, no compute" - that is because they are <strong>only the
build side</strong>; the compute-side code is simply not there. L10 said the backend "computes node by node" - that compute is the <strong>compute side</strong> dispatching by op and calling each forward.
So the build side and compute side correspond exactly to L09's "build" and L10's "execute" phases; one operator spans both phases, showing its face on the build side (define shape) and going all-out on
the compute side (actually compute). Straighten this thread and ggml's whole execution flow strings into one complete chain in your head.</p>
<p>This is also why ggml can <strong>fully separate model logic from hardware acceleration</strong>: writing a new model, you only assemble ready-made operators on the build side, touching no CPU/GPU
compute code at all; optimizing some operator's speed on some hardware, you only change that one forward on the compute side, affecting no model. This division - "<strong>model authors and kernel
authors each mind their own patch, without disturbing each other</strong>" - is the organizational basis for why engines like ggml can be widely reused and continuously optimized.</p>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does mul_mat's shape rule look reversed from math? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Because ggml is <strong>row-major</strong> and <span class="mono">ne[0]</span> is the innermost (memory-adjacent, fastest-changing) dim. In math we write "<span class="mono">A(m x k) . B(k x n)
    = C(m x n)</span>", requiring "A's columns k == B's rows k". But in ggml that contiguous k dim sits at <span class="mono">ne[0]</span>, so the rule is written "<span class="mono">a.ne[0] ==
    b.ne[0]</span>".</p>
    <p>In other words, <strong>math's "rows/columns" and ggml's ne dimension order are reversed</strong> - exactly L05's "dimension order opposite to PyTorch" trap, surfacing at the operator
    level. Keep L05's mnemonic "ne[0] is memory-adjacent" and any ggml operator's shape constraint stops being confusing.</p>
    <p>A practical tip: when reading ggml graph-build code, <strong>jot each tensor's ne on scratch paper</strong> and derive shapes operator by operator, checking at each mul_mat "do the two inner dims
    match". This is the most effective debugging method in ggml programming - most graph-build bugs are a shape mismatch somewhere, caught on the spot by that <span class="mono">GGML_ASSERT</span>. Once
    shape inference is second nature, you read even the most complex model graph-build code without panic.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What exactly do soft_max_ext's mask and scale do? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">scale</span> is a <strong>scaling factor</strong>, usually <span class="mono">1/sqrt(d)</span> (d is the per-head dimension). Attention scores are dot products of Q and K;
    the higher the dimension, the larger dot products tend to get, making softmax too "sharp" (nearly one-sided); multiplying by <span class="mono">1/sqrt(d)</span> first pulls scores back to a
    reasonable range, stabilizing gradients and values.</p>
    <p><span class="mono">mask</span> adds the <strong>causal mask</strong> into the scores: future positions get <span class="mono">-inf</span>, so their weights become 0 after softmax (from L04).
    <span class="mono">max_bias</span> controls ALiBi-style relative-position bias, 0 when unused. <span class="mono">soft_max_ext</span> <strong>fuses</strong> "multiply scale, add mask, softmax"
    into one operator, avoiding several large intermediate tensors - a very common "operator fusion" optimization in inference engines.</p>
    <p>Let me spell out <strong>fusion</strong> a bit more. Without fusion, the softmax step would build a "scaled tensor", then a "mask-added tensor", then an "exponentiated tensor"... each step writing
    out, for real in memory, an intermediate as big as the score matrix - costing memory and bandwidth. A fused operator does these steps <strong>in one loop, all at once</strong>, with intermediates only
    circling through registers/cache, never materializing as big tensors. For attention, whose "score matrix grows with the square of context length", the memory and bandwidth fusion saves are
    considerable - which is also why llama.cpp has even more aggressive fused implementations like flash-attention.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> So many operators - how does ggml manage them all? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>With a uniform <strong>enum + dispatch</strong> mechanism. Each operator is a value in <span class="mono">enum ggml_op</span> (GGML_OP_MUL_MAT, GGML_OP_SOFT_MAX...); at graph build this
    value is recorded in <span class="mono">tensor-&gt;op</span>. At execution the backend walks each node and a big <span class="mono">switch(node-&gt;op)</span> jumps to the matching
    <span class="mono">ggml_compute_forward_*</span> implementation.</p>
    <p>So adding a new operator is <strong>contained</strong> work: 1. add a value to the enum; 2. write a graph-build constructor (define the shape, fill src); 3. write a forward
    implementation in each backend you care about and wire it into that switch. The rest of the framework (graph build, memory planning, scheduling) <strong>needs no change at all</strong>. This
    "<strong>open for extension, closed for modification</strong>" structure is why ggml can keep growing hundreds of operators without falling apart.</p>
    <p>This mechanism also explains why ggml can <strong>support so many different model architectures</strong>. Llama, Qwen, Mistral, Gemma... the differences among these models are essentially "which
    operators, assembled in what order" - and the operators they use are mostly <strong>the same shared set</strong> (matmul, normalization, the attention pieces). So adding a new model architecture often
    needs <strong>not a single new operator</strong>, just a different assembly on the build side; only occasionally, when an architecture has a unique design, do you add one or two new operators. It is
    this shared operator library that lets llama.cpp keep up with the endless stream of new models without overhauling the engine for each one.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">mul_mat</span> requires <strong>equal inner dim ne[0]</strong> (eliminated); result <span class="mono">ne={a.ne[1], b.ne[1], ...}</span>, type F32, high dims broadcast.</li>
    <li>The shape rule reads <strong>reversed from math's "rows x columns"</strong>, because ggml is row-major and ne[0] is innermost (L05's trap).</li>
    <li><span class="mono">rms_norm</span> stabilizes values, <span class="mono">rope</span> injects position, <span class="mono">soft_max_ext</span> fuses "scale + mask + softmax" into attention weights.</li>
    <li>Each operator has <strong>two pieces of code</strong>: build side (ggml.c defines shape / fills op/src) + compute side (the backend's <span class="mono">ggml_compute_forward_*</span> actually computes).</li>
    <li>Execution dispatches via <span class="mono">enum ggml_op</span> + big <span class="mono">switch(op)</span>; adding an operator = add an enum + write a forward.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Splitting an operator into "<strong>declare the shape</strong>" and "<strong>each backend implements its own</strong>" - the former keeps graph-building light and catches errors right at
  assembly, the latter lets the same operator have an optimal implementation on CPU/CUDA/Metal. Model logic written once, hardware acceleration written several times - exactly the dividend of this
  <strong>declaration-implementation decoupling</strong>. Next lesson, we dig into the "material" these operators actually consume - the byte details of quantization formats.
</div>
""",
}

LESSON_12 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
课 06 我们建立了量化的<strong>直觉</strong>——"每一小块权重共享一个 scale，块内只存相对差"。这一课我们<strong>钻进字节级</strong>：打开 <span class="mono">ggml/src/ggml-common.h</span>，
看 <span class="mono">block_q4_0</span>、<span class="mono">block_q4_K</span> 这些块在内存里<strong>到底每个字节装了什么</strong>，再看解量化函数怎么把这堆字节还原成浮点，
最后看 <span class="mono">ggml_type_traits</span> 怎么把几十种量化类型<strong>统一接进引擎</strong>。
</p>
<p style="color:var(--muted);margin-top:.4rem">L06 回答的是"<strong>为什么能压、压完省多少</strong>"，这一课回答的是"<strong>压完在内存里长什么样、源码怎么把它读回来</strong>"——
两课正好是同一件事的"直觉面"与"实现面"。如果 L06 的块量化直觉你还有点模糊，建议先回头扫一眼；这一课会直接落到结构体的每一个字段上。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  一个量化 block 很像一个<strong>压缩包</strong>：开头几个字节是"解压参数"（scale、min），告诉你怎么把后面的数据放大还原；后面一长串是"压缩数据"（一个个挤在一起的低位量化值）。
  解量化，就是照着开头的参数，把后面每个小整数<strong>乘回它该有的大小</strong>。不同的量化格式，无非是"解压参数怎么记、数据怎么打包"的不同约定——读懂了字节布局，你就读懂了一种格式。
</div>

<h2>打开一个 block：q4_0 与 q8_0</h2>
<p>先看最经典的两个块。<span class="mono">q4_0</span> 把 <strong>32 个权重</strong>打成一块：开头 2 字节是一个 <span class="mono">half</span>（FP16 精度）的 scale，紧跟着 16 字节装下 32 个 4-bit 量化值
（每字节塞两个 nibble）。算下来一块 <strong>18 字节</strong>。<span class="mono">q8_0</span> 同样 32 个权重一块，但每个权重用一整字节的 <span class="mono">int8</span> 存、<strong>不打包</strong>——一块 <strong>34 字节</strong>
（2 字节 scale + 32 字节量化值），更准也更大。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>q4_0 / q8_0 字节布局</b>：开头是 scale，后面是量化值；q4_0 把 4-bit 打包，q8_0 用 int8 不打包</div>
  <div class="cells"><span class="lab">q4_0</span><span class="cell hl">d : 2B</span><span class="cell">qs : 16B（32 个 4-bit）</span><span class="lab">= 18 B / 32 权重</span></div>
  <div class="cells"><span class="lab">q8_0</span><span class="cell hl">d : 2B</span><span class="cell">qs : 32B（32 个 int8）</span><span class="lab">= 34 B / 32 权重</span></div>
</div>
<p>这些 block 在文件和内存里是<strong>一块紧挨着一块、连续排放</strong>的：一个权重矩阵就是一长串 block。知道了"一块多少字节"，就能算出"第 n 块在哪"——这正是后面解量化时能<strong>随机定位、并行处理</strong>的基础。
布局规整带来的好处，远不止省空间——它让"按需取一小块来解量化"成为可能，这也是后面"边算边解"能做到的前提。</p>
<p>对应到源码，结构体几乎和上图一一对应（<span class="mono">ggml/src/ggml-common.h</span>）：</p>
<pre class="code"><span class="cm">// 简化自 ggml/src/ggml-common.h</span>
<span class="kw">#define</span> QK4_0 32
<span class="kw">typedef struct</span> { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;  <span class="cm">// 2 + 16 = 18 B</span>
<span class="kw">typedef struct</span> { ggml_half d; int8_t  qs[QK8_0];   } block_q8_0;  <span class="cm">// 2 + 32 = 34 B</span></pre>
<p>逐个字段看：<span class="mono">d</span> 是那个块共享的 scale，<span class="mono">ggml_half</span> 就是 2 字节的半精度浮点（L06 说的"基准"）；<span class="mono">qs</span> 是量化值数组。
q4_0 的 <span class="mono">qs[QK4_0/2]</span>=<span class="mono">qs[16]</span>——32 个值只占 16 字节，因为每个值才 4 bit，<strong>两个挤进一个字节</strong>（一个放高 nibble、一个放低 nibble）。
q8_0 的 <span class="mono">qs[QK8_0]</span>=<span class="mono">qs[32]</span>——32 个值占满 32 字节，<strong>一个值独占一字节</strong>，不用拆位。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  有人会问：那个 scale 为什么用 2 字节的 <span class="mono">half</span>，而不是更精确的 4 字节 float？因为 scale <strong>每块才一个</strong>，它的精度对最终结果影响有限，却要乘进整块的体积里——用 half 省下的这 2 字节，摊到 32 个权重上虽小，乘以一个模型里成千上万个块，省下来的就很可观了。这是一个典型的工程取舍：在"几乎不影响精度"的地方<strong>能省则省</strong>，把字节预算留给真正重要的量化值。
</div>
<p>为什么 q8_0 更准也更大？因为 4-bit 只能表示 16 个档位、8-bit 能表示 256 个档位，<strong>同一个权重，8-bit 能贴得更近</strong>，量化误差更小。代价是体积翻倍：每权重从 0.5 字节涨到 1 字节。
这正是 L06 那张"显存对照表"背后的字节级真相——你在命令行里选 <span class="mono">Q4_0</span> 还是 <span class="mono">Q8_0</span>，本质就是在这两种 block 布局之间二选一。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里值得停下来想一个问题：为什么要<strong>分块</strong>，不干脆整个权重矩阵共享一个 scale？因为一个几千乘几千的大矩阵里，权重的大小范围差异很大——某些行很大、某些行很小。若全矩阵共用一个 scale，就得迁就那个最大值，小权重会被压得几乎只剩 0、精度全丢。分成 32 个一块后，<strong>每块各自找自己的 scale</strong>，大块用大 scale、小块用小 scale，量化误差立刻小一大截。这就是"块量化"四个字的全部用意：用"每块一个 scale"的少量开销，换回"局部精度"的大幅提升。
</div>
<p>还有个常被忽略的细节：q4_0 一块 18 字节、装 32 个权重，平均每权重 <strong>4.5 bit</strong>，而不是正好 4 bit。多出来的 0.5 bit，就是那 2 字节 scale 摊到 32 个权重头上的开销。
块越大，这份"管理开销"摊得越薄——这也为下一节 K-quant 用 256 的大超块埋下了伏笔。</p>

<h2>super-block：K-quant 的两层 scale</h2>
<p>q4_0 每 32 个权重配一个 scale，已经不错了，但还能更准。K-quant（带 K 的格式，如 <span class="mono">q4_K</span>、<span class="mono">q6_K</span>）的思路是：用一个 <strong>256 个权重的"超块"</strong>，
里面再切成若干个小子块，做成<strong>两层 scale</strong>——超块给一个"整体基准"，每个子块再各自记一个"细调 scale"。这样既摊薄了基准开销，又保住了局部精度。</p>
<p>以 <span class="mono">q4_K</span> 为例：256 个权重切成 8 个子块（每子块 32 个权重）。超块层有 <span class="mono">d</span> 和 <span class="mono">dmin</span> 两个 half；子块层有 8 组 6-bit 的 scale/min，
打包进 12 字节的 <span class="mono">scales[]</span>；再加 128 字节装 256 个 4-bit 量化值。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">超块</span><span class="name">d, dmin（各 1 个 half）</span></div><div class="ld">整个 256 权重共享的"整体 scale / 整体 min"</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">子块</span><span class="name">scales[12]：8 组 6-bit scale/min</span></div><div class="ld">每 32 个权重一组，各自细调，比单层 scale 贴得更准</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">数据</span><span class="name">qs[128]：256 个 4-bit 量化值</span></div><div class="ld">真正的权重量化值，打包存放</div></div>
</div>
<p>那 12 字节的 <span class="mono">scales[]</span> 是怎么来的？8 个子块、每块要存一个 scale 和一个 min，本可以各用一字节、共 16 字节；但 K-quant 把它们再压成 <strong>6-bit</strong>，
8×(6+6)=96 bit=12 字节，又省下 4 字节。<strong>连"管理用的 scale"本身都要量化</strong>——这种把每一处冗余都榨干的抠门劲，正是量化格式设计的日常。</p>
<p>两层 scale 是 K-quant 的灵魂。<strong>第一层</strong>（超块的 d/dmin）定个大致范围；<strong>第二层</strong>（子块的 6-bit scale/min）在这个范围里做局部微调——
哪一小段权重偏大偏小，子块 scale 立刻跟上。这就是 L06 spark 说的"同样的 bit 数，K-quant 往往更准"的<strong>字节级来源</strong>：精度不是靠多花 bit，而是靠<strong>更聪明地分配 scale</strong>。</p>
<pre class="code"><span class="cm">// 简化自 ggml/src/ggml-common.h（QK_K = 256, K_SCALE_SIZE = 12）</span>
<span class="kw">typedef struct</span> {
    ggml_half d;                   <span class="cm">// 超块整体 scale</span>
    ggml_half dmin;                <span class="cm">// 超块整体 min</span>
    uint8_t   scales[K_SCALE_SIZE];<span class="cm">// 8 个子块的 6-bit scale/min（无 qh）</span>
    uint8_t   qs[QK_K/2];          <span class="cm">// 256 个 4-bit 量化值</span>
} block_q4_K;</pre>
<p>把 q4_K 的字节数也算一下：2（d）+ 2（dmin）+ 12（scales）+ 128（qs）= <strong>144 字节</strong>装 256 个权重，平均每权重 144×8/256 = <strong>4.5 bit</strong>——和 q4_0 一样的位宽，精度却更高。
这就是"两层 scale"最直接的回报：没多花一个 bit，纯靠结构更精巧把误差压了下去。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  注意 <span class="mono">q4_K</span> 里<strong>没有 qh</strong>——只有 d、dmin、scales、qs 四个字段，这点很容易记错（见深挖 2）。而 <span class="mono">q6_K</span> 就<strong>有 qh</strong>：6-bit 的量化值被拆成"低 4 位"放 <span class="mono">ql</span>、"高 2 位"放 <span class="mono">qh</span>，再配 16 个 8-bit 的子块 scale 和一个超块 d。不同 K-quant 的字段并不统一，<strong>别想当然套用</strong>。
</div>
<p>顺带认识一下整个 K-quant 家族：从 <span class="mono">q2_K</span>、<span class="mono">q3_K</span> 一直到 <span class="mono">q6_K</span>，名字里的数字是大致的位宽，<strong>K</strong> 则代表都用"超块 + 两层 scale"这套结构。
位宽越低（如 q2_K）压得越狠、精度越险，往往只敢用在不那么敏感的层上；位宽越高（如 q6_K）越接近原始精度。这也是为什么实际下载模型时，你会看到 <span class="mono">Q4_K_M</span>、<span class="mono">Q5_K_S</span> 这种带后缀的名字——
它们是把<strong>不同层用不同 K-quant 档位</strong>混搭出来的方案，在体积和质量之间取不同的平衡点。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  多说一句那个 <span class="mono">dmin</span>。q4_0 只有一个 scale，量化是<strong>对称</strong>的（围绕 0）；而 q4_K 多了一个 min，做的是<strong>带偏移</strong>的量化——还原公式更像 <span class="mono">x = scale · q + min</span>（源码注释写的就是 "weight is represented as x = a·q + b"）。为什么要这个偏移？因为很多权重的分布并不以 0 为中心，有了 min 就能让量化区间<strong>整体平移</strong>去贴合真实分布，进一步压低误差。这是 K-quant 比 q4_0 更准的另一半原因：不只 scale 更细，连"零点"都能调。
</div>
<p>为什么超块要选 256 这么大？因为超块越大，"整体 d/dmin"这份固定开销摊到的权重越多、每权重的开销越小；而局部精度由更细的子块 scale 兜底，不会因为块大而变糊。
大块负责"高压缩"、子块负责"高精度"，两者兼得——这就是 K-quant 在相同位宽下能比 q4_0 更准的设计动机（详见深挖 1）。</p>

<h2>解量化：把字节还原成浮点</h2>
<p>权重以量化字节存着，可真要参与计算（比如喂进 L11 的 mul_mat），得先<strong>还原成浮点</strong>。这一步叫<strong>解量化</strong>（dequantize），由每种格式各自的 <span class="mono">dequantize_row_*</span> 函数负责
（在 <span class="mono">ggml/src/ggml-quants.c</span>）。以 q4_0 为例，逻辑出奇地简单：</p>
<pre class="code"><span class="cm">// 伪代码, 对应 ggml/src/ggml-quants.c 的 dequantize_row_q4_0</span>
<span class="kw">for</span> each block:
    d = half_to_float(block.d)          <span class="cm">// 取回这块的 scale</span>
    <span class="kw">for</span> j <span class="kw">in</span> 0..15:                     <span class="cm">// 每字节两个 nibble</span>
        q0 = (block.qs[j] &amp; 0x0F) - 8   <span class="cm">// 低 4 位, 减 8 回到有符号</span>
        q1 = (block.qs[j] &gt;&gt; 4)   - 8   <span class="cm">// 高 4 位, 减 8 回到有符号</span>
        y[j]      = q0 * d              <span class="cm">// x = (q - 8) * d</span>
        y[j + 16] = q1 * d</pre>
<div class="flow">
  <div class="node"><div class="nt">qs 取一个 nibble</div><div class="nd">q = 0..15</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">q - 8</div><div class="nd">平移成 -8..7<br>(有符号)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">× d</div><div class="nd">乘这块的 scale</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">float x</div><div class="nd">x = (q - 8) · d</div></div>
</div>
<p>拿一个真实的块走一遍，上面这条链路就具体了：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次解量化</b>：一个 q4_0 块的几个字节怎么还原成浮点（d=0.05 为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 字节 qs[j]</h5>
      <div class="cellrow"><span class="vc">0x96</span><span class="vc">0xC3</span></div>
      <div class="tlab">两个量化字节</div></div>
    <div class="op">&amp; 0x0F<br>&gt;&gt;4</div>
    <div class="stn"><h5>② 拆 nibble</h5>
      <div class="cellrow"><span class="vc">6</span><span class="vc">9</span><span class="vc">3</span><span class="vc">12</span></div>
      <div class="tlab">每字节拆出两个 4-bit 码</div></div>
    <div class="op">q - 8</div>
    <div class="stn"><h5>③ 减 8</h5>
      <div class="cellrow"><span class="vc">-2</span><span class="vc">+1</span><span class="vc">-5</span><span class="vc">+4</span></div>
      <div class="tlab">平移成有符号 -8..7</div></div>
    <div class="op">× d<br>d=0.05</div>
    <div class="stn"><h5>④ × d</h5>
      <div class="cellrow"><span class="vc blue">-.10</span><span class="vc blue">+.05</span><span class="vc blue">-.25</span><span class="vc blue">+.20</span></div>
      <div class="tlab">还原出近似浮点</div></div>
  </div>
</div>
<p>核心就一行：<span class="mono">x = (q - 8) * d</span>。<span class="mono">q</span> 是那个 0..15 的 4-bit 量化值，<strong>减 8</strong> 把它平移成 -8..7 的有符号数，再乘上这块的 scale <span class="mono">d</span>，
就还原出近似的原始浮点。q8_0 更直接：<span class="mono">x = q * d</span>，因为 int8 本身就是有符号的，不用减偏移。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  那个"<strong>减 8</strong>"也值得多想一层。4-bit 能存 0..15 共 16 个数，但权重有正有负，所以约定让 8 代表"0"、比 8 小的是负、比 8 大的是正——减去 8，就把 0..15 平移成 -8..7 这个<strong>大致对称</strong>的区间。这样正负权重都能表示，且 0 附近分布得最密（量化最准），恰好契合"大多数权重都集中在 0 附近"的事实。一个小小的减法，背后是对权重分布的理解。
</div>
<p>反过来，<strong>量化</strong>（把浮点压成字节）就是解量化的逆运算。q4_0 的参考实现 <span class="mono">quantize_row_q4_0_ref</span> 里，scale 取 <span class="mono">d = max / -8</span>——找出这块绝对值最大的权重，
让它对应到量化区间的端点（-8），其余权重按比例缩放取整。<strong>除以 -8 而不是 8</strong> 是个容易看走眼的细节：它和解量化时的"减 8"配套，保证一来一回数值对得上。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  顺势说清一件事：解量化是<strong>有损</strong>的——还原出来的浮点和原始权重并不完全相等，差的那一点就是量化误差。但为什么模型还能照常工作？因为深度网络对权重的微小扰动相当<strong>宽容</strong>：单个权重差一点点，经过成百上千次累加和非线性，整体输出几乎察觉不到。量化格式的全部艺术，就是在"压得更狠"和"误差还能被模型容忍"之间找平衡——这也是为什么会有 q4_0、q4_K、q6_K 这么多档位，让你按对精度的需求挑一个合适的折中点。
</div>
<p>还有一点值得记住：解量化不一定<strong>提前一次性</strong>把整个权重张量铺开成浮点（那会占很多内存）。在矩阵乘这种热点里，ggml 常常是<strong>边算边解</strong>——在内层循环里即时把用到的那一小块还原成浮点、立刻参与点积，
算完就丢。所以量化省的不只是显存，连"解压后的浮点"也大多不落地，带宽和缓存都跟着受益（呼应 L06 说的"省显存又提速"）。</p>

<h2>接进引擎：ggml_type_traits</h2>
<p>问题来了：q4_0、q8_0、q4_K、q6_K…… 几十种量化格式，字节布局各不相同，难道每种都要在 mul_mat、解量化里写一遍特判？当然不。ggml 用一张 <strong>类型特征表</strong>（<span class="mono">ggml_type_traits</span>）
把每种类型的"基本参数 + 怎么转 float"<strong>登记成一行</strong>，算子只管查表，不关心具体是哪种量化。</p>
<div class="flow">
  <div class="node"><div class="nt">tensor.type</div><div class="nd">= GGML_TYPE_Q4_K</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">type_traits[Q4_K]</div><div class="nd">blck_size / type_size<br>to_float = dequantize_row_q4_K</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">算子</div><div class="nd">按 traits 解量化后计算</div></div>
</div>
<p>张量只在 <span class="mono">type</span> 字段记着"我是 Q4_K"（L05 的 type 字段，到这里终于派上大用场）。算子要用它时，拿这个 type 去查 <span class="mono">type_traits</span> 表，就能拿到"这种类型一块多少元素、多少字节、用哪个函数转 float"，
照着做即可——<strong>算子代码里没有一个 if 在判断量化类型</strong>。</p>
<pre class="code"><span class="cm">// 简化自 ggml/include/ggml.h（每种 type 登记一行）</span>
<span class="kw">struct</span> ggml_type_traits {
    <span class="kw">const char</span> * type_name;     <span class="cm">// "q4_K"</span>
    int64_t       blck_size;     <span class="cm">// 一块多少元素, 如 256</span>
    size_t        type_size;     <span class="cm">// 一块多少字节, 如 144</span>
    bool          is_quantized;
    ggml_to_float_t   to_float;       <span class="cm">// 解量化函数指针</span>
    ggml_from_float_t from_float_ref; <span class="cm">// 量化函数指针(参考实现)</span>
};</pre>
<p>关键就在那两个<strong>函数指针</strong>：<span class="mono">to_float</span> 指向这种类型的解量化函数，<span class="mono">from_float_ref</span> 指向量化（参考）函数。算子拿到张量，按 traits 调 <span class="mono">to_float</span> 解量化、再计算；
要存盘时按 <span class="mono">from_float_ref</span> 量化。<strong>注意字段名是 <span class="mono">from_float_ref</span></strong>（带 _ref，参考实现），别记成 from_float。于是<strong>加一种新量化格式，主要工作就是填一行 traits + 写好这两个函数</strong>，
mul_mat、解量化那些算子代码<strong>基本不用动</strong>——这正是 L05 "结构不变、类型可换"在量化上的兑现。</p>

<p>退一步看，这一课其实把"量化"这件事讲全了三个层次：<strong>布局</strong>（block 里每个字节装什么）、<strong>还原</strong>（解量化怎么把字节变回浮点）、<strong>接入</strong>（traits 表怎么让算子统一处理）。
这三层正好对应你用 llama.cpp 时会碰到的三种场景：挑量化档位（布局决定体积与精度）、跑推理（解量化在背后默默进行）、以及读懂源码（traits 是所有量化类型的总入口）。
把这三层串起来，你就不再把 <span class="mono">Q4_K_M</span> 这种名字当成黑盒，而能说清它在内存里到底是什么、算的时候发生了什么。把一个量化名字拆解到字节这一层，你对"模型怎么落到磁盘和内存里"的理解，就又扎实了一截。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么 q4_K 用 256 的超块，而不是像 q4_0 那样 32？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>核心是<strong>摊薄固定开销 + 不牺牲局部精度</strong>。q4_0 每 32 个权重就得花 2 字节存一个 scale；超块做到 256，"整体 d/dmin"这份固定开销摊到的权重多了 8 倍，每权重的管理开销明显下降。</p>
    <p>但只把块放大、还用单层 scale 的话，局部精度会变差——256 个权重共享一个 scale，太粗了。K-quant 的解法是<strong>再加一层子块 scale</strong>：超块定大范围、子块定细节，于是"块大带来的高压缩"和"子块带来的高精度"同时拿到。</p>
    <p>这也解释了为什么不无脑把超块做到更大（比如 1024）：子块 scale 本身也要占空间，块太大、子块太多，第二层开销又上来了。256 + 8 子块是工程上压缩率与精度的一个甜点，经过大量实测调出来的。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> q4_K 有 qh 吗？q6_K 呢？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">q4_K</span> <strong>没有 qh</strong>。它的字段只有四个：<span class="mono">d</span>、<span class="mono">dmin</span>、<span class="mono">scales[12]</span>、<span class="mono">qs[128]</span>。4-bit 量化值正好一个 nibble，直接打包进 <span class="mono">qs</span>，不需要再拆高低位。</p>
    <p><span class="mono">q6_K</span> 才<strong>有 qh</strong>。6-bit 一个值放不进一个 nibble，于是拆开：低 4 位存进 <span class="mono">ql</span>、高 2 位存进 <span class="mono">qh</span>，再加 16 个 8-bit 的子块 scale 和一个超块 <span class="mono">d</span>。
    所以 q6_K 的结构和 q4_K 长得很不一样。</p>
    <p>这是初学者最容易栽的坑之一：以为所有 K-quant 字段都一样、把 q4_K 想象成"带 qh"。读这些结构体时一定<strong>对着源码逐字段核对</strong>，别凭格式名想当然——位宽不同、打包方式就不同，字段自然不同。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 算子怎么做到不为每种量化各写一遍？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>靠 <span class="mono">type_traits</span> 里的 <span class="mono">to_float</span> / <span class="mono">from_float_ref</span> 这两个<strong>函数指针</strong>。算子拿到一个量化张量，不去判断"这是 q4_0 还是 q4_K"，而是直接调它 traits 里登记的 <span class="mono">to_float</span> 解量化，再算。</p>
    <p>在 mul_mat 这种热点里更讲究：往往不是先整块解量化、再乘，而是在内层循环里<strong>即时解一小段、立刻点积</strong>，甚至为某些量化类型配了专门的高速点积内核。但对外暴露的接口是统一的——都是"按 traits 拿到怎么转 float"。</p>
    <p>结果就是：加一种新量化类型，<strong>算子代码一行都不用改</strong>，只要在 traits 表里填一行、写好解/量化函数。这种"用函数指针把差异收进一张表"的做法，正是 ggml 能容纳几十种量化格式还不臃肿的关键，和 L11 "switch(op) 派发算子"是同一种解耦思路。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li><span class="mono">q4_0</span> = [d 2B][qs 16B] = <strong>18 B / 32 权重</strong>（4-bit 打包）；<span class="mono">q8_0</span> 每权重一个 int8，一块 34 B，更准更大。</li>
    <li>K-quant 用 <strong>256 的超块 + 两层 scale</strong>（超块 d/dmin + 子块 6-bit），同位宽下比 q4_0 更准。</li>
    <li><span class="mono">q4_K</span> <strong>无 qh</strong>（只有 d/dmin/scales/qs）；<span class="mono">q6_K</span> 才有 ql+qh。别想当然套用。</li>
    <li>解量化核心一行：<span class="mono">x = (q - 8) * d</span>；量化方向 <span class="mono">d = max / -8</span>。</li>
    <li><span class="mono">ggml_type_traits</span> 用 <span class="mono">to_float</span> / <span class="mono">from_float_ref</span> 函数指针把每种类型接进引擎，算子<strong>无需为每种量化各写一遍</strong>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"<strong>每种量化类型长什么样、怎么转 float</strong>"全收进一张 traits 表、用函数指针暴露出来——于是几十种量化格式能共用同一套张量结构和同一批算子。L05 说"结构不变、类型可换"，
  到这里你看到了它在字节级的兑现：换格式只是换一行表项，引擎的主干<strong>纹丝不动</strong>。下一课，我们把视线从"一个张量怎么存"抬到"整个模型文件怎么存"——GGUF 格式。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Lesson 06 built the <strong>intuition</strong> of quantization - "each small block of weights shares one scale, and stores only relative offsets inside the block". This lesson goes
<strong>down to the byte level</strong>: we open <span class="mono">ggml/src/ggml-common.h</span> to see what every byte of blocks like <span class="mono">block_q4_0</span> and <span class="mono">block_q4_K</span>
<strong>actually holds</strong>, then how the dequantize functions restore those bytes back to floats, and finally how <span class="mono">ggml_type_traits</span> wires dozens of quantization types
<strong>uniformly into the engine</strong>.
</p>
<p style="color:var(--muted);margin-top:.4rem">L06 answered "<strong>why we can compress, and how much it saves</strong>"; this lesson answers "<strong>what it looks like in memory afterward, and how the source reads it back</strong>" -
the two are the "intuition side" and the "implementation side" of one thing. If L06's block-quantization intuition is still fuzzy, glance back first; this lesson lands directly on every struct field.</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  A quantization block is much like a <strong>zip archive</strong>: the first few bytes are the "decompression parameters" (scale, min) that tell you how to scale the following data back up; the long run
  after is the "compressed data" (low-bit quantized values packed together). Dequantizing means following those leading parameters to <strong>multiply each small integer back to its true size</strong>.
  Different quantization formats are just different conventions for "how the parameters are recorded and how the data is packed" - read the byte layout and you have read the format.
</div>

<h2>Open one block: q4_0 and q8_0</h2>
<p>Start with the two most classic blocks. <span class="mono">q4_0</span> packs <strong>32 weights</strong> into a block: the first 2 bytes are a <span class="mono">half</span> (FP16) scale, followed by 16 bytes
holding 32 4-bit quantized values (two nibbles per byte). That is <strong>18 bytes</strong> a block. <span class="mono">q8_0</span> also groups 32 weights, but stores each as a full byte of <span class="mono">int8</span>,
<strong>unpacked</strong> - <strong>34 bytes</strong> a block (2-byte scale + 32-byte values), more accurate and larger.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>q4_0 / q8_0 byte layout</b>: scale up front, values after; q4_0 packs 4-bit, q8_0 keeps int8 unpacked</div>
  <div class="cells"><span class="lab">q4_0</span><span class="cell hl">d : 2B</span><span class="cell">qs : 16B (32 x 4-bit)</span><span class="lab">= 18 B / 32 weights</span></div>
  <div class="cells"><span class="lab">q8_0</span><span class="cell hl">d : 2B</span><span class="cell">qs : 32B (32 x int8)</span><span class="lab">= 34 B / 32 weights</span></div>
</div>
<p>These blocks are laid out <strong>one right after another, contiguously</strong> in file and memory: a weight matrix is just a long run of blocks. Knowing "how many bytes per block" lets you compute "where block n is" - exactly the
basis for <strong>random addressing and parallel processing</strong> during dequantization later. A regular layout brings benefits far beyond saving space - it makes "fetch one small block on demand and dequantize it" possible, the very
precondition for the "dequantize on the fly" mentioned earlier.</p>
<p>In source, the struct matches the figure almost one-to-one (<span class="mono">ggml/src/ggml-common.h</span>):</p>
<pre class="code"><span class="cm">// simplified from ggml/src/ggml-common.h</span>
<span class="kw">#define</span> QK4_0 32
<span class="kw">typedef struct</span> { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;  <span class="cm">// 2 + 16 = 18 B</span>
<span class="kw">typedef struct</span> { ggml_half d; int8_t  qs[QK8_0];   } block_q8_0;  <span class="cm">// 2 + 32 = 34 B</span></pre>
<p>Field by field: <span class="mono">d</span> is the block's shared scale, and <span class="mono">ggml_half</span> is exactly a 2-byte half-precision float (L06's "baseline"); <span class="mono">qs</span> is the array of quantized
values. q4_0's <span class="mono">qs[QK4_0/2]</span>=<span class="mono">qs[16]</span> - 32 values in only 16 bytes, because each value is just 4 bits, <strong>two squeezed into one byte</strong> (one high nibble, one low). q8_0's
<span class="mono">qs[QK8_0]</span>=<span class="mono">qs[32]</span> - 32 values filling 32 bytes, <strong>one value per byte</strong>, no bit-splitting.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  One might ask: why is that scale a 2-byte <span class="mono">half</span> rather than a more precise 4-byte float? Because there is <strong>only one scale per block</strong>, its precision has limited impact on the final result, yet it counts against the whole block's size - the 2 bytes saved by using half, small per 32 weights, multiplied by the tens of thousands of blocks in a model, add up to a lot. This is a classic engineering trade-off: <strong>save wherever you can</strong> in places that "barely affect precision", leaving the byte budget for the quantized values that truly matter.
</div>
<p>Why is q8_0 both more accurate and larger? Because 4 bits can represent only 16 levels while 8 bits represent 256 levels, so for the same weight <strong>8-bit can sit much closer</strong>, with smaller
quantization error. The cost is double the size: from 0.5 byte per weight to 1 byte. This is the byte-level truth behind L06's "VRAM table" - choosing <span class="mono">Q4_0</span> vs <span class="mono">Q8_0</span>
on the command line is, in essence, picking between these two block layouts.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  It is worth pausing on one question here: why <strong>block</strong> at all, instead of sharing one scale across the whole weight matrix? Because in a matrix of thousands by thousands, weight magnitudes vary widely - some rows large, some small. With one scale for the whole matrix you must accommodate the largest value, and small weights get crushed almost to 0, losing all precision. Split into blocks of 32, <strong>each block finds its own scale</strong> - big blocks use a big scale, small blocks a small one - and quantization error drops sharply at once. That is the entire point of "block quantization": spend the small overhead of "one scale per block" to buy a large gain in local precision.
</div>
<p>One often-missed detail: a q4_0 block is 18 bytes for 32 weights, averaging <strong>4.5 bits</strong> per weight, not exactly 4. The extra 0.5 bit is the 2-byte scale amortized across 32 weights. The bigger the
block, the thinner this "management overhead" spreads - which already foreshadows why the next section's K-quant uses a big 256-weight super-block.</p>

<h2>Super-block: the two-level scale of K-quant</h2>
<p>q4_0's one scale per 32 weights is already decent, but it can be more accurate. The idea of K-quant (the K formats, such as <span class="mono">q4_K</span> and <span class="mono">q6_K</span>) is to use a
<strong>256-weight "super-block"</strong>, sliced into several sub-blocks, with a <strong>two-level scale</strong> - the super-block gives an "overall baseline", and each sub-block records its own "fine-tuning scale".
This both thins the baseline overhead and preserves local precision.</p>
<p>Take <span class="mono">q4_K</span>: 256 weights split into 8 sub-blocks (32 weights each). The super-block level has two halves, <span class="mono">d</span> and <span class="mono">dmin</span>; the sub-block level has 8 groups
of 6-bit scale/min packed into a 12-byte <span class="mono">scales[]</span>; plus 128 bytes holding 256 4-bit quantized values.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">super</span><span class="name">d, dmin (one half each)</span></div><div class="ld">the "overall scale / overall min" shared by all 256 weights</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">sub</span><span class="name">scales[12]: 8 groups of 6-bit scale/min</span></div><div class="ld">one group per 32 weights, each fine-tuned, sitting closer than a single-level scale</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">data</span><span class="name">qs[128]: 256 4-bit quantized values</span></div><div class="ld">the actual quantized weight values, packed</div></div>
</div>
<p>Where does the 12-byte <span class="mono">scales[]</span> come from? 8 sub-blocks, each needing a scale and a min, could use one byte each for 16 bytes total; but K-quant squeezes them to <strong>6 bits</strong>, 8x(6+6)=96 bits=12 bytes,
saving another 4 bytes. <strong>Even the "management" scales themselves are quantized</strong> - this squeezing of every last bit of redundancy is the daily business of quantization-format design.</p>
<p>The two-level scale is the soul of K-quant. The <strong>first level</strong> (the super-block's d/dmin) sets a rough range; the <strong>second level</strong> (the sub-block's 6-bit scale/min) fine-tunes locally within
that range - wherever a small run of weights skews high or low, the sub-block scale follows immediately. This is the <strong>byte-level source</strong> of L06's spark "for the same bit count, K-quant is usually more
accurate": precision comes not from spending more bits, but from <strong>allocating scale more cleverly</strong>.</p>
<pre class="code"><span class="cm">// simplified from ggml/src/ggml-common.h (QK_K = 256, K_SCALE_SIZE = 12)</span>
<span class="kw">typedef struct</span> {
    ggml_half d;                   <span class="cm">// super-block overall scale</span>
    ggml_half dmin;                <span class="cm">// super-block overall min</span>
    uint8_t   scales[K_SCALE_SIZE];<span class="cm">// 8 sub-blocks' 6-bit scale/min (no qh)</span>
    uint8_t   qs[QK_K/2];          <span class="cm">// 256 4-bit quantized values</span>
} block_q4_K;</pre>
<p>Compute q4_K's byte count too: 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) = <strong>144 bytes</strong> for 256 weights, averaging 144*8/256 = <strong>4.5 bits</strong> per weight - the same bit width as q4_0, yet higher precision. That is
the most direct payoff of the "two-level scale": not one extra bit spent, error pushed down purely by a smarter structure.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  Note that <span class="mono">q4_K</span> has <strong>no qh</strong> - only the four fields d, dmin, scales, qs, which is easy to misremember (see Dig deeper 2). <span class="mono">q6_K</span>, by contrast, <strong>does have qh</strong>: its 6-bit values are split into "low 4 bits" in <span class="mono">ql</span> and "high 2 bits" in <span class="mono">qh</span>, plus 16 8-bit sub-block scales and one super-block d. Fields differ across K-quants, so <strong>do not assume</strong>.
</div>
<p>Worth meeting the whole K-quant family: from <span class="mono">q2_K</span> and <span class="mono">q3_K</span> up to <span class="mono">q6_K</span>, the number in the name is the rough bit width, while <strong>K</strong> means they all use this
"super-block + two-level scale" structure. The lower the bit width (like q2_K) the harder the compression and the riskier the precision, so it is often used only on less sensitive layers; the higher (like q6_K) the closer to original
precision. This is why, downloading real models, you see suffixed names like <span class="mono">Q4_K_M</span> and <span class="mono">Q5_K_S</span> - schemes that <strong>mix different K-quant levels across different layers</strong>, striking
different balances between size and quality.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  A word on that <span class="mono">dmin</span>. q4_0 has only a scale, so its quantization is <strong>symmetric</strong> (around 0); q4_K adds a min, doing <strong>offset</strong> quantization - the restore formula is more like <span class="mono">x = scale * q + min</span> (the source comment reads "weight is represented as x = a*q + b"). Why the offset? Because many weight distributions are not centered on 0, and a min lets the quantization range <strong>shift as a whole</strong> to fit the real distribution, pushing error down further. This is the other half of why K-quant beats q4_0: not only finer scale, but even the "zero point" is adjustable.
</div>
<p>Why pick a super-block as big as 256? Because the bigger the super-block, the more weights the fixed "overall d/dmin" overhead spreads across, and the smaller the per-weight overhead; meanwhile local precision is
backstopped by the finer sub-block scales, so a big block does not get blurry. The big block handles "high compression", the sub-blocks handle "high precision", and you get both - this is the design motive for K-quant
being more accurate than q4_0 at the same bit width (see Dig deeper 1).</p>

<h2>Dequantize: restoring bytes to floats</h2>
<p>Weights sit in memory as quantized bytes, but to actually take part in computation (say, fed into L11's mul_mat) they must first be <strong>restored to floats</strong>. That step is <strong>dequantization</strong>,
handled by each format's own <span class="mono">dequantize_row_*</span> function (in <span class="mono">ggml/src/ggml-quants.c</span>). For q4_0 the logic is surprisingly simple:</p>
<pre class="code"><span class="cm">// pseudocode, mirrors dequantize_row_q4_0 in ggml/src/ggml-quants.c</span>
<span class="kw">for</span> each block:
    d = half_to_float(block.d)          <span class="cm">// fetch this block's scale</span>
    <span class="kw">for</span> j <span class="kw">in</span> 0..15:                     <span class="cm">// two nibbles per byte</span>
        q0 = (block.qs[j] &amp; 0x0F) - 8   <span class="cm">// low 4 bits, minus 8 back to signed</span>
        q1 = (block.qs[j] &gt;&gt; 4)   - 8   <span class="cm">// high 4 bits, minus 8 back to signed</span>
        y[j]      = q0 * d              <span class="cm">// x = (q - 8) * d</span>
        y[j + 16] = q1 * d</pre>
<div class="flow">
  <div class="node"><div class="nt">take a nibble from qs</div><div class="nd">q = 0..15</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">q - 8</div><div class="nd">shift to -8..7<br>(signed)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">x d</div><div class="nd">times this block's scale</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">float x</div><div class="nd">x = (q - 8) * d</div></div>
</div>
<p>Run one real block through it and the chain above gets concrete:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one dequant</b>: how a few bytes of a q4_0 block become floats (d=0.05 illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) byte qs[j]</h5>
      <div class="cellrow"><span class="vc">0x96</span><span class="vc">0xC3</span></div>
      <div class="tlab">two quantized bytes</div></div>
    <div class="op">&amp; 0x0F<br>&gt;&gt;4</div>
    <div class="stn"><h5>(2) split nibble</h5>
      <div class="cellrow"><span class="vc">6</span><span class="vc">9</span><span class="vc">3</span><span class="vc">12</span></div>
      <div class="tlab">two 4-bit codes per byte</div></div>
    <div class="op">q - 8</div>
    <div class="stn"><h5>(3) minus 8</h5>
      <div class="cellrow"><span class="vc">-2</span><span class="vc">+1</span><span class="vc">-5</span><span class="vc">+4</span></div>
      <div class="tlab">shift to signed -8..7</div></div>
    <div class="op">x d<br>d=0.05</div>
    <div class="stn"><h5>(4) x d</h5>
      <div class="cellrow"><span class="vc blue">-.10</span><span class="vc blue">+.05</span><span class="vc blue">-.25</span><span class="vc blue">+.20</span></div>
      <div class="tlab">restored approximate floats</div></div>
  </div>
</div>
<p>The core is one line: <span class="mono">x = (q - 8) * d</span>. <span class="mono">q</span> is the 0..15 4-bit value; <strong>subtracting 8</strong> shifts it to a signed -8..7, then
multiplying by this block's scale <span class="mono">d</span> restores the approximate original float. q8_0 is even more direct: <span class="mono">x = q * d</span>, since int8 is already signed and needs no offset.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  That "<strong>minus 8</strong>" deserves a second thought too. 4 bits store 0..15, sixteen values, but weights are both positive and negative, so the convention makes 8 mean "0", below 8 negative, above 8 positive - subtracting 8 shifts 0..15 into the <strong>roughly symmetric</strong> range -8..7. This represents both signs, and is densest (most accurate) near 0, matching the fact that "most weights cluster near 0". A tiny subtraction encodes an understanding of the weight distribution.
</div>
<p>In reverse, <strong>quantization</strong> (compressing floats to bytes) is the inverse. In q4_0's reference implementation <span class="mono">quantize_row_q4_0_ref</span>, the scale is <span class="mono">d = max / -8</span> - find the
weight with the largest magnitude in the block, map it to the endpoint of the quantization range (-8), and scale-and-round the rest proportionally. <strong>Dividing by -8 rather than 8</strong> is an easy-to-misread detail:
it pairs with the "minus 8" at dequantize time, ensuring the round trip lines up numerically.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  To make one thing explicit: dequantization is <strong>lossy</strong> - the restored floats are not exactly equal to the original weights, and that small gap is the quantization error. So why does the model still work? Because deep networks are quite <strong>tolerant</strong> of tiny perturbations to weights: a single weight off by a hair, after hundreds or thousands of accumulations and nonlinearities, is almost imperceptible in the overall output. The whole art of a quantization format is balancing "compress harder" against "error the model can still tolerate" - which is exactly why there are so many levels like q4_0, q4_K, q6_K, letting you pick a suitable trade-off for your precision needs.
</div>
<p>One more thing worth remembering: dequantization does not necessarily <strong>expand the whole weight tensor to floats up front</strong> (that would cost a lot of memory). In hotspots like matmul, ggml often
<strong>dequantizes on the fly</strong> - restoring just the small block it needs in the inner loop, feeding it into the dot product immediately, and discarding it. So quantization saves not only VRAM; even the "decompressed
floats" mostly never land, benefiting bandwidth and cache too (echoing L06's "saves VRAM and speeds up").</p>

<h2>Wiring into the engine: ggml_type_traits</h2>
<p>Here is the problem: q4_0, q8_0, q4_K, q6_K... dozens of quantization formats, each with a different byte layout - must every one be special-cased inside mul_mat and dequantize? Of course not. ggml uses a
<strong>type-traits table</strong> (<span class="mono">ggml_type_traits</span>) to <strong>register each type as one row</strong> of "basic parameters + how to convert to float", and operators just look it up, never caring which
quantization it actually is.</p>
<div class="flow">
  <div class="node"><div class="nt">tensor.type</div><div class="nd">= GGML_TYPE_Q4_K</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">type_traits[Q4_K]</div><div class="nd">blck_size / type_size<br>to_float = dequantize_row_q4_K</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">operator</div><div class="nd">dequantize per traits, then compute</div></div>
</div>
<p>A tensor only records "I am Q4_K" in its <span class="mono">type</span> field (L05's type field finally earns its keep here). When an operator needs it, it looks that type up in the <span class="mono">type_traits</span> table to get
"how many elements per block, how many bytes, which function converts to float" and just follows it - <strong>there is not a single if in the operator code branching on quantization type</strong>.</p>
<pre class="code"><span class="cm">// simplified from ggml/include/ggml.h (one row per type)</span>
<span class="kw">struct</span> ggml_type_traits {
    <span class="kw">const char</span> * type_name;     <span class="cm">// "q4_K"</span>
    int64_t       blck_size;     <span class="cm">// elements per block, e.g. 256</span>
    size_t        type_size;     <span class="cm">// bytes per block, e.g. 144</span>
    bool          is_quantized;
    ggml_to_float_t   to_float;       <span class="cm">// dequantize function pointer</span>
    ggml_from_float_t from_float_ref; <span class="cm">// quantize function pointer (reference)</span>
};</pre>
<p>The key lies in those two <strong>function pointers</strong>: <span class="mono">to_float</span> points to this type's dequantize function, <span class="mono">from_float_ref</span> to its (reference) quantize function. An operator takes a
tensor, calls <span class="mono">to_float</span> per its traits to dequantize, then computes; when saving, it quantizes via <span class="mono">from_float_ref</span>. <strong>Note the field name is <span class="mono">from_float_ref</span></strong>
(with _ref, the reference implementation), not from_float. So <strong>adding a new quantization format is mainly filling one traits row + writing these two functions</strong>, while mul_mat and the dequantize operators
<strong>need almost no change</strong> - exactly L05's "structure fixed, type swappable" cashed out for quantization.</p>

<p>Stepping back, this lesson actually covers "quantization" at three levels: <strong>layout</strong> (what each byte in a block holds), <strong>restoration</strong> (how dequantize turns bytes back into floats), and
<strong>wiring</strong> (how the traits table lets operators handle everything uniformly). These three map neatly onto the three situations you meet using llama.cpp: picking a quantization level (layout decides size and precision),
running inference (dequantization happens quietly behind the scenes), and reading the source (traits is the single entry point for all quantization types). String the three together and you stop treating a name like
<span class="mono">Q4_K_M</span> as a black box - you can say exactly what it is in memory and what happens when it computes. Breaking a quantization name down to the byte level makes your grasp of "how a model lands on disk and in memory"
that much more solid.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why does q4_K use a 256 super-block instead of q4_0's 32? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The core is <strong>amortizing fixed overhead without sacrificing local precision</strong>. q4_0 spends 2 bytes on a scale for every 32 weights; bringing the super-block to 256 spreads the fixed "overall d/dmin"
    overhead over 8x as many weights, clearly lowering per-weight management cost.</p>
    <p>But simply enlarging the block with a single-level scale would hurt local precision - 256 weights sharing one scale is too coarse. K-quant's answer is to <strong>add a second sub-block scale</strong>: the super-block
    sets the broad range, the sub-blocks set details, so you get both "high compression from the big block" and "high precision from the sub-blocks".</p>
    <p>This also explains why we do not blindly make the super-block even bigger (say 1024): sub-block scales themselves take space, and too big a block with too many sub-blocks raises the second-level overhead again. 256 + 8
    sub-blocks is an engineering sweet spot between compression ratio and precision, tuned through extensive measurement.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Does q4_K have qh? What about q6_K? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">q4_K</span> has <strong>no qh</strong>. It has only four fields: <span class="mono">d</span>, <span class="mono">dmin</span>, <span class="mono">scales[12]</span>, <span class="mono">qs[128]</span>. A 4-bit value is exactly one nibble, packed
    straight into <span class="mono">qs</span>, with no need to split high and low bits.</p>
    <p><span class="mono">q6_K</span> is the one that <strong>has qh</strong>. A 6-bit value does not fit in one nibble, so it is split: the low 4 bits go into <span class="mono">ql</span>, the high 2 bits into <span class="mono">qh</span>, plus 16
    8-bit sub-block scales and one super-block <span class="mono">d</span>. So q6_K's struct looks quite different from q4_K's.</p>
    <p>This is one of the most common beginner traps: assuming all K-quant fields are identical and imagining q4_K "with a qh". When reading these structs, always <strong>check field by field against the source</strong>, never
    guessing from the format name - different bit widths mean different packing, and thus different fields.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How do operators avoid being rewritten for each quantization? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Through the two <strong>function pointers</strong> <span class="mono">to_float</span> / <span class="mono">from_float_ref</span> in <span class="mono">type_traits</span>. Given a quantized tensor, an operator does not test "is this q4_0 or
    q4_K"; it directly calls the <span class="mono">to_float</span> registered in its traits to dequantize, then computes.</p>
    <p>In hotspots like mul_mat it is subtler: rather than dequantizing the whole block first and then multiplying, it often <strong>dequantizes a small run on the fly and dot-products immediately</strong>, even with dedicated
    fast dot-product kernels for certain quantization types. But the exposed interface is uniform - all "get how to convert to float from traits".</p>
    <p>The result: adding a new quantization type needs <strong>not a single line changed in operator code</strong>, only one traits row plus the dequant/quant functions. This "fold the differences into a table of function
    pointers" approach is the key to ggml hosting dozens of quantization formats without bloat - the same decoupling idea as L11's "switch(op) dispatch".</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li><span class="mono">q4_0</span> = [d 2B][qs 16B] = <strong>18 B / 32 weights</strong> (4-bit packed); <span class="mono">q8_0</span> is one int8 per weight, 34 B a block, more accurate and larger.</li>
    <li>K-quant uses a <strong>256 super-block + two-level scale</strong> (super-block d/dmin + sub-block 6-bit), more accurate than q4_0 at the same bit width.</li>
    <li><span class="mono">q4_K</span> has <strong>no qh</strong> (only d/dmin/scales/qs); <span class="mono">q6_K</span> is the one with ql+qh. Do not assume.</li>
    <li>Dequant core, one line: <span class="mono">x = (q - 8) * d</span>; quantize direction <span class="mono">d = max / -8</span>.</li>
    <li><span class="mono">ggml_type_traits</span> wires each type into the engine via <span class="mono">to_float</span> / <span class="mono">from_float_ref</span> function pointers, so operators <strong>need not be rewritten per quantization</strong>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Folding "<strong>what each quantization type looks like and how to convert it to float</strong>" into one traits table, exposed via function pointers - so dozens of quantization formats can share the same tensor structure and
  the same operators. L05 said "structure fixed, type swappable"; here you see it cashed out at the byte level: switching format is just switching a table row, while the engine's trunk <strong>does not move at all</strong>. Next
  lesson, we lift our gaze from "how one tensor is stored" to "how a whole model file is stored" - the GGUF format.
</div>
""",
}

LESSON_13 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前面好几课都在提那个 <span class="mono">.gguf</span> 文件——模型就装在里面。可它到底长什么样？这一课我们把一个 GGUF 文件<strong>从头到尾拆开</strong>：文件头、元数据、张量清单、对齐填充、数据段，
再看 llama.cpp 怎么用 <strong>mmap 零拷贝</strong>地把它加载进内存、实现大模型的"秒加载"。这是第三部分的收尾课，把前面学的内存、图、算子、量化，<strong>落到磁盘上的一个文件</strong>里。
</p>
<p style="color:var(--muted);margin-top:.4rem">为什么值得专门讲文件格式？因为 GGUF 是 llama.cpp 的"<strong>统一入口</strong>"：你从 HuggingFace 下载、用 L02 的转换脚本得到的，就是这个文件；运行时被读进来的，也是它。
读懂 GGUF，你就把"磁盘上的模型"和"内存里的张量"两端连了起来。</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  GGUF 像一个<strong>带目录的集装箱</strong>：箱门上贴着一张清单（里面有哪些张量、各是什么规格、从第几米开始——这就是 metadata 和 tensor info），箱内则整整齐齐码放着货物（张量数据）。
  卸货时不用把整箱倒出来，照着清单<strong>直接定位</strong>到要的那一件——这正是 mmap 加载的精髓：按需取用，不做无谓搬运。
</div>

<h2>整体布局：一个 GGUF 文件从头到尾</h2>
<p>先看全景。一个 GGUF 文件从头到尾是这样排的：开头 4 字节是 magic <span class="mono">"GGUF"</span>（一眼认出"这是 GGUF"），接着 version（当前是 3），然后是张量数量和 KV 数量，
再往后是一串 metadata 键值对、一串 tensor info（张量清单），按对齐补一段 padding，最后才是<strong>真正的张量数据</strong>（很可能就是 L12 那些量化块）。</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>magic "GGUF"（4 字节）</h4><p>文件最开头的 4 个字节，一眼认出"这是一个 GGUF 文件"。</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>version = 3（u32）</h4><p>格式版本号，当前是 3；向后兼容靠它来判断。</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>n_tensors · n_kv（各 i64）</h4><p>先报数：后面有多少个张量、多少个键值对。</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>metadata KV ×n_kv</h4><p>自描述信息：架构、超参、词表、聊天模板……模型的"说明书"。</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>tensor infos ×n_tensors</h4><p>张量清单：每个张量的 name / dims / type / offset。</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>padding（对齐）</h4><p>补几个字节，让数据段起点对齐到 32 字节边界。</p></div></div>
  <div class="step"><div class="num">7</div><div class="sc"><h4>tensor data</h4><p>真正的权重字节，常常就是 L12 讲的那些量化块。</p></div></div>
</div>
<p>把这张图翻译成"伪结构"，就是这样（布局见 <span class="mono">ggml/include/gguf.h</span> 的头部注释）：</p>
<pre class="code"><span class="cm">// 简化自 ggml/include/gguf.h 的格式说明</span>
"GGUF"                                  <span class="cm">// 4 字节 magic</span>
version      : u32                      <span class="cm">// = 3</span>
n_tensors    : i64
n_kv         : i64
kv_pairs     : [ (key:str, type:gguf_type, value), ... ]              <span class="cm">// 自描述</span>
tensor_infos : [ (name:str, n_dims:u32, dims[]:i64, type, offset:u64), ... ]
&lt;padding to alignment&gt;                  <span class="cm">// 默认 32 字节对齐</span>
tensor_data  : &lt;raw bytes&gt;              <span class="cm">// 权重(常是 L12 的量化块)</span></pre>
<p>逐段看：<strong>magic + version</strong> 是"身份证"，加载器一上来就核对——magic 不是 "GGUF" 直接拒绝，version 不认识就报错。<strong>n_tensors / n_kv</strong> 是两个计数，
告诉加载器"接下来要读多少条"。再往后两大块——metadata 和 tensor infos——是这一课的重点，分别回答"<strong>模型是什么</strong>"和"<strong>每个张量在哪</strong>"。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  还有个让 GGUF 能"到处跑"的底层细节：所有多字节的数都用<strong>固定的小端（little-endian）字节序</strong>写入，字符串则是"长度（u64）+ 内容"的形式存放、不带结尾的 \0。正因为字节序和编码方式是<strong>写死在格式里</strong>的，同一个 .gguf 文件在 x86、ARM、不同操作系统之间拷来拷去，读出来的数都一模一样——可移植性正是从这些不起眼的约定里来的。
</div>
<p>顺便说说名字：<strong>GGUF</strong> 是 "GGML Universal File" 的意思，G-G-M-L 来自作者 Georgi Gerganov 的名字缩写（也是 ggml 库名的由来）。它的前身是更简单的 GGML/GGJT 等格式，
因为不够灵活、扩展时老破坏兼容，才演进成今天这个带版本号、可自由扩展的 GGUF。了解这段渊源，能帮你看懂网上一些老教程里为什么会出现 ".bin"、"ggml-model" 这类旧叫法。</p>
<p>也许你会问：为什么不直接用现成的格式（比如 PyTorch 的 .pt、或 safetensors）？因为 llama.cpp 要的东西很特别——它要把<strong>量化块</strong>（L12 那些 q4_K、q6_K）原样存进去、要能 <strong>mmap 零拷贝</strong>加载、
还要把超参和词表<strong>自带</strong>在文件里，好让纯 C/C++ 端独立读取。这些需求叠加起来，催生了 GGUF 这个为"端侧推理"量身定做的格式。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  注意一个顺序上的讲究：<strong>所有"描述信息"都排在前面，真正的大块数据排在最后</strong>。这样加载器只要读文件开头一小段，就能把模型的全部结构搞清楚，而不必碰那几个 GB 的权重。这个"<strong>头部轻、尾部重</strong>"的布局，正是后面 mmap 零拷贝加载能成立的前提——先用头部信息建好张量清单，再把尾部数据按需映射进来。
</div>

<h2>元数据 KV：模型的"说明书"</h2>
<p>metadata 是一串<strong>键值对</strong>，专门回答"这个模型是什么"。它最重要的特性是<strong>自描述</strong>——加载器不必去别处找配置文件、也不必"猜"模型结构，所有超参、词表、聊天模板<strong>全写在文件里</strong>。
下面是几个真实的键：</p>
<table class="t">
  <tr><th>键 key</th><th>类型 type</th><th>含义</th></tr>
  <tr><td><span class="mono">general.architecture</span></td><td>str</td><td>模型架构，如 "llama"、"qwen2"——决定怎么建图</td></tr>
  <tr><td><span class="mono">llama.block_count</span></td><td>u32</td><td>层数（L04 的 n_layer）</td></tr>
  <tr><td><span class="mono">llama.embedding_length</span></td><td>u32</td><td>隐藏维度（L04 的 n_embd）</td></tr>
  <tr><td><span class="mono">tokenizer.ggml.tokens</span></td><td>array</td><td>词表：所有 token 的字符串</td></tr>
  <tr><td><span class="mono">general.alignment</span></td><td>u32</td><td>对齐字节数（可覆盖默认 32）</td></tr>
</table>
<p>看这些键就明白：L04 说"从 GGUF 头里直接读到 n_layer、n_embd"，读的就是 <span class="mono">llama.block_count</span>、<span class="mono">llama.embedding_length</span> 这两个 KV。
键名还带<strong>命名空间</strong>（<span class="mono">general.</span>、<span class="mono">llama.</span>、<span class="mono">tokenizer.</span>），架构相关的超参用架构名做前缀，于是同一套 GGUF 结构能装下任意模型。</p>
<p>每个值都有一个 <span class="mono">gguf_type</span> 标明类型：u8/i8/u32/i32/f32/bool/string/array 等等（见 <span class="mono">ggml/include/gguf.h</span> 的 <span class="mono">enum gguf_type</span>）。
正因为类型是写在文件里的，读取方不用预先知道"这个键是数还是字符串"，照着 type 解析即可——这就是"自描述"在字节层面的落实。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  一个真实模型的 metadata 往往有<strong>几十上百个 KV</strong>：除了上面几个，还有 RoPE 的频率参数、注意力头数 <span class="mono">llama.attention.head_count</span>、量化版本、训练信息等等。你可以用 <span class="mono">gguf-py</span> 里的脚本或 <span class="mono">llama-gguf</span> 工具把一个 .gguf 的所有 KV 打印出来——亲手翻一遍，会比看十遍讲解更有体感。
</div>
<p>还有一类 KV 专门描述"<strong>这个文件本身</strong>"：<span class="mono">general.name</span>（模型名）、<span class="mono">general.file_type</span>（整体量化档位，对应 L12 的 Q4_K_M 之类）、<span class="mono">general.quantization_version</span> 等。
它们不影响怎么建图，却让工具能一眼报出"这是什么模型、量化到几 bit"——你在加载日志里看到的那些模型信息，多半就是从这些 KV 读出来的。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  特别值一提的是<strong>聊天模板</strong>（chat template）也存在 KV 里。它是一段 Jinja 模板字符串，规定"system / user / assistant 的对话怎么拼成模型输入"。把它放进 GGUF 的好处是：换一个模型，对话格式<strong>自动跟着变</strong>，使用方不用手动去查"这个模型该用什么提示词格式"——又一个"自描述"省心的例子。
</div>
<p>再看 <span class="mono">gguf_type</span> 这套类型本身。它覆盖了从 8 位到 64 位的整数、32/64 位浮点、布尔、字符串，还有一个 <strong>array</strong> 表示"一串同类型的值"——词表 <span class="mono">tokenizer.ggml.tokens</span> 就是个字符串数组。
有了这套类型系统，metadata 几乎能装下任意结构化的配置，而读取方只靠一个 type 标记就知道该怎么解析每个值。</p>
<p>顺带一提，这套结构在 Python 侧由 <span class="mono">gguf-py</span> 读写（L02 的转换脚本就用它）：写入时先攒齐所有 KV 和 tensor info、算好对齐和 offset，再一次性落盘。所以一个 GGUF 文件总是<strong>头部完整、布局规整</strong>的，
不会出现"写了一半结构不全"的情况——这也方便了 mmap 这种"信任头部、直接定位"的读取方式。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  自描述带来一个很实在的好处：<strong>一个 .gguf 文件就是全部</strong>。不像有的格式要外挂一个 config.json、一个 tokenizer.json、一个 generation_config……GGUF 把超参、词表、聊天模板全收进同一个文件。拷走一个文件，模型就能在任何 llama.cpp 上自己读懂自己、跑起来——这正是 L01 说的"一个文件到处跑"的格式基础。
</div>

<h2>tensor info 与对齐：每个张量在哪</h2>
<p>metadata 讲完"模型是什么"，tensor info 回答"<strong>每个张量在哪、长什么样</strong>"。每条 tensor info 记四样东西：<span class="mono">name</span>（张量名，如 <span class="mono">blk.0.attn_q.weight</span>）、
<span class="mono">dims</span>（各维大小）、<span class="mono">type</span>（L05 的 ggml_type，也包括 L12 的量化类型）、以及 <span class="mono">offset</span>（在数据段里的相对偏移）。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>按 offset 定位张量</b>：tensor data 是一整段，每个张量从自己的 offset 处开始</div>
  <div class="cells"><span class="lab">data 段</span><span class="cell hl">张量A @0</span><span class="cell">张量B @offB</span><span class="cell">张量C @offC</span><span class="lab">...</span></div>
</div>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  关键在 <span class="mono">offset</span>：所有张量的原始字节<strong>首尾相连</strong>地排在 tensor data 段里。知道了"数据段起点 + 这个张量的 offset"，就能<strong>直接算出它在文件里的绝对位置</strong>，不用顺序扫描——这正是后面 mmap 能"指哪取哪"的基础。
</div>
<p>为什么 <span class="mono">offset</span> 记的是"相对数据段起点"的偏移，而不是文件里的绝对位置？因为这样更<strong>稳健</strong>：头部（KV、tensor info）的长度会随模型不同而变，要是用绝对偏移，头部一变所有 offset 都得重算；
用相对偏移，数据段内部的排布就和前面头部有多长<strong>解耦</strong>了——加载器把"数据段起点"算出来一次，再加上各张量的相对 offset 即可；这也是为什么往文件里追加张量时，已有张量的 offset 大多不用改动。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  还有个细节值得点出：读取 tensor info 和读取 tensor data 是<strong>分开的两件事</strong>。<span class="mono">gguf_init_from_file</span> 可以只把 name/dims/type/offset 这些"描述"读出来，先<strong>不</strong>碰真正的权重数据（靠一个 no_alloc 之类的选项控制）。正是这种"描述与数据解耦"，让"先建结构、再 mmap 数据"的两步加载成为可能——你能先知道模型多大、有哪些张量，再决定怎么把数据搬进来。
</div>
<p>张量的 <span class="mono">name</span> 不是随便起的，而是一套有规律的命名约定。比如 <span class="mono">blk.0.attn_q.weight</span> 表示"第 0 层（block 0）的注意力 Q 投影权重"，<span class="mono">token_embd.weight</span> 是词嵌入表。
加载器正是<strong>靠这套名字</strong>把文件里的张量一一对应到 L08 建出的模型结构上——名字对不上，权重就装不进对应的位置。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  那个 <span class="mono">dims</span> 也藏着一个 L05 提过的坑：维度按 ggml 的<strong>行优先、ne[0] 最内</strong>来记，和 PyTorch 里的形状顺序常常是反的。所以同一个权重矩阵，转成 GGUF 后 dims 的写法可能和你在 PyTorch 里看到的不一样——这不是出错，而是 L05 那条"维度顺序相反"的约定在文件格式里的延续。
</div>
<p>这里就用上了<strong>对齐</strong>。数据段的起点、以及每个张量的 offset，都会对齐到 <span class="mono">GGUF_DEFAULT_ALIGNMENT = 32</span> 字节（可被 <span class="mono">general.alignment</span> 覆盖）。
为什么要对齐？因为 CPU/GPU 的 SIMD 指令按对齐地址读取最快，mmap 也按内存页管理；让数据落在整齐的边界上，后端读起来更高效、也更省事（详见深挖 1）。</p>

<h2>加载：mmap 零拷贝</h2>
<p>有了清晰的布局，加载就分两步：先读头部、再映射数据。第一步用 <span class="mono">gguf_init_from_file</span> 把 magic、version、所有 KV 和 tensor info 读进来，建好一张"张量清单"；
第二步把整个文件用 <span class="mono">mmap</span> 只读映射进地址空间，每个张量的 <span class="mono">data</span> 指针直接落在映射上的对应位置。</p>
<pre class="code"><span class="cm">// 伪代码: gguf_init_from_file + llama_mmap (src/llama-mmap.cpp)</span>
ctx = <span class="fn">gguf_init_from_file</span>(path)        <span class="cm">// 读 magic/version/KV/tensor infos</span>
<span class="kw">assert</span> magic == "GGUF" <span class="kw">and</span> version == 3
mapping = <span class="fn">mmap</span>(file, PROT_READ)          <span class="cm">// 整文件只读映射, 不拷贝</span>
<span class="kw">for</span> t <span class="kw">in</span> tensors:
    t.data = mapping + data_off + t.offset  <span class="cm">// 张量数据直接指进映射</span></pre>
<div class="flow">
  <div class="node"><div class="nt">.gguf 磁盘文件</div><div class="nd">header | tensor data<br>(几 GB)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">mmap(PROT_READ)</div><div class="nd">只读映射, 不拷贝<br>用到的页才载入</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">张量 data 指针</div><div class="nd">直接指进映射<br>= 文件对应页</div></div>
</div>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  mmap 的妙处在于：它<strong>不把几 GB 权重先读进内存、再拷一遍</strong>，而是把文件"映射"进进程的地址空间——张量 <span class="mono">data</span> 指针看起来像普通内存指针，实际指向的是磁盘文件的对应页。真正读到哪一页，操作系统才把那一页从磁盘载入。于是启动时几乎不花时间在"搬数据"上，这就是大模型能<strong>秒加载</strong>的原因（实现见 <span class="mono">src/llama-mmap.cpp</span> 与 <span class="mono">src/llama-model-loader.cpp</span>）。
</div>
<p>再串一遍整条加载链路，把第三部分前几课都接起来：<span class="mono">gguf_init_from_file</span> 读出超参（metadata）-> 按超参建出 L08 的 ggml_context 和张量结构 -> 张量的 data 指针指进 mmap 映射（权重零拷贝就位）->
之后就是 L09 建图、L10 执行、L11 算子、L12 解量化。<strong>一个 .gguf 文件，就这样变成了一张能跑的计算图。</strong></p>

<p>为什么要"先读头部、再映射数据"分两步，而不是一股脑全读进来？因为这两步的代价天差地别：头部（metadata + tensor info）通常只有几十 KB，<strong>实读进内存</strong>毫无压力；而数据段动辄几 GB，
要是也老老实实读进来就太慢太占内存了，于是改用 mmap 把它<strong>留在磁盘上、按需取页</strong>。这种"小的实读、大的映射"的分工，是大模型加载又快又省的关键。</p>
<p>当然 mmap 也不是没有代价。它依赖操作系统的页缓存，<strong>第一次</strong>真正用到某页时仍要从磁盘读，所以"秒加载"省的是"启动时的整体拷贝"，而不是把磁盘读取变没了。
此外某些场景（如需要把权重整体搬上 GPU 显存）也未必用得上 mmap。但对"CPU 推理、内存就是权重所在地"的常见情形，mmap 几乎是免费的加速。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  最后，把这一课放回第三部分的大图里：L08 讲了张量和内存怎么组织，L09 讲了计算图怎么搭，L10 讲了图怎么调度执行，L11 讲了单个算子怎么算，L12 讲了权重以什么格式压缩存放，而这一课（L13）讲的是<strong>这一切怎么落进磁盘上的一个文件、又怎么被加载回来</strong>。至此，从"一个文件"到"一次推理"之间的每一环，你都看过了一遍。带着这张全景图再去翻 llama.cpp 的源码，你会发现每一块都能对上号、不再陌生。
</div>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么要对齐到 32 字节？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>两个原因。其一是<strong>计算效率</strong>：后端（CPU 的 SIMD、GPU 的 kernel）按对齐地址成批读数最快，地址错位会拖慢、甚至需要额外处理。让每个张量数据从对齐边界开始，后端就能用最高效的加载指令。</p>
    <p>其二是<strong>mmap 友好</strong>：mmap 按内存页映射，数据对齐到规整边界，按页处理更顺、更不容易跨页拖慢。<span class="mono">GGUF_DEFAULT_ALIGNMENT</span> 默认 32 字节，模型也可以用 <span class="mono">general.alignment</span> 这个 KV 覆盖它。</p>
    <p>这其实和 L12 的"字节布局"是同一种思维：<strong>为了让机器读得快，愿意花一点点空间在对齐/填充上</strong>。几个 padding 字节换来整个数据段的高效访问，非常划算。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> GGUF 比老的 GGML 格式好在哪？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>核心是<strong>自描述 + 可扩展</strong>。老格式把少量超参硬编码在文件头，加一个新字段就可能破坏兼容；GGUF 用键值对存元数据，<strong>加一个新超参只是多一个 KV</strong>，老加载器读到不认识的键直接跳过，不会崩。</p>
    <p>而且 GGUF 把超参、词表、聊天模板<strong>统一收进一个文件</strong>，免去了"权重 + 一堆外部配置"的拼凑。version 字段（现在是 3）则明确标记格式演进，让工具能判断兼容性。</p>
    <p>这种"自描述 + 版本化"的设计，是 GGUF 能成为生态通用格式的关键：模型作者、转换工具、推理引擎各自独立演进，只要遵守同一套 KV 约定就能互通。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> mmap 加载，模型算进"内存占用"吗？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>要分清<strong>虚拟内存</strong>和<strong>物理内存</strong>。mmap 会让进程的虚拟地址空间一下子"涨"出几 GB（整个文件的大小），但这只是地址映射，物理内存是<strong>按需、惰性</strong>载入的——用到哪页才占哪页。</p>
    <p>更妙的是这些页是<strong>文件页</strong>：可被操作系统在内存紧张时回收（反正磁盘上有原件），多个进程映射同一个文件时还能<strong>共享同一份物理页</strong>。所以同一台机器起多个实例，权重内存可以共用，省得多。</p>
    <p>这也解释了为什么用 <span class="mono">top</span> 看 llama.cpp 进程，VIRT（虚拟）很大、RES（常驻）却没那么夸张：差的那部分就是"映射了但还没真正读进来、或已被回收"的文件页。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>GGUF 布局：magic <span class="mono">"GGUF"</span> + version(3) + n_tensors/n_kv + <strong>metadata KV</strong> + <strong>tensor infos</strong> + padding + tensor data。</li>
    <li>metadata 是<strong>自描述</strong>的键值对：架构、超参、词表、聊天模板全在文件里，加载器无需猜测（每个值带 <span class="mono">gguf_type</span>）。</li>
    <li>每个 tensor info 记 <span class="mono">name/dims/type/offset</span>；数据段按 <span class="mono">GGUF_DEFAULT_ALIGNMENT=32</span> 对齐。</li>
    <li><span class="mono">gguf_init_from_file</span> 读头部建清单；权重用 <strong>mmap 只读零拷贝</strong>加载，张量 data 直接指进映射。</li>
    <li>自描述 + 可 mmap = "一个文件到处跑" + "秒加载"。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"<strong>模型是什么</strong>"（自描述元数据）和"<strong>模型的数</strong>"（对齐的张量数据）打包进<strong>同一个可 mmap 的文件</strong>——于是 L01 说的"一个文件到处跑"在格式层面真正落地：
  拷走一个 <span class="mono">.gguf</span>，任何 llama.cpp 都能自己读懂它、秒加载它。第三部分到此结束——从内存（L08）、图（L09）、执行（L10）、算子（L11）、量化（L12）到格式（L13），你已经看清了 ggml 引擎的全貌。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Several earlier lessons kept mentioning that <span class="mono">.gguf</span> file - the model lives inside it. But what does it actually look like? This lesson takes a GGUF file <strong>apart end to end</strong>:
the header, the metadata, the tensor list, the alignment padding, the data section - then sees how llama.cpp loads it into memory with <strong>mmap zero-copy</strong> to achieve a large model's "instant load". This is Part 3's
closing lesson, landing everything learned about memory, graphs, operators, and quantization <strong>into a single file on disk</strong>.
</p>
<p style="color:var(--muted);margin-top:.4rem">Why devote a lesson to a file format? Because GGUF is llama.cpp's "<strong>unified entry point</strong>": what you download from HuggingFace and convert with L02's script is this file; what gets read in at runtime is
also it. Read GGUF and you connect the two ends - "the model on disk" and "the tensors in memory".</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  GGUF is like a <strong>shipping container with a manifest</strong>: a list is posted on the door (which tensors are inside, each of what spec, starting at which meter - that is the metadata and tensor info), while the goods (the
  tensor data) are neatly stacked inside. To unload you do not dump the whole container out; you follow the manifest to <strong>address directly</strong> the item you want - exactly the essence of mmap loading: take on demand, no
  needless hauling.
</div>

<h2>Overall layout: a GGUF file end to end</h2>
<p>Start with the panorama. A GGUF file is laid out, front to back, like this: the first 4 bytes are the magic <span class="mono">"GGUF"</span> (instantly recognizing "this is GGUF"), then the version (currently 3), then the tensor
count and KV count, followed by a run of metadata key-value pairs, a run of tensor infos (the tensor list), a stretch of padding for alignment, and only then the <strong>actual tensor data</strong> (very likely those L12 quantized
blocks).</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc"><h4>magic "GGUF" (4 bytes)</h4><p>The first 4 bytes of the file, instantly recognizing "this is a GGUF file".</p></div></div>
  <div class="step"><div class="num">2</div><div class="sc"><h4>version = 3 (u32)</h4><p>The format version, currently 3; backward compatibility hinges on it.</p></div></div>
  <div class="step"><div class="num">3</div><div class="sc"><h4>n_tensors, n_kv (i64 each)</h4><p>A head count first: how many tensors and how many key-value pairs follow.</p></div></div>
  <div class="step"><div class="num">4</div><div class="sc"><h4>metadata KV x n_kv</h4><p>Self-describing info: architecture, hyperparameters, vocab, chat template... the model's "manual".</p></div></div>
  <div class="step"><div class="num">5</div><div class="sc"><h4>tensor infos x n_tensors</h4><p>The tensor list: each tensor's name / dims / type / offset.</p></div></div>
  <div class="step"><div class="num">6</div><div class="sc"><h4>padding (alignment)</h4><p>A few bytes added so the data section starts on a 32-byte boundary.</p></div></div>
  <div class="step"><div class="num">7</div><div class="sc"><h4>tensor data</h4><p>The actual weight bytes, often exactly those quantized blocks from L12.</p></div></div>
</div>
<p>Translating that figure into a "pseudo-struct" gives this (layout per the header comment in <span class="mono">ggml/include/gguf.h</span>):</p>
<pre class="code"><span class="cm">// simplified from the format description in ggml/include/gguf.h</span>
"GGUF"                                  <span class="cm">// 4-byte magic</span>
version      : u32                      <span class="cm">// = 3</span>
n_tensors    : i64
n_kv         : i64
kv_pairs     : [ (key:str, type:gguf_type, value), ... ]              <span class="cm">// self-describing</span>
tensor_infos : [ (name:str, n_dims:u32, dims[]:i64, type, offset:u64), ... ]
&lt;padding to alignment&gt;                  <span class="cm">// 32-byte default alignment</span>
tensor_data  : &lt;raw bytes&gt;              <span class="cm">// weights (often L12 quantized blocks)</span></pre>
<p>Section by section: <strong>magic + version</strong> are the "ID card", checked the moment the loader starts - if the magic is not "GGUF" it refuses outright, and an unknown version errors out. <strong>n_tensors / n_kv</strong>
are two counts telling the loader "how many entries to read next". After them come the two big blocks - metadata and tensor infos - the focus of this lesson, answering "<strong>what the model is</strong>" and "<strong>where each
tensor is</strong>" respectively.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Another low-level detail that lets GGUF "run everywhere": all multi-byte numbers are written in a <strong>fixed little-endian byte order</strong>, and strings are stored as "length (u64) + content" with no trailing \0. Because the byte order and encoding are <strong>fixed by the format</strong>, the same .gguf file copied across x86, ARM, and different operating systems reads back identical numbers - portability comes from these unglamorous conventions.
</div>
<p>A word on the name: <strong>GGUF</strong> stands for "GGML Universal File", and G-G-M-L comes from the initials of the author Georgi Gerganov (also the origin of the ggml library name). Its predecessors were simpler formats like GGML/GGJT,
which were too inflexible and kept breaking compatibility when extended, so they evolved into today's versioned, freely-extensible GGUF. Knowing this lineage helps you understand why some old online tutorials mention ".bin" or "ggml-model" names.</p>
<p>You might ask: why not use an existing format (PyTorch's .pt, or safetensors)? Because llama.cpp needs something special - it must store <strong>quantized blocks</strong> (L12's q4_K, q6_K) verbatim, load them <strong>mmap zero-copy</strong>,
and <strong>carry</strong> hyperparameters and vocab inside the file so a pure C/C++ side can read them independently. These needs stacked together gave rise to GGUF, a format tailored for "on-device inference".</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Note a deliberate ordering: <strong>all the "descriptive info" comes first, and the actual bulk data comes last</strong>. This way the loader reads just a small stretch at the file's start to learn the model's entire structure, without touching those several GB of weights. This "<strong>light head, heavy tail</strong>" layout is the very precondition for the later mmap zero-copy load - build the tensor list from the header first, then map the tail data in on demand.
</div>

<h2>Metadata KV: the model's "manual"</h2>
<p>Metadata is a run of <strong>key-value pairs</strong> that answers "what this model is". Its most important property is being <strong>self-describing</strong> - the loader need not hunt for a config file elsewhere, nor "guess" the
model structure; all hyperparameters, vocab, and chat template are <strong>written right in the file</strong>. Here are a few real keys:</p>
<table class="t">
  <tr><th>key</th><th>type</th><th>meaning</th></tr>
  <tr><td><span class="mono">general.architecture</span></td><td>str</td><td>model architecture, e.g. "llama", "qwen2" - decides how the graph is built</td></tr>
  <tr><td><span class="mono">llama.block_count</span></td><td>u32</td><td>layer count (L04's n_layer)</td></tr>
  <tr><td><span class="mono">llama.embedding_length</span></td><td>u32</td><td>hidden dimension (L04's n_embd)</td></tr>
  <tr><td><span class="mono">tokenizer.ggml.tokens</span></td><td>array</td><td>vocab: the strings of all tokens</td></tr>
  <tr><td><span class="mono">general.alignment</span></td><td>u32</td><td>alignment byte count (can override the default 32)</td></tr>
</table>
<p>These keys make it clear: when L04 said "read n_layer and n_embd straight from the GGUF header", it was reading exactly the <span class="mono">llama.block_count</span> and <span class="mono">llama.embedding_length</span> KVs. Keys also
carry a <strong>namespace</strong> (<span class="mono">general.</span>, <span class="mono">llama.</span>, <span class="mono">tokenizer.</span>), with architecture-specific hyperparameters prefixed by the architecture name, so one GGUF
structure can hold any model.</p>
<p>Every value carries a <span class="mono">gguf_type</span> marking its type: u8/i8/u32/i32/f32/bool/string/array and so on (see <span class="mono">enum gguf_type</span> in <span class="mono">ggml/include/gguf.h</span>). Because the type
is written in the file, the reader need not know in advance "is this key a number or a string"; it just parses per the type - this is "self-describing" realized at the byte level.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  A real model's metadata often has <strong>dozens to hundreds of KVs</strong>: besides the few above, there are RoPE frequency parameters, the attention head count <span class="mono">llama.attention.head_count</span>, quantization version, training info, and more. You can print all KVs of a .gguf with a script in <span class="mono">gguf-py</span> or the <span class="mono">llama-gguf</span> tool - flipping through them yourself gives more intuition than ten readings of an explanation.
</div>
<p>Another class of KV describes "<strong>the file itself</strong>": <span class="mono">general.name</span> (model name), <span class="mono">general.file_type</span> (the overall quantization level, matching L12's Q4_K_M and the like),
<span class="mono">general.quantization_version</span>, and so on. They do not affect how the graph is built, yet they let tools report at a glance "what model this is, quantized to how many bits" - the model info you see in load logs is mostly read
from these KVs.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Worth special mention: the <strong>chat template</strong> is also stored in a KV. It is a Jinja template string specifying "how system / user / assistant turns are assembled into the model input". The benefit of putting it in GGUF: switch models and the chat format <strong>changes automatically</strong>, so the caller need not manually look up "what prompt format this model uses" - another example of self-description saving effort.
</div>
<p>Look at the <span class="mono">gguf_type</span> system itself. It covers integers from 8 to 64 bits, 32/64-bit floats, bool, string, and an <strong>array</strong> meaning "a run of same-typed values" - the vocab <span class="mono">tokenizer.ggml.tokens</span>
is a string array. With this type system, metadata can hold almost any structured config, while the reader needs only a type tag to know how to parse each value.</p>
<p>By the way, this structure is read and written on the Python side by <span class="mono">gguf-py</span> (used by L02's conversion script): on write it first gathers all KVs and tensor infos, computes alignment and offsets, then flushes in one go.
So a GGUF file is always <strong>complete-headered and tidily laid out</strong>, never "half-written with a partial structure" - which also suits mmap's "trust the header, address directly" style of reading.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Self-description brings a very concrete benefit: <strong>one .gguf file is everything</strong>. Unlike formats that bolt on a config.json, a tokenizer.json, a generation_config..., GGUF folds hyperparameters, vocab, and chat template into the same file. Copy one file and the model can read and run itself on any llama.cpp - exactly the format-level basis for L01's "one file runs everywhere".
</div>

<h2>Tensor info and alignment: where each tensor is</h2>
<p>With metadata covering "what the model is", tensor info answers "<strong>where each tensor is and what it looks like</strong>". Each tensor info records four things: <span class="mono">name</span> (the tensor name, e.g.
<span class="mono">blk.0.attn_q.weight</span>), <span class="mono">dims</span> (the size of each dimension), <span class="mono">type</span> (L05's ggml_type, including L12's quantized types), and <span class="mono">offset</span> (the
relative offset within the data section).</p>
<div class="cellgroup">
  <div class="cg-cap"><b>addressing tensors by offset</b>: tensor data is one big section, each tensor starting at its own offset</div>
  <div class="cells"><span class="lab">data section</span><span class="cell hl">tensor A @0</span><span class="cell">tensor B @offB</span><span class="cell">tensor C @offC</span><span class="lab">...</span></div>
</div>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  The key is <span class="mono">offset</span>: all tensors' raw bytes are laid out <strong>end to end</strong> in the tensor data section. Knowing "the data section's start + this tensor's offset" lets you <strong>compute its absolute position in the file directly</strong>, with no sequential scan - exactly the basis for the later mmap to "point and fetch".
</div>
<p>Why does <span class="mono">offset</span> record an offset "relative to the data section start" rather than an absolute file position? Because it is more <strong>robust</strong>: the header (KVs, tensor info) length varies by model, and with absolute
offsets a header change would force recomputing every offset; with relative offsets, the data section's internal layout is <strong>decoupled</strong> from how long the header is - the loader computes the "data section start" once, then adds each tensor's
relative offset. This is also why, when appending tensors to a file, most existing tensors' offsets need no change.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  One more detail worth calling out: reading tensor info and reading tensor data are <strong>two separate things</strong>. <span class="mono">gguf_init_from_file</span> can read just the "descriptions" - name/dims/type/offset - while <strong>not</strong> touching the actual weight data yet (controlled by a no_alloc-style option). This very "description-data decoupling" is what makes the two-step "build structure first, mmap data later" load possible - you can learn how big the model is and which tensors exist before deciding how to bring the data in.
</div>
<p>A tensor's <span class="mono">name</span> is not arbitrary but follows a regular naming convention. For example <span class="mono">blk.0.attn_q.weight</span> means "the attention Q-projection weight of layer 0 (block 0)", and <span class="mono">token_embd.weight</span>
is the token-embedding table. The loader uses exactly these names to match the file's tensors one-to-one onto the model structure L08 builds - if a name does not match, the weight cannot be placed.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  That <span class="mono">dims</span> hides a pitfall L05 raised: dimensions are recorded in ggml's <strong>row-major, ne[0]-innermost</strong> order, often reversed from PyTorch's shape order. So the same weight matrix, converted to GGUF, may have its dims written differently from what you saw in PyTorch - not a bug, but L05's "reversed dimension order" convention carried into the file format.
</div>
<p>This is where <strong>alignment</strong> comes in. The start of the data section, and each tensor's offset, are aligned to <span class="mono">GGUF_DEFAULT_ALIGNMENT = 32</span> bytes (overridable by <span class="mono">general.alignment</span>).
Why align? Because CPU/GPU SIMD instructions read fastest from aligned addresses, and mmap manages memory in pages; letting data fall on tidy boundaries makes backend reads more efficient and simpler (see Dig deeper 1).</p>

<h2>Loading: mmap zero-copy</h2>
<p>With a clear layout, loading is two steps: read the header, then map the data. Step one uses <span class="mono">gguf_init_from_file</span> to read the magic, version, all KVs and tensor infos, building a "tensor list"; step two
maps the whole file read-only into the address space with <span class="mono">mmap</span>, and each tensor's <span class="mono">data</span> pointer lands directly at its place in the mapping.</p>
<pre class="code"><span class="cm">// pseudocode: gguf_init_from_file + llama_mmap (src/llama-mmap.cpp)</span>
ctx = <span class="fn">gguf_init_from_file</span>(path)        <span class="cm">// read magic/version/KV/tensor infos</span>
<span class="kw">assert</span> magic == "GGUF" <span class="kw">and</span> version == 3
mapping = <span class="fn">mmap</span>(file, PROT_READ)          <span class="cm">// whole-file read-only map, no copy</span>
<span class="kw">for</span> t <span class="kw">in</span> tensors:
    t.data = mapping + data_off + t.offset  <span class="cm">// tensor data points straight into the map</span></pre>
<div class="flow">
  <div class="node"><div class="nt">.gguf file on disk</div><div class="nd">header | tensor data<br>(several GB)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">mmap(PROT_READ)</div><div class="nd">read-only map, no copy<br>pages load on demand</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">tensor data pointer</div><div class="nd">points into the map<br>= file's page</div></div>
</div>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  The beauty of mmap is that it <strong>does not read several GB of weights into memory and copy them once more</strong>; it "maps" the file into the process's address space - a tensor's <span class="mono">data</span> pointer looks like an ordinary memory pointer but actually points at the corresponding page of the disk file. Only when a page is actually read does the OS load that page from disk. So startup spends almost no time "moving data" - that is why a large model can <strong>load instantly</strong> (implementation in <span class="mono">src/llama-mmap.cpp</span> and <span class="mono">src/llama-model-loader.cpp</span>).
</div>
<p>Stringing the whole load path once more, tying together Part 3's earlier lessons: <span class="mono">gguf_init_from_file</span> reads the hyperparameters (metadata) -> builds L08's ggml_context and tensor structures per those
hyperparameters -> tensors' data pointers point into the mmap mapping (weights in place, zero-copy) -> then comes L09 graph-building, L10 execution, L11 operators, L12 dequantization. <strong>A single .gguf file thus becomes a runnable
compute graph.</strong></p>

<p>Why two steps - "read the header, then map the data" - instead of reading it all at once? Because the two costs differ enormously: the header (metadata + tensor info) is usually only tens of KB, so <strong>actually reading it into
memory</strong> is trivial; the data section is often several GB, and dutifully reading it in would be far too slow and memory-hungry, so mmap is used to <strong>leave it on disk and fetch pages on demand</strong>. This "read the small, map the
large" division is the key to fast, frugal large-model loading.</p>
<p>Of course mmap is not free. It relies on the OS page cache, and the <strong>first</strong> real touch of a page still reads from disk, so "instant load" saves "the whole copy at startup", not disk reads themselves. Some scenarios (like moving
weights wholesale onto GPU VRAM) may not use mmap either. But for the common case of "CPU inference where memory is where the weights live", mmap is almost-free speedup.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Finally, put this lesson back into Part 3's big picture: L08 covered how tensors and memory are organized, L09 how the compute graph is built, L10 how the graph is scheduled and executed, L11 how a single operator computes, L12 in what format weights are compressed and stored, and this lesson (L13) covers <strong>how all of it lands into a file on disk and is loaded back</strong>. By now you have seen every link between "one file" and "one inference". With this panorama in hand, going back to read llama.cpp's source, you will find every piece falls into place and no longer feels foreign.
</div>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why align to 32 bytes? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Two reasons. First, <strong>compute efficiency</strong>: backends (CPU SIMD, GPU kernels) read in batches fastest from aligned addresses, while misaligned addresses slow things down or need extra handling. Starting each tensor's
    data on an aligned boundary lets the backend use its most efficient load instructions.</p>
    <p>Second, <strong>mmap-friendliness</strong>: mmap maps in memory pages, and data aligned to tidy boundaries is smoother to handle page by page and less prone to cross-page slowdowns. <span class="mono">GGUF_DEFAULT_ALIGNMENT</span>
    defaults to 32 bytes, and a model can override it with the <span class="mono">general.alignment</span> KV.</p>
    <p>This is the same mindset as L12's "byte layout": <strong>to let the machine read fast, spend a little space on alignment/padding</strong>. A few padding bytes buy efficient access to the whole data section - a great bargain.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> What makes GGUF better than the old GGML format? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>The core is <strong>self-describing + extensible</strong>. The old format hard-coded a few hyperparameters in the header, where adding a new field could break compatibility; GGUF stores metadata as key-value pairs, so
    <strong>adding a new hyperparameter is just one more KV</strong>, and an old loader simply skips keys it does not recognize without crashing.</p>
    <p>GGUF also <strong>folds hyperparameters, vocab, and chat template into one file</strong>, sparing you the "weights + a pile of external configs" patchwork. The version field (now 3) explicitly marks format evolution, letting tools
    judge compatibility.</p>
    <p>This "self-describing + versioned" design is the key to GGUF becoming the ecosystem's common format: model authors, conversion tools, and inference engines can each evolve independently, interoperating as long as they honor the
    same KV conventions.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> With mmap loading, does the model count as "memory used"? <span class="hint">Click to expand</span></summary>
  <div class="acc-body">
    <p>Distinguish <strong>virtual memory</strong> from <strong>physical memory</strong>. mmap makes the process's virtual address space suddenly "grow" by several GB (the whole file's size), but that is only an address mapping; physical
    memory is loaded <strong>on demand, lazily</strong> - a page is occupied only when used.</p>
    <p>Better still, these are <strong>file pages</strong>: the OS can reclaim them under memory pressure (the original is on disk anyway), and multiple processes mapping the same file can <strong>share the same physical pages</strong>. So
    running several instances on one machine can share weight memory, saving a lot.</p>
    <p>This also explains why, watching a llama.cpp process with <span class="mono">top</span>, VIRT (virtual) is huge while RES (resident) is not so dramatic: the difference is the file pages "mapped but not yet really read in, or already
    reclaimed".</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>GGUF layout: magic <span class="mono">"GGUF"</span> + version(3) + n_tensors/n_kv + <strong>metadata KV</strong> + <strong>tensor infos</strong> + padding + tensor data.</li>
    <li>metadata is <strong>self-describing</strong> key-value pairs: architecture, hyperparameters, vocab, chat template all in the file, no guessing for the loader (each value carries a <span class="mono">gguf_type</span>).</li>
    <li>Each tensor info records <span class="mono">name/dims/type/offset</span>; the data section is aligned to <span class="mono">GGUF_DEFAULT_ALIGNMENT=32</span>.</li>
    <li><span class="mono">gguf_init_from_file</span> reads the header to build the list; weights load via <strong>mmap read-only zero-copy</strong>, tensor data pointing straight into the mapping.</li>
    <li>Self-describing + mmap-able = "one file runs everywhere" + "instant load".</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Packing "<strong>what the model is</strong>" (self-describing metadata) and "<strong>the model's numbers</strong>" (aligned tensor data) into <strong>one mmap-able file</strong> - so L01's "one file runs everywhere" truly lands at the
  format level: copy one <span class="mono">.gguf</span> and any llama.cpp can read and instantly load it. Part 3 ends here - from memory (L08), graphs (L09), execution (L10), operators (L11), quantization (L12), to the format (L13), you
  have now seen the whole picture of the ggml engine.
</div>
""",
}
