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
<p><span class="mono">ggml_init</span> 做的事很直接：先为 <span class="mono">ggml_context</span> 这个管理结构本身分配一点空间，
然后<strong>准备好那块大 arena</strong>——如果你传了 <span class="mono">mem_buffer</span>，它就用你给的内存；传 <span class="mono">NULL</span>，
它就自己 <span class="mono">ggml_aligned_malloc(mem_size)</span> 要一块对齐过的内存（源码见 <span class="mono">ggml/src/ggml.c</span>）。
<span class="mono">ggml_free</span> 则把这块 arena 整体还掉（只有当这块内存是 ggml 自己分配的、即 <span class="mono">mem_buffer_owned</span> 时才释放）。
"<strong>能让你传入 mem_buffer</strong>"这一点很重要：它意味着 ggml 可以在别人给的内存上工作，方便嵌入到各种环境、或复用一块缓冲反复建图。</p>
<p>顺带说说<strong>对齐</strong>。<span class="mono">ggml_aligned_malloc</span> 要的不是普通内存，而是<strong>对齐到特定边界</strong>的内存——因为后端的 SIMD 指令
（AVX、NEON 等，L07 提过）往往要求数据地址对齐才能高效甚至正确地读取。arena 内部每切一个对象，也会按 <span class="mono">GGML_MEM_ALIGN</span> 对齐。
你可以把 arena 理解成一条<strong>带刻度的尺子</strong>，每个对象都落在整齐的刻度上，而不是随手乱放——这点整齐，换来的是计算时的速度。</p>
<p>所以严格说，<span class="mono">ggml_aligned_malloc</span> 与普通 <span class="mono">malloc</span> 的区别就在"<strong>对齐</strong>"二字：普通 malloc 只保证够大、不保证地址落在某个边界上；
而 ggml 要的内存，起始地址必须是某个对齐值（如 16 或 32 字节）的整数倍，这样后端才能放心地用对齐版的 SIMD 加载指令一次搬一大批数。对齐这件小事，体现的是 ggml"<strong>处处为后端计算让路</strong>"的取向。</p>
<p>还有一个常被问到的问题：<strong>一个程序里能开几个 <span class="mono">ggml_context</span>？</strong>答案是<strong>多个</strong>，而且这很常见。
比如可以用一个 ctx 装<strong>模型权重</strong>（活得久，整个推理期间都在）、另一个 ctx 装<strong>每步推理的计算图</strong>（活得短，算完就清）。
不同生命周期的东西放进不同的池子，<strong>该长留的长留、该速清的速清</strong>，互不干扰——这也是 arena 模型带来的便利：一次 <span class="mono">ggml_free</span> 就能精准回收一整批同寿命的对象。</p>
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
<p>为什么这种"只加不减"的游标能行得通？因为<strong>建图阶段几乎只增不删</strong>——你是在一口气把整张计算图搭出来，中途很少需要单独释放某个张量。
既然没有"挖东墙补西墙"的需求，那最简单的分配器（一个往前推的游标）就够用了，连记录空闲块、合并碎片这些复杂逻辑都省了。
这是一种典型的"<strong>用使用场景的特点，换分配器的极致简单</strong>"：等到 L10 真正要<strong>复用</strong>内存时，才会上更聪明的分配器；而这里的建图阶段，朴素的 bump 反而最合适。</p>
<pre class="code"><span class="cm"># 对应 ggml/src/ggml.c 的 ggml_new_object / ggml_new_tensor_impl</span>
<span class="kw">def</span> <span class="fn">new_object</span>(ctx, size):
    cur = ctx.objects_end.offs + ctx.objects_end.size   <span class="cm"># 当前游标</span>
    <span class="kw">if</span> cur + size &gt; ctx.mem_size:                       <span class="cm"># 池子不够了</span>
        abort("arena 空间不足")                          <span class="cm"># 不扩容, 直接报错!</span>
    obj = place_at(ctx.mem_buffer + cur)                <span class="cm"># 就地放下</span>
    link_into(ctx.objects, obj)                         <span class="cm"># 接入链表尾</span>
    <span class="kw">return</span> obj</pre>
<p>这里有两个要点。其一，<strong>张量的元数据（那个 <span class="mono">ggml_tensor</span> 结构）和它的数据缓冲，都从这同一块 arena 里切</strong>——
没有"每个张量单独 <span class="mono">malloc</span> 一次"这回事。其二，<strong>arena 不会自动扩容</strong>：游标一旦撞到边界，ggml 直接 <span class="mono">abort</span>。
所以使用者要<strong>事先把池子估得足够大</strong>。ggml 提供了 <span class="mono">ggml_tensor_overhead()</span> 帮你算"每个张量的元数据要占多少字节"，
建图时常按 <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> 个节点的规模留余量。</p>
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
<p><span class="mono">ggml_init</span> is straightforward: it allocates a little space for the <span class="mono">ggml_context</span> management struct itself, then <strong>prepares
that big arena</strong> - if you passed a <span class="mono">mem_buffer</span> it uses your memory; pass <span class="mono">NULL</span> and it does
<span class="mono">ggml_aligned_malloc(mem_size)</span> itself (see <span class="mono">ggml/src/ggml.c</span>). <span class="mono">ggml_free</span> returns the whole arena
(only freeing the block if ggml allocated it itself, i.e. <span class="mono">mem_buffer_owned</span>). That "<strong>you can pass in mem_buffer</strong>" matters: ggml can work
on memory someone else gave it, handy for embedding into various environments or reusing one buffer to build graphs repeatedly.</p>
<p>A word on <strong>alignment</strong>. <span class="mono">ggml_aligned_malloc</span> wants not just any memory but memory <strong>aligned to a particular boundary</strong> - because
backend SIMD instructions (AVX, NEON, from L07) often require aligned addresses to read efficiently or even correctly. Each object carved inside the arena is also aligned to
<span class="mono">GGML_MEM_ALIGN</span>. Think of the arena as a <strong>ruler with tick marks</strong>: every object lands on a tidy tick rather than wherever - and that bit of
tidiness buys speed at compute time.</p>
<p>So strictly, the difference between <span class="mono">ggml_aligned_malloc</span> and plain <span class="mono">malloc</span> is just
"<strong>alignment</strong>": plain malloc only guarantees big-enough, not that the address falls on a boundary; ggml's memory must start at a multiple of some alignment (16 or
32 bytes), so the backend can confidently use aligned SIMD loads to move a batch at once. This small thing reflects ggml's bias of "<strong>always making way for backend
compute</strong>".</p>
<p>One more often-asked question: <strong>how many <span class="mono">ggml_context</span>s can a program open?</strong> The answer is <strong>several</strong>, and that is common.
For instance, one ctx for the <strong>model weights</strong> (long-lived, present the whole inference) and another for <strong>each step's compute graph</strong> (short-lived,
cleared once computed). Putting things of different lifetimes into different pools lets <strong>the long-lived stay and the short-lived clear fast</strong>, without interfering -
another convenience of the arena model: one <span class="mono">ggml_free</span> precisely reclaims a whole batch of same-lifetime objects.</p>
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
<p>And why does this "only-add, never-remove" cursor work? Because <strong>the build phase is almost append-only</strong> - you are constructing the whole compute graph in one
go, rarely needing to free a single tensor mid-way. With no "rob Peter to pay Paul" need, the simplest allocator (a forward-pushing cursor) suffices, sparing all the complexity
of tracking free blocks and merging fragments. This is a classic "<strong>trade the scenario's traits for the allocator's utter simplicity</strong>": not until L10 actually
needs to <strong>reuse</strong> memory does a smarter allocator come in; here in the build phase, plain bump is the best fit.</p>
<pre class="code"><span class="cm"># cf. ggml_new_object / ggml_new_tensor_impl in ggml/src/ggml.c</span>
<span class="kw">def</span> <span class="fn">new_object</span>(ctx, size):
    cur = ctx.objects_end.offs + ctx.objects_end.size   <span class="cm"># current cursor</span>
    <span class="kw">if</span> cur + size &gt; ctx.mem_size:                       <span class="cm"># pool is out of room</span>
        abort("arena out of space")                     <span class="cm"># no growth, just fail!</span>
    obj = place_at(ctx.mem_buffer + cur)                <span class="cm"># place in situ</span>
    link_into(ctx.objects, obj)                         <span class="cm"># append to the list</span>
    <span class="kw">return</span> obj</pre>
<p>Two points here. One, <strong>a tensor's metadata (the <span class="mono">ggml_tensor</span> struct) and its data buffer are both carved from this same arena</strong> - there
is no "malloc once per tensor". Two, <strong>the arena does not auto-grow</strong>: the moment the cursor hits the edge, ggml <span class="mono">abort</span>s. So the user must
<strong>size the pool large enough up front</strong>. ggml offers <span class="mono">ggml_tensor_overhead()</span> to compute "how many bytes each tensor's metadata takes",
and graphs commonly leave headroom for <span class="mono">GGML_DEFAULT_GRAPH_SIZE = 2048</span> nodes.</p>
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
