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
<p>这里值得停一下，体会一下这个设计有多统一：无论是矩阵乘、加法、归一化还是注意力，几百个算子函数<strong>清一色都是"建张量 + 填 op/src + 返回"</strong>这个三步模板。
正因为如此一致，ggml 才能用<strong>同一套建图、同一套执行</strong>机制处理所有算子——加一个新算子，主要就是定义一个新的 <span class="mono">op</span> 枚举值、再写它的形状推导和计算实现，
建图这一环完全不用改。这种"<strong>用统一模板装下千变万化</strong>"的克制，是 ggml 代码读起来不乱的重要原因。</p>
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
<p>这也解释了一个新手常踩的坑：在 ggml 里，<strong>建完图就去读结果张量的数据，是读不到东西的</strong>——借条还没兑现呢。
必须先把图交给后端执行（下一课），结果张量的 <span class="mono">data</span> 才会被填上真实数值。把"建图"和"执行"分成两个明确的阶段，是用好 ggml API 的第一课。</p>

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
<p>"<strong>拓扑排序</strong>"这个词听起来唬人，其实道理就是一句大白话：<strong>要用到的东西，必须先准备好</strong>。
做菜时你不能在切菜之前就下锅，算 <span class="mono">y</span> 之前必须先有 <span class="mono">h</span>。拓扑排序就是把所有步骤排成一个合法的先后顺序，
让每一步用到的输入都在它之前已经备齐。一张图可能有<strong>不止一种</strong>合法顺序（比如两个互不依赖的分支谁先谁后都行），但只要满足"依赖在前"，
执行起来结果就一样。ggml 的回溯式建图，自动帮你算出了这样一个合法顺序，你完全不用操心。它内部还用一个"已访问"集合避免把同一个张量重复收进图——
当多个算子<strong>共享同一个输入</strong>时（这在神经网络里太常见了），那个输入只会被收一次、也只会被算一次。</p>
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
<p>It is worth pausing to feel how uniform this design is: whether matmul, add, normalization, or attention, the hundreds of operator functions are <strong>uniformly "build a tensor +
fill op/src + return"</strong>, this same three-step template. Precisely because of this consistency, ggml can handle all operators with <strong>one graph-building and one execution</strong>
mechanism - adding a new operator is mainly defining a new <span class="mono">op</span> enum value plus writing its shape inference and compute implementation; the graph-building part needs
no change. This restraint of "<strong>one uniform template holding endless variety</strong>" is a big reason ggml's code reads cleanly.</p>
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
<p>This also explains a pitfall beginners hit: in ggml, <strong>reading a result tensor's data right after building the graph gets you nothing</strong> - the IOU is not redeemed yet. You
must first hand the graph to the backend to execute (next lesson) before the result tensor's <span class="mono">data</span> is filled with real values. Splitting "build" and "execute" into
two clear phases is the first lesson of using the ggml API well.</p>

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
<p>"<strong>Topological sort</strong>" sounds intimidating but the idea is plain: <strong>what you need must be ready first</strong>. When cooking you cannot hit the pan before chopping;
before computing <span class="mono">y</span> you must have <span class="mono">h</span>. Topological sort arranges all steps into a legal order so each step's inputs are ready before it. A graph
may have <strong>more than one</strong> legal order (two independent branches can go either first), but as long as "dependencies first" holds, the result is the same. ggml's backtracking build
computes such a legal order automatically, no worry for you. It also uses a "visited" set internally to avoid collecting the same tensor twice - when multiple operators <strong>share one
input</strong> (extremely common in networks), that input is collected once and computed once.</p>
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
