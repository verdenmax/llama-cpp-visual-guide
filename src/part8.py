"""Content for Part 8 (practice & contributing)."""

LESSON_38 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
你从 HuggingFace 上下载一个模型，拿到的是一堆 <span class="mono">.safetensors</span> 权重分片，加上一个 <span class="mono">config.json</span>、一个 tokenizer。可 llama.cpp 从头到尾只认一种文件：<span class="mono">.gguf</span>。中间那一步"把 HF 模型转成 GGUF"，就是这一课的主角——它由仓库根目录那个 <span class="mono">convert_hf_to_gguf.py</span> 负责。很多人把它当成一个黑盒脚本：跑一条命令、等一会儿、出一个 <span class="mono">.gguf</span>。这一课要把这个黑盒拆开，看清它到底做了哪三件事。
</p>
<p style="color:var(--muted);margin-top:.4rem">这三件事其实很朴素：(1) <strong>认出这是什么架构</strong>——读 <span class="mono">config.json</span> 里的 <span class="mono">architectures</span>，分发到对应的转换类（Llama 走 <span class="mono">LlamaModel</span>、Qwen 走 <span class="mono">Qwen2Model</span>）；(2) <strong>把每个张量改名、对齐</strong>——HF 把注意力的 Q 投影叫 <span class="mono">model.layers.0.self_attn.q_proj.weight</span>，GGUF 要叫 <span class="mono">blk.0.attn_q.weight</span>，还要把超参（层数、维度、RoPE 设置）写成 GGUF 的元数据；(3) <strong>按 GGUF 的字节格式落盘</strong>——先写文件头，再写一段段元数据键值，再写张量信息表，最后对齐了写真正的权重。看懂这三步，你不仅会"用"这个脚本，还能在它不支持你的模型时，知道该去改哪里。</p>
<p style="color:var(--muted)">有一个结构上的变化值得先说：<span class="mono">convert_hf_to_gguf.py</span> 以前是个几千行的大文件，现在已经被<strong>重构成一个薄薄的命令行入口</strong>（约 300 行），真正干活的代码搬进了一个独立的 <span class="mono">conversion/</span> 包——每个架构一个模块。所以这一课讲的是"一个包怎么协作"，不是"一个大文件里有什么"。路线图：先看一条命令背后的分发流程（配一张追踪图），再看注册表怎么把"架构名"接到"转换类"，然后看张量改名与超参，最后拆开 GGUF 文件本身的字节布局。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  转换的本质，是一次<strong>格式翻译 + 重新打包</strong>。HF 那一堆文件里，信息是<strong>散</strong>的：权重在 <span class="mono">.safetensors</span> 里、超参在 <span class="mono">config.json</span> 里、词表在 tokenizer 文件里，张量的命名还跟着 PyTorch 的习惯走。GGUF 的目标恰恰相反——把这些信息<strong>收拢进一个自描述的单文件</strong>：打开一个 <span class="mono">.gguf</span>，里面既有"这个模型是什么"（架构、层数、维度、RoPE、词表全在元数据里），又有"模型的全部权重"，而且每个张量都用 llama.cpp 自己的规范名（<span class="mono">blk.N.attn_q</span> 这种）命好、按统一对齐排好。转换脚本干的，就是把散落的信息按这套规范重新组织一遍。理解了这一点，你就明白为什么 GGUF 能做到"下载一个文件、不依赖 Python、直接 mmap 就能跑"——所有自描述的功夫，都是在转换这一步一次性付清的（回顾 L13 的 GGUF 格式，这一课正是它的"写入端"）。还有一层好处常被忽略：因为"模型是什么"全写进了元数据，同一个加载器不用改一行代码，就能加载今天还没出现的新架构——只要转换脚本按规范把它的超参写进去，运行时照单全收。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把转换想成<strong>把一份外文书稿排版成一本正式出版物</strong>。原稿（HF 模型）是作者按自己的习惯写的：章节散在不同文件、术语用的是原文的叫法、还附了一堆零散的注释。出版社要做三件事：先<strong>认出这是哪一类书</strong>（架构分发——小说走小说的版式、教材走教材的版式）；再<strong>统一术语、对齐格式</strong>（张量改名——把作者口语化的叫法换成全书统一的规范名）；最后<strong>按出版标准装订成册</strong>（GGUFWriter 写盘——先扉页和版权页、再目录、最后正文，每一页都对齐好页边距）。装订好的这本书（<span class="mono">.gguf</span>）是<strong>自带说明书</strong>的：翻开第一页就知道它是什么、有多少章、怎么读——这正是 llama.cpp 不需要原始 <span class="mono">config.json</span> 就能加载模型的原因。
</div>
<h2>一条命令背后：薄 CLI 把活分发出去</h2>
<p>先跟着一条命令走一遍。你敲下 <span class="mono">python convert_hf_to_gguf.py /path/to/hf-model</span>，这个薄薄的入口脚本做的第一件事不是转换，而是<strong>认门</strong>：它打开模型目录里的 <span class="mono">config.json</span>，读出 <span class="mono">architectures</span> 字段——比如 <span class="mono">"LlamaForCausalLM"</span>。这就是模型的"身份证"。拿到身份后，它去一个注册表里查一句话："谁负责转这种架构？"查到的是一个 Python 类（<span class="mono">LlamaModel</span>），把它实例化，剩下的全交给这个类的 <span class="mono">write()</span> 方法。整个 <span class="mono">main()</span> 短得出乎意料：</p>
<pre class="code"><span class="cm"># 简化自 convert_hf_to_gguf.py 的 main()</span>
hparams = ModelBase.<span class="fn">load_hparams</span>(dir_model)            <span class="cm"># 读 config.json</span>
arch    = <span class="fn">get_model_architecture</span>(hparams, model_type)  <span class="cm"># 取 architectures[0]</span>
model_class = <span class="fn">get_model_class</span>(arch)                   <span class="cm"># 注册表里查到对应子类</span>
model = <span class="fn">model_class</span>(dir_model, output_type, fname_out, ...)
model.<span class="fn">write</span>()                                          <span class="cm"># 真正的转换 + 写盘都在这里</span></pre>
<p>这段代码的关键，是入口<strong>自己几乎什么都不做</strong>——<span class="mono">load_hparams</span> 读配置、<span class="mono">get_model_architecture</span> 取出架构名、<span class="mono">get_model_class</span> 去注册表查类，然后实例化、调 <span class="mono">write()</span>。真正的转换逻辑（怎么读张量、怎么改名、怎么写超参）全在那个查到的类里。这种"入口只管分发、细节交给插件"的结构，正是它能从几千行瘦成 300 行的原因：每种架构的特殊处理，都被搬进了 <span class="mono">conversion/</span> 包里各自的模块，互不打扰。那 <span class="mono">write()</span> 里到底发生了什么？它依次跑 <span class="mono">prepare_tensors</span>（逐张量改名、定量化类型、塞进 writer）和 <span class="mono">prepare_metadata</span>（把超参、词表收成元数据），再调 <span class="mono">GGUFWriter</span> 写文件头、写 KV、写张量数据，最后 <span class="mono">close</span> 收尾——前面那张追踪图的后半截，几乎全压缩在这一个 <span class="mono">write()</span> 调用里了。</p>
<p>把这条分发流水定格成一张图最直观：从一个 HF 目录进去，经过"认架构 -&gt; 查类 -&gt; 实例化 -&gt; 转换写盘"，最后落出一个 <span class="mono">.gguf</span>。看图时抓住一个对比：左边进去的是"按 PyTorch 习惯散落的一堆文件"，右边出来的是"按 GGUF 规范收拢的一个文件"，中间每一站做的都是同一件事——翻译 + 收拢。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次转换的分发</b>：薄 CLI 读出架构名、去注册表查到对应的转换类、实例化后调 write()，由它逐张量改名+量化、再交给 GGUFWriter 落盘（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① HF 目录</h5>
      <div class="cellrow"><span class="vc">.safetensors + config.json</span></div>
      <div class="tlab">下载来的原始模型</div></div>
    <div class="op">load_hparams<br>读 config</div>
    <div class="stn"><h5>② 架构名</h5>
      <div class="cellrow"><span class="vc blue">"LlamaForCausalLM"</span></div>
      <div class="tlab">architectures[0]</div></div>
    <div class="op">get_model_class<br>查注册表</div>
    <div class="stn"><h5>③ 转换类</h5>
      <div class="cellrow"><span class="vc hot">LlamaModel 实例</span></div>
      <div class="tlab">对应这个架构</div></div>
    <div class="op">set_gguf_parameters<br>prepare_tensors</div>
    <div class="stn"><h5>④ 规范张量</h5>
      <div class="cellrow"><span class="vc">blk.N.* + 量化</span></div>
      <div class="tlab">改名 + 定 dtype</div></div>
    <div class="op">GGUFWriter<br>写盘</div>
    <div class="stn"><h5>⑤ 文件</h5>
      <div class="cellrow"><span class="vc blue">model.gguf</span></div>
      <div class="tlab">自描述单文件</div></div>
  </div>
</div>
<h2>注册表：架构名怎么接到转换类</h2>
<p>上一步那句"去注册表里查谁负责"，是整个 conversion 包的枢纽。注册表本身就是一个普通的字典：架构名（字符串）-&gt; 转换类。神奇的地方在于<strong>怎么往里填</strong>——靠一个叫 <span class="mono">register</span> 的类方法当<strong>装饰器</strong>。每个架构模块在定义自己的类时，头顶都挂一行 <span class="mono">@ModelBase.register("LlamaForCausalLM", ...)</span>；Python 加载这个模块时就执行这行，把"这些架构名 -&gt; 这个类"登记进字典。一个类可以认领多个 HF 架构名（Llama / Mistral / Mixtral 共用一套转换逻辑），所以一行 register 往往列着好几个名字：</p>
<pre class="code"><span class="cm"># conversion/base.py: 注册表 + 装饰器工厂</span>
<span class="kw">class</span> <span class="fn">ModelBase</span>:
    _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}   <span class="cm"># 架构名 -&gt; 类</span>

    <span class="kw">@classmethod</span>
    <span class="kw">def</span> <span class="fn">register</span>(cls, *names):          <span class="cm"># 传入若干 HF 架构名</span>
        <span class="kw">def</span> <span class="fn">func</span>(modelcls):
            <span class="kw">for</span> name <span class="kw">in</span> names:
                cls._model_classes[model_type][name] = modelcls   <span class="cm"># 登记</span>
            <span class="kw">return</span> modelcls
        <span class="kw">return</span> func

<span class="cm"># conversion/llama.py: 一个架构 = 一个注册子类</span>
<span class="kw">@ModelBase.register</span>(<span class="st">"LlamaForCausalLM"</span>, <span class="st">"MistralForCausalLM"</span>, <span class="st">"MixtralForCausalLM"</span>)
<span class="kw">class</span> <span class="fn">LlamaModel</span>(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    <span class="kw">def</span> <span class="fn">set_gguf_parameters</span>(self): ...   <span class="cm"># 写层数/维度/RoPE 等</span>
    <span class="kw">def</span> <span class="fn">modify_tensors</span>(self, data, name, bid): ...  <span class="cm"># 改名/permute</span></pre>
<p>读懂这段，你就掌握了"给 llama.cpp 加一个新模型"的入口：<strong>不用改任何主干代码</strong>，只要在 <span class="mono">conversion/</span> 里新建一个模块，写一个 <span class="mono">@ModelBase.register("你的架构名")</span> 的子类，实现 <span class="mono">set_gguf_parameters</span>（写超参）和 <span class="mono">modify_tensors</span>（按需调整张量），它就自动被分发系统认领。官方的 <span class="mono">docs/development/HOWTO-add-model.md</span> 走的正是这条路。这就是"注册表 + 插件"的威力：几十种架构，就是 <span class="mono">conversion/</span> 下几十个互不影响的小文件，谁都能照着已有的一个改出下一个——这也是为什么 llama.cpp 能跟上社区里层出不穷的新模型。</p>
<h2>张量改名与超参：从 PyTorch 叫法到 GGUF 规范名</h2>
<p>分发到 <span class="mono">LlamaModel</span> 之后，真正的转换分两条线并行：一条写<strong>超参</strong>，一条搬<strong>张量</strong>。超参由 <span class="mono">set_gguf_parameters</span> 负责——它把 <span class="mono">config.json</span> 里的层数、隐藏维度、注意力头数、RoPE 设置等，一项项写成 GGUF 的元数据键值（比如 <span class="mono">llama.block_count</span>、<span class="mono">llama.embedding_length</span>）。张量这条线靠 <span class="mono">modify_tensors</span>：它遍历每个权重，调 <span class="mono">map_tensor_name</span> 把 HF 的命名翻成 GGUF 的规范名，必要时再调整张量本身的形状或排布。核心就这么几行：</p>
<pre class="code"><span class="cm"># conversion/base.py: 改名 (基类默认实现)</span>
<span class="kw">def</span> <span class="fn">map_tensor_name</span>(self, name):
    new_name = self.tensor_map.<span class="fn">get_name</span>(key=name, try_suffixes=(<span class="st">".weight"</span>, <span class="st">".bias"</span>))
    <span class="kw">if</span> new_name <span class="kw">is</span> <span class="kw">None</span>:
        <span class="kw">raise</span> ValueError(<span class="st">f"Can not map tensor {name!r}"</span>)
    <span class="kw">return</span> new_name

<span class="kw">def</span> <span class="fn">modify_tensors</span>(self, data_torch, name, bid):
    <span class="kw">return</span> [(self.<span class="fn">map_tensor_name</span>(name), data_torch)]   <span class="cm"># 默认: 只改名, 数据照搬</span></pre>
<p>改名靠的是一张<strong>映射表</strong> <span class="mono">TensorNameMap</span>：它把各家 HF 模型五花八门的命名，统一收敛到 llama.cpp 的规范名上。比如注意力的几个投影，HF 叫 <span class="mono">self_attn.{q,k,v}_proj</span>，GGUF 一律叫 <span class="mono">blk.N.attn_{q,k,v}</span>；名字里的层号按层展开。Llama 还有一个绕不过的坑：它的 Q/K 权重要先 <span class="mono">permute</span> 一下——HF 存 Q/K 的行序，和 llama.cpp 的 RoPE 实现所假设的不一样，不重排的话位置编码会算错。正是这种"每家模型一两个特例"的处理，让每个架构都需要自己一个子类。这也解释了一个常见报错 <span class="mono">"Can not map tensor ..."</span>：它往往不是模型坏了，而是当前架构的 <span class="mono">TensorNameMap</span> 还没收录这个张量名，需要在对应子类里补一条映射、或加一点特例处理。下面这张对照表，就是改名这一步最直观的样子：</p>
<table class="t">
  <tr><th>HF 名（PyTorch 习惯）</th><th>GGUF 规范名</th><th>说明</th></tr>
  <tr><td>model.layers.0.self_attn.q_proj.weight</td><td>blk.0.attn_q.weight</td><td>注意力 Q 投影（需 permute）</td></tr>
  <tr><td>model.layers.0.self_attn.k_proj.weight</td><td>blk.0.attn_k.weight</td><td>注意力 K 投影（需 permute）</td></tr>
  <tr><td>model.layers.0.mlp.gate_proj.weight</td><td>blk.0.ffn_gate.weight</td><td>FFN 门控投影</td></tr>
  <tr><td>model.embed_tokens.weight</td><td>token_embd.weight</td><td>词嵌入表</td></tr>
</table>
<h2>GGUF 文件长什么样：自描述的字节布局</h2>
<p>张量改好名、超参写成 KV 之后，最后一步是把这一切按 GGUF 的字节格式落盘——这活儿交给 <span class="mono">gguf-py</span> 里的 <span class="mono">GGUFWriter</span>。它写盘严格分四段，顺序不能乱：先<strong>文件头</strong>（一眼能看出有多少东西），再<strong>元数据 KV 段</strong>（模型的说明书），再<strong>张量信息表</strong>（每个权重的目录），最后<strong>张量数据段</strong>（真正的权重字节）。把这几段竖着叠起来，就是一个 <span class="mono">.gguf</span> 文件从头到尾的样子：</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">头</span><span class="name">header（24 字节）</span></div><div class="ld">magic "GGUF" + version=3 + tensor_count + kv_count；打开第一眼就知道有多少元数据、多少张量</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">元数据</span><span class="name">KV 段</span></div><div class="ld">一串"键 + 类型标记 + 值"：架构、层数、维度、RoPE、词表、chat 模板……模型的全部"说明书"都在这</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">张量表</span><span class="name">tensor info 段</span></div><div class="ld">每个张量一条：名字 + 维度 + dtype + 在数据区的偏移；相当于一份"目录"</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">对齐</span><span class="name">padding</span></div><div class="ld">补零到 32 字节边界，让后面的数据区起点对齐，便于 mmap 零拷贝</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">数据</span><span class="name">tensor data 段</span></div><div class="ld">所有权重的原始字节，按张量表里的偏移一块块排开；每块各自对齐</div></div>
</div>
<p>文件头是最简单也最关键的一段，就四个定长字段，<span class="mono">write_header_to_file</span> 几行写完——读它的人（llama.cpp 的加载器）也是先读这四个数，才知道后面该读多少：</p>
<pre class="code"><span class="cm"># gguf-py/gguf/gguf_writer.py: 写文件头 (四个字段)</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"&lt;I"</span>, GGUF_MAGIC))    <span class="cm"># 0x46554747 = "GGUF"</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"I"</span>,  GGUF_VERSION))  <span class="cm"># 3</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"Q"</span>,  len(tensors)))  <span class="cm"># 张量数 (u64)</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"Q"</span>,  len(kv_data)))  <span class="cm"># 元数据条数 (u64)</span></pre>
<p>张量信息表里每条记录都带一个 <strong>offset</strong>，指明这个张量的数据从数据区的第几个字节开始。写表时偏移是<strong>累加</strong>出来的：每写完一个张量，下一个的偏移就 <span class="mono">+= ggml_pad(本张量字节数, 32)</span>——也就是说每个张量都向上对齐到 32 字节。正是这一步让 <span class="mono">--outtype</span> 落到实处：你选 <span class="mono">f16</span> 还是 <span class="mono">q8_0</span>，决定的就是每个张量"占多少字节"、用哪个 <span class="mono">GGMLQuantizationType</span> 码写进表里（量化算法本身在 L06/L12 讲过，这里只是把结果按格式记下来）。读到这你应该已经看清：GGUF 没有任何"魔法"，它就是一套把"说明书 + 目录 + 数据"对齐着码进一个文件的朴素约定——而正因为朴素，它才能被任何语言、任何平台稳稳地读出来。再补一个常被忽略的点：元数据 KV 段里每个值都是<strong>自带类型</strong>的——开头一个 <span class="mono">GGUFValueType</span> 标记说明它是 u8、i32、f32、字符串还是数组，读的人据此就知道该取几个字节、怎么解析。正因为值自带类型，GGUF 才能把一个整数 <span class="mono">block_count</span>、一长串 token 字符串数组、几个浮点的 RoPE 参数全塞进同一段里互不混淆——这正是它能"自描述"的底层支撑。</p>
<h2>深入：对齐 与 词表</h2>
<p>两个折叠，补两个转换时绕不开、却容易被脚本"自动处理掉"而看不见的细节。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么所有东西都要对齐到 32 字节？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>还记得 L14 讲过 llama.cpp 用 <span class="mono">mmap</span> 把权重文件直接映射进内存、不做拷贝吗？mmap 要真正发挥威力（尤其是后端做 SIMD / 对齐访问时），数据在文件里的起始地址最好落在整齐的边界上。GGUF 把对齐定为 32 字节（<span class="mono">GGUF_DEFAULT_ALIGNMENT = 32</span>，可被元数据 <span class="mono">general.alignment</span> 覆盖）：每个张量的数据起点都向上取整到 32 的倍数，中间用零填充。代价是文件里多出一点点 padding 字节（每个张量最多浪费 31 字节，对动辄几十 MB 的权重可忽略不计），回报是加载时能整块 mmap、后端能按对齐地址做向量化读取，省掉一次拷贝和潜在的非对齐访问惩罚。这就是为什么上面字节布局那张图里，数据段前面专门留了一截 padding——它不是凑数，是在为运行时的零拷贝映射铺路。一个细节串起两课：L14 讲的是"加载时怎么 mmap 进来"，这一课讲的是"转换时怎么把文件码得能被那样 mmap"——同一个对齐约定的两端，一个写、一个读，严丝合缝。顺带一提，对齐这件事在 ggml 里无处不在——张量内存分配、后端缓冲区、KV cache，背后都有类似的"补齐到某个边界"的考量；GGUF 只是把这套思路也带到了磁盘文件上，让"文件里的布局"和"内存里的布局"天然合拍。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 词表是怎么被写进 GGUF 的？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>模型的权重只是一半，另一半是词表——没有它，模型吐出的 token id 没法变回文字（回顾 L20）。HF 的词表躺在 <span class="mono">tokenizer.json</span> / <span class="mono">tokenizer.model</span> 里，转换时由 <span class="mono">set_vocab</span> 读出来，把每个 token 的字符串、分数（score）、类型（普通 / 控制 / 未知 / 字节）一并写进 GGUF 的元数据。llama.cpp 支持几大类 tokenizer——SentencePiece（Llama 系常用）、BPE（GPT 系）等，<span class="mono">set_vocab</span> 会按模型选对应的读法。除了 token 本身，还有一批特殊 token 也要写：BOS / EOS、padding，以及 chat 模板（还记得 L22 吗？聊天模型的对话格式就存在 GGUF 的 <span class="mono">tokenizer.chat_template</span> 里）。所以一个 <span class="mono">.gguf</span> 是真正"自带电池"的：词表、特殊 token、chat 模板全在里面，加载后不需要再去找任何原始 tokenizer 文件——这也是为什么你只要下载一个 gguf，就能直接跑起一个会聊天的模型，而不必把 HF 仓库里那一堆配套文件也凑齐。给个实感：HF 仓库里那几十个文件（多个 <span class="mono">.safetensors</span> 分片、<span class="mono">config.json</span>、一堆 <span class="mono">tokenizer.*</span>），转换后浓缩成一个 <span class="mono">.gguf</span>；这种"一个文件就能跑"的体验，背后正是 <span class="mono">set_vocab</span> 这类步骤把零散信息一点点收进元数据的功劳。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>分工：<span class="mono">convert_hf_to_gguf.py</span> 现在是薄 CLI；转换机制在 <span class="mono">conversion/</span> 包，写盘在 <span class="mono">gguf-py</span>。</li>
    <li>分发：读 <span class="mono">config.json</span> 的 <span class="mono">architectures</span> -&gt; <span class="mono">get_model_class</span> -&gt; 注册表 <span class="mono">@ModelBase.register("LlamaForCausalLM", ...)</span> 找到对应子类。</li>
    <li>改名：<span class="mono">map_tensor_name</span> 经 <span class="mono">TensorNameMap</span> 把 HF 名翻成 GGUF 规范名（<span class="mono">model.layers.0.self_attn.q_proj</span> -&gt; <span class="mono">blk.0.attn_q</span>）；Llama 还要 <span class="mono">permute</span> Q/K。</li>
    <li>超参：<span class="mono">set_gguf_parameters</span> 把层数/维度/RoPE 等写成 GGUF 元数据键值。</li>
    <li>落盘：<span class="mono">GGUFWriter</span> 四段——头（magic/version/tensor_count/kv_count）-&gt; 元数据 KV -&gt; 张量信息表 -&gt; 对齐 32B -&gt; 权重数据；<span class="mono">--outtype</span> 决定每个张量存 f16/q8_0/...。</li>
    <li>扩展：新增一个架构 = 在 <span class="mono">conversion/</span> 加一个注册子类，实现 <span class="mono">set_gguf_parameters</span>/<span class="mono">modify_tensors</span>（见 HOWTO-add-model）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  这一课藏着两个反复出现的工程智慧。第一个是<strong>自描述格式</strong>：GGUF 把"怎么读我"写进了文件本身——元数据段说清架构、维度、词表，张量信息表说清每个权重的形状、类型、偏移。代价是转换时要多写一堆元数据，回报是运行时<strong>零外部依赖</strong>、能 mmap、跨语言都能解析。第二个是<strong>注册表 + 插件式扩展</strong>：<span class="mono">@ModelBase.register</span> 让"支持一个新架构"变成"加一个文件、不动主干"——几十种模型就是几十个互不干扰的小模块。这两招你在别处会一再遇到：自描述格式（想想 ELF、PNG、tar）、注册表分发（想想各种框架的 plugin 机制）。把"转换模型"看成"格式翻译 + 插件分发"，你就能在任何一个还没支持的模型面前，知道自己该往哪儿下手——这正是从"会用"迈向"能改、能贡献"的那一步（下一课就讲怎么把这一步真正提成一个 PR）。说到底，这一课是从"读者"到"作者"的转身：前面三十多课你一直在读 llama.cpp 怎么跑，从这里起，你开始有能力往里写——而写，正是贡献的起点，也是这门课最后想带你抵达的地方。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
You download a model from HuggingFace and get a pile of <span class="mono">.safetensors</span> weight shards, plus a <span class="mono">config.json</span> and a tokenizer. But llama.cpp speaks exactly one file format end to end: <span class="mono">.gguf</span>. That middle step - "convert an HF model to GGUF" - is the star of this lesson, handled by <span class="mono">convert_hf_to_gguf.py</span> at the repo root. Many treat it as a black box: run one command, wait a bit, out comes a <span class="mono">.gguf</span>. This lesson pries the box open to see the three things it actually does.
</p>
<p style="color:var(--muted);margin-top:.4rem">Those three things are plain: (1) <strong>recognize the architecture</strong> - read <span class="mono">architectures</span> from <span class="mono">config.json</span> and dispatch to the matching converter class (Llama goes to <span class="mono">LlamaModel</span>, Qwen to <span class="mono">Qwen2Model</span>); (2) <strong>rename and align every tensor</strong> - HF calls the attention Q projection <span class="mono">model.layers.0.self_attn.q_proj.weight</span>, GGUF wants <span class="mono">blk.0.attn_q.weight</span>, and the hyper-parameters (layer count, dims, RoPE settings) must be written as GGUF metadata; (3) <strong>serialize in GGUF's byte format</strong> - write the header, then the metadata key-values, then the tensor-info table, then the actual weights after alignment. Understand these three steps and you can not just "use" the script but, when it does not support your model, know where to go fix it.</p>
<p style="color:var(--muted)">One structural change is worth flagging first: <span class="mono">convert_hf_to_gguf.py</span> used to be a multi-thousand-line monolith; it has been <strong>refactored into a thin command-line entry</strong> (about 300 lines), and the real work moved into a standalone <span class="mono">conversion/</span> package - one module per architecture. So this lesson is about "how a package collaborates", not "what is inside one big file". Roadmap: first the dispatch flow behind one command (with a trace), then how the registry wires an "architecture name" to a "converter class", then tensor renaming and hyper-parameters, and finally the byte layout of the GGUF file itself.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Conversion is at heart a <strong>format translation plus repack</strong>. In the HF files the information is <strong>scattered</strong>: weights live in <span class="mono">.safetensors</span>, hyper-parameters in <span class="mono">config.json</span>, the vocab in tokenizer files, and tensor names follow PyTorch habits. GGUF aims for the opposite - gather it all into a <strong>self-describing single file</strong>: open one <span class="mono">.gguf</span> and it holds both "what this model is" (architecture, layer count, dims, RoPE, vocab, all in metadata) and "all of the model's weights", with each tensor named by llama.cpp's own canonical scheme (<span class="mono">blk.N.attn_q</span> and friends) and laid out at a uniform alignment. What the converter does is reorganize the scattered information to this spec. Grasp that and you see why GGUF can "download one file, no Python needed, just mmap and run" - all the self-describing effort is paid once, here at conversion time (recall L13 on the GGUF format; this lesson is its "write side"). One more benefit is easy to miss: because "what the model is" lives entirely in the metadata, the same loader can - without changing a line - load architectures that do not exist yet, as long as the converter wrote their hyper-params to spec; the runtime just takes them as given.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of conversion as <strong>typesetting a foreign manuscript into a formal publication</strong>. The manuscript (the HF model) was written to the author's own habits: chapters scattered across files, terminology in the original wording, a heap of loose notes attached. The publisher does three things: first <strong>recognize what kind of book this is</strong> (architecture dispatch - a novel gets a novel's layout, a textbook a textbook's); then <strong>unify the terminology and align the format</strong> (tensor renaming - swap the author's casual names for the book's one canonical naming); finally <strong>bind it to the publishing standard</strong> (GGUFWriter serialization - title and copyright page first, then the table of contents, then the body, every page aligned to the margins). The bound book (<span class="mono">.gguf</span>) <strong>carries its own manual</strong>: open the first page and you know what it is, how many chapters, how to read it - exactly why llama.cpp can load the model without the original <span class="mono">config.json</span>.
</div>
<h2>Behind one command: the thin CLI dispatches the work</h2>
<p>Follow one command through. You type <span class="mono">python convert_hf_to_gguf.py /path/to/hf-model</span>, and the first thing this thin entry script does is not convert but <strong>identify</strong>: it opens <span class="mono">config.json</span> in the model directory and reads the <span class="mono">architectures</span> field - say <span class="mono">"LlamaForCausalLM"</span>. That is the model's ID card. With the identity in hand it asks a registry one question: "who converts this architecture?" The answer is a Python class (<span class="mono">LlamaModel</span>); it instantiates that class and hands everything else to its <span class="mono">write()</span> method. The whole <span class="mono">main()</span> is surprisingly short:</p>
<pre class="code"><span class="cm"># simplified from main() in convert_hf_to_gguf.py</span>
hparams = ModelBase.<span class="fn">load_hparams</span>(dir_model)            <span class="cm"># read config.json</span>
arch    = <span class="fn">get_model_architecture</span>(hparams, model_type)  <span class="cm"># take architectures[0]</span>
model_class = <span class="fn">get_model_class</span>(arch)                   <span class="cm"># look up the subclass in the registry</span>
model = <span class="fn">model_class</span>(dir_model, output_type, fname_out, ...)
model.<span class="fn">write</span>()                                          <span class="cm"># the real conversion + serialization live here</span></pre>
<p>The key to this code is that the entry <strong>does almost nothing itself</strong> - <span class="mono">load_hparams</span> reads the config, <span class="mono">get_model_architecture</span> pulls out the architecture name, <span class="mono">get_model_class</span> looks up the class in the registry, then instantiate and call <span class="mono">write()</span>. The real conversion logic (how to read tensors, how to rename, how to write hyper-params) all lives in the looked-up class. This "the entry only dispatches, the details go to plugins" structure is exactly why it could slim from thousands of lines to 300: each architecture's special handling moved into its own module under the <span class="mono">conversion/</span> package, out of each other's way. So what happens inside <span class="mono">write()</span>? It runs <span class="mono">prepare_tensors</span> (rename each tensor, pick its quant type, hand it to the writer) then <span class="mono">prepare_metadata</span> (gather hyper-params and vocab into metadata), then calls <span class="mono">GGUFWriter</span> to write the header, the KV, and the tensor data, and finally <span class="mono">close</span> - the second half of that trace above is almost entirely compressed into this one <span class="mono">write()</span> call.</p>
<p>The dispatch flow is clearest frozen into one diagram: in goes an HF directory, through "identify arch -&gt; look up class -&gt; instantiate -&gt; convert and serialize", and out comes a <span class="mono">.gguf</span>. As you read the diagram, hold one contrast: what goes in on the left is "a pile of files scattered to PyTorch habits", what comes out on the right is "one file gathered to the GGUF spec", and every station in between does the same one thing - translate plus gather.</p>
<div class="trace">
  <div class="tcap"><b>Tracing the dispatch of one conversion</b>: the thin CLI reads the architecture name, looks up the matching converter class in the registry, instantiates it and calls write(), which renames+quantizes tensor by tensor and hands them to GGUFWriter to serialize (schematic).</div>
  <div class="stations">
    <div class="stn"><h5>(1) HF dir</h5>
      <div class="cellrow"><span class="vc">.safetensors + config.json</span></div>
      <div class="tlab">the downloaded model</div></div>
    <div class="op">load_hparams<br>read config</div>
    <div class="stn"><h5>(2) arch name</h5>
      <div class="cellrow"><span class="vc blue">"LlamaForCausalLM"</span></div>
      <div class="tlab">architectures[0]</div></div>
    <div class="op">get_model_class<br>look up registry</div>
    <div class="stn"><h5>(3) converter</h5>
      <div class="cellrow"><span class="vc hot">LlamaModel instance</span></div>
      <div class="tlab">matches this arch</div></div>
    <div class="op">set_gguf_parameters<br>prepare_tensors</div>
    <div class="stn"><h5>(4) canonical tensors</h5>
      <div class="cellrow"><span class="vc">blk.N.* + quantized</span></div>
      <div class="tlab">renamed + dtype set</div></div>
    <div class="op">GGUFWriter<br>serialize</div>
    <div class="stn"><h5>(5) file</h5>
      <div class="cellrow"><span class="vc blue">model.gguf</span></div>
      <div class="tlab">self-describing file</div></div>
  </div>
</div>
<h2>The registry: how an architecture name reaches a converter class</h2>
<p>That line "look up the registry for who is responsible" is the hinge of the whole conversion package. The registry itself is just an ordinary dictionary: architecture name (a string) -&gt; converter class. The clever part is <strong>how it gets filled</strong> - through a classmethod called <span class="mono">register</span> used as a <strong>decorator</strong>. Each architecture module, when it defines its class, hangs one line above it: <span class="mono">@ModelBase.register("LlamaForCausalLM", ...)</span>. Python runs that line when it loads the module, registering "these architecture names -&gt; this class" into the dictionary. One class can claim several HF architecture names (Llama / Mistral / Mixtral share one conversion path), so a single register line often lists several names:</p>
<pre class="code"><span class="cm"># conversion/base.py: the registry + decorator factory</span>
<span class="kw">class</span> <span class="fn">ModelBase</span>:
    _model_classes = {ModelType.TEXT: {}, ModelType.MMPROJ: {}}   <span class="cm"># arch name -&gt; class</span>

    <span class="kw">@classmethod</span>
    <span class="kw">def</span> <span class="fn">register</span>(cls, *names):          <span class="cm"># takes some HF architecture names</span>
        <span class="kw">def</span> <span class="fn">func</span>(modelcls):
            <span class="kw">for</span> name <span class="kw">in</span> names:
                cls._model_classes[model_type][name] = modelcls   <span class="cm"># register it</span>
            <span class="kw">return</span> modelcls
        <span class="kw">return</span> func

<span class="cm"># conversion/llama.py: one architecture = one registered subclass</span>
<span class="kw">@ModelBase.register</span>(<span class="st">"LlamaForCausalLM"</span>, <span class="st">"MistralForCausalLM"</span>, <span class="st">"MixtralForCausalLM"</span>)
<span class="kw">class</span> <span class="fn">LlamaModel</span>(TextModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    <span class="kw">def</span> <span class="fn">set_gguf_parameters</span>(self): ...   <span class="cm"># write layer count / dims / RoPE</span>
    <span class="kw">def</span> <span class="fn">modify_tensors</span>(self, data, name, bid): ...  <span class="cm"># rename / permute</span></pre>
<p>Read this and you hold the entry point for "adding a new model to llama.cpp": <strong>without touching any trunk code</strong>, you just create a module under <span class="mono">conversion/</span>, write a subclass decorated with <span class="mono">@ModelBase.register("YourArchName")</span>, implement <span class="mono">set_gguf_parameters</span> (write hyper-params) and <span class="mono">modify_tensors</span> (adjust tensors as needed), and it is automatically claimed by the dispatch system. The official <span class="mono">docs/development/HOWTO-add-model.md</span> walks exactly this path. That is the power of "registry plus plugins": dozens of architectures are dozens of independent little files under <span class="mono">conversion/</span>, and anyone can copy an existing one into the next - which is how llama.cpp keeps up with the steady stream of new community models.</p>
<h2>Tensor renaming and hyper-parameters: from PyTorch names to GGUF canonical names</h2>
<p>After dispatch to <span class="mono">LlamaModel</span>, the real conversion runs two lines in parallel: one writes <strong>hyper-parameters</strong>, the other moves <strong>tensors</strong>. Hyper-params are handled by <span class="mono">set_gguf_parameters</span> - it writes the layer count, hidden dim, attention head count, RoPE settings and so on from <span class="mono">config.json</span> as GGUF metadata key-values (such as <span class="mono">llama.block_count</span>, <span class="mono">llama.embedding_length</span>). The tensor line runs through <span class="mono">modify_tensors</span>: it walks each weight, calls <span class="mono">map_tensor_name</span> to translate the HF name into the GGUF canonical name, and reshapes or re-lays-out the tensor itself when needed. The core is just a few lines:</p>
<pre class="code"><span class="cm"># conversion/base.py: renaming (base-class default)</span>
<span class="kw">def</span> <span class="fn">map_tensor_name</span>(self, name):
    new_name = self.tensor_map.<span class="fn">get_name</span>(key=name, try_suffixes=(<span class="st">".weight"</span>, <span class="st">".bias"</span>))
    <span class="kw">if</span> new_name <span class="kw">is</span> <span class="kw">None</span>:
        <span class="kw">raise</span> ValueError(<span class="st">f"Can not map tensor {name!r}"</span>)
    <span class="kw">return</span> new_name

<span class="kw">def</span> <span class="fn">modify_tensors</span>(self, data_torch, name, bid):
    <span class="kw">return</span> [(self.<span class="fn">map_tensor_name</span>(name), data_torch)]   <span class="cm"># default: rename only, data as-is</span></pre>
<p>Renaming relies on a <strong>mapping table</strong>, <span class="mono">TensorNameMap</span>: it converges the wildly varied naming across HF models onto llama.cpp's canonical names. For example the attention projections that HF calls <span class="mono">self_attn.{q,k,v}_proj</span> all become <span class="mono">blk.N.attn_{q,k,v}</span> in GGUF, with the layer number expanded per layer. Llama also has one unavoidable wrinkle: its Q/K weights must first be <span class="mono">permute</span>d - the row order HF stores Q/K in differs from what llama.cpp's RoPE implementation assumes, and without the re-permute the positional encoding computes wrong. It is exactly this "one or two special cases per model" handling that makes each architecture need its own subclass. It also explains a common error <span class="mono">"Can not map tensor ..."</span>: usually it is not that the model is broken, but that this architecture's <span class="mono">TensorNameMap</span> has not yet recorded this tensor name, and you need to add a mapping or a small special case in the matching subclass. The table below is the most direct picture of this renaming step:</p>
<table class="t">
  <tr><th>HF name (PyTorch habit)</th><th>GGUF canonical name</th><th>note</th></tr>
  <tr><td>model.layers.0.self_attn.q_proj.weight</td><td>blk.0.attn_q.weight</td><td>attention Q projection (needs permute)</td></tr>
  <tr><td>model.layers.0.self_attn.k_proj.weight</td><td>blk.0.attn_k.weight</td><td>attention K projection (needs permute)</td></tr>
  <tr><td>model.layers.0.mlp.gate_proj.weight</td><td>blk.0.ffn_gate.weight</td><td>FFN gate projection</td></tr>
  <tr><td>model.embed_tokens.weight</td><td>token_embd.weight</td><td>token embedding table</td></tr>
</table>
<h2>What a GGUF file looks like: the self-describing byte layout</h2>
<p>Once the tensors are renamed and the hyper-params written as KV, the last step is to serialize all of it in GGUF's byte format - the job of <span class="mono">GGUFWriter</span> in <span class="mono">gguf-py</span>. It writes in four sections, strictly in order: first the <strong>header</strong> (so you can see at a glance how much is inside), then the <strong>metadata KV section</strong> (the model's manual), then the <strong>tensor-info table</strong> (a directory of every weight), and finally the <strong>tensor-data section</strong> (the actual weight bytes). Stack those sections vertically and you have a <span class="mono">.gguf</span> file end to end:</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">header</span><span class="name">header (24 bytes)</span></div><div class="ld">magic "GGUF" + version=3 + tensor_count + kv_count; the first glance tells you how much metadata and how many tensors</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">metadata</span><span class="name">KV section</span></div><div class="ld">a run of "key + type tag + value": architecture, layer count, dims, RoPE, vocab, chat template... the model's whole "manual"</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">tensor table</span><span class="name">tensor info section</span></div><div class="ld">one record per tensor: name + dims + dtype + offset into the data area; effectively a "table of contents"</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">align</span><span class="name">padding</span></div><div class="ld">zero-pad to a 32-byte boundary so the data area starts aligned, for zero-copy mmap</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">data</span><span class="name">tensor data section</span></div><div class="ld">the raw bytes of every weight, laid out block by block per the table's offsets; each block aligned too</div></div>
</div>
<p>The header is the simplest yet most crucial section - just four fixed-length fields, written in a few lines by <span class="mono">write_header_to_file</span>. Its reader (llama.cpp's loader) likewise reads these four numbers first to know how much follows:</p>
<pre class="code"><span class="cm"># gguf-py/gguf/gguf_writer.py: write the header (four fields)</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"&lt;I"</span>, GGUF_MAGIC))    <span class="cm"># 0x46554747 = "GGUF"</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"I"</span>,  GGUF_VERSION))  <span class="cm"># 3</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"Q"</span>,  len(tensors)))  <span class="cm"># tensor count (u64)</span>
fout.write(self.<span class="fn">_pack</span>(<span class="st">"Q"</span>,  len(kv_data)))  <span class="cm"># metadata entry count (u64)</span></pre>
<p>Each record in the tensor-info table carries an <strong>offset</strong> stating which byte of the data area this tensor's data starts at. The offset is <strong>accumulated</strong> as the table is written: after each tensor, the next offset goes <span class="mono">+= ggml_pad(this tensor's byte size, 32)</span> - that is, each tensor is rounded up to a 32-byte boundary. This is exactly where <span class="mono">--outtype</span> lands: choosing <span class="mono">f16</span> versus <span class="mono">q8_0</span> decides how many bytes each tensor occupies and which <span class="mono">GGMLQuantizationType</span> code is written into the table (the quantization algorithm itself was covered in L06/L12; here we only record the result per the format). By now you can see GGUF has no "magic" at all: it is a plain convention for laying "manual + directory + data" into one aligned file - and precisely because it is plain, it can be read back reliably from any language on any platform. One more easily-missed point: every value in the metadata KV section is <strong>self-typed</strong> - a leading <span class="mono">GGUFValueType</span> tag says whether it is a u8, i32, f32, string, or array, so the reader knows how many bytes to take and how to parse them. Because values carry their own type, GGUF can pack an integer <span class="mono">block_count</span>, a long array of token strings, and a few floating-point RoPE parameters all into the same section without confusion - exactly the underpinning that lets it be "self-describing".</p>
<h2>Deeper: alignment and the vocabulary</h2>
<p>Two accordions, filling in two details that conversion cannot avoid yet the script "auto-handles" so quietly you never see them.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why must everything align to 32 bytes? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Recall L14, where llama.cpp uses <span class="mono">mmap</span> to map the weight file straight into memory with no copy. For mmap to truly pay off (especially when the backend does SIMD / aligned access), the data's start address in the file should land on a tidy boundary. GGUF sets alignment to 32 bytes (<span class="mono">GGUF_DEFAULT_ALIGNMENT = 32</span>, overridable via the <span class="mono">general.alignment</span> metadata): each tensor's data start is rounded up to a multiple of 32, with zero padding in between. The cost is a few extra padding bytes in the file (at most 31 wasted per tensor, negligible against weights that run to tens of MB); the payoff is that loading can mmap whole blocks and the backend can do vectorized reads at aligned addresses, saving a copy and a potential misaligned-access penalty. That is why the byte-layout diagram above leaves a stretch of padding before the data section - it is not filler, it paves the way for zero-copy mapping at runtime. One detail ties two lessons together: L14 was about "how to mmap it in at load time", this lesson is about "how to lay the file out at conversion time so it can be mmap'd that way" - the two ends of one alignment convention, one writing, one reading, fitting exactly. Incidentally, alignment is everywhere in ggml - tensor memory allocation, backend buffers, the KV cache all carry a similar "round up to some boundary" consideration; GGUF merely carries that idea onto the on-disk file too, so that "the layout in the file" and "the layout in memory" naturally agree.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> How does the vocabulary get written into GGUF? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The weights are only half; the other half is the vocabulary - without it the token ids the model emits cannot turn back into text (recall L20). HF's vocab lives in <span class="mono">tokenizer.json</span> / <span class="mono">tokenizer.model</span>; at conversion time <span class="mono">set_vocab</span> reads it out and writes each token's string, score, and type (normal / control / unknown / byte) into the GGUF metadata. llama.cpp supports a few tokenizer families - SentencePiece (common for the Llama line), BPE (the GPT line), and others - and <span class="mono">set_vocab</span> picks the matching reader per model. Beyond the tokens themselves, a batch of special tokens must be written too: BOS / EOS, padding, and the chat template (remember L22? a chat model's conversation format is stored in GGUF's <span class="mono">tokenizer.chat_template</span>). So a <span class="mono">.gguf</span> is genuinely "batteries included": vocab, special tokens, chat template all inside, and after loading you need not hunt down any original tokenizer file - which is why downloading one gguf is enough to run a chatting model, without rounding up the pile of companion files from the HF repo. To make it concrete: the dozens of files in an HF repo (several <span class="mono">.safetensors</span> shards, <span class="mono">config.json</span>, a heap of <span class="mono">tokenizer.*</span>) condense after conversion into one <span class="mono">.gguf</span>; that "one file and it runs" experience is precisely the work of steps like <span class="mono">set_vocab</span> gathering scattered information bit by bit into the metadata.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Split: <span class="mono">convert_hf_to_gguf.py</span> is now a thin CLI; the conversion machinery is in the <span class="mono">conversion/</span> package, serialization in <span class="mono">gguf-py</span>.</li>
    <li>Dispatch: read <span class="mono">architectures</span> from <span class="mono">config.json</span> -&gt; <span class="mono">get_model_class</span> -&gt; registry <span class="mono">@ModelBase.register("LlamaForCausalLM", ...)</span> finds the subclass.</li>
    <li>Rename: <span class="mono">map_tensor_name</span> via <span class="mono">TensorNameMap</span> turns HF names into GGUF canonical names (<span class="mono">model.layers.0.self_attn.q_proj</span> -&gt; <span class="mono">blk.0.attn_q</span>); Llama also has to <span class="mono">permute</span> Q/K.</li>
    <li>Hyper-params: <span class="mono">set_gguf_parameters</span> writes layer count / dims / RoPE etc. as GGUF metadata key-values.</li>
    <li>Serialize: <span class="mono">GGUFWriter</span> in four sections - header (magic/version/tensor_count/kv_count) -&gt; metadata KV -&gt; tensor-info table -&gt; align to 32B -&gt; weight data; <span class="mono">--outtype</span> decides whether each tensor is stored f16/q8_0/...</li>
    <li>Extend: adding an architecture = add a registered subclass in <span class="mono">conversion/</span>, implement <span class="mono">set_gguf_parameters</span>/<span class="mono">modify_tensors</span> (see HOWTO-add-model).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Two recurring pieces of engineering wisdom hide in this lesson. The first is the <strong>self-describing format</strong>: GGUF writes "how to read me" into the file itself - the metadata section states architecture, dims, vocab; the tensor-info table states each weight's shape, type, offset. The cost is writing a pile of metadata at conversion time; the payoff is <strong>zero external dependencies</strong> at runtime, mmap-ability, and parseability from any language. The second is <strong>registry plus plugin-style extension</strong>: <span class="mono">@ModelBase.register</span> turns "support a new architecture" into "add a file, do not touch the trunk" - dozens of models become dozens of independent little modules. You will meet both moves elsewhere: self-describing formats (think ELF, PNG, tar), registry dispatch (think the plugin mechanism of countless frameworks). See "converting a model" as "format translation plus plugin dispatch" and, faced with any not-yet-supported model, you know where to start - the very step from "can use it" to "can change it, can contribute" (the next lesson is about turning that step into a real PR). At bottom, this lesson is a turn from "reader" to "author": for thirty-some lessons you have been reading how llama.cpp runs; from here on you start to be able to write into it - and writing is where contributing begins - and the place this course has been quietly leading you toward.
</div>
""",
}

LESSON_39 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
读懂了源码，下一步自然是动手：改一行、修个 bug、加个特性，然后把它提成一个能被合并的 PR。可"动手"这件事本身也有一套流程——怎么把这个上百万行的 C++ 大项目编译出来、改完怎么确认没把别的地方弄坏、提 PR 又要守哪些规矩。这一课不讲新的内部原理，而是带你把这条真实的开发回路完整走一遍，让前面三十多课学到的"读"，真正落到"能改、能贡献"上。
</p>
<p style="color:var(--muted);margin-top:.4rem">回路其实就三段：<strong>编译</strong>（用 CMake 一套命令把源码变成 <span class="mono">build/bin/</span> 里的二进制，换个开关就切到 CUDA / Metal / Vulkan 等后端）、<strong>测试</strong>（用 <span class="mono">ctest</span> 跑自动化用例，其中 <span class="mono">test-backend-ops</span> 专门校验各后端算子结果一致）、<strong>贡献</strong>（按 <span class="mono">CONTRIBUTING.md</span> 提 PR：一个 PR 一个功能、CPU 支持优先、用 <span class="mono">clang-format</span> 对齐风格）。把这三段串起来，你就有了一条从"本地改代码"到"被上游接纳"的完整路径。这条路你走通一次，以后面对任何一个陌生的大型 C++ 开源项目，套路其实都八九不离十。</p>
<p style="color:var(--muted)">有一件事必须先讲明白，尤其因为你可能正借助 AI 读这门课：llama.cpp 的 <span class="mono">CONTRIBUTING.md</span> 有一条<strong>明确的 AI 政策</strong>——不接受完全或主要由 AI 生成的 PR，且要求贡献者能独立理解、调试、维护自己提交的代码。这一课会把这条政策原样讲清楚，因为它直接决定了"怎样的贡献才会被这个项目接受"。路线图：编译（配一张"一次贡献的生命周期"追踪图）-> 测试 -> 调试 -> 贡献规范 -> 两个折叠深挖。一句话定个调：这一课的每个工具，最终都服务于同一个目标——让你的改动既改对了、又能被这个项目长期接住。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这一课的主线是一条<strong>回路</strong>：编译 -&gt; 改 -&gt; 测试 -&gt; 提 PR -&gt; CI 验证。它和前面所有"讲原理"的课不同——这里讲的是<strong>怎么真正参与进来</strong>。核心工具就三样：<span class="mono">CMake</span>（把源码编译成二进制，一套命令靠不同开关切换 CPU / CUDA / Metal / Vulkan 等后端）、<span class="mono">ctest</span>（跑自动化测试，其中 <span class="mono">test-backend-ops</span> 最关键，它逐算子比对不同后端的结果是否一致）、<span class="mono">CONTRIBUTING.md</span>（贡献规范，尤其是对 AI 生成 PR 的明确政策）。理解这条回路，你就从"读代码的人"变成了"能往里写、且写法能被项目接纳的人"。而且这条回路是<strong>自洽</strong>的：CONTRIBUTING 要求"改了 ggml 算子就要跑 test-backend-ops"、"新功能先只做 CPU"，正是因为 CPU 实现是所有后端的<strong>参考答案</strong>（呼应 L31/L33）——规范不是凭空定的规矩，而是从"怎么保证几十种后端都算对"这个工程现实里长出来的。还有一点值得先记在心里：这条回路是有方向的——总是"先能编、再改、再测、最后才提"，跳过任何一步（比如没跑测试就提 PR）都会在后面加倍还回来。把这个顺序刻进肌肉记忆，比记住任何一条具体命令都重要，因为命令可以现查，流程错了却要整段返工。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把参与 llama.cpp 想象成<strong>加入一个超大型开源厨房</strong>。你不能直接把菜端上桌：先得照菜谱（<span class="mono">build.md</span>）把灶台架起来——煤气灶、电磁炉、烤箱（CPU / CUDA / Metal）各有各的点火方式（不同 CMake 开关）。改好一道菜，不能自己觉得行就行，要先过<strong>标准化试吃</strong>（<span class="mono">ctest</span> / <span class="mono">test-backend-ops</span>：同一道菜在每个灶上做出来味道必须一致）。最后上桌前还得守<strong>餐厅规矩</strong>（<span class="mono">CONTRIBUTING</span>）：一次只上一道菜（一个 PR 一个功能）、新菜先出基础版（CPU 优先）、摆盘要符合统一格式（<span class="mono">clang-format</span>）。还有一条贴在墙上的硬规定：<strong>不收纯靠 AI 代做、你自己却讲不清怎么做的菜</strong>——因为出了问题，得是你、而不是 AI 来回锅。这套"标准化试吃 + 餐厅规矩"看着繁琐，其实全是在替你兜底：它让你改的东西在上桌前就被挡下明显的错，也让维护者敢放心收你的菜——没有这层保障，一个上百贡献者的厨房早就乱成一锅了。
</div>
<h2>编译：一套 CMake 命令，切换各种后端</h2>
<p>一切从编译开始。llama.cpp 用 CMake 构建，最基本的就两条命令：先<strong>配置</strong>（探测编译器和依赖、生成构建系统），再<strong>构建</strong>（真正把源码编译成二进制）。编译产物都落在 <span class="mono">build/bin/</span> 下——你熟悉的 <span class="mono">llama-cli</span>、<span class="mono">llama-server</span> 都在那。想换个后端（比如用上 NVIDIA GPU），不用改源码，只在<strong>配置那一步</strong>加一个开关：</p>
<pre class="code"><span class="cm"># 最基本的编译: 配置 + 构建 (来自 docs/build.md)</span>
cmake -B build                              <span class="cm"># 配置: 探测编译器/依赖, 生成构建系统</span>
cmake --build build --config Release        <span class="cm"># 构建: 真正编译, 产物落 build/bin/</span>

<span class="cm"># 换后端只改"配置"那步, 比如启用 CUDA:</span>
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release</pre>
<p>每个后端对应一个 <span class="mono">-DGGML_*</span> 开关，背后是 L31-L33 讲过的那套后端机制——同一份计算图，换一套算子实现。下面这张表把常用后端和它们的开关、适用平台列在一起，编译前对着选就行：</p>
<table class="t">
  <tr><th>后端</th><th>CMake 开关</th><th>适用平台</th></tr>
  <tr><td>CPU（默认）</td><td>无需开关</td><td>所有平台，参考实现</td></tr>
  <tr><td>CUDA</td><td>-DGGML_CUDA=ON</td><td>NVIDIA GPU</td></tr>
  <tr><td>Metal</td><td>默认开（-DGGML_METAL=OFF 可关）</td><td>Apple 芯片</td></tr>
  <tr><td>Vulkan</td><td>-DGGML_VULKAN=ON</td><td>跨厂商 GPU</td></tr>
  <tr><td>HIP / SYCL / MUSA</td><td>-DGGML_HIP / SYCL / MUSA=ON</td><td>AMD / Intel / 摩尔线程</td></tr>
</table>
<p>编译只是开发回路的起点。把"改一个东西、最后被合并"的全过程定格成一条流水，你会看到编译、测试、规范是怎么串成一条线的：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次贡献的生命周期</b>：从 clone 到合并，编译出二进制、改代码、跑测试确认没坏、格式化、提 PR、过 CI 矩阵，最后 squash 合并（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① clone</h5>
      <div class="cellrow"><span class="vc">本地拿到源码</span></div>
      <div class="tlab">fork + clone</div></div>
    <div class="op">cmake -B build<br>--build</div>
    <div class="stn"><h5>② 编译</h5>
      <div class="cellrow"><span class="vc blue">build/bin/ 二进制</span></div>
      <div class="tlab">确认能跑</div></div>
    <div class="op">改代码</div>
    <div class="stn"><h5>③ 改动</h5>
      <div class="cellrow"><span class="vc hot">你的修改</span></div>
      <div class="tlab">一个功能</div></div>
    <div class="op">ctest /<br>test-backend-ops</div>
    <div class="stn"><h5>④ 测试</h5>
      <div class="cellrow"><span class="vc blue">用例全过</span></div>
      <div class="tlab">没弄坏别处</div></div>
    <div class="op">clang-format<br>提 PR</div>
    <div class="stn"><h5>⑤ PR</h5>
      <div class="cellrow"><span class="vc">一功能 · CPU 优先</span></div>
      <div class="tlab">守 CONTRIBUTING</div></div>
    <div class="op">CI 矩阵<br>review</div>
    <div class="stn"><h5>⑥ 合并</h5>
      <div class="cellrow"><span class="vc blue">squash merge</span></div>
      <div class="tlab">进入 master</div></div>
  </div>
</div>
<h2>测试：ctest 与 test-backend-ops</h2>
<p>改完代码，怎么确认没把别处弄坏？跑测试。llama.cpp 的测试用 CTest 组织：<span class="mono">tests/</span> 下一堆 <span class="mono">test-*.cpp</span>，由 <span class="mono">tests/CMakeLists.txt</span> 里的几个宏注册成用例。最常用的是 <span class="mono">llama_build_and_test</span>（编译一个测试源文件并注册）和 <span class="mono">llama_test</span>（把同一个测试程序配上不同参数，注册成多个用例——比如 <span class="mono">test-tokenizer-0</span> 就对每个词表各跑一遍）。注册好之后，在 <span class="mono">build/</span> 里一条 <span class="mono">ctest</span> 就能把它们全跑起来：</p>
<pre class="code"><span class="cm"># tests/CMakeLists.txt: 用宏把一个 test-*.cpp 注册成 ctest 用例</span>
<span class="fn">llama_build_and_test</span>(test-backend-ops.cpp)         <span class="cm"># 编译并注册</span>
<span class="fn">llama_test</span>(test-tokenizer-0 NAME ... ARGS ...)     <span class="cm"># 参数化: 同一程序跑多个词表</span>

<span class="cm"># 在 build/ 里跑测试:</span>
ctest --test-dir build                 <span class="cm"># 跑全部用例</span>
ctest --test-dir build -R backend-ops  <span class="cm"># 只跑名字含 backend-ops 的</span></pre>
<p>测试分几类，各管一摊：<span class="mono">test-tokenizer-*</span> 验证分词结果和参考一致（呼应 L20）、<span class="mono">test-quantize-*</span> 验证量化/反量化的误差在范围内（呼应 L06/L12）、<span class="mono">test-sampling</span> 验证采样逻辑（呼应 L21）。但其中分量最重的是 <span class="mono">test-backend-ops</span>：它把 ggml 的每一个算子在不同后端上各算一遍，再逐元素比对结果是否一致。这就是 llama.cpp 敢同时维护十几种后端的底气——任何一个后端的任何一个算子写错了，这个测试都会当场抓出来。所以它不只是"一个测试"，而是整个多后端体系的正确性地基。顺带说一句它的运行成本：因为要在多个后端上各算一遍、还要逐元素比对，<span class="mono">test-backend-ops</span> 跑起来并不算快；但这恰恰是它的价值所在——它把"某个后端会不会悄悄算错"这种最难靠肉眼抓的 bug，提前压进了一次可重复的自动化比对里，用机器的耐心换人的安心。</p>
<h2>调试：Debug 构建、sanitizer 与算子核对</h2>
<p>测试告诉你"坏了"，调试帮你找到"哪坏了"。第一招是换 Debug 构建（<span class="mono">-DCMAKE_BUILD_TYPE=Debug</span>）：带上调试符号、关掉激进优化，崩溃时能看清调用栈、能单步跟。第二招是 sanitizer——CI 里专门有一条 <span class="mono">build-sanitize</span>，用 ASan / UBSan 跑测试，能逮住越界访问、未定义行为这类"平时不报、偶尔才炸"的 bug。在本地复现某个 CI 失败时，照着同样的 sanitizer 开关编一份，问题往往一下就现形。一条很实用的经验：Release 构建下那种偶发、难复现的崩溃，十有八九是内存越界或未定义行为，换成 Debug + sanitizer 重编一遍再跑，往往比盯着代码干想快得多——让工具替你把错误现场抓出来，是调试的第一性原则。</p>
<p>如果你动的是 ggml 算子，调试有个专属利器，还是它：<span class="mono">test-backend-ops</span>。它能把你改的那个算子在 CPU 和目标后端上各算一遍、逐元素比对，第一时间告诉你"结果对不对、从哪个元素开始偏"。CONTRIBUTING 也正因此点名：改了或新增 ggml 算子，必须跑（并补充）<span class="mono">test-backend-ops</span>。再配上各 example / 工具的 <span class="mono">--verbose</span> 日志，绝大多数推理层面的问题都能定位。这一节只点到为止——真正的调试功力得自己练，但你至少要知道这几件趁手的工具都摆在哪。还有一类问题不在 C++ 里、而在 Python 转换脚本或构建配置上——这时候 <span class="mono">python-lint</span> / <span class="mono">python-type-check</span> 这些 CI 检查就是你的第一道提示，本地照着跑一遍，能省掉一轮"提了才发现格式不过"的来回。</p>
<h2>贡献规范：怎样的 PR 会被接受</h2>
<p>工具会用了，最后这关是规矩。<span class="mono">CONTRIBUTING.md</span> 列的要求不多，但条条都为"让维护者能长期接住你的代码"：<strong>一个 PR 一个功能</strong>（改动聚焦才好审）、<strong>CPU 支持优先</strong>（新东西先做 CPU、其它后端放后续 PR）、改了 ggml 算子就<strong>跑并补 test-backend-ops</strong>、用 <span class="mono">clang-format</span>（clang-tools v15+）对齐风格。维护者合并时用 squash，提交标题写成 <span class="mono">&lt;module&gt; : &lt;title&gt; (#NNNN)</span> 这样的格式。把这些要件摆成一张过关清单：</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">范围</span><span class="name">一个 PR 一个功能</span></div><div class="ld">改动聚焦、好审；别把无关的重构和修复塞进同一个 PR</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">后端</span><span class="name">CPU 支持优先</span></div><div class="ld">新功能 / 新模型先只做 CPU，CUDA 等放后续 PR（CPU 是参考实现）</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">测试</span><span class="name">改 ggml 就跑 test-backend-ops</span></div><div class="ld">动了算子要跑、必要时补测例，保证各后端结果一致</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">风格</span><span class="name">clang-format + 标题格式</span></div><div class="ld">用 clang-format(v15+) 对齐；squash 标题写成 &lt;module&gt; : &lt;title&gt; (#NNNN)</div></div>
</div>
<p>最后是最该认真对待的一条——<strong>AI 使用政策</strong>。<span class="mono">CONTRIBUTING.md</span> 在很靠前的位置就写明：<strong>不接受完全或主要由 AI 生成的 PR</strong>；用 AI 先写、人再改，仍算 AI 生成。它要求贡献者能<strong>独立理解、调试、维护</strong>自己提交的代码，如实<strong>披露</strong> AI 的使用方式，并明确禁止用 AI 代写 PR 描述、issue、评论这类与人沟通的内容。这条政策的用意不难理解：一个 PR 是一份长期承诺——维护者要审它、集成它、长期支持它；项目要的从来不是"代码从哪来"，而是背后有没有一个能为它负责的人。所以如果你正用 AI 学这门课，最稳的姿势是：让它帮你<strong>读懂</strong>代码，而把"设计、决策、能讲清楚为什么"牢牢留在自己手里。说白了，这条政策保护的从来不是"纯手写"这个形式，而是"真有人懂这段代码、能在它出问题时扛起来"这个实质——而后者，恰恰是这整门课想帮你抵达的状态：不只是会用工具，而是心里真有底。</p>
<h2>深入：CPU 优先 与 CI 矩阵</h2>
<p>两个折叠，回答两个"为什么要这样规定"的问题——理解了它们，前面那些规矩就不再是死板的条文，而是有道理的工程选择。这也是读规范的正确姿势：别把它当成必须背的教条，而是去问每一条"它在防什么坏情况"——想通了防的是什么，你自然就记住了、也更愿意守。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么新功能要"CPU 支持优先"？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>乍看有点反直觉——大家不都冲着 GPU 加速来的吗，为什么新功能反而要先做 CPU？答案前面已经埋好了：CPU 实现是所有后端的<strong>参考答案</strong>。<span class="mono">test-backend-ops</span> 校验一个 CUDA / Metal 算子"对不对"，靠的就是拿它的结果和 CPU 版逐元素比对（呼应 L31 的 CPU 后端、L33 的后端调度）。如果一个新算子连 CPU 版都没有，就没有东西能给 GPU 版当 ground truth，正确性根本无从验证。再者，CPU 版人人能编、能跑（不挑硬件），维护者审 PR、别的贡献者复现问题都方便。所以"CPU 优先"不是看轻 GPU，而是先把"对"的基准立起来，再谈"快"——先正确、再加速，是这个项目一以贯之的工程顺序。把它和上一课连起来看也很自然：上一课新增一个模型也是"先让它在 CPU 上能转换、能跑通"，再谈别的；同一个"先立基准"的思路，在加模型和加算子两处各用了一次。反过来想也站得住：要是允许"先上 GPU 版、CPU 版以后再补"，那段时间里这个算子就没有任何参考答案，谁也说不清它到底算得对不对——规范要堵的正是这个洞。先把对错的尺子立起来，再谈谁跑得快，顺序错了，后面全是糊涂账。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> CI 为什么有几十个 workflow？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>翻一眼 <span class="mono">.github/workflows/</span>，你会看到几十个 yml 文件，乍看吓人。但理出来其实就两类。一类是<strong>构建矩阵</strong>：每个后端、每个平台各一套——<span class="mono">build-cpu</span>、<span class="mono">build-cuda-ubuntu</span> / <span class="mono">build-cuda-windows</span>、<span class="mono">build-vulkan</span>、<span class="mono">build-sycl</span>、<span class="mono">build-apple</span>、<span class="mono">build-android</span>……为什么这么多？因为 llama.cpp 的卖点就是"哪儿都能跑"（呼应 L33 后端调度），那就得在哪儿都编一遍、测一遍，少测一个组合就可能悄悄坏掉。另一类是<strong>质量门</strong>：<span class="mono">code-style</span>（命名 / 风格约定检查）、<span class="mono">editorconfig</span>、<span class="mono">python-lint</span> / <span class="mono">python-type-check</span>、<span class="mono">build-sanitize</span>（ASan/UBSan）等，把"风格统一、没低级错误"也自动卡住。你提一个 PR，这一整套会自动在各平台跑一遍——这就是为什么"在我机器上能编"远远不够：得在这个矩阵的每一格里都绿，才算真的没破坏跨平台支持。看懂这一点，你对"开源大项目怎么在几十种环境里保持不崩"也就有了答案：不是靠人去手动测，而是把"每种环境编一遍、测一遍"写成了自动跑的 CI。这也解释了一个常让新人困惑的现象：一个在自己电脑上明明没问题的 PR，却被 CI 拦了下来——很可能只是某个你手头根本没有的平台编不过。而正因为 CI 替所有人把这些平台都试了一遍，你才不必自己去凑一屋子设备，这是开源协作里一种隐形却巨大的便利。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>编译：<span class="mono">cmake -B build</span> 配置、<span class="mono">cmake --build build --config Release</span> 编译；后端靠开关切换（<span class="mono">-DGGML_CUDA=ON</span> 等）；产物在 <span class="mono">build/bin/</span>。</li>
    <li>测试：<span class="mono">ctest</span> 跑用例；<span class="mono">tests/CMakeLists.txt</span> 用 <span class="mono">llama_test</span>/<span class="mono">llama_build_and_test</span> 宏注册；<span class="mono">test-backend-ops</span> 逐算子比对各后端结果。</li>
    <li>调试：Debug 构建 + sanitizer（<span class="mono">build-sanitize</span> CI）；改 ggml 算子务必跑 <span class="mono">test-backend-ops</span>。</li>
    <li>贡献：<span class="mono">CONTRIBUTING.md</span> 规定一个 PR 一个功能、CPU 支持优先、squash 合并、标题 <span class="mono">&lt;module&gt; : &lt;title&gt; (#NNNN)</span>、<span class="mono">clang-format</span>（clang-tools v15+）。</li>
    <li>AI 政策：不接受完全或主要由 AI 生成的 PR；贡献者须能独立理解、调试、维护自己的代码，并如实披露 AI 的使用方式。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  这一课最值得带走的，是 llama.cpp 怎么把"对不对"变成一件<strong>可自动检验</strong>的事。<span class="mono">test-backend-ops</span> 不去抽象地论证某个 CUDA kernel "正确"，而是把正确<strong>操作性地定义</strong>成"和 CPU 参考实现逐元素一致"——于是几十种后端、上千个算子的正确性，统统化成一句可以一键跑的断言。这就是为什么 CONTRIBUTING 要"CPU 优先"：得先有那个谁都要对齐的<strong>参考答案</strong>，后面的 CUDA / Metal / Vulkan 才有"对"可言。把这个思路记住，你会发现它远不止用于内核：任何"多实现必须等价"的系统，都能用"挑一个当 ground truth、其余对齐它"来把质量问题自动化。至于那条 AI 政策，底层逻辑也一脉相承——项目要的不是"代码从哪来"，而是"有没有一个能对它负责、能在半夜出 bug 时把它修好的人"；可维护性，才是开源协作真正的硬通货。也给这门课收个尾：从第一课认识 llama.cpp 是什么，到这一课学会怎么给它提一个站得住脚的 PR，你已经走完了"读懂 -&gt; 动手 -&gt; 贡献"的整条路——剩下的，就是挑一个你真正在意的问题，去把它修好。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Once you can read the source, the natural next step is to act: change a line, fix a bug, add a feature, then turn it into a PR that can be merged. But "acting" has its own process - how to compile this million-line C++ project, how to confirm a change did not break something else, and what rules a PR must follow. This lesson teaches no new internals; it walks you through that real development loop end to end, so the "reading" from thirty-some lessons turns into "can change it, can contribute".
</p>
<p style="color:var(--muted);margin-top:.4rem">The loop is really three parts: <strong>build</strong> (use one set of CMake commands to turn source into binaries under <span class="mono">build/bin/</span>, flipping a flag to switch to CUDA / Metal / Vulkan and other backends), <strong>test</strong> (run automated cases with <span class="mono">ctest</span>, where <span class="mono">test-backend-ops</span> specifically checks that operators agree across backends), and <strong>contribute</strong> (open a PR per <span class="mono">CONTRIBUTING.md</span>: one feature per PR, CPU support first, align style with <span class="mono">clang-format</span>). String the three together and you have a full path from "edit code locally" to "accepted upstream". Walk this path once and the playbook for almost any unfamiliar large C++ open-source project looks much the same.</p>
<p style="color:var(--muted)">One thing must be stated up front, especially since you may be reading this course with AI help: llama.cpp's <span class="mono">CONTRIBUTING.md</span> has an <strong>explicit AI policy</strong> - it does not accept PRs that are fully or predominantly AI-generated, and it requires contributors to independently understand, debug, and maintain the code they submit. This lesson states that policy as-is, because it directly determines "what kind of contribution this project will accept". Roadmap: build (with a "lifecycle of one contribution" trace) -> test -> debug -> contribution rules -> two deep-dive accordions. To set the tone in one line: every tool in this lesson ultimately serves the same goal - making your change both correct and something the project can carry for the long haul.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  This lesson's spine is a <strong>loop</strong>: build -&gt; change -&gt; test -&gt; open a PR -&gt; CI. Unlike every prior "explain the internals" lesson, this one is about <strong>how to actually take part</strong>. The core tools are just three: <span class="mono">CMake</span> (compile source into binaries, one command set switching CPU / CUDA / Metal / Vulkan via different flags), <span class="mono">ctest</span> (run automated tests, of which <span class="mono">test-backend-ops</span> matters most - it compares operator results across backends), and <span class="mono">CONTRIBUTING.md</span> (the contribution rules, notably the explicit policy on AI-generated PRs). Understand the loop and you turn from "someone who reads the code" into "someone who can write into it in a way the project will accept". And the loop is <strong>self-consistent</strong>: CONTRIBUTING asks "if you change a ggml operator, run test-backend-ops" and "do a new feature on CPU first" precisely because the CPU implementation is the <strong>reference answer</strong> for all backends (recall L31/L33) - the rules are not arbitrary, they grow out of the engineering reality of "how do you guarantee dozens of backends all compute right". One more thing to fix in mind first: this loop has a direction - always "build first, then change, then test, and only then submit", and skipping any step (like opening a PR without running tests) comes back to bite you double later. Burning that order into muscle memory matters more than memorizing any single command, because commands you can look up, but a broken process means reworking whole chunks.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture joining llama.cpp as <strong>joining a giant open-source kitchen</strong>. You cannot just plate a dish and serve it: first set up the stove per the recipe (<span class="mono">build.md</span>) - gas burner, induction hob, oven (CPU / CUDA / Metal) each light differently (different CMake flags). After cooking a dish you do not get to call it good yourself; it must pass a <strong>standardized taste test</strong> (<span class="mono">ctest</span> / <span class="mono">test-backend-ops</span>: the same dish must taste identical made on every stove). Finally, before serving, you follow <strong>house rules</strong> (<span class="mono">CONTRIBUTING</span>): one dish at a time (one feature per PR), a new dish ships a basic version first (CPU first), and plating follows one format (<span class="mono">clang-format</span>). There is also a hard rule on the wall: <strong>no dishes that were entirely AI-cooked while you cannot explain how they were made</strong> - because when something goes wrong, it has to be you, not the AI, who reworks it. This "standardized taste test plus house rules" looks fussy but is all there to catch you: it stops obvious mistakes in your change before it reaches the table, and lets maintainers accept your dish with confidence - without that safety net a kitchen of hundreds of contributors would have descended into chaos long ago.
</div>
<h2>Build: one set of CMake commands, switching backends</h2>
<p>It all starts with the build. llama.cpp builds with CMake, and the basics are just two commands: first <strong>configure</strong> (detect the compiler and dependencies, generate the build system), then <strong>build</strong> (actually compile source into binaries). The build output lands under <span class="mono">build/bin/</span> - the <span class="mono">llama-cli</span> and <span class="mono">llama-server</span> you know are right there. To switch backends (say, to use an NVIDIA GPU) you do not touch the source; you add one flag at the <strong>configure step</strong>:</p>
<pre class="code"><span class="cm"># the basic build: configure + build (from docs/build.md)</span>
cmake -B build                              <span class="cm"># configure: detect compiler/deps, generate the build system</span>
cmake --build build --config Release        <span class="cm"># build: actually compile, output to build/bin/</span>

<span class="cm"># switching backend only changes the "configure" step, e.g. enable CUDA:</span>
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release</pre>
<p>Each backend has a <span class="mono">-DGGML_*</span> flag, backed by the backend machinery from L31-L33 - the same compute graph, a different set of operator implementations. The table below lists the common backends with their flags and target platforms; pick from it before you build:</p>
<table class="t">
  <tr><th>backend</th><th>CMake flag</th><th>target platform</th></tr>
  <tr><td>CPU (default)</td><td>no flag needed</td><td>all platforms, reference impl</td></tr>
  <tr><td>CUDA</td><td>-DGGML_CUDA=ON</td><td>NVIDIA GPU</td></tr>
  <tr><td>Metal</td><td>on by default (-DGGML_METAL=OFF to disable)</td><td>Apple silicon</td></tr>
  <tr><td>Vulkan</td><td>-DGGML_VULKAN=ON</td><td>cross-vendor GPU</td></tr>
  <tr><td>HIP / SYCL / MUSA</td><td>-DGGML_HIP / SYCL / MUSA=ON</td><td>AMD / Intel / Moore Threads</td></tr>
</table>
<p>Building is only the start of the development loop. Freeze the whole "change one thing, end up merged" path into a flow and you see how build, test, and rules string into one line:</p>
<div class="trace">
  <div class="tcap"><b>Tracing the lifecycle of one contribution</b>: from clone to merge - build the binary, change code, run tests to confirm nothing broke, format, open a PR, pass the CI matrix, finally squash-merge (schematic).</div>
  <div class="stations">
    <div class="stn"><h5>(1) clone</h5>
      <div class="cellrow"><span class="vc">source locally</span></div>
      <div class="tlab">fork + clone</div></div>
    <div class="op">cmake -B build<br>--build</div>
    <div class="stn"><h5>(2) build</h5>
      <div class="cellrow"><span class="vc blue">build/bin/ binaries</span></div>
      <div class="tlab">confirm it runs</div></div>
    <div class="op">change code</div>
    <div class="stn"><h5>(3) change</h5>
      <div class="cellrow"><span class="vc hot">your edit</span></div>
      <div class="tlab">one feature</div></div>
    <div class="op">ctest /<br>test-backend-ops</div>
    <div class="stn"><h5>(4) test</h5>
      <div class="cellrow"><span class="vc blue">cases all pass</span></div>
      <div class="tlab">nothing else broke</div></div>
    <div class="op">clang-format<br>open PR</div>
    <div class="stn"><h5>(5) PR</h5>
      <div class="cellrow"><span class="vc">one feature . CPU first</span></div>
      <div class="tlab">follow CONTRIBUTING</div></div>
    <div class="op">CI matrix<br>review</div>
    <div class="stn"><h5>(6) merge</h5>
      <div class="cellrow"><span class="vc blue">squash merge</span></div>
      <div class="tlab">into master</div></div>
  </div>
</div>
<h2>Test: ctest and test-backend-ops</h2>
<p>After a change, how do you confirm nothing else broke? Run the tests. llama.cpp's tests are organized with CTest: a pile of <span class="mono">test-*.cpp</span> under <span class="mono">tests/</span>, registered as cases by a few macros in <span class="mono">tests/CMakeLists.txt</span>. The most common are <span class="mono">llama_build_and_test</span> (compile a test source and register it) and <span class="mono">llama_test</span> (register the same test program with different arguments as several cases - e.g. <span class="mono">test-tokenizer-0</span> runs once per vocab). Once registered, one <span class="mono">ctest</span> in <span class="mono">build/</span> runs them all:</p>
<pre class="code"><span class="cm"># tests/CMakeLists.txt: macros register a test-*.cpp as a ctest case</span>
<span class="fn">llama_build_and_test</span>(test-backend-ops.cpp)         <span class="cm"># compile and register</span>
<span class="fn">llama_test</span>(test-tokenizer-0 NAME ... ARGS ...)     <span class="cm"># parameterized: one program, many vocabs</span>

<span class="cm"># run tests in build/:</span>
ctest --test-dir build                 <span class="cm"># run every case</span>
ctest --test-dir build -R backend-ops  <span class="cm"># only those whose name contains backend-ops</span></pre>
<p>Tests fall into a few groups, each minding its own patch: <span class="mono">test-tokenizer-*</span> checks that tokenization matches the reference (recall L20), <span class="mono">test-quantize-*</span> checks that quantize/dequantize error stays in range (recall L06/L12), <span class="mono">test-sampling</span> checks the sampling logic (recall L21). But the heaviest of all is <span class="mono">test-backend-ops</span>: it runs every ggml operator on each backend and compares results elementwise. This is what lets llama.cpp dare to maintain a dozen-plus backends at once - if any operator on any backend is wrong, this test catches it on the spot. So it is not just "a test" but the correctness foundation of the whole multi-backend system. A note on its running cost: because it computes on several backends and compares elementwise, <span class="mono">test-backend-ops</span> is not fast to run; but that is exactly its value - it presses the hardest-to-eyeball bug, "might some backend quietly compute wrong", into one repeatable automated comparison, trading the machine's patience for the human's peace of mind.</p>
<h2>Debug: Debug builds, sanitizers, and operator checking</h2>
<p>Tests tell you "it broke"; debugging helps you find "where it broke". The first move is a Debug build (<span class="mono">-DCMAKE_BUILD_TYPE=Debug</span>): with debug symbols and aggressive optimization off, you can read the call stack at a crash and single-step. The second is sanitizers - CI has a dedicated <span class="mono">build-sanitize</span> that runs tests under ASan / UBSan, catching out-of-bounds access and undefined behavior, the kind of "usually silent, occasionally explodes" bug. When reproducing a CI failure locally, building with the same sanitizer flags often makes the problem surface at once. A handy rule of thumb: an intermittent, hard-to-reproduce crash under a Release build is almost always a memory overrun or undefined behavior - rebuild with Debug + sanitizer and rerun, and that usually beats staring at the code guessing; letting the tools catch the error scene for you is the first principle of debugging.</p>
<p>If you are touching a ggml operator, debugging has a dedicated weapon - the same one: <span class="mono">test-backend-ops</span>. It runs the operator you changed on both CPU and the target backend and compares elementwise, telling you immediately "is the result right, and from which element does it diverge". This is exactly why CONTRIBUTING calls it out: change or add a ggml operator and you must run (and extend) <span class="mono">test-backend-ops</span>. Add the <span class="mono">--verbose</span> logging of the various examples / tools and you can locate the vast majority of inference-level problems. This section only points the way - real debugging skill is earned by practice, but you should at least know where these handy tools sit. There is also a class of problems that live not in the C++ but in the Python conversion scripts or build config - there the <span class="mono">python-lint</span> / <span class="mono">python-type-check</span> CI checks are your first hint; running them locally saves a round of "submit, then find out the format fails".</p>
<h2>Contribution rules: what kind of PR gets accepted</h2>
<p>With the tools in hand, the last gate is the rules. <span class="mono">CONTRIBUTING.md</span> lists few requirements, but each exists so maintainers can take on your code for the long haul: <strong>one feature per PR</strong> (a focused change is reviewable), <strong>CPU support first</strong> (do the new thing on CPU, leave other backends to follow-up PRs), <strong>run and extend test-backend-ops</strong> if you touched a ggml operator, and align style with <span class="mono">clang-format</span> (clang-tools v15+). Maintainers merge by squash, with a commit title in the form <span class="mono">&lt;module&gt; : &lt;title&gt; (#NNNN)</span>. Laid out as a pass/fail checklist:</p>
<div class="layers">
  <div class="layer l-core"><div class="lh"><span class="badge">scope</span><span class="name">one feature per PR</span></div><div class="ld">a focused, reviewable change; do not stuff unrelated refactors and fixes into one PR</div></div>
  <div class="layer l-main"><div class="lh"><span class="badge">backend</span><span class="name">CPU support first</span></div><div class="ld">new feature / model on CPU first, CUDA and others in follow-up PRs (CPU is the reference impl)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">test</span><span class="name">touch ggml -&gt; run test-backend-ops</span></div><div class="ld">run it when you change an operator, extend it if needed, keep backends in agreement</div></div>
  <div class="layer l-app"><div class="lh"><span class="badge">style</span><span class="name">clang-format + title format</span></div><div class="ld">align with clang-format (v15+); squash title as &lt;module&gt; : &lt;title&gt; (#NNNN)</div></div>
</div>
<p>Last, the one to take most seriously - the <strong>AI usage policy</strong>. <span class="mono">CONTRIBUTING.md</span> states it plainly: PRs that are <strong>fully or predominantly AI-generated are not accepted</strong>; AI-written-then-human-edited still counts as AI-generated. It requires contributors to <strong>independently understand, debug, and maintain</strong> the code they submit, to <strong>disclose</strong> how AI was used, and explicitly forbids using AI to write the human-facing parts - PR descriptions, issues, comments. The intent is not hard to see: a PR is a long-term commitment - maintainers review it, integrate it, support it for the long haul; the project never cares "where the code came from" but whether there is a person who can be responsible for it. So if you are learning this course with AI, the safest stance is: let it help you <strong>read</strong> the code, and keep "design, decisions, being able to explain the why" firmly in your own hands. Put plainly, this policy protects not the form of "handwritten only" but the substance of "someone really understands this code and can carry it when it breaks" - and that substance is exactly the state this whole course wants to bring you to: not just knowing how to use tools, but having solid ground under your feet.</p>
<h2>Deeper: CPU-first and the CI matrix</h2>
<p>Two accordions answering two "why is it required this way" questions - understand them and the earlier rules stop being rigid clauses and become reasoned engineering choices. This is also the right way to read the rules: do not treat them as dogma to memorize, but ask of each one "what bad case is it preventing" - once you see what is being prevented, you remember it naturally and are more willing to follow it.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why must a new feature be "CPU support first"? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>It looks counterintuitive at first - is not everyone here for GPU acceleration, so why do a new feature on CPU first? The answer was planted earlier: the CPU implementation is the <strong>reference answer</strong> for all backends. <span class="mono">test-backend-ops</span> checks whether a CUDA / Metal operator is "correct" by comparing its result elementwise against the CPU version (recall the CPU backend in L31, backend dispatch in L33). If a new operator does not even have a CPU version, there is nothing to serve as ground truth for the GPU version, and correctness cannot be verified at all. Moreover the CPU version builds and runs for everyone (no special hardware), making it easy for maintainers to review and for other contributors to reproduce issues. So "CPU first" does not slight the GPU; it stands up the baseline of "correct" before talking about "fast" - correctness first, then acceleration, is this project's consistent engineering order. It connects naturally with the last lesson: adding a model there was also "first make it convert and run on CPU", then the rest - the same "set the baseline first" idea, used once for adding models and once for adding operators. The reverse holds too: if "ship the GPU version first, add the CPU version later" were allowed, during that window the operator would have no reference answer at all and nobody could say whether it computes correctly - the rule plugs exactly this hole. Stand up the ruler of right and wrong first, then talk about who runs fast; get the order wrong and the rest is a muddle.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Why does CI have dozens of workflows? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Glance at <span class="mono">.github/workflows/</span> and you see dozens of yml files, daunting at first. But sorted out they are just two kinds. One is the <strong>build matrix</strong>: one per backend, per platform - <span class="mono">build-cpu</span>, <span class="mono">build-cuda-ubuntu</span> / <span class="mono">build-cuda-windows</span>, <span class="mono">build-vulkan</span>, <span class="mono">build-sycl</span>, <span class="mono">build-apple</span>, <span class="mono">build-android</span>... why so many? Because llama.cpp's whole selling point is "runs anywhere" (recall backend dispatch in L33), so it must compile and test everywhere; skip testing one combination and it can quietly break. The other kind is <strong>quality gates</strong>: <span class="mono">code-style</span> (naming / style-convention check), <span class="mono">editorconfig</span>, <span class="mono">python-lint</span> / <span class="mono">python-type-check</span>, <span class="mono">build-sanitize</span> (ASan/UBSan), and so on, automatically blocking "inconsistent style, silly mistakes". When you open a PR this whole set runs across platforms automatically - which is why "it builds on my machine" is nowhere near enough: it must be green in every cell of this matrix to count as truly not breaking cross-platform support. See this and you also have the answer to "how does a big open-source project stay un-broken across dozens of environments": not by manual testing, but by writing "compile and test in each environment" into CI that runs itself. This also explains a thing that often puzzles newcomers: a PR that clearly works on your own machine gets blocked by CI - likely just because some platform you do not even have fails to compile. And precisely because CI tries all those platforms for everyone, you do not have to assemble a roomful of devices yourself - an invisible but huge convenience of open-source collaboration.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Build: <span class="mono">cmake -B build</span> to configure, <span class="mono">cmake --build build --config Release</span> to compile; backends switch via flags (<span class="mono">-DGGML_CUDA=ON</span> etc.); output lands in <span class="mono">build/bin/</span>.</li>
    <li>Test: <span class="mono">ctest</span> runs cases; <span class="mono">tests/CMakeLists.txt</span> registers them via the <span class="mono">llama_test</span>/<span class="mono">llama_build_and_test</span> macros; <span class="mono">test-backend-ops</span> compares operator results across backends.</li>
    <li>Debug: Debug build + sanitizers (the <span class="mono">build-sanitize</span> CI); changing a ggml operator means running <span class="mono">test-backend-ops</span>.</li>
    <li>Contribute: <span class="mono">CONTRIBUTING.md</span> mandates one feature per PR, CPU support first, squash merge, title <span class="mono">&lt;module&gt; : &lt;title&gt; (#NNNN)</span>, and <span class="mono">clang-format</span> (clang-tools v15+).</li>
    <li>AI policy: PRs that are fully or predominantly AI-generated are not accepted; contributors must be able to independently understand, debug, and maintain their code, and must disclose how AI was used.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The best thing to take from this lesson is how llama.cpp turns "is it correct" into something <strong>automatically checkable</strong>. <span class="mono">test-backend-ops</span> does not argue abstractly that some CUDA kernel is "correct"; it <strong>operationally defines</strong> correct as "elementwise identical to the CPU reference" - so the correctness of dozens of backends and thousands of operators collapses into one assertion you can run with a single command. That is why CONTRIBUTING says "CPU first": you need that reference answer everyone aligns to before CUDA / Metal / Vulkan can even have a notion of "right". Keep this idea and you will see it far beyond kernels: any system where "multiple implementations must be equivalent" can automate quality by "picking one as ground truth and aligning the rest to it". As for the AI policy, the underlying logic is the same thread - the project does not care "where the code came from" but "whether there is a person who can be responsible for it, who can fix it when it breaks at 2am"; maintainability is the real hard currency of open-source collaboration. And to close the course: from lesson one learning what llama.cpp is, to this lesson learning how to open a PR that stands up, you have walked the whole path of "understand -&gt; act -&gt; contribute" - what is left is to pick a problem you genuinely care about and go fix it.
</div>
""",
}
