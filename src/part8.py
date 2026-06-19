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
    <div class="op">set_gguf_params<br>prepare_tensors</div>
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
    <div class="op">set_gguf_params<br>prepare_tensors</div>
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
