"""Content for Part 2 (foundations)."""

LESSON_04 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
课 03 看清了"循环怎么一圈圈转"。这一课往下钻一层：一个 transformer block 内部到底在算什么、
"<strong>自回归</strong>"为什么能成立，以及最关键的——<strong>KV cache 凭什么是精确的</strong>，而不是一种"差不多就行"的近似。
看懂这一层，你就明白课 03 那个循环为什么"可以这么省"。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  自回归生成像<strong>玩文字接龙</strong>：你每次只看<strong>已经写出来的那段话</strong>，猜出下一个最合适的字，写下去；
  然后把这个新字也算进"已写内容"，再猜下一个。模型也是这样——它永远只根据"<strong>到目前为止的全部 token</strong>"
  预测"<strong>下一个 token</strong>"，写一个、回头看一遍、再写一个。正因为"<strong>每一步都把自己刚写的字也读进去</strong>"，
  模型才会前后连贯——这也是"自回归"里"自"字的由来：用自己产生的输出，作为下一步的输入。
</div>

<h2>decoder-only：一个 block 在算什么</h2>
<p>课 02 说过 llama 系模型都是 <strong>decoder-only</strong> 结构。把它拆开，数据从下往上是一条很直的链：词 id 先查成向量，
再穿过<strong>许多层结构相同的 block</strong>，最后投影回词表，得到每个候选词的分数。一层 block 内部又分两个子层——
<strong>自注意力</strong>和<strong>前馈网络（FFN）</strong>：</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>词嵌入 Embedding</h4>
    <p>把每个 token id 查成一个稠密向量（词向量）；模型后面所有运算都在这些向量上进行。</p>
    <p class="mono">src/llama-graph.cpp · build_inp_embd</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>自注意力子层（每个 block 的前半，× N 个 block）</h4>
    <p>RMSNorm 归一化 -&gt; 自注意力（含 RoPE 位置编码）-&gt; 残差相加。<strong>token 之间唯一互相交流的地方就在这里。</strong></p>
    <p class="mono">ggml_rms_norm · ggml_rope · ggml_soft_max_ext</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>前馈子层（同一 block 的后半）</h4>
    <p>再一次 RMSNorm -&gt; 前馈网络（SwiGLU）-&gt; 残差相加。对<strong>每个 token 各自</strong>做一次非线性加工，位置之间互不影响。</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>末层归一化 Final Norm</h4>
    <p>所有 block 叠完后，再过一道 RMSNorm，稳定输出的数值尺度。</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>输出投影 -&gt; logits</h4>
    <p>用输出层（lm_head）把向量投影回<strong>词表大小</strong>，得到每个候选 token 的分数（logits）。</p>
    <p class="mono">llama_get_logits_ith 取出</p>
  </div></div>
</div>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里有个贯穿全课的关键认识：<strong>只有自注意力这一层，token 之间才会"互相看见"</strong>。FFN、归一化、投影都是<strong>对每个 token 各算各的</strong>，位置之间互不影响；唯独注意力会让"当前 token"去参考"其它 token"。所以"模型怎么利用上下文"这件事，<strong>全发生在注意力层</strong>——这也是后面因果掩码、KV cache 都围着注意力打转的原因。那两条<strong>残差</strong>支路和<strong>归一化</strong>也别小看：残差让梯度能顺畅地穿过几十层不衰减，归一化把每层输入拉回稳定范围，二者合起来才让"<strong>把很多层 block 叠很深</strong>"这件事在训练上变得可行。
</div>
<p>顺带说说<strong>词嵌入</strong>为什么重要。它不是简单的"查字典编号"，而是把每个 token 映射到高维空间里的一个点，
<strong>语义相近的词，向量也彼此靠近</strong>——这正是模型"理解"语言的起点。原始词向量只带"这个词大概什么意思"，还不含上下文；
真正让它"读懂整句"的，是随后那几十层 block。<strong>你可以把多层 block 想成一条逐级精炼的流水线</strong>：浅层更多在抓局部搭配与语法
（谁修饰谁、短语边界在哪），深层逐渐组合出更抽象的语义与长程关系（指代、逻辑、主题）。每过一层，token 的向量就被
"<strong>注入更多来自上下文的信息</strong>"；到顶层时，最后一个位置的向量已经浓缩了"接下来最该说什么"的全部线索，投影到词表就成了 logits。</p>
<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  再把<strong>注意力到底在干嘛</strong>说透一点：它像在做一次"<strong>带权重的检索</strong>"。当前 token 拿着自己的 Query（一个"我想找什么"的提问），去和每个历史 token 的 Key（一个"我是什么"的标签）逐一匹配，越像就给越高的注意力权重；然后按这些权重，把各历史 token 的 Value（"我能提供的内容"）<strong>加权求和</strong>，汇成当前 token 的新表示。于是"<strong>理解上下文</strong>"被落实成了"<strong>该重点参考前文哪几个词</strong>"。这也再次说明：为什么注意力是 token 之间唯一的交流通道——只有在这里，一个 token 的输出才真正取决于<strong>别的</strong> token。
</div>
<p>还有个常被忽略的点：注意力本身<strong>并不知道词的先后顺序</strong>——它只是在做"按相似度加权汇总"，把同样几个词打乱顺序喂进去，
纯注意力算出的结果竟然一样。可语言显然讲究语序（"狗咬人"和"人咬狗"天差地别）。<strong>位置信息</strong>就是为此补进来的：llama 系普遍用
<strong>RoPE（旋转位置编码）</strong>，它不是简单地给每个位置加一个"序号向量"，而是<strong>按位置给 Query / Key 做一次旋转</strong>，
让两个 token 的注意力分数自然带上"它们相距多远"的信息。这也是为什么前面 vflow 里，自注意力子层特意标了"含 RoPE"。</p>
<p>顺手把最后那一步<strong>输出投影</strong>也说明白：它把顶层那个 token 向量，乘上一个"<strong>词表大小 × 隐藏维度</strong>"的大矩阵（lm_head），
得到<strong>词表里每个 token 各一个分数</strong>——这就是 logits 的长度恒等于词表大小的原因。不少模型还让它和最底层的词嵌入矩阵
<strong>共享权重</strong>（weight tying），既省参数，又让"输入怎么编码"和"输出怎么打分"保持一致。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  顺便建立一点"体量感"：所谓 7B、13B 模型，<strong>参数量</strong>主要就堆在这些 block 里——每层的注意力投影矩阵、FFN 里两三个大矩阵，乘上几十层，再加上巨大的词嵌入与输出投影，加总就是几十亿个数。<strong>层数（n_layer）、隐藏维度（n_embd）、注意力头数</strong>这几个超参，基本决定了一个模型有多大、多强、跑起来多吃资源——后面看 GGUF 元数据时，你会在文件头里直接读到它们。
</div>
<p>把一层 block 的前向写成伪代码，就是"两条带残差的支路"：</p>
<pre class="code"><span class="cm"># 一层 block 的前向: 两条带残差的支路</span>
<span class="kw">def</span> <span class="fn">layer</span>(x):                 <span class="cm"># x: [n_tokens, n_embd]</span>
    a = <span class="fn">attn</span>(<span class="fn">rms_norm</span>(x))     <span class="cm"># tokens talk to each other here</span>
    x = x + a                 <span class="cm"># residual</span>
    f = <span class="fn">ffn</span>(<span class="fn">rms_norm</span>(x))      <span class="cm"># per-token non-linear mix</span>
    <span class="kw">return</span> x + f              <span class="cm"># residual</span></pre>

<h2>因果掩码：只能回头看</h2>
<p>既然注意力让 token 互相参考，那"写第 5 个字时能不能偷看第 6、第 7 个字"？<strong>绝对不能</strong>——生成时它们还不存在。
于是 decoder 给注意力加了一道<strong>因果掩码（causal mask）</strong>：第 i 个 token 只允许注意到<strong>位置 &lt;= i</strong> 的 token，
对"未来"的位置一律屏蔽。画成一张方格表，就是一个<strong>下三角</strong>：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>因果掩码</b>：第 i 个 token 只能注意到位置 &lt;= i 的 token；亮格=可见，灰格=屏蔽（在自己之后）</div>
  <div class="cells"><span class="lab">t1 看</span><span class="cell hl">k1</span><span class="cell dim">k2</span><span class="cell dim">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t2 看</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell dim">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t3 看</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell hl">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t4 看</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell hl">k3</span><span class="cell hl">k4</span></div>
</div>
<p>表里每一行是"某个 token 在看谁"：亮格表示<strong>可以注意</strong>（不晚于自己），灰格表示<strong>被屏蔽</strong>（在自己之后）。
实现上很直接：算完注意力分数后，把所有"未来位置"的分数<strong>置成 -inf</strong>，再过 softmax，这些位置的权重就变成 0，等于没看。
<strong>这就是"自回归"在数学上的样子</strong>——每个位置的输出只依赖它<strong>及它之前</strong>的输入，绝不泄露未来。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  <strong>因果掩码还有一个训练时的妙用，值得点破。</strong>训练时我们手里有<strong>完整的句子</strong>，可以把它整段一次喂进模型；因果掩码保证第 i 个位置<strong>只能看到 &lt;= i 的词</strong>，于是"用前文预测下一个词"这件事，在<strong>每一个位置上同时成立</strong>——一次前向就拿到了全句每个位置的"下一个词"预测，可以并行地和真实的下一个词比对、算损失。这种"<strong>把答案也一起喂进去、但靠掩码挡住未来</strong>"的训练方式叫 teacher forcing，让训练极其高效。有意思的是：<strong>这和推理时的 prefill 是同一套机制</strong>——prompt 里每个 token 都已知，于是能像训练那样一次并行算完，只不过推理时我们只关心最后一个位置的输出。
</div>
<p>也正因为训练和推理共用<strong>同一套前向</strong>，llama.cpp 里并没有"两套代码"：无论是 prefill 一次喂进几百个 token，
还是 decode 每次只喂 1 个新 token，走的都是<strong>同一张计算图</strong>，区别只在"这一批有几个 token、要输出哪些位置"。
理解了这点，再回看课 03 的七步数据流，就明白它为什么能用一个 <span class="mono">llama_decode</span> 同时扛起这两种节奏。</p>
<p>llama.cpp 里这一步由 <span class="mono">src/llama-graph.cpp</span> 的 <span class="mono">build_attn</span> / <span class="mono">build_attn_mha</span>
拼进计算图，掩码通过 <span class="mono">ggml_soft_max_ext</span> 在求 softmax 时一并施加，简化出来大致是：</p>
<pre class="code"><span class="cm">// 注意力打分 + 因果掩码 (对应 build_attn / build_attn_mha)</span>
kq  = <span class="fn">ggml_mul_mat</span>(ctx, k, q);            <span class="cm">// scores [n_kv, n_q]</span>
kq  = <span class="fn">ggml_soft_max_ext</span>(ctx, kq, mask,    <span class="cm">// mask: causal -inf on j&gt;i</span>
                        scale, max_bias);
kqv = <span class="fn">ggml_mul_mat</span>(ctx, v, kq);           <span class="cm">// weighted sum of values</span></pre>
<p>（以本仓库 2026-06-15 源码为准；真实实现散落在 <span class="mono">llama-graph.cpp</span>，这里只取主干、略去缩放与多头细节。）</p>

<h2>自回归 + 为什么 KV cache 是精确的</h2>
<p>把"每次只根据已有 token 预测下一个"画成回路，就是自回归循环：</p>
<div class="flow">
  <div class="node"><div class="nt">已有 token</div><div class="nd">x1 … xn</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">前向</div><div class="nd">N 层 block</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">末位 logits</div><div class="nd">词表打分</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">采样</div><div class="nd">选一个 token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">x(n+1)</div><div class="nd">回灌队尾, 继续</div></div>
</div>
<p>注意末端那条<strong>回灌</strong>箭头——新采样出的 token 被接到序列尾巴，成为下一轮的输入。这里就引出全课的"题眼"：
每往后写一个字，是不是都要把<strong>前面所有字</strong>重新算一遍注意力？<strong>不需要</strong>。原因恰恰藏在因果掩码里。</p>
<p>注意力里每个 token 会算出三样东西：<strong>Query</strong>（拿去"问"的向量）、<strong>Key</strong>（被查的"标签"）、<strong>Value</strong>（携带的"内容"）。
当我们新增第 n+1 个 token 时，它要拿自己的 Query 去和"<strong>所有历史 token 的 Key</strong>"逐一比对，再按比对出的权重把对应的 Value 加权汇总。
关键在于：<strong>历史 token 的 K 和 V，只取决于它自己和它的位置，而因果掩码保证它"看不到"任何后来的 token</strong>——
所以无论后面再追加多少新 token，<strong>前面那些 K/V 一个数都不会变</strong>。既然不变，就可以<strong>算一次、存下来、永远复用</strong>。这就是 KV cache：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>KV cache 是精确的</b>：新增 token 不改变任何历史 K/V，故缓存值与"从头重算"逐位相等</div>
  <div class="cells"><span class="lab">第 n 步</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span></div>
  <div class="cells"><span class="lab">第 n+1 步</span><span class="cell dim">K1（不变）</span><span class="cell dim">K2（不变）</span><span class="cell dim">K3（不变）</span><span class="cell hl">+K4</span></div>
</div>
<p>看这两行——第 n+1 步相比第 n 步，<strong>只多出最右边一个高亮新格</strong>，左边那些全是"<strong>原封不动</strong>"的旧值。
所以 KV cache <strong>不是</strong>一种"用精度换速度"的近似优化，而是一个<strong>数学上完全等价</strong>的复用：缓存里的值，
和"每步都从头重算"得到的值<strong>逐位相等</strong>。它省掉的是<strong>重复劳动</strong>，不是精度。顺带一提，预测下一个词时只需要
<strong>最后一个位置</strong>的那一行输出，所以取 logits 时只取末位：</p>
<pre class="code">logits = <span class="fn">llama_get_logits_ith</span>(ctx, -1);  <span class="cm">// 只取最后一个位置, 返回 n_vocab 个分数</span></pre>
<div class="card spark">
  <div class="tag">💡 实战</div>
  把这件事算笔账就更清楚了：<strong>没有缓存</strong>时，生成第 n 个 token 要重新为前面 n-1 个 token 全算一遍 K/V，整段生成下来总计算量大致是 <strong>n 的平方</strong>级别；<strong>有了缓存</strong>，每一步只为<strong>新来的那一个</strong> token 算 K/V，历史部分直接取用，于是每步的<strong>新增计算基本是常数</strong>。举个直观的例子：聊到第 2000 个 token 时，没有缓存就得把前面近 2000 个 token 的 K/V 从头再算一遍，<strong>越聊越慢</strong>；有了缓存，第 2000 步和第 2 步<strong>需要新算的 K/V 一样多（都只有一个 token）</strong>。请注意：这只是把"算过的别再算"，<strong>结果一个数都没变</strong>——这再次印证它是"精确复用"而非"近似加速"。
</div>
<p>当然，<strong>KV cache 也不是白来的</strong>：它要为<strong>每一层、每个历史 token</strong> 各存一份 K 和一份 V，占用的显存/内存随
<strong>上下文长度线性增长</strong>。上下文开到几万 token 时，这块缓存会吃掉相当可观的一片显存——这也是"上下文窗口"为什么总有上限、
长上下文为什么格外吃硬件的原因之一（分配、写入与复用都在 <span class="mono">src/llama-kv-cache.cpp</span>）。所以工程上既要<strong>靠它省计算</strong>，
又得<strong>想办法压它的体积</strong>——下面深挖里的 GQA / MQA 正是干这个的，这是一对需要一直权衡的矛盾。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  这里还藏着一个反直觉的事实：<strong>decode 阶段，瓶颈往往不是算力，而是"搬数据"</strong>。每生成一个 token，都要把模型那几个 GB 的权重、连同越来越大的 KV cache，从显存/内存里<strong>完整读一遍</strong>，可真正的计算量却只对应"一个 token"。于是单条对话的生成速度，更多是被<strong>内存带宽</strong>卡住，而不是被乘法次数卡住——这正好解释了为什么<strong>量化</strong>（把权重压小）能立竿见影地提速：要搬的字节少了，每步自然更快。下一部分讲量化时，我们会回到这条线索。
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么只有 decoder，没有 encoder？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>最初的 Transformer 是 <strong>encoder-decoder</strong> 结构，为翻译这类"读完整句再生成"的任务设计：encoder 双向读懂源句，
    decoder 据此逐词生成译文。但"<strong>预测下一个词</strong>"这件事并不需要双向——你写字时本来就只能依赖已经写出的部分。</p>
    <p>GPT 类模型于是<strong>只保留 decoder</strong>，配上因果掩码做纯粹的"续写"，结构更简单、训练目标更统一（始终是"猜下一个 token"），
    特别适合生成式任务。llama.cpp 支持的开源大模型，几乎清一色是这种 <strong>decoder-only</strong> 架构。</p>
    <p>反过来，只做"理解"不做"生成"的任务（如文本分类、检索、判断两句话是否同义），更适合 <strong>encoder-only</strong>
    （如 BERT）那种<strong>双向</strong>结构——它可以同时看左右两边的全部上下文；而"边读边写"的对话生成，则是 decoder-only 的主场。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> MHA / GQA / MQA：KV cache 还能更省 <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>标准多头注意力（<strong>MHA</strong>）里，Query、Key、Value 都有同样多的"头"。但 KV cache 的大小正比于 <strong>Key/Value 头的数量</strong>，
    于是出现了省内存的变体：<strong>GQA</strong>（分组查询注意力）让<strong>多个 Query 头共享一组 K/V 头</strong>，
    K/V 头数（<span class="mono">n_head_kv</span>）少于 Q 头数（<span class="mono">n_head</span>，见 <span class="mono">src/llama-hparams.h</span>）。</p>
    <p><strong>MQA</strong> 是极端情形——所有 Query 头只共享<strong>一组</strong> K/V。头数越少，KV cache 越小、长上下文越省显存，代价是表达力略降。
    如今多数大模型用 GQA 来折中：既显著压缩缓存，又几乎不掉效果。</p>
    <p>为什么共享了还几乎不掉效果？直觉是：相邻的若干 Query 头往往在关注<strong>差不多的东西</strong>，让它们共用一组 K/V，
    损失的表达力很有限，省下的显存却相当可观——这是一笔很划算的买卖，也是它能被广泛采用的原因。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> logits 是什么？temperature 怎么作用 <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><span class="mono">logits</span> 是<strong>词表上每个 token 的原始分数</strong>，可正可负、加起来也不等于 1，<strong>还不是概率</strong>。
    要变成概率，得过一道 <strong>softmax</strong>（指数化再归一化）。</p>
    <p><strong>温度（temperature, T）</strong>正作用在 softmax 之前：把所有 logits 同时除以 T——T 大于 1 会把分布"<strong>摊平</strong>"
    （更随机、更有创意），T 小于 1 会把分布"<strong>拉尖</strong>"（更确定、更保守），T 趋近 0 就退化成"<strong>每次都选分数最高的那个</strong>"（贪心）。
    所以同一份 logits，调温度就能在"稳重"和"放飞"之间滑动——这部分课 03 提过，后面还会有专门一课展开。</p>
    <p>顺便记住一个对照：<strong>贪心</strong>每次都选分数最高的词，稳定但容易重复、显得呆板；<strong>带温度的采样</strong>引入随机性，
    更生动也更容易"跑偏"。生成质量的调参，很大程度上就是在这两端之间找平衡。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>decoder-only = <strong>词嵌入 + N 层（自注意力 + FFN）+ 末层归一化 + 投影到词表 logits</strong>。</li>
    <li>token 之间<strong>唯一</strong>互相交流的地方是<strong>自注意力</strong>；FFN / 归一化 / 投影都对每个 token 各算各的。</li>
    <li><strong>因果掩码</strong>让第 i 个 token 只能注意到位置 &lt;= i 的 token——这就是"自回归"的数学实现。</li>
    <li><strong>KV cache 是精确的、不是近似</strong>：因果掩码保证旧 token 的 K/V 不随新 token 改变，缓存值与从头重算<strong>逐位相等</strong>。</li>
    <li>每步只需<strong>最后一个位置</strong>的输出，故取 logits 用 <span class="mono">llama_get_logits_ith(ctx, -1)</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  <strong>因果掩码</strong>这一个约束，同时换来了三样东西：prompt 可以<strong>整段并行</strong>地算（prefill）、历史 K/V 可以<strong>安全缓存</strong>（KV cache）、
  以及"<strong>自回归</strong>"这种逐字生成的形式本身。llama.cpp 后面几乎所有推理优化，追根溯源都站在这一条"<strong>只能回头看</strong>"的规则之上。把这条规则吃透，后面 ggml 怎么建图、KV cache 怎么管理、长上下文怎么优化，你都能找到它的影子。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Lesson 03 showed how the loop turns, round after round. This lesson drills one level deeper: what a transformer block
actually computes, why <strong>autoregression</strong> works at all, and - most importantly - <strong>why the KV cache is exact</strong>
rather than a "good enough" approximation. Understand this layer and you see why that loop can be so cheap.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Autoregressive generation is like a <strong>word-chain game</strong>: each time you look only at <strong>what has been written so far</strong>,
  guess the next best word, write it down; then count that new word as part of "what's written" and guess again. The model is the same -
  it always predicts the <strong>next token</strong> from <strong>all tokens so far</strong>, writes one, looks back, writes another.
  And because <strong>each step also reads back the word it just wrote</strong>, the model stays coherent - that is the "auto" in
  "autoregressive": using its own output as the next step's input.
</div>

<h2>decoder-only: what a block computes</h2>
<p>As lesson 02 noted, llama-family models are <strong>decoder-only</strong>. Unpacked, the data flows bottom-up in a straight chain:
token ids are looked up into vectors, pushed through <strong>many identical blocks</strong>, then projected back to the vocabulary to score
every candidate word. Inside one block there are two sub-layers - <strong>self-attention</strong> and the <strong>feed-forward network (FFN)</strong>:</p>
<div class="vflow">
  <div class="step"><div class="num">1</div><div class="sc">
    <h4>Embedding</h4>
    <p>Look up each token id into a dense vector (a word embedding); all later math runs on these vectors.</p>
    <p class="mono">src/llama-graph.cpp · build_inp_embd</p>
  </div></div>
  <div class="step"><div class="num">2</div><div class="sc">
    <h4>Self-attention sub-layer (first half of each block, x N blocks)</h4>
    <p>RMSNorm -&gt; self-attention (with RoPE positions) -&gt; residual add. <strong>This is the only place tokens talk to each other.</strong></p>
    <p class="mono">ggml_rms_norm · ggml_rope · ggml_soft_max_ext</p>
  </div></div>
  <div class="step"><div class="num">3</div><div class="sc">
    <h4>FFN sub-layer (second half of the same block)</h4>
    <p>Another RMSNorm -&gt; feed-forward network (SwiGLU) -&gt; residual add. A per-token non-linear mix; positions do not interact here.</p>
  </div></div>
  <div class="step"><div class="num">4</div><div class="sc">
    <h4>Final norm</h4>
    <p>After all blocks, one more RMSNorm to stabilize the output scale.</p>
  </div></div>
  <div class="step"><div class="num">5</div><div class="sc">
    <h4>Output projection -&gt; logits</h4>
    <p>The output head (lm_head) projects the vector back to <strong>vocabulary size</strong>, giving a score (logit) per candidate token.</p>
    <p class="mono">read out with llama_get_logits_ith</p>
  </div></div>
</div>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Here is the insight that runs through the whole lesson: <strong>only the self-attention layer lets tokens "see each other"</strong>. FFN, normalization, and projection all work <strong>on each token independently</strong>; positions do not interact. So "how the model uses context" <strong>happens entirely in attention</strong> - which is why the causal mask and the KV cache both revolve around it. Don't overlook the two <strong>residual</strong> branches and the <strong>norms</strong> either: residuals let gradients flow cleanly through dozens of layers without fading, and norms keep each layer's inputs in a stable range; together they make stacking blocks <strong>very deep</strong> trainable in the first place.
</div>
<p>A word on why <strong>embeddings</strong> matter. They are not a plain "dictionary lookup into an integer"; each token maps to a point in a
high-dimensional space where <strong>semantically similar words land close together</strong> - the starting point of the model "understanding" language.
A raw embedding only carries "roughly what this word means", with no context; what makes it "read the whole sentence" is the dozens of blocks that follow.
<strong>Think of the stacked blocks as a refinement pipeline</strong>: shallow layers capture local collocations and syntax (what modifies what, phrase
boundaries), deeper layers compose more abstract semantics and long-range relations (coreference, logic, topic). Each layer <strong>injects more context</strong>
into a token's vector; by the top, the last position's vector has distilled every clue about "what to say next", which the projection turns into logits.</p>
<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  To make <strong>what attention does</strong> concrete: it is a kind of <strong>weighted retrieval</strong>. The current token holds a Query (a "what am I looking for" question), matches it against every historical token's Key (a "what am I" tag) - the better the match, the higher the attention weight - then takes a <strong>weighted sum</strong> of those tokens' Values (the "content I can offer") into its new representation. So "understanding context" becomes "which earlier words to lean on". This is also why attention is the only channel between tokens: only here does a token's output truly depend on <strong>other</strong> tokens.
</div>
<p>An easily missed point: attention itself <strong>does not know word order</strong> - it only does similarity-weighted summing, so shuffling the same
words and feeding them in gives the same result from pure attention. But language clearly cares about order ("dog bites man" vs "man bites dog").
<strong>Position information</strong> is added for exactly this: llama-family models commonly use <strong>RoPE (rotary position embedding)</strong>, which
does not simply add an "index vector" per position but <strong>rotates Query/Key by position</strong>, so two tokens' attention score naturally carries
"how far apart they are". That is why the self-attention sub-layer above is tagged "with RoPE".</p>
<p>And the final <strong>output projection</strong>: it multiplies the top-layer token vector by a big "<strong>vocab-size x hidden-dim</strong>" matrix
(lm_head) to get <strong>one score per vocabulary token</strong> - which is why the logits length always equals the vocabulary size. Many models also
<strong>tie</strong> this with the bottom embedding matrix (weight tying), saving parameters and keeping "how inputs are encoded" consistent with
"how outputs are scored".</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  A quick sense of <strong>scale</strong>: the parameters of a "7B" or "13B" model live mostly in these blocks - each layer's attention projections, the FFN's two or three big matrices, times dozens of layers, plus the large embedding and output projections, sum to billions of numbers. A few hyperparameters - <strong>number of layers (n_layer), hidden dim (n_embd), number of heads</strong> - largely decide how big, how strong, and how resource-hungry a model is; you will read them straight from the GGUF header later.
</div>
<p>Written as pseudocode, one block's forward pass is "two residual branches":</p>
<pre class="code"><span class="cm"># one block's forward: two residual branches</span>
<span class="kw">def</span> <span class="fn">layer</span>(x):                 <span class="cm"># x: [n_tokens, n_embd]</span>
    a = <span class="fn">attn</span>(<span class="fn">rms_norm</span>(x))     <span class="cm"># tokens talk to each other here</span>
    x = x + a                 <span class="cm"># residual</span>
    f = <span class="fn">ffn</span>(<span class="fn">rms_norm</span>(x))      <span class="cm"># per-token non-linear mix</span>
    <span class="kw">return</span> x + f              <span class="cm"># residual</span></pre>

<h2>The causal mask: you may only look back</h2>
<p>Since attention lets tokens reference one another, can token 5 peek at tokens 6 and 7 while writing? <strong>Never</strong> - during generation
they do not exist yet. So the decoder adds a <strong>causal mask</strong> to attention: token i may attend only to positions <strong>&lt;= i</strong>,
masking out every "future" position. Drawn as a grid, it is a <strong>lower triangle</strong>:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Causal mask</b>: token i may attend only to positions &lt;= i; lit = visible, dim = masked (after itself)</div>
  <div class="cells"><span class="lab">t1 sees</span><span class="cell hl">k1</span><span class="cell dim">k2</span><span class="cell dim">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t2 sees</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell dim">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t3 sees</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell hl">k3</span><span class="cell dim">k4</span></div>
  <div class="cells"><span class="lab">t4 sees</span><span class="cell hl">k1</span><span class="cell hl">k2</span><span class="cell hl">k3</span><span class="cell hl">k4</span></div>
</div>
<p>Each row is "who one token may look at": a lit cell means <strong>attendable</strong> (no later than itself), a dim cell means <strong>masked</strong>
(after itself). The implementation is direct: after computing attention scores, set every "future" score to <strong>-inf</strong>, then softmax turns those
weights into 0 - as if unseen. <strong>This is what "autoregression" looks like mathematically</strong> - each position's output depends only on
<strong>itself and what came before</strong>, never leaking the future.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  <strong>The causal mask also has a neat training use worth spelling out.</strong> During training we have the <strong>full sentence</strong> and can feed it in all at once; the causal mask guarantees position i sees only tokens <strong>&lt;= i</strong>, so "predict the next word from the prefix" holds <strong>at every position simultaneously</strong> - one forward pass yields a "next word" prediction at every position, compared in parallel against the real next words to compute the loss. This "<strong>feed the answers in too, but block the future with a mask</strong>" training is called teacher forcing, and it is extremely efficient. Tellingly, <strong>this is the very same mechanism as prefill at inference time</strong> - every prompt token is known, so it can be computed in parallel like training, except at inference we only care about the last position's output.
</div>
<p>Because training and inference share <strong>the same forward pass</strong>, llama.cpp has no "two codebases": whether prefill feeds in hundreds of
tokens at once or decode feeds just 1 new token, both run the <strong>same compute graph</strong>, differing only in "how many tokens in this batch, and
which positions to output". With that in mind, revisit lesson 03's seven-step flow and you see why a single <span class="mono">llama_decode</span> can
carry both rhythms.</p>
<p>In llama.cpp this is assembled into the compute graph by <span class="mono">build_attn</span> / <span class="mono">build_attn_mha</span>
in <span class="mono">src/llama-graph.cpp</span>, with the mask applied during softmax via <span class="mono">ggml_soft_max_ext</span>. Simplified:</p>
<pre class="code"><span class="cm">// attention scores + causal mask (cf. build_attn / build_attn_mha)</span>
kq  = <span class="fn">ggml_mul_mat</span>(ctx, k, q);            <span class="cm">// scores [n_kv, n_q]</span>
kq  = <span class="fn">ggml_soft_max_ext</span>(ctx, kq, mask,    <span class="cm">// mask: causal -inf on j&gt;i</span>
                        scale, max_bias);
kqv = <span class="fn">ggml_mul_mat</span>(ctx, v, kq);           <span class="cm">// weighted sum of values</span></pre>
<p>(Per this repo's source as of 2026-06-15; the real implementation lives in <span class="mono">llama-graph.cpp</span>; only the trunk is shown,
omitting scaling and multi-head details.)</p>

<h2>Autoregression + why the KV cache is exact</h2>
<p>Draw "predict the next from what we already have" as a loop and you get the autoregressive cycle:</p>
<div class="flow">
  <div class="node"><div class="nt">existing tokens</div><div class="nd">x1 … xn</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">forward</div><div class="nd">N blocks</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">last-pos logits</div><div class="nd">vocab scores</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">sample</div><div class="nd">pick a token</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">x(n+1)</div><div class="nd">feed back, repeat</div></div>
</div>
<p>Note the <strong>feed-back</strong> arrow at the end - the newly sampled token is appended to the sequence and becomes the next round's input.
This raises the lesson's core question: to write each next word, must we recompute attention over <strong>all previous words</strong>?
<strong>No</strong> - and the reason is hidden in the causal mask.</p>
<p>In attention each token computes three things: <strong>Query</strong> (the vector that "asks"), <strong>Key</strong> (the "tag" being matched),
and <strong>Value</strong> (the "content" carried). To add token n+1, it takes its Query, compares against <strong>every historical token's Key</strong>,
and sums their Values by the resulting weights. The crucial point: <strong>a historical token's K and V depend only on itself and its position,
and the causal mask guarantees it cannot see any later token</strong> - so no matter how many new tokens are appended afterwards,
<strong>those earlier K/V never change by a single number</strong>. Since they never change, they can be <strong>computed once, stored, and reused forever</strong>.
That is the KV cache:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>The KV cache is exact</b>: a new token changes no historical K/V, so cached values equal "recompute from scratch" bit for bit</div>
  <div class="cells"><span class="lab">step n</span><span class="cell">K1</span><span class="cell">K2</span><span class="cell">K3</span></div>
  <div class="cells"><span class="lab">step n+1</span><span class="cell dim">K1 (same)</span><span class="cell dim">K2 (same)</span><span class="cell dim">K3 (same)</span><span class="cell hl">+K4</span></div>
</div>
<p>Look at the two rows - step n+1 differs from step n by <strong>just one highlighted new cell on the right</strong>; everything on the left is
<strong>untouched</strong> old values. So the KV cache is <strong>not</strong> an "accuracy-for-speed" approximation; it is a <strong>mathematically exact</strong>
reuse: the cached values equal what "recompute every step from scratch" would produce, <strong>bit for bit</strong>. What it saves is
<strong>repeated work</strong>, not precision. And since predicting the next word needs only the <strong>last position's</strong> row of output, we read logits at the end:</p>
<pre class="code">logits = <span class="fn">llama_get_logits_ith</span>(ctx, -1);  <span class="cm">// last position only, returns n_vocab scores</span></pre>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  Putting numbers on it makes it clearer: <strong>without the cache</strong>, generating the nth token recomputes K/V for all previous n-1 tokens, so the whole generation is roughly <strong>n-squared</strong> in cost; <strong>with the cache</strong>, each step computes K/V for <strong>only the one new</strong> token and reuses history, so the per-step extra work is essentially <strong>constant</strong>. A concrete feel: by the 2000th token, no cache means recomputing K/V for nearly 2000 prior tokens every step - <strong>slower the longer you chat</strong>; with the cache, step 2000 and step 2 compute the <strong>same amount of new K/V (just one token's)</strong>. Note this only avoids recomputing what was already computed - <strong>not a single number changes</strong>, which again shows it is exact reuse, not approximate speedup.
</div>
<p>Of course, the <strong>KV cache is not free</strong>: it stores one K and one V for <strong>every layer and every historical token</strong>, so its footprint
grows <strong>linearly with context length</strong>. At tens of thousands of tokens it eats a sizeable chunk of memory - one reason "context windows" always
have a ceiling and long contexts are so hardware-hungry (allocation, writes, and reuse all live in <span class="mono">src/llama-kv-cache.cpp</span>). So
engineering must both <strong>save compute with it</strong> and <strong>shrink its size</strong> - GQA/MQA below do exactly that - a tension to keep balancing.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  There is also a counter-intuitive fact here: <strong>during decode the bottleneck is usually not compute but moving data</strong>. Each generated token must read the model's several GB of weights, plus the ever-growing KV cache, <strong>in full</strong> from memory, yet the actual math corresponds to just "one token". So a single conversation's speed is limited more by <strong>memory bandwidth</strong> than by multiply count - which is exactly why <strong>quantization</strong> (shrinking the weights) speeds things up immediately: fewer bytes to move, faster steps. We will return to this thread in the next part on quantization.
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why only a decoder, no encoder? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The original Transformer was <strong>encoder-decoder</strong>, designed for tasks like translation ("read the whole sentence, then generate"):
    the encoder reads the source bidirectionally, the decoder generates the translation word by word. But <strong>predicting the next word</strong>
    does not need bidirectionality - while writing you can only depend on what you have already written.</p>
    <p>GPT-style models therefore <strong>keep only the decoder</strong>, pair it with a causal mask, and do pure "continuation". The structure is
    simpler and the training objective is uniform (always "guess the next token"), which suits generation. Almost every open model llama.cpp
    supports is this <strong>decoder-only</strong> architecture.</p>
    <p>Conversely, tasks that only "understand" without generating (classification, retrieval, deciding if two sentences mean the same) suit
    <strong>encoder-only</strong> models (like BERT) with their <strong>bidirectional</strong> structure - able to see all context on both sides at once;
    whereas "read-and-write" conversational generation is decoder-only's home turf.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> MHA / GQA / MQA: shrinking the KV cache <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>In standard multi-head attention (<strong>MHA</strong>), Query, Key, and Value all have the same number of heads. But the KV cache size is
    proportional to the number of <strong>Key/Value heads</strong>, so memory-saving variants appeared: <strong>GQA</strong> (grouped-query attention)
    lets <strong>several Query heads share one group of K/V heads</strong>, with fewer K/V heads (<span class="mono">n_head_kv</span>) than Q heads
    (<span class="mono">n_head</span>, see <span class="mono">src/llama-hparams.h</span>).</p>
    <p><strong>MQA</strong> is the extreme - all Query heads share <strong>one</strong> set of K/V. Fewer heads means a smaller KV cache and cheaper long
    contexts, at a small cost in expressiveness. Most large models today use GQA as the compromise: much smaller cache, almost no quality loss.</p>
    <p>Why does sharing barely hurt? Intuitively, neighboring Query heads often attend to <strong>much the same thing</strong>, so having them share one set
    of K/V loses little expressiveness while saving a lot of memory - a great trade, and why it is so widely adopted.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What are logits? How does temperature act? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><span class="mono">logits</span> are the <strong>raw scores for each token in the vocabulary</strong> - they can be positive or negative and do not
    sum to 1; they are <strong>not yet probabilities</strong>. To become probabilities they pass through a <strong>softmax</strong> (exponentiate, then normalize).</p>
    <p><strong>Temperature (T)</strong> acts just before softmax: divide all logits by T - T &gt; 1 <strong>flattens</strong> the distribution
    (more random, more creative), T &lt; 1 <strong>sharpens</strong> it (more deterministic, more conservative), and T near 0 degenerates into
    <strong>always picking the highest score</strong> (greedy). So with the same logits, tuning temperature slides you between "steady" and "wild" -
    lesson 03 touched on this, and a later lesson covers it in full.</p>
    <p>Keep one contrast in mind: <strong>greedy</strong> always picks the highest-scoring word - stable but prone to repetition and blandness;
    <strong>temperature sampling</strong> adds randomness - livelier but more prone to going off the rails. Tuning generation quality is largely about
    balancing these two ends.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>decoder-only = <strong>embedding + N layers (self-attention + FFN) + final norm + projection to vocab logits</strong>.</li>
    <li>The <strong>only</strong> place tokens talk to each other is <strong>self-attention</strong>; FFN / norm / projection are per-token.</li>
    <li>The <strong>causal mask</strong> lets token i attend only to positions &lt;= i - that is the mathematical form of "autoregression".</li>
    <li>The <strong>KV cache is exact, not approximate</strong>: the causal mask keeps old tokens' K/V unchanged by new tokens, so cached values match a from-scratch recompute <strong>bit for bit</strong>.</li>
    <li>Each step needs only the <strong>last position's</strong> output, so read logits with <span class="mono">llama_get_logits_ith(ctx, -1)</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The single constraint of the <strong>causal mask</strong> buys three things at once: the prompt can be computed <strong>fully in parallel</strong> (prefill),
  historical K/V can be <strong>safely cached</strong> (KV cache), and the very form of <strong>autoregressive</strong> token-by-token generation.
  Nearly every inference optimization later in llama.cpp ultimately stands on this one rule: <strong>you may only look back</strong>.
</div>
""",
}

LESSON_05 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
ggml 是 llama.cpp 的计算引擎，而在它眼里<strong>一切数据都是"张量"</strong>——权重、激活值、KV cache，全是张量。
这一课把 <span class="mono">ggml_tensor</span> 这个结构体讲到"摸得着"：什么是 shape、什么是 stride、为什么 ggml 用"行优先"，
以及为什么"转置一个张量"几乎不花钱。看懂这一层，后面所有计算图和算子的示意图你都能读懂；可以说，张量是读 ggml 源码的"<strong>第一块积木</strong>"。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把张量想成一排<strong>储物柜阵列</strong>：<span class="mono">ne[]</span> 告诉你"每排几个柜、一共几排"（形状），
  <span class="mono">nb[]</span> 告诉你"从一个柜走到下一个、或跳到下一排，各要迈多少步"（步长，单位是字节），
  <span class="mono">data</span> 则是这排柜子的<strong>起点地址</strong>。知道这三样，你就能算出任意一个柜子在哪。
  而"<strong>柜子里装的是什么规格</strong>"（<span class="mono">type</span>）则决定每个格子占多大、怎么读。把"形状（柜子怎么排）"和"数据（柜子里的东西）"
  分开记——这正是后面所有省内存、零拷贝把戏的根。
</div>

<h2>张量 = 形状 + 类型 + 一块连续内存</h2>
<p>一个张量说白了就是"<strong>一块连续内存 + 一份怎么解释它的说明书</strong>"。说明书里有：数据<strong>类型</strong>
（<span class="mono">type</span>，比如 F32、F16、Q4_0）、每个维度有<strong>多少元素</strong>（<span class="mono">ne[]</span>，ggml 最多 4 维），
以及一个指向那块内存的指针 <span class="mono">data</span>。先看"形状"长什么样：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>形状 ne</b>：一个 ne=[4,3] 的张量 - ne[0]=4 是"一排 4 个"（最内维），ne[1]=3 是"共 3 排"</div>
  <div class="cells"><span class="lab">第 0 排</span><span class="cell">a00</span><span class="cell">a01</span><span class="cell">a02</span><span class="cell">a03</span></div>
  <div class="cells"><span class="lab">第 1 排</span><span class="cell">a10</span><span class="cell">a11</span><span class="cell">a12</span><span class="cell">a13</span></div>
  <div class="cells"><span class="lab">第 2 排</span><span class="cell">a20</span><span class="cell">a21</span><span class="cell">a22</span><span class="cell">a23</span></div>
</div>
<p>顺便厘清"几维"这件事：<strong>0 维</strong>是一个标量（一个数），<strong>1 维</strong>是一个向量（一串数，比如一个词向量），
<strong>2 维</strong>是一个矩阵（比如一层的权重），<strong>3 维、4 维</strong>则常见于"带批次、带多头"的中间结果。ggml 看 <span class="mono">ne[]</span>
里实际大于 1 的维数来判断张量是几维；没用到的高维就填 1。所以当你看到 <span class="mono">ne=[4,3,1,1]</span> 时，它其实就是个 4×3 的二维张量。</p>
<p>注意 ggml 的约定：<span class="mono">ne[0]</span> 是<strong>最内层、变化最快</strong>的维度（在内存里挨着摆），<span class="mono">ne[1]</span> 是"下一排"，
以此类推。这和很多人习惯的"行 × 列"写法正好<strong>相反</strong>——后面有专门的深挖讲这个坑。把 <span class="mono">ggml_tensor</span>
的核心字段摊开看，就这么几样：</p>
<pre class="code"><span class="kw">struct</span> <span class="fn">ggml_tensor</span> {
    <span class="kw">enum</span> ggml_type   type;           <span class="cm">// F32 / F16 / Q4_0 ...</span>
    int64_t          ne[4];          <span class="cm">// #elements per dim (ne[0] = innermost)</span>
    size_t           nb[4];          <span class="cm">// byte strides</span>
    <span class="kw">enum</span> ggml_op     op;             <span class="cm">// how it was produced (graph node)</span>
    <span class="kw">struct</span> ggml_tensor * src[...];    <span class="cm">// inputs (back-pointers)</span>
    <span class="kw">struct</span> ggml_tensor * view_src;     <span class="cm">// set if this tensor is a view</span>
    <span class="kw">void</span> *           data;           <span class="cm">// the actual bytes</span>
    <span class="kw">char</span>             name[...];
};  <span class="cm">// 简化自 ggml/include/ggml.h, GGML_MAX_DIMS = 4</span></pre>
<p>这里有两组字段值得记住。<span class="mono">ne</span> / <span class="mono">nb</span> / <span class="mono">type</span> / <span class="mono">data</span>
描述"<strong>这块数据长什么样、在哪</strong>"；而 <span class="mono">op</span> / <span class="mono">src</span> 描述"<strong>它是怎么算出来的</strong>"——
<span class="mono">op</span> 是产生它的算子（比如矩阵乘），<span class="mono">src</span> 是指向输入张量的反向指针。后面这组正是 ggml
"<strong>先建计算图、再执行</strong>"的关键：每个张量都记得自己的来历，把它们顺着 <span class="mono">src</span> 串起来，就是一张计算图
（课 03 提过、第三部分会展开）。另外别忽略 <span class="mono">type</span>：同样 1000 个元素，F32 占 4000 字节、Q4_0 只占 500 多字节——
<strong>类型直接决定了这块内存有多大</strong>，这也是量化能省显存的根。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  抽象归抽象，张量到底用来装什么？在 llama.cpp 里，<strong>模型的每个权重矩阵</strong>是张量（比如词嵌入表是一个<span class="mono">[n_embd, n_vocab]</span> 的大张量、每层的注意力与 FFN 权重也都是张量），<strong>前向过程中流动的激活值</strong>是张量，连 <strong>KV cache</strong> 里存的那些 K 和 V 也是张量。可以说，一次推理从头到尾，就是<strong>一堆张量按计算图被算来算去</strong>。正因为如此，把"张量"这个抽象设计得又轻又灵活，对整个引擎的效率至关重要。
</div>
<p>再说说 <span class="mono">type</span>。ggml 支持一长串数据类型：全精度的 <span class="mono">F32</span>、半精度的 <span class="mono">F16</span> /
<span class="mono">BF16</span>，以及一大家子<strong>量化类型</strong>（<span class="mono">Q4_0</span>、<span class="mono">Q8_0</span>、
<span class="mono">Q4_K</span> 等等，下一课专门讲）。同一个张量结构、同一套 ne/nb 逻辑，能装下这么多种类型，靠的就是用 <span class="mono">type</span>
这一个字段统一描述"每个元素（或每块）多少字节、怎么解释"。这种"<strong>结构不变、类型可换</strong>"的设计，让量化能<strong>无缝接进</strong>
已有的张量与算子体系，而不必为每种精度各写一套。也正因如此，换个量化等级（比如从 <span class="mono">Q4_0</span> 换到 <span class="mono">Q5_K</span>）
对上层代码几乎是透明的——变的只是 <span class="mono">type</span> 和每块的字节数，ne/nb 的那套逻辑原封不动。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  张量从哪来？在 ggml 里，你先开一个 <span class="mono">ggml_context</span>（一个内存池），再用 <span class="mono">ggml_new_tensor_2d</span>这类函数在池子里"<strong>登记</strong>"一个张量——它会按 type 和 ne 算好需要多少字节、把 nb 填好。值得注意的是：<strong>建张量时通常并不立刻搬运那一大块数据</strong>，很多时候只是先把"形状说明书"建好（这正配合了 ggml 先建图、后执行的风格），真正的内存分配与计算留到后面统一来做。第三部分会专门讲 <span class="mono">ggml_context</span> 与这套"先描述、后执行"的内存管理。
</div>
<p>有人可能会问：装个多维数组，直接用 C++ 的 <span class="mono">std::vector</span> 或裸数组不就行了，何必单独造一个
<span class="mono">ggml_tensor</span>？因为 ggml 要的远不止"存一堆数"：它要能<strong>把运算描述成图</strong>（靠 op/src）、要能
<strong>让数据躺在不同后端的内存上</strong>（靠 buffer）、要能<strong>统一容纳量化等多种类型</strong>（靠 type 和按块的 nb）、还要能
<strong>零拷贝地变形</strong>（靠 ne/nb 与 view_src）。这些需求叠在一起，才有了这个看似简单、其实精心设计的结构体——它是整个引擎的"原子"。</p>

<h2>行优先与 stride：nb[] 是怎么算的</h2>
<p>内存其实是<strong>一维</strong>的（一长条字节），可张量是多维的，把多维"压平"到一维靠的就是 <strong>stride（步长）</strong>。
ggml 用 <span class="mono">nb[i]</span> 记录"<strong>第 i 维每加 1，要在内存里跳过多少字节</strong>"。ggml 是<strong>行优先</strong>的：
<span class="mono">ne[0]</span> 那一维在内存里连续摆放、步长最小。看一个 ne=[3,2]（每排 3 个、共 2 排）的 F32 张量是怎么躺在内存里的：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>行优先内存布局</b>：在内存里是一条连续字节流，先摆满第 0 排，紧接着第 1 排（数字是字节偏移）</div>
  <div class="cells"><span class="lab">元素</span><span class="cell">a00</span><span class="cell">a01</span><span class="cell">a02</span><span class="cell hl">a10</span><span class="cell">a11</span><span class="cell">a12</span></div>
  <div class="cells"><span class="lab">偏移</span><span class="cell dim">0</span><span class="cell dim">4</span><span class="cell dim">8</span><span class="cell dim">12</span><span class="cell dim">16</span><span class="cell dim">20</span></div>
</div>
<p>看这条字节流就懂了：同一排里相邻元素隔 <strong>4 字节</strong>（一个 F32），所以 <span class="mono">nb[0]=4</span>；从第 0 排跳到第 1 排
（高亮那个 a10）要跨过<strong>整整一排 3 个元素 = 12 字节</strong>，所以 <span class="mono">nb[1]=12</span>。这套规则写成公式，就藏在
<span class="mono">ggml.h</span> 的注释里：<span class="mono">nb[0]=ggml_type_size(type)</span>（一个元素多少字节）、
<span class="mono">nb[1]=nb[0]*(ne[0]/ggml_blck_size(type))</span>（跨一整排的字节，普通类型就是"每元素字节 × 一排元素数"）、
再往上 <span class="mono">nb[i]=nb[i-1]*ne[i-1]</span>（严格说公式里还有一个"对齐填充"项，连续、无填充的张量这一项为 0，这里略去）。于是给任意一个多维下标，把它和 <span class="mono">nb</span> 点乘一下，就得到字节偏移：</p>
<div class="flow">
  <div class="node"><div class="nt">多维下标</div><div class="nd">(i0, i1, i2, i3)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">点乘步长</div><div class="nd">Σ i_k × nb[k]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">字节偏移</div><div class="nd">offset</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">取到元素</div><div class="nd">data + offset</div></div>
</div>
<pre class="code"><span class="cm"># 多维下标 (i0, i1, i2, i3) -> 内存里的字节偏移</span>
offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
ptr    = (<span class="kw">char</span>*)tensor-&gt;data + offset   <span class="cm"># 就是这个元素的地址</span></pre>
<p>为什么是 <strong>4 维</strong>（<span class="mono">GGML_MAX_DIMS = 4</span>）？因为推理里的张量基本不超过 4 维——典型的就像
"批次 × 序列 × 头数 × 每头维度"这种组合，4 维足够覆盖；用一个<strong>定长</strong>的小数组存 ne/nb，既简单又快，不必动态分配。
还有一个容易忽略的点：<span class="mono">data</span> 指向的内存<strong>不一定在 CPU 上</strong>——它可能在 GPU 显存里，张量另有一个
<span class="mono">buffer</span> 字段记录"这块数据归哪个后端管"，这正好呼应课 07 要讲的多后端（同一个张量结构，数据可以躺在不同硬件上）。</p>
<p>把前面的公式用一个具体例子走一遍，印象会更深。还用上面那张内存图里的 <span class="mono">ne=[3,2]</span> 的 F32 张量：每个元素 4 字节，于是
<span class="mono">nb[0]=4</span>；跨一整排要越过 3 个元素，于是 <span class="mono">nb[1]=4×3=12</span>。想取第 1 排第 2 个元素（下标
<span class="mono">i1=1, i0=2</span>，从 0 数起，也就是图里的 a12），它的字节偏移就是 <span class="mono">1×nb[1] + 2×nb[0] = 12 + 8 = 20</span>——
正好是内存图里 a12 底下标的那个 20。整块张量一共 <span class="mono">3×2×4 = 24</span> 字节。你看，<strong>只要有 ne、nb 和 data，任意一个元素的地址
都能一步算出来</strong>，这就是 stride 的全部威力；至于 <span class="mono">ne=[4,3]</span> 之类的别的形状，留给本课末尾的思考题自己算一遍。</p>
<p>反过来，知道了字节布局，你也就明白了<strong>为什么"按行遍历"通常比"按列遍历"快</strong>：顺着 <span class="mono">ne[0]</span>（行内）走，
访问的是内存里<strong>连续相邻</strong>的字节，对 CPU 缓存最友好；而按高维跳着走，每次都跨一大步，更容易频繁缺失缓存、拖慢速度。
很多算子实现都刻意顺着连续维来安排循环，正是这个道理——<strong>布局决定性能</strong>，这条直觉在后面看内核实现时还会反复用到。</p>

<h2>view / 转置为什么不拷贝数据</h2>
<p>现在来看 ggml 一个非常"省"的设计。既然"<strong>形状（ne/nb）</strong>"和"<strong>数据（data）</strong>"是分开存的，那么很多"改变形状"
的操作，<strong>根本不用动数据</strong>，只要改几个数字。最典型的就是<strong>转置</strong>：把一个 [行, 列] 的矩阵转成 [列, 行]，ggml 只是
<strong>交换 ne[0] 与 ne[1]、同时交换 nb[0] 与 nb[1]</strong>，<span class="mono">data</span> 一个字节都不搬：</p>
<div class="cols">
  <div class="col"><h4>原张量</h4><p><span class="mono">ne=[3,2]</span>，<span class="mono">nb=[4,12]</span><br>data -&gt; 一块真实内存</p></div>
  <div class="col"><h4>转置后（一个 view）</h4><p><span class="mono">ne=[2,3]</span>，<span class="mono">nb=[12,4]</span><br>view_src -&gt; 指回原张量，data 不变</p></div>
</div>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  像 reshape、转置、切片、广播这类操作，ggml 大多用<strong>视图（view）</strong>实现：新张量复用<strong>同一块 data</strong>，只是带上不同的<span class="mono">ne</span>/<span class="mono">nb</span> 和偏移，并用 <span class="mono">view_src</span> 记住"我是谁的视图"。好处显而易见：<strong>零拷贝、省内存、还快</strong>。代价是：视图常常变得<strong>不连续</strong>（<span class="mono">ggml_is_contiguous</span> 为假）——比如转置之后，沿 <span class="mono">ne[0]</span>方向走，在内存里就不再是挨着的了。有些算子要求输入必须连续，这时要先用 <span class="mono">ggml_cont</span> 把它"压实"成一块新的连续内存（注意：<strong>这一步才真的发生拷贝</strong>）。所以你会在 ggml 代码里看到不少 <span class="mono">ggml_cont</span> 的调用，它们就是在"连续性"和"零拷贝"之间做权衡。
</div>
<p>顺带提一个常见操作：<strong>广播（broadcast）</strong>。当两个形状不完全相同的张量要做逐元素运算（比如给每一行都加上同一个偏置向量），
ggml 允许某些维度上"<strong>一个元素当很多元素用</strong>"——靠的还是 stride 的小把戏：把那一维的 <span class="mono">nb</span> 设成 0，下标怎么变、
地址都不动，于是同一个值被反复读取，看起来就像"复制"了一遍，实际上<strong>一个字节都没多占</strong>。这又是一次"改 nb 而不搬 data"的典型，
和 view、转置一脉相承。</p>
<p>一个最常见的 reshape 例子：把形状 <span class="mono">[n_embd, n_tokens]</span> 的激活，按多头注意力的需要"摊"成
<span class="mono">[head_dim, n_head, n_tokens]</span>——元素总数没变（<span class="mono">n_embd = head_dim × n_head</span>），数据也没搬，
只是<strong>重新解释了 ne/nb</strong>，就把"一个大向量"看成了"若干个头各自的小向量"。课 04 说的多头注意力里，大量这种"同一块数据、换个形状看"的操作，
靠的全是视图，几乎不产生额外拷贝。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  再比如<strong>切片</strong>：想从一大块张量里取出"第 k 层"或"第 h 个注意力头"那一小块，ggml 通常也不复制，而是<strong>算好一个起始偏移</strong>、配上裁剪过的 ne/nb，返回一个指回原数据的视图。所以在 ggml 代码里你会发现，"<strong>取一部分</strong>"和"<strong>换个形状看</strong>"在底层往往是同一种廉价操作——这套以视图为中心的玩法，是读懂后面计算图代码的一把钥匙。
</div>
<p>顺便说清"连续"到底指什么：一个张量<strong>连续</strong>，意思是它的元素在内存里就是<strong>紧挨着、按 ne[0]、ne[1]… 顺序一个不落地排</strong>的
（也就是 nb 严格按前面的公式递推）。转置、某些切片会打破这种整齐：元素还是那些元素，但"走的顺序"和"内存摆放"对不上了，于是
<span class="mono">ggml_is_contiguous</span> 返回假。多数算子能直接吃连续张量；遇到必须连续的场合，<span class="mono">ggml_cont</span>
会按当前形状把数据重新誊抄成一块整齐的新内存。记住这条，你调试形状相关的问题时会少踩很多坑。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  最后回头看 <span class="mono">op</span> 和 <span class="mono">src</span> 这两个字段，它们让张量有了"<strong>双重身份</strong>"：既是<strong>一块数据</strong>，又是<strong>计算图里的一个节点</strong>。当你写下 <span class="mono">c = ggml_mul_mat(ctx, a, b)</span>，ggml 并不立刻算矩阵乘，而是新建一个张量<span class="mono">c</span>，把它的 <span class="mono">op</span> 记成"矩阵乘"、<span class="mono">src[0]/src[1]</span> 指向 <span class="mono">a</span> 和<span class="mono">b</span>。于是整张计算图，其实就是<strong>张量们靠 src 互相牵着手连成的一张网</strong>；等图建完，再交给后端按这张网的顺序逐个算过去。理解了张量这层"图节点"身份，你就握住了第三部分 ggml 引擎的钥匙。
</div>
<p>结构体里还有个不起眼但很实用的字段 <span class="mono">name</span>：每个张量可以带一个名字。这在<strong>调试</strong>时很有用（打印计算图时一眼认出
"这是哪个权重"），而且 GGUF 文件里的每个权重张量本来就是<strong>带名字存的</strong>（像 <span class="mono">blk.0.attn_q.weight</span> 这种），
加载时按名字对号入座。所以"名字"不只是注释，它是模型权重和代码之间的<strong>索引</strong>。你在 GGUF 工具或调试日志里看到的那一串张量名，正是来自这个字段。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  给个实战提示：在 ggml 里写代码，<strong>最常见的报错就是形状对不上</strong>——矩阵乘要求左右两个张量在相乘的那一维上元素数相等，转置、reshape 之后维度顺序变了，很容易把该对齐的维弄错。养成<strong>随手在脑子里写出每个张量 ne</strong> 的习惯（甚至用 <span class="mono">name</span>标注、打印出来核对），能帮你省下大量调试时间。这也是为什么这一课要把 ne/nb 讲得这么细：<strong>形状是 ggml 编程的"语法"</strong>，语法错了，后面什么都跑不起来。
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> ggml 的维度顺序为什么和 PyTorch 相反？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>在 numpy / PyTorch 里，习惯把<strong>最后一维</strong>当作内存里连续的维度（行优先、C-order）：一个形状 <span class="mono">[batch, seq, dim]</span>
    的张量，<span class="mono">dim</span> 是连续的。ggml 反过来：它把<strong>连续的那一维放在 ne[0]</strong>（最前面），所以同一个张量在 ggml 里写成
    <span class="mono">ne = [dim, seq, batch]</span>——<strong>维度顺序整个反过来</strong>。</p>
    <p>这不是谁对谁错，只是约定不同；但读 ggml 代码、看张量形状时一定要在脑子里切换过来，否则很容易把行当成列、把 batch 看成 dim。
    一个好记的口诀：<strong>ggml 的 ne[0] 永远是"最贴着内存、变化最快"的那一维</strong>。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> nb 公式里为什么有个 /blck_size？量化类型的坑 <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>普通类型（F32、F16）里，<span class="mono">nb[0]</span> 就是一个元素的字节数。但<strong>量化类型</strong>（如 Q4_0）不是"一个元素一个值"，
    而是把<strong>一整块</strong>（如 32 个权重）打包压成定长字节，单个权重<strong>没法独立寻址</strong>。</p>
    <p>所以 ggml 用 <span class="mono">ggml_blck_size(type)</span>（一块里有几个元素）和 <span class="mono">ggml_type_size(type)</span>（一块多少字节）
    来描述。<span class="mono">nb[1]</span> 公式里那个 <span class="mono">ne[0] / ggml_blck_size(type)</span>，意思就是"这一排里有多少<strong>块</strong>"。
    明白这点，你就懂了为什么量化张量不能像普通数组那样随便按单元素下标去取——得<strong>按块解量化</strong>（第三部分的量化格式课会细讲）。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 怎么算一个张量占多少内存？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>直接用 <span class="mono">ggml_nbytes(tensor)</span>。直觉上，一个<strong>连续</strong>张量的字节数约等于"<strong>最高维元素数 × 最高维步长</strong>"
    （<span class="mono">ne[k] * nb[k]</span> 取最高维），也就是把各维元素数乘起来、再乘上每元素（或每块）的字节数。</p>
    <p>这在估算<strong>显存占用</strong>时很有用：模型权重占多少、KV cache 占多少，本质上都是这么一类张量的字节数加总。非连续张量、量化张量的算法略有不同，
    但 <span class="mono">ggml_nbytes</span> 已经替你把这些情况都处理好了，直接调用即可。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>张量 = <strong>type</strong>（类型）+ <strong>ne[4]</strong>（形状）+ <strong>nb[4]</strong>（字节步长）+ <strong>data</strong>（内存）+ <strong>op / src</strong>（它在计算图里怎么来的）。</li>
    <li>ggml 是<strong>行优先</strong>，<strong>ne[0] 是最内 / 连续维</strong>（步长最小），维度顺序和 numpy / PyTorch <strong>相反</strong>。</li>
    <li>多维下标 -&gt; 内存偏移：<span class="mono">offset = Σ i_k × nb[k]</span>。</li>
    <li>view / 转置 / reshape <strong>零拷贝</strong>：只改 ne/nb、复用 data、用 <span class="mono">view_src</span> 记来源；代价是可能<strong>非连续</strong>，必要时 <span class="mono">ggml_cont</span> 压实（这才真拷贝）。</li>
    <li>量化类型按<strong>块</strong>存储，<span class="mono">nb</span> 公式里的 /blck_size 就是这个原因。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"<strong>形状</strong>"（ne/nb）和"<strong>数据</strong>"（data）彻底分开存——就这一个设计，让转置、切片、reshape、广播统统变成"<strong>改几个数字</strong>"
  而非"搬一整块内存"，也让<strong>同一块权重</strong>能被计算图以不同视角反复使用。ggml 的高效，很大一部分就藏在这个朴素的拆分里。下一课讲量化，你会看到这套 ne/nb/type 的设计，如何让"把权重压成 4 bit"这件事，
  能不动声色地接进同一套张量与算子里。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
ggml is llama.cpp's compute engine, and in its eyes <strong>all data is "tensors"</strong> - weights, activations, the KV cache, all tensors.
This lesson makes the <span class="mono">ggml_tensor</span> struct tangible: what shape is, what stride is, why ggml is "row-major",
and why "transposing a tensor" costs almost nothing. Understand this layer and you can read every compute-graph and operator diagram that follows;
tensors are, you could say, the <strong>first building block</strong> for reading ggml source.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of a tensor as a grid of <strong>lockers</strong>: <span class="mono">ne[]</span> tells you "how many lockers per row, how many rows" (shape),
  <span class="mono">nb[]</span> tells you "how many steps to walk from one locker to the next, or to jump to the next row" (strides, in bytes),
  and <span class="mono">data</span> is the <strong>starting address</strong> of the grid. Know those three and you can locate any locker.
  And "<strong>what spec the lockers hold</strong>" (<span class="mono">type</span>) decides how big each cell is and how to read it. Keep "shape (how lockers
  are arranged)" and "data (what's inside)" separate - that is the root of every memory-saving, zero-copy trick later.
</div>

<h2>A tensor = shape + type + one contiguous block of memory</h2>
<p>A tensor is really "<strong>one contiguous block of memory + a manual for how to read it</strong>". The manual has: the data <strong>type</strong>
(<span class="mono">type</span>, e.g. F32, F16, Q4_0), how many elements per dimension (<span class="mono">ne[]</span>, up to 4 dims in ggml),
and a pointer <span class="mono">data</span> to that memory. First, what "shape" looks like:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>shape ne</b>: a tensor with ne=[4,3] - ne[0]=4 is "4 per row" (innermost dim), ne[1]=3 is "3 rows"</div>
  <div class="cells"><span class="lab">row 0</span><span class="cell">a00</span><span class="cell">a01</span><span class="cell">a02</span><span class="cell">a03</span></div>
  <div class="cells"><span class="lab">row 1</span><span class="cell">a10</span><span class="cell">a11</span><span class="cell">a12</span><span class="cell">a13</span></div>
  <div class="cells"><span class="lab">row 2</span><span class="cell">a20</span><span class="cell">a21</span><span class="cell">a22</span><span class="cell">a23</span></div>
</div>
<p>A quick clarification of "how many dims": <strong>0-D</strong> is a scalar (one number), <strong>1-D</strong> a vector (e.g. one embedding),
<strong>2-D</strong> a matrix (e.g. a layer's weights), and <strong>3-D / 4-D</strong> show up in "batched, multi-head" intermediates. ggml infers a tensor's
rank from how many of <span class="mono">ne[]</span> are greater than 1; unused higher dims are filled with 1. So <span class="mono">ne=[4,3,1,1]</span> is really
just a 4x3 two-dimensional tensor.</p>
<p>Note ggml's convention: <span class="mono">ne[0]</span> is the <strong>innermost, fastest-changing</strong> dimension (laid out contiguously in memory),
<span class="mono">ne[1]</span> is "the next row", and so on. This is the <strong>opposite</strong> of the "rows x cols" ordering many people are used to -
a dedicated deep-dive below covers this trap. Here are the core fields of <span class="mono">ggml_tensor</span>:</p>
<pre class="code"><span class="kw">struct</span> <span class="fn">ggml_tensor</span> {
    <span class="kw">enum</span> ggml_type   type;           <span class="cm">// F32 / F16 / Q4_0 ...</span>
    int64_t          ne[4];          <span class="cm">// #elements per dim (ne[0] = innermost)</span>
    size_t           nb[4];          <span class="cm">// byte strides</span>
    <span class="kw">enum</span> ggml_op     op;             <span class="cm">// how it was produced (graph node)</span>
    <span class="kw">struct</span> ggml_tensor * src[...];    <span class="cm">// inputs (back-pointers)</span>
    <span class="kw">struct</span> ggml_tensor * view_src;     <span class="cm">// set if this tensor is a view</span>
    <span class="kw">void</span> *           data;           <span class="cm">// the actual bytes</span>
    <span class="kw">char</span>             name[...];
};  <span class="cm">// simplified from ggml/include/ggml.h, GGML_MAX_DIMS = 4</span></pre>
<p>Two groups of fields are worth remembering. <span class="mono">ne</span> / <span class="mono">nb</span> / <span class="mono">type</span> / <span class="mono">data</span>
describe "<strong>what this data looks like and where it is</strong>"; while <span class="mono">op</span> / <span class="mono">src</span> describe
"<strong>how it was computed</strong>" - <span class="mono">op</span> is the operator that produced it (e.g. matmul), <span class="mono">src</span> are
back-pointers to input tensors. That second group is the key to ggml's "<strong>build the graph first, then execute</strong>": every tensor remembers its
origin, and stringing them together via <span class="mono">src</span> is a compute graph (mentioned in lesson 03, expanded in Part 3). Don't overlook
<span class="mono">type</span> either: the same 1000 elements take 4000 bytes as F32 but only ~500 as Q4_0 - <strong>the type alone decides how big this
memory is</strong>, which is the root of how quantization saves memory.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Abstractions aside, what do tensors actually hold? In llama.cpp, <strong>every weight matrix</strong> is a tensor (the embedding table is a big <span class="mono">[n_embd, n_vocab]</span> tensor; each layer's attention and FFN weights are tensors too), the <strong>activations flowing through the forward pass</strong> are tensors, and even the K and V stored in the <strong>KV cache</strong> are tensors. An entire inference, start to finish, is just <strong>a pile of tensors being computed along a graph</strong>. That is why making the "tensor" abstraction light and flexible is so crucial to the whole engine's efficiency.
</div>
<p>About <span class="mono">type</span>: ggml supports a long list of data types - full-precision <span class="mono">F32</span>, half-precision
<span class="mono">F16</span> / <span class="mono">BF16</span>, and a whole family of <strong>quantized types</strong> (<span class="mono">Q4_0</span>,
<span class="mono">Q8_0</span>, <span class="mono">Q4_K</span>, etc. - next lesson). The same tensor struct and the same ne/nb logic hold all of them, because one
field, <span class="mono">type</span>, uniformly describes "how many bytes per element (or per block), and how to interpret them". This "<strong>same structure,
swappable type</strong>" design lets quantization <strong>plug seamlessly</strong> into the existing tensor and operator machinery without rewriting it per
precision. Because of this, switching quantization levels (say <span class="mono">Q4_0</span> to <span class="mono">Q5_K</span>) is nearly transparent to higher-level
code - only the <span class="mono">type</span> and per-block byte count change; the ne/nb logic is untouched.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Where do tensors come from? In ggml you first open a <span class="mono">ggml_context</span> (a memory pool), then use functions like <span class="mono">ggml_new_tensor_2d</span> to "<strong>register</strong>" a tensor in the pool - it computes the needed bytes from type and ne, and fills in nb. Notably, <strong>creating a tensor usually does not immediately move that big block of data</strong>; often it just sets up the "shape manual" (matching ggml's build-graph-then-execute style), leaving the real allocation and computation for a unified later pass. Part 3 covers <span class="mono">ggml_context</span> and this "describe first, execute later" memory management.
</div>
<p>You might ask: to hold a multi-dim array, why not just use C++'s <span class="mono">std::vector</span> or a raw array - why a dedicated
<span class="mono">ggml_tensor</span>? Because ggml needs far more than "store some numbers": it must <strong>describe computation as a graph</strong> (via op/src),
<strong>let data live in different backends' memory</strong> (via buffer), <strong>uniformly hold many types including quantized ones</strong> (via type and
per-block nb), and <strong>reshape with zero copies</strong> (via ne/nb and view_src). All these needs together produced this seemingly simple but carefully
designed struct - the engine's "atom".</p>

<h2>Row-major and stride: how nb[] is computed</h2>
<p>Memory is actually <strong>one-dimensional</strong> (a long strip of bytes), but tensors are multi-dimensional; what flattens many dims into one is
<strong>stride</strong>. ggml uses <span class="mono">nb[i]</span> to record "<strong>how many bytes to skip in memory when dim i increases by 1</strong>".
ggml is <strong>row-major</strong>: the <span class="mono">ne[0]</span> dimension is laid out contiguously with the smallest stride. Here is how an
ne=[3,2] (3 per row, 2 rows) F32 tensor lies in memory:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Row-major layout</b>: one contiguous byte stream - fill row 0, then row 1 (numbers are byte offsets)</div>
  <div class="cells"><span class="lab">elem</span><span class="cell">a00</span><span class="cell">a01</span><span class="cell">a02</span><span class="cell hl">a10</span><span class="cell">a11</span><span class="cell">a12</span></div>
  <div class="cells"><span class="lab">offset</span><span class="cell dim">0</span><span class="cell dim">4</span><span class="cell dim">8</span><span class="cell dim">12</span><span class="cell dim">16</span><span class="cell dim">20</span></div>
</div>
<p>This byte stream makes it click: adjacent elements in a row are <strong>4 bytes</strong> apart (one F32), so <span class="mono">nb[0]=4</span>; jumping from
row 0 to row 1 (the highlighted a10) skips <strong>a whole row of 3 elements = 12 bytes</strong>, so <span class="mono">nb[1]=12</span>. As a formula -
straight from the comments in <span class="mono">ggml.h</span>: <span class="mono">nb[0]=ggml_type_size(type)</span> (bytes per element),
<span class="mono">nb[1]=nb[0]*(ne[0]/ggml_blck_size(type))</span> (bytes to cross one row; for plain types just "bytes-per-element x elements-per-row"),
and above that <span class="mono">nb[i]=nb[i-1]*ne[i-1]</span> (strictly there is also an alignment "padding" term, which is 0 for contiguous unpadded tensors and is omitted here). So given any multi-dim index, dot it with <span class="mono">nb</span> to get the byte offset:</p>
<div class="flow">
  <div class="node"><div class="nt">multi-dim index</div><div class="nd">(i0, i1, i2, i3)</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">dot with strides</div><div class="nd">sum i_k * nb[k]</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">byte offset</div><div class="nd">offset</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">reach element</div><div class="nd">data + offset</div></div>
</div>
<pre class="code"><span class="cm"># multi-dim index (i0, i1, i2, i3) -> byte offset in memory</span>
offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
ptr    = (<span class="kw">char</span>*)tensor-&gt;data + offset   <span class="cm"># the address of this element</span></pre>
<p>Why <strong>4 dims</strong> (<span class="mono">GGML_MAX_DIMS = 4</span>)? Because inference tensors rarely exceed 4 dims - a typical shape is
"batch x sequence x heads x per-head-dim", and 4 is enough; storing ne/nb in a small <strong>fixed-size</strong> array is simple and fast, with no dynamic
allocation. One more easily-missed point: the memory <span class="mono">data</span> points to is <strong>not necessarily on the CPU</strong> - it may live in GPU
memory, and the tensor has a separate <span class="mono">buffer</span> field recording "which backend owns this data". That foreshadows the multi-backend story
of lesson 07 (the same tensor struct, with data living on different hardware).</p>
<p>Walking the formula through a concrete example makes it stick. Using the same <span class="mono">ne=[3,2]</span> F32 tensor from the memory diagram above:
each element is 4 bytes, so <span class="mono">nb[0]=4</span>; crossing a whole row skips 3 elements, so <span class="mono">nb[1]=4x3=12</span>. To fetch row 1,
element 2 (index <span class="mono">i1=1, i0=2</span>, zero-based - a12 in the diagram), the byte offset is <span class="mono">1*nb[1] + 2*nb[0] = 12 + 8 = 20</span> -
exactly the 20 marked under a12. The whole tensor is <span class="mono">3x2x4 = 24</span> bytes. So with ne, nb, and data, <strong>any element's address is one step
away</strong> - that is the full power of stride; other shapes like <span class="mono">ne=[4,3]</span> are left for this lesson's closing exercise.</p>
<p>Conversely, knowing the byte layout explains <strong>why "row-major traversal" is usually faster than "column-major"</strong>: walking along
<span class="mono">ne[0]</span> (within a row) touches <strong>contiguous, adjacent</strong> bytes, friendliest to the CPU cache; jumping along higher dims takes a
big stride each time and misses the cache more often. Many operator implementations deliberately loop along the contiguous dimension for exactly this reason -
<strong>layout decides performance</strong>, an intuition we will reuse when reading kernel implementations later.</p>

<h2>Why view / transpose copy no data</h2>
<p>Now a very "thrifty" ggml design. Since "<strong>shape (ne/nb)</strong>" and "<strong>data</strong>" are stored separately, many "reshape" operations
<strong>need not touch the data</strong> at all - just change a few numbers. The classic is <strong>transpose</strong>: turning a [rows, cols] matrix into
[cols, rows], ggml merely <strong>swaps ne[0] with ne[1] and nb[0] with nb[1]</strong>, moving <span class="mono">data</span> by not a single byte:</p>
<div class="cols">
  <div class="col"><h4>Original</h4><p><span class="mono">ne=[3,2]</span>, <span class="mono">nb=[4,12]</span><br>data -&gt; a real block of memory</p></div>
  <div class="col"><h4>Transposed (a view)</h4><p><span class="mono">ne=[2,3]</span>, <span class="mono">nb=[12,4]</span><br>view_src -&gt; points back, data unchanged</p></div>
</div>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  Operations like reshape, transpose, slice, and broadcast are mostly implemented as <strong>views</strong> in ggml: the new tensor reuses the <strong>same data</strong>, just with different <span class="mono">ne</span>/<span class="mono">nb</span> and an offset, recording its origin in <span class="mono">view_src</span>. The upside is obvious: <strong>zero-copy, memory-saving, and fast</strong>. The cost: views often become <strong>non-contiguous</strong> (<span class="mono">ggml_is_contiguous</span> is false) - e.g. after a transpose, walking along <span class="mono">ne[0]</span> is no longer adjacent in memory. Some operators require contiguous inputs, in which case you first call <span class="mono">ggml_cont</span> to "compact" it into a fresh contiguous block (note: <strong>this step does copy</strong>). That is why you see many <span class="mono">ggml_cont</span> calls in ggml code - balancing "contiguity" against "zero-copy".
</div>
<p>One common operation in passing: <strong>broadcast</strong>. When two not-quite-same-shape tensors do an element-wise op (e.g. adding the same bias vector to
every row), ggml lets some dimension use "<strong>one element as many</strong>" - again via a stride trick: set that dim's <span class="mono">nb</span> to 0, so no
matter how the index changes the address does not, and the same value is re-read, looking "copied" while taking <strong>not one extra byte</strong>. Another
"change nb, don't move data" classic, of a piece with views and transpose.</p>
<p>A very common reshape: take activations of shape <span class="mono">[n_embd, n_tokens]</span> and "fan them out" for multi-head attention into
<span class="mono">[head_dim, n_head, n_tokens]</span> - the element count is unchanged (<span class="mono">n_embd = head_dim x n_head</span>) and no data moves; only
ne/nb are <strong>reinterpreted</strong>, turning "one big vector" into "several heads' small vectors". The multi-head attention from lesson 04 is full of these
"same data, different shape" operations, all done with views and almost no extra copies.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  Or <strong>slicing</strong>: to pull out "layer k" or "head h" from a big tensor, ggml usually does not copy either - it <strong>computes a starting offset</strong>, pairs it with trimmed ne/nb, and returns a view pointing back at the original data. So in ggml code "<strong>take a part</strong>" and "<strong>see it in a different shape</strong>" are often the same cheap operation underneath - this view-centric style is a key to reading the compute-graph code later.
</div>
<p>To spell out what "contiguous" means: a tensor is <strong>contiguous</strong> when its elements sit <strong>adjacent in memory, in ne[0], ne[1], ... order
without gaps</strong> (i.e. nb follows the formula above exactly). Transpose and some slices break this: same elements, but the "walk order" no longer matches the
memory layout, so <span class="mono">ggml_is_contiguous</span> returns false. Most operators handle contiguous tensors directly; where contiguity is required,
<span class="mono">ggml_cont</span> copies the data into a fresh tidy block in the current shape. Remember this and you will dodge many shape-related debugging traps.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Finally, back to <span class="mono">op</span> and <span class="mono">src</span> - they give a tensor a "<strong>dual identity</strong>": both <strong>a block of data</strong> and <strong>a node in the compute graph</strong>. When you write <span class="mono">c = ggml_mul_mat(ctx, a, b)</span>, ggml does not compute the matmul immediately; it creates a new tensor <span class="mono">c</span>, records its <span class="mono">op</span> as "matmul" and points <span class="mono">src[0]/src[1]</span> at <span class="mono">a</span> and <span class="mono">b</span>. So the whole compute graph is just <strong>tensors holding hands via src into a web</strong>; once built, it is handed to the backend to compute node by node in order. Grasp this "graph node" identity of tensors and you hold the key to Part 3's ggml engine.
</div>
<p>The struct also has a humble but handy field, <span class="mono">name</span>: each tensor can carry a name. This helps when <strong>debugging</strong> (spotting
"which weight is this" when printing the graph), and GGUF stores each weight tensor <strong>with a name</strong> (like <span class="mono">blk.0.attn_q.weight</span>),
matched up by name at load time. So a "name" is not just a comment - it is the <strong>index</strong> between model weights and code. The tensor names you see in
GGUF tools or debug logs come straight from this field.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  A practical tip: the <strong>most common error</strong> writing ggml code is a shape mismatch - matmul requires the two tensors to have equal element counts on the multiplied dimension, and after transpose/reshape the dim order changes, so it is easy to misalign. Get into the habit of <strong>writing out each tensor's ne in your head</strong> (or tagging with <span class="mono">name</span> and printing to check) - it saves a lot of debugging time. That is why this lesson belabors ne/nb: <strong>shape is the "grammar" of ggml programming</strong>; get the grammar wrong and nothing downstream runs.
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Why is ggml's dimension order the reverse of PyTorch? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>In numpy / PyTorch, the <strong>last</strong> dimension is the contiguous one (row-major, C-order): for a tensor of shape
    <span class="mono">[batch, seq, dim]</span>, <span class="mono">dim</span> is contiguous. ggml flips this: it puts the <strong>contiguous</strong> dimension at
    <span class="mono">ne[0]</span> (first), so the same tensor is written <span class="mono">ne = [dim, seq, batch]</span> - the <strong>whole order reversed</strong>.</p>
    <p>Neither is "right"; it is just a different convention. But when reading ggml code and shapes you must mentally switch, or you will mistake rows for
    columns and batch for dim. A handy mnemonic: <strong>ggml's ne[0] is always the "memory-adjacent, fastest-changing" dimension</strong>.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is there a /blck_size in the nb formula? The quantized-type trap <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>For plain types (F32, F16), <span class="mono">nb[0]</span> is just the bytes of one element. But <strong>quantized types</strong> (like Q4_0) are not
    "one value per element"; they pack a <strong>whole block</strong> (e.g. 32 weights) into fixed-size bytes, so a single weight is <strong>not independently
    addressable</strong>.</p>
    <p>So ggml uses <span class="mono">ggml_blck_size(type)</span> (elements per block) and <span class="mono">ggml_type_size(type)</span> (bytes per block).
    The <span class="mono">ne[0] / ggml_blck_size(type)</span> in the <span class="mono">nb[1]</span> formula means "how many <strong>blocks</strong> in a row". Once you
    get this, you see why a quantized tensor cannot be indexed element-by-element like a plain array - you must <strong>dequantize by block</strong> (Part 3's
    quantization-format lesson covers this in detail).</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How do you compute a tensor's memory footprint? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Use <span class="mono">ggml_nbytes(tensor)</span> directly. Intuitively, a <strong>contiguous</strong> tensor's byte count is about "<strong>highest-dim
    element count x highest-dim stride</strong>" (<span class="mono">ne[k] * nb[k]</span> at the top dim) - i.e. multiply all dims' element counts together, times
    bytes per element (or per block).</p>
    <p>This is handy for estimating <strong>memory use</strong>: how much the weights take, how much the KV cache takes, are all essentially sums of such tensor
    byte counts. Non-contiguous and quantized tensors compute slightly differently, but <span class="mono">ggml_nbytes</span> already handles those cases - just call it.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Tensor = <strong>type</strong> + <strong>ne[4]</strong> (shape) + <strong>nb[4]</strong> (byte strides) + <strong>data</strong> (memory) + <strong>op / src</strong> (how it arose in the graph).</li>
    <li>ggml is <strong>row-major</strong>, <strong>ne[0] is the innermost / contiguous dim</strong> (smallest stride); dimension order is the <strong>reverse</strong> of numpy / PyTorch.</li>
    <li>Multi-dim index -&gt; byte offset: <span class="mono">offset = sum i_k * nb[k]</span>.</li>
    <li>view / transpose / reshape are <strong>zero-copy</strong>: change ne/nb, reuse data, record origin in <span class="mono">view_src</span>; the cost is possible <strong>non-contiguity</strong>, compacted by <span class="mono">ggml_cont</span> when needed (that does copy).</li>
    <li>Quantized types store by <strong>block</strong>; the /blck_size in the <span class="mono">nb</span> formula is exactly why.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Storing "<strong>shape</strong>" (ne/nb) and "<strong>data</strong>" separately - that single design turns transpose, slice, reshape, and broadcast all into
  "<strong>change a few numbers</strong>" rather than "move a whole block of memory", and lets the <strong>same weights</strong> be reused by the graph from
  different viewpoints. Much of ggml's efficiency hides in this plain split. In the next lesson on quantization, you will see how this ne/nb/type design lets
  "compress weights to 4 bits" slot quietly into the very same tensor and operator machinery.
</div>
""",
}

LESSON_06 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
量化是 llama.cpp 能把一个 7B、甚至 70B 大模型塞进消费级显卡 / 内存的"<strong>压缩术</strong>"。课 01 提过它有 4/5/8 bit 多种档位；
这一课讲清三件事：<strong>为什么能压、怎么压（块量化）、Q4_0 / Q8_0 / K-quant 各是什么</strong>。
硬核的字节级细节留到第三部分的"量化格式"课，这里先把直觉建立起来。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  量化很像把一张高清照片<strong>按小块压缩</strong>：每个小方块里，先记一个"基准亮度"（scale），块内每个像素只存"相对基准差几档"。
  因为同一小块里的像素通常很接近，几档就够用，压完看起来还和原图差不多。把"像素"换成"模型权重"，这就是块量化——块越小、基准越贴合，还原得越像。
</div>

<h2>为什么要量化：显存与带宽</h2>
<p>训练好的大模型，本质是<strong>一大堆浮点数</strong>（权重）。默认用 16 位浮点（FP16 / BF16）存，每个权重 2 字节——一个 70 亿（7B）参数的模型，
光权重就要 <strong>约 14 GB</strong>，普通显卡和内存直接被劝退。量化就是<strong>用更少的位数近似地存这些权重</strong>：8 bit 砍掉一半、4 bit 再砍一半，
显存需求成倍下降。</p>
<div class="cols">
  <div class="col"><h4>FP16（原始）</h4><p>每权重 2 字节<br><strong>7B -&gt; 约 14 GB</strong></p></div>
  <div class="col"><h4>Q8_0（8-bit）</h4><p>约 1 字节 / 权重<br><strong>7B -&gt; 约 7 GB</strong></p></div>
  <div class="col"><h4>Q4_0（4-bit）</h4><p>约 0.56 字节 / 权重<br><strong>7B -&gt; 约 3.9 GB</strong></p></div>
</div>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  省显存只是其一。更关键的是<strong>带宽</strong>：课 04 说过，decode 阶段的瓶颈常常是"把权重从显存搬到计算单元"。权重越小，每生成一个 token 要搬的字节越少，<strong>速度直接变快</strong>。所以量化往往是"既省显存、又提速"的双赢，代价只是<strong>一点点精度损失</strong>——而大模型对这种损失通常相当宽容。
</div>
<p>为什么大模型能被压到这么狠还能用？因为它的权重里<strong>有大量冗余</strong>：每个权重的具体数值并不需要那么高的精度，模型真正依赖的是这些权重
<strong>整体的统计规律</strong>。把每个数从"非常精确"降到"大致正确"，单看一个损失明显，但成千上万个权重一起作用时，误差很大程度上互相抵消，最终输出几乎不受影响。
这也是为什么 4-bit 量化常常只让模型质量掉一点点，却换来几倍的体积和速度收益。</p>
<p>那为什么大家最常用 <strong>4-bit</strong> 而不是更狠的 2-bit、或更稳的 8-bit？这是个"<strong>甜点</strong>"问题：8-bit 几乎不掉质量，但省得不够多；
2-bit、3-bit 省得很狠，质量却开始明显滑坡。<strong>4-bit（尤其是 K-quant 的 4-bit）落在曲线的拐点上</strong>——体积压到原来的约四分之一，质量却只掉一点点，
于是成了社区下载量最大的档位。具体选哪档，还要看你的硬件、对质量的容忍度，以及模型本身大小（越大的模型，往往越扛得住激进量化）。</p>
<p>顺便补一点背景：前面反复出现的 <strong>FP32 / FP16 / BF16</strong> 都是浮点格式，区别在用多少位、以及怎么分配给"指数"和"尾数"。
FP32 是 4 字节的全精度，训练时常用；FP16 和 BF16 都是 2 字节的半精度，其中 BF16 牺牲一点尾数精度，换来和 FP32 一样大的表示范围，在大模型里很受欢迎。
量化则更进一步，把这些浮点直接换成<strong>更省的整数表示</strong>——可以理解为"在 FP16 已经省了一半的基础上，再往下狠压一截"。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  把这个体量感再放大一点：一个 70B 模型，FP16 要 <strong>约 140 GB</strong>——这是好几张顶级显卡才装得下的量；而 Q4_K_M 量化后只要 <strong>约 40 GB 上下</strong>，一张 48 GB 显存的卡、或一台大内存的机器就能跑起来。正是量化，把"只有大公司机房玩得起"的大模型，变成了"发烧友在家也能折腾"的东西。
</div>

<h2>块量化：每块一个 scale</h2>
<p>那"用更少位数近似"具体怎么做？最朴素的想法：给整个权重矩阵定一个统一的缩放系数（scale），把浮点数等比例映射到一个小整数范围。
但问题是，<strong>同一个矩阵里权重的大小可能差很多</strong>，用一个全局 scale，大值和小值没法兼顾，误差会很大。</p>
<p>块量化的办法是：把权重切成一个个<strong>小块</strong>（Q4_0 里每块 32 个权重），<strong>每块各自配一个 scale</strong>。块内动态范围小、scale 贴得准，
近似自然更精确。这就是"分而治之"在量化上的体现：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>块量化</b>：32 个浮点权重 -&gt; 1 个 scale（半精度）+ 32 个低位整数</div>
  <div class="cells"><span class="lab">原始</span><span class="cell">0.12</span><span class="cell">-0.08</span><span class="cell">0.21</span><span class="cell dim">…</span><span class="cell">-0.15</span><span class="lab">32 个 fp</span></div>
  <div class="cells"><span class="lab">量化后</span><span class="cell scale">d = scale</span><span class="cell q">9</span><span class="cell q">7</span><span class="cell q">11</span><span class="cell q">…</span><span class="cell q">5</span><span class="lab">1 个 scale + 32 个 4-bit</span></div>
</div>
<p>再换个角度理解"为什么块内范围小就更准"：量化的本质是"用有限的几档去近似连续的值"，<strong>这几档要覆盖的范围越窄，每一档之间的间隔就越小、分得越细</strong>。
一整个矩阵里既有 0.001 也有 5.0，硬用 16 档去分，间隔必然很粗；可一旦切成小块，每块内部的数往往挤在相近的量级，同样 16 档分得就细多了。这就是块量化精度更高的根本道理。</p>
<p>落到代码上，Q4_0 的"一块"就是一个紧凑的结构体（来自 <span class="mono">ggml/src/ggml-common.h</span>）：</p>
<pre class="code"><span class="cm">#define QK4_0 32       // 一块 32 个权重</span>
<span class="kw">typedef struct</span> {
    ggml_half d;            <span class="cm">// scale (fp16), 每块一个</span>
    uint8_t   qs[QK4_0/2];  <span class="cm">// 32 个权重, 每个压成 4-bit, 两个挤一字节</span>
} block_q4_0;              <span class="cm">// 2 + 16 = 18 字节 -> 平均 4.5 bit/权重</span></pre>
<p>算笔账：一块 32 个权重，用 <strong>2 字节</strong>存 scale（半精度浮点 <span class="mono">ggml_half</span>）、<strong>16 字节</strong>存 32 个 4-bit 量化值
（每个权重 4 bit，两个挤进一个字节），合计 <strong>18 字节</strong>。摊到每个权重就是 18 × 8 / 32 = <strong>4.5 bit</strong>——这就是"4-bit 量化"实际占用稍多于
4 bit 的原因：那多出来的 0.5 bit，是每块都要分摊的那个 scale。</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Q4_0 一块 = 18 字节</b>，装下 32 个权重</div>
  <div class="cells"><span class="lab">布局</span><span class="cell scale">d：2 字节 scale</span><span class="cell q">qs：32 个 4-bit = 16 字节</span></div>
  <div class="cells"><span class="lab">合计</span><span class="cell">2 + 16 = 18 字节</span><span class="cell dim">摊到每权重 = 4.5 bit</span></div>
</div>
<p>用的时候要"解量化"：把存的 4-bit 整数还原成近似的浮点值。Q4_0 的规则很简单（对应 <span class="mono">ggml/src/ggml-quants.c</span> 的
<span class="mono">dequantize_row_q4_0</span>）：</p>
<pre class="code"><span class="cm"># 每个 4-bit 值 q 在 0..15, 还原成带符号的权重:</span>
for i in range(32):
    q    = nibble(qs, i)     <span class="cm"># 0..15</span>
    x[i] = (q - 8) * d       <span class="cm"># 减 8 居中到 0 附近, 再乘以这一块的 scale d</span></pre>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  那个 <span class="mono">-8</span> 是把 0..15 的无符号范围<strong>平移到 -8..7</strong>，让它对称地分布在 0 两侧（权重有正有负）；乘以 <span class="mono">d</span>则把这个小整数还原回原来的尺度。整个过程没有查表、没有分支，就是一个减法加一个乘法，非常适合在 CPU / GPU 上批量快速跑——这正是 Q4_0 这种"<strong>对称量化</strong>"格式简单高效的原因。
</div>
<p>反过来，<strong>怎么从浮点权重得到那些 4-bit 整数</strong>？以对称量化为例：先在这一块里找出<strong>绝对值最大</strong>的那个权重，用它定出 scale——Q4_0 的具体做法是 <span class="mono">d = max / -8</span>（<span class="mono">max</span> 是块内<strong>带符号</strong>的极值权重），相当于把这个极值锚定到量化范围的端点 <span class="mono">q=0</span>，从而用满 <span class="mono">-8..7</span> 整个范围；再把每个权重除以 <span class="mono">d</span>、四舍五入、加上偏移，压进 0..15。所以"量化"和"解量化"是一对互逆操作：
量化时 <span class="mono">q = round(x/d) + 8</span>，用时 <span class="mono">x ≈ (q-8)*d</span>。两次取整之间丢掉的那点零头，正是量化误差的来源。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  注意那个 scale <span class="mono">d</span> 本身是用<strong>半精度浮点（fp16）</strong>存的，而不是再压成整数——因为每块只有一个 scale，占比很小，用 fp16 保住它的精度很划算，而它的准确度直接决定整块的还原质量。把视角拉远，你会看到一条清晰的<strong>"量化粒度"谱系</strong>：最粗的是整个张量共享一个 scale（per-tensor），中间是每块一个（per-block，如 Q4_0），最细的是超块里每个子块一个（K-quant）。<strong>粒度越细，精度越高，但要存的 scale 也越多</strong>——量化格式的演化，基本就是在这条线上找更好的平衡点。
</div>
<p>顺带一问：块大小为什么常取 <strong>32</strong>？这又是一处权衡——块越小，scale 越贴合局部、精度越高，但要存的 scale 越多、压缩率下降；块越大则相反。
32 是精度和体积之间一个经过实践检验的折中，也正好契合硬件上常见的并行宽度，算起来顺手。K-quant 用 256 的超块再细分，则是想"<strong>既要大块的高压缩率、又要小块的高精度</strong>"，
试图鱼和熊掌兼得。</p>
<p>把一组真实权重压一遍再还原，"有损但损得很小"就看得见了：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次量化往返</b>：4 个权重压成 4-bit 再还原，看误差有多小（数字为示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 原始权重</h5>
      <div class="cellrow"><span class="vc">0.46</span><span class="vc">-0.12</span><span class="vc">0.31</span><span class="vc">-0.40</span></div>
      <div class="tlab">块内 4 个 fp16</div></div>
    <div class="op">找最大<br>定 scale</div>
    <div class="stn"><h5>② scale d</h5>
      <div class="cellrow"><span class="vc hot">-0.058</span></div>
      <div class="tlab">d = max/-8 = 0.46/-8</div></div>
    <div class="op">量化<br>round(x/d)+8</div>
    <div class="stn"><h5>③ 4-bit 码</h5>
      <div class="cellrow"><span class="vc">0</span><span class="vc">10</span><span class="vc">3</span><span class="vc">15</span></div>
      <div class="tlab">存进 0..15</div></div>
    <div class="op">反量化<br>(q-8)×d</div>
    <div class="stn"><h5>④ 还原值</h5>
      <div class="cellrow"><span class="vc blue">0.46</span><span class="vc blue">-0.12</span><span class="vc blue">0.29</span><span class="vc blue">-0.40</span></div>
      <div class="tlab">误差 ≤ 0.02</div></div>
  </div>
</div>

<h2>Q8_0 / K-quant：精度与压缩的取舍</h2>
<p>Q4_0 只是量化大家庭里的一个。同样的"块 + scale"思路，换一换参数，就是不同档位：</p>
<table class="t">
  <tr><th>类型</th><th>每权重 bit</th><th>块 / 超块</th><th>特点</th></tr>
  <tr><td><strong>Q8_0</strong></td><td>约 8.5 bit</td><td>32</td><td>最接近 FP16，体积大，质量最稳</td></tr>
  <tr><td><strong>Q4_0</strong></td><td>约 4.5 bit</td><td>32</td><td>最轻最快，精度损失较明显</td></tr>
  <tr><td><strong>Q4_K</strong></td><td>约 4.5 bit</td><td>超块 256</td><td>子块各有 scale + 混合精度，同 bit 下更准</td></tr>
</table>
<p><strong>Q8_0</strong> 用 8 bit 存每个权重（块还是 32，结构是 <span class="mono">d + 32 个 int8</span>），约 8.5 bit/权重，体积大但精度最接近原始 FP16，
常用于对质量最敏感的场合。<strong>Q4_0</strong> 最轻，4.5 bit，速度快、省显存，但精度损失相对明显。</p>
<p>而 <strong>K-quant</strong>（名字里带 K，如 <span class="mono">Q4_K</span>、<span class="mono">Q5_K</span>）是更聪明的一档：它用更大的"<strong>超块</strong>"
（super-block，<span class="mono">QK_K = 256</span> 个权重），超块内再分成若干子块，每个子块有自己更精细的 scale 和 min，还会对不同张量<strong>混合用不同位宽</strong>。
结果是：在<strong>同样的平均 bit 数</strong>下，K-quant 的困惑度（perplexity，衡量模型预测好坏的指标，越低越好）通常明显低于对应的 Q4_0 / Q5_0。
今天大家从网上下载的 GGUF，大多就是 <span class="mono">Q4_K_M</span> 这类 K-quant 档位。</p>
<p>把这些名字连起来读就有规律了：<strong>字母 Q + 位数 + 可选的 K + 可选的档位后缀</strong>。<span class="mono">Q4_0</span> 是"4-bit、对称、基础块"，
<span class="mono">Q4_K_M</span> 是"4-bit、K-quant、中档混合精度"，<span class="mono">Q8_0</span> 是"8-bit、对称、基础块"。下次在下载页面看到一长串
<span class="mono">Q3_K_S</span>、<span class="mono">Q5_K_M</span>、<span class="mono">Q6_K</span>，你就能一眼读出它大概多大、多准了。</p>
<pre class="code"><span class="kw">typedef struct</span> {
    ggml_half d;          <span class="cm">// 超块整体 scale</span>
    ggml_half dmin;       <span class="cm">// 超块整体 min (非对称)</span>
    uint8_t scales[...];  <span class="cm">// 各子块更细的 scale/min (6-bit, 已量化)</span>
    uint8_t qs[...];      <span class="cm">// 量化值</span>
} block_q4_K;            <span class="cm">// QK_K = 256, 简化自 ggml-common.h</span></pre>
<p>还有两点值得知道。其一，<strong>llama.cpp 量化的主要是"权重"</strong>，推理时流动的"激活值"通常仍用较高精度（如 fp16/fp32）计算——因为权重是静态的、
占绝大多数内存，最值得压；激活是动态的，过度量化更容易伤精度。其二，<strong>并非每个张量都用同一档</strong>：像词嵌入、输出投影这种对质量影响大的张量，
常被刻意保留在更高的位宽，这正是 K-quant 的 <span class="mono">_M</span> / <span class="mono">_L</span> 档在做的"混合精度"。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  那这些量化文件是怎么来的？流程很直接：先用转换脚本把原始模型导出成一个<strong>高精度的 GGUF</strong>（通常是 fp16），再用 <span class="mono">llama-quantize</span>工具把它"压"成目标档位，比如 <span class="mono">llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M</span>。量化是<strong>一次性的离线操作</strong>，压完得到一个更小的 GGUF，之后每次加载运行的都是这个小文件——所以量化的开销只在"制作"时付一次，运行时只享受它带来的省与快。
</div>
<p>怎么衡量"量化掉了多少质量"？最常用的指标是<strong>困惑度（perplexity）</strong>：拿一段标准文本，看模型对"下一个词"预测得有多准，困惑度越低越好。
社区常做的事，就是把同一个模型的各个量化档位都跑一遍困惑度、列成表对比——你会看到 Q8_0 几乎和 fp16 持平、Q4_K_M 只高一丁点，而 Q2_K 则明显抬高。
<span class="mono">llama-perplexity</span> 工具就是干这个的，第七部分会专门讲怎么用它给量化"打分"。一个经验法则：<strong>在显存放得下的前提下，尽量选高一档</strong>，质量更有保障。</p>
<p>最后澄清一个边界：量化只压缩权重的<strong>表示方式</strong>（每个数用几位存），<strong>并不改变模型的结构</strong>——层数、维度、参数个数都原封不动。
这和<strong>剪枝</strong>（删掉一部分权重 / 神经元）、<strong>蒸馏</strong>（训练一个更小的新模型去模仿大模型）是完全不同的三条压缩路线。量化最大的好处就是
<strong>几乎免费、即插即用</strong>：不用重新训练，一个命令就能把现成模型变小变快，这也是它在本地推理里如此普及的原因。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  量化的损失也不是对所有任务一视同仁。<strong>闲聊、续写这类容错高的任务，4-bit 几乎感觉不出差别</strong>；而<strong>代码生成、数学推理、长链条逻辑</strong>这类"一步错步步错"的任务，对精度更敏感，激进量化时更容易看出退步。所以同一个 Q4 模型，你拿它聊天觉得很好、拿它写复杂代码却偶有翻车，并不奇怪——这时往上换一档（如 <span class="mono">Q5_K</span>、<span class="mono">Q6_K</span> 甚至 <span class="mono">Q8_0</span>）往往能找回不少。
</div>
<p>还有个实用细节：<strong>不同硬件后端对量化格式的支持和优化程度并不一样</strong>。同一个 Q4_K 模型，在 CPU 上靠 SIMD 指令快速解量化、在 CUDA 上有专门的核函数，
速度表现可能差别不小。所以"选哪个量化档位"有时也要看你打算在什么硬件上跑——这部分在第六部分讲内核时会更具体。整体而言，主流的 Q4_K / Q8_0 在各后端都有良好支持，闭眼选基本不会错。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  最后把这一课和课 02、课 05 串起来：一个量化后的 GGUF 文件，里面每个权重张量都带着自己的 <span class="mono">type</span>（课 05 讲过），有的标着 <span class="mono">Q4_K</span>、有的可能是 <span class="mono">Q6_K</span> 或 <span class="mono">F16</span>。加载时，<span class="mono">llama-model-loader</span>按张量的 type 决定怎么读、怎么解量化；计算时，ggml 的算子（如矩阵乘）能<strong>直接吃量化权重</strong>，在乘法的内层即时解量化，省去"先整体还原成 fp32 再算"的开销。所以"量化"不是一个孤立的步骤，而是<strong>贯穿存储（GGUF）、加载（loader）、计算（ggml 算子）的一条完整链路</strong>。
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Q4_0 / Q4_1 / Q8_0 后面的数字和 0/1 是什么意思？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>数字是<strong>每个权重的 bit 数</strong>：Q4 = 4 bit、Q8 = 8 bit。后缀 <span class="mono">_0</span> / <span class="mono">_1</span> 区分量化的"对称性"。</p>
    <p><span class="mono">_0</span> 是<strong>对称量化</strong>，只存一个 scale、零点固定（就像前面 Q4_0 的 <span class="mono">(q-8)*d</span>）；<span class="mono">_1</span> 是
    <strong>非对称量化</strong>，额外再存一个最小值 min（偏移量），还原公式变成 <span class="mono">q*d + min</span>。多存一个 min 让它能更贴合那些
    "不以 0 为中心"的权重分布，精度略高，但每块要多占几字节。要不要这个 min，就是 <span class="mono">_0</span> 和 <span class="mono">_1</span> 的区别。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> K-quant（Q4_K_M 等）凭什么更准？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>关键在<strong>更细粒度的 scale</strong>。普通 Q4_0 是"32 个权重共享 1 个 scale"；K-quant 用 256 个权重的超块，但超块内部再切成多个子块，
    <strong>每个子块有自己的 scale 和 min</strong>（这些子块 scale 本身又被量化成 6-bit 存起来，省空间）。粒度越细，scale 越能贴合局部，误差越小。</p>
    <p>名字里的 <span class="mono">_S</span> / <span class="mono">_M</span> / <span class="mono">_L</span>（small / medium / large）是不同的"<strong>混合精度</strong>"档：
    对模型里更重要的层用稍高的位宽、不重要的用低位宽，在体积和精度之间取不同平衡。所以同样标着"4-bit"，<span class="mono">Q4_K_M</span> 往往比
    <span class="mono">Q4_0</span> 又准又只大一点点——这也是它成为主流下载格式的原因。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> imatrix 是什么？和量化什么关系？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p><strong>imatrix（importance matrix，重要性矩阵）</strong>常被误解为"决定每个权重用几 bit"——其实<strong>不是</strong>。它是用一批<strong>校准数据</strong>
    跑一遍模型，统计出"每个权重对最终输出的影响有多大"。</p>
    <p>量化时，<strong>对更重要的权重，让它的量化误差更小</strong>（在选 scale、取整时更偏向保住它们）。换句话说，imatrix 改变的是"<strong>误差怎么分配</strong>"，
    让宝贵的精度用在刀刃上，而<strong>不改变</strong>位宽分配本身。它由 <span class="mono">llama-imatrix</span> 工具生成，再喂给 <span class="mono">llama-quantize</span>
    一起用，通常能在<strong>不增大体积</strong>的前提下进一步降低困惑度。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>量化 = 用更少 bit 近似存权重，<strong>省显存、省带宽、提速</strong>，代价是少量精度损失（大模型通常很宽容）。</li>
    <li><strong>块量化</strong>：把权重切成小块、每块一个 scale；Q4_0 每块 32 个权重 = 2 字节 scale + 16 字节量化值 = <strong>18 字节 = 4.5 bit/权重</strong>。</li>
    <li>解量化就是 <span class="mono">x = (q - 8) * d</span> 这么简单（Q4_0 对称量化）。</li>
    <li><strong>Q8_0</strong> 准而大、<strong>Q4_0</strong> 轻而糙、<strong>K-quant</strong>（超块 + 子块 scale + 混合精度）同 bit 下更准，是当下主流。</li>
    <li><strong>imatrix 是误差加权、不是位宽分配。</strong></li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  "<strong>每块一个 scale</strong>"——就这一个朴素的想法，把"浮点数动态范围太大"这个全局难题，拆成了无数个"块内范围很小"的局部小问题，
  于是低到 4 bit 也能保住可用的精度。大模型能从数据中心走进你的笔记本，这一招居功至伟。记住一句话：<strong>量化不是把模型变笨，
  而是把"过度精确"的浪费挤掉</strong>——用刚刚好的位数，装下模型真正需要的那部分信息。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Quantization is the "<strong>compression trick</strong>" that lets llama.cpp fit a 7B - or even 70B - model into consumer GPU / RAM. Lesson 01 mentioned its
4/5/8-bit tiers; this lesson nails down three things: <strong>why it compresses, how it compresses (block quantization), and what Q4_0 / Q8_0 / K-quant are</strong>.
The hardcore byte-level details wait for Part 3's "quantization format" lesson; here we build the intuition first.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Quantization is like compressing a high-res photo <strong>block by block</strong>: in each small block, record one "baseline brightness" (scale), and each pixel
  stores only "how many notches off the baseline". Since pixels in one block are usually close, a few notches suffice and the result still looks like the original.
  Swap "pixels" for "model weights" and that is block quantization - the smaller the block and the tighter the baseline, the closer the reconstruction.
</div>

<h2>Why quantize: memory and bandwidth</h2>
<p>A trained model is essentially <strong>a huge pile of floating-point numbers</strong> (weights). Stored by default in 16-bit float (FP16 / BF16), each weight is
2 bytes - a 7-billion-parameter (7B) model needs <strong>about 14 GB</strong> for weights alone, which ordinary GPUs and RAM simply refuse. Quantization
<strong>approximates these weights with fewer bits</strong>: 8-bit halves it, 4-bit halves it again, dropping memory needs several-fold.</p>
<div class="cols">
  <div class="col"><h4>FP16 (original)</h4><p>2 bytes / weight<br><strong>7B -&gt; ~14 GB</strong></p></div>
  <div class="col"><h4>Q8_0 (8-bit)</h4><p>~1 byte / weight<br><strong>7B -&gt; ~7 GB</strong></p></div>
  <div class="col"><h4>Q4_0 (4-bit)</h4><p>~0.56 bytes / weight<br><strong>7B -&gt; ~3.9 GB</strong></p></div>
</div>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Saving memory is only half of it. The bigger win is <strong>bandwidth</strong>: as lesson 04 noted, the decode bottleneck is often "moving weights from memory to the compute units". Smaller weights mean fewer bytes to move per generated token, so it is <strong>directly faster</strong>. Quantization is thus usually a win-win - less memory and more speed - at the cost of just <strong>a little accuracy</strong>, which large models tolerate quite well.
</div>
<p>Why can a big model be squeezed this hard and still work? Because its weights carry <strong>a lot of redundancy</strong>: each weight's exact value need not be so
precise; what the model really relies on is the <strong>overall statistical pattern</strong> of the weights. Dropping each number from "very precise" to "roughly
right" looks lossy one at a time, but across thousands of weights the errors largely cancel, leaving the output almost unaffected. That is why 4-bit quantization
often costs only a sliver of quality for several-fold gains in size and speed.</p>
<p>So why is <strong>4-bit</strong> the go-to rather than more aggressive 2-bit or safer 8-bit? It is a "<strong>sweet spot</strong>" question: 8-bit barely loses
quality but saves too little; 2-bit and 3-bit save a lot but quality starts to slide noticeably. <strong>4-bit (especially K-quant 4-bit) sits at the knee of the
curve</strong> - about a quarter the size, only a sliver of quality lost - so it is the most-downloaded tier. Which exact tier still depends on your hardware, your
quality tolerance, and the model's own size (bigger models usually withstand aggressive quantization better).</p>
<p>A bit of background: the <strong>FP32 / FP16 / BF16</strong> that keep coming up are all floating-point formats, differing in how many bits they use and how they
split them between "exponent" and "mantissa". FP32 is 4-byte full precision, common in training; FP16 and BF16 are both 2-byte half precision, with BF16 trading some
mantissa precision for the same wide range as FP32, which large models like. Quantization goes a step further, replacing these floats with <strong>cheaper integer
representations</strong> - think of it as "squeezing further down, on top of the half FP16 already saved".</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  To scale the intuition up: a 70B model in FP16 needs <strong>about 140 GB</strong> - several top-end GPUs' worth; quantized to Q4_K_M it needs only <strong>around 40 GB</strong>, runnable on a single 48 GB card or a big-RAM machine. Quantization is exactly what turned "only a corporate data center can afford it" models into something "an enthusiast can tinker with at home".
</div>

<h2>Block quantization: one scale per block</h2>
<p>So how exactly do we "approximate with fewer bits"? The naive idea: pick one global scale for the whole weight matrix and map the floats proportionally into a
small integer range. The problem: <strong>weight magnitudes within one matrix can vary a lot</strong>, and a single global scale cannot serve both large and small
values well, so the error is big.</p>
<p>Block quantization's answer: cut the weights into small <strong>blocks</strong> (32 weights per block in Q4_0) and give <strong>each block its own scale</strong>.
A block's dynamic range is small, the scale fits tightly, and the approximation is naturally more accurate. This is "divide and conquer" applied to quantization:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Block quantization</b>: 32 float weights -&gt; 1 scale (half-precision) + 32 low-bit integers</div>
  <div class="cells"><span class="lab">original</span><span class="cell">0.12</span><span class="cell">-0.08</span><span class="cell">0.21</span><span class="cell dim">…</span><span class="cell">-0.15</span><span class="lab">32 floats</span></div>
  <div class="cells"><span class="lab">quantized</span><span class="cell scale">d = scale</span><span class="cell q">9</span><span class="cell q">7</span><span class="cell q">11</span><span class="cell q">…</span><span class="cell q">5</span><span class="lab">1 scale + 32 4-bit</span></div>
</div>
<p>Another way to see "why a small in-block range is more accurate": quantization approximates continuous values with a few fixed levels, and <strong>the narrower
the range those levels must cover, the smaller the gap between levels and the finer the quantization</strong>. A whole matrix holding both 0.001 and 5.0 forced into 16
levels has coarse gaps; but cut into small blocks, the numbers in each block usually cluster at a similar magnitude, so the same 16 levels resolve them far more finely.
That is the root reason block quantization is more accurate.</p>
<p>In code, one Q4_0 "block" is a compact struct (from <span class="mono">ggml/src/ggml-common.h</span>):</p>
<pre class="code"><span class="cm">#define QK4_0 32       // 32 weights per block</span>
<span class="kw">typedef struct</span> {
    ggml_half d;            <span class="cm">// scale (fp16), one per block</span>
    uint8_t   qs[QK4_0/2];  <span class="cm">// 32 weights, each 4-bit, two packed per byte</span>
} block_q4_0;              <span class="cm">// 2 + 16 = 18 bytes -> 4.5 bit/weight on average</span></pre>
<p>Do the math: a block of 32 weights uses <strong>2 bytes</strong> for the scale (half-precision <span class="mono">ggml_half</span>) and <strong>16 bytes</strong> for 32
4-bit quants (4 bits each, two packed into a byte), totaling <strong>18 bytes</strong>. Per weight that is 18 x 8 / 32 = <strong>4.5 bit</strong> - which is why "4-bit
quantization" actually takes a bit more than 4 bits: the extra 0.5 bit is the per-block scale, amortized over the block.</p>
<div class="cellgroup">
  <div class="cg-cap"><b>Q4_0 one block = 18 bytes</b>, holding 32 weights</div>
  <div class="cells"><span class="lab">layout</span><span class="cell scale">d: 2-byte scale</span><span class="cell q">qs: 32 x 4-bit = 16 bytes</span></div>
  <div class="cells"><span class="lab">total</span><span class="cell">2 + 16 = 18 bytes</span><span class="cell dim">per weight = 4.5 bit</span></div>
</div>
<p>To use it you "dequantize": restore the stored 4-bit integers to approximate floats. Q4_0's rule is simple (cf. <span class="mono">dequantize_row_q4_0</span> in
<span class="mono">ggml/src/ggml-quants.c</span>):</p>
<pre class="code"><span class="cm"># each 4-bit value q in 0..15, restored to a signed weight:</span>
for i in range(32):
    q    = nibble(qs, i)     <span class="cm"># 0..15</span>
    x[i] = (q - 8) * d       <span class="cm"># subtract 8 to center near 0, then scale by the block's d</span></pre>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  The <span class="mono">-8</span> shifts the unsigned 0..15 range to <strong>-8..7</strong>, placing it symmetrically around 0 (weights are positive and negative); multiplying by <span class="mono">d</span> restores the small integer to the original scale. The whole thing has no lookup table and no branches - just a subtract and a multiply - perfect for running fast in bulk on CPU / GPU. That is why a "<strong>symmetric quantization</strong>" format like Q4_0 is so simple and efficient.
</div>
<p>Conversely, <strong>how do you get those 4-bit integers from float weights</strong>? For symmetric quantization: find the largest-magnitude weight in the block and
use it to set the scale; Q4_0's recipe is <span class="mono">d = max / -8</span> (<span class="mono">max</span> is the block's <strong>signed</strong> extreme weight),
anchoring that extreme to the end of the range (<span class="mono">q=0</span>) so the full <span class="mono">-8..7</span> range is used; then divide each weight by
<span class="mono">d</span>, round, and add the offset to pack into 0..15. So "quantize" and "dequantize" are inverse operations: quantize with
<span class="mono">q = round(x/d) + 8</span>, use with <span class="mono">x ~= (q-8)*d</span>. The little remainder lost between the two roundings is exactly where
quantization error comes from.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Note the scale <span class="mono">d</span> itself is stored in <strong>half precision (fp16)</strong>, not further squeezed to an integer - since there is only one scale per block its overhead is tiny, and keeping its precision in fp16 is well worth it because its accuracy directly determines the whole block's reconstruction quality. Zooming out, you see a clear <strong>"granularity spectrum"</strong>: coarsest is one scale for the whole tensor (per-tensor), middle is one per block (per-block, like Q4_0), finest is one per sub-block within a super-block (K-quant). <strong>Finer granularity means higher accuracy but more scales to store</strong> - the evolution of quantization formats is basically finding better balance points along this line.
</div>
<p>A side question: why is the block size often <strong>32</strong>? Another trade-off - smaller blocks fit the local scale better and raise accuracy but need more
scales, lowering the compression ratio; larger blocks do the reverse. 32 is a practice-tested compromise between accuracy and size, and it also matches common hardware
parallel widths, so it computes nicely. K-quant's 256-weight super-block with sub-division then tries to get "<strong>both the high compression of big blocks and the
high accuracy of small ones</strong>".</p>
<p>Push a few real weights through and back, and "lossy but barely" becomes visible:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one quantization round-trip</b>: 4 weights squeezed to 4-bit and restored - see how small the error is (numbers illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) original weights</h5>
      <div class="cellrow"><span class="vc">0.46</span><span class="vc">-0.12</span><span class="vc">0.31</span><span class="vc">-0.40</span></div>
      <div class="tlab">4 fp16 in a block</div></div>
    <div class="op">find max<br>set scale</div>
    <div class="stn"><h5>(2) scale d</h5>
      <div class="cellrow"><span class="vc hot">-0.058</span></div>
      <div class="tlab">d = max/-8 = 0.46/-8</div></div>
    <div class="op">quantize<br>round(x/d)+8</div>
    <div class="stn"><h5>(3) 4-bit codes</h5>
      <div class="cellrow"><span class="vc">0</span><span class="vc">10</span><span class="vc">3</span><span class="vc">15</span></div>
      <div class="tlab">stored in 0..15</div></div>
    <div class="op">dequant<br>(q-8)*d</div>
    <div class="stn"><h5>(4) restored</h5>
      <div class="cellrow"><span class="vc blue">0.46</span><span class="vc blue">-0.12</span><span class="vc blue">0.29</span><span class="vc blue">-0.40</span></div>
      <div class="tlab">error &lt;= 0.02</div></div>
  </div>
</div>

<h2>Q8_0 / K-quant: trading accuracy against compression</h2>
<p>Q4_0 is just one member of the quantization family. The same "block + scale" idea, with different parameters, gives different tiers:</p>
<table class="t">
  <tr><th>Type</th><th>bits / weight</th><th>block / super-block</th><th>character</th></tr>
  <tr><td><strong>Q8_0</strong></td><td>~8.5 bit</td><td>32</td><td>closest to FP16, large, most stable quality</td></tr>
  <tr><td><strong>Q4_0</strong></td><td>~4.5 bit</td><td>32</td><td>lightest and fastest, more visible accuracy loss</td></tr>
  <tr><td><strong>Q4_K</strong></td><td>~4.5 bit</td><td>super-block 256</td><td>per-sub-block scales + mixed precision, more accurate at the same bits</td></tr>
</table>
<p><strong>Q8_0</strong> stores each weight in 8 bits (block still 32, struct is <span class="mono">d + 32 int8</span>), ~8.5 bit/weight - large but closest to the original
FP16, used where quality matters most. <strong>Q4_0</strong> is lightest at 4.5 bit, fast and memory-saving, but with more visible accuracy loss.</p>
<p>And <strong>K-quant</strong> (the ones with K, like <span class="mono">Q4_K</span>, <span class="mono">Q5_K</span>) is a smarter tier: it uses a larger
"<strong>super-block</strong>" (<span class="mono">QK_K = 256</span> weights), splits it into several sub-blocks each with its own finer scale and min, and even
<strong>mixes bit-widths</strong> across tensors. The result: at the <strong>same average bits</strong>, K-quant's perplexity (a measure of prediction quality, lower
is better) is usually clearly lower than the matching Q4_0 / Q5_0. The GGUF files people download today are mostly K-quant tiers like
<span class="mono">Q4_K_M</span>.</p>
<p>Read these names together and a pattern emerges: <strong>letter Q + bit count + optional K + optional tier suffix</strong>. <span class="mono">Q4_0</span> is
"4-bit, symmetric, basic block", <span class="mono">Q4_K_M</span> is "4-bit, K-quant, medium mixed precision", <span class="mono">Q8_0</span> is "8-bit, symmetric, basic
block". Next time you see a long list of <span class="mono">Q3_K_S</span>, <span class="mono">Q5_K_M</span>, <span class="mono">Q6_K</span> on a download page, you can read
off roughly how big and how accurate each is.</p>
<pre class="code"><span class="kw">typedef struct</span> {
    ggml_half d;          <span class="cm">// super-block overall scale</span>
    ggml_half dmin;       <span class="cm">// super-block overall min (asymmetric)</span>
    uint8_t scales[...];  <span class="cm">// finer per-sub-block scale/min (6-bit, quantized)</span>
    uint8_t qs[...];      <span class="cm">// quants</span>
} block_q4_K;            <span class="cm">// QK_K = 256, simplified from ggml-common.h</span></pre>
<p>Two more things worth knowing. First, <strong>llama.cpp mainly quantizes "weights"</strong>; the "activations" flowing during inference are usually still computed
in higher precision (fp16/fp32) - weights are static and dominate memory, so they are most worth compressing, while activations are dynamic and over-quantizing them
hurts accuracy more easily. Second, <strong>not every tensor uses the same tier</strong>: high-impact tensors like the embedding and output projection are often
deliberately kept at higher bit-widths - exactly the "mixed precision" the K-quant <span class="mono">_M</span> / <span class="mono">_L</span> tiers do.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  So where do these quantized files come from? The flow is direct: first export the original model to a <strong>high-precision GGUF</strong> (usually fp16) with the conversion script, then "compress" it to the target tier with the <span class="mono">llama-quantize</span> tool, e.g. <span class="mono">llama-quantize model-f16.gguf model-Q4_K_M.gguf Q4_K_M</span>. Quantization is a <strong>one-time offline operation</strong>; you get a smaller GGUF and run that small file every time afterwards - so you pay the quantization cost once at "manufacture", and only enjoy the savings and speed at runtime.
</div>
<p>How do you measure "how much quality quantization cost"? The most common metric is <strong>perplexity</strong>: take a standard text and see how well the model
predicts the "next word" - lower is better. A common community exercise is to run perplexity across a model's quant tiers and tabulate them - you will see Q8_0 nearly
matching fp16, Q4_K_M only a hair higher, and Q2_K clearly higher. The <span class="mono">llama-perplexity</span> tool does exactly this, and Part 7 covers using it to
"grade" quantization. A rule of thumb: <strong>pick one tier higher whenever memory allows</strong>, for safer quality.</p>
<p>Finally, a boundary to clarify: quantization only compresses the weights' <strong>representation</strong> (how many bits each number uses); it <strong>does not change
the model's structure</strong> - layer count, dimensions, parameter count all stay. That makes it a different path from <strong>pruning</strong> (removing some
weights/neurons) and <strong>distillation</strong> (training a smaller new model to mimic a big one). Quantization's big advantage is being <strong>nearly free and
plug-and-play</strong>: no retraining, one command shrinks and speeds up an existing model - which is why it is so ubiquitous in local inference.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  Quantization's loss is not uniform across tasks. <strong>Forgiving tasks like chat and free continuation barely show a difference at 4-bit</strong>; but <strong>code generation, math reasoning, long chains of logic</strong> - where one wrong step cascades - are more sensitive to precision and show regressions more readily under aggressive quantization. So it is not odd that the same Q4 model feels great chatting but occasionally trips on complex code - bumping up a tier (<span class="mono">Q5_K</span>, <span class="mono">Q6_K</span>, even <span class="mono">Q8_0</span>) often recovers a lot.
</div>
<p>Another practical detail: <strong>different hardware backends support and optimize quant formats to different degrees</strong>. The same Q4_K model may perform quite
differently dequantizing via SIMD on CPU versus a dedicated kernel on CUDA. So "which tier" sometimes also depends on what hardware you will run on - Part 6 on kernels
gets concrete about this. Overall, the mainstream Q4_K / Q8_0 are well supported on all backends, so you can pick them blind without much risk.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Finally, tying this lesson to lessons 02 and 05: a quantized GGUF carries, for each weight tensor, its own <span class="mono">type</span> (from lesson 05) - some marked <span class="mono">Q4_K</span>, some maybe <span class="mono">Q6_K</span> or <span class="mono">F16</span>. At load time, <span class="mono">llama-model-loader</span> reads and dequantizes per tensor type; at compute time, ggml's operators (like matmul) can <strong>consume quantized weights directly</strong>, dequantizing on the fly in the inner loop, skipping the cost of fully restoring to fp32 first. So "quantization" is not an isolated step but <strong>a full chain across storage (GGUF), loading (loader), and computation (ggml operators)</strong>.
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> What do the numbers and the 0/1 in Q4_0 / Q4_1 / Q8_0 mean? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The number is <strong>bits per weight</strong>: Q4 = 4-bit, Q8 = 8-bit. The suffix <span class="mono">_0</span> / <span class="mono">_1</span> distinguishes the
    quantization's "symmetry".</p>
    <p><span class="mono">_0</span> is <strong>symmetric</strong> - it stores only a scale with a fixed zero point (like Q4_0's <span class="mono">(q-8)*d</span> above);
    <span class="mono">_1</span> is <strong>asymmetric</strong> - it stores an extra minimum (offset), so the formula becomes <span class="mono">q*d + min</span>. The extra
    min lets it fit weight distributions that are "not centered on 0", giving slightly better accuracy at a few extra bytes per block. Whether to keep that min is
    exactly the <span class="mono">_0</span> vs <span class="mono">_1</span> difference.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why is K-quant (Q4_K_M etc.) more accurate? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The key is <strong>finer-grained scales</strong>. Plain Q4_0 is "32 weights share 1 scale"; K-quant uses a 256-weight super-block but cuts it into several
    sub-blocks, <strong>each with its own scale and min</strong> (these sub-block scales are themselves quantized to 6 bits to save space). Finer granularity means the
    scale fits the local data better, so error shrinks.</p>
    <p>The <span class="mono">_S</span> / <span class="mono">_M</span> / <span class="mono">_L</span> (small / medium / large) in the name are different
    "<strong>mixed-precision</strong>" tiers: use slightly higher bits for the model's more important layers and lower bits for the rest, balancing size and accuracy
    differently. So even labeled "4-bit", <span class="mono">Q4_K_M</span> is often both more accurate and only slightly larger than <span class="mono">Q4_0</span> - which
    is why it became the mainstream download format.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> What is imatrix, and how does it relate to quantization? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p><strong>imatrix (importance matrix)</strong> is often misread as "deciding how many bits each weight gets" - it is <strong>not</strong>. It runs the model over a
    batch of <strong>calibration data</strong> to measure "how much each weight influences the final output".</p>
    <p>During quantization, <strong>more important weights are kept with smaller quantization error</strong> (the scale choice and rounding lean toward preserving them).
    In other words, imatrix changes <strong>how the error is distributed</strong>, spending precious precision where it matters, while <strong>not changing</strong> the
    bit-width allocation itself. It is produced by the <span class="mono">llama-imatrix</span> tool and fed to <span class="mono">llama-quantize</span>, usually lowering
    perplexity further <strong>without increasing size</strong>.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Quantization = approximate weights with fewer bits, <strong>saving memory and bandwidth and gaining speed</strong>, at a small accuracy cost (large models tolerate it well).</li>
    <li><strong>Block quantization</strong>: cut weights into blocks, one scale per block; Q4_0 has 32 weights/block = 2-byte scale + 16 bytes of quants = <strong>18 bytes = 4.5 bit/weight</strong>.</li>
    <li>Dequantization is just <span class="mono">x = (q - 8) * d</span> (Q4_0 symmetric quantization).</li>
    <li><strong>Q8_0</strong> accurate but large, <strong>Q4_0</strong> light but coarse, <strong>K-quant</strong> (super-block + per-sub-block scales + mixed precision) more accurate at the same bits and is today's mainstream.</li>
    <li><strong>imatrix is error weighting, not bit-width allocation.</strong></li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  "<strong>One scale per block</strong>" - that single plain idea turns the global problem of "floats have too wide a dynamic range" into countless local problems of
  "a block's range is small", so even 4 bits can hold usable accuracy. That large models can walk out of the data center and onto your laptop owes much to this one move. Remember one line:
  <strong>quantization does not make the model dumber; it squeezes out the waste of "over-precision"</strong> - using just enough bits to hold the information the model actually needs.
</div>
""",
}

LESSON_07 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前面几课都在讲"是什么"；这一课讲"<strong>怎么把它编出来、怎么挑硬件后端、编完有哪些产物</strong>"。读完这一课，你就能自己从源码 build 一个带 GPU 加速的 llama.cpp，
也能看懂为什么同一份代码能在 CPU、NVIDIA、苹果、AMD 各种硬件上跑。这一课也是第二部分的收尾。
</p>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  CMake 像一个<strong>装修总包工头</strong>：你在订单上勾选"要不要地暖（CUDA）、要不要中央空调（Metal / Vulkan）"，它就去联系对应的施工队（编译器和各家 GPU 工具链）、
  画出施工图（生成构建文件），最后交付一套能直接入住的房子（可执行文件）。你不用懂每个施工队的细节，只要会下订单——这一课就是教你"怎么下这张订单"。
</div>

<h2>后端抽象：同一张计算图，多种硬件</h2>
<p>课 03 说过，一次推理会被描述成一张<strong>计算图</strong>（一堆算子：矩阵乘、softmax、rope……）。但同样一个"矩阵乘"，在 CPU 上要用 SIMD 指令写、在 NVIDIA 上要用 CUDA 写、
在苹果上要用 Metal 写——实现天差地别。如果让上层的推理逻辑去操心这些，代码会乱成一团。</p>
<p>ggml 的解法是定义一个统一的<strong>后端（backend）接口</strong>（<span class="mono">ggml/include/ggml-backend.h</span>）：上层只管"我要算这张图"，至于"在哪种硬件上、用什么指令算"，
交给具体的后端实现（<span class="mono">ggml-cpu</span>、<span class="mono">ggml-cuda</span>、<span class="mono">ggml-metal</span>……）。这正是课 01 反复强调的"<strong>把'算什么'和'在哪算'解耦</strong>"。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">上层</span><span class="name">推理逻辑 / 计算图</span></div><div class="ld">课 03 的"建图"：只描述要算什么（matmul · softmax · rope ...）</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">接口</span><span class="name">ggml-backend</span></div><div class="ld">统一的后端接口：分配内存、调度算子、在设备间搬数据</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">实现</span><span class="name">ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …</span></div><div class="ld">每种硬件一份实现，真正把算子算出来</div></div>
</div>
<p>这个分层的好处是：<strong>加一种新硬件，只需新写一个后端，上层推理代码一行不用动</strong>；而编译时选哪些后端，就决定了你这份二进制"认得"哪些硬件。
所以"构建"和"后端"是同一件事的两面——构建系统的主要工作，就是按你的选择，把对应的后端代码编进来。</p>
<p>再补一点机制：每个后端在编进来时会向 ggml <strong>注册自己</strong>，声明"我能算哪些算子、管哪块内存"。运行时，调度器（<span class="mono">ggml-backend</span> 里的 sched）
拿到计算图后，会把每个算子<strong>分派给合适的后端</strong>去算，并在 CPU 和 GPU 内存之间按需搬运数据。万一某个算子在 GPU 后端里还没实现，调度器通常能
<strong>自动回退（fallback）到 CPU</strong> 把这一步算完——所以即使某个新算子 GPU 还没支持，整张图也不至于跑不起来，只是那一步慢一点。</p>
<p>还要补一句：<strong>"后端"不全是 GPU</strong>。CPU 本身就是一个后端；BLAS 是给 CPU 上大矩阵乘加速的库后端；苹果的 Accelerate 框架也能接进来。
所以"后端"更准确的说法是"<strong>一种把算子真正算出来的实现途径</strong>"，GPU 只是其中最受关注的一类。理解这一点，你看 <span class="mono">ggml/src</span> 下那一长串
<span class="mono">ggml-cpu</span>、<span class="mono">ggml-cuda</span>、<span class="mono">ggml-blas</span>、<span class="mono">ggml-metal</span>…… 目录时就不会困惑了。</p>

<h2>怎么编：CMake 两步走</h2>
<p>llama.cpp 用 <strong>CMake</strong> 作为构建系统。从源码编译，标准流程就两步：先<strong>配置（configure）</strong>，再<strong>构建（build）</strong>。</p>
<p>当然，最最开始还有一步别漏了：先把源码<strong>克隆</strong>下来——<span class="mono">git clone</span> 仓库地址、进到目录，再走下面那两步 CMake 就行。
想要某个稳定版本可以 checkout 对应的发布 tag；想跟最新进展就用默认的主分支。整个"<strong>克隆 -&gt; 配置 -&gt; 构建 -&gt; 运行</strong>"四步，
就是绝大多数人上手 llama.cpp 的完整路径。</p>
<div class="flow">
  <div class="node"><div class="nt">cmake -B build</div><div class="nd">配置: 探测 + 选后端</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">cmake --build build</div><div class="nd">编译源码</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">build/bin/</div><div class="nd">库 + 可执行程序</div></div>
</div>
<pre class="code"><span class="cm"># 仅 CPU (默认)</span>
cmake -B build
cmake --build build --config Release -j

<span class="cm"># 带 NVIDIA CUDA</span>
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j</pre>
<p>第一步 <span class="mono">cmake -B build</span> 是<strong>配置</strong>：CMake 会探测你的系统（有没有 CUDA 工具链、什么编译器、什么 CPU 指令集），根据你给的
<span class="mono">-D</span> 选项决定要编哪些后端，然后在 <span class="mono">build/</span> 目录里生成真正的构建文件。第二步 <span class="mono">cmake --build build</span> 才是
<strong>真正编译</strong>，把源码变成库和可执行文件。<span class="mono">-j</span> 让它多核并行编、快很多；<span class="mono">--config Release</span> 表示编优化过的发布版
（而非带调试信息、慢得多的 Debug 版）。整个过程对照前面那张流程图看，就很清楚了。</p>
<div class="card spark">
  <div class="tag">💡 实战</div>
  第一次完整编译可能要几分钟到十几分钟（尤其带上 CUDA 这种大后端）；好在 CMake 支持<strong>增量编译</strong>——你改一两个文件后再<span class="mono">cmake --build build</span>，它只重编受影响的部分，通常几秒就好。想更快，可以让 CMake 用 <strong>Ninja</strong> 作为底层构建工具（<span class="mono">cmake -G Ninja -B build</span>），它的并行调度比传统 Make 更高效。这些都是"配置一次、反复构建"的日常。
</div>
<p>顺便说：<strong>大多数人其实不用自己编</strong>。llama.cpp 官方在 GitHub Releases 提供了各平台的<strong>预编译包</strong>，下载解压即用；很多上层项目
（如 Ollama、LM Studio）也都内置了它。那什么时候才需要自己从源码编？——当你要<strong>打开某个预编译包没带的后端</strong>（比如针对你这张特定显卡的 CUDA）、
要<strong>用最新的开发版功能</strong>、或者要<strong>把库嵌进自己的程序</strong>时。这一课讲的就是这后一种"需要动手"的场景。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  再说说 <span class="mono">--config Release</span> 为什么重要。编译器在 Release 档会开启大量优化（向量化、内联、去掉断言和调试信息），跑起来可能比 Debug 档<strong>快好几倍</strong>。除非你在<strong>调试 llama.cpp 本身</strong>（要单步、看变量），否则平时一律用 Release。这也是为什么官方预编译包都是 Release 版——对"只是想跑模型"的人来说，没理由忍受 Debug 的慢。
</div>
<p>真正动手前，强烈建议先扫一眼仓库里的 <span class="mono">docs/build.md</span>：它把每个平台、每种后端的具体编译命令和注意事项都列全了
（包括 Windows、各家 GPU 的细节）。这一课给你的是"地图和直觉"，而 <span class="mono">docs/build.md</span> 是"逐条的操作手册"，两者配合着看，第一次编译就能少走很多弯路，遇到平台特有的问题也多半能在那里找到答案。</p>

<h2>选后端：CMake 选项一览</h2>
<p>要不要某个 GPU 后端，就靠配置时的一个 <span class="mono">-D</span> 开关。常用的几个列在下面：</p>
<table class="t">
  <tr><th>CMake 选项</th><th>启用的硬件 / 功能</th></tr>
  <tr><td><span class="mono">GGML_CPU</span>（默认 ON）</td><td>CPU 后端（自动用 AVX / NEON 等 SIMD）</td></tr>
  <tr><td><span class="mono">GGML_CUDA</span></td><td>NVIDIA GPU</td></tr>
  <tr><td><span class="mono">GGML_HIP</span></td><td>AMD GPU（ROCm）</td></tr>
  <tr><td><span class="mono">GGML_METAL</span></td><td>Apple GPU（macOS 上常默认 ON）</td></tr>
  <tr><td><span class="mono">GGML_VULKAN</span></td><td>跨厂商 GPU（含部分集显）</td></tr>
  <tr><td><span class="mono">GGML_SYCL</span></td><td>Intel GPU</td></tr>
  <tr><td><span class="mono">GGML_BLAS</span></td><td>用 BLAS 库加速大矩阵乘</td></tr>
</table>
<pre class="code">option(GGML_CPU    "ggml: enable CPU backend" ON)
option(GGML_CUDA   "ggml: use CUDA"           OFF)
option(GGML_METAL  "ggml: use Metal"          ...)
option(GGML_VULKAN "ggml: use Vulkan"         OFF)
<span class="cm"># ... HIP / SYCL / OPENCL / BLAS, 都在 ggml/CMakeLists.txt</span></pre>
<p>这些开关都定义在 <span class="mono">ggml/CMakeLists.txt</span> 里。注意 <strong>CPU 后端默认就是开的</strong>（<span class="mono">GGML_CPU=ON</span>），
所以你什么都不加，也能得到一个纯 CPU 能跑的 llama.cpp；GPU 后端则默认关闭，要哪个就显式 <span class="mono">-D...=ON</span> 打开。苹果设备上 Metal 通常默认开。
你也可以同时开多个后端，运行时再决定用哪个。</p>
<p>自己编最常见的坑，几乎都出在 GPU 工具链上。比如开了 <span class="mono">-DGGML_CUDA=ON</span> 却没装好 CUDA Toolkit、或者 CUDA 版本和显卡驱动对不上，
配置阶段就会报错——这其实是好事，<strong>CMake 在"配置"时就帮你把环境问题暴露出来了</strong>，省得编到一半才失败。遇到报错别慌，先看它提示缺什么：
缺 nvcc 就装 CUDA Toolkit、缺某个库就按提示装，多数问题照着错误信息走一遍就能解决。</p>

<h2>编完有什么：产物一览</h2>
<p>编译完成后，所有产物都落在 <span class="mono">build/bin</span> 目录里，分两类：</p>
<div class="cellgroup">
  <div class="cg-cap"><b>build/bin 产物</b>：分"库"和"可执行程序"两类</div>
  <div class="cells"><span class="lab">库</span><span class="cell">libggml</span><span class="cell">libllama</span><span class="lab">引擎本体, 可被链接</span></div>
  <div class="cells"><span class="lab">程序</span><span class="cell hl">llama-cli</span><span class="cell hl">llama-server</span><span class="cell">llama-quantize</span><span class="cell">llama-bench</span><span class="cell">llama-perplexity</span></div>
</div>
<p>一类是<strong>库</strong>：<span class="mono">libggml</span>（张量引擎）和 <span class="mono">libllama</span>（推理库），它们是"引擎本体"，可以被别的程序链接调用——
课 01 说的"可嵌入"，靠的就是它们。另一类是<strong>可执行程序</strong>，就是你平时直接用的命令：<span class="mono">llama-cli</span>（命令行对话）、
<span class="mono">llama-server</span>（起一个 HTTP 服务）、<span class="mono">llama-quantize</span>（课 06 用过的量化工具）、<span class="mono">llama-bench</span>（测速）、
<span class="mono">llama-perplexity</span>（测质量）等等。</p>
<div class="card detail">
  <div class="tag">🔬 细节 / 源码对应</div>
  前面说库可以"被别的程序链接"，具体怎么用？你的程序只要<strong>包含 <span class="mono">include/llama.h</span> 这个头文件、再链接上 <span class="mono">libllama</span></strong>，就能调用课 01 里那套 C API 来加载模型、跑推理。库可以编成<strong>静态库</strong>（直接打包进你的可执行文件，部署时是单个文件）或<strong>动态库</strong>（运行时再加载，多个程序可共享）。课 01 强调的"零依赖单文件可执行"，正是把 libllama 和后端<strong>静态链接</strong>进 llama-cli 的结果。
</div>
<p>这些可执行程序里，最常打交道的是两个：<span class="mono">llama-cli</span> 适合<strong>在命令行里快速试一把</strong>或写脚本；<span class="mono">llama-server</span>
则会起一个<strong>常驻的 HTTP 服务</strong>，对外提供和 OpenAI 兼容的接口，适合给前端、应用或其它服务调用——你常用的各种本地大模型 App，背后往往就是它。
两者用的是同一套 <span class="mono">libllama</span>，只是把"入口"包装成了不同形态。</p>
<p>除了主角库和那几个常用程序，<span class="mono">build/bin</span> 里其实还会有一堆小工具和示例：<span class="mono">llama-gguf</span>（查看 / 操作 GGUF 文件）、
<span class="mono">llama-tokenize</span>（单独试分词）、各种 <span class="mono">test-*</span> 测试程序，以及 <span class="mono">examples/</span> 下编出来的演示。平时用不到不用管，
但当你想深入某个细节（比如"这个模型到底被分成哪些张量"）时，往往能在这里找到一个正好趁手的小工具。</p>
<pre class="code"><span class="cm"># 跑起来: -ngl 把多少层卸载到 GPU</span>
./build/bin/llama-cli -m model.gguf -p "你好" -ngl 99</pre>
<div class="card spark">
  <div class="tag">💡 实战</div>
  这里的 <span class="mono">-ngl</span>（即 <span class="mono">n-gpu-layers</span>）很关键：它决定把模型的多少层放到 GPU 上算、其余留在 CPU。显存够就尽量多放（<span class="mono">-ngl 99</span> 基本是"能放的全放"），显存不够就只放一部分，CPU / GPU 混合跑。这也正呼应前面的后端抽象：<strong>同一个模型、同一张图，能灵活地切在不同硬件上算</strong>。
</div>
<p>怎么确认 GPU 后端真的编进去、也真的用上了？最简单的办法是看<strong>启动日志</strong>。跑 <span class="mono">llama-cli</span> 时，它会打印检测到的设备和每层的分配情况——
如果你看到类似 "offloaded 33/33 layers to GPU" 的字样，就说明层确实卸载到 GPU 上了。要是发现还在纯 CPU 跑，多半是 <span class="mono">-ngl</span> 没加、
或者那个后端根本没编进去，回到配置那一步检查 <span class="mono">-D</span> 选项即可。</p>
<p>把这一课和课 06 连起来看：你选的<strong>量化格式</strong>和你编的<strong>后端</strong>是要<strong>配合</strong>的。每个后端都为常见量化格式（如 Q4_K、Q8_0）写了专门的
解量化 + 矩阵乘内核，能直接吃量化权重、在算的时候顺手解量化。所以"<strong>选什么量化</strong>"和"<strong>用什么后端</strong>"共同决定了你的实际速度——
这也是为什么第六部分会专门去看这些内核到底怎么写。</p>
<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  因为后端是可插拔的、依赖又轻，llama.cpp 还能<strong>交叉编译</strong>到很多目标上：编成安卓 / iOS 的库塞进手机 App、编成 WebAssembly 跑在浏览器里、甚至编到树莓派这类小板子上。换一个目标平台，往往只是换一套工具链、调一调 CMake 选项的事，核心代码不用动。这正是"<strong>用 C/C++ + 后端抽象</strong>"换来的可移植红利——课 01 说的"到处跑"，在构建层面就是这么实现的。
</div>
<p>关于"可移植性"再多说一句，这里有个容易搞反的点。<strong>从源码自己编</strong>时，ggml 默认会打开 <span class="mono">GGML_NATIVE</span>（针对你这台机器的指令集做优化），
所以默认编出来的二进制是"<strong>本机特化</strong>"的、跑得快，但<strong>不保证能拷到别的机器上跑</strong>（换台 CPU 可能直接"非法指令"崩掉）。反倒是<strong>官方预编译包</strong>
为了"谁都能用"，特意关掉 native、改用更通用且运行时自适应的指令，所以它们才是<strong>可移植</strong>的那一档。这就是"<strong>预编译求通用、自编可求性能</strong>"的取舍——
想兼顾可移植，自己编时把 <span class="mono">GGML_NATIVE</span> 关掉即可。</p>
<p>最后给个动手的起点：想看"怎么用 libllama 写自己的程序"，仓库里的 <span class="mono">examples/simple</span> 是最好的入口——约两百行 C++ 就走完了加载模型、
分词、跑解码循环、输出文字的全过程，正好是课 03 那条主线的可运行版本。把它读懂、改一改，你就算真正上手 llama.cpp 的 API 了。</p>
<p>那为什么不干脆把<strong>所有后端</strong>都编进一个二进制、运行时谁有用谁？因为代价很大：每个 GPU 后端都<strong>拖着一大坨依赖</strong>（CUDA 要 CUDA 运行库、
Vulkan 要 Vulkan SDK……），全编进来体积暴涨、还要求目标机器装齐这些库——这恰恰违背了 llama.cpp"<strong>零依赖、轻量</strong>"的初心。所以它选择让你
<strong>按需挑选</strong>：纯 CPU 版可以小到拷哪都能跑，要加速时再单独编一个带某后端的版本。这正是本课末尾思考题的答案方向。</p>
<div class="card warn">
  <div class="tag">⚠ 注意</div>
  顺便厘清一个容易混的点：<strong>有些东西是编译期定的，有些是运行期定的</strong>。"<strong>支持哪些硬件后端</strong>"是编译期用 <span class="mono">-D</span> 选项定死的；而"<strong>用多大上下文、放几层到 GPU、几个线程、用什么采样</strong>"这些都是<strong>运行时</strong>的命令行参数，换一换不用重编。搞清这条界线，你就不会犯"想换个上下文长度还跑去重新编译"这种冤枉错。
</div>

<h2>深入一点（选读）</h2>
<p class="acc-intro">下面三个问题，想深究的同学点开看；只想抓主线的可以先跳过。</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> CPU 后端也要选吗？SIMD / BLAS 是什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>CPU 后端默认就开，你基本不用管它。它会<strong>自动探测并用上 CPU 的 SIMD 指令</strong>（x86 的 AVX、ARM 的 NEON 等）来加速矩阵运算——这是现代 CPU 上几乎免费的并行算力。</p>
    <p>你还可以选装 <span class="mono">GGML_BLAS</span>，用成熟的 BLAS 数学库进一步加速大矩阵乘（对 prefill 阶段帮助明显）。追求极致的人会用 <span class="mono">-march=native</span>
    让编译器针对你这台机器的指令集优化，但这样编出来的二进制就<strong>不能拷到别的机器</strong>用了——这又是一处"性能 vs 可移植"的老权衡。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么用 CMake，而不是手写 Makefile？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为 llama.cpp 要支持的平台和硬件太多了：Linux / macOS / Windows，CPU / CUDA / Metal / Vulkan / ROCm / SYCL……手写 Makefile 根本管不过来。</p>
    <p>CMake 的价值在于<strong>跨平台</strong>和<strong>自动探测</strong>：它能找到你系统里装的 CUDA、判断编译器支持哪些指令、再生成对应平台的构建文件
    （Linux 上是 Make 或 Ninja、Windows 上是 Visual Studio 工程）。项目早期其实有手写的 Makefile，但随着后端越来越多，现在已经<strong>统一以 CMake 为主</strong>。</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> 运行时怎么决定用哪个后端？多 GPU 怎么办？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>编译时可以把多个后端都编进来；<strong>运行时</strong>，ggml 有一个后端"注册表"，会枚举出当前机器上实际可用的设备（比如检测到一张 NVIDIA 卡）。
    <span class="mono">-ngl</span> 决定多少层放 GPU。</p>
    <p>如果有<strong>多张 GPU</strong>，还能按层或按张量把模型<strong>切分到几张卡</strong>上一起算（用 <span class="mono">--split-mode</span> 等参数）。
    所以"装哪些后端"是编译期的事，"具体用哪个、用几张卡"是运行期的事，两者分开，灵活又清晰。</p>
  </div>
</details>

<p>到这里，第二部分的四块基础就拼齐了：课 04 讲清了<strong>模型在算什么</strong>（decoder-only、注意力、KV cache），课 05 讲清了<strong>数据怎么表示</strong>
（张量、shape / stride），课 06 讲清了<strong>权重怎么压</strong>（量化），这一课讲清了<strong>引擎怎么编、怎么挑硬件</strong>（构建与后端）。有了这四块垫底，
第三部分我们就能放心地钻进 ggml 引擎内部，去看计算图、内存池、算子这些"机器零件"到底是怎么转起来的了。可以说，第二部分是"地基"，第三部分才开始盖"主楼"。</p>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>ggml 的<strong>后端抽象</strong>（<span class="mono">ggml-backend.h</span>）把"算什么（计算图）"和"在哪算（CPU / GPU）"解耦；加新硬件只需加一个后端。</li>
    <li>编译两步：<span class="mono">cmake -B build [-D 选项]</span> 配置，<span class="mono">cmake --build build</span> 构建。</li>
    <li>GPU 后端靠 <span class="mono">-DGGML_CUDA / METAL / VULKAN / HIP ...=ON</span> 在配置时打开；<strong>CPU 后端默认就开</strong>。</li>
    <li>产物在 <span class="mono">build/bin</span>：库（<span class="mono">libllama</span> / <span class="mono">libggml</span>）+ 程序（<span class="mono">llama-cli</span> / <span class="mono">llama-server</span> / <span class="mono">llama-quantize</span> ……）。</li>
    <li>运行时 <span class="mono">-ngl</span> 控制把多少层卸载到 GPU。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  把"后端"做成<strong>可插拔的编译期开关</strong>——同一份 ggml / llama 源码，既能编出一个零依赖、到处能跑的纯 CPU 版，也能编出榨干 CUDA / Metal 的加速版。
  "<strong>零依赖到处跑</strong>"和"<strong>有 GPU 就尽情加速</strong>"这两个看似矛盾的目标，就靠这套构建 + 后端体系优雅地同时满足了。这也正是第二部分的收尾——你已经备齐了读懂 ggml 引擎内部的全部基础。从下一部分起，我们会把 ggml 这台引擎彻底拆开——计算图怎么建、内存池怎么管、算子怎么算，
  一个零件一个零件地看个明白。打好了这一层地基，再往上盖楼就稳了。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
The last few lessons were about "what things are"; this one is about "<strong>how to build it, how to pick a hardware backend, and what the outputs are</strong>".
After this you can build a GPU-accelerated llama.cpp from source yourself, and you will see why one codebase runs on CPU, NVIDIA, Apple, and AMD hardware alike.
This lesson also closes Part 2.
</p>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  CMake is like a <strong>general contractor</strong>: you tick boxes on the order ("underfloor heating (CUDA)? central air (Metal / Vulkan)?"), and it lines up the
  right crews (compilers and each vendor's GPU toolchain), draws the blueprints (generates build files), and hands over a move-in-ready house (executables). You need
  not know each crew's details - just how to place the order, and this lesson teaches you how.
</div>

<h2>Backend abstraction: one compute graph, many kinds of hardware</h2>
<p>As lesson 03 noted, an inference is described as a <strong>compute graph</strong> (a pile of operators: matmul, softmax, rope...). But the same "matmul" is written
with SIMD on CPU, with CUDA on NVIDIA, with Metal on Apple - wildly different implementations. If the upper inference logic had to worry about all this, the code would
be a mess.</p>
<p>ggml's answer is a uniform <strong>backend interface</strong> (<span class="mono">ggml/include/ggml-backend.h</span>): the upper layer only says "compute this graph",
while "on which hardware, with which instructions" is left to a concrete backend (<span class="mono">ggml-cpu</span>, <span class="mono">ggml-cuda</span>,
<span class="mono">ggml-metal</span>...). This is exactly the "<strong>decouple 'what to compute' from 'where to compute'</strong>" that lesson 01 kept stressing.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">upper</span><span class="name">inference logic / compute graph</span></div><div class="ld">lesson 03's "graph building": only describes what to compute (matmul · softmax · rope ...)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">interface</span><span class="name">ggml-backend</span></div><div class="ld">uniform backend interface: allocate memory, schedule ops, move data between devices</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">impl</span><span class="name">ggml-cpu · ggml-cuda · ggml-metal · ggml-vulkan …</span></div><div class="ld">one implementation per hardware, actually computing the ops</div></div>
</div>
<p>The benefit of this layering: <strong>adding a new hardware needs only a new backend; not one line of upper inference code changes</strong>; and which backends you
pick at build time decides which hardware your binary "knows". So "build" and "backend" are two sides of the same coin - the build system's main job is to compile in
the backends you chose.</p>
<p>One more mechanism: each backend, when compiled in, <strong>registers itself</strong> with ggml, declaring "which ops I can compute, which memory I manage". At
runtime the scheduler (the sched in <span class="mono">ggml-backend</span>) takes the graph and <strong>dispatches each op to a suitable backend</strong>, moving data
between CPU and GPU memory as needed. If some op is not yet implemented in the GPU backend, the scheduler can usually <strong>fall back to CPU</strong> for that step -
so even an unsupported new op will not break the whole graph, it just runs that step a bit slower.</p>
<p>One more note: <strong>"backend" is not only GPUs</strong>. The CPU is itself a backend; BLAS is a library backend that speeds big matmuls on CPU; Apple's Accelerate
framework can plug in too. So "backend" is more precisely "<strong>a way to actually carry out the ops</strong>", with GPUs being the most-discussed kind. Grasp this and
the long list of <span class="mono">ggml-cpu</span>, <span class="mono">ggml-cuda</span>, <span class="mono">ggml-blas</span>, <span class="mono">ggml-metal</span>...
directories under <span class="mono">ggml/src</span> will not confuse you.</p>

<h2>How to build: two-step CMake</h2>
<p>llama.cpp uses <strong>CMake</strong> as its build system. From source the standard flow is two steps: first <strong>configure</strong>, then <strong>build</strong>.</p>
<p>Of course, there is one step before all this: <strong>clone</strong> the source - <span class="mono">git clone</span> the repo, cd in, then run the two CMake steps
below. For a stable version, checkout the matching release tag; to track the latest, use the default main branch. The whole "<strong>clone -&gt; configure -&gt; build
-&gt; run</strong>" four-step is how most people get started with llama.cpp.</p>
<div class="flow">
  <div class="node"><div class="nt">cmake -B build</div><div class="nd">configure: probe + pick backends</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node hl"><div class="nt">cmake --build build</div><div class="nd">compile sources</div></div>
  <div class="arrow">-&gt;</div>
  <div class="node"><div class="nt">build/bin/</div><div class="nd">libraries + executables</div></div>
</div>
<pre class="code"><span class="cm"># CPU only (default)</span>
cmake -B build
cmake --build build --config Release -j

<span class="cm"># with NVIDIA CUDA</span>
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j</pre>
<p>Step one <span class="mono">cmake -B build</span> is <strong>configure</strong>: CMake probes your system (is there a CUDA toolchain, which compiler, which CPU
instruction set), decides which backends to build from your <span class="mono">-D</span> options, and generates the real build files in <span class="mono">build/</span>.
Step two <span class="mono">cmake --build build</span> is the <strong>actual compile</strong>, turning sources into libraries and executables. <span class="mono">-j</span>
builds in parallel across cores (much faster); <span class="mono">--config Release</span> means an optimized release build (not the much slower, debug-info Debug build).
Read it against the flow diagram above and it is clear.</p>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  A first full build may take a few to a dozen-plus minutes (especially with a big backend like CUDA); fortunately CMake supports <strong>incremental builds</strong> - after changing a file or two, <span class="mono">cmake --build build</span> only recompiles what is affected, usually in seconds. For more speed, have CMake use <strong>Ninja</strong> as the underlying build tool (<span class="mono">cmake -G Ninja -B build</span>), whose parallel scheduling beats classic Make. This is the daily "configure once, build repeatedly".
</div>
<p>By the way: <strong>most people do not build at all</strong>. The llama.cpp project ships <strong>prebuilt packages</strong> per platform on GitHub Releases -
download, unzip, run; many higher-level projects (Ollama, LM Studio) bundle it too. So when do you build from source? When you need <strong>a backend the prebuilt
package lacks</strong> (e.g. CUDA for your specific card), <strong>the latest dev-branch features</strong>, or to <strong>embed the library into your own
program</strong>. This lesson is about that hands-on case.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  More on why <span class="mono">--config Release</span> matters. In Release the compiler turns on heavy optimization (vectorization, inlining, dropping asserts and debug info), often running <strong>several times faster</strong> than Debug. Unless you are <strong>debugging llama.cpp itself</strong> (stepping, inspecting variables), always use Release. That is why official prebuilt packages are Release - for someone who "just wants to run a model", there is no reason to suffer Debug's slowness.
</div>
<p>Before getting hands-on, skim the repo's <span class="mono">docs/build.md</span>: it lists the exact build commands and caveats for every platform and backend
(including Windows and each GPU's details). This lesson gives you "the map and the intuition", while <span class="mono">docs/build.md</span> is "the step-by-step
manual"; reading both together saves a lot of first-build detours, and most platform-specific snags have an answer there.</p>

<h2>Picking backends: a tour of the CMake options</h2>
<p>Whether you want a given GPU backend comes down to one <span class="mono">-D</span> switch at configure time. The common ones:</p>
<table class="t">
  <tr><th>CMake option</th><th>hardware / feature enabled</th></tr>
  <tr><td><span class="mono">GGML_CPU</span> (default ON)</td><td>CPU backend (auto-uses AVX / NEON SIMD)</td></tr>
  <tr><td><span class="mono">GGML_CUDA</span></td><td>NVIDIA GPU</td></tr>
  <tr><td><span class="mono">GGML_HIP</span></td><td>AMD GPU (ROCm)</td></tr>
  <tr><td><span class="mono">GGML_METAL</span></td><td>Apple GPU (often default ON on macOS)</td></tr>
  <tr><td><span class="mono">GGML_VULKAN</span></td><td>cross-vendor GPU (incl. some integrated)</td></tr>
  <tr><td><span class="mono">GGML_SYCL</span></td><td>Intel GPU</td></tr>
  <tr><td><span class="mono">GGML_BLAS</span></td><td>use a BLAS library to speed big matmuls</td></tr>
</table>
<pre class="code">option(GGML_CPU    "ggml: enable CPU backend" ON)
option(GGML_CUDA   "ggml: use CUDA"           OFF)
option(GGML_METAL  "ggml: use Metal"          ...)
option(GGML_VULKAN "ggml: use Vulkan"         OFF)
<span class="cm"># ... HIP / SYCL / OPENCL / BLAS, all in ggml/CMakeLists.txt</span></pre>
<p>These switches all live in <span class="mono">ggml/CMakeLists.txt</span>. Note the <strong>CPU backend is on by default</strong> (<span class="mono">GGML_CPU=ON</span>),
so with nothing extra you still get a CPU-runnable llama.cpp; GPU backends default off, so turn on whichever you want with an explicit <span class="mono">-D...=ON</span>.
On Apple devices Metal is usually on by default. You can also enable several backends at once and decide which to use at runtime.</p>
<p>The most common pitfall when building yourself is almost always the GPU toolchain. Turning on <span class="mono">-DGGML_CUDA=ON</span> without a proper CUDA Toolkit, or
a CUDA version mismatched with your driver, errors out at configure - which is actually good: <strong>CMake surfaces the environment problem at "configure" time</strong>,
instead of failing halfway through compiling. Don't panic at an error; read what it says is missing: no nvcc means install the CUDA Toolkit, a missing library means
install it as prompted - most issues resolve by following the error message.</p>

<h2>What you get: a tour of the outputs</h2>
<p>After compiling, all outputs land in <span class="mono">build/bin</span>, in two kinds:</p>
<div class="cellgroup">
  <div class="cg-cap"><b>build/bin outputs</b>: split into "libraries" and "executables"</div>
  <div class="cells"><span class="lab">libs</span><span class="cell">libggml</span><span class="cell">libllama</span><span class="lab">the engine itself, linkable</span></div>
  <div class="cells"><span class="lab">programs</span><span class="cell hl">llama-cli</span><span class="cell hl">llama-server</span><span class="cell">llama-quantize</span><span class="cell">llama-bench</span><span class="cell">llama-perplexity</span></div>
</div>
<p>One kind is <strong>libraries</strong>: <span class="mono">libggml</span> (the tensor engine) and <span class="mono">libllama</span> (the inference library) - the "engine
itself", which other programs can link against; lesson 01's "embeddable" rests on these. The other kind is <strong>executables</strong>, the commands you use directly:
<span class="mono">llama-cli</span> (command-line chat), <span class="mono">llama-server</span> (starts an HTTP service), <span class="mono">llama-quantize</span> (the
quantizer from lesson 06), <span class="mono">llama-bench</span> (speed), <span class="mono">llama-perplexity</span> (quality), and more.</p>
<div class="card detail">
  <div class="tag">🔬 Details / source</div>
  We said the libraries can "be linked by other programs" - how exactly? Your program just <strong>includes the <span class="mono">include/llama.h</span> header and links <span class="mono">libllama</span></strong> to call the C API from lesson 01 to load a model and run inference. The libraries can be built <strong>static</strong> (baked into your executable, a single file to deploy) or <strong>dynamic</strong> (loaded at runtime, shareable across programs). Lesson 01's "zero-dependency single-file executable" is exactly the result of <strong>statically linking</strong> libllama and the backends into llama-cli.
</div>
<p>Of these executables, two you will use most: <span class="mono">llama-cli</span> suits <strong>a quick command-line try</strong> or scripting;
<span class="mono">llama-server</span> starts a <strong>long-running HTTP service</strong> with an OpenAI-compatible API, for front-ends, apps, or other services to call -
the local-LLM apps you use are often it under the hood. Both use the same <span class="mono">libllama</span>, just wrapping the "entry point" in different forms.</p>
<p>Besides the star libraries and those common programs, <span class="mono">build/bin</span> also holds a pile of small tools and demos: <span class="mono">llama-gguf</span>
(inspect / manipulate GGUF files), <span class="mono">llama-tokenize</span> (try tokenization alone), various <span class="mono">test-*</span> programs, and the demos built
from <span class="mono">examples/</span>. You can ignore them day to day, but when you want to dig into a detail (e.g. "which tensors is this model split into") there is
often a handy little tool right here.</p>
<pre class="code"><span class="cm"># run it: -ngl offloads how many layers to the GPU</span>
./build/bin/llama-cli -m model.gguf -p "Hello" -ngl 99</pre>
<div class="card spark">
  <div class="tag">💡 Tip</div>
  Here <span class="mono">-ngl</span> (i.e. <span class="mono">n-gpu-layers</span>) is key: it decides how many of the model's layers run on the GPU, the rest on CPU. With enough VRAM put as many as you can (<span class="mono">-ngl 99</span> is basically "all that fit"); with too little, put only some and run CPU / GPU mixed. This echoes the backend abstraction: <strong>the same model, the same graph, flexibly split across different hardware</strong>.
</div>
<p>How do you confirm the GPU backend was really compiled in and is really being used? The simplest way is the <strong>startup log</strong>. When running
<span class="mono">llama-cli</span> it prints the detected devices and per-layer placement - if you see something like "offloaded 33/33 layers to GPU", layers really went
to the GPU. If it is still CPU-only, likely <span class="mono">-ngl</span> was omitted or that backend was not compiled in; go back to configure and check the
<span class="mono">-D</span> options.</p>
<p>Tying this lesson to lesson 06: the <strong>quantization format</strong> you pick and the <strong>backend</strong> you build must <strong>work together</strong>. Each
backend has dedicated dequant + matmul kernels for common quant formats (Q4_K, Q8_0), consuming quantized weights directly and dequantizing on the fly. So "<strong>which
quantization</strong>" and "<strong>which backend</strong>" jointly decide your real-world speed - which is why Part 6 goes to look at how those kernels are actually written.</p>
<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Because backends are pluggable and dependencies are light, llama.cpp can also <strong>cross-compile</strong> to many targets: a library for Android / iOS apps, WebAssembly to run in the browser, even small boards like a Raspberry Pi. Switching target platform is often just a different toolchain and a few CMake tweaks; the core code stays put. This is the portability dividend of "<strong>C/C++ plus a backend abstraction</strong>" - lesson 01's "run anywhere", realized at the build level.
</div>
<p>One more word on "portability", with an easy-to-get-backwards point. When you <strong>build from source</strong>, ggml turns on <span class="mono">GGML_NATIVE</span> by
default (optimizing for your machine's instruction set), so the default binary is <strong>machine-tuned</strong> and fast, but <strong>not guaranteed to run on other
machines</strong> (a different CPU may crash with "illegal instruction"). It is the <strong>official prebuilt packages</strong> that, to be "usable by everyone", turn
native off and use more generic, runtime-adaptive instructions - so those are the <strong>portable</strong> ones. That is the trade: <strong>prebuilt for portability,
self-build for performance</strong> - and if you want both, just turn <span class="mono">GGML_NATIVE</span> off when building.</p>
<p>A hands-on starting point: to see "how to write your own program with libllama", the repo's <span class="mono">examples/simple</span> is the best entry - around a couple hundred
lines of C++ walk the whole path of loading a model, tokenizing, running the decode loop, and printing text, a runnable version of lesson 03's main line. Read it, tweak
it, and you have truly started using the llama.cpp API.</p>
<p>So why not just compile <strong>all backends</strong> into one binary and let runtime pick? Because the cost is high: each GPU backend <strong>drags a big pile of
dependencies</strong> (CUDA needs the CUDA runtime, Vulkan the Vulkan SDK...), so compiling them all balloons the size and demands the target machine have all those
libraries - which contradicts llama.cpp's "<strong>zero-dependency, lightweight</strong>" ethos. So it lets you <strong>pick what you need</strong>: a CPU-only build can
be tiny and copy-anywhere, and you build a backend-specific version separately when you want acceleration. That is the direction of this lesson's closing question.</p>
<div class="card warn">
  <div class="tag">⚠ Heads-up</div>
  One easily-confused point: <strong>some things are set at compile time, others at runtime</strong>. "<strong>Which hardware backends</strong>" is fixed at compile time with <span class="mono">-D</span> options; but "<strong>context size, how many layers on GPU, thread count, sampling</strong>" are all <strong>runtime</strong> command-line flags, changeable without recompiling. Get this line straight and you will not make the mistake of "rebuilding just to change the context length".
</div>

<h2>Going deeper (optional)</h2>
<p class="acc-intro">Three questions below; open them if you want depth, skip them if you only want the main line.</p>

<details class="accordion">
  <summary><span class="badge-num">1</span> Do I even pick the CPU backend? What are SIMD / BLAS? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The CPU backend is on by default; you mostly need not touch it. It <strong>auto-detects and uses your CPU's SIMD instructions</strong> (AVX on x86, NEON on ARM)
    to speed up matrix math - nearly free parallel compute on modern CPUs.</p>
    <p>You can also opt into <span class="mono">GGML_BLAS</span> to further speed big matmuls with a mature BLAS library (noticeably helps prefill). The extreme route is
    <span class="mono">-march=native</span>, letting the compiler optimize for your exact instruction set - but the resulting binary <strong>cannot be copied to another
    machine</strong>. Another classic "performance vs portability" trade-off.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">2</span> Why CMake instead of a hand-written Makefile? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Because llama.cpp must support too many platforms and hardware: Linux / macOS / Windows, CPU / CUDA / Metal / Vulkan / ROCm / SYCL... a hand-written Makefile
    simply cannot keep up.</p>
    <p>CMake's value is <strong>cross-platform</strong> and <strong>auto-detection</strong>: it finds your installed CUDA, checks which instructions the compiler supports,
    and generates the right build files per platform (Make or Ninja on Linux, a Visual Studio project on Windows). The project did have a hand-written Makefile early on,
    but as backends multiplied it has <strong>standardized on CMake</strong>.</p>
  </div>
</details>

<details class="accordion">
  <summary><span class="badge-num">3</span> How is the backend chosen at runtime? What about multiple GPUs? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>You can compile several backends in; <strong>at runtime</strong>, ggml has a backend "registry" that enumerates the devices actually available on the machine
    (e.g. it detects an NVIDIA card). <span class="mono">-ngl</span> decides how many layers go on the GPU.</p>
    <p>With <strong>multiple GPUs</strong>, you can <strong>split the model across cards</strong> by layer or by tensor (via parameters like
    <span class="mono">--split-mode</span>). So "which backends to compile" is a build-time matter, "which to use and across how many cards" a runtime matter - kept
    separate, flexible and clear.</p>
  </div>
</details>

<p>And with that, Part 2's four foundations are complete: lesson 04 clarified <strong>what the model computes</strong> (decoder-only, attention, KV cache), lesson 05
<strong>how data is represented</strong> (tensors, shape/stride), lesson 06 <strong>how weights are compressed</strong> (quantization), and this one <strong>how the engine
is built and how hardware is chosen</strong> (build &amp; backends). With these four underneath, Part 3 can confidently dive into the ggml engine to see how the compute
graph, memory pool, and operators - the "machine parts" - actually turn. Part 2 is the "foundation"; Part 3 starts building the "main floors".</p>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>ggml's <strong>backend abstraction</strong> (<span class="mono">ggml-backend.h</span>) decouples "what to compute (the graph)" from "where (CPU / GPU)"; new hardware just needs a new backend.</li>
    <li>Two build steps: <span class="mono">cmake -B build [-D options]</span> to configure, <span class="mono">cmake --build build</span> to build.</li>
    <li>GPU backends are turned on at configure time with <span class="mono">-DGGML_CUDA / METAL / VULKAN / HIP ...=ON</span>; the <strong>CPU backend is on by default</strong>.</li>
    <li>Outputs are in <span class="mono">build/bin</span>: libraries (<span class="mono">libllama</span> / <span class="mono">libggml</span>) + programs (<span class="mono">llama-cli</span> / <span class="mono">llama-server</span> / <span class="mono">llama-quantize</span> ...).</li>
    <li>At runtime <span class="mono">-ngl</span> controls how many layers are offloaded to the GPU.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Making "backends" a <strong>pluggable compile-time switch</strong> - the same ggml / llama source can build a zero-dependency, run-anywhere CPU-only binary, or a
  CUDA / Metal-accelerated one. The two seemingly contradictory goals, "<strong>run anywhere with zero dependencies</strong>" and "<strong>go all-out when a GPU is
  present</strong>", are met at once, elegantly, by this build-plus-backend system. And that closes Part 2 - you now have all the groundwork to read the ggml engine internals.
  From the next part on, we take the ggml engine fully apart - how the compute graph is built, how the memory pool is managed, how operators compute - part by part.
  With this foundation laid, building upward is solid.
</div>
""",
}
