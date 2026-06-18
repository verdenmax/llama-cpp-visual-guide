"""Content for Part 7 (advanced topics)."""

LESSON_34 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
推理时模型一次只吐一个 token：把整个序列前向一遍、采样出下一个、再接上去前向一遍……一步接一步，纯串行。而上一部分刚说过（L30/L32），decode 阶段是<strong>访存密集</strong>的——大部分时间花在把几个 GB 的权重从显存搬进来，真正的乘加反而没占满算力。投机解码（speculative decoding）就是钻这个空子：让一个<strong>小模型先猜一串</strong>，大模型<strong>一次性批改</strong>，把"猜对的部分"白赚下来。
</p>
<p style="color:var(--muted);margin-top:.4rem">关键洞察只有一句：既然 decode 慢在带宽、算力有富余，那"验证 K 个 token"和"生成 1 个 token"几乎一样快——多验证的那几个 token 几乎是免费的。投机解码把这点免费算力，换成了实打实的加速：典型能快 2-3 倍，而且<strong>输出和直接采样完全等价</strong>，不掉一点质量。这听起来几乎像免费的午餐，而它确实接近免费——代价只是要额外备一个小草稿模型（或者连小模型都不用，走 n-gram）。这一课就把这件"又快又不掉质量"的事，从直觉一路讲到它为什么严格成立。</p>
<p style="color:var(--muted)">路线图：先讲清"为什么验证 K 个 ~= 生成 1 个"这点免费午餐，配一张追踪图看一轮投机怎么跑；再看草稿从哪来（小 draft model / n-gram）；最后看接受/拒绝怎么定、接受率怎么决定加速比，以及为什么它不改变输出质量。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  投机解码的全部精髓，就一句话：<strong>用富余的算力换串行的时间</strong>。自回归解码是 N 步串行，每步都被显存带宽卡着、算力闲着；投机解码让小模型一口气猜 K 个，大模型一次前向把这 K 个<strong>并行</strong>验证掉——猜对几个就等于一步顶几步。它不改模型、不改输出分布，纯粹是把"反正要把权重搬一遍，顺手多算几个 token"这点免费空间用起来。理解它，你就懂了为什么同样的模型、同样的显卡，挂一个小草稿模型就能快好几倍——快的不是"算得多"，而是"等得少"。顺带说，第七部分这几课的主题都是这种"换个角度重新看瓶颈"：投机解码省的是串行时间；下一课 MoE 省的是每个 token 的计算量；再后面的多模态在扩展"输入能是什么"，状态空间模型在重定义"历史怎么记"。四课各打一个新角度，但都还站在你前六部分搭好的地基上。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象你在<strong>批改听写</strong>：普通做法是你念一个词、学生写一个、你再念下一个，一来一回特别慢。投机解码像是请了个<strong>实习生先把整句都猜着写下来</strong>，然后你<strong>一眼扫过去</strong>，对的部分直接打勾、错的地方从那里改起。实习生（小模型）猜得快但不一定准；你（大模型）一次扫一整句、又快又权威。只要实习生猜对的比例够高，整体就比"一个一个念"快得多——而最终对错完全由你说了算，实习生再怎么瞎猜，也不会让最终答案出错。这个类比还能解释一个细节：实习生写得越像你的笔迹（draft 越接近 target），你打勾打得越顺、改得越少——这就是"接受率"在生活里的样子。
</div>

<h2>为什么能加速：免费的并行空间</h2>
<p>先把"免费午餐"算清楚。一次 <span class="mono">llama_decode</span> 的开销，绝大部分是把模型那几个 GB 的权重从显存流过一遍（L30/L32 说的访存密集）；至于这一遍里顺便算 1 个 token 还是 10 个 token 的乘加，差别小到可以忽略。换句话说：<strong>前向一次的成本几乎是固定的，和这次算几个位置无关</strong>。这就留出了一个空子——如果能一次性把好几个候选 token 都喂进去并行验证，那它们摊到的"额外成本"趋近于零。举个量级感受：一次前向的开销，几乎全压在"把几个 GB 的权重在显存里流一遍"上（动辄几十毫秒）；而这一遍里多验几个 token 位置，只是多做一点点乘加，相比那笔固定的搬运开销只是个零头。所以"顺手多验几个 token"的边际成本，低到几乎白送。这也是为什么投机解码只在 decode（memory-bound）上有效、在 prefill（compute-bound）上意义不大：prefill 本来就在并行算一大批 token，没有"闲着的算力"可榨。</p>
<p>自回归解码偏偏不给这个机会：它一步只产 1 个 token，下一个 token 依赖上一个，逼着你串行地把那几个 GB 反复搬 N 遍。投机解码的破法，就是<strong>先用别的办法弄出一串候选</strong>（猜的），再用大模型一次前向<strong>并行验证</strong>这一串——把"串行 N 步"压缩成"猜 1 次 + 验 1 次"。这里有个容易混淆的点：投机解码并没有"少算"——大模型该过的前向一次都没省，被拒的草稿 token 也实打实算了一遍。它省的纯粹是"串行的轮次"：把原本要排队等 N 次的前向，合并成了少数几次。所以它换来的是延迟（latency）下降，而不是总算力下降；在算力本就吃紧（大 batch、多并发）的服务器上，投机的收益反而会被压缩——这又一次呼应了 L30 的 pp/tg、L28 的连续批处理。</p>
<div class="cols">
  <div class="col"><h4>朴素 decode</h4><p>N 个 token = N 次前向，每次搬一遍几 GB 权重，纯串行。慢在"等数据"，不在"算"。</p></div>
  <div class="col"><h4>投机 decode</h4><p>小模型猜 K 个 -&gt; 大模型 1 次前向并行验 K 个 -&gt; 接受 m 个。猜得准时，1 次前向顶 m 步。</p></div>
</div>
<p>把一轮投机定格成一条流水看最清楚：</p>
<div class="trace">
  <div class="tcap"><b>追踪一轮投机解码</b>：草稿模型提 K 个候选，target 一次并行验证，从头接受匹配前缀、并在第一个不匹配处白送 1 个，下一轮从接受点续（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 草稿</h5>
      <div class="cellrow"><span class="vc">t1 t2 t3 t4 t5</span></div>
      <div class="tlab">小模型猜 K=5 个</div></div>
    <div class="op">target<br>1 次前向</div>
    <div class="stn"><h5>② 并行验证</h5>
      <div class="cellrow"><span class="vc blue">5 个位置的预测</span></div>
      <div class="tlab">一次 llama_decode</div></div>
    <div class="op">逐位<br>比对</div>
    <div class="stn"><h5>③ 比对</h5>
      <div class="cellrow"><span class="vc">t1 t2 t3 对 | t4 错</span></div>
      <div class="tlab">第 4 个起不匹配</div></div>
    <div class="op">接受<br>前缀</div>
    <div class="stn"><h5>④ 接受 + bonus</h5>
      <div class="cellrow"><span class="vc hot">t1 t2 t3 + 1 个</span></div>
      <div class="tlab">3 个 + 白送 1 个</div></div>
    <div class="op">续下<br>一轮</div>
    <div class="stn"><h5>⑤ 下一轮</h5>
      <div class="cellrow"><span class="vc">从接受点再猜</span></div>
      <div class="tlab">丢弃 t4 t5 之后</div></div>
  </div>
</div>
<p>这一轮里，大模型只做了<strong>一次</strong>前向，却一口气敲定了 4 个 token（3 个接受 + 1 个 bonus）。要是草稿全中，5 个候选就能换来 6 个 token / 一次前向；要是第一个就错，那就退化成普通 decode（白跑一次草稿）。换个角度看整段生成：原本要 N 次串行前向才能出 N 个 token，现在每次前向平均落定 m+1 个，前向次数就降到约 N/(m+1)——省下的全是那些"排队等显存"的时间。所以投机解码的加速比，全看<strong>接受率</strong>——草稿猜得越准，省得越多。</p>

<h2>草稿从哪来：小模型与 n-gram</h2>
<p>候选不能瞎猜——猜得越接近大模型的真实输出，接受率越高、越省。ggml 的 <span class="mono">common/speculative</span> 给了两条主要路子：</p>
<p><strong>(1) draft model（草稿模型）</strong>：一个和 target <strong>同词表</strong>的小模型（比如 target 是 70B、draft 用同系列的 1B）。它就是个迷你版的自回归解码——贪心地一个个续写，直到够 K 个、或者自己都没把握了：</p>
<pre class="code"><span class="cm">// 草稿模型贪心续写 K 个 (简化自 common/speculative.cpp)</span>
<span class="kw">while</span> (n_drafting &lt; n_max) {
    <span class="fn">llama_decode</span>(ctx_dft, batch);              <span class="cm">// 小模型前向一步</span>
    auto * cur_p = <span class="fn">get_probs</span>(ctx_dft);          <span class="cm">// 取这一步的概率分布</span>
    <span class="kw">if</span> (cur_p-&gt;data[0].p &lt; params.p_min)        <span class="cm">// 最可能的 token 都没把握?</span>
        <span class="kw">break</span>;                                  <span class="cm">// 收手, 不硬猜</span>
    draft.push_back(cur_p-&gt;data[0].id);          <span class="cm">// 把它当下一个候选</span>
    <span class="cm">// ... 把这个 token 接上, 继续猜下一个 ...</span>
}</pre>
<p>注意那个 <span class="mono">p_min</span> 阈值：草稿模型一旦"自己都没把握"（最可能 token 的概率都低于阈值），就<strong>提前收手</strong>——与其硬猜一个大概率被拒的 token、白占一个验证位，不如少猜几个。这正是接受率和草稿长度之间的权衡：猜太短省不到，猜太长又容易在后段翻车。（这说的是草稿的"长度" K；另一个旋钮是草稿模型的"大小"——越小越快但越不准，得和接受率一起权衡，两个旋钮一起决定那个甜区落在哪。）</p>
<p><strong>(2) n-gram</strong>：连小模型都不要，直接从<strong>已经生成的文本里查</strong>。原理很朴素：自然语言里很多片段会重复（人名、固定搭配、代码里的变量名），如果当前这几个 token 在前文出现过，那它后面跟过的那几个 token，很可能就是这次要续的。<span class="mono">common/speculative</span> 里有好几种 n-gram 变体（<span class="mono">ngram-simple</span> / <span class="mono">ngram-cache</span> 等），共同点是<strong>零额外模型、近乎零成本</strong>地凑出候选——特别适合"输出里有大量重复"的场景（改写、续写、结构化输出）。n-gram 这条路尤其适合 agent / 工具调用 / 长文档改写这类"输入和输出大量重叠"的活儿：要续的内容很多就照抄前文，命中率高得惊人，几乎白嫖一截速度，还省掉了加载和运行草稿模型的那份显存与算力。代价是 n-gram 只会照搬前文出现过的片段，遇到全新内容就没辙——所以实现里常把它和草稿模型<strong>串成一条链</strong>：先用近乎零成本的 n-gram 试一把，命中就直接用，没命中再退回让小草稿模型猜。前面源码事实里说的"多实现可链式"，就是这个意思。</p>
<table class="t">
  <tr><th>草稿来源</th><th>额外成本</th><th>准确度（接受率）</th><th>最适合</th></tr>
  <tr><td><span class="mono">draft model</span></td><td>要加载并运行一个小模型</td><td>高（draft 与 target 同系列时）</td><td>通用场景，draft 与 target 同源</td></tr>
  <tr><td><span class="mono">n-gram</span></td><td>近乎零（只查前文）</td><td>看输出的重复度</td><td>输入输出大量重叠：改写 / agent / 代码续写</td></tr>
</table>

<h2>接受、拒绝与接受率</h2>
<p>草稿有了，轮到大模型当裁判。调用方把草稿的 K 个 token 一次性追加进 target 的 batch，<span class="mono">llama_decode</span> 一次算出 target 在这 K 个位置各自的预测，然后<strong>从头逐位比对</strong>：target 在第 i 位的预测和草稿的第 i 个一致就接受，碰到第一个不一致就停，接受前面那段前缀，并把 target 自己在该位置的预测白送进来（见下一节）。这里有个关键的并行点：这 K 个位置的验证是在<strong>一次</strong> <span class="mono">llama_decode</span> 里同时算出来的——靠的是把草稿的 K 个 token 当作"已知输入"一起喂进去，让 target 在一趟前向里给出每个位置的预测。这正是前面那句"验证 K 个 ~= 生成 1 个"在代码层面的落地。最后告诉投机上下文这次接受了几个：</p>
<pre class="code"><span class="cm">// 调用方的验证循环 (示意; 接受语义见 common/speculative.h)</span>
<span class="kw">int</span> n_accepted = 0;
<span class="kw">for</span> (int i = 0; i &lt; n_draft; i++) {
    <span class="kw">if</span> (target_pred[i] != draft[i])             <span class="cm">// 第一个不匹配?</span>
        <span class="kw">break</span>;                                  <span class="cm">// 接受到此为止</span>
    n_accepted++;                               <span class="cm">// 这一位匹配, 接受</span>
}
<span class="fn">common_speculative_accept</span>(spec, seq_id, n_accepted); <span class="cm">// 告知接受了几个</span></pre>
<p>"接受率"就是<strong>平均每轮草稿被接受的比例</strong>。它直接决定加速比：设草稿长度 K、平均接受 m 个，那么大模型每做 1 次前向就推进了 m+1 个 token（含 bonus），理想加速约 m+1 倍——当然要减去跑草稿的开销。所以投机解码有个<strong>甜区</strong>：草稿模型既要够小（跑得快）、又要够准（接受率高），两者得平衡。draft 和 target 越像（同系列、同数据训练），接受率越高。实战里这对应 llama.cpp 的 <span class="mono">--model-draft</span> / <span class="mono">-md</span>（指定草稿模型）和 <span class="mono">--spec-draft-n-max</span> / <span class="mono">--spec-draft-n-min</span>（草稿长度上下限）等参数；草稿长度 K 也不是越大越好——K 越大，后段越容易翻车（一旦中途某个被拒，它后面猜的就全废了），所以通常取 4 到 8 这个量级最划算。总之，投机解码不是"开了就快"的开关，而是"配好草稿来源和长度才快"的调参题——配得好，免费的加速就到手了。</p>
<p>反过来，接受率低时投机解码会<strong>变慢</strong>：草稿老被拒，等于每轮都白跑一遍小模型、还多搬了一次草稿那几个 token 的 KV。所以它不是无脑开就赚——任务越"可预测"（代码、翻译、改写），越划算；越"发散"（创意写作、高温采样下的多样输出），越容易亏。一个实用的判断：如果你的场景里草稿接受率长期低于某条线（经验上大概 0.3 左右），那投机大概率在帮倒忙，不如关掉。好在 llama.cpp 跑投机时会打印接受率统计，看一眼就知道这笔买卖划不划算——这也呼应 L30 的态度：别猜，去量。</p>

<h2>深入：bonus token 与"不改变输出"</h2>
<p>两个折叠，回答投机解码最容易让人犯嘀咕的两个问题。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么第一个不匹配处能白送 1 个 token？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>因为 target 那一次前向，本来就把"每个位置的下一个 token 预测"都算出来了。验证草稿的第 i 个，用的就是 target 在第 i-1 位的预测；当草稿第 i 个被拒，意味着 target 在第 i-1 位给出的预测和草稿不同——而 <strong>target 这个预测，正是这一步本来就要采的那个 token</strong>。所以它不是"额外算的"，是"顺手已经算好的"，直接拿来用就行。于是每轮哪怕草稿一个都没中，至少也能靠这个 bonus 推进 1 个 token——保证投机解码<strong>最坏也不比普通 decode 少出 token</strong>（只是多花了草稿的开销）。顺带说，正因为有这个"保底 1 个"，投机解码的最坏情况是确定的：接受率为 0 时，它退化成"普通 decode + 每轮一次白跑的草稿"，慢一点但绝不会卡死或出错。这种"上不封顶、下有保底"的性质，正是它能放心默认开启的底气。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 投机解码会不会让输出变差？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>不会——这是投机解码最漂亮的地方：它<strong>在数学上等价于直接从 target 采样</strong>，输出分布一模一样。直觉是这样：被接受的 token，都是 target 自己也会给出的（验证通过）；被拒的位置，用的是 target 自己的预测（bonus）。草稿模型只在"提议"环节出现，它提议得好不好只影响<strong>速度</strong>（接受率），完全不参与"最终选哪个 token"的决定。换句话说，草稿是"加速器"不是"决策者"。<strong>贪心解码下</strong>这点最直观（逐位必须精确匹配）；<strong>带温度的随机采样</strong>下也一样：llama.cpp 在每个位置照常从 target 采一个 token，只有当草稿这一位和采到的<strong>完全相同</strong>才接受——发出去的永远是 target 自己采的那个，所以"开不开投机、采样分布完全一致"是天然成立的。（理论上还有一种更宽松的"修正拒绝采样"，能在保持分布不变的前提下提高接受率，但 llama.cpp 走的是更简单的"精确匹配才接受"这一路。）所以你可以放心：投机解码只换速度，不换质量。这一点之所以重要，是因为它把"要不要开投机"从一个质量问题降级成了纯粹的速度问题：你永远不必担心"开了投机会不会让模型变笨"，只需要关心"这个场景下它划不划算"——而后者，量一下接受率就有答案。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>原理：decode 是访存密集（L30/L32），一次前向验证 K 个 token ~= 生成 1 个；用富余算力换串行时间。</li>
    <li>一轮：草稿提 K 个 -&gt; target 一次前向并行验证 -&gt; 接受匹配前缀 + 1 个 bonus -&gt; 下一轮从接受点续。</li>
    <li>草稿来源：(1) 小 draft model（同词表，贪心续写、<span class="mono">p_min</span> 门控）；(2) n-gram（查前文重复，零额外模型）。</li>
    <li>加速比 = 接受率：平均接受 m 个则约 m+1 倍；draft 越像 target 越快；接受率太低反而变慢。</li>
    <li>不改变质量：数学上等价于直接从 target 采样；草稿只影响速度、不参与最终决策。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  投机解码是一个特别值得回味的"系统级"优化：它没改一行模型、没动一个权重，纯粹是<strong>看穿了瓶颈的性质</strong>（decode 慢在带宽、算力闲着），再用一个"提议-验证"的结构把闲置算力榨出来。这种"先猜后验、猜错不亏"的套路，在计算机系统里到处都是——CPU 的分支预测（猜下一条指令、猜错就回滚）、数据库的乐观锁、网络的预取，骨子里都是同一招：<strong>在有富余资源、且验证比生成便宜的地方，大胆地投机</strong>。把这个视角装进脑子，你会在很多看似无关的系统里认出它。其实大模型推理本身也越来越像一个"投机 + 验证"的系统：从这一课的 token 级投机，到把草稿缓存复用，再到各种"先用便宜的近似、再用昂贵的精确兜底"的设计——一旦认出这个模式，很多看似花哨的加速技巧，本质都是同一个朴素的赌注。下一课我们看另一种"用结构换效率"的思路：MoE 把一个大 FFN 拆成很多专家，每个 token 只走几个——用激活稀疏换参数容量。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
In inference the model emits one token at a time: run the whole sequence forward, sample the next token, append it, run forward again... step after step, purely serial. Yet as the last part showed (L30/L32), the decode phase is <strong>memory-bound</strong> - most of the time goes to hauling several GB of weights in from VRAM, while the actual multiply-adds leave compute underused. Speculative decoding exploits exactly this gap: let a <strong>small model guess a run of tokens first</strong>, have the big model <strong>grade them in one shot</strong>, and pocket the part it guessed right for free.
</p>
<p style="color:var(--muted);margin-top:.4rem">The key insight is one sentence: since decode is slow on bandwidth with compute to spare, "verifying K tokens" is about as fast as "generating 1 token" - those extra verified tokens are almost free. Speculative decoding turns that free compute into real speedup: typically 2-3x faster, and <strong>exactly equivalent to direct sampling</strong>, with no quality loss at all. It almost sounds like a free lunch, and it nearly is - the only cost is keeping a small draft model around (or no small model at all, going the n-gram route). This lesson takes this "faster with no quality loss" claim from intuition all the way to why it strictly holds.</p>
<p style="color:var(--muted)">Roadmap: first make clear why "verify K ~= generate 1" is a free lunch, with a trace of one speculative round; then where the draft comes from (a small draft model / n-gram); and finally how accept/reject is decided, how the acceptance rate sets the speedup, and why it does not change output quality.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The whole essence of speculative decoding is one sentence: <strong>trade spare compute for serial time</strong>. Autoregressive decoding is N serial steps, each stalled on VRAM bandwidth with compute sitting idle; speculative decoding lets a small model guess K tokens in one go, and the big model verifies all K in <strong>one parallel</strong> forward pass - guess a few right and one step does the work of several. It changes no model and no output distribution; it purely uses the free room of "since the weights get hauled in anyway, compute a few extra tokens while at it". Grasp it and you see why the same model on the same GPU runs several times faster once you attach a small draft model - the speedup is not "computing more" but "waiting less". By the way, this whole Part 7 shares that theme of "re-seeing the bottleneck from a new angle": speculative decoding saves serial time; next lesson's MoE saves per-token compute; later, multimodal extends "what the input can be" and state-space models redefine "how history is kept". Four lessons, four new angles - all still standing on the foundation the first six parts built.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture <strong>grading a dictation</strong>: the plain way is you read one word, the student writes one, you read the next - a slow back-and-forth. Speculative decoding is like hiring an <strong>intern to guess the whole sentence ahead</strong>, then you <strong>scan it at a glance</strong>, tick the right part and fix from where it went wrong. The intern (small model) guesses fast but not always right; you (big model) scan a whole sentence at once, fast and authoritative. As long as the intern's hit rate is high enough, the whole thing beats "one word at a time" - and the final right-or-wrong is entirely your call, so however wildly the intern guesses, the final answer is never wrong. The analogy also explains a detail: the more the intern's handwriting matches yours (the closer the draft is to the target), the smoother you tick and the less you fix - that is what "acceptance rate" looks like in everyday terms.
</div>

<h2>Why it speeds up: the free parallel room</h2>
<p>First, do the "free lunch" math. The cost of one <span class="mono">llama_decode</span> is mostly streaming the model's several GB of weights through from VRAM (the memory-bound point of L30/L32); whether you also compute the multiply-adds for 1 token or 10 in that pass is a negligible difference. In other words: <strong>one forward pass costs almost a fixed amount, regardless of how many positions it computes</strong>. That leaves a gap - if you can feed several candidate tokens in at once and verify them in parallel, their "extra cost" approaches zero. A sense of scale: the cost of one forward pass sits almost entirely on "streaming the several GB of weights through VRAM" (often tens of milliseconds); verifying a few more token positions in that same pass only adds a bit more multiply-add, a mere fraction of that fixed hauling cost. So the marginal cost of "verifying a few more tokens in passing" is low enough to be nearly free. This is also why speculative decoding helps only in decode (memory-bound) and means little in prefill (compute-bound): prefill is already computing a big batch of tokens in parallel, with no "idle compute" to wring out.</p>
<p>Autoregressive decoding refuses this chance: it produces just 1 token per step, and the next token depends on the last, forcing you to serially haul those GB N times over. Speculative decoding's trick is to <strong>produce a run of candidates by some other means</strong> (a guess), then use the big model's one forward pass to <strong>verify the run in parallel</strong> - compressing "N serial steps" into "1 guess + 1 verify". A common confusion here: speculative decoding does not "compute less" - the big model's forward passes are not saved, and rejected draft tokens really were computed. What it saves is purely "serial rounds": forward passes that used to queue up N times are merged into a few. So what it buys is lower latency, not lower total compute; on a server where compute is already tight (large batch, high concurrency), speculation's gains shrink instead - echoing once more L30's pp/tg and L28's continuous batching.</p>
<div class="cols">
  <div class="col"><h4>naive decode</h4><p>N tokens = N forward passes, each hauling several GB of weights, purely serial. Slow on "waiting for data", not "computing".</p></div>
  <div class="col"><h4>speculative decode</h4><p>small model guesses K -&gt; big model verifies K in 1 pass -&gt; accept m. When guesses are good, one pass does m steps' work.</p></div>
</div>
<p>Freezing one speculative round into a flow is clearest:</p>
<div class="trace">
  <div class="tcap"><b>Trace one speculative round</b>: the draft model proposes K candidates, the target verifies them in one parallel pass, accepts the matching prefix and adds 1 free bonus at the first mismatch, and the next round resumes from the accept point (illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>1 draft</h5>
      <div class="cellrow"><span class="vc">t1 t2 t3 t4 t5</span></div>
      <div class="tlab">small model guesses K=5</div></div>
    <div class="op">target<br>1 pass</div>
    <div class="stn"><h5>2 verify</h5>
      <div class="cellrow"><span class="vc blue">predictions at 5 positions</span></div>
      <div class="tlab">one llama_decode</div></div>
    <div class="op">compare<br>position-wise</div>
    <div class="stn"><h5>3 compare</h5>
      <div class="cellrow"><span class="vc">t1 t2 t3 ok | t4 no</span></div>
      <div class="tlab">mismatch from #4</div></div>
    <div class="op">accept<br>prefix</div>
    <div class="stn"><h5>4 accept + bonus</h5>
      <div class="cellrow"><span class="vc hot">t1 t2 t3 + 1</span></div>
      <div class="tlab">3 + 1 free bonus</div></div>
    <div class="op">next<br>round</div>
    <div class="stn"><h5>5 next round</h5>
      <div class="cellrow"><span class="vc">guess again from accept</span></div>
      <div class="tlab">discard after t4 t5</div></div>
  </div>
</div>
<p>In this round the big model did <strong>one</strong> forward pass yet nailed down 4 tokens at once (3 accepted + 1 bonus). If the draft were all correct, 5 candidates would buy 6 tokens per forward pass; if the very first is wrong, it degrades to plain decode (a wasted draft run). Seen across a whole generation: where you used to need N serial forward passes for N tokens, each pass now lands m+1 on average, so the pass count drops to about N/(m+1) - and all the savings are that "queuing for VRAM" time. So speculative decoding's speedup hinges entirely on the <strong>acceptance rate</strong> - the better the draft guesses, the more you save.</p>

<h2>Where the draft comes from: a small model and n-gram</h2>
<p>The candidates cannot be wild guesses - the closer they are to the big model's real output, the higher the acceptance rate and the more you save. ggml's <span class="mono">common/speculative</span> offers two main routes:</p>
<p><strong>(1) A draft model</strong>: a small model sharing the target's <strong>vocabulary</strong> (say target 70B, draft a 1B from the same family). It is just a mini autoregressive decode - greedily extending one token at a time until it has K, or until it is unsure:</p>
<pre class="code"><span class="cm">// draft model greedily extends K tokens (simplified from common/speculative.cpp)</span>
<span class="kw">while</span> (n_drafting &lt; n_max) {
    <span class="fn">llama_decode</span>(ctx_dft, batch);              <span class="cm">// small model, one forward step</span>
    auto * cur_p = <span class="fn">get_probs</span>(ctx_dft);          <span class="cm">// the step's probability distribution</span>
    <span class="kw">if</span> (cur_p-&gt;data[0].p &lt; params.p_min)        <span class="cm">// not even confident in the top token?</span>
        <span class="kw">break</span>;                                  <span class="cm">// stop, do not force a guess</span>
    draft.push_back(cur_p-&gt;data[0].id);          <span class="cm">// take it as the next candidate</span>
    <span class="cm">// ... append this token, keep guessing the next ...</span>
}</pre>
<p>Note that <span class="mono">p_min</span> threshold: once the draft model is "unsure even of itself" (the top token's probability falls below the threshold), it <strong>stops early</strong> - rather than force a token likely to be rejected and waste a verify slot, it guesses fewer. This is exactly the trade-off between acceptance rate and draft length: too short saves little, too long tends to derail in the later positions. (That is the draft's "length" K; the other knob is the draft model's "size" - smaller is faster but less accurate, to be weighed against the acceptance rate, and the two knobs together decide where the sweet spot falls.)</p>
<p><strong>(2) n-gram</strong>: no small model at all - look it up directly <strong>in the already-generated text</strong>. The idea is plain: natural language repeats a lot (names, fixed phrases, variable names in code), so if the current few tokens appeared earlier, the tokens that followed them then are likely what comes next now. <span class="mono">common/speculative</span> has several n-gram variants (<span class="mono">ngram-simple</span> / <span class="mono">ngram-cache</span>, etc.); what they share is producing candidates at <strong>zero extra model and near-zero cost</strong> - great for outputs with heavy repetition (rewriting, continuation, structured output). The n-gram route especially suits agent / tool-calling / long-document-rewriting work where input and output overlap heavily: much of what comes next is copied from earlier text, the hit rate is astonishingly high, you grab a chunk of free speed, and you save the VRAM and compute of loading and running a draft model too. The cost is that n-gram only copies fragments that appeared earlier and is helpless against genuinely new content - so implementations often <strong>chain</strong> it with a draft model: try the near-zero-cost n-gram first, use it on a hit, and fall back to the small draft model's guess on a miss. The "multiple implementations can chain" from the source facts above means exactly this.</p>
<table class="t">
  <tr><th>Draft source</th><th>Extra cost</th><th>Accuracy (acceptance)</th><th>Best for</th></tr>
  <tr><td><span class="mono">draft model</span></td><td>load and run a small model</td><td>high (when draft and target share a family)</td><td>general use, draft and target same origin</td></tr>
  <tr><td><span class="mono">n-gram</span></td><td>near zero (just look up earlier text)</td><td>depends on output repetition</td><td>heavy input/output overlap: rewriting / agent / code continuation</td></tr>
</table>

<h2>Accept, reject, and the acceptance rate</h2>
<p>With a draft in hand, the big model becomes the judge. The caller appends the draft's K tokens into the target's batch at once, <span class="mono">llama_decode</span> computes the target's prediction at each of those K positions in one pass, then compares <strong>position by position from the start</strong>: if the target's prediction at position i matches the draft's i-th token, accept it; at the first mismatch, stop, accept the prefix so far, and add the target's own prediction at that position for free (see next section). A key parallelism point here: these K positions are all verified in <strong>one</strong> <span class="mono">llama_decode</span> - by feeding the draft's K tokens in as "known inputs" so the target gives a prediction at every position in a single forward pass. This is exactly the code-level realization of "verify K ~= generate 1" from earlier. Finally tell the speculative context how many were accepted:</p>
<pre class="code"><span class="cm">// the caller's verify loop (illustrative; accept semantics in common/speculative.h)</span>
<span class="kw">int</span> n_accepted = 0;
<span class="kw">for</span> (int i = 0; i &lt; n_draft; i++) {
    <span class="kw">if</span> (target_pred[i] != draft[i])             <span class="cm">// first mismatch?</span>
        <span class="kw">break</span>;                                  <span class="cm">// accept up to here</span>
    n_accepted++;                               <span class="cm">// this position matches, accept</span>
}
<span class="fn">common_speculative_accept</span>(spec, seq_id, n_accepted); <span class="cm">// inform how many accepted</span></pre>
<p>The "acceptance rate" is the <strong>average fraction of the draft accepted per round</strong>. It directly sets the speedup: with draft length K and m accepted on average, each big-model forward pass advances m+1 tokens (including the bonus), an ideal speedup around m+1x - minus the cost of running the draft, of course. So speculative decoding has a <strong>sweet spot</strong>: the draft model must be small enough (fast) yet accurate enough (high acceptance), and the two must balance. The more alike draft and target are (same family, same training data), the higher the acceptance. In practice this maps to llama.cpp's <span class="mono">--model-draft</span> / <span class="mono">-md</span> (pick the draft model) and <span class="mono">--spec-draft-n-max</span> / <span class="mono">--spec-draft-n-min</span> (draft-length bounds); and draft length K is not better when bigger - the larger K is, the more the later positions tend to derail (once one mid-draft token is rejected, everything it guessed after is wasted), so 4 to 8 is usually the sweet range. In short, speculative decoding is not a "turn it on and it's fast" switch but a "fast once you tune the draft source and length" knob-turning exercise - tune it well and the free speedup is yours.</p>
<p>Conversely, when the acceptance rate is low, speculative decoding gets <strong>slower</strong>: the draft keeps getting rejected, so every round wastes a small-model run and the extra KV for those draft tokens. It is not a free win you flip on blindly - the more "predictable" the task (code, translation, rewriting), the more it pays; the more "divergent" (creative writing, diverse output under high-temperature sampling), the easier it loses. A practical test: if your scenario's draft acceptance rate stays below some line (around 0.3 by rule of thumb), speculation is probably doing more harm than good and is better turned off. Helpfully, llama.cpp prints acceptance-rate stats when speculating, so one glance tells you whether the trade pays - echoing L30's stance: do not guess, measure.</p>

<h2>Deeper: the bonus token and "no change to output"</h2>
<p>Two folds answering the two things about speculative decoding that most make people uneasy.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why can the first mismatch hand you 1 free token? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Because that one target forward pass already computed "the next-token prediction at every position". Verifying the draft's i-th token uses the target's prediction at position i-1; when the draft's i-th is rejected, it means the target's prediction at position i-1 differs from the draft - and <strong>that target prediction is exactly the token this step was going to sample anyway</strong>. So it is not "extra computed" but "already computed in passing", free to take. Thus even if a round's draft is all wrong, the bonus still advances 1 token - guaranteeing speculative decoding <strong>never emits fewer tokens than plain decode in the worst case</strong> (it only spent the draft's cost). Worth adding: because of this "at least 1" floor, speculative decoding's worst case is well-defined - at acceptance rate 0 it degrades to "plain decode plus one wasted draft run per round", a bit slower but never stalling or erroring. This "no ceiling above, a floor below" property is exactly what makes it safe to leave on by default.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Does speculative decoding make the output worse? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>No - this is the prettiest part: it is <strong>mathematically equivalent to sampling directly from the target</strong>, the output distribution is identical. The intuition: accepted tokens are ones the target itself would give (they passed verification); rejected positions use the target's own prediction (the bonus). The draft model appears only in "proposing", and how well it proposes affects only <strong>speed</strong> (acceptance rate), never the decision of "which token is final". In other words, the draft is an accelerator, not a decision-maker. <strong>Under greedy decoding</strong> this is most obvious (each position must match exactly); <strong>under temperature sampling</strong> it is the same: llama.cpp samples a token from the target at each position as usual, and accepts the draft only when it <strong>exactly equals</strong> the sampled token - what is emitted is always the target's own sample, so "the sampling distribution is identical whether speculation is on or off" holds by construction. (There is also a looser "modified rejection sampling" in theory that raises the acceptance rate while preserving the distribution, but llama.cpp takes the simpler "accept only on exact match" route.) So rest assured: speculative decoding trades only speed, not quality. This matters because it downgrades "should I enable speculation" from a quality question to a pure speed question: you never have to worry "will turning on speculation make the model dumber", only "is it worth it in this scenario" - and the latter is answered by measuring the acceptance rate.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Principle: decode is memory-bound (L30/L32), so one forward pass verifying K tokens ~= generating 1; trade spare compute for serial time.</li>
    <li>One round: draft proposes K -&gt; target verifies all K in one pass -&gt; accept the matching prefix + 1 bonus -&gt; next round resumes from the accept point.</li>
    <li>Draft sources: (1) a small draft model (same vocab, greedy extend, <span class="mono">p_min</span> gating); (2) n-gram (look up repeats in earlier text, no extra model).</li>
    <li>Speedup = acceptance rate: m accepted on average gives about m+1x; the more draft resembles target, the faster; too low an acceptance is slower instead.</li>
    <li>No quality change: mathematically equivalent to sampling from the target; the draft affects only speed, not the final decision.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  Speculative decoding is a "systems-level" optimization worth savoring: it changed not one line of the model and not one weight, purely <strong>seeing through the nature of the bottleneck</strong> (decode is slow on bandwidth with compute idle), then using a "propose-verify" structure to wring out that idle compute. This "guess first, verify after, lose nothing if wrong" pattern is everywhere in computer systems - a CPU's branch prediction (guess the next instruction, roll back if wrong), a database's optimistic locking, network prefetch - all the same move at heart: <strong>where there are spare resources and verifying is cheaper than generating, speculate boldly</strong>. Plant this view and you will recognize it in many seemingly unrelated systems. In fact large-model inference itself increasingly looks like a "speculate + verify" system: from this lesson's token-level speculation, to reusing cached drafts, to all sorts of "use a cheap approximation first, fall back to the expensive exact one" designs - once you recognize the pattern, many flashy-looking speedup tricks are at heart the same plain bet. Next lesson, another "structure for efficiency" idea: MoE splits one big FFN into many experts, each token taking only a few - trading activation sparsity for parameter capacity.
</div>
""",
}
