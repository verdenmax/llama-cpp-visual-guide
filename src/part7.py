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

LESSON_35 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前面每一层 FFN（前馈网络，L11），都是"每个 token 都老老实实从头到尾过一遍"。可现在最大的那些开源模型（Mixtral、DeepSeek、Qwen-MoE…）几乎都不这么干了——它们把一层 FFN 拆成几十甚至上百个"专家"（expert），每个 token 只挑其中两三个走。这就是 MoE（Mixture of Experts，专家混合）。这一课看 ggml 怎么实现它：一个 token 怎么被"路由"到几个专家、又怎么只算这几个而不浪费算力。这不是什么边角技巧——它已经是当下最强开源模型的标配架构，理解它，你才能读懂这一代模型为什么能又大又跑得动。
</p>
<p style="color:var(--muted);margin-top:.4rem">MoE 的魔力在一句话：<strong>参数容量像个大模型，单 token 的计算量却像个小模型</strong>。一个 8 专家、每 token 选 2 的 MoE 层，参数量约等于 8 个 FFN；可每个 token 只过其中 2 个——计算量只有"同样大的稠密模型"的四分之一。模型因此能用大参数"记"住多得多的东西，而每步推理的算力却省下一大截。</p>
<p style="color:var(--muted)">路线图：先看路由（router 怎么给每个 token 挑专家），配一张图追踪一个 token 的路由；再看 ggml 怎么用 <span class="mono">ggml_mul_mat_id</span> 只算被选中的专家；最后看这套"激活稀疏"的设计到底在用什么换什么、代价又在哪。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  MoE 的核心是一个朴素的赌注：<strong>不是每个 token 都需要整个网络的全部本事</strong>。一个讲代码的 token 和一个讲诗的 token，也许该交给不同的"专家"去处理。于是 MoE 把一层大 FFN 拆成 N 个小专家，再加一个"调度员"（router）给每个 token 挑最合适的 k 个。好处是参数能堆得很大（每个专家各记一点不同的东西），但每个 token 的实际计算只摊到 k 个专家上——<strong>用"激活稀疏"换"参数容量"</strong>。这也是为什么你会看到"总参数 600 亿、激活参数只有 100 亿"这种说法：前者是全部专家加起来，后者是单个 token 真正走过的那几个。这套"总参数大、激活参数小"的设计，正是为什么 MoE 模型下载下来动辄上百 GB、跑起来却没那么吃算力——你买的是"容量"，付的是"显存"。也正因如此，MoE 模型对个人玩家有点"门槛在显存"：它推理不费算力，却要求你先有足够的显存把全部专家装下——这又把皮球踢回了 L33 的多卡与 offload。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把稠密 FFN 想成一家<strong>什么都管的全科门诊</strong>：每个病人来了，都要把所有科室从头看一遍，又慢又浪费。MoE 则像一家<strong>分科的大医院 + 一个分诊台</strong>：分诊台（router）看一眼你的症状，把你只派给最相关的两三个科室（专家）。医院科室再多（参数再大），你这一趟也只跑两三个科室（算力不变）。代价也很直白：所有科室都得开着门、养着人（所有专家权重都要驻留显存），哪怕这一刻只有两个在接诊。这个类比也点出了 MoE 的甜与苦：分科让"看专科"更快更准（专家各管一摊），但维持一整座大医院的开销（养所有科室）始终都在。顺着这个类比你还能想到：要是分诊台水平不行（router 没训好），把急诊病人派去看眼科，那再好的专家也救不了——这就是后面要讲的"负载均衡 / 路由质量"为什么关键。
</div>

<h2>路由：router 怎么给 token 挑专家</h2>
<p>一切从一个小小的"打分"开始。MoE 层里有一个 <strong>router（也叫 gate，门控）</strong>，本质就是一个小线性层：拿当前 token 的向量，对每个专家打一个分。分越高，说明这个 token 越该交给那个专家。你可以把 router 想成一个"调度员"：它读一眼 token 的"语义指纹"（就是那个向量），凭训练得来的经验判断"这活儿该派给哪几位"。<span class="mono">ggml_argsort_top_k</span> 从这些分里挑出最高的 k 个（比如 8 选 2），就是这个 token 这一步要走的专家。这里有个常被问到的点：为什么是 top-k（选好几个）而不是 top-1（只选最好的一个）？因为只选一个，路由一旦判断失误就没有退路、训练也更不稳；选 2 个则让两个专家的输出加权融合，既容错、又能表达"这个 token 介于两类之间"的细腻语义。来看 ggml 里真实的路由几行（出自 <span class="mono">build_moe_ffn</span>）：</p>
<pre class="code"><span class="cm">// MoE 路由 (简化自 src/llama-graph.cpp build_moe_ffn)</span>
logits = <span class="fn">build_lora_mm</span>(gate_inp, cur);        <span class="cm">// router 线性层: 每个专家一个分 [n_expert, n_tokens]</span>
probs  = <span class="fn">ggml_soft_max</span>(logits);                 <span class="cm">// 转成概率 (有的模型用 sigmoid)</span>
selected = <span class="fn">ggml_argsort_top_k</span>(probs, n_expert_used); <span class="cm">// 选 top-k 个专家 (如 8 选 2)</span>
weights = <span class="fn">ggml_get_rows</span>(probs, selected);       <span class="cm">// 取这 k 个专家的门控权重</span>
weights = <span class="fn">normalize</span>(weights);                   <span class="cm">// 归一化, 让 k 个权重加起来为 1</span></pre>
<p>逐行看：<span class="mono">build_lora_mm(gate_inp, cur)</span> 是 router 线性层，给每个 token 算出对所有专家的打分（<span class="mono">logits</span>）；<span class="mono">ggml_soft_max</span> 把分变成概率；<span class="mono">ggml_argsort_top_k</span> 挑出最高的 <span class="mono">n_expert_used</span> 个（这就是"8 选 2"里的 2）；<span class="mono">ggml_get_rows</span> 把这 k 个专家对应的权重取出来、归一化——后面合并专家输出时，就按这组权重加权。整套路由极轻：相比专家本身的大矩阵乘，router 这点开销几乎可以忽略。值得强调的是，router 不是写死的规则，而是<strong>训练出来的</strong>：模型在海量数据上自己学会了"什么样的 token 该交给哪个专家"。所以你没法预先知道某个专家"专精"什么——它可能学成了"管标点的"、"管数字的"、"管某种语言的"，也可能是人类完全看不出规律的某种内部分工。这正是 MoE 既神奇又有点黑箱的地方：分工是涌现的，不是设计的。</p>
<p>把一个 token 的路由定格成一张图最直观：</p>
<div class="trace">
  <div class="tcap"><b>追踪一个 token 的 MoE 路由</b>：router 给 8 个专家打分，选出 top-2（带权重），只有这 2 个专家真正参与计算，最后按权重加权求和成输出（示意）。</div>
<svg viewBox="0 0 680 320" width="100%" role="img" aria-label="MoE 路由示例：一个 token 经 router 打分，从 8 个专家里选 top-2（带权重），两个专家各算后按权重加权求和成输出">
<g font-family="ui-monospace,monospace">
<rect x="30" y="150" width="72" height="40" rx="6" fill="#ffffff" stroke="#c2630e"/><text x="66" y="174" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">token</text>
<line x1="102" y1="170" x2="138" y2="170" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 144 170 L 136 166 L 136 174 z" fill="#9aa6b2"/>
<rect x="146" y="142" width="92" height="56" rx="6" fill="#ffffff" stroke="#2563eb"/><text x="192" y="168" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">router</text><text x="192" y="185" text-anchor="middle" fill="#5b6470" font-size="10">门控 gate</text>
<text x="332" y="18" text-anchor="middle" fill="#5b6470" font-size="11">8 个专家 (FFN)</text>
<line x1="238" y1="170" x2="296" y2="41" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="28" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="45" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E0</text>
<line x1="238" y1="170" x2="296" y2="73" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="60" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="77" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E1</text>
<line x1="238" y1="170" x2="296" y2="105" stroke="#7c3aed" stroke-width="1.8"/>
<path d="M 302 105 L 294 101 L 294 109 z" fill="#7c3aed"/>
<rect x="300" y="92" width="64" height="26" rx="5" fill="#ece3fb" stroke="#7c3aed"/><text x="332" y="109" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">E2</text>
<text x="372" y="109" fill="#7c3aed" font-weight="700" font-size="11">w1=0.7</text>
<line x1="238" y1="170" x2="296" y2="137" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="124" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="141" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E3</text>
<line x1="238" y1="170" x2="296" y2="169" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="156" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="173" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E4</text>
<line x1="238" y1="170" x2="296" y2="201" stroke="#7c3aed" stroke-width="1.8"/>
<path d="M 302 201 L 294 197 L 294 205 z" fill="#7c3aed"/>
<rect x="300" y="188" width="64" height="26" rx="5" fill="#ece3fb" stroke="#7c3aed"/><text x="332" y="205" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">E5</text>
<text x="372" y="205" fill="#7c3aed" font-weight="700" font-size="11">w2=0.3</text>
<line x1="238" y1="170" x2="296" y2="233" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="220" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="237" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E6</text>
<line x1="238" y1="170" x2="296" y2="265" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="252" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="269" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E7</text>
<text x="332" y="296" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">top-2 选中</text>
<text x="332" y="312" text-anchor="middle" fill="#9aa6b2" font-size="10">其余 6 个不算</text>
<line x1="426" y1="105" x2="470" y2="170" stroke="#7c3aed" stroke-width="1.6"/>
<line x1="426" y1="201" x2="470" y2="170" stroke="#7c3aed" stroke-width="1.6"/>
<path d="M 474 170 L 466 166 L 466 174 z" fill="#7c3aed"/>
<rect x="476" y="132" width="84" height="76" rx="6" fill="#ffffff" stroke="#7c3aed"/><text x="518" y="164" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">加权求和</text><text x="518" y="182" text-anchor="middle" fill="#5b6470" font-size="10">w1*E2 + w2*E5</text>
<line x1="560" y1="170" x2="588" y2="170" stroke="#c2630e" stroke-width="1.6"/><path d="M 594 170 L 586 166 L 586 174 z" fill="#c2630e"/>
<rect x="596" y="150" width="64" height="40" rx="6" fill="#ffffff" stroke="#c2630e"/><text x="628" y="174" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">输出</text>
</g></svg>
</div>

<h2>稀疏地算：ggml_mul_mat_id 只算选中的专家</h2>
<p>选好了专家，接下来是 MoE 最关键、也最容易想当然的一步：<strong>怎么"只算被选中的专家"</strong>。一个偷懒的实现可能是"8 个专家全算一遍，再把没选中的 6 个扔掉"——那 MoE 就一点没省算力了。事实上，naive 的"全算再扔"在某些早期实现里真的存在过，效果就是"参数稀疏了、算力没省"，白白浪费了 MoE 的好处。所以"真稀疏"是 MoE 能不能落地的关键，也是 ggml 专门为它做一个 <span class="mono">ggml_mul_mat_id</span> 算子的原因。ggml 用这个算子来做到真正的稀疏：</p>
<pre class="code"><span class="cm">// 稀疏专家矩阵乘 (src/llama-graph.cpp build_moe_ffn)</span>
<span class="cm">// selected = 上一节选出的 ids: 每个 token 选中了哪几个专家</span>
up   = <span class="fn">ggml_mul_mat_id</span>(up_exps,   cur, selected); <span class="cm">// 只对选中专家做 up 投影</span>
gate = <span class="fn">ggml_mul_mat_id</span>(gate_exps, cur, selected); <span class="cm">// 只对选中专家做 gate 投影</span>
act  = <span class="fn">silu</span>(gate) * up;                           <span class="cm">// 激活 (SwiGLU)</span>
out  = <span class="fn">ggml_mul_mat_id</span>(down_exps, act, selected); <span class="cm">// down 投影回原维度</span>
<span class="cm">// 最后按 router 的 weights 把 k 个专家的 out 加权求和</span></pre>
<p>关键就在 <span class="mono">ggml_mul_mat_id</span> 里那个 <span class="mono">id</span>：它比普通矩阵乘多吃一个 <span class="mono">ids</span> 张量（就是上一节的 <span class="mono">selected</span>），告诉这次乘法"每个 token 该乘哪几个专家的权重"。于是它不把 token 和全部 8 个专家相乘，而是<strong>按 ids 间接寻址</strong>、只取出被选中的那 2 个专家来算。<span class="mono">up_exps</span>/<span class="mono">gate_exps</span>/<span class="mono">down_exps</span> 是把所有专家权重打包在一起的大张量，<span class="mono">ids</span> 就是从里面挑"该用哪几片"的索引。算力因此实打实降到 <span class="mono">n_expert_used / n_expert</span>（8 选 2 就是四分之一），而不是"算完再扔"。顺带说清一个常见误解：MoE 省的是 <strong>FFN 这一块的算力</strong>，不是整个模型的算力。注意力、归一化、embedding 这些每个 token 还是照常全算——只有 FFN 被稀疏化了。但在大模型里 FFN 恰恰是参数和算力的大头，所以把它稀疏掉，整体收益就很可观。（这也是为什么 MoE 几乎只动 FFN、不碰注意力——注意力本来就不是参数大头，拆它收益不大，还会破坏全局信息的流动。）</p>
<div class="cols">
  <div class="col"><h4>等容量稠密</h4><p>同样多的参数做成一个大网络，每个 token 全部过一遍：算力 = 8 份专家。</p></div>
  <div class="col"><h4>MoE（8 选 2）</h4><p>每个 token 只激活 2 个专家：算力 = 2 份，是等容量稠密的 1/4。</p></div>
  <div class="col"><h4>共同点</h4><p>参数一样多（8 份专家全驻留显存）；差别只在"每步真正激活几份"。</p></div>
</div>

<h2>为什么这么设计：用稀疏换容量</h2>
<p>把前面两节连起来，MoE 的取舍就清楚了：它赌的是<strong>"知识可以分而治之"</strong>——与其让每个 token 都过一个无所不包的大 FFN，不如让它只过几个最相关的专家。这样模型能把参数堆得极大（每个专家分管一摊"知识"），而单 token 的计算量只跟"选几个"有关、跟"总共有多少专家"无关。这就是为什么近两年的旗舰开源模型几乎清一色 MoE：在固定的推理算力预算下，MoE 能塞进比稠密模型多得多的参数，从而更"聪明"。换个比喻：稠密模型像请了一个什么都懂一点的全才，MoE 像请了一个专家团再配个分诊台——团队的总知识量大得多，但每次只惊动相关的那两位。MoE 的兴起背后是一条朴素的经济学：训练和推理的算力都很贵，而参数（显存）相对便宜。MoE 正好顺着这条线——花便宜的显存换贵的算力，在同样的算力预算下做出更强的模型。这就是为什么一旦有人证明 MoE 能 scale，整个开源社区几乎一夜之间都跟了上来。当然，MoE 也不是万灵药：它更难训练（负载均衡、稳定性都是坑）、对显存更挑剔、在小规模上未必比稠密划算。它是"大模型时代"的产物——当你想把参数堆到稠密架构吃不消的量级时，MoE 才真正显出威力。</p>
<p>但天下没有免费的午餐，MoE 的代价也很实在，而且正好踩在前几课讲过的痛点上。<strong>显存</strong>：虽然每步只算 2 个专家，但 8 个专家的权重<strong>全都得待在显存里</strong>——你不知道下一个 token 会路由到哪几个。所以 MoE 的显存占用是按"总参数"算的，往往大得吓人（这也是为什么 MoE 模型特别依赖 L33 的多卡 / offload）。好在量化（L06/L12/L29）在这里帮了大忙：把专家权重压到 4-bit，显存一下小四倍，很多原本装不下的 MoE 才挤得进消费级显卡。</p>
<p><strong>访存不规整</strong>：相邻的 token 可能路由到完全不同的专家，<span class="mono">ggml_mul_mat_id</span> 的间接寻址让访存比稠密矩阵乘更跳跃、对缓存更不友好（呼应 L31/L32 的访存密集）。所以 MoE 是"省了算力、却更吃显存和带宽"的一笔交易——它把瓶颈从"算"挪向了"存和搬"。这也解释了一个实战现象：同样激活参数的 MoE 和稠密模型，MoE 跑起来不一定更快，但能在同样的显卡上装下聪明得多的模型。还有个微妙之处：prefill 时一批 token 一起算，不同 token 路由到不同专家，反而能把所有专家都用起来、并行度高；而 decode 时一次只有一个 token，往往只激活两个专家，硬件利用率偏低——这又一次呼应了 L18/L30 的 prefill vs decode。</p>

<h2>深入：负载均衡与 MoE 变体</h2>
<p>最后两个折叠，补两个真正落地 MoE 时绕不开的问题。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 为什么要费劲做"负载均衡"？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>设想 router 学偏了：它把绝大多数 token 都路由给同样那两个专家，剩下六个几乎没人光顾。那会发生两件坏事：被冷落的专家训练不充分、白占参数（容量浪费）；被挤爆的专家成了瓶颈，还可能超出它的"容量"（在某些训练和分布式推理实现里，一个专家一批能接的 token 有上限，超了就丢弃；但注意 llama.cpp 推理并不丢 token，永远把每个被选中的专家都算完）。所以训练 MoE 时通常会加一个<strong>负载均衡的辅助损失</strong>（auxiliary loss），鼓励 router 把 token 尽量均匀地分给所有专家；还会设"容量因子"给每个专家留出余量。这一步在<strong>推理</strong>时其实已经固化在权重里了（router 已经训练好），但理解它能帮你看懂 MoE 的很多设计——比如为什么有的模型要做 expert group、要做 token drop。推理侧 ggml 不需要再算 aux loss，但 router 给出的分布均不均衡，直接影响真实硬件上的专家利用率。还有个推理侧的现实问题，尤其在专家被分散到多张卡上时（L33 的专家并行）：一批 token（比如 server 同时处理的多个请求）如果恰好都挤到同一个专家，承载它的那张卡就成了串行瓶颈、别的卡却闲着——所以 MoE 在多卡高并发服务时，吞吐有时反而不如同等激活参数的稠密模型稳定。（在单卡上则相反：全挤到一个专家反而是一次规整的大矩阵乘，并不构成瓶颈。）（顺带说，正因为推理时 router 已经定型，换不同的输入、跑不同任务，专家的"忙闲"分布会跟着变——这也是为什么同一个 MoE 模型在不同任务上的实际速度会有波动。）</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> shared expert、expert group 是什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>纯 MoE 有个隐患：有些"通用本事"（语法、常识）每个 token 都用得到，让它去挤那 k 个名额有点浪费。于是 DeepSeek 等模型加了 <strong>shared expert（共享专家）</strong>：一个<strong>所有 token 都过</strong>的常驻专家，专管通用部分，再让 router 在剩下的专家里挑 k 个管"专门"部分。<span class="mono">build_moe_ffn</span> 里也支持 <strong>expert group（专家分组）</strong>：先把专家分成几组、先选组再在组内选专家（源码里的 <span class="mono">n_expert_groups</span> / <span class="mono">n_group_used</span>），这在专家数特别多（上百个）时能让路由更高效、也更利于把一组专家放在同一张卡上。这些变体的共同点是：都在"路由怎么挑、挑出来怎么组合"上做文章，而底层那个 <span class="mono">ggml_mul_mat_id</span> 的稀疏算法，始终不变。这也是读 ggml MoE 源码的一个好心态：别被各种模型五花八门的变体绕晕，抓住"打分 -&gt; 选 top-k -&gt; <span class="mono">ggml_mul_mat_id</span> 稀疏算 -&gt; 加权合并"这条主线，剩下的都是在这条线上加花样。举个具体的：DeepSeek-V3 有 256 个路由专家 + 1 个 shared expert、每 token 选 8 个；Mixtral 是 8 选 2；各家配置千差万别，但你拿这条主线去套，每一个都能对上号——这正是"理解机制远比记住配置更重要"的绝佳例子，也是这门课从头到尾想传达的态度。</p>
  </div>
</details>

<table class="t">
  <tr><th>维度</th><th>Mixtral 8x7B</th><th>DeepSeek-V3</th><th>不变的主线</th></tr>
  <tr><td>专家总数 <span class="mono">n_expert</span></td><td>8</td><td>256 (+1 shared)</td><td>全部权重都要驻留显存</td></tr>
  <tr><td>每 token 选 <span class="mono">n_expert_used</span></td><td>2</td><td>8</td><td>单步只算被选中的那几个</td></tr>
  <tr><td>shared expert</td><td>无</td><td>有 1 个常驻</td><td>都在"怎么挑、怎么合"上做文章</td></tr>
  <tr><td>稀疏算子</td><td><span class="mono">ggml_mul_mat_id</span></td><td><span class="mono">ggml_mul_mat_id</span></td><td>底层算法完全一样</td></tr>
</table>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>MoE = 把一层 FFN 拆成 N 个专家，每 token 只走 k 个（如 8 选 2）：参数容量像大模型、单 token 算力像小模型。</li>
    <li>路由：router 小线性层打分 -&gt; <span class="mono">ggml_soft_max</span> -&gt; <span class="mono">ggml_argsort_top_k</span> 选 top-k -&gt; 取门控权重归一化。</li>
    <li>稀疏：<span class="mono">ggml_mul_mat_id</span> 靠 <span class="mono">ids</span> 间接寻址，只算被选中的专家（算力 = n_expert_used / n_expert），不是"算完再扔"。</li>
    <li>取舍：用激活稀疏换参数容量；代价是所有专家权重都要驻留显存（吃显存）+ 路由导致访存不规整（吃带宽）。</li>
    <li>变体：负载均衡（训练期 aux loss）、shared expert（通用常驻）、expert group（先选组）——底层稀疏算法不变。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  MoE 和上一课的投机解码，骨子里是同一种智慧的两面：<strong>不是所有计算都同等重要，找出真正需要的那部分、只算它</strong>。投机解码在"时间"维度上偷懒（猜对的步骤不用真算），MoE 在"参数"维度上偷懒（不相关的专家不用激活）。这种"条件计算 / 稀疏激活"的思路，正在成为大模型继续变大的主要出路——因为稠密地把每个参数都用上，算力很快就撑不住了。往深一层看，这也很像生物大脑：你读这行字时，并没有点亮整个大脑，而只激活了相关的少数区域。顺便说，MoE 也提醒我们一件事：模型变强未必靠"让每个零件都更努力"，也可以靠"把零件分工得更聪明"——这种结构性的进步，往往比单纯堆算力更划算，也正是读底层实现最有意思的回报：你看到的不只是"怎么算得快"，还有"为什么这样组织最聪明"。下一课我们换一个维度：多模态——让模型不只读 token，还能"看见"图像，看 ggml 怎么把一张图变成模型能懂的 embedding。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Every FFN (feed-forward network, L11) so far has every token dutifully pass through the whole thing end to end. But the largest open models today (Mixtral, DeepSeek, Qwen-MoE...) almost all do otherwise - they split one FFN into dozens or even hundreds of "experts", and each token picks only two or three to go through. This is MoE (Mixture of Experts). This lesson looks at how ggml implements it: how one token is "routed" to a few experts, and how only those few are computed without wasting compute. This is no fringe trick - it is already the default architecture of today's strongest open models, and understanding it is what lets you grasp why this generation can be both huge and runnable.
</p>
<p style="color:var(--muted);margin-top:.4rem">MoE's magic is one sentence: <strong>the parameter capacity of a big model, but the per-token compute of a small one</strong>. An 8-expert, top-2 MoE layer has about the parameters of 8 FFNs; yet each token goes through only 2 of them - a quarter of the compute of an equally large dense model. The model can thus use its large parameters to "remember" far more, while the compute per inference step drops a great deal.</p>
<p style="color:var(--muted)">Roadmap: first routing (how the router picks experts for each token), with a trace of one token's routing; then how ggml uses <span class="mono">ggml_mul_mat_id</span> to compute only the selected experts; and finally what this "activation sparsity" design trades for what, and where the cost lies.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  MoE rests on a plain bet: <strong>not every token needs the whole network's full skill</strong>. A token about code and a token about poetry might be better handled by different "experts". So MoE splits one big FFN into N small experts, plus a "dispatcher" (the router) that picks the most fitting k for each token. The gain: parameters can pile up huge (each expert remembers something different), yet each token's actual compute is spread over only k experts - <strong>trading "activation sparsity" for "parameter capacity"</strong>. This is why you see phrasing like "60B total parameters, only 10B active": the former is all experts summed, the latter is the few a single token actually passes through. This "huge total, small active" design is why a MoE model downloads as hundreds of GB yet does not demand that much compute to run - you are buying "capacity" and paying in "VRAM". And for that reason MoE models are a bit "VRAM-gated" for hobbyists: inference is light on compute yet demands enough VRAM to hold all the experts up front - which kicks the ball back to L33's multi-GPU and offload.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Think of a dense FFN as a <strong>one-stop general clinic</strong>: every patient who comes must go through every department head to toe - slow and wasteful. MoE is like a <strong>specialized hospital plus a triage desk</strong>: triage (the router) glances at your symptoms and sends you only to the two or three most relevant departments (experts). However many departments the hospital has (however big the parameters), your visit touches only two or three (compute unchanged). The cost is just as plain: every department must keep its doors open and staff on hand (all expert weights must stay resident in VRAM), even if only two are seeing patients right now. The analogy also captures MoE's sweet and bitter: specialization makes "seeing a specialist" faster and sharper (each expert owns a domain), but the cost of running a whole big hospital (staffing every department) is always there. Following the analogy you can also see: if the triage desk is poor (the router is undertrained) and sends an emergency patient to ophthalmology, even the best specialist cannot help - which is why "load balancing / routing quality", covered later, matters so much.
</div>

<h2>Routing: how the router picks experts for a token</h2>
<p>It all starts with a tiny "scoring". An MoE layer has a <strong>router (also called the gate)</strong>, essentially a small linear layer: take the current token's vector and score every expert. The higher the score, the more this token should go to that expert. Think of the router as a "dispatcher": it reads the token's "semantic fingerprint" (that vector) and, on experience learned in training, judges "who should get this job". <span class="mono">ggml_argsort_top_k</span> picks the highest k of those scores (say 2 of 8) - the experts this token goes through this step. A frequently asked point here: why top-k (pick several) rather than top-1 (only the best one)? Because with only one, a single routing mistake has no fallback and training is less stable; picking 2 blends two experts' outputs by weight, which both tolerates errors and can express the subtler semantics of "this token is between two categories". Here are the real routing lines in ggml (from <span class="mono">build_moe_ffn</span>):</p>
<pre class="code"><span class="cm">// MoE routing (simplified from src/llama-graph.cpp build_moe_ffn)</span>
logits = <span class="fn">build_lora_mm</span>(gate_inp, cur);        <span class="cm">// router linear: one score per expert [n_expert, n_tokens]</span>
probs  = <span class="fn">ggml_soft_max</span>(logits);                 <span class="cm">// turn into probabilities (some models use sigmoid)</span>
selected = <span class="fn">ggml_argsort_top_k</span>(probs, n_expert_used); <span class="cm">// pick the top-k experts (e.g. 2 of 8)</span>
weights = <span class="fn">ggml_get_rows</span>(probs, selected);       <span class="cm">// take those k experts' gating weights</span>
weights = <span class="fn">normalize</span>(weights);                   <span class="cm">// normalize so the k weights sum to 1</span></pre>
<p>Line by line: <span class="mono">build_lora_mm(gate_inp, cur)</span> is the router linear layer, scoring every expert for each token (<span class="mono">logits</span>); <span class="mono">ggml_soft_max</span> turns scores into probabilities; <span class="mono">ggml_argsort_top_k</span> picks the highest <span class="mono">n_expert_used</span> (the "2" in "2 of 8"); <span class="mono">ggml_get_rows</span> pulls out those k experts' weights and normalizes them - later, when combining expert outputs, this is the weighting. The whole router is tiny: against the experts' own big matmuls, its cost is nearly negligible. Worth stressing: the router is not a hardcoded rule but <strong>learned</strong> - on vast data the model itself learns "what kind of token goes to which expert". So you cannot know in advance what an expert "specializes" in - it might have become "the punctuation one", "the numbers one", "the some-language one", or some internal division of labor with no pattern a human can see. This is what makes MoE both magical and a bit of a black box: the division of labor is emergent, not designed.</p>
<p>Freezing one token's routing into a picture is clearest:</p>
<div class="trace">
  <div class="tcap"><b>Trace one token's MoE routing</b>: the router scores 8 experts, picks the top-2 (with weights); only those 2 experts actually compute, and a weighted sum of their outputs forms the result (illustrative).</div>
<svg viewBox="0 0 680 320" width="100%" role="img" aria-label="MoE routing example: one token is scored by the router, top-2 of 8 experts are selected (with weights), each computes, and a weighted sum forms the output">
<g font-family="ui-monospace,monospace">
<rect x="30" y="150" width="72" height="40" rx="6" fill="#ffffff" stroke="#c2630e"/><text x="66" y="174" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">token</text>
<line x1="102" y1="170" x2="138" y2="170" stroke="#9aa6b2" stroke-width="1.6"/><path d="M 144 170 L 136 166 L 136 174 z" fill="#9aa6b2"/>
<rect x="146" y="142" width="92" height="56" rx="6" fill="#ffffff" stroke="#2563eb"/><text x="192" y="168" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">router</text><text x="192" y="185" text-anchor="middle" fill="#5b6470" font-size="10">gate</text>
<text x="332" y="18" text-anchor="middle" fill="#5b6470" font-size="11">8 experts (FFN)</text>
<line x1="238" y1="170" x2="296" y2="41" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="28" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="45" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E0</text>
<line x1="238" y1="170" x2="296" y2="73" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="60" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="77" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E1</text>
<line x1="238" y1="170" x2="296" y2="105" stroke="#7c3aed" stroke-width="1.8"/>
<path d="M 302 105 L 294 101 L 294 109 z" fill="#7c3aed"/>
<rect x="300" y="92" width="64" height="26" rx="5" fill="#ece3fb" stroke="#7c3aed"/><text x="332" y="109" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">E2</text>
<text x="372" y="109" fill="#7c3aed" font-weight="700" font-size="11">w1=0.7</text>
<line x1="238" y1="170" x2="296" y2="137" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="124" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="141" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E3</text>
<line x1="238" y1="170" x2="296" y2="169" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="156" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="173" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E4</text>
<line x1="238" y1="170" x2="296" y2="201" stroke="#7c3aed" stroke-width="1.8"/>
<path d="M 302 201 L 294 197 L 294 205 z" fill="#7c3aed"/>
<rect x="300" y="188" width="64" height="26" rx="5" fill="#ece3fb" stroke="#7c3aed"/><text x="332" y="205" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">E5</text>
<text x="372" y="205" fill="#7c3aed" font-weight="700" font-size="11">w2=0.3</text>
<line x1="238" y1="170" x2="296" y2="233" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="220" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="237" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E6</text>
<line x1="238" y1="170" x2="296" y2="265" stroke="#9aa6b2" stroke-width="1" stroke-dasharray="3 3"/>
<rect x="300" y="252" width="64" height="26" rx="5" fill="#ffffff" stroke="#cdd5df"/><text x="332" y="269" text-anchor="middle" fill="#9aa6b2" font-weight="400" font-size="11">E7</text>
<text x="332" y="296" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">top-2 selected</text>
<text x="332" y="312" text-anchor="middle" fill="#9aa6b2" font-size="10">other 6 not computed</text>
<line x1="426" y1="105" x2="470" y2="170" stroke="#7c3aed" stroke-width="1.6"/>
<line x1="426" y1="201" x2="470" y2="170" stroke="#7c3aed" stroke-width="1.6"/>
<path d="M 474 170 L 466 166 L 466 174 z" fill="#7c3aed"/>
<rect x="476" y="132" width="84" height="76" rx="6" fill="#ffffff" stroke="#7c3aed"/><text x="518" y="164" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">weighted sum</text><text x="518" y="182" text-anchor="middle" fill="#5b6470" font-size="10">w1*E2 + w2*E5</text>
<line x1="560" y1="170" x2="588" y2="170" stroke="#c2630e" stroke-width="1.6"/><path d="M 594 170 L 586 166 L 586 174 z" fill="#c2630e"/>
<rect x="596" y="150" width="64" height="40" rx="6" fill="#ffffff" stroke="#c2630e"/><text x="628" y="174" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">output</text>
</g></svg>
</div>

<h2>Computing sparsely: ggml_mul_mat_id runs only the selected experts</h2>
<p>With experts chosen, here comes MoE's most crucial and most easily-assumed step: <strong>how to "compute only the selected experts"</strong>. A lazy implementation might "compute all 8 experts then throw away the 6 not chosen" - then MoE saves no compute at all. In fact the naive "compute all then discard" really existed in some early implementations, with the effect "parameters got sparse but compute did not", squandering MoE's benefit. So "true sparsity" is what makes or breaks MoE in practice, and the reason ggml built a dedicated <span class="mono">ggml_mul_mat_id</span> op just for it. ggml uses that op to be genuinely sparse:</p>
<pre class="code"><span class="cm">// sparse expert matmul (src/llama-graph.cpp build_moe_ffn)</span>
<span class="cm">// selected = the ids picked last section: which experts each token chose</span>
up   = <span class="fn">ggml_mul_mat_id</span>(up_exps,   cur, selected); <span class="cm">// up projection on selected experts only</span>
gate = <span class="fn">ggml_mul_mat_id</span>(gate_exps, cur, selected); <span class="cm">// gate projection on selected experts only</span>
act  = <span class="fn">silu</span>(gate) * up;                           <span class="cm">// activation (SwiGLU)</span>
out  = <span class="fn">ggml_mul_mat_id</span>(down_exps, act, selected); <span class="cm">// down projection back to the original dim</span>
<span class="cm">// finally combine the k experts' out by the router's weights (weighted sum)</span></pre>
<p>The crux is the <span class="mono">id</span> in <span class="mono">ggml_mul_mat_id</span>: beyond a normal matmul it takes an extra <span class="mono">ids</span> tensor (the <span class="mono">selected</span> from last section), telling the multiply "which experts' weights each token should multiply by". So it does not multiply the token by all 8 experts but <strong>indirectly addresses by ids</strong>, pulling out only the 2 chosen experts to compute. <span class="mono">up_exps</span>/<span class="mono">gate_exps</span>/<span class="mono">down_exps</span> are big tensors packing all experts' weights together, and <span class="mono">ids</span> is the index of "which slices to use". Compute thus really drops to <span class="mono">n_expert_used / n_expert</span> (a quarter for 2 of 8), rather than "compute then discard". Let me clear up a common misconception: MoE saves the <strong>FFN's compute</strong>, not the whole model's. Attention, normalization, embeddings are still computed in full for every token - only the FFN part is sparsified. But in large models the FFN is exactly where most parameters and compute sit, so sparsifying it yields a big overall gain. (This is also why MoE almost only touches the FFN and leaves attention alone - attention is not the parameter-heavy part anyway, so splitting it gains little and would disrupt the flow of global information.)</p>
<div class="cols">
  <div class="col"><h4>equal-capacity dense</h4><p>the same parameters made into one big network, every token passing through all of it: compute = 8 experts' worth.</p></div>
  <div class="col"><h4>MoE (2 of 8)</h4><p>each token activates only 2 experts: compute = 2 experts' worth, a quarter of the equal-capacity dense.</p></div>
  <div class="col"><h4>in common</h4><p>the same parameter count (all 8 experts resident in VRAM); the only difference is "how many are actually activated per step".</p></div>
</div>

<h2>Why design it this way: sparsity for capacity</h2>
<p>Connecting the last two sections, MoE's trade-off is clear: it bets that <strong>"knowledge can be divided and conquered"</strong> - rather than have every token pass through one all-encompassing big FFN, let it pass through only a few most-relevant experts. The model can then pile parameters up enormously (each expert owning a patch of "knowledge"), while a single token's compute depends only on "how many are picked", not "how many experts there are in total". This is why the flagship open models of the last couple of years are almost uniformly MoE: under a fixed inference-compute budget, MoE can pack in far more parameters than a dense model, and so be "smarter". Another metaphor: a dense model is like hiring one generalist who knows a bit of everything; MoE is like hiring a panel of specialists plus a triage desk - the team's total knowledge is far greater, but each query only disturbs the relevant two. Behind MoE's rise is a plain economics: training and inference compute are expensive, while parameters (VRAM) are relatively cheap. MoE rides exactly that line - spend cheap VRAM to save expensive compute, building a stronger model under the same compute budget. That is why, once someone showed MoE scales, the whole open-source community followed almost overnight. Of course, MoE is no panacea: it is harder to train (load balancing and stability are pitfalls), pickier about VRAM, and at small scale not necessarily a better deal than dense. It is a product of the "large-model era" - MoE truly shines only when you want to pile parameters up to a scale a dense architecture cannot bear.</p>
<p>But there is no free lunch, and MoE's cost is concrete, landing exactly on the sore points of earlier lessons. <strong>VRAM</strong>: though only 2 experts compute per step, all 8 experts' weights <strong>must stay in VRAM</strong> - you do not know which the next token will route to. So MoE's memory footprint is counted by "total parameters", often frighteningly large (which is why MoE models lean so heavily on L33's multi-GPU / offload). Helpfully, quantization (L06/L12/L29) does a lot here: squeeze expert weights to 4-bit and VRAM shrinks fourfold, which is what squeezes many otherwise-too-big MoEs onto consumer GPUs at all.</p>
<p><strong>Irregular memory access</strong>: neighboring tokens may route to entirely different experts, and <span class="mono">ggml_mul_mat_id</span>'s indirect addressing makes memory access jumpier than a dense matmul and less cache-friendly (echoing the memory-bound theme of L31/L32). So MoE is a "saves compute but costs more VRAM and bandwidth" trade - it moves the bottleneck from "computing" toward "storing and moving". This also explains a real-world observation: at equal active parameters, a MoE and a dense model do not necessarily run at the same speed, but MoE fits a far smarter model onto the same GPU. One subtlety more: in prefill a batch of tokens is computed together, and different tokens routing to different experts can actually exercise all experts with high parallelism; while in decode a single token at a time often activates only two experts, with low hardware utilization - echoing once more the prefill-vs-decode of L18/L30.</p>

<h2>Deeper: load balancing and MoE variants</h2>
<p>Two last folds for two issues you cannot avoid when actually deploying MoE.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why bother with "load balancing"? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Suppose the router learned badly: it routes the vast majority of tokens to the same two experts, while the other six are barely visited. Two bad things follow: the neglected experts are undertrained and waste their parameters (capacity wasted); the overloaded experts become a bottleneck and may exceed their "capacity" (in some training and distributed-inference implementations an expert can take at most so many tokens per batch, beyond which tokens are dropped; but note llama.cpp inference drops nothing - it always computes every selected expert). So training a MoE usually adds a <strong>load-balancing auxiliary loss</strong>, encouraging the router to spread tokens evenly across experts; a "capacity factor" leaves each expert some headroom. At <strong>inference</strong> this is already baked into the weights (the router is trained), but understanding it helps you read many MoE designs - why some models use expert groups or token dropping. ggml at inference does not compute the aux loss, but how balanced the router's distribution is directly affects expert utilization on real hardware. There is also an inference-side reality, especially when experts are sharded across several GPUs (L33's expert parallelism): if a batch of tokens (say, the multiple requests a server handles at once) happen to all pile onto the same expert, the GPU holding it becomes a serial bottleneck while others sit idle - so under multi-GPU high-concurrency serving, MoE throughput is sometimes less stable than a dense model with equal active parameters. (On a single GPU it is the opposite: all piling onto one expert is just one regular large matmul, no bottleneck.) (Incidentally, because the router is fixed at inference, different inputs and different tasks shift the experts' busy/idle distribution - which is why the same MoE model's real-world speed varies across tasks.)</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> What are shared experts and expert groups? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Pure MoE has a pitfall: some "general skills" (grammar, common sense) are used by every token, and making them compete for the k slots is a bit wasteful. So models like DeepSeek add a <strong>shared expert</strong>: a resident expert <strong>every token passes through</strong>, handling the general part, leaving the router to pick k of the rest for the "specialized" part. <span class="mono">build_moe_ffn</span> also supports <strong>expert groups</strong>: split experts into groups, pick a group first then experts within it (<span class="mono">n_expert_groups</span> / <span class="mono">n_group_used</span> in the source), which keeps routing efficient when there are very many experts (hundreds) and helps put a group of experts on the same GPU. What these variants share: they all play with "how routing picks and how the picks are combined", while the underlying <span class="mono">ggml_mul_mat_id</span> sparse algorithm stays the same. This is also a good mindset for reading ggml's MoE source: do not get dizzy in the models' assorted variants - hold the through-line "score -&gt; pick top-k -&gt; <span class="mono">ggml_mul_mat_id</span> sparse compute -&gt; weighted combine", and the rest are just flourishes on that line. Concretely: DeepSeek-V3 has 256 routed experts + 1 shared expert, picking 8 per token; Mixtral is 2 of 8; configs differ wildly, but lay this through-line over each of them and every one lines up - a fine example of "understanding the mechanism beats memorizing the configs", and the attitude this course keeps trying to convey.</p>
  </div>
</details>

<table class="t">
  <tr><th>Dimension</th><th>Mixtral 8x7B</th><th>DeepSeek-V3</th><th>The unchanging through-line</th></tr>
  <tr><td>Total experts <span class="mono">n_expert</span></td><td>8</td><td>256 (+1 shared)</td><td>all weights must stay resident in VRAM</td></tr>
  <tr><td>Picked per token <span class="mono">n_expert_used</span></td><td>2</td><td>8</td><td>each step computes only the chosen few</td></tr>
  <tr><td>shared expert</td><td>none</td><td>1 resident</td><td>all play with "how to pick and combine"</td></tr>
  <tr><td>sparse op</td><td><span class="mono">ggml_mul_mat_id</span></td><td><span class="mono">ggml_mul_mat_id</span></td><td>the underlying algorithm is identical</td></tr>
</table>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>MoE = split one FFN into N experts, each token taking only k (e.g. 2 of 8): parameter capacity like a big model, per-token compute like a small one.</li>
    <li>Routing: a small router linear scores -&gt; <span class="mono">ggml_soft_max</span> -&gt; <span class="mono">ggml_argsort_top_k</span> picks top-k -&gt; take gating weights, normalize.</li>
    <li>Sparse: <span class="mono">ggml_mul_mat_id</span> uses <span class="mono">ids</span> to indirectly address and compute only the selected experts (compute = n_expert_used / n_expert), not "compute then discard".</li>
    <li>Trade: activation sparsity for parameter capacity; the cost is all expert weights resident in VRAM (memory) + routing's irregular access (bandwidth).</li>
    <li>Variants: load balancing (training-time aux loss), shared expert (general resident), expert groups (pick a group first) - the underlying sparse algorithm is unchanged.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  MoE and last lesson's speculative decoding are two faces of the same wisdom at heart: <strong>not all computation is equally important - find the part you actually need and compute only that</strong>. Speculative decoding economizes in the "time" dimension (steps guessed right need no real compute); MoE economizes in the "parameter" dimension (irrelevant experts need not activate). This "conditional computation / sparse activation" idea is becoming the main way for large models to keep growing - because using every parameter densely soon outstrips the compute budget. Deeper still, this resembles a biological brain: reading this line, you do not light up the whole brain, only the few relevant regions. By the way, MoE reminds us of something: a model gets stronger not only by "making every part work harder" but also by "dividing the parts' labor more cleverly" - this structural progress is often a better deal than piling on compute, and it is the most interesting reward of reading low-level implementations: you see not only "how to compute fast" but "why this organization is the smartest". Next lesson we switch dimensions: multimodal - letting the model not only read tokens but "see" images, watching how ggml turns a picture into an embedding the model understands.
</div>
""",
}

LESSON_36 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
到这里为止，你脑子里的 LLM 还只会读文字：一段 prompt 切成 token、查 embedding、过几十层 transformer。可现在的模型动不动就能"看图说话"——你发一张图，它就能描述、问答、读表格。它是怎么把"一张图"塞进一个只认 token 的模型里的？答案出乎意料地朴素：<strong>LLM 的输入层根本不在乎喂进来的 embedding 向量是从文字来的还是从图像来的</strong>。多模态要做的，就是把"一张图"也变成<strong>一串 embedding</strong>，和文本 token 的 embedding 拼在同一个序列里，一起送进 <span class="mono">llama_decode</span>。
</p>
<p style="color:var(--muted);margin-top:.4rem">这一课看 llama.cpp 的 <span class="mono">mtmd</span>（multimodal）子系统怎么干这件事。核心只有一句：模型主体一个字都不用改，多模态全靠在它<strong>前面</strong>接一段"翻译"流水——视觉编码器（clip / ViT）把图像压成视觉特征，projector（mmproj）再把这些特征<strong>投影到 LLM 的 embedding 空间</strong>，得到 N 个"看起来就像 token embedding"的向量，按 prompt 里 <span class="mono">&lt;image&gt;</span> 占位的地方插进序列。LLM 拿到这串向量，根本分不出哪些来自文字、哪些来自图像——它只管照常往下算。这一点初看会让人愣一下：模型明明"看懂"了图，怎么会"不知道那是图"？但这恰恰是这套设计最聪明的地方——把"看懂"这件事完全外包给了前面的视觉流水，LLM 只负责它最擅长的那件事："在向量序列上做推理"。</p>
<p style="color:var(--muted)">路线图：先看整条 <span class="mono">mtmd</span> 管线（切 chunk -> 编码 -> 取 embedding -> 和文本交织 decode），配一张追踪图看"一张图进 LLM"；再单独讲 projector 这座<strong>桥</strong>为什么不可少；最后两个折叠深挖 clip 的 ViT 内部，以及图像 embedding 在序列里怎么"占位置"、怎么进 KV cache。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  多模态听起来很玄，但拆开看就一句话：<strong>把"非文字"也翻译成 LLM 能吃的 embedding</strong>。回想 L04/L20：文本进 LLM 的第一步，是把每个 token 查成一个 embedding 向量；transformer 主体从头到尾处理的都是<strong>向量序列</strong>，它压根不知道"token"长什么样。这就留了一个口子——只要你能把一张图也变成<strong>同样形状、同样空间</strong>的一串向量，就能直接插进序列里，模型照单全收。所以 llama.cpp 没有去改动 LLM 本体，而是在前面挂一个 <span class="mono">mtmd</span> 流水：clip(ViT) 把图像编码成视觉特征，projector 把特征投影到 LLM 的 embedding 维度，产出 N 个 image embedding，按 <span class="mono">&lt;image&gt;</span> 标记插进文本序列。理解这一点，你就理解了当下绝大多数"多模态大模型"的套路：它们几乎都是"一个现成 LLM + 一个视觉编码器 + 一座 projector 桥"拼出来的——主体没变，只是学会了"看"。这种"冻结主体、外挂适配器"的思路，你其实在 L24 的 LoRA 里已经见过一次——区别只是 LoRA 适配的是"风格/任务"，而这里适配的是"输入模态"。同一个朴素的工程智慧——别动那个又大又难训的主体，只在它周围加可插拔的小零件——在两个完全不同的场景里各结了一次果。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  想象一位<strong>只读中文</strong>的专家（LLM），你要让他理解一张<strong>外文图表</strong>。你不会逼他去学外语（改模型代价太大），而是请一个<strong>翻译官</strong>：翻译官先<strong>看懂</strong>这张图（clip/ViT 把图像编码成视觉特征），再把它<strong>转述成一段中文</strong>（projector 投影到 LLM 的 embedding 空间），然后把这段中文<strong>插进你和专家的对话里</strong>。专家读到这段"中文"，自然地接着往下聊——他根本不知道这段话原本是一张图。这里有两个关键角色：<strong>看懂图的眼睛</strong>（clip）和<strong>把所见转成专家母语的嘴</strong>（projector）。眼睛再好，要是不会说专家的语言，专家也听不懂——所以 projector 这座"翻译桥"才是多模态能不能接上的关键。
</div>
<h2>总管线：mtmd 怎么把一张图喂进 LLM</h2>
<p>先把整条流水看一遍。用户给的是"图文混排"的输入——一段带 <span class="mono">&lt;image&gt;</span> 标记的文字，外加一张（或几张）图的像素。mtmd 把它走成五步：(1) <span class="mono">mtmd_init_from_file</span> 加载 projector（那个单独的 mmproj 文件）；(2) <span class="mono">mtmd_tokenize</span> 把输入<strong>切成一串 chunk</strong>——文字段落是 <span class="mono">TEXT</span> chunk，每张图变成一个 <span class="mono">IMAGE</span> chunk（音频则是 <span class="mono">AUDIO</span>）；(3) 对每个 image chunk 跑 <span class="mono">mtmd_encode_chunk</span>（内部就是 clip + projector）；(4) <span class="mono">mtmd_get_output_embd</span> 取出编码好的视觉 embedding；(5) 把 text chunk 的 token 和 image chunk 的 embedding<strong>按原顺序交织</strong>，一段段喂进 <span class="mono">llama_decode</span>。先看切 chunk 这一步：</p>
<pre class="code"><span class="cm">// 把"文字 + &lt;image&gt; 标记 + 图像 bitmap"切成有序 chunk (简化自 tools/mtmd/mtmd.h)</span>
mtmd_input_chunks * chunks = <span class="fn">mtmd_input_chunks_init</span>();
<span class="fn">mtmd_tokenize</span>(mtmd_ctx, chunks, &amp;text, bitmaps, n_bitmaps);
<span class="cm">// chunks 里现在是按原文顺序排好的一串:</span>
<span class="cm">//   [TEXT "这张图是"] [IMAGE 一张图] [TEXT "里面有什么?"]</span>
<span class="cm">// 每个 chunk 带一个类型: TEXT / IMAGE / AUDIO</span></pre>
<p>注意 prompt 里那个 <span class="mono">&lt;image&gt;</span>（源码里默认标记其实是 <span class="mono">&lt;__media__&gt;</span>）：它就是个<strong>占位符</strong>，告诉 mtmd"这张图该插在文字的哪个位置"。源码里那段注释举的例子很直白：形如"here is an image: &lt;__media__&gt; ..."这样一句，会被切成三个 chunk——标记前的文字、图像本身、标记后的文字，顺序和原文严丝合缝。切完 chunk，真正的重头戏是把 image chunk 编码成 embedding、再和文字交织着 decode。llama.cpp 把这套逻辑打包进了一个 helper，它的注释几乎就是整条管线的伪代码：</p>
<pre class="code"><span class="cm">// 逐个 chunk: 文字直接 decode, 图像先 encode 再 decode (简化自 mtmd-helper.cpp)</span>
<span class="kw">for</span> (each chunk : chunks) {
    <span class="kw">if</span> (<span class="fn">type</span>(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        <span class="fn">llama_decode</span>(lctx, <span class="fn">batch_of</span>(chunk.tokens));   <span class="cm">// 文字: 老路, 查 embedding 再算</span>
    } <span class="kw">else</span> {                                          <span class="cm">// IMAGE / AUDIO chunk:</span>
        <span class="fn">mtmd_encode_chunk</span>(mtmd_ctx, chunk);          <span class="cm">// 1) 内部跑 clip(ViT) + projector</span>
        <span class="kw">float</span> * embd = <span class="fn">mtmd_get_output_embd</span>(mtmd_ctx); <span class="cm">// 2) 取出 N 个视觉 embedding</span>
        <span class="fn">llama_decode</span>(lctx, <span class="fn">batch_of_embd</span>(embd));     <span class="cm">// 3) 直接把 embedding 喂进去</span>
    }
}</pre>
<p>看出门道了吗？文字和图像最后都落到同一个 <span class="mono">llama_decode</span> 上——区别只在"喂进去的是 token（让模型自己查 embedding）还是已经算好的 embedding（直接用）"。<span class="mono">llama_batch</span> 早就同时支持这两种输入（还记得 L18 那个 batch 里既能放 <span class="mono">token</span> 也能放 <span class="mono">embd</span> 吗？），所以图像 embedding 能<strong>无缝</strong>地塞进序列，模型主体一行都不用改。这套设计的妙处在于<strong>复用</strong>：mtmd 没有为图像另起一条推理通路，而是把图像"伪装"成 embedding、复用了文本那一整套 batch、KV cache、采样逻辑——多模态于是变成了一个"前处理"问题，而不是"重写引擎"问题。把"一张图进 LLM"的全过程定格成一条流水：</p>
<div class="cols">
  <div class="col"><h4>文字 chunk</h4><p>token id 序列 -&gt; <span class="mono">llama_decode</span> 内部查 embedding 表 -&gt; 算。走的是 L04/L20 的老路。</p></div>
  <div class="col"><h4>图像 chunk</h4><p>像素 -&gt; clip + projector 算出 embedding -&gt; 直接把 embedding 喂进 <span class="mono">llama_decode</span>。embedding 已备好，跳过查表。</p></div>
</div>
<div class="trace">
  <div class="tcap"><b>追踪一张图进 LLM</b>：像素先过 clip(ViT) 编成视觉特征，projector 投影成 N 个 image embedding，再和文字 token 按原顺序交织成一个序列，整体送进 llama_decode（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 一张图</h5>
      <div class="cellrow"><span class="vc">336x336 像素</span></div>
      <div class="tlab">原始 bitmap</div></div>
    <div class="op">切 patches<br>clip(ViT)</div>
    <div class="stn"><h5>② 视觉特征</h5>
      <div class="cellrow"><span class="vc blue">576 个特征向量</span></div>
      <div class="tlab">ViT 编码输出</div></div>
    <div class="op">projector<br>(mmproj)</div>
    <div class="stn"><h5>③ image embedding</h5>
      <div class="cellrow"><span class="vc hot">N 个 LLM embedding</span></div>
      <div class="tlab">投影到 LLM 维度</div></div>
    <div class="op">按 &lt;image&gt;<br>占位插入</div>
    <div class="stn"><h5>④ 交织序列</h5>
      <div class="cellrow"><span class="vc">文字 emb + 图 emb + 文字 emb</span></div>
      <div class="tlab">一个统一序列</div></div>
    <div class="op">一起<br>decode</div>
    <div class="stn"><h5>⑤ llama_decode</h5>
      <div class="cellrow"><span class="vc blue">LLM 照常前向</span></div>
      <div class="tlab">分不出图和字</div></div>
  </div>
</div>
<p>值得一提的是，这套"切 chunk -&gt; 编码 -&gt; 交织"的机制对<strong>音频</strong>一视同仁：把语音切成帧、过一个音频编码器（如 Whisper 的前端）、再过 projector 投影成 embedding，走的是和图像完全相同的通路——这就是为什么 <span class="mono">mtmd</span> 的 chunk 类型里 <span class="mono">AUDIO</span> 和 <span class="mono">IMAGE</span> 并列。换句话说，mtmd 的设计目标从一开始就不是"支持图像"，而是"支持任意能被编码成 embedding 的模态"。理解了图像这一条，音频、乃至将来更多模态，都是同一个模子里刻出来的。</p>
<h2>projector：把"看见的"翻译成 LLM 的母语</h2>
<p>上面那步 <span class="mono">mtmd_encode_chunk</span> 内部分两半：先 clip(ViT) 把图像"看"成视觉特征，再 projector 把特征"翻译"成 LLM 能读的 embedding。为什么非要这第二步？因为 clip 输出的视觉特征，和 LLM 的 token embedding<strong>根本不在一个空间</strong>——维度可能不一样（clip 也许输出 1024 维，LLM 要 4096 维），数值分布、语义含义更是两套体系。直接把 clip 的输出塞进 LLM，就像把一段没翻译的外文丢给只懂中文的专家，他只会一脸茫然。projector（就是那个单独的 mmproj 文件）就是这座桥：一个小网络，把视觉特征<strong>投影到 LLM 的 embedding 维度和空间</strong>，让它"看起来、用起来都像一个 token embedding"。打个比方，clip 的输出像一段"视觉速记"，每个数字的含义是按视觉任务编排的；LLM 的 embedding 空间则是按语言任务长出来的，同样长度的向量，"坐标系"也完全不同。projector 干的就是坐标变换：把视觉速记重新表达进语言的坐标系里，让"图里有只猫"这件事，落在 LLM 一向用来表示"猫"的那片向量空间附近。</p>
<p>两个关键的"对齐"由两个函数把关：<span class="mono">clip_n_mmproj_embd(ctx)</span> 返回 projector 的输出维度——它<strong>必须等于</strong> LLM 的 embedding 维度，否则向量塞不进序列；<span class="mono">clip_n_output_tokens(ctx, img)</span> 返回<strong>这一张图会占几个 embedding token</strong>（也就是前面 trace 里那个 N）。这个 N 不是随便定的：图越大、patch 越多，N 越大；有些 projector（resampler 类）还会<strong>主动压缩</strong> N，把几百个 patch 特征汇聚成几十个 embedding，省 KV cache、也省算力。这个 N 直接决定了一张图的"开销"：N 个 embedding 就要占 N 个序列位置、写 N 份 KV——所以同样一张图，projector 把它压成 64 个 embedding 还是铺成 576 个，对显存和速度的影响是数量级的。这也是高分辨率多模态模型的核心权衡之一：看得越细（patch 越多、N 越大）越准，但序列越长、越慢、越吃显存。</p>
<pre class="code"><span class="cm">// projector 内部: clip 先编码, 投影维度由 mmproj 决定 (简化自 tools/mtmd/clip.h)</span>
<span class="kw">int</span> n_embd   = <span class="fn">clip_n_mmproj_embd</span>(clip_ctx);       <span class="cm">// projector 输出维度 == LLM embedding 维度</span>
<span class="kw">int</span> n_tokens = <span class="fn">clip_n_output_tokens</span>(clip_ctx, img); <span class="cm">// 这张图占几个 embedding token</span>
std::vector&lt;float&gt; out_vec;
<span class="fn">clip_image_encode</span>(clip_ctx, n_threads, img, out_vec); <span class="cm">// 跑 ViT + projector, 输出 n_tokens x n_embd</span>
<span class="cm">// out_vec 现在是 n_tokens 个、每个 n_embd 维的视觉 embedding,</span>
<span class="cm">// 形状和 n_tokens 个 token 的 embedding 完全一样 -&gt; 直接进序列</span></pre>
<p>常见的 projector 有三档复杂度：<strong>线性层</strong>（一个矩阵乘，最简单，早期 LLaVA 用）、<strong>两层 MLP</strong>（多一层非线性，对齐更好，现在很常见）、<strong>resampler / cross-attention</strong>（用一组可学习 query 把变长的 patch 特征"重采样"成固定个数的 embedding，能压 N、也能处理任意分辨率，Qwen-VL 等用）。选哪一档是<strong>精度和成本的权衡</strong>：线性最省但表达力弱，resampler 最灵活但自己也带一摞参数和算力。不管哪一档，它的职责都一样：<strong>把视觉特征对齐到 LLM 的 embedding 空间</strong>。这也是为什么 mmproj 是个<strong>单独的文件</strong>、要单独加载——它是"某个视觉编码器 + 某个 LLM"这对组合<strong>专门训练</strong>出来的桥，换一个 LLM 或换一个 clip，桥就得重训。理解这一点，你就明白为什么下载多模态模型时，除了主模型那个大 GGUF，还得配一个小小的 mmproj 文件：少了那座桥，模型就只剩"读字"的本事，"看图"的能力无从谈起。一个实用的小知识：HuggingFace 上多模态模型的 GGUF 仓库里，那个名字带 mmproj 的小文件就是它，通常几百 MB 量级，千万别漏下。</p>
<table class="t">
  <tr><th>projector 类型</th><th>结构</th><th>特点</th><th>代表</th></tr>
  <tr><td>线性层</td><td>一个矩阵乘</td><td>最省，表达力弱，固定 N</td><td>早期 LLaVA</td></tr>
  <tr><td>两层 MLP</td><td>线性 + 非线性 + 线性</td><td>对齐更好，现在常见</td><td>LLaVA-1.5+</td></tr>
  <tr><td>resampler</td><td>可学习 query + cross-attention</td><td>能压 N、处理任意分辨率</td><td>Qwen-VL</td></tr>
</table>
<h2>深入：clip 的眼睛 与 图像的"位置"</h2>
<p>两个折叠，补两个真要落地多模态时绕不开的细节。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> clip(ViT) 内部到底在做什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>本课把 clip 当黑盒——给它一张图，它吐出一串视觉特征。掀开看，clip 的视觉编码器就是一个<strong>标准的 Vision Transformer（ViT）</strong>，和你前几部分学的文本 transformer 几乎一个套路，只是把"token"换成了"图像 patch"：(1) 把图切成固定大小的小块（patch，比如 14x14 像素一块），每块拉平、线性投影成一个向量——这就是图像版的"embedding"；(2) 加上位置编码（告诉模型每块在原图的哪个位置）；(3) 过若干层自注意力 + FFN，让每个 patch"看到"全图、聚合出语义。最后每个 patch 对应一个输出向量，合起来就是那串视觉特征。所以一张 336x336 的图、14 像素一块，就是 24x24 = 576 个 patch -&gt; 576 个特征（前面 trace 里的数字就是这么来的）。复杂度也从这来：patch 越多，自注意力的开销越大（O(patch^2)），这正是高分辨率图像为什么贵。为了又看得清、又不让 patch 数爆炸，很多实现会把大图<strong>切成几块</strong>分别编码（image tiling / 切片），再把各块的 embedding 拼起来——这也是为什么有的多模态模型吃一张大图会吐出成百上千个 embedding。真正的实现都在 <span class="mono">tools/mtmd/clip.cpp</span>，里面用 ggml 把这套 ViT 搭了出来——如果你已经读懂了 L16 的文本 build graph，那 clip.cpp 对你不会陌生，无非是换了一种 token。本课不逐行展开它，是因为它对"多模态怎么接进 LLM"这条主线不是重点：重点是它吐出的特征，要靠 projector 那座桥才能进 LLM。顺带一提，正因为 clip 内部也是个 transformer，它同样能用 ggml 那套算子、同样能量化、同样能跑在各种后端上——这就是为什么 llama.cpp 能把视觉编码器和 LLM 装进同一套推理框架，而不必再拉一个 PyTorch 进来。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 图像 embedding 在序列里怎么"占位置"、怎么进 KV cache？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>image embedding 一旦插进序列，对 LLM 来说它就是序列里实打实的 N 个位置——和文本 token 一样要分配 position、一样要写进 KV cache（呼应 L19）。这带来两个要处理的问题。<strong>一是位置编码</strong>：普通文本是一维位置（第 0、1、2 个 token），但图像是二维的（某 patch 在第几行第几列），硬拍平成一维会丢掉空间结构。于是不少模型用 <strong>M-RoPE（多维 RoPE）</strong>，给图像 token 一个能表达"行、列"的多维位置——llama.cpp 里 <span class="mono">mtmd_decode_use_mrope</span> 就是问"这个模型要不要用 M-RoPE"，而 <span class="mono">mtmd_helper_get_n_pos</span> 专门算一串 chunk 占了多少个"位置"（注释里点明：一般 n_pos == n_tokens，但 M-RoPE 下两者不同）。直觉上，M-RoPE 给图像 token 的位置不再是一根数轴上的一个点，而更像棋盘上的一个坐标格——这样模型才知道左上角那块和右下角那块在空间上离得远。<strong>二是注意力掩码</strong>：文本是因果的（只能看前面），但一张图内部的 patch 之间往往要<strong>互相都能看见</strong>（双向），所以有些模型（如 Gemma 3）在 decode 图像段时要临时切成非因果注意力——这正是 <span class="mono">mtmd_decode_use_non_causal</span> 在管的事。理解这两点，你就明白图像进 LLM 不只是"塞 N 个向量"那么简单：它还得在<strong>位置</strong>和<strong>注意力</strong>这两件 transformer 的根本机制上，和文本和谐共处。好在这些 llama.cpp 都替你处理好了，你只要知道：图像 embedding 进了序列，就和文本一样占 KV cache、一样参与后面每个 token 的注意力——这也是为什么图越多、KV cache 涨得越快。这件事在工程上很要命：一张高分辨率图可能就占掉几百上千个位置，几张图下来，KV cache 的占用直追一段长文本。所以多模态服务里，"图片预算"经常得和"上下文预算"一起算——这又一次把你带回 L19 的老问题：序列越长，KV cache 越大，能并发的请求就越少。多模态没有逃开这条铁律，只是让"序列里能有什么"变得更丰富了。所以下次看到"32K 上下文的多模态模型"，你心里要清楚：这 32K 是图和字<strong>共享</strong>的预算，一张高清图就能吃掉一大块。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>核心：LLM 主体只认 embedding 向量、不在乎来自文字还是图像；多模态 = 把"一张图"也变成 N 个 embedding 插进同一序列。</li>
    <li>mtmd 管线：<span class="mono">mtmd_tokenize</span> 切 chunk（TEXT/IMAGE/AUDIO）-&gt; 对 image chunk 跑 <span class="mono">mtmd_encode_chunk</span>(clip+projector) -&gt; <span class="mono">mtmd_get_output_embd</span> 取 embedding -&gt; 和文本交织进 <span class="mono">llama_decode</span>。</li>
    <li>projector(mmproj) 是桥：把 clip 视觉特征投影到 LLM 的 embedding 维度/空间；<span class="mono">clip_n_mmproj_embd</span>=LLM embd 维度，<span class="mono">clip_n_output_tokens</span>=一张图占几个 token。</li>
    <li>mmproj 是单独文件、为"某 clip + 某 LLM"专门训练；少了它模型只会读字、不会看图。</li>
    <li>图像 embedding 照常占 position、进 KV cache（L19）；M-RoPE 处理二维位置、非因果掩码处理图内双向（<span class="mono">mtmd_decode_use_mrope</span> / <span class="mono">_non_causal</span>）。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  多模态这一课最该带走的，不是 clip 或 projector 的细节，而是那个朴素到有点反直觉的事实：<strong>LLM 从没"看见"过图</strong>。它眼里永远只有 embedding 向量序列——多模态的全部魔法，是在它<strong>前面</strong>加了一道翻译，把图像翻成它早就习惯的那种向量。这是一种极有生命力的设计哲学：<strong>不改动核心，只在边界处做适配</strong>。同一个 LLM，配一座视觉桥就能看图、配一座音频桥就能听声、将来配别的桥还能接别的模态——核心始终是那个只懂 embedding 的 transformer。回头看你学过的整条链：L04 的 token、L20 的 embedding、L18 的 batch（能放 token 也能放 embd）、L19 的 KV cache——多模态没有推翻其中任何一块，只是站在它们肩上，把"输入能是什么"往外推了一大步。第七部分到这里，我们已经看了三种"在标准 transformer 之外做文章"的思路：投机解码改的是"怎么更快地出 token"、MoE 改的是"怎么用稀疏换容量"、多模态改的是"输入能是什么"。最后一课要动的是更根本的东西——状态空间模型（Mamba / RWKV）干脆把 transformer 赖以为生的注意力<strong>换掉</strong>，用另一套方式记住历史。
</div>

""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Up to here, the LLM in your head still only reads text: a prompt is split into tokens, each looked up to an embedding, then run through dozens of transformer layers. Yet today's models routinely "talk about pictures" - you send an image and they describe it, answer questions, read tables. How do they fit "an image" into a model that only knows tokens? The answer is surprisingly plain: <strong>the LLM's input layer does not care whether the embedding vectors fed in came from text or from an image</strong>. Multimodality just turns "an image" into <strong>a run of embeddings</strong> too, splices them into the same sequence as the text tokens' embeddings, and sends the lot into <span class="mono">llama_decode</span>.
</p>
<p style="color:var(--muted);margin-top:.4rem">This lesson looks at how llama.cpp's <span class="mono">mtmd</span> (multimodal) subsystem does it. The core is one sentence: the model body needs not a single change; multimodality rides entirely on a "translation" pipeline bolted <strong>in front</strong> of it - a vision encoder (clip / ViT) compresses the image into visual features, and the projector (mmproj) <strong>projects those features into the LLM's embedding space</strong>, yielding N vectors that "look just like token embeddings", spliced in where the prompt's <span class="mono">&lt;image&gt;</span> marker sits. The LLM, handed this run of vectors, cannot tell which came from text and which from the image - it just computes on as usual. This gives pause at first: if the model clearly "understood" the image, how can it "not know it was an image"? But that is precisely the cleverest part of the design - "understanding the picture" is fully outsourced to the vision pipeline in front, and the LLM does only what it is best at: reasoning over a sequence of vectors.</p>
<p style="color:var(--muted)">Roadmap: first the whole <span class="mono">mtmd</span> pipeline (split into chunks -> encode -> get embeddings -> interleave with text and decode), with a trace of "one image entering the LLM"; then a dedicated look at why the projector <strong>bridge</strong> is indispensable; and finally two folds digging into clip's ViT internals and how image embeddings "take positions" in the sequence and enter the KV cache.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  Multimodality sounds mystical, but unpacked it is one sentence: <strong>translate "non-text" into embeddings the LLM can eat too</strong>. Recall L04/L20: the first step for text entering an LLM is looking up each token to an embedding vector; the transformer body, start to finish, processes <strong>vector sequences</strong> - it has no idea what a "token" looks like. That leaves an opening - as long as you can turn an image into a run of vectors of the <strong>same shape, same space</strong>, you can splice it straight into the sequence and the model takes it all. So llama.cpp does not touch the LLM body; it hangs an <span class="mono">mtmd</span> pipeline in front: clip(ViT) encodes the image into visual features, the projector projects features to the LLM's embedding dimension, producing N image embeddings spliced in at the <span class="mono">&lt;image&gt;</span> marker. Grasp this and you grasp the recipe behind almost every "multimodal LLM" today: nearly all are "an off-the-shelf LLM + a vision encoder + a projector bridge" bolted together - the body unchanged, it merely learned to "see". This "freeze the body, bolt on an adapter" idea you have actually met once before, in L24's LoRA - the only difference is that LoRA adapts "style/task" while here we adapt "input modality". The same plain engineering wisdom - do not touch the big, hard-to-train body, just add pluggable little pieces around it - bearing fruit twice in two completely different settings.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture an expert who <strong>only reads Chinese</strong> (the LLM), and you want them to understand a <strong>foreign-language chart</strong>. You would not force them to learn the language (changing the model is too costly); you hire a <strong>translator</strong>: the translator first <strong>understands</strong> the chart (clip/ViT encodes the image into visual features), then <strong>retells it as a Chinese passage</strong> (the projector projects into the LLM's embedding space), then <strong>splices that passage into your conversation</strong> with the expert. Reading this "Chinese", the expert naturally carries on - never knowing the passage was originally a picture. Two key roles here: <strong>the eyes that see the picture</strong> (clip) and <strong>the mouth that turns what is seen into the expert's mother tongue</strong> (projector). However good the eyes, if they cannot speak the expert's language the expert understands nothing - which is why the projector "translation bridge" is what makes or breaks multimodality.
</div>
<h2>The whole pipeline: how mtmd feeds an image into the LLM</h2>
<p>First walk the whole pipeline. The user gives an "interleaved image-text" input - some text carrying an <span class="mono">&lt;image&gt;</span> marker, plus the pixels of one (or a few) images. mtmd runs it in five steps: (1) <span class="mono">mtmd_init_from_file</span> loads the projector (that separate mmproj file); (2) <span class="mono">mtmd_tokenize</span> <strong>splits the input into a run of chunks</strong> - text passages are <span class="mono">TEXT</span> chunks, each image becomes an <span class="mono">IMAGE</span> chunk (audio is <span class="mono">AUDIO</span>); (3) each image chunk is run through <span class="mono">mtmd_encode_chunk</span> (internally clip + projector); (4) <span class="mono">mtmd_get_output_embd</span> pulls out the encoded visual embeddings; (5) the text chunks' tokens and the image chunks' embeddings are <strong>interleaved in original order</strong> and fed segment by segment into <span class="mono">llama_decode</span>. First, the splitting:</p>
<pre class="code"><span class="cm">// split "text + &lt;image&gt; marker + image bitmap" into ordered chunks (simplified from tools/mtmd/mtmd.h)</span>
mtmd_input_chunks * chunks = <span class="fn">mtmd_input_chunks_init</span>();
<span class="fn">mtmd_tokenize</span>(mtmd_ctx, chunks, &amp;text, bitmaps, n_bitmaps);
<span class="cm">// chunks now hold, in original order:</span>
<span class="cm">//   [TEXT "this image is"] [IMAGE one picture] [TEXT "what is in it?"]</span>
<span class="cm">// each chunk carries a type: TEXT / IMAGE / AUDIO</span></pre>
<p>Note the <span class="mono">&lt;image&gt;</span> in the prompt (the source's default marker is actually <span class="mono">&lt;__media__&gt;</span>): it is just a <strong>placeholder</strong> telling mtmd "where in the text this image should be inserted". The source's comment gives a plain example: a line like "here is an image: &lt;__media__&gt; ..." splits into three chunks - the text before the marker, the image itself, the text after - in exact original order. Once chunks are split, the real show is encoding the image chunk into embeddings and decoding it interleaved with text. llama.cpp packs this logic into a helper whose comment is practically the whole pipeline's pseudo-code:</p>
<pre class="code"><span class="cm">// per chunk: text decodes directly, image encodes first then decodes (simplified from mtmd-helper.cpp)</span>
<span class="kw">for</span> (each chunk : chunks) {
    <span class="kw">if</span> (<span class="fn">type</span>(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        <span class="fn">llama_decode</span>(lctx, <span class="fn">batch_of</span>(chunk.tokens));   <span class="cm">// text: old path, look up embeddings then compute</span>
    } <span class="kw">else</span> {                                          <span class="cm">// IMAGE / AUDIO chunk:</span>
        <span class="fn">mtmd_encode_chunk</span>(mtmd_ctx, chunk);          <span class="cm">// 1) internally runs clip(ViT) + projector</span>
        <span class="kw">float</span> * embd = <span class="fn">mtmd_get_output_embd</span>(mtmd_ctx); <span class="cm">// 2) pull out N visual embeddings</span>
        <span class="fn">llama_decode</span>(lctx, <span class="fn">batch_of_embd</span>(embd));     <span class="cm">// 3) feed the embeddings straight in</span>
    }
}</pre>
<p>See the trick? Text and image both land on the same <span class="mono">llama_decode</span> - the only difference is "whether you feed in tokens (and let the model look up embeddings) or already-computed embeddings (used directly)". <span class="mono">llama_batch</span> has long supported both inputs (remember from L18 that a batch can carry either <span class="mono">token</span> or <span class="mono">embd</span>?), so image embeddings slot <strong>seamlessly</strong> into the sequence, with not one line of the model body changed. The beauty of this design is <strong>reuse</strong>: mtmd does not open a second inference path for images, it "disguises" images as embeddings and reuses the entire text machinery of batching, KV cache, and sampling - so multimodality becomes a "preprocessing" problem, not a "rewrite the engine" problem. Freezing one image's whole journey into the LLM as a pipeline:</p>
<div class="cols">
  <div class="col"><h4>text chunk</h4><p>a sequence of token ids -&gt; <span class="mono">llama_decode</span> looks up the embedding table inside -&gt; compute. The old path of L04/L20.</p></div>
  <div class="col"><h4>image chunk</h4><p>pixels -&gt; clip + projector compute embeddings -&gt; feed embeddings straight into <span class="mono">llama_decode</span>. Embeddings ready, table lookup skipped.</p></div>
</div>
<div class="trace">
  <div class="tcap"><b>Tracing one image into the LLM</b>: pixels first pass clip(ViT) into visual features, the projector projects them into N image embeddings, which are then interleaved with text tokens in original order into one sequence, sent whole into llama_decode (illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>1 one image</h5>
      <div class="cellrow"><span class="vc">336x336 pixels</span></div>
      <div class="tlab">raw bitmap</div></div>
    <div class="op">split patches<br>clip(ViT)</div>
    <div class="stn"><h5>2 visual features</h5>
      <div class="cellrow"><span class="vc blue">576 feature vectors</span></div>
      <div class="tlab">ViT encode output</div></div>
    <div class="op">projector<br>(mmproj)</div>
    <div class="stn"><h5>3 image embedding</h5>
      <div class="cellrow"><span class="vc hot">N LLM embeddings</span></div>
      <div class="tlab">projected to LLM dim</div></div>
    <div class="op">insert at<br>&lt;image&gt;</div>
    <div class="stn"><h5>4 interleaved seq</h5>
      <div class="cellrow"><span class="vc">text emb + image emb + text emb</span></div>
      <div class="tlab">one unified sequence</div></div>
    <div class="op">decode<br>together</div>
    <div class="stn"><h5>5 llama_decode</h5>
      <div class="cellrow"><span class="vc blue">LLM runs as usual</span></div>
      <div class="tlab">cannot tell image from text</div></div>
  </div>
</div>
<p>Worth noting: this "split chunks -&gt; encode -&gt; interleave" mechanism treats <strong>audio</strong> identically - cut speech into frames, run an audio encoder (like Whisper's front end), then a projector to embeddings, taking exactly the same path as images. That is why <span class="mono">mtmd</span>'s chunk types put <span class="mono">AUDIO</span> right beside <span class="mono">IMAGE</span>. In other words, mtmd's design goal from the start was not "support images" but "support any modality that can be encoded into embeddings". Once you understand the image path, audio - and more modalities to come - are cast from the same mold.</p>
<h2>The projector: translating "what is seen" into the LLM's mother tongue</h2>
<p>That <span class="mono">mtmd_encode_chunk</span> step splits internally into two halves: first clip(ViT) "sees" the image as visual features, then the projector "translates" those features into embeddings the LLM can read. Why is this second step mandatory? Because clip's visual features and the LLM's token embeddings <strong>are simply not in the same space</strong> - the dimensions may differ (clip might output 1024-d, the LLM wants 4096-d), and the value distributions and semantics are two different systems entirely. Feeding clip's output straight into the LLM is like handing an untranslated foreign passage to an expert who only reads Chinese - blank stares. The projector (that separate mmproj file) is the bridge: a small network that <strong>projects visual features into the LLM's embedding dimension and space</strong>, making them "look and behave just like a token embedding". By analogy, clip's output is a kind of "visual shorthand" whose numbers mean things arranged for a vision task; the LLM's embedding space grew out of a language task, so vectors of the same length live in completely different "coordinate systems". The projector does exactly that change of coordinates: re-expressing the visual shorthand into the language coordinate system, so "there is a cat in the image" lands near the patch of vector space the LLM has always used for "cat".</p>
<p>Two key "alignments" are guarded by two functions: <span class="mono">clip_n_mmproj_embd(ctx)</span> returns the projector's output dimension - it <strong>must equal</strong> the LLM's embedding dimension, or the vectors will not fit into the sequence; <span class="mono">clip_n_output_tokens(ctx, img)</span> returns <strong>how many embedding tokens this one image occupies</strong> (the N in the earlier trace). That N is not arbitrary: bigger image, more patches, larger N; some projectors (resampler types) even <strong>actively compress</strong> N, aggregating hundreds of patch features into a few dozen embeddings, saving KV cache and compute. This N directly sets an image's "cost": N embeddings take N sequence positions and write N entries of KV - so for the same image, whether the projector squeezes it to 64 embeddings or lays out 576 changes VRAM and speed by an order of magnitude. This is one of the core tradeoffs of high-resolution multimodal models: the finer it sees (more patches, larger N) the more accurate, but the longer the sequence, the slower and more VRAM-hungry.</p>
<pre class="code"><span class="cm">// inside the projector: clip encodes first, output dim set by mmproj (simplified from tools/mtmd/clip.h)</span>
<span class="kw">int</span> n_embd   = <span class="fn">clip_n_mmproj_embd</span>(clip_ctx);       <span class="cm">// projector output dim == LLM embedding dim</span>
<span class="kw">int</span> n_tokens = <span class="fn">clip_n_output_tokens</span>(clip_ctx, img); <span class="cm">// how many embedding tokens this image takes</span>
std::vector&lt;float&gt; out_vec;
<span class="fn">clip_image_encode</span>(clip_ctx, n_threads, img, out_vec); <span class="cm">// run ViT + projector, output n_tokens x n_embd</span>
<span class="cm">// out_vec is now n_tokens visual embeddings, each n_embd-dim,</span>
<span class="cm">// exactly the shape of n_tokens token embeddings -&gt; straight into the sequence</span></pre>
<p>Common projectors come in three tiers of complexity: a <strong>linear layer</strong> (one matmul, simplest, early LLaVA), a <strong>two-layer MLP</strong> (one more nonlinearity, better alignment, common today), and a <strong>resampler / cross-attention</strong> (a set of learned queries "resamples" the variable-length patch features into a fixed number of embeddings, compressing N and handling arbitrary resolution, used by Qwen-VL etc). Which tier is a <strong>tradeoff of accuracy and cost</strong>: linear is cheapest but least expressive, the resampler is most flexible but carries its own pile of parameters and compute. Whichever tier, its job is the same: <strong>align visual features into the LLM's embedding space</strong>. This is also why the mmproj is a <strong>separate file</strong>, loaded separately - it is a bridge <strong>specifically trained</strong> for one "this vision encoder + this LLM" pairing; swap the LLM or the clip and the bridge must be retrained. Grasp this and you see why, when downloading a multimodal model, besides the big main GGUF you also need a tiny mmproj file: without that bridge, the model keeps only its "read text" skill, and "see images" is off the table. A practical tip: in a multimodal model's GGUF repo on HuggingFace, the small file with mmproj in its name is exactly this, usually on the order of a few hundred MB - do not forget to grab it.</p>
<table class="t">
  <tr><th>projector type</th><th>structure</th><th>traits</th><th>example</th></tr>
  <tr><td>linear</td><td>one matmul</td><td>cheapest, least expressive, fixed N</td><td>early LLaVA</td></tr>
  <tr><td>two-layer MLP</td><td>linear + nonlinearity + linear</td><td>better alignment, common today</td><td>LLaVA-1.5+</td></tr>
  <tr><td>resampler</td><td>learned queries + cross-attention</td><td>compresses N, any resolution</td><td>Qwen-VL</td></tr>
</table>
<h2>Deeper: clip's eyes, and an image's "position"</h2>
<p>Two folds for two details you cannot avoid when really deploying multimodality.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> What is clip(ViT) actually doing inside? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>This lesson treats clip as a black box - give it an image, it spits out a run of visual features. Lift the lid and clip's vision encoder is just a <strong>standard Vision Transformer (ViT)</strong>, almost the same recipe as the text transformer from earlier parts, only with "token" swapped for "image patch": (1) cut the image into fixed-size blocks (patches, say 14x14 pixels each), flatten each and linearly project it to a vector - the image's version of an "embedding"; (2) add positional encoding (telling the model where each block sits in the original image); (3) run through several layers of self-attention + FFN, letting each patch "see" the whole image and aggregate semantics. Each patch ends up with one output vector, and together they are that run of visual features. So a 336x336 image at 14 pixels a block is 24x24 = 576 patches -&gt; 576 features (that is where the trace's number came from). The cost comes from here too: more patches, larger self-attention cost (O(patch^2)), which is why high-resolution images are expensive. To stay sharp without letting the patch count explode, many implementations <strong>cut a big image into tiles</strong>, encode each, and concatenate the tiles' embeddings - which is also why some multimodal models emit hundreds or thousands of embeddings for one large image. The real implementation lives in <span class="mono">tools/mtmd/clip.cpp</span>, where ggml builds this ViT out - if you already followed L16's text build-graph, clip.cpp will not feel foreign, just a different kind of token. This lesson does not unroll it line by line because, for the through-line of "how multimodality plugs into the LLM", it is not the point: the point is that the features it emits need the projector bridge to get into the LLM. Incidentally, because clip is internally a transformer too, it can use the same ggml ops, be quantized, and run on the same backends - which is why llama.cpp can fit the vision encoder and the LLM into one inference framework, without dragging in a separate PyTorch.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> How do image embeddings "take position" in the sequence and enter the KV cache? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Once image embeddings are spliced into the sequence, to the LLM they are N real positions in it - assigned positions like text tokens, written into the KV cache like text tokens (echoing L19). This brings two things to handle. <strong>First, positional encoding</strong>: plain text is one-dimensional position (the 0th, 1st, 2nd token), but an image is two-dimensional (which row and column a patch is in), and flattening to 1D loses spatial structure. So many models use <strong>M-RoPE (multi-dimensional RoPE)</strong>, giving image tokens a multi-dimensional position that can express "row, column" - in llama.cpp <span class="mono">mtmd_decode_use_mrope</span> asks "does this model need M-RoPE", and <span class="mono">mtmd_helper_get_n_pos</span> specifically counts how many "positions" a run of chunks occupies (its comment notes: normally n_pos == n_tokens, but under M-RoPE they differ). Intuitively, M-RoPE gives an image token's position not a single point on one number line but more like a coordinate cell on a chessboard - so the model knows the top-left patch and the bottom-right patch are far apart in space. <strong>Second, the attention mask</strong>: text is causal (can only see what came before), but the patches within one image usually need to <strong>all see each other</strong> (bidirectional), so some models (like Gemma 3) temporarily switch to non-causal attention while decoding the image segment - exactly what <span class="mono">mtmd_decode_use_non_causal</span> governs. Grasp these two and you see that an image entering the LLM is not just "splice in N vectors": it must also coexist with text on the two fundamental transformer mechanisms of <strong>position</strong> and <strong>attention</strong>. Helpfully llama.cpp handles all this for you; you just need to know: once image embeddings enter the sequence, they take KV cache like text and join the attention of every later token - which is why more images make the KV cache grow faster. This matters acutely in engineering: one high-resolution image can take hundreds or thousands of positions, and a few images in, the KV cache rivals a long passage of text. So in multimodal serving the "image budget" often has to be counted together with the "context budget" - which brings you right back to L19's old problem: the longer the sequence, the bigger the KV cache, the fewer requests you can run concurrently. Multimodality does not escape this iron law, it only makes "what can be in the sequence" richer. So next time you see a "32K-context multimodal model", be clear in your head: that 32K is a budget <strong>shared</strong> by images and text, and one high-res image can eat a big chunk of it.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Core: the LLM body only knows embedding vectors, not caring if they came from text or image; multimodality = turn "an image" into N embeddings too, spliced into the same sequence.</li>
    <li>mtmd pipeline: <span class="mono">mtmd_tokenize</span> splits chunks (TEXT/IMAGE/AUDIO) -&gt; run <span class="mono">mtmd_encode_chunk</span>(clip+projector) on image chunks -&gt; <span class="mono">mtmd_get_output_embd</span> gets embeddings -&gt; interleave with text into <span class="mono">llama_decode</span>.</li>
    <li>projector(mmproj) is the bridge: projects clip's visual features into the LLM's embedding dim/space; <span class="mono">clip_n_mmproj_embd</span>=LLM embd dim, <span class="mono">clip_n_output_tokens</span>=tokens one image takes.</li>
    <li>mmproj is a separate file, trained for one "this clip + this LLM" pairing; without it the model only reads text, cannot see images.</li>
    <li>Image embeddings take position and enter the KV cache as usual (L19); M-RoPE handles 2D position, non-causal mask handles in-image bidirectionality (<span class="mono">mtmd_decode_use_mrope</span> / <span class="mono">_non_causal</span>).</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The thing to take from this lesson is not clip's or the projector's details, but a fact plain to the point of being counterintuitive: <strong>the LLM has never "seen" an image</strong>. In its eyes there are only sequences of embedding vectors - all of multimodality's magic is a translation bolted <strong>in front</strong>, turning images into the kind of vector it was already used to. This is an enormously vital design philosophy: <strong>do not touch the core, only adapt at the boundary</strong>. The same LLM, given a vision bridge can see, given an audio bridge can hear, given some other bridge could take some other modality - the core staying that one embedding-only transformer. Look back over the whole chain you learned: L04's tokens, L20's embeddings, L18's batch (carrying token or embd), L19's KV cache - multimodality overturns none of them, it stands on their shoulders and pushes "what the input can be" a big step outward. By this point in Part 7 we have seen three takes on "working outside the standard transformer": speculative decoding changed "how to emit tokens faster", MoE changed "how to trade sparsity for capacity", multimodality changed "what the input can be". The last lesson touches something more fundamental - state-space models (Mamba / RWKV) simply <strong>replace</strong> the attention the transformer lives on, remembering history a different way.
</div>

""",
}

LESSON_37 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
transformer 的注意力有个绕不开的代价：要记住整段历史，它得把每个 token 的 Key/Value 都存进 KV cache，于是显存随序列<strong>线性</strong>地涨（L19 讲过），每生成一个新 token 还得回头看全部历史（算力 O(n^2)）。状态空间模型（State-Space Model，SSM；代表是 Mamba 和 RWKV）走了另一条完全不同的路：它不存全部历史，而是把历史<strong>压进一个固定大小的"状态" h_t</strong>，每来一个新 token，就用上一个状态 h_(t-1) 和当前输入算出新状态——显存 O(1)、不随序列长度涨。
</p>
<p style="color:var(--muted);margin-top:.4rem">这是注意力之外的另一条技术路线。它不是要彻底取代 transformer，而是换一种"怎么记住历史"的办法：注意力把历史摊开、一条条存着随时回查；SSM 把历史卷进一个滚动更新的状态里，只留摘要。两条路各有所长，所以这一课的落点，不只是看懂 Mamba/RWKV 怎么算，更是看懂"固定状态 vs 全量 KV"这个贯穿始终的取舍。把这层取舍想透，你以后再看到任何"线性注意力""高效 transformer 替代品"，都能一眼问到点子上：它拿什么压缩了历史？又因此丢了什么？</p>
<p style="color:var(--muted)">路线图：先把"递推 vs 注意力"摆在一起对照（一个把历史摊开存、一个把历史卷进一个状态），配一张图追踪状态怎么沿时间扫描；再看 ggml 用哪两个算子（<span class="mono">ggml_ssm_conv</span> + <span class="mono">ggml_ssm_scan</span>）实现它、状态怎么从 recurrent cache 读出写回；接着把"选择性扫描"这个核心概念讲清楚（A/B/C/Δ 各是什么、"选择性"到底选什么）；最后两个折叠聊 SSM 的代价和当下流行的"混合架构"。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  SSM 的核心，是把 transformer "记住一切" 的奢侈，换成 "记住一个摘要" 的精打细算。注意力的记忆方式是"把每个 token 的 KV 都留着、要用时回头查"——准，但越攒越多。SSM 换了个思路：维护一个<strong>固定大小的状态向量 h</strong>，把"到目前为止读到的东西"不断压缩进这个 h 里，每步只更新它、不另存历史。于是不管序列多长，状态都那么大——显存 O(1)、每步算力 O(1)，整段序列就是 O(n) 而不是注意力的 O(n^2)。代价也藏在这里：一个固定大小的状态装不下无限的历史，它是一种<strong>有损压缩</strong>，长程的精确检索（"第 3000 个 token 到底是什么"）不如能回头逐个查 KV 的注意力。理解这个取舍，你就理解了为什么 SSM 在超长序列、流式场景里特别香，却又常常要和注意力混着用。这一点本课最后会回到——纯粹路线很少是终点，工程上往往是"取两者之长"。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  注意力像一个<strong>带着全套逐字笔记的人</strong>：每听一句就原样记一条，要回答问题时翻笔记、逐条查——准确，但笔记越积越厚（KV cache 越来越大）。SSM 则像一个<strong>只在脑子里滚动更新一句话摘要的人</strong>：每听一句，就把"目前为止的大意"在心里更新一下，旧的原话不留。无论听多久，他脑子里要记的东西都是<strong>同样一句话那么多</strong>（状态固定大小）。代价也很真实：你问他"刚才第几句话的原话是什么"，带笔记的人能精确翻出来，只记摘要的人多半已经忘了细节——这正是 SSM "有损压缩" 的生活版。两种人没有谁更聪明，只是适合的场合不同：开会做纪要，摘要派更省力；法庭上要逐字举证，笔记派才靠得住。
</div>
<h2>递推 vs 注意力：两种记住历史的方式</h2>
<p>先把两条路并排放在一起。注意力（你在 L19 学的）记历史的方式是"全都留着"：每个 token 算出的 Key/Value 都进 KV cache，第 t 步要看前面所有 t 个位置——所以显存随序列<strong>线性</strong>增长，单步算力随历史<strong>线性</strong>增长（整段就是 O(n^2)）。SSM 反过来，把历史<strong>卷进一个固定大小的状态 h</strong>：第 t 步只拿上一步的状态 h_(t-1) 和当前输入 x_t，算出新状态 h_t——和 t 之前有多长完全无关。状态多大，是模型定死的超参（<span class="mono">ssm_d_state</span>），跟序列长度没关系。这一句听着平平无奇，却是 SSM 全部优势的源头：序列从 1k 拉到 100k，注意力的开销翻了一百倍，SSM 每步的开销却一个字节都没多。</p>
<div class="cols">
  <div class="col"><h4>注意力（KV cache）</h4><p>每个 token 的 KV 都存下来，一路变长。第 t 步回看全部 t 个历史：显存 O(n)、算力 O(n^2)。长程精确，但越来越重。</p></div>
  <div class="col"><h4>SSM（递推状态）</h4><p>只维护一个固定大小的状态 h，原地更新。第 t 步只看 h_(t-1) 和 x_t：显存 O(1)、算力 O(n)。轻、稳，但状态是有损摘要。</p></div>
</div>
<p>这个差别最直观的画面，就是"状态沿时间扫描"：输入 x_0, x_1, ... 一个个进来，状态 h 在原地一步步更新，每一步只依赖上一个状态和当前输入，旁边再顺手吐出一个输出 y。看下面这张图——注意 h 那一行<strong>始终是同样大的一个格子</strong>，不像 KV cache 那样越拖越长：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次状态扫描</b>：输入 x_0..x_4 依次进来，固定大小的状态 h 原地递推 h_0 -&gt; h_1 -&gt; ... -&gt; h_4，每步只依赖上一个 h 和当前 x，同时产出 y。状态大小自始至终不变（示意）。</div>
<svg viewBox="0 0 680 268" width="100%" role="img" aria-label="状态空间模型的状态扫描：输入 x0..x4 依次进来，一个固定大小的状态 h 原地递推 h0 -> h1 -> ... -> h4，每步只依赖上一个 h 和当前 x，并产出 y">
<g font-family="ui-monospace,monospace">
<text x="40" y="54" text-anchor="middle" fill="#5b6470" font-size="10">输入 x</text>
<text x="40" y="142" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="10">h</text>
<text x="40" y="224" text-anchor="middle" fill="#5b6470" font-size="10">输出 y</text>
<rect x="68" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="96" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x0</text>
<line x1="96" y1="64" x2="96" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 96 115 L 92 107 L 100 107 z" fill="#9aa6b2"/>
<rect x="68" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="96" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h0</text>
<line x1="96" y1="160" x2="96" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 96 205 L 92 197 L 100 197 z" fill="#9aa6b2"/>
<rect x="68" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="96" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y0</text>
<rect x="184" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="212" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x1</text>
<line x1="212" y1="64" x2="212" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 212 115 L 208 107 L 216 107 z" fill="#9aa6b2"/>
<rect x="184" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="212" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h1</text>
<line x1="126" y1="138" x2="178" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 183 138 L 175 134 L 175 142 z" fill="#7c3aed"/>
<line x1="212" y1="160" x2="212" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 212 205 L 208 197 L 216 197 z" fill="#9aa6b2"/>
<rect x="184" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="212" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y1</text>
<rect x="300" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="328" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x2</text>
<line x1="328" y1="64" x2="328" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 328 115 L 324 107 L 332 107 z" fill="#9aa6b2"/>
<rect x="300" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="328" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h2</text>
<line x1="242" y1="138" x2="294" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 299 138 L 291 134 L 291 142 z" fill="#7c3aed"/>
<line x1="328" y1="160" x2="328" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 328 205 L 324 197 L 332 197 z" fill="#9aa6b2"/>
<rect x="300" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="328" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y2</text>
<rect x="416" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="444" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x3</text>
<line x1="444" y1="64" x2="444" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 444 115 L 440 107 L 448 107 z" fill="#9aa6b2"/>
<rect x="416" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="444" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h3</text>
<line x1="358" y1="138" x2="410" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 415 138 L 407 134 L 407 142 z" fill="#7c3aed"/>
<line x1="444" y1="160" x2="444" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 444 205 L 440 197 L 448 197 z" fill="#9aa6b2"/>
<rect x="416" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="444" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y3</text>
<rect x="532" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="560" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x4</text>
<line x1="560" y1="64" x2="560" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 560 115 L 556 107 L 564 107 z" fill="#9aa6b2"/>
<rect x="532" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="560" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h4</text>
<line x1="474" y1="138" x2="526" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 531 138 L 523 134 L 523 142 z" fill="#7c3aed"/>
<line x1="560" y1="160" x2="560" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 560 205 L 556 197 L 564 197 z" fill="#9aa6b2"/>
<rect x="532" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="560" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y4</text>
<text x="154" y="129" text-anchor="middle" fill="#7c3aed" font-size="10">h=f(h,x)</text>
<text x="340" y="256" text-anchor="middle" fill="#5b6470" font-size="11">状态大小始终不变 -> 显存 O(1)，不随序列变长</text>
</g></svg>
</div>
<h2>ggml 怎么实现：两个算子 + 递推状态</h2>
<p>SSM 在 ggml 里主要落到两个算子上，加一套"递推状态"的读写。先是 <span class="mono">ggml_ssm_conv</span>——一个<strong>因果 1D 卷积</strong>：在做扫描之前，先让每个位置和它前面几个位置做一次局部混合（卷积宽度就是 <span class="mono">ssm_d_conv</span>，通常很小，比如 4）。它的作用类似给输入加一点"短期记忆"，让相邻 token 的信息先交融一下。然后是真正的核心 <span class="mono">ggml_ssm_scan</span>——<strong>选择性扫描</strong>，它吃一大把张量：状态 <span class="mono">s</span>、输入 <span class="mono">x</span>、时间步 <span class="mono">dt</span>(Δ)、状态矩阵 <span class="mono">A</span>/<span class="mono">B</span>/<span class="mono">C</span>、还有序列索引 <span class="mono">ids</span>，一趟把整段序列的递推都算出来。来看一个 Mamba 层里这两个算子的真实调用（出自 <span class="mono">src/models/mamba-base.cpp</span>）：</p>
<pre class="code"><span class="cm">// Mamba 层核心 (简化自 src/models/mamba-base.cpp)</span>
x = <span class="fn">ggml_ssm_conv</span>(ctx0, conv_x, layer.ssm_conv1d); <span class="cm">// 1) 因果卷积: 局部短期混合</span>
x = <span class="fn">ggml_silu</span>(ctx0, x);                          <span class="cm">//    激活</span>
<span class="cm">// 2) 从 x 投影出 dt, B, C —— 注意这几个量都依赖输入 x (这就是"选择性")</span>
dt = <span class="fn">build_lora_mm</span>(layer.ssm_dt, dt);            <span class="cm">//    每步的步长 Δ</span>
A  = layer.ssm_a;                                <span class="cm">//    状态转移矩阵 (学到的参数, 不随输入变)</span>
<span class="cm">// 3) 选择性扫描: 按 A/B/C/dt 把状态沿时间递推</span>
y = <span class="fn">ggml_ssm_scan</span>(ctx0, ssm, x, dt, A, B, C, ids);</pre>
<p>注意第 2 步那个关键细节：<span class="mono">dt</span>、<span class="mono">B</span>、<span class="mono">C</span> 都是从当前输入 <span class="mono">x</span> 算出来的（输入相关），而 <span class="mono">A</span> 是学到的固定参数。这正是 Mamba 比老式 SSM 强的地方——下一节细讲。这里还有一个容易被忽略的问题：状态 h 存在哪？它<strong>不进 KV cache</strong>，而是进一套专门的"递推状态缓存"（<span class="mono">llama-memory-recurrent</span>），每个序列只占<strong>固定大小</strong>的一块。<span class="mono">build_rs</span>（rs = recurrent state）就负责把上一步的状态从这块缓存里读出来、喂给 scan、再把更新后的新状态写回去：</p>
<pre class="code"><span class="cm">// 递推状态的读出 - 更新 - 写回 (示意; 见 src/llama-graph.cpp build_rs)</span>
<span class="cm">// 1) 从 recurrent cache 读出这些序列上一步的状态</span>
states = <span class="fn">get_state_rows</span>(state_cache, ids);
<span class="cm">// 2) 用 scan 在这些状态上往前推一段</span>
y = <span class="fn">ggml_ssm_scan</span>(ctx0, states, x, dt, A, B, C, ids);
<span class="cm">// 3) 把最后的新状态写回 cache, 留给下一步用</span>
<span class="fn">ggml_cpy</span>(ctx0, last_state, state_cache_view);</pre>
<p>对照 L17/L19 你会发现一个漂亮的呼应：transformer 用 KV cache 存"历史的全部 KV"，SSM 用 recurrent cache 存"历史压成的那个状态"。两者都是"把上一步的东西留到下一步"的缓存，但一个随序列变大、一个永远固定大小。这也是为什么 llama.cpp 要专门给 SSM 做一套 <span class="mono">llama-memory-recurrent</span>，而不是塞进原来的 KV cache——它们的"记忆形状"根本不同。也正因如此，一个纯 SSM 模型跑起来时，你在显存里几乎看不到"KV cache 随对话变长"这件事——取而代之的，是每个序列一小块大小恒定的状态。</p>
<h2>选择性扫描：A / B / C / Δ 到底在干什么</h2>
<p>"扫描"（scan）就是上面那个递推：从头到尾走一遍序列，状态一步步往前推。难点在"选择性"（selective）这三个字。先看最朴素的递推，去掉所有花哨，它就两行：</p>
<pre class="code"><span class="cm"># 选择性扫描的核心递推 (概念伪代码, 非真实 kernel)</span>
<span class="kw">for</span> t <span class="kw">in</span> 0..n:
    h = A * h + B * x[t]     <span class="cm"># 用上一个状态 h 和当前输入 x[t] 算新状态</span>
    y[t] = C * h             <span class="cm"># 从新状态读出这一步的输出</span></pre>
<p>四个量各司其职：<span class="mono">A</span> 管"上一个状态要保留多少"（衰减 / 记忆），<span class="mono">B</span> 管"当前输入怎么写进状态"，<span class="mono">C</span> 管"从状态里读出什么当输出"，<span class="mono">Δ</span>(dt) 是"步长"，控制这一步更新得多猛（可以理解成离散化的时间间隔：Δ 大≈多看一眼当前输入、Δ 小≈更依赖旧状态）。如果 A/B/C/Δ 都是<strong>固定不变</strong>的常数，那就是经典的线性 SSM——快，但呆板：它对每个 token 一视同仁，没法"看人下菜碟"。这也是早期 SSM 一直打不过 transformer 的根本原因——它记东西一视同仁，可语言里偏偏有的词关键、有的词是废话。</p>
<p>Mamba 的关键创新就一句话：<strong>让 B、C、Δ 随输入变化</strong>（前一节代码里 dt/B/C 都是从 x 投影出来的，就是这个意思）。这叫"选择性"——模型可以根据当前读到的内容，<strong>动态决定</strong>"这个 token 重要、多写进状态一点"或"这个 token 没用、让状态忽略它"。打个比方：固定 SSM 像一台匀速传送带，什么都按同样节奏处理；选择性 SSM 像一个会<strong>自己调速</strong>的阅读者，遇到关键句就放慢、细记，遇到废话就快进、略过。正是这点"输入相关的门控"，让 Mamba 在语言这种"信息密度不均"的任务上，第一次追平了 transformer。<span class="mono">A</span> 之所以仍保持固定（是学到的参数、不随输入变），是因为它管的是状态自身的稳定衰减，让它随输入乱跳反而会让递推不稳定——这是一个精心保留的"锚"。换句话说，Mamba 把 SSM 里"该死板的"（A，保稳定）和"该灵活的"（B/C/Δ，随输入）拆开对待，这种"有所变、有所不变"的拿捏，正是它比前代 SSM 高明的地方。</p>
<p>真实的 kernel 当然比这两行复杂得多：要做离散化（把连续的 A/Δ 变成每步的乘子）、要在 GPU 上做并行前缀和（parallel scan，把看似串行的递推拆成可并行的部分，这就是 Mamba 论文里那个"associative scan"），还要处理多头、分组。但抓住"<span class="mono">h = A*h + B*x; y = C*h</span>，而且 B/C/Δ 随输入走"这条主线，你就抓住了 SSM 的灵魂——剩下的都是把它高效地搬上硬件的工程。</p>
<table class="t">
  <tr><th>维度</th><th>注意力 (transformer)</th><th>SSM (Mamba/RWKV)</th></tr>
  <tr><td>记忆方式</td><td>存全部 token 的 KV</td><td>一个固定大小的状态 h</td></tr>
  <tr><td>显存随序列</td><td>线性增长 O(n)</td><td>不变 O(1)</td></tr>
  <tr><td>整段算力</td><td>O(n^2)</td><td>O(n)</td></tr>
  <tr><td>最擅长</td><td>长程精确检索 / copy</td><td>超长序列 / 流式 / 省显存</td></tr>
  <tr><td>llama.cpp 缓存</td><td>KV cache (L19)</td><td>llama-memory-recurrent</td></tr>
</table>
<h2>深入：长序列的甜与苦、混合架构</h2>
<p>最后两个折叠，回答 SSM 落地时最值得想清楚的两个问题。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> SSM 为什么适合长序列？代价又是什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>先说甜。注意力的 KV cache 随序列线性变大，第 t 步还要和前面全部 t 个位置算注意力——序列拉到几十万 token，显存和算力都会爆。SSM 的状态固定大小、每步只看上一个状态：显存 O(1)、整段算力 O(n)，序列再长，单步开销纹丝不动。这让它在<strong>超长上下文、流式输入、端侧低显存</strong>这些场景里格外香——你甚至可以近乎无限地往里喂 token，显存都不涨。再说苦。一个固定大小的状态，本质是把任意长的历史<strong>有损压缩</strong>进一个小向量。该记的太多、状态装不下时，它只能取舍着忘。于是 SSM 在"精确检索"类任务上天然吃亏：比如"把第 3000 个 token 原样复制出来"（copy 任务）、或者需要回头精确比对很久以前的某个细节——注意力能回头逐个查 KV，SSM 却只有一个被反复覆写的摘要。这不是实现不好，是"固定状态"这个选择的根本代价：你用"不随长度涨的开销"换走了"对任意历史的精确随机访问"。所以一个实用的直觉是：任务越偏"顺序处理、不怎么回头"（长文摘要、流式转写、音频），SSM 越占便宜；越偏"精确回查、大海捞针"（按 ID 检索、长文档里找某句原话），注意力越稳。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 既然各有长短，为什么不把两者混着用？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>这正是当下最流行的做法——<strong>混合架构</strong>。既然注意力擅长精确检索、SSM 擅长高效处理长序列，那就让一部分层用 SSM、一部分层用注意力，各取所长。很多新模型（如 Jamba，以及各种 Mamba-Transformer 混合）就这么搭：大多数层用 SSM 扛长度、省显存，少数几层插注意力补上精确检索的能力。llama.cpp 为此专门做了 <span class="mono">llama-memory-hybrid</span>——一个能<strong>同时</strong>管理两种记忆的内存子系统：注意力层那部分走 KV cache、SSM 层那部分走 recurrent state cache，两套缓存在同一个模型里并存。这也呼应了前面几课反复出现的态度：架构不是非此即彼的信仰之争，而是工程上的权衡组合。你前六部分搭好的那套地基（计算图 L09/L10、内存管理 L17、KV cache L19），到了混合架构这里依然管用——只是现在同一个模型里，有的层记"全部 KV"、有的层记"一个状态"，调度器要把两者都照顾好。这恰好说明了一件事：你学透的那套 transformer 缓存与调度，并不会因为 SSM 出现而作废——反而正是看懂混合架构的前提。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>核心取舍：注意力存全部 KV（显存 O(n)、算力 O(n^2)，长程精确）；SSM 用固定大小的递推状态 h（显存 O(1)、算力 O(n)，有损摘要）。</li>
    <li>递推：<span class="mono">h_t = f(A,B,C,Δ; h_(t-1), x_t)</span>，<span class="mono">y_t = C·h_t</span>——每步只看上一个状态和当前输入，状态大小由 <span class="mono">ssm_d_state</span> 定死。</li>
    <li>ggml 两算子：<span class="mono">ggml_ssm_conv</span>（因果卷积、局部混合）+ <span class="mono">ggml_ssm_scan(s,x,dt,A,B,C,ids)</span>（选择性扫描）；状态经 <span class="mono">build_rs</span> 从 <span class="mono">llama-memory-recurrent</span> 读出写回。</li>
    <li>"选择性"= 让 <span class="mono">B/C/Δ</span> 随输入变化（输入相关门控），模型学会动态"记住 / 忽略"——这是 Mamba 追平 transformer 的关键。</li>
    <li>代价：固定状态是有损压缩，长程精确检索（copy 类）不如注意力；故常用<strong>混合架构</strong>（<span class="mono">llama-memory-hybrid</span>）取两者之长。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  状态空间模型最该带走的，不是 Mamba 的某个公式，而是它逼你重新审视一个你早已习以为常的假设：<strong>"模型必须记住全部历史"其实是一个选择，不是一条定律</strong>。注意力选择了"全记下来、随时回查"，于是有了 O(n^2) 的代价；SSM 选择了"只记一个滚动摘要"，于是换来 O(n) 的轻盈，也接受了"记不清细节"的损失。没有哪个绝对更好——它们只是在"精确 vs 高效"这根轴上站了不同的位置。这正是第七部分四课共同的母题：投机解码重新看"出 token 的串行性"、MoE 重新看"算力和参数的绑定"、多模态重新看"输入的形态"、SSM 重新看"历史的存法"。每一课都在挑战一个"本来以为天经地义"的默认，再给出另一种可能。而它们全都站在你前六部分搭好的地基上——这，正是从"会用 llama.cpp"走向"看懂 llama.cpp、甚至改进它"的那一步。
</div>

<p>第七部分到此收束。从投机解码、MoE、多模态到状态空间模型，我们把四个"标准 transformer 之外"的进阶机制逐一拆开看了一遍——它们要么换个角度榨瓶颈、要么换种结构换效率、要么干脆换掉一根支柱。下一站，第八部分把视线从"模型怎么算"转向"工程怎么落地"：怎么把 HuggingFace 的模型转成 GGUF、怎么编译调试和测试、怎么真正参与到 llama.cpp 的贡献里来。学完前七部分，你已经从"模型怎么跑"一路看到了"前沿架构怎么变"；接下来，该把这份理解落到手上了。</p>

""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
The attention in a transformer has an unavoidable cost: to remember the whole history, it must store every token's Key/Value in the KV cache, so memory grows <strong>linearly</strong> with the sequence (as L19 showed), and emitting each new token means looking back over the whole history (compute O(n^2)). State-space models (SSMs; the well-known ones are Mamba and RWKV) take a completely different road: they do not store the whole history but <strong>compress it into a fixed-size "state" h_t</strong>; each new token computes a new state from the previous state h_(t-1) and the current input - memory O(1), not growing with sequence length.
</p>
<p style="color:var(--muted);margin-top:.4rem">This is a technical road alongside attention, not a replacement for the transformer - just a different way to "remember the history": attention spreads the history out and stores it entry by entry to look up at will; an SSM rolls the history into a continuously updated state and keeps only the gist. Each road has its strengths, so this lesson lands not only on how Mamba/RWKV compute, but on the through-line tradeoff of "fixed state vs full KV". Think this tradeoff through and, whenever you later meet any "linear attention" or "efficient transformer alternative", you can cut straight to the point: what did it use to compress the history, and what did it lose for it?</p>
<p style="color:var(--muted)">Roadmap: first put "recurrence vs attention" side by side (one spreads the history out and stores it, one rolls the history into a single state), with a trace of how the state scans along time; then the two ggml ops that implement it (<span class="mono">ggml_ssm_conv</span> + <span class="mono">ggml_ssm_scan</span>) and how the state is read from and written back to a recurrent cache; then the core concept of "selective scan" (what A/B/C/Delta are, what "selective" actually selects); and finally two folds on SSM's cost and today's popular "hybrid architectures".</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  At its core, an SSM trades the transformer's luxury of "remember everything" for the thrift of "remember a summary". Attention remembers by "keeping every token's KV and looking back when needed" - accurate, but ever-growing. An SSM takes a different tack: maintain a <strong>fixed-size state vector h</strong>, continuously compressing "what has been read so far" into this h, updating it each step and storing no separate history. So however long the sequence, the state stays that size - memory O(1), per-step compute O(1), the whole sequence O(n) instead of attention's O(n^2). The cost hides right here: a fixed-size state cannot hold infinite history, it is a <strong>lossy compression</strong>, and long-range exact retrieval ("what exactly was the 3000th token") is worse than attention, which can look back and check each KV. Grasp this tradeoff and you see why SSMs are especially sweet for very long sequences and streaming, yet so often get mixed with attention. We will return to this at the end - a pure route is rarely the destination; engineering usually "takes the best of both".
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Attention is like a person <strong>carrying a full verbatim transcript</strong>: each sentence heard is jotted down as-is, and to answer a question they flip through and look it up - accurate, but the notes pile up ever thicker (the KV cache keeps growing). An SSM is like a person <strong>keeping only a rolling one-sentence summary in their head</strong>: each sentence heard, they update "the gist so far" a little, keeping none of the old verbatim. However long they listen, what they hold in mind is always <strong>the same one sentence's worth</strong> (fixed-size state). The cost is real too: ask them "what exactly was the n-th sentence", the one with notes can find it precisely, the one with only a summary has likely forgotten the detail - exactly the lived version of SSM's "lossy compression". Neither person is smarter, they just suit different occasions: for meeting minutes the summary type is less work; in court, where every word must be cited, only the note-taker is reliable.
</div>
<h2>Recurrence vs attention: two ways to remember history</h2>
<p>First put the two roads side by side. Attention (which you learned in L19) remembers by "keeping it all": every token's Key/Value goes into the KV cache, and step t looks at all t positions before it - so memory grows <strong>linearly</strong> with the sequence and per-step compute grows <strong>linearly</strong> with the history (O(n^2) over the whole sequence). An SSM does the reverse, rolling the history <strong>into a fixed-size state h</strong>: step t takes only the previous state h_(t-1) and the current input x_t to compute the new state h_t - completely independent of how long the history before t is. How big the state is, is a hyperparameter the model fixes (<span class="mono">ssm_d_state</span>), unrelated to sequence length. That sentence sounds unremarkable but is the source of all of SSM's advantage: stretch the sequence from 1k to 100k and attention's cost grows a hundredfold, while an SSM's per-step cost does not gain a single byte.</p>
<div class="cols">
  <div class="col"><h4>Attention (KV cache)</h4><p>Every token's KV is stored, growing ever longer. Step t looks back over all t history: memory O(n), compute O(n^2). Long-range exact, but ever heavier.</p></div>
  <div class="col"><h4>SSM (recurrent state)</h4><p>Maintain only one fixed-size state h, updated in place. Step t sees only h_(t-1) and x_t: memory O(1), compute O(n). Light and steady, but the state is a lossy summary.</p></div>
</div>
<p>The most vivid picture of this difference is "the state scanning along time": inputs x_0, x_1, ... arrive one by one, the state h updates step by step in place, each step depending only on the previous state and the current input, emitting an output y on the side. In the figure below - note that the h row is <strong>always the same single cell</strong>, not dragging ever longer like the KV cache:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one state scan</b>: inputs x_0..x_4 arrive in turn, the fixed-size state h recurs h_0 -&gt; h_1 -&gt; ... -&gt; h_4, each step depending only on the previous h and the current x, while producing y. The state size never changes throughout (illustrative).</div>
<svg viewBox="0 0 680 268" width="100%" role="img" aria-label="State-space model state scan: inputs x0..x4 arrive in turn, a fixed-size state h recurs h0 -> h1 -> ... -> h4, each step depending only on the previous h and the current x, producing y">
<g font-family="ui-monospace,monospace">
<text x="40" y="54" text-anchor="middle" fill="#5b6470" font-size="10">input x</text>
<text x="40" y="142" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="10">h</text>
<text x="40" y="224" text-anchor="middle" fill="#5b6470" font-size="10">output y</text>
<rect x="68" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="96" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x0</text>
<line x1="96" y1="64" x2="96" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 96 115 L 92 107 L 100 107 z" fill="#9aa6b2"/>
<rect x="68" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="96" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h0</text>
<line x1="96" y1="160" x2="96" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 96 205 L 92 197 L 100 197 z" fill="#9aa6b2"/>
<rect x="68" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="96" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y0</text>
<rect x="184" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="212" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x1</text>
<line x1="212" y1="64" x2="212" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 212 115 L 208 107 L 216 107 z" fill="#9aa6b2"/>
<rect x="184" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="212" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h1</text>
<line x1="126" y1="138" x2="178" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 183 138 L 175 134 L 175 142 z" fill="#7c3aed"/>
<line x1="212" y1="160" x2="212" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 212 205 L 208 197 L 216 197 z" fill="#9aa6b2"/>
<rect x="184" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="212" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y1</text>
<rect x="300" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="328" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x2</text>
<line x1="328" y1="64" x2="328" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 328 115 L 324 107 L 332 107 z" fill="#9aa6b2"/>
<rect x="300" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="328" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h2</text>
<line x1="242" y1="138" x2="294" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 299 138 L 291 134 L 291 142 z" fill="#7c3aed"/>
<line x1="328" y1="160" x2="328" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 328 205 L 324 197 L 332 197 z" fill="#9aa6b2"/>
<rect x="300" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="328" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y2</text>
<rect x="416" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="444" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x3</text>
<line x1="444" y1="64" x2="444" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 444 115 L 440 107 L 448 107 z" fill="#9aa6b2"/>
<rect x="416" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="444" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h3</text>
<line x1="358" y1="138" x2="410" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 415 138 L 407 134 L 407 142 z" fill="#7c3aed"/>
<line x1="444" y1="160" x2="444" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 444 205 L 440 197 L 448 197 z" fill="#9aa6b2"/>
<rect x="416" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="444" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y3</text>
<rect x="532" y="36" width="56" height="28" rx="5" fill="#ffffff" stroke="#2563eb"/><text x="560" y="54" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">x4</text>
<line x1="560" y1="64" x2="560" y2="110" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 560 115 L 556 107 L 564 107 z" fill="#9aa6b2"/>
<rect x="532" y="116" width="56" height="44" rx="6" fill="#ece3fb" stroke="#7c3aed"/><text x="560" y="143" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="13">h4</text>
<line x1="474" y1="138" x2="526" y2="138" stroke="#7c3aed" stroke-width="2"/><path d="M 531 138 L 523 134 L 523 142 z" fill="#7c3aed"/>
<line x1="560" y1="160" x2="560" y2="200" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 560 205 L 556 197 L 564 197 z" fill="#9aa6b2"/>
<rect x="532" y="206" width="56" height="28" rx="5" fill="#ffffff" stroke="#c2630e"/><text x="560" y="224" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">y4</text>
<text x="154" y="129" text-anchor="middle" fill="#7c3aed" font-size="10">h=f(h,x)</text>
<text x="340" y="256" text-anchor="middle" fill="#5b6470" font-size="11">state size never changes -> O(1) memory, does not grow with the sequence</text>
</g></svg>
</div>
<h2>How ggml implements it: two ops + a recurrent state</h2>
<p>In ggml an SSM mainly comes down to two ops, plus a "recurrent state" read/write. First is <span class="mono">ggml_ssm_conv</span> - a <strong>causal 1D convolution</strong>: before the scan, it lets each position mix locally with the few positions before it (the convolution width is <span class="mono">ssm_d_conv</span>, usually small, e.g. 4). It acts like adding a bit of "short-term memory" to the input, letting neighboring tokens' information blend first. Then the real core, <span class="mono">ggml_ssm_scan</span> - the <strong>selective scan</strong>, which eats a whole armful of tensors: the state <span class="mono">s</span>, the input <span class="mono">x</span>, the timestep <span class="mono">dt</span> (Delta), the state matrices <span class="mono">A</span>/<span class="mono">B</span>/<span class="mono">C</span>, and the sequence indices <span class="mono">ids</span>, computing the whole sequence's recurrence in one pass. Here is the real call of these two ops in a Mamba layer (from <span class="mono">src/models/mamba-base.cpp</span>):</p>
<pre class="code"><span class="cm">// Mamba layer core (simplified from src/models/mamba-base.cpp)</span>
x = <span class="fn">ggml_ssm_conv</span>(ctx0, conv_x, layer.ssm_conv1d); <span class="cm">// 1) causal conv: local short-term mixing</span>
x = <span class="fn">ggml_silu</span>(ctx0, x);                          <span class="cm">//    activation</span>
<span class="cm">// 2) project dt, B, C from x - note these all depend on input x (this is "selective")</span>
dt = <span class="fn">build_lora_mm</span>(layer.ssm_dt, dt);            <span class="cm">//    the per-step step size Delta</span>
A  = layer.ssm_a;                                <span class="cm">//    state-transition matrix (learned param, input-independent)</span>
<span class="cm">// 3) selective scan: recur the state along time by A/B/C/dt</span>
y = <span class="fn">ggml_ssm_scan</span>(ctx0, ssm, x, dt, A, B, C, ids);</pre>
<p>Note the key detail in step 2: <span class="mono">dt</span>, <span class="mono">B</span>, <span class="mono">C</span> are all computed from the current input <span class="mono">x</span> (input-dependent), while <span class="mono">A</span> is a learned fixed parameter. This is exactly where Mamba beats old-style SSMs - detailed next section. There is also an easily-missed question: where does the state h live? It does <strong>not</strong> enter the KV cache, but a dedicated "recurrent state cache" (<span class="mono">llama-memory-recurrent</span>), each sequence taking a <strong>fixed-size</strong> block. <span class="mono">build_rs</span> (rs = recurrent state) reads the previous step's state out of this cache, feeds it to the scan, and writes the updated new state back:</p>
<pre class="code"><span class="cm">// recurrent state read - update - writeback (illustrative; see src/llama-graph.cpp build_rs)</span>
<span class="cm">// 1) read these sequences' previous-step state out of the recurrent cache</span>
states = <span class="fn">get_state_rows</span>(state_cache, ids);
<span class="cm">// 2) use the scan to advance those states forward a stretch</span>
y = <span class="fn">ggml_ssm_scan</span>(ctx0, states, x, dt, A, B, C, ids);
<span class="cm">// 3) write the final new state back to cache for the next step</span>
<span class="fn">ggml_cpy</span>(ctx0, last_state, state_cache_view);</pre>
<p>Against L17/L19 you find a neat echo: the transformer uses the KV cache to store "all the KV of the history", an SSM uses a recurrent cache to store "the one state the history compressed into". Both are caches that "keep the previous step's stuff for the next step", but one grows with the sequence and one is forever fixed-size. This is also why llama.cpp builds a dedicated <span class="mono">llama-memory-recurrent</span> for SSMs rather than stuffing them into the original KV cache - their "memory shapes" are fundamentally different. For the same reason, when a pure SSM model runs you will hardly see "the KV cache growing with the conversation" in VRAM - in its place is a small, constant-size block of state per sequence.</p>
<h2>Selective scan: what A / B / C / Delta actually do</h2>
<p>The "scan" is that recurrence above: walk the sequence start to end, pushing the state forward step by step. The hard part is the word "selective". First, the plainest recurrence, stripped of all frills, is just two lines:</p>
<pre class="code"><span class="cm"># the core recurrence of the selective scan (conceptual pseudo-code, not the real kernel)</span>
<span class="kw">for</span> t <span class="kw">in</span> 0..n:
    h = A * h + B * x[t]     <span class="cm"># new state from previous state h and current input x[t]</span>
    y[t] = C * h             <span class="cm"># read this step's output from the new state</span></pre>
<p>The four quantities each have a job: <span class="mono">A</span> governs "how much of the previous state to keep" (decay / memory), <span class="mono">B</span> governs "how the current input is written into the state", <span class="mono">C</span> governs "what is read out of the state as output", and <span class="mono">Delta</span> (dt) is the "step size", controlling how hard this step updates (think of it as the discretized time interval: large Delta ~= take one more look at the current input, small Delta ~= lean more on the old state). If A/B/C/Delta are all <strong>fixed</strong> constants, that is the classic linear SSM - fast, but rigid: it treats every token alike, unable to "tailor to who it sees". This is also the root reason early SSMs kept losing to transformers - they remember everything alike, yet in language some words are crucial and some are filler.</p>
<p>Mamba's key innovation is one sentence: <strong>let B, C, Delta vary with the input</strong> (in the previous section's code dt/B/C were all projected from x - that is what this means). This is "selectivity" - the model can, based on what it is currently reading, <strong>dynamically decide</strong> "this token matters, write it into the state a bit more" or "this token is useless, let the state ignore it". By analogy: a fixed SSM is a constant-speed conveyor belt, processing everything at the same pace; a selective SSM is a reader who <strong>adjusts their own speed</strong>, slowing to note key sentences and fast-forwarding past filler. It is exactly this "input-dependent gating" that let Mamba, for the first time, match transformers on a task like language with its uneven information density. <span class="mono">A</span> stays fixed (a learned parameter, input-independent) because it governs the state's own stable decay, and letting it jump around with the input would make the recurrence unstable - a carefully kept "anchor". In other words, Mamba treats separately what "should be rigid" in an SSM (A, for stability) and what "should be flexible" (B/C/Delta, following the input), and this judgment of "some things vary, some stay fixed" is exactly where it outdoes earlier SSMs.</p>
<p>The real kernel is of course far more complex than these two lines: it must discretize (turn the continuous A/Delta into per-step multipliers), do a parallel prefix-sum on the GPU (a parallel scan, splitting the seemingly-serial recurrence into parallelizable parts - this is the "associative scan" from the Mamba paper), and handle multiple heads and groups. But hold the through-line "<span class="mono">h = A*h + B*x; y = C*h</span>, with B/C/Delta following the input" and you have the soul of an SSM - the rest is engineering to move it efficiently onto hardware.</p>
<table class="t">
  <tr><th>Dimension</th><th>Attention (transformer)</th><th>SSM (Mamba/RWKV)</th></tr>
  <tr><td>How it remembers</td><td>stores every token's KV</td><td>one fixed-size state h</td></tr>
  <tr><td>Memory vs sequence</td><td>grows linearly O(n)</td><td>constant O(1)</td></tr>
  <tr><td>Whole-sequence compute</td><td>O(n^2)</td><td>O(n)</td></tr>
  <tr><td>Best at</td><td>long-range exact retrieval / copy</td><td>very long sequences / streaming / low VRAM</td></tr>
  <tr><td>llama.cpp cache</td><td>KV cache (L19)</td><td>llama-memory-recurrent</td></tr>
</table>
<h2>Deeper: the sweet and bitter of long sequences, and hybrid architectures</h2>
<p>Two last folds, answering the two questions most worth thinking through when deploying SSMs.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why are SSMs good for long sequences? And what is the cost? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The sweet first. Attention's KV cache grows linearly with the sequence, and step t must compute attention against all t positions before it - push the sequence to hundreds of thousands of tokens and both memory and compute blow up. An SSM's state is fixed-size and each step sees only the previous state: memory O(1), whole-sequence compute O(n), and however long the sequence the per-step cost does not budge. This makes it especially sweet for <strong>very long context, streaming input, and low-VRAM on-device</strong> - you can even feed tokens in almost endlessly and memory does not grow. Now the bitter. A fixed-size state is in essence a <strong>lossy compression</strong> of arbitrarily long history into a small vector. When there is too much worth remembering and the state cannot hold it, it can only forget selectively. So SSMs are naturally handicapped on "exact retrieval" tasks: e.g. "copy out the 3000th token verbatim" (the copy task), or needing to look back and exactly compare some detail from long ago - attention can look back and check each KV, while an SSM has only one repeatedly-overwritten summary. This is not poor implementation, it is the fundamental cost of choosing "fixed state": you traded "exact random access to arbitrary history" for "cost that does not grow with length". So a practical intuition: the more a task leans "sequential, rarely looking back" (long-document summary, streaming transcription, audio), the more SSM wins; the more it leans "exact look-up, needle-in-a-haystack" (retrieval by ID, finding one verbatim sentence in a long doc), the steadier attention is.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Since each has strengths, why not mix the two? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>That is exactly today's most popular approach - the <strong>hybrid architecture</strong>. Since attention excels at exact retrieval and SSMs at efficiently handling long sequences, let some layers use SSM and some use attention, each playing to its strength. Many new models (like Jamba, and various Mamba-Transformer hybrids) are built this way: most layers use SSM to carry the length and save memory, a few layers insert attention to restore exact-retrieval ability. llama.cpp built <span class="mono">llama-memory-hybrid</span> for this - a memory subsystem that manages both kinds <strong>at once</strong>: the attention layers' part goes through the KV cache, the SSM layers' part through the recurrent state cache, the two caches coexisting in one model. This echoes an attitude recurring across these lessons: architecture is not an either/or article of faith but an engineering combination of tradeoffs. The foundation you built in the first six parts (compute graph L09/L10, memory management L17, KV cache L19) still holds up here at the hybrid architecture - only now, within one model, some layers remember "all the KV" and some remember "one state", and the scheduler must serve both. This neatly shows one thing: mastering the transformer's caching and scheduling does not become obsolete when SSMs appear - it is precisely the prerequisite for understanding hybrid architectures.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>Core tradeoff: attention stores all KV (memory O(n), compute O(n^2), long-range exact); an SSM uses a fixed-size recurrent state h (memory O(1), compute O(n), a lossy summary).</li>
    <li>Recurrence: <span class="mono">h_t = f(A,B,C,Delta; h_(t-1), x_t)</span>, <span class="mono">y_t = C.h_t</span> - each step sees only the previous state and current input, state size fixed by <span class="mono">ssm_d_state</span>.</li>
    <li>ggml's two ops: <span class="mono">ggml_ssm_conv</span> (causal conv, local mixing) + <span class="mono">ggml_ssm_scan(s,x,dt,A,B,C,ids)</span> (selective scan); the state is read/written via <span class="mono">build_rs</span> from <span class="mono">llama-memory-recurrent</span>.</li>
    <li>"Selectivity" = let <span class="mono">B/C/Delta</span> vary with the input (input-dependent gating), the model learning to dynamically "remember / ignore" - the key to Mamba matching transformers.</li>
    <li>Cost: a fixed state is lossy compression, long-range exact retrieval (copy-like) worse than attention; hence the common <strong>hybrid architecture</strong> (<span class="mono">llama-memory-hybrid</span>) taking the best of both.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  The thing to take from state-space models is not some Mamba formula, but how it forces you to re-examine an assumption you long took for granted: <strong>"the model must remember the whole history" is actually a choice, not a law</strong>. Attention chose "keep it all, look back anytime", and got the O(n^2) cost; the SSM chose "keep only a rolling summary", and won the O(n) lightness while accepting the loss of being "fuzzy on details". Neither is absolutely better - they merely stand at different points on the "exact vs efficient" axis. This is precisely the shared motif of Part 7's four lessons: speculative decoding re-examined "the seriality of emitting tokens", MoE re-examined "the binding of compute and parameters", multimodality re-examined "the form of the input", and the SSM re-examined "how history is stored". Each challenges a default "taken as self-evident" and offers another possibility. And all of them stand on the foundation you built in the first six parts - which is exactly the step from "using llama.cpp" toward "understanding llama.cpp, even improving it".
</div>

<p>Part 7 closes here. From speculative decoding, MoE, and multimodality to state-space models, we have taken apart four advanced mechanisms "outside the standard transformer" one by one - each either squeezing the bottleneck from a new angle, trading structure for efficiency, or outright replacing a pillar. Next stop, Part 8 turns from "how the model computes" to "how the engineering lands": converting a HuggingFace model to GGUF, compiling/debugging/testing, and how to genuinely contribute to llama.cpp. Having finished the first seven parts, you have followed the path from "how a model runs" to "how frontier architectures change"; next, it is time to put that understanding into your own hands.</p>

""",
}
