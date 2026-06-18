"""Content for Part 6 (low-level kernels)."""

LESSON_31 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前面讲了算子"要做什么矩阵乘"（L11），可它最终是怎么在一块 CPU 上、用真实的机器指令一步步算出来的？这一课钻到最底层，看 ggml 的 CPU 后端（<span class="mono">ggml/src/ggml-cpu/</span>）怎么把一次点积/矩阵乘，从"标量一个一个乘"加速到"SIMD 一条指令算一排"，再切给多个线程并行。这是整个教程最硬核的一段，但也最能让你看清"性能到底从哪来"。
</p>
<p style="color:var(--muted);margin-top:.4rem">我们以<strong>量化点积</strong>（quantized dot product）这个推理里最热的内核为线索：它是矩阵乘的最内层循环，模型每生成一个 token 都要把它跑无数遍——把它算快，整个模型就快。这一课会逐行读真实的 AVX2 SIMD 代码；别怕，配着图你会发现，它的核心思路其实朴素得很。你会渐渐发现，所谓"硬核"，难的从来不是某一行代码在算什么，而是要同时在脑子里装下"数据怎么排布、指令怎么并行、缓存怎么命中"这么几条线——而图，正是帮你把这几条线一次看清的工具。</p>
<p style="color:var(--muted)">路线图：先看"标量 vs SIMD"的差别（一次算 1 个 vs 一次算 8 个），再看量化点积怎么把 4-bit 权重解包并向量化，最后看多线程怎么把一次大矩阵乘切开、让多个核一起算。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  CPU 后端的性能秘诀，归根结底就两招：<strong>数据并行</strong>（SIMD——一条指令同时处理一排数）和<strong>任务并行</strong>（多线程——多个核同时干活）。前者榨干单个核里闲置的算力，后者把活儿摊给所有核。这一课的真实代码看着满屏 intrinsic，但你只要抓住这两招，就抓住了主线：每一个看不懂的 <span class="mono">_mm256_*</span> 函数，本质都在做"把标量循环里一次一个的操作，变成一次一排"。读懂 CPU 后端，你就明白了为什么同一个模型，换个编译选项（开没开 AVX2）、改个线程数，速度能差好几倍。再把这件事说透一点：现代 CPU 其实"很宽"——一个核里有好几条能并行的运算流水线，还有一组很宽的向量寄存器。标量代码只用到其中很窄的一条，剩下的全闲着；SIMD 就是去占满那组宽寄存器，多线程是去占满所有核。所以"优化"在 CPU 上常常不是"想出更聪明的算法"，而是"把本来就有、却闲着的硬件用起来"。这也是为什么读底层内核时，最该问的不是"它在算什么"（那还是普通的点积），而是"它把硬件用满了没有"。把这个视角装进脑子，后面所有 <span class="mono">_mm256_*</span>、CUDA 线程、显存分块，都会从"天书"变成"占满硬件的不同手段"。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  SIMD 像流水线上的<strong>多头机械臂</strong>：普通机械臂一次拧一颗螺丝，多头机械臂一次同时拧 8 颗。活儿没变（还是拧螺丝），但一趟干 8 颗，自然快 8 倍。CPU 的 SIMD 寄存器就是这样一只"8 头机械臂"——一个 256 位的 <span class="mono">__m256</span> 寄存器，正好装下 8 个 32 位 float，一条指令就对这 8 个同时做乘加。多线程则像<strong>开了 8 条流水线</strong>：每条线都有自己的多头机械臂，一起开工。两招叠起来，就是"8 条线 × 每条一次 8 颗"的吞吐。这个类比还能帮你记住一个常见的坑：多头机械臂要快，前提是"8 颗螺丝得同时备好在手边"。要是螺丝得一颗颗从远处仓库取（数据从慢内存搬），机械臂再快也得干等——这就是 tiling（把螺丝先搬到手边的小料盒里）存在的理由。所以光有 SIMD 还不够，还得让数据"近"。记住这幅画面：算得快（SIMD / 多核）和喂得饱（缓存 / tiling）是一对孪生兄弟，少了谁都跑不快。
</div>

<h2>从标量到 SIMD</h2>
<p>先看最朴素的做法。点积就是"对应位置相乘再求和"：<span class="mono">sum += a[i]*b[i]</span>，循环 n 次。这是 <span class="mono">ggml-cpu/vec.cpp</span> 里 <span class="mono">ggml_vec_dot_f32</span> 标量参考实现的内核——正确、好懂，但慢：CPU 每个时钟周期只处理一个数，宽大的运算单元大半闲着。这份标量版还有一个常被忽视的用处：它是所有 SIMD 特化版的"正确性基准"。任何一个架构的向量实现，结果都必须和它对得上——整型量化内核要求逐位一致，浮点内核因为用了多个累加器、求和顺序变了，会有极小的舍入差异，但凡差得明显就是 bug。所以读底层内核时，先把这份慢而对的标量版看懂，再去看快的 SIMD 版，你心里就有了一把"对不对"的尺子——这也是一种很实用的读码顺序：先抓正确、再抓快。</p>
<pre class="code"><span class="cm">// 标量: 一次算一个 (简化自 ggml-cpu/vec.cpp 的 ggml_vec_dot_f32)</span>
float sum = 0;
<span class="kw">for</span> (int i = 0; i &lt; n; i++)
    sum += a[i] * b[i];                       <span class="cm">// 一个乘加, 重复 n 次</span>

<span class="cm">// SIMD / AVX2: 一条指令算 8 个 (vec.cpp 的 GGML_SIMD 路径)</span>
__m256 acc = <span class="fn">_mm256_setzero_ps</span>();             <span class="cm">// 8 路 float 累加器</span>
<span class="kw">for</span> (int i = 0; i &lt; n; i += 8) {
    __m256 va = <span class="fn">_mm256_loadu_ps</span>(a + i);       <span class="cm">// 一次载 8 个 a</span>
    __m256 vb = <span class="fn">_mm256_loadu_ps</span>(b + i);       <span class="cm">// 一次载 8 个 b</span>
    acc = <span class="fn">_mm256_fmadd_ps</span>(va, vb, acc);       <span class="cm">// acc += va*vb, 8 路同时</span>
}
float sum = <span class="fn">hsum_float_8</span>(acc);               <span class="cm">// 8 路水平求和 -> 标量</span></pre>
<p>右边的 SIMD（Single Instruction Multiple Data，单指令多数据）就是来榨干那部分闲置算力的。AVX2 提供 256 位的 <span class="mono">__m256</span> 寄存器，一个正好装 8 个 float；一条 <span class="mono">_mm256_fmadd_ps</span>（fused multiply-add，乘加融合）指令，让这 8 个 <strong>lane（通道）</strong>同时各做一次 <span class="mono">acc[i] += a[i]*b[i]</span>。循环步长因此从 1 变成 8，指令数少了八分之七。ARM 的 NEON 是 128 位、一次 4 个 float，思路完全一样，只是宽度减半。</p>
<p>把这"8 路并行"画出来最直观。下面追踪一次 SIMD 点积：8 对数同时乘加进 8 个累加器，循环若干轮后，再用一次<strong>水平求和</strong>（horizontal sum，hsum）把 8 个累加器合成最终的一个标量。</p>
<div class="trace">
  <div class="tcap"><b>追踪一次 SIMD 点积</b>：8 个 float 装进一个 256 位寄存器，一条 fmadd 指令让 8 路同时乘加，最后水平求和成一个标量（示意）。</div>
<svg viewBox="0 0 640 250" width="100%" role="img" aria-label="SIMD 点积示例：一条指令让 8 个通道同时乘加，再水平求和成一个标量">
<g font-family="ui-monospace,monospace">
<text x="50" y="42" text-anchor="end" fill="#5b6470" font-size="11">向量 a</text>
<text x="50" y="100" text-anchor="end" fill="#5b6470" font-size="11">向量 b</text>
<text x="50" y="200" text-anchor="end" fill="#5b6470" font-size="11">累加器</text>
<rect x="60" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="88" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a0</text>
<rect x="60" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="88" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b0</text>
<rect x="60" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="88" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s0</text>
<rect x="128" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="156" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a1</text>
<rect x="128" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="156" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b1</text>
<rect x="128" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="156" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s1</text>
<rect x="196" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="224" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a2</text>
<rect x="196" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="224" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b2</text>
<rect x="196" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="224" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s2</text>
<rect x="264" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="292" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a3</text>
<rect x="264" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="292" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b3</text>
<rect x="264" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="292" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s3</text>
<rect x="332" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="360" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a4</text>
<rect x="332" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="360" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b4</text>
<rect x="332" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="360" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s4</text>
<rect x="400" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="428" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a5</text>
<rect x="400" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="428" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b5</text>
<rect x="400" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="428" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s5</text>
<rect x="468" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="496" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a6</text>
<rect x="468" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="496" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b6</text>
<rect x="468" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="496" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s6</text>
<rect x="536" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="564" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a7</text>
<rect x="536" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="564" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b7</text>
<rect x="536" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="564" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s7</text>
<rect x="60" y="128" width="532" height="34" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<text x="320" y="150" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="13">_mm256_fmadd_ps：一条指令，8 路同时 s[i] += a[i]*b[i]</text>
<line x1="88" y1="114" x2="88" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 88 128 L 84 121 L 92 121 z" fill="#9aa6b2"/>
<line x1="88" y1="162" x2="88" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 88 180 L 84 173 L 92 173 z" fill="#9aa6b2"/>
<line x1="156" y1="114" x2="156" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 156 128 L 152 121 L 160 121 z" fill="#9aa6b2"/>
<line x1="156" y1="162" x2="156" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 156 180 L 152 173 L 160 173 z" fill="#9aa6b2"/>
<line x1="224" y1="114" x2="224" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 224 128 L 220 121 L 228 121 z" fill="#9aa6b2"/>
<line x1="224" y1="162" x2="224" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 224 180 L 220 173 L 228 173 z" fill="#9aa6b2"/>
<line x1="292" y1="114" x2="292" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 292 128 L 288 121 L 296 121 z" fill="#9aa6b2"/>
<line x1="292" y1="162" x2="292" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 292 180 L 288 173 L 296 173 z" fill="#9aa6b2"/>
<line x1="360" y1="114" x2="360" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 360 128 L 356 121 L 364 121 z" fill="#9aa6b2"/>
<line x1="360" y1="162" x2="360" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 360 180 L 356 173 L 364 173 z" fill="#9aa6b2"/>
<line x1="428" y1="114" x2="428" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 428 128 L 424 121 L 432 121 z" fill="#9aa6b2"/>
<line x1="428" y1="162" x2="428" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 428 180 L 424 173 L 432 173 z" fill="#9aa6b2"/>
<line x1="496" y1="114" x2="496" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 496 128 L 492 121 L 500 121 z" fill="#9aa6b2"/>
<line x1="496" y1="162" x2="496" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 496 180 L 492 173 L 500 173 z" fill="#9aa6b2"/>
<line x1="564" y1="114" x2="564" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 564 128 L 560 121 L 568 121 z" fill="#9aa6b2"/>
<line x1="564" y1="162" x2="564" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 564 180 L 560 173 L 568 173 z" fill="#9aa6b2"/>
<text x="320" y="74" text-anchor="middle" fill="#5b6470" font-size="11">8 个 float 装进一个 256 位寄存器</text>
<line x1="88" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 311 231 L 311 224 z" fill="#7c3aed"/>
<line x1="156" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 311 231 L 311 224 z" fill="#7c3aed"/>
<line x1="224" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 310 230 L 312 223 z" fill="#7c3aed"/>
<line x1="292" y1="208" x2="312" y2="224" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 310 227 L 315 221 z" fill="#7c3aed"/>
<line x1="360" y1="208" x2="324" y2="225" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 323 222 L 326 228 z" fill="#7c3aed"/>
<line x1="428" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 324 223 L 326 230 z" fill="#7c3aed"/>
<line x1="496" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 325 224 L 325 231 z" fill="#7c3aed"/>
<line x1="564" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 325 224 L 325 231 z" fill="#7c3aed"/>
<rect x="300" y="226" width="40" height="18" rx="4" fill="#ffffff" stroke="#7c3aed"/><text x="320" y="239" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">sum</text>
<text x="356" y="239" fill="#5b6470" font-size="11">水平求和 hsum -> 一个标量</text>
</g></svg>
</div>

<h2>量化点积怎么算</h2>
<p>真实推理里，权重是被量化压过的（L29），点积要先<strong>解包</strong>再算。以最常用的 <span class="mono">vec_dot_q4_0_q8_0</span> 为例：权重是 Q4_0——每 32 个一组打包成 16 字节的 4-bit 值（<span class="mono">block_q4_0</span> = 一个 fp16 的 <span class="mono">d</span>（scale）+ <span class="mono">qs[16]</span>），激活是 Q8_0 的 int8。所以一次量化点积的内层，是这么一串：解包 4-bit -> 减 8 偏移 -> 乘 int8 激活 -> 乘回 scale -> 累加。下面这段就是它真实的 AVX2 实现，逐行看。</p>
<pre class="code"><span class="cm">// 真实 AVX2 量化点积核心 (arch/x86/quants.c vec_dot_q4_0_q8_0)</span>
__m256 acc = <span class="fn">_mm256_setzero_ps</span>();
<span class="kw">for</span> (; ib &lt; nb; ++ib) {                               <span class="cm">// 遍历 block (每块 32 权重)</span>
    __m256  d  = <span class="fn">_mm256_set1_ps</span>(dx * dy);            <span class="cm">// 合并两块的 fp16 scale</span>
    __m256i qx = <span class="fn">bytes_from_nibbles_32</span>(x[ib].qs);    <span class="cm">// 解包: 16 字节 -> 32 个 [0..15]</span>
    qx = <span class="fn">_mm256_sub_epi8</span>(qx, <span class="fn">_mm256_set1_epi8</span>(8));   <span class="cm">// 偏移到 [-8..+7]</span>
    __m256i qy = <span class="fn">_mm256_loadu_si256</span>((const __m256i*)y[ib].qs); <span class="cm">// 载 32 个 int8 激活</span>
    __m256  q  = <span class="fn">mul_sum_i8_pairs_float</span>(qx, qy);     <span class="cm">// int8 点积 -> float</span>
    acc = <span class="fn">_mm256_fmadd_ps</span>(d, q, acc);                <span class="cm">// FMA: acc += d * q</span>
}
float sumf = <span class="fn">hsum_float_8</span>(acc);                      <span class="cm">// 8 路水平求和 -> 标量</span></pre>
<p>逐行拆开：<span class="mono">acc</span> 是 8 路 float 累加器；循环每轮处理一个 block——<span class="mono">bytes_from_nibbles_32</span> 把 16 字节解包成 32 个 [0..15] 的值（用 <span class="mono">0xF</span> 掩码取低 4 位、移位取高 4 位）；减 8 偏移到 [-8..+7]（4-bit 量化是有符号的）；<span class="mono">_mm256_loadu_si256</span> 一次载入 32 个 int8 激活；<span class="mono">mul_sum_i8_pairs_float</span> 做 int8 点积、得到 8 个 float；最后 <span class="mono">_mm256_fmadd_ps</span> 把它乘上合并的 scale 累加进 <span class="mono">acc</span>。所有 block 循环完，<span class="mono">hsum_float_8</span> 把 8 路累加器水平求和成最终标量。"解包 + 向量化乘加"这套，就是量化模型在 CPU 上跑得动的关键。顺带说一个容易被忽略的点：为什么激活用 Q8_0（int8）、权重用 Q4_0（4-bit），两边精度不一样？因为角色不同——权重是死的、量又大，压到 4-bit 省内存最划算；激活是活的、范围动态，留 int8 才稳。而在硬件层面，int8 的乘加有专门的快指令（如 <span class="mono">_mm256_maddubs_epi16</span>），比纯浮点点积还快。所以"权重 4-bit、激活 int8"这套搭配既省内存又跑得快，是社区量化方案的主流，也是这段内核为什么要先解包再算 int8 的根由。</p>
<p>把一个 block 的处理流程单独拎出来定格看，会更清楚每一步在干什么：</p>
<div class="trace">
  <div class="tcap"><b>追踪一个 block</b>：32 个权重从 16 字节 4-bit，一路解包、偏移、与激活点积、乘 scale 累加（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 打包</h5>
      <div class="cellrow"><span class="vc">qs[16]</span></div>
      <div class="tlab">32 个 4-bit 权重</div></div>
    <div class="op">解包<br>nibbles</div>
    <div class="stn"><h5>② 解包</h5>
      <div class="cellrow"><span class="vc">32 个 [0..15]</span></div>
      <div class="tlab">取低/高 4 位</div></div>
    <div class="op">-8<br>偏移</div>
    <div class="stn"><h5>③ 偏移</h5>
      <div class="cellrow"><span class="vc">[-8..+7]</span></div>
      <div class="tlab">有符号量化值</div></div>
    <div class="op">点积<br>int8</div>
    <div class="stn"><h5>④ 点积</h5>
      <div class="cellrow"><span class="vc blue">dot(qx, qy)</span></div>
      <div class="tlab">与 Q8_0 激活</div></div>
    <div class="op">×scale<br>累加</div>
    <div class="stn"><h5>⑤ 累加</h5>
      <div class="cellrow"><span class="vc hot">acc += d*dot</span></div>
      <div class="tlab">乘回 fp16 scale</div></div>
  </div>
</div>

<h2>多线程：让多个核一起算</h2>
<p>SIMD 榨干了单个核；多线程则让所有核一起上。一次大矩阵乘有很多输出行，而各行的计算彼此<strong>独立</strong>（算第 5 行不需要第 3 行的结果），所以天然适合并行：把行平均分给 N 个线程，每个线程算自己那一批，最后汇合。这种"互不依赖、可任意切分"的结构，正是数据并行最理想的对象。这里也藏着一个朴素却重要的判断：能不能并行，先看"有没有依赖"。行与行之间没有先后关系，就能随便切；一旦有依赖（后一步要等前一步的结果），并行就得加同步、加等待，收益立刻打折。后面 L32 看 GPU 时你会发现，是同一条判断标准在起作用。</p>
<div class="cols">
  <div class="col"><h4>线程 0</h4><p>算输出矩阵第 0..k 行（每行内部再用 SIMD 点积）。</p></div>
  <div class="col"><h4>线程 1</h4><p>算第 k..2k 行，和线程 0 同时进行、互不等待。</p></div>
  <div class="col"><h4>线程 2 / 3 / ...</h4><p>各算自己那批行；行间无依赖，切多少份都行。</p></div>
</div>
<p>ggml 的实现很轻量：每个算子 <span class="mono">ggml_compute_forward_*</span>（<span class="mono">ggml-cpu.c</span>）都拿到 <span class="mono">params-&gt;ith</span>（我是第几个线程）和 <span class="mono">params-&gt;nth</span>（一共几个线程），据此算出"我负责哪几行/哪几块"，各算各的。线程池由 <span class="mono">ggml-threading</span> 维护，避免反复创建销毁线程的开销。<strong>SIMD（核内一次 8 个）× 多线程（跨核同时干）</strong>两招叠加，就是 CPU 后端吞吐的全部来源——没有魔法，只是把同一份活儿尽可能地铺开同时做。值得一提的是，并不是所有算子都像矩阵乘这样好切。像 softmax、RMSNorm 这类要"先看全行再算"的归约操作，切分时得小心边界；而逐元素的算子（加法、激活函数）则和矩阵乘一样、随便切。ggml 给每个算子单独写切分逻辑，正是为了照顾这些差异。读源码时你会看到，几乎每个 <span class="mono">ggml_compute_forward_*</span> 开头都在用 <span class="mono">ith/nth</span> 算自己负责的范围——这就是多线程在算子层落地的样子。</p>

<h2>三招怎么叠起来：从点积到矩阵乘</h2>
<p>前面见过了 SIMD（核内一次 8 路）和多线程（跨核同时干），也提到过 tiling（让缓存里的数据多复用，细节在文末折叠里）。但在一次真实的矩阵乘里，这三招并不是各管各的，而是<strong>层层套在一起</strong>同时发力——把这层关系看清，才算真正读懂了"CPU 后端的快到底从哪来"。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">最外</span><span class="name">多线程 (跨核)</span></div><div class="ld">把输出矩阵的行切给 N 个线程, 每个核负责一批, 互不依赖</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">中间</span><span class="name">tiling (缓存)</span></div><div class="ld">每个线程把自己那批再切成小块, 一块块算, 让数据在缓存里反复复用</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">最内</span><span class="name">SIMD (核内)</span></div><div class="ld">算每个小块里的点积时, 用一条 fmadd 一次 8 路并行</div></div>
</div>
<p>三层一叠，效果是<strong>相乘</strong>的：多线程把活儿铺满所有核，tiling 让每个核都不必苦等内存，SIMD 再把每个核内部那只"机械臂"开到 8 头。任意一层缺位，整体都会被拖慢——只开多线程却不向量化，单核还是慢吞吞；只向量化却不分块，数据还在内存与缓存之间反复跑路。正因为三招齐上，ggml 才能在一台没有任何 GPU 的纯 CPU 机器上，把一个几 GB 的量化模型跑得有模有样。</p>
<p>这也正好解释了实战里那些调优经验：线程数通常设到物理核数附近最优（再多就互相争抢、得不偿失）；而把编译选项从无 SIMD 换成开启 AVX2、甚至 AVX-512，速度往往一下子翻倍——你现在知道，那是因为把"最内层"那只机械臂，从一次 1 个，换成了一次 8 个、16 个。底层内核看着玄，规律却很实在：哪一层没铺满，就在哪一层补；想知道补哪层，就回到 L30 的两把尺子去量。</p>

<h2>深入：缓存与多架构</h2>
<p>最后两个折叠，补两个让 CPU 后端真正快起来、却容易被忽略的工程细节：分块对缓存的意义，以及一份代码怎么适配各种 CPU。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> 分块（tiling）为什么比朴素三重循环快？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>朴素的矩阵乘是三重循环，每算一个输出元素都要把 A 的一整行、B 的一整列从内存扫一遍。矩阵一大，这些数据塞不进 CPU 的高速缓存（cache），于是反复从慢几十倍的主内存搬运——瓶颈不在"算"，而在"等数据"。<span class="mono">llamafile/sgemm.cpp</span> 和 <span class="mono">repack.cpp</span> 用的是<strong>分块（tiling）</strong>：把大矩阵切成刚好能放进缓存的小块，先把一小块载入缓存、把它能参与的计算全做完、再换下一块。同样的乘加次数，但每个数据载入后被<strong>充分复用</strong>，访存大大减少。这也呼应 L30：很多算子是"访存密集"的，省下访存就是省时间。tiling 是几乎所有高性能矩阵乘（CPU 的 BLAS、GPU 的 mmq，见 L32）的共同套路。顺带提一句和 tiling 搭档的 <span class="mono">repack</span>：它在加载权重时就把数据重排成"对缓存和 SIMD 都更友好"的布局，让运行时取数更顺、向量化更整齐。这是"预处理换运行时速度"的典型——多花一点加载时间，换之后无数次推理都更快，和 L29 的 imatrix（推理前先测量权重重要性）是同一种思路：把能提前做的事提前做掉。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 一份代码怎么适配各种 CPU？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>同样一个 <span class="mono">vec_dot_q4_0_q8_0</span>，x86 上想用 AVX2、ARM 上想用 NEON、老 CPU 上只能退回标量——怎么一份源码全照顾到？ggml 把架构特化的实现放在 <span class="mono">ggml-cpu/arch/{x86,arm,riscv,...}/</span> 下，用<strong>编译期 + 运行期</strong>两层分派：编译期用 <span class="mono">#if defined(__AVX2__)</span> 这类宏，只把当前架构支持的指令编进去；运行期再检测 CPU 实际有没有某条指令集（feature detection），挑一条最快的实现走。所以你下到的同一个二进制，在新 CPU 上自动用上 AVX-512、在老机器上稳稳退回标量，不会因为用了高级指令而崩掉。"一份代码、多架构最优"正是 ggml 能在五花八门的设备上跑起来的底气。这种"编译期裁剪 + 运行期挑选"的两层分派，其实是跨平台高性能库的通用做法。代价是源码里 <span class="mono">#if</span> 满天飞、同一个函数有好几份架构特化版，读起来枝杈很多；但换来的是"一次编译、处处最优"。所以你读 <span class="mono">arch/</span> 目录时不必把每个架构都啃下来——抓住 x86 这一支看懂，其余的无非是同一套思路换一组 intrinsic 名字而已。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>CPU 后端两招提速：<strong>SIMD</strong>（数据并行，一条指令算一排）+ <strong>多线程</strong>（任务并行，多个核同时算）。</li>
    <li>SIMD：AVX2 的 <span class="mono">__m256</span> 一次装 8 个 float，<span class="mono">_mm256_fmadd_ps</span> 一条指令做 8 路乘加；NEON 一次 4 个。末尾 <span class="mono">hsum</span> 水平求和成标量。</li>
    <li>量化点积 <span class="mono">vec_dot_q4_0_q8_0</span>：<span class="mono">block_q4_0</span> 每 32 权重共享一个 fp16 scale；内层 = 解包 4-bit -&gt; 偏移 -&gt; int8 点积 -&gt; <span class="mono">fmadd</span> 乘 scale 累加。</li>
    <li>多线程：算子用 <span class="mono">ith/nth</span> 把输出行切给各线程（行间独立、天然可并行）；线程池在 <span class="mono">ggml-threading</span>。</li>
    <li>更快的细节：分块（tiling）改善缓存复用；<span class="mono">arch/</span> 用编译期+运行期分派做到"一份代码多架构最优"。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  这一课最值得带走的，不是某个 intrinsic 的名字，而是一种看性能的眼光：<strong>同样的运算量，怎么把它"铺开"同时做、又怎么让数据少跑路</strong>。SIMD 是"把一次一个铺成一次一排"，多线程是"把一份活摊给多个核"，tiling 是"让载入缓存的数据多被复用几次"——三招都不改变要算的总量，只改变"怎么算"。这正是高性能计算的母题，下一课的 GPU（CUDA）会把同样的思路推到极致：几千上万个线程同时算、用片上内存复用数据。所以别被满屏的 <span class="mono">_mm256_*</span> 吓住——看穿它们，你看到的是同一个朴素而强大的念头：让闲着的硬件都忙起来。顺着这个念头，你甚至能预测很多优化的样子：看到一个慢的循环，先问"它能不能一次多算几个？"（SIMD）、"能不能拆给多个核？"（多线程）、"数据是不是反复在远处取？"（缓存 / tiling）。这三问几乎覆盖了 CPU 性能优化的大半江山。而它们背后是同一条朴素的经济学：硬件的算力和带宽是你买机器时就已付清的固定成本，闲着就是白白浪费；所谓高性能，不过是想方设法把这笔已付的成本用满。下一课进 GPU，你会看到把这套"用满硬件"的哲学推到几千个线程的极致——但内核思路，和这一课一模一样。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
We covered ops saying WHAT matmul to do (L11), but how does it finally get computed, step by step, on a CPU with real machine instructions? This lesson drops to the lowest level and watches ggml's CPU backend (<span class="mono">ggml/src/ggml-cpu/</span>) take one dot product / matmul from "scalar, one multiply at a time" up to "SIMD, one instruction does a whole row", then split it across threads. This is the most hardcore stretch of the whole guide - but also the best place to see where performance actually comes from.
</p>
<p style="color:var(--muted);margin-top:.4rem">We use the <strong>quantized dot product</strong> as our thread - the hottest kernel in inference. It is the innermost loop of matmul, run countless times for every token the model generates; make it fast and the whole model is fast. This lesson reads real AVX2 SIMD code line by line; do not be scared - with the diagrams you will find its core idea is actually plain. You will gradually see that "hardcore" is hard not because of what any line computes, but because you must hold several threads in your head at once - how data is laid out, how instructions run in parallel, how the cache hits - and a diagram is exactly the tool that lets you see all those threads at a glance.</p>
<p style="color:var(--muted)">Roadmap: first the "scalar vs SIMD" difference (one at a time vs eight at once), then how the quantized dot product unpacks 4-bit weights and vectorizes them, and finally how multithreading splits one big matmul so many cores compute together.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The CPU backend's performance secret boils down to two moves: <strong>data parallelism</strong> (SIMD - one instruction processes a whole row of numbers) and <strong>task parallelism</strong> (multithreading - many cores work at once). The first squeezes idle compute out of a single core; the second spreads the work across all cores. The real code looks like a screen of intrinsics, but grasp these two moves and you have the through-line: every cryptic <span class="mono">_mm256_*</span> function is essentially turning "one-at-a-time in a scalar loop" into "a whole row at once". Understand the CPU backend and you see why the same model, with a different build flag (AVX2 on or off) or a different thread count, can be several times faster or slower. To put it plainly: a modern CPU is actually "wide" - one core has several parallel execution pipelines and a set of wide vector registers. Scalar code uses only a narrow sliver of this, leaving the rest idle; SIMD goes to fill those wide registers, multithreading goes to fill all the cores. So "optimization" on a CPU is often not "invent a cleverer algorithm" but "put to use the hardware you already have sitting idle". That is why, reading low-level kernels, the question to ask is not "what is it computing" (still a plain dot product) but "is it using the hardware fully". Plant this view and every <span class="mono">_mm256_*</span>, CUDA thread, and VRAM tile later turns from gibberish into "different ways to fill the hardware".
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  SIMD is like a <strong>multi-head robotic arm</strong> on an assembly line: an ordinary arm tightens one screw at a time, a multi-head arm tightens 8 at once. The job is unchanged (still tightening screws), but doing 8 per pass is naturally 8x faster. A CPU's SIMD register is exactly such an "8-head arm" - one 256-bit <span class="mono">__m256</span> register holds 8 32-bit floats, and one instruction multiply-adds all 8 at once. Multithreading is like running <strong>8 assembly lines</strong>: each line has its own multi-head arm, all working together. Stack the two and you get "8 lines x 8 screws each per pass" of throughput. The analogy also helps you remember a common trap: a multi-head arm is only fast if "8 screws are ready at hand at once". If each screw must be fetched from a distant warehouse (data hauled from slow memory), the fastest arm still waits idle - which is exactly why tiling exists (move the screws into a small tray at hand first). So SIMD alone is not enough; data must also be "near". Keep this picture: computing fast (SIMD / many cores) and being fed enough (cache / tiling) are twin brothers - miss either and it will not run fast.
</div>

<h2>From scalar to SIMD</h2>
<p>First the plainest way. A dot product is "multiply matching positions and sum": <span class="mono">sum += a[i]*b[i]</span>, looped n times. This is the kernel of the scalar reference <span class="mono">ggml_vec_dot_f32</span> in <span class="mono">ggml-cpu/vec.cpp</span> - correct and readable, but slow: the CPU processes one number per cycle while its wide execution units sit mostly idle. This scalar version has an often-overlooked use too: it is the "correctness baseline" for every SIMD specialization. Any architecture's vector implementation must line up with it - bit for bit for the integer quantized kernels, within a tiny rounding error for float (multiple accumulators reorder the summation) - and anything off by more is a bug. So when reading low-level kernels, understand this slow-but-correct scalar version first, then read the fast SIMD version, and you hold a yardstick for "is it right" - a very practical reading order: grasp correct first, then grasp fast.</p>
<pre class="code"><span class="cm">// scalar: one at a time (simplified from ggml-cpu/vec.cpp ggml_vec_dot_f32)</span>
float sum = 0;
<span class="kw">for</span> (int i = 0; i &lt; n; i++)
    sum += a[i] * b[i];                       <span class="cm">// one multiply-add, repeated n times</span>

<span class="cm">// SIMD / AVX2: one instruction does 8 (vec.cpp GGML_SIMD path)</span>
__m256 acc = <span class="fn">_mm256_setzero_ps</span>();             <span class="cm">// 8-lane float accumulator</span>
<span class="kw">for</span> (int i = 0; i &lt; n; i += 8) {
    __m256 va = <span class="fn">_mm256_loadu_ps</span>(a + i);       <span class="cm">// load 8 a's at once</span>
    __m256 vb = <span class="fn">_mm256_loadu_ps</span>(b + i);       <span class="cm">// load 8 b's at once</span>
    acc = <span class="fn">_mm256_fmadd_ps</span>(va, vb, acc);       <span class="cm">// acc += va*vb, 8 lanes at once</span>
}
float sum = <span class="fn">hsum_float_8</span>(acc);               <span class="cm">// horizontal sum of 8 lanes -> scalar</span></pre>
<p>The SIMD (Single Instruction Multiple Data) on the right exists to squeeze out that idle compute. AVX2 offers 256-bit <span class="mono">__m256</span> registers, one holding exactly 8 floats; one <span class="mono">_mm256_fmadd_ps</span> (fused multiply-add) instruction makes these 8 <strong>lanes</strong> each do one <span class="mono">acc[i] += a[i]*b[i]</span> simultaneously. The loop stride goes from 1 to 8, cutting instruction count by seven-eighths. ARM's NEON is 128-bit, 4 floats at a time - the same idea at half the width.</p>
<p>Drawing this "8 lanes in parallel" is the clearest. Below we trace one SIMD dot product: 8 pairs multiply-add into 8 accumulators at once, and after a few rounds a single <strong>horizontal sum</strong> (hsum) folds the 8 accumulators into the final scalar.</p>
<div class="trace">
  <div class="tcap"><b>Tracing one SIMD dot product</b>: 8 floats packed in one 256-bit register, one fmadd instruction multiply-adds all 8 lanes, then a horizontal sum to one scalar (illustrative).</div>
<svg viewBox="0 0 640 250" width="100%" role="img" aria-label="SIMD dot product example: one instruction multiply-adds 8 lanes at once, then a horizontal sum to one scalar">
<g font-family="ui-monospace,monospace">
<text x="50" y="42" text-anchor="end" fill="#5b6470" font-size="11">vec a</text>
<text x="50" y="100" text-anchor="end" fill="#5b6470" font-size="11">vec b</text>
<text x="50" y="200" text-anchor="end" fill="#5b6470" font-size="11">acc</text>
<rect x="60" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="88" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a0</text>
<rect x="60" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="88" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b0</text>
<rect x="60" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="88" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s0</text>
<rect x="128" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="156" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a1</text>
<rect x="128" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="156" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b1</text>
<rect x="128" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="156" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s1</text>
<rect x="196" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="224" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a2</text>
<rect x="196" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="224" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b2</text>
<rect x="196" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="224" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s2</text>
<rect x="264" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="292" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a3</text>
<rect x="264" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="292" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b3</text>
<rect x="264" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="292" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s3</text>
<rect x="332" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="360" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a4</text>
<rect x="332" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="360" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b4</text>
<rect x="332" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="360" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s4</text>
<rect x="400" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="428" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a5</text>
<rect x="400" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="428" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b5</text>
<rect x="400" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="428" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s5</text>
<rect x="468" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="496" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a6</text>
<rect x="468" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="496" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b6</text>
<rect x="468" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="496" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s6</text>
<rect x="536" y="30" width="56" height="26" rx="4" fill="#ffffff" stroke="#c2630e"/><text x="564" y="48" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">a7</text>
<rect x="536" y="88" width="56" height="26" rx="4" fill="#ffffff" stroke="#2563eb"/><text x="564" y="106" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">b7</text>
<rect x="536" y="180" width="56" height="28" rx="4" fill="#c2630e" stroke="#c2630e"/><text x="564" y="199" text-anchor="middle" fill="#ffffff" font-weight="700" font-size="12">s7</text>
<rect x="60" y="128" width="532" height="34" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<text x="320" y="150" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="13">_mm256_fmadd_ps: one instruction, 8 lanes do s[i] += a[i]*b[i]</text>
<line x1="88" y1="114" x2="88" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 88 128 L 84 121 L 92 121 z" fill="#9aa6b2"/>
<line x1="88" y1="162" x2="88" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 88 180 L 84 173 L 92 173 z" fill="#9aa6b2"/>
<line x1="156" y1="114" x2="156" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 156 128 L 152 121 L 160 121 z" fill="#9aa6b2"/>
<line x1="156" y1="162" x2="156" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 156 180 L 152 173 L 160 173 z" fill="#9aa6b2"/>
<line x1="224" y1="114" x2="224" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 224 128 L 220 121 L 228 121 z" fill="#9aa6b2"/>
<line x1="224" y1="162" x2="224" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 224 180 L 220 173 L 228 173 z" fill="#9aa6b2"/>
<line x1="292" y1="114" x2="292" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 292 128 L 288 121 L 296 121 z" fill="#9aa6b2"/>
<line x1="292" y1="162" x2="292" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 292 180 L 288 173 L 296 173 z" fill="#9aa6b2"/>
<line x1="360" y1="114" x2="360" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 360 128 L 356 121 L 364 121 z" fill="#9aa6b2"/>
<line x1="360" y1="162" x2="360" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 360 180 L 356 173 L 364 173 z" fill="#9aa6b2"/>
<line x1="428" y1="114" x2="428" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 428 128 L 424 121 L 432 121 z" fill="#9aa6b2"/>
<line x1="428" y1="162" x2="428" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 428 180 L 424 173 L 432 173 z" fill="#9aa6b2"/>
<line x1="496" y1="114" x2="496" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 496 128 L 492 121 L 500 121 z" fill="#9aa6b2"/>
<line x1="496" y1="162" x2="496" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 496 180 L 492 173 L 500 173 z" fill="#9aa6b2"/>
<line x1="564" y1="114" x2="564" y2="121" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 564 128 L 560 121 L 568 121 z" fill="#9aa6b2"/>
<line x1="564" y1="162" x2="564" y2="173" stroke="#9aa6b2" stroke-width="1.4"/><path d="M 564 180 L 560 173 L 568 173 z" fill="#9aa6b2"/>
<text x="320" y="74" text-anchor="middle" fill="#5b6470" font-size="11">8 floats packed in one 256-bit register</text>
<line x1="88" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 311 231 L 311 224 z" fill="#7c3aed"/>
<line x1="156" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 311 231 L 311 224 z" fill="#7c3aed"/>
<line x1="224" y1="208" x2="311" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 310 230 L 312 223 z" fill="#7c3aed"/>
<line x1="292" y1="208" x2="312" y2="224" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 310 227 L 315 221 z" fill="#7c3aed"/>
<line x1="360" y1="208" x2="324" y2="225" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 323 222 L 326 228 z" fill="#7c3aed"/>
<line x1="428" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 324 223 L 326 230 z" fill="#7c3aed"/>
<line x1="496" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 325 224 L 325 231 z" fill="#7c3aed"/>
<line x1="564" y1="208" x2="325" y2="227" stroke="#7c3aed" stroke-width="1.4"/><path d="M 318 228 L 325 224 L 325 231 z" fill="#7c3aed"/>
<rect x="300" y="226" width="40" height="18" rx="4" fill="#ffffff" stroke="#7c3aed"/><text x="320" y="239" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="11">sum</text>
<text x="356" y="239" fill="#5b6470" font-size="11">horizontal sum (hsum) -> one scalar</text>
</g></svg>
</div>

<h2>How the quantized dot product works</h2>
<p>In real inference, weights are quantized (L29), so the dot product must <strong>unpack</strong> before computing. Take the most common <span class="mono">vec_dot_q4_0_q8_0</span>: weights are Q4_0 - grouped 32-at-a-time into 16 bytes of 4-bit values (<span class="mono">block_q4_0</span> = one fp16 <span class="mono">d</span> (scale) + <span class="mono">qs[16]</span>), activations are Q8_0 int8. So the inner of one quantized dot product is this chain: unpack 4-bit -> subtract 8 offset -> multiply int8 activation -> multiply back the scale -> accumulate. Below is its real AVX2 implementation, line by line.</p>
<pre class="code"><span class="cm">// real AVX2 quantized dot product core (arch/x86/quants.c vec_dot_q4_0_q8_0)</span>
__m256 acc = <span class="fn">_mm256_setzero_ps</span>();
<span class="kw">for</span> (; ib &lt; nb; ++ib) {                               <span class="cm">// loop over blocks (32 weights each)</span>
    __m256  d  = <span class="fn">_mm256_set1_ps</span>(dx * dy);            <span class="cm">// combine the two blocks' fp16 scale</span>
    __m256i qx = <span class="fn">bytes_from_nibbles_32</span>(x[ib].qs);    <span class="cm">// unpack: 16 bytes -> 32 values [0..15]</span>
    qx = <span class="fn">_mm256_sub_epi8</span>(qx, <span class="fn">_mm256_set1_epi8</span>(8));   <span class="cm">// offset to [-8..+7]</span>
    __m256i qy = <span class="fn">_mm256_loadu_si256</span>((const __m256i*)y[ib].qs); <span class="cm">// load 32 int8 activations</span>
    __m256  q  = <span class="fn">mul_sum_i8_pairs_float</span>(qx, qy);     <span class="cm">// int8 dot product -> float</span>
    acc = <span class="fn">_mm256_fmadd_ps</span>(d, q, acc);                <span class="cm">// FMA: acc += d * q</span>
}
float sumf = <span class="fn">hsum_float_8</span>(acc);                      <span class="cm">// horizontal sum of 8 lanes -> scalar</span></pre>
<p>Line by line: <span class="mono">acc</span> is an 8-lane float accumulator; each loop iteration processes one block - <span class="mono">bytes_from_nibbles_32</span> unpacks 16 bytes into 32 values in [0..15] (mask the low 4 bits with <span class="mono">0xF</span>, shift for the high 4); subtract 8 to offset into [-8..+7] (4-bit quants are signed); <span class="mono">_mm256_loadu_si256</span> loads 32 int8 activations at once; <span class="mono">mul_sum_i8_pairs_float</span> does the int8 dot product into 8 floats; finally <span class="mono">_mm256_fmadd_ps</span> multiplies by the combined scale and accumulates into <span class="mono">acc</span>. After all blocks, <span class="mono">hsum_float_8</span> horizontally sums the 8 lanes into the final scalar. This "unpack + vectorized multiply-add" is exactly what lets a quantized model run on CPU. One easy-to-miss point: why are activations Q8_0 (int8) and weights Q4_0 (4-bit), different precisions on the two sides? Because their roles differ - weights are fixed and bulky, so squeezing to 4-bit saves the most memory; activations are live with a dynamic range, so int8 keeps them stable. And at the hardware level, int8 multiply-add has dedicated fast instructions (like <span class="mono">_mm256_maddubs_epi16</span>), even faster than a pure-float dot product. So "weights 4-bit, activations int8" both saves memory and runs fast, the mainstream of community quantization - and the very reason this kernel unpacks first, then does int8.</p>
<p>Pulling one block's flow out and freezing it makes each step clearer:</p>
<div class="trace">
  <div class="tcap"><b>Tracing one block</b>: 32 weights from 16 bytes of 4-bit, unpacked, offset, dotted with activations, scaled and accumulated (illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>(1) packed</h5>
      <div class="cellrow"><span class="vc">qs[16]</span></div>
      <div class="tlab">32 4-bit weights</div></div>
    <div class="op">unpack<br>nibbles</div>
    <div class="stn"><h5>(2) unpack</h5>
      <div class="cellrow"><span class="vc">32 of [0..15]</span></div>
      <div class="tlab">low/high 4 bits</div></div>
    <div class="op">-8<br>offset</div>
    <div class="stn"><h5>(3) offset</h5>
      <div class="cellrow"><span class="vc">[-8..+7]</span></div>
      <div class="tlab">signed quants</div></div>
    <div class="op">dot<br>int8</div>
    <div class="stn"><h5>(4) dot</h5>
      <div class="cellrow"><span class="vc blue">dot(qx, qy)</span></div>
      <div class="tlab">with Q8_0 acts</div></div>
    <div class="op">x scale<br>accum</div>
    <div class="stn"><h5>(5) accumulate</h5>
      <div class="cellrow"><span class="vc hot">acc += d*dot</span></div>
      <div class="tlab">x the fp16 scale</div></div>
  </div>
</div>

<h2>Multithreading: many cores together</h2>
<p>SIMD squeezes a single core; multithreading puts all cores to work. One big matmul has many output rows, and each row's computation is <strong>independent</strong> (row 5 does not need row 3's result), so it is naturally parallel: split the rows evenly among N threads, each computes its batch, then merge. This "no dependencies, split however you like" structure is the ideal target for data parallelism. Hidden here is a plain but important test: whether you can parallelize comes down to "are there dependencies". Rows have no ordering between them, so you can split freely; once there is a dependency (a later step waits on an earlier result), parallelism needs synchronization and waiting, and the payoff drops immediately. When we look at the GPU in L32, you will find the very same test at work.</p>
<div class="cols">
  <div class="col"><h4>thread 0</h4><p>computes output rows 0..k (each row using a SIMD dot product inside).</p></div>
  <div class="col"><h4>thread 1</h4><p>computes rows k..2k, at the same time as thread 0, no waiting.</p></div>
  <div class="col"><h4>thread 2 / 3 / ...</h4><p>each takes its batch of rows; rows are independent, split into any number.</p></div>
</div>
<p>ggml's implementation is lightweight: each op <span class="mono">ggml_compute_forward_*</span> (<span class="mono">ggml-cpu.c</span>) gets <span class="mono">params-&gt;ith</span> (which thread am I) and <span class="mono">params-&gt;nth</span> (how many threads), and from them works out "which rows/blocks I own", computing on its own. The thread pool lives in <span class="mono">ggml-threading</span>, avoiding the cost of creating and destroying threads over and over. <strong>SIMD (8 at once within a core) x multithreading (many cores at once)</strong> stacked together is the entire source of the CPU backend's throughput - no magic, just spreading the same work out to be done as simultaneously as possible. Worth noting: not every op splits as nicely as matmul. Reduction ops like softmax and RMSNorm need to "see the whole row before computing", so splitting them needs care at the boundaries; elementwise ops (add, activations) split as freely as matmul. ggml writes per-op splitting logic exactly to handle these differences. Reading the source, you will see almost every <span class="mono">ggml_compute_forward_*</span> open by using <span class="mono">ith/nth</span> to work out its own range - that is what multithreading looks like landing at the op level.</p>

<h2>Stacking the three: from dot product to matmul</h2>
<p>We have seen SIMD (8 lanes within a core) and multithreading (many cores at once), and met tiling (reuse cached data, detailed in the fold below). But in one real matmul, the three are not independent - they <strong>nest layer upon layer</strong> and act together. See this relationship clearly and you truly understand "where the CPU backend's speed comes from".</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">outer</span><span class="name">multithreading (across cores)</span></div><div class="ld">split output rows among N threads, each core owns a batch, no dependencies</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">middle</span><span class="name">tiling (cache)</span></div><div class="ld">each thread cuts its batch into tiles, computing tile by tile, reusing cached data</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">inner</span><span class="name">SIMD (within a core)</span></div><div class="ld">computing each tile's dot product, one fmadd does 8 lanes at once</div></div>
</div>
<p>Stacked, the effect is <strong>multiplicative</strong>: multithreading spreads work over all cores, tiling keeps each core from waiting on memory, and SIMD turns each core's "arm" into an 8-head one. Drop any layer and the whole thing slows - only multithreading without vectorizing leaves single cores crawling; only vectorizing without tiling leaves data shuttling endlessly between memory and cache. Because all three are on, ggml can run a multi-GB quantized model respectably on a pure-CPU machine with no GPU at all.</p>
<p>This also explains the tuning lore: thread count is usually best near the physical core count (more just fights for resources); and switching the build from no-SIMD to AVX2 or even AVX-512 often doubles speed at a stroke - now you know it is because the "inner" arm went from 1-at-a-time to 8 or 16. Low-level kernels look arcane, but the rule is concrete: whichever layer is not full, fill it; to find which layer, go back to L30's two rulers and measure.</p>

<h2>Deep dive: cache and many architectures</h2>
<p>Two final folds for two easy-to-overlook engineering details that truly make the CPU backend fast: what tiling does for the cache, and how one codebase fits all kinds of CPU.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> Why is tiling faster than the naive triple loop? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>A naive matmul is a triple loop, and computing each output element rescans a whole row of A and a whole column of B from memory. Once the matrix is large, that data does not fit in the CPU's fast cache, so it is fetched over and over from main memory dozens of times slower - the bottleneck is not "computing" but "waiting for data". <span class="mono">llamafile/sgemm.cpp</span> and <span class="mono">repack.cpp</span> use <strong>tiling</strong>: cut the big matrix into small tiles that just fit in cache, load one tile, do all the computation it can take part in, then move to the next. Same number of multiply-adds, but each loaded datum is <strong>reused thoroughly</strong>, slashing memory traffic. This echoes L30: many ops are "memory-bound", and saving memory traffic saves time. Tiling is the common trick of nearly all high-performance matmuls (CPU BLAS, GPU mmq in L32). A quick word on tiling's partner <span class="mono">repack</span>: at weight-load time it rearranges the data into a layout "friendlier to cache and SIMD", so runtime fetches flow better and vectorization lines up neatly. This is a classic "preprocess to buy runtime speed" - spend a little more load time to make every one of countless later inferences faster, the same idea as L29's imatrix (measure weight importance before inference): do ahead of time whatever can be done ahead of time.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> How does one codebase fit all kinds of CPU? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>The same <span class="mono">vec_dot_q4_0_q8_0</span> wants AVX2 on x86, NEON on ARM, and a scalar fallback on old CPUs - how does one source cover them all? ggml puts arch-specific implementations under <span class="mono">ggml-cpu/arch/{x86,arm,riscv,...}/</span> and dispatches in two layers, <strong>compile-time + runtime</strong>: compile-time macros like <span class="mono">#if defined(__AVX2__)</span> compile in only what the current architecture supports; at runtime it then detects whether the CPU actually has a given instruction set (feature detection) and picks the fastest available. So the same binary you downloaded automatically uses AVX-512 on a new CPU and safely falls back to scalar on an old machine, without crashing for using an advanced instruction. "One codebase, optimal per architecture" is exactly what lets ggml run across such a motley range of devices. This two-layer "compile-time pruning + runtime selection" dispatch is in fact the standard approach of cross-platform high-performance libraries. The cost is that the source is full of <span class="mono">#if</span> and the same function has several arch-specialized versions, a branchy read; but in return you get "compile once, optimal everywhere". So when you read the <span class="mono">arch/</span> directory you need not chew through every architecture - grasp the x86 branch clearly, and the rest are just the same idea with a different set of intrinsic names.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>CPU backend, two speedups: <strong>SIMD</strong> (data parallel, one instruction per row) + <strong>multithreading</strong> (task parallel, many cores at once).</li>
    <li>SIMD: AVX2's <span class="mono">__m256</span> holds 8 floats, <span class="mono">_mm256_fmadd_ps</span> does 8 multiply-adds in one instruction; NEON does 4. A final <span class="mono">hsum</span> folds lanes into a scalar.</li>
    <li>Quantized dot <span class="mono">vec_dot_q4_0_q8_0</span>: <span class="mono">block_q4_0</span> shares one fp16 scale per 32 weights; inner = unpack 4-bit -&gt; offset -&gt; int8 dot -&gt; <span class="mono">fmadd</span> scale-accumulate.</li>
    <li>Multithreading: ops use <span class="mono">ith/nth</span> to split output rows across threads (rows independent, naturally parallel); thread pool in <span class="mono">ggml-threading</span>.</li>
    <li>Going faster: tiling improves cache reuse; <span class="mono">arch/</span> uses compile-time + runtime dispatch for "one codebase, optimal per architecture".</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  What is most worth taking from this lesson is not the name of some intrinsic, but a way of seeing performance: <strong>given the same amount of computation, how do you "spread it out" to run simultaneously, and how do you make data travel less</strong>. SIMD is "turn one-at-a-time into a-row-at-once", multithreading is "spread one job over many cores", tiling is "let cached data be reused a few more times" - none changes the total work, only "how" it is done. This is the central theme of high-performance computing, and the next lesson's GPU (CUDA) pushes the same ideas to the extreme: thousands of threads computing at once, on-chip memory reusing data. So do not be intimidated by the screen of <span class="mono">_mm256_*</span> - see through them, and what you see is one plain, powerful idea: keep all the idle hardware busy. Follow this idea and you can even predict what optimizations look like: facing a slow loop, first ask "can it compute several at once?" (SIMD), "can it split across cores?" (multithreading), "is data fetched from afar over and over?" (cache / tiling). These three questions cover most of CPU performance tuning. Behind them is one plain economics: the hardware's compute and bandwidth are a fixed cost you already paid for when buying the machine, and leaving it idle is pure waste; "high performance" is just contriving to use up that already-paid cost. Next lesson, the GPU pushes this "fill the hardware" philosophy to thousands of threads - but the kernel mindset is exactly this lesson's.
</div>
""",
}
