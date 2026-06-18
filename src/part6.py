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

LESSON_32 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
上一课看了 CPU 后端怎么靠 SIMD + 多线程把矩阵乘算快（L31）。可真正让大模型推理"飞"起来的，往往是 GPU——它不靠几个强核取胜，而靠<strong>成千上万个线程同时算</strong>。这一课钻进 ggml 的 CUDA 后端（<span class="mono">ggml/src/ggml-cuda/</span>），看它怎么把这么多线程组织起来：一次矩阵乘是怎么拆成 block 和 thread 的协作，又怎么靠片上的高速内存把它们喂饱。
</p>
<p style="color:var(--muted);margin-top:.4rem">单看一个 GPU 线程，它弱得可怜——主频比 CPU 还低，一次只算一格。GPU 的强，全在"数量"：几千个线程一起上，吞吐就压倒性地高。所以 GPU 内核的全部手艺，归根到底就两件事：把活儿切成成千上万个互不依赖的小块（让线程都有事干），再想办法让数据离线程足够近（别让它们饿着等显存）。这一课会读一段真实的 CUDA kernel 骨架，再看这一课的主角——分块矩阵乘，然后是显存层级，最后用概念的方式看一眼 flash attention。</p>
<p style="color:var(--muted)">路线图：先认识 CUDA 的执行模型（grid / block / thread / warp），再看分块矩阵乘怎么用片上 shared memory 复用数据，接着理解显存的三级层级与"搬数据比算还贵"，最后看 flash attention 为什么要把多个算子融合成一个 kernel。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  GPU 的性能秘诀，和 CPU（L31）是同一套母题、只是推到极致：<strong>海量并行</strong>（成千上万个线程同时算，而不是几个核）+ <strong>喂得饱</strong>（用片上高速内存复用数据，别让线程干等显存）。CPU 那只"8 头机械臂"（SIMD）在 GPU 上变成了一整座体育场的几千只手；但手越多，"备料"的压力越大——这就是为什么 GPU 内核里最关键的代码，往往不是"怎么算"，而是"怎么把数据搬进那块小而快的 shared memory、让一个 block 里的线程反复复用"。抓住"并行铺满 + 数据搬近"这两件事，满屏的 <span class="mono">blockIdx</span>、<span class="mono">__shared__</span>、<span class="mono">__syncthreads</span> 就都有了主线。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  上一课的 CPU 像几位<strong>老师傅</strong>，每人配一只 8 头机械臂，活儿精、人少；GPU 则像一座<strong>体育场里的几千名普通工人</strong>。单个工人慢，但几千人同时开工，总吞吐惊人——前提是你能把活儿拆成几千份谁也不等谁的小任务，还得喂得上料。每个班组（block）面前有一张<strong>小而快的工作台</strong>（shared memory），但料仓（global 显存）在很远的地方。聪明的做法是：班组先合力把一批料从料仓搬上工作台，然后大家在工作台上反复取用、互不打扰，算完再搬下一批。GPU 内核的全部技巧，几乎都在"怎么用好这张小工作台"上——这正是下面分块矩阵乘要讲的。
</div>

<h2>CUDA 执行模型：grid / block / thread</h2>
<p>CUDA 把成千上万个线程组织成三层：一次 kernel 启动是一个 <strong>grid（网格）</strong>，grid 切成很多 <strong>block（线程块）</strong>，每个 block 里又有几十到上千个 <strong>thread（线程）</strong>。还有一个绕不开的概念 <strong>warp</strong>：每 32 个线程编成一组、步调完全一致地执行同一条指令（这套模型叫 SIMT，单指令多线程）。同一个 block 里的线程能共享一块片上的 <span class="mono">__shared__</span> 内存、还能用 <span class="mono">__syncthreads()</span> 互相等一等；不同 block 之间则基本各管各的、不保证谁先谁后。</p>
<p>先看一个最简单的真实 kernel：把每个元素乘上一个 scale。它正好展示了 GPU 最基本的写法——<strong>先算出"我是谁"，再只算我那一份</strong>：</p>
<pre class="code"><span class="cm">// 真实 CUDA kernel: 每个线程算输出的一个/几个元素 (简化自 ggml-cuda/scale.cu)</span>
<span class="kw">__global__</span> void <span class="fn">scale_f32</span>(const float* x, float* dst, float scale, int64_t n) {
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;  <span class="cm">// 全局唯一线程号</span>
    int64_t stride = blockDim.x * gridDim.x;                 <span class="cm">// 一轮覆盖的线程总数</span>
    <span class="kw">for</span> (int64_t i = tid; i &lt; n; i += stride)             <span class="cm">// grid-stride 循环</span>
        dst[i] = scale * x[i];                              <span class="cm">// 各线程各算各的, 无依赖</span>
}
<span class="cm">// 启动: 指定 grid 和 block 大小</span>
<span class="fn">scale_f32</span>&lt;&lt;&lt;num_blocks, 256&gt;&gt;&gt;(x, dst, scale, n);   <span class="cm">// 256 = 每个 block 的线程数</span></pre>
<p>逐行看：<span class="mono">blockIdx.x * blockDim.x + threadIdx.x</span> 把"第几个 block、block 里第几个线程"换算成一个全局唯一的线程号 <span class="mono">tid</span>——这几乎是每个 CUDA kernel 的第一行。<span class="mono">stride</span> 是一轮里所有线程加起来能覆盖的元素数；当数据比线程还多时，用这个 <strong>grid-stride 循环</strong>让每个线程隔 <span class="mono">stride</span> 跳着多算几个。最关键的一点：所有线程跑的是<strong>同一份代码</strong>，只靠各自不同的 <span class="mono">tid</span> 落到不同的数据上——这就是 SIMT。启动时那串 <span class="mono">&lt;&lt;&lt;num_blocks, 256&gt;&gt;&gt;</span> 三尖括号语法，指定这次要起多少个 block、每个 block 多少线程。还有一点容易忽略却很关键：GPU 起的线程数往往<strong>远超</strong>它的物理核数（动辄几万），这不是浪费，而是 GPU 藏延迟的看家本领。当一批线程卡在等显存时，硬件立刻切去跑另一批就绪的线程，让计算单元一刻不闲。所以"线程多到用不完"恰恰是好事：它把访存的等待时间，用别的线程的计算填满了——这也解释了为什么 GPU 偏爱"小任务、海量并行"，而不像 CPU 那样靠少数强核。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">grid</span><span class="name">网格</span></div><div class="ld">一次 kernel 启动 = 一个 grid, 含成百上千个 block</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">block</span><span class="name">线程块</span></div><div class="ld">含几十~上千个线程, 共享一块片上 shared memory, 能用 __syncthreads 互等</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">thread</span><span class="name">线程</span></div><div class="ld">最小执行单位, 算自己那一格; 每 32 个线程组成一个 warp, 步调一致</div></div>
</div>

<h2>分块矩阵乘：GPU 内核的心脏</h2>
<p>矩阵乘是大模型里最重的算子，GPU 上把它写快的关键又是 tiling（分块）——和 L31 的 CPU 分块同一个思路，但这里分进的是 <strong>shared memory</strong>。先想想朴素写法的毛病：让每个线程独立去算 C 的一格，它得把 A 的一整行、B 的一整列从 global 显存（很慢）读一遍。相邻线程读的数据大量重叠，却各读各的，把本就紧张的显存带宽白白浪费掉——瓶颈又是"等数据"，不是"算"。举个直观的数：朴素版算一个 N x N 的输出，要从慢显存里读约 N 的三次方 量级的数据（每算一格都重新扫一整行 A 和一整列 B，相邻线程之间大量重复读）；机器真正擅长的乘加明明很快，结果大把时间都耗在了"等数据"上。这正是 tiling 要解决的核心矛盾。</p>
<p>分块的解法是让一个 block 的线程<strong>合作</strong>：先一起把 A 的一小块、B 的一小块从 global 搬进 block 共享的 <span class="mono">__shared__</span>（片上、快一两个数量级），用 <span class="mono">__syncthreads()</span> 等大家都搬完，然后 block 内每个线程都在这块快内存上反复取数、算自己负责的那一格，沿 K 维一块块推进、累加。下面是规范化的教学骨架（真实的 <span class="mono">mmq.cuh</span> 在此之上还加了 4-bit 解包和大量模板，但骨架就是这个）：</p>
<pre class="code"><span class="cm">// 教学版分块矩阵乘骨架 (真实的 ggml-cuda/mmq.cuh 在此之上加了 4-bit 解包和模板)</span>
<span class="kw">__global__</span> void <span class="fn">matmul_tiled</span>(const float* A, const float* B, float* C, int N) {
    <span class="kw">__shared__</span> float As[T][T];                 <span class="cm">// 片上共享: A 的一块</span>
    <span class="kw">__shared__</span> float Bs[T][T];                 <span class="cm">// 片上共享: B 的一块</span>
    int row = blockIdx.y*T + threadIdx.y;
    int col = blockIdx.x*T + threadIdx.x;
    float acc = 0.0f;                            <span class="cm">// 累加器在寄存器里(最快)</span>
    <span class="kw">for</span> (int k0 = 0; k0 &lt; N; k0 += T) {         <span class="cm">// 沿 K 维一块块推进</span>
        As[threadIdx.y][threadIdx.x] = A[row*N + (k0+threadIdx.x)]; <span class="cm">// 协作载入</span>
        Bs[threadIdx.y][threadIdx.x] = B[(k0+threadIdx.y)*N + col];
        <span class="fn">__syncthreads</span>();                       <span class="cm">// 等整块都载完再算</span>
        <span class="kw">for</span> (int k = 0; k &lt; T; ++k)             <span class="cm">// 块内复用, 全在快内存里</span>
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        <span class="fn">__syncthreads</span>();                       <span class="cm">// 等大家算完再覆盖下一块</span>
    }
    C[row*N + col] = acc;                        <span class="cm">// 一格只写回一次</span>
}</pre>
<p>关键就在那两个 <span class="mono">__syncthreads()</span>：第一个保证"整块都搬上工作台了"才开始算，第二个保证"大家都算完了"才去覆盖下一块——少了任何一个，就会有线程读到半截数据，结果全错。而一块 tile 一旦搬进 shared memory，就被 block 里所有线程<strong>反复复用</strong>很多次，把慢的 global 访问摊薄到极低。这正是把上一课"载入一次、复用多次"的 tiling 思想搬到了 GPU 的片上内存上：CPU 复用的是 cache，GPU 复用的是 shared memory，但省访存的算盘一模一样。粗算一下复用的威力：一块 T x T 的 tile 一旦载入 shared，块内线程会把它当行/列反复用上约 T 次——也就是每个从 global 搬来的数平均被复用约 T 倍；T 取 32，慢显存的访问量就直接降到约三十分之一。（代码里的 T 是 tile 的边长，常取 16 或 32；blockDim 也设成 T 行 T 列，正好一个线程管一格。）</p>
<p>把这个过程定格成一张图最直观——一个 block 把 A 的行块、B 的列块载入 shared，块内每个线程负责输出 C 的一格：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次分块矩阵乘</b>：一个 thread block 把 A 的行块、B 的列块载入 shared memory，块内每个线程算 C 的一格（As 的一行与 Bs 的一列做点积），沿 K 维循环累加（示意）。</div>
<svg viewBox="0 18 660 372" width="100%" role="img" aria-label="分块矩阵乘示例：一个 thread block 把 A 的行块、B 的列块载入共享内存，块内每个线程算输出 C 的一格">
<g font-family="ui-monospace,monospace">
<rect x="117" y="189" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="120" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="120" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="148" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="176" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="204" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="120" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="120" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="377" y="49" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="380" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="52" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="80" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="108" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="136" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="377" y="189" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="380" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="220" width="28" height="28" fill="#ece3fb" stroke="#7c3aed"/>
<rect x="436" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<text x="176" y="180" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">As（共享内存）</text>
<text x="436" y="40" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">Bs（共享内存）</text>
<text x="436" y="324" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="12">C 输出块</text>
<text x="110" y="238" text-anchor="end" fill="#c2630e" font-weight="700" font-size="11">行 i</text>
<text x="432" y="182" fill="#2563eb" font-weight="700" font-size="11">列 j</text>
<line x1="238" y1="234" x2="368" y2="234" stroke="#c2630e" stroke-width="1.6"/>
<path d="M 376 234 L 368 230 L 368 238 z" fill="#c2630e"/>
<line x1="422" y1="170" x2="422" y2="180" stroke="#2563eb" stroke-width="1.6"/>
<path d="M 422 188 L 418 180 L 426 180 z" fill="#2563eb"/>
<text x="504" y="238" fill="#7c3aed" font-weight="700" font-size="11">C[i][j]</text>
<line x1="500" y1="234" x2="438" y2="234" stroke="#7c3aed" stroke-width="1.2"/>
<rect x="60" y="332" width="540" height="44" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<text x="330" y="350" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="12">thread(i,j): C[i][j] += As[i][k] * Bs[k][j]</text>
<text x="330" y="368" text-anchor="middle" fill="#5b6470" font-size="11">循环 k0 = 0..K 步长 T：载入下一块 -> __syncthreads -> 累加 -> __syncthreads</text>
</g></svg>
</div>

<h2>显存层级：为什么"搬数据"比"算"还贵</h2>
<p>上面反复说"快内存""慢显存"，这里把它讲清楚。GPU 有三级内存，越往下越小、越快：<strong>global（显存 VRAM）</strong>有几 GB、所有线程都能访问，但延迟高、带宽常常就是整个内核的瓶颈；<strong>shared（片上共享内存）</strong>每个 block 只有几十 KB、只给 block 内线程共享，比 global 快一两个数量级；<strong>register（寄存器）</strong>每个线程私有、最快，分块矩阵乘里的累加器 <span class="mono">acc</span> 就住在这里。分块矩阵乘的全部意义，就是把数据尽量从 global 挪到 shared 和 register 上反复用。</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">大·慢</span><span class="name">global / 显存 VRAM</span></div><div class="ld">几 GB, 所有线程可见; 延迟高、带宽是瓶颈 (呼应 L30 的访存密集)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">中·快</span><span class="name">shared / 片上</span></div><div class="ld">每个 block 几十 KB, block 内共享; 比 global 快一两个数量级</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">小·最快</span><span class="name">register / 寄存器</span></div><div class="ld">每个线程私有, 最快; 累加器 acc 就放这里</div></div>
</div>
<p>这又一次印证了 L30 的观察：很多内核是"访存密集"而非"算力密集"的——GPU 的算力多到常常用不完，真正卡住它的，是把数据从显存喂进来的那点带宽。所以 GPU 优化的大半功夫，都花在"减少 global 访问、增加 shared/register 复用"上。看懂这一条，你就明白了为什么同一块卡上，一个写得好的 kernel 能比朴素版快好几倍——要算的总量没变，省下的全是搬运。这也是为什么 batch（一次喂多条序列/多 token）几乎总能提高 GPU 利用率：数据搬进来一次，多算几遍，把那条带宽用得更值。再补一个 GPU 特有的讲究：访问 global 显存时，一个 warp 里 32 个线程最好去读<strong>连续相邻</strong>的地址，硬件能把它们合并成一次大的内存事务（coalesced，合并访问）；要是各读各的乱地址，就退化成几十次小事务，带宽利用率骤降。所以 GPU 内核不仅要"少读 global"，还要"读得整齐"——这也是为什么张量在显存里的摆放顺序（layout）对性能影响很大，很多内核会专门重排数据来迁就这条规则。</p>

<h2>flash attention：把多个算子融合成一个 kernel</h2>
<p>注意力（attention）要算 <span class="mono">softmax(Q @ K^T) @ V</span>。朴素做法会先把巨大的 N x N 注意力分数矩阵整个算出来、写回显存，再读回来做 softmax、再读回来乘 V——光是把这个 N x N 大矩阵在显存里写一遍、读三遍，就把带宽吃干了（还得额外占一大块显存）。<strong>flash attention</strong> 的思路是把这几步<strong>融合（fuse）</strong>成一个 kernel：分块地算，一边算一边用"在线 softmax"修正，<strong>永不把整个 N x N 分数矩阵物化到显存</strong>，只把最终输出 O 写出去。</p>
<pre class="code"><span class="cm">// flash attention 思路 (概念伪代码, 真实实现见 ggml-cuda/fattn*.cu)</span>
acc = 0;  run_max = -inf;  run_sum = 0;
<span class="kw">for</span> each K/V tile:                  <span class="cm">// 分块, 永不物化整个 N x N 分数</span>
    s   = Q @ K_tile^T               <span class="cm">// 只算当前块的注意力分数</span>
    m   = max(run_max, rowmax(s))    <span class="cm">// 更新到目前为止的行最大值</span>
    p   = exp(s - m)                 <span class="cm">// 在线 softmax: 边算边修正</span>
    run_sum = run_sum * exp(run_max - m) + rowsum(p)
    acc     = acc     * exp(run_max - m) + p @ V_tile
    run_max = m
O = acc / run_sum                    <span class="cm">// 最后归一; 只写出 O, 不写 N x N</span></pre>
<p>诀窍在那个"在线 softmax"：softmax 本该先看到一整行才能定下分母，但 flash attention 用一个<strong>跑动的最大值和跑动的分母</strong>（<span class="mono">run_max</span> / <span class="mono">run_sum</span>），每来一个新块就把已累计的结果按新最大值重新缩放一下，于是不必等齐整行、也能分块往前算。代价是多了几次缩放运算，换来的是<strong>再也不用把那个 N x N 大矩阵搬进搬出显存</strong>——序列越长，省得越多。这就是"算子融合"的威力：把本该分几个 kernel、来回读写显存的活儿，并进一个 kernel 里一气呵成，省下的全是最贵的那部分——访存。这一点对长上下文尤其要命：序列长度翻倍，那个 N x N 矩阵的面积就翻四倍，朴素做法的显存和带宽开销爆炸式增长，而 flash attention 因为根本不物化它，开销只随序列长度线性上升——这正是如今动辄几万、几十万 token 的长上下文能跑起来的关键之一。</p>
<div class="card detail">
  <div class="tag">🔬 范围与源码</div>
  这一节只讲 flash attention 的<strong>思路</strong>，不逐行抠真实 kernel。原因是 ggml 的 <span class="mono">fattn-*</span> 系列文件（<span class="mono">fattn-tile</span> / <span class="mono">fattn-mma-f16</span> / <span class="mono">fattn-vec</span> 等）为了在不同 GPU、不同精度、不同 head 维度上都跑到最快，写了一大堆模板特化，行数多、枝杈密。但万变不离其宗：它们做的都是上面这套"分块 + 在线 softmax + 不物化 N x N"。先把思路记牢，真要读源码时就不会迷路。
</div>

<h2>深入：warp 内求和与内核变体</h2>
<p>最后两个折叠，补两个 GPU 内核里很常见、却容易看懵的细节：warp 怎么不靠 shared 就快速求和，以及为什么同一个算子会有那么多份 kernel。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> warp 内 32 个线程怎么不用 shared memory 就求和？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>很多内核（比如 softmax、norm、点积收尾）最后要把一组线程各自的部分和加成一个总和。如果走 shared memory，要写、要 <span class="mono">__syncthreads</span>、再读，比较费。而同一个 warp 里的 32 个线程本就步调一致，CUDA 提供了 <strong>warp shuffle</strong> 指令，让它们<strong>直接读彼此的寄存器</strong>，不碰任何内存。下面是 ggml 真实的 <span class="mono">warp_reduce_sum</span>（<span class="mono">ggml-cuda/common.cuh</span>）：</p>
<pre class="code"><span class="cm">// warp 内 32 线程求和, 不用 shared memory (来自 ggml-cuda/common.cuh)</span>
<span class="kw">__device__</span> float <span class="fn">warp_reduce_sum</span>(float x) {
    <span class="kw">for</span> (int offset = 16; offset &gt; 0; offset &gt;&gt;= 1)   <span class="cm">// 16 -> 8 -> 4 -> 2 -> 1</span>
        x += <span class="fn">__shfl_xor_sync</span>(0xffffffff, x, offset, 32); <span class="cm">// 直接读"距离 offset"的邻居</span>
    <span class="kw">return</span> x;                                          <span class="cm">// 结束后每个线程都拿到总和</span>
}</pre>
    <p>这是一棵<strong>蝶式（butterfly）归约树</strong>：第一轮每个线程和"隔 16"的伙伴交换并相加，第二轮隔 8、再隔 4、2、1，五轮（log2 32）之后，每个线程手里都是全 32 个值的总和。<span class="mono">__shfl_xor_sync</span> 的第一个参数 <span class="mono">0xffffffff</span> 是参与线程的掩码（全 32 个都参与），它让一个线程直接拿到另一个线程寄存器里的 <span class="mono">x</span>，完全不经过 shared 或 global。这就是为什么 warp 级归约又快又省——它把"线程间通信"压到了寄存器层面。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 为什么同一个矩阵乘有那么多份 kernel？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>翻 <span class="mono">ggml-cuda/</span> 你会看到一个算子常有好几份实现：<span class="mono">mmq</span>（量化矩阵乘）、<span class="mono">mmvq</span>（矩阵 x 向量，解码单 token 时用），还有针对不同量化格式（Q4_0/Q4_K/...）、不同 GPU 计算能力（compute capability）的特化版。为什么不写一份通用的？因为 GPU 性能对这些条件<strong>极其敏感</strong>：解码时是"瘦高"的矩阵 x 向量、prefill 时是"胖"的矩阵 x 矩阵（呼应 L18/L30 的 pp vs tg），最优的分块大小、用不用 tensor core、寄存器怎么分配都不一样。与其用一份折中代码处处平庸，不如为每种重要情形挑一份最快的——这跟 L31 里 CPU 的 <span class="mono">arch/</span> 多架构分派是同一种工程取舍：<strong>用代码量换性能</strong>。运行时再根据张量形状、量化类型、GPU 型号，挑一条最合适的 kernel 走。</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>CUDA 三层：<span class="mono">grid -&gt; block -&gt; thread</span>；每 32 线程一个 <strong>warp</strong> 步调一致；同一 block 内可用 <span class="mono">__shared__</span> + <span class="mono">__syncthreads</span>。</li>
    <li>每个线程算自己那份：<span class="mono">tid = blockIdx.x*blockDim.x + threadIdx.x</span>，数据比线程多时用 <strong>grid-stride 循环</strong>。</li>
    <li>分块矩阵乘是心脏：block 协作把 A/B 小块搬进 <span class="mono">__shared__</span>，<span class="mono">__syncthreads</span> 后反复复用、沿 K 累加；真实的 <span class="mono">mmq.cuh</span> 再加 4-bit 解包。</li>
    <li>显存三级 <span class="mono">global / shared / register</span>（大慢 -&gt; 小快 -&gt; 最快）；瓶颈常是带宽，优化 = 减少 global 访问、增加 shared 复用（呼应 L30）。</li>
    <li>flash attention：把 softmax + matmul 融合成一个 kernel，永不物化 N x N 分数，省显存与带宽；真实实现在 <span class="mono">fattn*.cu</span>。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  GPU 和 CPU 看似天差地别，内核思路却是<strong>同一个</strong>：同样的运算量，怎么铺开同时做、怎么让数据少跑路。CPU 把它推到几个核 × 8 路 SIMD；GPU 把它推到几千个线程 × 片上复用。所以你在 L31 学的那两问——"能不能一次多算几个？""数据是不是反复在远处取？"——到了 GPU 上原样还能用，只是答案换成了"几千个线程"和"shared memory"。看穿 <span class="mono">blockIdx</span>、<span class="mono">__shared__</span>、<span class="mono">__syncthreads</span> 这层语法，底下还是那个朴素的念头：让海量硬件都忙起来、别让它们饿着。这也是为什么"算子融合"（flash attention）和"分块复用"（tiling）会成为 GPU 优化的两大母题——它们都在和那条又慢又窄的显存带宽较劲。下一课我们退一步，看 ggml 怎么用一个统一的"后端"抽象，把 CPU、CUDA 以及更多后端管起来，并把一张计算图的算子分派到合适的硬件上。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
Last lesson watched the CPU backend make matmul fast with SIMD + multithreading (L31). But what really makes large-model inference "fly" is usually the GPU - it wins not with a few strong cores but with <strong>thousands of threads computing at once</strong>. This lesson drops into ggml's CUDA backend (<span class="mono">ggml/src/ggml-cuda/</span>) to see how it organizes all those threads: how one matmul is split into block-and-thread cooperation, and how on-chip fast memory keeps them fed.
</p>
<p style="color:var(--muted);margin-top:.4rem">A single GPU thread is feeble - lower clock than a CPU, computing one cell at a time. The GPU's strength is all in "numbers": thousands of threads at once give overwhelming throughput. So the whole craft of a GPU kernel comes down to two things: cut the work into thousands of independent little pieces (so every thread has something to do), and get the data close enough to the threads (so they do not starve waiting on VRAM). This lesson reads a real CUDA kernel skeleton, then this lesson's star - tiled matmul, then the memory hierarchy, and finally a conceptual look at flash attention.</p>
<p style="color:var(--muted)">Roadmap: first the CUDA execution model (grid / block / thread / warp), then how tiled matmul reuses data through on-chip shared memory, then the three-level memory hierarchy and "moving data costs more than computing", and finally why flash attention fuses several ops into one kernel.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The GPU's performance secret is the same theme as the CPU (L31), just pushed to the extreme: <strong>massive parallelism</strong> (thousands of threads at once, not a few cores) + <strong>keeping them fed</strong> (reuse data in on-chip fast memory, do not let threads idle on VRAM). The CPU's "8-head arm" (SIMD) becomes, on the GPU, a whole stadium of thousands of hands; but the more hands, the greater the "supply" pressure - which is why the most critical code in a GPU kernel is often not "how to compute" but "how to move data into that small, fast shared memory so threads in a block reuse it over and over". Hold these two - spread the parallelism, bring the data close - and the screenful of <span class="mono">blockIdx</span>, <span class="mono">__shared__</span>, <span class="mono">__syncthreads</span> all gains a through-line.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Last lesson's CPU is like a few <strong>master craftsmen</strong>, each with an 8-head robotic arm - skilled work, few people; the GPU is like <strong>thousands of ordinary workers in a stadium</strong>. One worker is slow, but thousands at once give astonishing total throughput - provided you can split the job into thousands of little tasks that wait on no one, and can keep them supplied. Each crew (block) has a <strong>small, fast workbench</strong> (shared memory) in front of it, but the warehouse (global VRAM) is far away. The smart move: the crew first hauls a batch of material from the warehouse onto the bench together, then everyone draws from the bench repeatedly without disturbing each other, and hauls the next batch only when done. Almost all the craft of a GPU kernel is in "using that small bench well" - which is exactly what tiled matmul below is about.
</div>

<h2>The CUDA execution model: grid / block / thread</h2>
<p>CUDA organizes thousands of threads into three levels: one kernel launch is a <strong>grid</strong>, the grid splits into many <strong>blocks</strong>, and each block holds tens to thousands of <strong>threads</strong>. One more unavoidable concept is the <strong>warp</strong>: every 32 threads form a group that executes the same instruction in perfect lockstep (this model is called SIMT, single instruction multiple threads). Threads in the same block can share a patch of on-chip <span class="mono">__shared__</span> memory and can wait for each other with <span class="mono">__syncthreads()</span>; different blocks are largely on their own, with no ordering guaranteed between them.</p>
<p>First a simplest real kernel: multiply every element by a scale. It shows the most basic GPU idiom - <strong>work out "who am I" first, then compute only my share</strong>:</p>
<pre class="code"><span class="cm">// real CUDA kernel: each thread computes one/few output elements (simplified from ggml-cuda/scale.cu)</span>
<span class="kw">__global__</span> void <span class="fn">scale_f32</span>(const float* x, float* dst, float scale, int64_t n) {
    int64_t tid    = blockIdx.x * blockDim.x + threadIdx.x;  <span class="cm">// globally unique thread id</span>
    int64_t stride = blockDim.x * gridDim.x;                 <span class="cm">// total threads in one sweep</span>
    <span class="kw">for</span> (int64_t i = tid; i &lt; n; i += stride)             <span class="cm">// grid-stride loop</span>
        dst[i] = scale * x[i];                              <span class="cm">// each thread on its own, no deps</span>
}
<span class="cm">// launch: set grid and block size</span>
<span class="fn">scale_f32</span>&lt;&lt;&lt;num_blocks, 256&gt;&gt;&gt;(x, dst, scale, n);   <span class="cm">// 256 = threads per block</span></pre>
<p>Line by line: <span class="mono">blockIdx.x * blockDim.x + threadIdx.x</span> turns "which block, and which thread within it" into a globally unique thread id <span class="mono">tid</span> - this is the first line of almost every CUDA kernel. <span class="mono">stride</span> is how many elements all threads together cover in one sweep; when there is more data than threads, this <strong>grid-stride loop</strong> lets each thread hop by <span class="mono">stride</span> and handle a few more. The key point: all threads run the <strong>same code</strong>, landing on different data only through their different <span class="mono">tid</span> - that is SIMT. The <span class="mono">&lt;&lt;&lt;num_blocks, 256&gt;&gt;&gt;</span> triple-angle-bracket syntax at launch sets how many blocks to start and how many threads per block. One easily-missed but crucial point: a GPU launches far <strong>more</strong> threads than it has physical cores (tens of thousands routinely) - not waste, but the GPU's signature trick for hiding latency. When one batch of threads stalls waiting on VRAM, the hardware instantly switches to another ready batch, keeping the compute units never idle. So "more threads than you can use" is exactly the point: it fills the memory-wait time with other threads' computation - which is why a GPU favors "small tasks, massive parallelism" rather than the CPU's few strong cores.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">grid</span><span class="name">grid</span></div><div class="ld">one kernel launch = one grid, holding hundreds-thousands of blocks</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">block</span><span class="name">thread block</span></div><div class="ld">tens-thousands of threads, sharing one patch of on-chip shared memory, syncable with __syncthreads</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">thread</span><span class="name">thread</span></div><div class="ld">smallest execution unit, computes its own cell; every 32 threads form a warp, in lockstep</div></div>
</div>

<h2>Tiled matmul: the heart of the GPU kernel</h2>
<p>Matmul is the heaviest op in a large model, and the key to writing it fast on a GPU is again tiling - the same idea as the CPU tiling in L31, but here we tile into <strong>shared memory</strong>. First, the flaw in the naive way: let each thread independently compute one cell of C, and it must read a whole row of A and a whole column of B from global VRAM (slow). Neighboring threads read heavily overlapping data, yet each reads its own, wasting the already-scarce VRAM bandwidth - the bottleneck is again "waiting for data", not "computing". A concrete sense: the naive version computing an N x N output reads on the order of N-cubed values from slow VRAM (every cell rescans a whole row of A and a whole column of B, with heavy duplicate reads between neighboring threads); the multiply-adds the machine is actually good at are fast, so most of the time goes to "waiting for data". That is exactly the core tension tiling resolves.</p>
<p>The tiled fix has a block's threads <strong>cooperate</strong>: together they first haul a small tile of A and a small tile of B from global into the block's shared <span class="mono">__shared__</span> memory (on-chip, one-to-two orders of magnitude faster), wait with <span class="mono">__syncthreads()</span> until everyone has loaded, then every thread in the block draws from that fast memory repeatedly to compute its own cell, advancing tile by tile along K and accumulating. Below is a normalized teaching skeleton (the real <span class="mono">mmq.cuh</span> adds 4-bit unpacking and heavy templating on top, but the skeleton is exactly this):</p>
<pre class="code"><span class="cm">// teaching tiled-matmul skeleton (real ggml-cuda/mmq.cuh adds 4-bit unpack + templates)</span>
<span class="kw">__global__</span> void <span class="fn">matmul_tiled</span>(const float* A, const float* B, float* C, int N) {
    <span class="kw">__shared__</span> float As[T][T];                 <span class="cm">// on-chip shared: a tile of A</span>
    <span class="kw">__shared__</span> float Bs[T][T];                 <span class="cm">// on-chip shared: a tile of B</span>
    int row = blockIdx.y*T + threadIdx.y;
    int col = blockIdx.x*T + threadIdx.x;
    float acc = 0.0f;                            <span class="cm">// accumulator in a register (fastest)</span>
    <span class="kw">for</span> (int k0 = 0; k0 &lt; N; k0 += T) {         <span class="cm">// advance along K, tile by tile</span>
        As[threadIdx.y][threadIdx.x] = A[row*N + (k0+threadIdx.x)]; <span class="cm">// cooperative load</span>
        Bs[threadIdx.y][threadIdx.x] = B[(k0+threadIdx.y)*N + col];
        <span class="fn">__syncthreads</span>();                       <span class="cm">// wait until the whole tile is loaded</span>
        <span class="kw">for</span> (int k = 0; k &lt; T; ++k)             <span class="cm">// reuse the tile, all in fast memory</span>
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        <span class="fn">__syncthreads</span>();                       <span class="cm">// wait before overwriting next tile</span>
    }
    C[row*N + col] = acc;                        <span class="cm">// write each cell back just once</span>
}</pre>
<p>The crux is those two <span class="mono">__syncthreads()</span>: the first guarantees "the whole tile is on the bench" before any compute starts, the second guarantees "everyone is done" before overwriting the next tile - drop either and some thread reads half-loaded data and the result is all wrong. Once a tile is in shared memory it is <strong>reused</strong> many times by all threads in the block, amortizing the slow global accesses down to almost nothing. This carries last lesson's "load once, reuse many" tiling onto the GPU's on-chip memory: the CPU reuses cache, the GPU reuses shared memory, but the memory-saving arithmetic is identical. A rough sense of the payoff: once a T x T tile is in shared, the block's threads reuse it as rows/columns about T times - so each value hauled from global is reused roughly T-fold; with T = 32, slow VRAM traffic drops to about one thirtieth. (T is the tile width, often 16 or 32; blockDim is set to T x T so one thread owns exactly one cell.)</p>
<p>Freezing the process into one picture is clearest - a block loads A's row-tile and B's col-tile into shared, and each thread in the block owns one cell of the output C:</p>
<div class="trace">
  <div class="tcap"><b>Trace one tiled matmul</b>: one thread block loads A's row-tile and B's col-tile into shared memory; each thread in the block computes one C cell (dot of an As row and a Bs col), looping and accumulating along K.</div>
<svg viewBox="0 18 660 372" width="100%" role="img" aria-label="Tiled matrix multiply example: one thread block loads A's row-tile and B's col-tile into shared memory, and each thread in the block computes one cell of C">
<g font-family="ui-monospace,monospace">
<rect x="117" y="189" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="120" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="120" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="148" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="176" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="204" y="220" width="28" height="28" fill="#fbe7d2" stroke="#c2630e"/>
<rect x="120" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="120" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="148" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="176" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="204" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="377" y="49" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="380" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="52" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="52" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="80" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="80" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="108" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="108" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="136" width="28" height="28" fill="#dbe7fc" stroke="#2563eb"/>
<rect x="436" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="136" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="377" y="189" width="118" height="118" rx="6" fill="none" stroke="#cdd5df"/>
<rect x="380" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="192" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="220" width="28" height="28" fill="#ece3fb" stroke="#7c3aed"/>
<rect x="436" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="220" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="248" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="380" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="408" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="436" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<rect x="464" y="276" width="28" height="28" fill="#ffffff" stroke="#cdd5df"/>
<text x="176" y="180" text-anchor="middle" fill="#c2630e" font-weight="700" font-size="12">As (shared)</text>
<text x="436" y="40" text-anchor="middle" fill="#2563eb" font-weight="700" font-size="12">Bs (shared)</text>
<text x="436" y="324" text-anchor="middle" fill="#7c3aed" font-weight="700" font-size="12">C output tile</text>
<text x="110" y="238" text-anchor="end" fill="#c2630e" font-weight="700" font-size="11">row i</text>
<text x="432" y="182" fill="#2563eb" font-weight="700" font-size="11">col j</text>
<line x1="238" y1="234" x2="368" y2="234" stroke="#c2630e" stroke-width="1.6"/>
<path d="M 376 234 L 368 230 L 368 238 z" fill="#c2630e"/>
<line x1="422" y1="170" x2="422" y2="180" stroke="#2563eb" stroke-width="1.6"/>
<path d="M 422 188 L 418 180 L 426 180 z" fill="#2563eb"/>
<text x="504" y="238" fill="#7c3aed" font-weight="700" font-size="11">C[i][j]</text>
<line x1="500" y1="234" x2="438" y2="234" stroke="#7c3aed" stroke-width="1.2"/>
<rect x="60" y="332" width="540" height="44" rx="6" fill="#ffffff" stroke="#cdd5df"/>
<text x="330" y="350" text-anchor="middle" fill="#1d2129" font-weight="700" font-size="12">thread(i,j): C[i][j] += As[i][k] * Bs[k][j]</text>
<text x="330" y="368" text-anchor="middle" fill="#5b6470" font-size="11">loop k0 = 0..K step T: load tile -> __syncthreads -> accumulate -> __syncthreads</text>
</g></svg>
</div>

<h2>The memory hierarchy: why "moving data" costs more than "computing"</h2>
<p>We kept saying "fast memory" and "slow VRAM" above; here is the clear version. A GPU has three levels of memory, each smaller and faster as you go down: <strong>global (VRAM)</strong> is several GB, visible to all threads, but high-latency, and its bandwidth is often the whole kernel's bottleneck; <strong>shared (on-chip)</strong> is only tens of KB per block, shared just among that block's threads, one-to-two orders of magnitude faster than global; <strong>register</strong> is private to each thread and fastest - the accumulator <span class="mono">acc</span> in tiled matmul lives here. The entire point of tiled matmul is to move data from global up into shared and registers and reuse it.</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">big-slow</span><span class="name">global / VRAM</span></div><div class="ld">several GB, all threads see it; high latency, bandwidth is the bottleneck (echoes L30's memory-bound)</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">mid-fast</span><span class="name">shared / on-chip</span></div><div class="ld">tens of KB per block, shared within a block; one-to-two orders faster than global</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">small-fastest</span><span class="name">register</span></div><div class="ld">private to each thread, fastest; the accumulator acc lives here</div></div>
</div>
<p>This confirms L30's observation once more: many kernels are "memory-bound", not "compute-bound" - a GPU has so much compute it often goes unused, and what actually stalls it is the bandwidth feeding data in from VRAM. So most of GPU optimization goes into "fewer global accesses, more shared/register reuse". Grasp this and you see why, on the same card, a well-written kernel can be several times faster than a naive one - the total work is unchanged; all the savings are in movement. It is also why batching (feeding many sequences/tokens at once) almost always raises GPU utilization: bring data in once, compute on it several times, and get more value out of that one bandwidth. One more GPU-specific subtlety: when accessing global VRAM, the 32 threads of a warp should ideally read <strong>contiguous neighboring</strong> addresses, so the hardware can merge them into one large memory transaction (coalesced access); if each reads a scattered address, it degrades into dozens of tiny transactions and bandwidth utilization collapses. So a GPU kernel must not only "read global less" but also "read it tidily" - which is why a tensor's layout in VRAM matters so much for performance, and why many kernels rearrange data specifically to satisfy this rule.</p>

<h2>Flash attention: fusing several ops into one kernel</h2>
<p>Attention computes <span class="mono">softmax(Q @ K^T) @ V</span>. The naive way first computes the whole huge N x N score matrix and writes it to VRAM, reads it back for softmax, reads it back again to multiply V - just writing that N x N matrix once and reading it three times drains the bandwidth (and costs a big slab of VRAM besides). <strong>Flash attention</strong> <strong>fuses</strong> these steps into one kernel: compute in tiles, correcting with an "online softmax" as you go, <strong>never materializing the whole N x N score matrix to VRAM</strong>, writing out only the final output O.</p>
<pre class="code"><span class="cm">// flash attention idea (concept pseudocode; real impl in ggml-cuda/fattn*.cu)</span>
acc = 0;  run_max = -inf;  run_sum = 0;
<span class="kw">for</span> each K/V tile:                  <span class="cm">// tiled; never materialize the full N x N scores</span>
    s   = Q @ K_tile^T               <span class="cm">// scores for this tile only</span>
    m   = max(run_max, rowmax(s))    <span class="cm">// update the running row max</span>
    p   = exp(s - m)                 <span class="cm">// online softmax: correct as you go</span>
    run_sum = run_sum * exp(run_max - m) + rowsum(p)
    acc     = acc     * exp(run_max - m) + p @ V_tile
    run_max = m
O = acc / run_sum                    <span class="cm">// normalize once; write only O, not the N x N</span></pre>
<p>The trick is that "online softmax": softmax normally needs a whole row before it can fix the denominator, but flash attention keeps a <strong>running max and a running denominator</strong> (<span class="mono">run_max</span> / <span class="mono">run_sum</span>), and on each new tile rescales the accumulated result to the new max - so it can march forward tile by tile without waiting for the full row. The cost is a few extra rescale operations; the reward is <strong>never shuttling that N x N matrix in and out of VRAM</strong> - the longer the sequence, the more you save. That is the power of "op fusion": work that would otherwise take several kernels reading and writing VRAM is merged into one kernel done in a single pass, and all the savings are in the most expensive part - memory traffic. This matters most for long context: double the sequence length and that N x N matrix quadruples in area, so the naive approach's VRAM and bandwidth costs explode, whereas flash attention, never materializing it, grows only linearly with the sequence - one of the key reasons today's tens-of-thousands to hundreds-of-thousands token contexts are feasible at all.</p>
<div class="card detail">
  <div class="tag">🔬 Scope &amp; source</div>
  This section conveys only the <strong>idea</strong> of flash attention, not a line-by-line real kernel. The reason is that ggml's <span class="mono">fattn-*</span> files (<span class="mono">fattn-tile</span> / <span class="mono">fattn-mma-f16</span> / <span class="mono">fattn-vec</span> and more) carry a pile of template specializations to run fastest across different GPUs, precisions, and head dimensions - many lines, dense branches. But it all comes back to the same thing: the "tile + online softmax + do not materialize N x N" above. Fix the idea first and you will not get lost when you do read the source.
</div>

<h2>Deeper: warp-level sum and kernel variants</h2>
<p>Two last folds for two GPU-kernel details that are common yet easy to find baffling: how a warp sums quickly without shared memory, and why one op has so many kernels.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> How do 32 threads in a warp sum without shared memory? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Many kernels (softmax, norm, the tail of a dot product) finally need to add up each thread's partial sum into one total. Going through shared memory means write, <span class="mono">__syncthreads</span>, then read - fairly costly. But the 32 threads in one warp are already in lockstep, and CUDA offers <strong>warp shuffle</strong> instructions that let them <strong>read each other's registers directly</strong>, touching no memory. Here is ggml's real <span class="mono">warp_reduce_sum</span> (<span class="mono">ggml-cuda/common.cuh</span>):</p>
<pre class="code"><span class="cm">// sum 32 threads within a warp, no shared memory (from ggml-cuda/common.cuh)</span>
<span class="kw">__device__</span> float <span class="fn">warp_reduce_sum</span>(float x) {
    <span class="kw">for</span> (int offset = 16; offset &gt; 0; offset &gt;&gt;= 1)   <span class="cm">// 16 -> 8 -> 4 -> 2 -> 1</span>
        x += <span class="fn">__shfl_xor_sync</span>(0xffffffff, x, offset, 32); <span class="cm">// read the neighbor "offset away"</span>
    <span class="kw">return</span> x;                                          <span class="cm">// afterwards every thread holds the total</span>
}</pre>
    <p>This is a <strong>butterfly reduction tree</strong>: round one, each thread swaps with and adds its partner "16 away", round two "8 away", then 4, 2, 1 - after five rounds (log2 of 32) every thread holds the sum of all 32 values. The first argument <span class="mono">0xffffffff</span> of <span class="mono">__shfl_xor_sync</span> is the mask of participating threads (all 32), and it lets a thread grab another thread's register <span class="mono">x</span> directly, never via shared or global. That is why warp-level reduction is fast and cheap - it pushes "inter-thread communication" down to the register level.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> Why are there so many kernels for one matmul? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Browse <span class="mono">ggml-cuda/</span> and you will find one op often has several implementations: <span class="mono">mmq</span> (quantized matmul), <span class="mono">mmvq</span> (matrix x vector, used when decoding a single token), plus specializations for different quant formats (Q4_0/Q4_K/...) and different GPU compute capabilities. Why not one generic version? Because GPU performance is <strong>extremely sensitive</strong> to these conditions: decoding is a "tall thin" matrix x vector, prefill is a "fat" matrix x matrix (echoing pp vs tg in L18/L30), and the best tile size, whether to use tensor cores, and how registers are allocated all differ. Rather than one compromise that is mediocre everywhere, pick the fastest for each important case - the same engineering trade-off as the CPU's multi-arch dispatch in <span class="mono">arch/</span> from L31: <strong>spend code to buy performance</strong>. At runtime it then picks the most suitable kernel by tensor shape, quant type, and GPU model.</p>
  </div>
</details>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>CUDA's three levels: <span class="mono">grid -&gt; block -&gt; thread</span>; every 32 threads a <strong>warp</strong> in lockstep; within a block, <span class="mono">__shared__</span> + <span class="mono">__syncthreads</span>.</li>
    <li>Each thread computes its share: <span class="mono">tid = blockIdx.x*blockDim.x + threadIdx.x</span>; with more data than threads, use a <strong>grid-stride loop</strong>.</li>
    <li>Tiled matmul is the heart: a block cooperatively loads A/B tiles into <span class="mono">__shared__</span>, then after <span class="mono">__syncthreads</span> reuses them, accumulating along K; the real <span class="mono">mmq.cuh</span> adds 4-bit unpack.</li>
    <li>Three memory levels <span class="mono">global / shared / register</span> (big-slow -&gt; small-fast -&gt; fastest); the bottleneck is usually bandwidth, so optimize = fewer global accesses, more shared reuse (echoes L30).</li>
    <li>Flash attention: fuse softmax + matmul into one kernel, never materializing the N x N scores, saving VRAM and bandwidth; real impl in <span class="mono">fattn*.cu</span>.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  GPU and CPU look worlds apart, yet the kernel mindset is <strong>the same</strong>: given the same amount of computation, how do you spread it out to run at once, and how do you make data travel less. The CPU pushes it to a few cores x 8-lane SIMD; the GPU pushes it to thousands of threads x on-chip reuse. So the two questions you learned in L31 - "can it compute several at once?" and "is data fetched from afar over and over?" - carry over verbatim to the GPU, only the answers become "thousands of threads" and "shared memory". See through the <span class="mono">blockIdx</span>, <span class="mono">__shared__</span>, <span class="mono">__syncthreads</span> syntax and underneath is still that plain idea: keep the massive hardware busy and do not let it starve. It is also why "op fusion" (flash attention) and "tiled reuse" (tiling) became the two great themes of GPU optimization - both are fighting that slow, narrow VRAM bandwidth. Next lesson we step back to see how ggml uses one unified "backend" abstraction to manage CPU, CUDA, and more, dispatching a compute graph's ops to the right hardware.
</div>
""",
}

LESSON_33 = {
    "zh": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
前两课分别看了 CPU（L31）和 CUDA（L32）各自怎么把算子算出来。可现实里，一台机器上常常同时有 CPU 和一块或多块 GPU——ggml 怎么把它们统一管起来，又怎么决定每个算子该交给谁算？这一课看 <span class="mono">ggml-backend</span> 这层后端抽象：它是第六部分的收束，也是把前面所有"怎么算"接回"在哪算"的关键一环。
</p>
<p style="color:var(--muted);margin-top:.4rem">想想 L09/L10 那张计算图：一串算子，从输入流到输出。现在问题来了——这串算子里，有的最好在 GPU 上算（矩阵乘），有的可能留在 CPU 更省事；张量还可能这会儿在显存、那会儿在内存。谁来决定每个算子去哪、谁来在设备之间搬张量？答案就是 ggml 的后端抽象层和调度器。这一课会看它怎么用一套统一接口屏蔽硬件差异、怎么在运行时动态加载后端、怎么把一张图分派下去执行。举个最贴近的场景：你用一块显存不太够的显卡跑一个大模型，于是把一部分层放 GPU、剩下的留 CPU——这时每跑一遍模型，数据都要在 CPU 和 GPU 之间来回过好几道，靠的全是这一课要讲的后端与调度。理解了它，你才真正明白 <span class="mono">-ngl</span> 这个参数背后到底发生了什么。</p>
<p style="color:var(--muted)">路线图：先看后端抽象（统一接口），再看注册与动态加载（一份二进制按硬件挑后端），然后是调度器（把图的算子分派到各后端、并按需搬张量），最后扫一眼 ggml 支持的其它后端。</p>

<div class="card macro">
  <div class="tag">🌍 宏观理解</div>
  后端抽象解决的是一个朴素却关键的问题：让上层代码<strong>不必关心硬件</strong>。计算图（L09）只管"要算什么"，至于某个算子最终落在 CPU、CUDA 还是 Metal 上，由各后端的实现去管。这层抽象就像给所有硬件套了一个统一插座：上层对着插座写一次，换什么"电器"（后端）都能插上。把"算什么"和"在哪算"解耦，正是 ggml 能同时支持十来种硬件、还能让它们协同工作的根本。这一课不再钻某一种硬件的内核（那是 L31/L32 的事），而是退到更高一层，看这套"管理 + 调度"的骨架。说得再直白点：前两课像"显微镜"，凑近看一种硬件的内核长什么样；这一课像"地图"，俯瞰所有硬件是怎么被统一接进来、又怎么协同的。两种视角缺一不可——没有显微镜，你不懂性能到底从哪来；没有地图，你不懂这一堆五花八门的硬件怎么被同一份代码驱动。
</div>

<div class="card analogy">
  <div class="tag">🔌 生活类比</div>
  把计算图想成一叠<strong>工单</strong>（每张是一个算子），后端就是各种<strong>工种的师傅</strong>（CPU 师傅、GPU 师傅……）。<span class="mono">ggml_backend_sched</span> 则是<strong>调度工头</strong>：他拿起每张工单，看看这活儿谁最合适、需要的料（输入张量）现在在谁手里，然后派给对应的师傅；要是料在 CPU 师傅手上、活儿却要 GPU 师傅做，工头就先把料搬过去。上层只管把一叠工单交给工头，至于具体派给谁、料怎么搬，全由这套抽象 + 调度兜住——这也是为什么你用 llama.cpp 时，只要用 <span class="mono">-ngl</span> 指定把几层放上 GPU，剩下的派活、搬料就全由这位"工头"在后台替你打理好了——你根本不用操心哪个算子具体跑在哪、张量又在什么时候被搬到哪里。
</div>

<h2>后端抽象：给所有硬件一个统一接口</h2>
<p>一个 <span class="mono">ggml_backend</span> 代表"一个能跑算子的设备"，它打包了三样东西：一个 <strong>device</strong>（设备句柄）、一套 <strong>buffer</strong>（在该设备上分配/读写张量的内存）、以及一组<strong>算子实现</strong>（这块硬件怎么算 matmul、softmax……）。每种后端（CPU、CUDA、Metal……）都去实现同一套接口（<span class="mono">ggml-backend-impl.h</span> 里的 <span class="mono">ggml_backend_i</span>），于是上层的计算图只需对着这个抽象写，完全不用管底下到底是哪种硬件。这里的 <span class="mono">buffer</span> 是个容易被忽略的关键：一个张量的数据到底躺在 CPU 内存还是 GPU 显存里，就由它决定；同一个张量在 CPU buffer 和 GPU buffer 里是两份不同的内存，后面调度器跨设备搬的，正是这些 buffer 里的数据。设备还带一个类型，方便上层"按需要挑"：</p>
<pre class="code"><span class="cm">// 设备类型 (ggml-backend.h)</span>
<span class="kw">enum</span> ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,    <span class="cm">// CPU</span>
    GGML_BACKEND_DEVICE_TYPE_GPU,    <span class="cm">// 独立 GPU</span>
    GGML_BACKEND_DEVICE_TYPE_ACCEL,  <span class="cm">// 加速器 (配合 CPU 用, 如 BLAS/AMX)</span>
    <span class="cm">// ... 实际还有 IGPU(集成显卡) / META 等</span>
};
<span class="cm">// 挑一个 GPU 设备; 没有就回退到 CPU (用法见 ggml-backend-reg.cpp)</span>
ggml_backend_dev_t dev = <span class="fn">ggml_backend_dev_by_type</span>(GGML_BACKEND_DEVICE_TYPE_GPU);
<span class="kw">if</span> (!dev) dev = <span class="fn">ggml_backend_dev_by_type</span>(GGML_BACKEND_DEVICE_TYPE_CPU);</pre>
<p>这套设计的好处，在 L31/L32 已经体现过一半：上层调 <span class="mono">matmul</span> 时根本不写"用 AVX2 还是 CUDA"，那是各后端实现里的事。后端抽象把这件事正式化——计算图（L09/L10）描述"做什么"，后端负责"怎么做、在哪做"。图和硬件之间彻底解耦：</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">上层</span><span class="name">计算图 / 模型</span></div><div class="ld">只描述算子和依赖 (L09/L10), 不关心硬件</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">中间</span><span class="name">ggml-backend 抽象</span></div><div class="ld">统一接口: device + buffer + 一组算子实现 (impl.h 里的几套接口)</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">底层</span><span class="name">各后端</span></div><div class="ld">CPU / CUDA / Metal / Vulkan ... 各自实现同一套接口</div></div>
</div>

<h2>注册与动态加载：一份二进制，按硬件挑后端</h2>
<p>抽象有了，可程序怎么知道这台机器上有哪些后端？答案是<strong>运行时动态加载</strong>。每个后端编译成一个动态库（.so/.dll），程序启动时调 <span class="mono">ggml_backend_load_all()</span> 去扫描、把能用的后端库一个个加载进来、登记进一张注册表。底层真正干活的是 <span class="mono">dl_load_library</span>：</p>
<pre class="code"><span class="cm">// 运行时加载所有可用后端 (ggml-backend-reg.cpp)</span>
<span class="kw">void</span> <span class="fn">ggml_backend_load_all</span>();   <span class="cm">// 扫描并加载 cuda/metal/vulkan... 动态库</span>

<span class="cm">// 底层加载单个库 (ggml-backend-dl.cpp), 跨平台两套</span>
<span class="cm">// POSIX:</span>
handle = <span class="fn">dlopen</span>(path, RTLD_NOW | RTLD_LOCAL);
<span class="cm">// Windows:</span>
handle = <span class="fn">LoadLibraryW</span>(path);</pre>
<p>加载完，程序就能问注册表"现在有几个后端、几个设备"：<span class="mono">ggml_backend_reg_count()</span>、<span class="mono">ggml_backend_dev_count()</span>，再用上一节的 <span class="mono">ggml_backend_dev_by_type</span> 挑设备。这套动态加载的意义很实在：<strong>一份主程序二进制，按机器实际硬件在运行时加载对应后端</strong>——有 CUDA 卡就加载 CUDA 后端，没有就不加载，绝不会因为缺少某个库而启动失败。你可以把注册表想成一本"电话簿"：每加载进一个后端，它就在簿子上登记一条"我是谁、我有哪些设备、怎么找到我"。上层要用 GPU 时，不是直接去认 CUDA，而是翻这本簿子按类型查——正是这层间接，让"上层不认识具体硬件"能真正成立。</p>
<div class="card detail">
  <div class="tag">🔬 为什么用动态加载</div>
  换个角度想：如果把所有后端都<strong>静态</strong>编进一个二进制，那它就得同时链接 CUDA、Vulkan、Metal、SYCL…… 一堆庞大又互相冲突的依赖，而且换台没有这些库的机器就跑不起来。动态加载把这件事推迟到运行时：发布一份精简的主程序，到了用户机器上，<strong>有什么硬件就加载什么后端</strong>。这也是为什么 llama.cpp 的预编译包能做到"一个包、到处能跑"——CUDA 后端在没有 N 卡的机器上只是没被加载，而不是让整个程序崩掉。
</div>

<h2>调度：把一张图的算子分派到各后端</h2>
<p>有了多个后端，最后一个问题是：一张计算图（L09/L10）里的算子，谁来决定每个去哪个后端算、又在后端之间搬张量？这就是 <span class="mono">ggml_backend_sched</span>（调度器）的活儿。它的两个主要接口是：</p>
<pre class="code"><span class="cm">// 建调度器, 交给它一组后端 (ggml-backend.h)</span>
sched = <span class="fn">ggml_backend_sched_new</span>({backend_gpu, backend_cpu}, ...);
<span class="cm">// 把整张计算图分派到各后端执行</span>
<span class="fn">ggml_backend_sched_graph_compute</span>(sched, graph);</pre>
<p>调度器拿到图后，逐个看每个算子：它的输入张量现在在哪个后端的 buffer 里？这个算子在哪个后端上算最合适（比如矩阵乘优先 GPU）？据此把算子<strong>指派</strong>给一个后端；如果某个输入还在别的设备上（算子要在 GPU 上跑、输入却还在 CPU 内存里），调度器就先插一次<strong>跨设备拷贝</strong>，把输入搬过去，再执行。整张图跑完，结果就落在该在的地方了。值得一提的是，调度器并不是把每个算子<strong>孤立</strong>地分派——跨设备拷贝很贵（又是访存，呼应 L30/L32），所以它会尽量把连续的、能在同一设备上算的算子<strong>成段</strong>地交给同一个后端，减少来回搬运。换句话说，它不只看"这个算子在哪算最快"，还看"怎么切这张图，整体的跨设备搬运最少"——这也是为什么 <span class="mono">-ngl</span> 一般是"前若干层整段放 GPU"，而不是东一层西一层地乱放。把"分派一个算子"这件事定格成一条流水看最清楚：</p>
<div class="trace">
  <div class="tcap"><b>追踪一次算子分派</b>：调度器看一个算子的输入在哪、挑后端、必要时跨设备拷贝、再执行写回（示意）。</div>
  <div class="stations">
    <div class="stn"><h5>① 取算子</h5>
      <div class="cellrow"><span class="vc">graph 里一个 op</span></div>
      <div class="tlab">如一次 matmul</div></div>
    <div class="op">看输入<br>在哪</div>
    <div class="stn"><h5>② 查位置</h5>
      <div class="cellrow"><span class="vc">输入在哪个 buffer</span></div>
      <div class="tlab">CPU 内存? 显存?</div></div>
    <div class="op">挑<br>后端</div>
    <div class="stn"><h5>③ 选后端</h5>
      <div class="cellrow"><span class="vc blue">输入在 GPU -> 派 GPU</span></div>
      <div class="tlab">就近、最合适</div></div>
    <div class="op">必要时<br>拷贝</div>
    <div class="stn"><h5>④ 搬张量</h5>
      <div class="cellrow"><span class="vc">跨设备拷贝输入</span></div>
      <div class="tlab">若输入在别处</div></div>
    <div class="op">执行<br>写回</div>
    <div class="stn"><h5>⑤ 算并写回</h5>
      <div class="cellrow"><span class="vc hot">该后端执行 -> 输出</span></div>
      <div class="tlab">结果留在该设备</div></div>
  </div>
</div>
<p>这正好把 L09/L10 那张"静态的图"接到了"动态的执行"上：图描述依赖，调度器按依赖顺序、结合每个张量的实际位置，把算子一个个落到具体硬件上跑。你平时用 <span class="mono">-ngl N</span> 把前 N 层放上 GPU、其余留 CPU，背后正是这个调度器在按层分派、并在 GPU 与 CPU 之间搬运边界处的张量。而当 GPU 显存实在放不下整个模型时，这种 CPU+GPU 混合执行往往是唯一能跑起来的办法——代价是边界处那几次跨设备拷贝，但总比完全跑不动强。这也解释了为什么 <span class="mono">-ngl</span> 调大调小，速度和显存占用会此消彼长：放上 GPU 的层越多，算得越快，但占的显存也越多、CPU-GPU 之间的搬运点也跟着变。</p>

<h2>其它后端一览</h2>
<p>除了已经细看的 CPU（L31）和 CUDA（L32），ggml 还实现了一大批后端，覆盖各家硬件。它们都遵循同一套 <span class="mono">ggml_backend_i</span> 接口，所以上层代码几乎不用改，换硬件只是换一个加载进来的后端：</p>
<table class="t">
  <tr><th>后端</th><th>面向的硬件 / 平台</th><th>典型场景</th></tr>
  <tr><td><span class="mono">Metal</span></td><td>Apple GPU（macOS / iOS）</td><td>苹果设备上的首选 GPU 后端</td></tr>
  <tr><td><span class="mono">Vulkan</span></td><td>跨平台 GPU</td><td>不限厂商的通用 GPU 加速</td></tr>
  <tr><td><span class="mono">SYCL</span></td><td>Intel GPU</td><td>Intel 独显 / 集显</td></tr>
  <tr><td><span class="mono">HIP</span></td><td>AMD GPU</td><td>A 卡（对标 CUDA）</td></tr>
  <tr><td><span class="mono">CANN</span></td><td>华为昇腾 NPU</td><td>昇腾加速卡</td></tr>
  <tr><td><span class="mono">OpenCL</span></td><td>跨平台（含移动 GPU）</td><td>移动 / 嵌入式（如高通 Adreno）</td></tr>
  <tr><td><span class="mono">BLAS</span></td><td>CPU（借现成数学库）</td><td>用 BLAS 库加速 CPU 矩阵乘</td></tr>
  <tr><td><span class="mono">RPC</span></td><td>远程机器 / 进程</td><td>把算子发到另一台机器上跑</td></tr>
</table>
<p>这张表最能说明后端抽象的价值：从苹果的 Metal 到华为的 CANN、从本机 GPU 到远程的 RPC，硬件天差地别，但对上层而言都只是"一个实现了 <span class="mono">ggml_backend_i</span> 的设备"。想支持一种新硬件，本质上就是再写一份后端实现、注册进来——上层的模型代码一行都不用动。事实上，这些后端里有不少是社区或硬件厂商贡献的：正因为接口是统一的，华为能自己来写 CANN、Intel 能来写 SYCL，而不必去动 ggml 的核心。一个好的抽象接口，等于给整个生态开了一扇"你来适配硬件、我保证上层不变"的门——这也是开源项目能在短时间里支持这么多硬件的组织学原因。</p>

<h2>深入：特殊后端与如何加一个后端</h2>
<p>最后两个折叠：看两个"不太像 GPU"的特殊后端，以及加一个新后端大致要实现什么。</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> BLAS 和 RPC 这种"特殊后端"是什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>大多数后端对应一种"算力硬件"，但有两个例外很有意思。<strong>BLAS 后端</strong>并不直接写 kernel，而是把矩阵乘转交给系统里现成的高性能 BLAS 数学库（如 OpenBLAS、MKL）去算——相当于"借别人造好的轮子"，在某些 CPU 上比 ggml 自带的实现还快。<strong>RPC 后端</strong>更特别：它根本不在本地算，而是把算子通过网络<strong>发到另一台机器（或进程）</strong>上跑，再把结果取回来。有了它，你可以把一个单机装不下的大模型，拆到好几台机器的显存里分布式地跑。这两个后端都套着同一层 <span class="mono">ggml_backend_i</span> 接口，所以对上层来说，"借数学库"和"发去远程"与"在本地 GPU 上算"没有任何区别——这正是抽象的威力。顺便说，正因为有 RPC 这种后端，"在一台机器上调度、把重活发到另一台或多台去跑"这类玩法才成为可能；而 BLAS 后端则提醒我们：所谓"后端"未必对应一种新硬件，也可以只是"换一套更快的算法实现"。抽象层只关心"你能不能实现这套接口"，至于你背后到底是一块芯片、一个数学库、还是一根网线，它一概不问。</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> 加一个新后端大致要实现什么？ <span class="hint">点击展开</span></summary>
  <div class="acc-body">
    <p>想给 ggml 加一种新硬件，核心就是去实现 <span class="mono">ggml-backend-impl.h</span> 里定义的那几套接口（设备接口 + 后端接口），主要包括几类：<strong>buffer 管理</strong>（怎么在这块硬件上分配内存、把张量数据拷进拷出）、<strong>算子支持查询</strong>（<span class="mono">supports_op</span>：这个后端能不能算某个算子——不能的就让调度器回退给别的后端）、以及<strong>跑计算图</strong>（<span class="mono">graph_compute</span>：把分给我的这串算子真正算出来）。实现完、再注册进注册表，上层的模型和调度器就能立刻用上它，完全不用改。这套"定义好接口、谁都能往里插"的设计，正是 ggml 能在这么短时间里长出十几种后端的原因——也是软件工程里"面向接口编程"最实在的一个例子。</p>
  </div>
</details>

<p>第六部分到这里就收束了。从 L31 的 CPU 指令（标量、SIMD、多线程）、L32 的 CUDA 线程（grid/block、分块、显存层级），到这一课的后端抽象与调度，"一个算子最终怎么落到硬件上算"这条线，算是从头讲透了。再往上回看：模型（第四部分）描述要算什么，计算图（第三部分）把它组织成依赖，后端（这一部分）把图落到具体硬件——三层一接，llama.cpp 跑模型的全貌就清楚了。下一站第七部分，我们去看一些进阶专题。学到这里，你已经把 llama.cpp 从"一个能跑模型的黑盒"，拆成了"一摞看得懂的层"——这本身就是啃源码最大的收获：再复杂的系统也不是一团乱麻，而是一层层抽象垒起来的，每一层只解决一个问题、只对上一层暴露一个干净的接口。把这套眼光带走，你以后去读任何一个大项目，都会比从前从容得多。</p>

<div class="card key">
  <div class="tag">✅ 关键要点</div>
  <ul>
    <li>后端抽象 <span class="mono">ggml_backend</span> = 一个能跑算子的设备（device + buffer + 一组算子实现），各后端实现同一套 <span class="mono">ggml_backend_i</span>，上层计算图不关心硬件。</li>
    <li>注册与动态加载：<span class="mono">ggml_backend_load_all</span> 运行时用 <span class="mono">dlopen</span> / <span class="mono">LoadLibraryW</span> 加载各后端动态库——一份二进制按实际硬件挑后端。</li>
    <li>调度 <span class="mono">ggml_backend_sched</span>：把一张计算图（L09/L10）的算子分派到各后端，并在设备之间按需拷贝张量。</li>
    <li>其它后端：Metal / Vulkan / SYCL / HIP / CANN / OpenCL / BLAS / RPC——同一接口，覆盖从苹果到华为、从本机到远程。</li>
    <li>加新后端 = 实现 <span class="mono">ggml-backend-impl.h</span> 的接口（buffer、<span class="mono">supports_op</span>、<span class="mono">graph_compute</span>）并注册。</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 设计洞察</div>
  这一课其实在讲软件工程里一个最经典的招式：<strong>用一层抽象，把"变化的部分"和"不变的部分"隔开</strong>。硬件年年在变（新 GPU、新 NPU、新指令集），但"计算图描述要算什么"这件事是稳定的。ggml 在这两者之间插了 <span class="mono">ggml_backend</span> 这层接口：上面的模型代码十年不用动，下面的硬件想加就加。L31/L32 让你看到了"内核怎么把硬件用满"，这一课让你看到"框架怎么把一堆不同的硬件统一管起来"——前者是深度，后者是广度，合起来才是 llama.cpp 能既快又到处能跑的全部秘密。带着这套"分层 + 抽象"的眼光回看整个教程，你会发现它处处都是：tokenizer 之于文本、GGUF 之于权重、计算图之于算子、后端之于硬件——每一层都在把复杂藏进一个干净的接口后面。
</div>
""",
    "en": r"""
<p class="lead" style="font-size:1.06rem;color:var(--muted);margin-top:-.6rem">
The last two lessons watched CPU (L31) and CUDA (L32) each compute ops on their own. But in reality a machine often has a CPU and one or more GPUs at once - how does ggml manage them uniformly, and how does it decide which op goes to whom? This lesson looks at the <span class="mono">ggml-backend</span> abstraction: the close of Part 6, and the key link tying all the earlier "how to compute" back to "where to compute".
</p>
<p style="color:var(--muted);margin-top:.4rem">Recall the compute graph from L09/L10: a chain of ops flowing from input to output. Now the question: within that chain, some ops are best on the GPU (matmul), some are simpler left on the CPU; and a tensor may be in VRAM now, in RAM later. Who decides where each op goes, and who moves tensors between devices? The answer is ggml's backend abstraction and scheduler. This lesson sees how it hides hardware differences behind one uniform interface, how it loads backends dynamically at runtime, and how it dispatches a whole graph for execution. A most relatable scenario: you run a big model on a card whose VRAM is a bit short, so you put some layers on the GPU and leave the rest on the CPU - now every pass of the model shuttles data back and forth between CPU and GPU several times, all riding on the backend and scheduler this lesson covers. Understand it and you truly see what happens behind that <span class="mono">-ngl</span> flag.</p>
<p style="color:var(--muted)">Roadmap: first the backend abstraction (one uniform interface), then registry and dynamic loading (one binary picks backends by hardware), then the scheduler (dispatch a graph's ops to backends and move tensors as needed), and finally a quick tour of the other backends ggml supports.</p>

<div class="card macro">
  <div class="tag">🌍 Big picture</div>
  The backend abstraction solves a plain but crucial problem: letting upper-layer code <strong>not care about hardware</strong>. The compute graph (L09) only says "what to compute"; whether a given op ends up on CPU, CUDA, or Metal is each backend's implementation's business. This abstraction is like fitting all hardware with one universal socket: the upper layer writes to the socket once, and any "appliance" (backend) plugs in. Decoupling "what to compute" from "where to compute" is exactly what lets ggml support a dozen kinds of hardware at once and have them work together. This lesson no longer digs into one hardware's kernels (that was L31/L32) but steps up a level to the "manage + schedule" skeleton. Put more plainly: the last two lessons were a "microscope", peering up close at what one hardware's kernel looks like; this lesson is a "map", looking down on how all the hardware is unified and made to cooperate. Both views are indispensable - without the microscope you do not see where performance comes from; without the map you do not see how this motley pile of hardware is driven by one codebase.
</div>

<div class="card analogy">
  <div class="tag">🔌 Analogy</div>
  Picture the compute graph as a stack of <strong>work orders</strong> (each one an op), and backends as <strong>tradespeople</strong> of various crafts (a CPU hand, a GPU hand, ...). <span class="mono">ggml_backend_sched</span> is the <strong>dispatching foreman</strong>: he picks up each order, sees who suits the job and where the materials (input tensors) currently are, then assigns it to the right hand; if the materials are with the CPU hand but the job needs the GPU hand, he moves them over first. The upper layer just hands the foreman a stack of orders; who gets each one and how materials move is all absorbed by this abstraction + scheduler - which is why, using llama.cpp, you only set <span class="mono">-ngl</span> for how many layers go on the GPU, and the rest - assigning the jobs, moving the materials - is quietly handled for you by this "foreman" in the background; you never have to worry which op runs where, or when a tensor gets moved.
</div>

<h2>The backend abstraction: one uniform interface for all hardware</h2>
<p>A <span class="mono">ggml_backend</span> represents "a device that can run ops", bundling three things: a <strong>device</strong> handle, a set of <strong>buffers</strong> (memory to allocate/read/write tensors on that device), and a set of <strong>op implementations</strong> (how this hardware computes matmul, softmax, ...). Every backend (CPU, CUDA, Metal, ...) implements the same interface (<span class="mono">ggml_backend_i</span> in <span class="mono">ggml-backend-impl.h</span>), so the upper compute graph only writes to this abstraction and never minds which hardware is underneath. The <span class="mono">buffer</span> here is an easily-overlooked key: it decides whether a tensor's data sits in CPU RAM or GPU VRAM; the same tensor is two different chunks of memory in a CPU buffer versus a GPU buffer, and what the scheduler later moves across devices is exactly the data in these buffers. Devices also carry a type, so the upper layer can "pick by need":</p>
<pre class="code"><span class="cm">// device types (ggml-backend.h)</span>
<span class="kw">enum</span> ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU,    <span class="cm">// CPU</span>
    GGML_BACKEND_DEVICE_TYPE_GPU,    <span class="cm">// discrete GPU</span>
    GGML_BACKEND_DEVICE_TYPE_ACCEL,  <span class="cm">// accelerator (used with the CPU, e.g. BLAS/AMX)</span>
    <span class="cm">// ... also IGPU (integrated GPU) / META, etc.</span>
};
<span class="cm">// pick a GPU device; fall back to CPU if none (usage in ggml-backend-reg.cpp)</span>
ggml_backend_dev_t dev = <span class="fn">ggml_backend_dev_by_type</span>(GGML_BACKEND_DEVICE_TYPE_GPU);
<span class="kw">if</span> (!dev) dev = <span class="fn">ggml_backend_dev_by_type</span>(GGML_BACKEND_DEVICE_TYPE_CPU);</pre>
<p>Half the payoff of this design already showed up in L31/L32: when the upper layer calls <span class="mono">matmul</span> it never writes "AVX2 or CUDA" - that lives inside each backend's implementation. The backend abstraction formalizes this: the compute graph (L09/L10) describes "what to do", the backend handles "how and where". Graph and hardware are fully decoupled:</p>
<div class="layers">
  <div class="layer l-app"><div class="lh"><span class="badge">top</span><span class="name">compute graph / model</span></div><div class="ld">describes ops and dependencies only (L09/L10), hardware-agnostic</div></div>
  <div class="layer l-part"><div class="lh"><span class="badge">middle</span><span class="name">ggml-backend abstraction</span></div><div class="ld">uniform interface: device + buffer + a set of op impls (the interfaces in impl.h)</div></div>
  <div class="layer l-core"><div class="lh"><span class="badge">bottom</span><span class="name">backends</span></div><div class="ld">CPU / CUDA / Metal / Vulkan ... each implements the same interface</div></div>
</div>

<h2>Registry and dynamic loading: one binary, backends by hardware</h2>
<p>We have the abstraction, but how does the program know which backends this machine has? The answer is <strong>runtime dynamic loading</strong>. Each backend compiles to a shared library (.so/.dll); at startup the program calls <span class="mono">ggml_backend_load_all()</span> to scan and load the usable backend libraries one by one and register them in a registry. The low-level worker is <span class="mono">dl_load_library</span>:</p>
<pre class="code"><span class="cm">// load all available backends at runtime (ggml-backend-reg.cpp)</span>
<span class="kw">void</span> <span class="fn">ggml_backend_load_all</span>();   <span class="cm">// scan and load cuda/metal/vulkan... shared libs</span>

<span class="cm">// low-level single-library load (ggml-backend-dl.cpp), two platform paths</span>
<span class="cm">// POSIX:</span>
handle = <span class="fn">dlopen</span>(path, RTLD_NOW | RTLD_LOCAL);
<span class="cm">// Windows:</span>
handle = <span class="fn">LoadLibraryW</span>(path);</pre>
<p>Once loaded, the program can ask the registry "how many backends, how many devices now": <span class="mono">ggml_backend_reg_count()</span>, <span class="mono">ggml_backend_dev_count()</span>, then pick a device with last section's <span class="mono">ggml_backend_dev_by_type</span>. The point of this dynamic loading is concrete: <strong>one main binary loads the matching backend at runtime per the machine's actual hardware</strong> - load the CUDA backend if there is a CUDA card, skip it otherwise, never failing to start over a missing library. Think of the registry as a "phone book": each loaded backend writes one entry - "who I am, what devices I have, how to reach me". When the upper layer wants a GPU, it does not go straight to CUDA but looks it up in the book by type - it is exactly this indirection that lets "the upper layer not know the concrete hardware" actually hold.</p>
<div class="card detail">
  <div class="tag">🔬 Why dynamic loading</div>
  Flip it around: if all backends were compiled <strong>statically</strong> into one binary, that binary would have to link CUDA, Vulkan, Metal, SYCL... a pile of huge, mutually conflicting dependencies, and would not run on a machine lacking those libraries. Dynamic loading defers this to runtime: ship one slim main program, and on the user's machine <strong>load whatever backend matches whatever hardware is present</strong>. This is why llama.cpp's prebuilt packages can be "one package, runs everywhere" - on a machine with no NVIDIA card the CUDA backend is simply not loaded, rather than crashing the whole program.
</div>

<h2>Scheduling: dispatching a graph's ops across backends</h2>
<p>With several backends, the last question is: in one compute graph (L09/L10), who decides which backend each op runs on, and who moves tensors between backends? That is <span class="mono">ggml_backend_sched</span>'s (the scheduler's) job. Its two main interfaces:</p>
<pre class="code"><span class="cm">// build a scheduler, hand it a set of backends (ggml-backend.h)</span>
sched = <span class="fn">ggml_backend_sched_new</span>({backend_gpu, backend_cpu}, ...);
<span class="cm">// dispatch the whole compute graph across backends</span>
<span class="fn">ggml_backend_sched_graph_compute</span>(sched, graph);</pre>
<p>Given the graph, the scheduler walks each op: which backend's buffer are its input tensors in now? Which backend best suits this op (matmul prefers GPU, say)? It <strong>assigns</strong> the op to a backend accordingly; if some input is still on another device (the op runs on GPU but the input is still in CPU memory), the scheduler first inserts a <strong>cross-device copy</strong> to move the input over, then executes. When the whole graph is done, results land where they should be. Worth noting: the scheduler does not dispatch each op in <strong>isolation</strong> - cross-device copies are expensive (memory traffic again, echoing L30/L32), so it tries to give consecutive ops that can run on the same device to one backend in <strong>segments</strong>, cutting the back-and-forth. In other words, it weighs not just "where does this op run fastest" but "how to cut this graph so total cross-device movement is least" - which is also why <span class="mono">-ngl</span> typically means "put the first several layers wholesale on the GPU" rather than scattering layers around. Freezing "dispatching one op" into a flow shows it clearest:</p>
<div class="trace">
  <div class="tcap"><b>Trace one op dispatch</b>: the scheduler checks where an op's inputs are, picks a backend, copies across devices if needed, then executes and writes back (illustrative).</div>
  <div class="stations">
    <div class="stn"><h5>1 take op</h5>
      <div class="cellrow"><span class="vc">an op in the graph</span></div>
      <div class="tlab">e.g. a matmul</div></div>
    <div class="op">where are<br>inputs</div>
    <div class="stn"><h5>2 locate</h5>
      <div class="cellrow"><span class="vc">which buffer</span></div>
      <div class="tlab">CPU RAM? VRAM?</div></div>
    <div class="op">pick<br>backend</div>
    <div class="stn"><h5>3 select</h5>
      <div class="cellrow"><span class="vc blue">inputs on GPU -> GPU</span></div>
      <div class="tlab">nearest, fittest</div></div>
    <div class="op">copy if<br>needed</div>
    <div class="stn"><h5>4 move</h5>
      <div class="cellrow"><span class="vc">cross-device copy</span></div>
      <div class="tlab">if input elsewhere</div></div>
    <div class="op">run +<br>write</div>
    <div class="stn"><h5>5 compute</h5>
      <div class="cellrow"><span class="vc hot">backend runs -> output</span></div>
      <div class="tlab">result stays there</div></div>
  </div>
</div>
<p>This is exactly where L09/L10's "static graph" connects to "dynamic execution": the graph describes dependencies, and the scheduler, following them and each tensor's actual location, drops the ops one by one onto concrete hardware. When you use <span class="mono">-ngl N</span> to put the first N layers on the GPU and leave the rest on CPU, this scheduler is what dispatches by layer and shuttles boundary tensors between GPU and CPU. And when the GPU's VRAM truly cannot hold the whole model, this CPU+GPU hybrid execution is often the only way to run at all - at the cost of a few cross-device copies at the boundary, but far better than not running. It also explains why turning <span class="mono">-ngl</span> up or down trades speed against VRAM use: the more layers on the GPU, the faster it computes, but the more VRAM it takes, and the CPU-GPU handoff points shift too.</p>

<h2>A tour of the other backends</h2>
<p>Beyond the CPU (L31) and CUDA (L32) we examined closely, ggml implements a whole crowd of backends covering everyone's hardware. They all follow the same <span class="mono">ggml_backend_i</span> interface, so upper-layer code barely changes - switching hardware just means loading a different backend:</p>
<table class="t">
  <tr><th>Backend</th><th>Target hardware / platform</th><th>Typical use</th></tr>
  <tr><td><span class="mono">Metal</span></td><td>Apple GPU (macOS / iOS)</td><td>the go-to GPU backend on Apple devices</td></tr>
  <tr><td><span class="mono">Vulkan</span></td><td>cross-platform GPU</td><td>vendor-agnostic general GPU acceleration</td></tr>
  <tr><td><span class="mono">SYCL</span></td><td>Intel GPU</td><td>Intel discrete / integrated graphics</td></tr>
  <tr><td><span class="mono">HIP</span></td><td>AMD GPU</td><td>AMD cards (CUDA's counterpart)</td></tr>
  <tr><td><span class="mono">CANN</span></td><td>Huawei Ascend NPU</td><td>Ascend accelerator cards</td></tr>
  <tr><td><span class="mono">OpenCL</span></td><td>cross-platform (incl. mobile GPU)</td><td>mobile / embedded (e.g. Qualcomm Adreno)</td></tr>
  <tr><td><span class="mono">BLAS</span></td><td>CPU (borrowing a math library)</td><td>accelerate CPU matmul via a BLAS library</td></tr>
  <tr><td><span class="mono">RPC</span></td><td>remote machine / process</td><td>send ops to another machine to run</td></tr>
</table>
<p>This table best shows the value of the backend abstraction: from Apple's Metal to Huawei's CANN, from a local GPU to remote RPC, the hardware is worlds apart, yet to the upper layer each is just "a device implementing <span class="mono">ggml_backend_i</span>". Supporting a new piece of hardware is essentially writing one more backend implementation and registering it - not a single line of the upper model code needs to change. In fact, many of these backends were contributed by the community or hardware vendors: precisely because the interface is uniform, Huawei can write CANN themselves and Intel can write SYCL, without touching ggml's core. A good abstract interface opens a door for the whole ecosystem - "you adapt the hardware, I guarantee the upper layer stays put" - and that is the organizational reason an open-source project can support so much hardware so fast.</p>

<h2>Deeper: special backends and how to add one</h2>
<p>Two last folds: two "not very GPU-like" special backends, and roughly what adding a new backend takes.</p>
<details class="accordion">
  <summary><span class="badge-num">1</span> What are "special backends" like BLAS and RPC? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>Most backends correspond to a piece of "compute hardware", but two exceptions are interesting. The <strong>BLAS backend</strong> does not write kernels itself; it hands matmul off to an existing high-performance BLAS math library on the system (OpenBLAS, MKL) - "borrowing someone else's ready-made wheel", and on some CPUs faster than ggml's own implementation. The <strong>RPC backend</strong> is more unusual: it does not compute locally at all but <strong>sends ops over the network to another machine (or process)</strong> to run, then fetches the results back. With it you can split a model too big for one machine across several machines' VRAM and run it distributed. Both wear the same <span class="mono">ggml_backend_i</span> interface, so to the upper layer "borrow a math library" and "send it remote" are no different from "compute on a local GPU" - exactly the power of abstraction. By the way, it is precisely because of a backend like RPC that tricks like "schedule on one machine and send the heavy work to another, or several others" become possible; and the BLAS backend reminds us that a "backend" need not correspond to new hardware at all - it can just be "a faster algorithm implementation". The abstraction only cares "can you implement this interface"; whether behind you is a chip, a math library, or a network cable, it never asks.</p>
  </div>
</details>
<details class="accordion">
  <summary><span class="badge-num">2</span> What does adding a new backend take? <span class="hint">click to expand</span></summary>
  <div class="acc-body">
    <p>To add a new piece of hardware to ggml, the core is implementing the interfaces defined in <span class="mono">ggml-backend-impl.h</span> (a device interface + a backend interface), mainly a few categories: <strong>buffer management</strong> (how to allocate memory on this hardware and copy tensor data in and out), <strong>op-support query</strong> (<span class="mono">supports_op</span>: can this backend compute a given op - if not, the scheduler falls back to another backend), and <strong>running the graph</strong> (<span class="mono">graph_compute</span>: actually compute the chain of ops assigned to me). Implement these, register it, and the upper model and scheduler can use it immediately with no changes. This "define the interface, anyone can plug in" design is why ggml grew a dozen backends in so short a time - and one of the most concrete examples of "program to an interface" in software engineering.</p>
  </div>
</details>

<p>Part 6 closes here. From L31's CPU instructions (scalar, SIMD, multithreading), through L32's CUDA threads (grid/block, tiling, the memory hierarchy), to this lesson's backend abstraction and scheduling, the thread of "how one op finally lands on hardware to compute" has been told end to end. Zooming back out: the model (Part 4) describes what to compute, the compute graph (Part 3) organizes it into dependencies, and the backend (this part) lands the graph on concrete hardware - join the three layers and the whole picture of how llama.cpp runs a model is clear. Next stop, Part 7, where we look at some advanced topics. By here, you have taken llama.cpp from "a black box that runs models" to "a stack of legible layers" - and that is the biggest reward of reading source: however complex, a system is never a tangle but a stack of abstractions, each solving one problem and exposing one clean interface to the layer above. Carry this lens away and you will read any large project far more calmly than before.</p>

<div class="card key">
  <div class="tag">✅ Key points</div>
  <ul>
    <li>The backend abstraction <span class="mono">ggml_backend</span> = a device that can run ops (device + buffer + a set of op impls); every backend implements the same <span class="mono">ggml_backend_i</span>, and the upper compute graph ignores hardware.</li>
    <li>Registry and dynamic loading: <span class="mono">ggml_backend_load_all</span> uses <span class="mono">dlopen</span> / <span class="mono">LoadLibraryW</span> at runtime to load backend shared libs - one binary picks backends by actual hardware.</li>
    <li>The scheduler <span class="mono">ggml_backend_sched</span>: dispatches a compute graph's (L09/L10) ops across backends and copies tensors between devices as needed.</li>
    <li>Other backends: Metal / Vulkan / SYCL / HIP / CANN / OpenCL / BLAS / RPC - one interface, covering Apple to Huawei, local to remote.</li>
    <li>Adding a backend = implement the <span class="mono">ggml-backend-impl.h</span> interfaces (buffer, <span class="mono">supports_op</span>, <span class="mono">graph_compute</span>) and register it.</li>
  </ul>
</div>

<div class="card spark">
  <div class="tag">💡 Design insight</div>
  This lesson is really about one of software engineering's most classic moves: <strong>use a layer of abstraction to separate "what changes" from "what stays"</strong>. Hardware changes every year (new GPUs, new NPUs, new instruction sets), but "the compute graph describes what to compute" is stable. ggml inserts the <span class="mono">ggml_backend</span> interface between the two: the model code above need not change for a decade, and hardware below can be added at will. L31/L32 showed you "how a kernel fills the hardware"; this lesson showed you "how the framework unifies a crowd of different hardware" - the former is depth, the latter breadth, and together they are the whole secret of llama.cpp being both fast and runnable everywhere. Carry this "layering + abstraction" lens back over the whole guide and you will see it everywhere: the tokenizer for text, GGUF for weights, the compute graph for ops, the backend for hardware - each layer hiding complexity behind a clean interface.
</div>
""",
}
