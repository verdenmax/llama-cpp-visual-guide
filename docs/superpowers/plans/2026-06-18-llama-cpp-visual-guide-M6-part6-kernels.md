# M6 · 第六部分（底层内核）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development 逐课执行本计划（每课一个 task，task 内步骤用 `- [ ]` 勾选）。
> **配套设计：** `docs/superpowers/specs/2026-06-18-m6-part6-kernels-design.md`（父级总设计/roadmap 见其顶部）。

**Goal:** 为图解教程补齐**第六部分 · 底层内核**（课 31-33，共 3 课），把"算子最终在硬件上怎么算"讲透：CPU 的 SIMD 与量化矩阵乘、CUDA 的 kernel 与分块矩阵乘、后端注册/动态加载/调度与其它后端概览。**硬核路线**：逐行讲解真实 SIMD/CUDA 代码（必要处最小简化），并用大量图把它讲透。

**Architecture:** 沿用既有零依赖 Python 静态站点生成器。**新建** `src/part6.py`，写 `LESSON_31..33`；每课在 `src/shell.py`（`PAGES`/`SUBTITLES`）、`src/registry.py`（`CONTENT`）、`src/quizzes.py`（`QUIZZES`）登记；`python3 src/build.py` 生成 HTML，`check_html.py`/`check_links.py` 校验。index 课数 30 -> 33、部分数 5 -> **6**（新增"第六部分"）。trace 复用既有 `.trace` 组件（Style A 纯 HTML / Style C 内联 SVG）。

**Tech Stack:** Python 3（仅标准库）· 自包含 HTML/CSS/JS · 既有 `.trace` 组件 · 关键 SVG 用 Python 预生成校验。

---

## 里程碑范围（M6 = 课 31-33）

| 课 | slug（产出文件） | 标题 zh / en | 内嵌 trace（风格） |
| --- | --- | --- | --- |
| 31 | `31-cpu-backend.html` | CPU 后端 / The CPU backend | ① SIMD 点积多通道并行（C·SVG）；② 量化点积一格（A） |
| 32 | `32-cuda-backend.html` | CUDA 后端 / The CUDA backend | 分块矩阵乘 tiling（C·SVG） |
| 33 | `33-backends-dispatch.html` | 后端调度与其它后端 / Backends & dispatch | 一次 op 分派到某后端（A·站点流） |

> 全部在"第六部分 · 底层内核 / Part 6 · Low-level kernels"标签下。**范围取舍**（已与用户确认）：L32 的 flash attention 仅概念+伪代码（不逐行真实 kernel）；L33 其它后端为概览（注册/动态加载/调度上真实代码，后端动物园点到为止）。

## 统一交付标准（每课硬性达标，与 M1-M5 一致）

每课 Step 4 写 `LESSON_NN = {"zh": r'''...''', "en": r'''...'''}`，须满足：

- **结构**：导语 `<p>` + 教学卡片（`macro`/`detail`/`analogy`/`key`/`spark` 酌用，≥2 张深挖 `<details>`）+ **≥3 个图示**（`flow`/`vflow`/`cols`/`cellgroup`/`layers`/`timeline`/`trace` 之一，单语 ≥3，含 trace）+ **≥2 段真实/简化代码**（`<pre class="code">`）+ **1 个内嵌 worked-example trace**（见各课）。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace` div 与内联 `<svg>` 不计入 `<p>`）。
- **中文密度**：zh CJK ≥ 4000（注意：`<span class="mono">`/代码/英文术语/标点都不计入 CJK，须写得足够"实"，目标 ~4200-4400 一次过）；**en CJK == 0**（纯 ASCII；代码/SVG 文本用 `-`/`->`/`...`/`~=`/`+/-`，不用 em-dash/unicode 箭头/`≈`/`±`）。
- **无文字墙**：连续顶层 `<p>` ≤ 3（遇墙用图/卡片/trace 打断；注意 vflow/details 里的 `<p>` 不是顶层）。
- **转义**：渲染 `<`/`>`/`&` 转义（`&lt;`/`&gt;`/`&amp;`）；无双重转义（`&amp;lt;`）。代码里 `&`/`<`/`>` 同样转义（如 `a &amp; b`、`x &lt;&lt; 2`）。
- **trace 规范**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn/.cellrow/.vc[.hot/.blue/.dim]/.op/.tlab`，无 `<svg>`）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>` + `<g font-family="ui-monospace,monospace">`，合法 XML，字面色板（白 `#ffffff`/text `#1d2129`、accent `#c2630e`/`#fff`、blue `#2563eb`、purple `#7c3aed`、label `#5b6470`、stroke `#cdd5df`、muted `#9aa6b2`），zh aria-label 中文、en aria-label + 所有 `<text>` 纯 ASCII；trace 不与 `<div class="card">` 紧邻（中间须有 `<p>`/`<h2>`）。
- **源码引用**：以"文件 + 符号名"为主、不写死行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验日期 2026-06-18）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测（沿用既有 `QUIZZES` 格式：`mcq` 列表，每题 `q`/`opts`(4)/`answer`/`why` 双语 + `open` 列表）。
- **登记**：`shell.PAGES`（filename、zh/en 短标题、`第六部分 · 底层内核`/`Part 6 · Low-level kernels`）、`shell.SUBTITLES`、`registry.CONTENT`（filename -> `part6.LESSON_NN`，registry 顶部加 `import part6`）。

## 执行方式

- superpowers:subagent-driven-development，**一课一个 task**（Task 1=课31 ... Task 3=课33；Task 4 收尾）。
- 每个 task：实现子代理 -> **spec 合规审查子代理 -> 质量审查子代理**（两段审查），修复回环后再标完成。子代理一律用当前主会话模型，显式传 `model`。
- **M5 经验**：后台 general-purpose 子代理写整课常中途失败（"completed"却零文件写入，疑似超时被终止）。控制器**每次都要独立核验 git 状态**（不信报告）；若实现子代理失败，则由控制器亲自照 LESSON 模板执笔，仍跑完整 spec+质量双重审查。
- **关键 Style-C SVG（L31 SIMD 点积、L32 分块矩阵乘）**：先用 Python 预生成并校验（well-formed XML + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 渲染目检），再喂给实现（沿用 M5 L28 的做法）。
- **HTML 是被 git 跟踪的**：每课提交须 `git add` 4 个源文件 **+ 重建后的全部 HTML**（`index.html` + `lessons/*.html`），提交后 `git status` 必须干净。
- 全程对照真实源码；commit 用 `Assisted-by: GitHub Copilot`（非 Co-authored-by）。
- 分支：在 master 上从本 plan 提交后，新建 `feature/part6-kernels` 分支做实现。

---


## Task 1: 课 31「CPU 后端 / The CPU backend」

**Files:** **新建** `src/part6.py`（写 `LESSON_31`，文件头 `"""Content for Part 6 (low-level kernels)."""`）、改 `src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `31-cpu-backend.html`。

**源码事实（核实 2026-06-18，`/home/verden/course/llama.cpp`，引用文件+符号、无行号）：**
- 后端目录 `ggml/src/ggml-cpu/`：`ggml-cpu.c`/`.cpp`（后端入口、逐算子 `ggml_compute_forward_*`、线程切分）；`quants.c`（量化点积的 generic 参考实现）；`arch/{x86,arm,...}/quants.c`（架构特化 SIMD intrinsics）；`ops.cpp`、`repack.cpp`、`llamafile/sgemm.cpp`（分块 GEMM）；`ggml-threading.{h,cpp}`。
- **量化块布局**（`ggml-common.h`）：`block_q4_0` = `ggml_half d`（缩放/delta）+ `uint8_t qs[QK4_0/2]`（16 字节，打包 32 个 4-bit 量化值），`QK4_0 == 32`；`block_q8_0` = `ggml_half d` + `int8_t qs[32]`，`QK8_0 == 32`。即权重每 32 个一组、共享一个 fp16 scale。
- **标量参考**（`quants.c` `ggml_vec_dot_q4_0_q8_0_generic`）：逐元素解包 4-bit、减 8 偏移、乘以 Q8_0 的 int8、乘 scale 累加——一次只算一个数。
- **真实 AVX2 SIMD**（`arch/x86/quants.c` `ggml_vec_dot_q4_0_q8_0`，`#if defined(__AVX2__)`），核心循环（一次处理一个 block 的 32 个量化值，8 通道并行）：
  ```c
  __m256 acc = _mm256_setzero_ps();                       // 8 路 float 累加器
  for (; ib < nb; ++ib) {                                 // 遍历 block (每块 32 权重)
      const __m256 d = _mm256_set1_ps(d_x * d_y);         // 合并两个块的 scale
      __m256i qx = bytes_from_nibbles_32(x[ib].qs);       // 解包: 16 字节 -> 32 个 [0..15]
      qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));      // 偏移到 [-8..+7]
      __m256i qy = _mm256_loadu_si256((const __m256i*)y[ib].qs); // 载入 32 个 int8 激活
      const __m256 q = mul_sum_i8_pairs_float(qx, qy);    // int8 点积 -> float
      acc = _mm256_fmadd_ps(d, q, acc);                   // FMA: acc += d * q
  }
  sumf = hsum_float_8(acc);                               // 8 路水平求和 -> 一个标量
  ```
  关键 intrinsic：`_mm256_setzero_ps`（清零）、`_mm256_fmadd_ps`（一条指令同时做 8 个乘加）、`bytes_from_nibbles_32`（4-bit 解包，用 `_mm256_set1_epi8(0xF)` 掩码 + 移位）、`hsum_float_8`（水平求和）。NEON 对应：`vdupq_n_f32`/`vmlaq_f32`/`vld1q_*`/`vaddvq_f32`（4 路并行）。
- **多线程**（`ggml-cpu.c`）：每个算子 `ggml_compute_forward_*` 拿到 `params->ith`（线程号）/`params->nth`（线程数），按输出的行/块把工作切给各线程（GEMM 各行独立、天然可并行）；底层用 `ggml-threading` 的线程池。
- **分块 GEMM**（`llamafile/sgemm.cpp`、`repack.cpp`）：把大矩阵乘按 tile 分块，提升缓存命中（朴素三重循环反复扫内存，慢）。

> **控制器预研（实现前完成）**：用 Python 预生成并校验 trace ① 的 Style-C SVG（"SIMD 点积 8 通道并行"），存到 session files；校验合法 XML + 英文 ASCII + 坐标不溢出 viewBox + 深色可读 + `rsvg-convert` 目检，再把两份 SVG（zh/en）逐字喂给实现子代理。

- [ ] **Step 1-3: 登记**（三处 + import）

```python
# registry.py 顶部: import part6  (新增)
# registry.py CONTENT 追加:
"31-cpu-backend.html": part6.LESSON_31,
# shell.py PAGES 追加（第六部分起点）:
("31-cpu-backend.html", "CPU 后端", "The CPU backend", "第六部分 · 底层内核", "Part 6 · Low-level kernels"),
# shell.py SUBTITLES 追加:
"31-cpu-backend.html": ("ggml-cpu：从标量到 SIMD、量化点积、多线程", "ggml-cpu: scalar to SIMD, quantized dot product, multithreading"),
```

- [ ] **Step 4: 执笔 `LESSON_31`（双语，新建 `src/part6.py`）。结构：**
  - 导语 `<p>`：算子（L11）讲了"做什么矩阵乘"，这一课讲它最终怎么落到 CPU 指令上算——从"标量一个个乘"加速到"SIMD 一条指令算一排"，再切给多线程。
  - `<h2>` 从标量到 SIMD：先看 generic 标量点积（一次一个数）；再看 SIMD 把它向量化——一条 `_mm256_fmadd_ps` 同时算 8 个（AVX2 `__m256`，256 位 / 32 位 = 8 个 float），NEON 一次 4 个。**真实代码片段**（标量循环 vs SIMD 循环对照，注明 intrinsic）。
    - **trace ①（Style C·SVG，控制器预生成）**："追踪一次 SIMD 点积"——画 8 条 lane 同时把 `a[i]*b[i]` 累加进 8 个累加器，循环若干轮后水平求和（hsum）成一个标量。
  - `<h2>` 量化点积怎么算：以 `vec_dot_q4_0_q8_0` 为例——权重每 32 个一组打包成 16 字节 4-bit（`block_q4_0`，共享一个 fp16 scale），激活是 Q8_0 的 int8。**真实 AVX2 循环逐行**（上面那段）：解包 nibble -> 偏移 -> 载入激活 -> int8 点积 -> `fmadd` 累加 -> 末尾 hsum，再乘 scale。
    - **trace ②（Style A·站点流）**："追踪一个 block"：`16 字节(4-bit 打包)` -> 解包成 `32 个量化值[0..15]` -> `-8 偏移[-8..7]` -> `× int8 激活点积` -> `× scale 累加`。
  - `<h2>` 多线程：`ggml_compute_forward_*` 用 `ith/nth` 把输出行切给 N 个线程（各行独立）；为什么 GEMM 适合数据并行。**图**：一个矩阵乘按行切给 4 个线程（`cols`/`cellgroup`/`layers`）。
  - `<h2>` 折叠深挖（`<details>` ≥2）：(1) `llamafile/sgemm` 的分块（tiling）为什么比朴素三重循环快——缓存友好、复用已载入的数据；(2) `arch/` 怎么按 CPU 特性选实现（编译期 `#if defined(__AVX2__)` + 运行时特性检测），一份代码多架构。
  - 硬性：zh CJK≥4000、en CJK==0、逐节对齐、≥3 图（含 2 trace）、≥2 深挖、≥2 真实代码片段。

- [ ] **Step 5: quiz（31）** 2-4 题：「SIMD 一条指令为什么能加速点积？（一次算 8/4 个，数据并行）」「`block_q4_0` 里 32 个权重共享什么？（一个 fp16 scale）」「`_mm256_fmadd_ps` 一次做几个乘加？（8 个 float）」「GEMM 为什么好并行？（各输出行独立）」。

- [ ] **Step 6: 重建+校验**：`python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py` 全绿；index 变"共 31 课 · **6** 个部分"；硬性达标；trace=2（`class="trace"`）、`<svg`=2（仅 trace ① 的 zh/en）、两 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII；grep 渲染无 `&amp;lt;` 双重转义、无裸 `<`（代码里 `&lt;&lt;`/`&amp;` 正确转义）。

- [ ] **Step 7: commit**：`feat: add lesson 31 CPU backend (bilingual) with SIMD trace + quiz` + `Assisted-by: GitHub Copilot`（暂存 4 源文件 + 重建的全部 HTML，提交后 git status 干净）。

---

## Task 2: 课 32「CUDA 后端 / The CUDA backend」

**Files:** `src/part6.py`（追加 `LESSON_32`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `32-cuda-backend.html`。

**源码事实（核实 2026-06-18）：**
- 后端目录 `ggml/src/ggml-cuda/`：`ggml-cuda.cu`（后端入口、把算子派给对应 kernel）；众多 `*.cu`/`*.cuh` 真实 kernel；`mmq.cuh`/`mmq.cu`（量化分块矩阵乘）、`mmvf.cu`/`mmvq.cu`（矩阵×向量）、`fattn*.cu`（flash attention）。
- **真实 kernel 骨架**（`scale.cu` `scale_f32`，最干净的范例）：
  ```c
  static __global__ void scale_f32(const float * x, float * dst, const float scale,
                                   const float bias, const int64_t nelements) {
      int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程号
      int64_t stride = (int64_t)blockDim.x * gridDim.x;                 // grid 总线程数
      for (int64_t i = tid; i < nelements; i += stride)                 // grid-stride 循环
          dst[i] = scale * x[i] + bias;                                 // 每个线程算它那份
  }
  ```
  要点：`__global__` 标记 GPU 入口；`blockIdx.x*blockDim.x + threadIdx.x` 是"我是第几个线程"的通用公式；成千上万线程各算一份，互不等待。
- **执行模型层级**：grid（整次发射）-> block（一组线程，可共享 `__shared__` 片上内存、可 `__syncthreads()` 同步）-> thread；warp = 32 个线程齐步走（SIMT）。kernel 发射写 `kernel&lt;&lt;&lt;gridDim, blockDim&gt;&gt;&gt;(...)`（渲染时 `<<<`/`>>>` 须转义）。
- **分块矩阵乘 tiling**（`mmq.cuh`，真实用 `__shared__` 缓存 tile + `__syncthreads`，但模板很重）。**教学用规范化骨架**（真实 CUDA tiled-GEMM 范式，mmq 在其上加解包量化）：
  ```c
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];   // 片上共享内存里的两块 tile
  float acc = 0;
  for (int k0 = 0; k0 < K; k0 += TILE) {             // 沿 K 维一块一块来
      As[ty][tx] = A[row*K + (k0+tx)];               // block 内线程协作把 tile 从
      Bs[ty][tx] = B[(k0+ty)*N + col];               // 慢速 global 载入快速 shared
      __syncthreads();                               // 等所有线程载完
      for (int k = 0; k < TILE; ++k)                 // 用 shared 里的数据复用计算
          acc += As[ty][k] * Bs[k][tx];
      __syncthreads();                               // 等所有线程算完再载下一块
  }
  C[row*N + col] = acc;                              // 每个 thread 写 C 的一格
  ```
  为什么快：tile 一旦载入 `__shared__`，block 里的线程**反复复用**，避免每次都去碰慢几十倍的 global 显存。
- **显存层级**：global（大、慢，GB 级）-> `__shared__`（每 block 几十 KB、快）-> register（每 thread、最快）。访存常比计算更贵（呼应 L30 decode 的"访存密集"）。
- **flash attention（仅概念，不逐行）**：`fattn*.cu` 共上千行、极复杂。要点：把 `softmax(QK^T)V` 这串运算**融合成一个 kernel**，用"分块在线 softmax"边算边累加，**避免把 N×N 的注意力分数矩阵写回 global 显存**——省显存、省带宽。给**简化伪代码**示意在线 softmax，并 `card` 指向 `fattn*.cu` 说明真实实现为何复杂（多 GPU 架构、多精度变体）。

> **控制器预研（实现前完成）**：用 Python 预生成并校验 trace 的 Style-C SVG（"分块矩阵乘 tiling"：一个 thread block 把 A 的行 tile、B 的列 tile 载入 shared memory，block 内每个 thread 算 C 的一格，沿 K 维循环累加）；校验同 Task 1，再逐字喂给实现。

- [ ] **Step 1-3: 登记**

```python
# registry.py CONTENT 追加（import part6 已在 Task 1 加过）:
"32-cuda-backend.html": part6.LESSON_32,
# shell.py PAGES:
("32-cuda-backend.html", "CUDA 后端", "The CUDA backend", "第六部分 · 底层内核", "Part 6 · Low-level kernels"),
# shell.py SUBTITLES:
"32-cuda-backend.html": ("ggml-cuda：线程网格、分块矩阵乘、显存层级", "ggml-cuda: thread grid, tiled matmul, the memory hierarchy"),
```

- [ ] **Step 4: 执笔 `LESSON_32`（双语）。结构：**
  - 导语：CPU 靠几个核 + SIMD（L31）；GPU 靠**成千上万个线程同时算**取胜。这一课看 ggml 的 CUDA kernel 怎么组织这些线程，把一次矩阵乘拆成 block/thread 的协作。
  - `<h2>` CUDA 执行模型：grid -> block -> thread 三层 + warp（32 线程齐步）+ `__shared__` 片上内存。**真实 kernel 骨架**（`scale_f32`，逐行讲 `blockIdx*blockDim+threadIdx` 与 grid-stride）。**图**：线程网格层级（`layers`/`cellgroup`：grid 含多个 block、block 含多个 thread）。
  - `<h2>` 分块矩阵乘（核心）：朴素 kernel 每算一格都去碰 global 显存，太慢；**tiling** 把 A、B 的小块先搬进 `__shared__`，block 内线程复用，再 `__syncthreads` 协同。**规范化真实骨架**（上面那段，逐行）；说明 `mmq.cuh` 在此之上加了 4-bit 解包。
    - **trace（Style C·SVG，控制器预生成）**："追踪一次分块矩阵乘"：一个 thread block 把 A 的行 tile、B 的列 tile 载入 shared memory（着色），block 内每个 thread 负责输出 C 的一格，沿 K 维循环累加。
  - `<h2>` 显存与搬运：global / shared / register 三级（大慢 -> 小快 -> 最快）；为什么"搬数据"常比"算"还贵（带宽瓶颈，呼应 L30）。**图**：显存层级金字塔 + 相对速度（`layers`/`cols`）。
  - `<h2>` flash attention（仅概念）：为什么把 softmax+matmul **融合**成一个 kernel——避免把巨大的 N×N 注意力分数写回显存。**简化伪代码**（分块在线 softmax，不逐行真实 kernel）。**card**：指向 `fattn*.cu`，真实复杂度来自多架构/多精度变体。
  - `<h2>` 折叠深挖（≥2）：(1) warp 内 reduction（`__shfl_down_sync` 等）怎么在 32 线程间快速求和（无需 shared）；(2) 为什么 CUDA kernel 这么多变体（`mmq`/`mmvq`/不同量化/不同 GPU 计算能力）——为各情形挑最快实现。
  - 硬性同 Task 1。

- [ ] **Step 5: quiz（32）** 2-4 题：「`blockIdx.x*blockDim.x+threadIdx.x` 算的是什么？（全局线程号）」「tiling 把 tile 搬进 `__shared__` 是为了什么？（复用、少碰慢的 global 显存）」「`__syncthreads()` 干嘛的？（block 内线程同步，等 tile 载完/算完）」「flash attention 融合 kernel 主要省了什么？（不把 N×N 注意力分数写回显存）」。

- [ ] **Step 6: 重建+校验**（同 Task 1；index "共 32 课 · 6 个部分"；trace=2、`<svg`=2（仅 tiling trace 的 zh/en）、合法 XML、英文 SVG 纯 ASCII；CUDA 代码里 `<<<`/`>>>` 用 `&lt;&lt;&lt;`/`&gt;&gt;&gt;` 转义，确认渲染无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 32 CUDA backend (bilingual) with tiling SVG trace + quiz` + `Assisted-by: GitHub Copilot`。

---

## Task 3: 课 33「后端调度与其它后端 / Backends & dispatch」

**Files:** `src/part6.py`（追加 `LESSON_33`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `33-backends-dispatch.html`。

**源码事实（核实 2026-06-18）：**
- 抽象层（`ggml/include/ggml-backend.h` + `ggml/src/ggml-backend.cpp`）：`ggml_backend_dev_t`（一个"能跑算子的设备"句柄）；`enum ggml_backend_dev_type { GGML_BACKEND_DEVICE_TYPE_CPU, ..._GPU, ..._ACCEL }`；`ggml_backend_dev_name`/`ggml_backend_dev_description`。每个后端实现一组统一接口（`ggml-backend-impl.h` 的 `ggml_backend_i`：分配 buffer、跑计算图等），上层只对着抽象层写、不管具体是 CPU 还是哪种 GPU。
- 注册表（`ggml-backend-reg.cpp`）：`ggml_backend_reg_count()`、`ggml_backend_dev_count()`、`ggml_backend_dev_by_type(type)`（按类型挑设备；找 GPU 找不到会回退 CPU）。
- **动态加载**（`ggml-backend-dl.cpp` + `ggml-backend-reg.cpp`）：`ggml_backend_load_all()` 在运行时扫描并加载各后端动态库；底层 `dl_load_library()` 在 POSIX 上是 `dlopen(path, RTLD_NOW | RTLD_LOCAL)`、在 Windows 上是 `LoadLibraryW`。意义：**一份主程序二进制**，按机器实际硬件在运行时加载对应后端（有 CUDA 卡就加载 CUDA 后端，没有就不加载），不必为每种硬件单独编译发布。
- **调度**（`ggml-backend.cpp` 的 `ggml_backend_sched`）：`ggml_backend_sched_new(...)` 建调度器，`ggml_backend_sched_graph_compute(sched, graph)` 把一张计算图（L09/L10）的各算子分派到不同后端执行，并在后端之间按需拷贝张量（比如某算子在 GPU、它的输入还在 CPU buffer，就先拷过去）。
- **其它后端（概览，不逐行）**：`ggml-metal`（Apple GPU，Metal）、`ggml-vulkan`（跨平台 GPU，Vulkan）、`ggml-sycl`（Intel GPU，SYCL）、`ggml-cann`（华为昇腾 NPU）、`ggml-hip`（AMD GPU，HIP）、`ggml-opencl`、`ggml-blas`（用 BLAS 库做 CPU 矩阵乘）、`ggml-rpc`（远程后端，把算子发到另一台机器/进程跑）。

> **trace 为 Style A（无 SVG），本课无需控制器预生成 SVG。**

- [ ] **Step 1-3: 登记**

```python
# registry.py CONTENT 追加:
"33-backends-dispatch.html": part6.LESSON_33,
# shell.py PAGES:
("33-backends-dispatch.html", "后端调度", "Backends & dispatch", "第六部分 · 底层内核", "Part 6 · Low-level kernels"),
# shell.py SUBTITLES:
"33-backends-dispatch.html": ("ggml-backend：抽象层、注册与动态加载、调度、其它后端一览", "ggml-backend: the abstraction, registry & dynamic load, scheduling, the backend zoo"),
```

- [ ] **Step 4: 执笔 `LESSON_33`（双语）。结构：**
  - 导语：前两课讲了 CPU（L31）、CUDA（L32）各自怎么算；可一台机器上常同时有 CPU + 一或多块 GPU，ggml 怎么统一管理、把每个算子派到合适的后端？这一课看 `ggml-backend` 抽象层——也是第六部分的收束。
  - `<h2>` 后端抽象：`ggml_backend` 是"一个能跑算子的设备"的**统一接口**（device + buffer + 一组算子实现）。上层计算图只对着抽象层，具体哪种硬件由各后端实现。**图**：抽象层把"计算图"与"各家后端（CPU/CUDA/Metal/...）"解耦（`layers`/`cols`）。**代码片段**：`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)`（找 GPU，回退 CPU）。
  - `<h2>` 注册与动态加载（真实代码）：`ggml_backend_load_all()` 运行时扫描、`dl_load_library`（`dlopen`/`LoadLibraryW`）把各后端动态库加载进来、登记进注册表。**代码片段**（简化的加载流程）。**card**：为什么用动态加载——一份二进制按实际硬件加载对应后端。
  - `<h2>` 调度：`ggml_backend_sched` 怎么把一张计算图的算子分派到不同后端、并在后端之间搬运张量（呼应 L09/L10）。
    - **trace（Style A·站点流）**："追踪一次 op 分派"：`计算图里一个算子` -> `调度器看它的输入张量在哪个 buffer/设备` -> `选后端（输入在 GPU 就派 GPU）` -> `（必要时跨设备拷贝输入）` -> `在该后端执行、写回输出`。
  - `<h2>` 其它后端一览（概览）：Metal / Vulkan / SYCL / CANN / HIP / OpenCL / BLAS / RPC 各是什么、何时用。**图/表**：后端 × 平台/硬件 × 用途（`table.t` 或 `cellgroup`）。
  - `<h2>` 折叠深挖（≥2）：(1) BLAS 后端、RPC 后端这种"特殊后端"（一个借现成数学库、一个把算子发去远程机器）；(2) 加一个新后端大致要实现哪些接口（指向 `ggml-backend-impl.h` 的 `ggml_backend_i`：buffer 类型、算子支持查询、`graph_compute` 等）。
  - 收尾 `<p>`：第六部分到此收束——从 CPU 指令、CUDA 线程，到后端抽象与调度，"算子最终怎么在硬件上算"讲完了；下一站第七部分讲进阶专题。
  - 硬性同 Task 1（但 trace 为 Style A，无 `<svg>`）。

- [ ] **Step 5: quiz（33）** 2-4 题：「`ggml_backend` 抽象层解决什么问题？（统一接口，让上层不管具体硬件）」「`ggml_backend_load_all` 为什么用动态加载（`dlopen`）？（一份二进制按实际硬件加载对应后端）」「`ggml_backend_sched` 干嘛的？（把计算图算子分派到各后端、按需跨设备搬张量）」「RPC 后端是什么？（把算子发到远程机器/进程执行）」。

- [ ] **Step 6: 重建+校验**（同 Task 1，但本课 **Style A、无 `<svg>`**：trace=2、`<svg`=0；index "共 33 课 · 6 个部分"；硬性达标；grep 渲染无双重转义）。

- [ ] **Step 7: commit**：`feat: add lesson 33 backends & dispatch (bilingual) with trace + quiz` + `Assisted-by: GitHub Copilot`。

---


## Task 4: 收尾（roadmap 勾选 + 全量验证 + 完成分支）

**Files:** `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md`（勾 M6）。

- [ ] **Step 1: 更新 roadmap**：里程碑总表 M6 行"状态"`待写` -> `完成`；状态追踪 `- [ ] M6 ...` -> `- [x] M6 ...`。commit `docs: mark M6 (part 6) done`。
- [ ] **Step 2: 全量验证**（master 合并前在分支上跑）：
  - `python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py` = `Wrote 34 files`、`structural check passed`（0 error/warning）、`all internal links resolve`。
  - index 显示"共 **33** 课 · **6** 个部分"；第六部分 3 课都在导航里。
  - 第六部分 3 课（31-33）：按 `<h2>` 分节 `<p>` 计数中英相等；zh CJK ≥ 4000、en CJK == 0、en 无 unicode（`≈`/`±`/箭头/em-dash/`…`）；无连续顶层 `<p>` > 3；无 `&amp;lt;`/`&amp;gt;`/`&amp;amp;` 双重转义。
  - 第六部分新增 trace：`class="trace"` 全站净增 6（3 课 × 2 语言）；`<svg` 净增 4（L31 SIMD 点积 + L32 tiling，各 zh/en）；新增 `<svg>` 均 `xml.dom.minidom` 可解析、英文 SVG 区纯 ASCII、深色色板。
- [ ] **Step 3: 第六部分整体复审**（建议）：派一个 superpowers:code-reviewer 子代理（当前模型）复审 `master..HEAD` 的 3 课跨课一致性（标题/卡片/图/trace 风格统一、真实源码引用准确、双语纪律、范围未越界——只改 part6 + 登记，不碰 1-30 课内容/build 基础设施/其它部分；flash attention 仅概念、L33 概览的取舍已落实）。
- [ ] **Step 4: 完成分支**：用 superpowers:finishing-a-development-branch，先过验证门，再按用户选择（历史偏好：本地 `--no-ff` 合并 master + 删分支）。

---

## 计划自审（writing-plans self-review）

- **Spec 覆盖**：设计 §每课设计 的 L31/L32/L33 三课 -> Task 1/2/3；统一交付标准 -> 各 task Step 4 硬性 + Step 6 校验；roadmap 勾选/全量验证/完成分支 -> Task 4。两个范围取舍（flash attention 仅概念、L33 概览）在 Task 2/Task 3 的源码事实与结构里均落实。✓ 无遗漏。
- **占位符扫描**：无 TBD/TODO；每个 task 的源码事实含**真实代码**（AVX2 vec_dot、CUDA scale_f32 + tiled-GEMM 骨架、backend dl/sched 符号），登记给出**精确字符串**，结构给出**具体 `<h2>` + trace 规格**。✓
- **类型/命名一致**：`part6.LESSON_31..33`、`import part6`、`31-cpu-backend/32-cuda-backend/33-backends-dispatch.html`、Part 标签"第六部分 · 底层内核 / Part 6 · Low-level kernels" 在各 task 与登记中一致。✓
- **风险点**：(1) 硬核真实代码的转义——CUDA `<<<>>>`、C 位运算 `<<`、`&` 须在 r-string 里写成 `&lt;`/`&amp;`，已在 Step 6 校验项点名；(2) zh CJK 密度——硬核课 `<span class="mono">`/代码占比高，CJK 易偏低，已在统一交付标准提醒"写实、目标 4200-4400 一次过"；(3) 子代理写整课可能失败——执行方式已写明"独立核验 git + 失败则控制器亲自执笔"；(4) 关键 SVG 预生成——Task 1/2 已写明控制器先 Python 预生成校验。


