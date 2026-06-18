# M6 · 第六部分（底层内核）设计文档

> **配套：** 父级总设计 `2026-06-13-llama-cpp-visual-guide-design.md`（§第六部分）· roadmap `2026-06-13-llama-cpp-visual-guide-roadmap.md`（M6 行）。
> **执笔基线：** 沿用 M1-M5 的零依赖 Python 静态站点生成器与双语机制；本部分新建 `src/part6.py`，写 `LESSON_31..33`。

## Goal

为图解教程补齐**第六部分 · 底层内核（硬核）**，共 **3 课（31-33）**，把"算子最终在硬件上怎么算"讲透：CPU 的 SIMD 与量化矩阵乘、CUDA 的 kernel 与分块矩阵乘、以及后端注册/动态加载/调度与其它后端概览。视角从前五部分的"对外用法"下沉到"底层实现"。

## 总基调（与用户确认）

- **硬核路线（depth=B）**：展开并**逐行讲解真实的 SIMD intrinsics（AVX2/NEON）和 CUDA kernel 代码**，目标是看懂真实 kernel 实现；必要处做**最小简化**但保持真实，并标注来源文件+符号。
- **用图把硬核讲透**：底层内核天然图形化（SIMD 通道、CUDA 线程网格、显存分块），关键处用**手绘 SVG（Style C）**，调度类用**站点流（Style A）**。
- **CPU/CUDA 为主线，其它后端做概览**（父设计 §约束）。
- 沿用 M1-M5 双语基线（见下"统一交付标准"）。

## 范围取舍（与用户确认）

1. **flash attention 不逐行**：L32 只对 **dequant + 分块矩阵乘（tiled matmul）**上真实 kernel；flash attention 作"**为什么要融合**"的概念 + 简化伪代码（真实 fattn kernel 过于复杂，逐行会让 L32 失控）。
2. **L33 其它后端是概览**：对**后端注册 / 动态加载 / 一次 op 的分派**上真实代码；Metal/Vulkan/SYCL/CANN/HIP/OpenCL/RPC 各"是什么、何时用"点到为止，不逐行。

## 里程碑范围（M6 = 课 31-33）

| 课 | slug（产出文件） | 标题 zh / en | 内嵌 trace（风格） |
| --- | --- | --- | --- |
| 31 | `31-cpu-backend.html` | CPU 后端 / The CPU backend | ① SIMD 点积"多通道并行"（C·SVG）；② 量化矩阵乘一格（A） |
| 32 | `32-cuda-backend.html` | CUDA 后端 / The CUDA backend | 分块矩阵乘 tiling：block 载 tile 入 shared、thread 算一格（C·SVG） |
| 33 | `33-backends-dispatch.html` | 后端调度与其它后端 / Backends & dispatch | 一次 op 从计算图分派到某后端执行（A·站点流） |

> 全部在"第六部分 · 底层内核 / Part 6 · Low-level kernels"标签下。

## 每课设计

### 课 31 · CPU 后端（`ggml/src/ggml-cpu/`）

**源码锚点（核实 2026-06-18）：** `ggml-cpu.c`/`.cpp`（后端入口、线程池）；`quants.c` `ggml_vec_dot_q4_0_q8_0_*`（量化点积）；`arch/`（x86/arm 等架构特化的 SIMD intrinsics，`_mm256_*` / `vmlaq_*`/`vdotq_*`）；`ops.cpp`（逐算子）；`repack.cpp` + `llamafile/sgemm.cpp`（分块 GEMM）；`ggml-threading.{h,cpp}`（多线程）。

**结构：**
- 导语：算子（L11）最终要落到 CPU 指令上算。这一课看 `ggml-cpu` 怎么把一次矩阵乘/点积，从"标量一个个乘"加速到"SIMD 一次算一排"，再切给多线程并行。
- `<h2>` 从标量到 SIMD：先看 generic 标量点积（`ggml_vec_dot_*_generic`）；再看 SIMD 把它向量化——一条指令同时处理 8 个（AVX2 `__m256`）/ 4 个（NEON `float32x4_t`）float。真实 intrinsic 逐行：load、FMA（`_mm256_fmadd_ps`/`vmlaq_f32`）、水平求和（horizontal sum）。
  - **trace ①（Style C·SVG）**："追踪一次 SIMD 点积"——画 8 条 lane 同时把 a[i]*b[i] 累加进 8 个累加器，最后水平求和成一个标量。
- `<h2>` 量化矩阵乘：`vec_dot_q4_0_q8_0` 怎么算——按 block（32 个权重一组）解包 4-bit 量化值、乘以共享 scale、与 Q8_0 激活做点积。真实代码（解包+FMA+缩放）。
  - **trace ②（Style A·站点流）**：一个 block 从"4-bit 打包字节 → 解包成 16 个量化值 → ×scale → 与激活点积 → 累加"。
- `<h2>` 多线程：`ggml-threading` 怎么把一个大矩阵乘按行/块切给 N 个线程；为什么 GEMM 适合并行（各行独立）。**图**：线程切分示意。
- `<h2>` 折叠深挖（≥2）：(1) `llamafile/sgemm` 的分块（tiling）为什么比朴素三重循环快（缓存友好）；(2) `arch/` 是怎么按 CPU 特性选实现的（运行时/编译期分派）。

### 课 32 · CUDA 后端（`ggml/src/ggml-cuda/`）

**源码锚点：** `ggml-cuda.cu`（后端入口）；真实 kernel `__global__`（`threadIdx/blockIdx`、`__shared__`）；`mmq.cu`（量化分块矩阵乘）、`mmvf.cu`/`mmvq.cu`（矩阵×向量）；`fattn*.cu`（flash attention，**仅概念**）；显存/数据搬运。

**结构：**
- 导语：GPU 靠"成千上万个线程同时算"取胜。这一课看 ggml 的 CUDA kernel 怎么组织这些线程，把一次矩阵乘拆成 block/thread 的协作。
- `<h2>` CUDA 执行模型：grid → block → thread 的三层；warp（32 线程齐步走）；`__shared__` 片上内存。**图**：线程网格层级。真实 kernel 骨架（`__global__ void k(...){ int i = blockIdx.x*blockDim.x + threadIdx.x; ... }`）。
- `<h2>` 分块矩阵乘（核心，真实但最小简化）：为什么要 tiling——把 A、B 的小块（tile）先搬进 `__shared__`，让一个 block 里的线程复用，减少慢速全局显存访问；每个 thread 算结果矩阵的一格。对照 `mmq.cu` 的真实结构。
  - **trace（Style C·SVG）**："追踪一次分块矩阵乘"——画一个 thread block 把 A 的一行 tile、B 的一列 tile 载入 shared memory，里面每个 thread 负责输出 C 的一格，循环累加。
- `<h2>` 显存与搬运：global / shared / register 三级；为什么"搬数据"常比"算"还贵（访存密集，呼应 L30 的 decode）。**图**：显存层级 + 带宽对比。
- `<h2>` flash attention（仅概念）：为什么把 softmax+matmul **融合**成一个 kernel——避免把巨大的注意力分数矩阵写回显存。简化伪代码（分块在线 softmax），不逐行真实 kernel。**card**：指向 `fattn*.cu`，告诉读者真实实现的复杂度来源。
- `<h2>` 折叠深挖（≥2）：(1) warp 内 reduction（`__shfl_*`）怎么快速求和；(2) 为什么 CUDA kernel 这么多变体（`mmq`/`mmvq`/不同精度/不同 GPU 架构）。

### 课 33 · 后端调度与其它后端（`ggml-backend*`，概览）

**源码锚点：** `ggml-backend.cpp`（`ggml_backend` 抽象、调度 `ggml_backend_sched`）；`ggml-backend-reg.cpp`（`ggml_backend_reg_count`/`ggml_backend_dev_by_type`/设备枚举）；`ggml-backend-dl.cpp`（`ggml_backend_load_all`，动态加载 .so/.dll）；其它后端目录 `ggml-metal/ggml-vulkan/ggml-sycl/ggml-cann/ggml-hip/ggml-opencl/ggml-rpc/...`。

**结构：**
- 导语：前两课讲了 CPU、CUDA 各自怎么算；可一台机器上可能同时有 CPU + 多种 GPU，ggml 怎么统一管理、把每个算子派到合适的后端？这一课看后端抽象层。
- `<h2>` 后端抽象：`ggml_backend` 是什么（一个"能跑算子的设备"的统一接口）；device / buffer / 一组算子实现。**图**：抽象层把上层计算图与各家后端解耦。
- `<h2>` 注册与动态加载（真实代码）：`ggml_backend_reg` 注册表；`ggml_backend_load_all` 怎么在运行时扫描并 `dlopen` 各后端动态库；`ggml_backend_dev_by_type` 按类型选设备。为什么用动态加载（一份二进制按机器实际硬件加载对应后端）。
- `<h2>` 调度：`ggml_backend_sched` 怎么把一张计算图的各算子分派到不同后端、并在后端之间搬运张量（呼应 L09/L10 的计算图）。
  - **trace（Style A·站点流）**："追踪一次 op 分派"——一个算子从计算图节点 → 调度器看它的输入在哪个 buffer/设备 → 选后端 → （必要时跨设备拷贝）→ 在该后端执行。
- `<h2>` 其它后端一览（概览）：Metal（Apple）、Vulkan（跨平台 GPU）、SYCL（Intel）、CANN（昇腾）、HIP（AMD）、OpenCL、RPC（远程后端）各是什么、何时用。**图/表**：后端 × 平台 × 用途。
- `<h2>` 折叠深挖（≥2）：(1) BLAS 后端与 RPC 后端这种"特殊后端"；(2) 加新后端大致要实现哪些接口（指向 `ggml-backend-impl.h`）。
- 收尾：本课收束"硬件层"，并衔接第七部分（进阶专题）。

## 统一交付标准（每课硬性达标，与 M1-M5 一致）

每课 `LESSON_NN = {"zh": r'''...''', "en": r'''...'''}`，须满足：

- **结构**：导语 `<p>` + 教学卡片（macro/detail/analogy/key/spark 酌用，≥2 张深挖 `<details>`）+ **≥3 个图示**（`flow`/`vflow`/`cols`/`cellgroup`/`layers`/`timeline`/`trace`，单语 ≥3，含 trace）+ **≥2 段真实/简化代码片段**（`<pre class="code">`）+ **≥1 个内嵌 worked-example trace**（见各课）。
- **双语对齐**：按 `<h2>` 分节，`<p>`/`<p ` 计数中英严格相等（`.trace` div 与内联 `<svg>` 不计入 `<p>`）。
- **中文密度**：zh CJK ≥ 4000；**en CJK == 0**（纯 ASCII；代码/SVG 文本用 `-`/`->`/`...`/`~=`/`+/-`，不用 em-dash/unicode 箭头/`≈`/`±`）。
- **无文字墙**：连续顶层 `<p>` ≤ 3。
- **转义**：渲染 `<`/`>`/`&` 须转义（`&lt;`/`&gt;`/`&amp;`）；无双重转义（`&amp;lt;`）。
- **trace 规范**：Style A 纯 HTML（`.trace/.tcap/.stations/.stn/.cellrow/.vc[.hot/.blue/.dim]/.op/.tlab`）；Style C 内联 `<svg viewBox=.. width="100%" role="img" aria-label=..>`（zh aria-label 中文、en 纯 ASCII），合法 XML，字面色板（白 `#ffffff`/`#1d2129`、accent `#c2630e`、blue `#2563eb`、purple `#7c3aed`、label `#5b6470` 等），深色可读；trace 不与 `<div class="card">` 紧邻。
- **源码引用**：以"文件 + 符号名"为主、不写死行号；对照真实 `/home/verden/course/llama.cpp` 核实（核验日期 2026-06-18）。
- **quiz**：`quizzes.py` 写该课 2-4 题双语自测。
- **登记**：`shell.PAGES`（filename、zh/en 短标题、`第六部分 · 底层内核`/`Part 6 · Low-level kernels`）、`shell.SUBTITLES`、`registry.CONTENT`（filename -> `part6.LESSON_NN`）；index 课数 30->33、部分数 5->**6**。

## 与 roadmap 衔接

- 完成后：roadmap M6 行"状态"`待写`->`完成`、状态追踪 `- [ ]`->`- [x]`。
- 执行：superpowers:subagent-driven-development（一课一个 task，顺序执行；收尾 task 勾 roadmap + 全量验证 + 完成分支）。**注意**：M5 经验表明后台 general-purpose 子代理写整课常中途失败（零写入），实现时若失败则由控制器亲自照模板执笔，仍保留完整 spec+质量双重审查。
- 关键 Style-C SVG（L31 SIMD 点积、L32 分块矩阵乘）建议先用 Python 预生成并校验（well-formed + 英文 ASCII + 坐标不溢出 + 深色可读 + rsvg 渲染目检），再喂给实现（沿用 M5 L28 的做法）。
