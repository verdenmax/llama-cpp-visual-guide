# M3 · 第三部分「ggml 张量引擎」实施计划 (Part 3 · The ggml engine)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写出第三部分的 6 课（lessons 08-13：ggml 核心对象 / 计算图惰性构建 / 图的执行与调度 / 核心算子 / 量化格式细节 / GGUF 文件格式），把第二部分建立的"张量 / 量化 / 后端"直觉，落到 ggml 引擎的真实数据结构与流程上；每课均达 M1.5/M1.6 确立的加强标准。

**Architecture:** 沿用既有"每课一组任务"模式——在新建 `src/part3.py` 里放 6 个 `LESSON_xx` 双语字典，并在 `shell.PAGES` / `registry.CONTENT` / `shell.SUBTITLES` / `quizzes.QUIZZES` 四处登记；不引入任何新组件或新依赖，全部复用既有设计系统与示意图 CSS 组件。每课产出后立即 `build.py` + `check_html.py` + `check_links.py` 全绿再 commit。

**Tech Stack:** 零依赖 Python 静态生成器（`src/*.py`）；纯 HTML/CSS 双语内容（`lang-zh`/`lang-en` + `bi()`）；校验脚本 `check_html.py`（结构/导航/计数/中文密度软检查）、`check_links.py`（死链）。源码事实核对对象：本机 llama.cpp 源树 `/home/verden/course/llama.cpp`（核验日期 2026-06-15）。

---

## 统一交付标准（每课都要满足，源自 Spec §5/§6）

每一课（一个 `LESSON_xx`，含 `"zh"` 与 `"en"` 两份）必须包含：

- **导语** `<p class="lead">`：点题（这一课回答什么问题、和上一课怎么衔接）。
- **教学卡片**：`macro`（🌍 宏观理解）、`detail`（🔬 细节/源码对应）、`analogy`（🔌 生活类比）、`key`（✅ 关键要点）、`spark`（💡 设计洞察），按需取用，建议齐全。
- **图：3-5 张，类型多样**（硬性）。复用（不要新造）的示意图 CSS 组件：`layers` / `flow`（横向）/ `vflow`（纵向步骤）/ `cols`（并排对比）/ `cellgroup`+`cells`+`cell`（`.scale`/`.hl`/`.q`/`.dim` + `.lab`/`.sep`）/ `timeline`+`lane`+`tslot`（`.span`/`.now`）/ `<table class="t">`。**原理课至少 1 张概念示意图**（用 cells/timeline 画原理草图，深色模式自适应、自包含、不外链）。
- **代码：2-3 段**（硬性）。先伪代码讲思路，再给从真实源码简化的片段，文中标注**来源文件 + 符号名**（不写死行号）。代码片段内部 **ASCII 优先**（`-`/`->`/`...`），不用 em-dash / unicode 箭头 / `×` / `…`。`<pre>` 里 `<`/`>`/`&` 写成实体。
- **折叠深挖：2-3 个** `<details class="accordion">`（配 `<p class="acc-intro">` 引导句 + `<div class="acc-body">` 正文）。
- **纯中文正文 ~4000+ 汉字**（CJK，按 `\u4e00-\u9fff` 计，不含英文/代码/路径）。英文 `"en"` 与中文**逐段对齐**（信息点齐全，可略简，但不得缺段），且 `"en"` 不含 CJK。

每课的登记清单（缺一处都会导致 build/check 失败或课程不显示）：

1. `src/shell.py` 的 `PAGES`：追加 `(filename, 标题_zh, 标题_en, "第三部分 · ggml 引擎", "Part 3 · The ggml engine")`。
2. `src/part3.py`：写 `LESSON_xx = {"zh": r"""...""", "en": r"""..."""}`。
3. `src/registry.py`：`import part3` 并在 `CONTENT` 追加 `"filename": part3.LESSON_xx`。
4. `src/shell.py` 的 `SUBTITLES`：追加 `"filename": ("副标题_zh", "subtitle_en")`。
5. `src/quizzes.py` 的 `QUIZZES`：追加该课 `{"mcq": [...2-4 题...], "open": [...]}`（双语；`answer` 为正确项在 `opts` 中的 0-based 下标）。
6. 运行 `cd src && python build.py && python check_html.py && python check_links.py`，**0 error / 0 warning + 全链接解析**后再 commit。

**门槛复核**：`check_html.py` 的软检查含 `MIN_DIAGRAMS`（视觉块密度，两语都数 >=3/语）、`MIN_CJK=3000`（zh 汉字数，目标 4000+）、`MAX_LESSON=40`（`第 N 课` 交叉引用上限）。新课必须让这些软检查都不报 WARN。

## 执行经验（来自 M2，务必照做）

- **控制者直接执笔**：M2 中两个内容子代理在"写大段 HTML"时都中途卡死（20-34 分钟无产出）。因此本里程碑由控制者（主会话）<strong>直接用 edit/create 执笔每课内容</strong>，但<strong>保留每课的 spec + 质量双重审查子代理</strong>（只读、卡死风险低，全程本模型 `claude-opus-4.8`）。
- **CJK 会被内联英文/代码稀释**：ggml 课里 `ggml_context`、`mem_buffer` 这类英文符号很多，纯中文字数会被压低。**初稿就要把中文写足**（目标 4200+），再用 `check_html.py` 的 CJK 计数校准；不足就补"是什么 / 为什么这么设计 / 还有什么替代"的讲解段，而不是堆英文符号。
- **逐段对齐**：每加一段中文，必须同步在 `"en"` 里加对应英文段（en 不含 CJK）。审查会逐段核对，缺段会被标 MISALIGNED。
- **每课产出后立即 commit**；加新课会改动所有已存在课的顶栏进度条（`/ N`、`width:%`），故 `git add -A` 把重建后的全部 HTML 一并提交，保持 `git status` 干净（M2 曾因只提交新课导致旧课进度条过期）。

## 源码事实（已核对的锚点，核验日期 2026-06-15；写作时仍以"文件 + 符号"引用，不写行号）

> 核对源树：`/home/verden/course/llama.cpp`。

- **L08**：`ggml_init_params {size_t mem_size; void * mem_buffer; bool no_alloc;}`（`ggml/include/ggml.h`）；`ggml_init` 分配<strong>单个预分配 arena</strong>（`mem_buffer = params.mem_buffer ? : ggml_aligned_malloc(mem_size)`），`ggml_free` 仅在 `mem_buffer_owned` 时释放。`ggml_new_object`（static，`ggml/src/ggml.c`）在 `ctx->mem_buffer` 末尾<strong>bump 分配</strong>、维护 `objects_begin/end` 链表、按 `GGML_MEM_ALIGN` 对齐、超出 `mem_size` 即 abort；`ggml_new_tensor_impl` 经它分配，张量元数据与（`!no_alloc` 时）数据都在 ctx 缓冲里——<strong>无 per-tensor malloc</strong>。`ggml_object {offs,size,next,type}`；`GGML_DEFAULT_GRAPH_SIZE=2048`；`ggml_tensor_overhead()`。
- **L09**：`ggml_cgraph {int size; int n_nodes; int n_leafs; ggml_tensor ** nodes; ggml_tensor ** grads; ggml_tensor ** leafs; ...}`（`ggml/src/ggml-impl.h`）；`ggml_build_forward_expand` -> `ggml_build_forward_impl`；`ggml_new_graph` / `ggml_new_graph_custom(ctx,size,grads)`。<strong>惰性</strong>：`ggml_mul_mat`/`ggml_add` 只 `ggml_new_tensor` 出结果张量、设 `result->op` 与 `result->src[0/1]`，<strong>不计算</strong>；op 参数经 `ggml_set_op_params_*` 存入。`ggml_visit_parents`：`op==GGML_OP_NONE && !PARAM` 的张量入 `leafs`（输入/常量），否则入 `nodes`（算子结果）。
- **L10**：`ggml_backend_graph_compute(backend,cgraph)`（`ggml/include/ggml-backend.h`，impl `ggml/src/ggml-backend.cpp`）；`ggml_backend_sched`（多后端：拆图、把算子指派到设备、跨设备拷贝；`_new/_reserve/_alloc_graph/_graph_compute/_split_graph`）。ggml-alloc：`ggml_gallocr`（`ggml/src/ggml-alloc.c`）用 `free_blocks[MAX_FREE_BLOCKS]` 做 best-fit，<strong>张量生命周期结束后其内存被后续张量复用</strong>（释放时与相邻块合并）。`ggml_backend_cpu_init`；注册表 `ggml_backend_reg_t`/`ggml_backend_dev_t`。
- **L11**：`ggml_mul_mat(ctx,a,b)`（`ggml/src/ggml.c`）：断言 `ggml_can_mul_mat`（要求 `a->ne[0]==b->ne[0]` 内维相等，且高两维可广播 `b->ne[2]%a->ne[2]==0` 等）、结果 `ne={a->ne[1], b->ne[1], b->ne[2], b->ne[3]}`、类型 F32。`ggml_rope`/`ggml_rope_ext(...,n_dims,mode,n_ctx_orig,freq_base,freq_scale,...)`；`ggml_soft_max_ext(ctx,a,mask,scale,max_bias)`（融合 `soft_max(a*scale + mask)`，max_bias=0 即无 ALiBi）；`ggml_rms_norm(ctx,a,eps)`。CPU 计算：`ggml_compute_forward_mul_mat`（`ggml/src/ggml-cpu/ggml-cpu.c`）+ `ggml_compute_forward_{rms_norm,soft_max,rope}`（`ggml/src/ggml-cpu/ops.cpp`），由大 `switch(op)` 派发。
- **L12**：`ggml/src/ggml-common.h`：`QK4_0=32`/`QK8_0=32`/`QK_K=256`；`block_q4_0{ggml_half d; uint8_t qs[QK4_0/2];}`（18 字节=4.5bit）、`block_q8_0{ggml_half d; int8_t qs[QK8_0];}`、`block_q4_K{ ggml_half d; ggml_half dmin;(或 ggml_half2 dm) uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2];}`（<strong>无 qh</strong>）、`block_q6_K{uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d;}`。`ggml/src/ggml-quants.c`：`dequantize_row_q4_0`（`x=(q-8)*d`）、`dequantize_row_q4_K`、`quantize_row_q4_0_ref`（`d = max / -8`）；`ggml_quantize_chunk`。`ggml_type_traits {type_name, blck_size, type_size, is_quantized, to_float, from_float_ref}`（注意是 `from_float_ref`）。
- **L13**：`ggml/include/gguf.h`：`GGUF_MAGIC="GGUF"`、`GGUF_VERSION=3`、`GGUF_DEFAULT_ALIGNMENT=32`；磁盘布局 = magic + version(uint32) + tensor_count(int64) + kv_count(int64) + KV 对(key,string;value 带 `gguf_type`；数组带元素类型+uint64 计数) + 每张量信息(name,n_dims,dims,`ggml_type`,offset) + 对齐 + 张量数据。`gguf_init_from_file`、`gguf_get_*`；`enum gguf_type {UINT8..STRING,ARRAY,UINT64,INT64,FLOAT64}`。mmap：`llama_mmap`（`src/llama-mmap.cpp`，POSIX `mmap(PROT_READ)` 只读零拷贝），由 `src/llama-model-loader.cpp` 经 `use_mmap`/`llama_mmap::SUPPORTED` 使用；张量数据段对齐。

---

## Task 1: 课 08「ggml 核心对象 / ggml core objects」

> 第三部分第 1 课，**创建** `src/part3.py`。讲 ggml 的"地基三件套"：`ggml_context`（内存池/arena）、`ggml_tensor`（已在 L05 讲过字段，这里讲它怎么从池子里分配出来）、以及 no-malloc 的 bump 分配设计。与 L05 区分：L05 讲张量"长什么样"，本课讲张量"从哪来、内存怎么管"。

**Files:** `src/shell.py`（PAGES+SUBTITLES 各加 08）、`src/part3.py`（CREATE，含 docstring + LESSON_08）、`src/registry.py`（`import part3` + CONTENT 加 08）、`src/quizzes.py`（QUIZZES 加 08）。产出 `index.html`（8 课 · 3 个部分）+ `lessons/08-ggml-core-objects.html`。

- [ ] **Step 1-3: 登记**（03/04...07 之后追加）
```python
# PAGES
("08-ggml-core-objects.html", "ggml 核心对象", "ggml core objects",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
# SUBTITLES
"08-ggml-core-objects.html": ("ggml_context · 内存池 arena · no-malloc bump 分配",
                              "ggml_context; the memory-pool arena; no-malloc bump allocation"),
# registry: import part3 ; "08-ggml-core-objects.html": part3.LESSON_08,
```

- [ ] **Step 4: 执笔 LESSON_08（双语）**。**结构（按序）**：
1. `lead`：第二部分给了直觉，第三部分开始拆引擎。第一站是"内存"——ggml 怎么管张量的内存？答案是一个叫 `ggml_context` 的内存池。
2. `analogy`（🔌）：`ggml_context` 像一块<strong>预先划好的停车场</strong>：开场一次性圈好一大片地（mem_size），之后每停一辆车（建一个张量/对象）就往后挪一个车位（bump），不必每次去物业重新申请地皮（malloc）。
3. `<h2>` 三件套总览 + **图1【三层关系】**（`layers` 或 `vflow`）：`ggml_init_params`（配置：mem_size/mem_buffer/no_alloc）-> `ggml_context`（持有 arena）-> 里面装着 `ggml_object` 链表（每个 object 包一个 tensor / graph）。
4. `<h2>` ggml_context：一个内存池 + **代码1（源码简化，`ggml/include/ggml.h` 的 `ggml_init_params`）**：
```
struct ggml_init_params {
    size_t mem_size;     // arena 总大小
    void * mem_buffer;   // 传 NULL 则内部分配
    bool   no_alloc;     // true = 只建元数据, 不为张量数据分配
};
ctx = ggml_init(params);   // 一次性拿到整块 arena
```
   讲 `ggml_init` 分配单块 arena、`ggml_free` 释放；mem_buffer 可外部传入（嵌入式/复用场景）。
5. `<h2>` no-malloc：bump 分配 + **图2【bump 分配概念图】**（`cellgroup`/`cells`，本课概念图）：一条 arena 字节带，已用部分 + 一个"游标"，每 `ggml_new_object` 就把游标往右推一格（obj1 | obj2 | obj3 | -> 空闲）；标注 `objects_begin/end` 链表。
6. 正文 + **代码2（伪代码，bump 的本质，对应 `ggml/src/ggml.c` 的 `ggml_new_object`/`ggml_new_tensor_impl`）**：
```
def new_object(ctx, size):
    cur = ctx.objects_end.offs + ctx.objects_end.size   # 当前游标
    if cur + size > ctx.mem_size: abort("arena 不够")    # 不扩容, 直接报错
    obj = place_at(ctx.mem_buffer + cur)                 # 就地放下
    link_into(ctx.objects, obj)                          # 接入链表
    return obj
```
   讲清"<strong>不 per-tensor malloc</strong>"：元数据和数据都从这块 arena 里切；超了不扩容、直接 abort（所以要预估够大，`ggml_tensor_overhead()` 帮你算每个张量的元数据开销）。
7. `<h2>` no_alloc 与"只建图不算" + 一段话：`no_alloc=true` 时只分配张量<strong>元数据</strong>、不分配数据缓冲——这正是"先建图、再由后端分配真正内存"的前提（接 L09/L10）。
8. **深挖1**"为什么不用 malloc / new 一个个分配张量？"：减少分配器开销与碎片、内存连续利于缓存与一次性释放、便于整体迁移到别的后端缓冲；代价是要预估 arena 大小。
9. **深挖2**"arena 不够会怎样？怎么估大小？"：超出即 abort；用 `ggml_tensor_overhead()` × 张量数 + 数据字节 估算；图通常用 `GGML_DEFAULT_GRAPH_SIZE=2048` 个节点的余量。
10. **深挖3**"ggml_context 和后端内存是一回事吗？"：不是——ctx 的 arena 常只放<strong>元数据</strong>（no_alloc），真正的张量数据由后端 buffer（L10 的 ggml-alloc）分配；澄清两层内存。
11. `key`（✅）：ggml_context = 预分配 arena；no-malloc bump 分配（`ggml_new_object` 往后推游标、不 per-tensor malloc）；超出即 abort；`no_alloc` 只建元数据，为"先建图后分配"铺路；`ggml_object` 链表串起所有对象。
12. `spark`（💡）：用"一次大分配 + 内部 bump"替代"成千上万次 malloc"——简单、快、可整体迁移，是 ggml 能把一张计算图轻量地搭起来、再整体丢给某个后端的根基。

必须讲到：arena/内存池；bump 分配 vs per-tensor malloc；no_alloc 的意义；超限 abort；两层内存（ctx 元数据 vs 后端数据）。

- [ ] **Step 5: quiz（08）** 2-3 mcq + 1 open：
- MCQ1 "ggml 为每个张量都单独 malloc 一次吗？" -> 正确："不，张量从 ggml_context 预分配的 arena 里 bump 切出，不 per-tensor malloc"；干扰：每次都 malloc / 用 GC / 存磁盘。
- MCQ2 "ggml_init_params 里 no_alloc=true 意味着什么？" -> 正确："只分配张量元数据、不分配数据缓冲（为先建图后由后端分配铺路）"；干扰：不分配任何东西 / 关闭量化 / 只读模式。
- OPEN "为什么 arena 满了 ggml 选择直接 abort，而不是自动扩容？这对使用者提出了什么要求？"

- [ ] **Step 6-7: 重建+校验+commit**
```bash
cd src && python build.py && python check_html.py && python check_links.py
# 期望 0/0、链接解析、index "共 8 课 · 3 个部分"、L08 CJK>=4000、en CJK=0
git add -A && git commit -m "feat: add lesson 08 ggml core objects (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 2: 课 09「计算图：惰性构建 / The compute graph: lazy build」

> 第三部分第 2 课。讲 ggml 最核心的思想之一：调用 `ggml_mul_mat` 这类函数时<strong>并不立刻计算</strong>，而是建一个记录"谁依赖谁"的<strong>计算图</strong>。接 L08（张量从 arena 来）、接 L03（"先建图后执行"曾一笔带过，这里展开）。

**Files:** shell.py（PAGES+SUBTITLES 加 09）、part3.py（追加 LESSON_09）、registry.py（加 09）、quizzes.py（加 09）。产出 `09-compute-graph.html`。

- [ ] **Step 1-3: 登记**
```python
("09-compute-graph.html", "计算图：惰性构建", "The compute graph: lazy build",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
"09-compute-graph.html": ("先建图后执行 · op/src 反向指针 · nodes vs leafs",
                          "build-then-run; op/src back-pointers; nodes vs leafs"),
"09-compute-graph.html": part3.LESSON_09,
```

- [ ] **Step 4: 执笔 LESSON_09（双语）**。**结构**：
1. `lead`：L08 知道张量从 arena 来。但你写 `c = ggml_mul_mat(a,b)` 时，乘法<strong>并没发生</strong>——ggml 只是记下"c 由 a、b 经矩阵乘得到"。这一课讲这种"惰性建图"。
2. `analogy`（🔌）：像写<strong>菜谱</strong>而不是马上做菜：先把"先切菜、再下锅、最后装盘"的步骤和依赖写成一张流程图，等真要开火时（执行）才照着做。建图 = 写菜谱，执行 = 照做。
3. `<h2>` 一次调用发生了什么 + **图1【op/src 反向指针】**（`flow` 或 `cells`）：`a`、`b` 两个张量 -> `ggml_mul_mat` -> 新张量 `c`，`c.op=MUL_MAT`、`c.src[0]=a`、`c.src[1]=b`（箭头从 c 指回 a/b，强调"反向")。
4. **代码1（源码简化，`ggml/src/ggml.c` 的 `ggml_mul_mat`）**：
```
struct ggml_tensor * ggml_mul_mat(ctx, a, b) {
    result = ggml_new_tensor(ctx, GGML_TYPE_F32, ...);  // 只建结果张量
    result->op     = GGML_OP_MUL_MAT;                   // 记下"怎么来的"
    result->src[0] = a;  result->src[1] = b;            // 记下输入
    return result;                                       // 不计算!
}
```
   讲：每个算子函数都是这套路——建结果张量、填 op/src、返回；真正的数乘一个都没做。
5. `<h2>` 把张量串成图 + **图2【一张小计算图】**（`vflow` 或 `flow`）：叶子 x、W1、W2 -> h=mul_mat(W1,x) -> y=mul_mat(W2,h)，画成有向图；标出 nodes（h,y）与 leafs（x,W1,W2）。
6. 正文 + **代码2（伪代码，`ggml_build_forward_expand` / `ggml_visit_parents`）**：
```
# 从输出张量出发, 沿 src 回溯, 拓扑排序进 graph
def build_forward(graph, t):
    for s in t.src: build_forward(graph, s)   # 先收集依赖
    if t.op == NONE and not t.is_param:
        graph.leafs.append(t)   # 输入/常量
    else:
        graph.nodes.append(t)   # 算子结果, 按依赖顺序
```
   讲 `ggml_cgraph` 结构（nodes/leafs/n_nodes/size）、`ggml_new_graph(_custom)`、`ggml_build_forward_expand(graph, out)`：从输出回溯、拓扑排序，保证执行时输入先于输出。
7. `<h2>` nodes vs leafs + **图3【nodes / leafs 分类表】**（`<table class="t">` 或 `cols`）：leafs=输入/权重/常量（op==NONE）；nodes=算子结果（有 op、按拓扑序）；执行时只算 nodes。
8. **深挖1**"惰性建图到底买到了什么？"：① 先看到完整图 -> 能整体优化/分配内存（L10 的内存复用要靠它）；② 同一张图可在不同后端执行；③ 把"描述"和"执行"解耦（呼应 L01）。
9. **深挖2**"src 反向指针，和 L05 说的 op/src 是一回事吗？"：是——L05 讲字段存在，这里讲它们<strong>串成图</strong>的用途；每个张量记得父节点，整张图就是张量靠 src 连成的网。
10. **深挖3**"图建好后存在哪？会很占内存吗？"：图本身只是<strong>指针数组</strong>（nodes/leafs 指向 arena 里的张量），很轻；张量元数据也在 ctx arena 里（L08）。真正占内存的是张量<strong>数据</strong>，那要等 L10 分配。
11. `key`（✅）：算子函数<strong>只建结果张量、填 op/src，不计算</strong>；`ggml_build_forward_expand` 从输出沿 src 回溯做拓扑排序；leafs=输入/常量、nodes=算子结果；惰性建图是后续"整体内存复用 + 多后端执行"的前提。
12. `spark`（💡）：先把整段运算<strong>画成一张图、再统一执行</strong>——这一步"延迟"换来了全局视野：内存可以复用、算子可以调度到不同硬件、还能反向求导。ggml 的威力，从"先不算"开始。

必须讲到：算子不立即计算、只填 op/src；拓扑排序建图；nodes vs leafs；惰性的三个好处。

- [ ] **Step 5: quiz（09）**：
- MCQ1 "调用 ggml_mul_mat(a,b) 时发生了什么？" -> 正确："新建一个结果张量并记下 op=MUL_MAT、src=[a,b]，但不做乘法"；干扰：立刻算出结果 / 修改 a / 写磁盘。
- MCQ2 "计算图里 leafs 和 nodes 的区别？" -> 正确："leafs 是输入/权重/常量（op==NONE），nodes 是算子结果（按拓扑序，执行时计算）"；干扰：leafs 是输出 / 没区别 / nodes 是叶子。
- OPEN "为什么 ggml 要'先建图、后执行'，而不是边调用边算？至少说出两个好处。"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 9 课 · 3 个部分"；commit `feat: add lesson 09 compute graph lazy build (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）

---

## Task 3: 课 10「图的执行与调度 / Graph execution & scheduling」

> 第三部分第 3 课。L09 建好了图，这一课讲<strong>怎么把它算出来</strong>：后端执行（`ggml_backend_graph_compute`）、多后端调度（`ggml_backend_sched`）、以及 ggml-alloc 怎么<strong>复用内存</strong>。接 L07（后端是什么）、L09（图是什么）。

**Files:** shell.py、part3.py（追加 LESSON_10）、registry.py、quizzes.py。产出 `10-graph-execution.html`。

- [ ] **Step 1-3: 登记**
```python
("10-graph-execution.html", "图的执行与调度", "Graph execution & scheduling",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
"10-graph-execution.html": ("backend 执行 · sched 多后端调度 · ggml-alloc 内存复用",
                            "backend compute; multi-backend sched; ggml-alloc memory reuse"),
"10-graph-execution.html": part3.LESSON_10,
```

- [ ] **Step 4: 执笔 LESSON_10（双语）**。**结构**：
1. `lead`：图建好了，但还只是"指针搭的骨架"。这一课让它真正<strong>跑起来</strong>：先分配内存、再按依赖顺序逐个算节点，必要时跨多个后端协作。
2. `analogy`（🔌）：像<strong>施工队照图纸盖楼</strong>：先按图纸算好要多少材料、堆在哪（内存分配），再按"先地基后楼层"的顺序施工（按拓扑序算节点）；工地不够大就<strong>腾挪复用</strong>同一块场地（内存复用）。
3. `<h2>` 执行三步 + **图1【建图->分配->执行】**（`flow`）：build graph(L09) -> ggml-alloc 规划内存 -> backend 逐节点 compute -> 输出。
4. `<h2>` 内存复用：ggml-alloc + **图2【内存复用概念图】**（`timeline` 或 `cellgroup`，本课概念图）：一条"内存槽"时间线，张量 A 用完（生命周期结束）后，它的槽被后来的张量 C 复用（同一格先标 A 再标 C）；标注 free_blocks 思想。
5. 正文 + **代码1（伪代码，`ggml/src/ggml-alloc.c` 的 gallocr best-fit/复用）**：
```
# 规划阶段: 按图遍历, 为每个张量找内存, 用完即归还
def plan(graph):
    for t in graph.nodes:                 # 拓扑序
        t.offset = free_blocks.best_fit(nbytes(t))  # 复用空闲块
        for s in t.src:
            if last_use(s) == t:          # s 之后不再用
                free_blocks.give_back(s)  # 归还, 供后面复用
```
   讲：因为 L09 已有<strong>完整的图</strong>，才能预知每个张量的生命周期、把内存<strong>复用</strong>到极致（峰值远小于"每个张量各占一块"）。
6. `<h2>` 后端执行 + **代码2（源码简化，`ggml/include/ggml-backend.h`）**：
```
ggml_backend_t be = ggml_backend_cpu_init();   // 或 cuda/metal...
ggml_gallocr_alloc_graph(galloc, graph);       // 真正分配
ggml_backend_graph_compute(be, graph);         // 逐节点算
```
   讲 `ggml_backend_graph_compute` 按拓扑序对每个 node 调对应后端的算子核函数（接 L11）。
7. `<h2>` 多后端调度：ggml_backend_sched + **图3【sched 拆图分派】**（`cols` 或 `layers`）：一张图被 sched 切成几段，分别指派给 CPU / GPU 后端，跨设备处自动插入数据拷贝。
8. 正文：`ggml_backend_sched` 干三件事——<strong>拆图、把算子指派到合适设备、在设备间拷贝张量</strong>；这正是 `-ngl` 把一部分层放 GPU、其余留 CPU 时背后的机制（接 L07）。
9. **深挖1**"为什么内存能复用得这么省？"：靠 L09 的<strong>完整图</strong> + 拓扑序 -> 能精确知道每个张量"最后一次被用"在哪，之后立刻归还内存；best-fit + 相邻空闲块合并，峰值内存大幅下降。
10. **深挖2**"sched 怎么决定哪段放哪个后端？"：按张量所在 buffer / 算子支持情况指派；不支持的算子回退 CPU（L07 提过的 fallback）；跨设备边界自动插 copy 节点。
11. **深挖3**"reserve / alloc 两步是干嘛的？"：先 `reserve` 用图<strong>预演</strong >一遍、量出峰值内存并一次性开好缓冲；之后每次 `alloc_graph` 复用这块缓冲，避免反复大分配——典型"配置一次、反复执行"。
12. `key`（✅）：执行 = 分配内存 + 按拓扑序逐节点 compute；ggml-alloc 靠完整图预知生命周期、<strong>复用内存</strong>把峰值压到很低；`ggml_backend_graph_compute` 跑单后端；`ggml_backend_sched` 拆图、跨多后端指派与拷贝（`-ngl` 的底层）。
13. `spark`（💡）：L09"先建图"的延迟，在这里<strong>连本带利还回来</strong>——正因为提前看到整张图，内存能复用、算子能跨设备调度。"先描述、后执行"不是麻烦，而是把全局优化的可能性攥在了手里。

必须讲到：分配+逐节点执行；内存复用靠完整图+生命周期；单后端 compute；sched 多后端拆图/指派/拷贝；reserve/alloc 两步。

- [ ] **Step 5: quiz（10）**：
- MCQ1 "ggml-alloc 为什么能大幅复用内存、压低峰值？" -> 正确："因为惰性建图提供了完整的图，能预知每个张量的生命周期，用完即归还供后续复用"；干扰：因为用了量化 / 因为内存便宜 / 因为不存中间结果。
- MCQ2 "ggml_backend_sched 主要负责什么？" -> 正确："把一张图拆开、把算子指派到合适的后端设备，并在设备间拷贝张量"；干扰：解析 GGUF / 量化权重 / 决定采样。
- OPEN "把模型一半层放 GPU、一半留 CPU（-ngl 设一半）时，sched 在背后大概做了哪些事？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 10 课 · 3 个部分"；commit `feat: add lesson 10 graph execution and scheduling (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）

---

## Task 4: 课 11「核心算子 / Core operators」

> 第三部分第 4 课。把 transformer 里最常见的几个算子讲透：矩阵乘 `mul_mat`、归一化 `rms_norm`、`rope`、`soft_max_ext`，重点是<strong>形状怎么推导</strong>、以及算子在 CPU 上"真正算"的地方。接 L04（注意力数学）、L09/L10（图与执行）。

**Files:** shell.py、part3.py（追加 LESSON_11）、registry.py、quizzes.py。产出 `11-core-operators.html`。

- [ ] **Step 1-3: 登记**
```python
("11-core-operators.html", "核心算子", "Core operators",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
"11-core-operators.html": ("mul_mat 形状推导 · rms_norm/rope/soft_max_ext · CPU 计算",
                           "mul_mat shapes; rms_norm/rope/soft_max_ext; CPU compute"),
"11-core-operators.html": part3.LESSON_11,
```

- [ ] **Step 4: 执笔 LESSON_11（双语）**。**结构**：
1. `lead`：图由算子组成。这一课挑出 transformer 里最核心的几个算子，看清它们各算什么、<strong>输入输出形状怎么对上</strong>，以及在 CPU 上真正落地计算的地方。
2. `analogy`（🔌）：算子像<strong>乐高积木</strong>：每块有固定的凸点/凹槽（输入输出形状），只有形状对得上才能拼起来；建图就是按形状把积木拼成模型，形状一错，拼装（断言）当场失败。
3. `<h2>` 头号算子：矩阵乘 mul_mat + **图1【mul_mat 形状推导】**（`cells` 或 `cols`，本课概念图）：a=[k, m]、b=[k, n]（ggml 约定 ne[0] 是内维 k）-> 内维 k 必须相等 -> 结果 c=[m, n]；高亮"被消去的 k"。
4. 正文 + **代码1（源码简化，`ggml/src/ggml.c` 的 `ggml_mul_mat` / `ggml_can_mul_mat`）**：
```
// 断言: a 的内维 == b 的内维 (ne[0] 相等)
GGML_ASSERT(a->ne[0] == b->ne[0]);          // k 必须对上
// 结果形状: 取 a 的"行"、b 的"列", 高两维来自 b
ne = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };  // 类型 F32
```
   讲 ggml 的内维约定（ne[0] 最内）让"内维相等"这条规则读起来和数学的"行列相乘"<strong>方向相反</strong>，要小心（呼应 L05 的维度顺序坑）；高两维支持广播。
5. `<h2>` 三个常客：rms_norm / rope / soft_max_ext + **图2【注意力里的算子流】**（`vflow` 或 `flow`）：x -> rms_norm -> mul_mat(Wq/k/v) -> rope(q,k) -> soft_max_ext(scores, mask) -> mul_mat(v) （把 L04 的注意力用算子串起来）。
6. **代码2（源码简化，算子签名，`ggml/include/ggml.h`）**：
```
ggml_rms_norm(ctx, a, eps);                       // 按最后一维归一化
ggml_rope_ext(ctx, a, pos, ff, n_dims, mode, ...);// 旋转位置编码
ggml_soft_max_ext(ctx, a, mask, scale, max_bias); // 融合 softmax(a*scale + mask)
```
   逐个一句话讲用途：rms_norm 稳定数值（L04 提过）、rope 注入位置（L04 提过）、soft_max_ext 把分数变权重并施加因果掩码（L04 的 -inf 掩码就在这）。
7. `<h2>` 算子在哪"真正算" + **图3【建图 vs 计算两处】**（`cols`）：左=`ggml.c` 里建图（填 op/src，L09）；右=`ggml-cpu/` 里 `ggml_compute_forward_*` 真正算（被 backend compute 的大 switch 派发，L10）。
8. 正文：强调"<strong>一个算子有两处代码</strong>"——`ggml_mul_mat`（建图、定形状）与 `ggml_compute_forward_mul_mat`（CPU 实现、真算）；GPU 后端则各有自己的 kernel（接 L12/第六部分）。
9. **深挖1**"为什么 mul_mat 的形状规则看起来和数学反着来？"：ggml 行优先、ne[0] 是内维，所以"内维相等"对应数学里的"左矩阵列数==右矩阵行数"；记 L05 的口诀"ne[0] 最贴内存"就不会错。
10. **深挖2**"soft_max_ext 的 mask 和 scale 是干嘛的？"：scale=1/sqrt(d) 缩放分数防止过大；mask 加上因果掩码（未来位置 -inf）；max_bias 控制 ALiBi（不用时为 0）——一个融合算子把这几步并成一次，省内存省带宽。
11. **深挖3"算子这么多，ggml 怎么管？"**：每个算子是 `enum ggml_op` 的一个值；建图时记在 `tensor->op`，执行时 backend 用一个大 `switch(op)` 派发到对应 `compute_forward`。加新算子=加一个 enum + 一个 forward 实现。
12. `key`（✅）：mul_mat 要求<strong>内维 ne[0] 相等</strong>、结果取 a/b 的"行/列"；rms_norm/rope/soft_max_ext 分别管归一化/位置/注意力权重；每个算子<strong>两处代码</strong>（建图定形状 + 后端真算）；执行靠 `switch(op)` 派发到 `ggml_compute_forward_*`。
13. `spark`（💡）：把一个算子拆成"<strong>声明形状</strong>"和"<strong>各后端各自实现</strong>"两半——前者让建图轻量且能查错，后者让同一个算子在 CPU/CUDA/Metal 上各有最优实现。模型逻辑写一遍，硬件加速写多份，正是这种拆分的红利。

必须讲到：mul_mat 形状规则与 ne[0] 内维；rms_norm/rope/soft_max_ext 各自作用；建图 vs compute_forward 两处；switch(op) 派发。

- [ ] **Step 5: quiz（11）**：
- MCQ1 "ggml_mul_mat(a,b) 对形状的核心要求是？" -> 正确："a 和 b 的内维 ne[0] 必须相等（被消去的那一维）"；干扰：形状完全相同 / b 必须是方阵 / 无要求。
- MCQ2 "soft_max_ext 里的 mask 起什么作用？" -> 正确："给分数加上掩码（如把未来位置设为 -inf 实现因果掩码）"；干扰：归一化 / 量化 / 缩放学习率。
- MCQ3 "为什么说一个算子有'两处代码'？" -> 正确："一处在 ggml.c 建图、定 op/src 与输出形状；另一处在后端（如 ggml-cpu 的 compute_forward）真正计算"；干扰：调试和发布 / 前端和后端网页 / 训练和推理。
- OPEN "ggml 的 mul_mat 形状规则为什么读起来和你在数学课学的'行×列'方向相反？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 11 课 · 3 个部分"；commit `feat: add lesson 11 core operators (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）

---

## Task 5: 课 12「量化格式细节 / Quantization formats in detail」

> 第三部分第 5 课。L06 讲了量化的直觉，这一课<strong>钻进字节级</strong>：block 结构体的精确布局、super-block（K-quant）、`ggml-quants.c` 的解量化、以及 `ggml_type_traits` 怎么把量化类型接进引擎。接 L06（量化直觉）、L05（type 字段）。

**Files:** shell.py、part3.py（追加 LESSON_12）、registry.py、quizzes.py。产出 `12-quant-formats.html`。

- [ ] **Step 1-3: 登记**
```python
("12-quant-formats.html", "量化格式细节", "Quantization formats in detail",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
"12-quant-formats.html": ("block 字节布局 · super-block K-quant · 解量化 · type_traits",
                          "block byte layout; super-block K-quant; dequant; type_traits"),
"12-quant-formats.html": part3.LESSON_12,
```

- [ ] **Step 4: 执笔 LESSON_12（双语）**。**结构**（与 L06 严格区分：L06 是"为什么/直觉"，本课是"字节怎么排/源码怎么写"）：
1. `lead`：L06 建立了"每块一个 scale"的直觉。这一课打开 `ggml-common.h`，看这些块在内存里<strong>到底每个字节装什么</strong>，以及解量化函数怎么把它还原。
2. `analogy`（🔌）：block 结构体像<strong>压缩包的文件头 + 数据段</strong>：开头几字节是"解压参数"（scale/min），后面是"压缩数据"（量化值）；解量化就是照着文件头把数据段还原。
3. `<h2>` 基础块：q4_0 / q8_0 + **图1【q4_0 / q8_0 字节布局】**（`cellgroup`/`cells`，本课概念图）：q4_0 = [d:2B][qs:16B]=18B；q8_0 = [d:2B][qs:32B]=34B；对齐标注每段字节数。
4. **代码1（源码，`ggml/src/ggml-common.h`）**：
```
#define QK4_0 32
typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0; // 18 B
typedef struct { ggml_half d; int8_t  qs[QK8_0];   } block_q8_0; // 34 B
```
   讲 q8_0 为什么更准更大（每权重 1 字节 int8，不打包）。
5. `<h2>` super-block：K-quant + **图2【q4_K super-block 分层】**（`layers` 或 `cellgroup`）：256 个权重的超块 = 整体 d + dmin + 8 个子块各自的 6-bit scale/min + 量化值；强调"两层 scale"。
6. **代码2（源码简化，`block_q4_K` / `block_q6_K`）**：
```
typedef struct {
    ggml_half d;     // 超块整体 scale
    ggml_half dmin;  // 超块整体 min
    uint8_t scales[K_SCALE_SIZE]; // 8 个子块的 6-bit scale/min
    uint8_t qs[QK_K/2];           // 256 个 4-bit 量化值
} block_q4_K;       // QK_K = 256, 无 qh
```
   讲 K-quant 的"超块整体 d/dmin + 子块细 scale"两层结构，正是 L06 说的"同 bit 更准"的来源。
7. `<h2>` 解量化：把字节还原成浮点 + **代码3（伪代码，`dequantize_row_q4_0`，对应 `ggml/src/ggml-quants.c`）**：
```
for each block:
    d = half_to_float(block.d)
    for i in 0..31:
        q = nibble(block.qs, i)   # 0..15
        x[i] = (q - 8) * d        # q4_0 解量化
```
   补一句量化方向：`quantize_row_q4_0_ref` 里 `d = max / -8`（L06 修订过的那条）。
8. `<h2>` 怎么接进引擎：ggml_type_traits + **图3【type -> traits -> 算子】**（`flow`）：tensor.type=Q4_K -> 查 `type_traits[Q4_K]`（blck_size=256、type_size、to_float=dequantize_row_q4_K）-> 算子按 traits 解量化后计算。
9. 正文 + 一句代码：`ggml_type_traits {type_name, blck_size, type_size, is_quantized, to_float, from_float_ref}`（注意是 `from_float_ref`）——每种类型一行，把"这类型一块多少元素/多少字节/怎么转 float"登记好，算子就能<strong>统一处理所有量化类型</strong>（呼应 L05"结构不变、类型可换"）。
10. **深挖1**"为什么 q4_K 用 256 的超块，而不是像 q4_0 一样 32？"：超块摊薄了"整体 d/dmin"的开销，又用子块 scale 保住局部精度——大块高压缩 + 小块高精度兼得（L06 spark 的字节级印证）。
11. **深挖2**"q4_K 有 qh 吗？q6_K 呢？"：q4_K <strong>没有</strong> qh（只有 d+dmin+scales+qs）；q6_K 才有 ql+qh（高低位分开存 6-bit）。别想当然套用。
12. **深挖3**"算子怎么不为每种量化各写一遍？"：靠 `type_traits` 的 `to_float`/`from_float_ref` 函数指针——算子拿到张量先按 traits 解量化（或在矩阵乘内层即时解量化），所以加一种新量化类型，主要是<strong>填一行 traits + 写解/量化函数</strong>，算子代码基本不动。
13. `key`（✅）：q4_0=[d 2B][qs 16B]=18B；q8_0 每权重 1 字节 int8；K-quant 用 256 超块 + 两层 scale（整体 d/dmin + 子块 6-bit）更准；解量化 `x=(q-8)*d`；`ggml_type_traits` 用函数指针把每种类型接进引擎（`from_float_ref`），算子无需为每种量化各写一遍。
14. `spark`（💡）：把"<strong>每种量化类型长什么样</strong>"收进一张 traits 表、用函数指针暴露"怎么转 float"——于是几十种量化格式能共用同一套张量与算子。L05 说"结构不变、类型可换"，到这里你看到了它字节级的兑现。

必须讲到：q4_0/q8_0 精确布局与字节数；K-quant 超块两层 scale；解量化公式；type_traits 函数指针接入；q4_K 无 qh 的细节。

- [ ] **Step 5: quiz（12）**：
- MCQ1 "q4_0 的一块 18 字节是怎么构成的？" -> 正确："2 字节的 half scale + 16 字节的 32 个 4-bit 量化值"；干扰：18 个权重 / 16B scale+2B 值 / 全是 int8。
- MCQ2 "K-quant（如 q4_K）为什么同 bit 下比 q4_0 更准？" -> 正确："用 256 的超块，整体 d/dmin 之外每个子块还有更细的 scale，局部更贴合"；干扰：用了更多 bit / 不打包 / 不丢数据。
- MCQ3 "ggml 怎么让算子统一处理几十种量化类型？" -> 正确："用 ggml_type_traits 表 + to_float/from_float_ref 函数指针，算子按 traits 解量化，无需为每种类型各写一遍"；干扰：为每种类型写一个算子 / 运行时编译 / 全转成 F32 存盘。
- OPEN "q4_K 和 q4_0 都是约 4-bit，但内存布局差别很大。试着说出至少两点结构上的不同。"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 12 课 · 3 个部分"；commit `feat: add lesson 12 quantization formats in detail (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）

---

## Task 6: 课 13「GGUF 文件格式 / The GGUF file format」

> 第三部分第 6 课（收尾课）。把"一个 .gguf 文件里到底装了什么、怎么被加载"讲透：magic/version、metadata KV、tensor info、对齐、以及 mmap 零拷贝加载。接 L02（gguf-py 转换）、L05（type）、L12（量化块就存在这里）。

**Files:** shell.py、part3.py（追加 LESSON_13）、registry.py、quizzes.py。产出 `13-gguf-format.html`。

- [ ] **Step 1-3: 登记**
```python
("13-gguf-format.html", "GGUF 文件格式", "The GGUF file format",
 "第三部分 · ggml 引擎", "Part 3 · The ggml engine"),
"13-gguf-format.html": ("magic/version · metadata KV · tensor info · 对齐 · mmap",
                        "magic/version; metadata KV; tensor info; alignment; mmap"),
"13-gguf-format.html": part3.LESSON_13,
```

- [ ] **Step 4: 执笔 LESSON_13（双语）**。**结构**：
1. `lead`：前面反复提到的 `.gguf` 文件，到底长什么样？这一课把它从头到尾拆开：文件头、元数据、张量清单、对齐、数据段，以及它怎么被<strong>零拷贝</strong>地加载进内存。
2. `analogy`（🔌）：GGUF 像一个<strong>带目录的集装箱</strong>：箱门上贴着清单（有哪些张量、各在第几米、什么规格=metadata+tensor info），箱内整齐码放货物（张量数据）；卸货时不用全搬出来，按清单<strong>直接定位</strong>（mmap）。
3. `<h2>` 整体布局 + **图1【GGUF 文件结构】**（`vflow` 或 `layers`，本课结构图）：magic "GGUF" -> version(=3) -> tensor_count -> kv_count -> [metadata KV pairs] -> [tensor infos] -> padding(对齐) -> [tensor data]。
4. **代码1（伪代码/注释，按 `ggml/include/gguf.h` 的布局）**：
```
"GGUF"            # 4 字节 magic
version   : u32   # = 3
n_tensors : i64
n_kv      : i64
kv_pairs  : [ (key:str, type:gguf_type, value) ... ]   # 超参/词表/模板...
tensors   : [ (name, n_dims, dims[], ggml_type, offset) ... ]
<padding to alignment>     # 默认 32 字节对齐
tensor_data : <raw bytes>  # 权重(可能是 L12 的量化块)
```
5. `<h2>` 元数据 KV：模型的"说明书" + **图2【metadata KV 举例】**（`<table class="t">`）：键 / 类型 / 含义，举真实键：`general.architecture`(str)、`llama.block_count`(u32 层数)、`llama.embedding_length`(u32 n_embd)、`tokenizer.ggml.tokens`(array)、`general.alignment`(u32)。
6. 正文：metadata 是<strong>自描述</strong>的关键——加载器不必"猜"模型结构，全部超参、词表、聊天模板都写在 KV 里（呼应 L04 说的"GGUF 头里直接读到 n_layer/n_embd"）；`gguf_type` 枚举定义了值的类型（u8/i32/f32/str/array/...）。
7. `<h2>` tensor info + 对齐 + **图3【按 offset 定位张量】**（`cells` 或 `timeline`）：每个 tensor info 记 name+dims+type+offset；数据段按 `GGUF_DEFAULT_ALIGNMENT=32` 对齐；offset 指向数据段内的相对位置。
8. `<h2>` 加载：mmap 零拷贝 + **代码2（伪代码，`gguf_init_from_file` + `llama_mmap`）**：
```
ctx = gguf_init_from_file(path)        # 读 magic/version/KV/tensor infos
assert magic == "GGUF" and version == 3
mapping = mmap(file, PROT_READ)        # 整文件只读映射, 不拷贝
for t in tensors:
    t.data = mapping + data_off + t.offset   # 张量数据直接指进映射
```
   讲 mmap 的妙处：<strong>不把几 GB 权重读进内存再拷一遍</strong>，而是把文件映射进地址空间，张量 data 指针直接落在映射上，用到哪页 OS 才加载哪页——这就是大模型"秒加载"的原因（`src/llama-mmap.cpp` / `llama-model-loader.cpp`）。
9. **深挖1**"为什么要对齐到 32 字节？"：让张量数据段起点对齐，便于 SIMD/后端按对齐地址高效读取、也便于 mmap 按页处理；`general.alignment` 可覆盖默认值。
10. **深挖2**"GGUF 和老的 GGML 格式比好在哪？"：GGUF 是<strong>自描述 + 可扩展</strong>——新增超参只是多一个 KV，不破坏旧文件；统一存超参/词表/模板，免去外部配置；版本号=3 标记格式演进。
11. **深挖3**"mmap 加载，模型会算进'内存占用'吗？"：mmap 的页是<strong>按需载入、可被系统回收</strong>的文件页，多进程还能共享同一映射；所以"看起来占了很多虚拟内存"但物理内存是惰性、可共享的——这也是同机起多个实例省内存的原因。
12. `key`（✅）：GGUF = magic "GGUF" + version(3) + KV 计数/张量计数 + metadata KV（自描述超参/词表/模板）+ tensor infos（name/dims/type/offset）+ 32 字节对齐 + 张量数据；`gguf_init_from_file` 读头部；权重用 <strong>mmap 只读零拷贝</strong>加载，张量 data 直接指进映射。
13. `spark`（💡）：把"<strong>模型是什么</strong>"（自描述元数据）和"<strong>模型的数</strong>"（对齐的张量数据）打包进<strong>一个可 mmap 的文件</strong>——于是"一个文件到处跑"（L01）在格式层面落地：拷走一个 .gguf，任何 llama.cpp 都能自己读懂它、秒加载它。第三部分到此结束，你已看清 ggml 引擎从内存、图、执行、算子到格式的全貌。

必须讲到：GGUF 整体布局；自描述 metadata KV；tensor info + 对齐；mmap 零拷贝加载；version=3/alignment=32。

- [ ] **Step 5: quiz（13）**：
- MCQ1 "GGUF 文件里的 metadata KV 主要存什么？" -> 正确："模型的自描述信息——架构、层数/维度等超参、词表、聊天模板等，让加载器无需猜测结构"；干扰：只有权重 / 只有版本号 / 源代码。
- MCQ2 "llama.cpp 用 mmap 加载 GGUF 权重的好处是？" -> 正确："只读映射文件、零拷贝，按需分页载入，不必把几 GB 权重先读进内存再拷一遍"；干扰：能修改权重 / 自动量化 / 加密。
- MCQ3 "GGUF 文件开头的 magic 和 version 是？" -> 正确："magic 是 'GGUF'，当前 version 是 3"；干扰：'GGML'/1 / 'LLMA'/2 / 没有 magic。
- OPEN "GGUF 把超参和词表都写进文件自描述，相比'权重文件 + 外部 config' 的老办法有什么好处？"

- [ ] **Step 6-7: 重建+校验+commit**（index "共 13 课 · 3 个部分"；commit `feat: add lesson 13 GGUF file format (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）

---

## Task 7: M3 验收（清重建 + 密度/CJK 审计 + 里程碑）

> 无新增内容；端到端验证第三部分 6 课全部达标、与前两部分衔接无回归。

**Files:** 无新增（仅可能补一处副标题/交叉引用的一致性微调）。

- [ ] **Step 1: 清重建 + 双校验（产物零漂移）**
```bash
cd /home/verden/course/llama-cpp-visual-guide
rm -f index.html lessons/*.html
cd src && python build.py && python check_html.py && python check_links.py
cd .. && git status --short        # 期望干净
```
Expected：0 error / 0 warning、全链接解析、`git status` 干净。

- [ ] **Step 2: 6 课密度 / CJK / 图 / 片段 / 深挖 审计**
```bash
cd /home/verden/course/llama-cpp-visual-guide && python3 - <<'PY'
import re, sys
sys.path.insert(0, "src")
import registry
DIAG = ("layers","vflow","flow","cols","cellgroup","timeline")
for f in ["08-ggml-core-objects.html","09-compute-graph.html","10-graph-execution.html",
          "11-core-operators.html","12-quant-formats.html","13-gguf-format.html"]:
    z = registry.CONTENT[f]["zh"]; e = registry.CONTENT[f]["en"]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", z))
    encjk = len(re.findall(r"[\u4e00-\u9fff]", e))
    diag = max(sum(z.count(f'class="{c}"') for c in DIAG) + z.count('<table class="t"'),
               sum(e.count(f'class="{c}"') for c in DIAG) + e.count('<table class="t"'))
    acc = z.count('class="accordion"'); pre = z.count("<pre")
    ok = cjk >= 4000 and encjk == 0 and diag >= 3 and acc >= 2 and pre >= 2
    print(f"{f:30s} CJK={cjk:5d} enCJK={encjk} diag>={diag} acc={acc} pre={pre}  {'OK' if ok else 'CHECK'}")
PY
```
Expected：每课 `CJK >= 4000`、`enCJK = 0`、`diag >= 3`、`acc >= 2`、`pre >= 2`。任一 `CHECK` 回到对应 Task 补足。

- [ ] **Step 3: 导航与交叉引用检查**
```bash
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 13 课 · 3 个部分' index.html && echo "index: 13 课 · 3 部分"
for n in 08 09 10 11 12 13; do grep -ql "href=\"lessons/$n-" index.html && echo "toc links $n"; done
grep -RoE '第 [0-9]+ 课' lessons/ | awk -F'第 | 课' '{if($2+0>40) print "OUT:", $0}'   # 期望无输出
```
Expected：index 显示 `13 课 · 3 个部分`、08-13 均在目录、无越界 `第 N 课` 引用、prev/next 链完整。

- [ ] **Step 4: 标记里程碑**

把 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 的 `- [ ] M3 ...` 改为 `- [x] M3 ...`，commit（`docs: mark M3 (Part 3 ggml engine) complete` + `Assisted-by: GitHub Copilot`）。随后按 finishing-a-development-branch 把 `build/m3-part3` 合并回 master（`--no-ff`）并删分支。

---

## Self-Review（计划自检，作者本人执行）

**1. Spec 覆盖**：第三部分 6 课（08 核心对象 / 09 计算图 / 10 执行调度 / 11 算子 / 12 量化格式 / 13 GGUF）= Spec §"第三部分"六条，逐一对应 Task 1-6；加强标准（3-5 图含概念图、2-3 片段、2-3 深挖、~4000+ CJK、en 逐段对齐无 CJK）写进"统一交付标准"并由 Task 7 Step 2 脚本量化校验。✅

**2. Placeholder 扫描**：每个 Task 的 Step 4 给了逐卡片/逐图/逐片段/逐深挖的具体简报与真实源码片段（`ggml_init_params`/`ggml_mul_mat`/`block_q4_K`/GGUF 布局等实体），Step 5 给了完整 quiz 要点，Step 6 给了精确命令与期望。无 TBD/TODO/"类似上文"占位。✅

**3. 一致性**：六课文件名（`08-ggml-core-objects`/`09-compute-graph`/`10-graph-execution`/`11-core-operators`/`12-quant-formats`/`13-gguf-format`）在 PAGES/SUBTITLES/registry/quizzes/校验命令中一致；part 标签统一 `第三部分 · ggml 引擎` / `Part 3 · The ggml engine`；`part3.py` 由 Task 1 创建、Task 2-6 追加；registry 的 `import part3` 仅 Task 1 引入。index 部分数从 2 -> 3。✅

**4. 防重复**：L08 vs L05（L05 字段、L08 内存来源）、L12 vs L06（L06 直觉、L12 字节级）、L11 vs L04（L04 数学、L11 算子与形状）、L10 vs L07（L07 后端是什么、L10 怎么调度）均已在 task 标题/导语划清边界。✅

**5. 源码事实**：均来自 2026-06-15 对真实源树的核验（explore 子代理已逐条确认 `from_float_ref` 命名、`block_q4_K` 无 qh、`d=max/-8`、`GGUF_VERSION=3`/`ALIGNMENT=32` 等易错点）。引用一律"文件+符号"。✅

---

## 执行方式

按用户既定：**subagent 驱动**（每课 spec 合规 + 代码质量两段审查，全程本模型 `claude-opus-4.8`）。但鉴于 M2 经验——内容子代理写大段 HTML 会卡死——**控制者直接执笔每课内容**，再对每课跑 spec + 质量双审子代理（只读）。分支 `build/m3-part3`，6 课顺序产出，Task 7 验收后合并回 master。






