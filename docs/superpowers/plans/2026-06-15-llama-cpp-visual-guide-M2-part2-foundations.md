# M2 · 第二部分「前置基础」实施计划 (Part 2 · Foundations)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 写出第二部分的 4 课（lessons 04-07：大模型推理基础 / 张量是什么 / 量化入门 / 构建系统与后端），为后续 ggml/llama 内部章节打好"概念地基"，每课均达到 M1.5/M1.6 确立的加强标准。

**Architecture:** 沿用 M0 脚手架与 M1 的"每课一组任务"模式——新建 `src/part2.py` 放 4 个 `LESSON_xx` 双语字典，并在 `shell.PAGES` / `registry.CONTENT` / `shell.SUBTITLES` / `quizzes.QUIZZES` 四处登记；不引入任何新组件或新依赖，全部复用既有设计系统与示意图 CSS 组件。每课产出后立即 `build.py` + `check_html.py` + `check_links.py` 全绿再 commit。

**Tech Stack:** 零依赖 Python 静态生成器（`src/*.py`）；纯 HTML/CSS 双语内容（`lang-zh`/`lang-en` + `bi()`）；校验脚本 `check_html.py`（结构/导航/计数/中文密度软检查）、`check_links.py`（死链）。源码事实核对对象：本机 llama.cpp 源树 `/home/verden/course/llama.cpp`。

---

## 统一交付标准（每课都要满足，源自 Spec §5/§6）

每一课（一个 `LESSON_xx`，含 `"zh"` 与 `"en"` 两份）必须包含：

- **导语** `<p class="lead">`：一段话点题（这一课要回答什么问题、为什么重要）。
- **教学卡片**（按需取用，建议齐全）：`macro`（🌍 宏观理解）、`detail`/卡片正文（拆解机制）、`analogy`（🔌 生活类比）、`key`（✅ 关键要点小结）、`spark`（💡 设计洞察收尾）。
- **图：3-5 张，类型多样**（硬性）。可用的示意图 CSS 组件（**只复用、不要新造**）：
  - `layers`（自底向上分层塔）、`flow`（横向流程）、`vflow`（纵向步骤）、`cols`（并排对比）、
  - `cellgroup`/`cells`/`cell`（格子示意图：`.scale`/`.hl`/`.q`/`.dim` 修饰 + `.lab` 标注 + `.sep` 分隔）、
  - `timeline`/`lane`/`tslot`（`.span`/`.now` 时间线）、`table class="t"`（对照表）。
  - **原理课至少 1 张"概念示意图"**（用 `cells`/`timeline` 画原理草图，深色模式自适应、自包含、不外链图片/SVG）。
- **代码：2-3 段**（硬性）。先**伪代码**讲思路，再给**从真实源码简化的片段**对照，并在文中标注**来源文件 + 符号名**（不写死行号）。代码片段内部 **ASCII 优先**（`-`/`->`/`...`），不用 em-dash / unicode 箭头 / `×` / `…`。
- **折叠深挖：2-3 个** `<details class="accordion">`（配 `<p class="acc-intro">` 引导句 + `<div class="acc-body">` 正文），新手可跳过。
- **纯中文正文 ~4000+ 汉字**（CJK，按 `\u4e00-\u9fff` 计，不含英文/代码/路径）。英文 `"en"` 与中文同义对齐（非逐字直译，可略简，但信息点齐全）。

每课的登记清单（缺一处都会导致 build/check 失败或课程不显示）：

1. `src/shell.py` 的 `PAGES`：追加 `(filename, 标题_zh, 标题_en, "第二部分 · 前置基础", "Part 2 · Foundations")`。
2. `src/part2.py`：写 `LESSON_xx = {"zh": r"""...""", "en": r"""..."""}`。
3. `src/registry.py`：`import part2` 并在 `CONTENT` 追加 `"filename": part2.LESSON_xx`。
4. `src/shell.py` 的 `SUBTITLES`：追加 `"filename": ("副标题_zh", "subtitle_en")`。
5. `src/quizzes.py` 的 `QUIZZES`：追加该课 `{"mcq": [...2-4 题...], "open": [...]}`（双语）。
6. 运行 `cd src && python build.py && python check_html.py && python check_links.py`，**0 error / 0 warning + 全链接解析**后再 commit。

**门槛复核**：`check_html.py` 的软检查含 `MIN_DIAGRAMS`（视觉块密度，两语都数）、`MIN_CJK`（zh 汉字数）、`MAX_LESSON=40`（`第 N 课` 交叉引用上限）。新课必须让这些软检查都不报 WARN。

## 避免与第一部分重复（重要）

第二部分是"基础打底"，但**不能复述第一部分已讲透的内容**，必须换更深/更细的角度：

- **L04 大模型推理基础**：L03 已讲"流水线节奏"（分词->图->logits->采样->循环、prefill/decode 两种节奏、KV cache 作为优化）。**L04 改讲底层"模型与数学"**：decoder-only transformer 一个 block 到底算了什么（embed -> 每层 注意力+FFN -> 末层 norm -> 词表 logits）、因果掩码（causal mask）、自回归的数学形式、以及**为什么 KV cache 在数学上是精确的**（因果掩码 => 旧 token 的 K/V 不随新 token 改变）。即"L03 那个循环凭什么成立"的下一层。
- **L05 张量是什么**：L02 提过 `ggml_tensor` 存在；**L05 讲清 shape/stride/行优先与字段语义**，以及 view/转置为何不拷贝数据。
- **L06 量化入门**：L01/L02 提过"4/5/8 bit、QK4_0=32、18 字节"；**L06 讲清块量化的机制与直觉**（每块一个 scale、Q4_0 字节布局、Q8_0、K-quant 超块思想）。格式硬核细节留给 L12，本课只到"入门直觉"。
- **L07 构建系统与后端**：第一部分未展开；**L07 讲 CMake 怎么编、后端怎么选、产物有哪些**。

## 源码事实（已核对的锚点，写作时仍需对照源树确认精确字段）

> 核对源树：`/home/verden/course/llama.cpp`。引用以"文件 + 符号名"为主，**不写死行号**，并在课内注明"以本仓库 <日期> 源码为准"。

- **L05**：`ggml/include/ggml.h` 的 `struct ggml_tensor` 字段 = `type, buffer, ne[GGML_MAX_DIMS], nb[GGML_MAX_DIMS], op, op_params, flags, src[GGML_MAX_SRC], view_src, view_offs, data, name, extra`；`GGML_MAX_DIMS = 4`；stride 注释：`nb[0]=ggml_type_size(type)`、`nb[1]=nb[0]*(ne[0]/ggml_blck_size(type))+padding`、`nb[i]=nb[i-1]*ne[i-1]`。辅助：`ggml_nbytes`、`ggml_is_contiguous`、`ggml_new_tensor_2d`。
- **L06**：`ggml/src/ggml-common.h`：`QK4_0=32`、`QK8_0=32`、`QK_K=256`；`block_q4_0`（`ggml_half d; uint8_t qs[QK4_0/2];` => 2+16=18 字节，4.5 bit/权重）、`block_q8_0`（`ggml_half d; int8_t qs[QK8_0];`）、`block_q4_K`（super-block 联合体，含 `d`/`dmin`）。解量化：`ggml/src/ggml-quants.c` 的 `dequantize_row_q4_0`（`x = (q - 8) * d`）。工具：`llama-quantize`。imatrix = 误差加权（**不是**位宽分配）。
- **L07**：`ggml/CMakeLists.txt` 选项 `GGML_CPU`(默认 ON)/`GGML_BLAS`/`GGML_CUDA`/`GGML_HIP`/`GGML_VULKAN`/`GGML_METAL`/`GGML_SYCL`/`GGML_OPENCL`；构建命令见 `docs/build.md`（`cmake -B build [-DGGML_CUDA=ON]` -> `cmake --build build -j`）；产物在 `build/bin`（`llama-cli`/`llama-server`/`llama-quantize`/`libllama`/`libggml` 等）；后端抽象 `ggml/include/ggml-backend.h`；层卸载 `-ngl`/`n_gpu_layers`（`common/`）。
- **L04**：decoder-only 计算图构建在 `src/llama-graph.cpp`/`src/llama-model.cpp`（`build_attn`/`build_attn_mha`、`llm_graph_context`）；GQA：`src/llama-hparams.h` 的 `n_head` 与 `n_head_kv`（k/v 头数 <= q 头数）；取 logits：`llama_get_logits_ith`（返回 `n_vocab` 个 float）；归一化/激活算子 `ggml_rms_norm`/`ggml_soft_max_ext`/`ggml_rope`（`ggml/include/ggml.h`）。

---

## Task 1: 课 04「大模型推理基础 / LLM inference fundamentals」

> 第二部分第 1 课，也是 `part2.py` 的首课（本任务**创建** `src/part2.py`）。按"详尽简报 + 子代理执笔"产出：先做登记（PAGES/SUBTITLES/registry），再据简报用中/英执笔 `LESSON_04`，加 quiz。**与 L03 严格区分**：L03 讲流水线节奏，本课讲模型与数学底座（block 结构 · 因果掩码 · 自回归 · KV cache 为何精确）。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 04 一项）
- Create: `src/part2.py`（新增 `LESSON_04`）
- Modify: `src/registry.py`（`import part2` + `CONTENT` 加 04）
- Modify: `src/quizzes.py`（`QUIZZES` 加 04）
- 产出：`index.html`（现 4 课 · 2 个部分）、`lessons/04-llm-inference-basics.html`

- [ ] **Step 1: 登记 PAGES**（03 那条之后追加）
```python
    ("04-llm-inference-basics.html", "大模型推理基础", "LLM inference fundamentals",
     "第二部分 · 前置基础", "Part 2 · Foundations"),
```

- [ ] **Step 2: 登记 SUBTITLES**
```python
    "04-llm-inference-basics.html": ("decoder-only · 因果掩码 · 自回归 · KV cache 为何精确",
                                     "decoder-only; causal mask; autoregression; why the KV cache is exact"),
```

- [ ] **Step 3: 登记 registry**（在 `registry.py` 顶部 `import part1` 旁加 `import part2`，并在 `CONTENT` 追加）
```python
    "04-llm-inference-basics.html": part2.LESSON_04,
```

- [ ] **Step 4: 在 part2.py 执笔 LESSON_04（中/英双语）**

新建 `src/part2.py`，文件头 docstring `"""Content for Part 2 (foundations)."""`，写 `LESSON_04 = {"zh": r"""...""", "en": r"""..."""}`，文风对齐课 01-03。**结构（按序）**：

1. `<p class="lead">`：L03 看过"循环怎么转"；这一课往下钻一层——一个 transformer block 到底算了什么、自回归为什么成立、**KV cache 凭什么是精确的（不是近似）**。
2. `<div class="card analogy">`（🔌 生活类比）：自回归像**接龙写句子**——每次只看已写出的字、猜下一个最可能的字，写下后再回头看全部已写内容，继续猜。
3. `<h2>` decoder-only：一个 block 在算什么 + **图1【decoder block 结构】**（`vflow` 纵向步骤）：词嵌入 -> 每层[ RMSNorm -> 自注意力(含 RoPE) -> 残差 -> RMSNorm -> FFN(SwiGLU) -> 残差 ] × N 层 -> 末层 RMSNorm -> 输出投影 -> 词表 logits。
4. 正文：注意力 = token 之间**唯一互相交流**的地方；FFN = 逐 token 的非线性加工；残差+norm = 训练稳定。**代码1（伪代码，一层 forward）**：
   ```
   def layer(x):                 # x: [n_tokens, n_embd]
       a = attn(rms_norm(x))     # tokens talk to each other here
       x = x + a                 # residual
       f = ffn(rms_norm(x))      # per-token non-linear mix
       return x + f              # residual
   ```
5. `<h2>` 因果掩码：只能回头看 + **图2【因果掩码概念图】**（`cellgroup`/`cells` 画下三角，本课"概念示意图"）：行=query token、列=key token；格子亮(`.hl`)=可见(j<=i)、暗(`.dim`)=屏蔽(j>i)。
6. 正文：因果掩码保证第 i 个 token 只能注意到 <=i 的 token——这就是"自回归"在注意力层的实现；softmax 前把被屏蔽位置置为 `-inf`。**代码2（源码简化片段）**：注意力打分（对应 `src/llama-graph.cpp` 的 `build_attn`/`build_attn_mha`，用 `ggml_soft_max_ext` 带 mask），简化几行：
   ```
   kq  = ggml_mul_mat(ctx, k, q);            // scores [n_kv, n_q]
   kq  = ggml_soft_max_ext(ctx, kq, mask,    // mask: causal -inf on j>i
                           scale, max_bias);
   kqv = ggml_mul_mat(ctx, v, kq);           // weighted sum of values
   ```
   注明"以本仓库 <核验日期> 源码为准，真实实现在 `llama-graph.cpp`"。
7. `<h2>` 自回归 + 为什么 KV cache 是精确的 + **图3【自回归循环】**（`flow` 横向）：x1..xn -> 模型 -> **末位** logits -> 采样 -> x(n+1) -> 追加 -> 重复。
8. **图4【KV cache 精确性概念图】**（`cells`，可选第 5 图）：第 n 步算出 token1..n 的 K/V；第 n+1 步只新增 1 列 K/V，前 n 列**完全不变**（`.hl` 标新列、旧列 `.dim` 标"复用"）。因为因果掩码下旧 token 看不到新 token，其 K/V 与未来无关 => 缓存精确、非近似。**代码3（可选）**：`llama_get_logits_ith(ctx, -1)` 只取最后位置、返回 `n_vocab` 个 float。
9. **深挖1**（accordion）"为什么只有 decoder，没有 encoder?"：原始 Transformer 是 encoder-decoder（翻译用），GPT 类只留 decoder 做"预测下一个词"，更简单、更适合生成；llama 系均为 decoder-only。
10. **深挖2**"MHA / GQA / MQA：KV cache 还能更省"：K/V 头数可少于 Q 头数（GQA），KV cache 体积按 `n_head_kv` 计（`src/llama-hparams.h` 的 `n_head`/`n_head_kv`）；MQA 是极端（1 个 KV 头）。
11. **深挖3**"logits 到底是什么？temperature 怎么作用"：logits = 词表上每个 token 的未归一化分数；`softmax(logits / T)` 得概率；T↑更随机、T↓更确定（呼应 L03 采样）。
12. `<div class="card key">`（✅ 关键要点）：decoder-only = 嵌入 + N×(注意力+FFN) + 末层投影到词表 logits；注意力是 token 间唯一交流处；**因果掩码 => 自回归**；**KV cache 精确而非近似**（旧 K/V 不随新 token 改变）；每步只在最后一个位置取 logits。
13. `<div class="card spark">`（💡 设计洞察）：**因果掩码这一个约束**同时带来三件事——可并行的 prefill、可缓存的 K/V、以及"自回归"本身；llama.cpp 几乎所有推理优化都建立在它之上。

**必须讲到的论点**：block 组成；注意力=交流点；因果掩码=自回归的实现；KV cache 精确性的数学理由；末位取 logits。
**要核实**（对照 `/home/verden/course/llama.cpp`）：`build_attn`/`build_attn_mha` 与 `ggml_soft_max_ext` 在 `llama-graph.cpp`；`n_head`/`n_head_kv` 在 `llama-hparams.h`；`llama_get_logits_ith` 返回 `n_vocab`；`ggml_rms_norm`/`ggml_rope`/`ggml_soft_max_ext` 算子名。引用只用"文件+符号"。

- [ ] **Step 5: 在 quizzes.py 增加 04 的双语 quiz**（schema 同 01；2-3 mcq + 1 open）
- **MCQ1**：q `{"zh":"为什么说 KV cache 是“精确优化”而不是“近似”？","en":"Why is the KV cache an exact optimization rather than an approximation?"}`；opts：①`因为因果掩码下旧 token 看不到新 token，它们的 K/V 不随新 token 改变`（✅）②`因为它把 logits 也缓存了` ③`因为权重被量化了` ④`因为只缓存了最后一层`；why 解释因果掩码 => 旧 K/V 与未来无关。
- **MCQ2**：q `{"zh":"decoder-only 模型里，token 之间“互相交流”主要发生在？","en":"In a decoder-only model, where do tokens mainly “talk to each other”?"}`；opts：①`带因果掩码的自注意力层`（✅）②`FFN（前馈层）` ③`RMSNorm` ④`输出投影`；why：FFN 是逐 token 的，注意力才让 token 互看。
- **OPEN**：`{"zh":"如果去掉因果掩码（允许看到未来 token），自回归生成还成立吗？KV cache 还精确吗？","en":"If you removed the causal mask (letting tokens see the future), would autoregressive generation still hold? Would the KV cache still be exact?"}`

- [ ] **Step 6: 重建 + 校验**

Run:
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 4 课 · 2 个部分' index.html && echo "index shows 4 lessons / 2 parts"
grep -q 'href="lessons/04-llm-inference-basics.html"' index.html && echo "toc links 04"
python3 -c "import re,sys; sys.path.insert(0,'src'); import registry; z=registry.CONTENT['04-llm-inference-basics.html']['zh']; print('L04 CJK:', len(re.findall(r'[\u4e00-\u9fff]', z)))"
```
Expected：`structural check passed`（0 error / 0 warning）、`all N internal links resolve`、index 显示 4 课 · 2 个部分、04 链接在目录、`L04 CJK >= 4000`（最低 3000 不报 WARN）。

- [ ] **Step 7: Commit**
```bash
git add src/shell.py src/registry.py src/part2.py src/quizzes.py index.html lessons/04-llm-inference-basics.html
git commit -m "feat: add lesson 04 LLM inference fundamentals (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 2: 课 05「张量是什么 / What is a tensor」

> 第二部分第 2 课。先登记（PAGES/SUBTITLES/registry），再据简报在 `part2.py` **追加** `LESSON_05`，加 quiz。把 `ggml_tensor` 讲到"摸得着"：shape/stride/行优先 + view 零拷贝。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 05）
- Modify: `src/part2.py`（追加 `LESSON_05`）
- Modify: `src/registry.py`（`CONTENT` 加 05）
- Modify: `src/quizzes.py`（`QUIZZES` 加 05）
- 产出：`index.html`（现 5 课 · 2 个部分）、`lessons/05-tensors.html`

- [ ] **Step 1: 登记 PAGES**（04 之后追加）
```python
    ("05-tensors.html", "张量是什么", "What is a tensor",
     "第二部分 · 前置基础", "Part 2 · Foundations"),
```

- [ ] **Step 2: 登记 SUBTITLES**
```python
    "05-tensors.html": ("shape/stride/行优先 · ggml_tensor 字段 · view 零拷贝",
                        "shape/stride/row-major; ggml_tensor fields; zero-copy views"),
```

- [ ] **Step 3: 登记 registry**
```python
    "05-tensors.html": part2.LESSON_05,
```

- [ ] **Step 4: 在 part2.py 执笔 LESSON_05（中/英双语）**。**结构（按序）**：

1. `<p class="lead">`：ggml 里一切数据都是"张量"。看懂 shape / stride / 行优先，就能看懂后面所有图与算子；这一课把 `ggml_tensor` 这个结构体讲到摸得着。
2. `<div class="card analogy">`（🔌 生活类比）：张量像一排**储物柜阵列**——`ne[]` 是"每排几个柜、共几排"，`nb[]` 是"走到下一个柜 / 下一排要迈多少步（字节）"，`data` 是阵列的起点地址。
3. `<h2>` 张量 = 形状 + 类型 + 一块连续内存 + **图1【shape 概念图】**（`cells`）：一个 ne=[2,3]（`ne[0]=2` 列、`ne[1]=3` 行）张量画成网格，标注 `ne[0]` 为"最内/最快维"。
4. 正文：`ne[GGML_MAX_DIMS]`(=4) 存每维元素数、`type` 存数据类型、`data` 指向内存、`op`/`src` 记录"它在计算图里怎么来的"。**代码1（源码简化，来自 `ggml/include/ggml.h` 的 `struct ggml_tensor`）**：
   ```
   struct ggml_tensor {
       enum ggml_type type;            // F32 / F16 / Q4_0 ...
       int64_t ne[4];                  // #elements per dim (ne[0] = innermost)
       size_t  nb[4];                  // byte strides
       enum ggml_op op;                // how it was produced (graph node)
       struct ggml_tensor * src[...];  // inputs (back-pointers)
       struct ggml_tensor * view_src;  // set if this tensor is a view
       void * data;                    // the actual bytes
       char name[...];
   };
   ```
   注明 `GGML_MAX_DIMS = 4`、"以本仓库 <核验日期> 源码为准"。
5. `<h2>` 行优先与 stride：`nb[]` 怎么算 + **图2【行优先内存布局概念图】**（`cells`/`timeline`，本课"概念示意图"）：ne=[3,2]（`ne[0]=3`）张量在**线性内存**里：行0 的 3 个元素连续摆、紧接行1 的 3 个；标注 `nb[0]=type_size`、`nb[1]=nb[0]*ne[0]`。
6. 正文：ggml 是**行优先**（`ne[0]` 最内、步长最小）。`ggml.h` 注释给了公式：`nb[0]=ggml_type_size(type)`、`nb[1]=nb[0]*(ne[0]/ggml_blck_size(type))+padding`、`nb[i]=nb[i-1]*ne[i-1]`。**代码2（伪代码，索引->偏移）**：
   ```
   # element (i0, i1, i2, i3) lives at:
   offset = i0*nb[0] + i1*nb[1] + i2*nb[2] + i3*nb[3]
   ptr    = (char*)tensor->data + offset
   ```
7. `<h2>` view / 转置为什么不拷贝数据 + **图3【view vs copy 对比图】**（`cols` 并排）：左=原张量 [2,3]；右=转置 [3,2]——只交换 `ne[]`/`nb[]`，`data` 与 `view_src` 指回原张量，**一个字节都不搬**。
8. 正文：转置 / reshape / view 只改 `ne`/`nb`/偏移、复用同一块 `data`（`view_src` 记住来源）；好处=零拷贝省内存，代价=结果**非连续**（`ggml_is_contiguous` 为假），某些算子需先 `ggml_cont`。**图4（可选）【连续 vs 非连续】**（`cells`）：连续=顺着 `nb` 紧挨；非连续=有跳步。
9. **深挖1**"ggml 的维度顺序为什么和 PyTorch 相反"：ggml `ne[0]` 是最内/连续维（步长最小）；numpy/PyTorch 习惯最后一维连续。`[batch, seq, dim]` 在 ggml 里写成 `ne=[dim, seq, batch]`，读图时当心。
10. **深挖2**"`nb`（strides）到底怎么来的 / 量化类型为何要 `/blck_size`"：量化类型把一"块"(如 32 个权重)打包成定长字节，故 `nb[0]` 要按"每块字节数 / 块内元素数"折算，ggml 用 `ggml_blck_size` + `ggml_type_size` 处理。
11. **深挖3**"怎么算一个张量占多少字节"：`ggml_nbytes(t)` 按 `ne` 与 `type` 算；连续张量约等于 `ne[最高维] * nb[最高维]`。
12. `<div class="card key">`（✅ 关键要点）：张量 = `type` + `ne[4]`(形状) + `nb[4]`(字节步长) + `data`(内存) + `op`/`src`(图里怎么来的)；ggml **行优先、`ne[0]` 最内**；view/转置**零拷贝**（改 `ne`/`nb`、复用 `data`）；非连续张量要留意。
13. `<div class="card spark">`（💡 设计洞察）：把"形状"(`ne`/`nb`)与"数据"(`data`)分开存——正是这个设计让转置/切片/广播都变成"改几个数字"而非"搬内存"，也让同一块权重能被计算图以不同视角反复使用，是 ggml 高效的根基之一。

**必须讲到的论点**：`ne`/`nb`/`type`/`data`/`op`/`src`/`view_src` 语义；行优先 + stride 公式；view/转置零拷贝；非连续概念。
**要核实**（对照源树）：`struct ggml_tensor` 字段名（`ggml/include/ggml.h`）；`GGML_MAX_DIMS=4`；`nb` 公式（注释）；`ggml_nbytes`/`ggml_is_contiguous`/`ggml_blck_size`/`ggml_type_size`/`ggml_cont` 存在。

- [ ] **Step 5: 在 quizzes.py 增加 05 的双语 quiz**（2-3 mcq + 1 open）
- **MCQ1**：q `{"zh":"在 ggml 里把一个张量“转置”，主要改变了什么？","en":"Transposing a ggml tensor mainly changes what?"}`；opts：①`只交换 ne[]/nb[]（步长），复用同一块 data，不搬数据`（✅）②`复制出一块新内存` ③`改变了 type` ④`重新量化了权重`；why：转置是改元数据的视图操作，零拷贝。
- **MCQ2**：q `{"zh":"ggml 张量里哪一维是“最内/连续（步长最小）”的维？","en":"Which dimension of a ggml tensor is the innermost/contiguous (smallest stride) one?"}`；opts：①`ne[0]`（✅）②`ne[3]` ③`由 type 决定` ④`都一样`；why：ggml 行优先，`nb[0]=type_size` 最小。
- **OPEN**：`{"zh":"给一个 ne=[4,3]（ne[0]=4）的 F32 张量，nb[0] 与 nb[1] 各是多少字节？（F32=4 字节）","en":"For an F32 tensor with ne=[4,3] (ne[0]=4), what are nb[0] and nb[1] in bytes? (F32 = 4 bytes)"}`

- [ ] **Step 6: 重建 + 校验**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 5 课 · 2 个部分' index.html && echo "index shows 5 lessons / 2 parts"
grep -q 'href="lessons/05-tensors.html"' index.html && echo "toc links 05"
python3 -c "import re,sys; sys.path.insert(0,'src'); import registry; z=registry.CONTENT['05-tensors.html']['zh']; print('L05 CJK:', len(re.findall(r'[\u4e00-\u9fff]', z)))"
```
Expected：0 error / 0 warning、全链接解析、index 显示 5 课 · 2 个部分、05 链接在目录、`L05 CJK >= 4000`。

- [ ] **Step 7: Commit**
```bash
git add src/shell.py src/registry.py src/part2.py src/quizzes.py index.html lessons/05-tensors.html
git commit -m "feat: add lesson 05 tensors (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 3: 课 06「量化入门 / Quantization, intuitively」

> 第二部分第 3 课。先登记，再在 `part2.py` **追加** `LESSON_06`，加 quiz。讲**机制与直觉**：为什么能压、块量化、Q4_0/Q8_0/K-quant；硬核字节细节留到 L12。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 06）
- Modify: `src/part2.py`（追加 `LESSON_06`）
- Modify: `src/registry.py`（`CONTENT` 加 06）
- Modify: `src/quizzes.py`（`QUIZZES` 加 06）
- 产出：`index.html`（现 6 课 · 2 个部分）、`lessons/06-quantization-intro.html`

- [ ] **Step 1: 登记 PAGES**（05 之后追加）
```python
    ("06-quantization-intro.html", "量化入门", "Quantization, intuitively",
     "第二部分 · 前置基础", "Part 2 · Foundations"),
```

- [ ] **Step 2: 登记 SUBTITLES**
```python
    "06-quantization-intro.html": ("为什么量化 · 块量化 · Q4_0/Q8_0/K-quant 一览",
                                   "why quantize; block quantization; Q4_0/Q8_0/K-quant tour"),
```

- [ ] **Step 3: 登记 registry**
```python
    "06-quantization-intro.html": part2.LESSON_06,
```

- [ ] **Step 4: 在 part2.py 执笔 LESSON_06（中/英双语）**。**结构（按序）**：

1. `<p class="lead">`：量化是 llama.cpp 能在消费级硬件跑大模型的"压缩术"。L01 提过有 4/5/8 bit，这一课讲**为什么能压、怎么压（块量化）、Q4_0/Q8_0/K-quant 各是什么**；硬核字节细节留到 L12，这里先建立直觉。
2. `<div class="card analogy">`（🔌 生活类比）：量化像把高清照片**按小块压缩**——每个小块记一个"亮度基准(scale)"，块内每个像素只存"相对基准的几档偏移"。块小、基准准，压完还很像原图。
3. `<h2>` 为什么要量化：显存与带宽 + **图1【内存账概念图】**（`cols` 或 `table class="t"`）：7B 参数 × `{fp16≈14GB, Q8_0≈7GB, Q4_0≈3.5GB}`；点明推理瓶颈常是**内存带宽**——权重更小 => 每步搬运更少 => 更快。
4. 正文：权重是一大堆浮点数，fp16 每个 2 字节、7B 就 14GB，消费级设备吃不下。量化 = 用更少 bit 近似存权重，显存↓带宽↓速度↑，代价是少量精度损失。
5. `<h2>` 块量化：每块一个 scale + **图2【块量化概念图】**（`cells`，本课"概念示意图"）：32 个 fp 权重 -> 1 个 scale(fp16) + 32 个低位整数；画一整块（`QK4_0=32`）。
6. 正文：不给整个张量一个 scale（动态范围差异大、误差大），而是切成**小块**（Q4_0 每块 32 个权重），每块自带 scale => 块内范围小、近似更准。**代码1（源码简化，来自 `ggml/src/ggml-common.h` 的 `block_q4_0`）**：
   ```
   #define QK4_0 32
   typedef struct {
       ggml_half d;            // scale (fp16), one per block
       uint8_t   qs[QK4_0/2];  // 32 weights packed as 4-bit nibbles
   } block_q4_0;               // 2 + 16 = 18 bytes  ->  4.5 bit/weight
   ```
7. **图3【Q4_0 字节布局】**（`cells`/`timeline`）：`[d: 2 字节][qs: 16 字节]` = 18 字节装 32 个权重。**代码2（伪代码，解量化，对应 `ggml/src/ggml-quants.c` 的 `dequantize_row_q4_0`）**：
   ```
   # each 4-bit nibble q in 0..15 maps back to a signed weight:
   for i in range(32):
       q    = nibble(qs, i)     # 0..15
       x[i] = (q - 8) * d       # recenter at 0, scale by block's d
   ```
   注明真实实现在 `ggml-quants.c`。
8. `<h2>` Q8_0 / K-quant：精度与压缩的取舍 + **图4【量化家族对比】**（`cols` 或 `table class="t"`）：`Q8_0`(8bit/块32，d+32×int8，≈8.5bit，最接近 fp16) / `Q4_0`(4.5bit，轻) / `Q4_K`(K-quant：super-block=256，多个子 scale，更准)。
9. 正文：Q8_0 精度高但体积大；Q4_0 轻但糙；**K-quant**(Q4_K 等)用 super-block(`QK_K=256`) + 多个子 scale + 量化的 min，相当于"在更细粒度上分配 scale"，同 bit 下更准。**代码3（可选，`block_q4_K` 形态）**：super-block 联合体含 `d`/`dmin` + 子块 scales。
10. **深挖1**"Q4_0 / Q4_1 / Q8_0 后面的数字和 0/1 是什么意思"：数字=每权重 bit；`_0`=只存 scale(对称、零点固定)、`_1`=额外存一个 min/zero-point(非对称、更准但更大)。
11. **深挖2**"K-quant（Q4_K_M 等）凭什么更准"：super-block 256 内再分子块，各有更高位的子 scale + min，并对不同张量混合精度；故同体积下困惑度更低，名字里的 `_S`/`_M`/`_L` 是不同混合档。
12. **深挖3**"imatrix 是什么、和量化什么关系"：importance matrix = 用校准数据统计"每个权重对输出的影响"，量化时**按重要性加权误差**（让重要权重更准），**不是**改变 bit 分配；用 `llama-imatrix` 生成、喂给 `llama-quantize`。
13. `<div class="card key">`（✅ 关键要点）：量化 = 用更少 bit 近似权重，省显存/带宽、提速、损一点精度；**块量化**每块一个 scale（Q4_0 每块 32 权重 = 18 字节 = 4.5bit）；解量化 `x=(q-8)*d`；Q8_0 准而大、Q4_0 轻而糙、K-quant 用 super-block 更准；**imatrix 是误差加权、不是位宽分配**。
14. `<div class="card spark">`（💡 设计洞察）："每块一个 scale"这一个简单想法，把一个全局难题（浮点动态范围大）拆成无数个局部小问题（块内范围小），于是低到 4 bit 也能保住可用精度——这正是大模型能塞进消费级硬件的关键一招。

**必须讲到的论点**：为什么量化（显存/带宽）；块量化每块一个 scale；Q4_0 布局与字节数；解量化公式；Q8_0/K-quant 取舍；imatrix 澄清。
**要核实**（对照源树）：`QK4_0=32`/`QK8_0=32`/`QK_K=256`、`block_q4_0`/`block_q8_0`/`block_q4_K`（`ggml/src/ggml-common.h`）；`dequantize_row_q4_0`（`ggml/src/ggml-quants.c`）；`llama-quantize`/`llama-imatrix` 工具名。

- [ ] **Step 5: 在 quizzes.py 增加 06 的双语 quiz**（2-3 mcq + 1 open）
- **MCQ1**：q `{"zh":"块量化（每块一个 scale）为什么比“整个张量共用一个 scale”更准？","en":"Why is block quantization (one scale per block) more accurate than one scale for the whole tensor?"}`；opts：①`每块自带 scale，块内动态范围更小，近似误差更小`（✅）②`因为用了更多 bit` ③`因为压缩率更高` ④`因为完全不丢数据`；why：局部范围小 => 近似更准。
- **MCQ2**：q `{"zh":"Q4_0 每个权重平均约几 bit？（每块 32 权重 = 2 字节 scale + 16 字节量化值）","en":"About how many bits per weight does Q4_0 use? (per 32-weight block = 2-byte scale + 16 bytes of quants)"}`；opts：①`约 4.5 bit`（✅）②`正好 4 bit` ③`8 bit` ④`2 bit`；why：(2+16)×8 / 32 = 4.5。
- **OPEN**：`{"zh":"同样压到约 4bit，为什么 Q4_K 通常比 Q4_0 困惑度更低？","en":"At roughly 4 bits, why does Q4_K usually have lower perplexity than Q4_0?"}`（提示：super-block 与子 scale）。

- [ ] **Step 6: 重建 + 校验**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 6 课 · 2 个部分' index.html && echo "index shows 6 lessons / 2 parts"
grep -q 'href="lessons/06-quantization-intro.html"' index.html && echo "toc links 06"
python3 -c "import re,sys; sys.path.insert(0,'src'); import registry; z=registry.CONTENT['06-quantization-intro.html']['zh']; print('L06 CJK:', len(re.findall(r'[\u4e00-\u9fff]', z)))"
```
Expected：0 error / 0 warning、全链接解析、index 显示 6 课 · 2 个部分、06 链接在目录、`L06 CJK >= 4000`。

- [ ] **Step 7: Commit**
```bash
git add src/shell.py src/registry.py src/part2.py src/quizzes.py index.html lessons/06-quantization-intro.html
git commit -m "feat: add lesson 06 quantization intro (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 4: 课 07「构建系统与后端 / Build system & backends」

> 第二部分第 4 课（收尾课）。先登记，再在 `part2.py` **追加** `LESSON_07`，加 quiz。讲怎么编、怎么选后端、产物有哪些——读完能自己从源码 build 一个带 GPU 的 llama.cpp。

**Files:**
- Modify: `src/shell.py`（`PAGES`、`SUBTITLES` 各加 07）
- Modify: `src/part2.py`（追加 `LESSON_07`）
- Modify: `src/registry.py`（`CONTENT` 加 07）
- Modify: `src/quizzes.py`（`QUIZZES` 加 07）
- 产出：`index.html`（现 7 课 · 2 个部分）、`lessons/07-build-and-backends.html`

- [ ] **Step 1: 登记 PAGES**（06 之后追加）
```python
    ("07-build-and-backends.html", "构建系统与后端", "Build system & backends",
     "第二部分 · 前置基础", "Part 2 · Foundations"),
```

- [ ] **Step 2: 登记 SUBTITLES**
```python
    "07-build-and-backends.html": ("CMake 两步走 · 后端选项 · 产物在 build/bin · -ngl",
                                   "two-step CMake; backend options; build/bin outputs; -ngl"),
```

- [ ] **Step 3: 登记 registry**
```python
    "07-build-and-backends.html": part2.LESSON_07,
```

- [ ] **Step 4: 在 part2.py 执笔 LESSON_07（中/英双语）**。**结构（按序）**：

1. `<p class="lead">`：前面都在讲"是什么"，这一课讲"怎么把它编出来、怎么选硬件后端、编完有哪些产物"——看懂这套，你就能自己从源码 build 一个带 GPU 加速的 llama.cpp。
2. `<div class="card analogy">`（🔌 生活类比）：CMake 像**装修总包**——你勾选要不要地暖(CUDA)、中央空调(Metal/Vulkan)，它去找对应施工队(工具链)、生成施工图(构建文件)，最后交付能住的房子(可执行文件)。
3. `<h2>` 后端抽象：同一张计算图，多种硬件 + **图1【后端分发分层图】**（`layers`）：计算图(ggml) -> `ggml-backend` 接口 -> { CPU, CUDA, Metal, Vulkan, BLAS, ... } 各后端实现。点明 ggml 把"算什么"和"在哪算"解耦（呼应 L01）。
4. 正文：ggml 定义统一的 `ggml-backend` 接口（`ggml/include/ggml-backend.h`），每种硬件实现一份（`ggml/src/ggml-cuda` 等）；运行时按可用设备把图的算子派发下去。**编译时选哪些后端 = 决定这份二进制支持哪些硬件**。
5. `<h2>` 怎么编：CMake 两步走 + **图2【构建流程图】**（`flow`）：`cmake -B build [-DGGML_CUDA=ON]`（配置：探测工具链、生成构建文件）-> `cmake --build build -j`（编译）-> 产物落在 `build/bin`。**代码1（真实命令，来自 `docs/build.md`）**：
   ```
   # CPU-only (default)
   cmake -B build
   cmake --build build --config Release -j

   # with NVIDIA CUDA
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ```
6. `<h2>` 选后端：CMake 选项一览 + **图3【后端选项表】**（`table class="t"` 两列 选项/启用什么）：`GGML_CPU`(默认 ON，CPU+SIMD) · `GGML_BLAS`(BLAS 加速大矩阵乘) · `GGML_CUDA`(NVIDIA) · `GGML_HIP`(AMD/ROCm) · `GGML_METAL`(Apple，macOS 默认 ON) · `GGML_VULKAN`(跨厂商) · `GGML_SYCL`(Intel) · `GGML_OPENCL`(部分移动/嵌入式 GPU)。**代码2（源码简化，来自 `ggml/CMakeLists.txt`）**：
   ```
   option(GGML_CPU    "ggml: enable CPU backend" ON)
   option(GGML_CUDA   "ggml: use CUDA"           OFF)
   option(GGML_METAL  "ggml: use Metal"          ${GGML_METAL_DEFAULT})
   option(GGML_VULKAN "ggml: use Vulkan"         OFF)
   # ... HIP / SYCL / OPENCL / BLAS
   ```
7. `<h2>` 编完有什么：产物一览 + **图4【产物分组图】**（`cellgroup`/`cells`）：**库**(`libllama` · `libggml`) + **程序**(`llama-cli` · `llama-server` · `llama-quantize` · `llama-bench` · `llama-perplexity` · ...)，都在 `build/bin`。**代码3（可选，跑起来）**：`./build/bin/llama-cli -m model.gguf -p "Hi" -ngl 99`（`-ngl` = 卸载多少层到 GPU）。
8. **深挖1**"CPU 后端也要选吗？SIMD / -march=native / BLAS"：CPU 后端默认开、AVX/NEON 自动探测；可选 BLAS 提升 prefill 大矩阵乘；追极致可 `-march=native`（牺牲可移植性）。
9. **深挖2**"为什么用 CMake 而不是手写 Makefile"：跨平台(Linux/macOS/Windows)、自动探测各家 GPU 工具链与依赖、可生成 Ninja/Make/VS 工程；llama.cpp 早期有 Makefile，现以 CMake 为主。
10. **深挖3**"运行时怎么决定用哪个后端 / 多 GPU 怎么办"：`ggml-backend` 注册表枚举可用设备；`-ngl`/`n_gpu_layers` 决定多少层放 GPU、其余留 CPU；多 GPU 可按层或张量切分。
11. `<div class="card key">`（✅ 关键要点）：`ggml-backend` 把"算什么/在哪算"解耦；编译两步 `cmake -B build [-D选项]` 再 `cmake --build`；GPU 靠 `-DGGML_CUDA/METAL/VULKAN/...` 开；产物在 `build/bin`（`libllama`/`libggml` + `llama-cli`/`server`/`quantize`...）；`-ngl` 控制 GPU 层卸载。
12. `<div class="card spark">`（💡 设计洞察）：把"后端"做成**可插拔的编译期开关**——同一份源码，按需编出 CPU-only 的极简二进制，或带 CUDA/Metal 的加速版；既守住"零依赖到处跑"，又能在有 GPU 时榨干性能。

**必须讲到的论点**：后端抽象解耦；CMake 两步编译；后端选项与对应硬件；产物在 `build/bin`；`-ngl` 层卸载。
**要核实**（对照源树）：`ggml/CMakeLists.txt` 的 `GGML_*` 选项名与默认值；`docs/build.md` 命令；`tools/` 程序名；`ggml/include/ggml-backend.h`；`-ngl`/`n_gpu_layers`。

- [ ] **Step 5: 在 quizzes.py 增加 07 的双语 quiz**（2-3 mcq + 1 open）
- **MCQ1**：q `{"zh":"想编一个支持 NVIDIA GPU 的 llama.cpp，应该怎么做？","en":"How do you build a llama.cpp with NVIDIA GPU support?"}`；opts：①`cmake -B build -DGGML_CUDA=ON`（✅）②`pip install cuda` ③`运行时加 --gpu` ④`手动改源码里的 if`；why：后端是编译期 CMake 开关。
- **MCQ2**：q `{"zh":"ggml 的“后端”(CPU/CUDA/Metal/...)主要解决什么问题？","en":"What does ggml's “backend” layer (CPU/CUDA/Metal/...) mainly solve?"}`；opts：①`同一张计算图能派发到不同硬件执行，把“算什么”和“在哪算”解耦`（✅）②`决定模型精度` ③`负责量化权重` ④`解析 GGUF 文件`；why：后端=执行层抽象。
- **OPEN**：`{"zh":"为什么“选哪些后端”是编译期开关，而不是运行时全都带上？","en":"Why are backends a compile-time switch rather than all bundled at runtime?"}`（提示：依赖、体积、零依赖哲学）。

- [ ] **Step 6: 重建 + 校验**
```bash
cd /home/verden/course/llama-cpp-visual-guide/src && python build.py && python check_html.py && python check_links.py
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 7 课 · 2 个部分' index.html && echo "index shows 7 lessons / 2 parts"
grep -q 'href="lessons/07-build-and-backends.html"' index.html && echo "toc links 07"
python3 -c "import re,sys; sys.path.insert(0,'src'); import registry; z=registry.CONTENT['07-build-and-backends.html']['zh']; print('L07 CJK:', len(re.findall(r'[\u4e00-\u9fff]', z)))"
```
Expected：0 error / 0 warning、全链接解析、index 显示 7 课 · 2 个部分、07 链接在目录、`L07 CJK >= 4000`。

- [ ] **Step 7: Commit**
```bash
git add src/shell.py src/registry.py src/part2.py src/quizzes.py index.html lessons/07-build-and-backends.html
git commit -m "feat: add lesson 07 build system and backends (bilingual) with quiz

Assisted-by: GitHub Copilot"
```

---

## Task 5: M2 验收（清重建 + 密度/CJK 审计 + 里程碑）

> 无新增内容；端到端验证第二部分 4 课全部达标、与第一部分衔接无回归。

**Files:** 无新增（仅可能补一处副标题/交叉引用的一致性微调）。

- [ ] **Step 1: 清重建 + 双校验（产物零漂移）**
```bash
cd /home/verden/course/llama-cpp-visual-guide
rm -f index.html lessons/*.html
cd src && python build.py && python check_html.py && python check_links.py
cd .. && git status --short        # 期望：重建后与已提交产物一致，无意外 diff
```
Expected：`structural check passed`（0 error / 0 warning）、`all N internal links resolve`、`git status` 干净（产物可复现）。

- [ ] **Step 2: 4 课密度 / CJK / 图 / 片段 / 深挖 审计**
```bash
cd /home/verden/course/llama-cpp-visual-guide && python3 - <<'PY'
import re, sys
sys.path.insert(0, "src")
import registry
DIAG = ("layers","vflow","flow","cols","cellgroup","timeline")
for f in ["04-llm-inference-basics.html","05-tensors.html",
          "06-quantization-intro.html","07-build-and-backends.html"]:
    z = registry.CONTENT[f]["zh"]; e = registry.CONTENT[f]["en"]
    cjk = len(re.findall(r"[\u4e00-\u9fff]", z))
    diag = max(sum(z.count(f'class="{c}') for c in DIAG) + z.count('class="t"'),
               sum(e.count(f'class="{c}') for c in DIAG) + e.count('class="t"'))
    acc = z.count('class="accordion"')
    pre = z.count("<pre")
    ok = cjk >= 4000 and diag >= 3 and acc >= 2 and pre >= 2
    print(f"{f:34s} CJK={cjk:5d} diagrams>={diag} accordions={acc} <pre>={pre}  {'OK' if ok else 'CHECK'}")
PY
```
Expected：每课 `CJK >= 4000`、`diagrams >= 3`（3-5）、`accordions >= 2`、`<pre> >= 2`。任一 `CHECK` 都要回到对应 Task 补足。

- [ ] **Step 3: 导航与交叉引用检查**
```bash
cd /home/verden/course/llama-cpp-visual-guide
grep -q '共 7 课 · 2 个部分' index.html && echo "index: 7 课 · 2 部分"
for n in 04 05 06 07; do grep -ql "href=\"lessons/$n-" index.html && echo "toc links $n"; done
grep -RoE '第 [0-9]+ 课' lessons/ | awk -F'第 | 课' '{if($2+0>40) print "OUT OF RANGE:", $0}'   # 期望无输出
```
Expected：index 显示 `7 课 · 2 个部分`、04-07 均在目录、无 `第 N 课`（N>40）越界引用、首页 prev/next 链路完整。

- [ ] **Step 4: 标记里程碑**

确认 1-3 全绿后，更新 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 状态行 `- [ ] M2 ...` 为 `- [x] M2 ...`；若有改动则：
```bash
git add docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md
git commit -m "docs: mark M2 (Part 2 foundations) complete

Assisted-by: GitHub Copilot"
```
随后按 finishing-a-development-branch 流程把 `build/m2-part2` 合并回 master（`--no-ff`）并删分支。

---

## Self-Review（计划自检，作者本人执行）

**1. Spec 覆盖**：第二部分 4 课（04 大模型推理基础 / 05 张量 / 06 量化入门 / 07 构建与后端）= Spec §"第二部分"四条，逐一对应 Task 1-4；加强标准（§5/§6：3-5 图含概念图、2-3 片段、2-3 深挖、~4000+ CJK）写进"统一交付标准"并由 Task 5 Step 2 脚本量化校验。✅ 无遗漏。

**2. Placeholder 扫描**：每个 Task 的 Step 4 给了**逐卡片/逐图/逐片段/逐深挖**的具体简报与真实源码片段（含 `block_q4_0`/`ggml_tensor`/`option(GGML_*)` 实体），Step 5 给了完整 quiz 文案，Step 6 给了精确命令与期望输出。无 "TBD/TODO/类似上文" 占位。✅

**3. 一致性**：四课文件名（`04-llm-inference-basics`/`05-tensors`/`06-quantization-intro`/`07-build-and-backends`）在 PAGES/SUBTITLES/registry/quizzes/校验命令中保持一致；part 标签统一 `第二部分 · 前置基础` / `Part 2 · Foundations`；`part2.py` 由 Task 1 创建、Task 2-4 追加；registry 的 `import part2` 仅 Task 1 引入。✅

**4. 防重复**：L04 与 L03（流水线）划清边界（模型/数学底座）、L05 深于 L02（字段语义）、L06 深于 L01（块量化机制），已在"避免与第一部分重复"明确。✅

---

## 执行方式

按用户既定：**subagent 驱动**（每课一个全新实现子代理 + spec 合规 + 代码质量两段审查），**全程使用当前主会话模型**。分支 `build/m2-part2`，4 课顺序产出（共享 `part2.py`/`registry.py`/`shell.py`/`quizzes.py`，串行避免冲突），Task 5 验收后合并回 master。





