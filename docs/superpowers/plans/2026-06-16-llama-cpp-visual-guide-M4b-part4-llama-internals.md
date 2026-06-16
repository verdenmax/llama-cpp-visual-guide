# M4b · 第四部分（下）llama 推理内部 — 文本 IO 与可控生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 产出本可视化指南第四部分（下）共 5 课（20-24），讲清"模型如何把文本切成 token（词表）、如何从 logits 采样出下一个词（采样器）、如何把多轮对话拼成一段提示词（对话模板）、如何用 GBNF 语法约束输出结构、以及如何用 LoRA 适配器轻量改变模型行为"——即 llama 推理在运行时核心（M4a）之上的"文本输入输出 + 可控生成"层。

**Architecture:** 沿用现有零依赖 Python 静态站点生成器。在既有 `src/part4.py` 追加 `LESSON_20..24` 双语内容字典；每课在 `src/shell.py`（PAGES/SUBTITLES）、`src/registry.py`（CONTENT）、`src/quizzes.py`（QUIZZES）登记；`python3 src/build.py` 生成 HTML，`check_html.py`/`check_links.py` 校验。index 课数 19 -> 24，部分数仍为 4（同属第四部分）。

**Tech Stack:** Python 静态站点生成器（build.py/shell.py/registry.py/quizzes.py/check_*.py）；内容在 part4.py；复用 CSS 图组件（layers/vflow/flow/cols/cellgroup/timeline/`table.t`）。

---

## 里程碑范围（M4b = 课 20-24）

| 课 | 标题（zh / en） | 主源文件 |
| --- | --- | --- |
| 20 | 词表 / Vocabulary | `src/llama-vocab.{h,cpp}` · `include/llama.h` |
| 21 | 采样 / Sampling | `src/llama-sampler.{h,cpp}` · `include/llama.h` |
| 22 | 对话模板 / Chat templates | `src/llama-chat.{h,cpp}` · `common/chat.{h,cpp}` · `common/jinja/` |
| 23 | 语法约束 / Grammar (GBNF) | `src/llama-grammar.{h,cpp}` · `grammars/README.md` |
| 24 | LoRA 适配器 / LoRA adapters | `src/llama-adapter.{h,cpp}` · `include/llama.h` |

衔接：本部分承接 M4a（14-19 加载与运行时核心：loader/arch/graph/context/batch/KV）。M4a 讲"模型怎么算出 logits"，M4b 讲"文本怎么进、token 怎么出、生成怎么被控制"，正好补上推理回路的两端（输入侧的 tokenize + 模板，输出侧的采样 + 语法）外加可控性（LoRA）。之后是 M5（第五部分 · 公共 API 与工具，25-30）。

## 统一交付标准（每课硬性达标，与 M1-M4a 一致）

- **zh 正文纯中文 CJK >= 4000**（`[\u4e00-\u9fff]` 计数）；**en 正文 0 个 CJK**，且与 zh **逐段对齐**（按 `<h2>` 切分，每段 `<p>`+`<p ` 数 zh==en）。
- **3-5 张图**，取自既有复用组件（layers/vflow/flow/cols/cellgroup/timeline/`table.t`），含 **>=1 张概念/结构图**；两种语言图数相同。
- **2-3 个 `<pre class="code">` 代码片段**：至少一个"简化自真实源码"的片段，**引用 文件+符号、不带行号**；其余可为伪代码（伪代码注释用 `#`，C++ 片段用 `//`）。
- **2-3 个 `<details class="accordion">` 深挖**（"为什么这么设计 / 还有什么替代 / 和前后怎么连"）。
- 卡片齐全：`lead`、`analogy`（🔌）、`key`（✅ 关键要点）、`spark`（💡 设计洞察）。
- quiz：3 个 MCQ（`answer` 为 0 基正确项下标，构建器会确定性洗牌）+ 1 个 open，全双语 `{"zh","en"}`；**quiz 文本是裸 HTML，`<`/`&` 必须写成 `&lt;`/`&amp;`**（quizzes.py 约定），正文里出现的 `<xxx>`/`<=` 同理需转义，构建后 grep 渲染 HTML 确认无裸标签泄漏。
- ASCII 守则：代码块内不用 `->`(U+2192 `→`)/`×`/`÷`/`…`/`≤`，用 `-&gt;`/`x`/`/`/`...`/`&lt;=`；散文沿用既有风格（`·`、`×`、`——`、`……` 可用于 zh，但 en 必须纯 ASCII，**全局不用 `→`**）。

## 执行方式

按用户既定：**subagent 驱动**（每课跑 spec 合规 + 代码质量两段审查子代理，全程 `claude-opus-4.8`）。鉴于内容子代理写大段 HTML 易卡死，**控制者直接执笔每课内容**，再对每课跑 spec + 质量双审子代理（只读）。分支 `build/m4b-part4`，5 课顺序产出，验收后 `--no-ff` 合并回 master 并删分支。

---

## 验证过的源码事实（2026-06-16 对真实源树 `/home/verden/course/llama.cpp` 核验；引用"文件+符号"，无行号）

> 源码已演进，以下为**当前真相**。Part 4 这几课的 C API 出现大量 2023 旧名被弃用/移除，写每课时严格据此，勿臆造已不存在的名字。

**L20 词表** — `src/llama-vocab.{h,cpp}` / `include/llama.h`
- `struct llama_vocab` 是 **pimpl**（私有 `struct impl; std::unique_ptr<impl> pimpl;`）；主方法：`tokenize`/`detokenize`/`token_to_piece`/`text_to_token`、`byte_to_token(uint8_t)`/`token_to_byte`、`n_tokens()`/`get_type()`、`is_normal/is_control/is_byte/is_eog`、`token_get_text/score/attr`、`get_add_bos()`/`get_add_eos()`。
- `enum llama_vocab_type`（`include/llama.h`）：`LLAMA_VOCAB_TYPE_NONE=0`/`_SPM=1`(SentencePiece, LLaMA 字节级 BPE + 字节回退)/`_BPE=2`(GPT-2 字节级 BPE)/`_WPM=3`(BERT WordPiece)/`_UGM=4`(T5 Unigram)/`_RWKV=5`(贪心)/**`_PLAMO2=6`**(Aho-Corasick+DP)。另有 `enum llama_vocab_pre_type`（~56 个 pre-tokenizer 变体）。
- Token 属性：当前用 **`enum llama_token_attr`**（位掩码：`UNKNOWN/UNUSED/NORMAL/CONTROL/USER_DEFINED/BYTE/NORMALIZED/LSTRIP/RSTRIP/SINGLE_WORD`）；旧 `enum llama_token_type` 标注 `//TODO: remove`，**用 attr**。
- 特殊 token 存 `impl` 的 `special_*_id`，访问器 `token_bos/eos/eot/eom/unk/sep/nl/pad/mask` + FIM `token_fim_pre/suf/mid/...`；C API `llama_vocab_bos/eos/eot/sep/nl/pad/mask`。⚠ `llama_vocab_cls` 已弃用 -> 用 `llama_vocab_bos`（CLS 并入 BOS）；UNK/EOM 无公开 C getter。
- 字节回退：`byte_to_token(uint8_t)`——SPM/UGM/PLAMO2 映到 `<0xXX>` 十六进制 token；BPE/WPM 用 `unicode_byte_to_utf8`；字节 token 带 `LLAMA_TOKEN_ATTR_BYTE`。
- C API：取词表 `llama_model_get_vocab(model)`；大小 **`llama_vocab_n_tokens(vocab)`**（旧 **`llama_n_vocab` 已 DEPRECATED**）；`llama_tokenize`/`llama_token_to_piece`/`llama_detokenize`（都收 `const llama_vocab *`）。⚠ 2023 旧名 `llama_token_bos/eos/...`、`llama_token_get_text/...`、`llama_add_bos_token` 等**全部 DEPRECATED -> `llama_vocab_*`**。

**L21 采样** — `src/llama-sampler.{h,cpp}`（⚠ **不是** `llama-sampling`）/ `include/llama.h` / `common/sampling.cpp`
- `struct llama_sampler_i`（`include/llama.h`）函数指针：`name`(可空)/`accept(smpl,token)`(可空)/`apply(smpl,cur_p)`(**必需**)/`reset`/`clone`/`free`（均可空）；另有 `[EXPERIMENTAL]` 后端 GPU 钩子 `backend_init/accept/apply/set_input`。`struct llama_sampler{ const llama_sampler_i * iface; llama_sampler_context_t ctx; }`。
- `struct llama_token_data{ llama_token id; float logit; float p; }`；`struct llama_token_data_array{ llama_token_data * data; size_t size; int64_t selected; bool sorted; }`（`selected` 是 `data` 的下标，不是 token id）。
- 链：`llama_sampler_chain_init`/`_add`(接管所有权)/`_get`/`_n`/`_remove`(放弃所有权)；`struct llama_sampler_chain_params{ bool no_perf; }` + `llama_sampler_chain_default_params()`。
- 构造器 `llama_sampler_init_*`：`greedy`(argmax)/`dist(seed)`(按概率随机)/`top_k`/`top_p`/`min_p`/`typical`/`top_n_sigma`/`temp`/`temp_ext`/`xtc`/`dry`/`mirostat`/`mirostat_v2`/`penalties`/`logit_bias`/`adaptive_p`/`grammar`/`grammar_lazy_patterns`/`infill`。⚠ 旧全局 `llama_sample_top_k/top_p/temperature/token(...)` **已全部移除**，改为采样器对象 + 链。
- 入口：`llama_sampler_sample(smpl,ctx,idx)`=读 idx 的 logits -> 建 `cur_p` -> `apply` -> 取 `cur_p.data[selected].id` -> `accept`；`llama_sampler_apply`(改/排候选)、`llama_sampler_accept`(把选中 token 喂回有状态采样器)、`llama_sampler_reset`。greedy=最大 logit 的下标；dist=softmax 后按 RNG 随机。
- 默认链顺序（`common/common.h` `common_params_sampling::samplers` + `common/sampling.cpp`）：`PENALTIES -> DRY -> TOP_N_SIGMA -> TOP_K -> TYPICAL_P -> TOP_P -> MIN_P -> XTC -> TEMPERATURE`，末尾追加 `dist`。⚠ grammar 采样器（`grmr`）**不在链内**，作为 `common_sampler` 的独立对象，按 `grammar_first` 在链前或链后施加。

---

**L22 对话模板** — `src/llama-chat.{h,cpp}` / `include/llama.h` / `common/chat.{h,cpp}` / `common/jinja/`
- 内建模板枚举 `enum llm_chat_template`（`src/llama-chat.h`，约 56 项含末尾 `LLM_CHAT_TEMPLATE_UNKNOWN` 哨兵）：`_CHATML`/`_LLAMA_2`(+`_SYS`变体)/`_LLAMA_3`/`_MISTRAL_V1/V3/V7`/`_PHI_3`/`_PHI_4`/`_GEMMA`/`_DEEPSEEK`/`_COMMAND_R`/`_GRANITE_*` 等。
- 检测+应用（`src/llama-chat.cpp`）：`llm_chat_detect_template(const std::string & tmpl)`->枚举（先按名 `llm_chat_template_from_str`，再按模板体子串启发式如含 `<|im_start|>`/`[INST]`）；`llm_chat_apply_template(llm_chat_template, const std::vector<const llama_chat_message *> & chat, std::string & dest, bool add_ass)`。
- `struct llama_chat_message{ const char * role; const char * content; }`（`include/llama.h`）。
- C API：`llama_chat_apply_template(const char * tmpl, const llama_chat_message * chat, size_t n_msg, bool add_ass, char * buf, int32_t length)`。⚠ **只收模板字符串、无 `llama_model *` 参数**（旧 2023 带 model 的重载已移除）；注释明确"不用 jinja 解析器，只支持预定义模板列表"。`llama_chat_builtin_templates(const char ** out, size_t len)` 列内建名。
- 硬编码 C++ vs Jinja：`src/llama-chat.cpp` 只处理上面这套固定枚举（纯字符串拼接、无 Jinja）；Jinja 引擎是 vendored 的 **`common/jinja/`**（命名空间 `jinja::`，⚠ 目录是 `common/jinja/` 不是 `minja`）。`common/chat.cpp` 是上层：`common_chat_templates`（`_init`/`_apply`/`_source`）在 `use_jinja=true`/模型自带模板时走真 Jinja，并加工具调用语法约束 + 输出解析；其 `enum common_chat_format` 近期重构为 PEG 模型（`CONTENT_ONLY`/`PEG_SIMPLE`/`PEG_NATIVE`/`PEG_GEMMA4`）。

**L23 语法约束** — `src/llama-grammar.{h,cpp}` / `include/llama.h` / `grammars/README.md`
- 元素类型 `enum llama_gretype`（`src/llama-grammar.h`）：`_END=0`/`_ALT=1`(`|`)/`_RULE_REF=2`(非终结符引用)/`_CHAR=3`(字面码点)/`_CHAR_NOT=4`(`[^...]`)/`_CHAR_RNG_UPPER=5`(范围上界 `[a-z]`)/`_CHAR_ALT=6`(类内追加字符)/`_CHAR_ANY=7`(`.`)/`_TOKEN=8`(指定 token `<[id]>`)/`_TOKEN_NOT=9`。⚠ `_CHAR_ANY/_TOKEN/_TOKEN_NOT` 是新增。
- `struct llama_grammar`（`src/llama-grammar.h`）：`const llama_vocab * vocab; const llama_grammar_rules rules; llama_grammar_stacks stacks; llama_partial_utf8 partial_utf8;` + 惰性/触发字段 `bool lazy; bool awaiting_trigger; std::string trigger_buffer; std::vector<llama_token> trigger_tokens; ... trigger_patterns`。
- 关键函数（`src/llama-grammar.cpp`）：`llama_grammar_init_impl`（两重载：原始规则数组 / `grammar_str`+`grammar_root`+惰性触发参数）；`llama_grammar_accept_impl(llama_grammar &, llama_token)`（按接受的 token 推进栈）；`llama_grammar_apply_impl(const llama_grammar &, llama_token_data_array * cur_p)`——**掩码**：把不允许的 EOG、空片、以及 `llama_grammar_reject_candidates` 拒绝集里的候选 `logit = -INFINITY`；`awaiting_trigger` 时提前返回。
- 语法采样器（`include/llama.h`）：`llama_sampler_init_grammar(const llama_vocab *, const char * grammar_str, const char * grammar_root)`；惰性 `llama_sampler_init_grammar_lazy_patterns(...)`。⚠ `llama_sampler_init_grammar_lazy` 已 **DEPRECATED**。其 `apply`->`llama_grammar_apply_impl`(掩码)，`accept`->`llama_grammar_accept_impl`(推进)。
- GBNF（`grammars/README.md`）：规则 `nonterminal ::= 序列`；`|` 备选；`[...]`/范围 `[a-z]`/取反 `[^\n]`；重复 `*`(`{0,}`)/`+`(`{1,}`)/`?`(`{0,1}`) 及 `{m}`/`{m,}`/`{m,n}`；`()` 分组；入口是 `root` 规则。仓库例：`grammars/json.gbnf`（`root ::= object`）、`grammars/chess.gbnf`。

**L24 LoRA 适配器** — `src/llama-adapter.{h,cpp}` / `src/llama-graph.cpp` / `include/llama.h`
- `struct llama_adapter_lora`（`src/llama-adapter.h`）：`std::unordered_map<std::string, llama_adapter_lora_weight> ab_map`(目标张量名 -> 低秩对) + `float alpha` + `std::vector<llama_token> alora_invocation_tokens`(aLoRA)。`struct llama_adapter_lora_weight{ ggml_tensor * a; ggml_tensor * b; }`，`get_scale(alpha, adapter_scale) = alpha ? adapter_scale*alpha/rank : adapter_scale`（`rank = b->ne[0]`）。
- LoRA 数学（`llm_graph_context::build_lora_mm`，`src/llama-graph.cpp`）：`res = W·cur`，再对每个生效适配器 `ab_cur = scale·(b·(a·cur))`，`res = res + ab_cur`——即有效增量 = **`scale · B · A`**（低秩）加到基础权重输出上。
- C API（⚠ 大量改名）：`llama_adapter_lora_init(llama_model *, const char * path_lora)`（旧 `llama_lora_adapter_init`）；`llama_adapter_lora_free`。⚠ **`llama_set_adapters_lora(ctx, llama_adapter_lora ** adapters, size_t n_adapters, float * scales)`**（复数、批量）——**单数 `llama_set_adapter_lora`/`llama_rm_adapter_lora`/`llama_clear_adapter_lora` 在当前代码里不存在**（清空 = `n_adapters==0`）。
- 控制向量：`struct llama_adapter_cvec`（`apply_to(ctx, cur, il)` 把每层方向向量加进残差流）；C API ⚠ **`llama_set_adapter_cvec(ctx, data, len, n_embd, il_start, il_end)`**（旧 `llama_apply_adapter_cvec`/`llama_control_vector_apply` 均已移除）。cvec 沿固定方向**平移激活**，LoRA 注入**低秩权重增量**。
- 生效位置：`llama_set_adapters_lora` -> `llama_context::set_adapters_lora` 把 `{适配器->scale}` 存到 context；之后 `decode` 建图时 `build_lora_mm` 读它、把增量折进每个相关 matmul（**不复制权重**）。

---

## Task 1: 课 20「词表 / Vocabulary」

> 第四部分第 7 课、M4b 第 1 课。M4a 讲清了模型怎么"算"（loader->arch->graph->context->batch->KV，输出 logits），但模型只认数字 token、不认文本。这一课讲 `llama_vocab`：怎么把文本切成 token（tokenize）、又怎么把 token 还原成文本（detokenize），背后有哪几种分词算法（SPM/BPE/WPM/UGM/RWKV/PLAMO2）、特殊 token 与字节回退。接 L15（`n_vocab` 来自词表）、预告 L21（采样在 token 空间里选下一个）。

**Files:** `src/part4.py`（追加 `LESSON_20`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `20-vocabulary.html`。

- [ ] **Step 1-3: 登记**
```python
# shell.py PAGES 追加：
("20-vocabulary.html", "词表", "Vocabulary",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
# shell.py SUBTITLES 追加：
"20-vocabulary.html": ("llama_vocab · tokenize/detokenize · SPM/BPE/WPM · 字节回退",
                       "llama_vocab; tokenize/detokenize; SPM/BPE/WPM; byte fallback"),
# registry.py CONTENT 追加：
"20-vocabulary.html": part4.LESSON_20,
```

- [ ] **Step 4: 执笔 LESSON_20（双语）**。**结构**：
1. `lead`：M4a 的模型已经能"算"，但它只认数字 token、不认文本。这一课讲 `llama_vocab`——把文本和 token **双向翻译**的那层：`tokenize` 把字符串切成 token id 序列喂给模型，`detokenize`/`token_to_piece` 把模型吐出的 token id 还原成文字。
2. `analogy`（🔌）：词表像一本**双向密码本**：编码时把人话按规则切成一串号码（token id），解码时按号码查回字符片段。不同模型用不同"切法"（SPM/BPE/WPM），但都是同一本"号码 <-> 文本片段"的对照表。
3. `<h2>` 为什么需要词表 + **图1【文本 <-> token 双向流】**（`flow`，本课概念图）：`"你好"` -> `tokenize` -> `[token id...]` -> model(L16) -> `[token id]` -> `token_to_piece` -> `"好"`。正文：模型内部全是数字；词表是文本世界和 token 世界之间唯一的翻译官，进出两头都要过它。
4. `<h2>` llama_vocab 是什么 + **代码1（简化自 `src/llama-vocab.h`）**：
```
struct llama_vocab {
    uint32_t n_tokens() const;                    // 词表大小(L15 的 n_vocab 来源)
    llama_token_attr token_get_attr(llama_token) const;
    int32_t tokenize(const char * text, ...) const;   // 文本 -> token id
    int32_t token_to_piece(llama_token, char * buf, ...) const;  // token id -> 文本片段
private:
    struct impl;                                  // pimpl: 藏起分词器细节
    std::unique_ptr<impl> pimpl;
};
```
   讲：`llama_vocab` 用 **pimpl** 把"具体哪种分词器"的实现细节藏在 `impl` 里，对外只露统一接口——所以换一种分词算法，用的人无感。
5. `<h2>` 几种分词算法 + **图2【vocab 类型】**（`<table class="t">`）：列 `LLAMA_VOCAB_TYPE_SPM`(SentencePiece, LLaMA)/`_BPE`(GPT-2 字节级)/`_WPM`(BERT WordPiece)/`_UGM`(T5 Unigram)/`_RWKV`(贪心)/`_PLAMO2`(Aho-Corasick)，列"含义/代表模型"。正文：同一句话，不同算法切出的 token 数和边界都不同；GGUF 的 `general.architecture`/tokenizer 元数据（L13）决定用哪种。
6. `<h2>` 特殊 token 与字节回退 + **图3【字节回退】**（`cellgroup`）：生僻字 `𓀀` -> 词表里没有 -> 拆成 UTF-8 字节 -> 每字节映到 `<0xF0>`/`<0x93>`... 这种字节 token（带 `LLAMA_TOKEN_ATTR_BYTE`）。正文：BOS/EOS/EOT/PAD 等特殊 token 由访问器 `token_bos()`/`token_eos()`/`token_eot()` 取；字节回退保证**任何** UTF-8 文本都能被编码，永不 OOV。**代码2（伪代码，tokenize 往返）**：
```
ids = vocab.tokenize("Hello", add_special=True)   # 可前置 BOS
# ids = [<bos>, 9906, ...]
text = ""
for id in ids:
    text += vocab.token_to_piece(id)              # 逐 token 还原拼接
```
7. `<h2>` C API 与衔接 + 正文：取词表 `llama_model_get_vocab(model)`；大小 `llama_vocab_n_tokens`（⚠ 旧 `llama_n_vocab` 已弃用）；`llama_tokenize`/`llama_token_to_piece`/`llama_detokenize`。⚠ 一大批 2023 旧名 `llama_token_bos/eos/...` 已弃用、改 `llama_vocab_*`。接 L21：tokenize 出的 id 进模型、模型出的 logits 由采样器在**这张词表的 token 空间**里选下一个 id。
8. **深挖1**"为什么有这么多分词算法（SPM/BPE/WPM…）？"：历史与取舍——不同模型家族沿用不同生态（LLaMA 系 SPM、GPT 系 BPE、BERG 系 WPM）；对多语言/代码/效率各有权衡。llama.cpp 用一套接口（pimpl）兼容它们。
9. **深挖2**"字节回退到底解决什么？"：避免 OOV（未登录词）——任何字符最差也能拆成 UTF-8 字节逐个编码，于是没有任何输入是"词表外"的；代价是生僻字会占多个 token。
10. **深挖3**"为什么 `n_vocab` 在词表里、不在 hparams（L15）？"：因为词表大小是**词表的属性**，由 tokenizer 决定；hparams 描述网络形状，词表描述 token 空间——职责分开。`llama_vocab::n_tokens()` 才是权威来源。
11. `key`（✅）：`llama_vocab`(pimpl) = `tokenize`(文本->id) + `token_to_piece`/`detokenize`(id->文本)；类型 `LLAMA_VOCAB_TYPE_SPM/BPE/WPM/UGM/RWKV/PLAMO2`；特殊 token `token_bos/eos/eot`；字节回退 `<0xXX>` 防 OOV；C API `llama_model_get_vocab`/`llama_vocab_n_tokens`/`llama_tokenize`（旧 `llama_n_vocab`/`llama_token_bos` 已弃用）。
12. `spark`（💡）：词表把"**文本世界**"和"**token 世界**"干净地隔开——模型只在 token 空间里活，永远不碰字符串；分词器用 pimpl 一藏，几种截然不同的切分算法在上层看起来就是同一个 `tokenize`。这层翻译官，是"模型只会算数字"和"人类只说文字"之间的唯一桥。

必须讲到：tokenize/detokenize 双向；pimpl；`LLAMA_VOCAB_TYPE_*` 几种；特殊 token 访问器；字节回退 `<0xXX>`/`ATTR_BYTE`；C API 新旧名（`llama_vocab_n_tokens` vs 弃用 `llama_n_vocab`）。

- [ ] **Step 5: quiz（20）**：
- MCQ1 "`tokenize` 做什么？" -> 正确："把文本字符串切成一串 token id（喂给模型）"；干扰：把 token 还原成文本 / 训练词表 / 给 token 打分。
- MCQ2 "字节回退（byte fallback）的作用？" -> 正确："让任何 UTF-8 字符都能被编码（拆成字节 token），避免未登录词 OOV"；干扰：压缩词表 / 加速推理 / 删除特殊 token。
- MCQ3 "取词表大小，当前应该用哪个 API？" -> 正确："`llama_vocab_n_tokens`（旧 `llama_n_vocab` 已弃用）"；干扰：`llama_n_ctx` / `strlen` / `llama_n_embd`。
- OPEN "结合 L21，描述一次对话生成里 `tokenize` 和 `token_to_piece` 各在什么时候被调用、各处理哪个方向。"

- [ ] **Step 6: 重建+校验**（`python3 src/build.py && python3 src/check_html.py && python3 src/check_links.py`；index 变 "共 20 课 · 4 个部分"；CJK>=4000、en CJK=0、逐段对齐、>=3 图、>=2 深挖、>=2 片段；grep 渲染 HTML 确认 `<0x`、`<bos>` 等已转义无裸标签泄漏）。
- [ ] **Step 7: commit**（`feat: add lesson 20 vocabulary (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 2: 课 21「采样 / Sampling」

> 第四部分第 8 课、M4b 第 2 课。L17 的 `llama_decode` 算出一排 logits（每个 token 一个分数），但怎么从几万个分数里挑出"下一个词"？这一课讲采样器：贪心 vs 随机、top-k/top-p/温度等"裁剪 + 塑形"候选分布的手段，以及 llama.cpp 把它们串成**采样链**的对象化设计。接 L17（logits 来源）、L20（在词表 token 空间里选）、预告 L23（grammar 是一种掩码采样器）。

**Files:** `src/part4.py`（追加 `LESSON_21`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `21-sampling.html`。

- [ ] **Step 1-3: 登记**
```python
("21-sampling.html", "采样", "Sampling",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"21-sampling.html": ("llama_sampler · 采样链 · top-k/top-p/温度 · greedy vs dist",
                     "llama_sampler; sampler chain; top-k/top-p/temp; greedy vs dist"),
"21-sampling.html": part4.LESSON_21,
```

- [ ] **Step 4: 执笔 LESSON_21（双语）**。**结构**：
1. `lead`：模型每步吐出一排 logits——词表里**每个** token 一个原始分数。可"下一个词"只能有一个，怎么从几万个分数里挑？这一课讲采样：先裁剪（top-k/top-p）、再塑形（温度/惩罚），最后选一个（greedy 取最大，或 dist 按概率随机）。
2. `analogy`（🔌）：采样像**摇号抽奖**：logits 是每个号码的原始权重，温度调"凭实力还是凭运气"，top-k/top-p 先划掉没希望的号，惩罚项压低最近出现过的号，最后 `dist` 按权重摇一个出来（或 `greedy` 直接选权重最大的那个）。
3. `<h2>` 从 logits 到 token + **图1【采样管线】**（`flow`，本课概念图）：`logits[n_vocab]` -> `penalties` -> `top_k` -> `top_p` -> `temp` -> `dist` -> `token id`。正文：每一步都在改写一个候选数组（`llama_token_data_array`），层层裁剪塑形，最后一步选中。
4. `<h2>` 采样器接口 + **代码1（简化自 `src/llama-sampler.h` / `include/llama.h`）**：
```
struct llama_sampler_i {
    const char * (*name)  (const llama_sampler *);
    void (*accept)(llama_sampler *, llama_token);        // 把选中 token 喂回(有状态采样器)
    void (*apply) (llama_sampler *, llama_token_data_array * cur_p);  // 改/排候选(必需)
    void (*reset)(llama_sampler *); /* clone; free */
};
struct llama_sampler { const llama_sampler_i * iface; llama_sampler_context_t ctx; };
```
   讲：每个采样器就是一组函数指针 + 一块状态 `ctx`；`apply` 是核心（改写候选数组），`accept` 让有状态采样器（惩罚/mirostat/grammar）记住已选的 token。
5. `<h2>` 采样链 + **图2【链：逐个 apply】**（`vflow` 或 `layers`）：`chain = [penalties, top_k, top_p, temp, dist]`，`sample` 时按顺序对 `cur_p` 逐个 `apply`。**代码2（伪代码，链的用法）**：
```
chain = llama_sampler_chain_init(params)
chain.add(llama_sampler_init_penalties(...))   # 压低重复
chain.add(llama_sampler_init_top_k(40))        # 留前 40
chain.add(llama_sampler_init_top_p(0.95, 1))   # 核采样
chain.add(llama_sampler_init_temp(0.8))        # 温度
chain.add(llama_sampler_init_dist(seed))       # 按概率随机选
id = llama_sampler_sample(chain, ctx, -1)      # 跑全链 -> 返回 token id
```
6. `<h2>` 常见采样器与 greedy vs dist + **图3【常见采样器】**（`<table class="t">`）：列 `greedy`(argmax)/`dist`(按概率随机)/`top_k`(留前 k)/`top_p`(核, 累积 p)/`min_p`(相对最大值的下限)/`temp`(缩放)/`penalties`(重复/频率/存在惩罚)/`mirostat`(目标困惑度)，列"作用"。正文：`greedy` 确定性（每次同输入同输出）；`dist` 随机（靠 seed）。⚠ 旧全局 `llama_sample_top_k/top_p/temperature` 已移除，统一为采样器对象 + 链。
7. **深挖1**"温度（temperature）到底在做什么？"：把 logits 除以 T 再 softmax。`T<1` 让分布更尖锐（更确定、更保守），`T>1` 更平（更随机、更有创意），`T->0` 退化成 greedy。它不改候选集，只改"软硬"。
8. **深挖2**"top-k 和 top-p（nucleus）有何不同？"：`top_k` 留**固定个数**（前 k 大）；`top_p` 留**累积概率达 p** 的最小集合（候选数随分布自适应——分布尖时少、平时多）。常组合使用：先 top_k 砍掉长尾，再 top_p 自适应收口。
9. **深挖3**"为什么做成'链'而不是一个大采样函数？"：可组合 + 可配置 + 状态隔离——每个采样器独立、自带状态（如 penalties 记历史、mirostat 记反馈），顺序可调；用户能用配置拼出任意策略，不必改引擎。⚠ 注意 grammar 采样器（L23）通常**不放进链**，作为独立对象按 `grammar_first` 在链前/后施加。
10. `key`（✅）：采样 = 在 `llama_token_data_array` 上逐步裁剪塑形再选一个；`llama_sampler_i`(name/accept/apply/reset) + `llama_sampler{iface,ctx}`；链 `chain_init`/`chain_add`/`sample`；`greedy`=argmax、`dist`=按概率随机；`top_k`/`top_p`/`temp`/`penalties`/`mirostat`；旧全局 `llama_sample_*` 已废。
11. `spark`（💡）：把采样从"一个写死的大函数"拆成"**一串可插拔的小变换**"，是典型的**责任链/管道**设计——和 L21 之前见过的 ggml 算子链、L16 的建图积木同一种味道：用小而独立的部件组合出复杂行为。于是"换个采样策略"只是换链里几个环，引擎主干一动不动。

必须讲到：logits -> 候选数组 -> 选 token；`llama_sampler_i`/`apply`/`accept`；链 `chain_init/add` + 顺序；greedy vs dist；top_k/top_p/temp/penalties 含义；旧全局 API 已移除；grammar 在链外（预告 L23）。

- [ ] **Step 5: quiz（21）**：
- MCQ1 "`greedy` 采样选哪个 token？" -> 正确："logit 最大的那个（argmax，确定性）"；干扰：随机一个 / 最后一个 / logit 最小的。
- MCQ2 "`top_p`（nucleus）采样保留哪些候选？" -> 正确："按概率从高到低累加、达到阈值 p 的最小候选集合"；干扰：固定前 50 个 / 概率大于 p 的 / 全部。
- MCQ3 "把采样做成'链'（chain）的主要好处？" -> 正确："可组合——按顺序施加多个独立、可配置的采样器"；干扰：跑得更快 / 省内存 / 只能用一个采样器。
- OPEN "结合 L20 和 L23，说说采样器为什么是在'词表的 token 空间'里工作，以及 grammar 如何作为一种'掩码'来约束这一步。"

- [ ] **Step 6: 重建+校验**（同上；index "共 21 课 · 4 个部分"；硬性达标；grep 渲染 HTML 确认无裸标签泄漏）。
- [ ] **Step 7: commit**（`feat: add lesson 21 sampling (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 3: 课 22「对话模板 / Chat templates」

> 第四部分第 9 课、M4b 第 3 课。你在聊天界面输入一句话，模型怎么知道"谁说的、一轮在哪结束、该从哪接着生成"？这一课讲对话模板：把 `[{role,content}...]` 消息列表按模型约定的格式拼成**一段带标记的提示词字符串**（如 ChatML 的 `<|im_start|>`、Llama-2 的 `[INST]`），以及 llama.cpp 的"内建模板表"与"Jinja"两条路。接 L20（模板输出要再 tokenize）。

**Files:** `src/part4.py`（追加 `LESSON_22`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `22-chat-templates.html`。
**⚠ 转义重点**：本课大量出现 `<|im_start|>`/`<|im_end|>`/`[INST]` 等标记，含 `<`/`>`/`|`——**正文与 quiz 中一律写成 `&lt;|im_start|&gt;`**，构建后 grep 渲染 HTML 确认无裸标签泄漏。

- [ ] **Step 1-3: 登记**
```python
("22-chat-templates.html", "对话模板", "Chat templates",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"22-chat-templates.html": ("llm_chat_template · 内建模板 vs Jinja · ChatML/Llama/Gemma",
                           "llm_chat_template; built-in vs Jinja; ChatML/Llama/Gemma"),
"22-chat-templates.html": part4.LESSON_22,
```

- [ ] **Step 4: 执笔 LESSON_22（双语）**。**结构**：
1. `lead`：模型吃的是一长串 token（L20），可对话是一条条带角色的消息。中间这步翻译就是对话模板：把 `system/user/assistant` 的消息列表，按这个模型训练时用的格式，拼成**一段带特殊标记的字符串**，再交给 L20 去 tokenize。格式不对，模型就读不懂"轮次"。
2. `analogy`（🔌）：对话模板像**公文信封格式**：同样的内容，不同机构抬头/落款不同。ChatML 把每条消息裹成 `&lt;|im_start|&gt;role ... &lt;|im_end|&gt;`，Llama-2 用 `[INST] ... [/INST]`。模板就是"把消息装进这个模型认得的信封"。
3. `<h2>` 为什么需要模板 + **图1【消息 -> 提示词 -> token】**（`flow`，本课概念图）：`[{system},{user}]` -> `apply_template` -> `"&lt;|im_start|&gt;system...&lt;|im_end|&gt;..."` -> `tokenize`(L20) -> model。正文：模板负责"消息 -> 字符串"，tokenize 负责"字符串 -> token"，两步接力。
4. `<h2>` 内建模板表 + **代码1（简化自 `src/llama-chat.h`）**：
```
enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML, LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_3, LLM_CHAT_TEMPLATE_GEMMA,
    LLM_CHAT_TEMPLATE_MISTRAL_V7, /* ... 约 55 种 ... */
    LLM_CHAT_TEMPLATE_UNKNOWN,
};
```
   **图2【代表模板的信封】**（`<table class="t">`）：列 模板 / 起止标记，行 ChatML(`&lt;|im_start|&gt;`/`&lt;|im_end|&gt;`)、Llama-2(`[INST]`/`[/INST]`)、Gemma(`&lt;start_of_turn&gt;`)、Llama-3(`&lt;|start_header_id|&gt;`)。
5. `<h2>` 检测与应用 + **代码2（伪代码，套模板）**：
```
tmpl = llm_chat_detect_template(template_str)   # 先按名, 再按模板体子串猜
dest = ""
llm_chat_apply_template(tmpl, messages, dest, add_ass=True)  # 渲染成字符串
# add_ass: 末尾追加 assistant 起始标记, 好让模型接着写回答
```
   正文：`detect` 把"模板字符串/名字"映到枚举（先精确匹配名，再按是否含 `&lt;|im_start|&gt;`/`[INST]` 等子串启发式猜）；`apply` 按枚举把消息拼出来。
6. `<h2>` 两条路：内建 vs Jinja + **图3【内建 vs Jinja】**（`cols`）：左"内建（`src/llama-chat.cpp`）：固定枚举、纯字符串拼接、无依赖、快"；右"Jinja（`common/jinja/` + `common/chat.cpp`）：模型 GGUF 自带任意模板时走真 Jinja，还能加工具调用约束/输出解析"。正文：C API `llama_chat_apply_template` ⚠ **只收模板字符串、不再带 `llama_model *`**，注释明确"不用 jinja 解析器，只支持预定义列表"；要任意模板/工具调用就上 `common/chat.cpp` 的 `common_chat_templates`。衔接：拼好的提示词进 L20 tokenize、再进 L17 decode。
7. **深挖1**"`add_ass`（add_assistant）这个开关是什么？"：渲染时在末尾追加 assistant 角色的**起始标记**（但不含内容），等于给模型"递话筒"——让它从 assistant 该说话的位置接着生成。补全提示（continue）时则关掉它。
8. **深挖2**"为什么 C API 的 `llama_chat_apply_template` 不再带 model 参数？"：解耦——套模板只需要"模板字符串 + 消息"，不需要整个模型；模板字符串本身可来自 GGUF 元数据或用户指定。注释也点明它只走内建预定义列表，要 Jinja 去上层 `common/`。
9. **深挖3**"内建硬编码模板 vs Jinja，何时用哪个？"：模型在 GGUF 里自带了 `chat_template`（Jinja 文本）时，上层 `common_chat_templates`（`use_jinja`）渲染它，最忠实；没有或想要零依赖/已知模型时，用 `llama-chat.cpp` 的内建枚举，快且稳。两者覆盖"已知模型"和"任意模型"两种情况。
10. `key`（✅）：对话模板 = 消息列表 -> 模型约定格式的提示词串；`enum llm_chat_template`(~55 种)；`llm_chat_detect_template`(名/子串) + `llm_chat_apply_template`(渲染, `add_ass`)；`struct llama_chat_message{role,content}`；C API `llama_chat_apply_template`(⚠ 无 model 参数) / `llama_chat_builtin_templates`；任意模板/工具调用走 `common/jinja` + `common/chat.cpp`。
11. `spark`（💡）：对话模板把"**模型的对话方言**"收进一张表——同样的消息，换个模型就换个信封，引擎主干不必关心。它和 L15 的"表驱动架构"是同一种智慧：把"每个模型各不相同的部分"沉淀成数据/模板，让通用代码照着办。读懂它，你就明白为什么同一个 llama.cpp 能聊几十种模型的"话"。

必须讲到：消息列表 -> 提示词字符串 -> tokenize 的接力；`llm_chat_template` 枚举 + 代表标记；`detect`/`apply` + `add_ass`；`llama_chat_message{role,content}`；C API 无 model 参数；内建 vs Jinja(`common/jinja`) 两条路。

- [ ] **Step 5: quiz（22）**（⚠ 选项里的标记也要转义 `&lt;|im_start|&gt;`）：
- MCQ1 "对话模板（chat template）做什么？" -> 正确："把带角色的消息列表拼成该模型约定格式的提示词字符串"；干扰：把文本切成 token / 给回答打分 / 压缩对话历史。
- MCQ2 "ChatML 模板用哪对标记包裹每条消息？" -> 正确："`&lt;|im_start|&gt;` 和 `&lt;|im_end|&gt;`"；干扰：`[INST]`/`[/INST]` / `&lt;s&gt;`/`&lt;/s&gt;` / `{{ }}`。
- MCQ3 "`add_ass`（add_assistant）为真时会？" -> 正确："在末尾追加 assistant 起始标记，让模型接着生成回答"；干扰：删除 system 消息 / 把回答翻译成英文 / 关闭采样。
- OPEN "结合 L20，说说为什么要'先套对话模板、再 tokenize'，如果把顺序反过来会出什么问题。"

- [ ] **Step 6: 重建+校验**（同上；index "共 22 课 · 4 个部分"；硬性达标；**重点 grep 渲染 HTML 确认 `&lt;|im_start|&gt;` 等无裸标签泄漏、无 `&amp;lt;` 双重转义**）。
- [ ] **Step 7: commit**（`feat: add lesson 22 chat templates (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 4: 课 23「语法约束 / Grammar (GBNF)」

> 第四部分第 10 课、M4b 第 4 课。想让模型**只**输出合法 JSON、或严格走某种格式？这一课讲 GBNF 语法约束：用一套类 BNF 规则描述"允许的输出长什么样"，在每一步采样前把**不符合语法的 token 掩掉**（`logit = -INFINITY`），于是模型只能在"语法允许的路"上走。接 L21（grammar 是一种掩码采样器）、L20（在 token 空间约束）。

**Files:** `src/part4.py`（追加 `LESSON_23`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `23-grammar.html`。
**⚠ 转义重点**：`llama_gretype` 的 `_TOKEN`(`&lt;[id]&gt;`)、负号 `[^\n]`、以及伪代码里的 `-&gt;`/`&lt;=`——正文/quiz 里 `<`/`>`/`&` 一律转义。

- [ ] **Step 1-3: 登记**
```python
("23-grammar.html", "语法约束", "Grammar (GBNF)",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"23-grammar.html": ("GBNF · llama_grammar · 掩码采样(-inf) · root 规则",
                    "GBNF; llama_grammar; mask sampling(-inf); root rule"),
"23-grammar.html": part4.LESSON_23,
```

- [ ] **Step 4: 执笔 LESSON_23（双语）**。**结构**：
1. `lead`：默认情况下模型想说啥说啥，但很多场景（调 API、填表单、结构化抽取）要求输出**严格合法**，比如必须是 JSON。GBNF 语法约束让你写一套规则，引擎在每步采样时只放行"此刻语法允许的 token"，于是模型**不可能**生成出格式非法的东西。
2. `analogy`（🔌）：语法约束像**表单的下拉框**：你不能随便填，只能从合法选项里挑。grammar 在每一步采样前，把"此刻语法不允许的 token"全划掉（设成负无穷），模型只能从剩下的合法 token 里选——一步都越不了界。
3. `<h2>` 为什么需要语法 + **图1【grammar 掩码采样】**（`flow`，本课概念图）：`logits` -> `grammar.apply`(非法 token 置 `-inf`) -> 采样(L21) -> 合法 `token` -> `grammar.accept`(推进语法状态)。正文：约束发生在 logits 和选 token 之间，是采样管线里的一个**掩码**环节。
4. `<h2>` GBNF 是什么 + **代码1（简化自 `grammars/json.gbnf`）**：
```
root   ::= object
object ::= "{" ws ( string ":" ws value )? "}"
value  ::= object | string | number | "true" | "false"
string ::= "\"" [^"]* "\""
```
   **图2【GBNF 语法元素】**（`<table class="t">`）：`::=`(定义规则)/`|`(备选)/`[...]`(字符类)/`[^...]`(取反)/`*`(0+)/`+`(1+)/`?`(可选)/`()`(分组)/`root`(入口)。正文：规则像递归的"积木拼装说明"，从 `root` 展开。
5. `<h2>` 语法怎么约束采样 + **代码2（伪代码，apply/accept）**：
```
# 采样前: 掩掉非法候选 (llama_grammar_apply_impl)
for cand in cur_p:
    if not grammar_allows(stacks, cand.id):
        cand.logit = -INFINITY        # 非法 -> 永不会被选中
# 选定 token 后: 推进语法状态 (llama_grammar_accept_impl)
grammar.accept(chosen_token)          # 沿规则栈往前走一步
```
   正文：`llama_grammar_apply_impl` 按当前规则栈（`stacks`）算出哪些 token 合法、把非法的 logit 砸到 `-inf`；选定后 `llama_grammar_accept_impl` 推进栈。两者一掩一进，逐 token 守住合法性。
6. `<h2>` 元素类型与作为采样器 + **图3【llama_gretype】**（`cellgroup` 或 `<table class="t">`）：`_CHAR`(字面)/`_CHAR_RNG_UPPER`(范围上界)/`_CHAR_NOT`(取反类)/`_CHAR_ANY`(`.`)/`_RULE_REF`(引用规则)/`_ALT`(`|`)/`_END`。正文：grammar 通过采样器接入——`llama_sampler_init_grammar(vocab, grammar_str, root)`，其 `apply` 调掩码、`accept` 推进。⚠ 惰性变体 `llama_sampler_init_grammar_lazy_patterns`（旧 `_grammar_lazy` 已弃用）。衔接：它就是 L21 那套采样器接口的一个实现，只不过通常在链外按 `grammar_first` 施加。
7. **深挖1**"grammar 和采样器（L21）是什么关系？"：grammar 本质是一个**采样器**——`llama_sampler_init_grammar` 返回的就是 `llama_sampler`，`apply`=掩码、`accept`=推进。只是它有状态（要记住语法走到哪），且通常**不混进主链**，而是作为独立对象按 `grammar_first` 在链前或链后单独施加。
8. **深挖2**"惰性/触发语法（lazy/trigger）是干嘛的？"：有时只想在**某个触发串/ token 出现后**才开始约束（典型：工具调用——模型先自由说话，遇到 `&lt;tool_call&gt;` 才切到 JSON 约束）。`llama_grammar` 的 `lazy`/`awaiting_trigger`/`trigger_patterns` 实现这点，对应 `grammar_lazy_patterns` 采样器。
9. **深挖3**"为什么用 token 级掩码、而不是'生成完再校验'？"：因为事后校验会**浪费**——生成到一半发现非法只能重来。token 级掩码保证**每一步都合法**，模型从不踏出语法边界，既高效又可靠（永远产出合法结果，无需重试）。
10. `key`（✅）：GBNF = 类 BNF 规则（`::=`/`|`/`[...]`/`*+?`/`root`）描述合法输出；`llama_grammar`(rules/stacks/partial_utf8 + lazy/trigger)；`apply`(非法 token 置 `-inf` 掩码) + `accept`(推进栈)；`enum llama_gretype`；接入采样 `llama_sampler_init_grammar`(/`_lazy_patterns`)；通常在链外按 `grammar_first`。
11. `spark`（💡）：语法约束把"**结构正确性**"从"事后祈祷"变成"**生成时保证**"——在 token 级别就堵死所有非法路径。这是 llama.cpp 把"自由生成"和"严格格式"统一进同一套采样接口的巧思：约束不是外挂的后处理，而是采样管线里的一个掩码环节。理解它，你就握住了"让 LLM 可靠输出结构化数据"的钥匙。

必须讲到：GBNF 规则/`root`；掩码采样（`-inf`）`apply` + `accept` 推进；`llama_grammar` 字段；`llama_gretype` 几种；接入 `llama_sampler_init_grammar`(/`_lazy_patterns`，旧 `_grammar_lazy` 弃用)；链外 `grammar_first`；token 级 vs 事后校验。

- [ ] **Step 5: quiz（23）**：
- MCQ1 "GBNF 语法约束怎么起作用？" -> 正确："采样时把不符合语法的 token 的 logit 设成负无穷（掩码），模型只能选合法 token"；干扰：生成完用正则校验 / 微调模型 / 改 prompt 提示。
- MCQ2 "GBNF 文法的入口（起始）规则叫什么？" -> 正确："`root`"；干扰：`main` / `start` / `entry`。
- MCQ3 "相比'生成完再校验'，token 级语法掩码的好处？" -> 正确："每一步都保证合法，不会生成到一半才发现非法而重来"；干扰：占内存更少 / 不需要词表 / 让模型更有创意。
- OPEN "结合 L21，说说 grammar 作为一种'掩码采样器'为什么常被放在采样链之外、按 `grammar_first` 单独施加。"

- [ ] **Step 6: 重建+校验**（同上；index "共 23 课 · 4 个部分"；硬性达标；grep 渲染 HTML 确认 `[^"]`、`&lt;[id]&gt;` 等无裸标签泄漏）。
- [ ] **Step 7: commit**（`feat: add lesson 23 grammar (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 5: 课 24「LoRA 适配器 / LoRA adapters」

> 第四部分第 11 课、M4b 第 5 课（第四部分收官）。想让大模型学会新风格/新任务，但全量微调要改几十 GB 权重、太贵？LoRA 用两个小矩阵 A、B 给权重加一个**低秩增量**，几 MB 的适配器就能改变行为，且即插即用、可叠加可卸下。这一课讲 llama.cpp 怎么加载、应用 LoRA 适配器与控制向量。接 L16（增量在建图时折进 matmul）、L17（适配器挂在 context 上）。

**Files:** `src/part4.py`（追加 `LESSON_24`）、`src/registry.py`、`src/shell.py`、`src/quizzes.py`。产出 `24-lora-adapters.html`。

- [ ] **Step 1-3: 登记**
```python
("24-lora-adapters.html", "LoRA 适配器", "LoRA adapters",
 "第四部分 · llama 推理内部", "Part 4 · Inside llama inference"),
"24-lora-adapters.html": ("LoRA 低秩增量 scale·B·A · llama_set_adapters_lora · 控制向量",
                          "LoRA low-rank delta scale*B*A; llama_set_adapters_lora; control vectors"),
"24-lora-adapters.html": part4.LESSON_24,
```

- [ ] **Step 4: 执笔 LESSON_24（双语）**。**结构**：
1. `lead`：让一个几十亿参数的模型适配新任务，全量微调要重训并存下**整套**权重，又贵又笨。LoRA 换个思路：冻结原权重，只学两个**小**矩阵 A、B，给每个目标权重加一个低秩增量。适配器往往只有几 MB，能即插即用、按比例叠加、随时卸下。
2. `analogy`（🔌）：LoRA 像给镜头套**滤镜**：原镜头（基础权重）一点不动，套上一片轻巧滤镜（`B·A` 低秩增量）就改变成像风格；不喜欢随时摘下，也能叠几片。控制向量则像一个**调色旋钮**，沿某个固定方向整体平移画面。
3. `<h2>` 为什么需要 LoRA + **图1【全量微调 vs LoRA】**（`vflow` 或 `cols`，本课概念图）：左"全量微调：改**所有**权重，存几十 GB，每个任务一份"；右"LoRA：冻结 W，只学小矩阵 A(降维)、B(升维)，存几 MB"。正文：低秩 `r` 远小于权重维度，于是参数量、显存、存储都大幅缩水。
4. `<h2>` LoRA 数学 + **图2【低秩增量】**（`flow`）：`x` -> `A`(d×r 降到 r) -> `B`(r×d 升回 d) -> `× scale` -> 加到 `W·x`。**代码1（简化自 `src/llama-graph.cpp` `build_lora_mm`）**：
```
res = W * cur;                       // 基础权重输出
ab_cur = B * (A * cur);              // 低秩两步: 先降维 A, 再升维 B
res = res + scale * ab_cur;          // 增量叠加; scale = alpha/rank * 用户比例
```
   讲：有效增量 = `scale · B · A`；`rank = b->ne[0]`；`scale` 由 `alpha` 和用户给的比例算出。
5. `<h2>` 加载与应用 + **代码2（伪代码，挂载）**：
```
adapter = llama_adapter_lora_init(model, "style.gguf")   # 读 A/B 张量(GGUF)
llama_set_adapters_lora(ctx, [adapter], n=1, scales=[0.8])  # 批量挂载, 各带 scale
# ... decode 若干步, 输出带上这个风格 ...
llama_set_adapters_lora(ctx, [], n=0, NULL)              # n=0 => 清空, 卸下全部
```
   正文：⚠ 当前是**批量复数** `llama_set_adapters_lora`（可同时挂多个、各带 scale）；**单数 `llama_set_adapter_lora`/`llama_rm_adapter_lora`/`llama_clear_adapter_lora` 已不存在**（清空 = `n=0`）。加载 `llama_adapter_lora_init`（旧 `llama_lora_adapter_init` 已改名）。
6. `<h2>` 控制向量与衔接 + **图3【LoRA vs 控制向量】**（`cols`）：左"LoRA：低秩**权重增量**，改 matmul 输出（`build_lora_mm`）"；右"控制向量（cvec）：沿固定**方向平移激活**，加进残差流（`llama_set_adapter_cvec`）"。正文：适配器挂在 `llama_context` 上（L17），`decode` 建图（L16）时 `build_lora_mm` 把增量**折进每个相关 matmul**——**不复制权重**、不改基础模型。⚠ cvec 的 C API 是 `llama_set_adapter_cvec`（旧 `llama_apply_adapter_cvec`/`llama_control_vector_apply` 已移除）。这也为第四部分收尾：从加载到可控生成，一个模型如何被"驱动 + 调味"。
7. **深挖1**"为什么'低秩'就够用？"：经验与理论都表明，微调给权重带来的变化往往集中在一个**低维子空间**里——用 `r=8/16` 的两个小矩阵就能捕捉大部分任务适配。于是花极小的参数，逼近全量微调的效果。
8. **深挖2**"为什么挂载 API 是批量复数 `llama_set_adapters_lora`？"：因为可以**同时挂多个 LoRA**、各给一个 scale（叠加多种风格/能力），所以接口天然是一组 `{adapters[], scales[]}`；清空就是传 `n=0`。旧的单数 `set/rm/clear` 三件套被这一个批量 setter 取代了——写老教程的名字会编译不过。
9. **深挖3**"LoRA 和控制向量有何不同？"：LoRA 改的是**权重**（给 matmul 加低秩增量，影响该层全部计算）；控制向量改的是**激活**（直接给某些层的残差流加一个固定方向向量，更轻、更像"调味"）。前者能学复杂适配，后者擅长沿某个语义方向（如"更正式""更乐观"）平移风格。
10. `key`（✅）：LoRA = 冻结 W、只学小矩阵 A/B，增量 `scale·B·A`（`build_lora_mm`，`src/llama-graph.cpp`）；`struct llama_adapter_lora`(`ab_map`/`alpha`)；加载 `llama_adapter_lora_init`、挂载 ⚠ 批量 `llama_set_adapters_lora`(单数已移除, `n=0` 清空)；控制向量 `llama_set_adapter_cvec`（沿方向平移激活）；适配器挂 context、`decode` 时折进 matmul，**不复制权重**。
11. `spark`（💡）：LoRA 把"**改变模型行为**"从"重训整套权重"降到"加一片几 MB 的低秩滤镜"——基础模型只读、增量即插即用。它和这一部分反复出现的主题一脉相承：**只读的知识**（权重）与**可变的部分**（适配器/上下文/采样策略）分开，于是一份大模型能被千变万化地复用。学到这里，你已经走完了第四部分——从一个 `.gguf` 文件被加载，到它如何被驱动、约束、并轻量改造成你想要的样子。

必须讲到：LoRA 低秩增量 `scale·B·A` + `build_lora_mm`；冻结基础权重、适配器几 MB；`llama_adapter_lora_init` + ⚠ 批量 `llama_set_adapters_lora`(单数已移除)；控制向量 `llama_set_adapter_cvec` vs LoRA 区别；适配器挂 context、decode 折进 matmul 不复制权重。

- [ ] **Step 5: quiz（24）**：
- MCQ1 "LoRA 怎么改变模型行为？" -> 正确："冻结原权重，用两个小矩阵 A、B 给权重加一个低秩增量 `scale·B·A`"；干扰：重训全部权重 / 换一个词表 / 改采样温度。
- MCQ2 "当前 llama.cpp 给上下文挂载 LoRA 用哪个 API？" -> 正确："批量的 `llama_set_adapters_lora`（单数 set/rm/clear 已移除，`n=0` 清空）"；干扰：`llama_set_adapter_lora`（单数）/ `llama_lora_apply` / 重新加载模型。
- MCQ3 "LoRA 和控制向量（control vector）的主要区别？" -> 正确："LoRA 给权重加低秩增量（改 matmul），控制向量沿固定方向平移激活（加进残差流）"；干扰：两者完全一样 / LoRA 改词表、cvec 改采样 / 都要重训模型。
- OPEN "结合 L16，说说为什么挂上 LoRA 能'不复制权重'就生效——`build_lora_mm` 在建图的哪一步把增量加进来。"

- [ ] **Step 6: 重建+校验**（同上；index "共 24 课 · 4 个部分"；硬性达标；grep 渲染 HTML 确认无裸标签泄漏）。
- [ ] **Step 7: commit**（`feat: add lesson 24 lora adapters (bilingual) with quiz` + `Assisted-by: GitHub Copilot`）。

---

## Task 6: M4b 验收与合并

> 5 课全部完成（且各自 spec + 质量双审通过）后做整体验收，再合并回 master。

- [ ] **Step 1: 干净重建零漂移**
```bash
cd /home/verden/course/llama-cpp-visual-guide && rm -f index.html lessons/*.html \
  && cd src && python3 build.py && python3 check_html.py && python3 check_links.py \
  && cd .. && git status --short
```
预期：0 error/0 warning、所有链接解析、`git status` 干净（HTML 与源同步无漂移）。

- [ ] **Step 2: 密度审计（课 20-24）**。对每课确认 zhCJK>=4000、enCJK=0、diag>=3、acc>=2、pre>=2、逐段 `<p>` 对齐：
```bash
cd /home/verden/course/llama-cpp-visual-guide && python3 - <<'PY'
import re, sys; sys.path.insert(0,'src'); import part4
def cjk(s): return len(re.findall(r'[\u4e00-\u9fff]', s))
DIAG=['layers','vflow','flow','cols','cellgroup','timeline']
def diag(s): return sum(s.count(f'class="{c}"')+s.count(f'class="{c} ') for c in DIAG)+len(re.findall(r'<table class="t"',s))
def par(zh,en):
    zs=re.split(r'<h2',zh); es=re.split(r'<h2',en)
    if len(zs)!=len(es): return f"H2!{len(zs)}/{len(es)}"
    return "OK" if all((z.count('<p>')+z.count('<p '))==(e.count('<p>')+e.count('<p ')) for z,e in zip(zs,es)) else "P-MISMATCH"
for n in range(20,25):
    L=getattr(part4,f"LESSON_{n}"); zh,en=L['zh'],L['en']
    print(n, "zhCJK",cjk(zh),"enCJK",cjk(en),"diagZ",diag(zh),"diagE",diag(en),
          "acc",zh.count('<details'),"pre",zh.count('<pre class="code"'),"parity",par(zh,en))
PY
```
预期：每课 zhCJK>=4000、enCJK=0、diagZ==diagE>=3、acc>=2、pre>=2、parity OK。

- [ ] **Step 3: 导航检查**：`grep -o '共 [0-9]* 课 · [0-9]* 个部分' index.html`（应 "共 24 课 · 4 个部分"）；`grep -o 'lessons/2[0-4]-[a-z-]*\.html' index.html | sort -u`（20-24 五条齐全）；确认无越界 `第 N 课`（N>24）。

- [ ] **Step 4: 标记路线图 M4b 完成**：把 `docs/superpowers/plans/2026-06-13-llama-cpp-visual-guide-roadmap.md` 里 M4b 行状态改"完成"、第 93 行复选框由 "M4a 完成，M4b 待写" 改为 "M4a/M4b 完成"（M4 整体可勾 `[x]`）。commit `docs: mark M4b (part 4 lessons 20-24) complete in roadmap` + `Assisted-by: GitHub Copilot`。

- [ ] **Step 5: finishing-a-development-branch**：用该 skill，向用户提 4 选项收尾（用户惯选"本地 `--no-ff` 合并 + 删分支"）-> checkout master、`git merge --no-ff build/m4b-part4`、在合并结果上复跑三个校验、删分支。

---

## Self-Review（写完即自查，对照 spec）

**1. spec 覆盖**：M4b spec = 第四部分下半 5 课（20 词表 / 21 采样 / 22 对话模板 / 23 语法 / 24 LoRA）。Task 1-5 一一对应；每课含登记、双语结构（lead/analogy/3-4 `<h2>` 含图与代码/3 深挖/key/spark）、quiz、重建、commit；Task 6 验收 + 合并。覆盖完整。

**2. 占位符扫描**：无 "TBD/TODO/类似上面"；每个 Step 给了真实代码块、图类型、quiz 选项与正确答案、commit 文案。代码片段均"简化自 文件+符号、无行号"。

**3. 类型/名字一致性**（对照"验证过的源码事实"，与 M4a 既有名字）：
- L20 `llama_vocab`(pimpl)/`tokenize`/`token_to_piece`/`LLAMA_VOCAB_TYPE_*`/`llama_vocab_n_tokens`(旧 `llama_n_vocab` 弃用)——一致。
- L21 `src/llama-sampler.{h,cpp}`(非 sampling)/`llama_sampler_i`(apply 必需)/`llama_token_data_array`/`chain_init/add`/`greedy`/`dist`/默认链顺序——一致；旧全局 `llama_sample_*` 已移除已注明。
- L22 `enum llm_chat_template`/`llm_chat_detect_template`/`llm_chat_apply_template`/`llama_chat_message{role,content}`/C API 无 model 参数/`common/jinja`(非 minja)——一致。
- L23 GBNF `root`/`llama_grammar`/`apply`(-inf 掩码)+`accept`/`llama_gretype`/`llama_sampler_init_grammar`(/`_lazy_patterns`，旧 `_grammar_lazy` 弃用)——一致。
- L24 `scale·B·A`/`build_lora_mm`(`src/llama-graph.cpp`)/`llama_adapter_lora`/`llama_adapter_lora_init`/⚠ 批量 `llama_set_adapters_lora`(单数 set/rm/clear 已移除)/`llama_set_adapter_cvec`(旧 `llama_apply_adapter_cvec` 移除)——一致。
- 跨课衔接编号：L20<->L21(token 空间)、L21<->L23(grammar 掩码采样器/链外 `grammar_first`)、L22<->L20(先模板后 tokenize)、L24<->L16(`build_lora_mm` 折进 matmul)/L17(挂 context)——一致。
- 文件名 `20-vocabulary`/`21-sampling`/`22-chat-templates`/`23-grammar`/`24-lora-adapters`、part 标签"第四部分 · llama 推理内部"——五课统一。

**4. 转义红线**：L22（`&lt;|im_start|&gt;` 等）、L23（`&lt;[id]&gt;`/`[^"]`/`-&gt;`）已在 Files/Step 6 标注"grep 渲染 HTML 防裸标签泄漏 + 防 `&amp;lt;` 双重转义"。伪代码注释用 `#`、C++ 片段用 `//`；en 纯 ASCII、不用 `→`。

