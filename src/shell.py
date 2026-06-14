"""Shared HTML shell (CSS design system + navigation) for the LangChain tutorial."""

import base64

# ---- favicon (inline SVG, base64) ----
_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='7' fill='#1a7f64'/>"
    "<text x='16' y='23' font-family='system-ui,sans-serif' font-size='20'"
    " font-weight='700' fill='#fff' text-anchor='middle'>λ</text></svg>"
)
FAVICON = "data:image/svg+xml;base64," + base64.b64encode(_FAVICON_SVG.encode()).decode()


def head_meta(title, description, og_type="website"):
    """SEO / social meta tags + favicon for a page <head>."""
    t = title.replace('"', "&quot;")
    d = description.replace('"', "&quot;")
    return (
        f'<meta name="description" content="{d}">\n'
        f'<meta name="theme-color" content="#1a7f64">\n'
        f'<link rel="icon" type="image/svg+xml" href="{FAVICON}">\n'
        f'<meta property="og:type" content="{og_type}">\n'
        f'<meta property="og:site_name" content="LangChain 图解教程">\n'
        f'<meta property="og:title" content="{t}">\n'
        f'<meta property="og:description" content="{d}">\n'
        f'<meta name="twitter:card" content="summary">\n'
        f'<meta name="twitter:title" content="{t}">\n'
        f'<meta name="twitter:description" content="{d}">'
    )


# Ordered list of all pages: (filename, short title, part label)
PAGES = [
    ("01-what-is-langchain.html", "LangChain 是什么", "第一部分 · 宏观全景"),
    ("02-monorepo.html", "Monorepo 全景", "第一部分 · 宏观全景"),
    ("03-lifecycle.html", "一次调用的生命周期", "第一部分 · 宏观全景"),
    ("04-messages.html", "消息系统", "第二部分 · 用户视角"),
    ("05-chat-models.html", "聊天模型", "第二部分 · 用户视角"),
    ("06-tools.html", "工具 Tools", "第二部分 · 用户视角"),
    ("07-agents-intro.html", "Agent 入门", "第二部分 · 用户视角"),
    ("08-runnable.html", "Runnable 万物之基", "第三部分 · 内部源码"),
    ("09-runnable-compose.html", "Runnable 如何组合", "第三部分 · 内部源码"),
    ("10-output-parsers.html", "输出解析器 Output Parsers", "第三部分 · 内部源码"),
    ("11-chat-internals.html", "聊天模型内部", "第三部分 · 内部源码"),
    ("12-tool-internals.html", "工具调用内部", "第三部分 · 内部源码"),
    ("13-agent-internals.html", "Agent 内部", "第三部分 · 内部源码"),
    ("14-streaming-callbacks.html", "Streaming 与 Callbacks", "第三部分 · 内部源码"),
    ("15-contributing.html", "读源码 / 调试 / 测试 / 贡献", "第四部分 · 进阶"),
    ("16-prompts.html", "提示词 Prompts", "第五部分 · 自己动手做 Agent"),
    ("17-rag.html", "RAG 检索增强", "第五部分 · 自己动手做 Agent"),
    ("18-custom-middleware.html", "写自己的中间件", "第五部分 · 自己动手做 Agent"),
    ("19-runtime-context.html", "运行时上下文与健壮性", "第五部分 · 自己动手做 Agent"),
    ("20-capstone.html", "端到端实战：拼一个客服 Agent", "第五部分 · 自己动手做 Agent"),
    ("21-langchain-vs-autogen.html", "横向对比：LangChain vs AutoGen", "第六部分 · 番外篇"),
    ("22-ai-stack.html", "全栈坐标系：从 LangChain 缩放到整个生态", "第六部分 · 番外篇"),
    ("23-learning-map.html", "隔壁层学习地图：L5 推理 · L6 向量检索", "第六部分 · 番外篇"),
    ("24-langgraph-mental-model.html", "深入 LangGraph：图 / 状态 / 节点 / 边", "第七部分 · 深入 LangGraph"),
    ("25-langgraph-pregel-engine.html", "执行引擎：Pregel 与超步", "第七部分 · 深入 LangGraph"),
    ("26-langgraph-persistence-control.html", "持久化 · 中断 · 控制流", "第七部分 · 深入 LangGraph"),
    ("27-glossary.html", "术语表 · 概念索引", "第八部分 · 速查"),
]

INDEX_FILE = "index.html"

CSS = r"""
* { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: #f6f7f9; --panel: #ffffff; --panel-2: #f0f2f5; --ink: #1d2129;
  --muted: #5b6470; --faint: #8a939f; --line: #e1e5ea;
  --accent: #1a7f64; --accent-soft: #e4f3ee; --accent-ink: #0f5c48;
  --blue: #2563eb; --blue-soft: #e7efff; --amber: #b4690e; --amber-soft: #fdf1dd;
  --purple: #7c3aed; --purple-soft: #f0e9ff; --red: #d23f3f; --red-soft: #fbe6e6;
  --code-bg: #0f172a; --code-ink: #e2e8f0; --code-line: #1e293b;
  --shadow: 0 1px 2px rgba(16,24,40,.06), 0 8px 24px rgba(16,24,40,.06);
  --radius: 14px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1116; --panel: #161b22; --panel-2: #1c232c; --ink: #e6edf3;
    --muted: #9aa6b2; --faint: #6e7a86; --line: #2a323c;
    --accent: #3fb892; --accent-soft: #14302a; --accent-ink: #8ee0c6;
    --blue: #6ea8fe; --blue-soft: #16243f; --amber: #e0a44a; --amber-soft: #33270f;
    --purple: #b794f6; --purple-soft: #271a40; --red: #f08080; --red-soft: #3a1a1a;
    --code-bg: #0a0f1a; --code-ink: #d8e2f0; --code-line: #14202f;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
html { scroll-behavior: smooth; overflow-x: hidden; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC",
    "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  background: var(--bg); color: var(--ink); line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
code, .mono { font-family: "SF Mono", "JetBrains Mono", "Fira Code", ui-monospace, Menlo, Consolas, monospace; overflow-wrap: break-word; }

/* ---- top progress bar ---- */
.topbar {
  position: sticky; top: 0; z-index: 50; background: var(--panel);
  border-bottom: 1px solid var(--line); backdrop-filter: blur(8px);
}
.topbar-inner {
  max-width: 960px; margin: 0 auto; padding: .7rem 1.25rem;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.topbar .home { font-size: .82rem; color: var(--muted); font-weight: 600; display:flex; gap:.5rem; align-items:center; }
.topbar .home b { color: var(--accent); }
.topbar .pill { font-size: .72rem; color: var(--muted); background: var(--panel-2);
  padding: .2rem .6rem; border-radius: 999px; border: 1px solid var(--line); white-space: nowrap; }
.progress { height: 3px; background: var(--panel-2); }
.progress > span { display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--blue)); }

.wrap { max-width: 820px; margin: 0 auto; padding: 2.4rem 1.25rem 5rem; }

/* ---- hero ---- */
.hero { margin-bottom: 2rem; }
.hero .part { font-size: .76rem; letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent); font-weight: 700; margin-bottom: .55rem; }
.hero h1 { font-size: 2.05rem; line-height: 1.2; letter-spacing: -.01em; font-weight: 750; }
.hero .lead { margin-top: .9rem; font-size: 1.06rem; color: var(--muted); }

h2 { font-size: 1.32rem; margin: 2.4rem 0 .9rem; letter-spacing: -.01em;
  display: flex; align-items: center; gap: .55rem; }
h2::before { content: ""; width: 4px; height: 1.05em; background: var(--accent); border-radius: 3px; display: inline-block; }
h3 { font-size: 1.05rem; margin: 1.4rem 0 .5rem; }
p { margin: .7rem 0; }
ul, ol { margin: .6rem 0 .6rem 1.3rem; }
li { margin: .3rem 0; }
strong { color: var(--ink); font-weight: 680; }
.inline { background: var(--panel-2); border: 1px solid var(--line); border-radius: 6px;
  padding: .08em .4em; font-size: .9em; color: var(--accent-ink); }

/* ---- callout cards ---- */
.card { border-radius: var(--radius); padding: 1.05rem 1.2rem; margin: 1.2rem 0;
  border: 1px solid var(--line); background: var(--panel); box-shadow: var(--shadow); }
.card .tag { font-size: .72rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  display: inline-flex; align-items: center; gap: .4rem; margin-bottom: .5rem; }
.card.macro { border-left: 4px solid var(--blue); }
.card.macro .tag { color: var(--blue); }
.card.detail { border-left: 4px solid var(--purple); }
.card.detail .tag { color: var(--purple); }
.card.analogy { border-left: 4px solid var(--amber); background: var(--amber-soft); }
.card.analogy .tag { color: var(--amber); }
.card.key { border-left: 4px solid var(--accent); background: var(--accent-soft); }
.card.key .tag { color: var(--accent-ink); }
.card.warn { border-left: 4px solid var(--red); background: var(--red-soft); }
.card.warn .tag { color: var(--red); }
.card.spark { border-left: 4px solid #e0a000;
  background: linear-gradient(100deg, rgba(224,160,0,.12), transparent 70%); }
.card.spark .tag { color: #c98a00; }
@media (prefers-color-scheme: dark) { .card.spark .tag { color: #f0c050; } }

/* ---- code file callout ---- */
.codefile { margin: 1.2rem 0; border-radius: 12px; overflow: hidden; border: 1px solid var(--line);
  box-shadow: var(--shadow); }
.codefile .cf-head { display: flex; align-items: center; gap: .55rem; padding: .5rem .85rem;
  background: var(--panel-2); border-bottom: 1px solid var(--line); font-size: .8rem; }
.codefile .cf-head .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--accent); flex-shrink:0; }
.codefile .cf-head .path { font-family: ui-monospace, monospace; color: var(--ink); font-weight: 600; }
.codefile .cf-head .ln { margin-left: auto; color: var(--faint); font-size: .72rem; }
.codefile pre { background: var(--code-bg); color: var(--code-ink); padding: .9rem 1rem;
  overflow-x: auto; font-size: .82rem; line-height: 1.6; }
.codefile pre .cm { color: #7d8aa3; }
.codefile pre .kw { color: #c792ea; }
.codefile pre .fn { color: #82aaff; }
.codefile pre .st { color: #c3e88d; }
.codefile pre .nb { color: #f78c6c; }

pre.code { background: var(--code-bg); color: var(--code-ink); padding: .9rem 1rem; border-radius: 12px;
  overflow-x: auto; font-size: .83rem; line-height: 1.6; margin: 1.1rem 0; box-shadow: var(--shadow); }
pre.code .cm { color: #7d8aa3; } pre.code .kw { color: #c792ea; }
pre.code .fn { color: #82aaff; } pre.code .st { color: #c3e88d; } pre.code .nb { color: #f78c6c; }

/* ---- collapsible accordion (details/summary) ---- */
.accordion { border: 1px solid var(--line); border-radius: 12px; background: var(--panel);
  margin: .7rem 0; box-shadow: var(--shadow); overflow: hidden; }
.accordion > summary { cursor: pointer; padding: .85rem 1.1rem; font-weight: 650; font-size: .96rem;
  list-style: none; display: flex; align-items: center; gap: .6rem; user-select: none; }
.accordion > summary::-webkit-details-marker { display: none; }
.accordion > summary::after { content: "▶"; font-size: .68rem; color: var(--accent);
  margin-left: auto; transition: transform .15s ease; }
.accordion[open] > summary::after { transform: rotate(90deg); }
.accordion > summary:hover { background: var(--panel-2); }
.accordion[open] > summary { border-bottom: 1px solid var(--line); }
.accordion .badge-num { background: var(--accent-soft); color: var(--accent-ink);
  width: 1.6rem; height: 1.6rem; border-radius: 7px; display: inline-flex; align-items: center;
  justify-content: center; font-size: .82rem; font-weight: 700; flex-shrink: 0; }
.accordion .hint { font-size: .72rem; color: var(--faint); font-weight: 400; }
.acc-body { padding: .9rem 1.1rem 1.1rem; }
.acc-intro { color: var(--muted); font-size: .9rem; margin: .2rem 0 .4rem; }
.qa { margin: 1rem 0; }
.qa:first-child { margin-top: .3rem; }
.qa .q { font-weight: 680; font-size: .9rem; display: flex; gap: .45rem; align-items: center; margin-bottom: .3rem; }
.qa .a { color: var(--muted); font-size: .9rem; }
.qa .a strong { color: var(--ink); }
.qa pre.code { margin: .5rem 0 0; font-size: .78rem; }

/* ---- flow diagram ---- */
.flow { display: flex; align-items: stretch; gap: 0; flex-wrap: wrap; margin: 1.3rem 0;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--radius);
  padding: 1.2rem 1rem; box-shadow: var(--shadow); }
.flow .node { flex: 1 1 0; min-width: 110px; text-align: center; padding: .7rem .5rem;
  border-radius: 10px; background: var(--panel-2); border: 1px solid var(--line); }
.flow .node .nt { font-weight: 700; font-size: .92rem; }
.flow .node .nd { font-size: .76rem; color: var(--muted); margin-top: .2rem; }
.flow .node.hl { background: var(--accent-soft); border-color: var(--accent); }
.flow .arrow { align-self: center; color: var(--faint); font-size: 1.3rem; padding: 0 .35rem; }

/* vertical flow */
.vflow { margin: 1.3rem 0; }
.vflow .step { display: flex; gap: .9rem; position: relative; padding-bottom: 1.1rem; }
.vflow .step:not(:last-child)::before { content:""; position:absolute; left: 15px; top: 34px; bottom: -2px;
  width: 2px; background: var(--line); }
.vflow .num { width: 32px; height: 32px; border-radius: 50%; background: var(--accent); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; flex-shrink: 0; z-index:1; }
.vflow .sc h4 { margin: .25rem 0 .2rem; font-size: 1rem; }
.vflow .sc p { margin: .15rem 0; font-size: .92rem; color: var(--muted); }
.vflow .sc .mono { font-size: .8rem; color: var(--accent-ink); }

/* layered architecture */
.layers { margin: 1.3rem 0; display: flex; flex-direction: column; gap: .55rem; }
.layer { border-radius: 12px; padding: .85rem 1.1rem; border: 1px solid var(--line); background: var(--panel);
  box-shadow: var(--shadow); }
.layer .lh { display: flex; align-items: center; gap: .6rem; }
.layer .lh .badge { font-size: .7rem; font-weight: 700; padding: .12rem .5rem; border-radius: 999px; }
.layer .lh .name { font-weight: 700; font-family: ui-monospace, monospace; }
.layer .ld { font-size: .85rem; color: var(--muted); margin-top: .35rem; }
.layer.l-core { border-left: 4px solid var(--accent); } .layer.l-core .badge { background: var(--accent-soft); color: var(--accent-ink); }
.layer.l-main { border-left: 4px solid var(--blue); } .layer.l-main .badge { background: var(--blue-soft); color: var(--blue); }
.layer.l-part { border-left: 4px solid var(--purple); } .layer.l-part .badge { background: var(--purple-soft); color: var(--purple); }
.layer.l-app { border-left: 4px solid var(--amber); } .layer.l-app .badge { background: var(--amber-soft); color: var(--amber); }

/* two-column compare */
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin: 1.2rem 0; }
@media (max-width: 640px) { .cols { grid-template-columns: 1fr; } }
.col { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 1rem 1.1rem; box-shadow: var(--shadow); min-width: 0; }
.col h4 { margin: 0 0 .4rem; font-size: .95rem; }

table.t { width: 100%; border-collapse: collapse; margin: 1.1rem 0; font-size: .9rem;
  background: var(--panel); border-radius: 12px; overflow: hidden; box-shadow: var(--shadow); }
table.t th, table.t td { padding: .6rem .8rem; text-align: left; border-bottom: 1px solid var(--line); }
table.t th { background: var(--panel-2); font-size: .8rem; letter-spacing: .02em; }
table.t tr:last-child td { border-bottom: none; }
table.t td.mono, table.t td .mono { font-family: ui-monospace, monospace; font-size: .82rem; color: var(--accent-ink); }
@media (max-width: 640px) {
  /* Wide multi-column tables: scroll within their own box instead of
     forcing page-level horizontal overflow (which clipped right columns). */
  table.t { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  table.t th, table.t td { padding: .5rem .6rem; }
}
.selftest { margin: 2.2rem 0 0; border-top: 2px dashed var(--line); padding-top: 1.2rem; }
.selftest > h2 { margin-top: .2rem; }
.quiz { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--blue);
  border-radius: 12px; padding: .9rem 1.1rem; margin: 1rem 0; box-shadow: var(--shadow); }
.quiz .qn { font-weight: 650; }
.quiz ol.opts { list-style: upper-alpha; margin: .55rem 0 .6rem 1.5rem; padding: 0; }
.quiz ol.opts li { margin: .3rem 0; padding-left: .15rem; }
.quiz details.accordion { margin: .5rem 0 0; }
.selftest code { font-family: ui-monospace, monospace; font-size: .9em; color: var(--accent-ink);
  background: var(--accent-soft); padding: 0 .28em; border-radius: 4px; }

/* footer nav */
.footnav { display: flex; justify-content: space-between; gap: 1rem; margin-top: 3rem;
  padding-top: 1.4rem; border-top: 1px solid var(--line); }
.footnav a { flex: 1; padding: .85rem 1.1rem; border-radius: 12px; border: 1px solid var(--line);
  background: var(--panel); box-shadow: var(--shadow); transition: .15s; }
.footnav a:hover { border-color: var(--accent); transform: translateY(-1px); }
.footnav a.next { text-align: right; }
.footnav .dir { font-size: .72rem; color: var(--faint); text-transform: uppercase; letter-spacing: .05em; }
.footnav .ttl { font-weight: 700; color: var(--ink); margin-top: .15rem; }
.footnav a.disabled { opacity: .35; pointer-events: none; }

/* index page */
.toc { display: grid; gap: .7rem; margin-top: 1.6rem; }
.toc-part { font-size: .78rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  color: var(--accent); margin: 1.4rem 0 .2rem; }
.toc a { display: flex; align-items: center; gap: .9rem; padding: .85rem 1.05rem; border-radius: 12px;
  background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); transition: .15s; }
.toc a:hover { border-color: var(--accent); transform: translateX(3px); }
.toc .n { width: 30px; height: 30px; border-radius: 8px; background: var(--accent-soft); color: var(--accent-ink);
  display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: .85rem; flex-shrink: 0; }
.toc .tt { font-weight: 650; color: var(--ink); }
.toc .ts { font-size: .8rem; color: var(--muted); margin-left: auto; text-align: right; }
.toc-search { position: relative; margin: 1.6rem 0 -.4rem; }
.toc-search input { width: 100%; box-sizing: border-box; padding: .75rem 2.8rem .75rem 1rem;
  border-radius: 12px; border: 1px solid var(--line); background: var(--panel); color: var(--ink);
  font-size: .98rem; box-shadow: var(--shadow); }
.toc-search input:focus { outline: none; border-color: var(--accent); }
.toc-search .qcount { position: absolute; right: 1rem; top: 50%; transform: translateY(-50%);
  color: var(--faint); font-size: .8rem; pointer-events: none; }
.toc a.hide, .toc .toc-part.hide { display: none; }
.toc-empty { display: none; color: var(--muted); padding: 1rem; text-align: center; }
.toc-empty.show { display: block; }
.hero.index h1 { font-size: 2.3rem; }
.legend { display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:1rem; font-size:.8rem; color:var(--muted); }
.legend span { display:flex; align-items:center; gap:.4rem; }
.legend i { width:12px; height:12px; border-radius:3px; display:inline-block; }
.pdf-btn { display:inline-flex; align-items:center; gap:.4rem; padding:.55rem 1.1rem;
  background:var(--accent); color:#fff; border-radius:10px; font-size:.9rem; font-weight:650;
  box-shadow:var(--shadow); transition:.15s; }
.pdf-btn:hover { background:var(--accent-ink); transform:translateY(-1px); }
"""

SEARCH_JS = """
(function(){
  var q=document.getElementById('q'); if(!q) return;
  var toc=document.querySelector('.toc');
  var empty=document.getElementById('tocempty');
  var count=document.getElementById('qcount');
  var links=[].slice.call(toc.querySelectorAll('a'));
  var heads=[].slice.call(toc.querySelectorAll('.toc-part'));
  links.forEach(function(a){ a.setAttribute('data-s',(a.textContent||'').toLowerCase()); });
  function run(){
    var t=(q.value||'').toLowerCase().trim(), n=0;
    links.forEach(function(a){
      var hit=!t||a.getAttribute('data-s').indexOf(t)>=0;
      a.classList.toggle('hide',!hit); if(hit)n++;
    });
    heads.forEach(function(h){
      var el=h.nextElementSibling, any=false;
      while(el && !el.classList.contains('toc-part')){
        if(el.tagName==='A' && !el.classList.contains('hide')){any=true;break;}
        el=el.nextElementSibling;
      }
      h.classList.toggle('hide',!any);
    });
    empty.classList.toggle('show', !!t && n===0);
    count.textContent = t ? (n+' \u8bfe') : '';
  }
  q.addEventListener('input',run);
})();
"""

NAV_SCRIPT = """
(function(){
  var onDisk = location.protocol === 'file:';
  document.querySelectorAll('[data-nav]').forEach(function(a){
    var n = a.getAttribute('data-nav');
    a.setAttribute('href', onDisk ? n : '/files/' + n);
  });
})();
"""


def page(filename, content, standalone=False, home_href=None):
    """Wrap lesson content in the full HTML shell with nav.

    When ``standalone`` is True, navigation uses plain relative ``href`` links
    (works via file:// and any static server). Otherwise it uses ``data-nav``
    plus a script that targets the brainstorm companion's ``/files/`` route.

    ``home_href`` overrides the link back to the index (use ``"../index.html"``
    when lesson pages live in a subdirectory). Sibling lesson links always use
    bare filenames, so lessons must share one directory.
    """
    idx = next(i for i, p in enumerate(PAGES) if p[0] == filename)
    fname, title, part = PAGES[idx]
    total = len(PAGES)
    pct = int((idx + 1) / total * 100)
    home = home_href or INDEX_FILE

    prev_link = (
        f'<a class="prev" data-nav="{PAGES[idx-1][0]}"><div class="dir">← 上一课</div>'
        f'<div class="ttl">{PAGES[idx-1][1]}</div></a>'
        if idx > 0 else
        f'<a class="prev" data-nav="{home}"><div class="dir">← 返回</div>'
        f'<div class="ttl">目录</div></a>'
    )
    next_link = (
        f'<a class="next" data-nav="{PAGES[idx+1][0]}"><div class="dir">下一课 →</div>'
        f'<div class="ttl">{PAGES[idx+1][1]}</div></a>'
        if idx + 1 < total else
        f'<a class="next" data-nav="{home}"><div class="dir">完成 →</div>'
        f'<div class="ttl">返回目录</div></a>'
    )

    script_tag = "" if standalone else f"<script>{NAV_SCRIPT}</script>"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{idx+1:02d} · {title} — LangChain 图解教程</title>
{head_meta(f"{idx+1:02d} · {title} — LangChain 图解教程", f"{part}｜{title}：面向新手的 LangChain 图解教程，配真实源码对应、折叠深挖与设计亮点。", og_type="article")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <a class="home" data-nav="{home}">📘 LangChain 图解教程 · <b>目录</b></a>
    <span class="pill">{part}</span>
    <span class="pill">{idx+1:02d} / {total:02d}</span>
  </div>
  <div class="progress"><span style="width:{pct}%"></span></div>
</div>
<div class="wrap">
  <div class="hero">
    <div class="part">{part}</div>
    <h1>{title}</h1>
  </div>
  {content}
  <div class="footnav">{prev_link}{next_link}</div>
</div>
{script_tag}
</body></html>"""
    if standalone:
        html = html.replace('data-nav="', 'href="')
    return html


def index_page(standalone=False, lesson_prefix=""):
    parts = {}
    order = []
    for i, (fname, title, part) in enumerate(PAGES):
        parts.setdefault(part, [])
        if part not in order:
            order.append(part)
        parts[part].append((i + 1, fname, title))

    blocks = []
    subtitles = {
        "01-what-is-langchain.html": "解决什么问题 · 核心心智模型",
        "02-monorepo.html": "core / langchain / partners 三层",
        "03-lifecycle.html": "从你的代码到 LLM 的完整数据流",
        "04-messages.html": "Human / AI / Tool / System 消息",
        "05-chat-models.html": "init_chat_model · invoke / stream / batch",
        "06-tools.html": "@tool 装饰器 · 工具调用",
        "07-agents-intro.html": "create_agent · Agent 循环",
        "08-runnable.html": "invoke/stream/batch · LCEL 管道 |",
        "09-runnable-compose.html": "Sequence / Parallel / Branch 组合",
        "10-output-parsers.html": "StrOutputParser · JsonOutputParser · 闭环",
        "11-chat-internals.html": "BaseChatModel 调用链",
        "12-tool-internals.html": "函数 → JSON Schema → tool_calls",
        "13-agent-internals.html": "LangGraph 状态图 · Send/Command · add_messages reducer",
        "14-streaming-callbacks.html": "流式输出与回调追踪",
        "15-contributing.html": "uv · 测试 · 调试 · 贡献",
        "16-prompts.html": "ChatPromptTemplate · MessagesPlaceholder · few-shot",
        "17-rag.html": "Document → 切块 → Embeddings → VectorStore → Retriever",
        "18-custom-middleware.html": "AgentMiddleware 钩子 · before/after/wrap",
        "19-runtime-context.html": "context_schema · with_fallbacks · stream_mode",
        "20-capstone.html": "把所有零件拼成一个完整可跑的 Agent",
        "21-langchain-vs-autogen.html": "两种范式对照：图/管道 vs 多 Agent 对话",
        "22-ai-stack.html": "Agent 编排 5 流派 · AI 全栈 7 层 · 你在哪",
        "23-learning-map.html": "vLLM/llama.cpp/Ollama · hnswlib/pgvector/Qdrant",
        "24-langgraph-mental-model.html": "为什么 LCEL 不够 · State/Node/Edge/compile",
        "25-langgraph-pregel-engine.html": "Pregel/BSP 超步 · Plan→Execution→Update · channels",
        "26-langgraph-persistence-control.html": "Checkpoint/StateSnapshot · interrupt · Send/Command",
        "27-glossary.html": "全书术语一句话查 + 点链接跳到对应课",
    }
    for part in order:
        blocks.append(f'<div class="toc-part">{part}</div>')
        for num, fname, title in parts[part]:
            sub = subtitles.get(fname, "")
            blocks.append(
                f'<a data-nav="{lesson_prefix}{fname}"><span class="n">{num:02d}</span>'
                f'<span class="tt">{title}</span>'
                f'<span class="ts">{sub}</span></a>'
            )
    toc = "\n".join(blocks)

    script_tag = "" if standalone else f"<script>{NAV_SCRIPT}</script>"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LangChain 图解教程 · 从零理解整个项目</title>
{head_meta("LangChain 图解教程 · 从零理解整个项目", f"从零理解整个 LangChain 项目的图解教程：宏观结构、用户用法、内部源码、自己动手做 Agent，外加横向对比 / AI 全栈坐标系 / LangGraph 引擎深入等番外。{len(order)} 部分 {len(PAGES)} 课，每课配真实代码对应、折叠深挖与设计亮点。", og_type="website")}
<style>{CSS}</style>
</head><body>
<div class="topbar">
  <div class="topbar-inner">
    <span class="home">📘 LangChain 图解教程</span>
    <span class="pill">共 {len(PAGES)} 课 · {len(order)} 个部分</span>
  </div>
  <div class="progress"><span style="width:100%"></span></div>
</div>
<div class="wrap">
  <div class="hero index">
    <div class="part">从零开始 · 面向完全新手</div>
    <h1>用图解理解整个 LangChain 项目</h1>
    <p class="lead">这套教程带你<strong>层层深入</strong>：先建立<strong>宏观全景</strong>，再从<strong>用户视角</strong>学会使用，
    然后深入<strong>内部源码</strong>，接着<strong>自己动手做 Agent</strong>，最后还有<strong>横向对比、AI 全栈坐标系、深入 LangGraph 引擎</strong>等番外。
    每一课都配有真实的代码文件对应，既有宏观理解，也有细节拆解。</p>
    <div class="legend">
      <span><i style="background:var(--blue)"></i>宏观理解</span>
      <span><i style="background:var(--purple)"></i>细节 / 源码</span>
      <span><i style="background:var(--amber)"></i>生活类比</span>
      <span><i style="background:var(--accent)"></i>关键要点</span>
    </div>
    <div style="margin-top:1.1rem">
      <a href="langchain-visual-guide.pdf" class="pdf-btn">📄 下载完整 PDF（全 {len(PAGES)} 课）</a>
    </div>
    <p style="margin:.8rem 0 0;color:var(--faint);font-size:.8rem">📌 对照 <strong>LangChain v1（<span class="mono">langchain_v1</span>）</strong> 与 <strong>LangGraph</strong> 源码讲解 · 最后核验 2026-06 · 源码引用以"文件 + 符号名"为主（行号会随上游更新而变）</p>
  </div>
  <div class="toc-search">
    <input id="q" type="search" placeholder="🔎 搜索课程：标题 / 关键词（如 RAG、Pregel、中间件）" autocomplete="off" aria-label="搜索课程">
    <span class="qcount" id="qcount"></span>
  </div>
  <div class="toc">{toc}</div>
  <div class="toc-empty" id="tocempty">没有匹配的课程，换个关键词试试。</div>
</div>
{script_tag}
<script>{SEARCH_JS}</script>
</body></html>"""
    if standalone:
        html = html.replace('data-nav="', 'href="')
    return html
