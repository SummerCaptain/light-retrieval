#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识库系统 - Web GUI 聊天界面

基于 Flask 的轻量级 Web 聊天界面。
启动: python app.py
访问: http://localhost:8080
"""

import sys, io, os, json, uuid
from datetime import datetime

# 强制行缓冲，确保 print 立刻显示（避免终端无输出）
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

import warnings, logging
warnings.filterwarnings('ignore')

# 只抑制第三方库的冗余日志，保留 Flask/Werkzeug 的启动信息
for _name in ['sentence_transformers', 'flag_embedding', 'urllib3', 'httpx', 'httpcore', 'elasticsearch']:
    logging.getLogger(_name).setLevel(logging.ERROR)

# 先导入核心模块（此时错误要能看到）
try:
    from flask import Flask, render_template_string, request, jsonify
    from core.workflow import run_knowledge_base, reset_session
except Exception as e:
    print(f"[启动失败] {type(e).__name__}: {e}")
    print("请检查: 1) pip install -r requirements.txt  2) 在项目目录下运行")
    sys.exit(1)

print("[OK] 模块加载完成")

app = Flask(__name__)

# 会话存储
sessions = {}


def get_session(session_id: str) -> dict:
    """获取或创建会话"""
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "messages": [],
        }
    return sessions[session_id]


# HTML 模板
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>知识库系统 - RAG</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#f0f2f5;--card:#fff;--primary:#4a90d9;--user:#e8f4fd;--ai:#fff;--border:#e0e0e0;--text:#2c3e50;--muted:#7f8c8d}
body{font-family:-apple-system,"Microsoft YaHei","Segoe UI",sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;flex-direction:column}
.header{background:linear-gradient(135deg,#4a90d9,#357abd);color:#fff;padding:14px 24px;display:flex;align-items:center;gap:12px;box-shadow:0 2px 8px rgba(0,0,0,0.15)}
.header h1{font-size:1.2em;font-weight:600}
.header .badge{background:rgba(255,255,255,0.2);padding:2px 10px;border-radius:12px;font-size:0.75em}
.chat-area{flex:1;overflow-y:auto;padding:20px;max-width:900px;margin:0 auto;width:100%}
.msg{display:flex;margin-bottom:16px;animation:fadeIn 0.3s}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.msg.user{justify-content:flex-end}
.bubble{max-width:75%;padding:12px 16px;border-radius:12px;line-height:1.7;font-size:0.92em;word-break:break-word;white-space:pre-wrap}
.msg.user .bubble{background:var(--primary);color:#fff;border-bottom-right-radius:4px}
.msg.ai .bubble{background:var(--card);border:1px solid var(--border);border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,0.06)}
.msg.ai .bubble .meta{font-size:0.78em;color:var(--muted);margin-top:8px;padding-top:6px;border-top:1px solid #f0f0f0}
.msg.system .bubble{background:#f8f9fa;color:var(--muted);font-size:0.84em;text-align:center;max-width:100%;border-radius:8px;padding:6px 16px}
.typing{display:none;margin-bottom:16px}
.typing .bubble{background:var(--card);border:1px solid var(--border);border-radius:12px;border-bottom-left-radius:4px;padding:12px 20px;font-size:0.88em;color:var(--muted)}
.typing .dot{display:inline-block;animation:blink 1.4s infinite}
.typing .dot:nth-child(2){animation-delay:0.2s}
.typing .dot:nth-child(3){animation-delay:0.4s}
@keyframes blink{0%,80%,100%{opacity:0.3}40%{opacity:1}}
.input-area{background:var(--card);border-top:1px solid var(--border);padding:14px 20px;max-width:900px;margin:0 auto;width:100%}
.input-wrap{display:flex;gap:10px;align-items:flex-end}
#query{flex:1;border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-size:0.92em;font-family:inherit;resize:none;outline:none;min-height:40px;max-height:120px;transition:border-color 0.2s}
#query:focus{border-color:var(--primary)}
#send{background:var(--primary);color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:0.92em;cursor:pointer;font-family:inherit;transition:background 0.2s;white-space:nowrap}
#send:hover{background:#357abd}
#send:disabled{background:#b0c4de;cursor:not-allowed}
.examples{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px}
.examples span{background:#e8f4fd;color:var(--primary);padding:4px 12px;border-radius:14px;font-size:0.82em;cursor:pointer;transition:all 0.2s;border:1px solid transparent}
.examples span:hover{border-color:var(--primary);background:#d0e8f9}
.welcome{text-align:center;padding:60px 20px 30px;color:var(--muted)}
.welcome h2{color:var(--text);font-size:1.4em;margin-bottom:8px}
</style>
</head>
<body>

<div class="header">
  <h1>知识库系统</h1>
  <span class="badge">BM25 + Vector + RRF</span>
  <span class="badge">Qwen3.7-Max</span>
</div>

<div class="chat-area" id="chatArea">
  <div class="welcome">
    <h2>Hi, 有什么可以帮你的?</h2>
    <p>输入问题或点击下方示例开始</p>
  </div>
  <div class="examples">
    <span onclick="ask(this)">什么是ETF？</span>
    <span onclick="ask(this)">降息对债券基金有什么影响？</span>
    <span onclick="ask(this)">基金定投的操作步骤是什么？</span>
    <span onclick="ask(this)">可转债中签后怎么操作？</span>
    <span onclick="ask(this)">ETF和普通指数基金哪个更好？</span>
  </div>
</div>

<div class="typing" id="typing">
  <div class="bubble">思考中<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span></div>
</div>

<div class="input-area">
  <div class="input-wrap">
    <textarea id="query" placeholder="输入问题..." rows="1" onkeydown="handleKey(event)"></textarea>
    <button id="send" onclick="sendQuery()">发送</button>
  </div>
</div>

<script>
const chatArea = document.getElementById('chatArea');
const queryInput = document.getElementById('query');
const sendBtn = document.getElementById('send');
const typingEl = document.getElementById('typing');

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
}

function ask(el) { queryInput.value = el.textContent; sendQuery(); }

function addMsg(role, html) {
  const d = document.createElement('div');
  d.className = 'msg ' + role;
  d.innerHTML = '<div class="bubble">' + html + '</div>';
  chatArea.appendChild(d);
  chatArea.scrollTop = chatArea.scrollHeight;
}

async function sendQuery() {
  const q = queryInput.value.trim();
  if (!q) return;
  queryInput.value = '';
  queryInput.style.height = 'auto';

  // 隐藏欢迎语和示例
  const wl = chatArea.querySelector('.welcome');
  if (wl) wl.remove();
  const el = chatArea.querySelector('.examples');
  if (el) el.remove();

  addMsg('user', escHtml(q));
  typingEl.style.display = 'flex';
  sendBtn.disabled = true;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query: q})
    });
    const data = await res.json();
    typingEl.style.display = 'none';
    sendBtn.disabled = false;

    let metaHtml = '';
    if (data.mode) metaHtml += data.mode + ' | ';
    if (data.doc_count) metaHtml += '检索 ' + data.doc_count + ' 条文档';
    if (data.sources && data.sources.length) metaHtml += ' | 来源: ' + data.sources.join(', ');
    if (data.elapsed) metaHtml += ' | ' + data.elapsed + 's';

    let answerHtml = escHtml(data.answer || '未生成响应');
    if (metaHtml) answerHtml += '<div class="meta">' + metaHtml + '</div>';
    addMsg('ai', answerHtml);
  } catch(e) {
    typingEl.style.display = 'none';
    sendBtn.disabled = false;
    addMsg('ai', '请求失败: ' + e.message);
  }
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
}

// 自动调整输入框高度
queryInput.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});
</script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({"answer": "请输入问题", "mode": "", "doc_count": 0, "sources": [], "elapsed": ""})

    session_id = "web_default"

    try:
        start_time = datetime.now()
        result = run_knowledge_base(query, session_id=session_id)
        end_time = datetime.now()

        mode = result.get("processing_mode", "")
        query_type = result.get("query_type", "")
        mode_text = ""
        if mode == "reactive":
            mode_text = f"直接回答 ({query_type})"
        elif mode == "deliberative":
            mode_text = f"RAG 检索 ({query_type})"

        final_docs = result.get("final_docs") or []
        sources = []
        for doc in final_docs:
            src = doc.get("metadata", {}).get("source_file", "")
            if src and src not in sources:
                sources.append(src)

        elapsed = f"{(end_time - start_time).total_seconds():.2f}"

        return jsonify({
            "answer": result.get("final_response", "未生成响应"),
            "mode": mode_text,
            "doc_count": len(final_docs),
            "sources": sources[:3],
            "elapsed": elapsed,
        })

    except Exception as e:
        return jsonify({"answer": f"处理出错: {str(e)}", "mode": "", "doc_count": 0, "sources": [], "elapsed": ""})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("=" * 55)
    print("  知识库系统 - Web GUI")
    print(f"  访问: http://localhost:{port}")
    print("=" * 55)
    app.run(host='0.0.0.0', port=port, debug=False)
