import sys, io, os

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

# 把 jieba 日志和 stdout 全部抑制
_old_out = sys.stdout
_old_err = sys.stderr
sys.stderr = io.StringIO()
sys.stdout = io.StringIO()

from core.workflow import run_knowledge_base, reset_session

# 恢复 stdout/stderr（jieba 已加载完）
sys.stdout = _old_out
sys.stderr = _old_err

QUERIES = [
    "什么是ETF？",
    "降息对债券基金有什么影响？",
    "基金定投的操作步骤是什么？",
    "可转债中签后怎么操作？",
]

results = []
for q in QUERIES:
    reset_session("e2e")
    try:
        r = run_knowledge_base(q, session_id="e2e")
        results.append({
            "query": q,
            "type": r.get("query_type", "?"),
            "mode": r.get("processing_mode", "?"),
            "doc_count": len(r.get("final_docs") or []),
            "top_docs": [
                {"file": d.get("metadata", {}).get("source_file", "?"), "score": round(d.get("score", 0), 4)}
                for d in (r.get("final_docs") or [])[:3]
            ],
            "answer": (r.get("final_response") or "N/A")[:500],
            "error": None,
        })
    except Exception as e:
        results.append({"query": q, "error": str(e)})

import json
out_path = os.path.join(os.path.expanduser("~"), "rag_result.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"SAVED to {out_path}")
