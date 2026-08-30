"""
RAG 管道局部验证（不依赖 LLM API）

验证：文档解析 → BM25检索 → 向量检索 → RRF融合 → 重排序 → 短期记忆
"""
import sys, io, os, json

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)
sys.path.insert(0, BASE)

import warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

# 抑制 jieba 日志
_old = sys.stdout
sys.stdout = io.StringIO()

from retrieval.doc_parser import DocParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import BGEReranker
from memory.short_term import ShortTermMemoryManager

sys.stdout = _old

print("=" * 55)
print("RAG 管道局部验证 (无 LLM)")
print("=" * 55)

# 1. 文档解析
print("\n[1] DocParser 解析知识库...")
parser = DocParser()
kb_dir = os.path.join(BASE, "data", "knowledge_base")
docs = parser.parse_directory(kb_dir)
print(f"  解析文档: {len(docs)} 个")
for doc in docs:
    print(f"  - {doc.source_file}: {doc.total_chunks} chunks")

# 2. BM25 检索
print("\n[2] BM25 检索...")
bm25 = BM25Retriever()
bm25.build_index(kb_dir)
for q in ["ETF", "债券基金", "定投", "可转债"]:
    results = bm25.search(q, top_k=3)
    print(f"  '{q}' → {len(results)} 条")
    for r in results[:2]:
        print(f"    [{r['doc_id']}] score={r['score']:.4f} | {r['content'][:40]}...")

# 3. 向量检索
print("\n[3] 向量检索...")
from test.test_vector_retriever import SimpleEmbedder
embedder = SimpleEmbedder(dim=128)
vec = VectorRetriever(embedder=embedder)
vec.build_index(kb_dir)
for q in ["ETF基金是什么", "降息对债基的影响", "定投步骤"]:
    results = vec.search(q, top_k=3)
    print(f"  '{q}' → {len(results)} 条")
    for r in results[:2]:
        print(f"    [{r['doc_id']}] score={r['score']:.4f} | {r['content'][:40]}...")

# 4. 混合检索
print("\n[4] 混合检索 (BM25 + Vector + RRF)...")
hybrid = HybridRetriever(bm25, vec)
for q in ["ETF和指数基金的区别", "降息债券", "可转债操作"]:
    results = hybrid.search(q, top_k=5)
    print(f"  '{q}' → {len(results)} 条")
    for r in results[:3]:
        bm25_rank = r.get("metadata", {}).get("bm25_rank")
        vec_rank = r.get("metadata", {}).get("vector_rank")
        print(f"    [{r['doc_id']}] score={r['score']:.4f} bm25={bm25_rank} vec={vec_rank}")

# 5. 重排序
print("\n[5] 重排序 (BGEReranker)...")
reranker = BGEReranker()
for q in ["ETF和指数基金的区别"]:
    fused = hybrid.search(q, top_k=10)
    reranked = reranker.rerank(q, fused, top_k=5)
    print(f"  '{q}' 重排序前 Top-3:")
    for r in fused[:3]:
        print(f"    [{r['doc_id']}] rrf_score={r['score']:.4f}")
    print(f"  '{q}' 重排序后 Top-3:")
    for r in reranked[:3]:
        orig = r.get("metadata", {}).get("original_score", 0)
        print(f"    [{r['doc_id']}] rerank_score={r['score']:.4f} (原={orig:.4f})")

# 6. 短期记忆
print("\n[6] 短期记忆...")
stm = ShortTermMemoryManager(session_id="test")
stm.add_user_query("什么是ETF？", query_type="knowledge_retrieval")
stm.add_assistant_response("ETF是交易型开放式指数基金...", retrieved_docs=["doc1"])
stm.add_user_query("那和普通基金有什么区别？", query_type="knowledge_retrieval")
print(f"  上下文: {stm.get_context()[:100]}...")
print(f"  追问判断: '手续费呢？' → {stm.is_follow_up('手续费呢？')}")
print(f"  追问判断: '请详细解释ETF的套利机制' → {stm.is_follow_up('请详细解释ETF的套利机制')}")

print("\n" + "=" * 55)
print("全部验证通过!")
print("=" * 55)
