# -*- coding: utf-8 -*-
"""
检索参数网格搜索

对 RRF_K / BM25_TOP_K / VECTOR_TOP_K 做网格搜索，在融合阶段（重排序前）
用 Golden Dataset（60 题）评估 Hit Rate@K / MRR，选出最优检索参数。

相关性锚点采用「来源文件 + 关键词」（见 evaluation/metrics.py），
与内容 MD5 无关，因此后续调整 CHUNK_SIZE 也不会使评估失效。

用法:
  python scripts/tune_retrieval_params.py
  python scripts/tune_retrieval_params.py --k 5

说明:
  - 需要本地已下载 BGE 向量模型（bge-large-zh-v1.5）
  - 不调用 LLM，不依赖 DASHSCOPE_API_KEY
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

from sentence_transformers import SentenceTransformer

from config.settings import (
    KNOWLEDGE_BASE_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
)
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from evaluation.metrics import hit_rate, mrr


def vector_search_by_vec(vector: VectorRetriever, query_vec, top_k: int) -> list:
    """用预计算的查询向量做 FAISS 检索（避免每次组合重复编码查询）"""
    top_k = min(top_k, vector.total_chunks)
    query_vec = query_vec.reshape(1, -1)  # FAISS 要求二维 (1, dim)
    scores, indices = vector._index.search(query_vec, top_k)
    scores, indices = scores[0], indices[0]

    results = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(vector._chunk_registry):
            continue
        reg = vector._chunk_registry[idx]
        results.append({
            "doc_id": reg["doc_id"],
            "chunk_id": reg["chunk_id"],
            "content": reg["content"],
            "score": float(score),
            "source": "vector",
            "metadata": {"source_file": reg["source_file"]},
        })
    return results


def load_questions() -> list:
    """加载 Golden Dataset 的题目列表"""
    dataset = json.loads(
        (PROJECT_ROOT / "evaluation" / "golden_dataset.json").read_text(encoding="utf-8")
    )
    return dataset["questions"]


def evaluate_combo(bm25, vector, hybrid, query_vecs, questions, rrf_k, bm25_top_k, vector_top_k, k):
    """对单个参数组合评估平均 Hit Rate@K 与 MRR"""
    hybrid.rrf_k = rrf_k
    bm25.top_k = bm25_top_k

    hit_sum = 0.0
    mrr_sum = 0.0
    for q, qvec in zip(questions, query_vecs):
        bm25_results = bm25.search(q["query"])
        vector_results = vector_search_by_vec(vector, qvec, vector_top_k)
        fused = hybrid._rrf_fusion(bm25_results, vector_results)
        fused.sort(key=lambda x: x["score"], reverse=True)
        top_docs = fused[:20]
        hit_sum += hit_rate(top_docs, q["relevant_source_files"], q["expected_keywords"], k=k)
        mrr_sum += mrr(top_docs, q["relevant_source_files"], q["expected_keywords"])

    n = len(questions)
    return hit_sum / n, mrr_sum / n


def main():
    parser = argparse.ArgumentParser(description="检索参数网格搜索")
    parser.add_argument("--k", type=int, default=5, help="Hit Rate 的 Top-K（默认 5）")
    args = parser.parse_args()

    rrf_grid = [20, 40, 60, 80, 100]
    bm25_grid = [10, 20, 30]
    vector_grid = [10, 20, 30]

    print("=" * 70)
    print("  检索参数网格搜索（融合阶段评估，重排序前）")
    print(f"  Hit Rate@{args.k} / MRR | 题目数: 60")
    print(f"  RRF_K ∈ {rrf_grid}  BM25_TOP_K ∈ {bm25_grid}  VECTOR_TOP_K ∈ {vector_grid}")
    print("=" * 70)

    # 构建索引（只构建一次，检索阶段复用）
    print("\n[1/3] 构建 BM25 + 向量索引...")
    bm25 = BM25Retriever()
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
    vector = VectorRetriever(embedder=embedder)
    kb_dir = str(KNOWLEDGE_BASE_DIR)
    bm25.build_index(kb_dir)
    vector.build_index(kb_dir)
    hybrid = HybridRetriever(bm25, vector)
    print(f"  BM25 chunks: {bm25.total_chunks} | Vector chunks: {vector.total_chunks}")

    questions = load_questions()

    # 预计算所有查询向量（一次性批量编码，网格搜索中复用）
    print("\n[2/3] 预计算查询向量...")
    query_vecs = vector._encode([q["query"] for q in questions])
    print(f"  已编码 {len(query_vecs)} 条查询")

    # 网格搜索
    print(f"\n[3/3] 网格搜索 {len(rrf_grid) * len(bm25_grid) * len(vector_grid)} 组参数...")
    results = []
    for rrf_k, bm25_top_k, vector_top_k in itertools.product(rrf_grid, bm25_grid, vector_grid):
        hr, mr = evaluate_combo(
            bm25, vector, hybrid, query_vecs, questions,
            rrf_k, bm25_top_k, vector_top_k, args.k,
        )
        results.append((hr, mr, rrf_k, bm25_top_k, vector_top_k))
        print(f"  RRF_K={rrf_k:<4} BM25={bm25_top_k:<3} VEC={vector_top_k:<3} "
              f"→ HR@{args.k}={hr:.4f} MRR={mr:.4f}", flush=True)

    # 按 Hit Rate 降序，MRR 次之
    results.sort(key=lambda x: (-x[0], -x[1]))

    print("\nTop-10 参数组合（按 Hit Rate 降序）:")
    print("-" * 70)
    for i, (hr, mr, rrf_k, bm25_top_k, vector_top_k) in enumerate(results[:10], 1):
        print(f"  {i:>2}. RRF_K={rrf_k:<4} BM25_TOP_K={bm25_top_k:<3} "
              f"VECTOR_TOP_K={vector_top_k:<3} → HR@{args.k}={hr:.4f} MRR={mr:.4f}")

    best = results[0]
    print("\n推荐参数（写入 config/settings.py）:")
    print(f"  RRF_K = {best[2]}")
    print(f"  BM25_TOP_K = {best[3]}")
    print(f"  VECTOR_TOP_K = {best[4]}")


if __name__ == "__main__":
    main()
