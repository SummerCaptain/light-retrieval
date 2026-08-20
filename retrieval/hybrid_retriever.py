# -*- coding: utf-8 -*-
"""
HybridRetriever - 混合检索策略 (BM25 + Vector + RRF 融合)

实现 BM25 关键词检索与向量语义检索的 RRF (Reciprocal Rank Fusion) 加权融合，
结合两路检索的优势：BM25 精确匹配专有名词，Vector 覆盖同义改写和口语化表达。
"""

from typing import Dict, List, Optional

from config.settings import RRF_K
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever


class HybridRetriever:
    """混合检索器 - BM25 + Vector 加权融合"""

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = RRF_K,
        reranker=None,
    ):
        """
        初始化混合检索器

        Args:
            bm25_retriever: BM25 关键词检索器
            vector_retriever: 向量语义检索器
            rrf_k: RRF 算法的平滑参数 k（默认 60）
            reranker: 可选的重排序模型（如 BGE-Reranker）
        """
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k
        self.reranker = reranker

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        混合检索：BM25 + Vector 两路召回 → RRF 融合 → 排序

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List[RetrievedDocument]: 按 RRF 得分降序排列
        """
        if not query or not query.strip():
            return []

        # 获取两路检索结果（使用配置的 top_k 作为候选数）
        bm25_results = self.bm25.search(query)
        vector_results = self.vector.search(query)

        # 如果两路都没有结果
        if not bm25_results and not vector_results:
            return []

        # RRF 融合
        fused = self._rrf_fusion(bm25_results, vector_results)

        # 排序：按 RRF 得分降序
        fused.sort(key=lambda x: x["score"], reverse=True)

        return fused[:top_k]

    # ================================================================
    # 内部方法
    # ================================================================

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
    ) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 加权融合

        公式: score(d) = sum_{r in rankers} 1 / (k + rank_r(d))
        其中 rank_r(d) 是文档 d 在检索器 r 中的排名（1-indexed），
        k 是平滑参数（默认 60），用于降低高排名文档的权重优势。

        Args:
            bm25_results: BM25 检索结果
            vector_results: 向量检索结果

        Returns:
            List[RetrievedDocument]: 融合后的结果列表
        """
        rrf_scores: Dict[str, float] = {}       # key -> RRF 得分
        doc_map: Dict[str, Dict] = {}            # key -> 文档内容

        def _doc_key(doc: Dict) -> str:
            """生成文档唯一标识"""
            return f"{doc['doc_id']}_{doc['chunk_id']}"

        # 处理 BM25 结果
        for rank, doc in enumerate(bm25_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        # 处理 Vector 结果
        for rank, doc in enumerate(vector_results, start=1):
            key = _doc_key(doc)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
            if key not in doc_map:
                doc_map[key] = doc

        # 构建融合结果
        fused = []
        for key, score in rrf_scores.items():
            doc = doc_map[key]
            fused.append({
                "doc_id": doc["doc_id"],
                "chunk_id": doc["chunk_id"],
                "content": doc["content"],
                "score": score,
                "source": "hybrid",
                "metadata": {
                    "source_file": doc["metadata"].get("source_file", ""),
                    "bm25_rank": self._get_rank(bm25_results, key),
                    "vector_rank": self._get_rank(vector_results, key),
                },
            })

        return fused

    def _get_rank(self, results: List[Dict], key: str) -> Optional[int]:
        """获取文档在结果列表中的排名（1-indexed），未出现返回 None"""
        for rank, doc in enumerate(results, start=1):
            if f"{doc['doc_id']}_{doc['chunk_id']}" == key:
                return rank
        return None