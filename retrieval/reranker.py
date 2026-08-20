# -*- coding: utf-8 -*-
"""
Reranker - BGE-Reranker 重排序模块

对混合检索的候选文档进行精细重排序，提升 Top-K 的相关性。
支持两种模式：
1. 生产模式：使用 BAAI/bge-reranker-large 模型
2. 降级模式：基于关键词重叠的简单重排序（无需 GPU/模型下载）
"""

from typing import Dict, List, Optional


class SimpleReranker:
    """
    简单重排序器（降级方案）

    基于查询与文档的关键词重叠度进行重排序，
    无需下载模型，适用于测试和轻量部署场景。
    """

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        基于关键词重叠度重排序

        Args:
            query: 查询文本
            documents: 候选文档列表（RetrievedDocument 格式）
            top_k: 返回数量

        Returns:
            重排序后的文档列表
        """
        if not query or not documents:
            return []

        # 对查询进行字符级分词（简化版，适配中文）
        query_chars = set(query)

        scored_docs = []
        for doc in documents:
            content = doc.get("content", "")
            # 计算字符重叠率
            content_chars = set(content)
            overlap = len(query_chars & content_chars)
            total = max(len(query_chars), 1)
            overlap_score = overlap / total

            # 保留原始 RRF 得分作为权重
            original_score = doc.get("score", 0.0)

            # 综合得分 = 原始得分权重(0.6) + 重叠率权重(0.4)
            combined_score = 0.6 * original_score + 0.4 * overlap_score

            reranked_doc = dict(doc)
            reranked_doc["score"] = combined_score
            reranked_doc["source"] = "reranked"
            reranked_doc["metadata"] = dict(doc.get("metadata", {}))
            reranked_doc["metadata"]["original_score"] = original_score
            reranked_doc["metadata"]["reranker"] = "simple"
            scored_docs.append(reranked_doc)

        # 按综合得分降序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


class BGEReranker:
    """
    BGE-Reranker 重排序器（生产模式）

    使用 BAAI/bge-reranker-large 交叉编码器对查询-文档对进行精细打分。
    首次使用会自动下载模型（约 1.3GB）。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"):
        """
        初始化 BGE-Reranker

        Args:
            model_name: 模型名称
            device: 运行设备 cpu/cuda
        """
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from FlagEmbedding import FlagReranker
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.device == "cuda",
            )

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        使用 BGE-Reranker 对文档重排序

        Args:
            query: 查询文本
            documents: 候选文档列表
            top_k: 返回数量

        Returns:
            重排序后的文档列表
        """
        if not query or not documents:
            return []

        self._load_model()

        # 构建查询-文档对
        pairs = [[query, doc.get("content", "")] for doc in documents]

        # 获取相关性分数
        scores = self._model.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # 组装结果
        scored_docs = []
        for doc, score in zip(documents, scores):
            reranked_doc = dict(doc)
            reranked_doc["score"] = float(score)
            reranked_doc["source"] = "reranked"
            reranked_doc["metadata"] = dict(doc.get("metadata", {}))
            reranked_doc["metadata"]["original_score"] = doc.get("score", 0.0)
            reranked_doc["metadata"]["reranker"] = "bge"
            scored_docs.append(reranked_doc)

        # 按相关性分数降序
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]


def create_reranker(use_bge: bool = False, device: str = "cpu") -> object:
    """
    工厂函数：创建重排序器

    Args:
        use_bge: 是否使用 BGE 模型（False 则使用 SimpleReranker）
        device: 运行设备

    Returns:
        重排序器实例
    """
    if use_bge:
        return BGEReranker(device=device)
    return SimpleReranker()
