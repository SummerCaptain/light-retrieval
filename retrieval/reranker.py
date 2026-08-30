# -*- coding: utf-8 -*-
"""
Reranker - BGE-Reranker 重排序模块

使用 BAAI/bge-reranker-large 交叉编码器对混合检索的候选文档进行精细重排序，
基于查询与文档的语义相关性打分。

直接基于 transformers 的 AutoModelForSequenceClassification 实现交叉编码打分，
不依赖 FlagEmbedding（其 compute_score 依赖的 tokenizer.prepare_for_model
已被新版 transformers 移除，存在版本不兼容）。
"""

from typing import Dict, List

import torch

from config.settings import RERANKER_MODEL_NAME, RERANKER_DEVICE


class BGEReranker:
    """
    BGE-Reranker 交叉编码器重排序器

    使用 BAAI/bge-reranker-large 对查询-文档对进行精细打分。
    首次使用会加载本地缓存的模型（约 2.2GB）。
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str = RERANKER_DEVICE,
        max_length: int = 512,
    ):
        """
        初始化 BGE-Reranker

        Args:
            model_name: 模型名称
            device: 运行设备 cpu/cuda
            max_length: 查询-文档对的最大截断长度
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型与分词器"""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()

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

        # 构建查询-文档对并打分
        pairs = [[query, doc.get("content", "")] for doc in documents]
        scores = self._score(pairs)

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

    def _score(self, pairs: List[List[str]]) -> List[float]:
        """对查询-文档对批量交叉编码打分"""
        queries = [p[0] for p in pairs]
        passages = [p[1] for p in pairs]

        with torch.no_grad():
            # 交叉编码：<s> query </s></s> passage </s>，超长时从尾部（passage 侧）截断
            inputs = self._tokenizer(
                queries,
                passages,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**inputs).logits.view(-1).float()

        return logits.cpu().tolist()


def create_reranker() -> BGEReranker:
    """工厂函数：创建 BGE-Reranker 重排序器"""
    return BGEReranker()
