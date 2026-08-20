# -*- coding: utf-8 -*-
"""
BM25Retriever - 基于 BM25 算法的关键词检索器

与 DocParser 协作，将文档分块构建为 BM25 倒排索引，
支持关键词精确匹配检索，解决专有名词召回率低的问题。
"""

import jieba
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from config.settings import BM25_TOP_K
from retrieval.doc_parser import DocParser, DocParseResult


class BM25Retriever:
    """BM25 关键词检索器"""

    def __init__(
        self,
        parser: Optional[DocParser] = None,
        top_k: int = BM25_TOP_K,
    ):
        """
        初始化 BM25 检索器

        Args:
            parser: DocParser 实例，用于解析文档
            top_k: 默认返回的文档数量
        """
        self.parser = parser or DocParser()
        self.top_k = top_k

        self._bm25_model: Optional[BM25Okapi] = None
        self._chunk_registry: List[Dict] = []  # 分块注册表，记录 doc_hash/chunk_id
        self._tokenized_corpus: List[List[str]] = []  # 分词后的文档集合
        self.total_chunks = 0

    # ================================================================
    # 公开接口
    # ================================================================

    def build_index(self, dir_path: str):
        """
        解析目录下所有文档并构建 BM25 索引

        Args:
            dir_path: 知识库文档目录路径
        """
        results = self.parser.parse_directory(dir_path)

        self._chunk_registry = []
        self._tokenized_corpus = []
        self.total_chunks = 0

        for result in results:
            for chunk_idx, chunk_text in enumerate(result.chunks):
                if not chunk_text.strip():
                    continue
                self._chunk_registry.append({
                    "doc_id": result.doc_hash,
                    "chunk_id": chunk_idx,
                    "content": chunk_text,
                    "source_file": result.source_file,
                })
                self._tokenized_corpus.append(self._tokenize(chunk_text))
                self.total_chunks += 1

        if self._tokenized_corpus:
            self._bm25_model = BM25Okapi(self._tokenized_corpus)

    def add_document(self, file_path: str):
        """
        添加单个文档到索引（增量更新）

        Args:
            file_path: 文档路径
        """
        result = self.parser.parse_file(file_path)

        new_chunks = []
        new_tokenized = []

        for chunk_idx, chunk_text in enumerate(result.chunks):
            if not chunk_text.strip():
                continue
            new_chunks.append({
                "doc_id": result.doc_hash,
                "chunk_id": chunk_idx,
                "content": chunk_text,
                "source_file": result.source_file,
            })
            new_tokenized.append(self._tokenize(chunk_text))

        if not new_chunks:
            return

        # 追加到现有注册表
        self._chunk_registry.extend(new_chunks)
        self._tokenized_corpus.extend(new_tokenized)
        self.total_chunks = len(self._chunk_registry)

        # 重建 BM25 索引（rank_bm25 不支持增量更新，需重建）
        self._bm25_model = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量，默认使用初始化时的 top_k

        Returns:
            List[RetrievedDocument]: 按 BM25 得分降序排列的检索结果
        """
        if top_k is None:
            top_k = self.top_k

        if not query or not query.strip():
            return []

        if self._bm25_model is None or self.total_chunks == 0:
            return []

        # 分词查询
        tokenized_query = self._tokenize(query)

        # BM25 检索
        scores = self._bm25_model.get_scores(tokenized_query)
        top_k = min(top_k, len(scores))

        # 获取 Top-K 索引（按得分降序）
        if top_k == 0:
            return []

        # 按得分降序排序，取 top_k 个索引
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            registry = self._chunk_registry[idx]
            results.append({
                "doc_id": registry["doc_id"],
                "chunk_id": registry["chunk_id"],
                "content": registry["content"],
                "score": float(score),
                "source": "bm25",
                "metadata": {
                    "source_file": registry["source_file"],
                },
            })

        return results

    # ================================================================
    # 内部方法
    # ================================================================

    def _tokenize(self, text: str) -> List[str]:
        """
        中文分词（使用 jieba 精确模式）

        Args:
            text: 待分词文本

        Returns:
            List[str]: 分词后的词列表
        """
        # 使用 jieba 精确模式分词，过滤空字符串
        tokens = [w for w in jieba.cut(text) if w.strip()]
        return tokens