# -*- coding: utf-8 -*-
"""
VectorRetriever - 基于 BGE Embedding + FAISS 的语义向量检索器

将文档分块编码为向量，构建 FAISS 索引，支持语义相似度检索。
与 BM25Retriever 互补，覆盖同义改写、口语化表达等场景。
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

import faiss

from config.settings import VECTOR_TOP_K, EMBEDDING_DIM
from retrieval.doc_parser import DocParser


class VectorRetriever:
    """向量语义检索器"""

    def __init__(
        self,
        parser: Optional[DocParser] = None,
        embedder=None,
        top_k: int = VECTOR_TOP_K,
        index_dir: Optional[str] = None,
    ):
        """
        初始化向量检索器

        Args:
            parser: DocParser 实例
            embedder: 向量编码器，需实现 encode(texts: List[str]) -> np.ndarray
            top_k: 默认返回数量
            index_dir: FAISS 索引存储目录
        """
        self.parser = parser or DocParser()
        self.embedder = embedder
        self.top_k = top_k
        self.index_dir = Path(index_dir) if index_dir else None

        self._index: Optional[faiss.Index] = None
        self._chunk_registry: List[Dict] = []
        self._dim = self._get_embedding_dim()
        self.total_chunks = 0

    def _get_embedding_dim(self) -> int:
        """获取向量维度"""
        if self.embedder is not None:
            # 用一段测试文本获取实际维度
            test_vec = self.embedder.encode(["test"])
            return test_vec.shape[1]
        return EMBEDDING_DIM

    # ================================================================
    # 公开接口
    # ================================================================

    def build_index(self, dir_path: str):
        """
        解析目录下所有文档并构建 FAISS 向量索引

        Args:
            dir_path: 知识库文档目录路径
        """
        results = self.parser.parse_directory(dir_path)

        self._chunk_registry = []
        all_texts = []
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
                all_texts.append(chunk_text)
                self.total_chunks += 1

        if not all_texts:
            return

        # 编码所有文本为向量
        vectors = self._encode(all_texts)

        # 构建 FAISS 索引（内积搜索，等价于余弦相似度）
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vectors)

    def add_document(self, file_path: str):
        """
        添加单个文档到向量索引

        Args:
            file_path: 文档路径
        """
        result = self.parser.parse_file(file_path)

        new_texts = []
        for chunk_idx, chunk_text in enumerate(result.chunks):
            if not chunk_text.strip():
                continue
            self._chunk_registry.append({
                "doc_id": result.doc_hash,
                "chunk_id": chunk_idx,
                "content": chunk_text,
                "source_file": result.source_file,
            })
            new_texts.append(chunk_text)

        if not new_texts:
            return

        self.total_chunks = len(self._chunk_registry)
        new_vectors = self._encode(new_texts)

        if self._index is None:
            self._index = faiss.IndexFlatIP(self._dim)
            self._index.add(new_vectors)
        else:
            self._index.add(new_vectors)

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        向量语义检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            List[RetrievedDocument]: 按相似度降序排列
        """
        if top_k is None:
            top_k = self.top_k

        if not query or not query.strip():
            return []

        if self._index is None or self.total_chunks == 0:
            return []

        # 编码查询向量
        query_vec = self._encode([query])
        top_k = min(top_k, self.total_chunks)

        # FAISS 搜索
        scores, indices = self._index.search(query_vec, top_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._chunk_registry):
                continue
            registry = self._chunk_registry[idx]
            results.append({
                "doc_id": registry["doc_id"],
                "chunk_id": registry["chunk_id"],
                "content": registry["content"],
                "score": float(score),
                "source": "vector",
                "metadata": {
                    "source_file": registry["source_file"],
                },
            })

        return results

    def save_index(self):
        """保存 FAISS 索引和分块注册表到磁盘"""
        if self._index is None:
            return

        if self.index_dir is None:
            raise ValueError("未设置 index_dir，无法保存索引")

        self.index_dir.mkdir(parents=True, exist_ok=True)

        # 保存 FAISS 索引
        faiss.write_index(self._index, str(self.index_dir / "vector.index"))

        # 保存分块注册表
        registry_path = self.index_dir / "registry.json"
        registry_path.write_text(
            json.dumps(self._chunk_registry, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_index(self):
        """从磁盘加载 FAISS 索引和分块注册表"""
        if self.index_dir is None:
            raise ValueError("未设置 index_dir，无法加载索引")

        index_path = self.index_dir / "vector.index"
        registry_path = self.index_dir / "registry.json"

        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        # 加载 FAISS 索引
        self._index = faiss.read_index(str(index_path))

        # 加载分块注册表
        if registry_path.exists():
            self._chunk_registry = json.loads(
                registry_path.read_text(encoding="utf-8")
            )
            self.total_chunks = len(self._chunk_registry)

    # ================================================================
    # 内部方法
    # ================================================================

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表编码为向量矩阵

        Args:
            texts: 文本列表

        Returns:
            np.ndarray: shape=(len(texts), dim)，已 L2 归一化
        """
        if self.embedder is None:
            raise RuntimeError("未设置 embedder，无法编码文本")

        vectors = self.embedder.encode(texts)
        vectors = np.array(vectors, dtype=np.float32)

        # L2 归一化，使内积等价于余弦相似度
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        vectors = vectors / norms

        return vectors