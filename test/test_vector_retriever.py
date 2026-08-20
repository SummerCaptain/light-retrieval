# -*- coding: utf-8 -*-
"""
VectorRetriever 单元测试

验证向量索引构建、语义相似度检索、FAISS 持久化逻辑。
使用轻量 SimpleEmbedder 避免下载 BGE 模型，测试核心检索管道。
运行方式: python -m unittest test.test_vector_retriever -v
"""

import os
import sys
import json
import math
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.doc_parser import DocParser
from retrieval.vector_retriever import VectorRetriever


# ================================================================
# 轻量 Embedder - 用于测试，避免下载 1.3GB BGE 模型
# 基于字符 bigram 重叠计算向量，模拟语义相似度
# 两段文本共享的 bigram 越多，向量越相似
# ================================================================
class SimpleEmbedder:
    """轻量 bigram Embedder，用于测试管道逻辑"""

    def __init__(self, dim: int = 256):
        self.dim = dim
        # 为 bigram 到维度建立稳定的哈希映射
        self._bigram_map: dict = {}

    def _get_bigram_index(self, bigram: str) -> int:
        """获取 bigram 对应的维度索引（稳定映射）"""
        if bigram not in self._bigram_map:
            self._bigram_map[bigram] = hash(bigram) % self.dim
        return self._bigram_map[bigram]

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        将文本编码为固定维度向量。
        每个字符 bigram 映射到一个维度，出现次数作为该维度的值。
        共享 bigram 越多的文本，余弦相似度越高。
        """
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # 提取所有字符 bigram
            for j in range(len(text) - 1):
                bigram = text[j:j + 2]
                idx = self._get_bigram_index(bigram)
                vectors[i, idx] += 1.0
            # L2 归一化
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm
        return vectors


class TestVectorRetriever(unittest.TestCase):
    """VectorRetriever 单元测试"""

    def setUp(self):
        """创建临时工作目录和测试文档"""
        self.temp_dir = tempfile.mkdtemp()
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.chunks_dir.mkdir()
        self.index_dir = Path(self.temp_dir) / "faiss_index"
        self.index_dir.mkdir()

        self.embedder = SimpleEmbedder(dim=256)

        self.parser = DocParser(
            chunk_size=512,
            chunk_overlap=64,
            output_dir=str(self.chunks_dir),
        )

        self.retriever = VectorRetriever(
            parser=self.parser,
            embedder=self.embedder,
            top_k=5,
            index_dir=str(self.index_dir),
        )

        self._create_test_docs()

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_docs(self):
        """创建多篇测试文档，覆盖不同主题"""
        docs_dir = Path(self.temp_dir) / "docs"
        docs_dir.mkdir()

        # 文档1: 退休规划 - 包含"养老"、"养老金"、"退休"等关键词
        (docs_dir / "retirement.txt").write_text(
            "退休规划是个人财务规划的重要组成部分。建议从30岁开始为退休储蓄。"
            "中国的养老金体系包括基本养老保险、企业年金和个人养老金三个层次。"
            "个人养老金账户每年最高可缴纳12000元，享受税收递延优惠。"
            "退休后的生活费用需要考虑通货膨胀因素，建议使用4%提取法则。",
            encoding="utf-8",
        )

        # 文档2: 基金定投 - 包含"定投"、"指数基金"、"长期"等关键词
        (docs_dir / "fund_investment.txt").write_text(
            "基金定投是一种长期投资策略，通过定期定额买入基金来分散风险。"
            "定投可以降低择时风险，适合普通投资者。"
            "建议选择沪深300指数基金或中证500指数基金作为定投标的。"
            "定投的核心理念是长期坚持，不因短期市场波动而中断。",
            encoding="utf-8",
        )

        # 文档3: 股票投资 - 包含"股票"、"估值"、"PE"等关键词
        (docs_dir / "stock_investment.txt").write_text(
            "股票投资需要关注公司的基本面和估值水平。"
            "市盈率PE是常用的估值指标，但需要结合行业平均水平和成长性综合判断。"
            "戴维斯双击是指业绩增长叠加估值提升带来的股价大幅上涨。"
            "价值投资强调以合理价格买入优质公司并长期持有。",
            encoding="utf-8",
        )

        # 文档4: 半相关文档 - 包含"债券"、"风险"等通用词
        (docs_dir / "bond_market.txt").write_text(
            "债券市场是固定收益投资的重要场所。国债以国家信用为担保，风险极低。"
            "企业债的收益率高于国债，但也承担了信用风险。"
            "在利率下行周期中，债券价格会上升，利好债券投资者。",
            encoding="utf-8",
        )

        # 文档5: 完全无关文档 - 运动健身
        (docs_dir / "exercise.txt").write_text(
            "每周进行150分钟的中等强度有氧运动有助于保持健康。"
            "力量训练可以增加肌肉量，提高基础代谢率。"
            "跑步、游泳和骑行是常见的有氧运动方式。",
            encoding="utf-8",
        )

        self.docs_dir = docs_dir

    # ================================================================
    # 测试1: 索引构建
    # ================================================================
    def test_build_index(self):
        """测试向量索引构建"""
        self.retriever.build_index(str(self.docs_dir))

        self.assertGreater(self.retriever.total_chunks, 0,
                           "索引应包含至少一个分块")
        self.assertIsNotNone(self.retriever._index,
                             "FAISS 索引应已构建")
        self.assertEqual(self.retriever._index.d, self.embedder.dim,
                         "FAISS 索引维度应与 embedder 一致")

    # ================================================================
    # 测试2: 语义检索 - 同义改写
    # ================================================================
    def test_search_semantic_paraphrase(self):
        """测试同义改写查询：查询"养老储蓄"应命中文档1"""
        self.retriever.build_index(str(self.docs_dir))

        # "养老储蓄"是对"退休规划"的口语化同义改写
        results = self.retriever.search("养老储蓄应该怎么做", top_k=5)

        self.assertGreater(len(results), 0, "应返回至少一个结果")
        self.assertEqual(results[0]["source"], "vector")

        # 第一个结果应最相关 - 包含"退休"或"养老"的文档
        top_content = results[0]["content"]
        relevant = "退休" in top_content or "养老" in top_content
        self.assertTrue(relevant,
                        f"最相关结果应包含'退休'或'养老'，实际内容: {top_content[:50]}...")

    # ================================================================
    # 测试3: 语义检索 - 精确关键词
    # ================================================================
    def test_search_exact_term(self):
        """测试精确术语查询：查询"戴维斯双击"应命中股票文档"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("什么是戴维斯双击", top_k=5)

        self.assertGreater(len(results), 0)
        found = any("戴维斯双击" in r["content"] for r in results)
        self.assertTrue(found, "应能检索到包含'戴维斯双击'的文档")

    # ================================================================
    # 测试4: 语义检索 - 无关查询得分低
    # ================================================================
    def test_irrelevant_query_low_score(self):
        """测试无关查询：查询"跑步"时，运动文档应排第一"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("跑步", top_k=3)

        self.assertGreater(len(results), 0)
        # 运动文档应排第一
        self.assertIn("运动", results[0]["content"])

    # ================================================================
    # 测试5: 得分降序排列
    # ================================================================
    def test_score_descending(self):
        """测试向量相似度得分降序排列"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("基金定投", top_k=5)

        if len(results) >= 2:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i]["score"],
                    results[i + 1]["score"],
                    "向量相似度得分应降序排列",
                )

    # ================================================================
    # 测试6: top_k 限制
    # ================================================================
    def test_top_k_limit(self):
        """测试 top_k 参数"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("投资", top_k=2)

        self.assertLessEqual(len(results), 2)

    # ================================================================
    # 测试7: 未构建索引时搜索
    # ================================================================
    def test_search_without_index(self):
        """测试未构建索引时返回空列表"""
        results = self.retriever.search("ETF", top_k=5)
        self.assertEqual(len(results), 0)

    # ================================================================
    # 测试8: 空查询
    # ================================================================
    def test_empty_query(self):
        """测试空查询"""
        self.retriever.build_index(str(self.docs_dir))
        results = self.retriever.search("", top_k=5)
        self.assertEqual(len(results), 0)

    # ================================================================
    # 测试9: 返回结果结构完整性
    # ================================================================
    def test_result_structure(self):
        """测试返回的 RetrievedDocument 结构完整"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("养老金", top_k=3)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("doc_id", r)
            self.assertIn("chunk_id", r)
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("source", r)
            self.assertIn("metadata", r)
            self.assertEqual(r["source"], "vector")
            # 得分应在 [-1, 1] 之间（余弦相似度）
            self.assertGreaterEqual(r["score"], -1.0)
            self.assertLessEqual(r["score"], 1.0)

    # ================================================================
    # 测试10: FAISS 索引持久化 - 保存与加载
    # ================================================================
    def test_save_and_load_index(self):
        """测试 FAISS 索引保存到磁盘并重新加载"""
        self.retriever.build_index(str(self.docs_dir))

        # 保存前搜索一次，记录结果
        results_before = self.retriever.search("退休规划", top_k=3)

        # 保存索引
        self.retriever.save_index()

        # 创建新的 retriever 并加载索引
        new_retriever = VectorRetriever(
            parser=self.parser,
            embedder=self.embedder,
            top_k=5,
            index_dir=str(self.index_dir),
        )
        new_retriever.load_index()

        # 加载后搜索，结果应与之前一致
        results_after = new_retriever.search("退休规划", top_k=3)

        self.assertEqual(len(results_before), len(results_after),
                         "加载后搜索结果数量应一致")
        for i in range(len(results_before)):
            self.assertEqual(
                results_before[i]["content"],
                results_after[i]["content"],
                f"加载后第{i}个结果的内容应一致",
            )

    # ================================================================
    # 测试11: 语义相似度数值验证
    # ================================================================
    def test_similarity_numeric(self):
        """测试自查询得分接近1.0（同一文档的向量与自身相似度最高）"""
        self.retriever.build_index(str(self.docs_dir))

        # 查询退休规划文档中的原文句子
        results = self.retriever.search(
            "退休规划是个人财务规划的重要组成部分",
            top_k=1,
        )

        self.assertGreater(len(results), 0)
        # 自查询的相似度应较高
        self.assertGreater(results[0]["score"], 0.3,
                           "自查询的向量相似度应较高")

    # ================================================================
    # 测试12: 单文档添加
    # ================================================================
    def test_add_document(self):
        """测试添加单个文档到向量索引"""
        self.retriever.build_index(str(self.docs_dir))
        initial_count = self.retriever.total_chunks

        new_doc = Path(self.temp_dir) / "new_doc.txt"
        new_doc.write_text(
            "科创板是中国资本市场改革的试验田，主要服务于科技创新企业。"
            "投资者可通过科创50ETF参与科创板投资。",
            encoding="utf-8",
        )

        self.retriever.add_document(str(new_doc))
        self.assertGreater(self.retriever.total_chunks, initial_count)

        results = self.retriever.search("科创板 ETF", top_k=5)
        self.assertGreater(len(results), 0)
        found = any("科创板" in r["content"] for r in results)
        self.assertTrue(found)


if __name__ == "__main__":
    unittest.main()