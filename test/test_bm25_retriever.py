# -*- coding: utf-8 -*-
"""
BM25Retriever 单元测试

验证 BM25 索引构建、关键词检索、结果排序的核心逻辑。
运行方式: python -m unittest test.test_bm25_retriever -v
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.doc_parser import DocParser
from retrieval.bm25_retriever import BM25Retriever


class TestBM25Retriever(unittest.TestCase):
    """BM25Retriever 单元测试"""

    def setUp(self):
        """每个测试用例运行前创建临时工作目录和测试文档"""
        self.temp_dir = tempfile.mkdtemp()
        self.chunks_dir = Path(self.temp_dir) / "chunks"
        self.chunks_dir.mkdir()

        self.parser = DocParser(
            chunk_size=512,
            chunk_overlap=64,
            output_dir=str(self.chunks_dir),
        )
        self.retriever = BM25Retriever(parser=self.parser)

        # 创建测试知识库文档
        self._create_test_docs()

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_docs(self):
        """创建多篇测试文档到临时目录"""
        docs_dir = Path(self.temp_dir) / "docs"
        docs_dir.mkdir()

        # 文档1: ETF 基础知识
        (docs_dir / "etf_basics.txt").write_text(
            "ETF（交易型开放式指数基金）是一种在交易所上市交易的指数基金。"
            "ETF结合了封闭式基金和开放式基金的特点，投资者可以像买卖股票一样交易ETF。"
            "ETF具有费率低、透明度高、交易灵活的优势。"
            "国内主要的ETF产品包括沪深300ETF、中证500ETF、科创50ETF等。",
            encoding="utf-8",
        )

        # 文档2: 债券基金
        (docs_dir / "bond_fund.txt").write_text(
            "债券基金主要投资于国债、企业债、金融债等固定收益类资产。"
            "债券基金的风险低于股票基金，预期收益高于货币基金。"
            "在央行降息周期中，已发行债券价格上升，债券基金净值上涨。"
            "投资者可根据风险偏好选择纯债基金、一级债基或二级债基。",
            encoding="utf-8",
        )

        # 文档3: 资产配置策略
        (docs_dir / "asset_allocation.txt").write_text(
            "资产配置是投资组合管理的核心环节。常见的资产配置策略包括："
            "1) 60/40策略：60%股票+40%债券的经典配置。"
            "2) 年龄法则：权益类资产占比=100-年龄。"
            "3) 全天候策略：根据经济周期动态调整各类资产比例。"
            "定期再平衡是维持目标配置比例的重要手段。",
            encoding="utf-8",
        )

        # 文档4: 风险管理
        (docs_dir / "risk_management.txt").write_text(
            "投资风险管理是保障资产安全的关键。主要风险类型包括："
            "市场风险、信用风险、流动性风险和操作风险。"
            "常用的风险管理工具有：止损单、期权对冲、分散投资。"
            "对于中小投资者，建议通过资产配置和定投策略来控制风险。"
            "保持足够的现金储备以应对突发情况也是风险管理的重要一环。",
            encoding="utf-8",
        )

        self.docs_dir = docs_dir

    # ================================================================
    # 测试1: 索引构建 - 基本功能
    # ================================================================
    def test_build_index(self):
        """测试从目录构建 BM25 索引"""
        self.retriever.build_index(str(self.docs_dir))

        self.assertGreater(self.retriever.total_chunks, 0,
                           "索引应包含至少一个分块")
        self.assertIsNotNone(self.retriever._bm25_model,
                             "BM25 模型应已初始化")

    # ================================================================
    # 测试2: 关键词检索 - 精确匹配
    # ================================================================
    def test_search_exact_keyword(self):
        """测试精确关键词检索：查询"ETF"应返回 ETF 相关文档"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("ETF", top_k=5)

        self.assertGreater(len(results), 0, "应返回至少一个结果")
        # 第一个结果应包含 "ETF"
        self.assertIn("ETF", results[0]["content"])
        self.assertEqual(results[0]["source"], "bm25")

    # ================================================================
    # 测试3: 关键词检索 - 语义相关
    # ================================================================
    def test_search_semantic_keyword(self):
        """测试语义相关关键词检索：查询"债券"应返回债券基金文档"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("债券基金 风险", top_k=5)

        self.assertGreater(len(results), 0)
        # 至少有一个结果包含"债券"
        found = any("债券" in r["content"] for r in results)
        self.assertTrue(found, "应至少有一个结果包含'债券'")

    # ================================================================
    # 测试4: 结果排序 - 相关性递减
    # ================================================================
    def test_score_descending(self):
        """测试 BM25 得分降序排列"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("ETF 指数基金", top_k=5)

        if len(results) >= 2:
            for i in range(len(results) - 1):
                self.assertGreaterEqual(
                    results[i]["score"],
                    results[i + 1]["score"],
                    "BM25 得分应降序排列",
                )

    # ================================================================
    # 测试5: top_k 限制
    # ================================================================
    def test_top_k_limit(self):
        """测试 top_k 参数限制返回数量"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("投资", top_k=2)

        self.assertLessEqual(len(results), 2)

    # ================================================================
    # 测试6: 未构建索引时搜索
    # ================================================================
    def test_search_without_index(self):
        """测试未构建索引时搜索应返回空列表"""
        results = self.retriever.search("ETF", top_k=5)
        self.assertEqual(len(results), 0, "未构建索引时应返回空列表")

    # ================================================================
    # 测试7: 空查询
    # ================================================================
    def test_empty_query(self):
        """测试空查询返回空列表"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("", top_k=5)
        self.assertEqual(len(results), 0, "空查询应返回空列表")

    # ================================================================
    # 测试8: 不相关查询得分低
    # ================================================================
    def test_irrelevant_query_low_score(self):
        """测试不相关查询的得分应很低"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("外星人 宇宙飞船", top_k=5)

        if len(results) > 0:
            # 所有得分应接近 0
            for r in results:
                self.assertLess(r["score"], 1.0,
                                "不相关查询的 BM25 得分应很低")

    # ================================================================
    # 测试9: 返回结果结构完整性
    # ================================================================
    def test_result_structure(self):
        """测试返回的 RetrievedDocument 结构完整"""
        self.retriever.build_index(str(self.docs_dir))

        results = self.retriever.search("资产配置", top_k=3)

        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("doc_id", r)
            self.assertIn("chunk_id", r)
            self.assertIn("content", r)
            self.assertIn("score", r)
            self.assertIn("source", r)
            self.assertIn("metadata", r)
            self.assertEqual(r["source"], "bm25")

    # ================================================================
    # 测试10: 单文档添加
    # ================================================================
    def test_add_document(self):
        """测试添加单个文档到索引"""
        self.retriever.build_index(str(self.docs_dir))
        initial_count = self.retriever.total_chunks

        new_doc = Path(self.temp_dir) / "new_doc.txt"
        new_doc.write_text(
            "科创板是中国资本市场改革的试验田，主要服务于科技创新企业。"
            "科创50ETF是跟踪科创板50指数的交易型开放式指数基金。"
            "投资者可通过科创50ETF参与科创板投资，分享科技创新红利。",
            encoding="utf-8",
        )

        self.retriever.add_document(str(new_doc))
        self.assertGreater(self.retriever.total_chunks, initial_count)

        results = self.retriever.search("科创板", top_k=5)
        self.assertGreater(len(results), 0)
        found = any("科创板" in r["content"] for r in results)
        self.assertTrue(found, "添加文档后应能检索到新内容")


if __name__ == "__main__":
    unittest.main()