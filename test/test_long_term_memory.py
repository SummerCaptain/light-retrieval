# -*- coding: utf-8 -*-
"""
LongTermMemoryManager 单元测试

验证 ES 索引创建、交互存储、相似查询搜索、用户偏好聚合等核心逻辑。
ES 不可用时自动跳过相关测试，不影响其他测试运行。
运行方式: python -m unittest test.test_long_term_memory -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.long_term import LongTermMemoryManager


# 测试用户 ID（避免污染生产数据）
TEST_USER_ID = "__test_user__"
TEST_SESSION_ID = "__test_session__"


class TestLongTermMemory(unittest.TestCase):
    """LongTermMemoryManager 单元测试"""

    @classmethod
    def setUpClass(cls):
        """所有测试用例运行前创建 ES 连接"""
        cls.ltm = LongTermMemoryManager()
        if not cls.ltm.is_connected:
            raise unittest.SkipTest("Elasticsearch 不可用，跳过 ES 相关测试")

    def setUp(self):
        """每个测试用例运行前清理测试数据"""
        if self.ltm.is_connected:
            self.ltm.delete_user_data(TEST_USER_ID)

    def tearDown(self):
        """每个测试用例运行后清理测试数据"""
        if self.ltm.is_connected:
            self.ltm.delete_user_data(TEST_USER_ID)

    # ================================================================
    # 测试1: 存储交互记录
    # ================================================================
    def test_store_interaction(self):
        """测试存储单条交互记录"""
        doc_id = self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="什么是ETF？",
            answer="ETF是交易型开放式指数基金...",
            query_type="knowledge_retrieval",
            retrieved_docs=["doc_001", "doc_002"],
            tags=["ETF", "指数基金"],
            response_time_ms=150.5,
        )

        self.assertIsNotNone(doc_id, "存储后应返回文档 ID")
        self.assertIsInstance(doc_id, str)

    # ================================================================
    # 测试2: 相似查询搜索
    # ================================================================
    def test_search_similar(self):
        """测试搜索相似历史查询"""
        # 先存储几条记录
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="ETF和指数基金有什么区别？",
            answer="主要区别在于...",
            tags=["ETF"],
        )
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="债券基金适合保守型投资者吗？",
            answer="债券基金风险较低...",
            tags=["债券基金"],
        )

        # 搜索与 ETF 相关的查询
        results = self.ltm.search_similar(
            user_id=TEST_USER_ID,
            query="ETF 交易规则",
            top_k=5,
        )

        self.assertGreater(len(results), 0, "应返回相似历史记录")
        # 第一条结果应包含 "ETF"
        self.assertIn("ETF", results[0]["query"])

    # ================================================================
    # 测试3: 用户标签聚合
    # ================================================================
    def test_get_user_tags(self):
        """测试用户偏好标签聚合"""
        # 存储多条带不同标签的记录
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="ETF是什么？",
            answer="...",
            tags=["ETF", "指数基金"],
        )
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="债券基金怎么选？",
            answer="...",
            tags=["债券基金"],
        )
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="科创50ETF怎么买？",
            answer="...",
            tags=["ETF", "科创板"],
        )

        tags = self.ltm.get_user_tags(TEST_USER_ID, limit=5)

        self.assertGreater(len(tags), 0, "应返回偏好标签")
        # "ETF" 出现了两次，应排第一
        self.assertEqual(tags[0], "ETF")

    # ================================================================
    # 测试4: 最近交互记录
    # ================================================================
    def test_get_recent(self):
        """测试获取最近 N 条交互记录"""
        for i in range(3):
            self.ltm.store_interaction(
                user_id=TEST_USER_ID,
                session_id=TEST_SESSION_ID,
                query=f"查询{i}",
                answer=f"回答{i}",
            )

        results = self.ltm.get_recent(TEST_USER_ID, limit=2)

        self.assertEqual(len(results), 2, "应返回指定数量的记录")
        # 按时间降序，最新的应排第一
        self.assertIn("查询", results[0]["query"])

    # ================================================================
    # 测试5: 反馈更新
    # ================================================================
    def test_update_feedback(self):
        """测试更新用户反馈"""
        doc_id = self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="测试查询",
            answer="测试回答",
        )

        self.assertIsNotNone(doc_id)

        # 更新为正面反馈
        success = self.ltm.update_feedback(doc_id, "positive")
        self.assertTrue(success)

        # 验证反馈统计
        stats = self.ltm.get_feedback_stats(TEST_USER_ID)
        self.assertEqual(stats["positive"], 1)
        self.assertEqual(stats["total"], 1)

    # ================================================================
    # 测试6: 反馈统计
    # ================================================================
    def test_feedback_stats(self):
        """测试反馈统计"""
        for fb in ["positive", "positive", "negative", "neutral"]:
            doc_id = self.ltm.store_interaction(
                user_id=TEST_USER_ID,
                session_id=TEST_SESSION_ID,
                query=f"测试{fb}",
                answer="回答",
            )
            self.ltm.update_feedback(doc_id, fb)

        stats = self.ltm.get_feedback_stats(TEST_USER_ID)
        self.assertEqual(stats["positive"], 2)
        self.assertEqual(stats["negative"], 1)
        self.assertEqual(stats["neutral"], 1)
        self.assertEqual(stats["total"], 4)

    # ================================================================
    # 测试7: 不同用户隔离
    # ================================================================
    def test_user_isolation(self):
        """测试不同用户之间数据隔离"""
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="用户A的查询",
            answer="...",
        )
        self.ltm.store_interaction(
            user_id="__other_user__",
            session_id=TEST_SESSION_ID,
            query="用户B的查询",
            answer="...",
        )

        # 搜索用户A的查询，不应包含用户B的数据
        results = self.ltm.search_similar(
            user_id=TEST_USER_ID,
            query="查询",
            top_k=5,
        )

        for r in results:
            self.assertEqual(r["user_id"], TEST_USER_ID,
                             "搜索结果不应包含其他用户的数据")

        # 清理
        self.ltm.delete_user_data("__other_user__")

    # ================================================================
    # 测试8: 删除用户数据
    # ================================================================
    def test_delete_user_data(self):
        """测试删除用户所有数据"""
        self.ltm.store_interaction(
            user_id=TEST_USER_ID,
            session_id=TEST_SESSION_ID,
            query="待删除的查询",
            answer="...",
        )

        success = self.ltm.delete_user_data(TEST_USER_ID)
        self.assertTrue(success)

        # 删除后应查不到数据
        results = self.ltm.get_recent(TEST_USER_ID, limit=10)
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()