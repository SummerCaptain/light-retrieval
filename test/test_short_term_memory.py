# -*- coding: utf-8 -*-
"""
ShortTermMemoryManager 单元测试

验证短期记忆的添加、裁剪、上下文格式化、追问判断等核心逻辑。
运行方式: python -m unittest test.test_short_term_memory -v
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.short_term import ShortTermMemoryManager


class TestShortTermMemory(unittest.TestCase):
    """ShortTermMemoryManager 单元测试"""

    def setUp(self):
        self.memory = ShortTermMemoryManager(
            session_id="test_session",
            max_rounds=5,
        )

    # ================================================================
    # 测试1: 基本添加
    # ================================================================
    def test_add_entry(self):
        """测试添加单条记忆条目"""
        self.memory.add_entry("user", "什么是ETF？", query_type="knowledge_retrieval")
        self.memory.add_entry("assistant", "ETF是交易型开放式指数基金...")

        self.assertEqual(self.memory.total_entries, 2)
        self.assertEqual(self.memory.total_rounds, 1)

    # ================================================================
    # 测试2: 便捷方法
    # ================================================================
    def test_convenience_methods(self):
        """测试 add_user_query 和 add_assistant_response"""
        self.memory.add_user_query("基金定投怎么操作？")
        self.memory.add_assistant_response(
            "定投需要先选择标的基金...",
            retrieved_docs=["doc_001", "doc_002"],
        )

        self.assertEqual(self.memory.total_entries, 2)
        self.assertEqual(
            self.memory.get_last_user_query(),
            "基金定投怎么操作？",
        )
        self.assertEqual(
            self.memory.get_last_answer(),
            "定投需要先选择标的基金...",
        )

    # ================================================================
    # 测试3: 上下文格式化
    # ================================================================
    def test_get_context(self):
        """测试格式化对话上下文"""
        self.memory.add_user_query("你好")
        self.memory.add_assistant_response("你好！有什么可以帮您的？")

        context = self.memory.get_context()

        self.assertIn("[历史对话]", context)
        self.assertIn("用户: 你好", context)
        self.assertIn("助手: 你好！有什么可以帮您的？", context)

    # ================================================================
    # 测试4: 空上下文
    # ================================================================
    def test_empty_context(self):
        """测试无历史时返回空字符串"""
        self.assertEqual(self.memory.get_context(), "")
        self.assertEqual(self.memory.get_recent_queries(), [])

    # ================================================================
    # 测试5: 容量裁剪 - 超出 max_rounds
    # ================================================================
    def test_trim_on_overflow(self):
        """测试超出容量时自动淘汰最旧记录"""
        memory = ShortTermMemoryManager(max_rounds=2)

        # 添加 3 轮对话（6 条记录）
        for i in range(3):
            memory.add_user_query(f"问题{i}")
            memory.add_assistant_response(f"回答{i}")

        # 应只保留最近 2 轮（4 条记录）
        self.assertEqual(memory.total_entries, 4)
        self.assertEqual(memory.total_rounds, 2)

        # 最旧的第一轮应被淘汰
        queries = memory.get_recent_queries()
        self.assertNotIn("问题0", queries)
        self.assertIn("问题1", queries)
        self.assertIn("问题2", queries)

    # ================================================================
    # 测试6: 最近查询获取
    # ================================================================
    def test_get_recent_queries(self):
        """测试获取最近 N 条用户查询"""
        for i in range(5):
            self.memory.add_user_query(f"查询{i}")
            self.memory.add_assistant_response(f"答案{i}")

        recent = self.memory.get_recent_queries(n=3)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent, ["查询2", "查询3", "查询4"])

    # ================================================================
    # 测试7: 追问判断 - 短查询 + 有历史
    # ================================================================
    def test_is_follow_up_short_query(self):
        """测试短查询 + 有历史 → 追问"""
        self.memory.add_user_query("ETF和指数基金有什么区别？")
        self.memory.add_assistant_response("ETF是交易所交易的...")

        # 短查询 + 包含指代词
        self.assertTrue(self.memory.is_follow_up("那怎么买呢？"))
        self.assertTrue(self.memory.is_follow_up("还有吗？"))
        self.assertTrue(self.memory.is_follow_up("为什么？"))

    # ================================================================
    # 测试8: 追问判断 - 无历史
    # ================================================================
    def test_is_follow_up_no_history(self):
        """测试无历史 → 不是追问"""
        self.assertFalse(self.memory.is_follow_up("那怎么买呢？"))

    # ================================================================
    # 测试9: 追问判断 - 长查询
    # ================================================================
    def test_is_follow_up_long_query(self):
        """测试长查询 → 不是追问（即使有历史）"""
        self.memory.add_user_query("ETF是什么？")
        self.memory.add_assistant_response("ETF是...")

        long_query = "请详细介绍一下ETF的交易规则、手续费和税务处理方式"
        self.assertFalse(self.memory.is_follow_up(long_query))

    # ================================================================
    # 测试10: 清空与重置
    # ================================================================
    def test_clear_and_reset(self):
        """测试清空和重置"""
        self.memory.add_user_query("测试")
        self.memory.add_assistant_response("回答")

        self.memory.clear()
        self.assertEqual(self.memory.total_entries, 0)
        self.assertTrue(self.memory.is_empty)

        # 重置并更改 session_id
        self.memory.add_user_query("新会话")
        self.memory.reset(new_session_id="new_session")

        self.assertEqual(self.memory.session_id, "new_session")
        self.assertEqual(self.memory.total_entries, 0)

    # ================================================================
    # 测试11: 导出
    # ================================================================
    def test_to_dict(self):
        """测试导出为字典"""
        self.memory.add_user_query("测试", query_type="knowledge_retrieval")
        self.memory.add_assistant_response("回答", retrieved_docs=["doc_001"])

        d = self.memory.to_dict()
        self.assertEqual(d["session_id"], "test_session")
        self.assertEqual(len(d["entries"]), 2)
        self.assertEqual(d["entries"][0]["query_type"], "knowledge_retrieval")
        self.assertEqual(d["entries"][1]["retrieved_docs"], ["doc_001"])

    # ================================================================
    # 测试12: 仅用户查询无回答
    # ================================================================
    def test_only_user_query(self):
        """测试只有用户查询没有回答时 get_last_answer 返回 None"""
        self.memory.add_user_query("只有问题没有回答")
        self.assertIsNone(self.memory.get_last_answer())

    # ================================================================
    # 测试13: 追问边界 - 空查询
    # ================================================================
    def test_is_follow_up_empty_query(self):
        """测试空查询 → 不是追问"""
        self.memory.add_user_query("之前问过ETF")
        self.memory.add_assistant_response("ETF是...")

        self.assertFalse(self.memory.is_follow_up(""))
        self.assertFalse(self.memory.is_follow_up("   "))

    # ================================================================
    # 测试14: 追问边界 - 恰好 15 字
    # ================================================================
    def test_is_follow_up_exact_15_chars(self):
        """测试恰好 15 字（边界值）→ 不是追问"""
        self.memory.add_user_query("ETF是什么？")
        self.memory.add_assistant_response("ETF是...")

        # 恰好 15 字，>= 15 应视为长查询，不是追问
        query_15 = "那请问这个ETF具体怎么操作呢"  # 15 字
        self.assertEqual(len(query_15), 15)
        self.assertFalse(self.memory.is_follow_up(query_15))

        # 14 字，带指示词，应视为追问
        query_14 = "那具体怎么操作呢？"  # 14 字
        self.assertTrue(self.memory.is_follow_up(query_14))

    # ================================================================
    # 测试15: 追问边界 - 短查询但无指示词
    # ================================================================
    def test_is_follow_up_short_no_indicator(self):
        """测试短查询但没有追问指示词 → 不是追问"""
        self.memory.add_user_query("ETF有什么特点？")
        self.memory.add_assistant_response("ETF特点是...")

        # 短但无指示词
        self.assertFalse(self.memory.is_follow_up("ETF"))
        self.assertFalse(self.memory.is_follow_up("费用"))
        self.assertFalse(self.memory.is_follow_up("风险"))

    # ================================================================
    # 测试16: 追问边界 - 数字类短查询
    # ================================================================
    def test_is_follow_up_numeric_query(self):
        """测试数字类短查询 → 追问"""
        self.memory.add_user_query("个人养老金每年能缴多少？")
        self.memory.add_assistant_response("每年最高12000元...")

        # 数字追问通常很短
        self.assertTrue(self.memory.is_follow_up("有上限吗？"))
        self.assertFalse(self.memory.is_follow_up("12000"))

    # ================================================================
    # 测试17: 追问边界 - 多轮追问链
    # ================================================================
    def test_is_follow_up_multi_turn(self):
        """测试多轮追问链（3 轮以上）"""
        self.memory.add_user_query("什么是债券基金？")
        self.memory.add_assistant_response("债券基金是...")
        self.memory.add_user_query("那风险大吗？")
        self.memory.add_assistant_response("风险较低...")
        self.memory.add_user_query("怎么买？")
        self.memory.add_assistant_response("可以通过银行...")

        # 第 4 轮短追问
        self.assertTrue(self.memory.is_follow_up("手续费多少？"))
        self.assertTrue(self.memory.is_follow_up("还有别的吗？"))

    # ================================================================
    # 测试18: 追问边界 - 符号类短查询
    # ================================================================
    def test_is_follow_up_symbols(self):
        """测试只含符号/标点的查询 → 不是追问"""
        self.memory.add_user_query("ETF怎么交易？")
        self.memory.add_assistant_response("ETF可以像股票一样交易...")

        self.assertFalse(self.memory.is_follow_up("？"))
        self.assertFalse(self.memory.is_follow_up("..."))
        self.assertFalse(self.memory.is_follow_up("！"))

    # ================================================================
    # 测试19: 追问边界 - 带反问的查询
    # ================================================================
    def test_is_follow_up_rhetorical(self):
        """测试带反问语句的查询"""
        self.memory.add_user_query("现在适合买债基吗？")
        self.memory.add_assistant_response("当前利率环境下...")

        # 反问式追问
        self.assertTrue(self.memory.is_follow_up("不是降息了吗？"))
        self.assertTrue(self.memory.is_follow_up("那为什么还跌？"))

    # ================================================================
    # 测试20: 追问边界 - 仅助手的单边历史
    # ================================================================
    def test_is_follow_up_assistant_only(self):
        """测试只有助手回答没有用户查询的历史"""
        self.memory.add_assistant_response("欢迎使用知识库系统！")

        # 有历史但无用户查询，短查询也可能是新话题
        self.assertFalse(self.memory.is_follow_up("那怎么用？"))


if __name__ == "__main__":
    unittest.main()