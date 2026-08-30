# -*- coding: utf-8 -*-
"""
LangGraph 工作流单元测试

验证知识库系统工作流的核心逻辑：
- 路由决策（reactive / deliberative）
- 混合检索 + 重排序
- 记忆注入与存储
- 完整 RAG 管道端到端

注意：测试需要 DASHSCOPE_API_KEY 环境变量才能调用 LLM。
未设置时自动跳过 LLM 相关测试。
运行方式: python -m unittest test.test_workflow -v
"""

import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.state import KnowledgeBaseState, RetrievedDocument


class TestWorkflowRouting(unittest.TestCase):
    """工作流路由逻辑测试（不依赖 LLM）"""

    def test_route_deliberative(self):
        """测试 deliberative 模式路由到 hybrid_search"""
        from core.workflow import route_after_assess

        state = {"processing_mode": "deliberative"}
        self.assertEqual(route_after_assess(state), "hybrid_search")

    def test_route_reactive(self):
        """测试 reactive 模式路由到 direct_answer"""
        from core.workflow import route_after_assess

        state = {"processing_mode": "reactive"}
        self.assertEqual(route_after_assess(state), "direct_answer")

    def test_route_default(self):
        """测试默认路由（无 processing_mode）"""
        from core.workflow import route_after_assess

        state = {}
        self.assertEqual(route_after_assess(state), "direct_answer")


class TestRerankerFactory(unittest.TestCase):
    """重排序器工厂函数测试"""

    def test_create_bge(self):
        """测试创建 BGEReranker（不加载模型）"""
        from retrieval.reranker import create_reranker, BGEReranker

        reranker = create_reranker()
        self.assertIsInstance(reranker, BGEReranker)

    def test_rerank_empty(self):
        """测试空查询或空文档列表直接返回空（不触发模型加载）"""
        from retrieval.reranker import create_reranker

        reranker = create_reranker()
        doc = {
            "doc_id": "doc1",
            "chunk_id": 0,
            "content": "a",
            "score": 0.5,
            "source": "hybrid",
            "metadata": {"source_file": "doc1.txt"},
        }
        self.assertEqual(reranker.rerank("", [doc], 3), [])
        self.assertEqual(reranker.rerank("测试", [], 3), [])


class TestStateDefinition(unittest.TestCase):
    """状态定义测试"""

    def test_retrieved_document_fields(self):
        """测试 RetrievedDocument 字段完整性"""
        doc = RetrievedDocument(
            doc_id="doc1",
            chunk_id=0,
            content="测试内容",
            score=0.95,
            source="hybrid",
            metadata={"source_file": "test.txt"},
        )

        self.assertEqual(doc["doc_id"], "doc1")
        self.assertEqual(doc["score"], 0.95)
        self.assertEqual(doc["source"], "hybrid")

    def test_memory_entry_fields(self):
        """测试 MemoryEntry 字段完整性"""
        from core.state import MemoryEntry

        entry = MemoryEntry(
            role="user",
            content="什么是ETF？",
            timestamp="2026-01-01T00:00:00",
            query_type="knowledge_retrieval",
            retrieved_docs=["doc1"],
        )

        self.assertEqual(entry["role"], "user")
        self.assertEqual(entry["query_type"], "knowledge_retrieval")

    def test_knowledge_base_state_fields(self):
        """测试 KnowledgeBaseState 字段完整性"""
        state = KnowledgeBaseState(
            user_query="测试查询",
            session_id="test",
            query_type=None,
            processing_mode=None,
            bm25_results=None,
            vector_results=None,
            fused_results=None,
            reranked_results=None,
            final_docs=None,
            short_term_memory=None,
            long_term_context=None,
            messages=[],
            final_response=None,
            sources=None,
            current_phase="assess",
            error=None,
            evaluation_trace=None,
        )

        self.assertEqual(state["user_query"], "测试查询")
        self.assertEqual(state["current_phase"], "assess")


class TestTagExtraction(unittest.TestCase):
    """标签提取测试"""

    def test_extract_tags(self):
        """测试查询类型到标签的映射"""
        from core.workflow import _extract_tags

        self.assertEqual(_extract_tags("simple_qa"), ["简单问答"])
        self.assertEqual(_extract_tags("knowledge_retrieval"), ["知识检索"])
        self.assertEqual(_extract_tags("comparative_analysis"), ["对比分析"])
        self.assertEqual(_extract_tags("procedural_guide"), ["步骤指南"])
        self.assertEqual(_extract_tags("unknown"), [])


if __name__ == "__main__":
    unittest.main()
