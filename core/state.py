# -*- coding: utf-8 -*-
"""
知识库系统状态定义

基于原有 WealthAdvisorState 扩展，新增检索结果、记忆上下文等字段，
支持混合检索 RAG 管道的全链路状态传递。
"""

from typing import Dict, List, Any, Literal, TypedDict, Optional, Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class RetrievedDocument(TypedDict):
    """检索到的文档片段"""
    doc_id: str                                  # 文档唯一标识
    chunk_id: int                                # 分块编号
    content: str                                 # 分块内容
    score: float                                 # 综合得分（融合后）
    source: str                                  # 来源：bm25 / vector / hybrid
    metadata: Dict[str, Any]                     # 元数据（文件名、页码等）


class MemoryEntry(TypedDict):
    """记忆条目"""
    role: Literal["user", "assistant"]           # 角色
    content: str                                 # 内容
    timestamp: str                               # 时间戳
    query_type: Optional[str]                    # 查询类型
    retrieved_docs: Optional[List[str]]          # 该轮检索到的文档ID列表


class KnowledgeBaseState(TypedDict):
    """知识库智能体的完整状态"""

    # ============================================================
    # 输入
    # ============================================================
    user_query: str                              # 用户查询
    session_id: Optional[str]                    # 会话ID（用于记忆关联）

    # ============================================================
    # 协调层 - 查询评估
    # ============================================================
    query_type: Optional[Literal[
        "simple_qa",            # 简单问答（直接回答，无需检索）
        "knowledge_retrieval",  # 知识检索（需要检索知识库）
        "comparative_analysis", # 对比分析（需要多文档综合）
        "procedural_guide"      # 步骤指南（需要结构化内容）
    ]]
    processing_mode: Optional[Literal[
        "reactive",             # 反应式：快速直接回答
        "deliberative"          # 深思熟虑：RAG 深度检索 + 分析
    ]]

    # ============================================================
    # 检索层 - 混合检索结果
    # ============================================================
    bm25_results: Optional[List[RetrievedDocument]]     # BM25 检索结果
    vector_results: Optional[List[RetrievedDocument]]   # 向量检索结果
    fused_results: Optional[List[RetrievedDocument]]    # RRF 融合后结果
    reranked_results: Optional[List[RetrievedDocument]] # Reranker 重排后结果
    final_docs: Optional[List[RetrievedDocument]]       # 最终使用的文档

    # ============================================================
    # 记忆层
    # ============================================================
    short_term_memory: Optional[List[MemoryEntry]]      # 短期记忆（当前会话）
    long_term_context: Optional[str]                    # 长期记忆检索到的上下文

    # ============================================================
    # 消息历史（用于 LLM 工具调用）
    # ============================================================
    messages: Annotated[List[BaseMessage], add_messages]

    # ============================================================
    # 输出
    # ============================================================
    final_response: Optional[str]               # 最终响应
    sources: Optional[List[str]]                # 引用来源列表

    # ============================================================
    # 控制流
    # ============================================================
    current_phase: Optional[str]                # 当前处理阶段
    error: Optional[str]                        # 错误信息
    evaluation_trace: Optional[Dict[str, Any]]  # 评估追踪数据（用于 LangFuse）