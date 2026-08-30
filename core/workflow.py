# -*- coding: utf-8 -*-
"""
知识库系统 LangGraph 工作流

基于原有投顾 Agent 工作流改造，串通 RAG 管道：
评估 → 混合检索 → 重排序 → 记忆注入 → LLM 生成 → 记忆存储

关键改造：
- 评估层：query_type 扩展为 4 种，新增 knowledge_retrieval / procedural_guide
- 检索层：替换原有 collect_data/analyze 为 HybridRetriever + Reranker
- 记忆层：新增短期/长期记忆注入与存储节点
- 生成层：基于检索结果 + 记忆上下文生成回答
"""

import json
import os
import uuid
import logging
import contextvars
from datetime import datetime
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from config.settings import (
    DASHSCOPE_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE,
    EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE,
    RERANKER_TOP_K, KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR,
)
from core.state import KnowledgeBaseState, RetrievedDocument
from retrieval.doc_parser import DocParser
from retrieval.bm25_retriever import BM25Retriever
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import create_reranker
from memory.short_term import ShortTermMemoryManager
from memory.long_term import LongTermMemoryManager

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 结构化日志（trace_id 关联单次请求的日志）
# ============================================================
logger = logging.getLogger("knowledge_base")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("kb_trace_id", default="")


def _log(msg: str, *args) -> None:
    """打印带 trace_id 前缀的结构化日志"""
    tid = _trace_id.get()
    prefix = f"[{tid}] " if tid else ""
    logger.info(prefix + msg, *args)


# ============================================================
# LLM 实例
# ============================================================
llm = ChatTongyi(
    model_name=LLM_MODEL_NAME,
    dashscope_api_key=DASHSCOPE_API_KEY,
    temperature=LLM_TEMPERATURE,
)

# ============================================================
# Prompt 模板
# ============================================================

ASSESSMENT_PROMPT = """你是一个知识库AI助手的协调层。请评估以下用户查询，确定其类型和应该采用的处理模式。

用户查询: {user_query}

请判断:
1. 查询类型:
   - "simple_qa": 简单问答，可以直接回答（如问候、闲聊、常识性问题）
   - "knowledge_retrieval": 需要检索知识库的查询（如专业概念、产品信息、政策解读）
   - "comparative_analysis": 需要对比分析的查询（如A和B的对比、优劣分析）
   - "procedural_guide": 需要步骤指南的查询（如操作流程、教程、法规步骤）

2. 处理模式:
   - "reactive": 简单问答，直接回答
   - "deliberative": 需要检索知识库后回答

请以JSON格式返回:
- query_type: 查询类型
- processing_mode: 处理模式
- reasoning: 决策理由
"""

RAG_GENERATION_PROMPT = """你是一个专业的知识库AI助手。请根据检索到的知识文档和对话历史，回答用户的问题。

要求：
1. 基于检索到的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请如实说明
3. 回答要清晰、准确、有条理
4. 引用文档时标注来源

{memory_context}

检索到的知识文档:
{retrieved_docs}

用户问题: {user_query}

请给出详细回答:"""

DIRECT_ANSWER_PROMPT = """你是一个友好的知识库AI助手。请直接回答用户的问题。

{memory_context}

用户问题: {user_query}

请给出简洁回答:"""


# ============================================================
# 节点函数
# ============================================================

def assess_query(state: KnowledgeBaseState) -> Dict[str, Any]:
    """评估查询类型和处理模式"""
    _log("节点: assess_query")

    prompt = ChatPromptTemplate.from_template(ASSESSMENT_PROMPT)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"user_query": state["user_query"]})

    _log(f"评估结果: {result}")

    processing_mode = result.get("processing_mode", "reactive")
    if processing_mode not in ["reactive", "deliberative"]:
        processing_mode = "reactive"

    query_type = result.get("query_type", "simple_qa")
    if query_type not in ["simple_qa", "knowledge_retrieval", "comparative_analysis", "procedural_guide"]:
        query_type = "simple_qa"

    _log(f"路由: mode={processing_mode}, type={query_type}")

    return {
        "query_type": query_type,
        "processing_mode": processing_mode,
    }


def hybrid_search(state: KnowledgeBaseState) -> Dict[str, Any]:
    """混合检索：BM25 + Vector → RRF 融合"""
    _log("节点: hybrid_search")

    # 从全局获取检索器实例
    retriever = _get_or_init_retriever()
    query = state["user_query"]

    results = retriever.search(query, top_k=20)

    _log(f"混合检索返回 {len(results)} 条结果")

    return {
        "fused_results": results,
        "current_phase": "rerank",
    }


def rerank(state: KnowledgeBaseState) -> Dict[str, Any]:
    """重排序：BGE-Reranker 交叉编码器"""
    _log("节点: rerank")

    reranker = _get_or_init_reranker()
    fused_results = state.get("fused_results") or []

    if not fused_results:
        _log("无候选文档，跳过重排序")
        return {
            "reranked_results": [],
            "final_docs": [],
            "current_phase": "inject_memory",
        }

    reranked = reranker.rerank(
        query=state["user_query"],
        documents=fused_results,
        top_k=RERANKER_TOP_K,
    )

    _log(f"重排序后保留 {len(reranked)} 条结果")

    return {
        "reranked_results": reranked,
        "final_docs": reranked,
        "current_phase": "inject_memory",
    }


def inject_memory(state: KnowledgeBaseState) -> Dict[str, Any]:
    """注入记忆上下文：短期记忆 + 长期记忆"""
    _log("节点: inject_memory")

    session_id = state.get("session_id") or "default"

    # 短期记忆
    stm = _get_or_init_stm(session_id)
    short_term_context = stm.get_context()

    # 追问判断
    is_follow = stm.is_follow_up(state["user_query"])

    # 长期记忆：用户偏好画像 + 相似历史问答（含答案）
    ltm = _get_or_init_ltm()
    long_term_context = ""
    user_id = state.get("session_id") or "default"

    if ltm.is_connected:
        parts = []

        # 用户偏好主题（历史交互最常问的标签）
        tags = ltm.get_user_tags(user_id, limit=5)
        if tags:
            parts.append("[用户偏好主题] " + "、".join(tags))

        # 相似历史问答（问题 + 答案，答案截断避免上下文过长）
        similar = ltm.search_similar(user_id, state["user_query"], top_k=3)
        if similar:
            lines = ["[相似历史问答]"]
            for i, record in enumerate(similar, 1):
                lines.append(f"{i}. Q: {record.get('query', '')}")
                answer = record.get("answer", "")
                if answer:
                    lines.append(f"   A: {answer[:200]}")
            parts.append("\n".join(lines))

        # 反馈信号（有正/负面反馈时提示，便于调整回答风格）
        feedback = ltm.get_feedback_stats(user_id)
        if feedback.get("positive", 0) > 0 or feedback.get("negative", 0) > 0:
            parts.append(
                f"[用户反馈] 正面 {feedback.get('positive', 0)} / 负面 {feedback.get('negative', 0)}"
            )

        long_term_context = "\n\n".join(parts)

    # 合并记忆上下文
    memory_parts = []
    if short_term_context:
        memory_parts.append(short_term_context)
    if long_term_context:
        memory_parts.append(long_term_context)
    if is_follow:
        memory_parts.append("[注意: 用户在追问上轮话题，请结合上下文回答]")

    combined_context = "\n\n".join(memory_parts) if memory_parts else ""

    _log(f"记忆注入: 短期={'有' if short_term_context else '无'}, "
         f"长期={'有' if long_term_context else '无'}, 追问={'是' if is_follow else '否'}")

    return {
        "long_term_context": combined_context,
        "short_term_memory": stm.to_dict()["entries"],
        "current_phase": "generate",
    }


def generate_answer(state: KnowledgeBaseState) -> Dict[str, Any]:
    """基于检索结果 + 记忆上下文生成回答"""
    _log("节点: generate_answer")

    # 格式化检索文档
    final_docs = state.get("final_docs") or []
    docs_text = ""
    sources = []
    if final_docs:
        docs_lines = []
        for i, doc in enumerate(final_docs, 1):
            source_file = doc.get("metadata", {}).get("source_file", "未知来源")
            docs_lines.append(f"[文档{i}] (来源: {source_file})\n{doc.get('content', '')}")
            if source_file not in sources:
                sources.append(source_file)
        docs_text = "\n\n".join(docs_lines)
    else:
        docs_text = "未检索到相关文档。"

    memory_context = state.get("long_term_context") or ""

    prompt = ChatPromptTemplate.from_template(RAG_GENERATION_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "retrieved_docs": docs_text,
        "memory_context": memory_context,
    })

    _log(f"生成回答完成, 长度: {len(result)}")

    return {
        "final_response": result,
        "sources": sources,
        "current_phase": "store_memory",
    }


def direct_answer(state: KnowledgeBaseState) -> Dict[str, Any]:
    """反应式：直接回答简单问题"""
    _log("节点: direct_answer")

    session_id = state.get("session_id") or "default"
    stm = _get_or_init_stm(session_id)
    memory_context = stm.get_context()

    prompt = ChatPromptTemplate.from_template(DIRECT_ANSWER_PROMPT)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "user_query": state["user_query"],
        "memory_context": memory_context,
    })

    return {
        "final_response": result,
        "sources": [],
        "current_phase": "store_memory",
    }


def store_memory(state: KnowledgeBaseState) -> Dict[str, Any]:
    """双写记忆：短期 + 长期"""
    _log("节点: store_memory")

    session_id = state.get("session_id") or "default"
    query = state["user_query"]
    answer = state.get("final_response") or ""
    query_type = state.get("query_type") or ""
    final_docs = state.get("final_docs") or []
    doc_ids = [d.get("doc_id", "") for d in final_docs if d.get("doc_id")]

    # 短期记忆
    stm = _get_or_init_stm(session_id)
    stm.add_user_query(query, query_type=query_type)
    stm.add_assistant_response(answer, retrieved_docs=doc_ids)

    # 长期记忆
    ltm = _get_or_init_ltm()
    if ltm.is_connected:
        tags = _extract_tags(query_type)
        ltm.store_interaction(
            user_id=session_id,
            session_id=session_id,
            query=query,
            answer=answer,
            query_type=query_type,
            retrieved_docs=doc_ids,
            tags=tags,
        )
        _log("长期记忆已存储")
    else:
        _log("ES 不可用，仅存储短期记忆")

    return {"current_phase": "respond"}


# ============================================================
# 路由函数
# ============================================================

def route_after_assess(state: KnowledgeBaseState) -> str:
    """评估后路由：reactive → 直接回答，deliberative → RAG 检索"""
    mode = state.get("processing_mode", "reactive")
    if mode == "deliberative":
        return "hybrid_search"
    return "direct_answer"


# ============================================================
# 全局实例（延迟初始化）
# ============================================================

_retriever = None
_reranker = None
_workflow = None
_stm_instances: Dict[str, ShortTermMemoryManager] = {}
_ltm_instance = None


def _get_or_init_retriever() -> HybridRetriever:
    """获取或初始化混合检索器"""
    global _retriever
    if _retriever is None:
        bm25 = BM25Retriever()
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
        vector = VectorRetriever(embedder=embedder, index_dir=str(VECTOR_STORE_DIR))

        kb_dir = str(KNOWLEDGE_BASE_DIR)

        # 向量索引优先从磁盘加载；知识库文档有更新时重建
        if _vector_index_fresh():
            vector.load_index()
            _log("向量索引已从磁盘加载")
        else:
            vector.build_index(kb_dir)
            vector.save_index()
            _log("向量索引已构建并保存")

        bm25.build_index(kb_dir)

        _retriever = HybridRetriever(bm25, vector)
    return _retriever


def _vector_index_fresh() -> bool:
    """判断磁盘上的向量索引是否与知识库文档同步（避免使用过期索引）"""
    index_file = VECTOR_STORE_DIR / "vector.index"
    registry_file = VECTOR_STORE_DIR / "registry.json"
    if not (index_file.exists() and registry_file.exists()):
        return False

    index_mtime = index_file.stat().st_mtime
    supported = {".txt", ".md", ".pdf", ".docx"}
    for f in KNOWLEDGE_BASE_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in supported:
            if f.stat().st_mtime > index_mtime:
                return False
    return True


def _get_or_init_reranker():
    """获取或初始化重排序器"""
    global _reranker
    if _reranker is None:
        _reranker = create_reranker()
    return _reranker


def _get_or_init_stm(session_id: str) -> ShortTermMemoryManager:
    """获取或初始化短期记忆"""
    if session_id not in _stm_instances:
        _stm_instances[session_id] = ShortTermMemoryManager(session_id=session_id)
    return _stm_instances[session_id]


def _get_or_init_ltm() -> LongTermMemoryManager:
    """获取或初始化长期记忆"""
    global _ltm_instance
    if _ltm_instance is None:
        _ltm_instance = LongTermMemoryManager()
    return _ltm_instance


def _extract_tags(query_type: str) -> List[str]:
    """从查询类型提取标签"""
    tag_map = {
        "simple_qa": ["简单问答"],
        "knowledge_retrieval": ["知识检索"],
        "comparative_analysis": ["对比分析"],
        "procedural_guide": ["步骤指南"],
    }
    return tag_map.get(query_type, [])


# ============================================================
# 工作流构建
# ============================================================

def create_knowledge_base_workflow() -> StateGraph:
    """创建知识库系统 LangGraph 工作流"""

    workflow = StateGraph(KnowledgeBaseState)

    # 添加节点
    workflow.add_node("assess", assess_query)
    workflow.add_node("hybrid_search", hybrid_search)
    workflow.add_node("rerank", rerank)
    workflow.add_node("inject_memory", inject_memory)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("direct_answer", direct_answer)
    workflow.add_node("store_memory", store_memory)

    # 设置入口
    workflow.set_entry_point("assess")

    # 评估后路由
    workflow.add_conditional_edges(
        "assess",
        route_after_assess,
        {
            "hybrid_search": "hybrid_search",
            "direct_answer": "direct_answer",
        },
    )

    # RAG 路径
    workflow.add_edge("hybrid_search", "rerank")
    workflow.add_edge("rerank", "inject_memory")
    workflow.add_edge("inject_memory", "generate_answer")
    workflow.add_edge("generate_answer", "store_memory")

    # 反应式路径
    workflow.add_edge("direct_answer", "store_memory")

    # 终点
    workflow.add_edge("store_memory", END)

    return workflow.compile()


def _get_or_init_workflow():
    """获取或初始化编译好的工作流图（模块级单例，避免每次请求重复编译）"""
    global _workflow
    if _workflow is None:
        _workflow = create_knowledge_base_workflow()
    return _workflow


# ============================================================
# 运行入口
# ============================================================

def run_knowledge_base(
    user_query: str,
    session_id: str = "default",
) -> Dict[str, Any]:
    """
    运行知识库系统

    Args:
        user_query: 用户查询
        session_id: 会话ID

    Returns:
        完整状态字典
    """
    agent = _get_or_init_workflow()

    trace_id = uuid.uuid4().hex[:8]
    token = _trace_id.set(trace_id)
    _log("开始处理 session=%s query=%s", session_id, user_query)

    initial_state = {
        "user_query": user_query,
        "session_id": session_id,
        "query_type": None,
        "processing_mode": None,
        "bm25_results": None,
        "vector_results": None,
        "fused_results": None,
        "reranked_results": None,
        "final_docs": None,
        "short_term_memory": None,
        "long_term_context": None,
        "messages": [],
        "final_response": None,
        "sources": None,
        "current_phase": "assess",
        "error": None,
        "evaluation_trace": None,
    }

    try:
        result = agent.invoke(initial_state)
    finally:
        _trace_id.reset(token)

    _log("处理完成 session=%s", session_id)
    return result


def reset_session(session_id: str = "default"):
    """重置会话（清空短期记忆）"""
    if session_id in _stm_instances:
        _stm_instances[session_id].clear()
        del _stm_instances[session_id]
