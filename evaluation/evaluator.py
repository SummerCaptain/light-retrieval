# -*- coding: utf-8 -*-
"""
RAG 效果评估体系 - 集成 LangSmith + OpenEval

基于 LangSmith 追踪和 OpenEvals 自动化评估，持续监控 RAG 管道的检索与生成质量。

评估维度：
1. 检索层：Hit Rate@K / MRR（基于 Golden Dataset 人工标注）
2. 生成层：Answer Relevance / RAG Helpfulness / RAG Groundedness / Retrieval Relevance（OpenEvals LLM-as-Judge）
3. 分组分析：按查询类型（knowledge_retrieval / comparative_analysis / procedural_guide）统计

使用方式：
    python run_rag_evaluation.py                         # 仅检索评估
    python run_rag_evaluation.py --full-rag              # 完整 RAG 评估（含 OpenEvals）
    python run_rag_evaluation.py --langsmith             # 上报 LangSmith
    python run_rag_evaluation.py --full-rag --langsmith  # 完整评估 + LangSmith 追踪
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Example, Run

from config.settings import (
    GOLDEN_DATASET_PATH,
    EVAL_HIT_RATE_K,
    EVAL_REPORTS_DIR,
    EVAL_LLM_MODEL,
)
from evaluation.metrics import (
    hit_rate,
    mrr,
    keyword_coverage,
    compute_batch_metrics,
    compute_metrics_by_type,
)


# ============================================================
# Golden Dataset 加载
# ============================================================

def load_golden_dataset(dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载 Golden Dataset

    Args:
        dataset_path: 数据集文件路径，默认使用配置中的路径

    Returns:
        解析后的 JSON 字典
    """
    if dataset_path:
        path = dataset_path
    elif GOLDEN_DATASET_PATH:
        path = str(GOLDEN_DATASET_PATH)
    else:
        from config.settings import PROJECT_ROOT
        path = str(PROJECT_ROOT / "evaluation" / "golden_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 检索管路执行
# ============================================================

def run_retrieval_for_query(query: str) -> Dict[str, Any]:
    """
    对单条查询执行检索管路（不经过 LLM），只评估检索质量
    流程：BM25 + Vector -> RRF 融合 -> 重排序

    Args:
        query: 用户查询

    Returns:
        包含各阶段检索结果的字典
    """
    from core.workflow import _get_or_init_retriever, _get_or_init_reranker

    retriever = _get_or_init_retriever()
    reranker = _get_or_init_reranker()

    # 混合检索
    fused_results = retriever.search(query, top_k=20)

    # 重排序
    reranked_results = reranker.rerank(
        query=query,
        documents=fused_results,
        top_k=EVAL_HIT_RATE_K,
    )

    # 提取文档ID
    fused_doc_ids = [f"{doc['doc_id']}_{doc['chunk_id']}" for doc in fused_results]
    reranked_doc_ids = [f"{doc['doc_id']}_{doc['chunk_id']}" for doc in reranked_results]

    # 拼接检索上下文（供 OpenEvals 检索相关性评估使用）
    context_parts = []
    for i, doc in enumerate(reranked_results[:EVAL_HIT_RATE_K], 1):
        source = doc.get("metadata", {}).get("source_file", "")
        content = doc.get("content", "")
        context_parts.append(f"[文档{i}](来源: {source})\n{content}")
    context_text = "\n\n".join(context_parts)

    return {
        "fused_results": fused_results,
        "reranked_results": reranked_results,
        "fused_doc_ids": fused_doc_ids,
        "reranked_doc_ids": reranked_doc_ids,
        "context": context_text,
        "fused_count": len(fused_results),
        "reranked_count": len(reranked_results),
    }


def run_full_rag_for_query(query: str) -> Dict[str, Any]:
    """
    对单条查询执行完整 RAG 管路（含 LLM 生成），用于端到端评估

    Args:
        query: 用户查询

    Returns:
        包含检索结果和生成回答的字典
    """
    from core.workflow import run_knowledge_base

    result = run_knowledge_base(query, session_id="eval_session")

    final_docs = result.get("final_docs") or []
    doc_ids = [f"{d['doc_id']}_{d['chunk_id']}" for d in final_docs]

    # 拼接检索上下文
    context_parts = []
    for i, doc in enumerate(final_docs, 1):
        source = doc.get("metadata", {}).get("source_file", "")
        content = doc.get("content", "")
        context_parts.append(f"[文档{i}](来源: {source})\n{content}")
    context_text = "\n\n".join(context_parts)

    return {
        "retrieved_docs": final_docs,
        "retrieved_doc_ids": doc_ids,
        "final_response": result.get("final_response", ""),
        "processing_mode": result.get("processing_mode", ""),
        "query_type": result.get("query_type", ""),
        "sources": result.get("sources", []),
        "context": context_text,
    }


# ============================================================
# LangSmith 集成
# ============================================================

def _init_langsmith_client():
    """初始化 LangSmith 客户端"""
    api_key = os.getenv("LANGSMITH_API_KEY")
    enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"

    if not api_key or not enabled:
        return None

    try:
        return Client(api_key=api_key)
    except Exception as e:
        print(f"[LangSmith] 客户端初始化失败: {e}")
        return None


def create_langsmith_dataset(
    client,
    dataset_name: str,
    golden_data: Dict[str, Any],
) -> str:
    """
    在 LangSmith 上创建/更新 Golden Dataset

    参考 2-langsmith_testing_evaluation.py 的 create_test_dataset 模式

    Args:
        client: LangSmith Client 实例
        dataset_name: 数据集名称
        golden_data: Golden Dataset JSON 数据

    Returns:
        数据集名称
    """
    if client is None:
        return dataset_name

    # 检查是否已存在
    try:
        existing = client.read_dataset(dataset_name=dataset_name)
        existing_examples = list(client.list_examples(dataset_name=dataset_name))
        if len(existing_examples) == len(golden_data["questions"]):
            print(f"[LangSmith] 数据集已存在且用例数量一致: {dataset_name} ({len(existing_examples)} 条)")
            return dataset_name
        else:
            print(f"[LangSmith] 数据集已存在但用例数量不同（现有 {len(existing_examples)} 条，"
                  f"预期 {len(golden_data['questions'])} 条），将删除重建")
            client.delete_dataset(dataset_name=dataset_name)
    except Exception:
        print(f"[LangSmith] 数据集不存在，将创建: {dataset_name}")

    # 创建数据集
    try:
        client.create_dataset(
            dataset_name=dataset_name,
            description=golden_data["metadata"]["description"],
        )
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise

    # 添加用例
    added = 0
    for q in golden_data["questions"]:
        try:
            client.create_example(
                inputs={
                    "query": q["query"],
                    "query_id": q["id"],
                    "query_type": q["query_type"],
                },
                outputs={
                    "relevant_doc_ids": q["relevant_doc_ids"],
                    "relevant_source_files": q["relevant_source_files"],
                    "expected_keywords": q["expected_keywords"],
                    "query_type": q["query_type"],
                },
                dataset_name=dataset_name,
            )
            added += 1
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  警告: 添加 {q['id']} 失败: {e}")

    print(f"[LangSmith] 数据集准备完成: {dataset_name} ({added} 条)")
    return dataset_name


# ============================================================
# LangSmith 评估器定义
# 参考 12-openevals_evaluators.py 的 create_llm_as_judge 模式
# 评估器为普通函数，签名 (run, example) -> dict
# ============================================================

def _get_example_outputs(example):
    """安全获取 example 的 outputs，支持 Example 对象和字典"""
    try:
        if hasattr(example, 'outputs') and example.outputs is not None:
            return example.outputs if isinstance(example.outputs, dict) else {}
        if isinstance(example, dict):
            if "outputs" in example:
                return example["outputs"] if isinstance(example["outputs"], dict) else {}
        if hasattr(example, '__dict__') and 'outputs' in example.__dict__:
            outputs = example.__dict__['outputs']
            return outputs if isinstance(outputs, dict) else {}
    except Exception:
        pass
    return {}


def _make_hit_rate_evaluator(k: int = 5):
    """创建 Hit Rate@K 评估器函数"""
    def hit_rate_evaluator(run, example):
        try:
            # 直接访问 run.outputs（参考 2-langsmith_testing_evaluation.py 模式）
            retrieved = run.outputs.get("retrieved_docs", []) if run.outputs else []
            example_outputs = _get_example_outputs(example)
            relevant = example_outputs.get("relevant_source_files", [])
            keywords = example_outputs.get("expected_keywords", [])

            hr = hit_rate(retrieved, relevant, keywords, k=k)

            return {
                "key": "hit_rate",
                "score": hr,
                "comment": f"Top-{k} 命中: {'是' if hr > 0 else '否'}",
            }
        except Exception as e:
            return {"key": "hit_rate", "score": 0, "comment": f"评估错误: {e}"}
    return hit_rate_evaluator


def _mrr_evaluator(run, example):
    """MRR 评估器：正确文档在排序中的位置倒数"""
    try:
        retrieved = run.outputs.get("retrieved_docs", []) if run.outputs else []
        example_outputs = _get_example_outputs(example)
        relevant = example_outputs.get("relevant_source_files", [])
        keywords = example_outputs.get("expected_keywords", [])

        mr = mrr(retrieved, relevant, keywords)

        return {
            "key": "mrr",
            "score": mr,
            "comment": f"MRR = {mr:.4f}",
        }
    except Exception as e:
        return {"key": "mrr", "score": 0, "comment": f"评估错误: {e}"}


def _keyword_coverage_evaluator(run, example):
    """关键词覆盖率评估器：回答中包含了多少期望关键词"""
    try:
        response = run.outputs.get("final_response", "") if run.outputs else ""
        example_outputs = _get_example_outputs(example)
        expected_keywords = example_outputs.get("expected_keywords", [])

        if not expected_keywords or not response:
            return {"key": "keyword_coverage", "score": None, "comment": "缺少关键词或回答"}

        kc = keyword_coverage(response, expected_keywords)
        return {
            "key": "keyword_coverage",
            "score": kc,
            "comment": f"覆盖率: {kc:.2%}",
        }
    except Exception as e:
        return {"key": "keyword_coverage", "score": 0, "comment": f"评估错误: {e}"}


def _create_openevals_evaluators():
    """
    创建 OpenEvals 评估器列表

    选取 RAG 场景最适用的 4 个评估器：
    - answer_relevance: 回答与问题的相关性
    - rag_helpfulness: 回答的帮助性
    - rag_groundedness: 回答是否基于检索文档（幻觉检测）
    - retrieval_relevance: 检索内容与问题的相关性

    注意：LangSmith 的 evaluate() 只会自动注入 inputs / outputs / reference_outputs，
    不会注入 context。而 RAG_GROUNDEDNESS_PROMPT 和 RAG_RETRIEVAL_RELEVANCE_PROMPT
    的提示词里包含 {context} 占位符，因此必须把 OpenEvals 评估器包一层 (run, example)
    函数，显式从 run.outputs 中取出 context 再传入，否则会报 KeyError: 'context'。
    """
    from langchain_community.chat_models import ChatTongyi
    from openevals.llm import create_llm_as_judge
    from openevals.prompts import (
        ANSWER_RELEVANCE_PROMPT,
        RAG_HELPFULNESS_PROMPT,
        RAG_GROUNDEDNESS_PROMPT,
        RAG_RETRIEVAL_RELEVANCE_PROMPT,
    )

    # max_retries=1：403 配额耗尽这类错误重试也不会成功，默认 10 次只会刷屏
    eval_llm = ChatTongyi(
        model_name=EVAL_LLM_MODEL,
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        temperature=0,
        max_retries=1,
    )

    # 创建四个 LLM-as-Judge 原函数
    judges = {
        "answer_relevance": create_llm_as_judge(
            prompt=ANSWER_RELEVANCE_PROMPT,
            feedback_key="answer_relevance",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        "rag_helpfulness": create_llm_as_judge(
            prompt=RAG_HELPFULNESS_PROMPT,
            feedback_key="rag_helpfulness",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        "rag_groundedness": create_llm_as_judge(
            prompt=RAG_GROUNDEDNESS_PROMPT,
            feedback_key="rag_groundedness",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
        "retrieval_relevance": create_llm_as_judge(
            prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
            feedback_key="retrieval_relevance",
            judge=eval_llm,
            continuous=True,
            use_reasoning=False,
        ),
    }

    # 各评估器需要填充的字段：inputs(问题) / outputs(回答) / context(检索上下文)
    specs = {
        "answer_relevance": {"inputs": True, "outputs": True, "context": False},
        "rag_helpfulness": {"inputs": True, "outputs": True, "context": False},
        "rag_groundedness": {"inputs": False, "outputs": True, "context": True},
        "retrieval_relevance": {"inputs": True, "outputs": False, "context": True},
    }

    def _extract_query(example):
        inputs = getattr(example, "inputs", None) or {}
        if isinstance(inputs, dict):
            return inputs.get("query", "")
        return inputs

    def _wrap(name, judge_fn, spec):
        def evaluator(run, example):
            outputs = run.outputs or {}
            if not isinstance(outputs, dict):
                outputs = {}
            kwargs = {}
            if spec["inputs"]:
                kwargs["inputs"] = _extract_query(example)
            if spec["outputs"]:
                kwargs["outputs"] = outputs.get("final_response", "")
            if spec["context"]:
                kwargs["context"] = outputs.get("context", "")
            return judge_fn(**kwargs)

        return evaluator

    return [_wrap(name, judges[name], specs[name]) for name in specs]


# ============================================================
# LangSmith 评估运行
# ============================================================

def run_langsmith_retrieval_eval(
    client,
    dataset_name: str,
    experiment_name: str,
    eval_k: int = 5,
):
    """
    通过 LangSmith evaluate 运行检索评估实验（仅检索，不调用 LLM）

    Args:
        client: LangSmith Client
        dataset_name: 数据集名称
        experiment_name: 实验名称
        eval_k: Hit Rate 的 Top-K

    Returns:
        评估结果
    """
    def target_fn(inputs: dict) -> dict:
        """
        检索目标函数

        LangSmith evaluate 传入的 inputs 就是 create_example 时的 inputs 字典
        """
        query = inputs.get("query", "")
        query_id = inputs.get("query_id", "unknown")

        retrieval_result = run_retrieval_for_query(query)

        return {
            "query_id": query_id,
            "query": query,
            "retrieved_docs": retrieval_result["reranked_results"],
            "retrieved_doc_ids": retrieval_result["reranked_doc_ids"],
            "fused_doc_ids": retrieval_result["fused_doc_ids"],
            "context": retrieval_result["context"],
        }

    evaluators = [
        _make_hit_rate_evaluator(k=eval_k),
        _mrr_evaluator,
    ]

    print(f"[LangSmith] 启动检索评估实验: {experiment_name}")
    print(f"  数据集: {dataset_name}")
    print(f"  评估器: Hit Rate@{eval_k}, MRR")

    results = evaluate(
        target_fn,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=experiment_name,
        max_concurrency=1,
    )

    return results


def run_langsmith_full_rag_eval(
    client,
    dataset_name: str,
    experiment_name: str,
    eval_k: int = 5,
):
    """
    通过 LangSmith evaluate 运行完整 RAG 评估实验（含 LLM 生成 + OpenEvals）

    评估器包括：
    - Hit Rate@K / MRR / 关键词覆盖率（确定性指标）
    - Answer Relevance / RAG Helpfulness / RAG Groundedness / Retrieval Relevance（OpenEvals LLM-as-Judge）

    Args:
        client: LangSmith Client
        dataset_name: 数据集名称
        experiment_name: 实验名称
        eval_k: Hit Rate 的 Top-K

    Returns:
        评估结果
    """
    def target_fn(inputs: dict) -> dict:
        """
        完整 RAG 目标函数

        返回检索结果和 LLM 生成回答，供 OpenEvals 评估器使用
        """
        query = inputs.get("query", "")
        query_id = inputs.get("query_id", "unknown")

        rag_result = run_full_rag_for_query(query)

        return {
            "query_id": query_id,
            "query": query,
            "retrieved_docs": rag_result["retrieved_docs"],
            "retrieved_doc_ids": rag_result["retrieved_doc_ids"],
            "final_response": rag_result["final_response"],
            "processing_mode": rag_result["processing_mode"],
            "context": rag_result["context"],
            # OpenEvals 评估器通过这些 key 提取数据
            "output": rag_result["final_response"],  # answer_relevance / rag_helpfulness
            "answer": rag_result["final_response"],  # rag_groundedness
        }

    # 确定性指标评估器
    evaluators = [
        _make_hit_rate_evaluator(k=eval_k),
        _mrr_evaluator,
        _keyword_coverage_evaluator,
    ]

    # OpenEvals LLM-as-Judge 评估器
    try:
        openevals_evaluators = _create_openevals_evaluators()
        evaluators.extend(openevals_evaluators)
        openevals_names = ["answer_relevance", "rag_helpfulness", "rag_groundedness", "retrieval_relevance"]
    except Exception as e:
        print(f"[警告] OpenEvals 评估器创建失败（跳过）: {e}")
        openevals_names = []

    eval_names = ["Hit Rate", "MRR", "Keyword Coverage"] + openevals_names
    print(f"[LangSmith] 启动完整 RAG 评估实验: {experiment_name}")
    print(f"  数据集: {dataset_name}")
    print(f"  评估器: {', '.join(eval_names)}")

    results = evaluate(
        target_fn,
        data=dataset_name,
        evaluators=evaluators,
        client=client,
        experiment_prefix=experiment_name,
        max_concurrency=1,
    )

    return results


# ============================================================
# 本地评估（不依赖 LangSmith）
# ============================================================

def run_local_evaluation(
    golden_data: Dict[str, Any],
    eval_k: int = 5,
    use_full_rag: bool = False,
) -> Dict[str, Any]:
    """
    本地运行 RAG 评估，无需 LangSmith

    Args:
        golden_data: Golden Dataset JSON 数据
        eval_k: Hit Rate 的 Top-K
        use_full_rag: 是否使用完整 RAG 管路（含 LLM 生成），否则只评估检索

    Returns:
        评估结果字典
    """
    questions = golden_data["questions"]
    all_results = []

    print(f"\n[评估] 开始评估，共 {len(questions)} 条查询")
    print(f"  评估模式: {'完整RAG' if use_full_rag else '仅检索'}")
    print(f"  Hit Rate@K: K={eval_k}")
    print("-" * 60)

    for i, q in enumerate(questions):
        query = q["query"]
        relevant_source_files = q["relevant_source_files"]
        expected_keywords = q.get("expected_keywords", [])
        query_type = q.get("query_type", "unknown")

        print(f"  [{i+1}/{len(questions)}] {query}", end=" ... ")

        try:
            if use_full_rag:
                rag_result = run_full_rag_for_query(query)
                retrieved_docs = rag_result["retrieved_docs"]
                final_response = rag_result["final_response"]
            else:
                retrieval_result = run_retrieval_for_query(query)
                retrieved_docs = retrieval_result["reranked_results"]
                final_response = ""

            hr = hit_rate(retrieved_docs, relevant_source_files, expected_keywords, k=eval_k)
            mr = mrr(retrieved_docs, relevant_source_files, expected_keywords)

            result_item = {
                "query_id": q["id"],
                "query": query,
                "query_type": query_type,
                "retrieved_docs": retrieved_docs,
                "relevant_source_files": relevant_source_files,
                "expected_keywords": expected_keywords,
                "final_response": final_response,
                "hit_rate": hr,
                "mrr": mr,
            }

            # 关键词覆盖率（仅完整RAG模式）
            if use_full_rag and expected_keywords and final_response:
                result_item["keyword_coverage"] = keyword_coverage(
                    final_response, expected_keywords
                )

            hit_mark = "HIT" if hr > 0 else "MISS"
            print(f"{hit_mark} (HR={hr:.1f}, MRR={mr:.3f})")

            all_results.append(result_item)

        except Exception as e:
            print(f"ERROR: {e}")
            all_results.append({
                "query_id": q["id"],
                "query": query,
                "query_type": query_type,
                "retrieved_docs": [],
                "relevant_source_files": relevant_source_files,
                "expected_keywords": expected_keywords,
                "final_response": "",
                "hit_rate": 0.0,
                "mrr": 0.0,
                "error": str(e),
            })

    # 计算汇总指标
    print()
    print("=" * 60)
    print("[评估] 计算汇总指标...")

    overall = compute_batch_metrics(all_results, k=eval_k)
    by_type = compute_metrics_by_type(all_results, k=eval_k)

    report = {
        "timestamp": datetime.now().isoformat(),
        "eval_k": eval_k,
        "use_full_rag": use_full_rag,
        "total_questions": len(questions),
        "overall": {k: v for k, v in overall.items() if k != "details"},
        "by_query_type": {
            qtype: {k: v for k, v in metrics.items() if k != "details"}
            for qtype, metrics in by_type.items()
        },
        "details": all_results,
    }

    return report


# ============================================================
# 报告输出
# ============================================================

def print_evaluation_report(report: Dict[str, Any]):
    """打印评估报告到控制台"""
    print()
    print("=" * 60)
    print("  RAG 检索效果评估报告")
    print("=" * 60)
    print(f"  时间: {report['timestamp']}")
    print(f"  查询数: {report['total_questions']}")
    print(f"  Hit Rate@{report['eval_k']}: {report['overall']['hit_rate']:.4f}")
    print(f"  MRR:            {report['overall']['mrr']:.4f}")

    if "keyword_coverage" in report["overall"]:
        print(f"  关键词覆盖率:   {report['overall']['keyword_coverage']:.4f}")

    print()
    print("-" * 60)
    print("  按查询类型分组:")
    print("-" * 60)

    type_names = {
        "knowledge_retrieval": "知识检索",
        "comparative_analysis": "对比分析",
        "procedural_guide": "步骤指南",
    }

    for qtype, metrics in report["by_query_type"].items():
        label = type_names.get(qtype, qtype)
        count = metrics.get("count", metrics.get("total", 0))
        print(f"  [{label}] ({count} 条)")
        print(f"    Hit Rate@{report['eval_k']}: {metrics['hit_rate']:.4f}")
        print(f"    MRR:            {metrics['mrr']:.4f}")
        if "keyword_coverage" in metrics:
            print(f"    关键词覆盖率:   {metrics['keyword_coverage']:.4f}")
        print()

    # 输出未命中的查询
    missed = [d for d in report["details"] if d.get("hit_rate", 0) == 0]
    if missed:
        print("-" * 60)
        print(f"  未命中查询 ({len(missed)} 条):")
        print("-" * 60)
        for d in missed:
            print(f"  - [{d['query_id']}] {d['query']}")
            print(f"    期望来源: {d['relevant_source_files']}")
            retrieved = d.get('retrieved_docs', [])
            if retrieved:
                top_sources = [doc.get("metadata", {}).get("source_file", "") for doc in retrieved[:report['eval_k']]]
                print(f"    实际Top-{report['eval_k']}: {top_sources}")
            else:
                print(f"    实际Top-{report['eval_k']}: (无检索结果)")
            if d.get("error"):
                print(f"    错误: {d['error']}")
        print()

    print("=" * 60)


def save_evaluation_report(
    report: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> str:
    """
    保存评估报告到 JSON 文件

    Args:
        report: 评估报告字典
        output_dir: 输出目录，默认为 evaluation/reports/

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        if EVAL_REPORTS_DIR:
            output_dir = str(EVAL_REPORTS_DIR)
        elif GOLDEN_DATASET_PATH:
            output_dir = str(GOLDEN_DATASET_PATH.parent / "reports")
        else:
            from config.settings import PROJECT_ROOT
            output_dir = str(PROJECT_ROOT / "evaluation" / "reports")

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"eval_report_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[评估] 报告已保存: {filepath}")
    return filepath
