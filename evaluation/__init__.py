# -*- coding: utf-8 -*-
"""
RAG 效果评估模块

提供基于 LangSmith + OpenEval 的 RAG 检索与生成效果评估能力：
- Golden Dataset 管理（50+ 真实业务问题）
- Hit Rate / MRR / 关键词覆盖率 确定性指标
- OpenEvals LLM-as-Judge 评估（answer_relevance, rag_helpfulness, rag_groundedness, retrieval_relevance）
- LangSmith 实验追踪与版本对比
- 按查询类型分组分析
"""

from evaluation.metrics import (
    hit_rate,
    mrr,
    keyword_coverage,
    compute_batch_metrics,
    compute_metrics_by_type,
)
from evaluation.evaluator import (
    load_golden_dataset,
    run_retrieval_for_query,
    run_full_rag_for_query,
    run_local_evaluation,
    run_langsmith_retrieval_eval,
    run_langsmith_full_rag_eval,
    create_langsmith_dataset,
    print_evaluation_report,
    save_evaluation_report,
    _make_hit_rate_evaluator,
    _mrr_evaluator,
    _keyword_coverage_evaluator,
)

__all__ = [
    "hit_rate",
    "mrr",
    "keyword_coverage",
    "compute_batch_metrics",
    "compute_metrics_by_type",
    "load_golden_dataset",
    "run_retrieval_for_query",
    "run_full_rag_for_query",
    "run_local_evaluation",
    "run_langsmith_retrieval_eval",
    "run_langsmith_full_rag_eval",
    "create_langsmith_dataset",
    "print_evaluation_report",
    "save_evaluation_report",
]
