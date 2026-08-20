# -*- coding: utf-8 -*-
"""
RAG 检索效果评估指标

实现 Hit Rate 和 MRR 两个核心检索指标，用于持续监控 RAG 管道检索质量。

- Hit Rate: 正确文档是否出现在 Top-K 结果中（召回率视角）
- MRR: 正确文档在排序中的位置倒数均值（排序质量视角）
"""

from typing import Dict, List, Any


def hit_rate(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    k: int = 5,
) -> float:
    """
    计算 Hit Rate@K

    只要 Top-K 结果中包含任意一个相关文档，即为命中。

    Args:
        retrieved_doc_ids: 检索返回的文档ID列表（已按相关性排序）
        relevant_doc_ids: 人工标注的相关文档ID列表
        k: 只看前 K 个结果

    Returns:
        1.0 表示命中，0.0 表示未命中
    """
    if not relevant_doc_ids:
        return 0.0

    top_k_ids = retrieved_doc_ids[:k]
    for doc_id in relevant_doc_ids:
        if doc_id in top_k_ids:
            return 1.0
    return 0.0


def mrr(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)

    找到第一个相关文档在检索结果中的排名 r，返回 1/r。
    未找到则返回 0。

    Args:
        retrieved_doc_ids: 检索返回的文档ID列表（已按相关性排序）
        relevant_doc_ids: 人工标注的相关文档ID列表

    Returns:
        1/r（第一个相关文档排名的倒数），未命中返回 0.0
    """
    if not relevant_doc_ids:
        return 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / rank
    return 0.0


def keyword_coverage(
    response: str,
    expected_keywords: List[str],
) -> float:
    """
    计算期望关键词覆盖率

    检查生成回答中包含了多少期望关键词，用于辅助评估生成质量。

    Args:
        response: 模型生成的回答
        expected_keywords: 期望包含的关键词列表

    Returns:
        覆盖率 (0.0 ~ 1.0)
    """
    if not expected_keywords:
        return 0.0

    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return found / len(expected_keywords)


def compute_batch_metrics(
    results: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Any]:
    """
    批量计算评估指标

    对一组查询的检索结果计算汇总指标，输出各指标的均值和逐条明细。

    Args:
        results: 每个元素包含:
            - query_id: 查询ID
            - retrieved_doc_ids: 检索返回的文档ID列表
            - relevant_doc_ids: 人工标注的相关文档ID列表
            - final_response: 模型生成的回答（可选）
            - expected_keywords: 期望关键词列表（可选）
        k: Hit Rate 的 Top-K

    Returns:
        汇总指标字典，包含:
            - hit_rate: 平均 Hit Rate@K
            - mrr: 平均 MRR
            - keyword_coverage: 平均关键词覆盖率（如有）
            - total: 查询总数
            - details: 逐条指标明细
    """
    hit_rates = []
    mrrs = []
    kw_coverages = []
    details = []

    for item in results:
        retrieved = item.get("retrieved_doc_ids", [])
        relevant = item.get("relevant_doc_ids", [])
        response = item.get("final_response", "")
        keywords = item.get("expected_keywords", [])

        hr = hit_rate(retrieved, relevant, k=k)
        mr = mrr(retrieved, relevant)

        hit_rates.append(hr)
        mrrs.append(mr)

        detail = {
            "query_id": item.get("query_id", "unknown"),
            "hit_rate": hr,
            "mrr": mr,
        }

        # 关键词覆盖率（可选）
        if keywords and response:
            kc = keyword_coverage(response, keywords)
            kw_coverages.append(kc)
            detail["keyword_coverage"] = kc

        details.append(detail)

    summary = {
        "hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0.0,
        "mrr": sum(mrrs) / len(mrrs) if mrrs else 0.0,
        "total": len(results),
        "k": k,
        "details": details,
    }

    if kw_coverages:
        summary["keyword_coverage"] = sum(kw_coverages) / len(kw_coverages)

    return summary


def compute_metrics_by_type(
    results: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    按查询类型分组计算指标

    分别统计 knowledge_retrieval / comparative_analysis / procedural_guide
    三种查询类型的 Hit Rate 和 MRR，定位不同场景的检索短板。

    Args:
        results: 同 compute_batch_metrics，额外需包含 query_type 字段
        k: Hit Rate 的 Top-K

    Returns:
        按类型分组的指标字典，key 为查询类型，value 为该类型的汇总指标
    """
    type_groups: Dict[str, List[Dict[str, Any]]] = {}

    for item in results:
        qtype = item.get("query_type", "unknown")
        if qtype not in type_groups:
            type_groups[qtype] = []
        type_groups[qtype].append(item)

    type_metrics = {}
    for qtype, items in type_groups.items():
        type_metrics[qtype] = compute_batch_metrics(items, k=k)
        type_metrics[qtype]["count"] = len(items)

    return type_metrics
