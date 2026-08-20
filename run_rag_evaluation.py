#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RAG 效果评估运行入口

基于 LangSmith 和 OpenEval 的 RAG 检索与生成效果评估体系。
使用 Golden Dataset（50+ 真实业务问题）持续监控检索与生成质量。

评估维度：
- 检索层: Hit Rate@K / MRR（确定性指标）
- 生成层: Answer Relevance / Helpfulness / Groundedness / Retrieval Relevance（OpenEvals LLM-as-Judge）

用法:
  python run_rag_evaluation.py                    # 仅检索评估（本地）
  python run_rag_evaluation.py --full-rag         # 完整 RAG 评估（含 LLM + OpenEvals）
  python run_rag_evaluation.py --k 3              # Hit Rate@3
  python run_rag_evaluation.py --langsmith        # 上报 LangSmith（仅检索指标）
  python run_rag_evaluation.py --full-rag --langsmith  # 完整评估 + LangSmith + OpenEvals

环境变量:
  DASHSCOPE_API_KEY:  通义千问 API 密钥（完整 RAG 模式和 OpenEvals 需要）
  LANGSMITH_API_KEY:  LangSmith API 密钥（--langsmith 需要）
  LANGCHAIN_TRACING_V2: 设为 true 启用 LangSmith 追踪
"""

import json
import os
import sys
import argparse
from datetime import datetime

# 强制行缓冲，确保 print 立刻显示
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 项目根目录加入路径
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import warnings
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="RAG 效果评估 - 基于 LangSmith + OpenEval"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Hit Rate 的 Top-K 值 (默认: 5)"
    )
    parser.add_argument(
        "--full-rag", action="store_true",
        help="使用完整 RAG 管路（含 LLM 生成），否则只评估检索质量"
    )
    parser.add_argument(
        "--langsmith", action="store_true",
        help="将评估结果上报到 LangSmith 实验"
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Golden Dataset 文件路径（默认使用配置中的路径）"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG 效果评估系统")
    print("  基于 LangSmith + OpenEval")
    print("=" * 60)
    print()

    from evaluation.evaluator import (
        load_golden_dataset,
        run_local_evaluation,
        run_langsmith_retrieval_eval,
        run_langsmith_full_rag_eval,
        create_langsmith_dataset,
        _init_langsmith_client,
        print_evaluation_report,
        save_evaluation_report,
    )

    # 加载 Golden Dataset
    try:
        golden_data = load_golden_dataset(args.dataset)
        meta = golden_data["metadata"]
        print(f"[数据集] {meta['name']} v{meta['version']}")
        print(f"  问题数: {meta['total_questions']}")
        print(f"  领域:   {', '.join(meta['domains'])}")
        print()
    except FileNotFoundError as e:
        print(f"错误: 找不到 Golden Dataset - {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误: Golden Dataset 格式错误 - {e}")
        sys.exit(1)

    # LangSmith 模式
    if args.langsmith:
        client = _init_langsmith_client()
        if client is None:
            print("[警告] LangSmith 未启用或配置不正确，将回退到本地评估模式")
            args.langsmith = False
        else:
            dataset_name = "rag-evaluation-golden-dataset"
            experiment_name = f"rag-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # 创建/更新 LangSmith 数据集
            create_langsmith_dataset(client, dataset_name, golden_data)

            # 根据模式选择评估函数
            if args.full_rag:
                # 完整 RAG 评估：确定性指标 + OpenEvals LLM-as-Judge
                run_langsmith_full_rag_eval(
                    client, dataset_name, experiment_name, eval_k=args.k
                )
            else:
                # 仅检索评估：Hit Rate + MRR
                run_langsmith_retrieval_eval(
                    client, dataset_name, experiment_name, eval_k=args.k
                )

            print()
            print("=" * 60)
            print(f"  LangSmith 评估完成!")
            print(f"  实验名称: {experiment_name}")
            if args.full_rag:
                print(f"  评估器: Hit Rate@{args.k}, MRR, Keyword Coverage,")
                print(f"          Answer Relevance, RAG Helpfulness,")
                print(f"          RAG Groundedness, Retrieval Relevance")
            else:
                print(f"  评估器: Hit Rate@{args.k}, MRR")
            print(f"  查看结果: https://smith.langchain.com")
            print("=" * 60)
            return

    # 本地评估模式
    report = run_local_evaluation(
        golden_data,
        eval_k=args.k,
        use_full_rag=args.full_rag,
    )

    # 打印报告
    print_evaluation_report(report)

    # 保存报告
    save_evaluation_report(report)


if __name__ == "__main__":
    main()
