#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
知识库系统 - 交互式入口

直接输入问题即可获得回答，输入 q 或 exit 退出。
"""

from datetime import datetime
from config.settings import LLM_MODEL_NAME
from core.workflow import run_knowledge_base, reset_session


def main():
    print("=" * 55)
    print("  知识库系统 - 混合智能体 RAG")
    print(f"  模型: {LLM_MODEL_NAME} | 检索: BM25 + Vector + RRF")
    print("=" * 55)
    print()
    print("直接输入问题即可，例如：")
    print("  - 什么是ETF？")
    print("  - 降息对债券基金有什么影响？")
    print("  - 基金定投的操作步骤是什么？")
    print("  - 可转债中签后怎么操作？")
    print("  - ETF和普通指数基金哪个更好？")
    print()
    print("输入 q 或 exit 退出")
    print("-" * 55)

    session_id = "session_1"

    while True:
        try:
            user_query = input("\n请输入问题: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break

        if not user_query:
            continue

        if user_query.lower() in ("q", "exit", "quit"):
            print("再见!")
            break

        print("处理中...")

        start_time = datetime.now()
        result = run_knowledge_base(user_query, session_id=session_id)
        end_time = datetime.now()

        # 输出处理模式
        mode = result.get("processing_mode", "未知")
        query_type = result.get("query_type", "未知")
        if mode == "reactive":
            print(f"[模式: 直接回答 | 类型: {query_type}]")
        else:
            print(f"[模式: RAG 检索 | 类型: {query_type}]")

        # 输出检索结果概况
        final_docs = result.get("final_docs") or []
        if final_docs:
            print(f"[检索: {len(final_docs)} 条文档]", end="")
            sources_short = []
            for doc in final_docs[:3]:
                src = doc.get("metadata", {}).get("source_file", "")
                if src and src not in sources_short:
                    sources_short.append(src)
            if sources_short:
                print(f" | 来源: {', '.join(sources_short)}", end="")
            print()

        # 输出回答
        print(f"\nAI: {result.get('final_response', '未生成响应')}")

        elapsed = (end_time - start_time).total_seconds()
        print(f"\n[耗时: {elapsed:.2f}s]")


if __name__ == "__main__":
    main()
