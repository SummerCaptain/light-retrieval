# -*- coding: utf-8 -*-
"""
校验 golden_dataset.json 的自洽性

1. 每个 relevant_doc_ids 指向真实存在的 chunk（hash + 序号）
2. 每个 expected_keywords 中的关键词都能在对应 chunk 文本中作为子串匹配到
3. 统计 query_type 分布是否与 metadata 声明一致

用法: python scripts/verify_golden_dataset.py
结果写入 scripts/_verify_result.txt（UTF-8）
"""
import json
import sys
from pathlib import Path

# 把项目根目录加入 sys.path，以便导入 retrieval 与 config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.doc_parser import DocParser
from config.settings import KNOWLEDGE_BASE_DIR

GOLDEN_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
OUT_PATH = PROJECT_ROOT / "scripts" / "_verify_result.txt"


def main():
    lines = []

    # 1. 解析知识库全部文档
    parser = DocParser()
    results = parser.parse_directory(str(KNOWLEDGE_BASE_DIR))
    # doc_id -> chunk 文本映射
    chunk_map = {}
    for r in results:
        if not r.doc_hash:
            continue
        for i, text in enumerate(r.chunks):
            chunk_map[f"{r.doc_hash}_{i}"] = text

    lines.append(f"解析到文档数: {len(results)}")
    lines.append(f"生成 chunk 总数: {len(chunk_map)}")
    lines.append("")

    # 2. 加载 golden dataset
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    meta = data["metadata"]
    lines.append(f"题目总数: {len(questions)}")
    lines.append("")

    # 3. 逐题校验
    fail_doc = []       # doc_id 不存在
    fail_kw = []        # 关键词未匹配
    type_counter = {}

    for q in questions:
        qid = q["id"]
        qtype = q["query_type"]
        type_counter[qtype] = type_counter.get(qtype, 0) + 1

        for doc_id in q["relevant_doc_ids"]:
            if doc_id not in chunk_map:
                fail_doc.append((qid, doc_id))
                continue
            chunk_text = chunk_map[doc_id]
            for kw in q["expected_keywords"]:
                if kw not in chunk_text:
                    fail_kw.append((qid, doc_id, kw))

    # 4. 输出统计
    lines.append("=" * 60)
    lines.append("[query_type 实际分布]")
    for k, v in sorted(type_counter.items()):
        lines.append(f"  {k}: {v}")
    lines.append(f"  (metadata 声明: {meta['query_type_distribution']})")
    lines.append("")

    lines.append("=" * 60)
    lines.append(f"[doc_id 不存在] 共 {len(fail_doc)} 处")
    for qid, doc_id in fail_doc:
        lines.append(f"  {qid}: {doc_id}")
    lines.append("")

    lines.append("=" * 60)
    lines.append(f"[关键词未匹配] 共 {len(fail_kw)} 处")
    for qid, doc_id, kw in fail_kw:
        lines.append(f"  {qid} [{doc_id}]: '{kw}'")
    lines.append("")

    lines.append("=" * 60)
    if not fail_doc and not fail_kw:
        lines.append("结果: 全部通过")
    else:
        lines.append("结果: 存在失败项，请修正")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("校验完成，结果见 scripts/_verify_result.txt")


if __name__ == "__main__":
    main()
