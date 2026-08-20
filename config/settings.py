# -*- coding: utf-8 -*-
"""
全局配置管理

集中管理知识库系统的所有配置项，包括模型参数、检索参数、路径配置等。
"""

import os
from pathlib import Path

# ============================================================
# 项目根路径
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ============================================================
# 数据路径
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"      # 原始知识文档
CHUNKS_DIR = DATA_DIR / "chunks"                       # DocParser 分块存储
VECTOR_STORE_DIR = DATA_DIR / "vector_store"           # FAISS 索引存储

# ============================================================
# LLM 配置
# ============================================================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3.7-max")
LLM_TEMPERATURE = 0.1
# 评估裁判 LLM（OpenEvals LLM-as-Judge）
EVAL_LLM_MODEL = os.getenv("EVAL_LLM_MODEL", "qwen3.7-max")

# ============================================================
# Embedding 配置
# ============================================================
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 有 GPU 时改为 "cuda"
EMBEDDING_DIM = 1024       # bge-large-zh-v1.5 的向量维度

# ============================================================
# BGE-Reranker 配置
# ============================================================
RERANKER_MODEL_NAME = "BAAI/bge-reranker-large"
RERANKER_DEVICE = "cpu"    # 有 GPU 时改为 "cuda"
RERANKER_TOP_K = 5          # 重排序后保留的文档数

# ============================================================
# 文档分块配置
# ============================================================
CHUNK_SIZE = 512            # 每个分块的最大字符数
CHUNK_OVERLAP = 64          # 相邻分块的重叠字符数
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# ============================================================
# 混合检索配置
# ============================================================
BM25_TOP_K = 20             # BM25 检索返回的候选数
VECTOR_TOP_K = 20           # 向量检索返回的候选数
RRF_K = 60                  # RRF 融合算法的 k 参数（平滑因子）

# ============================================================
# 分级记忆配置
# ============================================================
SHORT_TERM_MEMORY_SIZE = 5  # 短期记忆保留的最近对话轮数
ES_INDEX_NAME = "kb_memory" # 长期记忆 ES 索引名

# ============================================================
# Elasticsearch 配置
# ============================================================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")

# ============================================================
# 评估配置（默认不启用）
# ============================================================
EVAL_ENABLED = os.getenv("RAG_EVAL_ENABLED", "").lower() in ("true", "1", "yes")

if EVAL_ENABLED:
    GOLDEN_DATASET_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    EVAL_HIT_RATE_K = int(os.getenv("RAG_EVAL_K", "5"))          # Hit Rate 评估的 Top-K
    EVAL_REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"   # 评估报告输出目录
    LANGSMITH_DATASET_NAME = os.getenv("RAG_LANGSMITH_DATASET", "rag-evaluation-golden-dataset")
else:
    # 未启用时保留占位值，避免模块导入报错
    GOLDEN_DATASET_PATH = None
    EVAL_HIT_RATE_K = 5
    EVAL_REPORTS_DIR = None
    LANGSMITH_DATASET_NAME = None

# ============================================================
# 路径自动创建
# ============================================================
for _dir in [DATA_DIR, KNOWLEDGE_BASE_DIR, CHUNKS_DIR, VECTOR_STORE_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)