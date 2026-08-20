# -*- coding: utf-8 -*-
"""
LongTermMemoryManager - 长期记忆管理器

基于 Elasticsearch 的跨会话长期记忆存储。
记录用户历史交互、偏好标签、反馈信息，支持相似查询检索和用户画像构建。

特性：
- 自动创建 ES 索引（幂等，ik_smart 中文分词）
- 存储完整交互记录（查询、回答、检索文档、标签、反馈）
- 支持相似历史查询检索（ES 全文搜索）
- 支持用户偏好标签聚合
- ES 不可用时优雅降级（返回空结果，不抛异常）
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError

from config.settings import (
    ES_HOST,
    ES_USER,
    ES_PASSWORD,
    ES_INDEX_NAME,
)


# ES 索引 Mapping 定义
ES_INDEX_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                }
            }
        },
    },
    "mappings": {
        "properties": {
            "user_id":        {"type": "keyword"},
            "session_id":     {"type": "keyword"},
            "query":          {"type": "text", "analyzer": "ik_smart_analyzer"},
            "answer":         {"type": "text", "index": False},
            "query_type":     {"type": "keyword"},
            "retrieved_docs": {"type": "keyword"},
            "tags":           {"type": "keyword"},
            "feedback":       {"type": "keyword"},
            "timestamp":      {"type": "date"},
            "response_time_ms": {"type": "float"},
        }
    },
}


class LongTermMemoryManager:
    """长期记忆管理器 - Elasticsearch 存储"""

    def __init__(
        self,
        index_name: str = ES_INDEX_NAME,
        host: str = ES_HOST,
        user: str = ES_USER,
        password: str = ES_PASSWORD,
    ):
        """
        初始化长期记忆管理器

        Args:
            index_name: ES 索引名称
            host: ES 服务地址
            user: ES 用户名（可选）
            password: ES 密码（可选）
        """
        self.index_name = index_name
        self._connected = False

        # 构建 ES 客户端
        es_kwargs = {"hosts": [host]}
        if user and password:
            es_kwargs["basic_auth"] = (user, password)
        self._es = Elasticsearch(**es_kwargs)

        # 尝试连接并创建索引
        try:
            if self._es.ping():
                self._connected = True
                self._ensure_index()
        except ConnectionError:
            self._connected = False

    # ================================================================
    # 公开接口
    # ================================================================

    @property
    def is_connected(self) -> bool:
        """ES 是否可用"""
        return self._connected

    def store_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        query_type: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        response_time_ms: Optional[float] = None,
    ) -> Optional[str]:
        """
        存储一条交互记录

        Args:
            user_id: 用户ID
            session_id: 会话ID
            query: 用户查询
            answer: 助手回答
            query_type: 查询类型
            retrieved_docs: 检索到的文档ID列表
            tags: 自动标签
            response_time_ms: 响应耗时（毫秒）

        Returns:
            ES 文档 ID，连接失败时返回 None
        """
        if not self._connected:
            return None

        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "answer": answer,
            "query_type": query_type or "",
            "retrieved_docs": retrieved_docs or [],
            "tags": tags or [],
            "feedback": "neutral",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": response_time_ms or 0.0,
        }

        try:
            result = self._es.index(index=self.index_name, document=doc)
            return result["_id"]
        except Exception:
            return None

    def search_similar(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """
        搜索与当前查询相似的历史交互记录

        Args:
            user_id: 用户ID
            query: 查询文本
            top_k: 返回数量

        Returns:
            相似历史记录列表，ES 不可用时返回空列表
        """
        if not self._connected:
            return []

        try:
            result = self._es.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"user_id": user_id}},
                                {"match": {"query": {"query": query, "operator": "or"}}},
                            ]
                        }
                    },
                    "size": top_k,
                    "sort": [{"timestamp": {"order": "desc"}}],
                },
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception:
            return []

    def get_user_tags(self, user_id: str, limit: int = 10) -> List[str]:
        """
        获取用户最常访问的主题标签（偏好画像）

        Args:
            user_id: 用户ID
            limit: 返回标签数量

        Returns:
            标签列表（按频率降序），ES 不可用时返回空列表
        """
        if not self._connected:
            return []

        try:
            result = self._es.search(
                index=self.index_name,
                body={
                    "query": {
                        "bool": {
                            "must": [{"term": {"user_id": user_id}}],
                        }
                    },
                    "size": 0,
                    "aggs": {
                        "popular_tags": {
                            "terms": {
                                "field": "tags",
                                "size": limit,
                            }
                        }
                    },
                },
            )
            buckets = result["aggregations"]["popular_tags"]["buckets"]
            return [b["key"] for b in buckets]
        except Exception:
            return []

    def get_recent(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        """
        获取用户最近的交互记录

        Args:
            user_id: 用户ID
            limit: 返回数量

        Returns:
            最近交互记录列表（按时间降序），ES 不可用时返回空列表
        """
        if not self._connected:
            return []

        try:
            result = self._es.search(
                index=self.index_name,
                body={
                    "query": {"term": {"user_id": user_id}},
                    "size": limit,
                    "sort": [{"timestamp": {"order": "desc"}}],
                },
            )
            return [hit["_source"] for hit in result["hits"]["hits"]]
        except Exception:
            return []

    def update_feedback(self, doc_id: str, feedback: str) -> bool:
        """
        更新用户反馈

        Args:
            doc_id: ES 文档 ID
            feedback: 反馈类型 "positive" / "negative" / "neutral"

        Returns:
            是否更新成功
        """
        if not self._connected:
            return False

        try:
            self._es.update(
                index=self.index_name,
                id=doc_id,
                body={"doc": {"feedback": feedback}},
            )
            return True
        except Exception:
            return False

    def get_feedback_stats(self, user_id: str) -> Dict:
        """
        获取用户反馈统计

        Args:
            user_id: 用户ID

        Returns:
            {"positive": N, "negative": N, "neutral": N, "total": N}
        """
        if not self._connected:
            return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}

        try:
            result = self._es.search(
                index=self.index_name,
                body={
                    "query": {"term": {"user_id": user_id}},
                    "size": 0,
                    "aggs": {
                        "feedback_stats": {
                            "terms": {"field": "feedback"},
                        }
                    },
                },
            )
            buckets = result["aggregations"]["feedback_stats"]["buckets"]
            stats = {"positive": 0, "negative": 0, "neutral": 0, "total": 0}
            for b in buckets:
                stats[b["key"]] = b["doc_count"]
                stats["total"] += b["doc_count"]
            return stats
        except Exception:
            return {"positive": 0, "negative": 0, "neutral": 0, "total": 0}

    def delete_user_data(self, user_id: str) -> bool:
        """
        删除用户的所有数据（GDPR 合规）

        Args:
            user_id: 用户ID

        Returns:
            是否删除成功
        """
        if not self._connected:
            return False

        try:
            self._es.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"user_id": user_id}}},
            )
            return True
        except Exception:
            return False

    # ================================================================
    # 内部方法
    # ================================================================

    def _ensure_index(self):
        """创建 ES 索引（幂等，已存在时跳过）"""
        try:
            if not self._es.indices.exists(index=self.index_name):
                self._es.indices.create(
                    index=self.index_name,
                    body=ES_INDEX_MAPPING,
                )
        except Exception:
            pass