# -*- coding: utf-8 -*-
"""
ShortTermMemoryManager - 短期记忆管理器

基于内存列表的会话级短期记忆，记录最近 N 轮对话上下文。
用于多轮对话连续性、追问判断、查询去重。

特性：
- 固定容量（默认 5 轮），超出自动淘汰最旧记录
- 提供格式化的上下文文本，可直接注入 LLM Prompt
- 支持会话开始/结束/重置
"""

from datetime import datetime
from typing import Dict, List, Optional

from config.settings import SHORT_TERM_MEMORY_SIZE


class ShortTermMemoryManager:
    """短期记忆管理器 - 会话级别内存存储"""

    def __init__(
        self,
        session_id: str = "default",
        max_rounds: int = SHORT_TERM_MEMORY_SIZE,
    ):
        """
        初始化短期记忆

        Args:
            session_id: 会话标识
            max_rounds: 最大保留对话轮数（一轮 = 用户问 + 助手答）
        """
        self.session_id = session_id
        self.max_rounds = max_rounds
        self._entries: List[Dict] = []

    # ================================================================
    # 公开接口
    # ================================================================

    def add_entry(
        self,
        role: str,
        content: str,
        query_type: Optional[str] = None,
        retrieved_docs: Optional[List[str]] = None,
    ):
        """
        添加一条记忆条目

        Args:
            role: 角色 "user" 或 "assistant"
            content: 对话内容
            query_type: 查询类型（可选）
            retrieved_docs: 该轮检索到的文档 ID 列表（可选）
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "retrieved_docs": retrieved_docs or [],
        }
        self._entries.append(entry)
        self._trim()

    def add_user_query(self, query: str, query_type: Optional[str] = None):
        """添加用户查询的便捷方法"""
        self.add_entry("user", query, query_type=query_type)

    def add_assistant_response(
        self,
        response: str,
        retrieved_docs: Optional[List[str]] = None,
    ):
        """添加助手回答的便捷方法"""
        self.add_entry("assistant", response, retrieved_docs=retrieved_docs)

    def get_context(self) -> str:
        """
        获取格式化的对话上下文，用于注入 LLM Prompt

        Returns:
            格式化后的历史对话文本，无历史时返回空字符串
        """
        if not self._entries:
            return ""

        lines = ["[历史对话]"]
        for entry in self._entries:
            role_label = "用户" if entry["role"] == "user" else "助手"
            lines.append(f"{role_label}: {entry['content']}")
        return "\n".join(lines)

    def get_recent_queries(self, n: Optional[int] = None) -> List[str]:
        """
        获取最近 N 轮用户查询，用于去重和参考

        Args:
            n: 返回数量，默认返回全部

        Returns:
            用户查询文本列表（按时间升序）
        """
        user_entries = [e["content"] for e in self._entries if e["role"] == "user"]
        if n is not None:
            user_entries = user_entries[-n:]
        return user_entries

    def get_last_answer(self) -> Optional[str]:
        """
        获取上轮助手的回答，用于追问判断

        Returns:
            上一轮回答文本，无历史时返回 None
        """
        for entry in reversed(self._entries):
            if entry["role"] == "assistant":
                return entry["content"]
        return None

    def get_last_user_query(self) -> Optional[str]:
        """
        获取上轮用户查询

        Returns:
            上一轮用户查询文本，无历史时返回 None
        """
        for entry in reversed(self._entries):
            if entry["role"] == "user":
                return entry["content"]
        return None

    def is_follow_up(self, new_query: str) -> bool:
        """
        判断是否为追问（基于简短查询 + 有历史对话）

        启发式规则：
        - 有至少一轮用户参与的对话历史
        - 新查询长度 < 15 字
        - 包含指代词或省略特征

        Args:
            new_query: 新查询文本

        Returns:
            是否为追问
        """
        if not self._entries:
            return False

        # 必须有至少一条用户查询记录
        has_user_entry = any(e["role"] == "user" for e in self._entries)
        if not has_user_entry:
            return False

        if len(new_query) >= 15:
            return False

        follow_up_indicators = [
            "那", "这个", "那个", "它", "还有", "其次", "另外",
            "呢", "吗", "怎么", "为什么", "然后", "再", "如果",
            "多少", "什么", "哪些", "不是", "为什么",
        ]
        return any(indicator in new_query for indicator in follow_up_indicators)

    def clear(self):
        """清空所有记忆（会话结束调用）"""
        self._entries.clear()

    def reset(self, new_session_id: Optional[str] = None):
        """重置记忆（新会话开始调用）"""
        self._entries.clear()
        if new_session_id is not None:
            self.session_id = new_session_id

    @property
    def total_entries(self) -> int:
        """当前记忆条目数"""
        return len(self._entries)

    @property
    def total_rounds(self) -> int:
        """当前对话轮数（一轮 = 用户 + 助手）"""
        user_count = sum(1 for e in self._entries if e["role"] == "user")
        return user_count

    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._entries) == 0

    def to_dict(self) -> Dict:
        """导出为字典，便于序列化"""
        return {
            "session_id": self.session_id,
            "max_rounds": self.max_rounds,
            "entries": self._entries,
        }

    # ================================================================
    # 内部方法
    # ================================================================

    def _trim(self):
        """裁剪超出容量的旧记录（按对话轮数）"""
        max_entries = self.max_rounds * 2  # 每轮 = 用户 + 助手
        while len(self._entries) > max_entries:
            self._entries.pop(0)