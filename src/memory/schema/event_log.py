"""
事件日志模块 (Event Log)

定义事件日志数据结构，用于存储从情景记忆中提取的原子事实。
支持细粒度的事实检索和证据定位。

与其他记忆类型的关系:
====================

1. EventLog (事件日志):
   - 嵌入在 MemUnit 中
   - 包含时间点和原子事实列表
   - 每个原子事实是独立、可验证的陈述
   - 用于细粒度事实检索

2. SemanticMemoryItem (语义记忆项):
   - 同样嵌入在 MemUnit 中
   - 描述前瞻性关联预测
   - 用于上下文增强

使用场景:
========
- 提供细粒度的事实检索
- 支持事实级别的向量搜索
- 用于问答系统的证据定位

使用示例:
========
    from memory.schema import EventLog

    event_log = EventLog(
        time="May 08, 2023(Monday) at 01:56 PM",
        atomic_fact=[
            "Caroline greeted her friend Melanie with enthusiasm.",
            "Caroline expressed happiness to see Melanie.",
            "Melanie mentioned feeling overwhelmed with responsibilities.",
        ]
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class EventLog:
    """
    事件日志 - 原子事实

    存储从情景记忆中提取的时间点和原子事实列表。
    每个原子事实是一个独立、可验证的陈述。

    字段分组说明:
    =============

    1. 时间字段 (Timing):
        - time: 事件发生时间
          格式: "March 10, 2024(Sunday) at 2:00 PM"

    2. 内容字段 (Content):
        - atomic_fact: 原子事实列表
          每个事实是一个完整的句子，描述单一事件或状态

    3. 检索字段 (Retrieval):
        - fact_embeddings: 每个原子事实的向量嵌入
          预计算的语义向量，用于事实级别的相似度搜索

    使用场景:
    ========
    - 细粒度事实检索: 根据查询找到最相关的原子事实
    - 证据定位: 为回答提供具体的事实依据
    - 时间线构建: 基于时间和事实构建事件时间线

    示例:
    ====
    ```python
    event_log = EventLog(
        time="May 08, 2023(Monday) at 01:56 PM",
        atomic_fact=[
            "Caroline greeted her friend Melanie with enthusiasm.",
            "Caroline expressed happiness to see Melanie.",
            "Melanie mentioned feeling overwhelmed with responsibilities.",
        ],
        fact_embeddings=[
            [0.1, 0.2, ...],  # 第一个事实的向量
            [0.3, 0.4, ...],  # 第二个事实的向量
            [0.5, 0.6, ...],  # 第三个事实的向量
        ]
    )
    ```
    """

    # ===== 1. 时间字段 =====
    time: str  # 事件发生时间，格式如 "March 10, 2024(Sunday) at 2:00 PM"

    # ===== 2. 内容字段 =====
    atomic_fact: List[str]  # 原子事实列表，每个事实是一个完整的句子

    # ===== 3. 检索字段 =====
    fact_embeddings: Optional[List[List[float]]] = None  # 每个 atomic_fact 对应的 embedding

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，用于序列化存储。

        Returns:
            包含所有非空字段的字典
        """
        result = {
            "time": self.time,
            "atomic_fact": self.atomic_fact,
        }
        if self.fact_embeddings:
            result["fact_embeddings"] = self.fact_embeddings
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventLog":
        """
        从字典创建 EventLog 实例。

        Args:
            data: 包含 time、atomic_fact 等字段的字典

        Returns:
            EventLog 实例
        """
        return cls(
            time=data.get("time", ""),
            atomic_fact=data.get("atomic_fact", []),
            fact_embeddings=data.get("fact_embeddings"),
        )
