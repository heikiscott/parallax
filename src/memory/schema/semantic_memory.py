"""
语义记忆模块 (Semantic Memory)

定义语义记忆相关的数据结构，用于存储从情景记忆中提取的
客观事实性知识和前瞻性关联预测。

两种语义记忆类型:
================

1. SemanticMemory (语义记忆 - 事实性知识):
   - 存储在数据库中的客观事实
   - 独立于情景记忆存在
   - 可单独查询和检索
   - 示例: "用户熟悉 Python 编程"

2. SemanticMemoryItem (语义记忆项 - 前瞻性关联):
   - 附加在 MemUnit 或 Memory 上的预测
   - 描述事件可能对用户的影响
   - 用于上下文增强和主动推荐
   - 示例: "用户可能需要 Python 进阶教程"

区别说明:
========
- SemanticMemory 是 "已知事实"，描述现状
- SemanticMemoryItem 是 "关联预测"，描述可能性

使用场景:
========

SemanticMemory (事实性):
    - 用户画像补充
    - 知识图谱构建
    - 事实查询

SemanticMemoryItem (预测性):
    - 上下文推荐
    - 个性化建议
    - 时效性提醒

使用示例:
========
    from memory.schema import SemanticMemory, SemanticMemoryItem

    # 事实性知识
    semantic = SemanticMemory(
        user_id="user_123",
        content="用户精通 Python 编程",
        knowledge_type="skill"
    )

    # 前瞻性关联
    item = SemanticMemoryItem(
        content="用户可能需要 Python 高级课程",
        evidence="用户正在学习 Python",
        start_time="2024-01-01",
        duration_days=30
    )
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import datetime

from utils.datetime_utils import to_iso_format


@dataclass
class SemanticMemory:
    """
    语义记忆 - 事实性知识

    存储从情景记忆中提取的客观、可验证的事实信息。
    独立存储在数据库中，可单独查询。

    与 SemanticMemoryItem 的区别:
    - SemanticMemory: 描述已知事实 ("用户会 Python")
    - SemanticMemoryItem: 描述可能关联 ("用户可能需要 Python 资源")

    字段分组说明:
    =============

    1. 归属字段 (Ownership):
        - user_id: 知识所属用户ID

    2. 内容字段 (Content):
        - content: 事实性知识陈述
          示例: "用户有 React 开发经验"

    3. 分类字段 (Classification):
        - knowledge_type: 知识类型
          常见类型: "skill", "preference", "fact", "relationship"

    4. 溯源字段 (Provenance):
        - source_episodes: 知识来源的情景记忆ID列表
        - created_at: 知识提取时间

    5. 上下文字段 (Context):
        - group_id: 知识获取的群组上下文
        - participants: 相关事件的参与者

    6. 元数据字段 (Metadata):
        - metadata: 额外结构化信息
          可包含置信度、提取方法等

    使用场景:
    ========
    - 构建用户知识图谱
    - 支持事实查询: "用户会什么技能?"
    - 补充用户画像信息
    """

    # ===== 1. 归属字段 =====
    user_id: str  # 知识所属用户ID

    # ===== 2. 内容字段 =====
    content: str  # 事实性知识陈述

    # ===== 3. 分类字段 =====
    knowledge_type: str = "knowledge"  # 知识类型 (skill/preference/fact/relationship)

    # ===== 4. 溯源字段 =====
    source_episodes: List[str] = None  # 知识来源的情景记忆ID
    created_at: datetime.datetime = None  # 知识提取时间

    # ===== 5. 上下文字段 =====
    group_id: Optional[str] = None  # 群组上下文
    participants: Optional[List[str]] = None  # 事件参与者

    # ===== 6. 元数据字段 =====
    metadata: Optional[Dict[str, Any]] = None  # 额外结构化信息

    def __post_init__(self):
        """初始化可变字段的默认值"""
        if self.source_episodes is None:
            self.source_episodes = []
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        return {
            # 归属字段
            "user_id": self.user_id,
            # 内容字段
            "content": self.content,
            # 分类字段
            "knowledge_type": self.knowledge_type,
            # 溯源字段
            "source_episodes": self.source_episodes,
            "created_at": to_iso_format(self.created_at),
            # 上下文字段
            "group_id": self.group_id,
            "participants": self.participants,
            # 元数据字段
            "metadata": self.metadata,
        }


@dataclass
class SemanticMemoryItem:
    """
    语义记忆项 - 前瞻性关联预测

    描述事件可能对用户产生的影响或关联。
    附加在 MemUnit 和 Memory 上，用于上下文增强。

    与 SemanticMemory 的区别:
    - SemanticMemory: 已知事实 ("用户会 Python")
    - SemanticMemoryItem: 前瞻预测 ("用户可能需要 Python 资源")

    字段分组说明:
    =============

    1. 内容字段 (Content):
        - content: 预测性关联陈述
          示例: "用户可能需要项目管理工具"

    2. 证据字段 (Evidence):
        - evidence: 支持预测的简短事实 (≤30字符)
          示例: "用户提到要开始新项目"

    3. 时效字段 (Timing):
        - start_time: 关联生效时间 (YYYY-MM-DD)
        - end_time: 关联过期时间 (YYYY-MM-DD)
        - duration_days: 预期有效期 (天)
          用于检索时的时间衰减计算

    4. 溯源字段 (Provenance):
        - source_episode_id: 来源情景记忆ID
          用于追溯验证

    5. 检索字段 (Retrieval):
        - embedding: 向量嵌入
          预计算的语义向量，用于相似度搜索

    使用场景:
    ========
    - 上下文推荐: 根据用户近期行为推荐相关内容
    - 主动提醒: 在相关时间点主动提供信息
    - 个性化增强: 丰富对话上下文
    """

    # ===== 1. 内容字段 =====
    content: str  # 预测性关联陈述

    # ===== 2. 证据字段 =====
    evidence: Optional[str] = None  # 支持预测的简短事实 (≤30字符)

    # ===== 3. 时效字段 =====
    start_time: Optional[str] = None  # 关联生效时间 (YYYY-MM-DD)
    end_time: Optional[str] = None  # 关联过期时间 (YYYY-MM-DD)
    duration_days: Optional[int] = None  # 预期有效期 (天)

    # ===== 4. 溯源字段 =====
    source_episode_id: Optional[str] = None  # 来源情景记忆ID

    # ===== 5. 检索字段 =====
    embedding: Optional[List[float]] = None  # 向量嵌入

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于序列化"""
        return {
            # 内容字段
            "content": self.content,
            # 证据字段
            "evidence": self.evidence,
            # 时效字段
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_days": self.duration_days,
            # 溯源字段
            "source_episode_id": self.source_episode_id,
            # 检索字段
            "embedding": self.embedding,
        }
