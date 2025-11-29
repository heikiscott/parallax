"""Memory type enumeration."""

from enum import Enum


class MemoryType(Enum):
    """Types of memories that can be extracted."""

    EPISODE_SUMMARY = "episode_summary"  # 情节记忆
    BASE_MEMORY = "baseMemory"  # 稳定、客观、可验证 Who He Is
    PROFILE = "profile"  # 能力与经验画像
    PREFERENCES = "preferences"  # 偏好设置
    RELATIONSHIPS = "relationships"  # 人际关系
    SEMANTIC_SUMMARY = "semantic"  # 语义记忆
    EVENT_LOG = "event_log"  # 事件日志

    GROUP_PROFILE = "group_profile"  # 群组画像

    CORE = "core"  # 核心记忆
