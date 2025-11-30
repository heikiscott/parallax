"""
记忆数据结构模块 (Memory Schema Module)

包含所有记忆相关的数据结构和类型定义。
每个类型单独定义在各自的文件中，便于维护。

模块结构:
========
- memory_type.py: 记忆类型枚举 (MemoryType)
- source_type.py: 数据源类型枚举 (SourceType)
- memunit.py: 记忆单元 (MemUnit) - 对话边界检测的输出
- memory.py: 记忆基类 (Memory)
- semantic_memory.py: 语义记忆 (SemanticMemory, SemanticMemoryItem)
- episode_memory.py: 情景记忆 (EpisodeMemory)
- profile_memory.py: 用户画像 (ProfileMemory)
- group_profile_memory.py: 群体画像 (GroupProfileMemory)

类型层次:
========
    MemUnit (原始提取单元)
        │
        └── Memory (记忆基类)
                ├── EpisodeMemory (情景记忆)
                ├── ProfileMemory (用户画像)
                └── GroupProfileMemory (群体画像)

    独立类型:
        ├── SemanticMemory (语义记忆 - 事实性)
        └── SemanticMemoryItem (语义记忆项 - 预测性)

使用示例:
========
    from memory.schema import (
        MemoryType,
        SourceType,
        MemUnit,
        EpisodeMemory,
        ProfileMemory,
        GroupProfileMemory,
        SemanticMemory,
        SemanticMemoryItem,
    )
"""

from .memory_type import MemoryType
from .source_type import SourceType
from .memunit import MemUnit
from .memory import Memory
from .semantic_memory import SemanticMemory, SemanticMemoryItem
from .episode_memory import EpisodeMemory
from .profile_memory import ProfileMemory
from .group_profile_memory import GroupProfileMemory

__all__ = [
    # 枚举类型
    "MemoryType",
    "SourceType",
    # 核心数据结构
    "MemUnit",
    "Memory",
    # 具体记忆类型
    "SemanticMemory",
    "SemanticMemoryItem",
    "EpisodeMemory",
    "ProfileMemory",
    "GroupProfileMemory",
]
