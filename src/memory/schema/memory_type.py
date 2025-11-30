"""
记忆类型枚举模块 (Memory Type Enumeration)

定义 Parallax 记忆系统中可提取和存储的记忆类型。

记忆类型层次结构:
==================

    MemUnit (原始提取单元 - 对话边界检测的输出)
        │
        ├── EpisodeMemory (EPISODE_SUMMARY) - 情景记忆
        │       从用户视角描述的个人叙事性记忆
        │       用途: 记录用户的主观体验和经历
        │
        ├── SemanticMemory (SEMANTIC_SUMMARY) - 语义记忆
        │       从对话中提取的客观事实性知识
        │       用途: 存储可验证的事实信息
        │
        ├── EventLog (EVENT_LOG) - 事件日志
        │       带时间戳的原子事实记录
        │       用途: 精确的时间线回溯
        │
        ├── ProfileMemory (PROFILE) - 用户画像
        │       用户的技能、性格、偏好等特征
        │       用途: 构建用户个人档案
        │
        └── GroupProfileMemory (GROUP_PROFILE) - 群体画像
                群体的话题、角色、互动模式
                用途: 分析群体动态和关系

使用示例:
========
    from memory.schema import MemoryType

    if memory.memory_type == MemoryType.EPISODE_SUMMARY:
        # 处理情景记忆
        pass
    elif memory.memory_type == MemoryType.PROFILE:
        # 处理用户画像
        pass
"""

from enum import Enum


class MemoryType(Enum):
    """
    记忆类型枚举

    定义系统中所有可提取的记忆类型，每种类型代表不同的记忆存储维度。

    主要类型说明:
    =============

    EPISODE_SUMMARY (情景记忆):
        - 定义: 从特定用户视角描述的个人叙事性记忆
        - 内容: 包含事件描述、个人感受、主观解读
        - 特点: 同一事件会为每个参与者生成不同的情景记忆
        - 用途: 个性化回忆、上下文理解、对话连贯性
        - 示例: "今天我和小明讨论了项目进度，他提到下周要上线..."

    SEMANTIC_SUMMARY (语义记忆):
        - 定义: 从对话中提取的客观事实性知识
        - 内容: 可验证的事实、技能信息、关系描述
        - 特点: 客观中立，不包含主观判断
        - 用途: 知识库构建、事实查询、推理基础
        - 示例: "用户熟悉 Python 编程"

    EVENT_LOG (事件日志):
        - 定义: 带精确时间戳的原子事实记录
        - 内容: 单一事实 + 发生时间
        - 特点: 粒度细、可追溯、时间敏感
        - 用途: 时间线重建、精确回忆、审计追踪
        - 示例: "2024-03-14 14:30 - 用户提到明天有会议"

    PROFILE (用户画像):
        - 定义: 用户的综合特征档案
        - 内容: 技能、性格、偏好、工作习惯、价值观等
        - 特点: 基于证据的渐进式构建，可合并更新
        - 用途: 个性化服务、用户理解、行为预测
        - 示例: {"hard_skills": [{"value": "Python", "level": "expert"}]}

    GROUP_PROFILE (群体画像):
        - 定义: 群体的集体特征和动态
        - 内容: 讨论话题、成员角色、互动模式
        - 特点: 描述群体而非个人，关注关系和动态
        - 用途: 群体分析、角色识别、话题追踪
        - 示例: {"roles": {"DECISION_MAKER": ["user_123"]}}
    """

    # ===== 主要记忆类型 (当前使用) =====
    EPISODE_SUMMARY = "episode_summary"  # 情景记忆 - 个人叙事视角
    SEMANTIC_SUMMARY = "semantic"  # 语义记忆 - 客观事实知识
    EVENT_LOG = "event_log"  # 事件日志 - 时间戳原子事实
    PROFILE = "profile"  # 用户画像 - 个人特征档案
    GROUP_PROFILE = "group_profile"  # 群体画像 - 集体特征

    # ===== 遗留/保留类型 (向后兼容) =====
    BASE_MEMORY = "baseMemory"  # 基础记忆 (已废弃，请使用 PROFILE)
    PREFERENCES = "preferences"  # 偏好记忆 (已合并至 PROFILE)
    RELATIONSHIPS = "relationships"  # 关系记忆 (保留，未来使用)
    CORE = "core"  # 核心记忆 (已废弃，请使用 PROFILE)
