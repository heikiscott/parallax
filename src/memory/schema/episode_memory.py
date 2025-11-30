"""
情景记忆模块 (Episode Memory)

定义 EpisodeMemory，用于存储从特定用户视角描述的个人叙事性记忆。
情景记忆是记忆提取流水线的主要输出类型。

核心特点:
========
- 个人视角: 从单一用户的角度描述事件
- 叙事性: 以故事形式描述，而非简单的事实罗列
- 主观性: 包含个人感受、反应和解读
- 可追溯: 通过 memunit_id_list 关联到源 MemUnit

多视角生成:
==========
同一个 MemUnit (群体事件) 会为每个参与者生成不同的 EpisodeMemory:

    MemUnit (群体事件):
        "小明和小红讨论了新的 API 设计。小明建议用 REST，
         小红倾向于 GraphQL。他们决定分别做原型对比。"

    EpisodeMemory (小明视角):
        "今天我和小红讨论了 API 设计。我建议用 REST 因为简单，
         她提出了 GraphQL 的灵活性优势。我们达成共识，
         各自做原型来数据驱动决策。"

    EpisodeMemory (小红视角):
        "我和小明进行了一次富有成效的 API 设计讨论。
         他提议用 REST，我推荐了 GraphQL。
         我们决定同时做两个原型来比较。"

使用场景:
========
- 个性化回忆: 用户询问 "上次我们讨论了什么?"
- 上下文理解: 理解用户在特定事件中的角色和立场
- 对话连贯性: 保持对话的一致性和连续性

使用示例:
========
    from memory.schema import EpisodeMemory

    episode = EpisodeMemory(
        user_id="alice_123",
        timestamp=datetime.now(),
        memunit_id_list=["memunit_456"],
        event_id="episode_789",
        episode="今天我和团队讨论了项目时间线...",
        summary="项目时间线讨论",
        subject="Sprint 规划"
    )
"""

from dataclasses import dataclass, field

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class EpisodeMemory(Memory):
    """
    情景记忆 - 从用户视角描述的个人叙事记忆

    继承 Memory 基类，自动设置 memory_type 为 EPISODE_SUMMARY。

    每个 EpisodeMemory 代表一个用户对一组事件的主观体验。
    同一个 MemUnit 通常会为每个参与者生成独立的 EpisodeMemory，
    每份记忆都带有各自的视角。

    与其他记忆类型的区别:
    ====================
    - EpisodeMemory: 主观叙事 ("我和小明讨论了...")
    - SemanticMemory: 客观事实 ("小明会 Python")
    - ProfileMemory: 特征档案 (技能、性格、偏好)

    字段说明:
    ========

    情景记忆特有字段:
        - event_id: 情景记忆自身的唯一标识符
          不同于 memunit_id_list (源 MemUnit ID)
          用于存储和检索时的主键

    继承自 Memory 基类的字段:
        - memory_type: 自动设置为 MemoryType.EPISODE_SUMMARY
        - user_id: 记忆所属用户 (该视角的主人)
        - timestamp: 事件发生时间
        - memunit_id_list: 源 MemUnit ID 列表
        - episode: 完整的叙事文本 (主要内容)
        - summary: 简短摘要
        - subject: 话题/标题
        - group_id: 群组ID
        - participants: 参与者列表
        - keywords: 关键词
        - semantic_memories: 关联的语义记忆预测

    使用场景:
    ========
    - 记录用户的主观体验和经历
    - 支持 "我" 视角的回忆查询
    - 保持对话的个性化和连贯性
    - 追踪用户在群体事件中的参与

    示例:
        >>> episode = EpisodeMemory(
        ...     user_id="user_123",
        ...     timestamp=datetime.now(),
        ...     memunit_id_list=["memunit_1"],
        ...     event_id="episode_1",
        ...     episode="今天我了解了新功能的使用方法...",
        ...     summary="学习新功能",
        ...     subject="功能培训"
        ... )
    """

    # ===== 情景记忆特有字段 =====
    event_id: str = field(default=None)  # 情景记忆的唯一标识符

    def __post_init__(self):
        """
        初始化情景记忆

        自动设置 memory_type 为 EPISODE_SUMMARY，并调用父类初始化。
        """
        self.memory_type = MemoryType.EPISODE_SUMMARY
        super().__post_init__()
