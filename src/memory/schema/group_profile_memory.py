"""
群体画像模块 (Group Profile Memory)

定义 GroupProfileMemory，用于存储群体的集体特征，
包括讨论话题和成员角色等群体动态信息。

核心概念:
========
- 话题 (Topics): 群体讨论的主题/议题
- 角色 (Roles): 成员在群体中承担的功能性角色

与 ProfileMemory 的区别:
=======================
- ProfileMemory: 描述个人特征 (这个人是什么样的)
- GroupProfileMemory: 描述群体特征 (这个群体是什么样的)

群体画像关注:
- 群体关心什么 (话题)
- 成员之间的关系 (角色)
- 集体模式 (决策过程、互动风格)

话题数据结构:
============
topics 是 TopicInfo 对象列表 (定义在 extraction/memory/group_profile/):

    {
        "topic": "API 设计方案",      # 话题名称
        "status": "exploring",        # 状态: exploring/consensus/decided
        "confidence": "strong",       # 置信度: strong/weak
        "evidences": [...]            # 证据来源
    }

角色数据结构:
============
roles 是角色类型到用户列表的映射:

    {
        "DECISION_MAKER": [           # 决策者
            {
                "user_id": "u1",
                "user_name": "小明",
                "confidence": "strong",
                "evidences": [...]
            }
        ],
        "TOPIC_INITIATOR": [...]      # 话题发起者
    }

支持的角色类型 (GroupRole 枚举):
- DECISION_MAKER: 决策者 - 做出最终决定的人
- OPINION_LEADER: 意见领袖 - 影响群体观点的人
- TOPIC_INITIATOR: 话题发起者 - 主动开启新讨论的人
- EXECUTION_PROMOTER: 执行推动者 - 推动落地执行的人
- CORE_CONTRIBUTOR: 核心贡献者 - 主要贡献内容的人
- COORDINATOR: 协调者 - 协调各方工作的人
- INFO_SUMMARIZER: 信息整合者 - 总结归纳信息的人

使用场景:
========
- 群体分析: 了解群体的关注点和讨论热点
- 角色识别: 识别群体中的关键人物
- 会议记录: 追踪群体讨论的演进
- 团队协作: 优化团队分工和协作

使用示例:
========
    from memory.schema import GroupProfileMemory

    group_profile = GroupProfileMemory(
        user_id="group_admin",
        timestamp=datetime.now(),
        ori_event_id_list=["memunit_1", "memunit_2"],
        group_id="group_123",
        group_name="工程团队",
        topics=[topic_info_1, topic_info_2],
        roles={
            "DECISION_MAKER": [
                {"user_id": "alice", "user_name": "小明", "confidence": "strong"}
            ]
        }
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class GroupProfileMemory(Memory):
    """
    群体画像 - 群体的集体特征

    继承 Memory 基类，自动设置 memory_type 为 GROUP_PROFILE。

    描述群体的集体身份，而非个人档案:
    - 话题: 群体讨论/关心的内容
    - 角色: 成员在群体动态中的功能

    与其他记忆类型的区别:
    ====================
    - GroupProfileMemory: 群体特征 (群体关心什么、谁是领导者)
    - ProfileMemory: 个人特征 (个人技能、性格)
    - EpisodeMemory: 事件叙事 (发生了什么)

    字段说明:
    ========

    1. 群体身份 (Identity):
        - group_name: 群组显示名称
          示例: "工程团队", "产品讨论组"

    2. 群体话题 (Topics):
        - topics: 讨论话题列表
          类型: List[TopicInfo] (见 extraction/memory/group_profile/)
          每个话题包含:
            - topic: 话题名称
            - status: 状态 (exploring/consensus/decided)
            - confidence: 置信度 (strong/weak)
            - evidences: 证据来源

    3. 成员角色 (Roles):
        - roles: 角色到用户的映射
          格式: {
              "ROLE_TYPE": [
                  {
                      "user_id": "...",
                      "user_name": "...",
                      "confidence": "strong|weak",
                      "evidences": [...]
                  }
              ]
          }

          支持的角色类型:
            - DECISION_MAKER: 决策者
            - OPINION_LEADER: 意见领袖
            - TOPIC_INITIATOR: 话题发起者
            - EXECUTION_PROMOTER: 执行推动者
            - CORE_CONTRIBUTOR: 核心贡献者
            - COORDINATOR: 协调者
            - INFO_SUMMARIZER: 信息整合者

    注意:
    ====
    TopicInfo 和 GroupRole 定义在 extraction/memory/group_profile/ 模块中，
    因为它们与提取逻辑紧密相关。

    使用场景:
    ========
    - 群体分析: 了解群体关注点
    - 角色识别: 找出关键人物
    - 话题追踪: 追踪讨论演进
    - 团队优化: 改善团队协作

    示例:
        >>> profile = GroupProfileMemory(
        ...     user_id="admin_123",
        ...     timestamp=datetime.now(),
        ...     ori_event_id_list=["mu_1"],
        ...     group_id="grp_456",
        ...     group_name="开发组",
        ...     topics=[...],  # TopicInfo 对象列表
        ...     roles={"DECISION_MAKER": [{"user_id": "u1", ...}]}
        ... )
    """

    # ===== 1. 群体身份 =====
    group_name: Optional[str] = None  # 群组显示名称

    # ===== 2. 群体话题 =====
    # 类型: List[TopicInfo] (见 extraction/memory/group_profile/)
    # TopicInfo 包含: topic, status, confidence, evidences
    topics: Optional[List[Any]] = field(default_factory=list)

    # ===== 3. 成员角色 =====
    # 格式: {"ROLE_TYPE": [{"user_id": "...", "user_name": "...", "confidence": "...", "evidences": [...]}]}
    roles: Optional[Dict[str, List[Dict[str, str]]]] = field(default_factory=dict)

    def __post_init__(self):
        """
        初始化群体画像

        设置 memory_type 为 GROUP_PROFILE，确保 topics 和 roles 已初始化。
        """
        self.memory_type = MemoryType.GROUP_PROFILE
        if self.topics is None:
            self.topics = []
        if self.roles is None:
            self.roles = {}
        super().__post_init__()
