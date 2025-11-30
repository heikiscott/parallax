"""
用户画像模块 (Profile Memory)

定义 ProfileMemory，用于存储从对话中提取的用户综合特征档案，
包括技能、性格、偏好、工作习惯等多维度信息。

画像类别:
========
1. 技能维度: 硬技能 (技术能力) + 软技能 (人际能力)
2. 性格维度: 性格特征、决策风格、行为倾向
3. 职业维度: 工作职责、参与项目、职业目标
4. 行为维度: 工作习惯、沟通偏好
5. 动机维度: 价值观、动机系统、担忧恐惧

证据驱动设计:
============
所有画像属性都采用证据格式，确保可追溯和可验证:

    [
        {
            "value": "Python",           # 属性值
            "level": "expert",           # 熟练度 (技能专用)
            "evidences": [               # 证据来源
                "2024-01|conv_123",      # 日期|对话ID
                "2024-02|conv_456"
            ]
        }
    ]

这种设计支持:
- 追溯: 可查询属性来源于哪次对话
- 置信: 证据越多，属性越可信
- 合并: 多次对话的画像可增量合并
- 更新: 新证据可更新已有属性

使用场景:
========
- 个性化服务: 根据用户技能推荐内容
- 智能匹配: 根据画像匹配合适的任务/项目
- 行为预测: 基于性格和偏好预测用户行为
- 沟通优化: 根据沟通风格调整交互方式

使用示例:
========
    from memory.schema import ProfileMemory

    profile = ProfileMemory(
        user_id="user_123",
        timestamp=datetime.now(),
        memunit_id_list=["memunit_456"],
        user_name="小明",
        hard_skills=[
            {"value": "Python", "level": "expert", "evidences": ["2024-01|conv_1"]},
            {"value": "React", "level": "intermediate", "evidences": ["2024-02|conv_2"]}
        ],
        personality=[
            {"value": "分析型", "evidences": ["2024-01|conv_1"]}
        ]
    )
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .memory import Memory
from .memory_type import MemoryType


@dataclass
class ProfileMemory(Memory):
    """
    用户画像 - 从对话中提取的综合特征档案

    继承 Memory 基类，自动设置 memory_type 为 PROFILE。

    所有列表类型的属性都采用证据格式:
        [{"value": <string>, "evidences": [<date>|<conv_id>, ...], ...}]

    与其他记忆类型的区别:
    ====================
    - ProfileMemory: 用户特征档案 (是什么样的人)
    - EpisodeMemory: 主观叙事 (经历了什么)
    - GroupProfileMemory: 群体特征 (群体是什么样的)

    字段分组说明:
    =============

    1. 基本身份 (Identity):
        - user_name: 用户显示名称

    2. 技能维度 (Skills):
        - hard_skills: 硬技能/技术能力
          格式: [{"value": "Python", "level": "expert", "evidences": [...]}]
          level 可选值: beginner, intermediate, expert

        - soft_skills: 软技能/人际能力
          格式: [{"value": "沟通能力", "evidences": [...]}]

    3. 性格维度 (Personality):
        - personality: 性格特征
          示例: 分析型、外向型、细致型

        - way_of_decision_making: 决策风格
          示例: 数据驱动、直觉导向、共识型

        - tendency: 行为倾向
          示例: 风险偏好、保守稳健

    4. 职业维度 (Professional):
        - work_responsibility: 工作职责
          示例: 后端开发、项目管理

        - projects_participated: 参与的项目
          格式: List[ProjectInfo] (见 extraction/memory/profile/types.py)

        - user_goal: 职业/个人目标
          示例: 成为技术专家、创业

    5. 行为维度 (Behavioral):
        - working_habit_preference: 工作习惯偏好
          示例: 早起工作、喜欢安静环境

        - interests: 兴趣爱好
          示例: 阅读、编程、户外运动

    6. 动机维度 (Motivational):
        - motivation_system: 动机系统
          示例: 追求成就、渴望认可

        - fear_system: 担忧/恐惧系统
          示例: 害怕失败、担心落后

        - value_system: 价值观体系
          示例: 诚信、创新、团队合作

    7. 沟通维度 (Communication):
        - humor_use: 幽默使用风格
          示例: 自嘲型、讽刺型、温和型

        - colloquialism: 口语习惯/常用表达
          示例: 特定的口头禅或表达方式

    8. 群体动态 (Group Dynamics):
        - group_importance_evidence: 用户在群体中的重要性证据
          格式: GroupImportanceEvidence (见 extraction/memory/profile/types.py)

    9. 推理说明 (Reasoning):
        - output_reasoning: 画像提取的推理过程说明

    使用场景:
    ========
    - 个性化推荐: 根据技能和兴趣推荐内容
    - 团队匹配: 根据性格和技能匹配合适的角色
    - 沟通优化: 根据沟通风格调整交互方式
    - 目标追踪: 了解用户目标，提供相关支持
    """

    # ===== 1. 基本身份 =====
    user_name: Optional[str] = None  # 用户显示名称

    # ===== 2. 技能维度 =====
    # 格式: [{"value": "skill_name", "level": "beginner|intermediate|expert", "evidences": [...]}]
    hard_skills: Optional[List[Dict[str, Any]]] = None  # 硬技能/技术能力
    soft_skills: Optional[List[Dict[str, Any]]] = None  # 软技能/人际能力

    # ===== 3. 性格维度 =====
    # 格式: [{"value": "trait_description", "evidences": [...]}]
    personality: Optional[List[Dict[str, Any]]] = None  # 性格特征
    way_of_decision_making: Optional[List[Dict[str, Any]]] = None  # 决策风格
    tendency: Optional[List[Dict[str, Any]]] = None  # 行为倾向

    # ===== 4. 职业维度 =====
    work_responsibility: Optional[List[Dict[str, Any]]] = None  # 工作职责
    projects_participated: Optional[List[Any]] = None  # 参与的项目 (List[ProjectInfo])
    user_goal: Optional[List[Dict[str, Any]]] = None  # 职业/个人目标

    # ===== 5. 行为维度 =====
    working_habit_preference: Optional[List[Dict[str, Any]]] = None  # 工作习惯偏好
    interests: Optional[List[Dict[str, Any]]] = None  # 兴趣爱好

    # ===== 6. 动机维度 =====
    motivation_system: Optional[List[Dict[str, Any]]] = None  # 动机系统
    fear_system: Optional[List[Dict[str, Any]]] = None  # 担忧/恐惧系统
    value_system: Optional[List[Dict[str, Any]]] = None  # 价值观体系

    # ===== 7. 沟通维度 =====
    humor_use: Optional[List[Dict[str, Any]]] = None  # 幽默使用风格
    colloquialism: Optional[List[Dict[str, Any]]] = None  # 口语习惯

    # ===== 8. 群体动态 =====
    group_importance_evidence: Optional[Any] = None  # 群体重要性证据 (GroupImportanceEvidence)

    # ===== 9. 推理说明 =====
    output_reasoning: Optional[str] = None  # 画像提取推理说明

    def __post_init__(self) -> None:
        """初始化用户画像，设置正确的记忆类型"""
        self.memory_type = MemoryType.PROFILE
        super().__post_init__()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，用于序列化

        扩展基类 to_dict()，添加所有画像特定字段。
        处理嵌套对象 (ProjectInfo, GroupImportanceEvidence) 的序列化。

        返回:
            包含所有画像字段的字典
        """
        base_dict = super().to_dict()

        base_dict.update({
            # 基本身份
            "user_name": self.user_name,

            # 技能维度
            "hard_skills": self.hard_skills,
            "soft_skills": self.soft_skills,

            # 性格维度
            "personality": self.personality,
            "way_of_decision_making": self.way_of_decision_making,
            "tendency": self.tendency,

            # 职业维度
            "work_responsibility": self.work_responsibility,
            "projects_participated": [
                p.to_dict() if hasattr(p, 'to_dict') else p
                for p in (self.projects_participated or [])
            ] if self.projects_participated else None,
            "user_goal": self.user_goal,

            # 行为维度
            "working_habit_preference": self.working_habit_preference,
            "interests": self.interests,

            # 动机维度
            "motivation_system": self.motivation_system,
            "fear_system": self.fear_system,
            "value_system": self.value_system,

            # 沟通维度
            "humor_use": self.humor_use,
            "colloquialism": self.colloquialism,

            # 群体动态
            "group_importance_evidence": (
                self.group_importance_evidence.to_dict()
                if hasattr(self.group_importance_evidence, 'to_dict')
                else self.group_importance_evidence
            ) if self.group_importance_evidence else None,

            # 推理说明
            "output_reasoning": self.output_reasoning,
        })

        return base_dict
