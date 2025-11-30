"""
MemUnit Beanie ODM 模型

基于 Beanie ODM 的 MemUnit 数据模型定义，支持 MongoDB 分片集群。
"""

from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum

from beanie import Indexed
from core.oxm.mongo.document_base import DocumentBase
from pydantic import BaseModel, Field, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING
from core.oxm.mongo.audit_base import AuditBase


class DataTypeEnum(str, Enum):
    """数据类型枚举"""

    CONVERSATION = "Conversation"


class Message(BaseModel):
    """消息结构"""

    content: str = Field(..., description="消息文本内容")
    files: Optional[List[str]] = Field(default=None, description="文件链接列表")
    extend: Optional[Dict[str, str]] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "content": "今天的会议讨论了新功能的设计方案",
                "files": ["https://example.com/design_doc.pdf"],
                "extend": {
                    "sender": "张三",
                    "message_id": "msg_001",
                    "platform": "WeChat",
                },
            }
        }
    )


class RawData(BaseModel):
    """原始数据结构"""

    data_type: DataTypeEnum = Field(..., description="数据类型枚举")
    messages: List[Message] = Field(..., min_length=1, description="消息列表")
    meta: Optional[Dict[str, str]] = Field(default=None, description="元数据")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "data_type": "Conversation",
                "messages": [
                    {"content": "团队讨论新功能", "extend": {"sender": "张三"}}
                ],
                "meta": {"chat_id": "chat_12345", "platform": "WeChat"},
            }
        }
    )


class MemUnit(DocumentBase, AuditBase):
    """
    MemUnit 文档模型

    情景切分之后的结果存储模型，支持灵活扩展和高性能查询。

    字段生命周期:
    =============

    创建阶段 (ConvMemUnitExtractor.extract_memunit):
    - unit_id: ✅ 生成 UUID
    - user_id_list: ✅ 从请求获取
    - original_data: ✅ 处理后的消息列表
    - timestamp: ✅ 从最后消息获取
    - summary: ✅ 边界检测生成
    - group_id: ✅ 从请求获取
    - participants: ✅ 从消息提取
    - type: ✅ CONVERSATION
    - subject: ❌ None
    - narrative: ❌ None
    - keywords: ❌ None (预留字段，暂未实现)
    - linked_entities: ❌ None (预留字段，暂未实现)
    - semantic_memories: ❌ None
    - event_log: ❌ None
    - extend: ❌ None

    提取阶段 (EpisodeMemoryExtractor.extract_memory):
    - subject: ✅ LLM 提取的 title
    - narrative: ✅ LLM 提取的 content
    - extend['embedding']: ✅ 向量化后赋值

    语义提取阶段 (SemanticMemoryExtractor):
    - semantic_memories: ✅ 语义记忆列表

    事件提取阶段 (ExtractionOrchestrator):
    - event_log: ✅ 事件日志对象
    """

    # ===== 标识字段 =====
    unit_id: Optional[Indexed(str)] = Field(
        default=None,
        description="Schema MemUnit 中生成的 UUID，独立于 MongoDB _id，用于跨系统追踪"
    )

    # ===== 用户字段 =====
    user_id: Optional[Indexed(str)] = Field(
        default=None,
        description="用户ID，核心查询字段。群组记忆时为None，个人记忆时为用户ID"
    )
    user_id_list: Optional[List[str]] = Field(
        default=None,
        description="涉及的所有用户ID列表，用于权限过滤和生成个人视角记忆"
    )

    # ===== 时间字段 =====
    timestamp: Indexed(datetime) = Field(..., description="发生时间，分片键")

    # ===== 内容字段 =====
    summary: str = Field(..., min_length=1, description="记忆单元摘要")

    # ===== 上下文字段 =====
    group_id: Optional[Indexed(str)] = Field(
        default=None, description="群组ID，为空表示私聊"
    )
    original_data: Optional[List] = Field(default=None, description="原始信息")
    participants: Optional[List[str]] = Field(
        default=None, description="事件参与者名字"
    )
    type: Optional[DataTypeEnum] = Field(default=None, description="情景类型")

    subject: Optional[str] = Field(default=None, description="记忆单元主题")

    # 预留字段，暂未实现 LLM 提取逻辑，后续可能会实现
    keywords: Optional[List[str]] = Field(default=None, description="关键词（预留字段，暂未实现）")
    linked_entities: Optional[List[str]] = Field(
        default=None, description="关联的实体ID（预留字段，暂未实现）"
    )

    narrative: Optional[str] = Field(default=None, description="叙事描述（核心内容字段）")
    semantic_memories: Optional[List] = Field(default=None, description="语义记忆")
    event_log: Optional[Dict] = Field(default=None, description="Event Log 原子事实")
    extend: Optional[Dict] = Field(default=None, description="扩展字段")

    model_config = ConfigDict(
        # 集合名称
        collection="memunits",
        # 验证配置
        validate_assignment=True,
        # JSON 序列化配置
        json_encoders={datetime: lambda dt: dt.isoformat()},
        # 示例数据
        json_schema_extra={
            "example": {
                "user_id": "user_12345",
                "group_id": "group_67890",
                "timestamp": "2024-12-01T10:30:00.000Z",
                "summary": "团队讨论新功能设计方案，获得积极反馈",
                "original_data": [
                    {
                        "data_type": "Conversation",
                        "messages": [
                            {
                                "content": "今天的会议讨论了新功能的设计方案",
                                "files": ["https://example.com/design_doc.pdf"],
                                "extend": {"sender": "张三", "message_id": "msg_001"},
                            }
                        ],
                        "meta": {"chat_id": "chat_12345", "platform": "WeChat"},
                    }
                ],
                "participants": ["张三", "李四", "王五"],
                "type": "Conversation",
                "keywords": ["新功能", "设计方案", "会议"],
                "linked_entities": ["project_001", "feature_002"],
            }
        },
    )

    class Settings:
        """Beanie 设置"""

        # 集合名称
        name = "memunits"

        # 索引定义
        indexes = [
            # 2. 用户查询复合索引 - 核心查询模式
            IndexModel(
                [("user_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_user_timestamp",
            ),
            # 3. 群组查询复合索引 - 群聊场景优化
            IndexModel(
                [("group_id", ASCENDING), ("timestamp", DESCENDING)],
                name="idx_group_timestamp",
            ),
            # 4. 时间范围查询索引（分片键，MongoDB自动创建）
            # 注意：分片键索引会自动创建，无需手动定义
            # IndexModel([("timestamp", ASCENDING)], name="idx_timestamp"),
            # 5. 参与者查询索引 - 多值字段索引
            IndexModel(
                [("participants", ASCENDING)], name="idx_participants", sparse=True
            ),
            # 6. 用户类型查询复合索引 - 用户数据类型过滤场景优化
            IndexModel(
                [
                    ("user_id", ASCENDING),
                    ("type", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_user_type_timestamp",
            ),
            # 7. 群组类型查询复合索引 - 群组数据类型过滤场景优化
            IndexModel(
                [
                    ('group_id', ASCENDING),
                    ("type", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="idx_group_type_timestamp",
            ),
        ]

        # 验证设置
        validate_on_save = True
        use_state_management = True


# 导出模型
__all__ = ["MemUnit", "RawData", "Message", "DataTypeEnum"]
