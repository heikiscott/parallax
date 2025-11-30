"""
数据源类型枚举模块 (Source Type Enumeration)

定义记忆提取系统可处理的输入数据源类型。

当前支持的数据源:
================
- CONVERSATION: 聊天消息、群组讨论、对话记录

计划支持的数据源 (未来版本):
==========================
- EMAIL: 邮件往来
- DOCUMENT: 文档、笔记、文章
- MEETING: 会议记录、会议纪要

设计说明:
========
不同数据源类型可能需要不同的提取策略:
- CONVERSATION: 基于对话边界检测，提取话题转换点
- EMAIL: 基于邮件线程结构，提取讨论主题
- DOCUMENT: 基于段落/章节结构，提取知识点

使用示例:
========
    from memory.schema import SourceType

    raw_data = RawData(
        content={"speaker_id": "user_1", "content": "你好"},
        data_type=SourceType.CONVERSATION
    )

    # 字符串转换
    source_type = SourceType.from_string("conversation")  # 返回 SourceType.CONVERSATION
"""

from enum import Enum
from typing import Optional


class SourceType(Enum):
    """
    数据源类型枚举

    标识被处理的原始数据的来源和格式。
    不同的源类型可能需要不同的提取策略。

    类型说明:
    ========

    CONVERSATION (对话):
        - 定义: 聊天对话数据，包括群聊、私聊、对话记录
        - 数据格式要求:
            - speaker_id: 发言者ID
            - speaker_name: 发言者名称 (可选)
            - content: 消息内容
            - timestamp: 发送时间
        - 提取策略: 基于对话边界检测，识别话题转换点
        - 输出: MemUnit (记忆单元)

    使用场景:
        - 即时通讯消息 (微信、Slack、Discord 等)
        - 客服对话记录
        - 会议聊天记录
        - 论坛讨论帖
    """

    CONVERSATION = "Conversation"  # 对话数据

    # ===== 未来支持的数据源类型 (取消注释以启用) =====
    # EMAIL = "Email"              # 邮件数据
    # DOCUMENT = "Document"        # 文档数据
    # MEETING = "Meeting"          # 会议记录

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> Optional['SourceType']:
        """
        将字符串转换为 SourceType 枚举值

        支持大小写不敏感的匹配。

        参数:
            type_str: 类型字符串 (如 "Conversation", "conversation", "CONVERSATION")

        返回:
            匹配的 SourceType 枚举值，以下情况返回 None:
            - type_str 为 None 或空字符串
            - 没有找到匹配的枚举成员
            - 转换过程中发生任何错误

        示例:
            >>> SourceType.from_string("Conversation")
            SourceType.CONVERSATION
            >>> SourceType.from_string("conversation")
            SourceType.CONVERSATION
            >>> SourceType.from_string("invalid")
            None
        """
        if not type_str:
            return None

        try:
            enum_name = type_str.upper()
            return getattr(cls, enum_name)

        except AttributeError:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.error(f"未找到匹配的 SourceType: {type_str}，返回 None")
            return None
        except Exception as e:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"类型字段转换失败: {type_str}，错误: {e}")
            return None
