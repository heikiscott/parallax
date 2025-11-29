"""Raw data type enumeration."""

from enum import Enum
from typing import Optional


class RawDataType(Enum):
    """Types of content that can be processed."""

    CONVERSATION = "Conversation"

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> Optional['RawDataType']:
        """
        将字符串类型转换为RawDataType枚举

        Args:
            type_str: 类型字符串，如 "Conversation", "Email" 等

        Returns:
            RawDataType枚举值，如果转换失败则返回None
        """
        if not type_str:
            return None

        try:
            # 将字符串转换为枚举名称格式（如 "Conversation" -> "CONVERSATION"）
            enum_name = type_str.upper()
            return getattr(cls, enum_name)

        except AttributeError:
            # 如果没有找到对应的枚举，返回None
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.error(f"未找到匹配的RawDataType: {type_str}，返回None")
            return None
        except Exception as e:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"转换type字段失败: {type_str}, error: {e}")
            return None
