"""Source type enumeration for input data."""

from enum import Enum
from typing import Optional


class SourceType(Enum):
    """Types of input data sources that can be processed."""

    CONVERSATION = "Conversation"

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> Optional['SourceType']:
        """
        Convert string to SourceType enum.

        Args:
            type_str: Type string, e.g. "Conversation", "Email", etc.

        Returns:
            SourceType enum value, or None if conversion fails.
        """
        if not type_str:
            return None

        try:
            enum_name = type_str.upper()
            return getattr(cls, enum_name)

        except AttributeError:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.error(f"No matching SourceType found: {type_str}, returning None")
            return None
        except Exception as e:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to convert type field: {type_str}, error: {e}")
            return None


# Backward compatibility alias
RawDataType = SourceType
