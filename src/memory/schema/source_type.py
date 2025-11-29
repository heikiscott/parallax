"""
Source Type Enumeration.

This module defines the types of input data sources that can be processed
by the memory extraction system.

Currently supported sources:
- CONVERSATION: Chat messages, group discussions, dialogues

Future sources (planned):
- EMAIL: Email threads
- DOCUMENT: Documents, notes, articles
- MEETING: Meeting transcripts

Usage:
------
    from memory.schema import SourceType

    raw_data = RawData(
        content={"speaker_id": "user_1", "content": "Hello"},
        data_type=SourceType.CONVERSATION
    )
"""

from enum import Enum
from typing import Optional


class SourceType(Enum):
    """
    Types of input data sources that can be processed.

    This enum identifies the origin/format of raw data being processed.
    Different source types may require different extraction strategies.

    Attributes:
        CONVERSATION: Chat conversation data.
            Includes group chats, direct messages, and dialogues.
            Expected format: messages with speaker_id, content, timestamp.
    """

    CONVERSATION = "Conversation"

    # Future source types (uncomment when implemented):
    # EMAIL = "Email"
    # DOCUMENT = "Document"
    # MEETING = "Meeting"

    @classmethod
    def from_string(cls, type_str: Optional[str]) -> Optional['SourceType']:
        """
        Convert a string to SourceType enum value.

        Performs case-insensitive matching against enum member names.

        Args:
            type_str: Type string to convert (e.g., "Conversation", "conversation", "CONVERSATION").

        Returns:
            Matching SourceType enum value, or None if:
            - type_str is None or empty
            - No matching enum member found
            - Any conversion error occurs

        Examples:
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
            logger.error(f"No matching SourceType found: {type_str}, returning None")
            return None
        except Exception as e:
            from core.observation.logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Failed to convert type field: {type_str}, error: {e}")
            return None
