"""Utility functions for online retrieval.

Helper functions for datetime handling
and semantic memory filtering.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


def format_datetime_field(value: Any) -> Optional[str]:
    """Format datetime value to ISO string.

    Args:
        value: Datetime object or any value

    Returns:
        ISO formatted string if datetime, otherwise original value
    """
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def parse_datetime_value(value: Any) -> Optional[datetime]:
    """Parse datetime value from various formats.

    Args:
        value: Datetime object, ISO string, or any value

    Returns:
        Parsed datetime or None
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
    return None


def filter_semantic_memories_by_time(
    memories: List[Dict[str, Any]],
    data_source: str,
    current_time: Optional[datetime],
) -> List[Dict[str, Any]]:
    """Filter semantic memories by validity time range.

    Only applies to semantic_memory data source. Filters out memories
    whose start_time > current_time or end_time < current_time.

    Args:
        memories: List of memory dicts
        data_source: Data source type
        current_time: Current time for filtering

    Returns:
        Filtered list of memories
    """
    if data_source != "semantic_memory" or not current_time:
        return memories

    current_dt = (
        current_time
        if isinstance(current_time, datetime)
        else parse_datetime_value(current_time)
    )
    if current_dt is None:
        return memories

    filtered = []
    for memory in memories:
        start_dt = parse_datetime_value(memory.get("start_time"))
        end_dt = parse_datetime_value(memory.get("end_time"))

        if start_dt and start_dt > current_dt:
            continue
        if end_dt and end_dt < current_dt:
            continue
        filtered.append(memory)
    return filtered
