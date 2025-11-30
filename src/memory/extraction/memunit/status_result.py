"""
Status control result for MemUnit extraction.
"""

from dataclasses import dataclass


@dataclass
class StatusResult:
    """Status control result."""

    # 表示下次触发时，这次的对话会累积一起作为new message输入
    should_wait: bool
