"""
MemUnit extraction request data class.
"""

from dataclasses import dataclass
from typing import List, Optional

from memory.schema import Memory
from .raw_data import RawData


@dataclass
class MemUnitExtractRequest:
    """Request for extracting MemUnit from raw data."""

    history_raw_data_list: List[RawData]
    new_raw_data_list: List[RawData]
    # 整个群的user id
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None

    old_memory_list: Optional[List[Memory]] = None
    smart_mask_flag: Optional[bool] = False
