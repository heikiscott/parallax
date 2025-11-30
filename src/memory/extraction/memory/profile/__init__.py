"""
Profile memory extraction module.

This module provides classes for extracting profile memories from MemUnits.
"""

from .types import (
    ProjectInfo,
    ImportanceEvidence,
    GroupImportanceEvidence,
    ProfileMemoryExtractRequest,
)
from .profile_helpers import (
    remove_evidences_from_profile,
    accumulate_old_memory_entry,
    profile_payload_to_memory,
    merge_single_profile,
    merge_profiles,
)
from .project_helpers import (
    project_to_dict,
    convert_projects_to_dataclass,
    merge_projects_participated,
    filter_project_items_by_type,
)
from .skill_helpers import (
    normalize_skills_with_evidence,
    merge_skill_lists,
    merge_skill_lists_keep_highest_level,
)
from .value_helpers import (
    merge_value_with_evidences_lists,
    merge_value_with_evidences_lists_keep_highest_level,
    extract_values_with_evidence,
)
from .evidence_utils import (
    ensure_str_list,
    format_evidence_entry,
    conversation_id_from_evidence,
    merge_evidences_recursive,
)
from .profile_conversation import (
    extract_user_mapping_from_memunits,
    build_conversation_text,
    build_episode_text,
    annotate_relative_dates,
    extract_group_important_info,
    is_important_to_user,
    merge_group_importance_evidence,
    build_profile_prompt,
    build_evidence_completion_prompt,
)
from .empty_evidence_completion import complete_missing_evidences
from .profile_memory_extractor import ProfileMemoryExtractor
from .profile_memory_merger import ProfileMemoryMerger, convert_important_info_to_evidence

__all__ = [
    # Main classes
    "ProfileMemoryExtractor",
    "ProfileMemoryMerger",
    "convert_important_info_to_evidence",
    # Types
    "ProjectInfo",
    "ImportanceEvidence",
    "GroupImportanceEvidence",
    "ProfileMemoryExtractRequest",
    # Profile helpers
    "remove_evidences_from_profile",
    "accumulate_old_memory_entry",
    "profile_payload_to_memory",
    "merge_single_profile",
    "merge_profiles",
    # Project helpers
    "project_to_dict",
    "convert_projects_to_dataclass",
    "merge_projects_participated",
    "filter_project_items_by_type",
    # Skill helpers
    "normalize_skills_with_evidence",
    "merge_skill_lists",
    "merge_skill_lists_keep_highest_level",
    # Value helpers
    "merge_value_with_evidences_lists",
    "merge_value_with_evidences_lists_keep_highest_level",
    "extract_values_with_evidence",
    # Evidence utils
    "ensure_str_list",
    "format_evidence_entry",
    "conversation_id_from_evidence",
    "merge_evidences_recursive",
    # Conversation utils
    "extract_user_mapping_from_memunits",
    "build_conversation_text",
    "build_episode_text",
    "annotate_relative_dates",
    "extract_group_important_info",
    "is_important_to_user",
    "merge_group_importance_evidence",
    "build_profile_prompt",
    "build_evidence_completion_prompt",
    # Evidence completion
    "complete_missing_evidences",
]
