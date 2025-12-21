"""Classify stage prompts."""

from .batch_classification_prompts import (
    BATCH_CLASSIFICATION_PROMPT,
    CATEGORY_TO_QUESTION_TYPE,
    VALID_CATEGORIES,
    get_batch_classification_prompt,
)

from .single_classification_prompts import (
    SINGLE_CLASSIFICATION_PROMPT,
    get_single_classification_prompt,
)

__all__ = [
    # Batch classification
    "BATCH_CLASSIFICATION_PROMPT",
    "CATEGORY_TO_QUESTION_TYPE",
    "VALID_CATEGORIES",
    "get_batch_classification_prompt",
    # Single classification
    "SINGLE_CLASSIFICATION_PROMPT",
    "get_single_classification_prompt",
]
