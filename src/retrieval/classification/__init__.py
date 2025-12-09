"""Question classification for retrieval routing.

This module provides question classification to determine the best retrieval strategy:
- GEC_CLUSTER_RERANK: LLM selects clusters + Agentic fallback (for event-type questions)
- GEC_INSERT_AFTER_HIT: Original retrieval + cluster expansion (for reasoning/general questions)
- AGENTIC_ONLY: Pure Agentic retrieval (for attribute/preference questions)
"""

from .question_classifier import (
    QuestionType,
    RetrievalStrategy,
    ClassificationResult,
    QuestionClassifier,
    LLMQuestionClassifier,
    classify_question,
    should_use_group_event_cluster,
)

__all__ = [
    "QuestionType",
    "RetrievalStrategy",
    "ClassificationResult",
    "QuestionClassifier",
    "LLMQuestionClassifier",
    "classify_question",
    "should_use_group_event_cluster",
]
