"""Question classification for retrieval routing.

This module provides question classification to determine the best retrieval strategy:
- GEC_CLUSTER_RERANK: LLM selects clusters + Agentic fallback (for event-type questions)
- GEC_INSERT_AFTER_HIT: Original retrieval + cluster expansion (for reasoning/general questions)
- AGENTIC_ONLY: Pure Agentic retrieval (for attribute/preference questions)

Complexity levels for adaptive V4 routing:
- SIMPLE: Skip C-RAG evaluation, return Round 1 directly (saves ~300ms)
- MODERATE: Standard C-RAG flow (Round 1 → eval → maybe Round 2)
- COMPLEX: Force Round 2, higher recall budget
"""

from .question_classifier import (
    QuestionType,
    RetrievalStrategy,
    ComplexityLevel,
    ClassificationResult,
    QuestionClassifier,
    LLMQuestionClassifier,
    classify_question,
    should_use_group_event_cluster,
)

__all__ = [
    "QuestionType",
    "RetrievalStrategy",
    "ComplexityLevel",
    "ClassificationResult",
    "QuestionClassifier",
    "LLMQuestionClassifier",
    "classify_question",
    "should_use_group_event_cluster",
]
