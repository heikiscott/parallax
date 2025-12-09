"""Question classification node functions for LangGraph workflows.

This module contains node functions for:
- Question classification using the QuestionClassifier
"""

from typing import Dict, Any
from ...state import RetrievalState
from ...context import ExecutionContext
from .. import register_node


@register_node("classify_question_node")
async def classify_question_node(
    state: RetrievalState,
    context: ExecutionContext
) -> Dict[str, Any]:
    """Classify the question type using QuestionClassifier.

    Uses the rule-based QuestionClassifier to determine:
    - Question type (event_temporal, attribute_preference, etc.)
    - Retrieval strategy (GEC_CLUSTER_RERANK, AGENTIC_ONLY, etc.)

    The classification result is stored in state for:
    1. Routing decisions (via route_by_question_type router)
    2. Debugging and analytics

    Args:
        state: Current retrieval state containing query
        context: Execution context with services

    Returns:
        Dict with classification results:
        - question_type: Classified question type
        - retrieval_strategy: Recommended strategy
        - classification_confidence: Confidence score
        - metadata: Additional classification details
    """
    from src.retrieval.classification import classify_question

    query = state["query"]

    # Use the QuestionClassifier for classification
    result = classify_question(query)

    return {
        "question_type": result.question_type.value,
        "retrieval_strategy": result.strategy.value,
        "classification_confidence": result.confidence,
        "metadata": {
            "classification_method": "rule_based",
            "question_type": result.question_type.value,
            "strategy": result.strategy.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "detected_patterns": result.detected_patterns
        }
    }
