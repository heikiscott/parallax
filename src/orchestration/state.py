"""LangGraph State definitions for workflow orchestration.

This module defines the state structures shared across all nodes in workflows.
"""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


# Define Document type locally to avoid circular imports
class Document(TypedDict, total=False):
    """Document representation in retrieval workflows.

    Fields:
        id: Unique document identifier
        content: Document text content
        score: Relevance score
        metadata: Additional document metadata
    """
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class RetrievalState(TypedDict, total=False):
    """State for retrieval workflow.

    This state is shared across all nodes in the LangGraph workflow.
    Nodes read from and write to this state to pass information between steps.

    Fields:
        query: The user's query string
        question_type: Classification of the question (e.g., "event_time", "activity", "attribute")
        retrieval_strategy: Selected retrieval strategy based on question type
        documents: Retrieved documents (replaces previous results)
        all_documents: Accumulated documents from multiple retrieval rounds (uses add reducer)
        top_k: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking
        metadata: Metadata dictionary for tracking workflow execution
        error: Error message if any step fails

    Reducers:
        - all_documents: Uses `add` operator to accumulate documents from multiple rounds
        - metadata: Automatically merges dict updates
    """

    # Input
    query: str
    question_type: Optional[str]
    retrieval_strategy: Optional[str]

    # Document storage
    documents: List[Document]  # Current documents (replaces previous)
    all_documents: Annotated[List[Document], add]  # Accumulated across rounds

    # Configuration
    top_k: int
    rerank_top_k: int

    # Metadata & Error tracking
    metadata: Dict[str, Any]
    error: Optional[str]


class MemoryBuildingState(TypedDict, total=False):
    """State for memory building workflow.

    Used in the memory construction phase to build the knowledge base.
    """

    # Input
    conversation_text: str
    conversation_id: str

    # Extracted units
    mem_units: List[Dict[str, Any]]
    memories: List[Dict[str, Any]]

    # Embeddings & Storage
    embeddings: List[List[float]]
    stored_ids: List[str]

    # Metadata
    metadata: Dict[str, Any]
    error: Optional[str]


def create_initial_retrieval_state(
    query: str,
    top_k: int = 50,
    rerank_top_k: int = 20
) -> RetrievalState:
    """Create initial retrieval state with default values.

    Args:
        query: The user's question
        top_k: Number of documents to retrieve (default: 50)
        rerank_top_k: Number of documents after reranking (default: 20)

    Returns:
        Initial RetrievalState ready for workflow execution
    """
    return RetrievalState(
        query=query,
        question_type=None,
        retrieval_strategy=None,
        documents=[],
        all_documents=[],
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        metadata={},
        error=None
    )


def create_initial_memory_state(
    conversation_text: str,
    conversation_id: str
) -> MemoryBuildingState:
    """Create initial memory building state.

    Args:
        conversation_text: The conversation text to process
        conversation_id: Unique identifier for the conversation

    Returns:
        Initial MemoryBuildingState ready for workflow execution
    """
    return MemoryBuildingState(
        conversation_text=conversation_text,
        conversation_id=conversation_id,
        mem_units=[],
        memories=[],
        embeddings=[],
        stored_ids=[],
        metadata={},
        error=None
    )
