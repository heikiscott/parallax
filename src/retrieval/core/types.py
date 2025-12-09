"""Core data types for the retrieval system."""

from typing import TypedDict, Dict, Any, Optional


class Document(TypedDict, total=False):
    """Document representation for retrieval system.

    This is the core data structure used across all retrieval components
    (retrievers, rerankers, expanders).

    Fields:
        id: Unique identifier for the document
        content: The main text content of the document
        score: Relevance score (optional, set by retrievers/rerankers)
        metadata: Additional metadata (retrieval method, original memory object, etc.)
    """
    id: str
    content: str
    score: Optional[float]
    metadata: Dict[str, Any]
