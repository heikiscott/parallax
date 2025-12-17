"""Evaluation Workflow State Schema."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from operator import add


class EvalState(TypedDict, total=False):
    """Evaluation workflow state."""
    # Dataset
    dataset: Any
    conversations: List[Any]
    qa_pairs: List[Any]
    conv_id: Optional[int]

    # Results
    index: Optional[Dict[str, Any]]
    search_results: Optional[List[Any]]
    answer_results: Optional[List[Any]]
    eval_results: Optional[Any]

    # Tracking
    metadata: Dict[str, Any]
    completed_stages: Annotated[List[str], add]
    filter_categories: List[int]
