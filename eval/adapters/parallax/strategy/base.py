"""Base classes for retrieval strategies.

This module defines the abstract base class and data structures for retrieval strategies.
All concrete strategies should inherit from BaseRetrievalStrategy.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum

# Use TYPE_CHECKING to avoid import issues during testing
if TYPE_CHECKING:
    from memory.group_event_cluster import GroupEventClusterIndex
    from providers.llm.llm_provider import LLMProvider


class StrategyType(Enum):
    """Enumeration of available retrieval strategies."""
    GEC_CLUSTER_RERANK = "gec_cluster_rerank"
    GEC_INSERT_AFTER_HIT = "gec_insert_after_hit"
    AGENTIC_ONLY = "agentic_only"


@dataclass
class RetrievalContext:
    """Context object containing all resources needed for retrieval.

    This dataclass encapsulates all the indices, providers, and configuration
    needed to execute a retrieval strategy. It's passed to strategy.retrieve()
    to avoid tight coupling between strategies and specific resource acquisition.

    Attributes:
        emb_index: Embedding index for semantic search
        bm25: BM25 index for keyword search
        docs: List of documents (MemUnits)
        cluster_index: Optional cluster index for GEC strategies
        llm_provider: LLM provider for agentic retrieval
        llm_config: LLM configuration dictionary
        config: Experiment configuration
    """
    emb_index: Any  # Embedding index
    bm25: Any  # BM25 index
    docs: List[dict]
    cluster_index: Optional[Any] = None  # GroupEventClusterIndex
    llm_provider: Optional[Any] = None  # LLMProvider
    llm_config: Optional[Dict[str, Any]] = None
    config: Optional[Any] = None  # ExperimentConfig

    # Optional: traversal stats tracking
    enable_traversal_stats: bool = False


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.

    Attributes:
        results: List of (document, score) tuples
        metadata: Dictionary containing retrieval metadata
        strategy_type: The strategy that produced this result
    """
    results: List[Tuple[dict, float]]
    metadata: Dict[str, Any]
    strategy_type: StrategyType

    @property
    def count(self) -> int:
        """Number of results retrieved."""
        return len(self.results)

    @property
    def unit_ids(self) -> List[str]:
        """List of unit_ids in results."""
        return [doc.get("unit_id", "") for doc, _ in self.results]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": [
                {"document": doc, "score": score}
                for doc, score in self.results
            ],
            "metadata": self.metadata,
            "strategy_type": self.strategy_type.value,
            "count": self.count,
        }


class BaseRetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies.

    All retrieval strategies should inherit from this class and implement
    the retrieve() method. The strategy encapsulates the retrieval logic
    and can be dynamically selected based on question classification.

    Example:
        class MyCustomStrategy(BaseRetrievalStrategy):
            async def retrieve(self, query: str, context: RetrievalContext) -> RetrievalResult:
                # Custom retrieval logic
                results = await my_search(query, context.emb_index)
                return RetrievalResult(
                    results=results,
                    metadata={"strategy": "custom"},
                    strategy_type=self.strategy_type
                )
    """

    def __init__(self, strategy_type: StrategyType):
        """Initialize with strategy type.

        Args:
            strategy_type: The type of this strategy
        """
        self._strategy_type = strategy_type

    @property
    def strategy_type(self) -> StrategyType:
        """The type of this strategy."""
        return self._strategy_type

    @property
    def name(self) -> str:
        """Human-readable name of this strategy."""
        return self._strategy_type.value

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: RetrievalContext,
    ) -> RetrievalResult:
        """Execute the retrieval strategy.

        Args:
            query: The user's query string
            context: RetrievalContext containing indices and configuration

        Returns:
            RetrievalResult containing results and metadata
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.strategy_type.value})"
