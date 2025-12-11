"""Execution context for workflow orchestration.

This module provides dependency injection for all services and resources needed by nodes.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict
from pathlib import Path


@dataclass
class ExecutionContext:
    """Execution context providing access to all services and resources.

    This context is passed to node functions to access external services and resources.
    It acts as a dependency injection container, avoiding the need for global state.

    Retrieval Services:
        memory_index: Vector database for memory storage
        vectorize_service: Service for generating embeddings
        rerank_service: Service for reranking documents
        cluster_index: Cluster-based index for result expansion
        bm25_index: BM25 index for keyword search
        llm_provider: LLM service for answer generation and checks

    Evaluation Services:
        adapter: System adapter for eval pipeline
        evaluator: Evaluator for answer quality assessment
        output_dir: Output directory for results
        checkpoint_manager: Checkpoint manager for resuming
        logger: Logger instance
        console: Rich console for pretty printing

    Configuration:
        config: Additional configuration dictionary
        project_root: Path to project root directory
    """

    # Core services (retrieval workflows)
    memory_index: Any = None
    vectorize_service: Any = None
    rerank_service: Any = None

    # Optional services (retrieval workflows)
    cluster_index: Optional[Any] = None
    bm25_index: Optional[Any] = None
    llm_provider: Optional[Any] = None

    # Evaluation workflow services
    adapter: Optional[Any] = None
    evaluator: Optional[Any] = None
    output_dir: Optional[Any] = None
    checkpoint_manager: Optional[Any] = None
    logger: Optional[Any] = None
    console: Optional[Any] = None
    token_stats_collector: Optional[Any] = None  # Token usage statistics collector

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    project_root: Optional[Path] = None

    def __post_init__(self):
        """Initialize project root if not provided."""
        if self.project_root is None:
            # Default to current file's parent.parent.parent (src/orchestration -> src -> project root)
            self.project_root = Path(__file__).parent.parent.parent

    def get_prompt(self, prompt_path: str) -> str:
        """Load a prompt template from the prompts directory.

        Args:
            prompt_path: Relative path from prompts/ directory
                        e.g., "retrieval/sufficiency_check.txt"

        Returns:
            Prompt template as string

        Raises:
            FileNotFoundError: If prompt file doesn't exist
        """
        prompts_dir = self.project_root / "prompts"
        full_path = prompts_dir / prompt_path

        if not full_path.exists():
            raise FileNotFoundError(f"Prompt not found: {full_path}")

        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.

        Args:
            key: Configuration key (supports nested keys with dot notation, e.g., "retrieval.top_k")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def update_config(self, **kwargs):
        """Update configuration with new values.

        Args:
            **kwargs: Key-value pairs to update in config
        """
        self.config.update(kwargs)


def create_execution_context(
    memory_index: Any,
    vectorize_service: Any,
    rerank_service: Any,
    cluster_index: Optional[Any] = None,
    bm25_index: Optional[Any] = None,
    llm_provider: Optional[Any] = None,
    **config_kwargs
) -> ExecutionContext:
    """Factory function to create ExecutionContext.

    Args:
        memory_index: Vector database for memory storage
        vectorize_service: Service for generating embeddings
        rerank_service: Service for reranking documents
        cluster_index: Optional cluster-based index
        bm25_index: Optional BM25 index
        llm_provider: Optional LLM service
        **config_kwargs: Additional configuration key-value pairs

    Returns:
        Configured ExecutionContext instance
    """
    return ExecutionContext(
        memory_index=memory_index,
        vectorize_service=vectorize_service,
        rerank_service=rerank_service,
        cluster_index=cluster_index,
        bm25_index=bm25_index,
        llm_provider=llm_provider,
        config=config_kwargs
    )
