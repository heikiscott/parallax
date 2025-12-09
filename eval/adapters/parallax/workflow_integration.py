"""LangGraph Workflow Integration for Parallax Adapter.

This module provides helpers to integrate LangGraph workflows into the
Parallax evaluation pipeline, enabling YAML-driven retrieval configuration.

Usage:
    from eval.adapters.parallax.workflow_integration import (
        create_retrieval_workflow,
        run_workflow_retrieval,
    )

    # Create workflow once per conversation
    workflow = create_retrieval_workflow(
        workflow_name="adaptive_retrieval",
        context=execution_context
    )

    # Run retrieval for each query
    results, metadata = await run_workflow_retrieval(
        workflow=workflow,
        query="When did Alice go to the park?",
        top_k=20
    )
"""

import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def create_execution_context(
    memory_index: Any,
    bm25_index: Any,
    docs: List[Dict],
    cluster_index: Any = None,
    vectorize_service: Any = None,
    rerank_service: Any = None,
    llm_provider: Any = None,
    config: Optional[Dict] = None,
    project_root: Optional[Path] = None,
):
    """Create an ExecutionContext for LangGraph workflow.

    This wraps all the retrieval dependencies into a context object
    that can be passed to workflow nodes.

    Args:
        memory_index: Embedding index (from stage2)
        bm25_index: BM25 index object
        docs: List of document dicts for BM25
        cluster_index: Optional GroupEventClusterIndex
        vectorize_service: Optional vectorization service
        rerank_service: Optional rerank service
        llm_provider: Optional LLMProvider for agentic retrieval
        config: Optional config dict to merge into context
        project_root: Project root path for config file discovery

    Returns:
        ExecutionContext object ready for workflow execution
    """
    from src.orchestration.context import ExecutionContext

    # Create a wrapper for memory_index that provides get_all_docs
    class MemoryIndexWrapper:
        def __init__(self, emb_index, documents):
            self._emb_index = emb_index
            self._docs = documents
            # Expose the underlying index's attributes
            if emb_index is not None:
                self.embeddings = getattr(emb_index, 'embeddings', None)
                self.doc_ids = getattr(emb_index, 'doc_ids', None)

        def get_all_docs(self):
            """Return all documents for BM25 compatibility."""
            return self._docs

        @property
        def memories(self):
            """Compatibility property for legacy node code."""
            # Convert docs list to dict with unit_id as key
            class MemoryMock:
                def __init__(self, doc):
                    self.id = doc.get("unit_id", "")
                    self.narrative = doc.get("narrative", "")

            return {
                doc.get("unit_id", str(i)): MemoryMock(doc)
                for i, doc in enumerate(self._docs)
            }

        def __len__(self):
            """Return number of documents for len() compatibility."""
            # Prefer emb_index length if available (for embedding-based operations)
            if self._emb_index is not None and hasattr(self._emb_index, '__len__'):
                return len(self._emb_index)
            return len(self._docs)

        def __iter__(self):
            """Iterate over emb_index for embedding search compatibility.

            The emb_index is a list of {"doc": {...}, "embeddings": {...}} dicts,
            which is the format expected by search_utils.search_with_emb_index().
            """
            if self._emb_index is not None:
                return iter(self._emb_index)
            return iter(self._docs)

        def __getattr__(self, name):
            # Delegate unknown attributes to the underlying index
            if self._emb_index is not None:
                return getattr(self._emb_index, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    wrapped_memory_index = MemoryIndexWrapper(memory_index, docs)

    context = ExecutionContext(
        memory_index=wrapped_memory_index,
        bm25_index=bm25_index,
        cluster_index=cluster_index,
        vectorize_service=vectorize_service,
        rerank_service=rerank_service,
        llm_provider=llm_provider,
        project_root=project_root or Path(__file__).parent.parent.parent.parent,
    )

    # Merge additional config if provided
    if config:
        context.config.update(config)

    return context


def create_retrieval_workflow(
    workflow_name: str,
    context: Any,
):
    """Create a LangGraph retrieval workflow.

    Args:
        workflow_name: Name of the workflow (e.g., "adaptive_retrieval")
        context: ExecutionContext with all dependencies

    Returns:
        Compiled LangGraph workflow ready for execution
    """
    from src.orchestration import create_workflow

    workflow = create_workflow(workflow_name, context)
    logger.debug(f"Created workflow '{workflow_name}' with nodes: {list(workflow.nodes.keys())}")

    return workflow


async def run_workflow_retrieval(
    workflow: Any,
    query: str,
    top_k: int = 50,
    rerank_top_k: int = 20,
) -> Tuple[List[Tuple[Dict, float]], Dict[str, Any]]:
    """Run retrieval workflow and return results.

    Args:
        workflow: Compiled LangGraph workflow
        query: User query string
        top_k: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking

    Returns:
        Tuple of (results, metadata) where:
        - results: List of (doc_dict, score) tuples
        - metadata: Dict with retrieval metadata
    """
    from src.orchestration.state import create_initial_retrieval_state

    # Create initial state
    initial_state = create_initial_retrieval_state(
        query=query,
        top_k=top_k,
        rerank_top_k=rerank_top_k
    )

    # Execute workflow
    result = await workflow.ainvoke(initial_state)

    # Extract documents and convert to (doc, score) format
    documents = result.get("documents", [])

    # Convert Document TypedDicts back to (doc, score) format for compatibility
    results = []
    for doc in documents:
        # Document TypedDict has: id, content, score, metadata
        doc_dict = {
            "unit_id": doc.get("id", ""),
            "narrative": doc.get("content", ""),
        }
        # Merge original_doc from metadata if available
        if "metadata" in doc and "original_doc" in doc["metadata"]:
            doc_dict.update(doc["metadata"]["original_doc"])

        score = doc.get("score", 0.0)
        results.append((doc_dict, score))

    # Build metadata
    metadata = result.get("metadata", {})
    metadata["workflow_executed"] = True
    metadata["question_type"] = result.get("question_type")

    return results, metadata


async def workflow_search(
    query: str,
    workflow_name: str,
    emb_index: Any,
    bm25: Any,
    docs: List[Dict],
    cluster_index: Any = None,
    llm_provider: Any = None,
    config: Optional[Dict] = None,
    top_k: int = 50,
    rerank_top_k: int = 20,
) -> Tuple[List[Tuple[Dict, float]], Dict[str, Any]]:
    """One-shot workflow-based retrieval.

    Convenience function that creates context, workflow, and runs retrieval
    in a single call. Good for simple use cases.

    Args:
        query: User query string
        workflow_name: Name of the workflow to use
        emb_index: Embedding index
        bm25: BM25 index object
        docs: Document list for BM25
        cluster_index: Optional cluster index
        llm_provider: Optional LLM provider
        config: Optional config dict
        top_k: Number of documents to retrieve
        rerank_top_k: Number of documents after reranking

    Returns:
        Tuple of (results, metadata)
    """
    # Create context
    context = create_execution_context(
        memory_index=emb_index,
        bm25_index=bm25,
        docs=docs,
        cluster_index=cluster_index,
        llm_provider=llm_provider,
        config=config,
    )

    # Create workflow
    workflow = create_retrieval_workflow(workflow_name, context)

    # Run retrieval
    return await run_workflow_retrieval(
        workflow=workflow,
        query=query,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )
