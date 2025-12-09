"""Orchestration Layer - LangGraph Workflow Framework.

This module provides a generic workflow orchestration framework using LangGraph.
It is designed to be domain-agnostic and can orchestrate any type of workflow:
- Retrieval workflows
- Memory extraction workflows
- Conversation processing workflows
- Multi-agent collaboration workflows
- etc.

Usage:
    from orchestration import create_workflow, ExecutionContext
    from orchestration.state import create_initial_retrieval_state

    # Create context with dependencies
    context = ExecutionContext(
        memory_index=my_index,
        bm25_index=my_bm25,
        ...
    )

    # Create and run workflow
    workflow = create_workflow("adaptive_retrieval", context)
    result = await workflow.ainvoke(initial_state)
"""

from .config_loader import ConfigLoader, WorkflowConfig, NodeConfig
from .context import ExecutionContext
from .state import RetrievalState, Document, create_initial_retrieval_state
from .workflow_builder import create_workflow, build_workflow_from_config

__all__ = [
    # Config
    "ConfigLoader",
    "WorkflowConfig",
    "NodeConfig",
    # Context
    "ExecutionContext",
    # State
    "RetrievalState",
    "Document",
    "create_initial_retrieval_state",
    # Workflow
    "create_workflow",
    "build_workflow_from_config",
]
