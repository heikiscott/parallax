"""Node Registry for LangGraph workflows.

This module provides a registry system for all node functions used in workflows.
Nodes can be registered using the @register_node decorator or explicitly via register_node().

The registry enables dynamic node instantiation from YAML configuration files.

Node Organization:
- retrieval/: Retrieval-related nodes (classification, hybrid search, rerank, etc.)
- (future) memory/: Memory extraction nodes
- (future) conversation/: Conversation processing nodes
"""

from typing import Callable, Dict, Any
from ..state import RetrievalState, MemoryBuildingState
from ..context import ExecutionContext

# Type alias for node functions
NodeFunction = Callable[[Any, ExecutionContext], Any]

# Global registry of all node functions
NODE_REGISTRY: Dict[str, NodeFunction] = {}


def register_node(name: str):
    """Decorator to register a node function.

    Usage:
        @register_node("classify_question")
        async def classify_question_node(state: RetrievalState, context: ExecutionContext):
            # implementation
            return {"question_type": "event_time"}

    Args:
        name: Unique name for the node (used in YAML config)

    Returns:
        Decorator function
    """
    def decorator(func: NodeFunction) -> NodeFunction:
        if name in NODE_REGISTRY:
            raise ValueError(f"Node '{name}' is already registered")
        NODE_REGISTRY[name] = func
        return func
    return decorator


def get_node(name: str) -> NodeFunction:
    """Retrieve a node function by name.

    Args:
        name: Name of the node to retrieve

    Returns:
        The registered node function

    Raises:
        KeyError: If node name not found in registry
    """
    if name not in NODE_REGISTRY:
        raise KeyError(
            f"Node '{name}' not found in registry. "
            f"Available nodes: {list(NODE_REGISTRY.keys())}"
        )
    return NODE_REGISTRY[name]


def list_nodes() -> list:
    """List all registered node names.

    Returns:
        List of registered node names
    """
    return list(NODE_REGISTRY.keys())


def register_node_function(name: str, func: NodeFunction):
    """Explicitly register a node function without using decorator.

    Args:
        name: Unique name for the node
        func: The node function to register

    Raises:
        ValueError: If node name already registered
    """
    if name in NODE_REGISTRY:
        raise ValueError(f"Node '{name}' is already registered")
    NODE_REGISTRY[name] = func


# Import all node modules to trigger registration
# Retrieval nodes
try:
    from .retrieval import classification_nodes  # noqa: F401
except ImportError:
    pass

try:
    from .retrieval import legacy_nodes  # noqa: F401
except ImportError:
    pass

# Re-export commonly used nodes for convenience
try:
    from .retrieval import (
        classify_question_node,
        lightweight_retrieval_node,
        agentic_retrieval_node,
    )
except ImportError:
    pass


__all__ = [
    # Registry
    'NODE_REGISTRY',
    'register_node',
    'get_node',
    'list_nodes',
    'register_node_function',
    # Commonly used nodes
    "classify_question_node",
    "lightweight_retrieval_node",
    "agentic_retrieval_node",
]
