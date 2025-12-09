"""Workflow graph builder for LangGraph.

This module builds executable LangGraph StateGraph instances from YAML workflow configurations.
"""

from typing import Callable, Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from functools import partial
from dataclasses import dataclass

from .state import RetrievalState, MemoryBuildingState
from .context import ExecutionContext
from .config_loader import WorkflowConfig, ConfigLoader
from .nodes import get_node


# ============================================================================
# Router Registry
# ============================================================================

ROUTER_REGISTRY: Dict[str, Callable] = {}


def register_router(name: str):
    """Decorator to register a router function."""
    def decorator(func: Callable):
        ROUTER_REGISTRY[name] = func
        return func
    return decorator


def get_router(name: str) -> Callable:
    """Get a router function by name."""
    if name not in ROUTER_REGISTRY:
        raise KeyError(f"Router '{name}' not found. Available: {list(ROUTER_REGISTRY.keys())}")
    return ROUTER_REGISTRY[name]


# ============================================================================
# Built-in Routers
# ============================================================================

@register_router("route_by_question_type")
def route_by_question_type(state: RetrievalState) -> str:
    """Route based on question classification result.

    Uses the question_classifier to determine the retrieval strategy:
    - AGENTIC_ONLY -> lightweight (fast, no LLM)
    - GEC_CLUSTER_RERANK -> agentic (complex event queries)
    - GEC_INSERT_AFTER_HIT -> agentic (needs context expansion)

    For adaptive_retrieval workflow:
    - Simple attribute/preference queries -> lightweight_retrieval
    - Complex event/aggregation queries -> agentic_retrieval
    """
    from src.retrieval.classification import classify_question, RetrievalStrategy

    query = state.get("query", "")
    result = classify_question(query)

    # Store classification result in state for debugging
    # Note: This doesn't modify state, just returns routing decision

    # Route based on strategy
    if result.strategy == RetrievalStrategy.AGENTIC_ONLY:
        # Attribute/preference questions - simple retrieval is enough
        return "lightweight_retrieval"
    elif result.strategy == RetrievalStrategy.GEC_CLUSTER_RERANK:
        # Event questions - need agentic for cluster expansion
        return "agentic_retrieval"
    else:
        # GEC_INSERT_AFTER_HIT and others - use agentic for best coverage
        return "agentic_retrieval"


@register_router("route_by_sufficiency")
def route_by_sufficiency(state: RetrievalState) -> str:
    """Route based on sufficiency check result.

    Returns:
        "sufficient" if results are sufficient
        "insufficient" if need more retrieval
    """
    is_sufficient = state.get("is_sufficient", True)
    return "sufficient" if is_sufficient else "insufficient"


class WorkflowBuilder:
    """Builds LangGraph StateGraph from workflow configuration."""

    def __init__(self, context: ExecutionContext):
        """Initialize workflow builder.

        Args:
            context: Execution context for dependency injection
        """
        self.context = context

    def build_from_config(self, config: WorkflowConfig) -> StateGraph:
        """Build a StateGraph from workflow configuration.

        Args:
            config: Parsed workflow configuration

        Returns:
            Compiled StateGraph ready for execution
        """
        # Determine state schema based on workflow name
        # TODO: Make this more flexible (could be in config)
        state_schema = RetrievalState
        if "memory" in config.name.lower():
            state_schema = MemoryBuildingState

        # Create StateGraph
        graph = StateGraph(state_schema)

        # Add all nodes
        for node_config in config.nodes:
            node_func = get_node(node_config.function)

            # Create a wrapper that injects context
            async def node_wrapper(state, *, func=node_func, ctx=self.context, cfg=node_config.config):
                # Merge node config into context (handle empty config)
                if cfg:
                    ctx.update_config(**cfg)
                # Call the actual node function
                return await func(state, ctx)

            graph.add_node(node_config.name, node_wrapper)

        # Add all regular edges
        for edge_config in config.edges:
            from_node = edge_config.from_node
            to_node = edge_config.to_node

            # Handle special START and END nodes
            if from_node == "START":
                graph.set_entry_point(to_node)
            elif to_node == "END":
                graph.add_edge(from_node, END)
            else:
                graph.add_edge(from_node, to_node)

        # Add all conditional edges
        for cond_edge in config.conditional_edges:
            source = cond_edge.source
            router_func = get_router(cond_edge.router)
            destinations = cond_edge.destinations

            # Build path map for LangGraph
            # Map router return values to node names, handling "END" specially
            path_map = {}
            for route_key, target_node in destinations.items():
                if target_node == "END":
                    path_map[route_key] = END
                else:
                    path_map[route_key] = target_node

            # Add conditional edges using LangGraph's add_conditional_edges
            graph.add_conditional_edges(
                source,
                router_func,
                path_map
            )

        # Compile the graph
        return graph.compile()

    def build_from_yaml(self, config_name: str) -> StateGraph:
        """Build a StateGraph from YAML config file.

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Compiled StateGraph ready for execution
        """
        loader = ConfigLoader()
        config = loader.load(config_name)
        return self.build_from_config(config)


def create_workflow(
    config_name: str,
    context: ExecutionContext
) -> StateGraph:
    """Convenience function to create a workflow from config name.

    Args:
        config_name: Name of the config file (without .yaml extension)
        context: Execution context for dependency injection

    Returns:
        Compiled StateGraph ready for execution

    Example:
        >>> context = ExecutionContext(
        ...     memory_index=my_index,
        ...     vectorize_service=my_vectorizer,
        ...     rerank_service=my_reranker
        ... )
        >>> workflow = create_workflow("simple_retrieval", context)
        >>> result = await workflow.ainvoke({"query": "What happened yesterday?"})
    """
    builder = WorkflowBuilder(context)
    return builder.build_from_yaml(config_name)


def build_workflow_from_config(
    config: WorkflowConfig,
    context: ExecutionContext
) -> StateGraph:
    """Build a workflow from a config object.

    Args:
        config: Parsed workflow configuration
        context: Execution context for dependency injection

    Returns:
        Compiled StateGraph ready for execution
    """
    builder = WorkflowBuilder(context)
    return builder.build_from_config(config)
