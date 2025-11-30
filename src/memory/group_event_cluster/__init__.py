"""Group Event Cluster - LLM-driven event clustering for memory retrieval enhancement.

This module provides:
- GroupEventCluster: A semantic cluster of related MemUnits
- GroupEventClusterIndex: Index for managing clusters and lookups
- GroupEventClusterer: LLM-driven clustering engine
- Cluster-enhanced retrieval functions

Usage:
    from memory.group_event_cluster import (
        GroupEventClusterer,
        GroupEventClusterConfig,
        GroupEventClusterIndex,
        expand_with_cluster,
        ClusterRetrievalConfig,
    )

    # Configure and create clusterer
    config = GroupEventClusterConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )
    clusterer = GroupEventClusterer(config)

    # Cluster MemUnits
    index = await clusterer.cluster_memunits(memunit_list, conversation_id="conv_0")

    # Save index
    index.save_to_file(Path("clusters/conv_0.json"))

    # Use for retrieval enhancement
    retrieval_config = ClusterRetrievalConfig()
    expanded_results, metadata = expand_with_cluster(
        original_results=search_results,
        cluster_index=index,
        config=retrieval_config,
        all_docs_map=docs_map,
    )
"""

from .schema import (
    ClusterMember,
    GroupEventCluster,
    GroupEventClusterIndex,
)
from .config import (
    GroupEventClusterConfig,
    ClusterRetrievalConfig,
)
from .storage import (
    ClusterStorage,
    JsonClusterStorage,
    InMemoryClusterStorage,
)
from .retrieval import (
    expand_with_cluster,
    get_related_units_for_query,
)
from .clusterer import GroupEventClusterer

__all__ = [
    # Schema
    "ClusterMember",
    "GroupEventCluster",
    "GroupEventClusterIndex",
    # Config
    "GroupEventClusterConfig",
    "ClusterRetrievalConfig",
    # Storage
    "ClusterStorage",
    "JsonClusterStorage",
    "InMemoryClusterStorage",
    # Retrieval
    "expand_with_cluster",
    "get_related_units_for_query",
    # Clusterer
    "GroupEventClusterer",
]
