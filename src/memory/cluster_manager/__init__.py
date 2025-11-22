"""Cluster Manager - Automatic clustering of memunits with event notifications.

This module provides ClusterManager, a core component that clusters memunits
based on semantic similarity and temporal proximity, with event hooks for
downstream processing.

Key Features:
- Incremental clustering using embeddings and timestamps
- Event notifications on cluster assignments
- Flexible storage backends for cluster state
- Seamless integration with MemUnitExtractor

Usage:
    from memory.cluster_manager import ClusterManager, ClusterManagerConfig
    
    # Initialize
    config = ClusterManagerConfig(
        similarity_threshold=0.65,
        max_time_gap_days=7,
        enable_persistence=True
    )
    cluster_mgr = ClusterManager(config)
    
    # Attach to memunit extractor
    cluster_mgr.attach_to_extractor(memunit_extractor)
    
    # Register callbacks for cluster events
    cluster_mgr.on_cluster_assigned(my_callback)
    
    # Clusters are automatically assigned, callbacks notified!
"""

from memory.cluster_manager.config import ClusterManagerConfig
from memory.cluster_manager.manager import ClusterManager
from memory.cluster_manager.storage import (
    ClusterStorage,
    InMemoryClusterStorage,
)
from memory.cluster_manager.mongo_cluster_storage import MongoClusterStorage

__all__ = [
    "ClusterManager",
    "ClusterManagerConfig",
    "ClusterStorage",
    "InMemoryClusterStorage",
    "MongoClusterStorage",
]

