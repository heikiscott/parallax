"""Data structures for Group Event Cluster system.

This module defines the core data structures for clustering MemUnits by events
from a group perspective (third-person narrative).
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class ClusterMember:
    """
    Cluster member - records a MemUnit's information within a Cluster.

    Members are always sorted by timestamp within a cluster.
    """

    unit_id: str
    """MemUnit's unique identifier"""

    timestamp: datetime
    """MemUnit's timestamp, used for sorting"""

    summary: str
    """Brief summary of this MemUnit (1-2 sentences)"""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "unit_id": self.unit_id,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClusterMember":
        """Deserialize from dictionary."""
        timestamp = data["timestamp"]
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            unit_id=data["unit_id"],
            timestamp=timestamp,
            summary=data["summary"],
        )


@dataclass
class GroupEventCluster:
    """
    Group Event Cluster - a set of semantically related MemUnits (sorted by time).

    Design principles:
    - cluster_id: Unique identifier for indexing and reference
    - topic: Short topic name for LLM to determine new MemUnit membership
    - summary: Detailed third-person group perspective description
    - members: Member list, always sorted by timestamp ascending
    """

    # === Identification ===
    cluster_id: str
    """
    Unique identifier.
    Format: "gec_{index:03d}", e.g., "gec_001", "gec_002"
    gec = Group Event Cluster
    """

    # === Topic fields ===
    topic: str
    """
    Topic name (short, 10-30 characters).
    Usage:
    1. Used as Cluster identifier when LLM makes decisions
    2. Displayed in retrieval results
    Examples:
    - "Caroline's adoption plan"
    - "Melanie's book sharing"
    - "Weekend picnic activity"
    """

    summary: str
    """
    Summary description (detailed, 100-300 characters).
    Features:
    1. Third-person group perspective
    2. Contains key facts (people, time, place, events)
    3. Updated as Cluster members increase
    Usage:
    1. Provides rich context during retrieval
    2. Can serve as additional retrieval target (optional)
    """

    # === Member fields (sorted by time) ===
    members: List[ClusterMember] = field(default_factory=list)
    """
    Member list, always sorted by timestamp ascending.
    Auto-sorted after adding new members.
    """

    # === Time fields ===
    first_timestamp: Optional[datetime] = None
    """Earliest MemUnit timestamp"""

    last_timestamp: Optional[datetime] = None
    """Latest MemUnit timestamp"""

    # === Metadata ===
    created_at: datetime = field(default_factory=datetime.now)
    """Cluster creation time"""

    updated_at: datetime = field(default_factory=datetime.now)
    """Cluster last update time"""

    def add_member(self, member: ClusterMember) -> None:
        """
        Add a member and maintain time ordering.

        Args:
            member: The ClusterMember to add
        """
        self.members.append(member)
        self.members.sort(key=lambda m: m.timestamp)

        # Update time range
        self.first_timestamp = self.members[0].timestamp
        self.last_timestamp = self.members[-1].timestamp
        self.updated_at = datetime.now()

    def get_member_ids(self) -> List[str]:
        """Get all member unit_ids in time order."""
        return [m.unit_id for m in self.members]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "topic": self.topic,
            "summary": self.summary,
            "members": [m.to_dict() for m in self.members],
            "first_timestamp": self.first_timestamp.isoformat() if self.first_timestamp else None,
            "last_timestamp": self.last_timestamp.isoformat() if self.last_timestamp else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupEventCluster":
        """Deserialize from dictionary."""
        members = [ClusterMember.from_dict(m) for m in data.get("members", [])]

        first_ts = data.get("first_timestamp")
        last_ts = data.get("last_timestamp")
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        return cls(
            cluster_id=data["cluster_id"],
            topic=data["topic"],
            summary=data["summary"],
            members=members,
            first_timestamp=datetime.fromisoformat(first_ts) if first_ts else None,
            last_timestamp=datetime.fromisoformat(last_ts) if last_ts else None,
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.now(),
        )


@dataclass
class GroupEventClusterIndex:
    """
    Group Event Cluster Index - manages all Clusters and their mappings.

    Core functionality:
    1. Stores all GroupEventClusters
    2. Maintains unit_id -> cluster_id mapping (bidirectional lookup)
    3. Supports serialization/deserialization (JSON persistence)
    4. All returned member lists are sorted by time
    """

    # === Data storage ===
    clusters: Dict[str, GroupEventCluster] = field(default_factory=dict)
    """
    cluster_id -> GroupEventCluster mapping.
    Primary storage for all Clusters.
    """

    unit_to_clusters: Dict[str, List[str]] = field(default_factory=dict)
    """
    unit_id -> [cluster_id, ...] mapping (one-to-many).
    A MemUnit can belong to multiple Clusters if it discusses multiple topics.
    Used for quick lookup of which Clusters a MemUnit belongs to.
    """

    # === Metadata ===
    conversation_id: str = ""
    """Conversation ID this index belongs to"""

    total_units: int = 0
    """Total number of MemUnits"""

    created_at: datetime = field(default_factory=datetime.now)
    """Index creation time"""

    updated_at: datetime = field(default_factory=datetime.now)
    """Index last update time"""

    llm_model: str = ""
    """LLM model used for clustering"""

    # === Query methods ===
    def get_cluster(self, cluster_id: str) -> Optional[GroupEventCluster]:
        """Get Cluster by cluster_id."""
        return self.clusters.get(cluster_id)

    def get_clusters_by_unit(self, unit_id: str) -> List[GroupEventCluster]:
        """Get all Clusters that a MemUnit belongs to (supports multi-cluster assignment)."""
        cluster_ids = self.unit_to_clusters.get(unit_id, [])
        return [self.clusters[cid] for cid in cluster_ids if cid in self.clusters]

    def get_cluster_ids_by_unit(self, unit_id: str) -> List[str]:
        """Get all cluster_ids for a unit_id."""
        return self.unit_to_clusters.get(unit_id, [])

    def get_units_by_cluster(self, cluster_id: str) -> List[str]:
        """
        Get all member unit_ids of a Cluster.
        Returns time-sorted list.
        """
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return []
        return [m.unit_id for m in cluster.members]

    def get_related_units(self, unit_id: str, exclude_self: bool = True) -> List[str]:
        """
        Get other MemUnits in the same Cluster(s).
        Returns deduplicated, time-sorted list from ALL clusters the unit belongs to.
        This is the core method for retrieval expansion.
        """
        clusters = self.get_clusters_by_unit(unit_id)
        if not clusters:
            return []

        # Collect all related units from all clusters, deduplicate
        related_units = set()
        for cluster in clusters:
            for member in cluster.members:
                if not exclude_self or member.unit_id != unit_id:
                    related_units.add(member.unit_id)

        return list(related_units)

    def get_cluster_topics(self, unit_id: str) -> List[str]:
        """Get all topics of Clusters that unit_id belongs to."""
        clusters = self.get_clusters_by_unit(unit_id)
        return [c.topic for c in clusters if c.topic]

    # === Modification methods ===
    def add_cluster(self, cluster: GroupEventCluster) -> None:
        """Add a new cluster to the index."""
        self.clusters[cluster.cluster_id] = cluster
        # Update unit_to_clusters mapping (append, don't overwrite)
        for member in cluster.members:
            if member.unit_id not in self.unit_to_clusters:
                self.unit_to_clusters[member.unit_id] = []
            if cluster.cluster_id not in self.unit_to_clusters[member.unit_id]:
                self.unit_to_clusters[member.unit_id].append(cluster.cluster_id)
        self.updated_at = datetime.now()

    def add_unit_to_cluster(
        self,
        cluster_id: str,
        member: ClusterMember
    ) -> bool:
        """
        Add a MemUnit to an existing cluster.
        Supports multi-cluster assignment (unit can belong to multiple clusters).

        Returns:
            True if successful, False if cluster not found
        """
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return False

        # Check if unit already in this cluster
        existing_unit_ids = [m.unit_id for m in cluster.members]
        if member.unit_id in existing_unit_ids:
            return True  # Already a member, skip

        cluster.add_member(member)

        # Add to unit_to_clusters (append to list)
        if member.unit_id not in self.unit_to_clusters:
            self.unit_to_clusters[member.unit_id] = []
            self.total_units += 1  # Only count unique units
        if cluster_id not in self.unit_to_clusters[member.unit_id]:
            self.unit_to_clusters[member.unit_id].append(cluster_id)

        self.updated_at = datetime.now()
        return True

    # === Statistics methods ===
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics information."""
        cluster_sizes = [len(c.members) for c in self.clusters.values()]
        # Count units with multiple cluster assignments
        multi_cluster_units = sum(1 for cids in self.unit_to_clusters.values() if len(cids) > 1)
        return {
            "total_clusters": len(self.clusters),
            "total_units": self.total_units,
            "indexed_units": len(self.unit_to_clusters),
            "multi_cluster_units": multi_cluster_units,
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
        }

    # === Serialization methods ===
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "clusters": {
                cid: cluster.to_dict()
                for cid, cluster in self.clusters.items()
            },
            "unit_to_clusters": self.unit_to_clusters,
            "metadata": {
                "conversation_id": self.conversation_id,
                "total_units": self.total_units,
                "total_clusters": len(self.clusters),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "llm_model": self.llm_model,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupEventClusterIndex":
        """Deserialize from dictionary."""
        clusters = {
            cid: GroupEventCluster.from_dict(cdata)
            for cid, cdata in data.get("clusters", {}).items()
        }

        unit_to_clusters = data.get("unit_to_clusters", {})
        metadata = data.get("metadata", {})
        created_at = metadata.get("created_at")
        updated_at = metadata.get("updated_at")

        return cls(
            clusters=clusters,
            unit_to_clusters=unit_to_clusters,
            conversation_id=metadata.get("conversation_id", ""),
            total_units=metadata.get("total_units", 0),
            created_at=datetime.fromisoformat(created_at) if created_at else datetime.now(),
            updated_at=datetime.fromisoformat(updated_at) if updated_at else datetime.now(),
            llm_model=metadata.get("llm_model", ""),
        )

    def save_to_file(self, file_path: Path) -> None:
        """Save to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "GroupEventClusterIndex":
        """Load from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
