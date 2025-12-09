"""Core clustering logic for Group Event Cluster system.

This module contains the GroupEventClusterer class which performs
LLM-driven clustering of MemUnits into event clusters.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .types import GroupEventClusterConfig
from .schema import ClusterMember, GroupEventCluster, GroupEventClusterIndex
from .utils import (
    CLUSTER_ASSIGNMENT_PROMPT,
    CLUSTER_SUMMARY_PROMPT,
    UNIT_SUMMARY_PROMPT,
    CLUSTER_TOPIC_PROMPT,
    format_clusters_for_prompt,
    format_members_for_summary,
    parse_cluster_assignment_response,
)

logger = logging.getLogger(__name__)


class GroupEventClusterer:
    """
    Group Event Clusterer - LLM-driven clustering of MemUnits.

    Responsibilities:
    1. Receive MemUnits and call LLM to determine cluster assignment
    2. Manage GroupEventClusterIndex
    3. Generate summaries and topics
    4. Maintain time-sorted member lists
    """

    def __init__(
        self,
        config: GroupEventClusterConfig,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize the clusterer.

        Args:
            config: Clustering configuration
            llm_provider: LLM provider instance. If None, will create one from config.
        """
        self.config = config
        self.llm_provider = llm_provider or self._create_llm_provider()
        self.index: Optional[GroupEventClusterIndex] = None
        self._next_cluster_idx = 0

    def _create_llm_provider(self) -> Any:
        """Create LLM provider from configuration."""
        try:
            from providers.llm.llm_provider import LLMProvider

            return LLMProvider(
                provider_type=self.config.llm_provider,
                model=self.config.llm_model,
                api_key=self.config.llm_api_key,
                base_url=self.config.llm_base_url,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
        except ImportError:
            logger.warning("Could not import LLMProvider, using None")
            return None

    async def cluster_memunits(
        self,
        memunit_list: List[Dict[str, Any]],
        conversation_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GroupEventClusterIndex:
        """
        Cluster all MemUnits.

        Args:
            memunit_list: List of MemUnit dictionaries (will be sorted by time)
            conversation_id: Conversation ID
            progress_callback: Progress callback function (current, total, cluster_id)

        Returns:
            GroupEventClusterIndex with all clusters
        """
        # Sort by timestamp
        sorted_memunits = sorted(
            memunit_list,
            key=lambda x: self._parse_timestamp(x.get("timestamp", 0))
        )

        # Initialize index
        self.index = GroupEventClusterIndex(
            clusters={},
            unit_to_clusters={},
            conversation_id=conversation_id,
            total_units=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            llm_model=self.config.llm_model,
        )
        self._next_cluster_idx = 0

        logger.info(f"Starting clustering of {len(sorted_memunits)} MemUnits")

        # Process each MemUnit
        for i, memunit in enumerate(sorted_memunits):
            try:
                cluster_ids = await self._cluster_single_memunit(memunit)

                if progress_callback:
                    # Report first cluster_id for progress (backward compatible)
                    progress_callback(i + 1, len(sorted_memunits), cluster_ids[0] if cluster_ids else "")

                logger.debug(f"Processed {i+1}/{len(sorted_memunits)}: {memunit.get('unit_id')} -> {cluster_ids}")

            except Exception as e:
                logger.error(f"Error processing MemUnit {memunit.get('unit_id')}: {e}")
                # Continue with next MemUnit instead of failing completely
                continue

        logger.info(f"Clustering complete: {len(self.index.clusters)} clusters, {self.index.total_units} units")

        return self.index

    async def _cluster_single_memunit(self, memunit: Dict[str, Any]) -> List[str]:
        """
        Process a single MemUnit and assign it to one or more clusters.

        A MemUnit can belong to multiple clusters if it discusses multiple topics.

        Returns:
            List of cluster_ids the MemUnit was assigned to
        """
        unit_id = memunit.get("unit_id", "")
        narrative = memunit.get("narrative", "")
        timestamp = self._parse_timestamp(memunit.get("timestamp"))

        # Step 1: Generate unit summary
        unit_summary = await self._generate_unit_summary(narrative)

        # Step 2: If this is the first MemUnit, create a new cluster
        if not self.index.clusters:
            cluster_id = await self._create_new_cluster(
                unit_id=unit_id,
                unit_summary=unit_summary,
                narrative=narrative,
                timestamp=timestamp,
            )
            return [cluster_id]

        # Step 3: Use LLM to decide cluster assignment(s)
        decision = await self._llm_decide_cluster(
            unit_id=unit_id,
            unit_summary=unit_summary,
            narrative=narrative,
            timestamp=timestamp,
        )

        # Step 4: Handle multi-assignment response
        assigned_cluster_ids: List[str] = []
        assignments = decision.get("assignments", [])

        for assignment in assignments:
            assign_type = assignment.get("type", "")

            if assign_type == "NEW":
                cluster_id = await self._create_new_cluster(
                    unit_id=unit_id,
                    unit_summary=unit_summary,
                    narrative=narrative,
                    timestamp=timestamp,
                    topic=assignment.get("new_topic"),
                )
                assigned_cluster_ids.append(cluster_id)

            elif assign_type == "EXISTING":
                cluster_id = assignment.get("cluster_id", "")
                if cluster_id and cluster_id in self.index.clusters:
                    self._add_to_existing_cluster(
                        cluster_id=cluster_id,
                        unit_id=unit_id,
                        unit_summary=unit_summary,
                        timestamp=timestamp,
                    )
                    assigned_cluster_ids.append(cluster_id)
                else:
                    logger.warning(f"Cluster {cluster_id} not found, skipping")

        # Fallback: create new cluster if no valid assignments
        if not assigned_cluster_ids:
            logger.warning(f"No valid assignments for {unit_id}, creating new cluster")
            cluster_id = await self._create_new_cluster(
                unit_id=unit_id,
                unit_summary=unit_summary,
                narrative=narrative,
                timestamp=timestamp,
            )
            assigned_cluster_ids.append(cluster_id)

        return assigned_cluster_ids

    async def _generate_unit_summary(self, narrative: str) -> str:
        """Generate a brief summary for a MemUnit."""
        if not self.llm_provider:
            # Fallback: truncate narrative
            return narrative[:100] + "..." if len(narrative) > 100 else narrative

        try:
            prompt = UNIT_SUMMARY_PROMPT.format(narrative=narrative)
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=150,
            )
            return response.strip()
        except Exception as e:
            logger.warning(f"Failed to generate unit summary: {e}")
            return narrative[:100] + "..." if len(narrative) > 100 else narrative

    async def _llm_decide_cluster(
        self,
        unit_id: str,
        unit_summary: str,
        narrative: str,
        timestamp: datetime,
    ) -> Dict[str, Any]:
        """Use LLM to decide which cluster a MemUnit belongs to."""
        if not self.llm_provider:
            # Fallback: always create new cluster (use new multi-assignment format)
            return {
                "assignments": [{"type": "NEW", "new_topic": f"Event {self._next_cluster_idx + 1}"}],
                "reason": "No LLM provider available",
            }

        try:
            # Format existing clusters for prompt
            clusters_list = list(self.index.clusters.values())
            existing_clusters_text = format_clusters_for_prompt(
                clusters=clusters_list,
                max_clusters=self.config.max_clusters_in_prompt,
                max_members_per_cluster=self.config.max_members_per_cluster_in_prompt,
            )

            prompt = CLUSTER_ASSIGNMENT_PROMPT.format(
                existing_clusters=existing_clusters_text,
                unit_id=unit_id,
                timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                unit_summary=unit_summary,
                narrative=narrative[:500],  # Limit narrative length
            )

            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=200,
            )

            decision = parse_cluster_assignment_response(response)
            logger.debug(f"LLM decision for {unit_id}: {decision}")

            return decision

        except Exception as e:
            logger.warning(f"LLM cluster decision failed: {e}")
            return {
                "assignments": [{"type": "NEW", "new_topic": f"Event {self._next_cluster_idx + 1}"}],
                "reason": f"LLM error: {e}",
            }

    async def _create_new_cluster(
        self,
        unit_id: str,
        unit_summary: str,
        narrative: str,
        timestamp: datetime,
        topic: Optional[str] = None,
    ) -> str:
        """Create a new cluster and add the MemUnit as first member."""
        # Generate cluster_id
        self._next_cluster_idx += 1
        cluster_id = f"gec_{self._next_cluster_idx:03d}"

        # Generate topic if not provided
        if not topic:
            topic = await self._generate_topic(narrative)

        # Create member
        member = ClusterMember(
            unit_id=unit_id,
            timestamp=timestamp,
            summary=unit_summary,
        )

        # Create cluster
        cluster = GroupEventCluster(
            cluster_id=cluster_id,
            topic=topic,
            summary=unit_summary,  # Initial summary is just the first member's summary
            members=[member],
            first_timestamp=timestamp,
            last_timestamp=timestamp,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        # Add to index
        self.index.clusters[cluster_id] = cluster

        # Update unit_to_clusters mapping (one-to-many)
        if unit_id not in self.index.unit_to_clusters:
            self.index.unit_to_clusters[unit_id] = []
            self.index.total_units += 1
        self.index.unit_to_clusters[unit_id].append(cluster_id)
        self.index.updated_at = datetime.now()

        logger.info(f"Created new cluster {cluster_id}: {topic}")

        return cluster_id

    def _add_to_existing_cluster(
        self,
        cluster_id: str,
        unit_id: str,
        unit_summary: str,
        timestamp: datetime,
    ) -> None:
        """Add a MemUnit to an existing cluster."""
        cluster = self.index.clusters.get(cluster_id)
        if not cluster:
            return

        # Create member
        member = ClusterMember(
            unit_id=unit_id,
            timestamp=timestamp,
            summary=unit_summary,
        )

        # Add to cluster (maintains time ordering)
        cluster.add_member(member)

        # Update unit_to_clusters mapping (one-to-many)
        if unit_id not in self.index.unit_to_clusters:
            self.index.unit_to_clusters[unit_id] = []
            self.index.total_units += 1
        if cluster_id not in self.index.unit_to_clusters[unit_id]:
            self.index.unit_to_clusters[unit_id].append(cluster_id)
        self.index.updated_at = datetime.now()

        # Check if summary update is needed
        if len(cluster.members) % self.config.summary_update_threshold == 0:
            asyncio.create_task(self._update_cluster_summary(cluster))

        logger.debug(f"Added {unit_id} to cluster {cluster_id} (now {len(cluster.members)} members)")

    async def _generate_topic(self, description: str) -> str:
        """Generate a topic name for a new cluster."""
        if not self.llm_provider:
            return f"Event {self._next_cluster_idx}"

        try:
            prompt = CLUSTER_TOPIC_PROMPT.format(description=description[:300])
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=50,
            )
            topic = response.strip().strip('"\'')
            # Ensure reasonable length
            if len(topic) > 50:
                topic = topic[:47] + "..."
            return topic
        except Exception as e:
            logger.warning(f"Failed to generate topic: {e}")
            return f"Event {self._next_cluster_idx}"

    async def _update_cluster_summary(self, cluster: GroupEventCluster) -> None:
        """Update a cluster's summary based on all its members."""
        if not self.llm_provider:
            return

        try:
            members_info = format_members_for_summary(cluster.members)
            prompt = CLUSTER_SUMMARY_PROMPT.format(
                topic=cluster.topic,
                members_info=members_info,
            )

            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=400,
            )

            cluster.summary = response.strip()
            cluster.updated_at = datetime.now()

            logger.debug(f"Updated summary for cluster {cluster.cluster_id}")

        except Exception as e:
            logger.warning(f"Failed to update cluster summary: {e}")

    def _parse_timestamp(self, timestamp_value: Any) -> datetime:
        """Parse various timestamp formats to datetime."""
        if isinstance(timestamp_value, datetime):
            return timestamp_value

        if isinstance(timestamp_value, str):
            # Try various formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_value[:26], fmt[:len(timestamp_value)])
                except ValueError:
                    continue

            # Try ISO format
            try:
                return datetime.fromisoformat(timestamp_value.replace("Z", "+00:00"))
            except ValueError:
                pass

        if isinstance(timestamp_value, (int, float)):
            # Assume Unix timestamp
            try:
                return datetime.fromtimestamp(timestamp_value)
            except (ValueError, OSError):
                pass

        # Fallback to now
        logger.warning(f"Could not parse timestamp: {timestamp_value}, using now()")
        return datetime.now()

    def get_index(self) -> Optional[GroupEventClusterIndex]:
        """Get the current cluster index."""
        return self.index

    def get_stats(self) -> Dict[str, Any]:
        """Get clustering statistics."""
        if not self.index:
            return {"error": "No index available"}

        return {
            **self.index.get_stats(),
            "config": self.config.to_dict(),
        }
