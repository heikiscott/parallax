"""Storage abstractions for Group Event Cluster system.

This module provides storage backends for persisting cluster indices.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import json
import logging

from .schema import GroupEventClusterIndex

logger = logging.getLogger(__name__)


class ClusterStorage(ABC):
    """Abstract base class for cluster storage backends."""

    @abstractmethod
    async def save_index(
        self,
        conversation_id: str,
        index: GroupEventClusterIndex
    ) -> bool:
        """
        Save cluster index for a conversation.

        Args:
            conversation_id: Conversation identifier
            index: GroupEventClusterIndex to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def load_index(
        self,
        conversation_id: str
    ) -> Optional[GroupEventClusterIndex]:
        """
        Load cluster index for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            GroupEventClusterIndex if found, None otherwise
        """
        pass

    @abstractmethod
    async def exists(self, conversation_id: str) -> bool:
        """
        Check if cluster index exists for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, conversation_id: str) -> bool:
        """
        Delete cluster index for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def list_conversations(self) -> list[str]:
        """
        List all conversation IDs with stored indices.

        Returns:
            List of conversation IDs
        """
        pass


class JsonClusterStorage(ClusterStorage):
    """JSON file-based cluster storage.

    Stores each conversation's cluster index as a separate JSON file.
    File naming: {output_dir}/conv_{conversation_id}.json
    """

    def __init__(self, output_dir: Path):
        """
        Initialize JSON storage.

        Args:
            output_dir: Directory for storing JSON files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, conversation_id: str) -> Path:
        """Get file path for a conversation's cluster index.

        File naming convention: {conversation_id}.json
        If conversation_id already contains "conv_", use it as-is.
        Otherwise, the file will be named directly with conversation_id.

        Examples:
            - conversation_id="conv_0" -> conv_0.json
            - conversation_id="0" -> 0.json
        """
        # Sanitize conversation_id for use in filename
        safe_id = conversation_id.replace("/", "_").replace("\\", "_")
        return self.output_dir / f"{safe_id}.json"

    async def save_index(
        self,
        conversation_id: str,
        index: GroupEventClusterIndex
    ) -> bool:
        """Save cluster index to JSON file."""
        try:
            file_path = self._get_file_path(conversation_id)

            # Ensure conversation_id is set
            if not index.conversation_id:
                index.conversation_id = conversation_id

            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(index.to_dict(), f, ensure_ascii=False, indent=2)

            logger.info(f"Saved cluster index for {conversation_id} to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save cluster index for {conversation_id}: {e}")
            return False

    async def load_index(
        self,
        conversation_id: str
    ) -> Optional[GroupEventClusterIndex]:
        """Load cluster index from JSON file."""
        try:
            file_path = self._get_file_path(conversation_id)

            if not file_path.exists():
                logger.debug(f"No cluster index found for {conversation_id}")
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            index = GroupEventClusterIndex.from_dict(data)
            logger.info(f"Loaded cluster index for {conversation_id} from {file_path}")
            return index

        except Exception as e:
            logger.error(f"Failed to load cluster index for {conversation_id}: {e}")
            return None

    async def exists(self, conversation_id: str) -> bool:
        """Check if cluster index file exists."""
        file_path = self._get_file_path(conversation_id)
        return file_path.exists()

    async def delete(self, conversation_id: str) -> bool:
        """Delete cluster index file."""
        try:
            file_path = self._get_file_path(conversation_id)

            if file_path.exists():
                file_path.unlink()
                logger.info(f"Deleted cluster index for {conversation_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete cluster index for {conversation_id}: {e}")
            return False

    async def list_conversations(self) -> list[str]:
        """List all conversation IDs with stored indices."""
        conversations = []
        try:
            for file_path in self.output_dir.glob("*.json"):
                # The filename is the conversation_id
                # Skip summary files or other non-index files
                if file_path.stem.startswith("clustering_"):
                    continue
                conversations.append(file_path.stem)
        except Exception as e:
            logger.error(f"Failed to list conversations: {e}")

        return conversations


class InMemoryClusterStorage(ClusterStorage):
    """In-memory cluster storage for testing and development."""

    def __init__(self):
        """Initialize in-memory storage."""
        self._indices: dict[str, GroupEventClusterIndex] = {}

    async def save_index(
        self,
        conversation_id: str,
        index: GroupEventClusterIndex
    ) -> bool:
        """Save cluster index to memory."""
        try:
            # Ensure conversation_id is set
            if not index.conversation_id:
                index.conversation_id = conversation_id

            self._indices[conversation_id] = index
            return True

        except Exception as e:
            logger.error(f"Failed to save cluster index: {e}")
            return False

    async def load_index(
        self,
        conversation_id: str
    ) -> Optional[GroupEventClusterIndex]:
        """Load cluster index from memory."""
        return self._indices.get(conversation_id)

    async def exists(self, conversation_id: str) -> bool:
        """Check if cluster index exists in memory."""
        return conversation_id in self._indices

    async def delete(self, conversation_id: str) -> bool:
        """Delete cluster index from memory."""
        if conversation_id in self._indices:
            del self._indices[conversation_id]
        return True

    async def list_conversations(self) -> list[str]:
        """List all conversation IDs."""
        return list(self._indices.keys())

    def clear_all(self) -> None:
        """Clear all stored indices (useful for testing)."""
        self._indices.clear()
