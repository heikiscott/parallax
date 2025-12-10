"""Cluster-based result expansion.

This expander finds documents from the same cluster as retrieved results
and inserts them to improve coverage.
"""

from typing import List, Optional
from ..core import Document


class ClusterExpander:
    """Expander that adds documents from the same cluster."""

    def __init__(
        self,
        expansion_strategy: str = "insert_after_hit",
        max_expansion_per_hit: int = 3,
        min_cluster_score: float = 0.5
    ):
        """Initialize cluster expander.

        Args:
            expansion_strategy: How to insert cluster documents
                - "insert_after_hit": Insert after each hit from same cluster
                - "append_end": Append all cluster documents at the end
            max_expansion_per_hit: Maximum documents to add per cluster hit
            min_cluster_score: Minimum similarity score for cluster members
        """
        self.expansion_strategy = expansion_strategy
        self.max_expansion_per_hit = max_expansion_per_hit
        self.min_cluster_score = min_cluster_score

    async def expand(
        self,
        query: str,
        documents: List[Document],
        cluster_index,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Expand results by adding documents from same clusters.

        Args:
            query: Original search query
            documents: Retrieved documents
            cluster_index: Cluster index for finding related documents
            top_k: Maximum total documents to return (optional)

        Returns:
            Expanded list of documents
        """
        if not cluster_index or not documents:
            return documents

        expanded_docs = []
        seen_ids = set()
        cluster_docs_to_append = []  # For append_end strategy

        for doc in documents:
            # Add the original document
            doc_id = doc.get('id', '')
            if doc_id not in seen_ids:
                expanded_docs.append(doc)
                seen_ids.add(doc_id)

            # Find cluster members for this document
            try:
                cluster_members = await cluster_index.get_cluster_members(doc_id)

                if cluster_members:
                    if self.expansion_strategy == "insert_after_hit":
                        # Insert related documents after this hit
                        added_count = 0
                        for member in cluster_members:
                            member_id = str(getattr(member, 'id', ''))

                            # Skip if already in results or low score
                            if member_id in seen_ids:
                                continue

                            member_score = getattr(member, 'cluster_score', 0.0)
                            if member_score < self.min_cluster_score:
                                continue

                            # Create expanded document
                            expanded_doc = Document(
                                id=member_id,
                                content=getattr(member, 'narrative', '') or getattr(member, 'summary', ''),
                                score=member_score,
                                metadata={
                                    'expansion_method': 'cluster',
                                    'source_doc_id': doc_id,
                                    'cluster_score': member_score
                                }
                            )

                            expanded_docs.append(expanded_doc)
                            seen_ids.add(member_id)
                            added_count += 1

                            if added_count >= self.max_expansion_per_hit:
                                break

                    elif self.expansion_strategy == "append_end":
                        # Collect cluster documents to append at end
                        added_count = 0
                        for member in cluster_members:
                            member_id = str(getattr(member, 'id', ''))

                            # Skip if already in results or low score
                            if member_id in seen_ids:
                                continue

                            member_score = getattr(member, 'cluster_score', 0.0)
                            if member_score < self.min_cluster_score:
                                continue

                            # Create expanded document
                            expanded_doc = Document(
                                id=member_id,
                                content=getattr(member, 'narrative', '') or getattr(member, 'summary', ''),
                                score=member_score,
                                metadata={
                                    'expansion_method': 'cluster',
                                    'source_doc_id': doc_id,
                                    'cluster_score': member_score
                                }
                            )

                            cluster_docs_to_append.append(expanded_doc)
                            seen_ids.add(member_id)
                            added_count += 1

                            if added_count >= self.max_expansion_per_hit:
                                break

            except Exception as e:
                # Cluster index might not have this document, skip
                continue

        # For append_end strategy, add all collected cluster docs at the end
        if self.expansion_strategy == "append_end":
            expanded_docs.extend(cluster_docs_to_append)

        # Apply top_k limit if specified
        if top_k:
            expanded_docs = expanded_docs[:top_k]

        return expanded_docs


async def cluster_expand(
    query: str,
    documents: List[Document],
    cluster_index,
    expansion_strategy: str = "insert_after_hit",
    max_expansion_per_hit: int = 3,
    top_k: Optional[int] = None
) -> List[Document]:
    """Convenience function for cluster expansion.

    Args:
        query: Search query
        documents: Retrieved documents
        cluster_index: Cluster index
        expansion_strategy: How to insert cluster documents
        max_expansion_per_hit: Max documents to add per hit
        top_k: Maximum total documents

    Returns:
        Expanded documents
    """
    expander = ClusterExpander(
        expansion_strategy=expansion_strategy,
        max_expansion_per_hit=max_expansion_per_hit
    )
    return await expander.expand(query, documents, cluster_index, top_k)
