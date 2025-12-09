"""Embedding-based retriever.

This retriever uses vector embeddings to find semantically similar documents.
"""

from typing import List, Optional
from ..core import Document


class EmbeddingRetriever:
    """Retriever using vector embeddings for semantic search."""

    def __init__(self, top_k: int = 50):
        """Initialize embedding retriever.

        Args:
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k

    async def search(
        self,
        query: str,
        memory_index,
        vectorize_service,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Search for documents using embedding similarity.

        Args:
            query: Search query
            memory_index: Memory index to search
            vectorize_service: Service for generating embeddings
            top_k: Override default top_k

        Returns:
            List of retrieved documents with scores
        """
        k = top_k or self.top_k

        # Generate query embedding
        query_embedding = await vectorize_service.get_embedding(query)

        # Search in memory index
        results = await memory_index.search(
            query_embedding=query_embedding,
            top_k=k
        )

        # Convert to Document format
        documents = []
        for result in results:
            # Handle both dict and object result types
            if isinstance(result, dict):
                result_id = result.get('id', '')
                result_content = result.get('narrative', '') or result.get('summary', '')
                result_score = result.get('score')
            else:
                # Handle object attributes (e.g., MemoryObj)
                result_id = getattr(result, 'id', '')
                result_content = getattr(result, 'narrative', '') or getattr(result, 'summary', '')
                result_score = getattr(result, 'score', None)

            doc = Document(
                id=str(result_id),
                content=result_content,
                score=result_score,
                metadata={
                    'retrieval_method': 'embedding',
                    'original_result': result
                }
            )
            documents.append(doc)

        return documents


async def embedding_search(
    query: str,
    memory_index,
    vectorize_service,
    top_k: int = 50
) -> List[Document]:
    """Convenience function for embedding search.

    Args:
        query: Search query
        memory_index: Memory index to search
        vectorize_service: Service for generating embeddings
        top_k: Number of documents to retrieve

    Returns:
        List of retrieved documents
    """
    retriever = EmbeddingRetriever(top_k=top_k)
    return await retriever.search(query, memory_index, vectorize_service)
