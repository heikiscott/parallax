"""DeepInfra reranker component.

Uses DeepInfra's reranking API to rerank retrieved documents.
"""

from typing import List, Optional
from ..core import Document


class DeepInfraReranker:
    """Reranker using DeepInfra API."""

    def __init__(self, top_k: int = 20):
        """Initialize DeepInfra reranker.

        Args:
            top_k: Number of documents to return after reranking
        """
        self.top_k = top_k

    async def rerank(
        self,
        query: str,
        documents: List[Document],
        rerank_service,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Rerank documents using DeepInfra API.

        Args:
            query: Search query
            documents: Documents to rerank
            rerank_service: DeepInfra rerank service
            top_k: Override default top_k

        Returns:
            Reranked documents with updated scores
        """
        k = top_k or self.top_k

        if not documents:
            return []

        # Extract text content from documents
        texts = [doc.get('content', '') for doc in documents]

        # Call rerank service
        reranked_results = await rerank_service.rerank(
            query=query,
            documents=texts,
            top_k=k
        )

        # Map reranked results back to original documents
        reranked_documents = []
        for result in reranked_results:
            # result contains: index, score, text
            orig_idx = result.get('index', 0)
            if orig_idx < len(documents):
                doc = documents[orig_idx].copy()
                doc['score'] = result.get('score')
                doc['metadata'] = doc.get('metadata', {})
                doc['metadata']['rerank_score'] = result.get('score')
                doc['metadata']['rerank_method'] = 'deep_infra'
                reranked_documents.append(doc)

        return reranked_documents


async def deep_infra_rerank(
    query: str,
    documents: List[Document],
    rerank_service,
    top_k: int = 20
) -> List[Document]:
    """Convenience function for DeepInfra reranking.

    Args:
        query: Search query
        documents: Documents to rerank
        rerank_service: DeepInfra rerank service
        top_k: Number of documents to return

    Returns:
        Reranked documents
    """
    reranker = DeepInfraReranker(top_k=top_k)
    return await reranker.rerank(query, documents, rerank_service)
