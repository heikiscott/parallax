"""Hybrid retriever combining embedding and BM25 search.

This retriever performs both embedding-based semantic search and BM25 keyword search,
then fuses the results using Reciprocal Rank Fusion (RRF).
"""

from typing import List, Optional
from ..core import Document


class HybridRetriever:
    """Hybrid retriever using Embedding + BM25 + RRF fusion."""

    def __init__(
        self,
        emb_top_k: int = 50,
        bm25_top_k: int = 50,
        final_top_k: int = 20,
        rrf_k: int = 60
    ):
        """Initialize hybrid retriever.

        Args:
            emb_top_k: Number of documents to retrieve via embedding search
            bm25_top_k: Number of documents to retrieve via BM25 search
            final_top_k: Number of documents to return after fusion
            rrf_k: RRF fusion parameter (default 60)
        """
        self.emb_top_k = emb_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        memory_index,
        vectorize_service,
        bm25_index=None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Perform hybrid search (Embedding + BM25 + RRF).

        Args:
            query: Search query
            memory_index: Memory index to search
            vectorize_service: Service for generating embeddings
            bm25_index: Pre-built BM25 index (optional)
            top_k: Override default final_top_k

        Returns:
            List of fused documents with RRF scores
        """
        from .embedding_retriever import embedding_search
        from .bm25_retriever import bm25_search
        from ..core.utils import reciprocal_rank_fusion

        k = top_k or self.final_top_k

        # 1. Embedding search
        emb_documents = await embedding_search(
            query=query,
            memory_index=memory_index,
            vectorize_service=vectorize_service,
            top_k=self.emb_top_k
        )

        # 2. BM25 search
        bm25_documents = await bm25_search(
            query=query,
            memory_index=memory_index,
            bm25_index=bm25_index,
            top_k=self.bm25_top_k
        )

        # 3. Convert to (doc, score) tuples for RRF
        emb_results = [
            (self._doc_to_memory_obj(doc), doc.get('score', 0.0))
            for doc in emb_documents
        ]

        bm25_results = [
            (self._doc_to_memory_obj(doc), doc.get('score', 0.0))
            for doc in bm25_documents
        ]

        # 4. Apply Reciprocal Rank Fusion
        if not emb_results and not bm25_results:
            return []

        if not emb_results:
            fused_results = bm25_results[:k]
        elif not bm25_results:
            fused_results = emb_results[:k]
        else:
            fused_results = reciprocal_rank_fusion(
                emb_results,
                bm25_results,
                k=self.rrf_k
            )[:k]

        # 5. Convert back to Document format
        documents = []
        for mem, rrf_score in fused_results:
            doc = Document(
                id=str(getattr(mem, 'id', '')),
                content=getattr(mem, 'narrative', '') or getattr(mem, 'summary', ''),
                score=float(rrf_score),
                metadata={
                    'retrieval_method': 'hybrid_rrf',
                    'rrf_score': float(rrf_score),
                    'original_memory': mem
                }
            )
            documents.append(doc)

        return documents

    def _doc_to_memory_obj(self, doc: Document):
        """Convert Document back to memory object for RRF."""
        # Check if original_memory exists in metadata
        if 'original_memory' in doc.get('metadata', {}):
            return doc['metadata']['original_memory']

        # Otherwise create a simple object
        class MemoryObj:
            def __init__(self, doc_dict):
                self.id = doc_dict.get('id', '')
                self.narrative = doc_dict.get('content', '')
                self.summary = doc_dict.get('content', '')

        return MemoryObj(doc)


async def hybrid_search(
    query: str,
    memory_index,
    vectorize_service,
    bm25_index=None,
    emb_top_k: int = 50,
    bm25_top_k: int = 50,
    final_top_k: int = 20
) -> List[Document]:
    """Convenience function for hybrid search.

    Args:
        query: Search query
        memory_index: Memory index to search
        vectorize_service: Service for generating embeddings
        bm25_index: Pre-built BM25 index (optional)
        emb_top_k: Number of embedding results
        bm25_top_k: Number of BM25 results
        final_top_k: Number of final fused results

    Returns:
        List of fused documents
    """
    retriever = HybridRetriever(
        emb_top_k=emb_top_k,
        bm25_top_k=bm25_top_k,
        final_top_k=final_top_k
    )
    return await retriever.search(query, memory_index, vectorize_service, bm25_index)
