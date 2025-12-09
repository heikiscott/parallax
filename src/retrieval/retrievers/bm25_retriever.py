"""BM25-based retriever.

This retriever uses BM25 (Best Matching 25) algorithm for keyword-based search.
Supports both English and Chinese text.
"""

from typing import List, Optional
from ..core import Document


class BM25Retriever:
    """Retriever using BM25 algorithm for keyword search."""

    def __init__(self, top_k: int = 50):
        """Initialize BM25 retriever.

        Args:
            top_k: Number of documents to retrieve
        """
        self.top_k = top_k

    async def search(
        self,
        query: str,
        memory_index,
        bm25_index=None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Search for documents using BM25 scoring.

        Args:
            query: Search query
            memory_index: Memory index to search
            bm25_index: Pre-built BM25 index (optional, will build if None)
            top_k: Override default top_k

        Returns:
            List of retrieved documents with BM25 scores
        """
        from ..core.utils import build_bm25_index, search_with_bm25

        k = top_k or self.top_k

        # Get all candidates from memory index
        candidates = await memory_index.get_all_memories()

        if not candidates:
            return []

        # Build BM25 index if not provided
        if bm25_index is None:
            bm25, tokenized_docs, stemmer, stop_words = build_bm25_index(candidates)
        else:
            bm25, tokenized_docs, stemmer, stop_words = bm25_index

        if bm25 is None:
            # BM25 dependencies not available
            return []

        # Perform BM25 search
        results = await search_with_bm25(
            query=query,
            bm25=bm25,
            candidates=candidates,
            stemmer=stemmer,
            stop_words=stop_words,
            top_k=k
        )

        # Convert to Document format
        documents = []
        for mem, score in results:
            doc = Document(
                id=str(getattr(mem, 'id', '')),
                content=getattr(mem, 'narrative', '') or getattr(mem, 'summary', ''),
                score=float(score),
                metadata={
                    'retrieval_method': 'bm25',
                    'original_memory': mem
                }
            )
            documents.append(doc)

        return documents


async def bm25_search(
    query: str,
    memory_index,
    bm25_index=None,
    top_k: int = 50
) -> List[Document]:
    """Convenience function for BM25 search.

    Args:
        query: Search query
        memory_index: Memory index to search
        bm25_index: Pre-built BM25 index (optional)
        top_k: Number of documents to retrieve

    Returns:
        List of retrieved documents
    """
    retriever = BM25Retriever(top_k=top_k)
    return await retriever.search(query, memory_index, bm25_index)
