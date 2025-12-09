"""Multi-query expansion for retrieval.

This expander generates multiple reformulated queries using an LLM,
retrieves documents for each query, and fuses the results using RRF.
"""

from typing import List, Optional
import asyncio
from ..core import Document


class MultiQueryExpander:
    """Expander that generates multiple queries and fuses results."""

    def __init__(
        self,
        num_queries: int = 3,
        retrieval_top_k: int = 20,
        final_top_k: int = 50,
        rrf_k: int = 60
    ):
        """Initialize multi-query expander.

        Args:
            num_queries: Number of reformulated queries to generate
            retrieval_top_k: Documents to retrieve per query
            final_top_k: Final number of documents after fusion
            rrf_k: RRF fusion parameter
        """
        self.num_queries = num_queries
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.rrf_k = rrf_k

    async def expand(
        self,
        query: str,
        documents: List[Document],
        llm_provider,
        retriever_func,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """Expand results using multiple query variants.

        Args:
            query: Original search query
            documents: Initial retrieved documents
            llm_provider: LLM service for query generation
            retriever_func: Async function to call for retrieval
                Signature: async def retriever_func(query: str, top_k: int) -> List[Document]
            top_k: Override final_top_k

        Returns:
            Fused documents from multiple queries
        """
        from ..core.utils import multi_rrf_fusion

        k = top_k or self.final_top_k

        # Generate multiple query variants using LLM
        query_variants = await self._generate_query_variants(query, llm_provider)

        if not query_variants:
            # If query generation fails, return original documents
            return documents[:k]

        # Parallel retrieval for all queries
        retrieval_tasks = [
            retriever_func(q, self.retrieval_top_k)
            for q in query_variants
        ]

        all_results = await asyncio.gather(*retrieval_tasks)

        # Convert Document lists to (doc, score) tuples for RRF
        results_for_rrf = []
        for docs in all_results:
            doc_tuples = [
                (self._doc_to_memory_obj(doc), doc.get('score', 0.0))
                for doc in docs
            ]
            results_for_rrf.append(doc_tuples)

        # Add original documents to fusion
        original_tuples = [
            (self._doc_to_memory_obj(doc), doc.get('score', 0.0))
            for doc in documents
        ]
        results_for_rrf.append(original_tuples)

        # Fuse using RRF
        fused_results = multi_rrf_fusion(results_for_rrf, k=self.rrf_k)[:k]

        # Convert back to Document format
        fused_documents = []
        for mem, rrf_score in fused_results:
            doc = Document(
                id=str(getattr(mem, 'id', '')),
                content=getattr(mem, 'narrative', '') or getattr(mem, 'summary', ''),
                score=float(rrf_score),
                metadata={
                    'expansion_method': 'multi_query',
                    'rrf_score': float(rrf_score),
                    'num_queries': len(query_variants)
                }
            )
            fused_documents.append(doc)

        return fused_documents

    async def _generate_query_variants(
        self,
        query: str,
        llm_provider
    ) -> List[str]:
        """Generate multiple query variants using LLM.

        Args:
            query: Original query
            llm_provider: LLM service

        Returns:
            List of reformulated queries
        """
        prompt = f"""Given the following user query, generate {self.num_queries} different reformulations
that capture the same intent but use different wording or perspectives.

Original query: "{query}"

Requirements:
1. Each reformulation should be semantically similar but use different words
2. Cover different aspects or angles of the question
3. Keep the queries concise and focused
4. Output ONLY the reformulated queries, one per line, without numbering or extra text

Reformulated queries:"""

        try:
            response = await llm_provider.generate(prompt)

            # Parse response - expect one query per line
            lines = [line.strip() for line in response.strip().split('\n')]
            queries = [line for line in lines if line and not line.startswith('#')]

            # Include original query
            all_queries = [query] + queries[:self.num_queries]

            return all_queries

        except Exception as e:
            # If LLM fails, return only original query
            return [query]

    def _doc_to_memory_obj(self, doc: Document):
        """Convert Document back to memory object for RRF."""
        if 'original_memory' in doc.get('metadata', {}):
            return doc['metadata']['original_memory']

        class MemoryObj:
            def __init__(self, doc_dict):
                self.id = doc_dict.get('id', '')
                self.narrative = doc_dict.get('content', '')
                self.summary = doc_dict.get('content', '')

        return MemoryObj(doc)


async def multi_query_expand(
    query: str,
    documents: List[Document],
    llm_provider,
    retriever_func,
    num_queries: int = 3,
    retrieval_top_k: int = 20,
    final_top_k: int = 50
) -> List[Document]:
    """Convenience function for multi-query expansion.

    Args:
        query: Original query
        documents: Initial documents
        llm_provider: LLM service
        retriever_func: Async retrieval function
        num_queries: Number of query variants
        retrieval_top_k: Docs per query
        final_top_k: Final docs after fusion

    Returns:
        Fused documents
    """
    expander = MultiQueryExpander(
        num_queries=num_queries,
        retrieval_top_k=retrieval_top_k,
        final_top_k=final_top_k
    )
    return await expander.expand(query, documents, llm_provider, retriever_func, final_top_k)
