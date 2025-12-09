"""LLM utility functions for retrieval pipelines.

Provides LLM-guided retrieval capabilities:
1. Sufficiency Check: Determine if retrieval results are sufficient
2. Query Refinement: Generate improved queries
3. Multi-Query Generation: Generate multiple complementary queries
4. Document Formatting: Format documents for LLM input
"""

import json
import asyncio
from typing import List, Tuple, Optional

# Import prompts
from prompts.memory.en.eval.search.sufficiency_check_prompts import SUFFICIENCY_CHECK_PROMPT
from prompts.memory.en.eval.search.refined_query_prompts import REFINED_QUERY_PROMPT
from prompts.memory.en.eval.search.multi_query_prompts import MULTI_QUERY_GENERATION_PROMPT


def format_documents_for_llm(
    results: List[Tuple[dict, float]],
    max_docs: int = 10,
    use_episode: bool = True
) -> str:
    """Format retrieval results for LLM consumption.

    Args:
        results: List of (document, score) tuples
        max_docs: Maximum number of documents to include
        use_episode: True=use Episode Memory format, False=use Event Log format

    Returns:
        Formatted document string
    """
    formatted_docs = []

    for i, (doc, score) in enumerate(results[:max_docs], start=1):
        subject = doc.get("subject", "N/A")

        if use_episode:
            # Episode Memory format (full narrative)
            narrative = doc.get("narrative", "N/A")

            # Limit narrative length to avoid prompt overflow
            if len(narrative) > 500:
                narrative = narrative[:500] + "..."

            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {narrative}\n"
            )
            formatted_docs.append(doc_text)
        else:
            # Event Log format (atomic facts)
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                event_log = doc["event_log"]
                time_str = event_log.get("time", "N/A")
                atomic_facts = event_log.get("atomic_fact", [])

                if isinstance(atomic_facts, list) and atomic_facts:
                    facts_text = "\n     ".join(atomic_facts[:5])
                    if len(atomic_facts) > 5:
                        facts_text += f"\n     ... and {len(atomic_facts) - 5} more facts"

                    doc_text = (
                        f"Document {i}:\n"
                        f"  Title: {subject}\n"
                        f"  Time: {time_str}\n"
                        f"  Facts:\n"
                        f"     {facts_text}\n"
                    )
                    formatted_docs.append(doc_text)
                    continue

            # Fallback to narrative if no event_log
            narrative = doc.get("narrative", "N/A")
            if len(narrative) > 500:
                narrative = narrative[:500] + "..."

            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {narrative}\n"
            )
            formatted_docs.append(doc_text)

    return "\n".join(formatted_docs)


def parse_json_response(response: str) -> dict:
    """Parse LLM JSON response with robust error handling.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed JSON dictionary
    """
    try:
        # Extract JSON (LLM may add extra text before/after)
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)

        # Validate required fields
        if "is_sufficient" not in result:
            raise ValueError("Missing 'is_sufficient' field")

        # Set defaults
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("missing_information", [])

        return result

    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse LLM response: {e}")
        print(f"  Raw response: {response[:200]}...")

        # Conservative fallback: assume sufficient
        return {
            "is_sufficient": True,
            "reasoning": f"Failed to parse: {str(e)}",
            "missing_information": []
        }


def parse_refined_query(response: str, original_query: str) -> str:
    """Parse refined query from LLM response.

    Args:
        response: LLM response
        original_query: Original query for fallback

    Returns:
        Refined query string
    """
    refined = response.strip()

    # Remove common prefixes
    prefixes = ["Refined Query:", "Output:", "Answer:", "Query:"]
    for prefix in prefixes:
        if refined.startswith(prefix):
            refined = refined[len(prefix):].strip()

    # Validate length
    if len(refined) < 5 or len(refined) > 300:
        print(f"  ⚠️  Invalid refined query length ({len(refined)}), using original")
        return original_query

    # Avoid identical query
    if refined.lower() == original_query.lower():
        print(f"  ⚠️  Refined query identical to original, using original")
        return original_query

    return refined


def parse_multi_query_response(response: str, original_query: str) -> Tuple[List[str], str]:
    """Parse multi-query generation JSON response.

    Args:
        response: Raw LLM response string
        original_query: Original query for fallback

    Returns:
        (queries_list, reasoning)
    """
    try:
        # Extract JSON
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)

        # Validate required fields
        if "queries" not in result or not isinstance(result["queries"], list):
            raise ValueError("Missing or invalid 'queries' field")

        queries = result["queries"]
        reasoning = result.get("reasoning", "No reasoning provided")

        # Filter and validate queries
        valid_queries = []
        for q in queries:
            if isinstance(q, str) and 5 <= len(q) <= 300:
                if q.lower().strip() != original_query.lower().strip():
                    valid_queries.append(q.strip())

        if not valid_queries:
            print(f"  ⚠️  No valid queries generated, using original")
            return [original_query], "Fallback: used original query"

        # Limit to 3 queries
        valid_queries = valid_queries[:3]

        print(f"  ✅ Generated {len(valid_queries)} valid queries")
        return valid_queries, reasoning

    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse multi-query response: {e}")
        print(f"  Raw response: {response[:200]}...")

        return [original_query], f"Parse error: {str(e)}"


async def check_sufficiency(
    query: str,
    results: List[Tuple[dict, float]],
    llm_provider,
    llm_config: dict,
    max_docs: int = 10
) -> Tuple[bool, str, List[str]]:
    """Check if retrieval results are sufficient to answer the query.

    Args:
        query: User query
        results: Retrieval results (Top K)
        llm_provider: LLM Provider instance
        llm_config: LLM configuration dict
        max_docs: Maximum documents to evaluate

    Returns:
        (is_sufficient, reasoning, missing_information)
    """
    try:
        # Format documents
        retrieved_docs = format_documents_for_llm(
            results,
            max_docs=max_docs,
            use_episode=True
        )

        # Build prompt
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query,
            retrieved_docs=retrieved_docs
        )

        # Call LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=500,
        )

        # Parse response
        result = parse_json_response(result_text)

        return (
            result["is_sufficient"],
            result["reasoning"],
            result.get("missing_information", [])
        )

    except asyncio.TimeoutError:
        print(f"  ❌ Sufficiency check timeout (30s)")
        return True, "Timeout: LLM took too long", []
    except Exception as e:
        print(f"  ❌ Sufficiency check failed: {e}")
        import traceback
        traceback.print_exc()
        return True, f"Error: {str(e)}", []


async def generate_refined_query(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    max_docs: int = 10
) -> str:
    """Generate a refined query based on retrieval results.

    Args:
        original_query: Original query
        results: Round 1 retrieval results
        missing_info: List of missing information
        llm_provider: LLM Provider instance
        llm_config: LLM configuration
        max_docs: Maximum documents to use

    Returns:
        Refined query string
    """
    try:
        # Format documents
        retrieved_docs = format_documents_for_llm(
            results,
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"

        # Build prompt
        prompt = REFINED_QUERY_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )

        # Call LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=150,
        )

        # Parse and validate
        refined_query = parse_refined_query(result_text, original_query)

        return refined_query

    except asyncio.TimeoutError:
        print(f"  ❌ Query refinement timeout (30s)")
        return original_query
    except Exception as e:
        print(f"  ❌ Query refinement failed: {e}")
        import traceback
        traceback.print_exc()
        return original_query


async def generate_multi_queries(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    max_docs: int = 5,
    num_queries: int = 3
) -> Tuple[List[str], str]:
    """Generate multiple complementary queries for multi-query retrieval.

    Args:
        original_query: Original query
        results: Round 1 retrieval results
        missing_info: List of missing information
        llm_provider: LLM Provider instance
        llm_config: LLM configuration
        max_docs: Maximum documents to use (default 5)
        num_queries: Expected number of queries (default 3)

    Returns:
        (queries_list, reasoning)
    """
    try:
        # Format documents
        retrieved_docs = format_documents_for_llm(
            results,
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"

        # Build prompt
        prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )

        # Call LLM
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=300,
        )

        # Parse and validate
        queries, reasoning = parse_multi_query_response(result_text, original_query)

        print(f"  [Multi-Query] Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"    Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"  [Multi-Query] Strategy: {reasoning}")

        return queries, reasoning

    except asyncio.TimeoutError:
        print(f"  ❌ Multi-query generation timeout (30s)")
        return [original_query], "Timeout: used original query"
    except Exception as e:
        print(f"  ❌ Multi-query generation failed: {e}")
        import traceback
        traceback.print_exc()
        return [original_query], f"Error: {str(e)}"
