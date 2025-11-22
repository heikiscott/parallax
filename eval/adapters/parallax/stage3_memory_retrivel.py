import json
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import time

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import asyncio
from tqdm import tqdm



from eval.adapters.parallax.config import ExperimentConfig

# from eval.adapters.parallax.tools.embedding_provider import EmbeddingProvider
# from eval.adapters.parallax.tools.reranker_provider import RerankerProvider
from agents import vectorize_service
from agents import rerank_service

from eval.adapters.parallax.tools import agentic_utils

# 新增：使用 Memory Layer 的 LLMProvider
from providers.llm.llm_provider import LLMProvider


# This file depends on the rank_bm25 library.
# If you haven't installed it yet, run: pip install rank_bm25


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)
    
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)
    
    # 🔥 验证 stopwords 是否可用
    try:
        from nltk.corpus import stopwords
        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between a query vector and multiple document vectors.

    Args:
        query_vec: A 1D numpy array for the query.
        doc_vecs: A 2D numpy array where each row is a document vector.

    Returns:
        A 1D numpy array of cosine similarity scores.
    """
    # Calculate dot product
    dot_product = np.dot(doc_vecs, query_vec)

    # Calculate norms
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # Calculate cosine similarity, handling potential division by zero
    denominator = query_norm * doc_norms
    # Replace 0s in denominator with a small number to avoid division by zero
    denominator[denominator == 0] = 1e-9

    similarity_scores = dot_product / denominator

    return similarity_scores


def compute_maxsim_score(query_emb: np.ndarray, atomic_fact_embs: List[np.ndarray]) -> float:
    """
    计算 query 与多个 atomic_fact embeddings 的最大相似度（MaxSim策略）
    
    MaxSim 策略的核心思想：
    - 找到与 query 最相关的单个 atomic_fact
    - 只要有一个 atomic_fact 与 query 强相关，就认为整个 event_log 相关
    - 避免被不相关的 fact 稀释分数
    - 适用于记忆检索场景，用户通常只关注某一个方面
    
    优化：使用向量化矩阵运算，速度提升 2-3 倍
    
    Args:
        query_emb: query 的 embedding 向量（1D numpy array）
        atomic_fact_embs: 多个 atomic_fact 的 embedding 向量列表
    
    Returns:
        最大相似度分数（float，范围 [-1, 1]，通常为 [0, 1]）
    """
    if not atomic_fact_embs:
        return 0.0
    
    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return 0.0
    
    # 🔥 优化：使用矩阵运算代替循环（2-3倍加速）
    try:
        # 将 list 转换为矩阵：shape = (n_facts, embedding_dim)
        fact_matrix = np.array(atomic_fact_embs)
        
        # 批量计算所有 fact 的范数
        fact_norms = np.linalg.norm(fact_matrix, axis=1)
        
        # 过滤零向量
        valid_mask = fact_norms > 0
        if not np.any(valid_mask):
            return 0.0
        
        # 向量化计算所有相似度
        # sims = (fact_matrix @ query_emb) / (query_norm * fact_norms)
        dot_products = np.dot(fact_matrix[valid_mask], query_emb)
        sims = dot_products / (query_norm * fact_norms[valid_mask])
        
        # 返回最大相似度
        return float(np.max(sims))
    
    except Exception as e:
        # 回退到循环方式（兼容性保证）
        similarities = []
        for fact_emb in atomic_fact_embs:
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            sim = np.dot(query_emb, fact_emb) / (query_norm * fact_norm)
            similarities.append(sim)
        return max(similarities) if similarities else 0.0


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    This must be identical to the tokenization used during indexing.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    processed_tokens = [
        stemmer.stem(token)
        for token in tokens
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]

    return processed_tokens


def search_with_bm25_index(query: str, bm25, docs, top_n: int = 5):
    """
    Performs BM25 search using a pre-loaded index.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)

    if not tokenized_query:
        print("Warning: Query is empty after tokenization.")
        return []

    doc_scores = bm25.get_scores(tokenized_query)
    results_with_scores = list(zip(docs, doc_scores))
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def reciprocal_rank_fusion(
    emb_results: List[Tuple[dict, float]],
    bm25_results: List[Tuple[dict, float]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """
    使用 RRF (Reciprocal Rank Fusion) 融合 Embedding 和 BM25 检索结果
    
    RRF 是一种无需归一化的融合策略，对排序位置敏感。
    公式：RRF_score(doc) = Σ(1 / (k + rank_i))
    
    优势：
    1. 无需归一化分数（Embedding 和 BM25 分数范围不同）
    2. 简单有效，工业界广泛验证（Elasticsearch 等）
    3. 对头部结果更敏感（高排名贡献更大）
    4. 无需调参（k=60 是经验最优值）
    
    Args:
        emb_results: Embedding 检索结果 [(doc, score), ...]
        bm25_results: BM25 检索结果 [(doc, score), ...]
        k: RRF 常数，通常使用 60（经验值）
    
    Returns:
        融合后的结果 [(doc, rrf_score), ...]，按 RRF 分数降序排列
    
    Example:
        emb_results = [(doc1, 0.92), (doc2, 0.87), (doc3, 0.81)]
        bm25_results = [(doc2, 15.3), (doc1, 12.7), (doc4, 10.2)]
        
        Doc1: 1/(60+1) + 1/(60+2) = 0.0323
        Doc2: 1/(60+2) + 1/(60+1) = 0.0323  
        Doc3: 1/(60+3) + 0        = 0.0159
        Doc4: 0        + 1/(60+3) = 0.0159
        
        融合结果: [(doc1, 0.0323), (doc2, 0.0323), (doc3, 0.0159), (doc4, 0.0159)]
    """
    # 🔥 修复：使用 event_id 作为唯一标识，而不是 Python 内存地址
    # 原因：BM25 和 Embedding 索引分别加载 JSON，创建了不同的 Python 对象
    # 即使内容相同，id() 也会不同，导致无法去重
    doc_rrf_scores = {}  # {event_id: rrf_score}
    doc_map = {}         # {event_id: doc}
    
    # 处理 Embedding 检索结果
    for rank, (doc, score) in enumerate(emb_results, start=1):
        doc_id = doc.get("event_id", id(doc))  # 🔥 优先使用 event_id，回退到 id()
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 处理 BM25 检索结果
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        doc_id = doc.get("event_id", id(doc))  # 🔥 优先使用 event_id，回退到 id()
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 转换回 (doc, score) 格式
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]
    
    return fused_results


def multi_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]],
    k: int = 60
) -> List[Tuple[dict, float]]:
    """
    使用 RRF 融合多个查询的检索结果（多查询融合）
    
    与双路 RRF 类似，但支持融合任意数量的检索结果。
    每个结果集贡献的分数：1 / (k + rank)
    
    原理：
    - 在多个查询中都排名靠前的文档 → 分数累积高 → 最终排名靠前
    - 这是一种"投票机制"：多个查询都认为相关的文档更可能真正相关
    
    Args:
        results_list: 多个检索结果列表 [
            [(doc1, score), (doc2, score), ...],  # Query 1 结果
            [(doc3, score), (doc1, score), ...],  # Query 2 结果
            [(doc4, score), (doc2, score), ...],  # Query 3 结果
        ]
        k: RRF 常数（默认 60）
    
    Returns:
        融合后的结果 [(doc, rrf_score), ...]，按 RRF 分数降序排列
    
    Example:
        Query 1 结果: [(doc_A, 0.9), (doc_B, 0.8), (doc_C, 0.7)]
        Query 2 结果: [(doc_B, 0.88), (doc_D, 0.82), (doc_A, 0.75)]
        Query 3 结果: [(doc_A, 0.92), (doc_E, 0.85), (doc_B, 0.80)]
        
        RRF 分数计算：
        doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 0.0323  ← 在 Q1,Q2,Q3 都出现
        doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0323  ← 在 Q1,Q2,Q3 都出现
        doc_C: 1/(60+3) + 0        + 0        = 0.0159  ← 只在 Q1 出现
        doc_D: 0        + 1/(60+2) + 0        = 0.0161  ← 只在 Q2 出现
        doc_E: 0        + 0        + 1/(60+2) = 0.0161  ← 只在 Q3 出现
        
        融合结果: doc_A 和 doc_B 排名最高（被多个查询认可）
    """
    if not results_list:
        return []
    
    # 如果只有一个结果集，直接返回
    if len(results_list) == 1:
        return results_list[0]
    
    # 🔥 修复：使用 event_id 作为唯一标识，而不是 Python 内存地址
    # 原因：BM25 和 Embedding 索引分别加载 JSON，创建了不同的 Python 对象
    # 即使内容相同，id() 也会不同，导致无法去重
    doc_rrf_scores = {}  # {event_id: rrf_score}
    doc_map = {}         # {event_id: doc}
    
    # 遍历每个查询的检索结果
    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = doc.get("event_id", id(doc))  # 🔥 优先使用 event_id，回退到 id()
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # 累加 RRF 分数
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # 按 RRF 分数排序
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 转换回 (doc, score) 格式
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]
    
    return fused_results


async def lightweight_retrieval(
    query: str,
    emb_index,
    bm25,
    docs,
    config: ExperimentConfig,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    轻量级快速检索（无 LLM 调用，纯算法检索）
    
    流程：
    1. 并行执行 Embedding 和 BM25 检索
    2. 各取 Top-50 候选
    3. 使用 RRF 融合
    4. 返回 Top-20 结果
    
    优势：
    - 速度快：无 LLM 调用，纯向量/词法检索
    - 成本低：不消耗 LLM API 费用
    - 稳定：无网络依赖，纯本地计算
    
    适用场景：
    - 对延迟敏感的场景
    - 预算有限的场景
    - 查询简单明确的场景
    
    Args:
        query: 用户查询
        emb_index: Embedding 索引
        bm25: BM25 索引
        docs: 文档列表
        config: 实验配置
    
    Returns:
        (final_results, metadata)
    """
    start_time = time.time()
    
    metadata = {
        "retrieval_mode": "lightweight",
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    # ========== 并行执行 Embedding 和 BM25 检索 ==========
    emb_task = search_with_emb_index(
        query, 
        emb_index, 
        top_n=config.lightweight_emb_top_n  # 默认 50
    )
    bm25_task = asyncio.to_thread(
        search_with_bm25_index, 
        query, 
        bm25, 
        docs, 
        config.lightweight_bm25_top_n  # 默认 50
    )
    
    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)
    
    metadata["emb_count"] = len(emb_results)
    metadata["bm25_count"] = len(bm25_results)
    
    # ========== RRF 融合 ==========
    if not emb_results and not bm25_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    elif not emb_results:
        final_results = bm25_results[:config.lightweight_final_top_n]
    elif not bm25_results:
        final_results = emb_results[:config.lightweight_final_top_n]
    else:
        # 使用 RRF 融合
        fused_results = reciprocal_rank_fusion(
            emb_results, 
            bm25_results, 
            k=60  # 标准 RRF 参数
        )
        final_results = fused_results[:config.lightweight_final_top_n]  # 默认 20
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    return final_results, metadata


async def search_with_emb_index(
    query: str, 
    emb_index, 
    top_n: int = 5,
    query_embedding: Optional[np.ndarray] = None  # 🔥 支持预计算的 embedding
):
    """
    使用 MaxSim 策略执行 embedding 检索
    
    对于包含 atomic_facts 的文档：
    - 计算 query 与每个 atomic_fact 的相似度
    - 取最大相似度作为文档分数（MaxSim策略）
    
    对于传统文档：
    - 回退到使用 subject/summary/episode 字段
    - 取这些字段中的最大相似度
    
    优化：支持预计算的 query embedding，避免重复 API 调用
    
    Args:
        query: 查询文本
        emb_index: 预构建的 embedding 索引
        top_n: 返回的结果数量
        query_embedding: 可选的预计算 query embedding（避免重复计算）
    
    Returns:
        排序后的 (文档, 分数) 列表
    """
    # 获取 query 的 embedding（如果未提供则调用 API）
    if query_embedding is not None:
        query_vec = query_embedding
    else:
        query_vec = np.array(await vectorize_service.get_text_embedding(query))
    
    query_norm = np.linalg.norm(query_vec)
    
    # 如果 query 向量为零，返回空结果
    if query_norm == 0:
        return []
    
    # 存储每个文档的 MaxSim 分数
    doc_scores = []
    
    for item in emb_index:
        doc = item.get("doc")
        embeddings = item.get("embeddings", {})
        
        if not embeddings:
            continue
        
        # 优先使用 atomic_facts（MaxSim策略）
        if "atomic_facts" in embeddings:
            atomic_fact_embs = embeddings["atomic_facts"]
            if atomic_fact_embs:
                # 🔥 核心：使用 MaxSim 计算分数
                score = compute_maxsim_score(query_vec, atomic_fact_embs)
                doc_scores.append((doc, score))
                continue
        
        # 回退到传统字段（保持向后兼容）
        # 对于传统字段，也使用 MaxSim 策略（取最大值）
        field_scores = []
        for field in ["subject", "summary", "episode"]:
            if field in embeddings:
                field_emb = embeddings[field]
                field_norm = np.linalg.norm(field_emb)
                
                if field_norm > 0:
                    sim = np.dot(query_vec, field_emb) / (query_norm * field_norm)
                    field_scores.append(sim)
        
        if field_scores:
            score = max(field_scores)
            doc_scores.append((doc, score))
    
    if not doc_scores:
        return []
    
    # 按分数降序排序并返回 Top-N
    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


async def hybrid_search_with_rrf(
    query: str,
    emb_index,
    bm25,
    docs,
    top_n: int = 40,
    emb_candidates: int = 50,
    bm25_candidates: int = 50,
    rrf_k: int = 60,
    query_embedding: Optional[np.ndarray] = None  # 🔥 支持预计算的 embedding
) -> List[Tuple[dict, float]]:
    """
    使用 RRF 融合 Embedding 和 BM25 检索结果（混合检索）
    
    执行流程：
    1. 并行执行 Embedding (MaxSim) 和 BM25 检索
    2. 每种方法分别召回 top-N 候选文档
    3. 使用 RRF 融合两个结果集
    4. 返回融合后的 Top-N 文档
    
    为什么使用混合检索：
    - Embedding: 擅长语义匹配，但对罕见词和精确匹配较弱
    - BM25: 擅长精确匹配和罕见词，但语义理解较弱
    - RRF 融合: 结合两者优势，提升召回率 15-20%
    
    Args:
        query: 用户查询
        emb_index: Embedding 索引
        bm25: BM25 索引
        docs: 文档列表（用于 BM25）
        top_n: 最终返回的结果数量（默认 40）
        emb_candidates: Embedding 检索的候选数量（默认 50）
        bm25_candidates: BM25 检索的候选数量（默认 50）
        rrf_k: RRF 参数 k（默认 60，经验最优值）
    
    Returns:
        融合后的 Top-N 结果 [(doc, rrf_score), ...]
    
    Example:
        Query: "他喜欢吃什么？"
        
        Embedding Top-3:
        - (doc_A: "用户喜爱川菜", 0.92)  # 语义匹配"喜欢"="喜爱"
        - (doc_B: "用户偏好清淡口味", 0.78)
        - (doc_C: "成都是美食之都", 0.65)
        
        BM25 Top-3:
        - (doc_A: "用户喜爱川菜", 15.3)  # 精确匹配"喜欢"
        - (doc_D: "喜欢吃火锅", 12.7)  # 精确匹配"喜欢吃"
        - (doc_E: "最喜欢的菜是麻婆豆腐", 10.2)
        
        RRF 融合:
        - doc_A: 同时在两个结果中排名靠前 → 最高分 ✅
        - doc_D: 只在 BM25 中排名高
        - doc_B: 只在 Embedding 中排名高
        
        最终: [(doc_A, 0.0323), (doc_D, 0.0161), (doc_B, 0.0161), ...]
    """
    # 并行执行 Embedding 和 BM25 检索（提高效率）
    emb_task = search_with_emb_index(
        query, emb_index, top_n=emb_candidates, query_embedding=query_embedding
    )
    bm25_task = asyncio.to_thread(search_with_bm25_index, query, bm25, docs, bm25_candidates)
    
    # 等待两个检索任务完成
    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)
    
    # 如果其中一个检索结果为空，返回另一个
    if not emb_results and not bm25_results:
        return []
    elif not emb_results:
        print(f"Warning: Embedding search returned no results for query: {query}")
        return bm25_results[:top_n]
    elif not bm25_results:
        print(f"Warning: BM25 search returned no results for query: {query}")
        return emb_results[:top_n]
    
    # 使用 RRF 融合两个检索结果
    fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=rrf_k)
    
    # 打印融合统计信息（用于调试）
    print(f"Hybrid search: Emb={len(emb_results)}, BM25={len(bm25_results)}, Fused={len(fused_results)}, Returning top-{top_n}")
    
    return fused_results[:top_n]


async def agentic_retrieval(
    query: str,
    config: ExperimentConfig,
    llm_provider: LLMProvider,  # 改用 LLMProvider
    llm_config: dict,
    emb_index,
    bm25,
    docs,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    Agentic 多轮检索（LLM 引导）- 新流程
    
    流程：
    1. Round 1: 混合检索 → Top 20 → Rerank → Top 5 → LLM 判断充分性
    2. 如果充分：返回原始 Top 20（rerank 前的）
    3. 如果不充分：
       - LLM 生成改进查询
       - Round 2: 检索并合并到 40 个
       - Rerank 40 个 → 返回最终结果
    
    Args:
        query: 用户查询
        config: 实验配置
        llm_provider: LLM Provider (Memory Layer)
        llm_config: LLM 配置字典
        emb_index: Embedding 索引
        bm25: BM25 索引
        docs: 文档列表
    
    Returns:
        (final_results, metadata)
    """
    import time
    start_time = time.time()
    
    metadata = {
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "refined_query": None,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }
    
    print(f"\n{'='*60}")
    print(f"Agentic Retrieval: {query[:60]}...")
    print(f"{'='*60}")
    print(f"  [Start] Time: {time.strftime('%H:%M:%S')}")
    
    # ========== Round 1: 混合检索 Top 20 ==========
    print(f"  [Round 1] Hybrid search for Top 20...")
    
    round1_top20 = await hybrid_search_with_rrf(
        query=query,
        emb_index=emb_index,
        bm25=bm25,
        docs=docs,
        top_n=20,  # 🔥 只取 Top 20
        emb_candidates=config.hybrid_emb_candidates,
        bm25_candidates=config.hybrid_bm25_candidates,
        rrf_k=config.hybrid_rrf_k,
    )
    
    metadata["round1_count"] = len(round1_top20)
    print(f"  [Round 1] Retrieved {len(round1_top20)} documents")
    
    if not round1_top20:
        print(f"  [Warning] No results from Round 1")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    
    # ========== Rerank Top 20 → Top 5 用于 Sufficiency Check ==========
    print(f"  [Rerank] Reranking Top 20 to get Top 5 for sufficiency check...")
    
    if config.use_reranker:
        reranked_top5 = await reranker_search(
            query=query,
            results=round1_top20,
            top_n=5,  # 🔥 只取 Top 5 给 LLM 判断
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )
        metadata["round1_reranked_count"] = len(reranked_top5)
        print(f"  [Rerank] Got Top 5 for sufficiency check")
    else:
        # 如果不使用 reranker，直接取前 5 个
        reranked_top5 = round1_top20[:5]
        metadata["round1_reranked_count"] = 5
        print(f"  [No Rerank] Using original Top 5 for sufficiency check")
    
    if not reranked_top5:
        print(f"  [Warning] Reranking failed, falling back to original Top 20")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_top20, metadata
    
    # ========== LLM Sufficiency Check ==========
    print(f"  [LLM] Checking sufficiency on Top 5...")
    
    is_sufficient, reasoning, missing_info = await agentic_utils.check_sufficiency(
        query=query,
        results=reranked_top5,  # 🔥 使用 reranked Top 5
        llm_provider=llm_provider,  # 使用 LLMProvider
        llm_config=llm_config,
        max_docs=5  # 🔥 明确只检查 5 个文档
    )
    
    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning
    
    print(f"  [LLM] Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
    print(f"  [LLM] Reasoning: {reasoning}")
    
    # ========== 如果充分：返回原始 Round 1 的 Top 20 ==========
    if is_sufficient:
        print(f"  [Decision] Sufficient! Using original Round 1 Top 20 results")
        
        final_results = round1_top20  # 🔥 返回原始的 Top 20（不是 reranked 的）
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        
        print(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata
    
    # ========== 如果不充分：进入 Round 2 ==========
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    print(f"  [Decision] Insufficient, entering Round 2")
    print(f"  [Missing] {', '.join(missing_info) if missing_info else 'N/A'}")
    
    # ========== LLM 生成多个改进查询（多查询策略）==========
    use_multi_query = getattr(config, 'use_multi_query', True)  # 🔥 默认启用多查询
    
    if use_multi_query:
        print(f"  [LLM] Generating multiple refined queries...")
        
        # 🔥 生成 2-3 个互补查询
        refined_queries, query_strategy = await agentic_utils.generate_multi_queries(
            original_query=query,
            results=reranked_top5,  # 🔥 基于 Top 5 生成改进查询
            missing_info=missing_info,
            llm_provider=llm_provider,  # 使用 LLMProvider
            llm_config=llm_config,
            max_docs=5,
            num_queries=3  # 期望生成 3 个查询
        )
        
        metadata["refined_queries"] = refined_queries
        metadata["query_strategy"] = query_strategy
        metadata["num_queries"] = len(refined_queries)
        
        # ========== Round 2: 并行执行多个查询检索 ==========
        print(f"  [Round 2] Executing {len(refined_queries)} queries in parallel...")
        
        # 🔥 并行执行所有查询的混合检索
        multi_query_tasks = [
            hybrid_search_with_rrf(
                query=q,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=50,  # 🔥 每个查询召回 50 个候选
                emb_candidates=config.hybrid_emb_candidates,
                bm25_candidates=config.hybrid_bm25_candidates,
                rrf_k=config.hybrid_rrf_k,
            )
            for q in refined_queries
        ]
        
        # 等待所有查询完成
        multi_query_results = await asyncio.gather(*multi_query_tasks)
        
        # 打印每个查询的召回数
        for i, results in enumerate(multi_query_results, 1):
            print(f"    Query {i}: Retrieved {len(results)} documents")
        
        # ========== 使用 RRF 融合多个查询的结果 ==========
        print(f"  [Multi-RRF] Fusing results from {len(refined_queries)} queries...")
        
        # 🔥 使用多查询 RRF 融合
        round2_results = multi_rrf_fusion(
            results_list=multi_query_results,
            k=config.hybrid_rrf_k  # 使用相同的 k 参数
        )
        
        # 取 Top 40 用于后续合并
        round2_results = round2_results[:40]
        
        metadata["round2_count"] = len(round2_results)
        metadata["multi_query_total_docs"] = sum(len(r) for r in multi_query_results)
        
        print(f"  [Multi-RRF] Fused {metadata['multi_query_total_docs']} → {len(round2_results)} unique documents")
    
    else:
        # 🔥 回退到单查询模式（保持向后兼容）
        print(f"  [LLM] Generating single refined query (legacy mode)...")
        
        refined_query = await agentic_utils.generate_refined_query(
            original_query=query,
            results=reranked_top5,
            missing_info=missing_info,
            llm_provider=llm_provider,  # 使用 LLMProvider
            llm_config=llm_config,
            max_docs=5
        )
        
        metadata["refined_query"] = refined_query
        print(f"  [LLM] Refined query: {refined_query}")
        
        # ========== Round 2: 使用单个改进查询检索 ==========
        print(f"  [Round 2] Hybrid search with refined query...")
        
        round2_results = await hybrid_search_with_rrf(
            query=refined_query,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=40,
            emb_candidates=config.hybrid_emb_candidates,
            bm25_candidates=config.hybrid_bm25_candidates,
            rrf_k=config.hybrid_rrf_k,
        )
        
        metadata["round2_count"] = len(round2_results)
        print(f"  [Round 2] Retrieved {len(round2_results)} documents")
    
    # ========== 合并：确保总共 40 个文档 ==========
    print(f"  [Merge] Combining Round 1 and Round 2 to ensure 40 documents...")
    
    # 🔥 修复：使用 event_id 去重，而不是 Python 内存地址
    # 原因：BM25 和 Embedding 索引分别加载 JSON，创建了不同的 Python 对象
    round1_ids = {doc.get("event_id", id(doc)) for doc, _ in round1_top20}
    round2_unique = [(doc, score) for doc, score in round2_results 
                     if doc.get("event_id", id(doc)) not in round1_ids]
    
    # 合并：Round1 Top20 + Round2 去重后的文档（确保总数=40）
    combined_results = round1_top20.copy()  # 先加入 Round1 的 20 个
    needed_from_round2 = 40 - len(combined_results)  # 需要 20 个
    combined_results.extend(round2_unique[:needed_from_round2])
    
    actual_count = len(combined_results)
    duplicates_removed = len(round2_results) - len(round2_unique)
    round2_added = len(round2_unique[:needed_from_round2])
    
    print(f"  [Merge] Round1 Top20: 20 documents")
    print(f"  [Merge] Round2 duplicates removed: {duplicates_removed} documents")
    print(f"  [Merge] Round2 unique added: {round2_added} documents")
    print(f"  [Merge] Combined total: {actual_count} documents (target: 40)")
    
    # ========== Rerank 合并后的 40 个文档 ==========
    if config.use_reranker and len(combined_results) > 0:
        print(f"  [Rerank] Reranking {len(combined_results)} documents...")
        
        final_results = await reranker_search(
            query=query,  # 🔥 使用原始查询进行 rerank
            results=combined_results,
            top_n=20,  # 🔥 返回 Top 20 作为最终结果
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )
        
        print(f"  [Rerank] Final Top 20 selected")
    else:
        # 不使用 Reranker，直接返回 Top 20
        final_results = combined_results[:20]
        print(f"  [No Rerank] Returning Top 20 from combined results")
    
    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000
    
    print(f"  [Complete] Final: {len(final_results)} docs | Latency: {metadata['total_latency_ms']:.0f}ms")
    print(f"{'='*60}\n")
    
    return final_results, metadata


async def reranker_search(
    query: str,
    results: List[Tuple[dict, float]],
    top_n: int = 20,
    reranker_instruction: str = None,
    batch_size: int = 10,  # 🔥 批次大小（Reranker API 通常限制）
    max_retries: int = 3,  # 🔥 最大重试次数
    retry_delay: float = 2.0,  # 🔥 重试基础延迟
    timeout: float = 30.0,  # 🔥 单批次超时
    fallback_threshold: float = 0.3,  # 🔥 降级阈值
    config: ExperimentConfig = None,  # 🔥 新增：实验配置（用于获取并发数）
):
    """
    使用 reranker 模型对检索结果进行重排序（支持批量并发处理 + 增强稳定性）
    
    对于包含 event_log 的文档：
    - 格式化为多行文本：时间 + 每句 atomic_fact 单独一行
    - 例如：
      2024-10-31 14:30:00
      用户喜欢吃川菜
      用户最喜欢的川菜是麻婆豆腐
      用户不喜欢太辣的菜
    
    对于传统文档：
    - 回退到使用 episode 字段
    
    优化策略（稳定性优先）：
    - 将文档分批处理（每批 batch_size 个）
    - 串行处理批次（避免 API 限流）
    - 每个批次支持重试和指数退避
    - 成功率过低时自动降级到原始排序
    - 单批次超时保护
    
    Args:
        query: 用户查询
        results: 初步检索结果（来自 embedding 或 BM25）
        top_n: 返回的结果数量（默认 20）
        reranker_instruction: Reranker 指令
        batch_size: 每批处理的文档数量（默认 10）
        max_retries: 每个批次的最大重试次数（默认 3）
        retry_delay: 重试基础延迟秒数（默认 2.0，指数退避）
        timeout: 单批次超时时间（秒，默认 30）
        fallback_threshold: 成功率低于此值时降级（默认 0.3）
    
    Returns:
        重排序后的 Top-N 结果列表
    """
    if not results:
        return []

    # 第一步：格式化文档
    docs_with_episode = []
    doc_texts = []
    original_indices = []  # 🔥 记录原始索引，用于还原
    
    for idx, (doc, score) in enumerate(results):
        # 优先使用 event_log 格式化文本（如果存在）
        if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
            event_log = doc["event_log"]
            time_str = event_log.get("time", "")
            atomic_facts = event_log.get("atomic_fact", [])

            if isinstance(atomic_facts, list) and atomic_facts:
                # 🔥 格式化为多行文本（时间 + 每句 atomic_fact 单独一行）
                formatted_lines = []
                if time_str:
                    formatted_lines.append(time_str)
                
                # 🔥 修复：兼容两种格式（字符串 / 字典）
                for fact in atomic_facts:
                    if isinstance(fact, dict) and "fact" in fact:
                        # 新格式：{"fact": "...", "embedding": [...]}
                        formatted_lines.append(fact["fact"])
                    elif isinstance(fact, str):
                        # 旧格式：纯字符串
                        formatted_lines.append(fact)
                
                formatted_text = "\n".join(formatted_lines)

                docs_with_episode.append(doc)
                doc_texts.append(formatted_text)
                original_indices.append(idx)
                continue

        # 回退到原有的 episode 字段（保持向后兼容）
        if episode_text := doc.get("episode"):
            docs_with_episode.append(doc)
            doc_texts.append(episode_text)
            original_indices.append(idx)

    if not doc_texts:
        return []

    reranker = rerank_service.get_rerank_service()
    print(f"Reranking query: {query}")
    print(f"Reranking {len(doc_texts)} documents in batches of {batch_size}...")
    print(f"Reranking instruction: {reranker_instruction}")
    
    # 🔥 第二步：批量处理（串行 + 重试 + 降级）
    # 将文档分批，每批 batch_size 个
    batches = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        batches.append((i, batch))  # 保存起始索引和批次数据
    
    print(f"Split into {len(batches)} batches for serial reranking")
    
    # 🔥 处理单个批次（带重试 + 超时 + 指数退避）
    async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
        """处理单个批次（带重试和超时）"""
        for attempt in range(max_retries):
            try:
                # 🔥 添加超时保护
                batch_results = await asyncio.wait_for(
                    reranker._make_rerank_request(
                    query, batch_texts, instruction=reranker_instruction
                    ),
                    timeout=timeout
                )
                
                # 调整索引：将批次内的索引映射回全局索引
                for item in batch_results["results"]:
                    item["global_index"] = start_idx + item["index"]
                
                if attempt > 0:
                    print(f"  ✓ Batch at {start_idx} succeeded on attempt {attempt + 1}")
                return batch_results["results"]
                
            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # 指数退避：2s, 4s, 8s
                    print(f"  ⏱️  Batch at {start_idx} timeout (attempt {attempt + 1}), retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  ❌ Batch at {start_idx} timeout after {max_retries} attempts")
                    return []
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  ⚠️  Batch at {start_idx} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  ❌ Batch at {start_idx} failed after {max_retries} attempts: {e}")
                    return []
    
    # 🔥 可控并发处理（稳妥的并发策略）
    # 从配置获取并发数（默认 2，稳妥值）
    max_concurrent = getattr(config, 'reranker_concurrent_batches', 2)
    
    batch_results_list = []
    successful_batches = 0
    
    # 分组处理，每组最多 max_concurrent 个批次并发
    for group_start in range(0, len(batches), max_concurrent):
        group_batches = batches[group_start : group_start + max_concurrent]
        
        print(f"  Processing batch group {group_start//max_concurrent + 1} ({len(group_batches)} batches in parallel)...")
        
        # 🔥 并发处理当前组的所有批次
        tasks = [
            process_batch_with_retry(start_idx, batch) 
            for start_idx, batch in group_batches
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计成功的批次
        for result in group_results:
            if isinstance(result, list) and result:
                batch_results_list.append(result)
                successful_batches += 1
            else:
                batch_results_list.append([])
        
        # 组间延迟（进一步降低，激进加速）
        if group_start + max_concurrent < len(batches):
            await asyncio.sleep(0.3)  # 🔥 组间 0.3 秒间隔（从 0.8s 降低）
    
    # 🔥 第三步：合并所有批次的结果 + 降级策略
    all_rerank_results = []
    for batch_results in batch_results_list:
        all_rerank_results.extend(batch_results)
    
    # 🔥 计算成功率
    success_rate = successful_batches / len(batches) if batches else 0.0
    print(f"Reranker success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)")
    
    # 🔥 降级策略 1: 完全失败
    if not all_rerank_results:
        print("⚠️ Warning: All reranker batches failed, using original ranking as fallback")
        return results[:top_n]
    
    # 🔥 降级策略 2: 成功率过低
    if success_rate < fallback_threshold:
        print(f"⚠️ Warning: Reranker success rate too low ({success_rate:.1%} < {fallback_threshold:.1%}), using original ranking")
        return results[:top_n]
    
    print(f"Reranking complete: {len(all_rerank_results)} documents scored")
    
    # 第四步：按 reranker 分数排序并返回 Top-N
    sorted_results = sorted(
        all_rerank_results, 
        key=lambda x: x["relevance_score"], 
        reverse=True
    )[:top_n]
    
    # 映射回原始文档
    final_results = [
        (results[original_indices[item["global_index"]]][0], item["relevance_score"])
        for item in sorted_results
    ]
    
    return final_results


async def main():
    """Main function to perform batch search and save results in nemori format."""
    # --- Configuration ---
    config = ExperimentConfig()
    # 🔥 修正：实际文件在 locomo_eval/ 目录下，而不是 results/ 目录
    bm25_index_dir = (
        Path(__file__).parent / config.experiment_name / "bm25_index"
    )
    emb_index_dir = (
        Path(__file__).parent / config.experiment_name / "vectors"
    )
    save_dir = Path(__file__).parent / config.experiment_name

    dataset_path = config.datase_path
    results_output_path = save_dir / "search_results.json"
    
    # 🔥 断点续传：检查点文件路径
    checkpoint_path = save_dir / "search_results_checkpoint.json"

    # Ensure NLTK data is ready
    ensure_nltk_data()
    
    # 🔥 初始化 LLM Provider（用于 Agentic 检索）
    llm_provider = None
    llm_config = None
    if config.use_agentic_retrieval:
        if agentic_utils is None:
            print("Error: agentic_utils not found, cannot use agentic retrieval")
            print("Please check that tools/agentic_utils.py exists")
            return
        
        llm_config = config.llm_config.get(config.llm_service, config.llm_config["openai"])
        
        # 使用 Memory Layer 的 LLMProvider 替代 AsyncOpenAI
        llm_provider = LLMProvider(
            provider_type="openai",
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=int(llm_config.get("max_tokens", 32768)),
        )
        print(f"✅ LLM Provider initialized for agentic retrieval")
        print(f"   Model: {llm_config['model']}")

    # Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 🔥 断点续传：加载已有的检查点
    all_search_results = {}
    processed_conversations = set()
    
    if checkpoint_path.exists():
        print(f"\n🔄 Found checkpoint file: {checkpoint_path}")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                all_search_results = json.load(f)
            processed_conversations = set(all_search_results.keys())
            print(f"✅ Loaded {len(processed_conversations)} conversations from checkpoint")
            print(f"   Already processed: {sorted(processed_conversations)}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"   Starting from scratch...")
            all_search_results = {}
            processed_conversations = set()
    else:
        print(f"\n🆕 No checkpoint found, starting from scratch")

    # Iterate through the dataset, assuming the index of the dataset list
    # corresponds to the conversation index number.
    for i, conversation_data in enumerate(dataset):
        conv_id = f"locomo_exp_user_{i}"
        
        # 🔥 断点续传：跳过已处理的对话
        if conv_id in processed_conversations:
            print(f"\n⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            continue

        speaker_a = conversation_data["conversation"].get("speaker_a")
        speaker_b = conversation_data["conversation"].get("speaker_b")
        print(f"\n--- Processing Conversation ID: {conv_id} ({i+1}/{len(dataset)}) ---")

        if "qa" not in conversation_data:
            print(f"Warning: No 'qa' key found in conversation #{i}. Skipping.")
            continue

        # --- Load index once per conversation ---
        # 🔥 如果使用混合检索，需要同时加载 Embedding 和 BM25 索引
        if config.use_hybrid_search:
            # 加载 Embedding 索引
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                print(
                    f"Error: Embedding index not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)
            
            # 加载 BM25 索引
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                print(
                    f"Error: BM25 index not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]
            
            print(f"Loaded both Embedding and BM25 indexes for conversation {i} (Hybrid Search)")
        
        elif config.use_emb:
            # 仅加载 Embedding 索引
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                print(
                    f"Error: Index file not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)
        else:
            # 仅加载 BM25 索引
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                print(
                    f"Error: Index file not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]

        # Parallelize per-question retrieval with bounded concurrency
        # 🔥 增加并发数：Agentic 检索时也使用更高并发（10 → 20）
        max_concurrent = 20 if config.use_agentic_retrieval else 128
        sem = asyncio.Semaphore(max_concurrent)
        
        if config.use_agentic_retrieval:
            print(f"  🚀 Agentic retrieval enabled with HIGH CONCURRENCY: {max_concurrent} concurrent requests")

        async def process_single_qa(qa_pair):
            """处理单个 QA 对（支持多种检索模式）"""
            question = qa_pair.get("question")
            if not question:
                return None
            if qa_pair.get("category") == 5:
                print(f"Skipping question {question} because it is category 5")
                return None
            
            # 开始计时
            qa_start_time = time.time()
            
            try:
                async with sem:
                    retrieval_metadata = {}
                    
                    # ========== 检索模式选择 ==========
                    if config.retrieval_mode == "agentic":
                        # 🔥 Agentic 多轮检索（复杂但质量高）
                        top_results, retrieval_metadata = await agentic_retrieval(
                            query=question,
                            config=config,
                            llm_provider=llm_provider,  # 使用 LLMProvider
                            llm_config=llm_config,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                        )
                    
                    elif config.retrieval_mode == "lightweight":
                        # 🔥 轻量级快速检索（快速但质量略低）
                        top_results, retrieval_metadata = await lightweight_retrieval(
                            query=question,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                            config=config,
                        )
                    
                    else:
                        # 🔥 传统检索分支（保持向后兼容）
                        if config.use_reranker:
                            # 第一阶段：初步检索，召回 Top-N 候选
                            if config.use_hybrid_search:
                                # 混合检索：Embedding (MaxSim) + BM25 + RRF 融合
                                results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=config.emb_recall_top_n,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k
                                )
                            elif config.use_emb:
                                # 单独使用 Embedding + MaxSim 检索
                                results = await search_with_emb_index(
                                    query=question, 
                                    emb_index=emb_index, 
                                    top_n=config.emb_recall_top_n
                                )
                            else:
                                # 单独使用 BM25 检索
                                results = await asyncio.to_thread(
                                    search_with_bm25_index, 
                                    question, 
                                    bm25, 
                                    docs, 
                                    config.emb_recall_top_n
                                )
                            
                            # 第二阶段：Reranker 重排序
                            top_results = await reranker_search(
                                query=question,
                                results=results,
                                top_n=config.reranker_top_n,
                                reranker_instruction=config.reranker_instruction,
                                batch_size=config.reranker_batch_size,
                                max_retries=config.reranker_max_retries,
                                retry_delay=config.reranker_retry_delay,
                                timeout=config.reranker_timeout,
                                fallback_threshold=config.reranker_fallback_threshold,
                                config=config,
                            )
                        else:
                            # 单阶段检索（不使用 Reranker）
                            if config.use_hybrid_search:
                                top_results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=20,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k
                                )
                            elif config.use_emb:
                                top_results = await search_with_emb_index(
                                    query=question, emb_index=emb_index, top_n=20
                                )
                            else:
                                top_results = await asyncio.to_thread(
                                    search_with_bm25_index, question, bm25, docs, 20
                                )
                        
                        # 添加检索时间统计
                        retrieval_metadata = {
                            "retrieval_mode": "traditional",
                            "use_reranker": config.use_reranker,
                            "use_hybrid_search": config.use_hybrid_search,
                        }

                    # ========== 提取 event_ids ==========
                    event_ids = []
                    if top_results:
                        for doc, score in top_results:
                            event_id = doc.get('event_id')
                            if event_id:
                                event_ids.append(event_id)

                    # 计算处理时间
                    qa_latency_ms = (time.time() - qa_start_time) * 1000
                    
                    result = {
                        "query": question,
                        "event_ids": event_ids,  # 🔥 返回 event_ids 而不是 context
                        "original_qa": qa_pair,
                        "retrieval_metadata": {
                            **retrieval_metadata,
                            "qa_latency_ms": qa_latency_ms,
                            "target_event_ids_count": len(top_results),  # 记录目标数量
                            "actual_event_ids_count": len(event_ids),    # 记录实际提取的数量
                        }
                    }
                    
                    return result
                    
            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                import traceback
                traceback.print_exc()
                return None

        tasks = [
            asyncio.create_task(process_single_qa(qa_pair))
            for qa_pair in conversation_data["qa"]
        ]
        results_for_conv = [
            res for res in await asyncio.gather(*tasks) if res is not None
        ]

        all_search_results[conv_id] = results_for_conv
        
        # 🔥 断点续传：每处理完一个对话就保存检查点
        try:
            print(f"💾 Saving checkpoint after conversation {conv_id}...")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_search_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Checkpoint saved: {len(all_search_results)} conversations")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")

    # Save all results to a single JSON file in the specified format
    print(f"\n{'='*60}")
    print(f"🎉 All conversations processed!")
    print(f"{'='*60}")
    print(f"\nSaving final results to: {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Batch search and retrieval complete!")
    print(f"   Total conversations: {len(all_search_results)}")
    
    # 🔥 断点续传：完成后删除检查点文件
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"🗑️  Checkpoint file removed (task completed)")
        except Exception as e:
            print(f"⚠️  Failed to remove checkpoint: {e}")

    # Clean up resources
    reranker = rerank_service.get_rerank_service()
    # Assuming the service is DeepInfraRerankService, which has a close method.
    if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
        await reranker.close()


if __name__ == "__main__":
    asyncio.run(main())
