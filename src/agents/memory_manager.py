from __future__ import annotations

from typing import Any, List, Optional
import logging
import asyncio

from datetime import datetime, timedelta
import jieba
import numpy as np
import time
from typing import Dict, Any
from dataclasses import dataclass

from memory.schema import Memory
from services.mem_memorize import memorize
from memory.extraction import MemorizeRequest
from .fetch_memory_service import get_fetch_memory_service
from .dtos.memory_query import (
    FetchMemoryRequest,
    FetchMemoryResponse,
    RetrieveMemoryRequest,
    RetrieveMemoryResponse,
    Metadata,
)
from core.di import get_bean_by_type
from infra.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from core.observation.tracing.decorators import trace_logger
from core.nlp.stopwords_utils import filter_stopwords
from utils.datetime_utils import from_iso_format, get_now_with_timezone
from infra.adapters.out.persistence.repository.memunit_raw_repository import (
    MemUnitRawRepository,
)
from infra.adapters.out.persistence.document.memory.user_profile import (
    UserProfile,
)
from infra.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from retrieval.services.vectorize import get_vectorize_service
from retrieval.services.rerank import get_rerank_service
from retrieval.core.utils import lightweight_retrieval, build_bm25_index, search_with_bm25
import os

logger = logging.getLogger(__name__)


@dataclass
class EventLogCandidate:
    """Event Log 候选对象（用于从 atomic_fact 检索）"""
    episode_id: str
    user_id: str
    group_id: str
    timestamp: datetime
    episode: str  # atomic_fact 内容
    summary: str
    subject: str
    extend: dict  # 包含 embedding


class MemoryManager:
    """Unified memory interface.

    提供以下主要功能:
    - memorize: 接受原始数据并持久化存储
    - fetch_mem: 通过键检索记忆字段，支持多种记忆类型
    - retrieve_mem: 基于提示词检索方法的记忆读取
    """

    def __init__(self) -> None:
        # 获取记忆服务实例
        self._fetch_service = get_fetch_memory_service()

        logger.info(
            "MemoryManager initialized with fetch_mem_service and retrieve_mem_service"
        )

    # --------- Write path (raw data -> memorize) ---------
    @trace_logger(operation_name="agents 记忆存储")
    async def memorize(self, memorize_request: MemorizeRequest) -> List[Memory]:
        """Memorize a heterogeneous list of raw items.

        Accepts list[Any], where each item can be one of the typed raw dataclasses
        (ChatRawData / EmailRawData / MemoRawData / LincDocRawData) or any dict-like
        object. Each item is stored as a MemoryCell with a synthetic key.
        """
        memories = await memorize(memorize_request)
        return memories

    # --------- Read path (query -> fetch_mem) ---------
    # 基于kv的记忆读取，包括静态与动态记忆
    @trace_logger(operation_name="agents 记忆读取")
    async def fetch_mem(self, request: FetchMemoryRequest) -> FetchMemoryResponse:
        """获取记忆数据，支持多种记忆类型

        Args:
            request: FetchMemoryRequest 包含查询参数

        Returns:
            FetchMemoryResponse 包含查询结果
        """
        logger.debug(
            f"fetch_mem called with request: user_id={request.user_id}, memory_type={request.memory_type}"
        )

        # repository 支持 MemoryType.MULTIPLE 类型，默认就是corememory
        response = await self._fetch_service.find_by_user_id(
            user_id=request.user_id,
            memory_type=request.memory_type,
            version_range=request.version_range,
            limit=request.limit,
        )

        # 注意：response.metadata 已经通过 _get_employee_metadata 包含了完整的员工信息
        # 包括 source, user_id, memory_type, limit, email, phone, full_name
        # 这里不需要再次更新，因为 fetch_mem_service 已经提供了正确的信息

        logger.debug(
            f"fetch_mem returned {len(response.memories)} memories for user {request.user_id}"
        )
        return response

    # 基于retrieve_method的记忆读取，包括静态与动态记忆
    @trace_logger(operation_name="agents 记忆检索")
    async def retrieve_mem(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> RetrieveMemoryResponse:
        """检索记忆数据，根据 retrieve_method 分发到不同的检索方法

        Args:
            retrieve_mem_request: RetrieveMemRequest 包含检索参数

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 验证请求参数
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            # 根据 retrieve_method 分发到不同的检索方法
            from .memory_models import RetrieveMethod

            retrieve_method = retrieve_mem_request.retrieve_method

            logger.info(
                f"retrieve_mem 分发请求: user_id={retrieve_mem_request.user_id}, "
                f"retrieve_method={retrieve_method}, query={retrieve_mem_request.query}"
            )

            # 根据检索方法分发
            if retrieve_method == RetrieveMethod.KEYWORD:
                # 关键词检索
                return await self.retrieve_mem_keyword(retrieve_mem_request)
            elif retrieve_method == RetrieveMethod.VECTOR:
                # 向量检索
                return await self.retrieve_mem_vector(retrieve_mem_request)
            elif retrieve_method == RetrieveMethod.HYBRID:
                # 混合检索
                return await self.retrieve_mem_hybrid(retrieve_mem_request)
            else:
                raise ValueError(f"不支持的检索方法: {retrieve_method}")

        except Exception as e:
            logger.error(f"Error in retrieve_mem: {e}", exc_info=True)
            return RetrieveMemoryResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
                metadata=Metadata(
                    source="retrieve_mem_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve",
                ),
            )

    # 关键词检索方法（原来的 retrieve_mem 逻辑）
    @trace_logger(operation_name="agents 关键词记忆检索")
    async def retrieve_mem_keyword(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> RetrieveMemoryResponse:
        """基于关键词的记忆检索（原 retrieve_mem 的实现）

        Args:
            retrieve_mem_request: RetrieveMemRequest 包含检索参数

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 从 Request 中获取参数
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_keyword"
                )

            search_results = await self.get_keyword_search_results(retrieve_mem_request)

            if not search_results:
                logger.warning(
                    f"关键词检索未找到结果: user_id={retrieve_mem_request.user_id}, query={retrieve_mem_request.query}"
                )
                return RetrieveMemoryResponse(
                    memories=[],
                    original_data=[],
                    scores=[],
                    importance_scores=[],
                    total_count=0,
                    has_more=False,
                    query_metadata=Metadata(
                        source="episodic_memory_es_repository",
                        user_id=retrieve_mem_request.user_id,
                        memory_type="retrieve_keyword",
                    ),
                    metadata=Metadata(
                        source="episodic_memory_es_repository",
                        user_id=retrieve_mem_request.user_id,
                        memory_type="retrieve_keyword",
                    ),
                )

            # 使用通用的分组处理策略
            memories, scores, importance_scores, original_data, total_count = (
                await self.group_by_groupid_stratagy(search_results, source_type="es")
            )

            logger.debug(
                f"EpisodicMemoryEsRepository multi_search returned {len(memories)} groups for query: {retrieve_mem_request.query}"
            )

            return RetrieveMemoryResponse(
                memories=memories,
                scores=scores,
                importance_scores=importance_scores,
                original_data=original_data,
                total_count=total_count,
                has_more=False,
                query_metadata=Metadata(
                    source="episodic_memory_es_repository",
                    user_id=retrieve_mem_request.user_id,
                    memory_type="retrieve_keyword",
                ),
                metadata=Metadata(
                    source="episodic_memory_es_repository",
                    user_id=retrieve_mem_request.user_id,
                    memory_type="retrieve_keyword",
                ),
            )

        except Exception as e:
            logger.error(f"Error in retrieve_mem_keyword: {e}", exc_info=True)
            return RetrieveMemoryResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_keyword_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve_keyword",
                ),
                metadata=Metadata(
                    source="retrieve_mem_keyword_service",
                    user_id=(
                        retrieve_mem_request.user_id if retrieve_mem_request else ""
                    ),
                    memory_type="retrieve_keyword",
                ),
            )

    async def get_keyword_search_results(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> Dict[str, Any]:
        try:
            # 从 Request 中获取参数
            if not retrieve_mem_request:
                raise ValueError("retrieve_mem_request is required for retrieve_mem")

            top_k = retrieve_mem_request.top_k
            query = retrieve_mem_request.query
            user_id = retrieve_mem_request.user_id
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            # 获取 EpisodicMemoryEsRepository 实例
            es_repo = get_bean_by_type(EpisodicMemoryEsRepository)

            # 将查询字符串转换为搜索词列表
            # 使用jieba进行搜索模式分词，然后过滤停用词
            if query:
                raw_words = list(jieba.cut_for_search(query))
                query_words = filter_stopwords(raw_words, min_length=2)
            else:
                query_words = []

            logger.debug(f"query_words: {query_words}")

            # 构建时间范围过滤条件，处理 None 值
            date_range = {}
            if start_time is not None:
                date_range["gte"] = start_time
            if end_time is not None:
                date_range["lte"] = end_time

            # 调用 multi_search 方法，支持按 memory_sub_type 过滤
            search_results = await es_repo.multi_search(
                query=query_words,
                user_id=user_id,
                event_type=retrieve_mem_request.memory_sub_type,  # 按记忆子类型过滤
                size=top_k,
                from_=0,
                date_range=date_range,
            )
            return search_results
        except Exception as e:
            logger.error(f"Error in get_keyword_search_results: {e}")
            return {}

    # 基于向量的记忆检索
    @trace_logger(operation_name="agents 向量记忆检索")
    async def retrieve_mem_vector(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> RetrieveMemoryResponse:
        """基于向量相似性的记忆检索

        Args:
            request: Request 包含检索参数，包括 query 和 retrieve_mem_request

        Returns:
            RetrieveMemResponse 包含检索结果
        """
        try:
            # 从 Request 中获取参数
            logger.debug(
                f"retrieve_mem_vector called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_vector"
                )

            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_vector")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 获取向量化服务
            vectorize_service = get_vectorize_service()

            # 将查询文本转换为向量
            logger.debug(f"开始向量化查询文本: {query}")
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # 转换为列表格式
            logger.debug(f"查询文本向量化完成，向量维度: {len(query_vector_list)}")

            # 根据 memory_sub_type 选择对应的 Milvus Repository
            # - "semantic_memory": 使用 SemanticMemoryMilvusRepository
            # - "event_log": 使用 EventLogMilvusRepository
            # - 其他（episode/None）: 使用 EpisodicMemoryMilvusRepository（默认）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                from infra.adapters.out.search.repository.semantic_memory_milvus_repository import (
                    SemanticMemoryMilvusRepository,
                )
                milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
            elif retrieve_mem_request.memory_sub_type == "event_log":
                from infra.adapters.out.search.repository.event_log_milvus_repository import (
                    EventLogMilvusRepository,
                )
                milvus_repo = get_bean_by_type(EventLogMilvusRepository)
            else:
                milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

            # 处理时间范围过滤条件
            start_time_dt = None
            end_time_dt = None
            semantic_start_dt = None
            semantic_end_dt = None
            current_time_dt = None

            if start_time is not None:
                if isinstance(start_time, str):
                    # 如果是日期格式 "2024-01-01"，转换为当天的开始时间
                    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d")
                else:
                    start_time_dt = start_time

            if end_time is not None:
                if isinstance(end_time, str):
                    # 如果是日期格式 "2024-12-31"，转换为当天的结束时间
                    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d")
                    # 设置为当天的23:59:59，确保包含整天
                    end_time_dt = end_time_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_time_dt = end_time

            # 处理语义记忆时间范围（仅对 semantic_memory 有效）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                if retrieve_mem_request.semantic_start_time:
                    semantic_start_dt = datetime.strptime(retrieve_mem_request.semantic_start_time, "%Y-%m-%d")
                if retrieve_mem_request.semantic_end_time:
                    semantic_end_dt = datetime.strptime(retrieve_mem_request.semantic_end_time, "%Y-%m-%d")
                if retrieve_mem_request.current_time:
                    current_time_dt = datetime.strptime(retrieve_mem_request.current_time, "%Y-%m-%d")

            # 调用 Milvus 的向量搜索（根据记忆类型传递不同的参数）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                # 语义记忆：支持时间范围和有效期过滤，支持 radius 参数
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=semantic_start_dt,
                    end_time=semantic_end_dt,
                    current_time=current_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                    radius=retrieve_mem_request.radius,  # 从请求对象中获取相似度阈值参数
                )
            else:
                # 情景记忆和事件日志：使用 timestamp 过滤，支持 radius 参数
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                    radius=retrieve_mem_request.radius,  # 从请求对象中获取相似度阈值参数
                )

            logger.debug(f"Milvus向量搜索返回 {len(search_results)} 条结果")

            # 使用通用的分组处理策略
            memories, scores, importance_scores, original_data, total_count = (
                await self.group_by_groupid_stratagy(
                    search_results, source_type="milvus"
                )
            )

            logger.debug(
                f"EpisodicMemoryMilvusRepository vector_search returned {len(memories)} groups for query: {query}"
            )

            return RetrieveMemoryResponse(
                memories=memories,
                scores=scores,
                importance_scores=importance_scores,
                original_data=original_data,
                total_count=total_count,
                has_more=False,
                query_metadata=Metadata(
                    source="episodic_memory_milvus_repository",
                    user_id=user_id,
                    memory_type="retrieve_vector",
                ),
                metadata=Metadata(
                    source="episodic_memory_milvus_repository",
                    user_id=user_id,
                    memory_type="retrieve_vector",
                ),
            )

        except Exception as e:
            logger.error(f"Error in retrieve_mem_vector: {e}")
            return RetrieveMemoryResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_vector_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_vector",
                ),
                metadata=Metadata(
                    source="retrieve_mem_vector_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_vector",
                ),
            )

    async def get_vector_search_results(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> Dict[str, Any]:
        try:
            # 从 Request 中获取参数
            logger.debug(
                f"get_vector_search_results called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for get_vector_search_results"
                )
            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_vector")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_vector called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 获取向量化服务
            vectorize_service = get_vectorize_service()

            # 将查询文本转换为向量
            logger.debug(f"开始向量化查询文本: {query}")
            query_vector = await vectorize_service.get_embedding(query)
            query_vector_list = query_vector.tolist()  # 转换为列表格式
            logger.debug(f"查询文本向量化完成，向量维度: {len(query_vector_list)}")

            # 根据 memory_sub_type 选择对应的 Milvus Repository
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                from infra.adapters.out.search.repository.semantic_memory_milvus_repository import (
                    SemanticMemoryMilvusRepository,
                )
                milvus_repo = get_bean_by_type(SemanticMemoryMilvusRepository)
            elif retrieve_mem_request.memory_sub_type == "event_log":
                from infra.adapters.out.search.repository.event_log_milvus_repository import (
                    EventLogMilvusRepository,
                )
                milvus_repo = get_bean_by_type(EventLogMilvusRepository)
            else:
                milvus_repo = get_bean_by_type(EpisodicMemoryMilvusRepository)

            # 处理时间范围过滤条件
            start_time_dt = None
            end_time_dt = None
            semantic_start_dt = None
            semantic_end_dt = None
            current_time_dt = None

            if start_time is not None:
                if isinstance(start_time, str):
                    # 如果是日期格式 "2024-01-01"，转换为当天的开始时间
                    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d")
                else:
                    start_time_dt = start_time

            if end_time is not None:
                if isinstance(end_time, str):
                    # 如果是日期格式 "2024-12-31"，转换为当天的结束时间
                    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d")
                    # 设置为当天的23:59:59，确保包含整天
                    end_time_dt = end_time_dt.replace(hour=23, minute=59, second=59)
                else:
                    end_time_dt = end_time

            # 处理语义记忆时间范围（仅对 semantic_memory 有效）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                if retrieve_mem_request.semantic_start_time:
                    semantic_start_dt = datetime.strptime(retrieve_mem_request.semantic_start_time, "%Y-%m-%d")
                if retrieve_mem_request.semantic_end_time:
                    semantic_end_dt = datetime.strptime(retrieve_mem_request.semantic_end_time, "%Y-%m-%d")
                if retrieve_mem_request.current_time:
                    current_time_dt = datetime.strptime(retrieve_mem_request.current_time, "%Y-%m-%d")

            # 调用 Milvus 的向量搜索（根据记忆类型传递不同的参数）
            if retrieve_mem_request.memory_sub_type == "semantic_memory":
                # 语义记忆：支持时间范围和有效期过滤，支持 radius 参数
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=semantic_start_dt,
                    end_time=semantic_end_dt,
                    current_time=current_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                    radius=retrieve_mem_request.radius,  # 从请求对象中获取相似度阈值参数
                )
            else:
                # 情景记忆和事件日志：使用 timestamp 过滤，支持 radius 参数
                search_results = await milvus_repo.vector_search(
                    query_vector=query_vector_list,
                    user_id=user_id,
                    start_time=start_time_dt,
                    end_time=end_time_dt,
                    limit=top_k,
                    score_threshold=0.0,
                    radius=retrieve_mem_request.radius,  # 从请求对象中获取相似度阈值参数
                )
            return search_results
        except Exception as e:
            logger.error(f"Error in get_vector_search_results: {e}")
            return {}

    # 混合记忆检索
    @trace_logger(operation_name="agents 混合记忆检索")
    async def retrieve_mem_hybrid(
        self, retrieve_mem_request: 'RetrieveMemoryRequest'
    ) -> RetrieveMemoryResponse:
        """基于关键词和向量的混合记忆检索

        Args:
            retrieve_mem_request: RetrieveMemoryRequest 包含检索参数

        Returns:
            RetrieveMemoryResponse 包含混合检索结果
        """
        try:
            logger.debug(
                f"retrieve_mem_hybrid called with retrieve_mem_request: {retrieve_mem_request}"
            )
            if not retrieve_mem_request:
                raise ValueError(
                    "retrieve_mem_request is required for retrieve_mem_hybrid"
                )

            query = retrieve_mem_request.query
            if not query:
                raise ValueError("query is required for retrieve_mem_hybrid")

            user_id = retrieve_mem_request.user_id
            top_k = retrieve_mem_request.top_k
            start_time = retrieve_mem_request.start_time
            end_time = retrieve_mem_request.end_time

            logger.debug(
                f"retrieve_mem_hybrid called with query: {query}, user_id: {user_id}, top_k: {top_k}"
            )

            # 创建关键词检索请求
            keyword_request = RetrieveMemoryRequest(
                user_id=user_id,
                memory_types=retrieve_mem_request.memory_types,
                top_k=top_k,
                filters=retrieve_mem_request.filters,
                include_metadata=retrieve_mem_request.include_metadata,
                start_time=start_time,
                end_time=end_time,
                query=query,
            )

            # 创建向量检索请求
            vector_request = RetrieveMemoryRequest(
                user_id=user_id,
                memory_types=retrieve_mem_request.memory_types,
                top_k=top_k,
                filters=retrieve_mem_request.filters,
                include_metadata=retrieve_mem_request.include_metadata,
                start_time=start_time,
                end_time=end_time,
                query=query,
            )

            # 并行执行两种检索，获取原始搜索结果
            keyword_search_results = await self.get_keyword_search_results(
                keyword_request
            )
            vector_search_results = await self.get_vector_search_results(vector_request)

            logger.debug(f"关键词检索返回 {len(keyword_search_results)} 条原始结果")
            logger.debug(f"向量检索返回 {len(vector_search_results)} 条原始结果")

            # 合并原始搜索结果并进行rerank
            hybrid_result = await self._merge_and_rerank_search_results(
                keyword_search_results, vector_search_results, top_k, user_id, query
            )

            logger.debug(f"混合检索最终返回 {len(hybrid_result.memories)} 个群组")

            return hybrid_result

        except Exception as e:
            logger.error(f"Error in retrieve_mem_hybrid: {e}")
            return RetrieveMemoryResponse(
                memories=[],
                original_data=[],
                scores=[],
                importance_scores=[],
                total_count=0,
                has_more=False,
                query_metadata=Metadata(
                    source="retrieve_mem_hybrid_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_hybrid",
                ),
                metadata=Metadata(
                    source="retrieve_mem_hybrid_service",
                    user_id=user_id if 'user_id' in locals() else "",
                    memory_type="retrieve_hybrid",
                ),
            )

    def _extract_score_from_hit(self, hit: Dict[str, Any]) -> float:
        """从hit中提取得分

        Args:
            hit: 搜索结果hit

        Returns:
            得分
        """
        if '_score' in hit:
            return hit['_score']
        elif 'score' in hit:
            return hit['score']
        return 1.0

    async def _merge_and_rerank_search_results(
        self,
        keyword_search_results: List[Dict[str, Any]],
        vector_search_results: List[Dict[str, Any]],
        top_k: int,
        user_id: str,
        query: str,
    ) -> RetrieveMemoryResponse:
        """合并关键词和向量检索的原始搜索结果，并进行重新排序

        Args:
            keyword_search_results: 关键词检索的原始搜索结果
            vector_search_results: 向量检索的原始搜索结果
            top_k: 返回的最大群组数量
            user_id: 用户ID
            query: 查询文本

        Returns:
            RetrieveMemoryResponse: 合并和重新排序后的结果
        """
        # 提取搜索结果
        keyword_hits = keyword_search_results
        vector_hits = vector_search_results

        logger.debug(f"关键词检索原始结果: {len(keyword_hits)} 条")
        logger.debug(f"向量检索原始结果: {len(vector_hits)} 条")

        # 合并所有搜索结果并标记来源
        all_hits = []

        # 添加关键词检索结果，标记来源
        for hit in keyword_hits:
            hit_copy = hit.copy()
            hit_copy['_search_source'] = 'keyword'
            all_hits.append(hit_copy)

        # 添加向量检索结果，标记来源
        for hit in vector_hits:
            hit_copy = hit.copy()
            hit_copy['_search_source'] = 'vector'
            all_hits.append(hit_copy)

        logger.debug(f"合并后总结果数: {len(all_hits)} 条")

        # 使用rerank服务进行重排序
        try:
            rerank_service = get_rerank_service()
            reranked_hits = await rerank_service._rerank_all_hits(
                query, all_hits, top_k
            )

            logger.debug(f"使用rerank服务后取top_k结果数: {len(reranked_hits)} 条")

        except Exception as e:
            logger.error(f"使用rerank服务失败，回退到简单排序: {e}")
            # 如果rerank失败，回退到简单的得分排序
            reranked_hits = sorted(
                all_hits, key=self._extract_score_from_hit, reverse=True
            )[:top_k]

        # 对rerank后的结果进行分组处理
        memories, scores, importance_scores, original_data, total_count = (
            await self.group_by_groupid_stratagy(reranked_hits, source_type="hybrid")
        )

        # 构建最终结果
        return RetrieveMemoryResponse(
            memories=memories,
            scores=scores,
            importance_scores=importance_scores,
            original_data=original_data,
            total_count=total_count,
            has_more=False,
            query_metadata=Metadata(
                source="hybrid_retrieval",
                user_id=user_id,
                memory_type="retrieve_hybrid",
            ),
            metadata=Metadata(
                source="hybrid_retrieval",
                user_id=user_id,
                memory_type="retrieve_hybrid",
            ),
        )

    async def group_by_groupid_stratagy(
        self, search_results: List[Dict[str, Any]], source_type: str = "milvus"
    ) -> tuple:
        """通用的搜索结果分组处理策略

        Args:
            search_results: 搜索结果列表
            source_type: 数据源类型，支持 "es" 或 "milvus"

        Returns:
            tuple: (memories, scores, importance_scores, original_data, total_count)
        """
        memories_by_group = (
            {}
        )  # {group_id: {'memories': [Memory], 'scores': [float], 'importance_evidence': dict}}
        original_data_by_group = {}

        for hit in search_results:
            # 根据数据源类型提取数据
            if source_type == "es":
                # ES 搜索结果格式
                source = hit.get('_source', {})
                score = hit.get('_score', 1.0)
                user_id = source.get('user_id', '')
                group_id = source.get('group_id', '')
                timestamp_raw = source.get('timestamp', '')
                narrative = source.get('narrative', '')
                memunit_id_list = source.get('memunit_id_list', [])
                subject = source.get('subject', '')
                summary = source.get('summary', '')
                participants = source.get('participants', [])
                hit_id = source.get('episode_id', '')
                search_source = hit.get('_search_source', 'keyword')  # 默认为关键词检索
            elif source_type == "hybrid":
                # 混合检索结果格式，需要根据_search_source字段判断
                search_source = hit.get('_search_source', 'unknown')
                if search_source == 'keyword':
                    # 关键词检索结果格式
                    source = hit.get('_source', {})
                    score = hit.get('_score', 1.0)
                    user_id = source.get('user_id', '')
                    group_id = source.get('group_id', '')
                    timestamp_raw = source.get('timestamp', '')
                    narrative = source.get('narrative', '')
                    memunit_id_list = source.get('memunit_id_list', [])
                    subject = source.get('subject', '')
                    summary = source.get('summary', '')
                    participants = source.get('participants', [])
                    hit_id = source.get('episode_id', '')
                else:
                    # 向量检索结果格式
                    hit_id = hit.get('id', '')
                    score = hit.get('score', 1.0)
                    user_id = hit.get('user_id', '')
                    group_id = hit.get('group_id', '')
                    timestamp_raw = hit.get('timestamp')
                    narrative = hit.get('narrative', '')
                    metadata = hit.get('metadata', {})
                    memunit_id_list = metadata.get('memunit_id_list', [])
                    subject = metadata.get('subject', '')
                    summary = metadata.get('summary', '')
                    participants = metadata.get('participants', [])
            else:
                # Milvus 搜索结果格式
                hit_id = hit.get('id', '')
                score = hit.get('score', 1.0)
                user_id = hit.get('user_id', '')
                group_id = hit.get('group_id', '')
                timestamp_raw = hit.get('timestamp')
                narrative = hit.get('narrative', '')
                metadata = hit.get('metadata', {})
                memunit_id_list = metadata.get('memunit_id_list', [])
                subject = metadata.get('subject', '')
                summary = metadata.get('summary', '')
                participants = metadata.get('participants', [])
                search_source = 'vector'  # 默认为向量检索

            # 处理时间戳
            if timestamp_raw:
                if isinstance(timestamp_raw, datetime):
                    timestamp = timestamp_raw.replace(tzinfo=None)
                elif isinstance(timestamp_raw, (int, float)):
                    try:
                        timestamp = datetime.fromtimestamp(timestamp_raw)
                    except Exception as e:
                        logger.warning(
                            f"timestamp为数字但转换失败: {timestamp_raw}, error: {e}"
                        )
                        timestamp = datetime.now().replace(tzinfo=None)
                elif isinstance(timestamp_raw, str):
                    try:
                        timestamp = from_iso_format(timestamp_raw).replace(tzinfo=None)
                    except Exception as e:
                        logger.warning(
                            f"timestamp格式转换失败: {timestamp_raw}, error: {e}"
                        )
                        timestamp = datetime.now().replace(tzinfo=None)
                else:
                    logger.warning(
                        f"未知类型的timestamp_raw: {type(timestamp_raw)}, 使用当前时间"
                    )
                    timestamp = datetime.now().replace(tzinfo=None)
            else:
                timestamp = datetime.now().replace(tzinfo=None)

            # 获取 memunit 数据
            memunits = []
            if memunit_id_list:
                memunit_repo = get_bean_by_type(MemUnitRawRepository)
                for unit_id in memunit_id_list:
                    memunit = await memunit_repo.get_by_unit_id(unit_id)
                    if memunit:
                        memunits.append(memunit)
                    else:
                        logger.warning(f"未找到 memunit: unit_id={unit_id}")
                        continue

            # 为每个 memunit 添加原始数据
            for memunit in memunits:
                if group_id not in original_data_by_group:
                    original_data_by_group[group_id] = []
                original_data_by_group[group_id].append(memunit.original_data)

            # 创建 Memory 对象
            memory = Memory(
                memory_type="episode_summary",  # 情景记忆类型
                user_id=user_id,
                timestamp=timestamp,
                memunit_id_list=memunit_id_list or [hit_id],
                subject=subject,
                summary=summary,
                episode=episode,
                group_id=group_id,
                participants=participants,
            )

            # 添加搜索来源信息到 extend 字段
            if not hasattr(memory, 'extend') or memory.extend is None:
                memory.extend = {}
            memory.extend['_search_source'] = search_source

            # 从 user_profiles 中读取 group_importance_evidence
            group_importance_evidence = None
            if user_id and group_id:
                try:
                    user_profile = await UserProfile.find_one(
                        UserProfile.user_id == user_id,
                        UserProfile.group_id == group_id,
                        sort=[("version", -1)],
                    )

                    if user_profile:
                        group_importance_evidence = (
                            user_profile.profile_data.get("group_importance_evidence")
                            if isinstance(user_profile.profile_data, dict)
                            else None
                        )
                        if group_importance_evidence:
                            if not hasattr(memory, 'extend') or memory.extend is None:
                                memory.extend = {}
                            memory.extend['group_importance_evidence'] = (
                                group_importance_evidence
                            )
                            logger.debug(
                                "为memory添加group_importance_evidence: user_id=%s, group_id=%s",
                                user_id,
                                group_id,
                            )
                except Exception as e:
                    logger.warning(
                        "读取 user_profiles 失败: user_id=%s, group_id=%s, error=%s",
                        user_id,
                        group_id,
                        e,
                    )

            # 按group_id分组
            if group_id not in memories_by_group:
                memories_by_group[group_id] = {
                    'memories': [],
                    'scores': [],
                    'importance_evidence': group_importance_evidence,
                }

            memories_by_group[group_id]['memories'].append(memory)
            memories_by_group[group_id]['scores'].append(score)  # 保存原始得分
            # 更新group_importance_evidence（如果当前memory有更新的证据）
            if group_importance_evidence:
                memories_by_group[group_id][
                    'importance_evidence'
                ] = group_importance_evidence

        def calculate_importance_score(importance_evidence):
            """计算群组重要性得分"""
            if not importance_evidence or not isinstance(importance_evidence, dict):
                return 0.0

            evidence_list = importance_evidence.get('evidence_list', [])
            if not evidence_list:
                return 0.0

            total_speak_count = 0
            total_refer_count = 0
            total_conversation_count = 0

            for evidence in evidence_list:
                if isinstance(evidence, dict):
                    total_speak_count += evidence.get('speak_count', 0)
                    total_refer_count += evidence.get('refer_count', 0)
                    total_conversation_count += evidence.get('conversation_count', 0)

            if total_conversation_count == 0:
                return 0.0

            return (total_speak_count + total_refer_count) / total_conversation_count

        # 为每个group内的memories按时间戳排序，并计算重要性得分
        group_scores = []
        for group_id, group_data in memories_by_group.items():
            # 按时间戳排序memories
            group_data['memories'].sort(
                key=lambda m: m.timestamp if m.timestamp else ''
            )

            # 计算重要性得分
            importance_score = calculate_importance_score(
                group_data['importance_evidence']
            )
            group_scores.append((group_id, importance_score))

        # 按重要性得分排序groups
        group_scores.sort(key=lambda x: x[1], reverse=True)

        # 构建最终结果
        memories = []
        scores = []
        importance_scores = []
        original_data = []
        for group_id, importance_score in group_scores:
            group_data = memories_by_group[group_id]
            group_memories = group_data['memories']
            group_scores_list = group_data['scores']
            group_original_data = original_data_by_group.get(group_id, [])
            memories.append({group_id: group_memories})
            # scores结构与memories保持一致：List[Dict[str, List[float]]]
            scores.append({group_id: group_scores_list})
            # original_data结构与memories保持一致：List[Dict[str, List[Dict[str, Any]]]]
            original_data.append({group_id: group_original_data})
            importance_scores.append(importance_score)

        total_count = sum(
            len(group_data['memories']) for group_data in memories_by_group.values()
        )
        return memories, scores, importance_scores, original_data, total_count
    
    # --------- Lightweight 检索（Embedding + BM25 + RRF）---------
    # Delegates to retrieval.online.lightweight
    @trace_logger(operation_name="agents 轻量级检索")
    async def retrieve_lightweight(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        time_range_days: int = 365,
        top_k: int = 20,
        retrieval_mode: str = "rrf",  # "embedding" | "bm25" | "rrf"
        data_source: str = "episode",  # "episode" | "event_log" | "semantic_memory" | "profile"
        memory_scope: str = "all",  # "all" | "personal" | "group"
        current_time: Optional[datetime] = None,  # 当前时间，用于过滤有效期内的语义记忆
        radius: Optional[float] = None,  # COSINE 相似度阈值
    ) -> Dict[str, Any]:
        """
        轻量级记忆检索（统一使用 Milvus/ES 检索）

        Delegates to retrieval.online.lightweight.retrieve_lightweight
        """
        from retrieval.online.lightweight import retrieve_lightweight as _retrieve_lightweight

        return await _retrieve_lightweight(
            query=query,
            user_id=user_id,
            group_id=group_id,
            time_range_days=time_range_days,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            data_source=data_source,
            memory_scope=memory_scope,
            current_time=current_time,
            radius=radius,
        )
    
    async def _retrieve_from_vector_stores(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        top_k: int = 20,
        retrieval_mode: str = "rrf",
        data_source: str = "memunit",
        start_time: float = None,
        memory_scope: str = "all",
        current_time: Optional[datetime] = None,
        participant_user_id: Optional[str] = None,
        radius: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Delegates to retrieval.online.vector_store.retrieve_from_vector_stores"""
        from retrieval.online.vector_store import retrieve_from_vector_stores

        return await retrieve_from_vector_stores(
            query=query,
            user_id=user_id,
            group_id=group_id,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            data_source=data_source,
            start_time=start_time,
            memory_scope=memory_scope,
            current_time=current_time,
            participant_user_id=participant_user_id,
            radius=radius,
        )

    async def _retrieve_profile_memories(
        self,
        user_id: str,
        group_id: str,
        top_k: int,
        start_time: float,
    ) -> Dict[str, Any]:
        """从 user_profiles 集合直接读取用户画像"""
        doc = await UserProfile.find_one(
            UserProfile.user_id == user_id,
            UserProfile.group_id == group_id,
            sort=[("version", -1)],
        )

        memories: List[Dict[str, Any]] = []
        if doc:
            memories.append(
                {
                    "user_id": doc.user_id,
                    "group_id": doc.group_id,
                    "profile": doc.profile_data,
                    "scenario": doc.scenario,
                    "confidence": doc.confidence,
                    "version": doc.version,
                    "cluster_ids": doc.cluster_ids,
                    "memunit_count": doc.memunit_count,
                    "last_updated_cluster": doc.last_updated_cluster,
                    "updated_at": doc.updated_at.isoformat()
                    if doc.updated_at
                    else None,
                }
            )

        metadata = {
            "retrieval_mode": "direct",
            "data_source": "profile",
            "profile_count": len(memories),
            "total_latency_ms": (time.time() - start_time) * 1000,
        }

        return {
            "memories": memories[:top_k],
            "count": len(memories[:top_k]),
            "metadata": metadata,
        }

    @staticmethod
    def _format_datetime_field(value: Any) -> Optional[str]:
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @staticmethod
    def _parse_datetime_value(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str) and value:
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    return None
        return None

    def _filter_semantic_memories_by_time(
        self,
        memories: List[Dict[str, Any]],
        data_source: str,
        current_time: Optional[datetime],
    ) -> List[Dict[str, Any]]:
        if data_source != "semantic_memory" or not current_time:
            return memories
        current_dt = (
            current_time
            if isinstance(current_time, datetime)
            else self._parse_datetime_value(current_time)
        )
        if current_dt is None:
            return memories

        filtered = []
        for memory in memories:
            start_dt = self._parse_datetime_value(memory.get("start_time"))
            end_dt = self._parse_datetime_value(memory.get("end_time"))

            if start_dt and start_dt > current_dt:
                continue
            if end_dt and end_dt < current_dt:
                continue
            filtered.append(memory)
        return filtered
    
    # --------- Agentic 检索（LLM 引导的多轮检索）---------
    # Delegates to retrieval.online.agentic
    @trace_logger(operation_name="agents Agentic检索")
    async def retrieve_agentic(
        self,
        query: str,
        user_id: str = None,
        group_id: str = None,
        time_range_days: int = 365,
        top_k: int = 20,
        llm_provider = None,
        agentic_config = None,
    ) -> Dict[str, Any]:
        """Agentic 检索：LLM 引导的多轮智能检索

        Delegates to retrieval.online.agentic.retrieve_agentic
        """
        from retrieval.online.agentic import retrieve_agentic as _retrieve_agentic

        return await _retrieve_agentic(
            query=query,
            user_id=user_id,
            group_id=group_id,
            time_range_days=time_range_days,
            top_k=top_k,
            llm_provider=llm_provider,
            agentic_config=agentic_config,
        )

