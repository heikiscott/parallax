import os
from dotenv import load_dotenv

load_dotenv()


class ExperimentConfig:
    experiment_name: str = "locomo_evaluation"
    datase_path: str = "data/locomo10.json"
    use_emb: bool = True
    use_reranker: bool = True  # 启用 Reranker
    use_agentic_retrieval: bool = True
    use_multi_query: bool = True  #  启用多查询生成
    num_conv: int = 10 #最终使用记忆条数
    
    # 🔥 新增：MemUnit 提取功能开关
    enable_semantic_extraction: bool = False  # 是否启用语义记忆提取
    enable_clustering: bool = False            # 是否启用聚类
    enable_profile_extraction: bool = False    # 是否启用 Profile 提取
    
    # 🔥 聚类配置
    cluster_similarity_threshold: float = 0.65  # 聚类相似度阈值
    cluster_max_time_gap_days: float = 7.0     # 聚类最大时间间隔（天）
    
    # 🔥 Profile 配置
    profile_scenario: str = "assistant"       # Profile 场景：group_chat 或 assistant
    profile_min_confidence: float = 0.6        # Profile 价值判别阈值
    profile_min_memunits: int = 1              # Profile 提取最小 MemUnits 数量

    # ===== 群体事件聚类配置 (Group Event Cluster) =====
    enable_group_event_cluster: bool = True   # 是否启用群体事件聚类
    group_event_cluster_config: dict = {
        "llm_provider": "openai",
        "llm_model": None,  # None 表示使用主 LLM 配置
        "llm_api_key": None,  # None 表示从环境变量读取
        "llm_base_url": None,  # None 表示从环境变量读取
        "llm_temperature": 0.0,
        "summary_update_threshold": 5,
        "max_clusters_in_prompt": 20,
        "max_members_per_cluster_in_prompt": 3,
    }

    # ===== 群体事件聚类检索配置 =====
    group_event_cluster_retrieval_config: dict = {
        "enable_group_event_cluster_retrieval": True,
        # 可选策略: insert_after_hit, append_to_end, merge_by_score, replace_rerank, cluster_rerank
        "expansion_strategy": "cluster_rerank",
        # 通用扩展参数
        "max_expansion_per_hit": 3,
        "max_total_expansion": 10,
        "expansion_budget_ratio": 0.3,
        "prefer_time_adjacent": True,
        "time_window_hours": None,
        "expansion_score_decay": 0.7,
        "deduplicate_expanded": True,
        "rerank_after_expansion": False,
        "rerank_top_n_after_expansion": 20,
        # cluster_rerank 策略专用配置
        "cluster_rerank_max_clusters": 10,          # LLM 最多选择的 Cluster 数量（LLM 自己决定选几个）
        "cluster_rerank_max_members_per_cluster": 15,  # 每个 Cluster 最多返回的 MemUnits
        "cluster_rerank_total_max_members": 30,     # 最终返回的 MemUnits 总数上限
    }

    # 🔥 检索模式选择：'agentic' 或 'lightweight'
    # - agentic: 复杂的多轮检索，LLM引导，质量高但速度慢
    # - lightweight: 快速混合检索，BM25+Embedding混排，速度快但质量略低
    retrieval_mode: str = "agentic"  # 'agentic' | 'lightweight'

    # ===== 问题分类与策略路由配置 (Question Classification & Strategy Routing) =====
    enable_question_classification: bool = True  # 是否启用问题分类
    question_classification_config: dict = {
        # 分类器类型：'rule_based' (纯正则) 或 'llm' (LLM分类)
        # ⚠️ 注意：当前仅实现了 rule_based，llm 分类器尚未集成到路由流程
        "classifier_type": "rule_based",
        # 默认策略：当分类失败时使用
        # 可选: 'gec_cluster_rerank', 'gec_insert_after_hit', 'agentic_only'
        "default_strategy": "gec_insert_after_hit",
        # 是否记录分类结果到 metadata
        "log_classification": True,
        # 策略覆盖：可针对特定问题类型强制使用某策略
        # 格式: {"question_type": "strategy"}
        # 示例: {"EVENT_TEMPORAL": "gec_cluster_rerank", "ATTRIBUTE_IDENTITY": "agentic_only"}
        "strategy_overrides": {},
    }
    
    #  检索配置
    use_hybrid_search: bool = True  # 是否使用混合检索（Embedding + BM25 + RRF）
    emb_recall_top_n: int = 40      # Embedding/混合检索召回数量
    reranker_top_n: int = 20        # Reranker 重排序返回数量
    
    # 轻量级检索参数（仅在 retrieval_mode='lightweight' 时生效）
    lightweight_bm25_top_n: int = 50   # BM25 召回数量
    lightweight_emb_top_n: int = 50    # Embedding 召回数量
    lightweight_final_top_n: int = 20  # 混排后最终返回数量
    
    # 混合检索参数（仅在 use_hybrid_search=True 时生效）
    hybrid_emb_candidates: int = 50   # Embedding 候选数量
    hybrid_bm25_candidates: int = 50  # BM25 候选数量
    hybrid_rrf_k: int = 40             # RRF 参数 k
    
    #  多查询检索参数（仅在 use_multi_query=True 时生效）
    multi_query_num: int = 3           # 期望生成的查询数量
    multi_query_top_n: int = 50        # 每个查询召回的文档数
    
    # Reranker 优化参数（高性能配置）
    reranker_batch_size: int = 20      # Reranker 批次大小
    reranker_max_retries: int = 10     # 每个批次的最大重试次数，增加以确保评测完整性
    reranker_retry_delay: float = 0.8  # 重试间隔，指数退避
    reranker_timeout: float = 60.0     # 单个批次超时时间
    reranker_fallback_threshold: float = 0.3  # 成功率低于此值时降级到原始排序
    reranker_concurrent_batches: int = 5  #  增加并发：5 个批次并发
    
    reranker_instruction: str = (
    "Determine if the passage contains specific facts, entities (names, dates, locations), "
    "or details that directly answer the question.")
    
    # 🔥 Stage4 参数：从 unit_ids 中选择 top-k 构建 context
    response_top_k: int = 20  # 从检索到的 unit_ids 中选择前 k 个构建 context
    
    llm_service: str = "openai"  # openai, vllm
    llm_config: dict = {
        "openai": {
            "llm_provider": "openai",
            "model": os.getenv("LLM_MODEL", "openai/gpt-4.1-mini"),
            "base_url": os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1"),
            "api_key": os.getenv("LLM_API_KEY"),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "16384")),
        },
        "vllm": {
            "llm_provider": "openai",
            "model": "Qwen3-30B",
            "base_url": "http://0.0.0.0:8000/v1",
            "api_key": "123",
            "temperature": 0,
            "max_tokens": 16384,
        },
    }
    
    max_retries: int = 20  # 增加OpenAI API重试次数以容忍网络波动，确保评测完整性
    max_concurrent_requests: int = 10
