"""Configuration classes for Group Event Cluster system.

This module defines configuration for:
1. GroupEventClusterConfig - LLM clustering configuration
2. ClusterRetrievalConfig - Cluster-enhanced retrieval configuration
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GroupEventClusterConfig:
    """
    Group Event Cluster configuration.

    Supports specifying LLM model and parameters via config file or code.
    """

    # === LLM Configuration ===
    llm_provider: str = "openai"
    """
    LLM provider.
    Options: "openai", "anthropic", "azure", etc.
    Default: "openai"
    """

    llm_model: str = "gpt-4o-mini"
    """
    LLM model name.
    Recommend using small models to control cost.
    Default: "gpt-4o-mini"
    """

    llm_api_key: Optional[str] = None
    """
    API Key.
    If None, read from environment variable (OPENAI_API_KEY, etc.)
    """

    llm_base_url: Optional[str] = None
    """
    Custom API Base URL.
    For proxy or private deployment.
    """

    llm_temperature: float = 0.0
    """
    LLM temperature parameter.
    Recommend 0.0 for clustering decisions to ensure consistency.
    """

    llm_max_tokens: int = 1024
    """Maximum tokens for LLM response."""

    # === Clustering Configuration ===
    summary_update_threshold: int = 5
    """
    Update Summary when Cluster member count reaches multiples of this value.
    E.g., trigger update at 5, 10, 15, ...
    """

    max_clusters_in_prompt: int = 20
    """
    Maximum number of Clusters to show in LLM prompt.
    Prevents prompt from being too long.
    """

    max_members_per_cluster_in_prompt: int = 3
    """
    Number of recent members to show per Cluster in LLM prompt.
    """

    # === Storage Configuration ===
    output_dir: Optional[Path] = None
    """
    Output directory.
    Eval scenario: results/{experiment_name}/event_clusters/
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupEventClusterConfig":
        """Create config from dictionary."""
        return cls(
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            llm_api_key=data.get("llm_api_key"),
            llm_base_url=data.get("llm_base_url"),
            llm_temperature=data.get("llm_temperature", 0.0),
            llm_max_tokens=data.get("llm_max_tokens", 1024),
            summary_update_threshold=data.get("summary_update_threshold", 5),
            max_clusters_in_prompt=data.get("max_clusters_in_prompt", 20),
            max_members_per_cluster_in_prompt=data.get("max_members_per_cluster_in_prompt", 3),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "llm_base_url": self.llm_base_url,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "summary_update_threshold": self.summary_update_threshold,
            "max_clusters_in_prompt": self.max_clusters_in_prompt,
            "max_members_per_cluster_in_prompt": self.max_members_per_cluster_in_prompt,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }


@dataclass
class GroupEventClusterRetrievalConfig:
    """
    Group Event Cluster retrieval configuration.

    Controls how clustering results are integrated into the retrieval process.
    """

    # === Basic Switch ===
    enable_group_event_cluster_retrieval: bool = True
    """Whether to enable group event cluster retrieval"""

    # === Expansion Strategy ===
    expansion_strategy: str = "insert_after_hit"
    """
    Expansion strategy, determines how expanded documents are inserted into results.

    Options:
    - "insert_after_hit": Insert Cluster members after each hit document (recommended)
    - "append_to_end": Append all expanded documents to the end of results
    - "merge_by_score": Calculate decayed scores for expanded documents, merge with original results
    - "replace_rerank": Re-Rerank the entire result set after expansion (higher cost)
    """

    # === Expansion Quantity Control ===
    max_expansion_per_hit: int = 3
    """
    Maximum Cluster members to expand per hit document.

    Design considerations:
    - Too few (1): May miss important related information
    - Too many (5+): Will dilute original retrieval result quality
    - Recommended: 2-3
    """

    max_total_expansion: int = 10
    """
    Maximum total documents to expand per retrieval.

    Design considerations:
    - Prevents a single large Cluster from dominating results
    - Works with expansion_budget_ratio
    """

    expansion_budget_ratio: float = 0.3
    """
    Expansion budget ratio = expanded documents / original results count.

    E.g., original 20 results, budget=0.3, max expansion is 6
    Actual expansion = min(max_total_expansion, original_count * expansion_budget_ratio)
    """

    # === Time Adjacency Preference ===
    prefer_time_adjacent: bool = True
    """
    Whether to prefer time-adjacent members.

    Principle: Time-adjacent MemUnits more likely contain causal relationships
    or subsequent developments.

    Example:
    - Hit mu_007 (2023-03-20)
    - Cluster members: [mu_003, mu_007, mu_015, mu_023]
    - Prefer mu_003 (previous) and mu_015 (next)
    """

    time_window_hours: Optional[int] = None
    """
    Time window limit (hours).

    If set, only expand members within this time difference.
    None means no limit.

    Example: time_window_hours=168 (one week)
    """

    # === Score Decay (for merge_by_score strategy) ===
    expansion_score_decay: float = 0.7
    """
    Score decay coefficient for expanded documents.

    Expanded document score = original hit document score × expansion_score_decay

    Design considerations:
    - Too high (0.9+): Expanded documents may rank above original results
    - Too low (0.3-): Expanded documents sink to bottom, losing purpose
    - Recommended: 0.6-0.8
    """

    # === Deduplication Control ===
    deduplicate_expanded: bool = True
    """
    Whether to deduplicate expanded results.

    Scenario: Multiple hit documents belong to same Cluster, would expand same members
    """

    # === Rerank Related ===
    rerank_after_expansion: bool = False
    """
    Whether to re-Rerank after expansion.

    Note: Increases Rerank API call cost.
    Only effective when expansion_strategy="replace_rerank"
    """

    rerank_top_n_after_expansion: int = 20
    """Top N to return after expansion Rerank"""

    # ==========================================================
    # Cluster Rerank 策略专用配置
    # 仅当 expansion_strategy="cluster_rerank" 时生效
    # ==========================================================

    cluster_rerank_max_clusters: int = 10
    """
    [Cluster Rerank] LLM 最多可以选择的 Cluster 数量上限。

    LLM 会根据查询复杂度智能决定实际选择的数量：
    - 问题很具体时：可能只选 1 个
    - 问题涉及多个事件：可能选 2-3 个
    - 问题很宽泛/比较性问题：可能选 4+ 个

    此配置是硬上限，最终返回的 MemUnits 数量由
    cluster_rerank_total_max_members 控制。

    默认值: 10
    """

    cluster_rerank_max_members_per_cluster: int = 15
    """
    [Cluster Rerank] 每个被选中的 Cluster 最多返回的 MemUnits 数量。

    - 如果 Cluster 成员数 < 此值，返回全部成员
    - 如果 Cluster 成员数 >= 此值，按时间顺序取前 N 个

    默认值: 15
    """

    cluster_rerank_total_max_members: int = 30
    """
    [Cluster Rerank] 最终返回的 MemUnits 总数上限。

    这是所有选中 Clusters 成员总和的硬上限：
    - 按 Cluster 被选中的顺序依次添加成员
    - 达到此上限后停止添加
    - 防止返回过多内容给 LLM 生成答案

    默认值: 30
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupEventClusterRetrievalConfig":
        """Create config from dictionary."""
        return cls(
            enable_group_event_cluster_retrieval=data.get("enable_group_event_cluster_retrieval", True),
            expansion_strategy=data.get("expansion_strategy", "insert_after_hit"),
            max_expansion_per_hit=data.get("max_expansion_per_hit", 2),
            max_total_expansion=data.get("max_total_expansion", 10),
            expansion_budget_ratio=data.get("expansion_budget_ratio", 0.3),
            prefer_time_adjacent=data.get("prefer_time_adjacent", True),
            time_window_hours=data.get("time_window_hours"),
            expansion_score_decay=data.get("expansion_score_decay", 0.7),
            deduplicate_expanded=data.get("deduplicate_expanded", True),
            rerank_after_expansion=data.get("rerank_after_expansion", False),
            rerank_top_n_after_expansion=data.get("rerank_top_n_after_expansion", 20),
            # Cluster Rerank 策略配置
            cluster_rerank_max_clusters=data.get("cluster_rerank_max_clusters", 10),
            cluster_rerank_max_members_per_cluster=data.get("cluster_rerank_max_members_per_cluster", 15),
            cluster_rerank_total_max_members=data.get("cluster_rerank_total_max_members", 30),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_group_event_cluster_retrieval": self.enable_group_event_cluster_retrieval,
            "expansion_strategy": self.expansion_strategy,
            "max_expansion_per_hit": self.max_expansion_per_hit,
            "max_total_expansion": self.max_total_expansion,
            "expansion_budget_ratio": self.expansion_budget_ratio,
            "prefer_time_adjacent": self.prefer_time_adjacent,
            "time_window_hours": self.time_window_hours,
            "expansion_score_decay": self.expansion_score_decay,
            "deduplicate_expanded": self.deduplicate_expanded,
            "rerank_after_expansion": self.rerank_after_expansion,
            "rerank_top_n_after_expansion": self.rerank_top_n_after_expansion,
            # Cluster Rerank 策略配置
            "cluster_rerank_max_clusters": self.cluster_rerank_max_clusters,
            "cluster_rerank_max_members_per_cluster": self.cluster_rerank_max_members_per_cluster,
            "cluster_rerank_total_max_members": self.cluster_rerank_total_max_members,
        }
