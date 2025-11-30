"""
Cluster 阶段 (Stage 1.5)

负责对 MemUnits 进行 LLM 驱动的事件聚类。
运行在 Add 阶段之后，Search 阶段之前。
"""
from pathlib import Path
from typing import List, Any, Optional
from logging import Logger

from eval.core.data_models import Conversation
from eval.adapters.base import BaseAdapter
from eval.utils.checkpoint import CheckpointManager


async def run_cluster_stage(
    adapter: BaseAdapter,
    conversations: List[Conversation],
    output_dir: Path,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
    console: Any,
    completed_stages: set,
) -> dict:
    """
    执行 Cluster 阶段（群体事件聚类）

    Args:
        adapter: 系统适配器
        conversations: 对话列表
        output_dir: 输出目录
        checkpoint_manager: 断点续传管理器
        logger: 日志器
        console: 控制台对象
        completed_stages: 已完成的阶段集合

    Returns:
        包含 cluster_indices 的字典
    """
    # 调用 adapter 的 cluster 方法（仅传递数据相关参数）
    cluster_results = await adapter.cluster(
        conversations=conversations,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
    )

    logger.info("✅ Stage 1.5 completed")

    # 保存 checkpoint
    completed_stages.add("cluster")
    if checkpoint_manager:
        checkpoint_manager.save_checkpoint(completed_stages)

    return cluster_results
