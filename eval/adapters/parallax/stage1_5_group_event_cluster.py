"""Stage 1.5: Group Event Clustering

对 MemUnits 进行 LLM 驱动的事件聚类。
运行在 Add 阶段之后，Search 阶段之前。
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from memory.group_event_cluster import (
    GroupEventClusterer,
    GroupEventClusterConfig,
    JsonClusterStorage,
)
from providers.llm.llm_provider import LLMProvider


async def run_group_event_clustering(
    conversations: List[Any],
    memunits_dir: Path,
    clusters_dir: Path,
    config: dict,
    checkpoint_manager: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    执行群体事件聚类

    Args:
        conversations: 对话列表
        memunits_dir: MemUnits 目录
        clusters_dir: 聚类结果输出目录
        config: 配置字典（包含 llm, group_event_cluster_config 等）
        checkpoint_manager: 断点续传管理器

    Returns:
        聚类结果字典，包含 cluster_indices
    """
    console = Console()
    clusters_dir.mkdir(parents=True, exist_ok=True)

    # ========== Stage 1.5: Group Event Clustering ==========
    console.print(f"\n{'='*60}", style="bold cyan")
    console.print(f"Stage 1.5: Group Event Clustering", style="bold cyan")
    console.print(f"{'='*60}", style="bold cyan")

    # 获取聚类配置
    cluster_config_dict = config.get('group_event_cluster_config', {})
    llm_cfg = config.get('llm', {})

    # 创建 GroupEventClusterConfig
    cluster_config = GroupEventClusterConfig(
        llm_provider=cluster_config_dict.get("llm_provider", "openai"),
        llm_model=cluster_config_dict.get("llm_model") or llm_cfg.get("model", "gpt-4o-mini"),
        llm_api_key=cluster_config_dict.get("llm_api_key") or llm_cfg.get("api_key"),
        llm_base_url=cluster_config_dict.get("llm_base_url") or llm_cfg.get("base_url"),
        llm_temperature=cluster_config_dict.get("llm_temperature", 0.0),
        summary_update_threshold=cluster_config_dict.get("summary_update_threshold", 5),
        max_clusters_in_prompt=cluster_config_dict.get("max_clusters_in_prompt", 20),
        max_members_per_cluster_in_prompt=cluster_config_dict.get("max_members_per_cluster_in_prompt", 3),
        output_dir=clusters_dir,
    )

    console.print(f"🤖 Cluster LLM Model: {cluster_config.llm_model}", style="cyan")

    # 创建 LLM provider
    llm_provider = LLMProvider(
        provider_type=cluster_config.llm_provider,
        model=cluster_config.llm_model,
        api_key=cluster_config.llm_api_key,
        base_url=cluster_config.llm_base_url,
        temperature=cluster_config.llm_temperature,
        max_tokens=cluster_config.llm_max_tokens,
    )

    # 创建 clusterer 和 storage
    clusterer = GroupEventClusterer(
        config=cluster_config,
        llm_provider=llm_provider,
    )
    storage = JsonClusterStorage(output_dir=clusters_dir)

    # 检查已完成的会话（断点续传）
    # 注：文件本身就是 checkpoint，无论 checkpoint_manager 是否存在都要检查
    all_conv_ids = [i for i in range(len(conversations))]
    completed_convs = load_cluster_progress(clusters_dir, all_conv_ids)

    # 过滤出待处理的会话
    pending_conv_ids = [i for i in all_conv_ids if i not in completed_convs]

    console.print(f"\n📊 总会话数: {len(conversations)}", style="bold cyan")
    console.print(f"✅ 已完成: {len(completed_convs)}", style="bold green")
    console.print(f"⏳ 待处理: {len(pending_conv_ids)}", style="bold yellow")

    # 处理结果
    cluster_indices = {}
    total_clusters = 0
    total_units = 0

    # 加载已完成的聚类索引
    for conv_idx in completed_convs:
        conv_id = f"conv_{conv_idx}"
        existing_index = await storage.load_index(conv_id)
        if existing_index:
            cluster_indices[conv_id] = existing_index
            total_clusters += len(existing_index.clusters)
            total_units += existing_index.total_units

    if len(pending_conv_ids) == 0:
        console.print(f"\n🎉 所有会话已完成聚类，跳过！", style="bold green")
    else:
        console.print(f"🚀 开始处理...\n", style="bold green")

        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        ) as progress:
            task = progress.add_task(
                "[cyan]Clustering conversations...",
                total=len(pending_conv_ids),
            )

            for conv_idx in pending_conv_ids:
                conv_id = f"conv_{conv_idx}"
                # 兼容历史与当前文件命名
                memunit_candidates = [
                    memunits_dir / f"memunit_list_conv_{conv_idx}.json",
                    memunits_dir / f"memunits_conv_{conv_idx}.json",
                ]
                memunits_path = next((p for p in memunit_candidates if p.exists()), None)

                progress.update(task, description=f"[cyan]Processing {conv_id}...")

                # 加载 MemUnits
                if memunits_path is None:
                    console.print(f"  ⚠️  {conv_id}: memunits file not found, skipping", style="yellow")
                    progress.advance(task)
                    continue

                try:
                    with open(memunits_path, "r", encoding="utf-8") as f:
                        memunits_data = json.load(f)

                    # 处理不同格式
                    if isinstance(memunits_data, list):
                        memunits = memunits_data
                    elif isinstance(memunits_data, dict):
                        memunits = memunits_data.get("memunits", list(memunits_data.values()))
                    else:
                        memunits = []

                    if not memunits:
                        console.print(f"  ⚠️  {conv_id}: no memunits found, skipping", style="yellow")
                        progress.advance(task)
                        continue

                    # 执行聚类
                    index = await clusterer.cluster_memunits(
                        memunit_list=memunits,
                        conversation_id=conv_id,
                    )

                    # 保存结果（checkpoint: 每个会话完成后立即保存）
                    await storage.save_index(conv_id, index)

                    cluster_indices[conv_id] = index
                    total_clusters += len(index.clusters)
                    total_units += index.total_units

                    stats = index.get_stats()
                    console.print(
                        f"  ✓ {conv_id}: {stats['total_clusters']} clusters, "
                        f"{stats['indexed_units']} units, avg size {stats['avg_cluster_size']:.1f}",
                        style="green"
                    )

                except Exception as e:
                    console.print(f"  ❌ {conv_id}: clustering failed - {e}", style="red")

                progress.advance(task)

        elapsed_time = time.time() - start_time
        console.print(f"\n⏱️  耗时: {elapsed_time:.2f}s", style="dim")

    # 保存汇总信息
    config_dict = cluster_config.to_dict()
    # 过滤敏感字段，避免在结果文件中泄漏凭据
    if "llm_api_key" in config_dict:
        config_dict["llm_api_key"] = "[redacted]"

    summary = {
        "total_conversations": len(conversations),
        "clustered_conversations": len(cluster_indices),
        "total_clusters": total_clusters,
        "total_units": total_units,
        "config": config_dict,
    }
    summary_path = clusters_dir / "clustering_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    console.print(f"\n📊 Clustering complete: {total_clusters} clusters, {total_units} units", style="cyan")
    console.print(f"{'='*60}\n", style="dim")

    return {"cluster_indices": cluster_indices}


def load_cluster_progress(clusters_dir: Path, all_conv_ids: list) -> set:
    """
    加载 Cluster 阶段的细粒度进度（检查哪些会话已完成）

    Args:
        clusters_dir: 聚类结果目录
        all_conv_ids: 所有会话 ID 列表

    Returns:
        已完成的会话 ID 集合（数字索引）
    """
    completed_convs = set()

    if not clusters_dir.exists():
        return completed_convs

    for conv_idx in all_conv_ids:
        conv_id = f"conv_{conv_idx}"
        # JsonClusterStorage 的文件命名格式
        cluster_file = clusters_dir / f"{conv_id}.json"

        if cluster_file.exists():
            try:
                with open(cluster_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 验证文件有效性
                    if data and "clusters" in data:
                        completed_convs.add(conv_idx)
            except Exception:
                pass  # 文件损坏，将重新处理

    return completed_convs
