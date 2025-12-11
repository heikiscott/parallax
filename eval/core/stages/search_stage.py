"""
Search 阶段

负责检索相关记忆。
"""
import asyncio
import json
from pathlib import Path
from typing import List, Any, Optional, Dict, Set, Tuple
from logging import Logger
from tqdm import tqdm

from eval.core.data_models import QAPair, SearchResult
from eval.adapters.base import BaseAdapter
from eval.utils.checkpoint import CheckpointManager
from core.observation.logger import set_activity_id


def _build_evidence_to_cluster_mapping(
    output_dir: Path,
    conv_index: str,
    logger: Logger,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, List[str]]]:
    """
    构建 evidence (dia_id) → MemUnit → Cluster 的映射。

    Args:
        output_dir: 输出目录
        conv_index: 对话索引（例如 "0"）
        logger: 日志器

    Returns:
        Tuple of:
        - dia_id_to_unit_id: dia_id → unit_id 映射
        - unit_id_to_cluster_id: unit_id → cluster_id 映射
        - cluster_id_to_unit_ids: cluster_id → [unit_id] 映射
    """
    dia_id_to_unit_id: Dict[str, str] = {}
    unit_id_to_cluster_id: Dict[str, str] = {}
    cluster_id_to_unit_ids: Dict[str, List[str]] = {}

    # Step 1: 从 memunit 文件建立 dia_id → unit_id 映射
    memunit_file = output_dir / "memunits" / f"memunit_list_conv_{conv_index}.json"
    if memunit_file.exists():
        try:
            with open(memunit_file, "r", encoding="utf-8") as f:
                memunits = json.load(f)
            for memunit in memunits:
                unit_id = memunit.get("unit_id", "")
                original_data = memunit.get("original_data", [])
                for msg in original_data:
                    dia_id = msg.get("dia_id")
                    if dia_id:
                        dia_id_to_unit_id[dia_id] = unit_id
            logger.debug(f"  Loaded {len(dia_id_to_unit_id)} dia_id mappings from {memunit_file.name}")
        except Exception as e:
            logger.warning(f"  Failed to load memunit file: {e}")
    else:
        logger.warning(f"  Memunit file not found: {memunit_file}")

    # Step 2: 从 cluster index 建立 unit_id → cluster_id 映射
    cluster_file = output_dir / "event_clusters" / f"conv_{conv_index}.json"
    if cluster_file.exists():
        try:
            with open(cluster_file, "r", encoding="utf-8") as f:
                cluster_data = json.load(f)
            clusters = cluster_data.get("clusters", {})
            for cluster_id, cluster_info in clusters.items():
                members = cluster_info.get("members", [])
                cluster_id_to_unit_ids[cluster_id] = []
                for member in members:
                    unit_id = member.get("unit_id", "")
                    if unit_id:
                        unit_id_to_cluster_id[unit_id] = cluster_id
                        cluster_id_to_unit_ids[cluster_id].append(unit_id)
            logger.debug(f"  Loaded {len(unit_id_to_cluster_id)} unit_id mappings from {cluster_file.name}")
        except Exception as e:
            logger.warning(f"  Failed to load cluster file: {e}")
    else:
        logger.debug(f"  Cluster file not found: {cluster_file}")

    return dia_id_to_unit_id, unit_id_to_cluster_id, cluster_id_to_unit_ids


def _get_ground_truth_clusters(
    evidence_list: List[str],
    dia_id_to_unit_id: Dict[str, str],
    unit_id_to_cluster_id: Dict[str, str],
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    根据 evidence 获取 ground truth 的 MemUnits 和 Clusters。

    Args:
        evidence_list: evidence dia_id 列表
        dia_id_to_unit_id: dia_id → unit_id 映射
        unit_id_to_cluster_id: unit_id → cluster_id 映射

    Returns:
        Tuple of:
        - ground_truth_unit_ids: 应该包含的 unit_id 列表
        - ground_truth_cluster_ids: 应该选择的 cluster_id 列表
        - evidence_detail: 详细映射 {dia_id: [unit_id, cluster_id]}
    """
    ground_truth_unit_ids: List[str] = []
    ground_truth_cluster_ids: Set[str] = set()
    evidence_detail: Dict[str, List[str]] = {}

    for dia_id in evidence_list:
        unit_id = dia_id_to_unit_id.get(dia_id)
        cluster_id = unit_id_to_cluster_id.get(unit_id) if unit_id else None

        evidence_detail[dia_id] = [
            unit_id if unit_id else "NOT_FOUND",
            cluster_id if cluster_id else "NOT_FOUND"
        ]

        if unit_id and unit_id not in ground_truth_unit_ids:
            ground_truth_unit_ids.append(unit_id)
        if cluster_id:
            ground_truth_cluster_ids.add(cluster_id)

    return ground_truth_unit_ids, list(ground_truth_cluster_ids), evidence_detail


def _extract_cluster_selection_data(
    results_for_conv: List[dict],
    qa_list: List[QAPair],
    dia_id_to_unit_id: Dict[str, str],
    unit_id_to_cluster_id: Dict[str, str],
) -> Optional[dict]:
    """
    从检索结果中提取 Cluster Selection 信息，用于生成 checkpoint。

    Args:
        results_for_conv: 当前对话的所有 QA 检索结果
        qa_list: QA 对列表（包含 evidence 信息）
        dia_id_to_unit_id: dia_id → unit_id 映射
        unit_id_to_cluster_id: unit_id → cluster_id 映射

    Returns:
        Cluster selection checkpoint 数据，包含每个问题的选择详情和 ground truth；
        如果没有 cluster_rerank 数据则返回 None
    """
    checkpoint_data = {
        "qa_count": len(results_for_conv),
        "questions": [],
    }

    # 建立 question_id → QAPair 映射，用于获取 evidence
    qa_by_id = {qa.question_id: qa for qa in qa_list}

    for result in results_for_conv:
        if not result:
            continue

        retrieval_meta = result.get("retrieval_metadata", {})
        cluster_expansion = retrieval_meta.get("cluster_expansion", {})

        # 只有 cluster_rerank 策略才有这些信息
        if cluster_expansion.get("strategy") == "cluster_rerank":
            question_id = result.get("question_id", "")
            qa = qa_by_id.get(question_id)

            # 获取 ground truth
            ground_truth_unit_ids = []
            ground_truth_cluster_ids = []
            evidence_detail = {}

            if qa and qa.evidence:
                ground_truth_unit_ids, ground_truth_cluster_ids, evidence_detail = _get_ground_truth_clusters(
                    qa.evidence,
                    dia_id_to_unit_id,
                    unit_id_to_cluster_id,
                )

            # 计算 cluster selection 是否正确
            clusters_selected = cluster_expansion.get("clusters_selected", [])
            cluster_hit = bool(set(clusters_selected) & set(ground_truth_cluster_ids)) if ground_truth_cluster_ids else None

            # 🔥 新增：计算 memunit_hit（选中的 Cluster 是否包含 ground truth MemUnit）
            # 这个指标更准确，因为不同 Cluster 可能包含相同的 MemUnit
            unit_to_cluster = cluster_expansion.get("unit_to_cluster", {})
            selected_unit_ids = set(unit_to_cluster.keys())
            ground_truth_unit_set = set(ground_truth_unit_ids)

            # memunit_hit: 选中的 MemUnit 是否与 ground truth 有交集
            memunit_hit = bool(selected_unit_ids & ground_truth_unit_set) if ground_truth_unit_ids else None
            # memunit_coverage: 覆盖了多少比例的 ground truth MemUnit
            memunit_coverage = (
                len(selected_unit_ids & ground_truth_unit_set) / len(ground_truth_unit_set)
                if ground_truth_unit_set else None
            )

            question_data = {
                "query": result.get("query", ""),
                "question_id": question_id,
                # 实际选择
                "clusters_found": cluster_expansion.get("clusters_found", []),
                "clusters_selected": clusters_selected,
                "selection_reasoning": cluster_expansion.get("selection_reasoning", ""),
                "cluster_details": cluster_expansion.get("cluster_details", {}),
                "members_per_cluster": cluster_expansion.get("members_per_cluster", {}),
                "final_count": cluster_expansion.get("final_count", 0),
                "truncated": cluster_expansion.get("truncated", False),
                # Ground Truth
                "evidence": qa.evidence if qa else [],
                "evidence_detail": evidence_detail,
                "ground_truth_unit_ids": ground_truth_unit_ids,
                "ground_truth_cluster_ids": ground_truth_cluster_ids,
                # 评估指标
                "cluster_hit": cluster_hit,  # Cluster ID 匹配
                "memunit_hit": memunit_hit,  # MemUnit ID 匹配（更准确）
                "memunit_coverage": memunit_coverage,  # MemUnit 覆盖率
                "selected_unit_ids": list(selected_unit_ids),  # 选中的 MemUnit IDs
            }
            checkpoint_data["questions"].append(question_data)

    # 🔥 新增：汇总统计
    if checkpoint_data["questions"]:
        questions = checkpoint_data["questions"]
        total = len(questions)

        # cluster_hit 统计
        cluster_hits = [q["cluster_hit"] for q in questions if q["cluster_hit"] is not None]
        cluster_hit_count = sum(cluster_hits) if cluster_hits else 0

        # memunit_hit 统计（更准确的指标）
        memunit_hits = [q["memunit_hit"] for q in questions if q["memunit_hit"] is not None]
        memunit_hit_count = sum(memunit_hits) if memunit_hits else 0

        # memunit_coverage 平均值
        coverages = [q["memunit_coverage"] for q in questions if q["memunit_coverage"] is not None]
        avg_coverage = sum(coverages) / len(coverages) if coverages else 0

        checkpoint_data["summary"] = {
            "total_questions": total,
            "cluster_hit_count": cluster_hit_count,
            "cluster_hit_rate": cluster_hit_count / total if total else 0,
            "memunit_hit_count": memunit_hit_count,
            "memunit_hit_rate": memunit_hit_count / total if total else 0,
            "avg_memunit_coverage": avg_coverage,
        }

    return checkpoint_data if checkpoint_data["questions"] else None


async def run_search_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    index: Any,
    conversations: List,
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[SearchResult]:
    """
    并发执行检索，支持细粒度 checkpoint
    
    按会话分组处理，每处理完一个会话就保存 checkpoint
    
    Args:
        adapter: 系统适配器
        qa_pairs: QA 对列表
        index: 索引
        conversations: 对话列表（用于在线 API 重建缓存）
        checkpoint_manager: 断点续传管理器
        logger: 日志器
        
    Returns:
        检索结果列表
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Search stage")
    logger.info(f"{'='*60}")
    
    # 🔥 加载细粒度 checkpoint
    all_search_results_dict = {}
    if checkpoint_manager:
        all_search_results_dict = checkpoint_manager.load_search_progress()
    
    # 按会话分组 QA 对
    conv_to_qa = {}
    for qa in qa_pairs:
        conv_id = qa.metadata.get("conversation_id", "unknown")
        if conv_id not in conv_to_qa:
            conv_to_qa[conv_id] = []
        conv_to_qa[conv_id].append(qa)
    
    total_convs = len(conv_to_qa)
    processed_convs = set(all_search_results_dict.keys())
    remaining_convs = set(conv_to_qa.keys()) - processed_convs
    
    logger.info(f"Total conversations: {total_convs}")
    logger.info(f"Total questions: {len(qa_pairs)}")
    if processed_convs:
        logger.info(f"Already processed: {len(processed_convs)} conversations (from checkpoint)")
        logger.info(f"Remaining: {len(remaining_convs)} conversations")
    
    # 构建 conversation_id 到 conversation 的映射（用于在线 API 重建缓存）
    conv_id_to_conv = {conv.conversation_id: conv for conv in conversations}
    
    semaphore = asyncio.Semaphore(20)
    
    # 🔥 创建细粒度进度条（按问题追踪）
    total_questions = len(qa_pairs)
    processed_questions = sum(len(all_search_results_dict.get(conv_id, [])) for conv_id in processed_convs)
    
    pbar = tqdm(
        total=total_questions,
        initial=processed_questions,
        desc="🔍 Search Progress",
        unit="qa"
    )
    
    async def search_single_with_tracking(qa):
        # 设置 activity_id: search-{question_id}
        set_activity_id(f"search-{qa.question_id}")

        async with semaphore:
            conv_id = qa.metadata.get("conversation_id", "0")
            conversation = conv_id_to_conv.get(conv_id)
            result = await adapter.search(qa.question, conv_id, index, conversation=conversation)
            pbar.update(1)  # 每完成一个问题就更新进度条
            return result
    
    # 按会话逐个处理
    for idx, (conv_id, qa_list) in enumerate(sorted(conv_to_qa.items())):
        # 🔥 跳过已处理的会话
        if conv_id in processed_convs:
            logger.info(f"⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            continue
        
        logger.info(f"Processing Conversation ID: {conv_id} ({idx+1}/{total_convs}) - {len(qa_list)} questions")
        
        # 并发处理这个会话的所有问题
        tasks = [search_single_with_tracking(qa) for qa in qa_list]
        results_for_conv = await asyncio.gather(*tasks)
        
        # 将结果保存为字典格式
        results_for_conv_dict = [
            {
                "question_id": qa.question_id,
                "query": qa.question,
                "conversation_id": conv_id,
                "results": result.results,
                "retrieval_metadata": result.retrieval_metadata
            }
            for qa, result in zip(qa_list, results_for_conv)
        ]
        
        all_search_results_dict[conv_id] = results_for_conv_dict

        # 🔥 每处理完一个会话就保存检查点
        if checkpoint_manager:
            checkpoint_manager.save_search_progress(all_search_results_dict)

        # 🔥 保存 Cluster Selection Checkpoint（如果使用 cluster_rerank 策略）
        try:
            # 获取输出目录（从 index 或 checkpoint_manager）
            output_dir = None
            if isinstance(index, dict):
                output_dir = index.get("output_dir")
            if not output_dir and checkpoint_manager:
                output_dir = checkpoint_manager.output_dir

            if output_dir:
                output_dir = Path(output_dir)
                # 从 conv_id 提取数字索引（例如 "locomo_0" -> "0"）
                conv_index = conv_id.split("_")[-1] if "_" in conv_id else conv_id

                # 构建 evidence → MemUnit → Cluster 映射
                dia_id_to_unit_id, unit_id_to_cluster_id, _ = _build_evidence_to_cluster_mapping(
                    output_dir, conv_index, logger
                )

                cluster_selection_data = _extract_cluster_selection_data(
                    results_for_conv_dict,
                    qa_list,
                    dia_id_to_unit_id,
                    unit_id_to_cluster_id,
                )

                if cluster_selection_data:
                    cluster_selection_dir = output_dir / "cluster_selection"
                    cluster_selection_dir.mkdir(parents=True, exist_ok=True)

                    cluster_selection_path = cluster_selection_dir / f"conv_{conv_index}.json"

                    with open(cluster_selection_path, "w", encoding="utf-8") as f:
                        json.dump(cluster_selection_data, f, indent=2, ensure_ascii=False)

                    # 计算并打印 cluster hit 统计
                    total_questions = len(cluster_selection_data["questions"])
                    cluster_hits = sum(1 for q in cluster_selection_data["questions"] if q.get("cluster_hit") is True)
                    cluster_misses = sum(1 for q in cluster_selection_data["questions"] if q.get("cluster_hit") is False)
                    no_ground_truth = total_questions - cluster_hits - cluster_misses

                    logger.info(f"  💾 Cluster selection checkpoint saved: {cluster_selection_path.name}")
                    logger.info(f"     Cluster Hit: {cluster_hits}/{total_questions - no_ground_truth} ({cluster_hits/(total_questions - no_ground_truth)*100:.1f}%)" if (total_questions - no_ground_truth) > 0 else "")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to save cluster selection checkpoint: {e}")

    # 关闭进度条
    pbar.close()
    
    # 🔥 完成后删除细粒度检查点
    if checkpoint_manager:
        checkpoint_manager.delete_search_checkpoint()
    
    # 将字典格式转换为 SearchResult 对象列表（保持原有返回格式）
    all_results = []
    for conv_id in sorted(conv_to_qa.keys()):
        if conv_id in all_search_results_dict:
            for result_dict in all_search_results_dict[conv_id]:
                all_results.append(SearchResult(
                    query=result_dict["query"],
                    conversation_id=result_dict["conversation_id"],
                    results=result_dict["results"],
                    retrieval_metadata=result_dict.get("retrieval_metadata", {})
                ))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 All conversations processed!")
    logger.info(f"{'='*60}")
    logger.info(f"✅ Search completed: {len(all_results)} results\n")
    return all_results

