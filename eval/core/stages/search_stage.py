"""
Search 阶段

负责检索相关记忆。
"""
import asyncio
from typing import List, Any, Optional
from logging import Logger
from tqdm import tqdm

from eval.core.data_models import QAPair, SearchResult
from eval.adapters.base import BaseAdapter
from eval.utils.checkpoint import CheckpointManager
from eval.utils.logger import set_activity_id


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
    logger.info(f"Stage 2/4: Search")
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

