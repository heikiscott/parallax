"""
Answer 阶段

负责生成答案。
"""
import asyncio
import time
from typing import List, Optional
from logging import Logger
from tqdm import tqdm

from eval.core.data_models import QAPair, SearchResult, AnswerResult
from eval.adapters.base import BaseAdapter
from eval.utils.checkpoint import CheckpointManager
from core.observation.logger import set_activity_id
from config import load_config


def build_context(search_result: SearchResult) -> str:
    """
    从检索结果构建上下文
    
    优先使用预格式化的context（双speaker场景），否则使用简单序号格式化（单speaker场景）
    
    Args:
        search_result: 检索结果
        
    Returns:
        上下文字符串
    """
    # 优先使用预格式化的 context（由 adapter 提供）
    formatted_context = search_result.retrieval_metadata.get("formatted_context", "")
    if formatted_context:
        return formatted_context
    
    # 单 speaker 场景：简单格式化
    context_parts = []
    
    # 添加记忆内容
    for idx, result in enumerate(search_result.results[:10], 1):
        content = result.get("content", "")
        context_parts.append(f"{idx}. {content}")
    
    context = "\n\n".join(context_parts)
    
    # 对于 Memos 等支持 preferences 的系统，添加格式化的 pref_string
    preferences = search_result.retrieval_metadata.get("preferences", {})
    pref_string = preferences.get("pref_string", "")
    
    if pref_string:
        context += "\n\n" + pref_string
    
    return context


async def run_answer_stage(
    adapter: BaseAdapter,
    qa_pairs: List[QAPair],
    search_results: List[SearchResult],
    checkpoint_manager: Optional[CheckpointManager],
    logger: Logger,
) -> List[AnswerResult]:
    """
    生成答案，支持细粒度 checkpoint
    
    每 SAVE_INTERVAL 个问题保存一次 checkpoint
    
    Args:
        adapter: 系统适配器
        qa_pairs: QA 对列表
        search_results: 检索结果列表
        checkpoint_manager: 断点续传管理器
        logger: 日志器
        
    Returns:
        答案结果列表
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Answer stage")
    logger.info(f"{'='*60}")

    SAVE_INTERVAL = 400  # 每 400 个任务保存一次
    MAX_CONCURRENT = 50  # 最大并发数（semaphore 限制，batch 内并发）

    # Batch processing configuration from config/app.yaml
    app_config = load_config("app")
    eval_cfg = app_config.evaluation.answer
    BATCH_SIZE = int(eval_cfg.batch_size)  # 每批次处理的请求数
    BATCH_DELAY = int(eval_cfg.batch_delay)  # batch 间延迟（秒）
    REQUEST_INTERVAL = float(eval_cfg.request_interval)  # batch 内请求间隔（秒）

    logger.info(f"Batch processing enabled:")
    logger.info(f"  - Batch size: {BATCH_SIZE}")
    logger.info(f"  - Batch delay: {BATCH_DELAY}s (between batches)")
    logger.info(f"  - Request interval: {REQUEST_INTERVAL}s (within batch)")
    logger.info(f"  - Timeout & concurrency: controlled by Provider layer")
    
    # 加载细粒度 checkpoint
    all_answer_results = {}
    if checkpoint_manager:
        loaded_results = checkpoint_manager.load_answer_progress()
        # 转换为 {question_id: AnswerResult} 格式
        for result in loaded_results.values():
            all_answer_results[result["question_id"]] = result
    
    total_qa_count = len(qa_pairs)
    processed_count = len(all_answer_results)
    
    logger.info(f"Total questions: {total_qa_count}")
    if processed_count > 0:
        logger.info(f"Already processed: {processed_count} questions (from checkpoint)")
        logger.info(f"Remaining: {total_qa_count - processed_count} questions")

    # 准备待处理的任务
    pending_tasks = []
    for qa, sr in zip(qa_pairs, search_results):
        if qa.question_id not in all_answer_results:
            pending_tasks.append((qa, sr))

    if not pending_tasks:
        logger.info(f"✅ All questions already processed!")
        # 转换为 AnswerResult 对象列表（按原始顺序）
        results = []
        for qa in qa_pairs:
            if qa.question_id in all_answer_results:
                result_dict = all_answer_results[qa.question_id]
                results.append(AnswerResult(
                    question_id=result_dict["question_id"],
                    question=result_dict["question"],
                    answer=result_dict["answer"],
                    golden_answer=result_dict["golden_answer"],
                    category=result_dict.get("category"),
                    conversation_id=result_dict.get("conversation_id", ""),
                    formatted_context=result_dict.get("formatted_context", ""),  # 加载 formatted_context
                    # search_results 不再加载以节省空间
                ))
        return results

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    completed = processed_count
    failed = 0
    start_time = time.time()

    # 使用 tqdm 进度条
    pbar = tqdm(
        total=total_qa_count,
        initial=processed_count,
        desc="💬 Answer Progress",
        unit="qa"
    )

    async def answer_single_with_tracking(qa, search_result):
        nonlocal completed, failed

        # 设置 activity_id: answer-{question_id}
        set_activity_id(f"answer-{qa.question_id}")

        async with semaphore:
            try:
                # 构建 context
                context = build_context(search_result)
                
                # 检测是否为选择题，如果是则增强 question
                query = qa.question
                if "all_options" in qa.metadata:
                    options = qa.metadata["all_options"]
                    options_text = "\n".join([f"{key} {value}" for key, value in options.items()])
                    
                    # 将选项和要求整合到 question 中
                    query = f"""{qa.question}

OPTIONS:
{options_text}

IMPORTANT: This is a multiple-choice question. You MUST analyze the context and select the BEST option. In your FINAL ANSWER, return ONLY the option letter like (a), (b), (c), or (d), nothing else."""
                
                # 直接调用 adapter 的 answer 方法
                answer = await adapter.answer(
                    query=query,
                    context=context,
                    conversation_id=search_result.conversation_id,
                )
                
                answer = answer.strip()
            
            except Exception as e:
                logger.error(f"  ⚠️ Answer generation failed for {qa.question_id}: {e}")
                answer = "Error: Failed to generate answer"
                failed += 1
            
            result = AnswerResult(
                question_id=qa.question_id,
                question=qa.question,
                answer=answer,
                golden_answer=qa.answer,
                category=qa.category,
                conversation_id=search_result.conversation_id,
                formatted_context=context,  # 保存实际使用的上下文
                # search_results 不再保存以节省空间
            )
            
            # 保存结果
            all_answer_results[qa.question_id] = {
                "question_id": result.question_id,
                "question": result.question,
                "answer": result.answer,
                "golden_answer": result.golden_answer,
                "category": result.category,
                "conversation_id": result.conversation_id,
                "formatted_context": result.formatted_context,  # 保存 formatted_context
                # search_results 不再保存以节省空间
            }
            
            completed += 1
            pbar.update(1)  # 更新进度条
            
            # 定期保存 checkpoint
            if checkpoint_manager and (completed % SAVE_INTERVAL == 0 or completed == total_qa_count):
                elapsed = time.time() - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                eta = (total_qa_count - completed) / speed if speed > 0 else 0
                
                logger.info(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
                          f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
                
                checkpoint_manager.save_answer_progress(all_answer_results, completed, total_qa_count)
            
            return result

    # Batch processing: 分批执行，确保当前 batch 完全结束后才开始下一个 batch
    # 并发控制和超时已经在 Provider 层处理，这里负责批次管理和请求间隔控制
    total_batches = (len(pending_tasks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(pending_tasks), BATCH_SIZE):
        batch_tasks_data = pending_tasks[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1

        logger.info(f"\n{'─'*60}")
        logger.info(f"🚀 Processing Batch {batch_num}/{total_batches} ({len(batch_tasks_data)} requests)")

        batch_start_time = time.time()

        # 逐个发送请求，每个请求间隔 REQUEST_INTERVAL 秒
        batch_tasks = []
        for i, (qa, sr) in enumerate(batch_tasks_data):
            # 创建任务
            task = asyncio.create_task(answer_single_with_tracking(qa, sr))
            batch_tasks.append(task)

            # 如果不是最后一个请求，等待 REQUEST_INTERVAL 秒再发送下一个
            if i < len(batch_tasks_data) - 1:
                await asyncio.sleep(REQUEST_INTERVAL)

        # 等待当前 batch 的所有任务完成
        await asyncio.gather(*batch_tasks)

        batch_duration = time.time() - batch_start_time

        # 记录批次统计
        logger.info(f"✅ Batch {batch_num}/{total_batches} completed in {batch_duration:.1f}s")

        # 批次间延迟（除最后一个 batch 外）
        if batch_num < total_batches:
            logger.info(f"⏸️  Waiting {BATCH_DELAY}s before next batch...")
            await asyncio.sleep(BATCH_DELAY)

    # 关闭进度条
    pbar.close()
    
    # 统计信息
    elapsed_time = time.time() - start_time
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ All responses generated!")
    logger.info(f"   - Total questions: {total_qa_count}")
    logger.info(f"   - Successful: {completed - failed}")
    logger.info(f"   - Failed: {failed}")
    logger.info(f"   - Success rate: {success_rate:.1f}%")
    logger.info(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
    logger.info(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
    logger.info(f"{'='*60}\n")
    
    # 完成后删除细粒度检查点
    if checkpoint_manager:
        checkpoint_manager.delete_answer_checkpoints()
    
    # 转换为 AnswerResult 对象列表（按原始顺序）
    results = []
    for qa in qa_pairs:
        if qa.question_id in all_answer_results:
            result_dict = all_answer_results[qa.question_id]
            results.append(AnswerResult(
                question_id=result_dict["question_id"],
                question=result_dict["question"],
                answer=result_dict["answer"],
                golden_answer=result_dict["golden_answer"],
                category=result_dict.get("category"),
                conversation_id=result_dict.get("conversation_id", ""),
                formatted_context=result_dict.get("formatted_context", ""),
                search_results=result_dict.get("search_results", []),
            ))
    
    return results

