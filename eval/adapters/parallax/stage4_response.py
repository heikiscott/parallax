import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from time import time
from typing import List, Dict, Optional

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)



from eval.adapters.parallax.config import ExperimentConfig
from prompts.memory.en.eval.answer.answer_prompts import ANSWER_PROMPT

# 使用 Memory Layer 的 LLMProvider
from providers.llm.llm_provider import LLMProvider


# 🔥 Context 构建模板（从 stage3 迁移过来）
TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""


def load_memunits_by_conversation(conv_idx: int, memunits_dir: Path) -> Dict[str, dict]:
    """
    加载指定对话的所有 memunits，返回 unit_id -> memunit 的映射
    
    Args:
        conv_idx: 对话索引
        memunits_dir: memunits 目录路径
    
    Returns:
        {unit_id: memunit_dict} 的映射
    """
    memunit_file = memunits_dir / f"memunit_list_conv_{conv_idx}.json"
    
    if not memunit_file.exists():
        print(f"Warning: MemUnit file not found: {memunit_file}")
        return {}
    
    try:
        with open(memunit_file, "r", encoding="utf-8") as f:
            memunits = json.load(f)
        
        # 构建 unit_id -> memunit 的映射
        memunit_map = {}
        for memunit in memunits:
            unit_id = memunit.get("unit_id")
            if unit_id:
                memunit_map[unit_id] = memunit
        
        return memunit_map
    
    except Exception as e:
        print(f"Error loading memunits from {memunit_file}: {e}")
        return {}


def build_context_from_unit_ids(
    unit_ids: List[str],
    memunit_map: Dict[str, dict],
    speaker_a: str,
    speaker_b: str,
    top_k: int = 10
) -> str:
    """
    根据 unit_ids 从 memunit_map 中提取对应的 episode memory，构建 context

    Args:
        unit_ids: 检索到的 unit_ids 列表（已按相关性排序）
        memunit_map: unit_id -> memunit 的映射
        speaker_a: 说话者 A
        speaker_b: 说话者 B
        top_k: 选择前 k 个 unit_ids（默认 10）

    Returns:
        格式化的 context 字符串
    """
    # 🔥 选择 top-k unit_ids
    selected_unit_ids = unit_ids[:top_k]

    # 从 memunit_map 中提取对应的 episode memory
    retrieved_docs_text = []
    for unit_id in selected_unit_ids:
        memunit = memunit_map.get(unit_id)
        if not memunit:
            # 找不到对应的 memunit，跳过
            continue
        
        subject = memunit.get('subject', 'N/A')
        narrative = memunit.get('narrative', 'N/A')
        doc_text = f"{subject}: {narrative}\n---"
        retrieved_docs_text.append(doc_text)
    
    # 拼接所有文档
    speaker_memories = "\n\n".join(retrieved_docs_text)
    
    # 使用模板格式化最终 context
    context = TEMPLATE.format(
        speaker_1=speaker_a,
        speaker_2=speaker_b,
        speaker_memories=speaker_memories,
    )
    
    return context


async def locomo_response(
    llm_provider: LLMProvider,  # 改用 LLMProvider
    context: str,
    question: str,
    experiment_config: ExperimentConfig,
) -> str:
    """生成回答（使用 LLMProvider）
    
    Args:
        llm_provider: LLM Provider
        context: 检索到的上下文
        question: 用户问题
        experiment_config: 实验配置
    
    Returns:
        生成的答案
    """
    prompt = ANSWER_PROMPT.format(context=context, question=question)

    # 初始化 result 变量
    result = ""

    for i in range(experiment_config.max_retries):
        try:
            # Use 16384 as default max_tokens (matches gpt-4o-mini's output limit)
            result = await llm_provider.generate(
                prompt=prompt,
                temperature=0,
                max_tokens=int(os.getenv("LLM_MAX_TOKENS", "16384")),
            )

            # 安全解析 FINAL ANSWER
            if "FINAL ANSWER:" in result:
                parts = result.split("FINAL ANSWER:")
                if len(parts) > 1:
                    result = parts[1].strip()
                else:
                    # 分割失败，使用原始结果
                    result = result.strip()
            else:
                # 没有 FINAL ANSWER 标记，使用原始结果
                result = result.strip()

            if result == "":
                continue
            break
        except Exception as e:
            logger.warning(f"Answer generation error (attempt {i+1}/{experiment_config.max_retries}): {e}")
            # 如果是最后一次重试，记录错误但返回空字符串而不是抛出异常
            if i == experiment_config.max_retries - 1:
                logger.error(f"All {experiment_config.max_retries} retries failed. Returning empty answer.")
                result = ""
            else:
                # Aggressive exponential backoff: 5 * 2^i seconds, max 500s (~8 min)
                # i=0: 5s, i=1: 10s, i=2: 20s, i=3: 40s, i=4: 80s, i=5: 160s, i=6: 320s, i>=7: 500s
                backoff_time = min(5 * (2 ** i), 500)
                logger.info(f"Waiting {backoff_time}s before retry {i+2}/{experiment_config.max_retries}...")
                await asyncio.sleep(backoff_time)
            continue

    return result


async def process_qa(
    qa, 
    search_result, 
    llm_provider, 
    experiment_config,
    memunit_map: Dict[str, dict],
    speaker_a: str,
    speaker_b: str
):
    """
    处理单个 QA 对（新版：从 unit_ids 构建 context）

    Args:
        qa: 问题和答案对
        search_result: 检索结果（包含 unit_ids）
        llm_provider: LLM Provider
        experiment_config: 实验配置
        memunit_map: unit_id -> memunit 的映射
        speaker_a: 说话者 A
        speaker_b: 说话者 B

    Returns:
        包含问题、答案、类别等信息的字典
    """
    start = time()
    query = qa.get("question")
    gold_answer = qa.get("answer")
    qa_category = qa.get("category")

    # 🔥 从 unit_ids 构建 context（使用 top_k）
    unit_ids = search_result.get("unit_ids", [])
    top_k = experiment_config.response_top_k

    context = build_context_from_unit_ids(
        unit_ids=unit_ids,
        memunit_map=memunit_map,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        top_k=top_k
    )

    answer = await locomo_response(
        llm_provider, context, query, experiment_config
    )

    response_duration_ms = (time() - start) * 1000

    # 只在 verbose 模式下输出（减少日志）
    # print(f"Processed question: {query}")
    # print(f"Answer: {answer}")

    return {
        "question": query,
        "answer": answer,
        "category": qa_category,
        "golden_answer": gold_answer,
        "search_context": context,  # 保存构建的 context
        "unit_ids_used": unit_ids[:top_k],  # 🔥 记录实际使用的 unit_ids
        "response_duration_ms": response_duration_ms,
        "search_duration_ms": search_result.get("retrieval_metadata", {}).get("total_latency_ms", 0),
    }


async def main(search_path, save_path):
    """
    优化后的主函数
    
    性能优化：
    1. 全局并发处理：所有 QA 对并发处理，而不是按 conversation 串行
    2. 并发控制：使用 Semaphore 控制最大并发数
    3. 进度监控：实时显示处理进度
    4. 增量保存：每个 conversation 完成后立即保存（避免崩溃丢失数据）
    
    优化效果：
    - 优化前：77 分钟（串行）
    - 优化后：~8 分钟（并发 50）
    - 加速比：~10x
    """
    llm_config = ExperimentConfig.llm_config["openai"]
    experiment_config = ExperimentConfig()
    
    # 创建 LLM Provider（替代 AsyncOpenAI）
    llm_provider = LLMProvider(
        provider_type="openai",
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        base_url=llm_config["base_url"],
        temperature=llm_config.get("temperature", 0.0),
        max_tokens=int(llm_config.get("max_tokens", int(os.getenv("LLM_MAX_TOKENS", "32768")))),
    )
    
    locomo_df = pd.read_json(experiment_config.datase_path)
    with open(search_path) as file:
        locomo_search_results = json.load(file)

    num_users = len(locomo_df)
    
    # 🔥 加载 memunits 目录
    memunits_dir = Path(search_path).parent / "memunits"
    if not memunits_dir.exists():
        print(f"Error: MemUnits directory not found: {memunits_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Stage4: LLM Response Generation")
    print(f"{'='*60}")
    print(f"Total conversations: {num_users}")
    print(f"Response top-k: {experiment_config.response_top_k}")
    print(f"MemUnits directory: {memunits_dir}")
    
    # 🔥 优化1：全局并发控制（关键优化）
    # 控制同时处理的 QA 对数量，避免 API 限流
    MAX_CONCURRENT = int(os.getenv('EVAL_RESPONSE_MAX_CONCURRENT', '5'))  # 可根据 API 限制调整（10-100）
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    # 🔥 优化2：收集所有 QA 对（跨 conversation）
    all_tasks = []
    task_to_group = {}  # 用于追踪每个任务属于哪个 group
    
    # 🔥 优化3：定义带并发控制的处理函数
    async def process_qa_with_semaphore(qa, search_result, group_id, memunit_map, speaker_a, speaker_b):
        """带并发控制的 QA 处理"""
        async with semaphore:
            result = await process_qa(
                qa, search_result, llm_provider, experiment_config,
                memunit_map, speaker_a, speaker_b
            )
            return (group_id, result)
    
    total_qa_count = 0
    for group_idx in range(num_users):
        qa_set = locomo_df["qa"].iloc[group_idx]
        qa_set_filtered = [qa for qa in qa_set if qa.get("category") != 5]

        group_id = f"locomo_exp_user_{group_idx}"
        search_results = locomo_search_results.get(group_id)
        
        # 🔥 加载当前对话的 memunits
        memunit_map = load_memunits_by_conversation(group_idx, memunits_dir)
        print(f"Loaded {len(memunit_map)} memunits for conversation {group_idx}")
        
        # 🔥 获取 speaker 信息
        conversation_data = locomo_df["conversation"].iloc[group_idx]
        speaker_a = conversation_data.get("speaker_a", "Speaker A")
        speaker_b = conversation_data.get("speaker_b", "Speaker B")

        matched_pairs = []
        for qa in qa_set_filtered:
            question = qa.get("question")
            matching_result = next(
                (
                    result
                    for result in search_results
                    if result.get("query") == question
                ),
                None,
            )
            if matching_result:
                matched_pairs.append((qa, matching_result))
            else:
                print(
                    f"Warning: No matching search result found for question: {question}"
                )
        
        total_qa_count += len(matched_pairs)
        
        # 创建任务（全局并发）
        for qa, search_result in matched_pairs:
            task = process_qa_with_semaphore(qa, search_result, group_id, memunit_map, speaker_a, speaker_b)
            all_tasks.append(task)
    
    print(f"Total questions to process: {total_qa_count}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}")
    print(f"Estimated time: {total_qa_count * 3 / MAX_CONCURRENT / 60:.1f} minutes")
    print(f"\n{'='*60}")
    print(f"Starting parallel processing...")
    print(f"{'='*60}\n")
    
    # 🔥 优化4：全局并发执行所有任务（带进度监控）
    all_responses = {f"locomo_exp_user_{i}": [] for i in range(num_users)}
    
    import time as time_module
    start_time = time_module.time()
    completed = 0
    failed = 0
    
    # 🔥 优化5：分批处理 + 增量保存（避免崩溃丢失数据）
    CHUNK_SIZE = 200  # 每次处理 200 个任务
    SAVE_INTERVAL = 400  # 每 400 个任务保存一次
    
    for chunk_start in range(0, len(all_tasks), CHUNK_SIZE):
        chunk_tasks = all_tasks[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
        
        # 将结果分组到各个 conversation
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"  ❌ Task failed: {result}")
                failed += 1
                continue
            
            group_id, qa_result = result
            all_responses[group_id].append(qa_result)
        
        completed += len(chunk_tasks)
        elapsed = time_module.time() - start_time
        speed = completed / elapsed if elapsed > 0 else 0
        eta = (total_qa_count - completed) / speed if speed > 0 else 0
        
        print(f"Progress: {completed}/{total_qa_count} ({completed/total_qa_count*100:.1f}%) | "
              f"Speed: {speed:.1f} qa/s | Failed: {failed} | ETA: {eta/60:.1f} min")
        
        # 🔥 增量保存（每 SAVE_INTERVAL 个任务保存一次）
        if completed % SAVE_INTERVAL == 0 or completed == total_qa_count:
            temp_save_path = Path(save_path).parent / f"responses_checkpoint_{completed}.json"
            with open(temp_save_path, "w", encoding="utf-8") as f:
                json.dump(all_responses, f, indent=2, ensure_ascii=False)
            print(f"  💾 Checkpoint saved: {temp_save_path.name}")
    
    elapsed_time = time_module.time() - start_time
    success_rate = (completed - failed) / completed * 100 if completed > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"✅ All responses generated!")
    print(f"   - Total questions: {total_qa_count}")
    print(f"   - Successful: {completed - failed}")
    print(f"   - Failed: {failed}")
    print(f"   - Success rate: {success_rate:.1f}%")
    print(f"   - Time elapsed: {elapsed_time/60:.1f} minutes ({elapsed_time:.0f}s)")
    print(f"   - Average speed: {total_qa_count/elapsed_time:.1f} qa/s")
    print(f"{'='*60}\n")

    # 保存最终结果
    os.makedirs(Path(save_path).parent, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_responses, f, indent=2, ensure_ascii=False)
        print(f"✅ Final results saved to: {save_path}")
    
    # 清理 checkpoint 文件
    checkpoint_files = list(Path(save_path).parent.glob("responses_checkpoint_*.json"))
    for checkpoint_file in checkpoint_files:
        checkpoint_file.unlink()
        print(f"  🗑️  Removed checkpoint: {checkpoint_file.name}")


if __name__ == "__main__":
    config = ExperimentConfig()
    # 🔥 修正：实际文件在 locomo_eval/ 目录下，而不是 results/ 目录
    search_result_path = str(
        Path(__file__).parent
        / config.experiment_name  # 直接使用 experiment_name（即 "locomo_evaluation"）
        / "search_results.json"
    )
    save_path = (
        Path(__file__).parent / config.experiment_name / "responses.json"
    )
    # search_result_path = f"/Users/admin/Documents/Projects/b001-memsys/eval/locomo_eval/results/locomo_evaluation_0/nemori_locomo_search_results.json"

    asyncio.run(main(search_result_path, save_path))
