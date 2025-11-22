"""
LLM Judge 评估器

使用 LLM 作为评判器来评估答案的正确性。

对齐到 evaluation_archive 的评估逻辑：
- 保留每次 run 的独立判断 (judgment_1, judgment_2, judgment_3)
- 分别计算每次 run 的准确率
- 输出 mean 和 std
"""
import asyncio
import json
import logging
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict
from openai import AsyncOpenAI
from tqdm import tqdm

from eval.evaluators.base import BaseEvaluator
from eval.evaluators.registry import register_evaluator
from eval.core.data_models import AnswerResult, EvaluationResult
from eval.utils.prompts import get_prompt, format_prompt

logger = logging.getLogger(__name__)


@register_evaluator("llm_judge")
class LLMJudge(BaseEvaluator):
    """LLM 评判器"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # 初始化 OpenAI 客户端
        llm_config = config.get("llm", {})
        self.client = AsyncOpenAI(
            api_key=llm_config.get("api_key"),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1")
        )
        self.model = llm_config.get("model", "gpt-4o-mini")
        self.num_runs = config.get("num_runs", 3)
    
    async def evaluate(
        self, 
        answer_results: List[AnswerResult]
    ) -> EvaluationResult:
        """
        使用LLM评估答案，返回多次运行的统计结果
        
        Args:
            answer_results: 答案结果列表
            
        Returns:
            EvaluationResult: 包含mean和std的评估结果
        """
        print(f"\n{'='*60}")
        print(f"Evaluation: LLM Judge (model={self.model}, runs={self.num_runs})")
        print(f"{'='*60}")
        
        detailed_results = []
        
        # 并发评估所有答案
        semaphore = asyncio.Semaphore(10)  # 限制并发数
        
        # 使用 tqdm 进度条
        pbar = tqdm(total=len(answer_results), desc="⚖️  Evaluate Progress", unit="qa")
        
        async def evaluate_single(answer_result: AnswerResult):
            async with semaphore:
                result = await self._evaluate_single_answer(answer_result)
                pbar.update(1)  # 更新进度条
                return result
        
        tasks = [evaluate_single(ar) for ar in answer_results]
        results = await asyncio.gather(*tasks)
        
        # 关闭进度条
        pbar.close()
        
        # 收集结果
        for result in results:
            detailed_results.append(result)
        
        # 分别计算每次 run 的准确率
        run_scores = []
        category_stats = defaultdict(lambda: {"correct": [0] * self.num_runs, "total": 0})
        
        for i in range(self.num_runs):
            judgment_key = f"judgment_{i+1}"
            correct_count = 0
            total_count = 0
            
            for result in detailed_results:
                llm_judgments = result.get("llm_judgments", {})
                category = result.get("category")
                
                if judgment_key in llm_judgments:
                    total_count += 1
                    if llm_judgments[judgment_key]:
                        correct_count += 1
                        if category is not None:
                            category_stats[category]["correct"][i] += 1
                
                # 统计 category 总数（只需要一次）
                if i == 0 and category is not None:
                    category_stats[category]["total"] += 1
            
            if total_count > 0:
                run_accuracy = correct_count / total_count
                run_scores.append(run_accuracy)
        
        # 计算统计量
        mean_accuracy = np.mean(run_scores) if run_scores else 0.0
        std_accuracy = np.std(run_scores) if run_scores else 0.0
        
        # 计算每个 category 的准确率
        category_accuracies = {}
        for category, stats in category_stats.items():
            cat_accuracies = []
            for i in range(self.num_runs):
                if stats["total"] > 0:
                    cat_acc = stats["correct"][i] / stats["total"]
                    cat_accuracies.append(cat_acc)
            
            if cat_accuracies:
                category_accuracies[str(category)] = {
                    "mean": np.mean(cat_accuracies),
                    "std": np.std(cat_accuracies),
                    "individual_runs": cat_accuracies,
                    "total": stats["total"]
                }
        
        print(f"\n✅ 评估完成:")
        print(f"   - 总问题数: {len(answer_results)}")
        print(f"   - 平均准确率: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
        print(f"   - 标准差: {std_accuracy:.4f}")
        print(f"   - 各次 run 准确率: {[f'{s:.4f}' for s in run_scores]}")
        
        if category_accuracies:
            print(f"\n📊 按 Category 统计:")
            for cat, stats in sorted(category_accuracies.items()):
                print(f"   Category {cat}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['total']})")
        
        # 按 conversation 分组
        grouped_results = self._group_by_conversation(detailed_results)
        
        return EvaluationResult(
            total_questions=len(answer_results),
            correct=int(mean_accuracy * len(answer_results)),  # 使用 mean 计算
            accuracy=mean_accuracy,
            detailed_results=grouped_results,
            metadata={
                "model": self.model,
                "num_runs": self.num_runs,
                "mean_accuracy": mean_accuracy,
                "std_accuracy": std_accuracy,
                "run_scores": run_scores,
                "category_accuracies": category_accuracies
            }
        )
    
    def _group_by_conversation(self, detailed_results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        将结果按conversation分组（例如：locomo_exp_user_0, locomo_exp_user_1等）
        """
        grouped = defaultdict(list)
        
        for result in detailed_results:
            question_id = result.get("question_id", "")
            
            # 从 question_id 提取 conversation 信息
            # 例如: "locomo_0_qa0" -> "locomo_exp_user_0"
            # 例如: "personamem_5_qa2" -> "personamem_exp_user_5"
            if "_qa" in question_id:
                parts = question_id.split("_qa")
                conv_id = parts[0]  # "locomo_0" or "personamem_5"
                
                # 转换为 evaluation_archive 的格式
                if "_" in conv_id:
                    dataset_name, conv_num = conv_id.rsplit("_", 1)
                    group_key = f"{dataset_name}_exp_user_{conv_num}"
                else:
                    group_key = f"{conv_id}_exp_user_0"
            else:
                # 如果格式不符合预期，使用默认分组
                group_key = "default_group"
            
            grouped[group_key].append(result)
        
        return dict(grouped)
    
    async def _evaluate_single_answer(self, answer_result: AnswerResult) -> dict:
        """
        评估单个答案，保留每次run的独立判断
        """
        question = answer_result.question
        golden_answer = answer_result.golden_answer
        generated_answer = answer_result.answer
        
        # 多次评估，保留独立判断
        judgments = []
        for _ in range(self.num_runs):
            is_correct = await self._judge_answer(
                question, golden_answer, generated_answer
            )
            judgments.append(is_correct)
        
        # 使用 judgment_1, judgment_2, ... 格式
        llm_judgments = {
            f"judgment_{i+1}": judgment 
            for i, judgment in enumerate(judgments)
        }
        
        return {
            "question_id": answer_result.question_id,
            "question": question,
            "golden_answer": golden_answer,
            "generated_answer": generated_answer,
            "llm_judgments": llm_judgments,
            "category": answer_result.category,
        }
    
    async def _judge_answer(
        self, 
        question: str, 
        golden_answer: str, 
        generated_answer: str
    ) -> bool:
        """
        使用 LLM 判断答案是否正确
        
        Returns:
            True 如果正确，False 如果错误
        """
        # 使用配置化的 prompts
        system_prompt = get_prompt("llm_judge", "system_prompt")
        user_prompt = format_prompt(
            "llm_judge",
            "user_prompt",
            question=question,
            golden_answer=golden_answer,
            generated_answer=generated_answer
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            label = result.get("label", "WRONG")
            
            return label.strip().upper() == "CORRECT"

        except Exception as e:
            logger.warning(f"LLM Judge failed: {e}")
            return False

