"""
Pipeline 核心

评测流程的编排器，负责协调 Add → Search → Answer → Evaluate 四个阶段。
"""
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from eval.core.data_models import (
    Dataset, SearchResult, AnswerResult, EvaluationResult
)
from eval.adapters.base import BaseAdapter
from eval.evaluators.base import BaseEvaluator
from core.observation.logger import setup_logger, get_console
from eval.utils.saver import ResultSaver
from eval.utils.checkpoint import CheckpointManager

# 导入答案生成所需的组件
from providers.llm.llm_provider import LLMProvider

# 导入各个阶段的执行函数
from eval.core.stages.add_stage import run_add_stage
from eval.core.stages.search_stage import run_search_stage
from eval.core.stages.answer_stage import run_answer_stage
from eval.core.stages.evaluate_stage import run_evaluate_stage


class Pipeline:
    """
    评测 Pipeline
    
    四阶段流程：
    1. Add: 摄入对话数据并构建索引
    2. Search: 检索相关记忆
    3. Answer: 生成答案
    4. Evaluate: 评估答案质量
    """
    
    def __init__(
        self,
        adapter: BaseAdapter,
        evaluator: BaseEvaluator,
        llm_provider: LLMProvider,
        output_dir: Path,
        run_name: str = "default",
        use_checkpoint: bool = True,
        filter_categories: Optional[List[int]] = None,
    ):
        """
        初始化 Pipeline
        
        Args:
            adapter: 系统适配器
            evaluator: 评估器
            llm_provider: LLM Provider（用于答案生成）
            output_dir: 输出目录
            run_name: 运行名称（用于区分不同运行）
            use_checkpoint: 是否启用断点续传
            filter_categories: 需要过滤掉的问题类别列表（如 [5] 表示过滤掉 Category 5）
        """
        self.adapter = adapter
        self.evaluator = evaluator
        self.llm_provider = llm_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Determine run number from existing run*.log files
        run_number = self._get_next_run_number()
        self.logger = setup_logger(log_dir=self.output_dir, run_number=run_number)
        self.saver = ResultSaver(self.output_dir)
        self.console = get_console()
        
        # 断点续传支持
        self.use_checkpoint = use_checkpoint
        self.checkpoint = CheckpointManager(output_dir=output_dir, run_name=run_name) if use_checkpoint else None
        self.completed_stages: set = set()
        
        # 问题类别过滤配置（从数据集配置中读取）
        self.filter_categories = filter_categories or []

        # Store run_number for use in report generation
        self.run_number = run_number

    def _get_next_run_number(self) -> Optional[int]:
        """
        Determine the current run number by finding the most recently created run*.log file.

        This ensures the pipeline log number matches the run.log number, even when
        run.log is created before pipeline initialization.

        Returns:
            Run number (1, 2, 3, ...) for numbered runs, None for first run (run.log)
        """
        import glob
        import os

        # Find all run*.log files
        run_logs = list(self.output_dir.glob("run*.log"))

        if not run_logs:
            return None  # First run, no numbering needed

        # Find the most recently created run log
        latest_run_log = max(run_logs, key=lambda p: os.path.getctime(p))

        # Extract number from filename
        filename = latest_run_log.name
        if filename == "run.log":
            return None  # First run
        elif filename.startswith("run_") and filename.endswith(".log"):
            # Extract number from "run_N.log"
            number_str = filename[4:-4]  # Remove "run_" and ".log"
            try:
                return int(number_str)
            except ValueError:
                return None

        return None

    async def run(
        self,
        dataset: Dataset,
        stages: Optional[List[str]] = None,
        conv_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        运行完整 Pipeline

        Args:
            dataset: 标准格式数据集
            stages: 要执行的阶段列表，None 表示全部
                   可选值: ["add", "search", "answer", "evaluate"]
            conv_id: 指定处理的conversation索引（0-based），None表示处理所有

        Returns:
            评测结果字典
        """
        start_time = time.time()

        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print("🚀 Evaluation Pipeline", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        self.console.print(f"Dataset: {dataset.dataset_name}")
        self.console.print(f"System: {self.adapter.get_system_info()['name']}")
        self.console.print(f"Stages: {stages or 'all'}")
        self.console.print(f"{'='*60}\n", style="bold cyan")
        
        # 🔥 Filter by conversation index if specified
        if conv_id is not None:
            if 0 <= conv_id < len(dataset.conversations):
                selected_conv = dataset.conversations[conv_id]
                dataset.conversations = [selected_conv]
                
                # Filter QA pairs that belong to this conversation
                target_conv_id = selected_conv.conversation_id
                dataset.qa_pairs = [
                    qa for qa in dataset.qa_pairs 
                    if qa.metadata.get("conversation_id") == target_conv_id
                ]
                
                self.console.print(
                    f"[dim]🔍 Selected conversation {conv_id}: {target_conv_id} "
                    f"({len(dataset.qa_pairs)} QA pairs)[/dim]\n"
                )
            else:
                self.console.print(
                    f"[red]❌ Conversation index {conv_id} out of range "
                    f"(0-{len(dataset.conversations)-1})[/red]"
                )
                return {}
        
        
        # 根据配置过滤问题类别（如过滤掉 Category 5 对抗性问题）
        original_qa_count = len(dataset.qa_pairs)
        
        if self.filter_categories:
            # 将配置中的类别统一转为字符串（兼容 int 和 str 配置）
            filter_set = {str(cat) for cat in self.filter_categories}
            
            # 过滤掉指定类别的问题
            dataset.qa_pairs = [
                qa for qa in dataset.qa_pairs 
                if qa.category not in filter_set
            ]
            
            filtered_count = original_qa_count - len(dataset.qa_pairs)
            
            if filtered_count > 0:
                filtered_categories_str = ", ".join(sorted(filter_set))
                self.console.print(
                    f"[dim]🔍 Filtered out {filtered_count} questions from categories: {filtered_categories_str}[/dim]"
                )
                self.console.print(f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n")
        
        # 尝试加载 checkpoint
        search_results_data = None
        answer_results_data = None
        
        if self.use_checkpoint and self.checkpoint:
            checkpoint_data = self.checkpoint.load_checkpoint()
            if checkpoint_data:
                self.completed_stages = set(checkpoint_data.get('completed_stages', []))
                # 加载已保存的中间结果
                if 'search_results' in checkpoint_data:
                    search_results_data = checkpoint_data['search_results']
                if 'answer_results' in checkpoint_data:
                    answer_results_data = checkpoint_data['answer_results']
        
        # 默认执行所有阶段
        if stages is None:
            stages = ["add", "search", "answer", "evaluate"]
        
        results = {}
        
        # ===== Stage 1: Add =====
        add_just_completed = False  # 标记 add 是否刚刚完成
        
        if "add" in stages and "add" not in self.completed_stages:
            self.logger.info("Starting Stage 1: Add")
            
            stage_results = await run_add_stage(
                adapter=self.adapter,
                dataset=dataset,
                output_dir=self.output_dir,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
                console=self.console,
                completed_stages=self.completed_stages,
            )
            results.update(stage_results)
            add_just_completed = True  # Add 刚刚完成
            
        elif "add" in self.completed_stages:
            self.console.print("\n[yellow]⏭️  Skip Add stage (already completed)[/yellow]")
            # 重新构建索引元数据（由 adapter 负责，仅本地系统需要）
            # 对于在线 API，返回 None，但仍需设置 results["index"]
            index = self.adapter.build_lazy_index(dataset.conversations, self.output_dir)
            results["index"] = index  # 即使是 None 也要设置
        else:
            # 重新构建索引元数据（由 adapter 负责，仅本地系统需要）
            # 对于在线 API，返回 None，但仍需设置 results["index"]
            index = self.adapter.build_lazy_index(dataset.conversations, self.output_dir)
            results["index"] = index  # 即使是 None 也要设置
            if index is not None:
                self.logger.info("⏭️  Skipped Stage 1, using lazy loading")
        
        # ⏰ Post-Add Wait: 对于在线 API 系统，等待后台索引构建完成
        # 只有当 add 刚刚完成时才等待
        if add_just_completed:
            wait_seconds = self.adapter.config.get("post_add_wait_seconds", 0)
            if wait_seconds > 0 and "search" in stages:
                self.console.print(
                    f"\n[yellow]⏰ Waiting {wait_seconds}s for backend indexing to complete...[/yellow]"
                )
                self.logger.info(f"⏰ Waiting {wait_seconds}s for backend indexing")
                
                # 显示倒计时进度条
                from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    console=self.console
                ) as progress:
                    task = progress.add_task(
                        f"⏰ Backend indexing in progress...",
                        total=wait_seconds
                    )
                    for i in range(wait_seconds):
                        time.sleep(1)
                        progress.update(task, advance=1)
                
                self.console.print(f"[green]✅ Wait completed, ready for search[/green]\n")
                self.logger.info("✅ Post-add wait completed")
        
        # ===== Stage 2: Search =====
        if "search" in stages and "search" not in self.completed_stages:
            self.logger.info("Starting Stage 2: Search")
            
            search_results = await run_search_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                index=results["index"],
                conversations=dataset.conversations,  # 传递 conversations 用于重建缓存
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )
            
            self.saver.save_json(
                [self._search_result_to_dict(sr) for sr in search_results],
                "search_results.json"
            )
            results["search_results"] = search_results
            self.logger.info("✅ Stage 2 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("search")
            if self.checkpoint:
                search_results_data = [self._search_result_to_dict(sr) for sr in search_results]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data
                )
        elif "search" in self.completed_stages:
            self.console.print(f"\n[yellow]⏭️  Skip Search stage (already completed)[/yellow]")
            if search_results_data:
                # 从 checkpoint 加载
                search_results = [self._dict_to_search_result(d) for d in search_results_data]
                results["search_results"] = search_results
            elif self.saver.file_exists("search_results.json"):
                # 从文件加载
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
        elif "answer" in stages or "eval" in stages:
            # 只有当后续阶段需要 search_results 时，才尝试加载
            if self.saver.file_exists("search_results.json"):
                search_data = self.saver.load_json("search_results.json")
                search_results = [self._dict_to_search_result(d) for d in search_data]
                results["search_results"] = search_results
                self.logger.info("⏭️  Skipped Stage 2, loaded existing results")
            else:
                raise FileNotFoundError("Search results not found. Please run 'search' stage first.")
        else:
            # 不需要 search_results（例如只运行 add 阶段）
            search_results = None
        
        # ===== Stage 3: Answer =====
        if "answer" in stages and "answer" not in self.completed_stages:
            self.logger.info("Starting Stage 3: Answer")
            
            answer_results = await run_answer_stage(
                adapter=self.adapter,
                qa_pairs=dataset.qa_pairs,
                search_results=search_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )
            
            self.saver.save_json(
                [self._answer_result_to_dict(ar) for ar in answer_results],
                "answer_results.json"
            )
            results["answer_results"] = answer_results
            self.logger.info("✅ Stage 3 completed")
            
            # 保存 checkpoint
            self.completed_stages.add("answer")
            if self.checkpoint:
                answer_results_dict = [self._answer_result_to_dict(ar) for ar in answer_results]
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data if search_results_data else [self._search_result_to_dict(sr) for sr in search_results],
                    answer_results=answer_results_dict
                )
        elif "answer" in self.completed_stages:
            self.console.print(f"\n[yellow]⏭️  Skip Answer stage (already completed)[/yellow]")
            if answer_results_data:
                # 从 checkpoint 加载
                answer_results = [self._dict_to_answer_result(d) for d in answer_results_data]
                results["answer_results"] = answer_results
            elif self.saver.file_exists("answer_results.json"):
                # 从文件加载
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
        elif "evaluate" in stages:
            # 只有当 evaluate 阶段需要 answer_results 时，才尝试加载
            if self.saver.file_exists("answer_results.json"):
                answer_data = self.saver.load_json("answer_results.json")
                answer_results = [self._dict_to_answer_result(d) for d in answer_data]
                results["answer_results"] = answer_results
                self.logger.info("⏭️  Skipped Stage 3, loaded existing results")
            else:
                raise FileNotFoundError("Answer results not found. Please run 'answer' stage first.")
        else:
            # 不需要 answer_results（例如只运行 add 或 search）
            answer_results = None
        
        # ===== Stage 4: Evaluate =====
        if "evaluate" in stages and "evaluate" not in self.completed_stages:
            eval_result = await run_evaluate_stage(
                evaluator=self.evaluator,
                answer_results=answer_results,
                checkpoint_manager=self.checkpoint,
                logger=self.logger,
            )
            
            self.saver.save_json(
                self._eval_result_to_dict(eval_result),
                "eval_results.json"
            )
            results["eval_result"] = eval_result
            
            # 保存 checkpoint
            self.completed_stages.add("evaluate")
            if self.checkpoint:
                self.checkpoint.save_checkpoint(
                    self.completed_stages,
                    search_results=search_results_data if search_results_data else [self._search_result_to_dict(sr) for sr in search_results],
                    answer_results=answer_results_data if answer_results_data else [self._answer_result_to_dict(ar) for ar in answer_results],
                    eval_results=self._eval_result_to_dict(eval_result)
                )
        elif "evaluate" in self.completed_stages:
            self.console.print("\n[yellow]⏭️  Skip Evaluate stage (already completed)[/yellow]")
        
        # 生成报告
        elapsed_time = time.time() - start_time
        self._generate_report(results, elapsed_time)
        
        return results

    def _generate_report(self, results: Dict[str, Any], elapsed_time: float):
        """生成评测报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 Evaluation Report")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # 系统信息
        system_info = self.adapter.get_system_info()
        report_lines.append(f"System: {system_info['name']}")
        report_lines.append(f"Time Elapsed: {elapsed_time:.2f}s")
        report_lines.append("")
        
        # 评估结果
        if "eval_result" in results:
            eval_result = results["eval_result"]
            report_lines.append(f"Total Questions: {eval_result.total_questions}")
            report_lines.append(f"Correct: {eval_result.correct}")
            report_lines.append(f"Accuracy: {eval_result.accuracy:.2%}")
            report_lines.append("")
        
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)

        # 保存报告 (使用编号以避免覆盖)
        if self.run_number is not None:
            report_filename = f"report_{self.run_number}.txt"
        else:
            report_filename = "report.txt"
        report_path = self.output_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # 打印到控制台
        self.console.print("\n" + report_text, style="bold green")
        self.logger.info(f"Report saved to: {report_path}")
    
    # 序列化辅助方法
    def _search_result_to_dict(self, sr: SearchResult) -> dict:
        """将 SearchResult 对象转换为字典"""
        return {
            "query": sr.query,
            "conversation_id": sr.conversation_id,
            "results": sr.results,
            "retrieval_metadata": sr.retrieval_metadata,
        }
    
    def _dict_to_search_result(self, d: dict) -> SearchResult:
        """将字典转换为 SearchResult 对象"""
        return SearchResult(**d)
    
    def _answer_result_to_dict(self, ar: AnswerResult) -> dict:
        """将 AnswerResult 对象转换为字典"""
        # 处理空的 search_results
        return {
            "question_id": ar.question_id,
            "question": ar.question,
            "answer": ar.answer,
            "golden_answer": ar.golden_answer,
            "category": ar.category,
            "conversation_id": ar.conversation_id,
            "formatted_context": ar.formatted_context,
            "metadata": ar.metadata,
        }
    
    def _dict_to_answer_result(self, d: dict) -> AnswerResult:
        """将字典转换为 AnswerResult 对象"""
        return AnswerResult(**d)
    
    def _eval_result_to_dict(self, er: EvaluationResult) -> dict:
        """将 EvaluationResult 对象转换为字典"""
        return {
            "total_questions": er.total_questions,
            "correct": er.correct,
            "accuracy": er.accuracy,
            "detailed_results": er.detailed_results,
            "metadata": er.metadata,
        }
