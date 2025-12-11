"""
CLI 入口

评测框架的命令行接口。

Usage:
    python -m eval.cli --dataset locomo --system parallax
    python -m eval.cli --dataset locomo-mini --system parallax
    python -m eval.cli --dataset locomo --system parallax --stages search answer evaluate
    python -m eval.cli --dataset locomo --system parallax --conv 3
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path

# ===== 环境初始化 =====
# 必须在导入任何 Parallax 组件之前完成
# 参考 src/bootstrap.py 的初始化逻辑

# 1. 添加项目路径
project_root = Path(__file__).parent.parent.resolve()
src_path = project_root / "src"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# 2. 设置环境并检查必要的 secrets
# 注意：敏感信息从 config/secrets/secrets.yaml 加载
from utils.load_env import setup_environment
setup_environment(check_secrets=["openai_api_key"])

# ===== 现在可以安全地导入 Parallax 组件 =====
from eval.core.loaders import load_dataset
from eval.adapters.registry import create_adapter
from eval.evaluators.registry import create_evaluator
from config import load_yaml
from core.observation.logger import get_console, setup_logger

from providers.llm.llm_provider import LLMProvider


def deep_merge_config(base: dict, override: dict) -> dict:
    """
    深度合并配置字典
    
    Args:
        base: 基础配置
        override: 覆盖配置
        
    Returns:
        合并后的配置
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并嵌套字典
            result[key] = deep_merge_config(result[key], value)
        else:
            # 直接覆盖
            result[key] = value
    return result


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Memory System Evaluation Framework")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., locomo)"
    )
    parser.add_argument(
        "--system",
        type=str,
        required=True,
        help="System name (e.g., parallax)"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Workflow config name (e.g., standard_pipeline, search_only). If not specified, auto-selects based on system config."
    )
    parser.add_argument(
        "--conv",
        type=int,
        default=None,
        help="Conversation index to process (0-based). If not specified, all conversations will be processed."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: results/{dataset}-{system}"
    )
    
    args = parser.parse_args()
    
    console = get_console()
    
    # ===== 加载配置 =====
    console.print("\n[bold cyan]Loading configurations...[/bold cyan]")
    
    evaluation_root = Path(__file__).parent
    
    # 加载数据集配置
    dataset_config_path = project_root / "config" / "eval" / "datasets" / f"{args.dataset}.yaml"
    if not dataset_config_path.exists():
        console.print(f"[red]❌ Dataset config not found: {dataset_config_path}[/red]")
        return
    
    dataset_config = load_yaml(str(dataset_config_path))
    console.print(f"  ✅ Loaded dataset config: {args.dataset}")
    
    # 加载系统配置
    system_config_path = project_root / "config" / "eval" / "systems" / f"{args.system}.yaml"
    if not system_config_path.exists():
        console.print(f"[red]❌ System config not found: {system_config_path}[/red]")
        return
    
    system_config = load_yaml(str(system_config_path))
    console.print(f"  ✅ Loaded system config: {args.system}")
    
    # 应用数据集特定的配置覆盖
    if "dataset_overrides" in system_config and args.dataset in system_config["dataset_overrides"]:
        overrides = system_config["dataset_overrides"][args.dataset]
        # 深度合并覆盖配置（支持嵌套字段覆盖）
        system_config = deep_merge_config(system_config, overrides)
        console.print(f"  🔧 Applied dataset overrides for {args.dataset}: {list(overrides.keys())}")
    
    # ===== 加载数据集 =====
    console.print(f"\n[bold cyan]Loading dataset: {args.dataset}[/bold cyan]")
    
    data_path = dataset_config["data"]["path"]
    if not Path(data_path).is_absolute():
        # 优先从 eval/data/ 加载，如果不存在则从项目根目录加载
        eval_data_path = evaluation_root / "data" / data_path
        root_data_path = evaluation_root.parent / data_path

        if eval_data_path.exists():
            console.print(f"  📂 Using eval/data/{data_path}")
            data_path = eval_data_path
        elif root_data_path.exists():
            console.print(f"  📂 Using project root data/{data_path}")
            data_path = root_data_path
        else:
            console.print(f"[red]❌ Data not found in eval/data/ or project root data/[/red]")
            return
    
    # 智能加载（自动转换）
    dataset = load_dataset(args.dataset, str(data_path))
    
    console.print(f"  ✅ Loaded {len(dataset.conversations)} conversations, {len(dataset.qa_pairs)} QA pairs")
    
    # ===== 确定输出目录 =====
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # 使用简单的默认命名: {dataset}-{system}
        output_dir = evaluation_root / "results" / f"{args.dataset}-{args.system}"
    
    # 🔥 尽早初始化 Logger，以便捕获 Adapter 初始化时的日志
    setup_logger(log_dir=output_dir)
    
    # ===== 创建组件 =====
    console.print(f"\n[bold cyan]Initializing components...[/bold cyan]")
    
    # 创建适配器（传递 output_dir 用于持久化，启用 token 统计）
    adapter = create_adapter(
        system_config["adapter"],
        system_config,
        output_dir=output_dir,
        enable_token_stats=True  # 启用 token 统计
    )
    console.print(f"  ✅ Created adapter: {adapter.get_system_info()['name']}")
    
    # 创建评估器
    evaluator = create_evaluator(
        dataset_config["evaluation"]["type"],
        dataset_config["evaluation"]
    )
    console.print(f"  ✅ Created evaluator: {evaluator.get_name()}")
    
    # 创建 LLM Provider（用于答案生成）
    # 支持嵌套配置结构: llm.service + llm.{service}.* (e.g., llm.openai.model)
    llm_config = system_config.get("llm", {})
    llm_service = llm_config.get("service", "openai")
    llm_provider_config = llm_config.get(llm_service, llm_config)  # 优先嵌套，回退扁平

    llm_provider = LLMProvider(
        provider_type=llm_provider_config.get("provider", "openai"),
        model=llm_provider_config.get("model"),
        api_key=llm_provider_config.get("api_key"),
        base_url=llm_provider_config.get("base_url"),
        temperature=llm_provider_config.get("temperature", 0.0),
        max_tokens=int(llm_provider_config.get("max_tokens", 32768)),
    )
    console.print(f"  ✅ Created LLM provider: {llm_provider_config.get('model')}")
    
    # ===== 创建 Workflow =====
    # 从数据集配置中读取需要过滤的问题类别
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])

    console.print(f"\n[bold cyan]Building evaluation workflow...[/bold cyan]")

    # 注册 eval nodes（触发装饰器）
    import eval.core.workflow_nodes  # noqa

    # 创建 ExecutionContext（包含所有依赖）
    from src.orchestration.context import ExecutionContext
    from eval.utils.checkpoint import CheckpointManager
    from eval.utils.token_stats import TokenStatsCollector

    checkpoint_manager = CheckpointManager(output_dir=output_dir, run_name="default")
    token_stats_collector = TokenStatsCollector()  # 创建 token 统计收集器

    # 将 token_stats_collector 注入到 adapter（如果 adapter 支持）
    if hasattr(adapter, 'set_token_stats_collector'):
        adapter.set_token_stats_collector(token_stats_collector)

    context = ExecutionContext(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        checkpoint_manager=checkpoint_manager,
        logger=setup_logger(log_dir=output_dir),
        console=console,
        project_root=project_root,
        token_stats_collector=token_stats_collector,  # 添加 token 统计收集器
    )

    # 加载 workflow 配置
    from src.orchestration.workflow_builder import WorkflowBuilder
    from src.orchestration.config_loader import ConfigLoader

    # 选择 workflow：优先使用命令行参数，否则根据系统配置自动选择
    if args.workflow:
        workflow_config = args.workflow
    else:
        # 根据系统配置决定使用哪个 workflow
        # 从 group_event_cluster.enabled 字段读取配置
        enable_clustering = system_config.get("group_event_cluster", {}).get("enabled", False)
        if enable_clustering:
            workflow_config = "standard_pipeline"  # 包含 cluster 阶段
        else:
            workflow_config = "no_cluster"  # 跳过 cluster 阶段

    console.print(f"  📋 Workflow: {workflow_config}")
    console.print(f"  ✅ Output: {output_dir}")
    if filter_categories:
        console.print(f"  🔍 Filter categories: {filter_categories}")

    # 获取 enable_clustering 用于 state metadata
    enable_clustering = system_config.get("group_event_cluster", {}).get("enabled", False)

    # 构建 workflow（使用完整路径）
    eval_workflows_dir = project_root / "config" / "eval" / "workflows"
    workflow_file = eval_workflows_dir / f"{workflow_config}.yaml"

    if not workflow_file.exists():
        raise FileNotFoundError(f"Workflow config not found: {workflow_file}")

    loader = ConfigLoader(config_dir=eval_workflows_dir)
    workflow_config_obj = loader.load(workflow_config)

    builder = WorkflowBuilder(context)
    workflow = builder.build_from_config(workflow_config_obj)

    # ===== 数据过滤 =====
    # 🔥 Filter by conversation index if specified (same logic as old Pipeline)
    if args.conv is not None:
        if 0 <= args.conv < len(dataset.conversations):
            selected_conv = dataset.conversations[args.conv]
            dataset.conversations = [selected_conv]

            # Filter QA pairs that belong to this conversation
            target_conv_id = selected_conv.conversation_id
            dataset.qa_pairs = [
                qa for qa in dataset.qa_pairs
                if qa.metadata.get("conversation_id") == target_conv_id
            ]

            console.print(
                f"[dim]🔍 Selected conversation {args.conv}: {target_conv_id} "
                f"({len(dataset.qa_pairs)} QA pairs)[/dim]\n"
            )
        else:
            console.print(
                f"[red]❌ Conversation index {args.conv} out of range "
                f"(0-{len(dataset.conversations)-1})[/red]"
            )
            return

    # Filter by question categories (e.g., filter out Category 5 adversarial questions)
    original_qa_count = len(dataset.qa_pairs)

    if filter_categories:
        # Convert categories to strings (compatible with both int and str configs)
        filter_set = {str(cat) for cat in filter_categories}

        # Filter out questions from specified categories
        dataset.qa_pairs = [
            qa for qa in dataset.qa_pairs
            if qa.category not in filter_set
        ]

        filtered_count = original_qa_count - len(dataset.qa_pairs)

        if filtered_count > 0:
            filtered_categories_str = ", ".join(sorted(filter_set))
            console.print(
                f"[dim]🔍 Filtered out {filtered_count} questions from categories: {filtered_categories_str}[/dim]"
            )
            console.print(f"[dim]   Remaining questions: {len(dataset.qa_pairs)}[/dim]\n")

    # ===== 运行 Workflow =====
    console.print(f"[bold cyan]Running evaluation workflow...[/bold cyan]")

    # 创建初始 state
    from eval.core.workflow_nodes import EvalState

    # 🔥 从 checkpoint 加载已完成的 stages
    completed_stages_from_checkpoint = []
    if checkpoint_manager:
        checkpoint_data = checkpoint_manager.load_checkpoint()
        if checkpoint_data:
            completed_stages_from_checkpoint = checkpoint_data.get("completed_stages", [])
            if completed_stages_from_checkpoint:
                console.print(f"[dim]📋 Resuming from checkpoint: {completed_stages_from_checkpoint}[/dim]")

    initial_state = EvalState(
        dataset=dataset,
        conversations=dataset.conversations,
        qa_pairs=dataset.qa_pairs,
        conv_id=args.conv,
        filter_categories=filter_categories,
        metadata={"group_event_cluster": {"enabled": enable_clustering}},
        completed_stages=completed_stages_from_checkpoint,
    )

    try:
        # 执行 workflow
        final_state = await workflow.ainvoke(initial_state)

        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")

        # 🔥 保存最终的 completed_stages（统一 checkpoint 管理）
        if checkpoint_manager:
            final_completed = set(final_state.get("completed_stages", []))
            checkpoint_manager.save_checkpoint(final_completed)
            console.print(f"[dim]💾 Checkpoint saved: {sorted(list(final_completed))}[/dim]")

        # 打印结果
        if "eval_results" in final_state and final_state["eval_results"]:
            eval_results = final_state["eval_results"]
            if hasattr(eval_results, 'accuracy'):
                console.print(f"  Accuracy: {eval_results.accuracy:.2%}")
                console.print(f"  Correct: {eval_results.correct}/{eval_results.total_questions}")

        # 生成并输出 Token 统计报告
        if token_stats_collector:
            console.print()  # 空行
            token_report = token_stats_collector.generate_report()
            console.print(token_report)

            # 保存 token 统计到文件
            token_stats_file = output_dir / "token_stats.json"
            token_stats_collector.save_to_json(str(token_stats_file))
            console.print(f"[dim]Token statistics saved to: {token_stats_file}[/dim]\n")

        console.print(f"Results saved to: [cyan]{output_dir}[/cyan]\n")
    
    finally:
        # ===== 清理资源 =====
        # 只有使用了 rerank 的系统才需要清理
        systems_need_rerank = ["parallax"]
        if args.system in systems_need_rerank:
            try:
                from retrieval.services import rerank as rerank_service
                reranker = rerank_service.get_rerank_service()
                if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
                    await reranker.close()
                    console.print("[dim]🧹 Cleaned up rerank service resources[/dim]")
            except Exception as e:
                # 如果清理失败也不影响主流程
                console.print(f"[dim]⚠️  Failed to cleanup resources: {e}[/dim]")


if __name__ == "__main__":
    asyncio.run(main())

