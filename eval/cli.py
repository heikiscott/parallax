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

# 2. 加载环境变量
from utils.load_env import setup_environment
setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

# ===== 现在可以安全地导入 Parallax 组件 =====
from eval.core.loaders import load_dataset
from eval.core.pipeline import Pipeline
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
        "--stages",
        nargs="+",
        default=None,
        help="Stages to run (add, search, answer, evaluate). Default: all"
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
    
    # 创建适配器（传递 output_dir 用于持久化）
    adapter = create_adapter(
        system_config["adapter"],
        system_config,
        output_dir=output_dir
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
    
    # ===== 创建 Pipeline =====
    # 从数据集配置中读取需要过滤的问题类别
    filter_categories = dataset_config.get("evaluation", {}).get("filter_category", [])
    
    pipeline = Pipeline(
        adapter=adapter,
        evaluator=evaluator,
        llm_provider=llm_provider,
        output_dir=output_dir,
        filter_categories=filter_categories
    )
    
    console.print(f"  ✅ Created pipeline, output: {output_dir}")
    if filter_categories:
        console.print(f"  📋 Filter categories: {filter_categories}")
    
    # ===== 运行 Pipeline =====
    try:
        results = await pipeline.run(
            dataset=dataset,
            stages=args.stages,
            conv_id=args.conv,
        )
        
        console.print(f"\n[bold green]✨ Evaluation completed![/bold green]")
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

