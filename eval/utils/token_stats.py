"""
Token Statistics Collector

用于统计和跟踪评测过程中各阶段的 token 使用情况。
"""
from typing import Dict, List, Optional
from collections import defaultdict
import json
import contextvars

# Context variable for tracking current stage
_current_stage: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_stage', default=None
)


class TokenStatsCollector:
    """
    Token 统计收集器

    功能：
    1. 收集各阶段（add, cluster, search, answer）的 token 使用情况
    2. 统计每个阶段的总 tokens、平均 tokens、问题数量
    3. 生成统计报告
    4. 支持自动推断当前阶段（通过 context variable）
    """

    def __init__(self):
        """初始化统计收集器"""
        # 结构: {stage: [{"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}, ...]}
        self.stage_stats: Dict[str, List[Dict]] = defaultdict(list)

    def record(self, stage: Optional[str] = None, stats: Optional[Dict] = None) -> None:
        """
        记录一次 LLM 调用的 token 使用情况

        Args:
            stage: 阶段名称 (add, cluster, search, answer)
                   如果为 None，则尝试从 context variable 中获取
            stats: token 统计信息，格式:
                   {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
        """
        if stats is None:
            return

        # 确保包含必要字段
        if "total_tokens" not in stats:
            return

        # 如果未指定 stage，尝试从 context variable 获取
        if stage is None:
            stage = _current_stage.get()
            if stage is None:
                # 无法确定阶段，记录为 "unknown"
                stage = "unknown"

        self.stage_stats[stage].append({
            "prompt_tokens": stats.get("prompt_tokens", 0),
            "completion_tokens": stats.get("completion_tokens", 0),
            "total_tokens": stats.get("total_tokens", 0),
        })

    @staticmethod
    def set_current_stage(stage: Optional[str]) -> None:
        """
        设置当前阶段（用于自动推断）

        Args:
            stage: 阶段名称，或 None 表示清除
        """
        _current_stage.set(stage)

    @staticmethod
    def get_current_stage() -> Optional[str]:
        """获取当前阶段"""
        return _current_stage.get()

    def get_stage_summary(self, stage: str) -> Dict:
        """
        获取指定阶段的统计摘要

        Args:
            stage: 阶段名称

        Returns:
            统计摘要字典，包含 total_calls, total_tokens, avg_tokens, total_prompt_tokens, total_completion_tokens
        """
        stats_list = self.stage_stats.get(stage, [])

        if not stats_list:
            return {
                "total_calls": 0,
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "avg_prompt_tokens": 0,
                "avg_completion_tokens": 0,
                "avg_total_tokens": 0,
            }

        total_calls = len(stats_list)
        total_prompt = sum(s["prompt_tokens"] for s in stats_list)
        total_completion = sum(s["completion_tokens"] for s in stats_list)
        total_tokens = sum(s["total_tokens"] for s in stats_list)

        return {
            "total_calls": total_calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": total_prompt / total_calls,
            "avg_completion_tokens": total_completion / total_calls,
            "avg_total_tokens": total_tokens / total_calls,
        }

    def get_all_summaries(self) -> Dict[str, Dict]:
        """
        获取所有阶段的统计摘要

        Returns:
            {stage_name: summary_dict} 的字典
        """
        return {
            stage: self.get_stage_summary(stage)
            for stage in self.stage_stats.keys()
        }

    def generate_report(self) -> str:
        """
        生成可读的统计报告

        Returns:
            格式化的报告字符串
        """
        lines = []
        lines.append("=" * 70)
        lines.append("📊 Token Usage Statistics by Stage")
        lines.append("=" * 70)
        lines.append("")

        # 按阶段顺序输出
        stage_order = ["add", "cluster", "search", "answer"]
        stage_names = {
            "add": "Add (MemUnit Extraction)",
            "cluster": "Cluster (Event Clustering)",
            "search": "Search (Query Classification/Rewrite)",
            "answer": "Answer (Response Generation)",
        }

        total_all_tokens = 0
        total_all_calls = 0

        for stage in stage_order:
            if stage not in self.stage_stats:
                continue

            summary = self.get_stage_summary(stage)
            if summary["total_calls"] == 0:
                continue

            stage_name = stage_names.get(stage, stage.capitalize())
            lines.append(f"🔹 {stage_name}")
            lines.append(f"   Total LLM Calls:      {summary['total_calls']:,}")
            lines.append(f"   Total Tokens:         {summary['total_tokens']:,}")
            lines.append(f"     - Prompt Tokens:    {summary['total_prompt_tokens']:,}")
            lines.append(f"     - Completion Tokens: {summary['total_completion_tokens']:,}")
            lines.append(f"   Avg Tokens per Call:  {summary['avg_total_tokens']:.1f}")
            lines.append(f"     - Avg Prompt:       {summary['avg_prompt_tokens']:.1f}")
            lines.append(f"     - Avg Completion:   {summary['avg_completion_tokens']:.1f}")
            lines.append("")

            total_all_tokens += summary["total_tokens"]
            total_all_calls += summary["total_calls"]

        # 总计
        if total_all_calls > 0:
            lines.append("─" * 70)
            lines.append(f"📈 Overall Summary")
            lines.append(f"   Total LLM Calls:      {total_all_calls:,}")
            lines.append(f"   Total Tokens:         {total_all_tokens:,}")
            lines.append(f"   Avg Tokens per Call:  {total_all_tokens / total_all_calls:.1f}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        转换为字典格式（用于保存到 JSON）

        Returns:
            包含所有统计信息的字典
        """
        return {
            "summaries": self.get_all_summaries(),
            "raw_data": dict(self.stage_stats),
        }

    def save_to_json(self, filepath: str) -> None:
        """
        保存统计数据到 JSON 文件

        Args:
            filepath: 保存路径
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
