#!/usr/bin/env python3
"""
汇总评估结果脚本

功能：
1. 把 eval/results/ 里面单个 conv 跑的结果汇总成总结果
2. 每个 conv 如果有多轮的话直接选择最新的（例如 conv0, conv0-1, conv0-2 选择 conv0-2）
3. 生成包含总分数、具体路径、错题列表、分类统计的全面报告
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


def find_latest_conv_dirs(results_dir: Path) -> dict[int, Path]:
    """
    找到每个 conv 的最新目录。

    例如：conv0, conv0-1, conv0-2 -> 选择 conv0-2

    Returns:
        dict: {conv_id: latest_dir_path}
    """
    conv_pattern = re.compile(r'^locomo-conv(\d+)(?:-(\d+))?$')

    # 收集所有 conv 目录
    conv_dirs: dict[int, list[tuple[int, Path]]] = defaultdict(list)

    for d in results_dir.iterdir():
        if not d.is_dir():
            continue

        match = conv_pattern.match(d.name)
        if match:
            conv_id = int(match.group(1))
            version = int(match.group(2)) if match.group(2) else 0
            conv_dirs[conv_id].append((version, d))

    # 选择每个 conv 的最新版本
    latest_dirs = {}
    for conv_id, versions in conv_dirs.items():
        # 按版本号排序，选择最大的
        versions.sort(key=lambda x: x[0], reverse=True)
        latest_dirs[conv_id] = versions[0][1]

    return latest_dirs


def load_eval_results(eval_file: Path) -> Optional[dict]:
    """加载评估结果 JSON 文件"""
    if not eval_file.exists():
        return None

    try:
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  无法加载 {eval_file}: {e}")
        return None


def aggregate_results(latest_dirs: dict[int, Path]) -> dict:
    """
    汇总所有 conv 的结果

    Returns:
        dict: 包含汇总统计和详细信息的字典
    """
    total_questions = 0
    total_correct = 0

    # 按类别统计
    category_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

    # 错题列表
    wrong_answers = []

    # 各 conv 的详细信息
    conv_details = {}

    # 按 conv_id 排序处理
    for conv_id in sorted(latest_dirs.keys()):
        dir_path = latest_dirs[conv_id]
        eval_file = dir_path / 'eval_results.json'

        results = load_eval_results(eval_file)
        if results is None:
            conv_details[conv_id] = {
                'path': str(dir_path),
                'status': 'missing',
                'total': 0,
                'correct': 0,
                'accuracy': 0.0
            }
            continue

        conv_total = results.get('total_questions', 0)
        conv_correct = results.get('correct', 0)
        conv_accuracy = results.get('accuracy', 0.0)

        total_questions += conv_total
        total_correct += conv_correct

        conv_details[conv_id] = {
            'path': str(dir_path),
            'status': 'ok',
            'total': conv_total,
            'correct': conv_correct,
            'accuracy': conv_accuracy
        }

        # 处理详细结果
        detailed_results = results.get('detailed_results', {})
        for user_id, questions in detailed_results.items():
            for q in questions:
                category = str(q.get('category', 'unknown'))
                category_stats[category]['total'] += 1

                # 判断是否正确（多数判决）
                judgments = q.get('llm_judgments', {})
                true_count = sum(1 for v in judgments.values() if v)
                is_correct = true_count >= 2

                if is_correct:
                    category_stats[category]['correct'] += 1
                else:
                    wrong_answers.append({
                        'conv_id': conv_id,
                        'question_id': q.get('question_id', 'unknown'),
                        'question': q.get('question', ''),
                        'golden_answer': q.get('golden_answer', ''),
                        'generated_answer': q.get('generated_answer', ''),
                        'category': category,
                        'judgments': judgments
                    })

    # 计算总准确率
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0

    return {
        'total_questions': total_questions,
        'total_correct': total_correct,
        'overall_accuracy': overall_accuracy,
        'conv_details': conv_details,
        'category_stats': dict(category_stats),
        'wrong_answers': wrong_answers
    }


def get_category_name(category: str) -> str:
    """
    获取类别的可读名称

    LoCoMo 数据集类别定义（根据 evidence 数量和问题内容分析）:
    - Category 1: Multi-hop (多跳) - 平均 3+ 个 evidence，需要综合多处信息
    - Category 2: Temporal (时序) - 91% 是 When 问题，时间相关推理
    - Category 3: Commonsense (常识) - 需要常识或世界知识推理
    - Category 4: Single-hop (单跳) - 95% 单证据，直接事实查询
    - Category 5: Adversarial (对抗性) - 不可回答的问题
    """
    category_names = {
        '1': 'Multi-hop (多跳)',
        '2': 'Temporal (时序)',
        '3': 'Commonsense (常识)',
        '4': 'Single-hop (单跳)',
        '5': 'Adversarial (对抗性)',
        'unknown': 'Unknown (未知)'
    }
    return category_names.get(category, f'Category {category}')


def generate_summary(aggregated: dict) -> str:
    """生成摘要报告（用于屏幕打印）"""
    lines = []

    # 标题
    lines.append("=" * 70)
    lines.append("📊 LoCoMo 评估汇总报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 总体统计
    lines.append("-" * 70)
    lines.append("📈 总体统计")
    lines.append("-" * 70)
    lines.append(f"总问题数: {aggregated['total_questions']}")
    lines.append(f"正确数:   {aggregated['total_correct']}")
    lines.append(f"准确率:   {aggregated['overall_accuracy']:.2%}")
    lines.append(f"错题数:   {len(aggregated['wrong_answers'])}")
    lines.append("")

    # 各 conv 详情
    lines.append("-" * 70)
    lines.append("📁 各对话详情")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'Conv':<8} {'状态':<10} {'正确/总数':<12} {'准确率':<10} 路径")
    lines.append("-" * 70)

    for conv_id in sorted(aggregated['conv_details'].keys()):
        detail = aggregated['conv_details'][conv_id]
        status = '✅' if detail['status'] == 'ok' else '❌'
        if detail['status'] == 'ok':
            score = f"{detail['correct']}/{detail['total']}"
            accuracy = f"{detail['accuracy']:.2%}"
        else:
            score = "-"
            accuracy = "-"

        # 提取相对路径
        path = detail['path']
        if 'eval/results/' in path:
            path = path.split('eval/results/')[-1]
        elif 'eval\\results\\' in path:
            path = path.split('eval\\results\\')[-1]

        lines.append(f"conv{conv_id:<4} {status:<10} {score:<12} {accuracy:<10} {path}")

    lines.append("")

    # 按类别统计
    lines.append("-" * 70)
    lines.append("📊 按类别统计")
    lines.append("-" * 70)
    lines.append("")
    lines.append(f"{'类别':<30} {'正确/总数':<15} {'准确率':<10}")
    lines.append("-" * 70)

    for category in sorted(aggregated['category_stats'].keys()):
        stats = aggregated['category_stats'][category]
        cat_name = get_category_name(category)
        score = f"{stats['correct']}/{stats['total']}"
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        lines.append(f"{cat_name:<30} {score:<15} {accuracy:.2%}")

    lines.append("")
    lines.append("=" * 70)

    return '\n'.join(lines)


def generate_full_report(aggregated: dict) -> str:
    """生成完整报告（包含按类别归类的错题列表，用于保存到文件）"""
    lines = [generate_summary(aggregated)]

    # 按类别分组错题
    from collections import defaultdict
    wrong_by_category = defaultdict(list)
    for wrong in aggregated['wrong_answers']:
        wrong_by_category[wrong['category']].append(wrong)

    # 错题列表（按类别归类）
    lines.append("")
    lines.append("-" * 70)
    lines.append(f"❌ 错题列表 (共 {len(aggregated['wrong_answers'])} 题)")
    lines.append("-" * 70)

    total_idx = 0
    for category in sorted(wrong_by_category.keys()):
        wrongs = wrong_by_category[category]
        cat_name = get_category_name(category)

        lines.append("")
        lines.append(f"### {cat_name} ({len(wrongs)} 题)")
        lines.append("")

        for wrong in wrongs:
            total_idx += 1
            lines.append(f"[{total_idx}] {wrong['question_id']} (conv{wrong['conv_id']})")
            lines.append(f"    问题: {wrong['question']}")
            lines.append(f"    标准答案: {wrong['golden_answer']}")
            generated = wrong.get('generated_answer', '')
            if len(generated) > 200:
                lines.append(f"    生成答案: {generated[:200]}...")
            else:
                lines.append(f"    生成答案: {generated}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("报告结束")
    lines.append("=" * 70)

    return '\n'.join(lines)


def save_results(aggregated: dict, report: str, output_dir: Path):
    """保存结果到输出目录"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存 JSON 结果
    json_file = output_dir / 'aggregated_results.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 结果已保存到: {json_file}")

    # 保存文本报告
    report_file = output_dir / 'aggregated_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 文本报告已保存到: {report_file}")


def find_output_dir(base_name: str, results_dir: Path) -> Path:
    """
    找到下一个可用的输出目录。

    第一次运行: base_name (e.g., aggregated)
    第二次运行: base_name-1 (e.g., aggregated-1)
    第三次运行: base_name-2 (e.g., aggregated-2)
    """
    base_dir = results_dir / base_name

    if not base_dir.exists():
        return base_dir

    # 找到下一个可用的编号
    counter = 1
    while (results_dir / f"{base_name}-{counter}").exists():
        counter += 1

    return results_dir / f"{base_name}-{counter}"


def main():
    parser = argparse.ArgumentParser(
        description="汇总 LoCoMo 评估结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    汇总所有 conv 结果，输出到 eval/results/aggregated/
  %(prog)s --output custom    输出到 eval/results/custom/ 目录

注意: 每次运行会自动创建新目录 (aggregated, aggregated-1, aggregated-2, ...)
        """
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default='aggregated',
        help='输出目录基础名称 (默认: aggregated)'
    )

    args = parser.parse_args()

    # 路径设置
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    output_dir = find_output_dir(args.output, results_dir)  # 自动版本化

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        sys.exit(1)

    print("🔍 扫描评估结果目录...")
    print(f"   源目录: {results_dir}")
    print("")

    # 找到最新的 conv 目录
    latest_dirs = find_latest_conv_dirs(results_dir)

    if not latest_dirs:
        print("❌ 未找到任何 conv 评估结果")
        sys.exit(1)

    print(f"📁 找到 {len(latest_dirs)} 个对话的评估结果:")
    for conv_id in sorted(latest_dirs.keys()):
        print(f"   conv{conv_id}: {latest_dirs[conv_id].name}")
    print("")

    # 汇总结果
    print("📊 正在汇总结果...")
    aggregated = aggregate_results(latest_dirs)

    # 生成报告
    summary = generate_summary(aggregated)
    full_report = generate_full_report(aggregated)

    # 保存结果（完整报告含错题列表）
    print("")
    save_results(aggregated, full_report, output_dir)

    # 打印摘要到屏幕（不含错题详情）
    print("")
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
