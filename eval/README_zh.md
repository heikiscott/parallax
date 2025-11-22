# Parallax 评估框架

一个统一的模块化评估框架，用于在标准数据集上对记忆系统进行基准测试。

## 🌟 核心特性

### 统一且模块化的框架
- **一个代码库适用于所有场景**：无需为每个数据集或系统编写单独的代码
- **即插即用的系统支持**：支持多种记忆系统（Parallax、mem0、memOS、memU 等）
- **多种基准测试**：开箱即用支持 LoCoMo、LongMemEval、PersonaMem
- **一致的评估**：所有系统使用相同的流程和指标进行评估

### 自动兼容性检测
框架会自动检测并适配：
- **多用户 vs 单用户对话**：无缝处理两种对话类型
- **问答 vs 多项选择题**：根据问题格式自适应评估方式
- **有/无时间戳**：支持有或无时间信息的数据

### 强大的检查点系统
- **跨阶段检查点**：可从任何流程阶段恢复（添加 → 搜索 → 回答 → 评估）
- **细粒度恢复**：每个对话（搜索）和每 400 个问题（回答）保存进度


## 🏗️ 架构概览

### 代码结构

```
eval/
├── core/               # 流程编排和数据模型
├── adapters/           # 系统特定实现
├── evaluators/         # 答案评估（LLM 评判、精确匹配）
├── converters/         # 数据集格式转换器
├── utils/              # 配置、日志、I/O
├── config/
│   ├── datasets/       # 数据集配置（locomo.yaml 等）
│   ├── systems/        # 系统配置（parallax.yaml 等）
│   └── prompts.yaml    # 提示词模板
├── data/               # 基准数据集
└── results/            # 评估结果和日志
```

### 流程流转

评估包含 4 个连续阶段：

1. **添加（Add）**：摄取对话并构建索引
2. **搜索（Search）**：为每个问题检索相关记忆
3. **回答（Answer）**：使用检索到的上下文生成答案
4. **评估（Evaluate）**：使用 LLM 评判或精确匹配评估答案质量

每个阶段都会保存其输出，并可独立恢复。

## 🚀 快速开始

### 前置要求

- Python 3.10+
- Parallax 环境已配置（参见主项目的 `env.template`）

### 数据准备

将数据集文件放置在 `eval/data/` 目录中：

**LoCoMo**（原生格式，无需转换）：

```
eval/data/locomo/
└── locomo10.json
```

**LongMemEval**（自动转换为 LoCoMo 格式）：

```
eval/data/longmemeval/
└── longmemeval_s_cleaned.json  # 原始文件
# → 将自动生成：longmemeval_s_locomo_style.json
```

**PersonaMem**（自动转换为 LoCoMo 格式）：

```
eval/data/personamem/
├── questions_32k.csv           # 原始文件
└── shared_contexts_32k.jsonl   # 原始文件
# → 将自动生成：personamem_32k_locomo_style.json
```

框架会在首次运行时自动检测并转换非 LoCoMo 格式。您无需手动运行任何转换脚本。

### 安装

安装评估专用依赖：

```bash
# 用于评估本地系统（Parallax）
uv sync --group evaluation

# 用于评估在线 API 系统（mem0、memOS、memU 等）
uv sync --group evaluation-full
```

### 环境配置

评估框架重用主 Parallax `.env` 文件中的大部分环境变量：
- `LLM_API_KEY`、`LLM_BASE_URL`（用于答案生成）
- `DEEPINFRA_API_KEY`（用于嵌入向量/重排序）

要测试 Parallax，请先配置完整的 .env 文件。

**在线 API 系统的额外变量**（如测试这些系统，请添加到 `.env`）：

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# memOS
MEMOS_KEY=your_memos_api_key

# memU
MEMU_API_KEY=your_memu_api_key
```

### 快速测试（Mini 数据集）

使用 mini 数据集运行快速测试以验证一切正常：

```bash
# 导航到项目根目录
cd /path/to/memsys-opensource

# 运行 mini 数据集（单个对话，有限问题）
uv run python -m eval.cli --dataset locomo-mini --system parallax

# 或使用简化脚本
python eval/run_locomo.py --mini
```

### 测试单个对话

通过索引测试特定对话：

```bash
# 测试索引为 3 的对话
uv run python -m eval.cli --dataset locomo --system parallax --conv 3

# 或使用简化脚本
python eval/run_locomo.py --conv 3
```

### 完整评估

在所有对话上运行完整基准测试：

```bash
# 在 LoCoMo 上评估 Parallax（全部 10 个对话）
uv run python -m eval.cli --dataset locomo --system parallax

# 或使用简化脚本
python eval/run_locomo.py --all

# 评估其他系统
uv run python -m eval.cli --dataset locomo --system memos
uv run python -m eval.cli --dataset locomo --system memu
# 对于 mem0，建议先运行 add，在 Web 控制台检查记忆状态以确保完成，然后运行后续阶段。
uv run python -m eval.cli --dataset locomo --system mem0 --stages add
uv run python -m eval.cli --dataset locomo --system mem0 --stages search answer evaluate

# 在其他数据集上评估
uv run python -m eval.cli --dataset longmemeval --system parallax
uv run python -m eval.cli --dataset personamem --system parallax

# 使用 --run-name 区分多次运行（用于 A/B 测试）
# 结果将保存到：results/{dataset}-{system}-{run-name}/
uv run python -m eval.cli --dataset locomo --system parallax --run-name baseline
uv run python -m eval.cli --dataset locomo --system parallax --run-name experiment1
uv run python -m eval.cli --dataset locomo --system parallax --run-name 20241107

# 如果中断则从检查点恢复（自动）
# 只需重新运行相同命令 - 它会检测并从检查点恢复
uv run python -m eval.cli --dataset locomo --system parallax

```

### 查看结果

结果保存到 `eval/results/{dataset}-{system}[-{run-name}]/`：

```bash
# 查看摘要报告
cat eval/results/locomo-parallax/report.txt

# 查看详细评估结果
cat eval/results/locomo-parallax/eval_results.json

# 查看流程执行日志
cat eval/results/locomo-parallax/pipeline.log
```

**结果文件：**
- `report.txt` - 摘要指标（准确率、总问题数）
- `eval_results.json` - 每个问题的详细评估
- `answer_results.json` - 生成的答案和检索到的上下文
- `search_results.json` - 每个问题检索到的记忆
- `pipeline.log` - 详细执行日志

## 📊 理解结果

### 指标

- **准确率（Accuracy）**：正确答案的百分比（由 LLM 评判）
- **总问题数（Total Questions）**：评估的问题数量
- **正确数（Correct）**：正确回答的问题数量

### 详细结果

查看 `eval_results.json` 获取每个问题的详细信息：

**LoCoMo 示例（问答格式，由 LLM 评判评估）：**

```json
{
  "total_questions": ...,
  "correct": ...,
  "accuracy": ...,
  "detailed_results": {
      "locomo_exp_user_0": [
         {
            "question_id": "locomo_0_qa0",
            "question": "What is my favorite food?",
            "golden_answer": "Pizza",
            "generated_answer": "Your favorite food is pizza.",
            "judgments": [
               true,
               true,
               true
            ],
            "category": "1"
         }
         ...
      ]
  }
}
```

**PersonaMem 示例（多项选择格式，由精确匹配评估）：**

```json
{
  "overall_accuracy": ...,
  "total_questions": ...,
  "correct_count": ...,
  "detailed_results": [
    {
      "question_id": "acd74206-37dc-4756-94a8-b99a395d9a21",
      "question": "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.",
      "golden_answer": "(c)",
      "generated_answer": "(c)",
      "is_correct": true,
      "category": "recall_user_shared_facts"
    }
    ...
  ]
}
```

## 🔧 高级用法

### 运行特定阶段

跳过已完成的阶段以加快迭代：

```bash
# 仅运行搜索阶段（如果添加已完成）
uv run python -m eval.cli --dataset locomo --system parallax --stages search

# 运行搜索、回答和评估（跳过添加）
uv run python -m eval.cli --dataset locomo --system parallax \
    --stages search answer evaluate
```
如果您已经完成了搜索，并希望重新运行，请从 checkpoint_default.json 文件中的 completed_stages 中删除 "search"（以及后续阶段）：
```
  "completed_stages": [
    "answer",
    "search",
    "evaluate",
    "add"
  ]
```


### 自定义配置

修改系统或数据集配置：

```bash
# 复制并编辑配置
cp eval/config/systems/parallax.yaml eval/config/systems/parallax_custom.yaml
# 编辑 parallax_custom.yaml 进行修改

# 使用自定义配置运行
uv run python -m eval.cli --dataset locomo --system parallax_custom
```


## 🔌 支持的系统

### 本地系统
- **parallax** - Parallax，具有 MemCell 提取和双模式检索

### 在线 API 系统
- **mem0** - Mem0 API
- **memos** - memOS API  
- **memu** - memU HTTP API

## 📚 支持的数据集

- **locomo** - LoCoMo：长期对话记忆基准
- **longmemeval** - LongMemEval：扩展对话评估
- **personamem** - PersonaMem：人格一致性评估

## 📄 许可证

与父项目相同。


