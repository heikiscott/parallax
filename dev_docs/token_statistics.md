# Token Statistics 功能文档

## 概述

Token Statistics 功能用于统计和跟踪评测过程中各阶段的 LLM token 使用情况，帮助分析成本和性能。

## 核心特性

### ✅ 自动收集
- 在 `LLMProvider.generate()` 层面自动收集，无需手动添加统计代码
- 使用回调机制，低耦合设计
- 支持所有使用 `LLMProvider` 的代码路径

### ✅ 阶段追踪
- 使用 Python `contextvars` 自动推断当前阶段
- 支持的阶段：`add`, `cluster`, `search`, `answer`
- 可扩展到自定义阶段

### ✅ 鲁棒性
- 无论 LangGraph workflow 如何变化，都能正确统计
- 向后兼容，不影响不使用统计的场景
- 捕获所有 LLM 调用（包括 EventLogExtractor、问题分类、查询改写等）

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                   Workflow Node                         │
│  (设置阶段: TokenStatsCollector.set_current_stage())   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   LLMProvider                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │ async def generate():                            │  │
│  │   result = await self.provider.generate(...)    │  │
│  │   if self.stats_callback:                       │  │
│  │       stats = get_current_call_stats()          │  │
│  │       self.stats_callback(stats)  # 自动回调   │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│             TokenStatsCollector                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │ def record(stats):                               │  │
│  │   stage = _current_stage.get()  # 从上下文获取 │  │
│  │   self.stage_stats[stage].append(stats)         │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 使用方法

### 1. 基本使用（已自动启用）

运行评测时，token 统计已默认启用：

```bash
python eval/cli.py --dataset locomo-mini --system parallax
```

评测完成后会输出：

```
======================================================================
📊 Token Usage Statistics by Stage
======================================================================

🔹 Add (MemUnit Extraction)
   Total LLM Calls:      150
   Total Tokens:         1,234,567
     - Prompt Tokens:    987,654
     - Completion Tokens: 246,913
   Avg Tokens per Call:  8,230.4
     - Avg Prompt:       6,584.4
     - Avg Completion:   1,646.1

🔹 Search (Query Classification/Rewrite)
   Total LLM Calls:      45
   Total Tokens:         123,456
     - Prompt Tokens:    98,765
     - Completion Tokens: 24,691
   Avg Tokens per Call:  2,745.7
     - Avg Prompt:       2,195.0
     - Avg Completion:   548.7

🔹 Answer (Response Generation)
   Total LLM Calls:      30
   Total Tokens:         456,789
     - Prompt Tokens:    345,678
     - Completion Tokens: 111,111
   Avg Tokens per Call:  15,226.3
     - Avg Prompt:       11,522.6
     - Avg Completion:   3,703.7

──────────────────────────────────────────────────────────────────────
📈 Overall Summary
   Total LLM Calls:      225
   Total Tokens:         1,814,812
   Avg Tokens per Call:  8,066.3

======================================================================
```

统计结果也会保存到 `eval/results/{dataset}-{system}/token_stats.json`。

### 2. 在自定义 Workflow Node 中使用

如果你创建了自定义的 LangGraph workflow node，只需设置当前阶段：

```python
from eval.utils.token_stats import TokenStatsCollector
from src.orchestration.nodes import register_node

@register_node("my_custom_stage")
async def my_custom_stage_node(state, context):
    """自定义阶段"""
    # 设置当前阶段（用于 token 统计）
    TokenStatsCollector.set_current_stage("my_custom_stage")

    try:
        # 调用任何使用 llm_provider 的代码
        # 所有 LLM 调用都会自动被归类到 "my_custom_stage"
        result = await some_function(context.llm_provider)

        return {
            "result": result,
            "completed_stages": ["my_custom_stage"]
        }
    finally:
        # 清理阶段标记
        TokenStatsCollector.set_current_stage(None)
```

### 3. 手动指定阶段（不推荐）

如果确实需要手动指定阶段（绕过 context variable），可以：

```python
# 在 adapter 或其他代码中
if self.token_stats_collector:
    stats = self.llm_provider.get_current_call_stats()
    self.token_stats_collector.record(stage="custom_stage", stats=stats)
```

**注意：** 这种方式需要手动在每个 LLM 调用后添加代码，不推荐使用。优先使用方法 2（设置 context variable）。

## 数据格式

### JSON 输出格式

`token_stats.json` 文件格式：

```json
{
  "summaries": {
    "add": {
      "total_calls": 150,
      "total_prompt_tokens": 987654,
      "total_completion_tokens": 246913,
      "total_tokens": 1234567,
      "avg_prompt_tokens": 6584.4,
      "avg_completion_tokens": 1646.1,
      "avg_total_tokens": 8230.4
    },
    "search": {
      "total_calls": 45,
      "total_prompt_tokens": 98765,
      "total_completion_tokens": 24691,
      "total_tokens": 123456,
      "avg_prompt_tokens": 2195.0,
      "avg_completion_tokens": 548.7,
      "avg_total_tokens": 2745.7
    },
    "answer": {
      "total_calls": 30,
      "total_prompt_tokens": 345678,
      "total_completion_tokens": 111111,
      "total_tokens": 456789,
      "avg_prompt_tokens": 11522.6,
      "avg_completion_tokens": 3703.7,
      "avg_total_tokens": 15226.3
    }
  },
  "raw_data": {
    "add": [
      {"prompt_tokens": 5000, "completion_tokens": 1500, "total_tokens": 6500},
      {"prompt_tokens": 5200, "completion_tokens": 1600, "total_tokens": 6800},
      ...
    ],
    "search": [...],
    "answer": [...]
  }
}
```

## 实现细节

### 核心类

#### 1. `TokenStatsCollector` (eval/utils/token_stats.py)

```python
class TokenStatsCollector:
    def record(self, stage: Optional[str] = None, stats: Optional[Dict] = None):
        """记录 token 使用情况"""

    def get_stage_summary(self, stage: str) -> Dict:
        """获取指定阶段的统计摘要"""

    def generate_report(self) -> str:
        """生成可读的统计报告"""

    def save_to_json(self, filepath: str):
        """保存统计数据到 JSON 文件"""

    @staticmethod
    def set_current_stage(stage: Optional[str]):
        """设置当前阶段（使用 contextvars）"""
```

#### 2. `LLMProvider` (src/providers/llm/llm_provider.py)

```python
class LLMProvider:
    def __init__(
        self,
        provider_type: str,
        enable_stats: bool = False,
        stats_callback: Optional[Callable[[dict], None]] = None,
        **kwargs
    ):
        """初始化 LLM Provider，支持统计回调"""

    async def generate(self, prompt, ...):
        """生成文本，自动调用统计回调"""
        result = await self.provider.generate(...)

        # 自动收集统计
        if self.enable_stats and self.stats_callback:
            stats = self.provider.get_current_call_stats()
            if stats:
                self.stats_callback(stats)

        return result
```

### Context Variable 机制

使用 Python 的 `contextvars` 模块来跟踪当前阶段，这是线程安全和异步安全的：

```python
import contextvars

_current_stage = contextvars.ContextVar('current_stage', default=None)

# 在 workflow node 中设置
TokenStatsCollector.set_current_stage("answer")

# 在 callback 中自动获取
stage = _current_stage.get()  # 返回 "answer"
```

**优势：**
- 异步安全：每个异步任务有独立的上下文
- 无需手动传递：自动继承到子调用
- 自动清理：使用 try-finally 确保清理

## 常见问题

### Q1: 为什么有些 LLM 调用被归类到 "unknown"？

**A:** 这说明调用发生时没有设置 stage。检查：
1. 是否在 workflow node 中调用了 `TokenStatsCollector.set_current_stage()`
2. 是否在 finally 块中清理了 stage
3. 是否在异步上下文中正确传递了 context variable

### Q2: 如何禁用 token 统计？

**A:** 在 `cli.py` 中修改：

```python
# 将 enable_token_stats=True 改为 False
adapter = create_adapter(
    system_config["adapter"],
    system_config,
    output_dir=output_dir,
    enable_token_stats=False  # 禁用统计
)
```

### Q3: 统计会影响性能吗？

**A:** 几乎没有影响：
- 统计收集是纯内存操作（字典追加）
- 仅在 LLM 调用完成后执行（不阻塞主流程）
- Overhead < 1ms per call

### Q4: 如何为新的自定义 workflow 添加统计？

**A:** 遵循以下模板：

```python
@register_node("my_stage")
async def my_stage_node(state, context):
    TokenStatsCollector.set_current_stage("my_stage")
    try:
        # 你的代码
        result = await my_function(context.llm_provider)
        return {"result": result}
    finally:
        TokenStatsCollector.set_current_stage(None)
```

## 扩展和定制

### 添加新的统计维度

修改 `TokenStatsCollector.record()` 来收集额外信息：

```python
def record(self, stage: Optional[str] = None, stats: Optional[Dict] = None) -> None:
    # ... 现有代码 ...

    self.stage_stats[stage].append({
        "prompt_tokens": stats.get("prompt_tokens", 0),
        "completion_tokens": stats.get("completion_tokens", 0),
        "total_tokens": stats.get("total_tokens", 0),
        # 新增：记录时间戳
        "timestamp": time.time(),
        # 新增：记录模型
        "model": stats.get("model", "unknown"),
    })
```

### 自定义报告格式

修改 `TokenStatsCollector.generate_report()` 来定制输出格式。

## 总结

Token Statistics 功能提供了一个**鲁棒、低耦合、易扩展**的方案来追踪 LLM token 使用情况：

✅ **零侵入**：无需在业务代码中手动添加统计
✅ **自动化**：通过回调和 context variable 自动收集
✅ **可扩展**：支持任意自定义 workflow 和阶段
✅ **鲁棒性**：无论流程如何变化，都能正确统计

---

**最后更新：** 2025-12-10
**作者：** Claude Code
