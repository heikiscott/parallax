# LoCoMo 评估 Pipeline

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

LoCoMo (Long-Context Modeling) 评估系统，用于测试记忆系统在长对话场景下的检索和问答性能。

---

## 📋 目录结构

```
locomo_eval/
├── config.py                          # 配置文件
├── data/
│   └── locomo10.json                  # 测试数据集
├── prompts/                           # Prompt 模板
│   ├── sufficiency_check.txt          # 充分性检查
│   ├── refined_query.txt              # 查询改进
│   ├── multi_query_generation.txt     # 多查询生成
│   └── answer_prompts.py              # 回答生成
├── stage1_memunits_extraction.py      # 阶段 1：提取 MemUnits
├── stage2_index_building.py           # 阶段 2：构建索引
├── stage3_memory_retrivel.py          # 阶段 3：检索记忆
├── stage4_response.py                 # 阶段 4：生成回答
├── stage5_eval.py                     # 阶段 5：评估结果
└── tools/                             # 辅助工具
    ├── compute_acc.py                 # 准确率计算工具
    └── ...
```

---

## 🚀 快速开始

### 1. 环境配置

确保项目根目录的 `.env` 文件已配置：

```bash
# 必需的环境变量
LLM_API_KEY=your_llm_api_key           # LLM 相关环境变量
DEEPINFRA_API_KEY=your_deepinfra_key   # Embedding/Reranker 相关变量
```

### 2. 修改配置

编辑 `config.py`：

```python
class ExperimentConfig:
    experiment_name: str = "locomo_evaluation"  # 实验名称
    retrieval_mode: str = "lightweight"         # 'agentic' 或 'lightweight'
    # ... 其他配置
```

**关键配置项**：
- **并发数**：根据 API 限制设置并发请求数
- **Embedding 参数**：选择适合的 Embedding 模型和参数
- **Reranker 参数**：配置 Reranker 模型（仅 agentic 模式需要）
- **检索模式**：
  - `agentic`：复杂的多轮检索，质量高但速度慢
  - `lightweight`：快速混合检索，速度快但质量略低

### 3. 运行完整 Pipeline

```bash
# 阶段 1：提取 MemUnits
python eval/locomo_eval/stage1_memunits_extraction.py

# 阶段 2：构建索引
python eval/locomo_eval/stage2_index_building.py

# 阶段 3：检索记忆
python eval/locomo_eval/stage3_memory_retrivel.py

# 阶段 4：生成回答
python eval/locomo_eval/stage4_response.py

# 阶段 5：评估结果
python eval/locomo_eval/stage5_eval.py
```

### 4. 查看结果

```bash
# 查看最终评估结果
cat results/locomo_eval/judged.json

# 查看准确率统计
python eval/locomo_eval/tools/compute_acc.py
```

---

## 📊 结果说明

### 输出目录结构

```
results/locomo_eval/
├── memunits/                  # MemUnit 提取结果
│   ├── memunit_list_conv_0.json
│   └── ...
├── bm25_index/                # BM25 索引
│   └── *.pkl
├── vectors/                   # Embedding 索引
│   └── *.pkl
├── search_results.json        # 检索结果
├── responses.json             # 生成的回答
└── judged.json                # 最终评估结果
```

---

## ⚙️ 配置说明

### 切换检索模式

在 `config.py` 中修改：

```python
# 轻量级检索（快速）
retrieval_mode: str = "lightweight"

# Agentic 检索（高质量）
retrieval_mode: str = "agentic"
```

### 切换 LLM 服务

修改 `config.py`：

```python
llm_service: str = "openai"  # 或 "openrouter", "deepseek"

llm_config: dict = {
    "openai": {
        "model": "openai/gpt-4o-mini",
        "api_key": os.getenv("LLM_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.3,
        "max_tokens": 16384,
    }
}
```

---

## 🔗 相关文档

- [项目根目录 README](../../README_zh.md)
- [开发文档](../../docs/dev_docs/getting_started.md)
- [API 文档](../../docs/api_docs/)
