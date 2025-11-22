# LoCoMo Evaluation Pipeline

<p align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
</p>

LoCoMo (Long-Context Modeling) evaluation system for testing memory system performance in long conversation scenarios, including retrieval and question-answering capabilities.

---

## 📋 Directory Structure

```
locomo_eval/
├── config.py                          # Configuration file
├── data/
│   └── locomo10.json                  # Test dataset
├── prompts/                           # Prompt templates
│   ├── sufficiency_check.txt          # Sufficiency check
│   ├── refined_query.txt              # Query refinement
│   ├── multi_query_generation.txt     # Multi-query generation
│   └── answer_prompts.py              # Answer generation
├── stage1_memcells_extraction.py      # Stage 1: Extract MemCells
├── stage2_index_building.py           # Stage 2: Build indexes
├── stage3_memory_retrivel.py          # Stage 3: Retrieve memories
├── stage4_response.py                 # Stage 4: Generate responses
├── stage5_eval.py                     # Stage 5: Evaluate results
└── tools/                             # Utility tools
    ├── agentic_utils.py               # Agentic retrieval utilities
    ├── benchmark_embedding.py         # Embedding performance testing
    └── ...
```

---

## 🚀 Quick Start

### 1. Environment Setup

Ensure the `.env` file in the project root directory is configured:

```bash
# Required environment variables
LLM_API_KEY=your_llm_api_key           # LLM API key
DEEPINFRA_API_KEY=your_deepinfra_key   # Embedding/Reranker API key
```

### 2. Modify Configuration

Edit `config.py`:

```python
class ExperimentConfig:
    experiment_name: str = "locomo_evaluation"  # Experiment name
    retrieval_mode: str = "lightweight"         # 'agentic' or 'lightweight'
    # ... other configurations
```

**Key Configuration Options**:
- **Concurrency**: Set concurrent requests based on API limits
- **Embedding Parameters**: Choose appropriate embedding model and parameters
- **Reranker Parameters**: Configure reranker model (only for agentic mode)
- **Retrieval Mode**:
  - `agentic`: Complex multi-round retrieval, high quality but slower
  - `lightweight`: Fast hybrid retrieval, faster but slightly lower quality

### 3. Run Complete Pipeline

```bash
# Stage 1: Extract MemCells
python eval/locomo_eval/stage1_memcells_extraction.py

# Stage 2: Build indexes
python eval/locomo_eval/stage2_index_building.py

# Stage 3: Retrieve memories
python eval/locomo_eval/stage3_memory_retrivel.py

# Stage 4: Generate responses
python eval/locomo_eval/stage4_response.py

# Stage 5: Evaluate results
python eval/locomo_eval/stage5_eval.py
```

### 4. View Results

```bash
# View final evaluation results
cat results/locomo_eval/judged.json

# View accuracy statistics
python eval/locomo_eval/tools/compute_acc.py
```

---

## 📊 Results Overview

### Output Directory Structure

```
results/locomo_eval/
├── memcells/                  # MemCell extraction results
│   ├── memcell_list_conv_0.json
│   └── ...
├── bm25_index/                # BM25 indexes
│   └── *.pkl
├── vectors/                   # Embedding indexes
│   └── *.pkl
├── search_results.json        # Retrieval results
├── responses.json             # Generated responses
└── judged.json                # Final evaluation results
```

---

## ⚙️ Configuration Guide

### Switch Retrieval Mode

Modify in `config.py`:

```python
# Lightweight retrieval (fast)
retrieval_mode: str = "lightweight"

# Agentic retrieval (high quality)
retrieval_mode: str = "agentic"
```

### Switch LLM Service

Modify `config.py`:

```python
llm_service: str = "openai"  # or "openrouter", "deepseek"

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

## 🔗 Related Documentation

- [Project Root README](../../README.md)
- [Development Guide](../../docs/dev_docs/getting_started.md)
- [API Documentation](../../docs/api_docs/)
