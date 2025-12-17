"""
统一配置模块

所有配置文件都在此目录下管理，提供统一的加载接口。

目录结构:
    config/
    ├── __init__.py              # 本文件，导出公共 API
    ├── loader.py                # 配置加载工具
    ├── app.yaml                 # 应用级配置
    ├── secrets/                 # 敏感信息配置
    │   ├── secrets.template.yaml  # 模板文件（提交到 git）
    │   └── secrets.yaml           # 实际密钥（gitignore）
    ├── services/                # 服务/组件配置
    │   ├── providers.yaml       # LLM Provider 配置
    │   ├── databases.yaml       # 数据库配置
    │   ├── embedding.yaml       # Embedding 服务配置
    │   ├── rerank.yaml          # Rerank 服务配置
    │   ├── colbert.yaml         # ColBERT 模型配置
    │   └── llm_backends.yaml    # LLM 后端配置
    ├── workflows/               # 工作流配置
    │   ├── agentic_hybrid.yaml
    │   ├── adaptive_retrieval.yaml
    │   └── simple_retrieval.yaml
    └── eval/                    # 评估相关配置
        ├── datasets/            # 数据集配置
        ├── systems/             # 系统配置
        └── workflows/           # 评估专用工作流

使用示例:
    from config import load_config, load_secrets

    # 加载系统配置
    config = load_config("eval/systems/parallax")
    print(config.retrieval.mode)

    # 直接访问 secrets
    secrets = load_secrets()
    print(secrets.openai_api_key)
"""

from config.loader import (
    ConfigDict,
    load_config,
    load_raw_file,
    load_secrets,
    load_yaml,
    reload_config,
    save_yaml,
)

__all__ = [
    "ConfigDict",
    "load_config",
    "load_raw_file",
    "load_secrets",
    "load_yaml",
    "reload_config",
    "save_yaml",
]
