"""
统一配置模块

所有配置文件都在此目录下管理，提供统一的加载接口。

目录结构:
    config/
    ├── __init__.py              # 本文件，导出公共 API
    ├── loader.py                # 配置加载工具
    ├── eval/                    # 评估相关配置
    │   ├── datasets/            # 数据集配置
    │   │   ├── locomo.yaml
    │   │   ├── personamem.yaml
    │   │   └── longmemeval.yaml
    │   └── systems/             # 系统配置
    │       └── parallax.yaml
    ├── src/                     # src 相关配置
    │   └── llm_backends.yaml
    └── workflows/               # LangGraph 工作流配置
        └── adaptive_retrieval.yaml

使用示例:
    from config import load_config

    # 加载系统配置
    config = load_config("eval/systems/parallax")
    print(config.retrieval.mode)

    # 加载数据集配置
    dataset = load_config("eval/datasets/locomo")
    print(dataset.data.path)
"""

from config.loader import (
    ConfigDict,
    load_config,
    load_yaml,
    reload_config,
    save_yaml,
)

__all__ = [
    "ConfigDict",
    "load_config",
    "load_yaml",
    "reload_config",
    "save_yaml",
]
