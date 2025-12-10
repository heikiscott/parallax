"""
统一配置模块

所有配置文件都在此目录下管理，提供统一的加载接口。

目录结构:
    config/
    ├── __init__.py              # 本文件，导出公共 API
    ├── loader.py                # 配置加载工具
    ├── secrets/                 # 敏感信息配置
    │   ├── secrets.template.yaml  # 模板文件（提交到 git）
    │   └── secrets.yaml           # 实际密钥（gitignore）
    ├── eval/                    # 评估相关配置
    │   ├── datasets/            # 数据集配置
    │   └── systems/             # 系统配置
    ├── src/                     # src 相关配置
    └── app.yaml                 # 应用级配置

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
    load_secrets,
    load_yaml,
    reload_config,
    save_yaml,
)

__all__ = [
    "ConfigDict",
    "load_config",
    "load_secrets",
    "load_yaml",
    "reload_config",
    "save_yaml",
]
