#!/usr/bin/env python3
"""
统一的环境设置工具

提供 Python 路径设置和时区配置功能。

注意：敏感信息（API Keys、密码等）现在统一从 config/secrets/secrets.yaml 加载，
不再使用 .env 文件。
"""

import logging
import os
import sys
import time
from typing import Optional, List

from utils.app_meta import set_service_name

# 这里配置还没加载，所以不能使用 get_logger
logger = logging.getLogger(__name__)


class SecretNotConfiguredError(Exception):
    """Secret 未配置错误"""
    pass


def reset_timezone():
    """
    重置时区
    """
    timezone = os.environ.get("TZ") or "Asia/Shanghai"
    os.environ["TZ"] = timezone
    # tzset() is not available on Windows, only call it if available
    if hasattr(time, 'tzset'):
        time.tzset()


def sync_pythonpath_with_syspath():
    """
    同步 PYTHONPATH 和 sys.path，确保 sys.path 中的路径都在 PYTHONPATH 中

    注意：
    1. 只同步非标准库路径
    2. 排除 .venv 和类似的虚拟环境路径
    3. 保持原有 PYTHONPATH 的优先级
    """
    from pathlib import Path

    # 获取当前 PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "").split(":")
    pythonpath = [p for p in pythonpath if p]  # 移除空字符串

    # 需要排除的路径模式
    exclude_patterns = [
        ".venv",
        "site-packages",
        "dist-packages",
        "lib/python",
        "__pycache__",
    ]

    # 从 sys.path 中获取需要添加的路径
    new_paths = []
    for path in sys.path:
        # 跳过空路径
        if not path:
            continue

        # 转换为 Path 对象以便处理
        path_obj = Path(path)

        # 跳过不存在的路径
        if not path_obj.exists():
            continue

        # 跳过需要排除的路径
        if any(pattern in str(path_obj) for pattern in exclude_patterns):
            continue

        # 转换为字符串并规范化
        path_str = str(path_obj.resolve())

        # 如果路径不在当前 PYTHONPATH 中，添加到新路径列表
        if path_str not in pythonpath:
            new_paths.append(path_str)

    # 如果有新的路径需要添加
    if new_paths:
        # 将新路径添加到现有 PYTHONPATH 后面
        all_paths = pythonpath + new_paths
        # 更新环境变量
        os.environ["PYTHONPATH"] = ":".join(all_paths)
        logger.debug("已更新 PYTHONPATH: %s", os.environ["PYTHONPATH"])


def _check_secrets(required_secrets: List[str]) -> None:
    """
    检查必要的 secrets 是否已配置

    Args:
        required_secrets: 需要检查的 secret 键列表，支持点分隔格式
            例如: ["openai_api_key", "mongodb.password"]

    Raises:
        SecretNotConfiguredError: 如果必要的 secret 未配置
    """
    from config import load_secrets

    try:
        secrets = load_secrets()
    except FileNotFoundError as e:
        raise SecretNotConfiguredError(str(e)) from e

    missing = []
    for key in required_secrets:
        value = secrets.get(key)
        if value is None or value == "":
            missing.append(key)

    if missing:
        raise SecretNotConfiguredError(
            f"以下必要的 secrets 未配置: {', '.join(missing)}\n"
            f"请在 config/secrets/secrets.yaml 中设置这些值"
        )


def setup_environment(
    service_name: Optional[str] = None,
    check_secrets: Optional[List[str]] = None,
) -> bool:
    """
    统一的环境设置函数

    设置 Python 路径、时区和服务名称。
    敏感信息现在从 config/secrets/secrets.yaml 加载。

    Args:
        service_name: 当前启动服务的名称，将被存储在 APP_META_DATA 中
        check_secrets: 需要检查的 secret 键列表，支持点分隔格式
            例如: ["openai_api_key", "mongodb.password"]

    Returns:
        bool: 始终返回 True（保持向后兼容）

    Raises:
        SecretNotConfiguredError: 如果 check_secrets 中指定的 secret 未配置
    """
    # 同步 PYTHONPATH 和 sys.path
    sync_pythonpath_with_syspath()

    # 重置时区
    reset_timezone()

    # 设置服务名称
    if service_name:
        set_service_name(service_name)
        logger.debug("已设置服务名称: %s", service_name)

    # 检查必要的 secrets
    if check_secrets:
        _check_secrets(check_secrets)

    logger.info("环境设置完成")
    return True
