"""
统一配置加载工具

支持:
1. YAML 配置文件加载
2. 环境变量替换 (${VAR} 或 ${VAR:default})
3. 点访问 (config.retrieval.mode)
4. 配置缓存（避免重复加载）

使用示例:
    from config import load_config

    # 加载系统配置
    config = load_config("eval/systems/parallax")
    print(config.retrieval.mode)
    print(config.concurrency.extraction)

    # 加载数据集配置
    dataset = load_config("eval/datasets/locomo")
    print(dataset.data.path)
"""
import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache


class ConfigDict:
    """
    支持点访问的配置字典

    Examples:
        config = ConfigDict({"a": {"b": 1}})
        config.a.b  # -> 1
        config["a"]["b"]  # -> 1
        config.get("a.b")  # -> 1
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __repr__(self) -> str:
        return f"ConfigDict({self._data})"

    def get(self, key: str, default: Any = None) -> Any:
        """
        支持点分隔的 key 访问

        Examples:
            config.get("retrieval.mode")
            config.get("unknown.key", "default_value")
        """
        keys = key.split(".")
        value = self
        try:
            for k in keys:
                value = getattr(value, k)
            return value
        except AttributeError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """转换回普通字典"""
        return self._data

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()


def _get_config_dir() -> Path:
    """获取 config 目录路径"""
    return Path(__file__).parent


def _replace_env_vars(obj: Any) -> Any:
    """
    递归替换配置中的环境变量

    支持格式:
    - ${VAR_NAME} - 必须存在的环境变量
    - ${VAR_NAME:default_value} - 带默认值

    类型转换:
    - ${VAR:123} -> int 123
    - ${VAR:1.5} -> float 1.5
    - ${VAR:true} -> bool True
    """
    if isinstance(obj, dict):
        return {key: _replace_env_vars(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_replace_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'

        def replacer(match):
            var_name = match.group(1)
            default_value = match.group(2)

            value = os.environ.get(var_name)
            if value is None:
                if default_value is not None:
                    value = default_value
                else:
                    # 环境变量不存在且无默认值，保持原样
                    return match.group(0)

            return value

        result = re.sub(pattern, replacer, obj)

        # 尝试类型转换（仅当整个字符串是单一值时）
        if result != obj:  # 发生了替换
            result = _try_convert_type(result)

        return result
    else:
        return obj


def _try_convert_type(value: str) -> Union[str, int, float, bool]:
    """尝试将字符串转换为合适的类型

    转换优先级：
    1. 先尝试 float（包括整数形式如 "0", "1"）
    2. 只有纯文本布尔值才转换为 bool（"true", "false", "yes", "no", "on", "off"）

    这样可以避免 "0" 被错误转换为 False，"1" 被错误转换为 True
    """
    # 先尝试数值转换（避免 "0" 被误转为 False）
    # float（也能处理整数字符串如 "0", "1", "123"）
    try:
        float_val = float(value)
        # 如果是整数形式，返回 int
        if float_val == int(float_val) and '.' not in value:
            return int(float_val)
        return float_val
    except ValueError:
        pass

    # 只有纯文本布尔值才转换为 bool（不包括 "0" 和 "1"）
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False

    return value


def _find_config_file(name: str) -> Path:
    """
    查找配置文件

    Args:
        name: 配置名称，如 "eval/systems/parallax" 或 "eval/locomo"

    Returns:
        配置文件的完整路径
    """
    config_dir = _get_config_dir()

    # 尝试多种可能的路径
    candidates = [
        config_dir / f"{name}.yaml",
        config_dir / f"{name}.yml",
        config_dir / f"{name}.json",
        config_dir / name / "config.yaml",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Config file not found: {name}\n"
        f"Searched paths:\n" + "\n".join(f"  - {p}" for p in candidates)
    )


def load_yaml(file_path: str) -> Dict[str, Any]:
    """
    加载 YAML 配置文件（原始字典格式）

    Args:
        file_path: YAML 文件路径

    Returns:
        解析后的配置字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config = _replace_env_vars(config)
    return config


@lru_cache(maxsize=32)
def load_config(name: str) -> ConfigDict:
    """
    加载配置文件，返回支持点访问的配置对象

    Args:
        name: 配置名称，如:
            - "eval/systems/parallax" -> config/eval/systems/parallax.yaml
            - "eval/datasets/locomo" -> config/eval/datasets/locomo.yaml
            - "workflows/adaptive_retrieval" -> config/workflows/adaptive_retrieval.yaml

    Returns:
        ConfigDict 对象，支持点访问

    Examples:
        from config import load_config

        config = load_config("eval/systems/parallax")
        print(config.retrieval.mode)  # "agentic"
        print(config.concurrency.extraction)  # 5
        print(config.get("llm.openai.model"))  # "gpt-4.1-mini"
    """
    config_path = _find_config_file(name)
    raw_config = load_yaml(str(config_path))
    return ConfigDict(raw_config)


def reload_config(name: str) -> ConfigDict:
    """
    重新加载配置（清除缓存）

    用于需要动态更新配置的场景
    """
    load_config.cache_clear()
    return load_config(name)


def save_yaml(config: Union[Dict[str, Any], ConfigDict], file_path: str):
    """
    保存配置到 YAML 文件

    Args:
        config: 配置字典或 ConfigDict
        file_path: 保存路径
    """
    if isinstance(config, ConfigDict):
        config = config.to_dict()

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)


