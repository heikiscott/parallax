"""
统一日志模块

提供两种使用模式：
1. 默认模式：控制台输出，适合主应用运行时
2. 扩展模式：Rich 彩色控制台 + 多文件分级输出，适合评估/调试场景

基本用法:
    from core.observation.logger import get_logger
    logger = get_logger(__name__)
    logger.info("message")

扩展用法（评估/调试）:
    from core.observation.logger import setup_logger
    logger = setup_logger(log_dir=Path("logs"), run_number=1)
"""
import logging
import traceback
import contextvars
import uuid
import sys
import os
from pathlib import Path
from typing import Optional
from enum import Enum
from functools import lru_cache

# ============================================================================
# 可选依赖：Rich 库用于彩色控制台输出
# ============================================================================

try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = None
    RichHandler = None

# ============================================================================
# 常量定义
# ============================================================================

# 日志格式（统一格式，便于解析）
LOG_FORMAT = '%(asctime)s - [%(activity_id)s] - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'

# 需要抑制的第三方库日志
NOISY_LOGGERS = [
    'urllib3',
    'google',
    'googleapiclient',
    'httpx',
    'hpack',
    'httpcore',
    'pymongo',
    'aiokafka',
]

# ============================================================================
# Activity ID 追踪（用于关联同一请求的所有日志）
# ============================================================================

activity_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    'activity_id', default='-'
)


def set_activity_id(activity_id: Optional[str] = None) -> str:
    """设置当前上下文的 activity_id

    Args:
        activity_id: 自定义 ID，不提供则自动生成 8 位 UUID

    Returns:
        设置的 activity_id
    """
    if activity_id is None:
        activity_id = str(uuid.uuid4())[:8]
    activity_id_var.set(activity_id)
    return activity_id


def get_activity_id() -> str:
    """获取当前上下文的 activity_id"""
    return activity_id_var.get()


# ============================================================================
# 日志过滤器
# ============================================================================

class ActivityIdFilter(logging.Filter):
    """自动添加 activity_id 到日志记录"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.activity_id = activity_id_var.get()
        return True


class LevelFilter(logging.Filter):
    """只允许特定级别的日志通过"""

    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


# ============================================================================
# 日志级别枚举
# ============================================================================

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# 工具函数
# ============================================================================

def _get_caller_module_name(depth: int = 2) -> str:
    """获取调用者的模块名

    Args:
        depth: 调用栈深度，默认 2（跳过本函数和直接调用者）
    """
    frame = sys._getframe(depth)
    return frame.f_globals.get('__name__', 'unknown')


def _make_log_filename(base_name: str, run_number: Optional[int]) -> str:
    """生成日志文件名

    Args:
        base_name: 基础文件名，如 "pipeline.log"
        run_number: 运行编号，如果提供则生成 "pipeline_1.log"
    """
    if run_number is None:
        return base_name

    if '.' in base_name:
        name, ext = base_name.split('.', 1)
        return f"{name}_{run_number}.{ext}"
    return f"{base_name}_{run_number}"


def _create_file_handler(
    log_dir: Path,
    filename: str,
    level: int,
    formatter: logging.Formatter,
    activity_filter: ActivityIdFilter,
    level_filter: Optional[LevelFilter] = None,
) -> logging.FileHandler:
    """创建文件日志 Handler

    Args:
        log_dir: 日志目录
        filename: 文件名
        level: 日志级别
        formatter: 日志格式器
        activity_filter: Activity ID 过滤器
        level_filter: 级别过滤器（可选，用于只记录特定级别）
    """
    handler = logging.FileHandler(log_dir / filename, encoding='utf-8')
    handler.setLevel(level)
    handler.setFormatter(formatter)
    handler.addFilter(activity_filter)
    if level_filter:
        handler.addFilter(level_filter)
    return handler


# ============================================================================
# LoggerProvider 核心类
# ============================================================================

class LoggerProvider:
    """统一的日志管理类（单例模式）

    职责：
    1. 初始化根日志配置
    2. 抑制第三方库噪音日志
    3. 提供 logger 实例缓存
    4. 支持扩展模式（文件日志 + Rich 控制台）
    """

    _instance: Optional['LoggerProvider'] = None
    _initialized: bool = False
    _rich_console = None

    def __new__(cls) -> 'LoggerProvider':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._init_root_logger()
            self._suppress_noisy_loggers()
            LoggerProvider._initialized = True

    def _init_root_logger(self):
        """初始化根日志器"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()

        handler = logging.StreamHandler(sys.stdout)
        handler.addFilter(ActivityIdFilter())

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=LOG_FORMAT,
            handlers=[handler],
        )

    def _suppress_noisy_loggers(self):
        """抑制第三方库的冗余日志"""
        for logger_name in NOISY_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    @lru_cache(maxsize=1000)
    def _get_cached_logger(self, module_name: str) -> logging.Logger:
        """获取缓存的日志器"""
        return logging.getLogger(module_name)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """获取日志器

        Args:
            name: 日志器名称，推荐传入 __name__
        """
        if name is None:
            name = _get_caller_module_name(depth=3)
        return self._get_cached_logger(name)

    def setup_file_logging(
        self,
        log_dir: Path,
        name: Optional[str] = None,
        level: int = logging.INFO,
        run_number: Optional[int] = None,
        use_rich: bool = True,
    ) -> logging.Logger:
        """配置文件日志输出（扩展模式）

        Args:
            log_dir: 日志目录，将创建多个分级日志文件
            name: 日志器名称
            level: 控制台输出级别
            run_number: 运行编号，用于区分不同运行的日志文件
            use_rich: 是否使用 Rich 彩色输出

        Returns:
            配置好的 Logger 实例
        """
        if name is None:
            name = _get_caller_module_name(depth=3)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # 配置控制台输出
        self._add_console_handler(logger, level, use_rich)

        # 配置文件输出
        self._add_file_handlers(logger, log_dir, run_number)

        return logger

    def _add_console_handler(
        self,
        logger: logging.Logger,
        level: int,
        use_rich: bool
    ):
        """添加控制台 Handler"""
        if use_rich and RICH_AVAILABLE:
            handler = RichHandler(
                rich_tracebacks=True,
                show_time=False,
                show_path=False
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(LOG_FORMAT))

        handler.setLevel(level)
        handler.addFilter(ActivityIdFilter())
        logger.addHandler(handler)

    def _add_file_handlers(
        self,
        logger: logging.Logger,
        log_dir: Path,
        run_number: Optional[int]
    ):
        """添加文件 Handlers（分级输出到不同文件）"""
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(LOG_FORMAT)
        activity_filter = ActivityIdFilter()

        def filename(base: str) -> str:
            return _make_log_filename(base, run_number)

        # 所有日志（DEBUG 及以上）
        logger.addHandler(_create_file_handler(
            log_dir, filename("pipeline.log"),
            logging.DEBUG, formatter, activity_filter
        ))

        # 仅 DEBUG 级别
        logger.addHandler(_create_file_handler(
            log_dir, filename("pipeline.debug.log"),
            logging.DEBUG, formatter, activity_filter,
            LevelFilter(logging.DEBUG)
        ))

        # 仅 INFO 级别
        logger.addHandler(_create_file_handler(
            log_dir, filename("pipeline.info.log"),
            logging.INFO, formatter, activity_filter,
            LevelFilter(logging.INFO)
        ))

        # 仅 WARNING 级别
        logger.addHandler(_create_file_handler(
            log_dir, filename("pipeline.warning.log"),
            logging.WARNING, formatter, activity_filter,
            LevelFilter(logging.WARNING)
        ))

        # ERROR 及以上级别
        logger.addHandler(_create_file_handler(
            log_dir, filename("pipeline.error.log"),
            logging.ERROR, formatter, activity_filter
        ))

    def get_console(self):
        """获取 Rich Console 实例"""
        if not RICH_AVAILABLE:
            return None
        if self._rich_console is None:
            self._rich_console = Console()
        return self._rich_console

    # ========================================================================
    # 便捷日志方法（自动获取调用者模块名）
    # ========================================================================

    def debug(self, message: str, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).warning(message, *args, **kwargs)

    def warn(self, message: str, *args, **kwargs):
        self.warning(message, *args, **kwargs)

    def error(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).error(
            message, exc_info=exc_info, *args, **kwargs
        )

    def exception(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).exception(
            message, exc_info=exc_info, *args, **kwargs
        )

    def critical(self, message: str, exc_info: bool = True, *args, **kwargs):
        self._get_cached_logger(_get_caller_module_name()).critical(
            message, exc_info=exc_info, *args, **kwargs
        )

    def log_with_stack(self, level: LogLevel, message: str):
        """记录带完整堆栈的日志"""
        logger = self._get_cached_logger(_get_caller_module_name())
        stack_trace = traceback.format_stack()
        full_message = f"{message}\n堆栈跟踪:\n{''.join(stack_trace)}"
        getattr(logger, level.value.lower())(full_message)


# ============================================================================
# 全局实例
# ============================================================================

logger_provider = LoggerProvider()


# ============================================================================
# 公开 API - 推荐用法
# ============================================================================

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取日志器（推荐用法）

    推荐:
        logger = get_logger(__name__)  # 模块顶部获取一次
        logger.info("message")         # 后续直接使用

    Args:
        name: 模块名，推荐传入 __name__
    """
    if name is None:
        name = _get_caller_module_name()
    return logger_provider.get_logger(name)


def setup_logger(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    name: Optional[str] = None,
    run_number: Optional[int] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """配置日志器（扩展模式，适用于评估/调试）

    Args:
        log_dir: 日志目录，将创建以下文件：
            - pipeline.log: 所有日志
            - pipeline.debug.log: 仅 DEBUG
            - pipeline.info.log: 仅 INFO
            - pipeline.warning.log: 仅 WARNING
            - pipeline.error.log: ERROR 及以上
        level: 控制台输出级别，默认 INFO
        name: 日志器名称
        run_number: 运行编号，文件名会包含编号如 pipeline_1.log
        use_rich: 是否使用 Rich 彩色输出

    Returns:
        配置好的 Logger 实例
    """
    if name is None:
        name = _get_caller_module_name()

    if log_dir is not None:
        return logger_provider.setup_file_logging(
            log_dir=log_dir,
            name=name,
            level=level,
            run_number=run_number,
            use_rich=use_rich,
        )
    return logger_provider.get_logger(name)


def get_console():
    """获取 Rich Console 实例（用于高级终端输出）"""
    return logger_provider.get_console()


# ============================================================================
# 公开 API - 便捷用法（适合偶尔调用）
# ============================================================================

def debug(message: str, *args, **kwargs):
    """记录 DEBUG 日志"""
    logger_provider.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs):
    """记录 INFO 日志"""
    logger_provider.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """记录 WARNING 日志"""
    logger_provider.warning(message, *args, **kwargs)


def warn(message: str, *args, **kwargs):
    """记录 WARNING 日志（别名）"""
    logger_provider.warn(message, *args, **kwargs)


def error(message: str, exc_info: bool = True, *args, **kwargs):
    """记录 ERROR 日志"""
    logger_provider.error(message, exc_info=exc_info, *args, **kwargs)


def exception(message: str, exc_info: bool = True, *args, **kwargs):
    """记录异常日志（自动包含堆栈）"""
    logger_provider.exception(message, exc_info=exc_info, *args, **kwargs)


def critical(message: str, exc_info: bool = True, *args, **kwargs):
    """记录 CRITICAL 日志"""
    logger_provider.critical(message, exc_info=exc_info, *args, **kwargs)


def log_with_stack(level: LogLevel, message: str):
    """记录带完整堆栈的日志"""
    logger_provider.log_with_stack(level, message)
