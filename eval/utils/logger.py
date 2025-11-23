"""
日志工具

提供统一的日志记录功能，支持分级日志输出到不同文件。
"""
import logging
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler


def setup_logger(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    name: Optional[str] = None,
    run_number: Optional[int] = None
) -> logging.Logger:
    """
    设置日志器，支持分级日志输出

    Args:
        log_dir: 日志目录路径（可选）。如果指定，将创建以下日志文件：
            - pipeline.log: 所有级别的日志（DEBUG及以上）
            - pipeline.debug.log: DEBUG级别日志
            - pipeline.info.log: INFO级别日志
            - pipeline.warning.log: WARNING级别日志
            - pipeline.error.log: ERROR级别及以上日志
        level: 控制台输出的日志级别
        name: 日志器名称
        run_number: 运行编号（可选）。如果提供，日志文件名会包含编号，如 pipeline_1.log

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Logger本身设置为最低级别，由handler控制过滤

    # 清除已有的 handlers
    logger.handlers.clear()

    # 添加 Rich Console Handler（彩色输出）
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=False,
        show_path=False
    )
    console_handler.setLevel(level)  # 控制台只显示指定级别及以上
    logger.addHandler(console_handler)

    # 添加文件 Handlers（如果指定了日志目录）
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )

        # 如果提供了 run_number，使用带编号的文件名
        # 否则使用默认文件名（向后兼容）
        def get_log_filename(base_name: str) -> str:
            if run_number is not None:
                # pipeline.log -> pipeline_1.log
                if '.' in base_name:
                    name, ext = base_name.rsplit('.', 1)
                    return f"{name}_{run_number}.{ext}"
                return f"{base_name}_{run_number}"
            return base_name

        # 1. pipeline.log - 所有日志（DEBUG及以上）
        all_handler = logging.FileHandler(log_dir / get_log_filename("pipeline.log"), encoding='utf-8')
        all_handler.setLevel(logging.DEBUG)
        all_handler.setFormatter(formatter)
        logger.addHandler(all_handler)

        # 2. pipeline.debug.log - 只有DEBUG级别
        debug_handler = logging.FileHandler(log_dir / get_log_filename("pipeline.debug.log"), encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.addFilter(lambda record: record.levelno == logging.DEBUG)
        debug_handler.setFormatter(formatter)
        logger.addHandler(debug_handler)

        # 3. pipeline.info.log - 只有INFO级别
        info_handler = logging.FileHandler(log_dir / get_log_filename("pipeline.info.log"), encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)

        # 4. pipeline.warning.log - 只有WARNING级别
        warning_handler = logging.FileHandler(log_dir / get_log_filename("pipeline.warning.log"), encoding='utf-8')
        warning_handler.setLevel(logging.WARNING)
        warning_handler.addFilter(lambda record: record.levelno == logging.WARNING)
        warning_handler.setFormatter(formatter)
        logger.addHandler(warning_handler)

        # 5. pipeline.error.log - ERROR及以上级别
        error_handler = logging.FileHandler(log_dir / get_log_filename("pipeline.error.log"), encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    return logger


def get_console() -> Console:
    """获取 Rich Console 实例"""
    return Console()

