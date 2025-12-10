#!/usr/bin/env python3
"""
Task Worker - 异步任务处理器启动脚本

异步任务处理服务，负责：
- 后台任务队列处理
- 长时间运行的异步任务
- 定时任务和延迟任务
- 任务状态管理和监控

使用方法:
    arq scripts.task.WorkerSettings

配置来源:
    - Redis 配置: config/src/databases.yaml
    - 敏感信息: config/secrets/secrets.yaml
"""

import logging
import sys
from pathlib import Path

from arq.connections import RedisSettings

# 应用信息
APP_NAME = "Async Task Worker"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "异步任务处理服务"

# 这里环境变量还没加载，所以不能使用get_logger
logger = logging.getLogger(__name__)

# 添加src目录到Python路径
current_dir = Path(__file__).resolve().parent  # scripts/
project_root = current_dir.parent  # project root
src_dir = project_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 使用统一的环境加载工具
# 注意：敏感信息从 config/secrets/secrets.yaml 加载
from utils.load_env import setup_environment

setup_environment()

# 显示应用启动信息
logger.info("🚀 启动 %s v%s", APP_NAME, APP_VERSION)
logger.info("⚙️ %s", APP_DESCRIPTION)

# 运行主函数
# 扫描 component & task
from application_startup import setup_all

setup_all()


# Worker启动和关闭回调函数
async def startup(_ctx):
    """Worker启动时的回调函数"""
    logger.info("🔄 正在初始化异步任务Worker...")

    # 在worker启动时初始化应用上下文
    from app import app

    # 将应用信息添加到FastAPI应用中（必须在start_lifespan之前）
    app.title = APP_NAME
    app.version = APP_VERSION
    app.description = APP_DESCRIPTION

    if hasattr(app, "start_lifespan"):
        await app.start_lifespan()
        logger.info("✅ 应用lifespan启动完成")
    else:
        logger.warning("⚠️ app实例没有start_lifespan方法")

    logger.info("🎯 %s 启动完成，准备处理任务", APP_NAME)


async def shutdown(_ctx):
    """Worker关闭时的回调函数"""
    logger.info("🛑 正在关闭 %s...", APP_NAME)

    # 在worker关闭时清理应用上下文
    from app import app

    if hasattr(app, "exit_lifespan"):
        await app.exit_lifespan()
        logger.info("✅ 应用lifespan关闭完成")
    else:
        logger.warning("⚠️ app实例没有exit_lifespan方法")

    logger.info("👋 %s 已停止", APP_NAME)


from core.asynctasks.task_manager import get_task_manager
from config import load_config


def _get_redis_settings() -> RedisSettings:
    """从配置文件加载 Redis 设置"""
    cfg = load_config("src/databases")
    redis_cfg = cfg.redis
    return RedisSettings(
        host=redis_cfg.host,
        port=int(redis_cfg.port),
        database=int(redis_cfg.db),
        password=redis_cfg.password or None,
        ssl=bool(redis_cfg.ssl),
        username=redis_cfg.username or None,
    )


class WorkerSettings:
    functions = get_task_manager().get_worker_functions()
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = _get_redis_settings()
    health_check_interval = 30
    max_jobs = 10
    job_timeout = 300
    keep_result = 3600


#  arq scripts.task.WorkerSettings
