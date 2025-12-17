import uuid
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from arq import create_pool, ArqRedis
from arq.connections import RedisSettings
from arq.jobs import Job
from arq.worker import Worker, Function, func as arq_func

from config import load_config
from core.context.context_manager import ContextManager
from core.context.context import get_current_user_info
from core.di.decorators import component
from core.observation.logger import get_logger
from core.authorize.enums import Role


def _get_redis_config():
    """从 YAML 配置文件读取 Redis 配置"""
    return load_config("services/databases").redis

logger = get_logger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""

    PENDING = "pending"  # 等待执行
    RUNNING = "running"  # 正在执行
    SUCCESS = "success"  # 执行成功
    FAILED = "failed"  # 执行失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class TaskResult:
    """任务结果"""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    retry_count: int = 0
    user_id: Optional[int] = None
    user_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """重试配置"""

    max_retries: int = 1
    retry_delay: float = 1.0  # 秒
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0  # 秒


@dataclass
class TaskFunction:
    """任务函数"""

    name: str
    coroutine: Callable  # 包装处理用户上下文后的函数
    original_func: Callable  # 原始函数
    timeout: Optional[float] = None
    retry_config: Optional[RetryConfig] = None

    def to_arq_function(self) -> Function:
        """转换为arq函数"""
        return arq_func(
            self.coroutine,
            name=self.name,
            max_tries=self.retry_config.max_retries,
            timeout=self.timeout,
        )

    def __call__(self, *args, **kwargs) -> Any:
        """调用任务函数"""
        return self.original_func(*args, **kwargs)


@component(name="task_manager")
class TaskManager:
    """
    异步任务管理器

    基于arq框架实现的异步任务管理，提供任务添加、获取结果、删除任务等功能。
    使用ContextManager自动注入数据库会话和用户上下文。
    """

    def __init__(self, context_manager: ContextManager):
        """初始化任务管理器"""
        self._pool: Optional[ArqRedis] = None
        self._worker: Optional[Worker] = None
        self._redis_settings = self._get_redis_settings()
        self._context_manager = context_manager

        # 任务函数注册表
        self._task_registry: Dict[str, TaskFunction] = {}

        # 默认重试配置
        self._default_retry_config = RetryConfig()

        logger.info("任务管理器初始化完成")

    def _get_current_user_info(self) -> Optional[Dict[str, Any]]:
        """获取当前用户信息"""
        return get_current_user_info()

    def _get_current_user_id(self) -> Optional[int]:
        """获取当前用户ID"""
        user_info = self._get_current_user_info()
        return user_info.get("user_id") if user_info else None

    def _get_redis_settings(self) -> RedisSettings:
        """
        从 YAML 配置文件读取 Redis 配置

        Returns:
            RedisSettings: Redis连接配置
        """
        cfg = _get_redis_config()
        return RedisSettings(
            host=cfg.host,
            port=int(cfg.port),
            database=int(cfg.db),
            password=cfg.password if cfg.password else None,
            ssl=str(cfg.ssl).lower() == "true",
            username=cfg.username if cfg.username else None,
        )

    async def _get_pool(self) -> ArqRedis:
        """
        获取Redis连接池

        Returns:
            ArqRedis: Redis连接池
        """
        if self._pool is None:
            self._pool = await create_pool(self._redis_settings)
        return self._pool

    async def close(self) -> None:
        """关闭连接池"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        logger.info("任务管理器连接已关闭")

    def register_task(self, task_function: TaskFunction) -> None:
        """
        注册任务函数

        Args:
            name: 任务名称
            func: 任务函数
            retry_config: 重试配置（可选）
        """
        self._task_registry[task_function.name] = task_function
        logger.info(f"已注册任务: {task_function.name}")

    def scan_and_register_tasks(self, task_directories: List[str]) -> None:
        """
        扫描指定目录并自动注册任务

        Args:
            task_directories: 要扫描的任务目录列表
        """
        logger.info(f"开始扫描任务目录: {task_directories}")

        for directory in task_directories:
            self._scan_directory_for_tasks(directory)

        logger.info(f"任务扫描完成，共注册了 {len(self._task_registry)} 个任务")

    def _scan_directory_for_tasks(self, directory: str) -> None:
        """
        扫描单个目录中的任务

        Args:
            directory: 要扫描的目录路径
        """
        try:
            # 转换为绝对路径
            from utils.project_path import src_dir

            relative_path = Path(directory).resolve().relative_to(src_dir)
            package_name = ".".join(relative_path.parts)

            logger.info(f"扫描任务包: {package_name}")

            # 导入包并扫描
            try:
                package = importlib.import_module(package_name)

                # 扫描包中的所有模块
                if hasattr(package, '__path__'):
                    # 这是一个包，递归扫描所有子模块
                    for _, module_name, _ in pkgutil.walk_packages(
                        package.__path__, prefix=f"{package_name}."
                    ):
                        try:
                            module = importlib.import_module(module_name)
                            self._scan_module_for_tasks(module)
                        except Exception as e:
                            logger.error(f"导入模块失败: {module_name}, 错误: {e}")
                else:
                    # 这是一个模块，直接扫描
                    self._scan_module_for_tasks(package)

            except Exception as e:
                logger.error(f"导入包失败: {package_name}, 错误: {e}")

        except Exception as e:
            logger.error(f"扫描目录失败: {directory}, 错误: {e}")

    def _scan_module_for_tasks(self, module: Any) -> None:
        """
        扫描模块中的任务函数

        Args:
            module: 要扫描的模块对象
        """
        try:
            # 获取模块中的所有属性
            for attr_name in dir(module):
                # 跳过私有属性和特殊属性
                if attr_name.startswith('_'):
                    continue

                try:
                    attr = getattr(module, attr_name)

                    # 检查是否是TaskFunction实例
                    if isinstance(attr, TaskFunction):
                        self.register_task(attr)
                        logger.info(f"在模块 {module.__name__} 中发现任务: {attr.name}")

                except Exception as e:
                    logger.debug(
                        f"获取模块属性失败: {module.__name__}.{attr_name}, 错误: {e}"
                    )

        except Exception as e:
            logger.error(f"扫描模块任务失败: {module.__name__}, 错误: {e}")

    async def enqueue_task(
        self,
        task_name: Union[str, TaskFunction, Any],
        *args,
        task_id: Optional[str] = None,
        delay: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
        user_id: Optional[int] = None,
        user_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        添加任务到队列

        Args:
            task_name: 任务名称
            *args: 任务参数
            task_id: 任务ID（可选，不提供则自动生成）
            delay: 延迟执行时间（秒）
            retry_config: 重试配置（可选）
            user_id: 用户ID（可选，不提供则从当前上下文获取）
            user_data: 用户数据（可选，不提供则从当前上下文获取）
            metadata: 任务元数据（可选）
            **kwargs: 任务关键字参数

        Returns:
            str: 任务ID
        """
        if isinstance(task_name, TaskFunction):
            task_name = task_name.name
        elif isinstance(task_name, str):
            pass
        else:
            raise ValueError(f"类型错误: {type(task_name)}")

        assert task_name in self._task_registry, f"未找到任务: {task_name}"
        task_function = self._task_registry[task_name]
        current_retry_config = (
            retry_config or task_function.retry_config or self._default_retry_config
        )

        # 生成任务ID
        if task_id is None:
            task_id = str(uuid.uuid4())

        # 获取用户上下文（如果未提供）
        if user_data is None:
            current_user_context = self._get_current_user_info()
            if current_user_context is not None:
                user_data = current_user_context.copy()
            elif user_id is not None:
                user_data = {"user_id": user_id, "role": Role.USER}

        if user_data is None and user_id is None:
            # 尝试从当前上下文获取user_id
            current_user_id = self._get_current_user_id()
            if current_user_id is not None:
                user_data = {"user_id": current_user_id, "role": Role.USER}

        # 🔧 获取当前的 app_info_context（包含 task_id 等）
        from core.context.context import get_current_app_info

        current_app_info = get_current_app_info()

        # 准备任务上下文
        task_context = {
            "user_data": user_data,
            "app_info": current_app_info,  # 🔧 复制 app_info_context
            "metadata": metadata or {},
            "task_id": task_id,
            "retry_config": current_retry_config,
        }

        # 获取连接池
        pool = await self._get_pool()

        # 计算延迟时间
        defer_until = None
        if delay is not None:
            from utils.datetime_utils import get_now_with_timezone

            defer_until = get_now_with_timezone() + timedelta(seconds=delay)

        # 入队任务
        job = await pool.enqueue_job(
            task_name,
            task_context,
            *args,
            _job_id=task_id,
            _defer_until=defer_until,
            **kwargs,
        )

        user_id_for_log = user_data.get("user_id") if user_data else "unknown"
        logger.info(
            f"已添加任务到队列: {task_id}, 任务名称: {task_name}, 用户: {user_id_for_log}"
        )
        return task_id

    async def execute_task_with_context(
        self,
        task_func: Callable,
        context: Dict[str, Any],
        task_context: Dict[str, Any],
        *args,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        在上下文中执行任务

        Args:
            task_func: 任务函数
            context: 任务执行上下文（redis、job_id、job_try、score、enqueue_time）
            task_context: 业务上下文（用户数据、任务ID等）
            *args: 任务参数
            force_new_session: 是否强制创建新会话（默认False，避免不必要的会话创建）
            **kwargs: 任务关键字参数

        Returns:
            Any: 任务执行结果
        """
        user_data = task_context.get("user_data")
        app_info = task_context.get("app_info")  # 🔧 获取保存的 app_info_context

        # 🔧 恢复 app_info_context（包含 task_id 等）
        if app_info:
            from core.context.context import set_current_app_info

            set_current_app_info(app_info)
            logger.debug(f"🔧 已恢复 app_info_context: {app_info}")

        # 使用ContextManager执行任务，自动注入用户上下文和数据库会话
        # 🔧 可配置的会话隔离：只有在明确需要时才强制创建新会话
        result = await self._context_manager.run_with_full_context(
            task_func,
            *args,
            user_data=user_data,
            auto_inherit_user=False,
            auto_commit=True,
            force_new_session=force_new_session,  # 🔑 关键：可配置的会话隔离
            **kwargs,
        )

        task_id = task_context.get("task_id")
        user_id = user_data.get("user_id") if user_data else "unknown"
        logger.info(f"任务执行完成（独立会话）: {task_id}, 用户: {user_id}")
        return result

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        获取任务结果

        Args:
            task_id: 任务ID

        Returns:
            Optional[TaskResult]: 任务结果，如果任务不存在则返回None
        """
        pool = await self._get_pool()

        try:
            job = Job(task_id, pool)

            # 获取任务信息
            info = await job.info()
            if info is None:
                return None

            # 构造任务结果
            status = self._map_arq_status_to_task_status(info.job_status)

            # 尝试从任务上下文中获取用户信息（这可能不可用，因为arq不会保存我们的自定义上下文）
            user_id = None
            user_context_data = None

            result = TaskResult(
                task_id=task_id,
                status=status,
                result=info.result if status == TaskStatus.SUCCESS else None,
                error=str(info.result) if status == TaskStatus.FAILED else None,
                created_at=info.enqueue_time,
                started_at=info.start_time,
                finished_at=info.finish_time,
                retry_count=info.job_try or 0,
                user_id=user_id,
                user_context=user_context_data,
            )

            return result

        except Exception as e:
            logger.error(f"获取任务结果失败: {task_id}, 错误: {str(e)}")
            return None

    def _map_arq_status_to_task_status(self, arq_status: str) -> TaskStatus:
        """
        映射arq状态到任务状态

        Args:
            arq_status: arq任务状态

        Returns:
            TaskStatus: 任务状态
        """
        mapping = {
            "queued": TaskStatus.PENDING,
            "deferred": TaskStatus.PENDING,
            "in_progress": TaskStatus.RUNNING,
            "complete": TaskStatus.SUCCESS,
            "failed": TaskStatus.FAILED,
            "cancelled": TaskStatus.CANCELLED,
        }
        return mapping.get(arq_status, TaskStatus.PENDING)

    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功取消
        """
        pool = await self._get_pool()

        try:
            job = Job(task_id, pool)
            await job.abort()
            logger.info(f"已取消任务: {task_id}")
            return True
        except Exception as e:
            logger.error(f"取消任务失败: {task_id}, 错误: {str(e)}")
            return False

    async def delete_task(self, task_id: str) -> bool:
        """
        删除任务记录

        Args:
            task_id: 任务ID

        Returns:
            bool: 是否成功删除
        """
        pool = await self._get_pool()

        try:
            # 删除任务记录
            await pool.delete(f"arq:job:{task_id}")
            logger.info(f"已删除任务: {task_id}")
            return True
        except Exception as e:
            logger.error(f"删除任务失败: {task_id}, 错误: {str(e)}")
            return False

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[TaskResult]:
        """
        列出任务

        注意：由于arq的限制，无法有效地按用户ID过滤任务。
        这个方法会返回所有任务，然后在应用层过滤。
        在生产环境中，建议使用专门的任务状态存储系统。

        Args:
            status: 任务状态过滤（可选）
            user_id: 用户ID过滤（可选，但由于arq限制可能无效）
            limit: 返回数量限制

        Returns:
            List[TaskResult]: 任务列表
        """
        pool = await self._get_pool()

        try:
            # 获取所有任务键
            keys = await pool.keys("arq:job:*")
            tasks = []

            for key in keys[:limit]:
                task_id = key.decode().split(":")[-1]
                task_result = await self.get_task_result(task_id)

                if task_result is not None:
                    # 应用过滤条件
                    if status is not None and task_result.status != status:
                        continue

                    # 注意：由于arq的限制，user_id过滤可能不准确
                    if user_id is not None and task_result.user_id != user_id:
                        continue

                    tasks.append(task_result)

            return tasks

        except Exception as e:
            logger.error(f"列出任务失败: {str(e)}")
            return []

    async def get_task_count(self, status: Optional[TaskStatus] = None) -> int:
        """
        获取任务数量

        Args:
            status: 任务状态过滤（可选）

        Returns:
            int: 任务数量
        """
        tasks = await self.list_tasks(status=status)
        return len(tasks)

    def get_worker_functions(self) -> List[Function]:
        """
        获取worker函数映射

        Returns:
            Dict[str, Callable]: worker函数映射
        """
        return [v.to_arq_function() for v in self._task_registry.values()]

    def list_registered_task_names(self) -> List[str]:
        """
        获取所有已注册的任务名称

        Returns:
            List[str]: 任务名称列表
        """
        return list(self._task_registry.keys())


def task(retry_config: Optional[RetryConfig] = None, timeout: Optional[float] = 300):
    """
    任务装饰器

    Args:
        name: 任务名称（可选，不提供则使用函数名）
        retry_config: 重试配置（可选）

    Returns:
        装饰后的函数
    """

    if not retry_config:
        retry_config = RetryConfig()

    task_manager = get_task_manager()

    def decorator(func: Callable) -> Callable:

        async def _task_wrapper(*args, **kwargs):
            return await task_manager.execute_task_with_context(func, *args, **kwargs)

        function_name = func.__name__

        # 有些属性是arq worker 框架需要的
        return TaskFunction(
            name=function_name,
            coroutine=_task_wrapper,
            original_func=func,
            timeout=timeout,
            retry_config=retry_config,
        )

    return decorator


def get_task_manager() -> TaskManager:
    """
    获取任务管理器实例

    Returns:
        TaskManager: 任务管理器实例
    """
    from core.di.utils import get_bean_by_type

    return get_bean_by_type(TaskManager)
