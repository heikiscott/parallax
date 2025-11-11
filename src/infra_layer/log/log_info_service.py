"""
日志信息服务

提供日志上下文信息的注入和管理，支持异步上下文管理器模式。
主要处理：
- trace_id: 请求追踪ID
- group_id: 组ID
- user_id: 用户ID
"""

from typing import Optional
from contextlib import asynccontextmanager

from core.context import context
from core.observation.logger import get_logger
from core.di.decorators import component
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)


@component(name="log_info_service")
class LogInfoService:
    """日志信息服务，负责管理和注入日志相关的上下文信息

    使用@component装饰器确保单例模式，可以通过DI系统注入到其他组件中。
    """

    @asynccontextmanager
    async def inject_log_info(
        self,
        trace_id: Optional[str] = None,
        group_id: Optional[str] = None,
        from_user_id: Optional[str] = None,
    ):
        """
        注入日志信息到上下文中的异步上下文管理器

        Args:
            trace_id: 追踪ID，如果不提供则自动生成
            group_id: 组ID
            from_user_id: 操作发起者ID

        Yields:
            注入后的上下文信息字典
        """
        # 获取当前的app_info并创建新的副本
        current_app_info = context.get_current_app_info() or {}
        app_info = current_app_info.copy()

        try:
            # 更新新字典中的值
            if trace_id is not None:
                app_info['trace_id'] = trace_id
            # 更新group_id和from_user_id（如果提供了新值）
            if group_id is not None:
                app_info['group_id'] = group_id
            if from_user_id is not None:
                app_info['from_user_id'] = from_user_id

            # 设置更新后的app_info
            token = context.set_current_app_info(app_info)

            try:
                # 返回注入后的上下文信息
                yield app_info
            finally:
                # 使用token恢复到原始状态
                context.clear_current_app_info(token)

        except Exception as e:
            logger.error("注入日志信息时发生错误: %s", e)
            raise

    @asynccontextmanager
    async def override_trace_id(self, trace_id: str):
        """
        临时覆盖trace_id的异步上下文管理器

        Args:
            trace_id: 新的追踪ID

        Yields:
            更新后的上下文信息字典
        """
        async with self.inject_log_info(trace_id=trace_id):
            yield

    @asynccontextmanager
    async def override_group_id(self, group_id: str):
        """
        临时覆盖group_id的异步上下文管理器

        Args:
            group_id: 新的组ID

        Yields:
            更新后的上下文信息字典
        """
        async with self.inject_log_info(group_id=group_id):
            yield

    @asynccontextmanager
    async def override_from_user_id(self, from_user_id: str):
        """
        临时覆盖from_user_id的异步上下文管理器

        Args:
            from_user_id: 新的操作发起者ID

        Yields:
            更新后的上下文信息字典
        """
        async with self.inject_log_info(from_user_id=from_user_id):
            yield

    @staticmethod
    def get_current_trace_id() -> Optional[str]:
        """获取当前的trace_id"""
        app_info = context.get_current_app_info()
        return app_info.get('trace_id') if app_info else None

    @staticmethod
    def get_current_group_id() -> Optional[str]:
        """获取当前的group_id"""
        app_info = context.get_current_app_info()
        return app_info.get('group_id') if app_info else None

    @staticmethod
    def get_current_from_user_id() -> Optional[str]:
        """获取当前的操作发起者ID"""
        app_info = context.get_current_app_info()
        return app_info.get('from_user_id') if app_info else None


# 全局日志服务实例
_log_service: Optional[LogInfoService] = None


def get_log_service() -> LogInfoService:
    """获取全局日志服务实例"""
    global _log_service
    if _log_service is None:
        _log_service = get_bean_by_type(LogInfoService)
    return _log_service


@asynccontextmanager
async def log_context(
    *,
    trace_id: Optional[str] = None,
    group_id: Optional[str] = None,
    from_user_id: Optional[str] = None,
):
    """
    统一的日志上下文管理器

    示例:
        async with log_context(trace_id="123", group_id="456"):
            await some_operation()
    """
    async with get_log_service().inject_log_info(
        trace_id=trace_id, group_id=group_id, from_user_id=from_user_id
    ) as app_info:
        yield app_info


# 导出便捷获取函数
get_current_from_user_id = LogInfoService.get_current_from_user_id
get_current_group_id = LogInfoService.get_current_group_id
get_current_trace_id = LogInfoService.get_current_trace_id
