"""
Elasticsearch 客户端工厂

提供 Elasticsearch 客户端缓存和管理功能。

配置来源: config/services/databases.yaml
敏感信息（密码）来源: .env 文件
"""

import asyncio
from typing import Dict, Optional, List, Type, Any
from hashlib import md5
from elasticsearch import AsyncElasticsearch
from elasticsearch.dsl.async_connections import connections as async_connections

from core.di.decorators import component
from core.observation.logger import get_logger
from core.oxm.es.doc_base import DocBase
from config import load_config

logger = get_logger(__name__)


def _get_es_config():
    """获取 Elasticsearch 配置"""
    return load_config("services/databases").elasticsearch


def get_default_es_config() -> Dict[str, Any]:
    """
    从 YAML 配置获取默认的 Elasticsearch 配置

    配置来源: config/services/databases.yaml
    敏感信息（密码）来源: .env 文件（通过 YAML 的 ${VAR} 语法注入）

    Returns:
        Dict[str, Any]: 配置字典
    """
    cfg = _get_es_config()

    # 获取主机信息 - 支持单个 hosts 字符串或列表
    es_hosts_str = cfg.hosts
    if isinstance(es_hosts_str, list):
        es_hosts = es_hosts_str
    elif "," in es_hosts_str:
        es_hosts = [host.strip() for host in es_hosts_str.split(",")]
    else:
        es_hosts = [es_hosts_str]

    # 认证信息（从 .env 注入）
    es_username = cfg.username if cfg.username else None
    es_password = cfg.password if cfg.password else None
    es_api_key = None  # API Key 可后续在 YAML 中添加

    # 连接参数
    es_timeout = 120  # 默认超时
    es_verify_certs = cfg.verify_certs if hasattr(cfg, 'verify_certs') else False

    config = {
        "hosts": es_hosts,
        "timeout": es_timeout,
        "username": es_username,
        "password": es_password,
        "api_key": es_api_key,
        "verify_certs": es_verify_certs,
    }

    logger.info("获取默认 Elasticsearch 配置:")
    logger.info("  主机: %s", es_hosts)
    logger.info("  超时: %s 秒", es_timeout)
    logger.info(
        "  认证: %s", "API Key" if es_api_key else ("Basic" if es_username else "无")
    )

    return config


def get_cache_key(
    hosts: List[str], username: Optional[str] = None, api_key: Optional[str] = None
) -> str:
    """
    生成缓存键
    基于 hosts、认证信息生成唯一标识

    Args:
        hosts: Elasticsearch主机列表
        username: 用户名
        api_key: API密钥

    Returns:
        str: 缓存键
    """
    hosts_str = ",".join(sorted(hosts))
    auth_str = ""
    if api_key:
        auth_str = f"api_key:{api_key[:8]}..."
    elif username:
        auth_str = f"basic:{username}"

    key_content = f"{hosts_str}:{auth_str}"
    return md5(key_content.encode()).hexdigest()


class ElasticsearchClientWrapper:
    """Elasticsearch 客户端包装器"""

    def __init__(self, async_client: AsyncElasticsearch, hosts: List[str]):
        self.async_client = async_client
        self.hosts = hosts
        self._initialized = False
        self._document_classes: List[Type[DocBase]] = []

    async def initialize_indices(
        self, document_classes: Optional[List[Type[DocBase]]] = None
    ):
        """初始化索引"""
        if self._initialized:
            return

        if document_classes:
            try:
                logger.info(
                    "正在初始化 Elasticsearch 索引，共 %d 个文档类",
                    len(document_classes),
                )

                for doc_class in document_classes:
                    await self._init_document_index(doc_class)

                self._document_classes = document_classes
                self._initialized = True
                logger.info(
                    "✅ Elasticsearch 索引初始化成功，处理了 %d 个文档类",
                    len(document_classes),
                )

                for doc_class in document_classes:
                    logger.info(
                        "📋 初始化索引: class=%s -> index=%s",
                        doc_class.__name__,
                        (
                            doc_class._index._name
                            if hasattr(doc_class, '_index')
                            else 'unknown'
                        ),
                    )

            except Exception as e:
                logger.error("❌ Elasticsearch 索引初始化失败: %s", e)
                raise

    async def _init_document_index(self, doc_class: Type[DocBase]):
        """初始化单个文档类的索引"""
        try:
            # 获取别名名称
            if hasattr(doc_class, '_index') and hasattr(doc_class._index, '_name'):
                alias = doc_class._index._name
                # 检查别名是否为空
                if not alias or alias.strip() == '':
                    logger.warning("文档类 %s 的索引名称为空", doc_class.__name__)
                    return
            else:
                logger.warning("文档类 %s 没有正确的索引配置", doc_class.__name__)
                return

            logger.info("正在检查索引别名: %s (文档类: %s)", alias, doc_class.__name__)

            # 检查别名是否存在
            alias_exists = await self.async_client.indices.exists(index=alias)

            if not alias_exists:
                # 生成目标索引名
                if hasattr(doc_class, 'dest'):
                    dst = doc_class.dest()
                else:
                    from utils.datetime_utils import get_now_with_timezone

                    now = get_now_with_timezone()
                    dst = f"{alias}-{now.strftime('%Y%m%d%H%M%S%f')}"

                # 创建索引
                await doc_class.init(index=dst, using=self.async_client)

                # 创建别名
                await self.async_client.indices.update_aliases(
                    body={
                        "actions": [
                            {
                                "add": {
                                    "index": dst,
                                    "alias": alias,
                                    "is_write_index": True,
                                }
                            }
                        ]
                    }
                )
                logger.info("✅ 创建索引和别名: %s -> %s", dst, alias)
            else:
                logger.info("📋 索引别名已存在: %s", alias)

        except Exception as e:
            logger.error("❌ 初始化文档类 %s 的索引失败: %s", doc_class.__name__, e)
            raise

    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            await self.async_client.ping()
            logger.info("✅ Elasticsearch 连接测试成功: %s", self.hosts)
            return True
        except Exception as e:
            logger.error("❌ Elasticsearch 连接测试失败: %s, 错误: %s", self.hosts, e)
            return False

    async def close(self):
        """关闭连接"""
        try:
            if self.async_client:
                await self.async_client.close()
            logger.info("🔌 Elasticsearch 连接已关闭: %s", self.hosts)
        except Exception as e:
            logger.error("关闭 Elasticsearch 连接时出错: %s", e)

    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化索引"""
        return self._initialized


@component(name="elasticsearch_client_factory", primary=True)
class ElasticsearchClientFactory:
    """
    Elasticsearch 客户端工厂
    ### AsyncElasticsearch 是有状态的，因此可以在多个地方使用同一个实例 ###

    提供基于配置的 Elasticsearch 客户端缓存和管理功能
    """

    def __init__(self):
        """初始化 Elasticsearch 客户端工厂"""
        self._clients: Dict[str, ElasticsearchClientWrapper] = {}
        self._lock = asyncio.Lock()
        self._default_config: Optional[Dict[str, Any]] = None
        logger.info("ElasticsearchClientFactory initialized")

    async def create_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs,
    ) -> ElasticsearchClientWrapper:
        """
        创建 Elasticsearch 客户端实例

        Args:
            hosts: Elasticsearch主机列表
            username: 用户名
            password: 密码
            api_key: API密钥
            timeout: 超时时间（秒）
            **kwargs: 其他连接参数

        Returns:
            ElasticsearchClientWrapper 实例
        """
        # 构建连接参数
        conn_params = {
            "hosts": hosts,
            "timeout": timeout,
            "max_retries": 3,
            "retry_on_timeout": True,
            "verify_certs": False,  # 禁用 SSL 证书验证
            "ssl_show_warn": False,  # 禁用 SSL 警告
            **kwargs,
        }

        # 添加认证信息
        if api_key:
            conn_params["api_key"] = api_key
        elif username and password:
            conn_params["basic_auth"] = (username, password)

        # 创建异步客户端
        async_client = AsyncElasticsearch(**conn_params)

        client_wrapper = ElasticsearchClientWrapper(async_client, hosts)

        logger.info("Created Elasticsearch client for %s", hosts)
        return client_wrapper

    def create_async_connection(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        alias: str = "default",
        **kwargs,
    ):
        """
        创建 elasticsearch_dsl 的 async connection

        Args:
            hosts: Elasticsearch主机列表
            username: 用户名
            password: 密码
            api_key: API密钥
            timeout: 超时时间（秒）
            alias: 连接别名，默认为 "default"
            **kwargs: 其他连接参数

        Returns:
            elasticsearch_dsl 的 async connection 对象
        """
        # 构建连接参数
        conn_params = {
            "hosts": hosts,
            "timeout": timeout,
            "max_retries": 3,
            "retry_on_timeout": True,
            "verify_certs": False,  # 禁用 SSL 证书验证
            "ssl_show_warn": False,  # 禁用 SSL 警告
            **kwargs,
        }

        # 添加认证信息
        if api_key:
            conn_params["api_key"] = api_key
        elif username and password:
            conn_params["basic_auth"] = (username, password)

        # 创建 elasticsearch_dsl async 连接
        async_connections.configure(default=conn_params)  # 必须先配置，再获取连接
        async_es_connect = async_connections.get_connection(alias=alias)

        logger.info(
            "Created elasticsearch_dsl async connection for %s with alias '%s'",
            hosts,
            alias,
        )
        return async_es_connect

    async def get_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ) -> ElasticsearchClientWrapper:
        """
        获取 Elasticsearch 客户端实例

        Args:
            hosts: Elasticsearch主机列表
            username: 用户名
            password: 密码
            api_key: API密钥
            **kwargs: 其他配置参数

        Returns:
            ElasticsearchClientWrapper 实例
        """
        cache_key = get_cache_key(hosts, username, api_key)

        async with self._lock:
            # 检查缓存
            if cache_key in self._clients:
                logger.debug("Using cached Elasticsearch client for %s", hosts)
                return self._clients[cache_key]

            # 创建新的客户端实例
            logger.info("Creating new Elasticsearch client for %s", hosts)

            client_wrapper = await self.create_client(
                hosts=hosts,
                username=username,
                password=password,
                api_key=api_key,
                **kwargs,
            )

            # 测试连接
            if not await client_wrapper.test_connection():
                await client_wrapper.close()
                raise RuntimeError(f"Elasticsearch 连接测试失败: {hosts}")

            self._clients[cache_key] = client_wrapper
            logger.info(
                "Elasticsearch client %s created and cached with key %s",
                hosts,
                cache_key,
            )

        return client_wrapper

    async def get_default_client(self) -> ElasticsearchClientWrapper:
        """
        获取基于环境变量配置的默认 Elasticsearch 客户端实例

        Returns:
            ElasticsearchClientWrapper 实例
        """
        # 获取或创建默认配置
        if self._default_config is None:
            self._default_config = get_default_es_config()

        config = self._default_config
        return await self.get_client(
            hosts=config["hosts"],
            username=config.get("username"),
            password=config.get("password"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 120),
        )

    def get_default_connection(self, alias: str = "default"):
        """
        获取基于环境变量配置的默认 elasticsearch_dsl async connection

        Args:
            alias: 连接别名，默认为 "default"

        Returns:
            elasticsearch_dsl 的 async connection 对象
        """
        # 获取或创建默认配置
        if self._default_config is None:
            self._default_config = get_default_es_config()

        config = self._default_config
        return self.create_async_connection(
            hosts=config["hosts"],
            username=config.get("username"),
            password=config.get("password"),
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 120),
            alias=alias,
        )

    async def remove_client(
        self,
        hosts: List[str],
        username: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> bool:
        """
        移除指定的客户端

        Args:
            hosts: Elasticsearch主机列表
            username: 用户名
            api_key: API密钥

        Returns:
            bool: 是否成功移除
        """
        cache_key = get_cache_key(hosts, username, api_key)

        async with self._lock:
            if cache_key in self._clients:
                client_wrapper = self._clients[cache_key]
                try:
                    await client_wrapper.close()
                except Exception as e:
                    logger.error(
                        "Error closing Elasticsearch client during removal: %s", e
                    )

                del self._clients[cache_key]
                logger.info("Elasticsearch client %s removed from cache", hosts)
                return True
            else:
                logger.warning("Elasticsearch client %s not found in cache", hosts)
                return False

    async def close_all_clients(self) -> None:
        """关闭所有缓存的客户端"""
        async with self._lock:
            for cache_key, client_wrapper in self._clients.items():
                try:
                    await client_wrapper.close()
                except Exception as e:
                    logger.error(
                        "Error closing Elasticsearch client %s: %s", cache_key, e
                    )

            self._clients.clear()
            logger.info("All Elasticsearch clients closed and cleared from cache")
