"""MemUnit 同步服务

负责将 MemUnit.narrative 同步到 Milvus 和 Elasticsearch（群组记忆）。
PersonalSemanticMemory 和 PersonalEventLog 由 PersonalMemorySyncService 处理。
"""

from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

from infra.adapters.out.persistence.document.memory.memunit import MemUnit
from infra.adapters.out.search.repository.episodic_memory_milvus_repository import (
    EpisodicMemoryMilvusRepository,
)
from infra.adapters.out.search.repository.episodic_memory_es_repository import (
    EpisodicMemoryEsRepository,
)
from agents.deep_infra_vectorize_service import DeepInfraVectorizeServiceInterface
from core.di import get_bean_by_type, service
from utils.datetime_utils import get_now_with_timezone

logger = logging.getLogger(__name__)


@service(name="memunit_sync_service", primary=True)
class MemUnitSyncService:
    """MemUnit 同步服务

    负责将 MemUnit.narrative 同步到 Milvus 和 Elasticsearch（群组记忆）。
    PersonalSemanticMemory 和 PersonalEventLog 由 PersonalMemorySyncService 处理。
    """

    def __init__(
        self,
        episodic_milvus_repo: Optional[EpisodicMemoryMilvusRepository] = None,
        es_repo: Optional[EpisodicMemoryEsRepository] = None,
        vectorize_service: Optional[DeepInfraVectorizeServiceInterface] = None,
    ):
        """初始化同步服务
        
        Args:
            episodic_milvus_repo: 情景记忆 Milvus 仓库实例（可选，不提供则从 DI 获取）
            es_repo: ES 仓库实例（可选，不提供则从 DI 获取）
            vectorize_service: 向量化服务实例（可选，不提供则从 DI 获取）
        """
        self.episodic_milvus_repo = episodic_milvus_repo or get_bean_by_type(
            EpisodicMemoryMilvusRepository
        )
        self.es_repo = es_repo or get_bean_by_type(EpisodicMemoryEsRepository)
        
        if vectorize_service is None:
            from agents.deep_infra_vectorize_service import get_vectorize_service
            self.vectorize_service = get_vectorize_service()
        else:
            self.vectorize_service = vectorize_service
        
        logger.info("MemUnitSyncService 初始化完成（同步 narrative 到 Milvus + ES）")

    async def sync_memunit(
        self, memunit: MemUnit, sync_to_es: bool = True, sync_to_milvus: bool = True
    ) -> Dict[str, int]:
        """同步单个 MemUnit.narrative 到 Milvus 和 ES
        
        Args:
            memunit: MemUnit 文档对象
            sync_to_es: 是否同步到 ES（默认 True）
            sync_to_milvus: 是否同步到 Milvus（默认 True）
            
        Returns:
            同步统计信息 {"narrative": 1, "es_records": 1}
        """
        stats = {"narrative": 0, "es_records": 0}
        
        try:
            # 同步到 Milvus
            if sync_to_milvus:
                # 只同步 narrative（群组记忆）
                if hasattr(memunit, 'narrative') and memunit.narrative:
                    await self._sync_narrative(memunit)
                    stats["narrative"] = 1
                    logger.debug(f"✅ 已同步 narrative 到 Milvus: {memunit.unit_id}")

                # 刷新 Milvus，确保数据写入
                await self.episodic_milvus_repo.flush()
                logger.debug(f"✅ Milvus 数据已刷新: {memunit.unit_id}")

            # 同步到 ES
            if sync_to_es:
                es_count = await self._sync_to_es(memunit)
                stats["es_records"] = es_count
                logger.debug(f"✅ 已同步 {es_count} 条记录到 ES: {memunit.unit_id}")

                # 刷新 ES 索引，确保数据可搜索
                try:
                    client = await self.es_repo.get_client()
                    index_name = self.es_repo.get_index_name()
                    await client.indices.refresh(index=index_name)
                    logger.debug(f"✅ ES 索引已刷新: {index_name}")
                except Exception as e:
                    logger.warning(f"ES 索引刷新失败（可能不影响使用）: {e}")

            logger.info(
                f"MemUnit 同步完成: {memunit.unit_id}, 统计: {stats}"
            )

            return stats

        except Exception as e:
            logger.error(f"MemUnit 同步失败: {memunit.unit_id}, error={e}")
            raise

    async def _sync_narrative(self, memunit: MemUnit) -> None:
        """同步 narrative 到 Milvus

        Args:
            memunit: MemUnit 文档对象
        """
        # 从 MongoDB 读取 embedding（必须存在）
        vector = None
        if hasattr(memunit, 'extend') and memunit.extend and 'embedding' in memunit.extend:
            vector = memunit.extend['embedding']
            # 确保是 list 格式（可能是 numpy array）
            if hasattr(vector, 'tolist'):
                vector = vector.tolist()
            logger.debug(f"从 MongoDB 读取 narrative embedding: {memunit.unit_id}")

        if not vector:
            logger.warning(
                f"narrative 缺少 embedding，跳过同步到 Milvus: {memunit.unit_id}"
            )
            return

        # 准备搜索内容
        search_content = []
        if hasattr(memunit, 'subject') and memunit.subject:
            search_content.append(memunit.subject)
        if hasattr(memunit, 'summary') and memunit.summary:
            search_content.append(memunit.summary)
        if not search_content:
            search_content.append(memunit.narrative[:100])  # 使用 narrative 前 100 字符

        # 确保 vector 是 list 格式
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()

        # MemUnit 的 user_id 始终为 None（群组记忆）
        await self.episodic_milvus_repo.create_and_save_episodic_memory(
            episode_id=str(memunit.unit_id),
            user_id=memunit.user_id,  # None for group memory
            timestamp=memunit.timestamp or get_now_with_timezone(),
            narrative=memunit.narrative,
            search_content=search_content,
            vector=vector,
            user_name=memunit.user_id,
            title=getattr(memunit, 'subject', None),
            summary=getattr(memunit, 'summary', None),
            group_id=getattr(memunit, 'group_id', None),
            participants=getattr(memunit, 'participants', None),
            subject=getattr(memunit, 'subject', None),
            metadata="{}",
            memunit_id_list=[str(memunit.unit_id)],
        )
        logger.debug(f"✅ 已同步 narrative 到 Milvus: {memunit.unit_id}")

    async def _sync_to_es(self, memunit: MemUnit) -> int:
        """同步 MemUnit.narrative 到 ES

        Args:
            memunit: MemUnit 文档对象

        Returns:
            同步的记录数量
        """
        count = 0

        try:
            # 只同步 narrative（群组记忆）
            if hasattr(memunit, 'narrative') and memunit.narrative:
                search_content = []
                if hasattr(memunit, 'subject') and memunit.subject:
                    search_content.append(memunit.subject)
                if hasattr(memunit, 'summary') and memunit.summary:
                    search_content.append(memunit.summary)
                search_content.append(memunit.narrative[:500])

                await self.es_repo.create_and_save_episodic_memory(
                    episode_id=f"{str(memunit.unit_id)}_narrative",
                    user_id=memunit.user_id,
                    timestamp=memunit.timestamp or get_now_with_timezone(),
                    narrative=memunit.narrative,
                    search_content=search_content,
                    user_name=memunit.user_id,
                    title=getattr(memunit, 'subject', None),
                    summary=getattr(memunit, 'summary', None),
                    group_id=getattr(memunit, 'group_id', None),
                    participants=getattr(memunit, 'participants', None),
                    event_type="narrative",  # 标记类型
                    subject=getattr(memunit, 'subject', None),
                    memunit_id_list=[str(memunit.unit_id)],
                )
                count += 1

            return count

        except Exception as e:
            logger.error(f"同步到 ES 失败: {memunit.unit_id}, error={e}")
            return 0

    async def sync_memunits_batch(self, memunits: List[MemUnit]) -> Dict[str, Any]:
        """批量同步 MemUnits 到 Milvus
        
        Args:
            memunits: MemUnit 文档对象列表
            
        Returns:
            批量同步统计信息
        """
        total_stats = {
            "total_memunits": len(memunits),
            "success_memunits": 0,
            "failed_memunits": 0,
            "total_narrative": 0,
        }

        for memunit in memunits:
            try:
                stats = await self.sync_memunit(memunit)
                total_stats["success_memunits"] += 1
                total_stats["total_narrative"] += stats["narrative"]
            except Exception as e:
                logger.error(f"批量同步失败: {memunit.unit_id}, error={e}")
                total_stats["failed_memunits"] += 1
                continue
        
        logger.info(f"批量同步完成: {total_stats}")
        return total_stats


def get_memunit_sync_service() -> MemUnitSyncService:
    """获取 MemUnit 同步服务实例
    
    通过依赖注入框架获取服务实例，支持单例模式。
    """
    from core.di import get_bean
    return get_bean("memunit_sync_service")
