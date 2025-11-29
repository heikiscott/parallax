"""
批量将 MongoDB 中现有的 MemUnit.episode 重新同步到 Milvus / ES。

运行方式：
    uv run python src/bootstrap.py demo/tools/resync_memunits.py
"""

import asyncio
from typing import List

from core.di import get_bean_by_type
from core.observation.logger import get_logger
from infra.adapters.out.persistence.document.memory.memunit import MemUnit
from services.memunit_sync import MemUnitSyncService

logger = get_logger(__name__)


async def main() -> None:
    service = get_bean_by_type(MemUnitSyncService)

    memunits: List[MemUnit] = await MemUnit.find_all().to_list()
    if not memunits:
        logger.info("MongoDB 中没有 MemUnit 记录，跳过")
        return

    logger.info("开始重同步 %s 条 MemUnit 记录", len(memunits))
    success = 0
    for memunit in memunits:
        await service.sync_memunit(memunit, sync_to_es=True, sync_to_milvus=True)
        success += 1

    logger.info("完成重同步，成功 %s 条", success)


if __name__ == "__main__":
    asyncio.run(main())

