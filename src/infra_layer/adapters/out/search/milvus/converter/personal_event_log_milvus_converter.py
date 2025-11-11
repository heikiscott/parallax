"""
个人事件日志 Milvus 转换器

负责将 MongoDB 的 PersonalEventLog 文档转换为 Milvus Collection 实体。
"""

from typing import Dict, Any
import json

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from infra_layer.adapters.out.persistence.document.memory.personal_event_log import (
    PersonalEventLog as MongoPersonalEventLog,
)

logger = get_logger(__name__)


class PersonalEventLogMilvusConverter(BaseMilvusConverter[EpisodicMemoryCollection]):
    """
    PersonalEventLog Milvus 转换器
    
    将 MongoDB 的 PersonalEventLog 文档转换为 Milvus Collection 实体。
    注意：使用现有的 EpisodicMemoryCollection，通过 memory_sub_type 字段区分。
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoPersonalEventLog) -> Dict[str, Any]:
        """
        从 MongoDB PersonalEventLog 文档转换为 Milvus Collection 实体

        Args:
            source_doc: MongoDB 的 PersonalEventLog 文档实例

        Returns:
            Dict[str, Any]: Milvus 实体字典，可直接用于插入
        """
        if source_doc is None:
            raise ValueError("MongoDB 文档不能为空")

        try:
            # 转换时间戳
            timestamp = (
                int(source_doc.timestamp.timestamp())
                if source_doc.timestamp
                else 0
            )
            
            # 构建搜索内容
            search_content = cls._build_search_content(source_doc)
            
            # 创建 Milvus 实体字典
            milvus_entity = {
                # 基础标识字段
                "id": str(source_doc.id) if source_doc.id else "",
                "user_id": source_doc.user_id,
                "group_id": source_doc.group_id or "",
                "participants": getattr(source_doc, 'participants', []),  # 添加 participants
                # 时间字段
                "timestamp": timestamp,
                "start_time": 0,  # 事件日志没有时间范围
                "end_time": 0,
                # 核心内容字段 - 使用 atomic_fact
                "episode": source_doc.atomic_fact,
                "search_content": search_content,
                # 分类字段 - 标记为个人事件日志
                "memory_sub_type": "personal_event_log",
                "event_type": source_doc.event_type or "conversation",
                # 详细信息 JSON
                "metadata": json.dumps(cls._build_detail(source_doc), ensure_ascii=False),
                # 审计字段
                "created_at": (
                    int(source_doc.created_at.timestamp())
                    if source_doc.created_at
                    else 0
                ),
                "updated_at": (
                    int(source_doc.updated_at.timestamp())
                    if source_doc.updated_at
                    else 0
                ),
                # 向量字段
                "vector": source_doc.vector if source_doc.vector else [],
            }

            return milvus_entity

        except Exception as e:
            logger.error("从 MongoDB PersonalEventLog 文档转换为 Milvus 实体失败: %s", e)
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoPersonalEventLog) -> Dict[str, Any]:
        """构建详细信息字典"""
        detail = {
            "parent_episode_id": source_doc.parent_episode_id,
            "participants": source_doc.participants,
            "vector_model": source_doc.vector_model,
            "event_type": source_doc.event_type,
            "extend": source_doc.extend,
        }
        
        # 过滤掉 None 值
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoPersonalEventLog) -> str:
        """构建搜索内容（JSON 列表格式）"""
        text_content = []
        
        if source_doc.atomic_fact:
            text_content.append(source_doc.atomic_fact)
        
        return json.dumps(text_content, ensure_ascii=False)

