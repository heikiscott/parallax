"""
个人语义记忆 Milvus 转换器

负责将 MongoDB 的 PersonalSemanticMemory 文档转换为 Milvus Collection 实体。
"""

from typing import Dict, Any
import json
from datetime import datetime

from core.oxm.milvus.base_converter import BaseMilvusConverter
from core.observation.logger import get_logger
from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
    EpisodicMemoryCollection,
)
from infra_layer.adapters.out.persistence.document.memory.personal_semantic_memory import (
    PersonalSemanticMemory as MongoPersonalSemanticMemory,
)

logger = get_logger(__name__)


class PersonalSemanticMemoryMilvusConverter(BaseMilvusConverter[EpisodicMemoryCollection]):
    """
    PersonalSemanticMemory Milvus 转换器
    
    将 MongoDB 的 PersonalSemanticMemory 文档转换为 Milvus Collection 实体。
    注意：使用现有的 EpisodicMemoryCollection，通过 memory_sub_type 字段区分。
    """

    @classmethod
    def from_mongo(cls, source_doc: MongoPersonalSemanticMemory) -> Dict[str, Any]:
        """
        从 MongoDB PersonalSemanticMemory 文档转换为 Milvus Collection 实体

        Args:
            source_doc: MongoDB 的 PersonalSemanticMemory 文档实例

        Returns:
            Dict[str, Any]: Milvus 实体字典，可直接用于插入
        """
        if source_doc is None:
            raise ValueError("MongoDB 文档不能为空")

        try:
            # 解析时间字段
            start_time = 0
            end_time = 0
            
            if source_doc.start_time:
                if isinstance(source_doc.start_time, str):
                    start_dt = datetime.fromisoformat(source_doc.start_time.replace('Z', '+00:00'))
                    start_time = int(start_dt.timestamp())
                elif isinstance(source_doc.start_time, datetime):
                    start_time = int(source_doc.start_time.timestamp())
            
            if source_doc.end_time:
                if isinstance(source_doc.end_time, str):
                    end_dt = datetime.fromisoformat(source_doc.end_time.replace('Z', '+00:00'))
                    end_time = int(end_dt.timestamp())
                elif isinstance(source_doc.end_time, datetime):
                    end_time = int(source_doc.end_time.timestamp())
            
            # 使用 start_time 作为主时间戳
            timestamp = start_time if start_time > 0 else int(datetime.now().timestamp())
            
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
                "start_time": start_time,
                "end_time": end_time,
                # 核心内容字段
                "episode": source_doc.content,
                "search_content": search_content,
                # 分类字段 - 标记为个人语义记忆
                "memory_sub_type": "personal_semantic_memory",
                "event_type": "semantic",
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
            logger.error("从 MongoDB PersonalSemanticMemory 文档转换为 Milvus 实体失败: %s", e)
            raise

    @classmethod
    def _build_detail(cls, source_doc: MongoPersonalSemanticMemory) -> Dict[str, Any]:
        """构建详细信息字典"""
        detail = {
            "parent_episode_id": source_doc.parent_episode_id,
            "duration_days": source_doc.duration_days,
            "evidence": source_doc.evidence,
            "participants": source_doc.participants,
            "vector_model": source_doc.vector_model,
            "extend": source_doc.extend,
        }
        
        # 过滤掉 None 值
        return {k: v for k, v in detail.items() if v is not None}

    @staticmethod
    def _build_search_content(source_doc: MongoPersonalSemanticMemory) -> str:
        """构建搜索内容（JSON 列表格式）"""
        text_content = []
        
        if source_doc.content:
            text_content.append(source_doc.content)
        
        return json.dumps(text_content, ensure_ascii=False)

