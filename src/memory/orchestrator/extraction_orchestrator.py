from dataclasses import dataclass
import datetime
import time
from typing import List, Optional

from config import load_config
from core.observation.logger import get_logger

from providers.llm.llm_provider import LLMProvider


def _get_llm_config():
    """从 YAML 配置文件读取 LLM 配置"""
    return load_config("src/providers").llm


from ..extraction.memunit import (
    ConvMemUnitExtractor,
    ConversationMemUnitExtractRequest,
    RawData,
    StatusResult,
)
from ..schema import SourceType
from ..schema import MemUnit, MemoryType, Memory
from ..extraction.memory import (
    EpisodeMemoryExtractor,
    EpisodeMemoryExtractRequest,
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
    GroupProfileMemoryExtractor,
    GroupProfileMemoryExtractRequest,
    EventLogExtractor,
    SemanticMemoryExtractor,
)


logger = get_logger(__name__)


@dataclass
class MemorizeRequest:
    history_raw_data_list: list[RawData]
    new_raw_data_list: list[RawData]
    source_type: SourceType
    # 整个group全量的user_id列表
    user_id_list: List[str]
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    current_time: Optional[datetime] = None
    # 可选的提取控制参数
    enable_semantic_extraction: bool = True  # 是否提取语义记忆
    enable_event_log_extraction: bool = True  # 是否提取事件日志
    # 对话元数据字段（对应 ConversationMeta）
    scene: Optional[str] = None  # 场景标识符，如 "company"、"work" 等
    scene_desc: Optional[dict] = None  # 场景描述信息，如 {"bot_ids": ["aaa", "bbb"]}


@dataclass
class MemorizeOfflineRequest:
    memorize_from: datetime
    memorize_to: datetime


class ExtractionOrchestrator:
    """记忆提取编排器 - 负责编排各种记忆提取器"""

    def __init__(self):
        # 从 YAML 配置文件读取 LLM 配置
        llm_cfg = _get_llm_config()

        # Conversation MemUnit LLM Provider
        self.conv_memcall_llm_provider = LLMProvider(
            provider_type=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
            temperature=float(llm_cfg.temperature),
            max_tokens=int(llm_cfg.max_tokens),
        )

        # Event Log Extractor LLM Provider
        self.event_log_llm_provider = LLMProvider(
            provider_type=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
            temperature=float(llm_cfg.temperature),
            max_tokens=int(llm_cfg.max_tokens),
        )

        # Event Log Extractor - 延迟初始化
        self._event_log_extractor = None

        # Episode Memory Extractor LLM Provider
        self.episode_memory_extractor_llm_provider = LLMProvider(
            provider_type=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
            temperature=float(llm_cfg.temperature),
            max_tokens=int(llm_cfg.max_tokens),
        )

        # Profile Memory Extractor LLM Provider
        self.profile_memory_extractor_llm_provider = LLMProvider(
            provider_type=llm_cfg.provider,
            model=llm_cfg.model,
            base_url=llm_cfg.base_url,
            api_key=llm_cfg.api_key,
            temperature=float(llm_cfg.temperature),
            max_tokens=int(llm_cfg.max_tokens),
        )

        

    async def extract_memunit(
        self,
        history_raw_data_list: list[RawData],
        new_raw_data_list: list[RawData],
        source_type: SourceType,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        user_id_list: Optional[List[str]] = None,
        old_memory_list: Optional[List[Memory]] = None,
        enable_semantic_extraction: bool = True,
        enable_event_log_extraction: bool = True,
    ) -> tuple[Optional[MemUnit], Optional[StatusResult]]:
        """
        提取 MemUnit（包含可选的语义记忆和事件日志提取）
        
        Args:
            history_raw_data_list: 历史消息列表
            new_raw_data_list: 新消息列表
            raw_data_type: 数据类型
            group_id: 群组ID
            group_name: 群组名称
            user_id_list: 用户ID列表
            old_memory_list: 历史记忆列表
            enable_semantic_extraction: 是否提取语义记忆（默认True）
            enable_event_log_extraction: 是否提取事件日志（默认True）
            
        Returns:
            (MemUnit, StatusResult) 或 (None, StatusResult)
        """
        logger = get_logger(__name__)
        now = time.time()
        
        # 1. 提取基础 MemUnit（包括可选的语义记忆）
        request = ConversationMemUnitExtractRequest(
            history_raw_data_list,
            new_raw_data_list,
            user_id_list=user_id_list,
            group_id=group_id,
            group_name=group_name,
            old_memory_list=old_memory_list,
        )
        extractor = ConvMemUnitExtractor(self.conv_memcall_llm_provider)
        memunit, status_result = await extractor.extract_memunit(
            request, 
            use_semantic_extraction=enable_semantic_extraction
        )
        
        # 2. 如果成功提取 MemUnit，且启用了 Event Log 提取
        if memunit and enable_event_log_extraction and hasattr(memunit, 'narrative') and memunit.narrative:
            if self._event_log_extractor is None:
                self._event_log_extractor = EventLogExtractor(llm_provider=self.event_log_llm_provider)

            logger.debug(f"开始提取 Event Log: {memunit.unit_id}")
            event_log = await self._event_log_extractor.extract_event_log(
                episode_text=memunit.narrative,
                timestamp=memunit.timestamp
            )

            if event_log:
                memunit.event_log = event_log
                logger.debug(f"Event Log 提取成功: {memunit.unit_id}")
        
        logger.debug(
            f"提取MemUnit完成, raw_data_type: {raw_data_type}, "
            f"semantic_extraction={enable_semantic_extraction}, "
            f"event_log_extraction={enable_event_log_extraction}, "
            f"耗时: {time.time() - now}秒"
        )
        
        return memunit, status_result

    async def extract_memory(
        self,
        memunit_list: list[MemUnit],
        memory_type: MemoryType,
        user_ids: List[str],
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
        old_memory_list: Optional[List[Memory]] = None,
        user_organization: Optional[List] = None,
        episode_memory: Optional[Memory] = None,  # 用于个人语义记忆和事件日志提取
    ):
        """
        提取记忆
        
        Returns:
            - EPISODE_SUMMARY/PROFILE/GROUP_PROFILE: 返回 List[Memory]
            - SEMANTIC_SUMMARY: 返回 List[SemanticMemoryItem]
            - EVENT_LOG: 返回 EventLog
        """
        extractor = None
        request = None

        if memory_type == MemoryType.EPISODE_SUMMARY:
            extractor = EpisodeMemoryExtractor(
                self.episode_memory_extractor_llm_provider
            )
            request = EpisodeMemoryExtractRequest(
                memunit_list=memunit_list,
                user_id_list=user_ids,
                group_id=group_id,
                old_memory_list=old_memory_list,
            )
        elif memory_type == MemoryType.PROFILE:
            if memunit_list[0].type == SourceType.CONVERSATION:
                extractor = ProfileMemoryExtractor(
                    self.profile_memory_extractor_llm_provider
                )
                request = ProfileMemoryExtractRequest(
                    memunit_list=memunit_list,
                    user_id_list=user_ids,
                    group_id=group_id,
                    old_memory_list=old_memory_list,
                )
        elif memory_type == MemoryType.GROUP_PROFILE:
            extractor = GroupProfileMemoryExtractor(
                self.profile_memory_extractor_llm_provider
            )
            request = GroupProfileMemoryExtractRequest(
                memunit_list=memunit_list,
                user_id_list=user_ids,
                group_id=group_id,
                group_name=group_name,
                old_memory_list=old_memory_list,
                user_organization=None,
            )
        elif memory_type == MemoryType.SEMANTIC_SUMMARY and episode_memory:
            # 为个人 episode 提取语义记忆
            logger.debug(f"开始为个人 episode 提取语义记忆: user_id={episode_memory.user_id}")
            
            extractor = SemanticMemoryExtractor(
                llm_provider=self.episode_memory_extractor_llm_provider
            )
            
            semantic_memories = await extractor.generate_semantic_memories_for_episode(
                episode_memory
            )
                        
            return semantic_memories
        
        elif memory_type == MemoryType.EVENT_LOG and episode_memory:
            # 为个人 episode 提取事件日志
            logger.debug(f"开始为个人 episode 提取事件日志: user_id={episode_memory.user_id}")
            
            extractor = EventLogExtractor(
                    llm_provider=self.event_log_llm_provider
                )

            event_log = await extractor.extract_event_log(
                episode_text=episode_memory.narrative,
                timestamp=episode_memory.timestamp
            )
            
            return event_log

        if extractor == None or request == None:
            return []
        return await extractor.extract_memory(request)

