"""
Mem0 Adapter

适配 Mem0 在线 API 的评测框架。
参考：https://mem0.ai/

关键特性：
- 双视角处理：为 speaker_a 和 speaker_b 分别存储和检索记忆
- 支持自定义指令（custom_instructions）
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from eval.adapters.online.base import OnlineAPIAdapter
from eval.adapters.registry import register_adapter
from eval.core.data_models import Conversation, SearchResult
from eval.adapters.online.online_api_prompts import (
    MEM0_CUSTOM_INSTRUCTIONS,
    ONLINE_API_ANSWER_PROMPT,
    CONTEXT_TEMPLATE_DEFAULT,
)

logger = logging.getLogger(__name__)


@register_adapter("mem0")
class Mem0Adapter(OnlineAPIAdapter):
    """
    Mem0 在线 API 适配器
    
    支持：
    - 标准记忆存储和检索
    
    配置示例：
    ```yaml
    adapter: "mem0"
    api_key: "${MEM0_API_KEY}"
    batch_size: 2
    ```
    """
    
    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)
        
        # 导入 Mem0 客户端
        try:
            from mem0 import MemoryClient
        except ImportError:
            raise ImportError(
                "Mem0 client not installed. "
                "Please install: pip install mem0ai"
            )
        
        # 初始化 Mem0 客户端
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Mem0 API key is required. Set 'api_key' in config.")
        
        self.client = MemoryClient(api_key=api_key)
        self.batch_size = config.get("batch_size", 2)
        self.max_retries = config.get("max_retries", 5)
        self.max_content_length = config.get("max_content_length", 8000)
        self.console = Console()
        
        # 设置 custom instructions
        # 优先使用 config 中的设置（向后兼容），否则使用默认常量
        custom_instructions = config.get("custom_instructions") or MEM0_CUSTOM_INSTRUCTIONS

        if custom_instructions:
            try:
                self.client.update_project(custom_instructions=custom_instructions)
                logger.info("Custom instructions set successfully")
            except Exception as e:
                logger.warning(f"Failed to set custom instructions: {e}")

        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Max Content Length: {self.max_content_length}")
    
    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        """
        准备阶段：更新项目配置和清理已有数据
        
        Args:
            conversations: 标准格式的对话列表
            **kwargs: 额外参数
        """
        # 检查是否需要清理已有数据
        clean_before_add = self.config.get("clean_before_add", False)
        
        if not clean_before_add:
            self.console.print("   ⏭️  Skipping data cleanup (clean_before_add=false)", style="dim")
            return
        
        self.console.print(f"\n{'='*60}", style="bold yellow")
        self.console.print(f"Preparation: Cleaning existing data", style="bold yellow")
        self.console.print(f"{'='*60}", style="bold yellow")
        
        # 收集所有需要清理的 user_id
        user_ids_to_clean = set()
        
        for conv in conversations:
            # 获取 speaker_a 和 speaker_b 的 user_id
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")
            
            need_dual = self._need_dual_perspective(speaker_a, speaker_b)
            
            user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_a"))
            
            if need_dual:
                user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_b"))
        
        # 清理所有用户数据
        self.console.print(f"\n🗑️  Cleaning data for {len(user_ids_to_clean)} user(s)...", style="yellow")
        
        cleaned_count = 0
        failed_count = 0
        
        for user_id in user_ids_to_clean:
            try:
                self.client.delete_all(user_id=user_id)
                cleaned_count += 1
                self.console.print(f"   ✅ Cleaned: {user_id}", style="green")
            except Exception as e:
                failed_count += 1
                self.console.print(f"   ⚠️  Failed to clean {user_id}: {e}", style="yellow")
        
        self.console.print(
            f"\n✅ Cleanup completed: {cleaned_count} succeeded, {failed_count} failed",
            style="bold green"
        )
    
    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        """
        判断是否需要双视角处理
        
        单视角情况（不需要双视角）:
        - 标准角色: "user"/"assistant"
        - 大小写变体: "User"/"Assistant"
        - 带后缀: "user_123"/"assistant_456"
        
        双视角情况（需要双视角）:
        - 自定义名称: "Elena Rodriguez"/"Alex"
        """
        def is_standard_role(speaker: str) -> bool:
            speaker = speaker.lower()
            # 完全匹配
            if speaker in ["user", "assistant"]:
                return True
            # 以 user 或 assistant 开头
            if speaker.startswith("user") or speaker.startswith("assistant"):
                return True
            return False
        
        # 只有当两个 speaker 都不是标准角色时，才需要双视角
        return not (is_standard_role(speaker_a) or is_standard_role(speaker_b))
    
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Dict[str, Any]:
        """
        摄入对话数据到 Mem0
        
        关键特性：
        - 支持单视角和双视角处理
        - 单视角：标准 user/assistant 数据
        - 双视角：自定义 speaker 名称，为每个 speaker 分别存储记忆
        
        Mem0 API 特点：
        - 需要 user_id 来区分不同用户
        - 支持批量添加（建议 batch_size=2）
        - 支持图记忆（可选）
        - 需要时间戳（Unix timestamp）
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print(f"Stage 1: Adding to Mem0 (Dual Perspective)", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        
        conversation_ids = []
        
        for conv in conversations:
            conv_id = conv.conversation_id
            conversation_ids.append(conv_id)
            
            # 获取 speaker 信息
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")
            
            # 获取 user_id（从 metadata 中提取，已在数据加载时设置好）
            speaker_a_user_id = self._extract_user_id(conv, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conv, speaker="speaker_b")
            
            # 🔥 检测是否需要双视角处理
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
            
            # 获取时间戳（使用第一条消息的时间）
            timestamp = None
            is_fake_timestamp = False
            if conv.messages and conv.messages[0].timestamp:
                timestamp = int(conv.messages[0].timestamp.timestamp())
                is_fake_timestamp = conv.messages[0].metadata.get("is_fake_timestamp", False)
            
            self.console.print(f"\n📥 Adding conversation: {conv_id}", style="cyan")
            if is_fake_timestamp:
                self.console.print(f"   ⚠️  Using fake timestamp (original data has no timestamp)", style="yellow")
            
            if need_dual_perspective:
                # 双视角处理（Locomo 风格数据）
                self.console.print(f"   Mode: Dual Perspective", style="dim")
                await self._add_dual_perspective(conv, speaker_a, speaker_b, speaker_a_user_id, speaker_b_user_id, timestamp)
            else:
                # 单视角处理（标准 user/assistant 数据）
                self.console.print(f"   Mode: Single Perspective", style="dim")
                await self._add_single_perspective(conv, speaker_a_user_id, timestamp)
            
            self.console.print(f"   ✅ Added successfully", style="green")
        
        self.console.print(f"\n✅ All conversations added to Mem0", style="bold green")
        
        # 返回元数据（在线 API 不需要本地索引）
        return {
            "type": "online_api",
            "system": "mem0",
            "conversation_ids": conversation_ids,
        }
    
    async def _add_single_perspective(self, conv: Conversation, user_id: str, timestamp: int):
        """单视角添加（用于标准 user/assistant 数据）"""
        messages = []
        truncated_count = 0
        
        for msg in conv.messages:
            # 标准格式：直接使用 speaker_name: content
            content = f"{msg.speaker_name}: {msg.content}"
            
            # 截断过长的内容（Mem0 API 限制）
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                truncated_count += 1
            
            # 判断 role（user 或 assistant）
            role = "user" if msg.speaker_name.lower().startswith("user") else "assistant"
            messages.append({"role": role, "content": content})
        
        self.console.print(f"   User ID: {user_id}", style="dim")
        self.console.print(f"   Messages: {len(messages)}", style="dim")
        if truncated_count > 0:
            self.console.print(f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)", style="yellow")
        
        await self._add_messages_for_user(messages, user_id, timestamp, "Single User")
    
    async def _add_dual_perspective(
        self, 
        conv: Conversation, 
        speaker_a: str, 
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        timestamp: int
    ):
        """双视角添加（用于自定义 speaker 名称的数据）"""
        # 分别构造两个视角的消息列表
        speaker_a_messages = []
        speaker_b_messages = []
        truncated_count = 0
        
        for msg in conv.messages:
            # 格式：speaker_name: content
            content = f"{msg.speaker_name}: {msg.content}"
            
            # 截断过长的内容（Mem0 API 限制）
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length]
                truncated_count += 1
            
            if msg.speaker_name == speaker_a:
                # speaker_a 说的话
                speaker_a_messages.append({"role": "user", "content": content})
                speaker_b_messages.append({"role": "assistant", "content": content})
            elif msg.speaker_name == speaker_b:
                # speaker_b 说的话
                speaker_a_messages.append({"role": "assistant", "content": content})
                speaker_b_messages.append({"role": "user", "content": content})
        
        self.console.print(f"   Speaker A: {speaker_a} (user_id: {speaker_a_user_id})", style="dim")
        self.console.print(f"   Speaker A Messages: {len(speaker_a_messages)}", style="dim")
        self.console.print(f"   Speaker B: {speaker_b} (user_id: {speaker_b_user_id})", style="dim")
        self.console.print(f"   Speaker B Messages: {len(speaker_b_messages)}", style="dim")
        if truncated_count > 0:
            self.console.print(f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)", style="yellow")
        
        # 分别为两个 user_id 添加消息
        await self._add_messages_for_user(
            speaker_a_messages, 
            speaker_a_user_id, 
            timestamp, 
            f"Speaker A ({speaker_a})"
        )
        await self._add_messages_for_user(
            speaker_b_messages, 
            speaker_b_user_id, 
            timestamp, 
            f"Speaker B ({speaker_b})"
        )
    
    async def _add_messages_for_user(
        self, 
        messages: List[Dict], 
        user_id: str, 
        timestamp: int,
        description: str
    ):
        """
        为单个用户添加消息（带批量和重试）
        
        Args:
            messages: 消息列表
            user_id: 用户 ID
            timestamp: Unix 时间戳
            description: 描述（用于日志）
        """
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    self.client.add(
                        messages=batch_messages,
                        timestamp=timestamp,
                        user_id=user_id,
                    )
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.console.print(
                        f"   ⚠️  [{description}] Retry {attempt + 1}/{self.max_retries}: {e}", 
                            style="yellow"
                        )
                        time.sleep(2 ** attempt)
                    else:
                        self.console.print(
                            f"   ❌ [{description}] Failed after {self.max_retries} retries: {e}", 
                            style="red"
                        )
                        raise e
    
    async def search(
        self, 
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        从 Mem0 检索相关记忆
        
        关键特性：
        - 智能判断是否需要双视角搜索
        - 单视角：搜索一个 user_id
        - 双视角：同时搜索 speaker_a 和 speaker_b，合并结果
        
        Args:
            query: 查询文本
            conversation_id: 对话 ID
            index: 索引元数据（包含 conversation_ids）
            **kwargs: 可选参数，如 top_k, conversation（用于重建缓存）
        
        Returns:
            标准格式的检索结果
        """
        top_k = kwargs.get("top_k", 10)
        
        # 🔥 从 kwargs 直接获取对话信息（不使用缓存）
        conversation = kwargs.get("conversation")
        if conversation:
            speaker_a = conversation.metadata.get("speaker_a", "")
            speaker_b = conversation.metadata.get("speaker_b", "")
            speaker_a_user_id = self._extract_user_id(conversation, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conversation, speaker="speaker_b")
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
        else:
            # 回退方案：使用默认 user_id
            speaker_a_user_id = f"{conversation_id}_speaker_a"
            speaker_b_user_id = f"{conversation_id}_speaker_b"
            speaker_a = "speaker_a"
            speaker_b = "speaker_b"
            need_dual_perspective = False
        
        if need_dual_perspective:
            # 🔥 双视角搜索：从两个 speaker 的视角分别搜索
            return await self._search_dual_perspective(
                query, conversation_id, speaker_a, speaker_b, 
                speaker_a_user_id, speaker_b_user_id, top_k
            )
        else:
            # 单视角搜索（标准 user/assistant 数据）
            return await self._search_single_perspective(
                query, conversation_id, speaker_a_user_id, top_k
            )
    
    async def _search_single_perspective(
        self, query: str, conversation_id: str, user_id: str, top_k: int
    ) -> SearchResult:
        """单视角搜索（用于标准 user/assistant 数据）"""
        
        try:
            results = self.client.search(
                query=query,
                top_k=top_k,
                user_id=user_id,
                filters={"AND": [{"user_id": f"{user_id}"}]},
            )
            
            # 🔍 Debug: 打印原始搜索结果
            self.console.print(f"\n[DEBUG] Mem0 Search Results (Single):", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  User ID: {user_id}", style="dim")
            self.console.print(f"  Results: {json.dumps(results, indent=2, ensure_ascii=False)}", style="dim")
            
        except Exception as e:
            self.console.print(f"❌ Mem0 search error: {e}", style="red")
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": str(e)}
            )
        
        # 🔥 构建详细的 results 列表（为每条记忆添加 user_id）
        memory_results = []
        for memory in results.get("results", []):
            memory_results.append({
                "content": f"{memory['created_at']}: {memory['memory']}",
                "score": memory.get("score", 0.0),
                "user_id": user_id,  # 标记来源
                "metadata": {
                    "id": memory.get("id", ""),
                    "created_at": memory.get("created_at", ""),
                    "memory": memory.get("memory", ""),
                    "user_id": memory.get("user_id", ""),
                }
            })
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=memory_results,  # 🔥 返回详细的记忆列表（每条带 user_id）
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [user_id],
            }
        )
    
    async def _search_dual_perspective(
        self, 
        query: str, 
        conversation_id: str,
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int
    ) -> SearchResult:
        """双视角搜索（用于自定义 speaker 名称的数据）"""
        
        # 双视角搜索：分别搜索两个 user_id
        try:
            search_speaker_a_results = self.client.search(
                query=query,
                top_k=top_k,
                user_id=speaker_a_user_id,
                filters={"AND": [{"user_id": f"{speaker_a_user_id}"}]},
            )
            search_speaker_b_results = self.client.search(
                query=query,
                top_k=top_k,
                user_id=speaker_b_user_id,
                filters={"AND": [{"user_id": f"{speaker_b_user_id}"}]},
            )
            
            # 🔍 Debug: 打印原始搜索结果
            self.console.print(f"\n[DEBUG] Mem0 Search Results (Dual):", style="yellow")
            self.console.print(f"  Query: {query}", style="dim")
            self.console.print(f"  Speaker A ({speaker_a}, user_id={speaker_a_user_id}):", style="dim")
            self.console.print(f"    {json.dumps(search_speaker_a_results, indent=2, ensure_ascii=False)}", style="dim")
            self.console.print(f"  Speaker B ({speaker_b}, user_id={speaker_b_user_id}):", style="dim")
            self.console.print(f"    {json.dumps(search_speaker_b_results, indent=2, ensure_ascii=False)}", style="dim")
            
        except Exception as e:
            self.console.print(f"❌ Mem0 dual search error: {e}", style="red")
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": str(e)}
            )
        
        # 🔥 构建详细的 results 列表（为每条记忆添加 user_id）
        all_results = []
        
        # Speaker A 的记忆
        for memory in search_speaker_a_results.get("results", []):
            all_results.append({
                "content": f"{memory['created_at']}: {memory['memory']}",
                "score": memory.get("score", 0.0),
                "user_id": speaker_a_user_id,  # 标记来源
                "metadata": {
                    "id": memory.get("id", ""),
                    "created_at": memory.get("created_at", ""),
                    "memory": memory.get("memory", ""),
                    "user_id": memory.get("user_id", ""),
                }
            })
        
        # Speaker B 的记忆
        for memory in search_speaker_b_results.get("results", []):
            all_results.append({
                "content": f"{memory['created_at']}: {memory['memory']}",
                "score": memory.get("score", 0.0),
                "user_id": speaker_b_user_id,  # 标记来源
                "metadata": {
                    "id": memory.get("id", ""),
                    "created_at": memory.get("created_at", ""),
                    "memory": memory.get("memory", ""),
                    "user_id": memory.get("user_id", ""),
                }
            })
        
        # 格式化记忆（用于 formatted_context）
        speaker_a_memories = [
            f"{memory['created_at']}: {memory['memory']}"
            for memory in search_speaker_a_results.get("results", [])
        ]
        speaker_b_memories = [
            f"{memory['created_at']}: {memory['memory']}"
            for memory in search_speaker_b_results.get("results", [])
        ]
        
        # 格式化 memories 为可读文本（而不是 JSON 数组）
        speaker_a_memories_text = "\n".join(speaker_a_memories) if speaker_a_memories else "(No memories found)"
        speaker_b_memories_text = "\n".join(speaker_b_memories) if speaker_b_memories else "(No memories found)"
        
        # 使用标准 default template
        context = CONTEXT_TEMPLATE_DEFAULT.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )
        
        # 返回结果
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,  # 🔥 返回详细的记忆列表（每条带 user_id）
            retrieval_metadata={
                "system": "mem0",
                "top_k": top_k,
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": context,  # 🔥 套用 template 后的最终结果
                "speaker_a_memories_count": len(speaker_a_memories),
                "speaker_b_memories_count": len(speaker_b_memories),
            }
        )
    
    def _get_answer_prompt(self) -> str:
        """
        返回 answer prompt

        使用通用 default prompt
        """
        return ONLINE_API_ANSWER_PROMPT
    
    def get_system_info(self) -> Dict[str, Any]:
        """返回系统信息"""
        return {
            "name": "Mem0",
            "type": "online_api",
            "description": "Mem0 - Personalized AI Memory Layer",
            "adapter": "Mem0Adapter",
        }

