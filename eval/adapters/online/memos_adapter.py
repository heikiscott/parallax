"""
Memos Adapter

适配 Memos 在线 API 的评测框架。
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from rich.console import Console

from eval.adapters.online.base import OnlineAPIAdapter
from eval.adapters.registry import register_adapter
from eval.core.data_models import Conversation, SearchResult

logger = logging.getLogger(__name__)


@register_adapter("memos")
class MemosAdapter(OnlineAPIAdapter):
    """
    Memos 在线 API 适配器
    
    支持：
    - 记忆摄入（支持对话上下文）
    - 记忆检索
    
    官方 API 支持的参数：
    - user_id (必需) - 格式：{conv_id}_{speaker}，已包含会话信息
    - query (必需)
    - memory_limit_number (可选，默认 6)
    
    注意：不使用 conversation_id 参数，因为 user_id 已经包含了会话信息
    
    配置示例：
    ```yaml
    adapter: "memos"
    api_url: "${MEMOS_URL}"
    api_key: "${MEMOS_KEY}"
    ```
    """
    
    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)
        
        # 获取 API 配置
        self.api_url = config.get("api_url", "")
        if not self.api_url:
            raise ValueError("Memos API URL is required. Set 'api_url' in config.")
        
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memos API key is required. Set 'api_key' in config.")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }
        
        # 检索配置（只保留 batch_size 和 max_retries，其他参数不被官方 API 支持）
        self.batch_size = config.get("batch_size", 9999)  # Memos 支持大批量
        self.max_retries = config.get("max_retries", 5)

        self.console = Console()

        logger.info(f"API URL: {self.api_url}")
    
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Dict[str, Any]:
        """
        摄入对话数据到 Memos
        
        Memos API 特点：
        - 需要 user_id 和 conversation_id
        - 支持大批量添加
        - 消息需要包含 chat_time
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print(f"Stage 1: Adding to Memos", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        
        conversation_ids = []
        
        for conv in conversations:
            conv_id = conv.conversation_id
            conversation_ids.append(conv_id)
            
            # 🔥 检测是否需要双视角处理
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
            
            self.console.print(f"\n📥 Adding conversation: {conv_id}", style="cyan")
            
            if need_dual_perspective:
                # 双视角处理（Locomo 风格数据）
                self.console.print(f"   Mode: Dual Perspective", style="dim")
                self._add_dual_perspective(conv, conv_id)
            else:
                # 单视角处理（标准 user/assistant 数据）
                self.console.print(f"   Mode: Single Perspective", style="dim")
                self._add_single_perspective(conv, conv_id)
            
            self.console.print(f"   ✅ Added successfully", style="green")
        
        self.console.print(f"\n✅ All conversations added to Memos", style="bold green")
        
        # 返回元数据
        return {
            "type": "online_api",
            "system": "memos",
            "conversation_ids": conversation_ids,
        }
    
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
        speaker_a_lower = speaker_a.lower()
        speaker_b_lower = speaker_b.lower()
        
        # 检查是否是 user/assistant 相关的名称（放松条件）
        def is_standard_role(speaker: str) -> bool:
            speaker = speaker.lower()
            # 完全匹配
            if speaker in ["user", "assistant"]:
                return True
            # 以 user 或 assistant 开头（处理 user_123, assistant_456 等）
            if speaker.startswith("user") or speaker.startswith("assistant"):
                return True
            return False
        
        # 只有当两个 speaker 都不是标准角色时，才需要双视角
        return not (is_standard_role(speaker_a) or is_standard_role(speaker_b))
    
    def _add_single_perspective(self, conv: Conversation, conv_id: str):
        """单视角添加（用于标准 user/assistant 数据）"""
        messages = self._conversation_to_messages(conv, format_type="memos")
        user_id = self._extract_user_id(conv, speaker="speaker_a")
        
        self.console.print(f"   User ID: {user_id}", style="dim")
        self.console.print(f"   Messages: {len(messages)}", style="dim")
        
        self._send_messages_to_api(messages, user_id, conv_id)
    
    def _add_dual_perspective(self, conv: Conversation, conv_id: str):
        """双视角添加（用于 Locomo 风格数据）"""
        # 从 speaker_a 的视角
        speaker_a_messages = self._conversation_to_messages(
            conv, 
            format_type="memos",
            perspective="speaker_a"
        )
        speaker_a_id = self._extract_user_id(conv, speaker="speaker_a")
        
        # 从 speaker_b 的视角
        speaker_b_messages = self._conversation_to_messages(
            conv,
            format_type="memos",
            perspective="speaker_b"
        )
        speaker_b_id = self._extract_user_id(conv, speaker="speaker_b")
        
        self.console.print(f"   Speaker A ID: {speaker_a_id}", style="dim")
        self.console.print(f"   Speaker A Messages: {len(speaker_a_messages)}", style="dim")
        self.console.print(f"   Speaker B ID: {speaker_b_id}", style="dim")
        self.console.print(f"   Speaker B Messages: {len(speaker_b_messages)}", style="dim")
        
        # 分别发送
        self._send_messages_to_api(speaker_a_messages, speaker_a_id, conv_id)
        self._send_messages_to_api(speaker_b_messages, speaker_b_id, conv_id)
    
    def _send_messages_to_api(self, messages: List[Dict], user_id: str, conv_id: str):
        """发送消息到 Memos API"""
        url = f"{self.api_url}/add/message"
        
        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            
            payload = json.dumps(
                {
                    "messages": batch_messages,
                    "user_id": user_id,
                    "conversation_id": conv_id,
                },
                ensure_ascii=False
            )
            
            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    response = requests.post(url, data=payload, headers=self.headers, timeout=60)
                    
                    if response.status_code != 200:
                        raise Exception(f"HTTP {response.status_code}: {response.text}")
                    
                    result = response.json()
                    if result.get("message") != "ok":
                        raise Exception(f"API error: {result}")
                    
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        self.console.print(
                            f"   ⚠️  Retry {attempt + 1}/{self.max_retries}: {e}", 
                            style="yellow"
                        )
                        time.sleep(2 ** attempt)
                    else:
                        self.console.print(f"   ❌ Failed after {self.max_retries} retries: {e}", style="red")
                        raise e
    
    def _search_single_user(self, query: str, user_id: str, top_k: int) -> Dict[str, Any]:
        """
        单用户搜索（内部方法）
        
        Args:
            query: 查询文本
            user_id: 用户ID（格式：{conv_id}_{speaker}，已包含会话信息）
            top_k: 返回记忆数量
        
        Returns:
            搜索结果字典：
            {
                "text_mem": [{"memories": [...]}],
                "pref_string": "Explicit Preference:\n1. ..."
            }
        
        注意：
            不需要传递 conversation_id 参数，因为 user_id 已经包含了会话信息。
            例如：user_id="locomo_0_Caroline" 已经唯一标识了 locomo_0 这个会话。
        """
        url = f"{self.api_url}/search/memory"
        
        # 只使用官方必需的参数
        payload_dict = {
            "query": query,
            "user_id": user_id,
            "memory_limit_number": top_k,
        }
        
        payload = json.dumps(payload_dict, ensure_ascii=False)
        
        # 重试机制
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, data=payload, headers=self.headers, timeout=60)
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                
                result = response.json()
                if result.get("message") != "ok":
                    raise Exception(f"API error: {result}")
                
                data = result.get("data", {})
                text_mem_res = data.get("memory_detail_list", [])
                pref_mem_res = data.get("preference_detail_list", [])
                preference_note = data.get("preference_note", "")
                
                # 标准化字段名：将 memory_value 重命名为 memory
                for i in text_mem_res:
                    i.update({"memory": i.pop("memory_value", i.get("memory", ""))})
                
                # 格式化偏好字符串
                explicit_prefs = [
                    p["preference"]
                    for p in pref_mem_res
                    if p.get("preference_type", "") == "explicit_preference"
                ]
                implicit_prefs = [
                    p["preference"]
                    for p in pref_mem_res
                    if p.get("preference_type", "") == "implicit_preference"
                ]
                
                pref_parts = []
                if explicit_prefs:
                    pref_parts.append(
                        "Explicit Preference:\n"
                        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(explicit_prefs))
                    )
                if implicit_prefs:
                    pref_parts.append(
                        "Implicit Preference:\n"
                        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(implicit_prefs))
                    )
                
                pref_string = "\n".join(pref_parts) + preference_note
                
                return {"text_mem": [{"memories": text_mem_res}], "pref_string": pref_string}
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        
        return {"text_mem": [{"memories": []}], "pref_string": ""}
    
    async def search(
        self,
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        从 Memos 检索相关记忆
        
        Memos 特点：
        - 支持偏好提取（explicit/implicit preferences）
        - 支持多种检索模式
        - 🔥 支持双视角搜索（Locomo风格数据）
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
            search_data = self._search_single_user(query, user_id, top_k)
        except Exception as e:
            self.console.print(f"❌ Memos search error: {e}", style="red")
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": str(e)}
            )
        
        # 转换为标准 SearchResult 格式
        search_results = []
        for item in search_data["text_mem"][0]["memories"]:
            created_at = item.get("memory_time") or item.get("create_time", "")
            search_results.append({
                "content": item.get("memory", ""),
                "score": item.get("relativity", item.get("score", 0.0)),
                "user_id": user_id,  # 🔥 添加 user_id 标记记忆来源
                "metadata": {
                    "memory_id": item.get("id", ""),
                    "created_at": str(created_at) if created_at else "",
                    "memory_type": item.get("memory_type", ""),
                    "confidence": item.get("confidence", 0.0),
                    "tags": item.get("tags", []),
                }
            })
        
        # 偏好信息已经格式化好了
        pref_string = search_data.get("pref_string", "")
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=search_results,
            retrieval_metadata={
                "system": "memos",
                "preferences": {"pref_string": pref_string},
                "top_k": top_k,
                "user_ids": [user_id],  # 单视角：只有一个 user_id
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
        """
        双视角搜索（用于自定义 speaker 名称的数据）
        
        同时搜索两个 speaker 的记忆并合并结果
        """
        
        try:
            # 分别搜索两个 user_id
            search_a_results = self._search_single_user(query, speaker_a_user_id, top_k)
            search_b_results = self._search_single_user(query, speaker_b_user_id, top_k)
        except Exception as e:
            self.console.print(f"❌ Memos dual search error: {e}", style="red")
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "error": str(e),
                    "user_ids": [speaker_a_user_id, speaker_b_user_id],
                    "dual_perspective": True,
                }
            )
        
        # 🔥 构建详细的 results 列表（为每条记忆添加 user_id）
        all_results = []
        
        # Speaker A 的记忆
        for memory in search_a_results["text_mem"][0]["memories"]:
            all_results.append({
                "content": memory.get("memory", ""),
                "score": memory.get("relativity", 0.0),  # 🔥 修复：使用 relativity 字段
                "user_id": speaker_a_user_id,  # 标记来源
                "metadata": {
                    "memory_id": memory.get("memory_id", ""),
                    "created_at": memory.get("created_at", ""),
                    "memory_type": memory.get("memory_type", ""),
                    "confidence": memory.get("confidence", 0.0),
                    "tags": memory.get("tags", []),
                }
            })
        
        # Speaker B 的记忆
        for memory in search_b_results["text_mem"][0]["memories"]:
            all_results.append({
                "content": memory.get("memory", ""),
                "score": memory.get("relativity", 0.0),  # 🔥 修复：使用 relativity 字段
                "user_id": speaker_b_user_id,  # 标记来源
                "metadata": {
                    "memory_id": memory.get("memory_id", ""),
                    "created_at": memory.get("created_at", ""),
                    "memory_type": memory.get("memory_type", ""),
                    "confidence": memory.get("confidence", 0.0),
                    "tags": memory.get("tags", []),
                }
            })
        
        # 合并两个 speaker 的记忆和偏好（用于 formatted_context）
        speaker_a_context = (
            "\n".join([i["memory"] for i in search_a_results["text_mem"][0]["memories"]])
            + f"\n{search_a_results.get('pref_string', '')}"
        )
        speaker_b_context = (
            "\n".join([i["memory"] for i in search_b_results["text_mem"][0]["memories"]])
            + f"\n{search_b_results.get('pref_string', '')}"
        )
        
        # 使用 default template 格式化
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_context,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_context,
        )
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,  # 🔥 返回详细的记忆列表（每条带 user_id）
            retrieval_metadata={
                "system": "memos",
                "dual_perspective": True,
                "formatted_context": formatted_context,  # 套用 template 后的结果
                "top_k": top_k,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "preferences": {
                    "speaker_a_pref": search_a_results.get("pref_string", ""),
                    "speaker_b_pref": search_b_results.get("pref_string", ""),
                }
            }
        )
    
    def get_system_info(self) -> Dict[str, Any]:
        """返回系统信息"""
        return {
            "name": "Memos",
            "type": "online_api",
            "description": "Memos - Memory System with Preference Support",
            "adapter": "MemosAdapter",
        }

