"""
Memu Adapter

适配 Memu 在线 API 的评测框架。
使用 HTTP RESTful API 而不是 Python SDK，避免依赖冲突。
"""
import json
import time
import requests
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from eval.adapters.online_base import OnlineAPIAdapter
from eval.adapters.registry import register_adapter
from eval.core.data_models import Conversation, SearchResult


@register_adapter("memu")
class MemuAdapter(OnlineAPIAdapter):
    """
    Memu 在线 API 适配器
    
    使用 HTTP RESTful API 直接调用，避免 Python SDK 依赖冲突。
    
    支持：
    - 记忆摄入（基于对话上下文）
    - 异步任务状态监控
    - 记忆检索
    
    配置示例：
    ```yaml
    adapter: "memu"
    api_key: "${MEMU_API_KEY}"
    base_url: "https://api.memu.so"  # 可选，默认使用官方 API
    agent_id: "default_agent"  # 可选，默认 agent ID
    agent_name: "Assistant"  # 可选，默认 agent 名称
    task_check_interval: 3  # 可选，任务状态检查间隔（秒）
    task_timeout: 90  # 可选，任务超时时间（秒）
    ```
    """
    
    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)
        
        # 获取配置
        api_key = config.get("api_key", "")
        if not api_key:
            raise ValueError("Memu API key is required. Set 'api_key' in config.")
        
        self.base_url = config.get("base_url", "https://api.memu.so").rstrip('/')
        self.agent_id = config.get("agent_id", "default_agent")
        self.agent_name = config.get("agent_name", "Assistant")
        self.task_check_interval = config.get("task_check_interval", 3)
        self.task_timeout = config.get("task_timeout", 90)
        self.max_retries = config.get("max_retries", 5)
        
        # HTTP headers
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        self.console = Console()
        self.console.print(f"   Base URL: {self.base_url}", style="dim")
        self.console.print(f"   Agent: {self.agent_name} ({self.agent_id})", style="dim")
    
    async def add(
        self, 
        conversations: List[Conversation],
        **kwargs
    ) -> Dict[str, Any]:
        """
        摄入对话数据到 Memu
        
        Memu API 特点：
        - 使用 HTTP RESTful API 提交记忆
        - 返回异步任务 ID，需要轮询状态
        - 任务完成后才能搜索
        - 支持双视角处理（为两个 speaker 分别存储记忆）
        """
        self.console.print(f"\n{'='*60}", style="bold cyan")
        self.console.print(f"Stage 1: Adding to Memu", style="bold cyan")
        self.console.print(f"{'='*60}", style="bold cyan")
        
        conversation_ids = []
        task_ids = []  # 收集所有任务 ID
        
        for conv in conversations:
            conv_id = conv.conversation_id
            conversation_ids.append(conv_id)
            
            # 获取双视角信息
            speaker_a = conv.metadata.get("speaker_a", "User")
            speaker_b = conv.metadata.get("speaker_b", "Assistant")
            speaker_a_user_id = self._extract_user_id(conv, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conv, speaker="speaker_b")
            
            # 判断是否需要双视角
            need_dual_perspective = self._need_dual_perspective(speaker_a, speaker_b)
            
            self.console.print(f"\n📥 Adding conversation: {conv_id}", style="cyan")
            self.console.print(f"   Speaker A: {speaker_a} ({speaker_a_user_id})", style="dim")
            self.console.print(f"   Speaker B: {speaker_b} ({speaker_b_user_id})", style="dim")
            self.console.print(f"   Dual Perspective: {need_dual_perspective}", style="dim")
            
            # 获取 session_date（ISO 格式日期）
            session_date = None
            if conv.messages and conv.messages[0].timestamp:
                session_date = conv.messages[0].timestamp.strftime("%Y-%m-%d")
            else:
                from datetime import datetime
                session_date = datetime.now().strftime("%Y-%m-%d")
            
            # 根据视角需求添加记忆
            if need_dual_perspective:
                # 双视角：分别为 speaker_a 和 speaker_b 添加记忆
                task_id_a = await self._add_single_user(
                    conv, speaker_a_user_id, speaker_a, session_date, perspective="speaker_a"
                )
                task_id_b = await self._add_single_user(
                    conv, speaker_b_user_id, speaker_b, session_date, perspective="speaker_b"
                )
                if task_id_a:
                    task_ids.append(task_id_a)
                if task_id_b:
                    task_ids.append(task_id_b)
            else:
                # 单视角：只为 speaker_a 添加记忆
                task_id = await self._add_single_user(
                    conv, speaker_a_user_id, speaker_a, session_date, perspective="speaker_a"
                )
                if task_id:
                    task_ids.append(task_id)
        
        # 等待所有任务完成
        if task_ids:
            self.console.print(f"\n⏳ Waiting for {len(task_ids)} task(s) to complete...", style="bold yellow")
            self._wait_for_all_tasks(task_ids)
        
        self.console.print(f"\n✅ All conversations added to Memu", style="bold green")
        
        # 返回元数据
        return {
            "type": "online_api",
            "system": "memu",
            "conversation_ids": conversation_ids,
            "task_ids": task_ids,
        }
    
    def _need_dual_perspective(self, speaker_a: str, speaker_b: str) -> bool:
        """
        判断是否需要双视角处理
        
        单视角情况（不需要双视角）:
        - 标准角色: "user"/"assistant"
        - 大小写变体: "User"/"Assistant"
        - 带后缀: "user_123"/"assistant_456"
        
        双视角情况（需要双视角）:
        - 自定义名称: "Caroline"/"Manu"
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
        
        return not (is_standard_role(speaker_a) and is_standard_role(speaker_b))
    
    async def _add_single_user(
        self,
        conv: Conversation,
        user_id: str,
        user_name: str,
        session_date: str,
        perspective: str
    ) -> str:
        """
        为单个用户添加记忆
        
        Args:
            conv: 对话对象
            user_id: 用户 ID
            user_name: 用户名称
            session_date: 会话日期
            perspective: 视角（speaker_a 或 speaker_b）
        
        Returns:
            task_id: 任务 ID（如果成功）
        """
        # 转换为 Memu API 格式（指定视角）
        base_messages = self._conversation_to_messages(conv, format_type="basic", perspective=perspective)
        
        # 添加 Memu API 需要的额外字段
        conversation_messages = []
        for i, msg in enumerate(conv.messages):
            # 构造消息时间（ISO 格式）
            msg_time = msg.timestamp.isoformat() + "Z" if msg.timestamp else None
            
            conversation_messages.append({
                "role": base_messages[i]["role"],
                "name": msg.speaker_name or user_name,
                "time": msg_time,
                "content": base_messages[i]["content"]
            })
        
        self.console.print(f"   📤 Adding for {user_name} ({user_id}): {len(conversation_messages)} messages", style="dim")
        
        # 构造请求 payload
        payload = {
            "conversation": conversation_messages,
            "user_id": user_id,
            "user_name": user_name,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "session_date": session_date
        }
        
        # 提交任务（带重试）
        task_id = None
        for attempt in range(self.max_retries):
            try:
                url = f"{self.base_url}/api/v1/memory/memorize"
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                
                result = response.json()
                task_id = result.get("task_id")
                status = result.get("status")
                
                self.console.print(f"      ✅ Task created: {task_id} (status: {status})", style="green")
                break
                
            except Exception as e:
                if attempt < self.max_retries - 1:
                    self.console.print(
                        f"      ⚠️  Retry {attempt + 1}/{self.max_retries}: {e}", 
                        style="yellow"
                    )
                    time.sleep(2 ** attempt)
                else:
                    self.console.print(
                        f"      ❌ Failed after {self.max_retries} retries: {e}", 
                        style="red"
                    )
                    raise e
        
        return task_id
    
    def _wait_for_all_tasks(self, task_ids: List[str]) -> bool:
        """
        等待所有任务完成
        
        Args:
            task_ids: 任务 ID 列表
        
        Returns:
            是否所有任务都成功完成
        """
        if not task_ids:
            return True
        
        start_time = time.time()
        pending_tasks = set(task_ids)
        
        # 显示进度
        total_tasks = len(task_ids)
        
        while time.time() - start_time < self.task_timeout:
            completed_in_round = []
            failed_in_round = []
            
            for task_id in list(pending_tasks):
                try:
                    url = f"{self.base_url}/api/v1/memory/memorize/status/{task_id}"
                    response = requests.get(url, headers=self.headers)
                    response.raise_for_status()
                    result = response.json()
                    status = result.get("status")
                    
                    # Memu API 返回大写状态：PENDING/PROCESSING/SUCCESS/FAILED
                    if status in ["SUCCESS", "COMPLETED"]:
                        completed_in_round.append(task_id)
                    elif status in ["FAILED", "FAILURE"]:
                        failed_in_round.append(task_id)
                        self.console.print(
                            f"   ❌ Task {task_id} failed: {result.get('detail_info', 'Unknown error')}", 
                            style="red"
                        )
                    
                except Exception as e:
                    self.console.print(
                        f"   ⚠️  Error checking task {task_id}: {e}", 
                        style="yellow"
                    )
            
            # 移除已完成/失败的任务
            for task_id in completed_in_round + failed_in_round:
                pending_tasks.remove(task_id)
            
            # 更新进度
            completed_count = total_tasks - len(pending_tasks)
            if completed_in_round or failed_in_round:
                self.console.print(
                    f"   📊 Progress: {completed_count}/{total_tasks} tasks completed",
                    style="cyan"
                )
            
            # 如果所有任务都完成了
            if not pending_tasks:
                self.console.print(
                    f"   ✅ All {total_tasks} tasks completed!",
                    style="bold green"
                )
                return len(failed_in_round) == 0
            
            # 等待后重试
            if pending_tasks:
                elapsed = time.time() - start_time
                self.console.print(
                    f"   ⏳ {len(pending_tasks)} task(s) still processing... ({elapsed:.0f}s elapsed)",
                    style="dim"
                )
                time.sleep(self.task_check_interval)
        
        # 超时
        self.console.print(
            f"   ⚠️  Timeout: {len(pending_tasks)} task(s) not completed within {self.task_timeout}s",
            style="yellow"
        )
        return False
    
    async def search(
        self, 
        query: str,
        conversation_id: str,
        index: Any,
        **kwargs
    ) -> SearchResult:
        """
        从 Memu 检索相关记忆
        
        使用 HTTP RESTful API 直接调用搜索接口
        支持双视角搜索
        """
        # 获取对话信息
        conversation = kwargs.get("conversation")
        if conversation:
            speaker_a = conversation.metadata.get("speaker_a", "")
            speaker_b = conversation.metadata.get("speaker_b", "")
            speaker_a_user_id = self._extract_user_id(conversation, speaker="speaker_a")
            speaker_b_user_id = self._extract_user_id(conversation, speaker="speaker_b")
            need_dual = self._need_dual_perspective(speaker_a, speaker_b)
        else:
            # 回退方案：使用默认 user_id
            speaker_a_user_id = f"{conversation_id}_speaker_a"
            speaker_b_user_id = f"{conversation_id}_speaker_b"
            speaker_a = "speaker_a"
            speaker_b = "speaker_b"
            need_dual = False
        
        top_k = kwargs.get("top_k", 10)
        min_similarity = kwargs.get("min_similarity", 0.3)
        
        if need_dual:
            # 双视角搜索
            return await self._search_dual_perspective(
                query, conversation_id, speaker_a, speaker_b, 
                speaker_a_user_id, speaker_b_user_id, top_k, min_similarity
            )
        else:
            # 单视角搜索
            return await self._search_single_perspective(
                query, conversation_id, speaker_a_user_id, top_k, min_similarity
            )
    
    async def _search_single_perspective(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        top_k: int,
        min_similarity: float
    ) -> SearchResult:
        """单视角搜索"""
        try:
            url = f"{self.base_url}/api/v1/memory/retrieve/related-memory-items"
            payload = {
                "user_id": user_id,
                "agent_id": self.agent_id,
                "query": query,
                "top_k": top_k,
                "min_similarity": min_similarity
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
        except Exception as e:
            self.console.print(f"❌ Memu search error: {e}", style="red")
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={
                    "error": str(e),
                    "user_ids": [user_id]
                }
            )
        
        # 转换为标准格式
        search_results = []
        related_memories = result.get("related_memories", [])
        
        for item in related_memories:
            memory = item.get("memory", {})
            content = memory.get("content", "")
            score = item.get("similarity_score", 0.0)
            
            search_results.append({
                "content": content,
                "score": score,
                "user_id": user_id,  # 🔥 添加 user_id 标记记忆来源
                "metadata": {
                    "id": memory.get("memory_id", ""),
                    "category": memory.get("category", ""),
                    "created_at": memory.get("created_at", ""),
                    "happened_at": memory.get("happened_at", ""),
                }
            })
        
        # 构建定制的 context
        formatted_context = self._build_memu_context(search_results)
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=search_results,
            retrieval_metadata={
                "system": "memu",
                "user_ids": [user_id],
                "top_k": top_k,
                "min_similarity": min_similarity,
                "total_found": result.get("total_found", len(search_results)),
                "formatted_context": formatted_context,
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
        top_k: int,
        min_similarity: float
    ) -> SearchResult:
        """双视角搜索"""
        # 分别搜索两个 user 的记忆
        result_a = await self._search_single_perspective(
            query, conversation_id, speaker_a_user_id, top_k, min_similarity
        )
        result_b = await self._search_single_perspective(
            query, conversation_id, speaker_b_user_id, top_k, min_similarity
        )
        
        # 合并结果
        all_results = result_a.results + result_b.results
        
        # 按分数排序
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # 只保留 top_k 个
        all_results = all_results[:top_k]
        
        # 构建双视角的 context
        formatted_context = self._build_dual_perspective_context(
            speaker_a, speaker_b, result_a.results, result_b.results
        )
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "memu",
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "top_k": top_k,
                "min_similarity": min_similarity,
                "total_found": len(all_results),
                "formatted_context": formatted_context,
                "dual_perspective": True,
            }
        )
    
    def _build_dual_perspective_context(
        self,
        speaker_a: str,
        speaker_b: str,
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]]
    ) -> str:
        """
        构建双视角的 context，使用 default template
        
        步骤：
        1. 为每个 speaker 构建带 happened_at 的记忆列表
        2. 使用 online_api.templates.default 包装成双视角格式
        """
        # 构建 Speaker A 的记忆（带 happened_at 和 category）
        speaker_a_memories = []
        if results_a:
            for idx, result in enumerate(results_a[:5], 1):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                happened_at = metadata.get("happened_at", "")
                category = metadata.get("category", "")
                
                memory_text = f"{idx}. {content}"
                
                metadata_parts = []
                if happened_at:
                    date_str = happened_at.split("T")[0] if "T" in happened_at else happened_at
                    metadata_parts.append(f"Date: {date_str}")
                if category:
                    metadata_parts.append(f"Category: {category}")
                
                if metadata_parts:
                    memory_text += f" ({', '.join(metadata_parts)})"
                
                speaker_a_memories.append(memory_text)
        
        speaker_a_memories_text = "\n".join(speaker_a_memories) if speaker_a_memories else "(No memories found)"
        
        # 构建 Speaker B 的记忆（带 happened_at 和 category）
        speaker_b_memories = []
        if results_b:
            for idx, result in enumerate(results_b[:5], 1):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                happened_at = metadata.get("happened_at", "")
                category = metadata.get("category", "")
                
                memory_text = f"{idx}. {content}"
                
                metadata_parts = []
                if happened_at:
                    date_str = happened_at.split("T")[0] if "T" in happened_at else happened_at
                    metadata_parts.append(f"Date: {date_str}")
                if category:
                    metadata_parts.append(f"Category: {category}")
                
                if metadata_parts:
                    memory_text += f" ({', '.join(metadata_parts)})"
                
                speaker_b_memories.append(memory_text)
        
        speaker_b_memories_text = "\n".join(speaker_b_memories) if speaker_b_memories else "(No memories found)"
        
        # 使用 default template 包装
        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        return template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )
    
    def _build_memu_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        为 Memu 构建定制的 context，使用 happened_at 字段显示事件发生时间
        
        Args:
            search_results: 搜索结果列表
        
        Returns:
            格式化的 context 字符串
        """
        if not search_results:
            return ""
        
        context_parts = []
        
        for idx, result in enumerate(search_results[:10], 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})
            
            # 优先使用 happened_at（事件发生时间），如果没有则使用 created_at
            happened_at = metadata.get("happened_at", "")
            category = metadata.get("category", "")
            
            # 构建每条记忆的格式
            memory_text = f"{idx}. {content}"
            
            # 添加时间和分类信息（如果有的话）
            metadata_parts = []
            if happened_at:
                # 只显示日期部分（YYYY-MM-DD）
                date_str = happened_at.split("T")[0] if "T" in happened_at else happened_at
                metadata_parts.append(f"Date: {date_str}")
            if category:
                metadata_parts.append(f"Category: {category}")
            
            if metadata_parts:
                memory_text += f" ({', '.join(metadata_parts)})"
            
            context_parts.append(memory_text)
        
        return "\n\n".join(context_parts)
    
    def get_system_info(self) -> Dict[str, Any]:
        """返回系统信息"""
        return {
            "name": "Memu",
            "type": "online_api",
            "description": "Memu - Memory Management System (HTTP RESTful API)",
            "adapter": "MemuAdapter",
            "base_url": self.base_url,
            "agent_id": self.agent_id,
        }

