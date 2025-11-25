"""
Parallax Adapter

适配层，负责将评测框架与 Parallax 实现连接起来。
"""
import asyncio
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from eval.adapters.base import BaseAdapter
from eval.adapters.registry import register_adapter
from eval.core.data_models import Conversation, SearchResult
from eval.utils.logger import set_activity_id
from utils.datetime_utils import to_iso_format

logger = logging.getLogger(__name__)

# 导入 Parallax 实现
from eval.adapters.parallax import (
    stage1_memunits_extraction,
    stage2_index_building,
    stage3_memory_retrivel,
    stage4_response,
)

# 导入 Memory Layer 组件
from providers.llm.llm_provider import LLMProvider
from memory.memory_extractor.event_log_extractor import EventLogExtractor


@register_adapter("parallax")
class ParallaxAdapter(BaseAdapter):
    """
    Parallax 适配器

    职责：
    1. 接收评测框架的调用
    2. 转换数据格式（评测框架 ↔ Parallax）
    3. 调用 stage*.py 实现
    4. 返回评测框架需要的结果格式

    实现细节：
    - MemUnit 提取（stage1）
    - 索引构建（stage2）
    - 检索逻辑（stage3）
    - 答案生成（stage4）
    """
    
    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config)
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        
        # 初始化 LLM Provider（共享给所有 stage）
        # 从 YAML 的 llm 配置中读取
        llm_config = config.get("llm", {})
        
        self.llm_provider = LLMProvider(
            provider_type=llm_config.get("provider", "openai"),
            model=llm_config.get("model", "gpt-4o-mini"),
            api_key=llm_config.get("api_key", ""),
            base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=int(llm_config.get("max_tokens", 32768)),
        )
        
        # 初始化 Event Log Extractor（使用评估专用提示词）
        self.event_log_extractor = EventLogExtractor(
            llm_provider=self.llm_provider,
            use_eval_prompts=True  # 评估系统使用 eval/ 提示词
        )
        
        # 确保 NLTK 数据可用
        stage2_index_building.ensure_nltk_data()

        logger.info(f"Parallax Adapter initialized")
        logger.info(f"LLM Model: {llm_config.get('model')}")
        logger.info(f"Output Dir: {self.output_dir}")

        # Debug: Print environment configuration
        llm_api_key = os.getenv("LLM_API_KEY", "")
        if llm_api_key:
            logger.debug(f"🔑 LLM_API_KEY loaded: {llm_api_key[:20]}... (len={len(llm_api_key)})")
        else:
            logger.debug(f"⚠️  LLM_API_KEY not found in environment!")

        logger.debug(f"Concurrency Config - Extraction: {os.getenv('EVAL_EXTRACTION_MAX_CONCURRENT', '5')}")
        logger.debug(f"Concurrency Config - Indexing: {os.getenv('EVAL_INDEXING_MAX_CONCURRENT', '5')}")
        logger.debug(f"Concurrency Config - Retrieval: {os.getenv('EVAL_RETRIEVAL_MAX_CONCURRENT', '5')}")
        logger.debug(f"Concurrency Config - Response: {os.getenv('EVAL_RESPONSE_MAX_CONCURRENT', '5')}")
        logger.debug(f"Concurrency Config - Judgment: {os.getenv('EVAL_JUDGMENT_MAX_CONCURRENT', '5')}")
    
    @staticmethod
    def _extract_conv_index(conversation_id: str) -> str:
        """
        从 conversation_id 中提取数字索引部分
        
        例如：
        - "locomo_0" -> "0"
        - "personamem_42" -> "42"
        - "123" -> "123"
        - "test_abc_5" -> "5"
        
        策略：取最后一个下划线后的部分，如果没有下划线则返回原值
        """
        if "_" in conversation_id:
            return conversation_id.split("_")[-1]
        return conversation_id
    
    def _check_missing_indexes(
        self,
        index_dir: Path,
        num_conv: int,
        index_type: str = "bm25"
    ) -> List[int]:
        """
        检查缺失的索引文件
        
        Args:
            index_dir: 索引目录
            num_conv: 会话总数
            index_type: 索引类型（"bm25" 或 "embedding"）
        
        Returns:
            缺失索引的会话索引列表
        """
        missing_indexes = []
        
        for i in range(num_conv):
            if index_type == "bm25":
                index_file = index_dir / f"bm25_index_conv_{i}.pkl"
            else:  # embedding
                index_file = index_dir / f"embedding_index_conv_{i}.pkl"
            
            if not index_file.exists():
                missing_indexes.append(i)
        
        return missing_indexes
    
    async def add(
        self, 
        conversations: List[Conversation],
        output_dir: Path = None,
        checkpoint_manager = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Add 阶段：提取 MemUnits 并构建索引
        
        调用流程：
        1. Stage 1: 提取 MemUnits (stage1_memunits_extraction.py) - 并发处理
        2. Stage 2: 构建 BM25 和 Embedding 索引 (stage2_index_building.py)
        
        返回：索引元数据（方案 A：延迟加载）
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        memunits_dir = output_dir / "memunits"
        memunits_dir.mkdir(parents=True, exist_ok=True)
        bm25_index_dir = output_dir / "bm25_index"
        emb_index_dir = output_dir / "vectors"
        bm25_index_dir.mkdir(parents=True, exist_ok=True)
        emb_index_dir.mkdir(parents=True, exist_ok=True)
        
        console = Console()
        
        # ========== Stage 1: MemUnit Extraction (并发处理) ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 1: MemUnit Extraction", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")

        # 转换数据格式：评测框架 → Parallax
        raw_data_dict = {}
        for conv in conversations:
            conv_id = conv.conversation_id
            raw_data = []
            
            for idx, msg in enumerate(conv.messages):
                # 处理时间戳：如果为 None，使用基于索引的伪时间戳
                if msg.timestamp is not None:
                    timestamp_str = to_iso_format(msg.timestamp)
                else:
                    # 使用消息索引生成伪时间戳（保持相对顺序）
                    # 基准时间: 2023-01-01 00:00:00，每条消息间隔 30 秒
                    from datetime import datetime, timedelta
                    base_time = datetime(2023, 1, 1, 0, 0, 0)
                    pseudo_time = base_time + timedelta(seconds=idx * 30)
                    timestamp_str = to_iso_format(pseudo_time)
                
                message_dict = {
                    "speaker_id": msg.speaker_id,
                    "user_name": msg.speaker_name or msg.speaker_id,
                    "speaker_name": msg.speaker_name or msg.speaker_id,
                    "content": msg.content,
                    "timestamp": timestamp_str,
                }
                
                # 添加可选字段
                for optional_field in ["img_url", "blip_caption", "query"]:
                    if optional_field in msg.metadata and msg.metadata[optional_field] is not None:
                        message_dict[optional_field] = msg.metadata[optional_field]
                
                raw_data.append(message_dict)
            
            raw_data_dict[conv_id] = raw_data
        
        # 检查已完成的会话（断点续传）
        # 🔥 使用提取后的索引来检查文件（stage1 保存时用的是提取后的索引）
        completed_convs = set()
        if checkpoint_manager:
            all_conv_indices = [self._extract_conv_index(conv.conversation_id) for conv in conversations]
            completed_indices = checkpoint_manager.load_add_progress(memunits_dir, all_conv_indices)
            # 将完成的索引映射回原始 conversation_id
            for conv in conversations:
                if self._extract_conv_index(conv.conversation_id) in completed_indices:
                    completed_convs.add(conv.conversation_id)
        
        # 过滤出待处理的会话
        pending_conversations = [
            conv for conv in conversations
            if conv.conversation_id not in completed_convs
        ]
        
        console.print(f"\n📊 总会话数: {len(conversations)}", style="bold cyan")
        console.print(f"✅ 已完成: {len(completed_convs)}", style="bold green")
        console.print(f"⏳ 待处理: {len(pending_conversations)}", style="bold yellow")
        
        if len(pending_conversations) == 0:
            console.print(f"\n🎉 所有会话已完成，跳过 MemUnit 提取！", style="bold green")
        else:
            total_messages = sum(len(raw_data_dict[c.conversation_id]) for c in pending_conversations)
            console.print(f"📝 待处理消息数: {total_messages}", style="bold blue")
            console.print(f"🚀 开始并发处理...\n", style="bold green")
            
            # 使用 Rich 进度条并发处理
            start_time = time.time()
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("•"),
                TextColumn("[bold blue]{task.fields[status]}"),
                console=console,
                transient=False,
                refresh_per_second=1,
            ) as progress:
                # 创建主进度任务
                main_task = progress.add_task(
                    "[bold cyan]🎯 总进度",
                    total=len(conversations),
                    completed=len(completed_convs),
                    status="处理中",
                )
                
                # 为已完成的会话创建进度条（显示为完成）
                conversation_tasks = {}
                for conv_id in completed_convs:
                    conv_index = self._extract_conv_index(conv_id)
                    conv_task_id = progress.add_task(
                        f"[green]Conv-{conv_index}",
                        total=len(raw_data_dict.get(conv_id, [])),
                        completed=len(raw_data_dict.get(conv_id, [])),
                        status="✅ (已跳过)",
                    )
                    conversation_tasks[conv_id] = conv_task_id
                
                # 为待处理的会话创建进度条和任务
                processing_tasks = []
                for conv in pending_conversations:
                    conv_id = conv.conversation_id
                    conv_index = self._extract_conv_index(conv_id)  # 🔥 提取数字索引
                    conv_task_id = progress.add_task(
                        f"[yellow]Conv-{conv_index}",
                        total=len(raw_data_dict[conv_id]),
                        completed=0,
                        status="等待",
                    )
                    conversation_tasks[conv_id] = conv_task_id
                    
                    # 🔥 创建处理任务，传入提取后的索引
                    task = stage1_memunits_extraction.process_single_conversation(
                        conv_id=conv_index,  # 使用提取后的索引
                        conversation=raw_data_dict[conv_id],  # 数据用原始 ID
                        save_dir=str(memunits_dir),
                        llm_provider=self.llm_provider,
                        event_log_extractor=self.event_log_extractor,
                        progress_counter=None,
                        progress=progress,
                        task_id=conv_task_id,
                        config=self._convert_config_to_experiment_config(),
                    )
                    processing_tasks.append((conv_id, task))
                
                # 定义完成时更新函数
                async def run_with_completion(conv_id, task):
                    result = await task
                    progress.update(
                        conversation_tasks[conv_id],
                        status="✅",
                        completed=progress.tasks[conversation_tasks[conv_id]].total,
                    )
                    progress.update(main_task, advance=1)
                    return result
                
                # 🔥 并发执行所有待处理的任务
                if processing_tasks:
                    results = await asyncio.gather(
                        *[run_with_completion(conv_id, task) for conv_id, task in processing_tasks]
                    )
                else:
                    results = []
                
                progress.update(main_task, status="✅ 完成")
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            # 统计结果
            successful_convs = sum(1 for _, memunit_list in results if memunit_list)
            total_memunits = sum(len(memunit_list) for _, memunit_list in results)
            
            console.print("\n" + "=" * 60, style="dim")
            console.print("📊 MemUnit 提取完成统计:", style="bold")
            console.print(f"   ✅ 成功处理: {successful_convs}/{len(pending_conversations)}", style="green")
            console.print(f"   📝 总 memunits: {total_memunits}", style="blue")
            console.print(f"   ⏱️  总耗时: {elapsed:.2f} 秒", style="yellow")
            if len(pending_conversations) > 0:
                console.print(f"   🚀 平均每会话: {elapsed/len(pending_conversations):.2f} 秒", style="cyan")
            console.print("=" * 60, style="dim")
        
        # ========== Stage 2: Index Building ==========
        console.print(f"\n{'='*60}", style="bold cyan")
        console.print(f"Stage 2: Index Building", style="bold cyan")
        console.print(f"{'='*60}", style="bold cyan")
        
        # 调用 stage2 实现构建索引
        exp_config = self._convert_config_to_experiment_config()
        exp_config.num_conv = len(conversations)  # 设置会话数量
        
        # 🔥 智能跳过逻辑：检查已存在的索引文件
        bm25_need_build = self._check_missing_indexes(
            index_dir=bm25_index_dir,
            num_conv=len(conversations),
            index_type="bm25"
        )
        
        emb_need_build = []
        use_hybrid = self.config.get("search", {}).get("use_hybrid_search", True)
        if use_hybrid:
            emb_need_build = self._check_missing_indexes(
                index_dir=emb_index_dir,
                num_conv=len(conversations),
                index_type="embedding"
            )
        
        # 统计信息
        total_convs = len(conversations)
        bm25_to_build = len(bm25_need_build)
        emb_to_build = len(emb_need_build) if use_hybrid else 0
        
        console.print(f"\n📊 索引构建统计:")
        console.print(f"   总会话数: {total_convs}")
        console.print(f"   BM25 索引: 需要构建 {bm25_to_build}, 已存在 {total_convs - bm25_to_build}")
        if use_hybrid:
            console.print(f"   Embedding 索引: 需要构建 {emb_to_build}, 已存在 {total_convs - emb_to_build}")
        
        # 构建 BM25 索引
        if bm25_to_build > 0:
            # 设置 activity_id: 索引构建阶段
            set_activity_id("add-idx-bm25")
            console.print(f"\n🔨 构建 BM25 索引 ({bm25_to_build} 个会话)...", style="yellow")
            stage2_index_building.build_bm25_index(
                config=exp_config,
                data_dir=memunits_dir,
                bm25_save_dir=bm25_index_dir,
            )
            console.print("✅ BM25 索引构建完成", style="green")
        else:
            console.print("✅ BM25 索引已全部存在，跳过构建", style="green")

        # 构建 Embedding 索引（如果启用）
        if use_hybrid:
            if emb_to_build > 0:
                # 设置 activity_id: Embedding 索引构建阶段
                set_activity_id("add-idx-emb")
                console.print(f"\n🔨 构建 Embedding 索引 ({emb_to_build} 个会话)...", style="yellow")
                await stage2_index_building.build_emb_index(
                    config=exp_config,
                    data_dir=memunits_dir,
                    emb_save_dir=emb_index_dir,
                )
                console.print("✅ Embedding 索引构建完成", style="green")
            else:
                console.print("✅ Embedding 索引已全部存在，跳过构建", style="green")
        
        # ========== 方案 A：返回索引元数据（延迟加载） ==========
        # 不加载索引到内存，只返回路径和元数据
        index_metadata = {
            "type": "lazy_load",  # 标记为延迟加载
            "memunits_dir": str(memunits_dir),
            "bm25_index_dir": str(bm25_index_dir),
            "emb_index_dir": str(emb_index_dir),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": use_hybrid,
            "total_conversations": len(conversations),
        }
        
        console.print(f"\n{'='*60}", style="dim")
        console.print(f"✅ Add 阶段完成", style="bold green")
        console.print(f"   📁 MemUnits: {memunits_dir}", style="dim")
        console.print(f"   📁 BM25 索引: {bm25_index_dir}", style="dim")
        if use_hybrid:
            console.print(f"   📁 Embedding 索引: {emb_index_dir}", style="dim")
        console.print(f"   💡 使用延迟加载策略（内存友好）", style="cyan")
        console.print(f"{'='*60}\n", style="dim")
        
        return index_metadata
    
    async def search(self, query: str, conversation_id: str, index: Any, **kwargs) -> SearchResult:
        """
        Search 阶段：检索相关 MemUnits
        
        延迟加载：按需从文件加载索引（内存友好）
        """
        # 延迟加载 - 从文件读取索引
        bm25_index_dir = Path(index["bm25_index_dir"])
        emb_index_dir = Path(index["emb_index_dir"])
        
        # 🔥 修复：从 conversation_id 提取数字索引来查找索引文件
        # 例如：conversation_id = "locomo_0" -> conv_index = "0"
        conv_index = self._extract_conv_index(conversation_id)
        
        # 按需加载 BM25 索引（使用数字索引）
        bm25_file = bm25_index_dir / f"bm25_index_conv_{conv_index}.pkl"
        if not bm25_file.exists():
            return SearchResult(
                query=query,
                conversation_id=conversation_id,
                results=[],
                retrieval_metadata={"error": f"BM25 index not found: {bm25_file.name}"}
            )
        
        with open(bm25_file, "rb") as f:
            bm25_index_data = pickle.load(f)
        
        bm25 = bm25_index_data.get("bm25")
        docs = bm25_index_data.get("docs")
        
        # 按需加载 Embedding 索引（使用数字索引）
        emb_index = None
        if index.get("use_hybrid_search"):
            emb_file = emb_index_dir / f"embedding_index_conv_{conv_index}.pkl"
            if emb_file.exists():
                with open(emb_file, "rb") as f:
                    emb_index = pickle.load(f)
        
        # 调用 stage3 检索实现
        search_config = self.config.get("search", {})
        retrieval_mode = search_config.get("mode", "agentic")
        
        exp_config = self._convert_config_to_experiment_config()
        # 从 exp_config 获取正确格式的 llm_config
        llm_config = exp_config.llm_config.get(exp_config.llm_service, {})
        
        if retrieval_mode == "agentic":
            # Agentic 检索
            top_results, metadata = await stage3_memory_retrivel.agentic_retrieval(
                query=query,
                config=exp_config,
                llm_provider=self.llm_provider,
                llm_config=llm_config,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
            )
        elif retrieval_mode == "lightweight":
            # 轻量级检索
            top_results, metadata = await stage3_memory_retrivel.lightweight_retrieval(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                config=exp_config,
            )
        else:
            # 默认使用混合检索
            top_results = await stage3_memory_retrivel.hybrid_search_with_rrf(
                query=query,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=20,
                emb_candidates=search_config.get("hybrid_emb_candidates", 100),
                bm25_candidates=search_config.get("hybrid_bm25_candidates", 100),
                rrf_k=search_config.get("hybrid_rrf_k", 60),
            )
            metadata = {}
        
        # 转换为评测框架需要的格式
        results = []
        for doc, score in top_results:
            results.append({
                "content": doc.get("episode", ""),
                "score": float(score),
                "metadata": {
                    "subject": doc.get("subject", ""),
                    "summary": doc.get("summary", ""),
                }
            })
        
        # 🔥 构建 formatted_context
        formatted_context = ""
        conversation = kwargs.get("conversation")
        if conversation and top_results:
            # 获取 speaker 信息
            speaker_a = conversation.metadata.get("speaker_a", "Speaker A")
            speaker_b = conversation.metadata.get("speaker_b", "Speaker B")
            
            # 🔥 使用 config.response_top_k 而不是硬编码的 10
            response_top_k = exp_config.response_top_k
            
            # 构建 context
            retrieved_docs_text = []
            for doc, score in top_results[:response_top_k]:  # 使用 config 中的 response_top_k
                subject = doc.get('subject', 'N/A')
                episode = doc.get('episode', 'N/A')
                doc_text = f"{subject}: {episode}\n---"
                retrieved_docs_text.append(doc_text)
            
            speaker_memories = "\n\n".join(retrieved_docs_text)
            
            TEMPLATE = """Episodes memories for conversation between {speaker_1} and {speaker_2}:

    {speaker_memories}
"""
            formatted_context = TEMPLATE.format(
                speaker_1=speaker_a,
                speaker_2=speaker_b,
                speaker_memories=speaker_memories,
            )
        
        # 添加 formatted_context 到 metadata
        metadata["formatted_context"] = formatted_context
        
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata=metadata
        )
    
    async def answer(self, query: str, context: str, **kwargs) -> str:
        """
        Answer 阶段：生成答案
        
        调用 stage4_response.py 的实现
        """
        # 调用 stage4 答案生成实现
        exp_config = self._convert_config_to_experiment_config()
        
        answer = await stage4_response.locomo_response(
            llm_provider=self.llm_provider,
            context=context,
            question=query,
            experiment_config=exp_config,
        )
        
        return answer
    
    def get_system_info(self) -> Dict[str, Any]:
        """返回系统信息"""
        return {
            "name": "Parallax",
            "version": "1.0",
            "description": "Parallax memory system with agentic retrieval",
            "adapter": "Adapter connecting framework to Parallax implementation",
        }
    
    def _convert_config_to_experiment_config(self):
        """
        将评测框架的 config 转换为 ExperimentConfig 格式
        """
        from eval.adapters.parallax.config import ExperimentConfig
        import os
        
        exp_config = ExperimentConfig()
        
        # 映射 LLM 配置：将 YAML 的 llm 转换为 ExperimentConfig 的 llm_config 格式
        llm_cfg = self.config.get("llm", {})
        provider = llm_cfg.get("provider", "openai")
        
        exp_config.llm_service = provider
        exp_config.llm_config = {
            provider: {
                "llm_provider": provider,
                "model": llm_cfg.get("model", "gpt-4o-mini"),
                "api_key": llm_cfg.get("api_key") or os.getenv("LLM_API_KEY", ""),
                "base_url": llm_cfg.get("base_url") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
                "temperature": llm_cfg.get("temperature", 0.3),
                "max_tokens": int(llm_cfg.get("max_tokens", 32768)),
            }
        }
        
        # 映射 Add 阶段配置（只覆盖 YAML 中显式指定的）
        add_config = self.config.get("add", {})
        if "enable_semantic_extraction" in add_config:
            exp_config.enable_semantic_extraction = add_config["enable_semantic_extraction"]
        if "enable_clustering" in add_config:
            exp_config.enable_clustering = add_config["enable_clustering"]
        if "enable_profile_extraction" in add_config:
            exp_config.enable_profile_extraction = add_config["enable_profile_extraction"]
        
        # 映射 Search 阶段配置（只覆盖 YAML 中显式指定的）
        search_config = self.config.get("search", {})
        if "mode" in search_config:
            exp_config.retrieval_mode = search_config["mode"]
            exp_config.use_agentic_retrieval = (exp_config.retrieval_mode == "agentic")
        
        return exp_config
    
    def build_lazy_index(self, conversations: List[Conversation], output_dir: Any) -> Dict[str, Any]:
        """
        构建 Parallax 的延迟加载索引元数据

        🔥 Parallax 特点：
        - 本地索引（memunits, bm25, embeddings）
        - 延迟加载（只保存元数据，不加载实际索引文件）
        
        Args:
            conversations: 对话列表
            output_dir: 输出目录
            
        Returns:
            索引元数据字典
        """
        return {
            "type": "lazy_load",
            "memunits_dir": str(output_dir / "memunits"),
            "bm25_index_dir": str(output_dir / "bm25_index"),
            "emb_index_dir": str(output_dir / "vectors"),
            "conversation_ids": [conv.conversation_id for conv in conversations],
            "use_hybrid_search": True,
            "total_conversations": len(conversations),
        }
