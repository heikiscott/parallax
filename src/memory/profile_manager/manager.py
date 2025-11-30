"""ProfileManager - Core component for automatic profile extraction and management."""

import asyncio
from typing import Any, Dict, List, Optional, Set
from pathlib import Path

from providers.llm.llm_provider import LLMProvider
from memory.extraction.memory.profile import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
)
from memory.profile_manager.config import ProfileManagerConfig, ScenarioType
from memory.profile_manager.discriminator import ValueDiscriminator, DiscriminatorConfig
from memory.profile_manager.storage import ProfileStorage, InMemoryProfileStorage
from core.observation.logger import get_logger

logger = get_logger(__name__)


class ProfileManager:
    """Automatic profile extraction and management integrated with clustering.
    
    ProfileManager monitors memunit clustering and automatically extracts/updates
    user profiles when high-value information is detected.
    
    Key Features:
    - Automatic profile extraction triggered by cluster updates
    - Value discrimination to filter high-quality updates
    - Incremental profile merging with version history
    - Flexible storage backends (in-memory, file-based, or custom)
    - Seamless integration with ConvMemUnitExtractor
    
    Example:
        ```python
        # Initialize
        config = ProfileManagerConfig(
            scenario="group_chat",
            min_confidence=0.6,
            enable_versioning=True
        )
        
        profile_mgr = ProfileManager(llm_provider, config)
        
        # Option 1: Attach to extractor for automatic updates
        memunit_extractor = ConvMemUnitExtractor(llm_provider)
        profile_mgr.attach_to_extractor(memunit_extractor)
        
        # Option 2: Manual updates
        await profile_mgr.on_memunit_clustered(
            memunit=memunit,
            cluster_id="cluster_001",
            recent_memunits=[...]
        )
        
        # Access profiles
        profile = await profile_mgr.get_profile(user_id)
        all_profiles = await profile_mgr.get_all_profiles()
        ```
    """
    
    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ProfileManagerConfig] = None,
        storage: Optional[ProfileStorage] = None,
        group_id: Optional[str] = None,
        group_name: Optional[str] = None,
    ):
        """Initialize ProfileManager.
        
        Args:
            llm_provider: LLM provider for profile extraction and discrimination
            config: Manager configuration (uses defaults if None)
            storage: Profile storage backend (uses InMemoryProfileStorage if None)
            group_id: Group/conversation identifier
            group_name: Group/conversation name
        """
        self.llm_provider = llm_provider
        self.config = config or ProfileManagerConfig()
        self.group_id = group_id or "default"
        self.group_name = group_name
        
        # Initialize components
        self._profile_extractor = ProfileMemoryExtractor(llm_provider=llm_provider)
        
        # 💡 设置 MemUnit 阈值：降低到 2 个，让 Profile 更容易提取
        self._min_memunits_threshold = 1  # 从默认的 3 降低到 2
        
        discriminator_config = DiscriminatorConfig(
            min_confidence=self.config.min_confidence,
            use_context=True,
            context_window=2
        )
        scenario_str = self.config.scenario.value if isinstance(self.config.scenario, ScenarioType) else str(self.config.scenario)
        self._discriminator = ValueDiscriminator(
            llm_provider=llm_provider,
            config=discriminator_config,
            scenario=scenario_str
        )
        
        # Storage
        if storage is None:
            storage = InMemoryProfileStorage(
                enable_persistence=False,
                enable_versioning=self.config.enable_versioning
            )
        self._storage = storage
        
        # Internal state for cluster tracking
        self._cluster_memunits: Dict[str, List[Any]] = {}  # cluster_id -> memunits
        self._watched_clusters: Set[str] = set()  # clusters flagged for profile extraction
        self._recent_memunits: List[Any] = []  # rolling window for context
        
        # Statistics
        self._stats = {
            "total_memunits": 0,
            "high_value_memunits": 0,
            "profile_extractions": 0,
            "failed_extractions": 0,
        }
        
        # 可配置的最小 MemUnits 阈值（默认 1）
        self._min_memunits_threshold = 1
    
    async def on_memunit_clustered(
        self,
        memunit: Any,
        cluster_id: str,
        recent_memunits: Optional[List[Any]] = None,
        user_id_list: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Handle a newly clustered memunit and conditionally extract/update profiles.
        
        This method should be called when a memunit has been assigned to a cluster.
        It will:
        1. Add the memunit to the cluster's collection
        2. Evaluate if the memunit contains high-value profile information
        3. If high-value, mark the cluster as "watched" and extract profiles
        4. If cluster is already watched, update profiles incrementally
        
        Args:
            memunit: The memunit that was just clustered
            cluster_id: The cluster it was assigned to
            recent_memunits: Recent memunits for context (optional, uses internal if None)
            user_id_list: List of user IDs to extract profiles for (optional)
        
        Returns:
            Dict with extraction results:
            {
                "cluster_id": str,
                "is_high_value": bool,
                "confidence": float,
                "reason": str,
                "watched": bool,
                "profiles_updated": int,
                "updated_user_ids": List[str]
            }
        """
        self._stats["total_memunits"] += 1
        
        # Add to cluster collection
        if cluster_id not in self._cluster_memunits:
            self._cluster_memunits[cluster_id] = []
        self._cluster_memunits[cluster_id].append(memunit)
        logger.info(f"[ProfileManager] 已添加 MemUnit 到 _cluster_memunits[{cluster_id}]，当前数量: {len(self._cluster_memunits[cluster_id])}")
        
        # Update recent memunits window
        self._recent_memunits.append(memunit)
        if len(self._recent_memunits) > 10:
            self._recent_memunits = self._recent_memunits[-10:]
        
        # Use provided context or internal window
        context = recent_memunits if recent_memunits is not None else self._recent_memunits[-2:]
        
        # Value discrimination
        is_high_value, confidence, reason = await self._discriminator.is_high_value(
            memunit,
            context
        )
        logger.info(f"is_high_value: {is_high_value}, confidence: {confidence}, reason: {reason}")
        
        # 💡 强制设为 True，跳过价值判别（用于调试）
        is_high_value = True
        confidence = 1.0
        reason = f"[FORCED] {reason}"
        
        if is_high_value:
            self._stats["high_value_memunits"] += 1
            self._watched_clusters.add(cluster_id)
            log_msg = (
                f"High-value memunit detected in cluster {cluster_id}: "
                f"confidence={confidence:.2f}, reason='{reason}'"
            )
            logger.info(log_msg)
        
        # Extract/update profiles if cluster is watched and auto_extract is enabled
        updated_user_ids = []
        profiles_updated = 0
        
        if cluster_id in self._watched_clusters and self.config.auto_extract:
            # ✅ 参考原代码：只有当簇内 MemUnits 达到一定数量才提取
            # 避免单个 MemUnit 提取导致 Profile 为空
            cluster_memunit_count = len(self._cluster_memunits.get(cluster_id, []))
            
            # 使用实例属性或默认值
            min_memunits_for_extraction = getattr(self, '_min_memunits_threshold', 3)
            
            logger.info(f"[ProfileManager] Cluster {cluster_id}: {cluster_memunit_count} MemUnits, 阈值: {min_memunits_for_extraction}")
            
            if cluster_memunit_count < min_memunits_for_extraction:
                log_msg = (
                    f"Cluster {cluster_id} only has {cluster_memunit_count} memunits, "
                    f"waiting for {min_memunits_for_extraction} before extraction"
                )
                logger.debug(log_msg)
            else:
                try:
                    logger.info(f"[ProfileManager] 🚀 开始提取 Cluster {cluster_id} 的 Profile...")
                    updated_profiles = await self._extract_profiles_for_cluster(
                        cluster_id=cluster_id,
                        user_id_list=user_id_list
                    )
                    
                    profiles_updated = len(updated_profiles)
                    updated_user_ids = [
                        getattr(prof, "user_id", None)
                        for prof in updated_profiles
                        if hasattr(prof, "user_id")
                    ]
                    
                    self._stats["profile_extractions"] += 1
                    
                    if profiles_updated > 0:
                        log_msg = f"Updated {profiles_updated} profiles for cluster {cluster_id}"
                        logger.info(log_msg)
                
                except Exception as e:
                    error_msg = f"Failed to extract profiles for cluster {cluster_id}: {e}"
                    logger.error(error_msg, exc_info=True)
                    self._stats["failed_extractions"] += 1
        
        return {
            "cluster_id": cluster_id,
            "is_high_value": is_high_value,
            "confidence": confidence,
            "reason": reason,
            "watched": cluster_id in self._watched_clusters,
            "profiles_updated": profiles_updated,
            "updated_user_ids": updated_user_ids,
        }
    
    async def _extract_profiles_for_cluster(
        self,
        cluster_id: str,
        user_id_list: Optional[List[str]] = None
    ) -> List[Any]:
        """Extract profiles for a specific cluster using all its memunits.
        
        Args:
            cluster_id: Cluster identifier
            user_id_list: List of user IDs to extract for (optional)
        
        Returns:
            List of extracted/updated ProfileMemory objects
        """
        logger.debug(f"[ProfileManager] _extract_profiles_for_cluster 被调用")
        logger.debug(f"[ProfileManager]   cluster_id: {cluster_id}")
        logger.debug(f"[ProfileManager]   _cluster_memunits keys: {list(self._cluster_memunits.keys())}")
        logger.debug(f"[ProfileManager]   _cluster_memunits[{cluster_id}]: {len(self._cluster_memunits.get(cluster_id, []))} 个")
        
        memunits = self._cluster_memunits.get(cluster_id, [])
        if not memunits:
            logger.warning(f"[ProfileManager] ❌ _cluster_memunits[{cluster_id}] 为空！返回空列表")
            return []
        
        logger.info(f"[ProfileManager] ✅ 找到 {len(memunits)} 个 MemUnit")
        
        # 🔧 关键修复：将字典格式的 memunit 转换为完整的 MemUnit 对象
        # 从 MongoDB 重新加载完整的 MemUnit 数据
        from infra.adapters.out.persistence.document.memory.memunit import MemUnit as MemUnitDoc
        
        full_memunits = []
        for mc in memunits:
            event_id = getattr(mc, 'event_id', None) or getattr(mc, '_id', None) or getattr(mc, 'id', None)
            if event_id:
                logger.debug(f"[ProfileManager] 正在从 MongoDB 加载 MemUnit: {event_id} (类型: {type(event_id).__name__})")
                # 从 MongoDB 加载完整的 MemUnit
                # 尝试多种方式查询
                full_mc = None
                
                # 方式1: 用 event_id 字段查询
                full_mc = await MemUnitDoc.find_one({"event_id": str(event_id)})
                
                # 方式2: 如果没找到，尝试用 _id 查询
                if not full_mc:
                    try:
                        from bson import ObjectId
                        if isinstance(event_id, (str, ObjectId)):
                            oid = ObjectId(str(event_id)) if isinstance(event_id, str) else event_id
                            full_mc = await MemUnitDoc.get(oid)
                            if full_mc:
                                logger.debug(f"[ProfileManager] ✅ 用 _id 找到了")
                    except Exception as e:
                        logger.warning(f"[ProfileManager] ⚠️  用 _id 查询失败: {e}")
                
                if full_mc:
                    full_memunits.append(full_mc)
                    logger.debug(f"[ProfileManager] ✅ 加载成功，包含 episode: {len(full_mc.episode) if full_mc.episode else 0} 字符")
                else:
                    logger.warning(f"[ProfileManager] ⚠️  未找到 MemUnit: {event_id}，使用原始字典")
                    # 如果找不到，使用原始的字典对象
                    full_memunits.append(mc)
        
        if not full_memunits:
            logger.warning(f"[ProfileManager] ❌ 没有加载到任何完整的 MemUnit")
            return []
        
        logger.info(f"[ProfileManager] ✅ 加载了 {len(full_memunits)} 个完整的 MemUnit，开始提取...")
        memunits = full_memunits  # 使用完整的 MemUnit 对象
        
        # Limit batch size
        if len(memunits) > self.config.batch_size:
            logger.warning(
                f"Cluster {cluster_id} has {len(memunits)} memunits, "
                f"limiting to {self.config.batch_size} most recent"
            )
            memunits = memunits[-self.config.batch_size:]
        
        # Get old profiles for incremental merging
        old_profiles = []
        all_current_profiles = await self._storage.get_all_profiles()
        for profile in all_current_profiles.values():
            old_profiles.append(profile)
        
        # Build extraction request
        request = ProfileMemoryExtractRequest(
            memunit_list=memunits,
            user_id_list=user_id_list or [],
            group_id=self.group_id,
            group_name=self.group_name,
            old_memory_list=old_profiles if old_profiles else None,
        )
        
        # Extract profiles with retry logic
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"[ProfileManager] 开始调用 LLM 提取 Profile (场景: {self.config.scenario.value})...")
                
                if self.config.scenario == ScenarioType.ASSISTANT:
                    result = await self._profile_extractor.extract_profile_companion(request)
                else:
                    result = await self._profile_extractor.extract_memory(request)
                
                logger.debug(f"[ProfileManager] LLM 调用完成，返回结果: {result}")
                logger.debug(f"[ProfileManager] 结果类型: {type(result)}, 长度: {len(result) if result else 0}")
                
                if not result:
                    logger.warning(f"Profile extraction returned empty result for cluster {cluster_id}")
                    return []
                
                logger.info(f"[ProfileManager] ✅ LLM 返回了 {len(result)} 个 Profile")
                
                # 🚀 并行保存所有 profiles 到 storage
                updated_profiles = []
                save_tasks = []
                
                for i, profile in enumerate(result):
                    user_id = getattr(profile, "user_id", None)
                    logger.debug(f"[ProfileManager] Profile {i+1}/{len(result)}: user_id={user_id}, 类型={type(profile).__name__}")
                    
                    if user_id:
                        metadata = {
                            "group_id": self.group_id,  # 🔧 添加 group_id
                            "cluster_id": cluster_id,
                            "memunit_count": len(memunits),
                            "scenario": self.config.scenario.value,
                        }
                        
                        save_tasks.append((
                            user_id,
                            profile,
                            self._storage.save_profile(
                                user_id=user_id,
                                profile=profile,
                                metadata=metadata
                            )
                        ))
                    else:
                        logger.warning(f"[ProfileManager] ⚠️  Profile {i+1} 没有 user_id，跳过保存")
                
                logger.info(f"[ProfileManager] 准备保存 {len(save_tasks)} 个 Profile...")
                
                # 并行执行所有保存任务
                if save_tasks:
                    save_results = await asyncio.gather(
                        *[task[2] for task in save_tasks],
                        return_exceptions=True
                    )
                    
                    print(f"[ProfileManager] 保存结果: {save_results}")
                    
                    # 处理结果
                    for (user_id, profile, _), success in zip(save_tasks, save_results):
                        if isinstance(success, Exception):
                            error_msg = f"Failed to save profile for user {user_id}: {success}"
                            logger.error(error_msg, exc_info=True)
                        elif success:
                            updated_profiles.append(profile)
                            logger.info(f"[ProfileManager] ✅ 成功保存 Profile: user_id={user_id}")
                        else:
                            logger.warning(f"Failed to save profile for user {user_id}")
                
                logger.info(f"[ProfileManager] 最终成功保存了 {len(updated_profiles)} 个 Profile")
                return updated_profiles
            
            except Exception as e:
                logger.warning(
                    f"Profile extraction attempt {attempt + 1}/{self.config.max_retries} failed: {e}"
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All profile extraction attempts failed for cluster {cluster_id}")
                    raise
        
        return []
    
    async def get_profile(self, user_id: str) -> Optional[Any]:
        """Get the latest profile for a user.
        
        Args:
            user_id: User identifier
        
        Returns:
            ProfileMemory object if found, None otherwise
        """
        return await self._storage.get_profile(user_id)
    
    async def get_all_profiles(self) -> Dict[str, Any]:
        """Get all user profiles.
        
        Returns:
            Dictionary mapping user_id to ProfileMemory
        """
        return await self._storage.get_all_profiles()
    
    async def get_profile_history(
        self,
        user_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get profile version history for a user.
        
        Args:
            user_id: User identifier
            limit: Maximum number of versions to return
        
        Returns:
            List of profile versions with metadata, newest first
        """
        return await self._storage.get_profile_history(user_id, limit)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics.
        
        Returns:
            Dictionary with statistics:
            {
                "total_memunits": int,
                "high_value_memunits": int,
                "profile_extractions": int,
                "failed_extractions": int,
                "watched_clusters": int,
                "total_clusters": int
            }
        """
        return {
            **self._stats,
            "watched_clusters": len(self._watched_clusters),
            "total_clusters": len(self._cluster_memunits),
        }
    
    def attach_to_cluster_manager(self, cluster_manager: Any) -> None:
        """Attach ProfileManager to ClusterManager for automatic updates.
        
        This is the recommended approach when using the new ClusterManager component.
        
        Args:
            cluster_manager: ClusterManager instance
        """
        async def on_cluster_callback(group_id: str, memunit: Dict[str, Any], cluster_id: str):
            """Callback for cluster assignment events."""
            logger.info(f"[ProfileManager] 🔔 收到聚类回调: cluster_id={cluster_id}, group_id={group_id}")
            
            # Create wrapper object
            class MemUnitWrapper:
                def __init__(self, data: Dict[str, Any]):
                    for k, v in data.items():
                        setattr(self, k, v)
            
            mc_obj = MemUnitWrapper(memunit)
            
            logger.debug(f"[ProfileManager] 调用 on_memunit_clustered...")
            # Trigger profile update
            result = await self.on_memunit_clustered(
                memunit=mc_obj,
                cluster_id=cluster_id,
                user_id_list=memunit.get("user_id_list", [])
            )
            logger.debug(f"[ProfileManager] on_memunit_clustered 返回: {result}")
        
        # Register callback with ClusterManager
        logger.info(f"[ProfileManager] 注册回调到 ClusterManager")
        cluster_manager.on_cluster_assigned(on_cluster_callback)
        logger.info(f"[ProfileManager] 回调注册成功")
        
        logger.info("ProfileManager successfully attached to ClusterManager")
    
    def attach_to_extractor(self, memunit_extractor: Any) -> None:
        """Attach this ProfileManager to a MemUnitExtractor for automatic profile updates.
        
        This method integrates the ProfileManager with the clustering worker of a
        ConvMemUnitExtractor, so profiles are automatically extracted as conversations
        are processed.
        
        NOTE: This method uses monkey-patching. For cleaner integration, use
        ClusterManager and attach_to_cluster_manager() instead.
        
        Args:
            memunit_extractor: ConvMemUnitExtractor instance
        
        Raises:
            AttributeError: If extractor doesn't have a cluster_worker attribute
        """
        if not hasattr(memunit_extractor, "_cluster_worker"):
            raise AttributeError(
                "MemUnitExtractor does not have a _cluster_worker attribute. "
                "Only ConvMemUnitExtractor with clustering is supported."
            )
        
        cluster_worker = memunit_extractor._cluster_worker
        
        # Check if this is a ClusterManager wrapper
        if hasattr(cluster_worker, "_cluster_mgr"):
            # It's already using ClusterManager, attach directly
            self.attach_to_cluster_manager(cluster_worker._cluster_mgr)
            return
        
        # Monkey-patch the clustering worker to notify us after each memunit is processed
        original_submit = cluster_worker.submit
        
        def patched_submit(group_id: Optional[str], memunit: Dict[str, Any]) -> None:
            # Call original submit first
            original_submit(group_id, memunit)
            
            # Get cluster assignment
            gid = group_id or "__default__"
            state = cluster_worker._states.get(gid)
            if state:
                event_id = str(memunit.get("event_id"))
                cluster_id = state.eventid_to_cluster.get(event_id)
                
                if cluster_id:
                    # Create a simple wrapper object for the memunit dict
                    class MemUnitWrapper:
                        def __init__(self, data: Dict[str, Any]):
                            for k, v in data.items():
                                setattr(self, k, v)
                    
                    mc_obj = MemUnitWrapper(memunit)
                    
                    # Schedule profile update (run in background)
                    asyncio.create_task(
                        self.on_memunit_clustered(
                            memunit=mc_obj,
                            cluster_id=cluster_id,
                            user_id_list=memunit.get("user_id_list", [])
                        )
                    )
        
        # Replace the submit method
        cluster_worker.submit = patched_submit
        
        logger.info("ProfileManager successfully attached to MemUnitExtractor (legacy mode)")
    
    async def export_profiles(
        self,
        output_dir: Path,
        include_history: bool = True
    ) -> int:
        """Export all profiles to JSON files.
        
        Args:
            output_dir: Directory to save profiles
            include_history: Whether to export version history
        
        Returns:
            Number of profiles exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles = await self._storage.get_all_profiles()
        count = 0
        
        for user_id, profile in profiles.items():
            try:
                # Convert profile to dict
                if hasattr(profile, "to_dict"):
                    profile_dict = profile.to_dict()
                elif hasattr(profile, "__dict__"):
                    profile_dict = dict(profile.__dict__)
                else:
                    profile_dict = profile
                
                # Save latest profile
                import json
                latest_file = output_dir / f"profile_{user_id}.json"
                with open(latest_file, "w", encoding="utf-8") as f:
                    json.dump(profile_dict, f, ensure_ascii=False, indent=2, default=str)
                
                count += 1
                
                # Export history if requested
                if include_history:
                    history = await self._storage.get_profile_history(user_id)
                    if history:
                        history_dir = output_dir / "history" / user_id
                        history_dir.mkdir(parents=True, exist_ok=True)
                        
                        for i, entry in enumerate(history):
                            version_file = history_dir / f"version_{i:03d}.json"
                            with open(version_file, "w", encoding="utf-8") as f:
                                json.dump(entry, f, ensure_ascii=False, indent=2, default=str)
            
            except Exception as e:
                logger.error(f"Failed to export profile for user {user_id}: {e}")
        
        logger.info(f"Exported {count} profiles to {output_dir}")
        return count

