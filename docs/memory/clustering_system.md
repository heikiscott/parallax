# 聚类系统 (Clustering System)

## 概述

聚类系统（ClusterManager）是 Parallax 记忆系统的核心组件之一，用于将语义相近、时间相邻的 MemUnit 自动归类到同一"主题簇"中。通过聚类，系统能够：

1. **主题聚合**：将分散的对话片段按主题组织
2. **Profile 提取触发**：当簇内积累足够信息时，自动触发用户画像提取
3. **检索增强**（规划中）：支持基于簇的关联检索

## 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│                         消息处理流程                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ConvMemUnitExtractor                             │
│                     (边界检测 + MemUnit 生成)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ attach_to_extractor()
┌─────────────────────────────────────────────────────────────────────┐
│                       ClusterManager                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │   ClusterState   │  │  Vectorize       │  │   Callbacks      │   │
│  │   (聚类状态)      │  │  Service         │  │   (事件通知)      │   │
│  │                  │  │  (向量化服务)     │  │                  │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼ on_cluster_assigned callback
┌─────────────────────────────────────────────────────────────────────┐
│                       ProfileManager                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ ValueDiscriminator│  │ ProfileExtractor │  │  ProfileStorage  │   │
│  │   (价值判别)       │  │  (画像提取)       │  │   (画像存储)      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. ClusterManager

**位置**: `src/memory/cluster_manager/manager.py`

ClusterManager 是聚类系统的核心管理器，负责：
- 接收新的 MemUnit
- 计算语义相似度
- 分配到现有簇或创建新簇
- 触发下游回调

#### 主要方法

```python
class ClusterManager:
    async def cluster_memunit(
        self,
        group_id: str,
        memunit: Dict[str, Any]
    ) -> Optional[str]:
        """
        对 MemUnit 进行聚类

        Args:
            group_id: 群组/会话 ID
            memunit: MemUnit 字典，需包含:
                - unit_id: 唯一标识
                - narrative: 叙事文本（用于向量化）
                - timestamp: 时间戳

        Returns:
            cluster_id: 分配的簇 ID，如 "cluster_001"
        """

    def on_cluster_assigned(
        self,
        callback: Callable[[str, Dict[str, Any], str], None]
    ) -> None:
        """
        注册聚类完成回调

        回调签名: callback(group_id, memunit, cluster_id)
        """

    def attach_to_extractor(self, memunit_extractor: Any) -> None:
        """
        将 ClusterManager 附加到 MemUnitExtractor

        附加后，每次 MemUnit 提取完成会自动触发聚类
        """
```

### 2. ClusterManagerConfig

**位置**: `src/memory/cluster_manager/config.py`

```python
@dataclass
class ClusterManagerConfig:
    # 语义相似度阈值 (0.0-1.0)
    # 高于此阈值的 MemUnit 会被分配到现有簇
    similarity_threshold: float = 0.65

    # 最大时间间隔（天）
    # 超过此间隔的 MemUnit 不会加入现有簇
    max_time_gap_days: float = 7.0

    # 是否启用持久化
    enable_persistence: bool = False

    # 持久化目录（enable_persistence=True 时必填）
    persist_dir: str = None

    # 聚类算法: 'centroid' 或 'nearest'
    clustering_algorithm: str = "centroid"
```

### 3. ClusterState

**位置**: `src/memory/cluster_manager/manager.py`

每个群组维护独立的 ClusterState，存储：

```python
class ClusterState:
    # 所有 MemUnit ID
    unit_ids: List[str]

    # 时间戳列表
    timestamps: List[float]

    # 向量列表
    vectors: List[np.ndarray]

    # 簇 ID 列表
    cluster_ids: List[str]

    # unit_id -> cluster_id 映射
    unitid_to_cluster: Dict[str, str]

    # 下一个簇索引
    next_cluster_idx: int

    # 簇中心向量
    cluster_centroids: Dict[str, np.ndarray]

    # 簇成员数量
    cluster_counts: Dict[str, int]

    # 簇最后更新时间
    cluster_last_ts: Dict[str, Optional[float]]
```

### 4. ClusterStorage

**位置**: `src/memory/cluster_manager/storage.py`

抽象存储接口，支持多种后端：

| 实现类 | 位置 | 说明 |
|--------|------|------|
| `InMemoryClusterStorage` | `storage.py` | 内存存储，可选文件持久化 |
| `MongoClusterStorage` | `mongo_cluster_storage.py` | MongoDB 存储 |

```python
class ClusterStorage(ABC):
    @abstractmethod
    async def save_cluster_state(self, group_id: str, state: Dict) -> bool: ...

    @abstractmethod
    async def load_cluster_state(self, group_id: str) -> Optional[Dict]: ...

    @abstractmethod
    async def get_cluster_assignments(self, group_id: str) -> Dict[str, str]: ...

    @abstractmethod
    async def clear(self, group_id: Optional[str] = None) -> bool: ...
```

## 聚类算法

### Centroid-Based Clustering（默认）

使用簇中心（centroid）进行相似度匹配：

1. **新 MemUnit 到达**
   - 生成 narrative 的 embedding 向量

2. **查找匹配簇**
   - 遍历所有现有簇的中心向量
   - 计算 cosine similarity
   - 检查时间约束（不超过 max_time_gap_days）

3. **分配决策**
   - 如果最高相似度 ≥ `similarity_threshold`，加入该簇
   - 否则创建新簇

4. **更新中心**
   - 增量更新簇中心向量：
   ```
   new_centroid = (old_centroid * count + new_vector) / (count + 1)
   ```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `similarity_threshold` | 0.65 | 相似度阈值，越高越严格 |
| `max_time_gap_days` | 7.0 | 最大时间间隔，防止跨越太久的话题合并 |

## 与 ProfileManager 的集成

### 工作流程

```
MemUnit 提取完成
       │
       ▼
ClusterManager.cluster_memunit()
       │
       ├── 计算 embedding
       ├── 查找/创建簇
       ├── 更新簇状态
       │
       ▼
触发 on_cluster_assigned 回调
       │
       ▼
ProfileManager.on_memunit_clustered()
       │
       ├── 添加到 _cluster_memunits[cluster_id]
       ├── ValueDiscriminator 判断价值
       │
       ▼ (如果高价值且簇内 MemUnit 达到阈值)
       │
ProfileManager._extract_profiles_for_cluster()
       │
       ├── 从 MongoDB 加载完整 MemUnit
       ├── 调用 LLM 提取 Profile
       ├── 保存到 ProfileStorage
       │
       ▼
Profile 更新完成
```

### ProfileManager 配置

**位置**: `src/memory/profile_manager/config.py`

```python
class ScenarioType(Enum):
    GROUP_CHAT = "group_chat"  # 工作/群聊场景
    ASSISTANT = "assistant"     # 助手/陪伴场景

@dataclass
class ProfileManagerConfig:
    # 场景类型
    scenario: ScenarioType = ScenarioType.GROUP_CHAT

    # 价值判别最小置信度
    min_confidence: float = 0.6

    # 是否启用版本历史
    enable_versioning: bool = True

    # 是否自动提取 Profile
    auto_extract: bool = True

    # 每批次最大 MemUnit 数
    batch_size: int = 50

    # 最大重试次数
    max_retries: int = 3
```

### ValueDiscriminator（价值判别器）

**位置**: `src/memory/profile_manager/discriminator.py`

用于判断 MemUnit 是否包含高价值的 Profile 信息：

**group_chat 场景关注**:
- 角色/职责描述
- 技能展示
- 项目参与
- 工作习惯

**assistant 场景关注**:
- 性格特征
- 决策风格
- 兴趣爱好
- 价值观

### 配置示例

```python
from memory.cluster_manager import ClusterManager, ClusterManagerConfig
from memory.profile_manager import ProfileManager, ProfileManagerConfig

# 1. 创建 ClusterManager
cluster_config = ClusterManagerConfig(
    similarity_threshold=0.65,
    max_time_gap_days=7,
    enable_persistence=False
)
cluster_mgr = ClusterManager(config=cluster_config)

# 2. 创建 ProfileManager
profile_config = ProfileManagerConfig(
    scenario="assistant",  # 或 "group_chat"
    min_confidence=0.6,
    auto_extract=True
)
profile_mgr = ProfileManager(llm_provider, config=profile_config)

# 3. 连接组件
profile_mgr.attach_to_cluster_manager(cluster_mgr)

# 4. 附加到 MemUnitExtractor
cluster_mgr.attach_to_extractor(memunit_extractor)
```

## 在生产环境中的使用

**位置**: `src/services/mem_memorize.py` 中的 `_trigger_clustering()` 函数

```python
async def _trigger_clustering(group_id: str, memunit: MemUnit, scene: str = None):
    """
    在 MemUnit 保存后触发聚类

    Args:
        group_id: 群组 ID
        memunit: 保存的 MemUnit
        scene: 场景类型 ("assistant" 或 "group_chat")
    """
    # 1. 获取 MongoDB 存储
    mongo_storage = get_bean_by_type(MongoClusterStorage)

    # 2. 创建 ClusterManager
    cluster_manager = ClusterManager(
        config=ClusterManagerConfig(similarity_threshold=0.65, max_time_gap_days=7),
        storage=mongo_storage
    )

    # 3. 创建 ProfileManager
    profile_scenario = "assistant" if scene in ["assistant", "companion"] else "group_chat"
    profile_manager = ProfileManager(
        llm_provider=llm_provider,
        config=ProfileManagerConfig(scenario=profile_scenario),
        storage=mongo_profile_storage
    )

    # 4. 连接组件
    profile_manager.attach_to_cluster_manager(cluster_manager)

    # 5. 执行聚类
    cluster_id = await cluster_manager.cluster_memunit(group_id, memunit_dict)
```

## 在 Eval 中的使用

**位置**: `eval/adapters/parallax/stage1_memunits_extraction.py`

### 配置开关

在 `eval/adapters/parallax/config.py` 中：

```python
class ExperimentConfig:
    # 功能开关
    enable_semantic_extraction: bool = False  # 语义记忆提取
    enable_clustering: bool = False            # 聚类
    enable_profile_extraction: bool = False    # Profile 提取

    # 聚类配置
    cluster_similarity_threshold: float = 0.65
    cluster_max_time_gap_days: float = 7.0

    # Profile 配置
    profile_scenario: str = "assistant"  # "group_chat" 或 "assistant"
    profile_min_confidence: float = 0.6
    profile_min_memunits: int = 1
```

### 使用方式

```python
# 在 process_single_conversation() 中
if config.enable_clustering:
    cluster_mgr = ClusterManager(
        config=ClusterManagerConfig(
            similarity_threshold=config.cluster_similarity_threshold,
            max_time_gap_days=config.cluster_max_time_gap_days,
            enable_persistence=True,
            persist_dir=str(Path(save_dir) / "clusters" / f"conv_{conv_id}")
        )
    )
    cluster_mgr.attach_to_extractor(memunit_extractor)

if config.enable_profile_extraction and cluster_mgr:
    profile_mgr = ProfileManager(
        llm_provider=llm_provider,
        config=ProfileManagerConfig(scenario=config.profile_scenario),
        storage=InMemoryProfileStorage(enable_persistence=True)
    )
    profile_mgr.attach_to_cluster_manager(cluster_mgr)
```

## 聚类在检索中的应用（规划中）

### 当前状态

目前聚类主要用于 Profile 提取触发，**尚未在检索中使用**。

MemUnit schema 中**没有 cluster_id 字段**，聚类结果存储在独立的 `ClusterState` 中。

### 规划方案

#### 方案 A：Cluster-Aware Retrieval（检索后扩展）

```
Query → 检索 MemUnits (Top-K)
              │
              ▼
获取每个 MemUnit 的 cluster_id (从 ClusterState)
              │
              ▼
扩展：加入同簇的其他 MemUnits
              │
              ▼
重新排序/去重 → 最终 Context
```

**优点**：
- 不需要修改 MemUnit schema
- 利用现有聚类结果
- 实现简单

#### 方案 B：Cluster Episode（簇级别记忆）

为每个 Cluster 生成一个聚合的 Episode：

```
Cluster (多个 MemUnit)
       │
       ▼ LLM 聚合
Cluster Episode (汇总该主题的所有信息)
       │
       ▼ 索引
ES/Milvus (独立的 cluster_episode 类型)
```

**优点**：
- 直接检索到主题级别的信息
- 信息更完整

**缺点**：
- 需要额外的 LLM 调用
- 每次簇更新需要重新生成

## MongoDB 存储结构

### cluster_states 集合

**位置**: `src/infra/adapters/out/persistence/document/memory/cluster_state.py`

```python
class ClusterState(DocumentBase):
    # 主键
    group_id: str  # 群组 ID

    # 基础信息
    unit_ids: List[str]        # 所有 MemUnit ID
    timestamps: List[float]    # 时间戳列表
    cluster_ids: List[str]     # 簇 ID 列表

    # 映射
    unitid_to_cluster: Dict[str, str]  # unit_id -> cluster_id

    # 元数据
    next_cluster_idx: int              # 下一个簇索引

    # 簇中心
    cluster_centroids: Dict[str, List[float]]  # cluster_id -> 向量
    cluster_counts: Dict[str, int]             # cluster_id -> 成员数
    cluster_last_ts: Dict[str, float]          # cluster_id -> 最后时间
```

## 文件结构

```
src/memory/cluster_manager/
├── __init__.py              # 模块导出
├── config.py                # ClusterManagerConfig 配置类
├── manager.py               # ClusterManager 核心实现
├── storage.py               # ClusterStorage 抽象 + InMemoryClusterStorage
└── mongo_cluster_storage.py # MongoClusterStorage 实现

src/memory/profile_manager/
├── __init__.py              # 模块导出
├── config.py                # ProfileManagerConfig + ScenarioType
├── manager.py               # ProfileManager 核心实现
├── discriminator.py         # ValueDiscriminator (价值判别)
├── storage.py               # ProfileStorage 抽象 + InMemoryProfileStorage
└── mongo_profile_storage.py # MongoProfileStorage 实现

src/infra/adapters/out/persistence/document/memory/
└── cluster_state.py         # ClusterState MongoDB 文档模型
```

## 使用示例

### 基本使用

```python
from memory.cluster_manager import ClusterManager, ClusterManagerConfig

# 创建 ClusterManager
config = ClusterManagerConfig(
    similarity_threshold=0.65,
    max_time_gap_days=7
)
cluster_mgr = ClusterManager(config)

# 注册回调
def on_cluster(group_id, memunit, cluster_id):
    print(f"MemUnit {memunit['unit_id']} -> {cluster_id}")

cluster_mgr.on_cluster_assigned(on_cluster)

# 聚类 MemUnit
memunit = {
    "unit_id": "mu_001",
    "narrative": "今天讨论了项目进度...",
    "timestamp": datetime.now().timestamp()
}
cluster_id = await cluster_mgr.cluster_memunit("group_001", memunit)
```

### 与 MemUnitExtractor 集成

```python
from memory.extraction.memunit import ConvMemUnitExtractor
from memory.cluster_manager import ClusterManager

# 创建组件
extractor = ConvMemUnitExtractor(llm_provider)
cluster_mgr = ClusterManager()

# 附加聚类器
cluster_mgr.attach_to_extractor(extractor)

# 现在每次 extract_memunit() 成功后会自动触发聚类
```

### 导出聚类结果

```python
# 导出到文件
await cluster_mgr.export_clusters(Path("./output/clusters"))

# 获取统计信息
stats = cluster_mgr.get_stats()
print(f"总 MemUnits: {stats['total_memunits']}")
print(f"总簇数: {stats['total_clusters']}")
```

## 注意事项

1. **向量化服务依赖**
   - ClusterManager 依赖 `deep_infra_vectorize_service` 获取 embedding
   - 如果服务不可用，会创建单例簇（每个 MemUnit 一个簇）

2. **时间约束**
   - `max_time_gap_days` 防止跨越太久的话题合并
   - 对于持续性话题，建议适当放宽此值

3. **阈值调优**
   - `similarity_threshold=0.65` 是经验值
   - 太高：簇过于碎片化
   - 太低：不相关话题被合并

4. **性能考虑**
   - 聚类是增量式的，每个 MemUnit O(n) 复杂度（n = 现有簇数）
   - 对于大量簇，考虑使用近似最近邻算法

## 相关文档

- [记忆系统设计](memory_system_design.md)
- [MemUnit 规范](memunit_specification.md)
- [ProfileManager README](../../src/memory/profile_manager/README.md)
