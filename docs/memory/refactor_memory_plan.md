# src/memory 模块重构计划

## 一、现状分析

### 1.1 当前目录结构
```
src/memory/
├── __init__.py
├── schema/                      # 数据模型（底层）
│   ├── memory_type.py           # MemoryType 枚举
│   ├── source_type.py           # SourceType 枚举
│   ├── memunit.py               # MemUnit 数据类
│   ├── memory.py                # Memory 基类
│   ├── episode_memory.py        # EpisodeMemory
│   ├── profile_memory.py        # ProfileMemory
│   ├── group_profile_memory.py  # GroupProfileMemory
│   └── semantic_memory.py       # SemanticMemory
│
├── memunit_extractor/           # MemUnit 提取（中层）
│   ├── base_memunit_extractor.py    # 包含: MemUnitExtractor, RawData, StatusResult, MemUnitExtractRequest
│   └── conv_memunit_extractor.py    # 包含: ConvMemUnitExtractor, BoundaryDetectionResult, ConversationMemUnitExtractRequest
│
├── memory_extractor/            # Memory 提取（中层）
│   ├── base_memory_extractor.py     # 包含: MemoryExtractor, MemoryExtractRequest
│   ├── episode_memory_extractor.py  # 包含: EpisodeMemoryExtractor, EpisodeMemoryExtractRequest
│   ├── semantic_memory_extractor.py # SemanticMemoryExtractor
│   ├── profile_memory_extractor.py  # ProfileMemoryExtractor
│   ├── group_profile_memory_extractor.py
│   ├── event_log_extractor.py
│   ├── profile_memory/          # profile 辅助模块
│   └── group_profile/           # group_profile 辅助模块
│
├── cluster_manager/             # 聚类管理（中层）
│   ├── config.py                # ClusterManagerConfig
│   ├── manager.py               # ClusterManager, ClusterState
│   ├── storage.py               # ClusterStorage, InMemoryClusterStorage
│   └── mongo_cluster_storage.py # MongoClusterStorage
│
├── profile_manager/             # Profile管理（中层）
│   ├── config.py                # ProfileManagerConfig, ScenarioType
│   ├── manager.py               # ProfileManager
│   ├── discriminator.py         # ValueDiscriminator, DiscriminatorConfig
│   ├── storage.py               # ProfileStorage, InMemoryProfileStorage
│   └── mongo_profile_storage.py # MongoProfileStorage
│
├── orchestrator/                # 编排层（顶层）
│   └── extraction_orchestrator.py  # ExtractionOrchestrator, MemorizeRequest
│
└── prompts/                     # 提示词模板
```

### 1.2 发现的问题

#### 问题1: 循环依赖
```
conv_memunit_extractor.py
    ↓ import EpisodeMemoryExtractor (第33行)
episode_memory_extractor.py
```
`memunit_extractor` 应该是 `memory_extractor` 的下层，但现在却反向依赖。

#### 问题2: 文件名与类名不一致
| 文件名 | 主要类 | 问题 |
|--------|--------|------|
| `base_memunit_extractor.py` | `MemUnitExtractor`, `RawData`, `StatusResult` | 文件包含多个不相关的类 |
| `base_memory_extractor.py` | `MemoryExtractor`, `MemoryExtractRequest` | 同上 |
| `manager.py` (cluster) | `ClusterManager`, `ClusterState` | `ClusterState` 应该独立 |
| `storage.py` | 包含接口+实现 | 应该分离 |

#### 问题3: 职责不清晰
- `conv_memunit_extractor.py` 既做边界检测，又调用 episode 提取
- `manager.py` 命名过于泛化

#### 问题4: 存储层分散
- `cluster_manager/storage.py` + `mongo_cluster_storage.py`
- `profile_manager/storage.py` + `mongo_profile_storage.py`
- 相同模式重复，应该统一

---

## 二、重构目标

1. **单向依赖**: 上层依赖下层，无循环
2. **文件名=类名**: 一个文件一个主类（辅助类除外）
3. **职责单一**: 每个模块职责明确
4. **存储集中**: 统一的存储层抽象

### 目标依赖层次
```
L3: orchestrator/          ← 编排层（依赖 L2, L1, L0）
L2: clustering/, profiling/ ← 处理层（依赖 L1, L0）
L1: extraction/            ← 提取层（依赖 L0）
L0: schema/                ← 数据模型层（无依赖）
```

---

## 三、重构步骤（按顺序执行，每步独立可测试）

### 步骤 1: 拆分 base_memunit_extractor.py 中的类 ✅ 独立改动

**目标**: 将 `RawData`, `StatusResult`, `MemUnitExtractRequest` 移到独立文件

**改动**:
1. 创建 `memunit_extractor/raw_data.py` → 移入 `RawData` 类
2. 创建 `memunit_extractor/status_result.py` → 移入 `StatusResult` 类
3. 创建 `memunit_extractor/memunit_extract_request.py` → 移入 `MemUnitExtractRequest` 类
4. `base_memunit_extractor.py` 只保留 `MemUnitExtractor` 基类
5. 更新 `memunit_extractor/__init__.py` 导出
6. 更新所有引用这些类的文件

**影响范围**:
- `conv_memunit_extractor.py`
- `orchestrator/extraction_orchestrator.py`
- 外部引用

---

### 步骤 2: 拆分 base_memory_extractor.py 中的类 ✅ 独立改动

**目标**: 将 `MemoryExtractRequest` 移到独立文件

**改动**:
1. 创建 `memory_extractor/memory_extract_request.py` → 移入 `MemoryExtractRequest` 类
2. `base_memory_extractor.py` 只保留 `MemoryExtractor` 基类
3. 更新 `memory_extractor/__init__.py` 导出
4. 更新所有引用的文件

**影响范围**:
- `episode_memory_extractor.py`
- `profile_memory_extractor.py`
- `group_profile_memory_extractor.py`
- `semantic_memory_extractor.py`

---

### 步骤 3: 从 cluster_manager/manager.py 拆分 ClusterState ✅ 独立改动

**目标**: `ClusterState` 是独立的数据结构，应该有自己的文件

**改动**:
1. 创建 `cluster_manager/cluster_state.py` → 移入 `ClusterState` 类
2. `manager.py` 只保留 `ClusterManager`
3. 更新 import

**影响范围**:
- `cluster_manager/manager.py`
- `cluster_manager/storage.py`

---

### 步骤 4: 拆分存储层接口和实现 ✅ 独立改动

**目标**: 接口与实现分离

**改动**:
1. `cluster_manager/storage.py` → 拆分为:
   - `cluster_manager/cluster_storage.py` (接口 `ClusterStorage`)
   - `cluster_manager/in_memory_cluster_storage.py` (实现 `InMemoryClusterStorage`)

2. `profile_manager/storage.py` → 拆分为:
   - `profile_manager/profile_storage.py` (接口 `ProfileStorage`)
   - `profile_manager/in_memory_profile_storage.py` (实现 `InMemoryProfileStorage`)

3. 更新 `__init__.py` 导出

**影响范围**:
- `cluster_manager/__init__.py`
- `profile_manager/__init__.py`
- `cluster_manager/manager.py`
- `profile_manager/manager.py`

---

### 步骤 5: 解决循环依赖 - 核心改动 ⚠️ 需要设计决策

**问题**: `conv_memunit_extractor.py` 第33行导入了 `EpisodeMemoryExtractor`

**原因分析**:
```python
# conv_memunit_extractor.py 第69行
self.episode_extractor = EpisodeMemoryExtractor(llm_provider, use_eval_prompts)

# 第393行调用
episode_result = await self.episode_extractor.extract_memory(...)
```

`ConvMemUnitExtractor` 在检测到边界后，直接调用 `EpisodeMemoryExtractor` 提取情景记忆。

**解决方案A: 依赖注入（推荐）**
```python
class ConvMemUnitExtractor(MemUnitExtractor):
    def __init__(
        self,
        llm_provider=LLMProvider,
        use_eval_prompts: bool = False,
        episode_extractor=None,  # 可选注入
    ):
        self._episode_extractor = episode_extractor  # 延迟设置
```

调用方（orchestrator）负责组装:
```python
episode_extractor = EpisodeMemoryExtractor(llm_provider)
conv_extractor = ConvMemUnitExtractor(llm_provider, episode_extractor=episode_extractor)
```

**解决方案B: 回调模式**
```python
class ConvMemUnitExtractor(MemUnitExtractor):
    def __init__(self, ...):
        self._on_boundary_detected_callback = None

    def on_boundary_detected(self, callback):
        self._on_boundary_detected_callback = callback
```

**改动**:
1. 修改 `conv_memunit_extractor.py`:
   - 移除对 `episode_memory_extractor` 的 import
   - 构造函数添加 `episode_extractor` 参数（可选）
   - 内部使用注入的 extractor

2. 修改 `orchestrator/extraction_orchestrator.py`:
   - 负责创建并注入 `EpisodeMemoryExtractor`

**影响范围**:
- `conv_memunit_extractor.py`
- `orchestrator/extraction_orchestrator.py`

---

### 步骤 6: 重命名文件以匹配类名 ✅ 独立改动

**改动**:
| 原文件名 | 新文件名 | 主类 |
|----------|----------|------|
| `base_memunit_extractor.py` | `memunit_extractor.py` | `MemUnitExtractor` |
| `conv_memunit_extractor.py` | `conversation_memunit_extractor.py` | `ConvMemUnitExtractor` |
| `base_memory_extractor.py` | `memory_extractor.py` | `MemoryExtractor` |
| `cluster_manager/manager.py` | `cluster_manager/cluster_manager.py` | `ClusterManager` |
| `profile_manager/manager.py` | `profile_manager/profile_manager.py` | `ProfileManager` |
| `mongo_cluster_storage.py` | `mongo_cluster_storage.py` | ✓ 已匹配 |
| `mongo_profile_storage.py` | `mongo_profile_storage.py` | ✓ 已匹配 |

**影响范围**: 所有 import 语句

---

### 步骤 7: （可选）统一存储层到 storage/ 目录

**如果需要更进一步的统一**，可以将存储相关代码集中:

```
src/memory/
├── storage/
│   ├── __init__.py
│   ├── cluster/
│   │   ├── cluster_storage.py          # 接口
│   │   ├── in_memory_cluster_storage.py
│   │   └── mongo_cluster_storage.py
│   └── profile/
│       ├── profile_storage.py          # 接口
│       ├── in_memory_profile_storage.py
│       └── mongo_profile_storage.py
```

**这一步可以推迟**，因为当前存储层与各自 manager 放在一起也是合理的。

---

## 四、重构后的目标结构

```
src/memory/
├── __init__.py
│
├── schema/                              # L0: 数据模型层
│   ├── __init__.py
│   ├── memory_type.py
│   ├── source_type.py
│   ├── memunit.py
│   ├── memory.py
│   ├── episode_memory.py
│   ├── profile_memory.py
│   ├── group_profile_memory.py
│   └── semantic_memory.py
│
├── memunit_extractor/                   # L1: MemUnit 提取层
│   ├── __init__.py
│   ├── raw_data.py                      # RawData 类
│   ├── status_result.py                 # StatusResult 类
│   ├── memunit_extract_request.py       # MemUnitExtractRequest 类
│   ├── memunit_extractor.py             # MemUnitExtractor 基类（原 base_memunit_extractor.py）
│   └── conversation_memunit_extractor.py # ConvMemUnitExtractor（原 conv_memunit_extractor.py）
│
├── memory_extractor/                    # L1: Memory 提取层
│   ├── __init__.py
│   ├── memory_extract_request.py        # MemoryExtractRequest 类
│   ├── memory_extractor.py              # MemoryExtractor 基类（原 base_memory_extractor.py）
│   ├── episode_memory_extractor.py
│   ├── semantic_memory_extractor.py
│   ├── profile_memory_extractor.py
│   ├── group_profile_memory_extractor.py
│   ├── event_log_extractor.py
│   ├── profile_memory/
│   └── group_profile/
│
├── cluster_manager/                     # L2: 聚类处理层
│   ├── __init__.py
│   ├── config.py                        # ClusterManagerConfig
│   ├── cluster_state.py                 # ClusterState（从 manager.py 拆出）
│   ├── cluster_manager.py               # ClusterManager（原 manager.py）
│   ├── cluster_storage.py               # ClusterStorage 接口（从 storage.py 拆出）
│   ├── in_memory_cluster_storage.py     # InMemoryClusterStorage（从 storage.py 拆出）
│   └── mongo_cluster_storage.py
│
├── profile_manager/                     # L2: Profile 处理层
│   ├── __init__.py
│   ├── config.py
│   ├── profile_manager.py               # ProfileManager（原 manager.py）
│   ├── discriminator.py                 # ValueDiscriminator
│   ├── profile_storage.py               # ProfileStorage 接口（从 storage.py 拆出）
│   ├── in_memory_profile_storage.py     # InMemoryProfileStorage（从 storage.py 拆出）
│   └── mongo_profile_storage.py
│
├── orchestrator/                        # L3: 编排层
│   ├── __init__.py
│   └── extraction_orchestrator.py
│
└── prompts/                             # 辅助: 提示词
```

---

## 五、依赖关系图（重构后）

```
┌─────────────────────────────────────────────────────────────┐
│                   L3: orchestrator/                          │
│              ExtractionOrchestrator                          │
│   （负责组装所有组件，解决依赖注入）                            │
└────────────────────────┬────────────────────────────────────┘
                         │ 依赖
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐
│ L2:         │  │ L2:         │  │ L1: memunit_extractor/  │
│ cluster_    │  │ profile_    │  │ memory_extractor/       │
│ manager/    │  │ manager/    │  │                         │
└──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘
       │                │                      │
       │                │    ┌─────────────────┘
       │                │    │
       │                ▼    ▼
       │         ┌─────────────────────────┐
       │         │ L1: memory_extractor/   │
       │         │ (ProfileMemoryExtractor)│
       │         └───────────┬─────────────┘
       │                     │
       └─────────────────────┼─────────────────┐
                             │                 │
                             ▼                 ▼
                     ┌─────────────────────────────┐
                     │        L0: schema/          │
                     │  (MemUnit, Memory, etc.)    │
                     └─────────────────────────────┘
```

**关键变化**:
- `ConvMemUnitExtractor` 不再直接依赖 `EpisodeMemoryExtractor`
- `orchestrator` 负责组装和注入依赖

---

## 六、执行顺序建议

按照以下顺序执行，每步完成后运行测试确保不破坏功能：

1. **步骤 1**: 拆分 `base_memunit_extractor.py`（低风险）
2. **步骤 2**: 拆分 `base_memory_extractor.py`（低风险）
3. **步骤 3**: 拆分 `ClusterState`（低风险）
4. **步骤 4**: 拆分存储层（低风险）
5. **步骤 5**: 解决循环依赖（中风险，需要修改逻辑）
6. **步骤 6**: 重命名文件（低风险，但影响范围广）

每步完成后：
- 运行 `python -c "from memory import *"` 确保导入无误
- 运行相关单元测试
- 检查 IDE 中是否有红色波浪线（导入错误）

---

## 七、风险评估

| 步骤 | 风险级别 | 原因 |
|------|---------|------|
| 步骤1-4 | 🟢 低 | 纯文件拆分，不改变逻辑 |
| 步骤5 | 🟡 中 | 需要修改构造函数和调用方式 |
| 步骤6 | 🟢 低 | 只是重命名，IDE 可以批量替换 |
| 步骤7 | 🟡 中 | 可选，涉及目录结构变化 |

---

## 八、回滚方案

每个步骤都应该在单独的 git commit 中完成，便于回滚：

```bash
git checkout -b refactor/memory-module
# 执行步骤1
git add . && git commit -m "refactor(memory): extract RawData, StatusResult from base_memunit_extractor"
# 执行步骤2
git add . && git commit -m "refactor(memory): extract MemoryExtractRequest from base_memory_extractor"
# ...
```

如果某步骤出问题，可以 `git revert` 单个 commit。
