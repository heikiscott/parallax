# Extraction 层重构计划

## 0. 核心原则

1. **文件名 = 类名**：每个文件名与其内部主类名保持一致（snake_case 对应 PascalCase）
2. **目录名按逻辑简洁**：目录名不需要冗余，按功能模块命名即可

## 1. 目标结构

将现有的 `memunit_extractor/` 和 `memory_extractor/` 合并到统一的 `extraction/` 目录下：

```
src/memory/extraction/                          # L1: 提取层（只依赖 schema）
├── __init__.py                                 # 统一导出接口
│
├── memunit/                                    # 原始数据 → MemUnit
│   ├── __init__.py
│   ├── raw_data.py                             # RawData
│   ├── memunit_extract_request.py              # MemUnitExtractRequest
│   ├── status_result.py                        # StatusResult
│   ├── memunit_extractor.py                    # MemUnitExtractor (抽象基类)
│   └── conv_memunit_extractor.py               # ConvMemUnitExtractor
│
└── memory/                                     # MemUnit → Memory
    ├── __init__.py
    ├── memory_extractor.py                     # MemoryExtractor (抽象基类)
    ├── memory_extract_request.py               # MemoryExtractRequest
    ├── episode_memory_extractor.py             # EpisodeMemoryExtractor
    ├── semantic_memory_extractor.py            # SemanticMemoryExtractor
    ├── event_log_extractor.py                  # EventLogExtractor
    │
    ├── profile/                                # profile 相关子模块（目录名简洁）
    │   ├── __init__.py
    │   ├── profile_memory_extractor.py         # ProfileMemoryExtractor
    │   ├── profile_memory_merger.py            # ProfileMemoryMerger
    │   ├── profile_memory_extract_request.py   # ProfileMemoryExtractRequest
    │   ├── profile_conversation.py             # ProfileConversation (原 conversation.py)
    │   ├── data_normalize.py                   # (辅助模块，保持原名)
    │   ├── empty_evidence_completion.py
    │   ├── evidence_utils.py
    │   ├── profile_helpers.py
    │   ├── project_helpers.py
    │   ├── skill_helpers.py
    │   ├── value_helpers.py
    │   └── types.py
    │
    └── group_profile/                          # group_profile 相关子模块（目录名简洁）
        ├── __init__.py
        ├── group_profile_memory_extractor.py   # GroupProfileMemoryExtractor
        ├── data_processor.py                   # (辅助模块，保持原名)
        ├── llm_handler.py
        ├── role_processor.py
        └── topic_processor.py
```

---

## 2. 关键拆分：base_memunit_extractor.py

当前 `base_memunit_extractor.py` 混合了多个不相关的内容，需要拆分为独立文件（一个类一个文件）：

### 2.1 当前文件内容分析

| 内容 | 行数 | 职责 | 目标文件 |
|------|------|------|----------|
| `RawData` 数据类 | 28-250 | 原始数据封装、JSON序列化/反序列化 | `raw_data.py` |
| `MemUnitExtractRequest` 数据类 | 252-263 | 提取请求参数 | `memunit_extract_request.py` |
| `StatusResult` 数据类 | 265-271 | 状态控制结果 | `status_result.py` |
| `MemUnitExtractor` 抽象类 | 273-285 | 提取器基类 | `memunit_extractor.py` |

### 2.2 同样拆分 base_memory_extractor.py

| 内容 | 目标文件 |
|------|----------|
| `MemoryExtractRequest` 数据类 | `memory_extract_request.py` |
| `MemoryExtractor` 抽象类 | `memory_extractor.py` |

---

## 3. 完整文件映射表

### 3.1 memunit 模块（拆分 + 迁移）

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `base_memunit_extractor.py` → `RawData` | `extraction/memunit/raw_data.py` | 拆分 |
| `base_memunit_extractor.py` → `MemUnitExtractRequest` | `extraction/memunit/memunit_extract_request.py` | 拆分 |
| `base_memunit_extractor.py` → `StatusResult` | `extraction/memunit/status_result.py` | 拆分 |
| `base_memunit_extractor.py` → `MemUnitExtractor` | `extraction/memunit/memunit_extractor.py` | 拆分 |
| `conv_memunit_extractor.py` | `extraction/memunit/conv_memunit_extractor.py` | 迁移 |

### 3.2 memory 基础模块（拆分 + 迁移）

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `base_memory_extractor.py` → `MemoryExtractRequest` | `extraction/memory/memory_extract_request.py` | 拆分 |
| `base_memory_extractor.py` → `MemoryExtractor` | `extraction/memory/memory_extractor.py` | 拆分 |
| `episode_memory_extractor.py` | `extraction/memory/episode_memory_extractor.py` | 迁移 |
| `semantic_memory_extractor.py` | `extraction/memory/semantic_memory_extractor.py` | 迁移 |
| `event_log_extractor.py` | `extraction/memory/event_log_extractor.py` | 迁移 |

### 3.3 profile 子模块

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `profile_memory/extractor.py` | `extraction/memory/profile/profile_memory_extractor.py` | 重命名 |
| `profile_memory/merger.py` | `extraction/memory/profile/profile_memory_merger.py` | 重命名 |
| `profile_memory/types.py` (Request部分) | `extraction/memory/profile/profile_memory_extract_request.py` | 拆分 |
| `profile_memory/conversation.py` | `extraction/memory/profile/profile_conversation.py` | 重命名 |
| `profile_memory/*.py` (辅助模块) | `extraction/memory/profile/*.py` | 保持原名 |
| `profile_memory_extractor.py` | (删除) | 仅为转发文件 |

### 3.4 group_profile 子模块

| 原路径 | 新路径 | 说明 |
|--------|--------|------|
| `group_profile_memory_extractor.py` | `extraction/memory/group_profile/group_profile_memory_extractor.py` | 移入子目录 |
| `group_profile/*.py` | `extraction/memory/group_profile/*.py` | 迁移 |

---

## 4. 重构步骤

### Phase 1: 创建新目录结构

```bash
mkdir -p src/memory/extraction/memunit
mkdir -p src/memory/extraction/memory/profile
mkdir -p src/memory/extraction/memory/group_profile
```

### Phase 2: 拆分并迁移 memunit 模块

从 `base_memunit_extractor.py` 拆分为 4 个独立文件：

1. **创建 `raw_data.py`** - `RawData` 类
2. **创建 `memunit_extract_request.py`** - `MemUnitExtractRequest` 类
3. **创建 `status_result.py`** - `StatusResult` 类
4. **创建 `memunit_extractor.py`** - `MemUnitExtractor` 抽象基类
5. **迁移 `conv_memunit_extractor.py`** - 更新导入路径

6. **创建 `memunit/__init__.py`**

   ```python
   from .raw_data import RawData
   from .memunit_extract_request import MemUnitExtractRequest
   from .status_result import StatusResult
   from .memunit_extractor import MemUnitExtractor
   from .conv_memunit_extractor import (
       ConvMemUnitExtractor,
       ConversationMemUnitExtractRequest,
       BoundaryDetectionResult,
   )
   ```

### Phase 3: 拆分并迁移 memory 基础模块

从 `base_memory_extractor.py` 拆分：

1. **创建 `memory_extract_request.py`** - `MemoryExtractRequest` 类
2. **创建 `memory_extractor.py`** - `MemoryExtractor` 抽象基类
3. **迁移以下文件**（保持原名）：
   - `episode_memory_extractor.py`
   - `semantic_memory_extractor.py`
   - `event_log_extractor.py`
4. 更新各文件内部导入路径

### Phase 4: 迁移 profile 子模块

1. 复制所有文件到 `extraction/memory/profile/`
2. **重命名以匹配类名**：
   - `extractor.py` → `profile_memory_extractor.py`
   - `merger.py` → `profile_memory_merger.py`
   - `conversation.py` → `profile_conversation.py`
3. **拆分 `types.py`**：将 `ProfileMemoryExtractRequest` 拆分到独立文件
4. 更新所有内部导入路径
5. 创建 `profile/__init__.py`：

   ```python
   from .profile_memory_extractor import ProfileMemoryExtractor
   from .profile_memory_merger import ProfileMemoryMerger
   from .profile_memory_extract_request import ProfileMemoryExtractRequest
   from .types import (
       ProjectInfo,
       ImportanceEvidence,
       GroupImportanceEvidence,
   )
   ```

### Phase 5: 迁移 group_profile 子模块

1. 迁移 `group_profile_memory_extractor.py` 到 `extraction/memory/group_profile/`
2. 迁移 `group_profile/` 下所有辅助文件
3. 更新所有内部导入路径
4. 创建 `group_profile/__init__.py`：

   ```python
   from .group_profile_memory_extractor import GroupProfileMemoryExtractor
   ```

### Phase 6: 创建顶层导出

**`extraction/memory/__init__.py`**

```python
from .memory_extract_request import MemoryExtractRequest
from .memory_extractor import MemoryExtractor
from .episode_memory_extractor import EpisodeMemoryExtractor
from .semantic_memory_extractor import SemanticMemoryExtractor
from .event_log_extractor import EventLogExtractor
from .profile import (
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
    ProfileMemoryMerger,
    ProjectInfo,
    ImportanceEvidence,
    GroupImportanceEvidence,
)
from .group_profile import GroupProfileMemoryExtractor
```

**`extraction/__init__.py`**

```python
from .memunit import (
    RawData,
    MemUnitExtractor,
    MemUnitExtractRequest,
    StatusResult,
    ConvMemUnitExtractor,
    ConversationMemUnitExtractRequest,
    BoundaryDetectionResult,
)
from .memory import (
    MemoryExtractor,
    MemoryExtractRequest,
    EpisodeMemoryExtractor,
    SemanticMemoryExtractor,
    EventLogExtractor,
    ProfileMemoryExtractor,
    ProfileMemoryExtractRequest,
    ProfileMemoryMerger,
    GroupProfileMemoryExtractor,
)
```

### Phase 7: 更新外部引用

需要更新以下文件中的导入路径：

1. `src/memory/orchestrator/extraction_orchestrator.py`
2. `src/memory/cluster_manager/manager.py`
3. `src/memory/profile_manager/manager.py`
4. `src/memory/__init__.py`
5. 所有测试文件 (`tests/` 目录)
6. 评估相关文件 (`eval/` 目录)

### Phase 8: 清理旧目录

1. 运行所有测试确保通过
2. 删除 `src/memory/memunit_extractor/` 目录
3. 删除 `src/memory/memory_extractor/` 目录

---

## 5. 导入路径变更对照

### 旧导入方式

```python
from memory.memunit_extractor import ConvMemUnitExtractor, RawData
from memory.memunit_extractor.base_memunit_extractor import MemUnitExtractRequest
from memory.memory_extractor import EpisodeMemoryExtractor
from memory.memory_extractor.profile_memory import ProfileMemoryExtractor
from memory.memory_extractor.profile_memory.extractor import ProfileMemoryExtractor
```

### 新导入方式

```python
# 推荐：从顶层 extraction 导入
from memory.extraction import ConvMemUnitExtractor, RawData
from memory.extraction import EpisodeMemoryExtractor
from memory.extraction import ProfileMemoryExtractor

# 细粒度导入
from memory.extraction.memunit import ConvMemUnitExtractor, RawData
from memory.extraction.memunit.raw_data import RawData
from memory.extraction.memory.episode_memory_extractor import EpisodeMemoryExtractor
from memory.extraction.memory.profile import ProfileMemoryExtractor
from memory.extraction.memory.profile.profile_memory_extractor import ProfileMemoryExtractor
```

---

## 6. 命名规范总结

**核心原则：文件名 = 类名（snake_case ↔ PascalCase）**

| 类名 | 文件名 |
|------|--------|
| `RawData` | `raw_data.py` |
| `MemUnitExtractRequest` | `memunit_extract_request.py` |
| `StatusResult` | `status_result.py` |
| `MemUnitExtractor` | `memunit_extractor.py` |
| `ConvMemUnitExtractor` | `conv_memunit_extractor.py` |
| `MemoryExtractRequest` | `memory_extract_request.py` |
| `MemoryExtractor` | `memory_extractor.py` |
| `EpisodeMemoryExtractor` | `episode_memory_extractor.py` |
| `SemanticMemoryExtractor` | `semantic_memory_extractor.py` |
| `EventLogExtractor` | `event_log_extractor.py` |
| `ProfileMemoryExtractor` | `profile_memory_extractor.py` |
| `ProfileMemoryExtractRequest` | `profile_memory_extract_request.py` |
| `ProfileMemoryMerger` | `profile_memory_merger.py` |
| `ProfileConversation` | `profile_conversation.py` |
| `GroupProfileMemoryExtractor` | `group_profile_memory_extractor.py` |

---

## 7. 验证清单

- [ ] 所有类已拆分为独立文件（一类一文件）
- [ ] 所有文件名与主类名一致
- [ ] 目录名简洁（`profile/` 而非 `profile_memory/`）
- [ ] 所有文件已正确迁移
- [ ] 所有内部导入路径已更新
- [ ] 所有外部引用已更新
- [ ] 单元测试全部通过
- [ ] 集成测试全部通过
- [ ] 评估脚本可正常运行
- [ ] 旧目录已清理

---

## 8. 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 遗漏导入更新 | 运行时错误 | 全局搜索旧路径，确保全部替换 |
| 循环导入 | 启动失败 | 仔细规划导入顺序，必要时延迟导入 |
| 测试覆盖不足 | 隐藏问题 | 重构前确保测试覆盖率足够 |
| 文件拆分过细 | 导入路径变长 | 通过 `__init__.py` 提供简洁的顶层导入 |
