# MemUnit 标识字段重命名计划

## 决策

**选择方案: `unit_id`**

理由：看重自描述性，与 `user_id`、`group_id` 命名风格一致。

### 最终确认 (2024-12-XX)

- ✅ 选择 `unit_id` 作为新命名
- ✅ 全局统一：所有 MemUnit 的 ID 都叫 `unit_id`
- ✅ 数据库字段也统一改为 `unit_id`
- ✅ 无需迁移：当前处于开发阶段，无生产数据

## 命名映射

| 原命名 | 新命名 | 说明 |
|-------|-------|------|
| `MemUnit.event_id` | `MemUnit.unit_id` | 主标识字段 |
| `memunit_event_id_list` | `memunit_id_list` | Memory 中关联 MemUnit 的列表 |
| `get_by_event_id()` | `get_by_unit_id()` | Repository 方法 |
| `delete_by_event_id()` | `delete_by_unit_id()` | Repository 方法 |

**注意：以下不改动**
- `EpisodeMemory.event_id` - 情景记忆自身的标识，保持不变
- 其他实体的 `event_id` 字段 - 如 `BehaviorHistory.event_id` 等

---

## 改动清单

### 1. Schema 层 (src/memory/schema/)

#### 1.1 memunit.py
| 行号 | 原代码 | 新代码 | 类型 |
|-----|--------|--------|-----|
| 32 | `event_id="evt_123"` | `unit_id="evt_123"` | 示例 |
| 68 | `- event_id: 唯一标识符` | `- unit_id: 唯一标识符` | 注释 |
| 104 | `- event_id: 必填` | `- unit_id: 必填` | 注释 |
| 110 | `event_id: str` | `unit_id: str` | 字段定义 |
| 142 | `if not self.event_id:` | `if not self.unit_id:` | 验证 |
| 143 | `raise ValueError("event_id 是必填字段")` | `raise ValueError("unit_id 是必填字段")` | 错误信息 |
| 153 | `f"MemUnit(event_id={self.event_id}"` | `f"MemUnit(unit_id={self.unit_id}"` | __repr__ |
| 168 | `"event_id": self.event_id` | `"unit_id": self.unit_id` | to_dict |

#### 1.2 memory.py
| 行号 | 原代码 | 新代码 | 类型 |
|-----|--------|--------|-----|
| 79 | `memunit_event_id_list` | `memunit_id_list` | 注释 |
| 113 | `memunit_event_id_list: Optional[List[str]]` | `memunit_id_list: Optional[List[str]]` | 字段定义 |

#### 1.3 episode_memory.py
- 第 12, 46, 80, 81, 88, 109 行：仅注释中提及，说明 `event_id` 是 Episode 自身的标识
- **不需要修改**（EpisodeMemory.event_id 保持不变）

---

### 2. Extraction 层 (src/memory/extraction/)

#### 2.1 memunit/conv_memunit_extractor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 368 | `event_id=str(uuid.uuid4())` | `unit_id=str(uuid.uuid4())` |

#### 2.2 memory/episode_memory_extractor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 481 | `memunit.event_id for memunit in request.memunit_list` | `memunit.unit_id for memunit in request.memunit_list` |
| 490-491 | `memunit_event_id_list=[memunit.event_id ...]` | `memunit_id_list=[memunit.unit_id ...]` |
| 521 | `memunit.event_id for memunit in request.memunit_list` | `memunit.unit_id for memunit in request.memunit_list` |
| 534-535 | `memunit_event_id_list=[memunit.event_id ...]` | `memunit_id_list=[memunit.unit_id ...]` |

#### 2.3 memory/semantic_memory_extractor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 107 | `memunit.event_id` | `memunit.unit_id` |

#### 2.4 memory/group_profile/topic_processor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 88-89 | `hasattr(memunit, 'event_id')` / `memunit.event_id` | `hasattr(memunit, 'unit_id')` / `memunit.unit_id` |

#### 2.5 memory/group_profile/data_processor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 78-79 | `hasattr(memunit, 'event_id')` / `memunit.event_id` | `hasattr(memunit, 'unit_id')` / `memunit.unit_id` |
| 148-150 | `hasattr(memunit, 'event_id')` / `memunit.event_id` | `hasattr(memunit, 'unit_id')` / `memunit.unit_id` |
| 275 | `getattr(memunit, 'event_id', ...)` | `getattr(memunit, 'unit_id', ...)` |

#### 2.6 memory/group_profile/group_profile_memory_extractor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 306-308 | `mc.event_id` / `hasattr(mc, 'event_id')` | `mc.unit_id` / `hasattr(mc, 'unit_id')` |
| 361 | `mc.event_id for mc in memunit_list if hasattr(mc, 'event_id')` | `mc.unit_id for mc in memunit_list if hasattr(mc, 'unit_id')` |

#### 2.7 memory/profile/profile_memory_extractor.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 95 | `getattr(memunit, "event_id", None)` | `getattr(memunit, "unit_id", None)` |
| 549 | `event_id = build_episode_text(...)` | 变量名可保持（这里 event_id 是局部变量） |
| 585-587 | `getattr(memunit, "event_id", None)` | `getattr(memunit, "unit_id", None)` |
| 678-680 | `mc.event_id` / `hasattr(mc, 'event_id')` | `mc.unit_id` / `hasattr(mc, 'unit_id')` |
| 754-767 | `mc.event_id` / `hasattr(mc, 'event_id')` | `mc.unit_id` / `hasattr(mc, 'unit_id')` |

#### 2.8 memory/profile/profile_conversation.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 147 | `getattr(memunit, "event_id", None)` | `getattr(memunit, "unit_id", None)` |
| 192 | `getattr(memunit, "event_id", None)` | `getattr(memunit, "unit_id", None)` |

---

### 3. Services 层 (src/services/)

#### 3.1 mem_memorize.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 106 | `event_id={memunit.event_id}` | `unit_id={memunit.unit_id}` |
| 170 | `"event_id": str(memunit.event_id)` | `"unit_id": str(memunit.unit_id)` |
| 177-178 | `memunit_dict['event_id']` | `memunit_dict['unit_id']` |
| 189-192 | `memunit.event_id` | `memunit.unit_id` |
| 775 | `memunit.event_id` | `memunit.unit_id` |
| 805 | `get_by_event_id(str(memunit.event_id))` | `get_by_unit_id(str(memunit.unit_id))` |
| 815, 819, 823 | `memunit.event_id` | `memunit.unit_id` |
| 891 | `event_id=memunit.event_id` | `unit_id=memunit.unit_id` |
| 1055-1064 | `memunit.event_id` / `result.event_id` | `memunit.unit_id` / `result.unit_id` |

#### 3.2 mem_db_operations.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 283 | `memunit_event_id_list=getattr(...)` | `memunit_id_list=getattr(...)` |
| 891 | `event_id=memunit.event_id` | `unit_id=memunit.unit_id` |
| 1055 | `memunit.event_id` | `memunit.unit_id` |
| 1061-1064 | `memunit.event_id` / `result.event_id` | `memunit.unit_id` / `result.unit_id` |

#### 3.3 memunit_sync.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 81, 85, 91, 103, 109 | `memunit.event_id` | `memunit.unit_id` |
| 125, 129 | `memunit.event_id` | `memunit.unit_id` |
| 148 | `event_id=str(memunit.event_id)` | `event_id=str(memunit.unit_id)` (注意：这里 event_id 是参数名，指向 ES/Milvus 的 episode event_id) |
| 161 | `memunit_event_id_list=[str(memunit.event_id)]` | `memunit_id_list=[str(memunit.unit_id)]` |
| 163, 187 | `memunit.event_id` | `memunit.unit_id` |
| 199 | `memunit_event_id_list=[str(memunit.event_id)]` | `memunit_id_list=[str(memunit.unit_id)]` |
| 206, 231 | `memunit.event_id` | `memunit.unit_id` |

#### 3.4 memunit_milvus_sync.py
（与 memunit_sync.py 相同的改动模式）

---

### 4. Cluster Manager (src/memory/cluster_manager/)

#### 4.1 manager.py
| 行号 | 原代码 | 新代码 | 说明 |
|-----|--------|--------|-----|
| 28 | `self.event_ids: List[str]` | `self.unit_ids: List[str]` | ClusterState 字段 |
| 40 | `def assign_new_cluster(self, event_id: str)` | `def assign_new_cluster(self, unit_id: str)` | 方法参数 |
| 44 | 注释 `event_id: Event identifier` | `unit_id: Unit identifier` | |
| 57 | `event_id: str` | `unit_id: str` | 方法参数 |
| 65 | 注释 `event_id: Event identifier` | `unit_id: Unit identifier` | |
| 112 | `"event_ids": self.event_ids` | `"unit_ids": self.unit_ids` | to_dict |
| 136 | `state.event_ids = list(data.get("event_ids", []))` | `state.unit_ids = list(data.get("unit_ids", []))` | from_dict |
| 181 | `memunit['event_id']` | `memunit['unit_id']` | |
| 274-275 | `memunit.get("event_id", "")` | `memunit.get("unit_id", "")` | |
| 276 | `logger.warning("MemUnit missing event_id...")` | `logger.warning("MemUnit missing unit_id...")` | |
| 287 | `state.event_ids.append(event_id)` | `state.unit_ids.append(unit_id)` | |
| 309 | `state.event_ids.append(event_id)` | `state.unit_ids.append(unit_id)` | |
| 428 | `memunit.get("event_id", "")` | `memunit.get("unit_id", "")` | |

#### 4.2 storage.py
需检查是否有 `event_id` 引用

#### 4.3 mongo_cluster_storage.py
需检查是否有 `event_id` 引用

---

### 5. Profile Manager (src/memory/profile_manager/)

#### 5.1 manager.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 284-285 | `getattr(mc, 'event_id', None)` | `getattr(mc, 'unit_id', None)` |
| 292 | `{"event_id": str(event_id)}` | `{"unit_id": str(unit_id)}` |
| 560 | `memunit.get("event_id")` | `memunit.get("unit_id")` |

---

### 6. Orchestrator (src/memory/orchestrator/)

#### 6.1 extraction_orchestrator.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 164 | `memunit.event_id` | `memunit.unit_id` |
| 172 | `memunit.event_id` | `memunit.unit_id` |

---

### 7. Infrastructure 层 (src/infra/adapters/)

#### 7.1 persistence/document/memory/memunit.py
| 行号 | 原代码 | 新代码 | 说明 |
|-----|--------|--------|-----|
| 137-138 | `@property def event_id(self)` | `@property def unit_id(self)` | 属性名改为 unit_id |

**注意**: 这个属性返回 `self.id`（MongoDB 的 `_id`），所以实际值不变，只是属性名变了。

#### 7.2 persistence/repository/memunit_raw_repository.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 42 | `async def get_by_event_id(self, event_id: str)` | `async def get_by_unit_id(self, unit_id: str)` |
| 47 | 注释 `event_id: 事件 ID` | `unit_id: MemUnit ID` |
| 55-60 | 日志中的 `event_id` | `unit_id` |
| 71 | `memunit.event_id` | `memunit.unit_id` |
| 79-87 | 方法参数和注释中的 `event_id` | `unit_id` |
| 95-129 | 所有 `event_id` 相关 | `unit_id` |

#### 7.3 persistence/document/memory/episodic_memory.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 34 | `memunit_event_id_list: Optional[List[str]]` | `memunit_id_list: Optional[List[str]]` |

#### 7.4 search/elasticsearch/memory/episodic_memory.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 116 | `memunit_event_id_list = e_field.Keyword(multi=True)` | `memunit_id_list = e_field.Keyword(multi=True)` |

**注意**: ES 索引字段变更需要重建索引或使用别名迁移

#### 7.5 search/repository/episodic_memory_es_repository.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 94 | `memunit_event_id_list: Optional[List[str]]` | `memunit_id_list: Optional[List[str]]` |
| 118, 151 | `memunit_event_id_list` | `memunit_id_list` |
| 433, 466 | `memunit_event_id_list` | `memunit_id_list` |

#### 7.6 search/repository/episodic_memory_milvus_repository.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 60 | `memunit_event_id_list: Optional[List[str]]` | `memunit_id_list: Optional[List[str]]` |
| 86, 114 | `memunit_event_id_list` | `memunit_id_list` |

#### 7.7 search/milvus/converter/episodic_memory_milvus_converter.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 65-66 | `source_doc.event_id` / `hasattr(source_doc, 'event_id')` | 需要检查这里是 MemUnit 还是 Episode |
| 136 | `"memunit_event_id_list": getattr(...)` | `"memunit_id_list": getattr(...)` |

#### 7.8 search/elasticsearch/converter/episodic_memory_converter.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 88-89 | `memunit_event_id_list=getattr(source_doc, 'memunit_event_id_list', None)` | `memunit_id_list=getattr(source_doc, 'memunit_id_list', None)` |
| 202-203 | 同上 | 同上 |

#### 7.9 search/elasticsearch/converter/semantic_memory_converter.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 85 | `memunit_event_id_list=[source_doc.parent_episode_id]` | `memunit_id_list=[source_doc.parent_episode_id]` |

#### 7.10 search/elasticsearch/converter/event_log_converter.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 73 | `memunit_event_id_list=[source_doc.parent_episode_id]` | `memunit_id_list=[source_doc.parent_episode_id]` |

#### 7.11 search/repository/semantic_memory_es_repository.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 155 | `memunit_event_id_list=[]` | `memunit_id_list=[]` |

#### 7.12 search/repository/event_log_es_repository.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 144 | `memunit_event_id_list=[]` | `memunit_id_list=[]` |

---

### 8. Agents 层 (src/agents/)

#### 8.1 memory_models.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 141 | `memunit_event_id_list: Optional[List[str]]` | `memunit_id_list: Optional[List[str]]` |

#### 8.2 memory_manager.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 842, 859, 873, 886 | `memunit_event_id_list = source.get('memunit_event_id_list', [])` | `memunit_id_list = source.get('memunit_id_list', [])` |
| 922 | `if memunit_event_id_list:` | `if memunit_id_list:` |
| 924 | `for event_id in memunit_event_id_list:` | `for unit_id in memunit_id_list:` |
| 925 | `await memunit_repo.get_by_event_id(event_id)` | `await memunit_repo.get_by_unit_id(unit_id)` |
| 929 | `f"未找到 memunit: event_id={event_id}"` | `f"未找到 memunit: unit_id={unit_id}"` |
| 949 | `memunit_event_id_list=memunit_event_id_list` | `memunit_id_list=memunit_id_list` |

#### 8.3 fetch_memory_service.py
- 第 100-106, 651-664, 785-791 行：这些是获取 Episode 的方法，`event_id` 参数指的是 Episode 的 ID
- **不需要修改**

---

### 9. Prompts 层 (src/prompts/)

#### 9.1 memory/zh/semantic_mem_prompts.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 37, 83, 173, 235 | `"event_id": "..."` | `"unit_id": "..."` |

#### 9.2 memory/en/production/semantic_mem_prompts.py
| 行号 | 原代码 | 新代码 |
|-----|--------|--------|
| 35, 63, 131, 159 | `"event_id": "..."` | `"unit_id": "..."` |

---

### 10. Eval 目录 (eval/)

#### 10.1 adapters/parallax/stage1_memunits_extraction.py
需检查具体引用

#### 10.2 adapters/parallax/stage3_memory_retrivel.py
需检查具体引用

#### 10.3 adapters/parallax/stage4_response.py
需检查具体引用

#### 10.4 adapters/parallax/config.py
需检查具体引用

---

### 11. Demo 目录 (demo/)

#### 11.1 extract/extractor.py
需检查具体引用

#### 11.2 performance_test.py
需检查具体引用

---

## 不需要修改的部分

以下 `event_id` 属于其他实体，**不需要修改**：

1. **EpisodeMemory.event_id** - 情景记忆自身的唯一标识
2. **BehaviorHistory.event_id** - 行为历史的标识
3. **ES/Milvus 中存储 Episode 的 event_id 字段** - 这是 Episode 的 ID
4. **fetch_memory_service.py 中的 event_id 参数** - 指 Episode ID
5. **episodic_memory_raw_repository.py 中的 event_id** - 指 Episode ID

---

## 数据库迁移考虑

> **注意**: 当前处于开发阶段，无生产数据，无需迁移。直接修改字段名即可。

### MongoDB
- `MemUnit` 文档模型的 `event_id` 属性返回 `self.id`
- 实际数据库字段是 `_id`，不受影响
- 只需修改属性名为 `unit_id`

### Elasticsearch
- `memunit_event_id_list` 字段直接改为 `memunit_id_list`
- 无需迁移策略

### Milvus
- 同上，直接修改字段名

---

## 实施顺序建议

1. **Phase 1: Schema 层** (最先，其他依赖此)
   - memunit.py
   - memory.py

2. **Phase 2: Document 模型**
   - persistence/document/memory/memunit.py
   - persistence/document/memory/episodic_memory.py
   - elasticsearch/memory/episodic_memory.py

3. **Phase 3: Repository 层**
   - memunit_raw_repository.py
   - episodic_memory_*_repository.py

4. **Phase 4: Extraction 层**
   - 所有 extractor 文件

5. **Phase 5: Services 层**
   - mem_memorize.py
   - mem_db_operations.py
   - memunit_sync.py

6. **Phase 6: Manager 层**
   - cluster_manager
   - profile_manager

7. **Phase 7: Agents 层**
   - memory_manager.py
   - memory_models.py

8. **Phase 8: 其他**
   - prompts
   - eval
   - demo

---

## 风险与缓解

| 风险 | 级别 | 缓解措施 |
|-----|------|---------|
| 遗漏修改点 | 中 | 全局搜索验证 + 单元测试 |
| ES 索引不兼容 | 中 | 预先规划迁移策略 |
| 运行时错误 | 中 | 分阶段部署，逐步验证 |
| 外部 API 调用方 | 低 | 确认无外部依赖 |

---

## 验证清单

- [ ] 全局搜索 `event_id` 确认无遗漏
- [ ] 全局搜索 `memunit_event_id_list` 确认全部替换
- [ ] 单元测试全部通过
- [ ] 集成测试验证数据流
- [ ] ES/Milvus 数据迁移验证

---

*更新时间: 2024-XX-XX*
*状态: 待实施*
