# 群体事件聚类系统设计说明书 (Group Event Cluster)

## 1. 概述

### 1.1 背景

在 LoCoMo 评估中发现，当前的记忆检索系统存在以下问题：

| 问题 | 错题示例 | 说明 |
|------|----------|------|
| **信息分散** | qa23 (books read) | 同一主题（读书）的信息分布在多个 MemUnit 中，检索只命中部分 |
| **关联缺失** | qa11 (Sweden) | "home country" 和 "Sweden" 在不同 MemUnit 中，无法关联 |
| **上下文不足** | qa21 (picnic time) | 单个 MemUnit 缺乏时间推理所需的上下文 |

### 1.2 目标

设计一套基于 LLM 的群体事件聚类系统，实现：

1. **按事件聚类**：将讨论同一事件/主题的 MemUnit 归类到一起
2. **群体视角**：以第三人称描述事件，适用于多人对话场景
3. **时间有序**：聚类内的 MemUnit 按时间排序，查找和返回都保持时间顺序
4. **双向查找**：支持 MemUnit → Cluster 和 Cluster → MemUnits 的双向检索
5. **检索增强**：在检索时，通过 Cluster 扩展关联的 MemUnit，提高召回率
6. **通用设计**：代码放在 `src/` 中作为通用模块，Eval 是调用方之一

### 1.3 设计原则

| 原则 | 说明 |
|------|------|
| **LLM 驱动** | 核心聚类决策由 LLM 完成，而非纯向量相似度 |
| **离线处理** | 聚类在索引构建阶段完成，不影响检索延迟 |
| **可解释** | 每个 Cluster 有明确的主题和汇总描述 |
| **渐进式** | 支持增量添加 MemUnit 到现有 Cluster |
| **时间有序** | 所有成员列表按时间戳排序 |
| **配置化** | LLM 模型和 API Key 通过配置指定 |
| **简单存储** | 使用 JSON 文件存储，便于调试和分析 |

### 1.4 命名说明

| 名称 | 说明 |
|------|------|
| **GroupEventCluster** | 群体事件聚类，强调"群体视角"（第三人称） |
| **Group** | 表示多人对话场景，以群体视角描述 |
| **Event** | 表示聚类粒度是"事件"级别 |

---

## 2. 核心概念

### 2.1 术语定义

| 术语 | 英文 | 定义 |
|------|------|------|
| 群体事件聚类 | Group Event Cluster | 一组讨论同一事件/主题的 MemUnit 集合，以群体视角描述 |
| 主题 | Topic | Cluster 的核心主题，如 "Caroline 的领养计划" |
| 汇总 | Summary | Cluster 的第三人称群体视角描述 |
| 聚类索引 | Cluster Index | 存储所有 Cluster 及其映射关系的数据结构 |

### 2.2 与现有概念的关系

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Memory System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Message Stream                                                        │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                    MemUnit (记忆单元)                            │   │
│   │  - 边界检测的输出                                                │   │
│   │  - 包含 narrative (叙事), event_log (事件日志) 等                │   │
│   │  - 是聚类的基本单位                                              │   │
│   │  - 【注意】MemUnit 本身不存储 cluster_id                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       │ LLM 判断归属                                                    │
│       ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │              GroupEventCluster (群体事件聚类) [NEW]               │   │
│   │  - 多个 MemUnit 的逻辑分组（按时间排序）                          │   │
│   │  - 基于事件/主题的语义关联                                        │   │
│   │  - 包含 topic (主题), summary (汇总)                             │   │
│   │  - 支持检索时的上下文扩展                                         │   │
│   │  - 通过 unit_to_cluster 映射关联 MemUnit                         │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 MemUnit 与 Cluster 的关系

**设计决策：MemUnit 本身不存储 cluster_id**

| 方案 | 说明 | 采用 |
|------|------|------|
| MemUnit 添加 cluster_id 字段 | 修改 MemUnit schema | ❌ 不采用 |
| **通过 Index 维护映射关系** | `unit_to_cluster: {unit_id: cluster_id}` | ✅ 采用 |
| 存储在 MemUnit.extend 中 | `extend["group_event_cluster_id"]` | 可选（用于缓存） |

**理由**：
1. **保持 MemUnit 纯净**：MemUnit 是边界检测的输出，聚类是后续处理
2. **支持多种聚类策略**：同一 MemUnit 可能在不同策略下属于不同 Cluster
3. **避免循环依赖**：MemUnit 不依赖聚类模块
4. **灵活性**：可以在不修改 MemUnit 的情况下重新聚类

### 2.4 聚类粒度

**聚类的粒度是"事件"级别**，而非"话题"或"实体"级别：

| 粒度 | 示例 | 是否采用 |
|------|------|---------|
| 实体级 | "关于 Caroline 的所有对话" | ❌ 太粗，会混入无关内容 |
| **事件级** | "Caroline 的领养计划讨论" | ✅ 采用 |
| 话题级 | "关于孩子的讨论" | ❌ 太粗，不同人的孩子话题会混在一起 |
| 消息级 | 单条消息 | ❌ 太细，失去聚类意义 |

**事件的判断标准**：
1. 涉及相同的核心人物
2. 讨论同一件具体的事情（有因果或时间连续性）
3. 有共同的背景或目标

---

## 3. 数据结构设计

### 3.1 GroupEventCluster（群体事件聚类）

```python
@dataclass
class GroupEventCluster:
    """
    群体事件聚类 - 一组语义相关的 MemUnit（按时间排序）

    设计思路：
    - cluster_id: 唯一标识，用于索引和引用
    - topic: 简短的主题名称，便于 LLM 判断新 MemUnit 归属
    - summary: 详细的汇总描述，用于检索时提供上下文
    - members: 成员列表，【按时间戳排序】
    """

    # === 标识字段 ===
    cluster_id: str
    """
    唯一标识符
    格式: "gec_{index:03d}"，如 "gec_001", "gec_002"
    gec = Group Event Cluster
    """

    # === 主题字段 ===
    topic: str
    """
    主题名称（简短，10-30字）
    用途：
    1. 在 LLM 判断时作为 Cluster 的标识
    2. 在检索结果中展示
    示例：
    - "Caroline 的领养计划"
    - "Melanie 的读书分享"
    - "周末野餐活动"
    """

    summary: str
    """
    汇总描述（详细，100-300字）
    特点：
    1. 第三人称群体视角
    2. 包含关键事实（人物、时间、地点、事件）
    3. 随 Cluster 成员增加而更新
    用途：
    1. 检索时提供丰富的上下文
    2. 可作为额外的检索目标（可选）
    """

    # === 成员字段（按时间排序）===
    members: List[ClusterMember]
    """
    成员列表，【按 timestamp 升序排列】
    每次添加新成员后自动排序
    """

    # === 时间字段 ===
    first_timestamp: datetime
    """最早的 MemUnit 时间戳"""

    last_timestamp: datetime
    """最新的 MemUnit 时间戳"""

    # === 元数据 ===
    created_at: datetime
    """Cluster 创建时间"""

    updated_at: datetime
    """Cluster 最后更新时间"""


@dataclass
class ClusterMember:
    """
    聚类成员 - 记录 MemUnit 在 Cluster 中的信息
    """
    unit_id: str
    """MemUnit 的唯一标识"""

    timestamp: datetime
    """MemUnit 的时间戳，用于排序"""

    summary: str
    """该 MemUnit 的简短摘要（1-2句话）"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "timestamp": self.timestamp.isoformat(),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ClusterMember":
        return cls(
            unit_id=data["unit_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            summary=data["summary"],
        )
```

### 3.2 GroupEventClusterIndex（聚类索引）

```python
@dataclass
class GroupEventClusterIndex:
    """
    群体事件聚类索引 - 管理所有 Cluster 及其映射关系

    核心功能：
    1. 存储所有 GroupEventCluster
    2. 维护 unit_id → cluster_id 的映射（双向查找）
    3. 支持序列化/反序列化（JSON 持久化）
    4. 所有返回的成员列表都按时间排序
    """

    # === 数据存储 ===
    clusters: Dict[str, GroupEventCluster]
    """
    cluster_id → GroupEventCluster 映射
    所有 Cluster 的主存储
    """

    unit_to_cluster: Dict[str, str]
    """
    unit_id → cluster_id 映射
    用于快速查找 MemUnit 所属的 Cluster
    【这是 MemUnit 与 Cluster 关联的唯一来源】
    """

    # === 元数据 ===
    conversation_id: str
    """所属对话的 ID"""

    total_units: int
    """总 MemUnit 数量"""

    created_at: datetime
    """索引创建时间"""

    updated_at: datetime
    """索引最后更新时间"""

    llm_model: str
    """使用的 LLM 模型"""

    # === 查询方法 ===
    def get_cluster(self, cluster_id: str) -> Optional[GroupEventCluster]:
        """通过 cluster_id 获取 Cluster"""
        return self.clusters.get(cluster_id)

    def get_cluster_by_unit(self, unit_id: str) -> Optional[GroupEventCluster]:
        """通过 unit_id 获取所属的 Cluster"""
        cluster_id = self.unit_to_cluster.get(unit_id)
        return self.clusters.get(cluster_id) if cluster_id else None

    def get_cluster_id_by_unit(self, unit_id: str) -> Optional[str]:
        """通过 unit_id 获取 cluster_id"""
        return self.unit_to_cluster.get(unit_id)

    def get_units_by_cluster(self, cluster_id: str) -> List[str]:
        """
        获取 Cluster 的所有成员 unit_id
        【返回按时间排序的列表】
        """
        cluster = self.clusters.get(cluster_id)
        if not cluster:
            return []
        # members 已按时间排序
        return [m.unit_id for m in cluster.members]

    def get_related_units(self, unit_id: str, exclude_self: bool = True) -> List[str]:
        """
        获取与指定 unit_id 同 Cluster 的其他 MemUnit
        【返回按时间排序的列表】
        这是检索扩展的核心方法
        """
        cluster = self.get_cluster_by_unit(unit_id)
        if not cluster:
            return []

        # members 已按时间排序
        if exclude_self:
            return [m.unit_id for m in cluster.members if m.unit_id != unit_id]
        return [m.unit_id for m in cluster.members]

    def get_cluster_topic(self, unit_id: str) -> Optional[str]:
        """获取 unit_id 所属 Cluster 的主题"""
        cluster = self.get_cluster_by_unit(unit_id)
        return cluster.topic if cluster else None

    # === 统计方法 ===
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        cluster_sizes = [len(c.members) for c in self.clusters.values()]
        return {
            "total_clusters": len(self.clusters),
            "total_units": self.total_units,
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
            "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
            "singleton_clusters": sum(1 for s in cluster_sizes if s == 1),
        }

    # === 序列化方法 ===
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "clusters": {
                cid: cluster.to_dict()
                for cid, cluster in self.clusters.items()
            },
            "unit_to_cluster": self.unit_to_cluster,
            "metadata": {
                "conversation_id": self.conversation_id,
                "total_units": self.total_units,
                "total_clusters": len(self.clusters),
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "llm_model": self.llm_model,
            }
        }

    def save_to_file(self, file_path: Path) -> None:
        """保存到 JSON 文件"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_file(cls, file_path: Path) -> "GroupEventClusterIndex":
        """从 JSON 文件加载"""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
```

### 3.3 JSON 存储格式

**文件路径**（Eval 场景）：
```
eval/results/{experiment_name}/event_clusters/conv_{id}.json
```

**示例**：`eval/results/locomo-q30/event_clusters/conv_0.json`

```json
{
  "clusters": {
    "gec_001": {
      "cluster_id": "gec_001",
      "topic": "Caroline 的领养计划",
      "summary": "Caroline 向 Melanie 分享了她计划领养孩子的想法...",
      "members": [
        {
          "unit_id": "mu_003",
          "timestamp": "2023-01-15T10:30:00+08:00",
          "summary": "Caroline 首次提到想要领养孩子的计划"
        },
        {
          "unit_id": "mu_007",
          "timestamp": "2023-03-20T14:00:00+08:00",
          "summary": "讨论领养流程和所需准备"
        },
        {
          "unit_id": "mu_015",
          "timestamp": "2023-05-10T09:30:00+08:00",
          "summary": "Caroline 分享领养申请的进展"
        },
        {
          "unit_id": "mu_023",
          "timestamp": "2023-06-20T14:30:00+08:00",
          "summary": "领养审核通过的好消息"
        }
      ],
      "first_timestamp": "2023-01-15T10:30:00+08:00",
      "last_timestamp": "2023-06-20T14:30:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "updated_at": "2024-01-01T00:00:00+08:00"
    },
    "gec_002": {
      "cluster_id": "gec_002",
      "topic": "Melanie 的读书分享",
      "summary": "Melanie 分享了她最近阅读的书籍...",
      "members": [
        {
          "unit_id": "mu_005",
          "timestamp": "2023-02-10T16:00:00+08:00",
          "summary": "Melanie 分享了《Nothing is Impossible》的读后感"
        },
        {
          "unit_id": "mu_012",
          "timestamp": "2023-07-05T20:00:00+08:00",
          "summary": "讨论童年读过的书和新书推荐"
        }
      ],
      "first_timestamp": "2023-02-10T16:00:00+08:00",
      "last_timestamp": "2023-07-05T20:00:00+08:00",
      "created_at": "2024-01-01T00:00:00+08:00",
      "updated_at": "2024-01-01T00:00:00+08:00"
    }
  },
  "unit_to_cluster": {
    "mu_003": "gec_001",
    "mu_005": "gec_002",
    "mu_007": "gec_001",
    "mu_012": "gec_002",
    "mu_015": "gec_001",
    "mu_023": "gec_001"
  },
  "metadata": {
    "conversation_id": "conv_0",
    "total_units": 45,
    "total_clusters": 12,
    "created_at": "2024-01-01T00:00:00+08:00",
    "updated_at": "2024-01-01T00:00:00+08:00",
    "llm_model": "gpt-4o-mini"
  }
}
```

---

## 4. 配置设计

### 4.1 GroupEventClusterConfig

```python
@dataclass
class GroupEventClusterConfig:
    """
    群体事件聚类配置

    支持通过配置文件或代码指定 LLM 模型和参数
    """

    # === LLM 配置 ===
    llm_provider: str = "openai"
    """
    LLM 提供者
    可选值: "openai", "anthropic", "azure", 等
    默认: "openai"
    """

    llm_model: str = "gpt-4o-mini"
    """
    LLM 模型名称
    建议使用小模型控制成本
    默认: "gpt-4o-mini"
    """

    llm_api_key: Optional[str] = None
    """
    API Key
    如果为 None，从环境变量读取（OPENAI_API_KEY 等）
    """

    llm_base_url: Optional[str] = None
    """
    自定义 API Base URL
    用于代理或私有部署
    """

    llm_temperature: float = 0.0
    """
    LLM 温度参数
    聚类判断建议使用 0.0 保证一致性
    """

    # === 聚类配置 ===
    summary_update_threshold: int = 5
    """
    当 Cluster 成员数达到此值的倍数时，更新 Summary
    例如: 5, 10, 15, ... 时触发更新
    """

    max_clusters_in_prompt: int = 20
    """
    LLM 判断时最多展示的 Cluster 数量
    防止 Prompt 过长
    """

    max_members_per_cluster_in_prompt: int = 3
    """
    LLM 判断时每个 Cluster 展示的最近成员数
    """

    # === 存储配置 ===
    output_dir: Optional[Path] = None
    """
    输出目录
    Eval 场景: results/{experiment_name}/event_clusters/
    """

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupEventClusterConfig":
        """从字典创建配置"""
        return cls(
            llm_provider=data.get("llm_provider", "openai"),
            llm_model=data.get("llm_model", "gpt-4o-mini"),
            llm_api_key=data.get("llm_api_key"),
            llm_base_url=data.get("llm_base_url"),
            llm_temperature=data.get("llm_temperature", 0.0),
            summary_update_threshold=data.get("summary_update_threshold", 5),
            max_clusters_in_prompt=data.get("max_clusters_in_prompt", 20),
            max_members_per_cluster_in_prompt=data.get("max_members_per_cluster_in_prompt", 3),
            output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        )

    @classmethod
    def from_yaml(cls, file_path: Path) -> "GroupEventClusterConfig":
        """从 YAML 文件加载配置"""
        import yaml
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data.get("group_event_cluster", {}))
```

### 4.2 配置文件示例

**YAML 格式**（`config/group_event_cluster.yaml`）：

```yaml
group_event_cluster:
  # LLM 配置
  llm_provider: "openai"
  llm_model: "gpt-4o-mini"
  llm_api_key: null  # 从环境变量读取
  llm_base_url: null
  llm_temperature: 0.0

  # 聚类配置
  summary_update_threshold: 5
  max_clusters_in_prompt: 20
  max_members_per_cluster_in_prompt: 3

  # 存储配置（Eval 场景在代码中指定）
  output_dir: null
```

### 4.3 Eval 中的使用

```python
# eval/adapters/parallax/config.py

class ExperimentConfig:
    # ... 现有配置 ...

    # ===== 群体事件聚类配置 =====
    enable_group_event_cluster: bool = True
    """是否启用群体事件聚类"""

    group_event_cluster_config: dict = field(default_factory=lambda: {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.0,
        "summary_update_threshold": 5,
    })
    """聚类配置，会合并到 GroupEventClusterConfig"""

    # ===== 聚类增强检索配置 =====
    enable_cluster_retrieval: bool = True
    """是否在检索时启用聚类扩展"""

    cluster_expansion_limit: int = 3
    """每个 Cluster 最多扩展的 MemUnit 数量"""

    cluster_expansion_score: float = 0.3
    """扩展文档的分数系数"""
```

---

## 5. 代码结构

### 5.1 目录结构

```
src/memory/group_event_cluster/         # 通用模块，放在 src 中
├── __init__.py                         # 模块导出
├── schema.py                           # 数据结构定义
│   ├── ClusterMember
│   ├── GroupEventCluster
│   └── GroupEventClusterIndex
├── config.py                           # 配置类
│   └── GroupEventClusterConfig
├── clusterer.py                        # 核心聚类器
│   └── GroupEventClusterer
├── storage.py                          # 存储接口
│   ├── ClusterStorage (抽象类)
│   └── JsonClusterStorage
└── prompts.py                          # LLM 提示词模板

eval/adapters/parallax/
├── stage2_5_group_event_clustering.py  # Eval 调用入口
└── config.py                           # 添加聚类配置

eval/results/{experiment_name}/
└── event_clusters/                     # 聚类结果存储
    ├── conv_0.json
    ├── conv_1.json
    └── ...
```

### 5.2 模块导出

```python
# src/memory/group_event_cluster/__init__.py

from .schema import (
    ClusterMember,
    GroupEventCluster,
    GroupEventClusterIndex,
)
from .config import GroupEventClusterConfig
from .clusterer import GroupEventClusterer
from .storage import ClusterStorage, JsonClusterStorage

__all__ = [
    "ClusterMember",
    "GroupEventCluster",
    "GroupEventClusterIndex",
    "GroupEventClusterConfig",
    "GroupEventClusterer",
    "ClusterStorage",
    "JsonClusterStorage",
]
```

---

## 6. LLM 聚类算法

### 6.1 整体流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Group Event Clustering                               │
│                                                                         │
│  输入: MemUnit 列表（已按时间排序）                                       │
│  输出: GroupEventClusterIndex                                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  初始化: GroupEventClusterIndex (空)                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  For each MemUnit (按时间顺序) │
                    └───────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               │
┌─────────────────────────────────────────┐         │
│  Step 1: 提取 MemUnit 摘要               │         │
│  - 从 narrative 提取 1-2 句核心描述      │         │
│  - 识别关键实体（人物、事件、时间）       │         │
└─────────────────────────────────────────┘         │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────┐         │
│  Step 2: 判断是否为第一个 MemUnit        │         │
│  - 如果是 → 直接创建新 Cluster           │         │
│  - 如果不是 → 进入 LLM 判断              │         │
└─────────────────────────────────────────┘         │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────┐         │
│  Step 3: LLM 判断归属                    │         │
│  - 输入: MemUnit 摘要 + 现有 Clusters    │         │
│  - 输出: cluster_id 或 "NEW"             │         │
│  - 如果 "NEW": 同时返回新 topic          │         │
└─────────────────────────────────────────┘         │
                    │                               │
          ┌────────┴────────┐                       │
          ▼                 ▼                       │
┌─────────────────┐ ┌─────────────────┐             │
│  归入现有 Cluster │ │  创建新 Cluster  │             │
│  - 添加成员      │ │  - 生成 topic   │             │
│  - 按时间排序    │ │  - 生成 summary │             │
│  - 更新时间戳    │ │  - 添加到索引   │             │
└─────────────────┘ └─────────────────┘             │
                    │                               │
                    ▼                               │
┌─────────────────────────────────────────┐         │
│  Step 4: 更新 unit_to_cluster 映射       │         │
└─────────────────────────────────────────┘         │
                    │                               │
                    └───────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 5: 可选 - 更新大 Cluster 的 Summary                                │
│  - 当 Cluster 成员数达到阈值（如 5）时                                    │
│  - 重新生成更全面的 Summary                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  输出: 完整的 GroupEventClusterIndex                                     │
│  - 所有 Cluster 的成员列表已按时间排序                                    │
│  - 保存到 JSON 文件                                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 核心类设计

```python
class GroupEventClusterer:
    """
    群体事件聚类器

    职责：
    1. 接收 MemUnit，调用 LLM 判断归属
    2. 管理 GroupEventClusterIndex
    3. 生成摘要和汇总
    4. 维护成员的时间排序
    """

    def __init__(
        self,
        config: GroupEventClusterConfig,
        llm_provider: Optional[LLMProvider] = None,
    ):
        """
        初始化聚类器

        Args:
            config: 聚类配置
            llm_provider: LLM 提供者（如果为 None，根据配置创建）
        """
        self.config = config
        self.llm_provider = llm_provider or self._create_llm_provider()
        self.index: Optional[GroupEventClusterIndex] = None
        self._next_cluster_idx = 0

    async def cluster_memunits(
        self,
        memunit_list: List[Dict[str, Any]],
        conversation_id: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> GroupEventClusterIndex:
        """
        对所有 MemUnit 进行聚类

        Args:
            memunit_list: MemUnit 列表（会自动按时间排序）
            conversation_id: 对话 ID
            progress_callback: 进度回调 (current, total, cluster_id)

        Returns:
            GroupEventClusterIndex
        """
        # 按时间排序
        sorted_memunits = sorted(
            memunit_list,
            key=lambda x: self._parse_timestamp(x.get("timestamp", 0))
        )

        # 初始化索引
        self.index = GroupEventClusterIndex(
            clusters={},
            unit_to_cluster={},
            conversation_id=conversation_id,
            total_units=0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            llm_model=self.config.llm_model,
        )

        # 逐个处理
        for i, memunit in enumerate(sorted_memunits):
            cluster_id = await self._cluster_single_memunit(memunit)

            if progress_callback:
                progress_callback(i + 1, len(sorted_memunits), cluster_id)

        return self.index

    async def _cluster_single_memunit(self, memunit: Dict[str, Any]) -> str:
        """处理单个 MemUnit"""
        unit_id = memunit["unit_id"]
        narrative = memunit.get("narrative", "")
        timestamp = self._parse_timestamp(memunit.get("timestamp"))
        participants = memunit.get("participants", [])

        # Step 1: 生成摘要
        unit_summary = await self._generate_unit_summary(narrative)

        # Step 2: 如果是第一个，创建新 Cluster
        if not self.index.clusters:
            return await self._create_new_cluster(
                unit_id, unit_summary, narrative, timestamp
            )

        # Step 3: LLM 判断
        decision = await self._llm_decide_cluster(
            unit_summary, narrative, timestamp, participants
        )

        # Step 4: 处理决策
        if decision["decision"] == "NEW":
            cluster_id = await self._create_new_cluster(
                unit_id, unit_summary, narrative, timestamp,
                topic=decision.get("new_topic")
            )
        else:
            cluster_id = decision["decision"]
            self._add_to_cluster(cluster_id, unit_id, unit_summary, timestamp)

        # 更新索引
        self.index.unit_to_cluster[unit_id] = cluster_id
        self.index.total_units += 1
        self.index.updated_at = datetime.now()

        return cluster_id

    def _add_to_cluster(
        self,
        cluster_id: str,
        unit_id: str,
        unit_summary: str,
        timestamp: datetime,
    ) -> None:
        """
        添加成员到 Cluster，并保持时间排序
        """
        cluster = self.index.clusters[cluster_id]

        # 创建新成员
        member = ClusterMember(
            unit_id=unit_id,
            timestamp=timestamp,
            summary=unit_summary,
        )

        # 添加并按时间排序
        cluster.members.append(member)
        cluster.members.sort(key=lambda m: m.timestamp)

        # 更新时间范围
        cluster.first_timestamp = cluster.members[0].timestamp
        cluster.last_timestamp = cluster.members[-1].timestamp
        cluster.updated_at = datetime.now()

        # 检查是否需要更新 Summary
        if len(cluster.members) % self.config.summary_update_threshold == 0:
            asyncio.create_task(self._update_cluster_summary(cluster))
```

### 6.3 LLM Prompt 设计

见原文档第 4.2 节，将所有 `EventCluster` 替换为 `GroupEventCluster`。

---

## 7. 检索增强（详细设计）

聚类增强检索是本系统的核心价值所在。本章详细说明如何将聚类结果整合到现有检索流程中。

### 7.1 配置设计

#### 7.1.1 GroupEventClusterRetrievalConfig

```python
@dataclass
class GroupEventClusterRetrievalConfig:
    """
    聚类增强检索配置

    控制聚类结果如何整合到检索流程中
    """

    # === 基础开关 ===
    enable_group_event_cluster_retrieval: bool = True
    """是否启用聚类扩展"""

    # === 扩展策略 ===
    expansion_strategy: str = "insert_after_hit"
    """
    扩展策略，决定扩展文档如何插入结果列表

    可选值：
    - "insert_after_hit": 在每个命中文档后插入其 Cluster 成员（推荐）
    - "append_to_end": 将所有扩展文档追加到结果末尾
    - "merge_by_score": 为扩展文档计算衰减分数，与原结果混合排序
    - "replace_rerank": 扩展后整体重新 Rerank（成本较高）
    - "cluster_rerank": Cluster 级别重排，LLM 智能选择最相关 Clusters（需要额外配置）
    """

    # === 扩展数量控制 ===
    max_expansion_per_hit: int = 2
    """
    每个命中文档最多扩展的 Cluster 成员数

    设计考虑：
    - 太少（1）：可能遗漏重要关联信息
    - 太多（5+）：会稀释原检索结果的质量
    - 推荐值：2-3
    """

    max_total_expansion: int = 10
    """
    单次检索最多扩展的总文档数

    设计考虑：
    - 防止某个大 Cluster 占据过多结果
    - 与 expansion_budget_ratio 配合使用
    """

    expansion_budget_ratio: float = 0.3
    """
    扩展预算比例 = 扩展文档数 / 原检索结果数

    例如：原结果 20 个，budget=0.3，则最多扩展 6 个
    实际扩展数 = min(max_total_expansion, 原结果数 * expansion_budget_ratio)
    """

    # === 时间相邻性偏好 ===
    prefer_time_adjacent: bool = True
    """
    是否优先选择时间相邻的成员

    原理：时间上相邻的 MemUnit 更可能包含因果关系或后续发展

    例如：
    - 命中 mu_007（2023-03-20）
    - Cluster 成员：[mu_003, mu_007, mu_015, mu_023]
    - 优先选择 mu_003（前一个）和 mu_015（后一个）
    """

    time_window_hours: Optional[int] = None
    """
    时间窗口限制（小时）

    如果设置，只扩展时间差在此范围内的成员
    None 表示不限制

    例如：time_window_hours=168（一周）
    """

    # === 分数衰减（用于 merge_by_score 策略）===
    expansion_score_decay: float = 0.7
    """
    扩展文档的分数衰减系数

    扩展文档分数 = 原命中文档分数 × expansion_score_decay

    设计考虑：
    - 太高（0.9+）：扩展文档可能排到原结果前面
    - 太低（0.3-）：扩展文档会沉到底部，失去意义
    - 推荐值：0.6-0.8
    """

    # === 去重控制 ===
    deduplicate_expanded: bool = True
    """
    是否对扩展结果去重

    场景：多个命中文档属于同一 Cluster，会重复扩展相同成员
    """

    # === Rerank 相关 ===
    rerank_after_expansion: bool = False
    """
    扩展后是否重新 Rerank

    注意：会增加 Rerank API 调用成本
    仅在 expansion_strategy="replace_rerank" 时生效
    """

    rerank_top_n_after_expansion: int = 20
    """扩展后 Rerank 返回的 Top N"""
```

#### 7.1.2 在 ExperimentConfig 中的集成

```python
# eval/adapters/parallax/config.py

@dataclass
class ExperimentConfig:
    # ... 现有配置 ...

    # ===== 群体事件聚类配置 =====
    enable_group_event_cluster: bool = True
    """是否启用群体事件聚类"""

    group_event_cluster_config: dict = field(default_factory=lambda: {
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.0,
        "summary_update_threshold": 5,
    })
    """GroupEventClusterConfig 配置"""

    # ===== 聚类增强检索配置 =====
    group_event_cluster_retrieval_config: dict = field(default_factory=lambda: {
        "enable_group_event_cluster_retrieval": True,
        "expansion_strategy": "insert_after_hit",
        "max_expansion_per_hit": 2,
        "max_total_expansion": 10,
        "expansion_budget_ratio": 0.3,
        "prefer_time_adjacent": True,
        "time_window_hours": None,
        "expansion_score_decay": 0.7,
        "deduplicate_expanded": True,
        "rerank_after_expansion": False,
    })
    """GroupEventClusterRetrievalConfig 配置"""
```

### 7.2 扩展策略详解

#### 7.2.1 Strategy 1: insert_after_hit（推荐）

**原理**：在每个命中文档后立即插入其 Cluster 的相关成员，保持局部上下文连贯。

```
原检索结果:
  [1] mu_005 (score=0.92) - "Melanie 分享了《Nothing is Impossible》"
  [2] mu_020 (score=0.78) - "讨论周末计划"
  [3] mu_008 (score=0.71) - "Caroline 提到领养进展"
  ...

聚类扩展后:
  [1] mu_005 (score=0.92) - "Melanie 分享了《Nothing is Impossible》" ← 原命中
  [2] mu_012 (score=0.92×0.7=0.64) - "讨论童年读过的书" ← 扩展：同 Cluster
  [3] mu_020 (score=0.78) - "讨论周末计划" ← 原命中（无同 Cluster 成员）
  [4] mu_008 (score=0.71) - "Caroline 提到领养进展" ← 原命中
  [5] mu_003 (score=0.71×0.7=0.50) - "Caroline 首次提到领养计划" ← 扩展
  [6] mu_015 (score=0.71×0.7=0.50) - "领养申请进展更新" ← 扩展
  ...
```

**优点**：
- 保持语义连贯：相关内容紧邻展示
- 易于理解：用户看到命中文档后立即看到关联内容
- 适合 LLM 生成答案：上下文紧凑

**实现代码**：

```python
def expand_with_insert_after_hit(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: GroupEventClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],  # unit_id -> doc
) -> List[Tuple[dict, float]]:
    """
    insert_after_hit 策略：在每个命中文档后插入 Cluster 成员

    Args:
        original_results: 原检索结果 [(doc, score), ...]
        cluster_index: 聚类索引
        config: 聚类检索配置
        all_docs_map: 所有文档的映射（用于获取扩展文档内容）

    Returns:
        扩展后的结果列表 [(doc, score), ...]
    """
    expanded_results = []
    seen_unit_ids = set()  # 去重
    total_expanded = 0     # 总扩展计数

    # 计算扩展预算
    expansion_budget = min(
        config.max_total_expansion,
        int(len(original_results) * config.expansion_budget_ratio)
    )

    for doc, score in original_results:
        unit_id = doc.get("unit_id")

        # 添加原文档
        if unit_id not in seen_unit_ids:
            expanded_results.append((doc, score))
            seen_unit_ids.add(unit_id)

        # 检查是否还有扩展预算
        if total_expanded >= expansion_budget:
            continue

        # 获取该 MemUnit 所属的 Cluster
        cluster = cluster_index.get_cluster_by_unit(unit_id)
        if not cluster:
            continue

        # 获取 Cluster 的其他成员（已按时间排序）
        expansion_candidates = _select_expansion_candidates(
            cluster=cluster,
            hit_unit_id=unit_id,
            seen_unit_ids=seen_unit_ids,
            config=config,
        )

        # 为每个扩展候选计算衰减分数并添加
        for member in expansion_candidates[:config.max_expansion_per_hit]:
            if total_expanded >= expansion_budget:
                break
            if member.unit_id in seen_unit_ids:
                continue

            expanded_doc = all_docs_map.get(member.unit_id)
            if not expanded_doc:
                continue

            # 计算衰减分数
            expanded_score = score * config.expansion_score_decay

            expanded_results.append((expanded_doc, expanded_score))
            seen_unit_ids.add(member.unit_id)
            total_expanded += 1

    return expanded_results


def _select_expansion_candidates(
    cluster: GroupEventCluster,
    hit_unit_id: str,
    seen_unit_ids: Set[str],
    config: GroupEventClusterRetrievalConfig,
) -> List[ClusterMember]:
    """
    选择扩展候选成员

    策略：
    1. 如果 prefer_time_adjacent=True，优先选择时间相邻的
    2. 否则按时间顺序选择
    """
    # 过滤已见过的
    candidates = [
        m for m in cluster.members
        if m.unit_id != hit_unit_id and m.unit_id not in seen_unit_ids
    ]

    if not candidates:
        return []

    if not config.prefer_time_adjacent:
        # 简单策略：按时间顺序取前 N 个
        return candidates

    # 时间相邻策略：找到命中成员的位置，优先选前后相邻的
    hit_index = None
    for i, m in enumerate(cluster.members):
        if m.unit_id == hit_unit_id:
            hit_index = i
            break

    if hit_index is None:
        return candidates

    # 构建优先级：距离命中位置越近越优先
    # 交替选择前一个和后一个
    result = []
    before_idx = hit_index - 1
    after_idx = hit_index + 1

    while len(result) < len(candidates):
        # 先选后面的（时间上更新的信息）
        if after_idx < len(cluster.members):
            member = cluster.members[after_idx]
            if member.unit_id not in seen_unit_ids and member.unit_id != hit_unit_id:
                result.append(member)
            after_idx += 1

        # 再选前面的
        if before_idx >= 0:
            member = cluster.members[before_idx]
            if member.unit_id not in seen_unit_ids and member.unit_id != hit_unit_id:
                result.append(member)
            before_idx -= 1

        # 两边都没有了
        if after_idx >= len(cluster.members) and before_idx < 0:
            break

    # 应用时间窗口限制（如果配置了）
    if config.time_window_hours:
        hit_member = cluster.members[hit_index]
        hit_time = hit_member.timestamp
        max_delta = timedelta(hours=config.time_window_hours)

        result = [
            m for m in result
            if abs(m.timestamp - hit_time) <= max_delta
        ]

    return result
```

#### 7.2.2 Strategy 2: append_to_end

**原理**：将所有扩展文档追加到原结果末尾，不打乱原排序。

```
原检索结果:
  [1-20] 原检索 Top 20

扩展后:
  [1-20] 原检索 Top 20（保持不变）
  [21-26] 聚类扩展文档（按扩展顺序）
```

**优点**：
- 实现简单
- 不影响原排序
- 适合需要保持原检索结果完整性的场景

**缺点**：
- 扩展文档位置靠后，可能被截断
- 语义不连贯

#### 7.2.3 Strategy 3: merge_by_score

**原理**：为扩展文档计算衰减分数，与原结果统一排序。

```
原检索结果:
  mu_005 (score=0.92)
  mu_020 (score=0.78)
  mu_008 (score=0.71)

扩展候选（带衰减分数）:
  mu_012 (score=0.92×0.7=0.64) - mu_005 的 Cluster 成员
  mu_003 (score=0.71×0.7=0.50) - mu_008 的 Cluster 成员

合并排序后:
  [1] mu_005 (0.92)
  [2] mu_020 (0.78)
  [3] mu_008 (0.71)
  [4] mu_012 (0.64) ← 扩展
  [5] mu_003 (0.50) ← 扩展
```

**优点**：
- 高分扩展文档可以排到前面
- 灵活控制扩展文档的位置

**缺点**：
- 打乱原排序
- 需要调参 expansion_score_decay

#### 7.2.4 Strategy 4: replace_rerank

**原理**：扩展后对整体结果重新 Rerank。

```
流程:
  原检索 Top 20 → 聚类扩展到 30 个 → Rerank 30 个 → 返回 Top 20
```

**优点**：
- 最准确：Reranker 会重新评估相关性
- 扩展文档有机会排到前面

**缺点**：
- 成本高：增加一次 Rerank API 调用
- 延迟增加

#### 7.2.5 Strategy 5: cluster_rerank（Cluster 级别重排）

**原理**：让 LLM 智能选择最相关的 Clusters，返回选中 Clusters 的所有成员（按时间顺序）。

```
流程:
  原检索 Top 20 MemUnits
    → 提取对应的 Clusters（去重）
    → LLM 智能选择 1-N 个最相关 Clusters
    → 返回选中 Clusters 的 MemUnits（按时间顺序）
    → 应用数量限制
    → 生成答案
```

**核心思想**：

不是在 MemUnit 级别做扩展，而是退一步到 Cluster 级别：
- 先找到命中的 Clusters
- 让 LLM 根据查询判断哪些 Clusters 最相关
- LLM 可以灵活选择 1-N 个 Clusters（问题具体时选 1 个，宽泛时选多个）
- 返回完整的时间线上下文

**专用配置**：

```python
# 仅当 expansion_strategy="cluster_rerank" 时生效

cluster_rerank_max_clusters: int = 10
"""
LLM 最多可以选择的 Cluster 数量上限。
LLM 会根据查询复杂度智能决定实际选择的数量：
- 问题很具体时：可能只选 1 个
- 问题涉及多个事件：可能选 2-3 个
- 问题很宽泛/比较性问题：可能选 4+ 个
"""

cluster_rerank_max_members_per_cluster: int = 15
"""
每个 Cluster 最多返回的 MemUnits 数量。
如果 Cluster 成员少于此值，返回全部。
"""

cluster_rerank_total_max_members: int = 30
"""
最终返回的 MemUnits 总数上限。
这是所有选中 Clusters 成员总和的硬上限。
"""
```

**配置层级**：

```
限制层级（从宽到严）：
┌─────────────────────────────────────────────────────────────┐
│ 1. cluster_rerank_max_clusters = 5                          │
│    └─ LLM 最多选 5 个 Cluster                               │
│                                                             │
│ 2. cluster_rerank_max_members_per_cluster = 15              │
│    └─ 每个 Cluster 最多贡献 15 个 MemUnits                  │
│                                                             │
│ 3. cluster_rerank_total_max_members = 30 ← 最终硬上限       │
│    └─ 无论选了几个 Cluster，最终不超过 30 个 MemUnits       │
└─────────────────────────────────────────────────────────────┘
```

**示例场景**：

| 场景 | LLM 选择 | 实际返回 |
|------|---------|---------|
| 问题很具体（"Caroline 什么时候决定领养"） | 1 个 Cluster (12 成员) | 12 个 MemUnits |
| 问题涉及两个事件 | 2 个 Clusters (18 + 8 成员) | 26 个 MemUnits |
| 问题很宽泛 | 3 个 Clusters (15 + 15 + 10 成员) | 30 个 MemUnits (被总上限截断) |

**LLM 选择 Prompt**：

LLM 会看到查询和候选 Clusters 的信息（topic, summary, 成员数, 时间范围），
然后选择最相关的 Clusters 并提供选择理由。

**优点**：
- 智能选择：LLM 根据查询灵活决定需要几个 Clusters
- 完整上下文：返回整个事件的时间线，而非零散片段
- 时间顺序：返回的 MemUnits 按时间排序，保持叙事连贯
- 避免碎片化：不会只返回事件的部分内容

**缺点**：
- 成本：需要调用 LLM 进行 Cluster 选择
- 延迟：增加一次 LLM 调用

**适用场景**：
- 问题涉及事件的完整过程（如 "Caroline 的领养计划是怎么发展的"）
- 需要时间线上下文来回答问题
- 问题范围不确定，需要 LLM 智能判断

**Checkpoint 机制**：

cluster_rerank 策略会生成独立的 checkpoint 文件，保存在 `cluster_selection/` 目录下：

```
experiment_name/
├── search_results.json
├── search_results_checkpoint.json
└── cluster_selection/
    ├── locomo_exp_user_0.json
    ├── locomo_exp_user_1.json
    └── ...
```

每个 checkpoint 文件包含：

```json
{
    "qa_count": 10,
    "questions": [
        {
            "query": "When did Caroline decide to adopt?",
            "clusters_found": ["gec_001", "gec_003", "gec_007"],
            "clusters_selected": ["gec_001"],
            "selection_reasoning": "Only gec_001 contains adoption-related info",
            "cluster_details": {
                "gec_001": {"topic": "Caroline's adoption plan", "member_count": 12}
            },
            "members_per_cluster": {"gec_001": 12},
            "final_count": 12,
            "truncated": false,
            "evidence_cluster_analysis": {
                "unique_evidence_clusters": ["gec_001"],
                "cluster_coverage": 1.0
            }
        }
    ]
}
```

**结果标注**：

search_results.json 中每个问题的结果包含 cluster 信息：

```json
{
    "query": "When did Caroline decide to adopt?",
    "unit_ids": ["mu_001", "mu_003", "mu_007"],
    "unit_cluster_info": [
        {"unit_id": "mu_001", "cluster_id": "gec_001"},
        {"unit_id": "mu_003", "cluster_id": "gec_001"},
        {"unit_id": "mu_007", "cluster_id": "gec_001"}
    ],
    "evidence_cluster_analysis": {
        "evidence_details": [
            {"evidence_id": "mu_001", "cluster_id": "gec_001", "in_results": true}
        ],
        "unique_evidence_clusters": ["gec_001"],
        "clusters_in_results": ["gec_001"],
        "cluster_coverage": 1.0
    }
}
```

### 7.3 整合到检索流程

#### 7.3.1 整体流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Cluster-Enhanced Retrieval                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Step 1: 基础检索 (Hybrid Search / Agentic Retrieval)                    │
│                                                                         │
│  Query: "Melanie 读过哪些书？"                                           │
│  结果: [(mu_005, 0.92), (mu_020, 0.78), (mu_008, 0.71), ...]            │
│                                                                         │
│  ⚙️ 配置: 现有检索配置（retrieval_mode, use_reranker, etc.）             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  enable_group_event_cluster_retrieval?    │
                    └───────────────────────────────┘
                          │              │
                         Yes             No
                          │              │
                          ▼              └──────────────────┐
┌─────────────────────────────────────────────────────────┐ │
│  Step 2: 加载聚类索引                                     │ │
│                                                         │ │
│  cluster_index = load_cluster_index(conversation_id)    │ │
│                                                         │ │
│  如果索引不存在 → 跳过扩展，返回原结果                    │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    ▼                       │
┌─────────────────────────────────────────────────────────┐ │
│  Step 3: 识别命中文档的 Cluster                          │ │
│                                                         │ │
│  hit_clusters = {                                       │ │
│      mu_005 → gec_002 ("Melanie 的读书分享"),            │ │
│      mu_008 → gec_001 ("Caroline 的领养计划"),           │ │
│  }                                                      │ │
│                                                         │ │
│  注意：mu_020 没有命中任何 Cluster（可能是孤立的）        │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    ▼                       │
┌─────────────────────────────────────────────────────────┐ │
│  Step 4: 计算扩展预算                                     │ │
│                                                         │ │
│  原结果数: 20                                            │ │
│  expansion_budget_ratio: 0.3                            │ │
│  max_total_expansion: 10                                │ │
│                                                         │ │
│  实际预算 = min(10, 20 × 0.3) = 6                        │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    ▼                       │
┌─────────────────────────────────────────────────────────┐ │
│  Step 5: 选择扩展候选（按策略）                           │ │
│                                                         │ │
│  expansion_strategy: "insert_after_hit"                 │ │
│  prefer_time_adjacent: True                             │ │
│  max_expansion_per_hit: 2                               │ │
│                                                         │ │
│  gec_002 成员: [mu_005*, mu_012] → 扩展 mu_012          │ │
│  gec_001 成员: [mu_003, mu_007, mu_008*, mu_015, mu_023] │ │
│           → 扩展 mu_015 (后), mu_007 (前)               │ │
│                                                         │ │
│  * 表示命中的成员                                        │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    ▼                       │
┌─────────────────────────────────────────────────────────┐ │
│  Step 6: 执行扩展并整合结果                               │ │
│                                                         │ │
│  扩展后结果（insert_after_hit 策略）:                    │ │
│                                                         │ │
│  [1] mu_005 (0.92) ← 原命中                              │ │
│  [2] mu_012 (0.64) ← 扩展：gec_002 成员                  │ │
│  [3] mu_020 (0.78) ← 原命中（无 Cluster）                │ │
│  [4] mu_008 (0.71) ← 原命中                              │ │
│  [5] mu_015 (0.50) ← 扩展：gec_001 成员（后）            │ │
│  [6] mu_007 (0.50) ← 扩展：gec_001 成员（前）            │ │
│  [7] mu_XXX (0.65) ← 原命中                              │ │
│  ...                                                    │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    ▼                       │
                    ┌───────────────────────────────┐       │
                    │  rerank_after_expansion?      │       │
                    └───────────────────────────────┘       │
                          │              │                  │
                         Yes             No                 │
                          │              │                  │
                          ▼              └──────┐           │
┌─────────────────────────────────────────────┐ │           │
│  Step 7 (可选): 重新 Rerank                  │ │           │
│                                             │ │           │
│  Rerank 扩展后的结果                         │ │           │
│  返回 Top 20                                │ │           │
└─────────────────────────────────────────────┘ │           │
                          │                     │           │
                          ▼                     ▼           │
┌─────────────────────────────────────────────────────────┐ │
│  Step 8: 返回最终结果                                     │ │
│                                                         │ │
│  final_results = [...扩展后的结果...]                    │ │
│  metadata["cluster_expansion"] = {                      │ │
│      "expanded_count": 3,                               │ │
│      "clusters_hit": ["gec_001", "gec_002"],            │ │
│      "strategy": "insert_after_hit",                    │ │
│  }                                                      │ │
└─────────────────────────────────────────────────────────┘ │
                                    │                       │
                                    └───────────────────────┘
                                    │
                                    ▼
                              返回结果给调用方
```

#### 7.3.2 在 stage3_memory_retrieval.py 中的集成点

```python
# eval/adapters/parallax/stage3_memory_retrivel.py

from memory.group_event_cluster import (
    GroupEventClusterIndex,
    GroupEventClusterRetrievalConfig,
    expand_with_cluster,  # 核心扩展函数
)

async def agentic_retrieval(
    query: str,
    config: ExperimentConfig,
    llm_provider: LLMProvider,
    llm_config: dict,
    emb_index,
    bm25,
    docs,
    cluster_index: Optional[GroupEventClusterIndex] = None,  # 🔥 新增参数
    enable_traversal_stats: bool = False,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    Agentic 多轮检索（支持聚类增强）
    """
    # ... 现有检索逻辑 ...

    # 获取原检索结果
    final_results = ...  # 现有逻辑得到的结果

    # ========== 聚类增强扩展 ==========
    group_event_cluster_retrieval_config = GroupEventClusterRetrievalConfig.from_dict(
        config.group_event_cluster_retrieval_config
    )

    if (group_event_cluster_retrieval_config.enable_group_event_cluster_retrieval
        and cluster_index is not None):

        # 构建 all_docs_map（用于获取扩展文档内容）
        all_docs_map = {doc["unit_id"]: doc for doc in docs}

        # 执行聚类扩展
        expanded_results, expansion_metadata = expand_with_cluster(
            original_results=final_results,
            cluster_index=cluster_index,
            config=group_event_cluster_retrieval_config,
            all_docs_map=all_docs_map,
        )

        # 可选：扩展后重新 Rerank
        if group_event_cluster_retrieval_config.rerank_after_expansion:
            expanded_results = await reranker_search(
                query=query,
                results=expanded_results,
                top_n=group_event_cluster_retrieval_config.rerank_top_n_after_expansion,
                # ... 其他参数 ...
            )

        # 更新 metadata
        metadata["cluster_expansion"] = expansion_metadata

        final_results = expanded_results

    return final_results, metadata
```

#### 7.3.3 核心扩展函数接口

```python
# src/memory/group_event_cluster/retrieval.py

def expand_with_cluster(
    original_results: List[Tuple[dict, float]],
    cluster_index: GroupEventClusterIndex,
    config: GroupEventClusterRetrievalConfig,
    all_docs_map: Dict[str, dict],
) -> Tuple[List[Tuple[dict, float]], Dict[str, Any]]:
    """
    根据聚类索引扩展检索结果

    Args:
        original_results: 原检索结果 [(doc, score), ...]
        cluster_index: 聚类索引
        config: 聚类检索配置
        all_docs_map: unit_id -> doc 映射

    Returns:
        (expanded_results, metadata)
        - expanded_results: 扩展后的结果
        - metadata: 扩展统计信息
    """
    if config.expansion_strategy == "insert_after_hit":
        return expand_with_insert_after_hit(...)
    elif config.expansion_strategy == "append_to_end":
        return expand_with_append_to_end(...)
    elif config.expansion_strategy == "merge_by_score":
        return expand_with_merge_by_score(...)
    elif config.expansion_strategy == "replace_rerank":
        # 注意：实际 Rerank 在调用方执行
        return expand_for_rerank(...)
    else:
        raise ValueError(f"Unknown expansion strategy: {config.expansion_strategy}")
```

### 7.4 扩展元数据（用于分析）

```python
expansion_metadata = {
    "enabled": True,
    "strategy": "insert_after_hit",

    # 扩展统计
    "original_count": 20,           # 原结果数
    "expanded_count": 6,            # 扩展文档数
    "final_count": 26,              # 最终结果数

    # Cluster 统计
    "clusters_hit": ["gec_001", "gec_002"],  # 命中的 Cluster
    "clusters_expanded": {
        "gec_001": {
            "hit_unit_ids": ["mu_008"],
            "expanded_unit_ids": ["mu_015", "mu_007"],
        },
        "gec_002": {
            "hit_unit_ids": ["mu_005"],
            "expanded_unit_ids": ["mu_012"],
        },
    },

    # 预算使用
    "expansion_budget": 6,
    "budget_used": 3,

    # 配置快照
    "config": {
        "max_expansion_per_hit": 2,
        "expansion_budget_ratio": 0.3,
        "prefer_time_adjacent": True,
    }
}
```

### 7.5 最佳实践建议

| 场景 | 推荐配置 | 说明 |
|------|---------|------|
| **高精度场景** | `strategy="insert_after_hit"`, `max_expansion_per_hit=2` | 保持原排序，少量扩展 |
| **高召回场景** | `strategy="merge_by_score"`, `budget_ratio=0.5` | 更多扩展，混合排序 |
| **成本敏感** | `rerank_after_expansion=False` | 避免额外 Rerank 调用 |
| **质量优先** | `strategy="replace_rerank"`, `rerank_after_expansion=True` | 全量 Rerank，最准确 |
| **时间相关问答** | `prefer_time_adjacent=True`, `time_window_hours=168` | 优先近期关联 |

---

## 8. 成本估算

（与原文档相同，将 EventCluster 替换为 GroupEventCluster）

---

## 9. 实现计划

### Phase 1: 核心数据结构

- [ ] 创建 `src/memory/group_event_cluster/` 目录
- [ ] 实现 `schema.py`：ClusterMember, GroupEventCluster, GroupEventClusterIndex
- [ ] 实现 `config.py`：GroupEventClusterConfig
- [ ] 实现序列化/反序列化（保持时间排序）
- [ ] 单元测试

### Phase 2: LLM 聚类器

- [ ] 实现 `prompts.py`：LLM 提示词模板
- [ ] 实现 `clusterer.py`：GroupEventClusterer
- [ ] 实现 `storage.py`：JsonClusterStorage
- [ ] 集成测试

### Phase 3: Eval 集成

- [ ] 创建 `eval/adapters/parallax/stage2_5_group_event_clustering.py`
- [ ] 修改 `eval/adapters/parallax/config.py` 添加配置项
- [ ] 修改 `eval/adapters/parallax/stage3_memory_retrivel.py` 添加聚类增强

### Phase 4: 测试与调优

- [ ] 在 LoCoMo 数据集上运行
- [ ] 分析聚类质量
- [ ] 对比 Baseline vs Cluster-Enhanced 准确率
- [ ] 调优参数

---

## 10. 风险与缓解

（与原文档相同）

---

## 11. 附录

### 11.1 与现有 ClusterManager 的对比

| 方面 | 现有 ClusterManager | 新 GroupEventClusterer |
|------|---------------------|------------------------|
| 位置 | `src/memory/cluster_manager/` | `src/memory/group_event_cluster/` |
| 聚类依据 | 向量 cosine similarity | LLM 语义判断 |
| 视角 | 无明确视角 | 群体视角（第三人称） |
| 时间排序 | 无 | 成员按时间排序 |
| 主题名称 | 无 | 有（topic 字段） |
| 汇总描述 | 无 | 有（summary 字段） |
| 用途 | Profile 提取触发 | 检索增强 |

### 11.2 关键设计决策汇总

| 决策 | 选择 | 理由 |
|------|------|------|
| 命名 | GroupEventCluster | 强调群体视角 |
| 代码位置 | `src/memory/` | 通用模块，非 Eval 专用 |
| MemUnit 关联 | 通过 Index 映射 | 保持 MemUnit 纯净 |
| 时间排序 | 成员按时间升序 | 便于理解事件发展 |
| 存储格式 | JSON 文件 | 便于调试和分析 |
| LLM 配置 | 通过配置文件指定 | 灵活切换模型 |
