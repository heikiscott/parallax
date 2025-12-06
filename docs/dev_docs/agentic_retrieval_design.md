# Parallax 可配置检索 Pipeline 设计方案

## 目录

- [1. 概述](#1-概述)
  - [1.1 问题背景](#11-问题背景)
  - [1.2 设计目标](#12-设计目标)
- [2. 现有架构分析](#2-现有架构分析)
  - [2.1 现有 Pipeline 模式梳理](#21-现有-pipeline-模式梳理)
    - [2.1.1 评估 Pipeline](#211-评估-pipeline-evalcorepipelinepy)
    - [2.1.2 策略路由器](#212-策略路由器-evaladaptersparallaxstrategyrouterpy)
    - [2.1.3 Search Stage 细粒度检查点](#213-search-stage-细粒度检查点)
  - [2.2 兼容性问题识别](#22-兼容性问题识别)
  - [2.3 设计调整方案](#23-设计调整方案)
- [3. 核心架构](#3-核心架构)
  - [3.1 扁平可嵌套架构](#31-扁平可嵌套架构)
    - [核心设计原则：一切皆 Component](#核心设计原则一切皆-component)
    - [统一抽象：PipelineComponent](#统一抽象pipelinecomponent)
    - [Pipeline 也是 Component](#pipeline-也是-component)
    - [StrategyRouter 也是 Component](#strategyrouter-也是-component)
    - [使用示例：灵活组合](#使用示例灵活组合)
    - [与现有架构的关系](#与现有架构的关系)
  - [3.2 组件抽象](#32-组件抽象)
    - [组件分类体系](#组件分类体系)
    - [1. Memory Building（记忆构建）](#1-memory-building记忆构建)
    - [2. Question Classification & Routing（问题分类与路由）](#2-question-classification--routing问题分类与路由)
    - [3. Query Preprocessing（查询预处理）](#3-query-preprocessing查询预处理)
    - [4. Retrieval（检索）](#4-retrieval检索)
    - [5. Result Expansion（结果扩展）](#5-result-expansion结果扩展)
    - [6. Retrieval Postprocessing（检索后处理）](#6-retrieval-postprocessing检索后处理)
    - [7. Prompt Adaptation（Prompt 适配）](#7-prompt-adaptationprompt-适配)
    - [8. Judgment（评判）](#8-judgment评判)
    - [9. Answer Generation（答案生成）](#9-answer-generation答案生成)
  - [3.3 Pipeline 执行框架](#33-pipeline-执行框架)
    - [RetrievalPipeline](#retrievalpipeline)
    - [PipelineStage](#pipelinestage)
    - [PipelineContext](#pipelinecontext)
  - [3.4 配置系统](#34-配置系统)
    - [YAML 配置文件](#yaml-配置文件)
    - [ComponentRegistry](#componentregistry)
    - [PipelineConfigLoader](#pipelineconfigloader)
- [4. 配置示例](#4-配置示例)
  - [示例 1：完整的 Agentic RAG Pipeline](#示例-1完整的-agentic-rag-pipeline)
  - [示例 2：记忆构建 Pipeline](#示例-2记忆构建-pipeline)
  - [示例 3：ColBERT 检索 + 压缩](#示例-3colbert-检索--压缩)
  - [示例 4：轻量级检索（快速响应）](#示例-4轻量级检索快速响应)
- [5. 使用方式](#5-使用方式)
  - [方式 1：配置文件驱动](#方式-1配置文件驱动)
  - [方式 2：程序化构建](#方式-2程序化构建)
- [6. 实施路线图](#6-实施路线图)
  - [Phase 1: 基础框架（1-2周）](#phase-1-基础框架1-2周)
  - [Phase 2: 兼容适配（1周）](#phase-2-兼容适配1周)
  - [Phase 3: 新组件实现（2-3周）](#phase-3-新组件实现2-3周)
  - [Phase 4: 代码重构（可选，2周）](#phase-4-代码重构可选2周)
- [7. 关键文件清单](#7-关键文件清单)
  - [新增文件](#新增文件)
  - [修改文件（Phase 4）](#修改文件phase-4)
  - [保持不变（作为 Adapter 使用）](#保持不变作为-adapter-使用)
- [8. 优势总结](#8-优势总结)
  - [8.1 灵活性](#81-灵活性)
  - [8.2 扩展性](#82-扩展性)
  - [8.3 可维护性](#83-可维护性)
  - [8.4 性能](#84-性能)
  - [8.5 向后兼容](#85-向后兼容)
- [9. 风险与缓解](#9-风险与缓解)
  - [风险 1: 性能开销](#风险-1-性能开销)
  - [风险 2: 配置复杂度](#风险-2-配置复杂度)
  - [风险 3: 迁移成本](#风险-3-迁移成本)
- [10. 总结](#10-总结)
- [11. 相关文档](#11-相关文档)
  - [用户文档](#用户文档)
  - [技术文档](#技术文档)
  - [代码仓库](#代码仓库)
- [附录](#附录)
  - [A. 参考资料](#a-参考资料)

---

## 1. 概述

### 1.1 问题背景

当前 Parallax 检索系统存在以下问题：

1. **策略层与实现层耦合**：`strategy/` 依赖 `eval/` 层函数，应该依赖 `src/agents/`
2. **配置分散**：`AgenticConfig`, `ExperimentConfig`, `GECConfig` 参数重复定义
3. **扩展性受限**：添加新检索方法需要修改 3-6 个文件
4. **灵活性不足**：无法通过配置文件灵活组合检索方法（如 ColBERT、QRHead）

### 1.2 设计目标

设计一个**可配置的 Agentic Retrieval Pipeline**，实现：

- ✅ **声明式配置**：通过 YAML/JSON 配置文件定义检索流程
- ✅ **模块化组件**：每个检索步骤（Retriever, Reranker, Expander）独立可插拔
- ✅ **零代码扩展**：添加新检索方法只需实现接口 + 注册，无需修改现有代码
- ✅ **向后兼容**：现有代码可以无缝迁移到新架构

[⬆️ 返回目录](#目录)

---

## 2. 现有架构分析

### 2.1 现有 Pipeline 模式梳理

Parallax 项目中已经存在多个 Pipeline 实现，在设计新的统一架构前，需要先理解现有模式：

#### 2.1.1 评估 Pipeline (`eval/core/pipeline.py`)

**职责**: 五阶段评估工作流编排（Add → Cluster → Search → Answer → Evaluate）

**优势**：
- ✅ **检查点系统**: 阶段级别和对话级别的细粒度检查点，支持断点续跑
- ✅ **异步执行**: 基于 async/await，Search Stage 使用 `asyncio.Semaphore(20)` 控制并发
- ✅ **阶段控制**: 可选择性执行特定阶段（如 `stages=["search", "answer"]`）
- ✅ **结果管理**: ResultSaver 自动保存 JSON 结果

**局限**：
- ❌ **硬编码 5 阶段**: 无法动态添加阶段
- ❌ **函数式调用**: 阶段是函数而非对象，缺乏多态性
- ❌ **配置方式**: 使用 Python 类 (`ExperimentConfig`) 而非声明式配置

#### 2.1.2 策略路由器 (`eval/adapters/parallax/strategy/router.py`)

**职责**: 基于问题分类选择检索策略

**优势**：
- ✅ **问题分类**: 完善的 `QuestionClassifier`，支持多种问题类型（事件时间、活动、属性等）
- ✅ **组件注册**: `register_strategy()` / `unregister_strategy()` 机制
- ✅ **策略覆盖**: `strategy_overrides` 配置支持
- ✅ **元数据传播**: 分类结果自动附加到 `result.metadata`

**架构**：
```python
class StrategyRouter:
    def __init__(self, classifier: QuestionClassifier):
        self._classifier = classifier
        self._strategies: Dict[StrategyType, BaseRetrievalStrategy] = {}

    async def route_and_retrieve(self, query, context):
        classification = self.classify(query)
        strategy = self._strategies[classification.strategy]
        return await strategy.retrieve(query, context)
```

**发现**: StrategyRouter 实际上已经是一个成熟的"问题分类 + 路由"组件，与提议设计中的 `BaseQuestionClassifier` 高度重合。

#### 2.1.3 Search Stage 细粒度检查点

**位置**: `eval/core/stages/search_stage.py`

**核心机制**：
- 对话级别检查点：每个 conversation 处理完成后保存
- 并发控制：`asyncio.Semaphore(20)` 限制同时处理 20 个对话
- 渐进式保存：避免全部完成才保存导致的数据丢失

这是**生产级别的检查点实现**，必须保留并集成到新设计中。

[⬆️ 返回目录](#目录)

### 2.2 兼容性问题识别

对比提议设计与现有架构，发现 **4 个关键差异**：

| 维度 | 现有架构 | 提议设计 | 兼容性 |
|------|---------|---------|--------|
| **检查点** | ✅ 阶段 + 对话级别细粒度检查点 | ❌ 未提及 | **CRITICAL** - 必须集成 |
| **Pipeline 层次** | 单层（5 阶段流程） | 组件化（9 大类） | **需要双层架构** |
| **配置方式** | Python 类 + 部分 override | YAML 声明式 | **需要双轨支持** |
| **问题分类** | ✅ StrategyRouter 已实现 | BaseQuestionClassifier | **可直接复用** |

[⬆️ 返回目录](#目录)

### 2.3 设计调整方案

基于兼容性分析，对原设计进行以下调整：

#### 调整 1：集成 CheckpointManager（必需）

```python
# src/agents/pipeline/pipeline.py

class RetrievalPipeline:
    def __init__(
        self,
        name: str = "default",
        enable_checkpoint: bool = False,  # 新增
        checkpoint_dir: Optional[Path] = None,  # 新增
    ):
        self.checkpoint = CheckpointManager(checkpoint_dir) if enable_checkpoint else None

    async def execute(self, query, context):
        for stage in self.stages:
            # 检查点检查
            if self.checkpoint and self.checkpoint.exists(context.conversation_id, stage.name):
                result = self.checkpoint.load(context.conversation_id, stage.name)
                continue

            # 执行阶段
            result = await stage.execute(query, result, context)

            # 保存检查点
            if self.checkpoint:
                self.checkpoint.save(context.conversation_id, stage.name, result)

        return result
```

#### 调整 2：双层 Pipeline 架构（推荐）

**上层**: Evaluation Pipeline（现有的 5 阶段流程）
**下层**: Retrieval Pipeline（新设计的 9 大类组件）

```
EvaluationPipeline (eval/core/pipeline.py)
├── Add Stage
├── Cluster Stage (可选)
├── Search Stage  ← 内部使用 RetrievalPipeline
├── Answer Stage
└── Evaluate Stage

RetrievalPipeline (src/agents/pipeline/)
├── Question Classification & Routing (StrategyRouter)
├── Query Preprocessing
├── Retrieval
├── Result Expansion
├── Retrieval Postprocessing
├── Prompt Adaptation
├── Judgment
└── Answer Generation
```

**集成方式**：
```python
# eval/core/stages/search_stage.py (重构后)

from src.agents.pipeline import PipelineConfigLoader

async def run_search_stage(config, dataset):
    # 加载 Retrieval Pipeline
    pipeline = PipelineConfigLoader.build_pipeline(
        PipelineConfigLoader.load(config.retrieval_pipeline_config)
    )

    # 对话级别检查点
    async with asyncio.Semaphore(20):
        for conv in dataset.conversations:
            if checkpoint.exists(conv.id, "search"):
                continue

            result = await pipeline.execute(conv.query, context)
            checkpoint.save(conv.id, "search", result)
```

#### 调整 3：StrategyRouter 作为 BaseQuestionClassifier 的实现

**现状**: StrategyRouter 已经完整实现了问题分类 + 路由功能

**调整**: 将 StrategyRouter 定位为 `BaseQuestionClassifier` 的**参考实现**（而非从头重写）

```python
# src/agents/pipeline/components/question_classifier.py

class BaseQuestionClassifier(PipelineComponent):
    """问题分类器基类"""

    @abstractmethod
    async def classify(self, question: str, context: 'PipelineContext') -> QuestionType:
        pass

    @abstractmethod
    async def route(self, question_type: QuestionType, context: 'PipelineContext') -> str:
        pass

# eval/adapters/parallax/strategy/router.py (现有代码，标记为 BaseQuestionClassifier 实现)

class StrategyRouter(BaseQuestionClassifier):
    """问题分类 + 策略路由的生产实现"""
    # 现有代码保持不变
```

#### 调整 4：配置双轨支持（推荐）

同时支持 Python 类配置（向后兼容）和 YAML 配置（新功能）：

```python
# 方式 1: Python 类配置（现有方式）
config = ExperimentConfig(
    adapter="parallax",
    retrieval_strategy="gec_cluster_rerank",
    enable_clustering=True,
)

# 方式 2: YAML 配置（新方式）
config = ExperimentConfig.from_yaml("config/experiments/exp001.yaml")
config.retrieval_pipeline_config = "config/pipelines/agentic_hybrid.yaml"
```

[⬆️ 返回目录](#目录)

---

## 3. 核心架构

### 3.1 扁平可嵌套架构

#### 核心设计原则：一切皆 Component

本 Pipeline 设计采用**扁平可嵌套架构**，核心思想是：**一切皆 Component，Pipeline 也是 Component**。

这种设计带来极致的灵活性：
- ✅ 组件可以任意嵌套（Component 包含 Component）
- ✅ Pipeline 可以嵌套 Pipeline（Pipeline 本身也是 Component）
- ✅ 无需预设层次（简单场景扁平化，复杂场景自由嵌套）

#### 统一抽象：PipelineComponent

所有组件（包括 Pipeline 自身）都实现同一个接口：

```python
# src/agents/pipeline/components/base.py

from abc import ABC, abstractmethod
from typing import Optional

class PipelineComponent(ABC):
    """所有组件的基类 - 无论简单还是复合"""

    @abstractmethod
    async def process(
        self,
        query: str,
        context: 'PipelineContext'
    ) -> 'RetrievalResult':
        """处理输入，返回结果"""
        pass

    def is_composite(self) -> bool:
        """是否是复合组件（包含子组件）"""
        return False
```

#### Pipeline 也是 Component

Pipeline 本身实现 `PipelineComponent` 接口，因此可以作为组件嵌入到其他 Pipeline 中：

```python
# src/agents/pipeline/pipeline.py

class RetrievalPipeline(PipelineComponent):
    """Pipeline 也实现 Component 接口，可以嵌套"""

    def __init__(self, name: str, stages: List['PipelineStage'] = None):
        self.name = name
        self.stages = stages or []

    async def process(
        self,
        query: str,
        context: 'PipelineContext'
    ) -> 'RetrievalResult':
        """顺序执行各个 stage"""
        result = RetrievalResult(documents=[], metadata={})

        for stage in self.stages:
            # 每个 stage 的 component 也实现 process() 接口
            result = await stage.component.process(query, context)

            # 可以在这里做结果传递、条件判断等
            if stage.condition and not await stage.condition(result, context):
                continue

        return result

    def add_stage(self, component: PipelineComponent, name: str = None):
        """添加一个阶段"""
        self.stages.append(PipelineStage(component, name))
        return self

    def is_composite(self) -> bool:
        return True
```

#### StrategyRouter 也是 Component

StrategyRouter 是一个特殊的 Component，内部包含多个子 Component（策略），可以嵌入到任何 Pipeline 中：

```python
# eval/adapters/parallax/strategy/router.py (适配为新架构)

class StrategyRouter(PipelineComponent):
    """路由组件 - 根据分类选择子 Component 执行

    位置: eval/adapters/parallax/strategy/router.py
    类型: Question Classification & Routing 大类的一个实现
    """

    def __init__(
        self,
        classifier: 'QuestionClassifier',
        strategies: Dict[str, PipelineComponent]
    ):
        self._classifier = classifier
        self._strategies = strategies  # 每个策略也是 Component

    async def process(
        self,
        query: str,
        context: 'PipelineContext'
    ) -> 'RetrievalResult':
        # 1. 分类问题
        classification = self._classifier.classify(query)

        # 2. 选择策略（策略本身也是 Component）
        strategy = self._strategies[classification.strategy]

        # 3. 执行策略的 process() 方法
        result = await strategy.process(query, context)

        # 4. 添加分类元数据
        result.metadata["classification"] = {
            "question_type": classification.question_type,
            "confidence": classification.confidence,
        }

        return result

    def is_composite(self) -> bool:
        return True  # 因为内部包含多个 strategy
```

#### 使用示例：灵活组合

##### 场景 1: 简单 Pipeline（无嵌套）

```python
# 直接组合 Components
simple_pipeline = RetrievalPipeline("simple")
simple_pipeline.add_stage(HybridRetriever(top_k=50))
simple_pipeline.add_stage(DeepInfraReranker(top_k=20))
simple_pipeline.add_stage(ClusterExpander())

# 执行
result = await simple_pipeline.process(query, context)
```

##### 场景 2: Pipeline 嵌套 Pipeline

```python
# 创建子 Pipeline
round1_pipeline = RetrievalPipeline("round1")
round1_pipeline.add_stage(HybridRetriever())
round1_pipeline.add_stage(Reranker())

# 主 Pipeline 嵌套子 Pipeline
agentic_pipeline = RetrievalPipeline("agentic")
agentic_pipeline.add_stage(round1_pipeline)  # 嵌套：Pipeline 作为 Component
agentic_pipeline.add_stage(SufficiencyChecker())
agentic_pipeline.add_stage(
    MultiQueryExpander(),
    condition=lambda result, ctx: not result.metadata.get("sufficient")
)
agentic_pipeline.add_stage(round1_pipeline)  # 复用同一个 Pipeline

# 执行
result = await agentic_pipeline.process(query, context)
```

##### 场景 3: 带 Router 的 Pipeline

```python
# 定义各个策略（每个策略是一个 Pipeline）
simple_strategy = RetrievalPipeline("simple")
simple_strategy.add_stage(EmbeddingRetriever())
simple_strategy.add_stage(Reranker())

complex_strategy = RetrievalPipeline("complex")
complex_strategy.add_stage(HybridRetriever())
complex_strategy.add_stage(MultiQueryExpander())
complex_strategy.add_stage(Reranker())

# 创建 Router（Router 也是 Component）
router = StrategyRouter(
    classifier=QuestionClassifier(),
    strategies={
        "SIMPLE": simple_strategy,
        "COMPLEX": complex_strategy,
    }
)

# 主 Pipeline 使用 Router
main_pipeline = RetrievalPipeline("with_router")
main_pipeline.add_stage(router)  # Stage 1: 路由 + 执行对应策略
main_pipeline.add_stage(FinalDeduplicator())  # Stage 2: 通用后处理

# 执行
result = await main_pipeline.process(query, context)
```

#### 与现有架构的关系

虽然不预设固定层次，但实际使用中会自然形成层次关系：

```
现有评估流程 (eval/core/pipeline.py)
├── Add Memory Stage
├── Cluster Stage
├── Search Stage  ← 这里可以使用 RetrievalPipeline
│   └── RetrievalPipeline
│       ├── StrategyRouter (可选)
│       │   ├── Strategy A (也是 Pipeline)
│       │   └── Strategy B (也是 Pipeline)
│       ├── 或者直接是 Components
│       └── FinalPostprocessing
├── Answer Stage
└── Evaluate Stage
```

**关键点**：
- 现有的 `eval/core/pipeline.py` 保持不变（评估流程编排）
- 新的 `RetrievalPipeline` 在 Search Stage 内部使用
- StrategyRouter 是可选的，不需要路由时直接用 Pipeline 组合 Components

**优势**：
- **极致灵活**: 任意嵌套深度，无限制
- **简单一致**: 所有东西都是 Component，理解成本低
- **易于扩展**: 添加新组件只需实现 `PipelineComponent` 接口
- **易于测试**: 每个 Component 独立测试，组合后也可以测试

[⬆️ 返回目录](#目录)

---

### 3.2 组件抽象

Pipeline 组件按照 RAG 流程分为 9 大类，覆盖从记忆构建到答案评判的完整生命周期。

#### 组件分类体系

```
Pipeline Components
├── 1. Memory Building (记忆构建)
│   ├── BaseMemoryBuilder - 记忆构建器基类
│   │   ├── ContentExtractor (内容抽取)
│   │   ├── MemoryEncoder (记忆编码)
│   │   └── MemoryOrganizer (记忆组织)
│
├── 2. Question Classification & Routing (问题分类与路由)
│   ├── BaseQuestionClassifier - 问题分类器基类
│   │   ├── ComplexityClassifier (复杂度分类)
│   │   ├── DomainClassifier (领域分类)
│   │   ├── IntentClassifier (意图分类)
│   │   └── StrategyRouter (策略路由器)
│
├── 3. Query Preprocessing (查询预处理)
│   ├── BaseQueryPreprocessor - 查询预处理基类
│   │   ├── QueryRewriter (查询改写)
│   │   ├── QueryDecomposer (查询分解)
│   │   └── QueryExpander (查询扩展)
│
├── 4. Retrieval (检索)
│   ├── BaseRetriever - 检索器基类
│   │   ├── EmbeddingRetriever
│   │   ├── BM25Retriever
│   │   ├── HybridRetriever
│   │   ├── ColBERTRetriever
│   │   └── QRHeadRetriever
│
├── 5. Result Expansion (结果扩展)
│   ├── BaseExpander - 扩展器基类
│   │   ├── ClusterExpander
│   │   ├── MultiQueryExpander
│   │   └── GraphExpander
│
├── 6. Retrieval Postprocessing (检索后处理)
│   ├── BaseReranker (重排序)
│   ├── BaseCompressor (压缩)
│   ├── BaseDeduplicator (去重)
│   └── BaseHighlighter (高亮)
│
├── 7. Prompt Adaptation (Prompt 适配)
│   ├── BasePromptAdapter - Prompt 适配器基类
│   │   ├── TemplateSelector (模板选择器)
│   │   ├── ContextInjector (上下文注入器)
│   │   └── PromptComposer (Prompt 组合器)
│
├── 8. Judgment (评判)
│   ├── BaseJudge - 评判器基类
│   │   ├── RetrievalJudge (检索质量评判)
│   │   └── AnswerJudge (答案质量评判)
│
└── 9. Answer Generation (答案生成)
    └── BaseGenerator - 生成器基类
        ├── DirectGenerator
        ├── IterativeGenerator
        └── ChainOfThoughtGenerator
```

---

#### 1. Memory Building（记忆构建）

将原始输入转换为可存储和检索的记忆单元。

##### BaseMemoryBuilder

```python
class BaseMemoryBuilder(PipelineComponent):
    """记忆构建器基类"""

    @abstractmethod
    async def build(self, raw_content: str, context: 'PipelineContext') -> List[Memory]:
        """将原始内容构建为记忆单元"""
        pass
```

**子类型**：

- **ContentExtractor（内容抽取器）**：从原始内容中提取关键信息
  - `SummaryExtractor`：生成摘要
  - `EntityExtractor`：提取实体和关系
  - `EventExtractor`：提取事件时间线
  - `KeyPointExtractor`：提取关键观点

- **MemoryEncoder（记忆编码器）**：将提取的内容编码为向量或结构化表示
  - `EmbeddingMemoryEncoder`：生成向量表示
  - `StructuredMemoryEncoder`：生成结构化记忆（JSON/图结构）
  - `MultiModalMemoryEncoder`：多模态记忆编码（文本+图像）

- **MemoryOrganizer（记忆组织器）**：组织记忆的层次结构和关联
  - `ClusterMemoryOrganizer`：按主题聚类
  - `TemporalMemoryOrganizer`：按时间线组织
  - `HierarchicalMemoryOrganizer`：构建层次化记忆图
  - `GraphMemoryOrganizer`：构建知识图谱

---

#### 2. Question Classification & Routing（问题分类与路由）

对问题进行分类并路由到合适的处理策略。这是 Pipeline 的**决策中枢**。

##### BaseQuestionClassifier

```python
class BaseQuestionClassifier(PipelineComponent):
    """问题分类器基类"""

    @abstractmethod
    async def classify(self, question: str, context: 'PipelineContext') -> QuestionType:
        """分类问题，返回问题类型"""
        pass

    @abstractmethod
    async def route(self, question_type: QuestionType, context: 'PipelineContext') -> str:
        """根据问题类型路由到 Pipeline 配置"""
        pass
```

**子类型**：

- **ComplexityClassifier（复杂度分类器）**：判断问题复杂度
  - `SimpleQuestionClassifier`：简单事实查询 → 使用 Lightweight Pipeline
  - `ComplexQuestionClassifier`：复杂推理问题 → 使用 Agentic Pipeline
  - `MultiHopClassifier`：多跳问题 → 使用 Multi-Query Expansion

- **DomainClassifier（领域分类器）**：判断问题所属领域
  - `TechnicalDomainClassifier`：技术问题 → 使用专业术语 Prompt
  - `GeneralDomainClassifier`：通用问题 → 使用通用 Prompt
  - `MultiDomainClassifier`：跨领域问题 → 使用多领域检索

- **IntentClassifier（意图分类器）**：判断用户意图
  - `FactualIntentClassifier`：事实查询 → 直接检索
  - `AnalyticalIntentClassifier`：分析类问题 → 需要推理
  - `ComparativeIntentClassifier`：比较类问题 → 需要多文档对比

- **StrategyRouter（策略路由器）**：根据分类结果选择 Pipeline 策略
  - `RuleBasedRouter`：基于规则的路由
  - `LLMBasedRouter`：基于 LLM 的智能路由
  - `HybridRouter`：规则 + LLM 混合路由

**使用示例**：

```python
class LLMComplexityClassifier(ComplexityClassifier):
    async def classify(self, question: str, context: 'PipelineContext') -> QuestionType:
        prompt = f"Classify the complexity of this question: {question}\nOptions: simple, medium, complex"
        response = await context.llm_provider.generate(prompt)
        return self._parse_complexity(response)

    async def route(self, question_type: QuestionType, context: 'PipelineContext') -> str:
        if question_type.complexity == "simple":
            return "config/pipelines/lightweight.yaml"
        elif question_type.complexity == "medium":
            return "config/pipelines/hybrid_retrieval.yaml"
        else:
            return "config/pipelines/full_agentic_rag.yaml"
```

---

#### 3. Query Preprocessing（查询预处理）

在检索前对查询进行优化和转换。

##### BaseQueryPreprocessor

```python
class BaseQueryPreprocessor(PipelineComponent):
    """查询预处理基类"""

    @abstractmethod
    async def preprocess(self, query: str, context: 'PipelineContext') -> Union[str, List[str]]:
        """预处理查询，返回单个查询或多个查询"""
        pass
```

**子类型**：

- **QueryRewriter（查询改写器）**：改写查询以提升检索效果
  - `LLMQueryRewriter`：使用 LLM 改写查询
  - `ExpansionQueryRewriter`：添加同义词扩展
  - `ClarificationQueryRewriter`：澄清模糊查询

- **QueryDecomposer（查询分解器）**：将复杂查询分解为子问题
  - `SubQuestionDecomposer`：分解为子问题序列
  - `AspectDecomposer`：分解为多个方面
  - `TemporalDecomposer`：分解为时间段

- **QueryExpander（查询扩展器）**：扩展查询覆盖范围
  - `PRFExpander`：伪相关反馈扩展
  - `KnowledgeExpander`：基于知识库扩展
  - `SynonymExpander`：同义词扩展

---

#### 3. Retrieval（检索）

从索引中检索候选文档。

##### BaseRetriever

```python
class BaseRetriever(PipelineComponent):
    """检索器基类"""

    @abstractmethod
    async def search(self, query: str, top_k: int, context: 'PipelineContext') -> List[Document]:
        """执行检索，返回 top_k 文档"""
        pass
```

**具体实现**：
- `EmbeddingRetriever`：基于向量 Embedding 的检索
- `BM25Retriever`：基于 BM25 的关键词检索
- `HybridRetriever`：混合检索（Embedding + BM25 + RRF）
- `ColBERTRetriever`：ColBERT Late Interaction 检索
- `QRHeadRetriever`：QRHead 注意力分数检索

---

#### 4. Result Expansion（结果扩展）

扩展检索结果以提升覆盖度。

##### BaseExpander

```python
class BaseExpander(PipelineComponent):
    """结果扩展器基类"""

    @abstractmethod
    async def expand(self, query: str, documents: List[Document], context: 'PipelineContext') -> List[Document]:
        """扩展检索结果"""
        pass
```

**具体实现**：
- `ClusterExpander`：基于聚类的邻域扩展
- `MultiQueryExpander`：生成多个查询并行检索，融合结果
- `GraphExpander`：基于知识图谱扩展相关文档
- `TemporalExpander`：基于时间关系扩展

---

#### 5. Retrieval Postprocessing（检索后处理）

对检索结果进行精炼和优化。

##### BaseReranker（重排序器）

```python
class BaseReranker(PipelineComponent):
    """重排序器基类"""

    @abstractmethod
    async def rerank(self, query: str, documents: List[Document], top_k: int, context: 'PipelineContext') -> List[Document]:
        """重排序文档"""
        pass
```

**具体实现**：
- `DeepInfraReranker`：使用 DeepInfra Rerank API
- `ColBERTReranker`：使用 ColBERT 模型精细打分
- `CrossEncoderReranker`：使用 Cross-Encoder 模型

##### BaseCompressor（压缩器）

```python
class BaseCompressor(PipelineComponent):
    """压缩器基类"""

    @abstractmethod
    async def compress(self, query: str, documents: List[Document], context: 'PipelineContext') -> List[Document]:
        """压缩文档，移除无关内容"""
        pass
```

**具体实现**：
- `LLMCompressor`：使用 LLM 压缩无关内容
- `ExtractiveSummaryCompressor`：抽取式摘要压缩
- `SentenceFilterCompressor`：句子级别过滤

##### BaseDeduplicator（去重器）

```python
class BaseDeduplicator(PipelineComponent):
    """去重器基类"""

    @abstractmethod
    async def deduplicate(self, documents: List[Document], context: 'PipelineContext') -> List[Document]:
        """去除重复文档"""
        pass
```

**具体实现**：
- `SemanticDeduplicator`：语义去重
- `ExactDeduplicator`：精确去重
- `FuzzyDeduplicator`：模糊去重

##### BaseHighlighter（高亮器）

```python
class BaseHighlighter(PipelineComponent):
    """高亮器基类"""

    @abstractmethod
    async def highlight(self, query: str, documents: List[Document], context: 'PipelineContext') -> List[Document]:
        """高亮相关片段"""
        pass
```

**具体实现**：
- `KeywordHighlighter`：关键词高亮
- `RelevanceHighlighter`：相关片段高亮
- `EntityHighlighter`：实体高亮

---

#### 7. Prompt Adaptation（Prompt 适配）

**Prompt 元编程核心**：根据问题类型、领域、检索结果动态选择和组合 Prompt 模板。

##### BasePromptAdapter

```python
class BasePromptAdapter(PipelineComponent):
    """Prompt 适配器基类"""

    @abstractmethod
    async def adapt(
        self,
        query: str,
        documents: List[Document],
        question_type: QuestionType,
        context: 'PipelineContext'
    ) -> str:
        """根据上下文适配 Prompt"""
        pass
```

**子类型**：

- **TemplateSelector（模板选择器）**：根据问题类型选择 Prompt 模板
  - `QuestionTypeTemplateSelector`：
    - 事实查询 → 使用 "Based on the following documents, answer: {question}"
    - 分析类问题 → 使用 "Analyze the following information and provide insights: {question}"
    - 比较类问题 → 使用 "Compare the following aspects: {question}"
  - `DomainTemplateSelector`：
    - 技术领域 → 使用技术术语模板
    - 通用领域 → 使用通俗语言模板
  - `ComplexityTemplateSelector`：
    - 简单问题 → 使用简洁模板
    - 复杂问题 → 使用详细推理模板（Chain-of-Thought）

- **ContextInjector（上下文注入器）**：将检索结果注入 Prompt
  - `DocumentContextInjector`：格式化文档并注入
  - `MetadataContextInjector`：注入文档元数据（来源、时间等）
  - `RelevanceScoreInjector`：注入相关性分数
  - `HighlightContextInjector`：注入高亮片段

- **PromptComposer（Prompt 组合器）**：组合多个 Prompt 片段
  - `SequentialComposer`：按顺序组合（System Prompt + Context + Question）
  - `ConditionalComposer`：根据条件组合不同片段
  - `TemplateInterpolator`：模板变量插值

**使用示例**：

```python
class AdaptivePromptAdapter(BasePromptAdapter):
    def __init__(self):
        self.template_selector = QuestionTypeTemplateSelector()
        self.context_injector = DocumentContextInjector()
        self.composer = SequentialComposer()

    async def adapt(self, query, documents, question_type, context):
        # 1. 选择模板
        base_template = await self.template_selector.select(question_type)

        # 2. 注入文档上下文
        doc_context = await self.context_injector.inject(documents, max_tokens=2000)

        # 3. 组合 Prompt
        final_prompt = await self.composer.compose([
            "You are a helpful AI assistant.",
            doc_context,
            base_template.format(question=query)
        ])

        return final_prompt
```

**Prompt 模板仓库示例**：

```python
# src/agents/pipeline/prompts/templates.py

PROMPT_TEMPLATES = {
    "factual_simple": """Based on the following documents, provide a concise answer to the question.

Documents:
{documents}

Question: {question}

Answer:""",

    "analytical_complex": """Analyze the following information carefully and provide detailed insights.

Context:
{documents}

Question: {question}

Please think step by step:
1. Identify key information
2. Analyze relationships
3. Draw conclusions

Analysis:""",

    "comparative": """Compare the following aspects based on the provided documents.

Documents:
{documents}

Question: {question}

Comparison Table:
| Aspect | Option A | Option B |
|--------|----------|----------|

Conclusion:""",
}
```

---

#### 8. Judgment（评判）

评判检索和生成质量。

##### BaseJudge

```python
class BaseJudge(PipelineComponent):
    """评判器基类"""

    @abstractmethod
    async def judge(self, query: str, context: 'PipelineContext') -> Dict[str, Any]:
        """评判质量，返回评判结果"""
        pass
```

**子类型**：

- **RetrievalJudge（检索质量评判）**：
  - `SufficiencyJudge`：判断检索结果是否充分
  - `RelevanceJudge`：判断检索结果相关性
  - `CoverageJudge`：判断是否覆盖查询的所有方面
  - `DiversityJudge`：判断结果多样性

- **AnswerJudge（答案质量评判）**：
  - `FactualityJudge`：判断答案事实性
  - `CompletenessJudge`：判断答案完整性
  - `GroundingJudge`：判断答案是否有检索结果支撑
  - `HallucinationJudge`：检测幻觉

---

#### 9. Answer Generation（答案生成）

基于检索结果生成答案。

##### BaseGenerator

```python
class BaseGenerator(PipelineComponent):
    """生成器基类"""

    @abstractmethod
    async def generate(self, query: str, documents: List[Document], context: 'PipelineContext') -> str:
        """生成答案"""
        pass
```

**具体实现**：
- `DirectGenerator`：直接生成答案
- `IterativeGenerator`：迭代式生成（多轮检索-生成）
- `ChainOfThoughtGenerator`：思维链生成
- `StructuredGenerator`：结构化答案生成

[⬆️ 返回目录](#目录)

### 3.3 Pipeline 执行框架

#### RetrievalPipeline

组织多个阶段的执行流程：

```python
class RetrievalPipeline:
    """可配置的检索 Pipeline"""

    def __init__(self, name: str = "default"):
        self.name = name
        self.stages: List[PipelineStage] = []

    def add_stage(self, stage: 'PipelineStage') -> 'RetrievalPipeline':
        """添加一个 Pipeline 阶段"""
        self.stages.append(stage)
        return self

    async def execute(self, query: str, context: PipelineContext) -> RetrievalResult:
        """执行完整的 Pipeline"""
        current_result = RetrievalResult(documents=[], metadata={})

        for i, stage in enumerate(self.stages):
            stage_name = stage.name or f"stage_{i}"
            current_result = await stage.execute(query, current_result, context)
            current_result.metadata[f"{stage_name}_count"] = len(current_result.documents)

        return current_result
```

#### PipelineStage

单个阶段，支持条件执行：

```python
class PipelineStage:
    """Pipeline 的一个阶段"""

    def __init__(self, component: PipelineComponent, name: Optional[str] = None, condition: Optional[callable] = None):
        self.component = component
        self.name = name or component.__class__.__name__
        self.condition = condition  # 条件执行

    async def execute(self, query: str, prev_result: RetrievalResult, context: PipelineContext) -> RetrievalResult:
        """执行当前阶段"""
        if self.condition and not await self.condition(query, prev_result, context):
            return prev_result  # 跳过

        result = await self.component.process(query, context)
        result.metadata.update(prev_result.metadata)
        return result
```

#### PipelineContext

执行上下文，携带所有必要的资源：

```python
@dataclass
class PipelineContext:
    """Pipeline 执行上下文"""
    memory_index: Any  # MemoryIndex
    cluster_index: Optional[Any] = None
    vectorize_service: Optional[Any] = None
    rerank_service: Optional[Any] = None
    llm_provider: Optional[Any] = None
    config: Dict[str, Any] = None
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
```

### 3.4 配置系统

#### YAML 配置文件

通过 YAML 声明式定义 Pipeline：

```yaml
pipeline:
  name: "agentic_hybrid"
  stages:
    - name: "hybrid_search"
      component:
        type: "HybridRetriever"
        config: {top_k: 50}
    - name: "rerank"
      component:
        type: "DeepInfraReranker"
        config: {top_k: 20}
```

#### ComponentRegistry

组件注册表，实现零代码扩展：

```python
class ComponentRegistry:
    """组件注册表"""
    _registry: Dict[str, Type[PipelineComponent]] = {}

    @classmethod
    def register(cls, name: str, component_class: Type[PipelineComponent]):
        """注册组件"""
        cls._registry[name] = component_class

    @classmethod
    def create(cls, name: str, config: Dict) -> PipelineComponent:
        """创建组件实例"""
        component_class = cls._registry[name]
        return component_class(**config)
```

#### PipelineConfigLoader

加载配置并构建 Pipeline：

```python
class PipelineConfigLoader:
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """从 YAML 文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def build_pipeline(config: Dict[str, Any]) -> RetrievalPipeline:
        """根据配置构建 Pipeline"""
        pipeline = RetrievalPipeline(name=config['pipeline']['name'])

        for stage_config in config['pipeline']['stages']:
            component = ComponentRegistry.create(
                stage_config['component']['type'],
                stage_config['component'].get('config', {})
            )
            stage = PipelineStage(component=component, name=stage_config['name'])
            pipeline.add_stage(stage)

        return pipeline
```

[⬆️ 返回目录](#目录)

---

## 4. 配置示例

### 示例 1：完整的 Agentic RAG Pipeline

包含从查询预处理到答案评判的完整流程。

```yaml
# config/pipelines/full_agentic_rag.yaml

pipeline:
  name: "full_agentic_rag"

  stages:
    # Stage 1: 查询预处理
    - name: "query_classification"
      component:
        type: "ComplexityClassifier"
        config: {model: "gpt-4"}

    - name: "query_decomposition"
      component:
        type: "SubQuestionDecomposer"
        config: {max_subquestions: 3}
      condition: "is_complex_query"

    # Stage 2: 检索
    - name: "hybrid_retrieval"
      component:
        type: "HybridRetriever"
        config:
          emb_top_k: 50
          bm25_top_k: 50
          fusion_mode: "rrf"

    # Stage 3: 结果扩展
    - name: "cluster_expansion"
      component:
        type: "ClusterExpander"
        config:
          expansion_strategy: "insert_after_hit"
          max_expansion_per_hit: 3

    # Stage 4: 重排序
    - name: "rerank"
      component:
        type: "DeepInfraReranker"
        config: {top_k: 20}

    # Stage 5: 后处理
    - name: "deduplication"
      component:
        type: "SemanticDeduplicator"
        config: {threshold: 0.9}

    - name: "compression"
      component:
        type: "LLMCompressor"
        config: {max_tokens: 2000}

    # Stage 6: 检索质量评判
    - name: "sufficiency_check"
      component:
        type: "SufficiencyJudge"
        config: {top_n: 5}

    # Stage 7: 多轮检索（如果不充分）
    - name: "multi_query_expansion"
      component:
        type: "MultiQueryExpander"
        config: {num_queries: 3, top_k: 50}
      condition: "not_sufficient"

    - name: "final_rerank"
      component:
        type: "DeepInfraReranker"
        config: {top_k: 20}
      condition: "multi_query_executed"

    # Stage 8: 答案生成
    - name: "answer_generation"
      component:
        type: "DirectGenerator"
        config: {model: "gpt-4"}

    # Stage 9: 答案质量评判
    - name: "grounding_check"
      component:
        type: "GroundingJudge"
        config: {}

global_config:
  vectorize_service: "deep_infra"
  rerank_service: "deep_infra"
  llm_provider: "openai_gpt4"
```

### 示例 2：记忆构建 Pipeline

将原始对话内容转换为结构化记忆。

```yaml
# config/pipelines/memory_building.yaml

pipeline:
  name: "memory_building"

  stages:
    # Stage 1: 内容抽取
    - name: "summary_extraction"
      component:
        type: "SummaryExtractor"
        config: {model: "gpt-4", max_length: 200}

    - name: "entity_extraction"
      component:
        type: "EntityExtractor"
        config: {model: "gpt-4"}

    - name: "keypoint_extraction"
      component:
        type: "KeyPointExtractor"
        config: {model: "gpt-4", max_points: 5}

    # Stage 2: 记忆编码
    - name: "embedding_encoding"
      component:
        type: "EmbeddingMemoryEncoder"
        config: {embedding_service: "deep_infra"}

    - name: "structured_encoding"
      component:
        type: "StructuredMemoryEncoder"
        config: {format: "json"}

    # Stage 3: 记忆组织
    - name: "cluster_organization"
      component:
        type: "ClusterMemoryOrganizer"
        config: {num_clusters: 10, update_existing: true}

    - name: "temporal_organization"
      component:
        type: "TemporalMemoryOrganizer"
        config: {time_window: "1d"}
```

### 示例 3：ColBERT 检索 + 压缩

高性能检索配置。

```yaml
# config/pipelines/colbert_retrieval.yaml

pipeline:
  name: "colbert_retrieval"

  stages:
    # Stage 1: 查询改写
    - name: "query_rewrite"
      component:
        type: "LLMQueryRewriter"
        config: {model: "gpt-4"}

    # Stage 2: ColBERT 检索
    - name: "colbert_search"
      component:
        type: "ColBERTRetriever"
        config:
          model_name: "colbert-v2"
          doc_embeddings_path: "cache/colbert_embeddings.pkl"
          top_k: 100

    # Stage 3: ColBERT Rerank
    - name: "colbert_rerank"
      component:
        type: "ColBERTReranker"
        config: {top_k: 20}

    # Stage 4: 去重和压缩
    - name: "deduplication"
      component:
        type: "SemanticDeduplicator"
        config: {threshold: 0.85}

    - name: "compression"
      component:
        type: "ExtractiveSummaryCompressor"
        config: {compression_ratio: 0.5}

    # Stage 5: 聚类扩展
    - name: "cluster_expansion"
      component:
        type: "ClusterExpander"
        config:
          expansion_strategy: "insert_after_hit"
```

### 示例 4：轻量级检索（快速响应）

适用于简单查询的轻量级配置。

```yaml
# config/pipelines/lightweight.yaml

pipeline:
  name: "lightweight_retrieval"

  stages:
    # Stage 1: 简单检索
    - name: "embedding_search"
      component:
        type: "EmbeddingRetriever"
        config: {top_k: 10}

    # Stage 2: 快速 Rerank
    - name: "rerank"
      component:
        type: "DeepInfraReranker"
        config: {top_k: 5}

    # Stage 3: 高亮
    - name: "highlight"
      component:
        type: "KeywordHighlighter"
        config: {}
```

[⬆️ 返回目录](#目录)

---

## 5. 使用方式

### 方式 1：配置文件驱动

```python
from src.agents.pipeline import PipelineConfigLoader, PipelineContext

# 加载配置
pipeline_config = PipelineConfigLoader.load("config/pipelines/agentic_hybrid.yaml")
pipeline = PipelineConfigLoader.build_pipeline(pipeline_config)

# 创建上下文
context = PipelineContext(
    memory_index=memory_index,
    cluster_index=cluster_index,
    vectorize_service=get_vectorize_service(),
    rerank_service=get_rerank_service(),
    llm_provider=get_llm_provider()
)

# 执行
result = await pipeline.execute(query, context)
```

### 方式 2：程序化构建

```python
# 直接构建 Pipeline
pipeline = RetrievalPipeline("custom")

# 添加阶段
pipeline.add_stage(PipelineStage(
    ColBERTRetriever(model_name="colbert-v2", top_k=100),
    name="colbert_search"
))

pipeline.add_stage(PipelineStage(
    DeepInfraReranker(top_k=20),
    name="rerank"
))

# 执行
result = await pipeline.execute(query, context)
```

[⬆️ 返回目录](#目录)

---

## 6. 实施路线图

### Phase 1: 基础框架（1-2周）

**目标**：建立 Pipeline 抽象层和核心框架

1. 创建 `src/agents/pipeline/` 目录结构
2. 实现核心抽象接口：
   - `components/base.py` - PipelineComponent, Document, RetrievalResult
   - `components/retriever.py` - BaseRetriever
   - `components/reranker.py` - BaseReranker
   - `components/expander.py` - BaseExpander
   - `components/checker.py` - BaseChecker
3. 实现 Pipeline 执行框架：
   - `pipeline.py` - RetrievalPipeline, PipelineStage
   - `context.py` - PipelineContext
4. 实现组件注册机制：
   - `registry.py` - ComponentRegistry
5. 实现配置加载器：
   - `config_loader.py` - PipelineConfigLoader

**验收标准**：
- 所有接口定义完成
- 可以程序化构建简单 Pipeline 并执行

### Phase 2: 兼容适配（1周）

**目标**：包装现有检索逻辑，确保向后兼容

1. 实现 Adapter 层：
   - `adapters.py` - LegacyAgenticRetrieverAdapter, LegacyClusterExpanderAdapter
2. 创建默认配置文件：
   - `config/pipelines/agentic_hybrid.yaml` - 映射现有 agentic_retrieval 流程
3. 单元测试：
   - 验证新 Pipeline 与现有流程结果一致性
   - 测试配置加载和组件创建

**验收标准**：
- 新 Pipeline 运行结果与现有代码一致
- 所有单元测试通过

### Phase 3: 新组件实现（2-3周）

**目标**：实现 ColBERT 等新检索方法

1. **ColBERTRetriever**：
   - 集成 ColBERT 模型（如 `colbert-ir/colbertv2.0`）
   - 预计算 LoCoMo 数据集的文档 embeddings
   - 实现 late interaction 检索算法
   - 优化性能（缓存、批处理）

2. **ColBERTReranker**：
   - 对候选文档进行精细打分
   - 支持 top-k 截断

3. 配置文件：
   - `config/pipelines/colbert.yaml`
   - `config/pipelines/colbert_hybrid.yaml` - ColBERT + 聚类扩展

4. 性能对比测试：
   - ColBERT vs Embedding vs QRHead
   - 准确率 vs 速度权衡分析

**验收标准**：
- ColBERT 在 LoCoMo 数据集上运行成功
- 性能测试报告完成

### Phase 4: 代码重构（可选，2周）

**目标**：重构现有代码使用新 Pipeline API

1. 重构 `eval/adapters/parallax/stage3_memory_retrivel.py`：
   - 使用 PipelineConfigLoader 替代直接调用
2. 重构 `strategy/` 层：
   - 调用 `src/agents/pipeline/` 而非 `eval/` 层
3. 统一配置类：
   - 合并 `AgenticConfig`, `ExperimentConfig`
4. 清理冗余代码：
   - 移除重复的检索逻辑

**验收标准**：
- 策略层与实现层解耦
- 配置统一管理
- 代码量减少 20%+

[⬆️ 返回目录](#目录)

---

## 7. 关键文件清单

### 新增文件

**核心框架**：
- `src/agents/pipeline/components/base.py` - PipelineComponent 基类、Document、RetrievalResult
- `src/agents/pipeline/pipeline.py` - RetrievalPipeline, PipelineStage
- `src/agents/pipeline/context.py` - PipelineContext 执行上下文
- `src/agents/pipeline/registry.py` - ComponentRegistry 组件注册表
- `src/agents/pipeline/config_loader.py` - PipelineConfigLoader 配置加载器

**组件接口（9 大类）**：
- `src/agents/pipeline/components/memory_builder.py` - 记忆构建组件
  - BaseMemoryBuilder, ContentExtractor, MemoryEncoder, MemoryOrganizer
- `src/agents/pipeline/components/question_classifier.py` - 问题分类与路由组件
  - BaseQuestionClassifier, ComplexityClassifier, DomainClassifier, IntentClassifier, StrategyRouter
- `src/agents/pipeline/components/query_preprocessor.py` - 查询预处理组件
  - BaseQueryPreprocessor, QueryRewriter, QueryDecomposer, QueryExpander
- `src/agents/pipeline/components/retriever.py` - 检索组件
  - BaseRetriever, EmbeddingRetriever, BM25Retriever, HybridRetriever, ColBERTRetriever, QRHeadRetriever
- `src/agents/pipeline/components/expander.py` - 结果扩展组件
  - BaseExpander, ClusterExpander, MultiQueryExpander, GraphExpander, TemporalExpander
- `src/agents/pipeline/components/postprocessor.py` - 检索后处理组件
  - BaseReranker, BaseCompressor, BaseDeduplicator, BaseHighlighter
- `src/agents/pipeline/components/prompt_adapter.py` - Prompt 适配组件
  - BasePromptAdapter, TemplateSelector, ContextInjector, PromptComposer
- `src/agents/pipeline/components/judge.py` - 评判组件
  - BaseJudge, RetrievalJudge, AnswerJudge
- `src/agents/pipeline/components/generator.py` - 答案生成组件
  - BaseGenerator, DirectGenerator, IterativeGenerator, ChainOfThoughtGenerator

**Prompt 模板仓库**：
- `src/agents/pipeline/prompts/templates.py` - Prompt 模板定义
- `src/agents/pipeline/prompts/` - 各领域 Prompt 模板目录

**兼容适配**：
- `src/agents/pipeline/adapters.py` - 兼容适配器，包装现有检索逻辑

**配置文件**：
- `config/pipelines/full_agentic_rag.yaml` - 完整 Agentic RAG 配置
- `config/pipelines/memory_building.yaml` - 记忆构建配置
- `config/pipelines/colbert_retrieval.yaml` - ColBERT 检索配置
- `config/pipelines/lightweight.yaml` - 轻量级检索配置
- `config/pipelines/qrhead_retrieval.yaml` - QRHead 检索配置

### 修改文件（Phase 4）

- `eval/adapters/parallax/stage3_memory_retrivel.py` - 使用新 Pipeline API
- `eval/adapters/parallax/strategy/strategies.py` - 调用 `src/agents/pipeline/`
- `eval/adapters/parallax/config.py` - 简化配置，指向 Pipeline 配置文件

### 保持不变（作为 Adapter 使用）

- `src/agents/retrieval_utils.py` - 现有检索工具函数
- `src/agents/agentic_utils.py` - LLM 辅助工具

[⬆️ 返回目录](#目录)

---

## 8. 优势总结

### 8.1 灵活性
- **声明式配置**：修改 YAML 即可切换检索策略，无需改代码
- **组合自由**：任意组合 Retriever + Reranker + Expander
- **条件执行**：支持基于前序结果的动态流程（如充分性判断）

### 8.2 扩展性
- **零代码扩展**：新组件只需实现接口 + 注册到 Registry
- **插件化**：用户可以自定义组件并注册，无需修改框架代码

### 8.3 可维护性
- **关注点分离**：检索逻辑、配置、流程编排完全解耦
- **单一职责**：每个组件只负责一个检索步骤
- **易于测试**：每个组件可独立单元测试

### 8.4 性能
- **并行执行**：Pipeline 支持 async/await 并行阶段
- **缓存友好**：ColBERT 等方法的预计算结果可缓存
- **懒加载**：组件仅在需要时加载

### 8.5 向后兼容
- **渐进式迁移**：通过 Adapter 包装现有代码
- **双轨运行**：新旧系统可以并存
- **零风险**：新架构不影响现有功能

[⬆️ 返回目录](#目录)

---

## 9. 风险与缓解

### 风险 1: 性能开销

**风险**：引入抽象层可能增加性能开销

**缓解措施**：
- 使用 `async/await` 并行执行多个阶段
- 缓存中间结果（如 embeddings、rerank 分数）
- 性能测试确保开销 < 5%

### 风险 2: 配置复杂度

**风险**：配置文件可能过于复杂，增加学习成本

**缓解措施**：
- 提供多个预设配置模板（agentic_hybrid, colbert, lightweight）
- 99% 场景直接使用预设配置
- 详细文档和示例

### 风险 3: 迁移成本

**风险**：重构现有代码需要大量工作

**缓解措施**：
- 使用 Adapter 模式，逐步迁移
- 不强制一次性重写
- Phase 4 为可选阶段

[⬆️ 返回目录](#目录)

---

## 10. 总结

这个重构方案通过引入**扁平可嵌套架构 + 统一 Component 抽象 + 声明式配置**，实现了：

### 核心创新

1. **一切皆 Component**
   - Pipeline 本身也是 Component，可以任意嵌套
   - StrategyRouter 也是 Component，可以灵活组合
   - 无需预设层次，简单场景扁平化，复杂场景自由嵌套

2. **极致灵活性**
   - 任意搭配检索方法（Embedding, BM25, ColBERT, QRHead）
   - Pipeline 可以嵌套 Pipeline，Component 可以包含 Component
   - 支持条件执行、并行执行、结果复用

3. **零代码扩展**
   - 新组件只需实现 `PipelineComponent` 接口
   - 通过 ComponentRegistry 注册即可使用
   - YAML 配置文件即可组合新 Pipeline

4. **9 大组件分类**
   - 覆盖 RAG 全流程：Memory Building → Question Classification → Query Preprocessing → Retrieval → Result Expansion → Postprocessing → Prompt Adaptation → Judgment → Answer Generation
   - 作为推荐分类，帮助用户理解和组织代码
   - 每个大类下可以有任意多个子类实现

5. **向后兼容**
   - 现有 `eval/core/pipeline.py` 保持不变
   - 现有 `StrategyRouter` 可以适配为新架构
   - 通过 Adapter 包装现有检索逻辑

### 关键优势

**相比固定层次架构**：
- ✅ 不受层次限制，任意深度嵌套
- ✅ 简单场景无需过度设计
- ✅ 复杂场景有足够表达力

**相比函数式组合**：
- ✅ 统一接口，易于测试和组合
- ✅ 支持配置文件驱动
- ✅ 更好的类型安全和代码提示

**相比过度抽象**：
- ✅ 概念简单：只有 Component 和 Pipeline
- ✅ 学习成本低：一切皆 Component
- ✅ 调试友好：清晰的执行流程

### 最重要的是

这个设计为集成 **ColBERT、QRHead** 等新检索方法提供了清晰的路径：
1. 实现 `BaseRetriever` 接口
2. 注册到 `ComponentRegistry`
3. 在 YAML 配置文件中使用

**无需修改任何现有代码**，也无需在现有代码上"打补丁"。这将大大提升 Parallax 检索系统的灵活性和可维护性。

[⬆️ 返回目录](#目录)

---

---

## 11. 相关文档

### 用户文档
- **[Agentic Retrieval 使用指南](./agentic_retrieval_guide.md)**: 当前 Agentic 检索功能的快速开始、API 使用、配置说明和最佳实践

### 技术文档
- **[QRHead CPU 实施总结](./QRHead_CPU_实施总结.md)**: QRHead CPU 集成经验和性能优化
- **[QRHead 安装指南](./QRHEAD_SETUP.md)**: QRHead 环境配置和安装步骤

### 代码仓库
- `src/agents/pipeline/` - Pipeline 核心实现（待创建）
- `eval/adapters/parallax/` - 现有检索适配器
- `eval/core/pipeline.py` - 评估流程 Pipeline
- `config/pipelines/` - Pipeline 配置文件（待创建）

[⬆️ 返回目录](#目录)

---

## 附录

### A. 参考资料

- ColBERT 论文：[Efficient Passage Retrieval with Hashing for Open-domain Question Answering](https://arxiv.org/abs/2004.12832)
- QRHead 论文：[Query-Focused Retrieval Heads for Retrieval-Augmented Generation](https://arxiv.org/abs/2410.xxxxx)
- Parallax 现有架构文档：`docs/memory/`

[⬆️ 返回目录](#目录)
