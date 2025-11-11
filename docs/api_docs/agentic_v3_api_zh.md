# Agentic Layer V3 API 文档

## 概述

Agentic Layer V3 API 提供了专门用于处理群聊记忆的接口，采用简单直接的消息格式，无需任何预处理或格式转换。

## 主要特性

- ✅ **简单直接**：采用最简单的单条消息格式，无需复杂的数据结构
- ✅ **无需转换**：不需要任何格式转换或适配
- ✅ **逐条处理**：实时处理每条消息，适合消息流场景
- ✅ **集中式适配**：所有格式转换逻辑集中在 `group_chat_converter.py`，保持单一职责
- ✅ **向后兼容**：V2 接口依然可用，支持渐进式迁移
- ✅ **详细错误信息**：提供清晰的错误提示和数据统计

## 接口对比

| 特性 | V2 接口 | V3 接口 ⭐ |
|------|---------|-----------|
| 端点 | `/api/v2/agentic/memorize` | `/api/v3/agentic/memorize` |
| 输入格式 | 内部格式 | **简单直接格式** |
| 处理方式 | 逐条（需外部转换） | **逐条（无需转换）** |
| 格式复杂度 | 高 | **低（最简单）** |
| 推荐场景 | 已有转换逻辑 | **实时消息流（推荐）** |

**V3 接口的优势**：
- ✅ 格式最简单，直接传入必要字段即可
- ✅ 无需任何格式转换或适配
- ✅ 适合实时消息处理场景
- ✅ 性能最优（无转换开销）

## 接口说明

### POST `/api/v3/agentic/memorize`

逐条存储单条群聊消息记忆

#### 请求格式

**Content-Type**: `application/json`

**请求体**：简单直接的单条消息格式（无需预转换）

```json
{
  "group_id": "group_123",
  "group_name": "项目讨论组",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "今天讨论下新功能的技术方案",
  "refer_list": ["msg_000"]
}
```

**字段说明**：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| group_id | string | 否 | 群组ID |
| group_name | string | 否 | 群组名称 |
| message_id | string | 是 | 消息唯一标识 |
| create_time | string | 是 | 消息创建时间（ISO 8601格式） |
| sender | string | 是 | 发送者用户ID |
| sender_name | string | 否 | 发送者名称（默认使用 sender） |
| content | string | 是 | 消息内容 |
| refer_list | array | 否 | 引用的消息ID列表 |

#### 响应格式

**成功响应 (200 OK)**

```json
{
  "status": "ok",
  "message": "记忆存储成功，共保存 1 条记忆",
  "result": {
    "saved_memories": [
      {
        "memory_type": "episode_summary",
        "user_id": "user_001",
        "group_id": "group_123",
        "timestamp": "2025-01-15T10:00:00",
        "content": "用户讨论了新功能的技术方案"
      }
    ],
    "count": 1
  }
}
```

**错误响应 (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "数据格式错误：缺少必需字段 message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

**错误响应 (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "存储记忆失败，请稍后重试",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

---

## 使用场景

### 1. 实时消息流处理

适用于处理来自聊天应用的实时消息流，每收到一条消息就立即存储。

**示例**：
```json
{
  "group_id": "group_123",
  "group_name": "项目讨论组",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "今天讨论下新功能的技术方案",
  "refer_list": []
}
```

### 2. 聊天机器人集成

聊天机器人接收到用户消息后，可以直接调用 V3 接口存储记忆。

**示例**：
```json
{
  "group_id": "bot_conversation_123",
  "group_name": "与AI助手的对话",
  "message_id": "bot_msg_001",
  "create_time": "2025-01-15T10:05:00+08:00",
  "sender": "user_456",
  "sender_name": "李四",
  "content": "帮我总结下今天的会议内容",
  "refer_list": []
}
```

### 3. 消息队列消费

从消息队列（如 Kafka）消费消息时，可以逐条调用 V3 接口处理。

**Kafka 消费示例**：
```python
from kafka import KafkaConsumer
import httpx
import asyncio

async def process_message(message):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json={
                "group_id": message["group_id"],
                "group_name": message["group_name"],
                "message_id": message["message_id"],
                "create_time": message["create_time"],
                "sender": message["sender"],
                "sender_name": message["sender_name"],
                "content": message["content"],
                "refer_list": message.get("refer_list", [])
            }
        )
        return response.json()

# Kafka 消费者
consumer = KafkaConsumer('chat_messages')
for msg in consumer:
    asyncio.run(process_message(msg.value))
```

---

## 使用示例

### 使用 curl 调用

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "项目讨论组",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "张三",
    "content": "今天讨论下新功能的技术方案",
    "refer_list": []
  }'
```

### 使用 Python 代码调用

```python
import httpx
import asyncio

async def call_v3_memorize():
    # 简单直接的单条消息格式
    message_data = {
        "group_id": "group_123",
        "group_name": "项目讨论组",
        "message_id": "msg_001",
        "create_time": "2025-01-15T10:00:00+08:00",
        "sender": "user_001",
        "sender_name": "张三",
        "content": "今天讨论下新功能的技术方案",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json=message_data
        )
        result = response.json()
        print(f"保存了 {result['result']['count']} 条记忆")

asyncio.run(call_v3_memorize())
```

### 使用 run_memorize.py 脚本

对于 GroupChatFormat 格式的 JSON 文件，可以使用 `run_memorize.py` 脚本批量处理：

```bash
# V3 接口（推荐）
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize

# V2 接口（兼容模式）
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v2/agentic/memorize \
  --use-v2

# 仅验证格式
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

---

## 常见问题

### 1. V3 接口和 V2 接口应该如何选择？

**推荐使用 V3 接口**，理由如下：
- ✅ 格式更简单，只需要提供必要的字段
- ✅ 无需任何格式转换或适配
- ✅ 适合实时消息处理场景
- ✅ 性能更好（无转换开销）

**仅在以下情况使用 V2 接口**：
- 已有代码使用 V2 接口，需要保持兼容
- 已有完善的格式转换逻辑

### 2. 如何处理带引用的消息？

使用 `refer_list` 字段指定引用的消息ID列表：

```json
{
  "message_id": "msg_002",
  "content": "我同意你的方案",
  "refer_list": ["msg_001"]
}
```

### 3. group_id 和 group_name 是必需的吗？

不是必需的，但**强烈推荐提供**：
- `group_id` 用于标识群组，方便后续检索
- `group_name` 用于显示和理解，提升可读性

### 4. 如何处理私聊消息？

私聊消息可以不提供 `group_id`，或者使用特殊的私聊ID：

```json
{
  "group_id": "private_user001_user002",
  "group_name": "与张三的私聊",
  "message_id": "private_msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "张三",
  "content": "你好，最近怎么样？",
  "refer_list": []
}
```

### 5. 如何处理消息时间？

`create_time` 必须使用 ISO 8601 格式，支持带时区：

```json
{
  "create_time": "2025-01-15T10:00:00+08:00"  // 带时区
}
```

或不带时区（默认使用 UTC）：

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 6. 如何批量处理历史消息？

使用 `run_memorize.py` 脚本：

1. 准备 GroupChatFormat 格式的 JSON 文件
2. 运行脚本，脚本会自动逐条调用 V3 接口

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize
```

### 7. 接口调用频率有限制吗？

目前没有硬性限制，但建议：
- 实时场景：每秒不超过 100 次请求
- 批量导入：建议每条消息间隔 0.1 秒

### 8. 如何处理错误？

接口会返回详细的错误信息：

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "缺少必需字段: message_id"
}
```

建议在客户端实现重试机制，对于 5xx 错误可以重试 3 次。

---

## 架构说明

### 数据流

```
客户端
  ↓
  │ 简单直接的单条消息格式
  ↓
V3 Controller (agentic_v3_controller.py)
  ↓
  │ 调用 group_chat_converter.py
  ↓
格式转换 (convert_simple_message_to_memorize_input)
  ↓
  │ 内部格式
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ 记忆存储
  ↓
数据库 / 向量库
```

### 核心组件

1. **V3 Controller** (`agentic_v3_controller.py`)
   - 接收简单直接的单条消息
   - 调用 converter 进行格式转换
   - 调用 memory_manager 存储记忆

2. **Group Chat Converter** (`group_chat_converter.py`)
   - 集中式适配层
   - 负责所有格式转换逻辑
   - 保持单一职责

3. **Memory Manager** (`memory_manager.py`)
   - 记忆提取和存储
   - 向量化
   - 持久化

---

## 迁移指南

### 从 V2 迁移到 V3

#### V2 接口（旧）

```python
# 需要先转换格式
memorize_input = convert_group_chat_format_to_memorize_input(group_chat_data)

# 逐条调用
for message in memorize_input["messages"]:
    response = await client.post(
        "http://localhost:1995/api/v2/agentic/memorize",
        json={
            "messages": [message],
            "group_id": group_id,
            "raw_data_type": "Conversation"
        }
    )
```

#### V3 接口（新）

```python
# 直接调用，无需转换
response = await client.post(
    "http://localhost:1995/api/v3/agentic/memorize",
    json={
        "group_id": "group_123",
        "group_name": "项目讨论组",
        "message_id": "msg_001",
        "create_time": "2025-01-15T10:00:00+08:00",
        "sender": "user_001",
        "sender_name": "张三",
        "content": "今天讨论下新功能的技术方案",
        "refer_list": []
    }
)
```

**迁移优势**：
- 代码更简洁
- 无需格式转换
- 性能更好

---

## 相关文档

- [GroupChatFormat 格式规范](../../data_format/group_chat/group_chat_format.md)
- [V3 API 测试指南](../dev_docs/v3_api_testing_guide.md)
- [run_memorize.py 使用指南](../dev_docs/run_memorize_usage.md)
- [V2 API 文档](./agentic_v2_api.md)
