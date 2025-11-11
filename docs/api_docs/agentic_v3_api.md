# Agentic Layer V3 API Documentation

## Overview

The Agentic Layer V3 API provides specialized interfaces for processing group chat memories, using a simple and direct message format without any preprocessing or format conversion.

## Key Features

- ✅ **Simple and Direct**: Uses the simplest single message format, no complex data structures required
- ✅ **No Conversion Needed**: No format conversion or adaptation required
- ✅ **Sequential Processing**: Real-time processing of each message, suitable for message stream scenarios
- ✅ **Centralized Adaptation**: All format conversion logic centralized in `group_chat_converter.py`, maintaining single responsibility
- ✅ **Backward Compatible**: V2 interface still available, supporting gradual migration
- ✅ **Detailed Error Messages**: Provides clear error prompts and data statistics

## Interface Comparison

| Feature | V2 Interface | V3 Interface ⭐ |
|---------|-------------|----------------|
| Endpoint | `/api/v2/agentic/memorize` | `/api/v3/agentic/memorize` |
| Input Format | Internal Format | **Simple Direct Format** |
| Processing Method | Sequential (requires external conversion) | **Sequential (no conversion needed)** |
| Format Complexity | High | **Low (simplest)** |
| Recommended Scenario | Existing conversion logic | **Real-time message streams (recommended)** |

**Advantages of V3 Interface**:
- ✅ Simplest format, just pass required fields directly
- ✅ No format conversion or adaptation needed
- ✅ Suitable for real-time message processing scenarios
- ✅ Optimal performance (no conversion overhead)

## Interface Specification

### POST `/api/v3/agentic/memorize`

Store a single group chat message memory

#### Request Format

**Content-Type**: `application/json`

**Request Body**: Simple direct single message format (no pre-conversion needed)

```json
{
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Let's discuss the technical approach for the new feature today",
  "refer_list": ["msg_000"]
}
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| group_id | string | No | Group ID |
| group_name | string | No | Group name |
| message_id | string | Yes | Unique message identifier |
| create_time | string | Yes | Message creation time (ISO 8601 format) |
| sender | string | Yes | Sender user ID |
| sender_name | string | No | Sender name (defaults to sender) |
| content | string | Yes | Message content |
| refer_list | array | No | List of referenced message IDs |

#### Response Format

**Success Response (200 OK)**

```json
{
  "status": "ok",
  "message": "Memory stored successfully, 1 memory saved",
  "result": {
    "saved_memories": [
      {
        "memory_type": "episode_summary",
        "user_id": "user_001",
        "group_id": "group_123",
        "timestamp": "2025-01-15T10:00:00",
        "content": "User discussed technical approach for the new feature"
      }
    ],
    "count": 1
  }
}
```

**Error Response (400 Bad Request)**

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Data format error: missing required field message_id",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

**Error Response (500 Internal Server Error)**

```json
{
  "status": "failed",
  "code": "SYSTEM_ERROR",
  "message": "Failed to store memory, please try again later",
  "timestamp": "2025-01-15T10:30:00+00:00",
  "path": "/api/v3/agentic/memorize"
}
```

---

## Use Cases

### 1. Real-time Message Stream Processing

Suitable for processing real-time message streams from chat applications, storing each message as it arrives.

**Example**:
```json
{
  "group_id": "group_123",
  "group_name": "Project Discussion Group",
  "message_id": "msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Let's discuss the technical approach for the new feature today",
  "refer_list": []
}
```

### 2. Chatbot Integration

After a chatbot receives a user message, it can directly call the V3 interface to store the memory.

**Example**:
```json
{
  "group_id": "bot_conversation_123",
  "group_name": "Conversation with AI Assistant",
  "message_id": "bot_msg_001",
  "create_time": "2025-01-15T10:05:00+08:00",
  "sender": "user_456",
  "sender_name": "Li Si",
  "content": "Help me summarize today's meeting content",
  "refer_list": []
}
```

### 3. Message Queue Consumption

When consuming messages from a message queue (such as Kafka), you can call the V3 interface for each message.

**Kafka Consumer Example**:
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

# Kafka consumer
consumer = KafkaConsumer('chat_messages')
for msg in consumer:
    asyncio.run(process_message(msg.value))
```

---

## Usage Examples

### Using curl

```bash
curl -X POST http://localhost:1995/api/v3/agentic/memorize \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "group_123",
    "group_name": "Project Discussion Group",
    "message_id": "msg_001",
    "create_time": "2025-01-15T10:00:00+08:00",
    "sender": "user_001",
    "sender_name": "Zhang San",
    "content": "Let'\''s discuss the technical approach for the new feature today",
    "refer_list": []
  }'
```

### Using Python Code

```python
import httpx
import asyncio

async def call_v3_memorize():
    # Simple direct single message format
    message_data = {
        "group_id": "group_123",
        "group_name": "Project Discussion Group",
        "message_id": "msg_001",
        "create_time": "2025-01-15T10:00:00+08:00",
        "sender": "user_001",
        "sender_name": "Zhang San",
        "content": "Let's discuss the technical approach for the new feature today",
        "refer_list": []
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:1995/api/v3/agentic/memorize",
            json=message_data
        )
        result = response.json()
        print(f"Saved {result['result']['count']} memories")

asyncio.run(call_v3_memorize())
```

### Using run_memorize.py Script

For JSON files in GroupChatFormat, you can use the `run_memorize.py` script for batch processing:

```bash
# V3 interface (recommended)
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize

# V2 interface (compatibility mode)
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v2/agentic/memorize \
  --use-v2

# Validate format only
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --validate-only
```

---

## FAQ

### 1. How to choose between V3 and V2 interfaces?

**V3 interface is recommended** for the following reasons:
- ✅ Simpler format, only required fields needed
- ✅ No format conversion or adaptation required
- ✅ Suitable for real-time message processing scenarios
- ✅ Better performance (no conversion overhead)

**Use V2 interface only when**:
- Existing code uses V2 interface and needs compatibility
- Already have well-established format conversion logic

### 2. How to handle messages with references?

Use the `refer_list` field to specify the list of referenced message IDs:

```json
{
  "message_id": "msg_002",
  "content": "I agree with your approach",
  "refer_list": ["msg_001"]
}
```

### 3. Are group_id and group_name required?

Not required, but **strongly recommended**:
- `group_id` is used to identify the group for easier retrieval
- `group_name` is used for display and understanding, improving readability

### 4. How to handle private chat messages?

Private chat messages can omit `group_id`, or use a special private chat ID:

```json
{
  "group_id": "private_user001_user002",
  "group_name": "Private chat with Zhang San",
  "message_id": "private_msg_001",
  "create_time": "2025-01-15T10:00:00+08:00",
  "sender": "user_001",
  "sender_name": "Zhang San",
  "content": "Hi, how are you doing?",
  "refer_list": []
}
```

### 5. How to handle message timestamps?

`create_time` must use ISO 8601 format, timezone support:

```json
{
  "create_time": "2025-01-15T10:00:00+08:00"  // with timezone
}
```

Or without timezone (defaults to UTC):

```json
{
  "create_time": "2025-01-15T10:00:00"  // UTC
}
```

### 6. How to batch process historical messages?

Use the `run_memorize.py` script:

1. Prepare a JSON file in GroupChatFormat
2. Run the script, which will automatically call the V3 interface for each message

```bash
python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat.json \
  --api-url http://localhost:1995/api/v3/agentic/memorize
```

### 7. Are there rate limits for API calls?

Currently no hard limits, but we recommend:
- Real-time scenarios: No more than 100 requests per second
- Batch import: Suggest 0.1 second interval between messages

### 8. How to handle errors?

The interface returns detailed error messages:

```json
{
  "status": "failed",
  "code": "INVALID_PARAMETER",
  "message": "Missing required field: message_id"
}
```

We recommend implementing retry mechanism on the client side, with up to 3 retries for 5xx errors.

---

## Architecture

### Data Flow

```
Client
  ↓
  │ Simple direct single message format
  ↓
V3 Controller (agentic_v3_controller.py)
  ↓
  │ Call group_chat_converter.py
  ↓
Format Conversion (convert_simple_message_to_memorize_input)
  ↓
  │ Internal format
  ↓
Memory Manager (memory_manager.py)
  ↓
  │ Memory storage
  ↓
Database / Vector Database
```

### Core Components

1. **V3 Controller** (`agentic_v3_controller.py`)
   - Receives simple direct single messages
   - Calls converter for format conversion
   - Calls memory_manager to store memories

2. **Group Chat Converter** (`group_chat_converter.py`)
   - Centralized adaptation layer
   - Responsible for all format conversion logic
   - Maintains single responsibility

3. **Memory Manager** (`memory_manager.py`)
   - Memory extraction and storage
   - Vectorization
   - Persistence

---

## Migration Guide

### Migrating from V2 to V3

#### V2 Interface (Old)

```python
# Need to convert format first
memorize_input = convert_group_chat_format_to_memorize_input(group_chat_data)

# Call sequentially
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

#### V3 Interface (New)

```python
# Call directly, no conversion needed
response = await client.post(
    "http://localhost:1995/api/v3/agentic/memorize",
    json={
        "group_id": "group_123",
        "group_name": "Project Discussion Group",
        "message_id": "msg_001",
        "create_time": "2025-01-15T10:00:00+08:00",
        "sender": "user_001",
        "sender_name": "Zhang San",
        "content": "Let's discuss the technical approach for the new feature today",
        "refer_list": []
    }
)
```

**Migration Benefits**:
- Cleaner code
- No format conversion needed
- Better performance

---

## Related Documentation

- [GroupChatFormat Specification](../../data_format/group_chat/group_chat_format.md)
- [V3 API Testing Guide](../dev_docs/v3_api_testing_guide.md)
- [run_memorize.py Usage Guide](../dev_docs/run_memorize_usage.md)
- [V2 API Documentation](./agentic_v2_api.md)

