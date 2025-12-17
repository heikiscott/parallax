"""V4 Fine-grained MemUnit Extraction with Batch Processing.

This module provides fine-grained MemUnit extraction for V4 pipeline.
Key optimizations over V3:
- Session-based segmentation (1 LLM call per session)
- Batch narrative generation (N units per LLM call)
- Batch eventlog generation (N units per LLM call)

This reduces LLM calls from O(messages + 2*units) to O(sessions + 2*ceil(units/batch_size))
"""

from typing import Dict, List, Any, Optional
import json
import re
import uuid
import asyncio
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from core.observation.logger import set_activity_id
from utils.datetime_utils import (
    to_iso_format,
    from_iso_format,
    from_timestamp,
    get_now_with_timezone,
    format_datetime_for_prompt,
)
from providers.llm.llm_provider import LLMProvider
from memory.schema import MemUnit, SourceType, EventLog
from memory.extraction.memory import EventLogExtractor
from prompts.memory.en.eval.add.memunit_generation_batch_v4 import (
    SESSION_SEGMENTATION_PROMPT,
    BATCH_NARRATIVE_PROMPT,
    BATCH_EVENTLOG_PROMPT,
)

# Default batch sizes
DEFAULT_BATCH_SIZE_NARRATIVE = 25
DEFAULT_BATCH_SIZE_EVENTLOG = 25


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    if not response:
        return response

    # Try to extract JSON from markdown code block
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find JSON object directly
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response)
    if match:
        return match.group(0)

    return response.strip()


async def segment_session(
    session_messages: List[Dict[str, Any]],
    session_id: str,
    session_time: str,
    llm_provider: LLMProvider,
    conv_id: str = None,
) -> List[Dict[str, Any]]:
    """Segment a single session into fine-grained semantic units.

    Args:
        session_messages: List of message dicts for this session.
        session_id: Session identifier (e.g., "session_1").
        session_time: Session start time string.
        llm_provider: LLM provider instance.
        conv_id: Conversation ID for logging.

    Returns:
        List of unit dicts with messages and summary.
    """
    console = Console()

    # Format conversation for prompt
    conv_lines = []
    for msg in session_messages:
        dia_id = msg.get("dia_id", "unknown")
        speaker = msg.get("speaker_name") or msg.get("speaker_id", "Unknown")
        content = msg.get("content", "")
        conv_lines.append(f"[{dia_id}] {speaker}: {content}")

    conversation_text = "\n".join(conv_lines)

    prompt = SESSION_SEGMENTATION_PROMPT.format(
        session_id=session_id,
        session_time=session_time,
        conversation=conversation_text,
    )

    # Call LLM with retry
    for attempt in range(5):
        try:
            response = await llm_provider.generate(prompt)
            json_str = extract_json_from_response(response)

            if not json_str:
                raise ValueError("Empty JSON after extraction")

            result = json.loads(json_str)

            if "units" not in result:
                raise ValueError("Missing 'units' key in response")

            return result["units"]

        except json.JSONDecodeError as e:
            console.print(f"[yellow]⚠️ JSON parse error ({session_id}, attempt {attempt + 1}): {e}[/yellow]")
            if attempt == 4:
                raise
        except Exception as e:
            console.print(f"[yellow]⚠️ Segmentation error ({session_id}, attempt {attempt + 1}): {e}[/yellow]")
            if attempt == 4:
                raise

    return []


async def batch_generate_narratives(
    units: List[Dict[str, Any]],
    llm_provider: LLMProvider,
    batch_size: int = DEFAULT_BATCH_SIZE_NARRATIVE,
    conv_id: str = None,
) -> List[Dict[str, Any]]:
    """Generate narratives for multiple units in batches.

    Args:
        units: List of unit dicts with messages, timestamp, etc.
        llm_provider: LLM provider instance.
        batch_size: Number of units per LLM call.
        conv_id: Conversation ID for logging.

    Returns:
        List of narrative results: [{unit_id, title, content}, ...]
    """
    console = Console()
    all_results = []

    # Process in batches
    for batch_start in range(0, len(units), batch_size):
        batch_end = min(batch_start + batch_size, len(units))
        batch_units = units[batch_start:batch_end]

        set_activity_id(f"add-v4-{conv_id}-narr-batch{batch_start//batch_size}")

        # Prepare batch input
        units_for_prompt = []
        for i, unit in enumerate(batch_units):
            unit_id = batch_start + i
            # Format messages for this unit
            messages_text = "\n".join([
                f"{m.get('speaker_name', m.get('speaker', 'Unknown'))}: {m.get('content', '')}"
                for m in unit.get("original_data", [])
            ])

            # Get timestamp
            ts = unit.get("timestamp")
            if isinstance(ts, datetime):
                time_str = format_datetime_for_prompt(ts)
            elif isinstance(ts, str):
                time_str = ts
            else:
                time_str = "Unknown time"

            units_for_prompt.append({
                "unit_id": unit_id,
                "timestamp": time_str,
                "messages": messages_text,
            })

        units_json = json.dumps(units_for_prompt, ensure_ascii=False, indent=2)
        prompt = BATCH_NARRATIVE_PROMPT.format(units_json=units_json)

        # Call LLM with retry
        for attempt in range(5):
            try:
                response = await llm_provider.generate(prompt)
                json_str = extract_json_from_response(response)
                result = json.loads(json_str)

                if "results" not in result:
                    raise ValueError("Missing 'results' key")

                all_results.extend(result["results"])
                console.print(f"[dim]✅ Narrative batch {batch_start//batch_size + 1}: {len(result['results'])} generated[/dim]")
                break

            except Exception as e:
                console.print(f"[yellow]⚠️ Narrative batch error (attempt {attempt + 1}): {e}[/yellow]")
                if attempt == 4:
                    # On final failure, create empty results for this batch
                    for i in range(len(batch_units)):
                        all_results.append({
                            "unit_id": batch_start + i,
                            "title": "Extraction failed",
                            "content": batch_units[i].get("summary", ""),
                        })

    return all_results


async def batch_generate_eventlogs(
    units_with_narratives: List[Dict[str, Any]],
    llm_provider: LLMProvider,
    batch_size: int = DEFAULT_BATCH_SIZE_EVENTLOG,
    conv_id: str = None,
) -> List[Dict[str, Any]]:
    """Generate event logs for multiple units in batches.

    Args:
        units_with_narratives: List of dicts with unit_id, narrative, timestamp.
        llm_provider: LLM provider instance.
        batch_size: Number of units per LLM call.
        conv_id: Conversation ID for logging.

    Returns:
        List of eventlog results: [{unit_id, time, atomic_fact}, ...]
    """
    console = Console()
    all_results = []

    # Process in batches
    for batch_start in range(0, len(units_with_narratives), batch_size):
        batch_end = min(batch_start + batch_size, len(units_with_narratives))
        batch_units = units_with_narratives[batch_start:batch_end]

        set_activity_id(f"add-v4-{conv_id}-evlog-batch{batch_start//batch_size}")

        # Prepare batch input
        narratives_for_prompt = []
        for unit in batch_units:
            narratives_for_prompt.append({
                "unit_id": unit["unit_id"],
                "timestamp": unit["timestamp"],
                "narrative": unit["narrative"],
            })

        narratives_json = json.dumps(narratives_for_prompt, ensure_ascii=False, indent=2)
        prompt = BATCH_EVENTLOG_PROMPT.format(narratives_json=narratives_json)

        # Call LLM with retry
        for attempt in range(5):
            try:
                response = await llm_provider.generate(prompt)
                json_str = extract_json_from_response(response)
                result = json.loads(json_str)

                if "results" not in result:
                    raise ValueError("Missing 'results' key")

                all_results.extend(result["results"])
                console.print(f"[dim]✅ EventLog batch {batch_start//batch_size + 1}: {len(result['results'])} generated[/dim]")
                break

            except Exception as e:
                console.print(f"[yellow]⚠️ EventLog batch error (attempt {attempt + 1}): {e}[/yellow]")
                if attempt == 4:
                    # On final failure, create empty results
                    for unit in batch_units:
                        all_results.append({
                            "unit_id": unit["unit_id"],
                            "time": unit["timestamp"],
                            "atomic_fact": [],
                        })

    return all_results


def build_memunit_from_segment(
    segment: Dict[str, Any],
    speakers: set,
    unit_index: int,
    session_messages: List[Dict[str, Any]],
    session_time: Optional[datetime] = None,
) -> MemUnit:
    """Build a MemUnit from a segmented unit.

    Args:
        segment: Unit dict with messages and summary.
        speakers: Set of speaker IDs.
        unit_index: Global index of this unit.
        session_messages: Original session messages for lookup.
        session_time: Session start time.

    Returns:
        MemUnit instance.
    """
    messages = segment.get("messages", [])
    summary = segment.get("summary", "")

    # Build original_data from segment messages
    # Use a dict to merge consecutive entries with same dia_id (LLM sometimes incorrectly splits sentences)
    original_data_dict = {}
    for msg in messages:
        dia_id = msg.get("dia_id", "")
        msg_timestamp = None
        orig_speaker_name = msg.get("speaker", "")

        # Find timestamp from original session messages
        for orig_msg in session_messages:
            if orig_msg.get("dia_id") == dia_id:
                msg_timestamp = orig_msg.get("timestamp")
                orig_speaker_name = orig_msg.get("speaker_name") or orig_msg.get("speaker_id") or orig_speaker_name
                break

        if dia_id in original_data_dict:
            # Merge with existing entry - append content with space
            original_data_dict[dia_id]["content"] += " " + msg.get("content", "")
        else:
            original_data_dict[dia_id] = {
                "dia_id": dia_id,
                "speaker": orig_speaker_name,
                "speaker_name": orig_speaker_name,
                "content": msg.get("content", ""),
                "timestamp": msg_timestamp,
            }

    # Convert dict to list, preserving order of first appearance
    original_data = list(original_data_dict.values())

    # Get timestamp from first message
    timestamp = session_time or get_now_with_timezone()
    if messages and original_data:
        ts = original_data[0].get("timestamp")
        if ts:
            if isinstance(ts, str):
                timestamp = from_iso_format(ts)
            elif isinstance(ts, (int, float)):
                timestamp = from_timestamp(ts)
            elif isinstance(ts, datetime):
                timestamp = ts

    # Extract participants from this unit's messages
    participants = list(set(
        m.get("speaker_name") or m.get("speaker", "")
        for m in original_data
        if m.get("speaker_name") or m.get("speaker")
    ))

    return MemUnit(
        type=SourceType.CONVERSATION,
        unit_id=str(uuid.uuid4()),
        user_id_list=list(speakers),
        participants=participants,
        original_data=original_data,
        timestamp=timestamp,
        summary=summary,
    )


async def memunit_extraction_v4_optimized(
    conversation_by_session: Dict[str, Dict[str, Any]],
    llm_provider: LLMProvider,
    conv_id: str = None,
    batch_size_narrative: int = DEFAULT_BATCH_SIZE_NARRATIVE,
    batch_size_eventlog: int = DEFAULT_BATCH_SIZE_EVENTLOG,
    progress: Progress = None,
    task_id: int = None,
) -> List[MemUnit]:
    """Extract MemUnits using optimized V4 pipeline with batch processing.

    Flow:
    1. Segment each session independently (1 LLM call per session)
    2. Batch generate narratives (N units per LLM call)
    3. Batch generate event logs (N units per LLM call)

    Args:
        conversation_by_session: Dict mapping session_id to {messages, time}.
        llm_provider: LLM provider instance.
        conv_id: Conversation ID for logging.
        batch_size_narrative: Units per narrative batch.
        batch_size_eventlog: Units per eventlog batch.
        progress: Optional progress bar.
        task_id: Task ID for progress updates.

    Returns:
        List of MemUnit instances with narratives and event logs.
    """
    console = Console()
    set_activity_id(f"add-v4-{conv_id}")

    # Collect all speakers
    all_speakers = set()
    for session_data in conversation_by_session.values():
        for msg in session_data.get("messages", []):
            if msg.get("speaker_id"):
                all_speakers.add(msg["speaker_id"])

    # ========== Phase 1: Session-based Segmentation ==========
    if progress and task_id is not None:
        progress.update(task_id, status="Phase 1: Segmenting...")

    console.print(f"[cyan]📦 Phase 1: Segmenting {len(conversation_by_session)} sessions...[/cyan]")

    all_units = []  # List of (segment_dict, session_messages, session_time)
    session_ids = sorted(
        conversation_by_session.keys(),
        key=lambda x: int(x.replace("session_", "")) if x.startswith("session_") else 0
    )

    for session_id in session_ids:
        session_data = conversation_by_session[session_id]
        session_messages = session_data.get("messages", [])
        session_time_str = session_data.get("time", "")
        session_time = None

        if session_time_str:
            try:
                session_time = from_iso_format(session_time_str)
            except:
                pass

        # Segment this session
        segments = await segment_session(
            session_messages=session_messages,
            session_id=session_id,
            session_time=session_time_str,
            llm_provider=llm_provider,
            conv_id=conv_id,
        )

        console.print(f"[dim]  {session_id}: {len(segments)} units[/dim]")

        for seg in segments:
            all_units.append((seg, session_messages, session_time))

    total_units = len(all_units)
    console.print(f"[cyan]📦 Total units from all sessions: {total_units}[/cyan]")

    # Build MemUnits from segments
    memunit_list = []
    for idx, (segment, session_msgs, session_time) in enumerate(all_units):
        memunit = build_memunit_from_segment(
            segment=segment,
            speakers=all_speakers,
            unit_index=idx,
            session_messages=session_msgs,
            session_time=session_time,
        )
        memunit_list.append(memunit)

    # ========== Phase 2: Batch Narrative Generation ==========
    if progress and task_id is not None:
        progress.update(task_id, status="Phase 2: Narratives...")

    console.print(f"[cyan]📝 Phase 2: Generating narratives (batch_size={batch_size_narrative})...[/cyan]")

    # Prepare units for narrative generation
    units_for_narrative = []
    for idx, memunit in enumerate(memunit_list):
        ts = memunit.timestamp
        if isinstance(ts, datetime):
            time_str = format_datetime_for_prompt(ts)
        else:
            time_str = str(ts) if ts else "Unknown"

        units_for_narrative.append({
            "unit_id": idx,
            "timestamp": time_str,
            "original_data": memunit.original_data,
            "summary": memunit.summary,
        })

    narrative_results = await batch_generate_narratives(
        units=units_for_narrative,
        llm_provider=llm_provider,
        batch_size=batch_size_narrative,
        conv_id=conv_id,
    )

    # Apply narratives to MemUnits
    narrative_map = {r["unit_id"]: r for r in narrative_results}
    for idx, memunit in enumerate(memunit_list):
        if idx in narrative_map:
            result = narrative_map[idx]
            memunit.narrative = result.get("content", "")
            memunit.subject = result.get("title", "")
            # Update summary
            narrative = memunit.narrative
            memunit.summary = narrative[:200] + "..." if len(narrative) > 200 else narrative

    # ========== Phase 3: Batch EventLog Generation ==========
    if progress and task_id is not None:
        progress.update(task_id, status="Phase 3: EventLogs...")

    console.print(f"[cyan]📋 Phase 3: Generating event logs (batch_size={batch_size_eventlog})...[/cyan]")

    # Prepare units for eventlog generation
    units_for_eventlog = []
    for idx, memunit in enumerate(memunit_list):
        if memunit.narrative:
            ts = memunit.timestamp
            if isinstance(ts, datetime):
                time_str = format_datetime_for_prompt(ts)
            else:
                time_str = str(ts) if ts else "Unknown"

            units_for_eventlog.append({
                "unit_id": idx,
                "timestamp": time_str,
                "narrative": memunit.narrative,
            })

    eventlog_results = await batch_generate_eventlogs(
        units_with_narratives=units_for_eventlog,
        llm_provider=llm_provider,
        batch_size=batch_size_eventlog,
        conv_id=conv_id,
    )

    # Apply event logs to MemUnits
    eventlog_map = {r["unit_id"]: r for r in eventlog_results}
    for idx, memunit in enumerate(memunit_list):
        if idx in eventlog_map:
            result = eventlog_map[idx]
            memunit.event_log = EventLog(
                time=result.get("time", ""),
                atomic_fact=result.get("atomic_fact", []),
            )

    if progress and task_id is not None:
        progress.update(task_id, status="✅")

    # Print summary
    num_sessions = len(conversation_by_session)
    num_narrative_batches = (total_units + batch_size_narrative - 1) // batch_size_narrative
    num_eventlog_batches = (total_units + batch_size_eventlog - 1) // batch_size_eventlog
    total_llm_calls = num_sessions + num_narrative_batches + num_eventlog_batches

    console.print(f"[green]✅ V4 Extraction complete:[/green]")
    console.print(f"[dim]   Sessions: {num_sessions} → {total_units} units[/dim]")
    console.print(f"[dim]   LLM calls: {num_sessions} (seg) + {num_narrative_batches} (narr) + {num_eventlog_batches} (evlog) = {total_llm_calls} total[/dim]")

    return memunit_list


async def process_single_conversation_v4(
    conv_id: str,
    conversation: List[Dict[str, Any]],
    save_dir: str,
    llm_provider: LLMProvider,
    event_log_extractor: EventLogExtractor = None,  # Kept for API compatibility, not used
    progress_counter: dict = None,
    progress: Progress = None,
    task_id: int = None,
    config: Any = None,
) -> tuple:
    """Process a single conversation using optimized V4 extraction.

    Args:
        conv_id: Conversation ID.
        conversation: List of message dicts (flat, with session info in each msg).
        save_dir: Directory to save results.
        llm_provider: Shared LLM provider.
        event_log_extractor: Deprecated, event logs are now batch-generated.
        progress_counter: Progress counter dict.
        progress: Progress bar.
        task_id: Task ID for progress updates.
        config: Configuration object.

    Returns:
        Tuple of (conv_id, memunit_list).
    """
    console = Console()

    try:
        set_activity_id(f"add-v4-{conv_id}")

        if progress and task_id is not None:
            progress.update(task_id, status="Organizing sessions...")

        # Organize messages by session
        # Messages should have a 'session' field, otherwise treat as single session
        conversation_by_session: Dict[str, Dict[str, Any]] = {}

        for msg in conversation:
            session_id = msg.get("session", "session_1")
            if session_id not in conversation_by_session:
                conversation_by_session[session_id] = {
                    "messages": [],
                    "time": msg.get("timestamp", ""),
                }
            conversation_by_session[session_id]["messages"].append(msg)
            # Update session time to first message's timestamp
            if not conversation_by_session[session_id]["time"]:
                conversation_by_session[session_id]["time"] = msg.get("timestamp", "")

        # Get batch sizes from config
        batch_size_narrative = config.get("v4.batch_size_narrative", DEFAULT_BATCH_SIZE_NARRATIVE) if config else DEFAULT_BATCH_SIZE_NARRATIVE
        batch_size_eventlog = config.get("v4.batch_size_eventlog", DEFAULT_BATCH_SIZE_EVENTLOG) if config else DEFAULT_BATCH_SIZE_EVENTLOG

        # Extract MemUnits using optimized V4 method
        memunit_list = await memunit_extraction_v4_optimized(
            conversation_by_session=conversation_by_session,
            llm_provider=llm_provider,
            conv_id=conv_id,
            batch_size_narrative=batch_size_narrative,
            batch_size_eventlog=batch_size_eventlog,
            progress=progress,
            task_id=task_id,
        )

        # Normalize timestamps
        for memunit in memunit_list:
            if hasattr(memunit, 'timestamp'):
                ts = memunit.timestamp
                if isinstance(ts, (int, float)):
                    memunit.timestamp = from_timestamp(ts)
                elif isinstance(ts, str):
                    memunit.timestamp = from_iso_format(ts)
                elif not isinstance(ts, datetime):
                    memunit.timestamp = get_now_with_timezone()

        # Save results
        memunit_dicts = []
        for memunit in memunit_list:
            memunit_dict = memunit.to_dict()
            if hasattr(memunit, 'event_log') and memunit.event_log:
                memunit_dict['event_log'] = memunit.event_log.to_dict()
            memunit_dicts.append(memunit_dict)

        output_file = Path(save_dir) / f"memunit_list_conv_{conv_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(memunit_dicts, f, ensure_ascii=False, indent=2)

        # Save stats
        stats = {
            "conv_id": conv_id,
            "memunits": len(memunit_list),
            "sessions": len(conversation_by_session),
            "pipeline_version": "v4_optimized",
            "granularity": "fine-grained",
            "batch_size_narrative": batch_size_narrative,
            "batch_size_eventlog": batch_size_eventlog,
        }
        stats_file = Path(save_dir) / "stats" / f"conv_{conv_id}_stats.json"
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_file, "w") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        if progress_counter:
            progress_counter['completed'] += 1

        return conv_id, memunit_list

    except Exception as e:
        console.print(f"[red]❌ Conv-{conv_id} failed: {e}[/red]")
        import traceback
        traceback.print_exc()

        if progress_counter:
            progress_counter['completed'] += 1
            progress_counter['failed'] += 1

        return conv_id, []


