"""V4 Fine-grained MemUnit Extraction.

This module provides fine-grained MemUnit extraction for V4 pipeline.
Unlike V3 which detects boundaries per-message, V4 splits the entire
conversation in one LLM call into minimal semantic units.
"""

from typing import Dict, List, Any
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
)
from providers.llm.llm_provider import LLMProvider
from memory.schema import MemUnit, SourceType
from memory.extraction.memory import (
    EpisodeMemoryExtractRequest,
    EpisodeMemoryExtractor,
    EventLogExtractor,
)
from src.prompts.memory.en.eval.add.memunit_segmentation_v4 import FINE_GRAINED_SEGMENTATION_PROMPT


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response that may contain markdown formatting.

    Returns:
        Extracted JSON string.
    """
    if not response:
        return response

    # Try to extract JSON from markdown code block
    # Pattern matches ```json ... ``` or ``` ... ```
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, try to find JSON object directly
    # Look for content starting with { and ending with }
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response)
    if match:
        return match.group(0)

    # Return original if no patterns match
    return response.strip()


async def segment_conversation_v4(
    conversation: List[Dict[str, Any]],
    llm_provider: LLMProvider,
    conv_id: str = None,
) -> List[Dict[str, Any]]:
    """Segment conversation into fine-grained semantic units using LLM.

    Args:
        conversation: List of message dicts with keys: speaker_id, content, dia_id, etc.
        llm_provider: LLM provider instance.
        conv_id: Conversation ID for logging.

    Returns:
        List of unit dicts, each containing:
        - messages: List of {dia_id, speaker, content}
        - summary: Brief description
    """
    # Format conversation for prompt
    conv_lines = []
    for msg in conversation:
        dia_id = msg.get("dia_id", "unknown")
        # Use speaker_name if available, fallback to speaker_id
        speaker = msg.get("speaker_name") or msg.get("speaker_id", "Unknown")
        content = msg.get("content", "")
        conv_lines.append(f"[{dia_id}] {speaker}: {content}")

    conversation_text = "\n".join(conv_lines)

    prompt = FINE_GRAINED_SEGMENTATION_PROMPT.format(conversation=conversation_text)

    console = Console()
    console.print(f"[cyan]🔄 Starting segmentation for conv-{conv_id}, prompt length: {len(prompt)} chars[/cyan]")

    # Call LLM with retry
    for attempt in range(5):
        try:
            console.print(f"[dim]📡 Calling LLM (attempt {attempt + 1}/5)...[/dim]")
            response = await llm_provider.generate(prompt)
            console.print(f"[dim]✅ LLM response received[/dim]")

            # Debug: log response info
            response_len = len(response) if response else 0
            console.print(f"[dim]📝 LLM response length: {response_len} chars[/dim]")
            if response_len < 100:
                console.print(f"[dim]📝 Full response: {repr(response)}[/dim]")
            else:
                console.print(f"[dim]📝 Response preview: {repr(response[:200])}...[/dim]")

            # Extract JSON from response (handles markdown code blocks)
            json_str = extract_json_from_response(response)

            if not json_str:
                raise ValueError(f"Empty JSON after extraction. Original response length: {response_len}")

            result = json.loads(json_str)

            if "units" not in result:
                raise ValueError("Missing 'units' key in response")

            return result["units"]

        except json.JSONDecodeError as e:
            console.print(f"[yellow]⚠️ JSON parse error (attempt {attempt + 1}): {e}[/yellow]")
            if response:
                console.print(f"[yellow]   Response preview: {repr(response[:500] if len(response) > 500 else response)}[/yellow]")
            if attempt == 4:
                raise
        except Exception as e:
            console.print(f"[yellow]⚠️ Segmentation error (attempt {attempt + 1}): {e}[/yellow]")
            if attempt == 4:
                raise

    return []


def build_memunit_from_segment(
    segment: Dict[str, Any],
    speakers: set,
    unit_index: int,
    conversation: List[Dict[str, Any]],
) -> MemUnit:
    """Build a MemUnit from a segmented unit.

    Args:
        segment: Unit dict with messages and summary.
        speakers: Set of speaker IDs.
        unit_index: Index of this unit.
        conversation: Original conversation for timestamp lookup.

    Returns:
        MemUnit instance.
    """
    messages = segment.get("messages", [])
    summary = segment.get("summary", "")

    # Build original_data from segment messages
    # Need to include timestamp and speaker_name for EpisodeMemoryExtractor compatibility
    original_data = []
    for msg in messages:
        dia_id = msg.get("dia_id", "")
        # Find timestamp and speaker_name from original conversation
        msg_timestamp = None
        orig_speaker_name = msg.get("speaker", "")  # Fallback to LLM output
        for orig_msg in conversation:
            if orig_msg.get("dia_id") == dia_id:
                msg_timestamp = orig_msg.get("timestamp")
                # Use original speaker_name if available
                orig_speaker_name = orig_msg.get("speaker_name") or orig_msg.get("speaker_id") or orig_speaker_name
                break

        original_data.append({
            "dia_id": dia_id,
            "speaker": orig_speaker_name,
            "speaker_name": orig_speaker_name,
            "content": msg.get("content", ""),
            "timestamp": msg_timestamp,
        })

    # Get timestamp from first message's dia_id
    timestamp = get_now_with_timezone()
    if messages:
        first_dia_id = messages[0].get("dia_id", "")
        # Find matching message in original conversation
        for orig_msg in conversation:
            if orig_msg.get("dia_id") == first_dia_id:
                ts = orig_msg.get("timestamp")
                if ts:
                    if isinstance(ts, str):
                        timestamp = from_iso_format(ts)
                    elif isinstance(ts, (int, float)):
                        timestamp = from_timestamp(ts)
                    elif isinstance(ts, datetime):
                        timestamp = ts
                break

    memunit = MemUnit(
        type=SourceType.CONVERSATION,
        unit_id=str(uuid.uuid4()),
        user_id_list=list(speakers),
        original_data=original_data,
        timestamp=timestamp,
        summary=summary,
    )

    return memunit


async def memunit_extraction_v4(
    conversation: List[Dict[str, Any]],
    llm_provider: LLMProvider,
    conv_id: str = None,
    progress: Progress = None,
    task_id: int = None,
) -> List[MemUnit]:
    """Extract MemUnits using V4 fine-grained segmentation.

    Args:
        conversation: List of message dicts.
        llm_provider: LLM provider instance.
        conv_id: Conversation ID for logging.
        progress: Optional progress bar.
        task_id: Optional task ID for progress updates.

    Returns:
        List of MemUnit instances.
    """
    set_activity_id(f"add-v4-{conv_id}")

    # Get unique speakers
    speakers = {
        msg.get("speaker_id")
        for msg in conversation
        if msg.get("speaker_id")
    }

    # Update progress: segmentation phase
    if progress and task_id is not None:
        progress.update(task_id, completed=0, status="Segmenting...")

    # Segment conversation in one LLM call
    segments = await segment_conversation_v4(conversation, llm_provider, conv_id)

    total_segments = len(segments)
    console = Console()
    console.print(f"[cyan]📦 Conv-{conv_id}: Split into {total_segments} fine-grained units[/cyan]")

    # Build MemUnits from segments
    memunit_list = []
    episode_extractor = EpisodeMemoryExtractor(llm_provider=llm_provider, use_eval_prompts=True)

    for idx, segment in enumerate(segments):
        if progress and task_id is not None and total_segments > 0:
            progress.update(
                task_id,
                completed=int((idx / total_segments) * len(conversation)),
                status=f"Unit {idx + 1}/{total_segments}"
            )

        set_activity_id(f"add-v4-{conv_id}-ep{idx}")

        # Build base MemUnit
        memunit = build_memunit_from_segment(segment, speakers, idx, conversation)

        # Generate narrative (episode) from original_data
        episode_request = EpisodeMemoryExtractRequest(
            memunit_list=[memunit],
            user_id_list=list(speakers),
            participants=list(speakers),
        )

        try:
            episode_result = await episode_extractor.extract_memory(
                episode_request, use_group_prompt=True
            )
            memunit.narrative = episode_result.narrative
            memunit.subject = episode_result.subject
            memunit.summary = episode_result.narrative[:200] + "..." if len(episode_result.narrative) > 200 else episode_result.narrative
        except Exception as e:
            console.print(f"[yellow]⚠️ Episode extraction failed for unit {idx}: {e}[/yellow]")
            # Fallback: use segment summary
            memunit.narrative = segment.get("summary", "")
            memunit.subject = segment.get("summary", "")[:50]

        memunit_list.append(memunit)

    # Update progress: complete
    if progress and task_id is not None:
        progress.update(task_id, completed=len(conversation), status="✅")

    return memunit_list


async def process_single_conversation_v4(
    conv_id: str,
    conversation: List[Dict[str, Any]],
    save_dir: str,
    llm_provider: LLMProvider,
    event_log_extractor: EventLogExtractor = None,
    progress_counter: dict = None,
    progress: Progress = None,
    task_id: int = None,
    config: Any = None,
) -> tuple:
    """Process a single conversation using V4 fine-grained extraction.

    Args:
        conv_id: Conversation ID.
        conversation: List of message dicts.
        save_dir: Directory to save results.
        llm_provider: Shared LLM provider.
        event_log_extractor: Optional event log extractor.
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
            progress.update(task_id, status="Processing...")

        # Extract MemUnits using V4 method
        memunit_list = await memunit_extraction_v4(
            conversation=conversation,
            llm_provider=llm_provider,
            conv_id=conv_id,
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

        # Extract event logs concurrently (if enabled)
        if event_log_extractor:
            memunits_with_narrative = [
                (idx, mu) for idx, mu in enumerate(memunit_list)
                if hasattr(mu, 'narrative') and mu.narrative
            ]

            async def extract_event_log(idx: int, memunit: MemUnit):
                set_activity_id(f"add-v4-{conv_id}-el{idx}")
                try:
                    event_log = await event_log_extractor.extract_event_log(
                        episode_text=memunit.narrative,
                        timestamp=memunit.timestamp
                    )
                    return idx, event_log
                except Exception as e:
                    console.print(f"[yellow]⚠️ Event log failed (unit {idx}): {e}[/yellow]")
                    return idx, None

            max_concurrent = config.get("concurrency.extraction", 5) if config else 5
            sem = asyncio.Semaphore(max_concurrent)

            async def extract_with_sem(idx, mu):
                async with sem:
                    return await extract_event_log(idx, mu)

            event_log_tasks = [extract_with_sem(idx, mu) for idx, mu in memunits_with_narrative]
            event_log_results = await asyncio.gather(*event_log_tasks)

            for orig_idx, event_log in event_log_results:
                if event_log:
                    memunit_list[orig_idx].event_log = event_log

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
            "pipeline_version": "v4",
            "granularity": "fine-grained",
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
