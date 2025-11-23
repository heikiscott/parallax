"""
Test OpenAI Provider adaptive backoff logic.

Tests verify:
1. Fast success (< threshold): no delay
2. Slow success (≥ threshold): delay applied
3. Failed request: delay based on duration
4. Multiple retries: progressive backoff
"""

import asyncio
import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import openai

# Add src to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from providers.llm.openai_provider import OpenAIProvider
from providers.llm.protocol import LLMError


@pytest.fixture
def provider():
    """Create a test provider with known config."""
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_TIMEOUT': '30',
        'OPENAI_SDK_MAX_RETRIES': '0',
        'OPENAI_MAX_RETRIES': '3',
        'OPENAI_BACKOFF_FACTOR': '0.5',
        'OPENAI_MIN_BACKOFF': '2',
        'OPENAI_MAX_BACKOFF': '10',
        'OPENAI_SLOW_THRESHOLD': '5',
        'OPENAI_MAX_CONCURRENT_REQUESTS': '10',
    }):
        return OpenAIProvider(model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_fast_success_no_delay(provider):
    """Test: Fast success (< 5s) should not trigger delay."""

    # Mock successful fast response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Fast response"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Simulate fast response (2 seconds)
    async def fast_create(*args, **kwargs):
        await asyncio.sleep(0.1)  # Simulate 100ms network delay
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=fast_create)

    start = time.perf_counter()
    result = await provider.generate("Test prompt")
    elapsed = time.perf_counter() - start

    assert result == "Fast response"
    # Should complete in ~0.1s, no additional delay
    assert elapsed < 1.0, f"Fast request took {elapsed:.2f}s, should be < 1s"
    print(f"✅ Fast success: {elapsed:.2f}s (no delay)")


@pytest.mark.asyncio
async def test_slow_success_with_delay(provider):
    """Test: Slow success (≥ 5s) should trigger backoff delay."""

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Slow response"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Simulate slow response (6 seconds)
    async def slow_create(*args, **kwargs):
        await asyncio.sleep(0.2)  # Simulate 200ms but we'll mock the duration
        return mock_response

    with patch('time.perf_counter') as mock_time:
        # Mock timing to show 6 seconds elapsed
        mock_time.side_effect = [0, 6, 6]  # start, end, (next check)

        provider.client.chat.completions.create = AsyncMock(side_effect=slow_create)

        start = time.time()
        result = await provider.generate("Test prompt")
        elapsed = time.time() - start

        assert result == "Slow response"
        # Should have delay = 6 * 0.5 = 3 seconds
        # Total time ≈ 0.2s (request) + 3s (backoff) = 3.2s
        assert elapsed >= 2.5, f"Slow request should have backoff delay, got {elapsed:.2f}s"
        print(f"✅ Slow success: {elapsed:.2f}s (includes backoff)")


@pytest.mark.asyncio
async def test_retry_with_adaptive_backoff(provider):
    """Test: Failed requests trigger adaptive backoff based on duration."""

    call_count = 0

    async def failing_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # First two attempts fail, third succeeds
        if call_count < 3:
            await asyncio.sleep(0.1)
            # Create proper APIConnectionError with request object
            request = MagicMock()
            raise openai.APIConnectionError(request=request)

        # Third attempt succeeds
        await asyncio.sleep(0.05)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=failing_create)

    # Track sleep calls to verify backoff
    sleep_times = []

    async def track_sleep(duration):
        sleep_times.append(duration)
        # Don't actually sleep in test
        return None

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        start = time.time()
        result = await provider.generate("Test prompt")
        elapsed = time.time() - start

    assert result == "Success"
    assert call_count == 3, f"Should have 3 attempts, got {call_count}"

    # Filter out test's own sleeps (< 1s), keep only provider's backoff sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]
    assert len(backoff_sleeps) == 2, f"Should have 2 backoffs, got {len(backoff_sleeps)}"

    # Verify backoff times are reasonable (with jitter: base * 1.3)
    # First backoff: based on ~0.1s failure, so 0.1 * 0.5 = 0.05, clamped to min 2s, + jitter
    # Second backoff: similar
    for i, sleep_time in enumerate(backoff_sleeps):
        assert sleep_time >= 2.0, f"Backoff {i+1} should be >= min_backoff (2s), got {sleep_time:.2f}s"
        assert sleep_time <= 2.6, f"Backoff {i+1} should be <= min+jitter (2*1.3), got {sleep_time:.2f}s"

    print(f"✅ Retry backoff: attempts={call_count}, backoffs={backoff_sleeps}")


@pytest.mark.asyncio
async def test_all_retries_exhausted(provider):
    """Test: After all retries fail, raise LLMError."""

    async def always_fail(*args, **kwargs):
        await asyncio.sleep(0.05)
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=always_fail)

    # Mock sleep to avoid waiting in test
    async def mock_sleep(duration):
        return None

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=mock_sleep):
        with pytest.raises(LLMError, match="OpenAI API error"):
            await provider.generate("Test prompt")

    # Should have tried max_retries times (3)
    assert provider.client.chat.completions.create.call_count == 3
    print(f"✅ All retries exhausted correctly")


@pytest.mark.asyncio
async def test_concurrency_limit(provider):
    """Test: Semaphore limits concurrent requests."""

    active_count = 0
    max_active = 0

    async def slow_create(*args, **kwargs):
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)

        await asyncio.sleep(0.1)

        active_count -= 1

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=slow_create)

    # Launch 20 concurrent requests
    tasks = [provider.generate(f"Prompt {i}") for i in range(20)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 20
    assert all(r == "Response" for r in results)
    # Max concurrent should not exceed semaphore limit (10)
    assert max_active <= 10, f"Max concurrent was {max_active}, should be ≤ 10"
    print(f"✅ Concurrency limit respected: max_active={max_active}/10")


@pytest.mark.asyncio
async def test_backoff_clamping(provider):
    """Test: Backoff time is clamped to [min, max]."""

    # Test min clamping: very fast failure (0.5s)
    async def fast_fail(*args, **kwargs):
        await asyncio.sleep(0.01)
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=fast_fail)

    sleep_times = []
    async def track_sleep(duration):
        sleep_times.append(duration)
        return None

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        with pytest.raises(LLMError):
            await provider.generate("Test")

    # Filter out test's own sleeps, keep only provider's backoff sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]

    # All backoffs should be clamped to min (2s) + jitter (up to 30%)
    for sleep_time in backoff_sleeps:
        assert sleep_time >= 2.0, f"Backoff should be >= min (2s), got {sleep_time:.2f}s"
        assert sleep_time <= 2.6, f"Backoff should be <= min+jitter (2*1.3), got {sleep_time:.2f}s"

    print(f"✅ Backoff clamping: {backoff_sleeps}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])