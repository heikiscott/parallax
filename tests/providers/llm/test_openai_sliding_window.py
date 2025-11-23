"""
Test OpenAI Provider sliding window backoff logic.

Tests verify:
1. Backoff based on sliding window failure count (not attempt number)
2. No deadlock between semaphore and _request_lock
3. Success/failure tracking in sliding window
4. Exponential backoff: 1 * (2 ^ min(failures, 9))
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
        'OPENAI_MAX_RETRIES': '10',
        'OPENAI_MAX_BACKOFF': '512',
        'OPENAI_CONCURRENT_REQUESTS': '10',
    }):
        return OpenAIProvider(model="gpt-4o-mini")


@pytest.mark.asyncio
async def test_success_no_backoff(provider):
    """Test: Successful requests don't trigger backoff."""

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    async def fast_create(*args, **kwargs):
        await asyncio.sleep(0.01)
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=fast_create)

    start = time.perf_counter()
    result = await provider.generate("Test")
    elapsed = time.perf_counter() - start

    assert result == "Success"
    assert elapsed < 0.5, f"Success should be fast, got {elapsed:.2f}s"

    # Check sliding window recorded success
    assert len(provider._recent_requests) == 1
    assert provider._recent_requests[0] == 'success'

    print(f"✅ Success: {elapsed:.2f}s, sliding window: {list(provider._recent_requests)}")


@pytest.mark.asyncio
async def test_sliding_window_backoff():
    """Test: Backoff based on sliding window failure count.

    Each retry failure is recorded immediately, so backoff grows as failures accumulate.
    """

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '5',
        'OPENAI_MAX_BACKOFF': '512',
        'OPENAI_CONCURRENT_REQUESTS': '10',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    call_count = 0

    async def failing_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # Fail 3 times, then succeed
        if call_count <= 3:
            await asyncio.sleep(0.01)
            request = MagicMock()
            raise openai.APIConnectionError(request=request)

        # Success on 4th attempt
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=failing_create)

    sleep_times = []
    original_sleep = asyncio.sleep

    async def track_sleep(duration):
        sleep_times.append(duration)
        await original_sleep(0.001)

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        result = await provider.generate("Test")

    assert result == "Success"
    assert call_count == 4

    # Check sliding window: 3 failures + 1 success
    assert len(provider._recent_requests) == 4
    assert list(provider._recent_requests) == ['failure', 'failure', 'failure', 'success']

    # Filter out mock API sleeps (0.01s), keep only backoff sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]
    assert len(backoff_sleeps) == 3, f"Expected 3 backoffs, got {len(backoff_sleeps)}: {sleep_times}"

    # Backoff should increase based on sliding window failure count:
    # After 1st failure: 1 failure → 2^1 = 2s + jitter(0-5s) = 2-7s
    # After 2nd failure: 2 failures → 2^2 = 4s + jitter(0-5s) = 4-9s
    # After 3rd failure: 3 failures → 2^3 = 8s + jitter(0-5s) = 8-13s

    assert 2 <= backoff_sleeps[0] <= 7, f"Backoff 1: {backoff_sleeps[0]:.1f}s (expected 2-7s)"
    assert 4 <= backoff_sleeps[1] <= 9, f"Backoff 2: {backoff_sleeps[1]:.1f}s (expected 4-9s)"
    assert 8 <= backoff_sleeps[2] <= 13, f"Backoff 3: {backoff_sleeps[2]:.1f}s (expected 8-13s)"

    print(f"✅ Sliding window backoff: {[f'{t:.1f}s' for t in backoff_sleeps]}")
    print(f"   Window: {list(provider._recent_requests)}")


@pytest.mark.asyncio
async def test_sliding_window_caps_at_9():
    """Test: Sliding window caps backoff at 9 failures (2^9 = 512s)."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '12',
        'OPENAI_MAX_BACKOFF': '600',  # Higher than 512 to test the cap
        'OPENAI_CONCURRENT_REQUESTS': '10',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    async def always_fail(*args, **kwargs):
        await asyncio.sleep(0.01)
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=always_fail)

    sleep_times = []
    original_sleep = asyncio.sleep

    async def track_sleep(duration):
        sleep_times.append(duration)
        await original_sleep(0.001)

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        with pytest.raises(LLMError):
            await provider.generate("Test")

    # Filter out mock API sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]

    # Should have 11 backoffs (attempts 1-11, before final attempt 12)
    assert len(backoff_sleeps) == 11

    # Last 3 backoffs should all be ~512s (2^9, capped)
    for i in range(-3, 0):  # Last 3 backoffs
        assert 512 <= backoff_sleeps[i] <= 517, \
            f"Backoff {len(backoff_sleeps)+i+1} should be capped at 512s, got {backoff_sleeps[i]:.1f}s"

    print(f"✅ Capping at 9 failures: last 3 backoffs = {[f'{t:.1f}s' for t in backoff_sleeps[-3:]]}")


@pytest.mark.asyncio
async def test_no_deadlock_semaphore_and_lock():
    """
    Critical test: No deadlock between semaphore and _request_lock.

    This verifies the fix where success is recorded AFTER releasing semaphore,
    not while holding it.

    Test strategy:
    1. Set small concurrency limit (2)
    2. Launch many concurrent requests (10)
    3. Mix successes and failures
    4. If there's a deadlock, test will hang
    5. If fixed, all requests complete
    """

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '3',
        'OPENAI_MAX_BACKOFF': '10',
        'OPENAI_CONCURRENT_REQUESTS': '2',  # Only 2 concurrent
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    call_count = 0
    active_count = 0
    max_active = 0

    async def mixed_responses(*args, **kwargs):
        nonlocal call_count, active_count, max_active

        call_count += 1
        active_count += 1
        max_active = max(max_active, active_count)

        await asyncio.sleep(0.05)

        active_count -= 1

        # 50% success, 50% failure
        if call_count % 2 == 0:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            return mock_response
        else:
            request = MagicMock()
            raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=mixed_responses)

    # Mock sleep to make test faster
    original_sleep = asyncio.sleep

    async def fast_sleep(duration):
        await original_sleep(0.001)  # Very short sleep

    start = time.time()

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=fast_sleep):
        # Launch 10 concurrent requests
        tasks = [provider.generate(f"Prompt {i}") for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start

    # Some succeed, some fail
    successes = [r for r in results if isinstance(r, str)]
    failures = [r for r in results if isinstance(r, LLMError)]

    assert len(successes) + len(failures) == 10
    assert len(successes) > 0, "Should have some successes"

    # Verify concurrency limit respected
    assert max_active <= 2, f"Max active was {max_active}, should be ≤ 2"

    # Should complete quickly (no deadlock)
    assert elapsed < 10, f"Took {elapsed:.1f}s, possible deadlock"

    print(f"✅ No deadlock: {len(successes)} successes, {len(failures)} failures")
    print(f"   Completed in {elapsed:.1f}s, max_active={max_active}/2")


@pytest.mark.asyncio
async def test_sliding_window_max_size_20():
    """Test: Sliding window maintains max 20 entries."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '1',
        'OPENAI_MAX_BACKOFF': '10',
        'OPENAI_CONCURRENT_REQUESTS': '50',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    async def fast_success(*args, **kwargs):
        await asyncio.sleep(0.001)
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=fast_success)

    # Make 30 successful requests
    tasks = [provider.generate(f"Prompt {i}") for i in range(30)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 30

    # Sliding window should only keep last 20
    assert len(provider._recent_requests) == 20
    assert all(r == 'success' for r in provider._recent_requests)

    print(f"✅ Sliding window capped at 20: {len(provider._recent_requests)} entries")


@pytest.mark.asyncio
async def test_window_based_not_attempt_based():
    """Test: Backoff is window-based, not attempt-based.

    This is the key difference from old implementation:
    - Old: backoff based on attempt number (resets per request)
    - New: backoff based on window failure count (persists across requests)
    """

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '2',
        'OPENAI_MAX_BACKOFF': '100',
        'OPENAI_CONCURRENT_REQUESTS': '1',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    call_count = 0

    async def controlled_responses(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        await asyncio.sleep(0.01)

        # First request: fail
        if call_count == 1:
            request = MagicMock()
            raise openai.APIConnectionError(request=request)
        # First retry: succeed
        else:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Success"), finish_reason="stop")]
            mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=controlled_responses)

    sleep_times = []
    original_sleep = asyncio.sleep

    async def track_sleep(duration):
        sleep_times.append(duration)
        await original_sleep(0.001)

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        # First request: will fail once, then succeed
        result1 = await provider.generate("Request 1")
        assert result1 == "Success"

        # Window now has: ['failure', 'success']
        assert len(provider._recent_requests) == 2

        # Second request: should succeed on first try
        result2 = await provider.generate("Request 2")
        assert result2 == "Success"

        # Window now has: ['failure', 'success', 'success']
        assert len(provider._recent_requests) == 3

    # Filter backoff sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]

    # Should have exactly 1 backoff (after first failure)
    # Backoff = 2^1 = 2s + jitter(0-5s)
    assert len(backoff_sleeps) == 1
    assert 2 <= backoff_sleeps[0] <= 7

    print(f"✅ Window-based backoff: {backoff_sleeps[0]:.1f}s")
    print(f"   Final window: {list(provider._recent_requests)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
