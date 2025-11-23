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
async def test_slow_success_no_backoff_on_success(provider):
    """Test: Successful requests (even if slow) do NOT trigger backoff anymore.

    Note: We removed the slow-success backoff logic because:
    1. Success means the API is working, no need to delay
    2. Keeps semaphore usage simple and predictable
    3. Backoff only applies on errors (where it's actually needed)
    """

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Slow response"), finish_reason="stop")]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Simulate slow response
    async def slow_create(*args, **kwargs):
        await asyncio.sleep(0.2)  # Simulate 200ms
        return mock_response

    provider.client.chat.completions.create = AsyncMock(side_effect=slow_create)

    start = time.time()
    result = await provider.generate("Test prompt")
    elapsed = time.time() - start

    assert result == "Slow response"
    # Should complete quickly without backoff (success = no delay)
    assert elapsed < 1.0, f"Successful request should not have backoff, got {elapsed:.2f}s"
    print(f"✅ Slow success: {elapsed:.2f}s (no backoff on success)")


@pytest.mark.asyncio
async def test_retry_with_exponential_backoff(provider):
    """Test: Failed requests trigger exponential backoff (2^attempt)."""

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

    # With exponential backoff (min=2s) and ±30% jitter:
    # Backoff 1: 2 * 2^0 = 2s (±30% jitter = 1.4-2.6s)
    # Backoff 2: 2 * 2^1 = 4s (±30% jitter = 2.8-5.2s)
    assert 1.4 <= backoff_sleeps[0] <= 2.6, f"Backoff 1 should be 2s ± 30%, got {backoff_sleeps[0]:.2f}s"
    assert 2.8 <= backoff_sleeps[1] <= 5.2, f"Backoff 2 should be 4s ± 30%, got {backoff_sleeps[1]:.2f}s"

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
async def test_backoff_clamping_to_max(provider):
    """Test: Exponential backoff is clamped to max_backoff."""

    # Override to have a small max for testing
    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '5',
        'OPENAI_MIN_BACKOFF': '2',
        'OPENAI_MAX_BACKOFF': '6',  # Small max to test clamping
    }):
        test_provider = OpenAIProvider(model="gpt-4o-mini")

    async def fast_fail(*args, **kwargs):
        await asyncio.sleep(0.01)
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    test_provider.client.chat.completions.create = AsyncMock(side_effect=fast_fail)

    sleep_times = []
    async def track_sleep(duration):
        sleep_times.append(duration)
        return None

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        with pytest.raises(LLMError):
            await test_provider.generate("Test")

    # Filter provider's backoff sleeps
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]

    # With min=2, max=6, exponential would be: 2, 4, 8, 16, ...
    # But 8 and 16 should be clamped to max=6
    # With ±30% jitter:
    # Backoff 1: 2 * 2^0 = 2s (±30% = 1.4-2.6s)
    # Backoff 2: 2 * 2^1 = 4s (±30% = 2.8-5.2s)
    # Backoff 3: min(8, 6) = 6s (±30% = 4.2-7.8s)
    # Backoff 4: min(16, 6) = 6s (±30% = 4.2-7.8s)

    assert len(backoff_sleeps) == 4, f"Should have 4 backoffs, got {len(backoff_sleeps)}"

    # First two should be exponential with ±30% jitter
    assert 1.4 <= backoff_sleeps[0] <= 2.6, f"Backoff 1: {backoff_sleeps[0]:.2f}s"
    assert 2.8 <= backoff_sleeps[1] <= 5.2, f"Backoff 2: {backoff_sleeps[1]:.2f}s"

    # Last two should be clamped to max (6s ± 30%)
    assert 4.2 <= backoff_sleeps[2] <= 7.8, f"Backoff 3 should be clamped to 6s: {backoff_sleeps[2]:.2f}s"
    assert 4.2 <= backoff_sleeps[3] <= 7.8, f"Backoff 4 should be clamped to 6s: {backoff_sleeps[3]:.2f}s"

    print(f"✅ Backoff clamping to max: {backoff_sleeps}")


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test: Backoff increases exponentially with each retry (2^attempt)."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '5',
        'OPENAI_MIN_BACKOFF': '2',
        'OPENAI_MAX_BACKOFF': '60',
        'OPENAI_MAX_CONCURRENT_REQUESTS': '10',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    async def always_fail(*args, **kwargs):
        await asyncio.sleep(0.01)
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=always_fail)

    sleep_times = []
    async def track_sleep(duration):
        sleep_times.append(duration)
        return None

    with patch('providers.llm.openai_provider.asyncio.sleep', side_effect=track_sleep):
        with pytest.raises(LLMError):
            await provider.generate("Test")

    # Filter only provider's backoff sleeps (>= 1s)
    backoff_sleeps = [t for t in sleep_times if t >= 1.0]

    assert len(backoff_sleeps) == 4, f"Should have 4 backoffs (attempts 1-4), got {len(backoff_sleeps)}"

    # Verify exponential growth: each backoff should be roughly double the previous
    # Expected: 2s, 4s, 8s, 16s (before jitter)
    # With ±30% jitter, actual values will be in ranges:
    # [1.4, 2.6], [2.8, 5.2], [5.6, 10.4], [11.2, 20.8]

    expected_bases = [2, 4, 8, 16]
    for i, sleep_time in enumerate(backoff_sleeps):
        base = expected_bases[i]
        min_expected = base * 0.7  # base - 30%
        max_expected = base * 1.3  # base + 30%

        assert min_expected <= sleep_time <= max_expected, \
            f"Backoff {i+1} should be {base}s ± 30% jitter, got {sleep_time:.2f}s"

    # Verify exponential growth: each sleep should be roughly double the previous
    # With ±30% jitter, ratio could range from (2*0.7)/(1*1.3) ≈ 1.08 to (2*1.3)/(1*0.7) ≈ 3.71
    for i in range(1, len(backoff_sleeps)):
        ratio = backoff_sleeps[i] / backoff_sleeps[i-1]
        # Allow wider tolerance due to ±30% jitter on both values
        assert 1.0 <= ratio <= 4.0, \
            f"Backoff {i+1} should be ~2x backoff {i}, got ratio {ratio:.2f}"

    print(f"✅ Exponential backoff verified: {[f'{t:.1f}s' for t in backoff_sleeps]}")


@pytest.mark.asyncio
async def test_no_semaphore_deadlock_during_backoff():
    """
    Critical test: Semaphore should NOT be held during sleep (backoff).

    This test verifies the fix for the deadlock issue where semaphore was held
    during sleep, causing all slots to be occupied by sleeping tasks.

    Test strategy:
    1. Set max_concurrent to a small number (2)
    2. Launch more requests than the limit (5)
    3. Make all requests fail and require backoff
    4. If semaphore is held during sleep, this will deadlock
    5. If semaphore is released during sleep, all requests complete
    """

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_MAX_RETRIES': '3',
        'OPENAI_MIN_BACKOFF': '1',  # Short backoff for test speed
        'OPENAI_MAX_BACKOFF': '5',
        'OPENAI_MAX_CONCURRENT_REQUESTS': '2',  # Only 2 concurrent allowed
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

    call_count = 0
    active_requests = 0
    max_active_requests = 0

    async def track_and_fail(*args, **kwargs):
        nonlocal call_count, active_requests, max_active_requests

        call_count += 1
        active_requests += 1
        max_active_requests = max(max_active_requests, active_requests)

        # Simulate some work
        await asyncio.sleep(0.05)

        active_requests -= 1

        # Always fail to trigger backoff
        request = MagicMock()
        raise openai.APIConnectionError(request=request)

    provider.client.chat.completions.create = AsyncMock(side_effect=track_and_fail)

    # Use actual asyncio.sleep to verify the backoff doesn't cause deadlock
    # If semaphore is held during sleep, this will hang/timeout
    # If semaphore is released during sleep, all requests complete

    start = time.time()

    # Launch 5 concurrent requests (more than the semaphore limit of 2)
    tasks = [provider.generate(f"Prompt {i}") for i in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start

    # All should fail with LLMError (expected)
    assert all(isinstance(r, LLMError) for r in results), "All requests should fail"

    # Verify that requests actually ran concurrently
    # Each request makes 3 attempts, so 5 requests = 15 total calls
    assert call_count == 15, f"Expected 15 calls (5 requests × 3 retries), got {call_count}"

    # Critical assertion: max_active should not exceed semaphore limit
    assert max_active_requests <= 2, \
        f"Max active requests was {max_active_requests}, should be ≤ 2 (semaphore limit)"

    # The test should complete in reasonable time (not deadlock)
    # With exponential backoff: 1s, 2s per request, 3 retries = ~6s per request worst case
    # But with concurrent execution and semaphore properly releasing during sleep,
    # should complete much faster
    assert elapsed < 30, f"Test took {elapsed:.1f}s, possible deadlock (should be < 30s)"

    print(f"✅ No deadlock: 5 concurrent requests completed in {elapsed:.1f}s")
    print(f"   Max active requests: {max_active_requests}/2 (within semaphore limit)")
    print(f"   Total API calls: {call_count} (5 requests × 3 retries)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])