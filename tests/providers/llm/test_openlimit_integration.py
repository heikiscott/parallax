"""
Simple integration test for OpenAI Provider with openlimit.

This test verifies that the provider correctly initializes and uses openlimit.
"""

import pytest
from unittest.mock import patch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from providers.llm.openai_provider import OpenAIProvider


def test_provider_initialization_with_openlimit():
    """Test: Provider initializes with openlimit correctly."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
        'OPENAI_TIMEOUT': '30',
        'OPENAI_REQUEST_LIMIT': '500',
        'OPENAI_TOKEN_LIMIT': '150000',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")

        # Verify provider has rate limiter
        assert hasattr(provider, 'rate_limiter'), "Provider should have rate_limiter attribute"
        assert provider.rate_limiter is not None, "Rate limiter should not be None"

        # Verify configuration
        assert provider.request_limit == 500, f"Expected request_limit=500, got {provider.request_limit}"
        assert provider.token_limit == 150000, f"Expected token_limit=150000, got {provider.token_limit}"

        # Verify rate limiter configuration
        # Note: Rate limiter uses very high token_limit (999999999) to avoid deadlock,
        # even though provider.token_limit is 150000 for logging purposes
        assert provider.rate_limiter.request_limit == 500, f"Rate limiter request_limit should be 500"
        assert provider.rate_limiter.token_limit == 999999999, f"Rate limiter token_limit should be 999999999 (to prevent deadlock)"

        # Verify rate limiter type
        from openlimit import ChatRateLimiter
        assert isinstance(provider.rate_limiter, ChatRateLimiter), "Rate limiter should be ChatRateLimiter instance"

        print(f"✅ Provider initialized correctly")
        print(f"   Model: {provider.model}")
        print(f"   Request limit: {provider.request_limit} RPM")
        print(f"   Token limit: {provider.token_limit} TPM")
        print(f"   Rate limiter: {type(provider.rate_limiter).__name__}")


def test_provider_uses_environment_defaults():
    """Test: Provider uses correct defaults from environment."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
    }, clear=False):  # Don't clear other env vars
        provider = OpenAIProvider(model="gpt-4o-mini")

        # Should use default values
        assert provider.request_limit > 0, "Should have positive request limit"
        assert provider.token_limit > 0, "Should have positive token limit"

        print(f"✅ Provider uses defaults correctly")
        print(f"   Default request_limit: {provider.request_limit}")
        print(f"   Default token_limit: {provider.token_limit}")


def test_provider_repr():
    """Test: Provider string representation mentions openlimit."""

    with patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key',
    }):
        provider = OpenAIProvider(model="gpt-4o-mini")
        repr_str = repr(provider)

        assert 'openlimit' in repr_str, f"Provider repr should mention openlimit: {repr_str}"
        assert 'gpt-4o-mini' in repr_str, f"Provider repr should mention model: {repr_str}"

        print(f"✅ Provider repr correct: {repr_str}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
