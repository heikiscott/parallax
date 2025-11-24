"""
OpenAI official API provider implementation with openlimit.

This provider uses official OpenAI Python SDK (openai.AsyncOpenAI) with openlimit
for intelligent rate limiting and token management.
"""

import os
import time
import openai
from typing import Optional
from openlimit import ChatRateLimiter

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Official OpenAI API provider using openai.AsyncOpenAI with openlimit.

    This provider uses openlimit library which provides:
    - RPM (Requests Per Minute) rate limiting
    - TPM (Tokens Per Minute) rate limiting
    - Automatic token counting and scheduling
    - Leaky bucket algorithm for smooth traffic distribution
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_stats: bool = False,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (defaults to LLM_MODEL env var or "gpt-4o-mini")
            api_key: OpenAI API key (defaults to LLM_API_KEY/OPENAI_API_KEY env var)
            base_url: OpenAI base URL (defaults to LLM_BASE_URL env var)
            temperature: Sampling temperature (defaults to LLM_TEMPERATURE env var or 0.0)
            max_tokens: Maximum tokens to generate (defaults to LLM_MAX_TOKENS env var or 16384)
            enable_stats: Enable usage statistics accumulation (default: False)
            **kwargs: Additional arguments (ignored for now)

        Environment Variables:
            LLM_MODEL: Model name (default: "gpt-4o-mini")
            LLM_API_KEY / OPENAI_API_KEY: API key (required)
            LLM_BASE_URL: Base URL (default: "https://api.openai.com/v1")
            LLM_TEMPERATURE: Sampling temperature (default: 0.0)
            LLM_MAX_TOKENS: Max completion tokens (default: 16384)
            OPENAI_TIMEOUT: API timeout in seconds (default: 30)
            OPENAI_REQUEST_LIMIT: Max requests per minute (default: 500)
            OPENAI_TOKEN_LIMIT: Max tokens per minute (default: 150000)
        """
        # Read from environment variables with fallbacks
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature if temperature is not None else float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("LLM_MAX_TOKENS", "16384"))
        self.enable_stats = enable_stats

        # Use OpenAI official API key and base URL
        # Try LLM_API_KEY first (from .env), then fall back to OPENAI_API_KEY
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"

        # Read rate limiting configuration from environment
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        self.request_limit = int(os.getenv("OPENAI_REQUEST_LIMIT", "500"))
        self.token_limit = int(os.getenv("OPENAI_TOKEN_LIMIT", "150000"))

        # Initialize official OpenAI async client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        # Initialize openlimit rate limiter
        #
        # CRITICAL DEADLOCK FIX:
        # ======================
        # openlimit's bucket capacity is calculated as: capacity = rate_limit / 60
        # This means for TPM (tokens per minute), the per-second capacity is: TPM / 60
        #
        # Problem:
        # - With token_limit=150000 TPM, bucket capacity = 2500 tokens/second
        # - Our prompts can be 3500+ tokens
        # - When a request needs more tokens than bucket capacity, openlimit's
        #   wait_for_capacity() enters an infinite loop because:
        #   1. It checks: if self._capacity < amount: return False
        #   2. Then loops forever waiting for capacity that can never be satisfied
        #
        # Solution:
        # - Set token_limit=999999999 to give bucket capacity ~16.6M tokens/second
        # - This effectively disables TPM limiting while keeping RPM limiting active
        # - RPM limiting (500 requests/minute = ~8.3 requests/second) is sufficient
        #   for rate control without risking deadlock
        #
        # Note: self.token_limit (150000) is kept for logging purposes, but the actual
        # rate limiter uses 999999999 to prevent deadlock
        self.rate_limiter = ChatRateLimiter(
            request_limit=self.request_limit,
            token_limit=999999999,  # Prevent deadlock when single request exceeds per-second capacity
        )

        # Optional statistics tracking
        if self.enable_stats:
            self.current_call_stats = None

        logger.info(f"Initialized OpenAIProvider with model={model}")
        logger.info(f"  Request limit: {self.request_limit} RPM")
        logger.info(f"  Token limit: {self.token_limit} TPM")

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a response for the given prompt with openlimit rate limiting.

        Args:
            prompt: Input prompt
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request
            extra_body: Extra parameters (not used with SDK)
            response_format: Response format specification

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        start_time = time.perf_counter()

        try:
            # Convert prompt to messages format
            messages = [{"role": "user", "content": prompt}]

            # Prepare parameters
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
            }

            # Add optional parameters
            if max_tokens is not None:
                params["max_tokens"] = int(max_tokens) if isinstance(max_tokens, str) else max_tokens
            elif self.max_tokens is not None:
                params["max_tokens"] = int(self.max_tokens) if isinstance(self.max_tokens, str) else self.max_tokens

            if response_format is not None:
                params["response_format"] = response_format

            # Rate limiting with openlimit
            # We pass model and messages to let openlimit calculate prompt tokens
            # Don't pass max_tokens to avoid over-estimation of total tokens
            # (openlimit would add max_tokens to the estimate, but actual usage is usually less)
            limiter_params = {
                "model": params["model"],
                "messages": params["messages"],
                "temperature": params["temperature"],
            }
            async with self.rate_limiter.limit(**limiter_params):
                response = await self.client.chat.completions.create(**params)

            end_time = time.perf_counter()
            duration = end_time - start_time

            # Extract result
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            # Log completion details
            if finish_reason == 'stop':
                logger.debug(f"[OpenAI-{self.model}] 完成原因: {finish_reason}")
            else:
                logger.warning(f"[OpenAI-{self.model}] 完成原因: {finish_reason}")

            # Extract token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Log usage information
            logger.debug(f"[OpenAI-{self.model}] API调用完成")
            logger.debug(f"[OpenAI-{self.model}] 耗时: {duration:.2f}s")
            logger.debug(f"[OpenAI-{self.model}] Prompt Tokens: {prompt_tokens:,}")
            logger.debug(f"[OpenAI-{self.model}] Completion Tokens: {completion_tokens:,}")
            logger.debug(f"[OpenAI-{self.model}] 总Token数: {total_tokens:,}")

            # Record statistics if enabled
            if self.enable_stats:
                self.current_call_stats = {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens,
                    'duration': duration,
                    'timestamp': time.time(),
                }

            return content

        except openai.RateLimitError as e:
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.warning(f"[OpenAI-{self.model}] RateLimitError")
            logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
            logger.warning(f"   💬 错误: {str(e)}")
            raise LLMError(f"Rate limit error: {str(e)}")

        except (openai.APIError, openai.APIConnectionError) as e:
            error_time = time.perf_counter()
            duration = error_time - start_time
            error_type = type(e).__name__
            logger.warning(f"[OpenAI-{self.model}] {error_type}")
            logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
            logger.warning(f"   💬 错误: {str(e)}")
            raise LLMError(f"API error: {str(e)}")

        except Exception as e:
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"[OpenAI-{self.model}] Unexpected error")
            logger.error(f"   ⏱️  耗时: {duration:.2f}s")
            logger.error(f"   💬 错误信息: {str(e)}")
            raise LLMError(f"Request failed: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to the OpenAI API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"🔗 [OpenAI-{self.model}] 测试API连接...")
            test_response = await self.generate("Hello", temperature=0.1)
            success = len(test_response) > 0
            if success:
                logger.info(f"✅ [OpenAI-{self.model}] API连接测试成功")
            else:
                logger.error(f"❌ [OpenAI-{self.model}] API连接测试失败: 空响应")
            return success
        except Exception as e:
            logger.error(f"❌ [OpenAI-{self.model}] API连接测试失败: {e}")
            return False

    def get_current_call_stats(self) -> Optional[dict]:
        """Get statistics for the current call (if enabled)."""
        if self.enable_stats:
            return self.current_call_stats
        return None

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url}, limiter=openlimit)"
