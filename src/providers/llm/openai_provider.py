"""
OpenAI official API provider implementation with openlimit.

This provider uses official OpenAI Python SDK (openai.AsyncOpenAI) with openlimit
for intelligent rate limiting and token management.
"""

import os
import time
import asyncio
import httpx
import openai
from typing import Optional
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from aiolimiter import AsyncLimiter

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Official OpenAI API provider using openai.AsyncOpenAI with proactive rate limiting.

    This provider uses a multi-layer rate limiting strategy:
    1. HTTP connection pool (httpx) - large connection pool prevents TCP exhaustion
    2. Physical concurrency control (Semaphore) - limits concurrent requests
    3. Logical rate limiting (AsyncLimiter) - prevents 429 rate limit errors
    4. Exponential backoff retry (tenacity) - handles transient errors

    This "proactive throttling" approach prevents most 429 errors before they occur,
    maintaining high throughput without blocking due to excessive retries.
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
            OPENAI_TIMEOUT: API timeout in seconds (default: 40)
            OPENAI_MAX_RETRIES: SDK-level retry attempts (default: 0, disabled)
            OPENAI_MAX_CONCURRENT: Physical concurrency limit (default: 20)
            OPENAI_RPM_LIMIT: Requests per minute rate limit (default: 500)
            OPENAI_RETRY_MIN_WAIT: Minimum retry wait time in seconds (default: 5)
            OPENAI_RETRY_MAX_WAIT: Maximum retry wait time in seconds (default: 60)
            OPENAI_RETRY_ATTEMPTS: Maximum retry attempts (default: 20)
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
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "0"))  # Disable SDK retries, let eval framework handle it

        # Provider-level concurrency control (shared across all instances)
        max_concurrent = int(os.getenv("OPENAI_MAX_CONCURRENT", "20"))

        # Rate limiting configuration
        # Set to 80% of your OpenAI tier limit to leave safety margin
        # Common tiers: Free (~60 RPM), Tier 1 (~500 RPM), Tier 2+ (~5000 RPM)
        rpm_limit = int(os.getenv("OPENAI_RPM_LIMIT", "500"))

        # Configure httpx client with larger connection pool
        # This prevents TCP connection exhaustion at the HTTP client layer
        # max_keepalive_connections: Number of connections to keep alive for reuse
        # max_connections: Total number of connections allowed (should be >= max_concurrent)
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_keepalive_connections=max_concurrent,  # Match semaphore limit
                max_connections=max_concurrent * 4,  # 4x buffer for safety
            ),
            timeout=httpx.Timeout(self.timeout, connect=10.0),  # Bottom-layer timeout
        )

        # Initialize official OpenAI async client with custom httpx client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,  # Disable SDK-level retries
            http_client=http_client,  # Inject custom httpx client with larger connection pool
        )

        # Dual-layer rate limiting strategy:
        # Layer 1: Physical concurrency control (Semaphore) - prevents TCP connection exhaustion
        # Layer 2: Logical rate limiting (AsyncLimiter) - prevents 429 errors via proactive throttling
        # Layer 3: Exponential backoff retry (tenacity) - handles transient errors

        # Class-level semaphore for concurrency control (shared across all provider instances)
        # This ensures global concurrency limit even with multiple provider instances
        if not hasattr(OpenAIProvider, '_semaphore') or not hasattr(OpenAIProvider, '_semaphore_size'):
            OpenAIProvider._semaphore = asyncio.Semaphore(max_concurrent)
            OpenAIProvider._semaphore_size = max_concurrent
        elif OpenAIProvider._semaphore_size != max_concurrent:
            # If max_concurrent changed, recreate semaphore
            OpenAIProvider._semaphore = asyncio.Semaphore(max_concurrent)
            OpenAIProvider._semaphore_size = max_concurrent

        self.semaphore = OpenAIProvider._semaphore

        # Class-level rate limiter for proactive rate limiting (shared across all instances)
        # This prevents 429 errors by "braking before hitting the wall"
        if not hasattr(OpenAIProvider, '_rate_limiter') or not hasattr(OpenAIProvider, '_rpm_limit'):
            OpenAIProvider._rate_limiter = AsyncLimiter(max_rate=rpm_limit, time_period=60)
            OpenAIProvider._rpm_limit = rpm_limit
        elif OpenAIProvider._rpm_limit != rpm_limit:
            # If RPM limit changed, recreate rate limiter
            OpenAIProvider._rate_limiter = AsyncLimiter(max_rate=rpm_limit, time_period=60)
            OpenAIProvider._rpm_limit = rpm_limit

        self.rate_limiter = OpenAIProvider._rate_limiter

        # Optional statistics tracking
        if self.enable_stats:
            self.current_call_stats = None

        logger.info(f"Initialized OpenAIProvider with model={self.model}")
        logger.info(f"  Timeout: {self.timeout}s")
        logger.info(f"  Max retries (SDK): {self.max_retries}")
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  RPM limit: {rpm_limit} (proactive rate limiting)")
        logger.info(f"  HTTPX pool: keepalive={max_concurrent}, max={max_concurrent * 4}")

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
        # Read retry configuration from environment
        retry_min_wait = int(os.getenv("OPENAI_RETRY_MIN_WAIT", "1"))
        retry_max_wait = int(os.getenv("OPENAI_RETRY_MAX_WAIT", "60"))
        retry_attempts = int(os.getenv("OPENAI_RETRY_ATTEMPTS", "5"))

        try:
            # Layer 1: Physical concurrency control (outside retry loop)
            # This limits total concurrent requests to prevent TCP connection exhaustion
            async with self.semaphore:

                # Layer 2: Proactive rate limiting (before retry loop)
                # This prevents 429 errors by enforcing RPM limit BEFORE sending requests
                # Wait here in an orderly queue instead of hitting 429 and sleeping chaotically
                async with self.rate_limiter:
                    # Start timing AFTER acquiring both semaphore and rate limiter
                    # (exclude queue wait time from execution duration)
                    start_time = time.perf_counter()

                    # Layer 3: Exponential backoff retry for transient errors
                    # Should rarely trigger 429 now thanks to proactive rate limiting
                    async for attempt in AsyncRetrying(
                        wait=wait_random_exponential(min=retry_min_wait, max=retry_max_wait),
                        stop=stop_after_attempt(retry_attempts),
                        retry=retry_if_exception_type((
                            openai.RateLimitError,
                            openai.APIConnectionError,
                            openai.APIError,
                            openai.APITimeoutError,
                        )),
                        reraise=True
                    ):
                        with attempt:
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

                                # Make API call
                                # Dual-layer rate limiting now active:
                                # Layer 1: Semaphore (physical concurrency)
                                # Layer 2: AsyncLimiter (proactive RPM throttling)
                                # Layer 3: tenacity retry (exponential backoff for transient errors)
                                #
                                # Explicitly set timeout (double insurance, even though client has default timeout)
                                # This prevents zombie tasks from occupying semaphore slots indefinitely
                                response = await self.client.chat.completions.create(
                                    **params,
                                    timeout=self.timeout  # Use timeout from environment variable
                                )

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

                            except (openai.RateLimitError, openai.APIConnectionError, openai.APIError, openai.APITimeoutError) as e:
                                # 🚨 CRITICAL WARNING: If you see this log frequently, your OPENAI_RPM_LIMIT is set too high!
                                # The proactive rate limiter should prevent most 429 errors.
                                if isinstance(e, openai.RateLimitError):
                                    logger.warning(f"⚠️  [OpenAI-{self.model}] 依然触发了 429 RateLimitError!")
                                    logger.warning(f"   这说明 OPENAI_RPM_LIMIT 设置过高，请降低到你的 tier 限制的 80%")

                                # Log retry-able errors and let tenacity handle retry
                                error_time = time.perf_counter()
                                duration = error_time - start_time
                                error_type = type(e).__name__
                                attempt_num = attempt.retry_state.attempt_number

                                # Calculate next retry wait time
                                if attempt_num < retry_attempts:
                                    # Estimate next wait (exponential backoff range)
                                    next_min = min(retry_min_wait * (2 ** (attempt_num - 1)), retry_max_wait)
                                    next_max = retry_max_wait
                                    logger.info(f"[OpenAI-{self.model}] 重试 {attempt_num}/{retry_attempts}: {error_type}")
                                    logger.info(f"   ⏱️  已耗时: {duration:.2f}s, 将等待 {next_min:.1f}-{next_max:.1f}s 后重试")
                                else:
                                    logger.warning(f"[OpenAI-{self.model}] {error_type} (最后一次尝试 {attempt_num}/{retry_attempts})")
                                    logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
                                    logger.warning(f"   💬 错误: {str(e)}")

                                # Re-raise to let tenacity retry
                                raise

                            except Exception as e:
                                # Non-retryable errors: convert to LLMError and fail immediately
                                error_time = time.perf_counter()
                                duration = error_time - start_time
                                logger.error(f"[OpenAI-{self.model}] Non-retryable error")
                                logger.error(f"   ⏱️  耗时: {duration:.2f}s")
                                logger.error(f"   💬 错误信息: {str(e)}")
                                raise LLMError(f"Request failed: {str(e)}")

        except openai.RateLimitError as e:
            # Retry exhausted for rate limit
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"[OpenAI-{self.model}] RateLimitError - 重试{retry_attempts}次后仍失败")
            logger.error(f"   ⏱️  总耗时: {duration:.2f}s")
            logger.error(f"   💬 错误: {str(e)}")
            raise LLMError(f"Rate limit error after {retry_attempts} retries: {str(e)}")

        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError) as e:
            # Retry exhausted for API errors
            error_time = time.perf_counter()
            duration = error_time - start_time
            error_type = type(e).__name__
            logger.error(f"[OpenAI-{self.model}] {error_type} - 重试{retry_attempts}次后仍失败")
            logger.error(f"   ⏱️  总耗时: {duration:.2f}s")
            logger.error(f"   💬 错误: {str(e)}")
            raise LLMError(f"{error_type} after {retry_attempts} retries: {str(e)}")

        except LLMError:
            # Re-raise LLMError as-is (from non-retryable errors)
            raise

        except Exception as e:
            # Unexpected errors
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"[OpenAI-{self.model}] Unexpected error")
            logger.error(f"   ⏱️  耗时: {duration:.2f}s")
            logger.error(f"   💬 错误信息: {str(e)}")
            raise LLMError(f"Unexpected error: {str(e)}")

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
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url}, http_client=httpx+AsyncLimiter)"
