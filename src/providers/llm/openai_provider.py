"""
OpenAI official API provider implementation.

This provider uses official OpenAI Python SDK (openai.AsyncOpenAI).
"""

import os
import time
import asyncio
import random
import openai
from typing import Optional
from collections import deque

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(LLMProvider):
    """
    Official OpenAI API provider using openai.AsyncOpenAI.

    This provider uses the official OpenAI Python SDK which includes:
    - Automatic retry with exponential backoff
    - Rate limiting handling
    - Better error handling and connection management
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = 100 * 1024,
        enable_stats: bool = False,
        **kwargs,
    ):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o", "gpt-4-turbo")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: OpenAI base URL (defaults to official OpenAI endpoint)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_stats: Enable usage statistics accumulation (default: False)
            **kwargs: Additional arguments (ignored for now)

        Environment Variables:
            OPENAI_TIMEOUT: API timeout in seconds (default: 30)
            OPENAI_SDK_MAX_RETRIES: SDK internal retries (default: 0)
            OPENAI_MAX_RETRIES: Application-level max retries (default: 10)
            OPENAI_MAX_BACKOFF: Maximum backoff time in seconds (default: 300)
            OPENAI_CONCURRENT_REQUESTS: Fixed concurrent requests (default: 10)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats

        # Use OpenAI official API key and base URL
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"

        # Read retry and concurrency config from environment
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "30"))
        self.sdk_max_retries = int(os.getenv("OPENAI_SDK_MAX_RETRIES", "0"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "10"))
        self.max_backoff = float(os.getenv("OPENAI_MAX_BACKOFF", "300"))

        # Fixed concurrency (no dynamic adjustment)
        self.concurrent_requests = int(os.getenv("OPENAI_CONCURRENT_REQUESTS", "10"))

        # Initialize official OpenAI async client
        # We disable SDK's internal retry to have full control
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.sdk_max_retries,
        )

        # Fixed concurrency control with semaphore
        self.semaphore = asyncio.Semaphore(self.concurrent_requests)

        # Track recent request results (success/failure) in a sliding window
        self._recent_requests = deque(maxlen=20)
        self._request_lock = asyncio.Lock()

        # Optional statistics tracking
        if self.enable_stats:
            self.current_call_stats = None

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Generate a response for the given prompt with retry and backoff.

        Args:
            prompt: Input prompt
            temperature: Override temperature for this request
            max_tokens: Override max tokens for this request
            extra_body: Extra parameters (not used with SDK)
            response_format: Response format specification

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails after all retries
        """
        # Call retry logic directly - semaphore is inside the retry loop
        return await self._generate_with_retry(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def _generate_with_retry(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> str:
        """
        Internal method with intelligent retry and backoff.

        Backoff strategy:
        - Priority 1: Use Retry-After header from OpenAI response
        - Priority 2: Use x-ratelimit-reset-* headers to calculate wait time
        - Priority 3: Simple exponential backoff based on own_failures only
        - Backoff = 1 * (2 ^ min(own_failures, 9))
        - Add small random jitter 0~2 seconds to avoid synchronization
        - Sliding window tracks last 20 request outcomes (for monitoring only)

        Note: Semaphore is acquired ONCE for the entire request lifecycle (all retries).
        Backoff is performed INSIDE the semaphore to maintain strict concurrency control.
        This ensures at most concurrent_requests can be active (executing or waiting) at any time.
        """

        last_error = None
        last_duration = 0
        own_failures = 0  # Track this request's own failure count

        # Acquire semaphore ONCE for the entire request lifecycle (all retries)
        # This ensures we don't exceed concurrent_requests and prevents request pile-up
        async with self.semaphore:
            for attempt in range(self.max_retries):
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

                    # Call OpenAI API
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
                    logger.debug(f"[OpenAI-{self.model}] API调用完成 (attempt {attempt + 1})")
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

                    # Success - record in sliding window and return
                    async with self._request_lock:
                        self._recent_requests.append('success')

                    return content

                except openai.RateLimitError as e:
                    # Rate limit error - try to get server-suggested wait time
                    error_time = time.perf_counter()
                    duration = error_time - start_time
                    last_error = e
                    last_duration = duration
                    own_failures += 1

                    # Record failure to sliding window
                    async with self._request_lock:
                        self._recent_requests.append('failure')
                        failure_count = sum(1 for r in self._recent_requests if r == 'failure')
                        total_count = len(self._recent_requests)

                    logger.warning(f"[OpenAI-{self.model}] RateLimitError (attempt {attempt + 1}/{self.max_retries})")
                    logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
                    logger.warning(f"   💬 错误: {str(e)}")
                    logger.warning(f"   📊 失败率: {failure_count}/{total_count} ({failure_count/total_count*100:.1f}%)")

                except (openai.APIError, openai.APIConnectionError) as e:
                    # Other retriable errors
                    error_time = time.perf_counter()
                    duration = error_time - start_time
                    last_error = e
                    last_duration = duration
                    own_failures += 1

                    # Record failure to sliding window
                    async with self._request_lock:
                        self._recent_requests.append('failure')
                        failure_count = sum(1 for r in self._recent_requests if r == 'failure')
                        total_count = len(self._recent_requests)

                    error_type = type(e).__name__

                    logger.warning(f"[OpenAI-{self.model}] {error_type} (attempt {attempt + 1}/{self.max_retries})")
                    logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
                    logger.warning(f"   💬 错误: {str(e)}")
                    logger.warning(f"   📊 失败率: {failure_count}/{total_count} ({failure_count/total_count*100:.1f}%)")

                except Exception as e:
                    # Non-retriable errors - fail immediately
                    error_time = time.perf_counter()
                    duration = error_time - start_time
                    logger.error(f"   ⏱️  耗时: {duration:.2f}s")
                    logger.error(f"   💬 错误信息: {str(e)}")
                    raise LLMError(f"Request failed: {str(e)}")

                # Backoff logic INSIDE semaphore
                # Use intelligent backoff strategy based on server response
                if last_error and attempt < self.max_retries - 1:
                    backoff_time = None
                    backoff_source = "exponential"

                    # Priority 1: Try to get Retry-After header (for RateLimitError)
                    if isinstance(last_error, openai.RateLimitError) and hasattr(last_error, 'response'):
                        retry_after = last_error.response.headers.get('retry-after')
                        if retry_after:
                            try:
                                backoff_time = float(retry_after)
                                backoff_source = "retry-after"
                                logger.info(f"   📨 服务器建议等待: {retry_after}s (Retry-After header)")
                            except (ValueError, TypeError):
                                pass

                        # Priority 2: Try to get x-ratelimit-reset-requests header
                        if backoff_time is None:
                            reset_requests = last_error.response.headers.get('x-ratelimit-reset-requests')
                            if reset_requests:
                                try:
                                    wait_until = float(reset_requests)
                                    backoff_time = max(0, wait_until - time.time())
                                    backoff_source = "ratelimit-reset"
                                    logger.info(f"   📨 根据rate limit重置时间计算: {backoff_time:.1f}s")
                                except (ValueError, TypeError):
                                    pass

                    # Priority 3: Simple exponential backoff based on own_failures only
                    if backoff_time is None:
                        base_backoff = 1.0
                        retry_backoff = base_backoff * (2 ** min(own_failures, 9))
                        backoff_time = min(retry_backoff, self.max_backoff)
                        backoff_source = "exponential"

                    # Add small random jitter to avoid synchronization (0~2 seconds)
                    jitter = random.uniform(0, 2.0)
                    final_backoff = backoff_time + jitter

                    # Get current failure stats and concurrency info for logging
                    async with self._request_lock:
                        failure_count = sum(1 for r in self._recent_requests if r == 'failure')
                        total_count = len(self._recent_requests)

                    # Calculate current active requests: concurrent_requests - available semaphore slots
                    active_requests = self.concurrent_requests - self.semaphore._value

                    logger.info(f"⏳ [OpenAI-{self.model}] 退避 {final_backoff:.1f}s")
                    logger.info(f"   策略: {backoff_source}")
                    logger.info(f"   本请求失败数: {own_failures}")
                    logger.info(f"   全局失败率: {failure_count}/{total_count} ({failure_count/total_count*100:.1f}%)")
                    logger.info(f"   当前并发数: {active_requests}/{self.concurrent_requests}")
                    logger.info(f"   基础退避: {backoff_time:.1f}s")
                    logger.info(f"   随机抖动: +{jitter:.1f}s")
                    logger.info(f"   下次尝试: attempt {attempt + 2}/{self.max_retries}")

                    await asyncio.sleep(final_backoff)
                elif last_error:
                    # Last attempt failed
                    async with self._request_lock:
                        failure_count = sum(1 for r in self._recent_requests if r == 'failure')
                        total_count = len(self._recent_requests)

                    logger.error(f"[OpenAI-{self.model}] 所有 {self.max_retries} 次重试均失败")
                    logger.error(f"   ⏱️  最后耗时: {last_duration:.2f}s")
                    logger.error(f"   💬 最后错误: {str(last_error)}")
                    logger.error(f"   📊 请求失败率: {failure_count}/{total_count} ({failure_count/total_count*100:.1f}%)")
                    raise LLMError(f"OpenAI API error after {self.max_retries} retries: {str(last_error)}")

            # This should never be reached due to the raise in the loop
            raise LLMError(f"Request failed after {self.max_retries} retries: {str(last_error)}")

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
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url}, sdk=official)"
