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
            OPENAI_MAX_RETRIES: Application-level max retries (default: 5)
            OPENAI_BACKOFF_FACTOR: Backoff multiplier based on last request duration (default: 0.5)
            OPENAI_MIN_BACKOFF: Minimum backoff time in seconds (default: 5)
            OPENAI_MAX_BACKOFF: Maximum backoff time in seconds (default: 60)
            OPENAI_SLOW_THRESHOLD: Threshold to consider request as slow in seconds (default: 20)
            OPENAI_MAX_CONCURRENT_REQUESTS: Max concurrent requests (default: 50)
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
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "5"))

        # Adaptive backoff config: backoff based on last request duration
        self.backoff_factor = float(os.getenv("OPENAI_BACKOFF_FACTOR", "0.5"))
        self.min_backoff = float(os.getenv("OPENAI_MIN_BACKOFF", "5"))
        self.max_backoff = float(os.getenv("OPENAI_MAX_BACKOFF", "60"))
        self.slow_threshold = float(os.getenv("OPENAI_SLOW_THRESHOLD", "20"))

        self.max_concurrent_requests = int(os.getenv("OPENAI_MAX_CONCURRENT_REQUESTS", "50"))

        # Initialize official OpenAI async client
        # We disable SDK's internal retry to have full control
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.sdk_max_retries,
        )

        # Concurrency control: limit simultaneous API calls to avoid overload
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

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
        # Acquire semaphore to limit concurrent requests
        async with self.semaphore:
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
        Internal method with adaptive backoff logic.

        Simple and robust rule:
        - If last request was slow or failed, wait before next request
        - Wait time = last_duration * backoff_factor (clamped to min/max)
        """

        last_error = None
        last_duration = 0

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

                # Detect if service is busy (slow response)
                # Even on success, if the request was slow, backoff before allowing next request
                is_slow = duration > self.slow_threshold
                if is_slow:
                    logger.warning(f"[OpenAI-{self.model}] 耗时太长: {duration:.2f}s - 服务可能繁忙")

                    # Apply backoff even on success to give server breathing room
                    wait_time = duration * self.backoff_factor
                    wait_time = max(self.min_backoff, min(wait_time, self.max_backoff))

                    # Add jitter to prevent thundering herd (all requests waking up simultaneously)
                    jitter = random.uniform(0, wait_time * 0.3)  # 0-30% jitter
                    wait_time_with_jitter = wait_time + jitter

                    logger.info(f"[OpenAI-{self.model}] 虽然成功但耗时过长，退避 {wait_time_with_jitter:.1f}s (base={wait_time:.1f}s + jitter={jitter:.1f}s)...")
                    await asyncio.sleep(wait_time_with_jitter)

                # Success - return result
                return content

            except (openai.APIError, openai.APIConnectionError, openai.RateLimitError) as e:
                # Retriable errors
                error_time = time.perf_counter()
                duration = error_time - start_time
                last_error = e
                last_duration = duration

                error_type = type(e).__name__
                logger.warning(f"[OpenAI-{self.model}] {error_type} (attempt {attempt + 1}/{self.max_retries})")
                logger.warning(f"   ⏱️  耗时: {duration:.2f}s")
                logger.warning(f"   💬 错误: {str(e)}")

                # If not the last attempt, apply adaptive backoff
                if attempt < self.max_retries - 1:
                    # Adaptive backoff based on last request duration
                    # Formula: wait_time = duration * backoff_factor
                    # Clamped to [min_backoff, max_backoff]
                    wait_time = duration * self.backoff_factor
                    wait_time = max(self.min_backoff, min(wait_time, self.max_backoff))

                    # Add jitter to prevent thundering herd (all requests retrying simultaneously)
                    jitter = random.uniform(0, wait_time * 0.3)  # 0-30% jitter
                    wait_time_with_jitter = wait_time + jitter

                    logger.info(f"[OpenAI-{self.model}] 基于耗时 {duration:.1f}s 退避 {wait_time_with_jitter:.1f}s (base={wait_time:.1f}s + jitter={jitter:.1f}s, attempt {attempt + 2}/{self.max_retries})...")
                    await asyncio.sleep(wait_time_with_jitter)
                else:
                    # Last attempt failed
                    logger.error(f"[OpenAI-{self.model}] 所有 {self.max_retries} 次重试均失败")
                    logger.error(f"   ⏱️  最后耗时: {duration:.2f}s")
                    logger.error(f"   💬 最后错误: {str(e)}")
                    raise LLMError(f"OpenAI API error after {self.max_retries} retries: {str(e)}")

            except Exception as e:
                # Non-retriable errors - fail immediately
                error_time = time.perf_counter()
                duration = error_time - start_time
                logger.error(f"[OpenAI-{self.model}] Unexpected error: {e}")
                logger.error(f"   ⏱️  耗时: {duration:.2f}s")
                logger.error(f"   💬 错误信息: {str(e)}")
                raise LLMError(f"Request failed: {str(e)}")

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
