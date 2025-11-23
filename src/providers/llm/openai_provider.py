"""
OpenAI official API provider implementation.

This provider uses official OpenAI Python SDK (openai.AsyncOpenAI).
"""

import os
import time
import openai
from typing import Optional

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
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_stats = enable_stats

        # Use OpenAI official API key and base URL
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"

        # Initialize official OpenAI async client
        # The SDK automatically handles:
        # - Exponential backoff retry (default: 2 retries)
        # - Rate limiting (429 errors)
        # - Connection pooling and keep-alive
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600.0,  # 10 minutes timeout
            max_retries=5,  # Automatic retries with exponential backoff
        )

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
        Generate a response for the given prompt.

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

            # Call OpenAI API using official SDK
            # The SDK automatically handles retries with exponential backoff
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
            logger.debug(f"[OpenAI-{self.model}] API调用完成:")
            logger.debug(f"[OpenAI-{self.model}] 耗时: {duration:.2f}s")

            if duration > 30:
                logger.warning(f"[OpenAI-{self.model}] 耗时太长: {duration:.2f}s")

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

        except openai.APIError as e:
            # API errors (500, 503, etc.)
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"OpenAI APIError: {e}")
            logger.error(f"   ⏱️  耗时: {duration:.2f}s")
            logger.error(f"   💬 错误信息: {str(e)}")
            raise LLMError(f"OpenAI API error after retries: {str(e)}")

        except openai.APIConnectionError as e:
            # Connection errors (network issues, server disconnected, etc.)
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"OpenAI APIConnectionError: {e}")
            logger.error(f"   ⏱️  耗时: {duration:.2f}s")
            logger.error(f"   💬 错误信息: {str(e)}")
            raise LLMError(f"OpenAI connection error after retries: {str(e)}")

        except openai.RateLimitError as e:
            # Rate limit errors (429)
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"OpenAI RateLimitError: {e}")
            logger.error(f"   ⏱️  耗时: {duration:.2f}s")
            logger.error(f"   💬 错误信息: {str(e)}")
            raise LLMError(f"OpenAI rate limit exceeded after retries: {str(e)}")

        except Exception as e:
            # Other unexpected errors
            error_time = time.perf_counter()
            duration = error_time - start_time
            logger.error(f"Unexpected error: {e}")
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
        return f"OpenAIProvider(model={self.model}, base_url={self.base_url}, sdk=official)"
