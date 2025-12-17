"""
OpenAI official API provider implementation with openlimit.

This provider uses official OpenAI Python SDK (openai.AsyncOpenAI) with openlimit
for intelligent rate limiting and token management.
"""

import time
import asyncio
import httpx
import openai
from typing import Optional
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from aiolimiter import AsyncLimiter

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger
from config import load_config

logger = get_logger(__name__)

# 加载配置（带缓存）
def _get_provider_config():
    """获取 provider 配置"""
    return load_config("services/providers")


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
            model: Model name (defaults to config)
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: OpenAI base URL (defaults to config)
            temperature: Sampling temperature (defaults to config)
            max_tokens: Maximum tokens to generate (defaults to config)
            enable_stats: Enable usage statistics accumulation (default: False)
            **kwargs: Additional arguments (ignored for now)

        Configuration:
            配置来源: config/services/providers.yaml
            API Key 来源: config/secrets/secrets.yaml（通过 ${OPENAI_API_KEY} 注入）
        """
        # 加载配置文件
        cfg = _get_provider_config()
        llm_cfg = cfg.llm
        openai_cfg = cfg.openai

        # LLM 基础配置 (优先级: 参数 > YAML)
        self.model = model or llm_cfg.model
        self.temperature = temperature if temperature is not None else llm_cfg.temperature
        self.max_tokens = max_tokens if max_tokens is not None else llm_cfg.max_tokens
        self.enable_stats = enable_stats

        # API Key 从配置读取（secrets.yaml 会自动注入）
        self.api_key = api_key or llm_cfg.api_key
        self.base_url = base_url or llm_cfg.base_url

        # OpenAI Provider 配置
        self.timeout = float(openai_cfg.timeout)
        self.max_retries = int(openai_cfg.max_retries)

        # Provider-level concurrency control
        max_concurrent = int(openai_cfg.max_concurrent)

        # Rate limiting configuration
        rpm_limit = int(openai_cfg.rpm_limit)

        # Configure httpx client with connection settings optimized for stability
        #
        # 🔥 CRITICAL FIX: http2=False prevents "Server disconnected" errors
        # httpx's HTTP/2 implementation is unstable with OpenAI's servers
        http_client = httpx.AsyncClient(
            http2=False,  # 🔥 Disable HTTP/2 to prevent protocol errors
            limits=httpx.Limits(
                max_keepalive_connections=max_concurrent * 2,  # Match semaphore limit
                max_connections=max_concurrent * 4,  # 4x buffer for safety
                # 🔥【关键修改】设置连接保活过期时间
                # 默认是 5秒或更长。
                # 既然 OpenAI 那边容易断，我们设置得短一点，比如 5秒。
                # 超过 5秒没用的连接，本地直接扔掉，下次新建，避免去复用那个可能已经断掉的“僵尸连接”。
                keepalive_expiry=5.0
            ),
            timeout=httpx.Timeout(self.timeout, connect=30.0),
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
        logger.info(f"  Timeout: {self.timeout}s, connect=30.0s")
        logger.info(f"  Max retries (SDK): {self.max_retries}")
        logger.info(f"  Max concurrent: {max_concurrent}")
        logger.info(f"  RPM limit: {rpm_limit} (proactive rate limiting)")
        logger.info(f"  HTTPX: http2=False, keepalive={max_concurrent}, max_conn={max_concurrent * 4}")

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
        # Read retry configuration from YAML
        cfg = _get_provider_config()
        retry_cfg = cfg.openai.retry

        retry_min_wait = int(retry_cfg.min_wait)
        retry_max_wait = int(retry_cfg.max_wait)
        retry_attempts = int(retry_cfg.attempts)

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
                            httpx.RemoteProtocolError,
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

                                # ---------------------------------------------------
                                # 🔥 核心修改：开启 stream=True (流式传输)
                                # ---------------------------------------------------
                                # 即使你不需要实时显示，也必须用 stream。
                                # 因为 stream 会每隔几百毫秒传回一个数据包，
                                # 这能骗过防火墙/代理，告诉它们："别切断！我还活着！"
                                # ---------------------------------------------------

                                # 流式传输超时设置为 5 分钟，允许处理超长文本
                                stream_timeout = 300.0

                                response_stream = await self.client.chat.completions.create(
                                    **params,
                                    stream=True,  # 🔥 必须开启流式传输！
                                    stream_options={"include_usage": True},  # 请求返回 usage 信息
                                    timeout=stream_timeout,
                                    extra_headers={"Connection": "close"}
                                )

                                collected_content = []
                                finish_reason = None
                                usage_info = None

                                # 迭代流式数据
                                # 只要有数据包传回来，连接就会保持活跃，不会被代理切断
                                async for chunk in response_stream:
                                    # 收集内容片段
                                    if chunk.choices and len(chunk.choices) > 0:
                                        delta = chunk.choices[0].delta
                                        if delta and delta.content:
                                            collected_content.append(delta.content)
                                        # 记录完成原因
                                        if chunk.choices[0].finish_reason:
                                            finish_reason = chunk.choices[0].finish_reason

                                    # 流式模式下，usage 信息在最后一个 chunk 中
                                    if chunk.usage:
                                        usage_info = chunk.usage

                                # 拼接完整结果
                                content = "".join(collected_content)

                                # 校验结果
                                if not content:
                                    logger.warning(f"[OpenAI-{self.model}] 流式传输返回空内容，视作失败")
                                    raise ValueError("Empty response received from stream")

                                end_time = time.perf_counter()
                                duration = end_time - start_time

                                # Log completion details
                                if finish_reason == 'stop':
                                    logger.debug(f"[OpenAI-{self.model}] 完成原因: {finish_reason}")
                                elif finish_reason:
                                    logger.warning(f"[OpenAI-{self.model}] 完成原因: {finish_reason}")
                                else:
                                    logger.debug(f"[OpenAI-{self.model}] 流式传输完成 (无 finish_reason)")

                                # Extract token usage (流式模式下从最后一个 chunk 获取)
                                prompt_tokens = usage_info.prompt_tokens if usage_info else 0
                                completion_tokens = usage_info.completion_tokens if usage_info else 0
                                total_tokens = usage_info.total_tokens if usage_info else 0

                                # Log usage information
                                logger.debug(f"[OpenAI-{self.model}] API调用完成 (流式)")
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

                            except httpx.RemoteProtocolError as e:
                                # 网络中断错误，让 tenacity 重试
                                error_time = time.perf_counter()
                                duration = error_time - start_time
                                attempt_num = attempt.retry_state.attempt_number
                                logger.warning(f"🔌 [OpenAI-{self.model}] 流式传输网络中断! 重试 {attempt_num}/{retry_attempts}")
                                logger.warning(f"   ⏱️  已耗时: {duration:.2f}s")
                                logger.warning(f"   💬 错误: {str(e)[:200]}")
                                raise

                            except openai.BadRequestError as e:
                                # BadRequestError should NOT be retried - it indicates a problem with the request itself
                                error_time = time.perf_counter()
                                duration = error_time - start_time
                                logger.error(f"❌ [OpenAI-{self.model}] BadRequestError - 请求参数错误，不重试")
                                logger.error(f"   ⏱️  耗时: {duration:.2f}s")
                                logger.error(f"   💬 完整错误: {str(e)}")
                                # 提取并打印详细信息
                                if hasattr(e, 'body') and e.body:
                                    logger.error(f"   📋 错误详情: {e.body}")
                                if hasattr(e, 'message'):
                                    logger.error(f"   📋 错误消息: {e.message}")
                                raise LLMError(f"BadRequestError: {str(e)}")

                            except (openai.RateLimitError, openai.APIConnectionError, openai.APIError, openai.APITimeoutError) as e:
                                # Log retry-able errors and let tenacity handle retry
                                error_time = time.perf_counter()
                                duration = error_time - start_time
                                error_type = type(e).__name__
                                attempt_num = attempt.retry_state.attempt_number

                                # Determine log level based on error type
                                # Connection errors are more serious and should be warnings
                                is_connection_error = isinstance(e, (openai.APIConnectionError, openai.APITimeoutError))

                                # Special warning for rate limit errors
                                if isinstance(e, openai.RateLimitError):
                                    logger.warning(f"⚠️  [OpenAI-{self.model}] 触发了 429 RateLimitError!")
                                    logger.warning(f"   这说明 OPENAI_RPM_LIMIT 设置过高，请降低到你的 tier 限制的 80%")

                                # Calculate next retry wait time
                                if attempt_num < retry_attempts:
                                    next_min = min(retry_min_wait * (2 ** (attempt_num - 1)), retry_max_wait)
                                    next_max = retry_max_wait
                                    # Use warning for connection errors, info for rate limits
                                    if is_connection_error:
                                        logger.warning(f"🔌 [OpenAI-{self.model}] 连接异常! 重试 {attempt_num}/{retry_attempts}: {error_type}")
                                        logger.warning(f"   ⏱️  已耗时: {duration:.2f}s, 将等待 {next_min:.1f}-{next_max:.1f}s 后重试")
                                        logger.warning(f"   💬 错误: {str(e)[:200]}")  # Truncate long error messages
                                    else:
                                        logger.info(f"[OpenAI-{self.model}] 重试 {attempt_num}/{retry_attempts}: {error_type}")
                                        logger.info(f"   ⏱️  已耗时: {duration:.2f}s, 将等待 {next_min:.1f}-{next_max:.1f}s 后重试")
                                        logger.info(f"   💬 错误: {str(e)[:500]}")  # 打印更多错误信息
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

        except (openai.APIError, openai.APIConnectionError, openai.APITimeoutError, httpx.RemoteProtocolError) as e:
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
