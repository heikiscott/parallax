"""
OpenRouter LLM provider implementation.

This provider uses OpenRouter API to access multiple LLM models
with optional provider selection feature.
"""

from math import log
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import aiohttp
from typing import Optional, List
import asyncio
import random

from .protocol import LLMProvider, LLMError
from core.observation.logger import get_logger
from config import load_config

logger = get_logger(__name__)


def _get_provider_config():
    """获取 provider 配置"""
    return load_config("src/providers")


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM provider.

    This provider uses OpenRouter API to access multiple LLM models
    with optional provider selection for routing control.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_stats: bool = False,
        provider_order: List[str] | None = None,
        **kwargs,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            model: Model name (e.g., "openai/gpt-4o-mini", "x-ai/grok-4-fast")
            api_key: OpenRouter API key
            base_url: OpenRouter base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_stats: Enable usage statistics accumulation (default: False)
            provider_order: Optional list of provider names for routing
            **kwargs: Additional arguments (ignored for now)
        """
        # 从配置文件加载默认值
        cfg = _get_provider_config()
        openrouter_cfg = cfg.openrouter

        self.model = model or openrouter_cfg.model
        self.temperature = temperature if temperature is not None else openrouter_cfg.temperature
        self.max_tokens = max_tokens if max_tokens is not None else openrouter_cfg.max_tokens
        self.enable_stats = enable_stats

        # API Key 和 Base URL 从配置加载
        self.api_key = api_key or openrouter_cfg.api_key
        self.base_url = base_url or openrouter_cfg.base_url

        # Provider 路由顺序（可选）
        self.provider_order = provider_order or getattr(openrouter_cfg, 'provider_order', None)

        # 可选的单次调用统计
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

        Returns:
            Generated response text

        Raises:
            LLMError: If generation fails
        """
        # 使用 time.perf_counter() 获得更精确的时间测量
        start_time = time.perf_counter()

        # 构建 provider 路由配置（如果指定了 provider_order）
        openrouter_provider = None
        if self.provider_order:
            openrouter_provider = {"order": self.provider_order, "allow_fallbacks": False}

        # Prepare request data
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.temperature,
            "response_format": response_format,
        }
        # Only add provider parameter for OpenRouter
        if openrouter_provider is not None:
            data["provider"] = openrouter_provider
        # print(data)
        # print(data["extra_body"])
        # Add max_tokens if specified
        if max_tokens is not None:
            data["max_tokens"] = int(max_tokens) if isinstance(max_tokens, str) else max_tokens
        elif self.max_tokens is not None:
            data["max_tokens"] = int(self.max_tokens) if isinstance(self.max_tokens, str) else self.max_tokens

        # 使用异步的 aiohttp 替代同步的 urllib
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        max_retries = 5
        for retry_num in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=600)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions", json=data, headers=headers
                    ) as response:
                        chunks = []
                        async for chunk in response.content.iter_any():
                            chunks.append(chunk)
                        test = b"".join(chunks).decode()
                        response_data = json.loads(test)
                        # print(response_data)
                        # 处理错误响应
                        if response.status != 200:
                            error_msg = response_data.get('error', {}).get(
                                'message', f"HTTP {response.status}"
                            )
                            logger.error(
                                f"❌ [OpenAI-{self.model}] HTTP错误 {response.status}:"
                            )
                            logger.error(f"   💬 错误信息: {error_msg}")
                            # Debug: 429 Too Many Requests 断点调试
                            if response.status == 429:
                                logger.warning(
                                    f"429 Too Many Requests, waiting for 10 seconds"
                                )
                                await asyncio.sleep(random.randint(5, 20))

                            raise LLMError(f"HTTP Error {response.status}: {error_msg}")

                        # 使用 time.perf_counter() 获得更精确的时间测量
                        end_time = time.perf_counter()

                        # 提取finish_reason
                        finish_reason = response_data.get('choices', [{}])[0].get(
                            'finish_reason', ''
                        )
                        if finish_reason == 'stop':
                            logger.debug(
                                f"[OpenAI-{self.model}] 完成原因: {finish_reason}"
                            )
                        else:
                            logger.warning(
                                f"[OpenAI-{self.model}] 完成原因: {finish_reason}"
                            )

                        # 提取token使用信息
                        usage = response_data.get('usage', {})
                        prompt_tokens = usage.get('prompt_tokens', 0)
                        completion_tokens = usage.get('completion_tokens', 0)
                        total_tokens = usage.get('total_tokens', 0)

                        # 打印详细的使用信息

                        logger.debug(f"[OpenAI-{self.model}] API调用完成:")
                        logger.debug(
                            f"[OpenAI-{self.model}] 耗时: {end_time - start_time:.2f}s"
                        )
                        # 如果耗时太长
                        if end_time - start_time > 30:
                            logger.warning(
                                f"[OpenAI-{self.model}] 耗时太长: {end_time - start_time:.2f}s"
                            )
                        logger.debug(
                            f"[OpenAI-{self.model}] Prompt Tokens: {prompt_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] Completion Tokens: {completion_tokens:,}"
                        )
                        logger.debug(
                            f"[OpenAI-{self.model}] 总Token数: {total_tokens:,}"
                        )

                        # 新增：记录当前调用的统计信息（如果开启统计）
                        if self.enable_stats:
                            self.current_call_stats = {
                                'prompt_tokens': prompt_tokens,
                                'completion_tokens': completion_tokens,
                                'total_tokens': total_tokens,
                                'duration': end_time - start_time,
                                'timestamp': time.time(),
                            }

                        return response_data['choices'][0]['message']['content']

            except aiohttp.ClientError as e:
                error_time = time.perf_counter()
                logger.error("aiohttp.ClientError: %s", e)
                # logger.error(f"❌ [OpenAI-{self.model}] 请求失败:")
                logger.error(f"   ⏱️  耗时: {error_time - start_time:.2f}s")
                logger.error(f"   💬 错误信息: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                # raise LLMError(f"Request failed: {str(e)}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")
            except Exception as e:
                error_time = time.perf_counter()
                logger.error("Exception: %s", e)
                logger.error(f"   ⏱️  耗时: {error_time - start_time:.2f}s")
                logger.error(f"   💬 错误信息: {str(e)}")
                logger.error(f"retry_num: {retry_num}")
                if retry_num == max_retries - 1:
                    raise LLMError(f"Request failed: {str(e)}")

    async def test_connection(self) -> bool:
        """
        Test the connection to the OpenRouter API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info(f"🔗 [OpenAI-{self.model}] 测试API连接...")
            # Try a simple generation to test connection
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
        if self.enable_stats:
            return self.current_call_stats
        return None

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"OpenRouterProvider(model={self.model}, base_url={self.base_url})"
