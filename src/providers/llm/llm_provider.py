import os
from typing import Optional, Callable
from .openai_provider import OpenAIProvider


class LLMProvider:
    def __init__(
        self,
        provider_type: str,
        enable_stats: bool = False,
        stats_callback: Optional[Callable[[dict], None]] = None,
        **kwargs
    ):
        """
        Initialize LLM Provider with optional statistics tracking.

        Args:
            provider_type: Type of provider ("openai", etc.)
            enable_stats: Whether to enable token statistics tracking
            stats_callback: Optional callback function called after each generation
                           with token stats dict as argument
            **kwargs: Provider-specific arguments
        """
        self.provider_type = provider_type
        self.enable_stats = enable_stats
        self.stats_callback = stats_callback

        if provider_type == "openai":
            self.provider = OpenAIProvider(enable_stats=enable_stats, **kwargs)
        else:
            raise ValueError(
                f"Unsupported provider type: {provider_type}. Supported types: 'openai'"
            )
        # TODO: add other providers

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
        response_format: dict | None = None,
    ) -> str:
        """Generate text with automatic statistics collection via callback."""
        result = await self.provider.generate(
            prompt, temperature, max_tokens, extra_body, response_format
        )

        # Automatically invoke callback if enabled
        if self.enable_stats and self.stats_callback:
            stats = self.provider.get_current_call_stats()
            if stats:
                self.stats_callback(stats)

        return result

    def get_current_call_stats(self) -> Optional[dict]:
        """Get statistics for the most recent call (if stats enabled)."""
        if self.enable_stats:
            return self.provider.get_current_call_stats()
        return None

    def set_stats_callback(self, callback: Optional[Callable[[dict], None]]) -> None:
        """Set or update the statistics callback function."""
        self.stats_callback = callback
