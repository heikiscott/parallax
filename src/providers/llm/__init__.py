"""
LLM providers module for memory layer.

This module provides LLM providers for the memory layer functionality.
"""

from .openai_provider import OpenAIProvider
from .openrouter_provider import OpenRouterProvider
from .protocol import LLMProvider

__all__ = ["LLMProvider", "OpenAIProvider", "OpenRouterProvider"]


def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM providers.

    Args:
        provider_type: Type of provider ("openai", "openrouter")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider(**kwargs)
    elif provider_type == "openrouter":
        return OpenRouterProvider(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. Supported types: 'openai', 'openrouter'"
        )


def create_provider_from_env(provider_type: str, **kwargs) -> LLMProvider:
    """
    Create LLM provider from environment variables.

    Args:
        provider_type: Type of provider ("openai", "openrouter")
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "openai":
        return OpenAIProvider.from_env(**kwargs)
    elif provider_type == "openrouter":
        return OpenRouterProvider.from_env(**kwargs)
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. Supported types: 'openai', 'openrouter'"
        )