"""External service integrations for retrieval.

Uses lazy imports to avoid loading heavy dependencies (like core.di) unnecessarily.
"""

__all__ = [
    'get_vectorize_service',
    'get_rerank_service',
    'get_colbert_service',
    'ColBERTService',
    'ColBERTConfig',
]


def __getattr__(name):
    """Lazy import of services to avoid circular dependencies."""
    if name == 'get_vectorize_service':
        from .vectorize import get_vectorize_service
        return get_vectorize_service
    elif name == 'get_rerank_service':
        from .rerank import get_rerank_service
        return get_rerank_service
    elif name in ('get_colbert_service', 'ColBERTService', 'ColBERTConfig'):
        from .colbert_service import get_colbert_service, ColBERTService, ColBERTConfig
        if name == 'get_colbert_service':
            return get_colbert_service
        elif name == 'ColBERTService':
            return ColBERTService
        elif name == 'ColBERTConfig':
            return ColBERTConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
