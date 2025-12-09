"""Core retrieval system components."""

from .types import Document

# Note: utils are imported directly where needed to avoid initialization issues
# from .utils import build_bm25_index, etc.

__all__ = ['Document']
