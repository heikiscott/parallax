"""
MemUnit extraction abstract base class.
"""

from abc import ABC, abstractmethod
from typing import Optional

from providers.llm.llm_provider import LLMProvider
from memory.schema import MemUnit, SourceType
from .memunit_extract_request import MemUnitExtractRequest
from .status_result import StatusResult


class MemUnitExtractor(ABC):
    """Abstract base class for MemUnit extraction."""

    def __init__(
        self, source_type: SourceType, llm_provider=LLMProvider
    ):
        self.source_type = source_type
        self._llm_provider = llm_provider

    @abstractmethod
    async def extract_memunit(
        self, request: MemUnitExtractRequest
    ) -> tuple[Optional[MemUnit], Optional[StatusResult]]:
        pass
