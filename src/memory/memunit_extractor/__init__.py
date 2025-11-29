"""
MemUnit Extractor Module.

This module contains extractors that transform raw input data into MemUnits.

MemUnits are the intermediate representation between raw messages and
extracted memories. They represent coherent conversation segments
identified through boundary detection.

Extractors:
-----------
- MemUnitExtractor: Abstract base class for all memunit extractors
- ConvMemUnitExtractor: Extracts MemUnits from conversation messages
"""

from .base_memunit_extractor import MemUnitExtractor
from .conv_memunit_extractor import ConvMemUnitExtractor

__all__ = [
    "MemUnitExtractor",
    "ConvMemUnitExtractor",
]
