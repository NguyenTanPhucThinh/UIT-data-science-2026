"""Local Retrieval-Augmented Generation modules for Task 2 LegalQA."""

from .config import ContextConfig, LegalQAGenerationConfig, LLMConfig
from .pipeline import LegalQAPipeline
from .types import EvidenceChunk, QAInput, QAResult

__all__ = [
    "ContextConfig",
    "EvidenceChunk",
    "LegalQAGenerationConfig",
    "LegalQAPipeline",
    "LLMConfig",
    "QAInput",
    "QAResult",
]
