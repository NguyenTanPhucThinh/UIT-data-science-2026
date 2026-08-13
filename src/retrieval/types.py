from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalResult:
    """A chunk-level retrieval result."""

    chunk_index: int
    doc_id: str
    score: float
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class DocumentCandidate:
    """A document-level candidate represented by one chunk."""

    doc_id: str
    chunk_index: int
    score: float
    chunk_id: Optional[str] = None


@dataclass(frozen=True)
class RerankResult:
    """Final document-level ranking result."""

    doc_id: str
    score: float
    chunk_index: int
    chunk_id: Optional[str] = None
