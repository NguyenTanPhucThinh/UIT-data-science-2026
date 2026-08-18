from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EvidenceChunk:
    """A ranked, traceable corpus chunk supplied to the answer generator."""

    doc_id: str
    passage: str
    corpus_chunk_index: int
    chunk_id: Optional[str] = None
    legal_section: Optional[str] = None
    document_name: Optional[str] = None
    retrieval_rank: int = 0
    rrf_score: Optional[float] = None
    reranker_score: Optional[float] = None
    is_neighbor: bool = False

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "EvidenceChunk":
        return cls(
            doc_id=str(item["doc_id"]),
            passage=str(item.get("passage", "")),
            corpus_chunk_index=int(
                item.get("corpus_chunk_index", item.get("chunk_index", -1))
            ),
            chunk_id=(str(item["chunk_id"]) if item.get("chunk_id") is not None else None),
            legal_section=(
                str(item["legal_section"])
                if item.get("legal_section") is not None
                else None
            ),
            document_name=(
                str(item["document_name"])
                if item.get("document_name") is not None
                else (
                    str(item["doc_name"])
                    if item.get("doc_name") is not None
                    else None
                )
            ),
            retrieval_rank=int(item.get("retrieval_rank", item.get("rank", 0))),
            rrf_score=(
                float(item["rrf_score"]) if item.get("rrf_score") is not None else None
            ),
            reranker_score=(
                float(item["reranker_score"])
                if item.get("reranker_score") is not None
                else (
                    float(item["score"]) if item.get("score") is not None else None
                )
            ),
            is_neighbor=bool(item.get("is_neighbor", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QAInput:
    question_id: str
    question: str
    evidence: tuple[EvidenceChunk, ...]


@dataclass
class QAResult:
    question_id: str
    answer: str
    prompt_tokens: int
    evidence_used: list[EvidenceChunk] = field(default_factory=list)
    raw_answer: Optional[str] = None

    def submission_item(self) -> dict[str, str]:
        return {"answer": self.answer}
