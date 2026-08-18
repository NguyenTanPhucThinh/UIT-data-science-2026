from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import ContextConfig
from .types import EvidenceChunk


@dataclass(frozen=True)
class ContextBuildResult:
    text: str
    token_count: int
    evidence_used: tuple[EvidenceChunk, ...]


class EvidenceContextBuilder:
    """Build a compact, token-budgeted prompt context from ranked evidence chunks."""

    def __init__(self, tokenizer: Any, config: ContextConfig):
        self.tokenizer = tokenizer
        self.config = config

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def _decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def _neighbor_chunks(
        self,
        base: EvidenceChunk,
        corpus_chunks: Sequence[Mapping[str, Any]],
    ) -> list[EvidenceChunk]:
        if self.config.neighbor_window == 0 or base.corpus_chunk_index < 0:
            return []

        neighbors: list[EvidenceChunk] = []
        start = max(0, base.corpus_chunk_index - self.config.neighbor_window)
        end = min(len(corpus_chunks), base.corpus_chunk_index + self.config.neighbor_window + 1)
        for idx in range(start, end):
            if idx == base.corpus_chunk_index:
                continue
            raw = corpus_chunks[idx]
            if str(raw.get("doc_id")) != base.doc_id or not str(raw.get("passage", "")).strip():
                continue
            neighbors.append(
                EvidenceChunk(
                    doc_id=base.doc_id,
                    passage=str(raw["passage"]),
                    corpus_chunk_index=idx,
                    chunk_id=(str(raw["chunk_id"]) if raw.get("chunk_id") is not None else None),
                    legal_section=(
                        str(raw["legal_section"])
                        if raw.get("legal_section") is not None
                        else None
                    ),
                    document_name=(
                        str(raw["doc_name"]) if raw.get("doc_name") is not None else base.document_name
                    ),
                    retrieval_rank=base.retrieval_rank,
                    rrf_score=base.rrf_score,
                    reranker_score=base.reranker_score,
                    is_neighbor=True,
                )
            )
        return neighbors

    def _select(
        self,
        ranked_evidence: Sequence[EvidenceChunk],
        corpus_chunks: Sequence[Mapping[str, Any]],
    ) -> list[EvidenceChunk]:
        selected: list[EvidenceChunk] = []
        seen_chunk_keys: set[tuple[str, str]] = set()
        document_order: list[str] = []
        per_document: dict[str, int] = {}

        for base in sorted(
            ranked_evidence,
            key=lambda item: (
                item.retrieval_rank if item.retrieval_rank > 0 else 10**9,
                -(item.reranker_score or float("-inf")),
            ),
        ):
            if base.doc_id not in document_order:
                if len(document_order) >= self.config.max_documents:
                    continue
                document_order.append(base.doc_id)

            candidates = [base, *self._neighbor_chunks(base, corpus_chunks)]
            candidates.sort(key=lambda item: item.corpus_chunk_index)
            for item in candidates:
                if per_document.get(item.doc_id, 0) >= self.config.max_chunks_per_document:
                    break
                key = (item.doc_id, item.chunk_id or str(item.corpus_chunk_index))
                if key in seen_chunk_keys or not item.passage.strip():
                    continue
                seen_chunk_keys.add(key)
                per_document[item.doc_id] = per_document.get(item.doc_id, 0) + 1
                selected.append(item)

        selected.sort(
            key=lambda item: (
                document_order.index(item.doc_id),
                item.corpus_chunk_index,
            )
        )
        return selected

    def _header(self, evidence: EvidenceChunk, ordinal: int) -> str:
        fields = [f"Tài liệu {ordinal}", f"doc_id={evidence.doc_id}"]
        if self.config.include_document_name and evidence.document_name:
            fields.append(f"tên={evidence.document_name}")
        if self.config.include_legal_section and evidence.legal_section:
            fields.append(f"mục={evidence.legal_section}")
        return "[" + " | ".join(fields) + "]"

    def build(
        self,
        ranked_evidence: Sequence[EvidenceChunk | Mapping[str, Any]],
        corpus_chunks: Sequence[Mapping[str, Any]],
    ) -> ContextBuildResult:
        evidence_objects = [
            item if isinstance(item, EvidenceChunk) else EvidenceChunk.from_mapping(item)
            for item in ranked_evidence
        ]
        selected = self._select(evidence_objects, corpus_chunks)

        blocks: list[str] = []
        used: list[EvidenceChunk] = []
        used_tokens = 0
        document_ordinals: dict[str, int] = {}

        for evidence in selected:
            ordinal = document_ordinals.setdefault(
                evidence.doc_id, len(document_ordinals) + 1
            )
            header = self._header(evidence, ordinal)
            separator = "\n\n" if blocks else ""
            fixed = separator + header + "\n"
            fixed_tokens = self._encode(fixed)
            remaining = self.config.max_context_tokens - used_tokens - len(fixed_tokens)
            if remaining <= 0:
                break

            passage_tokens = self._encode(evidence.passage.strip())
            if len(passage_tokens) > remaining:
                passage_tokens = passage_tokens[:remaining]
            passage = self._decode(passage_tokens).strip()
            if not passage:
                continue
            block = fixed + passage
            block_tokens = self._encode(block)
            if used_tokens + len(block_tokens) > self.config.max_context_tokens:
                overflow = used_tokens + len(block_tokens) - self.config.max_context_tokens
                if overflow >= len(passage_tokens):
                    break
                passage = self._decode(passage_tokens[:-overflow]).strip()
                block = fixed + passage
                block_tokens = self._encode(block)
            blocks.append(block)
            used.append(evidence)
            used_tokens += len(block_tokens)
            if used_tokens >= self.config.max_context_tokens:
                break

        return ContextBuildResult(
            text="".join(blocks).strip(),
            token_count=used_tokens,
            evidence_used=tuple(used),
        )
