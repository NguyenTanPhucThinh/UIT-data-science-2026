from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

from .types import DocumentCandidate, RetrievalResult


DEFAULT_RRF_K = 60


def rrf_fuse(
    dense_results: Sequence[RetrievalResult],
    bm25_results: Sequence[RetrievalResult],
    rrf_k: int = DEFAULT_RRF_K,
) -> dict[int, float]:
    """Fuse two ranked chunk lists using Reciprocal Rank Fusion.

    RRF intentionally uses ranks, not raw scores, so dense cosine scores and
    BM25 scores can be combined without assuming they share a scale.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be > 0")

    fused: defaultdict[int, float] = defaultdict(float)

    for rank, result in enumerate(dense_results, start=1):
        fused[result.chunk_index] += 1.0 / (rrf_k + rank)

    for rank, result in enumerate(bm25_results, start=1):
        fused[result.chunk_index] += 1.0 / (rrf_k + rank)

    return dict(fused)


def aggregate_chunks_to_documents(
    fused_scores: dict[int, float],
    metadata: Sequence[dict],
    top_k: int = 20,
) -> list[DocumentCandidate]:
    """Map fused chunk ranking to document ranking with max-score aggregation."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    best_by_doc: dict[str, DocumentCandidate] = {}

    for chunk_index, score in fused_scores.items():
        item = metadata[chunk_index]
        doc_id = str(item["doc_id"])
        candidate = DocumentCandidate(
            doc_id=doc_id,
            chunk_index=chunk_index,
            chunk_id=str(item.get("chunk_id")) if item.get("chunk_id") is not None else None,
            score=float(score),
        )

        current = best_by_doc.get(doc_id)
        if current is None or candidate.score > current.score:
            best_by_doc[doc_id] = candidate

    ranked = sorted(best_by_doc.values(), key=lambda x: x.score, reverse=True)
    return ranked[:top_k]
