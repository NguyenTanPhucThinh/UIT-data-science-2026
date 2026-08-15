from __future__ import annotations

from pathlib import Path
from typing import Sequence

from sentence_transformers import CrossEncoder

from .types import DocumentCandidate, RerankResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 4


class Reranker:
    """Development cross-encoder reranker for document candidates."""

    def __init__(
        self,
        metadata: Sequence[dict],
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "auto",
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.metadata = metadata
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.batch_size = batch_size
        self.model = CrossEncoder(
            self.model_name,
            num_labels=1,
            max_length=max_length,
            device=self.device,
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device

        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def rerank(
        self,
        question: str,
        candidates: Sequence[DocumentCandidate],
        top_k: int = 5,
    ) -> list[RerankResult]:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if not candidates:
            return []

        pairs = [
            (question, self.metadata[candidate.chunk_index]["passage"])
            for candidate in candidates
        ]

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = [
            RerankResult(
                doc_id=candidate.doc_id,
                score=float(score),
                chunk_index=candidate.chunk_index,
                chunk_id=candidate.chunk_id,
            )
            for candidate, score in zip(candidates, scores)
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked[:top_k]
